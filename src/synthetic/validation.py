"""FiftyOne YOLO import validation plus project-specific integrity rules."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .config import SyntheticConfig
from .splitting import split_for_scene
from .types import DatasetValidationReport


def _validate_with_fiftyone(root: Path) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Import each split with FiftyOne and return its parsed sample labels."""
    from fiftyone.types import YOLOv5Dataset

    errors: list[str] = []
    parsed_labels: dict[str, Any] = {}
    importer_class = YOLOv5Dataset().get_dataset_importer_cls()
    for split in ("train", "val"):
        importer = importer_class(
            dataset_dir=str(root),
            split=split,
            label_type="detections",
            include_all_data=True,
        )
        try:
            importer.setup()
            for filepath, _metadata, label in importer:
                parsed_labels[Path(filepath).resolve().as_posix()] = label
        except Exception as exc:
            errors.append(f"FiftyOne could not import YOLO {split} split: {exc}")
        finally:
            importer.close()
    return tuple(errors), parsed_labels


def _validate_pedestrian_labels(
    parsed_labels: dict[str, Any], config: SyntheticConfig
) -> tuple[str, ...]:
    """Enforce the project class mapping and normalized in-image box rule."""
    errors: list[str] = []
    for filepath, detections in parsed_labels.items():
        for detection in detections.detections:
            if detection.label != config.labels.pedestrian_class_name:
                errors.append(
                    f"unexpected label {detection.label!r} in {filepath}; "
                    f"expected {config.labels.pedestrian_class_name!r}"
                )
            x, y, width, height = detection.bounding_box
            if (
                not all(math.isfinite(value) for value in (x, y, width, height))
                or x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or x + width > 1
                or y + height > 1
            ):
                errors.append(f"YOLO bounding box is outside normalized image bounds in {filepath}")
    return tuple(errors)


def _is_within_root(root: Path, relative_path: str) -> bool:
    try:
        (root / relative_path).resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _validate_project_contracts(config: SyntheticConfig) -> tuple[str, ...]:
    """Validate project-specific contracts such as manifest completeness and scene-level split isolation."""
    root = config.output.root
    errors: list[str] = []
    manifest_path = root / "manifest.jsonl"
    if not root.is_dir():
        return (f"dataset directory does not exist: {root}",)
    if not manifest_path.is_file():
        return (f"dataset manifest does not exist: {manifest_path}",)

    samples: dict[str, dict[str, Any]] = {}
    scene_splits: dict[str, str] = {}
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return (f"cannot read dataset manifest: {exc}",)

    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
            sample_id = record["sample_id"]
            scene_id = record["scene_id"]
            split = record["split"]
            image_ref = record["image"]
            label_ref = record["label"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append(f"invalid manifest record at line {line_number}: {exc}")
            continue

        if not all(isinstance(value, str) and value for value in (sample_id, scene_id)):
            errors.append(f"sample and scene ids must be non-empty strings at manifest line {line_number}")
            continue
        if sample_id in samples:
            errors.append(f"duplicate manifest sample id: {sample_id}")
        samples[sample_id] = record
        if split not in {"train", "val"}:
            errors.append(f"invalid split for manifest sample {sample_id}: {split}")
            continue

        previous_split = scene_splits.setdefault(scene_id, split)
        if previous_split != split:
            errors.append(f"scene {scene_id} appears in both train and val splits")
        if split != split_for_scene(scene_id, config):
            errors.append(f"scene {scene_id} is assigned to the wrong configured split")

        expected_image_prefix = f"images/{split}/"
        expected_label_prefix = f"labels/{split}/"
        if not isinstance(image_ref, str) or not image_ref.startswith(expected_image_prefix):
            errors.append(f"manifest image is in the wrong split for {sample_id}")
        if not isinstance(label_ref, str) or not label_ref.startswith(expected_label_prefix):
            errors.append(f"manifest label is in the wrong split for {sample_id}")
        for reference in (image_ref, label_ref):
            if not isinstance(reference, str) or not _is_within_root(root, reference):
                errors.append(f"unsafe or invalid manifest path for {sample_id}: {reference}")
            elif not (root / reference).is_file():
                errors.append(f"manifest reference does not exist for {sample_id}: {reference}")
        if isinstance(image_ref, str) and Path(image_ref).stem != sample_id:
            errors.append(f"manifest image name does not match sample id {sample_id}")
        if isinstance(label_ref, str) and Path(label_ref).stem != sample_id:
            errors.append(f"manifest label name does not match sample id {sample_id}")

    expected_images: set[str] = set()
    expected_labels: set[str] = set()
    for split in ("train", "val"):
        image_directory = root / "images" / split
        label_directory = root / "labels" / split
        expected_images.update(
            path.relative_to(root).as_posix()
            for path in image_directory.glob("*")
            if path.is_file()
        )
        expected_labels.update(
            path.relative_to(root).as_posix()
            for path in label_directory.glob("*.txt")
            if path.is_file()
        )
    manifest_images = {
        reference
        for record in samples.values()
        if isinstance((reference := record.get("image")), str)
    }
    manifest_labels = {
        reference
        for record in samples.values()
        if isinstance((reference := record.get("label")), str)
    }
    for path in sorted(expected_images - manifest_images):
        errors.append(f"image is missing from manifest: {path}")
    for path in sorted(expected_labels - manifest_labels):
        errors.append(f"label is missing from manifest: {path}")
    for path in sorted(manifest_images - expected_images):
        errors.append(f"manifest image is not in the dataset: {path}")
    for path in sorted(manifest_labels - expected_labels):
        errors.append(f"manifest label is not in the dataset: {path}")

    return tuple(errors)


def validate_dataset(config: SyntheticConfig) -> DatasetValidationReport:
    """Use FiftyOne's YOLO importer, then check manifest and split contracts."""
    root = config.output.root
    if not root.is_dir():
        return DatasetValidationReport(
            fiftyone_errors=(),
            project_errors=(f"dataset directory does not exist: {root}",),
        )
    fiftyone_errors, parsed_labels = _validate_with_fiftyone(root)
    project_errors = _validate_project_contracts(config) + _validate_pedestrian_labels(
        parsed_labels, config
    )
    return DatasetValidationReport(fiftyone_errors=fiftyone_errors, project_errors=project_errors)