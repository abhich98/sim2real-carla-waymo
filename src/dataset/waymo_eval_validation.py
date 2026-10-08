"""FiftyOne-backed validation and project contract checks for Waymo eval exports."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .waymo_eval_config import WaymoEvalConfig


def validate_waymo_eval_dataset(config: WaymoEvalConfig) -> tuple[str, ...]:
    """Import the test split with FiftyOne and check manifest/sampling invariants."""
    root = config.output.root
    errors: list[str] = []
    if not root.is_dir():
        return (f"evaluation output directory does not exist: {root}",)

    from fiftyone.types import YOLOv5Dataset

    importer_class = YOLOv5Dataset().get_dataset_importer_cls()
    importer = importer_class(
        dataset_dir=str(root),
        split="test",
        label_type="detections",
        include_all_data=True,
    )
    imported_labels: dict[str, Any] = {}
    try:
        importer.setup()
        for filepath, _metadata, labels in importer:
            imported_labels[Path(filepath).resolve().as_posix()] = labels
    except Exception as exc:
        errors.append(f"FiftyOne could not import the YOLO test split: {exc}")
    finally:
        importer.close()

    manifest_path = root / "manifest.jsonl"
    if not manifest_path.is_file():
        return tuple(errors + [f"manifest does not exist: {manifest_path}"])
    seen_samples: set[str] = set()
    frequencies: dict[tuple[str, int, str], int] = {}
    try:
        records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        return tuple(errors + [f"could not parse evaluation manifest: {exc}"])

    for line_number, record in enumerate(records, start=1):
        try:
            sample_id = record["sample_id"]
            context = record["scene_id"]
            camera = int(record["camera_name"])
            timestamp = int(record["timestamp_micros"])
            image_ref = record["image"]
            label_ref = record["label"]
            pedestrians = record["pedestrians"]
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid manifest record at line {line_number}: {exc}")
            continue
        if sample_id in seen_samples:
            errors.append(f"duplicate evaluation sample id: {sample_id}")
        seen_samples.add(sample_id)
        if camera not in config.source.cameras or context not in config.source.contexts:
            errors.append(f"sample {sample_id} has unconfigured source keys")
        for reference, prefix in ((image_ref, "images/test/"), (label_ref, "labels/test/")):
            if not isinstance(reference, str) or not reference.startswith(prefix):
                errors.append(f"sample {sample_id} has invalid test split path: {reference}")
                continue
            try:
                resolved = (root / reference).resolve()
                resolved.relative_to(root.resolve())
            except (OSError, ValueError):
                errors.append(f"sample {sample_id} path escapes output root: {reference}")
                continue
            if not resolved.is_file():
                errors.append(f"sample {sample_id} file is missing: {reference}")
        if Path(image_ref).stem != sample_id or Path(label_ref).stem != sample_id:
            errors.append(f"sample {sample_id} filenames do not match manifest ID")
        if len(pedestrians) != record.get("pedestrian_count"):
            errors.append(f"sample {sample_id} pedestrian count does not match manifest boxes")
        if len(pedestrians) == 0 and record.get("pedestrian_count") != 0:
            errors.append(f"empty-label sample has nonzero pedestrian count: {sample_id}")
        for pedestrian in pedestrians:
            identity = (context, camera, str(pedestrian["camera_object_id"]))
            frequencies[identity] = frequencies.get(identity, 0) + 1
            box = pedestrian.get("box_xyxy_pixels", [])
            if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
                errors.append(f"invalid source box for sample {sample_id}")
            if pedestrian.get("yolo", "").split(maxsplit=1)[0] != str(config.output.pedestrian_class_id):
                errors.append(f"wrong pedestrian class id for sample {sample_id}")

    expected_image_count = len(records)
    if len(imported_labels) != expected_image_count:
        errors.append(
            f"FiftyOne imported {len(imported_labels)} samples, manifest contains {expected_image_count}"
        )
    for (context, camera, object_id), count in frequencies.items():
        if count > config.sampling.max_appearances_per_identity:
            errors.append(
                f"identity exceeds configured cap: {context}/camera_{camera}/{object_id} ({count})"
            )
    return tuple(errors)