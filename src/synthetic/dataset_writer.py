"""Dataset output API for paired RGB images and YOLO label files."""

from __future__ import annotations

import hashlib
import json
import math
from typing import TextIO

import numpy as np
from PIL import Image
import yaml

from .config import SyntheticConfig
from .types import CapturedFrame, SplitName, WrittenSample, YoloAnnotation


class DatasetWriter:
    """Write images, labels, and a sample manifest under the configured root."""

    def __init__(self, config: SyntheticConfig) -> None:
        self.config = config
        self._manifest_file: TextIO | None = None

    def __enter__(self) -> DatasetWriter:
        root = self.config.output.root
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "manifest.jsonl"
        existing_samples = any(
            path.is_file()
            for split in ("train", "val")
            for directory in (root / "images" / split, root / "labels" / split)
            if directory.is_dir()
            for path in directory.iterdir()
        )
        if manifest_path.exists() or existing_samples:
            raise FileExistsError(
                f"dataset output is not empty: {root}; choose a new path or clear it explicitly"
            )
        dataset_yaml = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "names": [self.config.labels.pedestrian_class_name],
        }
        (root / "dataset.yaml").write_text(
            yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8"
        )
        self._manifest_file = manifest_path.open("x", encoding="utf-8")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.finalize()

    def write_sample(
        self,
        frame: CapturedFrame,
        split: SplitName,
        annotations: tuple[YoloAnnotation, ...],
    ) -> WrittenSample:
        if split not in {"train", "val"}:
            raise ValueError(f"unsupported dataset split: {split}")
        if self._manifest_file is None or self._manifest_file.closed:
            raise RuntimeError("DatasetWriter must be used as a context manager")
        if frame.frame_id < 0 or not frame.scene_id:
            raise ValueError("frame id must be non-negative and scene id must not be empty")
        if not math.isfinite(frame.timestamp_seconds):
            raise ValueError("frame timestamp must be finite")

        image = np.asarray(frame.rgb_image)
        expected_shape = (self.config.camera.height, self.config.camera.width, 3)
        if image.dtype != np.uint8 or image.shape != expected_shape:
            raise ValueError(f"RGB image must be uint8 with shape {expected_shape}; got {image.shape}")

        scene_token = hashlib.sha256(frame.scene_id.encode("utf-8")).hexdigest()[:12]
        sample_id = f"{scene_token}_{frame.frame_id:08d}"
        image_dir = self.config.output.root / "images" / split
        label_dir = self.config.output.root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        extension = "jpg" if self.config.output.image_format == "jpeg" else self.config.output.image_format
        pillow_format = "JPEG" if extension in {"jpg", "jpeg"} else "PNG"
        image_path = image_dir / f"{sample_id}.{extension}"
        label_path = label_dir / f"{sample_id}.txt"
        if image_path.exists() or label_path.exists():
            raise FileExistsError(f"sample already exists: {sample_id}")

        label_text = "\n".join(annotation.to_line() for annotation in annotations)
        if label_text:
            label_text += "\n"

        temporary_image = image_path.with_name(f".{image_path.name}.tmp")
        temporary_label = label_path.with_name(f".{label_path.name}.tmp")
        try:
            Image.fromarray(image).save(temporary_image, format=pillow_format)
            temporary_label.write_text(label_text, encoding="utf-8")
            temporary_image.replace(image_path)
            temporary_label.replace(label_path)
        except Exception:
            temporary_image.unlink(missing_ok=True)
            temporary_label.unlink(missing_ok=True)
            image_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)
            raise

        record = {
            "sample_id": sample_id,
            "frame_id": frame.frame_id,
            "scene_id": frame.scene_id,
            "timestamp_seconds": frame.timestamp_seconds,
            "split": split,
            "image": image_path.relative_to(self.config.output.root).as_posix(),
            "label": label_path.relative_to(self.config.output.root).as_posix(),
            "width": self.config.camera.width,
            "height": self.config.camera.height,
            "pedestrian_count": len(annotations),
        }
        self._manifest_file.write(json.dumps(record, sort_keys=True) + "\n")
        self._manifest_file.flush()
        return WrittenSample(frame_id=frame.frame_id, split=split, image_path=image_path, label_path=label_path)

    def finalize(self) -> None:
        if self._manifest_file is not None and not self._manifest_file.closed:
            self._manifest_file.flush()
            self._manifest_file.close()