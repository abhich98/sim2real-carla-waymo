"""Shared data contracts for capture, annotations, writing, and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray


SplitName = Literal["train", "val"]
RGBImage = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class PixelBox:
    """Image-space xyxy box in pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True, slots=True)
class YoloAnnotation:
    """YOLO class id and normalized center-x, center-y, width, height."""

    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float

    def to_line(self) -> str:
        """Convert the annotation to a YOLO-formatted line."""
        values = (self.center_x, self.center_y, self.width, self.height)
        if self.class_id < 0 or any(not math.isfinite(value) for value in values):
            raise ValueError("YOLO annotation contains invalid values")
        if any(not 0 <= value <= 1 for value in values) or self.width == 0 or self.height == 0:
            raise ValueError("YOLO coordinates must be within [0, 1] with positive size")
        if (
            self.center_x - self.width / 2 < 0
            or self.center_y - self.height / 2 < 0
            or self.center_x + self.width / 2 > 1
            or self.center_y + self.height / 2 > 1
        ):
            raise ValueError("YOLO box must lie within image bounds")
        return f"{self.class_id} " + " ".join(f"{value:.8f}" for value in values)


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    frame_id: int
    timestamp_seconds: float
    scene_id: str
    rgb_image: RGBImage
    pedestrian_boxes: tuple[PixelBox, ...]


@dataclass(frozen=True, slots=True)
class WrittenSample:
    frame_id: int
    split: SplitName
    image_path: Path
    label_path: Path


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    requested_frames: int
    written_frames: int
    skipped_frames: int
    output_root: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    fiftyone_errors: tuple[str, ...]
    project_errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.fiftyone_errors and not self.project_errors

    @property
    def errors(self) -> tuple[str, ...]:
        return self.fiftyone_errors + self.project_errors


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    image_count: int
    label_file_count: int
    pedestrian_count: int
    images_by_split: dict[str, int]
    pedestrians_by_split: dict[str, int]