"""Contracts for converting CARLA pedestrian boxes into YOLO labels."""

import math

from .config import CameraConfig, LabelConfig, PedestrianConfig
from .types import PixelBox, YoloAnnotation


def box_to_yolo(
    box: PixelBox,
    camera: CameraConfig,
    labels: LabelConfig,
    pedestrians: PedestrianConfig,
) -> YoloAnnotation | None:
    """Convert a visible pixel box to a normalized YOLO annotation."""
    coordinates = (box.x_min, box.y_min, box.x_max, box.y_max)
    if any(not math.isfinite(value) for value in coordinates):
        return None

    original_width = box.x_max - box.x_min
    original_height = box.y_max - box.y_min
    if original_width <= 0 or original_height <= 0:
        return None

    image_area = float(camera.width * camera.height)
    original_area = original_width * original_height
    if labels.clip_boxes_to_image:
        x_min = min(max(box.x_min, 0.0), float(camera.width))
        y_min = min(max(box.y_min, 0.0), float(camera.height))
        x_max = min(max(box.x_max, 0.0), float(camera.width))
        y_max = min(max(box.y_max, 0.0), float(camera.height))
    else:
        if (
            box.x_min < 0
            or box.y_min < 0
            or box.x_max > camera.width
            or box.y_max > camera.height
        ):
            return None
        x_min, y_min, x_max, y_max = coordinates

    width = x_max - x_min
    height = y_max - y_min
    clipped_area = width * height
    if width <= 0 or height <= 0 or clipped_area < pedestrians.min_box_area_pixels:
        return None
    if clipped_area / original_area < pedestrians.min_visible_fraction:
        return None
    if clipped_area > image_area:
        return None

    return YoloAnnotation(
        class_id=labels.pedestrian_class_id,
        center_x=((x_min + x_max) / 2) / camera.width,
        center_y=((y_min + y_max) / 2) / camera.height,
        width=width / camera.width,
        height=height / camera.height,
    )