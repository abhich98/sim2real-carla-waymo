"""YAML loading and validation for synthetic-data generation settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class SyntheticConfigError(ValueError):
    """Raised when the synthetic generation YAML is invalid."""


@dataclass(frozen=True, slots=True)
class CarlaConfig:
    host: str
    port: int
    traffic_manager_port: int
    timeout_seconds: float
    map_name: str | None

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise SyntheticConfigError("carla.host must not be empty")
        if not 1 <= self.port <= 65535:
            raise SyntheticConfigError("carla.port must be between 1 and 65535")
        if not 1 <= self.traffic_manager_port <= 65535:
            raise SyntheticConfigError("carla.traffic_manager_port must be between 1 and 65535")
        if self.timeout_seconds <= 0:
            raise SyntheticConfigError("carla.timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    num_frames: int
    frames_per_scene: int
    seed: int
    train_fraction: float
    split_unit: str
    fixed_delta_seconds: float
    ticks_per_frame: int
    warmup_ticks_per_scene: int
    max_scene_attempts: int

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise SyntheticConfigError("generation.num_frames must be positive")
        if self.frames_per_scene <= 0:
            raise SyntheticConfigError("generation.frames_per_scene must be positive")
        if self.seed < 0:
            raise SyntheticConfigError("generation.seed must be non-negative")
        if not 0 < self.train_fraction < 1:
            raise SyntheticConfigError("generation.train_fraction must be between 0 and 1")
        if self.split_unit != "scene":
            raise SyntheticConfigError("generation.split_unit must be 'scene'")
        if self.fixed_delta_seconds <= 0:
            raise SyntheticConfigError("generation.fixed_delta_seconds must be positive")
        if self.ticks_per_frame <= 0:
            raise SyntheticConfigError("generation.ticks_per_frame must be positive")
        if self.warmup_ticks_per_scene < 0:
            raise SyntheticConfigError("generation.warmup_ticks_per_scene must be non-negative")
        if self.max_scene_attempts <= 0:
            raise SyntheticConfigError("generation.max_scene_attempts must be positive")


@dataclass(frozen=True, slots=True)
class CameraConfig:
    width: int
    height: int
    field_of_view_degrees: float
    mount_x_meters: float
    mount_y_meters: float
    mount_z_meters: float
    projection_near_plane_meters: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise SyntheticConfigError("camera.width and camera.height must be positive")
        if not 0 < self.field_of_view_degrees < 180:
            raise SyntheticConfigError("camera.field_of_view_degrees must be between 0 and 180")
        if self.projection_near_plane_meters <= 0:
            raise SyntheticConfigError("camera.projection_near_plane_meters must be positive")


@dataclass(frozen=True, slots=True)
class PedestrianConfig:
    min_per_scene: int
    max_per_scene: int
    min_distance_meters: float
    max_distance_meters: float
    min_visible_fraction: float
    min_box_area_pixels: float
    front_spawn_cosine: float
    spawn_attempts_per_pedestrian: int
    min_walk_speed_meters_per_second: float
    max_walk_speed_meters_per_second: float

    def __post_init__(self) -> None:
        if self.min_per_scene < 0 or self.max_per_scene < self.min_per_scene:
            raise SyntheticConfigError("pedestrian count range is invalid")
        if self.min_distance_meters < 0 or self.max_distance_meters <= self.min_distance_meters:
            raise SyntheticConfigError("pedestrian distance range is invalid")
        if not 0 < self.min_visible_fraction <= 1:
            raise SyntheticConfigError("pedestrians.min_visible_fraction must be in (0, 1]")
        if self.min_box_area_pixels <= 0:
            raise SyntheticConfigError("pedestrians.min_box_area_pixels must be positive")
        if not 0 <= self.front_spawn_cosine <= 1:
            raise SyntheticConfigError("pedestrians.front_spawn_cosine must be in [0, 1]")
        if self.spawn_attempts_per_pedestrian <= 0:
            raise SyntheticConfigError("pedestrians.spawn_attempts_per_pedestrian must be positive")
        if (
            self.min_walk_speed_meters_per_second <= 0
            or self.max_walk_speed_meters_per_second < self.min_walk_speed_meters_per_second
        ):
            raise SyntheticConfigError("pedestrian walk speed range is invalid")


@dataclass(frozen=True, slots=True)
class RandomizationConfig:
    weather_presets: list[str]
    time_of_day_hours: list[int]
    min_vehicles: int
    max_vehicles: int
    min_vehicle_distance_from_camera_meters: float
    ego_autopilot: bool
    background_vehicle_autopilot: bool

    def __post_init__(self) -> None:
        if not self.weather_presets or any(not value for value in self.weather_presets):
            raise SyntheticConfigError("randomization.weather_presets must contain values")
        if not self.time_of_day_hours or any(not 0 <= hour <= 23 for hour in self.time_of_day_hours):
            raise SyntheticConfigError("randomization.time_of_day_hours must contain hours from 0 to 23")
        if self.min_vehicles < 0 or self.max_vehicles < self.min_vehicles:
            raise SyntheticConfigError("randomization vehicle count range is invalid")
        if self.min_vehicle_distance_from_camera_meters < 0:
            raise SyntheticConfigError("minimum vehicle distance must be non-negative")


@dataclass(frozen=True, slots=True)
class OcclusionConfig:
    pedestrian_semantic_tag: int
    depth_tolerance_meters: float
    min_visible_pixels: int
    min_visible_pixel_fraction: float

    def __post_init__(self) -> None:
        if not 0 <= self.pedestrian_semantic_tag <= 255:
            raise SyntheticConfigError("occlusion.pedestrian_semantic_tag must be in [0, 255]")
        if self.depth_tolerance_meters <= 0:
            raise SyntheticConfigError("occlusion.depth_tolerance_meters must be positive")
        if self.min_visible_pixels <= 0:
            raise SyntheticConfigError("occlusion.min_visible_pixels must be positive")
        if not 0 < self.min_visible_pixel_fraction <= 1:
            raise SyntheticConfigError("occlusion.min_visible_pixel_fraction must be in (0, 1]")


BOX_SOURCES = frozenset({"segmentation", "projected_3d"})


@dataclass(frozen=True, slots=True)
class LabelConfig:
    format: str
    pedestrian_class_id: int
    pedestrian_class_name: str
    clip_boxes_to_image: bool
    box_source: str

    def __post_init__(self) -> None:
        if self.format != "yolo":
            raise SyntheticConfigError("labels.format must be 'yolo'")
        if self.box_source not in BOX_SOURCES:
            raise SyntheticConfigError(
                f"labels.box_source must be one of {sorted(BOX_SOURCES)}"
            )
        if self.pedestrian_class_id < 0 or not self.pedestrian_class_name.strip():
            raise SyntheticConfigError("pedestrian label class metadata is invalid")


@dataclass(frozen=True, slots=True)
class OutputConfig:
    root: Path
    image_format: str

    def __post_init__(self) -> None:
        if self.image_format not in {"png", "jpg", "jpeg"}:
            raise SyntheticConfigError("output.image_format must be png, jpg, or jpeg")


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    carla: CarlaConfig
    generation: GenerationConfig
    camera: CameraConfig
    pedestrians: PedestrianConfig
    randomization: RandomizationConfig
    occlusion: OcclusionConfig
    labels: LabelConfig
    output: OutputConfig


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise SyntheticConfigError(f"configuration section '{name}' must be a mapping")
    return value


def load_synthetic_config(path: str | Path) -> SyntheticConfig:
    """Load and validate generation settings from a YAML file."""
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file)
    except OSError as exc:
        raise SyntheticConfigError(f"cannot read config file '{config_path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise SyntheticConfigError(f"invalid YAML in '{config_path}': {exc}") from exc

    if not isinstance(data, dict):
        raise SyntheticConfigError("configuration root must be a mapping")

    try:
        return SyntheticConfig(
            carla=CarlaConfig(**_section(data, "carla")),
            generation=GenerationConfig(**_section(data, "generation")),
            camera=CameraConfig(**_section(data, "camera")),
            pedestrians=PedestrianConfig(**_section(data, "pedestrians")),
            randomization=RandomizationConfig(**_section(data, "randomization")),
            occlusion=OcclusionConfig(**_section(data, "occlusion")),
            labels=LabelConfig(**_section(data, "labels")),
            output=OutputConfig(
                root=Path(_section(data, "output")["root"]),
                image_format=_section(data, "output")["image_format"],
            ),
        )
    except TypeError as exc:
        raise SyntheticConfigError(f"configuration fields are missing or unknown: {exc}") from exc
    except KeyError as exc:
        raise SyntheticConfigError(f"missing output configuration field: {exc}") from exc