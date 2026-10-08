"""YAML configuration for the fixed Waymo evaluation export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class WaymoEvalConfigError(ValueError):
    """Raised when the Waymo evaluation YAML is invalid."""


@dataclass(frozen=True, slots=True)
class WaymoEvalSourceConfig:
    root: Path
    contexts: tuple[str, ...]
    cameras: tuple[int, ...]
    pedestrian_type_id: int

    def __post_init__(self) -> None:
        if not self.contexts or len(set(self.contexts)) != len(self.contexts):
            raise WaymoEvalConfigError("source.contexts must be a non-empty unique list")
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise WaymoEvalConfigError("source.cameras must be a non-empty unique list")
        if any(camera not in {1, 2, 3, 4, 5} for camera in self.cameras):
            raise WaymoEvalConfigError("source.cameras must use Waymo camera IDs 1 through 5")
        if self.pedestrian_type_id < 0:
            raise WaymoEvalConfigError("source.pedestrian_type_id must be non-negative")


@dataclass(frozen=True, slots=True)
class WaymoEvalSamplingConfig:
    target_appearances_per_identity: int
    max_appearances_per_identity: int
    negative_fraction: float
    seed: int

    def __post_init__(self) -> None:
        if self.target_appearances_per_identity <= 0:
            raise WaymoEvalConfigError("sampling.target_appearances_per_identity must be positive")
        if self.max_appearances_per_identity < self.target_appearances_per_identity:
            raise WaymoEvalConfigError(
                "sampling.max_appearances_per_identity must be >= target appearances"
            )
        if not 0 <= self.negative_fraction < 1:
            raise WaymoEvalConfigError("sampling.negative_fraction must be in [0, 1)")
        if self.seed < 0:
            raise WaymoEvalConfigError("sampling.seed must be non-negative")


@dataclass(frozen=True, slots=True)
class WaymoEvalOutputConfig:
    root: Path
    image_format: str
    pedestrian_class_id: int
    pedestrian_class_name: str

    def __post_init__(self) -> None:
        if self.image_format not in {"jpg", "jpeg"}:
            raise WaymoEvalConfigError("output.image_format must be jpg or jpeg")
        if self.pedestrian_class_id < 0 or not self.pedestrian_class_name.strip():
            raise WaymoEvalConfigError("output pedestrian class configuration is invalid")


@dataclass(frozen=True, slots=True)
class WaymoEvalConfig:
    source: WaymoEvalSourceConfig
    sampling: WaymoEvalSamplingConfig
    output: WaymoEvalOutputConfig


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise WaymoEvalConfigError(f"configuration section '{name}' must be a mapping")
    return value


def load_waymo_eval_config(path: str | Path) -> WaymoEvalConfig:
    """Load and validate a Waymo evaluation config from YAML."""
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file)
    except OSError as exc:
        raise WaymoEvalConfigError(f"cannot read config '{config_path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise WaymoEvalConfigError(f"invalid YAML in '{config_path}': {exc}") from exc
    if not isinstance(data, dict):
        raise WaymoEvalConfigError("configuration root must be a mapping")

    try:
        source = _section(data, "source")
        sampling = _section(data, "sampling")
        output = _section(data, "output")
        return WaymoEvalConfig(
            source=WaymoEvalSourceConfig(
                root=Path(source["root"]),
                contexts=tuple(str(context) for context in source["contexts"]),
                cameras=tuple(int(camera) for camera in source["cameras"]),
                pedestrian_type_id=int(source["pedestrian_type_id"]),
            ),
            sampling=WaymoEvalSamplingConfig(**sampling),
            output=WaymoEvalOutputConfig(
                root=Path(output["root"]),
                image_format=output["image_format"],
                pedestrian_class_id=output["pedestrian_class_id"],
                pedestrian_class_name=output["pedestrian_class_name"],
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, WaymoEvalConfigError):
            raise
        raise WaymoEvalConfigError(f"missing or invalid evaluation config field: {exc}") from exc