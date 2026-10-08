from pathlib import Path
from typing import Any, Literal

__all__ = ["launch_fiftyone_app", "load_fiftyone_dataset"]


def load_fiftyone_dataset(
	dataset_root: str | Path,
	split: Literal["train", "val", "test"],
	name: str | None = None,
) -> Any:
	from .fiftyone_integration import load_fiftyone_dataset as load_dataset

	return load_dataset(dataset_root, split=split, name=name)


def launch_fiftyone_app(dataset: Any) -> None:
	from .fiftyone_integration import launch_fiftyone_app as launch_app

	launch_app(dataset)


def __getattr__(name: str):
	if name == "WaymoLoader":
		from .waymo_loader import WaymoLoader

		return WaymoLoader
	if name in {"launch_fiftyone_app", "load_fiftyone_dataset"}:
		from . import fiftyone_integration

		return getattr(fiftyone_integration, name)
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
