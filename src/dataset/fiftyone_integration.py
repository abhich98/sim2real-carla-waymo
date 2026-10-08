"""Shared FiftyOne helpers for YOLO-formatted project datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal


FiftyOneSplit = Literal["train", "val", "test"]


def load_fiftyone_dataset(
    dataset_root: str | Path,
    split: FiftyOneSplit,
    name: str | None = None,
) -> Any:
    """Import a YOLOv5-format split into a temporary FiftyOne dataset."""
    from fiftyone import Dataset
    from fiftyone.types import YOLOv5Dataset

    return Dataset.from_dir(
        dataset_dir=str(dataset_root),
        dataset_type=YOLOv5Dataset,
        name=name,
        persistent=False,
        split=split,
        label_type="detections",
        include_all_data=True,
    )


def launch_fiftyone_app(dataset: Any) -> None:
    """Open a FiftyOne dataset in the App and block until the session closes."""
    import fiftyone as fo

    fo.launch_app(dataset).wait()