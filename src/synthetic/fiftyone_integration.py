"""Optional FiftyOne dataset loading helpers for interactive inspection."""

from __future__ import annotations

from typing import Literal

from .config import SyntheticConfig


def load_fiftyone_dataset(
    config: SyntheticConfig,
    split: Literal["train", "val"],
    name: str | None = None,
):
    """Import a generated split as a temporary FiftyOne dataset for inspection."""
    import fiftyone as fo
    from fiftyone.types import YOLOv5Dataset

    return fo.Dataset.from_dir(
        dataset_dir=str(config.output.root),
        dataset_type=YOLOv5Dataset,
        name=name,
        persistent=False,
        split=split,
        label_type="detections",
        include_all_data=True,
    )