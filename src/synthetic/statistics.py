"""Dataset statistics collection, separate from validation."""

from __future__ import annotations

import json

from .config import SyntheticConfig
from .types import DatasetStatistics


def collect_dataset_statistics(config: SyntheticConfig) -> DatasetStatistics:
    """Count image files, label files, and non-empty YOLO rows by split."""
    root = config.output.root
    images_by_split: dict[str, int] = {}
    pedestrians_by_split: dict[str, int] = {}
    label_file_count = 0

    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images_by_split[split] = sum(
            1
            for path in image_dir.glob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        label_paths = [path for path in label_dir.glob("*.txt") if path.is_file()]
        label_file_count += len(label_paths)
        pedestrians_by_split[split] = sum(
            1
            for path in label_paths
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    return DatasetStatistics(
        image_count=sum(images_by_split.values()),
        label_file_count=label_file_count,
        pedestrian_count=sum(pedestrians_by_split.values()),
        images_by_split=images_by_split,
        pedestrians_by_split=pedestrians_by_split,
    )


def write_dataset_statistics(config: SyntheticConfig) -> DatasetStatistics:
    """Collect statistics and write them to stats.json explicitly."""
    statistics = collect_dataset_statistics(config)
    output_path = config.output.root / "stats.json"
    output_path.write_text(
        json.dumps(
            {
                "image_count": statistics.image_count,
                "label_file_count": statistics.label_file_count,
                "pedestrian_count": statistics.pedestrian_count,
                "images_by_split": statistics.images_by_split,
                "pedestrians_by_split": statistics.pedestrians_by_split,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return statistics