"""Command-line entry point for YAML-driven synthetic dataset generation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from .config import load_synthetic_config
from .fiftyone_integration import load_fiftyone_dataset
from .generator import generate_dataset
from .statistics import write_dataset_statistics
from .validation import validate_dataset


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "carla_synthetic.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a CARLA pedestrian image dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print YAML settings without connecting to CARLA or writing data",
    )
    actions.add_argument(
        "--validate-only",
        action="store_true",
        help="validate an existing generated dataset without writing statistics",
    )
    actions.add_argument(
        "--stats-only",
        action="store_true",
        help="collect and write statistics for an existing dataset",
    )
    actions.add_argument(
        "--fiftyone",
        choices=("train", "val"),
        help="load the selected split and open the FiftyOne App",
    )
    args = parser.parse_args(argv)

    config = load_synthetic_config(args.config)
    config_data = asdict(config)
    config_data["output"]["root"] = str(config.output.root)
    print(yaml.safe_dump(config_data, sort_keys=False))
    if args.dry_run:
        return 0
    if args.validate_only:
        report = validate_dataset(config)
        print(yaml.safe_dump(asdict(report), sort_keys=False))
        return 0 if report.valid else 1
    if args.stats_only:
        statistics = write_dataset_statistics(config)
        print(yaml.safe_dump(asdict(statistics), sort_keys=False))
        return 0
    if args.fiftyone:
        dataset = load_fiftyone_dataset(config, split=args.fiftyone)
        print(f"Loaded FiftyOne dataset {dataset.name!r}")
        import fiftyone as fo

        fo.launch_app(dataset).wait()
        return 0

    summary = generate_dataset(config)
    print(summary)
    return 0