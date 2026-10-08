"""Command-line tools for auditing/exporting a fixed Waymo evaluation set."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .waymo_eval_config import load_waymo_eval_config
from .waymo_pedestrian_eval import (
    audit_waymo_eval_candidates,
    export_waymo_eval_dataset,
    load_waymo_eval_candidates,
    select_waymo_eval_frames,
)
from .waymo_eval_validation import validate_waymo_eval_dataset
from .fiftyone_integration import launch_fiftyone_app, load_fiftyone_dataset


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "waymo_eval.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit and export Waymo pedestrian evaluation data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--audit", action="store_true", help="summarize scenes, cameras, and track frequencies")
    action.add_argument("--plan", action="store_true", help="report deterministic selection counts without writing images")
    action.add_argument("--export", action="store_true", help="export the selected image/label evaluation set")
    action.add_argument("--validate", action="store_true", help="validate an existing exported evaluation set")
    action.add_argument(
        "--fiftyone",
        action="store_true",
        help="load the exported test split and open the FiftyOne App",
    )
    args = parser.parse_args(argv)

    config = load_waymo_eval_config(args.config)
    if args.fiftyone:
        dataset = load_fiftyone_dataset(config.output.root, split="test")
        print(f"Loaded FiftyOne dataset {dataset.name!r}")
        launch_fiftyone_app(dataset)
        return 0
    if args.validate:
        errors = validate_waymo_eval_dataset(config)
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return 0 if not errors else 1

    candidates = load_waymo_eval_candidates(config)
    if args.audit:
        result = audit_waymo_eval_candidates(candidates)
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return 0

    selection = select_waymo_eval_frames(candidates, config)
    if args.plan:
        frequency_histogram = {}
        for frequency in selection.identity_frequencies.values():
            frequency_histogram[frequency] = frequency_histogram.get(frequency, 0) + 1
        result = {
            "positive_frames": selection.positive_frames,
            "negative_frames": selection.negative_frames,
            "total_frames": len(selection.selected_frames),
            "negative_fraction": selection.negative_frames / max(len(selection.selected_frames), 1),
            "under_target_identities": len(selection.under_target_identities),
            "achieved_appearance_histogram": dict(sorted(frequency_histogram.items())),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    exported = export_waymo_eval_dataset(config, candidates)
    print(
        json.dumps(
            {
                "output_root": str(config.output.root),
                "positive_frames": exported.positive_frames,
                "negative_frames": exported.negative_frames,
                "total_frames": len(exported.selected_frames),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())