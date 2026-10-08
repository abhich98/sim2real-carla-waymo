"""High-level generation API for the synthetic pedestrian dataset."""

from .config import SyntheticConfig
from .annotations import box_to_yolo
from .carla_runtime import CarlaCapture
from .dataset_writer import DatasetWriter
from .splitting import split_for_scene
from .types import GenerationSummary
from .validation import validate_dataset


def generate_dataset(config: SyntheticConfig) -> GenerationSummary:
    """Capture scenes and write a reproducible image/label dataset."""
    written_frames = 0
    with CarlaCapture(config) as capture, DatasetWriter(config) as writer:
        for frame in capture.capture_frames():
            annotations = tuple(
                annotation
                for box in frame.pedestrian_boxes
                if (
                    annotation := box_to_yolo(
                        box,
                        config.camera,
                        config.labels,
                        config.pedestrians,
                    )
                ) is not None
            )
            writer.write_sample(frame, split_for_scene(frame.scene_id, config), annotations)
            written_frames += 1

    if written_frames != config.generation.num_frames:
        raise RuntimeError(
            f"CARLA capture returned {written_frames} frames; "
            f"expected {config.generation.num_frames}"
        )

    report = validate_dataset(config)
    if not report.valid:
        joined_errors = "\n".join(report.errors)
        raise RuntimeError(f"generated dataset failed validation:\n{joined_errors}")

    return GenerationSummary(
        requested_frames=config.generation.num_frames,
        written_frames=written_frames,
        skipped_frames=0,
        output_root=config.output.root,
        manifest_path=config.output.root / "manifest.jsonl",
    )