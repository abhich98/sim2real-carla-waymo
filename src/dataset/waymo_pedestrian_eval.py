"""Track-aware export of a labeled Waymo camera evaluation set."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

from .waymo_eval_config import WaymoEvalConfig


@dataclass(frozen=True, order=True, slots=True)
class EvalFrameKey:
    context_name: str
    camera_name: int
    timestamp_micros: int


@dataclass(frozen=True, slots=True)
class EvalPedestrianBox:
    camera_object_id: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True, slots=True)
class EvalFrameCandidate:
    key: EvalFrameKey
    boxes: tuple[EvalPedestrianBox, ...]


@dataclass(frozen=True, slots=True)
class WaymoEvalAudit:
    contexts: int
    image_camera_samples: int
    positive_image_camera_samples: int
    negative_image_camera_samples: int
    pedestrian_box_rows: int
    camera_scoped_identities: int
    repeated_raw_ids_across_cameras: int
    track_frequency_histogram: dict[int, int]
    samples_by_context: dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class WaymoEvalSelection:
    selected_frames: tuple[EvalFrameCandidate, ...]
    positive_frames: int
    negative_frames: int
    identity_frequencies: dict[str, int]
    under_target_identities: tuple[str, ...]


def _component_path(root: Path, component: str, context_name: str) -> Path:
    return root / component / f"{context_name}.parquet"


def load_waymo_eval_candidates(config: WaymoEvalConfig) -> tuple[EvalFrameCandidate, ...]:
    """Load image keys and pedestrian boxes while leaving JPEG payloads on disk."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("install the waymo_eval dependency group to read Parquet data") from exc

    candidates: list[EvalFrameCandidate] = []
    camera_filter = set(config.source.cameras)
    for context_name in config.source.contexts:
        image_path = _component_path(config.source.root, "camera_image", context_name)
        box_path = _component_path(config.source.root, "camera_box", context_name)
        if not image_path.is_file() or not box_path.is_file():
            raise FileNotFoundError(
                f"missing camera_image or camera_box Parquet file for context {context_name}"
            )

        image_table = pq.read_table(
            image_path,
            columns=["key.camera_name", "key.frame_timestamp_micros"],
        )
        image_keys: set[tuple[int, int]] = set()
        for row in image_table.to_pylist():
            camera_name = int(row["key.camera_name"])
            if camera_name in camera_filter:
                key = (camera_name, int(row["key.frame_timestamp_micros"]))
                if key in image_keys:
                    raise ValueError(f"duplicate camera_image key in {context_name}: {key}")
                image_keys.add(key)

        box_columns = [
            "key.camera_name",
            "key.camera_object_id",
            "key.frame_timestamp_micros",
            "[CameraBoxComponent].type",
            "[CameraBoxComponent].box.center.x",
            "[CameraBoxComponent].box.center.y",
            "[CameraBoxComponent].box.size.x",
            "[CameraBoxComponent].box.size.y",
        ]
        box_table = pq.read_table(box_path, columns=box_columns)
        boxes_by_key: dict[tuple[int, int], list[EvalPedestrianBox]] = defaultdict(list)
        for row in box_table.to_pylist():
            camera_name = int(row["key.camera_name"])
            if (
                camera_name not in camera_filter
                or row["[CameraBoxComponent].type"] != config.source.pedestrian_type_id
            ):
                continue
            center_x = float(row["[CameraBoxComponent].box.center.x"])
            center_y = float(row["[CameraBoxComponent].box.center.y"])
            width = float(row["[CameraBoxComponent].box.size.x"])
            height = float(row["[CameraBoxComponent].box.size.y"])
            object_id = row["key.camera_object_id"]
            if object_id is None or width <= 0 or height <= 0:
                continue
            frame_key = (camera_name, int(row["key.frame_timestamp_micros"]))
            if frame_key not in image_keys:
                raise ValueError(
                    f"pedestrian annotation has no matching camera image: "
                    f"{context_name}/{frame_key}"
                )
            boxes_by_key[frame_key].append(
                EvalPedestrianBox(
                    camera_object_id=str(object_id),
                    x_min=center_x - width / 2,
                    y_min=center_y - height / 2,
                    x_max=center_x + width / 2,
                    y_max=center_y + height / 2,
                )
            )

        candidates.extend(
            EvalFrameCandidate(
                key=EvalFrameKey(context_name, camera_name, timestamp_micros),
                boxes=tuple(
                    sorted(
                        boxes_by_key.get((camera_name, timestamp_micros), []),
                        key=lambda box: box.camera_object_id,
                    )
                ),
            )
            for camera_name, timestamp_micros in sorted(image_keys)
        )

    return tuple(sorted(candidates, key=lambda candidate: candidate.key))


def audit_waymo_eval_candidates(candidates: tuple[EvalFrameCandidate, ...]) -> WaymoEvalAudit:
    """Summarize pedestrian identities, cross-camera ID reuse, and image negatives."""
    identity_frequency: Counter[tuple[str, int, str]] = Counter()
    raw_id_cameras: dict[tuple[str, str], set[int]] = defaultdict(set)
    box_rows = 0
    sample_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"images": 0, "positive_images": 0, "negative_images": 0, "pedestrian_boxes": 0}
    )
    for candidate in candidates:
        sample_counts[candidate.key.context_name]["images"] += 1
        if candidate.boxes:
            sample_counts[candidate.key.context_name]["positive_images"] += 1
        else:
            sample_counts[candidate.key.context_name]["negative_images"] += 1
        for box in candidate.boxes:
            identity = (candidate.key.context_name, candidate.key.camera_name, box.camera_object_id)
            identity_frequency[identity] += 1
            raw_id_cameras[(candidate.key.context_name, box.camera_object_id)].add(
                candidate.key.camera_name
            )
            box_rows += 1
            sample_counts[candidate.key.context_name]["pedestrian_boxes"] += 1

    histogram = Counter(identity_frequency.values())
    return WaymoEvalAudit(
        contexts=len(sample_counts),
        image_camera_samples=len(candidates),
        positive_image_camera_samples=sum(bool(candidate.boxes) for candidate in candidates),
        negative_image_camera_samples=sum(not candidate.boxes for candidate in candidates),
        pedestrian_box_rows=box_rows,
        camera_scoped_identities=len(identity_frequency),
        repeated_raw_ids_across_cameras=sum(
            len(camera_ids) > 1 for camera_ids in raw_id_cameras.values()
        ),
        track_frequency_histogram=dict(sorted(histogram.items())),
        samples_by_context={context: values for context, values in sorted(sample_counts.items())},
    )


def _identity_key(box: EvalPedestrianBox, key: EvalFrameKey) -> tuple[str, int, str]:
    return key.context_name, key.camera_name, box.camera_object_id


def _stable_rank(seed: int, key: EvalFrameKey) -> str:
    encoded = f"{seed}:{key.context_name}:{key.camera_name}:{key.timestamp_micros}"
    return sha256(encoded.encode("utf-8")).hexdigest()


def _negative_samples(
    candidates: list[EvalFrameCandidate], count: int, seed: int
) -> list[EvalFrameCandidate]:
    groups: dict[tuple[str, int], list[EvalFrameCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[(candidate.key.context_name, candidate.key.camera_name)].append(candidate)
    for group in groups.values():
        group.sort(key=lambda candidate: _stable_rank(seed, candidate.key))

    chosen: list[EvalFrameCandidate] = []
    group_keys = sorted(groups)
    while len(chosen) < count:
        advanced = False
        for group_key in group_keys:
            group = groups[group_key]
            if group:
                chosen.append(group.pop(0))
                advanced = True
                if len(chosen) == count:
                    break
        if not advanced:
            break
    return chosen


def select_waymo_eval_frames(
    candidates: tuple[EvalFrameCandidate, ...], config: WaymoEvalConfig
) -> WaymoEvalSelection:
    """Select positive images to target three views per identity, with a hard cap."""
    positives = [candidate for candidate in candidates if candidate.boxes]
    counts: Counter[tuple[str, int, str]] = Counter()
    selected: list[EvalFrameCandidate] = []
    selected_keys: set[EvalFrameKey] = set()
    selected_times: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    time_ranges: dict[tuple[str, int, str], tuple[int, int]] = {}
    for candidate in positives:
        for box in candidate.boxes:
            identity = _identity_key(box, candidate.key)
            if identity not in time_ranges:
                time_ranges[identity] = (
                    candidate.key.timestamp_micros,
                    candidate.key.timestamp_micros,
                )
            else:
                lower, upper = time_ranges[identity]
                time_ranges[identity] = (
                    min(lower, candidate.key.timestamp_micros),
                    max(upper, candidate.key.timestamp_micros),
                )

    target = config.sampling.target_appearances_per_identity
    cap = config.sampling.max_appearances_per_identity
    while True:
        best: EvalFrameCandidate | None = None
        best_score: tuple[int, float, str] | None = None
        for candidate in positives:
            if candidate.key in selected_keys:
                continue
            identities = {_identity_key(box, candidate.key) for box in candidate.boxes}
            if any(counts[identity] >= cap for identity in identities):
                continue
            deficient = [identity for identity in identities if counts[identity] < target]
            if not deficient:
                continue
            deficit_score = sum(target - counts[identity] for identity in deficient)
            spacing_score = 0.0
            for identity in deficient:
                previously_selected = selected_times[identity]
                lower, upper = time_ranges[identity]
                duration = max(upper - lower, 1)
                min_spacing = (
                    min(abs(candidate.key.timestamp_micros - timestamp) for timestamp in previously_selected)
                    / duration
                    if previously_selected
                    else 1.0
                )
                spacing_score += min_spacing
            score = (deficit_score, spacing_score, _stable_rank(config.sampling.seed, candidate.key))
            if best_score is None or score > best_score:
                best = candidate
                best_score = score
        if best is None:
            break
        selected.append(best)
        selected_keys.add(best.key)
        for box in best.boxes:
            identity = _identity_key(box, best.key)
            counts[identity] += 1
            selected_times[identity].append(best.key.timestamp_micros)

    negative_target = math.ceil(
        len(selected)
        * config.sampling.negative_fraction
        / (1 - config.sampling.negative_fraction)
    )
    negatives = _negative_samples(
        [candidate for candidate in candidates if not candidate.boxes],
        negative_target,
        config.sampling.seed,
    )
    all_identities = {
        _identity_key(box, candidate.key)
        for candidate in positives
        for box in candidate.boxes
    }
    frequency_names = {
        f"{context}|camera_{camera}|{object_id}": count
        for (context, camera, object_id), count in sorted(counts.items())
    }
    under_target = tuple(
        f"{context}|camera_{camera}|{object_id}"
        for context, camera, object_id in sorted(all_identities)
        if counts[(context, camera, object_id)] < target
    )
    return WaymoEvalSelection(
        selected_frames=tuple(sorted(selected + negatives, key=lambda candidate: candidate.key)),
        positive_frames=len(selected),
        negative_frames=len(negatives),
        identity_frequencies=frequency_names,
        under_target_identities=under_target,
    )


def _yolo_line(box: EvalPedestrianBox, image_width: int, image_height: int, class_id: int) -> str | None:
    x_min = min(max(box.x_min, 0.0), float(image_width))
    y_min = min(max(box.y_min, 0.0), float(image_height))
    x_max = min(max(box.x_max, 0.0), float(image_width))
    y_max = min(max(box.y_max, 0.0), float(image_height))
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return None
    values = (
        ((x_min + x_max) / 2) / image_width,
        ((y_min + y_max) / 2) / image_height,
        width / image_width,
        height / image_height,
    )
    return f"{class_id} " + " ".join(f"{value:.8f}" for value in values)


def export_waymo_eval_dataset(
    config: WaymoEvalConfig,
    candidates: tuple[EvalFrameCandidate, ...] | None = None,
) -> WaymoEvalSelection:
    """Export selected original JPEGs and YOLO labels using bounded Parquet batches."""
    import pyarrow.parquet as pq
    import yaml

    candidates = candidates if candidates is not None else load_waymo_eval_candidates(config)
    selection = select_waymo_eval_frames(candidates, config)
    output_root = config.output.root
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"evaluation output directory is not empty: {output_root}")
    image_dir = output_root / "images" / "test"
    label_dir = output_root / "labels" / "test"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "test": "images/test",
                "names": [config.output.pedestrian_class_name],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    selected_by_context: dict[str, dict[tuple[int, int], EvalFrameCandidate]] = defaultdict(dict)
    for candidate in selection.selected_frames:
        selected_by_context[candidate.key.context_name][
            (candidate.key.camera_name, candidate.key.timestamp_micros)
        ] = candidate

    manifest_path = output_root / "manifest.jsonl"
    written_keys: set[EvalFrameKey] = set()
    with manifest_path.open("x", encoding="utf-8") as manifest_file:
        for context_name in config.source.contexts:
            wanted = selected_by_context.get(context_name, {})
            if not wanted:
                continue
            image_path = _component_path(config.source.root, "camera_image", context_name)
            parquet_file = pq.ParquetFile(image_path)
            for batch in parquet_file.iter_batches(
                columns=[
                    "key.camera_name",
                    "key.frame_timestamp_micros",
                    "[CameraImageComponent].image",
                ],
                batch_size=16,
            ):
                for row in batch.to_pylist():
                    key_tuple = (
                        int(row["key.camera_name"]),
                        int(row["key.frame_timestamp_micros"]),
                    )
                    candidate = wanted.get(key_tuple)
                    if candidate is None:
                        continue
                    key = candidate.key
                    if key in written_keys:
                        raise ValueError(f"duplicate image key during export: {key}")
                    encoded_image = row["[CameraImageComponent].image"]
                    with Image.open(BytesIO(encoded_image)) as image:
                        image_width, image_height = image.size
                    sample_id = f"{context_name}_cam{key.camera_name}_{key.timestamp_micros}"
                    image_relative = f"images/test/{sample_id}.jpg"
                    label_relative = f"labels/test/{sample_id}.txt"
                    destination_image = output_root / image_relative
                    temporary_image = destination_image.with_name(f".{destination_image.name}.tmp")
                    temporary_image.write_bytes(encoded_image)
                    temporary_image.replace(destination_image)

                    annotations: list[dict[str, Any]] = []
                    label_lines: list[str] = []
                    for box in candidate.boxes:
                        line = _yolo_line(
                            box,
                            image_width,
                            image_height,
                            config.output.pedestrian_class_id,
                        )
                        if line is None:
                            raise ValueError(f"degenerate source box for {sample_id}: {box}")
                        label_lines.append(line)
                        annotations.append(
                            {
                                "camera_object_id": box.camera_object_id,
                                "box_xyxy_pixels": [box.x_min, box.y_min, box.x_max, box.y_max],
                                "yolo": line,
                            }
                        )
                    label_text = "\n".join(label_lines)
                    if label_text:
                        label_text += "\n"
                    (output_root / label_relative).write_text(label_text, encoding="utf-8")

                    record = {
                        "sample_id": sample_id,
                        "scene_id": context_name,
                        "timestamp_micros": key.timestamp_micros,
                        "camera_name": key.camera_name,
                        "image": image_relative,
                        "label": label_relative,
                        "width": image_width,
                        "height": image_height,
                        "pedestrian_count": len(annotations),
                        "pedestrians": annotations,
                    }
                    manifest_file.write(json.dumps(record, sort_keys=True) + "\n")
                    written_keys.add(key)

    missing_keys = {candidate.key for candidate in selection.selected_frames} - written_keys
    if missing_keys:
        raise LookupError(f"selected camera images were missing during export: {len(missing_keys)}")

    report = {
        "requested_target_appearances": config.sampling.target_appearances_per_identity,
        "maximum_appearances": config.sampling.max_appearances_per_identity,
        "positive_frames": selection.positive_frames,
        "negative_frames": selection.negative_frames,
        "negative_fraction_achieved": selection.negative_frames
        / max(len(selection.selected_frames), 1),
        "under_target_identities": list(selection.under_target_identities),
        "identity_frequencies": selection.identity_frequencies,
    }
    (output_root / "sampling_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return selection