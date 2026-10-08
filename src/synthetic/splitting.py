"""Deterministic dataset split assignment."""

import hashlib
from typing import Literal

from .config import SyntheticConfig


def split_for_scene(scene_id: str, config: SyntheticConfig) -> Literal["train", "val"]:
    """Assign all frames from a scene to one deterministic split."""
    if not scene_id:
        raise ValueError("scene_id must not be empty")
    digest = hashlib.sha256(f"{config.generation.seed}:{scene_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big") / 2**64
    return "train" if bucket < config.generation.train_fraction else "val"