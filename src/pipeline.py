from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.dataset import WaymoLoader
from src.preprocessing import (
    radius_outlier_removal,
    statistical_outlier_removal,
    voxel_downsample,
)


@dataclass(slots=True)
class Milestone123Result:
    frame_index: int
    lidar_raw: np.ndarray
    lidar_voxel: np.ndarray
    lidar_sor: np.ndarray
    lidar_ror: np.ndarray
    front_camera: np.ndarray
    calibration: Any


def run_milestones_1_to_3(
    loader: WaymoLoader,
    frame_index: int,
    camera_name: str | int | None = None,
    voxel_size: float = 0.10,
    sor_nb_neighbors: int = 20,
    sor_std_ratio: float = 2.0,
    ror_radius: float = 0.5,
    ror_min_points: int = 16,
) -> Milestone123Result:
    frame = loader.load_frame(
        frame_index=frame_index,
        return_array=True,
        camera_name=camera_name,
    )
    lidar_raw = frame["lidar"]
    lidar_voxel = voxel_downsample(lidar_raw, voxel_size=voxel_size, return_array=True)
    lidar_sor = statistical_outlier_removal(
        lidar_voxel,
        nb_neighbors=sor_nb_neighbors,
        std_ratio=sor_std_ratio,
        return_array=True,
    )
    lidar_ror = radius_outlier_removal(
        lidar_voxel,
        radius=ror_radius,
        min_points=ror_min_points,
        return_array=True,
    )

    return Milestone123Result(
        frame_index=frame_index,
        lidar_raw=lidar_raw,
        lidar_voxel=lidar_voxel,
        lidar_sor=lidar_sor,
        lidar_ror=lidar_ror,
        front_camera=frame["front_camera"],
        calibration=frame["calibration"],
    )
