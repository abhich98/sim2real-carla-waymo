from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from open3d import geometry, utility

from src.utils.lidar_data import as_point_cloud


@dataclass(slots=True)
class GroundSegmentationResult:
    plane_model: np.ndarray
    inlier_indices: np.ndarray
    ground_points: np.ndarray
    non_ground_points: np.ndarray


def segment_ground_plane(
    points: np.ndarray,
    distance_threshold: float = 0.2,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> GroundSegmentationResult:
    point_cloud = as_point_cloud(points)
    plane_model, inliers = point_cloud.segment_plane(
        distance_threshold=float(distance_threshold),
        ransac_n=int(ransac_n),
        num_iterations=int(num_iterations),
    )

    points_array = np.asarray(point_cloud.points)
    inlier_indices = np.asarray(inliers, dtype=np.int32)

    ground_mask = np.zeros(points_array.shape[0], dtype=bool)
    ground_mask[inlier_indices] = True

    return GroundSegmentationResult(
        plane_model=np.asarray(plane_model, dtype=np.float64),
        inlier_indices=inlier_indices,
        ground_points=points_array[ground_mask],
        non_ground_points=points_array[~ground_mask],
    )