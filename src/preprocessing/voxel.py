from __future__ import annotations

import numpy as np
from open3d import geometry, utility


def _as_point_cloud(points: np.ndarray) -> geometry.PointCloud:
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1] < 3:
        raise ValueError("points must have shape (N, >=3)")
    point_cloud = geometry.PointCloud()
    point_cloud.points = utility.Vector3dVector(points_array[:, :3])
    return point_cloud


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float = 0.10,
    return_array: bool = True,
) -> np.ndarray | geometry.PointCloud:
    point_cloud = _as_point_cloud(points)
    downsampled = point_cloud.voxel_down_sample(voxel_size=float(voxel_size))
    if not return_array:
        return downsampled
    return np.asarray(downsampled.points)
