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


def statistical_outlier_removal(
    points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    return_array: bool = True,
) -> np.ndarray | geometry.PointCloud:
    point_cloud = _as_point_cloud(points)
    filtered_cloud, _ = point_cloud.remove_statistical_outlier(
        nb_neighbors=int(nb_neighbors),
        std_ratio=float(std_ratio),
    )
    if not return_array:
        return filtered_cloud
    return np.asarray(filtered_cloud.points)


def radius_outlier_removal(
    points: np.ndarray,
    radius: float = 0.5,
    min_points: int = 16,
    return_array: bool = True,
) -> np.ndarray | geometry.PointCloud:
    point_cloud = _as_point_cloud(points)
    filtered_cloud, _ = point_cloud.remove_radius_outlier(
        nb_points=int(min_points),
        radius=float(radius),
    )
    if not return_array:
        return filtered_cloud
    return np.asarray(filtered_cloud.points)
