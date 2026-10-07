from __future__ import annotations

import numpy as np
from open3d import geometry, utility

from src.utils.lidar_data import as_point_cloud


def statistical_outlier_removal(
    points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    return_array: bool = True,
) -> np.ndarray | geometry.PointCloud:
    point_cloud = as_point_cloud(points)
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
    point_cloud = as_point_cloud(points)
    filtered_cloud, _ = point_cloud.remove_radius_outlier(
        nb_points=int(min_points),
        radius=float(radius),
    )
    if not return_array:
        return filtered_cloud
    return np.asarray(filtered_cloud.points)
