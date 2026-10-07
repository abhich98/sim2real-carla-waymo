from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from open3d import geometry, utility


def _as_point_cloud(points: np.ndarray) -> geometry.PointCloud:
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1] < 3:
        raise ValueError("points must have shape (N, >=3)")

    point_cloud = geometry.PointCloud()
    point_cloud.points = utility.Vector3dVector(points_array[:, :3])
    return point_cloud


@dataclass(slots=True)
class DBSCANResult:
    labels: np.ndarray
    cluster_indices: list[np.ndarray]
    clusters: list[np.ndarray]
    noise_points: np.ndarray


def run_dbscan(
    points: np.ndarray,
    eps: float = 0.8,
    min_points: int = 10,
    min_cluster_size: int = 30,
) -> DBSCANResult:
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1] < 3:
        raise ValueError("points must have shape (N, >=3)")

    if points_array.shape[0] == 0:
        empty_labels = np.empty((0,), dtype=np.int32)
        empty_points = np.empty((0, 3), dtype=np.float64)
        return DBSCANResult(
            labels=empty_labels,
            cluster_indices=[],
            clusters=[],
            noise_points=empty_points,
        )

    point_cloud = _as_point_cloud(points_array)
    raw_labels = np.asarray(
        point_cloud.cluster_dbscan(
            eps=float(eps),
            min_points=int(min_points),
            print_progress=False,
        ),
        dtype=np.int32,
    )

    filtered_labels = np.full(raw_labels.shape, -1, dtype=np.int32)
    cluster_indices: list[np.ndarray] = []
    clusters: list[np.ndarray] = []

    next_label = 0
    for label in sorted(set(raw_labels.tolist())):
        if label < 0:
            continue

        indices = np.flatnonzero(raw_labels == label)
        if indices.size < min_cluster_size:
            continue

        filtered_labels[indices] = next_label
        cluster_indices.append(indices)
        clusters.append(points_array[indices])
        next_label += 1

    noise_points = points_array[filtered_labels < 0]

    return DBSCANResult(
        labels=filtered_labels,
        cluster_indices=cluster_indices,
        clusters=clusters,
        noise_points=noise_points,
    )
