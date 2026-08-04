from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from open3d import geometry, utility, visualization


def show_point_cloud(points: np.ndarray, window_name: str = "Waymo LiDAR") -> None:
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1] < 3:
        raise ValueError("points must have shape (N, >=3)")

    point_cloud = geometry.PointCloud()
    point_cloud.points = utility.Vector3dVector(points_array[:, :3])
    visualization.draw_geometries([point_cloud], window_name=window_name)


def show_camera_image(image: np.ndarray, title: str = "Front Camera") -> None:
    image_array = np.asarray(image)
    plt.figure(figsize=(12, 7))
    plt.title(title)
    plt.imshow(image_array)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
