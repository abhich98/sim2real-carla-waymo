from __future__ import annotations

import argparse

from src.dataset import WaymoLoader
from src.pipeline import run_milestones_1_to_3
from src.visualization import show_camera_image, show_point_cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Waymo v2 Milestones 1-3 pipeline.")
    parser.add_argument("--dataset-dir", default="gs://waymo_open_dataset_v_2_0_1/training")
    parser.add_argument(
        "--context-name",
        default="10876852935525353526_1640_000_1660_000",
        help="Waymo context sequence name.",
    )
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--frame-index", type=int, default=35)
    parser.add_argument("--camera-name", default=None)

    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--sor-nb-neighbors", type=int, default=20)
    parser.add_argument("--sor-std-ratio", type=float, default=2.0)
    parser.add_argument("--ror-radius", type=float, default=0.5)
    parser.add_argument("--ror-min-points", type=int, default=16)

    parser.add_argument("--no-view", action="store_true")
    parser.add_argument(
        "--warm-cache-only",
        action="store_true",
        help="Download component parquet files to local cache and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader = WaymoLoader(
        dataset_dir=args.dataset_dir,
        context_name=args.context_name,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )

    if args.warm_cache_only:
        cache_map = loader.warm_cache(
            components=[
                "camera_image",
                "camera_calibration",
                "lidar",
                "lidar_calibration",
            ]
        )
        for component, paths in cache_map.items():
            print(f"{component}: cached {len(paths)} file(s)")
        return

    result = run_milestones_1_to_3(
        loader=loader,
        frame_index=args.frame_index,
        camera_name=args.camera_name,
        voxel_size=args.voxel_size,
        sor_nb_neighbors=args.sor_nb_neighbors,
        sor_std_ratio=args.sor_std_ratio,
        ror_radius=args.ror_radius,
        ror_min_points=args.ror_min_points,
    )

    print(f"Frame {result.frame_index}")
    print(f"LiDAR raw points: {result.lidar_raw.shape[0]:,}")
    print(f"LiDAR voxel points: {result.lidar_voxel.shape[0]:,}")
    print(f"LiDAR SOR points: {result.lidar_sor.shape[0]:,}")
    print(f"LiDAR ROR points: {result.lidar_ror.shape[0]:,}")
    print(f"Camera image shape: {result.front_camera.shape}")

    if not args.no_view:
        show_point_cloud(result.lidar_sor, window_name="LiDAR (Voxel + SOR)")
        show_camera_image(result.front_camera, title="Front Camera")


if __name__ == "__main__":
    main()
