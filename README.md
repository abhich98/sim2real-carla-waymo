# Waymo LiDAR Transformer

Milestones 1-3 are implemented as reusable Python modules with Waymo v2 loading, local cache support, Open3D/Matplotlib visualization, and LiDAR preprocessing.

## Implemented Milestones

- Milestone 1: Dataset loader (`WaymoLoader`) with:
	- `load_sequence()`
	- `load_frame(frame_index, return_array=...)`
	- `get_lidar(frame_index, return_array=...)`
	- `get_front_camera(frame_index, return_array=...)`
	- `get_calibration(frame_index, return_array=...)`
- Milestone 2: Visualization:
	- Open3D point cloud viewer
	- Matplotlib front camera viewer
- Milestone 3: Preprocessing:
	- Voxel downsampling (`voxel_size=0.10` default)
	- Statistical outlier removal (`nb_neighbors=20`, `std_ratio=2.0` defaults)
	- Radius outlier removal (`radius=0.5`, `min_points=16` defaults)

## Project Structure

```
waymo-lidar-transformer/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── lidar.ipynb
├── outputs/
├── src/
│   ├── dataset/
│   │   └── waymo_loader.py
│   ├── preprocessing/
│   │   ├── voxel.py
│   │   └── filters.py
│   ├── visualization/
│   │   └── viewer.py
│   ├── classical/
│   ├── transformer/
│   ├── utils/
│   └── pipeline.py
└── main.py
```

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Authenticate GCP (for gs:// access):

```bash
gcloud auth application-default login
```

## Run

Warm local cache only:

```bash
uv run python main.py --context-name 10876852935525353526_1640_000_1660_000 --warm-cache-only
```

Run milestones 1-3 end-to-end:

```bash
uv run python main.py \
	--context-name 10876852935525353526_1640_000_1660_000 \
	--frame-index 35
```

Headless/no visualization:

```bash
uv run python main.py --context-name 10876852935525353526_1640_000_1660_000 --frame-index 35 --no-view
```

## Notes

- Loader uses `return_array` for LiDAR, camera, and calibration getters as requested.
- Cache files are stored in `data/raw/<component>/<context>.parquet`.
- Some Waymo v2 LiDAR encodings may require a dataset-specific conversion path; if automatic array conversion is not possible, `get_lidar(..., return_array=False)` returns the raw component for custom conversion.
