from __future__ import annotations

from functools import wraps
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import logging

import dask.dataframe as dd
import numpy as np
import tensorflow as tf

try:
    from waymo_open_dataset import v2
except Exception as import_error:  # pragma: no cover - runtime environment dependent
    v2 = None
    _WAYMO_IMPORT_ERROR = import_error
else:
    _WAYMO_IMPORT_ERROR = None


logger = logging.getLogger(__name__)


CAMERA_NAMES_TO_INDEX = {
    "front": 1,
    "front_left": 2,
    "front_right": 3,
    "side_left": 4,
    "side_right": 5,
}
LIDAR_LASER_NAMES_TO_INDEX = {
    "top": 1,
    "front": 2,
    "side_left": 3,
    "side_right": 4,
    "rear": 5,
}


class WaymoLoader:

    def __init__(
        self,
        dataset_dir: str,
        context_name: str,
        cache_dir: str = "data/raw",
        use_cache: bool = True,
    ) -> None:
        self.dataset_dir = dataset_dir.rstrip("/")
        self.context_name = context_name
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

        # Extract frame timestamps from the lidar component to determine the number of frames
        lidar_df = self.read_component("lidar")
        lidar_timestamps = lidar_df["key.frame_timestamp_micros"].compute().unique()
        self.frame_timestamps_micros = sorted(lidar_timestamps)
        self.num_frames = len(self.frame_timestamps_micros)

    def _require_waymo(self) -> None:
        if v2 is None:
            raise ImportError(
                "waymo_open_dataset.v2 is required but not available in the current environment."
            ) from _WAYMO_IMPORT_ERROR

    def _load_component_class(self, tag: str) -> type | None:
        self._require_waymo()
        for component_class, component_tag in v2.TAG_BY_COMPONENT.items():
            if component_tag == tag:
                return component_class

        logger.warning(f"No valid component class found for tag '{tag}'.")
        return None

    @staticmethod
    def validate_frame_index(func):
        @wraps(func)
        def wrapper(self, frame_index, *args, **kwargs):
            if not isinstance(frame_index, int) or not (0 <= frame_index < self.num_frames):
                raise IndexError(
                    f"Frame index {frame_index} is out of bounds (0 to {self.num_frames - 1})."
                )
            return func(self, frame_index, *args, **kwargs)
        return wrapper

    def _component_uri(self, tag: str) -> str:
        return f"{self.dataset_dir}/{tag}/{self.context_name}.parquet"

    def _local_component_dir(self, tag: str) -> Path:
        return self.cache_dir / tag

    def _local_component_paths(self, tag: str) -> list[str]:
        directory = self._local_component_dir(tag)
        if not directory.exists():
            return []
        return sorted(str(path) for path in directory.glob(f"{self.context_name}*.parquet"))

    def _download_component_to_cache(self, tag: str) -> list[str]:
        source_glob = self._component_uri(tag)
        source_paths = tf.io.gfile.glob(source_glob)
        if not source_paths:
            raise FileNotFoundError(f"No source files found for component '{tag}' at {source_glob}")

        destination_dir = self._local_component_dir(tag)
        destination_dir.mkdir(parents=True, exist_ok=True)

        downloaded_paths: list[str] = []
        for source_path in source_paths:
            destination_path = destination_dir / Path(source_path).name

            if not destination_path.exists():
                logger.info(f"Downloading {source_path} to {destination_path}")
                tf.io.gfile.copy(source_path, str(destination_path), overwrite=False)

            downloaded_paths.append(str(destination_path))
        return downloaded_paths

    def _resolve_component_paths(self, tag: str, use_cache: bool | None = None) -> list[str]:
        resolved_use_cache = self.use_cache if use_cache is None else use_cache

        if resolved_use_cache:
            local_paths = self._local_component_paths(tag)
            if local_paths:
                return local_paths
            return self._download_component_to_cache(tag)

        remote_glob = self._component_uri(tag)
        remote_paths = tf.io.gfile.glob(remote_glob)
        if not remote_paths:
            raise FileNotFoundError(f"No component paths found for '{tag}' at {remote_glob}")
        return remote_paths

    def read_component(self, tag: str, use_cache: bool | None = None) -> dd.DataFrame:
        paths = self._resolve_component_paths(tag, use_cache=use_cache)
        return dd.read_parquet(paths)

    def warm_cache(self, components: list[str]) -> dict[str, list[str]]:
        return {component: self._resolve_component_paths(component, use_cache=True) for component in components}

    def load_sequence(self, components: list[str] | None = None) -> dict[str, dd.DataFrame]:
        component_list = components or [
            "camera_image",
            "camera_calibration",
            "lidar",
            "lidar_calibration",
        ]
        return {component: self.read_component(component) for component in component_list}

    def _to_component_dict(self, component: Any) -> dict[str, Any]:
        if isinstance(component, dict):
            return component

        if is_dataclass(component) and not isinstance(component, type):
            return asdict(component)

        to_dict = getattr(component, "to_dict", None)
        if callable(to_dict) and not isinstance(component, type):
            maybe_dict = to_dict()
            if isinstance(maybe_dict, dict):
                return maybe_dict

        if hasattr(component, "__dict__"):
            return dict(vars(component))

        return {"value": component}

    def _build_component(self, tag: str, row_dict: dict[str, Any]) -> Any:
        component_class = self._load_component_class(tag)
        if component_class is not None and hasattr(component_class, "from_dict"):
            return component_class.from_dict(row_dict)
        else:
            logger.warning(f"Cannot build component for tag '{tag}': No valid component class found.")
            return None

    def _get_matching_row_dict(self, tag: str, filters: dict[str, Any]) -> dict[str, Any]:
        component_df = self.read_component(tag)
        for key, value in filters.items():
            if key in component_df.columns:
                component_df = component_df[component_df[key] == value]

        rows = component_df.head(1, compute=True, npartitions=-1)
        if rows.empty:
            raise LookupError(f"No row found for component '{tag}' with filters {filters}.")
        return rows.iloc[0].to_dict()

    def _build_component_from_matching_row(self, tag: str, filters: dict[str, Any]) -> Any:
        filters["key.segment_context_name"] = self.context_name

        row_dict = self._get_matching_row_dict(tag, filters)
        return self._build_component(tag, row_dict)

    @DeprecationWarning
    def _first_numeric_array(self, payload: Any) -> np.ndarray | None:
        if isinstance(payload, np.ndarray) and np.issubdtype(payload.dtype, np.number):
            return payload

        if isinstance(payload, list) and payload:
            array_candidate = np.asarray(payload)
            if np.issubdtype(array_candidate.dtype, np.number):
                return array_candidate

        if isinstance(payload, dict):
            if {"x", "y", "z"}.issubset(payload):
                x = np.asarray(payload["x"])
                y = np.asarray(payload["y"])
                z = np.asarray(payload["z"])
                return np.column_stack((x, y, z))
            for key in payload:
                nested = self._first_numeric_array(payload[key])
                if nested is not None:
                    return nested

        if isinstance(payload, tuple):
            for entry in payload:
                nested = self._first_numeric_array(entry)
                if nested is not None:
                    return nested

        return None

    def _camera_to_array(self, component_or_row: Any) -> np.ndarray:
        image_bytes = getattr(component_or_row, "image", None)
        if image_bytes is None and isinstance(component_or_row, dict):
            image_bytes = component_or_row.get("image")
        if image_bytes is None:
            raise ValueError("Camera image bytes are not available in the selected row.")
        return tf.image.decode_jpeg(image_bytes).numpy()

    def _lidar_to_array(self, component_or_row: Any) -> np.ndarray:
        self._require_waymo()

        lidar_component = component_or_row
        if isinstance(component_or_row, dict):
            lidar_component = self._build_component("lidar", component_or_row)

        lidar_key = getattr(lidar_component, "key", None)
        if lidar_key is None:
            raise ValueError("LiDAR component is missing its key metadata.")

        calibration = self._build_component_from_matching_row(
            "lidar_calibration",
            {
                "key.laser_name": lidar_key.laser_name,
            },
        )
        vehicle_pose = self._build_component_from_matching_row(
            "vehicle_pose",
            {
                "key.frame_timestamp_micros": lidar_key.frame_timestamp_micros,
            },
        )

        if lidar_key.laser_name == 1:  # Top LiDAR
            lidar_pose = self._build_component_from_matching_row(
                "lidar_pose",
                {
                    "key.frame_timestamp_micros": lidar_key.frame_timestamp_micros,
                    "key.laser_name": lidar_key.laser_name,
                },
            )
            pose_range_image = getattr(lidar_pose, "range_image_return1", None)
        else:
            pose_range_image = None

        point_cloud_returns: list[np.ndarray] = []
        
        for index, range_image in enumerate(lidar_component.range_image_returns):
            if range_image is None:
                continue

            pixel_pose = pose_range_image if index == 0 else None
            # pixel_pose = None # TODO: Remove this later
            print("pixel pose is ", "there" if pixel_pose is not None else "not there")
            points_tensor = v2.convert_range_image_to_point_cloud(
                range_image=range_image,
                calibration=calibration,
                pixel_pose=pixel_pose,
                frame_pose=vehicle_pose if pixel_pose is not None else None,
                keep_polar_features=False,
            )
            points_array = points_tensor.numpy()
            if points_array.size:
                point_cloud_returns.append(points_array[:, :3])

        if not point_cloud_returns:
            return np.empty((0, 3), dtype=np.float32)

        return np.concatenate(point_cloud_returns, axis=0)

    @validate_frame_index
    def get_lidar(
            self, 
            frame_index: int, 
            lidar_name: str | int = "top",
            return_array: bool = False
        ) -> Any:

        frame_timestamp = self.frame_timestamps_micros[frame_index]
        if isinstance(lidar_name, str):
            lidar_name = LIDAR_LASER_NAMES_TO_INDEX[lidar_name.lower()]

        component = self._build_component_from_matching_row(
            "lidar", 
            filters = {
                            "key.segment_context_name": self.context_name,
                            "key.laser_name": lidar_name,
                            "key.frame_timestamp_micros": frame_timestamp,
                        }
            )

        if not return_array:
            return component
        return self._lidar_to_array(component)

    @validate_frame_index
    def get_camera(
        self,
        frame_index: int,
        camera_name: str | int = "front",
        return_array: bool = False,
    ) -> Any:
        frame_timestamp = self.frame_timestamps_micros[frame_index]
        if isinstance(camera_name, str):
            camera_name = CAMERA_NAMES_TO_INDEX[camera_name.lower()]

        component = self._build_component_from_matching_row(
            "camera_image",
            filters = {
                            "key.camera_name": camera_name,
                            "key.frame_timestamp_micros": frame_timestamp,
                        }
            )

        if not return_array:
            return component
        return self._camera_to_array(component)

    @validate_frame_index
    def load_frame(
        self,
        frame_index: int,
        return_array: bool = False,
        lidar_name: str | int = "top",
        camera_name: str | int | None = "front",
    ) -> dict[str, Any]:
        output = {
            "frame_index": frame_index,
            "frame_timestamp_micros": self.frame_timestamps_micros[frame_index],
            "lidar": self.get_lidar(frame_index=frame_index, lidar_name=lidar_name, return_array=return_array)
        }
        if camera_name is not None:
            output["camera"] = self.get_camera(
                frame_index=frame_index,
                camera_name=camera_name,
                return_array=return_array,
            )

        return output
