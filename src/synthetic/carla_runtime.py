"""CARLA capture API; this module intentionally does not import CARLA at import time."""

from __future__ import annotations

from collections.abc import Iterator
import math
import queue
import random
import importlib
import logging
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import OcclusionConfig, SyntheticConfig
from .types import CapturedFrame, PixelBox


# ---------------------------------------------------------------------------
# Pure helpers (no CARLA dependency, unit-testable)
# ---------------------------------------------------------------------------

_LOG = logging.getLogger(__name__)

_DEPTH_FAR_METERS = 1000.0
_DEPTH_SCALE = 256.0**3 - 1.0


def bgra_to_rgb(raw: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert a CARLA BGRA image array (H, W, 4) to a contiguous RGB array."""
    return raw[:, :, :3][:, :, ::-1].copy()


def decode_depth(raw: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Decode a CARLA depth-camera BGRA array (H, W, 4) into metres."""
    bgr = raw[:, :, :3].astype(np.float64)
    normalized = (bgr[:, :, 2] + bgr[:, :, 1] * 256.0 + bgr[:, :, 0] * 65536.0) / _DEPTH_SCALE
    return (normalized * _DEPTH_FAR_METERS).astype(np.float32)


def decode_semantic_tags(raw: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return the per-pixel semantic tag (red channel) of a raw CARLA segmentation image."""
    return raw[:, :, 2].copy()


def project_to_image(
    world_to_camera: NDArray[np.float64],
    points: NDArray[np.float64],
    width: int,
    height: int,
    field_of_view_degrees: float,
    near_plane_meters: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Project world points (N, 3) into pixels using CARLA's camera convention.

    CARLA cameras look along +x with +y right and +z up. Returns ``(pixels (N, 2),
    depths (N,))`` or ``None`` when any point lies at or behind the near plane.
    """
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    camera_points = homogeneous @ world_to_camera.T
    depths = camera_points[:, 0]
    if np.any(depths <= near_plane_meters):
        return None
    focal_length = width / (2.0 * math.tan(math.radians(field_of_view_degrees) / 2.0))
    pixel_x = focal_length * camera_points[:, 1] / depths + width / 2.0
    pixel_y = -focal_length * camera_points[:, 2] / depths + height / 2.0
    return np.stack([pixel_x, pixel_y], axis=1), depths


def visible_pedestrian_box(
    projected_box: PixelBox,
    min_depth_meters: float,
    max_depth_meters: float,
    semantic_tags: NDArray[np.uint8],
    depth_meters: NDArray[np.float32],
    occlusion: OcclusionConfig,
    min_in_image_fraction: float,
    tight: bool,
) -> PixelBox | None:
    """Decide whether a pedestrian is visible and return its label box.

    Visible pixels are those inside the in-image part of the projected box whose
    semantic tag is "pedestrian" and whose depth lies within the pedestrian's own
    3D-box depth range (plus tolerance). This rejects pedestrians hidden behind
    vehicles, buildings, or vegetation, which a pure 3D projection cannot detect.
    """
    height, width = semantic_tags.shape
    box_width = projected_box.x_max - projected_box.x_min
    box_height = projected_box.y_max - projected_box.y_min
    if box_width <= 0 or box_height <= 0:
        return None

    clipped_width = min(projected_box.x_max, width) - max(projected_box.x_min, 0.0)
    clipped_height = min(projected_box.y_max, height) - max(projected_box.y_min, 0.0)
    if clipped_width <= 0 or clipped_height <= 0:
        return None
    if (clipped_width * clipped_height) / (box_width * box_height) < min_in_image_fraction:
        return None

    x0 = max(0, int(math.floor(projected_box.x_min)))
    y0 = max(0, int(math.floor(projected_box.y_min)))
    x1 = min(width, int(math.ceil(projected_box.x_max)))
    y1 = min(height, int(math.ceil(projected_box.y_max)))
    if x1 <= x0 or y1 <= y0:
        return None

    region_depth = depth_meters[y0:y1, x0:x1]
    visible = (
        (semantic_tags[y0:y1, x0:x1] == occlusion.pedestrian_semantic_tag)
        & (region_depth >= min_depth_meters - occlusion.depth_tolerance_meters)
        & (region_depth <= max_depth_meters + occlusion.depth_tolerance_meters)
    )
    visible_count = int(np.count_nonzero(visible))
    if visible_count < occlusion.min_visible_pixels:
        return None
    if visible_count / float((y1 - y0) * (x1 - x0)) < occlusion.min_visible_pixel_fraction:
        return None

    if not tight:
        return projected_box
    rows, columns = np.nonzero(visible)
    return PixelBox(
        x_min=float(x0 + columns.min()),
        y_min=float(y0 + rows.min()),
        x_max=float(x0 + columns.max() + 1),
        y_max=float(y0 + rows.max() + 1),
    )


class _SensorStream:
    """Queue for one sensor that hands out the measurement for an exact frame."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._queue: queue.Queue[Any] = queue.Queue()
        self._pending: Any = None

    def put(self, data: Any) -> None:
        self._queue.put(data)

    def clear(self) -> None:
        """Drop everything received so far (e.g. after warm-up ticks)."""
        self._pending = None
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def get(self, frame: int, deadline: float) -> Any:
        """Return the measurement for ``frame``, discarding older ones."""
        while True:
            if self._pending is not None:
                item, self._pending = self._pending, None
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out waiting for CARLA {self.name} frame {frame}")
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty as exc:
                    raise TimeoutError(
                        f"timed out waiting for CARLA {self.name} frame {frame}"
                    ) from exc
            if item.frame == frame:
                return item
            if item.frame > frame:
                self._pending = item
                raise RuntimeError(
                    f"CARLA {self.name} skipped frame {frame} (next is {item.frame})"
                )
            # Older measurement (warm-up tick, skipped tick, or previous scene): drop it.


class _SceneSetupError(RuntimeError):
    """A scene could not be populated at the chosen ego spawn point."""


def _is_passenger_car(blueprint: Any) -> bool:
    """True for ordinary cars; excludes trucks, buses, vans and two-wheelers.

    A roof camera at a fixed mount position would end up inside the cab of a truck or bus
    (whole image one colour) or float in the air on a bicycle.
    """
    if blueprint.has_attribute("base_type"):
        return blueprint.get_attribute("base_type").as_str().lower() == "car"
    if blueprint.has_attribute("number_of_wheels"):
        return blueprint.get_attribute("number_of_wheels").as_int() == 4
    return True


# ---------------------------------------------------------------------------
# CARLA capture
# ---------------------------------------------------------------------------

_SENSORS = (
    ("rgb", "sensor.camera.rgb"),
    ("depth", "sensor.camera.depth"),
    ("semantic", "sensor.camera.semantic_segmentation"),
)


class CarlaCapture:
    """Own CARLA connection, synchronous world settings, actors, and sensors."""

    def __init__(self, config: SyntheticConfig) -> None:
        self.config = config
        self._carla: Any = None
        self._client: Any = None
        self._world: Any = None
        self._traffic_manager: Any = None
        self._original_settings: Any = None
        self._sensors: dict[str, Any] = {}
        self._streams: dict[str, _SensorStream] = {}
        self._walkers: list[Any] = []
        self._controllers: list[Any] = []
        self._scene_actors: list[Any] = []
        self._rng = random.Random(config.generation.seed)
        self._scene_id = ""
        self._active_scene_index: int | None = None

    def __enter__(self) -> CarlaCapture:
        # Delay importing CARLA until generation runs on a system with the CARLA server available.
        try:
            carla = importlib.import_module("carla")  # pyright: ignore[reportMissingImports]
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CARLA Python API is unavailable; install the API matching the CARLA server build"
            ) from exc

        self._carla = carla
        self._client = carla.Client(self.config.carla.host, self.config.carla.port)
        self._client.set_timeout(self.config.carla.timeout_seconds)
        self._world = self._client.get_world()
        if self.config.carla.map_name:
            current_map = self._world.get_map().name.rsplit("/", 1)[-1]
            if current_map != self.config.carla.map_name:
                self._world = self._client.load_world(self.config.carla.map_name)

        self._original_settings = self._world.get_settings()
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.config.generation.fixed_delta_seconds
        self._world.apply_settings(settings)

        randomization = self.config.randomization
        if randomization.ego_autopilot or randomization.background_vehicle_autopilot:
            self._traffic_manager = self._client.get_trafficmanager(
                self.config.carla.traffic_manager_port
            )
            self._traffic_manager.set_synchronous_mode(True)
            self._traffic_manager.set_random_device_seed(self.config.generation.seed)
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        try:
            self._stop_scene()
        finally:
            if self._traffic_manager is not None:
                try:
                    self._traffic_manager.set_synchronous_mode(False)
                except RuntimeError:
                    pass
            if self._world is not None and self._original_settings is not None:
                self._world.apply_settings(self._original_settings)

    def capture_frames(self) -> Iterator[CapturedFrame]:
        if self._world is None:
            raise RuntimeError("CarlaCapture must be used as a context manager")

        generation = self.config.generation
        for frame_id in range(generation.num_frames):
            scene_index = frame_id // generation.frames_per_scene
            if scene_index != self._active_scene_index:
                self._stop_scene()
                self._start_scene(scene_index)

            for _ in range(generation.ticks_per_frame):
                frame = self._world.tick()
            snapshot = self._world.get_snapshot()
            if snapshot.frame != frame:
                raise RuntimeError(
                    f"CARLA snapshot frame {snapshot.frame} does not match tick {frame}"
                )
            measurements = self._measurements_for(frame)
            rgb_image = measurements["rgb"]
            if (
                rgb_image.width != self.config.camera.width
                or rgb_image.height != self.config.camera.height
            ):
                raise RuntimeError("CARLA camera returned an image with unexpected dimensions")

            yield CapturedFrame(
                frame_id=frame_id,
                timestamp_seconds=float(rgb_image.timestamp),
                scene_id=self._scene_id,
                rgb_image=bgra_to_rgb(self._as_array(rgb_image)),
                pedestrian_boxes=self._pedestrian_boxes(
                    snapshot,
                    rgb_image.transform,
                    decode_semantic_tags(self._as_array(measurements["semantic"])),
                    decode_depth(self._as_array(measurements["depth"])),
                ),
            )
        self._stop_scene()

    # -- scene lifecycle ----------------------------------------------------

    def _start_scene(self, scene_index: int) -> None:
        carla = self._carla
        seed = self.config.generation.seed + scene_index
        self._rng = random.Random(seed)
        if hasattr(self._world, "set_pedestrians_seed"):
            self._world.set_pedestrians_seed(seed)

        weather_name = self._rng.choice(self.config.randomization.weather_presets)
        if not hasattr(carla.WeatherParameters, weather_name):
            raise ValueError(f"unknown CARLA weather preset: {weather_name}")
        weather = carla.WeatherParameters()
        preset = getattr(carla.WeatherParameters, weather_name)
        for attribute in dir(preset):
            if not attribute.startswith("_") and hasattr(weather, attribute):
                try:
                    setattr(weather, attribute, getattr(preset, attribute))
                except (AttributeError, TypeError):
                    continue
        hour = self._rng.choice(self.config.randomization.time_of_day_hours)
        weather.sun_altitude_angle = 90.0 * math.sin((hour - 6) * math.pi / 12)
        self._world.set_weather(weather)

        map_name = self._world.get_map().name.rsplit("/", 1)[-1]
        self._scene_id = f"{map_name}_{scene_index:04d}_{weather_name}_{hour:02d}h"
        spawn_points = list(self._world.get_map().get_spawn_points())
        if not spawn_points:
            raise RuntimeError("CARLA map has no vehicle spawn points for the camera rig")
        self._rng.shuffle(spawn_points)

        attempts = min(self.config.generation.max_scene_attempts, len(spawn_points))
        last_error: _SceneSetupError | None = None
        for attempt in range(attempts):
            try:
                self._populate_scene(spawn_points[attempt], spawn_points)
                break
            except _SceneSetupError as exc:
                last_error = exc
                self._destroy_scene_actors()
        else:
            raise RuntimeError(
                f"could not set up scene {self._scene_id} after {attempts} spawn points: {last_error}"
            )

        for _ in range(self.config.generation.warmup_ticks_per_scene):
            self._world.tick()
        for stream in self._streams.values():
            stream.clear()
        self._active_scene_index = scene_index

    def _populate_scene(self, ego_spawn_point: Any, spawn_points: list[Any]) -> None:
        blueprints = self._world.get_blueprint_library()
        vehicle_blueprints = list(blueprints.filter("vehicle.*"))
        ego_blueprints = [bp for bp in vehicle_blueprints if _is_passenger_car(bp)]
        if not ego_blueprints:
            raise RuntimeError("CARLA blueprint library has no passenger-car blueprints")

        ego = self._world.try_spawn_actor(self._rng.choice(ego_blueprints), ego_spawn_point)
        if ego is None:
            raise _SceneSetupError("ego spawn point is occupied")
        self._scene_actors.append(ego)
        self._attach_sensors(ego, blueprints)

        # In synchronous mode a freshly spawned actor reports a zero transform until the
        # next tick, so place everything relative to the spawn point, not ego.get_transform().
        self._spawn_pedestrians(ego_spawn_point, blueprints)
        self._spawn_background_vehicles(ego_spawn_point, vehicle_blueprints, spawn_points)
        if self.config.randomization.ego_autopilot:
            ego.set_autopilot(True, self.config.carla.traffic_manager_port)

    def _attach_sensors(self, ego: Any, blueprints: Any) -> None:
        camera = self.config.camera
        transform = self._carla.Transform(
            self._carla.Location(
                x=camera.mount_x_meters, y=camera.mount_y_meters, z=camera.mount_z_meters
            )
        )
        for name, blueprint_id in _SENSORS:
            blueprint = blueprints.find(blueprint_id)
            blueprint.set_attribute("image_size_x", str(camera.width))
            blueprint.set_attribute("image_size_y", str(camera.height))
            blueprint.set_attribute("fov", str(camera.field_of_view_degrees))
            blueprint.set_attribute("sensor_tick", "0.0")  # one measurement per world tick
            sensor = self._world.spawn_actor(blueprint, transform, attach_to=ego)
            self._scene_actors.append(sensor)
            stream = _SensorStream(name)
            sensor.listen(stream.put)
            self._sensors[name] = sensor
            self._streams[name] = stream

    def _spawn_pedestrians(self, ego_transform: Any, blueprints: Any) -> None:
        pedestrians = self.config.pedestrians
        walker_blueprints = list(blueprints.filter("walker.pedestrian.*"))
        controller_blueprint = blueprints.find("controller.ai.walker")
        if not walker_blueprints:
            if pedestrians.min_per_scene:
                raise RuntimeError("CARLA blueprint library has no pedestrian blueprints")
            return

        count = self._rng.randint(pedestrians.min_per_scene, pedestrians.max_per_scene)
        origin = ego_transform.location
        forward = ego_transform.get_forward_vector()
        attempts = max(count, 1) * pedestrians.spawn_attempts_per_pedestrian

        walkers: list[Any] = []
        for _ in range(attempts):
            if len(walkers) >= count:
                break
            location = self._world.get_random_location_from_navigation()
            if location is None:
                continue
            dx, dy = location.x - origin.x, location.y - origin.y
            distance = math.hypot(dx, dy)
            if distance < 1e-3:
                continue
            if not pedestrians.min_distance_meters <= distance <= pedestrians.max_distance_meters:
                continue
            if (dx * forward.x + dy * forward.y) / distance < pedestrians.front_spawn_cosine:
                continue

            walker_blueprint = self._rng.choice(walker_blueprints)
            if walker_blueprint.has_attribute("is_invincible"):
                walker_blueprint.set_attribute("is_invincible", "false")
            walker = self._world.try_spawn_actor(walker_blueprint, self._carla.Transform(location))
            if walker is not None:
                walkers.append(walker)
                self._scene_actors.append(walker)

        controllers: list[Any] = []
        for walker in walkers:
            controller = self._world.try_spawn_actor(
                controller_blueprint, self._carla.Transform(), walker
            )
            if controller is None:
                continue
            controllers.append((walker, controller))
            self._scene_actors.append(controller)
            self._controllers.append(controller)

        if len(controllers) < pedestrians.min_per_scene:
            raise _SceneSetupError(
                f"spawned {len(controllers)} controllable pedestrians, below configured "
                f"minimum {pedestrians.min_per_scene}"
            )

        # CARLA requires a tick after spawning walker controllers before starting them.
        self._world.tick()
        for walker, controller in controllers:
            controller.start()
            controller.set_max_speed(
                self._rng.uniform(
                    pedestrians.min_walk_speed_meters_per_second,
                    pedestrians.max_walk_speed_meters_per_second,
                )
            )
            destination = self._world.get_random_location_from_navigation()
            if destination is not None:
                controller.go_to_location(destination)
            self._walkers.append(walker)
        _LOG.info("scene %s: %d pedestrians spawned (target %d)", self._scene_id, len(self._walkers), count)

    def _spawn_background_vehicles(
        self, ego_transform: Any, vehicle_blueprints: list[Any], spawn_points: list[Any]
    ) -> None:
        randomization = self.config.randomization
        count = self._rng.randint(randomization.min_vehicles, randomization.max_vehicles)
        ego_location = ego_transform.location
        candidates = [
            point
            for point in spawn_points
            if point.location.distance(ego_location)
            >= randomization.min_vehicle_distance_from_camera_meters
        ]
        self._rng.shuffle(candidates)
        for transform in candidates[:count]:
            vehicle = self._world.try_spawn_actor(self._rng.choice(vehicle_blueprints), transform)
            if vehicle is None:
                continue
            self._scene_actors.append(vehicle)
            if randomization.background_vehicle_autopilot:
                vehicle.set_autopilot(True, self.config.carla.traffic_manager_port)

    def _stop_scene(self) -> None:
        self._destroy_scene_actors()
        self._active_scene_index = None

    def _destroy_scene_actors(self) -> None:
        for sensor in self._sensors.values():
            try:
                sensor.stop()
            except RuntimeError:
                pass
        for controller in self._controllers:
            try:
                controller.stop()
            except RuntimeError:
                pass

        if self._scene_actors:
            # Controllers before their walkers, sensors before their parent vehicle.
            ordered = self._controllers + [
                actor for actor in reversed(self._scene_actors) if actor not in self._controllers
            ]
            try:
                self._client.apply_batch_sync(
                    [self._carla.command.DestroyActor(actor.id) for actor in ordered]
                )
                self._world.tick()
            except RuntimeError:
                pass

        self._sensors.clear()
        self._streams.clear()
        self._walkers.clear()
        self._controllers.clear()
        self._scene_actors.clear()

    # -- per-frame helpers --------------------------------------------------

    def _measurements_for(self, frame: int) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.carla.timeout_seconds
        return {name: stream.get(frame, deadline) for name, stream in self._streams.items()}

    @staticmethod
    def _as_array(image: Any) -> NDArray[np.uint8]:
        return np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))

    def _pedestrian_boxes(
        self,
        snapshot: Any,
        camera_transform: Any,
        semantic_tags: NDArray[np.uint8],
        depth_meters: NDArray[np.float32],
    ) -> tuple[PixelBox, ...]:
        camera = self.config.camera
        world_to_camera = np.asarray(camera_transform.get_inverse_matrix(), dtype=np.float64)
        boxes: list[PixelBox] = []
        for walker in self._walkers:
            walker_snapshot = snapshot.find(walker.id)
            if walker_snapshot is None:
                continue
            vertices = walker.bounding_box.get_world_vertices(walker_snapshot.get_transform())
            points = np.array([[v.x, v.y, v.z] for v in vertices], dtype=np.float64)
            projection = project_to_image(
                world_to_camera,
                points,
                camera.width,
                camera.height,
                camera.field_of_view_degrees,
                camera.projection_near_plane_meters,
            )
            if projection is None:
                continue
            pixels, depths = projection
            projected_box = PixelBox(
                x_min=float(pixels[:, 0].min()),
                y_min=float(pixels[:, 1].min()),
                x_max=float(pixels[:, 0].max()),
                y_max=float(pixels[:, 1].max()),
            )
            box = visible_pedestrian_box(
                projected_box,
                float(depths.min()),
                float(depths.max()),
                semantic_tags,
                depth_meters,
                self.config.occlusion,
                self.config.pedestrians.min_visible_fraction,
                tight=self.config.labels.box_source == "segmentation",
            )
            if box is not None:
                boxes.append(box)
        return tuple(boxes)
