"""CARLA capture API; this module intentionally does not import CARLA at import time."""

from __future__ import annotations

from collections.abc import Iterator
import math
import queue
import random
import importlib
import time
from typing import Any

import numpy as np

from .config import SyntheticConfig
from .types import CapturedFrame, PixelBox


class CarlaCapture:
    """Own CARLA connection, synchronous world settings, actors, and sensors."""

    def __init__(self, config: SyntheticConfig) -> None:
        self.config = config
        self._carla: Any = None
        self._client: Any = None
        self._world: Any = None
        self._original_settings: Any = None
        self._camera: Any = None
        self._image_queue: queue.Queue[Any] = queue.Queue()
        self._scene_actors: list[Any] = []
        self._controllers: list[Any] = []
        self._rng = random.Random(config.generation.seed)
        self._scene_id = ""

    def __enter__(self) -> CarlaCapture:
        # This is setup this way to delay importing CARLA until the generation is run on a system with the CARLA server available.
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
        settings.synchronous_mode = self.config.generation.synchronous_mode
        settings.fixed_delta_seconds = (
            self.config.generation.fixed_delta_seconds
            if self.config.generation.synchronous_mode
            else None
        )
        self._world.apply_settings(settings)
        if hasattr(self._world, "set_pedestrians_seed"):
            self._world.set_pedestrians_seed(self.config.generation.seed)
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._stop_scene()
        if self._world is not None and self._original_settings is not None:
            self._world.apply_settings(self._original_settings)

    def capture_frames(self) -> Iterator[CapturedFrame]:
        if self._world is None:
            raise RuntimeError("CarlaCapture must be used as a context manager")

        for frame_id in range(self.config.generation.num_frames):
            scene_index = frame_id // self.config.generation.frames_per_scene
            if scene_index != getattr(self, "_active_scene_index", None):
                self._stop_scene()
                self._start_scene(scene_index)

            image = self._next_camera_image()
            rgb = np.frombuffer(image.raw_data, dtype=np.uint8)
            rgb = rgb.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1].copy()
            camera_transform = self._camera.get_transform()
            pedestrian_boxes = self._project_pedestrian_boxes(camera_transform)
            yield CapturedFrame(
                frame_id=frame_id,
                timestamp_seconds=float(image.timestamp),
                scene_id=self._scene_id,
                rgb_image=rgb,
                pedestrian_boxes=pedestrian_boxes,
            )
        self._stop_scene()

    def _start_scene(self, scene_index: int) -> None:
        carla = self._carla
        self._rng = random.Random(self.config.generation.seed + scene_index)
        if hasattr(self._world, "set_pedestrians_seed"):
            self._world.set_pedestrians_seed(self.config.generation.seed + scene_index)

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
        self._rng.shuffle(spawn_points)
        if not spawn_points:
            raise RuntimeError("CARLA map has no vehicle spawn points for the camera rig")

        blueprints = self._world.get_blueprint_library()
        vehicle_blueprints = list(blueprints.filter("vehicle.*"))
        if not vehicle_blueprints:
            raise RuntimeError("CARLA blueprint library has no vehicle blueprints")
        ego = self._spawn_first_available(
            [self._rng.choice(vehicle_blueprints)], spawn_points
        )
        if ego is None:
            raise RuntimeError("could not spawn the camera rig vehicle")
        self._scene_actors.append(ego)

        camera_blueprint = blueprints.find("sensor.camera.rgb")
        camera_blueprint.set_attribute("image_size_x", str(self.config.camera.width))
        camera_blueprint.set_attribute("image_size_y", str(self.config.camera.height))
        camera_blueprint.set_attribute(
            "fov", str(self.config.camera.field_of_view_degrees)
        )
        camera_blueprint.set_attribute(
            "sensor_tick", str(self.config.camera.sensor_tick_seconds)
        )
        camera_transform = carla.Transform(
            carla.Location(
                x=self.config.camera.mount_x_meters,
                y=self.config.camera.mount_y_meters,
                z=self.config.camera.mount_z_meters,
            )
        )
        self._camera = self._world.spawn_actor(
            camera_blueprint, camera_transform, attach_to=ego
        )
        self._scene_actors.append(self._camera)
        self._camera.listen(self._image_queue.put)
        self._spawn_pedestrians(ego, blueprints)
        self._spawn_background_vehicles(ego, blueprints, spawn_points)
        self._active_scene_index = scene_index

    def _spawn_first_available(self, blueprints: list[Any], transforms: list[Any]) -> Any:
        for transform in transforms:
            actor = self._world.try_spawn_actor(self._rng.choice(blueprints), transform)
            if actor is not None:
                return actor
        return None

    def _spawn_pedestrians(self, ego: Any, blueprints: Any) -> None:
        walker_blueprints = list(blueprints.filter("walker.pedestrian.*"))
        controller_blueprint = blueprints.find("controller.ai.walker")
        if not walker_blueprints:
            if self.config.pedestrians.min_per_scene:
                raise RuntimeError("CARLA blueprint library has no pedestrian blueprints")
            return

        count = self._rng.randint(
            self.config.pedestrians.min_per_scene,
            self.config.pedestrians.max_per_scene,
        )
        ego_transform = ego.get_transform()
        origin = ego_transform.location
        forward = ego_transform.get_forward_vector()
        spawned = 0
        attempts = max(
            count * self.config.pedestrians.spawn_attempts_per_pedestrian,
            self.config.pedestrians.spawn_attempts_per_pedestrian,
        )
        for _ in range(attempts):
            if spawned >= count:
                break
            location = self._world.get_random_location_from_navigation()
            if location is None:
                continue
            dx, dy = location.x - origin.x, location.y - origin.y
            distance = math.hypot(dx, dy)
            if not self.config.pedestrians.min_distance_meters <= distance <= self.config.pedestrians.max_distance_meters:
                continue
            if (
                (dx * forward.x + dy * forward.y) / distance
                < self.config.pedestrians.front_spawn_cosine
            ):
                continue

            walker_blueprint = self._rng.choice(walker_blueprints)
            if walker_blueprint.has_attribute("is_invincible"):
                walker_blueprint.set_attribute("is_invincible", "false")
            walker = self._world.try_spawn_actor(
                walker_blueprint, self._carla.Transform(location)
            )
            if walker is None:
                continue
            self._scene_actors.append(walker)
            controller = self._world.try_spawn_actor(
                controller_blueprint, self._carla.Transform(), walker
            )
            if controller is None:
                walker.destroy()
                self._scene_actors.remove(walker)
                continue
            self._scene_actors.append(controller)
            self._controllers.append(controller)
            controller.start()
            controller.set_max_speed(
                self._rng.uniform(
                    self.config.pedestrians.min_walk_speed_meters_per_second,
                    self.config.pedestrians.max_walk_speed_meters_per_second,
                )
            )
            destination = self._world.get_random_location_from_navigation()
            if destination is not None:
                controller.go_to_location(destination)
            spawned += 1
        if spawned < self.config.pedestrians.min_per_scene:
            raise RuntimeError(
                f"spawned {spawned} pedestrians, below configured minimum "
                f"{self.config.pedestrians.min_per_scene}"
            )

    def _spawn_background_vehicles(
        self, ego: Any, blueprints: Any, spawn_points: list[Any]
    ) -> None:
        vehicle_blueprints = list(blueprints.filter("vehicle.*"))
        if len(vehicle_blueprints) == 0:
            return
        count = self._rng.randint(
            self.config.randomization.min_vehicles,
            self.config.randomization.max_vehicles,
        )
        ego_location = ego.get_location()
        candidates = [
            point
            for point in spawn_points
            if (
                point.location.distance(ego_location)
                >= self.config.randomization.min_vehicle_distance_from_camera_meters
            )
        ]
        self._rng.shuffle(candidates)
        for transform in candidates[:count]:
            vehicle = self._world.try_spawn_actor(
                self._rng.choice(vehicle_blueprints), transform
            )
            if vehicle is not None:
                self._scene_actors.append(vehicle)

    def _next_camera_image(self) -> Any:
        sensor_interval = self.config.camera.sensor_tick_seconds
        timeout = max(self.config.carla.timeout_seconds, sensor_interval * 3)
        deadline = time.monotonic() + timeout
        while True:
            self._world.tick()
            try:
                image = self._image_queue.get(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
            except queue.Empty as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for a CARLA RGB camera frame") from exc
                continue
            if image.width != self.config.camera.width or image.height != self.config.camera.height:
                raise RuntimeError("CARLA camera returned an image with unexpected dimensions")
            return image

    def _project_pedestrian_boxes(self, camera_transform: Any) -> tuple[PixelBox, ...]:
        inverse = np.asarray(camera_transform.get_inverse_matrix(), dtype=np.float64)
        focal_length = self.config.camera.width / (
            2 * math.tan(math.radians(self.config.camera.field_of_view_degrees) / 2)
        )
        boxes: list[PixelBox] = []
        for actor in self._scene_actors:
            if not actor.type_id.startswith("walker.pedestrian."):
                continue
            world_vertices = actor.bounding_box.get_world_vertices(actor.get_transform())
            projected: list[tuple[float, float]] = []
            for vertex in world_vertices:
                camera_point = inverse @ np.array([vertex.x, vertex.y, vertex.z, 1.0])
                depth = camera_point[0]
                if depth <= self.config.camera.projection_near_plane_meters:
                    continue
                pixel_x = focal_length * camera_point[1] / depth + self.config.camera.width / 2
                pixel_y = -focal_length * camera_point[2] / depth + self.config.camera.height / 2
                projected.append((float(pixel_x), float(pixel_y)))
            if len(projected) != len(world_vertices):
                continue
            boxes.append(
                PixelBox(
                    x_min=min(point[0] for point in projected),
                    y_min=min(point[1] for point in projected),
                    x_max=max(point[0] for point in projected),
                    y_max=max(point[1] for point in projected),
                )
            )
        return tuple(boxes)

    def _stop_scene(self) -> None:
        if self._camera is not None:
            try:
                self._camera.stop()
            except RuntimeError:
                pass
            self._camera = None
        for controller in self._controllers:
            try:
                controller.stop()
            except RuntimeError:
                pass
        self._controllers.clear()
        for actor in reversed(self._scene_actors):
            try:
                actor.destroy()
            except RuntimeError:
                pass
        self._scene_actors.clear()
        self._active_scene_index = None
        while True:
            try:
                self._image_queue.get_nowait()
            except queue.Empty:
                break