"""A stand-in for the ``carla`` module, just large enough to drive the bridge.

Why this exists: the CARLA bridge is the one part of the stack that cannot be
exercised on a machine without a GPU, and untested bridge code is where
projects like this die -- the perception work lands, the bridge has a sign
error in the steering conversion, and two days go into debugging the planner.

What it does and does not prove. It verifies **our** side of the seam: the
handedness conversion, the tick/substep arithmetic, the actor registry, the
grids, the metric plumbing and -- above all -- that every actor we spawn is
destroyed again. It does **not** prove anything about CARLA's own semantics; it
is our reading of the API, so a misunderstanding here reproduces faithfully in
both places. The fake is therefore written to mirror CARLA's conventions
exactly where they matter: left-handed with y to the right, yaw in degrees
increasing clockwise, and ``VehicleControl.steer`` positive to the **right**.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional


class Vector3D:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = float(x), float(y), float(z)


class Location(Vector3D):
    pass


class Rotation:
    def __init__(self, pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> None:
        self.pitch, self.yaw, self.roll = float(pitch), float(yaw), float(roll)


class Transform:
    def __init__(self, location: Optional[Location] = None,
                 rotation: Optional[Rotation] = None) -> None:
        self.location = location or Location()
        self.rotation = rotation or Rotation()


class BoundingBox:
    def __init__(self, x: float, y: float, z: float = 0.75) -> None:
        self.extent = Vector3D(x, y, z)
        self.location = Location()


class VehicleControl:
    def __init__(self, throttle: float = 0.0, steer: float = 0.0,
                 brake: float = 0.0, **_kw) -> None:
        self.throttle, self.steer, self.brake = throttle, steer, brake


class WeatherParameters:
    def __init__(self, **kw) -> None:
        self.__dict__.update(kw)


class Wheel:
    def __init__(self, x: float, y: float, max_steer_angle: float = 0.0) -> None:
        self.position = Vector3D(x, y, 0.0)      # centimetres, as CARLA reports
        self.max_steer_angle = max_steer_angle


class PhysicsControl:
    def __init__(self) -> None:
        # 2.5 m wheelbase, 70 deg of lock -- a compact hatchback.
        self.wheels = [
            Wheel(250.0, -80.0, 70.0), Wheel(250.0, 80.0, 70.0),
            Wheel(0.0, -80.0, 0.0), Wheel(0.0, 80.0, 0.0),
        ]


class WorldSettings:
    def __init__(self) -> None:
        self.synchronous_mode = False
        self.fixed_delta_seconds = None
        self.no_rendering_mode = False


class Blueprint:
    def __init__(self, bid: str, attributes: Optional[Dict[str, str]] = None) -> None:
        self.id = bid
        self._attr = dict(attributes or {"role_name": "autopilot"})

    def has_attribute(self, key: str) -> bool:
        return key in self._attr

    def set_attribute(self, key: str, value: str) -> None:
        self._attr[key] = value


class BlueprintLibrary:
    def __init__(self, blueprints: List[Blueprint]) -> None:
        self._bps = blueprints

    def find(self, bid: str) -> Blueprint:
        for bp in self._bps:
            if bp.id == bid:
                return bp
        raise IndexError(bid)

    def filter(self, pattern: str) -> List[Blueprint]:
        prefix = pattern.rstrip("*")
        return [bp for bp in self._bps if bp.id.startswith(prefix)]


class Waypoint:
    """A point on the fake town's road: two straight lanes along CARLA +x."""

    def __init__(self, x: float, y: float, lane_width: float = 3.5) -> None:
        self.transform = Transform(Location(x, y, 0.0), Rotation(yaw=0.0))
        self.lane_width = lane_width

    def next(self, distance: float) -> List["Waypoint"]:
        nxt = self.transform.location.x + distance
        if nxt > 400.0:
            return []
        return [Waypoint(nxt, self.transform.location.y, self.lane_width)]


class Actor:
    def __init__(self, world: "World", aid: int, type_id: str,
                 transform: Transform, extent=(2.0, 0.9), speed: float = 0.0) -> None:
        self.id = aid
        self.type_id = type_id
        self._world = world
        self._tf = transform
        self.bounding_box = BoundingBox(*extent)
        self.speed = speed
        self.destroyed = False
        self.autopilot = False
        self.control = VehicleControl()
        self._listener = None
        self.started = False

    # -- state ---------------------------------------------------------
    def get_transform(self) -> Transform:
        return self._tf

    def get_velocity(self) -> Vector3D:
        yaw = math.radians(self._tf.rotation.yaw)
        return Vector3D(self.speed * math.cos(yaw), self.speed * math.sin(yaw), 0.0)

    def get_physics_control(self) -> PhysicsControl:
        return PhysicsControl()

    def apply_control(self, control: VehicleControl) -> None:
        self.control = control

    def set_autopilot(self, on: bool, _port: int = 8000) -> None:
        self.autopilot = on

    # -- sensors / walker controllers ----------------------------------
    def listen(self, callback) -> None:
        self._listener = callback

    def stop(self) -> None:
        self._listener = None

    def start(self) -> None:
        self.started = True

    def go_to_location(self, _loc) -> None:
        pass

    def set_max_speed(self, speed: float) -> None:
        self.speed = speed

    def destroy(self) -> bool:
        if self.destroyed:
            raise RuntimeError("actor already destroyed")
        self.destroyed = True
        self._world.actors = [a for a in self._world.actors if a is not self]
        return True


class SensorImage:
    """Enough of ``carla.Image`` for the decoders: a raw BGRA buffer."""

    def __init__(self, height: int, width: int, raw: bytes, frame: int) -> None:
        self.height, self.width, self.raw_data, self.frame = height, width, raw, frame


class ActorSnapshotView:
    def __init__(self, actor: Actor) -> None:
        self._a = actor

    def get_transform(self) -> Transform:
        return self._a.get_transform()

    def get_velocity(self) -> Vector3D:
        return self._a.get_velocity()


class WorldSnapshot:
    def __init__(self, actors: List[Actor]) -> None:
        self._by_id = {a.id: ActorSnapshotView(a) for a in actors}

    def find(self, aid: int):
        return self._by_id.get(aid)


class Map:
    def __init__(self) -> None:
        self.name = "FakeTown"

    def get_spawn_points(self) -> List[Transform]:
        return [Transform(Location(x, y, 0.0), Rotation(yaw=0.0))
                for x in (0.0, 20.0, 40.0, 60.0, 80.0) for y in (0.0, -3.5)]

    def generate_waypoints(self, distance: float) -> List[Waypoint]:
        out = []
        x = -30.0
        while x <= 400.0:
            out.append(Waypoint(x, 0.0))
            out.append(Waypoint(x, -3.5))
            x += distance
        return out

    def get_waypoint(self, location: Location) -> Waypoint:
        return Waypoint(location.x, location.y)


class World:
    """Ticks a kinematic bicycle for the ego and drifts everyone else forward."""

    def __init__(self) -> None:
        self.actors: List[Actor] = []
        self.settings = WorldSettings()
        self.applied_settings: List[WorldSettings] = []
        self.weather = None
        self.ticks = 0
        self.pedestrian_seed = None
        self.cross_factor = None
        self._next_id = 1
        self._map = Map()
        self._blueprints = BlueprintLibrary([
            Blueprint("vehicle.nissan.micra"), Blueprint("vehicle.tesla.model3"),
            Blueprint("vehicle.harley-davidson.low_rider"),
            Blueprint("vehicle.carlamotors.firetruck"),
            Blueprint("walker.pedestrian.0001"), Blueprint("walker.pedestrian.0002"),
            Blueprint("controller.ai.walker", {}),
            Blueprint("sensor.other.collision", {}),
            Blueprint("sensor.camera.semantic_segmentation",
                      {"image_size_x": "800", "image_size_y": "600", "fov": "90"}),
            Blueprint("sensor.camera.rgb",
                      {"image_size_x": "800", "image_size_y": "600", "fov": "90"}),
        ])

    # -- session -------------------------------------------------------
    def get_settings(self) -> WorldSettings:
        return WorldSettings()

    def apply_settings(self, settings: WorldSettings) -> None:
        self.settings = settings
        self.applied_settings.append(settings)

    def set_weather(self, weather) -> None:
        self.weather = weather

    def get_map(self) -> Map:
        return self._map

    def get_blueprint_library(self) -> BlueprintLibrary:
        return self._blueprints

    def get_snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(self.actors)

    def get_random_location_from_navigation(self) -> Location:
        return Location(float(10 + 3 * len(self.actors)), -1.5, 0.0)

    def set_pedestrians_seed(self, seed: int) -> None:
        self.pedestrian_seed = seed

    def set_pedestrians_cross_factor(self, factor: float) -> None:
        self.cross_factor = factor

    # -- spawning ------------------------------------------------------
    def spawn_actor(self, bp: Blueprint, transform: Transform, attach_to=None) -> Actor:
        extent = (0.3, 0.3) if bp.id.startswith("walker") else (1.95, 0.85)
        speed = 0.0 if (bp.id.startswith("sensor") or bp.id.startswith("controller")) else 5.0
        actor = Actor(self, self._next_id, bp.id, transform, extent, speed)
        if attach_to is not None:
            actor.speed = 0.0
            actor.attached_to = attach_to
        self._next_id += 1
        self.actors.append(actor)
        return actor

    def try_spawn_actor(self, bp: Blueprint, transform: Transform, attach_to=None):
        """Refuses an occupied spawn point, as CARLA does.

        Without this the fake happily stacks two cars on one point, the ego
        starts inside a motorcycle, and every run reports a collision on step
        zero -- which is also exactly what happens in real CARLA if the caller
        ignores the ``None`` return.
        """
        if attach_to is None:
            for a in self.actors:
                if a.type_id.startswith(("sensor", "controller")):
                    continue
                if math.hypot(a._tf.location.x - transform.location.x,
                              a._tf.location.y - transform.location.y) < 5.0:
                    return None
        return self.spawn_actor(bp, transform, attach_to)

    # -- physics -------------------------------------------------------
    def tick(self) -> int:
        dt = self.settings.fixed_delta_seconds or 0.05
        self.ticks += 1
        for a in list(self.actors):
            if a.type_id.startswith("sensor"):
                if a._listener is not None:
                    if "camera" in a.type_id:
                        h, w = 4, 4
                        # Tag 1 = Roads, in the red channel of a BGRA buffer.
                        raw = bytes([0, 0, 1, 255] * (h * w))
                        a._listener(SensorImage(h, w, raw, self.ticks))
                continue
            if getattr(a, "attached_to", None) is not None:
                continue
            if a.type_id.startswith("vehicle") and a.control.throttle + a.control.brake > 0.0 \
                    and not a.autopilot:
                # The ego. Kinematic bicycle in CARLA's own frame: positive
                # steer turns right, and yaw grows clockwise.
                c = a.control
                a.speed = max(0.0, a.speed + (2.0 * c.throttle - 4.0 * c.brake) * dt)
                yaw = math.radians(a._tf.rotation.yaw)
                delta = math.radians(70.0) * c.steer
                yaw += a.speed / 2.5 * math.tan(delta) * dt
                a._tf.rotation.yaw = math.degrees(yaw)
            yaw = math.radians(a._tf.rotation.yaw)
            a._tf.location.x += a.speed * math.cos(yaw) * dt
            a._tf.location.y += a.speed * math.sin(yaw) * dt
        return self.ticks


class TrafficManager:
    def __init__(self, port: int) -> None:
        self.port = port
        self.synchronous = None
        self.seed = None

    def set_synchronous_mode(self, on: bool) -> None:
        self.synchronous = on

    def set_random_device_seed(self, seed: int) -> None:
        self.seed = seed


class Client:
    def __init__(self, host: str, port: int) -> None:
        self.host, self.port = host, port
        self.timeout = None
        self.world = World()
        self.tm = TrafficManager(8000)

    def set_timeout(self, seconds: float) -> None:
        self.timeout = seconds

    def get_world(self) -> World:
        return self.world

    def load_world(self, name: str) -> World:
        self.world = World()
        self.world.get_map().name = name
        return self.world

    def get_trafficmanager(self, port: int = 8000) -> TrafficManager:
        self.tm = TrafficManager(port)
        return self.tm
