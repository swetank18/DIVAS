"""Tests for the CARLA bridge -- all of which run without CARLA installed.

Two halves, matching the module's own layout: the pure conversions, and then
:class:`CarlaWorld` driven against ``tests/fake_carla``. The second half proves
our side of the seam -- handedness, tick arithmetic, the actor registry, and
that everything spawned is destroyed again -- not CARLA's own behaviour.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from divas.sim import carla_bridge as cb
from divas.sim.interface import check
from divas.types import EgoState, VehicleParams

from tests import fake_carla


@pytest.fixture
def carla_world(monkeypatch):
    """A :class:`CarlaWorld` wired to the fake simulator."""
    monkeypatch.setattr(cb, "carla", fake_carla)
    monkeypatch.setattr(cb, "HAVE_CARLA", True)

    def _make(**kw):
        cfg = cb.CarlaConfig(n_vehicles=3, n_walkers=2, seed=7, **kw)
        return cb.CarlaWorld(cfg)

    return _make


# --------------------------------------------------------------------------
# frames and conversions
# --------------------------------------------------------------------------


def test_carla_frame_is_left_handed_and_round_trips():
    # A point 10 m ahead and 4 m to CARLA's y+ is 4 m to *our* right, i.e. -4.
    x, y, th = cb.carla_to_odom(10.0, 4.0, 30.0)
    assert (x, y) == (10.0, -4.0)
    assert th == pytest.approx(-math.radians(30.0))
    assert cb.odom_to_carla(x, y, th) == pytest.approx((10.0, 4.0, 30.0))


def test_steering_sign_flips_between_the_two_conventions():
    """A left turn for us must be a negative steer for CARLA.

    This is the mirrored-world bug: get it backwards and the stack still runs,
    it just steers into everything it was avoiding.
    """
    params = VehicleParams()
    _thr, left, _brk = cb.control_from_command(0.0, +0.35, params, 70.0)
    _thr, right, _brk = cb.control_from_command(0.0, -0.35, params, 70.0)
    assert left < 0.0 < right
    assert cb.steer_from_control(left, 70.0) == pytest.approx(0.35)


def test_throttle_and_brake_split_and_saturate():
    params = VehicleParams()
    thr, _s, brk = cb.control_from_command(params.max_accel * 2, 0.0, params, 70.0)
    assert (thr, brk) == (1.0, 0.0)
    thr, _s, brk = cb.control_from_command(params.min_accel * 2, 0.0, params, 70.0)
    assert (thr, brk) == (0.0, 1.0)


def test_steering_saturates_at_full_lock():
    params = VehicleParams()
    _t, steer, _b = cb.control_from_command(0.0, -10.0, params, 70.0)
    assert steer == 1.0


def test_blueprint_classes_are_specific_before_generic():
    assert cb.classify_blueprint("vehicle.harley-davidson.low_rider") == "motorcycle"
    assert cb.classify_blueprint("vehicle.carlamotors.firetruck") == "truck"
    assert cb.classify_blueprint("vehicle.mitsubishi.fusorosa") == "bus"
    assert cb.classify_blueprint("vehicle.nissan.micra") == "car"
    assert cb.classify_blueprint("walker.pedestrian.0031") == "pedestrian"
    assert cb.classify_blueprint("static.prop.streetbarrier") == "unknown"


def test_params_come_from_the_spawned_vehicle_not_the_defaults():
    base = VehicleParams()
    p = cb.params_from_physics(base, length=4.6, width=1.9, wheelbase=2.7,
                               max_steer_deg=45.0)
    assert (p.length, p.width, p.wheelbase) == (4.6, 1.9, 2.7)
    assert p.rear_overhang == pytest.approx((4.6 - 2.7) / 2)
    assert p.max_steer == pytest.approx(min(base.max_steer, math.radians(45.0)))


# --------------------------------------------------------------------------
# rasterisation
# --------------------------------------------------------------------------


def test_raster_window_marks_off_road_as_occupied():
    pts = np.stack([np.arange(0.0, 40.0, 0.5), np.zeros(80)], axis=1)
    raster = cb.DrivableRaster.from_points(pts, np.full(80, 6.0), 0.25)
    grid = raster.window(20.0, 0.0, 12.0, 0.25)
    assert grid.occupancy_at(20.0, 0.0)[0] == 0.0        # on the road
    assert grid.occupancy_at(20.0, 9.0)[0] == 1.0        # beyond the shoulder
    assert grid.occupancy_at(20.0, 30.0)[0] == 1.0       # off the raster entirely


def test_raster_contains_agrees_with_the_window():
    pts = np.stack([np.arange(0.0, 40.0, 0.5), np.zeros(80)], axis=1)
    raster = cb.DrivableRaster.from_points(pts, np.full(80, 6.0), 0.25)
    on = raster.contains(np.array([20.0, 20.0]), np.array([0.0, 9.0]))
    assert list(on) == [True, False]


def test_stamp_actors_paints_an_oriented_box():
    pts = np.stack([np.arange(0.0, 40.0, 0.5), np.zeros(80)], axis=1)
    raster = cb.DrivableRaster.from_points(pts, np.full(80, 8.0), 0.25)
    grid = raster.window(20.0, 0.0, 12.0, 0.25)
    stamped = cb.stamp_actors(grid, [(20.0, 0.0, 0.0, 2.0, 0.8)])
    assert grid.occupancy_at(21.5, 0.0)[0] == 0.0        # original untouched
    assert stamped.occupancy_at(21.5, 0.0)[0] == 1.0     # inside, along heading
    assert stamped.occupancy_at(20.0, 1.5)[0] == 0.0     # outside, across it


def test_route_progress_and_lookahead():
    route = cb.Route(np.stack([np.arange(0.0, 50.0, 1.0), np.zeros(50)], axis=1))
    assert route.progress(10.0, 0.4) == pytest.approx(10.0)
    assert route.point_ahead(10.0, 0.0, 8.0)[0] == pytest.approx(18.0)
    # Past the end it clamps rather than extrapolating into empty space.
    assert route.point_ahead(48.0, 0.0, 40.0)[0] == pytest.approx(49.0)


def test_route_from_waypoint_follows_the_lane_and_flips_y():
    rng = np.random.default_rng(0)
    route = cb.route_from_waypoint(fake_carla.Waypoint(0.0, -3.5), 40.0, rng, 2.0)
    assert route.length == pytest.approx(40.0, abs=2.0)
    assert np.allclose(route.points[:, 1], 3.5)          # CARLA -3.5 -> odom +3.5


# --------------------------------------------------------------------------
# sensor decoding
# --------------------------------------------------------------------------


def test_semantic_mask_reads_the_red_channel():
    # BGRA, so the tag byte is index 2. Two road pixels, two sidewalk.
    raw = bytes([0, 0, 1, 255, 0, 0, 1, 255, 0, 0, 2, 255, 0, 0, 2, 255])
    mask = cb.semantic_drivable_mask(raw, 2, 2)
    assert mask.tolist() == [[True, True], [False, False]]
    with_shoulder = cb.semantic_drivable_mask(
        bytes([0, 0, 25, 255] * 4), 2, 2, cb.DRIVABLE_TAGS_WITH_SHOULDER
    )
    assert with_shoulder.all()
    assert not cb.semantic_drivable_mask(bytes([0, 0, 25, 255] * 4), 2, 2).any()


def test_lidar_and_radar_decode_to_the_documented_shapes():
    lidar = cb.decode_lidar(np.arange(8, dtype=np.float32).tobytes())
    assert lidar.shape == (2, 4)
    radar = cb.decode_radar(np.arange(4, dtype=np.float32).tobytes())
    assert radar.shape == (1, 4)


# --------------------------------------------------------------------------
# CarlaWorld, against the fake simulator
# --------------------------------------------------------------------------


def test_world_satisfies_the_simworld_protocol(carla_world):
    with carla_world() as world:
        assert check(world) == []
        assert world.self_test() == []


def test_world_enters_synchronous_mode_with_a_fixed_delta(carla_world):
    with carla_world() as world:
        settings = world.world.settings
        assert settings.synchronous_mode is True
        assert settings.fixed_delta_seconds == world.cfg.fixed_delta_seconds
        assert world.tm.synchronous is True
        assert world.tm.seed == world.cfg.seed          # or the run is not repeatable


def test_ego_params_are_read_off_the_spawned_blueprint(carla_world):
    with carla_world() as world:
        assert world.params.length == pytest.approx(3.9)
        assert world.params.width == pytest.approx(1.7)
        assert world.params.wheelbase == pytest.approx(2.5, abs=0.05)
        assert world._max_steer_deg == pytest.approx(70.0)


def test_step_rejects_a_dt_that_is_not_a_whole_number_of_ticks(carla_world):
    with carla_world() as world:
        with pytest.raises(ValueError, match="whole number"):
            world.step(0.037, 0.0, 0.0)


def test_step_ticks_the_server_and_advances_our_clock(carla_world):
    with carla_world(fixed_delta_seconds=0.05) as world:
        before = world.world.ticks
        world.step(0.10, 1.0, 0.0)                      # two ticks
        assert world.world.ticks - before == 2
        assert world.t == pytest.approx(0.10)
        assert world.ego.t == pytest.approx(world.t)
        assert world.ego.v > 0.0                        # throttle did something


def test_a_left_command_turns_the_ego_left_in_our_frame(carla_world):
    """The end-to-end sign check: command, conversion, physics, read-back.

    The fake integrates in CARLA's frame, so this fails if either the control
    conversion or the pose conversion is mirrored -- and passes if *both* are,
    which is why the two are also tested separately above.
    """
    with carla_world() as world:
        for _ in range(20):
            world.step(0.05, 1.5, 0.0)                  # get some speed up
        theta0 = world.ego.theta
        for _ in range(20):
            world.step(0.05, 0.5, 0.30)                 # steer left, CCW positive
        assert world.ego.theta > theta0
        assert world.ego.y > 0.0                        # and moved to our left
        assert world.ego.delta > 0.0                    # reported in our convention


def test_grids_are_ego_centred_and_actors_appear_only_in_the_full_one(carla_world):
    with carla_world() as world:
        static, full = world.ground_truth_grids(half_extent=20.0, resolution=0.25)
        xmin, ymin, xmax, ymax = static.bounds
        assert xmin == pytest.approx(world.ego.x - 20.0)
        assert ymax == pytest.approx(world.ego.y + 20.0)
        # The predictor must not see actors in the grid or it double-counts its
        # own social repulsion, so the static layer has to be strictly emptier.
        assert full.data.sum() >= static.data.sum()
        assert static.occupancy_at(world.ego.x, world.ego.y + 12.0)[0] == 1.0


def test_tracks_carry_class_extent_and_range_dependent_noise(carla_world):
    with carla_world() as world:
        tracks = world.ground_truth_tracks()
        assert tracks, "the fake town should have traffic in range"
        for tr in tracks:
            assert tr.cls in ("car", "motorcycle", "truck", "pedestrian", "unknown")
            assert tr.position_uncertainty > 0.0
        far = max(tracks, key=lambda t: math.hypot(t.x - world.ego.x, t.y - world.ego.y))
        near = min(tracks, key=lambda t: math.hypot(t.x - world.ego.x, t.y - world.ego.y))
        if far is not near:
            assert far.position_uncertainty >= near.position_uncertainty


def test_actors_beyond_track_range_are_not_reported(carla_world):
    with carla_world(track_range=1.0) as world:
        assert world.ground_truth_tracks() == []
        assert world.actors == []


def test_collision_sensor_is_believed_and_labelled(carla_world):
    with carla_world() as world:
        assert world.collision() is None
        other = fake_carla.Actor(world.world, 999, "walker.pedestrian.0001",
                                 fake_carla.Transform())
        world._on_collision(type("E", (), {"other_actor": other})())
        assert world.collision() == "actor:pedestrian#999"
        world.clear_collisions()
        assert world.collision() is None


def test_leaving_the_drivable_raster_reads_as_off_road(carla_world):
    with carla_world() as world:
        world.ego.y += 40.0
        assert world.collision() == "off_road"


def test_sensors_attach_and_deliver_frames(carla_world):
    with carla_world(sensors=("semantic",)) as world:
        world.step(0.05, 0.0, 0.0)
        frame = world.frames["semantic"]
        mask = cb.semantic_drivable_mask(frame.raw_data, frame.height, frame.width)
        assert mask.all()                               # the fake paints all road


def test_close_destroys_everything_and_restores_the_server(carla_world):
    """The trap this whole class is shaped around.

    Leaked actors do not crash the next run -- they quietly make it worse, by
    filling the town before it starts.
    """
    world = carla_world(sensors=("rgb",))
    spawned = [world.vehicle, world._collision_sensor]
    spawned += [info.handle for info in world._spawned.values()]
    spawned += list(world._sensors.values())
    assert len(spawned) > 4
    world.close()

    assert all(a.destroyed for a in spawned), "an actor was left on the map"
    assert world.world.settings.synchronous_mode is False
    assert world.tm.synchronous is False
    world.close()                                       # idempotent


def test_context_manager_closes_even_when_the_body_raises(carla_world):
    world = carla_world()
    with pytest.raises(RuntimeError):
        with world:
            raise RuntimeError("boom")
    assert world._closed


def test_the_runner_closes_the_world_it_built(carla_world):
    """The other half of the leak fix: ``run`` owns the world it creates."""
    from divas.eval.runner import ABLATION, RunnerConfig, run
    from divas.eval.scenarios import Scenario

    world = carla_world()
    scenario = Scenario(
        name="carla_smoke",
        description="fake CARLA, straight lane",
        build=lambda seed: world,
        goal_progress=1e6,                              # never reached
        time_limit=1.0,
    )
    m = run(scenario, ABLATION[2], seed=0,
            cfg=RunnerConfig(sim_dt=world.cfg.fixed_delta_seconds))
    assert world._closed, "the runner leaked a CARLA session"
    assert m.plan_calls > 0
    assert m.sim_time == pytest.approx(1.0, abs=0.1)


def test_the_runner_keeps_the_vehicle_the_bridge_measured(carla_world):
    """``run`` must not overwrite CARLA-derived params with the defaults."""
    from divas.eval.runner import ABLATION, RunnerConfig, run
    from divas.eval.scenarios import Scenario

    world = carla_world()
    world.params = cb.params_from_physics(world.params, 4.8, 2.0, 2.9, 60.0)
    scenario = Scenario(name="carla_params", description="", build=lambda s: world,
                        goal_progress=1e6, time_limit=0.3)
    run(scenario, ABLATION[2], seed=0, cfg=RunnerConfig(sim_dt=0.05))
    assert world.params.length == pytest.approx(4.8)


def test_requesting_a_sensor_turns_rendering_back_on(carla_world):
    """No-rendering mode is a big speed-up that silently blanks every camera."""
    with carla_world() as fast:
        assert fast.world.settings.no_rendering_mode is True
    with carla_world(sensors=("rgb",)) as seeing:
        assert seeing.world.settings.no_rendering_mode is False


# --------------------------------------------------------------------------
# guarded by the real client -- these skip on a machine without it
# --------------------------------------------------------------------------

needs_client = pytest.mark.skipif(
    not cb.HAVE_CARLA, reason="needs the carla client wheel installed"
)


@needs_client
def test_drivable_tags_match_carlas_own_enum():
    """Pin ``DRIVABLE_TAGS`` to ``CityObjectLabel`` rather than to memory.

    The tag numbering changed once already (0.9.14). A silent shift turns the
    Phase 2 drivable-area ground truth into a mask of some other class, and
    nothing downstream would notice -- it would just train badly.
    """
    import carla

    labels = {n: int(v) for n, v in carla.CityObjectLabel.names.items()}
    assert set(cb.DRIVABLE_TAGS) == {labels["Roads"], labels["RoadLines"]}
    assert set(cb.DRIVABLE_TAGS_WITH_SHOULDER) == {
        labels["Roads"], labels["RoadLines"], labels["Ground"]
    }


@needs_client
def test_every_carla_name_the_bridge_uses_still_exists():
    """API drift check, run without a server.

    The bridge is the one module that can be broken by upgrading something
    other than this repository. Finding out at import time costs a second;
    finding out two hours into a batch costs the evening.
    """
    import carla

    surface = {
        carla: ["Client", "WeatherParameters", "Transform", "Location",
                "Rotation", "VehicleControl", "command"],
        carla.command: ["DestroyActor"],
        carla.Client: ["set_timeout", "get_world", "load_world",
                       "get_trafficmanager"],
        carla.World: ["get_settings", "apply_settings", "tick", "get_map",
                      "get_blueprint_library", "spawn_actor", "try_spawn_actor",
                      "set_weather", "get_snapshot",
                      "get_random_location_from_navigation",
                      "set_pedestrians_seed", "set_pedestrians_cross_factor"],
        carla.Map: ["get_spawn_points", "generate_waypoints", "get_waypoint"],
        carla.BlueprintLibrary: ["find", "filter"],
        carla.ActorBlueprint: ["has_attribute", "set_attribute", "id"],
        carla.Actor: ["get_transform", "get_velocity", "destroy", "type_id",
                      "id", "bounding_box"],
        carla.Vehicle: ["apply_control", "set_autopilot", "get_physics_control"],
        carla.Sensor: ["listen", "stop"],
        carla.WalkerAIController: ["start", "stop", "go_to_location",
                                   "set_max_speed"],
        carla.WorldSnapshot: ["find"],
        carla.Waypoint: ["next", "transform", "lane_width"],
        carla.WheelPhysicsControl: ["position", "max_steer_angle"],
        carla.WorldSettings: ["synchronous_mode", "fixed_delta_seconds",
                              "no_rendering_mode"],
        carla.TrafficManager: ["set_synchronous_mode", "set_random_device_seed"],
        carla.BoundingBox: ["extent"],
        carla.CollisionEvent: ["other_actor"],
    }
    missing = [f"{getattr(o, '__name__', o)}.{n}"
               for o, names in surface.items() for n in names if not hasattr(o, n)]
    assert missing == []


@needs_client
def test_every_weather_preset_is_constructible():
    """A typo'd preset key raises only when that weather is first selected --
    which is halfway through an --all-weather sweep, an hour in."""
    import carla

    for name, preset in cb.WEATHER_PRESETS.items():
        carla.WeatherParameters(**preset)          # must not raise
