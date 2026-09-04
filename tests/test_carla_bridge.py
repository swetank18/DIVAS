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

# --------------------------------------------------------------------------
# the longitudinal loop
# --------------------------------------------------------------------------


def _plant(tracker, params, resistance, v_ref, authority=1.0, dt=0.05, steps=800,
           throttle_gain=6.47, brake_gain=3.44):
    """Drive ``tracker`` against the longitudinal plant as measured in CARLA.

    ``tests/fake_carla`` integrates ``v += (2*throttle - 4*brake)*dt`` with no
    losses, so it cannot show what this loop exists to fix -- in a
    frictionless world with pedal gains that happen to equal the planner's
    comfort limits, the open-loop map is correct. Three things here that the
    real plant has and the fake does not:

    * ``resistance(v)``, which in CARLA is engine braking and is large;
    * pedal *gains*, the acceleration a unit of pedal actually delivers,
      which are not ``params.max_accel`` and ``params.min_accel``;
    * ``authority``, a scale on the throttle gain, for asking what happens
      when the pedal model is wrong.

    The default gains and the default :class:`ResistanceModel` are the ones
    ``scripts/calibrate_longitudinal.py`` measured, so this loop reproduces
    the live equilibrium -- which is what
    ``test_the_synthetic_plant_reproduces_the_live_equilibrium`` checks.

    Returns the final speed.
    """
    v, a_meas = 0.0, 0.0
    for _ in range(steps):
        a_cmd = float(np.clip(1.1 * (v_ref - v), params.min_accel, params.max_accel))
        throttle, brake = tracker.update(a_cmd, v, a_meas, dt)
        a_applied = throttle * throttle_gain * authority - brake * brake_gain
        v_next = max(0.0, v + (a_applied - resistance(v)) * dt)
        a_meas, v = (v_next - v) / dt, v_next
    return v


class _OpenLoop:
    """The longitudinal half of :func:`control_from_command`, as it was.

    Kept in the tests rather than in the module: it is the thing being argued
    against, and the only reason it still needs to run is to size the
    difference.
    """

    def __init__(self, params):
        self.params = params

    def update(self, a_cmd, v, a_meas, dt):
        thr, _steer, brk = cb.control_from_command(a_cmd, 0.0, self.params, 70.0)
        return thr, brk


def test_zero_command_at_speed_asks_for_throttle_not_coasting():
    """The one line that distinguishes the two mappings.

    Stage 6 emits ``1.1 * (v_ref - v)``, so the command decays to zero exactly
    when the ego arrives at its reference. Zero throttle there means coasting,
    and in CARLA coasting costs about 4 m/s^2.
    """
    params = VehicleParams()
    throttle, brake = cb.LongitudinalTracker(params).update(0.0, 9.0, 0.0, 0.05)
    assert throttle > 0.0 and brake == 0.0
    assert cb.control_from_command(0.0, 0.0, params, 70.0)[0] == 0.0


def test_the_synthetic_plant_reproduces_the_live_equilibrium():
    """Pin the plant model in these tests against the real measurement.

    ``scripts/calibrate_longitudinal.py`` ran the old open-loop mapping on a
    straight, empty Town10HD_Opt road with a 9.0 m/s reference and measured it
    settling at **7.88 m/s**. Feeding the same measured resistance and pedal
    gains to the loop above must land in the same place, or the tests below
    are arguing about a plant that does not exist.

    The equilibrium is where the proportional command balances engine
    braking: ``1.1 (9 - v) / 2.0 * 6.47 = 2.39 + 0.188 v``, i.e. v = 7.9.
    """
    params = VehicleParams()
    settled = _plant(_OpenLoop(params), params, cb.ResistanceModel(), 9.0)
    assert settled == pytest.approx(7.88, abs=0.25)


def test_the_closed_loop_holds_its_reference_where_the_open_loop_falls_short():
    """The claim, at the size the live run measured it: about 1.1 m/s."""
    params, resistance = VehicleParams(), cb.ResistanceModel()
    ref = params.cruise_speed

    open_loop = _plant(_OpenLoop(params), params, resistance, ref)
    closed = _plant(cb.LongitudinalTracker(params, resistance), params, resistance, ref)

    assert ref - open_loop == pytest.approx(1.12, abs=0.3)
    assert closed == pytest.approx(ref, abs=0.2)


def test_the_trim_absorbs_a_wrong_pedal_model():
    """Why the integral term is there and the feedforward alone is not enough.

    The feedforward is a fit with a 0.77 m/s^2 residual to a gear-dependent,
    hysteretic engine-braking curve, so it is wrong everywhere by a little.
    Take a fifth off the throttle authority it assumes -- a stand-in for a
    gradient, a different blueprint, or a stale fit -- and the loop must still
    arrive at the reference.
    """
    params, resistance = VehicleParams(), cb.ResistanceModel()
    held = _plant(cb.LongitudinalTracker(params, resistance), params, resistance,
                  params.cruise_speed, authority=0.8)
    assert held == pytest.approx(params.cruise_speed, abs=0.25)


def test_an_unreachable_reference_saturates_instead_of_winding_up():
    """The trim must not be asked to fix physics.

    Halve the throttle authority and the reference stops being reachable at
    all: full throttle then delivers 3.24 m/s^2 against 4.08 m/s^2 of engine
    braking at 9 m/s, so the plant tops out at ``3.24 = 2.39 + 0.188 v``, or
    about 4.5 m/s. The right behaviour is to sit there on a floored throttle
    with the trim at its clamp -- not to keep integrating against a limit the
    pedal cannot move.
    """
    params, resistance = VehicleParams(), cb.ResistanceModel()
    tracker = cb.LongitudinalTracker(params, resistance)
    held = _plant(tracker, params, resistance, params.cruise_speed, authority=0.5)

    assert held == pytest.approx(4.5, abs=0.4)              # the plant's ceiling
    assert abs(tracker.trim) <= tracker.i_limit + 1e-9
    assert tracker.update(2.0, held, 0.0, 0.05)[0] == 1.0   # still asking for all of it


def test_a_held_stop_does_not_creep():
    """The feedforward asks for rolling-resistance throttle at every speed
    including zero, so a held stop has to be an explicit branch."""
    throttle, brake = cb.LongitudinalTracker(VehicleParams()).update(0.0, 0.0, 0.0, 0.05)
    assert throttle == 0.0 and brake == 1.0


def test_integral_trim_is_clamped_under_sustained_saturation():
    """Braking saturates constantly in traffic. An unclamped integrator would
    spend the following seconds unwinding a term it accumulated against a
    pedal that was already on the floor, which reads as a controller that has
    stopped responding."""
    params = VehicleParams()
    tracker = cb.LongitudinalTracker(params, i_limit=1.0)
    for _ in range(500):                    # ask for full brake, achieve nothing
        tracker.update(params.min_accel, 8.0, 0.0, 0.05)
    assert abs(tracker.trim) <= 1.0 + 1e-9


def test_resistance_fit_recovers_known_coefficients():
    """The coast-down fit, checked without a server.

    ``scripts/calibrate_longitudinal.py`` is the only place the feedforward
    numbers come from, so its arithmetic is worth a test even though the
    measurement it consumes needs CARLA. Feed it a curve and it must give the
    curve back.
    """
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    import calibrate_longitudinal as cal

    true = cb.ResistanceModel(c0=0.12, c1=0.010, c2=0.0012)
    speeds = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0]
    decels = [true(v) for v in speeds]

    got = cal.fit_resistance(speeds, decels)
    assert (got.c0, got.c1, got.c2) == pytest.approx((0.12, 0.010, 0.0012), abs=1e-6)
    assert cal.fit_residual(got, speeds, decels) < 1e-9


def test_resistance_fit_refuses_too_few_samples_and_clamps_unphysical_drag():
    """A negative quadratic term means the samples were not measuring coasting
    -- a gradient, or a segment that clipped a kerb. Clamped, not published."""
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    import calibrate_longitudinal as cal

    with pytest.raises(ValueError):
        cal.fit_resistance([3.0, 9.0], [0.2, 0.3])

    # a curve that turns over: c2 must not come back negative
    speeds = [3.0, 5.0, 7.0, 9.0, 11.0]
    decels = [0.1, 0.5, 0.8, 0.9, 0.85]
    assert cal.fit_resistance(speeds, decels).c2 >= 0.0


# --------------------------------------------------------------------------
# static map obstacles
# --------------------------------------------------------------------------


def test_carve_clears_an_oriented_box_and_nothing_else():
    """Parked cars have to come out of the drivable mask.

    The mask is painted from the lane graph, which does not know a lane has a
    car standing in it. Town10HD carries 29 parked vehicles as static meshes
    rather than actors, so nothing spawns them and ``ground_truth_tracks``
    cannot report them -- without this they read as free space and a planner
    routes straight through them.
    """
    raster = cb.DrivableRaster((0.0, 0.0), 0.25, np.ones((80, 80), dtype=bool))
    cleared = raster.carve([(10.0, 10.0, 0.0, 2.0, 1.0)])

    assert cleared == pytest.approx((4.0 * 2.0) / 0.25 ** 2, rel=0.02)
    assert not raster.contains([10.0], [10.0])[0]        # inside the box
    assert raster.contains([15.0], [10.0])[0]            # 5 m away, untouched


def test_carve_respects_orientation():
    """A box rotated 90 degrees must clear the other way round, or every
    parked car is carved across the lanes instead of along them."""
    a = cb.DrivableRaster((0.0, 0.0), 0.25, np.ones((80, 80), dtype=bool))
    a.carve([(10.0, 10.0, 0.0, 2.5, 0.9)])
    b = cb.DrivableRaster((0.0, 0.0), 0.25, np.ones((80, 80), dtype=bool))
    b.carve([(10.0, 10.0, math.pi / 2, 2.5, 0.9)])

    # 2 m along the box's long axis: cleared in one, still drivable in the other
    assert not a.contains([12.0], [10.0])[0]
    assert b.contains([12.0], [10.0])[0]
    assert not b.contains([10.0], [12.0])[0]
    assert a.contains([10.0], [12.0])[0]


def test_carve_ignores_boxes_off_the_raster():
    """A box outside the rasterised town must not wrap around the array."""
    raster = cb.DrivableRaster((0.0, 0.0), 0.25, np.ones((40, 40), dtype=bool))
    assert raster.carve([(500.0, 500.0, 0.0, 2.0, 1.0)]) == 0
    assert raster.mask.all()


def test_the_world_carves_the_towns_parked_cars_out_of_free_space(carla_world):
    """End to end: a baked-in parked car must reach the planner as an obstacle.

    The geometry tests above prove ``carve`` clears a box. This proves the
    bridge actually asks CARLA for the boxes and applies them -- the wiring,
    which is the half that was missing and would have let the ego plan through
    29 parked vehicles in Town10HD.

    It also has to arrive in the *static* grid specifically. The static grid is
    the one the predictor sees, and a parked car is exactly the kind of thing
    it must treat as scenery rather than as an agent about to move.
    """
    world = carla_world()
    try:
        assert world._carved_cells > 0, "no cells cleared -- a frame conversion?"
        assert len(world._static_boxes) == len(fake_carla.World.PARKED)

        # The fake town parks two cars at CARLA y = -3.5, i.e. odom y = +3.5.
        for _name, cx, cy, _yaw, _hl, _hw in fake_carla.World.PARKED:
            x, y, _ = cb.carla_to_odom(cx, cy, 0.0)
            assert not world._raster.contains([x], [y])[0]

        # and the ego's own lane is untouched, or no episode could run at all
        assert world._raster.contains([60.0], [0.0])[0]
    finally:
        world.close()


def test_a_town_with_no_baked_obstacles_still_builds(carla_world, monkeypatch):
    """Most CARLA towns have few or none of most labels, and a bridge that
    needed them would fail on exactly the maps that are easiest to drive."""
    monkeypatch.setattr(fake_carla.World, "PARKED", ())
    world = carla_world()
    try:
        assert world._carved_cells == 0
        assert world._static_boxes == ()
        assert world._raster.mask.any()          # the road survived
    finally:
        world.close()


# --------------------------------------------------------------------------
# long routes that cross themselves
# --------------------------------------------------------------------------


def _lap():
    """One closed rectangular lap, 2 m spacing -- a town circuit in miniature."""
    pts = [[float(x), 0.0] for x in range(0, 101, 2)]
    pts += [[100.0, float(y)] for y in range(0, 61, 2)]
    pts += [[float(x), 60.0] for x in range(100, -1, -2)]
    pts += [[0.0, float(y)] for y in range(60, -1, -2)]
    return pts


def _two_way():
    """Out along one carriageway and back along the other, 3.5 m apart.

    The geometry that broke the first two attempts at windowing: spatially
    adjacent, tens of metres apart along the route.
    """
    out = [[float(x), 0.0] for x in range(0, 61, 2)]
    back = [[float(x), -3.5] for x in range(60, -1, -2)]
    return out, back


def _drive(route, points):
    return [route.progress(float(x), float(y)) for x, y in points]


def test_a_global_route_search_cannot_measure_a_lap_driven_twice():
    """The failure the windowed route exists to fix, pinned so it stays fixed.

    Every point of the second lap is nearest to a point of the first, so a
    global search reports the ego sliding back to the start line just as it
    finishes.
    """
    lap = _lap()
    route = cb.Route(np.array(lap + lap))
    progress = _drive(route, lap + lap)

    assert progress[-1] == pytest.approx(0.0)            # "back at the start"
    assert max(progress) < route.length * 0.6
    assert not all(b >= a for a, b in zip(progress, progress[1:]))


def test_a_windowed_route_measures_a_lap_driven_twice():
    lap = _lap()
    route = cb.Route(np.array(lap + lap), windowed=True)
    progress = _drive(route, lap + lap)

    assert progress[-1] == pytest.approx(route.length)
    assert all(b >= a for a, b in zip(progress, progress[1:]))


def test_the_cursor_cannot_teleport_to_the_opposite_carriageway():
    """The 72 m/s bug.

    Taking the argmin over a band of *arc length* mixes up two different
    notions of near: the opposite carriageway is 3.5 m away in space and tens
    of metres away along the route, so the cursor jumped to it and a 25 s run
    reported 1808 m of progress. Following the route to the first local
    minimum instead cannot cross that gap.
    """
    out, back = _two_way()
    route = cb.Route(np.array(out + back), windowed=True)
    progress = _drive(route, out + back)

    assert max(np.diff(progress)) < 5.0                  # no jump; 3.5 is the turn
    assert progress[-1] == pytest.approx(route.length)


def test_the_walk_does_not_start_behind_the_cursor():
    """The other half of the same geometry.

    Letting the walk begin 20 m back looks like harmless robustness and
    stalls the route: from there the nearest local minimum really is the
    outbound carriageway, 3.5 m away, so the cursor never follows onto the
    return leg. Measured: stalled at 67 m of 123 m.
    """
    out, back = _two_way()
    route = cb.Route(np.array(out + back), windowed=True)
    _drive(route, out)
    halfway = _drive(route, back[:len(back) // 2])[-1]

    assert halfway > route.length * 0.5


def test_progress_ratchets_when_the_ego_swings_wide():
    """Passing a parked lorry puts the nearest route point briefly behind.

    Reporting that as lost progress would show a backward jump in the metric
    and read as a stall to the runner's stuck detector.
    """
    route = cb.Route(np.array([[float(x), 0.0] for x in range(0, 101, 2)]),
                     windowed=True)
    _drive(route, [(float(x), 0.0) for x in range(0, 61, 2)])
    before = route.progress(60.0, 0.0)
    assert route.progress(57.0, 3.0) >= before


def test_route_from_waypoint_stops_on_a_loop_unless_told_otherwise(carla_world):
    """The default must not change: every published number was measured with
    the stop-on-loop rule and a globally searched route."""
    world = carla_world()
    try:
        assert world.road.windowed is False
    finally:
        world.close()


def test_a_long_route_is_windowed(carla_world):
    world = carla_world(long_route=True, route_length=2000.0)
    try:
        assert world.road.windowed is True
    finally:
        world.close()


def test_lane_context_reports_the_opendrive_ids(carla_world):
    """Telemetry only -- nothing above stage 3 may read it.

    The stack does not model lanes, so a lane change is an observation about
    the trajectory a free-space planner produced, not a manoeuvre it chose.
    """
    world = carla_world()
    try:
        ctx = world.lane_context()
        assert set(ctx) == {"road_id", "lane_id", "is_junction", "lane_width"}
        assert isinstance(ctx["lane_id"], int)
        assert ctx["lane_width"] > 0.0
    finally:
        world.close()


def _lane_events():
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    import record_carla
    return record_carla.lane_events


def test_lane_changes_are_counted_but_not_across_a_junction():
    """The count has to survive junctions or it means nothing.

    Lane ids inside a junction describe connecting roads, not lanes of a
    carriageway, and the road id changes on the far side. Comparing across one
    would score every turn as a lane change and inflate the headline number by
    roughly the number of junctions -- which on a town circuit is most of it.
    """
    lane_events = _lane_events()
    log = ([{"road_id": 1, "lane_id": -1, "is_junction": False}] * 5
           + [{"road_id": 1, "lane_id": -2, "is_junction": False}] * 5   # a real one
           + [{"road_id": 0, "lane_id": 9, "is_junction": True}] * 4
           + [{"road_id": 2, "lane_id": -1, "is_junction": False}] * 5)  # not one

    changes, traversals = lane_events(log)
    assert changes == [5]
    assert traversals == 1


def test_lane_id_alone_is_not_enough():
    """Ids are unique only within a road, so the pair is what changes."""
    lane_events = _lane_events()
    log = ([{"road_id": 1, "lane_id": -1, "is_junction": False}] * 3
           + [{"road_id": 2, "lane_id": -1, "is_junction": False}] * 3)
    changes, _ = lane_events(log)
    assert changes == [3]                   # same lane_id, different road


def test_missing_telemetry_samples_are_skipped_not_counted():
    """A frame off the mapped network must not read as two lane changes."""
    lane_events = _lane_events()
    log = ([{"road_id": 1, "lane_id": -1, "is_junction": False}] * 3
           + [{}]
           + [{"road_id": 1, "lane_id": -1, "is_junction": False}] * 3)
    assert lane_events(log)[0] == []


# --------------------------------------------------------------------------
# red lights
# --------------------------------------------------------------------------


def test_no_keepout_where_there_is_no_light(carla_world):
    """The normal state for most of any route. Code that assumes a light is
    always there fails on open road rather than at a junction."""
    world = carla_world()
    try:
        assert world.red_light_keepouts() == ()
    finally:
        world.close()


def test_a_red_light_becomes_a_keepout_across_every_stopped_lane(carla_world,
                                                                 monkeypatch):
    """Across *every* lane, not just the ego's.

    A free-space planner asked to avoid a barrier across one lane will drive
    round it into the oncoming one, which is a worse outcome than the red it
    was obeying.
    """
    world = carla_world()
    try:
        world.world.traffic_light = fake_carla.TrafficLight(
            fake_carla.TrafficLightState.Red
        )
        world.vehicle._tf.location.x = 55.0        # inside the trigger
        boxes = world.red_light_keepouts()

        assert len(boxes) == 2                           # both lanes stopped
        widths = [hw for _x, _y, _th, _hl, hw in boxes]
        assert all(w == pytest.approx(3.5 / 2) for w in widths)
    finally:
        world.close()


def test_green_and_yellow_do_not_stop_the_ego(carla_world):
    world = carla_world()
    try:
        world.vehicle._tf.location.x = 55.0
        for state in (fake_carla.TrafficLightState.Green,
                      fake_carla.TrafficLightState.Yellow):
            world.world.traffic_light = fake_carla.TrafficLight(state)
            assert world.red_light_keepouts() == ()
    finally:
        world.close()


def test_the_keepout_lands_in_the_static_grid_not_the_dynamic_one(carla_world):
    """A stop line is scenery, not an agent.

    In the dynamic grid it would reach the predictor as a stationary track to
    extrapolate, and a risk field around a stop line is not a thing.
    """
    world = carla_world()
    try:
        world.vehicle._tf.location.x = 55.0
        world.world.traffic_light = fake_carla.TrafficLight(
            fake_carla.TrafficLightState.Red
        )
        world.ego.x, world.ego.y = 55.0, 0.0
        static, _full = world.ground_truth_grids(half_extent=32.0)

        # The stop line sits at CARLA x=60, i.e. odom (60, 0) and (60, 3.5).
        assert static.data[static.world_to_cell(60.0, 0.0)[::-1]] == 1.0
    finally:
        world.close()


def test_obeying_lights_can_be_turned_off(carla_world):
    """Every CARLA number measured before this existed was measured without
    it, so it has to be possible to reproduce them."""
    world = carla_world(obey_traffic_lights=False)
    try:
        world.vehicle._tf.location.x = 55.0
        world.world.traffic_light = fake_carla.TrafficLight(
            fake_carla.TrafficLightState.Red
        )
        assert world.red_light_keepouts() == ()
    finally:
        world.close()
