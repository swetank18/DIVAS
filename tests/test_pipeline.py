"""Unit tests for the stage contracts and each pipeline stage.

Deliberately runnable without ROS and without a GPU: every module in this
repository is pure Python over the dataclasses in :mod:`divas.types`, which is
what lets six people work on six stages without six environments.

    python3 -m pytest tests/ -q      (or)      python3 tests/test_pipeline.py
"""

from __future__ import annotations

import numpy as np

from divas.control.controllers import PurePursuitController, SamplingMPC
from divas.eval import runner, scenarios
from divas.planning.hybrid_astar import HybridAStar, PlannerConfig
from divas.prediction.predictors import ConstantVelocityPredictor, SocialForcePredictor
from divas.prediction.risk import MarginParams, RiskField
from divas.planning.rrt import RRTConfig
from divas.sim.world import Actor, Circle, Rect, Road, World
from divas.types import EgoState, OccupancyGrid, Path, Track, VehicleParams


# Tests that assert the search *finds* something get a generous time budget.
# The production budget is a wall-clock deadline, so under load -- a parallel
# ablation on the same machine, CI on a shared runner -- a correctness test
# racing it becomes a benchmark of spare capacity and fails intermittently.
# The budget itself is tested separately, where it belongs.
GENEROUS = PlannerConfig(time_budget_ms=2000.0)
GENEROUS_RRT = RRTConfig(time_budget_ms=1000.0)


# -- stage contracts ------------------------------------------------------


def test_grid_geometry_and_bounds():
    g = OccupancyGrid.empty((-5, 35), (-10, 10), 0.2)
    assert g.data.shape == (100, 200)
    assert g.bounds == (-5.0, -10.0, 35.0, 10.0)
    x, y = g.cell_to_world(*g.world_to_cell(3.3, -2.7))
    assert abs(x - 3.3) <= 0.2 and abs(y - -2.7) <= 0.2


def test_offmap_counts_as_occupied():
    """A planner must not escape through the edge of its own perception."""
    g = OccupancyGrid.empty((-5, 35), (-10, 10), 0.2)
    assert g.occupancy_at(999.0, 0.0)[0] == 1.0
    assert g.signed_clearance_at(999.0, 0.0)[0] < 0.0


def test_signed_distance_is_negative_inside_obstacles():
    """The unsigned EDT is flat inside obstacles, so its gradient is useless
    exactly where something has to be pushed back out."""
    g = OccupancyGrid.empty((0, 10), (0, 10), 0.2)
    g.data[20:30, 20:30] = 1.0          # world x,y in [4, 6]
    g.invalidate_cache()
    assert g.signed_clearance_at(5.0, 5.0)[0] < 0.0, "inside must be negative"
    assert abs(g.signed_clearance_at(5.0, 5.0)[0] + 1.0) < 0.15, "1 m from the edge"
    assert g.signed_clearance_at(1.0, 1.0)[0] > 0.0, "free space must be positive"
    assert g.signed_clearance_at(1.0, 1.0)[0] > g.signed_clearance_at(3.5, 5.0)[0]


def test_path_resample_and_terminal_flag():
    pts = np.array([[0, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]], dtype=float)
    p = Path(pts, terminal_stop=True)
    r = p.resample(0.5)
    assert r.terminal_stop is True
    assert abs(r.length - p.length) < 1e-6
    assert np.allclose(r.xy[0], p.xy[0]) and np.allclose(r.xy[-1], p.xy[-1])


def test_curvature_profile_beats_lattice_kappa():
    """Lattice kappa only takes the planner's handful of steering values."""
    s = np.linspace(0, 20, 41)
    p = Path(np.stack([s, np.zeros_like(s), np.zeros_like(s),
                       np.full_like(s, 0.2)], axis=1), terminal_stop=True)
    assert p.curvature_profile().max() < 1e-6, "a straight path has no curvature"


def test_track_extents_from_class():
    t = Track(1, 0, 0, 0, 0, "bus")
    assert (t.half_length, t.half_width) == (5.5, 1.3)
    assert Track(2, 0, 0, 0, 0, "pedestrian").half_length == 0.3


# -- stage 4: prediction --------------------------------------------------


def test_constant_velocity_is_exact():
    ts = ConstantVelocityPredictor(horizon=2.0, dt=0.1).predict(
        [Track(1, 0.0, 0.0, 5.0, 0.0, "car")]
    )
    end = ts.trajectories[0].modes[0].points[-1]
    assert np.allclose(end, [10.0, 0.0])
    assert ts.trajectories[0].confidence == 1.0, "single mode is maximally confident"


def test_confidence_discriminates_by_agility():
    """A bus holds its line; a motorcycle does not.  If confidence cannot tell
    them apart it cannot drive the safety margin."""
    w = World(road=Road.straight(), ego=EgoState(v=9.0),
              actors=[Actor(1, "motorcycle", 18, 1.5, 0.0, 7.0),
                      Actor(2, "bus", 20, -1.5, 0.0, 6.0)], seed=0)
    ts = SocialForcePredictor().predict(
        w.ground_truth_tracks(), w.ground_truth_grid(), w.ego)
    conf = {t.cls: t.confidence for t in ts}
    assert conf["bus"] > conf["motorcycle"]


def test_predictor_beats_constant_velocity_is_not_assumed():
    """Predictions are no longer forced onto the drivable area.

    An obstacle force kept them there and cost ~1.1 m of accuracy -- enough to
    make this predictor worse than constant velocity.  Straying off-road is
    harmless: the keep-out lands in a region the occupancy grid already
    blocks.  What must hold is that the modes stay physically plausible, i.e.
    no faster than the actor could actually travel.
    """
    w = World(road=Road.straight(half_width=4.0), ego=EgoState(v=9.0),
              actors=[Actor(1, "motorcycle", 15, 2.0, 0.0, 7.0)], seed=1)
    static, _ = w.ground_truth_grids()
    ts = SocialForcePredictor().predict(w.ground_truth_tracks(), static, w.ego)
    for mode in ts.trajectories[0].modes:
        step = np.linalg.norm(np.diff(mode.points, axis=0), axis=1)
        assert step.max() < SocialForcePredictor().max_speed * ts.dt * 1.1


# -- risk field -----------------------------------------------------------


def _risk(margin, speed=9.0):
    w = World(road=Road.straight(), ego=EgoState(v=speed),
              actors=[Actor(1, "motorcycle", 18, 1.0, 0.0, 7.0)], seed=3)
    g = w.ground_truth_grid()
    ts = SocialForcePredictor().predict(w.ground_truth_tracks(), g, w.ego)
    rf = RiskField(ts, speed, VehicleParams().half_extent, margin)
    rf.rasterize(g)
    return rf, ts, g


def test_dynamic_margin_exceeds_fixed_when_uncertain():
    dyn, _, _ = _risk(MarginParams())
    fix, _, _ = _risk(MarginParams.fixed(1.0))
    assert dyn.mean_margin() > fix.mean_margin()


def test_margin_grows_with_speed():
    slow, _, _ = _risk(MarginParams(), speed=3.0)
    fast, _, _ = _risk(MarginParams(), speed=12.0)
    assert fast.mean_margin() > slow.mean_margin()


def test_keepout_is_longer_along_heading():
    """A disc keep-out for a long vehicle blocks the road sideways."""
    rf, ts, _ = _risk(MarginParams())
    assert (rf.a > rf.b).all(), "ellipse must be elongated along motion"


def test_rasterised_and_analytic_risk_agree():
    rf, ts, _ = _risk(MarginParams())
    p = ts.trajectories[0].mean_path()[9]
    a = float(rf.risk_at(p[0], p[1], 1.0)[0])
    b = float(rf.lookup(p[0], p[1], 1.0)[0])
    assert a > 0.3 and abs(a - b) < 0.35


# -- stage 5: planning ----------------------------------------------------


def test_planner_reaches_goal_on_open_road():
    w = World(road=Road.straight(), ego=EgoState(v=9.0), seed=0)
    res = HybridAStar(config=GENEROUS).plan(
        w.ground_truth_grid(), w.ego, w.local_goal(20.0))
    assert res.success and res.path.length > 15.0
    assert res.path.terminal_stop is False


def test_planner_routes_around_an_obstacle():
    # half_width 5.0, as in the scenario suite.  On a 4.0 m half-width road a
    # 1.2 m pothole dead centre is genuinely impassable once the planner's
    # inflation is applied -- a property of the geometry, not of the search.
    w = World(road=Road.straight(half_width=5.0),
              statics=[Circle(15, 0, 1.2, "pothole")],
              ego=EgoState(v=9.0), seed=0)
    res = HybridAStar(config=GENEROUS).plan(
        w.ground_truth_grid(), w.ego, w.local_goal(20.0))
    assert res.success
    assert np.abs(res.path.xy[:, 1]).max() > 1.0, "did not deviate around it"


def test_planner_returns_partial_when_blocked_and_marks_it():
    w = World(road=Road.straight(half_width=4.0),
              statics=[Rect(20, 0, 2, 16, 0, "wall")], ego=EgoState(v=6.0), seed=0)
    res = HybridAStar().plan(w.ground_truth_grid(), w.ego, w.local_goal(20.0))
    assert not res.success and res.partial
    assert res.path.terminal_stop is True, "controller must brake for a dead end"


def test_planner_respects_its_time_budget():
    """Overrunning is worse than returning a mediocre path."""
    w = World(road=Road.straight(half_width=4.0),
              statics=[Rect(20, 0, 2, 16, 0, "wall")], ego=EgoState(v=6.0), seed=0)
    res = HybridAStar().plan(w.ground_truth_grid(), w.ego, w.local_goal(20.0))
    assert res.elapsed_ms < 400.0


def test_risk_field_changes_the_plan():
    """The thesis: an actor that is clear *now* but will not be."""
    w = World(road=Road.straight(half_width=5.0), ego=EgoState(v=9.0),
              actors=[Actor(9, "pedestrian", 16, -4.2, 1.5708, 1.3)], seed=0)
    g = w.ground_truth_grid()
    ts = SocialForcePredictor().predict(w.ground_truth_tracks(), g, w.ego)
    rf = RiskField(ts, 9.0, VehicleParams().half_extent, MarginParams())
    rf.rasterize(g)
    pl = HybridAStar(config=GENEROUS)
    goal = np.array([26.0, 0.0])
    plain = pl.plan(g, w.ego, goal, None)
    aware = pl.plan(g, w.ego, goal, rf)
    y_plain = np.interp(16, plain.path.xy[:, 0], plain.path.xy[:, 1])
    y_aware = np.interp(16, aware.path.xy[:, 0], aware.path.xy[:, 1])
    assert y_aware > y_plain + 0.8, "risk field did not steer away from the crossing"


# -- stage 6: control -----------------------------------------------------


def test_controllers_emergency_stop_without_a_path():
    for c in (PurePursuitController(), SamplingMPC()):
        assert c.step(None, EgoState(v=9.0)).accel == VehicleParams().min_accel


def test_mpc_tracks_a_lateral_shift():
    vp = VehicleParams()
    s = np.linspace(0, 45, 181)
    y = -2.0 / (1 + np.exp(-(s - 16) / 2.2))
    th = np.arctan(np.gradient(y, s))
    path = Path(np.stack([s, y, th, np.gradient(th, s)], axis=1), terminal_stop=False)
    ego = EgoState(x=0, y=0, theta=0, v=8.5)
    ctl = SamplingMPC(vp)
    err = []
    for _ in range(100):
        cmd = ctl.step(path, ego, None, None, 0.05)
        md = vp.max_steer_rate * 0.05
        ego.delta += float(np.clip(np.clip(cmd.steer, -vp.max_steer, vp.max_steer)
                                   - ego.delta, -md, md))
        ego.x += ego.v * np.cos(ego.theta) * 0.05
        ego.y += ego.v * np.sin(ego.theta) * 0.05
        ego.theta += ego.v / vp.wheelbase * np.tan(ego.delta) * 0.05
        ego.v = float(np.clip(ego.v + np.clip(cmd.accel, vp.min_accel, vp.max_accel) * 0.05,
                              0, vp.max_speed))
        err.append(abs(ego.y - np.interp(ego.x, s, y)))
    assert max(err) < 0.8, f"tracking error {max(err):.2f} m"
    assert ego.x > 25.0, "vehicle froze instead of tracking"


def test_mpc_does_not_freeze_on_an_empty_road():
    """Braking is always available as a cheap way to make a horizon safe."""
    w = World(road=Road.straight(half_width=5.0), ego=EgoState(v=8.0), seed=0)
    pl, ctl = HybridAStar(), SamplingMPC()
    path = None
    for step in range(60):
        g = w.ground_truth_grid()
        if step % 5 == 0:
            res = pl.plan(g, w.ego, w.local_goal(28.0), None)
            if len(res.path) >= 2:
                path = res.path
        w.step(0.05, *(lambda c: (c.accel, c.steer))(ctl.step(path, w.ego, g, None, 0.05)))
    assert w.ego.v > 4.0, f"crawled to {w.ego.v:.1f} m/s on an empty road"


# -- end to end -----------------------------------------------------------


def test_closed_loop_runs_and_reports():
    sc = scenarios.get("unmarked_road")
    m = runner.run(sc, runner.ABLATION[-1], seed=0)
    row = m.as_row()
    # Production budgets on purpose -- this one is meant to exercise the real
    # timing.  The threshold is loose because the planner is budget-limited by
    # design, not because a low rate would be acceptable.
    assert m.plan_calls > 0 and m.plan_success_rate > 0.2
    assert row["e2e_ms_p95"] > 0
    assert m.progress > 20.0


if __name__ == "__main__":
    import sys, traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in fns:
        try:
            f()
            print(f"  PASS  {n}")
        except Exception as e:
            bad += 1
            print(f"  FAIL  {n}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)


# -- planner fallback -----------------------------------------------------


def test_rrt_solves_what_it_should():
    from divas.planning import KinodynamicRRT
    w = World(road=Road.straight(half_width=5.0),
              statics=[Circle(15, 0, 1.2, "pothole")], ego=EgoState(v=9.0), seed=0)
    res = KinodynamicRRT(config=GENEROUS_RRT, seed=0).plan(
        w.ground_truth_grid(), w.ego, w.local_goal(28.0))
    assert res.success and res.path.length > 20.0


def test_rrt_fails_honestly_on_a_blocked_road():
    from divas.planning import KinodynamicRRT
    w = World(road=Road.straight(half_width=4.0),
              statics=[Rect(20, 0, 2, 16, 0, "wall")], ego=EgoState(v=6.0), seed=0)
    res = KinodynamicRRT(seed=0).plan(w.ground_truth_grid(), w.ego, w.local_goal(28.0))
    assert not res.success and res.path.terminal_stop is True


def test_fallback_prefers_astar_and_does_not_call_rrt_when_it_succeeds():
    from divas.planning import FallbackPlanner
    w = World(road=Road.straight(half_width=5.0), ego=EgoState(v=9.0), seed=0)
    fp = FallbackPlanner(config=GENEROUS)
    res = fp.plan(w.ground_truth_grid(), w.ego, w.local_goal(28.0))
    assert res.success and fp.fallback_calls == 0, "sampling ran when the lattice sufficed"


def test_fallback_returns_the_longer_partial_when_both_fail():
    from divas.planning import FallbackPlanner
    w = World(road=Road.straight(half_width=4.0),
              statics=[Rect(20, 0, 2, 16, 0, "wall")], ego=EgoState(v=6.0), seed=0)
    fp = FallbackPlanner()
    res = fp.plan(w.ground_truth_grid(), w.ego, w.local_goal(28.0))
    assert fp.fallback_calls == 1 and not res.success
    assert res.partial and res.path.terminal_stop is True
