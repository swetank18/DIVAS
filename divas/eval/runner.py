"""Closed-loop runner -- the vertical slice.

Stages 1-3 are ground-truth stubs served by :mod:`divas.sim.world`; stages 4-6
are the real modules.  Everything talks through the contracts in
:mod:`divas.types`, so replacing a stub with a trained model changes nothing
here.

The three rates from ``EXECUTION_PLAN.md`` section 2 are honoured rather than
collapsed into one: perception/prediction at 10 Hz, planning at 4 Hz, control
at 20 Hz.  Collapsing them would make every result optimistic, because the
controller would never once have to act on a stale plan -- which on the real
vehicle is the normal case, not the exception.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np

from divas.control.controllers import (
    Controller,
    MPCConfig,
    PurePursuitController,
    SamplingMPC,
)
from divas.eval.metrics import RunMetrics
from divas.eval.scenarios import Scenario
from divas.planning import FallbackPlanner, PlannerConfig, RRTConfig
from divas.prediction.predictors import (
    ConstantVelocityPredictor,
    Predictor,
    SocialForcePredictor,
)
from divas.prediction.risk import MarginParams, RiskField
from divas.types import EgoState, VehicleParams


@dataclass
class StackConfig:
    """One arm of the ablation."""

    name: str
    predictor: str = "social_force"      # social_force | constant_velocity | none
    margin: MarginParams = field(default_factory=MarginParams)
    controller: str = "mpc"              # mpc | pure_pursuit
    description: str = ""

    def build(self, params: VehicleParams):
        pred: Optional[Predictor]
        if self.predictor == "social_force":
            pred = SocialForcePredictor()
        elif self.predictor == "constant_velocity":
            pred = ConstantVelocityPredictor()
        elif self.predictor == "none":
            pred = None
        else:
            raise ValueError(f"unknown predictor {self.predictor!r}")
        # Evaluation runs iteration-bounded, not clock-bounded, so a result
        # is a property of the algorithm rather than of the machine it ran on.
        # The deployment budget is exercised separately -- see
        # docs/decisions/ADR-008-deterministic-evaluation.md.
        planner = FallbackPlanner(
            params,
            PlannerConfig(time_budget_ms=None),
            RRTConfig(time_budget_ms=None),
        )
        ctrl: Controller = (
            SamplingMPC(params, MPCConfig())
            if self.controller == "mpc"
            else PurePursuitController(params)
        )
        return pred, planner, ctrl


# The ablation.  Each row removes exactly one thing from the row below it, so
# a difference in the table can be attributed to a single mechanism.
ABLATION: List[StackConfig] = [
    StackConfig(
        "baseline_conventional",
        predictor="constant_velocity",
        margin=MarginParams.fixed(1.0),
        controller="pure_pursuit",
        description="Constant-velocity prediction, fixed 1.0 m buffer, geometric control",
    ),
    StackConfig(
        "no_prediction",
        predictor="none",
        margin=MarginParams.fixed(1.0),
        controller="mpc",
        description="MPC over current occupancy only -- no prediction at all",
    ),
    StackConfig(
        "cv_pred_fixed_margin",
        predictor="constant_velocity",
        margin=MarginParams.fixed(1.0),
        controller="mpc",
        description="MPC + constant-velocity prediction + fixed margin",
    ),
    StackConfig(
        "interaction_pred_fixed_margin",
        predictor="social_force",
        margin=MarginParams.fixed(1.0),
        controller="mpc",
        description="MPC + interaction-aware prediction + fixed margin",
    ),
    StackConfig(
        "fixed_wide_margin",
        predictor="social_force",
        margin=MarginParams.fixed(1.4),
        controller="mpc",
        description="Control arm: fixed margin matching the dynamic arm's MEAN, "
                    "to separate the effect of margin SIZE from margin VARIATION",
    ),
    StackConfig(
        "full_dynamic_margin",
        predictor="social_force",
        margin=MarginParams(),
        controller="mpc",
        description="Full stack: interaction-aware prediction + confidence-scaled margin",
    ),
]


@dataclass
class RunnerConfig:
    sim_dt: float = 0.05          # 20 Hz control
    perception_every: int = 2     # 10 Hz
    plan_every: int = 5           # 4 Hz
    # Must stay inside the costmap's half-extent, with room for the goal
    # tolerance.  A route goal outside the local map is unreachable by
    # construction, and the planner then burns its entire time budget every
    # cycle proving it -- which reads as "the planner is slow" rather than
    # "the goal is off the map".
    # Must also exceed the MPC's own horizon (cruise_speed x MPC horizon,
    # ~18 m).  A route goal closer than that leaves every rollout running off
    # the end of the path, where cross-track cost grows without bound -- and
    # the controller's cheapest response is to brake until its horizon fits
    # inside the path, which looks exactly like unexplained timidity.
    goal_lookahead: float = 28.0
    stuck_speed: float = 0.15     # m/s
    stuck_time: float = 5.0       # s below stuck_speed before giving up
    record: bool = False
    record_every: int = 1         # trace stride; 1 = every control step


def run(
    scenario: Scenario,
    stack: StackConfig,
    seed: int = 0,
    params: Optional[VehicleParams] = None,
    cfg: Optional[RunnerConfig] = None,
) -> RunMetrics:
    cfg = cfg or RunnerConfig()
    world = scenario.build(seed)
    # A simulator may already know its own vehicle -- the CARLA bridge reads
    # length, width, wheelbase and steering limit off the blueprint it actually
    # spawned.  Overwriting that with the defaults would mean measuring
    # clearance for one vehicle while driving another, so the caller's params
    # win only when the caller supplied any.
    params = params or getattr(world, "params", None) or VehicleParams()
    world.params = params
    predictor, planner, controller = stack.build(params)
    ego_extent = params.half_extent

    m = RunMetrics(scenario=scenario.name, stack=stack.name, seed=seed)
    trace = {"t": [], "ego": [], "v": [], "actors": [], "path": [],
             "d_safe": [], "progress": []}

    try:
        grid = None
        risk: Optional[RiskField] = None
        path = None
        last_pred_t = 0.0
        prev_accel = 0.0
        prev_v = world.ego.v
        speeds: List[float] = []
        d_safes: List[float] = []
        stuck_for = 0.0
        start_progress = world.road.progress(world.ego.x, world.ego.y)
        step = 0

        while world.t < scenario.time_limit:
            ego = world.ego

            # -- stages 1-3 (stubbed) + stage 4 ---------------------------
            if step % cfg.perception_every == 0 or grid is None:
                static_grid, grid = world.ground_truth_grids()
                tracks = world.ground_truth_tracks()
                if predictor is not None:
                    t0 = time.perf_counter()
                    ts = predictor.predict(tracks, static_grid, ego)
                    risk = RiskField(ts, ego.v, ego_extent, stack.margin)
                    risk.rasterize(grid)
                    m.timer("predict").add((time.perf_counter() - t0) * 1e3)
                    # Only when there is traffic to keep a margin from.  Averaging
                    # in the zero-actor scenarios drags every stack's figure
                    # towards zero and makes the fixed/dynamic comparison -- the
                    # one number the ablation exists to produce -- meaningless.
                    if len(ts):
                        d_safes.append(risk.mean_margin())
                last_pred_t = world.t
            if risk is not None:
                risk.age = world.t - last_pred_t

            # -- stage 5 ---------------------------------------------------
            if step % cfg.plan_every == 0:
                goal = world.local_goal(cfg.goal_lookahead)
                t0 = time.perf_counter()
                res = planner.plan(grid, ego, goal, risk)
                m.timer("plan").add((time.perf_counter() - t0) * 1e3)
                m.plan_calls += 1
                m.plan_success += int(res.success)
                m.plan_partial += int(res.partial)
                # A partial plan is still useful -- it is the best reachable
                # prefix.  Only an empty result leaves the previous path standing.
                if len(res.path) >= 2:
                    path = res.path

            # -- stage 6 ---------------------------------------------------
            t0 = time.perf_counter()
            cmd = controller.step(path, ego, grid, risk, cfg.sim_dt)
            m.timer("control").add((time.perf_counter() - t0) * 1e3)

            if cfg.record and step % cfg.record_every == 0:
                trace["t"].append(world.t)
                trace["ego"].append((ego.x, ego.y, ego.theta))
                trace["v"].append(ego.v)
                trace["actors"].append(
                    [(a.x, a.y, a.theta, a.cls) for a in world.actors if a.alive]
                )
                trace["path"].append(path.xy.copy() if path is not None else None)
                trace["d_safe"].append(risk.mean_margin() if risk is not None else 0.0)
                trace["progress"].append(m.progress)

            world.step(cfg.sim_dt, cmd.accel, cmd.steer)
            step += 1

            # -- metrics ---------------------------------------------------
            e = world.ego
            speeds.append(e.v)
            # Jerk from the realised acceleration.  Samples where the speed hits
            # the zero floor are skipped: the clip means the vehicle decelerated
            # less than commanded, so the difference quotient reports a spike that
            # the occupant never felt.  Left in, it made a stalled run look like a
            # violently jerky one and buried the real comfort signal.
            a_act = (e.v - prev_v) / cfg.sim_dt
            if e.v > 1e-6 and prev_v > 1e-6:
                m.max_jerk = max(m.max_jerk, abs(a_act - prev_accel) / cfg.sim_dt)
            prev_accel, prev_v = a_act, e.v
            m.max_lat_accel = max(
                m.max_lat_accel, abs(e.v**2 * np.tan(e.delta) / params.wheelbase)
            )
            m.max_speed = max(m.max_speed, e.v)

            ttc = world.time_to_collision()
            m.ttc_samples.append(ttc)
            m.min_ttc = min(m.min_ttc, ttc)
            m.min_clearance = min(m.min_clearance, world.clearance_to_actors())

            hit = world.collision()
            if hit is not None:
                m.collision, m.collision_with = True, hit
                break

            if e.v < cfg.stuck_speed:
                stuck_for += cfg.sim_dt
                if stuck_for > cfg.stuck_time:
                    m.timeout = True
                    break
            else:
                stuck_for = 0.0

            m.progress = world.road.progress(e.x, e.y) - start_progress
            if m.progress >= scenario.goal_progress:
                m.success = True
                break

        m.sim_time = world.t
        m.mean_speed = float(np.mean(speeds)) if speeds else 0.0
        m.mean_d_safe = float(np.mean(d_safes)) if d_safes else 0.0
        if not m.success and not m.collision:
            m.timeout = True
        if cfg.record:
            m.trace = trace
        return m
    finally:
        # CARLA keeps every actor a disconnected client spawned, so a run that
        # dies without cleaning up leaves its traffic parked across the map and
        # the *next* run spawns into a town that is already full -- which does
        # not crash, it just quietly produces worse numbers.  The built-in
        # world has no close(); this costs it nothing.
        closer = getattr(world, "close", None)
        if callable(closer):
            closer()
