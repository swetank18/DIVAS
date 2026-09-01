#!/usr/bin/env python3
"""Is the prediction confidence calibrated?

The dynamic safety margin rests on one assumption: that when the predictor
reports high confidence it is actually more likely to be right.  Nothing in
`PredictedTrajectory.confidence` enforces that.  It measures how far apart the
predictor's own hypotheses are -- which is a statement about the model's
internal disagreement, not about the world.  A model can be confidently wrong,
and if it is, the margin *shrinks* exactly when it should widen.

This script measures it.  For every prediction made during a run it stores the
predicted position at each horizon step and the confidence reported for it,
then looks up where the actor actually went and reports error against
confidence.

A useful confidence is *monotone*: error should fall as confidence rises.  If
it is flat, the dynamic margin is adding noise to the safety buffer rather than
information, and `lambda` should be zero until the predictor is calibrated.

    python3 scripts/calibration.py --predictor social_force --seeds 4
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path as FsPath

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))

import numpy as np

from divas.control.controllers import SamplingMPC
from divas.eval import scenarios
from divas.planning import FallbackPlanner
from divas.prediction.predictors import ConstantVelocityPredictor, SocialForcePredictor
from divas.prediction.risk import MarginParams, RiskField
from divas.types import VehicleParams


def collect(scenario, predictor, seed, dt=0.05, horizon_s=32.0):
    """Run one scenario, returning (confidence, error_m, lead_time_s) triples."""
    world = scenario.build(seed)
    params = world.params = VehicleParams()
    planner = FallbackPlanner(params)
    ctl = SamplingMPC(params)
    ego_extent = params.half_extent

    truth: dict = defaultdict(dict)      # actor id -> {step: (x, y)}
    preds: list = []                     # (step0, actor_id, k, conf, px, py)
    path = None
    risk = None
    step = 0
    while world.t < scenario.time_limit:
        for a in world.actors:
            if a.alive:
                truth[a.id][step] = (a.x, a.y)

        grid = world.ground_truth_grid()
        if step % 2 == 0:
            ts = predictor.predict(world.ground_truth_tracks(), grid, world.ego)
            risk = RiskField(ts, world.ego.v, ego_extent, MarginParams())
            risk.rasterize(grid)
            for tr in ts:
                conf = tr.confidence_profile()
                mean = tr.mean_path()
                for k in range(0, ts.n_steps, 5):     # every 0.5 s of horizon
                    # prediction dt is 0.1 s, sim dt is 0.05 s -> 2 sim steps
                    preds.append((step + 2 * (k + 1), tr.track_id, k,
                                  float(conf[k]), mean[k, 0], mean[k, 1]))
            last_pred = world.t
        if risk is not None:
            risk.age = world.t - last_pred
        if step % 5 == 0:
            res = planner.plan(grid, world.ego, world.local_goal(28.0), risk)
            if len(res.path) >= 2:
                path = res.path
        cmd = ctl.step(path, world.ego, grid, risk, dt)
        world.step(dt, cmd.accel, cmd.steer)
        step += 1
        if world.collision() is not None:
            break

    out = []
    for target_step, aid, k, conf, px, py in preds:
        actual = truth.get(aid, {}).get(target_step)
        if actual is None:
            continue
        out.append((conf, float(np.hypot(px - actual[0], py - actual[1])),
                    (k + 1) * 0.1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", default="social_force",
                    choices=("social_force", "constant_velocity"))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    pred = (SocialForcePredictor() if args.predictor == "social_force"
            else ConstantVelocityPredictor())
    rows = []
    for name in scenarios.SCENARIOS:
        sc = scenarios.get(name)
        if not sc.build(0).actors:
            continue
        for seed in range(args.seeds):
            rows += collect(sc, pred, seed)

    if not rows:
        print("no samples")
        return 1
    conf = np.array([r[0] for r in rows])
    err = np.array([r[1] for r in rows])
    lead = np.array([r[2] for r in rows])

    print(f"predictor : {args.predictor}")
    print(f"samples   : {len(rows)}")
    print(f"error     : mean {err.mean():.2f} m, p90 {np.percentile(err, 90):.2f} m\n")

    print("CALIBRATION -- error should fall as confidence rises")
    print(f"  {'confidence bin':>18s} {'n':>7s} {'mean err':>9s} {'p90 err':>9s}")
    edges = np.quantile(conf, np.linspace(0, 1, args.bins + 1))
    edges[-1] += 1e-9
    for i in range(args.bins):
        sel = (conf >= edges[i]) & (conf < edges[i + 1])
        if sel.sum() == 0:
            continue
        print(f"  {edges[i]:.2f} - {edges[i+1]:.2f}      {sel.sum():7d} "
              f"{err[sel].mean():9.2f} {np.percentile(err[sel], 90):9.2f}")

    r = float(np.corrcoef(conf, err)[0, 1])
    print(f"\n  corr(confidence, error) = {r:+.3f}   "
          "(want clearly negative; ~0 means uninformative)")

    print("\nERROR BY LEAD TIME -- sanity check that the horizon behaves")
    for t in sorted(set(np.round(lead, 1))):
        sel = np.abs(lead - t) < 1e-6
        print(f"  t+{t:.1f}s  n={sel.sum():6d}  mean err {err[sel].mean():5.2f} m")

    if r > -0.1:
        print("\nVERDICT: confidence is NOT informative about error on this suite.")
        print("The dynamic margin is therefore modulating the safety buffer with")
        print("noise, not information. Set MarginParams.lam = 0 until the")
        print("predictor is calibrated -- see docs/decisions/.")
    else:
        print("\nVERDICT: confidence carries signal; the dynamic margin has a basis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
