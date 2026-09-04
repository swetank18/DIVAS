#!/usr/bin/env python3
"""How wide a safety margin does each predictor actually require?

A conventional predictor is scored with ADE/FDE -- average and final
displacement error, in metres, averaged over agents. Those numbers are hard to
act on: they say the predictor is 1.4 m wrong on average, and nothing about how
much room the planner must therefore leave.

This script measures the quantity the planner actually needs, which falls
straight out of conformal calibration:

    RSM(t, alpha) = the (1 - alpha) quantile of this predictor's displacement
                    error at lookahead t, in metres

Call it the **required safety margin**. It is the radius a keep-out must have
for the actor to be inside it (1 - alpha) of the time at that lookahead, and it
is measured on this predictor, on these scenarios, with no distributional
assumption. Two properties make it more useful than ADE:

* it is a *quantile*, not a mean, so it speaks about the tail that collides
  rather than the average that does not;
* it is denominated in metres of carriageway, which is a budget the vehicle
  either has or does not. A 10 m Indian carriageway with a 3.6 m vehicle and
  oncoming traffic does not have 5 m to give.

Run it before and after any change to the predictor. A predictor improvement
that does not reduce RSM has not bought the planner anything.

    python3 scripts/required_margin.py
    python3 scripts/required_margin.py --alpha 0.2 --predictors constant_velocity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as FsPath

import numpy as np

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))

from divas.eval import runner
from divas.eval.scenarios import SCENARIOS, get
from divas.prediction.conformal import ConformalCalibrator, ConformalConfig
from divas.prediction.predictors import (
    ConstantVelocityPredictor,
    SocialForcePredictor,
)
from divas.prediction.risk import MarginParams


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictors", nargs="+",
                    default=["constant_velocity", "social_force"])
    ap.add_argument("--scenarios", nargs="+", default=sorted(SCENARIOS))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--track-noise", type=float, default=None,
                    help="override the simulator's measurement noise. 0 isolates "
                         "model error from sensor noise: the velocity term is "
                         "extrapolated over the horizon, so at 3 s it dominates "
                         "the residual and would otherwise be read as a bad "
                         "motion model")
    ap.add_argument("--out", default="docs/required-margin.json")
    args = ap.parse_args()

    results = {}
    for name in args.predictors:
        # window huge and adaptation off: this is a measurement of the whole
        # distribution, not a controller tracking a moving one.
        cal = ShadowCalibrator(
            30, 0.1, ConformalConfig(alpha=args.alpha, adaptive=False,
                                     window=1000000, min_samples=50,
                                     max_margin=1e9)
        )
        arm = runner.StackConfig(name=f"probe_{name}", predictor=name,
                                 margin=MarginParams.fixed(1.0), controller="mpc")
        for scen in args.scenarios:
            for seed in range(1, args.seeds + 1):
                _drive_and_calibrate(arm, scen, seed, cal, args.track_noise)
        # margins() is silenced on the shadow; read the quantiles directly.
        q = np.array([cal._quantile_at(k) for k in range(cal.n_steps)])
        results[name] = {
            "alpha": args.alpha,
            "rsm": [round(float(v), 3) for v in q],
            "n_scored": cal.n_scored,
            "coverage": cal.coverage(),
        }
        print(f"\n{name}  ({cal.n_scored} residuals, "
              f"{len(args.scenarios)} scenarios x {args.seeds} seeds)")
        print(f"  lookahead  0.5 s   1.0 s   1.5 s   2.0 s   2.5 s   3.0 s")
        idx = [4, 9, 14, 19, 24, 29]
        print("  RSM, m    " + "".join(f"{q[i]:7.2f} " for i in idx))

    out = FsPath(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")

    if len(results) == 2:
        a, b = args.predictors[0], args.predictors[1]
        qa = np.array(results[a]["rsm"]); qb = np.array(results[b]["rsm"])
        print(f"\n{b} vs {a}, required margin at 3 s: "
              f"{qb[-1]:.2f} m vs {qa[-1]:.2f} m "
              f"({100*(qb[-1]-qa[-1])/qa[-1]:+.1f}%)")
    return 0


class ShadowCalibrator(ConformalCalibrator):
    """Scores every prediction, and reports no margin at all.

    This is the whole trick of the measurement. If the calibrated margin were
    fed back to the planner, the ego would drive differently -- more timidly,
    with an uncapped margin it would barely move -- and the predictor would
    then be scored on a distribution of situations that only exists because of
    the margin being measured. The measurement would be a function of itself.

    So ``margins()`` returns zeros: the risk field falls back to exactly the
    fixed 1.0 m keep-out of the baseline arm, the ego behaves identically to
    it, and the calibration happens silently alongside.
    """

    _zero = None

    def margins(self):
        # Zeros, and allocated once. The base class calls margins() every
        # cycle to record the margin in force; letting that fall through to a
        # real quantile would partition a million-sample window on every tick,
        # and the sweep would never finish.
        if self._zero is None:
            self._zero = np.zeros(self.n_steps)
        return self._zero


def _drive_and_calibrate(arm, scen, seed, cal, track_noise=None):
    """Run one episode, feeding the calibrator from the runner's own loop.

    Attached by monkeypatching rather than by threading a second parameter
    through ``StackConfig``: this is a measurement harness, and the runner
    should not grow an argument that only a script uses.
    """
    from divas.eval import runner as R
    original = R.ConformalCalibrator

    class _Attach(original):
        def __new__(cls, *a, **k):
            return cal

    R.ConformalCalibrator = _Attach
    scenario = get(scen)
    if track_noise is not None:
        original_build = scenario.build

        def _quiet_build(seed, _b=original_build, _n=track_noise):
            w = _b(seed)
            w.track_noise = float(_n)
            return w

        scenario = scenario.__class__(
            name=scenario.name, description=scenario.description,
            build=_quiet_build, goal_progress=scenario.goal_progress,
            time_limit=scenario.time_limit,
            tests=getattr(scenario, "tests", ""))
    arm2 = R.StackConfig(name=arm.name, predictor=arm.predictor,
                         margin=arm.margin, controller=arm.controller,
                         conformal=ConformalConfig())
    try:
        R.run(scenario, arm2, seed=seed)
    finally:
        R.ConformalCalibrator = original


if __name__ == "__main__":
    raise SystemExit(main())
