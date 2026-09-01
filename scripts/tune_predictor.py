#!/usr/bin/env python3
"""Open-loop prediction accuracy: sweep the social-force parameters.

Isolates stage 4.  The ego is driven along the road centreline at a fixed
speed and the controller is not run at all, so this measures the predictor and
nothing else -- and does it about fifty times faster than a closed-loop run.

Motivation: on the scenario suite the interaction-aware predictor is *less*
accurate than plain constant velocity (4.73 m vs 3.64 m mean error), and the
closed-loop ablation follows exactly that ordering.  Either the forces are
mis-tuned, or the benchmark cannot reward interaction-awareness.  This tells
the two apart.

    python3 scripts/tune_predictor.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path as FsPath

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))

import numpy as np

from divas.eval import scenarios
from divas.prediction.predictors import ConstantVelocityPredictor, SocialForcePredictor


def score(predictor, seeds=3, ego_speed=8.0, dt=0.05, static_actors=False,
          only=None) -> dict:
    """Mean / p90 prediction error over the suite, open loop."""
    errs, leads = [], []
    for name in scenarios.SCENARIOS:
        if only and name not in only:
            continue
        sc = scenarios.get(name)
        if not sc.build(0).actors:
            continue
        for seed in range(seeds):
            w = sc.build(seed)
            truth = defaultdict(dict)
            preds = []
            s0 = w.road.progress(w.ego.x, w.ego.y)
            g = None
            step = 0
            while w.t < sc.time_limit:
                for a in w.actors:
                    if a.alive:
                        truth[a.id][step] = (a.x, a.y)
                if step % 2 == 0:
                    if step % 20 == 0 or g is None:
                        # Static-only, and cached.  Static-only because the
                        # predictor already repels agents from each other via
                        # its social term -- feeding it a grid that also
                        # contains those agents applies the repulsion twice.
                        g = w.ground_truth_grid(include_actors=static_actors)
                    ts = predictor.predict(w.ground_truth_tracks(), g, w.ego)
                    for tr in ts:
                        mean = tr.mean_path()
                        for k in range(0, ts.n_steps, 5):
                            preds.append((step + 2 * (k + 1), tr.track_id,
                                          mean[k, 0], mean[k, 1], (k + 1) * 0.1))
                # Ego rides the centreline; the controller is deliberately absent.
                p = w.road.offset_point(s0 + ego_speed * w.t, 0.0)
                nxt = w.road.offset_point(s0 + ego_speed * w.t + 1.0, 0.0)
                w.ego.x, w.ego.y = float(p[0]), float(p[1])
                w.ego.theta = float(np.arctan2(nxt[1] - p[1], nxt[0] - p[0]))
                w.ego.v = ego_speed
                w.t += dt
                for a in w.actors:
                    if a.alive:
                        a.step(dt, w.t, w)
                step += 1
            for tstep, aid, px, py, lead in preds:
                act = truth.get(aid, {}).get(tstep)
                if act is not None:
                    errs.append(float(np.hypot(px - act[0], py - act[1])))
                    leads.append(lead)
    e = np.array(errs)
    l = np.array(leads)
    return {
        "n": len(e),
        "mean": float(e.mean()),
        "p90": float(np.percentile(e, 90)),
        "at_1s": float(e[np.abs(l - 1.1) < 1e-6].mean()) if (np.abs(l - 1.1) < 1e-6).any() else float("nan"),
        "at_3s": float(e[np.abs(l - 3.0) < 1e-6].mean()) if (np.abs(l - 3.0) < 1e-6).any() else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these scenarios")
    args = ap.parse_args()

    base = score(ConstantVelocityPredictor(), args.seeds, only=args.only)
    print(f"{'config':52s} {'mean':>7s} {'p90':>7s} {'@1.1s':>7s} {'@3.0s':>7s}")
    print("-" * 84)
    print(f"{'constant_velocity (baseline)':52s} {base['mean']:7.2f} {base['p90']:7.2f} "
          f"{base['at_1s']:7.2f} {base['at_3s']:7.2f}")

    print(f"{'social-force, grid WITH actors (current behaviour)':52s}", end=" ", flush=True)
    r = score(SocialForcePredictor(), args.seeds, static_actors=True, only=args.only)
    print(f"{r['mean']:7.2f} {r['p90']:7.2f} {r['at_1s']:7.2f} {r['at_3s']:7.2f}")

    grid = {
        "social_strength": [0.0, 0.6, 2.1],
        "obstacle_strength": [0.0, 1.5, 6.0],
        "relaxation": [0.6, 2.0],
    }
    rows = []
    for ss, os_, rel in itertools.product(*grid.values()):
        p = SocialForcePredictor(social_strength=ss, obstacle_strength=os_, relaxation=rel)
        r = score(p, args.seeds, only=args.only)
        rows.append(((ss, os_, rel), r))
        lbl = f"social={ss:<4} obstacle={os_:<4} relax={rel}"
        delta = r["mean"] - base["mean"]
        flag = "  <-- beats CV" if delta < 0 else ""
        print(f"{lbl:52s} {r['mean']:7.2f} {r['p90']:7.2f} {r['at_1s']:7.2f} "
              f"{r['at_3s']:7.2f}{flag}")

    best = min(rows, key=lambda kv: kv[1]["mean"])
    print(f"\nbest: social={best[0][0]} obstacle={best[0][1]} relax={best[0][2]} "
          f"-> {best[1]['mean']:.2f} m (CV baseline {base['mean']:.2f} m)")
    if best[1]["mean"] >= base["mean"]:
        print("\nNo setting of the social forces beats constant velocity here.")
        print("The suite's actors follow scripted policies and do not react to")
        print("each other, so there is no interaction for an interaction-aware")
        print("model to exploit. That is a limitation of the BENCHMARK, and the")
        print("fix is reactive actors -- not more predictor tuning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
