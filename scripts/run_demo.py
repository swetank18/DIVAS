#!/usr/bin/env python3
"""Run one scenario and render it.

    python3 scripts/run_demo.py --scenario mixed_traffic --stack full_dynamic_margin

Writes a PNG showing the corridor, the obstacles, the driven trajectory, the
actors, and the speed/margin traces.  With ``--gif`` it also writes an animation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as FsPath

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from divas.eval import runner, scenarios

CLS_COLOR = {
    "motorcycle": "#d1495b", "autorickshaw": "#edae49", "car": "#00798c",
    "bus": "#003d5b", "truck": "#30638e", "pedestrian": "#8f2d56",
    "animal": "#6a4c93", "bicycle": "#c17817",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="mixed_traffic",
                    choices=list(scenarios.SCENARIOS))
    ap.add_argument("--stack", default="full_dynamic_margin",
                    choices=[s.name for s in runner.ABLATION])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sc = scenarios.get(args.scenario)
    st = next(s for s in runner.ABLATION if s.name == args.stack)
    cfg = runner.RunnerConfig(record=True)
    m = runner.run(sc, st, seed=args.seed, cfg=cfg)

    row = m.as_row()
    print(f"scenario : {sc.name}  --  {sc.description}")
    print(f"tests    : {sc.tests}")
    print(f"stack    : {st.name}  --  {st.description}")
    outcome = ("SUCCESS" if m.success else
               f"COLLISION with {m.collision_with}" if m.collision else "TIMEOUT")
    print(f"outcome  : {outcome}  after {m.sim_time:.1f} s, {m.progress:.1f} m")
    for k in ("mean_speed", "min_ttc", "min_clearance_m", "max_lat_accel",
              "max_jerk", "mean_d_safe_m", "plan_success_rate",
              "predict_ms_p95", "plan_ms_p95", "control_ms_p95", "e2e_ms_p95"):
        print(f"  {k:20s} {row[k]}")

    tr = m.trace
    ego = np.array(tr["ego"])
    world = sc.build(args.seed)
    poly = world.road.polygon()

    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1, 1], hspace=0.45)

    ax = fig.add_subplot(gs[0])
    ax.fill(poly[:, 0], poly[:, 1], color="#e9e9e9", zorder=0, label="drivable area")
    for o in world.statics:
        if hasattr(o, "radius"):
            ax.add_patch(plt.Circle((o.x, o.y), o.radius, color="#444", zorder=3))
        else:
            ax.add_patch(plt.Rectangle(
                (o.x - o.length / 2, o.y - o.width / 2), o.length, o.width,
                angle=np.degrees(o.theta), color="#666", zorder=3))
    seen = set()
    for frame in tr["actors"][::6]:
        for ax_, ay, _, cls in frame:
            ax.plot(ax_, ay, ".", ms=3, alpha=0.45,
                    color=CLS_COLOR.get(cls, "#888"),
                    label=cls if cls not in seen else None)
            seen.add(cls)
    ax.plot(ego[:, 0], ego[:, 1], "-", lw=2.4, color="#1b998b", label="ego", zorder=5)
    ax.plot(ego[0, 0], ego[0, 1], "o", color="#1b998b", ms=8, zorder=6)
    if m.collision:
        ax.plot(ego[-1, 0], ego[-1, 1], "X", color="red", ms=15, zorder=7,
                label=f"collision: {m.collision_with}")
    ax.set_aspect("equal")
    ax.set_xlim(-5, max(ego[:, 0].max() + 25, 40))
    ax.set_title(f"{sc.name} / {st.name} -- {outcome}")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    ax.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(tr["t"], tr["v"], color="#1b998b")
    ax2.axhline(world.params.cruise_speed, ls="--", lw=0.8, color="#888",
                label="cruise target")
    ax2.set_ylabel("speed [m/s]"); ax2.grid(alpha=0.25); ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(tr["t"], tr["d_safe"], color="#d1495b")
    ax3.set_ylabel("mean $d_{safe}$ [m]"); ax3.set_xlabel("t [s]")
    ax3.grid(alpha=0.25)
    ax3.set_title("dynamic safety margin -- widens as prediction confidence falls",
                  fontsize=9, loc="left")

    out = args.out or f"demo_{sc.name}_{st.name}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
