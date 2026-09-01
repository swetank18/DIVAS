#!/usr/bin/env python3
"""Side-by-side comparison of two stacks on the SAME scenario and seed.

The single most persuasive artifact available to this project: the baseline
and the proposed stack, run on identical traffic from an identical seed, one
of them hitting something and the other not.  Runs are bit-reproducible
(ADR-008), so the video shows the same thing every time it is generated.

    python3 scripts/make_comparison.py --scenario mixed_traffic \\
        --left baseline_conventional --right cv_pred_fixed_margin

Writes an MP4 (or GIF without ffmpeg) plus a static PNG for a slide.
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
from matplotlib import animation
from matplotlib.patches import Circle as MplCircle, Rectangle

from divas.eval import runner, scenarios
from divas.types import CLASS_EXTENT

CLS_COLOR = {
    "motorcycle": "#d1495b", "autorickshaw": "#edae49", "car": "#00798c",
    "bus": "#003d5b", "truck": "#30638e", "pedestrian": "#8f2d56",
    "animal": "#6a4c93", "bicycle": "#c17817",
}
EGO_OK, EGO_HIT = "#1b998b", "#c1121f"


def _oriented(ax, x, y, theta, hl, hw, color, z=4, alpha=1.0, lw=0.0):
    """Draw a box centred on (x, y) at heading theta."""
    c, s = np.cos(theta), np.sin(theta)
    corner = (x - hl * c + hw * s, y - hl * s - hw * c)
    r = Rectangle(corner, 2 * hl, 2 * hw, angle=np.degrees(theta),
                  color=color, zorder=z, alpha=alpha, lw=lw)
    ax.add_patch(r)
    return r


def outcome(m) -> str:
    if m.success:
        return "REACHED GOAL"
    if m.collision:
        return f"COLLISION - {m.collision_with}"
    return "TIMED OUT"


def find_contrast(args) -> int:
    """Which scenario and seed shows the difference most clearly?

    A demo needs one case where the baseline visibly fails and the proposed
    stack visibly does not.  Guessing wastes rehearsal time, and picking a
    case where both succeed wastes the jury's attention.
    """
    left = next(s for s in runner.ABLATION if s.name == args.left)
    right = next(s for s in runner.ABLATION if s.name == args.right)
    cfg = runner.RunnerConfig()
    rows = []
    for name in scenarios.SCENARIOS:
        sc = scenarios.get(name)
        for seed in range(args.find_seeds):
            a = runner.run(sc, left, seed=seed, cfg=cfg)
            b = runner.run(sc, right, seed=seed, cfg=cfg)
            # Best demo: baseline fails, ours succeeds.
            score = (2 if (not a.success and b.success) else 0) \
                + (1 if a.collision and not b.collision else 0)
            rows.append((score, name, seed, outcome(a), outcome(b),
                         a.progress, b.progress))
            print(f"  {name:22s} seed{seed}  {outcome(a):32s} | {outcome(b)}")
    rows.sort(key=lambda r: (-r[0], -(r[6] - r[5])))
    print("\nBest demo cases (baseline fails, proposed succeeds):")
    for sc_ in rows[:5]:
        if sc_[0] == 0:
            break
        print(f"  --scenario {sc_[1]} --seed {sc_[2]}"
              f"   [{sc_[3]}] vs [{sc_[4]}]")
    if rows[0][0] == 0:
        print("  none found - both stacks behave the same everywhere scanned")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="mixed_traffic", choices=list(scenarios.SCENARIOS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--left", default="baseline_conventional",
                    choices=[s.name for s in runner.ABLATION])
    ap.add_argument("--right", default="cv_pred_fixed_margin",
                    choices=[s.name for s in runner.ABLATION])
    ap.add_argument("--window", type=float, default=55.0, help="metres of road shown")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-video", action="store_true", help="static PNG only")
    ap.add_argument("--find", action="store_true",
                    help="scan scenarios and seeds for the clearest contrast, "
                         "then exit without rendering")
    ap.add_argument("--find-seeds", type=int, default=4)
    args = ap.parse_args()

    if args.find:
        return find_contrast(args)

    sc = scenarios.get(args.scenario)
    cfg = runner.RunnerConfig(record=True)
    runs = []
    for name in (args.left, args.right):
        st = next(s for s in runner.ABLATION if s.name == name)
        m = runner.run(sc, st, seed=args.seed, cfg=cfg)
        runs.append((st, m))
        print(f"{name:32s} {outcome(m):34s} {m.progress:6.1f} m  "
              f"v={m.mean_speed:.1f} m/s  t={m.sim_time:.1f} s")

    world = sc.build(args.seed)
    poly = world.road.polygon()
    stem = args.out or f"compare_{sc.name}_{args.left}_vs_{args.right}"

    # ---------------------------------------------------------------- static
    fig, axes = plt.subplots(2, 1, figsize=(17, 7), sharex=True)
    handles, seen = [], set()
    for ax, (st, m) in zip(axes, runs):
        ego = np.array(m.trace["ego"])
        ax.fill(poly[:, 0], poly[:, 1], color="#ececec", zorder=0)
        for o in world.statics:
            if hasattr(o, "radius"):
                ax.add_patch(MplCircle((o.x, o.y), o.radius, color="#555", zorder=3))
            else:
                _oriented(ax, o.x, o.y, o.theta, o.length / 2, o.width / 2, "#777", z=3)
        # Traffic as it stood at the decisive moment.  Without it the reader
        # sees a red line stopping for no visible reason; with it, the vehicle
        # that was hit is right there.
        for ax_, ay, ath, cls in m.trace["actors"][-1]:
            hl, hw = CLASS_EXTENT.get(cls, (1.0, 0.6))
            _oriented(ax, ax_, ay, ath, hl, hw, CLS_COLOR.get(cls, "#888"),
                      z=4, alpha=0.85)
            if cls not in seen:
                handles.append((cls, CLS_COLOR.get(cls, "#888")))
                seen.add(cls)
        col = EGO_HIT if m.collision else EGO_OK
        ax.plot(ego[:, 0], ego[:, 1], color=col, lw=2.6, zorder=5)
        ax.plot(ego[0, 0], ego[0, 1], "o", color=col, ms=7, zorder=6)
        ehl, ehw = world.params.half_extent
        _oriented(ax, ego[-1, 0], ego[-1, 1], ego[-1, 2], ehl, ehw, col, z=6)
        if m.collision:
            ax.plot(ego[-1, 0], ego[-1, 1], "X", color=EGO_HIT, ms=20, zorder=8,
                    markeredgecolor="white", markeredgewidth=1.5)

        ax.set_ylabel("y [m]")
        ax.set_title(f"{st.name}   —   {outcome(m)}   ({m.progress:.0f} m, "
                     f"{m.mean_speed:.1f} m/s mean)",
                     loc="left", fontsize=11,
                     color=EGO_HIT if m.collision else "#111")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("x [m]")
    # Explicit limits.  Patches added with add_patch push the data limits
    # around, and with an equal aspect that silently rescales the whole
    # figure; a slide artifact needs framing that does not drift.
    span = max(np.array(m.trace["ego"])[:, 0].max() for _, m in runs)
    ymin, ymax = poly[:, 1].min(), poly[:, 1].max()
    for ax in axes:
        ax.set_xlim(-5, span + 20)
        ax.set_ylim(ymin - 2, ymax + 2)
    if handles:
        from matplotlib.lines import Line2D
        axes[0].legend(
            [Line2D([], [], marker="s", ls="", color=c) for _, c in handles],
            [n for n, _ in handles],
            loc="upper left", fontsize=8, ncol=len(handles), framealpha=0.9)
    fig.suptitle(f"{sc.name} — identical traffic, identical seed", fontsize=13)
    fig.savefig(f"{stem}.png", dpi=120, bbox_inches="tight")
    print(f"\nwrote {stem}.png")
    plt.close(fig)
    if args.no_video:
        return 0

    # ---------------------------------------------------------------- video
    n = max(len(m.trace["t"]) for _, m in runs)
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.1, 2.1, 1.0], hspace=0.42)
    panels, arts = [], []
    for row, (st, m) in enumerate(runs):
        ax = fig.add_subplot(gs[row])
        ax.fill(poly[:, 0], poly[:, 1], color="#ececec", zorder=0)
        for o in world.statics:
            if hasattr(o, "radius"):
                ax.add_patch(MplCircle((o.x, o.y), o.radius, color="#555", zorder=3))
            else:
                _oriented(ax, o.x, o.y, o.theta, o.length / 2, o.width / 2, "#777", z=3)
        ax.set_aspect("equal")
        ax.set_ylabel("y [m]")
        ax.grid(alpha=0.2)
        ax.set_title(st.name, loc="left", fontsize=11, fontweight="bold")
        trail, = ax.plot([], [], color=EGO_OK, lw=2.0, zorder=5, alpha=0.85)
        plan, = ax.plot([], [], color="#f4a261", lw=1.6, ls="--", zorder=4)
        banner = ax.text(0.5, 0.90, "", transform=ax.transAxes, ha="center",
                         fontsize=15, fontweight="bold", color=EGO_HIT, zorder=10)
        panels.append(ax)
        arts.append({"trail": trail, "plan": plan, "banner": banner, "boxes": []})

    axv = fig.add_subplot(gs[2])
    for (st, m), c in zip(runs, (EGO_HIT, EGO_OK)):
        axv.plot(m.trace["t"], m.trace["v"], color=c, lw=1.6, label=st.name)
    cursor = axv.axvline(0, color="#333", lw=1.0)
    axv.set_xlabel("t [s]"); axv.set_ylabel("speed [m/s]")
    axv.grid(alpha=0.25); axv.legend(fontsize=8, loc="lower right")

    def draw(k):
        for (st, m), ax, art in zip(runs, panels, arts):
            tr = m.trace
            i = min(k, len(tr["t"]) - 1)
            ex, ey, eth = tr["ego"][i]
            ax.set_xlim(ex - args.window * 0.30, ex + args.window * 0.70)
            ax.set_ylim(ey - 11, ey + 11)
            e = np.array(tr["ego"][: i + 1])
            hit = m.collision and i == len(tr["t"]) - 1
            art["trail"].set_data(e[:, 0], e[:, 1])
            art["trail"].set_color(EGO_HIT if hit else EGO_OK)
            p = tr["path"][i]
            art["plan"].set_data(p[:, 0], p[:, 1]) if p is not None else art["plan"].set_data([], [])
            for b in art["boxes"]:
                b.remove()
            art["boxes"] = []
            for ax_, ay, ath, cls in tr["actors"][i]:
                hl, hw = CLASS_EXTENT.get(cls, (1.0, 0.6))
                art["boxes"].append(_oriented(ax, ax_, ay, ath, hl, hw,
                                              CLS_COLOR.get(cls, "#888")))
            ehl, ehw = world.params.half_extent
            art["boxes"].append(_oriented(ax, ex, ey, eth, ehl, ehw,
                                          EGO_HIT if hit else EGO_OK, z=6))
            art["banner"].set_text(outcome(m) if i == len(tr["t"]) - 1 and not m.success else "")
        cursor.set_xdata([runs[0][1].trace["t"][min(k, len(runs[0][1].trace["t"]) - 1)]] * 2)
        return []

    fig.suptitle(f"{sc.name}  —  identical traffic, identical seed "
                 f"(seed {args.seed})", fontsize=13)
    anim = animation.FuncAnimation(fig, draw, frames=n, interval=1000 // args.fps)
    try:
        anim.save(f"{stem}.mp4", writer=animation.FFMpegWriter(fps=args.fps, bitrate=2400))
        print(f"wrote {stem}.mp4  ({n} frames, {n / args.fps:.1f} s)")
    except Exception as exc:                                   # noqa: BLE001
        print(f"ffmpeg unavailable ({exc}); writing GIF instead")
        anim.save(f"{stem}.gif", writer=animation.PillowWriter(fps=args.fps))
        print(f"wrote {stem}.gif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
