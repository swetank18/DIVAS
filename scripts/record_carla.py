#!/usr/bin/env python3
"""Record a CARLA episode as a jury video: chase camera beside the stack's view.

The 2-D comparison videos in ``demo/`` already make the algorithmic argument
-- baseline hits the pedestrian, ours does not, same seed. What they cannot
show is that the same six stages drive a full vehicle dynamics model in a
rendered town, which is the claim ADR-005 and the stage contracts exist to
make. This writes that: a chase camera on the left, and on the right the same
frame as the stack sees it -- drivable area, the planned path, tracked actors,
and the ego's own footprint.

Both panels come from **one** episode, so they cannot disagree. The left is
what CARLA rendered; the right is the trace the runner recorded while driving
it.

    ./scripts/carla_server.sh
    ~/carla-venv/bin/python3 scripts/record_carla.py --stack cv_pred_fixed_margin

Rendering is on, because a camera is attached -- ``no_rendering_mode`` returns
empty frames, which looks exactly like a sensor that failed to attach, and the
bridge overrides it whenever a sensor is requested. Expect this to run slower
than an evaluation batch; that is the cost of pixels and it does not affect
any number, because a synchronous server waits for our tick.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path as FsPath
from typing import List

import numpy as np

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))

from divas.eval import runner
from divas.eval.scenarios import Scenario
from divas.sim.carla_bridge import (
    HAVE_CARLA,
    WEATHER_PRESETS,
    CarlaConfig,
    CarlaWorld,
)
from divas.types import CLASS_EXTENT

CLS_COLOR = {"car": "#4a6fa5", "truck": "#3d5a80", "bus": "#3d5a80",
             "motorcycle": "#e07a5f", "bicycle": "#e07a5f",
             "pedestrian": "#c1121f", "unknown": "#888888"}
EGO_OK, EGO_HIT = "#2a9d8f", "#c1121f"


class ChaseRecorder:
    """A :class:`SimWorld` that also saves the chase frame on every step.

    A wrapper rather than a change to ``runner.run``: the runner's loop is the
    thing every published number comes out of, and adding a video hook to it
    would put frame-writing latency inside the timers it reports. Here the
    cost lands outside those timers and the episode is otherwise identical.

    Everything not defined here delegates, so the wrapper satisfies the
    protocol for free and cannot drift when the protocol grows.
    """

    def __init__(self, world: CarlaWorld, out_dir: FsPath, every: int = 1) -> None:
        self._world = world
        self._dir = out_dir
        self._every = max(int(every), 1)
        self._n = 0
        self.frames_written = 0
        #: One lane_context per captured frame, aligned with the runner's
        #: trace. Telemetry only: the stack does not model lanes, so this
        #: measures what the free-space planner did rather than informing it.
        self.lane_log: List[dict] = []
        out_dir.mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name):
        return getattr(self._world, name)

    def step(self, dt: float, accel: float, steer: float) -> None:
        self._world.step(dt, accel, steer)
        if self._n % self._every == 0:
            self._save()
            try:
                self.lane_log.append(self._world.lane_context())
            except Exception:
                # Off the mapped network entirely. One missing telemetry
                # sample must not end a three minute recording, so the frame
                # is kept and the sample is marked absent.
                self.lane_log.append({})
        self._n += 1

    def _save(self) -> None:
        img = self._world.frames.get("chase")
        if img is None:
            return
        # CARLA hands over BGRA bytes. Alpha is always 255 and the channel
        # order is not RGB, so both have to be dealt with before anything
        # displays it -- a forgotten swap here is the classic "why is the sky
        # orange" bug.
        buf = np.frombuffer(img.raw_data, dtype=np.uint8)
        buf = buf.reshape((img.height, img.width, 4))[:, :, :3][:, :, ::-1]
        try:
            import imageio.v2 as imageio
            imageio.imwrite(self._dir / f"cam_{self.frames_written:05d}.jpg", buf)
        except ImportError:
            import matplotlib.pyplot as plt
            plt.imsave(self._dir / f"cam_{self.frames_written:05d}.jpg", buf)
        self.frames_written += 1


def lane_events(lane_log):
    """Frame indices at which the ego changed lane, and junction frames.

    A lane change is a change of the ``(road_id, lane_id)`` pair. Both are
    needed: OpenDRIVE lane ids are only unique within a road, and the sign
    encodes the side of the carriageway, so ``lane_id`` alone changes at every
    road boundary.

    Junction frames are excluded and counted separately. Lane ids inside a
    junction describe connecting roads rather than lanes of a carriageway, and
    every junction traversal would otherwise read as a burst of lane changes --
    which would inflate the headline number by roughly the number of turns.

    Note what this measures. The planner has no concept of a lane; it plans
    over free space. These are lane changes *observed* in the trajectory, not
    manoeuvres the planner selected, and the deck should say it that way.
    """
    changes, junctions = [], []
    prev = None
    for i, ctx in enumerate(lane_log):
        if not ctx:
            continue
        if ctx.get("is_junction"):
            junctions.append(i)
            prev = None          # do not compare across a junction
            continue
        key = (ctx["road_id"], ctx["lane_id"])
        if prev is not None and key != prev:
            changes.append(i)
        prev = key
    # Collapse junction frames into traversals: consecutive runs are one.
    traversals = sum(1 for a, b in zip([-9] + junctions, junctions) if b - a > 1)
    return changes, traversals


def render(trace, world, cam_dir: FsPath, out: FsPath, metrics, fps: int,
           window: float, title: str, lane_log=None) -> None:
    """Composite the camera frames with the stack's own view, and encode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    cams = sorted(cam_dir.glob("cam_*.jpg"))
    n = min(len(cams), len(trace["t"]))
    if n == 0:
        raise RuntimeError("no frames captured -- was a chase camera attached?")

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle as MplCircle

    raster = world._raster
    ny, nx = raster.mask.shape
    extent = (raster.origin[0], raster.origin[0] + nx * raster.resolution,
              raster.origin[1], raster.origin[1] + ny * raster.resolution)
    # Two flat colours rather than a ramp: this raster is boolean, and a
    # colormap that interpolates invites the viewer to read a confidence into
    # it that is not there. Off-road is warm and light, drivable is neutral
    # grey, so the ego and the plan are the only saturated things on the panel.
    road_cmap = ListedColormap(["#f2ede6", "#c8c8c8"])

    frame_dir = cam_dir / "composite"
    frame_dir.mkdir(exist_ok=True)

    # Width-to-height of the right-hand axes, from the gridspec below. Set the
    # y limits from it rather than guessing: with ``aspect="equal"`` and both
    # limits given, matplotlib honours the aspect by shrinking the box, and
    # the panel then floats in white space instead of filling.
    PANEL_ASPECT = 1.36
    LEGEND = []

    ehl, ehw = world.params.half_extent
    changes, n_junction_traversals = lane_events(lane_log or [])
    junction_frames = [i for i, c in enumerate(lane_log or [])
                       if c and c.get("is_junction")]
    if lane_log:
        print(f"  {len(changes)} lane changes, "
              f"{n_junction_traversals} junction traversals observed")

    def box(ax, x, y, theta, hl, hw, color, z=4):
        c, s = np.cos(theta), np.sin(theta)
        pts = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        rot = pts @ np.array([[c, s], [-s, c]]) + np.array([x, y])
        patch = MplPolygon(rot, closed=True, color=color, zorder=z)
        ax.add_patch(patch)
        return patch

    for k in range(n):
        fig = plt.figure(figsize=(16, 5.4), dpi=120)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.06,
                              left=0.01, right=0.99, top=0.90, bottom=0.06)

        axc = fig.add_subplot(gs[0])
        axc.imshow(plt.imread(cams[k]))
        axc.set_xticks([]); axc.set_yticks([])
        axc.set_title("CARLA 0.9.16 — Town10HD, full vehicle dynamics",
                      fontsize=10, loc="left")

        ax = fig.add_subplot(gs[1])
        ax.imshow(raster.mask, origin="lower", extent=extent,
                  cmap=road_cmap, vmin=0, vmax=1, zorder=0, interpolation="nearest")
        ex, ey, eth = trace["ego"][k]
        # Lead the ego rather than centre it: the interesting half of the
        # frame is the half it is driving into.
        cx = ex + 0.15 * window * np.cos(eth)
        cy = ey + 0.15 * window * np.sin(eth)
        half_h = 0.5 * window / PANEL_ASPECT
        ax.set_xlim(cx - window * 0.5, cx + window * 0.5)
        ax.set_ylim(cy - half_h, cy + half_h)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("what the stack sees — free space, plan, tracks, safety margin",
                     fontsize=10, loc="left")

        # The margin, drawn where it acts. d_safe is the whole argument of the
        # project -- a buffer that widens when the predictor is unsure -- and a
        # number in a HUD does not show it. One halo per actor, at the radius
        # the risk field is actually holding this step.
        d_safe = float(trace["d_safe"][k])
        for ax_, ay, ath, cls in trace["actors"][k]:
            hl, hw = CLASS_EXTENT.get(cls, (1.0, 0.6))
            if d_safe > 0.0:
                ax.add_patch(MplCircle((ax_, ay), hl + d_safe, color="#e9c46a",
                                       alpha=0.30, zorder=2, lw=0))
        trail = np.array(trace["ego"][: k + 1])
        ax.plot(trail[:, 0], trail[:, 1], color=EGO_OK, lw=2.0, alpha=0.85, zorder=5)
        p = trace["path"][k]
        if p is not None:
            ax.plot(p[:, 0], p[:, 1], color="#e76f51", lw=2.2, ls="--", zorder=6)
        for ax_, ay, ath, cls in trace["actors"][k]:
            hl, hw = CLASS_EXTENT.get(cls, (1.0, 0.6))
            box(ax, ax_, ay, ath, hl, hw, CLS_COLOR.get(cls, "#888"), z=4)
        box(ax, ex, ey, eth, ehl, ehw, EGO_OK, z=7)

        if k == 0:
            handles = [
                MplPolygon([[0, 0]], color=EGO_OK, label="ego"),
                MplPolygon([[0, 0]], color=CLS_COLOR["car"], label="tracked vehicle"),
                MplPolygon([[0, 0]], color=CLS_COLOR["pedestrian"], label="pedestrian"),
                MplPolygon([[0, 0]], color="#e9c46a", alpha=0.5, label="d_safe margin"),
            ]
            LEGEND.extend(handles)
        ax.legend(handles=LEGEND, loc="lower left", fontsize=7, framealpha=0.9,
                  ncol=2)
        # A scale bar: without one, "it cleared the auto" has no size.
        bx = cx + window * 0.44 - 10.0
        by = cy - half_h * 0.88
        ax.plot([bx, bx + 10.0], [by, by], color="#333", lw=2.5, zorder=9)
        ax.text(bx + 5.0, by + half_h * 0.04, "10 m", fontsize=7, ha="center",
                color="#333", zorder=9)

        hud = (f"t {trace['t'][k]:5.1f}s   v {trace['v'][k]:4.1f} m/s   "
               f"driven {trace['progress'][k]:6.1f} m   "
               f"d_safe {trace['d_safe'][k]:.2f} m")
        if lane_log:
            done_changes = sum(1 for c in changes if c <= k)
            # Traversals, not frames: a junction takes a couple of seconds
            # to cross, so counting frames reported 33 junctions in 88 m.
            done_junctions = sum(
                1 for a, b in zip([-9] + junction_frames, junction_frames)
                if b - a > 1 and b <= k
            )
            hud += f"   lane changes {done_changes}   junctions {done_junctions}"
            ctx = lane_log[k] if k < len(lane_log) else {}
            if ctx.get("is_junction"):
                ax.text(0.5, 0.95, "JUNCTION", transform=ax.transAxes,
                        ha="center", va="top", fontsize=11, fontweight="bold",
                        color="#264653", zorder=10)
            # Hold the banner for a beat either side so it is readable at speed.
            if any(abs(k - c) <= 8 for c in changes):
                ax.text(0.5, 0.87, "LANE CHANGE", transform=ax.transAxes,
                        ha="center", va="top", fontsize=12, fontweight="bold",
                        color="#e76f51", zorder=10)
        fig.text(0.01, 0.955, title, fontsize=12, fontweight="bold", va="bottom")
        fig.text(0.99, 0.955, hud, fontsize=10, family="monospace",
                 va="bottom", ha="right", color="#333")
        fig.savefig(frame_dir / f"f_{k:05d}.png", facecolor="white")
        plt.close(fig)
        if k % 40 == 0:
            print(f"  composited {k}/{n}")

    if shutil.which("ffmpeg") is None:
        print(f"ffmpeg not found; {n} PNG frames left in {frame_dir}")
        return
    # Scaled and CRF-tuned deliberately. Straight from 1920-wide PNGs at
    # CRF 20 this comes out near 40 MB, which is several times the size of the
    # whole repository and is not a sensible thing to commit or to email to a
    # jury. 1600 wide at CRF 26 is ~5 MB and indistinguishable on a projector.
    # ``-vf scale`` keeps the height even (-2), which yuv420p requires.
    cmd = ["ffmpeg", "-y", "-framerate", str(fps),
           "-i", str(frame_dir / "f_%05d.png"),
           "-vf", "scale=1600:-2",
           "-c:v", "libx264", "-preset", "slow", "-crf", "26",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    print(f"\nwrote {out}  ({n} frames, {n / fps:.1f} s)")
    still = out.with_suffix(".png")
    shutil.copy(frame_dir / f"f_{max(n - 1, 0):05d}.png", still)
    print(f"wrote {still}  (final frame, for a slide)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default=None)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--vehicles", type=int, default=20)
    ap.add_argument("--walkers", type=int, default=10)
    ap.add_argument("--weather", default="clear_noon", choices=sorted(WEATHER_PRESETS))
    ap.add_argument("--stack", default="cv_pred_fixed_margin",
                    choices=[s.name for s in runner.ABLATION])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=35.0)
    ap.add_argument("--goal", type=float, default=200.0)
    ap.add_argument("--route-length", type=float, default=400.0)
    ap.add_argument("--long-route", action="store_true",
                    help="let the route cross itself, for drives longer than "
                         "the town. Town10HD is ~400x230 m, so anything past "
                         "about a kilometre has to be a circuit; this also "
                         "switches the route to a windowed progress search, "
                         "without which a self-crossing route measures nonsense")
    ap.add_argument("--fps", type=int, default=20, help="20 = real time at dt=0.05")
    ap.add_argument("--every", type=int, default=1, help="capture every Nth step")
    ap.add_argument("--window", type=float, default=54.0, help="bird's-eye width, m")
    ap.add_argument("--out", default="demo/carla_town10.mp4")
    ap.add_argument("--frames-dir", default=None,
                    help="where to keep intermediate frames (default: a temp dir)")
    args = ap.parse_args()

    if not HAVE_CARLA:
        print("The carla Python package is not installed.\n"
              "  ~/carla-venv/bin/python3 scripts/record_carla.py", file=sys.stderr)
        return 2

    cfg = CarlaConfig(
        host=args.host, port=args.port, town=args.town, timeout=args.timeout,
        fixed_delta_seconds=args.dt, n_vehicles=args.vehicles,
        n_walkers=args.walkers, weather=args.weather, seed=args.seed,
        route_length=args.route_length,
        long_route=args.long_route,
        sensors=("chase",),          # forces rendering back on -- see the bridge
        render=True,
    )
    frames_dir = FsPath(args.frames_dir) if args.frames_dir else \
        FsPath("/tmp/divas_record") / f"{args.stack}_{args.seed}"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    stack = next(s for s in runner.ABLATION if s.name == args.stack)
    world = CarlaWorld(cfg)
    recorder = ChaseRecorder(world, frames_dir, every=args.every)
    scenario = Scenario(
        name=f"carla:{world.map.name}:{args.weather}",
        description=f"CARLA closed loop, {args.vehicles} vehicles, "
                    f"{args.walkers} pedestrians",
        build=lambda _seed: recorder,
        goal_progress=args.goal,
        time_limit=args.time_limit,
    )
    rcfg = runner.RunnerConfig(sim_dt=args.dt, record=True,
                               record_every=args.every)
    print(f"recording {args.stack}, seed {args.seed}, "
          f"{args.time_limit:.0f}s limit, route {world.road.length:.0f} m"
          f"{' (windowed)' if world.road.windowed else ''} ...")
    m = runner.run(scenario, stack, seed=args.seed, cfg=rcfg)
    verdict = "SUCCESS" if m.success else ("COLLISION" if m.collision else "time out")
    print(f"  {verdict}   progress {m.progress:.1f} m   mean v {m.mean_speed:.2f} m/s   "
          f"min clearance {m.min_clearance}   e2e p95 {m.end_to_end_p95:.1f} ms")
    if m.collision:
        # *What* was hit decides whose fault it is. CARLA's traffic manager
        # will drive into a stopped ego, and a rear-end from behind is not a
        # planner failure -- but it is scored as a collision either way, so
        # the label has to be visible rather than inferred from the video.
        print(f"  hit: {m.collision}")
    print(f"  {recorder.frames_written} camera frames")

    out = FsPath(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    title = (f"DIVAS — {args.stack}  ·  free-space planning with "
             f"prediction-aware risk")
    render(m.trace, world, frames_dir, out, m, args.fps, args.window, title,
           lane_log=recorder.lane_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
