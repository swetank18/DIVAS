#!/usr/bin/env python3
"""Export reference trajectories for the MATLAB cross-validation.

The point of the MATLAB work is *independence*: a second implementation, in a
second tool, of the two models every published number rests on -- the
kinematic bicycle that the built-in simulator integrates, and the longitudinal
controller that was identified against CARLA tonight. If MATLAB reproduces
them, the models are not an artefact of one codebase; if it does not, one of
the two is wrong and we would rather find out here than on stage.

This script produces the reference. It drives the **real** shipped code --
``divas.sim.world.World.step`` and ``divas.sim.carla_bridge.LongitudinalTracker``
-- not a copy written for the occasion, because validating a copy against
another copy proves nothing.

    python3 scripts/export_for_matlab.py
    matlab -batch "cd('sim/matlab'); validate_against_python"

Writes CSVs into ``sim/matlab/reference/``. They are small and text, so a
reviewer can read them without either tool installed.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path as FsPath

import numpy as np

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))

from divas.sim.carla_bridge import LongitudinalTracker, ResistanceModel
from divas.sim.world import Road, World
from divas.types import EgoState, VehicleParams


def straight_road(length: float = 600.0, half_width: float = 8.0) -> Road:
    """A wide straight. The bicycle export never leaves it, and the road plays
    no part in the dynamics -- it is here because ``World`` requires one."""
    s = np.arange(0.0, length, 2.0)
    centre = np.column_stack([s, np.zeros_like(s)])
    return Road(centerline=centre, left_width=half_width, right_width=half_width)


def command(t: float, params: VehicleParams) -> tuple:
    """The open-loop input sequence, as a pure function of time.

    Deliberately not a controller: MATLAB must be able to generate the exact
    same inputs from ``t`` alone, with no planner, no costmap and no random
    numbers. It has to exercise the parts of the model that are easy to get
    wrong, so it includes

    * a steer reversal, which is where a sign error shows;
    * a step in steer larger than ``max_steer_rate * dt``, so the rate limiter
      actually engages;
    * a step from full braking to full acceleration, so the jerk limiter does;
    * a period of commanded braking at low speed, so the ``v >= 0`` clamp does.

    A smooth sinusoid through the middle of the envelope would agree between
    any two implementations and prove nothing.
    """
    if t < 3.0:
        return params.max_accel, 0.0
    if t < 6.0:
        return 0.5, 0.40                       # step in steer: rate limiter
    if t < 9.0:
        return 0.0, -0.40                      # reversal: sign convention
    if t < 12.0:
        return params.min_accel, 0.10          # full brake, then...
    if t < 15.0:
        return params.max_accel, -0.25         # ...full throttle: jerk limiter
    if t < 18.0:
        return params.min_accel, 0.0           # brake to a stop: v clamp
    return 0.3 * params.max_accel, 0.15 * math.sin(2.0 * math.pi * (t - 18.0) / 6.0)


def export_bicycle(out: FsPath, dt: float, seconds: float) -> int:
    params = VehicleParams()
    world = World(road=straight_road(), ego=EgoState(x=0.0, y=0.0, theta=0.0, v=0.0),
                  params=params)
    rows = []
    n = int(round(seconds / dt))
    for k in range(n):
        t = k * dt
        accel_cmd, steer_cmd = command(t, params)
        # Recorded *before* the step: these are the inputs applied over the
        # interval, and the state is the one they were applied to. MATLAB
        # replays exactly this and compares the state it reaches.
        e = world.ego
        rows.append((t, accel_cmd, steer_cmd, e.x, e.y, e.theta, e.v, e.delta, e.a))
        world.step(dt, accel_cmd, steer_cmd)
    e = world.ego
    rows.append((n * dt, 0.0, 0.0, e.x, e.y, e.theta, e.v, e.delta, e.a))

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "accel_cmd", "steer_cmd", "x", "y", "theta", "v",
                    "delta", "a"])
        for row in rows:
            w.writerow([f"{v:.12g}" for v in row])
    return len(rows)


def export_longitudinal(out: FsPath, dt: float, seconds: float, ref: float) -> int:
    """The identified CARLA plant, driven by the tracker and by the old map.

    Same plant, same reference, two controllers -- which is the comparison the
    calibration measured live (7.88 m/s against 9.01 m/s). MATLAB reproducing
    both curves is what makes tonight's fix independent of this codebase.
    """
    params = VehicleParams()
    resistance = ResistanceModel()
    throttle_gain, brake_gain = 6.47, 3.44

    def simulate(controller):
        v, a_meas, out_rows = 0.0, 0.0, []
        for k in range(int(round(seconds / dt))):
            a_cmd = float(np.clip(1.1 * (ref - v), params.min_accel, params.max_accel))
            throttle, brake = controller(a_cmd, v, a_meas, dt)
            a_applied = throttle * throttle_gain - brake * brake_gain
            v_next = max(0.0, v + (a_applied - resistance(v)) * dt)
            out_rows.append((k * dt, v, a_cmd, throttle, brake))
            a_meas, v = (v_next - v) / dt, v_next
        return out_rows

    tracker = LongitudinalTracker(params, resistance, throttle_gain=throttle_gain,
                                 brake_gain=brake_gain)
    closed = simulate(tracker.update)

    def open_loop(a_cmd, v, a_meas, _dt):
        if a_cmd >= 0.0:
            return float(np.clip(a_cmd / params.max_accel, 0.0, 1.0)), 0.0
        return 0.0, float(np.clip(-a_cmd / abs(params.min_accel), 0.0, 1.0))

    opened = simulate(open_loop)

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "v_closed", "a_cmd_closed", "throttle_closed",
                    "brake_closed", "v_open"])
        for (t, vc, ac, thc, brc), (_t, vo, _a, _th, _br) in zip(closed, opened):
            w.writerow([f"{x:.12g}" for x in (t, vc, ac, thc, brc, vo)])
    return len(closed)


def export_params(out: FsPath, ref: float, dt: float) -> None:
    """Every constant MATLAB needs, so nothing is duplicated by hand.

    A hand-copied wheelbase is the classic way a cross-validation "fails":
    two correct implementations of different vehicles.
    """
    params = VehicleParams()
    resistance = ResistanceModel()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "value"])
        for name, value in (
            ("wheelbase", params.wheelbase),
            ("max_steer", params.max_steer),
            ("max_steer_rate", params.max_steer_rate),
            ("max_accel", params.max_accel),
            ("min_accel", params.min_accel),
            ("max_speed", params.max_speed),
            ("cruise_speed", params.cruise_speed),
            ("max_jerk", params.max_jerk),
            ("resistance_c0", resistance.c0),
            ("resistance_c1", resistance.c1),
            ("resistance_c2", resistance.c2),
            ("throttle_gain", 6.47),
            ("brake_gain", 3.44),
            ("throttle_ki", 0.60),
            ("i_limit", 3.0),
            ("coast_band", 0.15),
            ("stop_speed", 0.2),
            ("speed_reference", ref),
            ("dt", dt),
        ):
            w.writerow([name, f"{float(value):.12g}"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="sim/matlab/reference")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--ref", type=float, default=9.0)
    args = ap.parse_args()

    out = FsPath(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n = export_bicycle(out / "bicycle_reference.csv", args.dt, args.seconds)
    print(f"wrote {out / 'bicycle_reference.csv'}  ({n} rows)")
    n = export_longitudinal(out / "longitudinal_reference.csv", args.dt,
                            args.seconds, args.ref)
    print(f"wrote {out / 'longitudinal_reference.csv'}  ({n} rows)")
    export_params(out / "params.csv", args.ref, args.dt)
    print(f"wrote {out / 'params.csv'}")
    print("\nnow, in MATLAB:\n    cd sim/matlab\n    validate_against_python")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
