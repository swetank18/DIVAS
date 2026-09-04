#!/usr/bin/env python3
"""Measure the ego's longitudinal plant in CARLA, and fit the pedal model.

Two questions, both of which were open after the first live run measured a
4.56 m/s mean speed against a 9.0 m/s cruise target.

**1. What does coasting actually cost?**  :class:`ResistanceModel` is a
quadratic in speed, ``c0 + c1 v + c2 v^2``, and its defaults were a textbook
guess.  ``--coastdown`` measures it: accelerate to a target, lift off, and
record the deceleration over a short window, at a spread of speeds.  Fitting
those points gives the feedforward term that lets a zero acceleration command
*hold* speed instead of decaying.  Paste the printed line into
:class:`CarlaConfig` per ego blueprint -- a Micra and a Tesla do not share a
drag area.

**2. How much of the speed gap is the pedal map, and how much is the road?**
This is the question that matters for what the deck may claim.  A dense urban
route has traffic, junctions and curvature, and the planner's own reference is
``min(sqrt(a_lat / kappa), cruise)`` -- which is simply *below* cruise for much
of Town10, entirely legitimately.  So this script asks the narrower question on
a straight, empty road, where the reference is cruise and nothing but the
controller can explain a shortfall: hold a constant reference with the old
open-loop map, then with :class:`LongitudinalTracker`, and print both.

Expect the honest answer to be a few tenths of a metre per second.  The
open-loop map has no integral action, so its standing error is
``resistance(v) / (1.1 * authority)``; that is a real defect and it is not
4 m/s.  Anything larger than about 1 m/s here means something else is wrong
and the MPC is worth suspecting after all.

Run the server first, through the wrapper -- launched plainly, Vulkan picks the
integrated GPU on a hybrid laptop and everything below renders at a few frames
a second::

    ./scripts/carla_server.sh
    ~/carla-venv/bin/python3 scripts/calibrate_longitudinal.py --plot

Traffic is off and the vehicle is teleported back to its spawn between
segments, so the whole thing needs one straight road and about a minute.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path as FsPath

import numpy as np

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))

from divas.sim.carla_bridge import (
    HAVE_CARLA,
    CarlaConfig,
    CarlaWorld,
    LongitudinalTracker,
    ResistanceModel,
    carla_to_odom,
    carla_vector_to_odom,
    control_from_command,
    normalised_steer,
)

if HAVE_CARLA:                                      # pragma: no cover
    import carla


# --------------------------------------------------------------------------
# the fit -- pure, and tested without CARLA
# --------------------------------------------------------------------------


def fit_resistance(speeds, decels) -> ResistanceModel:
    """Least-squares fit of ``a = c0 + c1 v + c2 v^2`` to coast-down samples.

    The fit is **unconstrained**, and then validated as a curve.  An earlier
    version clamped ``c0`` and ``c2`` to be non-negative *after* solving the
    joint normal equations, on the reasoning that drag cannot be negative.
    That is invalid: the three coefficients are not independent, so pinning
    one of them moves the curve away from every sample instead of toward the
    nearest physical one.  CARLA's coast-down rises steeply and then flattens
    above about 9 m/s -- engine braking, with the autobox downshifting -- which
    genuinely wants a negative quadratic term.  Clamping it turned a good fit
    into ``1.31 v``, which reads 11.8 m/s^2 at cruise, put that into the
    feedforward, and drove the ego to 22 m/s.

    So the coefficients are taken as solved and the *curve* is checked over
    the speed range that was actually sampled: it must stay positive, because
    a feedforward that goes negative would ask for brake to hold a steady
    speed.  If it does not, the fit falls back to a straight line and then to
    the sample mean -- both worse models, but neither can invert the sign of
    the term they feed.
    """
    v = np.asarray(speeds, dtype=float)
    a = np.asarray(decels, dtype=float)
    if v.size < 3:
        raise ValueError(f"need at least 3 coast-down samples, got {v.size}")

    grid = np.linspace(0.0, float(v.max()), 64)

    def _positive(model) -> bool:
        return bool(np.all(np.array([model(x) for x in grid]) > 0.0))

    for order in (2, 1):
        if v.size <= order:
            continue
        design = np.column_stack([v ** k for k in range(order + 1)])
        coef, *_ = np.linalg.lstsq(design, a, rcond=None)
        padded = list(coef) + [0.0] * (3 - len(coef))
        model = ResistanceModel(c0=float(padded[0]), c1=float(padded[1]),
                                c2=float(padded[2]))
        if _positive(model):
            return model
    return ResistanceModel(c0=float(np.mean(a)), c1=0.0, c2=0.0)


def fit_residual(model: ResistanceModel, speeds, decels) -> float:
    """RMS error of the fit, m/s^2. Quote it: a feedforward term is only worth
    trusting to the accuracy of the curve it came from."""
    pred = np.array([model(v) for v in np.asarray(speeds, dtype=float)])
    return float(np.sqrt(np.mean((pred - np.asarray(decels, dtype=float)) ** 2)))


# --------------------------------------------------------------------------
# the measurements -- these need a server
# --------------------------------------------------------------------------


def forward_speed(vehicle) -> float:
    """Signed forward speed, m/s, in this stack's convention.

    Signed, not ``|v|``: a vehicle rolling backwards must not read as coasting
    slowly forwards, which would bias every resistance sample low.
    """
    tf = vehicle.get_transform()
    _x, _y, theta = carla_to_odom(tf.location.x, tf.location.y, tf.rotation.yaw)
    vel = vehicle.get_velocity()
    vx, vy = carla_vector_to_odom(vel.x, vel.y)
    return float(vx * math.cos(theta) + vy * math.sin(theta))


def on_road(cmap, vehicle) -> bool:
    """Is the ego still on a driving lane?

    The validity guard for every sample below, and it is not optional. The
    first version of this script steered straight ahead on Town04, whose
    highway curves; the ego left the carriageway within seconds and the
    coast-down it then measured -- 1.5 to 3.3 m/s^2 -- was the rolling
    resistance of *grass*, an order of magnitude above the road value it was
    supposed to be measuring. A fit to those samples put 7.4 m/s^2 of
    feedforward into the controller, which floored the throttle and drove the
    car to 26 m/s. A wrong measurement is worse than no measurement, so any
    sample taken off-lane is discarded rather than fitted.
    """
    wp = cmap.get_waypoint(vehicle.get_transform().location,
                           project_to_road=False, lane_type=carla.LaneType.Driving)
    return wp is not None


def lane_keep_steer(cmap, vehicle, params, max_steer_deg, lookahead) -> float:
    """Normalised CARLA steer that follows the lane the ego is in.

    Pure pursuit onto a waypoint ``lookahead`` metres along the lane graph --
    the same geometry as :class:`PurePursuitController`, but reading the target
    straight off CARLA's map rather than needing a planned ``Path``. This is a
    test rig: it exists so the vehicle stays on the road for the length of a
    measurement, not to be a controller.

    The sign conversion goes through :func:`normalised_steer` rather than being
    written out again here. That function has a test pinning its sign, and this
    is exactly the kind of second copy that ends up mirrored.
    """
    tf = vehicle.get_transform()
    wp = cmap.get_waypoint(tf.location, project_to_road=True,
                           lane_type=carla.LaneType.Driving)
    nxt = wp.next(max(lookahead, 1.0))
    if not nxt:
        return 0.0
    target = nxt[0].transform.location
    ex, ey, theta = carla_to_odom(tf.location.x, tf.location.y, tf.rotation.yaw)
    tx, ty, _ = carla_to_odom(target.x, target.y, 0.0)
    dx, dy = tx - ex, ty - ey
    alpha = math.atan2(dy, dx) - theta
    alpha = (alpha + math.pi) % (2.0 * math.pi) - math.pi
    ld = max(math.hypot(dx, dy), 1e-3)
    delta = math.atan2(2.0 * params.wheelbase * math.sin(alpha), ld)
    delta = float(np.clip(delta, -params.max_steer, params.max_steer))
    return normalised_steer(delta, max_steer_deg)


def drive(world, cmap, vehicle, params, max_steer_deg, throttle, brake, ticks):
    """Hold one pedal position for ``ticks`` ticks, keeping to the lane.

    Returns ``(speeds, stayed_on_road)``. Steering is a rig concern only; it
    does not enter any longitudinal number, because a few degrees of steer at
    these speeds costs far less than the fit's own residual.
    """
    speeds, ok = [], True
    for _ in range(ticks):
        steer = lane_keep_steer(cmap, vehicle, params, max_steer_deg,
                                max(0.6 * max(forward_speed(vehicle), 0.0), 4.0))
        vehicle.apply_control(
            carla.VehicleControl(throttle=float(throttle), steer=float(steer),
                                 brake=float(brake))
        )
        world.tick()
        speeds.append(forward_speed(vehicle))
        ok = ok and on_road(cmap, vehicle)
    return speeds, ok


def settle(world, cmap, vehicle, params, max_steer_deg, ticks=40):
    """Brake to a near stop between segments.

    Deliberately *not* a teleport back to the spawn pose. ``set_transform``
    drops the vehicle at the new pose and it arrives scraping, which shows up
    in the following coast-down as extra resistance; and in synchronous mode
    the pose does not apply until the next tick, so the velocity has to be
    zeroed separately or the car turns up at the start point still doing 12
    m/s. Braking along the lane instead costs a second and cannot corrupt the
    measurement.
    """
    drive(world, cmap, vehicle, params, max_steer_deg, 0.0, 1.0, ticks)


def coast_slope(speeds, dt, skip=3) -> float:
    """Coast-down deceleration, m/s^2, as the regression slope of ``v(t)``.

    A slope over the whole window rather than the endpoint difference: one
    noisy sample at either end otherwise sets the value. The first few ticks
    after lift-off are dropped because the gearbox is still settling there,
    and that transient is not the steady resistance the feedforward wants.
    """
    v = np.asarray(speeds[skip:], dtype=float)
    if v.size < 4:
        return float("nan")
    t = np.arange(v.size) * dt
    slope = float(np.polyfit(t, v, 1)[0])
    return -slope


def coastdown(world, cmap, vehicle, params, max_steer_deg, dt, targets,
              window=2.0, timeout=15.0):
    """Resistance samples ``(v, a_resist)``, one per target speed.

    The window is short on purpose. A long coast spans a wide speed range, and
    fitting a curve to samples that each already average over most of that
    range throws away the very variation being measured.
    """
    samples = []
    for target in targets:
        settle(world, cmap, vehicle, params, max_steer_deg)
        elapsed = 0.0
        while forward_speed(vehicle) < target and elapsed < timeout:
            _s, ok = drive(world, cmap, vehicle, params, max_steer_deg, 1.0, 0.0, 1)
            elapsed += dt
            if not ok:
                break
        reached = forward_speed(vehicle)
        if reached < 0.85 * target:
            print(f"  {target:5.1f} m/s  SKIP -- only reached {reached:5.2f} m/s "
                  f"in {timeout:.0f}s")
            continue
        # lift off, both pedals up: this is the measurement
        trace, ok = drive(world, cmap, vehicle, params, max_steer_deg, 0.0, 0.0,
                          max(int(round(window / dt)), 8))
        if not ok:
            print(f"  {target:5.1f} m/s  DISCARD -- left the lane mid-coast")
            continue
        a_resist = coast_slope(trace, dt)
        v_mid = float(np.mean(trace))
        if not np.isfinite(a_resist) or a_resist <= 0.0:
            print(f"  {target:5.1f} m/s  DISCARD -- a_resist = {a_resist}")
            continue
        samples.append((v_mid, a_resist))
        print(f"  {target:5.1f} m/s  coast {trace[0]:5.2f} -> {trace[-1]:5.2f} "
              f"over {window:.1f}s   a_resist = {a_resist:5.3f} m/s^2")
    return samples


def measure_pedal_gain(world, cmap, vehicle, params, max_steer_deg, dt,
                       resistance, pedal, level, entry_speed, window=1.2):
    """Acceleration delivered per unit of pedal, m/s^2.

    ``pedal`` is ``"throttle"`` or ``"brake"``. The vehicle is brought to
    ``entry_speed``, the pedal is held at ``level`` for ``window`` seconds, and
    the *gross* acceleration is the net one plus the resistance the coast-down
    already measured -- resistance is acting throughout, so leaving it out
    would understate throttle authority and overstate brake authority.

    Why this is measured rather than assumed: the original mapping normalised
    by ``params.max_accel`` and ``params.min_accel``, which are the planner's
    comfort limits and have nothing to do with what the actuators can do.
    Returns ``(gain, gross_accel, v_mid)``, or ``None`` if the segment left
    the lane.
    """
    settle(world, cmap, vehicle, params, max_steer_deg)
    elapsed = 0.0
    while forward_speed(vehicle) < entry_speed and elapsed < 20.0:
        _s, ok = drive(world, cmap, vehicle, params, max_steer_deg, 1.0, 0.0, 1)
        elapsed += dt
        if not ok:
            return None
    throttle, brake = (level, 0.0) if pedal == "throttle" else (0.0, level)
    trace, ok = drive(world, cmap, vehicle, params, max_steer_deg,
                      throttle, brake, max(int(round(window / dt)), 6))
    if not ok or len(trace) < 6:
        return None
    t = np.arange(len(trace)) * dt
    net = float(np.polyfit(t, np.asarray(trace, dtype=float), 1)[0])
    v_mid = float(np.mean(trace))
    # Gross of resistance, and signed so both pedals report a positive gain.
    gross = net + resistance(v_mid) if pedal == "throttle" else -net - resistance(v_mid)
    return float(gross / max(level, 1e-3)), float(gross), v_mid


def speed_hold(world, cmap, vehicle, params, max_steer_deg, dt, controller,
               ref, seconds):
    """Track a constant speed reference along the lane; return the speed trace.

    ``controller`` is anything with ``update(a_cmd, v, a_meas, dt) ->
    (throttle, brake)``. The command is the same proportional law stage 6
    emits, so the comparison isolates the pedal mapping and nothing else --
    same reference, same road, same gain.
    """
    settle(world, cmap, vehicle, params, max_steer_deg)
    v, a_meas, trace = forward_speed(vehicle), 0.0, []
    for _ in range(int(round(seconds / dt))):
        a_cmd = float(np.clip(1.1 * (ref - v), params.min_accel, params.max_accel))
        throttle, brake = controller.update(a_cmd, v, a_meas, dt)
        steer = lane_keep_steer(cmap, vehicle, params, max_steer_deg,
                                max(0.6 * max(v, 0.0), 4.0))
        vehicle.apply_control(
            carla.VehicleControl(throttle=float(throttle), steer=float(steer),
                                 brake=float(brake))
        )
        world.tick()
        v_next = forward_speed(vehicle)
        a_meas, v = (v_next - v) / dt, v_next
        trace.append(v)
    return trace


class OpenLoopPedals:
    """The longitudinal half of :func:`control_from_command`, as the baseline.

    It is the mapping the first live run used, and the only reason it still
    runs is to put a number on the difference.
    """

    def __init__(self, params):
        self.params = params

    def update(self, a_cmd, v, a_meas, dt):
        throttle, _steer, brake = control_from_command(a_cmd, 0.0, self.params, 70.0)
        return throttle, brake


# --------------------------------------------------------------------------


def plot(path, samples, model, traces, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    if samples:
        v = np.array([s[0] for s in samples])
        a = np.array([s[1] for s in samples])
        grid = np.linspace(0, max(v.max() * 1.1, 1.0), 100)
        ax0.plot(grid, [model(x) for x in grid], "-", lw=2,
                 label=f"fit  {model.c0:.3f} + {model.c1:.4f}v + {model.c2:.5f}v²")
        ax0.plot(v, a, "o", ms=7, label="coast-down samples")
    ax0.set_xlabel("speed, m/s")
    ax0.set_ylabel("coast-down deceleration, m/s²")
    ax0.set_title("Measured longitudinal resistance")
    ax0.grid(alpha=0.3)
    ax0.legend(fontsize=8)

    ax1.axhline(ref, ls="--", c="k", lw=1, label=f"reference {ref:.1f} m/s")
    for name, trace in traces.items():
        ax1.plot(np.arange(len(trace)) * 0.05, trace, lw=1.8, label=name)
    ax1.set_xlabel("time, s")
    ax1.set_ylabel("speed, m/s")
    ax1.set_title("Speed hold on a straight, empty road")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nwrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default=None,
                    help="default: whatever is loaded, which avoids a map load "
                         "entirely. Town04's long highway is the ideal road for "
                         "this and it segfaulted the server on a 6 GB card -- "
                         "the large maps do not fit, so the measurement is taken "
                         "on Town10HD_Opt with lane keeping instead")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="client timeout, s. A map load on a laptop GPU takes "
                         "well over the 30 s default")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--blueprint", default="vehicle.nissan.micra")
    ap.add_argument("--targets", type=float, nargs="*",
                    default=[3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
                    help="speeds, m/s, to take a coast-down sample at")
    ap.add_argument("--window", type=float, default=2.0, help="coast window, s")
    ap.add_argument("--ref", type=float, default=9.0, help="speed-hold reference, m/s")
    ap.add_argument("--ki", type=float, default=0.60, help="integral trim gain, 1/s")
    ap.add_argument("--hold", type=float, default=25.0, help="speed-hold duration, s")
    ap.add_argument("--out", default="docs/longitudinal-calibration.json")
    ap.add_argument("--plot", nargs="?",
                    const="docs/longitudinal-calibration.png", default=None)
    args = ap.parse_args()

    if not HAVE_CARLA:
        print("The carla Python package is not installed.\n"
              "  ~/carla-venv/bin/python3 scripts/calibrate_longitudinal.py",
              file=sys.stderr)
        return 2

    cfg = CarlaConfig(
        host=args.host, port=args.port, town=args.town, timeout=args.timeout,
        fixed_delta_seconds=args.dt, ego_blueprint=args.blueprint,
        n_vehicles=0, n_walkers=0,          # the road must be empty to mean anything
        weather="clear_noon", seed=1, sensors=(), render=False,
    )
    world = CarlaWorld(cfg)
    try:
        vehicle, cworld, params = world.vehicle, world.world, world.params
        cmap, max_steer_deg = world.map, world._max_steer_deg
        print(f"map {world.map.name}   ego {args.blueprint}   "
              f"{params.length:.2f} x {params.width:.2f} m\n")

        print("coast-down:")
        samples = coastdown(cworld, cmap, vehicle, params, max_steer_deg,
                            args.dt, args.targets, args.window)
        if len(samples) < 3:
            print("\nnot enough clean segments to fit -- try a map with a longer "
                  "straight (--town Town05), or lower --targets", file=sys.stderr)
            return 1
        model = fit_resistance([s[0] for s in samples], [s[1] for s in samples])
        residual = fit_residual(model, [s[0] for s in samples], [s[1] for s in samples])

        print(f"\nfit   ResistanceModel(c0={model.c0:.4f}, c1={model.c1:.5f}, "
              f"c2={model.c2:.6f})")
        print(f"      RMS residual {residual:.4f} m/s^2 over "
              f"{len(samples)} samples")
        print(f"      at {args.ref:.0f} m/s coasting costs {model(args.ref):.3f} m/s^2 "
              f"-- that is what the old open-loop map left uncompensated")

        print("\npedal authority (gross of resistance):")
        gains = {}
        for pedal, level, entry in (("throttle", 0.5, 3.0), ("throttle", 1.0, 3.0),
                                    ("brake", 1.0, 10.0)):
            got = measure_pedal_gain(cworld, cmap, vehicle, params, max_steer_deg,
                                     args.dt, model, pedal, level, entry)
            if got is None:
                print(f"  {pedal:8s} @ {level:.1f}   DISCARD -- left the lane")
                continue
            gain, gross, v_mid = got
            gains.setdefault(pedal, []).append(gain)
            print(f"  {pedal:8s} @ {level:.1f}   {gross:6.2f} m/s^2 gross at "
                  f"{v_mid:5.2f} m/s   -> gain {gain:5.2f} m/s^2 per unit")

        throttle_gain = float(np.mean(gains.get("throttle", [5.5])))
        brake_gain = float(np.mean(gains.get("brake", [8.0])))
        print(f"\n  throttle_gain = {throttle_gain:.2f}   "
              f"brake_gain = {brake_gain:.2f}")
        print(f"  for comparison the old mapping used params.max_accel = "
              f"{params.max_accel:.1f} and |min_accel| = {abs(params.min_accel):.1f},")
        print("  which are the planner's comfort limits, not the vehicle's.")

        print(f"\nspeed hold, {args.ref:.1f} m/s reference, {args.hold:.0f}s, "
              f"straight and empty:")
        traces = {}
        for name, ctrl in (
            ("open loop (old)", OpenLoopPedals(params)),
            ("closed loop (new)", LongitudinalTracker(
                params, model, ki=args.ki, throttle_gain=throttle_gain,
                brake_gain=brake_gain)),
        ):
            trace = speed_hold(cworld, cmap, vehicle, params, max_steer_deg,
                               args.dt, ctrl, args.ref, args.hold)
            traces[name] = trace
            tail = trace[-int(round(5.0 / args.dt)):]        # last 5 s = settled
            print(f"  {name:20s} settled {np.mean(tail):6.2f} m/s   "
                  f"error {args.ref - np.mean(tail):+5.2f}   "
                  f"peak {max(trace):5.2f}")

        gap = float(np.mean(traces["closed loop (new)"][-100:])
                    - np.mean(traces["open loop (old)"][-100:]))
        print(f"\nthe pedal map is worth {gap:+.2f} m/s on a straight empty road.")
        print("Whatever remains in a town route is traffic, junctions and a\n"
              "curvature-limited reference, which are legitimate -- not the\n"
              "controller. Split the two before claiming either in the deck.")

        print("\npaste into CarlaConfig for this blueprint:\n")
        print(f"    resistance=ResistanceModel(c0={model.c0:.4f}, "
              f"c1={model.c1:.5f}, c2={model.c2:.6f}),")
        print(f"    throttle_gain={throttle_gain:.2f},")
        print(f"    brake_gain={brake_gain:.2f},")

        payload = {
            "map": world.map.name,
            "blueprint": args.blueprint,
            "samples": [{"v": v, "a_resist": a} for v, a in samples],
            "fit": {"c0": model.c0, "c1": model.c1, "c2": model.c2,
                    "rms_residual": residual},
            "pedal_gains": {"throttle": throttle_gain, "brake": brake_gain,
                            "planner_comfort_limits": {
                                "max_accel": params.max_accel,
                                "min_accel": params.min_accel}},
            "speed_hold": {
                "reference": args.ref,
                **{name: {"settled": float(np.mean(t[-100:])), "peak": float(max(t))}
                   for name, t in traces.items()},
                "pedal_map_worth_m_s": gap,
            },
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.out}")
        if args.plot:
            plot(args.plot, samples, model, traces, args.ref)
        return 0
    finally:
        world.close()


if __name__ == "__main__":
    raise SystemExit(main())
