#!/usr/bin/env python3
"""Drive the DIVAS stack closed-loop in CARLA.

The same runner, the same metrics and the same ablation arms as
``run_ablation.py`` -- only the simulator changes. That is the claim ADR-005
and :mod:`divas.sim.interface` were written to make, and this script is where
it either holds or does not.

Start the server first, in another terminal::

    ~/carla/CarlaUE4.sh -quality-level=Low -RenderOffScreen

then::

    python3 scripts/run_carla.py --check                    # connect, self-test, leave
    python3 scripts/run_carla.py --stack cv_pred_fixed_margin --seed 1
    python3 scripts/run_carla.py --weather hard_rain --sensors semantic rgb
    python3 scripts/run_carla.py --all-weather --seeds 3    # the robustness table

``--check`` is the one to run first on a new machine: it connects, spawns,
verifies the ``SimWorld`` protocol, ticks once and cleans up, in about ten
seconds. If that fails, nothing below it can work.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path as FsPath

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))

from divas.eval import runner
from divas.eval.metrics import aggregate
from divas.eval.scenarios import Scenario
from divas.sim.carla_bridge import (
    HAVE_CARLA,
    WEATHER_PRESETS,
    CarlaConfig,
    CarlaWorld,
    default_sensor_rig,
)


def build_config(args, seed: int, weather: str) -> CarlaConfig:
    return CarlaConfig(
        host=args.host,
        port=args.port,
        town=args.town,
        fixed_delta_seconds=args.dt,
        n_vehicles=args.vehicles,
        n_walkers=args.walkers,
        weather=weather,
        seed=seed,
        tm_port=args.tm_port,
        route_length=args.route_length,
        long_route=args.long_route,
        obey_traffic_lights=not args.ignore_lights,
        sensors=tuple(args.sensors),
        render=args.render,
    )


def one_run(args, stack, seed: int, weather: str):
    """One closed-loop episode. The world is built here and destroyed by the
    runner's ``finally`` -- see the leak note in :class:`CarlaWorld`."""
    world = CarlaWorld(build_config(args, seed, weather))
    scenario = Scenario(
        name=f"carla:{world.map.name}:{weather}",
        description=f"CARLA closed loop, {args.vehicles} vehicles, "
                    f"{args.walkers} pedestrians, {weather}",
        build=lambda _seed: world,
        goal_progress=args.goal,
        time_limit=args.time_limit,
    )
    # sim_dt must equal the server's fixed delta: CarlaWorld.step refuses any
    # dt that is not a whole number of ticks rather than silently rounding it.
    cfg = runner.RunnerConfig(sim_dt=args.dt)
    return runner.run(scenario, stack, seed=seed, cfg=cfg)


def check(args) -> int:
    """Connect, spawn, verify the protocol, tick once, clean up."""
    world = CarlaWorld(build_config(args, args.seed, args.weather))
    try:
        missing = world.self_test()
        if missing:
            print(f"FAIL: CarlaWorld is missing {missing} from the SimWorld protocol")
            return 1
        static, full = world.ground_truth_grids()
        tracks = world.ground_truth_tracks()
        world.step(args.dt, 0.0, 0.0)
        print(f"  map              {world.map.name}")
        print(f"  ego              {world.params.length:.2f} x {world.params.width:.2f} m, "
              f"wheelbase {world.params.wheelbase:.2f} m, "
              f"lock {world._max_steer_deg:.0f} deg")
        print(f"  route            {world.road.length:.0f} m")
        print(f"  drivable raster  {world._raster.mask.shape} cells "
              f"@ {world._raster.resolution} m")
        print(f"  static obstacles {len(world._static_boxes)} baked map meshes, "
              f"{world._carved_cells} cells carved out of free space")
        if world._static_boxes and world._carved_cells < 20:
            print("                   (few cells: this town parks in bays, off the "
                  "driving lanes -- expected)")
        print(f"  costmap          {full.data.shape}, "
              f"{100 * float(full.occupied_mask().mean()):.1f}% occupied "
              f"(static {100 * float(static.occupied_mask().mean()):.1f}%)")
        print(f"  tracks in range  {len(tracks)}")
        print(f"  sensors          {sorted(world.frames) or 'none attached'}")
        print(f"  clearance / TTC  {world.clearance_to_actors():.2f} m / "
              f"{world.time_to_collision():.1f} s")
        print("\nOK -- the bridge is live and satisfies SimWorld.")
        return 0
    finally:
        world.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--town", default=None, help="e.g. Town10HD_Opt; default: loaded map")
    ap.add_argument("--dt", type=float, default=0.05, help="server tick, s (20 Hz)")
    ap.add_argument("--vehicles", type=int, default=40)
    ap.add_argument("--walkers", type=int, default=25)
    ap.add_argument("--weather", default="clear_noon", choices=sorted(WEATHER_PRESETS))
    ap.add_argument("--all-weather", action="store_true",
                    help="sweep every preset -- the rain/fog/dust robustness table")
    ap.add_argument("--sensors", nargs="*", default=[],
                    choices=sorted(default_sensor_rig()), help="Phase 2 rig")
    ap.add_argument("--render", action="store_true",
                    help="render the server window; much slower, needed for video")
    ap.add_argument("--stack", default="full_dynamic_margin",
                    choices=[s.name for s in runner.ABLATION])
    ap.add_argument("--all-stacks", action="store_true", help="run the whole ablation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1, help="seeds per arm, from --seed")
    ap.add_argument("--route-length", type=float, default=400.0,
                    help="metres of lane graph to follow from the spawn point")
    ap.add_argument("--long-route", action="store_true",
                    help="let the route cross itself, and measure progress with "
                         "a windowed search. Required past about a kilometre: "
                         "Town10HD is ~400x230 m, and a global nearest-point "
                         "search on a self-crossing route reports nonsense")
    ap.add_argument("--ignore-lights", action="store_true",
                    help="do not stop at reds (reproduces runs measured before "
                         "red-light keep-outs existed)")
    ap.add_argument("--goal", type=float, default=200.0, help="route metres for success")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--out", default="carla_results.json")
    ap.add_argument("--check", action="store_true", help="connect, self-test, exit")
    args = ap.parse_args()

    if not HAVE_CARLA:
        print("The carla Python package is not installed.\n"
              "  pip install carla==0.9.16   # must match the server build exactly",
              file=sys.stderr)
        return 2

    if args.check:
        return check(args)

    stacks = (runner.ABLATION if args.all_stacks
              else [s for s in runner.ABLATION if s.name == args.stack])
    weathers = sorted(WEATHER_PRESETS) if args.all_weather else [args.weather]
    seeds = list(range(args.seed, args.seed + args.seeds))

    print(f"{len(stacks) * len(weathers) * len(seeds)} episodes: "
          f"{len(stacks)} stacks x {len(weathers)} weathers x {len(seeds)} seeds\n")

    results: dict = {}
    for stack in stacks:
        for weather in weathers:
            for seed in seeds:
                key = f"{stack.name}|{weather}"
                try:
                    m = one_run(args, stack, seed, weather)
                except Exception:
                    # One episode dying must not take the batch with it -- and
                    # the world is already closed by the runner's finally.
                    traceback.print_exc()
                    print(f"  [ERR] {stack.name} {weather} seed={seed}")
                    continue
                results.setdefault(key, []).append(m)
                flag = "OK " if m.success else ("HIT" if m.collision else "T/O")
                print(f"  {flag} {stack.name:32s} {weather:12s} seed={seed} "
                      f"prog={m.progress:6.1f}m v={m.mean_speed:.1f} "
                      f"minClr={m.min_clearance:.2f}")

    if not results:
        print("\nno episodes completed", file=sys.stderr)
        return 1

    print("\n" + "=" * 104)
    print(f"{'stack | weather':46s} {'succ':>6s} {'coll':>6s} {'prog_m':>8s} "
          f"{'v_mean':>7s} {'minClr':>7s} {'d_safe':>7s} {'e2e_ms':>7s}")
    print("-" * 104)
    table = []
    for key, runs in results.items():
        a = aggregate(runs)
        a["key"] = key
        table.append(a)
        print(f"{key:46s} {a['success_rate']:>6.2f} {a['collision_rate']:>6.2f} "
              f"{a['mean_progress_m']:>8.1f} {a['mean_speed']:>7.2f} "
              f"{str(a['min_clearance_m']):>7s} {a['mean_d_safe_m']:>7.2f} "
              f"{a['e2e_ms_p95']:>7.1f}")
    print("=" * 104)
    print("\nThese are CARLA numbers. They are NOT comparable row-for-row with the\n"
          "built-in-simulator ablation: different vehicle, different traffic model,\n"
          "different road. What transfers is the ordering between the arms.")

    with open(args.out, "w") as f:
        json.dump({"table": table,
                   "runs": {k: [m.as_row() for m in v] for k, v in results.items()}},
                  f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
