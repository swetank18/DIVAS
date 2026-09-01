# Where this stands — handoff, 1 Sep 2026

*Updated later the same day: **the stack drives in CARLA.** Driver fixed,
0.9.16 installed, bridge validated end-to-end on Town10HD — 101 m, no
collision, 46.7 ms p95. §2 has the numbers. The deck (§0c) is now the only
thing still waiting entirely on you.*

Read this first when you come back. It is written to be resumable without
re-reading the conversation.

---

## 0. What still needs you

### a) ~~Fix the NVIDIA driver~~ — DONE

The diagnosis held exactly: the module existed only for `6.17.0-22-generic`
while you were booted into `7.0.0-30-generic`. Installing
`linux-modules-nvidia-580-open-7.0.0-30-generic` pulled the driver forward to
**580.173.02** and `nvidia-smi` now reports the **RTX 3050 Laptop, 6144 MiB,
CUDA 13.0**. Secure Boot was not the blocker — the modules are
Canonical-signed, and `dmesg` shows the module loading cleanly under lockdown.

One loose end: `dmesg` carries an API mismatch for `gnome-shell`, which still
has the *old* 580.126.09 libraries mapped from before the upgrade. It is
harmless to CUDA and to a `-RenderOffScreen` CARLA, both of which are new
processes linking the new libraries — but **log out and back in, or reboot,
before trying to render a CARLA window in this session.**

### b) ~~Install CARLA~~ — DONE, and the version changed

**Use 0.9.16, not 0.9.15.** 0.9.15 downloaded fine and is unusable here:
`PythonAPI/carla/dist/` ships clients for Python **2.7 and 3.7 only**, and PyPI
has no `carla==0.9.15` for anything newer. This machine is Python 3.12, so
0.9.15 means standing up a second interpreter — and then the whole DIVAS stack
has to run under it too, because `divas` and `carla` must be importable in one
process.

0.9.16 is the **last Unreal Engine 4 release**, so the 6 GB VRAM reasoning is
unchanged and your card is still exactly at the floor (0.10/1.0 moved to UE5
and want 8 GB+). It also ships
`carla-0.9.16-cp312-cp312-manylinux_2_31_x86_64.whl`, which installs on system
Python with no venv gymnastics — already verified against PyPI.

Downloaded and extracted to `~/carla/0.9.16` (19 GB). Its own
`PythonAPI/carla/dist/` ships cp310/cp311/**cp312** wheels, which confirms the
choice from the other direction.

The client is **already installed**, in `~/carla-venv`:

```bash
python3 -m venv ~/carla-venv --system-site-packages   # done
~/carla-venv/bin/pip install carla==0.9.16            # done
```

A venv rather than `pip install --break-system-packages`, because Ubuntu 24.04
refuses system-Python installs (PEP 668) and because this is reversible with
`rm -rf ~/carla-venv`. `--system-site-packages` matters: `divas` and `carla`
have to be importable in **one** process, so the venv inherits the system
numpy 1.26.4, scipy 1.11.4 and matplotlib 3.6.3 rather than reinstalling them.
**Run everything CARLA through `~/carla-venv/bin/python3`.** The suite is 59
tests there and 56 + 3 skipped on system Python — the three extra check the
bridge against the real client.

To run it:

```bash
./scripts/carla_server.sh                                  # server, offscreen
~/carla-venv/bin/python3 scripts/run_carla.py --check      # bridge self-test
```

**Use `scripts/carla_server.sh`, not `CarlaUE4.sh` directly.** This is a
hybrid-graphics laptop — an AMD Radeon 780M in the CPU and the RTX 3050 on
`01:00.0` — and Vulkan enumerates the **AMD part as GPU0**. Launched plainly,
CARLA runs on the integrated chip. It does not error; it just renders at a few
frames a second, which makes a 20 Hz control loop meaningless and reads as a
slow planner rather than a wrong GPU. The script sets
`__NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only`, which puts the
RTX 3050 first. `libomp5` was also missing and is now installed; CARLA will not
start without it.

Two tarballs are still in `~/carla`, 8.4 GB each. Nothing can use the 0.9.15
one — `rm ~/carla/CARLA_0.9.15.tar.gz`, and the 0.9.16 one too once you trust
the extraction.

### c) Fix the two submission blockers in the deck

Still outstanding from the very first analysis, and independent of any code:

1. **Faculty Mentor and Industry Mentor rows on slide 7 are empty** and both
   are marked *(Mandatory)*. Hard submission blocker.
2. Slide 1 says "SMART INDIA HACKATHON **2025**"; the PS ID is SIH26037.
3. Two registration numbers look malformed — Swetank `RA25033010134` (13
   chars) and Aneek `RA251102601641` (14), where the others are 15.

---

## 1. What exists and works

Repo: <https://github.com/swetank18/autonomous_path> — all yours, no co-author
trailers. ~7,100 lines, **56 tests green**, 9 ADRs.

```bash
python3 -m pytest tests/ -q                       # 56 passed + 3 skipped, no GPU, ROS or CARLA
~/carla-venv/bin/python3 -m pytest tests/ -q      # 59 passed, incl. the real-client checks
python3 scripts/run_ablation.py --seeds 8 --jobs 12
python3 scripts/make_comparison.py --scenario pedestrian_crossing \
    --left baseline_conventional --right cv_pred_fixed_margin --seed 1
```

**Phases 0 and 1 are complete.** The full six-stage pipeline runs closed-loop
at three separate rates (perception/prediction 10 Hz, planning 4 Hz, control
20 Hz). Stages 1–3 are ground-truth stubs; stages 4–6 are real.

### The headline result — 48 runs per arm, reproducible

| stack | success | collision |
|---|---|---|
| baseline: CV + fixed 1.0 m + pure pursuit | 0.56 | **0.42** |
| MPC, no prediction | 0.65 | 0.33 |
| **MPC + CV prediction + fixed margin** | **0.92** | **0.06** |
| MPC + interaction-aware prediction + fixed | 0.90 | 0.08 |
| control arm: fixed 1.4 m (= dynamic mean) | 0.90 | 0.08 |
| full: + confidence-scaled dynamic margin | 0.90 | 0.08 |

**Claim this:** prediction-aware planning cuts collisions **0.42 → 0.06** and
raises success **0.56 → 0.92**.

**Do not claim these yet:** interaction-aware prediction shows no measurable
gain over constant velocity, and the dynamic margin shows none over a fixed
margin of the same average size — the fixed-1.4 m control arm proves the
variation adds nothing beyond its mean. Neither is refuted; both are unearned.
See `docs/decisions/ADR-009` and §3 below.

### The jury demo is rendered and pushed

`demo/compare_pedestrian_crossing_*.mp4` — baseline **hits a pedestrian at
29 m**, ours reaches the goal at 110 m, on every seed tried. Pedestrians are
20.4% of India's road deaths, so the demo and the motivation are one argument.
Lead with it.

---

## 2. The CARLA bridge — done, but never run against a real server

`divas/sim/carla_bridge.py` is finished. `CarlaWorld` implements
`divas.sim.interface.SimWorld`, so `divas.eval.runner.run` drives it with no
change to any stage above 3 — which is the claim ADR-005 and the stage
contracts exist to make.

**What landed:**

- `CarlaWorld` — connect, synchronous mode with a fixed delta, ego spawned from
  a blueprint, a seeded traffic manager, vehicles, walkers *with* their AI
  controllers, an optional sensor rig (RGB, semantic, depth, LiDAR, radar) and
  a collision sensor. `close()` is idempotent, `__exit__` calls it, and
  `runner.run` now has a `finally` that calls it — leaked actors were the
  named trap and they are shut.
- `params_from_physics` — length, width, wheelbase and steering lock read off
  the blueprint CARLA actually spawned. `runner.run` no longer overwrites them
  with the defaults, because measuring clearance for one vehicle while driving
  another is a silent way to publish a wrong number.
- `route_from_waypoint` — follows the lane graph with a seeded choice at each
  junction, rather than depending on `agents.navigation.GlobalRoutePlanner`,
  which lives in a directory on disk instead of in the wheel.
- `semantic_drivable_mask` and friends — per-pixel drivable ground truth from
  the segmentation camera. This is the thing that makes Phase 2 tractable.
- `scripts/run_carla.py` — `--check` connects, self-tests the protocol, ticks
  once and cleans up in about ten seconds. Run that first on any new machine.
  `--all-weather` sweeps the presets, which is how the deck's
  "fusion in rain/dust/glare" risk gets *tested* rather than asserted.

**One real bug was found and fixed on the way.** `control_from_command` passed
our road-wheel angle to CARLA unnegated. CARLA's `VehicleControl.steer` is
positive to the **right**; ours is positive to the **left**, counter-clockwise,
because that is what the bicycle model integrates. That is precisely the
mirrored-world failure the module's own docstring predicted — everything runs,
the vehicle just steers into what it was avoiding. There is now a test that
fails if the sign flips back.

**How it is tested without a GPU.** `tests/fake_carla.py` is a stand-in for the
`carla` module: left-handed, y to the right, yaw clockwise, steer positive
right, and it integrates a kinematic bicycle so a full closed-loop episode runs
against it. 30 of the 56 tests cover the bridge, including one that asserts the
runner destroys every actor it spawned. Be clear about what that proves: it
tests *our* side of the seam, not CARLA's semantics. A misunderstanding of the
API reproduces faithfully in both places.

**What has since been checked against the real 0.9.16 client**, without a
server: every CARLA name the bridge uses still exists, every weather preset is
constructible, and `DRIVABLE_TAGS` matches CARLA's own `CityObjectLabel` enum
(`Roads` 1, `RoadLines` 24, `Ground` 25). Those are now three tests that skip
when the client is absent, so upgrading CARLA cannot drift them silently — the
tag numbering already changed once, at 0.9.14, and a silent shift would turn
the Phase 2 drivable-area ground truth into a mask of some other class.

### It has now been run against a real server, and it works

CARLA 0.9.16, Town10HD_Opt, RTX 3050, 20 vehicles and 10 pedestrians:

```
ego              3.63 x 1.85 m, wheelbase 2.49 m, lock 70 deg   (read off the blueprint)
route            394 m from the lane graph
drivable raster  936 x 995 cells @ 0.25 m, rasterised once
costmap          256 x 256, 70.5% occupied      (Town10 is dense urban)
```

One closed-loop episode, `cv_pred_fixed_margin`, seed 1, 25 s:

| progress | collision | min clearance | mean speed | e2e p95 |
|---|---|---|---|---|
| **101.1 m** | none | 4.95 m | 4.56 m/s | 46.7 ms |

The full sensor rig delivers: semantic 540x960 with 18 distinct tags and
**41.4% drivable** on a forward view, RGB, 14,747 LiDAR points, 14 radar
detections at 36–68 m. `hard_rain` applies (precipitation 80). Requesting a
sensor correctly forced `no_rendering_mode` back off.

**One real bug only a live server could find.** The first `--check` passed and
then dumped core: a listening sensor callback firing into a half-torn-down
client throws on CARLA's own thread, where no `try` of ours can catch it; a
walker controller attached to an already-destroyed walker is itself already
dead; and in synchronous mode the server only *processes* a destruction on the
next tick, so destroy-then-disconnect leaves the commands queued. Teardown is
now one `apply_batch_sync(..., due_tick_cue=True)` with sensors stopped first
and controllers destroyed before their walkers. Clean exit, and `get_actors()`
afterwards shows only the map's own traffic lights and the spectator.

## 3. Next tasks, in priority order

1. **Perception, Phase 2.** Now unblocked, and the highest-value thing left.
   The semantic camera gives free drivable-area ground truth in the same frame
   as the RGB the model consumes — verified live at 41.4% drivable on a forward
   view.
1b. **Tune the longitudinal mapping.** Mean speed in CARLA was 4.56 m/s against
   a 9.0 m/s cruise target. Some of that is Town10 traffic, but
   `control_from_command` is deliberately crude — throttle proportional to
   requested acceleration, no engine or brake force curve. Check that before
   blaming the MPC, and re-check it before quoting any CARLA speed number.
2. **(was Phase 2, now folded into 1.)** The semantic camera gives free drivable-area ground
   truth in the same frame as the RGB the model consumes — `semantic` in
   `--sensors`, then `semantic_drivable_mask`. Note `DRIVABLE_TAGS` excludes
   tag 25, unpaved ground; Indian carriageways routinely include the shoulder,
   and `DRIVABLE_TAGS_WITH_SHOULDER` is there for when you decide to claim it.
   Then download IDD and run `python3 scripts/verify_idd.py --root ~/IDD`,
   which reports whether the label-id mapping is right before any training run
   depends on it.
3. **Give the predictor a car-following term.** The diagnosis is concrete: the
   social term's `social_range` is **1.4 m**, but real traffic interaction
   happens at 10–30 m, so the model literally cannot see the interactions it
   exists to capture. It also has no car-following term at all, which is where
   most longitudinal error lives. This is what would make claim #2 earnable.
   Caveat to state honestly when done: the built-in simulator's actors use IDM,
   so an IDM-like predictor is partly scored against its own model class —
   which is one more reason to re-measure it in CARLA, whose traffic manager is
   a different model again.
4. **ROS 2 node wrappers** so the rates are enforced by middleware rather than
   by the runner's loop counter. `ros2_ws/src/` is still empty.
5. **Real-time measurement** with the deployment clock budget enabled, ideally
   on Jetson. Current numbers are iteration-bounded and are deliberately *not*
   a real-time claim (ADR-008). Note that a *synchronous* CARLA session waits
   for our tick, so running the stack in CARLA does not by itself measure
   real-time behaviour either.

## 4. Traps already paid for — do not re-introduce

Nine ADRs in `docs/decisions/`. The three expensive ones:

- **ADR-006, horizon consistency.** The route goal was closer than the control
  horizon, so every MPC rollout ran off the path end where cross-track cost
  explodes, and braking became the cheapest way to keep the horizon inside the
  path. The vehicle crawled at 1.8 m/s on an empty road with the nearest actor
  25 m away. Invariant to preserve: **costmap ⊃ route lookahead > control
  horizon reach.**
- **ADR-008, deterministic evaluation.** The planner terminated on a wall-clock
  deadline, so identical code and seeds gave 0.62 vs 0.79 success purely from
  CPU contention. Evaluation is now iteration-bounded. **Never benchmark
  anything that terminates on wall-clock time.**
- **ADR-009, predictor forces.** The obstacle force was the single largest
  source of error in the stack — it pushed predicted agents off the road
  boundary in ways real agents do not move, and made the "interaction-aware"
  predictor *less accurate than constant velocity*.

Two more, both now guarded by tests:

- **CARLA steer is positive to the right, ours is positive to the left.** See
  §2. `control_from_command` negates; `steer_from_control` un-negates.
- **Vulkan picks the integrated GPU on this laptop.** See §0b. Always launch
  through `scripts/carla_server.sh`.
- **CARLA teardown order is load-bearing.** Sensors stopped first, walker
  controllers before their walkers, and one batched destroy with the tick cue —
  see §2. Getting it wrong aborts the process from a thread you cannot catch on.
- **`no_rendering_mode` blanks every camera.** It is a large speed-up on a
  laptop GPU and the right default for an evaluation batch, but it silently
  returns empty frames, which looks exactly like a sensor that failed to
  attach. Requesting any sensor now overrides it.

Also: `deck/` is gitignored on purpose. It carries six people's names,
registration numbers, emails and mobile numbers, and the repo is **public**.
Make the repo private before adding it.

---

## 5. Decisions still yours

| # | decision | status |
|---|---|---|
| 1 | Simulator | **resolved — CARLA**, and the bridge is written (MATLAB credentials later, for RoadRunner maps and control validation) |
| 2 | Hardware: physical rover, or simulation only? | open — decides whether Phase 6 exists |
| 3 | Actual deadline | open — decides how much of Phases 3–5 is realistic |
| 4 | Faculty + industry mentor | open — **submission blocker** |

One thing to know about CARLA and "hyper realistic": its stock maps are
Western roads *with* lane markings. Your stack ignores lane markings by
design, so free-space navigation still tests honestly — but the road *texture*
will not be Indian until you build a custom map in RoadRunner, which needs the
MATLAB licence. Worth knowing so the deck does not overclaim.
