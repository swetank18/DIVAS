# Where this stands — handoff, 1 Sep 2026

*Updated later the same day: the CARLA bridge is finished and tested (§2, §3),
and the NVIDIA driver is fixed — `nvidia-smi` sees the RTX 3050. The CARLA
version changed to 0.9.16 for a Python 3.12 reason worth reading in §0b. The
deck (§0c) is the one thing still waiting entirely on you.*

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

### b) Finish the CARLA download — running now, and the version changed

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

The download is running in the background; `tail -1 ~/carla/download.log`, and
`wget -c` resumes if it dies:

```bash
cd ~/carla
wget -c -O CARLA_0.9.16.tar.gz \
  "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.16.tar.gz"
```

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

Then, once the download lands:

```bash
tar -xzf CARLA_0.9.16.tar.gz            # ~20 GB extracted; 98 GB free
./CarlaUE4.sh -quality-level=Low -RenderOffScreen   # smoke test
~/carla-venv/bin/python3 scripts/run_carla.py --check
```

`~/carla/CARLA_0.9.15.tar.gz` is still on disk, 8.4 GB and complete. Nothing
here can use it — delete it once 0.9.16 is extracted and working.

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

**So the bridge is unvalidated, not untested.** What remains untested is
everything that needs a live server: spawning, ticking, the traffic manager,
and whether `vehicle.nissan.micra` is still a blueprint in 0.9.16 (if it is
not, `_blueprint()` falls back to another vehicle rather than dying).

## 3. Next tasks, in priority order

1. **Validate the bridge against a real CARLA server.** Driver, then extract,
   then `python3 scripts/run_carla.py --check`. Expect the first real failures
   here — blueprint ids that moved between releases, a town whose spawn points
   are all occupied, and the longitudinal mapping in `control_from_command`,
   which is deliberately crude (throttle proportional to requested
   acceleration, no engine or brake force curve). If the ego undershoots its
   speed reference in CARLA but not in the built-in sim, that function is the
   first place to look, not the MPC.
2. **Perception, Phase 2.** The semantic camera gives free drivable-area ground
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
