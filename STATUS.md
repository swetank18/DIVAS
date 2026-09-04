# Where this stands — handoff, 1 Sep 2026

*Updated 3 Sep 2026, the night before the presentation: **the ego now reaches
its goal in CARLA** — 201.7 m, no collision, 7.12 m/s mean, up from 101.1 m and
4.56 m/s. The cause was the pedal model, it was identified against the live
server rather than guessed at, and the honest size of the effect is smaller
than it first looked. There is a rendered jury video. §6 is the whole account,
and it corrects §3 item 1b, which blamed the wrong thing.*

*Updated 1 Sep 2026, later the same day: **the stack drives in CARLA.** Driver
fixed, 0.9.16 installed, bridge validated end-to-end on Town10HD — 101 m, no
collision, 46.7 ms p95. §2 has the numbers.*

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
   are marked *(Mandatory)*. Hard submission blocker, and the only one left
   that nobody but you can clear. `scripts/fill_deck_mentors.py` writes them
   in place, keeping the row's existing font — pass
   `--faculty "Name|Dept|Year|Sem|Gender|Email|Mobile"` and the same for
   `--industry`. It prints what is still blocking, so it is also the check.
2. ~~Slide 1 says "SMART INDIA HACKATHON **2025**"~~ — **fixed 3 Sep**, now
   2026. A timestamped backup of the previous file sits beside it.
3. Two registration numbers are still malformed — Swetank `RA25033010134`
   (13 chars) and Aneek `RA251102601641` (14), where the other four are 15.
   **Not guessed and not patched**: the pattern would suggest
   `RA2511026010134` and `RA2511026010641`, but these are identity numbers on
   a submitted form, and a plausible-looking wrong one is worse than a
   visibly wrong one. Confirm them and run
   `--fix-reg "SWETANK KUMAR=RA..."`, repeatable.

---

## 1. What exists and works

Repo: <https://github.com/swetank18/autonomous_path> — all yours, no co-author
trailers. ~8,300 lines, **70 tests green**, 9 ADRs.

```bash
python3 -m pytest tests/ -q                       # 70 passed + 3 skipped, no GPU, ROS or CARLA
~/carla-venv/bin/python3 -m pytest tests/ -q      # 73 passed, incl. the real-client checks
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
1b. ~~**Tune the longitudinal mapping.**~~ **DONE, 3 Sep — and the guess in
   this item was wrong.** The mapping was indeed the problem, but not for the
   reason written here, and not at the size implied. See §6.
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

---

## 6. The night before, 3 Sep — the pedal model, and a demo that reaches its goal

### What was wrong, and what was wrong about the diagnosis

§3 item 1b said the ego crawled because `control_from_command` was crude. Half
right, and the reasoning under it was wrong in a way worth recording, because
the wrong version is the one that sounds convincing.

The mapping was `throttle = accel / max_accel`, open loop. Stage 6 emits
`accel = 1.1 * (v_ref - v)`, so the command decays to zero exactly as the ego
arrives at its reference — and zero throttle means coasting. The first guess
was that this alone explained 4.56 m/s against a 9.0 m/s cruise target.

**It cannot.** The steady-state error of a proportional loop against a
disturbance is `disturbance / gain`. Producing a 4.4 m/s error at a gain of 1.1
needs a resistance of nearly 5 m/s², and a real car coasts at about 0.3. The
first test written for the fix failed on exactly this point, which is what the
test was for. Two things were actually true:

1. **CARLA's coast-down is enormous** — not 0.3 m/s² but **4.08 m/s² at
   9 m/s**, measured. The default vehicle physics carry
   `damping_rate_zero_throttle_clutch_engaged = 2.0` and an autobox that
   downshifts hard on lift-off (gear 3 → 2 → 1 within a second). This is
   engine braking, it is an order of magnitude beyond real drag, and it is
   *the plant*, so it has to be modelled rather than argued with.
2. **`max_accel` and `min_accel` are the planner's comfort limits**, 2.0 and
   −4.0 m/s², and were being used as the pedal normalisation. The vehicle's
   real authority is **6.47** and **3.44** m/s² per unit of pedal. Normalising
   a pedal by a comfort bound is a category error, and it is wrong here by
   more than a factor of one and a half in both directions.

Even together these are worth about **1.1 m/s**, not 4.4. The rest of the
original 4.56 m/s mean was always legitimate: Town10 traffic, junctions, and a
curvature-limited reference that is simply below cruise on a dense urban route.
**Do not let the deck claim otherwise.**

### The plant was identified, not guessed

`scripts/calibrate_longitudinal.py` measures it on the live server: coast-down
segments at 3, 5, 7, 9, 11 and 13 m/s with lane keeping, then a pedal-authority
sweep, then the same speed-hold run under the old mapping and the new one.
Record in `docs/longitudinal-calibration.{json,png}`.

```
resistance  ResistanceModel(c0=2.3915, c1=0.18802, c2=0.0)   RMS residual 0.77 m/s^2
pedals      throttle_gain 6.47   brake_gain 3.44   m/s^2 per unit
speed hold  open loop (old)   7.88 m/s      closed loop (new)   9.01 m/s
            against a 9.0 m/s reference, straight and empty
```

Two traps paid for in the rig itself, both now guarded:

- **Steering straight ahead is not a control experiment.** The first version
  held `steer = 0` on Town04, whose highway curves; the ego left the
  carriageway in seconds and the coast-down it measured — 1.5 to 3.3 m/s² —
  was the rolling resistance of *grass*. Every sample is now discarded unless
  the ego stayed on a `Driving` lane for the whole window.
- **Clamping fitted coefficients is not the same as constraining a fit.**
  `fit_resistance` originally forced `c0, c2 >= 0` after solving the joint
  normal equations, because drag cannot be negative. CARLA's curve flattens
  above 9 m/s and genuinely wants a negative quadratic term; pinning it turned
  a good fit into `1.31 v`, which reads 11.8 m/s² at cruise, and the resulting
  feedforward floored the throttle and drove the ego to **26 m/s** before it
  crashed. The fit is now unconstrained and the *curve* is validated for
  positivity, falling back to a line and then to the mean.

Also: **Town04 segfaults the server.** 6 GB of VRAM does not fit the large
maps. Calibration runs on Town10HD_Opt.

### The controller

`LongitudinalTracker` in the bridge: resistance feedforward, measured pedal
gains, and an integral trim on the *acceleration* tracking error with
anti-windup. The trim matters more than it looks — the feedforward is a
three-parameter fit with a 0.77 m/s² residual to a gear-dependent, hysteretic
curve, so it is wrong everywhere by a little. `control_from_command` is kept
as the open-loop baseline the calibration compares against.

`ki = 0.60`, `i_limit = 3.0`. The limit is generous on purpose: braking
saturates constantly in traffic. It is *not* enough to overcome a plant that
physically cannot reach the reference, and it should not be — there is a test
that asks for an unreachable speed and asserts the tracker sits on a floored
throttle at the plant's ceiling instead of integrating against physics.

### The result, same scenario and seed as §2

| | 1 Sep | 3 Sep |
|---|---|---|
| progress | 101.1 m | **201.7 m — goal reached** |
| mean speed | 4.56 m/s | **7.12 m/s** |
| min clearance | 4.95 m | 4.66 m |
| collision | none | none |
| e2e p95 | 46.7 ms | 51.6 ms |

### The jury video

```bash
~/carla-venv/bin/python3 scripts/record_carla.py --stack cv_pred_fixed_margin --seed 1
```

`demo/carla_town10_ours.mp4` — chase camera on the left, the stack's own view
on the right: drivable area, the planned path, tracked actors, and a halo at
`d_safe` around each one. Both panels are the *same* episode, so they cannot
disagree. A `chase` camera was added to the sensor rig for this and nothing
upstream of stage 1 may read it.

Encode settings are deliberate — straight from 1920-wide frames at CRF 20 this
came out at 39 MB, six times the size of the whole repository. 1600 wide at
CRF 26 is 8 MB and indistinguishable on a projector.

### The CARLA episode does not discriminate between the arms — know this

The baseline arm was recorded on the same map, same seed, same traffic
(`demo/carla_town10_baseline.mp4`) and it **also reaches the goal**: 201.7 m,
7.31 m/s, min clearance 4.39 m, no collision. Ours: 201.7 m, 7.12 m/s, 4.66 m,
no collision.

So be precise about what the CARLA video is evidence *for*. It is an
integration and feasibility claim — the same six stages drive a full vehicle
dynamics model in a rendered town at 51.6 ms p95, which is what ADR-005 and
the stage contracts were built to make possible. It is **not** evidence that
prediction-aware planning helps, and presenting it that way would be
overclaiming something the run does not show.

The reason is not subtle: CARLA's traffic manager drives *politely* and its
walkers use AI controllers that largely keep to the pavement, so this episode
never contains the conflict the method addresses. The 2-D scenarios were built
adversarially on purpose — a pedestrian steps off the kerb, a two-wheeler cuts
in — and that is where the baseline hits things and the ablation separates
0.42 from 0.06.

**In the deck:** lead the scientific claim with
`demo/compare_pedestrian_crossing_*.mp4` and the ablation table, and use the
CARLA video for "and it runs in a real simulator, on real vehicle dynamics".
Discriminating between arms *in CARLA* would need scripted adversarial
scenarios rather than the traffic manager, which is a real next task and is
not done.

### Free space now means free space

`DrivableRaster.carve` clears oriented boxes out of the drivable mask, and
`CarlaWorld` feeds it the town's baked-in obstacles from
`get_environment_objects`. This matters because the mask is painted from the
lane graph, and **Town10HD carries 48 parked vehicles as static meshes rather
than actors** — nothing spawns them, `ground_truth_tracks` cannot report them,
and left alone they read as drivable.

**Measured effect on Town10HD: essentially none, and say so.** Of the 32 that
fall inside the rasterised area, *zero* have their centre on a driving lane;
the median sits 3.38 m clear of one, because this is a US-style town that
parks in bays. One cell gets carved, from a corner overlap. The mechanism is
kept because it is the correct definition of free space and because the roads
this project is actually about do not park in bays — a lorry stopped in the
carriageway is the normal case on an Indian road, and it is precisely the case
a lane-graph raster gets silently wrong. **This is not a Town10 result.**

### MATLAB

Installer is `~/Downloads/matlab_R2026a_Linux` (the *online* installer, 672 MB,
which downloads products at install time). `sudo ~/Downloads/matlab_R2026a_Linux/install`,
and select MATLAB, Simulink, Automated Driving Toolbox, Model Predictive
Control Toolbox, Computer Vision Toolbox and Navigation Toolbox.

`sim/matlab/` holds an independent implementation of the two models every
published number rests on — the kinematic bicycle and the longitudinal
controller — plus `validate_against_python.m`, which replays references
exported by `scripts/export_for_matlab.py` and reports the disagreement. It
needs base MATLAB only; `divas_wrap_angle.m` exists so that `wrapToPi`
(Mapping Toolbox) is not required.

```bash
python3 scripts/export_for_matlab.py
matlab -batch "cd('sim/matlab'); validate_against_python"
```

The exported Python reference already reproduces the live server: closed loop
settles at 9.0000 m/s and the old mapping at 7.9100, against 9.01 and 7.88
measured in CARLA. **The MATLAB half has not been executed** — MATLAB is not
installed yet, and there is no Octave on this machine to stand in for it (and
Octave would not, honestly: `readtable`, `yline` and `exportgraphics` are not
in it). Treat the `.m` files as reviewed but unrun until that command prints
PASS.

Be careful what this buys. It establishes that the integrator and the pedal
controller are not artefacts of one codebase. It says nothing about the
planner, the predictor or the risk field, none of which are reimplemented, and
it cannot catch an error *shared* by both implementations. RoadRunner — the
actual reason the licence matters — is a multi-day asset job and the deck
should not claim an Indian custom map.

---

## 7. Extended navigation, 3 Sep — what a long drive exposed

Asked for a long run with full navigation, lane changes, parking and traffic.
Three of those four are now real; the fourth is not, and this section says why
rather than dressing something else up as it.

### Routes longer than the town

Town10HD is about 400 x 230 m, so any drive past roughly a kilometre is a
circuit -- and `Route.progress` searched the **whole** polyline for the nearest
point, which on a self-crossing route is not a small inaccuracy but a wrong
answer. `Route(windowed=True)` follows a cursor instead. It is opt-in
(`CarlaConfig.long_route`, `--long-route`) because every published number was
measured with the global search, and silently changing how progress is
measured would move all of them.

Getting it right took two goes, and both failures are worth keeping:

1. **Argmin over a band of arc length teleports.** The band is measured along
   the route; "near" is measured in space. On a two-way street the opposite
   carriageway is 3.5 m away spatially and tens of metres away along the
   route, so the cursor jumped to it. Measured: a 25 s run reporting
   **1808 m** of progress, or 72 m/s. Following the route forward to the first
   local minimum of distance cannot cross that gap.
2. **Starting the walk behind the cursor stalls it.** A 20 m backward window
   looks like harmless robustness. From there the nearest local minimum
   genuinely *is* the outbound carriageway, so the cursor never follows onto
   the return leg: an out-and-back route stalled at **67 m of 123 m**. The
   walk now starts at the cursor and progress ratchets, which is safe because
   neither simulator lets the vehicle reverse.

Both are pinned by tests using realistic geometry -- a closed lap driven twice,
and an out-and-back on separated carriageways. The earlier test used a route
that overlapped itself *exactly*, which no road does and which no algorithm
can resolve without heading.

### Lane changes, measured rather than asserted

`CarlaWorld.lane_context()` reports the OpenDRIVE `road_id`, `lane_id` and
junction flag; `record_carla.lane_events` turns a run into a count. Two things
it gets right and a naive version does not: lane ids are unique only *within* a
road, so the pair is what changes; and junction frames are excluded, because
lane ids inside a junction describe connecting roads and every turn would
otherwise read as a burst of lane changes -- the first HUD reported **33
junctions in 88 m** for exactly that reason.

State this carefully in the deck. **The planner has no concept of a lane.** It
plans over free space, which is the entire design. These are lane changes
*observed* in the trajectory a free-space planner produced, not manoeuvres it
selected. That is a more interesting claim than the one it is easy to overstate
into, and it is the true one.

### Red lights

The stack had no signal handling at all, which on a signalled Western town is a
real gap. It now arrives the only way it honestly can: a red light becomes a
**keep-out in the static costmap**, spanning every lane that light stops --
because a free-space planner asked to avoid a barrier across one lane will
drive round it into the oncoming one. Ground truth from the simulator, exactly
like the tracks and the drivable area, standing in for the perception that
would report it. Not a contribution; do not present it as one.

Verified live: with every light in Town10HD frozen red, the ego is handed two
keep-out boxes 3.8 m ahead, correctly oriented across both lanes.

**But be honest about how little this was exercised.** Instrumenting a 25 s
dense run: of 237 perception steps, the ego had **no light governing it 226
times and a green 11 times -- and never once a red.** So red-light handling
neither caused nor prevented anything measured below, and its only live test is
the forced one above.

### Dense traffic: it collides about half the time

Five seeds, 40 vehicles and 20 pedestrians, 25 s, long route:

| | collision rate | mean progress |
|---|---|---|
| without red-light keep-outs | 0.40 | 453 m |
| with them | 0.60 | 503 m |

Two of five versus three of five is not a difference at n=5, and it could not
be one given the ego never met a red. **The honest reading is that at this
density the stack collides roughly half the time**, always with a vehicle
(`actor:car`, `actor:truck`) and at a minimum time-to-collision of 0.3--0.8 s.

The likely cause is not signals and not the pedal model: **the stack has no
right-of-way or yielding logic.** It plans through free space, and at a
junction free space includes the path of a vehicle that is about to occupy it.
The risk field extends the predicted actors, but a constant-velocity prediction
of a car that has not yet started its turn does not cover where it will be.
This is a real limitation, it is the next thing worth fixing, and it should be
said out loud rather than discovered by a juror.

### The long-route crawl is congestion, not the controller

A 1343 m windowed route at 20 vehicles covers 189 m in the first 30 s -- the
same 6-7 m/s the short route managed -- and then slows to about 0.5 m/s for the
next minute. Two things it is **not**, both checked rather than assumed:

* **Not the progress metric.** Instrumented against distance actually driven:
  221.6 m of progress against 222.1 m driven. The windowed cursor tracks.
* **Not the red-light keep-outs.** Same seed, same route, 90 s:
  221.6 m with them and 225.1 m without. Within noise.

What is left is congestion: Town10's traffic manager queues at signals, the ego
queues behind it, and it has no overtaking policy that would justify pulling
out around a stopped queue into an oncoming lane. That is realistic urban
behaviour rather than a defect, but it does mean **mean speed over a long town
route is dominated by waiting**, and quoting it as a capability number would be
misleading in the other direction from the 4.56 m/s figure in §6.

### The extended run, and what it ends on

`demo/carla_town10_extended.mp4` -- seed 3, 1369 m windowed route, 20 vehicles
and 10 pedestrians, 3x speed:

```
415.9 m driven    4 lane changes    6 junction traversals
no collision      min clearance 1.08 m      mean speed 6.06 m/s
```

It **ends stopped at a junction after 68 s**, with no plan drawn, terminated by
the runner's stuck detector rather than by the 200 s limit. That is not the
pedal model and not the route: on the same seed over 45 s the long route plans
at **0.99 success and 81 ms e2e p95**, indistinguishable from the short route
(0.98, 89 ms), and the goal geometry is identical -- median 28.2 m ahead,
maximum 30.9 m, all comfortably inside the 32 m costmap.

What changes is congestion. When the ego is boxed in at a busy junction the
goal 28 m ahead is genuinely unreachable through stopped traffic, Hybrid A*
runs to its 4000-iteration ceiling proving it, and `plan_success_rate` falls to
0.48 with e2e p95 at 530-720 ms. **Those latency figures are the planner
failing, not the planner being slow**, and they must not be quoted as timing
for a working stack -- nor as a real-time claim at all, per ADR-008.

The honest summary for the deck: the stack navigates a kilometre-scale town
route through traffic, changes lanes and negotiates junctions, and gets stuck
when a junction jams. Fixing that is the yielding/right-of-way work above, plus
a fallback for a goal that is temporarily unreachable -- neither of which
exists yet.

### Parking — not supported, and not faked

Two independent blockers, both checked rather than assumed:

1. **The planner cannot reverse.** `HybridAStar._expand` generates forward arc
   primitives only (`+L`), and both simulators clamp `v >= 0` -- the built-in
   world at `divas/sim/world.py`, and the bridge through the same state. Any
   parking manoeuvre that needs a reverse leg is unreachable, and parallel
   parking needs one by definition.
2. **Town10HD has no parking bays in its road network.** Its OpenDRIVE carries
   only `Driving`, `Shoulder` and `NONE` lane types next to the carriageway;
   there is no `Parking` lane to target. The 48 parked vehicles are static
   meshes standing on shoulders, not bays.

What it would take, honestly: reverse motion primitives in Hybrid A* with a
direction-switch cost (the branching factor doubles), signed speed through the
MPC and the bicycle model, reverse gear in `control_from_command`, and a map
with somewhere to park. That is a day or two of careful work touching the
planner every published result depends on -- not a thing to attempt the night
before a presentation.

What the stack *does* do at the end of a route is taper to a controlled stop
(`Path.terminal_stop`), which the new longitudinal controller now holds
properly through its standstill branch. Call that "comes to a controlled stop
at its destination". Do not call it parking.

---

## 8. Phase 4, 4 Sep — the margin becomes a measurement

The Phase 1 ablation killed this project's headline mechanism: a fixed margin
equal to the dynamic arm's mean scored identically, 0.90/0.08. Phase 4 is the
answer to *why*, and to what replaces it.

### Why the adaptive margin could not have worked

Not because adaptivity is worthless, and not because the confidence signal is
noise. ADR-009 measured that signal and it is real: prediction error by
confidence quintile runs **7.01, 5.59, 4.47, 3.90, 2.66 m**, correlation −0.27.

**The problem is scale.** That is 4.35 m of error variation. `lam` is 0.8 m and
the margin is capped at 1.8 m, because ADR-003 found that ~5 m keep-outs block
a 10 m carriageway outright. The margin's entire dynamic range is about a fifth
of the variation it is responding to. With a range that small, only the mean
can matter — which is exactly what the control arm reported.

There is a second, sharper problem with the signal itself. Confidence is the
spatial dispersion of the predictor's modes: *the predictor disagreeing with
itself*, which is not *the predictor being wrong*. With the constant-velocity
predictor there is one mode, so dispersion is identically zero and the
"dynamic" margin is **literally a constant** — for every actor, at every step,
in every scenario. There is a test asserting that.

### What replaces it

```
tuned       d_safe(t) = d0 + k_v·v_ego + λ·(1 − confidence(t))
calibrated  d_safe(t) = d0 + k_v·v_ego + q_{1−α}(t)
```

`q_{1−α}(t)` is the conformal quantile of the predictor's **own recent error**
at lookahead `t`. Split conformal gives a distribution-free finite-sample
guarantee, `P(‖actual − predicted‖ ≤ q) ≥ 1 − α`, with no assumption about the
predictor, the noise or the traffic — a property no tuned coefficient has at
any value. Realised coverage: **0.941 / 0.891 / 0.793** against nominal
0.95 / 0.90 / 0.80, flat across every horizon step.

This is also the honest answer to the closest prior art. Fisac et al. (RSS
2018) get the same *shape* from Bayesian model confidence and Hamilton–Jacobi
reachability; this is weaker in per-step guarantee and stronger in assumptions,
because it needs none.

### Two bugs found in our own calibrator, and one reversed conclusion

1. **Coverage was self-referential.** `observe` scored each residual against a
   quantile recomputed *after* inserting that residual. It now scores against
   the margin that was in force when the prediction was made — which is both
   unbiased and the question the vehicle actually faced.
2. **The quantile was recomputed per residual**, re-partitioning the whole
   window every time. Quadratic in the window; a calibration sweep ran for
   forty minutes without finishing.

Fixing (1) reversed the conclusion on adaptive conformal inference (Gibbs &
Candès 2021). Post-shift coverage, honestly scored:

| window | plain rolling | with ACI |
|---|---|---|
| 240 (default) | 0.880 | **0.844** |
| 1200 | 0.848 | 0.834 |
| 4000 | 0.771 | 0.814 |

**ACI makes things worse at the default window** — the rolling quantile already
tracks the shift, and ACI is a second feedback loop correcting an error the
first one fixed. It is off by default, and a test fails if it is switched back
on silently. The earlier claim that it recovered 0.700 → 0.860 came from the
biased measurement and is withdrawn.

### The finding that matters most

The first live run of the conformal arm **saturated its own cap**: mean
`d_safe` 2.45 m against a 2.5 m ceiling, coverage still only 0.854.

**An honest 90% keep-out for constant-velocity prediction at a 3 s lookahead
does not fit on the road.** No margin policy — fixed, tuned or calibrated — can
cover an error wider than the carriageway. Calibration does not create
accuracy; it reveals its absence, as a number rather than as a mis-tuned
coefficient nobody can see.

Two instruments follow, and they are the Phase 4 deliverables:

* **Required Safety Margin**, `scripts/required_margin.py` — a predictor metric
  denominated in metres of carriageway rather than in ADE/FDE. A predictor
  change that does not reduce RSM has bought the planner nothing.
* **Actionable horizon** — the lookahead past which an honestly calibrated
  keep-out stops fitting on the road. `StackConfig.horizon` exists so the
  ablation can test whether a shorter, well-covered horizon beats a longer,
  under-covered one.

### What Phase 4 does not yet claim

That any of this reduces collisions. That is the ablation's job, and it will be
answered with the same control arm that killed the heuristic — a fixed margin
at the conformal arm's own mean. If it ties again, the conclusion is again that
size beat variation, and it gets reported the same way.

---

## 9. Phase 4 result — the required safety margin

`scripts/required_margin.py` measures the quantity the planner actually needs,
which falls straight out of conformal calibration:

> **RSM(t, α)** — the (1−α) quantile of the predictor's displacement error at
> lookahead `t`, in metres. The radius a keep-out must have for the actor to be
> inside it (1−α) of the time.

It is denominated in **metres of carriageway** rather than in ADE/FDE, because
that is a budget the vehicle either has or does not, and it is a *quantile*
rather than a mean, so it speaks about the tail that collides.

Over 323,414 residuals, 8 scenarios × 3 seeds, at 90% coverage:

| lookahead | 0.5 s | 1.0 s | 1.5 s | 2.0 s | 2.5 s | 3.0 s |
|---|---|---|---|---|---|---|
| constant velocity | 3.05 | 4.33 | 5.89 | 7.91 | 9.91 | **12.01 m** |
| social force | 3.09 | 4.41 | 6.05 | 8.19 | 10.20 | **12.43 m** |

**The margin the stack carries is capped at 1.8 m.** The margin an honest 90%
keep-out needs at a 3 s lookahead is **12 m**, and even at half a second it is
already 3 m. Short by a factor of seven at the horizon the planner actually
uses.

That is the whole explanation of the Phase 1 null result, and it is a better
one than "the confidence signal is weak". A fixed 1.4 m margin and a dynamic
one averaging 1.4 m scored identically because **both are equally inadequate**:
the difference between them is noise against an error an order of magnitude
larger. No margin policy can cover an error wider than the road.

It also refutes the interaction-aware arm on its own terms rather than by
ablation alone: social force needs **more** margin than constant velocity
(12.43 against 12.01 m), consistent with ADR-009's finding that its obstacle
term made it less accurate than the baseline.

### The decomposition, which decides what to claim

Part of that 12 m is the simulator's own measurement noise rather than the
motion model. `range_noise_sigmas` gives

```
sigma_position = 0.15 + 0.012 · range        0.51 m at 30 m
sigma_velocity = 0.30 + 0.020 · range        0.90 m/s at 30 m
```

and **velocity noise is extrapolated by the predictor over the whole horizon**:
0.9 m/s at a 3 s lookahead is 2.7 m per axis before the model makes a single
mistake. So a large part of RSM's growth with `t` is sensor noise amplified by
extrapolation, not a bad motion model, and the two lead to opposite fixes —
better tracking and a shorter horizon on one hand, a better predictor on the
other. `--track-noise 0` runs the same sweep with the noise removed to
separate them.

**Do not attribute the 12 m to the predictor until that split is measured.**

### The design consequence

RSM turns the prediction horizon from a constant into a measured variable. A
3 s horizon is only useful if the planner can carry a 12 m keep-out, and on a
10 m carriageway it cannot. The **actionable horizon** is the lookahead at
which RSM still fits the margin budget, and `StackConfig.horizon` exists so the
ablation can test whether a shorter, honestly-covered horizon beats a longer,
under-covered one.

---

## 10. Phase 5, 4 Sep — the stack finally sees

Until now stages 1–3 were ground-truth stubs: the simulator handed the planner
its free space. The project is *named* for segmenting drivable area instead of
detecting lanes, and that was the one part not implemented. It is now.

### What was built

```
IDD photograph → drivable-area segmentation → BEV free space → Hybrid A*
```

* `divas/perception/datasets/idd_polygons.py` — IDD ships **only**
  `_gtFine_polygons.json`; there are no label PNGs in the archive. That is
  better than it sounds, because polygons carry label *names*, which removes
  the need for `idd.py`'s flagged `DRIVABLE_LEVEL3_IDS` hypothesis entirely.
* `divas/perception/models/drivable.py` — LR-ASPP over MobileNetV3-Large,
  fine-tuned from ImageNet. **3.22 M parameters, 3.7 ms per frame at 512×288
  on an RTX 3050 — 267 FPS against a stage-1 budget of 10 Hz.**
* `divas/perception/bev.py` — ground-plane projection, BEV cell → image pixel.
* `scripts/{prepare_idd,train_drivable,demo_perception,eval_perception_planning}.py`

### What checking the data corrected

**There is no `parking` class in IDD.** `idd.py` names the drivable set as
`("road", "parking", "drivable fallback")`; scanning 400 annotation files found
32 distinct labels and no `parking` among them. The set is `road` plus
`drivable fallback`.

**And a claim made from one frame that the aggregate refuted.** A single val
image showed 19.5% road against 26.0% shoulder, which looked like a superb
line — *the shoulder is bigger than the road*. Over 800 masks the true split is
**road 25.40%, drivable fallback 4.27%, not drivable 70.34%**. The shoulder is
a *rare* class, which makes it the hard one to learn, and the line does not
survive. What does survive is narrower and still true: Cityscapes and KITTI
have no label for it at all, so a model trained on them cannot represent it.

That correction also fixed a real bug — the loss weights had been hard-coded
from that single frame, and would have trained the model to under-weight
exactly the class the dataset was chosen for. They are now measured from the
cache at startup.

### The metric that IoU cannot give

Every frame is planned **twice** — once on the network's free space, once on
the annotator's, same goal, same Hybrid A\* that produced the closed-loop
ablation — and the difference is reported in metres. A frame with IoU 0.85
where both planners produce the same path is a frame where perception was good
enough; a frame with IoU 0.95 where the paths diverge by two metres is not, and
the pixel metric cannot tell them apart.

### Three bugs found by looking at the figure rather than the numbers

1. **The near-field blind spot.** A forward camera cannot see the road under
   its own bumper — `min_visible_range` puts it at 3.55 m — and unseen ground
   is occupied, so the ego started *inside an obstacle* and every plan failed.
   Visible in the first figure as a black wedge directly ahead.
2. **"Furthest free cell ahead" is a trap.** Free space seen through a gap in a
   hedge is free, and disconnected. Half the goals chosen that way were
   unreachable, and the planner's failure to reach them read as a planning
   failure.
3. **Cell connectivity is not vehicle traversability.** Cells connect through a
   one-cell gap; a 3.9 × 1.7 m vehicle does not. The reachability flood fill now
   runs on configuration space — free space eroded by half the vehicle width
   plus the planner's inflation.

### What Phase 5 does not claim

It is stages 1 → 2 → 5 on real imagery. It is **not closed-loop**: a single
photograph has no next frame, so there is no control step and no prediction.
IDD also ships no camera calibration, so the bird's-eye metre scale comes from
a nominal field of view and mounting height — the geometry is self-consistent,
the absolute distances are an assumption, and that caption is printed on every
figure.

---

## 11. The decomposition — it is the tracker, not the motion model

`scripts/required_margin.py --track-noise 0` reruns the sweep with the
simulator's measurement noise removed, which separates the two things the
required margin was mixing together.

| lookahead | with sensor noise | model error only | noise share |
|---|---|---|---|
| 0.5 s | 3.05 m | **0.30 m** | **90%** |
| 1.0 s | 4.33 m | 1.09 m | 75% |
| 1.5 s | 5.89 m | 2.36 m | 60% |
| 2.0 s | 7.91 m | 4.09 m | 48% |
| 2.5 s | 9.91 m | 6.16 m | 38% |
| 3.0 s | 12.01 m | 8.43 m | 30% |

**The motion model is not the problem at the lookaheads that could fit.** At
half a second constant velocity is wrong by 0.30 m, and 90% of what the margin
must cover there is measurement noise. Model error only becomes the majority
beyond about two seconds.

The mechanism is `range_noise_sigmas`: velocity noise is
`0.30 + 0.020 · range`, so 0.9 m/s at 30 m, and the predictor **extrapolates
that over the whole horizon** — 2.7 m per axis at 3 s before the motion model
errs at all.

### The actionable horizon

Against the 1.8 m margin cap from ADR-003, the lookahead at which an honest
90% keep-out still fits:

```
with perfect sensing        1.2 s
with the current tracker    0.0 s
what the stack plans over   3.0 s
```

So the planner is committing three seconds ahead on predictions that are not
covered at **any** lookahead. That is a better explanation of the Phase 1 null
result than anything before it: the adaptive margin never had a chance,
because no margin inside the budget covers this prediction at this horizon.

### This reverses a roadmap item

ADR-009 concluded that fixing the predictor was the next task, on the evidence
that the social-force model was no better than constant velocity. That is
still true, and it is now also beside the point: **at the horizon where the
margin could fit, constant velocity is already accurate to 30 cm.** The
expensive term is the tracker, and the cheap fix is a shorter horizon.

Ordered by measured value:

1. **Shorten the planning horizon** to what is actually covered, and re-run the
   ablation. `StackConfig.horizon` exists for this.
2. **Reduce velocity noise** — smoothing, or a filter with a motion model, so
   that extrapolation is not amplifying a raw per-frame estimate.
3. **Improve the motion model**, which matters only beyond 2 s and only once
   the first two are done.

### The caveat to settle before quoting "0.0 s"

RSM here aggregates over *every* tracked actor, including ones at the 60 m
track range where the noise model is largest, while the planner mostly cares
about near ones. Conditioning RSM on range will raise the actionable horizon,
and it should be measured before the 0.0 s figure is used anywhere.
