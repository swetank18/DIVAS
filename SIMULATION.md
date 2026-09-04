# Simulation handoff — CARLA + MATLAB

For whoever's building the next CARLA and MATLAB simulation demos.
Written so you don't have to re-derive what's already built, what it
proves, and what it doesn't. Read `CONTEXT.md` first — this is the
sequel, specific to simulation work.

**Before doing anything:** `git pull`, then `.venv/bin/python -m pytest
tests/ -q` — confirm 121 passed, 3 skipped, on your machine, before
building on top of it. If that's not what you get, something's
different about your setup or the branch — fix that first.

---

## 1. What you're actually building on top of

- **2D sim** (`divas/sim/world.py`) — fast, no GPU, the source of every
  published ablation number. Scripted and reactive traffic (IDM
  car-following + overtaking), potholes (now a slow-down hazard, not a
  crash — see `CONTEXT.md`).
- **CARLA bridge** (`divas/sim/carla_bridge.py`) — drives the identical
  stack in a real simulator, real vehicle dynamics. Real Bengaluru road
  network available now (`docs/maps/bengaluru*.osm/.xodr`) alongside the
  stock Town10HD map, plus a bazaar/crowd scenario with cattle.
- **MATLAB cross-validation** (`sim/matlab/`) — an independent
  reimplementation of the bicycle model and longitudinal controller, to
  prove the Python integrator isn't hiding a transcription bug.
- **Perception** — a real IDD-trained segmentation model
  (`divas/perception/models/drivable_idd.pt`) and, from this session, real
  object detection + a detection→tracking→prediction bridge
  (`divas/perception/detection.py`, `tracking.py`,
  `scripts/run_prediction_demo.py`). **None of this perception work is
  wired into the closed-loop sim yet** — CARLA and the 2D sim both still
  run on simulator ground truth. If your demo needs "the car sees the
  road through a real camera model," that integration doesn't exist —
  budget time for it or scope it out and say so.

---

## 2. CARLA simulation — how to actually run one

### Setup (one-time, per machine)

CARLA needs its own venv, separate from the repo's `.venv/`:

```bash
python3 -m venv ~/carla-venv --system-site-packages
~/carla-venv/bin/pip install carla==0.9.16
```

`--system-site-packages` matters — `divas` and `carla` must be
importable in **one** process, and this inherits the system's
numpy/scipy/matplotlib instead of fighting a second copy.

Laptop with a hybrid iGPU/dGPU (common on gaming laptops)? Vulkan will
pick the integrated GPU and CARLA will "work" at a few FPS, silently.
**Always launch the server through the provided script, not
`CarlaUE4.sh` directly:**

```bash
./scripts/carla_server.sh
```

### Self-test before building anything

```bash
~/carla-venv/bin/python3 scripts/run_carla.py --check
```

Connects, self-tests the bridge protocol, ticks once, cleans up. ~10
seconds. Run this first on any new machine — if it fails, nothing built
on top of it will work either.

### Running a scenario

```bash
# Stock Town10HD, one stack, one seed
~/carla-venv/bin/python3 scripts/run_carla.py --stack full_dynamic_margin --seed 1

# Real Bengaluru road network instead of Town10
~/carla-venv/bin/python3 scripts/carla_osm_map.py --fetch    # once, downloads the OSM extract
~/carla-venv/bin/python3 scripts/carla_osm_map.py --load     # builds it in a running server
~/carla-venv/bin/python3 scripts/run_carla.py --town <whatever carla_osm_map printed>

# Bazaar / crowd scenario — cattle in the carriageway, a standing crowd
~/carla-venv/bin/python3 scripts/record_carla.py --bazaar --in-crowd \
    --standing 8 --crossing 10 --stack full_dynamic_margin --seed 1

# Whole ablation, one command
~/carla-venv/bin/python3 scripts/run_carla.py --all-stacks --seeds 5
```

### Recording a jury video

```bash
~/carla-venv/bin/python3 scripts/record_carla.py --stack cv_pred_fixed_margin --seed 1
```

Writes chase-cam + stack-view side by side, same encoding convention as
the existing `demo/*.mp4` files (1600 wide, CRF 26 — 8MB, indistinguishable
on a projector, don't re-encode at 1920/CRF20, that's 39MB for nothing).

### Traps already paid for — do not rediscover these

- **CARLA steer is positive right; ours is positive left.**
  `control_from_command` negates, `steer_from_control` un-negates. There's
  a test that fails if this flips back — if it fails, don't "fix" the
  test.
- **Teardown order is load-bearing.** Sensors stopped first, walker
  controllers destroyed before their walkers, one batched destroy with
  the tick cue. Getting it wrong aborts the process from a thread you
  cannot catch. Don't write your own teardown — reuse `CarlaWorld.close()`.
- **`no_rendering_mode` blanks every camera silently.** If you request any
  sensor, it's already overridden off for you — but if you're writing new
  CARLA code from scratch, know this trap exists.
- **CARLA's traffic manager drives politely and historically never
  created real conflict** — baseline and prediction-aware arms performed
  identically on stock Town10HD traffic. The bazaar/crowd scenario may
  have changed this (real Indian road, cattle, dense crowd) — **check
  whether it actually discriminates between stacks before claiming it
  proves the algorithm helps.** If baseline and the proposed stack both
  reach the goal untouched, that's an integration demo, not an ablation
  result — say so, don't blur the two.
- **Town04 segfaults the server** — 6GB VRAM doesn't fit the large maps.
  Stick to Town10HD_Opt or the Bengaluru import for anything on a laptop
  GPU.

---

## 3. MATLAB simulation — how to actually run one

**No toolboxes needed for the existing cross-validation** — base MATLAB
only. `divas_wrap_angle.m` exists specifically so `wrapToPi` (Mapping
Toolbox) isn't required.

```bash
python3 scripts/export_for_matlab.py          # writes sim/matlab/reference/
matlab -batch "cd('sim/matlab'); validate_against_python"
```

This checks that the MATLAB bicycle model and longitudinal controller
agree with the Python ones to floating-point noise — proves the
integrator isn't an artifact of one codebase. **It does not touch the
planner, predictor, risk field, or perception** — none of those are
reimplemented in MATLAB. Don't present it as validating anything beyond
the two models it actually covers.

### A real scenario replay now exists — `sim/MATLABS/`

`sim/MATLABS/replay_animation.m` plays back a real recorded closed-loop
episode (two stacks, same seed, side by side) from the same JSON
`scripts/export_replay.py` already writes for the web replay — not a
re-simulation, a player. No toolboxes.

```bash
python3 scripts/export_replay.py --scenario pedestrian_crossing   # if the JSON isn't already in docs/
matlab -batch "cd('sim/MATLABS'); replay_animation('../../docs/replay-pedestrian_crossing.json', true)"   # true = also write an .avi
```

**Written but not run on real MATLAB — this dev machine has none
installed.** Reviewed carefully, including two MATLAB `jsondecode`
gotchas it works around (mixed-type JSON arrays and objects with
differing field sets both decode to cell arrays, not matrices/struct
arrays — see `idx_any()`/`numel_rows()` in the file). **Run it once on
real MATLAB before trusting it or presenting it** — same rule the team
already applies to `sim/matlab/`'s files. If something breaks, it's
almost certainly in the `jsondecode` assumptions the README documents,
not in the geometry.

### Building an actual MATLAB/Simulink scenario simulation

The replay above visualises a run that already happened — it's not the
same thing as RoadRunner scenario authoring or Simulink control
validation. That (the Automated Driving Toolbox / Model Predictive
Control Toolbox / Navigation Toolbox stuff from `PROJECT_OVERVIEW.md`) is
still real, unstarted work — the toolbox list in the deck implies it, the
repo doesn't have it. Scope this explicitly before promising it in a
pitch; it needs a licence with those toolboxes, none of which the replay
or the base cross-validation need.

RoadRunner specifically (custom Indian-road-texture maps) is a
multi-day asset job, not a script — the CARLA Bengaluru OSM import
above is the faster path to "real Indian road network," just without
RoadRunner's polish.

---

## 4. Before you push

1. `.venv/bin/python -m pytest tests/ -q` (or the CARLA-venv equivalent
   if you touched the bridge) — must stay green.
2. If you changed anything under `divas/`, re-run whatever ablation your
   change could plausibly affect — a simulation change that silently
   shifts closed-loop numbers is worse than an honest "I didn't check."
3. Large files (videos, checkpoints, maps) — check `.gitignore` first.
   `*.pt`, `demo/blender_frames/`, `demo/perception/` are already
   excluded; anything else large and generated should probably join them
   rather than bloat the repo. GitHub's hard limit is 100MB/file — a
   dataset accidentally `git add`-ed wedges the repo irrecoverably (see
   the `*.tar.gz`/`IDD/` exclusions already in `.gitignore` — that
   happened once, don't repeat it).
4. Commit with a message that says what changed and why — match the
   existing style (`git log --oneline -10`), not a generic "update
   sim." Push directly to `main`, same as the rest of this repo's
   history — no branch workflow is in use yet.
5. Update `CONTEXT.md` if what you built changes "what's real vs. stub" —
   that file is the thing the next session (human or Claude) trusts
   first, and it's already gone stale twice this project.
