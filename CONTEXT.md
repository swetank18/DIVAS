# Context — read this first

Working memory for Claude across sessions on DIVAS. Source of truth for
narrative/status is still `STATUS.md`, `EXECUTION_PLAN.md`,
`PROJECT_OVERVIEW.md`, `README.md` — this file is the condensed map so a
fresh session doesn't have to re-read all of them before doing anything
useful. **This file goes stale fast** — a teammate pushed 18 commits
mid-session once already (real IDD model, BEV, conformal margin, Bengaluru
maps). Before trusting anything below about "what exists," run
`git log --oneline -10` and `git status` first.

## What this project is

SIH26037, Team DIVAS. Free-space-centric autonomy stack for unstructured
Indian roads — asks "where can I physically go" instead of "where is my
lane." Six-stage pipeline: perception → BEV projection → fusion →
prediction → planning → control.

## What's real vs. stub, right now

- **Real, working code:** stages 4-6 — prediction (`divas/prediction/`),
  planning (`divas/planning/`), control (`divas/control/`). The 2-D sim
  (`divas/sim/world.py`), the CARLA bridge (`divas/sim/carla_bridge.py`,
  now with real Bengaluru OSM/xodr maps in `docs/maps/`), the eval
  harness (`divas/eval/`).
- **Perception (stage 1) — real, trained, not a stub:**
  `divas/perception/models/drivable.py`'s `DrivableSegmenter` loads
  `drivable_idd.pt`, a checkpoint actually trained on IDD (not
  Cityscapes-pretrained). This is the one true "trained model" in the
  repo. `scripts/train_drivable.py` / `scripts/prepare_idd.py` are the
  training pipeline; `divas/perception/datasets/idd.py` is the real IDD
  loader (my earlier hand-written stand-in for this was superseded and
  removed when this landed).
- **BEV projection (stage 2) — real, no longer a stub:**
  `divas/perception/bev.py` — camera → bird's-eye free-space grid.
- **Fusion (stage 3) — still a ground-truth stub.** No Bayesian occupancy
  update, no EKF tracker built yet.
- **Object detection — real, added by me, additive not overlapping:**
  `divas/perception/detection.py`'s `ObjectDetector` — pretrained
  YOLOv8n/COCO, no training, CPU. Boxes + class + confidence for
  car/truck/bus/motorcycle/bicycle/pedestrian/animal. This is separate
  from the drivable-mask model above and from BEV — outputs **image-pixel
  boxes**, not world-frame tracks. Turning a box into a `divas.types.Track`
  (world x/y/vx/vy) still needs BEV projection to actually be wired
  through it, which hasn't happened yet.
  Demo: `.venv/bin/python scripts/run_perception.py <image.png>` (runs
  segmentation + detection together, writes one overlay). Two honest
  gaps: COCO has no "autorickshaw" class; the two perception models
  (mine, YOLO/COCO detection; teammate's, IDD/drivable segmentation)
  aren't fused into one call yet — separate scripts.
- **Pothole handling — changed this session.** `World.collision()` no
  longer ends the episode on a pothole hit; instead both controllers cap
  speed near one (`POTHOLE_SAFE_SPEED=2.0 m/s`, ramped over 6m — see
  `divas/control/controllers.py`). Verified: every seed that used to
  crash on a pothole now completes, with a measured real slowdown
  (`RunMetrics.pothole_encounters` / `min_speed_in_pothole`). Not wired
  into `CarlaWorld` — it has no scripted potholes, kept optional via
  `getattr` the same way `world.close()` already was.
- **Prediction margin — two mechanisms, both live in the ablation:**
  - **Heuristic** (`full_dynamic_margin`): `d0 + k_v·v + λ·(1-confidence)`,
    where confidence = how much the predictor's own guessed modes disagree
    with each other. Ties the fixed-margin control arm on every dataset
    tested — measures the wrong quantity (self-disagreement, not being
    wrong) per the team's own `ADR-010-conformal-margin.md`.
  - **Conformal** (`cv_pred_conformal_margin`, new): `d0 + k_v·v +
    q_(1-α)(t)`, where `q` is a real measured error percentile from the
    predictor's own recent mistakes, tracked per lookahead step
    (`divas/prediction/conformal.py`). Statistically principled (coverage
    guarantee holds for any predictor), but their own honest finding:
    margin saturates its cap (2.45m of a 2.5m ceiling) at only 85.4%
    actual coverage vs 90% target — the predictor itself is sometimes
    wrong by more than fits on the road. **Not yet proven to reduce
    collisions** — that's the next ablation to run, with the same
    control-arm check that killed the heuristic.
  - Neither margin mechanism is a trained model. Nothing in prediction is
    ML — it's a physics rollout (social-force) plus either a dispersion
    heuristic or a running error percentile. See "why not use the IDD
    model for this" — category mismatch: that model has no concept of
    time or actor identity, it segments one static photo.
- **No ROS 2 node wrappers** (`ros2_ws/src/` empty).
- **Tests:** 121 passed + 3 skipped (skipped need a live CARLA client).
  Run with `.venv/bin/python -m pytest tests/ -q`.

## Environment

System `python3` (Homebrew, 3.14) has no numpy/scipy/matplotlib/pytest and
is PEP-668-locked. A project venv exists at `.venv/` — **always use
`.venv/bin/python`**, not bare `python3`. `.venv` also has
`torch`/`torchvision`/`transformers`/`pillow`/`ultralytics` (CPU wheels,
for the perception modules) — not in any requirements file yet, just
installed directly into `.venv`. `yolov8n.pt` downloads to repo root on
first run of the detector; it's gitignored (`*.pt`), don't commit it.

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/run_ablation.py --seeds 8 --jobs 8
.venv/bin/python scripts/run_perception.py demo/carla_town10_ours.png
.venv/bin/python scripts/make_comparison.py --scenario pedestrian_crossing \
    --left baseline_conventional --right cv_pred_fixed_margin --seed 1
```

CARLA bridge needs `~/carla-venv/bin/python3` instead (separate venv,
`carla==0.9.16`, `--system-site-packages`). See `STATUS.md` for the full
CARLA setup story.

## The ablation picture — what's earned, what isn't

Headline (6 original scripted scenarios): prediction-aware planning cuts
collisions **0.42 → 0.06**, success **0.56 → 0.92**. Solid, reproducible,
lead with it.

**Reactive traffic** (`reactive_overtaking`, `reactive_dense` — actors
that actually yield/overtake, not scripted) was run at 48 seeds
(`eval_results_reactive48.json`, gitignored, local only — rerun if
needed). At that sample size:
- Interaction-aware prediction **did not** clearly beat CV prediction —
  an earlier 16-seed run showed a striking 1.00/0.00 vs 0.94/0.06 gap;
  at 48 seeds it collapsed to a tie (0.96/0.02 both). **That 16-seed
  result was noise, not signal** — don't repeat the claim.
- Dynamic margin (heuristic) tied the fixed-margin control arm again,
  same pattern as the original ablation.
- **Almost every remaining collision was a pothole**, not an actor —
  which is now fixed (see above). **The whole reactive ablation is stale
  and needs re-running** with the pothole fix + conformal margin arm both
  in place to get current honest numbers.

## CARLA — what it does and does NOT prove

CARLA bridge works (drives full stack, no collision, ~51.6ms p95 e2e
latency) on Town10HD. **Historically, CARLA's traffic manager drove
politely and never created real conflict** — baseline and prediction-aware
arms performed identically there. **This may have changed**: the team
added a bazaar/crowd scenario with cattle (`docs/replay-cattle_and_crowd.json`,
`demo/carla_bazaar_crowd.mp4`, `demo/compare_cattle_and_crowd_*.mp4`) on
real Bengaluru roads — check whether that scenario actually discriminates
between arms before assuming the old "CARLA doesn't prove the algorithm"
caveat still fully applies. Verify, don't assume either way.

## Known real gaps (say these out loud, don't let a judge find them)

- Fusion (stage 3) still a ground-truth stub — no real EKF tracker.
- Object detection outputs pixel boxes, not world-frame tracks — BEV
  projection exists but isn't wired through the detector yet.
- No yielding/right-of-way logic — dense traffic collided ~50% of the
  time at junctions in the last measurement (STATUS.md §7) — may be
  affected by the pothole fix, not re-measured since.
- Conformal margin not yet proven to reduce collisions (see above).
- COCO detector has no "autorickshaw" class.
- Planner can't reverse — no parking.

## Traps already paid for — do not re-introduce

Ten ADRs in `docs/decisions/` now (ADR-010 added — conformal margin).
Expensive ones: horizon consistency (ADR-006), never benchmark on
wall-clock time (ADR-008), predictor obstacle-force made prediction
*worse* than CV until disabled (ADR-009), heuristic margin measures the
wrong quantity (ADR-010). Also: CARLA steer sign flipped vs. ours, CARLA
teardown order load-bearing, Vulkan picks the integrated GPU unless
launched through `scripts/carla_server.sh`, `no_rendering_mode` silently
blanks every camera.

## Code navigation

A knowledge graph of this repo exists in `graphify-out/` (gitignored,
local only — rerun `/graphify --update` if it's gone or stale) —
**use it before grepping.** See `CLAUDE.md` for the tools and workflow.

## Submission blockers (independent of code)

Faculty/industry mentor rows on the deck, two malformed registration
numbers — check `scripts/fill_deck_mentors.py` and `STATUS.md` for
current status; may already be resolved given how much else landed.
