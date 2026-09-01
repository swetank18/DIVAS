# Execution Plan — SIH26037 / Team DIVAS

Working plan for building the stack described in `PROJECT_OVERVIEW.md`.
Status at time of writing: **repository is empty except the deck.** This is greenfield.

---

## 0. Guiding strategy

**Build a thin vertical slice before building any one module well.**

The failure mode for a six-person, six-stage pipeline is six excellent modules that have never run together, integrated in the final 48 hours. The antidote:

1. Get a **closed loop running end-to-end in week 1** using ground-truth stubs for every stage the simulator can fake (perfect segmentation, perfect tracks, perfect prediction).
2. Then replace one stub at a time with the real module, keeping the loop green after every swap.
3. Every stage boundary is a ROS 2 topic with a fixed message type, so a stub and the real thing are interchangeable with zero code changes downstream.

The demo is always runnable. Quality goes up monotonically. Nothing has to be integrated under deadline pressure.

**Second principle: measure from day one.** The metrics harness (§4, Phase 1) is built *before* the modules it measures. A judge's decisive question is "how much better than the baseline?" — the answer must be a table, not an adjective.

---

## 1. Repository layout

```
SIH/
├── README.md                    # entry point
├── PROJECT_OVERVIEW.md
├── EXECUTION_PLAN.md
├── deck/                        # pptx + exported slide assets
├── docs/decisions/              # ADRs — one file per architectural decision
├── msgs/                        # ROS 2 interface definitions (the contracts)
├── divas/                       # the importable package
│   ├── types.py                 # the stage contracts
│   ├── perception/datasets/     # IDD loaders, drivable-class remapping
│   ├── fusion/                  # BEV projection, occupancy grid, EKF tracker
│   ├── prediction/              # social-force + CV baseline, risk field
│   ├── planning/                # Hybrid A*, RRT* fallback
│   ├── control/                 # pure pursuit, risk-aware MPC
│   ├── sim/                     # 2-D world serving ground-truth stubs
│   └── eval/                    # metrics, scenarios, closed-loop runner
├── scripts/                     # run_demo, run_ablation, verify_idd
├── tests/
├── sim/{carla,matlab}/          # bridges, once decision #1 lands
├── ros2_ws/src/                 # thin node wrappers around divas/
└── rover/                       # Jetson deployment, drivers, calibration
```

The algorithm modules live under `divas/` as a package rather than as top-level
directories: `import planning` would collide with half of PyPI, and a package
keeps every module unit-testable without a ROS environment, which matters when
six people are on six machines.


Keep algorithm code **outside** `ros2_ws/`. ROS nodes are thin wrappers. This keeps every module unit-testable without a ROS environment, which matters when six people are on six machines.

---

## 2. Stage contracts (define these first — Phase 0)

These message definitions are the project's spine. Freeze them early; change them only via an ADR in `docs/decisions/`.

| Topic | Type | Rate | Producer → Consumer |
|---|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | 30 Hz | sensor → perception |
| `/perception/drivable_mask` | `sensor_msgs/Image` (mono8) | 15–30 Hz | perception → projection |
| `/fusion/occupancy_grid` | `nav_msgs/OccupancyGrid` (40×20 m @ 0.2 m) | 10 Hz | fusion → planning |
| `/fusion/tracks` | `divas_msgs/TrackArray` `{id, x, y, vx, vy, class, cov}` | 10 Hz | fusion → prediction |
| `/prediction/trajectories` | `divas_msgs/TrajectorySet` (3 s @ 0.1 s, per-mode probability) | 10 Hz | prediction → planning, control |
| `/planning/path` | `nav_msgs/Path` + curvature | 2–5 Hz | planning → control |
| `/control/cmd` | `ackermann_msgs/AckermannDrive` | 20–50 Hz | control → vehicle |

Every one of these gets a **ground-truth stub publisher** driven from simulator state. That is what makes the vertical slice possible.

---

## 3. Open decisions — needed before or during Phase 0

**Blocking (I cannot pick these for you):**

| # | Decision | Why it blocks |
|---|---|---|
| 1 | **CARLA-primary or MATLAB-primary simulation?** | Determines the entire sim harness. My recommendation: **CARLA for perception-in-the-loop and closed-loop scenarios, MATLAB/Simulink for control validation and the RoadRunner scenario library.** But if SIH26037's sponsoring organisation is MathWorks (the deck's toolbox list strongly suggests it), judges will expect MATLAB deliverables — check the PS text in the official catalogue. |
| 2 | **Hardware budget: is there a physical rover, or is this simulation-only?** | Phase 6 exists or does not. Also determines whether sensor drivers and calibration are on the critical path. |
| 3 | **Actual deadline date.** | Phase durations below are relative; they need anchoring. |
| 4 | **Faculty mentor + industry mentor.** | Submission blocker, independent of the build (see `PROJECT_OVERVIEW.md` §8). |

**Non-blocking (I will proceed on these defaults, override any time):** Python-first for all modules with C++ ports only where profiling proves it necessary · acados for MPC · social-force prediction baseline before any learned model · PyTorch for segmentation.

---

## 4. Phases

Durations are working estimates for a six-person team; compress or stretch once decision #3 lands. Each phase has a **Definition of Done** that is binary — no partial credit.

### Phase 0 — Ground truth and de-risking (2–3 days) — **partly done**

Verify assumptions before building on them.

*Done:* stage contracts frozen and committed (`divas/types.py`, `msgs/`); IDD
loader, drivable-area remapping and a `verify` report written
(`scripts/verify_idd.py`) — it runs against a real IDD download and reports
whether the label-id mapping is plausible. *Outstanding:* everything that needs
the dataset, the hardware, or decision #1.

- [ ] Download IDD; confirm license terms; count images with a usable drivable-area label; write the class-remapping to a binary drivable mask
- [ ] **Determine whether a temporal/multimodal IDD variant is actually obtainable.** If not, prediction and fusion data come from simulation — decide now, not in week 3
- [ ] Read the official SIH26037 problem statement text in the PS catalogue; note every mandated deliverable and tool
- [ ] Stand up CARLA and/or MATLAB per decision #1; confirm every team member can run it
- [ ] Benchmark the target compute (Jetson or whatever is available) on a reference segmentation model — get a real ms/frame number
- [ ] Specify the exact LiDAR and radar part numbers (range, rate, FOV) if hardware is in scope
- [ ] Freeze the §2 stage contracts; commit `msgs/`

**DoD:** Every assumption in `PROJECT_OVERVIEW.md` §4 is either confirmed or has a written fallback in `docs/decisions/`.

### Phase 1 — Vertical slice with stubs (4–6 days) — **done**

The whole loop runs end to end. `python3 -m pytest tests/ -q` — 22 tests.
Measured latencies: predict ~20 ms p95, plan 13–120 ms p95, control ~20 ms p95.

- [ ] ROS 2 workspace scaffolded; all seven topics from §2 exist
- [ ] Simulator publishes ground-truth occupancy grid, tracks, and ego state as stubs
- [ ] Constant-velocity prediction stub
- [ ] Hybrid A\* planner, Python, working over the stub grid
- [ ] Pure-pursuit controller (simplest thing that closes the loop — MPC comes later)
- [ ] **`eval/` metrics harness**: success rate, collision rate, min TTC, max lateral accel, max jerk, per-stage latency — logged automatically per run
- [ ] Five scenario definitions: unmarked road following, pothole avoidance, two-wheeler cut-in, pedestrian crossing, oncoming-vehicle negotiation on a shared carriageway

**DoD — met.** `python3 scripts/run_demo.py --scenario two_wheeler_cutin` drives
goal-to-goal (110 m, no collision, 8.4 m/s mean), and `scripts/run_ablation.py`
emits the metrics table. Delivered beyond the DoD: the risk-aware MPC of Phase 4
and the six-scenario suite of Phase 5 are already in place, because the ablation
needed both arms to compare.

Ablation, 6 scenarios x 8 seeds = **48 runs per arm**, iteration-bounded so the
numbers are reproducible ([ADR-008](docs/decisions/ADR-008-deterministic-evaluation.md)):

| stack | success | collision | timeout | progress | v_mean |
|---|---|---|---|---|---|
| baseline: CV prediction + fixed 1.0 m + pure pursuit | 0.56 | **0.42** | 0.02 | 90.0 m | 6.67 |
| MPC, no prediction | 0.65 | 0.33 | 0.02 | 97.4 m | 7.99 |
| MPC + CV prediction + fixed margin | **0.92** | **0.06** | 0.02 | 108.7 m | 7.81 |
| MPC + interaction-aware prediction + fixed margin | 0.90 | 0.08 | 0.02 | 108.0 m | 7.86 |
| control arm: fixed 1.4 m margin (matches dynamic mean) | 0.90 | 0.08 | 0.02 | 108.0 m | 7.83 |
| full: + confidence-scaled dynamic margin | 0.90 | 0.08 | 0.02 | 107.9 m | 7.85 |

**What the data supports.** Prediction is the whole story: collisions fall
**0.42 to 0.06** and success rises **0.56 to 0.92**. At 48 runs per arm that is
20 collisions versus 3 — a large, robust effect, and it is the claim the deck
should lead with.

**What the data does not yet support.** The bottom four rows are
indistinguishable (0.90-0.92 success, 3-4 collisions out of 48):

- *Interaction-aware prediction* shows no measurable gain over constant
  velocity. That is not a refutation — this benchmark **cannot test it**,
  because the scenario actors follow scripted policies and never react to one
  another. Fixing the benchmark is the top task below.
- *The confidence-scaled dynamic margin* shows no gain over a fixed margin, and
  the control arm proves why: a fixed 1.4 m margin matching the dynamic arm's
  mean performs identically. The variation is contributing nothing beyond its
  average. The underlying confidence signal *is* sound -- error by confidence
  quintile is 7.01, 5.59, 4.47, 3.90, 2.66 m -- so the mechanism has a basis,
  but the scenarios do not yet stress it.

Until that changes, the deck should claim the free-space + prediction + risk-aware
planning core, which is measured and holds, and describe the interaction model and
dynamic margin as implemented and instrumented rather than as demonstrated wins.

**Top task, ahead of any further tuning: give the scenario actors reactive
policies** (yield, car-following, gap acceptance). Two of the three novelty
claims cannot be evaluated at all until traffic in the suite actually interacts.

Two caveats to carry forward, both real:
- The ROS 2 node wrappers are not written. The runner drives the stages directly,
  at the correct separate rates; `ros2_ws/src/` is still empty.

This is also the **fallback demo** for the entire project. From here, everything is improvement rather than risk.

### Phase 2 — Real perception (5–7 days)

- [ ] IDD data pipeline + augmentation
- [ ] Train drivable-area segmentation; report mIoU on the drivable class
- [ ] Inverse-perspective mapping to the BEV free-space grid; validate against simulator ground truth
- [ ] Optimize for edge: prune, quantize, TensorRT export; report ms/frame on target hardware
- [ ] **Swap the perception stub for the real model.** Loop stays green.

**DoD:** Pipeline runs on camera images with no ground-truth free space, and the Phase 1 metrics have not regressed catastrophically.

### Phase 3 — Fusion and prediction (5–7 days)

- [ ] Bayesian occupancy-grid update fusing camera BEV + LiDAR + radar
- [ ] EKF multi-object tracker (constant-turn-rate model), with association and track lifecycle
- [ ] Degraded-sensor test: kill the camera and confirm radar/LiDAR fallback holds the loop up — this is a *demo moment*, script it
- [ ] Social-force interaction-aware prediction, multi-modal with per-mode probability
- [ ] Optional if time and data allow: GRU trajectory model; keep social-force as the fallback
- [ ] **Swap both stubs.** Loop stays green.

**DoD:** Prediction publishes a real `TrajectorySet` with calibrated probabilities; ADE/FDE reported against held-out trajectories.

### Phase 4 — Risk-aware MPC (5–7 days) — *the novelty phase* — **partly done**

*Done:* the MPC exists (`divas/control/controllers.py`), the margin equation is
implemented and tunable, and the ablation runs. *The remaining work is tuning,
and it is the highest-value work left in the project* — see the caveat under
Phase 1. The current ablation shows collisions falling 0.33 → 0.06 as prediction
is added, then the last two arms regressing into timeouts. Fixing that
conservatism is what turns the table into a result.

- [x] Kinematic bicycle model MPC: states `[x, y, θ, v]`, inputs `[a, δ]`, N = 20, dt = 0.1 s (2 s horizon)
- [ ] Cost: path tracking + control effort + input slew + obstacle barrier
- [x] Constraints: `a ∈ [-4, 2] m/s²`, `|δ| ≤ 0.5 rad`, `|δ̇| ≤ 0.3 rad/s`, per-actor ellipsoidal keep-out ([ADR-003](docs/decisions/ADR-003-elliptical-keepouts.md))
- [x] Solver: sampling MPC (MPPI), warm-started, fixed horizon — see [ADR-002](docs/decisions/ADR-002-mpc-solver.md) for why not acados, and the migration path back to it
- [x] **Implement `d_safe(t) = d₀ + k·v_ego + λ·(1 − confidence(t))`**  — [ ] *tune* `d₀`, `k`, `λ` against the timeout rate
- [ ] Validate against a CasADi/IPOPT offline reference for optimality gap
- [ ] Replace pure-pursuit. Handle late replans safely (track last valid path, degrade to controlled stop)

**DoD:** MPC solves within the control period on target hardware, and an **ablation table** shows fixed-margin vs. dynamic-margin performance on the scenario suite. This table is the single most persuasive artifact in the submission.

### Phase 5 — Validation and evidence (4–5 days)

- [ ] Expand to 15–20 scenarios incl. rain/dust/low-light and rare edge cases
- [ ] **Run the full ablation:** lane-detection + constant-velocity baseline · ours minus interaction-aware prediction · ours minus dynamic margin · full stack. Four rows, one table
- [ ] MATLAB/Simulink control validation; RoadRunner scenes for scenario realism
- [ ] Per-stage latency budget with measured numbers; end-to-end reaction time
- [ ] Generate demo video assets from the best runs

**DoD:** Every quantitative claim in the pitch traces to a logged run in `eval/`.

### Phase 6 — Rover prototype (5–7 days, conditional on decision #2)

- [ ] Jetson provisioned; sensor drivers; camera-LiDAR extrinsic calibration; time sync
- [ ] Deploy the *same* ROS 2 stack — no fork, no rewrite. If the code diverges here, the sim results stop backing the hardware claim
- [ ] Controlled course: cones as obstacles, a person walking across, an unmarked path
- [ ] Record the demo video

**DoD:** Rover completes the course autonomously on camera.

### Phase 7 — Submission polish (2–3 days)

- [ ] Fix all six deck issues in `PROJECT_OVERVIEW.md` §8 — **start with the empty mandatory mentor rows**
- [ ] Rebuild the methodology diagram with the prediction-uncertainty feedback edge
- [ ] Rewrite slide 6 as proper numbered citations
- [ ] Add the ablation table to the deck — it likely replaces a paragraph of prose
- [ ] Rehearse: 3-minute pitch, 2-minute demo, prepared answers on real-time feasibility, IDD generalization, and why this is not "just another AV project"

### Jury demo — what to show, ranked by risk

`scripts/make_comparison.py` renders two stacks on **identical traffic from an
identical seed**, as an MP4 and a slide PNG. Runs are bit-reproducible
(ADR-008), so what you rehearse is exactly what the jury sees.

`--find` scans scenarios and seeds for the clearest contrast. Its answer:

| case | baseline | proposed |
|---|---|---|
| **`pedestrian_crossing`, any seed** | **hits a pedestrian at 29 m** | reaches the goal, 110 m |
| `pothole_slalom` seed 1 | hits a pothole | reaches the goal |
| `mixed_traffic` seeds 0–3 | hits an autorickshaw at 91 m | reaches the goal, 120 m |

Lead with **`pedestrian_crossing`**. The baseline kills a pedestrian and the
proposed stack does not, on every seed tried — and pedestrians are 20.4% of
India's road deaths, which is already the strongest number in the deck. The
demo and the motivation are the same argument.

1. **Zero risk, always have it:** the pre-rendered MP4.
2. **Low risk, high impact:** run it live from the CLI. A judge picks the
   scenario; it runs on a laptop with no GPU. Judges remember the team that
   let them poke at it.
3. **The decisive slide:** the ablation table. Every panel asks "how much
   better than existing solutions" and most teams cannot answer with a number.
4. **Medium risk:** CARLA live, once decision #1 lands. Needs the GPU laptop.
5. **Highest risk:** the rover. Record the video first; treat the live run as
   a bonus.

Show `cv_pred_fixed_margin` as "our stack". It is the best-performing arm, and
it is still the project's actual thesis — free-space planning with prediction-
aware risk. It simply omits the two additions that have not yet earned their
place. Claiming the arm that measures best is honest; claiming the one with the
most machinery in it is not.

---

## 5. Suggested ownership

Six people, six stages, plus integration. Adjust to actual skills and interest — this is a starting proposal, not an assignment.

| Person | Primary | Secondary |
|---|---|---|
| **Rijul Rawal** (lead, ECE) | Integration, ROS 2 architecture, demo, deck | Interface discipline; owns "does the loop still run?" |
| **Akshat Gupta** (CINTEL, 3rd yr) | Planning — Hybrid A\*, RRT\* fallback | Most senior; also reviews the MPC |
| **Sidhant Sinha** (CINTEL) | Perception — segmentation, IDD training, edge optimization | BEV projection |
| **Sri Priya CH** (ECE) | Sensors and fusion — occupancy grid, EKF, calibration | Rover hardware in Phase 6 |
| **Swetank Kumar** (CINTEL) | Prediction — social-force, trajectory model, dataset pipeline | Data tooling |
| **Aneek Sen** (CINTEL) | Simulation, scenario suite, `eval/` metrics harness | MATLAB/Simulink validation |

**Control/MPC** is the hardest single module — pair Akshat with whoever finishes their phase first, rather than assigning it to one person alone.

Non-negotiable process rule: **nobody merges a module that breaks the vertical slice.** The loop stays green.

---

## 6. What I do next

Phases 0 and 1 are built on the simulator-agnostic path, so decisions #1–#3 are
still open and still yours. In priority order:

1. **Give the scenario actors reactive policies** (yield, car-following, gap
   acceptance). Interaction-aware prediction and the dynamic margin are both
   unevaluable against scripted traffic, so this gates two of the three novelty
   claims. Highest value by a distance.
2. **Write the ROS 2 node wrappers** so the rates are enforced by the middleware
   rather than by the runner's loop counter.
3. **Measure real-time performance properly** with the deployment clock budget
   enabled, ideally on Jetson-class hardware. The evaluation numbers are
   iteration-bounded and are deliberately *not* a real-time claim (ADR-008).
4. **Perception (Phase 2)**, once IDD is downloaded and `scripts/verify_idd.py`
   confirms the label mapping.

Still blocked on you: **#1 simulator** (CARLA vs MATLAB — check whether SIH26037
is MathWorks-sponsored), **#2 hardware** (does Phase 6 exist), **#3 deadline**.
**#4 mentors** remains a submission blocker independent of the build.
