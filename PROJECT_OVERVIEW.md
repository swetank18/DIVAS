# SIH26037 — Adaptive Path Planning & Collision Avoidance for Unstructured Indian Roads

**Team DIVAS · SRMIST · Smart India Hackathon 2026 · Theme: Smart Vehicles · Category: Software**

---

## 1. The crux, in one sentence

> Every mainstream autonomy stack is **lane-centric** — it asks *"where is my lane?"*. On Indian roads there often is no lane. We build a **free-space-centric** stack that asks *"where can I physically go, and who is about to get in the way?"*

Everything else in this project is a consequence of that single inversion.

---

## 2. Why the standard stack breaks here

The deck's five failure modes, restated as engineering mismatches:

| Assumption baked into Western AV stacks | Reality on Indian roads | Our replacement |
|---|---|---|
| Lane markings exist and are visible | Faded, absent, or ignored | **Drivable-area semantic segmentation** — a binary free-space mask, no lane hypothesis |
| A lane graph / HD map defines legal routes | No lane graph; traffic flows as a fluid | **Hybrid A\* / RRT\*** search over an open occupancy grid |
| Agents move at roughly constant velocity in their lane | Auto-rickshaws, two-wheelers, pedestrians, cattle — erratic, interactive, non-lane-bound | **Interaction-aware prediction** (social-force or GRU), multi-modal with uncertainty |
| Road surface is uniform and flat | Potholes, speed breakers, unmarked humps, debris | **Camera + radar + low-cost LiDAR fusion** for redundant geometric detection |
| A fixed safety buffer is enough | Cut-ins happen with sub-second warning | **Confidence-modulated dynamic safety margin** inside the MPC |

Training data matters as much as architecture: KITTI / Cityscapes / nuScenes encode Western road structure. We train on the **India Driving Dataset (IDD, IIIT-Hyderabad + Intel)**, which encodes ours.

---

## 3. Architecture — the six stages and the contract between them

The single most important discipline in this project is that **every stage has a typed, testable interface**. Six people cannot build one pipeline without this. Stage boundaries are ROS 2 topics; each can be swapped for a ground-truth stub during development.

```
 [RGB camera] ──▶ 1. PERCEPTION ──▶ 2. BEV PROJECTION ──▶ 3. FUSION ──┐
 [radar]      ─────────────────────────────────────────────────────────┤
 [2.5D LiDAR] ─────────────────────────────────────────────────────────┤
                                                                       ▼
        6. CONTROL ◀── 5. PLANNING ◀── 4. PREDICTION ◀── occupancy + tracks
             │
             ▼
      (steer, accel) ──▶ vehicle / rover / sim
```

**1. Perception** — `Image(H×W×3)` → `DrivableMask(H×W, binary)` + `Detections[]`
Semantic segmentation network trained on IDD. Output is *free space*, not lanes. Runs at camera rate.

**2. BEV projection** — `DrivableMask` + camera intrinsics/extrinsics → `FreeSpaceGrid`
Inverse-perspective mapping onto a bird's-eye grid. Working spec: a **64 × 64 m rolling window at 0.25 m/cell, ego-centred but world-aligned** — it translates with the vehicle and does not rotate with it. See [ADR-001](docs/decisions/ADR-001-local-costmap-frame.md) for why the original ego-frame spec was replaced.

**3. Fusion** — `FreeSpaceGrid ⊕ RadarPoints ⊕ LidarScan` → `OccupancyGrid(p_occ per cell)` + `Tracks[]`
Bayesian grid update for static geometry; EKF (constant-turn-rate model) for dynamic object tracks. Each track carries `{id, x, y, vx, vy, class, covariance}`. **Radar is the rain/dust/glare fallback path** — this is why the sensor set is redundant rather than merely additive.

**4. Prediction** — `Tracks[]` → `TrajectorySet[]` over a **3 s horizon @ 0.1 s**, each with a probability
Interaction-aware, so a predicted two-wheeler path bends around other agents instead of extrapolating straight. Output is consumed as a **time-indexed risk field**, not as hard obstacles — that distinction is what lets the controller be brave when confident and timid when not.

**5. Planning** — `OccupancyGrid` + `RiskField` + `Goal` → `Path[(x, y, θ, κ)]`, replanned at **2–5 Hz**
Hybrid A\* over the grid with kinematic feasibility built into the motion primitives (no post-hoc smoothing of an infeasible A\* path). RRT\* as the fallback when Hybrid A\* fails to find a solution in budget.

**6. Control** — `Path` + `EgoState` + `RiskField` → `(δ, a)` at **20–50 Hz**
Nonlinear MPC on a kinematic bicycle model. **This is where the project's actual novelty lives**, and it must be one explicit, tunable equation, not a hand-wave:

```
d_safe(t) = d₀ + k·v_ego + λ·(1 − confidence(t))
```

The safety buffer around each predicted actor *widens automatically* when the prediction module is uncertain and *tightens* when it is confident. It is implemented in `divas/prediction/risk.py`; current coefficients are `d₀ = 0.4 m`, `k = 0.10 s`, `λ = 0.8 m`, capped at 1.8 m. Setting `λ = 0` recovers the conventional fixed policy, which is exactly the ablation.

`confidence(t)` is defined as the **spatial dispersion of the prediction's modes**, not the entropy of their probabilities — see [ADR-004](docs/decisions/ADR-004-confidence-definition.md). Three equally likely modes that predict nearly the same path describe a certain future; entropy would score them as maximally uncertain. What the controller needs is how many *metres* the prediction could be wrong by.

---

## 4. Technology stack

| Layer | Choice | Note |
|---|---|---|
| Perception / ML | Python, PyTorch, OpenCV | Segmentation + prediction models |
| Real-time loop | C++ | Planner and controller, once Python prototype is validated |
| Middleware | ROS 2 | Stage boundaries = topics; enables the stub-swap discipline |
| Planning | Hybrid A\* / RRT\* (custom or OMPL) | |
| Control | **Sampling MPC (MPPI)** today; acados/OSQP is the hardware path — see [ADR-002](docs/decisions/ADR-002-mpc-solver.md) | CasADi/IPOPT for offline validation only |
| Simulation | CARLA (perception-in-the-loop) + MATLAB/Simulink, Automated Driving Toolbox, RoadRunner (scenarios, control validation) | See open decision #1 |
| Dataset | India Driving Dataset (IDD) | |
| Hardware | Mono/stereo camera, low-cost 2.5D LiDAR, short-range radar/ultrasonic, NVIDIA Jetson-class edge GPU | |

### Reality checks on the deck's stack

These are corrections worth making now rather than discovering in week 4:

1. **CasADi + IPOPT will not hit 20 Hz on a Jetson.** IPOPT is an interior-point solver built for offline optimality, not real-time embedded control. Use **acados** (SQP-RTI) or a linearized bicycle model with **OSQP**. Keep CasADi/IPOPT as the offline reference to validate that the fast solver is producing near-optimal solutions.
2. **IDD has no radar and no ego-motion, and its main release is single-image.** It is excellent for segmentation and detection; it does **not** by itself train a prediction model or validate sensor fusion. Multimodal/temporal IDD variants exist — confirm availability and licensing in Phase 0. Otherwise: train perception on IDD, and generate fusion + prediction data in simulation.
3. **"Low-cost 2.5D LiDAR"** almost certainly means a 2D scanner (e.g. RPLIDAR-class) tilted or on a nodding mount. Specify the exact part before writing the driver — the plan hinges on its actual range, rate, and FOV.
4. **Planner and controller run at different rates.** 2–5 Hz replanning against 20–50 Hz control is a real architectural constraint: the controller must track the *last valid* path and degrade safely when a replan is late. Design the failure behaviour explicitly.

---

## 5. Scope — what we demo vs. what we claim

The deck is already honest about this, and that honesty is a strength. Keep it sharp:

**In scope (demonstrable):**
- Full six-stage pipeline running closed-loop in simulation on Indian-style unstructured scenarios
- Trained IDD segmentation model with reported mIoU on the drivable class
- Quantified ablation: our stack vs. a lane-following + constant-velocity baseline
- Small-scale rover executing the same ROS 2 stack on a controlled indoor/campus course

**Explicitly out of scope:**
- A real autonomous vehicle on public roads. Say this out loud in the pitch. Claiming it invites a question you cannot win; disclaiming it makes every other claim more credible.

---

## 6. Impact narrative

Grounded in the figures already in the deck:

- 480,583 road accidents and 172,890 deaths in India (2023)
- Two-wheeler riders 44.8% and pedestrians 20.4% of road deaths — the **exact** road users a lane-centric stack handles worst and a free-space + interaction-aware stack handles best
- 83%+ of fatalities in the 18–60 working-age group
- Overspeeding implicated in ~68% of accident deaths — directly addressed by the enforced dynamic safety margin

Benefits: **social** (protects highest-risk users), **environmental** (smoother planning → less harsh braking → lower fuel and emissions), **economic** (healthcare, insurance, productivity losses avoided), **scalability** (low-cost sensor stack pushes safety below the premium-vehicle price point), **policy** (UN Decade of Action target to halve road deaths by 2030).

The strongest version of this argument is the second bullet: the demographic most killed on Indian roads is precisely the one Western AV stacks are worst at seeing. That is not a coincidence, and it is the core of the pitch.

---

## 7. Known gaps and risks

Carried from the deck's own challenge/strategy mapping, plus what analysis surfaces:

| Risk | Severity | Mitigation |
|---|---|---|
| Real-time inference latency on edge GPU | High | Lightweight/pruned models, TensorRT export, measured budget per stage |
| Fusion accuracy in rain / dust / glare | High | Sensor redundancy with radar/LiDAR fallback path |
| **No training data for the prediction module** | **High** | Simulation-generated trajectories, or a physics-based social-force model needing no training |
| **MPC solver too slow for real-time** | **High** | acados/OSQP, fixed horizon, warm-started solver |
| Limited labeled edge-case data | Medium | Synthetic augmentation of rare scenes |
| Generalization failure across regions | Medium | Broader IDD sampling, domain mixing |
| Hardware cost/integration within timeline | Medium | Simulation-first; rover is a bonus deliverable, not a dependency |
| **Six people, one pipeline, no interface discipline** | **High** | Typed stage contracts + ground-truth stubs from day one (§3) |

The two bolded rows the deck does not yet name — prediction training data and MPC solver speed — are the ones most likely to actually derail the build.

---

## 8. Deck issues to fix before submission

Found while analysing `SRMIST SIH2026-TEAM DIVAS.pptx`:

1. **Slide 1 says "SMART INDIA HACKATHON 2025"** — the PS ID is SIH26037 and the filename says SIH2026. Fix the year.
2. **Slide 7: Faculty Mentor and Industry Mentor rows are completely empty** and both are marked *(Mandatory)*. This is a hard submission blocker, not a polish item. Resolve first.
3. **Two registration numbers look malformed.** Four members have 15-character reg numbers (`RA2511043010009`); *Swetank Kumar* has `RA25033010134` (13) and *Aneek Sen* has `RA251102601641` (14). Submission portals validate this field — verify against ID cards.
4. **Slide 6 is a raw URL dump.** Convert to numbered citations with paper titles and authors. It currently reads as unreviewed.
5. **Slide 5 has zero editable text** — the entire impact argument is baked into images. If a judge or the portal needs text, it is unreachable. Also, one embedded clipping is Goa-specific road-accident data while the pitch argues national scale; either drop it or relabel it as an illustrative regional case.
6. **The methodology diagram (slide 3) shows a clean linear flow.** Add the feedback edge — prediction uncertainty feeding the controller's safety margin — because that loop *is* the novelty and the current diagram hides it.

---

*Companion document: `EXECUTION_PLAN.md` — phased build plan, repo layout, ownership, and open decisions.*
