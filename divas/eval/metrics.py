"""Run metrics.

Built before the modules they measure, deliberately.  The question that
decides this project is "how much better than the baseline?", and that has to
be answered with a table.  Anything not recorded here cannot be claimed in the
deck.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class StageTimer:
    """Latency accumulator for one pipeline stage."""

    samples: List[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples.append(float(ms))

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self.samples, 95)) if self.samples else 0.0

    @property
    def max(self) -> float:
        return float(np.max(self.samples)) if self.samples else 0.0


@dataclass
class RunMetrics:
    """Everything one closed-loop run produces."""

    scenario: str = ""
    stack: str = ""
    seed: int = 0

    success: bool = False
    collision: bool = False
    collision_with: str = ""
    timeout: bool = False

    sim_time: float = 0.0
    distance: float = 0.0
    progress: float = 0.0

    min_ttc: float = float("inf")
    ttc_samples: List[float] = field(default_factory=list)
    min_clearance: float = float("inf")

    max_lat_accel: float = 0.0
    max_jerk: float = 0.0
    max_speed: float = 0.0
    mean_speed: float = 0.0

    plan_calls: int = 0
    plan_success: int = 0
    plan_partial: int = 0
    mean_d_safe: float = 0.0

    timers: Dict[str, StageTimer] = field(default_factory=dict)
    trace: Optional[dict] = None

    #: Realised coverage of the conformal margin, or None for the arms that
    #: do not use one. Reported next to the collision rate on purpose: an arm
    #: that hits its coverage target and still collides has a calibrated
    #: margin and a planning problem, which is a different bug from an
    #: uncalibrated margin.
    conformal_coverage: Optional[float] = None
    conformal_alpha: Optional[float] = None
    conformal_scored: int = 0

    def timer(self, name: str) -> StageTimer:
        return self.timers.setdefault(name, StageTimer())

    # -- derived ----------------------------------------------------------
    @property
    def ttc_p5(self) -> float:
        """5th-percentile TTC.  The minimum alone hides how *often* we were
        close, which is what actually distinguishes a safe policy from a lucky
        one."""
        v = [t for t in self.ttc_samples if np.isfinite(t)]
        return float(np.percentile(v, 5)) if v else float("inf")

    @property
    def plan_success_rate(self) -> float:
        return self.plan_success / self.plan_calls if self.plan_calls else 0.0

    @property
    def end_to_end_p95(self) -> float:
        """Worst-case reaction latency through the whole pipeline."""
        return sum(t.p95 for t in self.timers.values())

    def as_row(self) -> dict:
        return {
            "scenario": self.scenario,
            "stack": self.stack,
            "seed": self.seed,
            "success": self.success,
            "collision": self.collision,
            "collision_with": self.collision_with,
            "timeout": self.timeout,
            "sim_time": round(self.sim_time, 2),
            "progress_m": round(self.progress, 1),
            "min_ttc": round(self.min_ttc, 2) if np.isfinite(self.min_ttc) else None,
            "ttc_p5": round(self.ttc_p5, 2) if np.isfinite(self.ttc_p5) else None,
            "min_clearance_m": round(self.min_clearance, 2)
            if np.isfinite(self.min_clearance) else None,
            "max_lat_accel": round(self.max_lat_accel, 2),
            "max_jerk": round(self.max_jerk, 2),
            "mean_speed": round(self.mean_speed, 2),
            "mean_d_safe_m": round(self.mean_d_safe, 2),
            "plan_success_rate": round(self.plan_success_rate, 3),
            "predict_ms_p95": round(self.timer("predict").p95, 1),
            "plan_ms_p95": round(self.timer("plan").p95, 1),
            "control_ms_p95": round(self.timer("control").p95, 1),
            "e2e_ms_p95": round(self.end_to_end_p95, 1),
        }


def aggregate(runs: List[RunMetrics]) -> dict:
    """Summarise a set of runs -- one row of the ablation table."""
    if not runs:
        return {}
    n = len(runs)
    finite_ttc = [r.min_ttc for r in runs if np.isfinite(r.min_ttc)]
    # Only runs that actually had traffic.  Averaging the obstacle-only
    # scenarios in drags every stack's margin towards zero and flattens the
    # fixed-versus-dynamic comparison, which is the one number this table
    # exists to produce.
    with_traffic = [r.mean_d_safe for r in runs if r.mean_d_safe > 0.0]
    # TTC on the runs that did not end in a collision: a collision drives the
    # minimum to zero by definition, so including it says nothing about how
    # close the surviving runs came.
    clean_ttc = [r.min_ttc for r in runs if not r.collision and np.isfinite(r.min_ttc)]
    finite_clr = [r.min_clearance for r in runs if np.isfinite(r.min_clearance)]
    return {
        "stack": runs[0].stack,
        "runs": n,
        "success_rate": round(sum(r.success for r in runs) / n, 3),
        "collision_rate": round(sum(r.collision for r in runs) / n, 3),
        "timeout_rate": round(sum(r.timeout for r in runs) / n, 3),
        "mean_progress_m": round(float(np.mean([r.progress for r in runs])), 1),
        "min_ttc": round(float(np.min(finite_ttc)), 2) if finite_ttc else None,
        "mean_min_ttc": round(float(np.mean(finite_ttc)), 2) if finite_ttc else None,
        "min_clearance_m": round(float(np.min(finite_clr)), 2) if finite_clr else None,
        "mean_speed": round(float(np.mean([r.mean_speed for r in runs])), 2),
        "max_lat_accel": round(float(np.max([r.max_lat_accel for r in runs])), 2),
        "max_jerk": round(float(np.max([r.max_jerk for r in runs])), 2),
        "mean_d_safe_m": round(float(np.mean(with_traffic)), 2) if with_traffic else 0.0,
        "min_ttc_no_collision": round(float(np.min(clean_ttc)), 2) if clean_ttc else None,
        "plan_success_rate": round(float(np.mean([r.plan_success_rate for r in runs])), 3),
        "e2e_ms_p95": round(float(np.mean([r.end_to_end_p95 for r in runs])), 1),
    }
