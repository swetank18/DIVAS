"""Stage 5 -- planning over free space."""

from __future__ import annotations

from typing import Optional

from divas.planning.hybrid_astar import HybridAStar, PlannerConfig, PlanResult
from divas.planning.rrt import KinodynamicRRT, RRTConfig
from divas.types import EgoState, OccupancyGrid, VehicleParams


class FallbackPlanner:
    """Hybrid A* first, sampling second.

    The lattice search is better when it works -- deterministic, repeatable,
    and fast enough to replan at 4 Hz.  Sampling is what is left when the
    corridor pinches below the lattice resolution or the manoeuvre needs a
    heading the bins do not contain.  Running the fallback only on failure
    keeps the common case cheap.

    When neither reaches the goal, the longer of the two partial paths is
    returned: a partial plan is still the best reachable prefix, and the
    controller knows from ``terminal_stop`` to treat its end as a stop.
    """

    def __init__(
        self,
        params: Optional[VehicleParams] = None,
        config: Optional[PlannerConfig] = None,
        rrt_config: Optional[RRTConfig] = None,
        seed: int = 0,
    ) -> None:
        self.params = params or VehicleParams()
        self.astar = HybridAStar(self.params, config)
        self.rrt = KinodynamicRRT(self.params, rrt_config, seed=seed)
        self.fallback_calls = 0
        self.fallback_rescues = 0
        # The previous plan, kept so the lattice search can be biased towards
        # it.  Held here rather than inside HybridAStar so the search itself
        # stays a pure function of its inputs and remains easy to test.
        self._previous = None

    def reset(self) -> None:
        self._previous = None

    def plan(self, grid, start, goal_xy, risk=None) -> PlanResult:
        res = self.astar.plan(grid, start, goal_xy, risk, previous=self._previous)
        if not res.success:
            self.fallback_calls += 1
            alt = self.rrt.plan(grid, start, goal_xy, risk)
            if alt.success:
                self.fallback_rescues += 1
                res = alt
            elif alt.path.length > res.path.length:
                res = alt
        if len(res.path) >= 2:
            self._previous = res.path
        return res


__all__ = [
    "HybridAStar", "PlannerConfig", "PlanResult",
    "KinodynamicRRT", "RRTConfig", "FallbackPlanner",
]
