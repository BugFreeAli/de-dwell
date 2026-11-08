"""
Exact dynamic programme for M0 (finite horizon, observable state, expectimax over kill outcomes).

V(state) = max over useful actions j of  sum_outcomes p * (reward + V(next_state)),  V = 0 if no useful action.
The recursion is memoised on the canonical state (t, cur, prog), so only reachable states are visited.
Because the time axis is part of the state, the recursion terminates after at most d_max / 1 levels.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from .m0 import EPS, Model, State


class DPResult:
    """Optimal value, optimal policy and the number of visited states."""

    def __init__(self, model: Model):
        self.model = model
        self._memo: Dict[State, Tuple[float, Optional[int]]] = {}
        self.value = self._solve(model.initial_state())[0]

    def _solve(self, state: State) -> Tuple[float, Optional[int]]:
        hit = self._memo.get(state)
        if hit is not None:
            return hit
        best, best_j = 0.0, None
        for j in self.model.actions(state):          # current target first -> ties keep it
            val = 0.0
            for p, r, nxt in self.model.outcomes(state, j):
                val += p * (r + self._solve(nxt)[0])
            if val > best + EPS:
                best, best_j = val, j
        self._memo[state] = (best, best_j)
        return best, best_j

    @property
    def n_states(self) -> int:
        return len(self._memo)

    def policy(self) -> Callable[[State], Optional[int]]:
        """Optimal action for any state reachable under the DP (all successors of every action are memoised)."""
        return lambda state: self._solve(state)[1]


def solve(model: Model) -> DPResult:
    return DPResult(model)
