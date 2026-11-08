"""
Simple, information-matched scheduling rules for M0. Every rule sees only the observable state
(t, cur, prog) and the prior, exactly like the DP. All rules share one feasibility filter: they only
choose among "useful" threats (a kill before the deadline is still possible), see Model.actions.
Ties keep the current target, then the lowest index (Model.actions lists the current target first
and a candidate must beat the incumbent by more than EPS).

Rule  | key                                              | source
FAIR  | least dwell received first (max -x)              | adapted from the fair-allocation result of Chen et al. 2026 (memoryless shots, no slew)
EDF   | earliest deadline first                          | classical
LLF   | least expected laxity: d - t - slew - E[rem]     | classical, with expected remaining dwell
SE    | shortest engagement: slew + E[rem] (min)         | Taylor 2021 (time-based rule, value-blind)
GMR   | v * P(kill before breach) / (slew + E[time spent]) | greedy marginal return per unit time, dedicated attempt
BI    | max over persistence limit tau of the same ratio  | Gittins-type belief index, ADAPTED to deterministic dwell thresholds;
      |                                                  | NOT the exact index of Glazebrook et al. (memoryless shots)
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from .m0 import EPS, Model, State

Policy = Callable[[State], Optional[int]]


def _stats(model: Model, state: State, j: int):
    """Per-threat quantities used by the rule keys."""
    t, cur, prog = state
    x = prog[j]
    sl = model.slew(cur, j)
    cap = model.d[j] - t - sl            # dwell steps still usable before breach
    post = model.post[j][x]              # [(p, r)] posterior over remaining dwell
    return x, sl, cap, post


def _ratio(v: float, sl: int, cap: int, post, tau: float) -> float:
    """v * P(kill within min(tau,cap)) / (slew + E[time spent within min(tau,cap)])."""
    lim = min(tau, cap)
    p_succ = sum(p for p, r in post if r <= lim)
    spent = sum(p * min(r, lim) for p, r in post)
    return v * p_succ / (sl + spent)


def _score(rule: str, model: Model, state: State, j: int) -> float:
    """Higher score = engage first."""
    t, _, _ = state
    x, sl, cap, post = _stats(model, state, j)
    e_rem = sum(p * r for p, r in post)
    if rule == "FAIR":
        return -float(x)
    if rule == "EDF":
        return -float(model.d[j])
    if rule == "LLF":
        return -(model.d[j] - t - sl - e_rem)
    if rule == "SE":
        return -(sl + e_rem)
    if rule == "GMR":
        return _ratio(model.v[j], sl, cap, post, float("inf"))
    if rule == "BI":
        taus = {r for _, r in post if r <= cap} | {cap}
        return max(_ratio(model.v[j], sl, cap, post, tau) for tau in taus)
    raise ValueError(f"unknown rule {rule}")


SIMPLE_RULES = ("FAIR", "EDF", "LLF", "SE", "GMR", "BI")


def make_rule(model: Model, rule: str) -> Policy:
    """Return the policy function of a rule for this model."""
    def policy(state: State) -> Optional[int]:
        best_j, best_s = None, -float("inf")
        for j in model.actions(state):           # current target first
            sc = _score(rule, model, state, j)
            if sc > best_s + EPS:
                best_j, best_s = j, sc
        return best_j
    return policy
