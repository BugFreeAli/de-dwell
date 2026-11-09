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

import math
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
ALL_RULES = SIMPLE_RULES + ("CEP", "LA1")


def _durations(model: Model, state: State, acts):
    t, cur, prog = state
    dur = {j: sum(p * r for p, r in model.post[j][prog[j]]) for j in acts}
    dur_left = None
    kappa = getattr(model, "kappa", 1.0)
    if kappa < 1.0 and cur in dur:
        xc, mc = divmod(prog[cur], model.K)
        code = int(math.floor(kappa * xc + 1e-9)) * model.K + mc
        post = model.post[cur][code]
        if post:
            dur_left = sum(p * r for p, r in post)
    return dur, dur_left


def _cep_policy(model: Model) -> Policy:
    fallback = make_rule(model, "GMR")

    def policy(state: State) -> Optional[int]:
        acts = model.actions(state)
        if not acts:
            return None
        t, cur, prog = state
        dur, dur_left = _durations(model, state, acts)
        best = [0.0, 0.0, None]

        def extend(time, last, used, value, first):
            if first is not None and (value > best[0] + EPS or (abs(value - best[0]) <= EPS and time < best[1] - EPS)):
                best[0], best[1], best[2] = value, time, first
            for j in acts:
                if j in used:
                    continue
                dj = dur_left if (dur_left is not None and j == cur and first is not None and first != cur) else dur[j]
                end = time + model.slew(last, j) + dj
                if end <= model.d[j] + 1e-9:
                    extend(end, j, used | {j}, value + model.v[j], j if first is None else first)

        extend(float(t), cur, frozenset(), 0.0, None)
        return best[2] if best[2] is not None else fallback(state)

    return policy


def _plan_value(model: Model, state: State, cache: dict) -> float:
    hit = cache.get(state)
    if hit is not None:
        return hit
    acts = model.actions(state)
    if not acts:
        cache[state] = 0.0
        return 0.0
    t, cur, prog = state
    dur, dur_left = _durations(model, state, acts)
    best = [0.0]

    def extend(time, last, used, value, first):
        if value > best[0]:
            best[0] = value
        for j in acts:
            if j in used:
                continue
            dj = dur_left if (dur_left is not None and j == cur and first is not None and first != cur) else dur[j]
            end = time + model.slew(last, j) + dj
            if end <= model.d[j] + 1e-9:
                extend(end, j, used | {j}, value + model.v[j], j if first is None else first)

    extend(float(t), cur, frozenset(), 0.0, None)
    cache[state] = best[0]
    return best[0]


def _la1_policy(model: Model) -> Policy:
    cache: dict = {}

    def policy(state: State) -> Optional[int]:
        best_j, best_v = None, -1.0
        for j in model.actions(state):
            val = 0.0
            for p, r, nxt in model.outcomes(state, j):
                val += p * (r + _plan_value(model, nxt, cache))
            if val > best_v + EPS:
                best_j, best_v = j, val
        return best_j

    return policy


def _rollout_policy(model: Model, base_name: str) -> Policy:
    base = make_rule(model, base_name)
    base_cache: dict = {}
    value_cache: dict = {}

    def base_action(state: State):
        if state not in base_cache:
            base_cache[state] = base(state)
        return base_cache[state]

    def value(state: State) -> float:
        hit = value_cache.get(state)
        if hit is not None:
            return hit
        j = base_action(state)
        v = 0.0 if j is None else sum(p * (r + value(nxt)) for p, r, nxt in model.outcomes(state, j))
        value_cache[state] = v
        return v

    def policy(state: State) -> Optional[int]:
        best_j, best_v = None, -1.0
        for j in model.actions(state):
            q = sum(p * (r + value(nxt)) for p, r, nxt in model.outcomes(state, j))
            if q > best_v + EPS:
                best_j, best_v = j, q
        return best_j

    return policy


def make_rule(model: Model, rule: str) -> Policy:
    """Return the policy function of a rule for this model."""
    if rule.startswith("ROLL_"):
        return _rollout_policy(model, rule[5:])
    if rule == "CEP":
        return _cep_policy(model)
    if rule == "LA1":
        return _la1_policy(model)

    def policy(state: State) -> Optional[int]:
        best_j, best_s = None, -float("inf")
        for j in model.actions(state):           # current target first
            sc = _score(rule, model, state, j)
            if sc > best_s + EPS:
                best_j, best_s = j, sc
        return best_j
    return policy
