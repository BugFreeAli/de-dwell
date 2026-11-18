"""
Methods for M1 (amendment A11.4): 11 rules x two variants = 22 methods, named like "EDF_M" or "ROLL_GMR_L".

  _M  myopic          : for threat j the dwell rate is frozen at g(d_j - t), the value at decision time t.
  _L  range-lookahead : the true rate profile is used from the earliest start t + slew on.
For a class with remaining units r the number of steps is
  M: ceil(r / g(d_j - t))        L: the smallest m with g(tau0)+g(tau0-1)+...+g(tau0-m+1) >= r, tau0 = d_j - start
(beyond the breach the profile is extended with g(1); this only matters for classes that cannot finish in time).
Expected remaining time is sum_k p_k n_k; the ratios of GMR and BI, the CEP durations and the LLF and SE keys use n_k
in place of the remaining dwell of M0. With constant g both variants equal the M0 rule (tests).
All methods choose among the useful dwell actions of the model. Only the six rollouts can choose IDLE (idle followed
by the base policy is one of their candidate actions); the other methods never idle. Ties: current target first.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from .m1 import EPS, IDLE, Model1, State

Policy = Callable[[State], Optional[int]]
BASE_RULES = ("FAIR", "EDF", "LLF", "SE", "GMR", "BI", "CEP", "LA1")
ROLL_RULES = ("ROLL_CEP", "ROLL_GMR", "ROLL_EDF")
RULES = BASE_RULES + ROLL_RULES
METHODS = tuple(f"{r}_{v}" for r in RULES for v in ("M", "L"))


def split(name: str):
    rule, variant = name.rsplit("_", 1)
    assert variant in ("M", "L"), name
    return rule, variant


def steps_list(model: Model1, variant: str, t: int, j: int, x: int, start: float):
    """[(p, n_k)]: posterior over the number of dwell steps needed by threat j with progress x."""
    post = model.post[j][x]
    if variant == "M":
        g = model.g_at(model.d[j] - t)
        return [(p, -(-r // g)) for p, r in post]
    st = int(math.ceil(start - 1e-9))
    tau0 = model.d[j] - st
    out = []
    for p, r in post:
        key = (j, tau0, r)
        n = model._steps.get(key)
        if n is None:
            acc, m = 0, 0
            while acc < r:
                acc += model.g_at(tau0 - m); m += 1
            n = model._steps[key] = m
        out.append((p, n))
    return out


def _ratio(v, sl, cap, sl_post, tau):
    lim = min(tau, cap)
    p_succ = sum(p for p, n in sl_post if n <= lim)
    spent = sum(p * min(n, lim) for p, n in sl_post)
    return v * p_succ / (sl + spent)


def _score(rule, variant, model, state, j):
    t, cur, prog, e = state
    x = prog[j]
    sl = model.slew(cur, j)
    cap = model.d[j] - t - sl
    sp = steps_list(model, variant, t, j, x, t + sl)
    e_n = sum(p * n for p, n in sp)
    if rule == "FAIR":
        return -float(x)
    if rule == "EDF":
        return -float(model.d[j])
    if rule == "LLF":
        return -(model.d[j] - t - sl - e_n)
    if rule == "SE":
        return -(sl + e_n)
    if rule == "GMR":
        return _ratio(model.v[j], sl, cap, sp, float("inf"))
    if rule == "BI":
        taus = {n for _, n in sp if n <= cap} | {cap}
        return max(_ratio(model.v[j], sl, cap, sp, tau) for tau in taus)
    raise ValueError(rule)


def _duration(model, variant, state, j, start):
    t, cur, prog, e = state
    return sum(p * n for p, n in steps_list(model, variant, t, j, prog[j], start))


def _plan_search(model, variant, state, acts, want_first):
    """Certainty-equivalent plan: ordered subsets, expected durations; returns (best value, first action)."""
    t, cur, prog, e = state
    best = [0.0, 0.0, None]

    def extend(time, last, used, value, first):
        if want_first and first is not None and (value > best[0] + EPS or
                                                (abs(value - best[0]) <= EPS and time < best[1] - EPS)):
            best[0], best[1], best[2] = value, time, first
        elif not want_first and value > best[0]:
            best[0] = value
        for j in acts:
            if j in used:
                continue
            start = time + model.slew(last, j)
            end = start + _duration(model, variant, state, j, start)
            if end <= model.d[j] + 1e-9:
                extend(end, j, used | {j}, value + model.v[j], j if first is None else first)

    extend(float(t), cur, frozenset(), 0.0, None)
    return best[0], best[2]


def _cep(model, variant) -> Policy:
    fallback = make_method(model, f"GMR_{variant}")

    def policy(state):
        acts = model.dwell_actions(state)
        if not acts:
            return None
        _, first = _plan_search(model, variant, state, acts, True)
        return first if first is not None else fallback(state)
    return policy


def _la1(model, variant) -> Policy:
    cache: dict = {}

    def plan_value(state):
        hit = cache.get(state)
        if hit is None:
            acts = model.dwell_actions(state)
            hit = cache[state] = _plan_search(model, variant, state, acts, False)[0] if acts else 0.0
        return hit

    def policy(state):
        best_a, best_v = None, -1.0
        for a in model.dwell_actions(state):
            val = sum(p * (r + plan_value(nxt)) for p, r, nxt in model.outcomes(state, a))
            if val > best_v + EPS:
                best_a, best_v = a, val
        return best_a
    return policy


def _rollout(model, base_name) -> Policy:
    base = make_method(model, base_name)
    bcache: dict = {}
    vcache: dict = {}

    def base_action(state):
        if state not in bcache:
            bcache[state] = base(state)
        return bcache[state]

    def value(state):
        hit = vcache.get(state)
        if hit is None:
            a = base_action(state)
            hit = vcache[state] = 0.0 if a is None else sum(p * (r + value(nxt)) for p, r, nxt in model.outcomes(state, a))
        return hit

    def policy(state):
        best_a, best_v = None, -1.0
        for a in model.actions(state):                  # dwell actions first, IDLE last
            q = sum(p * (r + value(nxt)) for p, r, nxt in model.outcomes(state, a))
            if q > best_v + EPS:
                best_a, best_v = a, q
        return best_a
    return policy


def make_method(model: Model1, name: str) -> Policy:
    rule, variant = split(name)
    if rule.startswith("ROLL_"):
        return _rollout(model, f"{rule[5:]}_{variant}")
    if rule == "CEP":
        return _cep(model, variant)
    if rule == "LA1":
        return _la1(model, variant)

    def policy(state):
        best_a, best_s = None, -float("inf")
        for j in model.dwell_actions(state):
            sc = _score(rule, variant, model, state, j)
            if sc > best_s + EPS:
                best_a, best_s = j, sc
        return best_a
    return policy
