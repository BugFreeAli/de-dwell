"""
A14: M0 with a probabilistic-goal objective ("asset survival", plan A14.2).

Model, instances and information are those of M0 (src/dwell/m0.py). Only the objective changes.
Let W be the total value of the n threats and lam in (0, 1) the leakage tolerance. The asset survives if the value of the
threats that leak (are not killed before breach) is at most lam * W, i.e. if the killed value reaches
    need = ceil((1 - lam) * W)          (values are integers, so need is an integer)
Objective: maximise P(killed value >= need)  ("probabilistic goal MDP", Xu and Mannor 2011).
The optimal policy depends on the accumulated killed value `acc`, so the exact DP runs on (t, cur, prog, acc).

Policies here are functions policy(state, acc) -> action or None. Every policy stops (returns None) once acc >= need,
because success is then certain; this changes no probability.

Methods (plan A14.4):
  EV rules      the 11 rules of A13 (built for expected killed value, they ignore acc)         EV_RULES
  DPEV          the exact optimum of the EXPECTED-VALUE objective, evaluated on the goal         (EV-based)
  CEPS          survival-aware certainty-equivalent planner (needs acc)                           (aware)
  RU_<base>     exact one-step policy improvement of <base> under the goal objective            (aware)
"""
import math
import time

import numpy as np

from .dp import solve
from .instances import make_instance_k
from .m0 import DEAD, EPS, Model
from .rules import make_rule

TOL = 1e-9
EV_RULES = ("FAIR", "EDF", "LLF", "SE", "GMR", "BI", "CEP", "LA1", "ROLL_CEP", "ROLL_GMR", "ROLL_EDF")
EV_BASED = EV_RULES + ("DPEV",)
AWARE = ("CEPS", "RU_CEP", "RU_GMR", "RU_EDF", "RU_DPEV")
METHODS = EV_BASED + AWARE
BASE_OF_RU = {"RU_CEP": "CEP", "RU_GMR": "GMR", "RU_EDF": "EDF", "RU_DPEV": "DPEV"}


def need_of(values, lam):
    """Killed value needed for survival: ceil((1 - lam) * W). A value <= 0 means the goal is reached at once."""
    return math.ceil((1.0 - lam) * float(sum(values)) - 1e-9)


class GoalDP:
    """Exact DP for max P(killed value >= need): memoised expectimax over (state, acc)."""

    def __init__(self, model, need):
        self.m, self.need = model, need
        self.memo = {}
        self.value = self.V(model.initial_state(), 0.0)

    def V(self, state, acc):
        if acc >= self.need - 1e-9:
            return 1.0                                   # the goal is reached, acc never decreases
        key = (state, acc)
        hit = self.memo.get(key)
        if hit is not None:
            return hit[0]
        potential = sum(self.m.v[j] for j, x in enumerate(state[2]) if x != DEAD)
        if acc + potential < self.need - 1e-9:           # even killing every killable threat cannot reach the goal
            self.memo[key] = (0.0, None)
            return 0.0
        best, best_j = 0.0, None
        for j in self.m.actions(state):                  # current target first, so ties keep it
            val = 0.0
            for p, r, nxt in self.m.outcomes(state, j):
                val += p * self.V(nxt, acc + r)
            if val > best + EPS:
                best, best_j = val, j
        self.memo[key] = (best, best_j)
        return best

    @property
    def n_states(self):
        return len(self.memo)

    def policy(self):
        def pol(state, acc):
            if acc >= self.need - 1e-9:
                return None
            self.V(state, acc)
            return self.memo[(state, acc)][1]
        return pol


def prob_goal(model, policy, need):
    """Exact P(killed value >= need) of policy(state, acc): enumerate every hidden class vector and weight by its prior."""
    cache, total = {}, 0.0
    for c, w in model.worlds():
        state, acc = model.initial_state(), 0.0
        while acc < need - 1e-9:
            key = (state, acc)
            if key in cache:
                j = cache[key]
            else:
                j = policy(state, acc)
                cache[key] = j
            if j is None:
                break
            t, cur, prog = state
            t2 = t + model.slew(cur, j) + 1
            p2 = list(prog)
            if prog[j] + 1 == model.D[c[j]]:             # the true class of j is completed by this step
                acc += model.v[j]
                p2[j] = DEAD
            else:
                p2[j] = prog[j] + 1
            state = (t2, j, model.canon(t2, p2))
        total += w * (1.0 if acc >= need - 1e-9 else 0.0)
    return total


def ceps_policy(model, need):
    """
    CEPS: survival-aware certainty-equivalent planner. Remaining dwell is replaced by its expectation; every ordered
    sequence of useful threats that is feasible with these durations is enumerated. Among the feasible sequences whose
    value reaches the remaining need, take the one with the largest minimum slack (ties: earlier finish, then larger
    value). If no sequence reaches the need, take the feasible sequence with the largest value (ties: earlier finish).
    If no sequence is feasible, use GMR. Re-planned at every step; ties keep the first sequence found (current target first).
    """
    fallback = make_rule(model, "GMR")

    def policy(state, acc):
        if acc >= need - 1e-9:
            return None
        acts = model.actions(state)
        if not acts:
            return None
        t, cur, prog = state
        dur = {j: sum(p * r for p, r in model.post[j][prog[j]]) for j in acts}
        need_left = need - acc
        reach, best = [None], [None]

        def extend(time_, last, used, value, minslack, first):
            if first is not None:
                if value >= need_left - 1e-9:
                    key = (minslack, -time_, value)
                    if reach[0] is None or key > reach[0][0]:
                        reach[0] = (key, first)
                key2 = (value, -time_)
                if best[0] is None or key2 > best[0][0]:
                    best[0] = (key2, first)
            for j in acts:
                if j in used:
                    continue
                end = time_ + model.slew(last, j) + dur[j]
                if end <= model.d[j] + 1e-9:
                    extend(end, j, used | {j}, value + model.v[j], min(minslack, model.d[j] - end),
                           j if first is None else first)

        extend(float(t), cur, frozenset(), 0.0, float("inf"), None)
        if reach[0] is not None:
            return reach[0][1]
        if best[0] is not None:
            return best[0][1]
        return fallback(state)

    return policy


def ru_policy(model, base, need):
    """Exact one-step policy improvement of `base` (a policy(state, acc)) under the goal objective."""
    vmemo = {}

    def bval(state, acc):
        if acc >= need - 1e-9:
            return 1.0
        key = (state, acc)
        if key in vmemo:
            return vmemo[key]
        j = base(state, acc)
        v = 0.0 if j is None else sum(p * bval(nxt, acc + r) for p, r, nxt in model.outcomes(state, j))
        vmemo[key] = v
        return v

    def policy(state, acc):
        if acc >= need - 1e-9:
            return None
        best_j, best_v = None, -1.0
        for j in model.actions(state):
            q = sum(p * bval(nxt, acc + r) for p, r, nxt in model.outcomes(state, j))
            if q > best_v + EPS:
                best_j, best_v = j, q
        return best_j

    return policy


def build_policies(model, need):
    """All 17 methods as policy(state, acc)."""
    pols = {}
    for m in EV_RULES:
        pols[m] = (lambda b: (lambda st, ac: b(st)))(make_rule(model, m))
    ev_pol = solve(model).policy()
    pols["DPEV"] = lambda st, ac: ev_pol(st)
    pols["CEPS"] = ceps_policy(model, need)
    for ru, b in BASE_OF_RU.items():
        pols[ru] = ru_policy(model, pols[b], need)
    return pols


def a14_instance(cfg, n, rho, s, a, lam, idx):
    """Instance (identical to A6 to A13 for the same arguments), its goal and the stored regime variables."""
    inst, real = make_instance_k(cfg["master_seed"], idx, n, rho, s, a, D=tuple(cfg["D0"]),
                                 value_set=tuple(cfg["value_set"]), u_low=cfg["u_low"])
    return inst, need_of(inst.values, lam), real


def run_instance(cfg, n, rho, s, a, lam, idx, audit=False):
    """One instance: exact goal optimum and the goal probability of every method; invariants are written to the row."""
    t0 = time.perf_counter()
    inst, need, real = a14_instance(cfg, n, rho, s, a, lam, idx)
    model = Model(inst)
    gdp = GoalDP(model, need)
    row = dict(n=n, rho=rho, s=s, a=a, lam=lam, idx=idx, W=float(sum(inst.values)), need=need, U_opt=gdp.value,
               n_states=gdp.n_states, **real)
    viol = []
    pols = build_policies(model, need)
    s0 = model.initial_state()
    for m in METHODS:
        u = prob_goal(model, pols[m], need)
        row["U_" + m] = u
        if not (-TOL <= u <= gdp.value + TOL):
            viol.append(f"range:{m}")
    for ru, b in BASE_OF_RU.items():
        if row["U_" + ru] < row["U_" + b] - TOL:
            viol.append(f"rollout_below_base:{ru}")
    if need <= 0 and abs(gdp.value - 1.0) > TOL:
        viol.append("trivial_goal")
    row["first_goal"], row["first_ev"] = gdp.policy()(s0, 0.0), pols["DPEV"](s0, 0.0)
    row["first_differs"] = int(row["first_goal"] != row["first_ev"])
    if audit and abs(prob_goal(model, gdp.policy(), need) - gdp.value) > TOL:
        viol.append("policy_value")
    row["violations"] = ";".join(viol)
    row["sec"] = time.perf_counter() - t0
    return row
