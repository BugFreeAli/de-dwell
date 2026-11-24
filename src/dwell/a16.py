"""
A16: planner-side confirmatory evaluation (plan A16.3). One row per instance: exact survival probability of the planners CEP
(expected-value rule), CEPS and RU_CEP (survival-aware, A14.4), the expected killed value of CEP and the goal tightness
tau = need / E[killed value of CEP]. At level "exact" (n <= 5) also the exact goal optimum, DPEV, RU_DPEV and the exact expected-value optimum.
Levels: "exact" (all of the above), "ru" (CEP, CEPS, RU_CEP), "cheap" (CEP, CEPS only).
"""
import math
import time

import numpy as np

from .dp import solve
from .instances import make_instance_k
from .m0 import DEAD, Model
from .m0u import GoalDP, ceps_policy, need_of, prob_goal, ru_policy
from .rules import make_rule

TOL = 1e-9


def expected_killed(model, policy):
    """Exact expected killed value of policy(state, acc): enumerate every hidden-class vector (policy stops only when it returns None)."""
    cache, total = {}, 0.0
    for c, w in model.worlds():
        state, acc = model.initial_state(), 0.0
        while True:
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
            if prog[j] + 1 == model.D[c[j]]:
                acc += model.v[j]
                p2[j] = DEAD
            else:
                p2[j] = prog[j] + 1
            state = (t2, j, model.canon(t2, p2))
        total += w * acc
    return total


def eval_row(cfg, group, n, rho, s, a, lam, idx, level, D0=None, value_set=None):
    t0 = time.perf_counter()
    D0 = tuple(D0 or cfg["D0"])
    value_set = tuple(value_set or cfg["value_set"])
    inst, real = make_instance_k(cfg["eval_seed"], idx, n, rho, s, a, D=D0, value_set=value_set, u_low=cfg["u_low"])
    model = Model(inst)
    need = need_of(inst.values, lam)
    row = dict(group=group, n=n, rho=rho, s=s, a=a, lam=lam, idx=idx, need=need, W=float(sum(inst.values)), **real)
    cep = make_rule(model, "CEP")
    pc = lambda st, ac: cep(st)
    pcs = ceps_policy(model, need)
    row["U_CEP"], row["U_CEPS"] = prob_goal(model, pc, need), prob_goal(model, pcs, need)
    row["E_CEP"] = expected_killed(model, pc)                    # CEP never stops at the goal: it is the expected-value rule
    row["tau"] = need / row["E_CEP"] if row["E_CEP"] > 1e-12 else float("inf")
    viol = []
    if level in ("ru", "exact"):
        row["U_RU_CEP"] = prob_goal(model, ru_policy(model, pc, need), need)
        if row["U_RU_CEP"] < row["U_CEP"] - TOL:
            viol.append("rollout_below_base")
    if level == "exact":
        gdp = GoalDP(model, need)
        ev = solve(model)
        evp = ev.policy()
        pd_ = lambda st, ac: evp(st)
        row["U_opt"], row["V_ev"] = gdp.value, ev.value
        row["tau_ev"] = need / ev.value if ev.value > 1e-12 else float("inf")
        row["U_DPEV"] = prob_goal(model, pd_, need)
        row["U_RU_DPEV"] = prob_goal(model, ru_policy(model, pd_, need), need)
        if row["U_RU_DPEV"] < row["U_DPEV"] - TOL:
            viol.append("rollout_below_base_dpev")
        if any(row[k] > row["U_opt"] + TOL for k in ("U_CEP", "U_CEPS", "U_RU_CEP", "U_DPEV", "U_RU_DPEV")):
            viol.append("above_optimum")
        if row["E_CEP"] > row["V_ev"] + TOL:
            viol.append("cep_above_ev_optimum")
    us = [row[k] for k in row if k.startswith("U_") and k != "U_opt"]
    if any(not (-TOL <= u <= 1 + TOL) for u in us):
        viol.append("range")
    row["violations"] = ";".join(viol)
    row["sec"] = time.perf_counter() - t0
    return row
