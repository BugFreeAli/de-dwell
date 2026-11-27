"""
A18: the survival objective under a wrong dwell scale (plan A18.2, A18.3).

The planners are built on the NOMINAL model (class dwell D0, the model every earlier amendment used). The world has the dwell
TRUE = scale * D0 (class k needs scale * D0[k] dwell steps); deadlines, values, slew, labels and priors are those of the nominal
instance (common random numbers across scales). The planner sees what it can observe (time, current target, dwell given to each
threat, which threats were killed) and maps it into its nominal state: a killed threat is DEAD; an alive threat has progress x;
the nominal canonical form marks a threat DEAD when the nominal model says it can no longer be killed by its deadline (this includes
progress beyond the largest nominal dwell). Nothing else about the true dwell is known to the planner.

Exact evaluation: every hidden class vector is enumerated, the true kill happens when the dwell given to a threat reaches the true
dwell of its class. Policy decisions are cached per (nominal state, accumulated value).

Methods (U = survival probability P(killed value >= need)):
  nominal planners in the true world: CEP, CEPS, RU_CEP (level ru, exact), DPEV and RU_DPEV (level exact: exact expected-value optimum
  of the nominal model and its goal rollout);
  informed references (built on the TRUE model and evaluated in it): CEP_T, CEPS_T; level exact also U_opt_T (exact goal optimum of the true model).
Invariants written to the row: range, nominal planners not above U_opt_T, scale 1 reproduces the informed values, no kill completes
after its deadline.
"""
import dataclasses
import time

from .dp import solve
from .instances import make_instance_k
from .m0 import DEAD, Model
from .m0u import GoalDP, ceps_policy, need_of, prob_goal, ru_policy
from .rules import make_rule

TOL = 1e-9


def scaled_dwell(D0, scale):
    D = tuple(int(round(scale * d)) for d in D0)
    assert all(abs(scale * d - x) < 1e-9 for d, x in zip(D0, D)), "scale must give integer dwell"
    return D


def prob_goal_true(model_b, D_true, policy, need):
    """
    Exact P(killed value >= need) of a policy built on the nominal model model_b when class k really needs D_true[k] dwell steps.
    policy(state, acc) sees the nominal state. Returns (probability, number of kills that would complete after the deadline).
    """
    cache, total, late = {}, 0.0, 0
    for c, w in model_b.worlds():
        t, cur = 0, -1
        x = [0] * model_b.n                              # dwell received (true, observable)
        prog = list(model_b.initial_state()[2])          # nominal progress or DEAD
        acc = 0.0
        while acc < need - 1e-9:
            state = (t, cur, tuple(prog))
            key = (state, acc)
            if key in cache:
                j = cache[key]
            else:
                j = policy(state, acc)
                cache[key] = j
            if j is None:
                break
            t2 = t + model_b.slew(cur, j) + 1
            x[j] += 1
            p2 = list(prog)
            if x[j] == D_true[c[j]]:                     # the threat is killed by this step
                if t2 <= model_b.d[j]:
                    acc += model_b.v[j]
                else:
                    late += 1
                p2[j] = DEAD
            else:
                p2[j] = min(x[j], model_b.D[-1])         # nominal progress (beyond the largest nominal dwell the canonical form gives DEAD)
            prog = list(model_b.canon(t2, p2))
            t, cur = t2, j
        total += w * (1.0 if acc >= need - 1e-9 else 0.0)
    return total, late


def eval_row(cfg, n, rho, s, a, lam, scale, idx, level):
    """One instance at one true scale. level: 'exact' (n <= 5) or 'ru'."""
    t0 = time.perf_counter()
    D0 = tuple(cfg["D0"])
    inst, real = make_instance_k(cfg["eval_seed"], idx, n, rho, s, a, D=D0, value_set=tuple(cfg["value_set"]), u_low=cfg["u_low"])
    Dt = scaled_dwell(D0, scale)
    mb = Model(inst)
    mt = Model(dataclasses.replace(inst, D=Dt))
    need = need_of(inst.values, lam)
    row = dict(n=n, rho=rho, s=s, a=a, lam=lam, scale=scale, idx=idx, need=need, W=float(sum(inst.values)), **real)
    cep = make_rule(mb, "CEP")
    pc = lambda st, ac: cep(st)
    pols = {"CEP": pc, "CEPS": ceps_policy(mb, need)}
    pols["RU_CEP"] = ru_policy(mb, pc, need)
    if level == "exact":
        evp = solve(mb).policy()
        pd_ = lambda st, ac: evp(st)
        pols["DPEV"] = pd_
        pols["RU_DPEV"] = ru_policy(mb, pd_, need)
    viol, late = [], 0
    for m, p in pols.items():
        u, lt = prob_goal_true(mb, Dt, p, need)
        row["U_" + m] = u
        late += lt
    cept = make_rule(mt, "CEP")
    row["U_CEP_T"] = prob_goal(mt, lambda st, ac: cept(st), need)
    row["U_CEPS_T"] = prob_goal(mt, ceps_policy(mt, need), need)
    if level == "exact":
        row["U_opt_T"] = GoalDP(mt, need).value
        if any(row["U_" + m] > row["U_opt_T"] + TOL for m in pols) or any(row[k] > row["U_opt_T"] + TOL for k in ("U_CEP_T", "U_CEPS_T")):
            viol.append("above_optimum")
    if scale == 1.0 and (abs(row["U_CEP"] - row["U_CEP_T"]) > TOL or abs(row["U_CEPS"] - row["U_CEPS_T"]) > TOL):
        viol.append("scale1_differs")
    if late:
        viol.append("late_kill")
    us = [v for k, v in row.items() if k.startswith("U_")]
    if any(not (-TOL <= u <= 1 + TOL) for u in us):
        viol.append("range")
    row["violations"] = ";".join(viol)
    row["sec"] = time.perf_counter() - t0
    return row
