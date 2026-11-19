"""
A12: budgeted hardening adversary (plan A12.2 and A12.3).

Truth model of a strategy (q, lambda_h): every threat is hardened independently with probability q. A hardened
threat of base class c needs round(lambda_h * D0[c]) + e dwell steps, a non-hardened one D0[c] + e, where e is the
within-class noise (offsets and probabilities as in A7). q = 1 is the shared scale of A6 and A7, q = 0 is the nominal model.
The defender sees only the dwell received and the kill flags. All values are exact expectations.

Everything that is not new is reused from A6/A7 (adapt.py, mismatch.py, m2.py, dp.py):
  * Planner / Kit / nominal_factory / eval_rec build and evaluate the nominal planners, QMDP and the Bayes-adaptive ceiling;
  * believed_factory turns the optimal policy of a believed model into a policy on observations.
"""
import dataclasses
import time

from .adapt import Kit, Planner, class_dwells, eval_rec, merged_instance, nominal_factory
from .dp import solve
from .instances import make_instance_k
from .m2 import Model2
from .mismatch import believed_factory, expand_prior, scaled

TOL = 1e-9

# (name, q, lambda_h); beta = q * (lambda_h - 1) is the hardening budget of plan A12.2.
STRATEGIES = (
    ("NONE", 0.0, 1.0),
    ("SH12", 1.0, 1.2),
    ("PH50_14", 0.5, 1.4),
    ("PH25_18", 0.25, 1.8),
    ("SH14", 1.0, 1.4),
    ("PH50_18", 0.5, 1.8),
)
STRAT_NAMES = tuple(s[0] for s in STRATEGIES)
PARTIAL = ("PH50_14", "PH25_18", "PH50_18")
NONADAPTIVE = ("DPMIS_1.0", "CEP_1.0", "ROLL_CEP_1.0", "DPMIS_1.2")
ADAPTIVE = ("QMDP_B", "QMDP_H", "BA_H", "IID_DP")
DEFENDERS = NONADAPTIVE + ADAPTIVE
PRIOR_B = (0.8, 1.0, 1.2)       # prior B of A7
PRIOR_H = (1.0, 1.2, 1.4)       # hardening-only prior (new)


def beta(name):
    for n, q, lam in STRATEGIES:
        if n == name:
            return q * (lam - 1.0)
    raise KeyError(name)


def budget_set(level):
    """Strategy names whose hardening budget does not exceed the level (B20 -> 0.2, B40 -> 0.4)."""
    cap = {"B20": 0.2, "B40": 0.4}[level]
    return tuple(n for n, q, lam in STRATEGIES if q * (lam - 1.0) <= cap + 1e-9)


def hard_truth(inst0, D0, offsets, probs, q, lam_h):
    """True instance of strategy (q, lam_h) as an M0 instance with merged classes (exact)."""
    D, cols = [], []
    for c, b in enumerate(D0):
        for hardened, pw in ((False, 1.0 - q), (True, q)):
            if pw <= 0.0:
                continue
            base = scaled((b,), lam_h)[0] if hardened else b
            for e, pe in zip(offsets, probs):
                D.append(base + e)
                cols.append((c, pw * pe))
    assert all(x >= 1 for x in D), "dwell below one step"
    prior = tuple(tuple(inst0.prior[i][c] * w for c, w in cols) for i in range(inst0.n))
    return merged_instance(dataclasses.replace(inst0, D=tuple(D), prior=prior), tuple(D))


def a12_instance(cfg, rho, s, a, idx):
    """Same generator and noise expansion as a7_instance, with the planner scales of A12."""
    base = tuple(cfg["D0"])
    offs, probs = tuple(cfg["noise_offsets"]), tuple(cfg["noise_probs"])
    inst0, real = make_instance_k(cfg["master_seed"], idx, cfg["n"], rho, s, a, D=base,
                                  value_set=tuple(cfg["value_set"]), u_low=cfg["u_low"])
    Dl = {lam: class_dwells(base, lam, offs) for lam in sorted(set(cfg["planner_scales"]) | {1.0})}
    inst = dataclasses.replace(inst0, D=Dl[1.0], prior=expand_prior(inst0.prior, probs))
    return inst, Dl, real, inst0


def _same_instance(x, y):
    return (x.D == y.D and x.s == y.s and x.values == y.values and x.deadlines == y.deadlines
            and all(abs(p - r) < 1e-12 for rx, ry in zip(x.prior, y.prior) for p, r in zip(rx, ry)))


def run_instance(cfg, rho, s, a, idx):
    """One instance: value of every defender under every strategy and the optimum of every strategy."""
    t_start = time.perf_counter()
    D0, offs, probs = tuple(cfg["D0"]), tuple(cfg["noise_offsets"]), tuple(cfg["noise_probs"])
    inst, Dl, real, inst0 = a12_instance(cfg, rho, s, a, idx)
    planner = Planner(inst, Dl, cfg["planner_scales"])
    row = dict(rho=rho, s=s, a=a, idx=idx, n=cfg["n"], total_value=sum(inst.values), **real)
    viol = []

    truth = {}                                           # name -> (true instance, model, DP result)
    for name, q, lam in STRATEGIES:
        ti = hard_truth(inst0, D0, offs, probs, q, lam)
        key = 1.0 if q == 0.0 else (lam if q == 1.0 else None)
        if key is not None:                              # shared or nominal truth: reuse the planner's DP
            pi, pm, pdp = planner.models[key]
            if not _same_instance(pi, ti):
                viol.append(f"truth_mismatch:{name}")
            truth[name] = (pi, pm, pdp)
        else:
            m = Model2(ti)
            truth[name] = (ti, m, solve(m))
        row[f"Vstar_{name}"] = truth[name][2].value
        row[f"nst_{name}"] = truth[name][2].n_states

    w = tuple(1.0 / 3 for _ in range(3))
    kit_b, kit_h = Kit(planner, PRIOR_B, w), Kit(planner, PRIOR_H, w)
    facs = {
        "DPMIS_1.0": nominal_factory(planner, "DPMIS", 1.0),
        "CEP_1.0": nominal_factory(planner, "CEP", 1.0),
        "ROLL_CEP_1.0": nominal_factory(planner, "ROLL_CEP", 1.0),
        "DPMIS_1.2": nominal_factory(planner, "DPMIS", 1.2),
        "QMDP_B": kit_b.factory("QMDP"),
        "QMDP_H": kit_h.factory("QMDP"),
        "BA_H": kit_h.factory("BA"),
        "IID_DP": believed_factory(truth["PH50_14"][1], truth["PH50_14"][2].policy()),
    }
    assert tuple(facs) == DEFENDERS
    for dname in DEFENDERS:
        t0 = time.perf_counter()
        for k in STRAT_NAMES:
            v = eval_rec(truth[k][0], facs[dname])
            row[f"V_{dname}_{k}"] = v
            if not (-TOL <= v <= truth[k][2].value + TOL):
                viol.append(f"range:{dname}:{k}")
        row[f"t_{dname}"] = time.perf_counter() - t0

    # exact identities of the plan (A12.6, audit invariants)
    for dname, k in (("DPMIS_1.0", "NONE"), ("DPMIS_1.2", "SH12"), ("IID_DP", "PH50_14")):
        if abs(row[f"V_{dname}_{k}"] - row[f"Vstar_{k}"]) > TOL:
            viol.append(f"not_calibrated:{dname}:{k}")
    dpba = kit_h.solve_ba().value
    row["V_BA_H_dp"] = dpba
    inside = ("NONE", "SH12", "SH14")                    # the three shared scales of prior H
    if abs(sum(w[i] * row[f"V_BA_H_{k}"] for i, k in enumerate(inside)) - dpba) > 1e-8:
        viol.append("ba_policy_value")
    if sum(w[i] * row[f"V_QMDP_H_{k}"] for i, k in enumerate(inside)) > dpba + 1e-8:
        viol.append("qmdp_above_ba")
    row["violations"] = ";".join(viol)
    row["sec"] = time.perf_counter() - t_start
    return row
