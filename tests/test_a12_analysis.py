"""Tests of the A12 analysis: linear program, split sample, regret matrix, decision logic (plan A12.6 items 6 to 8)."""
import itertools

import numpy as np
import pandas as pd
import pytest

from dwell.analysis_a12 import (analyse, build_arrays, decide, flags, minimax_mixture, regret_matrix, stats_from,
                                strata)
from dwell.hard import ADAPTIVE, DEFENDERS, NONADAPTIVE, PARTIAL, STRAT_NAMES, budget_set


# ---- linear program --------------------------------------------------------------------------------------
def test_matching_pennies():
    x, w, v = minimax_mixture([[0.0, 1.0], [1.0, 0.0]])
    assert np.allclose(x, [0.5, 0.5]) and np.allclose(w, [0.5, 0.5]) and abs(v - 0.5) < 1e-9


def test_dominated_defender_gets_no_weight():
    x, w, v = minimax_mixture([[0.1, 0.2], [0.3, 0.4]])
    assert np.allclose(x, [1.0, 0.0]) and abs(v - 0.2) < 1e-9


def test_pure_saddle():
    x, w, v = minimax_mixture([[3.0, 1.0, 2.0], [5.0, 4.0, 6.0]])
    assert np.allclose(x, [1.0, 0.0]) and abs(v - 3.0) < 1e-9
    assert np.allclose(w, [1.0, 0.0, 0.0])


def test_constant_and_zero_matrices():
    x, w, v = minimax_mixture(np.zeros((4, 3)))
    assert abs(v) < 1e-12 and abs(x.sum() - 1) < 1e-12
    x, w, v = minimax_mixture(np.full((3, 5), 0.07))
    assert abs(v - 0.07) < 1e-12


@pytest.mark.parametrize("seed", range(40))
def test_duality_certificate_random(seed):
    """x and w are feasible and their payoffs coincide: a proof of optimality that needs no other solver."""
    rng = np.random.default_rng(seed)
    m, k = int(rng.integers(1, 9)), int(rng.integers(1, 7))
    R = rng.random((m, k)) * (0.3 if seed % 2 else 1.0)
    if seed % 5 == 0:
        R = np.round(R, 1)                                    # ties and degeneracy
    x, w, v = minimax_mixture(R)
    assert x.min() >= -1e-12 and abs(x.sum() - 1) < 1e-9 and w.min() >= -1e-12 and abs(w.sum() - 1) < 1e-9
    assert (x @ R).max() <= v + 1e-9
    assert (R @ w).min() >= v - 1e-9


def test_matches_scipy_when_available():
    linprog = pytest.importorskip("scipy.optimize").linprog
    rng = np.random.default_rng(5)
    for _ in range(30):
        R = rng.random((int(rng.integers(2, 9)), int(rng.integers(2, 7))))
        m, k = R.shape
        res = linprog(np.r_[np.zeros(m), 1.0], A_ub=np.c_[R.T, -np.ones(k)], b_ub=np.zeros(k),
                      A_eq=[np.r_[np.ones(m), 0.0]], b_eq=[1.0], bounds=[(0, None)] * m + [(None, None)])
        assert abs(res.fun - minimax_mixture(R)[2]) < 1e-8


# ---- budget sets, regret matrix, split ------------------------------------------------------------------
def test_budget_filter_exact():
    assert set(budget_set("B20")) == {"NONE", "SH12", "PH50_14", "PH25_18"}
    assert set(budget_set("B40")) == set(STRAT_NAMES)


def test_regret_matrix_uses_own_denominator():
    V = np.array([[[1.0, 2.0]], [[3.0, 4.0]]]).transpose(0, 1, 2)       # N=2, D=1, K=2
    Vs = np.array([[2.0, 4.0], [4.0, 8.0]])
    R = regret_matrix(V, Vs, np.array([0, 1]))
    assert np.allclose(R, [[1 - 4 / 6, 1 - 6 / 12]])
    assert np.allclose(regret_matrix(V, Vs, np.array([0, 0, 1])), [[1 - 5 / 8, 1 - 8 / 16]])


def test_strata_split_even_odd():
    cell = np.repeat([0, 1], 6)
    idx = np.tile(np.arange(6), 2)
    fit, ev = strata(cell, idx)
    assert all((idx[g] % 2 == 0).all() for g in fit) and all((idx[g] % 2 == 1).all() for g in ev)
    assert sorted(np.concatenate(fit + ev)) == list(range(12))


def synthetic(Rtrue, n_cell=20, seed=0):
    """Run-like dataframe whose pooled regret equals Rtrue (rows: defenders, columns: strategies) up to small jitter."""
    rng = np.random.default_rng(seed)
    rows = []
    for rho, s, a in itertools.product((1.1, 1.5), (2, 4), (0.5, 0.9)):
        for i in range(n_cell):
            row = dict(rho=rho, s=s, a=a, idx=i, violations="")
            for j, k in enumerate(STRAT_NAMES):
                row[f"Vstar_{k}"] = 10.0
                for d_i, d in enumerate(DEFENDERS):
                    row[f"V_{d}_{k}"] = 10.0 * (1 - Rtrue[d_i, j]) + rng.normal(0, 0.05)
            rows.append(row)
    return pd.DataFrame(rows)


def base_R():
    R = np.full((8, 6), 0.02)
    return R


def test_flags_exposed_and_defended():
    R = base_R()
    for d in NONADAPTIVE:
        R[DEFENDERS.index(d), STRAT_NAMES.index("SH14")] = 0.20
    res = analyse(synthetic(R), "B40", B=60)
    f = flags(res)
    assert f["X"] and f["R"] and not f["S"]
    assert abs(res["point"]["E_nom"] - 0.20) < 0.01 and abs(res["point"]["E_def"] - 0.02) < 0.01


def test_flags_stop_when_nominal_is_robust():
    res = analyse(synthetic(base_R()), "B40", B=60)
    assert not flags(res)["X"]


def test_flags_exposed_when_nobody_defends():
    R = base_R()
    for d in DEFENDERS:
        R[DEFENDERS.index(d), STRAT_NAMES.index("SH14")] = 0.20
    f = flags(analyse(synthetic(R), "B40", B=60))
    assert f["X"] and not f["R"]


def test_S_flag_when_shared_adaptation_loses_on_partial_strategy():
    R = base_R()
    for d in NONADAPTIVE:
        R[DEFENDERS.index(d), STRAT_NAMES.index("SH14")] = 0.20
    for d in ("QMDP_B", "QMDP_H", "BA_H"):
        R[DEFENDERS.index(d), STRAT_NAMES.index("PH50_18")] = 0.08
    res = analyse(synthetic(R), "B40", B=60)
    assert res["point"]["S_PH50_18"] > 0.05 and flags(res)["S"]
    R2 = base_R()
    for d in NONADAPTIVE:
        R2[DEFENDERS.index(d), STRAT_NAMES.index("SH14")] = 0.20
    assert not flags(analyse(synthetic(R2), "B40", B=60))["S"]


def test_exploitability_is_max_over_strategies_and_uses_odd_half():
    df = synthetic(base_R())
    even = df["idx"] % 2 == 0
    for d in DEFENDERS:                                       # make the even half look terrible, odd half fine
        df.loc[even, f"V_{d}_SH14"] = 1.0
    res = analyse(df, "B40", B=20)
    assert res["point"]["E"].max() < 0.05                     # odd half only
    df2 = synthetic(base_R())
    df2.loc[~even, "V_DPMIS_1.0_NONE"] = 9.0                  # one strategy of one defender is bad on the odd half
    r2 = analyse(df2, "B40", B=20)
    assert abs(r2["point"]["E"][DEFENDERS.index("DPMIS_1.0")] - 0.10) < 0.01


def test_mixture_ignores_the_odd_half():
    df = synthetic(base_R())
    V, Vs, cell, idx = build_arrays(df)
    fit, ev = strata(cell, idx)
    Rf = regret_matrix(V, Vs, np.concatenate(fit))
    x1 = stats_from(Rf, regret_matrix(V, Vs, np.concatenate(ev)))["x"]
    V2 = V.copy(); V2[idx % 2 == 1] *= 0.5
    x2 = stats_from(regret_matrix(V2, Vs, np.concatenate(fit)), regret_matrix(V2, Vs, np.concatenate(ev)))["x"]
    assert np.allclose(x1, x2)


def test_decision_table_in_order():
    ok = dict(X=True, R=True, S=True)
    assert decide(dict(X=False, R=True, S=True), None) == "STOP"
    assert decide(dict(X=True, R=False, S=True), None) == "EXPOSED"
    assert decide(ok, None).startswith("PENDING")
    assert decide(ok, dict(X=False, R=True, S=True)) == "REGIME-LIMITED"
    assert decide(ok, dict(X=True, R=False, S=True)) == "REGIME-LIMITED"
    assert decide(dict(X=True, R=True, S=False), ok) == "ADAPTATION-SUFFICES"
    assert decide(ok, ok) == "STRONG-ADVERSARIAL"
