"""Tests of the A13 go-rule logic (src/dwell/analysis_a13.py)."""
import numpy as np
import pandas as pd

from dwell.analysis_a13 import cell_summary, decide, find_regions, outside_corner
from dwell.m0g import METHODS


def cells():
    return [(n, rho, s, a, g) for n in (4, 5) for rho in (1.1, 1.5) for s in (2, 4) for a in (0.5, 0.9)
            for g in ("SPREAD", "CLUSTER")]


def test_adjacent_on_each_axis_is_a_region():
    base = (4, 1.1, 2, 0.5, "SPREAD")
    for nb in [(5, 1.1, 2, 0.5, "SPREAD"), (4, 1.5, 2, 0.5, "SPREAD"), (4, 1.1, 4, 0.5, "SPREAD"),
               (4, 1.1, 2, 0.9, "SPREAD"), (4, 1.1, 2, 0.5, "CLUSTER")]:
        f = {c: False for c in cells()}
        f[base] = True; f[nb] = True
        assert decide(f, f)["outcome"] == "GO"


def test_diagonal_cells_are_not_a_region():
    f = {c: False for c in cells()}
    f[(4, 1.1, 2, 0.5, "SPREAD")] = True; f[(5, 1.5, 2, 0.5, "SPREAD")] = True
    assert find_regions(f) == []
    g = {c: False for c in cells()}
    g[(4, 1.1, 2, 0.5, "SPREAD")] = True; g[(4, 1.1, 2, 0.5, "CLUSTER")] = True; g[(5, 1.1, 2, 0.9, "CLUSTER")] = True
    assert len(find_regions(g)) == 1


def test_marginal_and_nogo():
    f = {c: False for c in cells()}
    p = dict(f); p[(4, 1.1, 2, 0.5, "SPREAD")] = True; p[(4, 1.1, 2, 0.5, "CLUSTER")] = True
    assert decide(f, p)["outcome"] == "MARGINAL"
    assert decide(f, f)["outcome"] == "NO-GO"


def test_corner_cells_excluded():
    assert not outside_corner(1.5, 0) and not outside_corner(2.0, 2) and outside_corner(1.5, 2)
    f = {(4, 1.5, 0, 0.5, "SPREAD"): True, (5, 1.5, 0, 0.5, "SPREAD"): True}
    assert find_regions(f) == []


def _df(regret, n=200, seed=0):
    rng = np.random.default_rng(seed)
    dp = rng.uniform(5, 9, size=n)
    d = {"idx": np.arange(n), "V_dp": dp}
    for m in METHODS:
        d["V_" + m] = dp * 0.5
    d["V_ROLL_CEP"] = dp * (1 - regret)
    return pd.DataFrame(d)


def test_cell_flag_uses_lower_bound_and_best_method():
    s = cell_summary(_df(0.10), 0)
    assert s["flagged"] and s["best_method"] == "ROLL_CEP" and s["best_is_rollout"]
    t = cell_summary(_df(0.04), 0)
    assert not t["flagged"] and not t["point_flagged"]
    assert cell_summary(_df(0.0501), 0)["point_flagged"] is True


def test_noisy_cell_is_point_flagged_but_not_flagged():
    """Point estimate 6% but a wide interval: the lower bound, not the point estimate, decides the flag."""
    n = 20
    dp = np.full(n, 8.0)
    d = {"idx": np.arange(n), "V_dp": dp}
    for m in METHODS:
        d["V_" + m] = dp * 0.5
    r = np.zeros(n); r[:6] = 0.2                       # mean regret 0.06
    d["V_ROLL_CEP"] = dp * (1 - r)
    s = cell_summary(pd.DataFrame(d), 0)
    assert s["point_flagged"] and not s["flagged"] and s["lo5"] < 0.05 <= s["regret_best"]
