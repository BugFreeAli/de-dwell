"""Tests of the A11 go rule logic (src/dwell/analysis_a11.py)."""
import numpy as np
import pandas as pd

from dwell.analysis_a11 import cell_summary, decide, find_regions, outside_corner
from dwell.rules1 import METHODS


def cells():
    return [(rho, s, a, g, e) for rho in (1.1, 1.5) for s in (2, 4) for a in (0.5, 0.9) for g in (2, 4) for e in ("0.5", "0.8", "inf")]


def test_adjacent_on_eps_axis_is_a_region_and_budget_driven():
    f = {c: False for c in cells()}
    f[(1.1, 2, 0.5, 2, "0.5")] = True; f[(1.1, 2, 0.5, 2, "0.8")] = True
    res = decide(f, f)
    assert res["outcome"] == "GO" and res["kind"] == "budget-driven"


def test_inf_only_region_is_ordering_only():
    f = {c: False for c in cells()}
    f[(1.1, 2, 0.5, 2, "inf")] = True; f[(1.1, 2, 0.5, 4, "inf")] = True
    assert decide(f, f)["kind"] == "ordering-only"


def test_diagonal_and_gap_are_not_regions():
    f = {c: False for c in cells()}
    f[(1.1, 2, 0.5, 2, "0.5")] = True; f[(1.5, 4, 0.5, 2, "0.5")] = True        # differ in two factors
    assert find_regions(f) == []
    g = {c: False for c in cells()}
    g[(1.1, 2, 0.5, 2, "0.5")] = True; g[(1.1, 2, 0.5, 2, "inf")] = True          # eps 0.5 and inf are not adjacent (0.8 between)
    assert find_regions(g) == []


def test_marginal_and_nogo():
    f = {c: False for c in cells()}
    p = dict(f); p[(1.1, 2, 0.5, 2, "0.5")] = True; p[(1.1, 2, 0.5, 2, "0.8")] = True
    assert decide(f, p)["outcome"] == "MARGINAL"
    assert decide(f, f)["outcome"] == "NO-GO"


def test_corner_cells_excluded():
    assert not outside_corner(1.5, 0) and not outside_corner(2.0, 2) and outside_corner(1.5, 2)
    f = {(1.5, 0, 0.5, 2, "0.5"): True, (1.5, 0, 0.5, 2, "0.8"): True}
    assert find_regions(f) == []


def _df(regret, n=200, seed=0):
    rng = np.random.default_rng(seed)
    dp = rng.uniform(5, 9, size=n)
    d = {"idx": np.arange(n), "V_dp": dp, "rho_hat": 1.0, "rho_boost": 0.8}
    for m in METHODS:
        d["V_" + m] = dp * (1 - 0.5)                    # weak methods
    d["V_ROLL_CEP_L"] = dp * (1 - regret)               # the best method has exactly this regret
    return pd.DataFrame(d)


def test_cell_flag_uses_lower_bound():
    assert cell_summary(_df(0.10), 0)["flagged"]
    s = cell_summary(_df(0.04), 0)
    assert not s["flagged"] and not s["point_flagged"]
    assert cell_summary(_df(0.10), 0)["best_method"] == "ROLL_CEP_L"
