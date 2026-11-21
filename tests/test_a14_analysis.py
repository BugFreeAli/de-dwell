"""Tests of the A14 analysis logic (plan A14.7, items 7 to 9): regions, informative cells, flags, decision table."""
import numpy as np
import pandas as pd

from dwell.analysis_a14 import (ABS_THRESH, AXES, REL_THRESH, best_over, cell_summary, decide, find_regions,
                                outside_corner, summarise)
from dwell.m0u import AWARE, EV_BASED, METHODS


def frame(u_opt, gaps, n=60, seed=0, key=(4, 1.1, 2, 0.5, 0.25)):
    """Synthetic cell: every method has U = u_opt - gap (gaps: dict method -> gap, default 0.5 so that it is clearly bad)."""
    rng = np.random.default_rng(seed)
    base = np.clip(u_opt + rng.normal(0, 0.01, n), 0, 1)
    d = {"idx": np.arange(n), "U_opt": base}
    for m in METHODS:
        d["U_" + m] = np.clip(base - gaps.get(m, 0.5), 0, 1)
    g = pd.DataFrame(d)
    for ax, v in zip(AXES, key):
        g[ax] = v
    return g


def test_corner_rule():
    assert outside_corner(1.5, 2) and outside_corner(0.8, 4) and not outside_corner(2.0, 2) and not outside_corner(1.1, 1)


def test_regions_need_one_factor_one_step():
    c1, c2, c3, c4 = (4, 0.8, 2, 0.5, 0.25), (4, 1.1, 2, 0.5, 0.25), (4, 1.5, 2, 0.5, 0.25), (4, 0.8, 4, 0.5, 0.5)
    allc = [c1, c2, c3, c4, (4, 1.1, 4, 0.5, 0.25)]
    assert find_regions({c1: True, c2: True}, allc) == [(c1, c2)]
    assert find_regions({c1: True, c3: True}, allc) == []                 # two steps apart in rho
    assert find_regions({c1: True, c4: True}, allc) == []                 # two factors differ (s and lam)
    assert find_regions({c1: True}, allc) == []


def test_informative_range_blocks_flags():
    for u_opt in (0.05, 0.995):                      # a large loss, but the cell is outside 0.10 <= mean(U_opt) <= 0.99
        s = cell_summary(frame(u_opt, {m: 0.5 for m in METHODS}), 0)
        assert not s["informative"] and not s["flagged_all"] and not s["flagged_ev"] and not s["point_all"]
    s = cell_summary(frame(0.5, {m: 0.2 for m in METHODS}), 0)
    assert s["informative"] and s["flagged_all"] and s["flagged_ev"] and s["point_all"]
    s = cell_summary(frame(0.5, {m: 0.0 for m in METHODS}), 0)
    assert s["informative"] and not s["flagged_all"] and not s["flagged_ev"]


def test_flag_needs_relative_and_absolute_gap():
    # all methods lose 0.04 of a survival probability near 0.3: relative 13% but absolute 4 points -> flagged
    s = cell_summary(frame(0.3, {m: 0.04 for m in METHODS}), 0)
    assert s["flagged_all"]
    # lose 0.012: relative 4% < 5% -> not flagged
    s = cell_summary(frame(0.3, {m: 0.012 for m in METHODS}), 0)
    assert not s["flagged_all"]
    # relative 20% but absolute 1 point (U_opt about 0.1 -> gap 0.02 is the floor; use 0.012 at U_opt 0.12)
    s = cell_summary(frame(0.12, {m: 0.012 for m in METHODS}), 0)
    assert s["regret_best"] > REL_THRESH and s["abs_best"] < ABS_THRESH and not s["flagged_all"] and not s["point_all"]


def test_flag_uses_the_bound_not_the_point_estimate():
    g = frame(0.4, {m: 0.0 for m in METHODS}, n=10)
    for m in METHODS:                                    # one instance in ten loses 0.5 of the probability for every method
        u = g["U_" + m].to_numpy().copy()
        u[0] = max(0.0, u[0] - 0.5)
        g["U_" + m] = u
    s = cell_summary(g, 0)
    assert s["point_all"] and not s["flagged_all"]       # point estimate above both thresholds, lower bound is not
    assert s["lo5"] < REL_THRESH <= s["regret_best"]


def test_best_over_matches_manual():
    g = frame(0.5, {"CEPS": 0.01, "DPEV": 0.1})
    U = np.stack([g["U_" + m].to_numpy() for m in METHODS])
    rel, ab = best_over(U, g["U_opt"].to_numpy())
    k = METHODS.index("CEPS")
    assert abs(rel - (1 - g["U_CEPS"].sum() / g["U_opt"].sum())) < 1e-12
    assert abs(ab - (g["U_opt"].sum() - g["U_CEPS"].sum()) / len(g)) < 1e-12


def test_sets_ev_and_all():
    # only an aware method is good: ALL is not flagged, EV is
    gaps = {m: 0.0 if m == "CEPS" else 0.2 for m in METHODS}
    s = cell_summary(frame(0.5, gaps), 0)
    assert s["best_method"] == "CEPS" and s["best_is_aware"] and not s["flagged_all"] and s["flagged_ev"]
    # only an EV-based method is good: both sets share the best method
    gaps = {m: 0.0 if m == "DPEV" else 0.2 for m in METHODS}
    s = cell_summary(frame(0.5, gaps), 0)
    assert s["best_method"] == "DPEV" and not s["best_is_aware"] and not s["flagged_all"] and not s["flagged_ev"]


def test_decision_table():
    a, b = (4, 0.8, 2, 0.5, 0.25), (4, 1.1, 2, 0.5, 0.25)
    t, f = {a: True, b: True}, {a: False, b: False}
    assert decide(t, t, t)["outcome"] == "GO"
    assert decide(f, t, t)["outcome"] == "MISMATCH"
    assert decide(f, f, t)["outcome"] == "MARGINAL"
    assert decide(f, f, f)["outcome"] == "NO-GO"
    assert decide({a: True, b: False}, {a: True, b: False}, {a: True, b: False})["outcome"] == "NO-GO"   # one cell is no region


def test_summarise_runs_on_two_cells_and_is_order_independent():
    g1, g2 = frame(0.5, {m: 0.05 for m in METHODS}, key=(4, 0.8, 2, 0.5, 0.25)), frame(0.5, {}, key=(4, 1.1, 2, 0.5, 0.25))
    a = summarise(pd.concat([g1, g2], ignore_index=True))
    b = summarise(pd.concat([g2, g1], ignore_index=True))
    assert a.equals(b) and len(a) == 2
