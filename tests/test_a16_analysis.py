"""Tests of the A16 analysis logic (plan A16.7, items 4 to 6): informative cells, bootstrap intervals, claims P1 to P3, outcomes."""
import numpy as np
import pandas as pd

from dwell.analysis_a16 import (BAND, P1_MIN, P2_MAX, P3_MIN, band_diff, boot_mean, claims, diff_strata, informative)


def cell(n, gain, ru_minus_ceps, tau, rho=1.1, s=2, a=0.5, lam=0.25, rows=40, ceps=0.5, ru=True):
    rng = np.random.default_rng(n + int(rho * 10))
    tau = np.asarray(tau, dtype=float) if np.ndim(tau) else np.full(rows, tau)
    g = np.asarray(gain, dtype=float) if np.ndim(gain) else np.full(rows, gain)
    d = dict(group="main", n=n, rho=rho, s=s, a=a, lam=lam, U_CEP=np.clip(ceps - g, 0, 1), U_CEPS=np.full(rows, ceps), tau=tau)
    if ru:
        d["U_RU_CEP"] = np.full(rows, ceps + ru_minus_ceps)
    return pd.DataFrame(d)


def dfs(gain_in=0.08, gain_out=0.0, ru_gap=0.01, ru7=True):
    out = {}
    for n in (5, 7, 9):
        tau = np.where(np.arange(40) < 20, 0.95, 1.6)
        gain = np.where(np.arange(40) < 20, gain_in, gain_out)
        out[n] = pd.concat([cell(n, gain, ru_gap, tau, rho=1.1), cell(n, gain, ru_gap, tau, rho=1.5)])
    return out


def test_informative_uses_the_reference_method():
    a = cell(7, 0.0, 0.0, 1.0, ceps=0.5)
    b = cell(7, 0.0, 0.0, 1.0, rho=1.5, ceps=0.995)
    assert len(informative(pd.concat([a, b]))) == 40
    # RU_CEP present: it decides, not CEPS
    c = cell(7, 0.0, -0.45, 1.0, ceps=0.5)                       # RU_CEP = 0.05: outside, although CEPS = 0.5
    assert len(informative(c)) == 0
    # no RU_CEP column: CEPS decides
    assert len(informative(cell(11, 0.0, 0.0, 1.0, ceps=0.5, ru=False))) == 40


def test_bootstrap_is_stratified_deterministic_and_ordered():
    s = [np.full(30, 0.05), np.full(20, 0.05)]
    assert boot_mean(s) == (0.05, 0.05, 0.05) or all(abs(x - 0.05) < 1e-12 for x in boot_mean(s))
    rng = np.random.default_rng(0)
    s2 = [rng.normal(0.02, 0.1, 50), rng.normal(0.02, 0.1, 40)]
    m, lo, hi = boot_mean(s2)
    assert lo < m < hi and boot_mean(s2) == (m, lo, hi)


def test_diff_strata_sign():
    d = cell(7, 0.05, 0.02, 1.0)
    assert abs(diff_strata(d, "U_CEPS", "U_CEP")[0].mean() - 0.05) < 1e-9 and abs(diff_strata(d, "U_RU_CEP", "U_CEPS")[0].mean() - 0.02) < 1e-9


def test_band_difference():
    strata = [(np.r_[np.full(20, 0.1), np.zeros(20)], np.r_[np.ones(20, bool), np.zeros(20, bool)])] * 2
    pt, lo = band_diff(strata)
    assert abs(pt - 0.1) < 1e-12 and lo > 0.09
    flat = [(np.zeros(40), np.r_[np.ones(20, bool), np.zeros(20, bool)])]
    assert abs(band_diff(flat)[0]) < 1e-12 and band_diff(flat)[1] <= 1e-12


def test_band_limits_are_both_used():
    g = np.r_[np.full(10, 0.1), np.zeros(30)]
    tau = np.r_[np.full(10, 0.95), np.full(10, 1.2), np.full(10, 1.6), np.full(10, 0.5)]
    d = pd.DataFrame(dict(group="main", n=7, rho=1.1, s=2, a=0.5, lam=0.25, U_CEP=0.5 - g, U_CEPS=0.5, U_RU_CEP=0.5, tau=tau))
    out = claims({5: d.assign(n=5), 7: d, 9: d.assign(n=9)})
    assert out["P3"]["diff"] > 0.07                                           # only the 10 instances with tau in [0.8, 1.1] are in the band


def test_confirmed_partial_not_confirmed_and_no_outcome():
    assert claims(dfs())["outcome"] == "CONFIRMED"
    assert claims(dfs(ru_gap=0.06))["outcome"] == "PARTIAL"                  # RU_CEP better than CEPS by 6 points: P2 fails
    assert claims(dfs(gain_in=0.0))["outcome"] == "NOT CONFIRMED"
    assert claims(dfs(gain_in=0.03, gain_out=0.03))["outcome"] == "PARTIAL"   # P1 and P2 hold, the regime difference is zero: P3 fails
    partial = dfs(); del partial[9]
    assert claims(partial)["outcome"] == "NO OUTCOME"


def test_thresholds_are_the_planned_ones():
    assert (P1_MIN, P2_MAX, P3_MIN, BAND) == (0.01, 0.03, 0.03, (0.8, 1.1))


def test_p1_uses_the_lower_bound_and_p2_the_upper_bound():
    noisy = {}
    for n in (5, 7, 9):
        g = np.r_[np.zeros(9), 0.5]                                           # mean 0.05 from one instance in ten: the lower bound is 0
        noisy[n] = cell(n, g, 0.01, 1.0, rows=10)
    out = claims(noisy)
    assert out["P1_n7"]["mean"] > P1_MIN and not out["P1_n7"]["holds"]
    wide = {n: cell(n, 0.05, np.r_[np.full(5, -0.2), np.full(5, 0.2)], 1.0, rows=10) for n in (5, 7, 9)}
    o2 = claims(wide)
    assert o2["P2_n7"]["mean"] <= P2_MAX and not o2["P2_n7"]["holds"]


def test_p3_uses_the_lower_bound():
    g = np.r_[np.zeros(9), 0.6, np.zeros(30)]                               # in-band mean gain 0.06 from one instance in ten
    tau = np.r_[np.full(10, 0.95), np.full(30, 1.6)]
    d = {n: cell(n, g, 0.01, tau, rows=40) for n in (5, 7, 9)}
    out = claims(d)
    assert out["P3"]["diff"] >= P3_MIN and not out["P3"]["holds"]
