"""
Analysis logic of amendment A18 (plan A18.4, A18.5): the survival objective under a wrong dwell scale. Reads per-instance rows only
(src/dwell/a18.py). One statement per true scale (0.8, 1.2, 1.4) and size n (5, 7, 9); scale 1.0 is the same-seed control.
Informative cell (scale, n, rho, s, a, lam): the mean survival probability of the INFORMED survival-aware planner CEPS_T lies in [0.10, 0.99]
(a reference that does not depend on the planners under test). Intervals: instance-level bootstrap stratified by cell (B = 2000,
seed 20261292), one-sided 95%.
P1(scale, n): lower bound of the mean of U_CEPS - U_CEP (both planners built on the nominal model, evaluated with the true dwell) >= 0.01.
Label (P1 only), first match: NO OUTCOME (a statement has no informative cell), ROBUST (P1 holds in all 9), MOSTLY (in at least 7),
NOT ROBUST. The control (scale 1.0) is reported and is not part of the label.
Secondary, not part of the label:
  cost of the wrong scale: mean of U_CEPS_T - U_CEPS; SMALL if its upper bound <= 0.03, LARGE if its lower bound >= 0.03, else INCONCLUSIVE.
  awareness against knowledge: mean of U_CEPS - U_CEP_T; AWARENESS-WINS if its lower bound >= 0.01, KNOWLEDGE-WINS if its upper bound <= -0.01, else MIXED.
  cheap planner: mean of U_RU_CEP - U_CEPS; ADEQUATE if its upper bound <= 0.03, ROLLOUT-WORTH if its lower bound >= 0.03, else INCONCLUSIVE.
"""
import numpy as np
import pandas as pd

from .analysis_a16 import B_BOOT, INFO_HI, INFO_LO, boot_mean

CELL = ["n", "scale", "rho", "s", "a", "lam"]
REF = "U_CEPS_T"
P1_MIN, C_MAX = 0.01, 0.03
SCALES, SIZES, CONTROL = (0.8, 1.2, 1.4), (5, 7, 9), 1.0
MOSTLY_MIN = 7
SEED18 = 20261292
EXACT_METHODS = ("CEP", "CEPS", "RU_CEP", "DPEV", "RU_DPEV")


def informative(d: pd.DataFrame) -> pd.DataFrame:
    keep = [g for _, g in d.groupby(CELL) if INFO_LO <= g[REF].mean() <= INFO_HI]
    return pd.concat(keep) if keep else d.iloc[0:0]


def strata(d, a, b):
    return [(g[a] - g[b]).to_numpy() for _, g in d.groupby(CELL)]


def stats(d: pd.DataFrame) -> dict:
    """d: the rows of one scale and size. Statistics over its informative cells."""
    inf = informative(d)
    if len(inf) == 0:
        return dict(undefined=True)
    g, g_lo, g_hi = boot_mean(strata(inf, "U_CEPS", "U_CEP"), seed=SEED18)
    c, c_lo, c_hi = boot_mean(strata(inf, "U_CEPS_T", "U_CEPS"), seed=SEED18)
    w, w_lo, w_hi = boot_mean(strata(inf, "U_CEPS", "U_CEP_T"), seed=SEED18)
    out = dict(undefined=False, instances=len(inf), cells=len(inf.groupby(CELL)), gain=g, gain_lo=g_lo, gain_hi=g_hi, p1=bool(g_lo >= P1_MIN),
               cost=c, cost_lo=c_lo, cost_hi=c_hi, cost_class="SMALL" if c_hi <= C_MAX else ("LARGE" if c_lo >= C_MAX else "INCONCLUSIVE"),
               aware=w, aware_lo=w_lo, aware_hi=w_hi,
               aware_class="AWARENESS-WINS" if w_lo >= P1_MIN else ("KNOWLEDGE-WINS" if w_hi <= -P1_MIN else "MIXED"))
    if "U_RU_CEP" in inf.columns and inf["U_RU_CEP"].notna().all():
        r, r_lo, r_hi = boot_mean(strata(inf, "U_RU_CEP", "U_CEPS"), seed=SEED18)
        out.update(ru_gap=r, ru_gap_lo=r_lo, ru_gap_hi=r_hi, adequacy="ADEQUATE" if r_hi <= C_MAX else ("ROLLOUT-WORTH" if r_lo >= C_MAX else "INCONCLUSIVE"))
    return out


def regret_table(d: pd.DataFrame) -> dict:
    """Exact level (rows with U_opt_T): mean regret against the exact optimum of the true model (points), informative cells."""
    inf = informative(d)
    inf = inf[inf["U_opt_T"].notna()] if "U_opt_T" in inf.columns else inf.iloc[0:0]
    if len(inf) == 0:
        return {}
    cols = [m for m in EXACT_METHODS if "U_" + m in inf.columns] + ["CEP_T", "CEPS_T"]
    return {m: float(100 * (inf["U_opt_T"] - inf["U_" + m]).mean()) for m in cols}


def claims18(df: pd.DataFrame, sizes=SIZES, scales=SCALES) -> dict:
    res = {}
    for sc in (CONTROL,) + tuple(scales):
        for n in sizes:
            d = df[(df["scale"] == sc) & (df["n"] == n)]
            res[(sc, n)] = stats(d) if len(d) else dict(undefined=True)
    keys = [(sc, n) for sc in scales for n in sizes]
    if any(res[k]["undefined"] for k in keys):
        label = "NO OUTCOME"
    else:
        holds = sum(bool(res[k]["p1"]) for k in keys)
        label = "ROBUST" if holds == len(keys) else ("MOSTLY" if holds >= MOSTLY_MIN else "NOT ROBUST")
    return dict(statements=res, label=label, n_statements=len(keys))
