"""
Analysis logic of amendment A17 (plan A17.4, A17.7): robustness of the planner claims across instance families.
Rows are those of src/dwell/a16.py at level "ru" (CEP, CEPS, RU_CEP); one group per family (base, K3, wide, equal, spread).
Informative cell: mean U of RU_CEP in [0.10, 0.99]. Intervals: instance-level bootstrap stratified by cell (B = 2000, seed 20261282), one-sided 95%.
Per variant family f and size n (K3 and the others at n = 7 and 9: 8 statements):
  P1(f, n): lower bound of the mean of U_CEPS - U_CEP >= 0.01.
  cheap planner: ADEQUATE if the upper bound of the mean of U_RU_CEP - U_CEPS is <= 0.03; ROLLOUT-WORTH if its lower bound is >= 0.03; else INCONCLUSIVE.
P3 pooled: over the informative instances of all variant families (n = 7 and 9), the mean gain U_CEPS - U_CEP for 0.8 <= tau <= 1.1 minus that of the others has a lower bound >= 0.03.
Overall label (P1 only): ROBUST if P1 holds in all 8 statements, MOSTLY if in at least 6, NOT ROBUST otherwise; NO OUTCOME if a statement is undefined.
The base family is the same-seed control and is reported, not part of the label.
"""
import numpy as np
import pandas as pd

from .analysis_a16 import BAND, CELL, P1_MIN, P2_MAX, P3_MIN, band_diff, boot_mean, diff_strata, informative

SEED17 = 20261282
VARIANTS = ("K3", "wide", "equal", "spread")
SIZES = (7, 9)
MOSTLY_MIN = 6


def family_stats(d: pd.DataFrame) -> dict:
    """d: the rows of one family and size. Returns the interval statistics of the informative cells."""
    inf = informative(d)
    if len(inf) == 0:
        return dict(undefined=True)
    m, lo, hi = boot_mean(diff_strata(inf, "U_CEPS", "U_CEP"), seed=SEED17)
    m2, lo2, hi2 = boot_mean(diff_strata(inf, "U_RU_CEP", "U_CEPS"), seed=SEED17)
    adequacy = "ADEQUATE" if hi2 <= P2_MAX else ("ROLLOUT-WORTH" if lo2 >= P2_MAX else "INCONCLUSIVE")
    return dict(undefined=False, instances=len(inf), cells=len(inf.groupby(CELL)), gain=m, gain_lo=lo, gain_hi=hi, p1=bool(lo >= P1_MIN),
                ru_gap=m2, ru_gap_lo=lo2, ru_gap_hi=hi2, adequacy=adequacy)


def claims17(df: pd.DataFrame, sizes_by_family: dict) -> dict:
    out, parts = {}, []
    for f in ("base",) + VARIANTS:
        for n in sizes_by_family.get(f, SIZES):
            d = df[(df["group"] == f) & (df["n"] == n)]
            out[(f, n)] = family_stats(d) if len(d) else dict(undefined=True)
            if f in VARIANTS and not out[(f, n)]["undefined"]:
                inf = informative(d)
                for _, g in inf.groupby(CELL):
                    tau = g["tau"].to_numpy()
                    parts.append(((g["U_CEPS"] - g["U_CEP"]).to_numpy(), (tau >= BAND[0]) & (tau <= BAND[1])))
    if parts:
        pt, lo = band_diff(parts, seed=SEED17)
        p3 = dict(diff=pt, lo5=lo, holds=bool(lo >= P3_MIN))
    else:
        p3 = dict(holds=False, undefined=True)
    keys = [(f, n) for f in VARIANTS for n in sizes_by_family.get(f, SIZES)]
    if any(out[k]["undefined"] for k in keys):
        label = "NO OUTCOME"
    else:
        holds = sum(bool(out[k]["p1"]) for k in keys)
        label = "ROBUST" if holds == len(keys) else ("MOSTLY" if holds >= MOSTLY_MIN else "NOT ROBUST")
    return dict(families=out, P3_pooled=p3, label=label, n_statements=len(keys))
