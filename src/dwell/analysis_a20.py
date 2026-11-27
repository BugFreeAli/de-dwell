"""
Analysis logic of amendment A20 (plan A20.4, A20.5): robustness of the scale-learning planner. Reads per-instance rows only (src/dwell/a20.py).
True dwell conditions: control "1.0" (dwell (5, 10)); in-support scales "0.8", "1.2", "1.4"; outside the prior "0.6" (3, 6) and "1.6" (8, 16); class specific "SP" (4, 12)
and "CP" (6, 8). Informative cell (cond, n, rho, s, a, lam): mean U_CEPS_T in [0.10, 0.99]. Instance-level bootstrap stratified by cell (B = 2000, seed 20261313),
one-sided 95% bounds.
Statements:
  Q0(n), control, n = 5, 7:            upper bound of mean(U_RU_CLAMP - U_ADAPT) <= 0.03      (learning costs little against the non-learning rollout planner when the nominal model is right)
  Q1(cond, n), cond in "0.8", "1.2", "1.4", n = 5, 7 (6):  lower bound of mean(U_ADAPT - U_RU_CLAMP) >= 0.01   (learning adds to not abandoning plus a rollout)
  Q2(cond, n), cond in "0.6", "1.6", "SP", "CP", n = 5, 7 (8):  lower bound of mean(U_ADAPT - U_NOM) >= 0.03  (the learner still beats the nominal planner outside its prior or with a class-specific error)
  Q3 (all 14 non-control statements): class of ADAPT against RU_CLAMP: LEARNING-BETTER (lower bound >= 0.01), CLAMP-BETTER (upper bound <= -0.01), else MIXED (descriptive).
Label, first match: NO OUTCOME (a Q0 or Q1 statement has no informative cell, or fewer than 6 of the 8 Q2 statements have one); ROBUST-LEARNER (Q1 in at least 5 of 6, Q0 in 2 of 2
and Q2 in all of its defined statements); USEFUL (Q2 in at least 5 statements and Q0 in 2 of 2); LIMITED (Q2 in at least 1); NO GAIN. Second label: LEARNING-NEEDED if Q1 holds
in at least 5 of 6, else CLAMP-ROLLOUT-ENOUGH. A statement without an informative cell is UNDEFINED and is not counted as holding.
"""
import numpy as np
import pandas as pd

from .analysis_a16 import INFO_HI, INFO_LO, boot_mean

CELL = ["n", "cond", "rho", "s", "a", "lam"]
REF = "U_CEPS_T"
Q0_MAX, Q1_MIN, Q2_MIN, Q3_MIN = 0.03, 0.01, 0.03, 0.01
CONTROL, IN_SUPPORT, OUT = "1.0", ("0.8", "1.2", "1.4"), ("0.6", "1.6", "SP", "CP")
SIZES = (5, 7)
Q1_MIN_COUNT, Q2_DEFINED_MIN, Q2_USEFUL_MIN = 5, 6, 5
SEED20 = 20261313
METHODS = ("NOM", "CLAMP", "RU_CLAMP", "ADAPT", "ADAPT_V", "CEPS_T", "RU_CEP_T")


def informative(d: pd.DataFrame) -> pd.DataFrame:
    keep = [g for _, g in d.groupby(CELL) if INFO_LO <= g[REF].mean() <= INFO_HI]
    return pd.concat(keep) if keep else d.iloc[0:0]


def strata(d, a, b):
    return [(g[a] - g[b]).to_numpy() for _, g in d.groupby(CELL)]


def stats(d: pd.DataFrame) -> dict:
    """d: the rows of one condition and size. Statistics over its informative cells."""
    inf = informative(d)
    if len(inf) == 0:
        return dict(undefined=True)
    a_r, a_r_lo, a_r_hi = boot_mean(strata(inf, "U_ADAPT", "U_RU_CLAMP"), seed=SEED20)
    r_a, r_a_lo, r_a_hi = boot_mean(strata(inf, "U_RU_CLAMP", "U_ADAPT"), seed=SEED20)
    a_n, a_n_lo, a_n_hi = boot_mean(strata(inf, "U_ADAPT", "U_NOM"), seed=SEED20)
    a_c, a_c_lo, a_c_hi = boot_mean(strata(inf, "U_ADAPT", "U_CLAMP"), seed=SEED20)
    a_i, a_i_lo, a_i_hi = boot_mean(strata(inf, "U_RU_CEP_T", "U_ADAPT"), seed=SEED20)
    return dict(undefined=False, instances=len(inf), cells=len(inf.groupby(CELL)),
                vs_ru_clamp=a_r, vs_ru_clamp_lo=a_r_lo, vs_ru_clamp_hi=a_r_hi,
                q0=bool(r_a_hi <= Q0_MAX), ru_clamp_minus_adapt=r_a, ru_clamp_minus_adapt_hi=r_a_hi,
                q1=bool(a_r_lo >= Q1_MIN),
                vs_nom=a_n, vs_nom_lo=a_n_lo, vs_nom_hi=a_n_hi, q2=bool(a_n_lo >= Q2_MIN),
                vs_clamp=a_c, vs_clamp_lo=a_c_lo, vs_clamp_hi=a_c_hi,
                informed_minus_adapt=a_i, informed_minus_adapt_hi=a_i_hi,
                q3="LEARNING-BETTER" if a_r_lo >= Q3_MIN else ("CLAMP-BETTER" if a_r_hi <= -Q3_MIN else "MIXED"))


def timing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean time per policy call (ms, memoised within the instance) and mean number of calls per instance, by size and method (all rows)."""
    rows = []
    for n, g in df.groupby("n"):
        for m in ("NOM", "CLAMP", "RU_CLAMP", "ADAPT", "ADAPT_V"):
            if "ms_" + m in g.columns:
                rows.append(dict(n=n, method=m, ms_per_call=float(g["ms_" + m].mean()), calls_per_instance=float(g["calls_" + m].mean())))
    return pd.DataFrame(rows)


def regret_table(d: pd.DataFrame) -> dict:
    """Exact level (rows with U_opt_T): mean regret against the exact optimum of the true model (points), informative cells."""
    inf = informative(d)
    inf = inf[inf["U_opt_T"].notna()] if "U_opt_T" in inf.columns else inf.iloc[0:0]
    if len(inf) == 0:
        return {}
    return {m: float(100 * (inf["U_opt_T"] - inf["U_" + m]).mean()) for m in METHODS}


def claims20(df: pd.DataFrame, sizes=SIZES, in_support=IN_SUPPORT, out=OUT) -> dict:
    res = {}
    for c in (CONTROL,) + tuple(in_support) + tuple(out):
        for n in sizes:
            d = df[(df["cond"] == c) & (df["n"] == n)]
            res[(c, n)] = stats(d) if len(d) else dict(undefined=True)
    q0k = [(CONTROL, n) for n in sizes]
    q1k = [(c, n) for c in in_support for n in sizes]
    q2k = [(c, n) for c in out for n in sizes]
    q2_defined = [k for k in q2k if not res[k]["undefined"]]
    if any(res[k]["undefined"] for k in q0k + q1k) or len(q2_defined) < Q2_DEFINED_MIN:
        label, needed = "NO OUTCOME", "NO OUTCOME"
    else:
        q0 = sum(bool(res[k]["q0"]) for k in q0k)
        q1 = sum(bool(res[k]["q1"]) for k in q1k)
        q2 = sum(bool(res[k]["q2"]) for k in q2_defined)
        if q1 >= Q1_MIN_COUNT and q0 == len(q0k) and q2 == len(q2_defined):
            label = "ROBUST-LEARNER"
        elif q2 >= Q2_USEFUL_MIN and q0 == len(q0k):
            label = "USEFUL"
        elif q2 >= 1:
            label = "LIMITED"
        else:
            label = "NO GAIN"
        needed = "LEARNING-NEEDED" if q1 >= Q1_MIN_COUNT else "CLAMP-ROLLOUT-ENOUGH"
    return dict(statements=res, label=label, needed_class=needed, n_q0=len(q0k), n_q1=len(q1k), n_q2=len(q2k), q2_defined=len(q2_defined))
