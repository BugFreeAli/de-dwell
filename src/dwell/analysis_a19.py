"""
Analysis logic of amendment A19 (plan A19.4, A19.5): learning the dwell scale during the episode. Reads per-instance rows only (src/dwell/a19.py).
Informative cell (scale, n, rho, s, a, lam): mean U_CEPS_T (informed survival-aware planner) in [0.10, 0.99]. Instance-level bootstrap stratified by
cell (B = 2000, seed 20261303), one-sided 95% bounds.
Statements (true scales 0.8, 1.2, 1.4; sizes 5, 7):
  P1(scale, n): lower bound of mean(U_ADAPT - U_NOM) >= 0.03                       (learning helps against the nominal planner; 6 statements)
  P2(n):        at scale 1.0 the upper bound of mean(U_NOM - U_ADAPT) <= 0.03       (learning costs little when the nominal model is right; 2 statements)
  S1(scale, n): lower bound of mean(U_ADAPT - U_CLAMP) >= 0.01                      (learning adds to not abandoning; 6 statements)
  recovery R = mean(U_ADAPT - U_NOM) / mean(U_RU_CEP_T - U_NOM) (share of the gap to the informed rollout planner that ADAPT closes), bootstrap with the
  same resampled instances in numerator and denominator; per statement and pooled over the six statements.
Label, first match: NO OUTCOME (a statement has no informative cell); WORKS (P1 in 6 of 6, P2 in 2 of 2, pooled R lower bound >= 0.5); HELPS (P1 in at
least 4 of 6 and P2 in 2 of 2); LIMITED (P1 in at least 1); NO GAIN. Second label: LEARNING-ADDS if S1 holds in at least 5 of 6, else CLAMP-COMPARABLE.
Pilot rule (n = 5 only, descriptive): GO if the pooled point recovery over the three scales >= 0.20 and mean(U_NOM - U_ADAPT) at scale 1.0 <= 0.05, else STOP.
"""
import numpy as np
import pandas as pd

from .analysis_a16 import B_BOOT, INFO_HI, INFO_LO, boot_mean

CELL = ["n", "scale", "rho", "s", "a", "lam"]
REF = "U_CEPS_T"
P1_MIN, P2_MAX, S1_MIN = 0.03, 0.03, 0.01
R_HIGH, R_SOME, R_POOL_MIN = 0.5, 0.2, 0.5
SCALES, SIZES, CONTROL = (0.8, 1.2, 1.4), (5, 7), 1.0
HELPS_MIN, CLAMP_ADDS_MIN = 4, 5
PILOT_R_MIN, PILOT_LOSS_MAX = 0.20, 0.05
GAP_MIN = 0.01
SEED19 = 20261303
METHODS = ("NOM", "CLAMP", "ADAPT", "ADAPT_V", "CEPS_T", "RU_CEP_T")


def informative(d: pd.DataFrame) -> pd.DataFrame:
    keep = [g for _, g in d.groupby(CELL) if INFO_LO <= g[REF].mean() <= INFO_HI]
    return pd.concat(keep) if keep else d.iloc[0:0]


def strata(d, a, b):
    return [(g[a] - g[b]).to_numpy() for _, g in d.groupby(CELL)]


def boot_ratio(num, den, B=B_BOOT, seed=SEED19):
    """Ratio of pooled means with the same resampled instances in numerator and denominator. Returns (point, 5th, 95th percentile)."""
    rng = np.random.default_rng(seed)
    dsum = sum(x.sum() for x in den)
    point = float(sum(x.sum() for x in num) / dsum) if abs(dsum) > 1e-12 else float("nan")
    out = np.full(B, np.nan)
    for b in range(B):
        ns = ds = 0.0
        for x, y in zip(num, den):
            idx = rng.integers(0, len(x), len(x))
            ns += x[idx].sum(); ds += y[idx].sum()
        if abs(ds) > 1e-12:
            out[b] = ns / ds
    ok = out[~np.isnan(out)]
    if len(ok) == 0:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(ok, 5)), float(np.percentile(ok, 95))


def recovery_class(point, lo, gap):
    if gap < GAP_MIN or np.isnan(lo):
        return "UNDEFINED"
    return "HIGH" if lo >= R_HIGH else ("SOME" if lo >= R_SOME else "LOW")


def stats(d: pd.DataFrame) -> dict:
    """d: the rows of one scale and size. Statistics over its informative cells."""
    inf = informative(d)
    if len(inf) == 0:
        return dict(undefined=True)
    g, g_lo, g_hi = boot_mean(strata(inf, "U_ADAPT", "U_NOM"), seed=SEED19)
    l, l_lo, l_hi = boot_mean(strata(inf, "U_NOM", "U_ADAPT"), seed=SEED19)
    c, c_lo, c_hi = boot_mean(strata(inf, "U_ADAPT", "U_CLAMP"), seed=SEED19)
    gap, gap_lo, gap_hi = boot_mean(strata(inf, "U_RU_CEP_T", "U_NOM"), seed=SEED19)
    r, r_lo, r_hi = boot_ratio(strata(inf, "U_ADAPT", "U_NOM"), strata(inf, "U_RU_CEP_T", "U_NOM"))
    return dict(undefined=False, instances=len(inf), cells=len(inf.groupby(CELL)),
                gain=g, gain_lo=g_lo, gain_hi=g_hi, p1=bool(g_lo >= P1_MIN),
                loss=l, loss_lo=l_lo, loss_hi=l_hi, p2=bool(l_hi <= P2_MAX),
                vs_clamp=c, vs_clamp_lo=c_lo, vs_clamp_hi=c_hi, s1=bool(c_lo >= S1_MIN),
                gap=gap, recovery=r, recovery_lo=r_lo, recovery_hi=r_hi, recovery_class=recovery_class(r, r_lo, gap))


def pooled_recovery(df: pd.DataFrame, sizes=SIZES, scales=SCALES) -> dict:
    parts = []
    for sc in scales:
        for n in sizes:
            parts.append(informative(df[(df["scale"] == sc) & (df["n"] == n)]))
    inf = pd.concat([p for p in parts if len(p)]) if any(len(p) for p in parts) else df.iloc[0:0]
    if len(inf) == 0:
        return dict(undefined=True)
    num, den = strata(inf, "U_ADAPT", "U_NOM"), strata(inf, "U_RU_CEP_T", "U_NOM")
    r, lo, hi = boot_ratio(num, den)
    gap = float(np.mean(np.concatenate(den)))
    return dict(undefined=False, recovery=r, recovery_lo=lo, recovery_hi=hi, gap=gap, recovery_class=recovery_class(r, lo, gap), instances=len(inf))


def regret_table(d: pd.DataFrame) -> dict:
    """Exact level (rows with U_opt_T): mean regret against the exact optimum of the true model (points), informative cells."""
    inf = informative(d)
    inf = inf[inf["U_opt_T"].notna()] if "U_opt_T" in inf.columns else inf.iloc[0:0]
    if len(inf) == 0:
        return {}
    return {m: float(100 * (inf["U_opt_T"] - inf["U_" + m]).mean()) for m in METHODS}


def claims19(df: pd.DataFrame, sizes=SIZES, scales=SCALES) -> dict:
    res = {}
    for sc in (CONTROL,) + tuple(scales):
        for n in sizes:
            d = df[(df["scale"] == sc) & (df["n"] == n)]
            res[(sc, n)] = stats(d) if len(d) else dict(undefined=True)
    p1_keys = [(sc, n) for sc in scales for n in sizes]
    p2_keys = [(CONTROL, n) for n in sizes]
    pool = pooled_recovery(df, sizes, scales)
    if any(res[k]["undefined"] for k in p1_keys + p2_keys) or pool["undefined"]:
        label, clamp = "NO OUTCOME", "NO OUTCOME"
    else:
        p1 = sum(bool(res[k]["p1"]) for k in p1_keys)
        p2 = sum(bool(res[k]["p2"]) for k in p2_keys)
        s1 = sum(bool(res[k]["s1"]) for k in p1_keys)
        if p1 == len(p1_keys) and p2 == len(p2_keys) and pool["recovery_lo"] >= R_POOL_MIN:
            label = "WORKS"
        elif p1 >= HELPS_MIN and p2 == len(p2_keys):
            label = "HELPS"
        elif p1 >= 1:
            label = "LIMITED"
        else:
            label = "NO GAIN"
        clamp = "LEARNING-ADDS" if s1 >= CLAMP_ADDS_MIN else "CLAMP-COMPARABLE"
    return dict(statements=res, pooled=pool, label=label, clamp_class=clamp, n_p1=len(p1_keys), n_p2=len(sizes))


def pilot_rule(df: pd.DataFrame) -> dict:
    """Descriptive go / stop rule on the n = 5 pilot."""
    d5 = df[df["n"] == 5]
    pool = pooled_recovery(d5, (5,), SCALES)
    ctl = informative(d5[d5["scale"] == CONTROL])
    if pool["undefined"] or len(ctl) == 0:
        return dict(decision="NO OUTCOME")
    loss = float((ctl["U_NOM"] - ctl["U_ADAPT"]).mean())
    go = bool(pool["recovery"] >= PILOT_R_MIN and loss <= PILOT_LOSS_MAX)
    return dict(decision="GO" if go else "STOP", recovery=pool["recovery"], loss_at_control=loss)
