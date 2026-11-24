"""
Analysis logic of amendment A16 (plan A16.5, A16.8). Reads per-instance rows only.
Informative cell: the mean survival probability of the reference method lies in [0.10, 0.99]; the reference is RU_CEP, or CEPS where
RU_CEP was not run (n = 11). Intervals: instance-level bootstrap stratified by cell (B = 2000, seed 20261272), one-sided 95%.
P1 (survival-aware planning pays): lower bound of the mean of U_CEPS - U_CEP >= 0.01 at n = 5, 7 and 9.
P2 (the cheap planner is adequate): upper bound of the mean of U_RU_CEP - U_CEPS <= 0.03 at n = 7 and 9.
P3 (regime variable): over the informative instances of n = 5, 7, 9, the mean gain U_CEPS - U_CEP of the instances with
    0.8 <= tau <= 1.1 minus that of the others has a lower bound >= 0.03 (tau = need / expected killed value of CEP).
CONFIRMED = P1, P2 and P3 all hold; PARTIAL = P1 holds at every size but P2 or P3 fails; NOT CONFIRMED = otherwise.
"""
import numpy as np
import pandas as pd

INFO_LO, INFO_HI = 0.10, 0.99
P1_MIN, P2_MAX, P3_MIN = 0.01, 0.03, 0.03
BAND = (0.8, 1.1)
B_BOOT, SEED = 2000, 20261272
CELL = ["group", "n", "rho", "s", "a", "lam"]


def reference(df):
    return "U_RU_CEP" if "U_RU_CEP" in df.columns and df["U_RU_CEP"].notna().all() else "U_CEPS"


def informative(df: pd.DataFrame) -> pd.DataFrame:
    ref = reference(df)
    keep = [g for _, g in df.groupby(CELL) if INFO_LO <= g[ref].mean() <= INFO_HI]
    return pd.concat(keep) if keep else df.iloc[0:0]


def boot_mean(strata, B=B_BOOT, seed=SEED):
    """strata: list of 1-D arrays (one per cell). Returns (mean, 5th percentile, 95th percentile) of the pooled mean."""
    rng = np.random.default_rng(seed)
    tot = sum(len(s) for s in strata)
    point = sum(s.sum() for s in strata) / tot
    out = np.empty(B)
    for b in range(B):
        out[b] = sum(s[rng.integers(0, len(s), len(s))].sum() for s in strata) / tot
    return float(point), float(np.percentile(out, 5)), float(np.percentile(out, 95))


def diff_strata(df, a, b):
    return [(g[a] - g[b]).to_numpy() for _, g in df.groupby(CELL)]


def band_diff(strata, B=B_BOOT, seed=SEED):
    """strata: list of (gain, in_band) array pairs. Returns (in-band mean minus out-of-band mean: point, 5th percentile)."""
    def stat(parts):
        g = np.concatenate([p[0] for p in parts]); m = np.concatenate([p[1] for p in parts])
        if m.sum() == 0 or (~m).sum() == 0:
            return float("nan")
        return g[m].mean() - g[~m].mean()
    point = stat(strata)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        parts = []
        for g, m in strata:
            ix = rng.integers(0, len(g), len(g))
            parts.append((g[ix], m[ix]))
        out.append(stat(parts))
    out = np.asarray(out, dtype=float)
    return float(point), (float(np.nanpercentile(out, 5)) if not np.all(np.isnan(out)) else float("nan"))


def claims(dfs: dict) -> dict:
    """dfs: size n -> DataFrame of the informative main rows (group 'main')."""
    out = {}
    for n in (5, 7, 9):
        d = dfs.get(n)
        if d is None or len(d) == 0:
            out[f"P1_n{n}"] = dict(holds=False, undefined=True)
            continue
        m, lo, hi = boot_mean(diff_strata(d, "U_CEPS", "U_CEP"))
        out[f"P1_n{n}"] = dict(mean=m, lo5=lo, hi95=hi, holds=bool(lo >= P1_MIN), undefined=False)
    for n in (7, 9):
        d = dfs.get(n)
        if d is None or len(d) == 0:
            out[f"P2_n{n}"] = dict(holds=False, undefined=True)
            continue
        m, lo, hi = boot_mean(diff_strata(d, "U_RU_CEP", "U_CEPS"))
        out[f"P2_n{n}"] = dict(mean=m, lo5=lo, hi95=hi, holds=bool(hi <= P2_MAX), undefined=False)
    parts = []
    for n in (5, 7, 9):
        d = dfs.get(n)
        if d is None or len(d) == 0:
            continue
        for _, g in d.groupby(CELL):
            tau = g["tau"].to_numpy()
            parts.append(((g["U_CEPS"] - g["U_CEP"]).to_numpy(), (tau >= BAND[0]) & (tau <= BAND[1])))
    if parts:
        pt, lo = band_diff(parts)
        out["P3"] = dict(diff=pt, lo5=lo, holds=bool(lo >= P3_MIN), undefined=False)
    else:
        out["P3"] = dict(holds=False, undefined=True)
    p1 = all(out[f"P1_n{n}"]["holds"] for n in (5, 7, 9))
    rest = all(out[k]["holds"] for k in ("P2_n7", "P2_n9", "P3"))
    if any(v.get("undefined") for v in out.values()):
        out["outcome"] = "NO OUTCOME"
    elif p1 and rest:
        out["outcome"] = "CONFIRMED"
    elif p1:
        out["outcome"] = "PARTIAL"
    else:
        out["outcome"] = "NOT CONFIRMED"
    return out
