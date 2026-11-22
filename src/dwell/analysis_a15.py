"""
Analysis logic of amendment A15 (plan A15.5, A15.9). Reads per-instance rows only.
Unit of inference: the training seed. For a variant V and size n the seed-level statistic is the mean survival probability of the
model of seed k over all instances of the informative cells of size n. A cell is informative if the mean U of RU_CEP lies in [0.10, 0.99].
Contrast Delta_k = (seed-level value of A) - (value of B): B is a planner (RU_CEP or CEPS, one number) or another variant paired by seed k.
Test: exact one-sided Wilcoxon signed-rank (H1: Delta > 0) over the seeds; Holm over the 8 primary contrasts, alpha = 0.05.
Material: mean Delta >= 0.01 (one point of survival probability).
"""
import itertools
import math

import numpy as np
import pandas as pd

ALPHA, MARGIN, INFO_LO, INFO_HI, TIME_RATIO = 0.05, 0.01, 0.10, 0.99, 10.0
VARIANTS = ("IL_G", "IL_EV", "IL_G_noacc", "IL_A", "IL_AD")
CONTRASTS = [("IL_A", "CEPS", 5), ("IL_AD", "CEPS", 7), ("IL_AD", "CEPS", 9), ("IL_AD", "RU_CEP", 7), ("IL_AD", "RU_CEP", 9),
             ("IL_A", "IL_G", 5), ("IL_G", "IL_EV", 5), ("IL_G", "IL_G_noacc", 5)]


def informative(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the informative cells (mean U of RU_CEP in [0.10, 0.99]) of one size."""
    keep = []
    for _, g in df.groupby(["rho", "s", "a", "lam"]):
        if INFO_LO <= g["U_RU_CEP"].mean() <= INFO_HI:
            keep.append(g)
    return pd.concat(keep) if keep else df.iloc[0:0]


def n_seeds(df) -> int:
    return len({c.split("_s")[-1] for c in df.columns if c.startswith("U_IL_G_s")})


def seed_values(df: pd.DataFrame, variant: str) -> np.ndarray:
    """Seed-level pooled mean U of `variant` (one value per training seed)."""
    return np.array([df[f"U_{variant}_s{k}"].mean() for k in range(n_seeds(df))])


def wilcoxon_greater(d) -> float:
    """Exact one-sided p of the signed-rank test (H1: median > 0). Zeros are dropped, ties get average ranks."""
    d = np.asarray([x for x in d if abs(x) > 1e-12], dtype=float)
    m = len(d)
    if m == 0:
        return 1.0
    order = np.argsort(np.abs(d), kind="mergesort")
    ranks = np.empty(m)
    srt = np.abs(d)[order]
    i = 0
    while i < m:
        j = i
        while j + 1 < m and abs(srt[j + 1] - srt[i]) < 1e-12:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    w_obs = ranks[d > 0].sum()
    cnt = 0
    for signs in itertools.product((0, 1), repeat=m):
        if (np.array(signs) * ranks).sum() >= w_obs - 1e-9:
            cnt += 1
    return cnt / 2 ** m


def holm(p, alpha=ALPHA):
    """Holm step-down: returns a list of booleans (reject) in the input order."""
    order = np.argsort(p)
    rej = [False] * len(p)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (len(p) - rank):
            rej[i] = True
        else:
            break
    return rej


def contrasts(dfs: dict) -> list:
    """dfs: size n -> DataFrame of the informative rows. One dict per primary contrast."""
    out = []
    for a, b, n in CONTRASTS:
        d = dfs.get(n)
        if d is None or len(d) == 0:
            out.append(dict(a=a, b=b, n=n, mean_delta=float("nan"), p=1.0, undefined=True)); continue
        va = seed_values(d, a)
        vb = float(d["U_" + b].mean()) if b in ("RU_CEP", "CEPS") else seed_values(d, b)
        delta = va - vb
        out.append(dict(a=a, b=b, n=n, mean_delta=float(delta.mean()), min_delta=float(delta.min()), max_delta=float(delta.max()),
                        p=wilcoxon_greater(delta), undefined=False))
    rej = holm([c["p"] for c in out])
    for c, r in zip(out, rej):
        c["holm_reject"] = bool(r and not c["undefined"])
        c["material"] = bool(c["mean_delta"] >= MARGIN) if not c["undefined"] else False
        c["win"] = c["holm_reject"] and c["material"]
    return out


def decide(cs: list, time_ratio) -> dict:
    """First-match decision table (A15.9). cs: output of contrasts(); time_ratio: decision time of RU_CEP over IL_AD at n = 9."""
    get = {(c["a"], c["b"], c["n"]): c for c in cs}
    if any(c["undefined"] for c in cs):
        return dict(outcome="NO OUTCOME", reason="a size has no informative cell")
    c7, c9 = get[("IL_AD", "RU_CEP", 7)], get[("IL_AD", "RU_CEP", 9)]
    sec = dict(objective_matters=get[("IL_G", "IL_EV", 5)]["win"], accumulated_value_needed=get[("IL_G", "IL_G_noacc", 5)]["win"],
               planner_feature_helps=get[("IL_A", "IL_G", 5)]["win"],
               improves_cheap_planner=all(get[k]["win"] for k in (("IL_A", "CEPS", 5), ("IL_AD", "CEPS", 7), ("IL_AD", "CEPS", 9))))
    if c7["win"] and c9["win"]:
        return dict(outcome="LEARN-BEATS", **sec)
    if c7["mean_delta"] >= -MARGIN and c9["mean_delta"] >= -MARGIN:
        if time_ratio is None:
            return dict(outcome="NO OUTCOME", reason="decision times not measured", **sec)
        if time_ratio >= TIME_RATIO:
            return dict(outcome="LEARN-MATCHES", time_ratio=time_ratio, **sec)
    return dict(outcome="LEARN-BEHIND", **sec)
