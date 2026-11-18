"""
Analysis logic of amendment A11 (plan A11.5, A11.9). Reads per-instance rows only; nothing is recomputed from memory.

regret_m(cell) = 1 - sum_i V_m,i / sum_i V_dp,i; best non-DP method = smallest regret among the 22 (recomputed in every
bootstrap resample); one-sided 95% lower bound = 5th percentile of the bootstrap distribution (B = 2000, seed 20261223,
one generator per cell so the result does not depend on the cell order). A cell is flagged if the bound is >= 0.05 and
point-flagged if the point estimate is >= 0.05. A region = two cells that differ in exactly one factor by one position
of the grid order (eps order: 0.5, 0.8, inf), both outside the corner (s >= 2 and rho <= 1.5).
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .rules1 import METHODS, RULES

THRESH = 0.05
B_BOOT = 2000
SEED_CELL = 20261223
SEED_LM = 20261224
AXES = ("rho", "s", "a", "gamma", "eps")


def outside_corner(rho: float, s: float) -> bool:
    return s >= 2 and rho <= 1.5


def eps_sort_key(e) -> float:
    return float("inf") if str(e) == "inf" else float(e)


def grid_values(cells: Sequence[tuple]):
    """Ordered grid values per axis from the cells present."""
    vals = []
    for k, ax in enumerate(AXES):
        u = sorted({c[k] for c in cells}, key=(eps_sort_key if ax == "eps" else float))
        vals.append(u)
    return vals


def find_regions(flags: Dict[tuple, bool], cells: Sequence[tuple] = None) -> List[Tuple[tuple, tuple]]:
    """Pairs of face-adjacent flagged cells; both cells must be outside the corner."""
    cells = list(flags) if cells is None else list(cells)
    grid = grid_values(cells)
    pos = [{v: i for i, v in enumerate(g)} for g in grid]
    out = []
    for c in cells:
        if not flags.get(c, False) or not outside_corner(c[0], c[1]):
            continue
        for ax in range(len(AXES)):
            i = pos[ax][c[ax]]
            if i + 1 < len(grid[ax]):
                nb = list(c); nb[ax] = grid[ax][i + 1]; nb = tuple(nb)
                if nb in flags and flags[nb] and outside_corner(nb[0], nb[1]):
                    out.append((c, nb))
    return out


def decide(flagged: Dict[tuple, bool], point_flagged: Dict[tuple, bool]) -> dict:
    """Outcome of the pre-written decision table (A11.9)."""
    reg = find_regions(flagged)
    if reg:
        budget = any(str(c[4]) != "inf" for pair in reg for c in pair)
        return dict(outcome="GO", kind="budget-driven" if budget else "ordering-only", regions=reg)
    reg_pt = find_regions(point_flagged)
    if reg_pt:
        return dict(outcome="MARGINAL", kind="", regions=reg_pt)
    return dict(outcome="NO-GO", kind="", regions=[])


def method_matrix(g: pd.DataFrame, methods=METHODS) -> np.ndarray:
    return np.stack([g["V_" + m].to_numpy() for m in methods])


def best_regret(V: np.ndarray, dp: np.ndarray, idx: np.ndarray = None):
    if idx is None:
        return (1.0 - V.sum(1) / dp.sum()).min()
    return (1.0 - V[:, idx].sum(2) / dp[idx].sum(1)).min(0)


def cell_summary(g: pd.DataFrame, cell_no: int) -> dict:
    dp = g["V_dp"].to_numpy()
    if dp.sum() <= 0:
        return dict(undefined=True)
    V = method_matrix(g)
    reg = 1.0 - V.sum(1) / dp.sum()
    k = int(np.argmin(reg))
    rng = np.random.default_rng([SEED_CELL, cell_no])
    idx = rng.integers(0, len(g), size=(B_BOOT, len(g)))
    boot = best_regret(V, dp, idx)
    lo, hi = np.percentile(boot, [5, 97.5])
    out = dict(undefined=False, n=len(g), best_method=METHODS[k], regret_best=float(reg[k]), lo5=float(lo), hi975=float(hi),
               flagged=bool(lo >= THRESH), point_flagged=bool(reg[k] >= THRESH),
               rho_hat=g["rho_hat"].mean(), rho_boost=g["rho_boost"].mean(), V_dp=dp.mean(),
               best_is_rollout=METHODS[k].startswith("ROLL"))
    out.update({f"regret_{m}": float(reg[i]) for i, m in enumerate(METHODS)})
    st = g[g["V_noidle"].notna()] if "V_noidle" in g else g.iloc[0:0]
    if len(st):
        out["value_of_idle"] = float(1.0 - st["V_noidle"].sum() / st["V_dp"].sum())
        out["p_wait_dp"] = float(st["p_wait_dp"].mean()); out["p_pre_dp"] = float(st["p_pre_dp"].mean())
        out["n_stats"] = len(st)
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["rho", "s", "a", "gamma", "eps"]
    for no, (key, g) in enumerate(df.groupby(keys, sort=True)):
        g = g.sort_values("idx")
        r = dict(zip(keys, key)); r.update(cell_summary(g, no)); rows.append(r)
    return pd.DataFrame(rows)


def lm_differences(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over cells of regret(L) - regret(M) per rule, paired instance bootstrap (same resample for M and L)."""
    rng = np.random.default_rng(SEED_LM)
    keys = ["rho", "s", "a", "gamma", "eps"]
    groups = [g.sort_values("idx") for _, g in df.groupby(keys, sort=True) if g["V_dp"].sum() > 0]
    point = {r: 0.0 for r in RULES}
    boot = {r: np.zeros(B_BOOT) for r in RULES}
    for g in groups:
        dp = g["V_dp"].to_numpy()
        idx = rng.integers(0, len(g), size=(B_BOOT, len(g)))
        for r in RULES:
            vm, vl = g[f"V_{r}_M"].to_numpy(), g[f"V_{r}_L"].to_numpy()
            point[r] += ((1 - vl.sum() / dp.sum()) - (1 - vm.sum() / dp.sum())) / len(groups)
            bm = 1 - vm[idx].sum(1) / dp[idx].sum(1); bl = 1 - vl[idx].sum(1) / dp[idx].sum(1)
            boot[r] += (bl - bm) / len(groups)
    return pd.DataFrame([dict(rule=r, mean_regret_L_minus_M=point[r], lo=np.percentile(boot[r], 2.5),
                              hi=np.percentile(boot[r], 97.5)) for r in RULES])
