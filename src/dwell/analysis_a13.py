"""
Analysis logic of amendment A13 (plan A13.5, A13.9). Reads per-instance rows only.

regret_m(cell) = 1 - sum_i V_m,i / sum_i V_dp,i; best non-DP method = smallest regret among the 11 (recomputed in every
bootstrap resample); one-sided 95% lower bound = 5th percentile of the bootstrap distribution (B = 2000, seed 20261242,
one generator per cell so the result does not depend on the cell order). A cell is flagged if the bound is >= 0.05 and
point-flagged if the point estimate is >= 0.05. A region = two cells that differ in exactly one factor by one position of
the grid order (n: 4, 5; rho: 1.1, 1.5; s_bar: 2, 4; a: 0.5, 0.9; geometry: SPREAD, CLUSTER), both outside the corner.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .m0g import GEOMETRIES, METHODS

THRESH = 0.05
B_BOOT = 2000
SEED_CELL = 20261242
AXES = ("n", "rho", "s", "a", "geom")


def outside_corner(rho, s):
    return s >= 2 and rho <= 1.5


def axis_key(ax):
    return (lambda g: GEOMETRIES.index(g)) if ax == "geom" else float


def grid_values(cells: Sequence[tuple]):
    return [sorted({c[k] for c in cells}, key=axis_key(ax)) for k, ax in enumerate(AXES)]


def find_regions(flags: Dict[tuple, bool], cells: Sequence[tuple] = None) -> List[Tuple[tuple, tuple]]:
    """Pairs of face-adjacent flagged cells; both cells must be outside the corner (cell = (n, rho, s, a, geom))."""
    cells = list(flags) if cells is None else list(cells)
    grid = grid_values(cells)
    pos = [{v: i for i, v in enumerate(g)} for g in grid]
    out = []
    for c in cells:
        if not flags.get(c, False) or not outside_corner(c[1], c[2]):
            continue
        for ax in range(len(AXES)):
            i = pos[ax][c[ax]]
            if i + 1 < len(grid[ax]):
                nb = list(c); nb[ax] = grid[ax][i + 1]; nb = tuple(nb)
                if flags.get(nb, False) and outside_corner(nb[1], nb[2]):
                    out.append((c, nb))
    return out


def decide(flagged: Dict[tuple, bool], point_flagged: Dict[tuple, bool]) -> dict:
    """Outcome of the pre-written decision table (A13.9)."""
    reg = find_regions(flagged)
    if reg:
        return dict(outcome="GO", regions=reg)
    reg_pt = find_regions(point_flagged)
    if reg_pt:
        return dict(outcome="MARGINAL", regions=reg_pt)
    return dict(outcome="NO-GO", regions=[])


def method_matrix(g: pd.DataFrame) -> np.ndarray:
    return np.stack([g["V_" + m].to_numpy() for m in METHODS])


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
    lo, hi = np.percentile(best_regret(V, dp, idx), [5, 97.5])
    out = dict(undefined=False, count=len(g), best_method=METHODS[k], regret_best=float(reg[k]), lo5=float(lo),
               hi975=float(hi), flagged=bool(lo >= THRESH), point_flagged=bool(reg[k] >= THRESH),
               best_is_rollout=METHODS[k].startswith("ROLL"), V_dp=float(dp.mean()))
    for col in ("first_differs", "cv_slew", "mean_slew", "n_closed"):
        if col in g:
            out[col] = float(g[col].mean())
    out.update({f"regret_{m}": float(reg[i]) for i, m in enumerate(METHODS)})
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    keys = list(AXES)
    rows = []
    order = sorted(df.groupby(keys).groups, key=lambda t: tuple(axis_key(ax)(v) for ax, v in zip(AXES, t)))
    for no, key in enumerate(order):
        g = df[(df[keys] == pd.Series(key, index=keys)).all(axis=1)].sort_values("idx")
        r = dict(zip(keys, key)); r.update(cell_summary(g, no)); rows.append(r)
    return pd.DataFrame(rows)
