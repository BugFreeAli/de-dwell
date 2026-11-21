"""
Analysis logic of amendment A14 (plan A14.5, A14.9). Reads per-instance rows only.

U_m,i = exact P(killed value >= need) of method m on instance i; U_opt,i = exact optimum of the goal objective.
Cell regret      rel_m = 1 - sum_i U_m,i / sum_i U_opt,i          absolute gap   abs_m = (sum_i U_opt,i - sum_i U_m,i) / N
A cell is INFORMATIVE if 0.10 <= mean(U_opt) <= 0.99 (otherwise a relative loss of a probability near 0 or 1 is not material).
Best method = smallest rel (equivalently smallest abs) in the set, re-selected in every bootstrap resample.
One-sided 95% lower bound = 5th percentile of B = 2000 resamples (seed 20261252, one generator per cell).
A cell is flagged for a set if it is informative and lo(rel) >= 0.05 and lo(abs) >= 0.02; point-flagged with the point estimates.
Sets: ALL = the 17 methods; EV = the 11 EV rules and DPEV (the information- and objective-matched EV-based methods).
Region = two flagged cells that differ in exactly one factor by one position of the grid order, both outside the corner.
Decision: GO (region for ALL) > MISMATCH (region for EV only) > MARGINAL (region of point-flagged cells for ALL) > NO-GO.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .m0u import EV_BASED, METHODS

REL_THRESH = 0.05
ABS_THRESH = 0.02
INFO_LO, INFO_HI = 0.10, 0.99
B_BOOT = 2000
SEED_CELL = 20261252
AXES = ("n", "rho", "s", "a", "lam")


def outside_corner(rho, s):
    return s >= 2 and rho <= 1.5


def grid_values(cells: Sequence[tuple]):
    return [sorted({c[k] for c in cells}, key=float) for k in range(len(AXES))]


def find_regions(flags: Dict[tuple, bool], cells: Sequence[tuple] = None) -> List[Tuple[tuple, tuple]]:
    """Pairs of face-adjacent flagged cells; cell = (n, rho, s, a, lam); both must be outside the corner."""
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


def decide(flag_all: Dict[tuple, bool], flag_ev: Dict[tuple, bool], point_all: Dict[tuple, bool]) -> dict:
    """Outcome of the pre-written decision table (A14.9)."""
    reg = find_regions(flag_all)
    if reg:
        return dict(outcome="GO", regions=reg)
    reg_ev = find_regions(flag_ev)
    if reg_ev:
        return dict(outcome="MISMATCH", regions=reg_ev)
    reg_pt = find_regions(point_all)
    if reg_pt:
        return dict(outcome="MARGINAL", regions=reg_pt)
    return dict(outcome="NO-GO", regions=[])


def method_matrix(g: pd.DataFrame, methods=METHODS) -> np.ndarray:
    return np.stack([g["U_" + m].to_numpy() for m in methods])


def best_over(U: np.ndarray, opt: np.ndarray, idx: np.ndarray = None):
    """(relative regret, absolute gap) of the best method; with idx (B x N) one value per resample."""
    if idx is None:
        s = U.sum(1)
        k = int(np.argmax(s))
        return 1.0 - s[k] / opt.sum(), (opt.sum() - s[k]) / len(opt)
    s = U[:, idx].sum(2)                                  # (methods, B)
    so = opt[idx].sum(1)                                  # (B,)
    sb = s.max(0)
    with np.errstate(divide="ignore", invalid="ignore"):  # a resample with sum(U_opt) = 0 gives nan (only in cells with U_opt near 0)
        return 1.0 - sb / so, (so - sb) / idx.shape[1]


def cell_summary(g: pd.DataFrame, cell_no: int) -> dict:
    opt = g["U_opt"].to_numpy()
    mean_opt = float(opt.mean())
    if opt.sum() <= 0:
        return dict(undefined=True, informative=False, U_opt=mean_opt)
    U = method_matrix(g)
    reg = 1.0 - U.sum(1) / opt.sum()
    k = int(np.argmin(reg))
    ev_ix = [METHODS.index(m) for m in EV_BASED]
    kev = ev_ix[int(np.argmin(reg[ev_ix]))]
    rng = np.random.default_rng([SEED_CELL, cell_no])
    idx = rng.integers(0, len(g), size=(B_BOOT, len(g)))
    rel_all, abs_all = best_over(U, opt, idx)
    rel_ev, abs_ev = best_over(U[ev_ix], opt, idx)
    info = INFO_LO <= mean_opt <= INFO_HI
    n = len(g)
    out = dict(undefined=False, count=n, U_opt=mean_opt, informative=bool(info),
               best_method=METHODS[k], regret_best=float(reg[k]), abs_best=float((opt.sum() - U[k].sum()) / n),
               lo5=float(np.percentile(rel_all, 5)), lo5_abs=float(np.percentile(abs_all, 5)),
               best_ev_method=METHODS[kev], regret_best_ev=float(reg[kev]),
               abs_best_ev=float((opt.sum() - U[kev].sum()) / n),
               lo5_ev=float(np.percentile(rel_ev, 5)), lo5_ev_abs=float(np.percentile(abs_ev, 5)))
    out["flagged_all"] = bool(info and out["lo5"] >= REL_THRESH and out["lo5_abs"] >= ABS_THRESH)
    out["flagged_ev"] = bool(info and out["lo5_ev"] >= REL_THRESH and out["lo5_ev_abs"] >= ABS_THRESH)
    out["point_all"] = bool(info and out["regret_best"] >= REL_THRESH and out["abs_best"] >= ABS_THRESH)
    out["point_ev"] = bool(info and out["regret_best_ev"] >= REL_THRESH and out["abs_best_ev"] >= ABS_THRESH)
    out["best_is_aware"] = not (METHODS[k] in EV_BASED)
    for col in ("first_differs", "n_states"):
        if col in g:
            out[col] = float(g[col].mean())
    out.update({f"regret_{m}": float(reg[i]) for i, m in enumerate(METHODS)})
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    keys = list(AXES)
    rows = []
    order = sorted(df.groupby(keys).groups, key=lambda t: tuple(float(v) for v in t))
    for no, key in enumerate(order):
        g = df[(df[keys] == pd.Series(key, index=keys)).all(axis=1)].sort_values("idx")
        r = dict(zip(keys, key)); r.update(cell_summary(g, no)); rows.append(r)
    return pd.DataFrame(rows)
