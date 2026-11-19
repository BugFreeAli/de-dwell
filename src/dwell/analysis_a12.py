"""
A12 analysis (plan A12.4): regret matrix, exploitability, minimax mixture (own simplex, no scipy needed),
split sample, stratified bootstrap and the mechanical decision rule.

Conventions
  * rows = defenders, columns = attacker strategies; R[d, k] = 1 - sum V(d, k) / sum V*(k) pooled over the chosen instances.
  * E(d) = max over the strategies of a budget set of R[d, k] (exploitability).
  * The mixture x is solved on the EVEN-index instances and evaluated on the ODD-index instances; every decision
    quantity is computed on the odd half.
  * Bootstrap: instances are resampled within every (cell, half) stratum; every selection and the linear program
    are recomputed in each resample.
"""
import numpy as np

from .hard import ADAPTIVE, DEFENDERS, NONADAPTIVE, PARTIAL, STRAT_NAMES, budget_set

X_MIN, R_MIN, R_P5, S_MIN = 0.08, 0.30, 0.20, 0.01      # thresholds of plan A12.4
BOOT_B = 2000
BOOT_SEED = 20261230
CELL_COLS = ["rho", "s", "a"]


def minimax_mixture(R):
    """
    Defender mixture x minimising max_k (x^T R)_k (R = loss matrix, rows defenders, columns attacker strategies).
    Standard reduction: A = R + c > 0, maximise sum(y) subject to A^T y <= 1, y >= 0, then v = 1 / sum(y), x = v y;
    the attacker's optimal mixture is read from the dual values. Simplex with Bland's rule on the slack basis.
    Returns (x, w, value) with value = min_x max_k (x^T R)_k.
    """
    R = np.asarray(R, dtype=float)
    m, k = R.shape
    shift = 1.0 - R.min()
    G = (R + shift).T.copy()                  # k x m, all entries >= 1
    T = np.zeros((k + 1, m + k + 1))
    T[:k, :m] = G
    T[:k, m:m + k] = np.eye(k)
    T[:k, -1] = 1.0
    T[k, :m] = -1.0
    basis = list(range(m, m + k))
    for _ in range(10000):
        neg = np.where(T[k, :-1] < -1e-12)[0]
        if len(neg) == 0:
            break
        e = int(neg[0])
        rows = [i for i in range(k) if T[i, e] > 1e-12]
        if not rows:
            raise RuntimeError("linear program unbounded (cannot happen for a positive matrix)")
        best = min(T[i, -1] / T[i, e] for i in rows)
        r = min((i for i in rows if T[i, -1] / T[i, e] <= best + 1e-12), key=lambda i: basis[i])
        T[r] /= T[r, e]
        for i in range(k + 1):
            if i != r:
                T[i] -= T[i, e] * T[r]
        basis[r] = e
    else:
        raise RuntimeError("simplex did not terminate")
    y = np.zeros(m)
    for i, b in enumerate(basis):
        if b < m:
            y[b] = T[i, -1]
    v = 1.0 / y.sum()
    z = T[k, m:m + k]
    return y * v, z / z.sum(), v - shift


def build_arrays(df):
    """(V[N, D, K], Vstar[N, K], cell id [N], idx [N]) from a run dataframe."""
    V = np.stack([[df[f"V_{d}_{k}"].to_numpy(float) for k in STRAT_NAMES] for d in DEFENDERS]).transpose(2, 0, 1)
    Vs = np.stack([df[f"Vstar_{k}"].to_numpy(float) for k in STRAT_NAMES], axis=1)
    cell = df.groupby(CELL_COLS).ngroup().to_numpy()
    return V, Vs, cell, df["idx"].to_numpy(int)


def regret_matrix(V, Vs, rows):
    """R[d, k] = 1 - sum V / sum V*(k) over the given instance rows (repeats allowed)."""
    return 1.0 - V[rows].sum(0) / Vs[rows].sum(0)


def strata(cell, idx):
    """Row indices per (cell, half): fit = even idx, evaluate = odd idx."""
    fit, ev = [], []
    for c in np.unique(cell):
        rows = np.where(cell == c)[0]
        fit.append(rows[idx[rows] % 2 == 0])
        ev.append(rows[idx[rows] % 2 == 1])
    return fit, ev


def _draw(rng, groups):
    return np.concatenate([g[rng.integers(0, len(g), size=len(g))] for g in groups])


def stats_from(Rfit, Rev, level="B40"):
    """All decision quantities from a fit-half matrix and an evaluation-half matrix."""
    cols = [STRAT_NAMES.index(n) for n in budget_set(level)]
    E = Rev[:, cols].max(axis=1)
    x, w, vfit = minimax_mixture(Rfit[:, cols])
    mix_k = x @ Rev[:, cols]
    E_mix = float(mix_k.max())
    nonad = [DEFENDERS.index(d) for d in NONADAPTIVE]
    adap = [DEFENDERS.index(d) for d in ADAPTIVE]
    E_nom = float(E[nonad].min())
    E_def = float(min(E[adap].min(), E_mix))
    out = dict(E=E, E_mix=E_mix, E_nom=E_nom, E_def=E_def, reduction=1.0 - E_def / E_nom, x=x, w=w, v_fit=vfit)
    shared_adapt = [DEFENDERS.index(d) for d in ("QMDP_B", "QMDP_H", "BA_H")]
    for p in PARTIAL:
        j = STRAT_NAMES.index(p)
        out["S_" + p] = float(Rev[shared_adapt, j].min() - Rev[nonad, j].min())
    return out


def analyse(df, level="B40", B=BOOT_B, seed=BOOT_SEED):
    """Point estimates and bootstrap 5th percentiles of E_nom, reduction and the S differences."""
    V, Vs, cell, idx = build_arrays(df)
    fit, ev = strata(cell, idx)
    point = stats_from(regret_matrix(V, Vs, np.concatenate(fit)), regret_matrix(V, Vs, np.concatenate(ev)), level)
    rng = np.random.default_rng(seed)
    keys = ["E_nom", "E_def", "reduction"] + ["S_" + p for p in PARTIAL]
    draws = {k: np.empty(B) for k in keys}
    for b in range(B):
        rf, re = _draw(rng, fit), _draw(rng, ev)
        st = stats_from(regret_matrix(V, Vs, rf), regret_matrix(V, Vs, re), level)
        for k in keys:
            draws[k][b] = st[k]
    p5 = {k: float(np.percentile(v, 5)) for k, v in draws.items()}
    return dict(point=point, p5=p5, level=level)


def flags(res):
    """X, R, S of plan A12.4 from an analyse() result at B40."""
    pt, p5 = res["point"], res["p5"]
    X = pt["E_nom"] >= X_MIN and p5["E_nom"] >= X_MIN
    R = pt["reduction"] >= R_MIN and p5["reduction"] >= R_P5
    S = any(pt["S_" + p] >= S_MIN and p5["S_" + p] > 0.0 for p in PARTIAL)
    return dict(X=bool(X), R=bool(R), S=bool(S))


def decide(main, noise2=None):
    """Outcome table of A12.4, in order. noise2 = flags of the noise +-2 run or None if not run yet."""
    if not main["X"]:
        return "STOP"
    if not main["R"]:
        return "EXPOSED"
    if noise2 is None:
        return "PENDING (a12_noise2 not analysed)"
    if not (noise2["X"] and noise2["R"]):
        return "REGIME-LIMITED"
    if not main["S"]:
        return "ADAPTATION-SUFFICES"
    return "STRONG-ADVERSARIAL"
