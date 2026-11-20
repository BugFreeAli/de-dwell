"""
A13: M0 with a geometry-dependent slew matrix (plan A13.2, A13.r1).

Threat j has a bearing theta_j in [0, 1]; slewing from i to j costs s_ij = floor(c * |theta_i - theta_j| + 0.5) steps,
then the matrix is replaced by its shortest-path closure so that the triangle inequality holds exactly (A13.r1 item 1).
Everything else is M0: the DP (dwell.dp), the rules (dwell.rules) and evaluate_policy only call Model.slew(cur, j),
so a subclass that overrides that one method is enough.
"""
import itertools
import math
import time

import numpy as np

from .dp import solve
from .instances import make_instance_k
from .m0 import EPS, NO_TARGET, Model, evaluate_policy
from .rules import make_rule

TOL = 1e-9
METHODS = ("FAIR", "EDF", "LLF", "SE", "GMR", "BI", "CEP", "LA1", "ROLL_CEP", "ROLL_GMR", "ROLL_EDF")
BASE_OF = {"ROLL_CEP": "CEP", "ROLL_GMR": "GMR", "ROLL_EDF": "EDF"}
GEOMETRIES = ("SPREAD", "CLUSTER")
C_FACTOR = {"SPREAD": 3.0, "CLUSTER": 2.7273}          # c = factor * s_bar (plan A13.2)
CLUSTER_CENTRES, CLUSTER_HALF_WIDTH = (0.15, 0.85), 0.05


class ModelG(Model):
    """M0 with a slew matrix; the first engagement is free, a different target costs sm[cur][j]."""

    def __init__(self, inst, slew_matrix):
        super().__init__(inst)
        sm = tuple(tuple(int(x) for x in row) for row in slew_matrix)
        assert len(sm) == self.n and all(len(r) == self.n for r in sm), "slew matrix shape"
        assert all(sm[i][i] == 0 for i in range(self.n)) and all(x >= 0 for r in sm for x in r), "slew matrix values"
        self.sm = sm

    def slew(self, cur, j):
        return self.sm[cur][j] if (cur != NO_TARGET and cur != j) else 0


def uniform_matrix(n, s):
    return tuple(tuple(0 if i == j else s for j in range(n)) for i in range(n))


def bearings(master_seed, idx, n, geom):
    """Bearings from their own random stream, independent of make_instance_k (plan A13.2)."""
    g = 0 if geom == "SPREAD" else 1
    rng = np.random.default_rng([master_seed, idx, 13, g])
    if geom == "SPREAD":
        return tuple(float(x) for x in rng.random(n))
    group = rng.random(n) < 0.5
    off = rng.uniform(-CLUSTER_HALF_WIDTH, CLUSTER_HALF_WIDTH, n)
    return tuple(float(CLUSTER_CENTRES[0] if gr else CLUSTER_CENTRES[1]) + float(o) for gr, o in zip(group, off))


def slew_matrix(theta, c):
    """Rounded separation, then shortest-path closure. Returns (matrix, number of entries changed by the closure)."""
    n = len(theta)
    m = [[0 if i == j else int(math.floor(c * abs(theta[i] - theta[j]) + 0.5)) for j in range(n)] for i in range(n)]
    raw = [row[:] for row in m]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                m[i][j] = min(m[i][j], m[i][k] + m[k][j])
    changed = sum(1 for i in range(n) for j in range(n) if m[i][j] != raw[i][j])
    return tuple(tuple(r) for r in m), changed


def a13_instance(cfg, n, rho, s, a, geom, idx, control=False):
    """Instance (as in A6 to A12 for the same arguments), its slew matrix and the stored regime variables."""
    inst, real = make_instance_k(cfg["master_seed"], idx, n, rho, s, a, D=tuple(cfg["D0"]),
                                 value_set=tuple(cfg["value_set"]), u_low=cfg["u_low"])
    theta = bearings(cfg["master_seed"], idx, n, geom)
    if control:
        sm, changed = uniform_matrix(n, s), 0
    else:
        sm, changed = slew_matrix(theta, C_FACTOR[geom] * s)
    return inst, sm, theta, changed, real


def oracle_g(model):
    """
    Class-aware upper bound for a slew matrix: in every world (hidden class vector) the exact deterministic optimum.
    Because the matrix satisfies the triangle inequality, splitting a dwell never helps, so an optimal schedule serves a
    subset contiguously; the oracle enumerates subsets and all orders of the subset (n <= 5). Not an implementable policy.
    """
    n = model.n
    total = 0.0
    for c, w in model.worlds():
        p = [model.D[k] for k in c]
        best = 0.0
        for mask in range(1, 1 << n):
            items = [i for i in range(n) if mask >> i & 1]
            val = sum(model.v[i] for i in items)
            if val <= best:
                continue
            for order in itertools.permutations(items):
                t, last, ok = 0, NO_TARGET, True
                for i in order:
                    t += model.slew(last, i) + p[i]
                    last = i
                    if t > model.d[i]:
                        ok = False
                        break
                if ok:
                    best = val
                    break
        total += w * best
    return total


def run_instance(cfg, n, rho, s, a, geom, idx, control=False, audit=False):
    """One instance: DP optimum and the exact value of every method; invariants are written to the row."""
    t0 = time.perf_counter()
    inst, sm, theta, changed, real = a13_instance(cfg, n, rho, s, a, geom, idx, control)
    model = ModelG(inst, sm)
    dp = solve(model)
    row = dict(n=n, rho=rho, s=s, a=a, geom=geom, idx=idx, V_dp=dp.value, n_states=dp.n_states, n_closed=changed, **real)
    off = [sm[i][j] for i in range(n) for j in range(n) if i != j]
    row["mean_slew"] = float(np.mean(off))
    row["cv_slew"] = float(np.std(off) / np.mean(off)) if np.mean(off) > 0 else 0.0
    viol = []
    pols = {}
    for m in METHODS:
        pols[m] = make_rule(model, m)
        v = evaluate_policy(model, pols[m])
        row["V_" + m] = v
        if not (-TOL <= v <= dp.value + TOL):
            viol.append(f"range:{m}")
    for r, b in BASE_OF.items():
        if row["V_" + r] < row["V_" + b] - TOL:
            viol.append(f"rollout_below_base:{r}")
    s0 = model.initial_state()
    row["first_dp"], row["first_cep"] = dp.policy()(s0), pols["CEP"](s0)
    row["first_differs"] = int(row["first_dp"] != row["first_cep"])
    if audit:
        if abs(evaluate_policy(model, dp.policy()) - dp.value) > TOL:
            viol.append("policy_value")
        orc = oracle_g(model)
        row["V_oracle"] = orc
        if orc < dp.value - TOL:
            viol.append("oracle_below_dp")
    row["violations"] = ";".join(viol)
    row["sec"] = time.perf_counter() - t0
    return row
