"""Tests of the A13 model, generator, oracle and row invariants (plan A13.6, items 1 to 5)."""
import math
from functools import lru_cache

import numpy as np
import pytest

from brute_g import brute_value_g
from dwell.dp import solve
from dwell.instances import make_instance_k
from dwell.m0 import Instance, Model, evaluate_policy
from dwell.m0g import (C_FACTOR, CLUSTER_CENTRES, CLUSTER_HALF_WIDTH, GEOMETRIES, METHODS, ModelG, a13_instance,
                       bearings, oracle_g, run_instance, slew_matrix, uniform_matrix)
from dwell.rules import make_rule

CFG = dict(master_seed=11, D0=[5, 10], value_set=[1, 2, 3], u_low=0.35)


def random_instance(rng, n, K):
    D = tuple(sorted(rng.choice(np.arange(2, 7), size=K, replace=False).tolist()))
    prior = []
    for _ in range(n):
        p = rng.random(K) + 0.05
        prior.append(tuple((p / p.sum()).tolist()))
    return Instance(D=D, s=int(rng.integers(0, 5)), values=tuple(int(x) for x in rng.integers(1, 4, size=n)),
                    deadlines=tuple(int(x) for x in rng.integers(3, 13, size=n)), prior=tuple(prior))


# ---- generator -----------------------------------------------------------------------------------------------
def test_bearings_ranges_and_groups():
    for idx in range(50):
        sp = bearings(5, idx, 5, "SPREAD")
        assert len(sp) == 5 and all(0.0 <= x <= 1.0 for x in sp)
        cl = bearings(5, idx, 5, "CLUSTER")
        for x in cl:
            assert any(abs(x - c) <= CLUSTER_HALF_WIDTH + 1e-12 for c in CLUSTER_CENTRES)
    assert bearings(5, 3, 4, "SPREAD") != bearings(5, 3, 4, "CLUSTER")
    assert bearings(5, 3, 4, "CLUSTER") == bearings(5, 3, 4, "CLUSTER")


def test_matrix_is_symmetric_zero_diagonal_and_metric():
    changed_any = False
    for geom in GEOMETRIES:
        for idx in range(300):
            th = bearings(7, idx, 5, geom)
            m, ch = slew_matrix(th, C_FACTOR[geom] * 2)
            changed_any |= ch > 0
            n = len(m)
            for i in range(n):
                assert m[i][i] == 0
                for j in range(n):
                    assert m[i][j] == m[j][i] >= 0
                    for k in range(n):
                        assert m[i][j] <= m[i][k] + m[k][j]
    assert changed_any          # the closure does act on rounded matrices (otherwise this test could not see it)


@pytest.mark.parametrize("geom", GEOMETRIES)
@pytest.mark.parametrize("sbar", [2, 4])
def test_mean_pairwise_slew_close_to_sbar(geom, sbar):
    vals = []
    for idx in range(2000):
        m, _ = slew_matrix(bearings(9, idx, 5, geom), C_FACTOR[geom] * sbar)
        vals.append(np.mean([m[i][j] for i in range(5) for j in range(5) if i != j]))
    assert abs(np.mean(vals) - sbar) < 0.3


def test_instances_equal_earlier_amendments():
    for idx in range(3):
        inst, sm, th, ch, real = a13_instance(CFG, 4, 1.1, 2, 0.9, "SPREAD", idx)
        i2, real2 = make_instance_k(CFG["master_seed"], idx, 4, 1.1, 2, 0.9, D=(5, 10), value_set=(1, 2, 3), u_low=0.35)
        assert inst == i2 and real == real2


# ---- model against M0 and against brute force ---------------------------------------------------------------
def test_uniform_matrix_reproduces_m0_for_dp_and_every_method():
    rng = np.random.default_rng(1)
    for _ in range(40):
        inst = random_instance(rng, int(rng.integers(2, 5)), int(rng.integers(2, 4)))
        m0, mg = Model(inst), ModelG(inst, uniform_matrix(inst.n, inst.s))
        assert abs(solve(m0).value - solve(mg).value) < 1e-9
        for m in METHODS:
            assert abs(evaluate_policy(m0, make_rule(m0, m)) - evaluate_policy(mg, make_rule(mg, m))) < 1e-9


@pytest.mark.parametrize("seed", range(6))
def test_dp_equals_bruteforce_with_asymmetric_matrices(seed):
    rng = np.random.default_rng(100 + seed)
    for n in (2, 3):
        inst = random_instance(rng, n, 2)
        sm = tuple(tuple(0 if i == j else int(rng.integers(0, 5)) for j in range(n)) for i in range(n))
        dp = solve(ModelG(inst, sm)).value
        assert abs(dp - brute_value_g(inst.D, sm, inst.values, inst.deadlines, inst.prior)) < 1e-9


def world_dp(model, p):
    """Exact deterministic optimum of one world with preemption allowed (independent of oracle_g)."""
    n = model.n

    @lru_cache(maxsize=None)
    def f(t, cur, prog):
        best = 0.0
        for j in range(n):
            if prog[j] >= p[j]:
                continue
            t2 = t + (model.sm[cur][j] if cur != -1 and cur != j else 0) + 1
            if t2 > model.d[j]:
                continue
            q = prog[:j] + (prog[j] + 1,) + prog[j + 1:]
            best = max(best, (model.v[j] if q[j] == p[j] else 0.0) + f(t2, j, q))
        return best

    return f(0, -1, (0,) * n)


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_oracle_equals_exact_world_dp_with_preemption(geom):
    for idx in range(6):
        inst, sm, th, ch, real = a13_instance(CFG, 3, 1.1, 4, 0.5, geom, idx)
        mg = ModelG(inst, sm)
        ref = sum(w * world_dp(mg, tuple(mg.D[k] for k in c)) for c, w in mg.worlds())
        assert abs(oracle_g(mg) - ref) < 1e-9


# ---- rows --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("geom", GEOMETRIES)
def test_row_invariants_and_columns(geom):
    for idx, (n, rho, s, a) in enumerate([(4, 1.1, 2, 0.9), (5, 1.5, 4, 0.5), (3, 1.1, 4, 0.9)]):
        row = run_instance(CFG, n, rho, s, a, geom, idx, audit=True)
        assert row["violations"] == ""
        for m in METHODS:
            assert -1e-9 <= row["V_" + m] <= row["V_dp"] + 1e-9
        assert row["V_oracle"] >= row["V_dp"] - 1e-9
        for r, b in (("ROLL_CEP", "CEP"), ("ROLL_GMR", "GMR"), ("ROLL_EDF", "EDF")):
            assert row["V_" + r] >= row["V_" + b] - 1e-9
        assert row["first_differs"] in (0, 1)


def test_control_is_uniform_and_equals_m0():
    for idx in range(4):
        inst, sm, th, ch, real = a13_instance(CFG, 4, 1.1, 2, 0.9, "CLUSTER", idx, control=True)
        assert sm == uniform_matrix(4, 2)
        row = run_instance(CFG, 4, 1.1, 2, 0.9, "CLUSTER", idx, control=True)
        assert abs(row["V_dp"] - solve(Model(inst)).value) < 1e-9
