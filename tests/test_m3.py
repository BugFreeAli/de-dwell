import numpy as np
import pytest

from brute3 import brute_value3
from test_dp_bruteforce import random_instance
from dwell.dp import solve
from dwell.m0 import Instance, Model
from dwell.m2 import Model2, evaluate_policy2
from dwell.m3 import Model3, evaluate_policy3
from dwell.oracle import oracle_value
from dwell.rules import make_rule


def pos_instance(rng, n, D, s, dmax_cap):
    while True:
        inst = random_instance(rng, n, D, s, dmax_cap=dmax_cap)
        if all(min(r) > 0.05 for r in inst.prior):
            return inst


@pytest.mark.parametrize("q", [0.5, 0.7, 0.95])
@pytest.mark.parametrize("D", [(2, 4), (3, 5), (1, 3)])
@pytest.mark.parametrize("n,s", [(2, 0), (2, 1), (3, 0), (3, 1)])
def test_model3_dp_equals_bruteforce(q, D, n, s):
    rng = np.random.default_rng(int(100 * q) + 13 * n + s + D[0])
    for _ in range(6):
        inst = pos_instance(rng, n, D, s, 8)
        dp = solve(Model3(inst, q)).value
        bf = brute_value3(inst.D, inst.s, inst.values, inst.deadlines, inst.prior, q)
        assert dp == pytest.approx(bf, abs=1e-9), (q, inst)


@pytest.mark.parametrize("D", [(2, 5), (3, 8)])
def test_uninformative_signal_equals_model0(D):
    rng = np.random.default_rng(8)
    for _ in range(8):
        inst = pos_instance(rng, 4, D, int(rng.integers(0, 4)), 14)
        assert solve(Model3(inst, 0.5)).value == pytest.approx(solve(Model(inst)).value, abs=1e-9)


def test_policy_recursion_matches_world_enumeration_on_model2():
    rng = np.random.default_rng(3)
    for _ in range(6):
        inst = random_instance(rng, 4, (2, 5), 2, dmax_cap=12)
        m = Model2(inst)
        for r in ("CEP", "GMR", "EDF"):
            assert evaluate_policy3(m, make_rule(m, r)) == pytest.approx(evaluate_policy2(m, make_rule(m, r)), abs=1e-9)


@pytest.mark.parametrize("q", [0.7, 0.95])
def test_model3_ordering_and_monotone_information(q):
    rng = np.random.default_rng(int(100 * q))
    for _ in range(6):
        inst = pos_instance(rng, 4, (3, 8), int(rng.integers(0, 4)), 16)
        m = Model3(inst, q)
        res = solve(m)
        assert evaluate_policy3(m, res.policy()) == pytest.approx(res.value, abs=1e-9)
        assert oracle_value(m) >= res.value - 1e-9
        assert res.value >= solve(Model(inst)).value - 1e-9
        for r in ("CEP", "GMR", "EDF", "ROLL_CEP", "ROLL_GMR"):
            v = evaluate_policy3(m, make_rule(m, r))
            assert -1e-12 <= v <= res.value + 1e-9
