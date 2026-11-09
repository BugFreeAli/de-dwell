import numpy as np
import pytest

from brute2 import brute_value2
from test_dp_bruteforce import random_instance
from dwell.dp import solve
from dwell.instances import make_instance, make_instance_k
from dwell.m0 import Model
from dwell.m2 import Model2, evaluate_policy2
from dwell.oracle import oracle_value
from dwell.rules import make_rule

KAPPAS = [1.0, 0.5, 0.0]
ALL = ["FAIR", "EDF", "LLF", "SE", "GMR", "BI", "CEP", "LA1"]


@pytest.mark.parametrize("n,D", [(3, (2, 5)), (4, (1, 3, 6)), (4, (2, 5))])
def test_model2_with_full_retention_equals_model0(n, D):
    rng = np.random.default_rng(31 + n)
    for _ in range(10):
        inst = random_instance(rng, n, D, int(rng.integers(0, 3)), dmax_cap=14)
        assert solve(Model2(inst, 1.0)).value == pytest.approx(solve(Model(inst)).value, abs=1e-9)


@pytest.mark.parametrize("kappa", KAPPAS)
@pytest.mark.parametrize("n,D", [(2, (1, 3)), (3, (1, 3)), (3, (2, 4)), (2, (1, 2, 4)), (3, (1, 2, 4))])
@pytest.mark.parametrize("s", [0, 1, 2])
def test_model2_dp_equals_bruteforce(kappa, n, D, s):
    rng = np.random.default_rng(int(100 * kappa) + 7 * n + s + len(D))
    for _ in range(8):
        inst = random_instance(rng, n, D, s, dmax_cap=7)
        dp_val = solve(Model2(inst, kappa)).value
        bf_val = brute_value2(inst.D, inst.s, inst.values, inst.deadlines, inst.prior, kappa)
        assert dp_val == pytest.approx(bf_val, abs=1e-9), (kappa, inst)


@pytest.mark.parametrize("kappa", KAPPAS)
def test_model2_policy_value_and_ordering(kappa):
    rng = np.random.default_rng(55 + int(10 * kappa))
    for _ in range(8):
        inst = random_instance(rng, 4, (1, 3, 6), int(rng.integers(0, 4)), dmax_cap=14)
        m = Model2(inst, kappa)
        res = solve(m)
        assert evaluate_policy2(m, res.policy()) == pytest.approx(res.value, abs=1e-9)
        assert oracle_value(m) >= res.value - 1e-9
        for r in ALL:
            v = evaluate_policy2(m, make_rule(m, r))
            assert -1e-12 <= v <= res.value + 1e-9, (r, kappa)


@pytest.mark.parametrize("kappa", KAPPAS)
def test_cep_and_la1_optimal_when_classes_known(kappa):
    rng = np.random.default_rng(77 + int(10 * kappa))
    for _ in range(12):
        inst = random_instance(rng, 4, (2, 5), int(rng.integers(0, 4)), dmax_cap=14, degenerate=True)
        m = Model2(inst, kappa)
        dp = solve(m).value
        for r in ("CEP", "LA1"):
            assert evaluate_policy2(m, make_rule(m, r)) == pytest.approx(dp, abs=1e-9)


def test_make_instance_k_matches_make_instance_for_two_classes():
    for idx in range(5):
        a_inst, a_real = make_instance(9, idx, 5, 1.1, 2, 0.7)
        b_inst, b_real = make_instance_k(9, idx, 5, 1.1, 2, 0.7)
        assert a_inst.values == b_inst.values and a_inst.deadlines == b_inst.deadlines
        assert np.allclose(np.array(a_inst.prior), np.array(b_inst.prior))
        assert a_real["d_max"] == b_real["d_max"] and a_real["info_bits"] == pytest.approx(b_real["info_bits"])
