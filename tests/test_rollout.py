import numpy as np
import pytest

from test_dp_bruteforce import random_instance
from dwell.dp import solve
from dwell.m0 import Model, evaluate_policy
from dwell.m2 import Model2, evaluate_policy2
from dwell.rules import make_rule

BASES = ["CEP", "GMR", "EDF", "BI"]


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("kappa,D", [(1.0, (2, 5)), (1.0, (1, 3, 9)), (0.5, (2, 5)), (0.0, (1, 8))])
def test_rollout_improves_base_and_stays_below_dp(base, kappa, D):
    rng = np.random.default_rng(int(10 * kappa) + len(D) + len(base))
    for _ in range(6):
        inst = random_instance(rng, 4, D, int(rng.integers(0, 4)), dmax_cap=14)
        m = Model2(inst, kappa)
        dp = solve(m).value
        v_base = evaluate_policy2(m, make_rule(m, base))
        v_roll = evaluate_policy2(m, make_rule(m, "ROLL_" + base))
        assert v_roll >= v_base - 1e-9
        assert v_roll <= dp + 1e-9


@pytest.mark.parametrize("base", ["CEP"])
def test_rollout_optimal_when_base_is_optimal(base):
    rng = np.random.default_rng(321)
    for _ in range(10):
        inst = random_instance(rng, 4, (2, 5), int(rng.integers(0, 4)), dmax_cap=14, degenerate=True)
        m = Model(inst)
        assert evaluate_policy(m, make_rule(m, "ROLL_" + base)) == pytest.approx(solve(m).value, abs=1e-9)
