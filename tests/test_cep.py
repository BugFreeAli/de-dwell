import numpy as np
import pytest

from test_dp_bruteforce import random_instance
from dwell.dp import solve
from dwell.m0 import Model, evaluate_policy
from dwell.rules import make_rule


@pytest.mark.parametrize("n,D,s", [(3, (2, 5), 0), (4, (2, 5), 2), (5, (2, 5), 1), (4, (1, 3, 6), 2)])
def test_cep_is_optimal_when_classes_are_known(n, D, s):
    rng = np.random.default_rng(500 + n + s)
    for _ in range(15):
        inst = random_instance(rng, n, D, s, dmax_cap=14, degenerate=True)
        m = Model(inst)
        assert evaluate_policy(m, make_rule(m, "CEP")) == pytest.approx(solve(m).value, abs=1e-9)


@pytest.mark.parametrize("n,s", [(3, 0), (4, 2), (5, 1)])
def test_cep_between_zero_and_dp(n, s):
    rng = np.random.default_rng(900 + n + s)
    for _ in range(10):
        inst = random_instance(rng, n, (2, 5), s, dmax_cap=16)
        m = Model(inst)
        v = evaluate_policy(m, make_rule(m, "CEP"))
        assert -1e-12 <= v <= solve(m).value + 1e-9


def test_cep_returns_useful_action_or_none():
    rng = np.random.default_rng(4)
    for _ in range(20):
        inst = random_instance(rng, 4, (2, 5), 2, dmax_cap=12)
        m = Model(inst)
        st = m.initial_state()
        j = make_rule(m, "CEP")(st)
        assert (j is None and not m.actions(st)) or j in m.actions(st)
