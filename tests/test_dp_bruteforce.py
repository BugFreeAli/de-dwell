"""Unit tests: DP vs independent brute force, exact policy evaluation, oracle bound, hand-checked cases."""
import itertools
import numpy as np
import pytest

from brute import brute_value
from dwell.m0 import Instance, Model, evaluate_policy
from dwell.dp import solve
from dwell.oracle import oracle_value
from dwell.rules import SIMPLE_RULES, make_rule


def random_instance(rng, n, D, s, dmax_cap=8, degenerate=False):
    K = len(D)
    prior = []
    for _ in range(n):
        if degenerate:
            p = np.zeros(K); p[rng.integers(K)] = 1.0
        else:
            p = rng.dirichlet(np.ones(K))
            if rng.random() < 0.3:                       # sometimes remove a class (zero prior mass)
                p[rng.integers(K)] = 0.0
                if p.sum() == 0:
                    p[0] = 1.0
                p = p / p.sum()
        prior.append(tuple(float(x) for x in p))
    values = tuple(float(v) for v in rng.integers(1, 4, size=n))
    deadlines = tuple(int(d) for d in rng.integers(1, dmax_cap + 1, size=n))
    return Instance(D=tuple(D), s=s, values=values, deadlines=deadlines, prior=tuple(prior))


TINY_CONFIGS = [(2, (1, 3)), (2, (2, 4)), (3, (1, 3)), (3, (2, 4)), (3, (1, 2, 4)), (2, (1, 2, 4))]


@pytest.mark.parametrize("n,D", TINY_CONFIGS)
@pytest.mark.parametrize("s", [0, 1, 2])
def test_dp_equals_bruteforce(n, D, s):
    rng = np.random.default_rng(1000 * n + 10 * s + len(D))
    for _ in range(12):
        inst = random_instance(rng, n, D, s, dmax_cap=7)
        dp_val = solve(Model(inst)).value
        bf_val = brute_value(inst.D, inst.s, inst.values, inst.deadlines, inst.prior)
        assert dp_val == pytest.approx(bf_val, abs=1e-9), inst


@pytest.mark.parametrize("n,D", [(3, (2, 5)), (4, (2, 5)), (4, (1, 3, 6))])
@pytest.mark.parametrize("s", [0, 1, 3])
def test_dp_policy_simulated_equals_dp_value(n, D, s):
    """Exact evaluation of the extracted DP policy (world enumeration) must reproduce the DP value."""
    rng = np.random.default_rng(7 * n + s)
    for _ in range(10):
        inst = random_instance(rng, n, D, s, dmax_cap=14)
        m = Model(inst)
        res = solve(m)
        assert evaluate_policy(m, res.policy()) == pytest.approx(res.value, abs=1e-9)


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("s", [0, 2])
def test_ordering_oracle_dp_rules(n, s):
    """oracle >= DP optimum >= every rule (up to float tolerance); all values >= 0."""
    rng = np.random.default_rng(31 * n + s)
    for _ in range(8):
        inst = random_instance(rng, n, (2, 5), s, dmax_cap=16)
        m = Model(inst)
        res = solve(m)
        orc = oracle_value(m)
        assert orc >= res.value - 1e-9
        for r in SIMPLE_RULES:
            v = evaluate_policy(m, make_rule(m, r))
            assert -1e-12 <= v <= res.value + 1e-9, (r, v, res.value, inst)


@pytest.mark.parametrize("n,D,s", [(3, (2, 5), 1), (4, (1, 3), 2), (3, (1, 2, 4), 0)])
def test_known_classes_dp_equals_oracle(n, D, s):
    """With one-hot priors there is no hidden information: DP optimum == class-aware oracle."""
    rng = np.random.default_rng(99 + n)
    for _ in range(15):
        inst = random_instance(rng, n, D, s, dmax_cap=12, degenerate=True)
        m = Model(inst)
        assert solve(m).value == pytest.approx(oracle_value(m), abs=1e-9)


def test_hand_case_single_threat():
    # D=(2,5), prior 50/50, deadline 5: both classes feasible -> value v.  deadline 4: only soft -> 0.5 v.
    for d, expected in [(5, 3.0), (4, 1.5), (1, 0.0)]:
        inst = Instance(D=(2, 5), s=1, values=(3.0,), deadlines=(d,), prior=((0.5, 0.5),))
        assert solve(Model(inst)).value == pytest.approx(expected)


def test_hand_case_slew():
    # Known class, D=(2,), s=1. Serve threat 0 first (done at t=2), slew 1 + dwell 2 -> done at t=5.
    for dl, expected in [((2, 5), 2.0), ((2, 4), 1.0)]:
        inst = Instance(D=(2,), s=1, values=(1.0, 1.0), deadlines=dl, prior=((1.0,), (1.0,)))
        m = Model(inst)
        assert solve(m).value == pytest.approx(expected)
        assert oracle_value(m) == pytest.approx(expected)
        assert evaluate_policy(m, make_rule(m, "EDF")) == pytest.approx(expected)


def test_hand_case_probe_and_abandon():
    # Threat 0: soft(2)/hard(5) 50/50, value 1, deadline 9.  Threat 1: known soft(2), value 1, deadline 9, s=0.
    # Optimal: kill the certain threat (2 steps), then work on threat 0 (finish by 2+5=7 <= 9): value 2 - 0 = 2.
    inst = Instance(D=(2, 5), s=0, values=(1.0, 1.0), deadlines=(9, 9), prior=((0.5, 0.5), (1.0, 0.0)))
    assert solve(Model(inst)).value == pytest.approx(2.0)


def test_rules_return_only_useful_actions():
    rng = np.random.default_rng(5)
    inst = random_instance(rng, 4, (2, 5), 2, dmax_cap=12)
    m = Model(inst)
    st = m.initial_state()
    for r in SIMPLE_RULES:
        j = make_rule(m, r)(st)
        assert j is None or j in m.actions(st)
