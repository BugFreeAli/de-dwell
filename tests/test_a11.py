"""Tests for M1 and the 22 methods (plan A11.7). Run: python -m pytest tests/test_a11.py -q"""
import math
import numpy as np
import pytest

from brute_m1 import brute_value
from dwell.dp import solve
from dwell.instances import make_instance_k
from dwell.m0 import Instance, Model, evaluate_policy
from dwell.m1 import (IDLE, Model1, energy_budget, evaluate_policy1, oracle_value1, rho_boost, solve1,
                      two_level_gtab)
from dwell.oracle import oracle_value
from dwell.rules import make_rule
from dwell.rules1 import METHODS, RULES, make_method, split

TOL = 1e-9


def rand_inst(rng, n, K, smax=4, dmax_cap=14):
    base = sorted(rng.choice(np.arange(1, 9), size=K, replace=False).tolist())
    prior = []
    for _ in range(n):
        w = rng.dirichlet(np.ones(K) * 0.7)
        w = np.maximum(w, 0.02); w = w / w.sum()
        prior.append(tuple(float(x) for x in w))
    dm = int(rng.integers(max(base[0], 4), dmax_cap))
    dl = [max(base[0], int(round(rng.uniform(0.35, 1) * dm))) for _ in range(n)]
    dl[int(rng.integers(n))] = dm
    return Instance(D=tuple(base), s=int(rng.integers(0, smax + 1)), values=tuple(float(rng.choice([1, 2, 3])) for _ in range(n)),
                    deadlines=tuple(dl), prior=tuple(prior))


def const_g(c, inst):
    return (1,) + (c,) * (max(inst.deadlines) + 2)


def monotone_g(rng, inst):
    """Rates that do not fall as the breach approaches (non-increasing in tau), integers >= 1."""
    T = max(inst.deadlines) + 2
    steps = sorted((int(x) for x in rng.integers(1, 5, size=T)))          # ascending over steps = descending in tau
    return (1,) + tuple(reversed(steps))


# ---- (1) regression: constant rate reproduces M0 -----------------------------------------------------------
def test_regression_rate_one_dp_equals_m0():
    rng = np.random.default_rng(11)
    for it in range(500):
        n = int(rng.integers(2, 5)); K = int(rng.integers(2, 4))
        inst = rand_inst(rng, n, K)
        v0 = solve(Model(inst)).value
        v1 = solve1(Model1(inst, const_g(1, inst))).value           # idle allowed, no budget
        assert abs(v0 - v1) < TOL, (it, v0, v1)


def test_regression_rate_one_every_method_equals_m0_rule():
    """The 16 non-rollout methods equal the M0 rule with idle enabled. The six rollouts equal the M0 rollout when idle is
    disabled (code-path regression); with idle enabled a rollout may strictly improve by deferring one step, which is
    counted in test_rollout_idle_difference_at_rate_one and reported, not asserted equal (plan sub-amendment A11.r1)."""
    rng = np.random.default_rng(12)
    for it in range(60):
        n = int(rng.integers(2, 5)); K = int(rng.integers(2, 4))
        inst = rand_inst(rng, n, K)
        m0 = Model(inst)
        m1 = Model1(inst, const_g(1, inst))
        m1_noidle = Model1(inst, const_g(1, inst), None, False)
        for name in METHODS:
            rule, var = split(name)
            v0 = evaluate_policy(m0, make_rule(m0, rule))
            mm = m1_noidle if name.startswith("ROLL") else m1
            v1 = evaluate_policy1(mm, make_method(mm, name))
            assert abs(v0 - v1) < TOL, (it, name, v0, v1)


def test_rollout_idle_difference_at_rate_one():
    """With idle enabled and g = 1 a rollout is never worse than its base, but may differ from the M0 rollout."""
    rng = np.random.default_rng(14)
    n_diff = n_tot = 0
    for it in range(40):
        inst = rand_inst(rng, int(rng.integers(2, 5)), 2)
        m0, m1 = Model(inst), Model1(inst, const_g(1, inst))
        for r in ("CEP", "GMR", "EDF"):
            v0 = evaluate_policy(m0, make_rule(m0, "ROLL_" + r))
            v1 = evaluate_policy1(m1, make_method(m1, f"ROLL_{r}_M"))
            base = evaluate_policy(m0, make_rule(m0, r))
            assert v1 >= base - 1e-9
            n_tot += 1; n_diff += abs(v0 - v1) > 1e-9
    print(f"rollout with idle differs from the M0 rollout at g=1 in {n_diff} of {n_tot} cases")


def test_regression_scaling_constant_rate():
    rng = np.random.default_rng(13)
    done = 0
    while done < 80:
        n = int(rng.integers(2, 5)); K = 2
        inst = rand_inst(rng, n, K)
        c = int(rng.choice([2, 3]))
        mode = done % 2
        if mode == 0:                                   # D' = c D
            D2 = tuple(c * d for d in inst.D); D0 = inst.D
        else:                                           # arbitrary D', M0 with ceil(D'/c)
            D2 = tuple(int(x) for x in sorted(rng.choice(np.arange(2, 16), size=K, replace=False)))
            D0 = tuple(-(-d // c) for d in D2)
            if len(set(D0)) < K:
                continue
        i0 = Instance(D=D0, s=inst.s, values=inst.values, deadlines=inst.deadlines, prior=inst.prior)
        i2 = Instance(D=D2, s=inst.s, values=inst.values, deadlines=inst.deadlines, prior=inst.prior)
        assert abs(solve(Model(i0)).value - solve1(Model1(i2, const_g(c, inst))).value) < TOL
        done += 1


# ---- DP against the independent brute force ---------------------------------------------------------------
@pytest.mark.parametrize("seed", [21, 22, 23])
def test_dp_equals_bruteforce(seed):
    rng = np.random.default_rng(seed)
    for it in range(40):
        n = int(rng.integers(2, 4)); K = 2
        inst = rand_inst(rng, n, K, smax=3, dmax_cap=10)
        T = max(inst.deadlines) + 2
        g = (1,) + tuple(int(x) for x in rng.integers(1, 4, size=T))      # arbitrary (non-monotone) profile
        B = [None, int(rng.integers(1, 6))][it % 2]
        for idle in (True, False):
            bf = brute_value(inst.D, inst.s, inst.values, inst.deadlines, inst.prior, g, B, idle)
            dp = solve1(Model1(inst, g, B, idle)).value
            assert abs(bf - dp) < 1e-9, (it, B, idle, bf, dp)


# ---- idle dominance --------------------------------------------------------------------------------------
def test_idle_never_helps_without_budget_monotone_rates():
    rng = np.random.default_rng(31)
    for it in range(300):
        n = int(rng.integers(2, 4)); K = 2
        inst = rand_inst(rng, n, K, smax=4, dmax_cap=12)
        g = monotone_g(rng, inst)
        a = solve1(Model1(inst, g, None, True)).value
        b = solve1(Model1(inst, g, None, False)).value
        assert abs(a - b) < TOL, (it, a, b)


def test_idle_never_hurts_and_helps_somewhere_with_budget():
    rng = np.random.default_rng(32)
    strict = 0
    for it in range(120):
        n = int(rng.integers(2, 4))
        inst = rand_inst(rng, n, 2, smax=3, dmax_cap=12)
        g = monotone_g(rng, inst)
        B = int(rng.integers(2, 6))
        a = solve1(Model1(inst, g, B, True)).value
        b = solve1(Model1(inst, g, B, False)).value
        assert a >= b - TOL
        strict += a > b + 1e-9
    assert strict > 0


def test_constructed_case_waiting_pays():
    # one threat, D=(4,), budget 2 dwell steps, rate 1 far away and 2 in the last two steps before breach:
    # starting now needs 4 steps; waiting until tau<=2 kills with 2 steps.
    inst = Instance(D=(4,), s=0, values=(1.0,), deadlines=(6,), prior=((1.0,),))
    g = (1, 2, 2, 1, 1, 1, 1, 1)
    assert solve1(Model1(inst, g, 2, False)).value == 0.0
    assert solve1(Model1(inst, g, 2, True)).value == 1.0
    assert solve1(Model1(inst, g, None, False)).value == 1.0


# ---- invariants ---------------------------------------------------------------------------------------------
def test_policy_value_oracle_and_methods_invariants():
    rng = np.random.default_rng(41)
    for it in range(14):
        n = int(rng.integers(2, 5))
        inst = rand_inst(rng, n, 2, smax=3, dmax_cap=12)
        g = monotone_g(rng, inst)
        B = [None, int(rng.integers(3, 9))][it % 2]
        m = Model1(inst, g, B)
        res = solve1(m)
        assert abs(evaluate_policy1(m, res.policy()) - res.value) < TOL
        assert oracle_value1(m) >= res.value - TOL
        vals = {}
        for name in METHODS:
            vals[name] = evaluate_policy1(m, make_method(m, name))
            assert -TOL <= vals[name] <= res.value + TOL, (name, vals[name], res.value)
        for rule in ("CEP", "GMR", "EDF"):
            for var in ("M", "L"):
                assert vals[f"ROLL_{rule}_{var}"] >= vals[f"{rule}_{var}"] - 1e-9, (rule, var)


def test_L_equals_M_when_rate_constant():
    rng = np.random.default_rng(42)
    for it in range(25):
        inst = rand_inst(rng, int(rng.integers(2, 5)), 2, smax=3, dmax_cap=12)
        m = Model1(inst, const_g(int(rng.integers(1, 4)), inst))
        for r in RULES:
            a = evaluate_policy1(m, make_method(m, r + "_M")); b = evaluate_policy1(m, make_method(m, r + "_L"))
            assert abs(a - b) < TOL, (r, a, b)


def test_non_rollout_methods_never_idle_and_dp_never_idles_without_budget():
    rng = np.random.default_rng(43)
    for it in range(10):
        inst = rand_inst(rng, 3, 2, smax=3, dmax_cap=12)
        m = Model1(inst, monotone_g(rng, inst), None)
        for name in METHODS:
            if not name.startswith("ROLL"):
                _, w, _ = evaluate_policy1(m, make_method(m, name), stats=True)
                assert w == 0.0, name
        _, w, _ = evaluate_policy1(m, solve1(m).policy(), stats=True)
        assert w == 0.0


def test_generator_budget_and_rho_boost():
    inst, real = make_instance_k(1, 0, 4, 1.1, 2, 0.7, D=(5, 10))
    g = two_level_gtab(2, 10, max(inst.deadlines) + 2)
    B = energy_budget(inst, 0.8)
    e_tot = sum(sum(p * d for p, d in zip(pr, inst.D)) for pr in inst.prior)
    assert B == math.ceil(0.8 * e_tot - 1e-9) and energy_budget(inst, None) is None
    assert 0 < rho_boost(inst, g) <= real["rho_hat"] + 1e-9


def test_instances_identical_to_m0_generator():
    a, _ = make_instance_k(20261221, 5, 4, 1.5, 4, 0.5, D=(5, 10))
    b, _ = make_instance_k(20261221, 5, 4, 1.5, 4, 0.5, D=(5, 10))
    assert a == b


def test_steps_list_values_myopic_vs_lookahead():
    from dwell.rules1 import steps_list
    inst = Instance(D=(6,), s=0, values=(1.0,), deadlines=(10,), prior=((1.0,),))
    g = (1,) + tuple(3 if tau <= 3 else 1 for tau in range(1, 13))
    m = Model1(inst, g)
    # at t = 4 the threat is at tau = 6: rate 1 now, rate 3 from tau = 3 on
    assert steps_list(m, "M", 4, 0, 0, 4)[0][1] == 6          # frozen rate 1: ceil(6/1)
    assert steps_list(m, "L", 4, 0, 0, 4)[0][1] == 4          # 1+1+1 (tau 6,5,4) then 3 (tau 3): 6 units in 4 steps
