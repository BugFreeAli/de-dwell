"""Tests of the A14 goal objective (plan A14.7, items 1 to 6): threshold rule, exact DP against an independent brute force,
policy values, rollout improvement, CEPS on known classes, instance identity, row invariants."""
import math

import numpy as np
import pytest

from brute_u import brute_goal
from dwell.dp import solve
from dwell.instances import make_instance_k
from dwell.m0 import DEAD, Instance, Model
from dwell.m0u import (AWARE, BASE_OF_RU, EV_BASED, METHODS, GoalDP, a14_instance, build_policies, ceps_policy,
                       need_of, prob_goal, run_instance, ru_policy)

CFG = dict(master_seed=11, D0=[5, 10], value_set=[1, 2, 3], u_low=0.35)


def random_instance(rng, n, K, onehot=False):
    D = tuple(sorted(rng.choice(np.arange(2, 7), size=K, replace=False).tolist()))
    prior = []
    for _ in range(n):
        if onehot:
            p = np.zeros(K)
            p[int(rng.integers(K))] = 1.0
        else:
            p = rng.random(K) + 0.05
            p = p / p.sum()
        prior.append(tuple(p.tolist()))
    return Instance(D=D, s=int(rng.integers(0, 5)), values=tuple(float(x) for x in rng.integers(1, 4, size=n)),
                    deadlines=tuple(int(x) for x in rng.integers(3, 13, size=n)), prior=tuple(prior))


def matrix(inst):
    n = len(inst.values)
    return [[0 if i == j else inst.s for j in range(n)] for i in range(n)]


# ---- 1. threshold rule ----------------------------------------------------------------------------------------------
def test_need_rule():
    assert need_of([1, 2, 3, 1], 0.25) == 6          # 0.75 * 7 = 5.25 -> 6
    assert need_of([1, 2, 3, 1], 0.5) == 4           # 3.5 -> 4
    assert need_of([2, 2, 2, 2], 0.25) == 6          # exactly 6.0 stays 6 (no float spill to 7)
    assert need_of([1, 2, 3, 1], 0.0) == 7           # kill everything
    assert need_of([1, 2], 1.0) <= 0                 # nothing needed


def test_lam_does_not_change_the_instance():
    a, na, _ = a14_instance(CFG, 4, 1.1, 2, 0.5, 0.25, 3)
    b, nb, _ = a14_instance(CFG, 4, 1.1, 2, 0.5, 0.5, 3)
    assert a == b and na >= nb
    ref, _ = make_instance_k(CFG["master_seed"], 3, 4, 1.1, 2, 0.5, D=(5, 10), value_set=(1, 2, 3), u_low=0.35)
    assert a == ref                                   # identical to the instances of A6 to A13


# ---- 2. exact DP against an independent brute force ---------------------------------------------------------------
@pytest.mark.parametrize("seed", range(12))
def test_goal_dp_equals_brute_force(seed):
    rng = np.random.default_rng(seed)
    n, K = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    inst = random_instance(rng, n, K)
    W = sum(inst.values)
    for need in (0, 1, math.ceil(0.5 * W), math.ceil(0.75 * W), W, W + 1):
        ref = brute_goal(inst.D, matrix(inst), inst.values, inst.deadlines, inst.prior, need)
        got = GoalDP(Model(inst), need).value
        assert abs(ref - got) < 1e-9, (seed, need, ref, got)


def test_goal_value_is_monotone_in_need():
    rng = np.random.default_rng(5)
    for _ in range(8):
        inst = random_instance(rng, 3, 2)
        vals = [GoalDP(Model(inst), need).value for need in range(0, int(sum(inst.values)) + 2)]
        assert vals[0] == 1.0 and vals[-1] == 0.0
        assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))


# ---- 3. policy value, bounds ---------------------------------------------------------------------------------------
def test_extracted_goal_policy_attains_the_value():
    rng = np.random.default_rng(7)
    for _ in range(6):
        inst = random_instance(rng, 3, 2)
        m = Model(inst)
        need = math.ceil(0.6 * sum(inst.values))
        g = GoalDP(m, need)
        assert abs(prob_goal(m, g.policy(), need) - g.value) < 1e-9


def test_every_method_is_between_zero_and_the_optimum():
    rng = np.random.default_rng(9)
    for _ in range(5):
        inst = random_instance(rng, 3, 2)
        m = Model(inst)
        need = math.ceil(0.7 * sum(inst.values))
        g = GoalDP(m, need)
        pols = build_policies(m, need)
        for name in METHODS:
            u = prob_goal(m, pols[name], need)
            assert -1e-9 <= u <= g.value + 1e-9, (name, u, g.value)


def test_trivial_goal_is_one_for_every_method():
    rng = np.random.default_rng(3)
    inst = random_instance(rng, 3, 2)
    m = Model(inst)
    pols = build_policies(m, 0)
    assert all(abs(prob_goal(m, pols[name], 0) - 1.0) < 1e-12 for name in METHODS)


def test_impossible_goal_is_zero():
    rng = np.random.default_rng(4)
    inst = random_instance(rng, 3, 2)
    m = Model(inst)
    need = int(sum(inst.values)) + 1
    assert GoalDP(m, need).value == 0.0


# ---- 4. rollout improvement ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_rollout_never_below_its_base(seed):
    rng = np.random.default_rng(100 + seed)
    inst = random_instance(rng, 3, 2)
    m = Model(inst)
    need = math.ceil(0.6 * sum(inst.values))
    pols = build_policies(m, need)
    for ru, base in BASE_OF_RU.items():
        assert prob_goal(m, pols[ru], need) >= prob_goal(m, pols[base], need) - 1e-9, (ru, base)


def test_rollout_of_the_goal_optimum_keeps_the_optimum():
    rng = np.random.default_rng(21)
    inst = random_instance(rng, 3, 2)
    m = Model(inst)
    need = math.ceil(0.6 * sum(inst.values))
    g = GoalDP(m, need)
    pol = ru_policy(m, g.policy(), need)
    assert abs(prob_goal(m, pol, need) - g.value) < 1e-9


# ---- 5. CEPS on known classes ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(10))
def test_ceps_is_optimal_when_classes_are_known(seed):
    rng = np.random.default_rng(300 + seed)
    inst = random_instance(rng, int(rng.integers(2, 5)), 2, onehot=True)
    m = Model(inst)
    for frac in (0.5, 0.75, 1.0):
        need = math.ceil(frac * sum(inst.values))
        g = GoalDP(m, need)
        assert abs(prob_goal(m, ceps_policy(m, need), need) - g.value) < 1e-9, (seed, frac)


# ---- 6. rows ---------------------------------------------------------------------------------------------------------
def test_row_has_all_columns_and_no_violation():
    row = run_instance(CFG, 4, 1.1, 2, 0.5, 0.25, 0, audit=True)
    assert row["violations"] == ""
    for m in METHODS:
        assert "U_" + m in row
    assert 0.0 <= row["U_opt"] <= 1.0 and row["need"] == need_of(
        make_instance_k(CFG["master_seed"], 0, 4, 1.1, 2, 0.5, D=(5, 10), value_set=(1, 2, 3), u_low=0.35)[0].values, 0.25)
    assert row["first_differs"] in (0, 1)


def test_row_is_deterministic():
    a = run_instance(CFG, 4, 1.5, 4, 0.9, 0.5, 2)
    b = run_instance(CFG, 4, 1.5, 4, 0.9, 0.5, 2)
    assert {k: v for k, v in a.items() if k != "sec"} == {k: v for k, v in b.items() if k != "sec"}


def test_method_sets():
    assert set(EV_BASED) | set(AWARE) == set(METHODS) and not set(EV_BASED) & set(AWARE) and len(METHODS) == 17


# ---- 7. DPEV is the optimum of the EXPECTED-VALUE objective ---------------------------------------------------------
def expected_killed_value(model, policy):
    cache, total = {}, 0.0
    for c, w in model.worlds():
        state, acc = model.initial_state(), 0.0
        while True:
            key = (state, acc)
            j = cache[key] if key in cache else cache.setdefault(key, policy(state, acc))
            if j is None:
                break
            t, cur, prog = state
            t2 = t + model.slew(cur, j) + 1
            p2 = list(prog)
            if prog[j] + 1 == model.D[c[j]]:
                acc += model.v[j]
                p2[j] = DEAD
            else:
                p2[j] = prog[j] + 1
            state = (t2, j, model.canon(t2, p2))
        total += w * acc
    return total


@pytest.mark.parametrize("seed", range(5))
def test_dpev_attains_the_expected_value_optimum(seed):
    rng = np.random.default_rng(500 + seed)
    inst = random_instance(rng, 3, 2)
    m = Model(inst)
    pols = build_policies(m, math.ceil(0.6 * sum(inst.values)))
    assert abs(expected_killed_value(m, pols["DPEV"]) - solve(m).value) < 1e-9


# ---- 8. cases that pin the prune, CEPS feasibility and the rollout's first action ---------------------------------------
def test_kill_everything_is_possible_and_matches_brute_force():
    inst = Instance(D=(2, 3), s=1, values=(1.0, 2.0, 1.0), deadlines=(9, 9, 9),
                    prior=((0.5, 0.5), (0.5, 0.5), (0.5, 0.5)))
    need = 4
    ref = brute_goal(inst.D, matrix(inst), inst.values, inst.deadlines, inst.prior, need)
    got = GoalDP(Model(inst), need).value
    assert ref > 0.1 and abs(ref - got) < 1e-12        # the prune must not cut the state in which every kill is still needed


def test_ceps_prefers_the_valuable_feasible_plan_when_the_goal_is_out_of_reach():
    inst = Instance(D=(2, 3), s=0, values=(1.0, 2.0), deadlines=(3, 3), prior=((1.0, 0.0), (1.0, 0.0)))
    m = Model(inst)
    # killing both needs 4 steps but both breach at 3: the goal (3) is out of reach, so the planner must take the value-2 threat
    assert ceps_policy(m, 3)(m.initial_state(), 0.0) == 1


@pytest.mark.parametrize("seed", range(8))
def test_rollout_first_action_is_the_best_deviation_from_its_base(seed):
    rng = np.random.default_rng(700 + seed)
    inst = random_instance(rng, 3, 2)
    m = Model(inst)
    need = math.ceil(0.6 * sum(inst.values))
    pols = build_policies(m, need)
    s0 = m.initial_state()
    for ru, base in (("RU_CEP", "CEP"), ("RU_DPEV", "DPEV")):
        def deviate(j, b=pols[base]):
            return lambda st, ac: j if (st == s0 and ac == 0.0) else b(st, ac)
        q = {j: prob_goal(m, deviate(j), need) for j in m.actions(s0)}
        if not q:
            continue
        j_ru = pols[ru](s0, 0.0)
        assert q[j_ru] >= max(q.values()) - 1e-9, (seed, ru, q, j_ru)


def test_rollout_credits_a_kill_made_by_its_own_first_action():
    # X (index 0) needs 2 steps and is worth 1; A (index 1) dies in one step and is worth 3; the goal is 3.
    # The base never acts, so only the value gained by the first action can tell the actions apart.
    inst = Instance(D=(1, 2), s=0, values=(1.0, 3.0), deadlines=(5, 5), prior=((0.0, 1.0), (1.0, 0.0)))
    m = Model(inst)
    assert ru_policy(m, lambda st, ac: None, 3)(m.initial_state(), 0.0) == 1
