import itertools
from functools import lru_cache

import pytest

from brute_ba import brute_value_ba
from dwell.adapt import (Kit, PRACTICAL, Planner, a7_instance, class_dwells, eval_rec, iid_truth, merge_classes,
                         merged_instance, nominal_factory)
from dwell.dp import solve
from dwell.instances import make_instance_k
from dwell.m2 import Model2
from dwell.mismatch import eval_obs

SCALES = (0.5, 1.0, 1.5)


def build(idx, n=3, rho=1.1, s=1, a=0.7, offs=(0, 1), probs=(0.5, 0.5), D0=(2, 4), scales=SCALES, seed=7):
    cfg = dict(master_seed=seed, n=n, D0=list(D0), value_set=[1, 2, 3], u_low=0.35,
               noise_offsets=[list(o) if isinstance(o, (list, tuple)) else o for o in offs], noise_probs=list(probs),
               scales=list(scales))
    inst, Dl, real, inst0 = a7_instance(cfg, rho, s, a, idx, make_instance_k)
    return cfg, inst, Dl, inst0


def truth(planner, lam):
    return planner.models[lam][0], planner.models[lam][2].value


def test_class_dwells_flat_and_nested():
    assert class_dwells((5, 10), 1.0, [-1, 0, 1]) == (4, 5, 6, 9, 10, 11)
    assert class_dwells((5, 10), 0.8, [-1, 0, 1]) == (3, 4, 5, 7, 8, 9)
    assert class_dwells((5, 10), 1.2, [[-1, 0, 1], [-2, 0, 2]]) == (5, 6, 7, 10, 12, 14)
    with pytest.raises(AssertionError):
        class_dwells((5, 10), 0.9, [0])
    with pytest.raises(AssertionError):
        class_dwells((5, 10), 0.6, [-3, 0, 3])


def test_merge_classes_sorts_merges_and_keeps_mass():
    D = (4, 5, 3, 5)
    prior = ((0.1, 0.2, 0.3, 0.4), (0.25, 0.25, 0.25, 0.25))
    Dm, pm = merge_classes(D, prior)
    assert Dm == (3, 4, 5)
    assert pm[0] == pytest.approx((0.3, 0.1, 0.6)) and pm[1] == pytest.approx((0.25, 0.25, 0.5))
    assert all(sum(r) == pytest.approx(1.0) for r in pm)


def test_instance_expansion_probabilities_and_shared_deadlines():
    cfg, inst, Dl, inst0 = build(0, offs=((0, 1), (0, 2)))
    assert all(sum(r) == pytest.approx(1.0) for r in inst.prior)
    assert inst.deadlines == inst0.deadlines and inst.values == inst0.values
    assert Dl[1.0] == (2, 3, 4, 6)
    mi = merged_instance(inst, Dl[0.5])
    assert mi.D == tuple(sorted(set(Dl[0.5])))
    Model2(mi)


def test_iid_truth_mass_and_expected_dwell():
    inst0, _ = make_instance_k(3, 0, 3, 1.1, 2, 0.7, D=(5, 10), value_set=(1, 2, 3))
    ti = iid_truth(inst0, (5, 10), (0.8, 1.0, 1.2), (0.25, 0.5, 0.25))
    assert ti.D == (4, 5, 6, 8, 10, 12)
    for row, r0 in zip(ti.prior, inst0.prior):
        assert sum(row) == pytest.approx(1.0)
        assert sum(p * d for p, d in zip(row, ti.D)) == pytest.approx(5 * r0[0] + 10 * r0[1])


def sim_obs_paths(inst_true, Dl_true_classes, rng_seed, steps=12):
    import random
    rnd = random.Random(rng_seed)
    n = inst_true.n
    K = len(inst_true.D)
    c = [rnd.choices(range(K), weights=inst_true.prior[i])[0] for i in range(n)]
    t, cur = 0, -1
    dwell = [0] * n
    kd = [0] * n
    out = []
    for _ in range(steps):
        alive = [j for j in range(n) if kd[j] == 0]
        if not alive:
            break
        j = rnd.choice(alive)
        sl = inst_true.s if (cur != -1 and cur != j) else 0
        t += sl + 1
        cur = j
        dwell[j] += 1
        if dwell[j] == inst_true.D[c[j]]:
            kd[j] = dwell[j]
        out.append((t, cur, tuple(-kd[i] if kd[i] > 0 else dwell[i] for i in range(n))))
    return out


@pytest.mark.parametrize("weights", [None, (0.2, 0.5, 0.3)])
def test_posterior_matches_joint_enumeration(weights):
    for idx in range(3):
        cfg, inst, Dl, _ = build(idx, offs=((0, 1), (0, 2)))
        pl = Planner(inst, Dl, SCALES)
        kit = Kit(pl, SCALES, weights)
        w0 = weights if weights is not None else (1 / 3,) * 3
        K = len(inst.D)
        ti, _ = truth(pl, 1.0)
        for seed in range(4):
            for (t, cur, obs) in sim_obs_paths(ti, None, seed):
                num = []
                for li, lam in enumerate(SCALES):
                    tot = 0.0
                    for c in itertools.product(range(K), repeat=inst.n):
                        w = w0[li]
                        ok = True
                        for i, k in enumerate(c):
                            D = Dl[lam][k]
                            w *= inst.prior[i][k]
                            if obs[i] < 0 and D != -obs[i]:
                                ok = False
                            if obs[i] >= 0 and D <= obs[i]:
                                ok = False
                        if ok:
                            tot += w
                    num.append(tot)
                s = sum(num)
                if s <= 0:
                    continue
                got, _ = kit.ba.belief(obs)
                assert got == pytest.approx(tuple(x / s for x in num), abs=1e-12)


@pytest.mark.parametrize("weights", [None, (0.125, 0.75, 0.125), (0.1, 0.2, 0.7)])
def test_bayes_adaptive_value_matches_independent_brute_force(weights):
    for idx in range(2):
        cfg, inst, Dl, _ = build(idx, offs=((0, 1), (0, 2)), n=3)
        pl = Planner(inst, Dl, SCALES)
        kit = Kit(pl, SCALES, weights)
        w0 = weights if weights is not None else (1 / 3,) * 3
        v_dp = kit.solve_ba().value
        v_bf = brute_value_ba(None, SCALES, w0, inst.s, inst.values, inst.deadlines, inst.prior, Dl=[Dl[l] for l in SCALES])
        assert v_dp == pytest.approx(v_bf, abs=1e-9)
        avg = sum(w * eval_rec(truth(pl, l)[0], kit.factory("BA")) for w, l in zip(w0, SCALES))
        assert avg == pytest.approx(v_dp, abs=1e-9)


def test_eval_rec_equals_world_enumeration_for_all_planners():
    for idx in range(3):
        cfg, inst, Dl, _ = build(idx)
        pl = Planner(inst, Dl, SCALES)
        kit = Kit(pl, SCALES, (0.2, 0.5, 0.3))
        facs = [nominal_factory(pl, m) for m in ("DPMIS", "CEP", "ROLL_CEP")] + [kit.factory(x) for x in ("BA",) + PRACTICAL]
        for lam in SCALES:
            ti, vc = truth(pl, lam)
            for fac in facs:
                a = eval_rec(ti, fac)
                b = eval_obs(ti, fac)
                assert a == pytest.approx(b, abs=1e-9)
                assert -1e-9 <= a <= vc + 1e-9


def test_single_scale_support_reduces_to_nominal_planners():
    for idx in range(4):
        cfg, inst, Dl, _ = build(idx)
        pl = Planner(inst, Dl, SCALES)
        kit = Kit(pl, (1.0,), None)
        for lam in SCALES:
            ti, _ = truth(pl, lam)
            for name, meth in (("PLUG_DP", "DPMIS"), ("QMDP", "DPMIS"), ("BA", "DPMIS"), ("PLUG_CEP", "CEP"), ("MIXCEP", "CEP")):
                a = eval_rec(ti, kit.factory(name))
                b = eval_rec(ti, nominal_factory(pl, meth, 1.0))
                assert a == pytest.approx(b, abs=1e-9), (idx, lam, name)


def brute_q(inst_lam, t, cur, dwell, kd, j):
    n, D, s, d, v = inst_lam.n, inst_lam.D, inst_lam.s, inst_lam.deadlines, inst_lam.values
    K = len(D)
    dmax = max(d)

    def qk(i, x):
        z = sum(inst_lam.prior[i][k] for k in range(K) if D[k] > x)
        return sum(inst_lam.prior[i][k] for k in range(K) if D[k] == x + 1) / z

    @lru_cache(maxsize=None)
    def V(t, cur, dwell, kd):
        best = 0.0
        if t + 1 <= dmax:
            best = V(t + 1, cur, dwell, kd)
        for jj in range(n):
            if kd[jj] > 0:
                continue
            best = max(best, Q(t, cur, dwell, kd, jj))
        return best

    def Q(t, cur, dwell, kd, jj):
        sl = s if (cur != -1 and cur != jj) else 0
        t2 = t + sl + 1
        if t2 > d[jj]:
            return -1.0
        x = dwell[jj]
        p = qk(jj, x)
        dw = dwell[:jj] + (x + 1,) + dwell[jj + 1:]
        val = 0.0
        if p > 0:
            val += p * (v[jj] + V(t2, jj, dw, kd[:jj] + (x + 1,) + kd[jj + 1:]))
        if p < 1:
            val += (1 - p) * V(t2, jj, dw, kd)
        return val

    return Q(t, cur, dwell, kd, j)


def test_qmdp_q_values_match_independent_expectimax():
    checked = 0
    for idx in range(3):
        cfg, inst, Dl, _ = build(idx)
        pl = Planner(inst, Dl, SCALES)
        kit = Kit(pl, SCALES, (0.3, 0.4, 0.3))
        ti, _ = truth(pl, 1.0)
        for seed in range(3):
            for (t, cur, obs) in sim_obs_paths(ti, None, seed, steps=8):
                po = kit.ba.project(obs)
                w, _ = kit.ba.belief(po)
                acts, q = kit.qvalues(t, cur, po)
                dwell = tuple(0 if o < 0 else o for o in po)
                kd = tuple(-o if o < 0 else 0 for o in po)
                for j in acts:
                    ref = 0.0
                    for li, lam in enumerate(SCALES):
                        if w[li] <= 1e-12:
                            continue
                        iml = pl.models[lam][0]
                        ref += w[li] * brute_q(iml, t, cur, dwell, kd, j)
                    assert q[j] == pytest.approx(ref, abs=1e-9)
                    checked += 1
    assert checked > 20


def test_practical_planners_obey_ceiling_and_calibrated_bounds():
    for idx in range(3):
        cfg, inst, Dl, _ = build(idx)
        pl = Planner(inst, Dl, SCALES)
        w0 = (0.25, 0.5, 0.25)
        kit = Kit(pl, SCALES, w0)
        v_ba = kit.solve_ba().value
        for name in PRACTICAL:
            vals = []
            for lam in SCALES:
                ti, vc = truth(pl, lam)
                v = eval_rec(ti, kit.factory(name))
                assert -1e-9 <= v <= vc + 1e-9
                vals.append(v)
            assert sum(w * v for w, v in zip(w0, vals)) <= v_ba + 1e-9


def test_out_of_support_truth_is_handled_without_failure():
    cfg, inst, Dl, _ = build(1, scales=(0.5, 1.0, 1.5, 2.0), offs=((0, 1), (0, 2)))
    pl = Planner(inst, Dl, (0.5, 1.0, 1.5, 2.0))
    kit = Kit(pl, (0.5, 1.0, 1.5), None)
    ti, vc = truth(pl, 2.0)
    for name in ("BA",) + PRACTICAL:
        v = eval_rec(ti, kit.factory(name))
        assert -1e-9 <= v <= vc + 1e-9
    w, _ = kit.ba.belief((-1, -7, 30))
    assert sum(w) == pytest.approx(1.0) and all(x > 0 for x in w)
    po = kit.ba.project((-1, -9, 30))
    assert po[0] in kit.ba.union[0] or -po[0] in kit.ba.union[0]
    assert po[2] == kit.ba.union[2][-1] - 1


def test_projection_is_identity_for_in_support_observations():
    cfg, inst, Dl, _ = build(2)
    pl = Planner(inst, Dl, SCALES)
    kit = Kit(pl, SCALES, None)
    for lam in SCALES:
        ti, _ = truth(pl, lam)
        for seed in range(4):
            for (t, cur, obs) in sim_obs_paths(ti, None, seed):
                assert kit.ba.project(obs) == obs


def test_nominal_planner_is_exact_at_nominal_truth():
    for idx in range(4):
        cfg, inst, Dl, _ = build(idx, n=4, D0=(5, 10), scales=(0.8, 1.0, 1.2), offs=(-1, 0, 1), probs=(0.25, 0.5, 0.25))
        pl = Planner(inst, Dl, (0.8, 1.0, 1.2))
        ti, vc = truth(pl, 1.0)
        assert eval_rec(ti, nominal_factory(pl, "DPMIS")) == pytest.approx(vc, abs=1e-9)


def indep_posterior(inst, Dl, lams, w0, obs):
    K = len(inst.D)
    num = []
    for li, lam in enumerate(lams):
        tot = 0.0
        for c in itertools.product(range(K), repeat=inst.n):
            w = w0[li]
            ok = True
            for i, k in enumerate(c):
                D = Dl[lam][k]
                w *= inst.prior[i][k]
                if (obs[i] < 0 and D != -obs[i]) or (obs[i] >= 0 and D <= obs[i]):
                    ok = False
            if ok:
                tot += w
        num.append(tot)
    s = sum(num)
    return [x / s for x in num]


def test_lam_hat_is_map_with_nominal_tiebreak():
    cfg, inst, Dl, _ = build(0)
    pl = Planner(inst, Dl, SCALES)
    kit = Kit(pl, SCALES, None)
    assert kit.lam_hat((0.2, 0.5, 0.3)) == 1.0
    assert kit.lam_hat((0.5, 0.0, 0.5)) == 0.5
    assert kit.lam_hat((0.4, 0.4, 0.2)) == 1.0
    assert kit.lam_hat((0.1, 0.2, 0.7)) == 1.5
    assert kit.lam_hat((0.7, 0.2, 0.1)) == 0.5


def test_plug_dp_acts_as_dp_of_map_scale():
    checked = 0
    for idx in range(3):
        cfg, inst, Dl, _ = build(idx)
        w0 = (0.25, 0.5, 0.25)
        pl = Planner(inst, Dl, SCALES)
        kit = Kit(pl, SCALES, w0)
        ti, _ = truth(pl, 1.5)
        for seed in range(4):
            for (t, cur, obs) in sim_obs_paths(ti, None, seed, steps=10):
                po = kit.ba.project(obs)
                if po != obs:
                    continue
                post = indep_posterior(inst, Dl, SCALES, w0, obs)
                top = max(post)
                cands = [l for l, p in zip(SCALES, post) if p >= top - 1e-12]
                lam = min(cands, key=lambda l: (abs(l - 1.0), l))
                im, M, dp = pl.models[lam]
                K = M.K
                prog = [-1 if o < 0 else min(o, M.D[-1] - 1) * K + min(sum(1 for k in range(K) if M.D[k] <= o), K - 1) for o in obs]
                st = (t, cur, M.canon(t, prog))
                want = dp.policy()(st)
                if want is None:
                    continue
                assert kit.act("PLUG_DP", t, cur, po) == want
                checked += 1
    assert checked > 10


def test_mixture_durations_match_independent_computation():
    checked = 0
    for idx in range(3):
        cfg, inst, Dl, _ = build(idx)
        w0 = (0.2, 0.5, 0.3)
        pl = Planner(inst, Dl, SCALES)
        kit = Kit(pl, SCALES, w0)
        ti, _ = truth(pl, 1.0)
        K = len(inst.D)
        for seed in range(4):
            for (t, cur, obs) in sim_obs_paths(ti, None, seed, steps=8):
                post = indep_posterior(inst, Dl, SCALES, w0, obs)
                acts = kit.ba.actions((t, cur, obs))
                dur = kit.durations(t, cur, obs, acts)
                for j in acts:
                    num = den = 0.0
                    for lam, pw in zip(SCALES, post):
                        if pw <= 1e-12:
                            continue
                        x = obs[j]
                        mass = [(inst.prior[j][k], Dl[lam][k] - x) for k in range(K) if Dl[lam][k] > x]
                        z = sum(p for p, _ in mass)
                        rmin = min(r for _, r in mass)
                        if t + rmin > inst.deadlines[j]:
                            continue
                        num += pw * sum(p * r for p, r in mass) / z
                        den += pw
                    if den > 1e-12:
                        assert dur[j] == pytest.approx(num / den, abs=1e-9)
                        checked += 1
    assert checked > 20


def brute_plan_first_actions(acts, dur, cur, t, s, d, v):
    state = {"value": 0.0, "time": 0.0, "firsts": set()}

    def rec(time, last, used, value, first):
        if first is not None:
            if value > state["value"] + 1e-12 or (abs(value - state["value"]) <= 1e-12 and time < state["time"] - 1e-12):
                state["value"], state["time"], state["firsts"] = value, time, {first}
            elif abs(value - state["value"]) <= 1e-12 and abs(time - state["time"]) <= 1e-12:
                state["firsts"].add(first)
        for j in acts:
            if j in used:
                continue
            sl = s if (last != -1 and last != j) else 0
            end = time + sl + dur[j]
            if end <= d[j] + 1e-9:
                rec(end, j, used | {j}, value + v[j], j if first is None else first)

    rec(float(t), cur, frozenset(), 0.0, None)
    return state["firsts"]


def test_mixcep_action_is_optimal_first_move_of_independent_plan_enumeration():
    checked = 0
    for idx in range(4):
        cfg, inst, Dl, _ = build(idx, n=3)
        pl = Planner(inst, Dl, SCALES)
        kit = Kit(pl, SCALES, (0.3, 0.4, 0.3))
        ti, _ = truth(pl, 1.0)
        for seed in range(4):
            for (t, cur, obs) in sim_obs_paths(ti, None, seed, steps=8):
                acts = kit.ba.actions((t, cur, obs))
                if not acts:
                    continue
                dur = kit.durations(t, cur, obs, acts)
                firsts = brute_plan_first_actions(acts, dur, cur, t, inst.s, inst.deadlines, inst.values)
                if firsts:
                    assert kit.act("MIXCEP", t, cur, obs) in firsts
                    checked += 1
    assert checked > 10


def test_plug_dp_tests_are_discriminating_between_scales():
    disc = 0
    for idx in range(6):
        for truth_lam in (0.5, 1.5):
            cfg, inst, Dl, _ = build(idx, rho=0.9, s=1)
            w0 = (0.25, 0.5, 0.25)
            pl = Planner(inst, Dl, SCALES)
            kit = Kit(pl, SCALES, w0)
            ti, _ = truth(pl, truth_lam)
            for seed in range(8):
                for (t, cur, obs) in sim_obs_paths(ti, None, seed, steps=10):
                    po = kit.ba.project(obs)
                    w, _ = kit.ba.belief(po)
                    lam = kit.lam_hat(w)
                    if lam == 1.0:
                        continue
                    st_hat = (t, cur, kit.prog_in(lam, t, po))
                    st_nom = (t, cur, kit.prog_in(1.0, t, po))
                    a_hat = pl.models[lam][2].policy()(st_hat)
                    a_nom = pl.models[1.0][2].policy()(st_nom)
                    if a_hat is not None and a_hat != a_nom:
                        assert kit.act("PLUG_DP", t, cur, po) == a_hat
                        disc += 1
    assert disc > 0


def test_projection_maps_unseen_kill_dwell_to_nearest_support_value():
    cfg, inst, Dl, _ = build(1, scales=(0.5, 1.0, 1.5, 2.0), offs=((0, 1), (0, 2)))
    pl = Planner(inst, Dl, (0.5, 1.0, 1.5, 2.0))
    kit = Kit(pl, (0.5, 1.0, 1.5), None)
    U = kit.ba.union[0]
    for k in range(1, 15):
        po = kit.ba.project((-k, 0, 0))
        want = k if k in U else min(U, key=lambda u: (abs(u - k), u))
        assert po[0] == -want


class StubbornPolicy:
    def __init__(self, order):
        self.order = order

    def __call__(self):
        order = self.order

        def pol(t, cur, dwell, kd):
            return order[(t // 3) % len(order)]
        return pol


def test_eval_rec_matches_world_enumeration_for_policies_that_choose_infeasible_or_dead_targets():
    for idx in range(4):
        cfg, inst, Dl, _ = build(idx)
        pl = Planner(inst, Dl, SCALES)
        for lam in SCALES:
            ti, _ = truth(pl, lam)
            for order in ([0], [1, 0], [2, 2, 1], [0, 1, 2]):
                fac = StubbornPolicy(order)
                assert eval_rec(ti, fac) == pytest.approx(eval_obs(ti, fac), abs=1e-9)
