"""Tests of the A15 learned scorer and pipeline (plan A15.7, items 1 to 6)."""
import math, os, sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from dwell.dp import solve
from dwell.instances import make_instance_k
from dwell.learned import features as base_features
from dwell.learned_g import (A_MAX, CLIP_HI, CLIP_LO, COLS, N_AUG, N_FULL, N_NOACC, Scorer, build_instance_samples, features_g, learned_policy,
                             rollout_states, to_arrays, train_variant)
from dwell.m0 import Model
from dwell.m0u import GoalDP, need_of

CFG = dict(pool_seed=5, train_n=[3, 4], rho_grid=[1.1, 1.5], s_grid=[2, 4], a_grid=[0.5, 0.9], lam_grid=[0.25, 0.5],
           D0=[5, 10], value_set=[1, 2, 3], u_low=0.35, worlds_per_instance=9)


def small_model(i=0, n=4, rho=1.1, s=2, a=0.5, lam=0.25):
    inst, _ = make_instance_k(77, i, n, rho, s, a, D=(5, 10), value_set=(1, 2, 3), u_low=0.35)
    return Model(inst), need_of(inst.values, lam)


def pool(k=6):
    rows = []
    for i in range(k):
        r, _ = build_instance_samples(CFG, i)
        rows += [(i, x) for x in r]
    return to_arrays(rows)


# ---- scorer -------------------------------------------------------------------------------------------------------------
def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    sc = Scorer(6, hidden=4, seed=1)
    Phi, mask, y = rng.normal(size=(7, 3, 6)), rng.random((7, 3)) > 0.2, rng.integers(0, 3, 7)
    mask[:, 0] = True
    y = np.where(mask[np.arange(7), y], y, 0)
    _, g = sc.loss_and_grads(Phi, mask, y)
    for k in sc.p:
        flat = sc.p[k].ravel()
        for idx in range(min(3, flat.size)):
            old = flat[idx]
            flat[idx] = old + 1e-6; lp, _ = sc.loss_and_grads(Phi, mask, y)
            flat[idx] = old - 1e-6; lm, _ = sc.loss_and_grads(Phi, mask, y)
            flat[idx] = old
            assert abs((lp - lm) / 2e-6 - g[k].ravel()[idx]) < 1e-5, (k, idx)


def test_save_load_roundtrip(tmp_path):
    sc = Scorer(N_FULL, seed=3)
    sc.save(str(tmp_path / "m.npz"), loss_last=0.5)
    sc2 = Scorer.load(str(tmp_path / "m.npz"))
    phi = np.random.default_rng(1).normal(size=(4, N_FULL))
    assert np.allclose(sc.scores(phi), sc2.scores(phi)) and sc2.n_feat == N_FULL


# ---- features -----------------------------------------------------------------------------------------------------------
def test_feature_layout_and_values():
    m, need = small_model()
    s0 = m.initial_state()
    acts = m.actions(s0)
    F = features_g(m, s0, acts, max(m.d), 0.0, need)
    assert F.shape == (len(acts), N_AUG)
    assert np.allclose(F[:, :12], np.clip(base_features(m, s0, acts, max(m.d)), CLIP_LO, CLIP_HI))
    W = float(sum(m.v))
    assert np.allclose(F[:, 12], need / W) and np.allclose(F[:, 14], need / W)
    F2 = features_g(m, s0, acts, max(m.d), 2.0, need)
    assert np.allclose(F2[:, 14], (need - 2.0) / W)
    for k, j in enumerate(acts):
        assert F2[k, 16] == (1.0 if m.v[j] >= need - 2.0 - 1e-9 else 0.0)
    assert F2.min() >= -3.0 - 1e-9 and np.all(F2[:, 20] == 0.0)                 # no CEPS action given: flag column is 0


def test_ceps_flag_marks_exactly_the_recommended_action():
    m, need = small_model(3)
    s0 = m.initial_state(); acts = m.actions(s0)
    F = features_g(m, s0, acts, max(m.d), 0.0, need, ceps_act=acts[1])
    assert F[1, 20] == 1.0 and F[:, 20].sum() == 1.0


def test_ia_policy_uses_the_ceps_recommendation():
    from dwell.m0u import ceps_policy
    m, need = small_model(4)
    sc = Scorer(N_AUG, seed=0)
    sc.p["W1"][:] = 0.0; sc.p["W1"][20, :] = 5.0; sc.p["w2"][:] = 1.0        # scores increase with the flag only
    s0 = m.initial_state()
    assert learned_policy(m, sc, need)(s0, 0.0) == ceps_policy(m, need)(s0, 0.0)


def test_noacc_columns_do_not_depend_on_acc_but_the_rest_do():
    m, need = small_model(2)
    s0 = m.initial_state(); acts = m.actions(s0)
    A, B = features_g(m, s0, acts, max(m.d), 0.0, need), features_g(m, s0, acts, max(m.d), 3.0, need)
    assert np.allclose(A[:, :N_NOACC], B[:, :N_NOACC]) and not np.allclose(A[:, N_NOACC:], B[:, N_NOACC:])
    assert COLS == {"IL_G": 20, "IL_EV": 20, "IL_G_noacc": 14, "IL_A": 21, "IL_AD": 21} and N_NOACC == 14


def test_policy_stops_at_the_goal_and_follows_the_argmax():
    m, need = small_model()
    sc = Scorer(N_FULL, seed=4)
    pol = learned_policy(m, sc, need)
    s0 = m.initial_state()
    assert pol(s0, float(need)) is None
    acts = m.actions(s0)
    sc_val = sc.scores(features_g(m, s0, acts, max(m.d), 0.0, need)[:, :N_FULL])
    assert pol(s0, 0.0) == acts[int(np.argmax(sc_val))]


# ---- teacher data -------------------------------------------------------------------------------------------------------
def test_pool_is_deterministic():
    a, b = build_instance_samples(CFG, 3)[0], build_instance_samples(CFG, 3)[0]
    assert len(a) == len(b) and all(np.array_equal(x[0], y[0]) and x[1:] == y[1:] for x, y in zip(a, b))


@pytest.mark.parametrize("i", range(4))
def test_goal_labels_are_optimal_and_ev_labels_are_the_ev_optimum(i):
    rows, meta = build_instance_samples(CFG, i)
    m = meta["model"]
    if not rows:
        return
    need = meta["need"]
    g, ev = GoalDP(m, need), solve(m).policy()
    for (phi, na, lg, le), (st, ac) in zip(rows, meta["states"]):
        acts = m.actions(st)
        assert na == len(acts)
        q = {j: sum(p * g.V(nxt, ac + r) for p, r, nxt in m.outcomes(st, j)) for j in acts}
        if lg >= 0:
            assert abs(q[acts[lg]] - max(q.values())) < 1e-9 and max(q.values()) > 1e-9      # a label only where some action can still help
        else:
            assert max(q.values()) <= 1e-9 or g.policy()(st, ac) is None      # no label only when no action can help
        assert (acts[le] if le >= 0 else None) == ev(st)


def test_rollout_follows_the_teacher_when_epsilon_is_zero_and_is_random_when_one():
    m, need = small_model(1)
    gp = GoalDP(m, need).policy()
    c = next(iter(m.worlds()))[0]
    tr = rollout_states(m, c, gp, need, np.random.default_rng(0), 0.0)
    assert tr[0][0] == m.initial_state() and all(tr[k][1] <= tr[k + 1][1] for k in range(len(tr) - 1))
    for (st, ac), (st2, ac2) in zip(tr, tr[1:]):                              # the next state is the teacher's action applied
        j = gp(st, ac)
        assert st2[1] == j
    tr1 = rollout_states(m, c, gp, need, np.random.default_rng(0), 1.0)
    assert all(st[1] in range(-1, len(m.v)) for st, _ in tr1)


def test_label_choice_follows_the_variant_and_subsample_is_a_fraction():
    rng = np.random.default_rng(2)
    S = 400
    Phi = rng.normal(0, 0.3, size=(S, A_MAX, N_AUG)); mask = np.ones((S, A_MAX), dtype=bool)
    Phi[:, 0, 0] = 3.0; Phi[:, 1, 1] = 3.0                                      # action 0 shows a signal in column 0, action 1 in column 1
    yg, ye, iid = np.zeros(S, dtype=int), np.ones(S, dtype=int), np.repeat(np.arange(20), 20)
    P = (Phi, mask, yg, ye, iid)
    scg, _, _, ng = train_variant("IL_G", 0, P, epochs=25)
    sce, _, _, ne = train_variant("IL_EV", 0, P, epochs=25)
    assert ng == ne == 14 * 20                                                # 70% of 20 instances = 14
    assert scg.accuracy(Phi[:, :, :N_FULL], mask, yg) > 0.9 and sce.accuracy(Phi[:, :, :N_FULL], mask, ye) > 0.9
    assert train_variant("IL_G_noacc", 0, P, epochs=2)[0].n_feat == 14


def test_training_is_reproducible_and_learns_the_teacher():
    P = pool(8)
    a = train_variant("IL_G", 1, P, epochs=25)
    b = train_variant("IL_G", 1, P, epochs=25)
    assert all(np.array_equal(a[0].p[k], b[0].p[k]) for k in a[0].p)
    assert a[1][-1] < a[1][0] and a[2] > 0.7


# ---- pipeline rows --------------------------------------------------------------------------------------------------------
def test_eval_row_invariants_and_planner_sets():
    import run_a15
    cfg = dict(eval_seed=9, D0=[5, 10], value_set=[1, 2, 3], u_low=0.35)
    P = pool(6)
    models = {(v, 0): train_variant(v, 0, P, epochs=8)[0] for v in ("IL_G", "IL_EV", "IL_G_noacc", "IL_A", "IL_AD")}
    r5 = run_a15.eval_instance(cfg, models, 4, 1.5, 4, 0.5, 0.25, 0)
    assert r5["violations"] == "" and "U_DPEV" in r5 and "U_RU_DPEV" in r5 and r5["U_opt"] >= r5["U_RU_CEP"] - 1e-9
    r7 = run_a15.eval_instance(cfg, models, 6, 1.5, 4, 0.5, 0.25, 0)
    assert "U_DPEV" not in r7 and math.isnan(r7["U_opt"])
    assert "U_RU_CEP" in r7 and "U_IL_G_s0" in r7
