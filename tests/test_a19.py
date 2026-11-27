"""Tests of A19 (plan A19.6): the scale-learning planner and its evaluation (src/dwell/a19.py), the analysis and the job layout."""
import dataclasses, json, os, sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.setrecursionlimit(200000)
from dwell import a18
from dwell.a18 import scaled_dwell
from dwell.a19 import Learner, SUPPORT, eval_row, prob_goal_obs
from dwell.analysis_a19 import (SCALES, SIZES, boot_ratio, claims19, pilot_rule, pooled_recovery, recovery_class, regret_table, stats)
from dwell.instances import make_instance_k
from dwell.m0 import DEAD, Instance, Model
from dwell.m0u import GoalDP, ceps_policy, need_of, prob_goal, ru_policy

CFG = dict(D0=[5, 10], value_set=[1, 2, 3], u_low=0.35, eval_seed=3)


def toy(deadline=20, D=(5, 10)):
    return Instance(D=D, s=0, values=(1.0,), deadlines=(deadline,), prior=((0.5, 0.5),))


# ---- belief ------------------------------------------------------------------------------------------------------------------
def test_posterior_from_alive_threats():
    L = Learner(toy(), 1, SUPPORT)
    w = L.posterior((0, -1, (0,), (False,)))
    assert all(abs(v - 0.25) < 1e-12 for v in w.values())
    w = L.posterior((4, 0, (4,), (False,)))                    # alive after 4 steps: scale 0.8 needs class 1 (prior 0.5), the others are untouched
    assert abs(w[0.8] / w[1.0] - 0.5) < 1e-12 and abs(w[1.2] - w[1.0]) < 1e-12
    w = L.posterior((5, 0, (5,), (False,)))                    # alive after 5 steps: scale 1.0 also needs class 1
    assert abs(w[0.8] - w[1.0]) < 1e-12 and abs(w[1.2] / w[1.0] - 2.0) < 1e-12


@pytest.mark.parametrize("x,scale", [(4, 0.8), (5, 1.0), (6, 1.2), (7, 1.4), (8, 0.8), (10, 1.0), (12, 1.2), (14, 1.4)])
def test_a_kill_shows_the_scale(x, scale):
    L = Learner(toy(), 1, SUPPORT)
    w = L.posterior((x, 0, (x,), (True,)))
    assert abs(w[scale] - 1.0) < 1e-12 and sum(w.values()) == pytest.approx(1.0)


def test_impossible_observation_falls_back_to_the_prior():
    L = Learner(toy(), 1, (1.0, 1.2))
    w = L.posterior((3, 0, (3,), (True,)))                     # no scale has a class with dwell 3
    assert w == {1.0: 0.5, 1.2: 0.5}


def test_posterior_never_excludes_the_truth_and_rows_are_clean():
    for sc in (0.8, 1.0, 1.2, 1.4):
        for idx in range(4):
            inst, _ = make_instance_k(5, idx, 5, 1.1, 2, 0.9, D=(5, 10), value_set=(1, 2, 3), u_low=0.35)
            need = need_of(inst.values, 0.25)
            L = Learner(inst, need, SUPPORT)

            def wrapped(obs, acc, L=L, sc=sc):
                assert L.posterior(obs)[sc] > 0.0
                return L.adapt(obs, acc)
            p, late = prob_goal_obs(inst, scaled_dwell((5, 10), sc), wrapped, need)
            assert -1e-9 <= p <= 1.0 + 1e-9 and late == 0


# ---- planners ----------------------------------------------------------------------------------------------------------------
def test_hand_case_nominal_abandons_clamp_continues_adapt_learns():
    inst = toy(deadline=20, D=(2, 4))                          # true dwell (4, 8): twice the nominal one
    L = Learner(inst, 1, (1.0, 2.0))
    Dt = (4, 8)
    assert abs(prob_goal_obs(inst, Dt, L.nom, 1)[0] - 0.5) < 1e-12        # class 1 survives the largest nominal dwell and is abandoned
    assert abs(prob_goal_obs(inst, Dt, L.clamp, 1)[0] - 1.0) < 1e-12      # never abandons
    assert abs(prob_goal_obs(inst, Dt, L.adapt, 1)[0] - 1.0) < 1e-12
    assert abs(prob_goal_obs(inst, Dt, L.adapt_v, 1)[0] - 1.0) < 1e-12


def test_singleton_support_equals_the_informed_planners():
    for sc in (0.8, 1.2, 1.4):
        for idx in range(3):
            inst, _ = make_instance_k(9, idx, 5, 1.1, 2, 0.5, D=(5, 10), value_set=(1, 2, 3), u_low=0.35)
            need = need_of(inst.values, 0.25)
            mt = Model(dataclasses.replace(inst, D=scaled_dwell((5, 10), sc)))
            ct = ceps_policy(mt, need)
            L = Learner(inst, need, (sc,))
            Dt = scaled_dwell((5, 10), sc)
            assert abs(prob_goal_obs(inst, Dt, L.adapt, need)[0] - prob_goal(mt, ru_policy(mt, ct, need), need)) < 1e-12
            assert abs(prob_goal_obs(inst, Dt, L.adapt_v, need)[0] - prob_goal(mt, ct, need)) < 1e-12


class _Stub:
    """A scale model with a fixed action set and deterministic one-step outcomes (used to test the weighting of the planners)."""
    def __init__(self, s):
        self.s = s

    def actions(self, st):
        return [0, 1]

    def outcomes(self, st, a):
        return [(1.0, 0.0, (self.s, a))]


def stubbed(support, weights, table=None, votes=None):
    L = Learner(toy(), 1, support)
    L.posterior = lambda obs: dict(zip(support, weights))
    L.state_in = lambda s, obs: ("st", s)
    L.mod = {s: _Stub(s) for s in support}
    if table is not None:
        L._bval = lambda s, nxt, acc: table[nxt]
    if votes is not None:
        L.ceps = {s: (lambda st, acc, s=s: votes[s]) for s in support}
    return L


def test_adapt_weights_the_scales_by_the_posterior():
    table = {(0.8, 0): 0.6, (0.8, 1): 0.4, (1.2, 0): 0.0, (1.2, 1): 1.0}
    assert stubbed((0.8, 1.2), (0.9, 0.1), table).adapt((0, -1, (0,), (False,)), 0.0) == 0      # 0.54 against 0.46
    assert stubbed((0.8, 1.2), (0.1, 0.9), table).adapt((0, -1, (0,), (False,)), 0.0) == 1      # 0.06 against 0.94


def test_adapt_v_votes_with_the_posterior_and_keeps_the_first_of_a_tie():
    sup = (0.8, 1.0, 1.2)
    votes = {0.8: 0, 1.0: 1, 1.2: 1}
    assert stubbed(sup, (0.6, 0.2, 0.2), votes=votes).adapt_v((0, -1, (0,), (False,)), 0.0) == 0   # 0.6 against 0.4; unweighted it would be 1 against 2
    assert stubbed(sup, (0.2, 0.4, 0.4), votes=votes).adapt_v((0, -1, (0,), (False,)), 0.0) == 1
    assert stubbed((0.8, 1.2), (0.5, 0.5), votes={0.8: 0, 1.2: 1}).adapt_v((0, -1, (0,), (False,)), 0.0) == 0


def test_the_support_of_the_config_reaches_the_learner(monkeypatch):
    import dwell.a19 as m
    seen = []
    orig = m.Learner

    def spy(inst, need, support=m.SUPPORT):
        seen.append(tuple(support))
        return orig(inst, need, support)
    monkeypatch.setattr(m, "Learner", spy)
    m.eval_row(dict(CFG, support=[1.2]), 5, 1.1, 2, 0.5, 0.25, 1.2, 0, "ru")
    m.eval_row(CFG, 5, 1.1, 2, 0.5, 0.25, 1.2, 0, "ru")
    assert seen == [(1.2,), SUPPORT]


def test_nominal_planner_reproduces_a18():
    for sc in (0.8, 1.2, 1.4):
        for idx in range(3):
            r19 = eval_row(CFG, 5, 1.5, 2, 0.5, 0.25, sc, idx, "ru")
            r18 = a18.eval_row(CFG, 5, 1.5, 2, 0.5, 0.25, sc, idx, "ru")
            assert abs(r19["U_NOM"] - r18["U_CEPS"]) < 1e-12 and abs(r19["U_CEPS_T"] - r18["U_CEPS_T"]) < 1e-12


def uncached(inst, Dt, policy, need):
    mb = Model(inst)
    total = 0.0
    for c, w in mb.worlds():
        t, cur, acc, x, kl = 0, -1, 0.0, [0] * mb.n, [False] * mb.n
        while acc < need - 1e-9:
            j = policy((t, cur, tuple(x), tuple(kl)), acc)
            if j is None:
                break
            t2 = t + (mb.s if (cur != -1 and cur != j) else 0) + 1
            x[j] += 1
            if x[j] == Dt[c[j]]:
                kl[j] = True
                acc += mb.v[j] if t2 <= mb.d[j] else 0.0
            t, cur = t2, j
        total += w * float(acc >= need - 1e-9)
    return total


def test_cache_does_not_change_the_probability():
    for sc in (0.8, 1.2):
        inst, _ = make_instance_k(13, 1, 5, 1.1, 4, 0.9, D=(5, 10), value_set=(1, 2, 3), u_low=0.35)
        need = need_of(inst.values, 0.5)
        L = Learner(inst, need, SUPPORT)
        Dt = scaled_dwell((5, 10), sc)
        for pol in (L.nom, L.clamp, L.adapt_v):
            assert abs(prob_goal_obs(inst, Dt, pol, need)[0] - uncached(inst, Dt, pol, need)) < 1e-12


# ---- rows --------------------------------------------------------------------------------------------------------------------
def test_rows_have_no_violation_and_learning_is_used():
    diff = 0
    for sc in (1.0, 0.8, 1.2, 1.4):
        for idx in range(4):
            r = eval_row(CFG, 5, 1.5, 2, 0.5, 0.25, sc, idx, "exact")
            assert r["violations"] == ""
            assert all(r["U_" + m] <= r["U_opt_T"] + 1e-9 for m in ("NOM", "CLAMP", "ADAPT", "ADAPT_V", "CEPS_T", "RU_CEP_T"))
            diff += abs(r["U_ADAPT"] - r["U_NOM"]) > 1e-9 and sc != 1.0
    assert diff > 0


def test_support_in_the_config_is_used_and_levels():
    cfg = dict(CFG, support=[1.2])
    for idx in range(3):
        r = eval_row(cfg, 5, 1.1, 2, 0.5, 0.25, 1.2, idx, "exact")
        assert abs(r["U_ADAPT"] - r["U_RU_CEP_T"]) < 1e-12 and abs(r["U_ADAPT_V"] - r["U_CEPS_T"]) < 1e-12
    ru = eval_row(CFG, 7, 1.1, 2, 0.5, 0.25, 1.2, 0, "ru")
    ex = eval_row(CFG, 5, 1.1, 2, 0.5, 0.25, 1.2, 0, "exact")
    assert "U_opt_T" not in ru and "U_opt_T" in ex and "U_ADAPT" in ru
    a, b = eval_row(CFG, 5, 1.5, 4, 0.9, 0.5, 0.8, 3, "exact"), eval_row(CFG, 5, 1.5, 4, 0.9, 0.5, 1.4, 3, "exact")
    assert (a["need"], a["W"], a["d_max"]) == (b["need"], b["W"], b["d_max"])
    r = eval_row(CFG, 5, 1.5, 4, 0.9, 0.5, 1.0, 3, "exact")
    assert abs(r["U_NOM"] - r["U_CEPS_T"]) < 1e-9


def test_informed_references_on_the_true_model():
    for sc in (0.8, 1.2, 1.4):
        for idx in range(2):
            r = eval_row(CFG, 5, 1.5, 2, 0.5, 0.25, sc, idx, "exact")
            inst, _ = make_instance_k(3, idx, 5, 1.5, 2, 0.5, D=(5, 10), value_set=(1, 2, 3), u_low=0.35)
            mt = Model(dataclasses.replace(inst, D=scaled_dwell((5, 10), sc)))
            need = need_of(inst.values, 0.25)
            ct = ceps_policy(mt, need)
            assert abs(r["U_CEPS_T"] - prob_goal(mt, ct, need)) < 1e-12
            assert abs(r["U_RU_CEP_T"] - prob_goal(mt, ru_policy(mt, ct, need), need)) < 1e-12
            assert abs(r["U_opt_T"] - GoalDP(mt, need).value) < 1e-12


# ---- analysis ----------------------------------------------------------------------------------------------------------------
def cell(n, scale, gain, rho=1.1, rows=40, clamp_gain=0.05, ceps_t=0.5, ru_t=0.5, base=0.3):
    g = np.asarray(gain, dtype=float) if np.ndim(gain) else np.full(rows, float(gain))
    m = len(g)
    return pd.DataFrame(dict(n=n, scale=scale, rho=rho, s=2, a=0.5, lam=0.25, U_NOM=np.full(m, base), U_ADAPT=np.full(m, base) + g,
                             U_CLAMP=np.full(m, base) + clamp_gain, U_ADAPT_V=np.full(m, base), U_CEPS_T=np.full(m, ceps_t), U_RU_CEP_T=np.full(m, ru_t),
                             U_opt_T=np.full(m, 0.8)))


def frame(gain=0.15, ctl_gain=-0.01, weak=(), sizes=SIZES, clamp_gain=0.05, **kw):
    parts = []
    for sc in (1.0,) + SCALES:
        for n in sizes:
            g = ctl_gain if sc == 1.0 else (0.0 if (sc, n) in weak else gain)
            for rho in (1.1, 1.5):
                parts.append(cell(n, sc, g, rho=rho, clamp_gain=clamp_gain, **kw))
    return pd.concat(parts)


def test_labels():
    assert claims19(frame())["label"] == "WORKS"
    assert claims19(frame(ctl_gain=-0.10))["label"] == "LIMITED"                                  # P2 fails: learning costs too much with the right model
    assert claims19(frame(weak=[(0.8, 5)]))["label"] == "HELPS"                                    # P1 in 5 of 6 although the pooled recovery bound is above 0.5
    assert claims19(frame(weak=[(0.8, 5), (1.4, 7)]))["label"] == "HELPS"                         # P1 in 4 of 6
    assert claims19(frame(gain=0.05))["label"] == "HELPS"                                         # P1 in 6 of 6 but the pooled recovery bound is only 0.25
    assert claims19(frame(weak=[(0.8, 5), (1.4, 7), (1.2, 5)]))["label"] == "LIMITED"             # P1 in 3 of 6
    assert claims19(frame(gain=0.0))["label"] == "NO GAIN"
    d = frame()
    assert claims19(d[~((d.scale == 1.2) & (d.n == 7))])["label"] == "NO OUTCOME"
    assert claims19(d[~((d.scale == 1.0) & (d.n == 5))])["label"] == "NO OUTCOME"                 # P2 statement without data


def test_clamp_class_and_control_outside_p1():
    assert claims19(frame(clamp_gain=0.05))["clamp_class"] == "LEARNING-ADDS"                    # ADAPT - CLAMP = +0.10
    assert claims19(frame(clamp_gain=0.15))["clamp_class"] == "CLAMP-COMPARABLE"
    assert claims19(frame(ctl_gain=0.30))["label"] == "WORKS" and claims19(frame(ctl_gain=0.30))["n_p1"] == 6
    r = claims19(frame(sizes=(5,)), sizes=(5,))
    assert r["label"] == "WORKS" and r["n_p1"] == 3 and r["n_p2"] == 1


def test_clamp_class_counts_the_statements():
    def with_s1_failing(keys):
        d = frame(clamp_gain=0.05)
        for sc, n in keys:
            m = (d.scale == sc) & (d.n == n)
            d.loc[m, "U_CLAMP"] = d.loc[m, "U_ADAPT"]                                             # ADAPT - CLAMP = 0 there
        return claims19(d)["clamp_class"]
    assert with_s1_failing([(0.8, 5)]) == "LEARNING-ADDS"                                         # S1 in 5 of 6
    assert with_s1_failing([(0.8, 5), (1.2, 5)]) == "CLAMP-COMPARABLE"                            # 4 of 6
    assert with_s1_failing([(0.8, 5), (1.2, 5), (1.4, 5), (0.8, 7), (1.2, 7)]) == "CLAMP-COMPARABLE"   # 1 of 6


def test_bounds_not_point_estimates():
    one = cell(7, 1.2, np.r_[np.zeros(9), 0.5], rows=10)                                          # mean gain 0.05 from one instance in ten
    assert stats(one)["gain"] > 0.03 and not stats(one)["p1"]
    d = cell(7, 1.0, 0.0, rows=40)
    d["U_ADAPT"] = 0.3 - np.r_[np.full(20, 0.3), np.full(20, -0.26)]                              # loss = U_NOM - U_ADAPT has mean 0.02 and a wide spread
    s = stats(d)
    assert s["loss"] <= 0.03 and not s["p2"]
    w = cell(7, 1.2, 0.0, rows=40, clamp_gain=0.0)
    w["U_CLAMP"] = 0.3 - np.r_[np.full(20, 0.3), np.full(20, -0.3 + 0.04)]                        # ADAPT - CLAMP: mean +0.02..., wide
    assert not stats(w)["s1"]
    assert stats(cell(7, 1.2, 0.15))["p1"] and stats(cell(7, 1.0, -0.01))["p2"] and stats(cell(7, 1.2, 0.15, clamp_gain=0.0))["s1"]


def test_informed_reference_selects_cells():
    assert stats(cell(7, 1.2, 0.15, ceps_t=0.995))["undefined"] and stats(cell(7, 1.2, 0.15, ceps_t=0.05))["undefined"]
    d = cell(7, 1.2, 0.15)
    d["U_ADAPT"] = 0.995                                                                         # the planner under test being saturated does not remove the cell
    assert not stats(d)["undefined"]


def test_recovery_ratio_pairs_the_resampled_instances():
    rng = np.random.default_rng(1)
    den = [rng.normal(0.2, 0.1, 30), rng.normal(0.3, 0.1, 30)]
    num = [0.5 * x for x in den]
    p, lo, hi = boot_ratio(num, den, B=200)
    assert abs(p - 0.5) < 1e-12 and abs(lo - 0.5) < 1e-9 and abs(hi - 0.5) < 1e-9
    assert recovery_class(0.9, 0.6, 0.2) == "HIGH" and recovery_class(0.9, 0.3, 0.2) == "SOME" and recovery_class(0.9, 0.1, 0.2) == "LOW"
    assert recovery_class(0.9, 0.6, 0.005) == "UNDEFINED"
    assert recovery_class(0.6, 0.1, 0.2) == "LOW" and recovery_class(0.3, 0.6, 0.2) == "HIGH"       # the class follows the bound, not the point estimate


def test_pooled_recovery_and_stats_fields():
    p = pooled_recovery(frame(gain=0.15))
    assert abs(p["recovery"] - 0.75) < 1e-9 and p["recovery_class"] == "HIGH"
    s = stats(cell(7, 1.2, 0.10))
    assert abs(s["recovery"] - 0.5) < 1e-9 and abs(s["gap"] - 0.2) < 1e-9


def test_pilot_rule():
    f = frame(sizes=(5,))
    assert pilot_rule(f)["decision"] == "GO"
    assert pilot_rule(frame(sizes=(5,), gain=0.02))["decision"] == "STOP"                        # recovery 0.10
    assert pilot_rule(frame(sizes=(5,), ctl_gain=-0.08))["decision"] == "STOP"                   # loss 0.08 at the control
    assert pilot_rule(frame(sizes=(5,)).query("scale != 1.0"))["decision"] == "NO OUTCOME"
    assert pilot_rule(frame(sizes=(5,), gain=0.045))["decision"] == "GO"                         # recovery 0.225


def test_regret_table_is_against_the_true_optimum_and_in_points():
    d = frame()
    d = d[(d.scale == 1.2) & (d.n == 5)]
    t = regret_table(d)
    assert abs(t["NOM"] - 50.0) < 1e-9 and abs(t["ADAPT"] - 35.0) < 1e-9 and abs(t["RU_CEP_T"] - 30.0) < 1e-9
    assert regret_table(frame().drop(columns=["U_opt_T"])) == {}


# ---- job layout and gate -----------------------------------------------------------------------------------------------------
def test_job_layout():
    import run_a19
    root = os.path.join(os.path.dirname(__file__), "..", "configs")
    main, pilot = (json.load(open(os.path.join(root, f"a19_{k}.json"))) for k in ("main", "pilot"))
    J, P = run_a19.jobs_of(main), run_a19.jobs_of(pilot)
    assert len(J) == 128 and sum(j[7] for j in J) == 7680 and {j[1] for j in J} == {5, 7}
    assert len(P) == 64 and sum(j[7] for j in P) == 1280 and {j[1] for j in P} == {5}
    assert main["eval_seed"] == run_a19.MAIN_SEED == 20261301 and pilot["eval_seed"] == run_a19.PILOT_SEED == 20261302
    assert main["support"] == list(SUPPORT) == pilot["support"] and set(main["scales"]) == set(SUPPORT)
    assert main["D0"] == [5, 10]


def test_gate_and_seed_rules(tmp_path, monkeypatch):
    import run_a19
    monkeypatch.setattr(run_a19, "ROOT", str(tmp_path))
    for role, seed in (("main", 20261301), ("pilot", 20261302)):
        with pytest.raises(SystemExit):
            run_a19.check_gate(dict(role=role, eval_seed=seed))                                  # no gate file
    run_a19.check_gate(dict(role="audit", eval_seed=3))
    for seed in (20261301, 20261302):
        with pytest.raises(SystemExit):
            run_a19.check_gate(dict(role="audit", eval_seed=seed))
    monkeypatch.setattr(run_a19, "code_hash", lambda: "x")
    os.makedirs(tmp_path / "results")
    json.dump({"code_hash": "x"}, open(tmp_path / "results" / "a19_gate.json", "w"))
    run_a19.check_gate(dict(role="main", eval_seed=20261301))
    run_a19.check_gate(dict(role="pilot", eval_seed=20261302))
    with pytest.raises(SystemExit):
        run_a19.check_gate(dict(role="main", eval_seed=20261302))
    with pytest.raises(SystemExit):
        run_a19.check_gate(dict(role="pilot", eval_seed=20261301))
    with pytest.raises(SystemExit):
        run_a19.check_gate(dict(role="other", eval_seed=20261301))
    json.dump({"code_hash": "y"}, open(tmp_path / "results" / "a19_gate.json", "w"))
    with pytest.raises(SystemExit):
        run_a19.check_gate(dict(role="main", eval_seed=20261301))
