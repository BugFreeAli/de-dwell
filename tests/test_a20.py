"""Tests of A20 (plan A20.6): RU_CLAMP, the learner outside its prior, timing columns (src/dwell/a20.py), the analysis and the job layout."""
import dataclasses, json, os, sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.setrecursionlimit(200000)
from dwell import a19, a20
from dwell.a18 import scaled_dwell
from dwell.a19 import Learner, SUPPORT, prob_goal_obs
from dwell.a20 import Learner20, eval_row
from dwell.analysis_a20 import CONTROL, IN_SUPPORT, OUT, SIZES, claims20, regret_table, stats, timing_table
from dwell.instances import make_instance_k
from dwell.m0 import Instance, Model
from dwell.m0u import GoalDP, ceps_policy, need_of, prob_goal, ru_policy

CONDS = {"1.0": [5, 10], "0.8": [4, 8], "1.2": [6, 12], "1.4": [7, 14], "0.6": [3, 6], "1.6": [8, 16], "SP": [4, 12], "CP": [6, 8]}
CFG = dict(D0=[5, 10], value_set=[1, 2, 3], u_low=0.35, eval_seed=3, conditions=CONDS)


def toy(deadline=20, D=(5, 10)):
    return Instance(D=D, s=0, values=(1.0,), deadlines=(deadline,), prior=((0.5, 0.5),))


# ---- planners ----------------------------------------------------------------------------------------------------------------
def test_ru_clamp_hand_case_and_cache():
    inst = toy(D=(2, 4))
    L = Learner20(inst, 1, (1.0, 2.0))
    assert abs(prob_goal_obs(inst, (4, 8), L.nom, 1)[0] - 0.5) < 1e-12
    assert abs(prob_goal_obs(inst, (4, 8), L.ru_clamp, 1)[0] - 1.0) < 1e-12              # never abandons, so class 1 is killed at x = 8
    assert abs(prob_goal_obs(inst, (4, 8), L.clamp, 1)[0] - 1.0) < 1e-12


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


def test_ru_clamp_cache_does_not_change_the_probability():
    for Dt in ((6, 12), (3, 6), (4, 12)):
        inst, _ = make_instance_k(13, 1, 5, 1.1, 4, 0.9, D=(5, 10), value_set=(1, 2, 3), u_low=0.35)
        need = need_of(inst.values, 0.5)
        L = Learner20(inst, need, SUPPORT)
        assert abs(prob_goal_obs(inst, Dt, L.ru_clamp, need)[0] - uncached(inst, Dt, L.ru_clamp, need)) < 1e-12


def test_the_learner_survives_an_observation_outside_its_prior():
    inst = toy()
    obs = (30, 0, (13,), (False,))                              # alive after 13 steps: no scale of this prior (largest dwell 10) allows it
    L = Learner20(inst, 1, (0.8, 1.0))
    assert L.posterior(obs) == {0.8: 0.5, 1.0: 0.5}              # no consistent scale: the prior
    assert L.adapt(obs, 0.0) in (None, 0)                        # no IndexError
    assert L.adapt_v(obs, 0.0) in (None, 0)
    with pytest.raises(IndexError):
        Learner(inst, 1, (0.8, 1.0)).adapt(obs, 0.0)             # the A19 learner failed here (found in the exploratory check of A20)


def test_learner20_equals_the_a19_learner_for_a_true_scale_in_the_prior():
    cfg19 = dict(D0=[5, 10], value_set=[1, 2, 3], u_low=0.35, eval_seed=3)
    for sc, cond in ((0.8, "0.8"), (1.2, "1.2"), (1.4, "1.4")):
        for idx in range(3):
            r19 = a19.eval_row(cfg19, 5, 1.5, 2, 0.5, 0.25, sc, idx, "exact")
            r20 = eval_row(CFG, 5, 1.5, 2, 0.5, 0.25, cond, idx, "exact")
            for m in ("NOM", "CLAMP", "ADAPT", "ADAPT_V"):
                assert abs(r19["U_" + m] - r20["U_" + m]) < 1e-12
            assert abs(r19["U_CEPS_T"] - r20["U_CEPS_T"]) < 1e-12 and abs(r19["U_RU_CEP_T"] - r20["U_RU_CEP_T"]) < 1e-12


# ---- rows --------------------------------------------------------------------------------------------------------------------
def test_condition_one_reproduces_the_informed_values_and_rows_are_clean():
    for idx in range(3):
        r = eval_row(CFG, 5, 1.1, 2, 0.5, 0.25, "1.0", idx, "exact")
        assert r["violations"] == "" and abs(r["U_NOM"] - r["U_CEPS_T"]) < 1e-12 and abs(r["U_RU_CLAMP"] - r["U_RU_CEP_T"]) < 1e-12


def test_all_conditions_run_without_violation_and_below_the_optimum():
    for cond in CONDS:
        for idx in range(3):
            r = eval_row(CFG, 5, 1.1, 4, 0.9, 0.5, cond, idx, "exact")
            assert r["violations"] == ""
            assert all(r["U_" + m] <= r["U_opt_T"] + 1e-9 for m in ("NOM", "CLAMP", "RU_CLAMP", "ADAPT", "ADAPT_V", "CEPS_T", "RU_CEP_T"))
            assert r["D_true"] == str(tuple(CONDS[cond]))


def test_the_condition_sets_the_true_dwell_and_instances_are_common():
    a, b = eval_row(CFG, 5, 1.5, 2, 0.5, 0.25, "0.6", 1, "exact"), eval_row(CFG, 5, 1.5, 2, 0.5, 0.25, "SP", 1, "exact")
    assert (a["need"], a["W"], a["d_max"]) == (b["need"], b["W"], b["d_max"]) and a["D_true"] != b["D_true"]
    diff = sum(abs(eval_row(CFG, 5, 1.5, 2, 0.5, 0.25, "1.2", i, "exact")["U_NOM"] - eval_row(CFG, 5, 1.5, 2, 0.5, 0.25, "1.0", i, "exact")["U_NOM"]) > 1e-9 for i in range(5))
    assert diff > 0


def test_informed_references_on_the_true_dwell():
    for cond in ("0.6", "SP", "CP", "1.6"):
        r = eval_row(CFG, 5, 1.5, 2, 0.5, 0.25, cond, 0, "exact")
        inst, _ = make_instance_k(3, 0, 5, 1.5, 2, 0.5, D=(5, 10), value_set=(1, 2, 3), u_low=0.35)
        mt = Model(dataclasses.replace(inst, D=tuple(CONDS[cond])))
        need = need_of(inst.values, 0.25)
        ct = ceps_policy(mt, need)
        assert abs(r["U_CEPS_T"] - prob_goal(mt, ct, need)) < 1e-12
        assert abs(r["U_RU_CEP_T"] - prob_goal(mt, ru_policy(mt, ct, need), need)) < 1e-12
        assert abs(r["U_opt_T"] - GoalDP(mt, need).value) < 1e-12


def test_timing_columns_and_levels_and_support():
    r = eval_row(CFG, 7, 1.1, 2, 0.5, 0.25, "1.2", 0, "ru")
    for m in ("NOM", "CLAMP", "RU_CLAMP", "ADAPT", "ADAPT_V"):
        assert r["ms_" + m] > 0 and r["calls_" + m] >= 1
    assert "U_opt_T" not in r and "U_RU_CLAMP" in r
    d = pd.DataFrame([r, r])
    t = timing_table(d)
    assert set(t["method"]) == {"NOM", "CLAMP", "RU_CLAMP", "ADAPT", "ADAPT_V"} and (t["n"] == 7).all()
    assert abs(float(t[t.method == "ADAPT"]["ms_per_call"].iloc[0]) - r["ms_ADAPT"]) < 1e-12
    for m in ("NOM", "ADAPT", "RU_CLAMP"):
        assert r["ms_" + m] * r["calls_" + m] / 1000.0 <= r["sec"]                          # the time of the calls cannot exceed the time of the row


def test_the_support_of_the_config_reaches_the_learner(monkeypatch):
    seen = []
    orig = a20.Learner20

    def spy(inst, need, support=SUPPORT):
        seen.append(tuple(support))
        return orig(inst, need, support)
    monkeypatch.setattr(a20, "Learner20", spy)
    a20.eval_row(dict(CFG, support=[1.2]), 5, 1.1, 2, 0.5, 0.25, "1.2", 0, "ru")
    a20.eval_row(CFG, 5, 1.1, 2, 0.5, 0.25, "1.2", 0, "ru")
    assert seen == [(1.2,), SUPPORT]


# ---- analysis ----------------------------------------------------------------------------------------------------------------
def cell(n, cond, a_nom=0.2, rows=40, rho=1.1, ceps_t=0.5, base=0.3):
    m = np.full(rows, base)
    return pd.DataFrame(dict(n=n, cond=cond, rho=rho, s=2, a=0.5, lam=0.25, U_NOM=m, U_CLAMP=m + 0.1, U_RU_CLAMP=m + 0.1, U_ADAPT=m + a_nom, U_ADAPT_V=m,
                             U_CEPS_T=np.full(rows, ceps_t), U_RU_CEP_T=np.full(rows, 0.6), U_opt_T=np.full(rows, 0.8)))
    # ADAPT = base + a_nom; RU_CLAMP = CLAMP = base + 0.1 (the frame() builder overrides ADAPT per condition)


def frame(q0_gap=0.0, q1_gain=0.05, q2_gain=0.2, bad_q1=(), bad_q2=(), drop=(), sizes=SIZES):
    parts = []
    for c in (CONTROL,) + IN_SUPPORT + OUT:
        for n in sizes:
            if (c, n) in drop:
                continue
            for rho in (1.1, 1.5):
                d = cell(n, c, rho=rho)
                if c == CONTROL:
                    d["U_ADAPT"] = d["U_RU_CLAMP"] - q0_gap                              # RU_CLAMP - ADAPT = q0_gap
                elif c in IN_SUPPORT:
                    d["U_ADAPT"] = d["U_RU_CLAMP"] + (0.0 if (c, n) in bad_q1 else q1_gain)
                else:
                    d["U_ADAPT"] = d["U_NOM"] + (0.0 if (c, n) in bad_q2 else q2_gain)
                parts.append(d)
    return pd.concat(parts)


def test_labels():
    r = claims20(frame())
    assert r["label"] == "ROBUST-LEARNER" and r["needed_class"] == "LEARNING-NEEDED" and (r["n_q0"], r["n_q1"], r["n_q2"], r["q2_defined"]) == (2, 6, 8, 8)
    assert claims20(frame(q0_gap=0.10))["label"] == "LIMITED"                              # Q0 fails: learning costs too much with the right model
    r = claims20(frame(bad_q1=[("0.8", 5), ("1.2", 7)]))                                    # Q1 in 4 of 6
    assert r["label"] == "USEFUL" and r["needed_class"] == "CLAMP-ROLLOUT-ENOUGH"
    assert claims20(frame(bad_q1=[("0.8", 5)]))["label"] == "ROBUST-LEARNER"                # Q1 in 5 of 6 is enough
    assert claims20(frame(bad_q2=[("0.6", 5)]))["label"] == "USEFUL"                        # one Q2 statement fails
    q2_four = [(c, n) for c in OUT for n in SIZES][:4]
    assert claims20(frame(bad_q2=q2_four))["label"] == "LIMITED"                            # Q2 in 4 of 8
    assert claims20(frame(bad_q2=[(c, n) for c in OUT for n in SIZES]))["label"] == "NO GAIN"
    assert claims20(frame(bad_q2=[(c, n) for c in OUT for n in SIZES][:3]))["label"] == "USEFUL"    # Q2 in 5 of 8


def test_undefined_statements():
    assert claims20(frame(drop=[("1.2", 7)]))["label"] == "NO OUTCOME"                      # a Q1 statement has no data
    assert claims20(frame(drop=[("1.0", 5)]))["label"] == "NO OUTCOME"                      # a Q0 statement has no data
    assert claims20(frame(drop=[("0.6", 5), ("0.6", 7)]))["label"] == "ROBUST-LEARNER"      # 6 of 8 Q2 statements defined, all hold
    assert claims20(frame(drop=[("0.6", 5), ("0.6", 7), ("SP", 5)]))["label"] == "NO OUTCOME"     # only 5 defined
    r = claims20(frame(drop=[("0.6", 5), ("0.6", 7)]))
    assert r["q2_defined"] == 6


def test_control_is_outside_q1_and_q2_and_sizes_are_configurable():
    f = frame()
    f.loc[f.cond == CONTROL, "U_NOM"] = 0.9                                                 # irrelevant to Q1 and Q2
    assert claims20(f)["label"] == "ROBUST-LEARNER"
    r = claims20(frame(sizes=(5,)), sizes=(5,))
    assert r["label"] == "NO OUTCOME" and r["q2_defined"] == 4                               # with one size only 4 Q2 statements exist


def test_bounds_not_point_estimates():
    wide = cell(7, "1.2", rows=40)
    wide["U_ADAPT"] = wide["U_RU_CLAMP"] + np.r_[np.zeros(36), np.full(4, 0.3)]               # mean +0.03 from 4 instances in 40
    s = stats(wide)
    assert s["vs_ru_clamp"] > 0.01 and not s["q1"]
    ctl = cell(5, CONTROL, rows=40)
    ctl["U_ADAPT"] = ctl["U_RU_CLAMP"] - np.r_[np.full(36, 0.0), np.full(4, 0.25)]            # RU_CLAMP - ADAPT: mean 0.025, wide
    s = stats(ctl)
    assert s["ru_clamp_minus_adapt"] <= 0.03 and not s["q0"]
    o = cell(5, "0.6", rows=40)
    o["U_ADAPT"] = o["U_NOM"] + np.r_[np.zeros(36), np.full(4, 0.4)]                           # mean +0.04, wide
    s = stats(o)
    assert s["vs_nom"] > 0.03 and not s["q2"]


def test_q3_classes_use_the_bounds():
    base = cell(7, "1.2", rows=40)
    d = base.copy(); d["U_ADAPT"] = d["U_RU_CLAMP"] + 0.05
    assert stats(d)["q3"] == "LEARNING-BETTER"
    d["U_ADAPT"] = d["U_RU_CLAMP"] - 0.05
    assert stats(d)["q3"] == "CLAMP-BETTER"
    d["U_ADAPT"] = d["U_RU_CLAMP"] + np.r_[np.zeros(36), np.full(4, 0.2)]                      # point estimate +0.02 but wide
    assert stats(d)["q3"] == "MIXED"
    d["U_ADAPT"] = d["U_RU_CLAMP"] - np.r_[np.zeros(36), np.full(4, 0.2)]
    assert stats(d)["q3"] == "MIXED"


def test_informed_reference_selects_cells():
    assert stats(cell(7, "1.2", ceps_t=0.995))["undefined"] and stats(cell(7, "1.2", ceps_t=0.05))["undefined"]
    d = cell(7, "1.2")
    d["U_ADAPT"] = 0.995
    assert not stats(d)["undefined"]


def test_regret_table_is_against_the_true_optimum_and_in_points():
    d = cell(5, "1.2", rows=10)
    t = regret_table(d)
    assert abs(t["NOM"] - 50.0) < 1e-9 and abs(t["ADAPT"] - 30.0) < 1e-9 and abs(t["RU_CLAMP"] - 40.0) < 1e-9
    assert regret_table(d.drop(columns=["U_opt_T"])) == {}


# ---- job layout and gate -----------------------------------------------------------------------------------------------------
def test_job_layout():
    import run_a20
    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "a20_main.json")))
    J = run_a20.jobs_of(cfg)
    assert len(J) == 256 and sum(j[7] for j in J) == 11520 and {j[1] for j in J} == {5, 7}
    assert {j[2] for j in J} == {"1.0", "0.8", "1.2", "1.4", "0.6", "1.6", "SP", "CP"} and {j[7] for j in J if j[1] == 5} == {50} and {j[7] for j in J if j[1] == 7} == {40}
    assert cfg["eval_seed"] == run_a20.MAIN_SEED == 20261311 and cfg["support"] == list(SUPPORT) and cfg["D0"] == [5, 10]
    assert cfg["conditions"] == CONDS
    for cond in IN_SUPPORT + (CONTROL,):
        assert tuple(cfg["conditions"][cond]) == scaled_dwell((5, 10), float(cond))


def test_gate_and_seed_rules(tmp_path, monkeypatch):
    import run_a20
    monkeypatch.setattr(run_a20, "ROOT", str(tmp_path))
    with pytest.raises(SystemExit):
        run_a20.check_gate(dict(role="main", eval_seed=20261311))
    run_a20.check_gate(dict(role="audit", eval_seed=3))
    with pytest.raises(SystemExit):
        run_a20.check_gate(dict(role="audit", eval_seed=20261311))
    monkeypatch.setattr(run_a20, "code_hash", lambda: "x")
    os.makedirs(tmp_path / "results")
    json.dump({"code_hash": "x"}, open(tmp_path / "results" / "a20_gate.json", "w"))
    run_a20.check_gate(dict(role="main", eval_seed=20261311))
    with pytest.raises(SystemExit):
        run_a20.check_gate(dict(role="main", eval_seed=1))
    with pytest.raises(SystemExit):
        run_a20.check_gate(dict(role="pilot", eval_seed=20261311))
    json.dump({"code_hash": "y"}, open(tmp_path / "results" / "a20_gate.json", "w"))
    with pytest.raises(SystemExit):
        run_a20.check_gate(dict(role="main", eval_seed=20261311))
