"""Tests of A18 (plan A18.5): the evaluation under a wrong dwell scale (src/dwell/a18.py), the analysis and the job layout."""
import dataclasses, json, os, sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.setrecursionlimit(200000)
from dwell.a18 import eval_row, prob_goal_true, scaled_dwell
from dwell.analysis_a18 import MOSTLY_MIN, SCALES, SIZES, claims18, regret_table, stats
from dwell.dp import solve
from dwell.instances import make_instance_k
from dwell.m0 import DEAD, Instance, Model
from dwell.m0u import ceps_policy, need_of, prob_goal, ru_policy
from dwell.rules import make_rule

CFG = dict(D0=[5, 10], value_set=[1, 2, 3], u_low=0.35, eval_seed=3)


def one_threat(deadline):
    inst = Instance(D=(2, 4), s=0, values=(1.0,), deadlines=(deadline,), prior=((0.5, 0.5),))
    m = Model(inst)
    pol = lambda st, ac: (m.actions(st) or [None])[0]
    return m, pol


# ---- row logic ---------------------------------------------------------------------------------------------------------------
def test_hand_case_dwell_twice_as_long_as_believed():
    m, pol = one_threat(5)                       # nominal D = (2, 4); true D = (4, 8); deadline 5
    p, late = prob_goal_true(m, (4, 8), pol, 1)
    assert abs(p - 0.5) < 1e-12 and late == 0    # class 0 is killed at t = 4; class 1 is abandoned when the nominal model rules it out (x = 4)


def test_hand_case_dwell_half_as_long_as_believed():
    m, pol = one_threat(3)                       # true D = (1, 2): both classes die by t = 2
    p, late = prob_goal_true(m, (1, 2), pol, 1)
    assert abs(p - 1.0) < 1e-12 and late == 0


def test_hand_case_abandon_when_the_nominal_model_says_the_deadline_is_missed():
    m, pol = one_threat(3)                       # true D = (4, 8) cannot be reached before t = 3: no kill, and no kill completes late
    p, late = prob_goal_true(m, (4, 8), pol, 1)
    assert p == 0.0 and late == 0


def test_scale_one_reproduces_the_exact_evaluation_for_every_planner():
    for idx in range(6):
        inst, _ = make_instance_k(7, idx, 4, 1.5, 2, 0.5, D=(5, 10), value_set=(1, 2, 3), u_low=0.35)
        m = Model(inst)
        need = need_of(inst.values, 0.25)
        cep = make_rule(m, "CEP")
        pc = lambda st, ac: cep(st)
        evp = solve(m).policy()
        pd_ = lambda st, ac: evp(st)
        for pol in (pc, ceps_policy(m, need), ru_policy(m, pc, need), pd_):
            assert abs(prob_goal_true(m, (5, 10), pol, need)[0] - prob_goal(m, pol, need)) < 1e-12


def uncached(mb, Dt, policy, need):
    """Independent re-implementation without the decision cache (same semantics, written separately)."""
    total = 0.0
    for c, w in mb.worlds():
        t, cur, acc, x = 0, -1, 0.0, [0] * mb.n
        prog = list(mb.initial_state()[2])
        while acc < need - 1e-9:
            j = policy((t, cur, tuple(prog)), acc)
            if j is None:
                break
            t2 = t + (mb.s if (cur != -1 and cur != j) else 0) + 1
            x[j] += 1
            new = list(prog)
            if x[j] == Dt[c[j]]:
                acc += mb.v[j] if t2 <= mb.d[j] else 0.0
                new[j] = DEAD
            else:
                new[j] = min(x[j], mb.D[-1])
            prog = list(mb.canon(t2, new))
            t, cur = t2, j
        total += w * float(acc >= need - 1e-9)
    return total


@pytest.mark.parametrize("scale", [0.8, 1.2, 1.4])
def test_cache_does_not_change_the_probability(scale):
    for idx in range(4):
        inst, _ = make_instance_k(11, idx, 5, 1.1, 4, 0.9, D=(5, 10), value_set=(1, 2, 3), u_low=0.35)
        m = Model(inst)
        need = need_of(inst.values, 0.5)
        pol = ceps_policy(m, need)
        Dt = scaled_dwell((5, 10), scale)
        assert abs(prob_goal_true(m, Dt, pol, need)[0] - uncached(m, Dt, pol, need)) < 1e-12


def test_scaled_dwell_must_be_integer():
    assert scaled_dwell((5, 10), 1.2) == (6, 12) and scaled_dwell((5, 10), 0.8) == (4, 8) and scaled_dwell((5, 10), 1.4) == (7, 14)
    with pytest.raises(AssertionError):
        scaled_dwell((5, 10), 0.9)


def test_rows_have_no_violation_and_the_scale_is_really_used():
    differs = 0
    for sc in (1.0, 0.8, 1.2, 1.4):
        for idx in range(5):
            r = eval_row(CFG, 5, 1.5, 2, 0.5, 0.25, sc, idx, "exact")
            assert r["violations"] == ""
            assert all(r["U_" + m] <= r["U_opt_T"] + 1e-9 for m in ("CEP", "CEPS", "RU_CEP", "DPEV", "RU_DPEV", "CEP_T", "CEPS_T"))
            differs += abs(r["U_CEPS"] - r["U_CEPS_T"]) > 1e-9 and sc != 1.0
    assert differs > 0                           # a wrong scale changes the nominal planner's result somewhere


def test_informed_references_are_computed_on_the_true_model():
    from dwell.m0u import GoalDP
    for sc in (0.8, 1.2, 1.4):
        for idx in range(3):
            r = eval_row(CFG, 5, 1.5, 2, 0.5, 0.25, sc, idx, "exact")
            inst, _ = make_instance_k(3, idx, 5, 1.5, 2, 0.5, D=(5, 10), value_set=(1, 2, 3), u_low=0.35)
            mt = Model(dataclasses.replace(inst, D=scaled_dwell((5, 10), sc)))
            need = need_of(inst.values, 0.25)
            cep = make_rule(mt, "CEP")
            assert abs(r["U_CEP_T"] - prob_goal(mt, lambda st, ac: cep(st), need)) < 1e-12
            assert abs(r["U_CEPS_T"] - prob_goal(mt, ceps_policy(mt, need), need)) < 1e-12
            assert abs(r["U_opt_T"] - GoalDP(mt, need).value) < 1e-12


def test_levels_and_common_random_numbers():
    ru = eval_row(CFG, 7, 1.1, 2, 0.5, 0.25, 1.2, 0, "ru")
    ex = eval_row(CFG, 5, 1.1, 2, 0.5, 0.25, 1.2, 0, "exact")
    assert "U_opt_T" not in ru and "U_DPEV" not in ru and "U_RU_CEP" in ru and "U_CEPS_T" in ru
    assert "U_opt_T" in ex and "U_DPEV" in ex and "U_RU_DPEV" in ex
    a, b = eval_row(CFG, 5, 1.5, 4, 0.9, 0.5, 0.8, 3, "exact"), eval_row(CFG, 5, 1.5, 4, 0.9, 0.5, 1.4, 3, "exact")
    assert (a["need"], a["W"], a["d_max"]) == (b["need"], b["W"], b["d_max"])      # same instance at every scale
    assert abs(eval_row(CFG, 5, 1.5, 4, 0.9, 0.5, 1.0, 3, "exact")["U_CEPS"] - eval_row(CFG, 5, 1.5, 4, 0.9, 0.5, 1.0, 3, "exact")["U_CEPS_T"]) < 1e-9


# ---- analysis ----------------------------------------------------------------------------------------------------------------
def cell(n, scale, gain, rho=1.1, rows=40, ceps_t=0.5, cost=0.0, aware=0.05, ru=0.01):
    g = np.asarray(gain, dtype=float) if np.ndim(gain) else np.full(rows, float(gain))
    ceps = 0.4
    return pd.DataFrame(dict(n=n, scale=scale, rho=rho, s=2, a=0.5, lam=0.25, U_CEP=np.clip(ceps - g, 0, 1), U_CEPS=np.full(len(g), ceps),
                             U_CEPS_T=np.full(len(g), ceps_t), U_CEP_T=np.full(len(g), ceps - aware), U_RU_CEP=np.full(len(g), ceps + ru),
                             U_opt_T=np.full(len(g), 0.8)))


def frame(gain=0.05, bad=(), control_gain=0.05, sizes=SIZES, **kw):
    parts = []
    for sc in (1.0,) + SCALES:
        for n in sizes:
            g = control_gain if sc == 1.0 else (0.0 if (sc, n) in bad else gain)
            for rho in (1.1, 1.5):
                parts.append(cell(n, sc, g, rho=rho, **kw))
    return pd.concat(parts)


def test_p1_uses_the_lower_bound_and_the_informed_reference_selects_cells():
    assert stats(cell(7, 1.2, 0.05))["p1"]
    assert not stats(cell(7, 1.2, 0.0))["p1"]
    one = cell(7, 1.2, np.r_[np.zeros(9), 0.5], rows=10)                                   # mean gain 0.05 from one instance in ten
    assert stats(one)["gain"] > 0.01 and not stats(one)["p1"]
    assert stats(cell(7, 1.2, 0.05, ceps_t=0.995))["undefined"]                             # the reference (CEPS_T) is outside [0.10, 0.99]
    assert stats(cell(7, 1.2, 0.05, ceps_t=0.05))["undefined"]
    d = cell(7, 1.2, 0.05, ceps_t=0.5)
    d["U_CEPS"] = 0.995                                                                    # the planner under test being saturated does not remove the cell
    assert not stats(d)["undefined"]


def test_labels():
    assert claims18(frame())["label"] == "ROBUST"
    assert claims18(frame(bad=[(0.8, 5)]))["label"] == "MOSTLY"                              # 8 of 9
    assert claims18(frame(bad=[(0.8, 5), (1.4, 9)]))["label"] == "MOSTLY"                    # 7 of 9
    assert claims18(frame(bad=[(0.8, 5), (1.4, 9), (1.2, 7)]))["label"] == "NOT ROBUST"      # 6 of 9
    assert MOSTLY_MIN == 7
    d = frame()
    assert claims18(d[~((d.scale == 1.2) & (d.n == 7))])["label"] == "NO OUTCOME"
    assert claims18(frame())["n_statements"] == 9


def test_control_is_not_part_of_the_label_and_sizes_are_configurable():
    assert claims18(frame(control_gain=0.0))["label"] == "ROBUST"
    r = claims18(frame(sizes=(5, 7)), sizes=(5, 7))
    assert r["label"] == "ROBUST" and r["n_statements"] == 6


def test_secondary_classes():
    s = stats(cell(7, 1.2, 0.05, ceps_t=0.5, aware=0.05, ru=0.01))                          # CEPS_T 0.5 vs CEPS 0.4: informed is 10 points better
    assert s["cost_class"] == "LARGE" and s["aware_class"] == "AWARENESS-WINS" and s["adequacy"] == "ADEQUATE"
    s2 = stats(cell(7, 1.2, 0.05, ceps_t=0.41, aware=-0.05, ru=0.08))
    assert s2["cost_class"] == "SMALL" and s2["aware_class"] == "KNOWLEDGE-WINS" and s2["adequacy"] == "ROLLOUT-WORTH"
    wide = cell(7, 1.2, 0.05, rows=40)
    wide["U_CEP_T"] = np.r_[np.full(20, 0.0), np.full(20, 0.8)]                              # mean difference 0.0 with a wide spread
    assert stats(wide)["aware_class"] == "MIXED"
    noisy = cell(7, 1.2, 0.05, rows=40)
    noisy["U_CEPS_T"] = np.r_[np.full(20, 0.1), np.full(20, 0.8)]                            # mean 0.45; vs CEPS 0.4 -> wide, neither class
    assert stats(noisy)["cost_class"] == "INCONCLUSIVE"


def test_classes_use_the_bounds_not_the_point_estimates():
    def wide(col, lo_v, hi_v):
        d = cell(7, 1.2, 0.05, rows=40)
        d[col] = np.r_[np.full(20, lo_v), np.full(20, hi_v)]
        return stats(d)
    # CEPS = 0.4 in every row; each series below has the stated mean and a wide spread
    assert abs(wide("U_CEPS_T", 0.1, 0.7)["cost"]) < 1e-9 and wide("U_CEPS_T", 0.1, 0.7)["cost_class"] == "INCONCLUSIVE"        # mean 0 <= 0.03 but upper bound above
    assert wide("U_CEPS_T", 0.1, 0.78)["cost"] > 0.03 and wide("U_CEPS_T", 0.1, 0.78)["cost_class"] == "INCONCLUSIVE"            # mean above 0.03, lower bound below
    w = wide("U_CEP_T", 0.0, 0.77)                                                                                         # aware = 0.4 - U_CEP_T, mean +0.015
    assert w["aware"] >= 0.01 and w["aware_class"] == "MIXED"
    w = wide("U_CEP_T", 0.1, 0.73)                                                                                         # mean -0.015
    assert w["aware"] <= -0.01 and w["aware_class"] == "MIXED"
    r = wide("U_RU_CEP", 0.1, 0.74)                                                                                        # gap mean +0.02
    assert r["ru_gap"] <= 0.03 and r["adequacy"] == "INCONCLUSIVE"
    r = wide("U_RU_CEP", 0.1, 0.8)                                                                                         # gap mean +0.05
    assert r["ru_gap"] >= 0.03 and r["adequacy"] == "INCONCLUSIVE"


def test_regret_table_is_against_the_true_optimum_and_in_points():
    d = frame()
    d = d[(d.scale == 1.2) & (d.n == 5)].assign(U_DPEV=0.7, U_RU_DPEV=0.75)
    t = regret_table(d)
    assert abs(t["CEPS"] - 40.0) < 1e-9 and abs(t["DPEV"] - 10.0) < 1e-9 and abs(t["CEPS_T"] - 30.0) < 1e-9
    assert regret_table(frame().drop(columns=["U_opt_T"])) == {}


# ---- job layout and gate -----------------------------------------------------------------------------------------------------
def test_job_layout():
    import run_a18
    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "a18_main.json")))
    J = run_a18.jobs_of(cfg)
    assert len(J) == 192 and sum(j[7] for j in J) == 12800
    assert {j[1] for j in J} == {5, 7, 9} and {j[2] for j in J} == {1.0, 0.8, 1.2, 1.4} and {j[7] for j in J if j[1] == 9} == {40}
    assert cfg["eval_seed"] == run_a18.MAIN_SEED == 20261291 and cfg["D0"] == [5, 10]
    for sc in cfg["scales"]:
        scaled_dwell(tuple(cfg["D0"]), sc)                                                  # every scale gives integer dwell


def test_gate_refuses_main_without_gate(tmp_path, monkeypatch):
    import run_a18
    monkeypatch.setattr(run_a18, "ROOT", str(tmp_path))
    with pytest.raises(SystemExit):
        run_a18.check_gate(dict(role="main", eval_seed=20261291))
    run_a18.check_gate(dict(role="audit", eval_seed=3))
    with pytest.raises(SystemExit):
        run_a18.check_gate(dict(role="audit", eval_seed=20261291))


def test_main_role_needs_the_main_seed_and_the_current_hash(tmp_path, monkeypatch):
    import run_a18
    monkeypatch.setattr(run_a18, "ROOT", str(tmp_path))
    monkeypatch.setattr(run_a18, "code_hash", lambda: "x")
    os.makedirs(tmp_path / "results")
    json.dump({"code_hash": "x"}, open(tmp_path / "results" / "a18_gate.json", "w"))
    run_a18.check_gate(dict(role="main", eval_seed=20261291))
    with pytest.raises(SystemExit):
        run_a18.check_gate(dict(role="main", eval_seed=1))
    json.dump({"code_hash": "y"}, open(tmp_path / "results" / "a18_gate.json", "w"))
    with pytest.raises(SystemExit):
        run_a18.check_gate(dict(role="main", eval_seed=20261291))
