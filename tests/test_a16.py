"""Tests of the A16 evaluation rows (plan A16.7, items 1 to 3)."""
import math

import numpy as np
import pytest

from dwell.a16 import expected_killed, eval_row
from dwell.dp import solve
from dwell.instances import make_instance_k
from dwell.m0 import Instance, Model
from dwell.m0u import need_of
from dwell.rules import make_rule

CFG = dict(eval_seed=9, D0=[5, 10], value_set=[1, 2, 3], u_low=0.35)


def small_model(i=0, n=4):
    inst, _ = make_instance_k(CFG["eval_seed"], i, n, 1.1, 2, 0.5, D=(5, 10), value_set=(1, 2, 3), u_low=0.35)
    return Model(inst)


@pytest.mark.parametrize("i", range(4))
def test_expected_killed_of_the_ev_policy_equals_the_ev_optimum(i):
    m = small_model(i)
    ev = solve(m)
    pol = ev.policy()
    assert abs(expected_killed(m, lambda st, ac: pol(st)) - ev.value) < 1e-9


def test_expected_killed_of_a_deterministic_instance():
    inst = Instance(D=(2, 3), s=1, values=(1.0, 3.0), deadlines=(9, 9), prior=((1.0, 0.0), (0.0, 1.0)))   # both killable in time
    m = Model(inst)
    cep = make_rule(m, "CEP")
    assert abs(expected_killed(m, lambda st, ac: cep(st)) - 4.0) < 1e-12
    assert expected_killed(m, lambda st, ac: None) == 0.0


def test_exact_row_invariants_and_columns():
    r = eval_row(CFG, "main", 4, 1.5, 4, 0.5, 0.25, 0, "exact")
    assert r["violations"] == ""
    for k in ("U_CEP", "U_CEPS", "U_RU_CEP", "U_opt", "U_DPEV", "U_RU_DPEV", "V_ev", "E_CEP", "tau", "tau_ev"):
        assert k in r
    assert r["U_opt"] >= max(r["U_CEP"], r["U_CEPS"], r["U_RU_CEP"], r["U_DPEV"], r["U_RU_DPEV"]) - 1e-9
    assert r["V_ev"] >= r["E_CEP"] - 1e-9 and abs(r["tau"] - r["need"] / r["E_CEP"]) < 1e-12


def test_levels_select_the_planners():
    ru = eval_row(CFG, "main", 6, 1.1, 2, 0.5, 0.25, 1, "ru")
    ch = eval_row(CFG, "main", 6, 1.1, 2, 0.5, 0.25, 1, "cheap")
    assert "U_RU_CEP" in ru and "U_DPEV" not in ru and "U_RU_CEP" not in ch and "U_opt" not in ch
    assert ru["U_CEP"] == ch["U_CEP"] and ru["U_CEPS"] == ch["U_CEPS"] and ru["tau"] == ch["tau"]


def test_rows_are_deterministic_and_use_the_variant_instances():
    a, b = eval_row(CFG, "main", 4, 1.5, 4, 0.5, 0.25, 2, "ru"), eval_row(CFG, "main", 4, 1.5, 4, 0.5, 0.25, 2, "ru")
    assert {k: v for k, v in a.items() if k != "sec"} == {k: v for k, v in b.items() if k != "sec"}
    k3 = eval_row(CFG, "K3", 4, 1.5, 4, 0.5, 0.25, 2, "ru", D0=[3, 6, 10])
    base = eval_row(CFG, "main", 4, 1.5, 4, 0.5, 0.25, 2, "ru")
    assert k3["group"] == "K3" and (k3["U_CEP"], k3["E_CEP"]) != (base["U_CEP"], base["E_CEP"])
    eq = eval_row(CFG, "equal", 4, 1.5, 4, 0.5, 0.25, 2, "ru", value_set=[2])
    assert eq["W"] == 8.0 and eq["need"] == need_of([2, 2, 2, 2], 0.25)



def test_a_method_above_the_optimum_is_flagged(monkeypatch):
    import dwell.a16 as a16

    class Fake:
        def __init__(self, model, need):
            self.value = 0.0
    monkeypatch.setattr(a16, "GoalDP", Fake)
    r = a16.eval_row(CFG, "main", 4, 1.1, 2, 0.5, 0.25, 0, "exact")
    assert r["U_CEP"] > 0 and "above_optimum" in r["violations"]
