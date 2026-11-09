import dataclasses
import importlib.util
import os

import numpy as np
import pytest

from dwell.dp import solve
from dwell.instances import make_instance_k
from dwell.m2 import Model2, evaluate_policy2
from dwell.rules import make_rule

spec = importlib.util.spec_from_file_location("run_a4", os.path.join(os.path.dirname(__file__), "..", "scripts", "run_a4.py"))
run_a4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_a4)


@pytest.mark.parametrize("D", [(2, 5), (1, 3, 9)])
def test_wrong_belief_never_beats_true_dp_and_correct_belief_matches(D):
    for idx in range(6):
        inst, _ = make_instance_k(5, idx, 4, 1.1, 2, 0.7, D=D)
        mt = Model2(inst)
        dp_t = solve(mt)
        for a_b in (0.5, 0.7, 0.9, 0.99):
            inst_b = run_a4.believed_instance(inst, a_b)
            mb = Model2(inst_b)
            dp_b = solve(mb)
            for m in run_a4.METHODS:
                v = evaluate_policy2(mt, run_a4.policy_for(m, mb, dp_b))
                assert -1e-12 <= v <= dp_t.value + 1e-9
            if a_b == 0.7:
                assert evaluate_policy2(mt, dp_b.policy()) == pytest.approx(dp_t.value, abs=1e-9)


def test_believed_instance_keeps_labels_and_deadlines():
    inst, _ = make_instance_k(3, 1, 5, 1.1, 2, 0.9, D=(1, 3, 9))
    b = run_a4.believed_instance(inst, 0.6)
    assert b.deadlines == inst.deadlines and b.values == inst.values
    for r_t, r_b in zip(inst.prior, b.prior):
        assert int(np.argmax(r_t)) == int(np.argmax(r_b))
        assert sum(r_b) == pytest.approx(1.0)
