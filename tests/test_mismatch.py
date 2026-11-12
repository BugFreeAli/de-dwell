import dataclasses

import numpy as np
import pytest

from brute_ba import brute_value_ba
from dwell.dp import solve
from dwell.instances import make_instance_k
from dwell.m0 import Instance
from dwell.m2 import Model2, evaluate_policy2
from dwell.mismatch import (ModelBA, ba_factory, base_policy_for, believed_factory, eval_obs, scaled)
from dwell.rules import make_rule


def nominal(idx, n=4, rho=1.1, s=2, a=0.7, D0=(5, 10)):
    inst, _ = make_instance_k(11, idx, n, rho, s, a, D=D0, value_set=(1, 2, 3))
    return inst


def test_scaled_is_exact_or_fails():
    assert scaled((5, 10), 0.8) == (4, 8) and scaled((5, 10), 1.2) == (6, 12)
    with pytest.raises(AssertionError):
        scaled((5, 10), 0.9)


@pytest.mark.parametrize("method", ["DPMIS", "CEP", "ROLL_CEP"])
def test_observation_evaluator_matches_exact_values_when_belief_is_true(method):
    for idx in range(4):
        inst = nominal(idx, n=4, D0=(2, 5))
        m = Model2(inst)
        dp = solve(m)
        base = base_policy_for(method, m, dp)
        v_obs = eval_obs(inst, believed_factory(m, base))
        v_exact = evaluate_policy2(m, base)
        assert v_obs == pytest.approx(v_exact, abs=1e-9)
        if method == "DPMIS":
            assert v_obs == pytest.approx(dp.value, abs=1e-9)


def test_wrong_belief_never_beats_calibrated_optimum():
    for idx in range(4):
        inst1 = nominal(idx)
        for lam_t in (0.8, 1.0, 1.2):
            inst_t = dataclasses.replace(inst1, D=scaled((5, 10), lam_t))
            v_cal = solve(Model2(inst_t)).value
            for lam_b in (0.6, 1.0, 1.4):
                inst_b = dataclasses.replace(inst1, D=scaled((5, 10), lam_b))
                mb = Model2(inst_b)
                dpb = solve(mb)
                for method in ("DPMIS", "CEP", "ROLL_CEP"):
                    v = eval_obs(inst_t, believed_factory(mb, base_policy_for(method, mb, dpb)))
                    assert -1e-12 <= v <= v_cal + 1e-9


def test_ba_with_single_scale_equals_model2():
    for idx in range(4):
        inst = nominal(idx)
        for lam in (0.8, 1.0, 1.2):
            ba = ModelBA(inst, (lam,))
            m2 = Model2(dataclasses.replace(inst, D=scaled((5, 10), lam)))
            assert solve(ba).value == pytest.approx(solve(m2).value, abs=1e-9)


@pytest.mark.parametrize("lams,D0", [((1.0, 2.0), (1, 2)), ((1.0, 2.0, 3.0), (1, 2)), ((1.0, 2.0), (1, 3))])
def test_ba_dp_equals_bruteforce(lams, D0):
    rng = np.random.default_rng(5)
    for _ in range(8):
        n = int(rng.integers(2, 4))
        prior = tuple(tuple(float(x) for x in rng.dirichlet(np.ones(2))) for _ in range(n))
        values = tuple(float(v) for v in rng.integers(1, 4, size=n))
        deadlines = tuple(int(x) for x in rng.integers(1, 8, size=n))
        s = int(rng.integers(0, 3))
        inst = Instance(D=D0, s=s, values=values, deadlines=deadlines, prior=prior)
        ba = ModelBA(inst, lams)
        lp = tuple(1.0 / len(lams) for _ in lams)
        assert solve(ba).value == pytest.approx(brute_value_ba(D0, lams, lp, s, values, deadlines, prior), abs=1e-9)


def test_ba_policy_value_consistency_and_information_ordering():
    lams = (0.8, 1.0, 1.2)
    for idx in range(4):
        inst = nominal(idx)
        ba = ModelBA(inst, lams)
        dp_ba = solve(ba)
        vba = []
        vcal = []
        vnom = []
        mnom = Model2(dataclasses.replace(inst, D=scaled((5, 10), 1.0)))
        dpnom = solve(mnom)
        for lam in lams:
            inst_t = dataclasses.replace(inst, D=scaled((5, 10), lam))
            vba.append(eval_obs(inst_t, ba_factory(ba, dp_ba.policy())))
            vcal.append(solve(Model2(inst_t)).value)
            vnom.append(eval_obs(inst_t, believed_factory(mnom, dpnom.policy())))
        assert np.mean(vba) == pytest.approx(dp_ba.value, abs=1e-9)
        assert np.mean(vcal) >= dp_ba.value - 1e-9
        assert dp_ba.value >= np.mean(vnom) - 1e-9


def test_ba_with_dwell_noise_equals_bruteforce_and_policy_consistency():
    from dwell.mismatch import expand_prior, noisy_classes
    rng = np.random.default_rng(9)
    base = (3, 6)
    offs = (-1, 0, 1)
    probs = (0.25, 0.5, 0.25)
    lams = (1.0, 2.0)
    Dl = [noisy_classes(base, lam, offs) for lam in lams]
    for _ in range(4):
        n = 2
        prior2 = tuple(tuple(float(x) for x in rng.dirichlet(np.ones(2))) for _ in range(n))
        values = tuple(float(v) for v in rng.integers(1, 4, size=n))
        deadlines = tuple(int(x) for x in rng.integers(4, 12, size=n))
        s = int(rng.integers(0, 3))
        inst = Instance(D=Dl[0], s=s, values=values, deadlines=deadlines, prior=expand_prior(prior2, probs))
        ba = ModelBA(inst, lams, Dl=Dl)
        dp = solve(ba)
        lp = tuple(1.0 / len(lams) for _ in lams)
        assert dp.value == pytest.approx(brute_value_ba(None, lams, lp, s, values, deadlines, inst.prior, Dl=Dl), abs=1e-9)
        vs = [eval_obs(dataclasses.replace(inst, D=Dl[i]), ba_factory(ba, dp.policy())) for i in range(len(lams))]
        assert np.mean(vs) == pytest.approx(dp.value, abs=1e-9)


def test_noisy_classes_are_ascending_for_all_scales():
    from dwell.mismatch import noisy_classes
    for lam in (0.6, 0.8, 1.0, 1.2, 1.4):
        D = noisy_classes((5, 10), lam, (-1, 0, 1))
        assert all(D[k] < D[k + 1] for k in range(len(D) - 1)) and D[0] >= 1
