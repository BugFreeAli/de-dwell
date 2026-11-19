"""Tests of the A12 model, truths, defenders and row invariants (plan A12.6, items 1 to 5 and 8)."""
import collections

import pytest

from brute import brute_value
from dwell.adapt import Kit, Planner, a7_instance, eval_rec, nominal_factory
from dwell.hard import (ADAPTIVE, DEFENDERS, NONADAPTIVE, PRIOR_B, STRATEGIES, STRAT_NAMES, a12_instance, beta,
                        budget_set, hard_truth, run_instance)
from dwell.instances import make_instance_k


def cfg(n=3, seed=7, noise=1):
    c = dict(master_seed=seed, n=n, D0=[5, 10], value_set=[1, 2, 3], u_low=0.35, planner_scales=[0.8, 1.0, 1.2, 1.4])
    if noise == 1:
        c.update(noise_offsets=[-1, 0, 1], noise_probs=[0.25, 0.5, 0.25])
    else:
        c.update(noise_offsets=[-2, -1, 0, 1, 2], noise_probs=[0.0625, 0.25, 0.375, 0.25, 0.0625])
    return c


def by_hand(inst0, D0, offs, probs, q, lam):
    """Class list and prior rows written out with plain loops (independent of hard_truth)."""
    D, rows = [], [[] for _ in range(inst0.n)]
    for c in range(len(D0)):
        for h, ph in ((0, 1.0 - q), (1, q)):
            for e, pe in zip(offs, probs):
                D.append((int(round(lam * D0[c])) if h else D0[c]) + e)
                for i in range(inst0.n):
                    rows[i].append(inst0.prior[i][c] * ph * pe)
    return D, rows


def test_budget_sets_and_beta():
    assert budget_set("B20") == ("NONE", "SH12", "PH50_14", "PH25_18")
    assert budget_set("B40") == STRAT_NAMES
    assert abs(beta("SH14") - 0.4) < 1e-12 and abs(beta("PH50_18") - 0.4) < 1e-12 and abs(beta("PH25_18") - 0.2) < 1e-12
    assert len(DEFENDERS) == 8 and DEFENDERS == NONADAPTIVE + ADAPTIVE


@pytest.mark.parametrize("noise", [1, 2])
def test_instance_matches_a7(noise):
    c = cfg(noise=noise)
    c7 = dict(c, scales=[0.8, 1.0, 1.2, 1.4])
    for idx in range(3):
        inst, Dl, real, inst0 = a12_instance(c, 1.1, 2, 0.9, idx)
        i7, Dl7, real7, i70 = a7_instance(c7, 1.1, 2, 0.9, idx, make_instance_k)
        assert inst == i7 and Dl == Dl7 and real == real7 and inst0 == i70


def test_shared_and_nominal_truth_equal_planner_models():
    c = cfg()
    inst, Dl, real, inst0 = a12_instance(c, 1.1, 2, 0.9, 1)
    pl = Planner(inst, Dl, c["planner_scales"])
    for lam in (1.0, 1.2, 1.4):
        q = 0.0 if lam == 1.0 else 1.0
        ti = hard_truth(inst0, tuple(c["D0"]), tuple(c["noise_offsets"]), tuple(c["noise_probs"]), q, lam)
        pi = pl.models[lam][0]
        assert ti.D == pi.D and ti.deadlines == pi.deadlines
        assert all(abs(a - b) < 1e-12 for r1, r2 in zip(ti.prior, pi.prior) for a, b in zip(r1, r2))


@pytest.mark.parametrize("noise", [1, 2])
@pytest.mark.parametrize("name,q,lam", [s for s in STRATEGIES if 0 < s[1] < 1])
def test_partial_truth_class_law_by_enumeration(name, q, lam, noise):
    c = cfg(noise=noise)
    inst, Dl, real, inst0 = a12_instance(c, 1.1, 2, 0.5, 2)
    ti = hard_truth(inst0, tuple(c["D0"]), tuple(c["noise_offsets"]), tuple(c["noise_probs"]), q, lam)
    D, rows = by_hand(inst0, c["D0"], c["noise_offsets"], c["noise_probs"], q, lam)
    for i in range(inst0.n):
        law = collections.defaultdict(float)
        for d, p in zip(D, rows[i]):
            law[d] += p
        assert set(law) == set(ti.D)
        for k, d in enumerate(ti.D):
            assert abs(law[d] - ti.prior[i][k]) < 1e-12
        assert abs(sum(ti.prior[i]) - 1.0) < 1e-12


@pytest.mark.parametrize("seed", [3, 4])
def test_true_model_dp_equals_bruteforce(seed):
    c = cfg(n=2, seed=seed)
    for idx in range(2):
        row = run_instance(c, 1.1, 2, 0.9, idx)
        inst, Dl, real, inst0 = a12_instance(c, 1.1, 2, 0.9, idx)
        for name, q, lam in STRATEGIES:
            D, rows = by_hand(inst0, c["D0"], c["noise_offsets"], c["noise_probs"], q if q > 0 else 0.0, lam)
            if q == 0.0 or q == 1.0:                           # drop zero-probability classes
                keep = [k for k in range(len(D)) if rows[0][k] > 0 or rows[-1][k] > 0]
                D = [D[k] for k in keep]; rows = [[r[k] for k in keep] for r in rows]
            bf = brute_value(tuple(D), inst0.s, inst0.values, inst0.deadlines, tuple(tuple(r) for r in rows))
            assert abs(bf - row[f"Vstar_{name}"]) < 1e-9, (name, bf, row[f"Vstar_{name}"])


@pytest.mark.parametrize("noise", [1, 2])
def test_row_invariants_and_columns(noise):
    c = cfg(noise=noise)
    for idx, (rho, s, a) in enumerate([(1.1, 2, 0.9), (1.5, 4, 0.5)]):
        row = run_instance(c, rho, s, a, idx)
        assert row["violations"] == ""
        for k in STRAT_NAMES:
            assert row[f"Vstar_{k}"] >= 0
            for d in DEFENDERS:
                assert -1e-9 <= row[f"V_{d}_{k}"] <= row[f"Vstar_{k}"] + 1e-9
        assert abs(row["V_DPMIS_1.0_NONE"] - row["Vstar_NONE"]) < 1e-9
        assert abs(row["V_DPMIS_1.2_SH12"] - row["Vstar_SH12"]) < 1e-9
        assert abs(row["V_IID_DP_PH50_14"] - row["Vstar_PH50_14"]) < 1e-9


def test_none_and_shared_reproduce_a7_values():
    c = cfg()
    c7 = dict(c, scales=[0.8, 1.0, 1.2, 1.4])
    for idx in range(2):
        row = run_instance(c, 1.1, 2, 0.9, idx)
        i7, Dl7, real7, inst0 = a7_instance(c7, 1.1, 2, 0.9, idx, make_instance_k)
        p7 = Planner(i7, Dl7, {0.8, 1.0, 1.2})
        assert abs(row["Vstar_NONE"] - p7.models[1.0][2].value) < 1e-12
        assert abs(row["Vstar_SH12"] - p7.models[1.2][2].value) < 1e-12
        for meth in ("DPMIS", "CEP", "ROLL_CEP"):
            fac = nominal_factory(p7, meth, 1.0)
            assert abs(row[f"V_{meth}_1.0_NONE"] - eval_rec(p7.models[1.0][0], fac)) < 1e-12
            assert abs(row[f"V_{meth}_1.0_SH12"] - eval_rec(p7.models[1.2][0], fac)) < 1e-12
        kit = Kit(p7, PRIOR_B, tuple(1.0 / 3 for _ in range(3)))
        assert abs(row["V_QMDP_B_SH12"] - eval_rec(p7.models[1.2][0], kit.factory("QMDP"))) < 1e-12
        assert abs(row["V_QMDP_B_NONE"] - eval_rec(p7.models[1.0][0], kit.factory("QMDP"))) < 1e-12


def test_bayes_adaptive_ceiling():
    row = run_instance(cfg(), 1.1, 2, 0.9, 1)
    w = 1.0 / 3
    ba = sum(w * row[f"V_BA_H_{k}"] for k in ("NONE", "SH12", "SH14"))
    qm = sum(w * row[f"V_QMDP_H_{k}"] for k in ("NONE", "SH12", "SH14"))
    assert abs(ba - row["V_BA_H_dp"]) < 1e-8
    assert qm <= row["V_BA_H_dp"] + 1e-8
