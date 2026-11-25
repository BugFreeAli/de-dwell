"""Tests of the A17 analysis and job layout (plan A17.6, items 1 to 3); the row logic is tested in tests/test_a16.py."""
import json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from dwell.analysis_a17 import MOSTLY_MIN, SIZES, VARIANTS, claims17, family_stats


def cell(group, n, gain, ru_gap, tau, rho=1.1, rows=40, ceps=0.5):
    g = np.asarray(gain, dtype=float) if np.ndim(gain) else np.full(rows, gain)
    t = np.asarray(tau, dtype=float) if np.ndim(tau) else np.full(rows, tau)
    return pd.DataFrame(dict(group=group, n=n, rho=rho, s=2, a=0.5, lam=0.25, U_CEP=np.clip(ceps - g, 0, 1), U_CEPS=np.full(rows, ceps),
                             U_RU_CEP=np.full(rows, ceps + ru_gap), tau=t))


def frame(gain=0.05, ru_gap=0.01, bad=(), n_groups=VARIANTS + ("base",), sizes=None):
    parts = []
    for f in n_groups:
        for n in (sizes or {}).get(f, SIZES):
            gg = 0.0 if (f, n) in bad else gain
            for rho in (1.1, 1.5):
                parts.append(cell(f, n, gg, ru_gap, 0.95, rho=rho))
    return pd.concat(parts)


def test_adequacy_classes_and_p1():
    d = cell("K3", 7, 0.05, 0.01, 1.0)
    f = family_stats(d)
    assert f["p1"] and f["adequacy"] == "ADEQUATE"
    assert family_stats(cell("K3", 7, 0.05, 0.08, 1.0))["adequacy"] == "ROLLOUT-WORTH"
    wide = cell("K3", 7, 0.05, np.r_[np.full(20, -0.2), np.full(20, 0.3)], 1.0)              # mean 0.05 with wide spread: neither class
    assert family_stats(wide)["adequacy"] == "INCONCLUSIVE"
    noisy_small = cell("K3", 7, 0.05, np.r_[np.full(20, -0.3), np.full(20, 0.32)], 1.0)        # mean 0.01 but a wide spread: not ADEQUATE
    assert family_stats(noisy_small)["ru_gap"] <= 0.03 and family_stats(noisy_small)["adequacy"] == "INCONCLUSIVE"
    one = cell("K3", 7, np.r_[np.zeros(9), 0.5], 0.01, 1.0, rows=10)                          # mean gain 0.05 from one instance in ten
    assert family_stats(one)["gain"] > 0.01 and not family_stats(one)["p1"]
    assert not family_stats(cell("K3", 7, 0.0, 0.01, 1.0))["p1"]
    assert family_stats(cell("K3", 7, 0.05, 0.01, 1.0, ceps=0.995))["undefined"]


def test_labels():
    sizes = {}
    assert claims17(frame(), sizes)["label"] == "ROBUST"
    assert claims17(frame(bad=[("K3", 7)]), sizes)["label"] == "MOSTLY"                                  # 7 of 8 hold: not ROBUST
    assert claims17(frame(bad=[("K3", 7), ("wide", 9)]), sizes)["label"] == "MOSTLY"                   # 6 of 8 hold
    assert claims17(frame(bad=[("K3", 7), ("wide", 9), ("equal", 7)]), sizes)["label"] == "NOT ROBUST"   # 5 of 8
    assert MOSTLY_MIN == 6
    d = frame()
    assert claims17(d[~((d.group == "equal") & (d.n == 9))], sizes)["label"] == "NO OUTCOME"


def test_k3_may_have_its_own_sizes_and_base_is_not_part_of_the_label():
    sizes = {"K3": [7]}
    d = frame(sizes=sizes)
    r = claims17(d, sizes)
    assert r["label"] == "ROBUST" and r["n_statements"] == 7
    d2 = frame(bad=[("base", 7), ("base", 9)])
    assert claims17(d2, {})["label"] == "ROBUST"                                                      # failures in the control do not change the label


def test_p3_pooled_uses_the_variants_only():
    gain = np.r_[np.full(20, 0.1), np.zeros(20)]
    tau = np.r_[np.full(20, 0.95), np.full(20, 1.6)]
    parts = [cell(f, n, gain, 0.01, tau, rho=r) for f in VARIANTS for n in SIZES for r in (1.1, 1.5)]
    good = claims17(pd.concat(parts), {})
    assert good["P3_pooled"]["holds"] and good["P3_pooled"]["diff"] > 0.09
    flat = [cell("base", n, gain, 0.01, tau, rho=r) for n in SIZES for r in (1.1, 1.5)] + \
           [cell(f, n, 0.05, 0.01, 0.95, rho=r) for f in VARIANTS for n in SIZES for r in (1.1, 1.5)]
    assert not claims17(pd.concat(flat), {})["P3_pooled"]["holds"]                                     # the structure only in the base is ignored


def test_job_layout():
    import run_a17
    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "a17_main.json")))
    J = run_a17.jobs_of(cfg)
    assert len(J) == 160 and sum(j[7] for j in J) == 5280
    k3 = [j for j in J if j[1] == "K3"]
    assert {j[2] for j in k3} == {7, 9} and {j[7] for j in k3 if j[2] == 9} == {20} and list(k3[0][8]) == [3, 6, 10]
    assert [list(j[9]) for j in J if j[1] == "equal"][0] == [2] and [list(j[9]) for j in J if j[1] == "spread"][0] == [1, 3, 9]
    assert cfg["eval_seed"] == run_a17.MAIN_SEED == 20261281


def test_gate_refuses_main_without_gate(tmp_path, monkeypatch):
    import run_a17
    monkeypatch.setattr(run_a17, "ROOT", str(tmp_path))
    try:
        run_a17.check_gate(dict(role="main", eval_seed=20261281))
        assert False
    except SystemExit:
        pass
    run_a17.check_gate(dict(role="audit", eval_seed=3))
    try:
        run_a17.check_gate(dict(role="audit", eval_seed=20261281))
        assert False
    except SystemExit:
        pass


def test_main_role_needs_the_main_seed_even_with_a_gate(tmp_path, monkeypatch):
    import run_a17
    monkeypatch.setattr(run_a17, "ROOT", str(tmp_path))
    monkeypatch.setattr(run_a17, "code_hash", lambda: "x")
    os.makedirs(tmp_path / "results")
    json.dump({"code_hash": "x"}, open(tmp_path / "results" / "a17_gate.json", "w"))
    run_a17.check_gate(dict(role="main", eval_seed=20261281))
    try:
        run_a17.check_gate(dict(role="main", eval_seed=1))
        assert False
    except SystemExit:
        pass
    json.dump({"code_hash": "y"}, open(tmp_path / "results" / "a17_gate.json", "w"))
    try:
        run_a17.check_gate(dict(role="main", eval_seed=20261281))
        assert False
    except SystemExit:
        pass
