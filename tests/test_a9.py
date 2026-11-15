import json
import os
import subprocess
import sys
from math import comb

import numpy as np
import pandas as pd
import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "analyze_a9.py")
RUNS = {
    "a9_m1": ([-1, 0, 1], [0.25, 0.5, 0.25], [0.6, 0.8, 1.2, 1.4]),
    "a9_m2": ([-2, -1, 0, 1, 2], [comb(4, k) / 16 for k in range(5)], [0.6, 0.8, 1.2, 1.4]),
    "a9_m3": ([-3, -2, -1, 0, 1, 2, 3], [comb(6, k) / 64 for k in range(7)], [0.8, 1.2, 1.4]),
    "a9_x2": ([-2, 0, 2], [0.25, 0.5, 0.25], [0.6, 0.8, 1.2, 1.4]),
}


def sd(offs, probs):
    o, p = np.array(offs, float), np.array(probs, float)
    return float(np.sqrt((p * o * o).sum() - (p * o).sum() ** 2))


def build(tmp_path, regret_fn, runs=RUNS, n_cells=True):
    dirs = []
    for name, (offs, probs, truths) in runs.items():
        d = tmp_path / name
        d.mkdir()
        cfg = dict(name=name, D0=[5, 10], noise_offsets=offs, noise_probs=probs, truths=truths, rho_grid=[1.1, 1.5],
                   s_grid=[2], a_grid=[0.5], n_instances=4)
        json.dump(dict(config=cfg), open(d / "run_meta.json", "w"))
        s = sd(offs, probs)
        for rho in cfg["rho_grid"][: (2 if n_cells else 1)]:
            rows = []
            for idx in range(4):
                r = dict(rho=rho, s=2, a=0.5, idx=idx)
                for t in truths:
                    R = regret_fn(abs(t - 1.0) * 5, s)
                    r[f"Vcal_{t}"] = 10.0
                    r[f"Vnom_DPMIS_t{t}"] = 10.0 * (1 - R)
                    r[f"Vnom_CEP_t{t}"] = 10.0 * (1 - R - 0.05)
                    r[f"Vnom_ROLL_CEP_t{t}"] = 10.0 * (1 - R - 0.03)
                rows.append(r)
            pd.DataFrame(rows).to_csv(d / f"cell_rho{rho}_s2_a0.5.csv", index=False)
        dirs.append(str(d))
    return dirs


def run(dirs, out):
    return subprocess.run([sys.executable, SCRIPT, "--dirs"] + dirs + ["--out", str(out)], capture_output=True, text=True)


def verdict(tmp_path, fn):
    dirs = build(tmp_path, fn)
    r = run(dirs, tmp_path / "dec.json")
    assert r.returncode == 0, r.stderr
    return json.load(open(tmp_path / "dec.json"))


def test_table_values_and_ratio(tmp_path):
    d = verdict(tmp_path, lambda delta, s: 0.01 * delta / s)
    tab = pd.DataFrame(d["table"])
    row = tab[(tab.run == "a9_m1") & (tab.truth == 1.4)].iloc[0]
    assert row.sigma == pytest.approx(0.5 ** 0.5)
    assert row.rho == pytest.approx(2 / 0.5 ** 0.5)
    assert row.regret == pytest.approx(0.01 * 2 / 0.5 ** 0.5)
    assert row.method == "DPMIS"
    assert row.side == "up" and row.delta == 2.0


def test_supported_when_loss_depends_only_on_ratio(tmp_path):
    d = verdict(tmp_path, lambda delta, s: 0.02 * (delta / s) ** 2 / (1 + 0.05 * (delta / s) ** 2))
    c = d["criteria"]
    assert c["C1_noise_scaling_at_one_step_shift"]["ok"]
    assert c["C2_monotone_in_ratio"]["ok"]
    assert c["C3_equal_ratio_pair"]["ok"]
    assert c["outcome"].startswith("SUPPORTED")


def test_noise_law_only_when_shift_enters_separately(tmp_path):
    d = verdict(tmp_path, lambda delta, s: 0.02 * delta / s ** 2)
    c = d["criteria"]
    assert c["C1_noise_scaling_at_one_step_shift"]["ok"]
    assert not c["C2_monotone_in_ratio"]["ok"]
    assert not c["C3_equal_ratio_pair"]["ok"]
    assert c["outcome"].startswith("NOISE-LAW-ONLY")


def test_not_supported_when_loss_ignores_noise(tmp_path):
    d = verdict(tmp_path, lambda delta, s: 0.02 * delta)
    c = d["criteria"]
    assert not c["C1_noise_scaling_at_one_step_shift"]["ok"]
    assert c["outcome"].startswith("NOT-SUPPORTED")


def test_c1_tolerance_boundary(tmp_path):
    base = lambda delta, s: 0.02 * (1 + (0.19 if s > 1.2 else 0.0)) * delta / s ** 2
    assert verdict(tmp_path, base)["criteria"]["C1_noise_scaling_at_one_step_shift"]["ok"]


def test_c1_tolerance_exceeded(tmp_path):
    fn = lambda delta, s: 0.02 * (1 + (0.6 if s > 1.2 else 0.0)) * delta / s ** 2
    assert not verdict(tmp_path, fn)["criteria"]["C1_noise_scaling_at_one_step_shift"]["ok"]


def test_missing_cell_file_stops_with_error(tmp_path):
    dirs = build(tmp_path, lambda delta, s: 0.01, n_cells=False)
    r = run(dirs, tmp_path / "dec.json")
    assert r.returncode != 0
    assert "expected" in r.stderr or "expected" in r.stdout


def test_best_method_is_chosen_per_truth(tmp_path):
    dirs = build(tmp_path, lambda delta, s: 0.02 * delta / s)
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith(".csv"):
                df = pd.read_csv(os.path.join(d, f))
                for t in (1.2, 1.4):
                    if f"Vcal_{t}" in df:
                        df[f"Vnom_ROLL_CEP_t{t}"] = df[f"Vcal_{t}"] * (1 - 0.001)
                df.to_csv(os.path.join(d, f), index=False)
    r = run(dirs, tmp_path / "dec.json")
    tab = pd.DataFrame(json.load(open(tmp_path / "dec.json"))["table"])
    up = tab[tab.side == "up"]
    assert (up.method == "ROLL_CEP").all()
    assert up.regret.max() == pytest.approx(0.001)
