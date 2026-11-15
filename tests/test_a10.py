import json
import os
import subprocess
import sys
from math import comb

import pandas as pd
import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "analyze_a10.py")


def build(tmp_path, e16, per_instance=None, drop=(), cells=2, asym=0.0):
    dirs = []
    for m in (1, 2):
        for n in (3, 4, 5):
            if (n, m) in drop:
                continue
            d = tmp_path / f"a10_n{n}_m{m}"
            d.mkdir()
            offs = list(range(-m, m + 1))
            cfg = dict(name=f"a10_n{n}_m{m}", n=n, noise_offsets=offs, noise_probs=[comb(2 * m, m + o) / 4 ** m for o in offs],
                       rho_grid=[1.1, 1.5], s_grid=[2], a_grid=[0.5], n_instances=4)
            json.dump(dict(config=cfg), open(d / "run_meta.json", "w"))
            r = 1.5 * e16[(n, m)]
            for rho in cfg["rho_grid"][:cells]:
                rows = []
                for idx in range(4):
                    f = 1.0 if per_instance is None else per_instance(idx)
                    row = dict(rho=rho, s=2, a=0.5, idx=idx)
                    for t, k in ((0.8, 1.0 - asym), (1.2, 1.0 + asym)):
                        row[f"Vcal_{t}"] = 10.0
                        row[f"Vnom_DPMIS_t{t}"] = 10.0 * (1 - r * k * f)
                        row[f"Vnom_CEP_t{t}"] = 10.0 * (1 - r * k * f - 0.05)
                        row[f"Vnom_ROLL_CEP_t{t}"] = 10.0 * (1 - r * k * f - 0.03)
                    rows.append(row)
                pd.DataFrame(rows).to_csv(d / f"cell_rho{rho}_s2_a0.5.csv", index=False)
            dirs.append(str(d))
    return dirs


def run(dirs, out):
    return subprocess.run([sys.executable, SCRIPT, "--dirs"] + dirs + ["--out", str(out)], capture_output=True, text=True)


def verdict(tmp_path, e16, **kw):
    dirs = build(tmp_path, e16, **kw)
    r = run(dirs, tmp_path / "dec.json")
    assert r.returncode == 0, r.stderr
    return json.load(open(tmp_path / "dec.json"))


def series(m1, m2):
    out = {}
    for i, n in enumerate((3, 4, 5)):
        out[(n, 1)] = m1[i]
        out[(n, 2)] = m2[i]
    return out


def test_table_values(tmp_path):
    d = verdict(tmp_path, series((0.04, 0.05, 0.06), (0.02, 0.03, 0.04)))
    row = [x for x in d["table"] if x["n"] == 4 and x["m"] == 2][0]
    assert row["R08"] == pytest.approx(0.045)
    assert row["R12"] == pytest.approx(0.045)
    assert row["E16"] == pytest.approx(0.03)
    assert row["instances"] == 8


def test_grows_material(tmp_path):
    d = verdict(tmp_path, series((0.04, 0.05, 0.06), (0.025, 0.03, 0.035)))
    assert d["criteria"]["m2"]["grows"] and d["criteria"]["m2"]["material_at_n5"]
    assert d["criteria"]["outcome"].startswith("GROWS-MATERIAL")


def test_grows_extrapolates_when_trend_reaches_three_percent_by_ten(tmp_path):
    d = verdict(tmp_path, series((0.04, 0.05, 0.06), (0.010, 0.015, 0.020)))
    c = d["criteria"]["m2"]
    assert c["grows"] and not c["material_at_n5"]
    assert c["n_star"] == pytest.approx(7.0)
    assert d["criteria"]["outcome"].startswith("GROWS-EXTRAPOLATES")


def test_flat_when_trend_reaches_three_percent_only_beyond_ten(tmp_path):
    d = verdict(tmp_path, series((0.04, 0.05, 0.06), (0.0, 0.003, 0.006)))
    assert d["criteria"]["m2"]["grows"]
    assert d["criteria"]["m2"]["n_star"] == pytest.approx(13.0)
    assert d["criteria"]["outcome"].startswith("FLAT-OR-FALLING")


def test_flat_when_loss_falls(tmp_path):
    d = verdict(tmp_path, series((0.05, 0.04, 0.03), (0.03, 0.025, 0.02)))
    assert not d["criteria"]["m2"]["grows"]
    assert d["criteria"]["outcome"].startswith("FLAT-OR-FALLING")


def test_not_monotone_is_not_growth(tmp_path):
    d = verdict(tmp_path, series((0.04, 0.05, 0.06), (0.02, 0.01, 0.03)))
    assert not d["criteria"]["m2"]["grows"]


def test_growth_threshold_boundary(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    below = verdict(tmp_path / "a", series((0.04, 0.05, 0.06), (0.020, 0.0225, 0.0249)))
    above = verdict(tmp_path / "b", series((0.04, 0.05, 0.06), (0.020, 0.0225, 0.0251)))
    assert not below["criteria"]["m2"]["grows"]
    assert above["criteria"]["m2"]["grows"]


def test_noisy_difference_fails_the_lower_bound(tmp_path):
    d = verdict(tmp_path, series((0.04, 0.05, 0.06), (0.020, 0.023, 0.026)), per_instance=lambda i: 2.0 if i % 2 == 0 else 0.0)
    c = d["criteria"]["m2"]
    assert c["difference_5_minus_3"] == pytest.approx(0.006)
    assert c["difference_lo5"] <= 0
    assert not c["grows"]


def test_materiality_needs_lower_bound(tmp_path):
    d = verdict(tmp_path, series((0.04, 0.05, 0.06), (0.025, 0.03, 0.032)), per_instance=lambda i: 2.0 if i % 2 == 0 else 0.0)
    assert d["criteria"]["m2"]["E16_n5_lo5"] < 0.02
    assert not d["criteria"]["m2"]["material_at_n5"]


def test_missing_run_gives_note_and_flat(tmp_path):
    d = verdict(tmp_path, series((0.04, 0.05, 0.06), (0.02, 0.03, 0.04)), drop={(5, 2)})
    assert "missing" in d["criteria"]["m2"]["note"]
    assert d["criteria"]["outcome"].startswith("FLAT-OR-FALLING")


def test_missing_cell_file_stops_with_error(tmp_path):
    dirs = build(tmp_path, series((0.04, 0.05, 0.06), (0.02, 0.03, 0.04)), cells=1)
    r = run(dirs, tmp_path / "dec.json")
    assert r.returncode != 0


def test_mixed_when_only_the_higher_noise_grows(tmp_path):
    d = verdict(tmp_path, series((0.06, 0.05, 0.04), (0.025, 0.03, 0.035)))
    assert d["criteria"]["m2"]["grows"] and d["criteria"]["m2"]["material_at_n5"] and not d["criteria"]["m1"]["grows"]
    assert d["criteria"]["outcome"].startswith("MIXED")


def test_asymmetric_truths_are_both_used(tmp_path):
    d = verdict(tmp_path, series((0.04, 0.05, 0.06), (0.02, 0.03, 0.04)), asym=0.4)
    row = [x for x in d["table"] if x["n"] == 5 and x["m"] == 2][0]
    assert row["R08"] == pytest.approx(0.06 * 0.6)
    assert row["R12"] == pytest.approx(0.06 * 1.4)
    assert row["E16"] == pytest.approx(0.04)
