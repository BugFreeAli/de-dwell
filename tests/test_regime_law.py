import json
import os
import subprocess
import sys
from math import comb

import numpy as np
import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "fit_regime_law.py")
SDS = {"SD10": 0.1, "SD14": 0.2 * (2 * 0.25) ** 0.5, "SD16": 0.2 * (2 / 3) ** 0.5, "SD20": 0.2}


def make(name, m, c, factor=1.0):
    offs = list(range(-m, m + 1))
    probs = [comb(2 * m, m + o) / 4 ** m for o in offs]
    s = float(np.sqrt(sum(p * o * o for o, p in zip(offs, probs))))
    d = {"name": name, "noise_offsets": offs, "noise_probs": probs, "scenarios": {}}
    for lab, sd in SDS.items():
        v = c * (sd / s) ** 2 * factor
        d["scenarios"][lab] = {"sd": sd, "nominal": v, "nominal_lo5": 0.95 * v, "nominal_hi95": 1.05 * v}
    return d


def run(args):
    return subprocess.run([sys.executable, SCRIPT] + args, capture_output=True, text=True)


def write(tmp_path, name, d):
    p = tmp_path / name
    p.write_text(json.dumps(d))
    return str(p)


def test_fit_recovers_coefficient_and_exponent(tmp_path):
    a = write(tmp_path, "a.json", make("a", 1, 0.5))
    b = write(tmp_path, "b.json", make("b", 2, 0.5))
    r = run(["--fit", a, b])
    assert r.returncode == 0
    assert "c = 50.00" in r.stdout
    assert "coefficient of a: 50.00" in r.stdout
    assert "coefficient of b: 50.00" in r.stdout


def test_check_holds_for_exact_prediction_and_fails_for_inflated(tmp_path):
    a = write(tmp_path, "a.json", make("a", 1, 0.5))
    b = write(tmp_path, "b.json", make("b", 2, 0.5))
    good = write(tmp_path, "g.json", make("g", 3, 0.5))
    bad = write(tmp_path, "x.json", make("x", 3, 0.5, 1.6))
    assert "verdict: HOLDS (4 of 4)" in run(["--fit", a, b, "--check", good]).stdout
    assert "verdict: FAILS (0 of 4)" in run(["--fit", a, b, "--check", bad]).stdout


def test_partly_when_one_level_is_outside(tmp_path):
    a = write(tmp_path, "a.json", make("a", 1, 0.5))
    b = write(tmp_path, "b.json", make("b", 2, 0.5))
    d = make("p", 3, 0.5)
    d["scenarios"]["SD20"]["nominal"] *= 2.0
    p = write(tmp_path, "p.json", d)
    assert "verdict: PARTLY (3 of 4)" in run(["--fit", a, b, "--check", p]).stdout


def test_prediction_matches_hand_computation(tmp_path):
    a = write(tmp_path, "a.json", make("a", 1, 0.5))
    b = write(tmp_path, "b.json", make("b", 2, 0.5))
    out = run(["--fit", a, b, "--predict-m", "3"]).stdout
    s3 = float(np.sqrt(sum(p * o * o for o, p in zip(range(-3, 4), [1 / 64, 6 / 64, 15 / 64, 20 / 64, 15 / 64, 6 / 64, 1 / 64]))))
    want = 100 * 0.5 * (0.2 / s3) ** 2
    assert f"predicted nominal regret {want:.2f}%" in out


def test_run_without_noise_is_rejected(tmp_path):
    d = make("z", 1, 0.5)
    d["noise_offsets"], d["noise_probs"] = [0], [1.0]
    p = write(tmp_path, "z.json", d)
    r = run(["--fit", p])
    assert r.returncode != 0
