#!/usr/bin/env python3
"""
Gate before the A11 main run (plan A11.6). Writes results/a11_gate.json only if ALL of these hold for the CURRENT code:
  1. tests/test_a11.py and tests/test_a11_analysis.py pass (includes the constant-rate regression tests);
  2. all mutations of scripts/mutate_a11.py are caught;
  3. results/a11_audit and results/a11_audit_ctrl are complete (all cells, n_instances rows each), were run with the
     invariant checks (columns V_orc and V_dp_policy present) on a non-main seed, and have no violation in any row;
  4. the audit was run with the same code hash as now.
scripts/run_a11.py refuses the main run without this file or when the code hash differs.
"""
import glob, json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "src"))
import pandas as pd
from run_a11 import MAIN_SEED, code_hash


def fail(msg):
    print("GATE CLOSED:", msg); sys.exit(1)


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1]


def check_audit(name, expect_cells):
    d = os.path.join(ROOT, "results", name)
    if not os.path.exists(os.path.join(d, "run_meta.json")):
        fail(f"{name}: no run_meta.json")
    meta = json.load(open(os.path.join(d, "run_meta.json"))); cfg = meta["config"]
    if cfg["master_seed"] == MAIN_SEED or cfg["role"] != "audit":
        fail(f"{name}: not an audit run on a non-main seed")
    if meta["code_hash"] != code_hash():
        fail(f"{name}: audit was run with different code; repeat the audit")
    files = glob.glob(os.path.join(d, "cell_*.csv"))
    if len(files) != expect_cells:
        fail(f"{name}: {len(files)} cells, expected {expect_cells}")
    df = pd.concat([pd.read_csv(f) for f in files])
    if len(df) != expect_cells * cfg["n_instances"]:
        fail(f"{name}: wrong number of rows")
    if "V_orc" not in df or "V_dp_policy" not in df or df["V_orc"].isna().any():
        fail(f"{name}: run without --audit")
    bad = int((df["violations"].fillna("") != "").sum())
    if bad:
        fail(f"{name}: {bad} rows with violations")
    return dict(cells=len(files), rows=len(df), seed=cfg["master_seed"])


def main():
    out = dict(code_hash=code_hash(), written=time.strftime("%Y-%m-%d %H:%M:%S"))
    rc, tail = run([sys.executable, "-m", "pytest", "tests/test_a11.py", "tests/test_a11_analysis.py", "-q", "-p", "no:cacheprovider"])
    out["tests"] = tail
    if rc:
        fail("tests: " + tail)
    rc, tail = run([sys.executable, "scripts/mutate_a11.py"])
    out["mutations"] = tail
    if rc:
        fail("mutations: " + tail)
    out["audit_main_grid"] = check_audit("a11_audit", 48)
    out["audit_control"] = check_audit("a11_audit_ctrl", 24)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "a11_gate.json"), "w"), indent=1)
    print("GATE OPEN:", json.dumps(out))


if __name__ == "__main__":
    main()
