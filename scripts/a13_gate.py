#!/usr/bin/env python3
"""
Writes results/a13_gate.json only if (plan A13.6): all A13 tests pass, all mutations are caught, and both audits
(results/a13_audit and results/a13_ctrl_audit, exploratory seed, all 32 cells) are complete with no invariant violation
and the SAME code hash. scripts/run_a13.py refuses the roles main and ctrl without this file for the current code.
Usage: python scripts/a13_gate.py
"""
import glob, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pandas as pd
from run_a13 import MAIN_SEED, code_hash


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1]


def check_audit(name):
    d = os.path.join(ROOT, "results", name)
    meta = json.load(open(os.path.join(d, "run_meta.json")))
    cfg = meta["config"]
    files = sorted(glob.glob(os.path.join(d, "cell_*.csv")))
    if len(files) != 32:
        sys.exit(f"GATE CLOSED: {name} has {len(files)} cells, expected 32")
    df = pd.concat([pd.read_csv(f) for f in files])
    if len(df) != 32 * cfg["n_instances"] or (df["violations"].fillna("") != "").any():
        sys.exit(f"GATE CLOSED: {name} incomplete or with violations")
    if meta["code_hash"] != code_hash() or cfg["master_seed"] == MAIN_SEED:
        sys.exit(f"GATE CLOSED: {name} was run with other code or with the main seed")
    return dict(cells=32, rows=len(df), seed=cfg["master_seed"])


def main():
    rc, tests = run([sys.executable, "-m", "pytest", "tests/test_a13.py", "tests/test_a13_analysis.py", "-q", "-p", "no:cacheprovider"])
    if rc != 0:
        sys.exit(f"GATE CLOSED: tests failed: {tests}")
    rc, muts = run([sys.executable, "scripts/mutate_a13.py"])
    if rc != 0:
        sys.exit(f"GATE CLOSED: a mutation survived: {muts}")
    gate = dict(code_hash=code_hash(), written=time.strftime("%Y-%m-%d %H:%M:%S"), tests=tests, mutations=muts,
                audit=check_audit("a13_audit"), ctrl_audit=check_audit("a13_ctrl_audit"))
    json.dump(gate, open(os.path.join(ROOT, "results", "a13_gate.json"), "w"), indent=1)
    print("GATE OPEN:", json.dumps(gate))


if __name__ == "__main__":
    main()
