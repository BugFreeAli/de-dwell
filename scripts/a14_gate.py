#!/usr/bin/env python3
"""
Writes results/a14_gate.json only if (plan A14.7): all A14 tests pass, all mutations are caught, and the audit
(results/a14_audit, exploratory seed, all 24 cells) is complete with no invariant violation and the SAME code hash.
scripts/run_a14.py refuses the roles main and spot without this file for the current code.
Usage: python scripts/a14_gate.py
"""
import glob, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pandas as pd
from run_a14 import SEEDS, code_hash


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1]


def check_audit(name, n_cells=24):
    d = os.path.join(ROOT, "results", name)
    meta = json.load(open(os.path.join(d, "run_meta.json")))
    cfg = meta["config"]
    files = sorted(glob.glob(os.path.join(d, "cell_*.csv")))
    if len(files) != n_cells:
        sys.exit(f"GATE CLOSED: {name} has {len(files)} cells, expected {n_cells}")
    df = pd.concat([pd.read_csv(f) for f in files])
    if len(df) != n_cells * cfg["n_instances"] or (df["violations"].fillna("") != "").any():
        sys.exit(f"GATE CLOSED: {name} incomplete or with violations")
    if meta["code_hash"] != code_hash() or cfg["master_seed"] in SEEDS.values():
        sys.exit(f"GATE CLOSED: {name} was run with other code or with a main or spot seed")
    return dict(cells=n_cells, rows=len(df), seed=cfg["master_seed"], wall_sec_sum=round(float(df["sec"].sum()), 1))


def main():
    rc, tests = run([sys.executable, "-m", "pytest", "tests/test_a14.py", "tests/test_a14_analysis.py", "-q", "-p", "no:cacheprovider"])
    if rc != 0:
        sys.exit(f"GATE CLOSED: tests failed: {tests}")
    rc, muts = run([sys.executable, "scripts/mutate_a14.py"])
    if rc != 0:
        sys.exit(f"GATE CLOSED: a mutation survived: {muts}")
    gate = dict(code_hash=code_hash(), written=time.strftime("%Y-%m-%d %H:%M:%S"), tests=tests, mutations=muts,
                audit=check_audit("a14_audit"))
    json.dump(gate, open(os.path.join(ROOT, "results", "a14_gate.json"), "w"), indent=1)
    print("GATE OPEN:", json.dumps(gate))


if __name__ == "__main__":
    main()
