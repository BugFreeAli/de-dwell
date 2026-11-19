#!/usr/bin/env python3
"""
Writes results/a12_gate.json only if (plan A12.6): all A12 tests pass, all mutations are caught, and the audit
(results/a12_audit, exploratory seed, all 8 cells) is complete with no invariant violation and the SAME code hash.
scripts/run_a12.py refuses the roles main and noise2 without this file for the current code.
Usage: python scripts/a12_gate.py
"""
import glob, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pandas as pd
from run_a12 import MAIN_SEEDS, code_hash


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1]


def main():
    rc, tests = run([sys.executable, "-m", "pytest", "tests/test_a12.py", "tests/test_a12_analysis.py", "-q", "-p", "no:cacheprovider"])
    if rc != 0:
        sys.exit(f"GATE CLOSED: tests failed: {tests}")
    rc, muts = run([sys.executable, "scripts/mutate_a12.py"])
    if rc != 0:
        sys.exit(f"GATE CLOSED: a mutation survived: {muts}")
    d = os.path.join(ROOT, "results", "a12_audit")
    meta = json.load(open(os.path.join(d, "run_meta.json")))
    cfg = meta["config"]
    files = sorted(glob.glob(os.path.join(d, "cell_*.csv")))
    if len(files) != 8:
        sys.exit(f"GATE CLOSED: audit has {len(files)} cells, expected 8")
    df = pd.concat([pd.read_csv(f) for f in files])
    if len(df) != 8 * cfg["n_instances"] or (df["violations"].fillna("") != "").any():
        sys.exit("GATE CLOSED: audit incomplete or with violations")
    if meta["code_hash"] != code_hash() or cfg["master_seed"] in MAIN_SEEDS.values():
        sys.exit("GATE CLOSED: audit was run with other code or with a main seed")
    gate = dict(code_hash=code_hash(), written=time.strftime("%Y-%m-%d %H:%M:%S"), tests=tests, mutations=muts,
                audit=dict(cells=8, rows=len(df), seed=cfg["master_seed"]))
    json.dump(gate, open(os.path.join(ROOT, "results", "a12_gate.json"), "w"), indent=1)
    print("GATE OPEN:", json.dumps(gate))


if __name__ == "__main__":
    main()
