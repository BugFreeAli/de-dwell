#!/usr/bin/env python3
"""
Writes results/a16_gate.json only if (plan A16.7): all A16 tests pass, all mutations are caught, and the audit (results/a16_audit, exploratory
seed) is complete with no violation and the SAME code hash. scripts/run_a16.py refuses the role main without this file for the current code.
Usage: python scripts/a16_gate.py
"""
import glob, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pandas as pd
from run_a16 import MAIN_SEED, code_hash, jobs_of


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1]


def check_audit(name="a16_audit"):
    d = os.path.join(ROOT, "results", name)
    meta = json.load(open(os.path.join(d, "run_meta.json"))); cfg = meta["config"]
    if meta["code_hash"] != code_hash() or cfg["eval_seed"] == MAIN_SEED:
        sys.exit("GATE CLOSED: the audit was run with other code or with the main seed")
    files = sorted(glob.glob(os.path.join(d, "cell_*.csv")))
    if len(files) != len(jobs_of(cfg)):
        sys.exit(f"GATE CLOSED: {len(files)} audit cells, expected {len(jobs_of(cfg))}")
    df = pd.concat([pd.read_csv(f) for f in files])
    if (df["violations"].fillna("") != "").any():
        sys.exit("GATE CLOSED: violations in the audit rows")
    return dict(cells=len(files), rows=len(df), eval_seed=cfg["eval_seed"], sec_sum=round(float(df["sec"].sum()), 1))


def main():
    rc, tests = run([sys.executable, "-m", "pytest", "tests/test_a16.py", "tests/test_a16_analysis.py", "-q", "-p", "no:cacheprovider"])
    if rc != 0:
        sys.exit(f"GATE CLOSED: tests failed: {tests}")
    rc, muts = run([sys.executable, "scripts/mutate_a16.py"])
    if rc != 0:
        sys.exit(f"GATE CLOSED: a mutation survived: {muts}")
    gate = dict(code_hash=code_hash(), written=time.strftime("%Y-%m-%d %H:%M:%S"), tests=tests, mutations=muts, audit=check_audit())
    json.dump(gate, open(os.path.join(ROOT, "results", "a16_gate.json"), "w"), indent=1)
    print("GATE OPEN:", json.dumps(gate))


if __name__ == "__main__":
    main()
