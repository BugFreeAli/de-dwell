#!/usr/bin/env python3
"""
Writes results/a15_gate.json only if (plan A15.7): all A15 tests pass, all mutations are caught, and the audit pipeline
(results/a15_audit: pool, train, eval and time stages on exploratory seeds) is complete with no violation and the SAME code hash.
scripts/run_a15.py refuses the role main without this file for the current code.   Usage: python scripts/a15_gate.py
"""
import glob, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pandas as pd
from run_a15 import MAIN, code_hash


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1]


def check_audit(name="a15_audit"):
    d = os.path.join(ROOT, "results", name)
    cfg = None
    for stage in ("pool", "train", "eval"):
        p = os.path.join(d, f"run_meta_{stage}.json")
        if not os.path.exists(p):
            sys.exit(f"GATE CLOSED: {name} has no {stage} stage")
        meta = json.load(open(p)); cfg = meta["config"]
        if meta["code_hash"] != code_hash():
            sys.exit(f"GATE CLOSED: stage {stage} of {name} was run with other code")
    if cfg["pool_seed"] in MAIN.values() or cfg["eval_seed"] in MAIN.values():
        sys.exit("GATE CLOSED: the audit used a main seed")
    n_sh = -(-cfg["pool_instances"] // cfg["shard_size"])
    n_shD = -(-cfg["distill_instances"] // cfg["shard_size"])
    n_models = len(glob.glob(os.path.join(d, "models", "*.npz")))
    cells = len(glob.glob(os.path.join(d, "eval", "cell_*.csv")))
    want_cells = len(cfg["eval_sizes"]) * len(cfg["eval_rho"]) * len(cfg["eval_s"]) * len(cfg["eval_a"]) * len(cfg["eval_lam"])
    if (len(glob.glob(os.path.join(d, "pool_shards", "*.npz"))) != n_sh or len(glob.glob(os.path.join(d, "poolD_shards", "*.npz"))) != n_shD
            or n_models != 5 * cfg["n_seeds"] or cells != want_cells):
        sys.exit("GATE CLOSED: the audit is incomplete (pool shards, models or cells)")
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(d, "eval", "cell_*.csv"))])
    if (df["violations"].fillna("") != "").any() or not os.path.exists(os.path.join(d, "time_a15.json")):
        sys.exit("GATE CLOSED: violations in the audit rows or no time stage")
    return dict(pool_shards=n_sh, pool_d_shards=n_shD, models=n_models, cells=cells, rows=len(df), pool_seed=cfg["pool_seed"], eval_seed=cfg["eval_seed"])


def main():
    rc, tests = run([sys.executable, "-m", "pytest", "tests/test_a15.py", "tests/test_a15_analysis.py", "-q", "-p", "no:cacheprovider"])
    if rc != 0:
        sys.exit(f"GATE CLOSED: tests failed: {tests}")
    rc, muts = run([sys.executable, "scripts/mutate_a15.py"])
    if rc != 0:
        sys.exit(f"GATE CLOSED: a mutation survived: {muts}")
    gate = dict(code_hash=code_hash(), written=time.strftime("%Y-%m-%d %H:%M:%S"), tests=tests, mutations=muts, audit=check_audit())
    json.dump(gate, open(os.path.join(ROOT, "results", "a15_gate.json"), "w"), indent=1)
    print("GATE OPEN:", json.dumps(gate))


if __name__ == "__main__":
    main()
