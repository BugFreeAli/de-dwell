#!/usr/bin/env python3
"""
Run an A12 grid (plan A12.5, A12.6). One CSV per cell (rho, s, a), resumable (finished cell files are kept).
Per instance: the optimum of every strategy and the value of every defender under every strategy (src/dwell/hard.py),
with the invariants of A12.6 checked in every row (violations are written to the row, never dropped; the run exits 1).

Roles (config "role"): audit may use any seed except the main seeds; "main" and "noise2" require the gate file
results/a12_gate.json (tests, mutations, audit) written for the SAME code hash, and their fixed master seeds.
Usage: python scripts/run_a12.py --config configs/a12_audit.json [--workers 4]
"""
import argparse, hashlib, itertools, json, os, platform, sys, time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.setrecursionlimit(50000)

import numpy as np
import pandas as pd

from dwell.hard import run_instance

MAIN_SEEDS = {"main": 20261228, "noise2": 20261229}
CODE_FILES = ["src/dwell/hard.py", "scripts/run_a12.py"]


def code_hash():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(open(os.path.join(ROOT, f), "rb").read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def run_cell(job):
    cfg, out_dir, rho, s, a = job
    path = os.path.join(out_dir, f"cell_rho{rho}_s{s}_a{a}.csv")
    if os.path.exists(path):
        return path, 0.0
    t0 = time.perf_counter()
    rows = [run_instance(cfg, rho, s, a, i) for i in range(cfg["n_instances"])]
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return path, time.perf_counter() - t0


def check_gate(cfg):
    role = cfg["role"]
    if role == "audit":
        if cfg["master_seed"] in MAIN_SEEDS.values():
            sys.exit("ERROR: a main seed may not be used for an audit.")
        return
    if role not in MAIN_SEEDS or cfg["master_seed"] != MAIN_SEEDS[role]:
        sys.exit(f"ERROR: role {role!r} requires master_seed {MAIN_SEEDS.get(role)}.")
    gp = os.path.join(ROOT, "results", "a12_gate.json")
    if not os.path.exists(gp):
        sys.exit("ERROR: no results/a12_gate.json. Run scripts/a12_gate.py first (tests, mutations, audit).")
    if json.load(open(gp)).get("code_hash") != code_hash():
        sys.exit("ERROR: code changed after the gate was written. Re-run tests, mutations, audit and a12_gate.py.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    check_gate(cfg)
    out_dir = os.path.join(ROOT, "results", cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(config=cfg, code_hash=code_hash(), python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__, platform=platform.platform(), cpu_count=os.cpu_count(), workers=args.workers,
                started=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(meta, open(os.path.join(out_dir, "run_meta.json"), "w"), indent=1)
    jobs = [(cfg, out_dir, rho, s, a) for rho, s, a in itertools.product(cfg["rho_grid"], cfg["s_grid"], cfg["a_grid"])]
    t0 = time.perf_counter()
    if args.workers > 1:
        with Pool(args.workers) as p:
            for path, dt in p.imap_unordered(run_cell, jobs):
                print(f"{os.path.basename(path)}  {dt:7.1f} s", flush=True)
    else:
        for j in jobs:
            path, dt = run_cell(j)
            print(f"{os.path.basename(path)}  {dt:7.1f} s", flush=True)
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("cell_") and f.endswith(".csv"))
    df = pd.concat([pd.read_csv(os.path.join(out_dir, f)) for f in files])
    nviol = int((df["violations"].fillna("") != "").sum())
    print(f"cells {len(jobs)}, rows {len(df)}, rows with violations {nviol}, wall {time.perf_counter() - t0:.1f} s")
    if nviol:
        print(df[df["violations"].fillna("") != ""][["rho", "s", "a", "idx", "violations"]].head(20).to_string())
        sys.exit(1)


if __name__ == "__main__":
    main()
