#!/usr/bin/env python3
"""
Run an A13 grid (plan A13.3, A13.6, A13.7). One CSV per cell (n, rho, s_bar, a, geometry), resumable.
Per instance: DP optimum and exact value of the 11 methods under the slew matrix (src/dwell/m0g.py); invariants are
written to the row (violations are never dropped; the run exits 1). Roles (config "role"):
  audit, ctrl_audit : exploratory, any seed except the main seed; audit-only extras (policy value, oracle) are computed
  main, ctrl        : fixed seed 20261241, refused unless results/a13_gate.json carries the hash of the current code
"control": true replaces the matrix by the uniform slew s_bar on the same instances (A13.8).
Usage: python scripts/run_a13.py --config configs/a13_audit.json [--workers 4]
"""
import argparse, hashlib, itertools, json, os, platform, sys, time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.setrecursionlimit(50000)

import numpy as np
import pandas as pd

from dwell.m0g import run_instance

MAIN_SEED = 20261241
CODE_FILES = ["src/dwell/m0g.py", "scripts/run_a13.py"]


def code_hash():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(open(os.path.join(ROOT, f), "rb").read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def cell_path(out_dir, n, rho, s, a, geom):
    return os.path.join(out_dir, f"cell_n{n}_rho{rho}_s{s}_a{a}_{geom}.csv")


def run_cell(job):
    cfg, out_dir, n, rho, s, a, geom = job
    path = cell_path(out_dir, n, rho, s, a, geom)
    if os.path.exists(path):
        return path, 0.0
    t0 = time.perf_counter()
    audit = cfg["role"] in ("audit", "ctrl_audit")
    rows = [run_instance(cfg, n, rho, s, a, geom, i, control=cfg["control"], audit=audit) for i in range(cfg["n_instances"])]
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return path, time.perf_counter() - t0


def check_gate(cfg):
    role = cfg["role"]
    if role in ("audit", "ctrl_audit"):
        if cfg["master_seed"] == MAIN_SEED:
            sys.exit("ERROR: the main seed may not be used for an audit.")
        return
    if role not in ("main", "ctrl") or cfg["master_seed"] != MAIN_SEED:
        sys.exit(f"ERROR: role {role!r} requires master_seed {MAIN_SEED}.")
    gp = os.path.join(ROOT, "results", "a13_gate.json")
    if not os.path.exists(gp):
        sys.exit("ERROR: no results/a13_gate.json. Run scripts/a13_gate.py first (tests, mutations, audits).")
    if json.load(open(gp)).get("code_hash") != code_hash():
        sys.exit("ERROR: code changed after the gate was written. Re-run tests, mutations, audits and a13_gate.py.")


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
    jobs = [(cfg, out_dir, n, rho, s, a, g) for n, rho, s, a, g in
            itertools.product(cfg["n_grid"], cfg["rho_grid"], cfg["s_grid"], cfg["a_grid"], cfg["geom_grid"])]
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
        print(df[df["violations"].fillna("") != ""][["n", "rho", "s", "a", "geom", "idx", "violations"]].head(20).to_string())
        sys.exit(1)


if __name__ == "__main__":
    main()
