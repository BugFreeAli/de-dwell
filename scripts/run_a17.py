#!/usr/bin/env python3
"""
Run an A17 grid (plan A17.3, A17.6, A17.7): planner robustness across instance families. One CSV per cell, resumable.
Roles: audit (any eval seed except 20261281) and main (eval seed 20261281, refused unless results/a17_gate.json carries the current code hash).
  python scripts/run_a17.py --config configs/a17_main.json --workers 4
Every cell is evaluated at level "ru" (CEP, CEPS, RU_CEP) by src/dwell/a16.py.
"""
import argparse, glob, hashlib, itertools, json, os, platform, sys, time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.setrecursionlimit(200000)
import numpy as np
import pandas as pd

from dwell.a16 import eval_row

MAIN_SEED = 20261281
CODE_FILES = ["src/dwell/a16.py", "src/dwell/analysis_a16.py", "src/dwell/analysis_a17.py", "src/dwell/m0u.py", "src/dwell/m0.py",
              "src/dwell/dp.py", "src/dwell/rules.py", "src/dwell/instances.py", "scripts/run_a17.py"]


def code_hash():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(open(os.path.join(ROOT, f), "rb").read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def check_gate(cfg):
    if cfg["role"] == "audit":
        if cfg["eval_seed"] == MAIN_SEED:
            sys.exit("ERROR: the main seed may not be used for an audit.")
        return
    if cfg["role"] != "main" or cfg["eval_seed"] != MAIN_SEED:
        sys.exit(f"ERROR: role main requires eval_seed {MAIN_SEED}.")
    gp = os.path.join(ROOT, "results", "a17_gate.json")
    if not os.path.exists(gp) or json.load(open(gp)).get("code_hash") != code_hash():
        sys.exit("ERROR: no gate for the current code. Run scripts/a17_gate.py (tests, mutations, audit).")


def jobs_of(cfg):
    J = []
    for fam in cfg["families"]:
        for sz in fam["sizes"]:
            for rho, s, a, lam in itertools.product(cfg["rho_grid"], cfg["s_grid"], cfg["a_grid"], cfg["lam_grid"]):
                J.append((cfg, fam["name"], sz["n"], rho, s, a, lam, sz["instances"], fam["D0"], fam["value_set"]))
    return J


def run_cell(job):
    cfg, group, n, rho, s, a, lam, N, D0, vs = job
    d = os.path.join(ROOT, "results", cfg["name"]); os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"cell_{group}_n{n}_rho{rho}_s{s}_a{a}_l{lam}.csv")
    if os.path.exists(path):
        return path, 0.0
    t0 = time.perf_counter()
    rows = [eval_row(cfg, group, n, rho, s, a, lam, i, "ru", D0, vs) for i in range(N)]
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return path, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    cfg = json.load(open(a.config)); check_gate(cfg)
    d = os.path.join(ROOT, "results", cfg["name"]); os.makedirs(d, exist_ok=True)
    json.dump(dict(config=cfg, code_hash=code_hash(), python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                   platform=platform.platform(), cpu_count=os.cpu_count(), workers=a.workers, started=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(os.path.join(d, "run_meta.json"), "w"), indent=1)
    J = sorted(jobs_of(cfg), key=lambda j: -(j[2] * 1000 + j[7]))      # largest sizes first
    t0 = time.perf_counter()
    if a.workers > 1:
        with Pool(a.workers) as p:
            for path, dt in p.imap_unordered(run_cell, J):
                print(f"{os.path.basename(path)}  {dt:8.1f} s", flush=True)
    else:
        for j in J:
            path, dt = run_cell(j); print(f"{os.path.basename(path)}  {dt:8.1f} s", flush=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(d, "cell_*.csv")))])
    nviol = int((df["violations"].fillna("") != "").sum())
    print(f"cells {len(J)}, rows {len(df)}, rows with violations {nviol}, wall {time.perf_counter() - t0:.1f} s")
    if nviol:
        print(df[df["violations"].fillna("") != ""][["group", "n", "rho", "s", "a", "lam", "idx", "violations"]].head(20).to_string()); sys.exit(1)


if __name__ == "__main__":
    main()
