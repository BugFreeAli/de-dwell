#!/usr/bin/env python3
"""
Run an A19 grid (plan A19.3, A19.6): learning the dwell scale during the episode. One CSV per cell, resumable.
Roles: audit (any eval seed except the main and pilot seeds), pilot (eval seed 20261302, n = 5 only) and main (eval seed 20261301); pilot and main are refused unless results/a19_gate.json carries the current code hash.
  python scripts/run_a19.py --config configs/a19_main.json --workers 4
Level per size: exact (n <= 5: also the exact goal optimum of the true model) and ru otherwise; every row has NOM, CLAMP, ADAPT, ADAPT_V and the informed
references CEPS_T and RU_CEP_T (src/dwell/a19.py).
"""
import argparse, glob, hashlib, itertools, json, os, platform, sys, time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.setrecursionlimit(200000)
import numpy as np
import pandas as pd

from dwell.a19 import eval_row

MAIN_SEED, PILOT_SEED = 20261301, 20261302
CODE_FILES = ["src/dwell/a19.py", "src/dwell/a18.py", "src/dwell/analysis_a16.py", "src/dwell/analysis_a19.py", "src/dwell/m0u.py", "src/dwell/m0.py",
              "src/dwell/dp.py", "src/dwell/rules.py", "src/dwell/instances.py", "scripts/run_a19.py"]


def code_hash():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(open(os.path.join(ROOT, f), "rb").read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def check_gate(cfg):
    if cfg["role"] == "audit":
        if cfg["eval_seed"] in (MAIN_SEED, PILOT_SEED):
            sys.exit("ERROR: the main and pilot seeds may not be used for an audit.")
        return
    want = {"main": MAIN_SEED, "pilot": PILOT_SEED}
    if cfg["role"] not in want or cfg["eval_seed"] != want[cfg["role"]]:
        sys.exit(f"ERROR: role {cfg['role']} requires its own eval seed ({want.get(cfg['role'])}).")
    gp = os.path.join(ROOT, "results", "a19_gate.json")
    if not os.path.exists(gp) or json.load(open(gp)).get("code_hash") != code_hash():
        sys.exit("ERROR: no gate for the current code. Run scripts/a19_gate.py (tests, mutations, audit).")


def jobs_of(cfg):
    J = []
    for sz in cfg["sizes"]:
        for sc in cfg["scales"]:
            for rho, s, a, lam in itertools.product(cfg["rho_grid"], cfg["s_grid"], cfg["a_grid"], cfg["lam_grid"]):
                J.append((cfg, sz["n"], sc, rho, s, a, lam, sz["instances"]))
    return J


def run_cell(job):
    cfg, n, sc, rho, s, a, lam, N = job
    d = os.path.join(ROOT, "results", cfg["name"]); os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"cell_n{n}_sc{sc}_rho{rho}_s{s}_a{a}_l{lam}.csv")
    if os.path.exists(path):
        return path, 0.0
    t0 = time.perf_counter()
    level = "exact" if n <= 5 else "ru"
    rows = [eval_row(cfg, n, rho, s, a, lam, sc, i, level) for i in range(N)]
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
    J = sorted(jobs_of(cfg), key=lambda j: -(j[1] * 1000 + j[7]))      # largest sizes first
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
        print(df[df["violations"].fillna("") != ""][["n", "scale", "rho", "s", "a", "lam", "idx", "violations"]].head(20).to_string()); sys.exit(1)


if __name__ == "__main__":
    main()
