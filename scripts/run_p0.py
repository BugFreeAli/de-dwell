#!/usr/bin/env python3
"""
Run the P0 regret grid: for every cell (rho, s, a) and instance, compute
  V_dp   exact DP optimum (information-matched: the policy sees only the observable state),
  V_orc  class-aware oracle (upper bound, not implementable),
  V_<rule> exact expected value of each simple rule (world enumeration, no Monte Carlo),
  V_dpsim exact value of the extracted DP policy (self-check, must equal V_dp).
Resumable at cell level: a finished cell is written atomically to <out>/cell_*.csv and skipped on re-run.
Invariants are checked on every instance and the run FAILS LOUDLY on any violation:
  V_orc >= V_dp >= V_rule >= 0  and  V_dpsim == V_dp  (tolerance 1e-9).

Usage:
  python scripts/run_p0.py --config configs/p0_main.json --out results/p0_main --workers 4
  python scripts/run_p0.py --config configs/p0_main.json --out results/p0_main --shard 0/2   # laptop 1 of 2
"""
import argparse, itertools, json, os, platform, subprocess, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from dwell.dp import solve
from dwell.instances import make_instance
from dwell.m0 import Model, evaluate_policy
from dwell.oracle import oracle_value
from dwell.rules import SIMPLE_RULES, make_rule

TOL = 1e-9


def cell_name(rho, s, a):
    return f"cell_rho{rho}_s{s}_a{a}.csv"


def run_cell(job):
    """Compute all instances of one cell and write its CSV atomically."""
    cfg, out_dir, rho, s, a = job
    path = os.path.join(out_dir, cell_name(rho, s, a))
    if os.path.exists(path):                              # resumable: skip finished cells
        return path, 0.0, "skipped"
    t_cell = time.time()
    rows = []
    for idx in range(cfg["n_instances"]):
        inst, real = make_instance(cfg["master_seed"], idx, cfg["n"], rho, s, a,
                                   D=tuple(cfg["D"]), value_set=tuple(cfg["value_set"]),
                                   u_low=cfg["u_low"])
        model = Model(inst)
        t0 = time.time()
        res = solve(model)
        t_dp = time.time() - t0
        row = dict(rho=rho, s=s, a=a, idx=idx, n=cfg["n"], total_value=sum(inst.values),
                   V_dp=res.value, V_orc=oracle_value(model),
                   V_dpsim=evaluate_policy(model, res.policy()),
                   n_states=res.n_states, t_dp=t_dp, **real)
        for r in SIMPLE_RULES:
            row["V_" + r] = evaluate_policy(model, make_rule(model, r))
        # ---- invariants: fail loudly --------------------------------------------------------
        ctx = f"cell rho={rho} s={s} a={a} idx={idx}"
        assert abs(row["V_dpsim"] - row["V_dp"]) < TOL, f"DP policy value != DP value ({ctx})"
        assert row["V_orc"] >= row["V_dp"] - TOL, f"oracle < DP ({ctx})"
        for r in SIMPLE_RULES:
            assert -TOL <= row["V_" + r] <= row["V_dp"] + TOL, f"rule {r} outside [0, V_dp] ({ctx}): {row['V_'+r]} vs {row['V_dp']}"
        rows.append(row)
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)                       # atomic rename
    return path, time.time() - t_cell, "done"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown (not a git repository)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", default="0/1", help="i/m: run cells with (cell index %% m) == i")
    ap.add_argument("--max-cells", type=int, default=None, help="smoke test: only the first k cells of this shard")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    os.makedirs(args.out, exist_ok=True)
    i, m = map(int, args.shard.split("/"))
    cells = list(itertools.product(cfg["rho_grid"], cfg["s_grid"], cfg["a_grid"]))
    cells = [c for k, c in enumerate(cells) if k % m == i]
    if args.max_cells:
        cells = cells[: args.max_cells]
    meta = dict(config=cfg, config_file=args.config, git_commit=git_commit(), python=sys.version,
                numpy=np.__version__, pandas=pd.__version__, platform=platform.platform(),
                machine=platform.machine(), cpu_count=os.cpu_count(), workers=args.workers,
                shard=args.shard, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(meta, open(os.path.join(args.out, f"run_meta_shard{i}of{m}.json"), "w"), indent=2)
    jobs = [(cfg, args.out, rho, s, a) for rho, s, a in cells]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, (path, dt, status) in enumerate(pool.imap_unordered(run_cell, jobs), 1):
            print(f"[{k}/{len(jobs)}] {status:7s} {os.path.basename(path)}  {dt:7.1f}s  elapsed {time.time()-t0:8.1f}s", flush=True)
    print("finished shard", args.shard, "in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
