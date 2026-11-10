#!/usr/bin/env python3
import argparse, itertools, json, os, platform, subprocess, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from dwell.dp import solve
from dwell.instances import make_instance_k
from dwell.m3 import Model3, evaluate_policy3
from dwell.oracle import oracle_value
from dwell.rules import make_rule

TOL = 1e-9
METHODS = ("CEP", "ROLL_CEP", "ROLL_GMR", "GMR", "EDF")


def run_cell(job):
    cfg, out_dir, rho, s, a, q = job
    path = os.path.join(out_dir, f"cell_rho{rho}_s{s}_a{a}_q{q}.csv")
    if os.path.exists(path):
        return path, 0.0, "skipped"
    t_cell = time.time()
    rows = []
    for idx in range(cfg["n_instances"]):
        inst, real = make_instance_k(cfg["master_seed"], idx, cfg["n"], rho, s, a,
                                     D=tuple(cfg["D"]), value_set=tuple(cfg["value_set"]), u_low=cfg["u_low"])
        model = Model3(inst, q)
        res = solve(model)
        row = dict(rho=rho, s=s, a=a, q=q, idx=idx, n=cfg["n"], total_value=sum(inst.values),
                   V_dp=res.value, V_orc=oracle_value(model), n_states=res.n_states, **real)
        assert abs(evaluate_policy3(model, res.policy()) - res.value) < TOL
        assert row["V_orc"] >= res.value - TOL
        for m in METHODS:
            v = evaluate_policy3(model, make_rule(model, m))
            assert -TOL <= v <= res.value + TOL, f"{m} outside [0, V_dp] rho={rho} s={s} a={a} q={q} idx={idx}: {v} vs {res.value}"
            row["V_" + m] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return path, time.time() - t_cell, "done"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-cells", type=int, default=None)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    os.makedirs(args.out, exist_ok=True)
    cells = list(itertools.product(cfg["rho_grid"], cfg["s_grid"], cfg["a_grid"], cfg["q_grid"]))
    if args.max_cells:
        cells = cells[: args.max_cells]
    meta = dict(config=cfg, config_file=args.config, git_commit=git_commit(), python=sys.version,
                numpy=np.__version__, pandas=pd.__version__, platform=platform.platform(),
                cpu_count=os.cpu_count(), workers=args.workers, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(meta, open(os.path.join(args.out, "run_meta_shard0of1.json"), "w"), indent=2)
    jobs = [(cfg, args.out, rho, s, a, q) for rho, s, a, q in cells]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, (path, dt, status) in enumerate(pool.imap_unordered(run_cell, jobs), 1):
            print(f"[{k}/{len(jobs)}] {status:7s} {os.path.basename(path)}  {dt:7.1f}s  elapsed {time.time()-t0:8.1f}s", flush=True)
    print("finished in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
