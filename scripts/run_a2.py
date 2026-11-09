#!/usr/bin/env python3
import argparse, itertools, json, os, platform, subprocess, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from dwell.dp import solve
from dwell.instances import make_instance_k
from dwell.m2 import Model2, evaluate_policy2
from dwell.oracle import oracle_value
from dwell.rules import SIMPLE_RULES, make_rule

TOL = 1e-9


def run_cell(job):
    cfg, out_dir, rho, s, a = job
    rules = tuple(SIMPLE_RULES) + tuple(cfg.get("extra_rules", []))
    path = os.path.join(out_dir, f"cell_rho{rho}_s{s}_a{a}.csv")
    if os.path.exists(path):
        return path, 0.0, "skipped"
    t_cell = time.time()
    rows = []
    for idx in range(cfg["n_instances"]):
        inst, real = make_instance_k(cfg["master_seed"], idx, cfg["n"], rho, s, a,
                                     D=tuple(cfg["D"]), value_set=tuple(cfg["value_set"]), u_low=cfg["u_low"])
        model = Model2(inst, cfg.get("kappa", 1.0))
        t0 = time.time()
        res = solve(model)
        t_dp = time.time() - t0
        row = dict(rho=rho, s=s, a=a, idx=idx, n=cfg["n"], total_value=sum(inst.values),
                   V_dp=res.value, V_orc=oracle_value(model),
                   V_dpsim=evaluate_policy2(model, res.policy()),
                   n_states=res.n_states, t_dp=t_dp, **real)
        for r in rules:
            row["V_" + r] = evaluate_policy2(model, make_rule(model, r))
        ctx = f"cell rho={rho} s={s} a={a} idx={idx}"
        assert abs(row["V_dpsim"] - row["V_dp"]) < TOL, f"DP policy value != DP value ({ctx})"
        assert row["V_orc"] >= row["V_dp"] - TOL, f"oracle < DP ({ctx})"
        for r in rules:
            assert -TOL <= row["V_" + r] <= row["V_dp"] + TOL, f"rule {r} outside [0, V_dp] ({ctx}): {row['V_'+r]} vs {row['V_dp']}"
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
    cells = list(itertools.product(cfg["rho_grid"], cfg["s_grid"], cfg["a_grid"]))
    if args.max_cells:
        cells = cells[: args.max_cells]
    meta = dict(config=cfg, config_file=args.config, git_commit=git_commit(), python=sys.version,
                numpy=np.__version__, pandas=pd.__version__, platform=platform.platform(),
                cpu_count=os.cpu_count(), workers=args.workers, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(meta, open(os.path.join(args.out, "run_meta_shard0of1.json"), "w"), indent=2)
    jobs = [(cfg, args.out, rho, s, a) for rho, s, a in cells]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, (path, dt, status) in enumerate(pool.imap_unordered(run_cell, jobs), 1):
            print(f"[{k}/{len(jobs)}] {status:7s} {os.path.basename(path)}  {dt:7.1f}s  elapsed {time.time()-t0:8.1f}s", flush=True)
    print("finished in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
