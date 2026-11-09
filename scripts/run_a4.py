#!/usr/bin/env python3
import argparse, dataclasses, itertools, json, os, platform, subprocess, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from dwell.dp import solve
from dwell.instances import make_instance_k
from dwell.m2 import Model2, evaluate_policy2
from dwell.oracle import oracle_value
from dwell.rules import make_rule

TOL = 1e-9
METHODS = ("DPMIS", "CEP", "ROLL_CEP", "ROLL_GMR", "GMR", "EDF")


def believed_instance(inst_true, a_b):
    K = len(inst_true.D)
    rows = []
    for row in inst_true.prior:
        label = max(range(K), key=lambda k: row[k])
        rows.append(tuple(a_b if k == label else (1.0 - a_b) / (K - 1) for k in range(K)))
    return dataclasses.replace(inst_true, prior=tuple(rows))


def policy_for(method, model_b, dp_b):
    if method == "DPMIS":
        return dp_b.policy()
    return make_rule(model_b, method)


def run_cell(job):
    cfg, out_dir, rho, s, a_true = job
    path = os.path.join(out_dir, f"cell_rho{rho}_s{s}_at{a_true}.csv")
    if os.path.exists(path):
        return path, 0.0, "skipped"
    t_cell = time.time()
    rows = []
    for idx in range(cfg["n_instances"]):
        inst, real = make_instance_k(cfg["master_seed"], idx, cfg["n"], rho, s, a_true,
                                     D=tuple(cfg["D"]), value_set=tuple(cfg["value_set"]), u_low=cfg["u_low"])
        model_t = Model2(inst, cfg.get("kappa", 1.0))
        dp_t = solve(model_t)
        row = dict(rho=rho, s=s, a_true=a_true, idx=idx, n=cfg["n"], total_value=sum(inst.values),
                   V_dp=dp_t.value, V_orc=oracle_value(model_t), n_states=dp_t.n_states, **real)
        for a_b in cfg["belief_grid"]:
            if abs(a_b - a_true) < 1e-12:
                model_b, dp_b = model_t, dp_t
            else:
                model_b = Model2(believed_instance(inst, a_b), cfg.get("kappa", 1.0))
                dp_b = solve(model_b)
            for m in METHODS:
                v = evaluate_policy2(model_t, policy_for(m, model_b, dp_b))
                ctx = f"cell rho={rho} s={s} at={a_true} idx={idx} belief={a_b} method={m}"
                assert -TOL <= v <= dp_t.value + TOL, f"value outside [0, V_dp] ({ctx}): {v} vs {dp_t.value}"
                if m == "DPMIS" and abs(a_b - a_true) < 1e-12:
                    assert abs(v - dp_t.value) < TOL, f"DP with correct belief differs from DP value ({ctx})"
                row[f"V_{m}_b{a_b}"] = v
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
    cells = list(itertools.product(cfg["rho_grid"], cfg["s_grid"], cfg["a_true_grid"]))
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
