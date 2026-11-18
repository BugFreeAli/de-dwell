#!/usr/bin/env python3
"""
Run an A11 grid (plan A11.3, A11.4, A11.6, A11.8). One CSV per cell, resumable (existing cell files are kept).
Per instance: exact DP value (with idle and energy), the 22 methods, timings, regime variables; for the first
`n_stats` instances of a cell the value of idle and the wait/preempt shares of the optimal policy; with --audit also the
invariants of A11.6.2 (violations are written to the row, never dropped, and make the run exit with code 1).

Roles (config "role"): smoke / audit / probe may only use seeds other than the main seed; "main" requires the gate file
results/a11_gate.json written by scripts/a11_gate.py (tests, mutations and audit passed) for the SAME code hashes.
Usage: python scripts/run_a11.py --config configs/a11_audit.json [--audit] [--workers 4]
"""
import argparse, hashlib, json, os, sys, time, platform
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.setrecursionlimit(50000)

import numpy as np
import pandas as pd

from dwell.dp import solve
from dwell.instances import make_instance_k
from dwell.m0 import Model, evaluate_policy
from dwell.m1 import (Model1, energy_budget, evaluate_policy1, oracle_value1, rho_boost, solve1, two_level_gtab)
from dwell.rules import make_rule
from dwell.rules1 import METHODS, make_method, split

MAIN_SEED = 20261222
CODE_FILES = ["src/dwell/m1.py", "src/dwell/rules1.py", "scripts/run_a11.py"]
TOL = 1e-9


def code_hash():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(open(os.path.join(ROOT, f), "rb").read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def eps_key(e):
    return "inf" if e is None else repr(float(e))


def cell_name(rho, s, a, gamma, eps):
    return f"cell_rho{rho}_s{s}_a{a}_g{gamma}_e{eps_key(eps)}.csv"


def run_instance(cfg, rho, s, a, gamma, eps, idx, audit):
    D = tuple(cfg["D"])
    inst, real = make_instance_k(cfg["master_seed"], idx, cfg["n"], rho, s, a, D=D,
                                 value_set=tuple(cfg["value_set"]), u_low=cfg["u_low"])
    tau_c = D[-1]
    gtab = two_level_gtab(gamma, tau_c, max(inst.deadlines) + 2)
    B = energy_budget(inst, eps)
    model = Model1(inst, gtab, B)
    row = dict(idx=idx, rho=rho, s=s, a=a, gamma=gamma, eps=eps_key(eps), B=-1 if B is None else B,
               rho_hat=real["rho_hat"], rho_dbf=real["rho_dbf"], rho_boost=rho_boost(inst, gtab),
               d_max=real["d_max"], total_value=sum(inst.values))
    viol = []
    t0 = time.perf_counter(); dp = solve1(model); row["t_dp"] = time.perf_counter() - t0
    row["V_dp"], row["n_states"] = dp.value, dp.n_states
    vals = {}
    for name in METHODS:
        t0 = time.perf_counter()
        vals[name] = evaluate_policy1(model, make_method(model, name))
        row["t_" + name] = time.perf_counter() - t0
        row["V_" + name] = vals[name]
        if not (-TOL <= vals[name] <= dp.value + TOL):
            viol.append(f"range:{name}")
    if idx < cfg.get("n_stats", 0) or audit:
        v, pw, pp = evaluate_policy1(model, dp.policy(), stats=True)
        row["V_dp_policy"], row["p_wait_dp"], row["p_pre_dp"] = v, pw, pp
        row["V_noidle"] = solve1(Model1(inst, gtab, B, allow_idle=False)).value
        if abs(v - dp.value) > TOL:
            viol.append("dp_policy_value")
        if eps is None and pw > 0:
            viol.append("dp_waits_without_budget")
    if audit:
        row["V_orc"] = oracle_value1(model)
        if row["V_orc"] < dp.value - TOL:
            viol.append("oracle_below_dp")
        for r in ("CEP", "GMR", "EDF"):
            for var in ("M", "L"):
                if vals[f"ROLL_{r}_{var}"] < vals[f"{r}_{var}"] - TOL:
                    viol.append(f"rollout_below_base:{r}_{var}")
        if gamma == 1:
            m0 = Model(inst); noidle = Model1(inst, gtab, B, allow_idle=False)
            n_diff = 0
            for name in METHODS:
                rule, var = split(name)
                v0 = evaluate_policy(m0, make_rule(m0, rule))
                if name.startswith("ROLL"):
                    n_diff += abs(v0 - vals[name]) > TOL
                    if eps is None and abs(evaluate_policy1(noidle, make_method(noidle, name)) - v0) > TOL:
                        viol.append(f"m0_regression:{name}")
                elif eps is None and abs(v0 - vals[name]) > TOL:
                    viol.append(f"m0_regression:{name}")
            if eps is None and abs(solve(m0).value - dp.value) > TOL:
                viol.append("m0_regression:dp")
            row["n_roll_idle_differs_from_m0"] = int(n_diff)
    row["violations"] = ";".join(viol)
    return row


def run_cell(args):
    cfg, rho, s, a, gamma, eps, out_dir, audit = args
    path = os.path.join(out_dir, cell_name(rho, s, a, gamma, eps))
    if os.path.exists(path):
        return path, 0.0
    t0 = time.perf_counter()
    rows = [run_instance(cfg, rho, s, a, gamma, eps, i, audit) for i in range(cfg["n_instances"])]
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return path, time.perf_counter() - t0


def check_gate(cfg):
    role = cfg["role"]
    if role != "main":
        if cfg["master_seed"] == MAIN_SEED:
            sys.exit("ERROR: the main seed may only be used with role 'main'.")
        return
    if cfg["master_seed"] != MAIN_SEED:
        sys.exit("ERROR: role 'main' requires master_seed 20261222.")
    gp = os.path.join(ROOT, "results", "a11_gate.json")
    if not os.path.exists(gp):
        sys.exit("ERROR: no results/a11_gate.json. Run scripts/a11_gate.py first (tests, mutations, audit).")
    gate = json.load(open(gp))
    if gate.get("code_hash") != code_hash():
        sys.exit("ERROR: code changed after the gate was written. Re-run the tests, the audit and scripts/a11_gate.py.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--audit", action="store_true", help="also check the invariants of A11.6.2")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    check_gate(cfg)
    out_dir = os.path.join(ROOT, "results", cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(config=cfg, code_hash=code_hash(), python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__, platform=platform.platform(), cpu_count=os.cpu_count(), workers=args.workers,
                started=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(meta, open(os.path.join(out_dir, "run_meta.json"), "w"), indent=1)
    eps_grid = [None if (e is None or e == "inf") else float(e) for e in cfg["eps_grid"]]
    jobs = [(cfg, rho, s, a, g, e, out_dir, args.audit) for rho in cfg["rho_grid"] for s in cfg["s_grid"]
            for a in cfg["a_grid"] for g in cfg["gamma_grid"] for e in eps_grid]
    t0 = time.perf_counter()
    if args.workers > 1:
        with Pool(args.workers) as p:
            for path, dt in p.imap_unordered(run_cell, jobs):
                print(f"{os.path.basename(path)}  {dt:7.1f} s", flush=True)
    else:
        for j in jobs:
            path, dt = run_cell(j)
            print(f"{os.path.basename(path)}  {dt:7.1f} s", flush=True)
    df = pd.concat([pd.read_csv(os.path.join(out_dir, f)) for f in sorted(os.listdir(out_dir)) if f.startswith("cell_") and f.endswith(".csv")])
    nviol = int((df["violations"].fillna("") != "").sum())
    print(f"cells {len(jobs)}, rows {len(df)}, rows with violations {nviol}, wall {time.perf_counter() - t0:.1f} s")
    if nviol:
        print(df[df["violations"].fillna("") != ""][["rho", "s", "a", "gamma", "eps", "idx", "violations"]].head(20).to_string())
        sys.exit(1)


if __name__ == "__main__":
    main()
