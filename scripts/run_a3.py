#!/usr/bin/env python3
import argparse, itertools, json, os, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from dwell.instances import make_instance_k
from dwell.m2 import Model2, evaluate_policy2
from dwell.rules import make_rule

TOL = 1e-9
RECHECK = ("CEP",)


def augment_cell(job):
    cfg, src_dir, out_dir, rho, s, a = job
    name = f"cell_rho{rho}_s{s}_a{a}.csv"
    out_path = os.path.join(out_dir, name)
    if os.path.exists(out_path):
        return name, 0.0, "skipped"
    t0 = time.time()
    df = pd.read_csv(os.path.join(src_dir, name))
    new_rules = [r for r in cfg["extra_rules"] if "V_" + r not in df.columns]
    cols = {r: [] for r in new_rules}
    times = {r: [] for r in new_rules + list(RECHECK)}
    for row in df.itertuples():
        inst, real = make_instance_k(cfg["master_seed"], int(row.idx), cfg["n"], rho, s, a,
                                     D=tuple(cfg["D"]), value_set=tuple(cfg["value_set"]), u_low=cfg["u_low"])
        ctx = f"cell rho={rho} s={s} a={a} idx={row.idx}"
        assert abs(sum(inst.values) - row.total_value) < TOL, f"instance mismatch (values) {ctx}"
        assert real["d_max"] == row.d_max, f"instance mismatch (d_max) {ctx}"
        assert abs(real["rho_dbf"] - row.rho_dbf) < TOL, f"instance mismatch (rho_dbf) {ctx}"
        model = Model2(inst, cfg.get("kappa", 1.0))
        for r in RECHECK:
            t1 = time.time()
            v = evaluate_policy2(model, make_rule(model, r))
            times[r].append(time.time() - t1)
            assert abs(v - getattr(row, "V_" + r)) < TOL, f"stored {r} differs from recomputed {ctx}: {v} vs {getattr(row, 'V_' + r)}"
        base_vals = {}
        for r in new_rules:
            t1 = time.time()
            v = evaluate_policy2(model, make_rule(model, r))
            times[r].append(time.time() - t1)
            assert -TOL <= v <= row.V_dp + TOL, f"rule {r} outside [0, V_dp] {ctx}: {v} vs {row.V_dp}"
            if r.startswith("ROLL_"):
                base = r[5:]
                stored = getattr(row, "V_" + base) if hasattr(row, "V_" + base) else None
                assert stored is not None, f"base rule {base} missing for {r} {ctx}"
                assert v >= stored - TOL, f"rollout {r} below its base rule {ctx}: {v} vs {stored}"
            cols[r].append(v)
    for r in new_rules:
        df["V_" + r] = cols[r]
        df["t_" + r] = times[r]
    for r in RECHECK:
        df["t_" + r] = times[r]
    df.to_csv(out_path + ".tmp", index=False)
    os.replace(out_path + ".tmp", out_path)
    return name, time.time() - t0, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-cells", type=int, default=None)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    os.makedirs(args.out, exist_ok=True)
    cells = list(itertools.product(cfg["rho_grid"], cfg["s_grid"], cfg["a_grid"]))
    if args.max_cells:
        cells = cells[: args.max_cells]
    jobs = [(cfg, args.src, args.out, rho, s, a) for rho, s, a in cells]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, (name, dt, status) in enumerate(pool.imap_unordered(augment_cell, jobs), 1):
            print(f"[{k}/{len(jobs)}] {status:7s} {name}  {dt:7.1f}s  elapsed {time.time()-t0:8.1f}s", flush=True)
    print("finished in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
