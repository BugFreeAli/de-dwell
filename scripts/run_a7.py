#!/usr/bin/env python3
import argparse, itertools, json, os, platform, subprocess, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from dwell.adapt import Kit, Planner, a7_instance, eval_rec, iid_truth, nominal_factory
from dwell.dp import solve
from dwell.instances import make_instance_k
from dwell.m2 import Model2

TOL = 1e-9
NOMINAL = ("DPMIS", "CEP", "ROLL_CEP")


def prior_weights(spec):
    w = spec.get("weights")
    return tuple(w) if w is not None else tuple(1.0 / len(spec["lams"]) for _ in spec["lams"])


def run_instance(cfg, rho, s, a, idx):
    t0 = time.time()
    inst, Dl, real, inst0 = a7_instance(cfg, rho, s, a, idx, make_instance_k)
    truths = cfg["truths"]
    float_truths = [t for t in truths if not isinstance(t, str)]
    need = set(float_truths) | {1.0}
    for spec in cfg["priors"].values():
        need |= set(spec["lams"])
    planner = Planner(inst, Dl, need)
    truth_inst, vcal = {}, {}
    for t in truths:
        if isinstance(t, str):
            ti = iid_truth(inst0, tuple(cfg["D0"]), cfg["iid_scales"], cfg["iid_probs"])
            truth_inst[t] = ti
            vcal[t] = solve(Model2(ti)).value
        else:
            truth_inst[t] = planner.models[t][0]
            vcal[t] = planner.models[t][2].value
    row = dict(rho=rho, s=s, a=a, idx=idx, n=cfg["n"], total_value=sum(inst.values), **real)
    ctx = f"cell rho={rho} s={s} a={a} idx={idx}"
    for t in truths:
        row[f"Vcal_{t}"] = vcal[t]

    def put(key, t, v):
        assert -TOL <= v <= vcal[t] + TOL, f"value outside [0, Vcal] ({ctx}) {key} truth {t}: {v} vs {vcal[t]}"
        row[key] = v

    for meth in NOMINAL:
        fac = nominal_factory(planner, meth, 1.0)
        for t in truths:
            put(f"Vnom_{meth}_t{t}", t, eval_rec(truth_inst[t], fac))
    if 1.0 in float_truths:
        assert abs(row["Vnom_DPMIS_t1.0"] - vcal[1.0]) < TOL, f"DP with correct belief differs from calibrated value ({ctx})"
    for pname, spec in cfg["priors"].items():
        lams, w = spec["lams"], prior_weights(spec)
        kit = Kit(planner, lams, w)
        dpba = kit.solve_ba()
        row[f"V_BA_{pname}"] = dpba.value
        names = ["BA"] + list(cfg["practical"].get(pname, []))
        inside = all(l in float_truths for l in lams)
        for name in names:
            fac = kit.factory(name)
            vals = {}
            for t in truths:
                v = eval_rec(truth_inst[t], fac)
                vals[t] = v
                put(f"V{name}_{pname}_t{t}", t, v)
            if inside:
                avg = sum(wi * vals[l] for wi, l in zip(w, lams))
                if name == "BA":
                    assert abs(avg - dpba.value) < 1e-8, f"BA policy value differs from BA DP value ({ctx}) {pname}"
                else:
                    assert avg <= dpba.value + 1e-8, f"practical planner beats Bayes-adaptive optimum ({ctx}) {pname} {name}"
    row["sec"] = time.time() - t0
    return row


def run_cell(job):
    cfg, out_dir, rho, s, a = job
    path = os.path.join(out_dir, f"cell_rho{rho}_s{s}_a{a}.csv")
    if os.path.exists(path):
        return path, 0.0, "skipped"
    t0 = time.time()
    rows = [run_instance(cfg, rho, s, a, idx) for idx in range(cfg["n_instances"])]
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return path, time.time() - t0, "done"


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
    ap.add_argument("--n-instances", type=int, default=None)
    ap.add_argument("--master-seed", type=int, default=None)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    if args.n_instances is not None:
        cfg["n_instances"] = args.n_instances
    if args.master_seed is not None:
        cfg["master_seed"] = args.master_seed
    os.makedirs(args.out, exist_ok=True)
    cells = list(itertools.product(cfg["rho_grid"], cfg["s_grid"], cfg["a_grid"]))
    if args.max_cells:
        cells = cells[: args.max_cells]
    meta = dict(config=cfg, config_file=args.config, git_commit=git_commit(), python=sys.version,
                numpy=np.__version__, pandas=pd.__version__, platform=platform.platform(),
                cpu_count=os.cpu_count(), workers=args.workers, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(meta, open(os.path.join(args.out, "run_meta.json"), "w"), indent=2)
    jobs = [(cfg, args.out, rho, s, a) for rho, s, a in cells]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, (path, dt, status) in enumerate(pool.imap_unordered(run_cell, jobs), 1):
            print(f"[{k}/{len(jobs)}] {status:7s} {os.path.basename(path)}  {dt:7.1f}s  elapsed {time.time()-t0:8.1f}s", flush=True)
    print("finished in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
