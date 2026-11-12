#!/usr/bin/env python3
import argparse, dataclasses, itertools, json, os, platform, subprocess, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from dwell.dp import solve
from dwell.instances import make_instance_k
from dwell.m2 import Model2
from dwell.mismatch import ModelBA, a6_instance, ba_factory, base_policy_for, believed_factory, eval_obs

TOL = 1e-9
METHODS = ("DPMIS", "CEP", "ROLL_CEP")


def run_instance(cfg, rho, s, a, idx):
    lam_tol, lam_ba = cfg["lam_tol"], cfg["lam_ba"]
    inst, Dl, real = a6_instance(cfg, rho, s, a, idx, make_instance_k)
    models = {}
    for lam in sorted(set(lam_tol) | set(lam_ba) | {1.0}):
        i_l = dataclasses.replace(inst, D=Dl[lam])
        m = Model2(i_l)
        models[lam] = (i_l, m, solve(m))
    row = dict(rho=rho, s=s, a=a, idx=idx, n=cfg["n"], total_value=sum(inst.values), **real)
    for lam, (_, _, dp) in models.items():
        row[f"Vcal_{lam}"] = dp.value
    i1 = models[1.0][0]
    base = {}
    for lam_b in sorted(set(lam_tol) | set(lam_ba) | {1.0}):
        _, mb, dpb = models[lam_b]
        for meth in METHODS:
            base[(meth, lam_b)] = believed_factory(mb, base_policy_for(meth, mb, dpb))
    ctx = f"cell rho={rho} s={s} a={a} idx={idx}"
    for lam_b in lam_tol:
        for meth in METHODS:
            v = eval_obs(i1, base[(meth, lam_b)])
            assert -TOL <= v <= row["Vcal_1.0"] + TOL, f"tolerance value outside [0, Vcal] ({ctx}) {meth} {lam_b}: {v} vs {row['Vcal_1.0']}"
            row[f"Vtol_{meth}_b{lam_b}"] = v
    assert abs(row["Vtol_DPMIS_b1.0"] - row["Vcal_1.0"]) < TOL, f"DP with correct belief differs from calibrated value ({ctx})"
    ba = ModelBA(inst, lam_ba, Dl=[Dl[l] for l in lam_ba])
    dpba = solve(ba)
    row["V_BAavg"] = dpba.value
    fac_ba = ba_factory(ba, dpba.policy())
    vs = []
    for lam_t in lam_ba:
        i_t = models[lam_t][0]
        vb = eval_obs(i_t, fac_ba)
        vs.append(vb)
        assert -TOL <= vb <= row[f"Vcal_{lam_t}"] + TOL, f"BA value outside [0, Vcal] ({ctx}) truth {lam_t}: {vb} vs {row[f'Vcal_{lam_t}']}"
        row[f"Vba_t{lam_t}"] = vb
        for meth in METHODS:
            vn = eval_obs(i_t, base[(meth, 1.0)])
            assert -TOL <= vn <= row[f"Vcal_{lam_t}"] + TOL, f"nominal value outside [0, Vcal] ({ctx}) {meth} truth {lam_t}"
            row[f"Vnom_{meth}_t{lam_t}"] = vn
    assert abs(float(np.mean(vs)) - dpba.value) < 1e-8, f"BA policy value differs from BA DP value ({ctx})"
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
