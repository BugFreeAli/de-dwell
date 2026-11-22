#!/usr/bin/env python3
"""
A15 pipeline (plan A15.3 to A15.8).  Resumable; roles: audit (any other seeds) and main (pool seed 20261262, eval seed 20261261,
refused unless results/a15_gate.json carries the hash of the current code).
  python scripts/run_a15.py pool  --config configs/a15_main.json --workers 4   # teacher-labelled training pool (shards)
  python scripts/run_a15.py train --config configs/a15_main.json --workers 4   # 3 variants x seeds models
  python scripts/run_a15.py eval  --config configs/a15_main.json --workers 4   # per-cell CSVs (planners and every model)
  python scripts/run_a15.py time  --config configs/a15_main.json               # per-decision time at n = 7 and 9
"""
import argparse, glob, hashlib, itertools, json, os, platform, sys, time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.setrecursionlimit(100000)
import numpy as np
import pandas as pd

from dwell.instances import make_instance_k
from dwell.learned_g import (Scorer, build_distill_samples, build_instance_samples, learned_policy, rollout_states, to_arrays, train_variant)
from dwell.m0 import Model
from dwell.m0u import GoalDP, a14_instance, ceps_policy, need_of, prob_goal, ru_policy
from dwell.rules import make_rule
from dwell.dp import solve

MAIN = {"pool_seed": 20261262, "eval_seed": 20261261}
CODE_FILES = ["src/dwell/learned_g.py", "src/dwell/analysis_a15.py", "src/dwell/m0u.py", "src/dwell/learned.py", "src/dwell/m0.py",
              "src/dwell/dp.py", "src/dwell/rules.py", "src/dwell/instances.py", "scripts/run_a15.py"]
VARIANTS = ("IL_G", "IL_EV", "IL_G_noacc", "IL_A", "IL_AD")


def code_hash():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(open(os.path.join(ROOT, f), "rb").read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def check_gate(cfg):
    if cfg["role"] == "audit":
        if cfg["pool_seed"] in MAIN.values() or cfg["eval_seed"] in MAIN.values():
            sys.exit("ERROR: a main seed may not be used for an audit.")
        return
    if cfg["role"] != "main" or any(cfg[k] != v for k, v in MAIN.items()):
        sys.exit(f"ERROR: role main requires {MAIN}.")
    gp = os.path.join(ROOT, "results", "a15_gate.json")
    if not os.path.exists(gp) or json.load(open(gp)).get("code_hash") != code_hash():
        sys.exit("ERROR: no gate for the current code. Run scripts/a15_gate.py (tests, mutations, audit).")


def out_dir(cfg):
    d = os.path.join(ROOT, "results", cfg["name"]); os.makedirs(d, exist_ok=True); return d


# ---- pool -------------------------------------------------------------------------------------------------------------
def pool_shard(job):
    cfg, kind, shard = job
    sub = "pool_shards" if kind == "G" else "poolD_shards"
    total = cfg["pool_instances"] if kind == "G" else cfg["distill_instances"]
    path = os.path.join(out_dir(cfg), sub, f"shard_{shard:04d}.npz")
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for i in range(shard * cfg["shard_size"], min((shard + 1) * cfg["shard_size"], total)):
        r, _ = build_instance_samples(cfg, i) if kind == "G" else build_distill_samples(cfg, i)
        rows += [(i, x) for x in r]
    arr = to_arrays(rows) if kind == "G" else to_arrays(rows, a_max=cfg.get("distill_a_max", 9))
    Phi, mask, yg, ye, iid = arr
    np.savez(path + ".tmp.npz", Phi=Phi, mask=mask, yg=yg, ye=ye, iid=iid)
    os.replace(path + ".tmp.npz", path)
    return path


def load_pool(cfg, kind="G"):
    sub, total = ("pool_shards", cfg["pool_instances"]) if kind == "G" else ("poolD_shards", cfg["distill_instances"])
    fs = sorted(glob.glob(os.path.join(out_dir(cfg), sub, "shard_*.npz")))
    n_sh = -(-total // cfg["shard_size"])
    if len(fs) != n_sh:
        sys.exit(f"ERROR: {len(fs)} {sub} found, expected {n_sh}; run the pool stage first.")
    Z = [np.load(f) for f in fs]
    return tuple(np.concatenate([z[k] for z in Z]) for k in ("Phi", "mask", "yg", "ye", "iid"))


# ---- training -----------------------------------------------------------------------------------------------------------
_POOL = None


def _init_pool(cfg):
    global _POOL
    _POOL = {"G": load_pool(cfg, "G"), "D": load_pool(cfg, "D")}


def train_one(job):
    cfg, variant, seed = job
    path = os.path.join(out_dir(cfg), "models", f"{variant}_s{seed}.npz")
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sc, hist, agree, ns = train_variant(variant, seed, _POOL["D" if variant == "IL_AD" else "G"], epochs=cfg["epochs"])
    sc.save(path[:-4] + ".tmp", loss_first=hist[0], loss_last=hist[-1], train_agreement=agree, n_samples=ns)
    os.replace(path[:-4] + ".tmp.npz", path)
    return path


# ---- evaluation ---------------------------------------------------------------------------------------------------------
_MODELS = None


def load_models(cfg):
    d = os.path.join(out_dir(cfg), "models")
    return {(v, k): Scorer.load(os.path.join(d, f"{v}_s{k}.npz")) for v in VARIANTS for k in range(cfg["n_seeds"])}


def _init_models(cfg):
    global _MODELS
    _MODELS = load_models(cfg)


def planner_policies(model, need, n):
    ev = lambda name: (lambda b: (lambda st, ac: b(st)))(make_rule(model, name))
    P = {"CEP": ev("CEP"), "CEPS": ceps_policy(model, need)}
    P["RU_CEP"] = ru_policy(model, P["CEP"], need)
    if n <= 5:
        evp = solve(model).policy()
        P["DPEV"] = lambda st, ac: evp(st)
        P["RU_DPEV"] = ru_policy(model, P["DPEV"], need)
    return P


def eval_instance(cfg, models, n, rho, s, a, lam, idx):
    t0 = time.perf_counter()
    ecfg = dict(master_seed=cfg["eval_seed"], D0=cfg["D0"], value_set=cfg["value_set"], u_low=cfg["u_low"])
    inst, need, real = a14_instance(ecfg, n, rho, s, a, lam, idx)
    model = Model(inst)
    row = dict(n=n, rho=rho, s=s, a=a, lam=lam, idx=idx, need=need, **real)
    viol = []
    opt = GoalDP(model, need).value if n <= 5 else float("nan")
    row["U_opt"] = opt
    P = planner_policies(model, need, n)
    for name, pol in P.items():
        row["U_" + name] = prob_goal(model, pol, need)
    for (v, k), sc in models.items():
        row[f"U_{v}_s{k}"] = prob_goal(model, learned_policy(model, sc, need), need)
    us = [x for kk, x in row.items() if kk.startswith("U_") and kk != "U_opt"]
    if any(not (-1e-9 <= u <= 1 + 1e-9) for u in us):
        viol.append("range")
    if n <= 5 and any(u > opt + 1e-9 for u in us):
        viol.append("above_optimum")
    if row["U_RU_CEP"] < row["U_CEP"] - 1e-9:
        viol.append("rollout_below_base")
    if n <= 5 and row["U_RU_DPEV"] < row["U_DPEV"] - 1e-9:
        viol.append("rollout_below_base_dpev")
    row["violations"] = ";".join(viol)
    row["sec"] = time.perf_counter() - t0
    return row


def eval_cell(job):
    cfg, n, rho, s, a, lam = job
    d = os.path.join(out_dir(cfg), "eval"); os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"cell_n{n}_rho{rho}_s{s}_a{a}_l{lam}.csv")
    if os.path.exists(path):
        return path, 0.0
    t0 = time.perf_counter()
    rows = [eval_instance(cfg, _MODELS, n, rho, s, a, lam, i) for i in range(cfg["eval_instances"][str(n)])]
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return path, time.perf_counter() - t0


# ---- decision time ------------------------------------------------------------------------------------------------------
def time_decisions(cfg):
    models = load_models(cfg)
    sc = models[("IL_AD", 0)]
    out = {}
    for n in cfg["time_sizes"]:
        ecfg = dict(master_seed=cfg["eval_seed"], D0=cfg["D0"], value_set=cfg["value_set"], u_low=cfg["u_low"])
        acc_t = {"IL_AD": [], "CEP": [], "CEPS": [], "RU_CEP": []}
        for idx in range(cfg["time_instances"]):
            inst, need, _ = a14_instance(ecfg, n, 1.1, 2, 0.5, 0.25, idx)
            model = Model(inst)
            if need <= 0:
                continue
            rng = np.random.default_rng([cfg["eval_seed"], n, idx])
            worlds = list(model.worlds()); w = np.array([x[1] for x in worlds])
            c = worlds[int(rng.choice(len(worlds), p=w / w.sum()))][0]
            base = (lambda b: (lambda st, ac: b(st)))(make_rule(model, "CEP"))
            for st, ac in rollout_states(model, c, ru_policy(model, base, need), need, rng, 0.0)[:cfg["time_states"]]:
                for name in acc_t:
                    t0 = time.perf_counter()
                    if name == "IL_AD":
                        learned_policy(model, sc, need)(st, ac)
                    elif name == "CEP":
                        make_rule(model, "CEP")(st)
                    elif name == "CEPS":
                        ceps_policy(model, need)(st, ac)
                    else:
                        ru_policy(model, base, need)(st, ac)             # cold: a fresh policy and an empty value cache
                    acc_t[name].append(1000 * (time.perf_counter() - t0))
        out[str(n)] = {k: dict(mean_ms=float(np.mean(v)), median_ms=float(np.median(v)), decisions=len(v)) for k, v in acc_t.items()}
    json.dump(dict(per_decision=out, cpu=platform.processor(), python=platform.python_version()),
              open(os.path.join(out_dir(cfg), "time_a15.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pool", "train", "eval", "time"]); ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    cfg = json.load(open(a.config)); check_gate(cfg)
    d = out_dir(cfg)
    json.dump(dict(config=cfg, code_hash=code_hash(), python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                   platform=platform.platform(), cpu_count=os.cpu_count(), workers=a.workers, stage=a.stage,
                   started=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(d, f"run_meta_{a.stage}.json"), "w"), indent=1)
    t0 = time.perf_counter()
    if a.stage == "pool":
        jobs = ([(cfg, "G", k) for k in range(-(-cfg["pool_instances"] // cfg["shard_size"]))] +
                [(cfg, "D", k) for k in range(-(-cfg["distill_instances"] // cfg["shard_size"]))])
        with Pool(a.workers) as p:
            for path in p.imap_unordered(pool_shard, jobs):
                print(os.path.basename(path), flush=True)
        for kind in ("G", "D"):
            pool = load_pool(cfg, kind)
            print(f"pool {kind}: {len(pool[0])} samples; labels {(pool[2] >= 0).mean():.3f}")
        print(f"wall {time.perf_counter() - t0:.1f} s")
    elif a.stage == "train":
        jobs = [(cfg, v, k) for v in VARIANTS for k in range(cfg["n_seeds"])]
        with Pool(a.workers, initializer=_init_pool, initargs=(cfg,)) as p:
            for path in p.imap_unordered(train_one, jobs):
                print(os.path.basename(path), flush=True)
        print(f"trained {len(jobs)} models, wall {time.perf_counter() - t0:.1f} s")
    elif a.stage == "eval":
        jobs = [(cfg, n, rho, s, al, lam) for n in cfg["eval_sizes"] for rho, s, al, lam in
                itertools.product(cfg["eval_rho"], cfg["eval_s"], cfg["eval_a"], cfg["eval_lam"])]
        with Pool(a.workers, initializer=_init_models, initargs=(cfg,)) as p:
            for path, dt in p.imap_unordered(eval_cell, jobs):
                print(f"{os.path.basename(path)}  {dt:8.1f} s", flush=True)
        fs = sorted(glob.glob(os.path.join(d, "eval", "cell_*.csv")))
        df = pd.concat([pd.read_csv(f) for f in fs])
        nviol = int((df["violations"].fillna("") != "").sum())
        print(f"cells {len(jobs)}, rows {len(df)}, rows with violations {nviol}, wall {time.perf_counter() - t0:.1f} s")
        if nviol:
            print(df[df["violations"].fillna("") != ""][["n", "rho", "s", "a", "lam", "idx", "violations"]].head(20).to_string()); sys.exit(1)
    else:
        time_decisions(cfg)


if __name__ == "__main__":
    main()
