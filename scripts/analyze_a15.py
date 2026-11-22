#!/usr/bin/env python3
"""
Analyse an A15 run (plan A15.5, A15.9): python scripts/analyze_a15.py --dir results/a15_main [--review]
Reads results/<name>/eval/cell_*.csv (and time_a15.json). A missing cell, wrong row count or a violation gives NO OUTCOME.
"""
import argparse, glob, itertools, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd
from dwell.analysis_a15 import VARIANTS, contrasts, decide, informative, seed_values


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--review", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.dir, "run_meta_eval.json")))["config"]
    n_cells = len(cfg["eval_sizes"]) * len(cfg["eval_rho"]) * len(cfg["eval_s"]) * len(cfg["eval_a"]) * len(cfg["eval_lam"])
    fs = sorted(glob.glob(os.path.join(a.dir, "eval", "cell_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    bad = []
    if len(fs) != n_cells: bad.append(f"{len(fs)} cell files, expected {n_cells}")
    for n in cfg["eval_sizes"]:
        if (df[df.n == n].groupby(["rho", "s", "a", "lam"]).size() != cfg["eval_instances"][str(n)]).any(): bad.append(f"n={n}: a cell has the wrong row count")
    if int((df["violations"].fillna("") != "").sum()): bad.append("rows with violations")
    if bad: print("NO OUTCOME:", "; ".join(bad)); sys.exit(1)
    inf = {n: informative(df[df.n == n]) for n in cfg["eval_sizes"]}
    planners = [c[2:] for c in df.columns if c.startswith("U_") and "_s" not in c[2:] and c not in ("U_opt",)]
    print(f"run {cfg['name']}  role {cfg['role']}  seeds {cfg['n_seeds']}  cells {n_cells}  rows {len(df)}")
    for n in cfg["eval_sizes"]:
        d = inf[n]; ncell = len(d.groupby(["rho", "s", "a", "lam"])) if len(d) else 0
        print(f"\nn = {n}: informative cells {ncell} of {len(df[df.n == n].groupby(['rho','s','a','lam']))}, instances {len(d)}")
        if not len(d): continue
        ref = d["U_opt"].mean() if n <= 5 else float("nan")
        if n <= 5: print(f"  mean U of the exact optimum {ref:.4f}")
        for p in planners: print(f"  {p:10s} mean U {d['U_' + p].mean():.4f}" + (f"   regret {100 * (1 - d['U_' + p].sum() / d['U_opt'].sum()):.2f} points" if n <= 5 else ""))
        for v in VARIANTS:
            sv = seed_values(d, v)
            print(f"  {v:10s} mean U {sv.mean():.4f}  SD over seeds {sv.std(ddof=1):.4f}  min {sv.min():.4f}  max {sv.max():.4f}" + (f"   regret {100 * (1 - sv.mean() / ref):.2f} points" if n <= 5 else ""))
    cs = contrasts(inf)
    print("\nPrimary contrasts (seed level, exact one-sided Wilcoxon, Holm over 8, alpha 0.05, material = mean delta >= 0.01):")
    T = pd.DataFrame(cs)
    print(T[[c for c in ["a", "b", "n", "mean_delta", "min_delta", "max_delta", "p", "holm_reject", "material", "undefined"] if c in T.columns]].round(4).to_string(index=False))
    tp = os.path.join(a.dir, "time_a15.json")
    ratio = None
    if os.path.exists(tp):
        t = json.load(open(tp))["per_decision"]
        for n, v in t.items(): print(f"\nDecision time at n = {n} (ms, mean): " + ", ".join(f"{k} {x['mean_ms']:.3f}" for k, x in v.items()))
        if "9" in t: ratio = t["9"]["RU_CEP"]["mean_ms"] / t["9"]["IL_AD"]["mean_ms"]; print(f"RU_CEP / IL_AD decision-time ratio at n = 9: {ratio:.1f}")
    if a.review: print("\nNo decision is applied (review mode)."); return
    res = decide(cs, ratio)
    print("\nOUTCOME:", json.dumps(res))
    json.dump(dict(result=res, contrasts=cs), open(os.path.join(a.dir, "outcome_a15.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
