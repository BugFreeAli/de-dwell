#!/usr/bin/env python3
"""
Analyse an A14 run (plan A14.5, A14.9). Reads only results/<name>/cell_*.csv.
  python scripts/analyze_a14.py --dir results/a14_main                 -> decision table (role main)
  python scripts/analyze_a14.py --dir results/a14_audit --review       -> no decision
  python scripts/analyze_a14.py --dir results/a14_n5                   -> descriptive only (role spot)
A missing cell, a wrong row count or a row with violations gives NO OUTCOME; nothing is dropped.
"""
import argparse, glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd
from dwell.analysis_a14 import AXES, decide, summarise
from dwell.m0u import AWARE, EV_BASED, METHODS


def load(d):
    cfg = json.load(open(os.path.join(d, "run_meta.json")))["config"]
    files = sorted(glob.glob(os.path.join(d, "cell_*.csv")))
    n_cells = len(cfg["n_grid"]) * len(cfg["rho_grid"]) * len(cfg["s_grid"]) * len(cfg["a_grid"]) * len(cfg["lam_grid"])
    problems = []
    if len(files) != n_cells:
        problems.append(f"{d}: found {len(files)} cell files, expected {n_cells}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if (df.groupby(list(AXES)).size() != cfg["n_instances"]).any():
        problems.append(f"{d}: a cell does not have n_instances rows")
    bad = int((df["violations"].fillna("") != "").sum())
    if bad:
        problems.append(f"{d}: {bad} rows with invariant violations")
    return cfg, df, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--review", action="store_true")
    a = ap.parse_args()
    cfg, df, problems = load(a.dir)
    if problems:
        print("NO OUTCOME:", "; ".join(problems)); sys.exit(1)
    S = summarise(df)
    S.to_csv(os.path.join(a.dir, "summary_cells.csv"), index=False)
    inf = S[S["informative"]]
    print(f"cells {len(S)}  informative {len(inf)}  instances per cell {cfg['n_instances']}  role {cfg['role']}  seed {cfg['master_seed']}")
    print("\nInformative cells: mean goal probability of the optimum and mean regret per method (points of relative regret):")
    print(f"  mean U_opt over informative cells: {inf['U_opt'].mean():.3f}")
    if len(inf):
        print((100 * inf[[f"regret_{m}" for m in METHODS]].mean()).rename(lambda c: c[7:]).round(2).to_string())
    print(f"\nBest method is objective-aware (CEPS or RU_*) in {100 * inf['best_is_aware'].mean() if len(inf) else float('nan'):.0f}% of informative cells")
    print(f"First action of the goal optimum differs from the EV optimum in {100 * df['first_differs'].mean():.1f}% of instances")
    show = ["n", "rho", "s", "a", "lam", "U_opt", "best_method", "regret_best", "abs_best", "lo5", "best_ev_method", "regret_best_ev", "lo5_ev"]
    print("\nInformative cells, largest regret of the best EV-based method first:")
    print(inf.sort_values("regret_best_ev", ascending=False)[show].head(12).round(4).to_string(index=False))
    print("\nInformative cells, largest regret of the best method of ALL 17 first:")
    print(inf.sort_values("regret_best", ascending=False)[show].head(8).round(4).to_string(index=False))
    # instance level tails of the best method per cell and of the best EV-based method per cell
    for label, col in (("best of ALL", "best_method"), ("best EV-based", "best_ev_method")):
        best = S.set_index(list(AXES))[col].to_dict()
        V = df[df["U_opt"] > 0].copy()
        key = list(zip(V.n, V.rho, V.s, V.a, V.lam))
        V["u"] = [getattr(r, "U_" + best[k]) for r, k in zip(V.itertuples(), key)]
        reg = 1 - V["u"] / V["U_opt"]
        print(f"Instance-level loss of {label} (all instances with U_opt > 0): mean {100 * reg.mean():.2f}%, p95 {100 * reg.quantile(.95):.1f}%, "
              f"max {100 * reg.max():.1f}%")
    rho_s = df["rho_dbf"].rank().corr((df["U_opt"] - df["U_DPEV"]).rank()) if "rho_dbf" in df else float("nan")
    print(f"Spearman correlation of the demand-bound load with the instance loss of DPEV: {rho_s:.3f}")
    if a.review or cfg["role"] == "spot":
        print("\nNo go rule is applied (review or spot-check mode)."); return
    key = lambda r: (r.n, r.rho, r.s, r.a, r.lam)
    res = decide({key(r): bool(r.flagged_all) for r in S.itertuples()}, {key(r): bool(r.flagged_ev) for r in S.itertuples()},
                 {key(r): bool(r.point_all) for r in S.itertuples()})
    print("\nOUTCOME:", res["outcome"], res["regions"][:6] if res["regions"] else "")
    json.dump(dict(outcome=res["outcome"], regions=[list(map(list, r)) for r in res["regions"]]),
              open(os.path.join(a.dir, "outcome_a14.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
