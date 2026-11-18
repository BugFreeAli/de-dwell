#!/usr/bin/env python3
"""
Analyse an A11 run (plan A11.5 and A11.9). Reads only results/<name>/cell_*.csv.
  python scripts/analyze_a11.py --dir results/a11_main                 -> go rule and decision table (main run only)
  python scripts/analyze_a11.py --dir results/a11_audit --review       -> non-gating audit review (no outcome is applied)
Nothing is dropped: a missing cell, a wrong row count, a row with violations or an undefined cell gives NO OUTCOME.
"""
import argparse, glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd
from dwell.analysis_a11 import decide, lm_differences, outside_corner, summarise
from dwell.rules1 import METHODS


def load(d):
    meta = json.load(open(os.path.join(d, "run_meta.json")))
    cfg = meta["config"]
    files = sorted(glob.glob(os.path.join(d, "cell_*.csv")))
    n_cells = (len(cfg["rho_grid"]) * len(cfg["s_grid"]) * len(cfg["a_grid"]) * len(cfg["gamma_grid"]) * len(cfg["eps_grid"]))
    problems = []
    if len(files) != n_cells:
        problems.append(f"found {len(files)} cell files, expected {n_cells}")
    df = pd.concat([pd.read_csv(f, dtype={"eps": str}) for f in files], ignore_index=True)
    counts = df.groupby(["rho", "s", "a", "gamma", "eps"]).size()
    if (counts != cfg["n_instances"]).any():
        problems.append("a cell does not have n_instances rows")
    if (df["violations"].fillna("") != "").any():
        problems.append(f"{int((df['violations'].fillna('') != '').sum())} rows with invariant violations")
    return cfg, df, problems


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--review", action="store_true")
    a = ap.parse_args()
    cfg, df, problems = load(a.dir)
    summ = summarise(df)
    if summ["undefined"].any():
        problems.append("a cell has sum V_dp = 0 (undefined)")
    summ.to_csv(os.path.join(a.dir, "summary_cells.csv"), index=False)
    lm = lm_differences(df); lm.to_csv(os.path.join(a.dir, "summary_L_minus_M.csv"), index=False)
    if problems:
        print("NO OUTCOME:", "; ".join(problems)); sys.exit(1)
    key = lambda r: (r.rho, r.s, r.a, r.gamma, str(r.eps))
    print(f"cells {len(summ)}  instances per cell {cfg['n_instances']}  role {cfg['role']}  seed {cfg['master_seed']}")
    mean_reg = {m: float(summ["regret_" + m].mean()) for m in METHODS}
    print("\nMean regret per method (mean over cells, in points):")
    print(pd.Series({m: 100 * v for m, v in mean_reg.items()}).round(2).to_string())
    print("\nMean regret L minus M per rule (points, paired 95% CI):")
    print(lm.assign(mean_regret_L_minus_M=100 * lm.mean_regret_L_minus_M, lo=100 * lm.lo, hi=100 * lm.hi).round(2).to_string(index=False))
    print(f"\nBest method is a rollout in {100 * summ['best_is_rollout'].mean():.0f}% of cells")
    if "value_of_idle" in summ:
        print(summ.groupby("eps")[["value_of_idle", "p_wait_dp", "p_pre_dp"]].mean().round(4).to_string())
    if a.review:
        worse = lm[lm.mean_regret_L_minus_M > 0.01]
        print("\nREVIEW (not gating): L worse than M by more than 1 point on average:", list(worse.rule) or "none")
        print(summ[["rho", "s", "a", "gamma", "eps", "best_method", "regret_best", "lo5"]].round(4).head(60).to_string(index=False))
        print("\nNo go rule is applied in review mode.")
        return
    flagged = {key(r): bool(r.flagged) for r in summ.itertuples()}
    point = {key(r): bool(r.point_flagged) for r in summ.itertuples()}
    res = decide(flagged, point)
    print("\nOUTCOME:", res["outcome"], res["kind"])
    for p in res["regions"]:
        print("  region:", p)
    json.dump(dict(outcome=res["outcome"], kind=res["kind"], regions=[[list(c) for c in p] for p in res["regions"]]),
              open(os.path.join(a.dir, "outcome.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
