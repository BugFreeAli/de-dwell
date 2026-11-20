#!/usr/bin/env python3
"""
Analyse A13 runs (plan A13.5, A13.9). Reads only results/<name>/cell_*.csv.
  python scripts/analyze_a13.py --dir results/a13_main [--ctrl results/a13_ctrl]    -> decision table
  python scripts/analyze_a13.py --dir results/a13_audit --ctrl results/a13_ctrl_audit --review   -> no decision
A missing cell, a wrong row count or a row with violations gives NO OUTCOME; nothing is dropped.
"""
import argparse, glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd
from dwell.analysis_a13 import AXES, decide, summarise
from dwell.m0g import METHODS


def load(d):
    cfg = json.load(open(os.path.join(d, "run_meta.json")))["config"]
    files = sorted(glob.glob(os.path.join(d, "cell_*.csv")))
    n_cells = len(cfg["n_grid"]) * len(cfg["rho_grid"]) * len(cfg["s_grid"]) * len(cfg["a_grid"]) * len(cfg["geom_grid"])
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
    ap.add_argument("--dir", required=True); ap.add_argument("--ctrl"); ap.add_argument("--review", action="store_true")
    a = ap.parse_args()
    cfg, df, problems = load(a.dir)
    cdf = None
    if a.ctrl:
        _, cdf, p2 = load(a.ctrl); problems += p2
    if problems:
        print("NO OUTCOME:", "; ".join(problems)); sys.exit(1)
    S = summarise(df)
    S.to_csv(os.path.join(a.dir, "summary_cells.csv"), index=False)
    print(f"cells {len(S)}  instances per cell {cfg['n_instances']}  role {cfg['role']}  seed {cfg['master_seed']}")
    print("\nMean regret per method (mean over cells, in points):")
    print((100 * S[[f"regret_{m}" for m in METHODS]].mean()).rename(lambda c: c[7:]).round(2).to_string())
    print(f"\nBest method is a rollout in {100 * S['best_is_rollout'].mean():.0f}% of cells")
    print(f"First action of the DP differs from that of CEP in {100 * df['first_differs'].mean():.1f}% of instances")
    print(f"Slew matrices: mean {df['mean_slew'].mean():.2f}, coefficient of variation {df['cv_slew'].mean():.2f}, "
          f"instances changed by the closure {100 * (df['n_closed'] > 0).mean():.0f}%")
    best = S.set_index(list(AXES))["best_method"].to_dict()
    V = df.copy()
    V["best"] = [best[(r.n, r.rho, r.s, r.a, r.geom)] for r in V.itertuples()]
    V["best_value"] = [getattr(r, "V_" + r.best) for r in V.itertuples()]
    V["inst_regret"] = np.where(V["V_dp"] > 0, 1 - V["best_value"] / V["V_dp"], np.nan)
    rho_s = V["cv_slew"].rank().corr(V["inst_regret"].rank())
    print(f"Spearman correlation of the slew coefficient of variation with the instance regret of the cell's best method: {rho_s:.3f}")
    show = ["n", "rho", "s", "a", "geom", "best_method", "regret_best", "lo5"]
    print("\nBest non-DP method per cell (regret as fraction), largest first:")
    print(S.sort_values("regret_best", ascending=False)[show].head(12).to_string(index=False))
    if cdf is not None:
        C = summarise(cdf).set_index(list(AXES))["regret_best"]
        D = S.set_index(list(AXES))["regret_best"] - C
        print(f"\nControl (uniform slew s_bar, same instances): mean best regret {100 * C.mean():.2f}% against {100 * S['regret_best'].mean():.2f}% with the matrix; "
              f"largest increase {100 * D.max():.2f} points")
    if a.review:
        print("\nNo go rule is applied in review mode."); return
    flagged = {(r.n, r.rho, r.s, r.a, r.geom): bool(r.flagged) for r in S.itertuples()}
    pointf = {(r.n, r.rho, r.s, r.a, r.geom): bool(r.point_flagged) for r in S.itertuples()}
    res = decide(flagged, pointf)
    print("\nOUTCOME:", res["outcome"], res["regions"][:6] if res["regions"] else "")
    json.dump(dict(outcome=res["outcome"], regions=[list(map(list, r)) for r in res["regions"]]),
              open(os.path.join(a.dir, "outcome_a13.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
