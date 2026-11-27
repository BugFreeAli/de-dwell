#!/usr/bin/env python3
"""
Analyse an A18 run (plan A18.4, A18.5):  python scripts/analyze_a18.py --dir results/a18_main [--review]
A missing cell, a wrong row count or a violation gives NO OUTCOME.
"""
import argparse, glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import pandas as pd
from dwell.analysis_a18 import CELL, SCALES, claims18, informative, regret_table


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--review", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.dir, "run_meta.json")))["config"]
    fs = sorted(glob.glob(os.path.join(a.dir, "cell_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    ncell = len(cfg["rho_grid"]) * len(cfg["s_grid"]) * len(cfg["a_grid"]) * len(cfg["lam_grid"])
    want_cells = len(cfg["sizes"]) * len(cfg["scales"]) * ncell
    bad = []
    if len(fs) != want_cells: bad.append(f"{len(fs)} cell files, expected {want_cells}")
    for s in cfg["sizes"]:
        if (df[df.n == s["n"]].groupby(CELL).size() != s["instances"]).any(): bad.append(f"n={s['n']}: wrong row count")
    if int((df["violations"].fillna("") != "").sum()): bad.append("rows with violations")
    if bad: print("NO OUTCOME:", "; ".join(bad)); sys.exit(1)
    sizes = tuple(s["n"] for s in cfg["sizes"]); scales = tuple(x for x in cfg["scales"] if x != 1.0)
    print(f"run {cfg['name']}  role {cfg['role']}  eval seed {cfg['eval_seed']}  cells {want_cells}  rows {len(df)}")
    res = claims18(df, sizes, scales)
    rows = []
    for (sc, n), v in res["statements"].items():
        d = df[(df.scale == sc) & (df.n == n)]
        inf = informative(d)
        r = dict(scale=sc, n=n, informative=f"{len(inf.groupby(CELL)) if len(inf) else 0} of {len(d.groupby(CELL))}")
        if not v["undefined"]:
            r.update(instances=v["instances"], CEPS_minus_CEP=100 * v["gain"], lo=100 * v["gain_lo"], P1=v["p1"],
                     informed_minus_nominal_CEPS=100 * v["cost"], cost_up=100 * v["cost_hi"], cost_class=v["cost_class"],
                     CEPS_minus_CEP_T=100 * v["aware"], aware_lo=100 * v["aware_lo"], aware_class=v["aware_class"])
            if "adequacy" in v: r.update(RU_minus_CEPS=100 * v["ru_gap"], ru_up=100 * v["ru_gap_hi"], cheap_planner=v["adequacy"])
            r.update(U_CEP=inf.U_CEP.mean(), U_CEPS=inf.U_CEPS.mean(), U_CEP_T=inf.U_CEP_T.mean(), U_CEPS_T=inf.U_CEPS_T.mean())
        rows.append(r)
    print("\nPer true scale and size (points of survival probability; bounds one-sided 95%; scale 1.0 = same-seed control):")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nRegret against the exact optimum of the true model, n = 5, informative cells (points):")
    for sc in (1.0,) + scales:
        print(sc, {k: round(v, 2) for k, v in regret_table(df[(df.scale == sc) & (df.n == 5)]).items()})
    if a.review or cfg["role"] != "main": print("\nNo decision is applied (review mode)."); return
    print(f"\nP1 statements: {res['n_statements']}; label (P1 only): {res['label']}")
    print("OUTCOME:", res["label"])
    out = {f"scale{k[0]}_n{k[1]}": v for k, v in res["statements"].items()}
    json.dump(dict(label=res["label"], statements=out), open(os.path.join(a.dir, "outcome_a18.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
