#!/usr/bin/env python3
"""
Analyse an A17 run (plan A17.4, A17.7):  python scripts/analyze_a17.py --dir results/a17_main [--review]
A missing cell, a wrong row count or a violation gives NO OUTCOME.
"""
import argparse, glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd
from dwell.analysis_a16 import CELL, informative
from dwell.analysis_a17 import claims17


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--review", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.dir, "run_meta.json")))["config"]
    fs = sorted(glob.glob(os.path.join(a.dir, "cell_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    ncell = len(cfg["rho_grid"]) * len(cfg["s_grid"]) * len(cfg["a_grid"]) * len(cfg["lam_grid"])
    want_cells = sum(len(f["sizes"]) for f in cfg["families"]) * ncell
    bad = []
    if len(fs) != want_cells: bad.append(f"{len(fs)} cell files, expected {want_cells}")
    sizes = {f["name"]: [s["n"] for s in f["sizes"]] for f in cfg["families"]}
    for f in cfg["families"]:
        for s in f["sizes"]:
            if (df[(df.group == f["name"]) & (df.n == s["n"])].groupby(CELL).size() != s["instances"]).any(): bad.append(f"{f['name']} n={s['n']}: wrong row count")
    if int((df["violations"].fillna("") != "").sum()): bad.append("rows with violations")
    if bad: print("NO OUTCOME:", "; ".join(bad)); sys.exit(1)
    print(f"run {cfg['name']}  role {cfg['role']}  eval seed {cfg['eval_seed']}  cells {want_cells}  rows {len(df)}")
    res = claims17(df, sizes)
    rows = []
    for (f, n), v in res["families"].items():
        d = df[(df.group == f) & (df.n == n)]
        inf = informative(d)
        r = dict(family=f, n=n, informative=f"{len(inf.groupby(CELL)) if len(inf) else 0} of {len(d.groupby(CELL))}")
        if not v["undefined"]:
            r.update(instances=v["instances"], CEPS_minus_CEP=100 * v["gain"], lo=100 * v["gain_lo"], P1=v["p1"], RU_minus_CEPS=100 * v["ru_gap"],
                     up=100 * v["ru_gap_hi"], cheap_planner=v["adequacy"])
            r.update(U_RU_CEP=inf.U_RU_CEP.mean(), U_CEPS=inf.U_CEPS.mean(), U_CEP=inf.U_CEP.mean())
        rows.append(r)
    print("\nPer family and size (gains in points of survival probability; bounds one-sided 95%; base = same-seed control):")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nP3 pooled over the variant families:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["P3_pooled"].items()})
    if a.review or cfg["role"] != "main": print("\nNo decision is applied (review mode)."); return
    print(f"\nP1 statements: {res['n_statements']}; label (P1 only): {res['label']}")
    print("OUTCOME:", res["label"])
    out = {f"{k[0]}_n{k[1]}": v for k, v in res["families"].items()}
    json.dump(dict(label=res["label"], P3_pooled=res["P3_pooled"], families=out), open(os.path.join(a.dir, "outcome_a17.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
