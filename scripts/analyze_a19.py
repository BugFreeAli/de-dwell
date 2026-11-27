#!/usr/bin/env python3
"""
Analyse an A19 run (plan A19.4, A19.5):  python scripts/analyze_a19.py --dir results/a19_main [--review]
A pilot directory (role pilot) prints the descriptive go / stop rule. A missing cell, a wrong row count or a violation gives NO OUTCOME.
"""
import argparse, glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import pandas as pd
from dwell.analysis_a19 import CELL, claims19, informative, pilot_rule, regret_table


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
    res = claims19(df, sizes, scales)
    rows = []
    for (sc, n), v in res["statements"].items():
        d = df[(df.scale == sc) & (df.n == n)]
        inf = informative(d)
        r = dict(scale=sc, n=n, informative=f"{len(inf.groupby(CELL)) if len(inf) else 0} of {len(d.groupby(CELL))}")
        if not v["undefined"]:
            r.update(instances=v["instances"], ADAPT_minus_NOM=100 * v["gain"], lo=100 * v["gain_lo"], P1=v["p1"],
                     NOM_minus_ADAPT_up=100 * v["loss_hi"], P2=v["p2"], ADAPT_minus_CLAMP=100 * v["vs_clamp"], s1_lo=100 * v["vs_clamp_lo"], S1=v["s1"],
                     gap_to_informed=100 * v["gap"], recovery=v["recovery"], rec_lo=v["recovery_lo"], rec_class=v["recovery_class"])
            r.update({("U_" + m): inf["U_" + m].mean() for m in ("NOM", "CLAMP", "ADAPT", "ADAPT_V", "CEPS_T", "RU_CEP_T")})
        rows.append(r)
    print("\nPer true scale and size (points of survival probability; bounds one-sided 95%; scale 1.0 = same-seed control):")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    p = res["pooled"]
    if not p["undefined"]:
        print(f"\nPooled recovery over scales != 1 and sizes {sizes}: {p['recovery']:.2f} (lower bound {p['recovery_lo']:.2f}, class {p['recovery_class']}), gap to the informed rollout planner {100 * p['gap']:.1f} points")
    print("\nRegret against the exact optimum of the true model, n = 5, informative cells (points):")
    for sc in (1.0,) + scales:
        print(sc, {k: round(v, 2) for k, v in regret_table(df[(df.scale == sc) & (df.n == 5)]).items()})
    if cfg["role"] == "pilot":
        print("\nPILOT RULE:", pilot_rule(df)); return
    if a.review or cfg["role"] != "main": print("\nNo decision is applied (review mode)."); return
    print(f"\nP1 statements: {res['n_p1']}; label: {res['label']}; clamp comparison: {res['clamp_class']}")
    print("OUTCOME:", res["label"], "/", res["clamp_class"])
    out = {f"scale{k[0]}_n{k[1]}": v for k, v in res["statements"].items()}
    json.dump(dict(label=res["label"], clamp_class=res["clamp_class"], pooled=p, statements=out), open(os.path.join(a.dir, "outcome_a19.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
