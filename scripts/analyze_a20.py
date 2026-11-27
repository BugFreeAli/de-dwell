#!/usr/bin/env python3
"""
Analyse an A20 run (plan A20.4, A20.5):  python scripts/analyze_a20.py --dir results/a20_main [--review]
A missing cell, a wrong row count or a violation gives NO OUTCOME.
"""
import argparse, glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import pandas as pd
from dwell.analysis_a20 import CELL, CONTROL, IN_SUPPORT, OUT, claims20, informative, regret_table, timing_table


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--review", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.dir, "run_meta.json")))["config"]
    fs = sorted(glob.glob(os.path.join(a.dir, "cell_*.csv")))
    df = pd.concat([pd.read_csv(f, dtype={"cond": str}) for f in fs], ignore_index=True)
    ncell = len(cfg["rho_grid"]) * len(cfg["s_grid"]) * len(cfg["a_grid"]) * len(cfg["lam_grid"])
    want_cells = len(cfg["sizes"]) * len(cfg["cond_order"]) * ncell
    bad = []
    if len(fs) != want_cells: bad.append(f"{len(fs)} cell files, expected {want_cells}")
    for s in cfg["sizes"]:
        if (df[df.n == s["n"]].groupby(CELL).size() != s["instances"]).any(): bad.append(f"n={s['n']}: wrong row count")
    if int((df["violations"].fillna("") != "").sum()): bad.append("rows with violations")
    if bad: print("NO OUTCOME:", "; ".join(bad)); sys.exit(1)
    sizes = tuple(s["n"] for s in cfg["sizes"])
    print(f"run {cfg['name']}  role {cfg['role']}  eval seed {cfg['eval_seed']}  cells {want_cells}  rows {len(df)}")
    res = claims20(df, sizes, IN_SUPPORT, OUT)
    rows = []
    for (c, n), v in res["statements"].items():
        d = df[(df["cond"] == c) & (df.n == n)]
        inf = informative(d)
        r = dict(cond=c, D_true=cfg["conditions"][c], n=n, informative=f"{len(inf.groupby(CELL)) if len(inf) else 0} of {len(d.groupby(CELL))}")
        if not v["undefined"]:
            r.update(instances=v["instances"], ADAPT_minus_RU_CLAMP=100 * v["vs_ru_clamp"], lo=100 * v["vs_ru_clamp_lo"], Q1=v["q1"] if c in IN_SUPPORT else "",
                     RU_CLAMP_minus_ADAPT_up=100 * v["ru_clamp_minus_adapt_hi"], Q0=v["q0"] if c == CONTROL else "",
                     ADAPT_minus_NOM=100 * v["vs_nom"], nom_lo=100 * v["vs_nom_lo"], Q2=v["q2"] if c in OUT else "",
                     ADAPT_minus_CLAMP=100 * v["vs_clamp"], Q3=v["q3"], informed_minus_ADAPT=100 * v["informed_minus_adapt"])
            r.update({("U_" + m): inf["U_" + m].mean() for m in ("NOM", "CLAMP", "RU_CLAMP", "ADAPT", "ADAPT_V", "CEPS_T", "RU_CEP_T")})
        rows.append(r)
    print("\nPer true dwell condition and size (points of survival probability; bounds one-sided 95%; control = same-seed reference):")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nRegret against the exact optimum of the true model, n = 5, informative cells (points):")
    for c in cfg["cond_order"]:
        print(c, {k: round(v, 2) for k, v in regret_table(df[(df["cond"] == c) & (df.n == 5)]).items()})
    print("\nTime per policy call (ms, memoised within the instance) and calls per instance, all rows:")
    print(timing_table(df).round(3).to_string(index=False))
    if a.review or cfg["role"] != "main": print("\nNo decision is applied (review mode)."); return
    print(f"\nstatements: Q0 {res['n_q0']}, Q1 {res['n_q1']}, Q2 {res['n_q2']} ({res['q2_defined']} defined); label: {res['label']}; against CLAMP+rollout: {res['needed_class']}")
    print("OUTCOME:", res["label"], "/", res["needed_class"])
    out = {f"cond{k[0]}_n{k[1]}": v for k, v in res["statements"].items()}
    json.dump(dict(label=res["label"], needed_class=res["needed_class"], statements=out), open(os.path.join(a.dir, "outcome_a20.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
