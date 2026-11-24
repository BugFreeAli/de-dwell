#!/usr/bin/env python3
"""
Analyse an A16 run (plan A16.5, A16.8):  python scripts/analyze_a16.py --dir results/a16_main [--review]
Primary claims P1 to P3 use only the group "main" at n = 5, 7, 9 (informative cells). n = 11 and the robustness variants are descriptive.
A missing cell, a wrong row count or a violation gives NO OUTCOME.
"""
import argparse, glob, itertools, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd
from dwell.analysis_a16 import BAND, CELL, boot_mean, claims, diff_strata, informative


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--review", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.dir, "run_meta.json")))["config"]
    n_cells = (len(cfg["sizes"]) * len(cfg["rho_grid"]) * len(cfg["s_grid"]) * len(cfg["a_grid"]) * len(cfg["lam_grid"]) +
               len(cfg.get("variants", [])) * len(cfg["rho_grid"]) * len(cfg["s_grid"]) * len(cfg.get("variant_a", [])) * len(cfg["lam_grid"]))
    fs = sorted(glob.glob(os.path.join(a.dir, "cell_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    bad = []
    if len(fs) != n_cells: bad.append(f"{len(fs)} cell files, expected {n_cells}")
    want = {("main", s["n"]): s["instances"] for s in cfg["sizes"]}
    for (g, n), N in want.items():
        if (df[(df.group == g) & (df.n == n)].groupby(CELL).size() != N).any(): bad.append(f"{g} n={n}: wrong row count")
    for v in cfg.get("variants", []):
        if (df[df.group == v["name"]].groupby(CELL).size() != cfg["variant_instances"]).any(): bad.append(f"{v['name']}: wrong row count")
    if int((df["violations"].fillna("") != "").sum()): bad.append("rows with violations")
    if bad: print("NO OUTCOME:", "; ".join(bad)); sys.exit(1)
    print(f"run {cfg['name']}  role {cfg['role']}  eval seed {cfg['eval_seed']}  cells {n_cells}  rows {len(df)}")
    dfs = {}
    for sz in cfg["sizes"]:
        n = sz["n"]; m = df[(df.group == "main") & (df.n == n)]; inf = informative(m); dfs[n] = inf
        ncell = len(inf.groupby(CELL)) if len(inf) else 0
        print(f"\nn = {n} ({sz['level']}): informative cells {ncell} of {len(m.groupby(CELL))}, instances {len(inf)}")
        if not len(inf): continue
        for c in ["U_opt", "U_RU_DPEV", "U_RU_CEP", "U_DPEV", "U_CEPS", "U_CEP"]:
            if c in inf.columns and inf[c].notna().all():
                extra = f"   regret {100 * (1 - inf[c].sum() / inf['U_opt'].sum()):.2f} points" if inf["U_opt"].notna().all() and c != "U_opt" else ""
                print(f"  {c[2:]:9s} mean U {inf[c].mean():.4f}{extra}")
        pt, lo, hi = boot_mean(diff_strata(inf, "U_CEPS", "U_CEP"))
        print(f"  CEPS minus CEP: {100 * pt:+.2f} points (one-sided 95% bounds: lower {100 * lo:+.2f}, upper {100 * hi:+.2f})")
        if "U_RU_CEP" in inf.columns and inf["U_RU_CEP"].notna().all():
            pt, lo, hi = boot_mean(diff_strata(inf, "U_RU_CEP", "U_CEPS"))
            print(f"  RU_CEP minus CEPS: {100 * pt:+.2f} points (lower {100 * lo:+.2f}, upper {100 * hi:+.2f})")
    parts = [dfs[n] for n in (5, 7, 9) if n in dfs and len(dfs[n])]
    if parts:
        print("\nGain of CEPS over CEP by goal tightness tau = need / E[killed value of CEP] (informative instances of n = 5, 7, 9):")
        allinf = pd.concat(parts)
        allinf = allinf.assign(gain=allinf.U_CEPS - allinf.U_CEP)
        allinf["tau_bin"] = pd.cut(allinf.tau, [0, 0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, np.inf])
        print(allinf.groupby("tau_bin", observed=True).agg(instances=("gain", "size"), gain_pts=("gain", lambda x: 100 * x.mean()),
                                                          U_CEP=("U_CEP", "mean"), U_CEPS=("U_CEPS", "mean")).round(3).to_string())
        print(f"Spearman correlation of |tau - 1| with the gain: {allinf['tau'].sub(1).abs().rank().corr(allinf['gain'].rank()):.3f}")
        if 5 in dfs and len(dfs[5]) and "U_DPEV" in dfs[5].columns:
            d5 = dfs[5]
            print(f"n = 5: Spearman correlation of |tau_ev - 1| with the DPEV loss (U_opt - U_DPEV): "
                  f"{d5['tau_ev'].sub(1).abs().rank().corr((d5['U_opt'] - d5['U_DPEV']).rank()):.3f}")
    for sz in cfg["sizes"]:
        if sz["level"] == "cheap":
            m = df[(df.group == "main") & (df.n == sz["n"])]; inf = informative(m)
            if len(inf):
                pt, lo, hi = boot_mean(diff_strata(inf, "U_CEPS", "U_CEP"))
                print(f"\nn = {sz['n']} (descriptive): informative cells {len(inf.groupby(CELL))}, CEPS minus CEP {100 * pt:+.2f} points (lower {100 * lo:+.2f})")
    for v in cfg.get("variants", []):
        g = df[df.group == v["name"]]; inf = informative(g)
        print(f"\nRobustness variant {v['name']} (n = {cfg['variant_n']}, D = {v['D0']}, values {v['value_set']}): informative cells {len(inf.groupby(CELL)) if len(inf) else 0} of {len(g.groupby(CELL))}")
        if len(inf):
            pt, lo, hi = boot_mean(diff_strata(inf, "U_CEPS", "U_CEP"))
            print(f"  CEPS minus CEP {100 * pt:+.2f} points (lower {100 * lo:+.2f}); DPEV regret {100 * (1 - inf['U_DPEV'].sum() / inf['U_opt'].sum()):.2f}, "
                  f"RU_DPEV {100 * (1 - inf['U_RU_DPEV'].sum() / inf['U_opt'].sum()):.2f}, RU_CEP {100 * (1 - inf['U_RU_CEP'].sum() / inf['U_opt'].sum()):.2f}, "
                  f"CEPS {100 * (1 - inf['U_CEPS'].sum() / inf['U_opt'].sum()):.2f}, CEP {100 * (1 - inf['U_CEP'].sum() / inf['U_opt'].sum()):.2f} points")
    if a.review or cfg["role"] != "main": print("\nNo decision is applied (review mode)."); return
    res = claims(dfs)
    print("\nPrimary claims:")
    for k, v in res.items():
        if k != "outcome": print(f"  {k}: " + ", ".join(f"{kk} {vv:.4f}" if isinstance(vv, float) else f"{kk} {vv}" for kk, vv in v.items()))
    print("\nOUTCOME:", res["outcome"])
    json.dump(res, open(os.path.join(a.dir, "outcome_a16.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
