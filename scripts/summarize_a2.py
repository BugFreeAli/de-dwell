#!/usr/bin/env python3
import json, os, sys
import numpy as np
import pandas as pd

BASE = ["FAIR", "EDF", "LLF", "SE", "GMR", "BI"]
VARIANTS = ["a2_main_la1", "a2_wide", "a2_k3", "a2_decay05", "a2_decay0"]
GRID = dict(rho_grid=[0.8, 1.1, 1.5, 2.0], s_grid=[0, 2, 4], a_grid=[0.5, 0.7, 0.9])
MIN_S = 2
MAX_RHO = 1.5


def noncorner_pairs(cfg, summ):
    R, S, A = cfg["rho_grid"], cfg["s_grid"], cfg["a_grid"]
    ok = {(r.rho, r.s, r.a) for r in summ.itertuples() if r.flagged and r.s >= MIN_S and r.rho <= MAX_RHO}
    pairs = []
    for c in ok:
        for ax, grid in enumerate((R, S, A)):
            i = grid.index(c[ax])
            if i + 1 < len(grid):
                n = list(c)
                n[ax] = grid[i + 1]
                if tuple(n) in ok:
                    pairs.append((c, tuple(n)))
    return pairs


def main():
    rows = []
    for name in VARIANTS:
        d, c = f"results/{name}", f"configs/{name}.json"
        if not os.path.exists(os.path.join(d, "summary_cells.csv")):
            print("skipped (no summary yet):", name)
            continue
        cfg = json.load(open(c))
        s = pd.read_csv(os.path.join(d, "summary_cells.csv"))
        s = s[s["rho"].isin(GRID["rho_grid"]) & s["s"].isin(GRID["s_grid"]) & s["a"].isin(GRID["a_grid"])]
        if len(s) != 36:
            sys.exit(f"ERROR: {name} has {len(s)} cells on the A2 grid, expected 36")
        cfg = dict(cfg, **GRID)
        rules = BASE + list(cfg.get("extra_rules", []))
        best = s[[f"regret_{r}" for r in rules]].min(axis=1)
        pairs = noncorner_pairs(cfg, s)
        rows.append(dict(
            variant=name, K=len(cfg["D"]), D=str(cfg["D"]), kappa=cfg.get("kappa", 1.0), n=cfg["n"],
            n_instances=cfg["n_instances"], cells=len(s),
            mean_best_pct=100 * best.mean(), max_best_pct=100 * best.max(),
            cells_ge5=int((best >= 0.05).sum()), cells_flagged=int(s["flagged"].sum()),
            noncorner_flagged=int(((s["flagged"]) & (s["s"] >= MIN_S) & (s["rho"] <= MAX_RHO)).sum()),
            mean_CEP_pct=100 * s["regret_CEP"].mean(), mean_LA1_pct=100 * s["regret_LA1"].mean(),
            mean_GMR_pct=100 * s["regret_GMR"].mean(),
            noncorner_gap=bool(pairs)))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(out.round(2).to_string(index=False))
    out.to_csv("results/a2_summary.csv", index=False)
    print("non-corner gap = a pair of adjacent flagged cells, both with s >= 2 and rho <= 1.5")


if __name__ == "__main__":
    main()
