#!/usr/bin/env python3
import glob, json, os, sys
import numpy as np
import pandas as pd

BASE = ["FAIR", "EDF", "LLF", "SE", "GMR", "BI"]
PREV = BASE + ["CEP", "LA1"]
VARIANTS = ["a3_main", "a3_wide", "a3_k3"]
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
        rules = BASE + list(cfg["extra_rules"])
        best_all = s[[f"regret_{r}" for r in rules]].min(axis=1)
        best_prev = s[[f"regret_{r}" for r in PREV]].min(axis=1)
        raw = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(d, "cell_*.csv"))])
        row = dict(variant=name, D=str(cfg["D"]), n_instances=cfg["n_instances"], cells=len(s),
                   mean_best_prev_pct=100 * best_prev.mean(), mean_best_all_pct=100 * best_all.mean(),
                   max_best_all_pct=100 * best_all.max(), cells_flagged=int(s["flagged"].sum()),
                   noncorner_flagged=int(((s["flagged"]) & (s["s"] >= MIN_S) & (s["rho"] <= MAX_RHO)).sum()),
                   noncorner_gap=bool(noncorner_pairs(cfg, s)))
        for r in ("CEP",) + tuple(x for x in cfg["extra_rules"] if x.startswith("ROLL_")):
            row[f"mean_{r}_pct"] = 100 * s[f"regret_{r}"].mean()
            row[f"ms_{r}"] = 1000 * raw[f"t_{r}"].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    print(out.round(2).to_string(index=False))
    out.to_csv("results/a3_summary.csv", index=False)
    print("non-corner gap = a pair of adjacent flagged cells, both with s >= 2 and rho <= 1.5; ms_* = mean seconds*1000 to evaluate the policy on one instance (all worlds)")


if __name__ == "__main__":
    main()
