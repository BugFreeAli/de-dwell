#!/usr/bin/env python3
import json, os, sys
import numpy as np
import pandas as pd

BASE = ["FAIR", "EDF", "LLF", "SE", "GMR", "BI"]
COMMON = {(r, s, a) for r in (0.8, 1.1, 1.5) for s in (0, 2, 4) for a in (0.5, 0.8, 1.0)}
RUNS = [("results/a1_s1", "configs/a1_s1.json"), ("results/a1_v2", "configs/a1_v2.json"),
        ("results/a1_main", "configs/a1_main.json"), ("results/a1_v4", "configs/a1_v4.json"),
        ("results/a1_ratio", "configs/a1_ratio.json")]


def main():
    rows = []
    for d, c in RUNS:
        cfg = json.load(open(c))
        if not os.path.exists(os.path.join(d, "summary_cells.csv")):
            print("skipped (no summary yet):", d)
            continue
        vals = np.array(cfg["value_set"], dtype=float)
        cv = float(vals.std() / vals.mean())
        s = pd.read_csv(os.path.join(d, "summary_cells.csv"))
        s = s[[(r.rho, r.s, r.a) in COMMON for r in s.itertuples()]]
        if len(s) != len(COMMON):
            sys.exit(f"ERROR: {d} has {len(s)} common cells, expected {len(COMMON)}")
        best6 = s[[f"regret_{r}" for r in BASE]].min(axis=1)
        best7 = s[[f"regret_{r}" for r in BASE + ["CEP"]]].min(axis=1)
        rows.append(dict(run=cfg["name"], D=str(cfg["D"]), value_set=str(cfg["value_set"]), value_cv=round(cv, 3),
                         n_instances=cfg["n_instances"],
                         mean_best6_pct=100 * best6.mean(), mean_best7_pct=100 * best7.mean(),
                         max_best7_pct=100 * best7.max(),
                         cells_best6_ge5=int((best6 >= 0.05).sum()), cells_best7_ge5=int((best7 >= 0.05).sum()),
                         cells_flagged_best7=int(s["flagged"].sum()), cells_info_flagged=int(s["info_flagged"].sum()),
                         mean_CEP_pct=100 * s["regret_CEP"].mean(), mean_GMR_pct=100 * s["regret_GMR"].mean(),
                         mean_BI_pct=100 * s["regret_BI"].mean(), mean_EDF_pct=100 * s["regret_EDF"].mean()))
    out = pd.DataFrame(rows).sort_values("value_cv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(out.round(2).to_string(index=False))
    out.to_csv("results/a1_summary.csv", index=False)
    print("Wrote results/a1_summary.csv (27 common cells: rho 0.8/1.1/1.5, s 0/2/4, a 0.5/0.8/1.0; best6 and cells_*_ge5 are point estimates)")


if __name__ == "__main__":
    main()
