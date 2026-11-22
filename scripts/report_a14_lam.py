#!/usr/bin/env python3
"""
Descriptive report of an A14 sweep over the leakage tolerance lam (plan A14.r1). Reads results/<name>/summary_cells.csv
(written by scripts/analyze_a14.py). No decision is made; the per-lam regions only describe where the flags of A14.5 fall.
  python scripts/report_a14_lam.py --dir results/a14_lam
Writes <dir>/lam_curve.csv.
"""
import argparse, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import pandas as pd
from dwell.analysis_a14 import find_regions


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True)
    d = ap.parse_args().dir
    S = pd.read_csv(os.path.join(d, "summary_cells.csv"))
    rows = []
    for lam, g in S.groupby("lam"):
        inf = g[g["informative"]]
        key = lambda r: (r.n, r.rho, r.s, r.a, r.lam)
        reg_ev = find_regions({key(r): bool(r.flagged_ev) for r in g.itertuples()})
        reg_all = find_regions({key(r): bool(r.flagged_all) for r in g.itertuples()})
        rows.append(dict(lam=lam, cells=len(g), informative=len(inf),
                         mean_U_opt=inf["U_opt"].mean() if len(inf) else float("nan"),
                         DPEV_pts=100 * inf["regret_DPEV"].mean() if len(inf) else float("nan"),
                         best_EV_pts=100 * inf["regret_best_ev"].mean() if len(inf) else float("nan"),
                         best_EV_max_pts=100 * inf["regret_best_ev"].max() if len(inf) else float("nan"),
                         best_ALL_pts=100 * inf["regret_best"].mean() if len(inf) else float("nan"),
                         best_ALL_max_pts=100 * inf["regret_best"].max() if len(inf) else float("nan"),
                         flagged_ev=int(g["flagged_ev"].sum()), flagged_all=int(g["flagged_all"].sum()),
                         regions_ev=len(reg_ev), regions_all=len(reg_all)))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(d, "lam_curve.csv"), index=False)
    print("Per lam, informative cells only; regret in points of relative regret (mean over informative cells unless 'max'):")
    print(T.round(2).to_string(index=False))
    print("\nNo decision is made here; flags and regions are those of plan A14.5.")


if __name__ == "__main__":
    main()
