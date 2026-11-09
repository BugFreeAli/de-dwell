#!/usr/bin/env python3
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

METHODS = ["DPMIS", "CEP", "ROLL_CEP", "ROLL_GMR", "GMR", "EDF"]
THRESH = 0.05
B = 2000
BOOT_SEED = 20261111
MIN_S = 2
MAX_RHO = 1.5


def load(cfg, d):
    files = sorted(glob.glob(os.path.join(d, "cell_*.csv")))
    n_cells = len(cfg["rho_grid"]) * len(cfg["s_grid"]) * len(cfg["a_true_grid"])
    if len(files) != n_cells:
        sys.exit(f"ERROR: found {len(files)} cell files, expected {n_cells}.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    counts = df.groupby(["rho", "s", "a_true"]).size()
    if (counts != cfg["n_instances"]).any():
        sys.exit("ERROR: a cell does not have n_instances rows.")
    return df


def analyse(cfg, df):
    rng = np.random.default_rng(BOOT_SEED)
    out = []
    for (rho, s, a_true), g in df.groupby(["rho", "s", "a_true"]):
        dp = g["V_dp"].to_numpy()
        idx = rng.integers(0, len(g), size=(B, len(g)))
        for a_b in cfg["belief_grid"]:
            V = np.stack([g[f"V_{m}_b{a_b}"].to_numpy() for m in METHODS])
            reg = 1.0 - V.sum(1) / dp.sum()
            k = int(np.argmin(reg))
            boot = (1.0 - V[:, idx].sum(2) / dp[idx].sum(1)).min(0)
            wrong = abs(a_b - a_true) > 1e-12
            lo = float(np.percentile(boot, 5))
            out.append(dict(rho=rho, s=s, a_true=a_true, a_b=a_b, wrong=wrong, n=len(g),
                            best_method=METHODS[k], regret_best=reg[k], lo5=lo,
                            flagged=bool(wrong and lo >= THRESH), point_flagged=bool(wrong and reg[k] >= THRESH),
                            **{f"regret_{m}": reg[i] for i, m in enumerate(METHODS)}))
    return pd.DataFrame(out)


def noncorner_pairs(cfg, summ):
    R, S, A = cfg["rho_grid"], cfg["s_grid"], cfg["belief_grid"]
    pairs = []
    for at, g in summ.groupby("a_true"):
        ok = {(r.rho, r.s, r.a_b) for r in g.itertuples() if r.flagged and r.s >= MIN_S and r.rho <= MAX_RHO}
        for c in ok:
            for ax, grid in enumerate((R, S, A)):
                i = grid.index(c[ax])
                if i + 1 < len(grid):
                    n = list(c)
                    n[ax] = grid[i + 1]
                    if tuple(n) in ok:
                        pairs.append((at, c, tuple(n)))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    df = load(cfg, args.dir)
    summ = analyse(cfg, df)
    summ.to_csv(os.path.join(args.dir, "summary_cells.csv"), index=False)
    pairs = noncorner_pairs(cfg, summ)
    wrong = summ[summ["wrong"]]
    res = dict(variant=cfg.get("name"), non_corner_misspecification_gap=bool(pairs), n_cells=int(len(summ)),
               n_wrong_cells=int(len(wrong)), n_flagged=int(summ["flagged"].sum()),
               n_noncorner_flagged=int((summ["flagged"] & (summ["s"] >= MIN_S) & (summ["rho"] <= MAX_RHO)).sum()),
               max_regret_wrong=float(wrong["regret_best"].max()), mean_regret_wrong=float(wrong["regret_best"].mean()),
               mean_regret_correct=float(summ[~summ["wrong"]]["regret_best"].mean()), n_instances_total=int(len(df)),
               pairs=[[a, list(b), list(c)] for a, b, c in pairs[:50]])
    json.dump(res, open(os.path.join(args.dir, "decision_a4.json"), "w"), indent=2)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    print("=== A4 misspecification verdict:", "GAP" if pairs else "NO GAP", "===")
    print(json.dumps({k: v for k, v in res.items() if k != "pairs"}, indent=2))
    print("\n--- mean regret (%) of the best method, by true accuracy and believed accuracy (all cells) ---")
    print((100 * summ.pivot_table(index="a_true", columns="a_b", values="regret_best", aggfunc="mean")).round(2).to_string())
    print("\n--- same, non-corner cells only (s >= 2, rho <= 1.5) ---")
    nc = summ[(summ["s"] >= MIN_S) & (summ["rho"] <= MAX_RHO)]
    print((100 * nc.pivot_table(index="a_true", columns="a_b", values="regret_best", aggfunc="mean")).round(2).to_string())
    print("\n--- mean regret (%) by method, wrong beliefs only ---")
    print((100 * wrong[[f"regret_{m}" for m in METHODS]].mean()).round(2).to_string())
    print("\n--- mean regret (%) by method, correct belief ---")
    print((100 * summ[~summ["wrong"]][[f"regret_{m}" for m in METHODS]].mean()).round(2).to_string())
    print("\n--- cells with point estimate >= 5% (wrong beliefs) ---")
    t = wrong[wrong["point_flagged"]]
    print(t[["rho", "s", "a_true", "a_b", "best_method", "regret_best", "lo5", "flagged"]].assign(regret_best=100 * t["regret_best"], lo5=100 * t["lo5"]).round(2).to_string(index=False))
    print("\nnon-corner pairs:", pairs[:10])


if __name__ == "__main__":
    main()
