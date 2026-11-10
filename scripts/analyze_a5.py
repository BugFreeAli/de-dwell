#!/usr/bin/env python3
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

METHODS = ["CEP", "ROLL_CEP", "ROLL_GMR", "GMR", "EDF"]
THRESH = 0.05
B = 2000
BOOT_SEED = 20261112
MIN_S = 2
MAX_RHO = 1.5


def load(cfg, d):
    files = sorted(glob.glob(os.path.join(d, "cell_*.csv")))
    n_cells = len(cfg["rho_grid"]) * len(cfg["s_grid"]) * len(cfg["a_grid"]) * len(cfg["q_grid"])
    if len(files) != n_cells:
        sys.exit(f"ERROR: found {len(files)} cell files, expected {n_cells}.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if (df.groupby(["rho", "s", "a", "q"]).size() != cfg["n_instances"]).any():
        sys.exit("ERROR: a cell does not have n_instances rows.")
    return df


def analyse(cfg, df):
    rng = np.random.default_rng(BOOT_SEED)
    out = []
    for (rho, s, a), g in df.groupby(["rho", "s", "a"]):
        base = g[g["q"] == 0.5].sort_values("idx")
        dp0 = base["V_dp"].to_numpy()
        orc = base["V_orc"].to_numpy()
        idx = rng.integers(0, len(base), size=(B, len(base)))
        for q, gq in g.groupby("q"):
            gq = gq.sort_values("idx")
            dp = gq["V_dp"].to_numpy()
            V = np.stack([gq["V_" + m].to_numpy() for m in METHODS])
            reg = 1.0 - V.sum(1) / dp.sum()
            k = int(np.argmin(reg))
            boot_reg = (1.0 - V[:, idx].sum(2) / dp[idx].sum(1)).min(0)
            gain = dp.sum() / dp0.sum() - 1.0
            boot_gain = dp[idx].sum(1) / dp0[idx].sum(1) - 1.0
            gap = orc.sum() - dp0.sum()
            closed = (dp.sum() - dp0.sum()) / gap if gap > 1e-12 else np.nan
            informative = q > 0.5
            out.append(dict(rho=rho, s=s, a=a, q=q, n=len(gq), informative=informative,
                            sense_gain=gain, sense_gain_lo5=float(np.percentile(boot_gain, 5)),
                            sense_flagged=bool(informative and np.percentile(boot_gain, 5) >= THRESH),
                            oracle_gap_closed=closed,
                            best_method=METHODS[k], regret_best=reg[k], regret_lo5=float(np.percentile(boot_reg, 5)),
                            planner_flagged=bool(informative and np.percentile(boot_reg, 5) >= THRESH),
                            **{f"regret_{m}": reg[i] for i, m in enumerate(METHODS)}))
    return pd.DataFrame(out)


def region(cfg, summ, col):
    R, S, A, Q = cfg["rho_grid"], cfg["s_grid"], cfg["a_grid"], [q for q in cfg["q_grid"]]
    ok = {(r.rho, r.s, r.a, r.q) for r in summ.itertuples() if getattr(r, col) and r.s >= MIN_S and r.rho <= MAX_RHO}
    pairs = []
    for c in ok:
        for ax, grid in enumerate((R, S, A, Q)):
            i = grid.index(c[ax])
            if i + 1 < len(grid):
                n = list(c)
                n[ax] = grid[i + 1]
                if tuple(n) in ok:
                    pairs.append((c, tuple(n)))
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
    sense_pairs = region(cfg, summ, "sense_flagged")
    plan_pairs = region(cfg, summ, "planner_flagged")
    inf = summ[summ["informative"]]
    nc = inf[(inf["s"] >= MIN_S) & (inf["rho"] <= MAX_RHO)]
    if sense_pairs and plan_pairs:
        outcome = "sensing is valuable and planners lose against the exact optimum: strong case for sensing-aware learning"
    elif sense_pairs:
        outcome = "sensing is valuable but belief-based planners handle it: sensing value is the contribution, not learning"
    else:
        outcome = "no robust value of in-dwell sensing in this model: no new direction"
    res = dict(variant=cfg.get("name"), sensing_value_region=bool(sense_pairs), planner_gap_region=bool(plan_pairs),
               outcome=outcome, n_cells=int(len(summ)), n_informative_cells=int(len(inf)),
               n_sense_flagged=int(inf["sense_flagged"].sum()), n_planner_flagged=int(inf["planner_flagged"].sum()),
               mean_sense_gain_noncorner=float(nc["sense_gain"].mean()), max_regret_best=float(inf["regret_best"].max()),
               mean_regret_best=float(inf["regret_best"].mean()), n_instances_total=int(len(df)),
               sense_pairs=[[list(a), list(b)] for a, b in sense_pairs[:30]], planner_pairs=[[list(a), list(b)] for a, b in plan_pairs[:30]])
    json.dump(res, open(os.path.join(args.dir, "decision_a5.json"), "w"), indent=2)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    print("=== A5 outcome:", outcome, "===")
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("pairs")}, indent=2))
    print("\n--- sensing gain of the exact optimum over q = 0.5 (%, mean over non-corner cells) by label accuracy a and signal quality q ---")
    print((100 * nc.pivot_table(index="a", columns="q", values="sense_gain", aggfunc="mean")).round(2).to_string())
    print("\n--- share of the oracle gap closed by the signal (mean over non-corner cells) ---")
    print(nc.pivot_table(index="a", columns="q", values="oracle_gap_closed", aggfunc="mean").round(3).to_string())
    print("\n--- regret (%) of the best planner against the exact optimum, by a and q (mean over non-corner cells) ---")
    print((100 * nc.pivot_table(index="a", columns="q", values="regret_best", aggfunc="mean")).round(2).to_string())
    print("\n--- mean regret (%) by method, informative cells ---")
    print((100 * inf[[f"regret_{m}" for m in METHODS]].mean()).round(2).to_string())
    print("\n--- flagged cells (sensing value or planner gap) ---")
    t = inf[inf["sense_flagged"] | inf["planner_flagged"]]
    print(t[["rho", "s", "a", "q", "sense_gain", "sense_flagged", "regret_best", "best_method", "planner_flagged"]].assign(sense_gain=100 * t["sense_gain"], regret_best=100 * t["regret_best"]).round(2).to_string(index=False))
    print("\nsensing pairs:", sense_pairs[:6], "\nplanner pairs:", plan_pairs[:6])


if __name__ == "__main__":
    main()
