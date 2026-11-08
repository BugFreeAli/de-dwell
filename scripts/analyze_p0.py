#!/usr/bin/env python3
"""
Analyse a P0 run according to docs/analysis_plan_P0.md. Reads only the raw per-instance CSVs.

Definitions (frozen in the plan):
  regret_r(cell)   = 1 - sum_i V_r,i / sum_i V_dp,i                     (ratio of sums over the instances of a cell)
  best simple rule = the rule with the smallest regret in that cell (selection on the same data: conservative for the go rule)
  CI               = instance-level percentile bootstrap (B=2000, resampling recomputes the best rule each time)
  flagged cell     = lower 5th percentile (one-sided 95%) of best-rule regret >= 5%
  point-flagged    = point estimate of best-rule regret >= 5%
  region           = two cells that are face-adjacent on the (rho, s, a) grid
  GO               = a region of flagged cells;  MARGINAL = no GO but a region of point-flagged cells;  NO-GO otherwise
  excess regret    = regret_best(a) - regret_best(a=1.0) at the same (rho, s): the part of the loss that exists only because the
                     hardness class is uncertain (a=1 reveals the class, so its regret is purely structural scheduling loss).
                     Paired instance bootstrap (same instance indices in both cells). excess-flagged: one-sided 95% lower bound >= 2 points.
  info attribution = 'information-sensitive' if a region of cells is both flagged and excess-flagged, else 'structural only'.
Usage: python scripts/analyze_p0.py --config configs/p0_main.json --dir results/p0_main
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

RULES = ["FAIR", "EDF", "LLF", "SE", "GMR", "BI"]
THRESH = 0.05
EXCESS_THRESH = 0.02   # percentage-point effect size for the companion 'information-sensitive' test (plan Section 5)
B = 2000
BOOT_SEED = 20261109


def load(cfg, d, allow_partial=False):
    files = sorted(glob.glob(os.path.join(d, "cell_*.csv")))
    n_cells = len(cfg["rho_grid"]) * len(cfg["s_grid"]) * len(cfg["a_grid"])
    if len(files) != n_cells and not allow_partial:
        sys.exit(f"ERROR: found {len(files)} cell files, expected {n_cells}. Finish the run or pass --allow-partial.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    counts = df.groupby(["rho", "s", "a"]).size()
    if (counts != cfg["n_instances"]).any():
        sys.exit("ERROR: a cell does not have n_instances rows (outliers/rows must never be dropped).")
    return df


def best_regret(V, dp, idx=None):
    """Regret of the best rule (min over rules); optional bootstrap index matrix idx of shape (B, N)."""
    if idx is None:
        return (1.0 - V.sum(1) / dp.sum()).min()
    return (1.0 - V[:, idx].sum(2) / dp[idx].sum(1)).min(0)


def add_excess(df, summ):
    """Paired bootstrap of regret_best(a) - regret_best(a=1) at the same (rho, s)."""
    rng = np.random.default_rng(BOOT_SEED + 1)
    cols = {}
    for (rho, s_), grp in df.groupby(["rho", "s"]):
        full = grp[grp["a"] == 1.0].sort_values("idx")
        if full.empty or full["V_dp"].sum() <= 0:
            continue
        Vf = np.stack([full["V_" + r].to_numpy() for r in RULES]); dpf = full["V_dp"].to_numpy()
        for a, g in grp[grp["a"] < 1.0].groupby("a"):
            g = g.sort_values("idx")
            if g["V_dp"].sum() <= 0:
                continue
            V = np.stack([g["V_" + r].to_numpy() for r in RULES]); dp = g["V_dp"].to_numpy()
            idx = rng.integers(0, len(g), size=(B, len(g)))
            diff = best_regret(V, dp, idx) - best_regret(Vf, dpf, idx)
            cols[(rho, s_, a)] = (best_regret(V, dp) - best_regret(Vf, dpf), np.percentile(diff, 5))
    summ["excess_regret"] = [cols.get((r.rho, r.s, r.a), (np.nan, np.nan))[0] for r in summ.itertuples()]
    summ["excess_lo5"] = [cols.get((r.rho, r.s, r.a), (np.nan, np.nan))[1] for r in summ.itertuples()]
    summ["excess_flagged"] = [bool(x >= EXCESS_THRESH) if x == x else False for x in summ["excess_lo5"]]
    summ["info_flagged"] = summ["flagged"].fillna(False).astype(bool) & summ["excess_flagged"]
    return summ


def analyse(df):
    rng = np.random.default_rng(BOOT_SEED)
    out = []
    for (rho, s, a), g in df.groupby(["rho", "s", "a"]):
        dp = g["V_dp"].to_numpy()
        if dp.sum() <= 0:
            out.append(dict(rho=rho, s=s, a=a, n=len(g), undefined=True))
            continue
        V = np.stack([g["V_" + r].to_numpy() for r in RULES])            # (R, N)
        reg = 1.0 - V.sum(1) / dp.sum()                                   # regret per rule
        k = int(np.argmin(reg))
        idx = rng.integers(0, len(g), size=(B, len(g)))
        boot = 1.0 - V[:, idx].sum(2) / dp[idx].sum(1)                    # (R, B)
        best_b = boot.min(0)                                              # best rule recomputed per resample
        lo, hi = np.percentile(best_b, [5, 97.5])
        lo95two = np.percentile(best_b, 2.5)
        inst_loss = (V.max(0) < dp - 1e-9).mean()                         # share of instances where NO rule is optimal
        out.append(dict(
            rho=rho, s=s, a=a, n=len(g), undefined=False,
            rho_hat=g["rho_hat"].mean(), sigma_hat=g["sigma_hat"].mean(), info_bits=g["info_bits"].iloc[0],
            V_dp=dp.mean(), V_orc=g["V_orc"].mean(), total_value=g["total_value"].mean(),
            best_rule=RULES[k], regret_best=reg[k], ci_lo_2p5=lo95two, ci_lo_one_sided_5=lo, ci_hi_97p5=hi,
            regret_best_norm_total=(dp.sum() - V[k].sum()) / g["total_value"].sum(),
            info_gap=1.0 - dp.sum() / g["V_orc"].sum(), share_instances_no_rule_optimal=inst_loss,
            **{f"regret_{r}": reg[i] for i, r in enumerate(RULES)},
            flagged=bool(lo >= THRESH), point_flagged=bool(reg[k] >= THRESH)))
    return add_excess(df, pd.DataFrame(out))


def has_region(cfg, summ, col):
    """True if two face-adjacent grid cells both have summ[col] == True; returns the pairs."""
    R, S, A = cfg["rho_grid"], cfg["s_grid"], cfg["a_grid"]
    F = np.zeros((len(R), len(S), len(A)), bool)
    for _, r in summ[~summ["undefined"]].iterrows():
        F[R.index(r["rho"]), S.index(r["s"]), A.index(r["a"])] = bool(r[col])
    pairs = []
    for ax in range(3):
        a = np.moveaxis(F, ax, 0)
        both = a[1:] & a[:-1]
        for pos in zip(*np.nonzero(both)):
            lo = [pos[0], pos[1], pos[2]]
            hi = [pos[0] + 1, pos[1], pos[2]]
            def to_cell(p):
                q = [0, 0, 0]
                order = [ax] + [x for x in range(3) if x != ax]
                for j, o in enumerate(order):
                    q[o] = p[j]
                return (R[q[0]], S[q[1]], A[q[2]])
            pairs.append((to_cell(lo), to_cell(hi)))
    return bool(pairs), pairs


def figure(cfg, summ, path, csv_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R, S, A = cfg["rho_grid"], cfg["s_grid"], cfg["a_grid"]
    ref = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, len(A), figsize=(3.2 * len(A), 3.4), sharey=True, constrained_layout=True)
    vmax = max(10.0, 100 * np.nanmax(summ["regret_best"]))
    for ax, a in zip(np.atleast_1d(axes), A):
        M = np.full((len(S), len(R)), np.nan)
        for _, r in summ[(summ["a"] == a) & (~summ["undefined"])].iterrows():
            M[S.index(r["s"]), R.index(r["rho"])] = 100 * r["regret_best"]
        # assert plotted values equal the saved table values
        for _, r in ref[(ref["a"] == a) & (~ref["undefined"].astype(bool))].iterrows():
            assert abs(M[S.index(r["s"]), R.index(r["rho"])] - 100 * r["regret_best"]) < 1e-9
        im = ax.imshow(M, origin="lower", vmin=0, vmax=vmax, cmap="viridis", aspect="auto")
        for i in range(len(S)):
            for j in range(len(R)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] < 0.6 * vmax else "black")
        ax.set_xticks(range(len(R)), [str(x) for x in R], fontsize=7)
        ax.set_yticks(range(len(S)), [str(x) for x in S], fontsize=7)
        ax.set_title(f"a = {a}", fontsize=9)
        ax.set_xlabel("load ratio (nominal)", fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("slew s (steps)", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.85, label="regret of best simple rule (%)")
    fig.savefig(path, dpi=200)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    df = load(cfg, args.dir, args.allow_partial)
    summ = analyse(df)
    csv_path = os.path.join(args.dir, "summary_cells.csv")
    summ.to_csv(csv_path, index=False)
    ok = summ[~summ["undefined"]]
    go, go_pairs = has_region(cfg, summ, "flagged")
    marg, marg_pairs = has_region(cfg, summ, "point_flagged")
    decision = "GO" if go else ("MARGINAL" if marg else "NO-GO")
    info, info_pairs = has_region(cfg, summ, "info_flagged")
    attribution = ("information-sensitive" if info else "structural only") if go else "n/a"
    res = dict(decision=decision, info_attribution=attribution, threshold=THRESH, excess_threshold=EXCESS_THRESH, n_cells=int(len(summ)), n_undefined=int(summ["undefined"].sum()),
               n_flagged=int(ok["flagged"].sum()), n_info_flagged=int(ok["info_flagged"].sum()), n_point_flagged=int(ok["point_flagged"].sum()),
               go_pairs=[list(map(list, p)) for p in go_pairs[:50]], marginal_pairs=[list(map(list, p)) for p in marg_pairs[:50]],
               max_regret_best=float(ok["regret_best"].max()), n_instances_total=int(len(df)))
    h1c = ok[(ok["rho"] == 0.5) & (ok["s"] <= 1) & (ok["a"] >= 0.9)]
    res["h1_supported"] = bool(len(h1c) > 0 and (h1c["ci_hi_97p5"] < THRESH).all())
    json.dump(res, open(os.path.join(args.dir, "decision.json"), "w"), indent=2)
    figure(cfg, summ, os.path.join(args.dir, "fig_p0_regret_best.png"), csv_path)

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)
    print("=== P0 decision:", decision, "| information attribution:", attribution, "===")
    print(json.dumps({k: v for k, v in res.items() if k not in ("go_pairs", "marginal_pairs")}, indent=2))
    print("\n--- regret of best simple rule (%), by cell (point estimate, one-sided 95% lower bound, best rule) ---")
    t = ok.assign(regret_pct=100 * ok["regret_best"], lo5_pct=100 * ok["ci_lo_one_sided_5"])
    print(t[["rho", "s", "a", "regret_pct", "lo5_pct", "best_rule", "flagged"]].round(2).to_string(index=False))
    h1 = ok[(ok["rho"] == 0.5) & (ok["s"] <= 1) & (ok["a"] >= 0.9)]
    h1_ok = bool(len(h1) > 0 and (h1["ci_hi_97p5"] < THRESH).all())
    print(f"\n--- H1 check (frozen cells: rho=0.5, s<=1, a>=0.9): upper 97.5% bound < 5% in all {len(h1)} cells? {h1_ok} ---")
    print(h1.assign(regret_pct=100 * h1["regret_best"], hi97_pct=100 * h1["ci_hi_97p5"])[["rho", "s", "a", "regret_pct", "hi97_pct"]].round(2).to_string(index=False))
    print("\n--- excess regret vs a=1.0 (percentage points; paired bootstrap one-sided 95% lower bound) ---")
    ex = ok[ok["a"] < 1.0]
    print(ex.assign(excess_pct=100 * ex["excess_regret"], lo5_pct=100 * ex["excess_lo5"])[["rho", "s", "a", "excess_pct", "lo5_pct", "excess_flagged"]].round(2).to_string(index=False))
    print("\n--- marginal means of best-rule regret (%) along each axis (descriptive) ---")
    for ax in ["rho", "s", "a"]:
        print(ok.groupby(ax)["regret_best"].mean().mul(100).round(2).to_string(), "\n")
    print("--- regret (%) by rule, averaged over cells (descriptive) ---")
    print((100 * ok[[f"regret_{r}" for r in RULES]].mean()).round(2).to_string())
    print("\n--- information gap (oracle vs DP), mean over cells by a (%) ---")
    print((100 * ok.groupby("a")["info_gap"].mean()).round(2).to_string())
    print("\nGO region pairs:", go_pairs[:10], "\nMARGINAL region pairs:", marg_pairs[:10])
    print("\nWrote:", csv_path, os.path.join(args.dir, "decision.json"), os.path.join(args.dir, "fig_p0_regret_best.png"))


if __name__ == "__main__":
    main()
