#!/usr/bin/env python3
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

METHODS = ["DPMIS", "CEP", "ROLL_CEP"]
THRESH = 0.05
B = 2000
BOOT_SEED = 20261116


def load(cfg, d):
    files = sorted(glob.glob(os.path.join(d, "cell_*.csv")))
    n_cells = len(cfg["rho_grid"]) * len(cfg["s_grid"]) * len(cfg["a_grid"])
    if len(files) != n_cells:
        sys.exit(f"ERROR: found {len(files)} cell files, expected {n_cells}.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if (df.groupby(["rho", "s", "a"]).size() != cfg["n_instances"]).any():
        sys.exit("ERROR: a cell does not have n_instances rows.")
    return df


def tolerance(cfg, df):
    rng = np.random.default_rng(BOOT_SEED)
    out = []
    for (rho, s, a), g in df.groupby(["rho", "s", "a"]):
        g = g.sort_values("idx")
        cal = g["Vcal_1.0"].to_numpy()
        idx = rng.integers(0, len(g), size=(B, len(g)))
        for lam in cfg["lam_tol"]:
            V = np.stack([g[f"Vtol_{m}_b{lam}"].to_numpy() for m in METHODS])
            reg = 1.0 - V.sum(1) / cal.sum()
            k = int(np.argmin(reg))
            boot = (1.0 - V[:, idx].sum(2) / cal[idx].sum(1)).min(0)
            e = round(lam - 1.0, 4)
            out.append(dict(rho=rho, s=s, a=a, error=e, best_method=METHODS[k], regret_best=reg[k],
                            lo5=float(np.percentile(boot, 5)), flagged=bool(e != 0 and np.percentile(boot, 5) >= THRESH),
                            **{f"regret_{m}": reg[i] for i, m in enumerate(METHODS)}))
    return pd.DataFrame(out)


def regions(cfg, tol, errors):
    R, S, A = cfg["rho_grid"], cfg["s_grid"], cfg["a_grid"]
    pairs = []
    for e in errors:
        ok = {(r.rho, r.s, r.a) for r in tol.itertuples() if r.flagged and abs(r.error - e) < 1e-9}
        for c in ok:
            for ax, grid in enumerate((R, S, A)):
                i = grid.index(c[ax])
                if i + 1 < len(grid):
                    n = list(c)
                    n[ax] = grid[i + 1]
                    if tuple(n) in ok:
                        pairs.append((e, c, tuple(n)))
    return pairs


def adaptation(cfg, df):
    rng = np.random.default_rng(BOOT_SEED + 1)
    cells = [g.sort_values("idx") for _, g in df.groupby(["rho", "s", "a"])]
    N = len(cells[0])
    idx = rng.integers(0, N, size=(B, N))
    rows = []
    for lam in cfg["lam_ba"]:
        if abs(lam - 1.0) < 1e-9:
            continue
        cal = np.stack([c[f"Vcal_{lam}"].to_numpy() for c in cells])
        ba = np.stack([c[f"Vba_t{lam}"].to_numpy() for c in cells])
        nom = {m: np.stack([c[f"Vnom_{m}_t{lam}"].to_numpy() for c in cells]) for m in METHODS}
        def pooled(V, ii=None):
            if ii is None:
                return 1.0 - V.sum() / cal.sum()
            return 1.0 - V[:, ii].sum(axis=(0, 2)) / cal[:, ii].sum(axis=(0, 2))
        reg_nom = {m: pooled(nom[m]) for m in METHODS}
        best = min(METHODS, key=lambda m: reg_nom[m])
        reg_ba = pooled(ba)
        boot_nom = np.min(np.stack([pooled(nom[m], idx) for m in METHODS]), axis=0)
        boot_ba = pooled(ba, idx)
        rec = 1.0 - reg_ba / reg_nom[best] if reg_nom[best] > 1e-12 else np.nan
        boot_rec = 1.0 - boot_ba / np.maximum(boot_nom, 1e-12)
        rows.append(dict(true_scale=lam, nominal_error=round(1.0 / lam - 1.0, 4), best_nominal_method=best,
                         regret_nominal_best=reg_nom[best], regret_BA=reg_ba, recovery=rec,
                         recovery_lo5=float(np.percentile(boot_rec, 5)), recovery_hi95=float(np.percentile(boot_rec, 95)),
                         **{f"regret_nominal_{m}": reg_nom[m] for m in METHODS}))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    df = load(cfg, args.dir)
    tol = tolerance(cfg, df)
    tol.to_csv(os.path.join(args.dir, "summary_tolerance.csv"), index=False)
    ad = adaptation(cfg, df)
    ad.to_csv(os.path.join(args.dir, "summary_adaptation.csv"), index=False)
    p20 = regions(cfg, tol, [-0.2, 0.2])
    p40 = regions(cfg, tol, [-0.4, 0.4])
    fragile20, fragile40 = bool(p20), bool(p40)
    adapt_ok = bool(((ad["regret_nominal_best"] >= THRESH) & (ad["recovery"] >= 0.5) & (ad["recovery_lo5"] >= 0.3)).any())
    if fragile20 and adapt_ok:
        outcome = "FRAGILE at 20% and adaptation recovers: strong case for adaptive scheduling under dwell uncertainty"
    elif fragile20:
        outcome = "FRAGILE at 20% but adaptation does not recover enough: tolerance and calibration analysis is the contribution"
    elif fragile40:
        outcome = "FRAGILE only at 40%: moderate sensitivity, weak direction"
    else:
        outcome = "TOLERANT: stop rule met, drop this direction"
    res = dict(variant=cfg.get("name"), fragile_20=fragile20, fragile_40=fragile40, adaptation_recovers=adapt_ok,
               outcome=outcome, n_instances_total=int(len(df)),
               pairs_20=[[e, list(a), list(b)] for e, a, b in p20[:20]], pairs_40=[[e, list(a), list(b)] for e, a, b in p40[:20]])
    json.dump(res, open(os.path.join(args.dir, "decision_a6.json"), "w"), indent=2)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    print("=== A6 outcome:", outcome, "===")
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("pairs")}, indent=2))
    print("\n--- tolerance curve: mean regret (%) over cells of the best method and of each method, by dwell error ---")
    t = tol.groupby("error")[["regret_best", "regret_DPMIS", "regret_CEP", "regret_ROLL_CEP"]].mean() * 100
    t["max_best"] = tol.groupby("error")["regret_best"].max() * 100
    t["cells_flagged"] = tol.groupby("error")["flagged"].sum()
    print(t.round(2).to_string())
    print("\n--- mean regret (%) of the best method by dwell error and label accuracy ---")
    print((100 * tol.pivot_table(index="a", columns="error", values="regret_best", aggfunc="mean")).round(2).to_string())
    print("\n--- by dwell error and load ---")
    print((100 * tol.pivot_table(index="rho", columns="error", values="regret_best", aggfunc="mean")).round(2).to_string())
    print("\n--- adaptation: planners built for the nominal scale against the Bayes-adaptive optimum (pooled over cells) ---")
    print(ad.round(4).to_string(index=False))
    print("\nflagged cells (|error| = 20%):")
    f = tol[tol["flagged"] & (tol["error"].abs() < 0.3)]
    print(f[["rho", "s", "a", "error", "best_method", "regret_best", "lo5"]].assign(regret_best=100 * f["regret_best"], lo5=100 * f["lo5"]).round(2).to_string(index=False))
    print("\nregion pairs at 20%:", p20[:6], "\nregion pairs at 40%:", p40[:6])


if __name__ == "__main__":
    main()
