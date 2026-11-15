#!/usr/bin/env python3
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

NOMINAL = ["DPMIS", "CEP", "ROLL_CEP"]
B = 2000
BOOT_SEED = 20261205
K_TOL = 0.20
PAIR_REL, PAIR_ABS = 0.25, 0.005


def sigma(cfg):
    o = np.array(cfg["noise_offsets"], float)
    p = np.array(cfg["noise_probs"], float)
    return float(np.sqrt((p * o ** 2).sum() - (p * o).sum() ** 2))


def load(cfg, d):
    files = sorted(glob.glob(os.path.join(d, "cell_*.csv")))
    n_cells = len(cfg["rho_grid"]) * len(cfg["s_grid"]) * len(cfg["a_grid"])
    if len(files) != n_cells:
        sys.exit(f"ERROR: {d}: found {len(files)} cell files, expected {n_cells}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if (df.groupby(["rho", "s", "a"]).size() != cfg["n_instances"]).any():
        sys.exit(f"ERROR: {d}: a cell does not have n_instances rows")
    return df


def rows_for_run(cfg, df):
    cells = [g.sort_values("idx") for _, g in df.groupby(["rho", "s", "a"])]
    N = len(cells[0])
    idx = np.random.default_rng(BOOT_SEED).integers(0, N, size=(B, N))
    out = []
    s = sigma(cfg)
    for t in cfg["truths"]:
        cal = np.stack([c[f"Vcal_{t}"].to_numpy() for c in cells])
        pts, boots = {}, {}
        for m in NOMINAL:
            V = np.stack([c[f"Vnom_{m}_t{t}"].to_numpy() for c in cells])
            pts[m] = 1.0 - V.sum() / cal.sum()
            boots[m] = 1.0 - V[:, idx].sum(axis=(0, 2)) / cal[:, idx].sum(axis=(0, 2))
        best = min(NOMINAL, key=lambda m: pts[m])
        boot = np.min(np.stack([boots[m] for m in NOMINAL]), axis=0)
        delta = abs(t - 1.0) * cfg["D0"][0]
        out.append(dict(run=cfg["name"], sigma=s, truth=t, side="down" if t < 1 else "up", delta=round(delta, 6),
                        rho=delta / s, regret=pts[best], lo5=float(np.percentile(boot, 5)),
                        hi95=float(np.percentile(boot, 95)), method=best))
    return out


def criteria(tab, main_names, pair_name, pair_ref):
    res = {}
    c1_ok, c1 = True, {}
    for side in ("down", "up"):
        sub = tab[(tab.side == side) & (tab.delta == 1.0) & tab.run.isin(main_names)]
        k = (sub.regret * sub.sigma ** 2).to_numpy()
        ok = len(k) >= 2 and bool(np.all(np.abs(k - k.mean()) <= K_TOL * k.mean()))
        c1[side] = dict(K=[float(x) for x in k], runs=list(sub.run), ok=ok)
        c1_ok = c1_ok and ok
    res["C1_noise_scaling_at_one_step_shift"] = dict(ok=c1_ok, detail=c1)
    c2_ok, c2 = True, {}
    for side in ("down", "up"):
        sub = tab[(tab.side == side) & tab.run.isin(main_names)].sort_values("rho")
        ok = bool(np.all(np.diff(sub.regret.to_numpy()) > 0))
        c2[side] = dict(rho=[float(x) for x in sub.rho], regret=[float(x) for x in sub.regret], ok=ok)
        c2_ok = c2_ok and ok
    res["C2_monotone_in_ratio"] = dict(ok=c2_ok, detail=c2)
    c3_ok, c3 = True, {}
    for side in ("down", "up"):
        a = tab[(tab.side == side) & (tab.run == pair_name) & (tab.delta == 2.0)]
        b = tab[(tab.side == side) & (tab.run == pair_ref) & (tab.delta == 1.0)]
        if len(a) != 1 or len(b) != 1:
            c3[side] = dict(ok=False, note="pair missing")
            c3_ok = False
            continue
        ra, rb = float(a.regret.iloc[0]), float(b.regret.iloc[0])
        ok = abs(ra - rb) <= max(PAIR_REL * rb, PAIR_ABS)
        c3[side] = dict(doubled=ra, reference=rb, rho_doubled=float(a.rho.iloc[0]), rho_reference=float(b.rho.iloc[0]), ok=ok)
        c3_ok = c3_ok and ok
    res["C3_equal_ratio_pair"] = dict(ok=c3_ok, detail=c3)
    if not res["C1_noise_scaling_at_one_step_shift"]["ok"]:
        res["outcome"] = "NOT-SUPPORTED: the inverse-variance scaling at a fixed shift does not hold on fresh seeds; drop the law and decide the direction"
    elif res["C2_monotone_in_ratio"]["ok"] and res["C3_equal_ratio_pair"]["ok"]:
        res["outcome"] = "SUPPORTED: loss is a function of shift over noise; proceed to the frozen confirmatory protocol A10"
    else:
        res["outcome"] = "NOISE-LAW-ONLY: inverse-variance scaling holds at a fixed shift but the joint ratio law does not; narrow the claim and decide with the author"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--main-runs", nargs="+", default=["a9_m1", "a9_m2", "a9_m3"])
    ap.add_argument("--pair-run", default="a9_x2")
    ap.add_argument("--pair-ref", default="a9_m1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for d in args.dirs:
        cfg = json.load(open(os.path.join(d, "run_meta.json")))["config"]
        rows += rows_for_run(cfg, load(cfg, d))
    tab = pd.DataFrame(rows)
    show = tab.copy()
    for c in ("regret", "lo5", "hi95"):
        show[c] = (100 * show[c]).round(2)
    show["sigma"] = show["sigma"].round(3)
    show["rho"] = show["rho"].round(3)
    pd.set_option("display.width", 200)
    print(show.sort_values(["side", "delta", "run"]).to_string(index=False))
    res = criteria(tab, args.main_runs, args.pair_run, args.pair_ref)
    print("\n" + json.dumps(res, indent=2))
    print("\n=== A9 outcome:", res["outcome"], "===")
    out = args.out or os.path.join(args.dirs[0], "..", "decision_a9.json")
    json.dump(dict(table=rows, criteria=res), open(out, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
