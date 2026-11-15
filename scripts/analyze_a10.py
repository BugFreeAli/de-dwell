#!/usr/bin/env python3
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

NOMINAL = ["DPMIS", "CEP", "ROLL_CEP"]
B = 2000
BOOT_SEED = 20261220
P16 = 1.0 / 3.0
GROW_MIN = 0.005
MAT, MAT_LO = 0.03, 0.02
N_MAX = 10


def load(cfg, d):
    files = sorted(glob.glob(os.path.join(d, "cell_*.csv")))
    n_cells = len(cfg["rho_grid"]) * len(cfg["s_grid"]) * len(cfg["a_grid"])
    if len(files) != n_cells:
        sys.exit(f"ERROR: {d}: found {len(files)} cell files, expected {n_cells}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if (df.groupby(["rho", "s", "a"]).size() != cfg["n_instances"]).any():
        sys.exit(f"ERROR: {d}: a cell does not have n_instances rows")
    return df


def run_stats(cfg, df):
    cells = [g.sort_values("idx") for _, g in df.groupby(["rho", "s", "a"])]
    N = len(cells[0])
    idx = np.random.default_rng(BOOT_SEED).integers(0, N, size=(B, N))
    per = {}
    for t in (0.8, 1.2):
        cal = np.stack([c[f"Vcal_{t}"].to_numpy() for c in cells])
        pts, boots = {}, {}
        for m in NOMINAL:
            V = np.stack([c[f"Vnom_{m}_t{t}"].to_numpy() for c in cells])
            pts[m] = 1.0 - V.sum() / cal.sum()
            boots[m] = 1.0 - V[:, idx].sum(axis=(0, 2)) / cal[:, idx].sum(axis=(0, 2))
        best = min(NOMINAL, key=lambda m: pts[m])
        per[t] = dict(method=best, point=pts[best], boot=np.min(np.stack([boots[m] for m in NOMINAL]), axis=0))
    S = per[0.8]["point"] + per[1.2]["point"]
    S_boot = per[0.8]["boot"] + per[1.2]["boot"]
    off = max(abs(x) for x in cfg["noise_offsets"])
    return dict(run=cfg["name"], n=cfg["n"], m=off, instances=len(df), R08=per[0.8]["point"], R12=per[1.2]["point"],
                m08=per[0.8]["method"], m12=per[1.2]["method"], S=S, E16=P16 * S, E16_boot=P16 * S_boot)


def decide(stats):
    res = {}
    for m in (1, 2):
        sub = sorted([s for s in stats if s["m"] == m and s["n"] in (3, 4, 5)], key=lambda s: s["n"])
        if len(sub) != 3:
            res[f"m{m}"] = dict(ok=False, note="runs for n = 3, 4, 5 missing")
            continue
        e = [s["E16"] for s in sub]
        diff = sub[2]["E16_boot"] - sub[0]["E16_boot"]
        g1 = bool(e[0] < e[1] < e[2] and (e[2] - e[0]) >= GROW_MIN and np.percentile(diff, 5) > 0)
        lo5 = float(np.percentile(sub[2]["E16_boot"], 5))
        mat = bool(e[2] >= MAT and lo5 >= MAT_LO)
        slope = float(np.polyfit([3, 4, 5], e, 1)[0])
        n_star = float(5 + (MAT - e[2]) / slope) if slope > 0 else float("inf")
        res[f"m{m}"] = dict(E16=e, difference_5_minus_3=float(e[2] - e[0]), difference_lo5=float(np.percentile(diff, 5)),
                            grows=g1, material_at_n5=mat, E16_n5_lo5=lo5, slope_per_n=slope, n_star=n_star)
    r2 = res.get("m2", {})
    if r2.get("grows") and r2.get("material_at_n5") and res.get("m1", {}).get("grows"):
        out = "GROWS-MATERIAL: the loss grows with swarm size and is material at noise +-2; continue toward the top-tier path (A11 scale-up planning)"
    elif r2.get("grows") and r2.get("material_at_n5"):
        out = "MIXED: the loss grows and is material at noise +-2 but does not grow at noise +-1; decide with the author before any scale-up"
    elif r2.get("grows") and not r2.get("material_at_n5") and r2.get("n_star", float("inf")) <= N_MAX:
        out = "GROWS-EXTRAPOLATES: the loss grows but is not yet material at n = 5; the linear trend reaches 3% by n = 10; continue only after a larger-n check and the author's acceptance of the risk"
    else:
        out = "FLAT-OR-FALLING: no growth with swarm size that would make the loss material; write the compact paper"
    res["outcome"] = out
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    stats = []
    for d in args.dirs:
        cfg = json.load(open(os.path.join(d, "run_meta.json")))["config"]
        stats.append(run_stats(cfg, load(cfg, d)))
    pd.set_option("display.width", 200)
    tab = pd.DataFrame([{k: v for k, v in s.items() if k != "E16_boot"} for s in stats]).sort_values(["m", "n"])
    show = tab.copy()
    for c in ("R08", "R12", "S", "E16"):
        show[c] = (100 * show[c]).round(2)
    print(show.to_string(index=False))
    res = decide(stats)
    print("\n" + json.dumps(res, indent=2, default=float))
    print("\n=== A10 outcome:", res["outcome"], "===")
    out = args.out or os.path.join(args.dirs[0], "..", "decision_a10.json")
    json.dump(dict(table=tab.to_dict(orient="records"), criteria=res), open(out, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
