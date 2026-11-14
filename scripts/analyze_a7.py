#!/usr/bin/env python3
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

NOMINAL = ["DPMIS", "CEP", "ROLL_CEP"]
B = 2000
BOOT_SEED = 20261130
MIX_P = {"SD10": 0.125, "SD14": 0.25, "SD16": 1.0 / 3.0, "SD20": 0.5}
CORE = [0.8, 1.0, 1.2]


def load(cfg, d):
    files = sorted(glob.glob(os.path.join(d, "cell_*.csv")))
    n_cells = len(cfg["rho_grid"]) * len(cfg["s_grid"]) * len(cfg["a_grid"])
    if len(files) != n_cells:
        sys.exit(f"ERROR: found {len(files)} cell files, expected {n_cells}.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if (df.groupby(["rho", "s", "a"]).size() != cfg["n_instances"]).any():
        sys.exit("ERROR: a cell does not have n_instances rows.")
    return df


class Pool:
    def __init__(self, df, seed):
        self.cells = [g.sort_values("idx") for _, g in df.groupby(["rho", "s", "a"])]
        self.N = len(self.cells[0])
        rng = np.random.default_rng(seed)
        self.idx = rng.integers(0, self.N, size=(B, self.N))

    def mat(self, col):
        return np.stack([c[col].to_numpy() for c in self.cells])

    def regret(self, col, truth):
        V, cal = self.mat(col), self.mat(f"Vcal_{truth}")
        pt = 1.0 - V.sum() / cal.sum()
        bt = 1.0 - V[:, self.idx].sum(axis=(0, 2)) / cal[:, self.idx].sum(axis=(0, 2))
        return pt, bt


def build(cfg, df):
    P = Pool(df, BOOT_SEED)
    truths = cfg["truths"]
    R = {}
    for t in truths:
        for m in NOMINAL:
            R[("nom", m, t)] = P.regret(f"Vnom_{m}_t{t}", t)
        for pn in cfg["priors"]:
            R[("BA", pn, t)] = P.regret(f"VBA_{pn}_t{t}", t)
            for m in cfg["practical"].get(pn, []):
                R[("prac", f"{pn}:{m}", t)] = P.regret(f"V{m}_{pn}_t{t}", t)
    return P, R


def mixture(R, kind, name, weights):
    pt = sum(w * R[(kind, name, t)][0] for t, w in weights.items())
    bt = sum(w * R[(kind, name, t)][1] for t, w in weights.items())
    return pt, bt


def weights_for(p):
    return {0.8: p, 1.0: 1.0 - 2.0 * p, 1.2: p}


def best_of(R, kind, names, weights):
    pts = {n: mixture(R, kind, n, weights) for n in names}
    best = min(names, key=lambda n: pts[n][0])
    boot = np.min(np.stack([pts[n][1] for n in names]), axis=0)
    return best, pts[best][0], boot, pts


def scenario_metrics(cfg, R):
    out = {}
    for label, p in MIX_P.items():
        w = weights_for(p)
        nb, nom_pt, nom_boot, _ = best_of(R, "nom", NOMINAL, w)
        entry = dict(p=p, sd=float(np.sqrt(2 * p) * 0.2), nominal_method=nb, nominal=nom_pt,
                     nominal_lo5=float(np.percentile(nom_boot, 5)), nominal_hi95=float(np.percentile(nom_boot, 95)))
        for pn in cfg["priors"]:
            if not all(t in cfg["truths"] for t in CORE):
                continue
            ceil_pt, ceil_boot = mixture(R, "BA", pn, w)
            names = [f"{pn}:{m}" for m in cfg["practical"].get(pn, [])]
            e = dict(ceiling=ceil_pt)
            if names:
                pb, pr_pt, pr_boot, per = best_of(R, "prac", names, w)
                gain = nom_pt - pr_pt
                gain_boot = nom_boot - pr_boot
                rec_boot = gain_boot / np.maximum(nom_boot, 1e-12)
                e.update(practical_method=pb.split(":")[1], practical=pr_pt, practical_lo5=float(np.percentile(pr_boot, 5)),
                         practical_hi95=float(np.percentile(pr_boot, 95)), gain=gain,
                         gain_lo5=float(np.percentile(gain_boot, 5)),
                         recovery=gain / nom_pt if nom_pt > 1e-12 else float("nan"),
                         recovery_lo5=float(np.percentile(rec_boot, 5)),
                         attainment=gain / (nom_pt - ceil_pt) if (nom_pt - ceil_pt) >= 0.0025 else float("nan"),
                         ceiling_recovery=(nom_pt - ceil_pt) / nom_pt if nom_pt > 1e-12 else float("nan"),
                         per_method={k.split(":")[1]: v[0] for k, v in per.items()})
            entry[pn] = e
        out[label] = entry
    return out


def safety_metrics(cfg, R, primary_prior="B", primary_label="SD16"):
    sc = scenario_metrics(cfg, R)
    meth = sc[primary_label][primary_prior].get("practical_method")
    out = dict(primary_method=meth)
    for t in cfg["truths"]:
        if isinstance(t, str):
            continue
        nb = min(NOMINAL, key=lambda m: R[("nom", m, t)][0])
        nom_pt = R[("nom", nb, t)][0]
        nom_boot = np.min(np.stack([R[("nom", m, t)][1] for m in NOMINAL]), axis=0)
        e = dict(nominal_method=nb, nominal=nom_pt)
        for pn in cfg["priors"]:
            e[f"BA_{pn}"] = R[("BA", pn, t)][0]
            for m in cfg["practical"].get(pn, []):
                e[f"{pn}:{m}"] = R[("prac", f"{pn}:{m}", t)][0]
            if pn == primary_prior and meth is not None:
                key = ("prac", f"{pn}:{meth}", t)
                diff = R[key][1] - nom_boot
                e["primary_minus_nominal"] = R[key][0] - nom_pt
                e["primary_minus_nominal_hi95"] = float(np.percentile(diff, 95))
        out[str(t)] = e
    return out


def iid_metrics(cfg, R):
    out = {}
    t = "iid"
    for m in NOMINAL:
        out[f"nominal_{m}"] = R[("nom", m, t)][0]
    for pn in cfg["priors"]:
        out[f"BA_{pn}"] = R[("BA", pn, t)][0]
        for m in cfg["practical"].get(pn, []):
            out[f"{pn}:{m}"] = R[("prac", f"{pn}:{m}", t)][0]
    return out


def per_cell_primary(cfg, df, R, meth, p=1.0 / 3.0):
    rows = []
    w = weights_for(p)
    nb = min(NOMINAL, key=lambda m: mixture(R, "nom", m, w)[0])
    for (rho, s, a), g in df.groupby(["rho", "s", "a"]):
        def reg(col, t):
            return 1.0 - g[col].sum() / g[f"Vcal_{t}"].sum()
        nom = sum(wi * reg(f"Vnom_{nb}_t{t}", t) for t, wi in w.items())
        pr = sum(wi * reg(f"V{meth}_B_t{t}", t) for t, wi in w.items()) if meth else float("nan")
        ba = sum(wi * reg(f"VBA_B_t{t}", t) for t, wi in w.items())
        rows.append(dict(rho=rho, s=s, a=a, nominal=100 * nom, practical=100 * pr, ceiling=100 * ba))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.dir, "run_meta.json")))
    cfg = meta["config"]
    df = load(cfg, args.dir)
    P, R = build(cfg, df)
    res = dict(name=cfg.get("name"), n_instances_total=int(len(df)), git_commit=meta.get("git_commit"),
               master_seed=cfg["master_seed"], noise_offsets=cfg["noise_offsets"], noise_probs=cfg["noise_probs"])
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    print(f"=== {cfg.get('name')}: {len(df)} instances, sec/instance mean {df['sec'].mean():.2f}, max {df['sec'].max():.2f} ===")
    if cfg["truths"] == ["iid"]:
        res["iid"] = iid_metrics(cfg, R)
        print("regret (%) against the exact optimum under independent per-threat scale:")
        for k, v in res["iid"].items():
            print(f"  {k:24s} {100*v:6.2f}")
        json.dump(res, open(os.path.join(args.dir, "metrics_a7.json"), "w"), indent=2)
        return
    print("\n--- pooled regret (%) by truth ---")
    rows = []
    for t in cfg["truths"]:
        r = {"truth": t}
        for m in NOMINAL:
            r[f"nom_{m}"] = 100 * R[("nom", m, t)][0]
        for pn in cfg["priors"]:
            r[f"BA_{pn}"] = 100 * R[("BA", pn, t)][0]
            for m in cfg["practical"].get(pn, []):
                r[f"{pn}:{m}"] = 100 * R[("prac", f"{pn}:{m}", t)][0]
        rows.append(r)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    if all(t in cfg["truths"] for t in CORE):
        sc = scenario_metrics(cfg, R)
        res["scenarios"] = sc
        print("\n--- expected regret (%) under symmetric scale error with SD of 10, 14, 16, 20 percent (truths 0.8, 1.0, 1.2 with weights p, 1-2p, p) ---")
        for label, e in sc.items():
            print(f"{label} (p={e['p']:.3f}, SD={100*e['sd']:.1f}%): nominal best {e['nominal_method']} {100*e['nominal']:.2f} [lo5 {100*e['nominal_lo5']:.2f}, hi95 {100*e['nominal_hi95']:.2f}]")
            for pn in cfg["priors"]:
                x = e[pn]
                line = f"   prior {pn}: ceiling {100*x['ceiling']:.2f}"
                if "practical" in x:
                    line += (f" | practical best {x['practical_method']} {100*x['practical']:.2f} gain {100*x['gain']:.2f} "
                             f"[lo5 {100*x['gain_lo5']:.2f}] recovery {x['recovery']:.2f} [lo5 {x['recovery_lo5']:.2f}] "
                             f"attainment {x['attainment']:.2f} ceiling recovery {x['ceiling_recovery']:.2f}")
                print(line)
        sf = safety_metrics(cfg, R)
        res["safety"] = sf
        print("\n--- per-truth regret (%) of nominal best, primary practical method (prior B) and ceilings ---")
        for t in cfg["truths"]:
            e = sf[str(t)]
            print(f"truth {t}: nominal {e['nominal_method']} {100*e['nominal']:.2f}", end="")
            if "primary_minus_nominal" in e:
                print(f" | primary({sf['primary_method']}) minus nominal {100*e['primary_minus_nominal']:+.2f} [hi95 {100*e['primary_minus_nominal_hi95']:+.2f}]", end="")
            print(" | " + " ".join(f"{k} {100*v:.2f}" for k, v in e.items() if k.startswith("BA_")))
        meth = sf["primary_method"]
        if meth:
            pc = per_cell_primary(cfg, df, R, meth)
            res["per_cell_SD16"] = pc.round(4).to_dict(orient="records")
            print("\n--- per cell, SD16 mixture, regret (%) ---")
            print(pc.round(2).to_string(index=False))
    json.dump(res, open(os.path.join(args.dir, "metrics_a7.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
