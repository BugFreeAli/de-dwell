#!/usr/bin/env python3
import argparse, glob, os
import pandas as pd

M = ["DPMIS", "CEP", "ROLL_CEP"]
ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
args = ap.parse_args()
files = sorted(glob.glob(os.path.join(args.dir, "cell_*.csv")))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"{args.dir}: {len(files)} files, {len(df)} rows")


def reg(col, t):
    return 1 - df[col].sum() / df[f"Vcal_{t}"].sum()


R = {}
for t in (0.8, 1.0, 1.2):
    nom = {m: reg(f"Vnom_{m}_t{t}", t) for m in M}
    b = min(nom, key=nom.get)
    R[t] = (nom[b], b, reg(f"Vba_t{t}", t))
    print(f"truth {t}: best nominal {b} {100*nom[b]:.2f}%  ceiling {100*R[t][2]:.2f}%")
for lab, p in (("SD10", 0.125), ("SD14", 0.25), ("SD16", 1 / 3), ("SD20", 0.5)):
    w = {0.8: p, 1.0: 1 - 2 * p, 1.2: p}
    E = {m: sum(w[t] * reg(f"Vnom_{m}_t{t}", t) for t in w) for m in M}
    b = min(E, key=E.get)
    print(f"{lab}: best nominal {b} {100*E[b]:.2f}%  ceiling (prior B) {100*sum(w[t]*R[t][2] for t in w):.2f}%")
