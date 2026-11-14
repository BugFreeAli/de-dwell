#!/usr/bin/env python3
import argparse, json, sys
import numpy as np

LABELS = ("SD10", "SD14", "SD16", "SD20")
REL_TOL = 0.25
ABS_TOL = 0.001


def noise_sd(m):
    o = np.array(m["noise_offsets"], float)
    p = np.array(m["noise_probs"], float)
    return float(np.sqrt((p * o ** 2).sum() - (p * o).sum() ** 2))


def points(m):
    s = noise_sd(m)
    if s <= 0:
        sys.exit("ERROR: run without noise has no finite ratio")
    return [(m["name"], lab, m["scenarios"][lab]["sd"] / s, m["scenarios"][lab]["nominal"],
             m["scenarios"][lab]["nominal_lo5"], m["scenarios"][lab]["nominal_hi95"]) for lab in LABELS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", nargs="+", required=True)
    ap.add_argument("--check", default=None)
    ap.add_argument("--predict-m", type=int, default=None)
    args = ap.parse_args()
    pts = []
    for f in args.fit:
        pts += points(json.load(open(f)))
    r = np.array([p[2] for p in pts])
    y = np.array([p[3] for p in pts])
    c = float((r ** 2 * y).sum() / (r ** 4).sum())
    b = np.polyfit(np.log(r), np.log(y), 1)
    print(f"fit on {len(pts)} points from {len(args.fit)} runs: nominal regret = c * r^2, c = {100*c:.2f} (percent)")
    print(f"free power-law exponent {b[0]:.3f}; max relative residual {np.abs((y - c * r ** 2) / y).max():.3f}")
    for p in sorted(pts, key=lambda x: x[2]):
        print(f"  {p[0]:10s} r={p[2]:.4f} observed {100*p[3]:5.2f}% [{100*p[4]:.2f}, {100*p[5]:.2f}] fitted {100*c*p[2]**2:5.2f}%")
    if args.predict_m is not None:
        m = args.predict_m
        from math import comb
        offs = list(range(-m, m + 1))
        probs = [comb(2 * m, m + o) / 4 ** m for o in offs]
        s = float(np.sqrt(sum(p * o * o for o, p in zip(offs, probs))))
        print(f"\npre-registered prediction for noise +-{m} (noise SD {s:.4f} steps):")
        for lab, sd in zip(LABELS, tuple(0.2 * (2 * q) ** 0.5 for q in (0.125, 0.25, 1 / 3, 0.5))):
            print(f"  {lab}: r={sd/s:.4f} predicted nominal regret {100*c*(sd/s)**2:.2f}%")
    if args.check:
        mc = json.load(open(args.check))
        ok = 0
        print(f"\ncheck against {args.check} (tolerance: {int(REL_TOL*100)}% relative or {100*ABS_TOL:.2f} points absolute, whichever larger)")
        for p in points(mc):
            pred = c * p[2] ** 2
            tol = max(REL_TOL * pred, ABS_TOL)
            hit = abs(p[3] - pred) <= tol
            ok += hit
            print(f"  {p[1]}: predicted {100*pred:.2f}% observed {100*p[3]:.2f}% [{100*p[4]:.2f}, {100*p[5]:.2f}] {'inside' if hit else 'OUTSIDE'}")
        verdict = "HOLDS" if ok == 4 else ("PARTLY" if ok == 3 else "FAILS")
        print(f"verdict: {verdict} ({ok} of 4)")


if __name__ == "__main__":
    main()
