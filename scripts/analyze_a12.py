#!/usr/bin/env python3
"""
Analyse A12 runs (plan A12.4). Reads only results/<name>/cell_*.csv.
  python scripts/analyze_a12.py --dir results/a12_main                              -> X, R, S (N pending)
  python scripts/analyze_a12.py --dir results/a12_main --dir2 results/a12_noise2    -> X, R, S, N and the outcome
  python scripts/analyze_a12.py --dir results/a12_audit --review                    -> non-gating audit review
A missing cell, a wrong row count or a row with violations gives NO OUTCOME; nothing is dropped.
"""
import argparse, glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd
from dwell.analysis_a12 import analyse, build_arrays, decide, flags, minimax_mixture, regret_matrix, strata
from dwell.hard import ADAPTIVE, DEFENDERS, NONADAPTIVE, STRAT_NAMES, budget_set


def load(d):
    cfg = json.load(open(os.path.join(d, "run_meta.json")))["config"]
    files = sorted(glob.glob(os.path.join(d, "cell_*.csv")))
    n_cells = len(cfg["rho_grid"]) * len(cfg["s_grid"]) * len(cfg["a_grid"])
    problems = []
    if len(files) != n_cells:
        problems.append(f"{d}: found {len(files)} cell files, expected {n_cells}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if (df.groupby(["rho", "s", "a"]).size() != cfg["n_instances"]).any():
        problems.append(f"{d}: a cell does not have n_instances rows")
    bad = int((df["violations"].fillna("") != "").sum())
    if bad:
        problems.append(f"{d}: {bad} rows with invariant violations")
    return cfg, df, problems


def pct(x):
    return f"{100 * x:6.2f}"


def report(df, res):
    V, Vs, cell, idx = build_arrays(df)
    fit, ev = strata(cell, idx)
    rf, re = np.concatenate(fit), np.concatenate(ev)
    Rev, Rfit, Rall = regret_matrix(V, Vs, re), regret_matrix(V, Vs, rf), regret_matrix(V, Vs, np.arange(len(df)))
    print("\nRegret (%) of every defender under every strategy, odd half (evaluation):")
    t = pd.DataFrame(100 * Rev, index=DEFENDERS, columns=STRAT_NAMES).round(2)
    print(t.to_string())
    for level in ("B40", "B20"):
        cols = [STRAT_NAMES.index(n) for n in budget_set(level)]
        print(f"\nExploitability E(d) at {level} (odd half) and the attacker's best response:")
        for i, d in enumerate(DEFENDERS):
            k = cols[int(np.argmax(Rev[i, cols]))]
            print(f"  {d:13s} {pct(Rev[i, cols].max())}   best response {STRAT_NAMES[k]}")
    cols = [STRAT_NAMES.index(n) for n in budget_set("B40")]
    x, w, v = minimax_mixture(Rfit[:, cols])
    print("\nMinimax mixture (solved on the even half, B40):")
    print("  weights:", {d: round(float(xi), 3) for d, xi in zip(DEFENDERS, x) if xi > 1e-9})
    print("  attacker mixture:", {STRAT_NAMES[c]: round(float(wi), 3) for c, wi in zip(cols, w) if wi > 1e-9})
    print(f"  worst-case regret: even half {pct(float((x @ Rfit[:, cols]).max()))}  odd half {pct(float((x @ Rev[:, cols]).max()))}")
    allc = np.where(np.isin(cell, np.unique(cell)))[0]
    print("\nMaximum over cell and strategy (attacker also chooses the cell), all instances, B40, descriptive:")
    for d in (min(NONADAPTIVE, key=lambda d: Rall[DEFENDERS.index(d), cols].max()), min(ADAPTIVE, key=lambda d: Rall[DEFENDERS.index(d), cols].max())):
        best = max(regret_matrix(V, Vs, np.where(cell == c)[0])[DEFENDERS.index(d), cols].max() for c in np.unique(cell))
        print(f"  {d:13s} {pct(best)}")
    p, q = res["point"], res["p5"]
    print(f"\nB40: E_nom {pct(p['E_nom'])} (5th pct {pct(q['E_nom'])}), E_def {pct(p['E_def'])}, reduction {p['reduction']:.3f} (5th pct {q['reduction']:.3f})")
    for s in ("PH50_14", "PH25_18", "PH50_18"):
        print(f"  S difference {s}: {pct(p['S_' + s])} points (5th pct {pct(q['S_' + s])})   [three strategies, no multiplicity correction]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--dir2"); ap.add_argument("--review", action="store_true")
    a = ap.parse_args()
    cfg, df, problems = load(a.dir)
    cfg2, df2 = None, None
    if a.dir2:
        cfg2, df2, p2 = load(a.dir2); problems += p2
    if problems:
        print("NO OUTCOME:", "; ".join(problems)); sys.exit(1)
    res = analyse(df, "B40")
    print(f"run {cfg['name']} role {cfg['role']} seed {cfg['master_seed']}  instances per cell {cfg['n_instances']}")
    report(df, res)
    print("\nReported without a rule: B20")
    r20 = analyse(df, "B20")
    print(f"  B20: E_nom {pct(r20['point']['E_nom'])}, E_def {pct(r20['point']['E_def'])}, reduction {r20['point']['reduction']:.3f}")
    if a.review:
        print("\nNo decision is applied in review mode."); return
    f1 = flags(res)
    f2 = None
    if df2 is not None:
        res2 = analyse(df2, "B40"); f2 = flags(res2)
        print(f"\nnoise run {cfg2['name']}: E_nom {pct(res2['point']['E_nom'])} (5th pct {pct(res2['p5']['E_nom'])}), reduction {res2['point']['reduction']:.3f} (5th pct {res2['p5']['reduction']:.3f})")
    outcome = decide(f1, f2)
    print(f"\nX {f1['X']}  R {f1['R']}  S {f1['S']}  N {None if f2 is None else bool(f2['X'] and f2['R'])}")
    print("OUTCOME:", outcome)
    json.dump(dict(flags=f1, noise2_flags=f2, outcome=outcome), open(os.path.join(a.dir, "outcome_a12.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
