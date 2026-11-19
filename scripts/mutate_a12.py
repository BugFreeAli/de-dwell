#!/usr/bin/env python3
"""
Mutation check for A12 (plan A12.6). Each mutation is (name, file, old, new, test files); `old` must match EXACTLY ONCE.
The mutated copy of src/ and tests/ is run against the listed test files; a mutation is caught if pytest fails.
Exit code 1 if any mutation survives.   Usage: python scripts/mutate_a12.py
"""
import os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H, A = "src/dwell/hard.py", "src/dwell/analysis_a12.py"
TM, TA = ["tests/test_a12.py", "tests/test_a12_analysis.py"], ["tests/test_a12_analysis.py"]
MUTATIONS = [
    ("lambda_h applied to the wrong class", H,
     "base = scaled((b,), lam_h)[0] if hardened else b", "base = scaled((D0[0],), lam_h)[0] if hardened else b", TM),
    ("q and 1 - q swapped", H,
     "for hardened, pw in ((False, 1.0 - q), (True, q)):", "for hardened, pw in ((False, q), (True, 1.0 - q)):", TM),
    ("noise offsets shifted", H, "D.append(base + e)", "D.append(base + e + 1)", TM),
    ("DPMIS_1.2 built for scale 1.0", H, '"DPMIS_1.2": nominal_factory(planner, "DPMIS", 1.2),',
     '"DPMIS_1.2": nominal_factory(planner, "DPMIS", 1.0),', TM),
    ("IID_DP built for the wrong strategy", H, 'believed_factory(truth["PH50_14"][1], truth["PH50_14"][2].policy())',
     'believed_factory(truth["PH25_18"][1], truth["PH25_18"][2].policy())', TM),
    ("budget filter off (B20 = B40)", H, 'cap = {"B20": 0.2, "B40": 0.4}[level]', 'cap = {"B20": 0.4, "B40": 0.4}[level]', TM),
    ("regret denominator from another strategy", A, "return 1.0 - V[rows].sum(0) / Vs[rows].sum(0)",
     "return 1.0 - V[rows].sum(0) / np.roll(Vs[rows].sum(0), 1)", TA),
    ("linear program: sign of the constraint right-hand side", A, "T[:k, -1] = 1.0", "T[:k, -1] = -1.0", TA),
    ("even and odd halves swapped", A,
     "fit.append(rows[idx[rows] % 2 == 0])\n        ev.append(rows[idx[rows] % 2 == 1])",
     "fit.append(rows[idx[rows] % 2 == 1])\n        ev.append(rows[idx[rows] % 2 == 0])", TA),
    ("exploitability uses the minimum instead of the maximum", A, "E = Rev[:, cols].max(axis=1)", "E = Rev[:, cols].min(axis=1)", TA),
    ("sign of the S difference reversed", A, 'out["S_" + p] = float(Rev[shared_adapt, j].min() - Rev[nonad, j].min())',
     'out["S_" + p] = float(Rev[nonad, j].min() - Rev[shared_adapt, j].min())', TA),
    ("decision quantities taken from the fit half", A,
     "point = stats_from(regret_matrix(V, Vs, np.concatenate(fit)), regret_matrix(V, Vs, np.concatenate(ev)), level)",
     "point = stats_from(regret_matrix(V, Vs, np.concatenate(fit)), regret_matrix(V, Vs, np.concatenate(fit)), level)", TA),
]


def main():
    survivors = []
    for name, rel, old, new, tests in MUTATIONS:
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        if src.count(old) != 1:
            sys.exit(f"ERROR: mutation '{name}': pattern found {src.count(old)} times in {rel} (must be exactly 1)")
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(os.path.join(ROOT, "src"), os.path.join(tmp, "src"))
            shutil.copytree(os.path.join(ROOT, "tests"), os.path.join(tmp, "tests"))
            open(os.path.join(tmp, rel), "w", encoding="utf-8").write(src.replace(old, new))
            try:
                r = subprocess.run([sys.executable, "-m", "pytest", *tests, "-q", "-x", "-p", "no:cacheprovider"],
                                   cwd=tmp, capture_output=True, text=True, timeout=1200)
                caught, tail = r.returncode != 0, (r.stdout.strip().splitlines() or [""])[-1]
            except subprocess.TimeoutExpired:
                caught, tail = True, "timeout (counted as caught: the mutant hangs)"
        print(f"{'CAUGHT  ' if caught else 'SURVIVED'} | {name} | {tail}")
        if not caught:
            survivors.append(name)
    print(f"{len(MUTATIONS) - len(survivors)} of {len(MUTATIONS)} mutations caught")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
