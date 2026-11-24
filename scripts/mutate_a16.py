#!/usr/bin/env python3
"""
Mutation check for A16 (plan A16.7). Each mutation is (name, file, old, new, test files); `old` must match EXACTLY ONCE.
A mutation is caught if pytest fails on the mutated copy of src/ and tests/. Exit code 1 if any mutation survives.
Usage: python scripts/mutate_a16.py
"""
import os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E, A = "src/dwell/a16.py", "src/dwell/analysis_a16.py"
TE, TA = ["tests/test_a16.py"], ["tests/test_a16_analysis.py"]
MUTATIONS = [
    ("tau inverted", E, 'row["tau"] = need / row["E_CEP"] if row["E_CEP"] > 1e-12 else float("inf")',
     'row["tau"] = row["E_CEP"] / need if need > 0 else float("inf")', TE),
    ("expected value ignores the world weights", E, "        total += w * acc", "        total += acc", TE),
    ("goal built from the wrong share", E, "need = need_of(inst.values, lam)", "need = need_of(inst.values, 1.0 - lam)", TE),
    ("ru level also runs the exact methods", E, 'if level == "exact":', 'if level in ("exact", "ru"):', TE),
    ("variant instances ignored", E, "D0 = tuple(D0 or cfg[\"D0\"])", "D0 = tuple(cfg[\"D0\"])", TE),
    ("variant values ignored", E, "value_set = tuple(value_set or cfg[\"value_set\"])", "value_set = tuple(cfg[\"value_set\"])", TE),
    ("optimum comparison removed", E, 'if any(row[k] > row["U_opt"] + TOL for k in ("U_CEP", "U_CEPS", "U_RU_CEP", "U_DPEV", "U_RU_DPEV")):',
     "if False:", TE),
    ("bootstrap bounds swapped", A, "return float(point), float(np.percentile(out, 5)), float(np.percentile(out, 95))",
     "return float(point), float(np.percentile(out, 95)), float(np.percentile(out, 5))", TA),
    ("P1 uses the point estimate", A, 'holds=bool(lo >= P1_MIN)', 'holds=bool(m >= P1_MIN)', TA),
    ("P2 uses the point estimate", A, 'holds=bool(hi <= P2_MAX)', 'holds=bool(m <= P2_MAX)', TA),
    ("band without upper limit", A, "(tau >= BAND[0]) & (tau <= BAND[1])", "(tau >= BAND[0])", TA),
    ("P3 uses the point estimate", A, 'holds=bool(lo >= P3_MIN)', 'holds=bool(pt >= P3_MIN)', TA),
    ("CONFIRMED needs only P1 or the rest", A, "elif p1 and rest:", "elif p1 or rest:", TA),
    ("informative filter removed", A, "if INFO_LO <= g[ref].mean() <= INFO_HI]", "if True]", TA),
    ("reference always CEPS", A, 'return "U_RU_CEP" if "U_RU_CEP" in df.columns and df["U_RU_CEP"].notna().all() else "U_CEPS"',
     'return "U_CEPS"', TA),
    ("difference sign", A, "return [(g[a] - g[b]).to_numpy() for _, g in df.groupby(CELL)]",
     "return [(g[b] - g[a]).to_numpy() for _, g in df.groupby(CELL)]", TA),
    ("bootstrap not stratified", A, "out[b] = sum(s[rng.integers(0, len(s), len(s))].sum() for s in strata) / tot",
     "out[b] = sum(s[rng.integers(0, len(s), len(s))].sum() for s in strata) / (tot + 1)", TA),
]


def main():
    survivors = []
    for name, rel, old, new, tests in MUTATIONS:
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read().replace("\r\n", "\n")
        if src.count(old) != 1:
            sys.exit(f"ERROR: mutation '{name}': pattern found {src.count(old)} times in {rel} (must be exactly 1)")
        with tempfile.TemporaryDirectory() as tmp:
            for d in ("src", "tests", "scripts"):
                shutil.copytree(os.path.join(ROOT, d), os.path.join(tmp, d))
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
