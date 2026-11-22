#!/usr/bin/env python3
"""
Mutation check for A15 (plan A15.7). Each mutation is (name, file, old, new, test files); `old` must match EXACTLY ONCE.
A mutation is caught if pytest fails on the mutated copy of src/ and tests/. Exit code 1 if any mutation survives.
Usage: python scripts/mutate_a15.py
"""
import os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L, A = "src/dwell/learned_g.py", "src/dwell/analysis_a15.py"
TM, TA = ["tests/test_a15.py"], ["tests/test_a15_analysis.py"]
MUTATIONS = [
    ("features ignore the accumulated value", L, "left = max(need - acc, 0.0)", "left = max(need, 0.0)", TM),
    ("no-acc variant keeps all columns", L, "N_FULL, N_NOACC, N_AUG = 20, 14, 21", "N_FULL, N_NOACC, N_AUG = 20, 20, 21", TM),
    ("CEPS flag always zero", L, "1.0 if (ceps_act is not None and j == ceps_act) else 0.0", "0.0", TM),
    ("policy does not stop at the goal", L, "    def pol(state, acc):\n        if acc >= need - 1e-9:\n            return None",
     "    def pol(state, acc):\n        if False:\n            return None", TM),
    ("policy takes the lowest score", L, "acts[int(np.argmax(scorer.scores(phi)))]", "acts[int(np.argmin(scorer.scores(phi)))]", TM),
    ("EV variant trained on the goal labels", L, 'y = ye if variant == "IL_EV" else yg', "y = yg", TM),
    ("no subsample of the instances", L, "size=max(1, int(frac * len(ids)))", "size=len(ids)", TM),
    ("exploration inverted", L, "if rng.random() < eps:", "if rng.random() > eps:", TM),
    ("hopeless states labelled 0", L, "acts.index(lg) if lg in acts else -1", "acts.index(lg) if lg in acts else 0", TM),
    ("gradient of the scorer wrong", L, "dZ = (G[:, :, None] * w2[None, None, :]) * (1.0 - H ** 2)",
     "dZ = (G[:, :, None] * w2[None, None, :])", TM),
    ("exact Wilcoxon counts only strictly larger", A, "if (np.array(signs) * ranks).sum() >= w_obs - 1e-9:",
     "if (np.array(signs) * ranks).sum() > w_obs + 1e-9:", TA),
    ("Holm without correction", A, "if p[i] <= alpha / (len(p) - rank):", "if p[i] <= alpha:", TA),
    ("materiality threshold removed", A, 'c["material"] = bool(c["mean_delta"] >= MARGIN) if not c["undefined"] else False',
     'c["material"] = bool(c["mean_delta"] >= -1.0) if not c["undefined"] else False', TA),
    ("BEATS needs only one size", A, 'if c7["win"] and c9["win"]:', 'if c7["win"] or c9["win"]:', TA),
    ("MATCHES margin sign", A, 'c7["mean_delta"] >= -MARGIN and c9["mean_delta"] >= -MARGIN',
     'c7["mean_delta"] >= MARGIN and c9["mean_delta"] >= MARGIN', TA),
    ("time ratio ignored", A, "if time_ratio >= TIME_RATIO:", "if time_ratio >= 0:", TA),
    ("informative filter removed", A, 'if INFO_LO <= g["U_RU_CEP"].mean() <= INFO_HI:', "if True:", TA),
    ("seed paired with itself", A, 'vb = float(d["U_" + b].mean()) if b in ("RU_CEP", "CEPS") else seed_values(d, b)',
     'vb = float(d["U_" + b].mean()) if b in ("RU_CEP", "CEPS") else seed_values(d, a)', TA),
    ("cheap-planner flag needs one size only", A, 'for k in (("IL_A", "CEPS", 5), ("IL_AD", "CEPS", 7), ("IL_AD", "CEPS", 9))))',
     'for k in (("IL_A", "CEPS", 5),)))', TA),
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
