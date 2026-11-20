#!/usr/bin/env python3
"""
Mutation check for A13 (plan A13.6). Each mutation is (name, file, old, new, test files); `old` must match EXACTLY ONCE.
The mutated copy of src/ and tests/ is run against the listed tests; a mutation is caught if pytest fails.
Exit code 1 if any mutation survives.   Usage: python scripts/mutate_a13.py
"""
import os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G, A = "src/dwell/m0g.py", "src/dwell/analysis_a13.py"
TM, TA = ["tests/test_a13.py"], ["tests/test_a13_analysis.py"]
MUTATIONS = [
    ("model uses the uniform s instead of the matrix", G,
     "return self.sm[cur][j] if (cur != NO_TARGET and cur != j) else 0",
     "return self.s if (cur != NO_TARGET and cur != j) else 0", TM),
    ("first engagement charged a slew", G,
     "return self.sm[cur][j] if (cur != NO_TARGET and cur != j) else 0",
     "return self.sm[cur][j] if cur != j else 0", TM),
    ("row and column swapped in the model", G,
     "return self.sm[cur][j] if (cur != NO_TARGET and cur != j) else 0",
     "return self.sm[j][cur] if (cur != NO_TARGET and cur != j) else 0", TM),
    ("oracle uses the uniform s", G, "t += model.slew(last, i) + p[i]", "t += model.s + p[i]", TM),
    ("rounding rule changed", G, "int(math.floor(c * abs(theta[i] - theta[j]) + 0.5))",
     "int(math.floor(c * abs(theta[i] - theta[j])))", TM),
    ("bearings of CLUSTER drawn as SPREAD", G, 'if geom == "SPREAD":', "if True:", TM),
    ("shortest-path closure omitted", G, "m[i][j] = min(m[i][j], m[i][k] + m[k][j])", "m[i][j] = m[i][j]", TM),
    ("control run uses the matrix", G, "if control:\n        sm, changed = uniform_matrix(n, s), 0",
     "if False:\n        sm, changed = uniform_matrix(n, s), 0", TM),
    ("regions allowed across two factors", A, "if flags.get(nb, False) and outside_corner(nb[1], nb[2]):",
     "if True:", TA),
    ("flag uses the point estimate instead of the bound", A, "flagged=bool(lo >= THRESH)", "flagged=bool(reg[k] >= THRESH)", TA),
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
