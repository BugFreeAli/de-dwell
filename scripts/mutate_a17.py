#!/usr/bin/env python3
"""
Mutation check for A17 (plan A17.6). Each mutation is (name, file, old, new); `old` must match EXACTLY ONCE.
A mutation is caught if pytest on tests/test_a17.py fails for the mutated copy. Exit code 1 if any mutation survives.
Usage: python scripts/mutate_a17.py
"""
import os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A, R = "src/dwell/analysis_a17.py", "scripts/run_a17.py"
MUTATIONS = [
    ("MOSTLY needs only one statement", A, '("MOSTLY" if holds >= MOSTLY_MIN else "NOT ROBUST")', '("MOSTLY" if holds >= 1 else "NOT ROBUST")'),
    ("ROBUST allows one failure", A, '"ROBUST" if holds == len(keys)', '"ROBUST" if holds >= len(keys) - 1'),
    ("ADEQUATE uses the point estimate", A, '"ADEQUATE" if hi2 <= P2_MAX', '"ADEQUATE" if m2 <= P2_MAX'),
    ("ROLLOUT-WORTH uses the point estimate", A, '"ROLLOUT-WORTH" if lo2 >= P2_MAX', '"ROLLOUT-WORTH" if m2 >= P2_MAX'),
    ("P1 uses the point estimate", A, "p1=bool(lo >= P1_MIN)", "p1=bool(m >= P1_MIN)"),
    ("P3 pooled includes the base family", A, 'if f in VARIANTS and not out[(f, n)]["undefined"]:', 'if not out[(f, n)]["undefined"]:'),
    ("label includes the base family", A, "keys = [(f, n) for f in VARIANTS for n in sizes_by_family.get(f, SIZES)]",
     'keys = [(f, n) for f in ("base",) + VARIANTS for n in sizes_by_family.get(f, SIZES)]'),
    ("informative filter skipped", A, "    inf = informative(d)\n    if len(inf) == 0:", "    inf = d\n    if len(inf) == 0:"),
    ("main role accepts any seed", R, 'if cfg["role"] != "main" or cfg["eval_seed"] != MAIN_SEED:', "if False:"),
    ("gate file not required", R, 'if not os.path.exists(gp) or json.load(open(gp)).get("code_hash") != code_hash():', "if False:"),
    ("audit may use the main seed", R, '        if cfg["eval_seed"] == MAIN_SEED:\n            sys.exit("ERROR: the main seed may not be used for an audit.")',
     '        if False:\n            sys.exit("ERROR: the main seed may not be used for an audit.")'),
    ("only the first size of a family is run", R, 'for sz in fam["sizes"]:', 'for sz in fam["sizes"][:1]:'),
]


def main():
    survivors = []
    for name, rel, old, new in MUTATIONS:
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read().replace("\r\n", "\n")
        if src.count(old) != 1:
            sys.exit(f"ERROR: mutation '{name}': pattern found {src.count(old)} times in {rel} (must be exactly 1)")
        with tempfile.TemporaryDirectory() as tmp:
            for d in ("src", "tests", "scripts", "configs"):
                shutil.copytree(os.path.join(ROOT, d), os.path.join(tmp, d))
            open(os.path.join(tmp, rel), "w", encoding="utf-8").write(src.replace(old, new))
            try:
                r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_a17.py", "-q", "-x", "-p", "no:cacheprovider"],
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
