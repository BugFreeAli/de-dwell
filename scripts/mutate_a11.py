#!/usr/bin/env python3
"""
Mutation check for A11 (plan A11.7, item 6). Each mutation is a (file, old, new) text replacement that must match
EXACTLY ONCE (the script stops loudly otherwise). The mutated copy of src/ and tests/ is run against tests/test_a11.py;
a mutation is 'caught' if pytest fails. Exit code 1 if any mutation survives.
Usage: python scripts/mutate_a11.py
"""
import os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M1, R1 = "src/dwell/m1.py", "src/dwell/rules1.py"
MUTATIONS = [
    ("rate evaluated at decision time instead of at the step start", M1,
     "        g = self.g_at(self.d[j] - ts)\n        x = prog[j]", "        g = self.g_at(self.d[j] - t)\n        x = prog[j]"),
    ("rate profile reversed", M1,
     "        return self.gtab[min(max(tau, 1), self.T)]", "        return self.gtab[self.T + 1 - min(max(tau, 1), self.T)]"),
    ("kill with strict inequality", M1,
     "        q = sum(p for p, r in post if r <= g)", "        q = sum(p for p, r in post if r < g)"),
    ("slew skipped", M1,
     "        return self.s if (cur != NO_TARGET and cur != j) else 0\n\n    def energy_left",
     "        return 0\n\n    def energy_left"),
    ("idle charged to the budget", M1,
     "return [(1.0, 0.0, (t + 1, cur, self.canon(t + 1, prog), e))]",
     "return [(1.0, 0.0, (t + 1, cur, self.canon(t + 1, prog), e + 1 if self.budget is not None else 0))]"),
    ("budget check skipped", M1,
     "        m = T if left is None else min(T, left)", "        m = T"),
    ("over-aggressive DEAD canonicalisation", M1,
     "            out.append(DEAD if p_max < self.rmin[i][x] else x)", "            out.append(DEAD if p_max <= self.rmin[i][x] else x)"),
    ("L variant uses the M duration", R1,
     '    if variant == "M":\n        g = model.g_at(model.d[j] - t)', '    if True:\n        g = model.g_at(model.d[j] - t)'),
    ("idle does not advance time", M1,
     "return [(1.0, 0.0, (t + 1, cur, self.canon(t + 1, prog), e))]", "return [(1.0, 0.0, (t, cur, self.canon(t, prog), e))]"),
    ("deadline off by one in the feasibility test", M1,
     "        T = self.d[j] - ts\n        if T < 1:\n            return False", "        T = self.d[j] - ts + 1\n        if T < 1:\n            return False"),
    ("idle always judged possible (idle_possible always true)", M1,
     "        if not self.allow_idle:\n            return False\n        t, cur, prog, e = state\n        return any(",
     "        if not self.allow_idle:\n            return False\n        t, cur, prog, e = state\n        return True or any("),
]


def main():
    survivors = []
    for name, rel, old, new in MUTATIONS:
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        if src.count(old) != 1:
            sys.exit(f"ERROR: mutation '{name}': pattern found {src.count(old)} times in {rel} (must be exactly 1)")
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(os.path.join(ROOT, "src"), os.path.join(tmp, "src"))
            shutil.copytree(os.path.join(ROOT, "tests"), os.path.join(tmp, "tests"))
            open(os.path.join(tmp, rel), "w", encoding="utf-8").write(src.replace(old, new))
            try:
                r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_a11.py", "-q", "-x", "-p", "no:cacheprovider"],
                                   cwd=tmp, capture_output=True, text=True, timeout=900)
                caught = r.returncode != 0
                tail = (r.stdout.strip().splitlines() or [""])[-1]
            except subprocess.TimeoutExpired:
                caught, tail = True, "timeout (counted as caught: the mutant hangs)"
        print(f"{'CAUGHT  ' if caught else 'SURVIVED'} | {name} | {tail}")
        if not caught:
            survivors.append(name)
    print(f"{len(MUTATIONS) - len(survivors)} of {len(MUTATIONS)} mutations caught")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
