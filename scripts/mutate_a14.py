#!/usr/bin/env python3
"""
Mutation check for A14 (plan A14.7). Each mutation is (name, file, old, new, test files); `old` must match EXACTLY ONCE.
The mutated copy of src/ and tests/ is run against the listed tests; a mutation is caught if pytest fails.
Exit code 1 if any mutation survives.   Usage: python scripts/mutate_a14.py
"""
import os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U, A = "src/dwell/m0u.py", "src/dwell/analysis_a14.py"
TM, TA = ["tests/test_a14.py"], ["tests/test_a14_analysis.py"]
NEED = "return math.ceil((1.0 - lam) * float(sum(values)) - 1e-9)"
FLAG = 'out["flagged_all"] = bool(info and out["lo5"] >= REL_THRESH and out["lo5_abs"] >= ABS_THRESH)'
MUTATIONS = [
    ("threshold rounded down instead of up", U, NEED, NEED.replace("math.ceil", "math.floor"), TM),
    ("tolerance used instead of survival share", U, NEED, "return math.ceil(lam * float(sum(values)) - 1e-9)", TM),
    ("goal reached counted only when strictly exceeded", U,
     "if acc >= self.need - 1e-9:\n            return 1.0", "if acc > self.need + 1e-9:\n            return 1.0", TM),
    ("accumulated value not updated in the DP", U, "val += p * self.V(nxt, acc + r)", "val += p * self.V(nxt, acc)", TM),
    ("hopeless-state prune too aggressive", U, "if acc + potential < self.need - 1e-9:", "if acc + potential < self.need + 1e-9:", TM),
    ("DP memo ignores the accumulated value", U, "key = (state, acc)\n        hit = self.memo.get(key)",
     "key = state\n        hit = self.memo.get(key)", TM),
    ("evaluation counts success only above the goal", U, "total += w * (1.0 if acc >= need - 1e-9 else 0.0)",
     "total += w * (1.0 if acc > need else 0.0)", TM),
    ("evaluation ignores the slew", U, "t2 = t + model.slew(cur, j) + 1", "t2 = t + 1", TM),
    ("CEPS accepts infeasible sequences", U, "if end <= model.d[j] + 1e-9:", "if True:", TM),
    ("rollout picks the worst action", U, "if q > best_v + EPS:", "if q < best_v - EPS:", TM),
    ("rollout ignores the accumulated value", U, "q = sum(p * bval(nxt, acc + r) for p, r, nxt in model.outcomes(state, j))",
     "q = sum(p * bval(nxt, acc) for p, r, nxt in model.outcomes(state, j))", TM),
    ("DPEV replaced by CEP", U, 'pols["DPEV"] = lambda st, ac: ev_pol(st)', 'pols["DPEV"] = pols["CEP"]', TM),
    ("regions allowed across two factors", A, "if flags.get(nb, False) and outside_corner(nb[1], nb[2]):", "if True:", TA),
    ("flag uses the point estimate instead of the bound", A, FLAG,
     'out["flagged_all"] = bool(info and out["regret_best"] >= REL_THRESH and out["abs_best"] >= ABS_THRESH)', TA),
    ("absolute floor removed", A, FLAG, 'out["flagged_all"] = bool(info and out["lo5"] >= REL_THRESH)', TA),
    ("informative filter removed", A, "info = INFO_LO <= mean_opt <= INFO_HI", "info = True", TA),
    ("MISMATCH decided on the ALL set", A, "reg_ev = find_regions(flag_ev)", "reg_ev = find_regions(flag_all)", TA),
    ("EV set contains every method", A, "ev_ix = [METHODS.index(m) for m in EV_BASED]", "ev_ix = list(range(len(METHODS)))", TA),
    ("best method chosen by the smallest sum", A, "k = int(np.argmax(s))", "k = int(np.argmin(s))", TA),
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
