#!/usr/bin/env python3
"""
Mutation check for A20 (plan A20.6). Each mutation is (name, file, old, new); `old` must match EXACTLY ONCE.
A mutation is caught if pytest on tests/test_a20.py fails for the mutated copy. Exit code 1 if any mutation survives.
Usage: python scripts/mutate_a20.py            (all mutations)
       python scripts/mutate_a20.py 0 8        (index range, for partial runs)
"""
import os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E, A, R = "src/dwell/a20.py", "src/dwell/analysis_a20.py", "scripts/run_a20.py"
MUTATIONS = [
    # planners and evaluation
    ("RU_CLAMP clamps like NOM", E, "min(x[j], self.mb.D[-1] - 1) for j in range(len(x)))\n        return self.ru_nom", "min(x[j], self.mb.D[-1]) for j in range(len(x)))\n        return self.ru_nom"),
    ("RU_CLAMP has no rollout", E, "return self.ru_nom((t, cur, self.mb.canon(t, prog)), acc)", "return self.ceps_nom((t, cur, self.mb.canon(t, prog)), acc)"),
    ("scale models not clamped outside the prior", E, "min(x[j], Dmax - 1) for j in range(len(x)))", "x[j] for j in range(len(x)))"),
    ("true dwell taken from the nominal model", E, 'Dt = tuple(cfg["conditions"][cond])', 'Dt = tuple(cfg["D0"])'),
    ("true model built with the nominal dwell", E, "mt = Model(dataclasses.replace(inst, D=Dt))", "mt = Model(inst)"),
    ("support of the planner ignored", E, 'L = Learner20(inst, need, cfg.get("support", SUPPORT))', "L = Learner20(inst, need, SUPPORT)"),
    ("optimum taken from the nominal model", E, 'row["U_opt_T"] = GoalDP(mt, need).value', 'row["U_opt_T"] = GoalDP(Model(inst), need).value'),
    ("informed rollout reference is not a rollout", E, 'row["U_RU_CEP_T"] = prob_goal(mt, ru_policy(mt, ceps_t, need), need)', 'row["U_RU_CEP_T"] = prob_goal(mt, ceps_t, need)'),
    ("time per call not divided by the calls", E, 'row["ms_" + name] = 1000.0 * tp.t / tp.n if tp.n else 0.0', 'row["ms_" + name] = 1000.0 * tp.t if tp.n else 0.0'),
    ("calls not counted", E, "        self.n += 1\n", "        self.n += 0\n"),
    # analysis
    ("Q0 uses the point estimate", A, "q0=bool(r_a_hi <= Q0_MAX)", "q0=bool(r_a <= Q0_MAX)"),
    ("Q1 uses the point estimate", A, "q1=bool(a_r_lo >= Q1_MIN)", "q1=bool(a_r >= Q1_MIN)"),
    ("Q2 uses the point estimate", A, "q2=bool(a_n_lo >= Q2_MIN)", "q2=bool(a_n >= Q2_MIN)"),
    ("informative cells selected with the planner under test", A, 'REF = "U_CEPS_T"', 'REF = "U_ADAPT"'),
    ("informative filter skipped", A, "    inf = informative(d)\n    if len(inf) == 0:\n        return dict(undefined=True)\n    a_r,", "    inf = d\n    if len(inf) == 0:\n        return dict(undefined=True)\n    a_r,"),
    ("LEARNING-BETTER uses the point estimate", A, '"LEARNING-BETTER" if a_r_lo >= Q3_MIN', '"LEARNING-BETTER" if a_r >= Q3_MIN'),
    ("CLAMP-BETTER uses the point estimate", A, '"CLAMP-BETTER" if a_r_hi <= -Q3_MIN', '"CLAMP-BETTER" if a_r <= -Q3_MIN'),
    ("ROBUST-LEARNER needs only 4 of 6 for Q1", A, "if q1 >= Q1_MIN_COUNT and q0", "if q1 >= 4 and q0"),
    ("ROBUST-LEARNER ignores Q0", A, "and q0 == len(q0k) and q2 == len(q2_defined):", "and q2 == len(q2_defined):"),
    ("ROBUST-LEARNER allows a Q2 failure", A, 'and q2 == len(q2_defined):\n            label = "ROBUST-LEARNER"', 'and q2 >= len(q2_defined) - 1:\n            label = "ROBUST-LEARNER"'),
    ("USEFUL needs only one Q2 statement", A, "elif q2 >= Q2_USEFUL_MIN and q0 == len(q0k):", "elif q2 >= 1 and q0 == len(q0k):"),
    ("USEFUL ignores Q0", A, "elif q2 >= Q2_USEFUL_MIN and q0 == len(q0k):", "elif q2 >= Q2_USEFUL_MIN:"),
    ("one defined Q2 statement is enough", A, "len(q2_defined) < Q2_DEFINED_MIN", "len(q2_defined) < 1"),
    ("LEARNING-NEEDED needs only one statement", A, '"LEARNING-NEEDED" if q1 >= Q1_MIN_COUNT', '"LEARNING-NEEDED" if q1 >= 1'),
    ("control is part of Q1", A, "q1k = [(c, n) for c in in_support for n in sizes]", "q1k = [(c, n) for c in (CONTROL,) + tuple(in_support) for n in sizes]"),
    ("in-support scales are part of Q2", A, "q2k = [(c, n) for c in out for n in sizes]", "q2k = [(c, n) for c in tuple(out) + tuple(in_support) for n in sizes]"),
    ("regret against the wrong sign", A, '100 * (inf["U_opt_T"] - inf["U_" + m])', '100 * (inf["U_" + m] - inf["U_opt_T"])'),
    # runner
    ("main role accepts any seed", R, 'if cfg["role"] != "main" or cfg["eval_seed"] != MAIN_SEED:', 'if cfg["role"] != "main":'),
    ("gate file not required", R, 'if not os.path.exists(gp) or json.load(open(gp)).get("code_hash") != code_hash():', "if False:"),
    ("audit may use the main seed", R, '        if cfg["eval_seed"] == MAIN_SEED:\n            sys.exit("ERROR: the main seed may not be used for an audit.")', '        if False:\n            sys.exit("ERROR: the main seed may not be used for an audit.")'),
    ("only the first condition is run", R, 'for cd in cfg["cond_order"]:', 'for cd in cfg["cond_order"][:1]:'),
]


def main():
    survivors = []
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (0, len(MUTATIONS))
    for name, rel, old, new in MUTATIONS[lo:hi]:
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read().replace("\r\n", "\n")
        if src.count(old) != 1:
            sys.exit(f"ERROR: mutation '{name}': pattern found {src.count(old)} times in {rel} (must be exactly 1)")
        with tempfile.TemporaryDirectory() as tmp:
            for d in ("src", "tests", "scripts", "configs"):
                shutil.copytree(os.path.join(ROOT, d), os.path.join(tmp, d))
            open(os.path.join(tmp, rel), "w", encoding="utf-8").write(src.replace(old, new))
            try:
                r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_a20.py", "-q", "-x", "-p", "no:cacheprovider"],
                                   cwd=tmp, capture_output=True, text=True, timeout=1200)
                caught, tail = r.returncode != 0, (r.stdout.strip().splitlines() or [""])[-1]
            except subprocess.TimeoutExpired:
                caught, tail = True, "timeout (counted as caught: the mutant hangs)"
        print(f"{'CAUGHT  ' if caught else 'SURVIVED'} | {name} | {tail}", flush=True)
        if not caught:
            survivors.append(name)
    n_run = len(MUTATIONS[lo:hi])
    print(f"{n_run - len(survivors)} of {n_run} mutations caught")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
