#!/usr/bin/env python3
"""
Mutation check for A18 (plan A18.5). Each mutation is (name, file, old, new); `old` must match EXACTLY ONCE.
A mutation is caught if pytest on tests/test_a18.py fails for the mutated copy. Exit code 1 if any mutation survives.
Usage: python scripts/mutate_a18.py
"""
import os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E, A, R = "src/dwell/a18.py", "src/dwell/analysis_a18.py", "scripts/run_a18.py"
MUTATIONS = [
    # evaluation under the true dwell
    ("true kill uses the nominal dwell", E, "if x[j] == D_true[c[j]]:", "if x[j] == model_b.D[c[j]]:"),
    ("slew time ignored in the true world", E, "t2 = t + model_b.slew(cur, j) + 1\n            x[j] += 1", "t2 = t + 1\n            x[j] += 1"),
    ("killed value counted as 1", E, "acc += model_b.v[j]\n                else:\n                    late += 1", "acc += 1.0\n                else:\n                    late += 1"),
    ("nominal planners evaluated with the nominal dwell", E, "u, lt = prob_goal_true(mb, Dt, p, need)", "u, lt = prob_goal_true(mb, D0, p, need)"),
    ("scale fixed at 1", E, "Dt = scaled_dwell(D0, scale)", "Dt = scaled_dwell(D0, 1.0)"),
    ("true model built with the nominal dwell", E, "mt = Model(dataclasses.replace(inst, D=Dt))", "mt = Model(inst)"),
    ("informed CEP built on the nominal model", E, 'cept = make_rule(mt, "CEP")', 'cept = make_rule(mb, "CEP")'),
    ("informed CEPS built on the nominal model", E, 'row["U_CEPS_T"] = prob_goal(mt, ceps_policy(mt, need), need)', 'row["U_CEPS_T"] = prob_goal(mt, ceps_policy(mb, need), need)'),
    ("optimum taken from the nominal model", E, 'row["U_opt_T"] = GoalDP(mt, need).value', 'row["U_opt_T"] = GoalDP(mb, need).value'),
    ("exact level does not run the exact optimum", E, 'if level == "exact":\n        row["U_opt_T"]', 'if False:\n        row["U_opt_T"]'),
    # analysis
    ("MOSTLY needs only one statement", A, '("MOSTLY" if holds >= MOSTLY_MIN else "NOT ROBUST")', '("MOSTLY" if holds >= 1 else "NOT ROBUST")'),
    ("ROBUST allows one failure", A, '"ROBUST" if holds == len(keys)', '"ROBUST" if holds >= len(keys) - 1'),
    ("P1 uses the point estimate", A, "p1=bool(g_lo >= P1_MIN)", "p1=bool(g >= P1_MIN)"),
    ("informative cells selected with the planner under test", A, 'REF = "U_CEPS_T"', 'REF = "U_CEPS"'),
    ("control is part of the label", A, "keys = [(sc, n) for sc in scales for n in sizes]", "keys = [(sc, n) for sc in (CONTROL,) + tuple(scales) for n in sizes]"),
    ("informative filter skipped", A, "    inf = informative(d)\n    if len(inf) == 0:\n        return dict(undefined=True)", "    inf = d\n    if len(inf) == 0:\n        return dict(undefined=True)"),
    ("SMALL uses the point estimate", A, '"SMALL" if c_hi <= C_MAX', '"SMALL" if c <= C_MAX'),
    ("AWARENESS-WINS uses the point estimate", A, '"AWARENESS-WINS" if w_lo >= P1_MIN', '"AWARENESS-WINS" if w >= P1_MIN'),
    ("ADEQUATE uses the point estimate", A, '"ADEQUATE" if r_hi <= C_MAX', '"ADEQUATE" if r <= C_MAX'),
    ("LARGE uses the point estimate", A, '"LARGE" if c_lo >= C_MAX', '"LARGE" if c >= C_MAX'),
    ("KNOWLEDGE-WINS uses the point estimate", A, '"KNOWLEDGE-WINS" if w_hi <= -P1_MIN', '"KNOWLEDGE-WINS" if w <= -P1_MIN'),
    ("ROLLOUT-WORTH uses the point estimate", A, '"ROLLOUT-WORTH" if r_lo >= C_MAX', '"ROLLOUT-WORTH" if r >= C_MAX'),
    ("regret against the wrong sign", A, '100 * (inf["U_opt_T"] - inf["U_" + m])', '100 * (inf["U_" + m] - inf["U_opt_T"])'),
    # runner
    ("main role accepts any seed", R, 'if cfg["role"] != "main" or cfg["eval_seed"] != MAIN_SEED:', "if False:"),
    ("gate file not required", R, 'if not os.path.exists(gp) or json.load(open(gp)).get("code_hash") != code_hash():', "if False:"),
    ("audit may use the main seed", R, '        if cfg["eval_seed"] == MAIN_SEED:\n            sys.exit("ERROR: the main seed may not be used for an audit.")',
     '        if False:\n            sys.exit("ERROR: the main seed may not be used for an audit.")'),
    ("only the first scale is run", R, 'for sc in cfg["scales"]:', 'for sc in cfg["scales"][:1]:'),
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
                r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_a18.py", "-q", "-x", "-p", "no:cacheprovider"],
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
