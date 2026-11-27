#!/usr/bin/env python3
"""
Mutation check for A19 (plan A19.6). Each mutation is (name, file, old, new); `old` must match EXACTLY ONCE.
A mutation is caught if pytest on tests/test_a19.py fails for the mutated copy. Exit code 1 if any mutation survives.
Usage: python scripts/mutate_a19.py            (all mutations)
       python scripts/mutate_a19.py 0 8        (index range, for partial runs)
"""
import os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E, A, R = "src/dwell/a19.py", "src/dwell/analysis_a19.py", "scripts/run_a19.py"
MUTATIONS = [
    # belief and planners
    ("kill likelihood uses 'dwell at least x'", E, "if Ds[k] == xj)", "if Ds[k] >= xj)"),
    ("alive likelihood uses 'dwell at least x'", E, "if Ds[k] > xj)", "if Ds[k] >= xj)"),
    ("no fallback for an impossible observation", E, "if z <= 0.0:", "if False:"),
    ("killed threats not DEAD in the scale models", E, "prog = tuple(DEAD if killed[j] else x[j] for j in range(len(x)))", "prog = tuple(x[j] for j in range(len(x)))"),
    ("ADAPT ignores the posterior weights", E, "q[a] = q.get(a, 0.0) + w[s] * sum(", "q[a] = q.get(a, 0.0) + sum("),
    ("ADAPT_V ignores the posterior weights", E, "votes[a] = votes.get(a, 0.0) + w[s]", "votes[a] = votes.get(a, 0.0) + 1.0"),
    ("ADAPT_V takes the last best action", E, "if votes[a] > best_v + EPS:", "if votes[a] >= best_v - EPS:"),
    ("CLAMP clamps like NOM", E, "min(x[j], self.mb.D[-1] - 1)", "min(x[j], self.mb.D[-1])"),
    ("NOM clamps like CLAMP", E, "min(x[j], self.mb.D[-1]) for j in range(len(x)))\n        return self.ceps_nom((t, cur, self.mb.canon(t, prog)), acc)\n\n    def clamp",
     "min(x[j], self.mb.D[-1] - 1) for j in range(len(x)))\n        return self.ceps_nom((t, cur, self.mb.canon(t, prog)), acc)\n\n    def clamp"),
    ("rollout value uses the wrong base policy", E, "j = self.ceps[s](state, acc)\n        m = self.mod[s]", "j = self.ceps_nom(state, acc)\n        m = self.mod[s]"),
    # evaluation under the true dwell
    ("true kill uses the nominal dwell", E, "if x[j] == D_true[c[j]]:", "if x[j] == mb.D[c[j]]:"),
    ("killed value counted as 1", E, "acc += mb.v[j]", "acc += 1.0"),
    ("slew time ignored in the true world", E, "t2 = t + mb.slew(cur, j) + 1", "t2 = t + 1"),
    ("scale fixed at 1", E, "Dt = scaled_dwell(D0, scale)", "Dt = scaled_dwell(D0, 1.0)"),
    ("support of the planner ignored", E, 'L = Learner(inst, need, cfg.get("support", SUPPORT))', "L = Learner(inst, need, SUPPORT)"),
    ("true model built with the nominal dwell", E, "mt = Model(dataclasses.replace(inst, D=Dt))", "mt = Model(inst)"),
    ("informed rollout reference is not a rollout", E, 'row["U_RU_CEP_T"] = prob_goal(mt, ru_policy(mt, ceps_t, need), need)', 'row["U_RU_CEP_T"] = prob_goal(mt, ceps_t, need)'),
    ("optimum taken from the nominal model", E, 'row["U_opt_T"] = GoalDP(mt, need).value', 'row["U_opt_T"] = GoalDP(Model(inst), need).value'),
    # analysis
    ("P1 uses the point estimate", A, "p1=bool(g_lo >= P1_MIN)", "p1=bool(g >= P1_MIN)"),
    ("P2 uses the point estimate", A, "p2=bool(l_hi <= P2_MAX)", "p2=bool(l <= P2_MAX)"),
    ("S1 uses the point estimate", A, "s1=bool(c_lo >= S1_MIN)", "s1=bool(c >= S1_MIN)"),
    ("informative cells selected with the planner under test", A, 'REF = "U_CEPS_T"', 'REF = "U_ADAPT"'),
    ("informative filter skipped", A, "    inf = informative(d)\n    if len(inf) == 0:\n        return dict(undefined=True)\n    g,", "    inf = d\n    if len(inf) == 0:\n        return dict(undefined=True)\n    g,"),
    ("WORKS ignores the pooled recovery", A, 'and pool["recovery_lo"] >= R_POOL_MIN', "and True"),
    ("WORKS ignores P2", A, 'and p2 == len(p2_keys) and pool["recovery_lo"]', 'and pool["recovery_lo"]'),
    ("WORKS allows one failure of P1", A, "if p1 == len(p1_keys) and p2", "if p1 >= len(p1_keys) - 1 and p2"),
    ("HELPS needs only one statement", A, "elif p1 >= HELPS_MIN and", "elif p1 >= 1 and"),
    ("HELPS ignores P2", A, "elif p1 >= HELPS_MIN and p2 == len(p2_keys):", "elif p1 >= HELPS_MIN:"),
    ("LEARNING-ADDS needs only one statement", A, "s1 >= CLAMP_ADDS_MIN", "s1 >= 1"),
    ("control is part of P1", A, "p1_keys = [(sc, n) for sc in scales for n in sizes]", "p1_keys = [(sc, n) for sc in (CONTROL,) + tuple(scales) for n in sizes]"),
    ("recovery class uses the point estimate", A, 'return "HIGH" if lo >= R_HIGH else ("SOME" if lo >= R_SOME else "LOW")', 'return "HIGH" if point >= R_HIGH else ("SOME" if point >= R_SOME else "LOW")'),
    ("ratio bootstrap resamples numerator and denominator separately", A, "            ns += x[idx].sum(); ds += y[idx].sum()", "            ns += x[idx].sum(); ds += y[rng.integers(0, len(y), len(y))].sum()"),
    ("pilot ignores the loss at the control", A, "and loss <= PILOT_LOSS_MAX", "and True"),
    ("pilot threshold on recovery dropped", A, 'bool(pool["recovery"] >= PILOT_R_MIN and', "bool(True and"),
    ("regret against the wrong sign", A, '100 * (inf["U_opt_T"] - inf["U_" + m])', '100 * (inf["U_" + m] - inf["U_opt_T"])'),
    # runner
    ("main role accepts any seed", R, 'if cfg["role"] not in want or cfg["eval_seed"] != want[cfg["role"]]:', 'if cfg["role"] not in want:'),
    ("gate file not required", R, 'if not os.path.exists(gp) or json.load(open(gp)).get("code_hash") != code_hash():', "if False:"),
    ("audit may use the pilot seed", R, 'if cfg["eval_seed"] in (MAIN_SEED, PILOT_SEED):', "if cfg[\"eval_seed\"] in (MAIN_SEED,):"),
    ("only the first scale is run", R, 'for sc in cfg["scales"]:', 'for sc in cfg["scales"][:1]:'),
]


def main():
    survivors = []
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (0, len(MUTATIONS))      # optional index range (for partial runs)
    for name, rel, old, new in MUTATIONS[lo:hi]:
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read().replace("\r\n", "\n")
        if src.count(old) != 1:
            sys.exit(f"ERROR: mutation '{name}': pattern found {src.count(old)} times in {rel} (must be exactly 1)")
        with tempfile.TemporaryDirectory() as tmp:
            for d in ("src", "tests", "scripts", "configs"):
                shutil.copytree(os.path.join(ROOT, d), os.path.join(tmp, d))
            open(os.path.join(tmp, rel), "w", encoding="utf-8").write(src.replace(old, new))
            try:
                r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_a19.py", "-q", "-x", "-p", "no:cacheprovider"],
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
