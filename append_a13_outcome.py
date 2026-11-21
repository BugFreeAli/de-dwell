#!/usr/bin/env python3
"""
Append the A13 outcome and the missing A11 to A13 entries to the repo records (append-only, fails loudly).

Run from the repository root:   python append_a13_outcome.py
Files changed (text appended at the end, nothing above is touched):
  docs/analysis_plan_P0.md   A13 outcome (NO-GO)
  docs/record.md             A11, A12, A13 entries and a repository-state note (record.md stopped at A10)
  docs/CHANGELOG.md          entries for A11, A12, A13
  README.md                  A11 to A13 file list
Safety: the script stops without writing anything if a marker line is already present (applied twice), if a file is
missing, or if a file does not end with a newline. It keeps the line-ending style of each file (CRLF or LF).
"""
import os
import sys

ROOT = os.getcwd()

PLAN = """

#### A13 outcome (2026-11-20): NO-GO
Main run seed 20261241 after commit a603068 (gate hash 16bb7279...0098): 32 cells, 6400 rows, 0 invariant violations, wall 2021.2 s on 4 workers (laptop, console output supplied by the author). Decision table A13.9 applied by scripts/analyze_a13.py. No cell is flagged and none is point-flagged. Largest best-method regret 1.38% (n = 5, rho 1.5, s_bar 4, label accuracy 0.9, SPREAD, ROLL_CEP, lower bound 0.88%); the 12 largest cells lie between 1.0% and 1.4%, all ROLL_CEP. Best method is a rollout in 100% of cells. Mean regret over cells (points): ROLL_CEP 0.89, CEP 2.76, ROLL_GMR 3.12, BI 10.16, GMR 10.28, ROLL_EDF 10.06, LA1 9.59, SE 21.18, EDF 23.81, LLF 63.87, FAIR 69.62. First action of the DP differs from that of CEP in 38.3% of instances. Slew matrices: mean 2.80, coefficient of variation 0.71, closure changed 42% of instances. Spearman correlation of the slew coefficient of variation with the instance regret of the best method: 0.008. The audit (exploratory seed 1) had shown a maximum of 1.5%; the main run agrees. The control on the main seed (A13.8) has not been run. [DOC: console output; raw cell files not re-analysed by me]
Time: my hedge was about 15 to 35 minutes; measured 33.7 minutes (n = 5 cells took 80 to 1267 s each).
Meaning [OPEN, my reading]: geometry-dependent slew creates no headroom for the best of 11 methods at n = 4 and 5; a certainty-equivalent plan over all orders with exact slews, improved by one exact rollout step, stays within about 1.4% of the DP optimum. Together with A1 to A12 this extends the pattern that a certainty-equivalent plan with one exact rollout step stays within a few percent of the DP optimum in every M0-type structure tested at n up to 5.
Action per A13.9: A13 is closed as a negative result for the best of 11 methods at n = 4 and 5. The proposed rule (no further structure test, write the compact robustness paper) is not yet confirmed by the author; the decision is the author's.
"""

RECORD = """

## A11 (2026-11-17 to 2026-11-19, added 2026-11-20; record.md had stopped at A10)
- [CODE] Plan frozen in commit d44d9be, code and gate in 910f210 (22 tests, 11 of 11 mutations caught, audit 960 rows main grid and 480 rows control, seed 1, no violation). Main run seed 20261222: 48 cells, 9600 rows, 0 violations, wall 3571 s on 4 workers.
- [CODE] Outcome NO-GO: 0 flagged cells, largest best-method regret 4.12% (rho 1.1, s 2, a 0.5, gamma 2, eps 0.5; ROLL_CEP_L), mean over cells 0.62%; best method is a rollout in 95.8% of cells. Value of idle 0.42% at eps 0.5 and 0 at eps 0.8 and infinity. Recomputed from the raw cell files on 2026-11-20 (summary_cells.csv).
- [DOC] Plan text A11.r1 was committed in 910f210 and appended to the plan only after the run (append_a11_r1.py failed its end-of-file check); disclosed in the plan.

## A12 (2026-11-19 to 2026-11-20)
- [CODE] Plan frozen in commit 4d795d4, code and gate in ac3a94a (70 tests passed, 1 skipped, 12 of 12 mutations caught, audit 160 rows seed 1). Main run seed 20261228 (1600 rows) and noise run seed 20261229 (800 rows), 0 violations, commit ac3a94a.
- [CODE] Outcome REGIME-LIMITED: X true (best non-adaptive worst case 10.24%, 5th percentile 9.63), R true (best adaptive 3.62%, reduction 0.646, 5th percentile 0.614), S false (largest difference +0.30 points, 5th percentile 0.08), N false (noise +-2: 5.38%, 5th percentile 4.80). Minimax mixture equals the pure defender IID_DP. Saved analysis output re-read on 2026-11-20; matches.
- [OPEN] The three best adaptive defenders are built with knowledge of the attack family, and IID_DP has zero regret under PH50_14 by construction.

## A13 (2026-11-20)
- [CODE] Plan frozen in commit d412965 (before any code), code, r1, audits and gate in a603068 (25 tests, 10 of 10 mutations caught; audit seed 1, 640 rows matrix and 640 rows control, no violation; gate hash 16bb7279...0098). 
- [CODE] Audit review recomputed from the raw audit files on 2026-11-20 equals the console output: best regret at most 1.52%, mean best regret 0.72% (matrix) against 0.35% (uniform control).
- [DOC] Main run seed 20261241: 32 cells, 6400 rows, 0 violations, wall 2021.2 s. Outcome NO-GO: largest best-method regret 1.38%, no cell flagged, ROLL_CEP best in all cells. Details in the plan, A13 outcome. Control on the main seed not run.
- [CODE] Check of A13.r1 item 1 (sandbox, 2000 draws, n = 5, master seed 9): closure changed 60%, 60%, 27%, 61% of the matrices (SPREAD 2 and 4, CLUSTER 2 and 4); mean slew after rounding and closure 1.81, 3.80, 1.87, 3.82 against s_bar 2, 4, 2, 4, so the geometry cells have about 0.90 to 0.96 times the mean slew of the control.

## Repository state notes (review of 2026-11-20)
- [CODE] Full test suite: 346 passed in 39.7 s (sandbox, one core). The plan file is append-only over all commits (the A1 commit replaced the placeholder line "None yet.").
- [CODE] No git tags exist after a10-result (workflow decision of 2026-11-17: no branches, no tags); freeze points after A10 are commits d44d9be (A11), 4d795d4 (A12), d412965 (A13). Earlier text in the literature document that mentions tags p0-a11-plan and p0-a12-plan does not match the repository.
- [CODE] The plan holds the paragraph "A12 outcome (2026-11-19): REGIME-LIMITED" twice with identical text.
- [CODE] environment-lock.yml is a UTF-16 export of the conda base environment, not of de-dwell; run_meta.json files hold the actual Python, numpy and pandas versions.
- [CODE] The A13 gate hashes only src/dwell/m0g.py and scripts/run_a13.py; changes to dp.py, m0.py, rules.py, instances.py or analysis_a13.py would not close it.
- [OPEN] Raw per-instance cell files are git-ignored; keep a dated archive with SHA-256 outside git (results/a13_main, a12_main, a12_noise2, a11_main).
"""

CHANGELOG = """

## 2026-11-17 to 2026-11-19, A11 and A12 (added 2026-11-20)
- A11 (range-dependent dwell, idle, energy budget): src/dwell/m1.py, rules1.py, analysis_a11.py, scripts/run_a11.py, analyze_a11.py, a11_gate.py, mutate_a11.py, configs a11_*, tests/test_a11.py, test_a11_analysis.py, brute_m1.py. Plan A11 and A11.r1. Outcome NO-GO.
- A12 (budgeted hardening adversary): src/dwell/hard.py, analysis_a12.py, scripts/run_a12.py, analyze_a12.py, a12_gate.py, mutate_a12.py, configs a12_*, tests/test_a12.py, test_a12_analysis.py. Plan A12 and A12.r1. Outcome REGIME-LIMITED.

## 2026-11-20, A13 (geometry-dependent slew)
- Added src/dwell/m0g.py, analysis_a13.py, scripts/run_a13.py, analyze_a13.py, a13_gate.py, mutate_a13.py, configs a13_*, tests/test_a13.py, test_a13_analysis.py, brute_g.py. Plan A13 (d412965) and A13.r1 (a603068).
- Outcome NO-GO (main seed 20261241, 6400 rows, largest best-method regret 1.38%). docs/record.md: A11 to A13 entries and repository-state notes added.
"""

README = """

## Amendments A11 to A13 files
    src/dwell/m1.py, rules1.py        A11 model with approach, idle and energy budget; 22 methods
    src/dwell/hard.py                 A12 hardening adversary and defenders
    src/dwell/m0g.py                  A13 slew matrix model, bearings, oracle, row runner
    src/dwell/analysis_a11.py, analysis_a12.py, analysis_a13.py   go-rule and decision logic
    scripts/run_a11.py, run_a12.py, run_a13.py                     resumable runners (gate file required for main runs)
    scripts/a11_gate.py, a12_gate.py, a13_gate.py                  write results/aNN_gate.json (tests, mutations, audits)
    scripts/analyze_a11.py, analyze_a12.py, analyze_a13.py         decision tables
    scripts/mutate_a11.py, mutate_a12.py, mutate_a13.py            mutation checks
"""

TARGETS = [
    ("docs/analysis_plan_P0.md", PLAN, "#### A13 outcome (2026-11-20)"),
    ("docs/record.md", RECORD, "## A13 (2026-11-20)"),
    ("docs/CHANGELOG.md", CHANGELOG, "## 2026-11-20, A13 (geometry-dependent slew)"),
    ("README.md", README, "## Amendments A11 to A13 files"),
]


def main():
    plan = []
    for rel, text, marker in TARGETS:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            sys.exit(f"ERROR: {rel} not found (run from the repository root)")
        raw = open(path, "rb").read()
        if marker.encode("utf-8") in raw:
            sys.exit(f"ERROR: {rel} already contains '{marker}'; nothing written")
        if not raw.endswith(b"\n"):
            sys.exit(f"ERROR: {rel} does not end with a newline; nothing written")
        crlf = b"\r\n" in raw
        new = text.replace("\r\n", "\n")
        if crlf:
            new = new.replace("\n", "\r\n")
        plan.append((path, raw, new.encode("utf-8")))
    for path, raw, add in plan:           # all checks passed: write
        with open(path, "wb") as f:
            f.write(raw + add)
        print("appended", os.path.relpath(path, ROOT), f"(+{len(add)} bytes)")


if __name__ == "__main__":
    main()
