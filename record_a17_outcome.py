#!/usr/bin/env python3
"""
Record the A17 plan (committed late, see the deviation note) and the A17 outcome (ROBUST) in the repo records (append-only, fails loudly).
Run from the repository root:   python record_a17_outcome.py
  docs/analysis_plan_P0.md, docs/record.md, docs/CHANGELOG.md
Stops without writing anything if a marker is already present, a file is missing, or a file does not end with a newline.
Keeps the line-ending style (CRLF or LF) of each file.
"""
import os, sys

PLAN = """

### A17 (2026-11-24): planner robustness across instance families at n = 7 and 9 (plan written before any A17 code is committed)
Status: plan only. The freeze point is the commit that adds this text (workflow of 2026-11-17: no tags). No A17 run has been made on the author's machine when this text is committed.

**A17.1 Why, and what is already known (disclosure).** A16 confirmed on a fresh seed that CEPS beats CEP at n = 5, 7, 9 in the base instance family (two hardness classes D = (5, 10), values in {1, 2, 3}). Three robustness variants were run only at n = 5 and only descriptively: in each of them CEPS beat CEP (+4.58, +2.22, +3.95 points), but the regret gap between CEPS and RU_CEP was 5.1 (K3), 5.0 (wide) and 8.7 (equal) points against exact optimum, so the claim that the cheap planner is adequate (P2) held in the base family only. A17 tests, at the sizes where no exact optimum exists, whether the planner claims hold across instance families, with fresh seeds and pre-written rules. Nothing else was looked at: the only exploratory runs are timing runs (seed 41) and a 27-row audit (seed 3), none of which is used.
Expectation stated before the run [OPEN]: P1 is likely to hold in most statements (the effect was +2 to +7 points in every earlier look); the cheap planner is likely not adequate in at least some variants (ROLLOUT-WORTH or INCONCLUSIVE); the regime variable is likely to separate again.

**A17.2 Claims and labels (fixed now).** Informative cell: mean survival probability of RU_CEP in [0.10, 0.99]. Variant families: K3 (three hardness classes, D = (3, 6, 10)), wide (D = (3, 12)), equal (all values 2), spread (values 1, 3, 9; D = (5, 10)); the base family is run on the same seed as a control. Statements per variant family and size (K3, wide, equal and spread at n = 7 and 9, eight statements):
- P1(f, n): the one-sided 95% lower bound of the mean of U_CEPS minus U_CEP is at least 0.01.
- Cheap-planner class (not part of the label): ADEQUATE if the one-sided 95% upper bound of the mean of U_RU_CEP minus U_CEPS is at most 0.03; ROLLOUT-WORTH if its lower bound is at least 0.03; INCONCLUSIVE otherwise.
- P3 pooled (not part of the label): over the informative instances of all variant families at n = 7 and 9, the mean gain U_CEPS minus U_CEP for 0.8 <= tau <= 1.1 minus that of the other instances has a one-sided 95% lower bound of at least 0.03 (tau = need / E[killed value of CEP]).
- Label (P1 only), first match: NO OUTCOME (a statement has no informative cell); ROBUST (P1 holds in all 8 statements); MOSTLY (in at least 6); NOT ROBUST (otherwise). The base family is reported as the same-seed control and is not part of the label.

**A17.3 Design.** Eval seed 20261281 (not used before), bootstrap seed 20261282. Cells: rho in {1.1, 1.5}, s in {2, 4}, label accuracy in {0.5, 0.9}, lam in {0.25, 0.5} (16 cells per family and size). Instances per cell: n = 7: 40 (K3: 30); n = 9: 30 (K3: 20); 160 cells and 5280 rows in all. Every row is evaluated exactly (all hidden-class vectors enumerated) by src/dwell/a16.py at level ru (CEP, CEPS, RU_CEP; A14.4). The instance generator, the goal (need = ceil((1 - lam) W)) and the survival objective are those of A14 to A16; the label accuracy spreads the remaining probability evenly over the other classes (K3).
**A17.4 Statistics.** As in A16: instance-level bootstrap stratified by cell, 2000 resamples, one-sided 95% bounds, pooled mean over the informative cells of a family and size. Everything not listed in A17.2 is descriptive (mean survival probabilities, number of informative cells, the base control).
**A17.5 Verification before any main run.** tests/test_a17.py (7 tests; the row logic is covered by tests/test_a16.py): classes and P1 with intervals, labels ROBUST, MOSTLY, NOT ROBUST and NO OUTCOME, family-specific sizes, the base control outside the label, P3 pooled over the variants only, the job layout (160 cells, 5280 rows, K3 sizes and parameters, equal and spread values), gate and seed refusal. scripts/mutate_a17.py: 12 mutations, all caught. Audit on an exploratory seed. scripts/a17_gate.py writes results/a17_gate.json with the SHA-256 of nine code files (a16.py, analysis_a16.py, analysis_a17.py, m0u.py, m0.py, dp.py, rules.py, instances.py, run_a17.py); run_a17.py refuses the role main without a gate that carries the current hash.
**A17.6 Procedure and cost.** Commit this text. Add the code, run tests, mutations, audit and gate, commit. Run configs/a17_main.json once with 4 workers and analyse with scripts/analyze_a17.py --dir results/a17_main. Cost: from my sandbox timing runs on one core (per instance at n = 7 about 0.05 to 0.5 s, K3 up to 0.5 s; at n = 9 about 0.4 to 3.6 s per instance, K3 3.5 to 4.2 s on the two cells timed) I estimate about 6000 s in all; hedged for the author's laptop with 4 workers: 60 to 120 minutes [OPEN, not measured there]. The hardest K3 cells at n = 9 were not timed.
**A17.7 What each label means (not commitments).** ROBUST: the mismatch finding and the planner gain hold across the four families at the sizes without exact reference, and the classes show where the cheap planner is enough. MOSTLY: the families in which P1 fails are reported as limits. NOT ROBUST: the confirmed result is restricted to the base family and the paper must say so.
**A17.8 Limits stated now.** Planners only; the families are my choice and cover hardness classes, dwell spread, value equality and value heterogeneity, not other generator parameters (deadline shapes, arrival patterns, slew values); no exact reference exists at n = 7 and 9, so all statements are relative to CEP and CEPS; informative cells are selected with RU_CEP; the thresholds are those of A16 and the class boundaries for the cheap planner are my choice; the amendments A1 to A17 are post-hoc relative to each other and no multiplicity correction is applied across amendments.

**Deviation (2026-11-25): this plan was committed after the main run.** The text above was written and delivered to the author in chat on 2026-11-24, before any A17 code existed, but it was added to this file only after the main run. The reason: append_a17_plan.py was run before A17_amendment.md had been copied into the repository root (console message "run from the repository root with docs/analysis_plan_P0.md and A17_amendment.md present"), so the commit fc33970 labelled "A17 plan (frozen before code)" contains no plan text. The rules that decide the label (thresholds, labels, informative cells, seeds, grid, instance counts) were in committed code and configuration before the run: src/dwell/analysis_a17.py (docstring and constants), configs/a17_main.json, scripts/run_a17.py and the gate, all in commit 2913a1f, before the main run. What the late commit adds is the disclosure and the stated expectations, which no code enforces. No decision depended on the A17 results. The freeze by plan commit did not happen for A17; this is a deviation from the workflow.

#### A17 outcome (2026-11-25): ROBUST
Main run after commit 2913a1f (gate hash 12537126...0065), role main, eval seed 20261281, bootstrap seed 20261282. Console output supplied by the author [DOC]; raw files not re-analysed by me. 160 cells, 5280 rows, 0 violations, wall 4413.7 s on 4 workers (hedged estimate 60 to 120 minutes).
P1 (CEPS minus CEP, points; one-sided 95% lower bound; informative cells of 16; instances), all holding: K3 n = 7: +5.32 (4.65), 11 cells, 330; K3 n = 9: +6.78 (5.92), 8, 160; wide n = 7: +4.96 (4.11), 10, 400; wide n = 9: +6.28 (5.40), 10, 300; equal n = 7: +5.51 (4.57), 4, 160; equal n = 9: +4.24 (3.36), 6, 180; spread n = 7: +6.46 (5.57), 10, 400; spread n = 9: +6.00 (5.14), 9, 270. Same-seed base control: n = 7: +6.65 (5.70), 8 cells, 320; n = 9: +6.96 (5.97), 9, 270. Label (P1 only): ROBUST (8 of 8).
Cheap-planner class (RU_CEP minus CEPS, points; upper bound; class): K3 n = 7: +2.22 (2.84) ADEQUATE; K3 n = 9: +2.36 (3.31) INCONCLUSIVE; wide n = 7: +3.00 (3.72) INCONCLUSIVE; wide n = 9: +2.96 (3.91) INCONCLUSIVE; equal n = 7: +2.21 (3.13) INCONCLUSIVE; equal n = 9: +3.78 (5.00) INCONCLUSIVE; spread n = 7: +0.45 (0.76) ADEQUATE; spread n = 9: +0.84 (1.28) ADEQUATE. Base control: n = 7: +2.09 (2.83) ADEQUATE; n = 9: +2.92 (3.83) INCONCLUSIVE. No statement is ROLLOUT-WORTH. P3 pooled over the variant families: in-band minus out-of-band gain +5.19 points (lower bound 4.54), holds.
Mean survival probability (RU_CEP, CEPS, CEP): K3 n = 7: 0.63, 0.61, 0.55; K3 n = 9: 0.69, 0.66, 0.59; wide 7: 0.63, 0.60, 0.56; wide 9: 0.65, 0.62, 0.56; equal 7: 0.75, 0.73, 0.67; equal 9: 0.56, 0.52, 0.48; spread 7: 0.74, 0.73, 0.67; spread 9: 0.75, 0.74, 0.68; base 7: 0.67, 0.65, 0.59; base 9: 0.62, 0.59, 0.52.
Meaning [OPEN, my reading]: (1) The planner gain of the survival-aware planner CEPS over its expected-value counterpart CEP (4.2 to 6.8 points) holds in all four families at n = 7 and 9 on a fresh seed, so the mismatch finding is not specific to the base family. (2) The regime variable separates in the pooled variants (+5.2 points). (3) The rollout RU_CEP adds 0.5 to 3.8 points over CEPS; the cheap planner is ADEQUATE in 3 of the 8 variant statements and INCONCLUSIVE in 5, never ROLLOUT-WORTH; it is closest to RU_CEP with heterogeneous values (spread, under 1 point). In the base control at n = 9 the gap (+2.92, upper bound 3.83) is larger than in A16 (+1.69, upper bound 2.11), so a statement of adequacy near the 3-point limit is seed-sensitive. (4) The earlier A16 variant regret gaps at n = 5 (5.0 to 8.7 points of regret against the exact optimum) are on a different scale and are not directly comparable with these absolute differences.
Limits as in A17.8, and: the equal family has few informative cells (4 and 6 of 16); the class boundaries are my choice; planners only.
"""

RECORD = """

## A17 (2026-11-24 to 2026-11-25)
- [CODE] Code, 7 tests, 12 of 12 mutations, audit and gate in commit 2913a1f (gate hash 12537126...0065, identical to my sandbox). Deviation: the A17 plan text was not committed before the run (append_a17_plan.py was run before A17_amendment.md was in the repository root); the deciding rules were in committed code and configs before the run (see the plan file).
- [DOC] Main run: 160 cells, 5280 rows, 0 violations, 4413.7 s. Outcome ROBUST: CEPS minus CEP +4.24 to +6.78 points in all 8 variant statements (lower bounds 3.36 to 5.92); P3 pooled +5.19 (lower 4.54). Cheap-planner class: ADEQUATE in 3 of 8 variant statements, INCONCLUSIVE in 5, never ROLLOUT-WORTH (RU_CEP minus CEPS +0.45 to +3.78).
- [OPEN] Reading: the mismatch and the planner gain are robust across the families; adequacy of the cheap planner is borderline and seed-sensitive near 3 points. Next: freeze the evidence and write.
"""

CHANGELOG = """

## 2026-11-25, A17 (planner robustness across instance families)
- Added src/dwell/analysis_a17.py, scripts/run_a17.py, analyze_a17.py, a17_gate.py, mutate_a17.py, configs a17_*, tests/test_a17.py (reuses src/dwell/a16.py). Code and gate 2913a1f; plan text committed after the run (deviation noted in the plan).
- Outcome ROBUST (main run, 5280 rows). docs/record.md: A17 entry.
"""

TARGETS = [("docs/analysis_plan_P0.md", PLAN, "### A17 (2026-11-24)"),
           ("docs/record.md", RECORD, "## A17 (2026-11-24 to 2026-11-25)"),
           ("docs/CHANGELOG.md", CHANGELOG, "## 2026-11-25, A17 (planner robustness across instance families)")]


def main():
    plan = []
    for rel, text, marker in TARGETS:
        if not os.path.exists(rel):
            sys.exit(f"ERROR: {rel} not found (run from the repository root)")
        raw = open(rel, "rb").read()
        if marker.encode() in raw:
            sys.exit(f"ERROR: {rel} already contains '{marker}'; nothing written")
        if not raw.endswith(b"\n"):
            sys.exit(f"ERROR: {rel} does not end with a newline; nothing written")
        new = text.replace("\r\n", "\n")
        if b"\r\n" in raw:
            new = new.replace("\n", "\r\n")
        plan.append((rel, raw, new.encode("utf-8")))
    for rel, raw, add in plan:
        open(rel, "wb").write(raw + add)
        print("appended", rel, f"(+{len(add)} bytes)")


if __name__ == "__main__":
    main()
