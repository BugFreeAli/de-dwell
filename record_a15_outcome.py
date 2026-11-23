#!/usr/bin/env python3
"""
Record the A15 outcome (LEARN-BEHIND) in the repo records (append-only, fails loudly). Run from the repository root:
    python record_a15_outcome.py
  docs/analysis_plan_P0.md, docs/record.md, docs/CHANGELOG.md
Stops without writing anything if a marker is already present, a file is missing, or a file does not end with a newline.
Keeps the line-ending style (CRLF or LF) of each file.
"""
import os, sys

PLAN = """

#### A15 outcome (2026-11-23): LEARN-BEHIND
Main run after commit df36152 (gate hash cc3f06b0...7287), role main, pool seed 20261262, eval seed 20261261. Console output supplied by the author [DOC]; raw files not re-analysed by me. Stages on 4 workers: pool 170.3 s (pool G 61,335 samples with goal labels for 64.4%; pool D 44,475 samples), training of 50 models 474.7 s, evaluation 48 cells and 2560 rows with 0 violations 3468.3 s, then the time stage; about 70 minutes in all (hedged estimate 2 to 4 hours).
Informative cells: n = 5: 11 of 16 (880 instances); n = 7: 8 of 16 (400); n = 9: 8 of 16 (240).
Mean survival probability over the informative cells (regret against the exact optimum, n = 5 only). n = 5: exact optimum 0.6407; RU_DPEV 0.6344 (0.98 points), RU_CEP 0.6198 (3.26), DPEV 0.6114 (4.57), CEPS 0.6034 (5.82), CEP 0.5714 (10.82); learned: IL_A 0.6055 (5.49), IL_AD 0.5887 (8.12), IL_EV 0.5598 (12.63), IL_G 0.5539 (13.54), IL_G_noacc 0.5114 (20.19). n = 7: RU_CEP 0.6639, CEPS 0.6462, IL_AD 0.6373, IL_A 0.6151, IL_EV 0.5860, CEP 0.5818, IL_G 0.5085, IL_G_noacc 0.5063. n = 9: RU_CEP 0.6583, CEPS 0.6346, IL_AD 0.6187, CEP 0.5613, IL_EV 0.5501, IL_A 0.5172, IL_G 0.3783, IL_G_noacc 0.3675. Standard deviation over seeds of IL_AD: 0.0149, 0.0109, 0.0141 at n = 5, 7, 9.
Primary contrasts (seed level, exact one-sided Wilcoxon, Holm over 8; mean difference, minimum and maximum over seeds, p, Holm reject, material): IL_A minus CEPS n = 5: +0.0021 (-0.0212, +0.0163), p 0.348, no, no. IL_AD minus CEPS n = 7: -0.0089 (-0.0271, +0.0074), p 0.986, no, no; n = 9: -0.0159 (-0.0392, +0.0091), p 0.995, no, no. IL_AD minus RU_CEP n = 7: -0.0267 (-0.0448, -0.0104), p 1.000, no, no; n = 9: -0.0396 (-0.0629, -0.0146), p 1.000, no, no. IL_A minus IL_G n = 5: +0.0516 (+0.0301, +0.0740), p 0.0010, yes, yes. IL_G minus IL_EV n = 5: -0.0058 (-0.0238, +0.0192), p 0.903, no, no. IL_G minus IL_G_noacc n = 5: +0.0426 (+0.0253, +0.0630), p 0.0010, yes, yes.
Decision table A15.8: IL_AD is more than one point below RU_CEP at n = 7 and 9 (all ten seeds negative at both sizes), so the outcome is LEARN-BEHIND. Secondary flags: objective_matters false; accumulated_value_needed true; planner_feature_helps true; improves_cheap_planner false.
Decision time (cold, mean ms): n = 7: IL_AD 0.801, CEP 0.454, CEPS 0.473, RU_CEP 25.54; n = 9: IL_AD 10.46, CEP 9.05, CEPS 10.33, RU_CEP 439.2; ratio RU_CEP over IL_AD at n = 9: 42.0. IL_AD costs about as much as CEPS because it calls CEPS.
Descriptive, not primary (computed by me from the printed pooled means; no paired intervals yet [OPEN]): CEPS minus CEP: +3.2 points (n = 5), +6.4 (n = 7), +7.3 (n = 9); RU_CEP minus CEPS: +1.6, +1.8, +2.4. IL_EV minus IL_G: +7.8 points at n = 7 and +17.2 at n = 9. IL_A is within 0.2 points of CEPS at n = 5 (0.6055 against 0.6034).
Meaning [OPEN, my reading]: (1) The learned scorers, trained by imitation, neither match nor beat the objective-aware planners at this training budget; the best learned variant (IL_AD) is 0.9 and 1.6 points below CEPS and 2.7 and 4.0 points below RU_CEP at n = 7 and 9. As predicted in A15.1, LEARN-BEATS did not occur. (2) The accumulated value is needed by the learner (IL_G_noacc is 4.3 points below IL_G at n = 5, Holm win) and the planner recommendation helps it (IL_A is 5.2 points above IL_G), but IL_A is only as good as CEPS. The effect of the teacher objective on the learner is not shown (IL_G minus IL_EV is -0.6 points, not significant), and the plain goal-trained scorer generalises worse across size than the expected-value one. (3) At scale the cheap objective-aware planner CEPS keeps most of the gain over its expected-value counterpart CEP (+6.4 and +7.3 points at n = 7 and 9) at the same decision cost, and RU_CEP adds 1.8 and 2.4 points at about 40 times the cost; the quality-time frontier in these results is CEPS to RU_CEP, not a learned scorer.
Limits as in A15.9: one imitation architecture, no reinforcement learning or search, training sizes up to 7, pooled over informative cells, instance variability not in the p-values, thresholds mine.
"""

RECORD = """

## A15 (2026-11-22 to 2026-11-23)
- [CODE] Plan commit 5170031 before any code; code, 26 tests, 19 of 19 mutations, audit pipeline and gate in df36152 (gate hash cc3f06b0...7287). The author's audit and gate run matched my sandbox (same hash).
- [DOC] Main run: pool 170 s, training of 50 models 475 s, evaluation 2560 rows with 0 violations 3468 s; time stage separate. Outcome LEARN-BEHIND: IL_AD minus RU_CEP -2.7 and -4.0 points at n = 7 and 9 (all seeds negative); IL_AD minus CEPS -0.9 and -1.6; IL_A minus CEPS +0.2 (n = 5, not significant). Accumulated value needed (IL_G_noacc -4.3 points, Holm win); planner feature helps (IL_A over IL_G +5.2 points); teacher objective effect not shown.
- [DOC] Decision time at n = 9: RU_CEP 439 ms, CEPS 10.3 ms, IL_AD 10.5 ms (42 times faster than RU_CEP, but equal to CEPS).
- [OPEN] Reading: learning does not add over the objective-aware planners here; CEPS keeps +6.4 and +7.3 points over its expected-value counterpart CEP at n = 7 and 9 at equal cost (descriptive, no paired intervals yet). Next candidate: a planner-side confirmatory plan (A16).
"""

CHANGELOG = """

## 2026-11-23, A15 (learned scorers against objective-aware planners)
- Added src/dwell/learned_g.py, analysis_a15.py, scripts/run_a15.py (stages pool, train, eval, time), analyze_a15.py, a15_gate.py, mutate_a15.py, configs a15_*, tests/test_a15.py, test_a15_analysis.py. Plan 5170031, code and gate df36152.
- Outcome LEARN-BEHIND (main run, 2560 rows). docs/record.md: A15 entry.
"""

TARGETS = [("docs/analysis_plan_P0.md", PLAN, "#### A15 outcome (2026-11-23)"),
           ("docs/record.md", RECORD, "## A15 (2026-11-22 to 2026-11-23)"),
           ("docs/CHANGELOG.md", CHANGELOG, "## 2026-11-23, A15 (learned scorers against objective-aware planners)")]


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
