#!/usr/bin/env python3
"""
Record the A16 outcome (CONFIRMED) in the repo records (append-only, fails loudly). Run from the repository root:
    python record_a16_outcome.py
  docs/analysis_plan_P0.md, docs/record.md, docs/CHANGELOG.md
Stops without writing anything if a marker is already present, a file is missing, or a file does not end with a newline.
Keeps the line-ending style (CRLF or LF) of each file.
"""
import os, sys

PLAN = """

#### A16 outcome (2026-11-24): CONFIRMED
Main run after commit ee8d676 (gate hash 0ba38840...0349), role main, eval seed 20261271, bootstrap seed 20261272; plan commit 9f2f4d1. Console output supplied by the author [DOC]; raw files not re-analysed by me. 88 cells, 4480 rows, 0 violations, wall 3596.5 s on 4 workers (hedged estimate 20 to 60 minutes).
Primary claims (mean, one-sided 95% bounds, holds): P1 n = 5: +3.22 points (lower 2.62), holds; P1 n = 7: +4.90 (lower 4.45), holds; P1 n = 9: +6.68 (lower 5.89), holds. P2 n = 7: RU_CEP minus CEPS +1.76 (upper 2.13), holds; P2 n = 9: +1.69 (upper 2.11), holds. P3: in-band minus out-of-band mean gain +4.70 points (lower 3.99), holds. Outcome CONFIRMED by the pre-written rule; the stated expectation was that PARTIAL was plausible (A16.1).
Informative cells and instances: n = 5: 12 of 16 (720); n = 7: 10 of 16 (800); n = 9: 9 of 16 (450); n = 11: 8 of 16 (240).
Mean survival probability (regret against the exact optimum at n = 5): n = 5: optimum 0.6674, RU_DPEV 0.6614 (0.90 points), DPEV 0.6456 (3.27), RU_CEP 0.6444 (3.45), CEPS 0.6272 (6.03), CEP 0.5950 (10.86). n = 7: RU_CEP 0.6232, CEPS 0.6056, CEP 0.5566. n = 9: RU_CEP 0.5705, CEPS 0.5536, CEP 0.4868. n = 11 (descriptive, CEP and CEPS only): CEPS 0.6734, CEP 0.6037; CEPS minus CEP +6.97 points (lower 6.03, upper 8.03). RU_CEP minus CEPS at n = 5: +1.72 (1.28 to 2.23).
Gain of CEPS over CEP by tau (informative instances of n = 5, 7, 9; instances, gain in points): (0.6, 0.8]: 197, 1.24; (0.8, 0.9]: 236, 4.52; (0.9, 1.0]: 402, 7.17; (1.0, 1.1]: 403, 8.18; (1.1, 1.25]: 412, 3.74; (1.25, 1.5]: 268, 0.79; above 1.5: 52, 0.0. Spearman correlation of |tau - 1| with the gain: -0.401. At n = 5 the Spearman correlation of |tau_ev - 1| with the DPEV loss is +0.111, so the regime variable does not explain the loss of the exact expected-value optimum (descriptive).
Robustness variants (n = 5, label accuracy 0.5; descriptive; regret against the exact optimum in points: DPEV, RU_DPEV, RU_CEP, CEPS, CEP): K3 (three classes): 7 of 8 cells informative, CEPS minus CEP +4.58 (lower 3.70); 6.14, 0.87, 3.33, 8.46, 15.35. wide (D = (3, 12)): 7 of 8, +2.22 (lower 1.61); 4.80, 1.07, 6.95, 11.91, 15.28. equal (all values 2): 5 of 8, +3.95 (lower 3.03); 2.85, 0.91, 2.40, 11.12, 18.34. By the rule of A16.4 all three variants show the mismatch (lower bound at least 0.01).
Meaning [OPEN, my reading]: (1) On fresh seeds the survival-aware certainty-equivalent planner CEPS beats its expected-value counterpart CEP by 3 to 7 points of survival probability from n = 5 to 11, and RU_CEP adds about 1.7 points at n = 5, 7 and 9, so the A15 pattern is confirmed with intervals. (2) The gain is largest when the goal is tight (tau between 0.8 and 1.1). (3) At n = 5, RU_CEP (3.45 points of regret) is as good as the exact expected-value optimum DPEV (3.27) on the survival objective. (4) The claim that CEPS is adequate (P2) holds for the base instance family only: in the three variants the regret difference between CEPS and RU_CEP is 5.1 (K3), 5.0 (wide) and 8.7 (equal) points (my arithmetic on the printed regrets), so outside the base family the rollout is worth its cost; this was not a planned claim. (5) tau explains the gain of CEPS over CEP but not the loss of DPEV at n = 5.
Limits as in A16.8, and: the band of tau was fixed after exploratory looks; the variants are n = 5 only and descriptive; informative cells are selected with RU_CEP.
"""

RECORD = """

## A16 (2026-11-23 to 2026-11-24)
- [CODE] Plan commit 9f2f4d1 before any code; code, 18 tests, 17 of 17 mutations, audit and gate in ee8d676 (gate hash 0ba38840...0349, identical to my sandbox).
- [DOC] Main run: 88 cells, 4480 rows, 0 violations, 3596.5 s. Outcome CONFIRMED: P1 +3.22, +4.90, +6.68 points at n = 5, 7, 9 (lower bounds 2.62, 4.45, 5.89); P2 RU_CEP minus CEPS +1.76 and +1.69 (upper bounds 2.13, 2.11); P3 regime difference +4.70 points (lower 3.99). n = 11 CEPS minus CEP +6.97 (descriptive). All three robustness variants show the mismatch (CEPS minus CEP +4.58, +2.22, +3.95).
- [OPEN] Reading: P2 holds for the base family only (CEPS to RU_CEP regret gap 5.1, 5.0 and 8.7 points in the variants); tau explains the gain of CEPS over CEP (Spearman -0.401) but not the DPEV loss at n = 5 (+0.111). Next candidate: planner robustness across instance families at n = 7 and 9 (A17).
"""

CHANGELOG = """

## 2026-11-24, A16 (planner-side confirmatory evaluation)
- Added src/dwell/a16.py, analysis_a16.py, scripts/run_a16.py, analyze_a16.py, a16_gate.py, mutate_a16.py, configs a16_*, tests/test_a16.py, test_a16_analysis.py. Plan 9f2f4d1, code and gate ee8d676.
- Outcome CONFIRMED (main run, 4480 rows). docs/record.md: A16 entry.
"""

TARGETS = [("docs/analysis_plan_P0.md", PLAN, "#### A16 outcome (2026-11-24)"),
           ("docs/record.md", RECORD, "## A16 (2026-11-23 to 2026-11-24)"),
           ("docs/CHANGELOG.md", CHANGELOG, "## 2026-11-24, A16 (planner-side confirmatory evaluation)")]


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
