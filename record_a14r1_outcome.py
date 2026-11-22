#!/usr/bin/env python3
"""
Record the outcomes of the two A14.r1 descriptive runs (tolerance sweep, n = 5 spot check) in the repo records (append-only).
Run from the repository root:   python record_a14r1_outcome.py
  docs/analysis_plan_P0.md, docs/record.md, docs/CHANGELOG.md
Stops without writing anything if a marker is already present, a file is missing, or a file does not end with a newline.
Keeps the line-ending style (CRLF or LF) of each file.
"""
import os, sys

PLAN = """

#### A14.r1 outcome (2026-11-21): descriptive runs, no go rule
Both runs were made after the commit 8ad2ec2 that holds A14.r1, with the gate of A14 open (gate hash 6aa7a603...7719). Console output supplied by the author [DOC]; raw files not re-analysed by me. Role spot, seed 20261253 for both runs.
Tolerance sweep (configs/a14_lam.json): n = 4, rho in {1.1, 1.5}, s in {2, 4}, a in {0.5, 0.9}, lam in {0.1, 0.25, 0.4, 0.5, 0.6}; 40 cells, 8000 rows, 0 violations, wall 534.9 s on 4 workers (estimate 10 minutes). 25 of 40 cells informative; mean U_opt 0.722. Mean regret over informative cells (points): DPEV 5.00, RU_DPEV 1.65, RU_CEP 3.42, CEPS 7.35, ROLL_CEP 6.84, CEP 10.09. First action of the goal optimum differs from DPEV in 53.6% of instances; Spearman correlation of the demand-bound load with the DPEV loss 0.007. Instance level: loss of the best method of ALL mean 4.18%, 95th percentile 6.3%; of the best EV-based method mean 9.87%, 95th percentile 100%.
Per lam (informative cells; mean regret of DPEV, of the best EV-based method, maximum of the best EV-based method; mean and maximum of the best of ALL; flagged cells for EV and ALL; regions for EV and ALL):
- lam 0.10: 0 of 8 cells informative.
- lam 0.25: 5 of 8; mean U_opt 0.36; 15.07, 15.07, 37.99; 6.46 and 18.78; 3 EV-flagged cells, 1 ALL-flagged cell; 2 EV regions, 0 ALL regions.
- lam 0.40: 8 of 8; 2.70, 2.70, 4.83; 0.70 and 2.54; none flagged.
- lam 0.50: 8 of 8; 2.01, 1.86, 3.99; 0.24 and 0.87; none flagged.
- lam 0.60: 4 of 8; 2.95, 2.67, 4.52; 0.20 and 0.47; none flagged.
Reading by the rule of A14.r1.4: (1) The first outcome is read as replicated, because two regions of EV-flagged cells appear at lam 0.25 on the new seed (descriptive, not a test). (2) The mean regret of the best EV-based method is not non-increasing along the five values (15.07, 2.70, 1.86, 2.67; lam 0.10 has no informative cell), so no statement that the loss falls with lam is made; the number of informative cells differs between the values (5, 8, 8, 4), so the means are not over the same cells. (3) Largest cell, rho 1.5, s 2, a 0.5, lam 0.25 (U_opt 0.143): best EV-based method (DPEV) loses 38.0% (lower bound 31.4%), the best of ALL (RU_DPEV) 18.8% (lower bound 14.2%, absolute gap 2.7 points). One cell is flagged for ALL at lam 0.25; from the printed values it is this cell [OPEN: inferred, the flag columns of the cells are not printed]. In the main run the same cell gave 13.5% for RU_DPEV. No region for ALL exists, so the outcome MISMATCH is not changed and no GO appears.
n = 5 spot check (configs/a14_n5.json): rho in {1.1, 1.5}, s in {2, 4}, a in {0.5, 0.9}, lam in {0.25, 0.5}; 16 cells, 1600 rows, 0 violations, wall 889.3 s (hedged estimate 15 to 60 minutes; the per-instance cost relative to the same n = 4 cells was 3.6 to 10.3 times, from the printed cell times). 12 of 16 cells informative; mean U_opt 0.663. Mean regret over informative cells (points): DPEV 5.34, RU_DPEV 1.27, RU_CEP 3.70, CEPS 9.00, ROLL_CEP 7.80, CEP 13.13, RU_GMR 11.59, EDF 68.65. First action differs in 52.9% of instances; Spearman correlation 0.093. Instance level: best of ALL mean 4.55%, 95th percentile 30.2%; best EV-based method mean 11.14%, 95th percentile 100%.
Largest cells (rho, s, a, lam: best EV-based regret and lower bound; best of ALL regret): 1.5, 2, 0.5, 0.25: 19.3% (13.0%), DPEV; RU_DPEV 6.8% (absolute gap 1.4 points, lower bound 3.9%). 1.5, 2, 0.9, 0.25: 10.0% (6.1%), DPEV; 1.5%. 1.5, 4, 0.5, 0.5: 5.3% (3.3%), ROLL_GMR; 0.0%. 1.1, 2, 0.5, 0.25: 5.2% (3.7%), DPEV; 1.6%. 1.5, 2, 0.5, 0.5: 4.7%, ROLL_CEP. The flag columns are not printed, so regions for the spot check are not stated.
Meaning [OPEN, my reading]: the mismatch of the expected-value optimum (up to 19% to 38% of the survival probability in the worst cell) is reproduced on a new seed and persists at n = 5. The residual loss of the best objective-aware method in the worst cell is 13.5% and 18.8% at n = 4 and 6.8% at n = 5 (absolute 2.2, 2.7 and 1.4 points); different informative cells enter at n = 5, so the sizes are not directly comparable. The effect is confined to tolerance 0.25 at load 1.1 to 1.5 and low label accuracy; at tolerance 0.4 and above the best EV-based method loses at most 4.8%. Open: whether objective-aware planning that does not need the exact expected-value optimum as a base (RU_CEP, CEPS) or a learned policy keeps this quality at sizes where no exact DP exists.
"""

RECORD = """

## A14.r1 (2026-11-21)
- [DOC] Tolerance sweep (seed 20261253, n = 4, 40 cells, 8000 rows, 0 violations, 534.9 s): at lam 0.25 two EV regions replicate the A14 finding (best EV-based method up to 38.0%, best of all 17 up to 18.8%, cell rho 1.5, s 2, a 0.5); at lam 0.4 and above the best EV-based method loses at most 4.8%; lam 0.1 has no informative cell. No monotone trend is claimed (informative cells differ per lam).
- [DOC] n = 5 spot check (16 cells, 1600 rows, 0 violations, 889.3 s): best EV-based method up to 19.3% (DPEV), best of all 17 up to 6.8% (RU_DPEV); mean regret over informative cells DPEV 5.34, RU_DPEV 1.27, RU_CEP 3.70, CEPS 9.00. Cost per instance 3.6 to 10.3 times the n = 4 cost.
- [OPEN] Reading: the mismatch persists at n = 5 and on a new seed; the open question is a scalable objective-aware method. No go rule applied; the A14 outcome MISMATCH is unchanged.
"""

CHANGELOG = """

## 2026-11-21, A14.r1 (descriptive runs)
- Added scripts/report_a14_lam.py, configs/a14_lam.json (tolerance sweep) and configs/a14_n5.json (n = 5 spot check, rho 1.1 and 1.5). Outcomes recorded in docs/analysis_plan_P0.md (A14.r1 outcome) and docs/record.md.
"""

TARGETS = [("docs/analysis_plan_P0.md", PLAN, "#### A14.r1 outcome (2026-11-21)"),
           ("docs/record.md", RECORD, "## A14.r1 (2026-11-21)"),
           ("docs/CHANGELOG.md", CHANGELOG, "## 2026-11-21, A14.r1 (descriptive runs)")]


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
