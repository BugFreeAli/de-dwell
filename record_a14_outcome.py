#!/usr/bin/env python3
"""
Record the A14 outcome (MISMATCH) and the A14.r1 plan in the repo records (append-only, fails loudly).
Run from the repository root:   python record_a14_outcome.py
  docs/analysis_plan_P0.md   A14 outcome and A14.r1 (plan for two descriptive runs, written before those runs)
  docs/record.md             A14 entry with provenance tags
  docs/CHANGELOG.md          A14 entry
Stops without writing anything if a marker is already present, a file is missing, or a file does not end with a newline.
Keeps the line-ending style (CRLF or LF) of each file.
"""
import os, sys

PLAN = """

#### A14 outcome (2026-11-21): MISMATCH
Main run seed 20261251 after commit 5f3f913 (gate hash 6aa7a603...7719): 24 cells, 4800 rows, 0 invariant violations, wall 1064.1 s on 4 workers (my estimate was 10 to 25 minutes; console output supplied by the author [DOC]; raw files not re-analysed by me). The audit on seed 1 (480 rows, 0 violations) gave the same picture. Decision table A14.9 applied by scripts/analyze_a14.py: regions exist for the EV set and not for ALL, outcome MISMATCH. Regions (n, rho, s, a, lam): (4, 1.1, 2, 0.5, 0.25) with (4, 1.5, 2, 0.5, 0.25), and (4, 1.1, 2, 0.5, 0.25) with (4, 1.1, 4, 0.5, 0.25).
14 of 24 cells are informative (mean U_opt between 0.10 and 0.99); mean U_opt over them 0.707. Mean relative regret over informative cells (points): FAIR 99.43, LLF 94.26, SE 56.60, EDF 49.16, GMR 31.81, BI 31.14, ROLL_EDF 29.80, LA1 26.89, RU_EDF 21.42, CEP 13.92, ROLL_GMR 12.73, CEPS 9.79, RU_GMR 9.21, ROLL_CEP 8.49, DPEV 6.27, RU_CEP 4.43, RU_DPEV 2.00. The best method is objective-aware in 100% of informative cells; RU_DPEV is the best method in every cell of the top-8 table printed by the analysis. The first action of the goal optimum differs from that of DPEV in 49.3% of instances. Spearman correlation of the demand-bound load with the instance loss of DPEV: 0.086.
Best EV-based method, largest regrets (rho, s, a, lam: regret, lower bound; method): 1.5, 2, 0.5, 0.25: 32.1%, 26.0%, DPEV; 1.1, 4, 0.5, 0.25: 16.0%, 12.2%, DPEV; 1.1, 2, 0.5, 0.25: 8.0%, 6.4%, DPEV; 1.1, 4, 0.9, 0.25: 7.2%, 5.3%, DPEV; 1.5, 4, 0.5, 0.5: 5.2%, 3.4%, ROLL_GMR. Best of all 17 methods: RU_DPEV; largest regrets 13.5% (rho 1.5, s 2, a 0.5, lam 0.25; U_opt 0.162; absolute gap 2.2 points; lower bound of the relative regret 9.6%), 6.5% (1.1, 4, 0.5, 0.25; 1.25 points), 3.1%, 2.7% and smaller elsewhere. Applying the point-flag rule of A14.5 by hand to the printed values, one cell is point-flagged for ALL (rho 1.5, s 2, a 0.5, lam 0.25); it is not flagged because the lower bound of its absolute gap does not reach the 2-point floor (the bound is not printed), and a single cell is no region. The floor was fixed in A14.5 before the run and is not changed.
Instance level (instances with U_opt > 0): loss of the best method of ALL has mean 3.19%, 95th percentile 0.0%, maximum 100.0%; loss of the best EV-based method has mean 8.31%, 95th percentile 100.0%, maximum 100.0% (the loss of one instance is mostly all or nothing).
Meaning [OPEN, my reading]: the exact optimum of the expected-value objective (DPEV) loses 2% to 32% of the optimal survival probability (more than 7% in four cells) in the informative cells of tight tolerance (lam 0.25) at load 1.1 and above, and the best EV-based method is DPEV itself in four of the five cells listed; so the loss comes from the objective, not from a weak planner. One exact step of policy improvement of DPEV under the goal objective (RU_DPEV) recovers most of it at n = 4, with one exception (13.5% in the cell above, where the audit had shown 4.4% with 20 instances). The effect is concentrated at lam 0.25, low label accuracy and load 1.1 to 1.5; at lam 0.5 only one cell exceeds 5% for the best EV-based method. The claim of A1 to A13 that exact-model planning is near-optimal holds for the expected value and does not carry over to a survival objective. The gain of objective-aware planning needs the accumulated value in the state and, for RU_DPEV, the exact EV optimum as base, which exists only for small n.
Not shown: n above 4 (spot check below), other forms of the goal, an unknown prior, learned policies. Chen et al. (2026) already show in a memoryless model that the optimal policy depends on the objective [DOC, per Section 4.1 of the literature document]; what is new here, if confirmed, is the dwell, slew and deadline structure and the size of the mismatch for the exact expected-value optimum [OPEN].

### A14.r1 (2026-11-21): two descriptive extensions after MISMATCH (written before those runs)
A14.r1.1 Purpose. The main grid has two values of lam and n = 4 only. The two runs below describe how the loss depends on lam and whether it persists at n = 5. They are post-hoc, follow a decided outcome, apply no go rule and change no result of A14.
A14.r1.2 Change to A14.3. The n = 5 spot check uses rho in {1.1, 1.5} (not {0.8, 1.1}), s in {2, 4}, a in {0.5, 0.9}, lam in {0.25, 0.5}: 16 cells, 100 instances, seed 20261253, role spot (configs/a14_n5.json). Reason: the rho 0.8 cells took 3327 s of the 4228 s total cell time of the main run and are mostly uninformative. The cost at n = 5 is not measured; the A13 audit timings were 6 to 16 times higher for n = 5 than for n = 4, which gives a hedged estimate of 15 to 60 minutes on 4 workers [OPEN].
A14.r1.3 Tolerance sweep. n = 4, rho in {1.5, 1.1}, s in {2, 4}, a in {0.5, 0.9}, lam in {0.1, 0.25, 0.4, 0.5, 0.6}: 40 cells, 200 instances, seed 20261253, role spot (configs/a14_lam.json). The cells at lam 0.25 and 0.5 repeat A14 cells on a new seed. Estimated cost from the main-run cell times: about 10 minutes on 4 workers [OPEN]. The sweep and the n = 5 run use the same seed stream; their instances differ because n differs. Many cells at lam 0.1 will be uninformative (U_opt below 0.10 at high load); they are reported as such.
A14.r1.4 Outputs and reading. scripts/analyze_a14.py (descriptive in role spot) and scripts/report_a14_lam.py (table per lam: informative cells, mean U_opt, mean regret of DPEV, of the best EV-based method and of the best of ALL, their maxima, number of flagged cells and of regions). No thresholds are added. A statement that the loss falls with lam is made only if the mean regret of the best EV-based method is non-increasing along the five values; otherwise it is reported as non-monotone. The first outcome is read as replicated if, at lam 0.25 on the new seed, at least one region of EV-flagged cells appears; this is a descriptive statement, not a test.
A14.r1.5 Limits. Same model and objective form as A14; the goal is a threshold on killed value; the prior is known; only methods that use the exact model are compared. No learned policy is included. If the n = 5 run is too slow, it is stopped, the finished cells are reported as such and nothing is dropped.
"""

RECORD = """

## A14 (2026-11-21)
- [CODE] Plan commit before code; code, tests (64), mutations (19 of 19), audit (seed 1, 480 rows, 0 violations) and gate in 5f3f913 (gate hash 6aa7a603...7719). The author's audit output equals my sandbox run to the last digit.
- [DOC] Main run seed 20261251: 24 cells, 4800 rows, 0 violations, 1064.1 s on 4 workers. Outcome MISMATCH (regions for the EV set, none for ALL). Best EV-based method loses up to 32.1% of the survival probability (rho 1.5, s 2, a 0.5, lam 0.25; DPEV); best of all 17 loses at most 13.5% (RU_DPEV; absolute gap 2.2 points, not flagged because of the absolute floor). Details in the plan, A14 outcome.
- [OPEN] Reading: the loss is caused by the objective (the exact expected-value optimum is mismatched), concentrated at lam 0.25 and low label accuracy; objective-aware one-step improvement recovers most of it at n = 4. n = 5 and the dependence on lam are described in A14.r1 (planned, not run).
- [CODE] Previous gate hashes covered 2 files; the A14 gate covers 7 (m0u, analysis_a14, m0, dp, rules, instances, run_a14).
"""

CHANGELOG = """

## 2026-11-21, A14 (asset-survival objective)
- Added src/dwell/m0u.py, analysis_a14.py, scripts/run_a14.py, analyze_a14.py, a14_gate.py, mutate_a14.py, report_a14_lam.py, configs a14_*, tests/test_a14.py, test_a14_analysis.py, brute_u.py. Plan A14 (commit before code), code and gate 5f3f913.
- Outcome MISMATCH (main seed 20261251, 4800 rows). A14.r1 plans a tolerance sweep and an n = 5 spot check (descriptive). docs/record.md: A14 entry.
"""

TARGETS = [("docs/analysis_plan_P0.md", PLAN, "#### A14 outcome (2026-11-21)"),
           ("docs/record.md", RECORD, "## A14 (2026-11-21)"),
           ("docs/CHANGELOG.md", CHANGELOG, "## 2026-11-21, A14 (asset-survival objective)")]


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
