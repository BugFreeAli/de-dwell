#!/usr/bin/env python3
"""
Correct an unverified count in the A13 outcome text of docs/analysis_plan_P0.md and log the correction in docs/record.md.
The first version of append_a13_outcome.py wrote "this is the seventh structure test without a gap larger than 5% for a strong
planner"; the count was not checked. The sentence is replaced by the statement already made in the record.
Run from the repository root:   python fix_a13_sentence.py
Stops without writing if the old sentence is not found exactly once, if the correction is already logged, or if a file is missing.
"""
import os, sys

PLAN, REC = "docs/analysis_plan_P0.md", "docs/record.md"
OLD = "Together with A1 to A12 this is the seventh structure test without a gap larger than 5% for a strong planner."
NEW = ("Together with A1 to A12 this extends the pattern that a certainty-equivalent plan with one exact rollout step stays within "
       "a few percent of the DP optimum in every M0-type structure tested at n up to 5.")
LOG = ("\n- [CODE] Correction 2026-11-21: the A13 outcome text first said \"the seventh structure test without a gap larger than 5%\"; "
       "that count was not verified and was replaced (docs/analysis_plan_P0.md, same sentence) by the pattern statement above.\n")
MARK = "Correction 2026-11-21"
for f in (PLAN, REC):
    if not os.path.exists(f):
        sys.exit(f"ERROR: {f} not found (run from the repository root)")
plan, rec = open(PLAN, "rb").read(), open(REC, "rb").read()
if plan.count(OLD.encode()) != 1:
    sys.exit(f"ERROR: old sentence found {plan.count(OLD.encode())} times in {PLAN} (must be exactly 1); nothing written")
if MARK.encode() in rec:
    sys.exit(f"ERROR: {REC} already logs the correction; nothing written")
if not rec.endswith(b"\n"):
    sys.exit(f"ERROR: {REC} does not end with a newline; nothing written")
log = LOG.replace("\n", "\r\n") if b"\r\n" in rec else LOG
open(PLAN, "wb").write(plan.replace(OLD.encode(), NEW.encode()))
open(REC, "wb").write(rec + log.encode())
print("sentence corrected in", PLAN, "and logged in", REC)
