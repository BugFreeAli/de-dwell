#!/usr/bin/env python3
"""
Append the A18 plan (A18_amendment.md in the repository root) to docs/analysis_plan_P0.md. Run from the repository root:
    python append_a18_plan.py
Append-only; stops without writing if the plan already contains the A18 marker, if a file is missing, or if the plan does not
end with a newline. Keeps the line-ending style of the plan (CRLF or LF).
"""
import os, sys

PLAN, SRC, MARK = "docs/analysis_plan_P0.md", "A18_amendment.md", "### A18 (2026-11-26)"
if not (os.path.exists(PLAN) and os.path.exists(SRC)):
    sys.exit("ERROR: run from the repository root with docs/analysis_plan_P0.md and A18_amendment.md present")
raw = open(PLAN, "rb").read()
if MARK.encode() in raw:
    sys.exit(f"ERROR: {PLAN} already contains '{MARK}'; nothing written")
if not raw.endswith(b"\n"):
    sys.exit(f"ERROR: {PLAN} does not end with a newline; nothing written")
add = open(SRC, encoding="utf-8").read().replace("\r\n", "\n").strip()
if not add.startswith(MARK):
    sys.exit("ERROR: A18_amendment.md does not start with the A18 heading")
add = "\n\n" + add + "\n"
if b"\r\n" in raw:
    add = add.replace("\n", "\r\n")
open(PLAN, "wb").write(raw + add.encode("utf-8"))
print(f"appended A18 plan to {PLAN} (+{len(add)} bytes)")
