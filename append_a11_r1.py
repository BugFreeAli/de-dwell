"""Append the A11.r1 sub-amendment to docs/analysis_plan_P0.md, plus short entries in docs/CHANGELOG.md and docs/record.md.

Run from the repository root:  conda run -n de-dwell python append_a11_r1.py A11_r1_amendment.md
Fails loudly (exit code 1, nothing written) if a file does not end with the expected text, if A11 is already
present, or if the amendment file has an unexpected heading. Originals are copied to backups/ first (git-ignored).
"""
import hashlib, pathlib, shutil, sys

PLAN = pathlib.Path("docs/analysis_plan_P0.md")
CHG = pathlib.Path("docs/CHANGELOG.md")
REC = pathlib.Path("docs/record.md")

PLAN_TAIL = "Any change of this text is a dated sub-amendment with a new tag.\r\n"
CHG_TAIL = "- Not done: no M1 code, no configs, no run.\r\n"
REC_TAIL = "- [OPEN] Novelty against time-dependent scheduling not checked; cost of the M1 DP not measured.\r\n"

CHG_ENTRY = (
    "\r\n## 2026-11-17, amendment A11.r1 (code, tests and audit built; no main run)\r\n"
    "- Plan: Section 12 of docs/analysis_plan_P0.md now also holds A11.r1 (corrections to A11.2 and A11.7, open definitions, procedure, cost probe). Frozen by tag `p0-a11-plan-r1`; earlier text unchanged.\r\n"
    "- Code: src/dwell/m1.py, rules1.py, analysis_a11.py; tests/test_a11.py, brute_m1.py, test_a11_analysis.py; scripts/run_a11.py, analyze_a11.py, mutate_a11.py, a11_gate.py; configs/a11_*.json.\r\n"
)
REC_ENTRY = (
    "\r\n## A11 code and audit (2026-11-17)\r\n"
    "- [CODE] 22 tests pass; 11 of 11 mutations caught; control audit (gamma = 1, 24 cells, 480 rows, exploratory seed 1) without violations: DP, 16 non-rollout methods and idle-free rollouts equal M0; value of idle 0 at every epsilon. Main-grid audit: see results/a11_audit and results/a11_gate.json.\r\n"
    "- [CODE] With idle enabled a rollout differs from the M0 rollout at gamma = 1 in 3 of 120 test cases (A11.r1, item 1b).\r\n"
    "- [OPEN] Cost on the author's laptops not measured; main seed 20261222 unused.\r\n"
)

def fail(msg):
    print("FAIL:", msg); sys.exit(1)

def read(p):
    b = p.read_bytes()
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        fail(f"{p} is not utf-8")

if len(sys.argv) != 2:
    fail("usage: append_a11_r1.py A11_r1_amendment.md")
amend = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
if "\r" in amend:
    fail("amendment file must use LF line endings")
if not amend.lstrip("\n").startswith("### A11.r1 (2026-11-17)"):
    fail("amendment heading not as expected")
amend_crlf = amend.replace("\n", "\r\n")
if not amend_crlf.startswith("\r\n"):
    amend_crlf = "\r\n" + amend_crlf

plan, chg, rec = read(PLAN), read(CHG), read(REC)
if "### A11.r1 (" in plan:
    fail("plan already contains A11.r1")
if "### A11 (" not in plan:
    fail("plan does not contain A11 yet; apply append_a11.py first")
for name, txt, tail in ((PLAN, plan, PLAN_TAIL), (CHG, chg, CHG_TAIL), (REC, rec, REC_TAIL)):
    if not txt.endswith(tail):
        fail(f"{name} does not end with the expected text")
    if "\r\n" not in txt or txt.replace("\r\n", "").count("\n") != 0:
        fail(f"{name} has non-CRLF line endings")

bk = pathlib.Path("backups"); bk.mkdir(exist_ok=True)
for p in (PLAN, CHG, REC):
    shutil.copy2(p, bk / (p.stem + ".pre-A11" + p.suffix))

PLAN.write_bytes((plan + amend_crlf).encode("utf-8"))
CHG.write_bytes((chg + CHG_ENTRY).encode("utf-8"))
REC.write_bytes((rec + REC_ENTRY).encode("utf-8"))

new = PLAN.read_bytes()
assert new.startswith(plan.encode("utf-8")), "original plan text changed"
assert new.count(b"\n") == new.count(b"\r\n"), "mixed line endings"
print("OK. plan sha256:", hashlib.sha256(new).hexdigest())
print("backups in", bk.resolve())
