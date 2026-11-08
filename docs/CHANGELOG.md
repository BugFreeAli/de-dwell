# Changelog

## 2026-11-08, step 1 (plan v1, code v0.1)
- Added: M0 model, exact DP, rules FAIR/EDF/LLF/SE/GMR/BI, class-aware oracle, instance generator, grid runner, analysis script, 40 unit tests, configs, analysis plan P0 v1.
- Decisions on the draft M0 of literature doc Section 10.5 (see docs/record.md): label-accuracy construction of information; grid values; exact rule definitions; oracle treated as a bound, not a rule; slew and retention semantics; companion information-attribution test; S1 control.
- Not done: no confirmatory run yet.

## 2026-11-09, amendment A1 (code v0.2)
- Added: rule CEP (certainty-equivalent planner), scripts/run_a1.py (same-instance augmentation), scripts/summarize_a1.py, configs a1_main, a1_s1, a1_v2, a1_v4, a1_ratio, tests/test_cep.py (8 new tests, 48 in total); `extra_rules` option in run_p0.py and analyze_p0.py.
- Plan: Section 12 of docs/analysis_plan_P0.md now holds amendment A1; Sections 1 to 11 are unchanged.
