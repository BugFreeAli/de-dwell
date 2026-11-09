# Changelog

## 2026-11-08, step 1 (plan v1, code v0.1)
- Added: M0 model, exact DP, rules FAIR/EDF/LLF/SE/GMR/BI, class-aware oracle, instance generator, grid runner, analysis script, 40 unit tests, configs, analysis plan P0 v1.
- Decisions on the draft M0 of literature doc Section 10.5 (see docs/record.md): label-accuracy construction of information; grid values; exact rule definitions; oracle treated as a bound, not a rule; slew and retention semantics; companion information-attribution test; S1 control.
- Not done: no confirmatory run yet.

## 2026-11-09, amendment A1 (code v0.2)
- Added: rule CEP (certainty-equivalent planner), scripts/run_a1.py (same-instance augmentation), scripts/summarize_a1.py, configs a1_main, a1_s1, a1_v2, a1_v4, a1_ratio, tests/test_cep.py (8 new tests, 48 in total); `extra_rules` option in run_p0.py and analyze_p0.py.
- Plan: Section 12 of docs/analysis_plan_P0.md now holds amendment A1; Sections 1 to 11 are unchanged.

## 2026-11-09, amendment A2 (code v0.3)
- Added: src/dwell/m2.py (K classes, retention factor kappa), make_instance_k, rule LA1, decay-aware CEP, scripts/run_a2.py, scripts/summarize_a2.py, configs a2_wide, a2_k3, a2_decay05, a2_decay0, a2_main_la1, tests/brute2.py and tests/test_m2.py (103 tests in total).
- Plan: Section 12 of docs/analysis_plan_P0.md now also holds amendment A2.

## 2026-11-09, amendment A3 (code v0.4)
- Added: rollout rules ROLL_CEP, ROLL_GMR, ROLL_EDF, scripts/run_a3.py, scripts/summarize_a3.py, configs a3_main, a3_wide, a3_k3, tests/test_rollout.py (120 tests in total).
- Plan: Section 12 of docs/analysis_plan_P0.md now also holds amendment A3.

## 2026-11-09, amendment A4 (code v0.5)
- Added: scripts/run_a4.py, scripts/analyze_a4.py, configs a4_k3 and a4_wide, tests/test_misspec.py (123 tests in total).
- Plan: Section 12 of docs/analysis_plan_P0.md now also holds amendment A4.
