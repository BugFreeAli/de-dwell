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

## 2026-11-10, amendments A5 and B0 (code v0.6)
- Added: src/dwell/m3.py (noisy in-dwell signal model), src/dwell/learned.py (numpy scorer), scripts/run_a5.py, scripts/analyze_a5.py, scripts/run_b0.py, configs a5_long, a5_sense, b0_learn, tests/brute3.py, tests/test_m3.py, tests/test_learned.py.
- Plan: Section 12 of docs/analysis_plan_P0.md now also holds amendments A5 and B0.

## 2026-11-11, amendment A6 (code v0.7)
- Added: src/dwell/mismatch.py (observation-based evaluator, exact Bayes-adaptive model with a hidden dwell scale), scripts/run_a6.py, scripts/analyze_a6.py, configs a6_main and a6_det, tests/brute_ba.py, tests/test_mismatch.py.
- Plan: Section 12 of docs/analysis_plan_P0.md now also holds amendment A6.

## 2026-11-12, A6 results recorded
- docs/analysis_plan_P0.md: A6 outcome note appended at the end of Section 12 (no earlier text changed). docs/record.md: A6 results.

## 2026-11-14, amendment A7 (code v0.8)
- Added: src/dwell/adapt.py, scripts/run_a7.py, analyze_a7.py, decide_a7.py, burley_stats.py, burley_transcribe.py, configs a7_main, a7_noise0, a7_noise2, a7_noise3, a7_h1, a7_h2, tests/test_adapt.py, docs/burley_tables_transcribed.csv, docs/burley_stats_output.txt.
- Plan: Section 12 of docs/analysis_plan_P0.md now also holds amendment A7 (appended; no earlier text changed). docs/record.md: A7 inputs and checks.
