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

## 2026-11-14, A7 results
- Added docs/results_a7/ (metrics, decision, run_meta of a7_main and a7_noise2), docs/results_a6_ceiling_check.txt, scripts/analyze_a6_ceiling.py. Plan A7.7 (outcome) appended; earlier text unchanged. record.md: A7 results.

## 2026-11-14, A8
- Added scripts/fit_regime_law.py, tests/test_regime_law.py, docs/results_a8_regime_law_fit.txt. Plan A8 (literature check, regime law, pre-registered prediction for a7_noise3) appended; earlier text unchanged. record.md: A8 entry.

## 2026-11-14, A8 outcome
- Plan A8.4 (outcome and correction) appended; docs/results_a8_noise3_check.txt, docs/results_a7/a7_noise3_metrics.json added; fit_regime_law.py prints per-run coefficients instead of the exponent; test updated. Earlier text unchanged.

## 2026-11-14, A9
- Added configs/a9_m1.json, a9_m2.json, a9_m3.json, a9_x2.json, scripts/analyze_a9.py, tests/test_a9.py. Plan A9 appended (earlier text unchanged). record.md: A9 design.

## 2026-11-14, A9 outcome
- Added docs/results_a9/decision_a9.json; plan A9.5 (outcome NOT-SUPPORTED) appended, earlier text unchanged; record.md: A9 outcome.

## 2026-11-14, A10
- Added configs/a10_*.json, scripts/analyze_a10.py, tests/test_a10.py. Plan A10 appended (earlier text unchanged). record.md: A10 design.

## 2026-11-14, A10 outcome
- Added docs/results_a10/decision_a10.json; plan A10.5 appended (earlier text unchanged); record.md: A10 outcome.


## 2026-11-17 to 2026-11-19, A11 and A12 (added 2026-11-20)
- A11 (range-dependent dwell, idle, energy budget): src/dwell/m1.py, rules1.py, analysis_a11.py, scripts/run_a11.py, analyze_a11.py, a11_gate.py, mutate_a11.py, configs a11_*, tests/test_a11.py, test_a11_analysis.py, brute_m1.py. Plan A11 and A11.r1. Outcome NO-GO.
- A12 (budgeted hardening adversary): src/dwell/hard.py, analysis_a12.py, scripts/run_a12.py, analyze_a12.py, a12_gate.py, mutate_a12.py, configs a12_*, tests/test_a12.py, test_a12_analysis.py. Plan A12 and A12.r1. Outcome REGIME-LIMITED.

## 2026-11-20, A13 (geometry-dependent slew)
- Added src/dwell/m0g.py, analysis_a13.py, scripts/run_a13.py, analyze_a13.py, a13_gate.py, mutate_a13.py, configs a13_*, tests/test_a13.py, test_a13_analysis.py, brute_g.py. Plan A13 (d412965) and A13.r1 (a603068).
- Outcome NO-GO (main seed 20261241, 6400 rows, largest best-method regret 1.38%). docs/record.md: A11 to A13 entries and repository-state notes added.
