# Record, step 1 (provenance-tagged)

Tags: [CODE] produced or checked by code in this repository, [DOC] stated in a document read, [OPEN] unverified or inference, [MEM] memory.

## Verified in the sandbox on 2026-11-08 (Python 3.12.3, numpy 2.5.4, pandas 3.0.6, 1 CPU core, Linux x86_64)
- [CODE] 40 unit tests pass: DP equals an independent brute-force expectimax (n = 2 and 3, K = 2 and 3, s = 0 to 2, random priors including zero-mass classes, brute force allows idling and unkillable targets); the exact value of the extracted DP policy equals the DP value; oracle >= DP >= every rule; with one-hot priors DP = oracle; four hand-checked cases.
- [CODE] Two deliberate mutations (slew set to 0; over-aggressive canonicalisation) make 18 and 23 tests fail; the restored code passes all 40.
- [CODE] Full 100-cell grid with 10 instances per cell on a throwaway seed (2): all invariants held (1000 instances); wall time 159 s on one core, of which 122 s in the rho = 0.5 cells. Extrapolation to 500 instances per cell: about 8000 s on one such core [OPEN: extrapolation; laptop speed unknown].
- [CODE] Exploratory smoke run (seed 1, 20 instances, 12 cells): invariants held; regret table seen (disclosed in the analysis plan, Section 8).

## Not verified
- [OPEN] The BI rule is an adaptation of a Gittins-type index; it is not the published Glazebrook et al. index.
- [OPEN] The 2-point threshold of the companion test is a judgment.
- [OPEN] Timings on the author's laptops.

## Amendment A1 checks (2026-11-09, sandbox, Python 3.12.3)
- [CODE] 48 tests pass; CEP equals the DP optimum when classes are known (random one-hot priors, n = 3 to 5), and stays within [0, V_dp] with random priors.
- [CODE] Augmenting an existing cell file with CEP gives the same V_CEP and V_dp as a full re-run on the same seed (exploratory seed 1, 12 cells, 240 instances); augmentation of 240 instances took 0.4 s.
- [CODE] On exploratory seed 1 (N = 20 per cell, 12 cells) the mean CEP regret was 2.1% against 9.5% for GMR (disclosed in the plan, amendment A1). Not a result.

## Amendment A1 results (2026-11-09, from the author's runs; Windows 11, 4 workers)
- [DOC] a1_main: GO, 4 of 100 cells flagged (load 1.5 and 2.0, slew 0), max best-rule regret 8.98%. a1_s1: GO, 11 flagged, max 9.97%. a1_v2: NO-GO, 0 flagged, max 4.94%. a1_v4: NO-GO, 1 flagged, max 5.68%. a1_ratio: GO, 5 of 27 flagged, max 7.49%.
- [DOC] a1_summary (27 common cells): mean best-of-six regret 7.78% (main) against 2.11% best-of-seven; CEP alone 2.30%.
- [DOC] Wall times from the author's logs: a1_main 57.6 s, a1_s1 53.8 s, a1_v2 298.6 s, a1_v4 300.0 s, a1_ratio 1476.4 s (4 workers).
- [OPEN] The A1 runs may have used a Python outside the conda env (the first analysis call lacked matplotlib); run_meta of a1_v2 requested.

## Amendment A2 checks (2026-11-09, sandbox, Python 3.12.3)
- [CODE] 103 tests pass: Model2 with kappa = 1 equals M0; Model2 DP equals an independent brute force with explicit evidence for kappa in {1, 0.5, 0}, K in {2, 3}, n in {2, 3}, slew 0 to 2; CEP and LA1 are exact when classes are known; oracle >= DP >= every rule.
- [CODE] After the decay-aware change, CEP values on M0 exploratory data are bit-identical to those before.
- [CODE] Timing on a throwaway seed (sandbox core, 3 instances per setting): hardest cells 0.56 s per instance (wide), 1.04 s (k3), 0.11 s (decay05), 0.09 s (decay0) including all eight rules.

## Amendment A2 results (2026-11-09, from the author's runs; Windows 11, conda env de-dwell, Python 3.12.16, numpy 2.5.4, pandas 3.0.6, 4 workers; run_meta of a2_wide confirms the environment, commit c77420f)
- [DOC] On the 36-cell A2 grid, best of eight rules: reference mean 3.46% (max 8.98%, 3 flagged, 0 non-corner); a2_wide 4.70% (9.15%, 13, 9); a2_k3 5.12% (8.41%, 18, 11); a2_decay05 3.65% (9.47%, 4, 0); a2_decay0 3.76% (9.50%, 6, 0).
- [DOC] Mean CEP regret 3.70, 6.06, 6.72, 3.96, 4.13%; mean LA1 regret 8.10, 7.55, 8.87, 8.45, 8.50%.
- [DOC] Information gap (oracle vs DP) in a2_wide: 8.0, 6.1, 2.2% at label accuracy 0.5, 0.7, 0.9; in a2_k3: 10.3, 6.9, 2.3%.
- [DOC] By the A2 table: non-corner gap in a2_wide and a2_k3; none in the decay variants.

## Amendment A3 checks (2026-11-09, sandbox, Python 3.12.3)
- [CODE] 120 tests pass: rollout >= base rule and <= DP for CEP, GMR, EDF, BI on M0, K = 3 and decay models; rollout of CEP is optimal when classes are known.
- [CODE] Exploratory seed 1 (40 instances, K = 3): mean evaluation time per instance 8.6 ms (CEP), 39 ms (ROLL_CEP), 30 ms (ROLL_GMR), 19 ms (ROLL_EDF); mean regret 6.0% (CEP), 2.2% (ROLL_CEP). Disclosed in the plan; not a result.

## Amendment A3 results (2026-11-09, from the author's runs; same environment as A2)
- [DOC] Best of eleven rules on the A2 grid: a3_main mean 0.83% (max 2.95%), a3_wide 1.12% (2.95%), a3_k3 1.57% (3.79%); no flagged cell; NO-GO in all three. ROLL_CEP mean 1.11, 1.54, 2.29%; ROLL_GMR 1.79, 2.87, 2.75%; ROLL_EDF 7.76, 6.57, 7.92%.
- [DOC] Evaluation time per instance (all worlds): CEP 3.95, 4.88, 25.07 ms; ROLL_CEP 19.86, 33.08, 79.03 ms.

## Amendment A4 checks (2026-11-09, sandbox, Python 3.12.3)
- [CODE] 123 tests pass: every method built from a wrong belief has a true-model value in [0, V_dp]; the DP built from the correct belief reproduces V_dp; believed instances keep labels, values and deadlines.
- [CODE] Timing on a throwaway seed (K = 3): about 2.7 s per DP at load 0.8, slew 0; one instance needs one DP for the truth and one per wrong belief. Slew 0 was removed from the A4 grid for cost.
- [CODE] Exploratory seed 1, 12 instances per cell, 8 cells: best-method regret 0.2 to 4.6% depending on the belief (disclosed in the plan; not a result).

## Amendment A4 results (2026-11-10, from the author's runs; same environment)
- [DOC] a4_k3: no non-corner gap; 1 flagged cell (load 2.0); max regret 6.57%; mean regret over wrong-belief cells 1.03%. a4_wide: criterion met (5 flagged, 3 non-corner; max 14.64%), but every cell with a point estimate of at least 5% has believed accuracy 0.5; beliefs 0.7, 0.9, 0.99 cost at most 0.84% on average.

## Amendments A5 and B0 checks (2026-11-10, sandbox, Python 3.12.3)
- [CODE] Model3 DP equals an independent brute force with explicit signal likelihoods (q in {0.5, 0.7, 0.95}, D in {(2,4), (3,5), (1,3)}, n = 2 and 3, slew 0 and 1); q = 0.5 reproduces M0; policy recursion equals world enumeration; sensing never lowers the optimum.
- [CODE] Learned scorer: gradient equals finite differences; training loss falls; policy returns useful actions.
- [CODE] Exploratory (seed 1, small samples): in-dwell signal gains of 0.3 to 1.1% for D = (3, 8) and 4.3 to 5.7% for D = (5, 12); disclosed in the plan; not results.
- [CODE] DAgger training check on the training seed: agreement with the DP teacher about 90%.

## Amendments A5 and B0 results (2026-11-11, from the author's runs; same environment)
- [DOC] a5_long: mean sensing gain 2.1% (non-corner), 18 to 36% of the oracle gap closed, 2 flagged cells (one cell at two q), best planner regret at most 2.4%; a5_sense: gain 0.1 to 0.4%, no flagged cell.
- [DOC] b0_learn: regret against the DP at n = 5: CEP 6.68%, ROLL_CEP 1.96%, ROLL_GMR 2.70%, learned 2.83%; slews outside the training range: ROLL_CEP 0.90%, learned 4.00%; value relative to ROLL_CEP: learned -1.36% (n = 7), -1.73% (n = 9), -3.23% (n = 11); time per decision at n = 9: learned 0.12 ms, ROLL_CEP 33 ms.

## Amendment A6 checks (2026-11-11, sandbox, Python 3.12.3)
- [CODE] Tests: the observation-based evaluator reproduces the exact value of DPMIS, CEP and ROLL_CEP when the believed model is the true one; a wrong belief never beats the calibrated optimum; the Bayes-adaptive DP equals an independent brute force (with and without dwell noise) and with a single scale equals the M2 DP; the value of the Bayes-adaptive policy averaged over the scales equals its DP value; calibrated >= Bayes-adaptive >= nominal planner on average.
- [CODE] Timing (sandbox core): noisy-dwell smoke about 3.6 s per instance and cell; extrapolated a6_main about 8000 s single-core [OPEN].
- [CODE] Exploratory smoke results disclosed in the plan; not results.

## Amendment A6 results (2026-11-12, from the author's runs; Windows 11, conda env de-dwell, Python 3.12.16, numpy 2.5.4, pandas 3.0.6, 4 workers; run_meta of a6_main: commit 8c8dce8, started 2026-11-12 16:11)
- [DOC] a6_main (noise, n = 4, D0 = (5, 10), 8 cells, 200 instances each): mean regret of the best method at dwell error -40, -20, +20, +40%: 13.99, 5.48, 4.81, 14.82%; max 21.70, 7.33, 7.49, 21.61%; flagged cells 8, 4, 2, 8. By method at -20%: DPMIS 6.36, CEP 10.13, ROLL_CEP 5.67%; at +20%: 4.81, 8.27, 6.58%. By label accuracy at -20%: 4.10% (0.5) and 6.85% (0.9).
- [DOC] Adaptation, pooled: truth 0.8: nominal best (DPMIS) 4.76%, Bayes-adaptive 1.48%, recovery 0.689 (0.655 to 0.722); truth 1.2: nominal best (ROLL_CEP) 6.25%, Bayes-adaptive 2.09%, recovery 0.666 (0.627 to 0.702).
- [DOC] a6_det (no noise): mean regret 20.65, 16.92, 9.21, 20.32%; recovery 0.854 (nominal 9.07% to 1.32%) and 0.900 (22.69% to 2.27%).
- [DOC] Table outcome: FRAGILE_20 and adaptation recovers (recovery criterion met through truth 1.2 only).
- [OPEN] Mean regret at 20% is close to the 5% bar; realistic dwell error range not verified (Burley tables unread); one scale shared by all threats; Bayes-adaptive policy is an upper bound.

## Amendment A7 inputs and checks (2026-11-14, sandbox, Python 3.12.3, numpy 2.5.4, pandas 3.0.6, 1 core)
- [FILE] Burley AFIT 2012 (DTIC ADA557011), Tables 10 to 15 (printed pages 107 to 112) and Conclusions (printed pages 93 to 95) read from the author's PDF (132 PDF pages). Tables are images; values transcribed by eye into docs/burley_tables_transcribed.csv (144 rows); 94 of 144 rows confirmed by an independent OCR pass; the rest checked by eye only.
- [CODE] scripts/burley_stats.py (output in docs/burley_stats_output.txt): relative dwell error RMSE / DT at 1.045 micrometre, all six atmosphere methods: median 13%, quartiles 10 to 19%, maximum 46%; 20% of rows above 20% and 5% above 40%. GFS forecasts: median 10%, quartiles 8 to 13%, maximum 30%. Climatology and ground tab: median 16%, quartiles 13 to 26%, maximum 46%. 1976 US Standard: median 19%, quartiles 16 to 27%, maximum 31%. COIL 1.31525 micrometre: median 48%, range 14 to 327% (not used for calibration).
- [DOC] Burley limits: the truth is a software dwell from radiosonde profiles with assumed aerosol and turbulence (printed pages 34, 35, 42); atmosphere only; one airborne ELLA engagement of an extremely hard ground target at about 3,977 m ground range (printed pages 42, 44, 45); RMSE has no sign; 25 days per row (printed page 37); the thesis states that solid state lasers reduce the impact of a dynamic atmosphere (printed page 95).
- [CODE] New code: src/dwell/adapt.py (class sets with merging, robust Bayes-adaptive model with projection of out-of-support observations, practical adaptive planners PLUG_DP, PLUG_CEP, QMDP, MIXCEP, exact recursive evaluator eval_rec), scripts/run_a7.py, analyze_a7.py, decide_a7.py, configs a7_main, a7_noise0, a7_noise2, a7_noise3, a7_h1, a7_h2, tests/test_adapt.py (23 tests; 202 tests in the suite pass).
- [CODE] 16 deliberate mutations of adapt.py: 15 caught; 1 equivalent (class lower bound m is a function of dwell). Five mutations survived the first version of the tests and led to new tests.
- [CODE] Timing (sandbox core, exploratory seed 1): a7_main about 4.2 s per instance (24 instances: 101 s); noise m = 0, 2, 3 about 0.7, 4.9, 4.4 s per instance in the small smoke samples, 1.2, 7, 11 s in the slower cells; h1 about 2.6 s; h2 about 0.4 s. Full A7 extrapolated about 30,000 s single-core [OPEN, laptop speed not measured].
- [CODE] Exploratory smoke results (seed 1, 8 to 24 instances; not results) are disclosed in the plan, A7.5.
- Plan: Section 12 of docs/analysis_plan_P0.md now also holds amendment A7 (dated 2026-11-14; tag p0-a7-plan to be set before the first run).

## A7 results (2026-11-14, author's laptop, 4 workers, Python 3.12.16, numpy 2.5.4, pandas 3.0.6, Windows 11)
- [CODE] a7_main: 1600 instances, 7,940 s, mean 16.0 s per instance (max 245 s). a7_noise2: 800 instances, 4,786 s, mean 18.7 s per instance (max 216 s). Commit d94034b. Files: docs/results_a7/.
- [CODE] Decision script outcome: REGIME-LIMITED (M, P, S1, S2, S3 met; N not met). Values in plan A7.7.
- [CODE] A6 files re-analysed (scripts/analyze_a6_ceiling.py, docs/results_a6_ceiling_check.txt): ceiling cost at truth 1.0 is 0.78% (noise) and 0.76% (no noise); best nominal planner at SD16: 3.88% (noise), 11.01% (no noise).
- [OPEN] Open: noise level of real targets; descriptive runs a7_noise0, a7_noise3, a7_h1, a7_h2 not yet made.

## A8 literature check and regime law (2026-11-14)
- [DOC] Full text read: Rocha et al., Electronics 15(21):4952 (2026) and Schaeffer et al., J. Compos. Sci. 8(11):471 (2024). Neither reports replicate statistics. Rocha: +-25% coupling change gives -23% / +42% threshold time in one case. Schaeffer: CFRP perforation 0.4 to 0.5 s at 120 kW, 20 mm, 4 mm.
- [REF] Snippet-level only: aluminium coupons in shear flow, 2A12 model-experiment error 6.2%, Johnson NPS dwell table, an AFIT EADSIM thesis (random dwell draw per target), HPM Monte Carlo arXiv:2602.08477. 9 searches, 2 page reads; not systematic.
- [CODE] scripts/fit_regime_law.py on docs/results_a7/*: nominal expected regret = 77.3% x r^2 with r = scale-error SD / noise SD (steps); exponent 1.98; max relative residual 1.6%; 8 points, 2 runs. Output: docs/results_a8_regime_law_fit.txt.
- [CODE] Pre-registered prediction for a7_noise3 (percent): SD10 0.52, SD14 1.03, SD16 1.37, SD20 2.06. Criterion in plan A8.2. tests/test_regime_law.py (5 tests; 5 mutants caught); suite 207 tests pass.

## A8 outcome (2026-11-14)
- [CODE] a7_noise3 (800 instances, commit 7811ae2): pre-registered check HOLDS 4 of 4; observed nominal regret 0.60, 1.21, 1.61, 2.42% against predicted 0.52, 1.03, 1.37, 2.06% (biased low by 15 to 18%).
- [CODE] Correction: the earlier exponent 1.98 and 8-point collapse are by construction (the mixture is linear in p). Independent evidence: S = R(0.8) + R(1.2) = 12.32, 6.28, 4.83 at noise SD 0.707, 1.0, 1.225; S x SD^2 = 6.16, 6.28, 7.25. Shift dependence is not quadratic (R(0.6)/R(0.8) = 1.8, R(1.4)/R(1.2) = 2.6).

## A9 design (2026-11-14, sandbox)
- [CODE] configs a9_m1, a9_m2, a9_m3, a9_x2 (seeds 20261201 to 20261204, 200 instances per cell, nominal planners only), scripts/analyze_a9.py (criteria C1 to C3), tests/test_a9.py (8 tests, 8 mutants caught; suite 215 tests pass).
- [CODE] Smoke on seed 1 (24 instances per run): C1 not met in the tiny sample; doubled-noise run 36 to 43% above the reference. Disclosed in plan A9.3. No run made.

## A9 outcome (2026-11-14, author's laptop, 4 workers)
- [CODE] Four runs of 1600 instances: 551.6, 790.7, 733.4, 554.3 s (about 44 minutes in total). Files: docs/results_a9/decision_a9.json.
- [CODE] Outcome NOT-SUPPORTED. C1 down side fails (K x 100 = 2.64, 2.48, 3.86), up side passes (3.47, 3.29, 3.43); C2 fails on the down side; C3 fails on both sides: doubled noise at a two-step shift gives 7.04% (down) and 9.47% (up) against 5.27% and 6.95% at the same ratio with noise +-1 (+34% and +36%).
- [OPEN] Direction decision pending (author).

## A10 design (2026-11-14, sandbox)
- [CODE] configs a10_n3_m1, a10_n4_m1, a10_n5_m1, a10_n3_m2, a10_n4_m2, a10_n5_m2 (seeds 20261211 to 20261216), a10_n6_m1 (optional, 20261217); scripts/analyze_a10.py; tests/test_a10.py (13 tests, 11 mutants caught; suite 228 tests pass).
- [CODE] Smoke on seed 1 (24 instances per run): no growth of E16 with n (4.59, 5.39, 3.02 at noise +-1; 2.57, 2.59, 1.71 at noise +-2). No run made.

## A10 outcome (2026-11-14, author's laptop, 4 workers)
- [CODE] Six runs, 34, 288, 1548, 54, 527, 3417 s (about 98 minutes; my hedge of 40 to 90 minutes was slightly low). Files: docs/results_a10/decision_a10.json.
- [CODE] E16 (percent) at n = 3, 4, 5: 4.56, 3.81, 3.45 (noise +-1); 2.23, 1.97, 1.84 (noise +-2). Outcome FLAT-OR-FALLING: the loss falls with n. Decision: write the compact paper.
- [OPEN] Optional before writing: adaptive (QMDP) gain at n = 5 without the ceiling; a10_n6_m1.
- [REF] Candidate venue seen on the publisher pages: Journal of Defense Modeling and Simulation (SAGE); the publisher guidelines say papers should ideally be 10 to 20 pages; indexing and ranking come from third-party listings and are unverified.


## A11 (2026-11-17 to 2026-11-19, added 2026-11-20; record.md had stopped at A10)
- [CODE] Plan frozen in commit d44d9be, code and gate in 910f210 (22 tests, 11 of 11 mutations caught, audit 960 rows main grid and 480 rows control, seed 1, no violation). Main run seed 20261222: 48 cells, 9600 rows, 0 violations, wall 3571 s on 4 workers.
- [CODE] Outcome NO-GO: 0 flagged cells, largest best-method regret 4.12% (rho 1.1, s 2, a 0.5, gamma 2, eps 0.5; ROLL_CEP_L), mean over cells 0.62%; best method is a rollout in 95.8% of cells. Value of idle 0.42% at eps 0.5 and 0 at eps 0.8 and infinity. Recomputed from the raw cell files on 2026-11-20 (summary_cells.csv).
- [DOC] Plan text A11.r1 was committed in 910f210 and appended to the plan only after the run (append_a11_r1.py failed its end-of-file check); disclosed in the plan.

## A12 (2026-11-19 to 2026-11-20)
- [CODE] Plan frozen in commit 4d795d4, code and gate in ac3a94a (70 tests passed, 1 skipped, 12 of 12 mutations caught, audit 160 rows seed 1). Main run seed 20261228 (1600 rows) and noise run seed 20261229 (800 rows), 0 violations, commit ac3a94a.
- [CODE] Outcome REGIME-LIMITED: X true (best non-adaptive worst case 10.24%, 5th percentile 9.63), R true (best adaptive 3.62%, reduction 0.646, 5th percentile 0.614), S false (largest difference +0.30 points, 5th percentile 0.08), N false (noise +-2: 5.38%, 5th percentile 4.80). Minimax mixture equals the pure defender IID_DP. Saved analysis output re-read on 2026-11-20; matches.
- [OPEN] The three best adaptive defenders are built with knowledge of the attack family, and IID_DP has zero regret under PH50_14 by construction.

## A13 (2026-11-20)
- [CODE] Plan frozen in commit d412965 (before any code), code, r1, audits and gate in a603068 (25 tests, 10 of 10 mutations caught; audit seed 1, 640 rows matrix and 640 rows control, no violation; gate hash 16bb7279...0098). 
- [CODE] Audit review recomputed from the raw audit files on 2026-11-20 equals the console output: best regret at most 1.52%, mean best regret 0.72% (matrix) against 0.35% (uniform control).
- [DOC] Main run seed 20261241: 32 cells, 6400 rows, 0 violations, wall 2021.2 s. Outcome NO-GO: largest best-method regret 1.38%, no cell flagged, ROLL_CEP best in all cells. Details in the plan, A13 outcome. Control on the main seed not run.
- [CODE] Check of A13.r1 item 1 (sandbox, 2000 draws, n = 5, master seed 9): closure changed 60%, 60%, 27%, 61% of the matrices (SPREAD 2 and 4, CLUSTER 2 and 4); mean slew after rounding and closure 1.81, 3.80, 1.87, 3.82 against s_bar 2, 4, 2, 4, so the geometry cells have about 0.90 to 0.96 times the mean slew of the control.

## Repository state notes (review of 2026-11-20)
- [CODE] Full test suite: 346 passed in 39.7 s (sandbox, one core). The plan file is append-only over all commits (the A1 commit replaced the placeholder line "None yet.").
- [CODE] No git tags exist after a10-result (workflow decision of 2026-11-17: no branches, no tags); freeze points after A10 are commits d44d9be (A11), 4d795d4 (A12), d412965 (A13). Earlier text in the literature document that mentions tags p0-a11-plan and p0-a12-plan does not match the repository.
- [CODE] The plan holds the paragraph "A12 outcome (2026-11-19): REGIME-LIMITED" twice with identical text.
- [CODE] environment-lock.yml is a UTF-16 export of the conda base environment, not of de-dwell; run_meta.json files hold the actual Python, numpy and pandas versions.
- [CODE] The A13 gate hashes only src/dwell/m0g.py and scripts/run_a13.py; changes to dp.py, m0.py, rules.py, instances.py or analysis_a13.py would not close it.
- [OPEN] Raw per-instance cell files are git-ignored; keep a dated archive with SHA-256 outside git (results/a13_main, a12_main, a12_noise2, a11_main).
