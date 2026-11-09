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
