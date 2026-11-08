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
