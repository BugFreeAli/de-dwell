
### A11 (2026-11-17): range-dependent dwell, engage now or wait (post-hoc origin)

Written before any A11 code and before any A11 run. Tag for the frozen text: `p0-a11-plan`. Naming: the row "plan A11, scale-up beyond the exact DP" in the table of A10.2 was not triggered (outcome FLAT-OR-FALLING, A10.5). A11 now denotes this test, decided by the author on 2026-11-16 (literature document Section 10.26). A12 (budgeted adversary) has its own plan. Everything under A11 is post-hoc in origin; the amendment is a kill-test, not a confirmatory study, and a GO would still require a frozen protocol with fresh seeds.

#### A11.1 Question

Does a deterministic approach, which makes the dwell rate grow as the range falls, create a structure in which the best of the available non-DP methods, including exact rollouts, loses at least 5% of the DP optimum outside the M0 corner? The question is asked in a model with an explicit idle action, so that "engage now or wait" is part of the decision.

#### A11.2 Model M1 (final specification)

Everything in Section 2 holds (integer time grid, one effector, n threats, hidden class, uniform slew s, first engagement free of slew, slew and one dwell step form one atomic action, progress retained, kill counts only if it completes at a time <= d_j) except the following.

1. Approach. Threat j has range R_j(t) = u (d_j - t) with one closing speed u for all threats; breach at R_j = 0, that is at t = d_j (the M0 deadline). With a common speed the range is a relabelling of the time to breach tau = d_j - t. Different speeds per threat are not part of A11.
2. Dwell rate. A dwell step that starts at time t0 on threat j (after the slew) adds g(tau) progress units, tau = d_j - t0 >= 1: g = 1 if tau > tau_c, g = gamma if tau <= tau_c, with gamma an integer, tau_c = D_K (the largest class requirement). One unit is one dwell step at far range. Class k is killed when progress >= D_k at the end of the step (overshoot allowed). gamma = 1 is M0.
3. Idle. A new action advances t by one step, leaves the pointing direction unchanged, adds no progress and uses no energy. It is available while t < max_j d_j.
4. Energy factor epsilon. If epsilon is finite, the number of dwell steps is at most B = ceil(epsilon x sum_i E_i), E_i = prior mean of D (rate-1 steps). Slews and idle steps cost nothing; no recharge. epsilon = infinity means no budget. The state is (t, cur, prog, e) with e the dwell steps used; for epsilon = infinity e is omitted.
5. Observation and belief: kill flags and progress units; the belief of an alive threat with progress x is the prior restricted to classes with D_k > x, renormalised.
6. Canonicalisation. A threat is DEAD if it is killed, or if x + P_j(t) < min over possible classes of D_k, with P_j(t) the sum of g over all steps from t to d_j - 1 (zero slew, unlimited energy). This bound is conservative, so the merge is exact. Actions are the alive threats that can still be killed given the slew and the remaining energy (best case: the last B - e steps before d_j), plus idle. Idle makes the restriction exact even though the value can now rise with t: a dwell step on an unkillable threat is weakly dominated by idle (same time, no slew, no energy, pointing unchanged).
7. Objective and exact DP as in Section 2: maximise the expected value of threats killed before breach; memoised expectimax over the canonical state, with idle.

Known property, tested (A11.7): with epsilon = infinity, idle is weakly dominated when progress is retained and dwelling costs nothing. An idle step before a slew can be replaced by starting the slew one step earlier and the extra dwell step adds progress at a lower or equal rate; an idle step while pointing at an alive threat is replaced by a dwell step. The DP value with and without idle must therefore be equal for epsilon = infinity. Waiting can only pay when dwelling has a cost, which is why epsilon is a factor. The cells with epsilon = infinity are the model as asked for on 2026-11-16; they are expected to show a wait share of exactly zero [OPEN until tested].

#### A11.3 Instances and grid

Generator `make_instance_k` unchanged (K = 2, n = 4, D = (5, 10), values {1, 2, 3}, u_low = 0.35): for the same (master seed, index, rho, s, a) the instances equal those of A6 to A10, so M1 cells with gamma = 1 can be compared instance by instance with M0.

| Factor | Values |
|---|---|
| load rho (rate-1 expected dwell over the last deadline) | 1.1, 1.5 |
| slew s | 2, 4 |
| label accuracy a | 0.5, 0.9 |
| approach gain gamma | 2, 4 |
| energy factor epsilon | 0.5, 0.8, infinity |

48 cells, 200 instances per cell (instances 0 to 199 identical in every cell), main master seed 20261222. Stored per instance: rho_hat (rate 1), rho_boost (all dwell placed as late as possible at the boosted rate), the energy budget B, and the usual regime variables. The grid contains no cell with s < 2 or rho > 1.5, so every cell is outside the M0 corner as defined in A2 (s >= 2 and rho <= 1.5). Control cells (gamma = 1) are used in the regression tests only.

#### A11.4 Methods

Two variants of each rule: `_M` (myopic: for threat j the rate is frozen at g(d_j - t), the value at decision time) and `_L` (range-lookahead: the true profile g from the earliest start t + slew on). For a class with remaining units r the number of steps is n_k = ceil(r / g) under M and the smallest m with sum of the first m rates >= r under L. Expected remaining time is sum_k p_k n_k; P(kill within a limit) and the ratios of GMR and BI use n_k in place of the remaining dwell; CEP durations use n_k with M or L applied to each plan element at its own start time (L) or at the decision-time rate (M). All methods use the exact useful-action filter of the model (A11.2, item 6), ties as in Section 4.

Rules (11): FAIR, EDF, LLF, SE, GMR, BI, CEP, LA1, ROLL_CEP, ROLL_GMR, ROLL_EDF. The rollouts use the exact one-step transitions of M1 and the M or L variant of the named rule as base policy (ROLL_X_M, ROLL_X_L). Total 22 methods. Only the DP and the six rollouts can choose idle (a rollout evaluates idle followed by the base policy); the other 16 methods never idle, which is part of their definition. The DP optimum (with idle) is the reference. Value of every method: exact, by enumerating all 2^n class vectors.

#### A11.5 Metrics and go rule

As in Sections 5 and 6 and A2: regret = 1 - sum V_method / sum V_dp per cell; best non-DP method per cell (best of the 22, recomputed in every resample); instance-level percentile bootstrap, B = 2000, seed 20261223, one-sided 95% lower bound; a cell is flagged if the bound is at least 0.05 and point-flagged if the point estimate is at least 0.05. A region is two cells that are face-adjacent on the grid (they differ in exactly one factor, by one position in the order given in A11.3), both outside the corner (s >= 2 and rho <= 1.5, true for the whole grid). GO if there is a region of flagged cells, MARGINAL if no GO but a region of point-flagged cells, NO-GO otherwise.

Reported with every outcome (descriptive, no decision): (i) per cell the best method and its regret; (ii) mean regret of each method, and of each L variant minus its M variant (paired bootstrap, seed 20261224); (iii) for the DP in the first 50 instances of each cell: the value of idle, 1 - V_noidle / V_dp, the probability-weighted share of instances in which the optimal policy waits at least once, and the share in which it leaves an alive threat that has progress (preemption), both by exact world enumeration; (iv) rho_hat and rho_boost of flagged cells.

#### A11.6 Baseline audit (before the main run)

1. Hard gates: the whole test suite passes, including A11.7; all mutations of A11.7 are caught. No main run before both.
2. Audit run, audit seed 20261221, 20 instances in each of the 48 cells and in the control cells. Invariants, all hard: V_dp >= V_method >= 0 for all 22 methods; the exact value of the extracted DP policy equals V_dp; every rollout >= its base policy (tolerance 1e-9); for epsilon = infinity the DP waits in no instance; with gamma = 1 every M and L variant equals the M0 rule.
3. Reviewed, not gating: per rule, mean regret of M and L; if an L variant is worse than its M variant by more than 1 point on average, the implementation is checked (L is not guaranteed to be better); the identity of the best method per cell; the share of cells in which the best method is a rollout.
4. The audit may only trigger the repair of an implementation error against the invariants above. It never changes the grid, the thresholds, the method list or the go rule. A repair is a dated sub-amendment with a new tag `p0-a11-plan-r1`; the main run uses the last tag. A repeated audit uses seed 20261219.
5. Cost probe, throwaway seed 2, three instances in each of the four slowest-looking cells (rho 1.5, s 2, gamma 2, epsilon 0.5 and infinity; rho 1.1, s 2, gamma 2, epsilon 0.5 and infinity): seconds per instance for the DP and for each method. Predicted laptop time = sandbox time x 3.8 / 4 workers (the factor of A10.3). If the prediction exceeds 10 hours, reduce in this order and record it before the run: (a) 200 to 100 instances per cell; (b) drop epsilon = 0.8; (c) drop rho = 1.5. The 10 hours are my limit [OPEN].

#### A11.7 Tests, written before the main run

`tests/test_a11.py`, with an independent brute force `tests/brute_m1.py` (explicit class-vector expectimax, unrestricted dwell actions, idle, energy):
1. Regression, rate: gamma = 1 (g = 1 everywhere), epsilon = infinity, idle allowed: the M1 DP value equals the M0 DP value to 1e-9 on at least 500 random instances (n = 2 to 4, K = 2 and 3, s = 0 to 4, random priors); the same for every method (M and L) against the M0 rule.
2. Regression, scaling: constant rate c in {2, 3} with D' = c D equals M0 with D; constant rate c with arbitrary D' equals M0 with ceil(D' / c) (overshoot).
3. DP equals brute force for n = 2 and 3, random rate profiles, random tau_c, s = 0 to 3, with and without budget.
4. Idle dominance: for epsilon = infinity the DP value with idle equals the value without idle on at least 300 random instances; for finite epsilon the value with idle is never lower, and is strictly higher in at least one constructed case.
5. Policy value: the exact value of the extracted DP policy equals the DP value; V_orc >= V_dp >= V_method; rollout >= base; L = M when g is constant.
6. Mutations that must make at least one test fail: rate evaluated at the wrong time, rate profile reversed, kill with strict instead of weak inequality, slew skipped, idle charged to the budget, budget check skipped, over-aggressive DEAD canonicalisation, L variant using the M duration, idle shifting time by zero, deadline off by one. A surviving mutant requires a new test or a documented equivalence.

#### A11.8 Runs

| Run | Config | Instances | Seed |
|---|---|---|---|
| `a11_audit` | `configs/a11_audit.json` | 20 per cell | 20261221 |
| `a11_main` | `configs/a11_main.json` | 200 per cell | 20261222 |
| `a11_ctrl` (optional, after the decision) | gamma = 1 | 200 per cell | 20261222 |

Smoke runs use exploratory seed 1 and are never results. Raw per-instance files are archived with SHA-256 as in Section 11; software versions, hardware, commit and seed are written to `run_meta_*.json`.

#### A11.9 Pre-written decision table (applied mechanically by `scripts/analyze_a11.py`)

| Outcome | Rule | Action |
|---|---|---|
| GO, budget-driven | a region of flagged cells with at least one cell with epsilon < infinity | A11 is a candidate for the paper spine. Before that: literature check on time-dependent scheduling (processing times that depend on the start time), a frozen confirmatory protocol with fresh seeds, a spot check at n = 5. The decision between A11 and A12 is taken after both |
| GO, ordering-only | regions only in cells with epsilon = infinity | Report. The wait share must be zero there (A11.2); the loss comes from range-aware ordering. Check that the best method is not an artefact of one base rule before any further step |
| MARGINAL | no GO, a region of point-flagged cells | Staged extension, pre-specified: 1000 instances per cell, same seed (instances 0 to 199 identical), analysed again, disclosed |
| NO-GO | otherwise | A11 is closed and reported as a negative result for the best of 22 methods at n = 4. Direction decided after A12 |
| No outcome | an assertion fails, a cell file is missing, a hard gate of A11.6 fails | Nothing is applied |

A GO does not show that learning is needed: it shows that no method among the 22 closes the gap. A learned policy would still have to be compared against these baselines at equal decision time.

#### A11.10 Disclosure

- Before this text was written I ran one throwaway script outside the repository (not repository code, not the M1 generator): 400 random instances with n = 3, D = (3, 8), random priors, values, deadlines, slew 0 to 3, a three-level rate profile with random thresholds, DP with and without idle. Without a dwell budget idle improved the value in 0 of 400 instances; with a dwell budget of 2 to 5 steps it did in 325 of 400. It is not a result and no threshold was derived from it. It led to the energy factor and to item 4 of A11.7. [CODE, scratch]
- The idea of range-dependent dwell comes from the author's 2026-11-16 decision, which follows the negative outcome of A10. No A11 data exist.
- Literature: not checked for time-dependent scheduling (processing times that depend on the start time). From memory, unverified, this is an established operations research topic; novelty of any claim is not established [MEM, OPEN].

#### A11.11 Limits

Common closing speed (range equals time to breach); a two-level step profile instead of a smooth 1/R^2 law; integer gain; no atmospheric extinction, pointing error or minimum engagement range; progress is retained (A2 showed that decay shrinks the M0 gap); a plain dwell-step energy budget without recharge or thermal state; n = 4, K = 2 and one D only; waiting is available only to the DP and the rollouts; the 22 methods are not exhaustive; the thresholds (5%, 10 hours, the epsilon levels, tau_c = D_K) are mine except the 5% rule decided on 2026-11-07; the corner definition is that of A2.

#### A11.12 Freeze

This text is committed and tagged `p0-a11-plan` before any A11 code exists. Code, configs, tests and the audit follow on branch `a11-range-dwell`. Any change of this text is a dated sub-amendment with a new tag.
