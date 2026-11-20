### A13 (2026-11-20): geometry-dependent slew, ordering as a routing problem (post-hoc origin)

Written before any A13 code and before any A13 run. Everything under A13 is post-hoc in origin; it is a kill-test, not a confirmatory study, and a GO would still require a frozen protocol with fresh seeds. A13 follows the NO-GO of A11 and the outcome REGIME-LIMITED of A12 (both recorded above) and the author's decision of 2026-11-20 to run one more structure test before choosing between a further direction and the compact robustness paper. The plan is frozen by the git commit that adds this text (no branches, no tags); its hash is written to `docs/record.md` before any code is written.

#### A13.1 Question

In M0 the slew between two different threats is the same constant s. Real beam directors slew by an amount that depends on the angular separation of the two targets (Taylor 2021 attributes the success of the line formation to longer slews between angularly separated targets). With a slew matrix, the order of engagement is a routing problem with deadlines and random service times. Does this create a structure in which the best of the available non-DP methods, including a certainty-equivalent plan over all orders and exact rollouts, loses at least 5% of the DP optimum? Asked in M0 (retained progress, no idle, no budget), because A11 showed that waiting and budgets add no headroom.

#### A13.2 Model M0-G (final specification)

Everything in Section 2 holds (integer time grid, one effector, n threats, hidden class, first engagement free of slew, slew and one dwell step form one atomic action, progress retained, a kill counts only if it completes at a time <= d_j) except the slew. Threat j has a bearing theta_j in [0, 1]. Slewing from threat i to threat j (i different from j) costs s_ij = floor(c x |theta_i - theta_j| + 0.5) steps, s_ii = 0. The slew matrix is symmetric, known to every method, and observable from the start. Instance.s remains the mean slew s_bar and is used only by the control (A13.8).

Geometry factor:

| Geometry | Bearings | c |
|---|---|---|
| SPREAD (line formation) | independent uniform on [0, 1] | 3.0 x s_bar |
| CLUSTER (two groups) | each threat in group A (centre 0.15) or B (centre 0.85) with probability 1/2, bearing = centre + uniform(-0.05, 0.05) | 2.7273 x s_bar |

c is chosen so that the expected pairwise slew before rounding equals s_bar (E|theta_i - theta_j| is 1/3 for SPREAD and 0.3667 for CLUSTER). Bearings are drawn from their own random stream keyed by (master seed, instance index), after and independently of `make_instance_k`, so values, deadlines and priors equal those of A6 to A12 for the same (master seed, index, rho, s, a).

#### A13.3 Instances and grid

K = 2, D = (5, 10), values {1, 2, 3}, u_low = 0.35 as in A11.

| Factor | Values |
|---|---|
| swarm size n | 4, 5 |
| load rho (expected dwell over the last deadline, slew excluded) | 1.1, 1.5 |
| mean slew s_bar | 2, 4 |
| label accuracy a | 0.5, 0.9 |
| geometry | SPREAD, CLUSTER |

32 cells, 200 instances per cell (instances 0 to 199 identical in every cell), main master seed 20261241. Every cell is outside the M0 corner as defined in A2 (s >= 2 and rho <= 1.5). Stored per instance: the slew matrix, its mean and coefficient of variation, and the usual regime variables.

#### A13.4 Methods

The 11 rules of A11 without the myopic and lookahead variants (there is no approach): FAIR, EDF, LLF, SE, GMR, BI, CEP, LA1, ROLL_CEP, ROLL_GMR, ROLL_EDF, all using the slew matrix wherever a slew enters (SE, LLF, GMR, BI, the feasibility filter, CEP plans, rollout transitions). CEP enumerates all orders of the useful threats with the exact slews and expected remaining dwell, so it is a strong routing baseline; no further routing heuristic is added. The DP optimum is the reference. Value of every method: exact, by enumerating all 2^n class vectors.

#### A13.5 Metrics and go rule

As in A11.5 with the 11 methods: regret = 1 - sum V_method / sum V_dp per cell; best non-DP method per cell, recomputed in every resample; instance-level percentile bootstrap, B = 2000, seed 20261242, one-sided 95% lower bound; a cell is flagged if the bound is at least 0.05 and point-flagged if the point estimate is at least 0.05. A region is two cells that are face-adjacent on the grid (they differ in exactly one factor, by one position in the order given in A13.3). GO if there is a region of flagged cells, MARGINAL if no GO but a region of point-flagged cells, NO-GO otherwise.

Reported without a decision: per cell the best method and its regret; mean regret of each method; the control of A13.8 on the audit; the share of instances in which the first action of the DP differs from that of CEP; regret against the coefficient of variation of the slew matrix.

#### A13.6 Audit, tests and gate (before the main run)

Hard gates, same mechanism as A11 and A12 (`scripts/a13_gate.py`; the main run refuses to start without the gate file for the same code hash): all tests pass; all mutations are caught; the audit has no violation.

Tests, with an independent brute force that takes a slew matrix: (1) a matrix with every off-diagonal entry equal to s reproduces the M0 DP value and the value of every method to 1e-9 on at least 500 random instances; (2) the DP equals brute force for n = 2 and 3 with random, also asymmetric, matrices; (3) the exact value of the extracted DP policy equals the DP value; oracle >= DP >= every method >= 0; every rollout >= its base policy; (4) the generator: the matrix is symmetric with zero diagonal, bearings of CLUSTER lie in the two groups, and over 2000 draws the mean pairwise slew is within 0.3 steps of s_bar for both geometries and both s_bar; instances equal those of A6 to A12 for the same arguments; (5) the class-aware oracle uses the matrix. Mutations that must be caught: the model uses the uniform s instead of the matrix; the first engagement is charged a slew; row and column swapped in CEP; the oracle uses the uniform s; the rounding rule changed; the bearings of CLUSTER taken from SPREAD; the slew charged after a kill omitted.

Audit run: exploratory seed 1 (as in A11 and A12, stated explicitly), 20 instances in each of the 32 cells and in the 32 control cells. Invariants, all hard: values within [0, V_dp]; policy value equals DP value; every rollout >= its base policy; oracle >= DP. The audit never changes the grid, the thresholds, the method list or the go rule; an implementation repair is a dated sub-amendment committed before the main run. The cost is measured in the audit (the A11 and A12 estimates were off in both directions, so none is promised here). If the measured audit time predicts more than 6 hours for the main run, reduce in this order and record it before the run: (a) 200 to 100 instances per cell; (b) drop the cells with n = 4; (c) drop s_bar = 4.

#### A13.7 Runs

| Run | Instances per cell | Seed |
|---|---|---|
| `a13_audit` (32 cells) | 20 | 1 |
| `a13_ctrl_audit` (uniform slew s_bar, same instances) | 20 | 1 |
| `a13_main` (32 cells) | 200 | 20261241 |
| `a13_ctrl` (optional, after the decision) | 200 | 20261241 |

#### A13.8 Control

The control uses the same instances with the uniform slew s_bar in place of the matrix. It shows whether regret that appears with the matrix is caused by the geometry. It is run on the audit before the main run (exploratory) and on the main seed only after the decision.

#### A13.9 Pre-written decision table (applied mechanically by `scripts/analyze_a13.py`)

| Outcome | Rule | Action |
|---|---|---|
| GO | a region of flagged cells | A13 is a candidate for the paper spine. Before that: literature check on dynamic orienteering and stochastic routing with deadlines, the control on the main seed, a frozen confirmatory protocol with fresh seeds, a spot check at n = 6 |
| MARGINAL | no GO, a region of point-flagged cells | Staged extension, pre-specified: 1000 instances per cell, same seed (instances 0 to 199 identical), analysed again, disclosed |
| NO-GO | otherwise | A13 is closed and reported as a negative result for the best of 11 methods at n = 4 and 5. Proposed rule, to be confirmed by the author before the freeze: no further structure test; the compact robustness paper from A1 to A12 is written |
| No outcome | an assertion fails, a cell file is missing, a hard gate fails | Nothing is applied |

#### A13.10 Disclosure

- No A13 code or data exist. The idea comes from Taylor 2021 (the line formation succeeds through longer slews) and from the author's decision.
- My expectation, not a result [OPEN]: NO-GO is more likely than GO. CEP enumerates all orders with the exact slew matrix, so the only room is the interaction of random dwell and routing; A11 and A12 found strong baselines in every earlier structure.
- Literature: not checked for dynamic orienteering or stochastic routing with deadlines. From memory, unverified, routing heuristics can lose 10 to 25% against exact solutions in deadline-constrained routing, which is the reason for the test; the numbers are not used [MEM, OPEN].

#### A13.11 Limits

Bearings are one-dimensional; slew is linear in angular separation with no acceleration limits, no minimum engagement range and no atmosphere; progress is retained; n = 4 and 5, K = 2, one D; two geometries chosen by me; the 22 earlier methods of A11 with idle, range and budget are not used; the 11 methods are not exhaustive; thresholds (5%, 6 hours, the cluster parameters, 20 audit instances) are mine except the 5% rule decided on 2026-11-07; the corner definition is that of A2.

#### A13.12 Freeze

This text is committed before any A13 code exists. Code, configs, tests, audit and gate follow on `main`. Any change after the freeze is a dated sub-amendment committed before the run it concerns.
