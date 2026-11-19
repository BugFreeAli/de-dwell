### A12 (2026-11-19): budgeted hardening adversary, exploitability and minimax defender (post-hoc origin)

Written before any A12 code and before any A12 run. Everything under A12 is post-hoc in origin; it is a kill-test, not a confirmatory study, and a positive outcome still requires a frozen protocol with fresh seeds. A12 follows the NO-GO of A11 (outcome recorded in A11) and the author's decision of 2026-11-16 (literature document, Section 10.26). The plan is frozen by the git commit that adds this text (workflow decision of 2026-11-17: no branches, no tags); its hash is written to `docs/record.md` before any code is written.

#### A12.1 Question

An attacker with a hardening budget chooses how strongly and how widely the threats are hardened (longer dwell), without the defender knowing. (i) How much does each defender in a fixed set lose against the attacker's best choice (exploitability)? (ii) Does an adaptive defender, or a minimax mixture of defenders, remove a material part of the worst-case loss of the best non-adaptive defender? (iii) Is adaptation built on a shared-scale belief itself exploitable when only some threats are hardened?

Not tested: label spoofing (inverted labels) and attacker choice of deadlines. A4 found that plausible errors in label accuracy cost the best method at most 0.84% on average when labels are not ignored; deadlines are observed by the defender, so the attacker's choice of arrival spacing is a choice of cell, reported as a secondary maximum.

#### A12.2 Model and attacker strategies

Model, instances, cell grid and planner knowledge as in A6 and A7 (n = 4, base classes (5, 10), values {1, 2, 3}, load 1.1 and 1.5, slew 2 and 4, label accuracy 0.5 and 0.9, retained progress, noise offsets (-1, 0, 1) with probabilities (0.25, 0.5, 0.25) in the main run). A hardened threat of base class c has dwell round(lambda_h x D0[c]) + e, a non-hardened one D0[c] + e. Each threat is hardened independently with probability q; q = 1 is the shared scale of A6 and A7. The defender observes only the dwell received and the kill flags.

Hardening budget beta = q (lambda_h - 1), the expected scale increase per threat. Pure attacker strategies:

| Name | q | lambda_h | beta |
|---|---|---|---|
| NONE | 0 | 1.0 | 0 |
| SH12 | 1 | 1.2 | 0.2 |
| PH50_14 | 0.5 | 1.4 | 0.2 |
| PH25_18 | 0.25 | 1.8 | 0.2 |
| SH14 | 1 | 1.4 | 0.4 |
| PH50_18 | 0.5 | 1.8 | 0.4 |

Budget level B20 = strategies with beta at most 0.2 (NONE, SH12, PH50_14, PH25_18); B40 = all six. B40 is primary. The attacker is not allowed to harden below the nominal dwell. Noise is an environment factor, not an attacker choice.

#### A12.3 Defenders (eight, all existing code except the two marked new)

All know the class prior and the noise law and see only dwell and kill flags; each is an exact function of the observable state.

| Name | Definition |
|---|---|
| DPMIS_1.0, CEP_1.0, ROLL_CEP_1.0 | Nominal planners of A6, built for scale 1.0 (no adaptation) |
| DPMIS_1.2 | Optimal policy of the model with shared scale 1.2 (constant insurance; new, same code with lam = 1.2) |
| QMDP_B | QMDP of A7 with prior B, uniform on {0.8, 1.0, 1.2} |
| QMDP_H | QMDP with a hardening-only prior, uniform on {1.0, 1.2, 1.4} (new prior, same code) |
| BA_H | Bayes-adaptive exact optimum for the prior {1.0, 1.2, 1.4}; a ceiling for shared strategies, not deployable |
| IID_DP | Optimal policy of the model in which every threat is independently hardened with q = 0.5 and lambda_h = 1.4 (new truth builder, merged classes, exact DP) |

#### A12.4 Quantities and decision rules (applied mechanically by `scripts/analyze_a12.py`)

Value of every defender under every strategy: exact expectation over the true class law by recursion over observable states (`eval_rec` of A7, equal to world enumeration in the tests). Reference V*(k): the DP optimum of the true model of strategy k (it knows q and lambda_h, and the scale in the shared case). Regret R(d, k) = 1 - sum V / sum V*(k), pooled over the 8 cells and the instances of the half in use. Exploitability E(d) = max over strategies of R(d, k) at the stated budget. Stackelberg and minimax coincide in this zero-sum regret game, so the minimax mixture is computed by the linear program minimise v subject to sum_d x_d R(d, k) <= v for every k, x a probability vector. A mixture means choosing one defender at random before the engagement, which only makes sense if the attacker cannot see the choice.

Split sample: instances with even index are used to solve the mixture, instances with odd index to evaluate it; the decision uses the odd half (the pure defenders are also evaluated on the odd half). Instance-level paired bootstrap, 2000 resamples, seed 20261230, one-sided 95% bound; every selection (best nominal defender, best adaptive defender) and the linear program are recomputed in each resample.

Let E_nom = min over {DPMIS_1.0, CEP_1.0, ROLL_CEP_1.0, DPMIS_1.2} of E(d) at B40 (the best non-adaptive defender), E_def = min of E over the four adaptive defenders (QMDP_B, QMDP_H, BA_H, IID_DP) and the minimax mixture.

- X (exploitable): E_nom at least 8% and its lower bound at least 8%. This is expected to hold from A6 (best nominal planner 14.8% at +40%, noise +-1) and is not a new finding; it is stated so that the decision does not rest on it.
- R (a defence exists): 1 - E_def / E_nom at least 0.30 and its 5th percentile at least 0.20.
- S (adaptation is exploitable): in at least one partial strategy (PH50_14, PH25_18, PH50_18) the best of QMDP_B, QMDP_H and BA_H has a regret above that of the best non-adaptive defender by at least 1 point, lower bound above 0.
- N (noise): X and R both hold in the noise +-2 run (offsets -2 to 2, binomial 1, 4, 6, 4, 1 over 16), thresholds unchanged.

| Outcome | Rule, in this order | Action |
|---|---|---|
| STOP | not X | The attack does not cost the best non-adaptive defender enough. Direction dropped |
| EXPOSED | X, not R | No defender in the set removes the loss. Report the exploitability; any new defender needs a new amendment before any run |
| REGIME-LIMITED | X, R, not N | Effect needs low within-class noise. Narrow the claim; the noise level must be justified from the literature before the decision |
| ADAPTATION-SUFFICES | X, R, N, not S | Exploitability plus QMDP defence. Moderate tier |
| STRONG-ADVERSARIAL | X, R, N, S | Candidate for the paper spine (with A11 closed). Before that: frozen confirmatory protocol with fresh seeds, a check at n = 5, literature check |

Reported without a rule: B20 results; per-strategy regrets of every defender; the attacker's best response to each pure defender and to the mixture; mixture weights and the value of the mixture on the even and odd halves; the maximum over cell and strategy (attacker also chooses the cell); expected regret of the nominal planners under the iid scale distributions of A7 for continuity.

#### A12.5 Runs

| Run | Content | Instances per cell (8 cells) | Master seed |
|---|---|---|---|
| `a12_audit` | all strategies and defenders, exploratory | 20 | 1 |
| `a12_main` | noise +-1, decisions X, R, S | 200 | 20261228 |
| `a12_noise2` | noise +-2, decision N | 100 | 20261229 |

The audit uses the exploratory seed 1 (as in A11, where the plan text named a different audit seed and the code used seed 1). The cost is measured in the audit; the A11 estimates were wrong in both directions, so no time is promised here. If the measured audit time predicts more than 6 hours for the two main runs, reduce in this order and record it before the run: `a12_main` to 100 instances per cell, then `a12_noise2` to 50.

#### A12.6 Audit, tests and gate (before any main run)

Hard gates, same mechanism as A11 (`scripts/a12_gate.py`; the main run refuses to start without the gate file for the same code hash): all tests pass; all mutations are caught; the audit has no violation.

Tests, with an independent brute force on small instances: (1) NONE reproduces the A7 nominal values on the same instances; (2) q = 1 reproduces the shared-scale truths of A6 and A7 and q = 0 reproduces NONE; (3) the true-model DP equals brute force for partial hardening; (4) the value of every defender lies between 0 and V*(k); (5) the Bayes-adaptive value lies at or above QMDP on average and at or above the nominal planners under a matched strategy; (6) the linear program equals a known solution on small games (matching pennies, a dominated strategy, a game with a pure saddle) and a brute-force enumeration on random small matrices; (7) the even-odd split never mixes halves; (8) the budget filter keeps exactly the strategies of A12.2. Mutations that must be caught: lambda_h applied to the wrong class, q and 1 - q swapped, noise offsets shifted, regret denominator taken from another strategy, constraint sign in the linear program, even and odd halves swapped, budget filter off.

Audit invariants (all hard): regrets within [0, 1]; the value of the extracted optimal policy equals the DP value; no defender above V*(k) by more than 1e-9.

#### A12.7 Disclosure

- Known before this plan: A6 (best nominal planner 14.0% and 14.8% at -40% and +40%, noise +-1), A7 (shared-scale adaptive planners recover about half at SD16 and almost nothing at noise +-2) and the exploratory `a7_h2` smoke (16 instances: nominal DPMIS 4.2%, shared-scale adaptive planners 7.1 to 10.3% when every threat has its own scale). Those numbers motivated the partial strategies and the IID_DP defender. X will therefore very likely hold; the open questions are R, S and N.
- My expectation, not a result [OPEN]: REGIME-LIMITED is the most plausible outcome, because A7 and A10 show that the loss falls with within-class noise.
- Literature: two extended web searches (Stackelberg and adaptive-adversary models of drone-swarm and laser defence) found general Stackelberg security work on drone attacks but no dwell-based directed-energy game; attacker-composition models for directed-energy defence (Gaver and Jacobs 2013, Hock and Weinberg 2025) are known only second-hand from the literature document and are unread. Novelty is not established [OPEN].

#### A12.8 Limits

n = 4, two base classes, one D, one effector; a finite attacker set of six strategies chosen by me, so exploitability is a maximum within that set and not over all attackers; independent per-threat hardening (no value-targeted or label-targeted hardening); noise is not an attacker choice; the defender set is finite and the minimax defender is a mixture over it, not a new policy class; no learned defender (B0 gave no case for learning at this scale); hardening levels, budgets, the 8%, 30%, 20% and 1-point thresholds and the 6-hour limit are mine and are the author's to confirm before the freeze; realistic hardening and within-class noise are unknown (Burley is atmospheric only).

#### A12.9 Freeze

This text is committed before any A12 code exists. Code, configs, tests, audit and gate follow on `main`. Any change after the freeze is a dated sub-amendment committed before the run it concerns.
