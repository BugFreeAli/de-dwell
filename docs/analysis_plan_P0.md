# Analysis plan for pilot P0 (stylized model M0, exact DP)

- Project: Research Project 2, learning versus index policies for dwell-based directed-energy scheduling
- Plan version: v1, dated 2026-11-08 (written before any confirmatory run)
- Frozen by: git tag `p0-plan-v1` (see Section 11). Any later change is a dated amendment in Section 12; the text above it is never edited.
- Source of the questions: `docs/literature_review_2_2026-11-07.pdf`, Sections 9 and 10 (provenance tags used there: [DOC] [OPEN] [CODE] [MEM]).

## 1. Purpose and decision

P0 is a design-stage gate, not a hypothesis test. It asks whether, in a small stylized model that can be solved exactly, the best simple scheduling rule loses a material share of the optimal value. The agreed go rule (decided by the author on 2026-11-07, [DOC] Section 10.1):

> GO if the best simple rule loses at least 5% of the DP-optimal value in at least one region of the regime grid.

This plan sharpens "region", "best simple rule" and "loses" so the rule can be applied mechanically (Section 6), and adds a companion test that separates loss caused by hardness uncertainty from loss that exists even when the class is known (Section 6, item 5). The companion test and the S1 control (Section 7) were added after an exploratory look (Section 8).

Hypotheses touched by P0: H1 (slack, small slew, informative prior: regret below 5%) and H2 (regret exceeds 5% somewhere; growth with load, slew and prior uncertainty is reported descriptively only). H3 and H4 need the simulator and RL and are not tested here.

## 2. Model M0 (final specification)

All times are integers on a grid of step 1.

- One effector, n = 5 threats. Threat i has value v_i, a breach deadline d_i and a hidden class c_i in {soft, hard} (K = 2). Class k needs D_k steps of accumulated dwell: D = (2, 5).
- Decision: "dwell on threat j for one more step". Engaging a threat different from the current one costs a slew of s steps first (uniform, independent of distance). The first engagement of an episode is free of slew. After a kill the next engagement of a different target pays the slew (the effector still points at the dead target). No information arrives during a slew, so "slew then dwell one step" is one atomic action (exact).
- Progress is retained when the effector leaves a threat (no cooling or decay). Kills are deterministic given the class: threat j dies when its dwell reaches D of its class. A kill completing at time t' counts only if t' <= d_j.
- Observation: for each threat, dead or alive and the dwell received. The belief of an alive threat with dwell x is its prior restricted to classes with D_k > x, renormalised (a deterministic function of the observation).
- Objective: maximise the expected total value of threats killed before breach.
- State (t, cur, prog). prog[i] = dwell of an alive threat, or DEAD when killed or when no possible class can still finish before d_i (exact canonicalisation, merges states with identical value). Actions are restricted to threats that can still be killed before the deadline including the slew to reach them. The restriction is exact; `tests/test_dp_bruteforce.py` checks it against an unrestricted brute force with idling.
- Exact DP: memoised expectimax over (t, cur, prog). No discretisation error inside M0; the grid step is part of the model definition. Sensitivity to the time resolution is checked in the optional run S4.

## 3. Instances and regime variables

Instance generator `src/dwell/instances.py`; base random numbers depend only on (master_seed, instance index) so cells share the same underlying draws (common random numbers).

- Values: uniform on {1, 2, 3}, independent (main run). All equal to 1 in control S1.
- Label: each threat gets a class label, Bernoulli(0.5) hard. Label accuracy a: P(true class = label) = a. The prior of a threat is (1-a, a) if labelled hard and (a, 1-a) if labelled soft. a = 0.5 is uninformative, a = 1 reveals the class.
- Deadlines: d_max = max(2, round(sum_i E_i / rho)) with E_i the prior mean dwell of threat i; each other threat gets d_i = max(2, round(u_i d_max)), u_i ~ U(0.35, 1); one random threat gets d_max exactly. `round` is Python's (half to even).
- Regime grid (main run): rho in {0.5, 0.8, 1.1, 1.5, 2.0}; s in {0, 1, 2, 4}; a in {0.5, 0.7, 0.8, 0.9, 1.0}; 100 cells; 500 instances per cell; master seed 20261108.
- Regime variables, defined per instance and stored:
  - load ratio rho_hat = sum_i E_i / d_max (expected total dwell over time available before the last deadline; slew excluded). Nominal rho is the grid value.
  - rho_dbf = max over deadlines t of (sum of E_i with d_i <= t) / t (demand-bound load, sees early-deadline congestion). Stored for the regime predictor, not used in the go rule.
  - slew ratio sigma_hat = s / mean_i E_i.
  - information I(a) = 1 - H2(a) bits per threat (0 at a = 0.5, 1 at a = 1).

## 4. Policies

All simple rules and the DP see only the observable state and the prior. All simple rules choose only among threats that can still be killed before the deadline (same filter as the DP). Ties: keep the current target, otherwise the lowest index (a candidate must beat the incumbent by more than 1e-12).

For an alive threat j: x = dwell received, sl = slew to j from the current target, cap = d_j - t - sl, posterior (p, r) over remaining dwell r = D_k - x, E[r] = sum p r. Let ratio(tau) = v_j * P(r <= min(tau, cap)) / (sl + E[min(r, tau, cap)]).

| Rule | Engage the threat that ... | Origin and caveat |
|---|---|---|
| FAIR | has received the least dwell | adapted from the fair-allocation result of Chen et al. 2026 (memoryless shots, no slew); expected to be weak when s > 0 |
| EDF | has the earliest deadline | classical, value-blind |
| LLF | has the least expected laxity d_j - t - sl - E[r] | classical with expected remaining dwell, value-blind |
| SE | has the smallest sl + E[r] (shortest engagement) | Taylor 2021, value-blind |
| GMR | maximises ratio(infinity): value times kill probability per unit expected time, dedicated attempt | greedy marginal return, adaptation |
| BI | maximises max over tau in {r <= cap} union {cap} of ratio(tau) | Gittins-type belief index ADAPTED to deterministic dwell thresholds; it is not the index of Glazebrook et al. (memoryless shots) [OPEN] |

- DP optimum: the exact value above. "Information-matched" means it uses the same prior as the rules.
- Class-aware oracle: knows all classes (upper bound, not implementable, not a competitor). Value = prior-weighted mean over class vectors of the best deterministic subset schedule (EDF order inside the subset). It is used for the information gap only and never in the go rule.
- Exact evaluation: the value of every rule is computed by enumerating all 2^5 class vectors and weighting by the prior, so there is no simulation noise. The only sampling error is the choice of instances.

## 5. Metrics

For a cell with instances i = 1..N:

- regret_r = 1 - sum_i V_r,i / sum_i V_dp,i (ratio of sums, so instances with small optimal value do not dominate). Reported as a percentage.
- best simple rule: the rule with the smallest regret in the cell, chosen on the same data. This selection favours the rules and therefore makes the go rule harder to satisfy (conservative).
- Secondary: regret normalised by total value; share of instances where no simple rule reaches V_dp; information gap 1 - sum V_dp / sum V_orc.
- Companion excess regret at (rho, s, a < 1): regret_best(a) - regret_best(a = 1) in percentage points; the a = 1 cell has the same instance indices, so the bootstrap is paired.

## 6. Decision procedure (applied mechanically by `scripts/analyze_p0.py`)

1. Per cell, instance-level percentile bootstrap (B = 2000, seed 20261109, resampling instances; the best rule is recomputed inside every resample).
2. A cell is flagged if the one-sided 95% lower bound of the best-rule regret is at least 0.05. A cell is point-flagged if the point estimate is at least 0.05.
3. A region is two cells that are face-adjacent on the (rho, s, a) grid.
4. Decision: GO if there is a region of flagged cells. MARGINAL if there is no GO but a region of point-flagged cells. NO-GO otherwise.
5. Information attribution (reported with every GO): a cell is excess-flagged if the one-sided 95% lower bound of its excess regret (paired bootstrap, seed 20261110) is at least 2 percentage points. If a region contains cells that are both flagged and excess-flagged, the loss is "information-sensitive"; otherwise it is "structural only". The 2-point effect size is a judgment [OPEN], set before the confirmatory run.
6. H1 check (descriptive): cells rho = 0.5, s <= 1, a >= 0.9. H1 is supported only if the upper 97.5% bootstrap bound of the best-rule regret is below 5% in all of them ("not significant" is never read as "equivalent").

Actions after the decision:

| Outcome | Action |
|---|---|
| GO, information-sensitive | proceed to step 2 (simulator) as planned |
| GO, structural only | report to the author before step 2. The loss comes from deadline packing and slew rather than from hardness uncertainty, which weakens G2 as the paper's centre. Decide together between keeping candidate A with a "structure plus uncertainty" framing and a question about deadline-constrained dwell scheduling |
| MARGINAL | staged extension, pre-specified: re-run the main grid with 1000 instances (`n_instances` = 1000, same seed, instances 0 to 499 identical), analyse the extended data, disclose the extension |
| NO-GO | follow Section 9 of the literature document: switch to the "when rules suffice" framing (candidate C) or add structure |

## 7. Runs

| Run | Config | Role |
|---|---|---|
| main | `configs/p0_main.json` | confirmatory for the go rule |
| S1 | `configs/p0_s1_homog.json` | control, required before interpreting the main run: all values equal; separates value packing from hardness uncertainty. Descriptive; no separate decision |
| S2 | `configs/p0_s2_n6.json` | optional: n = 6 (rho = 0.5 omitted: about 5 s per instance there) |
| S3 | `configs/p0_s3_ratio.json` | optional: hardness ratio D = (2, 8) |
| S4 | `configs/p0_s4a_coarse_n3.json`, `p0_s4b_fine_n3.json` | optional: time-resolution check (grid refined by 2, slews doubled), compared cell by cell |
| smoke | `configs/p0_smoke.json` | pipeline test only, exploratory seed 1, never used for a decision |

Secondary runs are analysed with the same script and reported as sensitivity, with the same labelling rules. Only the main run decides.

## 8. Exploratory work before this plan (disclosure)

- Exploratory seed 1 (never reused for the main run) was used for debugging, timing and a pipeline test: 20 instances in each of 12 cells (n = 5, rho in {0.5, 1.1, 2.0}, s in {0, 2}, a in {0.5, 1.0}). Timings were also taken with seed 1 on further cells.
- What I looked at in that run: the invariants (all held for 240 instances) and the regret table. Best-rule regret was near 0 at rho = 0.5 and between about 3% and 12% at rho >= 1.1, including cells with a = 1 where the class is known. Excess regret relative to a = 1 ranged from about -5.5 to +3.4 points with wide intervals (N = 20).
- Consequences for the plan: the companion information-attribution test (Section 6, item 5) and the S1 control were added because the 5% go rule alone can be met by value packing and deadlines without any hardness uncertainty. The grid ranges and the 5% rule were not changed on the basis of these numbers. The 2-point threshold was chosen after seeing the exploratory excess values and was not tuned further.
- The exploratory regrets are N = 20 and are not reported as results.

## 9. Integrity rules

- Every instance of every cell is kept. The analysis script stops if a cell has fewer than `n_instances` rows. No outlier removal.
- The run stops with an error if any invariant fails: V_orc >= V_dp >= V_rule >= 0 and the exact value of the extracted DP policy equals V_dp (tolerance 1e-9).
- Statistics are recomputed from the raw per-instance CSVs by the script; no remembered numbers.
- P0 is a design gate. No p-values are reported. The adjacency requirement reduces, but does not remove, selection over up to 100 cells; confirmatory claims in the paper (H1 to H4) use the later frozen plan with Holm correction.
- Post-hoc analyses and staged extensions are labelled as such.

## 10. Limitations of M0 (stated, not hidden)

- Retained progress favours preemption and probing: with cooling or decay, leaving a threat would cost progress. The simulator (step 2) must model this and a sensitivity run is needed.
- Within a class the required dwell is deterministic; K = 2 classes only. More classes or within-class variation are not covered.
- Slew is uniform; kill assessment is a plain dead/alive flag; one effector; independent classes; no thermal or energy limit.
- Values and deadlines are synthetic; regimes are nominal grids, not calibrated to any system.
- The BI rule is an adaptation, not the published index. The rule list is not exhaustive; "best simple rule" means best among these six.

## 11. Reproducibility and freeze

- Code, configs and this plan are committed; the tag `p0-plan-v1` marks the frozen state before the first confirmatory run. Every run records the git commit, Python, numpy and pandas versions, platform, CPU count, seed and config in `run_meta_*.json`.
- Raw per-instance CSVs are archived (tar.gz with SHA-256) and referenced in the run record.

## 12. Amendments

### A1 (2026-11-09): certainty-equivalent planner and value-spread runs (post-hoc)

Written before any A1 run. Everything under A1 is post-hoc and is reported as such.

Reason. The main run (tag `p0-main-run1`, commit ef13c2a) gave GO with attribution "structural only": 59 of 100 cells flagged, none information-flagged, maximum best-rule regret 12.3%. The S1 control gave GO "information-sensitive": 38 flagged, 23 information-flagged. None of the six rules plans for packing, so the main-run loss may only show that a planning rule is missing.

Exploratory disclosure. After the main run I tested the planner below on exploratory seed 1 (20 instances in each of 12 cells). Its mean regret was 2.1%, against 9.5% for GMR; the best-rule decision on that sample was NO-GO. This is N = 20 and is not a result. It is the reason A1 is run before any simulator work.

Additions:

1. Rule CEP (certainty-equivalent planner). Replace the remaining dwell of every alive threat by its expected value under the belief. Enumerate every ordered sequence of useful threats; a sequence is feasible if every threat finishes by its deadline with these durations (slews as in M0). Choose the feasible sequence with the largest total value (ties: earlier finish, then the sequence that starts with the current target) and dwell on its first threat. Re-plan at every step. If no sequence is feasible, use GMR for that step. CEP is a cheap planning baseline, not the strong optimiser of step 3 (receding horizon over sampled scenarios).
2. Same-instance augmentation: CEP is evaluated on the identical instances of the main and S1 runs (`scripts/run_a1.py` rebuilds every instance and stops if it differs from the stored one). The analysis applies the rules of Section 6 unchanged to seven rules (six plus CEP). Seeds and thresholds are unchanged.
3. Value-spread runs: value sets {1, 2} (`a1_v2`) and {1, 3, 9} (`a1_v4`), master seed 20261108, 300 instances, grid rho in {0.8, 1.1, 1.5}, s in {0, 2, 4}, a in {0.5, 0.8, 1.0}. They are not paired with the main instances (the value draw changes the random stream). With the main run ({1, 2, 3}) and S1 ({1}) the coefficients of variation of the values are 0.79, 0.41, 0.33 and 0.
4. Hardness-ratio run `a1_ratio`: D = (2, 8), values {1, 2, 3}, same reduced grid, 300 instances (S3 of Section 7 with CEP and the reduced grid).
5. Summary: `scripts/summarize_a1.py` on the 27 common cells.

Interpretation, fixed before the run:

| A1 main result (seven rules, Section 6 procedure) | Meaning | Action |
|---|---|---|
| GO | The gap survives a planning baseline in M0 | Continue; add value spread as a regime variable; strong planners are still required in step 3 |
| MARGINAL or NO-GO | CEP closes the gap in the tested M0 regimes; "simple rules lose" is not supported in M0 | No step 2 yet. Decide with the author between (i) a "when planning suffices" framing (candidate C) and (ii) adding structure that breaks certainty-equivalent planning (larger hardness ratio, more classes, noisy kill assessment, progress loss on leaving), tested first in M0 as A2 |

No new thresholds are introduced. The results of the value-spread and hardness-ratio runs are descriptive.

Limits: post-hoc; CEP is one planner; dwell is deterministic within a class; the findings hold for M0 only.

### A2 (2026-11-09): structures that may break certainty-equivalent planning (post-hoc)

Written before any A2 run. Everything under A2 is post-hoc and is reported as such.

A1 outcome (from the run logs and decision files). Main with CEP: GO by the Section 6 rule, but only 4 of 100 cells flagged, all with load 1.5 or 2.0 and slew 0; maximum best-rule regret 9.0%; on the 27 common cells the mean best-of-six regret was 7.8% and the mean best-of-seven regret 2.1%. S1 with CEP: GO, 11 flagged cells, maximum 10.0%. Value sets {1, 2} and {1, 3, 9}: NO-GO (0 and 1 flagged cells, maxima 4.9% and 5.7%). D = (2, 8): GO, 5 of 27 flagged, maximum 7.5%. The "information-sensitive" label of the companion test is not informative for runs with CEP, because CEP is exactly optimal when the classes are known (tested), so the excess regret equals the CEP regret. That test is not used in A2.

Reason. In M0 a cheap planner leaves little room for a learned policy outside a corner (load of at least 1.5, no slew). A2 asks whether any structure that the DP can still solve exactly makes the planner lose clearly in a region that is not a corner.

Exploratory disclosure. After writing the code I ran the K = 3 variant on exploratory seed 1 (10 instances in each of 8 cells, grid reduced). Mean regrets were 6.5% for CEP and 7.7% for LA1. This is N = 10 and is not a result.

Additions:

1. Model generalisation (`src/dwell/m2.py`): K classes (K = 2 reproduces M0 exactly when kappa = 1; tested) and a progress-retention factor kappa: when the effector leaves an alive threat, its dwell becomes floor(kappa x); the evidence that classes are ruled out is kept. kappa = 1 is M0. Exactness is checked against an independent brute force that tracks the evidence explicitly.
2. K-class instances (`make_instance_k`): the label picks one of K classes with probability a, the others share 1 - a equally. For K = 2 the instances equal those of M0.
3. Rule changes: CEP and the plan value are made decay-aware (when kappa < 1 and the plan leaves the current threat first, the current threat's duration is computed from its decayed dwell). For M0 (kappa = 1) CEP results are unchanged; verified on the exploratory smoke data. New rule LA1: one-step lookahead; for every useful action take the expectation over the outcomes of the reward plus the value of the best CEP plan from the next state, and choose the best. LA1 equals the DP optimum when classes are known (tested). It is not guaranteed to beat CEP.
4. Variants, all with n = 5, values {1, 2, 3}, master seed 20261108, 300 instances per cell, grid rho in {0.8, 1.1, 1.5, 2.0}, s in {0, 2, 4}, a in {0.5, 0.7, 0.9} (36 cells; a = 1 is omitted because CEP and LA1 are exact there):
   - `a2_wide`: K = 2, D = (1, 8), kappa = 1.
   - `a2_k3`: K = 3, D = (1, 3, 9), kappa = 1.
   - `a2_decay05`: K = 2, D = (2, 5), kappa = 0.5.
   - `a2_decay0`: K = 2, D = (2, 5), kappa = 0.
   - Reference: `a2_main_la1`, the A1 main run (n = 5, D = (2, 5), kappa = 1) with LA1 added on the identical instances; only its 36 cells on the A2 grid are used in the comparison.
   The decay variants are robustness checks of the assumption that progress is retained; they are expected to shrink the gap, not to widen it.
5. All eight rules (six old, CEP, LA1) enter the best-rule selection. Section 6 procedure (bootstrap, flagged cell) applies unchanged per variant.

Criterion, fixed before the run. A variant shows a non-corner gap if two face-adjacent cells are both flagged (one-sided 95% lower bound of the best-rule regret at least 5%) and both have s >= 2 and rho <= 1.5. `scripts/summarize_a2.py` applies it.

| Result | Action |
|---|---|
| a2_wide or a2_k3 shows a non-corner gap | That structure (hardness spread or number of classes) becomes a core axis of the simulator specification; the novelty claim is limited to it; strong planners (step 3) are still required |
| No variant shows a non-corner gap | M0-type structure is exhausted: no step 2 as planned; decide with the author between a "when planning suffices" paper (candidate C) and a different source of structure (for example noisy kill assessment, several effectors) |
| a2_decay05 or a2_decay0 removes the corner gap of the reference | Report that the corner gap of M0 depends on full retention of progress |

No new thresholds. Limits: post-hoc; n = 5 and two or three classes only; deterministic dwell within a class; CEP and LA1 are cheap planners, not the strong optimiser of step 3.

### A3 (2026-11-09): exact rollout baselines (post-hoc)

Written before any A3 run. Everything under A3 is post-hoc and is reported as such.

A2 outcome (from the run logs and decision files; best of eight rules, 36-cell grid, n = 5). Reference (D = (2, 5), full retention): mean best-rule regret 3.46%, maximum 8.98%, 3 flagged cells, none outside the corner. a2_wide (D = (1, 8)): 4.70%, 9.15%, 13 flagged, 9 non-corner. a2_k3 (D = (1, 3, 9)): 5.12%, 8.41%, 18 flagged, 11 non-corner. Decay with kappa = 0.5 and kappa = 0: 3.65% and 3.76%, maxima 9.47% and 9.50%, 4 and 6 flagged cells, none non-corner. Mean CEP regret: 3.70% (reference), 6.06% (wide), 6.72% (k3), 3.96% and 4.13% (decay). Mean LA1 regret: 8.10%, 7.55%, 8.87%, 8.45%, 8.50%. By the A2 table, a2_wide and a2_k3 show a non-corner gap. In these two variants the loss is concentrated at label accuracy 0.5 and 0.7 and is below 5% at 0.9.

Reason. CEP and LA1 are cheap planners. A planner with access to the model and more computation could close the remaining gap. A3 tests this before any simulator work.

Exploratory disclosure. After writing the code I ran the K = 3 variant on exploratory seed 1 (10 instances in each of 8 cells, grid reduced). Mean regrets were 6.0% for CEP and 2.2% for the CEP rollout defined below, 4.2% for the GMR rollout and 5.3% for the EDF rollout. This is N = 40 and is not a result. It suggests that a rollout may close much of the gap.

Additions:

1. Rules ROLL_CEP, ROLL_GMR, ROLL_EDF (`src/dwell/rules.py`): exact one-step policy improvement of a base rule. For every useful action take the expectation over the chance outcomes of the reward plus the exact expected value of following the base rule from the next state (memoised evaluation of the base rule over the reachable states); choose the best action, ties keep the current target. By the policy-improvement theorem the rollout never has a lower value than its base rule; this is checked on every instance. It is optimal when the base rule is optimal (tested for CEP with known classes).
2. The rollouts use the exact model (transitions and belief), so they are strong references, not deployable baselines at scale. The time to evaluate each policy on an instance is stored (`t_*` columns) and reported as a descriptive indication of decision cost.
3. Same-instance augmentation (`scripts/run_a3.py`) of the stored A2 data of the reference (`a3_main`, restricted to the 36 A2 cells, 500 instances), `a3_wide` and `a3_k3`. The script rebuilds each instance, stops if it differs from the stored one, recomputes CEP and stops if it differs from the stored value, and stops if a rollout is below its base rule or above the DP value.
4. Criterion, unchanged from A2: a non-corner gap is two face-adjacent flagged cells (one-sided 95% lower bound of the best-rule regret at least 5%), both with s >= 2 and rho <= 1.5; the best rule is now the best of eleven rules (six old, CEP, LA1, three rollouts). `scripts/summarize_a3.py` applies it.

| Result on a3_wide and a3_k3 | Action |
|---|---|
| At least one still shows a non-corner gap | The gap survives a strong planner. Step 2 starts with that structure as a core regime axis |
| Neither shows a non-corner gap | Planning with model access closes the gap in M0-type structure. No step 2 as planned. Decide with the author between a paper on learning versus planning at equal decision time and at sizes where planning is infeasible, and the "when planning suffices" framing (candidate C) |

No new thresholds. Limits: post-hoc; n = 5; at most three classes; deterministic dwell within a class; the rollouts need the exact model.

### A4 (2026-11-09): prior misspecification in M0 (post-hoc)

Written before any A4 run. Everything under A4 is post-hoc and is reported as such.

A3 outcome (from the run logs and decision files). Exact rollouts removed the gap: best of eleven rules, mean regret 0.83% (reference), 1.12% (D = (1, 8)), 1.57% (three classes), maxima 2.95%, 2.95%, 3.79%, no flagged cell, NO-GO in all three. The CEP rollout alone: 1.11%, 1.54%, 2.29%, at about 3 to 7 times the evaluation time of CEP. By the A3 table, step 2 is not started as planned.

Reason. The rollouts and the DP use the exact model, including the label accuracy. Learning could still matter if the model is wrong. A4 measures, with the exact tools, how much a planner built on a wrong label accuracy loses against the optimum that knows the true accuracy.

Exploratory disclosure. After writing the code I ran A4 on exploratory seed 1 for the three-class variant (12 instances in each of 8 cells: load 1.1 and 1.5, slew 2 and 4, true accuracy 0.7 and 0.9). Mean regret of the best method was about 0.2% when the believed accuracy was 0.9 or 0.99 and the true one 0.7, 1.8% for belief 0.5 at true 0.7, and 4.6% for belief 0.5 at true 0.9. This is N = 12 and is not a result. It suggests that losses are small except when the labels are ignored.

Design:

1. True model: K classes, label accuracy a_true in {0.7, 0.9}, instances as in A2 (`make_instance_k`), n = 5, values {1, 2, 3}, full retention, master seed 20261108, 300 instances per cell, load in {0.8, 1.1, 1.5, 2.0}, slew in {2, 4}. Slew 0 is left out: it is the corner by definition and the most expensive cell.
2. Believed model: the same instance with a believed accuracy a_b in {0.5, 0.7, 0.9, 0.99} (labels and deadlines unchanged). Every method is built from the believed model and then run in the true model: the value is the exact expected value under the true prior. a_b = a_true is the correct-belief reference. a_b = 0.5 is a planner that ignores the labels.
3. Methods: DPMIS (the optimal policy for the believed model), CEP, ROLL_CEP, ROLL_GMR, GMR, EDF. Regret is measured against the DP optimum of the true model (ratio of sums over the instances of a cell).
4. A cell with a_b different from a_true is flagged if the one-sided 95% lower bound (instance bootstrap, 2000 resamples, seed 20261111, best method recomputed inside every resample) of the best method's regret is at least 5%. The correct-belief cells are reported but never flagged.
5. Criterion, fixed before the run. A variant shows a non-corner misspecification gap if two cells with the same a_true are face-adjacent on the (load, slew, believed accuracy) grid, both flagged, both with load at most 1.5. Slew is at least 2 everywhere. `scripts/analyze_a4.py` applies it.
6. Runs: `a4_k3` (D = (1, 3, 9)) and `a4_wide` (D = (1, 8)).

| Result | Action |
|---|---|
| Gap in a4_k3 or a4_wide | A wrong prior costs the best planner at least 5%. Learning has an exact motivation. Next amendment A5: a Bayes-adaptive baseline (unknown accuracy, learned from the outcomes inside an episode), then step 2 |
| No gap in either | Planners are robust to a wrong label accuracy in M0-type structure. No case for learning here. Recommend the "planning suffices" paper (candidate C); step 2 is not started as planned |

No new thresholds. Limits: post-hoc; only the label accuracy is misspecified, not the physics; the believed accuracy is fixed during an episode (a planner that adapts is not tested here); n = 5; at most three classes.

### A5 and B0 (2026-11-10): in-dwell sensing and a learned policy against planners (post-hoc)

Written before any A5 or B0 run. Everything under A5 and B0 is post-hoc and is reported as such.

A4 outcome (from the run reports). Three classes (1, 3, 9): no non-corner gap, mean best-method regret under a wrong label accuracy 1.0%, maximum 6.6% (one flagged cell, load 2.0). Hardness (1, 8): the criterion is met (5 flagged cells, 3 outside the corner, maximum 14.6%), but every cell with a regret of at least 5% has the believed accuracy 0.5, a planner that ignores the labels; beliefs 0.7, 0.9 and 0.99 cost at most 0.84% on average. The criterion was too permissive; the substantive reading is that plausible errors in the label accuracy give no strong case for learning.

Decision of the author on 2026-11-10: pursue both a sensing direction (A5) and a first stage of learning versus planning (B0).

**A5: noisy in-dwell feedback.**

Reason. The gap between the DP optimum and the class-aware oracle (6 to 10% at label accuracy 0.5 to 0.7) cannot be closed by scheduling. It can only be reduced by information. A5 asks how much a noisy signal during dwell is worth and whether belief-based planners can use it.

Exploratory disclosure. I ran the model on exploratory seed 1 before fixing the design. With D = (3, 8), n = 5, load 1.1, slew 2, label accuracy 0.5 (10 instances), the exact optimum gained 0.34%, 0.83% and 1.08% over q = 0.5 for signal quality q = 0.7, 0.9 and 0.99 (oracle gap 7.8%). With D = (5, 12), n = 4 (6 instances, load 1.1 and 1.5) the gains were 4.3 to 4.5% (q = 0.9) and 4.8 to 5.7% (q = 0.99), with oracle gaps of 7.3 and 14.0%. A smoke run (15 instances, 4 informative cells, D = (3, 8)) showed gains of about 0.7 to 0.9%. These samples are small and are not results. The long informative window (D1 = 5) was chosen as the primary variant because of them.

Model (`src/dwell/m3.py`, `Model3`): two classes. After every dwell step that does not kill, the sensor reports one binary signal that is correct with probability q (hard-looking with probability q for a hard threat and 1 - q for a soft one). q = 0.5 is no information and reproduces M0 exactly (tested). The belief is a function of the dwell and the number of hard-looking signals; once the soft class is ruled out the class is known. The DP is exact; it equals an independent brute force with explicit signal likelihoods on tiny cases (tested). Policy values are computed by exact recursion over the observable state.

Runs: `a5_long` (primary): D = (5, 12), n = 4, values {1, 2, 3}, 200 instances, load in {1.1, 1.5}, slew in {2, 4}, label accuracy in {0.5, 0.7, 0.9}, q in {0.5, 0.9, 0.99}. `a5_sense` (secondary): D = (3, 8), n = 5, 300 instances, load in {0.8, 1.1, 1.5}, slew in {2, 4}, label accuracy in {0.5, 0.7, 0.9}, q in {0.5, 0.7, 0.9}. Master seed 20261108. Methods: CEP, ROLL_CEP, ROLL_GMR, GMR, EDF, with the belief updated by the signals.

Quantities (ratio of sums over the instances of a cell, instance bootstrap, 2000 resamples, seed 20261112, one-sided 95% lower bound):

- Sensing gain of a cell with q > 0.5: DP value at q divided by the DP value at q = 0.5 on the same instances, minus 1. This is the exact value of using the signal.
- Planner regret: 1 minus the value of the best method divided by the DP value at the same q; the best method is recomputed inside every resample.
- A cell is sensing-flagged if the lower bound of the gain is at least 5%, and planner-flagged if the lower bound of the best-method regret is at least 5%.
- A region is two face-adjacent flagged cells on the (load, slew, label accuracy, q) grid with load at most 1.5; slew is at least 2 everywhere.

| Result (applied per variant by `scripts/analyze_a5.py`) | Action |
|---|---|
| Sensing region and planner region | Strong case for sensing-aware learning: step 2 builds a simulator with kill assessment and tracking noise |
| Sensing region only | Sensing value is the contribution (a value-of-information analysis), not learning; path (a) with this extension |
| No sensing region | No new direction from sensing; path (a) |

**B0: learned policy against planners.**

Reason. First, cheap stage of the path in which learning is compared with planning under decision time, size and regime shift. The physics mismatch of that path needs the simulator and is not tested here.

Disclosure. The training procedure was checked on the training seed only (agreement with the teacher about 90% on 6000 samples). A pipeline smoke test with a tiny training set evaluated 6 instances of one cell and 3 instances of a larger-size cell with the old test seed 20261108; the test seed was then changed to 20261120, which has not been evaluated.

Method (`src/dwell/learned.py`, `scripts/run_b0.py`): a shared scorer (one hidden layer of 32 units) maps per-threat features (value, time to deadline, expected and extreme remaining dwell, kill probability, slack, slew, current-target flag, greedy index, time left, number of useful threats) to a score; the policy takes the best score, so it applies at every n. Training: imitation of the exact DP at n = 5 (D = (1, 8), load in {1.1, 1.5, 2.0}, slew in {2, 4}, label accuracy in {0.5, 0.7, 0.9}) with three DAgger rounds of 300 instances and 4 sampled worlds each, training seed 777001. The gradient is checked against finite differences (tested). Tests (seed 20261120): T1 n = 5 in distribution (12 cells, 100 instances, regret against the DP), T2 n = 5 with slews 1 and 6 outside the training range (8 cells, 60 instances), T3 n = 7, T4 n = 9 (8 cells, 60 and 40 instances), T5 n = 11 (4 cells, 20 instances). For n >= 7 there is no DP; the reference is ROLL_CEP and the quantity is the relative difference of values. Time per decision is measured on the most likely world.

| Result (fixed before the run) | Meaning |
|---|---|
| T2 regret of LEARNED minus regret of ROLL_CEP at most -1 point with the upper confidence limit below 0, or in T3 to T5 the relative value of LEARNED above ROLL_CEP by at least 1% with the lower limit above 0 | Learning beats planning in a region: strong; continue to the simulator for physics mismatch |
| Not the above, and in T1 regret(LEARNED) - regret(ROLL_CEP) at most 1 point, in T3 and T4 relative value at least -1%, and time per decision of LEARNED at most 0.2 of ROLL_CEP at n = 9 | Learning matches planning at far lower decision cost: a speed result of moderate novelty |
| Otherwise, in particular if regret(LEARNED) in T1 exceeds regret(CEP) | The learned policy is not competitive: path (a) |

No new thresholds beyond those written here. Limits: post-hoc; K = 2 and D = (1, 8) only; the learner is a small imitation scorer, not a reinforcement learning method; ROLL_CEP is a strong reference but not the optimum for n >= 7; timings are Python timings.

### A6 (2026-11-11): tolerance of the scheduler to dwell-model error and adaptation (post-hoc)

Written before any A6 run. Everything under A6 is post-hoc and is reported as such. A6 was prepared in the same chat as A5 and B0 after the author rejected the "planning suffices" paper (path (a)) and the simulator study (path (b)) and asked for a new question from the literature (literature document, Section 10.16).

A5 and B0 outcome (from the run reports). A5, D = (5, 12): mean value of the in-dwell signal 2.1% over non-corner cells, one cell passing the 5% bound (the "region" is that cell at two signal qualities), best planner regret at most 2.4%; D = (3, 8): no sensing value. B0: the imitation-learned scorer lost 2.8% at n = 5 against 2.0% for the CEP rollout and 6.7% for CEP, 4.0% against 0.9% with slews outside the training range, and was 1.4 to 3.2% below the CEP rollout at n = 7 to 11, at about 300 times lower time per decision at n = 9. By the pre-written tables, neither gives a case for learning over planning.

Literature status. Burley (AFIT 2012, DTIC ADA557011) was read only to about page 64 of about 120 and could not be fetched again by the author; its dwell-error tables are not read. The realistic range of dwell-time error is therefore not verified [OPEN]. A6 scans errors of 20% and 40% in both directions and reports the whole tolerance curve.

Exploratory disclosure. (1) A prototype of the observation-based evaluator on exploratory seed 1 (10 instances in each of 3 cells, true dwell (2, 5), label accuracy 0.7, n = 5): regret against the true DP of 3.2 to 9.4% for a believed dwell scale of 0.8 and 1.6 to 9.4% for 1.2, 14.7 to 19.8% for 0.6 and 11.7 to 23.0% for 1.4 (best planner of the believed model). (2) A smoke run of the final code on exploratory seed 1 with deterministic dwell (D = (5, 10), n = 4, 12 instances in each of 4 cells, load 1.1 and 1.5, slew 2): regret of the best planner 16.7% at -20% and 6.9% at +20%, 23.7% at -40% and 18.4% at +40%; the Bayes-adaptive policy recovered 91% and 96% of the loss of the nominal planner. Because one kill reveals the scale exactly when dwell is deterministic, this recovery is too easy, and within-class dwell noise was added to the design before freezing it. (3) A smoke run with noise (8 instances in each of 4 cells): 3.7 to 6.3% at +20% and -20%, 13.7 to 16.1% at +40% and -40%; recovery 70% and 74%. These samples are small and are not results.

Model. Two base classes (soft, hard) with nominal dwell (5, 10). A hidden dwell scale lambda is shared by all threats of an engagement: the dwell of a threat of base class c is round(lambda D0[c]) + e, where the noise e is -1, 0, +1 with probabilities 0.25, 0.5, 0.25 (a6_main) or always 0 (a6_det, secondary). Classes (base class, noise) are independent across threats with the label prior of M0. Deadlines, values, labels and the load are generated at lambda = 1 exactly as in M0, so a wrong scale changes the real load. Dwell is retained when the effector leaves a threat; slew, deadlines and the objective are those of M0. n = 4, values {1, 2, 3}.

Quantities (all values are exact expectations; ratio of sums over the instances of a cell; instance bootstrap of 2000 resamples, seed 20261116, one-sided 95% lower bound):

1. Planners with a wrong model. The planner believes a scale lambda_b in {0.6, 0.8, 1.0, 1.2, 1.4} while the truth is lambda = 1. Methods: DPMIS (the optimal policy of the believed model), CEP and ROLL_CEP built from the believed model. They are run in the true model by an observation-based evaluator: the planner sees only the dwell received and the kill flags, translates them into the state of its believed model (dwell beyond the largest believed dwell is clamped to the largest believed class) and acts; the value is the exact expectation over the true class vectors. Regret = 1 minus value divided by the optimum of the true model (the calibrated DP). The error is lambda_b - 1.
2. A cell and error with lower bound of the best-method regret at least 5% is flagged. A region is two face-adjacent flagged cells on the (load, slew, label accuracy) grid at the same error. FRAGILE_20: a region at an error of +20% or -20%. FRAGILE_40: a region at +40% or -40%.
3. Adaptation ceiling. The Bayes-adaptive optimum (exact DP over the observable history) knows the prior of lambda, uniform on {0.8, 1.0, 1.2}, and the noise law, and learns lambda from kills and from dwell without a kill. It is evaluated when the truth is lambda = 0.8 and 1.2, against planners built for the nominal scale 1.0 (relative errors +25% and -17%). Regret is measured against the calibrated DP of the true scale, pooled over all cells. Recovery = 1 - regret(Bayes-adaptive) / regret(best nominal planner), with a bootstrap interval over instances. The Bayes-adaptive policy is well specified here (the truth lies in its support); this is an upper bound for adaptation, not a deployable method.
4. Adaptation recovers if, for at least one true scale, the pooled regret of the best nominal planner is at least 5%, the recovery is at least 0.5 and its 5th percentile is at least 0.3.

Runs: `a6_main` (noise, K = 6 classes, 200 instances in each of 8 cells: load 1.1 and 1.5, slew 2 and 4, label accuracy 0.5 and 0.9, master seed 20261115) and `a6_det` (no noise, same grid, secondary: shows the effect of identifiability). Decisions are taken on a6_main.

| Result on a6_main (`scripts/analyze_a6.py`) | Meaning and action |
|---|---|
| FRAGILE_20 and adaptation recovers | Strong case for adaptive scheduling under dwell uncertainty. Next: a learned adaptive policy (A7) and a physics-based dwell model with realistic variability (simulator) |
| FRAGILE_20, no recovery | Mismatch is costly but not recoverable inside an engagement. The contribution is the tolerance and calibration analysis; moderate tier |
| FRAGILE_40 only | Moderate sensitivity; weak direction; report the tolerance curve and decide with the author |
| Neither | The stop rule is met: the direction is dropped and this is reported plainly |

No new thresholds. Limits: post-hoc; one dwell scale shared by all threats and a discrete scale grid; the Bayes-adaptive prior contains the truth and the noise law is known to it; dwell noise is symmetric and small; planners do not adapt in the tolerance part; n = 4; the realistic error range is not verified; a learned adaptive policy is not tested here.

#### A6 outcome (recorded 2026-11-12, from the author's run reports; this note changes no earlier text)

a6_main (noise, 8 cells, 200 instances each, master seed 20261115, commit 8c8dce8, conda-forge Python 3.12.16, numpy 2.5.4, pandas 3.0.6, Windows 11, 4 workers): mean regret of the best method by dwell error -40%, -20%, +20%, +40%: 14.0%, 5.5%, 4.8%, 14.8% (maxima 21.7%, 7.3%, 7.5%, 21.6%; flagged cells 8, 4, 2, 8 of 8). Regions at 20%: four pairs at -20% (all label accuracy 0.9) and one pair at +20% (load 1.5, slew 4); regions at 40% in all cells. Adaptation (pooled over cells): truth 0.8 (nominal error +25%): best nominal planner 4.76% (below the 5% bar), Bayes-adaptive 1.48%, recovery 0.69 (5th percentile 0.65); truth 1.2 (nominal error -17%): 6.25% and 2.09%, recovery 0.67 (0.63). By the pre-written table: FRAGILE_20 and adaptation recovers (the recovery criterion is met only through the scenario with truth 1.2, because the nominal regret is below 5% for truth 0.8). The mean regret at +-20% is close to the 5% bar.

a6_det (no noise): mean regret by error 20.7%, 16.9%, 9.2%, 20.3%; recovery 0.85 and 0.90 (nominal 9.1% and 22.7%, Bayes-adaptive 1.3% and 2.3%). Within-class dwell noise lowers both the fragility at 20% (from 9 to 17% down to about 5%) and the recovery (from 0.85 to 0.90 down to 0.67 to 0.69).

Reading [OPEN]: the strong case holds under the assumptions listed under Limits; in particular the Bayes-adaptive policy is an upper bound for adaptation, the realistic dwell error is unverified, and one scale is shared by all threats.



### A7 (2026-11-14): practical adaptation to dwell-model error (post-hoc)

Written before any A7 run. Everything under A7 is post-hoc and is reported as such. Prepared in a new chat after the A6 outcome. Tag for the frozen plan: `p0-a7-plan`.

#### A7.0 Literature input: Burley (AFIT 2012, DTIC ADA557011), read 2026-11-14

Source and read status [DOC]: Tables 10 to 15 (printed pages 107 to 112) and the Conclusions (printed pages 93 to 95), read from the PDF supplied by the author (file `Comparison_of_High_Energy_Laser_Expected_Dwell_Times_and_Probabil.pdf`, 132 PDF pages, printed page = PDF page - 16). The tables are images. The values were transcribed by eye into `docs/burley_tables_transcribed.csv` (144 rows: 3 sites, 2 paths, 4 season and time blocks, 6 atmosphere methods); 94 of 144 rows were confirmed by an independent OCR pass and the other rows were checked by eye only [CODE, FILE]. Nothing else of the thesis was re-read for this amendment.

What the tables give. For each atmosphere method (1976 US Standard, 50th percentile climatology, ground tab, and 48, 24 and 12 hour GFS forecasts) the predicted dwell time DT (s) and the RMSE (s) of the predicted dwell time against the dwell time computed from the sounding, for a 1.045 micrometre laser (ELLA) and a 1.31525 micrometre laser (COIL). Each row aggregates 25 days of one season and time of day at one site (printed page 37). Relative error is RMSE / DT with DT the predicted dwell of the same row; the control dwell is not tabulated [DOC, CODE: `scripts/burley_stats.py`].

Realistic range at 1.045 micrometre (144 rows) [CODE on transcribed FILE values]:

| Atmosphere method | Rows | Median | Quartiles | Maximum | Share above 20% | Share above 40% |
|---|---|---|---|---|---|---|
| GFS 48, 24, 12 h | 72 | 10% | 8 to 13% | 30% | 3% | 0% |
| Climatology and ground tab | 48 | 16% | 13 to 26% | 46% | 33% | 15% |
| 1976 US Standard | 24 | 19% | 16 to 27% | 31% | 46% | 0% |
| All six | 144 | 13% | 10 to 19% | 46% | 20% | 5% |

The COIL wavelength gives relative errors of 14 to 327% (median 48%); the thesis attributes its worse behaviour to thermal blooming from molecular absorption and states that the move to solid state lasers reduces the impact of a dynamic atmosphere (printed page 95) [DOC]. COIL is not used for calibration here.

Reading for A6 and A7 [OPEN, my reading]: a dwell prediction error of about 10% (forecast-based atmosphere) to about 16% (climatology or ground tab) in the median, with 20% as an upper-typical value and 40% as a tail value (5% of rows, all climatology or ground tab at the top of the boundary layer). The A6 error of 20% is therefore an upper-typical error and 40% a tail error. The thesis convention (error relative to the prediction) equals the convention truth scale = 1 +- r with the nominal scale 1.

What the thesis does not support [DOC unless marked]:
1. The "truth" is itself a model result: the control is the dwell computed by the thesis software (HELTDA) from in-situ radiosonde profiles (printed pages 34 and 35), with GADS aerosol and HV 5/7 turbulence profiles assumed in all simulations (printed page 42). Real engagement dwell error is not measured, and aerosol and turbulence errors are not in the figures. They are a lower bound for the atmospheric part only.
2. Only the atmosphere is varied. Target hardness, material and aspect, jitter, tracking, thermal state and beam quality are not varied. A solid target-to-target spread within a class (our noise e) is not estimated by the thesis at all.
3. One airborne ELLA prototype (150 kW, 0.4 m aperture, beam quality 2.9, 150 m/s) in a circular orbit at about 3,977 m ground range around a stationary ground target; the damage threshold of 1e5 J/cm2 is described as an extremely hard target chosen to extend dwell (printed pages 42, 44 and 45), giving predicted dwell of 17 to 38 s at 1.045 micrometre (tables). It is not a counter-UAV scenario.
4. The RMSE has no sign. Bias and random error are not separated, so a shared scale offset cannot be told from day-to-day scatter.
5. The distribution of errors inside a row is not given (25 days are aggregated); no quantiles, no correlation between paths or azimuths.
6. The table lists the predicted DT, not the control DT. Where a method over-predicts strongly (for example ground tab at the Brunei boundary layer, DT 31.7 s against about 17.5 s for the GFS rows [OPEN, proxy]), the error relative to the control dwell would be larger than RMSE / DT.
7. How the number of threats, their ranges and azimuths change the shared share of the error is not studied.

#### A7.1 Critical review of A6 (Section 10.17, `docs/record.md`, A6 outcome note above)

1. The effect at 20% is borderline. Mean regret of the best method was 5.48% at -20% and 4.81% at +20%, with 4 and 2 of 8 cells flagged; the regions are four pairs at -20% (all at label accuracy 0.9) and one pair at +20% (load 1.5, slew 4). With Burley, 20% is an upper-typical error, so the FRAGILE_20 outcome rests on a tail, not on the typical case. Using the A6 pooled numbers only (best nominal planner 4.76% at truth 0.8, 6.25% at truth 1.2, 0% at truth 1.0), a symmetric scale distribution with standard deviation 16% (probabilities 1/3 each on 0.8, 1.0, 1.2) would give about 3.7% and one with standard deviation 10% about 1.4% [OPEN, back-of-envelope, not a result].
2. Conventions are mixed. The tolerance part uses believed scale minus 1 (the planner is wrong by -20% or +20% of the truth), the adaptation part uses the truths 0.8 and 1.2 against the nominal 1.0 (errors +25% and -17% relative to the truth). A7 uses the Burley convention: truth = nominal x (1 +- r).
3. The recovery criterion was met only through truth 1.2 (nominal regret 6.25%); at truth 0.8 the nominal regret (4.76%) was below the 5% bar.
4. The Bayes-adaptive policy is a ceiling, not a method: its prior is uniform on exactly the three scales that are later used as truths, it knows the noise law and the class prior, the scale is shared by all threats, and it is exact (unbounded compute). Its regret at the nominal truth 1.0 was computed in A6 (column `Vba_t1.0`) but not reported; adaptation can cost value when the forecast is right.
5. The result depends on the within-class noise. Without noise the regret at 20% is 9 to 17% and the recovery 0.85 to 0.90; with noise of plus or minus one step it is about 5% and 0.67 to 0.69. The true level of target-to-target variability is unknown (Burley does not give it).
6. n = 4, K = 6 classes, a three-point scale grid, symmetric noise, one effector. Deadlines and load are generated at the nominal scale, so a wrong scale changes the real load; this couples the scale error to the load.
7. All A1 to A6 are post-hoc; the 5%, 0.5 and 0.3 bars were set after earlier amendments; the cell flags use 200 instances per cell and no multiplicity control over 8 cells x 5 errors.
8. In A6 the planners had no way to adapt (tolerance part) and the only adaptive policy was the ceiling. Whether a practical planner gets most of the ceiling is untested.

#### A7.2 Design

Instance generation and model are those of A6 (base classes (5, 10), n = 4, values {1, 2, 3}, same cell grid: load 1.1 and 1.5, slew 2 and 4, label accuracy 0.5 and 0.9; deadlines, values and labels generated at the nominal scale). The hidden scale is shared by all threats. Dwell of a threat of base class c is round(scale x D0[c]) + e; classes are (base class, noise), independent across threats. Classes with equal dwell are merged before the DP (exact). Truth and planner models use the same noise law (the planner knows everything except the scale, as in A6).

Planners (all see only the dwell received and the kill flags; all are exact functions of the observable state; the value of each is an exact expectation computed by recursion over observable states, `eval_rec`, which equals the world-enumeration evaluator `eval_obs` in the tests):

| Name | Definition |
|---|---|
| Nominal DPMIS, CEP, ROLL_CEP | As in A6, built for the nominal scale 1.0 (the forecast). No adaptation |
| BA | Bayes-adaptive optimum for a stated prior over scales (exact DP). Ceiling |
| PLUG_DP | Plug-in with re-planning: after every step the posterior over the scale grid is computed from the dwell and kill history; the planner acts as the optimal policy (DP) of the model of the most probable scale (ties go to the scale nearest 1.0). If that model sees no useful action but the posterior still allows a kill, it keeps the current target |
| PLUG_CEP | The same with the CEP rule of the plug-in model |
| QMDP | Planning on the posterior mixture: the action with the highest posterior-weighted Q-value, where Q is the exact value of acting once and then following the optimal policy of the model of each scale. It does not value future learning |
| MIXCEP | CEP plan enumeration with the expected remaining dwell of every threat taken under the posterior mixture over scales (re-planned at every step) |
| Learned adaptive scorer | Not built in A7. Gated: see A7.6, outcome HEADROOM |

Observations outside the support of a planner's prior (truth outside the grid) are handled by a fixed convention (documented limitation): an alive dwell above the largest class dwell of the support is clamped to the largest minus one; a kill at a dwell that no class of the support has is mapped to the nearest support dwell (ties to the smaller); if all scales still have zero likelihood, the likelihoods are floored at 1e-6 for that observation only. In-support observations are never changed (tested).

Priors of the adaptive planners:

| Prior | Support and weights | Meaning |
|---|---|---|
| B | {0.8, 1.0, 1.2}, uniform | Matches a scale error with standard deviation 16.3% (as A6) |
| O | {0.8, 1.0, 1.2}, (0.125, 0.75, 0.125) | Forecast-calibrated: matches a standard deviation of 10% (Burley GFS median); narrower than B |
| W | {0.6, 0.8, 1.0, 1.2, 1.4}, uniform | Wider than the truths of interest |

Truths: scales 0.6, 0.8, 1.0, 1.2, 1.4 in `a7_main` (0.6 and 1.4 lie outside the support of B and O); 0.8, 1.0, 1.2 in the other runs. Scale error distributions Pi_p: probability p on 0.8, p on 1.2 and 1 - 2p on 1.0, with standard deviation 0.2 x sqrt(2p). p = 0.125 gives 10% (Burley GFS median), p = 0.25 gives 14.1%, p = 1/3 gives 16.3% (Burley climatology and ground tab median), p = 0.5 gives 20% (stress). Names SD10, SD14, SD16, SD20.

Runs (all in `configs/`; fresh master seeds, none used before; all instances paired across methods):

| Run | Content | Instances per cell (8 cells) | Master seed |
|---|---|---|---|
| `a7_main` | Noise +-1 (0.25, 0.5, 0.25), truths 0.6 to 1.4, priors B, O, W, all planners (W: PLUG_DP and QMDP only) | 200 | 20261124 |
| `a7_noise2` | Noise +-2 (binomial 1, 4, 6, 4, 1 over 16), truths 0.8, 1.0, 1.2, priors B and O | 100 | 20261126 |
| `a7_noise0` | No noise, same as noise2 otherwise (descriptive) | 100 | 20261125 |
| `a7_noise3` | Noise +-3 (binomial of order 6), same (descriptive) | 100 | 20261127 |
| `a7_h1` | Class-proportional idiosyncratic spread: offsets (-1, 0, 1) for the soft class and (-2, 0, 2) for the hard class, shared scale unknown (descriptive) | 100 | 20261128 |
| `a7_h2` | No shared scale: every threat has its own independent scale in {0.8, 1.0, 1.2} with probabilities (0.25, 0.5, 0.25); planners built for deterministic dwell; adaptive planners use the (wrong) shared-scale model B (descriptive) | 200 | 20261129 |

Heterogeneity (task d) is exact in three ways: the noise sweep (additive spread per threat), `a7_h1` (spread that grows with the class) and `a7_h2` (fully independent multiplicative scale per threat, i.e. no shared component). A partly shared scale is the sum of a shared scale and idiosyncratic spread, which `a7_h1` and the noise runs represent. The per-threat scale is not modelled as a hidden vector with correlation; that would need a different exact solver.

Metrics. Regret = 1 - (sum of values) / (sum of calibrated optima of the true scale), pooled over all cells and instances of a run. Expected regret under Pi_p = sum over the three truths of the probability times the regret. Paired bootstrap over instance index, 2000 resamples, seed 20261130; selections (best nominal planner, best practical planner) are recomputed in every resample; the lower bound is the 5th percentile. Gain = expected regret of the best nominal planner minus that of the best practical planner of the matched prior. Recovery = gain / expected regret of the best nominal planner. Attainment = gain / (nominal minus ceiling), defined only if that difference is at least 0.25 points.

#### A7.3 Pre-written decision table (applied mechanically by `scripts/decide_a7.py` to `a7_main` and `a7_noise2`)

Primary scenario: Pi at SD16, prior B (matched), the best practical planner of `a7_main` (the method is then fixed for S1 to S3).

| Criterion | Rule |
|---|---|
| M, material loss | Expected regret of the best nominal planner at SD16 at least 3.0% and its lower bound at least 2.0% (3% is about twice the mean regret of the best planners in A1 to A5 [MEM, from the summaries in project memory, unverified]) |
| P, practical recovery | Recovery at least 0.5 with lower bound at least 0.3, and attainment at least 0.75 |
| S1 | With prior O the primary method loses at most 1.0 point at the nominal truth 1.0 |
| S2 | At the truths 0.6 and 1.4 (outside the support of B) the primary method with prior B is not worse than the best nominal planner by more than 0.5 points |
| S3 | With the narrow prior O under SD16 truths the primary method is not worse than the best nominal planner |
| N | M and P both hold in `a7_noise2` (SD16, prior B) |

| Outcome | Meaning and action |
|---|---|
| Not M | STOP. At the Burley-level error the best nominal planner loses less than the materiality bar. The direction does not support a top-tier claim; report plainly |
| M, P, S1, S2, S3, N | STRONG-PRACTICAL. Proceed to the frozen confirmatory protocol with fresh seeds |
| M, P, S1, S2, S3, not N | REGIME-LIMITED. The effect needs low idiosyncratic noise. Continue only with a narrowed claim (noise regime map); the noise level must then be justified from the literature |
| M, P, not (S1, S2, S3) | UNSAFE-ADAPTATION. Add a safeguard (A7b) first |
| M, not P, ceiling recovery at least 0.5 | HEADROOM. Run A7-L (a learned adaptive scorer; specified in a further amendment before any run) |
| M, not P, ceiling recovery below 0.5 | CEILING-WEAK. Neither practical planners nor the ceiling recover enough: tolerance analysis only, moderate tier |

Reported without a rule: the same quantities at SD10 (the forecast case; scope note: material or not), SD14 and SD20; per-truth regrets; per-cell values; `a7_noise0`, `a7_noise3`, `a7_h1`, `a7_h2`; the wide prior W.

Stop rule. (1) Any assertion in `scripts/run_a7.py` fails (values outside [0, calibrated optimum], BA policy value different from its DP value, a practical planner above the Bayes-adaptive optimum, DPMIS at the nominal truth different from the calibrated value): no table is applied; the cause is fixed and the affected runs are repeated. (2) `a7_main` and `a7_noise2` give "Not M": the direction is dropped. (3) If the wall time of `a7_main` exceeds 4 hours on the author's laptop, stop and reduce the instances per cell before continuing. (4) The descriptive runs never change the outcome.

#### A7.4 Tests and checks [CODE, sandbox, Python 3.12.3, numpy 2.5.4, pandas 3.0.6, 1 core]

`tests/test_adapt.py` (23 tests; 202 tests in the whole suite pass): class dwells (flat and class-specific offsets, integer scaling fails loudly); merging of equal-dwell classes keeps mass and order; the posterior over scales equals an independent joint enumeration (uniform and non-uniform prior); the Bayes-adaptive DP value equals an independent brute force (non-uniform priors, class-specific offsets) and the average of the policy values over the scales equals the DP value; `eval_rec` equals world enumeration for all planners and for deliberately bad policies (infeasible or dead targets); QMDP Q-values equal an independent expectimax; the mixture durations and the first move of MIXCEP equal an independent enumeration; PLUG_DP equals the DP of the most probable scale; with a single-scale support all planners reduce to the nominal ones; practical planners lie between 0 and the calibrated optimum and below the Bayes-adaptive optimum on average; out-of-support truths run without failure; projection is the identity in support. 16 deliberate code mutations of `src/dwell/adapt.py` were tried: 15 are caught by the tests (five survived the first version of the tests and led to new tests); the 16th (dropping the class lower bound m from the translated state) is an equivalent mutant, because m is a function of the dwell in a model with sorted classes.

#### A7.5 Exploratory disclosure (seed 1, small samples; not results)

Smoke runs of the final code on exploratory seed 1 (not a frozen seed), in the sandbox, with the analysis scripts. a7_main, 24 instances (3 per cell): best nominal planner 6.9% (truth 0.8) and 9.3% (truth 1.2); QMDP 1.1% and 3.4%; the Bayes-adaptive ceiling with prior B 1.8% and 2.7%; at SD16 expected regret nominal 5.7%, QMDP 2.4%, ceiling 2.1%; at SD10 with prior O nominal 2.2%, QMDP 1.4%, ceiling 1.1%. PLUG_CEP and MIXCEP were poor (about 14 and 12% at the nominal truth). With prior B the ceiling itself loses 1.9% at the nominal truth 1.0. Noise runs (16 instances for m = 0 and 2, 8 for m = 3): SD16 nominal 8.9% (m = 0), 3.7% (m = 2), 3.2% (m = 3); recovery by QMDP 0.70, 0.26, 0.29. `a7_h1` (16 instances): nominal 3.2%, recovery 0.24. `a7_h2` (16 instances): nominal DPMIS 4.2%, the shared-scale adaptive planners 7.1 to 10.3%. Changes made after seeing these numbers: (1) the safety criterion S1 was moved from prior B to prior O because the ceiling itself pays about 1.9% at the nominal truth under B; (2) the attainment ratio is defined only when the nominal-to-ceiling gap is at least 0.25 points; (3) the forecast-case scope note was added. The main thresholds M, P and N were fixed before these smoke runs were analysed in detail. The smoke suggests the effect shrinks quickly with noise, so a REGIME-LIMITED or weaker outcome is plausible [OPEN].

#### A7.6 Limits

Post-hoc; one shared scale on a three-point (five-point for W) grid; the planners know the noise law and class prior; the Bayes-adaptive policy and QMDP are exact on a small model (n = 4) and their compute does not scale; the realistic within-class noise is unknown; Burley's error is atmospheric only and model-to-model; the fixed convention for out-of-support observations affects the results for truths 0.6 and 1.4; instance (not seed) is the unit because no policy is trained; no learned adaptive scorer is tested.
