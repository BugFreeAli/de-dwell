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

