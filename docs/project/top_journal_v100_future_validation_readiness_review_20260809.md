# v100 Future-Validation Top-Journal Readiness Review

reviewed_at: 2026-08-09
literature_cutoff: 2026-08-09
target_venue: IEEE Transactions on Mobile Computing (TMC)
artifact_run_id: top_journal_v100_future_validation_v20_20260809/main_results_full_stratified_20260809_041440_446507
policy_version: tmc_review_policy_v3_20260621
git_commit: dfe1514 (v100 checkpoint provenance)
evidence_level: E2_ARTIFACT_AUDITED
verdict: Not TMC-ready; strong cross-split reward candidate, major revision

## Evidence Inventory

- v100 formal package: 11 agents, 20 frozen v71 formal windows, 2 workflows, 3 seeds and 1320 raw summaries.
- Independent v20 future-validation package: 11 agents, 15 outcome-blind frozen windows, 2 workflows, 3 seeds and 990 raw summaries.
- Future SA versus Popularity reward delta: `+3.992`, BCa 95% CI `[+2.077111,+8.290975]`, paired `75/15/0`, Holm-adjusted sign-test p=`0.0`; reward BCa lower bound is positive against all ten baselines.
- Future windows have no frame-interval overlap with v71 train/dev/formal/hidden plans. Integrity audit passed with zero missing references and zero JSON errors.
- The historical v71 hidden package was consumed by v98 and is not reused. The v20 future package is treated as a one-time frozen validation and no post-result tuning is allowed.

## Remaining Blockers

- Only NGSIM + Alibaba is covered; cross-mobility and cross-workflow generalization is absent.
- Mechanism realization and cost metrics are not uniformly dominant. On the future split, SA mechanism realization is `0.600` versus Popularity `0.567`, but learned baselines can have higher ready/realization values and SA backhaul/migration trade-offs remain.
- Complete component ablations and unified wall-clock, memory, parameter and inference-cost accounting are absent.
- The large future reward margin is evidence for the frozen candidate, not permission to claim universal or deployment-level superiority.

## Scorecard

| dimension | score | evidence and deduction |
|---|---:|---|
| Novelty and literature positioning | 15/20 | urgency-safe multi-horizon resource-constrained MAPPO policy improvement is concrete; broader nearest-neighbor positioning is incomplete |
| Technical correctness and modeling | 12/15 | exact legal-action branch targets and policy-prior-constrained planner are implemented and checkpointed; complexity analysis is incomplete |
| Baseline fairness and independence | 13/15 | 11-agent, 3-seed formal and future packages share the controller contract; compute parity needs a unified audit |
| Experiments, statistics and holdout | 17/20 | formal plus independent future BCa/paired statistics are available; only one data combination is covered |
| Mechanism realization and system metrics | 6/10 | reward lead is replicated, but mechanism and cost dominance is not universal |
| Robustness, generalization and scalability | 6/10 | future split is positive; cross-dataset, full noise and scalability packages are missing |
| Reproducibility and claim completeness | 9/10 | raw summaries, manifests, command logs, statistics and integrity audit exist; compute package is incomplete |
| **Total** | **78/100** | **Not TMC-ready** |

## Safe Claims

- Under the frozen NGSIM + Alibaba controller-level protocol and the independent v20 future split, SA-GHMAPPO ranks first in total reward among all 11 evaluated methods.
- The future split confirms a positive reward lead over Popularity with a hierarchical BCa interval fully above zero; this is stronger evidence than the prior v100 formal replication alone.
- The algorithmic source is MAPPO-side exact legal-action multi-horizon policy improvement with urgency/resource constraints and an auditable online counterfactual planner; environment reward and evaluator filtering were unchanged.

## Prohibited Claims

- Do not claim universal superiority, all-metric dominance, cross-dataset generalization, real deployment performance or TMC acceptance.
- Do not reopen v71 hidden or tune against the v20 future result. Any next algorithm change requires a new pre-frozen split and protocol.

## Required Actions Before Re-review

1. Add an independent mobility/workflow combination and a second system-scale setting.
2. Run component ablations for multi-horizon targets, urgency weighting, resource penalty and planner coupling with matched seeds and budgets.
3. Add unified wall-clock, peak-memory, parameter-count and inference/branch-cost accounting for every learned baseline and the proposed planner.
4. Report mechanism and cost trade-offs as primary results or constrained objectives, not only total reward.
