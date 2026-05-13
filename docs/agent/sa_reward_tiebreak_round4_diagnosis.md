# sa_reward_tiebreak_round4 Diagnosis

Scope: diagnose why `sa_mechanism_policy_round2` still trails `popularity_cache_heuristic` by `0.108333` reward on `mixed_informative`, and whether checkpoint selection can close it. This diagnosis did not modify `VecWorkflowCoreEnv`, reward formula, benchmark split, or `popularity_cache_heuristic`.

## Sources

- Round2 mixed: `artifacts/benchmarks/sa_mechanism_policy_round2/mixed_informative/main_results_mixed_informative_20260425_182051_736385/`
- Round2 full: `artifacts/benchmarks/sa_mechanism_policy_round2/full_stratified/main_results_full_stratified_20260425_182234_412726/`
- Round2 update sweep: `artifacts/eval/sa_reward_tiebreak_round4_round2_selection_sweep/round2_update_sweep_rows.csv`

## Mixed Gap

Overall mixed delta:

| metric | sa_ghmappo | popularity | SA - popularity |
|---|---:|---:|---:|
| total_reward | 83.405000 | 83.513333 | -0.108333 |
| continuity | 1.000000 | 1.000000 | 0.000000 |
| handoff_failure | 0.000000 | 0.000000 | 0.000000 |
| backhaul | 124.444444 | 170.666667 | -46.222222 |
| handoff_ready | 0.416667 | 0.416667 | 0.000000 |
| mechanism | 0.500000 | 0.500000 | 0.000000 |

The gap is not from delay, continuity, handoff failure, handoff readiness, mechanism realization, migration overhead, or backhaul. All are tied except backhaul, where SA is much lower.

By window class:

| window_class | SA reward | popularity reward | delta | SA backhaul | popularity backhaul |
|---|---:|---:|---:|---:|---:|
| active_non_mechanism | 69.975000 | 69.975000 | 0.000000 | 64.000000 | 64.000000 |
| mechanism_activating | 90.120000 | 90.282500 | -0.162500 | 154.666667 | 224.000000 |

The whole mixed gap is in `mechanism_activating` windows. `active_non_mechanism` is exactly tied.

By seed:

| seed | SA reward | popularity reward | delta | SA backhaul | popularity backhaul |
|---:|---:|---:|---:|---:|---:|
| 7 | 83.388333 | 83.513333 | -0.125000 | 117.333333 | 170.666667 |
| 13 | 83.438333 | 83.513333 | -0.075000 | 138.666667 | 170.666667 |
| 29 | 83.388333 | 83.513333 | -0.125000 | 117.333333 | 170.666667 |

No single seed is collapsing; seed 7 and 29 are slightly worse, seed 13 is closer.

## Checkpoint Selection Audit

The round2 checkpoint sweep evaluated all `update_*.pt` for each seed on formal `mixed_informative` and `full_stratified` windows.

| seed | selected mixed checkpoint | selected mixed reward | best mixed checkpoint | best mixed reward |
|---:|---|---:|---|---:|
| 7 | update_0006 | 83.388333 | update_0006 | 83.388333 |
| 13 | update_0008 | 83.438333 | update_0008 | 83.438333 |
| 29 | update_0008 | 83.388333 | update_0008 | 83.388333 |

There is no round2 checkpoint that improves mixed reward while keeping the existing safety profile. For seed29, `update_0009` is better on full reward/readiness, but lower on mixed reward than `update_0008`; it does not solve this task.

## Interpretation

The current exported benchmark rows do not include full per-step reward component sums for service/cache/internal action terms, so the exact `0.108333` mixed reward loss cannot be decomposed into every reward subcomponent from benchmark rows alone. The observable evidence says:

- not delay-related: `end_to_end_workflow_delay` is tied on mixed;
- not handoff-related: continuity and failure are tied;
- not readiness/realization-related: both are tied;
- not migration-overhead-related: exported overhead is tied;
- not backhaul-cost-related in the usual direction: SA has much lower backhaul but slightly lower reward.

The most likely explanation is a small action-mix/service/cache realization difference inside mechanism windows that is not exposed by the current formal benchmark rows. It is not a checkpoint selection miss.

## Round4 Risk Found During Diagnosis

An initial round4 profile changed inference-facing event-head temperature/sharpening settings. That produced worse mixed reward (`82.202222`) and higher backhaul (`156.444444`) while still not closing the gap. The profile was corrected to preserve round2 action-side behavior; the final round4 selector then chose the warm-start checkpoint for all seeds.

Conclusion: round4 selection/fine-tune did not find a safe improvement. Keep round2 as the current qualified candidate.
