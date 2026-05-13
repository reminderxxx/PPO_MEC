# SA-GHMAPPO mechanism policy round2 diagnosis

Date: 2026-04-25

Scope: this round did not modify `VecWorkflowCoreEnv`, reward formula, benchmark split, or `popularity_cache_heuristic`. The diagnosis below uses existing round1 V2 artifacts plus the new round2 training and benchmark artifacts.

## Inputs

- Round1 V2 diagnosis: `docs/agent/sa_advantage_round1_mechanism_diagnosis.md`
- Round1 V2 report: `docs/agent/sa_advantage_round1_mechanism_improvement_report.md`
- Round1 V2 selection summary: `artifacts/training/sa_advantage_round1/mechanism_advantage_selection_summary.json`
- Round2 seed29 trace export: `artifacts/training/sa_mechanism_policy_round2/seed29_mechanism_window_diagnostics_round2.csv`
- Round2 train summary: `artifacts/training/sa_mechanism_policy_round2/sa_ghmappo/sa_ghmappo_train_20260425_181700_754342_seed29/train_summary.json`

## Seed29 mechanism-window blocker

Round1 V2 seed29 selected update 15 had candidates and timing, but low prepare probability:

| signal | round1 V2 seed29 update15 |
|---|---:|
| raw_handoff_candidate_step_count | 34 |
| valid_handoff_target_step_count | 34 |
| timing_active_step_count | 34 |
| predictor_invoked_step_count | 84 |
| gate_pass_step_count | 16 |
| event_prepare_prob_mean | 0.139703 |
| event_prepare_prob_p75 | 0.091803 |
| guard_prefetch_to_prepare_count | 1 |
| migration_prepare_rate | 0.062500 |
| handoff_ready_ratio | 0.187500 |
| mechanism_realization_rate | 0.375000 |

This confirms the earlier blocker: candidate and timing signals were present, but the policy head did not put enough mass on prepare/prefetch/migration behavior in the critical window.

## Round2 policy signal

Round2 selected `best_by_round2_mechanism_score.pt` for seed29 maps to `update_0008.pt`. At that selected update:

| signal | round2 seed29 update8 |
|---|---:|
| raw_handoff_candidate_step_count | 34 |
| valid_handoff_target_step_count | 34 |
| timing_active_step_count | 34 |
| predictor_invoked_step_count | 84 |
| gate_pass_step_count | 16 |
| mechanism_prepare_action_legal_count | 45 |
| mechanism_prefetch_action_legal_count | 18 |
| mechanism_guided_action_count | 15 |
| weighted_mechanism_transition_ratio | 0.333333 |
| mechanism_head_entropy | 0.520547 |
| event_prepare_prob_before_update | 0.728025 |
| event_prepare_prob_after_update | 0.680621 |
| update-eval event_prepare_prob_mean | 0.452759 |
| guard_prefetch_to_prepare_count | 26 |
| migration_prepare_rate | 0.357639 |
| handoff_ready_ratio | 0.312500 |
| mechanism_realization_rate | 0.375000 |

The direct improvement is policy-side: the same candidate/timing/gate structure now receives substantially higher prepare probability and more guard-to-prepare conversions. There is no evidence that the original issue was a missing candidate or an over-strict valid-target filter.

## Legality and masks

For round2 seed29 update8, `mechanism_prepare_action_legal_count=45` and `mechanism_prefetch_action_legal_count=18` were recorded during learning. This indicates that prepare/prefetch-related choices were available in the mechanism-window training batch. The observed issue was low policy preference, not action illegality.

The benchmark/training summaries do not currently persist full per-step `action_info` traces for every benchmark step. They persist per-window/workflow step counts and rates, plus training-batch legal/guidance counts. The exported CSV therefore diagnoses at window-row granularity rather than raw step-row granularity.

## Gate and candidate filtering

Seed29 still shows:

| signal | value |
|---|---:|
| candidate_block_reason_no_next_rsu_count | 4 |
| candidate_block_reason_same_rsu_count | 46 |
| candidate_block_reason_eta_outside_window_count | 4 |
| invalid_reason_low_confidence_count | 18 |
| gate_pass_rate | 0.190476 |

These are stable between round1 and round2. The gate is selective, but it is not the round2 bottleneck: valid handoff target and timing-active counts remain nonzero, and round2 improves prepare probability without changing the gate.

## Direct cause of low guard conversion in round1

Round1 V2 seed29 had `guard_prefetch_to_prepare_count=1` at the selected update despite `target_mismatch_guard_count=4`. The prepare head was weak (`event_prepare_prob_mean=0.139703`), so the guard rarely converted prefetch intent into prepare behavior. Round2 raises this to `guard_prefetch_to_prepare_count=26` at the selected update.

## Remaining risk

Round2 improves selected-update behavior, but later updates still drift downward in seed29: latest update event prepare probability is `0.206564`, with `guard_prefetch_to_prepare_count=15`. This is why `best_by_round2_mechanism_score.pt` selects update 8 instead of the final update. The next minimal optimization, if needed, is to improve late-training retention of the mechanism auxiliary signal rather than changing the environment or reward.

