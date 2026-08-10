# v100 Future Validation Execution Record

- `executed_at`: 2026-08-09
- `candidate`: `top_journal_mechanism_v100_urgency_safe_resource_mappo`
- `validation_split`: `future_validation`
- `frozen_plan`: `configs/experiment/top_journal_v20_future_validation_time_audited_20260717/future_validation_window_plan.json`
- `artifact_run_id`: `top_journal_v100_future_validation_v20_20260809/main_results_full_stratified_20260809_041440_446507`
- `evidence_level`: `E2_ARTIFACT_AUDITED`
- `protocol`: NGSIM + Alibaba, 11 agents, 3 seeds (`7,13,29`), 15 frozen windows, 2 workflows, 22 steps, zero reward offset

## Result

| agent | total reward | continuity | handoff ready | mechanism realization |
|---|---:|---:|---:|---:|
| SA-GHMAPPO | 33.342 | 0.867 | 0.600 | 0.600 |
| Popularity | 29.350 | 0.862 | 0.533 | 0.567 |
| DQN | 24.236 | 0.824 | 0.400 | 0.400 |
| Dueling-DQN | 19.159 | 0.846 | 0.200 | 0.200 |
| DT-Handoff | 10.027 | 0.582 | 0.544 | 0.544 |
| PPO | 9.169 | 0.670 | 0.311 | 0.311 |
| QMIX | 8.924 | 0.451 | 0.822 | 0.822 |
| DAG-Offload | 8.600 | 0.445 | 0.822 | 0.822 |
| Cache-Offload | 3.125 | 0.359 | 0.867 | 0.867 |
| MAPPO | -1.066 | 0.275 | 0.933 | 0.933 |
| Controller-MAT | -1.271 | 0.275 | 0.933 | 0.933 |

SA-GHMAPPO versus Popularity total-reward difference was `+3.992`, hierarchical window-bootstrap BCa 95% CI `[+2.077111, +8.290975]`, paired wins/ties/losses `75/15/0`, Holm-adjusted sign-test p=`0.0`. Against all ten baselines, the total-reward BCa lower bound was positive.

The v20 future windows have no frame-interval overlap with v71 train, dev, formal or hidden plans. The v100 checkpoint manifest was frozen before this one-time evaluation, and no tuning was performed after the result was read.

## Evidence Boundary

This is strong cross-split reward evidence for the frozen v100 candidate, not proof of universal superiority. SA does not dominate every mechanism or cost metric: the reward advantage is partly associated with delay/continuity behavior, while backhaul and migration trade-offs must remain in the paper table. Cross-mobility/workflow validation, complete ablations and unified compute accounting remain open before a TMC-ready claim.
