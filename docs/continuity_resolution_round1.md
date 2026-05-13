# Continuity Resolution Round 1

更新日期：2026-04-24

## 问题

三 seed formal rerun 后，`sa_ghmappo` 使用 `best_by_reward` checkpoint 时，
`mixed_informative` 主协议下的 continuity 为 `0.979630`，仍低于
`popularity_cache_heuristic` 的 `1.000000`。

本轮不修改 `VecWorkflowCoreEnv`、reward、handoff、migration 或 adapter cache 语义；
只检查 checkpoint selection 和 episode stall 归因。

## 处理方式

新增两个 seed-level manifest：

- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/seed_checkpoint_manifest_formal_round1_3seed_sa_best_continuity.json`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/seed_checkpoint_manifest_formal_round1_3seed_sa_best_mechanism_balanced.json`

并重跑同协议 benchmark：

- `benchmark_formal_round1_3seed_sa_best_continuity/main_results_mixed_informative_20260424_191504_118596/aggregate_summary.json`
- `benchmark_formal_round1_3seed_sa_best_mechanism_balanced/main_results_mixed_informative_20260424_191504_207066/aggregate_summary.json`
- `benchmark_formal_round1_3seed_sa_best_continuity_full_stratified/main_results_full_stratified_20260424_191724_597766/aggregate_summary.json`

## 主协议结果

`mixed_informative`，agents `sa_ghmappo reactive_greedy popularity_cache_heuristic flat_ppo flat_mappo`，
seeds `7 13 29`。

| selection | agent | reward | continuity | handoff_ready | mechanism |
|---|---|---:|---:|---:|---:|
| best_by_reward | sa_ghmappo | 84.104444 | 0.979630 | 0.472222 | 0.611111 |
| best_by_reward | popularity_cache_heuristic | 83.513333 | 1.000000 | 0.416667 | 0.500000 |
| best_by_continuity | sa_ghmappo | 83.218889 | 1.000000 | 0.416667 | 0.500000 |
| best_by_continuity | popularity_cache_heuristic | 83.513333 | 1.000000 | 0.416667 | 0.500000 |
| best_by_mechanism_balanced | sa_ghmappo | 83.163333 | 1.000000 | 0.416667 | 0.500000 |
| best_by_mechanism_balanced | popularity_cache_heuristic | 83.513333 | 1.000000 | 0.416667 | 0.500000 |

结论：`best_by_continuity` 已解决主协议下主方法 continuity 低于 popularity heuristic 的问题；
代价是 reward 比 `best_by_reward` 低约 `0.885555`，且略低于 popularity heuristic。

## Window-Class 复查

`full_stratified` 下，`best_by_continuity` 覆盖 `mechanism_activating`、
`active_non_mechanism` 和 `idle_or_sparse`。

| agent | reward | continuity | handoff_ready | mechanism |
|---|---:|---:|---:|---:|
| sa_ghmappo | 75.623333 | 0.915404 | 0.208333 | 0.250000 |
| popularity_cache_heuristic | 74.898333 | 0.915404 | 0.208333 | 0.250000 |
| reactive_greedy | 67.070833 | 0.886237 | 0.000000 | 0.000000 |
| flat_ppo | 59.716667 | 0.591667 | 0.000000 | 0.444444 |
| flat_mappo | 59.716667 | 0.591667 | 0.000000 | 0.444444 |

结论：在三类窗口混合复查下，`best_by_continuity` 与 popularity heuristic 的 continuity、
handoff_ready 和 mechanism 持平，reward 更高。

## Stall 归因

诊断 artifact：

- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/continuity_resolution_round1/checkpoint_selection_and_stall_attribution.json`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/continuity_resolution_round1/checkpoint_selection_summary.csv`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/continuity_resolution_round1/stall_attribution_summary.csv`

`best_by_reward` 在主协议下：

- `sa_ghmappo` 共 `191` step，`4` stall，stall rate `0.020942`。
- 4 个 stall 全部发生在 `mechanism_activating` window。
- 4 个 stall 全部对应 action `predictive_next_rsu_prefetch`。
- 归因均为 `handoff_failed + cache_miss + cache_target_alignment_mismatch + handoff_event + prefetch_related`。

`best_by_continuity` 在主协议下：

- `sa_ghmappo` 共 `189` step，`0` stall。

`best_by_continuity` 在 full-stratified 复查下：

- `sa_ghmappo` 共 `393` step，`36` stall，stall rate `0.091603`。
- stall 全部来自 `idle_or_sparse` window。
- action 归因：`predictive_next_rsu_prefetch` 24 次，`current_rsu_steady_offload` 12 次。
- 该 full-stratified aggregate 中主方法与 popularity heuristic 的 continuity 持平，说明这部分不是主方法相对劣势的主要来源。

## 本轮结论

本轮问题已通过 checkpoint selection 解决：主方法在三 seed 主协议下可用
`best_by_continuity` 达到 `1.000000` continuity，追平 popularity heuristic。

后续若要同时保持 `best_by_reward` 的 reward 优势和 `best_by_continuity` 的连续性，需要做 agent 侧
continuity guard 或训练期 imitation/regularization；本轮不强行改策略，以免把尚未归因充分的问题写进主方法实现。

推荐后续两条汇报口径分开：

- reward/mechanism 主结果：使用 `best_by_reward`，说明 continuity 为 `0.979630`，机制指标更高。
- continuity-safe 结果：使用 `best_by_continuity`，说明 continuity 追平 heuristic，full-stratified reward 仍高于 popularity heuristic。
