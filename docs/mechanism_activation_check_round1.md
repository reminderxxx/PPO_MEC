# Mechanism Activation Check Round 1

更新日期：2026-04-24

诊断来源：`artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/comparison_summary_by_window_class.csv`

本文件只检查实验是否触发关键机制，不做论文结论。

## Window Class 覆盖

- `mechanism_activating`：已选中 2 个窗口，`window_off524_len16_t576_765` 和 `window_off246_len16_t298_313`。
- `active_non_mechanism`：已选中 1 个窗口，`window_off150_len16_t202_217`。
- `idle_or_sparse`：本轮未选中窗口，CSV 中保留空行，`episode_count=0`。

补充检查：已基于同一批 per-seed checkpoint 执行 `full_stratified` benchmark-only 复跑：

- aggregate：`artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark_full_stratified/main_results_full_stratified_20260424_174124_983467/aggregate_summary.json`
- rows：`artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark_full_stratified/main_results_full_stratified_20260424_174124_983467/benchmark_rows.csv`
- selected windows：`mechanism_activating=2`，`active_non_mechanism=2`，`idle_or_sparse=2`
- episode count：`60`

## Mechanism Activating

在 `mechanism_activating` 窗口上，机制链路已经被部分触发：

- `popularity_cache_heuristic`：`predictive_prefetch_request_count_mean=1.0`，`validated_predictive_prefetch_count_mean=0.5`，`migration_prepare_count_mean=4.0`，`mechanism_realization_rate_mean=0.5`。
- `sa_ghmappo`：`predictive_prefetch_request_count_mean=0.0`，`validated_predictive_prefetch_count_mean=0.0`，`migration_prepare_count_mean=2.5`，`mechanism_realization_rate_mean=0.5`。
- `reactive_greedy`：prefetch、validated prefetch 和 migration prepare 均为 `0.0`，`mechanism_realization_rate_mean=0.0`。
- `flat_ppo` / `flat_mappo`：prefetch、migration prepare 和 realization 均为 `0.0`。

连续性差异已经出现但不稳定：`popularity_cache_heuristic` 在机制窗口上 `workflow_continuity_rate_mean=1.0`，`reactive_greedy=0.916667`，`sa_ghmappo=0.833333`，`flat_ppo/flat_mappo=0.0`。

## Active Non Mechanism

在 `active_non_mechanism` 窗口上，各 agent 的关键机制计数均为 `0.0`：

- `predictive_prefetch_request_count_mean=0.0`
- `validated_predictive_prefetch_count_mean=0.0`
- `migration_prepare_count_mean=0.0`
- `handoff_ready_count_mean=0.0`
- `mechanism_realization_rate_mean=0.0`

这说明当前 prediction-aware heuristic 没有在该类窗口出现明显误触发。

## Idle Or Sparse

原始 `mixed_informative` run 没有选中 `idle_or_sparse` 窗口。补充 `full_stratified` benchmark 覆盖了 2 个 `idle_or_sparse` 窗口，结果如下：

- `sa_ghmappo`：`predictive_prefetch_request_count_mean=0.0`，`migration_prepare_count_mean=0.0`，`handoff_ready_count_mean=0.0`，`mechanism_realization_rate_mean=0.0`。
- `reactive_greedy`：上述关键机制计数均为 `0.0`。
- `popularity_cache_heuristic`：上述关键机制计数均为 `0.0`。
- `flat_ppo` / `flat_mappo`：上述关键机制计数均为 `0.0`。

这说明在本轮选中的 idle/sparse 窗口上，prediction-aware heuristic 和主方法都没有出现 prefetch / migration prepare 误触发。

## Handoff Ready

所有已选窗口上 `handoff_ready_count_mean=0.0`。这表示 round1 已触发 predictive prefetch 和 migration prepare，但还没有稳定触发 handoff-ready 的最终状态。

## 当前判断

- 机制没有“完全没进入动作链”：`popularity_cache_heuristic` 的 predictive prefetch 和 migration prepare 已进入动作链，`sa_ghmappo` 至少触发了 migration prepare / realization。
- 差异没全面拉开的主要原因更像训练预算和 handoff-ready 条件问题：`flat_ppo` / `flat_mappo` 训练预算很小；补充 `full_stratified` 已覆盖 idle/sparse，未观察到误触发；handoff-ready 条件仍没有被当前窗口稳定打到。
- 主方法当前 checkpoint 在这组窗口上没有表现出 predictive prefetch 优势；下一轮应扩大机制窗口数量和训练预算，再判断主方法策略是否学到 prefetch 链路。
