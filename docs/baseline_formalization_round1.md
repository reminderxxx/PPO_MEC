# Baseline Formalization Round 1

更新日期：2026-04-23

本轮目标是把 baseline 闭环从 smoke 推进到可追踪的多 seed round1。核心环境、reward、handoff、migration 和 adapter cache 语义未改。

## 运行记录

执行命令：

```bash
python scripts/run_baseline_experiment.py --config configs/experiment/baseline/minimal_ngsim_alibaba.yaml
```

结果入口：

- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260423_160112/comparison_summary.json`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260423_160112/comparison_summary.csv`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260423_160112/comparison_summary_detailed.json`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260423_160112/comparison_summary_by_window_class.csv`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260423_160112/run_manifest.json`

协议摘要：

- seeds：`7`、`13`
- selected windows：`mechanism_activating=2`、`active_non_mechanism=1`、`idle_or_sparse=0`
- trainable baseline：`flat_ppo`、`flat_mappo`
- heuristic baseline：`reactive_greedy`、`popularity_cache_heuristic`
- skeleton：`td3`、`qmix`

## 已拉开的差异

`popularity_cache_heuristic` 是 prediction-aware heuristic。它会使用 predicted next RSU / handoff target 做 prefetch 或 migration prepare，不能被写成无预测 baseline。

在 `mechanism_activating` 窗口上，`popularity_cache_heuristic` 相比 `reactive_greedy` 已出现机制差异：

- `predictive_prefetch_request_count_mean`：`+1.0`
- `validated_predictive_prefetch_count_mean`：`+0.5`
- `workflow_continuity_rate_mean`：`+0.083333`
- `total_reward_mean`：`49.81` vs `37.525`

`sa_ghmappo` 相比 flat baselines 已明显分开：

- 相比 `flat_ppo` / `flat_mappo`，`mechanism_activating` 上 `total_reward_mean +22.735`
- 相比 `flat_ppo` / `flat_mappo`，`workflow_continuity_rate_mean +0.833333`
- `flat_ppo` 和 `flat_mappo` 已能完成 per-seed train/eval/benchmark，但当前预算下策略仍弱。

在 `active_non_mechanism` 窗口上，prediction-aware baseline 未出现额外误触发：

- `popularity_cache_heuristic - reactive_greedy` 的 `predictive_prefetch_request_count`、`migration_prepare_count`、`backhaul_traffic_cost`、`workflow_continuity_rate` delta 均为 `0.0`

## 尚未拉开的差异

`sa_ghmappo` 在 `mechanism_activating` 窗口上没有全面优于 `popularity_cache_heuristic`：

- `sa_ghmappo - popularity_cache_heuristic` 的 `workflow_continuity_rate_mean` 为 `-0.166667`
- `sa_ghmappo - popularity_cache_heuristic` 的 `predictive_prefetch_request_count_mean` 为 `-1.0`
- `sa_ghmappo - popularity_cache_heuristic` 的 `validated_predictive_prefetch_count_mean` 为 `-0.5`

`handoff_ready_count_mean` 当前仍未拉开，各 agent 在机制窗口上均为 `0.0`。这说明本轮窗口能触发 prefetch 差异，但还不足以稳定触发 handoff-ready / migration-ready 的最终机制差异。

`idle_or_sparse` 未进入 selected windows，当前 by-window-class CSV 中保留空行；该类别还没有实际对照样本。

## 原因判断

- 窗口问题：`mixed_informative` 当前选中了机制窗口和 active non-mechanism 窗口，但没有选中 idle/sparse；handoff-ready 仍缺少足够触发样本。
- 训练预算问题：`flat_ppo` / `flat_mappo` 只用 round1 最小预算训练，能验证 per-seed artifact 闭环，但不能代表收敛性能。
- 机制触发问题：prediction-aware heuristic 已触发 predictive prefetch 并产生 validated hit；主方法在当前 checkpoint 与窗口组合下未表现出更高 prefetch 请求。

## 下轮建议

- 将 selected `mechanism_activating` window 增至更多窗口，优先包含 predicted handoff target non-null ratio 更高的窗口。
- 增加 `flat_ppo` / `flat_mappo` 训练预算，再比较 PPO-family baseline。
- 增加 `idle_or_sparse` selected window，用于检查 prediction-aware policy 的误触发成本。
- 保持 TD3 / QMIX skeleton，除非先冻结连续动作或 multi-agent wrapper contract。
