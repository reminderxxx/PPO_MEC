# SA-GHMAPPO Advantage Round 1 Mechanism Improvement Report

本轮只做 checkpoint selection 和诊断增强；未改环境语义、reward 主定义、benchmark split 或 popularity baseline。

## 新增 Selection V2

新增 checkpoint label: `best_by_mechanism_advantage_score.pt`

score 版本：`mechanism_advantage_score_v2`

公式：

```text
reward_term
+ 45 * continuity_advantage
+ 35 * handoff_failure_reduction
+ 0.08 * backhaul_cost_reduction
+ 30 * handoff_ready_advantage
+ 28 * mechanism_realization_advantage
+ 8 * prefetch_validated_hit_rate
+ 5 * migration_prepare_rate
- 0.2 * adapter_state_migration_overhead
- stability_penalty
```

如果 reference metrics 缺失，自动回退到 self-metrics 评分，并在 selection reason 中记录 `reference_metrics_available=false` 和缺失字段。stability penalty 使用窗口级 `continuity_floor`、`handoff_failure_ceiling`、`reward_std`、`handoff_ready_floor`、`mechanism_realization_floor`。

## V2 选择结果

V2 从现有 64-episode 训练 run 中选择出的 checkpoint 与 v1 `best_by_advantage_score` 相同：

| seed | selected update | score | checkpoint |
| ---: | ---: | ---: | --- |
| 7 | 6 | 136.847478 | `artifacts/training/sa_advantage_round1/sa_ghmappo/sa_ghmappo_train_20260425_122902_278394_seed7/checkpoints/best_by_mechanism_advantage_score.pt` |
| 13 | 7 | 136.847478 | `artifacts/training/sa_advantage_round1/sa_ghmappo/sa_ghmappo_train_20260425_123152_951083_seed13/checkpoints/best_by_mechanism_advantage_score.pt` |
| 29 | 15 | 125.309554 | `artifacts/training/sa_advantage_round1/sa_ghmappo/sa_ghmappo_train_20260425_123439_169766_seed29/checkpoints/best_by_mechanism_advantage_score.pt` |

Manifest:

- `artifacts/training/sa_advantage_round1/seed_checkpoint_manifest_sa_advantage_round1_best_by_mechanism_advantage_score.json`

Selection summary:

- `artifacts/training/sa_advantage_round1/mechanism_advantage_selection_summary.json`

## 新 Benchmark 目录

- Mixed: `artifacts/benchmarks/sa_advantage_round1_mechanism_v2/mixed_informative/main_results_mixed_informative_20260425_130801_604446/`
- Full: `artifacts/benchmarks/sa_advantage_round1_mechanism_v2/full_stratified/main_results_full_stratified_20260425_131017_570497/`

## 旧结果 vs 新结果

### Mixed Informative

| 指标 | old SA | old popularity | old delta | new SA | new popularity | new delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| total_reward | 80.872222 | 83.513333 | -2.641111 | 80.872222 | 83.513333 | -2.641111 |
| workflow_continuity_rate | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 |
| handoff_failure_rate | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| backhaul_traffic_cost | 119.111111 | 170.666667 | -51.555556 | 119.111111 | 170.666667 | -51.555556 |
| handoff_ready_ratio | 0.361111 | 0.416667 | -0.055556 | 0.361111 | 0.416667 | -0.055556 |
| mechanism_realization_rate | 0.500000 | 0.500000 | 0.000000 | 0.500000 | 0.500000 | 0.000000 |
| adapter_state_migration_overhead | 0.544444 | 0.453333 | +0.091111 | 0.544444 | 0.453333 | +0.091111 |

mixed reward 未改善。原因是 V2 没有选出不同 checkpoint；现有训练预算内更高 reward 的 seed29 候选在窗口级 failure/continuity 上风险更高，V2 按稳定性惩罚没有选它。

### Full Stratified

| 指标 | old SA | old popularity | old delta | new SA | new popularity | new delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| total_reward | 75.525000 | 75.492778 | +0.032222 | 75.525000 | 75.492778 | +0.032222 |
| workflow_continuity_rate | 0.979938 | 0.924242 | +0.055696 | 0.979938 | 0.924242 | +0.055696 |
| handoff_failure_rate | 0.037037 | 0.180556 | -0.143519 | 0.037037 | 0.180556 | -0.143519 |
| backhaul_traffic_cost | 153.777778 | 158.222222 | -4.444444 | 153.777778 | 158.222222 | -4.444444 |
| handoff_ready_ratio | 0.175926 | 0.250000 | -0.074074 | 0.175926 | 0.250000 | -0.074074 |
| mechanism_realization_rate | 0.259259 | 0.277778 | -0.018519 | 0.259259 | 0.277778 | -0.018519 |
| adapter_state_migration_overhead | 1.216667 | 1.226667 | -0.010000 | 1.216667 | 1.226667 | -0.010000 |

full 的最低成功标准保持：reward、continuity、handoff failure、backhaul cost、migration overhead 都优于 popularity。

## 验收问题回答

1. mixed 中 reward 是否改善：没有，V2 选择与 V1 相同。
2. full 中优势是否保持：保持。
3. handoff_ready_ratio 是否提升：没有。
4. mechanism_realization_rate 是否提升：没有。
5. backhaul cost 是否仍低于 popularity：是，mixed 低 `51.555556`，full 低 `4.444444`。
6. 是否牺牲 continuity / handoff failure 换 reward：没有。V2 保留了安全 checkpoint，mixed continuity/failure 仍持平，full continuity/failure 仍领先。
7. 是否建议 freeze 为 paper candidate：如果论文 claim 是 full_stratified 下的综合 utility、continuity、failure 和 backhaul 优势，可以作为最低 freeze candidate；如果论文 claim 要强调 mechanism readiness/realization 全面领先，则不建议 freeze。

## 下一步最小修改

selection 已经无法从现有 checkpoint 中找到更优安全点。下一轮最小有效动作应放在训练侧，而不是评估侧：

- 针对 seed29 的 mechanism windows 增加训练预算或机制窗口重复采样；
- 保持 reward/continuity 不变的前提下，提高 prepare 行为在 valid target + timing active 状态下的概率；
- 降低 imitation decay 后期过快失效的风险，重点观察 `event_prepare_prob_mean` 和 `guard_prefetch_to_prepare_count`；
- 继续使用当前 V2 selection，避免选择高 reward 但窗口级 failure 风险更高的 checkpoint。
