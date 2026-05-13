# SA-GHMAPPO Advantage Round 1 Mechanism Diagnosis

本诊断只读取 `sa_advantage_round1` 已完成训练和正式 benchmark 产物；未修改环境语义、reward、benchmark split 或 popularity baseline。

## 读取产物

- Training manifest: `artifacts/training/sa_advantage_round1/seed_checkpoint_manifest_sa_advantage_round1_best_by_advantage_score.json`
- Mixed benchmark: `artifacts/benchmarks/sa_advantage_round1/mixed_informative/main_results_mixed_informative_20260425_123738_344990/`
- Full benchmark: `artifacts/benchmarks/sa_advantage_round1/full_stratified/main_results_full_stratified_20260425_123916_274224/`
- V2 selection summary: `artifacts/training/sa_advantage_round1/mechanism_advantage_selection_summary.json`

## Mixed Informative 差距来源

`mixed_informative` 中 `sa_ghmappo` 的平均 reward 比 `popularity_cache_heuristic` 低 `2.641111`，但不是 delay 或 cache miss 造成：

| 项 | SA - popularity |
| --- | ---: |
| total_reward | -2.641111 |
| end_to_end_workflow_delay | 0.000000 |
| cache_miss_penalty_sum | 0.000000 |
| delay_penalty_sum | 0.000000 |
| backhaul_traffic_cost | -51.555556 |
| adapter_state_migration_overhead | +0.091111 |
| continuity_bonus_sum | -0.994444 |
| mechanism_exploration_bonus_sum | -1.333333 |

主要拖累集中在 `seed=29` 的 `mechanism_activating` 窗口。按 seed 看，`seed=7` 和 `seed=13` 的平均 reward delta 都只有 `-0.125`，`seed=29` 为 `-7.673333`。按窗口类别看，`active_non_mechanism` 为 `0.0`，`mechanism_activating` 为 `-3.961667`。

最差场景集中在：

| window | workflow | seed | reward delta | ready delta | backhaul delta | migration overhead delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `window_off246_len24_t293_316` | `j_8` | 29 | -17.870000 | -0.500000 | -80.000000 | +0.820000 |
| `window_off248_len24_t295_318` | `j_8` | 29 | -17.720000 | -0.500000 | -16.000000 | +0.820000 |
| `window_off248_len24_t295_318` | `j_3` | 29 | -6.300000 | 0.000000 | -128.000000 | 0.000000 |
| `window_off246_len24_t293_316` | `j_3` | 29 | -4.150000 | 0.000000 | -64.000000 | 0.000000 |

这说明 mixed 的 reward 落后主要来自机制窗口内的 continuity/mechanism bonus 没完全兑现，以及少量 migration overhead，而不是主方法 backhaul 更高。事实上 SA 的 backhaul 更低，但 reward 里的机制奖励和 continuity bonus 对 prepare/handoff-ready 的兑现更敏感。

## Full Stratified 为什么能赢

`full_stratified` 中 SA 达到最低成功标准：reward、continuity、handoff failure、backhaul cost 和 migration overhead 都优于 popularity。

| 指标 | sa_ghmappo | popularity | SA - popularity |
| --- | ---: | ---: | ---: |
| total_reward | 75.525000 | 75.492778 | +0.032222 |
| workflow_continuity_rate | 0.979938 | 0.924242 | +0.055696 |
| handoff_failure_rate | 0.037037 | 0.180556 | -0.143519 |
| backhaul_traffic_cost | 153.777778 | 158.222222 | -4.444444 |
| adapter_state_migration_overhead | 1.216667 | 1.226667 | -0.010000 |

优势不是来自机制窗口。按 window class 的 reward delta：

| window_class | SA - popularity reward |
| --- | ---: |
| `mechanism_activating` | -5.061667 |
| `active_non_mechanism` | 0.000000 |
| `idle_or_sparse` | +5.158333 |

也就是说 full 的胜利主要来自 `idle_or_sparse` 上 SA 明显少 stall、少 handoff failure。step trace 汇总显示：full 中 popularity 在 `idle_or_sparse` 出现 `48` 次 stall，SA 只有 `3` 次；对应 handoff failure rate 也从 popularity 的 `0.180556` 降到 SA 的 `0.037037`。

按 seed 看，`seed=7` 和 `seed=13` reward delta 都是 `+1.213889`，`seed=29` 是 `-2.331111`。因此 full 的优势在 2/3 seeds 上稳定，但 seed29 仍是机制兑现短板。

## Ready / Realization 仍低的原因

当前没有证据表明 prediction/handoff candidate 没产生，也没有证据表明 ready 或 realization 指标漏记。更像是策略动作在机制窗口没有稳定兑现到足够的 prepare/prefetch 强度。

从 selected update 的训练评估诊断看：

| seed | selected update | valid_handoff_target_rate | timing_active_step_count | gate_pass_rate | event_prepare_prob_mean | guard_prefetch_to_prepare_count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 6 | 0.404762 | 34 | 0.190476 | 0.464480 | 30 |
| 13 | 7 | 0.404762 | 34 | 0.190476 | 0.450924 | 30 |
| 29 | 15 | 0.404762 | 34 | 0.190476 | 0.139703 | 1 |

candidate 和 timing 信号在三个 seed 上基本一致，seed29 的主要差异是 event prepare probability 和 guard-to-prepare 转换明显低。这指向 agent 策略/训练稳定性，而不是窗口缺信号或 valid target 过度过滤。

正式 benchmark step trace 也支持这个判断：

| mode | agent | prefetch_requested | migration_prepare_requested | handoff_ready | mechanism_bonus_awarded |
| --- | --- | ---: | ---: | ---: | ---: |
| mixed | sa_ghmappo | 13 | 65 | 10 | 78 |
| mixed | popularity | 27 | 75 | 12 | 102 |
| full | sa_ghmappo | 17 | 108 | 13 | 125 |
| full | popularity | 39 | 120 | 18 | 159 |

`mechanism_realization_rate` 是 episode-level boolean，由 validated prefetch / handoff_ready / migration_during_handoff 是否出现推导；它能反映“是否触发过”，但不反映触发强度。因此 mixed 中 realization 可以追平 `0.5`，但 reward 仍因 mechanism bonus 和 continuity bonus 次数不足而落后。当前不建议改指标口径，只应把这一点作为诊断解释。

## 结论

- `mixed_informative` 的主要问题是 seed29 在 mechanism windows 上 prepare/prefetch 强度不足，导致 mechanism/continuity bonus 少，而不是 delay、cache miss 或 backhaul。
- `full_stratified` 的优势主要来自 idle/sparse 场景更少 stall 和更低 handoff failure，方向与 reward、continuity、backhaul 一致。
- `handoff_ready_ratio` 和 `mechanism_realization_rate` 低于 popularity 的直接原因是 SA 选择 prepare/prefetch 的频率和时机还不够稳定，尤其 seed29；不是 benchmark split、candidate 产生或指标漏记问题。
- 新的 mechanism-aware checkpoint score 已补入代码和 post-hoc selector，但现有训练预算内 v2 选择与 v1 相同，说明 selection 不能单独修复该短板。
