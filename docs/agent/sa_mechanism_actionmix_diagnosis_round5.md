# sa_mechanism_actionmix_diagnosis_round5

## Scope

本轮只做 telemetry / evaluator / analysis 诊断增强。

未修改：

- `VecWorkflowCoreEnv` 环境主逻辑
- reward 主公式
- action mask / handoff / cache 语义
- `sa_ghmappo` policy 网络、checkpoint selection
- benchmark split
- `popularity_cache_heuristic`
- round2 / round4 历史 artifacts

新增诊断字段来自 benchmark episode `step_trace` 中已有的 `metrics_protocol`、`action_name`、`control_action` 和 `reward_dict`。本轮没有新增训练。

## Code Changes

- `src/evaluators/main_results_support.py`
  - `summary_to_row()` 新增兼容列：`mode`、`scenario_id`、`window_tag`、`policy_name`。
  - 新增 service / cache / action mix / prefetch / migration / reward proxy 聚合列。
  - 新增列只追加到 `benchmark_rows.csv` 和 aggregate 统计，不删除或重命名旧列。
- `scripts/analyze_mechanism_actionmix_gap.py`
  - 读取 mixed/full benchmark rows。
  - 按 `mode + window_id + window_class + workflow_id + seed` 配对 `sa_ghmappo` 与 `popularity_cache_heuristic`。
  - 输出 action/cache/service/reward proxy 差异。

未修改 `src/evaluators/real_eval_support.py` 和 `src/envs/core/vec_workflow_core_env.py`。当前需要的字段已经在 evaluator 层可安全聚合。

## Artifacts

诊断 benchmark：

- Mixed: `artifacts/analysis/sa_mechanism_actionmix_diagnosis_round5/benchmark_mixed/main_results_mixed_informative_20260426_023736_742838/`
- Full: `artifacts/analysis/sa_mechanism_actionmix_diagnosis_round5/benchmark_full/main_results_full_stratified_20260426_023920_694043/`

诊断分析输出：

- `artifacts/analysis/sa_mechanism_actionmix_diagnosis_round5/actionmix_gap_summary.csv`
- `artifacts/analysis/sa_mechanism_actionmix_diagnosis_round5/mechanism_activating_policy_diff.csv`
- `artifacts/analysis/sa_mechanism_actionmix_diagnosis_round5/cache_service_breakdown.csv`
- `artifacts/analysis/sa_mechanism_actionmix_diagnosis_round5/reward_proxy_breakdown.csv`
- `artifacts/analysis/sa_mechanism_actionmix_diagnosis_round5/diagnosis_summary.json`

## Mixed Diagnosis

Mixed benchmark 复现 round2 结果：

| policy | reward | continuity | failure | ready | mechanism | backhaul |
|---|---:|---:|---:|---:|---:|---:|
| sa_ghmappo | 83.405000 | 1.000000 | 0.000000 | 0.416667 | 0.500000 | 124.444444 |
| popularity_cache_heuristic | 83.513333 | 1.000000 | 0.000000 | 0.416667 | 0.500000 | 170.666667 |
| SA - popularity | -0.108333 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -46.222223 |

按 window class 聚合后，gap 完全集中在 `mechanism_activating`：

| mode | window_tag | paired episodes | mean reward delta | losing pairs |
|---|---|---:|---:|---:|
| mixed_informative | active_non_mechanism | 6 | 0.000000 | 0 |
| mixed_informative | mechanism_activating | 12 | -0.162500 | 11 |

`-0.162500 * 12 / 18 = -0.108333`，与 mixed overall reward gap 对齐。

在 mixed 的 `mechanism_activating` 窗口中，observable service/cache 指标基本不是 gap 来源：

| metric delta, SA - popularity | mean |
|---|---:|
| adapter_miss_count | 0.000000 |
| service_wait_sum | 0.000000 |
| service_delay_sum | 0.000000 |
| service_reward_component | 0.000000 |
| delay_reward_component | 0.000000 |
| cache_reward_component | 0.000000 |
| handoff_reward_component | 0.000000 |
| failure_reward_component | 0.000000 |
| continuity_reward_component | -0.162500 |
| backhaul_traffic_cost | -69.333333 |

关键差异是 action mix：

| metric delta, SA - popularity | mean |
|---|---:|
| prefetch_attempt_count | -1.083333 |
| migration_attempt_count | +1.083333 |
| cache_admission_count | 0.000000 overall, -0.181818 on losing pairs |
| cache_noop_count | 0.000000 overall, +0.181818 on losing pairs |

结论：mixed 的剩余 `-0.108333` 不是 delay、cache miss、failure、ready 或 mechanism realization 问题，而是一个很小的 reward tie-break。SA 在 mechanism 窗口更偏 `handoff_migration_prepare`，popularity 更偏 `predictive_next_rsu_prefetch`。在这些 paired samples 上，popularity 偶尔通过 prefetch/cache admission 多拿 `continuity_bonus` 中的 `cache_result["added_new_adapter"] and cache_hit` 小额奖励；SA 虽然 backhaul 更低，但该 `backhaul_traffic_cost` 是 benchmark metric，不等价于 reward 中的 `migration_cost`，因此 mixed reward 没有体现这部分 backhaul 优势。

## Full Diagnosis

Full benchmark 也复现 round2 结果：

| policy | reward | continuity | failure | ready | mechanism | backhaul |
|---|---:|---:|---:|---:|---:|---:|
| sa_ghmappo | 76.654815 | 0.956229 | 0.097222 | 0.212963 | 0.277778 | 147.555556 |
| popularity_cache_heuristic | 75.492778 | 0.924242 | 0.180556 | 0.250000 | 0.277778 | 158.222222 |
| SA - popularity | +1.162037 | +0.031987 | -0.083334 | -0.037037 | 0.000000 | -10.666666 |

Full 的 advantage 不是来自 mechanism 子集，而是来自 `idle_or_sparse`：

| mode | window_tag | paired episodes | mean reward delta |
|---|---|---:|---:|
| full_stratified | active_non_mechanism | 18 | 0.000000 |
| full_stratified | idle_or_sparse | 18 | +4.538889 |
| full_stratified | mechanism_activating | 18 | -1.052778 |

Full mechanism 子集中最大的两条拖累样本是：

- `window_off250_len24_t297_320 / j_3 / seed 7`
- `window_off250_len24_t297_320 / j_3 / seed 29`

这两条中 SA 出现：

- `adapter_miss_count +5`
- `service_wait_sum +5`
- `handoff_failure_rate +1`
- `handoff_ready_ratio -1`
- `prefetch_attempt_count -2`
- `migration_attempt_count +5`
- `backhaul_traffic_cost -128`

这说明 full 下仍有少数 mechanism window 是真实 service/cache 兑现问题，但 full overall 被 `idle_or_sparse` 的服务成功、cache miss 降低和 reward 优势抵消并反超。

## Field Limitations

- `service_wait_sum` 是基于 `stall_occurred` 的 wait proxy，不是独立排队等待时间字段。
- `service_restart_count` 当前没有底层事件字段，诊断层只在存在 `service_restart` 时计数；本轮结果中为 0。
- `cache_eviction_count` 当前没有底层事件字段，诊断层只在存在 `cache_eviction` 时计数；本轮结果中为 0。
- reward proxy 使用现有 `reward_dict`：`service_reward`、`delay_penalty`、`cache_miss_penalty`、`migration_cost`、`continuity_bonus`、`mechanism_exploration_bonus`、`constraint_penalty`。没有重写 reward。
- full 中某些 paired episode 的 `total_reward` 差异还受每步 positive offset / episode length 影响；mixed 的剩余 gap 已由 exported reward components 中的 `continuity_reward_component` 对齐解释。

## Answer To Round5 Question

是否成功定位 mixed 的 `-0.108333` reward gap：成功。

定位结果：

- 不是 cache miss / cold start：mixed mechanism 下 `adapter_miss_count`、`adapter_warm_hit_count`、`service_wait_sum`、`delay_reward_component` 都持平。
- 不是 failure / readiness：mixed 下 continuity、failure、ready、mechanism realization 都持平。
- 不是 backhaul 差：SA backhaul 明显更低，但该 benchmark metric 没有转化成 reward component advantage。
- 主要是 action-mix tie-break：SA 少做 prefetch、多做 prepare；popularity 偶尔通过 prefetch/cache admission 获得小额 `continuity_bonus`，造成 mechanism rows 平均 `-0.162500`，最终 weighted 成 mixed overall `-0.108333`。

下一轮建议：

- 不建议优先改 selector：round4 已证明 warm-start/selection 没找到更优 checkpoint。
- 不建议改 reward：本轮已定位 reward component 来源，改 reward 会改变正式口径。
- 不建议继续盲训：gap 是小额 action-mix tie-break，不是训练预算无法解释的黑箱差距。
- 若进入优化，最小方向是 policy-side action bias / auxiliary：在 mechanism window 中保留 `prepare` 优势的同时，允许安全的 limited prefetch/cache admission，不要用 eval-time override。
- 如果要继续补 telemetry，优先补明确的 `cache_admission_added_new_adapter_count`、`reward_positive_offset_proxy`、`service_restart` 和 `cache_eviction` 底层透传；这属于 logging，不应改环境语义。

## Validation

已执行：

```bash
python -m py_compile src\evaluators\main_results_support.py scripts\analyze_mechanism_actionmix_gap.py
python scripts\benchmark_main_results.py --agents sa_ghmappo popularity_cache_heuristic --seed_checkpoint_manifest_path artifacts\training\sa_mechanism_policy_round2\seed_checkpoint_manifest_sa_mechanism_policy_round2_best_by_round2_mechanism_score.json --seeds 7 13 29 --max_mobility_rows 2500 --max_workflows 2 --max_steps 12 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_length 24 --window_count 3 --window_scan_stride 2 --window_selector max_handoff_candidate --window_mode mixed_informative --min_tasks 5 --max_tasks 20 --output_root artifacts\analysis\sa_mechanism_actionmix_diagnosis_round5\benchmark_mixed
python scripts\benchmark_main_results.py --agents sa_ghmappo popularity_cache_heuristic --seed_checkpoint_manifest_path artifacts\training\sa_mechanism_policy_round2\seed_checkpoint_manifest_sa_mechanism_policy_round2_best_by_round2_mechanism_score.json --seeds 7 13 29 --max_mobility_rows 2500 --max_workflows 2 --max_steps 12 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_length 24 --window_count 3 --window_scan_stride 2 --window_selector max_handoff_candidate --window_mode full_stratified --min_tasks 5 --max_tasks 20 --output_root artifacts\analysis\sa_mechanism_actionmix_diagnosis_round5\benchmark_full
python scripts\analyze_mechanism_actionmix_gap.py --mixed_benchmark_dir artifacts\analysis\sa_mechanism_actionmix_diagnosis_round5\benchmark_mixed\main_results_mixed_informative_20260426_023736_742838 --full_benchmark_dir artifacts\analysis\sa_mechanism_actionmix_diagnosis_round5\benchmark_full\main_results_full_stratified_20260426_023920_694043 --output_dir artifacts\analysis\sa_mechanism_actionmix_diagnosis_round5
```

验证结果：

- py_compile 通过。
- mixed 复现 round2：SA `83.405`，popularity `83.513`。
- full 复现 round2：SA `76.655`，popularity `75.493`。
- 诊断 CSV/JSON 已生成。
