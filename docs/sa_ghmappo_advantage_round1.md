# SA-GHMAPPO Advantage Round 1

更新日期：2026-04-24

## 目标和边界

本轮只做 agent 侧、训练策略、checkpoint selection 和诊断增强，不修改 `VecWorkflowCoreEnv`、reward、handoff / migration 语义、adapter cache 指标或 benchmark 指标口径。

对照方法 `popularity_cache_heuristic` 仍保持 prediction-aware heuristic：它继续使用 `predicted_next_rsu` 和 `predicted_handoff_target` 做 predictive prefetch / migration prepare。

## 已实现

- `sa_ghmappo` 增加可配置 `continuity_guard_enabled` / `handoff_target_alignment_guard_enabled`。
- guard 在 predicted next RSU 与 predicted handoff target 不一致、handoff imminent 或 high-confidence handoff target 场景下，对 prefetch logit 降权、对 migration prepare logit 加权。
- guard 诊断进入 `action_info`：`guard_triggered`、`original_action`、`guarded_action`、`predicted_next_rsu_id`、`predicted_handoff_target_rsu_id`、`current_rsu_id`、`reason`。
- `sa_ghmappo` 训练增加可选 heuristic-informed imitation loss，teacher 来自 `popularity_cache_heuristic`，仅在 mechanism / handoff / adapter-missing 相关状态启用。
- imitation 记录 `teacher_action`、`student_action`、`imitation_applied`、`heuristic_imitation_match_rate`。
- 训练 profile 增加 `sa_advantage_round1`，支持 mechanism window oversampling，但 eval window 仍使用原始 selected windows。
- checkpoint selection 增加 `best_by_advantage_score.pt`，只做外部排序，不改变环境 reward。
- benchmark aggregate 增加 `comparison_against_popularity.json` 和 `sa_advantage_diagnosis.json`。
- 新增消融汇总脚本 `scripts/build_sa_advantage_ablation.py`，从四组 benchmark aggregate 生成 `ablation_summary.csv/json`。

## Checkpoint Selection

`best_by_advantage_score` 使用：

```text
total_reward
+ 40 * workflow_continuity_rate
+ 15 * handoff_ready_ratio
+ 20 * mechanism_realization_rate
- 30 * handoff_failure_rate
- 0.1 * backhaul_traffic_cost
- 0.2 * adapter_state_migration_overhead
```

该分数只用于 checkpoint 排序和选择，不写回环境 reward。

## 当前正式结果状态

本轮代码已实现并完成最小验证；增强版三 seed 正式训练和 mixed / full benchmark 仍需执行后才能声明新主方法是否真实领先。

本轮 smoke 验证产物：

- `artifacts/training/sa_advantage_round1_smoke/sa_ghmappo/sa_ghmappo_train_20260424_210845_380573_seed7/train_summary.json`
- `artifacts/training/sa_advantage_round1_smoke/sa_ghmappo/sa_ghmappo_train_20260424_210845_380573_seed7/checkpoints/best_by_advantage_score.pt`
- `artifacts/benchmarks/sa_advantage_round1_smoke/main_results_mixed_informative_20260424_211202_990677/comparison_against_popularity.json`
- `artifacts/benchmarks/sa_advantage_round1_ablation_smoke/smoke_round1_check/ablation_summary.json`

短版 `sa_advantage_round1` profile 链路验证产物：

- `artifacts/training/sa_advantage_round1_quick/sa_ghmappo/sa_ghmappo_train_20260425_122208_746305_seed7/train_summary.json`
- `artifacts/training/sa_advantage_round1_quick/sa_ghmappo/sa_ghmappo_train_20260425_122208_746305_seed7/checkpoints/best_by_advantage_score.pt`
- `artifacts/benchmarks/sa_advantage_round1_quick/main_results_mixed_informative_20260425_122226_162889/comparison_against_popularity.json`
- `artifacts/benchmarks/sa_advantage_round1_ablation_quick/quick_round1_check/ablation_summary.json`

短版 profile 结果仅用于链路诊断，不用于正式结论：`sa_ghmappo` 相比 popularity 的 `total_reward` 为 `-1.675`，`workflow_continuity_rate` 为 `-0.125`，`backhaul_traffic_cost` 为 `-96.0`；guard 触发 `6` 次，`target_mismatch_guard_count=0`。

正式训练预算已于 2026-04-25 跑满，配置为 `sa_advantage_round1`、seeds `7/13/29`、每 seed `64 episodes`、`16 updates`。

训练产物：

- seed `7`: `artifacts/training/sa_advantage_round1/sa_ghmappo/sa_ghmappo_train_20260425_122902_278394_seed7/train_summary.json`
- seed `13`: `artifacts/training/sa_advantage_round1/sa_ghmappo/sa_ghmappo_train_20260425_123152_951083_seed13/train_summary.json`
- seed `29`: `artifacts/training/sa_advantage_round1/sa_ghmappo/sa_ghmappo_train_20260425_123439_169766_seed29/train_summary.json`
- per-seed checkpoint manifest: `artifacts/training/sa_advantage_round1/seed_checkpoint_manifest_sa_advantage_round1_best_by_advantage_score.json`

正式 benchmark 产物：

- mixed informative: `artifacts/benchmarks/sa_advantage_round1/mixed_informative/main_results_mixed_informative_20260425_123738_344990/aggregate_summary.json`
- full stratified: `artifacts/benchmarks/sa_advantage_round1/full_stratified/main_results_full_stratified_20260425_123916_274224/aggregate_summary.json`

正式 benchmark 对比 popularity：

| protocol | reward delta | continuity delta | handoff failure delta | backhaul delta | ready delta | mechanism delta | guard triggers |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed_informative | -2.641111 | 0.000000 | 0.000000 | -51.555556 | -0.055556 | 0.000000 | 102 |
| full_stratified | 0.032222 | 0.055696 | -0.143519 | -4.444444 | -0.074074 | -0.018519 | 165 |

结论：`full_stratified` 下达到本轮最低成功标准，主方法 reward、continuity、handoff failure、backhaul cost 和 migration overhead 均优于 popularity；但 `handoff_ready_ratio` 和 `mechanism_realization_rate` 仍低于 popularity。`mixed_informative` 下未达到 reward 领先，虽然 continuity 和 failure 追平且 backhaul cost 更低。

可作为 pre-change 参考的最近 round1 结果：

- `mixed_informative`，`best_by_reward`：`sa_ghmappo` reward `84.104444` 高于 popularity `83.513333`，但 continuity `0.979630` 低于 popularity `1.000000`。
- `mixed_informative`，`best_by_continuity`：`sa_ghmappo` continuity `1.000000` 追平 popularity，但 reward `83.218889` 低于 popularity `83.513333`。
- `full_stratified`，`best_by_continuity`：`sa_ghmappo` reward `75.623333` 高于 popularity `74.898333`，continuity / handoff_ready / mechanism 与 popularity 持平。

因此当前优化目标是把 `best_by_reward` 的 reward 优势和 `best_by_continuity` 的 continuity 安全性合到同一个训练/选择链路中。

## 待复查问题

1. 主方法在哪些指标超过 popularity？
   - 增强版正式 benchmark 未跑完前不能下结论；已有 pre-change full-stratified 中 reward 高于 popularity，mixed 中 reward 或 continuity 只能二选一。
2. 哪些指标仍接近或落后？
   - pre-change mixed 下，`best_by_reward` 的 continuity 落后；`best_by_continuity` 的 reward 落后。
3. continuity guard 触发多少次？
   - 新字段已进入 episode summary 和 benchmark rows，需查看 `continuity_guard_trigger_count`。
4. guard 是否减少 target mismatch stall？
   - 需对比 `target_mismatch_guard_count`、stall attribution 和 continuity/handoff failure。
5. imitation 是否提升 early stability？
   - 需对比 `heuristic_imitation_match_rate`、early update reward/continuity trend。
6. 是否保持低 backhaul cost 优势？
   - 由 `comparison_against_popularity.json` 的 `backhaul_traffic_cost` delta 判断。
7. 若没有明显领先，下一步原因是什么？
   - 优先检查：guard 触发不足、teacher imitation 过强导致复制 heuristic、mechanism windows 仍不足、或 post-handoff warm-ready 条件没有成立。

## 正式运行命令

训练 guard + imitation 主配置：

```bash
python scripts/train_sa_ghmappo_real_sample.py --agent_name sa_ghmappo --profile sa_advantage_round1 --random_seed 7 --mobility_source ngsim --workflow_csv_path data/raw/workflow/alibaba2018/batch_task.csv --max_mobility_rows 2500 --max_workflows 2 --workflow_selector ordered --rsu_layout auto_dominant_tight --frame_offset 0 --window_length 24 --window_selector max_handoff_candidate --window_count 3 --window_scan_stride 2 --window_mode mixed_informative --train_window_mode sampled --max_steps 12 --min_tasks 5 --max_tasks 20 --mechanism_window_oversample_ratio 2.0 --handoff_imminent_oversample_ratio 1.5 --target_mismatch_sample_weight 1.5 --min_mechanism_activating_windows 2 --output_root artifacts/training/sa_advantage_round1
```

对 seeds `13`、`29` 重复该命令并替换 `--random_seed`。

最小消融四组：

```bash
python scripts/train_sa_ghmappo_real_sample.py --agent_name sa_ghmappo --profile formal_main_stable --random_seed 7 --no-continuity_guard_enabled --heuristic_imitation_coef 0.0 --output_root artifacts/training/sa_advantage_round1/original
python scripts/train_sa_ghmappo_real_sample.py --agent_name sa_ghmappo --profile sa_advantage_round1 --random_seed 7 --continuity_guard_enabled --handoff_target_alignment_guard_enabled --heuristic_imitation_coef 0.0 --output_root artifacts/training/sa_advantage_round1/guard
python scripts/train_sa_ghmappo_real_sample.py --agent_name sa_ghmappo --profile sa_advantage_round1 --random_seed 7 --no-continuity_guard_enabled --no-handoff_target_alignment_guard_enabled --heuristic_imitation_coef 0.12 --output_root artifacts/training/sa_advantage_round1/imitation
python scripts/train_sa_ghmappo_real_sample.py --agent_name sa_ghmappo --profile sa_advantage_round1 --random_seed 7 --continuity_guard_enabled --handoff_target_alignment_guard_enabled --heuristic_imitation_coef 0.12 --output_root artifacts/training/sa_advantage_round1/guard_plus_imitation
```

benchmark mixed：

```bash
python scripts/benchmark_main_results.py --agents sa_ghmappo reactive_greedy popularity_cache_heuristic ppo mappo --seeds 7 13 29 --window_mode mixed_informative --sa_ghmappo_checkpoint_path <sa_advantage_checkpoint> --flat_ppo_checkpoint_path <ppo_checkpoint> --flat_mappo_checkpoint_path <mappo_checkpoint> --mobility_source ngsim --workflow_csv_path data/raw/workflow/alibaba2018/batch_task.csv --max_mobility_rows 2500 --max_workflows 2 --workflow_selector ordered --rsu_layout auto_dominant_tight --frame_offset 0 --window_length 24 --window_selector max_handoff_candidate --window_count 3 --window_scan_stride 2 --max_steps 12 --min_tasks 5 --max_tasks 20 --output_root artifacts/benchmarks/sa_advantage_round1/mixed_informative
```

benchmark full：

```bash
python scripts/benchmark_main_results.py --agents sa_ghmappo reactive_greedy popularity_cache_heuristic ppo mappo --seeds 7 13 29 --window_mode full_stratified --sa_ghmappo_checkpoint_path <sa_advantage_checkpoint> --flat_ppo_checkpoint_path <ppo_checkpoint> --flat_mappo_checkpoint_path <mappo_checkpoint> --mobility_source ngsim --workflow_csv_path data/raw/workflow/alibaba2018/batch_task.csv --max_mobility_rows 2500 --max_workflows 2 --workflow_selector ordered --rsu_layout auto_dominant_tight --frame_offset 0 --window_length 24 --window_selector max_handoff_candidate --window_count 3 --window_scan_stride 2 --max_steps 12 --min_tasks 5 --max_tasks 20 --output_root artifacts/benchmarks/sa_advantage_round1/full_stratified
```

消融汇总：

```bash
python scripts/build_sa_advantage_ablation.py --variant original=<original_aggregate_summary.json> --variant guard=<guard_aggregate_summary.json> --variant imitation=<imitation_aggregate_summary.json> --variant guard_plus_imitation=<guard_plus_imitation_aggregate_summary.json> --output_root artifacts/benchmarks/sa_advantage_round1/ablation
```

## 预期 artifacts

- `artifacts/training/sa_advantage_round1/sa_ghmappo/<run_id>/train_summary.json`
- `artifacts/training/sa_advantage_round1/sa_ghmappo/<run_id>/best_checkpoint_record.json`
- `artifacts/training/sa_advantage_round1/sa_ghmappo/<run_id>/checkpoints/best_by_advantage_score.pt`
- `artifacts/benchmarks/sa_advantage_round1/<mode>/<run_id>/aggregate_summary.json`
- `artifacts/benchmarks/sa_advantage_round1/<mode>/<run_id>/benchmark_rows.csv`
- `artifacts/benchmarks/sa_advantage_round1/<mode>/<run_id>/comparison_against_popularity.json`
- `artifacts/benchmarks/sa_advantage_round1/<mode>/<run_id>/sa_advantage_diagnosis.json`
- `artifacts/benchmarks/sa_advantage_round1/ablation/<run_id>/ablation_summary.csv`
- `artifacts/benchmarks/sa_advantage_round1/ablation/<run_id>/ablation_summary.json`
