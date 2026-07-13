# Strict-Full v8 Execution Record

- `recorded_at`: `2026-06-21T20:33:44+08:00`
- `candidate_profile`: `top_journal_mechanism_v8_strict_full`
- `split_protocol`: `strict_split_v1_20260621`
- `statistics_protocol`: `hierarchical_window_bootstrap_v1_20260621`
- `candidate_checkpoint_manifest`: `artifacts/experiments/top_journal_closed_loop/strict_full_v8_dev_screen_20260621_v2/seed_checkpoint_manifest.json`
- `candidate_seeds`: `7, 13, 29, 41, 53`
- `git_base_commit`: `0db0d9175ecbc2f5db615caa7cf5835150cea746`
- `record_status`: `reconstructed_from_in-thread_execution_and_aggregate_metadata`

用途：记录 strict-full v8 的候选冻结、formal、一次性 hidden 和外部 LuST 执行链。该记录是在运行完成后由会话命令与 aggregate metadata 重建，不冒充原始终端逐字日志；因此本轮最多记为 `E2_ARTIFACT_AUDITED`，不能记为 `E3_REPRODUCED`。

## 候选冻结

- dev 只进行了两轮候选筛选；v2 为最终冻结候选，之后未再修改 profile、checkpoint、split 或统计脚本。
- v8 关闭 v7 latency fallback；仅在当前 adapter 已 warm、且没有 distinct handoff target 时启用 steady-RSU soft bias。它不修改 reward、action contract 或环境。
- 最终 profile 参数记录于 `configs/experiment/top_journal_mechanism_v8_strict_full.yaml`；五个 seed 的主方法与全部 learned baseline checkpoint 记录于上述 manifest。

冻结 split 的等价重建命令：

```bash
.venv/bin/python scripts/freeze_strict_split_protocol.py --output_dir configs/experiment/top_journal_v8_strict_split_20260621 --max_mobility_rows 10000 --window_length 24 --window_scan_stride 2 --minimum_gap_frames 24 --windows_per_split 20 --mechanism_windows_per_split 6 --active_non_mechanism_windows_per_split 2 --random_seed 7
```

## Formal

- `opened_after_candidate_freeze`: `true`
- `window_plan`: `configs/experiment/top_journal_v8_strict_split_20260621/formal_window_plan.json`
- `benchmark_run_id`: `main_results_full_stratified_20260621_025440_591857`
- `output_root`: `artifacts/experiments/strict_full_v8_formal_all_baselines_20260621_v1`
- `episode_count`: `2400` (`20 windows x 5 seeds x 2 workflows x 12 agents`)
- `statistics_output`: `artifacts/analysis/strict_full_v8_formal_all_baselines_statistics_20260621_v1`

等价 benchmark 命令的关键参数：

```bash
.venv/bin/python scripts/benchmark_main_results.py --agents sa_ghmappo ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl popularity_cache_heuristic reactive_greedy --seed_checkpoint_manifest_path artifacts/experiments/top_journal_closed_loop/strict_full_v8_dev_screen_20260621_v2/seed_checkpoint_manifest.json --seeds 7 13 29 41 53 --max_mobility_rows 10000 --max_workflows 2 --max_steps 24 --workflow_selector ordered --primary_vehicle_selection handoff_pressure --window_plan_path configs/experiment/top_journal_v8_strict_split_20260621/formal_window_plan.json --window_mode full_stratified --output_root artifacts/experiments/strict_full_v8_formal_all_baselines_20260621_v1
```

## Hidden holdout one-time opening

- `sealed_in_frozen_manifest`: `true`
- `opened_at`: `2026-06-21T20:13:53+08:00`
- `open_count`: `1`
- `open_condition`: candidate v2 frozen and formal reward CI positive against all learned baselines
- `candidate_changed_after_open`: `false`
- `window_plan`: `configs/experiment/top_journal_v8_strict_split_20260621/hidden_holdout_window_plan.json`
- `benchmark_run_id`: `main_results_full_stratified_20260621_201353_886213`
- `output_root`: `artifacts/experiments/strict_full_v8_hidden_holdout_20260621_v1`
- `episode_count`: `2400`
- `statistics_output`: `artifacts/analysis/strict_full_v8_hidden_all_baselines_statistics_20260621_v1`

等价 benchmark 命令与 formal 相同，仅将 `--window_plan_path` 改为 `hidden_holdout_window_plan.json`，并将 `--output_root` 改为 `artifacts/experiments/strict_full_v8_hidden_holdout_20260621_v1`。冻结 `split_manifest.json` 保持不变；本节是独立开启记录。

## External LuST support

- 首次默认 layout 只得到 2 个无 handoff 窗口，判为无效诊断，不用于论文结论。
- 修正后使用 `auto_grid_tight`，得到 4 个独立 mechanism windows；因外层窗口少于 12，只能作为低功效 supporting evidence。
- `benchmark_run_id`: `main_results_full_stratified_20260621_202424_612488`
- `output_root`: `artifacts/experiments/strict_full_v8_external_lust_grid_20260621_v2`
- `statistics_output`: `artifacts/analysis/strict_full_v8_external_lust_grid_statistics_20260621_v2`

## 统计与完整性

formal、hidden 和 LuST 都使用：

```bash
.venv/bin/python scripts/analyze_top_journal_statistics.py --rows_path <benchmark_rows.csv> --candidate_agent sa_ghmappo --outer_cluster_keys window_id --inner_cluster_keys seed workflow_id --ci_method bca --bootstrap_samples 5000 --random_seed 7 --output_root <statistics_output>
```

完整性审计：

```bash
.venv/bin/python scripts/audit_artifact_integrity.py --run_root <dev_candidate_root> --run_root <formal_benchmark_root> --run_root <formal_statistics_root> --run_root <hidden_benchmark_root> --run_root <hidden_statistics_root> --run_root <lust_benchmark_root> --run_root <lust_statistics_root> --output_dir artifacts/audits/strict_full_v8_integrity_20260621
```

结果：`passed=true`，`inventory_file_count=11457`，`missing_reference_count=0`，`json_error_count=0`。SHA-256 清单位于 `artifacts/audits/strict_full_v8_integrity_20260621/sha256_manifest.txt`。

## Provenance limitation

本记录可复核运行语义和输出，但不是运行时自动保存的原始 argv/stdout/stderr。因此后续正式 canonical wrapper 必须在运行开始时自动写 `command.json`、环境版本、Git dirty state 和 stdout/stderr log；在独立目录重跑并逐字段复核前不得声称 `E3_REPRODUCED`。
