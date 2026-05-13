# Experiment Runbook Round 1

更新日期：2026-04-24

所有命令默认从仓库根目录执行。本文固定 formal experiment execution round 1 的最小正式命令，不包含 smoke 命令。

## Baseline Runner 正式命令

```bash
python scripts/run_baseline_experiment.py --config configs/experiment/baseline/minimal_ngsim_alibaba.yaml
```

用途：按配置完成 `sa_ghmappo` eval、heuristic eval、`ppo` / `mappo` per-seed train、per-seed checkpoint、per-seed eval 和 multi-seed benchmark。

预期输出目录：

```text
artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_<timestamp>/
```

关键输出：

- `comparison_summary.csv`
- `comparison_summary.json`
- `comparison_summary_detailed.json`
- `comparison_summary_by_window_class.csv`
- `run_manifest.json`
- `seed_checkpoint_manifest.json`
- `command_log.json`

## 只跑部分 Agent

只跑 trainable baseline：

```bash
python scripts/run_baseline_experiment.py --config configs/experiment/baseline/minimal_ngsim_alibaba.yaml --agents ppo mappo
```

只跑启发式 baseline：

```bash
python scripts/run_baseline_experiment.py --config configs/experiment/baseline/minimal_ngsim_alibaba.yaml --agents reactive_greedy popularity_cache_heuristic
```

只跑主方法与启发式对照：

```bash
python scripts/run_baseline_experiment.py --config configs/experiment/baseline/minimal_ngsim_alibaba.yaml --agents sa_ghmappo reactive_greedy popularity_cache_heuristic
```

旧名 `flat_ppo` 和 `flat_mappo` 仍可作为 registry alias 使用，主要用于消费历史 manifest。

## 只做 Benchmark 聚合

基于已完成 run 的 seed checkpoint manifest 重新执行 benchmark：

```bash
python scripts/benchmark_main_results.py --agents sa_ghmappo reactive_greedy popularity_cache_heuristic ppo mappo --seeds 7 13 --output_root artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark_rerun --seed_checkpoint_manifest_path artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/seed_checkpoint_manifest.json --sa_ghmappo_checkpoint_path artifacts/training/main_agents/sa_ghmappo/sa_ghmappo_train_20260415_154335_734767_seed7/checkpoints/best_by_reward.pt --flat_ppo_checkpoint_path artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/training/algo_pool/seed_7/flat_ppo/flat_ppo_train_20260424_145837_825858_seed7/checkpoints/latest.pt --flat_mappo_checkpoint_path artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/training/algo_pool/seed_7/flat_mappo/flat_mappo_train_20260424_145935_676708_seed7/checkpoints/latest.pt --mobility_source ngsim --workflow_csv_path data/raw/workflow/alibaba2018/batch_task.csv --max_mobility_rows 1500 --max_workflows 1 --workflow_selector ordered --rsu_layout auto_dominant_tight --frame_offset 0 --window_length 16 --max_steps 6 --min_tasks 5 --max_tasks 20 --window_selector max_handoff_candidate --window_count 3 --window_scan_stride 2 --window_mode mixed_informative
```

`--seed_checkpoint_manifest_path` 会让 benchmark 在 seed `7/13` 上分别使用对应 seed checkpoint。命令中的 `--flat_ppo_checkpoint_path` 和 `--flat_mappo_checkpoint_path` 是兼容参数名，只作为代表性 checkpoint 和审计 fallback；它们现在分别供 `ppo` 和 `mappo` 使用。

补充执行 full-stratified window-class 诊断：

```bash
python scripts/benchmark_main_results.py --agents sa_ghmappo reactive_greedy popularity_cache_heuristic ppo mappo --seeds 7 13 --output_root artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark_full_stratified --seed_checkpoint_manifest_path artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/seed_checkpoint_manifest.json --sa_ghmappo_checkpoint_path artifacts/training/main_agents/sa_ghmappo/sa_ghmappo_train_20260415_154335_734767_seed7/checkpoints/best_by_reward.pt --flat_ppo_checkpoint_path artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/training/algo_pool/seed_7/flat_ppo/flat_ppo_train_20260424_145837_825858_seed7/checkpoints/latest.pt --flat_mappo_checkpoint_path artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/training/algo_pool/seed_7/flat_mappo/flat_mappo_train_20260424_145935_676708_seed7/checkpoints/latest.pt --mobility_source ngsim --workflow_csv_path data/raw/workflow/alibaba2018/batch_task.csv --max_mobility_rows 1500 --max_workflows 1 --workflow_selector ordered --rsu_layout auto_dominant_tight --frame_offset 0 --window_length 16 --max_steps 6 --min_tasks 5 --max_tasks 20 --window_selector max_handoff_candidate --window_count 2 --window_scan_stride 2 --window_mode full_stratified
```

## 跳过阶段的 Runner 命令

只验证 manifest / summary 导出结构：

```bash
python scripts/run_baseline_experiment.py --config configs/experiment/baseline/minimal_ngsim_alibaba.yaml --dry_run
```

已有 checkpoint 路径写入配置且只想跑 eval + benchmark：

```bash
python scripts/run_baseline_experiment.py --config configs/experiment/baseline/minimal_ngsim_alibaba.yaml --skip_train
```

当前 `minimal_ngsim_alibaba.yaml` 中只内置了 `sa_ghmappo` checkpoint；若对 `ppo` / `mappo` 使用 `--skip_train`，需要先在配置里显式填入 checkpoint，否则 trainable baseline 会被标记为 `missing_checkpoint`。
