# v98 Hidden Holdout Execution Record

- `executed_at`: 2026-08-08
- `target_venue`: IEEE Transactions on Mobile Computing (TMC)
- `artifact_run_id`: `main_results_full_stratified_20260808_160805_690263`
- `candidate_freeze_record`: `docs/project/top_journal_v98_freeze_20260808.md`
- `candidate_manifest`: `artifacts/experiments/top_journal_v98_ucc_full_20260808/seed_checkpoint_manifest.json`
- `hidden_plan`: `configs/experiment/top_journal_v71_strict_split_20260730/hidden_holdout_window_plan.json`
- `window_independence_audit`: `artifacts/experiments/top_journal_v98_ucc_full_20260808/window_independence_formal_hidden.json`
- `hidden_rows_sha256`: `990fd0207a5756c63787e9552738ebe1e415d126801044430582f5fc7cdb1364`
- `hidden_aggregate_sha256`: `329ed76cb92158bdf1bbbe1aaf9bf638923055fb2f6c8c84544619707e2be176`
- `hidden_statistics_sha256`: `1d02d30b3eb9b3f88effe26daab41a469ef30122f0185cf2dff79b69b19bb1d0`
- `policy_after_opening`: consumed once; no checkpoint, hyperparameter, split, baseline or claim selection was changed after hidden output.

## Command

```bash
.venv/bin/python scripts/benchmark_main_results.py --agents sa_ghmappo ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl popularity_cache_heuristic --seed_checkpoint_manifest_path artifacts/experiments/top_journal_v98_ucc_full_20260808/seed_checkpoint_manifest.json --mobility_source ngsim --primary_vehicle_selection handoff_pressure --workflow_csv_path data/raw/workflow/alibaba2018/batch_task.csv --seeds 7 13 29 --max_mobility_rows 5000000 --max_workflows 2 --max_steps 22 --reward_positive_offset 0.0 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_length 24 --window_scan_stride 2 --window_mode full_stratified --window_plan_path configs/experiment/top_journal_v71_strict_split_20260730/hidden_holdout_window_plan.json --prediction_horizon 16 --min_tasks 5 --max_tasks 20 --output_root artifacts/experiments/top_journal_v98_ucc_full_20260808/benchmarks/hidden_holdout_full_stratified
```

## Result

- SA-GHMAPPO: `17.049`; Popularity: `16.814`; delta: `+0.2350`.
- Window-outer hierarchical BCa 95% CI: `[0.078333, 0.483290]`; paired win/tie/loss: `36/84/0`; Holm sign-test p=`0.0`.
- SA also exceeded PPO `2.285`, MAPPO `-2.143`, DQN `12.317`, and every other evaluated baseline on mean total reward. This record preserves the raw rows and statistics; it does not claim superiority on every mechanism metric.
