# v100 Execution Record

- `executed_at`: 2026-08-08
- `target_venue`: IEEE Transactions on Mobile Computing (TMC)
- `candidate`: `top_journal_mechanism_v100_urgency_safe_resource_mappo`
- `candidate_freeze`: `docs/project/top_journal_v100_freeze_20260808.md`
- `manifest`: `artifacts/experiments/top_journal_v100_urgency_safe_resource_full_20260808/seed_checkpoint_manifest.json`
- `manifest_sha256`: `418657c69731b45cfca1df0b45ecc1757adcd555cbde5b27106bc4ef6f7b9d09`

## Training Command

The following command was run once for each seed `7`, `13` and `29`; `{seed}` was replaced by the seed value.

```bash
./.venv/bin/python scripts/train_sa_ghmappo_real_sample.py --agent_name sa_ghmappo --profile top_journal_mechanism_v100_urgency_safe_resource_mappo --random_seed {seed} --episodes 256 --update_every 8 --train_window_count 20 --max_steps 22 --max_mobility_rows 5000000 --max_workflows 2 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_plan_path configs/experiment/top_journal_v71_strict_split_20260730/train_window_plan.json --eval_window_plan_path configs/experiment/top_journal_v71_strict_split_20260730/dev_window_plan.json --window_mode full_stratified --primary_vehicle_selection handoff_pressure --min_tasks 5 --max_tasks 20 --reward_positive_offset 0.0 --post_training_audit_mode compact --output_root artifacts/experiments/top_journal_v100_urgency_safe_resource_full_20260808/training_seed_{seed}
```

## Formal Benchmark

```bash
./.venv/bin/python scripts/benchmark_main_results.py --agents sa_ghmappo ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl popularity_cache_heuristic --seed_checkpoint_manifest_path artifacts/experiments/top_journal_v100_urgency_safe_resource_full_20260808/seed_checkpoint_manifest.json --mobility_source ngsim --primary_vehicle_selection handoff_pressure --workflow_csv_path data/raw/workflow/alibaba2018/batch_task.csv --seeds 7 13 29 --max_mobility_rows 5000000 --max_workflows 2 --max_steps 22 --reward_positive_offset 0.0 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_length 24 --window_scan_stride 8 --window_mode full_stratified --window_plan_path configs/experiment/top_journal_v71_strict_split_20260730/formal_window_plan.json --prediction_horizon 16 --min_tasks 5 --max_tasks 20 --output_root artifacts/experiments/top_journal_v100_urgency_safe_resource_full_20260808/benchmarks/formal_full_stratified
```

## Statistics

```bash
./.venv/bin/python scripts/analyze_top_journal_statistics.py --rows_path artifacts/experiments/top_journal_v100_urgency_safe_resource_full_20260808/benchmarks/formal_full_stratified/main_results_full_stratified_20260808_183339_526215/benchmark_rows.csv --candidate_agent sa_ghmappo --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl popularity_cache_heuristic --pair_keys seed window_id workflow_id prediction_setting_id robustness_setting_id scalability_setting_id ablation_label --outer_cluster_keys window_id --inner_cluster_keys seed workflow_id --ci_method bca --bootstrap_samples 5000 --random_seed 100 --output_root artifacts/experiments/top_journal_v100_urgency_safe_resource_full_20260808/statistics/formal_full_stratified
```

## Support and Audit

The compact prediction support command used the frozen five-window support plan at `artifacts/experiments/top_journal_v100_urgency_safe_resource_full_20260808/support/compact_support_window_plan_5.json`, with agents `sa_ghmappo popularity_cache_heuristic`, seeds `7 13 29`, and output root `.../support/prediction_robustness_compact`. The full noise sweep was attempted but terminated without output after excessive runtime; it is explicitly excluded from claims.

```bash
./.venv/bin/python scripts/audit_artifact_integrity.py --run_root artifacts/experiments/top_journal_v100_urgency_safe_resource_full_20260808 --output_dir artifacts/experiments/top_journal_v100_urgency_safe_resource_full_20260808/audit_artifact_integrity
```

Audit result: `passed=true`, `missing_reference_count=0`, `json_error_count=0`.
