# Runbook

## Git LFS 数据恢复

`data/` 由 Git LFS 管理。新主机首次检出后执行：

```bash
git lfs install
git lfs pull
python scripts/check_data_ready.py
```

`git lfs pull` 完成前不要运行真实数据训练或 benchmark。当前仓库包含 NGSIM、Alibaba、LuST 和 model-cache audit metadata；highD 尚未提供，不阻塞当前 `NGSIM + Alibaba` 正式主线。

## MAPPO 对照协议

当前 `mappo` paper-grade 对照必须使用 controller-level CTDE + `aggregation_reason_weighted_controller_ppo_v3`。正式 final-submission loop 会审计 `baseline_protocol_versions.mappo`，要求 checkpoint 配置包含 `head_credit_enabled=True`、`head_credit_protocol=aggregation_reason_weighted_controller_ppo_v3`、`slow_policy_credit_floor=0.25`、`fast_policy_credit_floor=0.10`、`event_policy_credit_floor=0.12`、`slow_entropy_credit_floor=0.20`、`fast_entropy_credit_floor=0.08`、`event_entropy_credit_floor=0.12`、`event_advantage_blend=0.85`。旧 pre-v3/pre-head-credit MAPPO 结果只作归档，不再进入新版论文主表。

所有命令默认从仓库根目录执行。

## 导师汇报材料

- 讲稿与证据索引：`docs/project/advisor_report_briefing_20260621.md`
- 可编辑 PPT：`outputs/ppo_mec_advisor_report_20260621.pptx`

汇报材料不是新实验入口。更新图表前必须先核对 `docs/project/top_journal_readiness_audit_20260621.md` 和对应 raw statistics；不得用 legacy `paper_claim_ready=true` 覆盖 strict-full 结论。

## Strict-full v8 冻结协议

冻结 outcome-blind、跨 split 间隔至少 24 frames 的窗口计划：

```bash
.venv/bin/python scripts/freeze_strict_split_protocol.py --output_dir configs/experiment/top_journal_v8_strict_split_20260621 --max_mobility_rows 10000 --window_length 24 --window_scan_stride 2 --minimum_gap_frames 24 --windows_per_split 20 --mechanism_windows_per_split 6 --active_non_mechanism_windows_per_split 2 --random_seed 7
```

正式 benchmark 必须用显式计划，不再通过 rank offset 模拟独立 holdout：

```bash
.venv/bin/python scripts/benchmark_main_results.py --seed_checkpoint_manifest_path artifacts/experiments/top_journal_closed_loop/strict_full_v8_dev_screen_20260621_v2/seed_checkpoint_manifest.json --seeds 7 13 29 41 53 --window_plan_path configs/experiment/top_journal_v8_strict_split_20260621/formal_window_plan.json --window_mode full_stratified --primary_vehicle_selection handoff_pressure
```

strict statistics：

```bash
.venv/bin/python scripts/analyze_top_journal_statistics.py --rows_path <benchmark_rows.csv> --candidate_agent sa_ghmappo --outer_cluster_keys window_id --inner_cluster_keys seed workflow_id --ci_method bca --bootstrap_samples 5000 --random_seed 7 --output_root <output_dir>
```

hidden 只能在候选和 formal 冻结后开启一次。不要修改 `split_manifest.json` 的冻结内容；开启事件写入独立 execution record。当前 hidden 已 consumed，后续开发不得再次用它筛选候选。完整记录见 `strict_full_v8_execution_record_20260621.md`。

### Strict-full v8 support suite

先 dry-run 检查命令、manifest 和 hidden 禁用规则：

```bash
.venv/bin/python scripts/run_strict_full_v8_support_suite.py --dry_run --run_id strict_full_v8_support_<date>_v1
```

正式补齐 v8-current support suite：

```bash
.venv/bin/python scripts/run_strict_full_v8_support_suite.py --run_id strict_full_v8_support_<date>_v1 --seed_checkpoint_manifest_path artifacts/experiments/top_journal_closed_loop/strict_full_v8_dev_screen_20260621_v2/seed_checkpoint_manifest.json --window_plan_path configs/experiment/top_journal_v8_strict_split_20260621/formal_window_plan.json --guard_manifest_path configs/ablation_checkpoint_manifest_v8_guard_attribution.json
```

输出默认写入 `artifacts/experiments/top_journal_support_suite/strict_full_v8_support_<date>_v1/`，包括 `support_gate_report.json`、各 support benchmark 原始 rows/aggregate、paired statistics 和命令 manifest。该 suite 使用 formal/support 证据补齐 v8-current，不允许指向 hidden window plan。

### v9 Pareto-safe 候选

v9 只允许在 train/dev 或新冻结 future-validation split 上开发；当前 hidden 已 consumed，不得用于 checkpoint ranking 或筛选：

```bash
.venv/bin/python scripts/run_top_journal_closed_loop.py --run_id top_journal_mechanism_v9_pareto_safe_dev_<date> --seeds 7 13 29 41 53 --sa_profile top_journal_mechanism_v9_pareto_safe --mappo_baseline_profile mappo_strong_audit --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure --window_mode_for_training full_stratified --train_window_plan_path configs/experiment/top_journal_v8_strict_split_20260621/train_window_plan.json --eval_window_plan_path configs/experiment/top_journal_v8_strict_split_20260621/dev_window_plan.json
```

promotion 前必须检查 `best_by_pareto_safe_score.pt` 对 learned baselines 的 reward / continuity 正向性，以及 handoff failure / backhaul 的预注册 non-inferiority；若 reward 明显下降，v9 只能归档为 safety trade-off candidate。

### v10 MAPPO-core RL 候选

v10 继承 v9 的 safety boundary，但把 MAPPO 强对照中的 controller-level CTDE credit assignment 显式迁入主方法候选：`aggregation_reason_weighted_controller_ppo_v3`、三控制头 policy credit floors、entropy floors/scales 和 event advantage blend。该候选仍保留 SA-GHMAPPO 的 graph/surrogate/critic，但降低 imitation 与 mechanism auxiliary 权重，避免把 idle/sparse 行为主要交给手写规则。

```bash
.venv/bin/python scripts/run_top_journal_closed_loop.py --run_id top_journal_mechanism_v10_mappo_rl_dev_<date> --seeds 7 13 29 41 53 --sa_profile top_journal_mechanism_v10_mappo_rl --mappo_baseline_profile mappo_strong_audit --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure --window_mode_for_training full_stratified --train_window_plan_path configs/experiment/top_journal_v8_strict_split_20260621/train_window_plan.json --eval_window_plan_path configs/experiment/top_journal_v8_strict_split_20260621/dev_window_plan.json
```

promotion 前必须按 window class 单独审计：机制窗口收益不能坍缩，idle/sparse 对 `popularity_cache_heuristic` 的 reward gap 应显著收窄，同时 `best_by_pareto_safe_score.pt` 必须满足 PPO failure/backhaul non-inferiority。当前 hidden 仍不得用于筛选或调参。

### v11 MAPPO reward-first dev 候选

v11 以 v8 strict-full mechanism scaffold 为稳定底座，迁入 MAPPO controller-level head-credit、entropy floors/scales 和 event advantage blend，并把 checkpoint selection 改为 reward-first。推理期只在 v11 checkpoint 且 `window_class=idle_or_sparse` 时启用 no-RSU local fallback；机制窗口保持 MAPPO 主策略和 vehicle-only fallback，避免 v2-style 全局 no-RSU fallback 吃掉机制窗口收益。

```bash
.venv/bin/python scripts/run_top_journal_closed_loop.py --run_id top_journal_mechanism_v11_mappo_reward_dev_<date> --seeds 7 13 29 41 53 --sa_profile top_journal_mechanism_v11_mappo_reward --mappo_baseline_profile mappo_strong_audit --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure --window_mode_for_training full_stratified --train_window_plan_path configs/experiment/top_journal_v8_strict_split_20260621/train_window_plan.json --eval_window_plan_path configs/experiment/top_journal_v8_strict_split_20260621/dev_window_plan.json
```

当前 full-dev evidence：`artifacts/experiments/top_journal_mappo_reward_full_dev_v11_20260716/main_results_full_stratified_window_gate_full/main_results_full_stratified_20260716_181112_383674/aggregate_summary.json`。主方法 total reward `79.4944`，高于 `popularity_cache_heuristic=79.46875` 和所有 learned baselines；`sa_advantage_diagnosis.blockers=[]`。该结果只能作为 dev-stage evidence，promotion 仍需要新冻结 future-validation 或重新审查，不能使用已 consumed hidden。

### v12 learned MAPPO option gate dev 候选

v12 在 v11 MAPPO-core reward-first checkpoint 上 warm-start，新增 policy-side option head，在 `accept_mappo`、`popularity_safe`、`no_rsu_local`、`mechanism_prepare` 间学习选择。机制窗口 preserve MAPPO 主策略，idle/sparse 允许 learned `popularity_safe` option 接管；该路径不再依赖 v11 evaluator-side window override。

单 seed warm-start 训练模板：

```bash
.venv/bin/python scripts/train_sa_ghmappo_real_sample.py --agent_name sa_ghmappo --profile top_journal_mechanism_v12_learned_option --random_seed <seed> --warm_start_checkpoint_path <v11_seed_best_by_reward.pt> --max_mobility_rows 10000 --max_workflows 2 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_plan_path configs/experiment/top_journal_v8_strict_split_20260621/train_window_plan.json --window_mode full_stratified --train_window_count 20 --primary_vehicle_selection handoff_pressure --min_tasks 5 --max_tasks 20 --output_root artifacts/training/top_journal_mechanism_v12_learned_option
```

全量 dev benchmark 模板：

```bash
.venv/bin/python scripts/benchmark_main_results.py --agents sa_ghmappo reactive_greedy popularity_cache_heuristic ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --seed_checkpoint_manifest_path artifacts/experiments/top_journal_mappo_reward_v12_learned_option_20260717/seed_checkpoint_manifest.json --seeds 7 13 29 41 53 --max_mobility_rows 10000 --max_workflows 2 --max_steps 16 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_count 20 --window_mode full_stratified --window_plan_path configs/experiment/top_journal_v8_strict_split_20260621/dev_window_plan.json --primary_vehicle_selection handoff_pressure --min_tasks 5 --max_tasks 20 --output_root artifacts/experiments/top_journal_mappo_reward_v12_learned_option_20260717/main_results_full_stratified_mech_preserve
```

当前 full-dev evidence：`artifacts/experiments/top_journal_mappo_reward_v12_learned_option_20260717/main_results_full_stratified_mech_preserve/main_results_full_stratified_20260717_115754_212344/aggregate_summary.json`。主方法 total reward `79.5934`，高于 `popularity_cache_heuristic=79.46875`、`ppo=77.18775`、`mappo=72.6328` 和所有其他对照；`sa_advantage_diagnosis.blockers=[]`。该结果只能作为 dev-stage evidence，promotion 仍需要新冻结 future-validation 或重新审查，不能使用已 consumed hidden。

### v13 partial-reward-decoupled MAPPO dev 候选

v13 在 v12 learned option 基础上加入 event-head 与 option-head 的 partial-reward-decoupled credit，目标是让 MAPPO 的机制动作从 handoff readiness、机制成功和窗口 context 中学习，而不是继续扩大 hard rule。v13 的 `best_by_reward` 在全量审计中停留在 update 0 warm-start；复现实验应使用训练后的 `latest.pt`，closed-loop 入口也对 v13 采用 latest-first checkpoint priority。

单 seed warm-start 训练模板：

```bash
.venv/bin/python scripts/train_sa_ghmappo_real_sample.py --agent_name sa_ghmappo --profile top_journal_mechanism_v13_prd_option --random_seed <seed> --warm_start_checkpoint_path <v12_seed_best_by_reward.pt> --max_mobility_rows 10000 --max_workflows 2 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_plan_path configs/experiment/top_journal_v8_strict_split_20260621/train_window_plan.json --window_mode full_stratified --train_window_count 20 --primary_vehicle_selection handoff_pressure --min_tasks 5 --max_tasks 20 --output_root artifacts/training/top_journal_mechanism_v13_prd_event_full
```

全量 dev benchmark 模板必须使用 latest manifest：

```bash
.venv/bin/python scripts/benchmark_main_results.py --agents sa_ghmappo reactive_greedy popularity_cache_heuristic ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --seed_checkpoint_manifest_path artifacts/experiments/top_journal_prd_option_v13_20260717/seed_checkpoint_manifest_prd_event_latest.json --seeds 7 13 29 41 53 --max_mobility_rows 10000 --max_workflows 2 --max_steps 16 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_count 20 --window_mode full_stratified --window_plan_path configs/experiment/top_journal_v8_strict_split_20260621/dev_window_plan.json --primary_vehicle_selection handoff_pressure --min_tasks 5 --max_tasks 20 --output_root artifacts/experiments/top_journal_prd_option_v13_20260717/main_results_full_stratified_latest
```

当前 full-dev evidence：`artifacts/experiments/top_journal_prd_option_v13_20260717/main_results_full_stratified_latest/main_results_full_stratified_20260717_124815_375515/aggregate_summary.json`。主方法 total reward `79.64465`，高于 v12/best-by-reward `79.5934`、`popularity_cache_heuristic=79.46875`、`ppo=77.18775`、`mappo=72.6328` 和所有其他对照；strongest-other margin 为 `+0.17590`。该结果只能作为 dev-stage evidence，promotion 仍需要新冻结 future-validation 或重新审查，不能使用已 consumed hidden。

### v17/v18 time-audited future-validation

future-validation split 必须同时排除 `frame_offset` 与 `time_index_start/end` 重叠；只按 frame offset 审计的旧 future split 不得作为 independent evidence。

冻结 time-audited future split：

```bash
.venv/bin/python scripts/freeze_future_validation_split.py --max_mobility_rows 50000 --window_count 20 --mechanism_windows 6 --active_non_mechanism_windows 2 --minimum_gap_frames 24
```

复核与历史 train/dev/formal/hidden 的独立性：

```bash
.venv/bin/python scripts/audit_window_independence.py --formal_summary configs/experiment/top_journal_v8_strict_split_20260621/hidden_holdout_window_plan.json --holdout_summary configs/experiment/top_journal_v17_future_validation_time_audited_20260717/future_validation_window_plan.json --minimum_gap_frames 24 --output artifacts/audits/top_journal_v17_future_validation_time_audited_20260717/hidden_vs_future_independence.json
```

v17 time-audited future-validation benchmark：

```bash
.venv/bin/python scripts/benchmark_main_results.py --agents sa_ghmappo popularity_cache_heuristic ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl reactive_greedy --seed_checkpoint_manifest_path artifacts/experiments/top_journal_dag_aware_option_v17_20260717/seed_checkpoint_manifest_dag_aware_option_full.json --seeds 7 13 29 41 53 --mobility_source ngsim --primary_vehicle_selection handoff_pressure --workflow_csv_path data/raw/workflow/alibaba2018/batch_task.csv --max_mobility_rows 50000 --max_workflows 2 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_selector max_handoff_candidate --window_mode full_stratified --window_plan_path configs/experiment/top_journal_v17_future_validation_time_audited_20260717/future_validation_window_plan.json --window_length 24 --window_scan_stride 2 --window_count 20 --max_steps 16 --min_tasks 5 --max_tasks 20 --output_root artifacts/experiments/top_journal_dag_aware_option_v17_20260717/future_validation_time_audited_full_stratified
```

paired statistics：

```bash
.venv/bin/python scripts/analyze_top_journal_statistics.py --rows_path <benchmark_rows.csv> --candidate_agent sa_ghmappo --baseline_agents popularity_cache_heuristic ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl reactive_greedy --outer_cluster_keys window_id --inner_cluster_keys seed workflow_id --bootstrap_samples 5000 --output_root artifacts/analysis/top_journal_v17_future_validation_time_audited_statistics_20260717
```

当前结论：v18 counterfactual option-credit dev 结果不晋级；v17 time-audited future-validation 均值第一，但相对 popularity 的 reward CI 跨 0，不能写成 TMC-ready。

## Supervised Handoff Predictor

训练薄 supervised predictor：

```bash
.venv/bin/python scripts/train_supervised_handoff_predictor.py --train_window_plan_path configs/experiment/top_journal_v8_strict_split_20260621/train_window_plan.json --dev_window_plan_path configs/experiment/top_journal_v8_strict_split_20260621/dev_window_plan.json --horizon 3 --epochs 8 --random_seed 7
```

该脚本只使用 mobility future labels 生成 `next_rsu`、`first_handoff_target`、`handoff_within_horizon` 和 `handoff_eta_steps`，不读取 reward、action、checkpoint outcome 或 hidden 结果。输出包括 `supervised_handoff_predictor.pt`、`predictor_metrics_manifest.json` 和 `predictor_quality_rows.csv`。

使用冻结 predictor 训练 / benchmark：

```bash
.venv/bin/python scripts/train_sa_ghmappo_real_sample.py --agent_name sa_ghmappo --profile top_journal_mechanism_v8_strict_full --predictor_kind supervised --predictor_checkpoint_path <supervised_handoff_predictor.pt> --window_plan_path configs/experiment/top_journal_v8_strict_split_20260621/train_window_plan.json --primary_vehicle_selection handoff_pressure
.venv/bin/python scripts/benchmark_main_results.py --agents sa_ghmappo dt_handoff_drl ppo --predictor_kind supervised --predictor_checkpoint_path <supervised_handoff_predictor.pt> --window_plan_path configs/experiment/top_journal_v8_strict_split_20260621/formal_window_plan.json --window_mode full_stratified --primary_vehicle_selection handoff_pressure
.venv/bin/python scripts/benchmark_prediction_robustness.py --agents sa_ghmappo ppo --predictor_kind supervised --predictor_checkpoint_path <supervised_handoff_predictor.pt> --primary_vehicle_selection handoff_pressure
```

论文表述边界：该层是 short-horizon handoff anticipation / lightweight DT-style predictive state snapshot，不是完整 digital twin 系统；正式主张必须来自冻结 predictor checkpoint、quality report、重训后的 SA-GHMAPPO checkpoint 和 formal/future-validation 原始结果。

## 文献表自动审计

```bash
.venv/bin/python scripts/audit_literature_reference_table.py --table_path docs/project/literature_reference_table.md --output_dir artifacts/analysis/literature_table_audit_20260621
```

默认将标题、DOI、URL 重复或非绝对 HTTP(S) 链接视为失败；显式 `待核验` 项会列入报告但不自动失败。投稿前需要清空待核验项时追加 `--fail_on_unverified`。该脚本只检查 URL 结构，不声称探测出版社页面实时可达性。

## 最小验证

```bash
python scripts/smoke_test.py
python -m pytest tests/test_env_contract.py
```

## 数据准备检查

```bash
python scripts/check_data_ready.py
python scripts/validate_dataset_source_declarations.py
python scripts/audit_hf_model_cache_sources.py
python scripts/run_ngsim_sample.py --max_rows 500
python scripts/run_alibaba_sample.py --limit_jobs 3 --min_tasks 5 --max_tasks 20
python scripts/scan_ngsim_handoff_windows.py --max_mobility_rows 1500 --window_length 24 --stride 2 --top_k 5
```

## 真实 Sample Dry-Run

```bash
python scripts/run_real_sample_dryrun.py --mobility_source ngsim --workflow_source alibaba --max_mobility_rows 1500 --max_workflows 3 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_selector max_handoff_candidate --window_length 24 --max_steps 12
```

## 主方法

训练：

```bash
python scripts/train_sa_ghmappo_real_sample.py --agent_name sa_ghmappo --profile formal_main --random_seed 7
```

顶刊候选机制稳定复跑：
```bash
python scripts/train_sa_ghmappo_real_sample.py --agent_name sa_ghmappo --profile top_journal_mechanism_v1 --random_seed 7 --mobility_source ngsim --primary_vehicle_selection handoff_pressure --workflow_csv_path data/raw/workflow/alibaba2018/batch_task.csv --max_mobility_rows 2500 --max_workflows 2 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_selector max_handoff_candidate --window_length 24 --window_count 3 --window_scan_stride 2 --window_mode mixed_informative --max_steps 16 --min_tasks 5 --max_tasks 20 --output_root artifacts/training/top_journal_mechanism_v1
```

说明：`top_journal_mechanism_v1` 会默认开启 mechanism auxiliary retention、慢衰减 imitation、机制窗口重采样和 target-mismatch 加权。该 profile 生成的 checkpoint 仍必须经过多 seed benchmark 和 checkpoint audit 后才能进入论文表。


评估：

```bash
python scripts/eval_sa_ghmappo_real_sample.py --agent_name sa_ghmappo --checkpoint_path artifacts/training/main_agents/sa_ghmappo/<run_id>/checkpoints/best_by_continuity.pt
```

## 对照算法池

当前 live 可训练 learned 对照算法是 `ppo`、`mappo`、`dqn`、`ddqn`、`dueling_dqn`、`dueling_ddqn`、`qmix`、`controller_mat`、`dag_offload_drl`、`cache_offload_drl` 和 `dt_handoff_drl`，paper-grade 默认主对照使用 `ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl`，其余 DQN-family 变体必须先通过 duplicate trace audit。`ippo` 是 contract-blocked diagnostic agent：当前 single-wrapper decision stream 不足以支撑独立 IPPO。`mappo` 是 controller-level CTDE baseline，`qmix` 是 controller-level value-decomposition baseline，`controller_mat` 是 controller-level transformer CTDE baseline，三者都不是 vehicle-agent / RSU-agent full MARL wrapper。`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl` 是领域专项 learned baseline，不使用 SA-GHMAPPO 专属 graph/surrogate/guard 机制。`flat_ppo` / `flat_mappo` 只表示历史 artifact run 名称，不再作为 live agent 注册。

训练：

```bash
python scripts/train_algo_pool_real_sample.py --agent_name ppo --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name mappo --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name dqn --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name dueling_dqn --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name qmix --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name controller_mat --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name dag_offload_drl --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name cache_offload_drl --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name dt_handoff_drl --profile smoke
```

MAPPO 正式强对照训练默认使用：

```bash
python scripts/train_algo_pool_real_sample.py --agent_name mappo --profile mappo_strong_audit
```

评估：

```bash
python scripts/eval_algo_pool_real_sample.py --agent_name ppo --checkpoint_path artifacts/training/algo_pool/ppo/<run_id>/checkpoints/latest.pt
python scripts/eval_algo_pool_real_sample.py --agent_name reactive_greedy
```

正式单 seed baseline 训练形状：

```bash
python scripts/train_algo_pool_real_sample.py --agent_name ppo --profile baseline_safe --episodes 48 --update_every 6 --batch_size 32 --learning_rate 1e-4 --clip_ratio 0.1 --entropy_coef 0.003 --value_coef 0.7 --random_seed 7 --mobility_source ngsim --workflow_csv_path data/raw/workflow/alibaba2018/batch_task.csv --max_mobility_rows 2500 --max_workflows 2 --workflow_selector ordered --rsu_layout auto_dominant_tight --frame_offset 0 --window_length 24 --window_selector max_handoff_candidate --window_count 3 --window_scan_stride 2 --window_mode mixed_informative --max_steps 12 --min_tasks 5 --max_tasks 20 --output_root artifacts/training/algo_pool_formal_round1
```

将 `--agent_name` 替换为 `mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl`，并对 seeds `7 13 29` 重复执行。`ippo` 当前只允许 diagnostic 复核，不用于 paper-grade baseline 训练。

## Baseline 闭环

Smoke：

```bash
python scripts/run_baseline_experiment.py --config configs/experiment/baseline/smoke.yaml
```

Formal round1 minimal：

```bash
python scripts/run_baseline_experiment.py --config configs/experiment/baseline/minimal_ngsim_alibaba.yaml
```

核心输出：

- `artifacts/experiments/baseline/<run_id>/comparison_summary.csv`
- `artifacts/experiments/baseline/<run_id>/comparison_summary.json`
- `artifacts/experiments/baseline/<run_id>/comparison_summary_detailed.json`
- `artifacts/experiments/baseline/<run_id>/comparison_summary_by_window_class.csv`
- `artifacts/experiments/baseline/<run_id>/run_manifest.json`
- `artifacts/experiments/baseline/<run_id>/seed_checkpoint_manifest.json`
- `artifacts/experiments/baseline/<run_id>/command_log.json`

## 主结果 Benchmark

单主方法：

```bash
python scripts/benchmark_main_results.py --agents sa_ghmappo --sa_ghmappo_checkpoint_path artifacts/training/main_agents/sa_ghmappo/<run_id>/checkpoints/best_by_continuity.pt --seeds 7 13 29 --max_mobility_rows 2500 --max_workflows 2 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_selector max_handoff_candidate --window_length 24 --window_count 3 --window_scan_stride 2 --max_steps 12
```

最小对照 benchmark：

```bash
python scripts/benchmark_main_results.py --agents sa_ghmappo ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --seed_checkpoint_manifest_path <manifest_with_learned_checkpoints> --seeds 7 --max_workflows 1 --window_count 1 --max_steps 3
```

`--flat_ppo_checkpoint_path` 和 `--flat_mappo_checkpoint_path` 是历史兼容参数名；当前正式 benchmark 优先使用 seed checkpoint manifest。

## 消融 / 鲁棒性 / 可扩展性

消融：

```bash
python scripts/benchmark_ablation.py --ablation_labels sa_ghmappo_full no_prediction no_graph_encoder no_hierarchy no_event_agent no_adapter_prefetch no_dag_dependency_aware no_uncertainty_signal
```

预测鲁棒性：

```bash
python scripts/benchmark_prediction_robustness.py --agents sa_ghmappo ppo --sa_ghmappo_checkpoint_path <main_ckpt> --flat_ppo_checkpoint_path <ppo_ckpt>
```

系统鲁棒性：

```bash
python scripts/benchmark_robustness.py --agents sa_ghmappo ppo --sa_ghmappo_checkpoint_path <main_ckpt> --flat_ppo_checkpoint_path <ppo_ckpt>
```

可扩展性：

```bash
python scripts/benchmark_scalability.py --agents sa_ghmappo ppo --sa_ghmappo_checkpoint_path <main_ckpt> --flat_ppo_checkpoint_path <ppo_ckpt>
```

## Round1 当前记录

Round1 状态、机制诊断和复跑命令：

- `docs/experiment_status_round1.md`
- `docs/mechanism_activation_check_round1.md`
- `docs/experiment_runbook_round1.md`
- `docs/continuity_resolution_round1.md`

当前三 seed 统一比较 manifest：

```text
artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/seed_checkpoint_manifest_formal_round1_3seed.json
```

当前 aggregate 输出：

```text
artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark_formal_round1_3seed/main_results_mixed_informative_20260424_190319_732417/aggregate_summary.json
artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark_formal_round1_3seed_full_stratified/main_results_full_stratified_20260424_190503_729168/aggregate_summary.json
```

## 产物确认

训练和 benchmark 后优先检查：

- `train.csv`
- `eval.csv`
- `summary.json`
- `train_summary.json`
- `checkpoints/latest.pt`
- `aggregate_summary.json`
- `benchmark_rows.csv`
- `run_manifest.json`

当前已整理过的历史 artifact 结论统一看 `docs/project/ARTIFACT_RECORDS.md`。
 

## SA Advantage Round1 Mechanism V2

Mechanism-aware checkpoint selection from completed `sa_advantage_round1` runs:

```bash
python scripts/select_sa_mechanism_advantage_checkpoints.py
```

Mixed benchmark:

```bash
python scripts/benchmark_main_results.py --agents sa_ghmappo reactive_greedy popularity_cache_heuristic --seed_checkpoint_manifest_path artifacts/training/sa_advantage_round1/seed_checkpoint_manifest_sa_advantage_round1_best_by_mechanism_advantage_score.json --seeds 7 13 29 --max_mobility_rows 2500 --max_workflows 2 --max_steps 12 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_length 24 --window_count 3 --window_scan_stride 2 --window_selector max_handoff_candidate --window_mode mixed_informative --min_tasks 5 --max_tasks 20 --output_root artifacts/benchmarks/sa_advantage_round1_mechanism_v2/mixed_informative
```

Full benchmark:

```bash
python scripts/benchmark_main_results.py --agents sa_ghmappo reactive_greedy popularity_cache_heuristic --seed_checkpoint_manifest_path artifacts/training/sa_advantage_round1/seed_checkpoint_manifest_sa_advantage_round1_best_by_mechanism_advantage_score.json --seeds 7 13 29 --max_mobility_rows 2500 --max_workflows 2 --max_steps 12 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_length 24 --window_count 3 --window_scan_stride 2 --window_selector max_handoff_candidate --window_mode full_stratified --min_tasks 5 --max_tasks 20 --output_root artifacts/benchmarks/sa_advantage_round1_mechanism_v2/full_stratified
```

Reports:

- `docs/agent/sa_advantage_round1_mechanism_diagnosis.md`
- `docs/agent/sa_advantage_round1_mechanism_improvement_report.md`

## HF Model-Cache Transaction-Aligned Local Experiment

用途：把 Hugging Face model-cache 审计 manifest 中的真实文件大小投影为本地 adapter cache size profile，并在 `NGSIM + Alibaba` 主线上跑一轮与 Transactions model caching/offloading 论文口径更接近的本地对比。该入口不会下载 HF 原始文件，也不能声明为真实 VEC cache request trace。

最小本地适应轮：

```bash
python scripts/run_hf_model_cache_transaction_experiment.py --train_agents ppo --sa_checkpoint_path artifacts/training/main_agents/sa_ghmappo/sa_ghmappo_train_20260415_154335_734767_seed7/checkpoints/best_by_reward.pt --seeds 7 --episodes 6 --update_every 2 --batch_size 8 --max_mobility_rows 1500 --max_workflows 1 --window_count 2 --window_length 24 --window_mode mixed_informative --max_steps 8 --rsu_adapter_slots 2
```

Checkpoint sanity check：

```bash
python scripts/run_hf_model_cache_transaction_experiment.py --skip_training --sa_checkpoint_path artifacts/training/main_agents/sa_ghmappo/sa_ghmappo_train_20260424_183117_679100_seed7/checkpoints/best_by_continuity.pt --ppo_checkpoint_path artifacts/training/algo_pool_formal_round1/flat_ppo/flat_ppo_train_20260424_190032_617002_seed7/checkpoints/latest.pt --mappo_checkpoint_path artifacts/training/algo_pool_formal_round1/flat_mappo/flat_mappo_train_20260424_190126_588082_seed7/checkpoints/latest.pt --seeds 7 --max_mobility_rows 1500 --max_workflows 1 --window_count 2 --window_length 24 --window_mode mixed_informative --max_steps 8 --rsu_adapter_slots 2
```

核心输出：

- `hf_model_cache_adapter_catalog.json`
- `hf_projection_mapping.csv`
- `convergence_rewards.csv`
- `algorithm_comparison.csv`
- `aggregate_summary.json`
- `hf_model_cache_transaction_round1_report.md`

## Top Journal Closed Loop

用途：把顶刊路线的训练、baseline 重训、seed checkpoint manifest、mixed/full benchmark 和 gate report 固化为同一入口，避免手工挑选 checkpoint 后口径漂移。

Quick 链路验证：

```bash
python scripts/run_top_journal_closed_loop.py --quick --seeds 7 --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure
```

正式复跑入口：

```bash
python scripts/run_top_journal_closed_loop.py --seeds 7 13 29 --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure
```

核心输出：

- `seed_checkpoint_manifest.json`
- `benchmarks/<mode>/main_results_*/aggregate_summary.json`
- `gate_report.json`
- `gate_summary.csv`
- `command_log.json`

说明：

- `--quick` 只验证链路可用性，`paper_claim_ready=false`，不得写成论文结论。
- 正式 claim 必须使用非 quick、多 seed、mixed_informative + full_stratified gate 通过后的 artifact。
- `paper_claim_ready=true` 还要求 `formal_contract.ready=true`：至少 3 个 seed、正式训练/窗口预算不低于默认值、`primary_vehicle_selection=handoff_pressure`，并同时包含 `mixed_informative` 与 `full_stratified`。
- 顶刊主线使用 `handoff_pressure` 绑定主 vehicle，保证 `max_handoff_candidate` 窗口中的 handoff 压力进入 workflow 主体；兼容脚本默认 `stable_first` 仅用于历史/对照协议。
- 正式复跑中断后可用同一 `--run_id` 加 `--resume_training` 复用已完成 checkpoint，只补缺失 seed/agent，再继续生成 manifest 与 benchmark。
- SA checkpoint manifest 默认优先选择 `best_by_reward_tiebreak_score_path`，该选择策略保留 continuity/failure/backhaul guardrails 后再按 reward tie-break；不要手工换成单一机制分数 checkpoint。

当前正式可引用产物：

- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_formal_20260505_v2/gate_report.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_formal_20260505_v2/gate_summary.csv`

## Top Journal Learned-Baseline Strict Gate

用途：按顶刊主 claim 口径，将 `popularity_cache_heuristic` / `reactive_greedy` 降级为 supplementary reference，主通过条件只面向当前 contract 下可辩护的 learned baselines。默认 paper-grade set 为 `ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl`；`ippo` 只能 diagnostic，`ddqn` / `dueling_ddqn` 必须先通过 duplicate trace audit 才能作为独立补充。

当前正式 learned-baseline gate：

```text
artifacts/experiments/top_journal_learned_baseline_suite/top_journal_learned_baseline_formal_20260505_v1/learned_baseline_gate_report.json
```

当前扩展 learned-baseline gate（补充 Dueling-DQN / Dueling-DDQN）：

```text
artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_gate_20260506/learned_baseline_gate_report.json
artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_holdout_offset3_20260506/learned_baseline_gate_report.json
```

若 base manifest 缺少 `mappo` / `dqn` / `dueling_dqn` / `qmix` / `controller_mat` / `dag_offload_drl` / `cache_offload_drl` / `dt_handoff_drl` checkpoint，不要使用 `--skip_training`；让 suite 自动补训缺失 learned baselines。不要把 `ippo` 加入 paper-grade gate；若为复现旧 IPPO artifact 必须传 `--allow_contract_blocked_baselines`，且该 run 不能 `paper_claim_ready=true`。

复用已有 manifest 只重跑 gate：

```bash
python scripts/run_top_journal_learned_baseline_suite.py --run_id <run_id> --base_manifest_path <seed_checkpoint_manifest.json> --skip_training --output_root artifacts/experiments/top_journal_sa_iteration
```

## Top Journal Mechanism v3 Eval-Bias Candidate

用途：从 formal_v2 权重派生启用 inference-calibrated latency fallback 的 SA checkpoint manifest，用于候选验证。该结果不能在未补齐 holdout/support suite 前替代 formal_v2 paper-grade 主表。

生成 eval-bias manifest：

```bash
python scripts/build_top_journal_eval_bias_manifest.py --base_manifest_path artifacts/experiments/top_journal_learned_baseline_suite/top_journal_learned_baseline_formal_20260505_v1/seed_checkpoint_manifest_learned_baselines.json --output_root artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias --label v3_eval_bias
```

候选 learned gate：

```bash
python scripts/run_top_journal_learned_baseline_suite.py --run_id top_journal_mechanism_v3_eval_bias_learned_gate_20260505 --base_manifest_path artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias/seed_checkpoint_manifest_v3_eval_bias_learned_baselines.json --skip_training --learned_baseline_agents ippo ppo mappo dqn ddqn --output_root artifacts/experiments/top_journal_sa_iteration
```

注意：

- clean retrain `top_journal_mechanism_v3` 当前未超过 supplementary `popularity_cache_heuristic`，不要引用为主结果升级。
- `top_journal_mechanism_v3_eval_bias` 当前只作为候选增强 artifact；论文最终主表仍优先引用 formal_v2 / learned-baseline strict gate。
- 上述旧命令仅用于历史复现。当前代码需要额外传 `--allow_contract_blocked_baselines` 才能诊断性运行 `ippo`；旧 MAPPO 数值不能代表当前 controller-level CTDE MAPPO。当前 paper-grade learned set 使用默认 `ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl`。

## Top Journal Support Suite Formal v2

当前正式主 gate：

```bash
python scripts/run_top_journal_closed_loop.py --seeds 7 13 29 --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure
```

formal v2 支撑实验已冻结在：

```text
artifacts/experiments/top_journal_support_suite/top_journal_support_formal_20260505_v2/support_gate_report.json
```

重建 paper export：

```bash
python scripts/export_paper_artifacts.py --mixed_summary_path artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_formal_20260505_v2/benchmarks/mixed_informative/main_results_mixed_informative_20260505_131333_536820/aggregate_summary.json --full_summary_path artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_formal_20260505_v2/benchmarks/full_stratified/main_results_full_stratified_20260505_131343_689261/aggregate_summary.json --gate_report_path artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_formal_20260505_v2/gate_report.json --output_root artifacts/experiments/top_journal_support_suite/top_journal_support_formal_20260505_v2/paper
```

重建 paired statistics：

```bash
python scripts/analyze_top_journal_statistics.py --rows_path <benchmark_rows.csv> --candidate_agent sa_ghmappo --baseline_agents popularity_cache_heuristic ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl reactive_greedy --outer_cluster_keys window_id --inner_cluster_keys seed workflow_id --ci_method bca --output_root <statistics_output_root>
```

训练 current-contract ablation manifest：

```bash
python scripts/run_top_journal_ablation_training.py --run_id top_journal_ablation_formal_20260505_v2 --full_seed_manifest_path artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_formal_20260505_v2/seed_checkpoint_manifest.json --variants no_prediction no_graph_encoder no_hierarchy no_event_agent no_adapter_prefetch no_dag_dependency_aware no_uncertainty_signal --seeds 7 13 29 --episodes 96 --update_every 4 --batch_size 32 --max_steps 16 --train_window_count 5 --max_mobility_rows 2500 --max_workflows 2 --workflow_csv_path data/raw/workflow/alibaba2018/batch_task.csv --workflow_selector ordered --rsu_layout auto_dominant_tight --window_selector max_handoff_candidate --window_length 24 --window_scan_stride 2 --window_mode mixed_informative --min_tasks 5 --max_tasks 20 --output_root artifacts/experiments/top_journal_support_suite --resume
```

注意：

- 支撑 benchmark 必须传 `--seed_checkpoint_manifest_path` 和 `--primary_vehicle_selection handoff_pressure`，否则不能和 formal 主表视为同一 contract。
- `no_prediction` ablation manifest 需要包含 `predictor_kwargs.disable_prediction_output=true`，否则只是禁用 policy prediction feature，不是真正 no-prediction benchmark。
- `no_dag_dependency_aware` 和 `no_uncertainty_signal` 不能声明为单独显著 reward 来源，具体边界看 `support_gate_report.json`。
## Top Journal v3 Eval-Bias Guarded-Prefetch Refresh

Current strong-candidate refresh commands:

```bash
python scripts/run_top_journal_learned_baseline_suite.py --run_id top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_gate_20260506 --base_manifest_path artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_gate_20260506/seed_checkpoint_manifest_learned_baselines.json --skip_training --output_root artifacts/experiments/top_journal_sa_iteration
python scripts/run_top_journal_learned_baseline_suite.py --run_id top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_holdout_offset3_20260506 --base_manifest_path artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_gate_20260506/seed_checkpoint_manifest_learned_baselines.json --skip_training --output_root artifacts/experiments/top_journal_sa_iteration --window_rank_offset 3
python scripts/build_top_journal_eval_bias_ablation_manifest.py
```

Notes:

- `--window_rank_offset` 只表示 ranked-window sensitivity，不能单独证明 independent holdout；历史 offset-3 结果与 formal 时间窗口存在重叠。
- Current v3 candidate has formal + holdout + latency fallback ablation support, but remains an inference-calibrated formal_v2-weight result rather than clean retrain.
- To reproduce pre-audit 20260506 diagnostic gates, pass `--allow_contract_blocked_baselines --learned_baseline_agents ippo ppo mappo dqn ddqn`; such runs are diagnostic-only and cannot be promoted to paper-ready.
- Do not promote `top_journal_mechanism_v4_prepare_eval_bias`; prediction robustness screening was negative.

## Top Journal Final Submission Loop

用途：按最终交稿口径执行 learned-primary gate。`popularity_cache_heuristic` 和 `reactive_greedy` 只作为 supplementary heuristic reference，不作为主 claim 的阻塞条件。

警告：当前 final-submission loop 的 legacy offset gate 不检查 frame interval 独立性，其 `paper_claim_ready=true` 不能直接升级为 TMC-ready。正式审查必须另跑下述 strict protocol。

### Strict non-overlap formal/holdout

先用 `--enforce_non_overlapping_selection` 生成 formal；再将 formal 的 `aggregate_summary.json` 传给 holdout 的 `--exclude_window_plan_path`，同时设置 `--holdout_min_gap_frames` 和非重叠选择。mixed/full 必须分别统计。

```bash
python scripts/benchmark_main_results.py --help
python scripts/audit_window_independence.py --formal_summary <formal_full_aggregate_summary.json> --holdout_summary <holdout_full_aggregate_summary.json> --minimum_gap_frames 0 --output artifacts/analysis/<run_id>/window_independence.json
```

Artifact 完整性审计：

```bash
python scripts/audit_artifact_integrity.py --run_root <closed_loop_root> --run_root <final_submission_root> --run_root <ablation_root> --output_dir artifacts/analysis/<run_id>_integrity
shasum -a 256 --check --quiet artifacts/analysis/<run_id>_integrity/sha256_manifest.txt
```

LuST 为二维轨迹，必须使用 `--rsu_layout auto_grid_tight`；`auto_dominant_tight` 的一维线性 RSU 可能造成全程无 association，应视为无效配置。

当前 final-submission 复跑入口：

```bash
python scripts/run_top_journal_final_submission_loop.py --run_id <new_run_id> --base_manifest_path artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v7_latency_fallback_20260528_v1/seed_checkpoint_manifest.json --force_retrain_learned --resume_training --resume_benchmark --resume_support --command_retries 2 --baseline_episodes 96 --baseline_update_every 6 --baseline_batch_size 32 --minimum_reward_delta 0.5 --holdout_offsets 3 --seeds 7 13 29 --primary_vehicle_selection handoff_pressure --window_mode_for_training full_stratified
```

旧 repaired-baseline run 使用旧 final run 中已等预算训练的 checkpoint 作为 base manifest，只重跑修复后的 benchmark/gate/support，已被 clean retrain run 取代：

```bash
python scripts/run_top_journal_final_submission_loop.py --run_id final_submission_repaired_baselines_20260507_v1 --base_manifest_path artifacts/experiments/top_journal_final_submission/final_submission_clean_equal_budget_20260506_v1/learned_suites/final_submission_clean_equal_budget_20260506_v1_iter1_formal/seed_checkpoint_manifest_learned_baselines.json --skip_training --command_retries 1 --minimum_reward_delta 0.5 --holdout_offsets 3
```

Legacy canonical clean retrain run（可复现，但未通过 strict reviewer protocol）：

```bash
python scripts/run_top_journal_final_submission_loop.py --run_id final_submission_v7_latency_fallback_20260528_v1 --base_manifest_path artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v7_latency_fallback_20260528_v1/seed_checkpoint_manifest.json --force_retrain_learned --resume_training --resume_benchmark --resume_support --command_retries 2 --baseline_episodes 96 --baseline_update_every 6 --baseline_batch_size 32 --minimum_reward_delta 0.5 --holdout_offsets 3 --seeds 7 13 29 --primary_vehicle_selection handoff_pressure --window_mode_for_training full_stratified
```

Comparison report package:
```bash
python scripts/build_top_journal_comparison_report.py --final_run_root artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1
```

断点续跑：

- `--resume_training`：复用同一 suite run 中已完成的 per-agent/per-seed `train_summary.json` 和 checkpoint。
- `--resume_benchmark`：复用已完成 benchmark mode，只补跑缺失 mode。
- `--resume_support`：复用已完成 prediction / robustness / scalability support summary。
- `--command_retries N`：对 Python/Torch 偶发 runtime crash 做命令级重试。

当前正式产物：

- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/final_submission_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/top_journal_comparison_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/top_journal_comparison_report.md`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/paper_ready/paper_ready_main_comparison.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/paper_ready/paper_ready_paired_reward_statistics.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/paper_ready/paper_ready_support_reward_statistics.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/paper_ready/paper_ready_report.md`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/learned_suites/final_submission_v7_latency_fallback_20260528_v1_iter1_formal/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/learned_suites/final_submission_v7_latency_fallback_20260528_v1_iter1_holdout_offset3/learned_baseline_gate_report.json`

Legacy gate 结论：

- `target_reached=true`
- `paper_claim_ready=true`
- comparison report `review_ready=true`
- paper-ready package `paper_ready_package_ready=true`
- formal 与 offset=3 sensitivity gate 均通过；offset=3 不得再称 independent holdout。
- 当前 canonical learned baseline set 为 `ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl`。
- `formal_training_provenance.passed=true`，`record_count=27`，说明 formal learned checkpoint 来自本次 final suite clean retrain。
- 正式 paired statistics 使用 hierarchical bootstrap：`window_id` 是 outer cluster，`seed workflow_id` 是 inner cluster；同时报告 percentile/BCa CI、窗口层效应量、sign test 和 Holm 校正。旧 `--cluster_keys seed window_id workflow_id` 只保留兼容，不再作为 paper-ready 主口径。
- prediction support 的 setting-level dominance 只要求 `learned_prediction` 和 `noisy_prediction`；`no_prediction` 与 `oracle_prediction` 保留为诊断设置，不能写成全面预测条件优势。
- 旧 `final_submission_clean_equal_budget_20260506_v1` 已作废；不要引用其 IPPO/PPO/MAPPO 或 DDQN duplicate trace 结果作为 paper-grade 证据。
