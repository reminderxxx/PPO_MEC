# PPO_MEC

PPO_MEC 是面向 AI-driven VEC 的研究原型，主线围绕跨 RSU 连续 DAG workflow 执行、车载 base model 与路侧 adapter cache 协同、handoff 状态迁移、surrogate prediction 和多时间尺度控制。

当前正式数据主线是 `NGSIM + Alibaba`。`LuST` 与 `highD` 保留 provider / 检查骨架，但不阻塞正式主线。

数据源声明统一维护在 `docs/project/DATASET_SOURCES.md` 和 `configs/data/dataset_sources.json`。当前已审计并 metadata-only 接入 Hugging Face model-cache 候选全集，用于 catalog/report 审计和后续 file-size profile 设计；不会自动下载模型文件，也不会替换正式 benchmark 默认 cache 行为。接入边界见 `configs/data/hf_model_cache_integration_plan.json` 和 `docs/agent/hf_model_cache_dataset_audit_round14_report.md`。

## 当前模型层

主方法：

- `sa_ghmappo`: `Surrogate-Assisted Graph Hierarchical Multi-Agent PPO`

方向匹配对照算法池：

- 可训练 learned 对照：`ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl`；`ddqn` / `dueling_ddqn` 仅在 duplicate trace audit 通过时作为可选补充。
- contract-blocked diagnostic 对照：`ippo`。当前 single-wrapper decision stream 不能支撑 paper-grade independent IPPO；`mappo` 已实现为 controller-level CTDE baseline，并且当前 paper-grade 协议要求启用 aggregation-reason controller head-credit，避免三控制头共享错误 credit；`controller_mat` 已实现为 controller-level transformer CTDE baseline，二者都不是 vehicle-agent / RSU-agent full MARL wrappers；`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl` 分别作为 DAG/cache/DT 领域专项 learned baseline。
- 非学习启发式对照：`reactive_greedy`、`popularity_cache_heuristic`
- 历史 artifact 路径中仍可能出现 `flat_ppo` / `flat_mappo` run 名称，但它们不再是 live agent 名称。
- TD3 / SAC / MADDPG 当前不进入 live registry；后续接入前必须先冻结匹配的 observation/action contract。`qmix` 已按 controller-level value-decomposition contract 接入，不是 vehicle-agent / RSU-agent full QMIX。

## Agent 结构

`src/agents/` 只按算法分文件：

- `base_agent.py`
- `registry.py`
- `sa_ghmappo_agent.py`
- `sa_ghmappo_core.py`
- `ippo_agent.py`
- `ppo_agent.py`
- `mappo_agent.py`
- `dqn_agent.py`
- `reactive_greedy_agent.py`
- `popularity_cache_heuristic_agent.py`

`registry.py` 直接导入算法文件。PPO / MAPPO 不再通过 `ppo_family.py` 或分类 package 组织。

## 目录概览

- `src/envs/`：核心环境、预测层和 Gym wrapper
- `src/envs/specs/action_schema.py`：语义动作 schema、mask 和 action adapter
- `src/data/`：mobility、workflow 和 model catalog 数据层
- `src/encoders/`：DAG、RSU 状态、flat semantic 和融合编码器
- `src/agents/`：主方法和对照算法 agent 接入
- `src/trainers/`：训练驱动协议
- `src/evaluators/`：benchmark、checkpoint 和真实 sample 辅助模块
- `scripts/`：数据检查、dry-run、训练、评估和 benchmark 脚本
- `artifacts/`：训练 checkpoint、benchmark 报告和论文表格产物
- `docs/project/`：长期项目文档入口

## 最小验证

```bash
python scripts/smoke_test.py
python -m pytest tests/test_env_contract.py
```

真实数据链路最小检查：

```bash
python scripts/run_ngsim_sample.py --max_rows 500
python scripts/run_alibaba_sample.py --limit_jobs 3 --min_tasks 5 --max_tasks 20
python scripts/run_real_sample_dryrun.py --mobility_source ngsim --workflow_source alibaba --max_mobility_rows 1500 --max_workflows 3 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_selector max_handoff_candidate --window_length 24 --max_steps 12
```

## 训练与评估

主方法训练：

```bash
python scripts/train_sa_ghmappo_real_sample.py --agent_name sa_ghmappo --profile formal_main --random_seed 7
```

对照算法训练：

```bash
python scripts/train_algo_pool_real_sample.py --agent_name ppo --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name dqn --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name dueling_dqn --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name controller_mat --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name dag_offload_drl --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name cache_offload_drl --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name dt_handoff_drl --profile smoke
```

对照算法评估：

```bash
python scripts/eval_algo_pool_real_sample.py --agent_name ppo --checkpoint_path artifacts/training/algo_pool/ppo/<run_id>/checkpoints/latest.pt
python scripts/eval_algo_pool_real_sample.py --agent_name reactive_greedy
```

## Benchmark

```bash
python scripts/benchmark_main_results.py --agents sa_ghmappo ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --sa_ghmappo_checkpoint_path <main_ckpt> --seed_checkpoint_manifest_path <manifest_with_learned_checkpoints> --seeds 7 13 29 --max_mobility_rows 2500 --max_workflows 2 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_selector max_handoff_candidate --window_length 24 --window_count 3 --window_scan_stride 2 --max_steps 12
```

`--flat_ppo_checkpoint_path` 和 `--flat_mappo_checkpoint_path` 是历史兼容参数名；当前 paper-grade 主表优先使用 seed checkpoint manifest 管理 learned baseline checkpoint。

## Baseline 闭环

```bash
python scripts/run_baseline_experiment.py --config configs/experiment/baseline/smoke.yaml
python scripts/run_baseline_experiment.py --config configs/experiment/baseline/minimal_ngsim_alibaba.yaml
```

核心输出包括：

- `comparison_summary.csv`
- `comparison_summary.json`
- `comparison_summary_detailed.json`
- `comparison_summary_by_window_class.csv`
- `run_manifest.json`
- `seed_checkpoint_manifest.json`
- `command_log.json`

Round1 状态、机制诊断和复跑命令见：

- `docs/experiment_status_round1.md`
- `docs/mechanism_activation_check_round1.md`
- `docs/experiment_runbook_round1.md`

## Top Journal Closed Loop

顶刊路线优先使用统一闭环入口，自动完成 SA-GHMAPPO、paper-grade learned baselines、seed checkpoint manifest、mixed/full benchmark 和 gate report：

```bash
python scripts/run_top_journal_closed_loop.py --quick --seeds 7 --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure
python scripts/run_top_journal_closed_loop.py --seeds 7 13 29 --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure
```

`--quick` 只验证链路，不用于论文结论；正式 claim 必须满足 `gate_report.json` 中的 `passed=true`、`formal_contract.ready=true` 和 `paper_claim_ready=true`。顶刊主线默认使用 `handoff_pressure` primary vehicle selection，让 NGSIM 窗口中的 handoff 压力进入 workflow 主体。

当前可引用的正式闭环产物：

- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_formal_20260505_v2/gate_report.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_formal_20260505_v2/gate_summary.csv`

Learned-baseline strict gate：

```bash
python scripts/run_top_journal_learned_baseline_suite.py --run_id <run_id> --base_manifest_path <seed_checkpoint_manifest.json> --skip_training --output_root artifacts/experiments/top_journal_sa_iteration
```

当前正式 learned-baseline 产物：

- `artifacts/experiments/top_journal_learned_baseline_suite/top_journal_learned_baseline_formal_20260505_v1/learned_baseline_gate_report.json`

当前默认 paper-grade learned-baseline set 为 `ppo` / `mappo` / `dqn` / `dueling_dqn` / `qmix` / `controller_mat` / `dag_offload_drl` / `cache_offload_drl` / `dt_handoff_drl`。`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl` 分别覆盖 DAG offloading、model/adapter cache offloading 和 Digital Twin handoff/service migration 领域对照。`ippo` 属于当前 contract-blocked diagnostic baseline；`ddqn` / `dueling_ddqn` 只有在 duplicate trace audit 通过时才能作为独立补充。复现旧 IPPO gate 必须显式使用 `--allow_contract_blocked_baselines`，且不能写成 paper-ready 结果。

`top_journal_mechanism_v3_eval_bias` 是基于 formal_v2 权重的 inference calibration 候选增强；生成入口为：

```bash
python scripts/build_top_journal_eval_bias_manifest.py --base_manifest_path artifacts/experiments/top_journal_learned_baseline_suite/top_journal_learned_baseline_formal_20260505_v1/seed_checkpoint_manifest_learned_baselines.json --output_root artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias --label v3_eval_bias
```

该候选不能在未补齐 holdout/support suite 前替代 formal_v2 paper-grade 主表。
## Current v3 Eval-Bias Candidate

Current guarded-prefetch refresh artifacts:

- formal gate: `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_gate_20260506/learned_baseline_gate_report.json`
- holdout gate: `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_holdout_offset3_20260506/learned_baseline_gate_report.json`
- latency fallback ablation: `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_support/statistics/latency_fallback_holdout_ablation_guarded_prefetch/paired_statistics.csv`

This is a strong inference-calibrated candidate, not a clean-retrain replacement. Prediction robustness still has an oracle-setting boundary, so do not claim universal superiority under oracle prediction.

## Current Learned-Baseline Expansion

The current top-journal paper-grade learned-baseline gate defaults to `ppo`, `mappo`, `dqn`, `dueling_dqn`, `qmix`, `controller_mat`, `dag_offload_drl`, `cache_offload_drl`, and `dt_handoff_drl`. `dag_offload_drl`, `cache_offload_drl`, and `dt_handoff_drl` are domain baselines for DAG offloading, model/adapter cache offloading, and Digital Twin handoff/service migration. `ippo` is diagnostic-only until a real independent per-agent wrapper/action contract is implemented; `mappo`, `qmix`, and `controller_mat` are controller-level CTDE/value-decomposition/transformer baselines, not vehicle-agent or RSU-agent full MARL wrappers. `ddqn` and `dueling_ddqn` are optional only if the duplicate-trace audit proves independence. `reactive_greedy` and `popularity_cache_heuristic` remain supplementary heuristic reference lines, not the primary publication gate.

Latest expanded artifacts:

- formal plus-dueling gate: `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_gate_20260506/learned_baseline_gate_report.json`
- holdout plus-dueling gate: `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_holdout_offset3_20260506/learned_baseline_gate_report.json`

## Current Final-Submission Loop

当前可交稿闭环入口为：

```bash
python scripts/run_top_journal_final_submission_loop.py --run_id <new_run_id> --base_manifest_path artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_formal_20260505_v2/seed_checkpoint_manifest.json --force_retrain_learned --resume_training --resume_benchmark --resume_support --command_retries 2 --baseline_episodes 96 --baseline_update_every 6 --baseline_batch_size 32 --minimum_reward_delta 0.5 --holdout_offsets 3
```

生成顶刊对比报告包：

```bash
python scripts/build_top_journal_comparison_report.py --final_run_root artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1
```

当前正式产物：

- `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/final_submission_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/comparison_report/top_journal_comparison_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/comparison_report/top_journal_comparison_report.md`
- `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/comparison_report/paper_ready/paper_ready_main_comparison.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/comparison_report/paper_ready/paper_ready_paired_reward_statistics.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/comparison_report/paper_ready/paper_ready_report.md`
- `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/learned_suites/final_submission_controller_mappo_qmix_20260509_v1_iter1_formal/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/learned_suites/final_submission_controller_mappo_qmix_20260509_v1_iter1_holdout_offset3/learned_baseline_gate_report.json`

`final_submission_controller_mappo_qmix_20260509_v1` 当前为 pre-Controller-MAT / pre-DAG-cache-DT-domain-baseline / pre-MAPPO-head-credit final-submission package：旧数值可用于追溯，但不能再作为当前 MAPPO 强对照的 canonical 主表。新增 `controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl` 以及 MAPPO head-credit 协议后，论文主表必须重跑 final-submission loop。`mappo`、`qmix` 和 `controller_mat` 是 controller-level learned baselines，不应写成 vehicle-agent / RSU-agent full MARL wrappers。旧 `final_submission_clean_retrain_repaired_baselines_20260507_v1` 是 pre-MAPPO/QMIX-controller-level package；旧 `final_submission_clean_equal_budget_20260506_v1` 因 duplicate trace 已作废。
