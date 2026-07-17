# PPO_MEC

## 2026-06-21 strict-full v8 审查状态

v8 已按冻结的 train/dev/formal/hidden 协议完成 5-seed formal 与一次性 hidden holdout。对全部 learned baselines 的 total reward hierarchical BCa 95% CI 在 formal/hidden 均为正，对 DT handoff DRL 的 workflow continuity 也为正；v7 的 strict-full statistical blocker 已修复。

当前 reviewer verdict 为 `Major revision (78/100)`，不是 `TMC-ready candidate`：hidden 相对 PPO 的 handoff failure 更差，formal/hidden 的 backhaul cost 更高，对 popularity heuristic 未形成显著 reward 优势；v8-current robustness/scalability/ablation 与更大外部验证仍待补齐。详见 `docs/project/top_journal_readiness_audit_20260621.md` 和 `docs/project/strict_full_v8_execution_record_20260621.md`。

2026-07-13 已接入 v8-current support suite 入口和 v9 Pareto-safe 候选路径：`scripts/run_strict_full_v8_support_suite.py` 负责补齐 prediction/system/scalability/guard attribution，`top_journal_mechanism_v9_pareto_safe` 负责在 dev / future-validation 上把 handoff failure 与 backhaul 纳入 checkpoint ranking。2026-07-16 进一步新增 `top_journal_mechanism_v10_mappo_rl` 与 `top_journal_mechanism_v11_mappo_reward`，把 MAPPO 的 controller-level CTDE head-credit / entropy-floor 机制迁入 SA-GHMAPPO 候选 profile，同时降低 imitation / mechanism auxiliary 牵引，并在 v11 中加入 reward-first checkpoint priority 与 idle/sparse window-context inference gate。v11 full-dev benchmark 已在 frozen dev plan 上让 SA-GHMAPPO total reward 高于全部对照：`79.4944` vs `popularity_cache_heuristic=79.46875`，artifact 为 `artifacts/experiments/top_journal_mappo_reward_full_dev_v11_20260716/main_results_full_stratified_window_gate_full/main_results_full_stratified_20260716_181112_383674/aggregate_summary.json`。

2026-07-17 新增 `top_journal_mechanism_v12_learned_option`：在 v11 MAPPO-core checkpoint 上 warm-start，加入可学习 contextual option gate，让策略在 `accept_mappo`、`popularity_safe`、`no_rsu_local` 和 `mechanism_prepare` 间学习选择；机制窗口显式保留 MAPPO 主策略，idle/sparse 窗口用 learned option 吸收 popularity-safe 行为。v12 full-dev 5-seed / 20-window / 2-workflow 全量 benchmark 已完成：SA-GHMAPPO total reward `79.5934`，高于 `popularity_cache_heuristic=79.46875`、`ppo=77.18775`、`mappo=72.6328` 及全部其他对照，artifact 为 `artifacts/experiments/top_journal_mappo_reward_v12_learned_option_20260717/main_results_full_stratified_mech_preserve/main_results_full_stratified_20260717_115754_212344/aggregate_summary.json`。v12 仍是 dev evidence，不是 hidden/future-validation 或 paper-ready 结论；hidden holdout 已 consumed，不能再用于筛选。

## 2026-06-21 导师汇报材料

- 可编辑汇报 PPT：`outputs/ppo_mec_advisor_report_20260621.pptx`
- 中文讲稿、创新点、模型架构与结果边界：`docs/project/advisor_report_briefing_20260621.md`

该材料基于 2026-06-18 E3 独立复现证据。可展示结论是：严格非重叠 mixed formal/holdout 对最强 learned baseline 的 reward 置信区间为正；full formal/holdout 仅点估计领先、95% CI 跨 0。材料不得被解释为项目已经达到 `TMC-ready`。

## 完整克隆（含真实数据）

仓库中的 `data/` 通过 Git LFS 版本化。首次克隆或切换到包含数据的分支前，需先安装 Git LFS，然后执行：

```bash
git lfs install
git lfs pull
python scripts/check_data_ready.py
```

若未执行 `git lfs pull`，`data/` 下只会保留 LFS 指针，真实数据链路无法运行。当前正式主线所需的 NGSIM、Alibaba 和 LuST 数据已纳入；highD 仍是未提供的后补数据源。

## 2026-06-18 v7 独立重建与严格审查

当前主机已用 `top_journal_mechanism_v7_latency_fallback_20260618_rebuild_v1` 和 `final_submission_v7_latency_fallback_20260618_rebuild_v1` 独立 clean retrain，复现 2026-05-28 legacy formal/final gate，并完成 SHA-256、manifest、checkpoint provenance、机制消融和 LuST external mobility 检查。

严格审查发现旧 `offset=3 holdout` 与 formal 滑动窗口重叠，不能称独立 holdout。改用 split 内及 split 间时间不重叠窗口后，mixed formal/holdout 对 `dt_handoff_drl` 的 paired CI 为正，但 full formal/holdout CI 跨 0。因此 v7 的 verdict 为 `Not TMC-ready`；legacy `paper_claim_ready=true` 只说明旧项目 gate 可复现。该 blocker 已由上文 v8 协议修复，v7 历史审查见 `docs/project/top_journal_readiness_audit_20260618.md`。

## 2026-05-28 SA v7 legacy final-submission package

- Current paper-ready package: `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/`.
- Final gate: `target_reached=true`, `paper_claim_ready=true`, `blockers=[]`; comparison package: `review_ready=true`, `paper_ready_package_ready=true`.
- Main method profile: `top_journal_mechanism_v7_latency_fallback`, a clean-retrain profile that keeps v6 freshness/admission guards and enables latency fallback fast-timescale execution control.
- Paper-grade learned baselines in this package: `ppo`, `mappo`, `dqn`, `dueling_dqn`, `qmix`, `controller_mat`, `dag_offload_drl`, `cache_offload_drl`, `dt_handoff_drl`.
- 在 legacy formal/offset-3 协议中，SA-GHMAPPO ranks first；这些数值已复现，但 offset-3 不能再标为 independent holdout。
- `popularity_cache_heuristic` remains a close supplementary reference, not a learned-baseline gate blocker: SA margins are `+0.250000`, `+0.479629`, `+0.355556`, and `+0.376191` across formal/holdout mixed/full.
- Reviewer-facing limitations from the generated self-review must be preserved: heuristic gap is close, mechanism realization is not uniformly a standalone CI-positive advantage, and backhaul savings are not universal.

## 2026-05-27 MAPPO v3 / SA v6 update

- `mappo` live baseline now uses `aggregation_reason_weighted_controller_ppo_v3`: controller-head credit floors and entropy floors/scales are applied to slow / fast / event heads to reduce action-mix collapse while keeping MAPPO free of SA-GHMAPPO-only graph/surrogate/guard mechanisms.
- Paper-grade learned-baseline loops accept `--mappo_baseline_profile mappo_strong_audit`; this profile is the default MAPPO profile inside the learned-suite/final-loop wrappers.
- Main-method optimization adds `top_journal_mechanism_v6_strong_competition` and `configs/experiment/top_journal_mechanism_v6_strong_competition.yaml` for a future same-budget rerun against optimized baselines.
- SA v6 now uses a freshness-aware cache-warm guard (`cache_warm_start_guard_max_prefetch_countdown=6.0`) so predictive prefetch is not forced before the recorder validation window.
- SA v6 also uses a confidence/alignment prefetch admission guard (`predictive_prefetch_admission_min_confidence=0.55`) so low-confidence prefetch is deferred until next-RSU and handoff-target evidence align.
- The latest 3-seed freshness-guard closed loop (`top_journal_mechanism_v6_freshness_guard_20260527_v1`) remains a negative candidate: SA still trails `popularity_cache_heuristic` by `0.055556` mixed / `0.018519` full reward and is not paper-ready.
- This v6 note is historical. The v7 legacy result has been reproduced, but the current strict reviewer verdict is `Not TMC-ready`.

PPO_MEC 是面向 AI-driven VEC 的研究原型，主线围绕跨 RSU 连续 DAG workflow 执行、车载 base model 与路侧 adapter cache 协同、handoff 状态迁移、surrogate prediction 和多时间尺度控制。

当前正式数据主线是 `NGSIM + Alibaba`。`LuST` 与 `highD` 保留 provider / 检查骨架，但不阻塞正式主线。

数据源声明统一维护在 `docs/project/DATASET_SOURCES.md` 和 `configs/data/dataset_sources.json`。当前已审计并 metadata-only 接入 Hugging Face model-cache 候选全集，用于 catalog/report 审计和后续 file-size profile 设计；不会自动下载模型文件，也不会替换正式 benchmark 默认 cache 行为。接入边界见 `configs/data/hf_model_cache_integration_plan.json` 和 `docs/agent/hf_model_cache_dataset_audit_round14_report.md`。

## Supervised Handoff Predictor v1

当前代码已接入薄 supervised handoff predictor 路径：`scripts/train_supervised_handoff_predictor.py` 可从冻结 train/dev window plan 训练短时 next-RSU / handoff-target / ETA predictor；`PredictorManager` 支持 `predictor_kind=supervised` 与显式 `predictor_checkpoint_path`。该层用于 handoff anticipation 和 lightweight DT-style predictive state snapshot，不是完整数字孪生系统，也不代表 predictor 本身已经形成 paper-ready 主结论；正式 claim 仍需要冻结 checkpoint、quality report、SA-GHMAPPO v9 重训和 formal/future-validation benchmark。

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
python scripts/build_top_journal_comparison_report.py --final_run_root artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1
```

当前正式产物：

- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/final_submission_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/top_journal_comparison_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/top_journal_comparison_report.md`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/paper_ready/paper_ready_main_comparison.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/paper_ready/paper_ready_paired_reward_statistics.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/paper_ready/paper_ready_report.md`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/learned_suites/final_submission_v7_latency_fallback_20260528_v1_iter1_formal/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/learned_suites/final_submission_v7_latency_fallback_20260528_v1_iter1_holdout_offset3/learned_baseline_gate_report.json`

`final_submission_v7_latency_fallback_20260528_v1` 是 legacy paper-ready package，2026-06-18 rebuild 证明它可复现；但严格非重叠 holdout 审查已否决其当前 TMC-ready 状态。`final_submission_controller_mappo_qmix_20260509_v1`、`final_submission_full_current_baselines_20260511_v1` 和更早 package 只用于历史追溯。`mappo`、`qmix` 和 `controller_mat` 是 controller-level learned baselines，不应写成 vehicle-agent / RSU-agent full MARL wrappers；`popularity_cache_heuristic` 是 close supplementary reference。
