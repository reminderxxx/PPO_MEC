# Directory Structure

G14R6：active Protocol v1.6、scientific config、binding schema与environment/index位于
`configs/experiment/typed_model_cache_formal_protocol_v1_6_20260825/`；runtime binding实例只允许写入未来
durable run root。共享验证位于`src/runtime/formal_training_identity.py`，生成器为
`scripts/repair_typed_model_cache_formal_training_binding.py`，机器审计包位于
`artifacts/analysis/typed_model_cache_formal_training_binding_repair_20260825_g14r6_v1/`。该结构不含正式
checkpoint或performance result。

G14R6：active Protocol v1.6、scientific config、binding schema与environment/index位于
`configs/experiment/typed_model_cache_formal_protocol_v1_6_20260825/`；runtime binding实例只允许写入未来
durable run root。共享验证位于`src/runtime/formal_training_identity.py`，生成器为
`scripts/repair_typed_model_cache_formal_training_binding.py`，机器审计包位于
`artifacts/analysis/typed_model_cache_formal_training_binding_repair_20260825_g14r6_v1/`。该结构不含正式
checkpoint或performance result。

G14R2：Protocol v1.2 与 window contract 位于
`configs/experiment/typed_model_cache_formal_protocol_v1_2_20260820/`；复用 v1.1 的 science/runtime/
fairness assets，不复制或改写 G14B 四个 window plans。loader 合同位于
`src/evaluators/formal_window_consumption.py`，修复/preflight/rehearsal 入口为
`scripts/repair_typed_model_cache_formal_windows.py`、`scripts/validate_formal_window_consumption.py` 与
`scripts/run_typed_model_cache_window_rehearsal.py`。机器审计包位于
`artifacts/analysis/typed_model_cache_formal_window_repair_20260820_g14r2_v1/`；只提交根级 JSON，
`rehearsal_runs/` 下 tiny checkpoints/raw rows 保持 ignored。该结构不含 G14C v3 正式输出。

G14R：v1.1 配置与 companion 位于
`configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820/`，包括 protocol、agent config、
三档 runtime、formal/dev 与 setting-specific fairness manifests、split companion 和 index。代码入口为
`scripts/restart_typed_model_cache_formal_protocol.py`、
`scripts/run_typed_model_cache_formal_protocol.py`、typed support/dev/statistics/artifact wrappers；共享合同位于
`src/runtime/formal_training_contract.py` 与
`src/evaluators/typed_model_cache_formal_execution.py`。审计包位于
`artifacts/analysis/typed_model_cache_formal_protocol_restart_20260820_g14r_v1/`；根级 JSON 纳入 Git，
non-formal rehearsal checkpoints/raw episodes 保持 ignored。该结构不包含 G14C v2 正式输出。

G14B：`src/evaluators/typed_model_cache_formal_protocol.py` 承载历史 interval registry、完整 NGSIM
inventory、split/overlap/hash/formal/seal/readiness 纯合同；
`scripts/freeze_typed_model_cache_formal_protocol.py` 是 create-only 非训练入口。冻结 plans 位于
`configs/experiment/typed_model_cache_formal_protocol_v1_20260820/`，机器审计包位于
`artifacts/analysis/typed_model_cache_formal_protocol_freeze_20260820_g14b_v1/`。该目录只含 JSON
protocol/index/audit，不含 checkpoint 或 performance result。

`scripts/audit_cache_event_telemetry.py` 是单 episode CacheEvent 对账入口；输出应写入 `artifacts/audits/<audit_id>/`，不得覆盖历史 run summary。

G12：`src/predictors/calibration.py` 与 `src/predictors/causal_snapshot.py` 承载pure calibration reducer和snapshot validator；`scripts/audit_predictor_calibration.py` 是稳定入口；小型validation bundle位于 `artifacts/analysis/causal_predictor_snapshot_validation_<run_id>/`。该入口不训练、不调参、不读取formal/holdout/hidden或RL reward。

G11：`configs/data/model_cache_dataset_registry.json` 是 public model-cache dataset qualification 事实源；`src/data/model_catalog/model_cache_dataset_registry.py` 负责纯验证和 deterministic artifact projection；`scripts/validate_model_cache_dataset_registry.py` 是稳定入口；产物固定写入 `artifacts/analysis/model_cache_dataset_discovery_20260819_g11_v1/`。该目录只含 metadata/audit JSON，不存 raw trace 或模型文件。

G08：`src/oracles/` 放置纯request replay/oracle solver；`scripts/build_cache_request_replay.py`、`scripts/run_future_horizon_cache_oracle.py`、`scripts/audit_cache_oracle_gap.py` 为稳定入口；validation bundle写入 `artifacts/analysis/future_horizon_cache_oracle_validation_<run_id>/`，不得覆盖历史artifact。

## 根目录

- `README.md`：项目定位、当前阶段、主线命令和实验入口总览
- `AGENTS.md`：AI 协作和维护规则
- `configs/`：正式实验、baseline 协议和消融相关 manifest
- `configs/data/`：统一数据源声明、G11 model-cache dataset registry 和 HF compatibility integration plan
- `configs/algo/`：方向匹配对照算法配置
- `configs/experiment/baseline/`：baseline 训练、评估和 benchmark 闭环配置
- `configs/experiment/top_journal_mechanism_v1.yaml`：顶刊路线机制训练 profile 与 benchmark 计划
- `configs/experiment/top_journal_mechanism_v8_strict_full.yaml`：strict-full v8 冻结候选参数、统计协议和 claim gate
- `configs/experiment/top_journal_mechanism_v9_pareto_safe.yaml`：v9 dev/future-validation 安全候选参数、non-inferiority 目标和 hidden 禁用边界
- `configs/experiment/top_journal_mechanism_v10_mappo_rl.yaml`：v10 MAPPO-core RL 候选参数，迁入 controller-level head-credit / entropy floors，并降低 imitation / auxiliary 牵引
- `configs/experiment/top_journal_mechanism_v11_mappo_reward.yaml`：v11 MAPPO reward-first dev 候选参数，记录 reward-first checkpoint priority 与 idle/sparse window-context fallback gate
- `configs/experiment/top_journal_mechanism_v12_learned_option.yaml`：v12 learned MAPPO option gate dev 候选参数，记录 option labels、contextual prior、warm-start 和 hidden 禁用边界
- `configs/experiment/top_journal_mechanism_v13_prd_option.yaml`：v13 partial-reward-decoupled MAPPO dev 候选参数，记录 event/option PRD credit、latest-after-training checkpoint policy 和 hidden 禁用边界
- `configs/experiment/top_journal_mechanism_v18_counterfactual_option.yaml`：v18 counterfactual option-credit MAPPO 候选参数，记录 selected-vs-expected legal-option utility credit 和负向晋级边界
- `configs/ablation_checkpoint_manifest_v8_guard_attribution.json`：v8 同 checkpoint 机制归因消融 manifest
- `configs/experiment/top_journal_v8_strict_split_20260621/`：outcome-blind train/dev/formal/hidden 固定窗口计划与 SHA-256 manifest
- `configs/experiment/top_journal_v17_future_validation_time_audited_20260717/`：v17 time-audited future-validation 固定窗口计划；按 `frame_offset` 和 `time_index_start/end` 同时排除历史 split
- `configs/experiment/typed_model_cache_formal_protocol_v1_20260820/`：G14B train/dev/formal/sealed-holdout 四个 outcome-blind window plans 与 protocol index；holdout 仍 sealed
- `configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820/`：G14R executable v1.1 protocol、agent/runtime/fairness/split companions 与完整 command matrix；不含正式结果
- `configs/experiment/typed_model_cache_formal_protocol_v1_2_20260820/`：G14R2 executable v1.2 protocol、
  frozen window consumption contract、agent/split companions 与 index；不含正式 checkpoint 或结果
- `configs/experiment/typed_model_cache_formal_protocol_v1_6_20260825/`：G14R6 active v1.6 protocol、
  execution-neutral scientific config、binding schema、environment manifest与index；runtime binding不写入该目录
- `configs/experiment/typed_model_cache_formal_protocol_v1_6_20260825/`：G14R6 active v1.6 protocol、
  execution-neutral scientific config、binding schema、environment manifest与index；runtime binding不写入该目录
- `data/`：原始数据与处理后数据；通过 Git LFS 版本化，完整克隆后需执行 `git lfs pull`
- `docs/`：长期维护文档，`docs/project/` 为事实来源，`docs/project/DATASET_SOURCES.md` 记录数据源声明，`docs/project/literature_reference_table.md` 记录顶刊/顶会 related-work 参考表，`docs/benchmark_plan_or_baseline_plan.md`、`docs/baseline_formalization_round1.md`、`docs/experiment_status_round1.md`、`docs/mechanism_activation_check_round1.md` 和 `docs/experiment_runbook_round1.md` 记录 baseline 计划、round1 状态、机制诊断与复跑命令
- `scripts/`：数据检查、dry-run、训练、评估和 benchmark 入口
- `scripts/run_top_journal_final_submission_loop.py`：最终交稿 learned-primary 自循环入口，编排 learned baseline 重训、formal/holdout gate、cluster bootstrap statistics 和 support suites
- `scripts/build_top_journal_comparison_report.py`：最终交稿 comparison package 生成入口，汇总 baseline protocol matrix、reward margins、mechanism paired statistics、support statistics、paper-ready LaTeX 表格和作者自审报告
- `scripts/audit_artifact_integrity.py`：run-root SHA-256、JSON path reference、external dependency 和 parse error 审计
- `scripts/audit_window_independence.py`：formal/holdout selected window plan 的 split 内与 split 间 frame/time interval 独立性审计
- `scripts/freeze_future_validation_split.py`：从 outcome-blind mobility covariates 生成 future-validation window plan，并排除已 consumed train/dev/formal/hidden frame/time intervals
- `scripts/freeze_strict_split_protocol.py`：生成跨 split 互斥、带 minimum frame gap 的固定窗口计划
- `scripts/run_strict_full_v8_support_suite.py`：编排 v8-current support suite、guard attribution、BCa/Holm statistics 和 support gate report；拒绝 hidden window plan
- `scripts/train_supervised_handoff_predictor.py`：从冻结 train/dev window plan 训练短时 supervised handoff predictor，并输出 checkpoint、metrics manifest 和 quality rows
- `scripts/audit_predictor_calibration.py`：从非hidden三段split审计binary calibration、reliability/selective gate、causal snapshots、staleness和pre-action trace，不运行RL benchmark
- `scripts/analyze_strict_full_failure_modes.py`：在非 hidden split 上分解 strict-full reward、continuity、failure 和 action-mix 失败模式
- `scripts/audit_literature_reference_table.py`：检查文献表标题/DOI/URL 重复、链接结构和显式待核验项
- `scripts/validate_model_cache_dataset_registry.py`：校验 G11 taxonomy、字段、评分、hard gates与兼容投影，并确定性生成机器审计包
- `scripts/validate_typed_model_cache.py`：生成 G13 typed base/adapter/state 受控验证包，覆盖原子事务、readiness、容量、五种 eviction、公平性、小规模 oracle 和真实数据最小链路
- `scripts/run_typed_model_cache_runtime_rehearsal.py`：G14A non-formal typed training/checkpoint/benchmark/CacheEvent/metrics闭环验证；不运行formal/holdout/hidden
- `scripts/freeze_typed_model_cache_formal_protocol.py`：G14B create-only 历史排除、split、formal protocol、holdout seal 与 readiness freeze；不运行 episode 或生成 checkpoint
- `scripts/run_typed_model_cache_formal_protocol.py`：G14R 13 阶段 append-only G14C v2 执行入口；普通 runner 无 holdout capability
- `scripts/run_typed_model_cache_formal_repair_rehearsal.py`：G14R bounded non-formal cadence/config/endpoint/support/phase rehearsal
- `src/`：核心实现
- `tests/`：自动化测试
- `artifacts/`：当前保留的训练 checkpoint、benchmark 报告和论文表格产物
- `outputs/`：面向用户的可编辑汇报导出物；不作为训练、benchmark 或 canonical artifact 根目录

## 数据目录

- `data/raw/mobility/ngsim/`：NGSIM 官方轨迹 CSV
- `data/raw/mobility/LuSTScenario/`：LuST SUMO 场景
- `data/raw/mobility/highD/`：highD 原始 CSV
- `data/raw/workflow/alibaba2018/`：Alibaba batch task 数据
- `data/raw/model_cache/`：外部 model-cache 数据源审计 manifest；默认不自动下载模型文件
- `data/processed/mobility/lust/`：LuST FCD 导出 CSV
- `data/processed/sampled_vec_dags/`：采样后的 workflow DAG JSONL

## 代码目录

- `src/agents/`：agent 基类、注册表和按算法分文件的主方法 / 对比方法接入；不再保留 `baselines/` 或 `marl/` 分类目录
- `src/data/`：mobility、workflow 和 model catalog 数据层；model-cache dataset qualification 与 `AdapterCatalog` runtime schema 保持职责分离
- `src/data/model_catalog/typed_model_cache_controlled.json`：G13 runtime controlled catalog；`hf_metadata_diagnostic_model_profile.json` 仅是非正式 metadata diagnostic，不参与 runtime 初始 cache
- `src/encoders/`：DAG、RSU 状态和融合编码器
- `src/envs/`：核心环境、预测层和 Gym/vector wrapper
- `src/envs/specs/action_schema.py`：语义动作 schema、mask 和 action adapter
- `src/predictors/`：监督 handoff predictor 的 feature schema、MLP checkpoint loader 和 runtime 推理封装
- `src/runtime/`：跨config、training、checkpoint、fairness与benchmark共享的resolved runtime/provenance合同；不承载算法或环境mutation
- `src/evaluators/`：真实 sample、主结果和 checkpoint 评估辅助
- `src/metrics/`：episode recorder、指标 reducer 和论文指标
- `src/trainers/`：PPO/MARL 训练驱动和 buffer
- `src/utils/`：通用工具

## 产物目录

- `outputs/ppo_mec_advisor_report_20260621.pptx`：基于 E3 复现证据整理的导师汇报 deck；数据来源与结论边界见 `docs/project/advisor_report_briefing_20260621.md`
- `artifacts/training/`：被保留 benchmark 引用的训练 run、checkpoint 和训练审计
- `artifacts/training/algo_pool/`：方向匹配对照算法训练产物
- `artifacts/training/supervised_predictors/`：supervised handoff predictor 的 checkpoint、quality report 和 metrics manifest
- `artifacts/training/algo_pool_formal_round1/`：round1 三 seed formal flat baseline 训练产物
- `artifacts/eval/algo_pool/`：方向匹配对照算法评估产物
- `artifacts/experiments/baseline/`：config-driven baseline 闭环产物、per-seed manifest、comparison summary 和 by-window-class summary
- `artifacts/experiments/top_journal_closed_loop/`：顶刊路线闭环产物，包括训练记录、seed checkpoint manifest、benchmark aggregate 和 gate report
- `artifacts/experiments/top_journal_mappo_reward_full_dev_v11_20260716/`：v11 full-dev 训练 manifest、checkpoint-selection probes 和最终 window-gate full benchmark；当前成功主表为 `main_results_full_stratified_window_gate_full/main_results_full_stratified_20260716_181112_383674/aggregate_summary.json`
- `artifacts/experiments/top_journal_mappo_reward_v12_learned_option_20260717/`：v12 learned option full-dev probes、seed checkpoint manifest 和最终 mechanism-preserve full benchmark；当前成功主表为 `main_results_full_stratified_mech_preserve/main_results_full_stratified_20260717_115754_212344/aggregate_summary.json`
- `artifacts/experiments/top_journal_prd_option_v13_20260717/`：v13 PRD option probes、latest/best-reward seed manifest 和全量 dev benchmark；当前 latest 成功主表为 `main_results_full_stratified_latest/main_results_full_stratified_20260717_124815_375515/aggregate_summary.json`
- `artifacts/experiments/top_journal_counterfactual_option_v18_20260717/`：v18 counterfactual option-credit 训练 manifest 和全量 dev benchmark；当前结果为负向探索，不作为主候选
- `artifacts/experiments/top_journal_dag_aware_option_v17_20260717/future_validation_time_audited_full_stratified/`：v17 time-audited future-validation 全量 benchmark；均值第一但对 popularity reward CI 跨 0
- `artifacts/analysis/top_journal_v17_future_validation_time_audited_statistics_20260717/`：time-audited future-validation window-outer hierarchical statistics
- `artifacts/analysis/typed_model_cache_validation_20260819_g13_v1/`：G13 小规模 deterministic validation；`formal=false`、`training=false`，不得作为算法收益论文证据
- `artifacts/analysis/typed_model_cache_runtime_plumbing_validation_20260819_g14a_v1/`：G14A non-formal typed MB plumbing、tiny checkpoint gate与reconciliation；不得作为正式checkpoint或论文证据
- `artifacts/analysis/typed_model_cache_formal_protocol_freeze_20260820_g14b_v1/`：G14B 历史账本、完整 interval inventory、split/overlap、formal/statistics/claim/seal/readiness 与 integrity JSON；checkpoint/performance result count均为0
- `artifacts/analysis/typed_model_cache_formal_protocol_restart_20260820_g14r_v1/`：G14R v1 failure reference、execution matrix、v1.1 protocol/hash、endpoint/support/command/phase/rehearsal/readiness/integrity JSON；正式结果 count=0
- `artifacts/audits/top_journal_v17_future_validation_time_audited_20260717/`：future split 与 train/dev/formal/hidden 的 frame/time 双区间独立性审计
- `artifacts/experiments/top_journal_support_suite/`：v8-current support suite、机制归因、paired statistics 和 support gate report 输出根目录；dry-run 不能作为论文证据
- `artifacts/experiments/strict_full_v8_*`：v8 formal、一次性 hidden 与 LuST external benchmark；正式结论只引用 `top_journal_readiness_audit_20260621.md` 列出的 run ID
- `artifacts/audits/strict_full_v8_integrity_20260621/`：v8 11457-file SHA-256 inventory 与引用完整性报告；保持 Git ignored
- `artifacts/experiments/top_journal_learned_baseline_suite/`：learned-baseline strict gate 产物；当前新 run 的 paper-grade 默认 learned set 为 PPO/MAPPO/DQN/Dueling-DQN/QMIX/Controller-MAT/DAG-Offload-DRL/Cache-Offload-DRL/DT-Handoff-DRL，IPPO 旧产物只作 diagnostic/历史审计
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_*`：补充 Dueling-DQN / Dueling-DDQN 后的 learned-baseline 扩展 gate 和 holdout 产物
- `artifacts/experiments/top_journal_sa_iteration/`：主方法优势迭代和候选验证产物，包括 v2/v3 retrain、eval-bias manifest、screen benchmark 和 learned gate；负向迭代不作为 paper-grade 主表
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_*`：当前 v3 eval-bias formal/holdout gate refresh。
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_support/`：v3 eval-bias latency fallback 消融、prediction robustness、robustness 和 scalability 支撑产物。
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v4_prepare_eval_bias*`：v4 prepare override 负向筛选产物，不作为主结果。
- `artifacts/experiments/top_journal_final_submission/`：最终交稿闭环产物；`final_submission_v7_latency_fallback_20260528_v1` 是 legacy paper-ready package，`final_submission_v7_latency_fallback_20260618_rebuild_v1` 为 E3 historical rebuild。旧 offset=3 与 formal 重叠；最新 readiness 以 `top_journal_readiness_audit_20260621.md` 为准。
- `artifacts/benchmarks/`：当前可引用的主结果、预测鲁棒性、消融、robustness 和可扩展性 benchmark
- `artifacts/analysis/model_cache_dataset_discovery_20260819_g11_v1/`：G11 19候选 registry snapshot、字段矩阵、评分、HF复核、mapping、validation与integrity manifest
- `artifacts/analysis/hf_model_cache_dataset_audit_round14/`：历史 HF model-cache 候选适配性审计路径；当前结论以 G11 artifact 为准
- `artifacts/paper/`：历史 paper export；legacy v7 表格只能在明确标注 overlap limitation 时使用，strict reviewer 结论以最新审计为准

新产物应写入明确的 run 目录，不应散落到仓库根目录。
`artifacts/analysis/cache_capacity_mb_validation_<run_id>/` 保存 MB capacity contract validation 的 summary、scenario/event snapshots 与 invariant 结果，不覆盖历史 validation。

`artifacts/analysis/classical_cache_baseline_validation_<run_id>/` 保存五种 matched reactive baseline 的 controlled mechanism validation；非 formal 结果。
