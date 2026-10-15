# Directory Structure

## 2026-10-14 显式目标站工程入口

- `src/envs/core/multi_request_target_control_env.py`：新参数化执行/cache/state/priority 合同，输入/输出送达与逐帧数据链路账本。
- `scripts/verify_joint_target_control.py`：原开发 manifest/hash/窗口重建和 train/dev 固定规则工程验证；输出初始/完成 manifest、`results.jsonl` 与 `events.jsonl`。
- `tests/test_multi_request_target_control_env.py`、`tests/test_verify_joint_target_control.py`：动作差异、依赖/字节/版本、旧策略拒绝、源数据/窗口身份和输出消费链。
- `artifacts/joint_target_control_engineering/`：独立工程回放产物，不进入 Git，不作为 RL 排名。

## 2026-10-14 开发训练曲线与机制机会

- `scripts/launch_eight_vehicle_joint_experiment.py`：可选 `--development_eval_every`；在 update 边界输出 `curve_checkpoints/`、`curve_checkpoints.jsonl` 与 `dev_curve.jsonl`。训练仅用原 agent，开发评价读取独立恢复副本并恢复随机状态；默认值 0 不生成曲线。
- `artifacts/joint_mechanism_opportunity_20261014/`：仅训练窗口的固定负载与固定动作资格诊断，含事前条件、原始逐窗口 JSON 和脚本；该目录为本地实验产物，不进入 Git。
- `artifacts/eight_vehicle_joint_v2_mechanism_budget96_20261014/`：预定的 96 episode 开发实验根目录；manifest 记录负载和曲线间隔，formal/holdout 不运行。

## 2026-10-14 八车冷缓存动作合同诊断

- `src/envs/core/multi_request_joint_control_env.py`：v2 可选共同 cold-cache mask；adapter 未就绪时 action 3 不合法，action 0 仍可请求加载。原 v2 默认行为保留。
- `scripts/launch_eight_vehicle_joint_experiment.py`：`--strict_cold_cache_mask` 启用独立开发协议与动作合同，写入 manifest；各算法在同一设置下重训。
- `artifacts/analysis/joint_predictor_qualification_20261014/`：开发窗口上的常速度换站预检，仅本地诊断，不含策略收益。

## 2026-10-12 八车 v2 修复

- `src/envs/core/joint_model_cache.py`：八车开发合同的独立 typed base/adapter 字节缓存。
- `src/envs/core/joint_causal_observation.py`：按帧因果预测和带延迟的 RSU 状态广播。
- `src/encoders/joint_observation_features.py`：actor 与集中 critic 的独立特征边界。
- `tests/test_masked_env_action_ppo.py`、`test_joint_model_cache.py`、`test_multi_request_joint_v2_execution.py`、
  `test_joint_causal_observation.py`、`test_joint_observation_encoder.py`、`test_joint_experiment_launcher.py`：
  概率合同、物理费用、信息隔离及七算法 producer/consumer 联合验证。
- 原 `artifacts/eight_vehicle_joint_sim/` 根目录沿用；v2 run manifest 标记 `eight_vehicle_joint_sim_v2_development`，
  新增 `dev_sampled.jsonl` 和含 greedy/sample 标签及逐帧费用收据的 `dev_events.jsonl`。
  旧目录不移动、不覆盖；同一 run ID 冲突立即拒绝。

## 2026-10-12 八车联合控制仿真实验

- `src/envs/core/multi_request_joint_control_env.py`：逐请求五动作、四站共享计算/链路/cache、显式 adapter 拉取和状态准备；成本仍为仿真配置。
- `scripts/launch_eight_vehicle_joint_experiment.py`：原 registry 算法的训练、开发和正式窗口执行入口；封存 holdout 不读取结果。
- `tests/test_multi_request_joint_control_env.py`：缓存动作、容量、车端计算时长及预测/准备的合同测试。
- `artifacts/eight_vehicle_joint_sim/`：运行 manifest、checkpoint、训练更新、窗口结果及逐请求事件；不进入 Git。

## 2026-10-12 八车四 RSU 轨迹驱动诊断

- `scripts/benchmark_eight_vehicle_algorithms.py`：统一八车合同上的六种学习策略和四种规则策略，保存排除清单、最终 checkpoint、逐窗口结果和逐请求事件。
- `src/agents/dqn_agent.py` / `mappo_agent.py`：新增独立 `TraceDrivenMultiRequest*` 类，不替换正式 registry 算法。
- `tests/test_eight_vehicle_algorithm_pool.py`：合法动作 mask、Q 更新和集中 critic 输入边界。
- `artifacts/training/eight_vehicle_algorithm_diagnostic/`：算法池诊断产物，不进入 Git 或正式主表。
- `src/agents/ppo_agent.py` 中的 `TraceDrivenMultiRequestPPO`：独立实验性共享策略 PPO，不注册为正式 baseline。
- `scripts/train_eight_vehicle_ppo.py`：多个不重叠 NGSIM 窗口的训练、间隔与评估，保存诊断 checkpoint 和逐窗口奖励。
- `tests/test_trace_driven_ppo.py`：非法 RSU 动作 mask 回归。
- `artifacts/training/eight_vehicle_ppo_diagnostic/`：训练 manifest、最终 checkpoint、逐 episode 和逐窗口结果；不进入 Git。
- `src/envs/core/multi_request_replay_env.py`：独立多请求帧级环境，共享四站 compute/link/cache 容量。
- `scripts/run_eight_vehicle_simulation.py`：NGSIM 连续八车与 Alibaba 八 DAG 的确定性跨源配对及回放入口。
- `tests/test_multi_request_replay_env.py`：八请求争用同一计算槽、窗口连续性和非法目标测试。
- `artifacts/diagnostics/eight_vehicle_simulation/`：本地诊断 manifest、逐请求 JSONL 和 summary；不进入 Git 或正式结果。

## 2026-10-05 最小机制证据闭环

- `configs/experiment/mechanism_evidence_closure_v1.json`：A/C 冻结矩阵、成本模型、预算和命令。
- `configs/acceptance/two_node_workflow_reexecution_comparison_v1.json`：B 的模型、输入、三臂、保真判据和预算。
- `scripts/run_mechanism_evidence_closure.py`：消费既有四配置账本并执行 12 个 bounded native design points。
- `scripts/run_two_node_workflow_reexecution_comparison.py`：复用两节点执行器，比较 continuous/restart/recovery。
- `artifacts/analysis/mechanism_evidence_closure_20261005_v1/`：A/C 原始 JSON、CSV、回执和完整性清单。
- `artifacts/analysis/two_node_workflow_reexecution_comparison_20261005_v1/`：B 四进程 receipt、状态包和终态回执。

2026-09-30 adapter/state calibration：`scripts/calibrate_workflow_state_recovery.py` 只实现明确的两节点 base-only
calibration 状态边界，`scripts/synthesize_adapter_state_recovery_calibration.py` 只读复算上一轮冻结长尾账本与有限
敏感性区间，专项测试为 `tests/test_workflow_state_recovery_calibration.py`。报告位于
`docs/project/adapter_state_recovery_calibration_20260930.md`，机器证据位于
`artifacts/analysis/adapter_state_recovery_calibration_20260930_v1/`；不修改 production action codec 或冻结实验。

2026-09-30 typed-cache 顺序依赖重算候选：显式配置位于
`configs/benchmark/typed_model_cache_controlled_lru_sequential_recompute.yaml`，验收入口为
`scripts/validate_typed_cache_sequential_replacement.py`，专项测试为
`tests/test_typed_cache_sequential_recompute.py`。审查报告位于
`docs/project/native_typed_cache_replacement_witness_20260930.md`，机器证据位于
`artifacts/analysis/native_typed_cache_replacement_20260930_v1/`；该候选不替换默认静态语义或历史正式产物。

单场景引用包入口`scripts/build_driving_workflow_package.py`，产物根`artifacts/datasets/`；不含原图/QA。
小实验计划见`scripts/prepare_drivelm_pilot.py`、`tests/test_drivelm_pilot_plan.py`，
模型下载入口为`scripts/download_driving_pilot_model.py`，本地payload位于被忽略的`data/raw/ai_workflow_pilot/`；
独立venv位于`artifacts/environments/`，都不进入Git。
公开资源元数据为`configs/experiment/drivelm_pilot_resources.json`。
数据资格工具：`scripts/audit_drivelm_workflow_sample.py`，测试为
`tests/test_drivelm_workflow_preview.py`（派生预览与输入组装）及
`tests/test_drivelm_workflow_sample_audit.py`，证据说明位于
`docs/project/drivelm_sample_qualification_20260929.md`；不保存原始标注或模型。

G14R18：唯一 active Protocol 2.9、ready index 与 Readiness v21 位于
`configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906/`；共享 8-field parser 位于
`src/runtime/formal_training_identity.py`，17-field envelope 的 producer 位于
`scripts/manage_typed_model_cache_formal_artifacts.py`，真实 loader/gate 位于 `scripts/benchmark_main_results.py`，
typed checkpoint consumer 位于 `src/runtime/typed_model_cache_runtime.py`。freeze 与审计构建入口为
`scripts/repair_formal_checkpoint_provenance_envelope_contract.py` 和
`scripts/build_formal_checkpoint_provenance_envelope_artifacts.py`；机器证据位于
`artifacts/analysis/typed_model_cache_formal_provenance_envelope_repair_20260906_g14r18_v1/`。test-only checkpoint
未提交；Protocol 2.8 目录 historical/audit-only，G14C v16 尚未创建。

G14R17（historical/audit-only）：Protocol 2.8、旧 ready index 与 Readiness v20 位于
`configs/experiment/typed_model_cache_formal_protocol_v2_8_20260906/`；共享 checkpoint identity projection 位于
`src/runtime/formal_training_identity.py`，producer/read-back 位于 `scripts/train_algo_pool_real_sample.py`，严格
dev/freeze consumers 位于 `scripts/run_typed_model_cache_formal_dev_selection.py` 与
`scripts/manage_typed_model_cache_formal_artifacts.py`。Protocol/artifact 构建入口为
`scripts/repair_formal_checkpoint_nullable_identity_contract.py` 与
`scripts/build_formal_checkpoint_identity_repair_artifacts.py`；机器证据位于
`artifacts/analysis/typed_model_cache_formal_checkpoint_identity_repair_20260906_g14r17_v1/`。测试 checkpoint 不保留，
Protocol 2.7 目录与 G14C v15 run 均为 historical/audit-only。

G14R16（historical/audit-only）：Protocol 2.7、ready index 与 Readiness v19 位于
`configs/experiment/typed_model_cache_formal_protocol_v2_7_20260905/`；resolver 位于
`src/runtime/formal_training_contract.py`，正式 outer gate 位于
`scripts/run_typed_model_cache_formal_protocol.py`，150-cell 生产入口验收复用
`scripts/run_typed_model_cache_formal_training_binding_acceptance.py`。Protocol/artifact 构建入口为
`scripts/repair_formal_training_entrypoint_contract.py` 与
`scripts/build_formal_training_entrypoint_repair_artifacts.py`；机器证据位于
`artifacts/analysis/typed_model_cache_formal_training_entrypoint_repair_20260905_g14r16_v1/`，不含正式
episode、checkpoint 或 performance result。Protocol 2.6 目录保留为 historical/audit-only。

G14R15（historical/audit-only）：Protocol 2.6、Cell Artifact Publication Contract 1.0.0、Generated Checkpoint Resource Contract
1.1.0、ready index 与 Readiness v18 位于
`configs/experiment/typed_model_cache_formal_protocol_v2_6_20260905/`；共享 cell transaction 在
`src/evaluators/formal_cell_transaction.py`，协议/证据构建入口在
`scripts/repair_formal_cell_artifact_publication.py` 与
`scripts/build_formal_cell_artifact_publication_artifacts.py`，非正式验收证据位于
`artifacts/analysis/typed_model_cache_formal_cell_publication_repair_20260905_g14r15_v1/`。

G14R14（historical/audit-only）：Protocol 2.5、Generated Checkpoint Resource Contract 1.0.0、ready index 与 Readiness v17 位于
`configs/experiment/typed_model_cache_formal_protocol_v2_5_20260905/`；共享 generated registry builder/resolver
位于 `src/runtime/generated_checkpoint_resources.py`；Protocol generator、真实 consumer rehearsal、artifact
builder 分别为 `scripts/repair_formal_generated_checkpoint_resources.py`、
`scripts/run_formal_generated_checkpoint_resource_rehearsal.py`、
`scripts/build_formal_generated_checkpoint_resource_artifacts.py`。机器证据位于
`artifacts/analysis/typed_model_cache_formal_generated_resource_closure_20260905_g14r14_v1/`，不含正式 checkpoint。

G14R13：唯一 active Protocol 2.4、Capability Routing Contract 1.0.0、ready index、environment、Scientific/
Order/nullable/lifecycle/exogenous/binding/context 与 Readiness v16 位于
`configs/experiment/typed_model_cache_formal_protocol_v2_4_20260905/`。共享 registry 位于
`src/runtime/formal_protocol_capabilities.py`；freeze、phase rehearsal、artifact builder 分别为
`scripts/repair_formal_protocol_capability_routing.py`、
`scripts/run_formal_protocol_capability_phase_chain_rehearsal.py`、
`scripts/build_formal_protocol_capability_routing_artifacts.py`。机器证据位于
`artifacts/analysis/typed_model_cache_formal_preflight_validator_dispatch_repair_20260905_g14r13_v1/`，不含正式
checkpoint、training 或 performance result。v1.0–v2.3 目录均 audit-only。

G14R8：唯一active Protocol v1.9、resource-resolution contract、ready index、environment、Scientific/Order、
binding/context schema与Readiness v11位于`configs/experiment/typed_model_cache_formal_protocol_v1_9_20260829/`；
共享resolver位于`src/runtime/active_formal_bundle.py`，生成器位于
`scripts/repair_typed_model_cache_formal_bundle_resources.py`，专项测试为
`tests/test_active_bundle_resource_resolution_v19.py`。机器证据位于
`artifacts/analysis/typed_model_cache_formal_bundle_resource_repair_20260829_g14r8_v1/`，不含正式checkpoint、
training或performance result。v1.0–v1.8目录全部audit-only。

G14R7A：唯一active Protocol v1.8、ready index、environment、Scientific Config、Order Contract、binding/context
schema与Readiness v10位于`configs/experiment/typed_model_cache_formal_protocol_v1_8_20260827/`；共享门禁位于
`src/runtime/active_formal_bundle.py`，pending generator/finalizer/acceptance builder位于
`scripts/repair_typed_model_cache_formal_active_bundle.py`、
`scripts/finalize_typed_model_cache_formal_active_bundle.py`与
`scripts/build_typed_model_cache_formal_active_bundle_acceptance.py`。机器证据位于
`artifacts/analysis/typed_model_cache_formal_active_bundle_closure_20260827_g14r7a_v1/`；只含JSON审计，不含
正式checkpoint、training或performance result。v1.0–v1.7目录均为audit-only。

G14R7：active Protocol v1.7、Formal Agent Order Contract `1.0.0`、scientific config与binding/environment/index
位于`configs/experiment/typed_model_cache_formal_protocol_v1_7_20260827/`；共享resolver位于
`src/runtime/formal_agent_order.py`，生成/验收/finalize入口位于`scripts/repair_typed_model_cache_formal_agent_order.py`、
`scripts/run_typed_model_cache_formal_agent_order_acceptance.py`与
`scripts/finalize_typed_model_cache_formal_agent_order_repair.py`。机器证据位于
`artifacts/analysis/typed_model_cache_formal_agent_order_repair_20260827_g14r7_v1/`；只提交根级JSON，
tiny checkpoint/raw rows与临时worktree保持ignored并在交付前移除。该结构不含正式checkpoint或performance。

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
# Protocol 2.0 additions

# Protocol 2.1 additions

- `configs/experiment/typed_model_cache_formal_protocol_v2_1_20260831/`：唯一 active Protocol/index、environment
  manifest、projection contract、binding/context schema 与 Readiness v13。
- `artifacts/analysis/typed_model_cache_formal_environment_identity_repair_20260831_g14r10_v1/`：G14R10
  pre-execution stop、projection/matrix/negative/clean acceptance/integrity 证据。
- `docs/project/formal_environment_identity_projection_contract.md`：projection 规范与 claim boundary。

# Protocol 2.2 additions

- `configs/experiment/typed_model_cache_formal_protocol_v2_2_20260901/`：唯一 active Protocol/index、request
  lifecycle/trace contract、environment projection、binding/context schema 与 Readiness v14。
- `artifacts/analysis/typed_model_cache_formal_request_subject_repair_20260901_g14r11_v1/`：G14R11 root cause、
  matrix、negative、eligibility/parity、exact/clean acceptance 与 integrity 证据。
- `docs/project/formal_request_subject_lifecycle_contract.md`：单 workflow 单连续车辆、RSU/time、runtime 与 claim boundary。
```text
configs/experiment/typed_model_cache_formal_protocol_v2_0_20260831/
  protocol_index.json
  protocol_v2_0_manifest.json
  formal_exogenous_request_execution_contract.json
  formal_request_exposure_schema.json
  readiness_v12.json

src/runtime/formal_exogenous_request_execution.py
scripts/run_formal_exogenous_request_rehearsal.py
scripts/run_formal_exogenous_phase_chain_rehearsal.py
artifacts/analysis/typed_model_cache_formal_exogenous_request_repair_20260831_g14r9_v1/
```

v1.0–v1.9 Protocol 目录仅保留 historical audit，不再是 live active path。

# Protocol 2.3 additions

- `configs/experiment/typed_model_cache_formal_protocol_v2_3_20260903/`：唯一 active Protocol/index、Nullable
  Metric Aggregation Contract 1.0.0、environment/binding/context resources 与 Readiness v15。
- `artifacts/analysis/typed_model_cache_formal_nullable_metric_repair_20260903_g14r12_v1/`：G14R12 compact audit
  package；其 `rehearsal_runtime/` 是 ignored non-formal runtime evidence，不提交大型 checkpoints。
- `docs/project/formal_nullable_metric_aggregation_contract.md`：aggregation、selection、statistics、gate 和 claim
  availability规范。

Protocol v2.2 及更早目录只作 historical audit；live execution 只接受 v2.3 唯一 active index。

## 2026-09-08 G14R20-A

`src/runtime/fixed_commit_continuation.py`、`scripts/preflight_fixed_commit_continuation.py` 与 `scripts/probe_fixed_commit_continuation.py` 为独立只读合同/入口；schema 位于 `configs/experiment/fixed_commit_continuation_v1/`，新增验收位于 `artifacts/analysis/g14r20_a_continuation_20260908/`。 详见 `fixed_commit_continuation_contract.md`。

## 2026-10-03 technical workflow suffix recovery

- `configs/acceptance/two_node_workflow_suffix_recovery_v1.json`：一次性两节点恢复验收的冻结机器方案。
- `src/runtime/workflow_suffix_recovery.py`：与模型加载解耦的状态封装、校验和后缀输入合同。
- `scripts/run_two_node_workflow_suffix_recovery.py`：三进程科学执行与 supervisor 入口。
- `artifacts/analysis/two_node_workflow_suffix_recovery_20261003_v3/`：一次科学执行的机器证据；v1/v2 均在
  run-root 创建和模型加载前失败并记录于 companion review，旧 adapter 验收目录和原始 FAIL 回执不覆盖、不改写。
- `artifacts/analysis/two_node_workflow_suffix_recovery_20261003_review_v1/`：pre-execution 失败、main 七文件保护、
  旧证据 hash 和独立完整性复算。

- `scripts/train_joint_parameterized_control.py`：固定预算 v3 学习/消融；`artifacts/joint_parameterized_learning/`：对应非正式产物。
- `src/envs/core/joint_learned_prediction.py`：训练窗口拟合、冻结因果 GRU 推断。

- `artifacts/joint_parameterized_learning_productive/seed<seed>/`：独立 productive v2 开发再训练，保留原 v1 目录。

- `scripts/audit_joint_parameterized_control.py`：两轮固定参数化开发产物的只读hash/费用/窗口/首窗重放审查。
- `artifacts/joint_parameterized_learning_audit_20261014/`：原始审查JSON、诊断与日志。

- `src/envs/core/joint_control_planning.py`：声明服务量、因果覆盖机会/净收益代理、预算平衡请求账本。
- `tests/test_coordinated_control_learning.py`：预约/迁移条件/prefix信用/概率/checkpoint验证。
- `artifacts/joint_coordinated_control/seed<seed>/`：新一轮同预算产物，保留旧两轮。

- `artifacts/joint_coordinated_partial_control/seed<seed>/`：三个主比较配置的部分预复制阶段，独立于严格版本七配置目录。

2026-10-15：artifacts/joint_coordinated_partial_control_v2保存生命周期修正后三组同预算开发比较；joint_coordinated_partial_control保留已中止试跑及run_abort.json。

2026-10-15：artifacts/joint_scheduled_control/{main,baselines,stability}/seedN/run_id/保存scheduled同预算开发比较；根comparison_plan.json事前冻结各组/seed/预算/source，旧轮原样保留。joint_scheduled_smoke与joint_scheduled_final_smoke为dirty-tree入口验证，不入效果主表。

- `scripts/prepare_joint_tail_control.py`、`train_joint_tail_control.py`、`audit_joint_tail_control.py`：tail公共场景生产/学习/只读消费。
- `src/envs/core/joint_training_scenarios.py`：仅train原时间内外生几何cohort、共享tight场景；不依赖agent。
- `artifacts/joint_tail_control/protocol/`及`<agent>/<run_id>/`：新scenario_id合同，旧scheduled产物保留。

- `scripts/audit_joint_tail_capacity.py`：与算法分数无关的资源必要容量上界。
- `scripts/joint_tail_mobility_support.py`、root/mobility_support_plan.json与artifacts/joint_tail_control/mobility_support/：独立零训练高移动支持集；evaluation.json不混为训练manifest。

2026-10-15：`artifacts/joint_delivery_control/`为团队节点开发比较独立root（protocol字节复用、十group、comparison_plan、audit/prediction_probe）；`joint_delivery_preflight*`与`joint_delivery_witness`为工程/只读证据，非主结果。

2026-10-15：joint_delivery_control为停止v1历史；本轮已完成的开发产物为artifacts/joint_delivery_control_v2，v2工程为joint_delivery_v2_preflight*。不移动/删除v1产物或拼接预算。

2026-10-16：delivery v2收尾原件为该root下audit.json、mobility_support/audit.json、legacy_cache_reference/audit.json与launcher_result.json；post_training_review/保留独立数值重建和两次零更新诊断（本地产物不入Git）。完整结论见joint_delivery_v2_results_review_20261016.md，非canonical更新。

## 2026-10-16 持续工作负载工程

- `src/envs/core/multi_request_stream_env.py`：外生FIFO请求、执行状态隔离、跨请求缓存与新控制身份合同。
- `src/envs/core/joint_stream_workloads.py`：原DAG分层、两族异构catalog、固定到达计划和必要容量下界。
- `src/envs/core/joint_stream_rules.py`：三种无训练工程规则与异构LRU/净收益保护。
- `scripts/run_joint_stream_workloads.py` / `scripts/audit_joint_stream_workloads.py`：score-blind216单元生产与独立raw消费/6次重放。
- `artifacts/joint_stream_workloads/<timestamp>_engineering/`：manifest、protocol.pt、results/events/receipts及audit.json；不进入Git、不替代既有delivery结果。

2026-10-16 FIFO修正：当前完整工程root为`artifacts/joint_stream_workloads_v2/`，workload/protocol v2；v1 `joint_stream_workloads/`停止129单元，仅历史诊断，不能resume/拼接。三规则和216预算不变，raw auditor另验证发布清单的FIFO启动顺序。
