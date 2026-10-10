# Directory Structure

## 2026-10-11 CSCWD 原始时间只读根因审计

- `scripts/diagnose_cscwd_raw_time_root_cause.py`、`tests/test_cscwd_raw_time_root_cause.py`：只读重放冻结 v4 147 步、各阶段/接触门账本和合成决策 clone 后缀泄漏见证；不改变环境。
- `artifacts/analysis/cscwd_raw_time_root_cause_20261011_v2/diagnosis.json`：本地 create-only 轻量机器证据，忽略 Git；`docs/project/cscwd_raw_time_root_cause_audit_20261011.md` 是审查台账。

## 2026-10-10 CSCWD 原始 NGSIM 事件时间开发剖面

- `src/data/mobility/ngsim_event_trace.py`：只读取冻结的三个已暴露窗口，核验原始 100 ms 时间/车辆帧身份和单位；不保存坐标。
- `src/envs/core/raw_ngsim_event_time_env.py`：独立 opt-in 时间、位置、几何接触及保守原子准入；公共观测只用轨迹前缀。
- `scripts/audit_cscwd_raw_ngsim_event_time.py`：create-only 固定动作、双时间剖面有界开发检查；原件保存在忽略 Git 的 `artifacts/analysis/cscwd_raw_ngsim_event_time_20261010_v*/`。
- `tests/test_raw_ngsim_event_time_env.py` 与 `docs/project/cscwd_raw_ngsim_event_time_contract_20261010.md`：边界/时间/因果验收和事前合同。

## 2026-10-09 因果前缀强基线版本

- `docs/project/cscwd_prepared_state_prefix_interface_20261010.md` 与 `tests/test_calibrated_workflow_prepared_state_prefix.py`：显式 v4 公共观测接口交接和合成验收；B runner/protocol 与实验原件不在本 A 分支改动。

- `scripts/diagnose_cscwd_sa_long_budget_cost.py` 与 `scripts/audit_cscwd_long_budget_state_alias.py`：只读 B 长预算原件，分别生成本地 `artifacts/analysis/cscwd_sa_long_budget_cost_diagnosis_20261009_v1/` 和 `cscwd_sa_long_budget_state_alias_20261009_v2/`；v1 别名审计的过滤错误原件保留，结论以 v2 为准。
- `docs/project/cscwd_sa_long_budget_cost_diagnosis_plan_20261009.md`、`cscwd_sa_long_budget_cost_diagnosis_20261009.md` 与 `cscwd_sa_long_budget_cost_summary_20261009.json`：事前规则、只读报告和小型机器摘要。

- `scripts/diagnose_cscwd_sa_behavior.py`：固定科学原件的只读重放、train/dev 共同状态前向和预测误差分层；本地 create-only 诊断根为 `artifacts/analysis/cscwd_sa_behavior_diagnosis_20261009_v1/`。
- `docs/project/cscwd_sa_behavior_diagnosis_plan_20261009.md`、`cscwd_sa_behavior_diagnosis_20261009.md` 与 `cscwd_sa_behavior_diagnosis_summary_20261009.json`：事前规则、独立报告和小型机器摘要。

- `src/envs/core/causal_rsu_predictor.py`：训练拟合的低容量前缀 RSU 预测器。
- `configs/experiment/calibrated_workflow_strong_baselines_development_v2_prefix_only.json`：一次开发执行的 hash、方法、预算和权限冻结。
- `scripts/run_calibrated_workflow_strong_baselines.py`、`scripts/analyze_calibrated_workflow_strong_baselines.py`、`scripts/launch_calibrated_workflow_strong_baselines.py`：预检/训练、自动分析、持久单次监督。
- `artifacts/experiments/cscwd_causal_strong_baselines_dev_20261009_v1/`：create-only 开发原件，忽略 Git；checkpoint 与真实数据不上传。
- `docs/project/cscwd_causal_strong_baseline_protocol_20261009.md`：事前协议和贡献边界。
- `docs/project/cscwd_2027_strong_baseline_prediction_permission_blocker_20261009.md`：旧版公共预测回退的科学权限阻断；无 run artifact。

## 2026-10-09 CSCWD 强基线开发链接线

- `scripts/run_calibrated_workflow_strong_baselines.py`：复用冻结 service/PPO 采样及评价消费者的 guarded 开发比较入口；默认仅可只读 preflight。
- `configs/experiment/calibrated_workflow_strong_baselines_development_v1.json`：未授权的 raw critic、同预算草案；仅指向已消费的 36-instance 开发 manifest。
- `tests/test_calibrated_workflow_strong_baselines.py`：完全合成实例的 DT 训练/选模/checkpoint 与 Popularity 逐实例重置验收。
- `docs/project/cscwd_2027_strong_baseline_wiring_20261009.md`：能力标签、预算、开发命令草案与 claim 边界。

## 2026-10-09 service-reward learning diagnosis

- `scripts/diagnose_calibrated_workflow_service_reward_learning.py`：create-only 只读重放、value/GAE 重算、固定 dev 梯度尺度与
  checkpoint probability trace；不训练或评价。
- `configs/experiment/calibrated_workflow_value_normalization_ab_v1.json`：未实现、未授权的单变量 PopArt A/B 冻结协议。
- `artifacts/analysis/calibrated_workflow_service_reward_learning_diagnosis_20261009_v1/`：18-cell matrix、状态分组、GAE/
  truncation/gradient/checkpoint 见证、身份回执与 integrity；不含 checkpoint。
- `docs/project/calibrated_workflow_service_reward_{learning_diagnosis,claim_change}_20261009.md`：独立诊断与 A 线主张边界。

## 2026-10-06 service-reward alignment

- `configs/experiment/calibrated_workflow_service_reward_alignment_v1.json`：两套 reward、单一候选权重理由、3×192 身份、
  `82,944` 理论 step cap 和 reward-free checkpoint selection。
- `scripts/audit_calibrated_workflow_service_reward.py`：公式分项案例、transition invariance、truncation bootstrap 的 create-only
  训练前门禁。
- `scripts/run_calibrated_workflow_service_reward_alignment.py`：两奖励×SA/MAPPO/PPO 一次性训练、dev 选模、规则对照、
  双公式同轨迹重评分和完整行为账本。
- `scripts/analyze_calibrated_workflow_service_reward_alignment.py`：source-window paired bootstrap、全 seed/strata、负区域、
  reward decomposition 和 paper table。
- `tests/test_calibrated_workflow_service_reward.py`：reward 排序、转移不变、bootstrap、selection 与 frozen identity。
- `artifacts/analysis/calibrated_workflow_service_reward_preflight_20261006_v1/`：训练前冻结原件。
- `artifacts/benchmarks/calibrated_workflow_service_reward_alignment_20261006_v1/`：零步启动失败现场。
- `artifacts/benchmarks/calibrated_workflow_service_reward_alignment_20261006_v2/`：唯一科学结果；checkpoint 本地保留。

## 2026-10-06 SA prepare-balance single-factor ablation

- `configs/experiment/calibrated_workflow_prepare_balance_ablation_v1.json`：A 历史复用、B `auxiliary_coef=0.0`、
  3×192 与 13,824-step 上限的冻结身份。
- `scripts/diagnose_calibrated_workflow_prepare_balance.py`：最多 12 个 dev 状态的 raw→temperature→sharpen→margin
  logits/prob/action trace；只诊断，不训练或计性能。
- `scripts/run_calibrated_workflow_prepare_balance_ablation.py`：只训练候选，重放 A 扩展行为字段，复用匹配 controls，保存
  完整曲线、checkpoint selection、readiness/feasibility ledger 与回执。
- `scripts/analyze_calibrated_workflow_prepare_balance_ablation.py`：只读重建 source-window paired bootstrap、行为计数、
  seed/strata 与完整性。
- `tests/test_calibrated_workflow_prepare_balance.py`：单因素、旧调用兼容、executed likelihood、环境/reward identity 和
  readiness logging 回归。
- `artifacts/analysis/calibrated_workflow_prepare_balance_diagnosis_20261006_v1/` 与
  `artifacts/benchmarks/calibrated_workflow_prepare_balance_ablation_20261006_v1/`：诊断和有界负结果；checkpoint 本地保留。
- `docs/project/sa_prepare_balance_{localization,ablation_results}_20261006.md`：事前定位/候选和完整结果边界。

## 2026-10-06 calibrated continuous-workflow pilot

- `src/envs/core/calibrated_continuous_workflow_env.py`：独立的小型 event-driven 校准仿真；共享五动作、依赖安全
  bundle cache、原子 rollback、state prepare/import 与 DAG predecessor recompute。
- `configs/experiment/calibrated_continuous_workflow_pilot_v1.json`：实测/轨迹/人工来源分离、八个分层模板和共同训练预算。
- `configs/experiment/calibrated_continuous_workflow_pilot_v1_manifest.json`：结果盲冻结的 12/4/12 train/dev/evaluation 实例。
- `scripts/freeze_calibrated_continuous_workflow_pilot.py`：从既有 NGSIM window plans 和 Alibaba JSONL create-only 冻结 workload。
- `scripts/run_calibrated_continuous_workflow_pilot.py`：SA-GHMAPPO/PPO/MAPPO 匹配训练、two-step rule、window-outer CI、
  checkpoint 与论文表入口。
- `tests/test_calibrated_continuous_workflow.py`：action 4 恢复、DAG 重算、依赖安全替换、失败 rollback 和 rule side-effect 回归。
- `artifacts/benchmarks/calibrated_continuous_workflow_pilot_20261006_v1/`：9 个 checkpoint、108 learned evaluation rows、
  12 rule rows、aggregate、paper table、receipt 与 integrity。
- `docs/project/calibrated_continuous_workflow_pilot_20261006.md`：负向结果、论文实验表、审查元数据与 claim 边界。

### v2 纠正版与 interface-blocked 训练尝试

- `configs/experiment/calibrated_continuous_workflow_pilot_v2.json` / `_manifest.json`：128-episode 限额、
  actual/estimated link error strata、dev checkpoint selection、同信息消融与 12/4/12 frozen instances。
- `scripts/diagnose_calibrated_continuous_workflow_v2.py`：depth-4 可达状态、对称计费、信息泄漏与规则分歧的 create-only 诊断。
- `scripts/run_calibrated_continuous_workflow_pilot_v2.py`：SA/PPO/controller-MAPPO/no-dependency 训练、两条 model-based planner、dev
  选模、evaluation-window outer CI、分层、曲线与 completion receipt。
- `artifacts/calibrated_continuous_workflow_{non_degeneracy,pilot_v2}_20261006/`：诊断、checkpoint、全部 seed/window 结果、
  曲线、分层、日志指针与 integrity。
- `docs/project/sa_ghmappo_innovation_fair_training_v2_20261006.md` 与
  `manuscript_experiment_section_sa_ghmappo_pilot_v2_20261006.md`：最近邻矩阵、事后接口失效边界和待合并诊断章节。

### v3 interface repair（独立新 profile）

- `src/encoders/calibrated_workflow_features.py`：typed bundle、byte occupancy、模型大小和公共 calibrated context 的共享特征函数。
- `configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json` / `_manifest.json`：继承 v2 数值配置，冻结
  12/4/12/8 train/dev/regression/frozen-check、3 seeds × 192 episodes 和 41,472-step 上限。
- `scripts/preflight_calibrated_workflow_interface_repair.py`：hash、split interval、预算、权限与三方法最小更新门禁。
- `scripts/run_calibrated_workflow_interface_repair.py`：一次性 bounded retrain、dev 选模、已暴露 regression、新冻结开发检查、
  raw-head/action 对账、分层和 checkpoint-free 交付入口。
- `scripts/analyze_calibrated_workflow_interface_repair.py`：从已完成 artifact 重建 window-outer bootstrap、分层、训练曲线、
  历史 regression 对照与 integrity；不训练、不选择 checkpoint。
- `scripts/reconcile_calibrated_workflow_action_heads.py`：只读重放 selected checkpoints，逐步保存 raw logits/probs、head actions、
  aggregate、projection 与 executed action，并验证原 action ledger 不变。
- `tests/test_calibrated_workflow_interface_repair.py`：单位/字段消费、动作聚合与 likelihood、action 0/4、重复 prepare 和
  failure-time mobility 回归。
- `artifacts/benchmarks/calibrated_continuous_workflow_interface_repair_20261006_v1/`：完成的 3-seed bounded run、200-row
  evaluation、1,477-row action ledger、1,340-row head reconciliation、曲线、统计、回执与 65-file integrity；checkpoint 不提交。
- `docs/project/calibrated_workflow_interface_repair_results_20261006.md`：缺陷、修复、负向公平比较与 claim 边界。

## 2026-10-06 symmetric recovery cost correction

- `src/runtime/symmetric_recovery_cost.py`：双方 event ledger 的纯函数 scorer 与三种在线决策。
- `scripts/run_symmetric_recovery_cost_validation.py`：四环境隔离的 preview/score runner、offline reference 和 old→new 输出。
- `configs/experiment/eviction_aware_recovery_corrected_v2.json`：按旧 config SHA 继承原 12 点的纠正版。
- `configs/experiment/recovery_cost_boundary_check_v1.json`：最多 6 点的一次性机制边界冻结配置。
- `tests/test_symmetric_recovery_cost.py`：共同成本、分支隔离、依赖安全、reload 守恒和权限边界。
- `artifacts/analysis/eviction_aware_recovery_corrected_20261006_v2/`：12 点、48 path、old→new 48 行与 7-file manifest。
- `artifacts/analysis/recovery_cost_boundary_check_20261006_v1/`：6 点、24 path、全量结果与 5-file manifest。
- `docs/project/recovery_cost_{defect_impact_report,corrected_matrix_results,boundary_check_results,correction_execution_record}_20261006.md`：
  只读缺陷、复算、边界和交付记录。

## 2026-10-05 real adapter victim→reload

- `configs/acceptance/real_cache_victim_reload_v1.json`：真实资源 identity/字节容量、两条件、逐臂成本和 12-call 上限。
- `src/runtime/peft_adapter_lifecycle.py`：显式 opt-in、adapter-only 的 PEFT unload/load bridge；不提交 logical ledger。
- `scripts/run_real_cache_victim_reload.py`：native preview→runtime unload/load→native commit 的 create-only 测量入口。
- `tests/test_real_cache_victim_reload.py`：预算、真实字节 victim、无驱逐、缺模型/非法 base victim 与 commit 顺序合同。
- `docs/project/real_cache_victim_reload_plan_20261005.md`：读取新输出前冻结的人工方案与历史公式缺陷边界。
- `artifacts/analysis/real_cache_victim_reload_20261005_v1/`：create-only 12-call 结果、6 进程回执、两 condition
  lifecycle、negative checks、terminal 与 22-file integrity manifest。
- `docs/project/real_cache_victim_reload_results_20261005.md`：逐事件卸载/加载、全路径成本、误差和 claim 边界。

## 2026-10-05 independent recovery cost measurement

- `configs/acceptance/independent_recovery_cost_measurement_v1.json`：两条件、六次配对、随机种子、事前成本/动作和
  24-call 硬预算的机器冻结方案。
- `scripts/run_independent_recovery_cost_measurement.py`：只复用既有模型、输入、状态包与 production action-4 路径的
  新进程配对测量及原始值/误差/决策回执生成器。
- `tests/test_independent_recovery_cost_measurement.py`：预算、预测、共同模型准备成本和未覆盖真实 reload 的静态合同。
- `docs/project/independent_recovery_cost_measurement_plan_20261005.md`：模型输出前冻结的人工可读计划。
- `artifacts/analysis/independent_recovery_cost_measurement_20261005_v1/`：已完成的 create-only 结果根，含 6 次配对、
  18 个进程回执、raw JSON/CSV、aggregate、terminal 与 72-file integrity manifest。
- `docs/project/independent_recovery_cost_measurement_results_20261005.md`：逐项误差、动作一致性、全部原始重复、
  同侧无边界结论与投稿边界。

## 2026-10-05 eviction-aware recovery

- `src/runtime/eviction_aware_recovery.py`：纯函数 opt-in 成本规则、原阈值与同信息两步对照；不增加动作/网络。
- `scripts/run_eviction_aware_recovery_validation.py`：当前 native victim preview、两条 action branch、四方法和证据输出。
- `configs/experiment/eviction_aware_recovery_v1.json`：12 点、事前成本、误差和 claim boundary freeze。
- `tests/test_eviction_aware_recovery.py`：对象去重、保守 fallback、合法 victim 与 native action transition 回归。
- `artifacts/analysis/eviction_aware_recovery_validation_20261005_v1/`：48 行结果、逐事件证据、汇总、回执与完整性清单。
- `docs/project/eviction_aware_recovery_plan_20261005.md` / `eviction_aware_recovery_results_20261005.md`：计划与审查。

## 2026-10-05 production action 4 / workload v0.1

- `configs/experiment/workload_v0_1_cost_mismatch_robustness_v1.json`：固定事前估计与 12 个独立实现成本点。
- `scripts/run_workload_v0_1_cost_mismatch_robustness.py`：0-call 成本失配评估，exact 明确为事后 oracle。
- `configs/acceptance/production_action4_independent_repeat_v1.json`、
  `scripts/run_production_action4_independent_repeat.py`：三次交错 restart/recovery、总 12-call 的固定实现复测。
- `src/runtime/production_action4_state.py`：create-only 状态导出、目标模型门禁、完整校验和执行权提交。
- `src/data/workflow/measurement_calibrated_vec_workload.py`：v0.1 生成器与三方法 bounded evaluator。
- `configs/experiment/measurement_calibrated_vec_workload_v0_1.json`：RQ、测量校准、8 点、3 seed、方法、目标与预算冻结。
- `scripts/run_measurement_calibrated_vec_workload.py`：生成 24 个实例并一次性输出 72 方法行和完整性回执。
- `artifacts/analysis/production_action4_workflow_20261005_v2/`：最终 real technical 正负链与成本分解。
- `artifacts/analysis/measurement_calibrated_vec_workload_20261005_v4/`：最终 24 实例、72 方法行、数据卡和完整性证据。
- `artifacts/analysis/workload_v0_1_cost_mismatch_robustness_20261005_v1/`：12 点、36 行估计/实现分离结果。
- `artifacts/analysis/production_action4_independent_repeat_20261005_v1/`：三次交错同版本配对复测及逐进程原始回执。
- 同名 real v1、workload v1/v2/v3 保留为非最终执行/代码失败证据，不覆盖、不冒充独立样本。
- `tests/test_production_action4_state.py`、`tests/test_measurement_calibrated_vec_workload.py`：正负接线与矩阵合同。
## 2026-10-09 冻结窗口身份校验

- `src/evaluators/main_results_support.py`：普通 frozen-plan mobility loader 在返回 bundle 前比较计划与实际窗口身份。
- `scripts/benchmark_main_results.py`、`scripts/train_algo_pool_real_sample.py`、`scripts/train_sa_ghmappo_real_sample.py`：把计划身份传到共享 loader。
- `tests/test_frozen_window_loader_identity.py`：覆盖同 offset 错 source 与原始区间错位的 rollout 前拒绝。
- `docs/project/cscwd_2027_baseline_contribution_audit_20261009.md`：记录旧 v70 审查与新基线实验边界；旧 artifact 原样保留。

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

## 2026-10-05 shared cache × recovery coupling additions

- `configs/experiment/shared_cache_recovery_coupling_v1.json`：A/B/C 三点、方法权限、成本目标、命令和输出清单的事前冻结协议。
- `scripts/run_shared_cache_recovery_coupling.py`：只走正常 `GymVecEnv.step` 的两分支原生见证入口；不训练 RL。
- `tests/test_shared_cache_recovery_coupling.py`：A/B/C later-load 差异与 `reset()` 切断跨 workflow resident state 的局部合同测试。
- `artifacts/shared_cache_recovery_coupling_20261005_v1/`：单次执行 JSON/CSV、逐事件 C witness、create-only state packages、receipt 与 integrity manifest。
- `docs/project/shared_cache_recovery_coupling_{plan,audit}_20261005.md`：事前冻结和事后审查。

## 2026-10-09 critic PopArt development A/B additions

```text
src/trainers/popart.py
configs/experiment/calibrated_workflow_value_normalization_ab_v1.json
configs/experiment/calibrated_workflow_value_normalization_ab_authorized_v1.json
scripts/preflight_calibrated_workflow_value_normalization_ab.py
scripts/run_calibrated_workflow_value_normalization_ab.py
scripts/launch_calibrated_workflow_value_normalization_ab.py
tests/test_popart_value_normalization.py
tests/test_calibrated_workflow_value_normalization_ab.py
tests/test_calibrated_workflow_value_normalization_launcher.py
docs/project/calibrated_workflow_value_normalization_ab_execution_protocol_20261009.md
artifacts/analysis/calibrated_workflow_value_normalization_ab_preflight_20261009_v1/
artifacts/benchmarks/calibrated_workflow_value_normalization_ab_20261009_v1/
```

后两个 artifact root 为 create-only runtime 产物；其中 `checkpoints/` 只本地保存，禁止提交或上传。
