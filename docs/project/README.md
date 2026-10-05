> **2026-09-08 追加勘误（G14R20-A）**：G14R18 补验任务新增正式执行为零，不表示既有 v16 未创建。
> 既有 v16 已完成 150 train cells、24 dev cells、1,200 candidates、150 selected/frozen checkpoints 与 6 个 generated resources；
> 无 failed terminal，尚无后续 formal 阶段。恢复门禁受阻不等于 run 失效。下文历史正文保留；涉及项目当前状态时以本勘误为准。
> 独立合同、原声明/hash 及证据指针见 [fixed_commit_continuation_contract.md](fixed_commit_continuation_contract.md)。执行器未实现、独立批准未签发，`execution_authorized=false`。

﻿# Project Maintenance Docs

这是 PPO_MEC 的项目化维护文档入口，用来把通用 AI 协作规范落到当前仓库。

## Live 文档

- `real_cache_victim_reload_plan_20261005.md`：真实 SmolVLM base + Helmet/ALPR adapter 的 12-call 事前冻结方案；
  对称核算 restart/recovery、合法 adapter-only victim、后续 reload 与无驱逐对照，尚未读取新模型输出

- `manuscript_evidence_progress_20261005.md`：论文 v1.1 的成果、未解决问题与后续追加记录规范；无新增实验。

- `independent_recovery_cost_measurement_plan_20261005.md`：独立成本检查的实例、随机顺序、既有校准、事前预测、
  24-call 硬预算、评分和未覆盖条件；模型输出产生前冻结
- `independent_recovery_cost_measurement_results_20261005.md`：24/24 调用、6 次逐项误差/动作/原始重复、输出保真、
  单侧无边界结论、最强主张与两个投稿缺口

- `eviction_aware_recovery_results_20261005.md`：12 点四方法完整结果、逐事件解释、主张—实验—原件—限制表与
  投稿关键缺口；新规则与正确两步前瞻持平，证据仅为 bounded implementation correction
- `eviction_aware_recovery_plan_20261005.md`：事前冻结公式、字段、复杂度、伪代码、12 点与执行边界

- `workload_v0_1_self_consistency_audit_20261005.md`：workload v0.1 自证循环、seed 有效输入、sharing 语义、
  version-specific real run 审查，以及 12 点成本失配与 12-call 固定实现复测结果；保留简单阈值，不启动 RL

- `measurement_calibrated_vec_workload_v0_1.md`：production action 4 technical 正负链、RQ—机制—假设—指标、
  workload v0.1 数据卡/来源表、全部 72 方法行、收益/无收益/交换区和唯一下一步

- `mechanism_evidence_closure_results_20261005.md`：共享缓存公平成本、两节点重跑/恢复和 12 场景决策空间结果；
  明确 Pareto 边界、ALPR 负结果和不启动 RL 的结论
- `mechanism_evidence_closure_plan_20261005.md`：本轮命令、调用、墙钟、12 格与状态成本的事前冻结方案

- `two_node_workflow_suffix_recovery_acceptance_20261003.md`：固定 4-generate 三进程见证、状态/输入/token 对照、
  负例、单次成本分层与严格 claim boundary；任务正确性 unavailable

- `two_node_workflow_suffix_recovery_plan_20261003.md`：真实两节点 technical workflow 的前缀保存、独立进程
  后缀恢复、负例、目标端读取边界和 4-generate 一次性验收方案；不代表任务正确性或 production action 4

- `adapter_state_recovery_calibration_20260930.md`：同 base 两个公开 adapter 的固定资源清单与授权门槛、
  独立进程状态恢复失败见证、action 4 语义及原生/敏感性分层；adapter 兼容性和真实净收益均未验证

- `remaining_workflow_decision_value_audit_20260930.md`：原生四配置账本、本机base/应用状态最小实测、动作可达性与
  16-episode有界配对见证；真实adapter/完整状态/网络成本仍缺，不支持算法晋级

- `native_typed_cache_request_audit_20260930.md`：固定四配置、288请求的原生 typed-cache 逐请求对账、首个
  dependency-safe feasible-set 差异、LRU 决策空间与 claim boundary；不是机制修复或训练结果

- `driving_workflow_minipackage_20260929.md`：公开来源检索、单场景引用包及成功后扩展条件；不是已运行数据集

- `driving_pilot_resource_setup_20260929.md`：已授权模型下载与隔离安装交接，图像条款待确认；非完成回执

- `driving_workflow_pilot_readiness_20260929.md`：模型/12图资源定位、16调用计划和实验前缺口；不宣称真实推理已完成

- `drivelm_workflow_preview_20260929.md`：9个驾驶工作流派生预览及参考答案隔离，接口通过但未执行模型

- `drivelm_sample_qualification_20260929.md`：官方 demo 字节/结构核验；790 QA 关系字段为空，区分驾驶语义与派生执行 DAG

- `vec_ai_public_resource_recovery_20260928.md`：找回旧19-source登记，DriveLM/V2X-Seq与请求/cache数据复核；复用优先，不从零编造语义

- `vec_ai_workload_stage_one_20260928.md`：数据集首阶段本地资源核查、最小车载应用模板、旧新字段映射及有界测量草案；模型输入待确认

- `vec_ai_workload_dataset_design_20260928.md`：车联网AI工作流/缓存数据集贡献候选，提出依据、校准生成方案、D01–D12证据池和验收边界；尚未生成或发布

- `system_mechanism_manuscript_working_draft.md`：系统机制主线英文工作稿，含问题、模型、方法正确性边界、开发结果、限制与独立证据映射；不是投稿终稿

- `problem_literature_traceability_20260928.md`：P01–P07问题、L01–L10论文、官方出处/等级边界、补证与失败判据的固定追溯索引

- `research_problem_and_evidence_plan_20260928.md`：主问题/三个子问题、近邻文献边界、设计实现差距、失败成本独立复算与有停止规则的补证路线

- `g14s01_innovation_algorithm_diagnosis_20260928.md`：相对最近邻的创新边界、formal/capacity/holdout 原件复核、
  controller-level 算法身份、核心机制调用证据、576/864 MB 非绑定根因，以及两阶段最小补证实验与 claim freeze

- `g14a01_formal_results_independent_review_20260921.md`：G14E07 formal-only 原始统计方向、window-level
  统计单位、延迟缺失、三容量、消融/support、oracle 边界与 holdout 前 claim freeze 的独立复核

- `formal_checkpoint_provenance_envelope_contract.md`：G14R18 17-field provenance envelope / 8-field shared identity
  边界、真实 companion→benchmark gate 验收、Protocol 2.9、Readiness v21 与 G14C v16 启动授权暂缓

- `formal_checkpoint_nullable_identity_contract.md`：G14R17 checkpoint identity producer/read-back/pre-dev/pre-sort/
  freeze/provenance closure、G14C v15 永久失败边界、Protocol 2.8 与 Readiness v20

- `formal_cell_artifact_publication_contract.md`：G14R15 统一 cell identity/staging/descriptor/payload validation/
  atomic publication/marker-ledger recovery，并冻结 Protocol 2.6、registry recovery、gate completion 与 Readiness v18

- `formal_generated_checkpoint_resource_identity_contract.md`：G14R14 static/generated 双层资源、create-only
  freeze publication、downstream consumer closure、capacity mapping、exact gate 与 Readiness v17

- `formal_protocol_capability_routing_contract.md`：G14R13 fail-closed capability registry、Protocol 2.4、
  persisted context outer/nested identity、G14C v13 pre-execution boundary 与 Readiness v16

- `active_formal_bundle_contract.md`：G14R7A Active Formal Bundle Contract 1.0、Protocol v1.8、唯一active
  index、Readiness v10原子finalization、outer pre-write gate与全链provenance

- `formal_agent_order_contract.md`：G14R7 order contract 1.0、Protocol v1.7、15-agent序列、consumer
  fail-fast、G14C v7永久拒绝、clean non-formal验收与Readiness v9边界

- `typed_model_cache_formal_training_identity_contract.md`：G14R6 scientific config 2.0、runtime execution
  binding 1.0、resolved context 2.0、checkpoint/downstream provenance 与 Readiness v8

- `typed_model_cache_formal_protocol.md`：G14B-G14R2 agent/seed/budget/capacity/metrics/statistics/claims/
  execution/holdout seal 冻结合同
- `typed_model_cache_formal_window_consumption_contract.md`：G14R2 raw/provider offset 语义、source-range
  推导、60-window reachability、command binding、ledger v2 与 Readiness v4
- `typed_model_cache_formal_portable_resource_contract.md`：G14R3 content-addressed external resources、fairness/window/
  checkpoint location、dev workflow binding、clean-tree rehearsal 与 Readiness v5
- `typed_model_cache_split_exclusion_audit.md`：G14B 全历史 interval ledger、NGSIM inventory、24/12/12/12 split 与 pairwise independence 审计

- `model_cache_dataset_discovery_audit_20260819.md`：G11 public model-serving/KV/model-artifact dataset taxonomy、qualification、HF复核、mapping与claim boundary
- `cache_information_sufficiency_marl_audit_contract.md`：G10 observation coverage、recoverability、aliasing、information gain和entity-level MARL necessity门禁
- `cache_request_replay_contract.md`：G08 policy-neutral request replay schema、fingerprint和outcome隔离规则
- `future_horizon_cache_oracle_contract.md`：G08 exact rolling finite-horizon oracle、可行域、objective、gap和artifact合同
- `cache_oracle_identifiability_feasibility_audit.md`：G08修改前的request内生性、真实时序和可识别性源码审计

- `../../AGENTS.md`：AI 协作硬约束和项目主线规则
- `CONTEXT.md`：当前稳定上下文、正式入口和结论边界
- `PROGRESS.md`：已确认阶段事实和整理动作
- `BUGS.md`：当前有效问题、风险和禁止误读项
- `ARTIFACT_RECORDS.md`：从 `artifacts/` 整理出的规范化实验记录
- `research_skill_integration_20260727.md`：外部科研 Skill 的选择性采用边界、项目原生证据链和 Policy-Learning Gate
- `sa_ghmappo_v47_v51_learning_audit_20260727.md`：v47--v51 dev 阶段训练/动作归因审计；当前不可晋级
- `current_results_audit_20260527.md`：当前 canonical / v5 / MAPPO v3 / SA v6 结果状态、缺口和阻塞审计表
- `top_journal_review_policy.md`：以 IEEE TMC 为主目标的长期 AI reviewer 证据等级、blocker、评分和固定输出规范
- `top_journal_readiness_audit_20260621.md`：strict-full v8 formal、一次性 hidden 与 LuST supporting evidence 的最新审查；当前 verdict 为 `Major revision`
- `strict_full_v8_execution_record_20260621.md`：v8 候选冻结、hidden 开启、运行语义、统计与完整性审计记录
- `sa_ghmappo_paper_method_report_20260716.md`：面向论文 Problem Formulation / Method / Algorithm Design 的主算法详细报告，固化研究问题、状态/动作合同、encoder、三控制头、PPO objective、guard 和写作边界
- `sa_ghmappo_current_results_feasibility_20260716.md`：当前主算法结果可行性报告，单独展示 strict-full v8 formal/hidden、全部 learned baseline、DT continuity 与 LuST supporting evidence
- `RUNBOOK.md`：包含 v8-current support suite 与 v9 Pareto-safe 候选的运行入口；这些入口不等于新 paper-grade 结果
- `top_journal_readiness_audit_20260618.md`：v7 strict non-overlap 历史审查；verdict 为 `Not TMC-ready`
- `novelty_review_20260621.md`：面向 TMC 的最新一手文献检索、最近邻矩阵和四项创新点新颖性审查；结论为 `Conditionally defensible, but crowded`
- `advisor_report_briefing_20260621.md`：面向导师汇报的创新点、整体框架、模型结构、实验结果、结论边界和问答讲稿
- `../../outputs/ppo_mec_advisor_report_20260621.pptx`：与讲稿配套的 14 页可编辑导师汇报 PPT，含逐页 speaker notes
- `CLEANUP_LOG.md`：旧文档和旧产物清理记录
- `DIRECTORY_STRUCTURE.md`：目录边界和产物写入位置
- `DATASET_SOURCES.md`：当前数据集名称、角色、本地路径和下载页声明
- `literature_reference_table.md`：顶刊/顶会 related-work 参考表，记录每篇论文可引用点和 PPO_MEC 相对优化点
- `../../configs/data/hf_model_cache_integration_plan.json`：HF model-cache 候选审计后的接入边界和 importer 前置条件
- `CODE_MODULE_MAP.md`：代码模块职责和主要依赖方向
- `RUNBOOK.md`：常用运行、验证、训练和 benchmark 命令
- `DECISION_LOG.md`：长期有效的设计和流程决策
- `STATUS_TAGS.md`：文档状态标签约定
- `ALGO_POOL.md`：方向匹配型强化学习对照算法池状态和运行入口
- `../benchmark_plan_or_baseline_plan.md`：baseline 盘点、对照矩阵和统一训练评估协议
- `../baseline_formalization_round1.md`：baseline formalization round1 机制差异诊断
- `../experiment_status_round1.md`：formal experiment execution round1 执行状态总表
- `../mechanism_activation_check_round1.md`：round1 机制触发诊断
- `../experiment_runbook_round1.md`：round1 正式复跑命令

## 使用方式

开始新任务时先读 `../../AGENTS.md`、`CONTEXT.md`、`PROGRESS.md`、`BUGS.md` 和本文件，再读相关脚本、配置和模块。  
改动代码后，根据影响面更新对应文档；只影响实现细节且入口、路径、协议、产物不变时，不需要机械更新所有文档。

## 模板来源

通用模板内容已整理进本目录。当前事实来源只保留 `docs/project/` 和根目录 `AGENTS.md`。

## G13 type-aware model cache

- 合同：`typed_model_cache_contract.md`
- 验证报告：`typed_model_cache_validation_report.md`
- 机器证据：`../../artifacts/analysis/typed_model_cache_validation_20260819_g13_v1/`
- 默认继续使用 legacy adapter-only profile；typed profile 必须显式启用。
- 顺序重算LRU候选：`native_typed_cache_replacement_witness_20260930.md`；仅显式non-formal candidate，旧v1.0默认不变。

## G14A typed MB runtime plumbing

- 合同：`typed_model_cache_runtime_contract.md`
- 验证报告：`typed_model_cache_runtime_validation_report.md`
- 机器证据：`../../artifacts/analysis/typed_model_cache_runtime_plumbing_validation_20260819_g14a_v1/`
- 状态：plumbing/rehearsal通过；后续 G14B 已冻结 split/protocol 并通过 readiness v2，但正式 checkpoint 仍不存在。

## G14B formal protocol freeze

- 合同：`typed_model_cache_formal_protocol.md`
- 排除审计：`typed_model_cache_split_exclusion_audit.md`
- 机器证据：`../../artifacts/analysis/typed_model_cache_formal_protocol_freeze_20260820_g14b_v1/`
- 状态：`READY_FOR_G14C_CLEAN_TRAIN_AND_FORMAL`；formal episode/checkpoint/performance result均为0，holdout sealed/unopened，不是paper-ready。

## G14R3 portable execution path repair

- Protocol v1.3 semantic：`1525b7cb...ac17`
- Readiness v5：`READY_FOR_G14C_V4_CLEAN_TRAIN_AND_FORMAL`
- 机器证据：`../../artifacts/analysis/typed_model_cache_formal_path_repair_20260821_g14r3_v1/`
- 状态：portable binding 与 non-formal exact rehearsal 已通过；G14C v4/formal/holdout/G15 尚未启动。

## G14R4+ transactional portable execution repair

- 环境合同：`typed_model_cache_formal_execution_environment_contract.md`
- Resume 合同：`typed_model_cache_formal_execution_resume_contract.md`
- Protocol v1.4 semantic：`4429531d...4155d`
- Readiness v6：`READY_FOR_G14C_V5_CLEAN_TRAIN_AND_FORMAL`
- 机器证据：`../../artifacts/analysis/typed_model_cache_formal_execution_repair_20260825_g14r4_v1/`
- 状态：no-.venv exact rehearsal 与 transaction/resume/finalize 验证通过；G14C v5/formal/holdout/G15 未启动。

## G14R5 resolved formal execution context repair

- 合同：`typed_model_cache_formal_resolved_execution_context_contract.md`
- Protocol v1.5 semantic：`feb7ccc4...d829a`
- Readiness v7：`READY_FOR_G14C_V6_CLEAN_TRAIN_AND_FORMAL`
- 机器证据：`../../artifacts/analysis/typed_model_cache_formal_preflight_context_repair_20260825_g14r5_v1/`
- 状态：detached no-.venv clean preflight/tests 已通过；G14C v5永久invalid；正式training/checkpoint/performance为0，
  holdout sealed/unopened，G14C v6/G14D/G15未启动。

## G14R6 formal training identity repair

- 合同：`typed_model_cache_formal_training_identity_contract.md`
- Protocol v1.6 semantic：`f2c9e729...a95c0`
- Scientific config semantic：`f83587cd...49bc8`
- 机器证据：`../../artifacts/analysis/typed_model_cache_formal_training_binding_repair_20260825_g14r6_v1/`
- 状态：只完成配置/执行身份合同与非正式验收；G14C v6永久invalid，正式training/checkpoint/performance仍为0，
  holdout sealed/unopened，未启动G14C v7/G14D/G15。

## G14R7A active formal bundle closure

- 合同：`active_formal_bundle_contract.md`
- Protocol v1.8 semantic：`9799bf2c...b3de`
- active bundle core/final：`96627ac4...5b65` / `793f5106...38bd`
- 机器证据：`../../artifacts/analysis/typed_model_cache_formal_active_bundle_closure_20260827_g14r7a_v1/`
- 状态：Readiness v10与ready index一致；v1.0–v1.7 audit-only。只完成pre-execution gate与clean验收，
  正式training/checkpoint/performance仍为0，holdout sealed/unopened，未启动G14C v8/G14D/G15。
# 2026-08-31 formal request contract

- `formal_exogenous_request_execution_contract.md`：Protocol 2.0 外生 request exposure、因果信息边界、Endpoint 2.0 与 fail-fast 规则。

# 2026-08-31 formal environment projection contract

- `formal_environment_identity_projection_contract.md`：Protocol 2.1 full scientific environment projection、
  Protocol-bound extensions、canonical fingerprint、host audit 与 v10 pre-execution stop 边界。

# 2026-09-02 formal request subject lifecycle contract

- `formal_request_subject_lifecycle_contract.md`：Protocol 2.2 持续主体资格、冻结选择证据、RSU/time 对齐、
  runtime reselection 禁止、analytical replay闭环与 v11 terminal 边界。

# 2026-09-04 formal nullable metric contract

- `formal_nullable_metric_aggregation_contract.md`：Protocol 2.3 的 finite/null、required missing、CSV/JSON、Dev
  selection、paired statistics、Holm、gate/claim `UNAVAILABLE` 规范。
- 机器证据：`../../artifacts/analysis/typed_model_cache_formal_nullable_metric_repair_20260903_g14r12_v1/`
- 状态：v12 永久 invalid；exact 256-episode 与 13-phase non-formal rehearsal 已闭环，正式
  training/checkpoint/performance仍为0，holdout sealed/unopened，未启动G14C v13/G14D/G15。

## 2026-10-05 shared cache × recovery coupling

- 事前冻结：`shared_cache_recovery_coupling_plan_20261005.md`
- 事后审查：`shared_cache_recovery_coupling_audit_20261005.md`
- 机器证据：`../../artifacts/shared_cache_recovery_coupling_20261005_v1/`
- 当前边界：同一 episode 内有限容量 resident-state 耦合成立；跨 workflow 持久缓存及共享无线/计算资源未实现；研究决定 3，不启动 RL。
