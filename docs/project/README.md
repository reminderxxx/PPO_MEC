## 2026-10-18 第二阶段已完成

[joint_stream_learning_diagnosis_review_20261018.md](joint_stream_learning_diagnosis_review_20261018.md)：DQN回放顺序一阶缺陷、15格参考全轨迹对账、同状态0/32/64评分、共同裁剪/物理链反例及第三阶段任务。训练/优化0；[执行合同](joint_stream_learning_diagnosis_contract_20261018.md)冻结7200评估帧，上限内完成7032。第三阶段未实施；下文阶段一“阶段二未执行”为历史。

## 2026-10-18 学习有效性阶段一

[joint_stream_learning_effectiveness_review_20261018.md](joint_stream_learning_effectiveness_review_20261018.md)：完整48条件0/32/64、8方法/3seed的学习增益、基线退化、全部72行模型指标、曲线和第二阶段任务点。仅开发证据，不证明收敛；阶段二未执行。

## 2026-10-17 最新完整回收

- [持续请求v2唯一确认轮独立审查](joint_stream_confirmation_results_review_20261017.md)：8组全部固定最终、费用/对象生命周期与baseline独立性；均值优势及seed17/预取/曲线覆盖负例，1/1已结束，非论文晋级。

## 2026-10-17 新增入口

- [持续请求v1独立回收审查](joint_stream_results_review_20261017.md)
- [持续请求v2有界确认合同](joint_stream_refinement_contract_20261017.md)

> **2026-09-08 追加勘误（G14R20-A）**：G14R18 补验任务新增正式执行为零，不表示既有 v16 未创建。
> 既有 v16 已完成 150 train cells、24 dev cells、1,200 candidates、150 selected/frozen checkpoints 与 6 个 generated resources；
> 无 failed terminal，尚无后续 formal 阶段。恢复门禁受阻不等于 run 失效。下文历史正文保留；涉及项目当前状态时以本勘误为准。
> 独立合同、原声明/hash 及证据指针见 [fixed_commit_continuation_contract.md](fixed_commit_continuation_contract.md)。执行器未实现、独立批准未签发，`execution_authorized=false`。

﻿# Project Maintenance Docs

这是 PPO_MEC 的项目化维护文档入口，用来把通用 AI 协作规范落到当前仓库。

## Live 文档

- [joint_stream_learning_contract_20261016.md](joint_stream_learning_contract_20261016.md)：持续请求八配置共同学习合同、固定训练/开发划分、预算与自动回收；配套学习前审、predictor partition审查、首次软件消费端失败审查同日保留。

- [joint_stream_workload_results_review_20261016.md](joint_stream_workload_results_review_20261016.md)：v3完整216单元零训练工程结果、配对负例、对象驻留使用与学习合同缺口。
- `joint_stream_workload_review_20261016.md` / `joint_stream_workload_contract_20261016.md`：工作负载空缺前审、独立持续请求/多模型合同和216单元零训练工程验证计划。

- [joint_delivery_v2_results_review_20261016.md](joint_delivery_v2_results_review_20261016.md)：十组3840回合固定最终结果、120checkpoint/主补充重放、预测开关/缓存成本、后期选站回退和剩余任务；双指标目标未达。

- `joint_target_control_engineering_20261014.md`：显式目标站、未来模型对象、状态/结果转发与共享数据链路的工程实现、参考来源与验收边界。
- `joint_reward_identity_audit_20261014.md`：9 个网络权重/加载/更新、72 次独立回放、未舍入回报与反事实动作检查；确认精确同分来自当前比较缺乏动作辨识力。
- `joint_mechanism_budget96_review_20261014.md`：多帧工作量、96 回合曲线、开发奖励同分与候选站容量上限的原件审查；不作为论文结果。
- `joint_control_v3_design_20261014.md`：可控执行站、base/adapter 预置、状态预复制和同合同算法对照的下一版设计草案；尚未实现。
- `joint_cold_mask_budget96_cache_baseline_review_20261014.md`：同冷缓存合同四倍训练量、缓存基线身份与特征、奖励上限和主方法可辨识性审查；仅开发证据。
- `eight_vehicle_baseline_validity_review_20261012.md`：纠正八车选站诊断与 SA-GHMAPPO/正式 MAPPO/论文方法对比的合同错位，明确有效 baseline 三类和重新运行门槛。
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

## 2026-10-15 参数化学习回收

- `joint_parameterized_results_review_20261015.md`：两轮3840 episode的奖励/完成、消融、费用收据、checkpoint重放与下一阶段问题；结果不支持SA主机制优势。

- `joint_coordinated_control_contract_20261015.md`：保留预测的协同选站、机会净收益迁移、prefix信用与七配置同预算方案。

- `joint_coordinated_gate_review_20261015.md`：严格完成条件排除9个离站请求帧的独立审查。
- `joint_partial_preparation_contract_20261015.md`：部分预复制/cache-only与三配置同预算再训练。

- joint_output_lifecycle_review_20261015.md：输出阶段冷模型约束的独立缺陷审查；修复及新同预算比较见joint_partial_preparation_contract_20261015.md。

- joint_coordinated_results_review_20261015.md：七配置同预算完整回收、原始账本及checkpoint重放审计，预测收益尚未证明。

- joint_partial_prediction_support_review_20261015.md：生命周期v3固定权重预测贡献及负例，主比较排序待完整回收。

- joint_feature_scale_review_20261015.md：首dev图表示尺度及激活饱和独立诊断，尚未做因果修复实验。

- joint_partial_control_results_review_20261015.md：生命周期v3三组同预算最终结果、全部原始账本/9模型重放、预测局部贡献和剩余缺口。

- joint_scheduled_control_contract_20261015.md：8帧预测/链路预约、动作信用及尺度修复，恢复七算法和去预测主比较、后期稳定性对照的事前合同。

- joint_scheduled_prediction_support_review_20261015.md：8帧预约执行、固定权重预测局部收益/负例、图尺度原始witness；完整排序见joint_scheduled_control_results_review_20261015.md。

- joint_scheduled_control_results_review_20261015.md：八主配置同预算最终排序、DQN信用勘误、120 checkpoint/30模型重放、跨seed回退与cache费用/交付边界。

- joint_tail_regression_review_20261015.md：seed27组件/逐头冻结干预，定位priority surrogate及尾部目标不一致。
- joint_tail_control_contract_20261015.md：独立tail实现、train-only cohort/双panel、八主两support固定预算及生产消费端。

- joint_tail_capacity_review_20261015.md：tight7/8必要容量上界不可全完成，区分过载与预测失效。
- joint_tail_mobility_support_contract_20261015.md：固定原布局/多站可选高移动的零训练探索支持协议，保留冻结主结果。

- joint_tail_control_results_review_20261015.md：十组完整回收、120checkpoint/60主重放、零训练移动分层/30重放、预测挤占冷模型负例和剩余目标/信用缺口。

- joint_delivery_control_contract_20261015.md：预测准备资源保护、团队节点信用、缓存残差与固定完整基线预算的新实现合同。

- joint_delivery_completion_guard_review_20261015.md：离站前结束却准备的有界错误、v1停止和成本记录。
- joint_delivery_control_v2_contract_20261015.md：保留predeparture交付、合同v2拒绝v1与从头同预算的独立修复。

- `joint_stream_fifo_review_20261016.md`：同帧名字排序破坏FIFO对照的独立审查与v1停止/v2修复边界。

- `joint_stream_admission_review_20261016.md`：延迟cache视图导致真实LRU误驱逐的独立witness、v2审计失败和共同本地准入保护v3。
