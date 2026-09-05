# Project Maintenance Docs

这是 PPO_MEC 的项目化维护文档入口，用来把通用 AI 协作规范落到当前仓库。

## Live 文档

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
