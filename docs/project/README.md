# Project Maintenance Docs

这是 PPO_MEC 的项目化维护文档入口，用来把通用 AI 协作规范落到当前仓库。

## Live 文档

- `../../AGENTS.md`：AI 协作硬约束和项目主线规则
- `CONTEXT.md`：当前稳定上下文、正式入口和结论边界
- `PROGRESS.md`：已确认阶段事实和整理动作
- `BUGS.md`：当前有效问题、风险和禁止误读项
- `ARTIFACT_RECORDS.md`：从 `artifacts/` 整理出的规范化实验记录
- `current_results_audit_20260527.md`：当前 canonical / v5 / MAPPO v3 / SA v6 结果状态、缺口和阻塞审计表
- `top_journal_review_policy.md`：以 IEEE TMC 为主目标的长期 AI reviewer 证据等级、blocker、评分和固定输出规范
- `top_journal_readiness_audit_20260618.md`：v7 独立重建、严格非重叠 holdout、机制消融与 LuST 外部迁移的最新审查；当前 verdict 为 `Not TMC-ready`
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
