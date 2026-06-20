# Directory Structure

## 根目录

- `README.md`：项目定位、当前阶段、主线命令和实验入口总览
- `AGENTS.md`：AI 协作和维护规则
- `configs/`：正式实验、baseline 协议和消融相关 manifest
- `configs/data/`：统一数据源声明 manifest 和 HF model-cache 接入方案
- `configs/algo/`：方向匹配对照算法配置
- `configs/experiment/baseline/`：baseline 训练、评估和 benchmark 闭环配置
- `configs/experiment/top_journal_mechanism_v1.yaml`：顶刊路线机制训练 profile 与 benchmark 计划
- `data/`：原始数据与处理后数据；通过 Git LFS 版本化，完整克隆后需执行 `git lfs pull`
- `docs/`：长期维护文档，`docs/project/` 为事实来源，`docs/project/DATASET_SOURCES.md` 记录数据源声明，`docs/project/literature_reference_table.md` 记录顶刊/顶会 related-work 参考表，`docs/benchmark_plan_or_baseline_plan.md`、`docs/baseline_formalization_round1.md`、`docs/experiment_status_round1.md`、`docs/mechanism_activation_check_round1.md` 和 `docs/experiment_runbook_round1.md` 记录 baseline 计划、round1 状态、机制诊断与复跑命令
- `scripts/`：数据检查、dry-run、训练、评估和 benchmark 入口
- `scripts/run_top_journal_final_submission_loop.py`：最终交稿 learned-primary 自循环入口，编排 learned baseline 重训、formal/holdout gate、cluster bootstrap statistics 和 support suites
- `scripts/build_top_journal_comparison_report.py`：最终交稿 comparison package 生成入口，汇总 baseline protocol matrix、reward margins、mechanism paired statistics、support statistics、paper-ready LaTeX 表格和作者自审报告
- `scripts/audit_artifact_integrity.py`：run-root SHA-256、JSON path reference、external dependency 和 parse error 审计
- `scripts/audit_window_independence.py`：formal/holdout selected window plan 的 split 内与 split 间 frame interval 独立性审计
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
- `src/data/`：mobility、workflow 和 model catalog 数据层
- `src/encoders/`：DAG、RSU 状态和融合编码器
- `src/envs/`：核心环境、预测层和 Gym/vector wrapper
- `src/envs/specs/action_schema.py`：语义动作 schema、mask 和 action adapter
- `src/evaluators/`：真实 sample、主结果和 checkpoint 评估辅助
- `src/metrics/`：episode recorder、指标 reducer 和论文指标
- `src/trainers/`：PPO/MARL 训练驱动和 buffer
- `src/utils/`：通用工具

## 产物目录

- `outputs/ppo_mec_advisor_report_20260621.pptx`：基于 E3 复现证据整理的导师汇报 deck；数据来源与结论边界见 `docs/project/advisor_report_briefing_20260621.md`
- `artifacts/training/`：被保留 benchmark 引用的训练 run、checkpoint 和训练审计
- `artifacts/training/algo_pool/`：方向匹配对照算法训练产物
- `artifacts/training/algo_pool_formal_round1/`：round1 三 seed formal flat baseline 训练产物
- `artifacts/eval/algo_pool/`：方向匹配对照算法评估产物
- `artifacts/experiments/baseline/`：config-driven baseline 闭环产物、per-seed manifest、comparison summary 和 by-window-class summary
- `artifacts/experiments/top_journal_closed_loop/`：顶刊路线闭环产物，包括训练记录、seed checkpoint manifest、benchmark aggregate 和 gate report
- `artifacts/experiments/top_journal_learned_baseline_suite/`：learned-baseline strict gate 产物；当前新 run 的 paper-grade 默认 learned set 为 PPO/MAPPO/DQN/Dueling-DQN/QMIX/Controller-MAT/DAG-Offload-DRL/Cache-Offload-DRL/DT-Handoff-DRL，IPPO 旧产物只作 diagnostic/历史审计
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_*`：补充 Dueling-DQN / Dueling-DDQN 后的 learned-baseline 扩展 gate 和 holdout 产物
- `artifacts/experiments/top_journal_sa_iteration/`：主方法优势迭代和候选验证产物，包括 v2/v3 retrain、eval-bias manifest、screen benchmark 和 learned gate；负向迭代不作为 paper-grade 主表
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_*`：当前 v3 eval-bias formal/holdout gate refresh。
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_support/`：v3 eval-bias latency fallback 消融、prediction robustness、robustness 和 scalability 支撑产物。
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v4_prepare_eval_bias*`：v4 prepare override 负向筛选产物，不作为主结果。
- `artifacts/experiments/top_journal_final_submission/`：最终交稿闭环产物；`final_submission_v7_latency_fallback_20260528_v1` 是 legacy paper-ready package，`final_submission_v7_latency_fallback_20260618_rebuild_v1` 为 E3 independent rebuild。旧 offset=3 与 formal 重叠，不能作为 strict canonical holdout；最新 readiness 以 `top_journal_readiness_audit_20260618.md` 为准。
- `artifacts/benchmarks/`：当前可引用的主结果、预测鲁棒性、消融、robustness 和可扩展性 benchmark
- `artifacts/analysis/hf_model_cache_dataset_audit_round14/`：HF model-cache 候选适配性审计产物
- `artifacts/paper/`：历史 paper export；legacy v7 表格只能在明确标注 overlap limitation 时使用，strict reviewer 结论以最新审计为准

新产物应写入明确的 run 目录，不应散落到仓库根目录。

