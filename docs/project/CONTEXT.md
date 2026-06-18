# Context

更新日期：2026-06-18

用途：记录 PPO_MEC 当前稳定上下文。这里写长期有效事实，不写单次运行细节。

## 项目状态

- 项目：`PPO_MEC`
- 定位：面向 AI-driven VEC 的研究原型
- 主线问题：跨 RSU 连续 DAG workflow 执行、车载 base model 与路侧 adapter cache 协同、handoff 状态迁移、多时间尺度控制
- 当前正式数据主线：`NGSIM + Alibaba`
- 当前数据源声明入口：`docs/project/DATASET_SOURCES.md`
- 当前外部 model-cache metadata source：Hugging Face `model-cache` 候选全集已审计并 metadata-only 接入；当前只支持数据源声明和真实 file-size/cache-volume profile 设计，不替换 benchmark 默认 cache 行为
- 当前论文协议：`paper_protocol_v1_20260409`
- 当前正式结果入口：`docs/project/ARTIFACT_RECORDS.md`
- 当前顶刊审查规范：`docs/project/top_journal_review_policy.md`；最新审查为 `docs/project/top_journal_readiness_audit_20260618.md`。
- 文档记录的 canonical 为 `final_submission_v7_latency_fallback_20260528_v1`，但当前主机缺少其 closed-loop/final-submission 原始 artifact root；在恢复并校验前，TMC readiness 只能标 `Unverifiable`。
- 当前 live 模型层：主方法 `sa_ghmappo` + 方向匹配型对照算法池；`mappo` 对照采用 controller-level CTDE + `aggregation_reason_weighted_controller_ppo_v3`。
- 当前新增后 live paper-grade learned 对照算法池：`ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl`
- 当前 optional learned 变体：`ddqn`、`dueling_ddqn`；只有 duplicate trace audit 证明独立后才能作为补充对照。
- 当前 diagnostic / contract-blocked 对照：`ippo`；当前 single-wrapper contract 不支撑独立 IPPO。`mappo` / `qmix` / `controller_mat` 已实现为 controller-level CTDE / value-decomposition / transformer baselines，不是 vehicle-agent / RSU-agent full MARL wrapper。
- 当前非学习启发式对照：`reactive_greedy`、`popularity_cache_heuristic`
- 历史 artifact 路径中仍可能出现 `flat_ppo` / `flat_mappo` run 名称，但它们不再作为 live agent 注册。
- 当前未注册骨架算法：`td3`、`sac`、`maddpg`；后续接入前必须先冻结匹配的 observation/action contract

## 正式入口

- 数据检查：`python scripts/check_data_ready.py`
- NGSIM 检查：`python scripts/run_ngsim_sample.py --max_rows 500`
- Alibaba 检查：`python scripts/run_alibaba_sample.py --limit_jobs 3 --min_tasks 5 --max_tasks 20`
- 真实 dry-run：`python scripts/run_real_sample_dryrun.py --mobility_source ngsim --workflow_source alibaba --max_mobility_rows 1500 --max_workflows 3 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_selector max_handoff_candidate --window_length 24 --max_steps 12`
- 主结果：`python scripts/benchmark_main_results.py`
- Baseline 闭环：`python scripts/run_baseline_experiment.py --config configs/experiment/baseline/smoke.yaml`
- 对照算法训练：`python scripts/train_algo_pool_real_sample.py`
- 对照算法评估：`python scripts/eval_algo_pool_real_sample.py`
- 消融：`python scripts/benchmark_ablation.py`
- 鲁棒性：`python scripts/benchmark_robustness.py` / `python scripts/benchmark_prediction_robustness.py`
- 可扩展性：`python scripts/benchmark_scalability.py`

## 当前可引用结论边界

- 文档化结果：`final_submission_v7_latency_fallback_20260528_v1` 是 2026-05-28 记录的 paper-ready canonical，正式摘要见 `ARTIFACT_RECORDS.md`；当前主机未恢复原始 artifact，不能把文档化数值写成本轮已硬复核结论。
- 当前主结果 claim 边界：恢复 artifact 后必须按 `top_journal_review_policy.md` 重新核验 formal/holdout/support、统计和 checkpoint provenance。`mappo` / `qmix` / `controller_mat` 是 controller-level baselines，不是 vehicle-agent / RSU-agent full MARL wrappers；`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl` 是当前 contract 下的领域专项 learned baselines；`ippo` 仍为 diagnostic/contract-blocked；`reactive_greedy` / `popularity_cache_heuristic` 只作 supplementary reference。
- 注意：历史记录可包含旧对比算法；这些只代表归档结果，不代表当前方向匹配算法池的 live 结果。
- 可引用但需说明限制：早期 robustness 最新保留记录，协议早于 frozen main table，不应单独支撑最终主张。
- 不再引用：toy benchmark、tmp quickcheck、LuST micro 激活窗口试验、早期 dry-run、阶段性 reward shaping / recalibration / uncertainty tuning。

## 当前保留原则

- `artifacts/paper/` 只保留当前 canonical paper record。
- `artifacts/benchmarks/` 只保留主线可引用报告和每类最新有效报告。
- `artifacts/training/` 只保留被保留 benchmark 引用的 checkpoint run。
- `docs/project/` 是唯一长期文档目录。
- `maintainable_engineering_docs(1)/` 和旧 `docs/*.md` 阶段文档不再作为事实来源。

