# Context

更新日期：2026-07-21

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
- 当前顶刊审查规范：`docs/project/top_journal_review_policy.md`；最新审查为 `docs/project/top_journal_readiness_audit_20260621.md`。
- v7 的 strict-full blocker 已由 v8 修复：冻结 20-window/split 协议下，formal/一次性 hidden 对全部 learned baselines 的 reward CI 为正，对 DT continuity 的 CI 也为正。当前 reviewer verdict 为 `Major revision (78/100)`，证据等级为 `E2_ARTIFACT_AUDITED`；PPO handoff failure/backhaul trade-off、v8-current support suite 与外部样本量仍未达 TMC-ready。
- 2026-07-21 v39-v41 dev-probe 显示：MAPPO-core / delayed-credit / advantage-weighted behavior regularization 可以让 SA-GHMAPPO 在 frozen dev 单 seed/20-window 上略高于 MAPPO、PPO 和 popularity heuristic，但仍未超过 `cache_offload_drl` 与 `dt_handoff_drl`。当前 `reward_positive_offset=5.0` 会按 step 累加，存在 reward ranking 与 workflow completion/continuity 不一致风险；v39-v41 不替代 v8/v20 记录，也不是 paper-ready / all-baseline-winner 结论。
- 当前 live 模型层：主方法 `sa_ghmappo` + 方向匹配型对照算法池；`mappo` 对照采用 controller-level CTDE + `aggregation_reason_weighted_controller_ppo_v3`。
- 当前 predictor 层：默认仍可使用 `baseline_predictor_v2`；代码已新增 `supervised_handoff_predictor_v1` 训练与 runtime 接口，需显式传入冻结 checkpoint。该层定位为短时 handoff anticipation / lightweight DT-style predictive state snapshot，不是完整数字孪生系统；在生成正式 checkpoint、quality report 和 v9 benchmark 前，不自动替代 v8 主结论。
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
- Supervised predictor：`python scripts/train_supervised_handoff_predictor.py`
- 可扩展性：`python scripts/benchmark_scalability.py`

## 当前可引用结论边界

- 文档化结果：`final_submission_v7_latency_fallback_20260528_v1` 是 legacy paper-ready package；`final_submission_v7_latency_fallback_20260618_rebuild_v1` 已复现旧协议数值，但 strict non-overlap 结果取代 offset-3 作为 readiness 判断依据。
- 当前主结果 claim 边界：可安全表述 v8 strict-full formal/一次性 hidden 对全部 learned baselines 的 reward CI 为正，且对 DT continuity CI 为正；必须同时报告相对 PPO 的 handoff-failure/backhaul trade-off及未超过 popularity heuristic。`mappo` / `qmix` / `controller_mat` 是 controller-level baselines，不是 vehicle-agent / RSU-agent full MARL wrappers。
- 当前 dev-probe claim 边界：v39 update_0005 full-pool SA-GHMAPPO `106.041` 高于 MAPPO/PPO/popularity，但低于 DT/cache；v41 conservative recovery `105.686` 只恢复稳定性，没有扩大 MAPPO 差距。不得把这组结果写为 all-baseline winner、canonical 晋级或投稿主 claim。
- 注意：历史记录可包含旧对比算法；这些只代表归档结果，不代表当前方向匹配算法池的 live 结果。
- 可引用但需说明限制：早期 robustness 最新保留记录，协议早于 frozen main table，不应单独支撑最终主张。
- 不再引用：toy benchmark、tmp quickcheck、LuST micro 激活窗口试验、早期 dry-run、阶段性 reward shaping / recalibration / uncertainty tuning。

## 当前保留原则

- `artifacts/paper/` 只保留当前 canonical paper record。
- `artifacts/benchmarks/` 只保留主线可引用报告和每类最新有效报告。
- `artifacts/training/` 只保留被保留 benchmark 引用的 checkpoint run。
- `docs/project/` 是唯一长期文档目录。
- `maintainable_engineering_docs(1)/` 和旧 `docs/*.md` 阶段文档不再作为事实来源。
