# Decision Log

## 2026-07-17: v17 用 DAG-aware option termination 约束 MAPPO 机制动作时机

决策：新增 `top_journal_mechanism_v17_dag_aware_option`，在 v16 conservative terminal option 基础上加入 DAG-aware option termination。该 gate 使用已有 graph-continuity critic 特征和 DAG topology：低置信 idle no-RSU predictive prefetch 被终止；机制窗口中，若 workflow 是短 DAG、critical path 低于阈值且当前节点分叉较多，则终止低机会 prefetch，改由 option 层选择 `popularity_safe`。

原因：v13 的 PRD-MAPPO 学习能提高机制收益，但少数 idle prefetch 会造成 backhaul blocker；v15/v16 在 reward 与 backhaul 之间来回摆动。逐行诊断显示 `window_off336.../j_3` 这类短 DAG 场景中“机制动作成功”不等于“总 reward 净收益”，而长 DAG 的 `j_8` 场景同类动作有明显正收益。因此需要把 MAPPO option 从 handoff/prediction evidence 升级为 DAG opportunity-aware termination，而不是继续扩大 hard rule 或改 reward。

影响：v17 不修改 `VecWorkflowCoreEnv` reward、不修改 `semantic_discrete_5` action contract、不改变 baseline contract、不读取 hidden。full-dev 证据显示 SA total reward `79.70825`，高于 v13 `79.64465`、v16 `79.627`、`popularity_cache_heuristic=79.46875`、`ppo=77.18775`、`mappo=72.6328` 和全部其他对照；`sa_advantage_diagnosis.blockers=[]`，backhaul 与 popularity 持平。该结果仍是 dev evidence，不能替代 future-validation / readiness audit。

## 2026-07-17: v13 用 partial-reward-decoupled MAPPO credit，并以 latest 评估学习后策略

决策：新增 `top_journal_mechanism_v13_prd_option`，在 v12 learned option gate 基础上加入 event-head 与 option-head 的 partial-reward-decoupled credit。`event_prd_advantage_*` 把机制准备、handoff readiness、prediction confidence 和机制窗口 context 转成 event advantage 补充项；`option_gate_prd_*` 让 option loss 学习机制动作和安全动作的部分信用。closed-loop 中 v13 使用 `latest_checkpoint_path` 优先，因为 full-dev 审计显示 `best_by_reward_path` 停留在 update 0 warm-start，无法代表 PRD 训练后的策略。

原因：v12 的 reward margin 已主要受机制窗口净收益限制，普通/idle 窗口多与 popularity 持平。继续扩大 hard rule 会被审稿人视为规则包装，也可能牺牲机制窗口；PRD-MAPPO 风格的部分信用分配更符合 MAPPO 的 CTDE 学习核心，目标是让 event/option head 从机制成功和净收益中学习，而不是只靠手写 fallback。

影响：v13 不修改 `VecWorkflowCoreEnv` reward、不修改 `semantic_discrete_5` action contract、不改变 baseline contract、不读取 hidden。full-dev latest 证据显示 SA total reward `79.64465`，高于 v12/best-by-reward `79.5934`、`popularity_cache_heuristic=79.46875`、`ppo=77.18775`、`mappo=72.6328` 和全部其他对照；strongest-other margin 从 `+0.12465` 扩大到 `+0.17590`。该结果仍是 dev evidence，不能替代 v8 canonical 或 future-validation 审查。

## 2026-07-17: v12 采用 learned contextual MAPPO option gate，而不是继续扩大 hard rule

决策：新增 `top_journal_mechanism_v12_learned_option`，在 v11 MAPPO-core reward-first checkpoint 上 warm-start，并在 SA-GHMAPPO policy 内加入四类 option head：`accept_mappo`、`popularity_safe`、`no_rsu_local`、`mechanism_prepare`。训练使用 PPO-style option loss、entropy 和随 update 衰减的 contextual prior；推理时机制窗口 preserve MAPPO 主策略，idle/sparse 才允许 learned popularity-safe option 接管。

原因：v11 full-dev 虽已超过 popularity，但 margin 只有 `+0.02565`，弱点集中在 idle/sparse；直接扩大规则 fallback 会牺牲机制窗口，而纯 learned option probe 会在低机制窗口和机制窗口之间误分配 credit。v12 把 MAPPO 的 controller-level credit assignment 留在主策略中，再让 option head 学习什么时候接受 MAPPO、什么时候借用规则安全动作，从而解决“MAPPO 与 PPO 拉不开、主算法偶尔低于规则”的一阶原因。

影响：v12 不修改 `VecWorkflowCoreEnv` reward、不修改 `semantic_discrete_5` action contract、不改变 baseline contract，也不读取 hidden。full-dev 证据显示 SA total reward `79.5934` 高于 `popularity_cache_heuristic=79.46875`、`ppo=77.18775`、`mappo=72.6328` 和全部其他对照；但这是 dev evidence，不能替代 v8 canonical 或现有 hidden/future-validation 审查。若要晋级，需要新冻结 future-validation split，并继续报告 PPO 在 failure/backhaul 上仍更优的 trade-off。

## 2026-07-16: v11 采用 MAPPO-core reward-first + idle/sparse window-context option gate

决策：新增 `top_journal_mechanism_v11_mappo_reward`，以 v8 strict-full scaffold 保留机制稳定性，同时迁入 MAPPO controller-level head-credit、entropy floors/scales 和 event advantage blend；checkpoint selection 使用 reward-first priority。推理期不全局替换 MAPPO 决策，只在 v11 checkpoint 且 `window_class=idle_or_sparse` 时打开 no-RSU local fallback，并在机制窗口保持 MAPPO 主策略和 vehicle-only fallback。

原因：first-order diagnosis 显示初始 v11 在机制窗口已经优于 popularity，但 idle/sparse 的 `vehicle_fallback` / no-RSU offload 行为拉低总 reward。全局 no-RSU local fallback 虽能改善 idle/sparse，却会压低机制窗口 reward 和 mechanism realization；因此需要把规则先验限定为 outcome-blind mobility regime 下的推理期 option gate，而不是替换 MAPPO 学习策略。

影响：v11 不修改 `VecWorkflowCoreEnv` reward、不修改 `semantic_discrete_5` action contract、不改变 baseline contract。full-dev 证据显示 SA total reward `79.4944` 高于全部对照，`sa_advantage_diagnosis.blockers=[]`；但这是 dev evidence，不能替代 v8 canonical 或现有 hidden/future-validation 审查。若要晋级，需要新冻结 future-validation split，并明确报告 idle/sparse 仍略低于 popularity 的边界。

## 2026-07-16: v10 用 MAPPO controller credit 强化 SA-GHMAPPO 的学习更新

决策：新增 `top_journal_mechanism_v10_mappo_rl` profile，在 v9 Pareto-safe 边界上显式迁入 MAPPO 强对照的 `aggregation_reason_weighted_controller_ppo_v3`、slow/fast/event policy credit floors、entropy floors/scales 和 `event_advantage_blend=0.85`。同时降低 `heuristic_imitation_coef`、`mechanism_aux_coef`、`mechanism_window_weight` 和 `prepare_action_prior_weight`，并关闭 `mechanism_aux_current_cache_fill_enabled`，让 idle/sparse 行为更多由 PPO/CTDE credit 学习，而不是由手写 imitation 或辅助目标拉动。

原因：v8/v9 的核心缺口不是 learned-baseline reward，而是相对 strong heuristic 的 idle/sparse 小亏，以及相对 PPO 的 failure/backhaul trade-off。继续只加 hard guard 容易把贡献写成规则系统；MAPPO 的 controller-level credit assignment 正好能解决三控制头共享错误 credit 和 action-mix collapse 风险，使 cache、execution 和 event head 从它们实际控制的动作中学习。

影响：v10 不修改 `VecWorkflowCoreEnv` reward、不修改 `semantic_discrete_5` action contract、不改变 baseline contract，也不替换 v8 canonical。v10 只能在 dev 或新冻结 future-validation split 上筛选；当前 hidden 已 consumed，仍不得用于候选选择或调参。晋级前必须按 window class 报告机制窗口收益、idle/sparse gap、PPO failure/backhaul non-inferiority 和 learned-baseline reward/continuity。

## 2026-07-13: v9 采用 Pareto-safe checkpoint ranking，不改 reward/action contract

决策：新增 `top_journal_mechanism_v9_pareto_safe` profile，把 handoff failure 与 backhaul trade-off 纳入 checkpoint selection / gate 的约束目标；实现上只调整 policy-side guard 参数、prefetch admission、steady RSU bias 和 checkpoint ranking，不修改 `semantic_discrete_5` action contract、环境 reward 或 baseline observation contract。

原因：v8 已修复 strict-full reward/continuity blocker，但 formal/hidden 对 PPO 的 handoff failure 和 backhaul trade-off 仍是顶刊 blocker。直接改 reward 或动作空间会破坏 v8/v9 可比性，也会重开 baseline 公平性和 checkpoint provenance 风险；因此先把 v9 限定为 dev/future-validation 上的 Pareto-safe candidate。

影响：v9 只有在 dev 与新 future-validation 上同时满足 reward、DT continuity、handoff failure 和 backhaul non-inferiority 后，才能进入重新审查；若 reward 明显下降，则归档为 safety trade-off candidate，不替换 v8。当前 hidden 已 consumed，不得用于 v9 筛选。

## 2026-07-06: predictor 升级为薄 supervised handoff anticipation 层

决策：新增 `supervised_handoff_predictor_v1`，把 prediction/DT 口径限定为短时 next-RSU、handoff target 和 ETA anticipation。该层通过显式 checkpoint 接入 `PredictorManager`，服务于 SA-GHMAPPO 的 cache / prefetch / migration prepare 控制；不改变 `semantic_discrete_5` action contract，不把 predictor 提升为主算法贡献。

原因：仅靠 baseline/calibrated predictor 难以支撑 TMC 级 predictive / DT claim；但完整 digital twin 子系统会稀释主问题并扩大实现风险。薄 supervised predictor 能提供可训练、可审计、可冻结的预测证据，同时保持主线问题聚焦在 reliability-gated continuous workflow control。

影响：正式论文只能写 `supervised short-horizon handoff predictor` 和 `lightweight DT-style predictive state snapshot`；不得写完整 digital twin、轨迹预测 SOTA 或 predictor 单独解决连续 cache。v9 结果必须使用冻结 predictor checkpoint 重训并在 formal/future-validation 上重新验证。

## 2026-06-21: strict-full v8 采用冻结 split、一次性 hidden 与层级统计

决策：顶刊主结果默认使用 outcome-blind 固定 window plan；时间窗口是外层独立抽样单元，seed/workflow 只在窗口内重采样。候选最多进行两轮 dev 筛选，冻结后运行 formal；formal 通过后 hidden 只开启一次，之后永久作为 consumed holdout。

原因：v7 的 rank-offset holdout 与 formal 重叠，且把 seed/workflow 行当独立 cluster 会高估有效样本量。固定 20-window/split、minimum gap 24 frames 与 hierarchical BCa/Holm 能把 selection、temporal overlap 和 pseudo-replication 风险拆开。

影响：`window_rank_offset` 只保留为 sensitivity 工具；正式 independent split 必须传 `--window_plan_path`。hidden 开启事件写入独立 execution record，不回写冻结 manifest。当前 hidden 已 consumed，任何后续算法修复必须使用 dev 或新冻结 future validation split。

## 2026-06-21: v8 用 steady-RSU soft bias 替换 v7 latency fallback

决策：v8 关闭会在 idle/no-handoff 状态偏向 vehicle fallback 的 latency fallback，改为仅在 current adapter warm 且没有 distinct handoff target 时对 current-RSU execution 施加 soft bias。该机制不修改 reward、action contract、环境或 baseline 输入。

原因：v7 failure diagnosis 显示 service wait/miss 与 continuity/failure 高度相关，而 latency fallback 在 steady window 中增加 vehicle execution，造成 strict-full continuity 退化。v8 需要保留跨 RSU mechanism 行为，同时避免在无 handoff 状态破坏已 warm 的本地服务。

影响：v8 修复了相对 DT 的 strict-full reward/continuity blocker，但引入相对 PPO 的 handoff failure 和 backhaul trade-off。该 trade-off 必须作为后续约束优化目标，不能通过再次查看现有 hidden 调参。

## 2026-05-28: v7 final-submission package 升级为当前 canonical

决策：`final_submission_v7_latency_fallback_20260528_v1` 升级为当前 paper-ready canonical final-submission package。旧 `final_submission_full_current_baselines_20260511_v1` 和 `final_submission_controller_mappo_qmix_20260509_v1` 降为历史 package；后续论文主表优先引用 v7 final-submission 的 `comparison_report/paper_ready/`。

原因：

- v7 final gate `target_reached=true`、`paper_claim_ready=true`、`blockers=[]`。
- formal 与 offset=3 holdout learned gates 均通过 contract、duplicate-trace independence、cluster-bootstrap reward CI；learned baselines 在 final suite 内 clean retrain，`formal_training_provenance.passed=true`、`record_count=27`。
- v7 package 覆盖当前 paper-grade learned set：PPO、controller-level MAPPO、DQN、Dueling-DQN、controller-level QMIX、Controller-MAT、DAG-Offload-DRL、Cache-Offload-DRL、DT-Handoff-DRL。
- comparison package `review_ready=true`、`paper_ready_package_ready=true`，作者自审无 blocker。

影响：

- `README.md`、`docs/project/PROGRESS.md`、`docs/project/ARTIFACT_RECORDS.md`、`docs/project/current_results_audit_20260527.md`、`docs/project/DIRECTORY_STRUCTURE.md` 和 `docs/project/RUNBOOK.md` 需要把当前 canonical 指向 v7。
- 论文主结果可写“SA-GHMAPPO ranks first among clean-retrained primary learned baselines in all formal and offset-3 holdout splits”，但必须保留 generated self-review 中的 3 个 limitation。
- `popularity_cache_heuristic` 继续作为 close supplementary reference，而非 primary learned-baseline blocker；不得写成大幅超过手写 heuristic。

## 2026-05-28: v7 clean-retrain 启用 latency fallback 快时标控制

决策：新增主方法 profile `top_journal_mechanism_v7_latency_fallback`，以 `top_journal_mechanism_v6_strong_competition` 为基线，保留 freshness / confidence-aware prefetch admission guards，并重新启用 `latency_fallback_bias_enabled=true`、`latency_fallback_bias_strength=1.20`、`latency_fallback_confidence_floor=0.62`、`latency_fallback_slow_suppression_strength=1.20`。该 profile 必须作为 clean retrain 候选运行，不复用旧 v3 eval-bias 结论。

原因：

- `top_journal_mechanism_v6_prefetch_admission_20260528_v1` 已消除相对 `popularity_cache_heuristic` 的负收益，但 mixed/full 仍与 heuristic 严格打平，无法通过要求 total reward 严格 win 的 gate。
- formal trace 里低风险快时标机会主要出现在 current adapter 已 warm、无 handoff pressure 的 steady execution 步；将这类步切换为 `vehicle_fallback` 可以降低 delay penalty，同时不改变 cache/backhaul/migration/handoff contract。
- 旧 `top_journal_mechanism_v3_eval_bias` 已证明 latency fallback 有潜力，但它是 inference calibration artifact；v7 的目的就是把该机制纳入独立 profile 和 clean-retrain 证据链。

影响：

- 不修改 `VecWorkflowCoreEnv` reward、不修改 `semantic_discrete_5` action schema、不修改 learned/domain baseline contract。
- `scripts/train_sa_ghmappo_real_sample.py`、`scripts/run_top_journal_closed_loop.py`、`configs/experiment/top_journal_mechanism_v7_latency_fallback.yaml` 和 profile contract tests 需要同步维护 v7 参数。
- `top_journal_mechanism_v7_latency_fallback_20260528_v1` closed-loop formal 已通过，但仍不是 final-submission canonical；替换 canonical 前必须继续跑 final-submission/holdout/support package。

## 2026-05-27: SA predictive prefetch 增加 confidence/alignment admission guard

决策：`top_journal_mechanism_v6_strong_competition` 增加 `predictive_prefetch_admission_guard_enabled=true`，并采用 `predictive_prefetch_admission_min_confidence=0.55`、`predictive_prefetch_admission_require_distinct_next=true`。当 selected action 为 predictive prefetch，但 `predicted_next_rsu_id` 仍未离开当前 RSU、prefetch target 仅来自后续序列或 handoff target，且预测置信度低于阈值时，将 prefetch 延期为 `handoff_migration_prepare`。

原因：

- `top_journal_mechanism_v6_freshness_guard_20260527_v1` 仍未通过 gate，剩余 gap 与 freshness 上界无关，而是低置信度/next-RSU 未对齐时的过早 prefetch realization gap。
- 负例 `window_off246_len24_t293_316` / `j_8` / seed `13` 中，SA 在 `prediction_confidence=0.383` 且 `predicted_next_rsu_id` 为当前 RSU 时 prefetch，最终 `expired_miss`；popularity heuristic 等到后续 `prediction_confidence=0.611` 且 next-RSU 已对齐时 prefetch 并命中。
- 该策略对应 VEC service caching / handoff migration 中常见的 confidence-aware admission control：在预测未稳定前优先准备迁移，不提前占用 prefetch freshness window。

影响：

- 默认关闭，历史 checkpoint 和旧 profile 行为不变；v6 profile 显式开启。
- 训练 summary、checkpoint 恢复、benchmark rows 和 eval-bias manifest builder 必须保留该字段，否则复现会丢失 admission 行为。
- 该决策只修正 policy-side admission，不改变 action schema、reward 或 formal gate；正式效果仍需重新跑 3-seed closed-loop / final-submission gate。

## 2026-05-27: SA cache-warm guard 增加 freshness window 上界

决策：`top_journal_mechanism_v6_strong_competition` 的 cache-warm start guard 增加 `cache_warm_start_guard_max_prefetch_countdown=6.0`。当 target adapter 未 warm 但预测 handoff countdown 超出该上界时，guard 不再把 event prepare 强制替换为 predictive prefetch，而是等待进入 freshness window 后再触发 prefetch。

原因：

- v6 masked full-train 诊断显示剩余 popularity gap 集中在单个 mechanism window：SA 过早 prefetch 后 validation expired，而 heuristic 在更接近 handoff 的 step prefetch 并命中。
- 当前 recorder 的 `prefetch_validation_window=6` 已定义 prefetch 的有效兑现窗口；policy-side guard 应与该 freshness contract 对齐，避免奖励/环境不变时制造过早机制动作。
- 该设计贴合 VEC service caching / service migration 文献中的 deadline-aware、freshness-aware placement 思路，但不引入新动作空间或 reward 改写。

影响：

- 历史 profile 和 checkpoint 默认上界为 `0.0`，语义为禁用该上界；只有 v6 profile 显式启用。
- 训练、checkpoint 恢复、eval-bias manifest 和配置文件需要保留该字段，否则 benchmark 复现会丢失 guard 行为。
- 本决策只解决 timing guard，不代表 v6 已 paper-ready；仍需通过 formal/holdout gate 后才能替换 canonical。

## 2026-05-27: MAPPO 强对照升级为 controller head-credit v3

决策：`mappo` 继续保持 controller-level CTDE baseline，但 paper-grade 强对照协议升级为 `aggregation_reason_weighted_controller_ppo_v3`，并新增 `mappo_strong_audit` 训练 profile。主算法新增 `top_journal_mechanism_v6_strong_competition` profile，用于后续与优化后的 learned baselines 同预算重跑。

原因：

- 2026-05-11 canonical 审计显示 MAPPO 虽然 protocol-valid，但存在 action-mix collapse，不能作为主优势证据。
- 若论文把 MAPPO 写入主表，必须给 MAPPO 合理的 controller-head credit floor、entropy floor/scale 和更稳健的 PPO 更新配置，避免弱化对照。
- 这些增强只属于 MAPPO 的通用 controller credit assignment，不引入 SA-GHMAPPO 的 graph/surrogate/guard/auxiliary 专属机制。

影响：

- `scripts/run_top_journal_learned_baseline_suite.py`、`scripts/run_top_journal_final_submission_loop.py`、`scripts/run_top_journal_closed_loop.py` 和 `scripts/build_top_journal_comparison_report.py` 的 MAPPO protocol audit 均要求 v3 字段。
- 旧 pre-v3 MAPPO checkpoint 只能作为历史 artifact；新的 MAPPO 论文 claim 必须来自 v3 final-submission package。
- 本次只是协议和训练入口更新；在 v6 + MAPPO v3 正式 final-submission gate 通过前，不替换 `final_submission_full_current_baselines_20260511_v1` 的已验证结果。

## 2026-05-12: SA-GHMAPPO 先修补 semantic_discrete_5 闭环，不切换 parameterized DAG action

决策：近期主算法保持 `semantic_discrete_5`，先补强 predictive action mask、invalid reason、predictor audit、mechanism success gate、DAG diagnostics 和 guard/projection delta；不在本轮引入 `target_node` / `target_rsu` parameterized action，也不改成 full vehicle/RSU multi-agent wrapper。

原因：

- 当前训练、评估、benchmark 和历史 checkpoint 都围绕五类 semantic discrete action；直接切换 parameterized action 会破坏现有主表消费端和 checkpoint contract。
- 当前最紧迫的审稿风险是非法 predictive action 静默 fallback、机制尝试与机制成功混淆、预测层 claim 边界不清，而不是动作维度不足本身。
- DAG graph encoder 已能提供结构信息；在冻结 DAG-level action 前，只能把 frontier / dependency / critical-path pressure 作为诊断和 ablation 证据。

影响：

- `ActionMaskBuilder.build_mask_info()` 成为 wrapper 和 policy 诊断的正式 precondition 来源。
- `ControlAction.metadata.invalid_action/invalid_reason`、raw/projected/final action、guard delta 和 mechanism success gate 进入新产物字段。
- 论文当前只能写 controller-level hierarchical multi-controller PPO / MAPPO-style 主方法；不能写 full vehicle/RSU multi-agent wrapper 或 learned surrogate predictor。

## 2026-05-10: MAPPO 对照必须启用 controller head-credit

决策：`mappo` 保持 controller-level CTDE baseline，但当前 paper-grade 协议必须启用 aggregation-reason controller head-credit，并由 learned suite、final loop 和 comparison report 审计 checkpoint config。

原因：

- 旧 MAPPO 三控制头在 PPO loss 中共享相同 credit，容易把 slow / fast / event head 的责任混淆。
- 主算法属于 MAPPO 路线增强；若弱化 MAPPO baseline，会被审稿人质疑对照不公平。
- head-credit 是当前 multi-controller action aggregation contract 下的通用 credit assignment，不引入 SA-GHMAPPO 的 graph/surrogate/guard 专属机制。

影响：

- `final_submission_controller_mappo_qmix_20260509_v1` 中的 MAPPO 结果降级为 pre-head-credit 历史归档。
- 新论文主表必须重跑 final-submission loop，并要求 `baseline_protocol_versions.mappo` 与新版协议一致。

## 2026-04-27: HF model-cache 不能直接进入正式 benchmark

决策：HF `model-cache` 候选全集先进入审计 manifest、统一数据源声明和 catalog metadata；正式 benchmark 不直接消费这些数据，除非先实现显式 file-size importer、adapter_id 映射、结果标签和 claim 边界。

原因：

- 审计到的 HF 候选主要是模型文件、embedding 文件、分块模型文件或图像生成 cache-like WebDataset，不是 VEC cache request/event trace。
- 现有候选不提供 cache hit/miss、RSU locality、handoff demand 或 adapter state migration trace。
- 直接把 HF 文件大小投影到 `cache_objects` 会改变 reward/backhaul 成本，必须作为单独 profile 输出，不能覆盖 `NGSIM + Alibaba` 主线结果。

影响：

- `data/raw/model_cache/huggingface_model_cache_sources.json` 升级为候选全集审计 manifest。
- `configs/data/hf_model_cache_integration_plan.json` 记录后续 importer 所需 contract 和 guardrail。
- 报告只能声明“HF 真实模型文件大小/缓存体量 profile”，不能声明“HF 真实 cache request trace”，除非后续找到并验证相应数据。

## 2026-04-27: HF model-cache 数据源先按 metadata-only 接入

决策：接入 Hugging Face dataset `ClemSummer/qwen-model-cache` 时，先只作为外部真实 model-cache metadata source 写入 `AdapterCatalog.model_cache_datasets`、`configs/data/dataset_sources.json` 和报告，不自动下载模型文件，也不改变 benchmark cache 行为。

原因：

- 当前正式结论仍绑定 `NGSIM + Alibaba`，不能把外部 HF 文件突然混入正式 benchmark。
- 现有 cache reward/telemetry 消费的是本地 `sample_model_catalog.json` 中的 adapter/cache object 语义；文件级 model-cache importer 尚未冻结字段 contract。
- metadata-only 接入可以先解决数据源可追踪性和报告声明缺口，同时避免不可控下载和历史结果不可复现。

影响：

- 后续若要用真实 HF model-cache 文件大小、文件层级或访问热度，需要新增 importer、schema 和 benchmark 消费端检查。
- 所有报告应优先引用 `docs/project/DATASET_SOURCES.md` 作为数据源声明入口。

## 2026-04-22: 项目化 AI 维护文档入口

决策：将通用模板落到根目录 `AGENTS.md` 和 `docs/project/`，作为 PPO_MEC 的 live 维护入口。

原因：

- 根目录 `AGENTS.md` 更容易被 AI coding 工具自动发现。
- `docs/project/` 可以承载结构、模块、运行手册和长期决策，避免 README 继续膨胀。
- 通用模板目录已整理吸收，不再保留为 live 文档入口。

影响：

- 新任务应先读 `AGENTS.md` 和 `docs/project/README.md`。
- 影响路径、入口、协议、产物和长期结论的改动，需要同步更新对应维护文档。

## 2026-04-22: live 模型层只保留主方法

决策：当前 live 模型层只保留 `sa_ghmappo`，删除对比算法实现、注册项和 baseline 专用训练评估入口。

原因：

- 当前任务要求除本实验主方法外删除对比算法。
- 后续 live 训练、评估和 benchmark 统一围绕主方法 checkpoint。

影响：

- `src/agents/registry.py` 仅注册 `sa_ghmappo`。
- 新 benchmark 输出不再生成对比算法 pairwise / win-tie-loss 结论。
- 已删除算法的专属 artifact 目录已清理；历史混合 aggregate 只作为归档快照，不代表当前可运行模型层。

## 既有长期决策摘要

- 当前正式真实数据主线是 `NGSIM + Alibaba`。
- `LuST` 保留 provider 和导出脚本，但当前不阻塞主线。
- `highD` 保留为后补数据源和 provider 骨架。
- `smoke_run` 只用于联调，不用于正式论文结论。
- 正式主表优先使用 `scripts/benchmark_main_results.py` 的多窗口、多 seed 输出。

## 2026-04-23: 建立方向匹配型对照算法池

决策：在保留主方法 `sa_ghmappo` 的同时，恢复对照算法池，但只接入与研究方向匹配的强化学习算法。

原因：

- 当前论文需要公平对比：Flat PPO / Flat MAPPO 提供同家族弱化对照。
- TD3 / SAC / MADDPG 只有在自然连续动作 contract 存在时才合理；当前不应强行改写环境动作定义。
- QMIX 需要稳定 multi-agent discrete wrapper，本轮只预留接口。

影响：

- 该决策已被 2026-04-24 的 agent 目录收敛决策部分替代：当前 live registry 使用 `ippo` / `ppo` / `mappo` 主名，`flat_ppo` / `flat_mappo` 仅保留为历史 artifact run 名称。
- TD3 / SAC / MADDPG / QMIX 不再注册 skeleton；后续接入前必须先冻结匹配的 observation/action contract。该 QMIX 阻塞口径已被 2026-05-09 的 controller-level QMIX 决策取代。

## 2026-04-23: 建立 baseline 训练与评测闭环

决策：在不改核心环境语义的前提下，新增 `reactive_greedy` 和 `popularity_cache_heuristic` 非学习对照，并用 `scripts/run_baseline_experiment.py` 统一调度训练、评估、benchmark 和 `comparison_summary` 导出。

原因：

- 当前用户目标是先开始训练并跑公平对照，不是重构训练框架。
- Reactive / popularity heuristic 可以通过既有 `semantic_discrete_5` 动作 contract 接入，不需要改 reward、handoff 或 adapter cache 语义。
- TD3 / QMIX 当前动作与 wrapper 条件不足，保留骨架比硬写不可靠实现更可解释。

影响：

- `ALGO_REGISTRY` 增加 `reactive_greedy` 和 `popularity_cache_heuristic`。
- `benchmark_main_results.py` 以及 robustness / scalability benchmark 的 agent choices 扩展为可评估 agent。
- `configs/experiment/baseline/` 成为 baseline 闭环配置入口。

## 2026-04-24: Round1 实验管理产物改为 agent × seed 粒度

决策：`run_baseline_experiment.py` 的 round1 summary 和 manifest 以 `agent × seed` 为主粒度，同时在 detailed JSON 中保留 agent-level 汇总。

原因：

- formal experiment execution 需要直接追踪每个 seed 的 train、checkpoint、eval 和 benchmark 输入。
- 后续扩展更多 seed、窗口或训练预算时，单纯 agent-level summary 不足以定位缺失 checkpoint 或 seed-specific 失败。

影响：

- `comparison_summary.csv/json`、`comparison_summary_by_window_class.csv` 和 `run_manifest.json` 均包含 `agent`、`seed`、`checkpoint_path`、`train_summary_path`、`eval_summary_path`、`benchmark_aggregate_path`、`support_level` 和 `status`。
- 旧 run 仍按原 schema 保留，新 round1 run 从 `baseline_minimal_ngsim_alibaba_20260424_145836` 起使用增强 schema。

## 2026-04-24: agent 目录收敛为按算法分文件

决策：`src/agents/` 不再按 `baselines/` 和 `marl/` 分层，也不再使用 `ppo_family.py` 同时承载 PPO 与 MAPPO。主方法、PPO、MAPPO 和 heuristic baseline 都在根目录独立算法文件中实现，`registry.py` 直接导入算法文件。

原因：

- 当前用户要求 agent 结构以算法为唯一组织维度，降低 AI 协作时误用 family 包装层的风险。
- PPO 与 MAPPO 需要作为两个独立基础算法维护，不能共用一个 family 导出层。
- TD3 / SAC / MADDPG / QMIX 当时没有匹配的 live contract，继续注册 skeleton 会干扰正式算法池。该 QMIX 阻塞口径已被 2026-05-09 的 controller-level QMIX 决策取代。

影响：

- live registry 只保留真实可用 agent 名：`sa_ghmappo`、`ippo`、`ppo`、`mappo`、`reactive_greedy`、`popularity_cache_heuristic`；`flat_ppo` 和 `flat_mappo` 仅作为历史 artifact run 名称保留。
- `sa_ghmappo_core.py` 作为共享 on-policy 核心保留；它不是算法分类目录，也不再承载 PPO/MAPPO 注册。
- `configs/algo/` 只保留当前可运行算法配置和 heuristic 配置。


## 2026-05-04: 顶刊路线算法闭环第一阶段

决策：保留当前单 wrapper 决策流，不做一次性 multi-agent wrapper 大重构；先修复两个影响审稿可信度的 live contract：MAPPO 不再与 PPO 共享完全相同的 flat critic 输入，改为 flat actor + centralized global semantic critic context；所有学习策略在 flat action 分布上应用 wrapper 提供的 `action_mask`。

原因：

- 旧 MAPPO 只设置 `centralized_critic=True`，但 flat 分支没有消费该标志，导致 PPO/MAPPO 结果可完全相同。
- 当前 action mask 虽然多数场景为 all-true，但 learned policy 不消费 mask 会让后续更细合法动作约束无法审查。
- 完整 multi-agent CTDE 仍需要稳定 multi-agent wrapper；在该 contract 冻结前，先提供可训练、可评估、可复现实验的 centralized-critic baseline。

影响：

- `FlatSemanticEncoder` 输出 `centralized_critic_context`，MAPPO flat value head 使用该 context，PPO/IPPO 继续使用 independent `critic_context`。
- `sa_ghmappo_core.py` 在 flat policy 采样和 PPO update 中应用 `action_mask`，并在 `action_info` 记录 `action_mask_applied`、`valid_action_count` 和 critic context key。
- 新增 `top_journal_mechanism_v1` 训练 profile 与配置文件，用于机制辅助损失、慢衰减 imitation、机制窗口重采样和后期 retention 的正式复跑候选。
- 旧 MAPPO checkpoint 可以继续被加载用于 actor-only evaluation，但旧结果不能代表新 centralized critic training contract；正式论文表需要重训 PPO/MAPPO 后再 benchmark。

## 2026-05-05: 机制辅助与 reactive cache fill 解耦

决策：

- `top_journal_mechanism_v1` 的机制辅助目标只用于 predictive prefetch / prepare 等机制动作，不再把普通 `current_rsu_cache_fill` 作为机制目标强推。
- 新增 backhaul guard：无 handoff/prefetch 预测信号时，同一 adapter 每 episode 的 reactive cache fill 有预算上限，超出后投影到 `no_cache_change`。

原因：

- quick gate 显示旧策略 reward 较高但 backhaul 明显高于 popularity heuristic，主要来自重复 reactive cache fill。
- 顶刊审查更容易接受“机制动作带来收益且 backhaul 不劣化”的证据，不应靠泛化的 cache fill bias 提升 reward。

边界：

- 该 guard 是 SA-GHMAPPO 策略内部的资源成本约束，不修改环境语义、reward、handoff、migration 或 benchmark 指标。
- quick 通过不等于 paper claim ready；正式结论仍需非 quick 多 seed artifact。

## 2026-05-05: 顶刊主线使用 handoff pressure 主体绑定

决策：

- 顶刊闭环默认使用 `primary_vehicle_selection=handoff_pressure`，让 `max_handoff_candidate` 选出的 NGSIM handoff 窗口绑定到 workflow 主 vehicle。
- 单独训练、评估和 dry-run 入口保留 `stable_first` 默认值，用于历史兼容；正式顶刊路线必须显式或通过闭环默认使用 `handoff_pressure`。
- `paper_claim_ready` 不再只由非 quick gate 决定，必须同时满足 `formal_contract.ready=true`。

原因：

- 原协议只保证窗口内存在 handoff vehicle，不保证 workflow 主体就是该 vehicle，导致机制训练信号与 benchmark handoff 压力错位。
- 顶刊审查需要明确区分 quick/smoke、低预算非 quick 和正式多 seed artifact，避免产物被误写成论文结论。

边界：

- 该变更只影响 primary vehicle 选择协议，不修改 mobility trace、handoff 语义、reward 或 benchmark 指标。
- quick-plus v6 只证明闭环可用，不能替代正式主表实验。

## 2026-05-05: 顶刊主 claim 采用 learned-baseline strict gate

决策：

- `reactive_greedy` 和 `popularity_cache_heuristic` 保留为 supplementary heuristic reference，不再作为顶刊主 claim 的通过条件。
- 主 learned-baseline gate 面向 `ippo`、`ppo`、`mappo`、`dqn` 和 `ddqn`，并保留 heuristic 对照行用于解释工程边界。

原因：

- `popularity_cache_heuristic` 是手写规则，不能替代文献级 learned baseline 作为主对照。
- 当前 `semantic_discrete_5` 动作 contract 可公平承载 PPO-family 与 DQN/DDQN，但不能直接承载 TD3/SAC/MADDPG 的连续动作假设。

边界：

- heuristic reference 仍应报告，尤其用于检查主方法是否只是复现手写规则。
- 连续控制 baseline 需要先冻结新的 action/observation contract，不能为了凑 baseline 强行接入。

## 2026-05-05: Mechanism v3 eval-bias 只作为候选增强

决策：

- 在 `sa_ghmappo` 中保留低风险 latency fallback inference calibration，但 clean retrain `top_journal_mechanism_v3` 未冻结为主结果。
- `top_journal_mechanism_v3_eval_bias` 作为候选增强 artifact 记录；正式 paper-grade 主表仍优先引用 formal_v2 与 learned-baseline strict gate。

原因：

- derived eval-bias manifest 在 formal_v2 权重上显著扩大了相对 heuristic reference 的 paired reward delta，且不牺牲 continuity / handoff / backhaul。
- 但 clean retrain v3 相对 heuristic 未形成稳定优势，说明该机制当前更像 inference calibration，不是已收敛的新训练 profile。

边界：

- `top_journal_mechanism_v3_eval_bias` 在补齐独立 holdout/support suite 前不能替代 formal_v2 paper-grade 主表。
- 讨论 heuristic reference 时必须同时报告 failed prepare / migration 诊断，避免把 reward shaping 中的机制探索奖励误读为真实迁移收益。

## 2026-05-06: v3 候选升级为强候选但不推广 v4

决策：

- 接受 `top_journal_mechanism_v3_eval_bias_guarded_prefetch_*` 作为当前最强候选结果：formal gate、offset=3 holdout、latency fallback ablation、robustness 和 scalability 均有 artifact 支撑。
- 不推广 `top_journal_mechanism_v4_prepare_eval_bias`；predictive prepare hard override 在 prediction robustness 的 oracle setting 上失败。
- `--window_rank_offset` 作为 holdout split 参数保留在主表和消融 benchmark 中，默认 0，不改变 frozen formal protocol。

原因：

- v3 在 learned baselines 和 supplementary popularity heuristic 上都有 paired 正向优势，且 holdout 不依赖 formal windows。
- v3 的 prediction robustness 仍存在 oracle heuristic upper-bound 边界，因此不能写成“所有预测条件全面领先”。
- v4 没有修复该边界，继续投入会偏离当前核心主表/holdout/消融闭环。

边界：

- v3 仍是 formal_v2 权重上的 inference calibration，不是 clean retrain 收敛结论。
- 论文可写 v3 的 latency fallback 机制贡献，但必须同时披露 oracle prediction robustness 边界。

## 2026-05-06: learned-baseline gate 补充 Dueling-DQN 系列

决策：

- 在 learned-baseline strict gate 中补充 `dueling_dqn` 和 `dueling_ddqn`，与 `ippo`、`ppo`、`mappo`、`dqn`、`ddqn` 一起作为主 learned baseline 集合。
- `reactive_greedy` 和 `popularity_cache_heuristic` 继续保留为 supplementary heuristic reference，不作为主 claim 的 gate 条件。

原因：

- 近期 VEC offloading / service caching 论文常使用 DQN-family 变体作 learned baseline；补充 dueling value/advantage 结构可以减少“只和 PPO 比”的审稿风险。
- 新 baseline 仍使用当前 `semantic_discrete_5` contract，不需要强行引入 TD3/SAC/MADDPG 这类连续控制 baseline。

边界：

- 不人为削弱 heuristic；论文中只是把 heuristic 从主 gate 降级为附表/参考线。
- 历史 learned-baseline artifact 未覆盖 `dueling_dqn` / `dueling_ddqn`，引用时应优先使用 plus-dueling gate 或显式说明旧 baseline set。

## 2026-05-06: 最终交稿 gate 采用 learned-primary 闭环

决策：

- 最终交稿主 gate 当时只比较 learned baselines：`ippo`、`ppo`、`mappo`、`dqn`、`ddqn`、`dueling_dqn`、`dueling_ddqn`。该 learned set 已被 2026-05-07 的 baseline independence 决策取代，不能继续作为 paper-grade 默认集合。
- `reactive_greedy` 和 `popularity_cache_heuristic` 保留为 supplementary heuristic reference，不作为主 claim 的阻塞条件。
- prediction support 的 setting-level dominance 只要求 `learned_prediction` 与 `noisy_prediction`；`no_prediction` 和 `oracle_prediction` 作为诊断设置保留。

原因：

- `popularity_cache_heuristic` 是项目内手写规则，适合作为工程参考线和 sanity check，但不应决定顶刊主 claim 是否成立。
- `no_prediction` 是预测机制下界，`oracle_prediction` 是诊断上界；二者不对应“实际 learned predictor 条件下的机制贡献”主声明。
- 主表需要一个可复跑、可断点续跑、可审计的统一 final loop，而不是依赖手工拼接 formal、holdout 和 support 产物。

影响：

- 新增 `scripts/run_top_journal_final_submission_loop.py`，正式产物写入 `artifacts/experiments/top_journal_final_submission/<run_id>/`。
- final loop 支持 `--resume_training`、`--resume_benchmark`、`--resume_support` 和 `--command_retries`，用于长实验自循环。
- `final_submission_clean_equal_budget_20260506_v1` 后续被 duplicate trace audit 重新判定为 `target_reached=false` / `paper_claim_ready=false`。

边界：

- heuristic reference 仍应在附表报告，尤其用于说明主方法不是简单复现手写 popularity 规则。
- 不能把 prediction support 写成所有预测设置全面领先；`no_prediction` 和 `oracle_prediction` 的数值必须作为诊断边界披露。

## 2026-05-06: learned-baseline independence audit 成为 final gate 硬约束

决策：`scripts/run_top_journal_learned_baseline_suite.py` 的 final gate 必须审计 learned baselines 的 benchmark trace 是否完全重复。若两个 learned baseline 在相同 `source_rows_path/window_id/scenario_id/mode/workflow_id/seed` 下除 agent/checkpoint 字段外全行一致，则该 suite 不再 `paper_claim_ready`。

原因：

- 当前 `ippo` 与 `ppo` 都是 `PPOBaseAgent` 的 flat single-wrapper 配置，除 `agent_name` / `policy_type` 外没有独立训练语义，因此同 seed 下 checkpoint 可 bitwise identical。
- 当前 `mappo` 只在 flat branch 中改变 critic context；actor 仍消费同一个 `shared_embedding` 并在 deterministic benchmark 中取 argmax，因此可以与 PPO/IPPO 产生完全相同动作轨迹和指标。
- DQN/DDQN 与 Dueling-DQN/Dueling-DDQN 也在最新 final run 中出现完全重复 trace，说明独立性问题不只限于 PPO-family。

影响：

- `final_submission_clean_equal_budget_20260506_v1` 被重新判定为 `target_reached=false` / `paper_claim_ready=false`。
- 后续 paper-grade 主表必须先修复真实 multi-agent/CTDE contract，或将非独立 baseline 降级为同构实现说明而非独立对照，再重跑 final loop。

## 2026-05-07: IPPO/MAPPO 降级为 diagnostic，paper-grade 默认 baseline 去重

决策：当前 single-wrapper decision stream 下，`ippo` 和 `mappo` 不再是 paper-grade learned baseline。live registry 保留它们为 `diagnostic`，用于历史 artifact 复核和 contract test；`scripts/run_top_journal_learned_baseline_suite.py` 与 `scripts/run_top_journal_final_submission_loop.py` 的默认主 learned set 改为 `ppo`、`dqn`、`dueling_dqn`。

原因：

- 当前 wrapper 没有 per-agent observation/action surface，不能真实实现独立 IPPO。
- 当前 MAPPO 只有 centralized critic context，actor 仍是 single-stream flat actor；它不是 full CTDE multi-agent MAPPO。
- `ddqn` 与 `dueling_ddqn` 在最新 final run 中分别与 `dqn` / `dueling_dqn` 完全重复，不能默认作为独立证据。

影响：

- 显式运行 `ippo` / `mappo` 必须使用 `--allow_contract_blocked_baselines`，并且该 suite 仍不能 `paper_claim_ready=true`。
- 后续可交稿主表必须使用去重后的 learned baseline set，且 duplicate trace audit 必须通过。
- `final_submission_repaired_baselines_20260507_v1` 按修复后的默认 set 复跑并通过 final gate，但仍复用旧 learned checkpoint；随后被 clean retrain run 取代。
- `final_submission_clean_retrain_repaired_baselines_20260507_v1` 新增 formal training provenance gate 并 clean retrain `ppo/dqn/dueling_dqn` 三 seed，成为当前 canonical final-submission run。

## 2026-05-07: final comparison package 成为顶刊审稿出口

决策：新增 `scripts/build_top_journal_comparison_report.py`，将 canonical final-submission gate、learned suite statistics 和 support rows 汇总为 `comparison_report/`。后续对外主对比不再手工从多个 JSON/CSV 拼接，而是优先引用该 comparison package。

原因：

- 顶刊审稿会同时检查 baseline 口径、训练来源、重复 trace、主表 reward、机制分解、holdout 和 support robustness；这些材料必须在同一 artifact 中可追踪。
- `popularity_cache_heuristic` 与主方法机制指标接近，必须在 protocol matrix 中明确其 supplementary reference 身份，避免被误写成 learned gate 对手。
- prediction support 的 `learned_prediction` / `noisy_prediction` 需要 setting-level CI，不能只给 aggregate 数值。

影响：

- 当时 canonical report 为 `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/comparison_report/top_journal_comparison_report.json`，其中 `review_ready=true`；当前已被 `final_submission_controller_mappo_qmix_20260509_v1/comparison_report/` 取代。
- `comparison_report/paper_ready/` 进一步生成论文可直接引用的 LaTeX/CSV 表格、copy-ready result statement 和 self-review，当前 `paper_ready_package_ready=true`。
- `comparison_report/baseline_protocol_matrix.csv` 记录 `ippo/mappo` 为 diagnostic/contract-blocked，`reactive_greedy` 和 `popularity_cache_heuristic` 为 supplementary reference。
- `comparison_report/support_paired_statistics.csv` 同时保留 support aggregate 和 prediction setting-level paired CI。
- self-review 必须随主表一起查看；当前无 blocker，但限制项包括 popularity heuristic 接近、no_prediction/oracle 诊断设置不支持全面预测优势、mechanism_realization_rate 不构成独立正向优势、holdout backhaul 对 PPO 不具备正 CI。

## 2026-05-09: MAPPO/QMIX 恢复为 controller-level paper-grade baselines

决策：`mappo` 不再按 2026-05-07 的 diagnostic/contract-blocked 口径处理；当前实现恢复为可训练、可评估、可进入 learned-baseline gate 的 controller-level CTDE MAPPO。新增 `qmix` 作为 controller-level value-decomposition learned baseline，同样可训练、可评估、可进入 learned-baseline gate。`ippo` 仍保持 contract-blocked diagnostic。

原因：

- 用户指出只保留 PPO/DQN-family 会形成“只和 PPO 比”的审稿风险，且旧 IPPO/PPO/MAPPO 完全相同确实不可接受。
- 当前环境尚无 vehicle-agent / RSU-agent per-agent wrapper，但主方法本身是 cache、execution/offload、handoff-event 多控制器结构；以这些 controller 作为 learned baseline 的 agent 粒度，能在现有 `semantic_discrete_5` contract 下形成可审查的 CTDE/value-decomposition baseline。
- 旧 MAPPO 问题不只是文档状态：层级分支 centralized critic 曾回落到普通 `critic_context`，event head 也未进入动作聚合；这些已修复为 `centralized_critic_context` 和三控制器动作聚合。
- 只补 PPO/MAPPO/DQN-family 仍可能被审稿人质疑缺少 value-decomposition MARL 对照；controller-level QMIX 可在不伪造 vehicle/RSU multi-agent wrapper 的前提下补齐这一类对比。

影响：

- `src/agents/mappo_agent.py` 使用 flat semantic encoder、三头 controller actor、centralized flat semantic critic，并关闭 SA-GHMAPPO 的 graph/prediction/uncertainty/dependency-aware/auxiliary/guard/imitation 机制。
- `src/agents/qmix_agent.py` 使用 flat semantic encoder、三组 controller Q heads 和 centralized monotonic mixer，并关闭 SA-GHMAPPO 专属机制。
- `scripts/run_top_journal_learned_baseline_suite.py` 与 `scripts/run_top_journal_final_submission_loop.py` 的默认 learned baseline set 更新为 `ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`。
- `scripts/build_top_journal_comparison_report.py` 支持 MAPPO/QMIX 作为当前 primary learned comparators；旧 artifact 若不含 MAPPO/QMIX，会在 protocol matrix 中标注为“当前代码可用但本 artifact 未覆盖”。
- `final_submission_clean_retrain_repaired_baselines_20260507_v1` 保留为 pre-MAPPO/QMIX-controller-level historical package；当前 MAPPO/QMIX 主表以 `final_submission_controller_mappo_qmix_20260509_v1` 为准。
- `final_submission_controller_mappo_qmix_20260509_v1` 已按该口径重跑并成为当前 canonical final package；旧 clean-retrain repaired-baseline package 降为 historical pre-MAPPO/QMIX artifact。

边界：

- 当前 MAPPO 是 controller-level CTDE，不应写成 vehicle-level 或 RSU-level full MAPPO。
- 当前 QMIX 是 controller-level monotonic value-decomposition，不应写成 vehicle-level 或 RSU-level full QMIX。
- MAPPO/QMIX smoke train/eval 只证明实现链路和 contract；论文数值必须来自多 seed formal/holdout final-submission artifact。

## 2026-05-10: 新增 Controller-MAT 并采用同环境交互预算

决策：新增 `controller_mat` 作为 controller-level Multi-Agent Transformer style learned baseline。它在当前 `semantic_discrete_5` 动作 contract 下，以 cache、execution/offload 和 handoff-event controller 作为 transformer tokens，使用 flat semantic encoder 与 centralized transformer critic；默认进入后续 learned-baseline / final-submission gate。

原因：

- 用户要求新增对照算法，并且参考顶刊顶会口径强化 baseline set；相比 TD3/SAC/MADDPG，Controller-MAT 不需要伪造连续动作 contract，也不需要立即引入 vehicle/RSU-level wrapper。
- Multi-Agent Transformer 属于 NeurIPS 2022 级别的近期 MARL 方向；按 controller-agent contract 适配后，可以补齐“attention/transformer MARL”对照类别。
- 训练预算不照搬 Atari/SMAC 的绝对 frame/step 数，而采用同环境交互预算：同 seeds、同 NGSIM+Alibaba 窗口、同 workflow、同 episode/max-step 预算、同 formal/holdout/support gate、同 cluster bootstrap 与 duplicate trace audit。

影响：

- 新增 `src/agents/mat_agent.py`、`configs/algo/controller_mat.yaml`，并接入 `registry.py`、checkpoint loader、contract tests、learned suite、final-submission loop、closed-loop 默认 baseline set 和 comparison report protocol matrix。
- 新 run 的 paper-grade learned set 为 `ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`。
- `final_submission_controller_mappo_qmix_20260509_v1` 不含 `controller_mat`，因此只能作为 pre-Controller-MAT canonical package；含 Controller-MAT 的论文主表必须重新跑 final-submission loop。

边界：

- `controller_mat` 是 controller-level transformer CTDE baseline，不是 vehicle-agent / RSU-agent full MAT。
- 它不能使用 SA-GHMAPPO 的 graph encoder、surrogate prediction features、uncertainty、DAG dependency-aware features、mechanism auxiliary loss、heuristic imitation 或 policy guards。
- 论文中应把新增 baseline 用作强化 learned comparator set 和突出 SA-GHMAPPO 机制贡献的证据链，而不是声称所有 MARL transformer 变体都已穷尽。

## 2026-05-10: 新增 DAG/cache/DT 领域专项 learned baselines

决策：在当前 `semantic_discrete_5` controller-agent contract 下新增 `dag_offload_drl`、`cache_offload_drl` 和 `dt_handoff_drl`。三者分别作为 DAG task offloading、model/adapter cache + offloading、Digital Twin handoff/service migration 的领域专项 learned comparators，默认进入后续 learned-baseline / final-submission gate。

原因：

- 当前项目主线不是泛化 MARL 排名，而是跨 RSU 连续 DAG workflow、adapter/model cache、handoff state migration、Digital Twin/surrogate prediction 与多时间尺度控制。
- 只增加通用 PPO/MAPPO/QMIX/MAT 容易被审稿人质疑缺少领域近邻对照；DAG/cache/DT 三条 baseline 能分别回应 workflow dependency、model cache/offloading 和 handoff/migration/DT 三类审稿点。
- 三个 baseline 使用同环境交互预算：同 seeds、同 NGSIM+Alibaba 窗口、同 workflow、同 episode/max-step 预算、同 formal/holdout/support gate 和 duplicate trace audit；不照搬其他 benchmark 的绝对 step 数，也不给领域 baseline 额外 synthetic rollout。

影响：

- 新增 `src/agents/dag_offload_agent.py`、`src/agents/cache_offload_agent.py`、`src/agents/dt_handoff_agent.py` 和对应 `configs/algo/*.yaml`。
- `registry.py`、`__init__.py`、checkpoint loader、contract tests、learned suite、final-submission loop、closed-loop 默认 baseline set 和 comparison report protocol matrix 已接入新 baseline。
- 新 run 的 paper-grade learned set 为 `ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl`。
- `final_submission_controller_mappo_qmix_20260509_v1` 不含 `controller_mat` 和这三个领域 baseline；含新增对照的论文主表必须重新跑 final-submission loop。

边界：

- `dag_offload_drl` 使用 DAG progress/frontier/critical-path/node-IO 等标量，不使用 SA-GHMAPPO 的 DAG graph message passing encoder。
- `cache_offload_drl` 使用 cache occupancy、adapter readiness、cache demand 和 future load 等标量，不使用 SA-GHMAPPO 的 surrogate/guard 机制。
- `dt_handoff_drl` 使用 raw Digital Twin prediction snapshot，不使用 SA-GHMAPPO 的 calibrated surrogate gate、uncertainty-aware event scaling、temporal smoothing 或 handoff guards。
- 这些新增 baseline 的 smoke train/eval 只能证明 contract 和链路可用；正式数值结论必须来自后续多 seed formal/holdout final-submission artifact。
