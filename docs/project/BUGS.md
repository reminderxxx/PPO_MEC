# Bugs And Risks

用途：记录当前有效问题、风险和禁止误读项。

## 2026-06-21: strict-full statistical blocker（RESOLVED）

- v7 的负向结论保持有效，但 v8 已按修复条件完成：四个 split 各 20 个互斥 outer windows、minimum gap 24 frames、5 seeds、window-outer hierarchical BCa/Holm、候选冻结后 formal 与一次性 hidden。
- v8 对 DT 的 full reward 与 continuity 在 formal/hidden 的 BCa 95% CI 均为正；原 strict-full blocker 标记 resolved，不再用 v7 legacy gate 充当修复证据。
- 修复不等于全面 TMC-ready。最新判定仍为 `Major revision`，详见 `top_journal_readiness_audit_20260621.md`。

## 2026-06-21: v8 system-tradeoff 与泛化缺口（OPEN）

- hidden 相对 PPO 的 handoff failure 显著更差；formal/hidden 相对 PPO 的 backhaul cost 显著更高。任何“failure-safe”或“降低回传开销”主张当前都会成为 blocker。
- formal/hidden 相对 popularity heuristic 的 reward CI 均跨 0；不能声称显著优于 strong heuristic。
- v8-current prediction robustness、system robustness、scalability 和逐机制消融已有统一入口，但尚未完成 full run 与 raw-row/statistics 审计；旧 v7 support suite 不能替代。
- LuST 修正 grid 只有 4 个独立 outer windows，低于 12-window 门槛；只能作为低功效辅助证据。
- hidden 已开启一次并永久 consumed。后续优化只能使用 dev 或新冻结 future validation split，不得再次读取现有 hidden 做候选选择。

## 当前限制

- `top_journal_mechanism_v21_efficiency_prd` 与 `top_journal_mechanism_v22_validated_utility_prd` 未能形成 paper-ready stronger-heuristic 优势。v21 dev 对 popularity reward delta 为 `+0.29125`，BCa CI `[0.01425, 0.811673]`，但 Holm p=`0.0789`；formal delta 只有 `+0.17275`，CI 跨 0，且仍有 backhaul 和 mechanism-attempt-without-validated-success blocker。v22 dev 提升了 mechanism realization（delta `+0.05`，Holm p=`0.046872`），但 reward delta 降至 `+0.268` 且 Holm p=`1.0`；formal reward delta 降至 `+0.10025`，blocker 未消除。hidden holdout 未开启，不能把 v21/v22 写成论文主 claim。
- v20/v21/v22 formal 诊断已经使用 `configs/experiment/top_journal_v20_formal_time_audited_20260717/future_validation_window_plan.json` 反馈算法设计，因此该 formal split 不能再作为未调参消费的最终 holdout。若后续候选看起来达标，必须重新冻结未消费的 formal/hidden 或只在预先冻结、未读取的 hidden 上做一次最终验证，并清楚记录开启条件。
- v20 formal/hidden 诊断 split 在排除历史窗口后没有可用 idle/sparse 窗口，只包含 10 mechanism / 10 active non-mechanism。它适合暴露 mechanism/active-heavy blocker，但不能替代 balanced strict split，也不能证明 idle/sparse 泛化。
- `top_journal_mechanism_v20_idle_execution_prd` 是当前最强算法候选：在 frozen dev 上 SA-GHMAPPO reward `79.7195` > popularity `79.46875`，在 time-audited v20 future-validation 上 reward `67.561867` > popularity `65.754`、PPO `65.866133`、MAPPO `64.0888` 和全部其他对照；相对 popularity 的 future reward delta `+1.807867`，BCa 95% CI `[0.373706, 3.793892]`，Holm sign-test p=`0.047208`。但它仍不能写成 TMC-ready / paper-ready：future split 只有 15 个 outer windows 且 `minimum_gap_frames=0`，还缺新 formal/hidden/support suite，训练 summary 仍有 collapse flags，future 上 `mechanism_realization_rate` 低于 popularity、adapter migration overhead 略高。
- v20 的收益来自 learning-side idle-execution partial-reward-decoupled MAPPO credit：改动 PPO/MAPPO actor advantage 与 option-gate advantage，让低风险 idle/current-RSU delay 与 local fallback 的 credit 分离。不得写成环境 reward shaping、evaluation wrapper、baseline 削弱或 window 筛选收益；也不得声称每个机制指标都全面优于 heuristic。
- `top_journal_mechanism_v19_handoff_risk_prd` 是 v17 之后的 handoff-risk PRD 中间候选，frozen dev reward `79.69925` 高于 popularity 但低于 v17 `79.70825` 和 v20 `79.7195`，当前不作为主候选。它的 handoff-risk credit / dual-cost 逻辑只作为 v20 的组成和后续风险约束依据。
- `top_journal_mechanism_v18_counterfactual_option` 是一次算法性 counterfactual option-credit MAPPO 尝试，但 frozen dev 结果低于 v17，且出现 continuity / handoff failure / mechanism readiness blocker；当前不得把 v18 写成主算法改进成功，只能作为负向探索和后续 credit-assignment 依据。
- `top_journal_mechanism_v17_dag_aware_option` 在 time-audited future-validation 上均值仍为第一，但相对 `popularity_cache_heuristic` 的 reward margin 只有 `+0.04815`，BCa 95% CI `[-0.396869, 0.636962]` 且 Holm sign-test p=`1.0`；不能声称显著优于 strong heuristic，不能据此判为 TMC-ready candidate。
- future-validation split 必须使用 `future_validation_split_v2_time_audited_20260717` 或后续更严格版本；只按 `frame_offset` 审计的 split 可能漏掉 `time_index_start/end` 重叠。任何旧 `top_journal_v17_future_validation_20260717` 结果不得作为 independent holdout / future evidence。
- `top_journal_mechanism_v17_dag_aware_option` 已在 frozen dev full_stratified 上完成 5-seed / 20-window / 2-workflow / 12-agent 全量 benchmark，并把 strongest-other reward margin 提升到 `+0.2395`，同时清除 v13/v16 的 `backhaul_cost_above_popularity` blocker；这仍不是 hidden/future-validation 证据。当前 hidden 已 consumed，v17 不得用现有 hidden 做进一步筛选；promotion 必须新冻结 future-validation split 并按 top-journal review policy 重新审查。
- v17 改进来自 policy-side DAG-aware MAPPO option termination，不是环境 reward、action contract、baseline contract、window plan 或评估包装改动。论文或汇报中可以说 v17 是当前 dev 主候选，但不能声称已 TMC-ready、全面优于 PPO 的所有系统指标，或已经通过独立 holdout。
- v17 的 mechanism realization `0.195` 低于 v16 `0.265`，说明 DAG-aware gate 用更保守的机制动作换取 reward / backhaul trade-off；不能把它写成“机制触发越多越好”，必须同时报告 validated success、continuity、backhaul 和 total reward。
- `top_journal_mechanism_v13_prd_option` latest 已在 frozen dev full_stratified 上完成 5-seed / 20-window / 2-workflow / 12-agent 全量 benchmark，并把 strongest-other reward margin 从 v12 `+0.12465` 扩大到 `+0.17590`；这仍不是 hidden/future-validation 证据。当前 hidden 已 consumed，v13 不得用现有 hidden 做进一步筛选；promotion 必须新冻结 future-validation split 并按 top-journal review policy 重新审查。
- v13 的 `best_by_reward` checkpoint 会停留在 warm-start update 0，不能代表 PRD 学习后的策略；本轮正向结果来自 `latest_checkpoint_path`，必须在论文或汇报中如实说明 checkpoint policy。不得把 latest-after-training 包装成 hidden-validated 或 reward-oracle selection。
- v13 改进来自 policy-side partial-reward-decoupled MAPPO event/option credit，不是环境 reward、action contract、baseline contract 或 window plan 改动。PPO 在 handoff failure/backhaul trade-off 上仍需单独报告，不能因为 reward margin 扩大而声称系统指标全面优于 PPO 或 heuristic。
- `top_journal_mechanism_v12_learned_option` 已在 frozen dev full_stratified 上完成 5-seed / 20-window / 2-workflow 全量 benchmark，并超过 `popularity_cache_heuristic` 和全部 learned baselines；这仍不是 hidden/future-validation 证据。当前 hidden 已 consumed，v12 不得用现有 hidden 做进一步筛选；promotion 必须新冻结 future-validation split 并按 top-journal review policy 重新审查。
- v12 的 full-dev 胜出来自 regime-aware 组合：mechanism window SA `82.758` > popularity `82.3425`，active non-mechanism 与 popularity/PPO 持平，idle/sparse 与 popularity 持平。不能声称每个 window class 或每个系统指标都全面优于规则；PPO 在 handoff failure `0.02` 和 backhaul `100.64` 上仍优于 v12 的 `0.075` / `110.72`。
- v12 的 learned option gate 是 policy-side MAPPO option head + contextual prior，不是环境 reward 改动、action contract 改动或 baseline contract 改动。若论文要作为算法创新表述，必须说明 `window_class` 的 outcome-blind 来源、mechanism window preserve-MAPPO 规则、option loss/prior 的训练角色，以及和 v11 evaluator-side hard gate 的区别。
- `top_journal_mechanism_v11_mappo_reward` 已在 frozen dev full_stratified 上超过 `popularity_cache_heuristic` 和全部 learned baselines，但这不是 hidden/future-validation 证据。当前 hidden 已 consumed，v11 不得用现有 hidden 做进一步筛选；promotion 必须新冻结 future-validation split 并按 top-journal review policy 重新审查。
- v11 的 full-dev 胜出不是所有 window class 全面胜出：机制窗口 SA `82.788` > popularity `82.3425`，active non-mechanism 持平，但 idle/sparse 仍为 SA `77.2175` < popularity `77.3975`。论文或汇报只能说总体 full-dev reward 过线，不能声称 idle/sparse 已彻底优于规则。
- v11 的 window-context no-RSU local fallback 由 outcome-blind `window_class=idle_or_sparse` gate 触发；它是推理期 regime-aware safety option，不是环境 reward 改动，也不是 learned predictor。若论文要将其作为算法创新，需要把 window class 的可观测来源、非 reward 选择边界和对 baselines 的公平性说明清楚。
- `top_journal_mechanism_v9_pareto_safe` 目前只是 dev/future-validation 安全候选 profile 和 checkpoint-ranking 入口；在完成 5-seed train/dev、learned-baseline 同窗口比较、future-validation split 互斥审计和新 readiness audit 前，不能替换 v8 canonical，也不能声称已解决 handoff failure / backhaul blocker。
- v9 的 `best_by_pareto_safe_score.pt` 是 checkpoint selection heuristic，不是新 reward function 或环境约束；论文必须把 reward、DT continuity、handoff failure 和 backhaul non-inferiority 分开报告，不能把 safety guard 收益写成纯 learned policy 收益。
- `top_journal_mechanism_v10_mappo_rl` 目前只是把 MAPPO controller-level credit / entropy floor 迁入 SA-GHMAPPO 的 RL 候选 profile；v11 dev result 才是本轮 reward-first follow-up。若未完成 future-validation、window-class gap、learned-baseline、failure/backhaul non-inferiority 与新 readiness audit，不得声称 MAPPO 路线已 paper-ready 修复 popularity gap 或系统 trade-off。
- 当前 `sa_ghmappo` 预测层默认仍是 `baseline_predictor_v2`。代码已新增 `predictor_kind=supervised` 和 `supervised_handoff_predictor_v1` checkpoint runtime，但在正式冻结 checkpoint、quality report、SA-GHMAPPO v9 重训和 formal/future-validation benchmark 前，不能把当前主结果写成已经使用 learned predictor。`predictor_kind=learned_or_calibrated` 仍只表示 calibrated baseline surrogate interface。
- `supervised_handoff_predictor_v1` 的安全定位是短时 next-RSU / handoff-target / ETA anticipation；不得写成完整 digital twin、轨迹预测 SOTA 或独立解决连续 cache 的核心算法。
- 当前 action contract 仍是 `semantic_discrete_5`，DAG graph encoder 与 DAG pressure diagnostics 已接入，但环境动作不选择 DAG frontier / target node；不能声明 DAG-level parameterized decision，除非后续冻结 `action_type + target_node + target_rsu/adapter` contract。
- `mechanism_exploration_bonus` 已标记为 shaping/diagnostic，但历史 reward 字段仍存在；正式机制收益必须优先用 validated prefetch hit、realized prepare、handoff ready、continuity 和 mechanism success gate，避免把 prepare/prefetch 尝试次数解释为机制兑现。
- `action_mask_info`、`ControlAction.metadata.invalid_reason`、action projection 和 guard delta 已进入新链路；历史 artifacts 没有这些字段，跨版本比较时必须显式标注 protocol version 或缺失字段。
- 旧 `final_submission_controller_mappo_qmix_20260509_v1` 中 `mappo` 是 pre-head-credit MAPPO。该结果里 `mappo` 在 continuity / handoff / backhaul 上更保守，但 total reward 弱于 `ppo`，作为“主算法基于 MAPPO 增强”的顶刊主表存在审稿风险；新的 MAPPO claim 必须改用 controller-level CTDE + `aggregation_reason_weighted_controller_ppo_v3` MAPPO 重跑。
- `paper_claim_summary.json` 中部分中文说明存在历史编码乱码；正式记录以 `docs/project/ARTIFACT_RECORDS.md` 的整理版为准。
- `smoke_run` 和 early toy benchmark 不能用于论文结论。
- `LuST` 场景仍保留 provider 价值，但当前不作为 `NGSIM + Alibaba` 主线的阻塞项或正式结论来源。
- 部分 ablation 记录使用早期 baseline checkpoint，适合作为机制对照，不适合单独声明最终 SOTA 结论。
- robustness 最新保留记录早于主表 frozen rerun，应该作为辅助压力测试，不应压过 frozen main table。
- 历史混合 aggregate 可能仍包含已删除算法记录，只能作为归档快照；当前 live 论文表格需要重新生成主方法单算法结果。
- `td3` / `sac` / `maddpg` 仍未进入当前 live registry；当前动作空间是 `semantic_discrete_5`，不应强行改写为纯连续控制实验。
- `mappo` 当前是 controller-level CTDE baseline：flat semantic encoder + cache / execution-offload / handoff-event controller actors + centralized flat semantic critic，并启用 `aggregation_reason_weighted_controller_ppo_v3`。它可以进入当前 paper-grade learned baseline gate，但 checkpoint 必须通过 `baseline_protocol_versions.mappo` 审计；pre-v3/pre-head-credit MAPPO 只能作为历史归档。它不是 vehicle-agent / RSU-agent full MAPPO；若论文声称 full multi-agent MAPPO，仍需 future multi-agent wrapper/action contract。`flat_mappo` 只表示历史 artifact run 名称，不再是 live agent 名称。
- `qmix` 当前是 controller-level value-decomposition baseline：三 controller Q heads + centralized monotonic mixer。它可以进入当前 paper-grade learned baseline gate，但不是 vehicle-agent / RSU-agent full QMIX；若论文声称 full multi-agent QMIX，仍需 future multi-agent wrapper/action contract。
- `controller_mat` 当前是 controller-level MAT-style transformer baseline：三 controller tokens + centralized transformer critic。它可以进入后续 paper-grade learned baseline gate，但不是 vehicle-agent / RSU-agent full MAT；`final_submission_controller_mappo_qmix_20260509_v1` 尚未包含它，不能把该旧 package 写成含 Controller-MAT 的最终对比。
- `dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl` 是围绕主线新增的领域专项 learned baselines：分别覆盖 DAG offloading、model/adapter cache offloading 和 Digital Twin handoff/service migration。它们按当前 controller-level `semantic_discrete_5` contract 实现，不是 full vehicle-agent/RSU-agent wrappers，也不使用 SA-GHMAPPO 的 graph message passing、calibrated surrogate gate、uncertainty-aware event scaling、mechanism auxiliary loss、heuristic imitation 或 policy guards。旧 `final_submission_controller_mappo_qmix_20260509_v1` 不含这些新增 baseline，不能把该旧 package 写成已覆盖 DAG/cache/DT 领域对照。
- `reactive_greedy` 和 `popularity_cache_heuristic` 是非学习 heuristic baseline，只用于提供规则对照，不应解释为 RL 训练结果。
- Hugging Face `model-cache` 候选全集当前只是审计、metadata 和 file-size reference；在实现文件级 importer、adapter 映射和独立 benchmark profile 前，不能声称 benchmark cache events 直接采样自这些数据集。
- formal v2 支撑实验已补齐 current-contract ablation；但 `no_dag_dependency_aware` 的 reward CI 跨 0，`no_uncertainty_signal` 不体现独立 reward 正贡献。论文中不能把这两项写成单独显著 reward 来源，只能作为机制设计组成或辅助稳定性因素谨慎描述。
- `reactive_greedy` 和 `popularity_cache_heuristic` 已降级为 supplementary heuristic reference；顶刊主 claim 应优先引用 canonical clean-retrain final comparison package，不要再把手写规则当主对照或 gate 阻塞项。旧 `top_journal_learned_baseline_formal_20260505_v1` 未覆盖当前去重后的 final-submission 口径，只能作为旧 baseline set 记录。
- DQN-family learned baselines（`dqn`、`ddqn`、`dueling_dqn`、`dueling_ddqn`）只适配当前 `semantic_discrete_5` 动作 contract；它们不能替代 TD3/SAC/MADDPG 这类连续控制 baseline。连续控制类 baseline 仍需先改变或扩展动作 contract。
- `train_sa_ghmappo_real_sample.py` 默认不再全量审计 `update_*.pt` 中间 checkpoint；如需复现完整 checkpoint consistency audit，必须显式加 `--audit_update_checkpoints`，并预期可能遇到损坏中间 checkpoint 需要容错记录。
- `top_journal_mechanism_v2` 和 clean retrain `top_journal_mechanism_v3` 虽然 learned-baseline gate 通过，但相对 supplementary `popularity_cache_heuristic` 未形成稳定优势，不能替代 formal_v2 主结果。
- `top_journal_mechanism_v3_eval_bias` 已补齐 formal/holdout 主表、latency fallback 消融、robustness 和 scalability；但它仍是在 formal_v2 权重上启用 inference calibration，不是 clean retrain，论文中必须如实说明。
- `top_journal_mechanism_v3_eval_bias` 的 prediction robustness 不满足“全面优于 heuristic upper-bound”：四类 prediction setting 汇总 SA `89.927917` 低于 popularity `90.94375`，主要由 `oracle_prediction` setting 拖累；不能写成 oracle 条件也全面领先。
- `top_journal_mechanism_v4_prepare_eval_bias` 是负向筛选结果；predictive prepare hard override 没有修复 oracle setting，反而使 prediction robustness 总 reward 低于 v3，不应推广或写入主结果。
- 当前 pre-Controller-MAT canonical `final_submission_controller_mappo_qmix_20260509_v1/comparison_report/paper_ready/paper_ready_report.md` 的自审没有 blocker，但有 4 个必须随论文主张保留的限制：`popularity_cache_heuristic` 与 SA-GHMAPPO reward 很接近；`no_prediction` / `oracle_prediction` 不支持全面预测条件优势；`mechanism_realization_rate` 不是每个 split 上的独立正向优势；holdout backhaul 对 PPO 不具备正 CI。新增 `controller_mat` 后需要重跑 final-submission loop 才能升级 canonical。
- `final_submission_clean_retrain_repaired_baselines_20260507_v1` 是 pre-MAPPO/QMIX-controller-level historical package；所有 legacy canonical 标签均受 2026-06-18 strict window audit 结论约束。

## 当前风险

- 2026-06-21 最近邻审查确认：TMC 2026 已有 DAG timing/data dependency + MADDPG，以及 mobility-aware parallel-task cross-RSU collaborative offloading；IoT Journal 2025 已有 dependency-aware hierarchical VEC offloading。论文不能把 `DAG + mobility + MARL`、`DAG + hierarchy` 或 graph-assisted VEC offloading 单独写成 novelty。
- `Dual Dependency-Aware Collaborative Service Caching and Task Offloading in VEC` 已覆盖 DAG/task dependency、service dependency、hierarchical cache 和 PPO；若 PPO_MEC 缺少 adapter size/load/warm/migration latency 或 serving-profile 证据，adapter cache 容易被审稿人视为 service cache 重命名。
- 当前可守 novelty 是完整联合 contract，而非单组件：跨 RSU continuous workflow state、adapter warm-state lifecycle、predictive handoff preparation/state migration 和 cache/execution/event 三时间尺度控制。任一元素拆开都已有强近邻，相关绝对首次表述会形成 novelty blocker。

- 2026-06-18 rebuild 已达到 `E3_REPRODUCED`，但旧 `window_rank_offset=3` formal/holdout 时间区间重叠；历史 offset holdout 只能作为 near-window sensitivity，不能作为 independent holdout。
- v7 严格非重叠协议下，SA 对 `dt_handoff_drl` 的 formal-full 与 holdout-full total-reward 95% CI 均跨 0；该历史 blocker 已由 v8 冻结 formal/hidden 修复，但不得反向把 legacy v7 gate 写成有效 strict evidence。
- mixed/full 会复用部分窗口，必须按 mode 分开报告，不能把 mode 当独立 cluster 合并扩大样本量。
- `no_prediction` 与 `no_adapter_prefetch` 消融高度耦合，不能解释为正交因果贡献或将 delta 相加。
- LuST grid 外部迁移的 reward 对 learned baselines 为正，但 backhaul 高于 popularity heuristic；不得声称全面改善系统指标。
- paired comparisons 尚未做 family-wise/FDR 校正，论文必须说明 multiplicity 策略。
- 若继续删除 artifacts，需要先确认对应路径没有被 `ARTIFACT_RECORDS.md` 的保留记录引用。
- 若重新生成主表，必须同步更新 `paper_main_table.json`、`paper_claim_summary.json` 和本目录下的整理记录。
- 若更换 checkpoint，必须同步检查 benchmark 消费端、manifest 和训练审计字段。
- 当前 reward 中 `mechanism_exploration_bonus` 会奖励预测 handoff 信号下的 prepare/prefetch 选择，未区分 prepare 是否最终成功；分析 heuristic reference 时需要同时报告 `migration_success_count`、`migration_failed_count`、handoff ready 和 continuity，避免把失败 prepare 尝试误读成真实系统收益。
- `--window_rank_offset` 只能用于 ranked-window sensitivity；独立 holdout 必须同时使用 interval exclusion、非重叠选择和 `scripts/audit_window_independence.py` 校验。
- 根目录下少数 `pytest-cache-files-*` 临时目录在本次清洗中被系统权限锁定，内容无法枚举；它们不属于项目 live 逻辑，但仍需在句柄释放后删除。
- 若后续启用 MADDPG 或 full vehicle/RSU-level QMIX，需要先冻结 multi-agent observation/action schema，再接入训练和 benchmark；当前 `qmix` 仅是 controller-level value-decomposition baseline。

## 已清理或不再阻塞

- 通用模板目录不再作为 live 文档入口。
- 旧阶段文档不再作为事实来源。
- toy / tmp / quickcheck / 单次 dry-run 产物不再参与当前结论。
