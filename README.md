# PPO_MEC

G14R14 已冻结 Formal Protocol `2.5.0`、Generated Checkpoint Resource Identity Contract `1.0.0` 与
Readiness v17=`READY_FOR_G14C_V14_CLEAN_TRAIN_AND_FORMAL`。static registry 只登记 pre-run immutable inputs；
当前 run 的 checkpoint manifest/provenance 由 committed `checkpoint_freeze` 原子发布到独立、create-only 的
generated registry。cache outer/child、controller、ablation、support、scalability、statistics/integrity/gate
统一先验 identity。clean detached non-formal rehearsal 已真实完成 13/13 phases、30 checkpoint loads 与三档
consumer closure，但 formal training/checkpoint/performance 均为 0，未启动 G14C v14、G14D、G15 或 holdout。
详见 `docs/project/formal_generated_checkpoint_resource_identity_contract.md`。

G14R13 已修复 active preflight 的 Protocol capability routing 漏配，冻结 Formal Protocol `2.4.0`、
Capability Routing Contract `1.0.0` 与 Readiness v16=`READY_FOR_G14C_V14_CLEAN_TRAIN_AND_FORMAL`。唯一
active index 为 `configs/experiment/typed_model_cache_formal_protocol_v2_4_20260905/protocol_index.json`；
v1.0–v2.3 全部 audit-only，未知版本 fail-closed。G14C v13 仅登记为
`PRE_EXECUTION_STOP / VALIDATOR_VERSION_DISPATCH_MISMATCH`，没有 run、ledger、checkpoint 或 performance；
本轮 formal training/checkpoint/performance 仍为 0，holdout sealed/unopened/unconsumed，未启动 G14C v14、
G14D 或 G15。详见 `docs/project/formal_protocol_capability_routing_contract.md`。

G14R11 已修复 G14C v11 首个训练单元暴露的 request subject 生命周期分叉，冻结 Formal Request Subject
Lifecycle Contract `1.0.0`、Formal Request Exposure Trace `2.0.0`、Formal Exogenous Request Execution
Contract `1.1.0`、Environment Identity Projection `1.1.0`、Execution Environment `1.2.0` 和 Protocol
`2.2.0`。唯一 active index 为
`configs/experiment/typed_model_cache_formal_protocol_v2_2_20260901/protocol_index.json`；Protocol 2.1/A12
及更早版本全部 audit-only。v11 永久登记为 `INVALID_PROTOCOL_OR_IMPLEMENTATION /`
`invalid_during_first_training_cell_before_first_episode_commit`，禁止 resume、retry、finalize、salvage 或
checkpoint reuse。G14R11 的 formal training/checkpoint/performance 均为 0，holdout sealed/unopened；没有启动
G14C v12、G14D 或 G15。详见 `docs/project/formal_request_subject_lifecycle_contract.md`。

G14R8 已建立 Active Bundle Resource Resolution Contract `1.0.0`，冻结 Protocol `1.9.0`、Active Formal
Bundle Contract `1.1.0` 与 Readiness v11=`READY_FOR_G14C_V9_CLEAN_TRAIN_AND_FORMAL`。唯一 active index 为
`configs/experiment/typed_model_cache_formal_protocol_v1_9_20260829/protocol_index.json`；所有 active consumer
只能从已验证 bundle 的 `active_bundle_resources` 解析资源，不得读取历史 `runtime_configs`/
`dev_fairness_manifests` 顶层字段。G14C v8 永久 invalid 于150/150 training、1,200 candidates完成后且首个
dev performance row前；其 checkpoint/candidate/partial dev input全部禁止复用。v1.0–v1.8均为audit-only，
holdout sealed/unopened；本轮未启动G14C v9、formal、G14D或G15。详见
`docs/project/active_bundle_resource_resolution_contract.md`。

G14R7A 已修复 Protocol active index 与 Readiness 分叉，并冻结 Active Formal Bundle Contract `1.0.0`、
Protocol `1.8.0` 和 Readiness v10=`READY_FOR_G14C_V8_CLEAN_TRAIN_AND_FORMAL`。唯一active index为
`configs/experiment/typed_model_cache_formal_protocol_v1_8_20260827/protocol_index.json`；Protocol semantic
SHA-256=`9799bf2c2f4b4665b8390c6fc5d5aa235faf11d6525e043eac289c061633b3de`，active bundle core/final SHA-256为
`96627ac414cb5dc80785c907ded2c9588dcdcf69469a5821b75fc07dc25e5b65`/
`793f5106b83f9687044aeeac122179a8c5805688d4a041c0418292345f9138bd`。outer runner在任何run-root写入前
自动从index验证Protocol、environment、scientific/order/schema、portable/fairness/data/runtime、Readiness和
clean `HEAD==origin/main`；手工CLI不能覆盖错误index。v1.0–v1.7全部audit-only。本轮正式training/
checkpoint/performance仍为0，holdout sealed/unopened，未启动G14C v8/G14D/G15。详见
`docs/project/active_formal_bundle_contract.md`。

G14R7 已冻结 Formal Agent Order Contract `1.0.0` 与 Protocol `1.7.0`。唯一主序列为 5 个
reactive（LRU/FIFO/LFU/Aging-LFU/Random）后接 10 个 learned（SA-GHMAPPO、PPO、MAPPO、DQN、
Dueling DQN、QMIX、Controller MAT、DAG Offload DRL、Cache Offload DRL、DT Handoff DRL）；JSON
mapping 插入顺序与 alphabetical sort 均不再构成身份。Order contract semantic SHA-256 为
`82e562755dadd4341c950bf71efc488d3527b7f45b7f02512f8064d189b655e0`，Protocol semantic SHA-256 为
`5a1c2070529674ecf65c8b836706849f0937853a59b6dfbc3b987d88ac4f50a5`。G14C v7 永久 invalid 于
150/150 training、1,200 candidates完成后且dev performance前，旧 checkpoint/candidate/ledger均禁止复用。
clean detached验收完成150-command、60/60真实preflight、15-agent non-formal dev/freeze/statistics链路与
全量tests，Readiness v9=`READY_FOR_G14C_V8_CLEAN_TRAIN_AND_FORMAL`；它只授权未来独立G14C v8，
不代表formal或paper-ready。本轮正式training/checkpoint/performance均为0，holdout sealed/unopened，未启动
G14C v8/G14D/G15。详见`docs/project/formal_agent_order_contract.md`。

G14R6 已将 formal learned-agent 超参数的科学身份与每次执行的 Protocol/commit/runtime 绑定拆分，冻结
scientific config `2.0.0`、execution binding `1.0.0`、resolved context `2.0.0` 与 Protocol `1.6.0`。
scientific config semantic SHA-256 为 `f83587cd13c126a0d8a6bdc26402e34ac1391bd6fc8ef504736458872d649bc8`，
Protocol semantic SHA-256 为 `f2c9e729f126d9e87f56fcdccf13f2ecd018c28ca3102b8d02b2bbd6abca95c0`。
G14C v6 永久 invalid 于首个 training cell、episode 0 前，episode/interaction/update/checkpoint全部为0且
禁止retry/resume/finalize/salvage/reuse。Readiness v8仅授权未来独立 G14C v7 clean run；本轮未启动
正式training/formal/holdout/G14D/G15，holdout保持sealed/unopened。详见
`docs/project/typed_model_cache_formal_training_identity_contract.md`。

G14R5 已修复 G14C v5 暴露的 outer/nested execution context 分叉，并冻结 Protocol v1.5 与 resolved context
`1.0.0`。G14C v5 永久 invalid，tests/training/dev/formal/holdout 均为0且禁止resume/finalize/checkpoint reuse。
detached Commit A6候选无本地`.venv`，使用共享绝对Python完成186-command dry-run和真实非正式
`preflight → tests`：NGSIM完整`11,850,526` rows、60/60 frozen windows可达，outer/nested expansion hash一致，
全仓`1038 passed`。Protocol semantic SHA-256为
`feb7ccc489d66aeba502fbef2fef70c911ecdd66218a8ea3d475725ec61d829a`，Readiness v7为
`READY_FOR_G14C_V6_CLEAN_TRAIN_AND_FORMAL`。该状态仅授权未来独立任务新建G14C v6 run；正式training/
checkpoint/performance仍为0，holdout sealed/unopened，G14D/G15未启动。详见
`docs/project/typed_model_cache_formal_resolved_execution_context_contract.md`。

G14R4+ 已联合修复 G14C v4 暴露的长 phase 终结、per-cell 事务/same-run resume 与 clean-worktree
Python 解析问题，并在未训练正式 checkpoint、未运行 formal/holdout/hidden、未启动 G14C v5/G15 的
前提下冻结 `typed_model_cache_formal_protocol_version=1.4.0`。两个 v4 run 分别永久登记为
`invalid_after_training_before_dev_performance_execution`（150/150 cells、1,200 candidates、dev/formal=0）
和 `invalid_before_first_frozen_subcommand`（所有执行计数为 0）；两者均禁止 resume、finalize 或 checkpoint
salvage。新合同包含 portable execution environment `1.0.0`、phase ledger `3.0.0`、cell ledger `1.0.0`、
completion candidate/finalize-only 和 hash-bound atomic commit。真正不含 `.venv` 的临时 clean snapshot 使用
共享绝对 Python 完成 16-cell 非正式 exact chain，并通过 8/16、75/150 中断恢复及 train terminal append
失败模拟；项目 import 全部来自 clean snapshot。Protocol v1.4 semantic SHA-256 为
`4429531dc3cf98e7ef332367e55e1d0a3dbc33773c20a3fe2e53e57d3534155d`，Readiness v6 为
`READY_FOR_G14C_V5_CLEAN_TRAIN_AND_FORMAL`。该状态只授权未来独立任务从最终 pushed Commit A5 的 clean
worktree 新建 run；holdout 仍 sealed/unopened，旧 v4 产物不可复用，G14 尚未完成。详见
`docs/project/typed_model_cache_formal_execution_environment_contract.md` 与
`docs/project/typed_model_cache_formal_execution_resume_contract.md`。

G14R2 已修复 G14C v2 暴露的 formal frozen-window 消费与 phase ledger 缺口，并在未运行正式训练、
formal、holdout 或 hidden 的前提下冻结 `typed_model_cache_formal_protocol_version=1.2.0`。G14C v2
`typed_model_cache_formal_20260820_164251_g14c_v2` 因训练命令遗漏显式 source range、回落到
`max_mobility_rows=1500` 而在首个训练单元前失败，永久标记为
`INVALID_PROTOCOL_OR_IMPLEMENTATION / invalid_before_performance_execution`：0/150 training、0 checkpoint、
0 formal，禁止 resume。新 window contract 按真实 loader 语义将 11,850,526 raw rows、73,871 provider
frames 与原 60-window split 绑定，60/60 identity/interval/fingerprint 可达；150/150 training commands 与
30 条 dev/formal/support commands 均显式传递完整 source/window 参数。Ledger `2.0.0` 记录 running/terminal
时间、wall-clock、失败枚举和 previous/current hash chain。Protocol v1.2 semantic SHA-256 为
`718c0f78aabd5d01012df31267626eab74a51b2b621aaa67a535c5b60e655ca9`，split hash 保持
`aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a`，Readiness v4 为
`READY_FOR_G14C_V3_CLEAN_TRAIN_AND_FORMAL`。这只授权未来从 Commit A3 clean worktree 另立任务，
不表示 G14/formal 完成或 paper-ready；holdout 仍 sealed/unopened。详见
`docs/project/typed_model_cache_formal_window_consumption_contract.md` 与
`docs/project/typed_model_cache_formal_protocol_restart.md`。

G14B 历史上冻结了 `historical_window_usage_registry_version=1.0.0`、
`typed_model_cache_split_protocol_version=1.0.0` 与
`typed_model_cache_formal_protocol_version=1.0.0`。Metadata-only 历史账本汇总 34,661 个窗口引用并
折叠为 668 个 outer intervals；无法恢复的 418 个 interval 对 lankershim/peachtree/us_101 形成保守
排除。完整 NGSIM 扫描在未消费 I-80 上构造 train/dev/formal/sealed-holdout=`24/12/12/12`，1,770
个 pairwise interval 全部满足 raw frame/time/segment-run 互斥与 24-frame gap。Readiness v2 为
`READY_FOR_G14C_CLEAN_TRAIN_AND_FORMAL`；这不是 formal 完成、checkpoint、性能结果或 paper-ready，
holdout 仍 sealed。协议见 `docs/project/typed_model_cache_formal_protocol.md`，排除审计见
`docs/project/typed_model_cache_split_exclusion_audit.md`。

G13 冻结 `typed_model_cache_contract_version=1.0.0`：显式区分 RSU resident base model、adapter 与 migration-only workflow state；legacy adapter-only继续默认。typed profile只使用MB capacity，一个旧cache action确定性解析为最多2对象的atomic base+adapter dependency bundle，并提供dependency-safe eviction、CacheEvent 1.3 layered readiness/per-type bytes、G07 typed fairness binding、G08 tiny exact oracle和G06 type-aware metrics 1.1。受控及最小NGSIM+Alibaba验证位于`artifacts/analysis/typed_model_cache_validation_20260819_g13_v1/`；不是G14、算法优势或paper-ready证据。详见`docs/project/typed_model_cache_contract.md`。

G11 冻结 `model_cache_dataset_registry_version=1.0.0`：截至 `2026-08-19` 从官方页面/API 核验 19 个 public model-serving、KV/prefix、model-artifact 与 AI workload 候选，固定 A–I taxonomy、19 项 field coverage、100 分 qualification 和 hard gates。结论为 A 类 joint VEC trace=0、C 类真实 adapter/LoRA request trace=0；BurstGPT 是最佳 B 类 request trace，Qwen-Bailian/Mooncake 是 D 类 KV trace，三个 HF 项仅可作 E 类 size metadata。接入严格 metadata-only，不改变 `NGSIM + Alibaba` 主线或正式 benchmark。详见 `docs/project/model_cache_dataset_discovery_audit_20260819.md`。

G10 冻结 `information_sufficiency_audit_contract_version=1.0.0`：`scripts/audit_cache_information_sufficiency.py` 基于源码真实 observation/action/actor contract 和严格匹配的 G07–G09 artifact，只读审计字段覆盖、recoverability、observation aliasing、固定 projection、entropy/NMI/CMI 与实体级 MARL 必要条件。当前真实 validation 只有 1 request/1 evaluation unit且无 decision-time observation trace，因此总 verdict 为 `UNVERIFIABLE`；MAPPO/SA-GHMAPPO 是 controller-level CTDE，不是 vehicle/RSU-level MARL。详见 `docs/project/cache_information_sufficiency_marl_audit_contract.md`。

G09 冻结 `cache_opportunity_analyzer_contract_version=1.0.0`：`scripts/analyze_cache_opportunities.py` 在严格匹配的 G07 manifest、G08 external replay/exact action trace、initial cache/capacity 与五个 baseline raw CacheEvent outcome 上，独立输出 demand reuse、feasible oracle opportunity、baseline capture/loss、互斥 taxonomy、gap decomposition、concentration 与 information-requirement labels。latency saved 继续 unavailable；该分析不是 causal regret、MARL 必要性、formal/holdout/hidden 或算法优劣证据。详见 `docs/project/cache_opportunity_analyzer_contract.md`。

G08 冻结 `cache_request_replay_version=1.0.0` 和 `future_horizon_cache_oracle_contract_v1.0.0`：从 G07 evaluation unit 独立生成 policy-neutral DAG/mobility request replay，在相同 initial per-RSU cache、slot/MB容量、同 step admission时序、transfer和multi-victim约束下运行 H=1/3/6/12 exact rolling oracle，并只对 fingerprint/capacity/initial-state 匹配的 raw baseline outcome输出 placement opportunity gap。入口为 `scripts/build_cache_request_replay.py`、`scripts/run_future_horizon_cache_oracle.py` 和 `scripts/audit_cache_oracle_gap.py`；详见 `docs/project/cache_request_replay_contract.md` 与 `docs/project/future_horizon_cache_oracle_contract.md`。G08 validation 不是 formal、holdout、hidden 或 latency-gain 证据。

G07 提供 paper-grade cache baseline fairness manifest `1.0.0`：五个 `reactive_*` baseline 在同一 NGSIM+Alibaba 数据、raw frame/time窗口、DAG request plan、seed、capacity、catalog、initial cache、G01/G03/G06合同下比较，唯一主要变量为 eviction policy。使用 `scripts/build_cache_baseline_fairness_manifest.py` 构建、`scripts/validate_cache_baseline_fairness_manifest.py` 校验，并通过 `benchmark_main_results.py --cache_baseline_fairness_manifest_path` 显式消费。详见 `docs/project/cache_baseline_fairness_manifest_contract.md`；validated/controlled结果不是formal或paper-ready证据。

Cache capacity supports backward-compatible `adapter_slots` and resident-size `mb` modes. The auditable eviction factory registers LRU, FIFO, LFU, Aging-LFU and seeded Random. Five matched `reactive_*` classical baselines share one reactive admission/control contract and differ only in eviction; see `docs/project/classical_cache_baseline_contract.md` and validate with `python scripts/validate_classical_cache_baselines.py`. These are benchmark-ready mechanism baselines, not paper-grade results or full cooperative caching algorithms.

CacheEvent `1.x` episode telemetry 可通过 `scripts/audit_cache_event_telemetry.py` 从 raw `cache_event_trace` 独立重算并与 legacy step/episode 字段分类对账；使用方法和口径见 `docs/project/RUNBOOK.md` 与 `docs/project/cache_event_contract.md`。该能力用于 contract 验证，不是新增论文指标。

G06 cache-efficiency contract 通过 `scripts/audit_cache_efficiency_metrics.py` 从 raw CacheEvent + trace context 独立重算 request/byte、churn、capacity、pollution 和 future-reuse proxy；latency saved 在缺少逐请求 counterfactual latency 时保持 unavailable。精确定义见 `docs/project/cache_efficiency_metrics_contract.md`；controlled/smoke 输出不是性能结论。

## 2026-07-29 v70 sparse-tail option MAPPO 正式结果

v70 `top_journal_mechanism_v70_sparse_tail_option_mappo` 已按 3 seed、12-agent、frozen mixed/full 窗口完成全量 benchmark，并首次在当前 offset-free formal-min full_stratified 协议下让 SA-GHMAPPO 同时高于 DT 规则/专项对照、popularity heuristic、PPO 和 MAPPO。关键 artifact：

- mixed: `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v70_sparse_tail_option_formal_min_20260730/benchmarks/mixed_informative_config_loaded/main_results_mixed_informative_20260730_010306_535310/aggregate_summary.json`
- full: `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v70_sparse_tail_option_formal_min_20260730/benchmarks/full_stratified_config_loaded/main_results_full_stratified_20260730_010523_184238/aggregate_summary.json`
- full hierarchical statistics: `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v70_sparse_tail_option_formal_min_20260730/statistics/full_stratified_hierarchical/paired_statistics.json`

结果：mixed 中 SA `33.763250` > DT `32.922750` > popularity `31.756250` > PPO `29.017167` > MAPPO `18.310000`；full 中 SA `32.385729` > DT `31.426667` > popularity `29.969271` > PPO `27.301597` > MAPPO `16.261458`。full-only window-outer hierarchical bootstrap 显示 SA 相对 DT 的 total reward delta 为 `+0.959063`，95% CI `[0.554171, 1.691468]`；相对 PPO 为 `+5.084132`，相对 MAPPO 为 `+16.124271`。收益来自 v70 sparse-tail option-prior 把原本 v67/v69 的 `idle_or_sparse` 短板从 `27.1015` 提升到 `30.7365`，并反超 DT `29.29275`。

结论边界：这是本轮“主算法 reward 高于其他算法”的正式 artifact 证据，但仍不是 `TMC-ready` 或 paper-ready package。当前还缺独立 hidden/future holdout、support suite、完整 artifact integrity/command-log package、消融与 robustness/scalability 复核；并且相对 DT 的 continuity 微弱为负、backhaul cost 略高，不能写成所有系统指标全面优于规则。

## 2026-07-21 MAPPO 行为正则 dev-probe 状态

v40/v41 已按用户要求做算法侧 MAPPO 改进并完成 frozen dev full-pool 复核，但没有证明“主算法高于全部算法”。当前本轮最高 SA 结果来自 v39 update_0005 full-pool：SA-GHMAPPO total reward `106.041`，高于 MAPPO `105.5875`、popularity `105.25` 和 PPO `94.77375`，但低于 `cache_offload_drl=119.14875` 与 `dt_handoff_drl=119.22625`，artifact 为 `artifacts/experiments/top_journal_closed_loop/top_journal_v39_delayed_credit_dev_probe/benchmarks/update_0005_full_pool/main_results_full_stratified_20260721_011956_616135/aggregate_summary.json`。

关键诊断是当前 `reward_positive_offset=5.0` 按 step 累加，DT/cache 的高 reward 与较低 completion/continuity 同时出现：DT/cache `successful_episode_rate=0.65`、`workflow_continuity_rate=0.583014`，SA 为 `1.0` / `0.970369`。因此 v39-v41 仍是 dev-probe，不是 paper-ready 或 all-baseline-winner 结论；若要继续冲击论文级结果，必须先冻结 completion-constrained / time-normalized objective 或重新设计 reward 后让全部 baseline 同协议重训重评。详见 `docs/project/PROGRESS.md`、`docs/project/BUGS.md` 和 `docs/project/ARTIFACT_RECORDS.md`。

## 2026-06-21 strict-full v8 审查状态

v8 已按冻结的 train/dev/formal/hidden 协议完成 5-seed formal 与一次性 hidden holdout。对全部 learned baselines 的 total reward hierarchical BCa 95% CI 在 formal/hidden 均为正，对 DT handoff DRL 的 workflow continuity 也为正；v7 的 strict-full statistical blocker 已修复。

当前 reviewer verdict 为 `Major revision (78/100)`，不是 `TMC-ready candidate`：hidden 相对 PPO 的 handoff failure 更差，formal/hidden 的 backhaul cost 更高，对 popularity heuristic 未形成显著 reward 优势；v8-current robustness/scalability/ablation 与更大外部验证仍待补齐。详见 `docs/project/top_journal_readiness_audit_20260621.md` 和 `docs/project/strict_full_v8_execution_record_20260621.md`。

2026-07-13 已接入 v8-current support suite 入口和 v9 Pareto-safe 候选路径：`scripts/run_strict_full_v8_support_suite.py` 负责补齐 prediction/system/scalability/guard attribution，`top_journal_mechanism_v9_pareto_safe` 负责在 dev / future-validation 上把 handoff failure 与 backhaul 纳入 checkpoint ranking。2026-07-16 进一步新增 `top_journal_mechanism_v10_mappo_rl` 与 `top_journal_mechanism_v11_mappo_reward`，把 MAPPO 的 controller-level CTDE head-credit / entropy-floor 机制迁入 SA-GHMAPPO 候选 profile，同时降低 imitation / mechanism auxiliary 牵引，并在 v11 中加入 reward-first checkpoint priority 与 idle/sparse window-context inference gate。v11 full-dev benchmark 已在 frozen dev plan 上让 SA-GHMAPPO total reward 高于全部对照：`79.4944` vs `popularity_cache_heuristic=79.46875`，artifact 为 `artifacts/experiments/top_journal_mappo_reward_full_dev_v11_20260716/main_results_full_stratified_window_gate_full/main_results_full_stratified_20260716_181112_383674/aggregate_summary.json`。

2026-07-17 新增 `top_journal_mechanism_v12_learned_option`：在 v11 MAPPO-core checkpoint 上 warm-start，加入可学习 contextual option gate，让策略在 `accept_mappo`、`popularity_safe`、`no_rsu_local` 和 `mechanism_prepare` 间学习选择；机制窗口显式保留 MAPPO 主策略，idle/sparse 窗口用 learned option 吸收 popularity-safe 行为。v12 full-dev 5-seed / 20-window / 2-workflow 全量 benchmark 已完成：SA-GHMAPPO total reward `79.5934`，高于 `popularity_cache_heuristic=79.46875`、`ppo=77.18775`、`mappo=72.6328` 及全部其他对照，artifact 为 `artifacts/experiments/top_journal_mappo_reward_v12_learned_option_20260717/main_results_full_stratified_mech_preserve/main_results_full_stratified_20260717_115754_212344/aggregate_summary.json`。v12 仍是 dev evidence，不是 hidden/future-validation 或 paper-ready 结论；hidden holdout 已 consumed，不能再用于筛选。

同日新增 `top_journal_mechanism_v13_prd_option`：在 v12 基础上加入 partial-reward-decoupled MAPPO event/option credit，并把 v13 closed-loop checkpoint policy 改为 `latest_checkpoint_path` 优先，以评估 PRD 训练后的学习策略。v13 latest full-dev 5-seed / 20-window / 2-workflow / 12-agent benchmark 中 SA-GHMAPPO total reward `79.64465`，高于 v12/best-by-reward `79.5934`、`popularity_cache_heuristic=79.46875`、`ppo=77.18775`、`mappo=72.6328` 及全部其他对照；strongest-other margin 从 v12 `+0.12465` 扩大到 `+0.17590`，artifact 为 `artifacts/experiments/top_journal_prd_option_v13_20260717/main_results_full_stratified_latest/main_results_full_stratified_20260717_124815_375515/aggregate_summary.json`。v13 仍是 dev evidence；promotion 需要新冻结 future-validation split 和 readiness audit。

同日进一步新增 `top_journal_mechanism_v17_dag_aware_option`：在 v16 conservative terminal option 上加入 DAG-aware MAPPO option termination，用 critical-path / workflow-size / branching / prediction confidence 判断机制动作时机。v17 full-dev 5-seed / 20-window / 2-workflow / 12-agent benchmark 中 SA-GHMAPPO total reward `79.70825`，高于 `popularity_cache_heuristic=79.46875`、`ppo=77.18775`、`mappo=72.6328` 及全部其他对照；对 popularity margin `+0.2395`，`sa_advantage_diagnosis.blockers=[]`，backhaul 与 popularity 持平 `110.8`。artifact 为 `artifacts/experiments/top_journal_dag_aware_option_v17_20260717/main_results_full_stratified_latest/main_results_full_stratified_20260717_154951_203519/aggregate_summary.json`。v17 是当前 dev 主候选，但仍不是 TMC-ready；promotion 需要新冻结 future-validation split、层级统计、support suite 和 readiness audit。

同日执行 `top_journal_mechanism_v18_counterfactual_option`：在 v17 基础上加入 COMA-style counterfactual option credit，即 selected option partial utility 减去同状态合法 option 的 policy-probability-weighted utility。该改动不修改 reward、action/environment contract、baseline contract 或 window plan。v18 full-dev 结果为 SA-GHMAPPO `79.4897` vs popularity `79.46875`，弱于 v17，且诊断 blocker 包含 continuity / handoff failure / mechanism readiness，因此不晋级为主候选。

随后冻结 `future_validation_split_v2_time_audited_20260717`，要求 frame_offset 与 time_index 双区间都和 train/dev/formal/hidden 保持至少 24 frames gap。time-audited future-validation 结果为 SA-GHMAPPO `77.56665`，高于 popularity `77.5185`、PPO `76.53095`、MAPPO `70.5285` 和全部其他对照；但对 popularity 的 reward delta 仅 `+0.04815`，BCa 95% CI `[-0.396869, 0.636962]`，Holm 后 sign-test p=`1.0`，且 handoff failure 仍略高于 popularity。结论：当前算法性改进能拉开对 MAPPO/PPO 的差距，尚不足以支撑“显著优于强规则基线”的 TMC-ready 主张。

## 2026-06-21 导师汇报材料

- 可编辑汇报 PPT：`outputs/ppo_mec_advisor_report_20260621.pptx`
- 中文讲稿、创新点、模型架构与结果边界：`docs/project/advisor_report_briefing_20260621.md`

该材料基于 2026-06-18 E3 独立复现证据。可展示结论是：严格非重叠 mixed formal/holdout 对最强 learned baseline 的 reward 置信区间为正；full formal/holdout 仅点估计领先、95% CI 跨 0。材料不得被解释为项目已经达到 `TMC-ready`。

## 完整克隆（含真实数据）

仓库中的 `data/` 通过 Git LFS 版本化。首次克隆或切换到包含数据的分支前，需先安装 Git LFS，然后执行：

```bash
git lfs install
git lfs pull
python scripts/check_data_ready.py
```

若未执行 `git lfs pull`，`data/` 下只会保留 LFS 指针，真实数据链路无法运行。当前正式主线所需的 NGSIM、Alibaba 和 LuST 数据已纳入；highD 仍是未提供的后补数据源。

## 2026-06-18 v7 独立重建与严格审查

当前主机已用 `top_journal_mechanism_v7_latency_fallback_20260618_rebuild_v1` 和 `final_submission_v7_latency_fallback_20260618_rebuild_v1` 独立 clean retrain，复现 2026-05-28 legacy formal/final gate，并完成 SHA-256、manifest、checkpoint provenance、机制消融和 LuST external mobility 检查。

严格审查发现旧 `offset=3 holdout` 与 formal 滑动窗口重叠，不能称独立 holdout。改用 split 内及 split 间时间不重叠窗口后，mixed formal/holdout 对 `dt_handoff_drl` 的 paired CI 为正，但 full formal/holdout CI 跨 0。因此 v7 的 verdict 为 `Not TMC-ready`；legacy `paper_claim_ready=true` 只说明旧项目 gate 可复现。该 blocker 已由上文 v8 协议修复，v7 历史审查见 `docs/project/top_journal_readiness_audit_20260618.md`。

## 2026-05-28 SA v7 legacy final-submission package

- Current paper-ready package: `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/`.
- Final gate: `target_reached=true`, `paper_claim_ready=true`, `blockers=[]`; comparison package: `review_ready=true`, `paper_ready_package_ready=true`.
- Main method profile: `top_journal_mechanism_v7_latency_fallback`, a clean-retrain profile that keeps v6 freshness/admission guards and enables latency fallback fast-timescale execution control.
- Paper-grade learned baselines in this package: `ppo`, `mappo`, `dqn`, `dueling_dqn`, `qmix`, `controller_mat`, `dag_offload_drl`, `cache_offload_drl`, `dt_handoff_drl`.
- 在 legacy formal/offset-3 协议中，SA-GHMAPPO ranks first；这些数值已复现，但 offset-3 不能再标为 independent holdout。
- `popularity_cache_heuristic` remains a close supplementary reference, not a learned-baseline gate blocker: SA margins are `+0.250000`, `+0.479629`, `+0.355556`, and `+0.376191` across formal/holdout mixed/full.
- Reviewer-facing limitations from the generated self-review must be preserved: heuristic gap is close, mechanism realization is not uniformly a standalone CI-positive advantage, and backhaul savings are not universal.

## 2026-05-27 MAPPO v3 / SA v6 update

- `mappo` live baseline now uses `aggregation_reason_weighted_controller_ppo_v3`: controller-head credit floors and entropy floors/scales are applied to slow / fast / event heads to reduce action-mix collapse while keeping MAPPO free of SA-GHMAPPO-only graph/surrogate/guard mechanisms.
- Paper-grade learned-baseline loops accept `--mappo_baseline_profile mappo_strong_audit`; this profile is the default MAPPO profile inside the learned-suite/final-loop wrappers.
- Main-method optimization adds `top_journal_mechanism_v6_strong_competition` and `configs/experiment/top_journal_mechanism_v6_strong_competition.yaml` for a future same-budget rerun against optimized baselines.
- SA v6 now uses a freshness-aware cache-warm guard (`cache_warm_start_guard_max_prefetch_countdown=6.0`) so predictive prefetch is not forced before the recorder validation window.
- SA v6 also uses a confidence/alignment prefetch admission guard (`predictive_prefetch_admission_min_confidence=0.55`) so low-confidence prefetch is deferred until next-RSU and handoff-target evidence align.
- The latest 3-seed freshness-guard closed loop (`top_journal_mechanism_v6_freshness_guard_20260527_v1`) remains a negative candidate: SA still trails `popularity_cache_heuristic` by `0.055556` mixed / `0.018519` full reward and is not paper-ready.
- This v6 note is historical. The v7 legacy result has been reproduced, but the current strict reviewer verdict is `Not TMC-ready`.

PPO_MEC 是面向 AI-driven VEC 的研究原型，主线围绕跨 RSU 连续 DAG workflow 执行、车载 base model 与路侧 adapter cache 协同、handoff 状态迁移、surrogate prediction 和多时间尺度控制。

当前正式数据主线是 `NGSIM + Alibaba`。`LuST` 与 `highD` 保留 provider / 检查骨架，但不阻塞正式主线。

数据源声明统一维护在 `docs/project/DATASET_SOURCES.md` 和 `configs/data/dataset_sources.json`。当前已审计并 metadata-only 接入 Hugging Face model-cache 候选全集，用于 catalog/report 审计和后续 file-size profile 设计；不会自动下载模型文件，也不会替换正式 benchmark 默认 cache 行为。接入边界见 `configs/data/hf_model_cache_integration_plan.json` 和 `docs/agent/hf_model_cache_dataset_audit_round14_report.md`。

## Supervised Handoff Predictor v1

当前代码已接入薄 supervised handoff predictor 路径：`scripts/train_supervised_handoff_predictor.py` 可从冻结 train/dev window plan 训练短时 next-RSU / handoff-target / ETA predictor；`PredictorManager` 支持 `predictor_kind=supervised` 与显式 `predictor_checkpoint_path`。该层用于 handoff anticipation 和 lightweight DT-style predictive state snapshot，不是完整数字孪生系统，也不代表 predictor 本身已经形成 paper-ready 主结论；正式 claim 仍需要冻结 checkpoint、quality report、SA-GHMAPPO v9 重训和 formal/future-validation benchmark。

G12进一步冻结 `causal_predictor_snapshot_contract_version=1.0.0`：generated/as-of/consumed时间、calibrated probabilities、staleness、abstention/mask和oracle隔离。入口仍默认关闭，不进入canonical profile。非hidden calibration/validation审计可运行：

```bash
.venv/bin/python scripts/audit_predictor_calibration.py \
  --run_id causal_predictor_snapshot_validation_20260819_g12_v1
```

该命令不训练、不调参、不运行formal/holdout/hidden或RL性能比较。合同和claim边界见`docs/project/causal_calibrated_predictor_snapshot_contract.md`。

## 当前模型层

主方法：

- `sa_ghmappo`: `Surrogate-Assisted Graph Hierarchical Multi-Agent PPO`

方向匹配对照算法池：

- 可训练 learned 对照：`ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl`；`ddqn` / `dueling_ddqn` 仅在 duplicate trace audit 通过时作为可选补充。
- contract-blocked diagnostic 对照：`ippo`。当前 single-wrapper decision stream 不能支撑 paper-grade independent IPPO；`mappo` 已实现为 controller-level CTDE baseline，并且当前 paper-grade 协议要求启用 aggregation-reason controller head-credit，避免三控制头共享错误 credit；`controller_mat` 已实现为 controller-level transformer CTDE baseline，二者都不是 vehicle-agent / RSU-agent full MARL wrappers；`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl` 分别作为 DAG/cache/DT 领域专项 learned baseline。
- 非学习启发式对照：`reactive_greedy`、`popularity_cache_heuristic`
- 历史 artifact 路径中仍可能出现 `flat_ppo` / `flat_mappo` run 名称，但它们不再是 live agent 名称。
- TD3 / SAC / MADDPG 当前不进入 live registry；后续接入前必须先冻结匹配的 observation/action contract。`qmix` 已按 controller-level value-decomposition contract 接入，不是 vehicle-agent / RSU-agent full QMIX。

## Agent 结构

`src/agents/` 只按算法分文件：

- `base_agent.py`
- `registry.py`
- `sa_ghmappo_agent.py`
- `sa_ghmappo_core.py`
- `ippo_agent.py`
- `ppo_agent.py`
- `mappo_agent.py`
- `dqn_agent.py`
- `reactive_greedy_agent.py`
- `popularity_cache_heuristic_agent.py`

`registry.py` 直接导入算法文件。PPO / MAPPO 不再通过 `ppo_family.py` 或分类 package 组织。

## 目录概览

- `src/envs/`：核心环境、预测层和 Gym wrapper
- `src/envs/specs/action_schema.py`：语义动作 schema、mask 和 action adapter
- `src/data/`：mobility、workflow 和 model catalog 数据层
- `src/encoders/`：DAG、RSU 状态、flat semantic 和融合编码器
- `src/agents/`：主方法和对照算法 agent 接入
- `src/trainers/`：训练驱动协议
- `src/evaluators/`：benchmark、checkpoint 和真实 sample 辅助模块
- `scripts/`：数据检查、dry-run、训练、评估和 benchmark 脚本
- `artifacts/`：训练 checkpoint、benchmark 报告和论文表格产物
- `docs/project/`：长期项目文档入口

## 最小验证

```bash
python scripts/smoke_test.py
python -m pytest tests/test_env_contract.py
```

真实数据链路最小检查：

```bash
python scripts/run_ngsim_sample.py --max_rows 500
python scripts/run_alibaba_sample.py --limit_jobs 3 --min_tasks 5 --max_tasks 20
python scripts/run_real_sample_dryrun.py --mobility_source ngsim --workflow_source alibaba --max_mobility_rows 1500 --max_workflows 3 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_selector max_handoff_candidate --window_length 24 --max_steps 12
```

## 训练与评估

主方法训练：

```bash
python scripts/train_sa_ghmappo_real_sample.py --agent_name sa_ghmappo --profile formal_main --random_seed 7
```

对照算法训练：

```bash
python scripts/train_algo_pool_real_sample.py --agent_name ppo --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name dqn --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name dueling_dqn --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name controller_mat --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name dag_offload_drl --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name cache_offload_drl --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name dt_handoff_drl --profile smoke
```

对照算法评估：

```bash
python scripts/eval_algo_pool_real_sample.py --agent_name ppo --checkpoint_path artifacts/training/algo_pool/ppo/<run_id>/checkpoints/latest.pt
python scripts/eval_algo_pool_real_sample.py --agent_name reactive_greedy
```

## Benchmark

```bash
python scripts/benchmark_main_results.py --agents sa_ghmappo ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --sa_ghmappo_checkpoint_path <main_ckpt> --seed_checkpoint_manifest_path <manifest_with_learned_checkpoints> --seeds 7 13 29 --max_mobility_rows 2500 --max_workflows 2 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_selector max_handoff_candidate --window_length 24 --window_count 3 --window_scan_stride 2 --max_steps 12
```

`--flat_ppo_checkpoint_path` 和 `--flat_mappo_checkpoint_path` 是历史兼容参数名；当前 paper-grade 主表优先使用 seed checkpoint manifest 管理 learned baseline checkpoint。

## Baseline 闭环

```bash
python scripts/run_baseline_experiment.py --config configs/experiment/baseline/smoke.yaml
python scripts/run_baseline_experiment.py --config configs/experiment/baseline/minimal_ngsim_alibaba.yaml
```

核心输出包括：

- `comparison_summary.csv`
- `comparison_summary.json`
- `comparison_summary_detailed.json`
- `comparison_summary_by_window_class.csv`
- `run_manifest.json`
- `seed_checkpoint_manifest.json`
- `command_log.json`

Round1 状态、机制诊断和复跑命令见：

- `docs/experiment_status_round1.md`
- `docs/mechanism_activation_check_round1.md`
- `docs/experiment_runbook_round1.md`

## Top Journal Closed Loop

顶刊路线优先使用统一闭环入口，自动完成 SA-GHMAPPO、paper-grade learned baselines、seed checkpoint manifest、mixed/full benchmark 和 gate report：

```bash
python scripts/run_top_journal_closed_loop.py --quick --seeds 7 --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure
python scripts/run_top_journal_closed_loop.py --seeds 7 13 29 --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure
```

`--quick` 只验证链路，不用于论文结论；正式 claim 必须满足 `gate_report.json` 中的 `passed=true`、`formal_contract.ready=true` 和 `paper_claim_ready=true`。顶刊主线默认使用 `handoff_pressure` primary vehicle selection，让 NGSIM 窗口中的 handoff 压力进入 workflow 主体。

当前可引用的正式闭环产物：

- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_formal_20260505_v2/gate_report.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_formal_20260505_v2/gate_summary.csv`

Learned-baseline strict gate：

```bash
python scripts/run_top_journal_learned_baseline_suite.py --run_id <run_id> --base_manifest_path <seed_checkpoint_manifest.json> --skip_training --output_root artifacts/experiments/top_journal_sa_iteration
```

当前正式 learned-baseline 产物：

- `artifacts/experiments/top_journal_learned_baseline_suite/top_journal_learned_baseline_formal_20260505_v1/learned_baseline_gate_report.json`

当前默认 paper-grade learned-baseline set 为 `ppo` / `mappo` / `dqn` / `dueling_dqn` / `qmix` / `controller_mat` / `dag_offload_drl` / `cache_offload_drl` / `dt_handoff_drl`。`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl` 分别覆盖 DAG offloading、model/adapter cache offloading 和 Digital Twin handoff/service migration 领域对照。`ippo` 属于当前 contract-blocked diagnostic baseline；`ddqn` / `dueling_ddqn` 只有在 duplicate trace audit 通过时才能作为独立补充。复现旧 IPPO gate 必须显式使用 `--allow_contract_blocked_baselines`，且不能写成 paper-ready 结果。

`top_journal_mechanism_v3_eval_bias` 是基于 formal_v2 权重的 inference calibration 候选增强；生成入口为：

```bash
python scripts/build_top_journal_eval_bias_manifest.py --base_manifest_path artifacts/experiments/top_journal_learned_baseline_suite/top_journal_learned_baseline_formal_20260505_v1/seed_checkpoint_manifest_learned_baselines.json --output_root artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias --label v3_eval_bias
```

该候选不能在未补齐 holdout/support suite 前替代 formal_v2 paper-grade 主表。
## Current v3 Eval-Bias Candidate

Current guarded-prefetch refresh artifacts:

- formal gate: `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_gate_20260506/learned_baseline_gate_report.json`
- holdout gate: `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_holdout_offset3_20260506/learned_baseline_gate_report.json`
- latency fallback ablation: `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_support/statistics/latency_fallback_holdout_ablation_guarded_prefetch/paired_statistics.csv`

This is a strong inference-calibrated candidate, not a clean-retrain replacement. Prediction robustness still has an oracle-setting boundary, so do not claim universal superiority under oracle prediction.

## Current Learned-Baseline Expansion

The current top-journal paper-grade learned-baseline gate defaults to `ppo`, `mappo`, `dqn`, `dueling_dqn`, `qmix`, `controller_mat`, `dag_offload_drl`, `cache_offload_drl`, and `dt_handoff_drl`. `dag_offload_drl`, `cache_offload_drl`, and `dt_handoff_drl` are domain baselines for DAG offloading, model/adapter cache offloading, and Digital Twin handoff/service migration. `ippo` is diagnostic-only until a real independent per-agent wrapper/action contract is implemented; `mappo`, `qmix`, and `controller_mat` are controller-level CTDE/value-decomposition/transformer baselines, not vehicle-agent or RSU-agent full MARL wrappers. `ddqn` and `dueling_ddqn` are optional only if the duplicate-trace audit proves independence. `reactive_greedy` and `popularity_cache_heuristic` remain supplementary heuristic reference lines, not the primary publication gate.

Latest expanded artifacts:

- formal plus-dueling gate: `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_gate_20260506/learned_baseline_gate_report.json`
- holdout plus-dueling gate: `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_holdout_offset3_20260506/learned_baseline_gate_report.json`

## Current Final-Submission Loop

当前可交稿闭环入口为：

```bash
python scripts/run_top_journal_final_submission_loop.py --run_id <new_run_id> --base_manifest_path artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_formal_20260505_v2/seed_checkpoint_manifest.json --force_retrain_learned --resume_training --resume_benchmark --resume_support --command_retries 2 --baseline_episodes 96 --baseline_update_every 6 --baseline_batch_size 32 --minimum_reward_delta 0.5 --holdout_offsets 3
```

生成顶刊对比报告包：

```bash
python scripts/build_top_journal_comparison_report.py --final_run_root artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1
```

当前正式产物：

- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/final_submission_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/top_journal_comparison_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/top_journal_comparison_report.md`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/paper_ready/paper_ready_main_comparison.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/paper_ready/paper_ready_paired_reward_statistics.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/paper_ready/paper_ready_report.md`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/learned_suites/final_submission_v7_latency_fallback_20260528_v1_iter1_formal/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/learned_suites/final_submission_v7_latency_fallback_20260528_v1_iter1_holdout_offset3/learned_baseline_gate_report.json`

`final_submission_v7_latency_fallback_20260528_v1` 是 legacy paper-ready package，2026-06-18 rebuild 证明它可复现；但严格非重叠 holdout 审查已否决其当前 TMC-ready 状态。`final_submission_controller_mappo_qmix_20260509_v1`、`final_submission_full_current_baselines_20260511_v1` 和更早 package 只用于历史追溯。`mappo`、`qmix` 和 `controller_mat` 是 controller-level learned baselines，不应写成 vehicle-agent / RSU-agent full MARL wrappers；`popularity_cache_heuristic` 是 close supplementary reference。

## Typed MB Runtime Plumbing

训练、评估、benchmark与cache fairness现在共享`typed_model_cache_runtime_contract_v1.0.0`。Typed入口示例配置为`configs/benchmark/typed_model_cache_controlled_lru.yaml`（320 MB）和`typed_model_cache_controlled_lru_384mb.yaml`；legacy slot/MB配置继续可用。共享训练入口`train_algo_pool_real_sample.py`支持SA-GHMAPPO、PPO、MAPPO和domain cache/offload agent的显式runtime binding。

非正式端到端检查入口：

```bash
.venv/bin/python scripts/run_typed_model_cache_runtime_rehearsal.py
```

该入口只生成`non_formal_typed_runtime_rehearsal`，不冻结G14 split/protocol，不训练正式checkpoint，不运行formal、holdout、hidden或G15。合同与运行方法见`docs/project/typed_model_cache_runtime_contract.md`和`docs/project/typed_model_cache_runtime_validation_report.md`。
# Protocol 2.3 nullable formal metrics

正式 typed model-cache 执行合同现使用 Protocol 2.3。Protocol 2.2 的持续 subject lifecycle、外生 request、split、
window、catalog、训练预算与选择顺序保持冻结；新增 Nullable Metric Aggregation Contract 1.0.0，统一 train/eval、
Dev selection、checkpoint freeze、benchmark、statistics、gate 与 claim map 的 `float|null` 语义。唯一 active index：

```text
configs/experiment/typed_model_cache_formal_protocol_v2_3_20260903/protocol_index.json
```

G14C v12 `typed_model_cache_formal_20260902_162203_g14c_v12` 永久无效，run/root/staging/checkpoint 不得 resume、
retry、finalize、salvage、选择、冻结或进入 formal consumer。Protocol v1.0–v2.2 仅供审计。合同说明见
`docs/project/formal_nullable_metric_aggregation_contract.md`、
`docs/project/formal_request_subject_lifecycle_contract.md`、
`docs/project/formal_exogenous_request_execution_contract.md` 与
`docs/project/formal_environment_identity_projection_contract.md`。
