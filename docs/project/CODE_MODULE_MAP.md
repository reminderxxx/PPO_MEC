# Code Module Map

## G14R7 unified formal agent order

- `src/runtime/formal_agent_order.py`：versioned order contract加载、strict schema/canonical hash、reactive/
  learned/main/report-only/statistics角色解析及Protocol/scientific/fairness/template exact-order验证。
- `src/evaluators/cache_baseline_fairness.py`：`BASELINE_NAMES`来自order contract；baseline matrix与
  `controller_agents`均逐元素验证，不接受set equality。
- `scripts/run_typed_model_cache_formal_dev_selection.py`：只用resolver learned order构建candidate与nested
  benchmark；G14C v7路径硬拒绝。
- `scripts/manage_typed_model_cache_formal_artifacts.py`：candidate selection、freeze和seed/provenance companion
  按权威顺序输出并绑定order hash；v1–v7 invalid roots全部拒绝。
- `scripts/benchmark_main_results.py`、`src/evaluators/main_results_support.py`：15-agent argv exact gate、raw row
  稳定排序、aggregate/display顺序与checkpoint path拒绝。
- `scripts/analyze_top_journal_statistics.py`、`run_typed_model_cache_formal_statistics.py`：candidate/baseline
  配对身份、完整15-agent pair、重复/缺失检测、稳定key traversal与Holm顺序绑定。
- `src/runtime/formal_training_identity.py`、`formal_training_contract.py`、
  `resolved_formal_execution_context.py`：order semantic hash进入binding、context、training/checkpoint provenance。
- `scripts/repair_typed_model_cache_formal_agent_order.py`：生成Protocol v1.7/config/audit骨架；
  `run_typed_model_cache_formal_agent_order_acceptance.py`执行clean non-formal验收；finalizer生成Readiness v9。
- `tests/test_formal_agent_order_v17.py`：G14C v7复现、order/schema/role/consumer/hash/holdout 18类负例。

## G14R6 formal training identity split

- `src/runtime/formal_training_identity.py`：scientific config `2.0.0`、execution binding `1.0.0`、strict JSON、
  canonical hash、Protocol parity、runtime binding与checkpoint identity验证。
- `src/runtime/formal_training_contract.py`：v1.6同时消费scientific config/binding/context，legacy companion只供
  v1.1–v1.5审计，所有漂移在episode 0前拒绝。
- `src/runtime/resolved_formal_execution_context.py`：context `2.0.0`写入scientific config和binding SHA；
  host path仍只属于runtime audit/full context。
- `scripts/run_typed_model_cache_formal_protocol.py`：binding唯一producer；在Protocol hash、observed clean commit、
  environment和186-command matrix确定后生成binding，再生成resolved context。
- `scripts/train_algo_pool_real_sample.py`：真实训练入口的四件套解析、agent实例config审计、checkpoint/summary
  provenance与episode-0 contract-only rehearsal。
- `scripts/run_typed_model_cache_formal_dev_selection.py`、`manage_typed_model_cache_formal_artifacts.py`、
  `benchmark_main_results.py`：dev candidate、selection/freeze/provenance与formal checkpoint loading的binding消费端。
- `scripts/repair_typed_model_cache_formal_training_binding.py`：生成Protocol v1.6、config/index/schema与G14R6审计包；
  不训练、不运行formal/holdout。
- `tests/test_formal_training_identity_v16.py`：G14C v6复现、两层正例、18类负例、150-cell与Readiness v8回归。

## G14R2 formal window consumption and ledger v2

- `src/evaluators/formal_window_consumption.py`：冻结 window/source/loader identity、显式 raw-prefix scanner、
  plan/CLI binding、direct `RealMobilityBundle` loader、frame/time/provider/fingerprint 校验与 60-window
  reachability；sealed holdout 仅暴露 identity-only 模式。
- `src/evaluators/main_results_support.py`：formal/rehearsal 分支通过 contract+split+window ID 加载同一 bundle，
  不再对 frozen window 重新 rank/select。
- `scripts/train_algo_pool_real_sample.py` 与 `scripts/benchmark_main_results.py`：消费显式 contract/source/
  plan/split/mode，episode/checkpoint 前验证 range、selector、length、RSU、vehicle selection 与完整 split。
- `scripts/validate_formal_window_consumption.py`：`--validate-window-plan-only` 身份可达性入口；不构建 agent、
  不训练、不生成 performance result，holdout 只读 metadata。
- `scripts/repair_typed_model_cache_formal_windows.py`：生成 Protocol v1.2、window/command/ledger/restart/
  readiness/integrity 审计包；不启动 G14C v3。
- `scripts/run_typed_model_cache_window_rehearsal.py`：四 agent、两 seed、两 capacity 的 bounded non-formal
  frozen-train rehearsal，以及 train boundary tiny evaluation。
- `scripts/run_typed_model_cache_formal_dev_selection.py`、`run_typed_model_cache_formal_cache_policy.py`、
  `run_typed_model_cache_formal_support.py`：解析并向 benchmark 传递 frozen window contract；普通 runner
  不展开 holdout。
- `scripts/validate_typed_model_cache_formal_restart.py`：在 train 前执行真实 60-window loader reachability。
- `src/evaluators/typed_model_cache_formal_execution.py`：兼容 v1.1/v1.2 protocol validator、Readiness v4、
  failure classification 与 ledger `2.0.0` running/terminal hash chain。
- `src/runtime/formal_training_contract.py`：接受 v1.1/v1.2 formal training manifest，保持 cadence/config
  science contract 不变。
- `tests/test_formal_window_consumption.py`：逐窗口 60-case reachability，以及 source/binding/commands/
  protocol/ledger/rehearsal/readiness 专项回归。

## G14R executable formal protocol v1.1

- `src/runtime/formal_training_contract.py`：legacy/formal training budget resolver、checkpoint cadence/index/
  resume contract 与 instantiated agent-config audit；不拥有训练 loop。
- `scripts/train_algo_pool_real_sample.py`：消费 protocol v1.1/agent companion/cadence，保存 selection-eligible
  scheduled checkpoints 与 selection-ineligible `latest.pt`，写 resolved config/schedule/provenance。
- `src/metrics/cache_efficiency_metrics.py`：metrics 1.2 pure reducer；生产 byte-ready 与 typed
  transfer/request primary fields并执行 stored/raw reconciliation。
- `src/evaluators/typed_model_cache_formal_execution.py`：endpoint schema、support/scalability identity、
  command-matrix expansion、readiness v3 与 append-only phase ledger。
- `scripts/run_typed_model_cache_formal_dev_selection.py`：按 frozen dev fairness 评估全部 scheduled candidate
  updates，并执行 outcome-blind lexicographic selection。
- `scripts/manage_typed_model_cache_formal_artifacts.py`：selected checkpoint hash freeze、分 capacity
  seed/provenance manifests、integrity inventory 与 completeness-only formal gate。
- `scripts/run_typed_model_cache_formal_support.py`：统一 typed ablation/robustness/prediction/scalability/oracle
  setting consumer；setting ID 外无自由语义 override。
- `scripts/run_typed_model_cache_formal_cache_policy.py`：cache-policy benchmark 与 policy-neutral request replay
  的同 phase wrapper。
- `scripts/run_typed_model_cache_formal_statistics.py`：只收集 frozen controller rows 并调用预注册层级统计。
- `scripts/run_typed_model_cache_formal_protocol.py`：13 阶段 append-only runner；不含 holdout execution API。
- `scripts/restart_typed_model_cache_formal_protocol.py`：生成 protocol/config/Readiness v3/audit package；不训练。
- `scripts/run_typed_model_cache_formal_repair_rehearsal.py`：controlled non-hidden bounded rehearsal；不产正式证据。
- `tests/test_typed_model_cache_formal_execution.py`：48 类专项合同（参数化后 52 tests）。

## G14B historical exclusion、split 与 formal protocol

- `src/evaluators/typed_model_cache_formal_protocol.py`：metadata-only 历史 plan parser、完整 NGSIM interval inventory、raw frame/time/segment-run overlap、result-blind deterministic split、canonical/full/semantic hash、formal/statistics/claim/holdout/readiness 合同与 fail-fast validator；不执行算法或环境 episode。
- `scripts/freeze_typed_model_cache_formal_protocol.py`：create-only transactional freeze 入口；编排历史账本、I-80 candidates、24/12/12/12 split、agent/capacity/fairness/CLI preflight、seal/readiness 和 integrity manifest，不训练或生成 checkpoint。
- `src/evaluators/cache_baseline_fairness.py`：fairness window identity 绑定扩展为 raw frame/time 和 source segment/run，避免把 provider offset 冒充 raw interval；legacy optional 字段保持兼容。
- `configs/experiment/typed_model_cache_formal_protocol_v1_20260820/`：四个冻结 window plans 与 protocol index；sealed holdout 只允许 validator 读取 identity/interval。
- `tests/test_typed_model_cache_formal_protocol.py`：历史、overlap/gap、12+12、result-blind、hash、agent/budget/capacity/statistics/claim、CLI、seal/append-only/readiness/round-trip 专项合同测试。
- `docs/project/typed_model_cache_formal_protocol.md` 与 `typed_model_cache_split_exclusion_audit.md`：协议与证据边界的长期事实源。

## G12 causal calibrated predictor snapshots

- `src/predictors/calibration.py`：pure binary/multiclass/ETA/reliability/selective reducers，deterministic temperature fit、三段split interval audit与canonical hash。
- `src/predictors/causal_snapshot.py`：snapshot/calibration artifact版本、JSON-safe builder/consumer、causal/oracle/probability fail-fast validator和K=1/3/6/12 staleness诊断。
- `src/predictors/supervised_handoff_predictor.py`：保留旧runtime值，同时暴露完整raw logits/probabilities、semantic feature order/hash、normalization和availability。
- `src/envs/core/predictor_manager.py`：默认关闭的K-step慢快照、历史delay、expiry/abstention mask与显式fallback；不切换oracle。
- `src/envs/wrappers/gym_vec_env.py`、`src/metrics/recorder.py`：pre-action decision observation trace及step/episode snapshot provenance；`src/evaluators/main_results_support.py`把provenance投影到benchmark row。
- `scripts/audit_predictor_calibration.py`：稳定非训练入口；只用v71非hidden train/dev与v112 quality/checkpoint，生成G12完整audit bundle。
- `tests/test_causal_predictor_snapshot.py`：causality/probability/metrics/calibration/split/abstention/runtime/trace合同与负例。

## G11 public model-cache dataset registry

- `src/data/model_catalog/model_cache_dataset_registry.py`：纯 metadata validator/artifact projector；冻结 A–I taxonomy、19项 field availability、100分score、hard gates、compatibility检查、canonical JSON与SHA-256，不发起网络请求或下载。
- `configs/data/model_cache_dataset_registry.json`：19个候选的唯一 qualification 事实源；包含identity/access/fields/evidence/fitness/score/recommendation/mapping与online verification metadata。
- `scripts/validate_model_cache_dataset_registry.py`：加载 registry、`dataset_sources`、HF兼容manifest与legacy sample catalog，fail-fast后生成 deterministic G11 artifact。
- `scripts/check_data_ready.py`、`scripts/validate_dataset_source_declarations.py`：只检查 metadata readiness 和引用一致性；不把外部候选变成 runtime dataset。
- `src/data/model_catalog/adapter_catalog.py`：继续只承载环境所需 catalog schema；不负责外部来源qualification、license决策或下载。
- `tests/test_model_cache_dataset_registry.py`：覆盖taxonomy/score/hard-gate/compatibility/NaN/URL/date/determinism/legacy catalog/raw-download boundary。

## G10 information sufficiency and MARL necessity audit

- `src/analysis/information_sufficiency_audit.py`：纯只读 G10 reducer；冻结源码architecture facts、15项field map、trace leakage/alignment、recoverability、aliasing/projection、plug-in entropy/NMI/CMI和entity-level必要条件门禁。
- `scripts/audit_cache_information_sufficiency.py`：严格消费 G07/G08/G09 与可选 matched pre-action trace；默认拒绝覆盖，输出 resolved config、command log、synthetic validation与integrity manifest，不训练或修改checkpoint。
- `tests/test_information_sufficiency_audit.py`：覆盖single/factorized/controller-CTDE/entity MARL、field/index/normalization/leakage、recoverability、local/global aliasing、fixed projection、统计小样本、identity/integrity与determinism。
- `docs/project/cache_information_sufficiency_marl_audit_contract.md`：冻结controller-level/entity-level、GNN/GAT、prohibited claims和G11边界。

## G09 cache opportunity analyzer

- `src/analysis/cache_opportunity_analyzer.py`：纯 G09 reducer；严格校验 G07/G08/raw baseline identity，生成 demand/oracle/capture-loss request rows、taxonomy、fixed buckets、concentration、information labels和reconciliation。
- `scripts/analyze_cache_opportunities.py`：默认拒绝覆盖的稳定入口；只读取显式 raw artifact，输出 command log 与 integrity manifest，不执行训练或 benchmark。
- `tests/test_cache_opportunity_analyzer.py`：覆盖 demand、oracle、五 baseline、所有 primary reason、MB multi-victim、cross-RSU/handoff、identity/integrity、determinism与 JSON round-trip。

## G08 oracle

- `src/oracles/cache_request_replay.py`：policy-neutral replay schema、canonical fingerprint、G07/provider/workflow reconstruction和validation。
- `src/oracles/future_horizon_cache_oracle.py`：slot/MB exact rolling solver、capacity/action trace、optimality status、raw outcome alignment和matched gap；不依赖reward、agent或aggregate。
- `scripts/build_cache_request_replay.py`、`scripts/run_future_horizon_cache_oracle.py`、`scripts/audit_cache_oracle_gap.py`：稳定构建、运行和审计入口。

## 2026-08-18 G07 cache baseline fairness manifest

- `src/evaluators/cache_baseline_fairness.py`：canonical JSON/hash、schema builder/validator、dataset/window/workload/seed/cache/metric invariants、10组pairwise diff、runtime fingerprint/provenance enforcement。
- `scripts/build_cache_baseline_fairness_manifest.py`、`scripts/validate_cache_baseline_fairness_manifest.py`：显式构建与结构化验证入口；不下载数据、不运行formal/hidden。
- `scripts/benchmark_main_results.py`：显式消费已验证manifest，拒绝CLI覆盖冻结字段，核对observed request stream，并写resolved manifest、run/audit/integrity provenance。
- `src/evaluators/main_results_support.py`：raw row新增manifest ID/full/semantic hash、evaluation unit、expected/observed fingerprint；未传manifest保持`unavailable`兼容。
- `tests/test_cache_baseline_fairness_manifest.py`：覆盖schema、漂移、canonical hash、CLI/runtime、provenance、pairwise与legacy兼容。

## 2026-08-14 cache event independent reducer

- `src/metrics/cache_event_metrics.py`：纯 `CacheEvent` reducer、历史 summary 的 unavailable/empty 区分、schema/invariant validation 和 legacy telemetry structured comparison；reducer 不依赖 step/system/evaluator aggregate。
- `scripts/audit_cache_event_telemetry.py`：读取单个 episode `summary.json`，将 event-derived/legacy value、差值、mapping class 和解释写入独立 audit JSON，拒绝覆盖已有审计。
- `tests/test_cache_event_metrics.py`：覆盖人工精确值、denominator、来源/执行、admission/eviction/transfer/migration、非法 schema/event、历史兼容和 mapping mismatch。

## 2026-08-14 request-level cache event contract

- `src/envs/specs/semantic_objects.py`：冻结 `CacheEvent` v1 dataclass、required fields 与 event/object/hit-source 枚举，并执行 hit、admission、eviction 和 capacity-null 不变量。
- `src/envs/core/vec_workflow_core_env.py`：在真实 request/cache/handoff/execution 结果汇合处生产单 lifecycle event；cache action 同时保留 admission 前后 capacity 快照。
- `src/metrics/recorder.py`：检查 episode 内 `event_id` 唯一性，将 raw events 导出为 `cache_event_trace`；既有 `step_trace` 和 summary 字段保持不变。
- `docs/project/cache_event_contract.md`：记录 version、lifecycle、null/compatibility 边界和后续 byte capacity / LRU/LFU / oracle 消费方式。
- `tests/test_cache_event_contract.py`：覆盖 contract、场景、序列化和 request-denominator/invariant regression。

## 2026-08-11 v118-v121 planner-internalization boundary

- `src/agents/sa_ghmappo_core.py`：用 forked PyTorch RNG 初始化 optional transition ensemble，避免改变 policy 初始权重；实验性 teacher loss 支持 realized-GAE agreement gate 和 logit-margin projection，均默认关闭。
- `src/agents/sa_ghmappo_agent.py`：维护新增 checkpoint/config 字段白名单，确保训练与 evaluator restore contract 一致。
- `scripts/train_sa_ghmappo_real_sample.py`：维护 v118-v121 development profiles；这些 profile 已被 dual-domain matched evidence 拒绝，不属于 canonical v100。
- `tests/test_algo_pool_contract.py`：覆盖 learned-dynamics on/off 初始 policy 相等、model/realized advantage 双门控、低优势拒绝与 logit-margin 梯度路径。
- `docs/project/top_journal_v118_v121_negative_execution_20260811.md`：记录完整训练、双域 raw/planner 结果、统计边界与下一阶段 multi-entity MAPPO contract 要求。

## 2026-08-11 v100 LuST outcome-blind future validation

- `scripts/freeze_future_validation_split.py`：在保持 NGSIM 默认排除逻辑不变的前提下，支持 LuST trace 的 outcome-blind window freezing、历史 LuST plan 排除、source hash 记录和 frame/time/segment gap audit。
- `configs/experiment/top_journal_v100_lust_future_validation_20260810/`：保存 12-window sealed LuST plan 与 source manifest；它只定义验证窗口，不改变算法、reward 或 benchmark evaluator。
- `scripts/benchmark_main_results.py`、`scripts/analyze_top_journal_statistics.py`：执行冻结 v100 manifest 的 11-agent LuST full benchmark 和 outer-window hierarchical statistics。
- `scripts/benchmark_ablation.py`：支持 frozen-plan fast path、显式 zero-offset reward protocol 和 v100 no-online-planner inference attribution；该消融不替代 matched retraining ablation。

## 2026-08-09 v113-v117 MAPPO policy-iteration experiments

- `src/agents/sa_ghmappo_core.py`：保留 factorized counterfactual head loss、selective teacher-only label、training-only planner 和 conservative model-advantage gate；这些能力默认关闭，且未晋级 canonical v100。
- `src/trainers/marl_on_policy_trainer.py`：区分 online planner、training-only planner 和 teacher-only label，保证训练期辅助信号不会被误报为 native evaluation 行为。
- `src/evaluators/real_eval_support.py`：恢复新增 checkpoint config 字段，确保实验 profile 的 producer/consumer contract 对齐。
- `scripts/train_sa_ghmappo_real_sample.py`、`tests/test_algo_pool_contract.py`：维护 v113-v117 negative profiles 与 MAPPO action/loss contract regression tests。

## 2026-08-08 v97/v98 UCC counterfactual calibration and policy improvement

- `src/trainers/marl_on_policy_trainer.py`：在保持原 PPO rollout contract 的前提下，把 exact one-step branch 的真实 transition samples（含当前 MAPPO critic bootstrap）交给 UCC replay；v98 同时保留 exact action TD targets 供第二层 model policy improvement 使用。
- `src/agents/sa_ghmappo_core.py`：训练阶段可按 UCB、确定性评估按 LCB；v97/v98 通过 UCC calibration、MAPPO policy prior、action mask 和 margin 约束模型辅助动作，不修改 environment reward 或 evaluator filtering。
- `scripts/train_sa_ghmappo_real_sample.py`：维护 v95/v96 negative profiles 与 v97/v98 candidate profiles；v97/v98 的 formal winner status 仍待 multi-seed evidence。
- `tests/test_algo_pool_contract.py`：覆盖 UCB-train/LCB-eval、planned action propagation 和 counterfactual calibration sample contract。

## 2026-08-08 v94 uncertainty-calibrated model-assisted MAPPO

- `src/models/uncertainty_transition_ensemble.py`：负责 UCC-MAPPO 的 replay-backed bootstrap transition/TD-target ensemble、validation calibration、LCB prediction 和 checkpoint state；它只消费 rollout rows，不反向依赖 evaluator 或 benchmark。
- `src/trainers/ppo_buffer.py`、`src/trainers/marl_on_policy_trainer.py`：记录 `next_value`，为 model-assisted action improvement 提供 critic-bootstrap TD target，并维持原 PPO/SA rollout contract 的向后兼容默认值。
- `src/agents/sa_ghmappo_core.py`：将 UCC target 与 MAPPO policy prior、margin 和 PPO actor loss 连接；保存/恢复 ensemble、replay 和 protocol config。
- `scripts/train_sa_ghmappo_real_sample.py`、`scripts/run_top_journal_closed_loop.py`：维护 v94 profile、等预算 256-episode closed-loop、冻结 train/dev/formal plan 和 latest-first checkpoint selection。
- `tests/test_uncertainty_transition_ensemble.py`：覆盖 TD target、replay accumulation、uncertainty output 和 torch checkpoint serialization。
- `src/data/mobility/ngsim_provider.py`：提供 bounded-chunk pandas C-parser fast path 和原 csv fallback；两者输出同一 NGSIM segment/frame/VehicleState contract。
- `scripts/train_algo_pool_real_sample.py`：当传入冻结 window plan 时直接消费 plan，跳过不必要的全量窗口扫描；无冻结 plan 时仍保留原 outcome-blind candidate resolution。

## 2026-08-08 v93 mechanism-aware online counterfactual MAPPO

- `src/trainers/marl_on_policy_trainer.py`：在线反事实 branch replay 新增 delayed predictive-prefetch validation queue 和当前步 handoff-aligned prepare target，确保机制 credit 与 `EpisodeRecorder` 的兑现协议一致。
- `src/agents/sa_ghmappo_core.py`、`src/agents/sa_ghmappo_agent.py`：v93 保留 MAPPO centralized critic / controller credit，融合 robust counterfactual return advantage、mechanism advantage 和 policy-prior constrained online planner；不改变 reward/action contract。
- `src/evaluators/main_results_support.py`、`src/evaluators/real_eval_support.py`、`scripts/benchmark_main_results.py`：记录 planner、validated mechanism、offset-free reward 和 checkpoint provenance，支持同窗口 PPO/MAPPO/heuristic 比较。
- `scripts/train_sa_ghmappo_real_sample.py`、`scripts/run_top_journal_closed_loop.py`、`configs/experiment/top_journal_mechanism_v93_mechanism_aware_online_mappo.yaml`：维护 v71-v93 profile chain、训练预算与 paper-claim boundary；v93 当前仅为 dev candidate。
- `tests/test_algo_pool_contract.py`、`tests/test_checkpoint_compat.py`、`tests/test_strict_split_protocol.py`、`tests/test_top_journal_closed_loop.py`：覆盖 v93 contract、checkpoint compatibility、冻结窗口和 closed-loop 入口。

## 2026-07-22 v42-v46 offset-free MAPPO opportunity / net-utility profiles

- `src/envs/specs/semantic_objects.py`、`src/envs/core/vec_workflow_core_env.py`：`RewardBreakdown` 和环境 info 新增 `positive_offset` / `reward_positive_offset_component`，支撑 offset-free 与 legacy reward protocol 同时审计。
- `src/evaluators/main_results_support.py`、`scripts/benchmark_main_results.py`：主结果 rows/aggregate 新增 `reward_positive_offset`、`reward_positive_offset_component`、`offset_adjusted_total_reward`、`episode_step_count` 和 `reward_protocol`，benchmark 支持 `--reward_positive_offset 0.0`。
- `scripts/train_sa_ghmappo_real_sample.py`、`scripts/train_algo_pool_real_sample.py`：训练入口新增 `--reward_positive_offset` 并在 v42+ profile 默认置 0；新增 v42-v46 profile defaults 和 SA-GHMAPPO profile kwargs。
- `src/agents/sa_ghmappo_core.py`：v43 新增 strict opportunity delayed-credit 判定；v44 新增 opportunity-constrained actor logits，将 trusted handoff context、prediction reliability、prepare timing 和 cache readiness 写入 event/slow/fast/env-action logits；v46 复用已有 net-utility PRD / dual-cost credit 作为 MAPPO policy-side learning signal。
- `scripts/run_top_journal_closed_loop.py`：把 v42-v46 纳入 latest-first checkpoint selection、offset-free reward profile 和 strict full-budget override，避免 reward-first checkpoint 或正 offset 回退。
- `configs/experiment/top_journal_mechanism_v42_completion_aligned_mappo.yaml` 至 `configs/experiment/top_journal_mechanism_v46_net_utility_constrained_mappo.yaml`：记录各候选 profile、算法差异、训练预算和 paper-claim boundary。
- `tests/test_algo_pool_contract.py`、`tests/test_top_journal_closed_loop.py`：覆盖 v42-v46 profile 参数、offset-free effective setting、latest checkpoint selection 和 opportunity/net-utility contract。

## 2026-07-17 v18 counterfactual option credit and time-audited future split

- `src/agents/sa_ghmappo_core.py`：新增 option-gate counterfactual partial credit，计算 selected option partial utility 与同状态合法 option policy-weighted expected utility 的差值；保留 v17 DAG-aware option termination，不修改 action/reward/environment contract。
- `src/agents/sa_ghmappo_agent.py`、`src/evaluators/real_eval_support.py`：新增并恢复 `option_gate_counterfactual_prd_*` checkpoint/config 字段，保证 v18 训练和 benchmark inference contract 一致。
- `scripts/train_sa_ghmappo_real_sample.py`：新增 `top_journal_mechanism_v18_counterfactual_option` profile，继承 v17 并启用 counterfactual option credit。
- `scripts/run_top_journal_closed_loop.py`：把 v18 纳入 latest-first SA profile 和 strict-full dev budget override。
- `scripts/freeze_future_validation_split.py`：新增 outcome-blind future-validation split 生成器，v2 同时按 `frame_offset` 与 `time_index_start/end` 排除 train/dev/formal/hidden 历史窗口。
- `scripts/audit_window_independence.py`：窗口独立性审计从 frame-only 升级为 frame/time 双区间检查。
- `configs/experiment/top_journal_mechanism_v18_counterfactual_option.yaml`、`configs/experiment/top_journal_v17_future_validation_time_audited_20260717/`、`tests/test_algo_pool_contract.py`、`tests/test_top_journal_closed_loop.py`：记录 v18 profile、time-audited future split 和 contract tests。

## 2026-07-17 v13 partial-reward-decoupled MAPPO credit

- `src/agents/sa_ghmappo_core.py`：新增 event-head partial reward decoupling、option-gate partial credit、机制窗口 preserve-MAPPO 下的 option credit 计算，以及 checkpoint config 中的 `event_prd_advantage_*` / `option_gate_prd_*` 字段。
- `src/evaluators/real_eval_support.py`：恢复 v13 PRD credit config 字段，保证训练 checkpoint 与 benchmark inference contract 一致。
- `scripts/train_sa_ghmappo_real_sample.py`：新增 `top_journal_mechanism_v13_prd_option` profile，继承 v12 learned option 并启用 PRD event / option credit。
- `scripts/run_top_journal_closed_loop.py`：v13 使用 latest-first checkpoint priority，避免 `best_by_reward` 固定在 warm-start update 0；budget override 与 v12 strict-full dev screen 对齐。
- `configs/experiment/top_journal_mechanism_v13_prd_option.yaml`、`tests/test_algo_pool_contract.py`、`tests/test_top_journal_closed_loop.py`：记录 v13 PRD 参数、latest checkpoint selection、fallback selection 和 contract tests。

## 2026-07-17 v12 learned MAPPO option gate

- `src/agents/sa_ghmappo_core.py`：新增 policy-side `option_actor`、四类 option label、PPO-style option loss、entropy bonus、decayed contextual prior、idle/sparse popularity-safe prior、mechanism-window preserve-MAPPO 分支，以及 v11 checkpoint 缺少 option head 时的 partial warm-start load。
- `src/trainers/marl_on_policy_trainer.py`：把 `run_metadata.window_class` 传给 `agent.act()` / value evaluation，并在 rollout summary 中统计 option gate enabled/applied、label 和 selection reason。
- `src/evaluators/main_results_support.py`：把 option gate 诊断字段纳入主结果 metrics/rows；v12 checkpoint 不触发 v11 evaluator-side idle/sparse hard override。
- `src/evaluators/real_eval_support.py`：恢复 v12 option gate config 字段，保证训练 checkpoint 与 benchmark inference contract 一致。
- `scripts/train_sa_ghmappo_real_sample.py`：新增 `top_journal_mechanism_v12_learned_option` profile，继承 v11 MAPPO-core reward-first 设置，降低 imitation 牵引并启用 learned contextual option gate。
- `scripts/run_top_journal_closed_loop.py`：把 v12 纳入 reward-first profile set 和 strict-full dev budget override。
- `configs/experiment/top_journal_mechanism_v12_learned_option.yaml`、`tests/test_algo_pool_contract.py`、`tests/test_top_journal_closed_loop.py`：记录 v12 config、warm-start contract、mechanism-window preserve 行为、reward checkpoint selection 和 contract tests。

## 2026-07-16 v11 MAPPO reward-first candidate

- `src/agents/sa_ghmappo_core.py`：新增 `idle_popularity_fallback_*` 与 no-RSU local fallback 可选开关；默认只在 v11 inference 中对 deterministic `vehicle_fallback` 做 popularity candidate replacement，no-RSU local 由 evaluator 的 idle/sparse window gate 控制。
- `src/evaluators/real_eval_support.py`：从 checkpoint config 和相邻 `train_summary.json` 恢复 v11 profile，并为旧 v11 checkpoint 注入 reward-first inference defaults，保证已训练 checkpoint 可复现新评估行为。
- `src/evaluators/main_results_support.py`：新增 `build_window_context_agent_overrides()`；当 checkpoint profile 为 `top_journal_mechanism_v11_mappo_reward` 且 benchmark `window_class=idle_or_sparse` 时，评估端打开 no-RSU local fallback，机制窗口不启用该 override。
- `scripts/train_sa_ghmappo_real_sample.py`：新增 `top_journal_mechanism_v11_mappo_reward` profile，继承 v8 strict scaffold 并迁入 MAPPO head-credit / entropy floors / event advantage blend；训练预算为 128 episodes、20 train windows。
- `scripts/run_top_journal_closed_loop.py`：v11 使用 reward-first checkpoint priority，`best_by_reward_path` 先于 tiebreak/continuity 字段；closed-loop budget override 与 strict-full dev screen 对齐。
- `configs/experiment/top_journal_mechanism_v11_mappo_reward.yaml`、`tests/test_algo_pool_contract.py`、`tests/test_top_journal_closed_loop.py`：记录 v11 profile、window gate 语义和 contract tests。

## 2026-07-13 v8 support suite and v9 Pareto-safe candidate

- `scripts/run_strict_full_v8_support_suite.py`：统一运行 v8-current prediction robustness、system robustness、scalability 和 guard attribution，并生成 `support_gate_report.json`；脚本只接受非 hidden window plan。
- `scripts/benchmark_prediction_robustness.py`、`scripts/benchmark_robustness.py`、`scripts/benchmark_scalability.py`、`scripts/benchmark_ablation.py`：支持 `--window_plan_path`，让 support benchmark 消费冻结 strict split。
- `src/evaluators/real_eval_support.py`、`src/evaluators/main_results_support.py`：新增 `agent_config_overrides` 评估端覆盖，用于同一 checkpoint 下的 guard attribution，不改变 checkpoint 文件、reward 或 action schema。
- `scripts/train_sa_ghmappo_real_sample.py`：新增 `top_journal_mechanism_v9_pareto_safe` profile，并在 checkpoint selection 中输出 `best_by_pareto_safe_score.pt`；该 ranking 显式惩罚 handoff failure、backhaul 和 continuity regression。
- `scripts/run_top_journal_closed_loop.py`：新增 v9 budget override，保持 strict-full 训练预算与 v8 可比。
- `configs/experiment/top_journal_mechanism_v9_pareto_safe.yaml`、`configs/ablation_checkpoint_manifest_v8_guard_attribution.json`：分别记录 v9 候选边界和 v8 guard attribution manifest。

## 2026-07-06 supervised handoff predictor v1

- `src/predictors/supervised_handoff_predictor.py`：定义薄 MLP predictor、冻结 feature schema、checkpoint schema 和 runtime loader；只输出短时 next-RSU / handoff-target / ETA / confidence prediction。
- `scripts/train_supervised_handoff_predictor.py`：从冻结 train/dev window plan 构建 mobility future-label 样本，训练 predictor checkpoint 并写出 metrics manifest / quality rows；不读取 reward、action 或 checkpoint outcome。
- `src/envs/core/predictor_manager.py`：新增 `predictor_kind=supervised` 和 `predictor_checkpoint_path`，将 supervised predictor 输出映射回现有 `predictions` contract；缺失 checkpoint、schema 或 RSU map 不匹配时 fail fast。
- `scripts/train_sa_ghmappo_real_sample.py`、`scripts/benchmark_main_results.py`、`scripts/benchmark_prediction_robustness.py`：接收 supervised predictor checkpoint，支撑 SA-GHMAPPO v9 重训、主结果 benchmark 和 prediction robustness 五组设置。

## 2026-06-21 strict-full v8 protocol and analysis

- `scripts/freeze_strict_split_protocol.py`：只按 mobility covariate 分层，冻结 train/dev/formal/hidden 计划、源数据 hash 与 interval independence audit。
- `src/evaluators/main_results_support.py`、`scripts/benchmark_main_results.py`：读取显式 `--window_plan_path`，保证 benchmark 消费冻结窗口而非重新扫描选择。
- `scripts/run_top_journal_closed_loop.py`、`scripts/train_sa_ghmappo_real_sample.py`、`scripts/train_algo_pool_real_sample.py`：把 train/eval window plan 传入主方法和 baseline 训练链。
- `scripts/analyze_top_journal_statistics.py`：window outer、seed/workflow inner hierarchical bootstrap，输出 percentile/BCa CI、effect size、sign test 与 Holm correction。
- `scripts/analyze_strict_full_failure_modes.py`：只允许非 hidden 标签，按窗口/action/reward component 诊断 failure mode。
- `src/agents/sa_ghmappo_core.py`、`src/evaluators/real_eval_support.py`：实现并恢复 v8 steady-RSU soft bias；只在 current adapter warm 且无 distinct handoff 时生效，不修改 reward/action schema。
- `configs/experiment/top_journal_mechanism_v8_strict_full.yaml`：冻结 v8 profile、两轮 dev 上限、split/统计协议与 promotion boundary。
- `tests/test_top_journal_statistics.py`、`tests/test_strict_split_protocol.py`、`tests/test_strict_full_failure_modes.py`、`tests/test_algo_pool_contract.py`：覆盖统计层级、split hash/间隔、hidden 标签禁用和 v8 checkpoint contract。
- `scripts/audit_literature_reference_table.py`、`tests/test_literature_reference_audit.py`：解析六列文献表，归一化标题/DOI/URL，报告重复、无效链接结构与待核验条目。

## 2026-05-28 SA v7 latency fallback clean-retrain profile

- `scripts/train_sa_ghmappo_real_sample.py`：新增 `top_journal_mechanism_v7_latency_fallback` profile；继承 v6 guards，并启用 `latency_fallback_bias_*` / `latency_fallback_slow_suppression_strength`，用于 clean retrain 而非旧 eval-bias 复用。
- `scripts/run_top_journal_closed_loop.py`：将 v7 纳入 formal budget override，默认 `sa_episodes=128`、`train_window_count=6`。
- `configs/experiment/top_journal_mechanism_v7_latency_fallback.yaml`：记录 v7 训练、closed-loop、final-submission 和 promotion gate 参数；不修改 reward、action schema 或 baseline contract。
- `tests/test_algo_pool_contract.py`、`tests/test_top_journal_closed_loop.py`：覆盖 v7 profile 参数和 closed-loop budget。

## 2026-05-27 SA confidence-aware prefetch admission guard

- `src/agents/sa_ghmappo_core.py`：新增 `predictive_prefetch_admission_guard_*`，在低置信度且 next-RSU / prefetch target 未对齐时把 selected predictive prefetch 延期为 event prepare；默认关闭，v6 profile 显式开启。
- `src/agents/sa_ghmappo_agent.py`、`scripts/train_sa_ghmappo_real_sample.py`、`src/evaluators/real_eval_support.py`：同步维护该字段的构造参数、profile 默认值、checkpoint config、训练 summary 和 benchmark 恢复路径。
- `src/trainers/marl_on_policy_trainer.py`、`src/evaluators/main_results_support.py`：新增 `predictive_prefetch_admission_guard_count/rate` 诊断消费，避免 formal benchmark rows 丢失 guard 触发计数。
- `configs/experiment/top_journal_mechanism_v6_strong_competition.yaml` 和 `configs/algo/*.yaml`：v6 记录 admission guard 参数；learned/domain baselines 继续声明排除该 SA-only guard。

## 2026-05-27 SA freshness-aware prefetch guard

- `src/agents/sa_ghmappo_core.py`：`cache_warm_start_guard` 新增 `cache_warm_start_guard_max_prefetch_countdown`，用于把 target-adapter prefetch 限制在 freshness window 内；默认 `0.0` 保持历史无上界行为。
- `src/agents/sa_ghmappo_agent.py`、`scripts/train_sa_ghmappo_real_sample.py`、`src/evaluators/real_eval_support.py`：共同维护该字段的训练 profile、checkpoint config 和 benchmark 恢复路径。
- `configs/experiment/top_journal_mechanism_v6_strong_competition.yaml`：v6 profile 显式设置上界 `6.0`，与 `EpisodeRecorder(prefetch_validation_window=6)` 对齐；该机制属于 policy guard，不修改环境 reward 或 `semantic_discrete_5` schema。

## 2026-05-12 SA-GHMAPPO contract notes

- `src/envs/specs/action_schema.py`：维护 `semantic_discrete_5` action schema、precondition mask、invalid reason 和 `ActionAdapter` 到 `ControlAction` 的转换；`build_mask_info()` 是 wrapper/policy/report 消费 action legality 的来源。
- `src/envs/specs/semantic_objects.py`：`ControlAction.metadata` 承载 action id/name、invalid action 和 invalid reason，不改变 cache/offload/migration 三个语义动作主体。
- `src/envs/core/predictor_manager.py`：统一输出 `baseline`、`oracle`、`learned_or_calibrated`、`supervised`、`no_prediction` 的 predictor kind 和 runtime audit；当前默认仍不是 learned predictor，只有显式 supervised checkpoint 才设置 `learned_predictor_attached=true`。
- `src/envs/core/vec_workflow_core_env.py`：在 `metrics_protocol` 汇总 predictor audit proxy、DAG frontier/critical-path pressure、mechanism success gate 和 action invalid 字段。
- `src/agents/sa_ghmappo_core.py`：主算法 action info 同时记录 raw head action、mask projection 后 action、guard 后 final action 和 guard delta。
- `src/trainers/marl_on_policy_trainer.py`、`scripts/train_sa_ghmappo_real_sample.py`、`src/evaluators/main_results_support.py`、`scripts/benchmark_main_results.py`、`scripts/build_top_journal_comparison_report.py`：消费并汇总 action projection、invalid attempt、DAG diagnostics 和 mechanism validated success gate。

## 核心链路

- `src/data/mobility/`：读取或回放车辆轨迹，生成车辆状态、RSU 关联和 handoff 事件。
- `src/data/workflow/`：生成 toy DAG 或解析 Alibaba workflow，输出环境可消费的 DAG 结构。
- `src/data/model_catalog/`：描述车载 base model、路侧 adapter cache、state bundle 和外部 model-cache metadata source。
- `src/envs/`：消费 mobility、workflow 和 catalog，执行 cache/offload/migration 动作，输出状态、奖励和 continuity 信息。
- `src/envs/specs/action_schema.py`：维护语义动作 schema、mask 和 `ControlAction` 适配。
- `src/agents/`：agent 基类、算法文件和注册表。当前规则是只按算法分文件，不再保留 `baselines/`、`marl/` 或 PPO family 包装目录。
- `src/encoders/`：为主方法和 baseline 提供 DAG、RSU 状态、flat semantic 和融合编码器。
- `src/trainers/`：负责 on-policy 训练循环、buffer 和 checkpoint 写出。
- `src/evaluators/`：负责 checkpoint 选择、真实 sample 支持、主结果和 benchmark 聚合。
- `src/metrics/`：负责 episode 记录、指标 reducer 和论文指标。

## Agent 组织

`src/agents/registry.py` 直接从算法文件导入并注册：

- `sa_ghmappo` -> `src/agents/sa_ghmappo_agent.py`
- `ippo` -> `src/agents/ippo_agent.py`
- `ppo` -> `src/agents/ppo_agent.py`
- `mappo` -> `src/agents/mappo_agent.py`
- `qmix` -> `src/agents/qmix_agent.py`
- `controller_mat` -> `src/agents/mat_agent.py`
- `dag_offload_drl` -> `src/agents/dag_offload_agent.py`
- `cache_offload_drl` -> `src/agents/cache_offload_agent.py`
- `dt_handoff_drl` -> `src/agents/dt_handoff_agent.py`
- `dqn` / `ddqn` / `dueling_dqn` / `dueling_ddqn` -> `src/agents/dqn_agent.py`
- `reactive_greedy` -> `src/agents/reactive_greedy_agent.py`
- `popularity_cache_heuristic` -> `src/agents/popularity_cache_heuristic_agent.py`

2026-05-27 MAPPO protocol update：
- `src/agents/mappo_agent.py` 当前负责 controller-level CTDE MAPPO + `aggregation_reason_weighted_controller_ppo_v3` controller head-credit。
- `src/agents/sa_ghmappo_core.py` 承载通用 controller head credit floors / entropy floors / entropy scales；默认仍兼容旧 v2 行为，`mappo` 显式启用 v3。
- `scripts/train_algo_pool_real_sample.py` 提供 `mappo_strong_audit` profile。
- `scripts/run_top_journal_learned_baseline_suite.py`、`scripts/run_top_journal_final_submission_loop.py`、`scripts/run_top_journal_closed_loop.py` 和 `scripts/build_top_journal_comparison_report.py` 负责审计 `mappo` checkpoint protocol，避免 pre-v3/pre-head-credit MAPPO 进入新版主表。
- `src/evaluators/real_eval_support.py` 在恢复 `mappo` checkpoint 时保留 v3 head-credit 相关 config 字段。

`flat_ppo` / `flat_mappo` 只表示历史 artifact run 名称，不再作为 live agent 注册。

公共核心：

- `src/agents/sa_ghmappo_core.py` 保留主方法共享的 on-policy rollout、checkpoint、flat policy 网络等实现。它不是算法 family 包装层，不能重新承载 PPO/MAPPO 注册。

## 脚本入口分组

- 数据检查：`check_data_ready.py`、`validate_dataset_source_declarations.py`、`audit_hf_model_cache_sources.py`、`run_ngsim_sample.py`、`run_alibaba_sample.py`
- 窗口与 dry-run：`scan_ngsim_handoff_windows.py`、`run_real_sample_dryrun.py`
- 最小联调：`smoke_test.py`、`run_toy_episode.py`、`benchmark_toy_runs.py`
- 主方法训练：`train_sa_ghmappo_real_sample.py`
- 对照算法训练：`train_algo_pool_real_sample.py`
- Baseline 闭环：`run_baseline_experiment.py`
- 主方法评估：`eval_sa_ghmappo_real_sample.py`、`run_checkpoint_sweep.py`
- 对照算法评估：`eval_algo_pool_real_sample.py`
- 主结果：`benchmark_main_results.py`
- 消融和压力测试：`benchmark_ablation.py`、`benchmark_prediction_robustness.py`、`benchmark_robustness.py`、`benchmark_scalability.py`
- 论文导出：`export_paper_artifacts.py`

## 依赖方向

数据层和环境层是下游算法的基础。训练、评估和 benchmark 可以依赖环境、agent、metrics，但环境不应反向依赖具体训练脚本。

任何输出字段、manifest、checkpoint 或路径变化，都要同步检查 `scripts/` 生产者、`src/evaluators/` 消费者和 `configs/` manifest。

## 2026-05-04 算法 contract 更新

- `src/encoders/fusion_encoder.py` 的 `FlatSemanticEncoder` 同时输出 actor/local `shared_embedding` 和 MAPPO centralized critic 使用的 `centralized_critic_context`。
- `src/agents/mappo_agent.py` 是当前 controller-level CTDE MAPPO baseline：flat semantic encoder + cache / execution-offload / handoff-event 三个 controller actors + centralized flat semantic critic；它不是 vehicle-agent / RSU-agent full MAPPO wrapper。
- `src/agents/qmix_agent.py` 是当前 controller-level QMIX baseline：flat semantic encoder + cache / execution-offload / handoff-event controller Q heads + centralized monotonic mixer；它不是 vehicle-agent / RSU-agent full QMIX wrapper。
- `src/agents/mat_agent.py` 是当前 controller-level MAT-style transformer baseline：flat semantic encoder + 三个 controller tokens + centralized transformer critic；它不是 vehicle-agent / RSU-agent full MAT wrapper。
- `src/agents/dag_offload_agent.py` 是 dependency-aware DAG offloading 领域对照：flat semantic encoder + DAG progress/frontier/critical-path/node-IO scalar block + controller-level centralized critic；它不使用主算法 DAG graph message passing。
- `src/agents/cache_offload_agent.py` 是 model/adapter cache + offloading 领域对照：flat semantic encoder + cache occupancy、adapter readiness、cache demand 和 future-load scalar block；它不使用主算法 surrogate/guard 机制。
- `src/agents/dt_handoff_agent.py` 是 Digital Twin handoff/service migration 领域对照：flat semantic encoder + raw DT prediction sequence、dwell time、confidence、future-load 和 boundary-pressure scalar block；它不使用主算法 calibrated surrogate gate 或 uncertainty-aware event scaling。
- `src/agents/sa_ghmappo_core.py` 仍是共享 on-policy core，但 PPO/IPPO 与 MAPPO 的 actor/action contract 已分离；MAPPO 使用层级三头 actor，`centralized_critic=True` 时消费 `centralized_critic_context`。
- learned policy 的 flat action distribution 在 `act()` 和 `learn()` 中应用 `decision_info["action_mask"]`，mask audit 字段随 `action_info` 写入 rollout。

## 2026-05-05 Top Journal Closed Loop

- `scripts/run_top_journal_closed_loop.py` 统一调度 SA-GHMAPPO、PPO/MAPPO/QMIX/DQN-family baseline 训练、seed checkpoint manifest、benchmark 和 gate report，并支持 `--resume_training` 复用已完成 seed checkpoint。
- `src/envs/core/vec_workflow_core_env.py` 提供 `primary_vehicle_selection`，顶刊闭环使用 `handoff_pressure` 把 handoff 压力绑定到 workflow 主 vehicle。
- `src/agents/sa_ghmappo_core.py` 新增机制辅助 current-cache-fill 解耦、backhaul guard 和 cache-warm start guard；这些逻辑属于 agent policy，不修改环境语义。
- `src/evaluators/real_eval_support.py` 负责从 checkpoint config 恢复 backhaul/cache-warm guard 与机制辅助开关，保证训练/benchmark contract 一致。
- `src/trainers/marl_on_policy_trainer.py` 与 `src/evaluators/main_results_support.py` 消费并汇总 cache-warm guard 诊断字段。
- `scripts/benchmark_main_results.py` 继续作为正式主表 benchmark 消费端，并兼容 UTF-8 BOM manifest。

## 2026-05-05 Top Journal Support Suite

- `src/evaluators/main_results_support.py` 现在提供 seed checkpoint manifest helper，支撑 benchmark 消费端按 seed 选择 checkpoint。
- `scripts/benchmark_prediction_robustness.py`、`scripts/benchmark_robustness.py`、`scripts/benchmark_scalability.py` 和 `scripts/benchmark_ablation.py` 支持 `--seed_checkpoint_manifest_path` / `--primary_vehicle_selection`，用于和 formal 主表保持 checkpoint 与 handoff-pressure contract 一致。
- `scripts/run_top_journal_ablation_training.py` 负责训练 current-contract SA-GHMAPPO ablation variants 并生成 per-seed ablation manifest。
- `scripts/analyze_top_journal_statistics.py` 负责从 rows 生成 paired bootstrap CI、win/tie/loss 和 sign-test 摘要。
- `scripts/build_top_journal_support_gate_report.py` 汇总主 gate、支撑实验完成状态、关键统计和 claim warning。
- `scripts/export_paper_artifacts.py` 负责 formal_v2 baseline-aware paper table / claim summary 导出。
- `scripts/train_sa_ghmappo_real_sample.py` 的 checkpoint consistency audit 默认只审计 latest/warm_start/best 系列；`update_*.pt` 全量审计需要显式 `--audit_update_checkpoints`。

## 2026-05-05 Mechanism v3 Iteration

- `src/agents/sa_ghmappo_core.py` 新增 latency fallback inference calibration 字段：在当前 adapter 已 warm 且无跨 RSU / handoff 预测时，可压低 slow cache/prefetch/event heads，使 fast head 的 `vehicle_fallback` 在低风险状态生效；机制窗口不触发该抑制。
- `src/evaluators/real_eval_support.py` 从 checkpoint config 恢复 latency fallback calibration 字段，保证 derived checkpoint manifest 在 benchmark 中可复现。
- `scripts/train_sa_ghmappo_real_sample.py` 新增 `top_journal_mechanism_v3` profile；当前 clean retrain 结果未冻结为主结果。
- `scripts/build_top_journal_eval_bias_manifest.py` 从已有 seed checkpoint manifest 派生启用 latency fallback calibration 的 SA checkpoint manifest，用于候选验证。
## 2026-05-06 Holdout And Eval-Bias Support

- `src/evaluators/main_results_support.py` 的 `resolve_window_candidates()` 支持 ranked offset、formal interval exclusion、minimum gap 和 greedy non-overlap selection；独立 holdout 必须使用 interval 约束，不能只使用 offset。
- `scripts/audit_window_independence.py` 对两个 aggregate summary 的 selected window plans 做 split 内/跨 split interval 审计。
- `src/evaluators/real_sample_support.py` 的 `auto_grid_tight` 为 LuST 等二维 mobility 建立 RSU grid；一维 NGSIM 仍可使用 `auto_dominant_tight`。
- `scripts/benchmark_main_results.py`、`scripts/run_top_journal_learned_baseline_suite.py` 和 `scripts/benchmark_ablation.py` 消费 `--window_rank_offset`。
- `src/agents/sa_ghmappo_core.py` 增加 predictive prepare hard override 配置字段；默认关闭，只用于派生候选筛选。continuity guard 现在在 target cache ready 前不压制 prefetch。
- `src/evaluators/real_eval_support.py` 从 checkpoint config 恢复新增的 predictive prepare override 字段。
- `scripts/build_top_journal_eval_bias_manifest.py` 可写入 predictive prepare override 和 cache warm countdown 配置；`scripts/build_top_journal_eval_bias_ablation_manifest.py` 生成 v3 latency fallback 消融 manifest。
## 2026-05-06 Final Submission Loop

- `scripts/run_top_journal_final_submission_loop.py` 负责最终交稿 learned-primary 闭环编排：formal learned suite、offset holdout、prediction robustness、system robustness、scalability、final gate report 和断点续跑。
- `scripts/run_top_journal_learned_baseline_suite.py` 负责 learned baseline 等预算训练、manifest 增强、mixed/full benchmark、cluster bootstrap statistics、duplicate trace audit 和 learned-only gate。
- `scripts/build_top_journal_comparison_report.py` 负责从 final gate 和 rows/aggregate artifacts 生成顶刊对比包：baseline protocol matrix、reward margins、main paired statistics、support setting-level paired statistics、paper-ready LaTeX/CSV 表格、copy-ready result statement、self-review 和 markdown/json report。
- `src/agents/ippo_agent.py` 当前仅保留 diagnostic/contract-blocked agent；不能作为 paper-grade learned baseline，除非后续先实现真实 independent per-agent wrapper/action contract。`src/agents/mappo_agent.py`、`src/agents/qmix_agent.py` 与 `src/agents/mat_agent.py` 已接入为 controller-level paper-grade learned baselines，但不支持 full vehicle-agent / RSU-agent MAPPO/QMIX/MAT 声明。`src/agents/dag_offload_agent.py`、`src/agents/cache_offload_agent.py`、`src/agents/dt_handoff_agent.py` 已接入为主线领域专项 learned baselines，需通过 final-submission loop 后才能引用正式数值。
- `scripts/analyze_top_journal_statistics.py` 支持 `--cluster_keys`，当前 final loop 使用 `seed window_id workflow_id` 作为 total_reward cluster bootstrap unit。
- `src/data/mobility/ngsim_provider.py` 返回 loaded frames 时使用显式 `VehicleState` 字段复制，避免长实验中通用 `deepcopy` 在 Python 3.14 下偶发崩溃。
# 2026-08-17 Cache capacity dependency update

- `src/data/model_catalog/adapter_catalog.py` 提供唯一 resident adapter MB resolver，区分 explicit CacheObject 与 64 MB legacy fallback。
- `src/envs/core/cache_eviction.py` 负责通用 eviction lifecycle、只读 `EvictionPlan`、LRU 1.0 状态/排序和仅注册 `lru` 的 fail-fast factory；不写 cache/catalog，也不承载 RL agent。
- `src/envs/core/vec_workflow_core_env.py` 负责 slot/MB normalization、size/oversized/required-free、plan 验证、initial/runtime 原子 mutation 和 snapshots；通过 factory 依赖 policy，不再直接维护 LRU 排序状态。
- `src/envs/specs/semantic_objects.py`、`src/metrics/cache_event_metrics.py` 负责 CacheEvent 1.1 multi-victim 生产、1.0/1.x 兼容消费与 legacy 对账。

# 2026-08-18 classical cache baseline modules

- `src/envs/core/cache_eviction.py`：五种 eviction policy、统一 plan/state factory。
- `src/agents/classical_cache_agent.py`：共享 reactive control 的身份薄层；不拥有 eviction state。
- `src/agents/registry.py`：完整 baseline spec 与 identity-policy binding validator。
- `scripts/validate_classical_cache_baselines.py`：matched controlled benchmark 与诊断 artifact。
## 2026-08-18 G06 cache efficiency metrics

- `src/metrics/cache_efficiency_metrics.py`：pure CacheEvent/trace-context reducer，冻结 request/byte、lifecycle/churn、capacity、pollution、future-reuse proxy 和 latency-unavailable contract。
- `src/envs/specs/semantic_objects.py`：CacheEvent 1.2 optional admission identity/size 与逐 victim sizes；1.0/1.1 consumer compatibility 保留。
- `src/envs/core/vec_workflow_core_env.py`、`src/envs/wrappers/gym_vec_env.py`、`src/metrics/recorder.py`：生产版本化 initial/final per-RSU cache snapshot，不将环境对象泄露到 metrics 层。
- `src/evaluators/main_results_support.py`：仅接入 G06 nullable scalar；大型 raw trace 仍由 episode summary 承载。
- `scripts/audit_cache_efficiency_metrics.py`：读取真实 summary 并输出 JSON-safe 独立重算结果。
- `tests/test_cache_efficiency_metrics.py`：覆盖 byte denominator、multi-victim、pollution/censoring、reuse horizon、invalid/missing/null 与 aggregate semantics。

## 2026-08-19 G13 type-aware model cache

- `src/data/model_catalog/adapter_catalog.py`：同时承载 legacy adapter-only projection 与冻结的 typed catalog 校验、fingerprint、依赖/兼容解析；不下载或加载模型权重。
- `src/envs/core/vec_workflow_core_env.py`：拥有 RSU typed residency、分层 readiness、base→adapter 单动作原子事务、容量守恒和 dependency-safe mutation；workflow state 只记录迁移 readiness，不计 resident cache 容量。
- `src/envs/core/cache_eviction.py`：继续只生成只读 victim plan；typed 环境负责过滤 pinned/non-evictable/dependency-blocked 对象并验证完整 plan 后一次提交。
- `src/envs/specs/semantic_objects.py`：冻结 `CacheEvent 1.3.0` 的 typed 可选字段，同时兼容消费 1.0–1.2。
- `src/metrics/cache_event_metrics.py`、`src/metrics/cache_efficiency_metrics.py`：从 raw event 独立重算 layered readiness、逐类型 transfer/residency/admission/eviction、base reuse、orphan/churn；legacy 缺失项保持 unavailable/null。
- `src/oracles/cache_request_replay.py`、`src/oracles/typed_model_cache_oracle.py`、`src/oracles/future_horizon_cache_oracle.py`：冻结 typed replay 与小规模精确 atomic oracle；legacy oracle 路径不变。
- `src/evaluators/cache_baseline_fairness.py`：校验 catalog/profile/初始状态/事务/指标/oracle 的 policy-invariant binding。
- `scripts/validate_typed_model_cache.py`：只运行受控确定性验证与真实 NGSIM + Alibaba 最小链路，不训练、不运行 formal/holdout/hidden。

## 2026-08-19 G14A typed formal runtime plumbing

- `src/runtime/typed_model_cache_runtime.py`：config/fairness/training/checkpoint/benchmark共享resolver、runtime hash、catalog consumer check与三态checkpoint provenance gate；不拥有环境cache mutation。
- `scripts/train_algo_pool_real_sample.py`：统一learned training入口，显式消费runtime catalog/MB profile并写summary/checkpoint provenance；支持SA-GHMAPPO、PPO、MAPPO和domain cache/offload baseline。
- `scripts/eval_algo_pool_real_sample.py`、`scripts/benchmark_main_results.py`：消费同一resolved runtime；typed learned evaluation执行checkpoint gate；benchmark支持legacy slot、legacy MB、typed MB并把raw event留在episode summary。
- `src/evaluators/cache_baseline_fairness.py`：typed manifest 1.1 producer/validator及runtime CLI enforcement；五reactive pairwise matrix不被controller companion改变。
- `scripts/run_typed_model_cache_runtime_rehearsal.py`：只生成non-formal tiny training/restore/benchmark/reconciliation与integrity证据。
- `tests/test_typed_runtime_plumbing.py`：G14A配置、fairness、training、checkpoint、legacy、event、metrics、hash/JSON专项合同测试。

## 2026-08-24 G14R3 portable formal resources

- `src/runtime/portable_resource_identity.py`：portable contract、registry、scientific fingerprint、六类 root resolver、
  conflict/symlink/content audit 与共享 CLI binding。
- `src/evaluators/cache_baseline_fairness.py`、`formal_window_consumption.py`：content-identical relocation、legacy
  exact-path compatibility 与 portable companion enforcement。
- `scripts/run_typed_model_cache_formal_dev_selection.py`：explicit workflow propagation、真实 dev candidate evaluation
  与受控 non-formal rehearsal matrix。
- `scripts/manage_typed_model_cache_formal_artifacts.py`：path-invariant selection、portable checkpoint freeze/companions、
  invalid G14C v3 hard reject、integrity 与 non-formal completeness gate。
- `scripts/repair_typed_model_cache_formal_paths.py`：Protocol v1.3/config/evidence/readiness generator、186-command parity 与
  12 negative cases。
- `scripts/run_typed_model_cache_formal_path_rehearsal.py`：16-cell public controlled exact phase-chain rehearsal；不具备
  formal/holdout/hidden capability。
- `tests/test_portable_resource_identity.py`：portable identity、resolver、parser、checkpoint、command/readiness regressions。

## G14R4+ execution transaction modules

- `src/runtime/formal_execution_environment.py`：Python resolver、dependency/environment fingerprint、clean-worktree
  import probe、child environment parity 与 runtime-location audit。
- `src/evaluators/formal_phase_transaction.py`：phase ledger v3、monotonic timing、completion candidate、immutable terminal
  commit 与 finalize-only。
- `src/evaluators/formal_cell_transaction.py`：cell ledger v1、stable cell/episode ID、attempt staging、artifact inventory、
  atomic marker/commit、same-run resume 与 committed-only matrix validation。
- `scripts/run_typed_model_cache_formal_protocol.py`：v1.4 resolver 与 transactional phase入口；legacy v1.1-v1.3 继续
  保留旧 runner 兼容但不能使用 finalize-only。
- `scripts/manage_typed_model_cache_formal_artifacts.py`：checkpoint freeze 同时硬拒绝 G14C v3 与两个永久无效
  G14C v4 run root，旧 checkpoint 不可进入新 freeze manifest。
- `scripts/repair_typed_model_cache_formal_execution.py`：Protocol v1.4/config/root-cause/artifact/Readiness v6 generator。
- `scripts/run_typed_model_cache_formal_execution_rehearsal.py`：no-.venv import、8/16、75/150 与 finalize-only 非正式审计。
- `tests/test_typed_model_cache_formal_execution_v14.py`：environment/timing/cell/dev-formal/protocol 事务专项回归。

## 2026-08-25 G14R5 resolved context modules

- `src/runtime/resolved_formal_execution_context.py`：resolved context `1.0.0` canonical hash、finite/path/identity校验、
  atomic create-only、same-run load与nested Python parity。
- `scripts/run_typed_model_cache_formal_protocol.py`：v1.5唯一 context producer；fresh/dry-run/resume/finalize共用完整
  resolution，context/file SHA进入input、run identity和phase ledger；v1.0–v1.4 active execution关闭。
- `scripts/validate_typed_model_cache_formal_restart.py`：加载 outer context，重展开全部templates并比对186-command hash，
  同时执行11,850,526-row/60-window metadata-only reachability。
- `scripts/run_typed_model_cache_formal_dev_selection.py`、`run_typed_model_cache_formal_support.py`、
  `run_typed_model_cache_formal_statistics.py`：nested subprocess只消费context内绝对Python。
- `scripts/repair_typed_model_cache_formal_preflight_context.py`：Protocol v1.5/config、G14C v5登记、producer/consumer
  matrix、G14R5 artifact与Readiness v7 generator。
- `tests/test_typed_model_cache_formal_execution_v15.py`：outer/nested parity、context tamper/drift/fallback、invalid-run和
  holdout边界专项回归。
