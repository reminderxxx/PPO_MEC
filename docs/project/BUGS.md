# Bugs And Risks

用途：记录当前有效问题、风险和禁止误读项。

## 2026-08-18: G05 unavailable/null aggregation regression resolved

- `FIXED / nullable aggregate`: benchmark aggregate 曾把 `None`/missing 通过通用 default-zero 聚合成 `0.0`，违反 G03 Cache Capacity Contract；现全 unavailable group 保持 JSON `null`，mixed group 只统计 available finite numeric values并记录 available/unavailable counts。
- `FIXED / consumer compatibility`: pairwise、win/tie/loss、robustness、checkpoint sweep 与 transaction summary 已能读取 nullable mean；不可用值不参与 delta 或 best-value 排名。
- `BOUNDARY`: 该修复不追溯改写历史 artifact，也不新增 G06 byte-hit/pollution/regret/latency-saved 指标；历史 capacity-disabled artifact 仍不能支持 cache-efficiency claim。

## 2026-08-14: cache observability contract resolved; metric/baseline work remains open

- `RESOLVED / event observability`: request-level `CacheEvent` v1 now records lookup, mutually exclusive hit source, admission, eviction victim, transfer/migration, capacity before/after and execution result in raw episode summaries.
- `OPEN / derived metrics`: byte hit, pollution, eviction regret and latency-saved metrics are not implemented in this task; v1 events only provide their future raw input contract.
- `OPEN / algorithm comparison`: byte capacity, capacity-matched LRU/LFU/FIFO/Random and future-horizon oracle remain absent. Capacity-disabled historical artifacts still cannot support cache-efficiency claims.

## 2026-08-11: planner distillation is not a stable native-policy improvement

- `FIXED / RNG fairness`: transition-ensemble initialization previously shifted the global PyTorch RNG before policy construction. Historical `no_learned_dynamics` retraining results are confounded and must not be cited causally. New enabled/disabled runs share identical initial policy parameters.
- `REJECTED / v118 full`: conservative planner distillation regressed RNG-aligned v100 raw reward on LuST by `-2.2450` and NGSIM by `-2.4081`; NGSIM BCa `[-6.0801,-0.2678]`. Planner execution masked most of the loss but did not improve v100.
- `REJECTED / early checkpoint`: v118 update8 improved only seed7 on LuST and catastrophically reversed on NGSIM. Fixed shorter training or checkpoint selection is not a valid recovery route.
- `REJECTED / v119-v121`: realized-GAE gating removed the negative tail but produced no material deterministic policy change. Stronger coefficients and logit-margin projection remained effectively tied to v100 in three-seed dual-domain probes.
- `BLOCKER / contract`: PPO and MAPPO currently share one global semantic observation and one environment action; MAPPO only factorizes the controller heads and centralizes the critic. A large generic MAPPO-over-PPO claim is structurally unsupported until a vehicle/RSU-level decentralized observation/action contract is frozen and benchmarked.
- `OPEN / selection bias`: NGSIM formal and LuST future outcomes were consumed repeatedly during v118-v121 development. These profiles cannot be promoted on the same splits; any new architecture requires an untouched frozen holdout.

## 2026-08-09: v100 reward lead confirmed on independent future split, mechanism dominance unresolved

- `RESOLVED / cross-split reward`: frozen v100 SA-GHMAPPO reached `33.342` versus Popularity `29.350` on the one-time v20 future-validation package; reward BCa CI was fully positive and SA won `75/15/0` paired comparisons.
- `OPEN / metric trade-off`: SA's mechanism realization (`0.600`) is above Popularity (`0.567`) on this split but not uniformly above all learned baselines; backhaul and migration costs remain trade-offs. Do not compress the result into “all metrics improved.”
- `OPEN / paper readiness`: one mobility/workflow combination, complete component ablations and unified compute accounting remain absent. The independent future split strengthens evidence but does not make the package TMC-ready.

## 2026-08-10: LuST support is positive but not an independent generalization claim

- `RESOLVED / external direction`: v100 wins on the available LuST support package (`34.200` vs Popularity `27.215`, MAPPO `28.946`).
- `OPEN / support power`: only 4 outer windows are available and the historical plan metadata is `outcome_blind_selection=false`; do not use this as primary cross-city evidence or pool it with the NGSIM future rows.
- `OPEN / workflow scope`: LuST support still uses Alibaba workflows; a second workflow source remains unavailable in the current data root.

## 2026-08-11: LuST independent split confirms reward lead but exposes regime trade-off

- `RESOLVED / cross-mobility reward`: the new outcome-blind 12-window LuST split gives SA `-25.638` versus Popularity `-32.961`, with BCa reward CI `[+2.324,+15.186]` and paired `24/48/0`.
- `OPEN / regime coverage`: all methods are negative in the aggregate; SA's lead concentrates in mechanism-activating windows (`31.218` vs Popularity `13.644`), while active/idle strata remain negative. The algorithm must not be described as uniformly robust across regimes.
- `OPEN / paper readiness`: component ablations and unified compute accounting remain incomplete even after cross-mobility evidence was added.

## 2026-08-11: inference ablation contract fixed and planner attribution recorded

- `FIXED / ablation reward protocol`: `scripts/benchmark_ablation.py` previously inherited a `+5.0` reward offset by omission; the invalid diagnostic is excluded, and the corrected runner defaults to explicit zero offset.
- `RESOLVED / planner attribution`: full v100 beats the no-online-planner inference variant by `+2.084722` with a positive BCa interval on all 12 LuST windows.
- `OPEN / training ablation`: this is an inference-side component ablation. Matched retraining ablations for every loss/constraint component are still required for a full causal training claim.

## 2026-08-09: v113-v117 native-policy internalization boundary

- `REJECTED / factorized target`: v113's joint-to-head marginal target is a valid MAPPO training loss but destabilized the event head and reduced reward; it is not enabled by the canonical v100 profile.
- `REJECTED / hard teacher distillation`: v114 generated exact model-improved labels with positive support, but its rollout behavior and native PPO returns were misaligned. Nonzero teacher support must not be interpreted as policy improvement.
- `REJECTED / training-only planner`: v115-v117 kept planner execution out of evaluation, but the native policy did not internalize the online planner advantage within the tested budget. Any future result must report training behavior and native evaluation separately.
- `OPEN / metric interpretation`: v100 formal uplift was `+0.3475`, while the one-time independent v20 future split produced `+3.992`; the larger cross-split reward gap is now established for the frozen checkpoint, but mechanism/cost metrics are not uniformly dominant. No reward-only selection or runtime wrapper may be used to manufacture an advantage.

## 2026-08-09: learned predictor integration remains blocked

- `BLOCKED / predictor-policy mismatch`: the supervised predictor has strong next-RSU classification but low handoff probability under the current class imbalance. Feeding its low-confidence targets into the v100 planner caused `mean_total_reward=16.366` and mechanism collapse on the dev probe. Do not enable `predictor_kind=supervised` in the canonical v100 run until a separately frozen calibration and policy-gating protocol is designed.
- `FIXED / variable RSU slots`: v71 windows expose different RSU slot counts. The predictor now accepts a runtime RSU subset of the checkpoint slot map and pads newly observed slots during train-sample construction. This is an input-contract fix, not evidence of algorithmic gain.
- `FIXED / threshold-selection complexity`: supervised predictor threshold selection previously rebuilt predictions for every candidate score, creating quadratic behavior on full mobility plans. It now uses one sorted cumulative scan.

## 2026-08-09: CAMA follow-up did not widen the reward gap

- `OPEN / candidate performance`: native CAMA v102/v103 remains below Popularity on the dev protocol (`18.4101`/`18.4093` vs `21.1033`). Do not claim that counterfactual head credit has already improved end-to-end reward.
- `OPEN / planner dependence`: v100's positive margin is produced by the agent-side online counterfactual planner; CAMA head distillation did not reproduce that margin when planner execution was disabled. Any future claim must report planner-enabled and native-policy ablations separately.
- `RESOLVED / negative stress test`: strong CAMA target distillation was tested and rejected after mean reward `-15.038` and continuity `0.256`. A non-triggered `collapse_detected=false` flag is not evidence of safe learning; the raw probe remains a negative boundary condition.
- `OPEN / paper evidence`: v101-v105 have no untouched independent hidden result, complete component ablation package or unified compute audit. They remain development evidence, not paper-ready proof.

## 2026-08-08: v100 paper-readiness risks remain

- `BLOCKER`: v100 has no untouched hidden holdout. The historical v71 hidden split was consumed by v98 and must not be reused for v100 tuning or claimed as v100 evidence.
- `BLOCKER`: only the NGSIM + Alibaba controller-level combination is covered. Cross-mobility, cross-workflow and larger system-scale validation are absent.
- `MAJOR`: formal mechanism realization, continuity, handoff-ready ratio and backhaul/migration outcomes tie Popularity; the reward margin is mainly delay-based and small (`+0.3475`).
- `MAJOR`: v100 and v98 formal reward rows are identical, so the v100 formal package is a replication/winner confirmation rather than proof of a new formal uplift.
- `MAJOR`: full noise sweep, complete component ablations and unified wall-clock/compute accounting are not yet available. Compact prediction support is support-only and must not replace these packages.

## 2026-08-08 v98 hidden 已消费后的剩余风险

- `RESOLVED / formal-hidden independence`: `scripts/audit_window_independence.py` 对 v71 formal 与 hidden 计划通过，20+20 windows 的 frame/time/segment intervals 均无重叠；hidden 已一次性消费，不能再用于调参或换 checkpoint。
- `RESOLVED / reward main claim`: v98 full formal 与 hidden 均相对 Popularity 为正，且 window-outer BCa CI 完全高于 0；结果不能解释为所有机制指标全面领先，因为 continuity、handoff-ready、mechanism realization 与 Popularity 持平。
- `RESOLVED / policy-improvement attribution`: v97 calibration-only 三 seed full formal 未超过 Popularity；v98 第二层 exact one-step policy improvement 后超过，v97 collapse seed29 未被筛除。
- `OPEN / support scope`: prediction robustness 已完成 5-window frozen support subset；未完成 full 20-window noise sweep、system robustness 和 scalability 的当前 v98 package，不能宣称完整 support suite 已闭环。
- `OPEN / compute audit`: 训练/评估包含 exact branch samples、UCC ensemble 和 online planner；当前 artifact 有训练 summary 与 update count，但尚未形成统一 wall-clock/peak-memory/branch-cost table。
- `OPEN / metric tradeoff`: v98 的 reward 胜出主要来自 total_reward；与 Popularity 的机制 realization、ready、continuity 持平，backhaul/migration 等指标必须随主表一起报告，不能只给 reward。

## 2026-08-08 v98 candidate risks

- `OPEN / multi-seed evidence`: v98 目前只有 seed7、48 episode 的 formal probe，不能与 v94/v70 的 multi-seed full result 等量齐观；必须完成 seeds `[7,13,29]`、256 episodes、formal aggregate 和 paired statistics。
- `OPEN / holdout freeze`: hidden holdout 尚未消费。v98 formal candidate、checkpoint manifest、stats protocol 和 claim 必须先冻结，再做一次性 holdout；不得根据 holdout 结果继续调参。
- `OPEN / compute cost`: v97/v98 每个训练 step 使用 five-action one-step branch calibration，训练 wall-clock 高于 v94；最终必须记录 branch transition count、runtime 和 checkpoint provenance，并做 compute-aware comparison。
- `OPEN / ablation`: v98 的增益尚未拆分为 UCC calibration、exact policy improvement、policy prior 和 no-model/no-calibration ablations；在补齐前不能声称每个组件独立显著贡献。

## 2026-08-08 v94 UCC-MAPPO 开放风险

- `OPEN / evidence pending`: UCC-MAPPO 的正式多 seed benchmark 尚未完成；当前不能把首个 seed、真实 smoke 或历史 v93 开发结果写成全量优胜结论。
- `OPEN / calibration`: ensemble uncertainty 在当前真实小跑中出现较高 validation error 和 clipped calibration scale；正式结果必须报告 calibration、no-uncertainty、no-policy-prior、no-model ablation，不能只报告 reward。
- `OPEN / contract scope`: 该 model 是 controller-level action-conditioned TD surrogate，不是 vehicle-level / RSU-level multi-agent world model；论文中不能扩大 MARL 或 digital-twin claim。
- `OPEN / protocol`: hidden holdout 仍 sealed；formal candidate、checkpoint manifest、statistics 和主 claim 冻结前不得开启 hidden，也不得用 formal 结果继续调参。
- `RESOLVED / data-loading`: baseline 长跑曾由纯 Python `csv.DictReader` 解析 5M-row NGSIM 造成首个 episode 长时间无输出；`NGSIMProvider` 现用 bounded-chunk pandas C-parser 并保留 fallback，已用 segment contract、真实 1500-row sample 和 5M-row read benchmark 验证。
- `RESOLVED / frozen-plan startup`: `train_algo_pool_real_sample.py` 在冻结训练计划存在时曾重复执行全量窗口扫描；现直接消费冻结 plan，与 SA 入口保持一致。低行数 smoke 若覆盖不到冻结 plan 的 frame offset 会明确报错，不能用缩短数据集伪造正式协议。
- `RESOLVED / evaluation checkpoint contract`: 首轮 v94 benchmark 的 SA evaluator 白名单漏掉 UCC ensemble/planner 配置，导致恢复时 planner 默认关闭；已补齐字段、补充 checkpoint compatibility regression test，并用真实 checkpoint 验证 model state/replay 恢复。首轮 aggregate 降级为 diagnostic，需用同一 frozen manifest 重跑 benchmark。
- `RESOLVED / learned action propagation`: evaluator 修复后发现 planner 的 `applied` 只写入 action metadata，未同步更新 `env.step()` 使用的 action；已修复并用 fake environment 验证 planned action 到达环境。此前 repaired-evaluator aggregate 仍只能作为诊断，需再次重跑。

## 2026-08-08: v93 开发集已领先但仍有训练预算与顶刊证据 blocker（OPEN）

- v93 在两个独立 seed 的同窗口 zero-offset full-stratified dev benchmark 中均高于 PPO、controller-level MAPPO 和 Popularity；seed13 为 `17.901250` vs `16.947000`，seed7 为 `18.068000` vs `16.947000`。这只能作为开发集事实，不能替代 formal/holdout 统计。
- 当前最大 blocker 是训练预算不对称：v93 seed13 checkpoint 仅 16 episodes，seed7 的独立训练在 episode19 / update2 后因 branch `deepcopy` 反事实计算过慢中止；PPO/MAPPO checkpoint 为 96 episodes。后续 paper comparison 必须统一 episode/update budget，或给出预注册的 compute-matched protocol。
- v93 仍未完成未消费 formal/hidden、window-outer hierarchical CI、paired sign test/Holm、prediction/system robustness、scalability、完整 ablation package、command log 和 artifact integrity manifest。按 `tmc_review_policy_v3_20260621` 当前 verdict 仍为 `Not TMC-ready` / evidence below `E2_ARTIFACT_AUDITED`。
- v93 机制目标已修复 delayed prefetch validation 和当前步 handoff alignment，但机制指标仍不是所有 strata 全面领先；完整结果必须同时报告 validated hit、handoff ready、continuity、backhaul 和失败率，不能只报告 total reward。
- v93 在线 planner 需要每个候选动作做 digital-twin branch replay，训练计算成本高。后续应先做不改变 objective 的 branch reuse / state snapshot 优化，并记录 wall-clock、branch count 和 peak memory；不能通过减少候选或缩短 rollout 后把结果称为等预算。

## 2026-07-29: v70 formal-min all-baseline reward winner 仍非 paper-ready（OPEN）

- v70 `top_journal_mechanism_v70_sparse_tail_option_mappo` 已在当前 formal-min mixed/full 全量 benchmark 中让 SA-GHMAPPO total reward 排名第一。full_stratified artifact 为 `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v70_sparse_tail_option_formal_min_20260730/benchmarks/full_stratified_config_loaded/main_results_full_stratified_20260730_010523_184238/aggregate_summary.json`：SA `32.385729` > DT `31.426667` > popularity `29.969271` > PPO `27.301597` > MAPPO `16.261458`。
- 当前最关键正向证据是 full-only window-outer hierarchical statistics：`artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v70_sparse_tail_option_formal_min_20260730/statistics/full_stratified_hierarchical/paired_statistics.json`。SA vs DT total reward delta 为 `+0.959063`，95% CI `[0.554171, 1.691468]`；SA vs popularity/PPO/MAPPO 的 reward CI 也均为正。
- 不得把 v70 写成 paper-ready / TMC-ready：本轮是 formal-min benchmark + statistics，缺独立 hidden/future holdout、support suite、artifact integrity/command-log package、ablation、prediction/system robustness、scalability 和完整 readiness audit，证据等级低于 `E2_ARTIFACT_AUDITED`。
- 不得声称所有系统指标全面优于规则。v70 相对 DT 的 reward 与 mechanism readiness/realization 为正，但 workflow continuity delta 微弱为负且 CI 触及 0；backhaul cost 相对 DT 略高，SA vs DT backhaul signed benefit 为 `-0.444444`，95% CI `[-1.515039, 0.0]`，Holm p=`0.07812`。相对 popularity 的 reward/continuity/readiness/realization 为正，但 backhaul cost 仍更高：raw delta `+1.5`。
- v70 的安全表述边界：提升来自 policy-side sparse-tail option prior 修复 `idle_or_sparse` option boundary，使该 strata reward 从 v67/v69 的 `27.1015` 提到 `30.7365` 并反超 DT `29.29275`；不是 reward shaping、environment/action schema、baseline contract 或 evaluator filtering。后续若冲击投稿，需要冻结未消费 holdout 并补 support/ablation，不得继续在已查看 formal-min 结果上反复调参后再把它当独立验证。

## 2026-07-27: v55 coverage-recovery dev evidence 与后处理审计风险（OPEN）

- v52/v53 已确认存在策略路径接入问题：net-advantage prepare gate 没有进入 `SAGHMAPPOBaseAgent` 覆盖后的 `_apply_policy_adjustments`。本轮已在真实 policy path 中修复，并用单元测试覆盖；历史 v52/v53 结果不得解释为 gate 已实际改善主策略。
- v54 虽修复 gate 路径并加入 service-completion gate，但 no-current-RSU 覆盖缺口仍大量选择 action2 vehicle fallback，导致主算法低于 MAPPO。v55 的 coverage-recovery MAPPO credit/option candidate/guard 在三组 full dev 训练中把 no-current action 稳定转为 `{4:1064}`，这是当前 reward 提升的主要机制证据。
- v55 仍不是 paper-ready：`artifacts/training/top_journal_v55_coverage_recovery_full_dev_summary/sa_ghmappo/` 已补齐三 seed `train_summary.json`，full-dev mean reward `9.102656`，高于同协议 PPO/MAPPO dev 对照，但缺 formal/holdout/support suite、窗口外层统计、完整 manifest/command log 和 paper-grade full checkpoint consistency audit。禁止写成“足以发论文”“TMC-ready”或“显著高于全部算法”。
- 训练脚本后处理风险已部分修复：`scripts/train_sa_ghmappo_real_sample.py` 新增 compact post-training audit，v55 默认先生成完整 `train_summary.json` 并把 compact scope 明确标为非 paper-grade；compact 审计不会修复 best checkpoint record，也不能作为 checkpoint family 的最终一致性证据。剩余风险仍然存在：正式 `E2` / paper-ready package 必须补跑 `--post_training_audit_mode full` 或独立 full checkpoint consistency audit，并生成完整 manifest / command log / formal-support package。

## 2026-07-27: v47--v51 policy-attribution blocker（OPEN；门禁已实现）

- `top_journal_mechanism_v47`--`v51` 目前只有 dev-stage single-seed training evidence，且各 update 的 deterministic evaluation 指标不变。v51 update 1--16 的 reward `15.358`、continuity `0.706148`、handoff-ready `0.45`、mechanism realization `0.525` 相同；`best_by_reward` 位于 update 1。不能把 checkpoint selection、physical-transfer feature 或 reward 数值解释为 MAPPO 学习改进。
- v51 diagnostics 的 `deterministic_event_prepare_rate_on_valid_target=1.0`、`event_margin_mean=5.998730`、`guard_action_delta_rate=0.684751` 表明当前推理路径混合了 learned logits 和 runtime policy interventions。任何主 claim 必须先比较 raw-policy 与 safety-projected-policy；环境 action mask 可以保留，heuristic logit bias/guard/option replacement 的贡献必须作为独立机制消融报告。
- Policy-Learning Gate 已实现：raw learned policy 与 safety-projected policy 分开评估，自动 checkpoint selection/consistency audit 固定使用 raw-policy metrics，且 raw action/metrics invariant 的 run 被拒绝选择。这只消除了“看不见策略是否改变”的测量缺口，不证明 v47--v51 有效。
- 在完成 multi-seed dev ablation 和新的 frozen formal/holdout 协议前，v47--v51 仍为 `Unverifiable`，不得晋级 canonical、paper-ready 或 TMC-ready。

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

- 2026-07-22 v46 是当前 offset-free dev full-pool 最强候选，但仍不是 paper-ready。`top_journal_mechanism_v46_net_utility_constrained_mappo` 在 20 outer windows / 40 paired rows / seed7 / full_stratified 上得到 SA-GHMAPPO `38.791`，高于 PPO `38.3375` 与 `popularity_cache_heuristic=38.0`；相对 popularity 的 reward BCa CI 为 `[0.18, 2.239]`，但 Holm p=`0.210924`，相对 PPO 的 CI 为 `[-0.09125, 2.330993]` 且 Holm p=`1.0`。禁止把它写成显著高于 PPO、显著高于全部 baseline 或 TMC-ready。
- v46 的 `sa_advantage_diagnosis` 仍有 `backhaul_cost_above_popularity` blocker：SA backhaul `170.8` vs popularity `170.0`，mechanism realization 与 readiness 均与 popularity 持平，reward gap 主要来自少数窗口 reward / migration-overhead 侧收益。论文表述必须同时报告 backhaul、migration overhead、mechanism success gate 和 PPO trade-off，不能只报告 total reward。
- v42-v46 是 offset-free protocol 修复后的 dev-probe 线索：`reward_positive_offset=0.0` 已进入训练、benchmark 和 aggregate reporting，但只有 seed7 的 full-pool dev evidence。任何 formal/holdout/support 结论必须重新训练或冻结 manifest 后按 top-journal protocol 运行，不得把 v46 dev statistics 当 final package。
- 2026-07-21 v39/v40/v41 dev-probe 不能证明“主算法高于全部算法”。当前最强 SA full-pool 复核是 v39 update_0005：SA-GHMAPPO `106.041`，高于 MAPPO `105.5875`、popularity `105.25` 和 PPO `94.77375`，但低于 `cache_offload_drl=119.14875` 与 `dt_handoff_drl=119.22625`。v41 conservative recovery 为 `105.686`，同样只略高于 MAPPO/popularity，未超过 DT/cache。禁止将 v39/v41 写成 all-baseline winner 或 paper-ready 结果。
- 历史 v39/v41 的 `VecWorkflowCoreEnv.reward_positive_offset=5.0` 是已确认 artifact risk：它按 step 累加，导致未完成但拖到更长 horizon 的策略可能比更快完成 workflow 的策略得到更高 `total_reward`。v42+ 已用 `reward_positive_offset=0.0` 修正该 ranking 协议；旧 v39/v41 只能作为目标不一致诊断，不能再作为论文主 reward ranking。
- `top_journal_mechanism_v40_advantage_weighted_behavior_mappo` 是负向算法探索：train-window update 5/6 reward 很高，但 frozen dev targeted benchmark 只有 `85.788`，说明 positive-deviation advantage-weighted behavior cloning 过拟合窗口行为。它只能作为 AWR/AWAC-style behavior regularization 的失败消融，不能晋级。
- `top_journal_mechanism_v41_conservative_recovery_mappo` 修复了 v40 的严重 dev collapse，但没有扩大 MAPPO gap：SA `105.686` vs MAPPO `105.5875`，差距只有 `+0.0985`；对 popularity 差距 `+0.436`，对 DT 差距 `-13.54025`。它可以作为 conservative recovery ablation，不是论文主候选。
- checkpoint audit best-source fallback 已修复，但历史 v40/v41 修复前生成的 `best_by_reward` 记录需要看 `source_checkpoint_path` 和 `expected_best_sources`，不要仅凭 `best_by_reward.pt` 文件名判断是否真来自最佳 update。正式重跑前应保留 `tests/test_sa_checkpoint_repair.py` 覆盖。
- 2026-07-19 v23-v26 reward-gap 扩大实验没有形成比 v22 更强的主结果。v23 counterfactual constrained PRD 在 dev 上低于 popularity；v24 tail-risk PRD 只取得 dev `+0.195`、formal `+0.05625` 的 popularity reward delta 且 CI 跨 0；v25 opportunity PRD 退化到 dev `+0.100`；v26 safe-counterfactual PRD 为 dev `+0.16425`，仍低于 v22 `+0.268`。这些 profile 只能作为负向探索和机制诊断，不能写成 canonical 晋级或 paper-ready 改进。
- 当前 strict/full dev 和 time-audited formal 中，`popularity_cache_heuristic` 是非常强的 supplementary rule reference；formal split 上 SA、popularity、PPO、MAPPO 的 `mechanism_realization_rate` 均为 `0.0`，因此很难从 handoff/cache mechanism 上拉开大 reward gap。若继续要求显著高于 popularity，需要新冻结更能覆盖 validated mechanism opportunities 且未消费的 formal/hidden split，不能在已查看 formal 上继续筛 profile。
- v23-v26 的失败说明“增加机制尝试”不是单调有效：v23 会产生 failed mechanism tail loss，v25/v26 降低 validated mechanism realization 或压低 v22 的正窗口收益。后续算法设计必须先提供可检验的一阶机制假设，例如可学习 counterfactual evaluator、offline safe policy improvement 或更强 predictor，而不是单纯调大 PRD/auxiliary 系数。
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

# 2026-08-14: cache observability / baseline / oracle blockers

- `BLOCKER / baseline identity`: live registry 没有独立 LRU、LFU、FIFO、Random cache agents；环境中的 LRU 仅是 capacity-enabled admission 时的 eviction primitive，不能当成已评估 baseline。
- `BLOCKER / capacity protocol`: v100 formal/future 主 rows 的 cache capacity 未启用，aggregate occupancy/eviction 为 0；这些 artifacts 不支持 cache pollution、turnover、eviction regret、byte efficiency 或容量受限 placement claim。
- `BLOCKER / request observability`: 当前 raw rows 没有 request-level object bytes、hit source（local/neighbor/cloud）、delay decomposition、future reuse 与 counterfactual cloud latency，无法计算 byte hit、P95/P99、useful-cache ratio 或 latency-saved-per-MB。
- `OPEN / cache oracle`: `oracle_prediction` 不是 future-request capacity oracle；尚未建立固定 horizon、相同容量/transfer cost 下的 offline optimal placement/eviction upper bound。
- 完整审计与 P0--P3 路线见 `docs/project/full_system_cache_algorithm_audit_20260814.md`。
# 2026-08-17 MB capacity implementation status

- `RESOLVED / live capacity semantics`: 环境已支持 slot/MB、initial enforcement、原子 multi-victim 与 oversized 拒绝。这不追溯改变 v100 capacity-disabled artifacts，也未解决独立 cache baseline、正式 byte metrics 或 oracle blocker。

# 2026-08-18 Classical baseline boundary

- `RESOLVED`: FIFO/LFU/Aging-LFU/Random、明确 baseline identity 与 agent-policy mismatch 检查已实现。
- `OPEN`: 正式 byte-hit/pollution/regret/latency-saved、capacity oracle 与 paper fairness manifest 不属于本 Goal；controlled validation 不能替代 formal evidence。
