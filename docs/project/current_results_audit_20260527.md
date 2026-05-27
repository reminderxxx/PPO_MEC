# Current Results Audit

## 2026-05-27 freshness guard formal run 与 admission-guard 后续状态

本节优先于下方 v6 masked full-train 段落读取。

- 最新完成的 3-seed formal run 为 `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_freshness_guard_20260527_v1/`。
- Gate: `formal_contract.ready=true`，`passed=false`，`paper_claim_ready=false`。
- `mixed_informative`：SA `98.091111`，`popularity_cache_heuristic` `98.146667`，差值 `-0.055556`；blockers 为 `sa_total_reward_not_above_popularity` 与 `benchmark_minimum_success_not_reached`。
- `full_stratified`：SA `90.153148`，`popularity_cache_heuristic` `90.171667`，差值 `-0.018519`；blockers 相同。
- 诊断 `artifacts/analysis/top_journal_mechanism_v6_freshness_guard_actionmix_diagnosis_20260527/` 显示剩余负例来自低置信度、next-RSU 未对齐时的过早 prefetch；因此代码已新增 `predictive_prefetch_admission_guard_*`，但该新增 guard 目前只有 quick/debug chain 验证，不构成新的 formal 结果。
- 当前 canonical 仍为 `final_submission_full_current_baselines_20260511_v1`；v6 必须重新通过 3-seed formal/holdout gate 后才能替换。

## 2026-05-27 最新修复版 v6 closed-loop 审计结论

本节优先于下方旧 v6 审计段落读取。旧 run `top_journal_mechanism_v6_strong_competition_20260527_v1` 保留为修复前 negative baseline；最新修复版为：

- Run root: `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_masked_fulltrain_20260527_v1/`
- Gate: `formal_contract.ready=true`、`baseline_protocol_audit.passed=true`、`passed=false`、`paper_claim_ready=false`
- Settings: `sa_episodes=128`、`train_window_count=6`、`window_mode_for_training=full_stratified`、seeds `7/13/29`

| Benchmark mode | Gate blockers | SA reward | Popularity reward | SA - Popularity | Strongest learned | SA - strongest learned | Projection / invalid | Mechanism success gate |
|---|---|---:|---:|---:|---|---:|---:|---|
| `mixed_informative` | `sa_total_reward_not_above_popularity`; `benchmark_minimum_success_not_reached` | `98.091111` | `98.146667` | `-0.055556` | `mappo=82.555000` | `+15.536111` | `0.0 / 0.0` | `18/125=0.144000` |
| `full_stratified` | `sa_total_reward_not_above_popularity`; `benchmark_minimum_success_not_reached` | `90.153148` | `90.171667` | `-0.018519` | `mappo=86.142222` | `+4.010926` | `0.0 / 0.0` | `33/188=0.175532` |

修复确认：

- invalid action / action projection 已在 formal benchmark 中归零。旧 v6 run 的 gate total 为 mixed `85/85`、full `432/432`；新 run 为 mixed `0/0`、full `0/0`。
- full split continuity 已与 popularity heuristic 持平：SA `workflow_continuity_rate=0.927399`、`handoff_failure_rate=0.123148`；本轮不再存在旧 v6 的 `cache_offload_drl` learned-side blocker。
- SA 在本轮超过所有 learned baselines，但仍没有超过 supplementary `popularity_cache_heuristic`，且 mechanism success gate 未达标。因此该 run 仍不能替换 canonical。

当前处理：

- 主论文 canonical 仍是 `final_submission_full_current_baselines_20260511_v1`。
- `top_journal_mechanism_v6_masked_fulltrain_20260527_v1` 可作为“修复 invalid/projection 后的最新 negative candidate”和下一轮优化起点。
- 下一步应优先提高 validated mechanism success rate，并检查 reward shaping / guard-delta 计分，使 SA 在不依赖非法动作投影的情况下稳定超过 `popularity_cache_heuristic`。

更新日期：2026-05-27

用途：给论文写作和下一轮实验调度提供明确状态判定。本文只汇总已存在 artifact 与已实现代码能力；所有数值结论以 `artifacts/` 下 JSON/CSV 为准。

## 状态判定规则

- `Results 可引用`：已有 formal/holdout/support artifact，且 gate 中 `paper_claim_ready=true` 或 comparison package `paper_ready_package_ready=true`。
- `Methods 可描述`：代码、profile、配置或审计协议已实现，但没有 3 seed formal/holdout final-submission artifact；不得写任何性能提升数值。
- `Negative / Appendix only`：artifact 已生成但 gate 未通过；只能作为失败候选、诊断或审计边界，不进入主结果。
- `Related Work only`：外部论文只支撑动机、定位、baseline rationale 或 reviewer response；不得支撑 PPO_MEC 数值 claim。

## 当前结果审计表

| 对象 | 判定 | 精确证据 | 论文写作处理 | 必须修改 / 补齐的项 |
|---|---|---|---|---|
| `final_submission_full_current_baselines_20260511_v1` final gate | `Results 可引用` | `final_submission_gate_report.json`：`paper_claim_ready=true`、`target_reached=true`、`blockers=[]` | Results 主表可引用；句子必须限定为 “under the 2026-05-11 current-baseline protocol”。 | 若论文强调 MAPPO 已按 v3 优化，需要另跑 MAPPO v3 / SA v6 final-submission；不得用该 artifact 替代。 |
| `final_submission_full_current_baselines_20260511_v1` comparison package | `Results 可引用` | `top_journal_comparison_report.json`：`review_ready=true`、`paper_ready_package_ready=true`、`self_review_summary.blocker_count=0`、`limitation_count=5`、`pass_count=13` | 可导出 paper-ready table；正文必须同步写 5 个 limitation。 | 不得删除 MAPPO action-mix risk、heuristic 接近、prediction/oracle 边界。 |
| canonical learned-baseline CI | `Results 可引用` | formal/holdout 对 9 个 learned baselines 的 paired total reward CI 全为正；最弱 vs `ppo`：formal mean `+4.745278`、CI `[2.3372, 7.028835]`；holdout mean `+6.975`、CI `[4.155505, 9.63982]` | 可写 “SA-GHMAPPO outperforms all clean-retrained learned baselines in the canonical package”。 | 主 claim 必须按 actual strongest learned baseline 排序，不得硬写“只比 MAPPO 强”。 |
| canonical split-level margin | `Results 可引用，弱项需标注` | Formal Mixed `+7.871667`、Formal Full `+3.703148`、Holdout Mixed `+10.097777`、Holdout Full `+5.636667`；四个 split 的 strongest learned baseline 均为 `ppo` | 可写 split-level margin；Formal Full 是 weakest split，需要如实报告。 | 下一轮 v6 必须重新检查 Formal Full；若被 MAPPO v3 或 `cache_offload_drl` 拉低，则不得沿用 canonical margin。 |
| support suites | `Results 可引用` | vs `ppo` 最弱项：Prediction `+2.794583`、Robustness `+6.879236`、Scalability `+2.159306`；CI 均为正 | Supplementary / robustness 表可引用。 | 不得把 `no_prediction` / `oracle_prediction` diagnostic setting 写成 universal prediction dominance。 |
| canonical MAPPO action-mix audit | `Results 可引用为风险说明` | 8 条 MAPPO-vs-PPO/DQN audit 均为 `high`；MAPPO prefetch `0.0`；vs PPO reward delta 约 `-29.25` 到 `-31.81` | MAPPO 只能写作有效 controller-level CTDE baseline，且在该 artifact 中存在 action-mix collapse。 | 主优势证据锚定 PPO / strongest learned baseline；不得把旧 MAPPO 低分写成主贡献。 |
| `final_submission_v5_perf_robust_20260515_v1` | `Negative / Appendix only` | `paper_claim_ready=false`；blockers：`cache_offload_drl` formal CI low `-1.008212`、holdout CI low `-3.372274` | 可在 appendix 或 internal audit 写 “v5 failed promotion gate”。 | 不得替换 canonical；不得输出 paper-ready claim。 |
| v5 split margins | `Negative / Appendix only` | Formal Full `+0.837777`、Holdout Mixed `+1.443333`、Holdout Full `+1.360953` | 只能解释 v5 为什么没有晋级。 | 后续优化目标必须提高 weak split margin，并消除 `cache_offload_drl` CI blocker。 |
| v5 MAPPO behavior | `Negative / Appendix only` | MAPPO action-mix audit 降为 `tracked`；prefetch 约 `0.24` 到 `0.5` | 可写“v5 中 MAPPO collapse 缓解，但 package 仍失败”。 | 不能把 MAPPO 改善单独写成主结果，因为 final gate 未通过。 |
| MAPPO v3 / `mappo_strong_audit` implementation | `Methods 可描述` | 已实现 `aggregation_reason_weighted_controller_ppo_v3`、三头 policy/entropy floors、checkpoint config/load 审计；debug training 和 checkpoint load 通过 | Methods / implementation 可描述；不得写 MAPPO v3 的 final-submission 性能 claim。 | 必须跑 3 seed formal + holdout + comparison report，且生成新版 `mappo_action_mix_audit.csv/.tex` 后才可替换旧 MAPPO 论述。 |
| `top_journal_mechanism_v6_strong_competition_20260527_v1` closed-loop | `Negative / Appendix only` | `gate_report.json`：`formal_contract.ready=true`、`baseline_protocol_audit.passed=true`、`passed=false`、`paper_claim_ready=false` | 可作为最新失败候选和诊断依据；不得替换 canonical，不得继续写“v6 已优于强基线”。 | 先修 SA v6 weak split 与 guard/projection 问题，再决定是否跑 final-submission。 |
| 2025-2026 文献补充 | `Related Work only` | `literature_reference_table.md` 已补 TITS 2025 distributed VEC offloading、TITS 2025 SAGIN offloading、FGCS 2026 V2V/MARL offloading、TMC 2025 EdgeLLM、TMC 2026 H2O | 只能用于 Related Work、motivation、baseline rationale 或 reviewer response。 | H2O DOI 仍需 IEEE Xplore 复核；FGCS 条目不可称为 IEEE/ACM 顶刊主证据。 |

## 最新 v6 closed-loop 结果

Run root：

- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_strong_competition_20260527_v1/`
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_strong_competition_20260527_v1/gate_report.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_strong_competition_20260527_v1/gate_summary.csv`

执行命令：

```bash
python scripts/run_top_journal_closed_loop.py --run_id top_journal_mechanism_v6_strong_competition_20260527_v1 --seeds 7 13 29 --sa_profile top_journal_mechanism_v6_strong_competition --mappo_baseline_profile mappo_strong_audit --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure
```

总体结论：

- `formal_contract.ready=true`：seed、预算、benchmark modes、`primary_vehicle_selection=handoff_pressure` 等正式合同满足。
- `baseline_protocol_audit.passed=true`：MAPPO v3 profile 审计通过，`baseline_protocol_versions.mappo.head_credit_protocol=aggregation_reason_weighted_controller_ppo_v3`。
- `passed=false`、`paper_claim_ready=false`：本轮 v6 不能晋级，不能替换 2026-05-11 canonical。
- `next_optimization_hints` 指向：提高 SA reward advantage，扩大 train windows/seeds，检查 event/value losses，并 sweep `event_logit_temperature_final`。

| Benchmark mode | Gate blockers | SA reward | Popularity reward | SA - Popularity | Strongest learned | SA - strongest learned | SA - MAPPO v3 | 机制 success gate |
|---|---|---:|---:|---:|---|---:|---:|---|
| `mixed_informative` | `sa_total_reward_not_above_popularity`; `benchmark_minimum_success_not_reached` | `96.874444` | `98.146667` | `-1.272223` | `cache_offload_drl=92.024444` | `+4.850000` | `+13.439444` | `24/103=0.233010` |
| `full_stratified` | `sa_total_reward_not_above_cache_offload_drl`; `sa_total_reward_not_above_popularity`; `benchmark_minimum_success_not_reached` | `89.692037` | `90.171667` | `-0.479630` | `cache_offload_drl=90.168889` | `-0.476852` | `+4.692778` | `39/163=0.239264` |

关键诊断：

- v6 对 PPO 仍有大幅优势：mixed `+26.327222`，full `+19.590185`；但 PPO 已不是本轮 strongest learned baseline。
- `cache_offload_drl` 成为 strongest learned baseline；mixed 下 SA 仍领先 `+4.85`，full 下 SA 反而低 `-0.476852`，这是本轮 learned-side blocker。
- SA 与 `popularity_cache_heuristic` 的 continuity、handoff failure、backhaul、handoff ready、mechanism realization、migration overhead 在两种模式下基本同形，但 reward 低于 popularity。主要差异来自 SA 被 guard/projection 大量改写后，机制 attempt/shaping/handoff reward 少于 popularity。
- SA 的 action projection 仍偏高：mixed `4.722222`，full `8.0`；invalid action attempt rate 分别约 `0.381559` 和 `0.608726`。这说明 v6 策略本身尚未学到足够干净的可执行动作，仍依赖 guard 修正。
- MAPPO v3 在本轮 closed-loop 中未形成 paper-ready 结果：mixed reward `83.435`、full reward `84.999259`；prefetch count 仍为 `0.0`，但 continuity 和 handoff failure 明显优于旧 canonical MAPPO。它证明 v3 协议可运行，不证明 MAPPO v3 已足够强。

## 与 canonical / v5 对比

| Run | Gate 状态 | Strongest learned baseline | 最弱主 split margin | MAPPO 状态 | 论文处理 |
|---|---|---|---:|---|---|
| `final_submission_full_current_baselines_20260511_v1` | `paper_claim_ready=true` | `ppo` | `+3.703148` | action-mix `high` risk；prefetch `0.0` | 当前唯一主论文 canonical |
| `final_submission_v5_perf_robust_20260515_v1` | `paper_claim_ready=false` | `cache_offload_drl` | `+0.837777` | action-mix 降为 `tracked`；prefetch `0.24` 到 `0.5` | negative / appendix |
| `top_journal_mechanism_v6_strong_competition_20260527_v1` | `paper_claim_ready=false` | `cache_offload_drl` | `-0.476852` | v3 协议审计通过；closed-loop 中仍 prefetch `0.0` | negative / appendix；暂不跑 final-submission promotion |

## 下一轮验收表

| 优先级 | 验收对象 | 必须生成的 artifact | 通过条件 | 失败处理 |
|---:|---|---|---|---|
| P0 | SA v6 修复候选 closed-loop | 新 `artifacts/experiments/top_journal_closed_loop/<run_id>/gate_report.json` | `formal_contract.ready=true`、`baseline_protocol_audit.passed=true`、`passed=true`、`paper_claim_ready=true` | 若仍不超过 `popularity_cache_heuristic` 或 `cache_offload_drl`，只作 negative candidate。 |
| P0 | learned-baseline paired CI | final comparison report | formal 和 holdout 对全部 learned baselines 的 total reward CI 下界均 `> 0` | 哪个 baseline CI 下界 `<=0`，哪个就是 blocker。 |
| P0 | strongest learned baseline 排名 | `strongest_comparator_audit` | 每个 split 中 SA-GHMAPPO 都高于 actual strongest learned baseline | 不允许 hard-code PPO；若 strongest 是 MAPPO v3 或 `cache_offload_drl`，按实际写。 |
| P0 | MAPPO v3 action-mix audit | `mappo_action_mix_audit.csv` / `.tex` | 不再出现 prefetch 全 0；若 risk 为 high，报告风险并避免用 MAPPO 弱点支撑主 claim | high risk 不一定阻断 final gate，但必须进入 limitation。 |
| P1 | SA invalid action / guard projection | benchmark aggregate summary | `action_projection_count` 和 `invalid_action_attempt_rate` 明显低于本轮 v6，且 reward 不低于 popularity | 若系统指标同形但 reward 仍低，优先检查 shaping reward 与 guard-delta 计分。 |
| P1 | `cache_offload_drl` blocker | paired reward statistics | formal/holdout vs `cache_offload_drl` CI 下界均 `> 0` | 若任一边界 `<=0`，final package 不是 paper-ready。 |
| P1 | prediction claim boundary | prediction robustness report | learned/noisy predictor setting 对 learned baselines 为正 CI | `oracle_prediction` / `no_prediction` 仍只写 diagnostic。 |
| P2 | real model-cache backend | importer + independent benchmark profile | 有真实 request/event 或 file-size profile contract、adapter_id 映射、独立结果标签 | 未实现前不得声称 benchmark cache events 来自真实 model-cache trace。 |
| P2 | full vehicle/RSU MARL | 新 observation/action contract + retrained baselines | vehicle/RSU-level wrapper、训练、评估、benchmark 消费端全部冻结 | 未完成前 MAPPO/QMIX/Controller-MAT 只写 controller-level。 |

## 当前改进方案

1. 先不跑 `final_submission_v6_mappo_v3_strong_20260527_v1` promotion。closed-loop 已显示 `paper_claim_ready=false`，继续跑 final-submission 会消耗时间且大概率得到 negative package。
2. 修 SA v6 的可执行动作学习：降低 invalid action attempt 与 action projection，重点查 `event_logit_temperature_final`、event head credit、guard-delta 后的学习信号，以及 cache/prefetch/action mask 的训练分布。
3. 对 full split 单独优化 `cache_offload_drl` blocker：full 下 SA 的 continuity `0.927399` 低于 `cache_offload_drl` 的 `0.982744`，handoff failure `0.123148` 高于 `0.037037`，需要提高 full-stratified 下的 service continuity 和 migration timing。
4. MAPPO v3 继续保留为强对照实现，但不能急着写入主结果。下一轮要看 `mappo_action_mix_audit` 是否不再 prefetch 全 0，并确认 MAPPO v3 是否成为 strongest learned baseline。
5. 论文当前主结果仍用 `final_submission_full_current_baselines_20260511_v1`；v5 和 v6 都只作为失败筛选、审计表和改进路径。
