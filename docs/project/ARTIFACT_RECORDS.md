# Artifact Records

## 2026-07-17 v13 partial-reward-decoupled MAPPO latest full-dev

状态：`[dev-full-success]` `[candidate]` `[not-hidden]`

路径：

- `artifacts/experiments/top_journal_prd_option_v13_20260717/seed_checkpoint_manifest_prd_event_latest.json`
- `artifacts/experiments/top_journal_prd_option_v13_20260717/main_results_full_stratified_latest/main_results_full_stratified_20260717_124815_375515/aggregate_summary.json`
- `artifacts/experiments/top_journal_prd_option_v13_20260717/main_results_full_stratified_latest/main_results_full_stratified_20260717_124815_375515/sa_advantage_diagnosis.json`
- `artifacts/experiments/top_journal_prd_option_v13_20260717/main_results_full_stratified_best_reward/main_results_full_stratified_20260717_124435_673243/aggregate_summary.json`

确认结果：v13 latest 使用 5 seeds、20 frozen dev windows、2 workflows/window、12 agents、`full_stratified`、`primary_vehicle_selection=handoff_pressure`。SA-GHMAPPO total reward `79.64465`，高于 `popularity_cache_heuristic=79.46875`、`ppo=77.18775`、`mappo=72.6328` 和全部其他对照；相对 strongest other 的 margin 为 `+0.17590`，高于 v12 的 `+0.12465`。机制 realization 为 `0.245`，高于 v12/best-by-reward `0.195` 和 popularity `0.175`；continuity 为 `0.811677`，高于 v12 `0.807893`。同一 v13 profile 的 `best_by_reward` full-dev 结果与 v12 完全一致，因为 checkpoint audit 显示 selected checkpoint source update 为 0 warm-start。

结论边界：这是 frozen dev evidence，不是 formal/hidden/future-validation 或 paper-ready 证据。hidden holdout 已 consumed 且未用于 v13 筛选。v13 不能替换 v8 canonical；promotion 需要新冻结 future-validation split、统计审查、baseline fairness 审查和 readiness audit。`latest_after_prd_training` 是为了评估 PRD 学习后的策略，不能对外包装成 hidden-validated checkpoint selection。

## 2026-07-17 v12 learned MAPPO option gate full-dev

状态：`[dev-full-success]` `[candidate]` `[not-hidden]`

路径：

- `artifacts/experiments/top_journal_mappo_reward_v12_learned_option_20260717/seed_checkpoint_manifest.json`
- `artifacts/experiments/top_journal_mappo_reward_v12_learned_option_20260717/main_results_full_stratified_mech_preserve/main_results_full_stratified_20260717_115754_212344/aggregate_summary.json`
- `artifacts/experiments/top_journal_mappo_reward_v12_learned_option_20260717/main_results_full_stratified_mech_preserve/main_results_full_stratified_20260717_115754_212344/sa_advantage_diagnosis.json`

确认结果：v12 使用 5 seeds、20 frozen dev windows、2 workflows/window、`full_stratified`、`primary_vehicle_selection=handoff_pressure`。SA-GHMAPPO total reward `79.5934`，高于 `popularity_cache_heuristic=79.46875`、`ppo=77.18775`、`mappo=72.6328` 和全部其他对照；`sa_advantage_diagnosis.blockers=[]`、`minimum_success_reached=true`。分层结果为 mechanism window SA `82.758` > popularity `82.3425`，active non-mechanism 三者 `83.275` 持平，idle/sparse SA 与 popularity 均为 `77.3975`。

结论边界：这是 frozen dev evidence，不是 formal/hidden/future-validation 或 paper-ready 证据。hidden holdout 已 consumed 且未用于 v12 筛选。v12 不能替换 v8 canonical；promotion 需要新冻结 future-validation split、统计审查、baseline fairness 审查和 readiness audit。PPO 在 handoff failure/backhaul 上仍更优，不能据此声明 v12 已全面解决系统 trade-off。

## 2026-06-21 strict-full v8 frozen candidate, formal and hidden

状态：`[E2-artifact-audited]` `[strict-full-blocker-resolved]` `[major-revision]`

路径：

- `artifacts/experiments/top_journal_closed_loop/strict_full_v8_dev_screen_20260621_v2/`
- `artifacts/experiments/strict_full_v8_formal_all_baselines_20260621_v1/main_results_full_stratified_20260621_025440_591857/`
- `artifacts/experiments/strict_full_v8_hidden_holdout_20260621_v1/main_results_full_stratified_20260621_201353_886213/`
- `artifacts/experiments/strict_full_v8_external_lust_grid_20260621_v2/main_results_full_stratified_20260621_202424_612488/`
- `artifacts/analysis/strict_full_v8_formal_all_baselines_statistics_20260621_v1/`
- `artifacts/analysis/strict_full_v8_hidden_all_baselines_statistics_20260621_v1/`
- `artifacts/analysis/strict_full_v8_external_lust_grid_statistics_20260621_v2/`
- `artifacts/audits/strict_full_v8_integrity_20260621/`

确认结果：v8 使用冻结的 20-window/split、5-seed 协议；formal 与一次性 hidden 对全部 learned baselines 的 reward BCa CI 为正，对 DT continuity 的 CI 也为正。原 v7 strict-full blocker已解除。完整性审计 11457 个文件通过，missing reference 与 JSON error 为 0。

结论边界：hidden 相对 PPO 的 handoff failure 显著更差，formal/hidden backhaul cost 更高，对 popularity heuristic 的 reward CI 跨 0；LuST 只有 4 个 outer windows。最新 verdict 为 `Major revision (78/100)`，数值和禁止表述见 `top_journal_readiness_audit_20260621.md`。

## 2026-06-18 v7 independent rebuild and strict review

状态：`[E3-reproduced]` `[legacy-gate-pass]` `[strict-review-not-ready]`

路径：

- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v7_latency_fallback_20260618_rebuild_v1/`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260618_rebuild_v1/`
- `artifacts/experiments/top_journal_support_suite/top_journal_v7_mechanism_ablation_20260618_v1/`
- `artifacts/analysis/top_journal_v7_rebuild_integrity_20260618/`

旧 v7 formal 核心值和 final gate 已独立 clean retrain 复现；SHA-256、manifest references、27 个 baseline checkpoint provenance 和 comparison rebuild 均通过。旧 offset-3 与 formal 时间窗口重叠。严格 non-overlap formal/holdout 中 mixed 对 DT 的 CI 为正，但 full formal/holdout CI 跨 0，因此此 rebuild 不是新的 TMC-ready canonical。完整数值见 `top_journal_readiness_audit_20260618.md`。

## 2026-05-28 Top Journal Final Submission v7 Latency Fallback

状态：`[legacy-paper-ready]` `[reproduced]` `[strict-review-not-ready]`

路径：
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/final_submission_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/learned_suites/final_submission_v7_latency_fallback_20260528_v1_iter1_formal/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/learned_suites/final_submission_v7_latency_fallback_20260528_v1_iter1_holdout_offset3/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/support/prediction/prediction_robustness_20260528_202640_186901/prediction_robustness_summary.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/support/robustness/robustness_20260528_202823/aggregate_summary.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/support/scalability/scalability_20260528_203154/aggregate_summary.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/top_journal_comparison_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/paper_ready/paper_ready_report.md`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/paper_ready/paper_ready_main_comparison.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/paper_ready/paper_ready_paired_reward_statistics.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/paper_ready/paper_ready_support_reward_statistics.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/comparison_report/paper_ready/paper_ready_self_review.json`

用途：当前 paper-ready canonical final-submission package。使用 v7 latency fallback clean-retrain SA checkpoint，并在 final suite 内 clean retrain 9 个 primary learned baselines 的 3 个 seed。

确认结果：
- Final gate：`target_reached=true`、`paper_claim_ready=true`、`blockers=[]`。
- Comparison package：`review_ready=true`、`paper_ready_package_ready=true`，self-review `blocker_count=0`、`limitation_count=3`、`pass_count=15`。
- Formal learned gate 与 offset=3 sensitivity gate 均为 `passed=true`、legacy `paper_claim_ready=true`；offset-3 与 formal 时间窗口重叠，不是 independent holdout。
- `formal_training_provenance.passed=true`，required agents 为 `ppo/mappo/dqn/dueling_dqn/qmix/controller_mat/dag_offload_drl/cache_offload_drl/dt_handoff_drl`，`record_count=27`。
- Formal split：strongest learned baseline 均为 `dt_handoff_drl`；mixed margin `+11.176111`，full margin `+3.377407`。
- Holdout offset=3：strongest learned baseline 均为 `dt_handoff_drl`；mixed margin `+8.442778`，full margin `+5.242143`。
- Paired CI：formal weakest vs `dt_handoff_drl` mean `+5.327083`、95% CI `[1.594094, 8.963719]`；holdout weakest vs `dt_handoff_drl` mean `+6.202333`、95% CI `[1.607076, 10.593939]`。
- Support weakest learned margins：prediction vs `dt_handoff_drl` `+4.833472` CI `[3.170913, 6.600080]`；robustness `+9.799097` CI `[8.329792, 11.297618]`；scalability `+4.133380` CI `[3.245373, 5.016079]`。

结论边界：
- `popularity_cache_heuristic` 是 close supplementary reference，formal/holdout mixed/full margins 分别为 `+0.250000`、`+0.479629`、`+0.355556`、`+0.376191`；不要写成大幅超过手写 heuristic。
- 论文中必须保留 comparison self-review 的限制项：heuristic gap close、mechanism realization rate 不构成每个 split 的 standalone CI-positive 优势、backhaul savings 不是 universal headline。
- MAPPO/QMIX/Controller-MAT/DAG/cache/DT 均按 controller-level 或 semantic-discrete contract 表述，不得写成 vehicle/RSU-level full MARL wrapper。

## 2026-05-28 Top Journal Mechanism v7 Latency Fallback Closed Loop

状态：`[formal-pass]` `[candidate]` `[optimization-validation]`

路径：
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v7_latency_fallback_20260528_v1/`
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v7_latency_fallback_20260528_v1/gate_report.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v7_latency_fallback_20260528_v1/gate_summary.csv`
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v7_latency_fallback_20260528_v1/seed_checkpoint_manifest.json`
- `artifacts/analysis/top_journal_mechanism_v7_latency_fallback_actionmix_diagnosis_20260528/`

用途：记录 v6 guards + clean-retrain latency fallback 的 3-seed formal closed-loop 结果。该 run 通过 closed-loop formal gate，但尚未生成 final-submission holdout/support/comparison package，因此不是当前 canonical。

确认结果：
- `formal_contract.ready=true`、`baseline_protocol_audit.passed=true`、`passed=true`、`paper_claim_ready=true`。
- `mixed_informative`：SA `98.396667`，`popularity_cache_heuristic` `98.146667`，SA delta `+0.250000`；strongest learned baseline `mappo=82.555000`。
- `full_stratified`：SA `90.651296`，`popularity_cache_heuristic` `90.171667`，SA delta `+0.479629`；strongest learned baseline `mappo=86.142222`。
- 两个 benchmark mode 下 SA 与 popularity 的 continuity、handoff failure、backhaul 均持平；收益来自 latency fallback 带来的 delay penalty 下降。
- action-mix 诊断：active/idle 非机制窗口贡献主要正收益；mechanism_activating 窗口基本持平，mixed 下有一个轻微 losing pair，后续 final-submission 仍需检查 holdout 稳定性。

结论边界：
- 该 artifact 可作为 v7 候选优化证据；不能单独替换 `final_submission_full_current_baselines_20260511_v1` 或写成最终 paper-grade package。
- 下一步必须运行 final-submission/holdout/support，并生成 comparison report / paper-ready package。

## 2026-05-27 Top Journal Mechanism v6 Freshness Guard Closed Loop

状态：`[negative-result]` `[audit]` `[optimization-diagnosis]`

路径：
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_freshness_guard_20260527_v1/`
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_freshness_guard_20260527_v1/gate_report.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_freshness_guard_20260527_v1/gate_summary.csv`
- `artifacts/analysis/top_journal_mechanism_v6_freshness_guard_actionmix_diagnosis_20260527/`

用途：记录 freshness-aware cache-warm guard 加入后的一次 3-seed formal closed-loop 结果，并作为后续 confidence/alignment prefetch admission guard 的诊断依据。该 run 不是 final-submission package，也不是当前 canonical。

确认结果：
- `formal_contract.ready=true`，但 `passed=false`、`paper_claim_ready=false`。
- `mixed_informative`：SA `98.091111`，`popularity_cache_heuristic` `98.146667`，SA 差值 `-0.055556`。
- `full_stratified`：SA `90.153148`，`popularity_cache_heuristic` `90.171667`，SA 差值 `-0.018519`。
- blockers 为 `sa_total_reward_not_above_popularity` 与 `benchmark_minimum_success_not_reached`；SA 仍超过所有 learned baselines，但未超过 supplementary heuristic。
- action-mix 诊断定位到 `window_off246_len24_t293_316` / `j_8` / seed `13` 的 prefetch realization gap：低置信度且 next-RSU 未对齐时过早 prefetch 导致 `expired_miss`。

结论边界：
- 该 artifact 证明 freshness countdown guard 单独不足以通过 gate。
- 后续新增 `predictive_prefetch_admission_guard_*` 只完成 quick/debug chain 验证，不能把 quick 结果写成 formal improvement。

## 2026-05-27 Top Journal Mechanism v6 Masked Fulltrain Closed Loop

状态：`[negative-result]` `[audit]` `[repair-validation]`

路径：
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_masked_fulltrain_20260527_v1/`
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_masked_fulltrain_20260527_v1/gate_report.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_masked_fulltrain_20260527_v1/gate_summary.csv`
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_masked_fulltrain_20260527_v1/seed_checkpoint_manifest.json`

用途：记录 SA v6 invalid action / action projection 修复后，使用 full-stratified training windows 与 v6 budget 的最新 closed-loop 结果。该 run 不是 final-submission package，也不是当前 canonical。

确认结果：
- `formal_contract.ready=true`，正式 seed、预算、benchmark modes 与 `primary_vehicle_selection=handoff_pressure` 合同满足。
- `baseline_protocol_audit.passed=true`，MAPPO v3 checkpoint protocol 记录为 `aggregation_reason_weighted_controller_ppo_v3`。
- gate 未通过：`passed=false`、`paper_claim_ready=false`。
- `mixed_informative`：SA `98.091111`，`popularity_cache_heuristic` `98.146667`，SA 差值 `-0.055556`；strongest learned baseline 为 `mappo=82.555000`，SA 差值 `+15.536111`。
- `full_stratified`：SA `90.153148`，`popularity_cache_heuristic` `90.171667`，SA 差值 `-0.018519`；strongest learned baseline 为 `mappo=86.142222`，SA 差值 `+4.010926`。
- formal benchmark aggregate 中 `action_projection_count=0.0`、`invalid_action_attempt_count=0.0`；修复前 v6 run 的 gate total 为 mixed `85/85`、full `432/432`。
- mechanism success gate 未达标：mixed `18/125=0.144000`，full `33/188=0.175532`。

结论边界：
- 该结果确认 SA v6 的 invalid action / projection 问题已修复，且 SA 重新超过所有 learned baselines。
- 当前 blocker 已收窄为 supplementary `popularity_cache_heuristic` 的极小 reward gap 和 mechanism success gate；在两者通过前，不替换 `final_submission_full_current_baselines_20260511_v1`，也不运行 v6 final-submission promotion。

## 2026-05-27 Top Journal Mechanism v6 Strong-Competition Closed Loop

状态：`[negative-result]` `[audit]`

路径：

- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_strong_competition_20260527_v1/`
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_strong_competition_20260527_v1/gate_report.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_strong_competition_20260527_v1/gate_summary.csv`
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_strong_competition_20260527_v1/seed_checkpoint_manifest.json`

用途：记录 SA v6 与 MAPPO v3 强对照 profile 的最新 closed-loop 结果。该 run 不是 final-submission package，也不是当前 canonical。

确认结果：

- `formal_contract.ready=true`，正式 seed、预算、benchmark modes 和 `primary_vehicle_selection=handoff_pressure` 合同满足。
- `baseline_protocol_audit.passed=true`，MAPPO v3 checkpoint protocol 记录为 `aggregation_reason_weighted_controller_ppo_v3`。
- gate 未通过：`passed=false`、`paper_claim_ready=false`。
- `mixed_informative`：SA `96.874444`，`popularity_cache_heuristic` `98.146667`，`cache_offload_drl` `92.024444`；blockers 为 `sa_total_reward_not_above_popularity` 和 `benchmark_minimum_success_not_reached`。
- `full_stratified`：SA `89.692037`，`popularity_cache_heuristic` `90.171667`，`cache_offload_drl` `90.168889`；blockers 为 `sa_total_reward_not_above_cache_offload_drl`、`sa_total_reward_not_above_popularity` 和 `benchmark_minimum_success_not_reached`。
- MAPPO v3 本轮 closed-loop 中 mixed reward `83.435`、full reward `84.999259`；协议可运行，但不能支撑 paper-ready MAPPO 性能 claim。

结论边界：

- 该结果只能作为 negative candidate / optimization diagnosis，不替换 `final_submission_full_current_baselines_20260511_v1`。
- 后续若要把 MAPPO v3 和 SA v6 写进主论文结果，需要先生成新的 final-submission package，并要求 final gate `paper_claim_ready=true`、comparison package `paper_ready_package_ready=true`。
- 当前已知 blocker 是 SA v6 在 full split 下落后 `cache_offload_drl`，并且在 mixed/full 下均没有超过 supplementary `popularity_cache_heuristic`。

## 2026-05-11 Full Current-Baseline Final Submission

状态：`[canonical]` `[paper-grade]`

路径：
- `artifacts/experiments/top_journal_final_submission/final_submission_full_current_baselines_20260511_v1/`
- `artifacts/experiments/top_journal_final_submission/final_submission_full_current_baselines_20260511_v1/final_submission_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_full_current_baselines_20260511_v1/comparison_report/`

用途：当前可进论文主表的 final-submission package。该 run 对 `ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl` 执行同环境交互预算 clean retrain，并完成 formal / holdout / prediction robustness / robustness / scalability gate。

关键事实：
- `final_submission_gate_report.json` 中 `paper_claim_ready=true`、`target_reached=true`、`blockers=[]`。
- comparison report 中 `review_ready=true`、`paper_ready_package_ready=true`；自审 `blocker_count=0`、`limitation_count=5`、`pass_count=13`。
- 该 artifact 中 MAPPO 为 controller-level CTDE + 当时的 aggregation-reason controller head-credit protocol，`baseline_protocol_versions.mappo` 记录 `head_credit_enabled=True`、`event_policy_credit_floor=0.05`、`event_entropy_credit_floor=0.05`、`event_advantage_blend=1.0`。2026-05-27 之后新的 MAPPO claim 必须使用 v3 protocol 重新跑 final-submission gate。
- Formal / holdout 主表中 SA-GHMAPPO 均超过最强 learned baseline；最强 learned baseline 均为 `ppo`，reward margin 范围为 `+3.703148` 到 `+10.097777`。
- Cluster-bootstrap paired total reward 对全部 9 个 learned baselines 均为 positive CI；formal 最弱 vs `ppo` mean delta `+4.745278`，95% CI `[2.3372, 7.028835]`；holdout 最弱 vs `ppo` mean delta `+6.975`，95% CI `[4.155505, 9.63982]`。
- Prediction / robustness / scalability support suites 对全部 9 个 learned baselines 均为 positive CI；最弱项分别为 vs `ppo`：`+2.794583`、`+6.879236`、`+2.159306`。

引用边界：
- 主 claim 面向 clean-retrained learned baselines；`reactive_greedy` 和 `popularity_cache_heuristic` 只作 supplementary reference。
- `popularity_cache_heuristic` 与 SA-GHMAPPO 很接近，最小 reward margin `+0.183333`，不能写成大幅超过 heuristic。
- `no_prediction` / `oracle_prediction` 是 diagnostic stress cases；`mechanism_realization_rate` 和部分 backhaul saving 不作为单独主 claim。

## 2026-05-10 MAPPO head-credit artifact boundary

状态：`[canonical-boundary-update]`

- 当前 live `mappo` 已升级为 controller-level CTDE + aggregation-reason controller head-credit baseline。
- `final_submission_controller_mappo_qmix_20260509_v1` 是 pre-MAPPO-head-credit package；它的 MAPPO 数值只能追溯历史，不能作为新版顶刊主表的 MAPPO 强对照。
- 新 canonical final-submission package 必须由当前代码重跑，且 `final_submission_gate_report.json` / learned suite report 中应包含 `baseline_protocol_versions.mappo`。

更新日期：2026-04-24

用途：记录当前项目可继续消费的 artifact 边界、来源路径和限制。这里维护 live 事实来源，不把历史混合结果写成当前结论。

## 状态标签

- `[canonical]`：当前正式事实来源。
- `[paper-grade]`：可用于论文主表或补充表，但必须确认协议与当前 live 算法池一致。
- `[supporting]`：辅助分析，可引用但不能替代主结果。
- `[archived]`：历史快照，只保留追溯价值，不作为当前 live 事实来源。
- `[cleaned]`：已从当前项目中删除或清空。

## Live Artifact Policy

状态：`[canonical]`

- live 模型层包含主方法 `sa_ghmappo` 和方向匹配对照算法池。
- 当前可训练 learned 对照算法为 `ppo`、`mappo`、`dqn`、`ddqn`、`dueling_dqn`、`dueling_ddqn`、`qmix`、`controller_mat`、`dag_offload_drl`、`cache_offload_drl` 和 `dt_handoff_drl`；paper-grade 默认主对照使用 `ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl`。`ippo` 是当前 single-wrapper contract 下的 diagnostic agent，不能作为独立 paper-grade 对照；`mappo` 是 controller-level CTDE baseline，`qmix` 是 controller-level value-decomposition baseline，`controller_mat` 是 controller-level transformer CTDE baseline，三者都不是 vehicle-agent / RSU-agent full MARL wrapper。`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl` 是领域专项 learned baselines，不使用 SA-GHMAPPO 专属 graph/surrogate/guard 机制。旧名 `flat_ppo`、`flat_mappo` 仅作为历史 artifact 路径和 run 名称保留。现有 `final_submission_controller_mappo_qmix_20260509_v1` 不含 `controller_mat` 或 DAG/cache/DT 领域 baseline，含新增对照的主表需重跑。
- 当前非学习启发式对照为 `reactive_greedy`、`popularity_cache_heuristic`，不需要 checkpoint。
- TD3 / SAC / MADDPG 当前不进入 live registry，也不作为可训练结果来源；`qmix` 已按 controller-level contract 进入 live registry。
- 对照算法正式结果需要通过 `scripts/train_algo_pool_real_sample.py`、`scripts/eval_algo_pool_real_sample.py` 和 `scripts/benchmark_main_results.py` 重新生成。
- 统一 baseline 闭环产物写入 `artifacts/experiments/baseline/<run_id>/`，并以 `comparison_summary.csv/json`、`comparison_summary_detailed.json`、`comparison_summary_by_window_class.csv` 和 `run_manifest.json` 作为实验管理入口。
- 历史 artifact 路径中出现的 `flat_ppo` / `flat_mappo` 表示旧 run 名称，不表示当前源码仍使用 PPO family 包装层。

## Current Checkpoint Sources

状态：`[canonical]`

当前主方法 checkpoint：

- `sa_ghmappo`: `artifacts/training/main_agents/sa_ghmappo/sa_ghmappo_train_20260415_154335_734767_seed7/checkpoints/best_by_reward.pt`

当前 round1 三 seed 对照训练 artifacts：

- `artifacts/training/algo_pool_formal_round1/flat_ppo/flat_ppo_train_20260424_190032_617002_seed7/`
- `artifacts/training/algo_pool_formal_round1/flat_ppo/flat_ppo_train_20260424_190032_624481_seed13/`
- `artifacts/training/algo_pool_formal_round1/flat_ppo/flat_ppo_train_20260424_190032_683702_seed29/`
- `artifacts/training/algo_pool_formal_round1/flat_mappo/flat_mappo_train_20260424_190126_588082_seed7/`
- `artifacts/training/algo_pool_formal_round1/flat_mappo/flat_mappo_train_20260424_190126_621893_seed13/`
- `artifacts/training/algo_pool_formal_round1/flat_mappo/flat_mappo_train_20260424_190126_618080_seed29/`

说明：这些目录是重构前生成的历史路径。当前源码主名为 `ppo` / `mappo`，benchmark 入口仍能通过 checkpoint 参数或 manifest 路径消费这些旧 checkpoint，但 `flat_ppo` / `flat_mappo` 不再是 live agent 名称。

## HF Model Cache Dataset Audit Round14

状态：`[supporting]`

本轮用于回答“HF 数据集是否适合全面接入真实数据实验，以及如何接入”。它是数据源审计和接入方案，不是 benchmark 结果。

Audit outputs：

- `artifacts/analysis/hf_model_cache_dataset_audit_round14/hf_model_cache_dataset_audit.csv`
- `artifacts/analysis/hf_model_cache_dataset_audit_round14/diagnosis_summary.json`
- `docs/agent/hf_model_cache_dataset_audit_round14_report.md`

Manifest / plan：

- `data/raw/model_cache/huggingface_model_cache_sources.json`
- `configs/data/hf_model_cache_integration_plan.json`

说明：

- 候选全集包含 `ClemSummer/qwen-model-cache`、`ClemSummer/cbow-model-cache`、`Efficient-Large-Model/imagenet-llamagen-cache`、`Kuperberg/bert-model-cache`、`amansapkota/examsathi-model-cache`。
- 结论是当前无直接 benchmark-ready 数据集；4 个候选可作为真实 file-size/cache-volume profile 后续接入，`amansapkota/examsathi-model-cache` 当前不适合。
- 该轮不下载 HF 原始文件，也不改变当前 `NGSIM + Alibaba` 主线 benchmark。

## Large-scale Real Dataset Comparison Round13

状态：`[supporting]`

本轮用于回答“大规模真实数据集对比试验”和主方法优劣，不替换 frozen paper 主表。

Benchmark outputs：

- `artifacts/benchmarks/large_scale_real_dataset_round13/mixed_informative/main_results_mixed_informative_20260427_134259_037473/aggregate_summary.json`
- `artifacts/benchmarks/large_scale_real_dataset_round13/mixed_informative/main_results_mixed_informative_20260427_134259_037473/benchmark_rows.csv`
- `artifacts/benchmarks/large_scale_real_dataset_round13/full_stratified/main_results_full_stratified_20260427_134547_361821/aggregate_summary.json`
- `artifacts/benchmarks/large_scale_real_dataset_round13/full_stratified/main_results_full_stratified_20260427_134547_361821/benchmark_rows.csv`

Analysis outputs：

- `artifacts/analysis/large_scale_real_dataset_round13/diagnosis_summary.json`
- `artifacts/analysis/large_scale_real_dataset_round13/agent_metric_summary.csv`
- `artifacts/analysis/large_scale_real_dataset_round13/sa_pairwise_deltas.csv`
- `artifacts/analysis/large_scale_real_dataset_round13/window_class_metric_summary.csv`
- `docs/agent/large_scale_real_dataset_round13_report.md`

说明：

- 数据链路为 `NGSIM mobility + Alibaba cluster-trace-v2018 batch_task DAG`。
- Hugging Face `ClemSummer/qwen-model-cache` 只作为 metadata-only 数据源声明，不直接驱动本轮 benchmark cache event。
- highD 原始文件缺失，未纳入本轮 benchmark。

## Formal Experiment Execution Round 1

状态：`[supporting]`

当前 round1 增强 schema 的 canonical run：

- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/comparison_summary.json`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/comparison_summary.csv`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/comparison_summary_detailed.json`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/comparison_summary_by_window_class.csv`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/run_manifest.json`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/seed_checkpoint_manifest.json`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark/main_results_mixed_informative_20260424_150518_946716/aggregate_summary.json`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark/main_results_mixed_informative_20260424_150518_946716/benchmark_rows.csv`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark_full_stratified/main_results_full_stratified_20260424_174124_983467/aggregate_summary.json`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark_full_stratified/main_results_full_stratified_20260424_174124_983467/benchmark_rows.csv`

该 run 可用于 formal experiment execution round1 的结构化检查，仍不是最终论文主表预算。

## Round1 Three-Seed Formal Baseline Rerun

状态：`[supporting]`

统一 manifest：

- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/seed_checkpoint_manifest_formal_round1_3seed.json`

Benchmark outputs：

- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark_formal_round1_3seed/main_results_mixed_informative_20260424_190319_732417/aggregate_summary.json`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark_formal_round1_3seed/main_results_mixed_informative_20260424_190319_732417/benchmark_rows.csv`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark_formal_round1_3seed_full_stratified/main_results_full_stratified_20260424_190503_729168/aggregate_summary.json`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark_formal_round1_3seed_full_stratified/main_results_full_stratified_20260424_190503_729168/benchmark_rows.csv`

说明：该 rerun 解决了 flat baseline 缺 seed `29` 且训练预算过低的问题。路径中的 `flat_*` 是历史 artifact 名称。

## Round1 Continuity Resolution

状态：`[supporting]`

报告：

- `docs/continuity_resolution_round1.md`

新增 manifest：

- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/seed_checkpoint_manifest_formal_round1_3seed_sa_best_continuity.json`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/seed_checkpoint_manifest_formal_round1_3seed_sa_best_mechanism_balanced.json`

Benchmark outputs：

- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark_formal_round1_3seed_sa_best_continuity/main_results_mixed_informative_20260424_191504_118596/aggregate_summary.json`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark_formal_round1_3seed_sa_best_mechanism_balanced/main_results_mixed_informative_20260424_191504_207066/aggregate_summary.json`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark_formal_round1_3seed_sa_best_continuity_full_stratified/main_results_full_stratified_20260424_191724_597766/aggregate_summary.json`

Diagnostics：

- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/continuity_resolution_round1/checkpoint_selection_and_stall_attribution.json`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/continuity_resolution_round1/checkpoint_selection_summary.csv`
- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/continuity_resolution_round1/stall_attribution_summary.csv`

说明：本轮通过 checkpoint selection 解决 continuity 对比问题，未修改环境或指标口径。

## Archived Families

状态：`[archived]`

以下历史结果只保留追溯价值：

- 旧 mixed/full 主表 aggregate 和对应 paper 导出。
- 旧 robustness、prediction robustness 和 scalability aggregate。
- 早期 benchmark、dry-run、real sample、eval、export job 和 log 目录。

历史 aggregate 可能包含旧算法名称；引用前必须显式标注为历史快照。

## Regeneration Guide

状态：`[canonical]`

重新生成当前主方法结果：

```bash
python scripts/train_sa_ghmappo_real_sample.py
python scripts/eval_sa_ghmappo_real_sample.py --checkpoint_path <checkpoint_path>
python scripts/benchmark_main_results.py --agents sa_ghmappo --sa_ghmappo_checkpoint_path <checkpoint_path>
```

重新生成当前对照算法结果：

```bash
python scripts/run_baseline_experiment.py --config configs/experiment/baseline/smoke.yaml
python scripts/train_algo_pool_real_sample.py --agent_name ppo --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name dqn --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name dueling_dqn --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name dag_offload_drl --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name cache_offload_drl --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name dt_handoff_drl --profile smoke
python scripts/eval_algo_pool_real_sample.py --agent_name ppo --checkpoint_path <checkpoint_path>
python scripts/benchmark_main_results.py --agents sa_ghmappo ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --seed_checkpoint_manifest_path <manifest_with_learned_checkpoints>
```

生成新正式结果后需要同步更新：

- `README.md`
- `docs/project/ARTIFACT_RECORDS.md`
- `docs/project/RUNBOOK.md`

## HF Model-Cache Transaction-Aligned Round1

状态：`[supporting]`

用途：本轮用于把 Hugging Face model-cache 审计 manifest 中的真实文件大小投影为本地 adapter cache size profile，并在 `NGSIM + Alibaba` 主线下做 Transactions model caching/offloading 口径的本地小样本对比。它不是 paper-grade 主表，也不是 HF 真实 cache request trace。

Local adaptation run：
- `artifacts/experiments/hf_model_cache_transaction_round1/hf_model_cache_transaction_round1_20260430_160913_653774/aggregate_summary.json`
- `artifacts/experiments/hf_model_cache_transaction_round1/hf_model_cache_transaction_round1_20260430_160913_653774/algorithm_comparison.csv`
- `artifacts/experiments/hf_model_cache_transaction_round1/hf_model_cache_transaction_round1_20260430_160913_653774/convergence_rewards.csv`
- `artifacts/experiments/hf_model_cache_transaction_round1/hf_model_cache_transaction_round1_20260430_160913_653774/hf_projection_mapping.csv`

Checkpoint sanity run：
- `artifacts/experiments/hf_model_cache_transaction_round1/hf_model_cache_transaction_round1_20260430_161055_695912/aggregate_summary.json`
- `artifacts/experiments/hf_model_cache_transaction_round1/hf_model_cache_transaction_round1_20260430_161055_695912/algorithm_comparison.csv`
- `artifacts/experiments/hf_model_cache_transaction_round1/hf_model_cache_transaction_round1_20260430_161055_695912/hf_projection_mapping.csv`

已确认边界：
- HF 数据只作为 file-size/cache-volume profile；不能声明为真实 VEC cache hit/miss、RSU locality、handoff demand 或 adapter state migration trace。
- 本轮单 seed、2 个 benchmark episode，不能支撑最终领先性 claim。
- 当前结果显示 `sa_ghmappo` 高于 PPO/MAPPO 和 `reactive_greedy`，但未超过 `popularity_cache_heuristic` 的 total reward 与 continuity。
- 相关 manifest 和 paper export 文件

## Top Journal Closed Loop Quick v3

状态：`[validation]`

用途：验证顶刊闭环 orchestrator、checkpoint manifest、mixed/full benchmark 和 gate report 可用；不是论文正式结论。

Run root：
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_quick_seed7_20260505_v3/`

核心产物：
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_quick_seed7_20260505_v3/seed_checkpoint_manifest.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_quick_seed7_20260505_v3/gate_report.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_quick_seed7_20260505_v3/gate_summary.csv`
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_quick_seed7_20260505_v3/benchmarks/mixed_informative/main_results_mixed_informative_20260505_030901_139250/aggregate_summary.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_quick_seed7_20260505_v3/benchmarks/full_stratified/main_results_full_stratified_20260505_030931_129078/aggregate_summary.json`

结论边界：
- quick gate 通过：mixed/full 下 SA reward 略高于 popularity，continuity/handoff/backhaul 持平。
- `paper_claim_ready=false`；该结果只有 seed 7、2-episode 训练和 tiny windows，不能用于正式论文 claim。

Latest quick v3 benchmark refresh with backhaul guard diagnostics:
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_quick_seed7_20260505_v3/benchmarks/mixed_informative/main_results_mixed_informative_20260505_031709_606215/aggregate_summary.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_quick_seed7_20260505_v3/benchmarks/full_stratified/main_results_full_stratified_20260505_031739_558338/aggregate_summary.json`

## Top Journal Closed Loop Quick v6

状态：`[validation]`

用途：验证 handoff pressure 主体绑定、cache-warm start guard、MAPPO CTDE/action mask 修复后的闭环 gate；不是论文正式结论。

Run root：
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_quick_seed7_20260505_v6/`

核心产物：
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_quick_seed7_20260505_v6/seed_checkpoint_manifest.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_quick_seed7_20260505_v6/gate_report.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_quick_seed7_20260505_v6/gate_summary.csv`
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_quick_seed7_20260505_v6/benchmarks/mixed_informative/main_results_mixed_informative_20260505_105754_448829/aggregate_summary.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_quick_seed7_20260505_v6/benchmarks/full_stratified/main_results_full_stratified_20260505_105825_548572/aggregate_summary.json`

结论边界：
- quick-plus gate 通过：mixed/full 下 SA reward 略高于 popularity，continuity/handoff/backhaul 持平。
- mixed_informative：SA total reward 57.600000，高于 popularity 57.250000；continuity 1.000000，handoff failure 0.000000，backhaul 128.000000，与 popularity 持平。
- full_stratified：SA total reward 61.273333，高于 popularity 60.906667；continuity 1.000000，handoff failure 0.000000，backhaul 133.333333，与 popularity 持平。
- `paper_claim_ready=false`；该结果只有 seed 7、2-episode 训练和 tiny windows，不能用于正式论文 claim。

## Top Journal Closed Loop Formal v2

状态：`[paper-grade]`

用途：当前顶刊主表正式闭环，使用 `NGSIM + Alibaba`、三 seed `7/13/29`、`primary_vehicle_selection=handoff_pressure`、正式训练预算和 mixed/full benchmark。

核心产物：

- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_formal_20260505_v2/gate_report.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_formal_20260505_v2/gate_summary.csv`
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_formal_20260505_v2/seed_checkpoint_manifest.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_formal_20260505_v2/benchmarks/mixed_informative/main_results_mixed_informative_20260505_131333_536820/aggregate_summary.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_formal_20260505_v2/benchmarks/full_stratified/main_results_full_stratified_20260505_131343_689261/aggregate_summary.json`

确认结果：

- `passed=true`
- `formal_contract.ready=true`
- `paper_claim_ready=true`
- mixed：SA total reward `98.330000`，popularity `98.146667`，continuity / handoff failure / backhaul 与 popularity 持平。
- full：SA total reward `90.464259`，popularity `90.171667`，continuity / handoff failure / backhaul 与 popularity 持平。

## Top Journal Support Suite Formal v2

状态：`[paper-grade]`

用途：在 formal v2 主 gate 之后补齐顶刊支撑证据包，包括 paper export、paired statistics、prediction robustness、system robustness、scalability 和 current-contract ablation。

核心入口：

- `artifacts/experiments/top_journal_support_suite/top_journal_support_formal_20260505_v2/support_gate_report.json`
- `artifacts/experiments/top_journal_support_suite/top_journal_support_formal_20260505_v2/paper/paper_claim_summary.json`
- `artifacts/experiments/top_journal_support_suite/top_journal_support_formal_20260505_v2/statistics/main_results/paired_statistics.csv`
- `artifacts/experiments/top_journal_support_suite/top_journal_support_formal_20260505_v2/prediction_robustness/prediction_robustness_20260505_155047_385240/prediction_robustness_summary.json`
- `artifacts/experiments/top_journal_support_suite/top_journal_support_formal_20260505_v2/robustness/robustness_20260505_155152/aggregate_summary.json`
- `artifacts/experiments/top_journal_support_suite/top_journal_support_formal_20260505_v2/scalability/scalability_20260505_155309/aggregate_summary.json`
- `artifacts/experiments/top_journal_support_suite/top_journal_support_formal_20260505_v2/ablation/mixed_informative/ablation_mixed_informative_20260505_174855_410627/ablation_summary.json`
- `artifacts/experiments/top_journal_support_suite/top_journal_support_formal_20260505_v2/ablation/full_stratified/ablation_full_stratified_20260505_174938_932616/ablation_summary.json`
- `artifacts/experiments/top_journal_support_suite/top_journal_ablation_formal_20260505_v2/ablation_checkpoint_manifest.json`

规模：

- 主结果 paired statistics：mixed + full 共 `360` episode rows。
- prediction robustness：`432` episode rows。
- robustness：`720` episode rows。
- scalability：`1440` episode rows。
- ablation：mixed `144` rows，full `432` rows。

结论边界：

- 支撑闭环完整：`support_suite_complete=true`。
- 主结果可声明：SA 相对 popularity 的 paired total reward delta `+0.265278`，95% bootstrap CI `[0.169444, 0.368056]`，win/tie/loss `38/34/0`。
- ablation 支持 prediction、graph encoder、hierarchy、event agent 和 adapter prefetch 的贡献。
- `no_dag_dependency_aware` 的 reward CI 跨 0，`no_uncertainty_signal` 不体现独立 reward 正贡献；不能把这两项写成单独显著 reward 来源。

## Top Journal Mechanism v3 Eval-Bias Candidate

状态：`[candidate-validation]`

用途：验证在 formal_v2 已训练权重上启用保守 inference-calibrated latency fallback 后，主方法是否能在不牺牲 continuity / handoff / backhaul guardrails 的情况下扩大相对 learned baselines 和 supplementary heuristic 的 reward 优势。

核心产物：

- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias/seed_checkpoint_manifest_v3_eval_bias_learned_baselines.json`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_learned_gate_20260505/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_learned_gate_20260505/learned_baseline_gate_summary.csv`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_learned_gate_20260505/statistics/main_and_heuristic_results/paired_statistics.csv`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_learned_gate_20260505/benchmarks/mixed_informative/main_results_mixed_informative_20260505_221107_217441/aggregate_summary.json`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_learned_gate_20260505/benchmarks/full_stratified/main_results_full_stratified_20260505_221119_729619/aggregate_summary.json`

确认结果：

- learned strict gate：`passed=true`、`formal_contract.ready=true`、`paper_claim_ready=true`。
- mixed：SA total reward `98.596667`，strongest learned baseline MAPPO `90.458333`，delta `+8.138334`。
- full：SA total reward `90.916111`，strongest learned baseline MAPPO `86.761111`，delta `+4.155000`。
- supplementary heuristic：SA 相对 `popularity_cache_heuristic` 的 paired total reward delta `+0.670833`，95% bootstrap CI `[0.545833, 0.797222]`，win/tie/loss `69/3/0`。
- mixed/full 中 SA 与 popularity 的 continuity、handoff failure、backhaul、handoff ready 和 mechanism realization 持平。

结论边界：

- 该候选使用 `scripts/build_top_journal_eval_bias_manifest.py` 从 formal_v2 权重派生 checkpoint config，属于 inference calibration 验证，不是 clean retrain。
- 在补齐独立 holdout 或支撑实验复跑前，它不能替代 `top_journal_closed_loop_formal_20260505_v2` 作为最终 paper-grade 主表。

## Top Journal Mechanism v2/v3 Negative Iterations

状态：`[negative-result]`

用途：记录未冻结的主方法优势迭代，避免后续误引用。

核心产物：

- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v2_learned_gate_20260505/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_learned_gate_20260505/learned_baseline_gate_report.json`

结论边界：

- `top_journal_mechanism_v2` learned gate 通过，但 mixed 下 SA `96.146667` 低于 popularity `98.146667`。
- clean retrain `top_journal_mechanism_v3` learned gate 通过，但 mixed 下 SA `96.474` 低于 popularity `98.146667`，full 下 SA `89.509` 低于 popularity `90.171667`。
- 这两轮不能作为主方法优于 heuristic reference 的论文结论，只保留为调参诊断。
## Top Journal Mechanism v3 Eval-Bias Guarded-Prefetch Refresh

状态：`[invalidated-baseline-duplicate-trace]`

用途：在当时代码下复跑 v3 eval-bias formal gate、offset=3 ranked-window sensitivity、latency fallback ablation、prediction robustness、robustness 和 scalability。2026-06-18 interval audit 后不得再称 independent holdout。

核心产物：

- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_gate_20260506/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_gate_20260506/statistics/main_and_heuristic_results/paired_statistics.csv`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_holdout_offset3_20260506/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_holdout_offset3_20260506/statistics/main_and_heuristic_results/paired_statistics.csv`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_support/statistics/latency_fallback_holdout_ablation_guarded_prefetch/paired_statistics.csv`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_support/prediction_robustness_guarded_prefetch/prediction_robustness_20260506_092412_434998/prediction_robustness_summary.json`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_support/robustness/robustness_20260506_091639/aggregate_summary.json`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_support/scalability/scalability_20260506_091639/aggregate_summary.json`

确认结果：

- formal learned gate：`passed=true`、`formal_contract.ready=true`、`paper_claim_ready=true`；mixed SA `98.596667` vs MAPPO `90.458333`，full SA `90.916111` vs MAPPO `86.761111`。
- holdout learned gate：`passed=true`、`window_rank_offset=3`；mixed SA `99.936667` vs MAPPO `89.516667`，full SA `93.416429` vs MAPPO `87.367857`。
- formal paired heuristic：SA vs popularity total reward delta `+0.670833`，95% CI `[0.545833, 0.797222]`，win/tie/loss `69/3/0`。
- holdout paired heuristic：SA vs popularity total reward delta `+0.61`，95% CI `[0.493333, 0.726708]`，win/tie/loss `48/12/0`。
- latency fallback holdout ablation：`sa_ghmappo_full` vs `no_latency_fallback` total reward delta `+0.385`，95% CI `[0.275, 0.5]`，win/tie/loss `32/28/0`。
- support：robustness SA `97.050764` > popularity `96.550764`；scalability SA `92.05125` > popularity `90.38875`。

结论边界：

- 该候选可作为“inference-calibrated latency fallback”强结果使用，但仍不是 clean retrain。
- prediction robustness 不能声称全面领先：汇总 SA `89.927917` 低于 popularity `90.94375`，oracle setting 仍是未闭合边界。

## Top Journal Mechanism v3 Plus-Dueling Learned-Baseline Refresh

状态：`[strong-candidate]`

用途：在 v3 guarded-prefetch candidate 上补充 Dueling-DQN / Dueling-DDQN learned baselines，减少“只对比 PPO-family 或普通 DQN-family”的审稿风险；heuristic 仍为 supplementary reference。

核心产物：
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_gate_20260506/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_gate_20260506/statistics/learned_main_results/paired_statistics.csv`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_gate_20260506/statistics/heuristic_supplementary/paired_statistics.csv`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_holdout_offset3_20260506/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_holdout_offset3_20260506/statistics/learned_main_results/paired_statistics.csv`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_holdout_offset3_20260506/statistics/heuristic_supplementary/paired_statistics.csv`

确认结果：
- formal plus-dueling gate 原始报告为 `passed=true`、`formal_contract.ready=true`、`paper_claim_ready=true`，但 learned baseline set 为 `ippo/ppo/mappo/dqn/ddqn/dueling_dqn/dueling_ddqn`，已被后续 baseline independence audit 否决；不能作为 paper-grade 主表。
- formal mixed：SA `98.596667`，strongest learned baseline `mappo` `90.458333`，delta `+8.138334`；formal full：SA `90.916111`，`mappo` `86.761111`，delta `+4.155`。
- holdout offset=3：mixed SA `99.936667` vs `mappo` `89.516667`，delta `+10.42`；full SA `93.416429` vs `mappo` `87.367857`，delta `+6.048572`。
- paired total reward：formal vs `dueling_dqn/dueling_ddqn` 为 `+35.691111`，holdout 为 `+38.441667`；bootstrap CI 均显著为正。
- supplementary heuristic：formal vs `popularity_cache_heuristic` delta `+0.670833`，holdout delta `+0.61`。

结论边界：
- 新增 dueling baselines 是 learned-baseline 完整性补充，但 `dueling_dqn` / `dueling_ddqn` 在最新 final run 中出现完全重复 trace；不能把二者同时写成独立证据。
- 按 2026-05-06/2026-05-07 当时 contract，`mappo` 被视为 diagnostic/contract-blocked baseline；这些旧 MAPPO 数值不能作为当前 controller-level CTDE MAPPO 的 paper-grade 结果。

## Top Journal Mechanism v4 Prepare Override Negative Screen

状态：`[negative-result]`

核心产物：

- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v4_prepare_eval_bias/seed_checkpoint_manifest_v4_prepare_eval_bias_learned_baselines.json`
- `artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v4_prepare_eval_bias_support/prediction_robustness/prediction_robustness_20260506_093008_229757/prediction_robustness_summary.json`

结论边界：

- v4 predictive prepare hard override 未修复 oracle prediction：prediction robustness 汇总 SA `89.45625`，popularity `90.94375`；oracle setting SA `85.163333`，popularity `93.188333`。
- 不作为主方法升级，不进入 paper-grade 主表。
## Top Journal Final Submission Learned-Primary v1

状态：`[invalidated-baseline-duplicate-trace]`

用途：最终交稿 learned-primary 闭环审计产物。主 gate 只比较 learned baselines；`reactive_greedy` 和 `popularity_cache_heuristic` 作为 supplementary heuristic reference 保留，不作为主 claim 的阻塞条件。该 run 已被 duplicate trace audit 否决，不能作为 paper-grade 主表。

核心产物：

- `artifacts/experiments/top_journal_final_submission/final_submission_clean_equal_budget_20260506_v1/final_submission_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_equal_budget_20260506_v1/learned_suites/final_submission_clean_equal_budget_20260506_v1_iter1_formal/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_equal_budget_20260506_v1/learned_suites/final_submission_clean_equal_budget_20260506_v1_iter1_formal/statistics/learned_main_results/paired_statistics.csv`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_equal_budget_20260506_v1/learned_suites/final_submission_clean_equal_budget_20260506_v1_iter1_holdout_offset3/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_equal_budget_20260506_v1/learned_suites/final_submission_clean_equal_budget_20260506_v1_iter1_holdout_offset3/statistics/learned_main_results/paired_statistics.csv`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_equal_budget_20260506_v1/support/prediction/prediction_robustness_20260506_153116_016161/prediction_robustness_summary.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_equal_budget_20260506_v1/support/robustness/robustness_20260506_154200/aggregate_summary.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_equal_budget_20260506_v1/support/scalability/scalability_20260506_154241/aggregate_summary.json`

确认结果：

- final gate：`target_reached=false`、`paper_claim_ready=false`，blockers 为 `learned_gate_failed:offset_0` 和 `learned_gate_failed:offset_3`。
- formal learned gate：reward margin 与 formal contract 本身通过，但 `baseline_independence.passed=false`；duplicate trace blockers 为 `ippo:ppo:n=72`、`ippo:mappo:n=72`、`ppo:mappo:n=72`、`dqn:ddqn:n=72`、`dueling_dqn:dueling_ddqn:n=72`。
- holdout offset=3 learned gate：reward margin 与 formal contract 本身通过，但 `baseline_independence.passed=false`；duplicate trace blockers 为 `ippo:ppo:n=60`、`ippo:mappo:n=60`、`ppo:mappo:n=60`、`dqn:ddqn:n=60`、`dueling_dqn:dueling_ddqn:n=60`。
- formal mixed：SA `98.33` vs strongest learned baseline `90.458333`，delta `+7.871667`；formal full：SA `90.464259` vs `86.761111`，delta `+3.703148`。
- holdout mixed：SA `99.614444` vs strongest learned baseline `89.516667`，delta `+10.097777`；holdout full：SA `93.004524` vs `87.367857`，delta `+5.636667`。
- formal total_reward cluster bootstrap vs MAPPO：mean delta `+4.745278`，95% CI `[2.221578, 7.209456]`。
- holdout total_reward cluster bootstrap vs MAPPO：mean delta `+6.975`，95% CI `[4.070388, 9.804966]`。

结论边界：

- `popularity_cache_heuristic` 不能再被写成主 gate 阻塞条件；它是 supplementary reference。
- `ippo`/`ppo`/`mappo` 在当前 single-wrapper learned baseline contract 下不能被写成三个独立有效对照；DQN/DDQN 与 Dueling-DQN/Dueling-DDQN 也不能在重复 trace 未解决前作为独立证据。
- prediction support 的 `learned_prediction` 和 `noisy_prediction` setting 支持主 claim；`no_prediction` 和 `oracle_prediction` 保留为诊断设置，不能写成全面预测条件优势。

## Top Journal Final Submission Repaired Baselines v1

状态：`[superseded-by-clean-retrain]`

用途：在 baseline independence 修复后，使用当前 paper-grade learned set `ppo`、`dqn`、`dueling_dqn` 重跑 final-submission learned-primary 闭环。该 run 复用了旧 final run 中等预算训练好的 learned checkpoint，已被 clean retrain run 取代，不作为当前 canonical final 主表。

核心产物：

- `artifacts/experiments/top_journal_final_submission/final_submission_repaired_baselines_20260507_v1/final_submission_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_repaired_baselines_20260507_v1/learned_suites/final_submission_repaired_baselines_20260507_v1_iter1_formal/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_repaired_baselines_20260507_v1/learned_suites/final_submission_repaired_baselines_20260507_v1_iter1_formal/statistics/learned_main_results/paired_statistics.csv`
- `artifacts/experiments/top_journal_final_submission/final_submission_repaired_baselines_20260507_v1/learned_suites/final_submission_repaired_baselines_20260507_v1_iter1_holdout_offset3/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_repaired_baselines_20260507_v1/learned_suites/final_submission_repaired_baselines_20260507_v1_iter1_holdout_offset3/statistics/learned_main_results/paired_statistics.csv`
- `artifacts/experiments/top_journal_final_submission/final_submission_repaired_baselines_20260507_v1/support/prediction/prediction_robustness_20260507_114053_709960/prediction_robustness_summary.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_repaired_baselines_20260507_v1/support/robustness/robustness_20260507_114133/aggregate_summary.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_repaired_baselines_20260507_v1/support/scalability/scalability_20260507_114247/aggregate_summary.json`

确认结果：

- final gate：`target_reached=true`、`paper_claim_ready=true`、`blockers=[]`。
- formal learned gate：`passed=true`、`baseline_contract.passed=true`、`baseline_independence.passed=true`、`blockers=[]`。
- holdout offset=3 learned gate：`passed=true`、`baseline_contract.passed=true`、`baseline_independence.passed=true`、`blockers=[]`。
- formal strongest learned baseline 为 `ppo`：mixed SA `98.33` vs PPO `90.458333`；full SA `90.464259` vs PPO `86.761111`。
- holdout strongest learned baseline 为 `ppo`：mixed SA `99.614444` vs PPO `89.516667`；full SA `93.004524` vs PPO `87.367857`。
- formal total_reward cluster bootstrap vs PPO：mean delta `+4.745278`，95% CI `[2.189981, 7.342312]`。
- holdout total_reward cluster bootstrap vs PPO：mean delta `+6.975`，95% CI `[4.126826, 9.849931]`。

结论边界：

- 本 run 复用旧 final run 中等预算训练好的 learned checkpoint，重跑的是修复后的 baseline set、benchmark、statistics 和 support gate。
- smoke 级链路中 DQN 与 Dueling-DQN 可完全重复；正式 paper-grade 判断必须继续依赖 `baseline_independence` hard gate。

## Top Journal Final Submission Clean-Retrain Repaired Baselines v1

状态：`[superseded-by-controller-mappo-qmix]`

用途：pre-MAPPO/QMIX-controller-level historical final-submission learned-primary 闭环。使用当时修复后的 paper-grade learned set `ppo`、`dqn`、`dueling_dqn`，并在 final suite 内 clean retrain 这三个 learned baselines 的 3 个 seed。该 run 在 2026-05-09 的 controller-level MAPPO/QMIX 接入前生成；当前已被 `final_submission_controller_mappo_qmix_20260509_v1` 取代。

核心产物：

- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/final_submission_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/learned_suites/final_submission_clean_retrain_repaired_baselines_20260507_v1_iter1_formal/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/learned_suites/final_submission_clean_retrain_repaired_baselines_20260507_v1_iter1_formal/statistics/learned_main_results/paired_statistics.csv`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/learned_suites/final_submission_clean_retrain_repaired_baselines_20260507_v1_iter1_holdout_offset3/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/learned_suites/final_submission_clean_retrain_repaired_baselines_20260507_v1_iter1_holdout_offset3/statistics/learned_main_results/paired_statistics.csv`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/support/prediction/prediction_robustness_20260507_123207_556011/prediction_robustness_summary.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/support/robustness/robustness_20260507_123248/aggregate_summary.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/support/scalability/scalability_20260507_123401/aggregate_summary.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/comparison_report/top_journal_comparison_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/comparison_report/top_journal_comparison_report.md`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/comparison_report/baseline_protocol_matrix.csv`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/comparison_report/support_paired_statistics.csv`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/comparison_report/paper_ready/paper_ready_main_comparison.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/comparison_report/paper_ready/paper_ready_paired_reward_statistics.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/comparison_report/paper_ready/paper_ready_support_reward_statistics.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/comparison_report/paper_ready/paper_ready_prediction_setting_audit.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/comparison_report/paper_ready/paper_ready_report.md`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/comparison_report/paper_ready/paper_ready_self_review.json`

审稿复核摘要：
- comparison report：`review_ready=true`、`paper_ready_package_ready=true`；baseline protocol matrix 按当时 contract 明确 `ippo/mappo` 为 contract-blocked diagnostic，`reactive_greedy` / `popularity_cache_heuristic` 为 supplementary reference。2026-05-09 后该 matrix 对 MAPPO/QMIX 状态已过期，不能作为当前 MAPPO/QMIX 主表结论。
- paper-ready 自审：`blocker_count=0`，`limitation_count=4`；限制项为 popularity heuristic 接近、no_prediction/oracle 不支持全面预测优势、mechanism_realization_rate 不构成独立正向优势、holdout backhaul 对 PPO 不具备正 CI。
- prediction required setting-level CI vs PPO 均为正：`learned_prediction` mean delta `+7.871667`，95% CI `[3.748194, 11.912917]`；`noisy_prediction` mean delta `+4.353889`，95% CI `[1.244903, 7.842389]`。

确认结果：

- final gate：`target_reached=true`、`paper_claim_ready=true`、`blockers=[]`。
- `formal_training_provenance.passed=true`，required agents 为 `ppo/dqn/dueling_dqn`，required seeds 为 `7/13/29`，record_count `9`。
- formal learned gate：`passed=true`、`baseline_contract.passed=true`、`baseline_independence.passed=true`、`blockers=[]`。
- holdout offset=3 learned gate：`passed=true`、`baseline_contract.passed=true`、`baseline_independence.passed=true`、`blockers=[]`。
- formal strongest learned baseline 为 `ppo`：mixed SA `98.33` vs PPO `90.458333`；full SA `90.464259` vs PPO `86.761111`。
- holdout strongest learned baseline 为 `ppo`：mixed SA `99.614444` vs PPO `89.516667`；full SA `93.004524` vs PPO `87.367857`。
- formal total_reward cluster bootstrap vs PPO：mean delta `+4.745278`，95% CI `[2.189981, 7.342312]`。
- holdout total_reward cluster bootstrap vs PPO：mean delta `+6.975`，95% CI `[4.126826, 9.849931]`。

结论边界：

- `popularity_cache_heuristic` 与 `reactive_greedy` 仍为 supplementary reference，不作为主 learned gate 阻塞条件。
- 真正 independent IPPO 与 vehicle-agent / RSU-agent full MAPPO 仍需 future multi-agent wrapper/action contract；当前已实现的是 controller-level CTDE MAPPO。

## Top Journal Final Submission Controller MAPPO/QMIX v1

状态：`[paper-grade-current]`

用途：pre-Controller-MAT / pre-DAG-cache-DT-domain-baseline canonical final-submission learned-primary 闭环。使用 `ppo`、controller-level `mappo`、`dqn`、`dueling_dqn`、controller-level `qmix` 作为 paper-grade learned baselines，并在 final suite 内 clean retrain 5 个 learned baselines 的 3 个 seed。`ippo` 仍为 diagnostic/contract-blocked；`reactive_greedy` 和 `popularity_cache_heuristic` 只作 supplementary reference。新增 `controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl` 后，含这些新增对照的主表需要重新跑 final-submission loop。

核心产物：

- `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/final_submission_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/learned_suites/final_submission_controller_mappo_qmix_20260509_v1_iter1_formal/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/learned_suites/final_submission_controller_mappo_qmix_20260509_v1_iter1_holdout_offset3/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/comparison_report/top_journal_comparison_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/comparison_report/baseline_protocol_matrix.csv`
- `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/comparison_report/paired_statistics.csv`
- `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/comparison_report/paper_ready/paper_ready_main_comparison.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/comparison_report/paper_ready/paper_ready_paired_reward_statistics.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/comparison_report/paper_ready/paper_ready_report.md`
- `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/comparison_report/paper_ready/paper_ready_self_review.json`

审稿复核摘要：

- final gate：`target_reached=true`、`paper_claim_ready=true`、`blockers=[]`。
- comparison report：`review_ready=true`、`paper_ready_package_ready=true`。
- paper-ready 自审：`blocker_count=0`、`limitation_count=4`。
- formal/holdout learned gate 均为 `passed=true`，`baseline_contract.passed=true`，`baseline_independence.passed=true`，无 duplicate trace blocker。
- protocol matrix 将 `ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix` 全部标为 primary learned comparator；`ippo` 仍为 excluded diagnostic；该 artifact 尚未覆盖 `controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl`。

确认结果：

- formal mixed：SA `98.33` vs strongest learned baseline PPO `90.458333`，delta `+7.871667`。
- formal full：SA `90.464259` vs PPO `86.761111`，delta `+3.703148`。
- holdout offset=3 mixed：SA `99.614444` vs PPO `89.516667`，delta `+10.097777`。
- holdout offset=3 full：SA `93.004524` vs PPO `87.367857`，delta `+5.636667`。
- formal total_reward cluster bootstrap：vs PPO `+4.745278`，95% CI `[2.189981, 7.342312]`；vs MAPPO `+7.009167`，95% CI `[3.056153, 10.706199]`；vs QMIX `+41.100833`，95% CI `[32.650604, 49.539073]`。
- holdout total_reward cluster bootstrap：vs PPO `+6.975`，95% CI `[4.126826, 9.849931]`；vs MAPPO `+9.636667`，95% CI `[5.475946, 13.531958]`；vs QMIX `+43.745`，95% CI `[34.501264, 53.23284]`。
- support suites 对全部 primary learned baselines 的 `total_reward` CI 均为正（15/15）；最弱项为 `scalability` vs `ppo`：mean delta `+2.159306`，95% CI `[1.435874, 2.92841]`。PPO 仍是三个 support suite 的最弱 learned 参照：prediction `+2.794583`，95% CI `[1.725233, 3.839785]`；robustness `+6.879236`，95% CI `[5.789302, 7.998894]`；scalability `+2.159306`，95% CI `[1.435874, 2.92841]`。

结论边界：

- `mappo` 和 `qmix` 是 controller-level baselines，不是 vehicle-agent / RSU-agent full MARL wrappers；新增 DAG/cache/DT 领域 baseline 尚未进入该 artifact。
- `popularity_cache_heuristic` 与 SA-GHMAPPO 非常接近，仍需作为 supplementary reference 报告，不应宣称大幅领先手写规则。
- prediction robustness 只能声明 `learned_prediction` / `noisy_prediction` 条件下优于 PPO；`no_prediction` / `oracle_prediction` 是诊断边界。
