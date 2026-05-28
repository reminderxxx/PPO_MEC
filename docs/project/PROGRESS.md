# Progress

用途：记录已确认的阶段事实和整理动作。未验证内容不写成事实。

## 2026-05-28: v7 final-submission paper-ready package

已完成：

- 使用 `top_journal_mechanism_v7_latency_fallback_20260528_v1/seed_checkpoint_manifest.json` 作为 SA 基础，运行 final-submission loop：`final_submission_v7_latency_fallback_20260528_v1`。
- final suite clean retrain 9 个 learned baselines 的 3 个 seed：`ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl`；`formal_training_provenance.passed=true`，`record_count=27`。
- 运行 comparison package builder，生成 `comparison_report/` 和 `comparison_report/paper_ready/`。

已完成验证：

- `python scripts\run_top_journal_final_submission_loop.py --run_id final_submission_v7_latency_fallback_20260528_v1 --base_manifest_path artifacts\experiments\top_journal_closed_loop\top_journal_mechanism_v7_latency_fallback_20260528_v1\seed_checkpoint_manifest.json --force_retrain_learned --resume_training --resume_benchmark --resume_support --command_retries 2 --baseline_episodes 96 --baseline_update_every 6 --baseline_batch_size 32 --minimum_reward_delta 0.5 --holdout_offsets 3 --seeds 7 13 29 --primary_vehicle_selection handoff_pressure --window_mode_for_training full_stratified`
- `python scripts\build_top_journal_comparison_report.py --final_run_root artifacts\experiments\top_journal_final_submission\final_submission_v7_latency_fallback_20260528_v1`

关键结果：

- Final gate：`target_reached=true`、`paper_claim_ready=true`、`blockers=[]`。
- Comparison package：`review_ready=true`、`paper_ready_package_ready=true`；self-review `blocker_count=0`、`limitation_count=3`、`pass_count=15`。
- Formal split margins over strongest learned baseline `dt_handoff_drl`：mixed `+11.176111`，full `+3.377407`。
- Holdout offset=3 split margins over strongest learned baseline `dt_handoff_drl`：mixed `+8.442778`，full `+5.242143`。
- Cluster-bootstrap paired total reward CI 均为正；最弱 formal learned CI 为 vs `dt_handoff_drl` mean `+5.327083`、95% CI `[1.594094, 8.963719]`，最弱 holdout learned CI 为 vs `dt_handoff_drl` mean `+6.202333`、95% CI `[1.607076, 10.593939]`。
- Support suites 对全部 primary learned baselines 也为正：最弱 prediction vs `dt_handoff_drl` mean `+4.833472`、95% CI `[3.170913, 6.600080]`；robustness `+9.799097`、95% CI `[8.329792, 11.297618]`；scalability `+4.133380`、95% CI `[3.245373, 5.016079]`。
- Supplementary `popularity_cache_heuristic` 仍很接近：formal/holdout mixed/full reward margins 分别为 `+0.250000`、`+0.479629`、`+0.355556`、`+0.376191`。

结论边界：

- `final_submission_v7_latency_fallback_20260528_v1` 是当前 paper-ready canonical final-submission package。
- 论文表述必须保留 generated self-review 的 3 个 limitation：heuristic gap close、mechanism realization rate 不构成每个 split 的 standalone CI-positive 优势、backhaul savings 不作为 universal headline。

## 2026-05-28: SA v7 latency fallback clean-retrain formal pass

已完成：

- 新增 `top_journal_mechanism_v7_latency_fallback` profile 和 `configs/experiment/top_journal_mechanism_v7_latency_fallback.yaml`。v7 以 v6 strong-competition profile 为基线，保留 freshness / confidence-aware prefetch admission guards，并重新启用 clean-retrain latency fallback：`latency_fallback_bias_enabled=true`、`latency_fallback_bias_strength=1.20`、`latency_fallback_confidence_floor=0.62`、`latency_fallback_slow_suppression_strength=1.20`。
- `scripts/run_top_journal_closed_loop.py` 已把 v7 纳入 formal budget override：`sa_episodes=128`、`train_window_count=6`。
- `tests/test_algo_pool_contract.py` 和 `tests/test_top_journal_closed_loop.py` 已覆盖 v7 profile 参数与 formal budget。

已完成验证：

- `python -m py_compile scripts\train_sa_ghmappo_real_sample.py scripts\run_top_journal_closed_loop.py`
- `python -m pytest tests/test_algo_pool_contract.py::AlgoPoolContractTestCase::test_sa_v6_profile_is_registered_for_strong_competition tests/test_algo_pool_contract.py::AlgoPoolContractTestCase::test_sa_v7_profile_combines_v6_guards_with_latency_fallback tests/test_top_journal_closed_loop.py::test_effective_settings_honor_v6_sa_profile_budget tests/test_top_journal_closed_loop.py::test_effective_settings_honor_v7_sa_profile_budget`
- `python scripts\run_top_journal_closed_loop.py --quick --run_id top_journal_mechanism_v7_latency_fallback_quick_20260528 --seeds 7 --sa_profile top_journal_mechanism_v7_latency_fallback --mappo_baseline_profile mappo_strong_audit --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure --window_mode_for_training full_stratified`
- `python scripts\run_top_journal_closed_loop.py --run_id top_journal_mechanism_v7_latency_fallback_20260528_v1 --seeds 7 13 29 --sa_profile top_journal_mechanism_v7_latency_fallback --mappo_baseline_profile mappo_strong_audit --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure --window_mode_for_training full_stratified --sa_episodes 128 --train_window_count 6 --resume_training`
- `python scripts\analyze_mechanism_actionmix_gap.py --mixed_benchmark_dir artifacts\experiments\top_journal_closed_loop\top_journal_mechanism_v7_latency_fallback_20260528_v1\benchmarks\mixed_informative\main_results_mixed_informative_20260528_200213_838926 --full_benchmark_dir artifacts\experiments\top_journal_closed_loop\top_journal_mechanism_v7_latency_fallback_20260528_v1\benchmarks\full_stratified\main_results_full_stratified_20260528_200240_428173 --output_dir artifacts\analysis\top_journal_mechanism_v7_latency_fallback_actionmix_diagnosis_20260528`

关键结果：

- `top_journal_mechanism_v7_latency_fallback_20260528_v1`：`formal_contract.ready=true`、`baseline_protocol_audit.passed=true`、`passed=true`、`paper_claim_ready=true`。
- `mixed_informative`：SA `98.396667` vs `popularity_cache_heuristic` `98.146667`，delta `+0.250000`；continuity、handoff failure 和 backhaul 均与 popularity 持平；strongest learned baseline 为 `mappo=82.555000`。
- `full_stratified`：SA `90.651296` vs `popularity_cache_heuristic` `90.171667`，delta `+0.479629`；continuity、handoff failure 和 backhaul 均与 popularity 持平；strongest learned baseline 为 `mappo=86.142222`。
- action-mix 诊断显示主要收益来自 placement/action mix：SA 在 active/idle 非机制窗口用 `vehicle_fallback` 替换一部分 current-RSU steady execution，降低 delay reward penalty；cache/backhaul/handoff 指标未被改写。

结论边界：

- 这是 closed-loop formal pass，不是 final-submission package，也不是新的 canonical。
- 论文主结果替换前仍必须跑 final-submission/holdout/support，并生成新的 comparison report / paper-ready package；当前 canonical 仍不自动替换。

## 2026-05-27: SA v6 confidence-aware prefetch admission guard

已完成审计：

- 正式 closed-loop `top_journal_mechanism_v6_freshness_guard_20260527_v1` 已完成，`formal_contract.ready=true`，但 `passed=false`、`paper_claim_ready=false`。
- gate blocker 仍为 `sa_total_reward_not_above_popularity` 和 `benchmark_minimum_success_not_reached`：mixed SA `98.091111` vs popularity `98.146667`；full SA `90.153148` vs popularity `90.171667`。
- 诊断产物 `artifacts/analysis/top_journal_mechanism_v6_freshness_guard_actionmix_diagnosis_20260527/` 显示剩余负例集中在 `window_off246_len24_t293_316` / `j_8` / seed `13`：SA 在低置信度且 `predicted_next_rsu_id` 仍为当前 RSU 时提前 prefetch，最终 `expired_miss`；heuristic 等到后续更高置信度、next-RSU 对齐时 prefetch 并命中。

已完成维护：

- `src/agents/sa_ghmappo_core.py` 新增 `predictive_prefetch_admission_guard_*`，默认关闭；v6 profile 显式开启 `predictive_prefetch_admission_guard_enabled=true`、`predictive_prefetch_admission_min_confidence=0.55`、`predictive_prefetch_admission_require_distinct_next=true`。
- guard 只在 selected action 为 predictive prefetch、当前 adapter 已 warm、target adapter 未 warm、存在 distinct handoff target，且 prediction confidence 低于阈值并且 next-RSU / prefetch target 未对齐时触发；触发后把动作延期为 `handoff_migration_prepare`。
- 训练 summary、checkpoint 恢复、main-results benchmark rows、eval-bias manifest builder、baseline excluded-SA-mechanism 配置和 contract tests 已同步新增字段。

已完成验证：

- `python -m py_compile src\agents\sa_ghmappo_core.py src\agents\sa_ghmappo_agent.py src\evaluators\real_eval_support.py src\evaluators\main_results_support.py src\trainers\marl_on_policy_trainer.py scripts\train_sa_ghmappo_real_sample.py scripts\build_top_journal_eval_bias_manifest.py`
- `python -m pytest tests\test_algo_pool_contract.py`
- `python -m pytest tests\test_env_contract.py`
- `python -m pytest tests\test_top_journal_closed_loop.py`
- `python scripts\smoke_test.py`
- `python scripts\run_top_journal_closed_loop.py --quick --run_id top_journal_mechanism_v6_prefetch_admission_quick_20260527 --seeds 7 --sa_profile top_journal_mechanism_v6_strong_competition --mappo_baseline_profile mappo_strong_audit --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure --window_mode_for_training full_stratified`

结论边界：

- quick run 仅验证训练、checkpoint、恢复和 benchmark 消费链路；`paper_claim_ready=false`、`passed=false` 不构成论文结论。
- 该维护是低置信度预测下的 policy-side admission control，不修改 `semantic_discrete_5` action contract、环境 reward 或 formal gate；仍需后续 3-seed formal/holdout 验证才能判断是否缩小 popularity gap。

## 2026-05-27: SA v6 freshness-aware prefetch guard

已完成维护：

- `src/agents/sa_ghmappo_core.py` 为 `cache_warm_start_guard` 增加 `cache_warm_start_guard_max_prefetch_countdown`，默认 `0.0` 表示保持历史无上界行为；当显式设置为正数时，target adapter 未 warm 但 handoff countdown 超过 freshness window 时不再把 prepare 强制改成 predictive prefetch。
- `top_journal_mechanism_v6_strong_competition` profile 将该上界设为 `6.0`，与当前 `EpisodeRecorder(prefetch_validation_window=6)` 对齐，减少机制窗口中过早 prefetch 变成 `expired_miss` 的风险。
- `real_eval_support`、训练 summary、eval-bias manifest builder、配置文件和 contract tests 已同步识别该字段，保持 checkpoint 生产端/消费端一致。

已完成验证：

- `python -m py_compile src\agents\sa_ghmappo_core.py src\agents\sa_ghmappo_agent.py src\evaluators\real_eval_support.py scripts\train_sa_ghmappo_real_sample.py scripts\build_top_journal_eval_bias_manifest.py`
- `python -m pytest tests\test_algo_pool_contract.py`
- `python -m pytest tests\test_env_contract.py`
- `python -m pytest tests\test_top_journal_closed_loop.py`
- `python scripts\smoke_test.py`
- `python scripts\run_top_journal_closed_loop.py --quick --run_id top_journal_mechanism_v6_freshness_guard_quick_20260527 --seeds 7 --sa_profile top_journal_mechanism_v6_strong_competition --mappo_baseline_profile mappo_strong_audit --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure --window_mode_for_training full_stratified`

quick run 结果边界：

- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_freshness_guard_quick_20260527/` 可执行完成；训练 summary 记录 `cache_warm_start_guard_max_prefetch_countdown=6.0`。
- quick/debug run 的 `paper_claim_ready=false`、`passed=false` 符合预期；不得作为论文结果或 v6 promotion 依据。

结论边界：

- 本次是方向匹配的 policy-side freshness gating，小步修复 v6 机制窗口诊断中暴露的 prefetch timing 问题；不修改 `semantic_discrete_5` action contract、环境 reward 或 benchmark gate。
- 该代码变更本身不构成新的论文结果；仍需 formal/holdout v3 final-submission protocol 重跑后，才能判断是否替代当前 canonical。

## 2026-05-27: SA v6 action-mask 修复与 full-stratified closed-loop 重跑

已完成维护：

- `src/envs/specs/action_schema.py` 的 predictive prefetch precondition 改为优先读取 `predicted_next_rsu_by_vehicle`，并在预测序列首项仍为当前 RSU 时扫描第一个 non-current RSU，避免把后续真实 handoff target 误判为 invalid。
- `src/agents/sa_ghmappo_core.py` 的层级策略在存在有效 `action_mask` 时先在 masked env-action score 上采样/argmax，再反解为 slow/fast/event head target，避免训练和评估阶段先采样非法 head 组合再投影。
- `scripts/run_top_journal_closed_loop.py` 对 `top_journal_mechanism_v6_strong_competition` profile 使用 v6 预算默认值：`sa_episodes=128`、`train_window_count=6`；配置文件同步记录 closed-loop full-stratified 训练窗口。

已完成运行：

- `python scripts\run_top_journal_closed_loop.py --quick --run_id top_journal_mechanism_v6_masked_fulltrain_quick_20260527 --seeds 7 --sa_profile top_journal_mechanism_v6_strong_competition --mappo_baseline_profile mappo_strong_audit --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure --window_mode_for_training full_stratified`
- `python scripts\run_top_journal_closed_loop.py --run_id top_journal_mechanism_v6_masked_fulltrain_20260527_v1 --seeds 7 13 29 --sa_profile top_journal_mechanism_v6_strong_competition --mappo_baseline_profile mappo_strong_audit --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure --window_mode_for_training full_stratified --sa_episodes 128 --train_window_count 6`

核心产物：

- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_masked_fulltrain_20260527_v1/gate_report.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_masked_fulltrain_20260527_v1/gate_summary.csv`
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_masked_fulltrain_20260527_v1/seed_checkpoint_manifest.json`

关键结果：

- 修复版 run 完成且 `formal_contract.ready=true`，`baseline_protocol_audit.passed=true`；但 closed-loop gate 仍未通过：`passed=false`、`paper_claim_ready=false`。
- `action_projection_count` 和 `invalid_action_attempt_count` 在 formal benchmark 的 `mixed_informative` 与 `full_stratified` 下均为 `0.0`；旧 v6 run 的 gate total 分别为 mixed `85/85`、full `432/432`。
- `mixed_informative`：SA reward `98.091111`，`popularity_cache_heuristic` `98.146667`，差值 `-0.055556`；strongest learned baseline 为 `mappo=82.555`，SA 差值 `+15.536111`。
- `full_stratified`：SA reward `90.153148`，`popularity_cache_heuristic` `90.171667`，差值 `-0.018519`；strongest learned baseline 为 `mappo=86.142222`，SA 差值 `+4.010926`。
- full split continuity 已恢复到 heuristic 同形：SA `workflow_continuity_rate=0.927399`、`handoff_failure_rate=0.123148`，与 `popularity_cache_heuristic` 持平；旧 v6 的 full learned-side blocker `cache_offload_drl` 已不再是本轮 strongest learned baseline。
- 新 blocker 集中在 supplementary popularity 的极小 reward gap 和 mechanism success gate：mixed success `18/125=0.144`，full success `33/188=0.175532`，均未达到 benchmark minimum success gate。

结论边界：

- `top_journal_mechanism_v6_masked_fulltrain_20260527_v1` 证明 invalid action / projection 问题已修复，且 SA 在本轮重新超过所有 learned baselines。
- 该 run 仍是 negative candidate，不替换 `final_submission_full_current_baselines_20260511_v1`；在超过 `popularity_cache_heuristic` 并通过 mechanism success gate 前，不运行 v6 final-submission promotion。

## 2026-05-27: v6 强竞争 closed-loop 结果审计

已完成运行：

- `python scripts\run_top_journal_closed_loop.py --run_id top_journal_mechanism_v6_strong_competition_20260527_v1 --seeds 7 13 29 --sa_profile top_journal_mechanism_v6_strong_competition --mappo_baseline_profile mappo_strong_audit --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure`

核心产物：

- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_strong_competition_20260527_v1/gate_report.json`
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_strong_competition_20260527_v1/gate_summary.csv`
- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_strong_competition_20260527_v1/seed_checkpoint_manifest.json`

关键结果：

- 本轮 run 完成且 `formal_contract.ready=true`，MAPPO v3 的 `baseline_protocol_audit.passed=true`，`baseline_protocol_versions.mappo.head_credit_protocol=aggregation_reason_weighted_controller_ppo_v3`。
- closed-loop gate 未通过：`passed=false`、`paper_claim_ready=false`。
- `mixed_informative` blocker：`sa_total_reward_not_above_popularity`、`benchmark_minimum_success_not_reached`。SA `96.874444`，`popularity_cache_heuristic` `98.146667`，差值 `-1.272223`；SA 仍高于 strongest learned baseline `cache_offload_drl=92.024444`，差值 `+4.85`。
- `full_stratified` blocker：`sa_total_reward_not_above_cache_offload_drl`、`sa_total_reward_not_above_popularity`、`benchmark_minimum_success_not_reached`。SA `89.692037`，`popularity_cache_heuristic` `90.171667`，`cache_offload_drl` `90.168889`；SA 相对 strongest learned baseline 差值 `-0.476852`。
- v6 对 PPO 仍有明显优势：mixed `+26.327222`，full `+19.590185`；但本轮 strongest learned baseline 已变为 `cache_offload_drl`，不能再把 PPO 写成默认最强对照。
- MAPPO v3 协议可运行，但 closed-loop 结果还不是 paper-ready：mixed reward `83.435`，full reward `84.999259`，本轮 benchmark 中 prefetch count 仍为 `0.0`。

结论边界：

- `top_journal_mechanism_v6_strong_competition_20260527_v1` 是 negative candidate，不替换当前 canonical。
- 当前可写入主论文的正式结果仍是 `final_submission_full_current_baselines_20260511_v1`。
- 暂不继续运行 v6 final-submission promotion；需要先降低 SA invalid action / action projection，并修复 full split 下相对 `cache_offload_drl` 的 reward 与 continuity 弱项。

## 2026-05-27: MAPPO v3 强对照与 SA v6 候选入口

已完成维护：

- `mappo` 升级为 `aggregation_reason_weighted_controller_ppo_v3`：slow / fast / event 三个 controller head 均有 policy credit floor、entropy credit floor 和 entropy scale，降低 MAPPO action-mix collapse 风险。
- 新增 baseline profile `mappo_strong_audit`，并让 learned-baseline suite、final-submission loop 和 closed-loop 默认对 MAPPO 使用该 profile。
- `real_eval_support`、checkpoint config、learn/action info 和 comparison protocol audit 均保留并检查 v3 字段。
- 新增主算法 profile `top_journal_mechanism_v6_strong_competition` 与配置 `configs/experiment/top_journal_mechanism_v6_strong_competition.yaml`，用于后续与优化后 learned baselines 同预算重跑。

结论边界：

- 本轮只更新算法实现、训练入口、审计协议和文档；尚未重跑正式 formal/holdout final-submission benchmark。
- 现有已验证 canonical 仍是 `final_submission_full_current_baselines_20260511_v1`；新的 MAPPO v3 / SA v6 论文 claim 必须等 v6 final-submission package 通过 `paper_claim_ready=true` 后再替换。

## 2026-05-15: v5 性能/robustness 候选实验维护启动

已完成维护：

- `GymVecEnv` observation 中当前 RSU cache size 改为按 `state["primary_vehicle_id"]` 解析主车辆；缺失时才 fallback 到 `vehicles[0]`。
- `PredictorManager` 增加 oracle 请求但 oracle frames 不可用时的 audit 字段：`requested_predictor_kind`、`oracle_requested`、`oracle_available`、`oracle_fallback_to_baseline`。
- `_prediction_history` 增加长度上限，最多保留 `prediction_delay_steps + 1` 条，避免长跑累计。
- 新增 SA-GHMAPPO 训练 profile `top_journal_mechanism_v5_perf_robust` 和实验配置 `configs/experiment/top_journal_mechanism_v5_perf_robust.yaml`；保持 `semantic_discrete_5` action contract、reward 定义和 baseline contract 不变。
- 新增 contract 测试覆盖 primary vehicle observation cache、oracle fallback audit 和 prediction history trim。

已验证：

- `python -m py_compile src\envs\wrappers\gym_vec_env.py src\envs\core\predictor_manager.py scripts\train_sa_ghmappo_real_sample.py`
- `python -m pytest tests\test_env_contract.py tests\test_algo_pool_contract.py`：32 passed；仍有既有 `.pytest_cache` WinError 183 warning，不影响 contract 结果。
- `python scripts\smoke_test.py`
- `python scripts\run_top_journal_closed_loop.py --quick --run_id top_journal_mechanism_v5_perf_robust_quick_20260515 --seeds 7 --sa_profile top_journal_mechanism_v5_perf_robust --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure`：脚本链通过；quick/debug gate 不作为论文结论。
- `python scripts\run_top_journal_closed_loop.py --run_id top_journal_mechanism_v5_perf_robust_20260515_v1 --seeds 7 13 29 --sa_profile top_journal_mechanism_v5_perf_robust --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure --resume_training`：首次运行在 `train_dueling_dqn_seed_29` 出现一次 `import torch` 异常，原命令 resume 后完成；`formal_contract_ready=true`，`paper_claim_ready=false`。
- `python scripts\run_top_journal_final_submission_loop.py --run_id final_submission_v5_perf_robust_20260515_v1 --base_manifest_path artifacts\experiments\top_journal_closed_loop\top_journal_mechanism_v5_perf_robust_20260515_v1\seed_checkpoint_manifest.json --force_retrain_learned --resume_training --resume_benchmark --resume_support --command_retries 2 --baseline_episodes 96 --baseline_update_every 6 --baseline_batch_size 32 --minimum_reward_delta 0.5 --holdout_offsets 3 --seeds 7 13 29`：命令级 retry 后完成，`formal_training_provenance.passed=true`，但 `target_reached=false`、`paper_claim_ready=false`。
- `python scripts\build_top_journal_comparison_report.py --final_run_root artifacts\experiments\top_journal_final_submission\final_submission_v5_perf_robust_20260515_v1 --bootstrap_samples 5000`：完成，`review_ready=false`、`paper_ready_package_ready=false`。

关键结果：

- Final gate blockers：`offset_0:cluster_ci_not_positive:cache_offload_drl:ci95_low=-1.008212`；`offset_3:cluster_ci_not_positive:cache_offload_drl:ci95_low=-3.372274`。
- comparison self-review：`blocker_count=3`、`limitation_count=1`、`pass_count=14`。
- 主 split 相对最强 learned baseline 的 reward margin：Formal Mixed `+7.845556`，Formal Full `+0.837777`，Holdout Mixed `+1.443333`，Holdout Full `+1.360953`；最弱项低于当前 canonical promotion threshold `+3.703148`。
- support weakest all-setting reward delta：Prediction `+1.687778`、Robustness `+6.910833`、Scalability `+1.752685`；Prediction 和 Scalability 低于当前 canonical references，Robustness 略高于 `+6.879236`。

结论边界：

- v5 不通过 promotion gate，不替换当前 canonical；继续保留 `final_submission_full_current_baselines_20260511_v1` 为正式 canonical。
- 本轮 v5 记录为性能/robustness 候选失败结果；不更新 `README.md`、`ARTIFACT_RECORDS.md`、`RUNBOOK.md` 或 `DIRECTORY_STRUCTURE.md` 的 canonical 指向。

## 2026-05-12: SA-GHMAPPO 主算法 contract 闭环修补

已完成：

- `src/envs/specs/action_schema.py` 将 `ActionMaskBuilder` 从粗粒度“有当前 DAG 节点即可全放行”改为语义 precondition mask：无 distinct predicted target 时屏蔽 predictive prefetch/prepare，目标 adapter 已 warm 时屏蔽 prefetch，并输出 `action_mask_info.invalid_reasons`。
- `ActionAdapter.decode()` 不再把非法 predictive prefetch 静默 fallback 到 current RSU；非法 predictive/prepare 动作返回 no-op 控制并在 `ControlAction.metadata` 标记 `invalid_action` / `invalid_reason`。
- `src/envs/core/predictor_manager.py` 显式输出 `predictor_kind`、`predictor_interfaces_available`、`learned_predictor_attached=false`、`surrogate_claim_boundary` 和 `prediction_quality_audit`。当前默认仍是 `baseline_predictor_v2`，`learned_or_calibrated` 仅表示 calibrated baseline surrogate interface，不代表 learned predictor checkpoint。
- `src/envs/core/vec_workflow_core_env.py` 的 `metrics_protocol` 增加 action invalid、predictor audit proxy、DAG frontier / critical-path pressure、mechanism attempt / strict success / pending validation gate 字段；`mechanism_exploration_bonus` 标记为 `shaping_diagnostic`。
- `src/agents/sa_ghmappo_core.py` 在 action info 中记录 raw head action、projected action、final guarded action、projection count 和 guard delta，避免把 mask/projection/guard 效果误写成纯 learned policy。
- `scripts/train_sa_ghmappo_real_sample.py`、`src/evaluators/main_results_support.py`、`scripts/benchmark_main_results.py` 和 `scripts/build_top_journal_comparison_report.py` 已接入新增 diagnostics/report gate 字段。

已验证：

- `python -m py_compile src\agents\sa_ghmappo_core.py src\envs\core\predictor_manager.py src\envs\core\vec_workflow_core_env.py src\envs\specs\action_schema.py src\envs\specs\semantic_objects.py src\envs\wrappers\gym_vec_env.py src\trainers\marl_on_policy_trainer.py scripts\train_sa_ghmappo_real_sample.py src\evaluators\main_results_support.py scripts\benchmark_main_results.py scripts\build_top_journal_comparison_report.py`
- `python -m pytest tests\test_env_contract.py tests\test_algo_pool_contract.py`：29 passed；仍有 `.pytest_cache` WinError 183 warning，不影响 contract 结果。
- `python scripts\run_real_sample_dryrun.py --mobility_source ngsim --workflow_source alibaba --max_mobility_rows 1500 --max_workflows 3 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_selector max_handoff_candidate --window_length 24 --max_steps 12`
- `python scripts\benchmark_main_results.py --agents sa_ghmappo popularity_cache_heuristic --sa_ghmappo_checkpoint_path artifacts\training\top_journal_mechanism_v1_smoke\sa_ghmappo\sa_ghmappo_train_20260504_172243_521865_seed7\checkpoints\best_by_reward.pt --seeds 7 --max_mobility_rows 1500 --max_workflows 1 --max_steps 4 --window_count 1 --window_length 24 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_selector max_handoff_candidate --window_mode mixed_informative --output_root artifacts\benchmarks\main_results_smoke_field_audit`
- `python scripts\build_top_journal_comparison_report.py --final_run_root artifacts\experiments\top_journal_final_submission\final_submission_controller_mappo_qmix_20260509_v1 --output_dir artifacts\reports\top_journal_report_field_audit_20260512 --bootstrap_samples 50 --random_seed 7`

结论边界：

- 当前主算法仍保持 `semantic_discrete_5` contract，不声明 DAG-level target-node action，也不声明 full vehicle/RSU multi-agent wrapper。
- 当前预测层结果应写作 prediction-aware / surrogate-feature-assisted；只有接入真实 learned predictor checkpoint 后，才能写 learned surrogate。
- 机制收益优先看 `prefetch_validated_hit`、`migration_prepare_realized`、`handoff_ready`、continuity 和新增 success gate；`mechanism_exploration_bonus` 只作为 shaping/diagnostic。

## 2026-05-11: MAPPO 低分审稿风险闭环

已完成：

- `scripts/build_top_journal_comparison_report.py` 新增 action-mix 审计闭环，直接从 formal/holdout `benchmark_rows.csv` 聚合 `total_reward`、`workflow_continuity_rate`、`handoff_failure_rate`、`prefetch_action_count`、`local_exec_count`、`current_rsu_exec_count`、`migration_action_count` 等行级指标。
- 新增 `action_mix_summary.csv` 和 `mappo_action_mix_audit.csv`，并生成 paper-ready `paper_ready_mappo_action_mix_audit.csv` / `.tex`。
- 自审表新增 `MAPPO action-mix risk`：当前 canonical run 中 MAPPO 是 protocol-valid controller-level CTDE baseline，但其低分被明确归因到 action-mix collapse，而不是作为主算法优势的主要证据。
- `paper_ready_report.md` 新增 MAPPO action-mix 诊断段落和 claim boundary：若 MAPPO 弱于 PPO/DQN，论文必须报告该审计；最强 learned-baseline 主张仍锚定 PPO，不依赖 MAPPO 低分。

已验证：

- `python -m py_compile scripts\build_top_journal_comparison_report.py`
- `python scripts\build_top_journal_comparison_report.py --final_run_root artifacts\experiments\top_journal_final_submission\final_submission_full_current_baselines_20260511_v1 --bootstrap_samples 5000`
- `python -m pytest tests\test_algo_pool_contract.py tests\test_env_contract.py`

关键结果：

- 重建后的 `comparison_report/top_journal_comparison_report.json` 仍为 `review_ready=true`、`paper_ready_package_ready=true`、`blocker_count=0`；新增 MAPPO 风险后 `limitation_count=5`、`pass_count=13`。
- `mappo_action_mix_audit.csv` 共 8 条 MAPPO-vs-PPO/DQN 审计行，均为 high risk：MAPPO 在 formal/holdout mixed/full 中相对 PPO 的 reward delta 约为 `-29.25` 到 `-31.81`，continuity delta 约为 `-0.57` 到 `-0.66`，prefetch 均为 `0.0`，而 PPO prefetch 约为 `12.5` 到 `13.21`。
- 当前论文可写边界：MAPPO 可以作为当前协议有效的 controller-level CTDE 对照，但不能把 MAPPO 低分写成主算法优势的核心证据；主表强对照口径应写为 SA-GHMAPPO 相对最强 learned baseline PPO 的增益，并用 MAPPO action-mix 审计主动解释该 baseline 的弱点。

## 2026-05-11: 完整 final-submission 重跑通过

已完成：

- 新增 canonical paper-ready run：`artifacts/experiments/top_journal_final_submission/final_submission_full_current_baselines_20260511_v1/`。
- 以 `top_journal_closed_loop_formal_20260505_v2/seed_checkpoint_manifest.json` 为 base manifest，执行 `force_retrain_learned`，对 `ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl` 做同环境交互预算 clean retrain。
- formal offset 0、holdout offset 3、prediction robustness、system robustness 和 scalability support suite 均完成。
- `baseline_protocol_versions.mappo` 记录当时的 head-credit protocol：`head_credit_enabled=True`、`event_policy_credit_floor=0.05`、`event_entropy_credit_floor=0.05`、`event_advantage_blend=1.0`；2026-05-27 之后新的 MAPPO claim 需使用 v3 protocol 重跑。
- 新 comparison package：`artifacts/experiments/top_journal_final_submission/final_submission_full_current_baselines_20260511_v1/comparison_report/`，其中 `review_ready=true`、`paper_ready_package_ready=true`。

已验证：

- `python scripts\run_top_journal_final_submission_loop.py --run_id final_submission_full_current_baselines_20260511_v1 --base_manifest_path artifacts\experiments\top_journal_closed_loop\top_journal_closed_loop_formal_20260505_v2\seed_checkpoint_manifest.json --force_retrain_learned --resume_training --resume_benchmark --resume_support --command_retries 2 --baseline_episodes 96 --baseline_update_every 6 --baseline_batch_size 32 --minimum_reward_delta 0.5 --holdout_offsets 3`
- `python scripts\build_top_journal_comparison_report.py --final_run_root artifacts\experiments\top_journal_final_submission\final_submission_full_current_baselines_20260511_v1 --bootstrap_samples 5000`

可写入论文的主结果：

- `final_submission_gate_report.json`：`paper_claim_ready=true`、`target_reached=true`、`blockers=[]`。
- 自审：`blocker_count=0`、`limitation_count=4`、`pass_count=13`。
- 主表四个 split 均超过最强 learned baseline，最强 learned baseline 均为 `ppo`：Formal Mixed `+7.871667`，Formal Full `+3.703148`，Holdout Mixed `+10.097777`，Holdout Full `+5.636667`。
- paired total reward 统计对 9 个 learned baselines 全部为 positive CI；最弱项仍为 vs `ppo`：formal mean delta `+4.745278`，95% CI `[2.3372, 7.028835]`；holdout mean delta `+6.975`，95% CI `[4.155505, 9.63982]`。
- support suites 对 9 个 learned baselines 全部 positive CI；最弱项均为 vs `ppo`：Prediction `+2.794583`，95% CI `[1.725233, 3.839785]`；Robustness `+6.879236`，95% CI `[5.789302, 7.998894]`；Scalability `+2.159306`，95% CI `[1.435874, 2.92841]`。

结论边界：

- `reactive_greedy` 和 `popularity_cache_heuristic` 仍只作为 supplementary references；`popularity_cache_heuristic` 与 SA-GHMAPPO 很接近，最小 reward margin `+0.183333`，不能声称大幅超过 heuristic。
- `no_prediction` 和 `oracle_prediction` 是 diagnostic stress cases，不支撑 universal prediction dominance claim。
- `mechanism_realization_rate` 和部分 backhaul saving 不应作为单独主 claim。

## 2026-05-11: 对比表 claimability 与预算口径收紧

已完成：

- `scripts/build_top_journal_comparison_report.py` 新增 reviewer-facing `algorithm_comparison_table`，覆盖主算法、primary learned baselines、available domain baselines、optional DQN-family variants 和 supplementary heuristics。
- 对比表补齐 `algorithm_family`、`learning_type`、`contract_granularity`、`uses_sa_only_mechanisms`、`training_budget_policy`、`current_artifact_status`、`literature_anchor`、`claim_boundary` 等字段。
- 预算口径改为分层表达：claimable learned rows 使用同 NGSIM + Alibaba 环境交互预算；available rows 需要同预算重跑 final-submission loop；`ddqn` / `dueling_ddqn` 只有 duplicate-trace independence audit 通过后才能作为补充。
- `paper_ready_algorithm_comparison.tex` 的 Role/Budget/Status 改为论文可读标签，CSV/JSON 仍保留机器可审计字段。
- `paper_ready_report.md` 在 package 被 blocker 阻断时改为 `Comparison Package Audit` / `Conditional Numeric Summary`，不再输出可直接复制的 paper-ready 结论；旧 `final_submission_controller_mappo_qmix_20260509_v1` 中 MAPPO 明确标为 `pre_head_credit_protocol_missing`，只保留数值审计，不可作为当前 MAPPO claim。
- `summarize_deltas` 改为显式均值、样本标准差和 cluster sum/count bootstrap，避免 Python 3.14 `statistics.stdev` 异常，并将 bootstrap 从样本展开优化为 cluster 聚合抽样。

已验证：

- `python -m py_compile scripts\build_top_journal_comparison_report.py`
- `python -m pytest tests\test_algo_pool_contract.py`
- `python scripts\build_top_journal_comparison_report.py --final_run_root artifacts\experiments\top_journal_final_submission\final_submission_controller_mappo_qmix_20260509_v1 --output_dir artifacts\experiments\top_journal_final_submission\final_submission_controller_mappo_qmix_20260509_v1\comparison_report_mappo_head_credit_audit_fast --bootstrap_samples 1`
- `python scripts\build_top_journal_comparison_report.py --final_run_root artifacts\experiments\top_journal_final_submission\final_submission_controller_mappo_qmix_20260509_v1 --output_dir artifacts\experiments\top_journal_final_submission\final_submission_controller_mappo_qmix_20260509_v1\comparison_report_mappo_head_credit_audit_200_direct_new --bootstrap_samples 200`

结论边界：

- 当时审计产物 `comparison_report_mappo_head_credit_audit_200_direct_new` 的 `review_ready=true`，但 `paper_ready_package_ready=false`，原因是旧 MAPPO checkpoint 缺少该轮要求的 head-credit protocol 记录。
- 可声明子集仅为该 artifact 中 contract-valid 的 `ppo`、`dqn`、`dueling_dqn`、`qmix`；含当前 MAPPO、Controller-MAT、DAG/cache/DT 领域对照的正式主表仍需重新跑 final-submission loop。

## 2026-05-10: 新增 DAG/cache/DT 领域专项 learned baselines

已完成：

- 审阅当前主线后确认对照算法应围绕跨 RSU DAG workflow、adapter/model cache、handoff/state migration、Digital Twin/surrogate prediction 和多时间尺度控制，而不是继续堆泛化 MARL 名称。
- 新增 `dag_offload_drl`：flat semantic encoder + DAG progress/frontier/critical-path/node-IO/adapter-readiness scalar block + controller-level centralized critic。
- 新增 `cache_offload_drl`：flat semantic encoder + cache occupancy、adapter readiness、cache demand 和 future-load scalar block + controller-level centralized critic。
- 新增 `dt_handoff_drl`：flat semantic encoder + raw Digital Twin prediction sequence、dwell time、confidence、future load 和 boundary-pressure scalar block + controller-level centralized critic。
- 三个新增 baseline 已接入 live registry、`configs/algo/*.yaml`、checkpoint loader、算法池 contract tests、learned-baseline suite、final-submission loop、top-journal closed-loop 和 comparison report protocol matrix。
- 新 run 的默认 paper-grade learned baseline set 扩展为 `ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl`。

已验证：

- `python -m py_compile src\agents\dag_offload_agent.py src\agents\cache_offload_agent.py src\agents\dt_handoff_agent.py src\agents\registry.py src\agents\__init__.py src\evaluators\real_eval_support.py scripts\run_top_journal_learned_baseline_suite.py scripts\run_top_journal_final_submission_loop.py scripts\run_top_journal_closed_loop.py scripts\build_top_journal_comparison_report.py`
- `python -m pytest tests\test_algo_pool_contract.py`
- `python scripts\train_algo_pool_real_sample.py --agent_name dag_offload_drl --profile smoke --episodes 2 --update_every 1 --batch_size 4 --random_seed 7 --max_mobility_rows 500 --max_workflows 1 --window_length 12 --window_count 1 --window_scan_stride 2 --window_mode mixed_informative --max_steps 3 --min_tasks 5 --max_tasks 10 --primary_vehicle_selection handoff_pressure --output_root artifacts\training\algo_pool_contract_validation`
- `python scripts\train_algo_pool_real_sample.py --agent_name cache_offload_drl --profile smoke --episodes 2 --update_every 1 --batch_size 4 --random_seed 7 --max_mobility_rows 500 --max_workflows 1 --window_length 12 --window_count 1 --window_scan_stride 2 --window_mode mixed_informative --max_steps 3 --min_tasks 5 --max_tasks 10 --primary_vehicle_selection handoff_pressure --output_root artifacts\training\algo_pool_contract_validation`
- `python scripts\train_algo_pool_real_sample.py --agent_name dt_handoff_drl --profile smoke --episodes 2 --update_every 1 --batch_size 4 --random_seed 7 --max_mobility_rows 500 --max_workflows 1 --window_length 12 --window_count 1 --window_scan_stride 2 --window_mode mixed_informative --max_steps 3 --min_tasks 5 --max_tasks 10 --primary_vehicle_selection handoff_pressure --output_root artifacts\training\algo_pool_contract_validation`
- `python scripts\eval_algo_pool_real_sample.py --agent_name dag_offload_drl --checkpoint_path artifacts\training\algo_pool_contract_validation\dag_offload_drl\dag_offload_drl_train_20260510_160340_898580_seed7\checkpoints\latest.pt --max_mobility_rows 500 --max_workflows 1 --window_length 12 --max_steps 3 --min_tasks 5 --max_tasks 10 --primary_vehicle_selection handoff_pressure --output_root artifacts\eval\algo_pool_contract_validation`
- `python scripts\eval_algo_pool_real_sample.py --agent_name cache_offload_drl --checkpoint_path artifacts\training\algo_pool_contract_validation\cache_offload_drl\cache_offload_drl_train_20260510_160412_388955_seed7\checkpoints\latest.pt --max_mobility_rows 500 --max_workflows 1 --window_length 12 --max_steps 3 --min_tasks 5 --max_tasks 10 --primary_vehicle_selection handoff_pressure --output_root artifacts\eval\algo_pool_contract_validation`
- `python scripts\eval_algo_pool_real_sample.py --agent_name dt_handoff_drl --checkpoint_path artifacts\training\algo_pool_contract_validation\dt_handoff_drl\dt_handoff_drl_train_20260510_160340_898673_seed7\checkpoints\latest.pt --max_mobility_rows 500 --max_workflows 1 --window_length 12 --max_steps 3 --min_tasks 5 --max_tasks 10 --primary_vehicle_selection handoff_pressure --output_root artifacts\eval\algo_pool_contract_validation`
- `python -m pytest tests\test_env_contract.py`
- `python scripts\smoke_test.py`
- `python scripts\build_top_journal_comparison_report.py --final_run_root artifacts\experiments\top_journal_final_submission\final_submission_controller_mappo_qmix_20260509_v1 --output_dir artifacts\experiments\top_journal_final_submission\final_submission_controller_mappo_qmix_20260509_v1\comparison_report_domain_baseline_code_audit --bootstrap_samples 200`
- `python -m pytest tests`

结论边界：

- `dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl` 是领域专项 learned baselines，不是 SA-GHMAPPO 变体，也不是 vehicle-agent / RSU-agent full MARL wrappers。
- 它们不能使用 SA-GHMAPPO 的 graph message passing、calibrated surrogate gate、uncertainty-aware event scaling、mechanism auxiliary loss、heuristic imitation、continuity/backhaul/cache-warm guards。
- `final_submission_controller_mappo_qmix_20260509_v1` 不含 `controller_mat` 和这些领域 baseline；含新增对照的论文主表需要重新跑 final-submission loop。
- 本节记录代码接入状态；正式数值结论必须来自后续多 seed formal/holdout final-submission artifact。

## 2026-05-10: 新增 Controller-MAT learned baseline 接入

已完成：

- 新增 `controller_mat` learned baseline：flat semantic encoder、cache / execution-offload / handoff-event 三个 controller tokens、transformer encoder 和 centralized transformer critic。
- `controller_mat` 已接入 live registry、`configs/algo/controller_mat.yaml`、checkpoint loader、算法池 contract tests、learned-baseline suite、final-submission loop、top-journal closed-loop 和 comparison report protocol matrix。
- 新 run 的默认 paper-grade learned baseline set 扩展为 `ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`。
- 预算口径写入 learned suite / final gate：采用同环境交互预算，保持 seeds、窗口、workflow、episode/max-step、formal/holdout/support gate 和 duplicate trace audit 一致；不把 Atari/SMAC 绝对训练步数硬套到 VEC 真实数据链路。

已验证：

- `python -m py_compile src\agents\mat_agent.py src\agents\registry.py src\agents\__init__.py src\evaluators\real_eval_support.py scripts\run_top_journal_learned_baseline_suite.py scripts\run_top_journal_final_submission_loop.py scripts\run_top_journal_closed_loop.py scripts\build_top_journal_comparison_report.py`
- `python -m pytest tests\test_algo_pool_contract.py`
- `python scripts\train_algo_pool_real_sample.py --agent_name controller_mat --profile smoke --episodes 2 --update_every 1 --batch_size 4 --random_seed 7 --max_mobility_rows 500 --max_workflows 1 --window_length 12 --window_count 1 --window_scan_stride 2 --window_mode mixed_informative --max_steps 3 --min_tasks 5 --max_tasks 10 --primary_vehicle_selection handoff_pressure --output_root artifacts\training\algo_pool_contract_validation`
- `python scripts\eval_algo_pool_real_sample.py --agent_name controller_mat --checkpoint_path artifacts\training\algo_pool_contract_validation\controller_mat\controller_mat_train_20260510_144754_351672_seed7\checkpoints\latest.pt --max_mobility_rows 500 --max_workflows 1 --window_length 12 --max_steps 3 --min_tasks 5 --max_tasks 10 --primary_vehicle_selection handoff_pressure --output_root artifacts\eval\algo_pool_contract_validation`
- `python -m pytest tests\test_env_contract.py`
- `python scripts\smoke_test.py`
- `python -m pytest tests`
- `python scripts\build_top_journal_comparison_report.py --final_run_root artifacts\experiments\top_journal_final_submission\final_submission_controller_mappo_qmix_20260509_v1 --output_dir artifacts\experiments\top_journal_final_submission\final_submission_controller_mappo_qmix_20260509_v1\comparison_report_controller_mat_code_audit --bootstrap_samples 200`

结论边界：

- `controller_mat` 是 controller-level MAT-style transformer baseline，不是 vehicle-agent / RSU-agent full MAT wrapper。
- `final_submission_controller_mappo_qmix_20260509_v1` 不含 `controller_mat`，仍是 pre-Controller-MAT canonical package；含 Controller-MAT 的论文主表需要重新跑 final-submission loop。
- 本节记录代码接入状态；正式数值结论必须来自后续多 seed formal/holdout final-submission artifact。

## 2026-05-09: controller-level MAPPO/QMIX 对照恢复为可训练 baselines

已完成：

- `mappo` 从 diagnostic/contract-blocked 状态恢复为当前 contract 下可训练的 controller-level CTDE MAPPO baseline。
- MAPPO 使用 flat semantic encoder、cache / execution-offload / handoff-event 三个 controller actors，以及 centralized flat semantic critic；它不是 vehicle-agent / RSU-agent full MAPPO wrapper。
- 修复层级分支 centralized critic 的消费端：`centralized_critic=True` 时 value head 读取 `centralized_critic_context`，不再回落到普通 `critic_context`。
- MAPPO 的 event head 已进入动作聚合，能产生 handoff prepare action；同时显式关闭 SA-GHMAPPO 的 graph encoder、surrogate prediction features、uncertainty、dependency-aware features、mechanism auxiliary loss、heuristic imitation 和 policy guards。
- 新增 `qmix` controller-level value-decomposition baseline：flat semantic encoder、cache / execution-offload / handoff-event 三组 controller Q heads，以及 centralized monotonic mixer；它不是 vehicle-agent / RSU-agent full QMIX wrapper。
- `qmix` 已接入 live registry、`configs/algo/qmix.yaml`、训练/评估 checkpoint loader、comparison report protocol matrix 和算法池 contract tests。
- `scripts/train_algo_pool_real_sample.py` 的 `smoke` profile 对 replay-based baselines（DQN-family / QMIX）会自动把 `min_replay_size` 压到 smoke rollout 可触发的范围，避免 smoke 只保存 checkpoint 但不发生梯度更新；正式 `baseline_safe` 不受该规则影响。
- `scripts/run_top_journal_learned_baseline_suite.py`、`scripts/run_top_journal_final_submission_loop.py`、`scripts/run_top_journal_closed_loop.py` 和 `scripts/build_top_journal_comparison_report.py` 的默认 paper-grade learned set 更新为 `ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`；`ippo` 仍为 contract-blocked diagnostic。
- 新 final-submission run `final_submission_controller_mappo_qmix_20260509_v1` 已完成：clean retrain `ppo/mappo/dqn/dueling_dqn/qmix` 三 seed，formal + offset=3 holdout learned gate、prediction/robustness/scalability support 和 comparison report 均通过。
- 新 comparison package `artifacts/experiments/top_journal_final_submission/final_submission_controller_mappo_qmix_20260509_v1/comparison_report/` 中 `review_ready=true`、`paper_ready_package_ready=true`，自审 `blocker_count=0`、`limitation_count=4`。
- support suite 自审已从单独对 PPO 收紧为对全部 primary learned baselines（`ppo/mappo/dqn/dueling_dqn/qmix`）检查；15/15 个 support `total_reward` CI 组合均为正，最弱项为 `scalability` vs `ppo`。

已验证：

- `python -m py_compile src\agents\mappo_agent.py src\agents\registry.py src\agents\sa_ghmappo_core.py scripts\run_top_journal_learned_baseline_suite.py scripts\run_top_journal_final_submission_loop.py scripts\build_top_journal_comparison_report.py`
- `python -m py_compile scripts\train_algo_pool_real_sample.py src\agents\qmix_agent.py src\agents\registry.py src\evaluators\real_eval_support.py scripts\run_top_journal_learned_baseline_suite.py scripts\run_top_journal_final_submission_loop.py scripts\run_top_journal_closed_loop.py scripts\build_top_journal_comparison_report.py`
- `python -m pytest tests\test_algo_pool_contract.py`
- `python -m pytest tests`
- `python scripts\smoke_test.py`
- `python scripts\train_algo_pool_real_sample.py --agent_name mappo --profile smoke --episodes 2 --update_every 1 --batch_size 4 --random_seed 7 --max_mobility_rows 500 --max_workflows 1 --window_length 12 --window_count 1 --window_scan_stride 2 --window_mode mixed_informative --max_steps 3 --min_tasks 5 --max_tasks 10 --primary_vehicle_selection handoff_pressure --output_root artifacts\training\algo_pool_contract_validation`
- `python scripts\eval_algo_pool_real_sample.py --agent_name mappo --checkpoint_path artifacts\training\algo_pool_contract_validation\mappo\mappo_train_20260509_131747_668213_seed7\checkpoints\latest.pt --max_mobility_rows 500 --max_workflows 1 --window_length 12 --max_steps 3 --min_tasks 5 --max_tasks 10 --primary_vehicle_selection handoff_pressure --output_root artifacts\eval\algo_pool_contract_validation`
- `python scripts\build_top_journal_comparison_report.py --final_run_root artifacts\experiments\top_journal_final_submission\final_submission_clean_retrain_repaired_baselines_20260507_v1 --output_dir artifacts\experiments\top_journal_final_submission\final_submission_clean_retrain_repaired_baselines_20260507_v1\comparison_report_mappo_code_audit --bootstrap_samples 200`
- `python scripts\train_algo_pool_real_sample.py --agent_name qmix --profile smoke --random_seed 7 --max_mobility_rows 500 --max_workflows 1 --window_length 12 --window_count 1 --window_scan_stride 2 --window_mode mixed_informative --max_steps 3 --min_tasks 5 --max_tasks 10 --primary_vehicle_selection handoff_pressure --output_root artifacts\training\algo_pool_contract_validation`
- `python scripts\eval_algo_pool_real_sample.py --agent_name qmix --checkpoint_path artifacts\training\algo_pool_contract_validation\qmix\qmix_train_20260509_222108_508852_seed7\checkpoints\latest.pt --max_mobility_rows 500 --max_workflows 1 --window_length 12 --max_steps 3 --min_tasks 5 --max_tasks 10 --primary_vehicle_selection handoff_pressure --output_root artifacts\eval\algo_pool_contract_validation`
- `python scripts\build_top_journal_comparison_report.py --final_run_root artifacts\experiments\top_journal_final_submission\final_submission_clean_retrain_repaired_baselines_20260507_v1 --output_dir artifacts\experiments\top_journal_final_submission\final_submission_clean_retrain_repaired_baselines_20260507_v1\comparison_report_qmix_current_audit --bootstrap_samples 200`
- `python scripts\run_top_journal_final_submission_loop.py --run_id final_submission_controller_mappo_qmix_20260509_v1 --force_retrain_learned --resume_training --resume_benchmark --resume_support --command_retries 1`
- `python scripts\build_top_journal_comparison_report.py --final_run_root artifacts\experiments\top_journal_final_submission\final_submission_controller_mappo_qmix_20260509_v1 --output_dir artifacts\experiments\top_journal_final_submission\final_submission_controller_mappo_qmix_20260509_v1\comparison_report --bootstrap_samples 5000`

关键结果：

- final gate：`target_reached=true`、`paper_claim_ready=true`、`blockers=[]`。
- formal/holdout learned gate 均为 `passed=true`，`baseline_contract.passed=true`，`baseline_independence.passed=true`，无 duplicate trace blocker。
- formal mixed：SA `98.33` vs strongest learned PPO `90.458333`，delta `+7.871667`；formal full：SA `90.464259` vs PPO `86.761111`，delta `+3.703148`。
- holdout offset=3 mixed：SA `99.614444` vs PPO `89.516667`，delta `+10.097777`；holdout full：SA `93.004524` vs PPO `87.367857`，delta `+5.636667`。
- total_reward cluster bootstrap：formal vs MAPPO `+7.009167` CI `[3.056153, 10.706199]`，formal vs QMIX `+41.100833` CI `[32.650604, 49.539073]`；holdout vs MAPPO `+9.636667` CI `[5.475946, 13.531958]`，holdout vs QMIX `+43.745` CI `[34.501264, 53.23284]`。

结论边界：

- 以上 MAPPO/QMIX smoke train/eval 只证明 contract 和链路可用，不能作为论文数值结论。
- 当前 canonical final package 是 `final_submission_controller_mappo_qmix_20260509_v1`；`final_submission_clean_retrain_repaired_baselines_20260507_v1` 降为 pre-MAPPO/QMIX-controller-level historical package。
- `ippo` 仍需 future independent per-agent wrapper/action surface；不能把当前 IPPO 写成 paper-grade learned baseline。
- `td3` / `sac` / `maddpg` 仍未接入 live registry；当前 `semantic_discrete_5` contract 不支撑把它们伪装成连续控制对照。

## 2026-05-07: clean retrain repaired-baseline final loop 闭合

已完成：

- 新增 `scripts/build_top_journal_comparison_report.py`，把 final-submission gate 转成顶刊审稿口径的 comparison package：baseline protocol matrix、formal/holdout reward margins、paired mechanism statistics、support-suite statistics 和 markdown/json 报告。
- 对 canonical run `final_submission_clean_retrain_repaired_baselines_20260507_v1` 生成 `comparison_report/`，`top_journal_comparison_report.json` 中 `review_ready=true`。
- `comparison_report/paper_ready/` 已补齐可直接放入论文的 LaTeX/CSV 表格、copy-ready result statement 和作者自审报告；`top_journal_comparison_report.json` 中 `paper_ready_package_ready=true`。
- `scripts/run_top_journal_final_submission_loop.py` 新增 `formal_training_provenance` final gate：formal learned checkpoint 必须由本次 final suite 训练或从同一 run 断点恢复，外部 checkpoint 复用不再能直接 `paper_claim_ready=true`。
- 按修复后的默认 learned baseline set `ppo`、`dqn`、`dueling_dqn` 对 3 个 seed 执行 clean retrain，并重跑 formal、offset=3 holdout、prediction robustness、system robustness 和 scalability。
- `ippo` / `mappo` 继续保持 diagnostic/contract-blocked，不进入 paper-grade 主表。

核心产物：

- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/final_submission_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/comparison_report/top_journal_comparison_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/comparison_report/top_journal_comparison_report.md`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/comparison_report/paper_ready/paper_ready_main_comparison.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/comparison_report/paper_ready/paper_ready_paired_reward_statistics.tex`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/comparison_report/paper_ready/paper_ready_report.md`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/learned_suites/final_submission_clean_retrain_repaired_baselines_20260507_v1_iter1_formal/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_retrain_repaired_baselines_20260507_v1/learned_suites/final_submission_clean_retrain_repaired_baselines_20260507_v1_iter1_holdout_offset3/learned_baseline_gate_report.json`

关键结果：

- final gate：`target_reached=true`、`paper_claim_ready=true`、`blockers=[]`。
- comparison report：`review_ready=true`、`paper_ready_package_ready=true`；protocol matrix 明确 `ippo/mappo` 为 contract-blocked diagnostic，`popularity_cache_heuristic` / `reactive_greedy` 为 supplementary reference。
- paper-ready 自审：`blocker_count=0`、`limitation_count=4`，限制项为 popularity heuristic 很接近、no_prediction/oracle 不支持全面预测条件主张、mechanism_realization_rate 不是独立正向优势、holdout backhaul 对 PPO 不具备正 CI。
- `formal_training_provenance.passed=true`，required agents 为 `ppo/dqn/dueling_dqn`，required seeds 为 `7/13/29`，record_count `9`。
- formal / holdout learned gate 均为 `passed=true`，`baseline_contract.passed=true`，`baseline_independence.passed=true`，无 duplicate trace blocker。
- formal：mixed SA `98.33` vs PPO `90.458333`；full SA `90.464259` vs PPO `86.761111`。
- holdout offset=3：mixed SA `99.614444` vs PPO `89.516667`；full SA `93.004524` vs PPO `87.367857`。
- formal total_reward cluster bootstrap vs PPO：mean delta `+4.745278`，95% CI `[2.189981, 7.342312]`。
- holdout total_reward cluster bootstrap vs PPO：mean delta `+6.975`，95% CI `[4.126826, 9.849931]`。
- support：prediction aggregate SA `89.452917` > PPO `86.658333`；robustness SA `96.782708` > PPO `89.903472`；scalability SA `86.008611` > PPO `83.849306`。
- prediction required setting-level CI vs PPO 均为正：`learned_prediction` mean delta `+7.871667`，95% CI `[3.748194, 11.912917]`；`noisy_prediction` mean delta `+4.353889`，95% CI `[1.244903, 7.842389]`。

结论边界：

- 当时 canonical final-submission run 是 `final_submission_clean_retrain_repaired_baselines_20260507_v1`；当前已被 `final_submission_controller_mappo_qmix_20260509_v1` 取代。
- `final_submission_repaired_baselines_20260507_v1` 已被 clean retrain run supersede；不要把 checkpoint-reuse run 作为最终主表。
- 真正 IPPO/MAPPO 仍需未来 multi-agent wrapper/action contract；当前不能写成已实现。

## 2026-05-07: baseline independence 修复后 final-submission 闭环通过

已完成：

- `ippo` / `mappo` 在 live registry 中降级为 `diagnostic`，不再进入 paper-grade 默认 learned baseline set。
- `scripts/run_top_journal_learned_baseline_suite.py` 新增 contract-blocked baseline hard gate；显式加入 `ippo` / `mappo` 时必须使用 `--allow_contract_blocked_baselines`，且不能 `paper_claim_ready=true`。
- `scripts/run_top_journal_final_submission_loop.py` 默认 learned baselines 改为 `ppo`、`dqn`、`dueling_dqn`，support agents 同步使用 `sa_ghmappo` 加这三个 learned baselines。
- 使用旧 final run 中已等预算训练的 learned checkpoint，按修复后的默认 baseline set 重跑 formal、offset=3 holdout、prediction robustness、system robustness 和 scalability。

核心产物：

- `artifacts/experiments/top_journal_final_submission/final_submission_repaired_baselines_20260507_v1/final_submission_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_repaired_baselines_20260507_v1/learned_suites/final_submission_repaired_baselines_20260507_v1_iter1_formal/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_repaired_baselines_20260507_v1/learned_suites/final_submission_repaired_baselines_20260507_v1_iter1_holdout_offset3/learned_baseline_gate_report.json`

关键结果：

- final gate：`target_reached=true`、`paper_claim_ready=true`、`blockers=[]`。
- formal / holdout learned gate 均为 `passed=true`，`baseline_contract.passed=true`，`baseline_independence.passed=true`，无 duplicate trace blocker。
- learned baseline set：`ppo`、`dqn`、`dueling_dqn`；formal strongest learned baseline 为 `ppo`。
- formal：mixed SA `98.33` vs PPO `90.458333`，full SA `90.464259` vs PPO `86.761111`。
- holdout offset=3：mixed SA `99.614444` vs PPO `89.516667`，full SA `93.004524` vs PPO `87.367857`。
- formal total_reward cluster bootstrap vs PPO：mean delta `+4.745278`，95% CI `[2.189981, 7.342312]`。
- holdout total_reward cluster bootstrap vs PPO：mean delta `+6.975`，95% CI `[4.126826, 9.849931]`。
- support：prediction aggregate SA `89.452917` > PPO `86.658333` > DQN `70.933333` > Dueling-DQN `57.597222`；robustness SA `96.782708` > PPO `89.903472`；scalability SA `86.008611` > PPO `83.849306`。

结论边界：

- 该 run 没有重新训练 learned checkpoint；它复用 `final_submission_clean_equal_budget_20260506_v1` 中的等预算 checkpoint，修复的是 paper-grade baseline set 与 gate 口径。
- smoke 级 `contract_repair_quick_20260507` 中 DQN 与 Dueling-DQN 仍可重复，说明 duplicate trace audit 必须继续作为硬门槛；正式修复 run 已通过该审计。
- `ippo` / `mappo` 只能作为 diagnostic artifact 讨论，不能写成独立 learned baseline。

## 2026-05-06: final submission run 被 baseline independence audit 否决

已完成：

- 新增 `scripts/run_top_journal_final_submission_loop.py`，编排 learned-baseline suite、offset holdout、prediction robustness、system robustness、scalability 和 final gate report。
- `scripts/run_top_journal_learned_baseline_suite.py` 支持 `--resume_training`、`--resume_benchmark`、`--command_retries`、等预算 learned baseline 重训、minimum reward margin、cluster bootstrap statistics 和 learned-baseline duplicate trace audit。
- `scripts/analyze_top_journal_statistics.py` 支持 `--cluster_keys`，当前 total_reward 主判据使用 `seed window_id workflow_id` 做 cluster bootstrap。
- `src/data/mobility/ngsim_provider.py` 用显式 `VehicleState` 字段复制替代 `deepcopy` 返回 loaded frames，避免 Python 3.14 在长实验中偶发 runtime/deepcopy 崩溃。
- final support gate 将 `popularity_cache_heuristic` 降为 supplementary reference；prediction support 的 setting-level dominance 只要求 `learned_prediction` 和 `noisy_prediction`，`no_prediction` / `oracle_prediction` 保留为诊断设置。

核心产物：

- `artifacts/experiments/top_journal_final_submission/final_submission_clean_equal_budget_20260506_v1/final_submission_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_equal_budget_20260506_v1/learned_suites/final_submission_clean_equal_budget_20260506_v1_iter1_formal/learned_baseline_gate_report.json`
- `artifacts/experiments/top_journal_final_submission/final_submission_clean_equal_budget_20260506_v1/learned_suites/final_submission_clean_equal_budget_20260506_v1_iter1_holdout_offset3/learned_baseline_gate_report.json`

关键结果：

- final gate：`target_reached=false`，`paper_claim_ready=false`，当前 blocker 为 `learned_gate_failed:offset_0` 和 `learned_gate_failed:offset_3`。
- formal learned gate：reward margin 本身通过，但 baseline independence audit 失败；重复 trace 包括 `ippo:ppo:n=72`、`ippo:mappo:n=72`、`ppo:mappo:n=72`、`dqn:ddqn:n=72`、`dueling_dqn:dueling_ddqn:n=72`。
- holdout offset=3：reward margin 本身通过，但 baseline independence audit 失败；重复 trace 包括 `ippo:ppo:n=60`、`ippo:mappo:n=60`、`ppo:mappo:n=60`、`dqn:ddqn:n=60`、`dueling_dqn:dueling_ddqn:n=60`。
- formal learned gate 数值：mixed SA `98.33` vs strongest learned baseline `90.458333`，full SA `90.464259` vs `86.761111`。
- holdout offset=3 数值：mixed SA `99.614444` vs strongest learned baseline `89.516667`，full SA `93.004524` vs `87.367857`。
- formal total_reward cluster bootstrap vs MAPPO：mean delta `+4.745278`，95% CI `[2.221578, 7.209456]`。
- holdout total_reward cluster bootstrap vs MAPPO：mean delta `+6.975`，95% CI `[4.070388, 9.804966]`。
- support：prediction aggregate SA `89.452917` > MAPPO `86.658333`；robustness SA `96.782708` > MAPPO `89.903472`；scalability SA `86.008611` > MAPPO `83.849306`。

结论边界：

- `popularity_cache_heuristic` 只作为手写 heuristic reference，不作为主 learned-baseline gate 的阻塞条件。
- `ippo` 与 `ppo` 在当前 single-wrapper decision stream 下是同构 on-policy baseline；`mappo` 仅改变 critic context，actor 决策流仍可能与 PPO/IPPO 完全重合。当前 final run 不能作为 paper-ready learned-baseline 证据。
- DQN/DDQN 与 Dueling-DQN/Dueling-DDQN 也出现完全重复 benchmark trace；后续必须先修复或降级这些非独立 baseline，再重跑 final loop。
- prediction 的 `no_prediction` 和 `oracle_prediction` 是机制诊断设置；不能写成所有预测条件全面领先。
- 本轮 final run 使用 formal_v2 SA checkpoint 与等预算重训 learned baselines，但已被 duplicate trace audit 否决，不是可交稿闭环。

## 2026-04-27: HF model-cache 候选全集审计 round14

已完成：

- 审计 Hugging Face 上 5 个 `model-cache` 候选：`ClemSummer/qwen-model-cache`、`ClemSummer/cbow-model-cache`、`Efficient-Large-Model/imagenet-llamagen-cache`、`Kuperberg/bert-model-cache`、`amansapkota/examsathi-model-cache`。
- 将候选全集写入 `data/raw/model_cache/huggingface_model_cache_sources.json`，记录下载页、Hub 文件规模、Dataset Viewer 状态和适配性边界。
- 新增 `configs/data/hf_model_cache_integration_plan.json`，明确 HF 数据只能先作为 real file-size/cache-volume profile 接入，不能直接声明为真实 VEC cache event trace。
- 新增 `scripts/audit_hf_model_cache_sources.py`，生成 `artifacts/analysis/hf_model_cache_dataset_audit_round14/` 和 `docs/agent/hf_model_cache_dataset_audit_round14_report.md`。
- `sample_model_catalog.json` 的 `model_cache_datasets` 已覆盖上述候选全集；`validate_dataset_source_declarations.py` 现在检查所有 HF model-cache source 都在 catalog 中声明。

结论：

- 本轮仍没有把 HF 数据集作为正式 benchmark 输入。
- 4 个候选可作为真实文件大小/体量 profile 的后续 importer 输入；`amansapkota/examsathi-model-cache` 审计时没有可用模型/cache 文件，不适合当前实验。
- 现有 HF 候选不提供真实 cache hit/miss、RSU locality、handoff demand 或 adapter state migration trace；正式 benchmark 若要使用，必须先加显式 importer 和独立结果标签。

## 2026-04-27: Large-scale real dataset comparison round13

已完成：

- 在 `NGSIM + Alibaba` 真实数据链路上执行大规模对比 benchmark。
- 使用三 seed `7/13/29`，5 个 agent：`sa_ghmappo`、`reactive_greedy`、`popularity_cache_heuristic`、`ppo`、`mappo`。
- 使用 5000 条 NGSIM row、4 个 Alibaba workflow、window_length 32、max_steps 20。
- 已跑 `mixed_informative` 和 `full_stratified` 两组；full-stratified 总 episode 数为 1080，每个 agent 216。
- 生成主方法优劣报告：`docs/agent/large_scale_real_dataset_round13_report.md`。

关键结论：

- `sa_ghmappo` 在 mixed 和 full-stratified overall reward 上均为最高。
- 相比 PPO/MAPPO，`sa_ghmappo` 的 reward、success、continuity 和 adapter miss 明显更好。
- 相比 `popularity_cache_heuristic`，`sa_ghmappo` reward、backhaul 和 mechanism realization 更好，但 continuity、handoff failure、handoff ready 和 adapter miss 更弱。

说明：

- 本轮不下载 HF model-cache 文件；`ClemSummer/qwen-model-cache` 仍是 metadata-only source，不直接驱动 benchmark cache event。
- highD 原始文件仍缺失，未纳入本轮 benchmark。

## 2026-04-27: HF model-cache metadata 接入与数据源声明补齐

已完成：

- 新增 `configs/data/dataset_sources.json` 和 `docs/project/DATASET_SOURCES.md`，统一记录当前数据源名称、角色、本地路径和下载页。
- 新增 `data/raw/model_cache/huggingface_model_cache_sources.json`，metadata-only 接入 Hugging Face dataset `ClemSummer/qwen-model-cache`。
- `AdapterCatalog` 支持 `model_cache_datasets`，并在 `sample_model_catalog.json` 中声明 HF model-cache 数据源。
- `check_data_ready.py` 增加 HF model-cache metadata manifest 检查。
- 新增 `scripts/validate_dataset_source_declarations.py` 与 `tests/test_model_catalog_sources.py`，验证所有数据集声明都有名称和下载页。

说明：

- 本轮没有下载 HF 模型文件，没有替换正式 benchmark 默认 cache 行为，也没有把 controlled profile 写成真实数据集。
- `NGSIM + Alibaba` 仍是正式数据主线；HF model-cache 当前只作为 catalog/report 可追踪的外部真实数据源引用。

## 2026-04-22: AI 规范文档落地与 artifacts 记录整理

已完成：

- 建立根目录 `AGENTS.md`，作为 AI 协作硬约束入口。
- 建立 `docs/project/`，作为唯一长期维护文档目录。
- 将 artifacts 中可引用报告整理到 `docs/project/ARTIFACT_RECORDS.md`。
- 将当前稳定上下文整理到 `docs/project/CONTEXT.md`。
- 将当前限制和问题整理到 `docs/project/BUGS.md`。
- 将过期阶段文档、通用模板目录、偏离主线或小修小补产物列入清理。

整理原则：

- 保留 frozen paper protocol 相关结果。
- 保留当前主线 `NGSIM + Alibaba` 可追溯的主结果、补充表、消融、预测鲁棒性和可扩展性报告。
- 删除 toy、tmp、quickcheck、LuST micro、早期 dry-run 和阶段性调参产物。
- 删除旧文档后，只以 `docs/project/` 作为项目维护记录。

## 2026-04-22: 模型层收敛为主方法

已完成：

- `src/agents/` live registry 收敛为仅注册 `sa_ghmappo`。
- 删除对比算法实现和 baseline 专用训练评估脚本。
- 主方法训练、评估、benchmark、鲁棒性和可扩展性脚本默认只接受 `sa_ghmappo`。
- `README.md`、`RUNBOOK.md`、`CODE_MODULE_MAP.md` 和 `DECISION_LOG.md` 已同步为主方法单算法入口。
- 删除已删除对比算法专属训练目录、checkpoint 目录和 benchmark episode 明细目录。
- `ARTIFACT_RECORDS.md` 已改为当前主方法 live artifact 策略，不再把历史对比表作为当前事实来源。

说明：

- 历史混合 aggregate 和 paper 导出只作为归档快照；新论文表格需要重新生成主方法单算法结果。

## 已确认主线

- 正式主线：`NGSIM + Alibaba`
- 正式协议：`paper_protocol_v1_20260409`
- 主结果源：`artifacts/benchmarks/main_results/main_results_mixed_informative_20260415_154627_405291/aggregate_summary.json`
- 补充表源：`artifacts/benchmarks/main_results/main_results_full_stratified_20260415_154815_801060/aggregate_summary.json`
- 论文导出源：`artifacts/paper/paper_protocol_v1_20260409_rerun_20260415_ngsim_v2/`
- 以上历史结果源已降级为归档快照；当前 live 主结果待主方法单算法 benchmark 重新生成。

## 不再推进方向

- 早期 toy placeholder benchmark
- 单次 dry-run 产物作为结论来源
- `tmp_quick` / `quickcheck` 训练产物
- LuST micro 激活窗口作为正式主线
- 未进入 frozen protocol 的阶段性 reward shaping、recalibration、uncertainty stress 训练批次

## 2026-04-23: 建立方向匹配型强化学习对照算法池

已完成：

- 新增 `ActionSchema`、`ActionAdapter` 和 `ActionMaskBuilder`，将 wrapper 动作解码从 `GymVecEnv` 内部逻辑抽到独立动作适配层。
- 新增可训练对照算法 `flat_ppo` 和 `flat_mappo`，均复用现有 PPO-family rollout、checkpoint 和 metrics 链路。
- 新增 `td3`、`sac`、`maddpg`、`qmix` 注册骨架，并在配置中记录当前阻塞点。
- 新增 `configs/algo/*.yaml` 算法配置。
- 新增 `scripts/train_algo_pool_real_sample.py` 和 `scripts/eval_algo_pool_real_sample.py`，导出 `train.csv`、`eval.csv` 和 `summary.json`。
- `benchmark_main_results.py`、prediction robustness、robustness 和 scalability benchmark 已接入 `flat_ppo` / `flat_mappo` checkpoint 参数。

说明：

- 当前动作 contract 是 `semantic_discrete_5`，不自然支持 TD3 / SAC / MADDPG 的标准连续控制动作。
- 完整 MADDPG / QMIX 需要先定义稳定 multi-agent wrapper contract。
- 2026-04-23 的 `algo_pool_smoke` 仅用于链路验证，不作为论文结论。

## 2026-04-23: Baseline 训练与评测闭环落地

已完成：

- 新增 `reactive_greedy` 和 `popularity_cache_heuristic` 非学习 baseline，并注册进统一 `ALGO_REGISTRY`。
- 新增 `configs/experiment/baseline/smoke.yaml` 和 `minimal_ngsim_alibaba.yaml`。
- 新增 `scripts/run_baseline_experiment.py`，按配置调用现有训练、评估、benchmark 入口并导出 `comparison_summary.csv` / `comparison_summary.json`。
- benchmark 汇总补齐 `end_to_end_workflow_delay`、`cross_rsu_cold_start_frequency` 和 `adapter_state_migration_overhead`。
- 新增 `docs/benchmark_plan_or_baseline_plan.md`，记录算法盘点、对照矩阵和统一协议。
- 已执行 smoke baseline 闭环，结果入口为 `artifacts/experiments/baseline/baseline_smoke_20260423_110751/comparison_summary.json`。

说明：

- 本轮没有修改核心环境语义、reward、handoff 规则或主方法结构。
- TD3 / QMIX 继续保留接口和配置占位，不进入当前训练闭环。

## 2026-04-23: Baseline Formalization Round 1

已完成：

- `scripts/run_baseline_experiment.py` 支持 per-seed training、per-seed checkpoint、per-seed eval 和 seed checkpoint manifest。
- `benchmark_main_results.py` 支持 `--seed_checkpoint_manifest_path`，multi-seed benchmark 可按 seed 使用对应 baseline checkpoint。
- `comparison_summary.csv/json` 增加 per-seed train/eval/checkpoint 追踪字段。
- 新增 `comparison_summary_detailed.json`、`comparison_summary_by_window_class.csv` 和 `run_manifest.json`。
- `scripts/eval_sa_ghmappo_real_sample.py` 补齐 `eval.csv` 导出，主方法 eval artifact 与 baseline eval 产物协议对齐。
- `configs/experiment/baseline/minimal_ngsim_alibaba.yaml` 已纳入 `flat_mappo`，并跑通 seed `7` / `13` 的 round1 最小对照。
- 新增 `docs/baseline_formalization_round1.md` 记录机制差异可分性检查。

结果入口：

- `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260423_160112/comparison_summary_detailed.json`

说明：

- `popularity_cache_heuristic` 是 prediction-aware heuristic，不应被写成无预测 baseline。
- 当前 round1 已拉开 popularity heuristic 与 reactive greedy 的 predictive prefetch 差异；`sa_ghmappo` 未在机制窗口上全面优于 popularity heuristic，需后续增加窗口与训练预算。

## 2026-04-24: Formal Experiment Execution Round 1 增量推进

已完成：

- `run_baseline_experiment.py` 的 `comparison_summary.csv/json`、`comparison_summary_by_window_class.csv` 和 `run_manifest.json` 已提升为可直接追踪 `agent × seed` 的实验管理产物。
- 新 run `artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/` 已完成非 smoke baseline round1 闭环。
- `flat_ppo` 和 `flat_mappo` 已按 seed `7` / `13` 分别训练、导出 checkpoint、分别 eval，并通过 `seed_checkpoint_manifest.json` 进入 benchmark。
- `sa_ghmappo`、`reactive_greedy`、`popularity_cache_heuristic`、`flat_ppo` 和 `flat_mappo` 均已在当前 round1 配置下完成 benchmark。
- 使用同一批 per-seed checkpoint 补跑 `full_stratified` benchmark，已覆盖 `mechanism_activating`、`active_non_mechanism` 和 `idle_or_sparse` 三类窗口。
- 新增 `docs/experiment_status_round1.md`、`docs/mechanism_activation_check_round1.md` 和 `docs/experiment_runbook_round1.md`。

说明：

- 本轮未修改核心环境语义、reward、handoff、migration 或 adapter cache 规则。
- 当时 `td3`、`sac`、`maddpg`、`qmix` 仍按 skeleton / contract blocked 状态记录；2026-04-24 agent 结构收敛后已从 live registry 移除。`qmix` 已在 2026-05-09 按 controller-level contract 重新接入。

## 2026-04-24: Round1 三 seed baseline 阻塞点处理

已完成：

- `scripts/train_algo_pool_real_sample.py` 支持 `--window_selector`、`--window_mode`、`--window_count` 和 `--window_scan_stride`，flat baseline 训练可使用正式窗口选择协议。
- `flat_ppo` 和 `flat_mappo` 已补齐 seed `7` / `13` / `29` 正式训练，48 episodes、8 updates、mixed-informative windows。
- 新增三 seed manifest：`artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/seed_checkpoint_manifest_formal_round1_3seed.json`。
- 已重跑三 seed 统一 benchmark：
  - `benchmark_formal_round1_3seed/main_results_mixed_informative_20260424_190319_732417/aggregate_summary.json`
  - `benchmark_formal_round1_3seed_full_stratified/main_results_full_stratified_20260424_190503_729168/aggregate_summary.json`

说明：

- 该轮解决了 flat baselines 缺 seed `29` 且训练预算过低的问题。
- 新 flat baselines 的 continuity 已从旧 benchmark 的 `0.0` 提升到 mixed-informative 下约 `0.646`，但仍低于 `sa_ghmappo` 和 heuristic baselines。
- 当前剩余主要问题转为主方法 continuity guard / checkpoint selection 诊断，而不是 baseline artifact 不完整。

## 2026-04-24: Round1 continuity checkpoint-selection 处理

已完成：

- 新增 `docs/continuity_resolution_round1.md`。
- 新增 `seed_checkpoint_manifest_formal_round1_3seed_sa_best_continuity.json` 和 `seed_checkpoint_manifest_formal_round1_3seed_sa_best_mechanism_balanced.json`。
- 重跑三 seed `mixed_informative` benchmark，确认 `best_by_continuity` 与 `best_by_mechanism_balanced` 都可将 `sa_ghmappo` continuity 提升到 `1.000000`。
- 重跑 `best_by_continuity` 的 `full_stratified` benchmark，确认三类 window 下 `sa_ghmappo` 与 popularity heuristic continuity 持平，且 reward 更高。
- 生成 stall attribution artifact：`artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/continuity_resolution_round1/`。

说明：

- `best_by_reward` 的 mixed 主协议掉点来自 4 个 `mechanism_activating` step，均为 `predictive_next_rsu_prefetch` 在 handoff/cache target mismatch 下导致 stall。
- 本轮采用 checkpoint selection 解决 continuity 对比问题，暂不改环境语义、reward 或主方法策略实现。

## 2026-04-24: Agent 文件夹结构收敛

已完成：

- `src/agents/` 收敛为按算法分文件：`sa_ghmappo_agent.py`、`ippo_agent.py`、`ppo_agent.py`、`mappo_agent.py`、`reactive_greedy_agent.py`、`popularity_cache_heuristic_agent.py`。
- 删除 `ppo_family.py`、`off_policy_placeholders.py`、`baselines/` 和 `marl/` 分类目录。
- `registry.py` 改为直接导入算法文件；当前 live registry 不再保留 `flat_ppo` / `flat_mappo` 算法别名，这两个名称仅作为历史 artifact run 名称保留。
- TD3 / SAC / MADDPG / QMIX 从 live registry 和当时算法配置中移除，后续接入前需要先冻结匹配 contract。`qmix` 已在 2026-05-09 按 controller-level contract 重新接入。

说明：

- `src/agents/sa_ghmappo_core.py` 保留为共享 on-policy 核心，避免把训练、checkpoint 和 rollout 逻辑拆成语义级重写。
- 本轮没有修改环境语义、reward、handoff、migration 或 benchmark 指标口径。



## 2026-05-04: 顶刊路线算法 contract 修复启动

已完成：

- `mappo` 的 flat critic 不再与 `ppo` 复用完全相同输入；MAPPO 在 `centralized_critic=True` 下使用 `FlatSemanticEncoder` 输出的 `centralized_critic_context`。
- learned PPO-family flat policy 在 `act()` 与 `learn()` 中应用 `action_mask`，并将 mask audit 字段写入 `action_info`。
- 新增 `top_journal_mechanism_v1` 主方法训练 profile 和 `configs/experiment/top_journal_mechanism_v1.yaml`，覆盖机制辅助损失、慢衰减 imitation、机制窗口重采样和后期 retention。
- 新增目标 contract 测试，验证 MAPPO critic context 与 PPO 区分、flat action mask 可约束 learned policy。

说明：

- 本轮没有重跑正式训练或 benchmark；上述为代码 contract 和复跑入口修复。
- 旧 PPO/MAPPO artifact 不能直接代表新 MAPPO centralized critic training contract；顶刊主表需要按新代码重训并重跑多 seed benchmark。

## 2026-05-05: 顶刊闭环 gate 与 backhaul guard

已完成：

- 新增 `scripts/run_top_journal_closed_loop.py`，统一执行 SA-GHMAPPO 训练、PPO/MAPPO baseline 重训、seed checkpoint manifest 生成、mixed/full benchmark 和 gate report。
- gate report 显式检查 SA 相对 PPO/MAPPO、popularity heuristic 的 reward、continuity、handoff failure、backhaul 和机制兑现/ready 指标，并输出 blocker。
- `top_journal_mechanism_v1` 的机制辅助不再把 reactive `current_rsu_cache_fill` 当作机制目标强推，避免辅助损失放大 backhaul。
- 新增 `backhaul_guard_enabled`，在无预测机制信号时限制同一 adapter 的重复 reactive cache fill，并将 guard 配置写入 checkpoint、训练 summary 和 eval loader。
- `benchmark_main_results.py` 的 seed manifest loader 兼容 UTF-8 BOM，避免 PowerShell 生成 manifest 后 benchmark 读失败。
- 新增闭环 gate 单测和 backhaul guard contract 测试。

验证：

- `top_journal_closed_loop_quick_seed7_20260505_v3` quick gate 通过；mixed/full 中 SA reward 略高于 popularity，continuity/handoff/backhaul 持平。
- quick 不是论文结论；当前机制兑现仍为 0，原因是 tiny quick 窗口中有效 handoff target 缺失，正式结论仍需多 seed、长窗口、正式训练预算复跑。

## 2026-05-05: handoff pressure 主体绑定与 cache-warm 闭环

已完成：

- `VecWorkflowCoreEnv` 新增 `primary_vehicle_selection` 协议；顶刊闭环默认使用 `handoff_pressure`，把 NGSIM `max_handoff_candidate` 窗口中的 handoff 压力绑定到 workflow 主 vehicle。
- `top_journal_mechanism_v1` 新增 cache-warm start guard：当前 adapter 未 warm 时优先 current cache fill；预测目标 adapter 未 warm 时先 prefetch，再执行 prepare，避免 event prepare 覆盖 slow cache 动作。
- 训练、评估、benchmark、dry-run 与 checkpoint config 消费端均透传 `primary_vehicle_selection`，并新增 cache-warm guard 诊断字段。
- `scripts/run_top_journal_closed_loop.py` 新增 `formal_contract` 审计；只有通过正式 seed、预算、benchmark mode 和 primary vehicle selection 门槛时，才允许 `paper_claim_ready=true`。
- `scripts/run_top_journal_closed_loop.py` 新增 `--resume_training`，正式复跑中断后可以复用同一 `run_id` 下已完成 checkpoint。

验证：

- `top_journal_closed_loop_quick_seed7_20260505_v6` quick-plus gate 通过；mixed/full 中 SA reward 略高于 popularity，continuity/handoff/backhaul 持平。
- 该 quick-plus run 仍不是论文结论；正式主表需要非 quick、多 seed、正式预算复跑并满足 `formal_contract.ready=true`。

## 2026-05-05: 顶刊 formal_v2 支撑实验闭环

已完成：

- `top_journal_closed_loop_formal_20260505_v2` 已通过正式 gate：`passed=true`、`formal_contract.ready=true`、`paper_claim_ready=true`。
- 基于同一 `seed_checkpoint_manifest.json`、`primary_vehicle_selection=handoff_pressure`、三 seed `7/13/29` 和 `NGSIM + Alibaba` 主线，补齐 paper export、paired statistics、prediction robustness、system robustness、scalability 和 current-contract ablation。
- 新增 `scripts/analyze_top_journal_statistics.py`，对 benchmark rows 生成 paired bootstrap CI、win/tie/loss 和 sign-test 摘要。
- 新增 `scripts/run_top_journal_ablation_training.py`，按当前顶刊 contract 训练 7 个 SA-GHMAPPO ablation variants，并生成 per-seed ablation manifest。
- 新增 `scripts/build_top_journal_support_gate_report.py`，汇总主 gate、支撑实验完成情况和 claim 边界。
- `benchmark_prediction_robustness.py`、`benchmark_robustness.py`、`benchmark_scalability.py` 和 `benchmark_ablation.py` 已支持 per-seed checkpoint manifest / `primary_vehicle_selection` 消费端一致性。
- `export_paper_artifacts.py` 已更新为 formal_v2 baseline-aware 导出，不再只导出单算法表。
- `train_sa_ghmappo_real_sample.py` 的 checkpoint audit 对损坏中间 checkpoint 做容错记录；`update_*.pt` 全量审计改为显式 `--audit_update_checkpoints`。

核心产物：

- 主 gate：`artifacts/experiments/top_journal_closed_loop/top_journal_closed_loop_formal_20260505_v2/gate_report.json`
- 支撑 gate：`artifacts/experiments/top_journal_support_suite/top_journal_support_formal_20260505_v2/support_gate_report.json`
- paper export：`artifacts/experiments/top_journal_support_suite/top_journal_support_formal_20260505_v2/paper/`
- ablation manifest：`artifacts/experiments/top_journal_support_suite/top_journal_ablation_formal_20260505_v2/ablation_checkpoint_manifest.json`

关键结论边界：

- 主结果中 SA-GHMAPPO 相对 `popularity_cache_heuristic` 的 paired reward delta 为 `+0.265278`，95% bootstrap CI `[0.169444, 0.368056]`，win/tie/loss 为 `38/34/0`。
- ablation 支持 `no_prediction`、`no_graph_encoder`、`no_hierarchy`、`no_event_agent` 和 `no_adapter_prefetch` 的 reward 贡献。
- `no_dag_dependency_aware` 的 reward CI 跨 0，`no_uncertainty_signal` 不表现为独立 reward 正贡献；论文中不能把 DAG dependency-aware 或 uncertainty signal 单独写成显著 reward 来源。

## 2026-05-05: 顶刊 learned-baseline strict 闭环

已完成：

- 新增 `src/agents/dqn_agent.py`，注册 `dqn` 与 `ddqn`，作为当前 `semantic_discrete_5` 动作 contract 下的文献级 value-based learned baseline。
- 新增 `scripts/run_top_journal_learned_baseline_suite.py`：复用 formal_v2 的 SA/PPO/MAPPO checkpoint，只补训缺失的 `ippo`、`dqn`、`ddqn` 三 seed checkpoint，并在同一 manifest 下重跑 mixed/full 主表。
- learned-baseline gate 不再把 `reactive_greedy` 或 `popularity_cache_heuristic` 作为主通过条件；heuristic 只保留为 supplementary reference。
- `top_journal_learned_baseline_formal_20260505_v1` 原始报告通过 learned strict gate，但其包含当前已降级为 diagnostic 的 IPPO/MAPPO；在 2026-05-07 baseline independence 修复后不能继续作为 paper-grade 主表。

核心产物：

- learned gate：`artifacts/experiments/top_journal_learned_baseline_suite/top_journal_learned_baseline_formal_20260505_v1/learned_baseline_gate_report.json`
- learned manifest：`artifacts/experiments/top_journal_learned_baseline_suite/top_journal_learned_baseline_formal_20260505_v1/seed_checkpoint_manifest_learned_baselines.json`
- learned paired statistics：`artifacts/experiments/top_journal_learned_baseline_suite/top_journal_learned_baseline_formal_20260505_v1/statistics/learned_main_results/paired_statistics.csv`

关键结论：

- mixed_informative 中 strongest learned baseline 是 `mappo`，SA-GHMAPPO reward `98.33` vs `90.458333`，delta `+7.871667`。
- full_stratified 中 strongest learned baseline 是 `mappo`，SA-GHMAPPO reward `90.464259` vs `86.761111`，delta `+3.703148`。
- paired total reward delta：vs `ippo/ppo` 为 `+5.877222`，vs `mappo` 为 `+4.745278`，vs `dqn/ddqn` 为 `+21.414028`。

## 2026-05-05: 主方法优势迭代 v2/v3 诊断

已完成：

- 新增 `top_journal_mechanism_v2` / `top_journal_mechanism_v3` 主方法 profile，并在 `sa_ghmappo` checkpoint config 中接入 latency fallback 相关字段。
- `src/agents/sa_ghmappo_core.py` 新增低风险 latency fallback 机制：当前 adapter 已 warm 且无跨 RSU / handoff 预测时，允许 fast head 倾向 `vehicle_fallback`；v3 进一步在该候选状态下抑制 slow cache/prefetch/event heads，避免 low-risk 执行被 slow head 覆盖。
- 新增 `scripts/build_top_journal_eval_bias_manifest.py`，把 formal_v2 权重派生为可复现的 inference-calibrated manifest。
- 对 `top_journal_mechanism_v2`、clean retrain `top_journal_mechanism_v3` 和 `top_journal_mechanism_v3_eval_bias` 分别跑 learned-baseline gate。

关键结果：

- `top_journal_mechanism_v2_learned_gate_20260505`：learned gate 通过，但 mixed 下 SA `96.146667` 低于 popularity `98.146667`，full 下 SA `90.393889` 仅略高于 popularity `90.171667`，不能作为主结果升级。
- clean retrain `top_journal_mechanism_v3_learned_gate_20260505`：learned gate 通过，但 mixed 下 SA `96.474` 低于 popularity `98.146667`，full 下 SA `89.509` 低于 popularity `90.171667`，不能冻结。
- `top_journal_mechanism_v3_eval_bias_learned_gate_20260505`：基于 formal_v2 权重的 inference-calibrated candidate 通过 learned gate；mixed 下 SA `98.596667` vs MAPPO `90.458333`，full 下 SA `90.916111` vs MAPPO `86.761111`。
- `top_journal_mechanism_v3_eval_bias` 相对 supplementary `popularity_cache_heuristic` 的 paired total reward delta 为 `+0.670833`，95% bootstrap CI `[0.545833, 0.797222]`，win/tie/loss `69/3/0`。

结论边界：

- 当前最干净的 SA 训练来源仍是 `top_journal_closed_loop_formal_20260505_v2`，但 learned-baseline 主表必须按 2026-05-07 去重后的 `ppo/dqn/dueling_dqn` contract 重跑，不能继续引用旧 IPPO/MAPPO gate 作为 paper-grade 证据。
- `top_journal_mechanism_v3_eval_bias` 是可复现的候选增强结果，但它是 formal_v2 权重上的 inference calibration；在补齐独立 holdout/support suite 前，不应替代 formal_v2 写成最终 paper-grade 主表。

## 2026-05-06: v3 eval-bias holdout 与支撑闭环筛选

已完成：

- `benchmark_main_results.py`、`run_top_journal_learned_baseline_suite.py` 和 `benchmark_ablation.py` 新增 `--window_rank_offset`，用于在同一窗口排序协议下按 strata 跳过前 N 个窗口，构造独立 holdout。
- `src/agents/sa_ghmappo_core.py` 的 continuity guard 改为目标 cache ready 前不压制 predictive prefetch；新增可关闭的 `predictive_prepare_hard_override_*` 配置，用于后续候选筛选，默认关闭。
- 新增 `scripts/build_top_journal_eval_bias_ablation_manifest.py`，生成 `sa_ghmappo_full` vs `no_latency_fallback` 的机制消融 manifest。
- 当前代码下复跑 `top_journal_mechanism_v3_eval_bias_guarded_prefetch_gate_20260506` 和 `top_journal_mechanism_v3_eval_bias_guarded_prefetch_holdout_offset3_20260506`，formal 与 holdout learned gate 均通过。
- 补充 v3 support：latency fallback holdout ablation、prediction robustness、system robustness 和 scalability。

关键结果：

- formal windows：mixed SA `98.596667` vs MAPPO `90.458333`；full SA `90.916111` vs MAPPO `86.761111`。
- offset=3 holdout：mixed SA `99.936667` vs MAPPO `89.516667`；full SA `93.416429` vs MAPPO `87.367857`。
- formal + full paired：SA vs `popularity_cache_heuristic` total reward delta `+0.670833`，95% CI `[0.545833, 0.797222]`，win/tie/loss `69/3/0`。
- holdout paired：SA vs `popularity_cache_heuristic` total reward delta `+0.61`，95% CI `[0.493333, 0.726708]`，win/tie/loss `48/12/0`。
- latency fallback holdout ablation：`sa_ghmappo_full` vs `no_latency_fallback` total reward delta `+0.385`，95% CI `[0.275, 0.5]`，win/tie/loss `32/28/0`。
- robustness：SA `97.050764` > popularity `96.550764` > MAPPO `89.903472`。
- scalability：SA `92.05125` > popularity `90.38875` > MAPPO `88.635417`。

结论边界：

- v3 eval-bias 已从“只过 formal windows 的候选”推进到“formal + independent holdout + 机制消融成立”的强候选，但仍是 formal_v2 权重上的 inference calibration，不是 clean retrain。
- prediction robustness 汇总中 SA `89.927917` 低于 popularity `90.94375`，主要由 `oracle_prediction` setting 拖累；该项不能写成全面优于 oracle/heuristic upper-bound。
- v4 prepare hard-override 筛选失败：prediction robustness 中 SA `89.45625` 仍低于 popularity `90.94375`，oracle setting SA `85.163333` vs popularity `93.188333`，不推广。

## 2026-05-06: 顶刊 learned-baseline 扩展到 Dueling-DQN 系列

已完成：

- `src/agents/dqn_agent.py` 扩展为 DQN-family baseline 文件，新增 `dueling_dqn` 与 `dueling_ddqn`，保留旧 `dqn` / `ddqn` checkpoint 的网络层名兼容。
- `src/agents/registry.py`、`src/evaluators/real_eval_support.py`、`scripts/run_top_journal_learned_baseline_suite.py` 和 `tests/test_algo_pool_contract.py` 已接入新增 learned baselines。
- 新增 `configs/algo/dqn.yaml`、`ddqn.yaml`、`dueling_dqn.yaml` 和 `dueling_ddqn.yaml`，记录 value-based baseline contract。
- 在 v3 guarded-prefetch candidate 上补训 `dueling_dqn` / `dueling_ddqn` 三 seed checkpoint，并重跑 formal 与 offset=3 holdout learned-baseline gate。

核心产物：

- formal plus-dueling gate：`artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_gate_20260506/learned_baseline_gate_report.json`
- holdout plus-dueling gate：`artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_holdout_offset3_20260506/learned_baseline_gate_report.json`
- formal learned paired statistics：`artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_gate_20260506/statistics/learned_main_results/paired_statistics.csv`
- holdout learned paired statistics：`artifacts/experiments/top_journal_sa_iteration/top_journal_mechanism_v3_eval_bias_guarded_prefetch_plus_dueling_holdout_offset3_20260506/statistics/learned_main_results/paired_statistics.csv`

关键结果：

- formal plus-dueling gate 原始报告为 `paper_claim_ready=true`、`formal_contract.ready=true`、`passed=true`，但该结果已因 IPPO/MAPPO contract-blocked 与 DQN-family duplicate trace 风险降级为 diagnostic，不再是 paper-grade 证据。
- formal mixed：SA `98.596667` vs `mappo` `90.458333`，delta `+8.138334`；formal full：SA `90.916111` vs `mappo` `86.761111`，delta `+4.155`。
- holdout offset=3 mixed：SA `99.936667` vs `mappo` `89.516667`，delta `+10.42`；holdout full：SA `93.416429` vs `mappo` `87.367857`，delta `+6.048572`。
- paired total reward：formal vs `dueling_dqn/dueling_ddqn` 为 `+35.691111`，holdout 为 `+38.441667`；两者 95% CI 均显著为正。

结论边界：

- 新增 dueling baselines 用于强化 learned-baseline 对照，不改变 heuristic 的 supplementary reference 定位。
- `dueling_dqn` / `dueling_ddqn` 在当前训练预算下弱于 DQN/DDQN；论文中应作为完整性补充，不应把它们包装成 strongest baseline。
# 2026-05-10: MAPPO head-credit 公平性审查与修正

本轮回应审稿风险：旧结果中 `ppo` 在 reward 上强于 `mappo`，而主算法是 MAPPO 路线增强，容易被质疑为 MAPPO 对照被弱化。

修正：

- `src/agents/mappo_agent.py` 的 `mappo` baseline 继续保持 flat semantic encoder + controller-level CTDE critic，不引入 SA-GHMAPPO 的 graph/surrogate/guard 机制。
- `mappo` 启用通用 `head_credit_enabled=True`，按 `aggregation_reason` 将 PPO policy credit 分给实际控制环境动作的 cache / execution / handoff-event head；这是 controller-level MAPPO 公平 credit assignment，不是主算法专属机制。
- `configs/algo/mappo.yaml`、`tests/test_algo_pool_contract.py`、`src/evaluators/real_eval_support.py`、`scripts/run_top_journal_learned_baseline_suite.py`、`scripts/run_top_journal_final_submission_loop.py`、`scripts/run_top_journal_closed_loop.py`、`scripts/build_top_journal_comparison_report.py` 同步记录和审计新版 MAPPO checkpoint protocol。

结论边界：

- `final_submission_controller_mappo_qmix_20260509_v1` 是 pre-MAPPO-head-credit package；其中 MAPPO 结果只作历史归档，不再作为当前顶刊主表的 MAPPO 强对照。
- 新的顶刊主表必须重跑 final-submission loop，并要求 `baseline_protocol_versions.mappo` 记录当前 MAPPO v3 protocol。
- 若新版 MAPPO 变强导致 SA-GHMAPPO margin 变小，应以新版结果为准；不能为突出主算法而保留弱 MAPPO。

已验证：

- `python -m py_compile src\agents\mappo_agent.py src\agents\registry.py src\evaluators\real_eval_support.py scripts\run_top_journal_learned_baseline_suite.py scripts\run_top_journal_final_submission_loop.py scripts\run_top_journal_closed_loop.py scripts\build_top_journal_comparison_report.py`
- `python -m pytest tests\test_algo_pool_contract.py`
- `python scripts\train_algo_pool_real_sample.py --agent_name mappo --profile smoke --episodes 2 --update_every 1 --batch_size 4 --random_seed 7 --max_mobility_rows 500 --max_workflows 1 --window_length 12 --window_count 1 --window_scan_stride 2 --window_mode mixed_informative --max_steps 3 --min_tasks 5 --max_tasks 10 --primary_vehicle_selection handoff_pressure --output_root artifacts\training\algo_pool_contract_validation`
- `python scripts\eval_algo_pool_real_sample.py --agent_name mappo --checkpoint_path artifacts\training\algo_pool_contract_validation\mappo\mappo_train_20260510_215921_523181_seed7\checkpoints\latest.pt --max_mobility_rows 500 --max_workflows 1 --window_length 12 --max_steps 3 --min_tasks 5 --max_tasks 10 --primary_vehicle_selection handoff_pressure --output_root artifacts\evals\algo_pool_contract_validation`
- `python scripts\build_top_journal_comparison_report.py --final_run_root artifacts\experiments\top_journal_final_submission\final_submission_controller_mappo_qmix_20260509_v1 --output_dir artifacts\experiments\top_journal_final_submission\final_submission_controller_mappo_qmix_20260509_v1\comparison_report_mappo_head_credit_audit --bootstrap_samples 200`
- `python -m pytest tests\test_env_contract.py`
- `python scripts\smoke_test.py`
- `python -m pytest tests`

验证结论：新版 MAPPO smoke checkpoint 通过协议审计；旧 `final_submission_controller_mappo_qmix_20260509_v1` 生成的新 comparison audit 为 `paper_ready_package_ready=false`，确认旧 pre-head-credit MAPPO 不再作为当前 paper-ready package。
