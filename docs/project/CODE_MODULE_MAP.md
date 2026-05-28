# Code Module Map

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
- `src/envs/core/predictor_manager.py`：统一输出 `baseline`、`oracle`、`learned_or_calibrated`、`no_prediction` 的 predictor kind 和 runtime proxy audit；当前默认不是 learned predictor。
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

- `src/evaluators/main_results_support.py` 的 `resolve_window_candidates()` 支持 `window_rank_offset`，按 stratum 跳过 ranked prefix，用于 formal window 之外的 holdout。
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
