# Algorithm Pool

更新日期：2026-05-27

用途：记录方向匹配型强化学习对照算法池的 live 接入状态、动作边界和运行入口。

## 2026-05-27 MAPPO v3 protocol

- `mappo` 仍是 controller-level CTDE baseline：flat semantic encoder、cache / execution-offload / handoff-event 三 controller actor、centralized flat semantic critic。
- 当前 paper-grade `mappo` 使用 `aggregation_reason_weighted_controller_ppo_v3`，在 aggregation-reason head-credit 基础上加入 slow / fast / event 三头的 policy credit floor、entropy credit floor 和 entropy scale，目标是降低 action-mix collapse 风险。
- `mappo_strong_audit` 是正式强对照训练 profile；learned suite、final-submission loop 和 closed-loop 入口会对 MAPPO 默认使用该 profile。
- v3 仍不使用 SA-GHMAPPO 的 graph encoder、surrogate prediction features、uncertainty-aware scaling、mechanism auxiliary loss、heuristic imitation 或 guards；它不是 vehicle-agent / RSU-agent full MAPPO wrapper。
- 旧 pre-v3/pre-head-credit MAPPO checkpoint 只保留为历史 artifact；新的 MAPPO 论文 claim 必须来自 v3 checkpoint protocol 审计通过的 final-submission package。

## 2026-05-10 MAPPO protocol

- `mappo` 是当前 controller-level CTDE baseline：flat semantic encoder、cache / execution-offload / handoff-event 三 controller actor、centralized flat semantic critic。
- 当前 paper-grade `mappo` 必须启用 aggregation-reason controller head-credit，即根据最终环境动作由哪个 controller head 主导来分配 PPO policy credit。
- 该 head-credit 是通用 MAPPO credit assignment，不使用 SA-GHMAPPO 的 graph encoder、surrogate prediction features、uncertainty-aware scaling、mechanism auxiliary loss 或 guards。
- pre-head-credit MAPPO checkpoint 只保留为历史 artifact，不再进入当前顶刊主表。

## 当前目标

算法池服务于 AI-driven VEC 主线下的公平对比，保持以下研究对象不变：

- DAG 连续工作流执行
- cross-RSU handoff / state migration
- digital twin / surrogate prediction
- adapter / parameter-sharing caching
- 多时间尺度控制

## Agent 文件规则

- `src/agents/` 只按算法分文件。
- `registry.py` 直接从算法文件导入并注册。
- PPO 和 MAPPO 分别在 `ppo_agent.py` 与 `mappo_agent.py` 中实现，不再使用 `ppo_family.py`。
- 不再保留 `src/agents/baselines/` 和 `src/agents/marl/` 分类目录。
- `sa_ghmappo_core.py` 仅作为共享 on-policy 核心保留，不作为 PPO/MAPPO family 包装层。

## 可运行算法

| agent | tier | observation | action | 当前状态 |
|---|---|---|---|---|
| `sa_ghmappo` | main method | graph + surrogate semantic encoder | semantic discrete 5 via multi-head aggregation | 可训练、可评估、可 benchmark |
| `ippo` | diagnostic | flat semantic encoder | semantic discrete 5 | 可评估/历史复核；当前不作为 paper-grade learned baseline |
| `ppo` | tier1 | flat semantic encoder | semantic discrete 5 | 可训练、可评估、可 benchmark |
| `mappo` | tier1 | flat semantic controller-level CTDE | semantic discrete 5 via cache / execution / handoff-event controllers | 可训练、可评估、可 benchmark |
| `dqn` | tier1 | flat semantic encoder | semantic discrete 5 | trainable/evaluable/benchmark value-based baseline |
| `ddqn` | tier1 | flat semantic encoder | semantic discrete 5 | trainable/evaluable/benchmark Double-DQN baseline |
| `dueling_dqn` | tier1 | flat semantic encoder | semantic discrete 5 | trainable/evaluable/benchmark Dueling-DQN baseline |
| `dueling_ddqn` | tier1 | flat semantic encoder | semantic discrete 5 | trainable/evaluable/benchmark Dueling Double-DQN baseline |
| `qmix` | tier1 | flat semantic controller-level value decomposition | semantic discrete 5 via cache / execution / handoff-event controller Q heads | 可训练、可评估、可 benchmark |
| `controller_mat` | tier1 | flat semantic controller-level transformer CTDE | semantic discrete 5 via cache / execution / handoff-event controller tokens | 可训练、可评估、可 benchmark；新增后需重跑 final loop 才能进入当前 canonical 主表 |
| `dag_offload_drl` | tier1_domain | flat semantic + DAG scalar offload features | semantic discrete 5 via cache / execution / handoff-event controller heads | 可训练、可评估、可 benchmark；DAG workflow/offloading 领域对照 |
| `cache_offload_drl` | tier1_domain | flat semantic + model/adapter cache scalar features | semantic discrete 5 via cache / execution / handoff-event controller heads | 可训练、可评估、可 benchmark；model cache/offloading 领域对照 |
| `dt_handoff_drl` | tier1_domain | flat semantic + Digital Twin handoff snapshot features | semantic discrete 5 via cache / execution / handoff-event controller heads | 可训练、可评估、可 benchmark；DT handoff/service migration 领域对照 |
| `reactive_greedy` | heuristic | semantic state info | semantic discrete 5 | 可评估、可 benchmark |
| `popularity_cache_heuristic` | heuristic | semantic state info | semantic discrete 5 | 可评估、可 benchmark |

`flat_ppo` / `flat_mappo` 只表示历史 artifact run 名称，不再作为 live agent 注册。

说明：

- `ippo` 是 independent-style PPO diagnostic baseline；当前 wrapper 是单共享决策流，不能支撑 paper-grade independent IPPO。
- `ppo` 是基础单智能体 PPO，不使用图结构、层次结构或 surrogate-specific policy head。
- `mappo` 当前是 controller-level CTDE baseline：三头 controller actor 分别处理 cache、execution/offload 和 handoff-event，critic 消费 `centralized_critic_context`。它不是 vehicle-agent / RSU-agent full MAPPO。
- `qmix` 当前是 controller-level value-decomposition baseline：三组 controller Q heads 分别处理 cache、execution/offload 和 handoff-event，并通过 centralized monotonic mixer 聚合。它不是 vehicle-agent / RSU-agent full QMIX。
- `controller_mat` 当前是 controller-level MAT-style transformer baseline：cache、execution/offload 和 handoff-event controller 作为 transformer tokens，共享 centralized critic。它不是 vehicle-agent / RSU-agent full MAT wrapper，也不使用 SA-GHMAPPO 的 graph/surrogate/guard 机制。
- `dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl` 是领域专项 learned baseline，分别衬托 DAG offloading、model/adapter cache offloading 和 Digital Twin handoff/service migration；它们不使用 SA-GHMAPPO 的 graph message passing、calibrated surrogate gate、uncertainty-aware scaling、mechanism auxiliary loss、heuristic imitation 或 guards。
- TD3 / SAC / MADDPG 仍不进入 live registry；当前 `semantic_discrete_5` 动作空间不适合强行包装成连续控制实验。

## 动作规范

- 规范模块：`src/envs/specs/action_schema.py`
- wrapper 接线：`src/envs/wrappers/gym_vec_env.py`
- 当前 action contract：`semantic_discrete_5`
- 当前不支持把 TD3 / SAC / MADDPG 强行接成连续控制实验。

## 配置

- `configs/algo/ppo.yaml`
- `configs/algo/ippo.yaml`
- `configs/algo/mappo.yaml`
- `configs/algo/dqn.yaml`
- `configs/algo/ddqn.yaml`
- `configs/algo/dueling_dqn.yaml`
- `configs/algo/dueling_ddqn.yaml`
- `configs/algo/qmix.yaml`
- `configs/algo/controller_mat.yaml`
- `configs/algo/dag_offload_drl.yaml`
- `configs/algo/cache_offload_drl.yaml`
- `configs/algo/dt_handoff_drl.yaml`
- `configs/algo/reactive_greedy.yaml`
- `configs/algo/popularity_cache_heuristic.yaml`

## 运行入口

训练 PPO：

```bash
python scripts/train_algo_pool_real_sample.py --agent_name ppo --profile smoke
```

训练 IPPO：

```bash
python scripts/train_ippo_smoke_round12.py
```

训练 MAPPO：

```bash
python scripts/train_algo_pool_real_sample.py --agent_name mappo --profile mappo_strong_audit
```

训练 QMIX：

```bash
python scripts/train_algo_pool_real_sample.py --agent_name qmix --profile smoke
```

训练 Controller-MAT：

```bash
python scripts/train_algo_pool_real_sample.py --agent_name controller_mat --profile smoke
```

训练领域专项 learned baseline：

```bash
python scripts/train_algo_pool_real_sample.py --agent_name dag_offload_drl --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name cache_offload_drl --profile smoke
python scripts/train_algo_pool_real_sample.py --agent_name dt_handoff_drl --profile smoke
```

正式单 seed baseline 训练可使用同一入口，并显式指定 benchmark-style window selection：

```bash
python scripts/train_algo_pool_real_sample.py --agent_name ppo --profile baseline_safe --episodes 48 --update_every 6 --batch_size 32 --learning_rate 1e-4 --clip_ratio 0.1 --entropy_coef 0.003 --value_coef 0.7 --random_seed 7 --mobility_source ngsim --workflow_csv_path data/raw/workflow/alibaba2018/batch_task.csv --max_mobility_rows 2500 --max_workflows 2 --workflow_selector ordered --rsu_layout auto_dominant_tight --frame_offset 0 --window_length 24 --window_selector max_handoff_candidate --window_count 3 --window_scan_stride 2 --window_mode mixed_informative --max_steps 12 --min_tasks 5 --max_tasks 20 --output_root artifacts/training/algo_pool_formal_round1
```

将 `--agent_name` 替换为 `mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`、`dag_offload_drl`、`cache_offload_drl` 或 `dt_handoff_drl`，并对 seeds `7 13 29` 重复执行。`ddqn` / `dueling_ddqn` 只有在 duplicate trace audit 通过时才作为独立补充。

评估：

```bash
python scripts/eval_algo_pool_real_sample.py --agent_name ppo --checkpoint_path artifacts/training/algo_pool/<agent>/<run_id>/checkpoints/latest.pt
python scripts/eval_algo_pool_real_sample.py --agent_name reactive_greedy
```

统一 baseline 闭环：

```bash
python scripts/run_baseline_experiment.py --config configs/experiment/baseline/smoke.yaml
```

最小 benchmark：

```bash
python scripts/benchmark_main_results.py --agents sa_ghmappo ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --sa_ghmappo_checkpoint_path <main_ckpt> --seed_checkpoint_manifest_path <manifest_with_learned_checkpoints> --seeds 7 --max_workflows 1 --window_count 1 --max_steps 3
```

`--flat_ppo_checkpoint_path` 和 `--flat_mappo_checkpoint_path` 是历史兼容参数名；正式新实验优先用 seed checkpoint manifest 管理所有 learned baseline checkpoint。

## 导出约定

训练入口写出：

- `train.csv`
- `summary.json`
- `train_summary.json`
- `checkpoints/latest.pt`

评估入口写出：

- `eval.csv`
- `summary.json`

benchmark 入口继续写出：

- `benchmark_rows.csv`
- `aggregate_summary.json`
- per-episode `*.summary.json`

baseline 闭环额外写出：

- `comparison_summary.csv`
- `comparison_summary.json`
- `comparison_summary_detailed.json`
- `comparison_summary_by_window_class.csv`
- `run_manifest.json`
- `seed_checkpoint_manifest.json`
- `command_log.json`

## 2026-05-04 顶刊路线更新

- `mappo` 当前不再与 `ppo` 使用相同 actor/action 结构：PPO 是 flat actor + independent critic；MAPPO 是 controller-level 三头 actor + centralized flat semantic critic。
- 该实现是当前 contract 下的 paper-grade controller-level CTDE MAPPO baseline；完整 vehicle-agent / RSU-agent CTDE MAPPO 仍需要后续冻结 multi-agent wrapper contract 后扩展。
- learned PPO-family policy 已在 flat action distribution 上消费 wrapper `action_mask`；`action_info` 会记录 `action_mask_applied` 和 `valid_action_count` 供训练/benchmark audit。
- 新增 `top_journal_mechanism_v1` 主方法训练 profile，作为机制稳定性复跑候选；其结果只有在多 seed benchmark 和 checkpoint audit 通过后才能写入论文主表。

## 2026-05-10 Controller-MAT 扩展

- 新增 `controller_mat`：参考 Multi-Agent Transformer 路线，但按当前项目冻结的 cache / execution-offload / handoff-event controller-agent contract 实现。
- 新 run 的 paper-grade 默认 learned set 扩展为 `ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`。
- 训练预算采用同环境交互预算，而不是照搬 Atari/SMAC 的绝对步数：同 seeds、同 NGSIM+Alibaba 窗口、同 workflow、同 episode/max-step 预算、同 formal/holdout/support gate。
- `final_submission_controller_mappo_qmix_20260509_v1` 不含 `controller_mat`，仍是 pre-Controller-MAT canonical package；含 `controller_mat` 的论文主表需要重跑 final-submission loop。

## 2026-05-10 DAG/cache/DT 领域对照扩展

- 新增 `dag_offload_drl`：参考 dependency-aware task offloading DRL 路线，按当前 controller-agent contract 使用 DAG progress、frontier、critical path、node IO 和 adapter readiness 标量，不使用主算法图消息传递。
- 新增 `cache_offload_drl`：参考 service/model caching + computation offloading DRL 路线，使用 cache occupancy、adapter readiness、cache demand 和 future load 标量，不使用主算法 surrogate/guard 机制。
- 新增 `dt_handoff_drl`：参考 Digital Twin assisted VEC offloading/service migration DRL 路线，使用 raw DT prediction snapshot，包括 predicted RSU sequence、dwell time、confidence、future load 和 boundary pressure，不使用主算法 calibrated surrogate gate 或 uncertainty-aware event scaling。
- 新 run 的 paper-grade 默认 learned set 扩展为 `ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl`。
- 训练预算继续采用同环境交互预算：同 seeds、同 NGSIM+Alibaba 窗口、同 workflow、同 episode/max-step 预算、同 formal/holdout/support gate 和 duplicate trace audit。
- `final_submission_controller_mappo_qmix_20260509_v1` 不含这些领域专项 baseline；含 DAG/cache/DT 对照的论文主表必须重跑 final-submission loop。

## 2026-05-09 顶刊对比算法池更新

- 当时 paper-grade 默认 learned set 为 `ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`。
- `mappo` 和 `qmix` 均按主方法的 cache / execution-offload / handoff-event controller 粒度实现，可在当前 `semantic_discrete_5` contract 下训练、评估和 benchmark。
- `ippo` 仍为 diagnostic；`td3`、`sac`、`maddpg` 仍需新的连续或 multi-agent action/observation contract 后才能接入。
- 既有 `final_submission_clean_retrain_repaired_baselines_20260507_v1` 是 pre-MAPPO/QMIX-controller-level 结果；当前主表若包含 `mappo` / `qmix`，必须重跑 final-submission loop。
