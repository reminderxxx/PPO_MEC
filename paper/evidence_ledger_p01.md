# P01 论文证据总账

## 1. 审查元数据

| 字段 | 值 |
|---|---|
| `reviewed_at` | `2026-09-14` |
| `literature_cutoff` | `2026-06-21`（最近一次有审计记录的 novelty cutoff；文献表虽更新至 2026-08-19，但 P01 未重新联网核验） |
| `target_venue` | `IEEE Transactions on Mobile Computing (TMC)` |
| `artifact_run_id` | `typed_model_cache_evaluation_only_20260913_g14r20_i3_pending` |
| `policy_version` | `tmc_review_policy_v3_20260621` |
| scientific Git commit | `a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d` |
| executor Git commit | `a028ea291941a484ae9cd2e316d1b52adde3d1f2` |
| frozen protocol | `2.9.0`, semantic SHA-256 `c059ae03eb93323d14c6768c6867156c0f6c030307a80bfcb88eda8d6169a968` |
| model source run | `typed_model_cache_formal_20260906_152847_g14c_v16` |
| evidence level | `E0_UNAVAILABLE`（论文结果包级；已有 raw cell 仅可索引，完整 formal/statistics/gate 缺失） |
| verdict | `Unverifiable` |

`Unverifiable` 的含义是当前无法验证论文级数值 claim，不等于算法无效。P01 不执行评分，因为正式统计、support、scalability、gate 和 completion 尚未产生，holdout 未开启。

## 2. 证据优先级

发生冲突时按以下顺序采用证据：

1. 冻结 scientific checkout 中的源代码与 protocol 2.9 manifest；
2. immutable model-source reference、selected checkpoint 及其实际配置；
3. 当前 evaluation-only run 的 phase/cell ledger、committed cell 产物与独立 failure report；
4. 长期项目文档；
5. 历史 method report、旧 benchmark 摘要和探索性记录。

历史文档不能覆盖冻结 checkpoint 的实际开关，也不能用旧结果补齐当前 formal 缺口。

## 3. 权威位置索引

### 3.1 科学实现与协议

- Scientific checkout：`/Users/howen/Projects/PPO_MEC/artifacts/execution_checkouts/g14r20_i_scientific_a6d1fd8`
- Executor checkout：`/Users/howen/Projects/PPO_MEC/artifacts/execution_checkouts/g14r20_i3_executor_release`
- Active protocol manifest：`.../configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906/protocol_v2_9_manifest.json`
- Agent core：`.../src/agents/sa_ghmappo_core.py`
- Environment action adapter：`.../src/envs/specs/action_schema.py`
- Runtime transaction：`.../src/runtime/typed_model_cache_runtime.py`
- Formal exogenous request execution：`.../src/runtime/formal_exogenous_request_execution.py`
- Reward implementation：`.../src/envs/core/vec_workflow_core_env.py`
- Endpoint recomputation：`.../src/metrics/cache_efficiency_metrics.py`

### 3.2 冻结模型

- Model source run：`/Users/howen/Projects/PPO_MEC/artifacts/experiments/typed_model_cache_formal/typed_model_cache_formal_20260906_152847_g14c_v16`
- Model source reference：current evaluation-only run 下 `evaluation_model_source_reference.json`；其 `source_run_root` 指向上述 model source run。
- 模型矩阵：10 个 learned agent × 5 seeds × 3 capacities = 150 个 immutable selected models。
- 示例 selected checkpoint：`training/sa_ghmappo/formal_constrained_288mb_sa_ghmappo_seed7/checkpoints/update_0012.pt`，SHA-256 `440dd4ae5496da152f0b95c173e8023536a00d63e0b1ca15dea1129ef03ae18b`。

### 3.3 当前 evaluation-only 产物

- Run root：`/Users/howen/Projects/PPO_MEC/artifacts/experiments/typed_model_cache_evaluation_only/typed_model_cache_evaluation_only_20260913_g14r20_i3_pending`
- Phase ledger：`phase_state.jsonl`
- Cell ledger：`cell_state.jsonl`
- Failure report：`/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14e03_host_retry_authorization_20260913/formal_ablation_failure_report.json`
- Continuation lock：位于 evaluation-only 根目录的 `.continuation_locks/`，P01 未读取后修改、未清理、未续跑。

## 4. 已核验的系统与方法事实

| 主题 | 可写事实 | 关键边界 | 证据状态 |
|---|---|---|---|
| 控制拓扑 | 每 step 一个 controller call，输出一个 `semantic_discrete_5` action。 | 不是 vehicle-level / RSU-level full MARL。 | 代码与协议已核验 |
| 三类角色头 | slow/cache、fast/execution、event/handoff 三头，共享 encoder，层级条件化。 | role heads 不是独立 agents。 | 代码与 checkpoint config 已核验 |
| 动作 | 当前 RSU fill、预测下一 RSU prefetch、vehicle fallback、当前 RSU steady offload、handoff migration prepare。 | action 4 同时包含目标预取/准备和当前稳态执行语义。 | action adapter 已核验 |
| typed cache | base model + adapter 共同决定 readiness，adapter 依赖 base。 | 不能把 adapter hit 单独写成 full-service ready。 | catalog/protocol/runtime 已核验 |
| workflow state | 有 transfer size，`counts_toward_capacity=false`，只用于迁移。 | 不是长期 cache object。 | catalog/protocol 已核验 |
| KV prefix | `enabled=false`。 | 不得声称 KV cache 结果。 | protocol 已核验 |
| admission 顺序 | cache-changing action 在 same-step lookup 之前执行。 | same-step hit 仍可能产生 transfer；hit 高不等于 transfer 低。 | runtime/formal execution 已核验 |
| encoder | DAG 10-d node feature、2 轮 predecessor/successor mean message passing；RSU 10-d、vehicle 10-d、prediction 13-d。 | 只描述冻结实现，不延伸为通用 GNN 新颖性。 | source + checkpoint 已核验 |
| predictor | baseline trajectory/boundary short-horizon predictor；confidence 是 heuristic composition。 | 不是 learned predictor，不是 calibrated probability，不是 full digital twin。 | predictor source/protocol 已核验 |
| PPO | role-level clipped PPO；selected config `clip_ratio=0.2`。 | 不能用旧 profile 中与 checkpoint 不一致的值。 | checkpoint/source 已核验 |
| reward | service + continuity + exploration − delay − miss − migration − constraint。 | shaping reward 不是论文 endpoint。 | environment source 已核验 |
| primary endpoints | 6 个冻结 endpoint。 | `latency_saved` 与 counterfactual regret 当前 unavailable。 | protocol endpoint schema 已核验 |
| oracle | report-only exact H=1/3/6/12 cells。 | 只对 exact cell；不是无条件全局上界。 | protocol 已核验 |

## 5. 冻结 checkpoint 的开关核对

### 5.1 可进入主方法描述的 enabled 项

- `encoder_kind=graph`
- `centralized_critic=true`
- `hierarchical_conditioning=true`
- `use_hierarchy=true`
- `use_prediction_features=true`
- `use_uncertainty_signal=true`
- `use_dependency_aware=true`
- `event_head_enabled=true`
- `adapter_prefetch_enabled=true`
- graph/RSU/vehicle/prediction feature encoders
- baseline predictor

### 5.2 必须排除出主方法 claim 的 disabled/inert 项

- `graph_continuity_critic_enabled=false`
- `uncertainty_aware_event_scaling_enabled=false`
- `uncertainty_aware_critic_enabled=false`
- `head_credit_enabled=false`
- `mechanism_logit_bias_strength=0`
- `continuity_guard_enabled=false`
- `handoff_target_alignment_guard_enabled=false`
- heuristic imitation coefficient = 0
- mechanism auxiliary coefficient path虽记录 `0.06`，但 `_compute_auxiliary_loss` 返回零，实际 inert
- digital-twin planner、option gate 和后续 advanced mechanisms 均未进入本冻结模型主路径

历史 `docs/project/sa_ghmappo_paper_method_report_20260716.md` 包含部分后来设计机制和旧 strict-full v8 结果口径。它可作为历史背景，但不能作为当前 frozen paper method 或结果证据。

## 6. 协议事实

### 6.1 数据与对象

- Mobility：NGSIM I-80。
- Workflow：Alibaba 2018 batch DAG，冻结 workflow IDs 为 `j_3`, `j_8`, `j_15`。
- Model/cache：controlled typed catalog；2 个 base models、8 个 adapters、1 个 migration-only workflow state。
- 正确口径：`NGSIM + Alibaba DAG + controlled typed catalog` 的组合实验。
- 禁止口径：joint real-world VEC model-cache trace、真实车联网 adapter request trace。

### 6.2 窗口、容量与训练

- split：24 train、12 dev、12 formal、12 sealed holdout windows。
- 每窗口 24 frames；episode 最多 22 steps。
- capacity：288 / 576 / 864 MB。
- seeds：7 / 13 / 29 / 43 / 71。
- 每 agent-seed-capacity：256 episodes，最多 5632 environment interactions，batch 64，32 updates，每 4 updates checkpoint。
- selection：只用 dev，finite-before-null lexicographic；formal/holdout 禁止参与选择。

### 6.3 统计

- 外层独立单位：raw-time mobility window `(source_segment_run_id, window_id)`。
- seed、workflow、row 是 nested observations，不是独立 outer clusters。
- hierarchical bootstrap：10,000 replicates，seed 1401；percentile 95% 与 BCa 95%。
- effect size：paired Cohen `d_z` 与 outer-window standardized mean delta。
- paired test：exact two-sided sign test，按 `1e-9` 预注册容差删除数值 ties。
- multiplicity：6 个 primary endpoints × 所有预注册 primary comparisons 同一 Holm family，`alpha=0.05`。
- zero pair：effect/CI/test/Holm 均为 null，不能记成 tie/pass/fail。
- formal 与 holdout 独立性必须按原始 frame/time interval 检查，window ID 或 rank offset 不足以证明互斥。

## 7. 当前正式产物状态

| Phase | 状态 | Committed cells | 行/episodes | outer windows | 可用于论文结论？ |
|---|---:|---:|---:|---:|---|
| `formal_cache_policy` | completed | 3 | 8100 | 12 | 仅可索引 raw/aggregate 文件；统计未完成，不能 claim |
| `formal_controller` | completed | 3 | 8100 | 12 | 仅可索引 raw/aggregate 文件；统计未完成，不能 claim |
| `formal_ablation` | failed | 0 | 0 committed | 0 committed | 不可用 |
| `formal_support` | not started | 0 | 0 | 0 | 不可用 |
| `formal_scalability` | not started | 0 | 0 | 0 | 不可用 |
| `formal_statistics` | not started | 0 | 0 | 0 | 不可用 |
| `formal_gate` | not started | 0 | 0 | 0 | 不可用 |
| `complete_without_holdout` | not started | 0 | 0 | 0 | 不可用 |

总计 6/22 expected committed cells。两张已完成 phase 表来自同一组 12 个 outer windows，不能把 8100 + 8100 写成 16,200 个独立样本。

`formal_cache_policy` 三个 aggregate 入口：

- `constrained_288mb/benchmark/main_results_mixed_informative_20260913_135111_044450/aggregate_summary.json`
- `medium_576mb/benchmark/main_results_mixed_informative_20260913_141822_718319/aggregate_summary.json`
- `relaxed_864mb/benchmark/main_results_mixed_informative_20260913_144047_513954/aggregate_summary.json`

`formal_controller` 三个 aggregate 入口：

- `formal_controller-ca75f4fd6cba10d56d8210cb/aggregate_summary.json`
- `formal_controller-4931a055d167dfcf76cf6ded/aggregate_summary.json`
- `formal_controller-d81f46ce6883684144cfe40d/aggregate_summary.json`

以上路径均相对于 current evaluation-only run root。P01 未从 aggregate 中提取、筛选或解释 performance 数值。

## 8. 失败与停止边界

首个 `formal_ablation` cell `formal_ablation-3e9322fac172fcae01f2cc58` 在 attempt 1 以 `failed_terminal`, return code 1 结束且未 committed。failure report 将根因分类为 `frozen_ablation_command_active_bundle_root_mismatch`：child 从 executor checkout 运行 support script，但 frozen protocol index 位于独立 scientific checkout；active-bundle validator 以 executor root 做相对路径校验并 fail closed。

P01 不授权也不实施修复、retry、recovery、finalize、lock cleanup 或后续 phase dispatch。held lock 未触碰，holdout 未开启。

## 9. Claim 状态总表

| Claim family | 当前状态 | P01 可写内容 |
|---|---|---|
| problem/system contract | implemented / source-audited | 可写精确定义和边界 |
| role-factored graph PPO | implemented / checkpoint-audited | 可写 enabled 路径；禁写 disabled 机制 |
| data realism | provenance-only | 只能写组合数据，不写 joint trace |
| typed base sharing | `UNVERIFIED` | 只能写机制定义，不能写效果 |
| byte hit / request readiness | `UNVERIFIED` | 结果占位 |
| transfer overhead | `UNVERIFIED` | 结果占位；hit 与 transfer 联合解释 |
| workflow continuity | `UNVERIFIED` | 结果占位 |
| capacity pressure | `UNVERIFIED` | 结果占位 |
| eviction-policy effect | `UNVERIFIED` | 结果占位 |
| controller comparison | `UNVERIFIED` | 结果占位 |
| predictor boundary | `UNVERIFIED` | 只写 baseline predictor 实现边界 |
| ablation attribution | `UNAVAILABLE` | formal ablation 无 committed cell |
| oracle opportunity | `UNAVAILABLE` | 等待 exact cells；不能写 global upper bound |
| significance / superiority | `UNAVAILABLE` | 统计与 gate 未开始 |
| holdout generalization | `UNAVAILABLE` | holdout sealed |
| paper-ready | `UNAVAILABLE` | 不得声明 |

## 10. 现有写作资产与处置

- 历史方法材料：`docs/project/sa_ghmappo_paper_method_report_20260716.md`。
- 历史导出：`outputs/sa_ghmappo_paper_method_report_20260716.docx`。
- 论文产物导出脚本：`scripts/export_paper_artifacts.py`。
- 当前仓库未发现成熟完整的 `.tex`/LaTeX manuscript 主线。

P01 新建独立 `paper/` 写作区，不覆盖历史材料，不修改代码、脚本、科学配置或 artifact。

## 11. 顶刊审查结论

- Artifact completeness：`BLOCKED`，16 个 expected cells 未 committed，关键后续 phases 未开始。
- Protocol/provenance：冻结 protocol、code/model source 和已有 cell provenance 可追踪；但最终 complete manifest/gate 缺失。
- Statistics/baseline fairness：协议已冻结，正式统计尚未执行，结论 `Unverifiable`。
- Mechanism realization：实现路径可核验；效果与必要性因 ablation 不可用而 `Unverifiable`。
- Claim boundary：可写系统、实现、协议；所有比较性/显著性/泛化 claim 暂禁。
- Literature novelty：最近有审计的 cutoff 为 2026-06-21；投稿前必须由 P02 或后续任务从一手来源刷新并核验。

当前唯一合规结论是：**论文结构和方法描述可以继续推进；结果结论、晋级判断和 paper-ready 判断均不可验证。**
