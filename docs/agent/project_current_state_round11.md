# project_current_state_round11

## 当前主线目标

当前项目主线仍是 `NGSIM + Alibaba`，研究对象是 AI-driven VEC 中跨 RSU 连续 DAG workflow 执行、adapter cache 协同、handoff 状态迁移与多时间尺度控制。

当前主算法是 `sa_ghmappo`。当前不应继续盲调 policy；下一步应先补齐对照方法 rows 与指标定义审计。

## 数据集声明

统一数据集声明已补到 `docs/project/DATASET_SOURCES.md` 和 `configs/data/dataset_sources.json`。当前所有外部数据源均要求记录 dataset name 与可到达下载页。

| 数据源 | 当前角色 | 下载页 |
| --- | --- | --- |
| Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data | 正式 mobility trace 主线 | https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories-and-supporting-data |
| Alibaba Cluster Trace Program - cluster-trace-v2018 | 正式 workflow DAG 主线 | https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018 |
| Luxembourg SUMO Traffic (LuST) Scenario | 保留 mobility provider | https://github.com/lcodeca/LuSTScenario |
| The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways | 保留 mobility provider 骨架 | https://levelxdata.com/highd-dataset/ |
| ClemSummer/qwen-model-cache | Hugging Face 真实 model-cache metadata source | https://huggingface.co/datasets/ClemSummer/qwen-model-cache |

`ClemSummer/qwen-model-cache` 已接入 `src/data/model_catalog/sample_model_catalog.json` 的 `model_cache_datasets`。本接入是 metadata-only，不自动下载模型文件，不改变正式 benchmark 默认 cache 行为。

## Benchmark 层级

- `mixed_informative`：正式候选层，用于当前 round2 qualified candidate 的 mixed 对照。
- `full_stratified`：正式候选层，用于当前 round2 qualified candidate 的 full 对照。
- `multi_adapter_hard_joint_proposal`：proposal only，不是新真实数据集，不可 freeze。
- `multi_adapter_hard_joint_smoke_round10`：proposal smoke，不是正式论文结果，不可 freeze。

明确：`multi_adapter_hard_joint_smoke_round10` 不是新数据集，不是正式论文结果，不可 freeze。它只是在真实 mobility trace + 真实 Alibaba DAG structure 上叠加 controlled AI-service adapter assignment 与 controlled cache-capacity stress。

## round5-round10 状态摘要

- round5：mixed 中 SA reward `83.405000`，popularity `83.513333`，SA-pop `-0.108333`；full 中 SA reward `76.654815`，popularity `75.492778`，SA-pop `+1.162037`。mixed gap 集中在 `mechanism_activating`，SA 少 prefetch、多 migration prepare、backhaul 更低，剩余 gap 主要表现为 continuity reward tie-break。
- scenario innovation audit：当前数据不是 easy static；`hard_joint=0.666667`，`mechanism_activating=0.333333`。`j_3` 是 9 nodes / critical path 5，`j_8` 是 17 nodes / critical path 9。handoff/cross-RSU pressure 存在，但 adapter diversity 只有 1。
- round7：catalog 中有 5 个 adapter，但 benchmark 只出现 `adapter_batch_type_1`；原因是 j_3/j_8 的 Alibaba `task_type=1` 被 legacy parser 映射到单一 adapter。base model 只有 `veh_base_v1`，且当时没有 capacity/eviction。
- round8：新增 `adapter_assignment_profile`，默认 `legacy_batch_type` 不变；`semantic_ai_service` 只在显式启用时生效，j_3 产生 4 个 adapter，j_8 产生 5 个 adapter，全部在 catalog 中。
- round9：新增默认关闭的 cache capacity/eviction telemetry。`enabled=false` 保持 append-only；`enabled=true, rsu_adapter_slots=2` 时 LRU eviction 生效。
- round10：生成 `24` 条 proposal smoke rows，`semantic_ai_service_active=True`，`adapter_diversity_activated=True`，`cache_capacity_profile_active=True`，`eviction_activated=True`，`handoff_pressure_active=True`。

## round10 policy 状态

- `sa_ghmappo`: reward `92.556667`, continuity `0.583334`, failure `0.0`, backhaul `221.333333`。
- `ppo`: reward `72.213333`, continuity `0.527778`；row 来自 current `ppo` registry + existing `flat_ppo` checkpoint alias。
- `popularity_cache_heuristic`: reward `88.395`, continuity `1.0`。
- `reactive_greedy`: reward `65.175`, continuity `0.916666`。
- `ippo`: missing，原因是 `not_registered_or_not_evaluable`。

## 为什么现在不能 freeze

- round10 是 proposal smoke，不是正式 benchmark split。
- IPPO rows 缺失，SA vs IPPO 仍无法回答。
- continuity 指标在 hard joint smoke 下暴露出 SA 相对 popularity 的短板，需要先做 failure diagnosis，而不是直接 freeze。
- controlled AI-service/cache stress 不应被写成真实数据集或正式论文结论。

## 下一步优先级

1. 先补 IPPO live eval/checkpoint rows。
2. 确认是否存在遗漏 hand-written rule rows；当前 live heuristic rows 是 `popularity_cache_heuristic` 与 `reactive_greedy`。
3. 做 `hard_joint_policy_failure_diagnosis`，定位 SA continuity 低但 reward 高的具体 step/window/workflow 原因。
4. 最后才考虑 policy-side prefetch/cache-admission bias。
