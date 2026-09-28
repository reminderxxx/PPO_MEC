# Dataset Sources

更新日期：2026-09-28（新增复用候选；历史G11登记未改写）

用途：统一记录项目使用、保留或 metadata-only 接入的数据源。数据源声明不会自动下载或覆盖原始数据。

## 正式与保留数据源

| 数据源 | 当前角色 | 本地路径 | 官方页 |
|---|---|---|---|
| NGSIM Vehicle Trajectories | 正式 mobility trace 主线 | `data/raw/mobility/ngsim/` | https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories-and-supporting-data |
| Alibaba cluster-trace-v2018 | 正式 workflow DAG 主线 | `data/raw/workflow/alibaba2018/` | https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018 |
| LuST Scenario | 保留 mobility provider / support | `data/raw/mobility/LuSTScenario/` | https://github.com/lcodeca/LuSTScenario |
| highD | 后补 mobility provider 骨架 | `data/raw/mobility/highD/` | https://levelxdata.com/highd-dataset/ |

`NGSIM + Alibaba` 仍是唯一正式数据主线；G11 没有改变 benchmark 默认数据、split 或算法语义。

2026-09-28设计增补：候选 `vec_ai_workload_v0_1` 为车辆关联AI工作流/缓存的经校准合成工作负载，
尚未生成、校准、发布或接入正式benchmark，不替代上述历史主线。提出过程、D01–D12证据池、
模型/adapter/数据/状态边界及许可要求见 `vec_ai_workload_dataset_design_20260928.md`。

## G11 public model-cache dataset registry

2026-09-28找回及在线复核：旧19项确为metadata-only，不能误读为已经下载或不存在候选。
新增DriveLM（ECCV 2024）与V2X-Seq（CVPR 2023）作为驾驶AI语义/车路数据候选；
BurstGPT/Mooncake/Qwen-Bailian仍有请求或复用profile价值。逐项入口、许可边界及计划见
`vec_ai_public_resource_recovery_20260928.md`。本轮未下载或启用任何新源。

统一 registry：`configs/data/model_cache_dataset_registry.json`，版本 `1.0.0`。截至 `2026-08-19` 核验 19 个来源；完整证据、字段矩阵、分数与拒绝理由见 `docs/project/model_cache_dataset_discovery_audit_20260819.md`。

已接入 `configs/data/dataset_sources.json` 的 metadata-only 来源：

| 来源 | 分类 / recommendation | 可安全使用的真实字段 | 主要边界 |
|---|---|---|---|
| BurstGPT v2.0 | B / request trace candidate | time、session、model、tokens、elapsed time | 无 mobility/RSU/bytes/cache outcome |
| Qwen-Bailian traces | D / KV reuse profile | time、chat/turn、token lengths、prefix hashes | KV不是adapter；无RSU/latency |
| Mooncake FAST'25 traces | D / KV reuse profile | time、token lengths、prefix hashes | KV不是adapter；无model/client/mobility |
| Azure LLM 2023/2024、LMM 2025 | F / arrival-token profile | time、input/output tokens、image count | 无稳定model identity |
| AcmeTrace | F / coarse arrival-resource profile | job/user/type/times/resources | generic AI job，非serving request |
| Alibaba PAI GPU trace | F / metadata reference | job/task/time/resource metadata | license和inference subset未解决 |
| HF qwen/cbow/bert model-cache | E / model size profile | revision、file path/count/bytes | license未知；无request/cache event |

五个 HF 历史候选仍完整保留在 `data/raw/model_cache/huggingface_model_cache_sources.json` 供审计追溯，但 `sample_model_catalog.json` 只投影 qwen/cbow/bert 三个 E 类 size metadata。ImageNet/LlamaGen 内容 cache 与空 Examsathi 候选已拒绝，不进入 live catalog。

## 受控 profile

这些不是外部真实数据集，不能在论文中写成新数据集：

| Profile | 路径 | 边界 |
|---|---|---|
| PPO_MEC sample model catalog | `src/data/model_catalog/sample_model_catalog.json` | repo-local base model、adapter/cache/state schema；其中外部 dataset rows 仅为 metadata |
| multi-adapter hard-joint proposal | `configs/benchmark/multi_adapter_hard_joint_proposal.yaml` | 叠加在 NGSIM + Alibaba 上的 controlled stress profile |

## Claim boundary

- 未发现公开 A 类 joint VEC AI cache trace，也未发现 C 类真实 adapter/LoRA request trace。
- 独立的 request/KV/size 来源可在后续单独批准的 G12 中与 NGSIM 做明确标注的 exogenous/synthetic alignment，但不能称 jointly observed trace。
- G11 未下载原始 payload，未实现 importer，未启用任何新源进入 formal benchmark。
- HF 模型文件不能称 request trace；KV/prefix cache 不能等同 adapter cache。
