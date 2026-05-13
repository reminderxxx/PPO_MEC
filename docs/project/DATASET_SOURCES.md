# Dataset Sources

更新日期：2026-04-27

用途：统一记录当前项目使用、保留或接入为 metadata 的数据源。这里的“下载页”指可到达的官方或托管页；不会自动下载或覆盖原始数据。

## 数据集声明

| 数据源 | 当前角色 | 本地路径 | 下载页 |
| --- | --- | --- | --- |
| Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data | 正式 mobility trace 主线 | `data/raw/mobility/ngsim/` | https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories-and-supporting-data |
| Alibaba Cluster Trace Program - cluster-trace-v2018 | 正式 workflow DAG 主线，当前消费 `batch_task.csv` | `data/raw/workflow/alibaba2018/` | https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018 |
| Luxembourg SUMO Traffic (LuST) Scenario | 保留 mobility provider 和导出链路，不阻塞正式主线 | `data/raw/mobility/LuSTScenario/` | https://github.com/lcodeca/LuSTScenario |
| The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways | 保留 mobility provider 骨架，作为后补数据源 | `data/raw/mobility/highD/` | https://levelxdata.com/highd-dataset/ |
| ClemSummer/qwen-model-cache | HF 真实 model-file cache 候选；metadata/file-size only | `data/raw/model_cache/huggingface_model_cache_sources.json` | https://huggingface.co/datasets/ClemSummer/qwen-model-cache |
| ClemSummer/cbow-model-cache | HF 真实 embedding cache 候选；metadata/file-size only | `data/raw/model_cache/huggingface_model_cache_sources.json` | https://huggingface.co/datasets/ClemSummer/cbow-model-cache |
| Efficient-Large-Model/imagenet-llamagen-cache | HF 大规模 cache-like WebDataset 体量参考；非 VEC model/adapter cache trace | `data/raw/model_cache/huggingface_model_cache_sources.json` | https://huggingface.co/datasets/Efficient-Large-Model/imagenet-llamagen-cache |
| Kuperberg/bert-model-cache | HF 分块 model-file cache 候选；metadata/file-size only | `data/raw/model_cache/huggingface_model_cache_sources.json` | https://huggingface.co/datasets/Kuperberg/bert-model-cache |
| amansapkota/examsathi-model-cache | HF 候选排除记录；审计时无可用模型/cache 文件 | `data/raw/model_cache/huggingface_model_cache_sources.json` | https://huggingface.co/datasets/amansapkota/examsathi-model-cache |

## 受控 Profile

这些不是外部真实数据集，不能在论文中写成新数据集。

| Profile | 路径 | 边界 |
| --- | --- | --- |
| PPO_MEC sample_model_catalog AI-service adapter profile | `src/data/model_catalog/sample_model_catalog.json` | repo-local controlled catalog，定义 base model、adapter cache、state bundle 和 cache object 语义。 |
| multi_adapter_hard_joint_proposal | `configs/benchmark/multi_adapter_hard_joint_proposal.yaml` | proposal-only controlled stress profile，叠加在真实 NGSIM mobility 和 Alibaba DAG structure 上。 |

## 当前边界

- `NGSIM + Alibaba` 仍是正式数据主线。
- HF model-cache 候选全集已写入统一数据源声明、`AdapterCatalog.model_cache_datasets` 和 `data/raw/model_cache/huggingface_model_cache_sources.json`。
- 当前 HF 接入是 audit/metadata/file-size 层，不自动下载模型文件，也不直接驱动 benchmark cache event。
- 审计结论：现有 HF 候选不提供真实 VEC cache hit/miss、RSU locality、handoff demand 或 adapter state migration trace；正式 benchmark 若使用 HF，只能先作为单独的 file-size/profile projection，并在报告里明确边界。
