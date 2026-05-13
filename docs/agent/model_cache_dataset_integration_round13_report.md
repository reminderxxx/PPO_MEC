# model_cache_dataset_integration_round13

## 范围

本轮接入真实 Hugging Face model-cache 数据源的 metadata，不下载模型文件，不覆盖原始数据，不修改 reward、policy、baseline、checkpoint selection 或正式 benchmark split。

## 数据集声明检查

检查结论：此前报告里有 `NGSIM + Alibaba` 和 controlled stress 边界说明，但没有统一的“数据集名称 + 可到达下载页”声明表。本轮新增统一声明入口：

- `docs/project/DATASET_SOURCES.md`
- `configs/data/dataset_sources.json`

## 新接入的真实 Model Cache 数据源

| 数据集名称 | 提供方 | 当前接入方式 | 下载页 |
| --- | --- | --- | --- |
| ClemSummer/qwen-model-cache | Hugging Face Datasets | metadata-only，写入 `AdapterCatalog.model_cache_datasets` | https://huggingface.co/datasets/ClemSummer/qwen-model-cache |

边界：该数据源现在只作为真实外部 model-cache dataset reference 进入 catalog/report 审计；当前 benchmark cache event 仍由环境和本地 catalog 语义产生，不能声称直接采样自该 HF 数据集。

## 当前所有数据源

| 数据源 | 当前角色 | 下载页 |
| --- | --- | --- |
| Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data | 正式 mobility trace 主线 | https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories-and-supporting-data |
| Alibaba Cluster Trace Program - cluster-trace-v2018 | 正式 workflow DAG 主线 | https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018 |
| Luxembourg SUMO Traffic (LuST) Scenario | 保留 mobility provider | https://github.com/lcodeca/LuSTScenario |
| The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways | 保留 mobility provider 骨架 | https://levelxdata.com/highd-dataset/ |
| ClemSummer/qwen-model-cache | HF model-cache metadata source | https://huggingface.co/datasets/ClemSummer/qwen-model-cache |

## 改动文件

- `src/data/model_catalog/adapter_catalog.py`
- `src/data/model_catalog/sample_model_catalog.json`
- `data/raw/model_cache/huggingface_model_cache_sources.json`
- `configs/data/dataset_sources.json`
- `scripts/check_data_ready.py`
- `scripts/validate_dataset_source_declarations.py`
- `tests/test_model_catalog_sources.py`
- `docs/project/DATASET_SOURCES.md`
- `docs/agent/model_cache_dataset_integration_round13_report.md`

## 验证

已执行：

```bash
python -m py_compile scripts/validate_dataset_source_declarations.py src/data/model_catalog/adapter_catalog.py scripts/check_data_ready.py
python scripts/validate_dataset_source_declarations.py
python scripts/check_data_ready.py
python -m pytest tests/test_model_catalog_sources.py tests/test_env_contract.py
```

预期：

- 所有数据集声明都有 `dataset_name` 和 `download_page_url`。
- `ClemSummer/qwen-model-cache` 出现在 `sample_model_catalog.json` 的 `model_cache_datasets` 中。
- `check_data_ready.py` 能看到 HF model-cache metadata manifest。

结果：

- `py_compile`: passed。
- `validate_dataset_source_declarations.py`: passed，`all_dataset_declarations_have_name_and_download_page=True`，`model_cache_dataset_declared_in_catalog=True`。
- `check_data_ready.py`: `ready_count=4/5`；NGSIM、Alibaba、LuST、HF model-cache metadata 已就绪；`highD` 原始文件仍缺失且保持为后补数据源。
- `pytest`: `4 passed`；有一个既有 `.pytest_cache` 创建 warning，不影响测试结果。
