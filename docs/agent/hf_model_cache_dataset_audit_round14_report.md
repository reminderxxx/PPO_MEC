# hf_model_cache_dataset_audit_round14

## 结论

本轮没有把 Hugging Face 数据集作为正式 benchmark 输入。审计后也不建议直接接入正式对比实验：当前候选只能支撑真实模型文件大小、分块大小或大规模 cache-like 体量参考，不能支撑真实 VEC cache hit/miss、RSU locality、handoff demand 或 adapter state migration trace。

因此当前安全接入边界是 metadata + file-size profile；benchmark consumption 需要先实现显式 importer、adapter_id 映射和单独结果标签。

## 候选审计

| Dataset | 可用文件/规模 | Viewer | 当前判定 | 下载页 |
|---|---:|---|---|---|
| ClemSummer/qwen-model-cache | 2288.932 MB / 11 files | no | not_ready | https://huggingface.co/datasets/ClemSummer/qwen-model-cache |
| ClemSummer/cbow-model-cache | 950.776 MB / 3 files | no | not_ready | https://huggingface.co/datasets/ClemSummer/cbow-model-cache |
| Efficient-Large-Model/imagenet-llamagen-cache | 6881.76 MB / 2 files | yes | limited_not_vec_semantic | https://huggingface.co/datasets/Efficient-Large-Model/imagenet-llamagen-cache |
| Kuperberg/bert-model-cache | 4676.155 MB / 29 files | no | not_ready | https://huggingface.co/datasets/Kuperberg/bert-model-cache |
| amansapkota/examsathi-model-cache | 0.002 MB / 1 files | no | not_suitable | https://huggingface.co/datasets/amansapkota/examsathi-model-cache |

## 如何接入

1. 保留 `data/raw/model_cache/huggingface_model_cache_sources.json` 作为 HF 候选全集审计 manifest，不自动下载原始大文件。
2. 只把具备真实模型/cache 文件的候选投影成单独的 `hf_file_size_profile`，从 Hub metadata 的 file size 生成 `CacheObject.size_mb`。
3. `adapter_id` 不能从 HF 文件名自动推断，必须增加显式映射表，例如 `hf_file -> adapter_tracking`，并在报告里标注这是 size-profile projection。
4. benchmark 入口需要显式选择新的 catalog/profile，输出目录必须独立命名为 `hf_model_cache_*`，不能覆盖当前 `NGSIM + Alibaba` 主线结论。
5. 论文或报告中只能声明“使用 HF 真实模型文件大小/缓存体量 profile”，不能声明“使用 HF 真实 cache request trace”。

## 产物

- audit_csv: `D:\PPO_MEC\artifacts\analysis\hf_model_cache_dataset_audit_round14\hf_model_cache_dataset_audit.csv`
- diagnosis_summary: `D:\PPO_MEC\artifacts\analysis\hf_model_cache_dataset_audit_round14\diagnosis_summary.json`
- integration_plan: `configs\data\hf_model_cache_integration_plan.json`
