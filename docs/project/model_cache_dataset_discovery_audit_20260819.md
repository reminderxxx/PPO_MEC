# G11 Public Model-Cache Dataset Discovery, Qualification and Metadata Integration Audit

## Outcome

G11 已冻结为 `model_cache_dataset_registry_version = 1.0.0`。本轮截至 `2026-08-19` 从官方页面、官方仓库、官方 API metadata 和论文 artifact 链接核验 19 个候选，未下载模型权重、完整 trace、Parquet 或 WebDataset payload，未执行 G12、训练、formal、holdout 或 hidden benchmark。

结论是：没有发现公开的 A 类 `joint_vec_ai_cache_trace`，也没有发现可验证的 C 类真实 LoRA/adapter 请求 trace。找到 1 个 B 类 model-serving request trace、2 个 D 类 KV/prefix trace、3 个 E 类模型文件大小 metadata、5 个 F 类通用 AI workload trace，以及 8 个应拒绝或仅作检索记录的来源。真实 request trace、KV cache、模型文件 metadata 与内容 dataset 已严格分离。

审查元数据：

- `reviewed_at`: `2026-08-19`
- `literature_cutoff`: `2026-08-19`
- `target_venue`: IEEE TMC evidence support；本报告不是 paper-ready verdict
- `artifact_run_id`: `model_cache_dataset_discovery_20260819_g11_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- 基线 Git commit: `39da1da594ffdc2cdb1f0c049ea9f06385b995cd`
- evidence level: `E2_ARTIFACT_AUDITED_FOR_PUBLIC_METADATA_ONLY`

## 本地现状复核

G11 前项目在统一数据源声明、HF 兼容 manifest 与 `AdapterCatalog` 中登记同一组 5 个 HF 名称，但它们都不是真实 model-cache request trace。`ClemSummer/qwen-model-cache`、`ClemSummer/cbow-model-cache` 和 `Kuperberg/bert-model-cache` 实际提供模型/embedding/shard 文件与文件大小；`Efficient-Large-Model/imagenet-llamagen-cache` 是 ImageNet/LlamaGen 内容 cache WebDataset；`amansapkota/examsathi-model-cache` 只有 `.gitattributes`。

这些 HF 源均没有 request timestamp、真实请求频率、cache hit/miss、eviction、RSU locality、mobility/handoff 或 adapter-state migration 字段。G11 前 5 项都进入了 `sample_model_catalog.json`，但没有一项直接进入正式 NGSIM + Alibaba benchmark。`run_hf_model_cache_transaction_experiment.py` 真正使用的是 HF Hub 文件大小 metadata、真实 NGSIM mobility 与真实 Alibaba DAG；adapter 语义投影、request order/reuse、RSU/cache event 和 handoff 结果由硬编码映射与 PPO_MEC 环境生成。因此它是“真实文件大小 metadata + 跨源真实 mobility/workflow + 合成 adapter/cache 事务”，不是联合观测 trace。

本轮已移除 live catalog 中的 ImageNet/LlamaGen 内容 cache 和空 Examsathi 候选，并让实验入口拒绝非 `model_size_profile_candidate` 来源。三个保留 HF 项仍只表示 metadata 兼容，不会自动启用正式 benchmark。

## 固定 taxonomy 与评分合同

唯一 primary class 为：A `joint_vec_ai_cache_trace`、B `model_serving_request_trace`、C `adapter_or_lora_workload_trace`、D `kv_prefix_cache_trace`、E `model_artifact_size_metadata`、F `generic_ai_workload_trace`、G `content_dataset_not_cache_trace`、H `paper_only_or_unavailable`、I `rejected_or_unsafe`。

0–100 权重固定为：时间请求序列 20、稳定 model/adapter/cache-object identity 20、bytes/size 15、reuse/cache semantics 10、load/inference/transfer latency 10、client/tenant identity 5、mobility/location/RSU 10、license/provenance/reproducibility 10。validator 同时强制 A/B/C/D identity 与时间 hard gate、A 类 mobility/RSU gate、E 类不得 request-ready、未知 license 不得 formal-ready、rejected 不得 live projection。

## 候选、分类与评分

| source key | primary class | score | recommendation | access/license conclusion |
|---|---:|---:|---|---|
| `burstgpt_v2` | B | 68 | direct request trace | public, CC-BY-4.0 |
| `qwen_bailian_kv_traces` | D | 65 | KV reuse profile | public Git LFS, Apache-2.0 |
| `mooncake_fast25_traces` | D | 60 | KV reuse profile | public, Apache-2.0 |
| `acme_trace` | F | 45 | arrival/resource profile | public, CC-BY-4.0 |
| `alibaba_pai_gpu_trace_2020` | F | 40 | metadata reference only | public metadata, data license unresolved |
| `hf_qwen_model_cache` | E | 38 | model size profile | public metadata, license unknown |
| `hf_cbow_model_cache` | E | 38 | model size profile | public metadata, license unknown |
| `hf_bert_model_cache` | E | 38 | model size profile | public metadata, license/provenance blocked |
| `azure_llm_inference_2023` | F | 30 | arrival/token profile | public, CC-BY-4.0 |
| `azure_llm_inference_2024` | F | 30 | arrival/token profile | public, CC-BY-4.0 |
| `azure_lmm_inference_2025` | F | 30 | arrival/token profile | public, CC-BY-4.0 |
| `lmsys_chat_1m` | G | 25 | rejected | gated/custom terms; conversation content, no time/cache trace |
| `serverlessllm_artifact` | H | 20 | metadata reference only | code/artifact; no public production trace |
| `hf_imagenet_llamagen_cache` | G | 18 | rejected | license unknown; content cache, not serving trace |
| `slora_artifact` | H | 10 | rejected | synthetic workload only |
| `distserve_artifact` | H | 10 | rejected | benchmark/artifact, no public production trace |
| `trimcaching_paper` | H | 10 | metadata reference only | paper claim; no public trace found |
| `azure_functions_invocation_2021` | I | 45 | rejected | licensed but generic serverless semantic mismatch |
| `hf_examsathi_model_cache` | I | 2 | rejected | only `.gitattributes`; no usable payload/schema |

分布为 B=1、D=2、E=3、F=5、G=2、H=4、I=2，A=0、C=0。高分不覆盖 hard gate；例如 Azure Functions 即使有稳定时间与函数 identity，也因非 AI model/cache workload 被拒绝。

## 深入字段核验

### BurstGPT v2.0

[官方仓库](https://github.com/HPMLL/BurstGPT)的 v2.0 release 提供多份 CSV，字段包括 `Timestamp`、`Session ID`、`Elapsed time`、`Model`、request/response/total token 和 `Log Type`。它满足时间序列与稳定 model identity，适合 model request 到达/token/总体 elapsed-time replay；缺失模型 bytes、adapter ID、cache outcome、latency decomposition、RSU 与 mobility。它是本轮最佳 B 类候选，但不能单独称为 VEC trace。

### Qwen-Bailian anonymous usage traces

[官方仓库](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon)提供生产派生的两小时 JSONL trace，字段包括 `chat_id`、`parent_chat_id`、`timestamp`、input/output length、type、turn 与经盐处理的 16-token `hash_ids`。它适合 session-aware prefix overlap/reuse；没有 user identity、精确 model version、bytes、latency、eviction、RSU 或 mobility。KV/prefix object 不能映射成 adapter object。

### Mooncake FAST'25 traces

[官方 trace 说明](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md)区分真实 conversation/tool-agent trace 与另列的 synthetic trace，并给出 `timestamp`、input/output length 与 512-token `hash_ids`。它适合 KV/prefix reuse 计算，但没有 model/client/bytes/latency/eviction/mobility 字段，且不能作为 LoRA/adapter trace。

### Azure LLM/LMM inference traces

Microsoft 官方说明分别覆盖 [2023](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md)、[2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md) 与 [2025 LMM](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLMMInferenceDataset2025.md)。公开 schema 主要是 `TIMESTAMP`、context/generated token，LMM 另有 image count；没有稳定 service/model/adapter identity，所以只能校准 arrival/token profile，不能成为 B 类 model-cache request trace。

### AcmeTrace

[官方仓库](https://github.com/InternLM/AcmeTrace)公开 job/user/type/submit-start-end time/duration/queue/resource 字段。它适合粗粒度 AI job arrival/resource profile，不是在线 inference request trace；`type` 不得伪装为 `model_id`。

完整 19×19 field coverage matrix、逐项评分组成及证据 URL 见机器 artifact。

## 五个既有 HF 候选逐项结论

| candidate | page/card/license | files and semantics | request/cache fields | decision |
|---|---|---|---|---|
| `ClemSummer/qwen-model-cache` | page 存在；无 card；license 未声明 | 11 files，约 2.40 GB，Qwen model/tokenizer/config | 无 timestamp、hit/eviction、adapter ID | E；保留 metadata-only size profile，formal blocked |
| `ClemSummer/cbow-model-cache` | page 存在；无 card；license 未声明 | 3 files，约 997 MB，CBOW embedding files | 无 request/cache sequence | E；保留 metadata-only size profile，formal blocked |
| `Efficient-Large-Model/imagenet-llamagen-cache` | page 存在；无 card；license 未声明 | 2 files，约 7.22 GB，ImageNet/LlamaGen WebDataset content cache | 无 model/adapter/request/cache-event identity | G；修正旧分类并移除 live recommendation |
| `Kuperberg/bert-model-cache` | page 存在；无 card；license 未声明 | 29 files，约 4.90 GB，chunked BERT files | 无 request/cache sequence；aggregate size provenance 异常 | E；保留 metadata-only，formal blocked |
| `amansapkota/examsathi-model-cache` | page 存在；无 card；license 未声明 | 1 file、2461 bytes，仅 `.gitattributes` | 无任何可用 request/model/cache schema | I；拒绝并移除 live recommendation |

五项的 revision、格式、证据摘要与旧分类纠正记录保留在 `existing_hf_reaudit.json`，避免以后因测试期待或名称含 `cache` 而重新误收。

## Metadata-only integration 与 mapping plan

统一 registry 记录 19 个候选；`dataset_sources.json` 只声明 10 个推荐/参考来源及 3 个 HF size 候选，所有 G11 项的 raw-download 标记均为 false。`AdapterCatalog` 仅保留 qwen/cbow/bert 三个 E 类 compatibility metadata，不承担 qualification、下载或正式启用职责。

已冻结的主要 mapping：

| source field | normalized field | PPO_MEC target | transformation | information loss / validation |
|---|---|---|---|---|
| HF path/bytes/LFS SHA | object ID/size/provenance | future explicit size-profile artifact | bytes ÷ 1048576，保留 path/revision | 丢失全部 request/locality/cache-event；需先解决 license/checksum |
| BurstGPT timestamp/session/model/tokens/elapsed | order/requester/model/tokens/latency/provenance | future external request replay | 相对时间排序，row index 作 request ID | 无 bytes/adapter/cache/RSU/mobility；需小样本单调性与枚举检查 |
| Mooncake timestamp/length/hash IDs | order/object/tokens/prefix/provenance | future KV-only replay | opaque prefix-block key，真实/合成文件分离 | 无 model/client/bytes/latency/mobility；不得映射 adapter |
| Qwen-Bailian chat/parent/time/turn/hash IDs | order/session/object/tokens/prefix/provenance | future session-aware KV-only replay | chat 作 session，hash 作 opaque block key | 无 exact model/bytes/latency/mobility；需 privacy/parent consistency 检查 |
| Azure timestamp/token/image count | order/tokens/provenance | future arrival/token calibration | row index request ID，保留 service file boundary | 不得虚构 model/client/cache identity |
| Acme job/user/type/times/resources | order/requester/latency/provenance | optional coarse AI-job profile | job-level mapping，type 仅作 workload type | 非 inference-request 粒度；不得称 model trace |

这只是 G11 mapping plan；没有实现 raw importer 或 G12 calibration。

## 在线核验、访问与 license 风险

19 个 canonical landing page 在 `2026-08-19` 均通过官方页面或官方 API metadata GET 核验为 HTTP 200，未发生 redirect；状态、最终 URL 与核验方法逐项记录于 `source_verification_rows.json`。HF 五项使用官方 Hub API file metadata；其他项使用官方 GitHub/作者 artifact/官方数据说明。LMSYS-Chat-1M 为 gated/custom terms，不能视作公开直接下载；HF 三个 size 候选与 Alibaba PAI trace 的 license 未解决，均不得 formal-ready。HTTP 200 只证明页面/metadata 可达，不证明内容语义、license 或 payload 可消费。

## 十三个必答结论

1. **真正 joint VEC AI model-cache trace：**未发现。A=0。
2. **可用 model-serving request trace：**有，BurstGPT v2.0 是最佳 B 类候选；缺 mobility、RSU、bytes 与 cache outcome。
3. **adapter/LoRA request trace：**未发现公开可验证 C 类 trace。S-LoRA/DistServe/ServerlessLLM artifact 中的 workload 不是公开真实 adapter 请求 trace。
4. **KV/prefix reuse trace：**有，Qwen-Bailian 与 Mooncake；它们是 KV/prefix，不是 adapter cache。
5. **现有 HF 五项真正有用者：**qwen、cbow、bert 仅对文件/分块大小 profile 有用；ImageNet/LlamaGen 与 Examsathi 不适合 live catalog。
6. **只能提供 model size：**qwen、cbow、bert。三者 license 均未知，bert 另有 provenance anomaly。
7. **适合与 NGSIM 明确标注 cross-source 对齐者：**BurstGPT 可提供 model request arrival/token/session；Azure 可提供 arrival/token；Qwen-Bailian/Mooncake 可分别提供 KV prefix reuse；HF 三项可提供 object size。只能称 exogenous/synthetic alignment。
8. **对齐损失的真实性：**丢失真实 vehicle→model preference、共同时间因果、RSU locality、mobility–request coupling、cache hit/eviction、load/transfer latency 与 handoff migration correlation；独立来源之间没有联合观测关系。
9. **可进入后续 G12 calibration 的字段：**BurstGPT/Azure 的 arrival/token，Qwen-Bailian/Mooncake 的 KV/prefix reuse，HF metadata 的 capacity/transfer-size 参数，Acme 的粗粒度 arrival/resource。没有来源可校准真实 mobility/handoff predictor。
10. **不能进入正式 benchmark：**未知 license/provenance 的 HF/Alibaba PAI；rejected G/I；没有真实 trace 的 H；gated/custom-term LMSYS；以及任何未经独立 G12 protocol、sample validation 与 claim boundary 审查的新源。
11. **最大不可识别字段：**同一真实系统内的 `vehicle/client ↔ RSU/location ↔ model/adapter/object ↔ bytes ↔ cache outcome ↔ load/inference/transfer latency ↔ handoff` 联合键与时间关联。
12. **若用户授权，下一最小下载：**BurstGPT v2.0 的单个有界 chronological CSV 小样本（header + 少量连续行），用于 schema、timestamp、model enum、session 与 token range 验证；不先下载完整 trace。
13. **是否应发布 synthetic/exogenous replay：**需要。应发布明确标注来源、独立采样假设、随机种子、映射、信息损失和非联合真实性的 replay；不能命名为 real joint VEC trace。

## Artifact 与完整性

机器产物位于 `artifacts/analysis/model_cache_dataset_discovery_20260819_g11_v1/`，包含 registry snapshot、URL verification、field matrix、score components、recommended/rejected sources、HF re-audit、mapping plan、validation report、search log 与 SHA-256 integrity manifest。validator 确保 JSON 无 NaN/Inf、输出可确定性重建、source key/URL 唯一且 compatibility declarations 一致。

## Claim boundary 与剩余风险

G11 只达到公开 metadata 的 E2 artifact-audited evidence。它不证明任何新源已适配环境、不证明 predictor calibration、不证明算法性能或 paper readiness。页面后续可能变更；HF 缺失 license、BERT 异常大小、Alibaba PAI license、公开 adapter trace 缺口与 joint VEC trace 缺口仍是 blocker。所有跨源组合必须保留 source provenance，并把独立性与信息损失写入未来 artifact。
