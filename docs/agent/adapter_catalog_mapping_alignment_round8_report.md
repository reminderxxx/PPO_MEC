# adapter_catalog_mapping_alignment_round8

## 本轮范围

本轮没有训练、没有 freeze、没有修改 reward、policy、baseline 或 checkpoint selection，也没有替换 mixed/full benchmark。

本轮只做三件事：

- 给 Alibaba DAG parser 增加可关闭的 `adapter_assignment_profile`，默认仍是 `legacy_batch_type`。
- 新增 `semantic_ai_service` controlled AI-service adapter assignment profile，用于 proposal 验证。
- 新增只读验证脚本和 proposal-only 配置草案。

## 改动文件

- `src/data/workflow/alibaba_dag_parser.py`
- `src/data/workflow/workflow_dataset_builder.py`
- `scripts/validate_adapter_mapping_profile.py`
- `configs/benchmark/multi_adapter_hard_joint_proposal.yaml`
- `docs/agent/adapter_catalog_mapping_alignment_round8_report.md`

## 生成产物

- `artifacts/analysis/adapter_catalog_mapping_alignment_round8/adapter_mapping_profile_validation.csv`
- `artifacts/analysis/adapter_catalog_mapping_alignment_round8/adapter_catalog_alignment_check.csv`
- `artifacts/analysis/adapter_catalog_mapping_alignment_round8/cache_capacity_stress_proposal.csv`
- `artifacts/analysis/adapter_catalog_mapping_alignment_round8/diagnosis_summary.json`

## 核心回答

1. 为什么当前 benchmark 只有 `adapter_batch_type_1`？

当前 selected workflows 是 `j_3` 和 `j_8`。它们在 Alibaba `batch_task.csv` 中的节点 `task_type` 都是 `1`。legacy parser 规则是：

```text
required_adapter = adapter_batch_type_<task_type>
```

因此所有节点都变成 `adapter_batch_type_1`。

2. `adapter_batch_type_1` 为什么与 `sample_model_catalog.json` 不对齐？

`src/data/model_catalog/sample_model_catalog.json` 当前定义的是 5 个 AI-service adapter：

- `adapter_perception`
- `adapter_tracking`
- `adapter_fusion`
- `adapter_intent`
- `adapter_control`

它不包含 `adapter_batch_type_1`。所以问题不是 catalog 只有一个 adapter，而是 legacy Alibaba task_type mapping 和 sample catalog 的 adapter 命名体系没有对齐。

3. `legacy_batch_type` 和 `semantic_ai_service` 的区别是什么？

- `legacy_batch_type`：保持旧行为，`task_type=1 -> adapter_batch_type_1`，用于复现实验和兼容旧 artifact。
- `semantic_ai_service`：基于真实 Alibaba DAG structure 和 task_type 字段，按 DAG 位置/依赖形态映射到现有 5 个 AI-service adapter。它是 controlled AI-service adapter assignment profile，不声称这些 adapter ID 来自 Alibaba 原始数据。

4. `semantic_ai_service` 是否能在 `j_3/j_8` 上产生多 adapter？

能。验证结果：

- `j_3`: 4 个 adapter
  - `adapter_control`
  - `adapter_fusion`
  - `adapter_perception`
  - `adapter_tracking`
- `j_8`: 5 个 adapter
  - `adapter_control`
  - `adapter_fusion`
  - `adapter_intent`
  - `adapter_perception`
  - `adapter_tracking`

5. 这些 adapter 是否都存在于 `sample_model_catalog.json`？

是。`semantic_ai_service_all_adapters_in_catalog = true`。

6. 当前是否已有 cache capacity 参数？

未发现可靠的 `cache_capacity` 参数或正式配置入口。当前环境代码更接近 append/ensure cached adapter 行为。

7. 当前是否已有 eviction 机制？

未发现可审计的 eviction 机制或底层 eviction event telemetry。

8. 本轮是否只提出 proposal，没有改变现有环境行为？

是。`cache_pressure_profile.enabled = false`，`proposal_only = true`。本轮没有修改 cache 行为，也没有让 proposal 接管正式 benchmark。

9. `multi_adapter_hard_joint_proposal` 是否是新数据集？

不是。它是基于真实 mobility trace + 真实 Alibaba DAG structure 的 controlled AI-service stress profile proposal。

10. 哪些部分是真实的？

- NGSIM mobility trace
- Alibaba DAG structure
- Alibaba `task_type` 原始字段

11. 哪些部分是可控构造的？

- workflow-to-adapter assignment: `semantic_ai_service`
- adapter/model size profile: 对齐 `sample_model_catalog.json`
- cache capacity stress setting: proposal-only，当前未启用

12. 下一轮建议

不建议继续盲调 SA policy。建议顺序：

1. 先补 cache capacity / eviction / cache occupancy / admission-added-new-adapter telemetry。
2. 再做只读 split proposal，明确哪些 window/workflow 属于 multi-adapter hard-joint。
3. 再显式启用 `semantic_ai_service` proposal profile。
4. 再跑 SA vs IPPO/PPO/popularity 的公平 benchmark。

## 验证结果

已执行：

```bash
python -m py_compile scripts\validate_adapter_mapping_profile.py src\data\workflow\alibaba_dag_parser.py src\data\workflow\workflow_dataset_builder.py
python scripts\validate_adapter_mapping_profile.py
```

结果：

- `semantic_ai_service_produces_multiple_adapters: True`
- `semantic_ai_service_all_adapters_in_catalog: True`
- `cache_capacity_eviction_remains_proposal_only: True`

## 结论

当前 benchmark 已经激活 DAG 与 handoff/cross-RSU pressure，但尚未激活多 adapter/model cache competition。本轮已经把 catalog 与 mapping 的对齐路径打通为 proposal，但默认 legacy 行为不变，旧结果仍可复现。
