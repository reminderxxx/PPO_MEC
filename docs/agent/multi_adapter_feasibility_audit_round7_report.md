# multi_adapter_feasibility_audit_round7

## 范围

本轮只做只读审计。没有修改环境行为、reward、policy、checkpoint selection、baseline，没有新增正式 benchmark split，没有训练，也没有创建新数据集。

## 关键结论

当前 benchmark 已经激活 DAG 与 handoff/cross-RSU pressure，但尚未激活多 adapter/model cache competition。因此不建议立即继续调 SA policy，应先补齐 adapter catalog/mapping/cache telemetry。

## 逐项回答

1. 当前项目是否真的只有一个 adapter？
   - 不是。默认 catalog `src/data/model_catalog/sample_model_catalog.json` 中观测到 `5` 个 adapter：`adapter_control;adapter_fusion;adapter_intent;adapter_perception;adapter_tracking`。
2. 如果不是，为什么 benchmark 里只体现出一个 required_adapter？
   - Alibaba workflow mapping 使用 `required_adapter=f"adapter_batch_type_{task_type}"`，但当前 benchmark 选中的 `j_3/j_8` 只有 `task_type=1`，所以所有节点都是 `adapter_batch_type_1`。
3. adapter 是在哪里定义或写死的？
   - catalog 在 `src/data/model_catalog/sample_model_catalog.json`；workflow required_adapter 在 `src/data/workflow/alibaba_dag_parser.py` 中由 task_type 生成，不是 catalog 直接采样。
4. 当前是否存在多个 base model？
   - 当前默认 catalog 只有 `1` 个 base model：`veh_base_v1`。
5. base model 和 adapter 是否在系统中解耦？
   - 数据结构上解耦，`WorkflowNode` 有独立 `required_base_model` 与 `required_adapter`；但当前 Alibaba parser 固定 `required_base_model='veh_base_v1'`，adapter 由 task_type 生成。
6. workflow 节点的 required_adapter 是怎么来的？
   - 来自 Alibaba task row 的 `task_type`，映射为 `adapter_batch_type_<task_type>`；不使用 workflow_id，不使用 random seed。
7. 当前是否有 cache capacity 约束？
   - 未发现可靠 cache capacity 配置。环境当前 cache 行为是 append/ensure cached adapter，没有容量上限审计证据。
8. 当前 cache capacity 是否足以容纳所有 adapter？
   - 缺少 capacity 字段，不能量化；从源码行为看没有 eviction/capacity guard，因此更接近无限容量或无显式容量竞争。
9. 当前是否真的发生 eviction / cold start / warm hit？
   - warm hit、cold start、admission、hit/miss telemetry 存在；eviction 未观察到，且缺少底层 eviction 事件。
10. 当前 mixed/full benchmark 是否足以证明 model/adapter caching 创新点？
   - 不足。它能说明 DAG + mobility/handoff + prefetch/prepare，但不能充分说明多 adapter、多 base model、有限 cache capacity、eviction competition。
11. 后续 multi_adapter_hard_joint 是已有真实数据筛选，还是 trace-driven synthetic stress profile？
   - mobility trace 和 DAG structure 可以继续来自真实 NGSIM + Alibaba；adapter/model size profile、workflow-to-adapter assignment、cache capacity stress setting 需要明确标注为可控构造，不能写成真实数据集。
12. stress profile 边界：
   - 真实：mobility trace、DAG structure、task_type 字段。
   - 可控构造：adapter/model size profile、workflow-to-adapter assignment、cache capacity stress setting。
13. 下一轮应先做什么？
   - 先补 telemetry 和只读 split proposal；随后扩展/对齐 adapter catalog 与 workflow-to-adapter mapping，再跑 IPPO/PPO rows。暂不建议继续优化 SA policy。

## 状态摘要

- adapter_catalog_status: `observed_multiple_adapters`
- base_model_catalog_status: `single_base_model`
- workflow_mapping_status: `task_type_mapping_but_selected_workflows_single_type`
- cache_pressure_status: `capacity_missing_eviction_not_observed`
- telemetry_coverage_status: `partial`
- multi_adapter_feasibility: `requires_adapter_catalog_extension + requires_workflow_to_adapter_mapping_extension + requires_cache_capacity_config + requires_new_telemetry`

## 缺失字段

`cache_admission_added_new_adapter_count, cache_capacity, cache_eviction_event, cache_occupancy_rate, workflow_required_adapter_not_in_default_catalog`

## 产物

- `artifacts\analysis\multi_adapter_feasibility_audit_round7\adapter_catalog_audit.csv`
- `artifacts\analysis\multi_adapter_feasibility_audit_round7\base_model_catalog_audit.csv`
- `artifacts\analysis\multi_adapter_feasibility_audit_round7\workflow_adapter_mapping_audit.csv`
- `artifacts\analysis\multi_adapter_feasibility_audit_round7\cache_capacity_eviction_audit.csv`
- `artifacts\analysis\multi_adapter_feasibility_audit_round7\telemetry_field_coverage.csv`
- `artifacts\analysis\multi_adapter_feasibility_audit_round7\per_mode_adapter_diversity.csv`
- `artifacts\analysis\multi_adapter_feasibility_audit_round7\per_scenario_adapter_diversity.csv`
- `artifacts\analysis\multi_adapter_feasibility_audit_round7\per_bucket_adapter_diversity.csv`
- `artifacts\analysis\multi_adapter_feasibility_audit_round7\multi_adapter_feasibility_summary.csv`
- `docs\agent\multi_adapter_feasibility_audit_round7_report.md`
- `artifacts\analysis\multi_adapter_feasibility_audit_round7\diagnosis_summary.json`
