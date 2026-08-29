# Active Bundle Resource Resolution Contract

- contract version：`1.0.0`
- active Protocol：`1.9.0`
- active index：`configs/experiment/typed_model_cache_formal_protocol_v1_9_20260829/protocol_index.json`
- status：`READY_FOR_G14C_V9_CLEAN_TRAIN_AND_FORMAL`

## 问题与边界

Protocol v1.8 已将资源身份统一登记为 `active_bundle_resources`，但 dev selector 仍读取已删除的
`index["runtime_configs"]` 与 `index["dev_fairness_manifests"]`，导致 G14C v8 在 150/150 training、
1,200 candidates 后、任何 dev performance row 前失败。v1.9 不恢复旧顶层字段；v1.0–v1.8 只允许历史审计，
不得启动 active execution。

G14C v8 `typed_model_cache_formal_20260828_101804_g14c_v8` 永久标记为
`invalid_after_training_before_dev_performance_execution`。failure/integrity/inventory SHA-256 分别为
`2c09cd14028051a012ddedf756bd6b186b4d1680582c5944acc0da986aa40ba5`、
`d2a02fb61bd5b1f9964a7516441ac3ec31d95c0b4451190291be6a9bd1bf3bba`、
`025b616efcbf9a41289f0a05a0f07bd2a8d1afaa22698ef70fc21c15d034aba5`。resume、retry、finalize、
salvage、checkpoint/candidate/partial-dev-input 复用全部禁止。

## 唯一 API

`src/runtime/active_formal_bundle.py` 提供：

- `resolve_active_bundle_resource(bundle, logical_id, expected_role=...)`
- `resolve_active_bundle_group(bundle, prefix, expected_role=...)`
- `resolve_capacity_resource_pairs(bundle, fairness_group=...)`
- `resolve_support_resource(bundle, setting_id)`
- `validate_registered_resource_path(resource, supplied_path)`
- `build_active_bundle_resource_resolution_audit(bundle)`

这些 API 只接受 `validate_active_formal_bundle()` 返回且带有进程内 validation token 的结果；裸 index、
手工构造 mapping 或历史 index 均不能作为可信输入。

## 解析规则

`active_bundle_resources` 是唯一资源目录。每项解析结果固定包含 logical ID、role、logical path、绝对路径、
version scope、content SHA-256、size、可选 semantic SHA、active bundle SHA 与 validation status。logical ID
必须唯一，prefix 与 role 必须匹配，显式 CLI 路径必须与登记路径、hash、size 完全一致，路径逃逸和 symlink
禁止。

Capacity 顺序固定为 `constrained_288mb`、`medium_576mb`、`relaxed_864mb`。runtime 与 formal/dev fairness
必须按同名 label 一一配对，不允许 missing、extra 或错配。Support fairness 按冻结 setting ID 解析；capacity
support 使用同 capacity 的 formal fairness。资源列表物理顺序不是身份，resolver 总按冻结顺序输出。

## Provenance 与执行

outer runner 在 run-root 写入前重新验证 active bundle，并生成完整 resource-resolution audit；audit hash 进入
resolved expansion context 与 command matrix，因此被 execution binding、resolved context、phase/cell input
间接绑定。dev selector 在读取 checkpoint 前再次从 run-local context 解析 index、核对 active bundle hash，
写出 `dev_resource_resolution_audit.json`，并将 audit identity 写入 checkpoint provenance、cell input 与 candidate
rows。formal support 对 runtime/fairness 显式路径执行同样登记验证。nonformal dev rehearsal 与 formal dev 共用
同一三档 capacity resolver，不存在独立 schema 绕行。

## 科学与安全边界

Scientific Config `2.0.0`、Agent Order Contract `1.0.0`、agent/seed/budget/capacity/data/window/statistics/
holdout 与 dependency fingerprint 均未改变。Readiness v11 只证明执行合同和非正式验收，不是正式 checkpoint、
performance、holdout 或 paper-ready 证据。Holdout 保持 `sealed=true/opened=false`。
