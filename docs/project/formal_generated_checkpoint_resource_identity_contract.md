# Formal Generated Checkpoint Resource Identity Contract

版本：`1.0.0`（G14R14，2026-09-05）

本合同解决 run 启动前不可知的 checkpoint content identity。static portable registry 只登记已存在的 immutable
dataset、catalog、window plan、runtime config 与 fairness manifest；不得预声明、占位或伪造未来 checkpoint hash。
当前 run 的 `checkpoint_freeze` 形成 committed terminal 后，runner 从 selection/freeze companions 构造独立的
`generated_checkpoint_resource_registry.json`，以原子 create-only 方式发布。

registry 固定包含三档容量各一个 seed checkpoint manifest 和 provenance manifest，共 6 个资源。每行绑定 logical
ID、role、schema、capacity label/value、run-root-relative path、size/content hash，并由 provenance 完整覆盖 manifest
中的每个 frozen checkpoint。registry 顶层绑定 Protocol semantic/full、active bundle、execution commit、resolved
context、Scientific Config、training binding、dev selection、freeze、source phase committed ledger 与 current run ID。

cache-policy outer、nested benchmark、controller、ablation、support、scalability、statistics、integrity、gate、
artifact inventory 与 claim evidence 都必须在 checkpoint read 前验证 registry。outer/child 必须逐项传递相同 static
registry、generated registry、checkpoint manifest/provenance IDs；显式 path 与 logical identity 冲突时拒绝。

三档映射不可回落：`constrained_288mb ↔ 288`、`medium_576mb ↔ 576`、`relaxed_864mb ↔ 864`。runtime、fairness、
checkpoint manifest/provenance 的 path、ID、hash 与 capacity metadata 必须一致。missing/duplicate/static collision、
wrong role/schema/capacity、hash/size drift、symlink/path escape、cross-run、staging/uncommitted/failed phase、stale
context/bundle/selection/freeze、G14C v1–v13 reference、post-publication rewrite 和 wrapper/child bypass 均 fail-closed。

Formal exact gate 只审计完整性、追溯性与预注册执行，不按性能筛选。non-formal rehearsal 使用自己的缩小计数并
强制 `formal=false`、`performance_evidence=false`；它不能进入 canonical formal artifacts 或论文结论。Holdout
capability 固定为 false，sealed holdout 只允许 metadata identity reachability。
