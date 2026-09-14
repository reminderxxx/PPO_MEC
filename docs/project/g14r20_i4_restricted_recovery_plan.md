# G14R20-I4 跨-checkout修复与受限恢复方案

- `plan_version`: `g14r20_i4_restricted_recovery_plan_v1.0.0`
- `reviewed_at`: `2026-09-14T00:00:00+08:00`
- `literature_cutoff`: `2026-06-21`（本轮不做 novelty 复审）
- `target_venue`: `IEEE TMC`
- `artifact_run_id`: `typed_model_cache_evaluation_only_20260913_g14r20_i3_pending`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `minimum_reviewed_executor_code_commit`: `c1abc4ba96319bda19db7d05a2e106cf4230a38e`
- `minimum_reviewed_executor_code_tree`: `a44f9582e8110e747f9c478ffdbe7415b69f7962`
- `evidence_level`: `E2_ARTIFACT_AUDITED_FOR_RECOVERY_SCOPE`
- `paper_verdict`: `Unverifiable`（正式矩阵未完成，本轮没有性能审查）
- `recovery_grant_issued`: `false`
- `real_recovery_started`: `false`
- `holdout_opened`: `false`

## 根因和最小修复

evaluation-only executor 从 I3 checkout 运行新代码，冻结 Protocol 与 active bundle 则属于独立 scientific
checkout。`run_typed_model_cache_formal_support.py` 先正确验证 executor-side resolved context，随后错误地将
executor `ROOT` 传给 scientific `protocol_index` 的 active-bundle validator，导致合法 index 无法
`relative_to(ROOT)`，并以 `only the unique active protocol index is accepted` fail closed。

修复保持两类身份独立：

1. `resolve_scientific_bundle_root()` 从完整重验的 model source reference、source context、scientific
   commit/tree 和 evaluation context 解析唯一 scientific root，同时再次核对 executor absolute root、HEAD/tree
   和 clean code scope。
2. active index、Protocol、resource role、logical ID、content SHA-256、size 与 bundle hash 仍由原 validator
   校验，不接受任意 index 或路径。
3. frozen command 中 executor-local runtime/fairness path 只能作为同 relative logical path 的 checkout mirror；
   mirror 必须位于已验证 executor 内、无 symlink，并与 scientific bundle 登记的 hash/size 完全一致。
4. 不改变全局 `repository_root`，不复制或链接 scientific 文件，不改变参数、模型、窗口、统计或原请求。
5. `formal_ablation`、`formal_support`、`formal_scalability` 共用同一 public support consumer，均受同一修复覆盖；
   scalability 的 frozen `fairness_manifest.formal.medium_576mb` 按容量 logical ID 校验，不再误查不存在的
   setting-specific support fairness。

## I3 只读保全结论

两份原生 ledger validator 均通过，且当前完整文件就是拟议恢复锚点：

| ledger | records | bytes | full-prefix SHA-256 | terminal hash/status |
|---|---:|---:|---|---|
| phase | 8 | 168899 | `a12731365111da83422ad4062ab482b08f3ee30c967084f6a32f4369bbce9adb` | `bfd7f4071b58d3a6f9cb61355dcfb6785c39ab490bfa4e90ebe34ac1807bb362` / `formal_ablation failed` |
| cell | 14 | 4090388 | `574ca9084db9ce377507edf7d210854dd69f575eed7135f4d567a35895310016` | `012ba1af0174f723352cf1ff7fb5aad262f80c2c6be0371cb34efee4050d149a` / `failed_terminal` |

六个 committed cell 的 transaction inventory、committed marker、producer manifest、producer exact membership、
SHA-256/size 和 publication relocation preimage/non-path semantics 均通过。保留资格只依据身份与双层完整性，
没有读取效果决定取舍：

| phase / capacity | cell ID | transaction inventory SHA-256 | files | producer files |
|---|---|---|---:|---:|
| cache / 288 | `formal_cache_policy-458f1a514d3de2a9855adae8` | `0c4b3524a194373566efafa441861f840acf1c72cf1abf5d719d138c8ccf3179` | 2712 | 2708 |
| cache / 576 | `formal_cache_policy-57ea8942f9af2530278a6888` | `991b30e61a56dd182584c8fafe24693a271274bea2bed5a7bbcb8354efc3c6c7` | 2712 | 2708 |
| cache / 864 | `formal_cache_policy-b995253ce76fcb798c98cf9e` | `c3c9f647d591a7e16038a42492e2b63ed3c43e8932433cd7856d93ebad3aec4c` | 2712 | 2708 |
| controller / 288 | `formal_controller-ca75f4fd6cba10d56d8210cb` | `b920a0a749d03abf752988e0d5826150975e10409e280067b4cc68997f7d7c5d` | 2711 | 2708 |
| controller / 576 | `formal_controller-4931a055d167dfcf76cf6ded` | `0f22d399cc7849fbc8cbf6bbd39d7bb417669fad874e4157dba9c6393efb1fd4e` | 2711 | 2708 |
| controller / 864 | `formal_controller-d81f46ce6883684144cfe40d` | `ecc223765507a0d09e128021e2d4ab7b9df9e98c3ae98edc786b18ea259a1462` | 2711 | 2708 |

> 注：最终机器可读 evidence 是上述表的权威逐项值；文档中的任何抄录错误不得替代 evidence/validator。

失败边界保持不变：`formal_ablation-3e9322fac172fcae01f2cc58` attempt 1 为
`failed_terminal`；staging 仅有空 stdout 和 SHA-256 为
`51b2569c1b7c98c950ba2f509122613f66606a2821368b0756007b828e3db414` 的 stderr。另一个 cell
`formal_ablation-e40a9c86c8fdbf3b7962a689` 尚无 ledger record。held lock 状态与 inode
不变；lock owner canonical SHA-256 为
`0cfeecba26816805d6445b9513263f183a3db651c5bdbbb9e204431b9872d11f`。

150 个 source models 全量复算通过：10 agents × 5 seeds × 3 capacities，每 capacity 50；source reference
canonical SHA-256 为 `0ba289caa663d38dd35f61feafa509a3f8dc2eb4269b3220f18d383be81c3ea3`，
scientific commit/tree 仍为 `a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d` /
`eb8e83c6e4532bc45c22fb7a816388e576753e30`。

## 现有合同的恢复可行性

结论：当前 executor 不能仅凭新 grant 安全恢复，现有合同也不能直接完成“新代码 + 原 run 原地续写”。原因是：

- 原 grant 与 execution context 固定 executor `a028ea2`、tree、command paths 和 command matrix；新代码 commit
  会改变这些身份，grant 不能重写合同事实。
- `reconciliation.py` 明确拒绝任何 `failed` phase 和 `failed_terminal` cell。
- `FormalCellLedger.begin_cell()` 明确以 `terminal cell failure forbids resume` 拒绝新 attempt。
- phase ledger v3 将 `failed` 设为该 phase 的不可变 terminal；同 phase 后续 record 会被
  `phase terminal record is immutable` 拒绝。
- 现有 project grant 明确 `failure_cell_recovery=false`、`failure_phase_recovery=false`，且 held lock 的 crash
  owner/quiescence 也未区分普通新 grant 与 cold recovery。

因此，签发另一份普通 grant、清锁、改旧 context/manifest 或让新代码冒充旧 executor 都不能成为恢复方案。

## 最小受限恢复方案（待另行实现与批准）

现有合同缺少所需 primitive。本轮不实现大型恢复平台，也不创建 recovery run。可审查的最小扩展应满足：

1. 建立一个新、独立 recovery execution identity，显式绑定上表两个完整 ledger 前缀、原 source reference、
   六个 committed marker/inventory、原 failed phase/cell terminal、新 executor commit/tree 以及唯一 recovery scope。
   原 run、原 ledger 和 failed terminal 永久只读。
2. 通过只读 external-cell reference inventory 消费六个 committed cell；每次消费前重复双层校验。不得复制、
   symlink、改写或重新 dispatch 这六个 cell。
3. 对失败 cell 记录 `source_attempt=1` 与 `recovery_attempt=2`，在新 recovery identity 下从头执行一个新 staging
   attempt；attempt 1 继续作为原 run 的 failed terminal。不得把 attempt 2 写成 attempt 1 的 finalize。
4. 第二个从未启动的 ablation cell 不是 retry：它只在 recovery attempt 2 成功并 committed 后，以自己的首次
   attempt 执行。两者不得并行、合并或交换 setting ID。
5. 本阶段成功条件仅为两个 ablation cell 的新产物通过 producer/transaction 双层完整性并形成独立 recovery
   phase terminal；随后停止。`formal_support`、`formal_scalability`、statistics、gate 和 completion 均需项目负责人
   根据计划窗口审查另行授权。
6. cold recovery 前必须由独立方证明 owner digest
   `0cfeecba26816805d6445b9513263f183a3db651c5bdbbb9e204431b9872d11f` 无 live process/descendants，并证明 kernel
   lock 可取得；固定 inode 不删除。prefix/source/executor 任一漂移即停止。

由于 external-cell reference/recovery ledger consumer 尚未实现并验收，当前方案状态为
`DESIGN_REVIEWABLE_NOT_EXECUTABLE`。随附 unsigned draft 不是 grant，不能用于 dispatch。

## 明确停止点

工程修复已通过隔离验收；恢复方案待批准。当前保持：

- `recovery_grant_issued=false`
- `real_recovery_started=false`
- `holdout_opened=false`
