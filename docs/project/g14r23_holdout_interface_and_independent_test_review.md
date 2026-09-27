# G14R23 holdout 组合接口修复与新独立测试可行性审查

- `reviewed_at`: `2026-09-27`
- `literature_cutoff`: `2026-09-27`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `g14r23_holdout_interface_and_independent_test_20260927`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `implementation_git_commit`: `78bc38486616b65ee0c13f49ebea24d965ded436`
- `evidence_level`: `E2_METADATA_AND_SYNTHETIC_EXECUTION_CONTRACT_VALIDATED_NO_REAL_HOLDOUT_PERFORMANCE`
- `verdict`: `接口冲突已修复；无法建立新的可靠独立测试；paper-ready 仍为 Unverifiable`

## Executive verdict

G14R23 关闭了 dedicated opening 与 window consumption 的一阶合同冲突。原实现有三层互斥：benchmark
preflight 无条件拒绝 `sealed_holdout`，window binding 只允许 `identity_only`，最终 bundle consumer 又无条件拒绝
holdout。现在真实执行拥有独立的 `dedicated_holdout_execution` mode；它只能在 issued、hash-bound authorization
与原 append-only opening record 同时有效时通过。`identity_only` 仍只用于 metadata revalidation，普通 runner 的
sealed guard 没有删除或放宽。

组合链已经在 synthetic sealed split 上实际走通：production launcher/command builder → actual
`benchmark_main_results.py` parser/preflight → dedicated capability → complete 12-window binding → per-window consumer →
actual benchmark episode。负例覆盖未 opening、普通 formal mode、缺 capability、plan identity drift、opening 后 child
失败、永久 consumed 与二次 opening 拒绝。它证明接口组合兼容，不是模块分别通过；但 synthetic 结果不是正式
holdout 或性能证据。

新独立测试审计没有运行算法，也没有读取 reward/cache/agent outcomes。全历史 metadata 扩展与车辆 identity 扫描
确认仍有时间不重叠的 I-80 区间，但通过车辆复现过滤的 14 个候选全部集中在同一个 `i_80_run_001` capture
session 的相邻时段。预注册式可靠性规则要求 12 个窗口覆盖至少 3 个 segment-runs、每个 run 最多 4 个，同时与
全部已用区间和候选间零观察车辆复现；现有范围无法满足。因此结论是
`NO_RELIABLE_UNUSED_RANGE_ESTABLISHED`，不生成新 split、不创建 seal、不签 grant、不运行新测试。

## 1. 组合接口修复

### 根因

1. `scripts/benchmark_main_results.py` 在解析 frozen contract 后直接拒绝任何 `sealed_holdout`。
2. `validate_window_plan_binding()` 对 holdout 只接受 `identity_only`；真实执行若借用该 mode 会产生虚假 provenance。
3. `load_window_bundle_from_contract()` 即便前两层被绕过，仍无条件拒绝 holdout bundle。
4. 旧单测只分别验证 seal 与 identity-only reachability，没有执行最终生产命令到科学 episode 的组合链。

### 修复后的能力边界

- `identity_only`：只允许 sealed metadata/interval identity 检查，不能加载 bundle 或运行 policy。
- `formal` / `rehearsal`：仍不能消费 `sealed_holdout`。
- `dedicated_holdout_execution`：只允许 `sealed_holdout`，且必须同时验证 authorization 和已经 create-only 写入的
  opening record。
- authorization 冻结 exact benchmark argv、Git commit、seal、window contract/plan、candidate checkpoint manifest、
  corrected statistics contract、opening path、output root/run ID；任何字段或文件 identity 漂移都在 episode 前拒绝。
- launcher 继续调用原 `append_holdout_execution_record()`；opening record 在 child 前写入。child 失败仍永久
  consumed，重复 opening 拒绝，不删除或改写旧 record。
- benchmark parser/preflight 和最终 window consumer 分别重验 authorization/opening identity，不把 outer launcher
  的一次验证当作传递信任。

### 组合证据

`tests/test_dedicated_holdout_execution.py` 的 synthetic fixture 使用 60 个 24-frame metadata windows，其中 12 个
标记为 synthetic sealed split；成功案例执行 1 个 checkpoint-free reactive agent、1 seed、1 workflow、12 windows、
每 episode 1 step。验证对象是调用链和 guard，不评价数值。

| Case | 最终入口 | 预期/结果 |
|---|---|---|
| 合法 dedicated opening | actual benchmark episode | 通过；12/12 windows 被 consumer 加载 |
| 未 opening / 未授权 | actual parser/preflight | 拒绝；没有 output root |
| ordinary formal / 缺 capability | window binding | 拒绝；`identity_only` 仍非执行 mode |
| plan identity conflict | actual parser/preflight | episode 前拒绝；没有 output root |
| opening 后 workflow error | actual benchmark child | child 失败但 opening record 永久 consumed |
| 同 authorization 二次启动 | original sealed guard | `already exists`，拒绝二次 opening |

真实 sealed holdout 读取/执行数为 0；真实 one-time token、grant、checkpoint 和 performance output 均未创建或消费。

## 2. 全历史 metadata-only 可行性审计

### 输入与禁止项

- 基线：G14B `historical_window_usage_registry.json`，1,209 个 metadata plan/result 文件、34,661 个窗口引用、668
  个 unique outer intervals，其中 418 个 unknown interval 已按非 I-80 scope 保守排除。
- 扩展：当前完整项目历史共发现 3,957 个带 `selected_window_plan`/`selected_windows` 的 JSON，90,640 个引用。
- 后续去重：60 个 unique I-80 intervals；全部属于已冻结/已使用的 train/dev/formal/sealed allocation。
- 解析闭合：6,995 个 checkout/config 复制引用由冻结基线 `window_id` 账本消歧；24 个明确
  `synthetic/test_only` 引用隔离；3 个明确非 I-80 unresolved scope 隔离；可能触及 I-80 的 unresolved 引用为 0。
- 车辆检查：扫描完整 11,850,526 rows，只读取 `Vehicle_ID`、`Location`、`Global_Time`，不读取或推导任何
  performance/outcome 字段。

禁止输入包括 reward、cache hit/byte hit、transfer、handoff outcome、agent ranking、checkpoint performance、claim
status 和任何 proposed-window 算法结果。本轮没有先跑算法再选窗口。

### 时间、segment 与车辆相关性

- 2026-08-20 的 result-blind inventory 有 579 个 I-80 candidates。
- 与后续 60 个 used/allocated intervals 做 raw frame/time/segment-run 检查后，60 个 exact overlap 被排除，519 个仅
  达到不重叠和 24-frame embargo。
- 与全部已用 I-80 区间做 Vehicle_ID 复现检查后只剩 14 个零观察复现候选。
- 14 个候选全部位于 `i_80_run_001`，并集中在同一 capture session 的邻近时段；没有跨 run 覆盖。
- 同一 NGSIM I-80 source/road segment、capture session、Alibaba workflow source 与 controlled typed catalog 仍构成
  共同相关性边界。时间不重叠和零观察车辆复现不能把这些窗口升级为 pristine/external independence。

### 确定性选择规则及失败点

规则在任何新结果之前冻结：从 579-window inventory 开始，排除全历史 used/allocated interval 及 24-frame embargo；
要求 candidate 对历史和 candidate 之间均零 Vehicle_ID recurrence；以
`SHA-256(seed=1423, segment-run, raw-frame-start, raw-time-start)` 排序，跨 run round-robin；至少 3 个 segment-runs、
每 run 最多 4 个，目标 12 个。

现有 metadata 只能从一个 run 提供合格候选；再执行候选间零 Vehicle_ID recurrence 后，确定性 correlation preview
仅剩 2 个窗口，达不到 12-window/3-run 硬条件。
故不形成窗口名单、不创建 split/seal proposal。这个失败与算法效果无关，不能通过放宽相关性条件、改 seed 或先看
结果后换窗口修复。

## 3. 若未来有新数据范围，批准前必须绑定的方案

本轮没有可签发方案；以下只是未来解除 blocker 后仍必须满足的预先条件：

1. 全历史 ledger 扩展必须对新增数据源/segment-run 达到零可能触及目标 scope 的 unresolved references。
2. 结果无关规则必须在执行前冻结 exact 12-window identities、至少 3 runs、每 run 最多 4、pairwise temporal gap 与
   pairwise/history Vehicle_ID recurrence=0；不得按 agent outcome 更换窗口。
3. new split manifest 与 new seal 必须 create-only，绑定本审计、source identity 和 window contract；批准前保持
   `execution_authorized=false / grant_signed=false / seal_created=false`。
4. 现有 frozen candidate/checkpoint manifest 及每个 checkpoint SHA-256 原样绑定；禁止训练、调参、重选、替换或
   从旧 invalid/consumed run salvage。
5. corrected statistics 预绑定：signed-positive claim rule、window-level sign test、84-family Holm、capacity 分开报告、
   completed-workflow conditional delay availability/censoring 与 trade-off claim；不得按新结果删 endpoint/subgroup。
6. 执行 authorization 必须绑定 exact Git commit、benchmark argv、window contract/plan、opening record 和 deterministic
   output identity；opening 后任何失败都永久 consumed。

## 4. 保护与科学限制

- G14C v1–v15 invalid/failed evidence、G14C v16、G14E07 formal output、旧 window plans、ledger、opening state 和
  failure records均未修改、覆盖、移动或删除。
- 本任务训练=0、调参=0、checkpoint replacement=0、原结果修改=0、真实 holdout episode=0、新测试 episode=0。
- G14E07 的 delay survivor conditioning、4/6 typed ablations unavailable、capacity/eviction discrimination、oracle
  gap unavailable、同一 NGSIM+Alibaba+controlled catalog 数据边界均继续存在。
- 接口修复只证明未来在合法 authorization 下存在一致的执行通路；它不签发 authorization，也不把 formal-only
  evidence 升级为 algorithm superiority、canonical 或 paper-ready。

## Evidence inventory

- `artifacts/analysis/g14r23_holdout_interface_and_independent_test_20260927/interface_validation.json`
- `artifacts/analysis/g14r23_holdout_interface_and_independent_test_20260927/targeted_tests_junit.xml`
- `artifacts/analysis/g14r23_holdout_interface_and_independent_test_20260927/full_tests_junit.xml`
- `artifacts/analysis/g14r23_holdout_interface_and_independent_test_20260927/new_independent_test_feasibility.json`
- `artifacts/analysis/g14r23_holdout_interface_and_independent_test_20260927/completion_receipt.json`

验证结果：G14R23 定向测试 `209 passed`；当前提交全量测试 `1366 passed, 2 skipped`，无 failure。两个 skip
为项目既有条件性跳过，不属于 G14R23 新增链路；完整明细保存在上述 JUnit 文件中。

## Safe claims

- dedicated opening 与真实 window consumer 的组合接口已在 synthetic 数据上闭合，且 `identity_only` 没有被滥用。
- 全历史 metadata 审计没有发现满足预冻结相关性条件的 12-window/3-run 新独立测试范围。
- 当前不签 grant、不创建新 seal、不运行新测试是科学上正确的保守结论。

## Prohibited claims

- “真实 sealed holdout 已可直接运行”或“G14R23 已授权 opening”。
- “14 个零车辆复现窗口就是 14 个完全独立样本”。
- “没有 raw overlap 等于 pristine/external independence”。
- “接口通过或存在剩余数据意味着算法 paper-ready”。
