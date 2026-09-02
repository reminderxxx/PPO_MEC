# Formal Request Subject Lifecycle Contract 1.0.0

## Scope

Protocol 2.2 为 formal train、dev 与 formal evaluation 冻结同一套“单 workflow、单连续车辆”request subject
语义。合同修复执行身份，不改变 agent architecture、observation/action ID、reward/loss、超参数、seed、capacity、
split、window、workflow、checkpoint cadence、endpoint 或 Holm 规则。

本合同的 readiness 只授权未来独立任务从全新 clean worktree 启动 G14C v12；它不是 formal performance、
holdout、G14、TMC-ready 或 paper-ready 证据。

## Request horizon and eligibility

每个 evaluation unit 的 horizon 为：

```text
request_count = min(DAG execution-order length, max_steps, mobility frame count - 1)
```

候选车辆必须在 frame `0..request_count` 每一帧存在，并在每对相邻帧通过项目既有 displacement/speed
physical-continuity guard。跨 segment-run、同 ID teleport/reuse 或任何中途缺失均不合格。资格只读取冻结
mobility、workflow horizon、RSU layout 与 selection contract；不得读取 action、reward、cache/service outcome、
checkpoint 或 holdout performance。

在合格集合内继续使用既有 `handoff_pressure` 定义和确定性 vehicle-ID tie-break。若未封存 train/dev/formal
单元没有合格车辆，producer 在任何训练前以
`BLOCKED_BY_FORMAL_REQUEST_SUBJECT_ELIGIBILITY` fail-fast；不得缩短 DAG、减少 max_steps、换窗口、删 seed
或回退到动态换车。

## Frozen lifecycle evidence

Exposure trace `2.0.0` 严格记录并纳入 request fingerprint：

- lifecycle contract/version、selection mode/version 与 exposure horizon；
- selected primary vehicle ID、eligible candidate count 与 canonical candidate fingerprint；
- physical-continuity rule/version；
- `reselection_policy=forbidden_during_formal_episode`；
- selection evidence 对 actor/controller 均不可见；
- `outcome_independence=true`。

缺失、额外、错误类型、版本漂移、可见性变化、subject 篡改、非 finite JSON、非 canonical round-trip 或 SHA-256
不一致全部拒绝。execution-specific source provenance 不进入 exposure fingerprint，因此相同 evaluation unit 的
agent、capacity、phase 或 runtime path 不改变外生 demand identity。

## RSU and time semantics

所有 request 使用同一 selected vehicle 和同一 frozen `RSUMapper`：

- `request_rsu_id` 来自 frame `step_index - 1`；
- `current_service_rsu_id` 与 `time_index` 来自 frame `step_index`；
- association 为 `null` 时保留 `null`，不得换车寻找可服务车辆；
- action-before-lookup 语义保持不变。

Runtime reset 从已验证 trace 绑定 subject，并从 frozen mobility 独立复算候选 count/fingerprint、time 和逐步 RSU。
每步 subject 缺失、physical continuity 失败或 request/runtime/CacheEvent identity 漂移均立即失败；runtime 不得反向
改写 exposure。Legacy/non-formal endogenous 路径继续保留原有动态 reselection。

## Consumer closure

同一 lifecycle identity 进入 exposure producer、`VecWorkflowCoreEnv`、training/dev、execution binding、resolved
context、checkpoint provenance、formal cache-policy/controller、CacheEvent alignment、endpoint reducer、fairness、
request replay/oracle、opportunity analyzer、statistics/integrity/gate 与 active bundle。Analytical replay 复用同一
lifecycle producer，只把 frozen exposure 转换为 oracle replay；它不得独立选择另一车辆。

机器可复核的 producer/consumer matrix、18 个 fail-closed 负例、exact v11 failure-unit rehearsal、144-unit
eligibility audit 和 6,480-cell cross-agent/capacity parity 位于：

```text
artifacts/analysis/typed_model_cache_formal_request_subject_repair_20260901_g14r11_v1/
```

## G14C v11 terminal boundary

`typed_model_cache_formal_20260901_155201_g14c_v11` 永久为
`INVALID_PROTOCOL_OR_IMPLEMENTATION / invalid_during_first_training_cell_before_first_episode_commit`。其
resume、retry、finalize、salvage、checkpoint/candidate/partial-output reuse 全部禁止；formal performance evidence
为 0，holdout 保持 sealed/unopened。
