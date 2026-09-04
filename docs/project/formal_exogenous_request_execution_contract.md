# Formal Exogenous Request Execution Contract

## 冻结身份

- 合同版本：`formal_exogenous_request_execution_contract_version=1.0.0`
- exposure schema：`formal_request_exposure_trace_version=1.0.0`
- endpoint contract：`formal_endpoint_metrics_contract_version=2.0.0`
- active Protocol：`2.3.0`（外生 request 语义保持 2.0 冻结；nullable aggregate 由独立 Contract 1.0.0 约束）
- 默认行为：关闭；Protocol 2.x 的 train/dev/formal 命令必须显式传入 `--formal-exogenous-request-execution`。

## 科学 estimand

每个 evaluation unit 的 request exposure 在任何被比较 agent 执行动作前，由冻结的 mobility、workflow、seed、window、`handoff_pressure` vehicle selection、typed catalog 和 dependency bundle 生成。它与 decision/action、cache/service outcome 和 workflow outcome 分离：后面三者可以因策略而异，但不得添加、删除、重排、重试或抑制已冻结 exposure。

`request_exposure_fingerprint` 只标识外生 exposure；`outcome_fingerprint` 标识策略结果。二者不是同一身份。每个 exposure 必须且只能对齐一个 request-level `CacheEvent 1.3`。

train、dev、formal 均使用 `replay_driven_exogenous_request_exposure`，没有隐藏的训练/评估 progression shift。legacy/non-formal 路径继续默认使用原 endogenous progression。

## 因果与信息边界

每个 request 保留 current/request/eligible RSU、typed base+adapter atomic dependency、DAG provenance 和 actor 不可见的 `oracle_only_future_topology`。执行顺序仍为 action-before-lookup。future topology、cache/service/reward/victim/transfer outcome 不得进入 exposure 或 actor/controller observation；污染或身份漂移立即失败。

G08 replay 仅用于 analytical oracle，不能作为 formal execution producer。G09 opportunity consumer 必须记录 exposure provenance，不能从 outcome 反推 denominator。

## Endpoint 2.0

以下四项共享完全相同的 external request denominator：

- `full_service_ready_byte_hit_rate`
- `joint_base_adapter_hit_rate`
- `full_service_ready_request_rate`
- `transfer_mb_per_request`

`workflow_continuity_rate` 为成功 service outcome 数除以 external exposure 数；predecessor failure 不抑制后续 exposure。`end_to_end_workflow_delay` 仅在 workflow 完整、未 right-censor 且所有 exposure 成功时可用；失败、不完整或 right-censored 时为 `null/unavailable`，不得用零、reward 或早终止时长替代。跨 episode 聚合、Dev 选择和统计消费遵循 `formal_nullable_metric_aggregation_contract.md`。

## Fail-fast 边界

缺失、duplicate、extra、out-of-order、跨 agent fingerprint 不同、event/replay identity 漂移、catalog/dependency/size/evaluation-unit 漂移、outcome 污染、formal endogenous fallback、future leak、historical active bundle 或任一永久 invalid G14C run（包括 v12）路径/checkpoint 引用都必须终止执行。
