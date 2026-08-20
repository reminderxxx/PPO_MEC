# Cache Efficiency Metrics Contract

版本：`1.2.0`（G14R primary endpoint extension，2026-08-20；G06/G13 legacy 指标保持兼容）
输入：raw `CacheEvent 1.x` trace + 可选 `cache_trace_context 1.0.0`。参考实现为 `src/metrics/cache_efficiency_metrics.py`。

## 总原则

Reducer 是纯函数，不读取 reward、`system_metrics`、legacy step telemetry 或 benchmark aggregate。只有 `event_type=request` 进入 request/byte 分母；`not_applicable` 排除。未知值使用 JSON `null`，零分母 rate 为 `null`。非有限/负数、重复 event ID、矛盾 lifecycle、episode 内 capacity unit/value 变化和 residency 不一致均 fail-fast。

每组输出 `availability={available,partial,unavailable,not_applicable}`、`unavailable_reason`、`required_fields`、available/unavailable event count。部分缺失不得转成零。

## Request 与 byte efficiency

- `request_count = count(request events)`，单位 request。
- `object_hit_count = count(cache_hit=true)`；`object_miss_count=request_count-object_hit_count`。
- `object_hit_rate=object_hit_count/request_count`。
- hit-source count/rate 分母均为 `request_count`。`vehicle_local` 遵循冻结的 `cache_hit=true` 语义计为 hit；`cloud`、`unserved` 为 miss。
- `requested_size_mb_sum = Σ request.size_mb`；`hit_size_mb_sum = Σ size_mb where cache_hit=true`；`miss_size_mb_sum` 同理。
- `byte_hit_rate=hit_size_mb_sum/requested_size_mb_sum`。按 source 的 MB/rate 使用同一 byte denominator。

Byte 采用严格 complete-case contract：任一 request 的 `size_mb=null` 时 byte group 为 partial/unavailable，所有总 MB 和 byte rate 为 `null`；同时输出 count/rate coverage。NaN、inf、负值直接报错。合法全零 byte denominator 的 rate 为 `null`。

## Admission、eviction、churn 与 transfer

- admission requested/added count 分别计 flag；`admitted_size_mb_sum=Σ admitted_size_mb`。
- rejection count 以 requested 且未 added 计数，按 `capacity_rejection_reason`（缺失时 admission reason）分解。
- oversized rejection 是 `capacity_rejection_reason=object_exceeds_total_capacity`；MB numerator 为 `requested_object_size_mb`。
- `eviction_event_count` 计发生 eviction 的 request；`eviction_victim_count=Σ eviction_count`；multi-victim 逐 victim 保留 identity/size。
- `cache_churn_mb=admitted_size_mb_sum+evicted_size_mb_sum`。
- `total_transfer_size_mb_sum=adapter_transfer_size_mb_sum+state_migration_size_mb_sum`。
- `transfer_per_hit_mb=total_transfer_size_mb_sum/object_hit_count`，单位 MB/hit。
- `transfer_amplification_ratio=total_transfer_size_mb_sum/hit_size_mb_sum`，无量纲。

Slot capacity 不会被当作 MB。即使 capacity unit 为 `adapter_slots`，object/transfer 流量仍使用 catalog `size_mb`。

## Capacity utilization

Enabled trace 的每个 request 提供 before/after observation：`occupancy=used/capacity`。`mean_occupancy` 与 `peak_occupancy` 分别对全部有效 observation 求均值/最大值；saturation event 是 before 或 after `remaining=0` 的 request，rate 分母为 enabled request count。`rejected_due_to_capacity_count` 计非空 `capacity_rejection_reason`。

Capacity disabled 输出 `not_applicable` 和 null metrics，不是零。episode 内 enabled 状态、unit 或 capacity value 改变，以及 `used+remaining != capacity` 均报错。跨 episode aggregate 只聚合相同 metric 的 available finite values，并保留 available/unavailable count；unit 必须作为分组/provenance 字段，禁止混合 slot/MB 后赋予同一物理单位。

## Pollution 与 censoring

`cache_trace_context 1.0.0` 提供 initial/final per-RSU resident `(object_id, adapter_id, size_mb)` snapshot 和 `episode_end_step_index`。事件按 step 重建 resident interval；RSU hit 标记该 resident 在本 interval 内被复用。

- `unused_admitted_object_count/size_mb`：admission 后直到 eviction 都未产生该 RSU/object cache hit、且已由 eviction 完整闭合的 admission。
- `polluted_resident_mb_steps=Σ(size_mb × resident step duration)`，只对上述已闭合未复用 admission。
- `total_resident_mb_steps` 包含 initial resident 与 runtime admission 的全部可重建 interval。
- `cache_pollution_ratio=polluted_resident_mb_steps/total_resident_mb_steps`。

episode-end 未复用但仍 resident 的 admission 是 right-censored，单独输出 count/MB，排除于 pollution numerator；不能称为永久污染。final snapshot 必须与事件重建完全一致，否则 fail-fast。旧 trace 或缺 context 时本组为 unavailable。

## Eviction future-reuse proxy

默认 horizon `H={1,3,6,12}` step。对每个 victim（包括 multi-victim）查找同 episode 后续相同 `object_id` 的首个 request：

- `evicted_then_requested_within_h_count/mb`：next request gap `<=H` 的 victim count/size。
- `eviction_future_reuse_rate_h=count/referenced victim count`。
- `time_to_next_request_after_eviction_steps_mean`：有后续请求 victim 的平均 gap。

这是 future-request reuse diagnostic proxy，不是 causal eviction regret，也不是相同容量/请求流的 oracle gap。G08 才负责 counterfactual oracle；本 Goal 不实现。

## Latency saved 可识别性

当前环境没有逐 request 对齐且单位明确的 observed service latency、同请求 cold/cloud counterfactual latency、transfer latency 与 stall/restart latency。因此 `latency_saved_metrics.availability=unavailable`，所有 latency-saved 数值为 `null`，并列出所需字段。reward delay、workflow span 或不同请求平均 delay 均不得替代。

## Schema 与消费端

`CacheEvent 1.2.0` 仅新增 optional `admitted_object_id/admitted_adapter_id/admitted_size_mb/evicted_sizes_mb`；不删除或重定义 1.0/1.1 字段。`EpisodeRecorder` 新增 `cache_trace_context`。旧 summary 继续读取；缺证据的组返回 partial/unavailable。

Benchmark row 只接入 object/byte hit、churn、pollution、transfer amplification、capacity occupancy 与 latency-saved nullable scalar；大型 trace 仍保留在 episode summary。aggregate 不把 null 变成零。

审计入口：

```bash
.venv/bin/python scripts/audit_cache_efficiency_metrics.py --summary_path <episode.summary.json> --output_path <new.audit.json>
```

这些结果是 contract/机制验证。Smoke、synthetic、controlled classical baseline 或最小 real dry-run 均不能支持 paper-ready、算法 superiority、causal regret 或 latency-saved claim。

## G13 type-aware metrics 1.1.0

CacheEvent 1.3 raw fields独立重算base/adapter/joint hit、state/full readiness、missing type、compatibility failure、per-type requested/hit/resident/admitted/evicted/transfer MB、base/adapter occupancy、pinned MB、bundle rejection/churn、adapters per base、base reuse/sharing、严格可识别的avoided base transfer与orphan。pollution重建支持multi-object typed admission/eviction，right-censoring不变。旧1.0–1.2 trace的type-aware group为unavailable而非0。latency saved仍unavailable。

G14A benchmark raw summary继续保留完整CacheEvent 1.3与trace context；`summary_to_row()`只输出nullable scalar和轻量runtime/fairness/checkpoint provenance，aggregate不复制raw event。`typed_model_cache_runtime_plumbing_validation_20260819_g14a_v1`对28个typed episode从raw event独立重算metrics 1.1并与benchmark scalar一致；这只是reconciliation，不是性能比较。

## G14R primary endpoints 1.2.0

`full_service_ready_byte_hit_rate` 的 eligible unit 是 typed request event。每个 request 将其 unique
dependency objects 的 `resident_size_mb` 求和，依赖类型为 base model + adapter；同一 base 在同一
request 只计一次，不按 lookup row 或 resident inventory 重复，但在另一个独立 request 中作为该请求
所需 service bytes 再计一次。`full_service_ready=true` 时该 request 的全部 dependency bytes 进入
numerator；partial readiness 的整请求 numerator 为 0。任一 eligible request 缺 dependency size 时
primary value 为 `null/partial` 并输出 coverage；合法零 denominator 为 `null`；legacy trace 为
`unavailable`。

`transfer_mb_per_request = (base_model_transfer_mb + adapter_transfer_mb +
workflow_state_migration_transfer_mb) / typed_request_event_count`，单位 decimal MB/request。
`other_typed_transfer_mb` 独立报告且不进入 primary total，避免把未知 migration 类型静默混入。

参考 reducer、episode summary、benchmark row 与 nullable aggregate 使用同名 canonical 字段。
若 summary 中已有 reducer snapshot，`cache_efficiency_row_fields()` 必须与 raw event 重算完全一致，
否则 fail-fast。G14R 非正式 rehearsal 对 36 个 summary 完成六 primary fields 的 raw/row 对账；
这不是正式性能证据。
