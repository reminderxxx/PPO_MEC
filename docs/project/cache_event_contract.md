# Cache Event Contract

Version: `1.2.0`

## Lifecycle

采用单事件 request lifecycle：每个 episode step 的当前 workflow node 服务请求生成一个 `request` event，cache lookup、最终 hit/miss、admission、eviction、transfer/migration 和 execution result 都是该事件的字段。admission 与 eviction 不生成子事件，因此不会扩大 request denominator。workflow 已无当前节点时仍生成 `not_applicable` event，用于区分“环境明确无请求”和“trace 未记录”。

事件由 `VecWorkflowCoreEnv` 在掌握真实 cache/action/handoff/service 结果的位置生成；evaluator 不依据 reward 或 aggregate 反推。`EpisodeRecorder` 原样保存到 episode summary 的 `cache_event_trace`，并保留 `cache_event_schema_version`。benchmark 的逐 episode `.summary.json` 是 raw trace artifact；既有 row/aggregate 通过 `summary_path` 追溯，不把大型 trace 嵌入 aggregate。

## Frozen enums

- `event_type`: `request`, `not_applicable`
- `object_type`: `adapter`, `not_applicable`
- `hit_source`: `vehicle_local`, `current_rsu`, `target_rsu`, `neighbor_rsu`, `cloud`, `unserved`, `not_applicable`

当前真实 cache object 只冻结 `adapter`。不把不存在的 content 或 inference-result cache 写入正式 schema。

## Fields

- Identity: `event_id`, `event_schema_version`, `event_type`, `time_index`, `episode_step_index`
- Workload: `vehicle_id`, `workflow_id`, `node_id`, `object_id`, `adapter_id`, `object_type`, `size_mb`
- Service path: `request_rsu_id`, `selected_target_rsu_id`, `served_rsu_id`, `predicted_next_rsu_id`, `predicted_handoff_target_rsu_id`, `hit_source`
- Lookup/admission: `cache_lookup_performed`, `cache_hit`, `was_cached_before`, `admission_requested`, `admission_added`, `admission_reason`, `cache_target_rsu_id`
- Eviction: frozen singular fields plus optional 1.1 `eviction_count`, ordered `evicted_object_ids`, ordered `evicted_adapter_ids`, `evicted_size_mb_sum`; singular fields identify the first LRU victim
- Efficiency extension: optional 1.2 `admitted_object_id`, `admitted_adapter_id`, `admitted_size_mb`, ordered `evicted_sizes_mb`; episode summary carries separate `cache_trace_context 1.0.0` initial/final per-RSU snapshots
- Transfer/migration: `adapter_transfer_size_mb`, `state_migration_size_mb`, `transfer_source`, `migration_requested`, `migration_realized`
- Capacity: existing before/after fields plus optional `requested_object_size_mb`, `capacity_rejection_reason`
- Control/result: `action_id`, `action_name`, `cache_strategy`, `offload_mode`, `service_success`, `stall_occurred`, `handoff_event_count`

## Null and not-applicable rules

- Unknown identifiers and unavailable numeric values use JSON `null`; categorical non-applicability uses `not_applicable`.
- Capacity disabled is `cache_capacity_enabled=false` and all six before/after capacity values are `null`. It is not capacity `0` and not occupancy `0`.
- No eviction uses `eviction_occurred=false`, null victim fields and `eviction_reason=not_occurred`.
- `size_mb` comes from catalog `CacheObject`; when the adapter lacks a cache object it uses the catalog's explicit transfer-size fallback and `transfer_source=catalog_fallback`.
- `cache_hit=true` requires exactly one concrete hit source. RSU hit sources are forbidden when `cache_hit=false`.
- `admission_added=true` requires target RSU and object; eviction requires a victim object.

## Compatibility

`step_trace`, `system_metrics`, `prefetch_summary`, `handoff_summary` and existing benchmark row fields are unchanged. Old summaries without `cache_event_trace` remain valid because benchmark conversion does not require the new field. New consumers must use `.get("cache_event_trace", [])` when reading mixed historical/current artifacts.

Schema `1.x` may add optional consumer-safe fields but cannot delete or redefine v1 fields/enums. A changed required-field meaning or enum removal requires a new major version and coordinated producer/consumer migration.

## Current boundary and future consumers

The contract enables request hit/source audit, byte denominators from `size_mb`, admission/eviction counts, transfer volumes, residency reconstruction and future reuse joins. G06 implements the identifiable derived metrics. G08 now provides a separate policy-neutral replay and finite-horizon placement opportunity oracle; it does not reinterpret observed CacheEvent as oracle input. Causal eviction regret and latency saved remain unavailable. Five G05 classical baselines and slot/MB capacity are separate frozen contracts.

Future byte-capacity and LRU/LFU implementations should consume ordered `request` events and before/after capacity snapshots. A future-horizon oracle should join by episode/time/object and compare placement/victim decisions under the same capacity and request stream. Those are separate tasks and must not reinterpret capacity-disabled artifacts.

## Independent reducer and denominator

`src/metrics/cache_event_metrics.py::reduce_cache_events()` is the stateless reference reducer. Its only inputs are `list[CacheEvent | dict]`, the schema version, and frozen G01 fields; it does not read `system_metrics`, step telemetry, reward, evaluator rows or aggregate output. `reduce_cache_event_summary()` is the compatibility wrapper: a missing trace yields `availability=unavailable`, while a present empty trace is available with `total_event_count=0` and undefined (`null`) rates.

Only `event_type=request` enters the request denominator. `not_applicable` is counted separately; admission and eviction remain attributes of one request. `cache_hit=true` defines a hit. `vehicle_local` is a hit under the frozen environment semantics; `cloud` and `unserved` are misses; `not_applicable` is excluded. A zero-request hit or migration rate is `null`, not a fabricated zero.

Execution source is derived without guessing: successful vehicle/cloud requests require matching `offload_mode`; successful RSU requests require `offload_mode=rsu` and non-null `served_rsu_id`; a failed request must be stalled and `unserved`. Contradictions fail fast. Since 1.1, one request may carry an ordered multi-victim eviction.

Current `1.x` events are accepted, including consumer-safe optional fields. Unknown major versions, malformed events, duplicate IDs, hit/source contradictions, invalid admission/eviction/migration lifecycles and invalid capacity-disabled snapshots fail fast.

## Legacy telemetry mapping

| Event-derived field | Legacy candidate | Class | Boundary |
|---|---|---|---|
| `request_event_count` | episode step count | `compatible_but_different_scope` | steps may include `not_applicable` |
| `cache_hit_count` / `cache_miss_count` | `adapter_hit_count` / `adapter_miss_count` | `exact` for G01 current summaries | legacy derives from executable steps |
| `admission_request_count` | `cache_admission_count` | `exact` | one lifecycle admission flag |
| `admission_added_count` | `cache_admission_added_new_adapter_count` | `exact` | new adapter only |
| `eviction_count` | `cache_eviction_count`, `eviction_count` | `exact` | G01 has at most one victim |
| `migration_request_count` | `migration_attempt_count` | `exact` | prepare or migrate request |
| `migration_request_count` | `migration_prepare_count` | `compatible_but_different_scope` | prepare-only summary |
| `migration_realized_count` | `migration_success_count` | `exact` for G01 current summaries | realized prepare or handoff migration |
| `migration_realized_count` | `migration_during_handoff_count` | `compatible_but_different_scope` | handoff-only subset |
| adapter + state transfer MB | `backhaul_traffic_cost` | `compatible_but_different_scope` | event bundle semantics vs handoff-booked migration cost |
| `cache_hit_rate` | `adapter_warm_hit_ratio` | `not_equivalent` | request hit vs step warm readiness |
| `stall_count` | `workflow_continuity_rate` | `not_equivalent` | count vs step rate |
| hit-source distribution | complete legacy field | `unavailable` | old summary has no full distribution |

G06 derived metrics are frozen separately in `cache_efficiency_metrics_contract.md`. Future reuse remains a non-causal proxy; latency saved remains unavailable; G08 cache oracle remains unimplemented. Controlled metrics are not paper performance evidence.
