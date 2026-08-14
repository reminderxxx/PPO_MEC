# Cache Event Contract

Version: `1.0.0`

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
- Eviction: `eviction_occurred`, `eviction_policy`, `evicted_object_id`, `evicted_adapter_id`, `eviction_reason`
- Transfer/migration: `adapter_transfer_size_mb`, `state_migration_size_mb`, `transfer_source`, `migration_requested`, `migration_realized`
- Capacity: `cache_capacity_enabled`, `cache_capacity_unit`, `cache_capacity_before`, `cache_used_before`, `cache_remaining_before`, `cache_capacity_after`, `cache_used_after`, `cache_remaining_after`
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

The contract enables request hit/source audit, byte denominators from `size_mb`, admission/eviction counts, transfer volumes and future reuse joins. It does not yet implement byte-hit metrics, cache pollution, eviction regret, latency saved, LRU/LFU/FIFO/Random baselines, byte capacity or an oracle.

Future byte-capacity and LRU/LFU implementations should consume ordered `request` events and before/after capacity snapshots. A future-horizon oracle should join by episode/time/object and compare placement/victim decisions under the same capacity and request stream. Those are separate tasks and must not reinterpret capacity-disabled artifacts.
