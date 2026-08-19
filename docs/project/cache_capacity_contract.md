# Cache Capacity Contract

`cache_capacity_profile.enabled=false` preserves the historical unbounded cache: no capacity rejection/eviction runs and all capacity snapshots are JSON `null`. `enabled=true, unit=adapter_slots` uses integer `rsu_adapter_slots`; omitted `unit` remains this legacy default. `enabled=true, unit=mb` requires a finite positive `capacity_mb` and measures capacity, used and remaining in resident adapter MB. Unknown units and invalid values fail fast.

```yaml
cache_capacity_profile:
  enabled: true
  unit: mb
  capacity_mb: 256.0
  eviction_policy: lru
  telemetry_enabled: true
```

`AdapterCatalog.resolve_adapter_resident_size_mb()` is the single resolver used by initialization, admission, snapshots and CacheEvent. Explicit `CacheObject.size_mb` yields `catalog_cache_object`; an absent object uses the existing nonzero 64 MB `catalog_fallback`. Missing IDs and non-finite, zero or negative sizes fail fast. Resident and adapter-transfer size currently share this value; state-bundle migration size is separate.

Already-resident admission is a cache-content no-op and records an LRU hit. An object larger than total capacity is rejected as `object_exceeds_total_capacity` before victim planning and without policy-state mutation. Otherwise the environment computes required free capacity, asks the configured policy for a read-only plan, validates it, then atomically removes all victims and adds the object. An insufficient plan causes no partial eviction. Reset uses the same resolver and policy plan to trim initial cache. Initialization trimming is not a request event.

CacheEvent `1.1.0` adds optional `eviction_count`, ordered `evicted_object_ids`, ordered `evicted_adapter_ids`, `evicted_size_mb_sum`, `requested_object_size_mb` and `capacity_rejection_reason`. Legacy singular victim fields identify the first LRU victim. The reducer accepts old 1.0 events (one implied victim) and new 1.x events, while unknown majors fail fast.

Legacy telemetry remains; `cache_capacity_unit` declares the unit. Disabled aggregate capacity/used/remaining remain `null`. In MB mode `rsu_adapter_slots` is compatibility metadata only, not capacity. Policy lifecycle, audit plan and exact LRU ordering are frozen in `cache_eviction_policy_contract.md`. LRU remains an environment primitive, not an evaluated baseline. LFU/FIFO/Random, formal byte metrics and a future-request oracle are not implemented.

## Nullable benchmark aggregation

`cache_capacity`、`cache_used_size`、`cache_remaining_size` 与 `cache_occupancy_rate` 是 nullable capacity snapshots。`null`/字段缺失表示 unavailable 或 not applicable，数值 `0` 表示真实观测零；聚合层不得互换二者。`cache_capacity_enabled` 是实际 0/1 状态指标，不属于 nullable capacity value。

每个 metric 的 aggregate 只对 available finite numeric values 计算 `mean/std/min/max`，并输出 `available_count` 与 `unavailable_count`：

- group 全部 unavailable 时，四个统计值均为 JSON `null`；
- mixed group 忽略 unavailable 样本，仅以 available 数值作为统计分母，同时由两个 count 保留审计范围；
- 真实零参与统计并保持数值 `0.0`；
- pairwise、win/tie/loss、robustness 与报告消费者不得把 unavailable 转为零；无法比较时结果为/保持 `unavailable` 或 `null`。

该规则适用于 benchmark 通用聚合，因而普通完整 numeric metrics 的既有统计值不变；缺失、`null`、非数值或非有限值不再由通用 default-zero 伪造成观测零。

# 2026-08-18 policy compatibility

五种 registered policy 均支持 `adapter_slots` 与 `mb`。`eviction_policy_seed` 与 `eviction_policy_config` 是向后兼容 profile 字段；Random 要求 seed，Aging-LFU 验证 interval/factor。环境继续计算 required free、验证 plan 并原子提交。

## G13 typed profile

`typed_base_adapter_state_v1`只允许enabled MB capacity；base+adapter共享同一per-RSU byte capacity。workflow state固定为handoff payload且不计入，KV disabled。initial typed state必须无需policy-specific trim即可装入。dependency bundle一次计算required-free、一次plan并原子commit；任一对象/整体bundle oversized或victim不足时rollback。snapshot增加used MB by type、bundle MB、admitted/evicted MB by type与orphan=0。legacy profile的disabled/slot/MB语义不变。
