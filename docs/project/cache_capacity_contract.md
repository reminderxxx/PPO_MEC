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

Already-resident admission is a no-op and only touches LRU metadata. An object larger than total capacity is rejected as `object_exceeds_total_capacity` without eviction. Otherwise admission plans the smallest deterministic LRU prefix that frees enough space, then atomically removes victims and adds the object. Reset uses the same resolver/order to trim initial cache. Initialization trimming is not a request event.

CacheEvent `1.1.0` adds optional `eviction_count`, ordered `evicted_object_ids`, ordered `evicted_adapter_ids`, `evicted_size_mb_sum`, `requested_object_size_mb` and `capacity_rejection_reason`. Legacy singular victim fields identify the first LRU victim. The reducer accepts old 1.0 events (one implied victim) and new 1.x events, while unknown majors fail fast.

Legacy telemetry remains; `cache_capacity_unit` declares the unit. Disabled aggregate capacity/used/remaining remain `null`. In MB mode `rsu_adapter_slots` is compatibility metadata only, not capacity. LRU remains an environment primitive, not an evaluated baseline. LFU/FIFO/Random, formal byte metrics and a future-request oracle are not implemented.
