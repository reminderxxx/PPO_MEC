# Cache Eviction Policy Contract

## Scope and identity

`src/envs/core/cache_eviction.py` defines the cache-victim policy boundary. The factory accepts a case-insensitive canonical name and currently registers only `lru`; unknown names fail fast and never fall back silently. LRU identity is `policy_name=lru`, `policy_version=1.0.0`, `deterministic=true`, `requires_seed=false`, with `adapter_slots` and `mb` support. A seed is accepted for a stable future factory signature but LRU does not consume it.

This is a structural foundation for fair policy work, not a performance improvement or a formal benchmark baseline. FIFO, LFU, Aging-LFU and Random are not implemented.

## Responsibility boundary

`VecWorkflowCoreEnv` owns capacity enablement and unit/value parsing, resident-size resolution, oversized rejection, required-free-capacity arithmetic, plan validation, atomic cache/catalog mutation, admission, CacheEvent and legacy telemetry. The eviction policy owns per-RSU ordering metadata, deterministic victim planning and detached audit export. It does not own reward, workflow/handoff/offloading logic, actor actions, catalog writes, benchmark output or artifact paths.

`plan_victims` is read-only. It receives the RSU, current residents and sizes, required free capacity, protected incoming object, capacity unit and episode step. It returns ordered victims and sizes, cumulative freed capacity, sufficient/insufficient status, policy identity, ordered candidates, recency evidence and a selection reason. The environment applies no victims unless the complete plan is sufficient.

## Lifecycle

- `reset`: `rsu_id=None` clears the whole episode; per-RSU calls initialize ordered resident state. RSUs are isolated and no state survives the next episode.
- `on_admission`: called only after an object is actually added. Rejected and oversized requests do not enter state.
- `on_hit`: called only for a real hit at the RSU cache that served it. Miss, vehicle, cloud and unserved paths do not touch an RSU entry. Already-resident cache actions count as a hit, not a new admission.
- `on_eviction`: called only after the environment removes the object; an unapplied plan cannot clean state.
- `plan_victims`: creates a deterministic, non-mutating minimum-prefix plan.
- `export_state`: returns JSON-safe identity, selection key, clock, per-RSU oldest-first order and resident metadata without exposing live Python objects.

## Frozen LRU semantics

Initial residents preserve catalog order. For `N` residents their initial `last_used_step` values are `-N, ..., -1`, so the first listed resident is oldest. Runtime admission and real hit use the integer episode step, not trace wall-clock. Victims sort by `(last_used_step, adapter_id)`; therefore simultaneous operations at one step use lexical adapter ID as the G03-compatible tie-break. An admission becomes most recent, and a real hit refreshes recency.

Slot mode gives every resident size `1`. MB mode uses the environment's validated resident size. Multi-victim selection returns the oldest-to-newest minimum prefix whose cumulative size reaches required free capacity. The incoming object is protected, non-residents cannot be selected, and identical inputs/state produce identical plans.

Initialization trimming and runtime admission use the same policy state and plan contract. Disabled capacity does not invoke victim planning. Oversized objects are rejected before planning. Already-cached requests do not admit or evict. Environment-side eviction plus admission remains atomic.

## Audit and compatibility

An `EvictionPlan` contains `policy_name`, `policy_version`, `ordered_candidates`, `candidate_recency`, `ordered_victim_ids`, `victim_sizes`, `required_free_capacity`, `cumulative_freed_capacity`, `capacity_unit`, `sufficient` and `selection_reason`. `VecWorkflowCoreEnv.export_cache_eviction_policy_state()` and `export_last_eviction_plan()` expose detached debug snapshots; existing CacheEvent 1.1 fields and legacy first-victim compatibility are unchanged.

G03 parity evidence is produced by `scripts/validate_cache_eviction_policy.py` under `artifacts/analysis/cache_eviction_policy_lru_validation_20260817_v1/`. It covers disabled admission, slot and MB victims, MB multi-victim order, hit refresh, same-step tie-break, RSU/reset isolation, initial trim, final cache and exported state. This artifact is contract validation only and cannot support claims of LRU performance superiority.

# 1.1 classical policy extension (2026-08-18)

Factory 精确注册 `lru/fifo/lfu/aging_lfu/random`。冻结排序、aging 与 seed 语义见 `classical_cache_baseline_contract.md`。`EvictionPlan.candidate_recency` 保持 1.x 兼容字段名并承载各 policy evidence；未升级 CacheEvent schema。

## G13 typed object generalization

policy boundary本来使用`object_id`，legacy环境实际传adapter ID；typed profile现在传stable typed object ID。五policy算法不变。typed victim candidate排除pinned/non-evictable对象，并冻结“base仍有resident adapter依赖时禁止evictbase”；一次bundle只调用一次read-only multi-victim plan。没有dependency-safe sufficient plan时环境不应用任何victim。legacy adapter victim排序/seed parity保持不变。
