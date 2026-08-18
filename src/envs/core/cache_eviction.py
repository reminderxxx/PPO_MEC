"""Stable, auditable cache-eviction policy contract and LRU implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import math
import random
from typing import Any, Mapping, Sequence


CACHE_EVICTION_POLICY_VERSION = "1.0.0"
CACHE_CAPACITY_UNITS = frozenset({"adapter_slots", "mb"})
CACHE_CAPACITY_EPSILON = 1.0e-9


@dataclass(frozen=True)
class EvictionPlan:
    """Serializable, non-mutating victim-selection result."""

    rsu_id: str
    ordered_victim_ids: list[str]
    victim_sizes: list[float]
    cumulative_freed_capacity: float
    required_free_capacity: float
    capacity_unit: str
    policy_name: str
    policy_version: str
    ordered_candidates: list[str]
    candidate_recency: list[dict[str, Any]]
    sufficient: bool
    selection_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvictionPolicy(ABC):
    """Lifecycle contract for environment-owned, atomic cache admission."""

    policy_name: str
    policy_version: str
    deterministic: bool
    requires_seed: bool
    capacity_units_supported: frozenset[str]

    @abstractmethod
    def reset(
        self,
        *,
        rsu_id: str | None = None,
        initial_resident_ids: Sequence[str] = (),
        current_step: int = 0,
    ) -> None:
        """Clear the episode when rsu_id is None, or initialize one RSU."""

    @abstractmethod
    def on_admission(self, *, rsu_id: str, object_id: str, current_step: int) -> None:
        """Record an object only after the environment admits it."""

    @abstractmethod
    def on_hit(self, *, rsu_id: str, object_id: str, current_step: int) -> None:
        """Record a real hit at the RSU that served it."""

    @abstractmethod
    def on_eviction(self, *, rsu_id: str, object_id: str, current_step: int) -> None:
        """Remove policy metadata only after the environment applies eviction."""

    @abstractmethod
    def plan_victims(
        self,
        *,
        rsu_id: str,
        resident_ids: Sequence[str],
        resident_sizes: Mapping[str, float],
        required_free_capacity: float,
        protected_object_id: str | None,
        capacity_unit: str,
        current_step: int,
    ) -> EvictionPlan:
        """Return an ordered plan without mutating cache or policy state."""

    @abstractmethod
    def export_state(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable policy-state snapshot."""


@dataclass
class _LRUEntry:
    last_used_step: int
    admitted_step: int
    initial_admission_order: int | None
    last_event: str
    last_event_sequence: int


class LRUEvictionPolicy(EvictionPolicy):
    """G03-compatible deterministic least-recently-used eviction policy."""

    policy_name = "lru"
    policy_version = CACHE_EVICTION_POLICY_VERSION
    deterministic = True
    requires_seed = False
    capacity_units_supported = CACHE_CAPACITY_UNITS

    def __init__(self, *, seed: int | None = None) -> None:
        # Accepted for a stable factory signature; deterministic LRU never consumes it.
        self._seed_supplied = seed is not None
        self._rsu_entries: dict[str, dict[str, _LRUEntry]] = {}
        self._event_sequence = 0

    def reset(
        self,
        *,
        rsu_id: str | None = None,
        initial_resident_ids: Sequence[str] = (),
        current_step: int = 0,
    ) -> None:
        if rsu_id is None:
            self._rsu_entries = {}
            self._event_sequence = 0
            return
        ordered_ids = list(dict.fromkeys(str(item) for item in initial_resident_ids))
        entry_count = len(ordered_ids)
        entries: dict[str, _LRUEntry] = {}
        for index, object_id in enumerate(ordered_ids):
            self._event_sequence += 1
            entries[object_id] = _LRUEntry(
                last_used_step=-entry_count + index,
                admitted_step=int(current_step),
                initial_admission_order=index,
                last_event="reset_initial_admission",
                last_event_sequence=self._event_sequence,
            )
        self._rsu_entries[str(rsu_id)] = entries

    def on_admission(self, *, rsu_id: str, object_id: str, current_step: int) -> None:
        self._record_touch(
            rsu_id=rsu_id,
            object_id=object_id,
            current_step=current_step,
            event="admission",
            is_admission=True,
        )

    def on_hit(self, *, rsu_id: str, object_id: str, current_step: int) -> None:
        self._record_touch(
            rsu_id=rsu_id,
            object_id=object_id,
            current_step=current_step,
            event="hit",
            is_admission=False,
        )

    def _record_touch(
        self,
        *,
        rsu_id: str,
        object_id: str,
        current_step: int,
        event: str,
        is_admission: bool,
    ) -> None:
        rsu_key = str(rsu_id)
        object_key = str(object_id)
        entries = self._rsu_entries.setdefault(rsu_key, {})
        previous = entries.get(object_key)
        self._event_sequence += 1
        entries[object_key] = _LRUEntry(
            last_used_step=int(current_step),
            admitted_step=int(current_step) if is_admission or previous is None else previous.admitted_step,
            initial_admission_order=(None if previous is None else previous.initial_admission_order),
            last_event=event,
            last_event_sequence=self._event_sequence,
        )

    def on_eviction(self, *, rsu_id: str, object_id: str, current_step: int) -> None:
        del current_step
        self._event_sequence += 1
        self._rsu_entries.setdefault(str(rsu_id), {}).pop(str(object_id), None)

    def plan_victims(
        self,
        *,
        rsu_id: str,
        resident_ids: Sequence[str],
        resident_sizes: Mapping[str, float],
        required_free_capacity: float,
        protected_object_id: str | None,
        capacity_unit: str,
        current_step: int,
    ) -> EvictionPlan:
        del current_step
        unit = str(capacity_unit).strip().lower()
        if unit not in self.capacity_units_supported:
            raise ValueError(f"eviction policy lru does not support capacity unit: {capacity_unit}")
        required = float(required_free_capacity)
        if not math.isfinite(required) or required < 0.0:
            raise ValueError("required_free_capacity must be finite and non-negative")
        entries = self._rsu_entries.get(str(rsu_id), {})
        protected = None if protected_object_id is None else str(protected_object_id)
        residents = list(dict.fromkeys(str(item) for item in resident_ids))
        candidates = [item for item in residents if item != protected]
        candidates.sort(key=lambda item: (entries.get(item, _LRUEntry(-10**9, 0, None, "missing", 0)).last_used_step, item))

        evidence = []
        for rank, object_id in enumerate(candidates):
            entry = entries.get(object_id)
            evidence.append(
                {
                    "rank": rank,
                    "object_id": object_id,
                    "last_used_step": entry.last_used_step if entry else -10**9,
                    "admitted_step": entry.admitted_step if entry else None,
                    "initial_admission_order": entry.initial_admission_order if entry else None,
                    "last_event": entry.last_event if entry else "missing_policy_state",
                    "selection_key": [entry.last_used_step if entry else -10**9, object_id],
                }
            )

        selected: list[str] = []
        sizes: list[float] = []
        freed = 0.0
        for object_id in candidates:
            if freed + CACHE_CAPACITY_EPSILON >= required:
                break
            size = float(resident_sizes.get(object_id, 0.0))
            if not math.isfinite(size) or size <= 0.0:
                raise ValueError(f"invalid resident size for eviction candidate {object_id!r}: {size!r}")
            selected.append(object_id)
            sizes.append(size)
            freed += size
        sufficient = freed + CACHE_CAPACITY_EPSILON >= required
        reason = (
            "no_eviction_required"
            if required <= CACHE_CAPACITY_EPSILON
            else "lru_minimum_prefix_sufficient"
            if sufficient
            else "insufficient_evictable_capacity"
        )
        return EvictionPlan(
            rsu_id=str(rsu_id),
            ordered_victim_ids=selected,
            victim_sizes=sizes,
            cumulative_freed_capacity=freed,
            required_free_capacity=required,
            capacity_unit=unit,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            ordered_candidates=candidates,
            candidate_recency=evidence,
            sufficient=sufficient,
            selection_reason=reason,
        )

    def export_state(self) -> dict[str, Any]:
        rsus: dict[str, Any] = {}
        for rsu_id, entries in sorted(self._rsu_entries.items()):
            order = sorted(entries, key=lambda item: (entries[item].last_used_step, item))
            rsus[rsu_id] = {
                "lru_order_oldest_first": order,
                "resident_metadata": {
                    object_id: asdict(entries[object_id]) for object_id in sorted(entries)
                },
            }
        return {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "deterministic": self.deterministic,
            "requires_seed": self.requires_seed,
            "capacity_units_supported": sorted(self.capacity_units_supported),
            "selection_key": ["last_used_step", "object_id"],
            "clock": "episode_step",
            "seed_consumed": False,
            "rsus": rsus,
        }


@dataclass
class _FrequencyEntry:
    frequency: int
    last_used_step: int
    admitted_step: int
    admission_order: int
    last_event: str


class FIFOEvictionPolicy(EvictionPolicy):
    policy_name = "fifo"
    policy_version = CACHE_EVICTION_POLICY_VERSION
    deterministic = True
    requires_seed = False
    capacity_units_supported = CACHE_CAPACITY_UNITS

    def __init__(self, *, seed: int | None = None) -> None:
        self._seed_supplied = seed is not None
        self._entries: dict[str, dict[str, dict[str, Any]]] = {}
        self._rsu_sequences: dict[str, int] = {}

    def reset(self, *, rsu_id: str | None = None, initial_resident_ids: Sequence[str] = (), current_step: int = 0) -> None:
        if rsu_id is None:
            self._entries, self._rsu_sequences = {}, {}
            return
        rsu_key = str(rsu_id)
        entries: dict[str, dict[str, Any]] = {}
        for index, object_id in enumerate(dict.fromkeys(map(str, initial_resident_ids))):
            entries[object_id] = {"admission_order": index, "admitted_step": int(current_step), "initial": True}
        self._entries[rsu_key] = entries
        self._rsu_sequences[rsu_key] = len(entries)

    def on_admission(self, *, rsu_id: str, object_id: str, current_step: int) -> None:
        rsu_key = str(rsu_id)
        entries = self._entries.setdefault(rsu_key, {})
        if str(object_id) in entries:
            return
        order = self._rsu_sequences.get(rsu_key, len(entries))
        self._rsu_sequences[rsu_key] = order + 1
        entries[str(object_id)] = {"admission_order": order, "admitted_step": int(current_step), "initial": False}

    def on_hit(self, *, rsu_id: str, object_id: str, current_step: int) -> None:
        del rsu_id, object_id, current_step

    def on_eviction(self, *, rsu_id: str, object_id: str, current_step: int) -> None:
        del current_step
        self._entries.setdefault(str(rsu_id), {}).pop(str(object_id), None)

    def plan_victims(self, **kwargs: Any) -> EvictionPlan:
        entries = self._entries.get(str(kwargs["rsu_id"]), {})
        return _ordered_plan(
            policy=self, kwargs=kwargs,
            key=lambda item: (int(entries.get(item, {}).get("admission_order", -10**9)), item),
            evidence=lambda rank, item: {"rank": rank, "object_id": item, **entries.get(item, {}), "selection_key": [entries.get(item, {}).get("admission_order", -10**9), item]},
            reason="fifo_minimum_prefix_sufficient",
        )

    def export_state(self) -> dict[str, Any]:
        return _export_mapping_state(self, self._entries, "fifo_order_oldest_first", lambda entries, item: (entries[item]["admission_order"], item), {"selection_key": ["admission_order", "object_id"], "seed_consumed": False})


class LFUEvictionPolicy(EvictionPolicy):
    policy_name = "lfu"
    policy_version = CACHE_EVICTION_POLICY_VERSION
    deterministic = True
    requires_seed = False
    capacity_units_supported = CACHE_CAPACITY_UNITS

    def __init__(self, *, seed: int | None = None) -> None:
        self._seed_supplied = seed is not None
        self._entries: dict[str, dict[str, _FrequencyEntry]] = {}
        self._sequence = 0

    def reset(self, *, rsu_id: str | None = None, initial_resident_ids: Sequence[str] = (), current_step: int = 0) -> None:
        if rsu_id is None:
            self._entries, self._sequence = {}, 0
            return
        entries: dict[str, _FrequencyEntry] = {}
        for object_id in dict.fromkeys(map(str, initial_resident_ids)):
            self._sequence += 1
            entries[object_id] = _FrequencyEntry(0, int(current_step), int(current_step), self._sequence, "reset_initial_admission")
        self._entries[str(rsu_id)] = entries

    def on_admission(self, *, rsu_id: str, object_id: str, current_step: int) -> None:
        entries = self._entries.setdefault(str(rsu_id), {})
        if str(object_id) in entries:
            return
        self._sequence += 1
        entries[str(object_id)] = _FrequencyEntry(0, int(current_step), int(current_step), self._sequence, "admission")

    def on_hit(self, *, rsu_id: str, object_id: str, current_step: int) -> None:
        entries = self._entries.setdefault(str(rsu_id), {})
        item = str(object_id)
        entry = entries.get(item)
        if entry is None:
            self._sequence += 1
            entry = _FrequencyEntry(0, int(current_step), int(current_step), self._sequence, "implicit_hit_state")
        entry.frequency += 1
        entry.last_used_step = int(current_step)
        entry.last_event = "hit"
        entries[item] = entry

    def on_eviction(self, *, rsu_id: str, object_id: str, current_step: int) -> None:
        del current_step
        self._entries.setdefault(str(rsu_id), {}).pop(str(object_id), None)

    def _selection_key(self, entries: Mapping[str, _FrequencyEntry], item: str) -> tuple[Any, ...]:
        entry = entries.get(item, _FrequencyEntry(0, -10**9, 0, 0, "missing"))
        return (entry.frequency, entry.last_used_step, item)

    def plan_victims(self, **kwargs: Any) -> EvictionPlan:
        entries = self._entries.get(str(kwargs["rsu_id"]), {})
        return _ordered_plan(policy=self, kwargs=kwargs, key=lambda item: self._selection_key(entries, item), evidence=lambda rank, item: self._frequency_evidence(entries, rank, item), reason=f"{self.policy_name}_minimum_prefix_sufficient")

    def _frequency_evidence(self, entries: Mapping[str, _FrequencyEntry], rank: int, item: str) -> dict[str, Any]:
        entry = entries.get(item, _FrequencyEntry(0, -10**9, 0, 0, "missing_policy_state"))
        return {"rank": rank, "object_id": item, **asdict(entry), "selection_key": list(self._selection_key(entries, item))}

    def export_state(self) -> dict[str, Any]:
        rsus = {}
        for rsu_id, entries in sorted(self._entries.items()):
            rsus[rsu_id] = {"eviction_order_first": sorted(entries, key=lambda item: self._selection_key(entries, item)), "resident_metadata": {item: asdict(entries[item]) for item in sorted(entries)}}
        return _base_export(self, rsus, {"initial_frequency": 0, "selection_key": ["frequency", "last_used_step", "object_id"], "seed_consumed": False})


class AgingLFUEvictionPolicy(LFUEvictionPolicy):
    policy_name = "aging_lfu"

    def __init__(self, *, seed: int | None = None, aging_interval: int = 8, aging_factor: float = 0.5) -> None:
        if isinstance(aging_interval, bool) or not isinstance(aging_interval, int) or aging_interval <= 0:
            raise ValueError("aging_interval must be a positive integer")
        if not math.isfinite(float(aging_factor)) or not 0.0 < float(aging_factor) < 1.0:
            raise ValueError("aging_factor must be finite and strictly between 0 and 1")
        super().__init__(seed=seed)
        self.aging_interval = int(aging_interval)
        self.aging_factor = float(aging_factor)
        self._clocks: dict[str, int] = {}
        self._last_aging_event: dict[str, int] = {}

    def reset(self, **kwargs: Any) -> None:
        rsu_id = kwargs.get("rsu_id")
        super().reset(**kwargs)
        if rsu_id is None:
            self._clocks, self._last_aging_event = {}, {}
        else:
            self._clocks[str(rsu_id)] = 0
            self._last_aging_event[str(rsu_id)] = 0

    def _advance(self, rsu_id: str) -> None:
        key = str(rsu_id)
        clock = self._clocks.get(key, 0) + 1
        self._clocks[key] = clock
        if clock % self.aging_interval == 0:
            for entry in self._entries.setdefault(key, {}).values():
                entry.frequency = max(0, math.floor(entry.frequency * self.aging_factor))
            self._last_aging_event[key] = clock

    def on_admission(self, **kwargs: Any) -> None:
        self._advance(str(kwargs["rsu_id"]))
        super().on_admission(**kwargs)

    def on_hit(self, **kwargs: Any) -> None:
        self._advance(str(kwargs["rsu_id"]))
        super().on_hit(**kwargs)

    def on_eviction(self, **kwargs: Any) -> None:
        self._advance(str(kwargs["rsu_id"]))
        super().on_eviction(**kwargs)

    def export_state(self) -> dict[str, Any]:
        state = super().export_state()
        state.update({"aging_interval": self.aging_interval, "aging_factor": self.aging_factor, "aging_timing": "before_hit_admission_or_eviction", "aging_events": "per_rsu_policy_callbacks", "rsu_clocks": dict(sorted(self._clocks.items())), "last_aging_event": dict(sorted(self._last_aging_event.items()))})
        return state


class RandomEvictionPolicy(EvictionPolicy):
    policy_name = "random"
    policy_version = CACHE_EVICTION_POLICY_VERSION
    deterministic = False
    requires_seed = True
    capacity_units_supported = CACHE_CAPACITY_UNITS

    def __init__(self, *, seed: int | None = None) -> None:
        if seed is None:
            raise ValueError("random eviction policy requires an explicit seed")
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self._residents: dict[str, set[str]] = {}

    def reset(self, *, rsu_id: str | None = None, initial_resident_ids: Sequence[str] = (), current_step: int = 0) -> None:
        del current_step
        if rsu_id is None:
            self._rng = random.Random(self._seed)
            self._residents = {}
            return
        self._residents[str(rsu_id)] = set(map(str, initial_resident_ids))

    def on_admission(self, *, rsu_id: str, object_id: str, current_step: int) -> None:
        del current_step
        self._residents.setdefault(str(rsu_id), set()).add(str(object_id))

    def on_hit(self, *, rsu_id: str, object_id: str, current_step: int) -> None:
        del rsu_id, object_id, current_step

    def on_eviction(self, *, rsu_id: str, object_id: str, current_step: int) -> None:
        del current_step
        self._residents.setdefault(str(rsu_id), set()).discard(str(object_id))

    def plan_victims(self, **kwargs: Any) -> EvictionPlan:
        residents = list(dict.fromkeys(map(str, kwargs["resident_ids"])))
        protected = None if kwargs.get("protected_object_id") is None else str(kwargs["protected_object_id"])
        eligible = sorted(item for item in residents if item != protected)
        sampled = self._rng.sample(eligible, len(eligible))
        return _ordered_plan(policy=self, kwargs=kwargs, ordered=sampled, evidence=lambda rank, item: {"rank": rank, "object_id": item, "seed": self._seed, "eligible_candidate_order": eligible, "sampled_rank": rank}, reason="random_sampled_minimum_prefix_sufficient")

    def export_state(self) -> dict[str, Any]:
        return _base_export(self, {key: {"resident_ids": sorted(value)} for key, value in sorted(self._residents.items())}, {"seed": self._seed, "seed_identity": "explicit_policy_seed", "reproducibility": "seeded_reproducible", "rng_private": True})


def _base_export(policy: EvictionPolicy, rsus: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    return {"policy_name": policy.policy_name, "policy_version": policy.policy_version, "deterministic": policy.deterministic, "requires_seed": policy.requires_seed, "capacity_units_supported": sorted(policy.capacity_units_supported), "rsus": rsus, **extra}


def _export_mapping_state(policy: EvictionPolicy, source: Mapping[str, Mapping[str, Any]], order_key: str, key: Any, extra: dict[str, Any]) -> dict[str, Any]:
    rsus = {rsu_id: {order_key: sorted(entries, key=lambda item: key(entries, item)), "resident_metadata": {item: dict(entries[item]) for item in sorted(entries)}} for rsu_id, entries in sorted(source.items())}
    return _base_export(policy, rsus, extra)


def _ordered_plan(*, policy: EvictionPolicy, kwargs: Mapping[str, Any], evidence: Any, reason: str, key: Any | None = None, ordered: Sequence[str] | None = None) -> EvictionPlan:
    unit = str(kwargs["capacity_unit"]).strip().lower()
    if unit not in policy.capacity_units_supported:
        raise ValueError(f"eviction policy {policy.policy_name} does not support capacity unit: {unit}")
    required = float(kwargs["required_free_capacity"])
    if not math.isfinite(required) or required < 0:
        raise ValueError("required_free_capacity must be finite and non-negative")
    protected = None if kwargs.get("protected_object_id") is None else str(kwargs["protected_object_id"])
    candidates = list(ordered) if ordered is not None else [item for item in dict.fromkeys(map(str, kwargs["resident_ids"])) if item != protected]
    if key is not None:
        candidates.sort(key=key)
    sizes_map = kwargs["resident_sizes"]
    selected, sizes, freed = [], [], 0.0
    for item in candidates:
        if freed + CACHE_CAPACITY_EPSILON >= required:
            break
        size = float(sizes_map.get(item, 0.0))
        if not math.isfinite(size) or size <= 0:
            raise ValueError(f"invalid resident size for eviction candidate {item!r}: {size!r}")
        selected.append(item); sizes.append(size); freed += size
    sufficient = freed + CACHE_CAPACITY_EPSILON >= required
    selection_reason = "no_eviction_required" if required <= CACHE_CAPACITY_EPSILON else reason if sufficient else "insufficient_evictable_capacity"
    return EvictionPlan(str(kwargs["rsu_id"]), selected, sizes, freed, required, unit, policy.policy_name, policy.policy_version, candidates, [evidence(rank, item) for rank, item in enumerate(candidates)], sufficient, selection_reason)


def build_eviction_policy(name: str, *, seed: int | None = None, **policy_config: Any) -> EvictionPolicy:
    """Construct a registered policy; aliases and silent fallback are forbidden."""

    normalized = str(name or "").strip().lower()
    factories = {"lru": LRUEvictionPolicy, "fifo": FIFOEvictionPolicy, "lfu": LFUEvictionPolicy, "aging_lfu": AgingLFUEvictionPolicy, "random": RandomEvictionPolicy}
    if normalized not in factories:
        raise ValueError(f"unsupported eviction policy: {name!r}; registered policies: {', '.join(factories)}")
    if normalized != "aging_lfu" and policy_config:
        raise ValueError(f"eviction policy {normalized} does not accept config keys: {sorted(policy_config)}")
    return factories[normalized](seed=seed, **policy_config)
