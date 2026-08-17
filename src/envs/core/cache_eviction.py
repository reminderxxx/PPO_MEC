"""Stable, auditable cache-eviction policy contract and LRU implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import math
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


def build_eviction_policy(name: str, *, seed: int | None = None) -> EvictionPolicy:
    """Construct a registered policy; aliases and silent fallback are forbidden."""

    normalized = str(name or "").strip().lower()
    if normalized == "lru":
        return LRUEvictionPolicy(seed=seed)
    raise ValueError(f"unsupported eviction policy: {name!r}; registered policies: lru")
