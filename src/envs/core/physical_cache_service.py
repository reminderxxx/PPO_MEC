"""Asynchronous typed-cache admission backed by the repository catalog.

Catalog sizes are MiB. This service converts them to bytes once, reserves
resident capacity before loading, and delegates transfer time and consumed
bytes to PhysicalRuntime through caller-provided callbacks.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Callable, Mapping, Sequence

from src.data.model_catalog.adapter_catalog import AdapterCatalog, TypedCacheObject


MIB = 1024 * 1024


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


class PhysicalTypedCache:
    """Per-RSU typed object residency with dependency-safe LRU admission.

    ``request`` returns ``ready``, ``loading``, or ``rejected``. A loading
    object's space is reserved once and becomes resident without another
    capacity increment. Every missing typed object receives its own transfer
    job, allowing completion times to differ. A completed adapter waits for
    its base dependency before becoming resident. ``cancel`` also cancels
    in-flight dependents when their required object is cancelled.
    """

    def __init__(
        self,
        catalog: AdapterCatalog,
        rsu_ids: Sequence[str],
        capacity_bytes_per_rsu: Mapping[str, float] | float,
        initial_residents: Mapping[str, Sequence[str]] | None = None,
        eviction_policy: str = "lru",
    ) -> None:
        if eviction_policy != "lru":
            raise ValueError("physical typed-cache currently supports eviction_policy='lru'")
        ids = list(rsu_ids)
        if not ids or any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
            raise ValueError("rsu_ids must contain unique nonempty strings")
        self.catalog = catalog
        self.rsu_ids = tuple(ids)
        if isinstance(capacity_bytes_per_rsu, Mapping):
            if set(capacity_bytes_per_rsu) != set(ids):
                raise ValueError("capacity_bytes_per_rsu must contain exactly the RSU IDs")
            self.capacity_bytes_per_rsu = {
                rsu_id: _nonnegative(capacity_bytes_per_rsu[rsu_id], f"capacity for {rsu_id}")
                for rsu_id in ids
            }
        else:
            capacity = _nonnegative(capacity_bytes_per_rsu, "capacity_bytes_per_rsu")
            self.capacity_bytes_per_rsu = {rsu_id: capacity for rsu_id in ids}
        if initial_residents is not None and set(initial_residents) - set(ids):
            raise ValueError("initial_residents contains an unknown RSU")

        self._objects: dict[str, TypedCacheObject] = {
            item.object_id: item for item in catalog.typed_cache_objects
        }
        if len(self._objects) != len(catalog.typed_cache_objects):
            raise ValueError("typed catalog contains duplicate object IDs")
        self._state: dict[str, dict[str, str]] = {}
        self._job_for_object: dict[tuple[str, str], str] = {}
        self._object_for_job: dict[str, tuple[str, str]] = {}
        self._pins: dict[tuple[str, str], int] = {}
        self._last_used: dict[tuple[str, str], int] = {}
        self._touch_sequence = 0
        for rsu_id in ids:
            residents = list(
                catalog.get_initial_typed_residents(rsu_id)
                if initial_residents is None
                else initial_residents.get(rsu_id, ())
            )
            if len(set(residents)) != len(residents):
                raise ValueError(f"duplicate initial resident at {rsu_id}")
            for object_id in residents:
                self._object(object_id)
            resident_set = set(residents)
            for object_id in residents:
                if not set(self._object(object_id).dependency_ids).issubset(resident_set):
                    raise ValueError(f"initial resident has missing dependency: {object_id}")
            self._state[rsu_id] = {object_id: "resident" for object_id in residents}
            if self._used_bytes(rsu_id) > self.capacity_bytes_per_rsu[rsu_id]:
                raise ValueError(f"initial residents exceed capacity at {rsu_id}")
            for object_id in residents:
                self._touch(rsu_id, object_id)

    def _object(self, object_id: str) -> TypedCacheObject:
        try:
            return self._objects[object_id]
        except KeyError as exc:
            raise ValueError(f"unknown typed object: {object_id}") from exc

    def _resident_bytes(self, object_id: str) -> float:
        item = self._object(object_id)
        return float(item.resident_size_mb) * MIB if item.counts_toward_capacity else 0.0

    def _transfer_bytes(self, object_id: str) -> float:
        return float(self._object(object_id).transfer_size_mb) * MIB

    def _used_bytes(self, rsu_id: str) -> float:
        return sum(self._resident_bytes(item) for item in self._state.get(rsu_id, {}))

    def _touch(self, rsu_id: str, object_id: str) -> None:
        self._touch_sequence += 1
        self._last_used[(rsu_id, object_id)] = self._touch_sequence

    def _validate_rsu(self, rsu_id: str) -> None:
        if rsu_id not in self._state:
            raise KeyError(f"unknown rsu_id: {rsu_id}")

    def _closure(self, object_id: str) -> list[str]:
        ordered: list[str] = []
        visited: set[str] = set()

        def visit(current: str) -> None:
            if current in visited:
                return
            visited.add(current)
            for dependency in self._object(current).dependency_ids:
                visit(dependency)
            ordered.append(current)

        visit(object_id)
        return ordered

    def _victims(self, rsu_id: str, needed_bytes: float, protected: set[str]) -> list[str] | None:
        remaining = set(self._state[rsu_id])
        victims: list[str] = []
        freed = 0.0
        while freed < needed_bytes:
            candidates = []
            for object_id in remaining:
                item = self._object(object_id)
                if (
                    object_id in protected
                    or self._state[rsu_id][object_id] != "resident"
                    or item.evictability != "evictable"
                    or self._pins.get((rsu_id, object_id), 0)
                ):
                    continue
                if any(
                    object_id in self._object(other).dependency_ids
                    for other in remaining if other != object_id
                ):
                    continue
                candidates.append(object_id)
            if not candidates:
                return None
            victim = min(candidates, key=lambda item: (self._last_used.get((rsu_id, item), 0), item))
            remaining.remove(victim)
            victims.append(victim)
            freed += self._resident_bytes(victim)
        return victims

    def is_ready(self, rsu_id: str, adapter_id: str) -> bool:
        self._validate_rsu(rsu_id)
        adapter = self.catalog.get_typed_adapter(adapter_id)
        return all(self._state[rsu_id].get(item) == "resident" for item in self._closure(adapter.object_id))

    def request(
        self,
        rsu_id: str,
        adapter_id: str,
        time_s: float,
        submit_transfer: Callable[[float, dict[str, Any]], str],
    ) -> dict[str, Any]:
        self._validate_rsu(rsu_id)
        _nonnegative(time_s, "time_s")
        adapter = self.catalog.get_typed_adapter(adapter_id)
        # The native plan fixes base-before-adapter object order and identity.
        plan = self.catalog.resolve_typed_placement_plan(
            adapter_id=adapter_id, resident_object_ids=list(self._state[rsu_id])
        )
        closure = self._closure(adapter.object_id)
        if self.is_ready(rsu_id, adapter_id):
            for object_id in closure:
                self._touch(rsu_id, object_id)
            return {
                "status": "ready", "rsu_id": rsu_id, "adapter_id": adapter_id,
                "job_ids": [], "missing_object_ids": [], "evicted_object_ids": [],
                "ordered_object_ids": list(plan.ordered_object_ids),
            }

        missing = list(plan.missing_object_ids)
        need = sum(self._resident_bytes(item) for item in missing)
        deficit = max(0.0, self._used_bytes(rsu_id) + need - self.capacity_bytes_per_rsu[rsu_id])
        victims = self._victims(rsu_id, deficit, set(closure)) if deficit > 0 else []
        if victims is None:
            return {
                "status": "rejected", "reason": "insufficient_dependency_safe_capacity",
                "rsu_id": rsu_id, "adapter_id": adapter_id,
                "job_ids": [], "missing_object_ids": missing, "evicted_object_ids": [],
                "ordered_object_ids": list(plan.ordered_object_ids),
            }
        for object_id in victims:
            del self._state[rsu_id][object_id]
            self._last_used.pop((rsu_id, object_id), None)
        job_ids: list[str] = []
        for object_id in missing:
            item = self._object(object_id)
            self._state[rsu_id][object_id] = "loading"
            metadata = {
                "rsu_id": rsu_id, "object_id": object_id,
                "object_type": item.object_type, "adapter_id": adapter_id,
            }
            try:
                job_id = submit_transfer(self._transfer_bytes(object_id), metadata)
            except Exception:
                # Keep submitted jobs and their reservations: their transfer
                # cost has already started and must be completed or cancelled.
                del self._state[rsu_id][object_id]
                raise
            if not isinstance(job_id, str) or not job_id or job_id in self._object_for_job:
                del self._state[rsu_id][object_id]
                raise ValueError("submit_transfer must return a unique nonempty job_id")
            self._job_for_object[(rsu_id, object_id)] = job_id
            self._object_for_job[job_id] = (rsu_id, object_id)
            job_ids.append(job_id)
            self._touch(rsu_id, object_id)
        return {
            "status": "loading", "rsu_id": rsu_id, "adapter_id": adapter_id,
            "job_ids": job_ids, "missing_object_ids": missing,
            "evicted_object_ids": victims,
            "ordered_object_ids": list(plan.ordered_object_ids),
        }

    def _promote(self, rsu_id: str) -> list[str]:
        promoted: list[str] = []
        changed = True
        while changed:
            changed = False
            for object_id, state in list(self._state[rsu_id].items()):
                if state != "loaded_waiting_dependency":
                    continue
                if all(self._state[rsu_id].get(dep) == "resident" for dep in self._object(object_id).dependency_ids):
                    self._state[rsu_id][object_id] = "resident"
                    promoted.append(object_id)
                    changed = True
        return promoted

    def on_transfer_complete(self, job_id: str, time_s: float) -> dict[str, Any]:
        _nonnegative(time_s, "time_s")
        try:
            rsu_id, object_id = self._object_for_job.pop(job_id)
        except KeyError as exc:
            raise KeyError(f"unknown or completed cache transfer: {job_id}") from exc
        if self._state[rsu_id].get(object_id) != "loading":
            raise RuntimeError(f"cache job has no loading reservation: {job_id}")
        del self._job_for_object[(rsu_id, object_id)]
        self._state[rsu_id][object_id] = "loaded_waiting_dependency"
        promoted = self._promote(rsu_id)
        return {
            "status": self._state[rsu_id][object_id], "job_id": job_id,
            "rsu_id": rsu_id, "object_id": object_id, "time_s": float(time_s),
            "became_resident_object_ids": promoted,
            "used_bytes": self._used_bytes(rsu_id),
        }

    def cancel(
        self, job_id: str, cancel_transfer: Callable[[str], dict[str, Any]]
    ) -> dict[str, Any]:
        try:
            rsu_id, object_id = self._object_for_job[job_id]
        except KeyError as exc:
            raise KeyError(f"unknown or completed cache transfer: {job_id}") from exc
        affected = {object_id}
        changed = True
        while changed:
            changed = False
            for other in self._state[rsu_id]:
                if other not in affected and any(dep in affected for dep in self._object(other).dependency_ids):
                    affected.add(other)
                    changed = True
        cancelled_events = []
        released = []
        for current in sorted(affected, key=lambda item: (item != object_id, item)):
            current_job_id = self._job_for_object.get((rsu_id, current))
            if current_job_id is not None:
                event = cancel_transfer(current_job_id)
                cancelled_events.append(deepcopy(event))
                del self._object_for_job[current_job_id]
                del self._job_for_object[(rsu_id, current)]
            if current in self._state[rsu_id]:
                del self._state[rsu_id][current]
                released.append(current)
                self._last_used.pop((rsu_id, current), None)
        return {
            "status": "cancelled", "rsu_id": rsu_id, "object_id": object_id,
            "released_object_ids": released, "cancelled_events": cancelled_events,
            "used_bytes": self._used_bytes(rsu_id),
        }

    def pin(self, rsu_id: str, adapter_id: str) -> None:
        if not self.is_ready(rsu_id, adapter_id):
            raise ValueError("adapter and dependencies must be resident before pin")
        adapter = self.catalog.get_typed_adapter(adapter_id)
        for object_id in self._closure(adapter.object_id):
            key = (rsu_id, object_id)
            self._pins[key] = self._pins.get(key, 0) + 1
            self._touch(rsu_id, object_id)

    def unpin(self, rsu_id: str, adapter_id: str) -> None:
        self._validate_rsu(rsu_id)
        adapter = self.catalog.get_typed_adapter(adapter_id)
        closure = self._closure(adapter.object_id)
        if any(self._pins.get((rsu_id, item), 0) <= 0 for item in closure):
            raise ValueError("adapter dependency closure is not pinned")
        for object_id in closure:
            key = (rsu_id, object_id)
            self._pins[key] -= 1
            if self._pins[key] == 0:
                del self._pins[key]

    def snapshot(self) -> dict[str, Any]:
        return {
            "capacity_bytes_per_rsu": dict(self.capacity_bytes_per_rsu),
            "rsus": {
                rsu_id: {
                    "used_bytes": self._used_bytes(rsu_id),
                    "remaining_bytes": self.capacity_bytes_per_rsu[rsu_id] - self._used_bytes(rsu_id),
                    "objects": [
                        {
                            "object_id": object_id, "state": state,
                            "resident_bytes": self._resident_bytes(object_id),
                            "transfer_bytes": self._transfer_bytes(object_id),
                            "job_id": self._job_for_object.get((rsu_id, object_id)),
                            "pin_count": self._pins.get((rsu_id, object_id), 0),
                        }
                        for object_id, state in self._state[rsu_id].items()
                    ],
                }
                for rsu_id in self.rsu_ids
            },
        }
