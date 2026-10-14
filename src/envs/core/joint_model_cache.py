"""Byte-accounted model cache for the opt-in joint-control diagnostic contract.

This is a controlled frame simulator, not a measured transfer implementation.
Admission is a policy action; dependency-safe LRU replacement is automatic.
Resident and in-flight objects both reserve capacity. Model weights are shared
by compatible adapters; workflow state is accounted separately by the caller.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import ceil
from typing import Any, Iterable, Mapping

from src.envs.specs.semantic_objects import WorkflowGraphState


@dataclass(frozen=True)
class ModelCacheObject:
    object_id: str
    object_type: str
    resident_size_bytes: int
    transfer_size_bytes: int
    dependency_ids: tuple[str, ...] = ()
    base_model_id: str | None = None
    adapter_id: str | None = None
    pinned: bool = False

    def __post_init__(self) -> None:
        if self.object_type not in {"base_model", "adapter"}:
            raise ValueError("joint model cache supports base_model and adapter objects")
        if min(self.resident_size_bytes, self.transfer_size_bytes) <= 0:
            raise ValueError("model sizes must be positive bytes")
        if self.object_type == "base_model" and (not self.base_model_id or self.dependency_ids):
            raise ValueError("base models require an identity and no dependencies")
        if self.object_type == "adapter" and (not self.adapter_id or len(self.dependency_ids) != 1):
            raise ValueError("adapters require an identity and exactly one base dependency")


def controlled_catalog_from_workflows(
    workflows: Mapping[str, WorkflowGraphState] | Iterable[WorkflowGraphState],
    *,
    base_size_bytes: int,
    adapter_size_bytes: int,
) -> list[ModelCacheObject]:
    """Build an explicitly synthetic catalog; these sizes are not measurements."""
    objects: dict[str, ModelCacheObject] = {}
    adapter_bases: dict[str, str] = {}
    rows = workflows.values() if isinstance(workflows, Mapping) else workflows
    for workflow in rows:
        for node in workflow.nodes:
            base, adapter = node.required_base_model, node.required_adapter
            if adapter in adapter_bases and adapter_bases[adapter] != base:
                raise ValueError(f"adapter {adapter} has conflicting base dependencies")
            adapter_bases[adapter] = base
            base_id = f"base:{base}"
            objects[base_id] = ModelCacheObject(
                base_id, "base_model", base_size_bytes, base_size_bytes,
                base_model_id=base,
            )
            adapter_id = f"adapter:{adapter}"
            objects[adapter_id] = ModelCacheObject(
                adapter_id, "adapter", adapter_size_bytes, adapter_size_bytes,
                dependency_ids=(base_id,), adapter_id=adapter,
            )
    return list(objects.values())


class JointModelCache:
    """Per-RSU dependencies, capacity reservations, and shared-link transfers.

    Frame order: ``begin_frame()``, zero or more ``request_adapter()`` calls,
    then ``advance()``. Pass the caller's remaining shared-link slots to each
    request and subtract its ``link_slots_consumed`` before handling state
    transfers. Pending dependency waits reserve memory but use no link slot.
    """

    contract_version = "joint_typed_model_cache_v1"
    eviction_policy = "automatic_dependency_safe_lru"

    def __init__(
        self,
        rsu_ids: Iterable[str],
        *,
        capacity_bytes: int | Mapping[str, int],
        objects: Iterable[ModelCacheObject],
        link_slots: int,
        transfer_bytes_per_frame: int,
    ) -> None:
        ids = list(rsu_ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("RSU IDs must be nonempty and unique")
        if link_slots < 1 or transfer_bytes_per_frame < 1:
            raise ValueError("link slots and transfer bytes per frame must be positive")
        self.capacity_bytes = {
            rsu: int(capacity_bytes[rsu] if isinstance(capacity_bytes, Mapping) else capacity_bytes)
            for rsu in ids
        }
        if min(self.capacity_bytes.values()) < 1:
            raise ValueError("cache capacity must be positive bytes")
        rows = list(objects)
        self.objects = {obj.object_id: obj for obj in rows}
        if len(self.objects) != len(rows):
            raise ValueError("duplicate model object identity")
        self.adapters = {obj.adapter_id: obj.object_id for obj in rows if obj.object_type == "adapter"}
        if len(self.adapters) != sum(obj.object_type == "adapter" for obj in rows):
            raise ValueError("duplicate adapter identity")
        for obj in rows:
            if any(dep not in self.objects or self.objects[dep].object_type != "base_model"
                   for dep in obj.dependency_ids):
                raise ValueError(f"missing or invalid base dependency for {obj.object_id}")
        self.link_slots = int(link_slots)
        self.transfer_bytes_per_frame = int(transfer_bytes_per_frame)
        self.resident: dict[str, OrderedDict[str, None]] = {rsu: OrderedDict() for rsu in ids}
        self.pending: dict[str, OrderedDict[str, int]] = {rsu: OrderedDict() for rsu in ids}
        self.active: dict[str, set[str]] = {rsu: set() for rsu in ids}
        self.transferred_bytes = {rsu: 0 for rsu in ids}
        self.evicted_bytes = {rsu: 0 for rsu in ids}
        self.eviction_count = {rsu: 0 for rsu in ids}
        self.last_frame_transfer_bytes = {rsu: 0 for rsu in ids}
        self._frame_open = False

    def _adapter_object(self, adapter_id: str, required_base_model_id: str) -> ModelCacheObject | None:
        object_id = self.adapters.get(adapter_id)
        if object_id is None:
            return None
        obj = self.objects[object_id]
        if self.objects[obj.dependency_ids[0]].base_model_id != required_base_model_id:
            return None
        return obj

    def is_ready(self, rsu_id: str, adapter_id: str, required_base_model_id: str) -> bool:
        obj = self._adapter_object(adapter_id, required_base_model_id)
        return bool(obj is not None and obj.object_id in self.resident[rsu_id]
                    and all(dep in self.resident[rsu_id] for dep in obj.dependency_ids))

    def touch(self, rsu_id: str, adapter_id: str, required_base_model_id: str) -> None:
        if not self.is_ready(rsu_id, adapter_id, required_base_model_id):
            raise ValueError("cannot serve a model before adapter and base are ready")
        obj = self.objects[self.adapters[adapter_id]]
        for object_id in (*obj.dependency_ids, obj.object_id):
            self.resident[rsu_id].move_to_end(object_id)

    def begin_frame(self) -> dict[str, int]:
        if self._frame_open:
            raise RuntimeError("advance must close the previous cache frame")
        self._frame_open = True
        for rsu, pending in self.pending.items():
            self.last_frame_transfer_bytes[rsu] = 0
            for object_id in pending:
                if len(self.active[rsu]) >= self.link_slots:
                    break
                if all(dep in self.resident[rsu] for dep in self.objects[object_id].dependency_ids):
                    self.active[rsu].add(object_id)
        return {rsu: len(active) for rsu, active in self.active.items()}

    def _used_bytes(self, rsu_id: str) -> int:
        return sum(self.objects[object_id].resident_size_bytes
                   for object_id in (*self.resident[rsu_id], *self.pending[rsu_id]))

    def _eviction_plan(self, rsu_id: str, needed_bytes: int, protected: set[str]) -> list[str] | None:
        """Plan without mutation; never leave a resident or pending orphan."""
        remaining = list(self.resident[rsu_id])
        used = self._used_bytes(rsu_id)
        victims: list[str] = []
        while used + needed_bytes > self.capacity_bytes[rsu_id]:
            dependents = {
                dep for key in (*remaining, *self.pending[rsu_id])
                for dep in self.objects[key].dependency_ids
            }
            victim = next((key for key in remaining
                           if key not in protected and key not in dependents
                           and not self.objects[key].pinned), None)
            if victim is None:
                return None
            remaining.remove(victim)
            victims.append(victim)
            used -= self.objects[victim].resident_size_bytes
        return victims

    def request_adapter(
        self,
        rsu_id: str,
        adapter_id: str,
        required_base_model_id: str,
        available_link_slots: int | None = None,
    ) -> dict[str, Any]:
        if not self._frame_open:
            raise RuntimeError("begin_frame must precede admission")
        result: dict[str, Any] = {"status": "unknown_or_incompatible_model", "link_slots_consumed": 0,
                                  "admitted_object_ids": [], "evicted_object_ids": []}
        obj = self._adapter_object(adapter_id, required_base_model_id)
        if obj is None:
            return result
        if self.is_ready(rsu_id, adapter_id, required_base_model_id):
            self.touch(rsu_id, adapter_id, required_base_model_id)
            return {**result, "status": "already_cached"}
        if obj.object_id in self.pending[rsu_id]:
            return {**result, "status": "already_loading"}
        closure = (*obj.dependency_ids, obj.object_id)
        return self._request_closure(rsu_id, closure, result, available_link_slots)

    def request_base_model(self, rsu_id: str, base_model_id: str,
                           available_link_slots: int | None = None) -> dict[str, Any]:
        """Explicit base-only admission for the parameterized engineering contract."""
        if not self._frame_open:
            raise RuntimeError("begin_frame must precede admission")
        result = {"status": "unknown_or_incompatible_model", "link_slots_consumed": 0,
                  "admitted_object_ids": [], "evicted_object_ids": []}
        obj = next((obj for obj in self.objects.values()
                    if obj.object_type == "base_model" and obj.base_model_id == base_model_id), None)
        if obj is None:
            return result
        if obj.object_id in self.resident[rsu_id]:
            self.resident[rsu_id].move_to_end(obj.object_id)
            return {**result, "status": "already_cached"}
        if obj.object_id in self.pending[rsu_id]:
            return {**result, "status": "already_loading"}
        return self._request_closure(rsu_id, (obj.object_id,), result, available_link_slots,
                                     fetch_status="base_model_fetch")

    def validate_eviction(self, rsu_id: str, object_ids: Iterable[str]) -> list[str]:
        """Validate the complete ordered eviction before any cache mutation."""
        victims = list(object_ids)
        if len(victims) != len(set(victims)):
            raise ValueError("duplicate eviction object")
        remaining = set(self.resident[rsu_id])
        for key in victims:
            if key not in remaining or self.objects[key].pinned:
                raise ValueError("eviction requires an unpinned resident object")
            dependents = {dep for item in (*remaining, *self.pending[rsu_id])
                          for dep in self.objects[item].dependency_ids}
            if key in dependents:
                raise ValueError("eviction would orphan a resident or pending adapter")
            remaining.remove(key)
        return victims

    def evict_objects(self, rsu_id: str, object_ids: Iterable[str]) -> dict[str, Any]:
        if not self._frame_open:
            raise RuntimeError("begin_frame must precede eviction")
        victims = self.validate_eviction(rsu_id, object_ids)
        size = sum(self.objects[key].resident_size_bytes for key in victims)
        for key in victims:
            self.resident[rsu_id].pop(key)
        self.evicted_bytes[rsu_id] += size
        self.eviction_count[rsu_id] += len(victims)
        return {"evicted_object_ids": victims, "evicted_bytes": size}

    def _request_closure(self, rsu_id: str, closure: tuple[str, ...], result: dict[str, Any],
                         available_link_slots: int | None, *,
                         fetch_status: str = "adapter_fetch") -> dict[str, Any]:
        resident, pending = self.resident[rsu_id], self.pending[rsu_id]
        missing = [key for key in closure if key not in resident and key not in pending]
        closure_bytes = sum(self.objects[key].resident_size_bytes for key in closure)
        if closure_bytes > self.capacity_bytes[rsu_id]:
            return {**result, "status": "capacity_infeasible", "required_bytes": closure_bytes}
        needed = sum(self.objects[key].resident_size_bytes for key in missing)
        victims = self._eviction_plan(rsu_id, needed, set(closure))
        if victims is None:
            return {**result, "status": "capacity_blocked", "required_free_bytes": needed}
        ready_to_start = [key for key in missing
                          if all(dep in resident for dep in self.objects[key].dependency_ids)]
        available = self.link_slots - len(self.active[rsu_id])
        if available_link_slots is not None:
            available = min(available, max(0, available_link_slots))
        if ready_to_start and available < 1:
            return {**result, "status": "link_queue"}
        for key in victims:
            resident.pop(key)
            self.evicted_bytes[rsu_id] += self.objects[key].resident_size_bytes
            self.eviction_count[rsu_id] += 1
        for key in missing:
            pending[key] = self.objects[key].transfer_size_bytes
        started = ready_to_start[:available]
        self.active[rsu_id].update(started)
        return {**result, "status": fetch_status, "link_slots_consumed": len(started),
                "admitted_object_ids": missing, "evicted_object_ids": victims,
                "reserved_bytes": needed}

    def advance(self) -> list[dict[str, Any]]:
        if not self._frame_open:
            raise RuntimeError("begin_frame must precede advance")
        events = []
        for rsu, active in self.active.items():
            for object_id in sorted(active):
                remaining = self.pending[rsu][object_id]
                transferred = min(self.transfer_bytes_per_frame, remaining)
                self.pending[rsu][object_id] -= transferred
                self.transferred_bytes[rsu] += transferred
                self.last_frame_transfer_bytes[rsu] += transferred
                finished = self.pending[rsu][object_id] == 0
                events.append({"rsu_id": rsu, "object_id": object_id,
                               "object_type": self.objects[object_id].object_type,
                               "transferred_bytes": transferred, "completed": finished})
                if finished:
                    self.pending[rsu].pop(object_id)
                    self.resident[rsu][object_id] = None
            active.intersection_update(self.pending[rsu])
        self._frame_open = False
        return events

    def snapshot(self, rsu_id: str) -> dict[str, Any]:
        resident, pending = self.resident[rsu_id], self.pending[rsu_id]
        cached = [self.objects[key].adapter_id for key in resident
                  if self.objects[key].object_type == "adapter"
                  and all(dep in resident for dep in self.objects[key].dependency_ids)]
        loading = [self.objects[key].adapter_id for key in pending
                   if self.objects[key].object_type == "adapter"]
        remaining_frames = {key: ceil(remaining / self.transfer_bytes_per_frame)
                            for key, remaining in pending.items()}
        resident_bytes = sum(self.objects[key].resident_size_bytes for key in resident)
        reserved_bytes = sum(self.objects[key].resident_size_bytes for key in pending)
        return {
            "contract_version": self.contract_version,
            "eviction_policy": self.eviction_policy,
            "cached_adapter_ids": cached,
            "loading_adapter_ids": loading,
            "ready_base_model_ids": [self.objects[key].base_model_id for key in resident
                                     if self.objects[key].object_type == "base_model"],
            "resident_object_ids": list(resident),
            "loading_object_ids": list(pending),
            "active_transfer_object_ids": sorted(self.active[rsu_id]),
            "loading_remaining_frames": remaining_frames,
            "loading_remaining_bytes": dict(pending),
            "dependency_ready": {key: all(dep in resident for dep in self.objects[key].dependency_ids)
                                 for key in (*resident, *pending)},
            "capacity_bytes": self.capacity_bytes[rsu_id],
            "resident_bytes": resident_bytes,
            "reserved_bytes": reserved_bytes,
            "occupied_bytes": resident_bytes + reserved_bytes,
            "free_bytes": self.capacity_bytes[rsu_id] - resident_bytes - reserved_bytes,
            "transferred_bytes": self.transferred_bytes[rsu_id],
            "last_frame_transfer_bytes": self.last_frame_transfer_bytes[rsu_id],
            "evicted_bytes": self.evicted_bytes[rsu_id],
            "eviction_count": self.eviction_count[rsu_id],
        }
