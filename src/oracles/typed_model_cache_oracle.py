"""Tiny exact oracle for atomic typed base-model + adapter dependency bundles."""

from __future__ import annotations

import itertools
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from src.evaluators.cache_baseline_fairness import sha256_value


EPSILON = 1.0e-9


class TypedOracleError(ValueError):
    pass


class _StateLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class _TypedAction:
    action: str
    rsu_id: str | None
    admitted_object_ids: tuple[str, ...] = ()
    evicted_object_ids: tuple[str, ...] = ()
    rejection_reason: str | None = None

    @property
    def tie_key(self) -> tuple[Any, ...]:
        return (
            self.rsu_id or "",
            self.admitted_object_ids,
            self.action,
            self.evicted_object_ids,
        )


def _state_key(state: dict[str, set[str]]) -> tuple[Any, ...]:
    return tuple((rsu, tuple(sorted(objects))) for rsu, objects in sorted(state.items()))


def _state_fingerprint(state: dict[str, set[str]]) -> str:
    return sha256_value({rsu: sorted(objects) for rsu, objects in sorted(state.items())})


def _type_sums(object_ids: tuple[str, ...] | list[str], objects: dict[str, dict[str, Any]], field: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for object_id in object_ids:
        row = objects[object_id]
        object_type = str(row["object_type"])
        totals[object_type] = totals.get(object_type, 0.0) + float(row[field])
    return {key: round(value, 6) for key, value in sorted(totals.items())}


def _used(residents: set[str], objects: dict[str, dict[str, Any]]) -> float:
    return sum(float(objects[item]["resident_size_mb"]) for item in residents)


def _dependencies_valid(residents: set[str], objects: dict[str, dict[str, Any]]) -> bool:
    return all(set(objects[item].get("dependency_ids") or []).issubset(residents) for item in residents)


def _candidate_actions(
    request: dict[str, Any],
    state: dict[str, set[str]],
    objects: dict[str, dict[str, Any]],
    capacity_mb: float,
) -> list[_TypedAction]:
    bundle = tuple(request["dependency_bundle"]["ordered_object_ids"])
    actions = [_TypedAction("noop", None)]
    for raw_rsu_id in request["eligible_cache_target_rsu_ids"]:
        rsu_id = str(raw_rsu_id)
        residents = state.setdefault(rsu_id, set())
        missing = tuple(item for item in bundle if item not in residents)
        if not missing:
            actions.append(_TypedAction("already_resident", rsu_id))
            continue
        requested_mb = sum(float(objects[item]["resident_size_mb"]) for item in missing)
        if any(float(objects[item]["resident_size_mb"]) > capacity_mb + EPSILON for item in missing):
            actions.append(_TypedAction("reject_oversized_object", rsu_id, rejection_reason="object_exceeds_total_capacity"))
            continue
        if requested_mb > capacity_mb + EPSILON:
            actions.append(_TypedAction("reject_oversized_bundle", rsu_id, rejection_reason="dependency_bundle_exceeds_total_capacity"))
            continue
        required = max(_used(residents, objects) + requested_mb - capacity_mb, 0.0)
        if required <= EPSILON:
            actions.append(_TypedAction("admit_bundle", rsu_id, missing))
            continue
        evictable = sorted(
            item
            for item in residents
            if objects[item].get("evictability") == "evictable"
        )
        for count in range(1, len(evictable) + 1):
            for victims in itertools.combinations(evictable, count):
                remaining = (residents - set(victims)) | set(missing)
                if not _dependencies_valid(remaining, objects):
                    continue
                freed = sum(float(objects[item]["resident_size_mb"]) for item in victims)
                if freed + EPSILON < required:
                    continue
                if any(
                    freed - float(objects[item]["resident_size_mb"]) + EPSILON >= required
                    and _dependencies_valid((residents - (set(victims) - {item})) | set(missing), objects)
                    for item in victims
                ):
                    continue
                actions.append(_TypedAction("admit_bundle", rsu_id, missing, victims))
    return sorted(set(actions), key=lambda item: item.tie_key)


def _apply(
    action: _TypedAction,
    state: dict[str, set[str]],
    objects: dict[str, dict[str, Any]],
    capacity_mb: float,
) -> tuple[dict[str, set[str]], float, float]:
    result = {rsu: set(residents) for rsu, residents in state.items()}
    transfer = evicted = 0.0
    if action.action == "admit_bundle" and action.rsu_id:
        residents = result.setdefault(action.rsu_id, set())
        for object_id in action.evicted_object_ids:
            residents.remove(object_id)
            evicted += float(objects[object_id]["resident_size_mb"])
        for object_id in action.admitted_object_ids:
            residents.add(object_id)
            transfer += float(objects[object_id]["transfer_size_mb"])
        if not _dependencies_valid(residents, objects):
            raise TypedOracleError("oracle action created an orphan")
    for rsu_id, residents in result.items():
        if _used(residents, objects) > capacity_mb + EPSILON:
            raise TypedOracleError(f"typed oracle capacity violation at {rsu_id}")
    return result, transfer, evicted


def _hit(request: dict[str, Any], state: dict[str, set[str]]) -> bool:
    bundle = set(request["dependency_bundle"]["ordered_object_ids"])
    return any(bundle.issubset(state.get(str(rsu_id), set())) for rsu_id in request["eligible_service_rsu_ids"])


def _better(left: tuple[Any, ...], right: tuple[Any, ...] | None) -> bool:
    if right is None:
        return True
    left_primary = (round(left[0], 12), left[1], round(-left[2], 12), round(-left[3], 12))
    right_primary = (round(right[0], 12), right[1], round(-right[2], 12), round(-right[3], 12))
    return left_primary > right_primary if left_primary != right_primary else left[4] < right[4]


def _plan(
    requests: list[dict[str, Any]],
    state: dict[str, set[str]],
    objects: dict[str, dict[str, Any]],
    capacity_mb: float,
    state_limit: int,
) -> tuple[list[_TypedAction], tuple[Any, ...], int]:
    visited = 0
    memo: dict[tuple[Any, ...], tuple[list[_TypedAction], tuple[Any, ...]]] = {}

    def recurse(index: int, current: dict[str, set[str]]) -> tuple[list[_TypedAction], tuple[Any, ...]]:
        nonlocal visited
        key = (index, _state_key(current))
        if key in memo:
            return memo[key]
        visited += 1
        if visited > state_limit:
            raise _StateLimit
        if index >= len(requests):
            return [], (0.0, 0, 0.0, 0.0, ())
        request = requests[index]
        best_actions = None
        best_score = None
        for action in _candidate_actions(request, current, objects, capacity_mb):
            next_state, transfer, evicted = _apply(action, current, objects, capacity_mb)
            hit = _hit(request, next_state)
            future_actions, future = recurse(index + 1, next_state)
            score = (
                (float(request["object_size_mb"]) if hit else 0.0) + future[0],
                int(hit) + future[1],
                transfer + future[2],
                evicted + future[3],
                (action.tie_key,) + future[4],
            )
            if _better(score, best_score):
                best_actions, best_score = [action, *future_actions], score
        assert best_actions is not None and best_score is not None
        memo[key] = (best_actions, best_score)
        return memo[key]

    actions, score = recurse(0, state)
    return actions, score, visited


def solve_typed_model_cache_oracle(
    *, replay: dict[str, Any], manifest: dict[str, Any], horizon: int,
    state_limit: int, full_trace_diagnostic: bool,
    contract_version: str, solver_identity: str, objective_identity: str,
) -> dict[str, Any]:
    cache = manifest["cache_contract"]
    typed_contract = cache.get("typed_model_cache") or {}
    if typed_contract.get("profile_id") != "typed_base_adapter_state_v1":
        raise TypedOracleError("typed replay requires matching fairness typed profile")
    if typed_contract.get("contract_version") != "1.0.0":
        raise TypedOracleError("typed fairness contract version mismatch")
    if cache.get("capacity", {}).get("unit") != "mb":
        raise TypedOracleError("typed oracle requires MB capacity")
    capacity_mb = float(cache["capacity"]["capacity_mb"])
    objects = {
        str(item["object_id"]): deepcopy(item)
        for item in typed_contract.get("resident_objects", [])
    }
    if not objects:
        raise TypedOracleError("typed oracle resident object table is missing")
    state = {
        str(item["rsu_id"]): set(map(str, item["resident_object_ids"]))
        for item in typed_contract.get("initial_typed_cache_contents", [])
    }
    for residents in state.values():
        if not _dependencies_valid(residents, objects):
            raise TypedOracleError("typed oracle initial state contains orphan adapter")
        if _used(residents, objects) > capacity_mb + EPSILON:
            raise TypedOracleError("typed oracle initial state exceeds capacity")
    requests = list(replay["requests"])
    if any(request.get("catalog_fingerprint") != typed_contract.get("catalog_fingerprint") for request in requests):
        raise TypedOracleError("typed replay/catalog fingerprint mismatch")
    initial_fingerprint = _state_fingerprint(state)
    trace = []
    status = "optimal"
    visited_total = 0
    for index, request in enumerate(requests):
        end = len(requests) if full_trace_diagnostic else min(len(requests), index + horizon)
        visible = requests[index:end]
        pre_state = {rsu: set(residents) for rsu, residents in state.items()}
        try:
            actions, score, visited = _plan(visible, pre_state, objects, capacity_mb, state_limit)
        except _StateLimit:
            status = "unknown_state_limit"
            break
        visited_total += visited
        action = actions[0]
        pre_hit = _hit(request, pre_state)
        state, transfer, evicted = _apply(action, pre_state, objects, capacity_mb)
        post_hit = _hit(request, state)
        trace.append({
            "request_id": request["request_id"], "step_index": request["step_index"],
            "request_order": request["request_order"], "visible_request_ids": [item["request_id"] for item in visible],
            "visible_start_index": index, "visible_end_index_exclusive": end,
            "pre_action_hit": pre_hit, "post_action_hit": post_hit, "action": action.action,
            "cache_target_rsu_id": action.rsu_id,
            "admitted_object_id": action.admitted_object_ids[-1] if action.admitted_object_ids else None,
            "admitted_object_ids": list(action.admitted_object_ids),
            "evicted_object_ids": list(action.evicted_object_ids), "rejection_reason": action.rejection_reason,
            "adapter_transfer_mb": _type_sums(action.admitted_object_ids, objects, "transfer_size_mb").get("adapter", 0.0),
            "transfer_mb_by_type": _type_sums(action.admitted_object_ids, objects, "transfer_size_mb"),
            "evicted_mb_by_type": _type_sums(action.evicted_object_ids, objects, "resident_size_mb"),
            "transfer_mb": transfer, "evicted_mb": evicted, "cache_churn_mb": transfer + evicted,
            "rolling_plan_objective": {"hit_mb": score[0], "hit_count": score[1], "transfer_mb": score[2], "evicted_mb": score[3]},
            "post_state": {rsu: sorted(items) for rsu, items in sorted(state.items())},
            "post_state_fingerprint": _state_fingerprint(state),
            "capacity_occupancy": {rsu: {"used": _used(items, objects), "capacity": capacity_mb, "occupancy_rate": _used(items, objects) / capacity_mb} for rsu, items in sorted(state.items())},
            "atomic_dependency_bundle": True, "orphan_count": 0,
        })
    hit_count = sum(int(item["post_action_hit"]) for item in trace)
    requested_mb = sum(float(item["object_size_mb"]) for item in requests[:len(trace)])
    hit_mb = sum(float(requests[index]["object_size_mb"]) for index, item in enumerate(trace) if item["post_action_hit"])
    transfer_by_type: dict[str, float] = {}
    for item in trace:
        for object_type, value in item["transfer_mb_by_type"].items():
            transfer_by_type[object_type] = transfer_by_type.get(object_type, 0.0) + float(value)
    return {
        "identity": {
            "oracle_contract_version": contract_version,
            "oracle_identity": "full_trace_exact_diagnostic_v1.0.0" if full_trace_diagnostic else "rolling_finite_horizon_exact_v1.0.0",
            "solver_identity": solver_identity, "objective_identity": objective_identity,
            "objective_lexicographic": ["maximize_future_joint_model_hit_mb", "maximize_future_full_model_hit_count", "minimize_typed_transfer_mb", "minimize_evicted_mb_and_cache_churn", "canonical_tie_break"],
            "horizon": "full_trace" if full_trace_diagnostic else horizon,
            "horizon_includes_current_request": True, "rolling_horizon": not full_trace_diagnostic,
            "admission_timing": "action_before_service_lookup_same_step_atomic_bundle_can_hit",
            "optimality_status": status, "request_replay_fingerprint": replay["request_replay_fingerprint"],
            "g07_manifest_id": manifest["identity"]["manifest_id"],
            "g07_manifest_full_sha256": manifest["hashes"]["full_manifest_sha256"],
            "g07_manifest_semantic_sha256": manifest["hashes"]["semantic_protocol_sha256"],
            "git_commit": manifest["identity"]["git_commit"], "capacity_unit": "mb", "capacity_value": capacity_mb,
            "initial_state_fingerprint": initial_fingerprint, "state_limit": state_limit, "states_visited": visited_total,
            "model_cache_profile_id": "typed_base_adapter_state_v1", "typed_model_cache_contract_version": "1.0.0",
            "catalog_fingerprint": typed_contract["catalog_fingerprint"], "atomic_dependency_bundle": True,
        },
        "performance": {
            "request_count": len(trace), "total_replay_request_count": len(requests),
            "object_hit_count": hit_count, "object_hit_rate": hit_count / len(trace) if trace else None,
            "joint_model_hit_count": hit_count, "full_service_ready_count": hit_count,
            "requested_mb": requested_mb, "hit_mb": hit_mb, "byte_hit_rate": hit_mb / requested_mb if requested_mb else None,
            "admission_count": sum(bool(item["admitted_object_ids"]) for item in trace),
            "admission_mb": sum(item["transfer_mb"] for item in trace),
            "eviction_event_count": sum(bool(item["evicted_object_ids"]) for item in trace),
            "eviction_victim_count": sum(len(item["evicted_object_ids"]) for item in trace),
            "evicted_mb": sum(item["evicted_mb"] for item in trace), "transfer_mb": sum(item["transfer_mb"] for item in trace),
            "transfer_mb_by_type": {key: round(value, 6) for key, value in sorted(transfer_by_type.items())},
            "churn_mb": sum(item["cache_churn_mb"] for item in trace),
            "latency_saved": {"availability": "unavailable", "value": None, "reason": "counterfactual latency absent"},
        },
        "action_trace": trace,
        "capacity_invariant_audit": {"status": "pass" if status == "optimal" else "incomplete", "checked_step_count": len(trace), "capacity_never_exceeded": True, "dependency_orphan_count": 0, "per_rsu_isolation": True},
        "horizon_information_audit": {"status": "pass" if status == "optimal" else "incomplete", "full_trace_diagnostic": full_trace_diagnostic, "max_visible_request_count": max((len(item["visible_request_ids"]) for item in trace), default=0), "outside_horizon_read": False},
    }
