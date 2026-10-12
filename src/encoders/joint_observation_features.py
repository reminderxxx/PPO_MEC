"""Bounded v2 features with an explicit actor/central-critic data boundary."""

from __future__ import annotations

import math
from typing import Any


JOINT_ACTOR_FEATURE_DIM = 37
JOINT_CRITIC_FEATURE_DIM = 20


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _ratio(value: Any, scale: Any = 1.0) -> float:
    return min(1.0, max(0.0, _number(value) / max(_number(scale), 1.0)))


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def joint_actor_features(state: dict[str, Any]) -> list[float]:
    """Read only the acting request and available (possibly stale) RSU views.

    The centralized_critic_state key is deliberately never read here. Numeric
    scales are fixed feature transforms, not physical calibration parameters.
    """
    vehicle_id = state.get("primary_vehicle_id")
    vehicle = next((row for row in state.get("vehicles", [])
                    if row.get("vehicle_id") == vehicle_id), {})
    current = vehicle.get("associated_rsu_id")
    predictions = state.get("predictions", {})
    target = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
    node = state.get("current_workflow_node") or {}
    adapter, base = node.get("required_adapter"), node.get("required_base_model")
    rsus = {row.get("rsu_id"): row for row in state.get("rsus", [])}
    features = []
    for rsu_id in (current, target):
        row = rsus.get(rsu_id, {})
        capacity = row.get("capacity_bytes", 1)
        remaining_frames = row.get("loading_remaining_frames", {})
        features.extend([
            float(bool(row)),
            float(bool(row.get("shared_state_available", bool(row)))),
            float(bool(adapter and adapter in row.get("cached_adapter_ids", []))),
            float(bool(base and base in row.get("ready_base_model_ids", []))),
            float(bool(adapter and adapter in row.get("loading_adapter_ids", []))),
            _ratio(row.get("occupied_bytes", 0), capacity),
            _ratio(row.get("reserved_bytes", 0), capacity),
            _ratio(row.get("free_bytes", 0), capacity),
            _ratio(remaining_frames.get(f"adapter:{adapter}", 0), 32),
            _ratio(remaining_frames.get(f"base:{base}", 0), 32),
        ])
    shared = state.get("shared_state", {})
    copy = state.get("state_copies_by_rsu", {}).get(target, {})
    transfer = state.get("state_transfers_by_rsu", {}).get(target, {})
    version = _number(state.get("execution_state_version", 0))
    payload = state.get("execution_state_payload_bytes", 0)
    features.extend([
        _ratio(state.get("current_node_work_units_progress", 0), state.get("current_node_work_units_required", 1)),
        _ratio(state.get("current_node_service_steps_remaining", 0), 32),
        _ratio(state.get("vehicle_service_steps_remaining", 0), 128),
        float(bool(state.get("current_node_parents_ready", False))),
        float(bool(state.get("current_adapter_ready", False))),
        float(bool(state.get("current_state_ready", False))),
        _ratio(version, 256),
        _ratio(payload, 8 * 1024 * 1024),
        float(bool(shared.get("available", False))),
        _ratio(shared.get("age_frames", 0), 32),
        float(bool(shared.get("available", False)) and _number(shared.get("age_frames")) == 0),
        float(bool(copy)),
        _ratio(version - _number(copy.get("version", 0)), 256),
        _ratio(copy.get("payload_bytes", 0), payload),
        _ratio(transfer.get("remaining_bytes", 0), payload),
        _ratio(transfer.get("remaining_frames", 0), 32),
        float(bool(copy and _number(copy.get("version")) == version)),
    ])
    assert len(features) == JOINT_ACTOR_FEATURE_DIM
    return features


def joint_critic_features(state: dict[str, Any]) -> list[float]:
    """Aggregate true joint execution/cache state supplied only to the critic."""
    central = state.get("centralized_critic_state", {})
    rows = list(central.get("joint_workflows", {}).values())
    rsus = list(central.get("rsus", []))
    remaining_nodes = [_ratio(row.get("remaining_nodes", 0), 32) for row in rows]
    remaining_work = [_ratio(row.get("remaining_work_units", 0), 32) for row in rows]
    versions = [_ratio(row.get("state_version", 0), 256) for row in rows]
    occupancy = [_ratio(row.get("occupied_bytes", 0), row.get("capacity_bytes", 1)) for row in rsus]
    reserved = [_ratio(row.get("reserved_bytes", 0), row.get("capacity_bytes", 1)) for row in rsus]
    associations = central.get("associations", {})
    loads = [sum(value == rsu.get("rsu_id") for value in associations.values()) for rsu in rsus]
    features = [
        _ratio(len(rows), 32),
        _ratio(sum(_number(row.get("remaining_nodes", 0)) > 0 for row in rows), 32),
        _mean([float(_number(row.get("remaining_nodes", 0)) == 0) for row in rows]),
        _mean(remaining_nodes), max(remaining_nodes, default=0.0),
        _mean(remaining_work), max(remaining_work, default=0.0),
        _mean([_ratio(row.get("work_progress", 0), _number(row.get("work_progress", 0))
                      + _number(row.get("remaining_work_units", 0))) for row in rows]),
        _mean([float(bool(row.get("parents_ready", False))) for row in rows]),
        _mean(versions), max(versions, default=0.0),
        _mean([float(str(row.get("execution_location", "")).startswith("vehicle:")) for row in rows]),
        _mean([float(row.get("execution_location") is None) for row in rows]),
        _mean(occupancy), max(occupancy, default=0.0),
        _mean(reserved), max(reserved, default=0.0),
        _ratio(sum(len(rsu.get("loading_object_ids", [])) for rsu in rsus), 32),
        _ratio(_mean(loads), 32), _ratio(max(loads, default=0), 32),
    ]
    assert len(features) == JOINT_CRITIC_FEATURE_DIM
    return features
