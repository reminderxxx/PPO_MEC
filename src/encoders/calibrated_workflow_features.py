"""Shared public-feature helpers for the calibrated-workflow interface profile."""

from __future__ import annotations

import math
from typing import Any


CALIBRATED_WORKFLOW_INTERFACE_V2 = "calibrated_workflow_interface_v2"


def uses_calibrated_workflow_interface_v2(semantic_state: dict[str, Any]) -> bool:
    return str(semantic_state.get("interface_profile", "")) == CALIBRATED_WORKFLOW_INTERFACE_V2


def primary_vehicle(semantic_state: dict[str, Any]) -> dict[str, Any]:
    vehicles = list(semantic_state.get("vehicles", []) or [])
    primary_vehicle_id = semantic_state.get("primary_vehicle_id")
    if primary_vehicle_id is not None:
        for vehicle in vehicles:
            if str(vehicle.get("vehicle_id")) == str(primary_vehicle_id):
                return dict(vehicle)
    return dict(vehicles[0]) if vehicles else {}


def rsu_by_id(semantic_state: dict[str, Any], rsu_id: Any) -> dict[str, Any]:
    for rsu in semantic_state.get("rsus", []) or []:
        if str(rsu.get("rsu_id")) == str(rsu_id):
            return dict(rsu)
    return {}


def predicted_target_rsu_id(semantic_state: dict[str, Any]) -> Any:
    vehicle = primary_vehicle(semantic_state)
    vehicle_id = str(vehicle.get("vehicle_id", ""))
    predictions = semantic_state.get("predictions", {}) or {}
    return predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)


def required_bundle_ids(
    semantic_state: dict[str, Any],
    node: dict[str, Any] | None = None,
) -> list[str]:
    node = dict(node or semantic_state.get("current_workflow_node") or {})
    adapter_id = node.get("required_adapter")
    base_id = node.get("required_base_model")
    context = semantic_state.get("calibrated_context", {}) or {}
    adapter_to_bundle = context.get("adapter_to_bundle", {}) or {}
    bundle = [str(item) for item in adapter_to_bundle.get(adapter_id, [])]
    if base_id is not None and str(base_id) not in bundle:
        bundle.insert(0, str(base_id))
    if adapter_id is not None:
        adapter_object_id = f"adapter:{adapter_id}"
        if adapter_object_id not in bundle:
            bundle.append(adapter_object_id)
    return bundle


def bundle_ready(
    semantic_state: dict[str, Any],
    rsu: dict[str, Any],
    node: dict[str, Any] | None = None,
) -> float:
    required = set(required_bundle_ids(semantic_state, node))
    residents = {str(item) for item in rsu.get("typed_resident_object_ids", []) or []}
    if required and residents:
        return float(required.issubset(residents))
    node = dict(node or semantic_state.get("current_workflow_node") or {})
    adapter_id = node.get("required_adapter")
    return float(
        adapter_id is not None
        and str(adapter_id) in {str(item) for item in rsu.get("cached_adapter_ids", []) or []}
    )


def base_ready(
    semantic_state: dict[str, Any],
    rsu: dict[str, Any],
    node: dict[str, Any] | None = None,
) -> float:
    node = dict(node or semantic_state.get("current_workflow_node") or {})
    base_id = node.get("required_base_model")
    residents = {str(item) for item in rsu.get("typed_resident_object_ids", []) or []}
    return float(base_id is not None and str(base_id) in residents)


def adapter_ready(rsu: dict[str, Any], node: dict[str, Any] | None) -> float:
    node = dict(node or {})
    adapter_id = node.get("required_adapter")
    return float(
        adapter_id is not None
        and str(adapter_id) in {str(item) for item in rsu.get("cached_adapter_ids", []) or []}
    )


def cache_occupancy(rsu: dict[str, Any]) -> float:
    capacity = max(float(rsu.get("cache_capacity", 0.0) or 0.0), 1.0)
    used = float(rsu.get("cache_used_bytes", 0.0) or 0.0)
    return max(0.0, min(used / capacity, 1.0))


def bundle_resident_bytes(
    semantic_state: dict[str, Any],
    node: dict[str, Any] | None = None,
) -> int:
    context = semantic_state.get("calibrated_context", {}) or {}
    catalog = context.get("object_catalog", {}) or {}
    return int(
        sum(
            int((catalog.get(object_id, {}) or {}).get("resident_bytes", 0) or 0)
            for object_id in required_bundle_ids(semantic_state, node)
        )
    )


def log_scale(value: float, reference: float) -> float:
    return max(0.0, min(math.log1p(max(float(value), 0.0)) / max(math.log1p(reference), 1e-9), 1.0))
