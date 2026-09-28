"""CRDCM 可观测决策状态的冻结特征合同。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch


CRDCM_OBSERVATION_CONTRACT_VERSION = "crdcm_observation_v1"
CRDCM_DECISION_CONTRACT_VERSION = "crdcm_decision_v1"

CRDCM_FEATURE_NAMES = (
    "current_base_missing",
    "current_adapter_missing",
    "current_bundle_missing_mb_scaled",
    "current_capacity_remaining_mb_scaled",
    "current_cache_occupancy",
    "target_base_missing",
    "target_adapter_missing",
    "target_state_missing",
    "target_bundle_missing_mb_scaled",
    "target_capacity_remaining_mb_scaled",
    "target_cache_occupancy",
    "target_eviction_shortfall_mb_scaled",
    "dag_remaining_nodes_scaled",
    "dag_remaining_ratio",
    "dag_critical_path_scaled",
    "dag_critical_path_pressure",
    "dag_frontier_scaled",
    "current_adapter_remaining_reuse_scaled",
    "current_adapter_remaining_reuse_ratio",
    "workflow_completed_ratio",
    "handoff_target_differs",
    "handoff_confidence",
    "handoff_uncertainty",
    "handoff_eta_scaled",
    "migration_enabled",
    "migration_state_ready",
    "migration_capacity_conflict",
    "predicted_target_available",
)


def _bounded(value: Any, *, scale: float = 1.0) -> float:
    try:
        numeric = float(value or 0.0) / max(float(scale), 1e-8)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(numeric, 1.0))


def build_crdcm_feature_values(payload: dict[str, Any]) -> list[float]:
    """Derive the actor vector only from fields declared observable in v1."""

    current = dict(payload.get("current_rsu", {}) or {})
    target = dict(payload.get("predicted_target_rsu", {}) or {})
    dag = dict(payload.get("remaining_dag", {}) or {})
    migration = dict(payload.get("migration", {}) or {})
    prediction = dict(payload.get("handoff_prediction", {}) or {})
    current_bundle = dict(current.get("required_bundle", {}) or {})
    target_bundle = dict(target.get("required_bundle", {}) or {})
    return [
        float(not bool(current_bundle.get("base_ready", False))),
        float(not bool(current_bundle.get("adapter_ready", False))),
        _bounded(current_bundle.get("missing_resident_mb"), scale=512.0),
        _bounded(current.get("capacity_remaining"), scale=1024.0),
        _bounded(current.get("occupancy_rate")),
        float(not bool(target_bundle.get("base_ready", False))),
        float(not bool(target_bundle.get("adapter_ready", False))),
        float(bool(migration.get("state_required", False)) and not bool(migration.get("state_ready", False))),
        _bounded(target_bundle.get("missing_resident_mb"), scale=512.0),
        _bounded(target.get("capacity_remaining"), scale=1024.0),
        _bounded(target.get("occupancy_rate")),
        _bounded(target_bundle.get("eviction_shortfall_mb"), scale=512.0),
        _bounded(dag.get("remaining_nodes"), scale=32.0),
        _bounded(dag.get("remaining_ratio")),
        _bounded(dag.get("critical_path_length"), scale=32.0),
        _bounded(dag.get("critical_path_pressure")),
        _bounded(dag.get("frontier_size"), scale=16.0),
        _bounded(dag.get("current_adapter_remaining_reuse_count"), scale=16.0),
        _bounded(dag.get("current_adapter_remaining_reuse_ratio")),
        _bounded(dag.get("completed_ratio")),
        float(bool(prediction.get("target_differs_from_current", False))),
        _bounded(prediction.get("confidence")),
        _bounded(prediction.get("uncertainty")),
        _bounded(prediction.get("eta_steps"), scale=12.0),
        float(bool(migration.get("enabled", False))),
        float(bool(migration.get("state_ready", False))),
        float(bool(migration.get("capacity_conflict", False))),
        float(bool(prediction.get("target_available", False))),
    ]


def finalize_crdcm_observation(payload: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(payload)
    result["contract_version"] = CRDCM_OBSERVATION_CONTRACT_VERSION
    result["oracle_future_fields_actor_visible"] = False
    result["outcome_fields_present"] = False
    result["feature_names"] = list(CRDCM_FEATURE_NAMES)
    result["feature_vector"] = build_crdcm_feature_values(result)
    validate_crdcm_observation(result)
    return result


def validate_crdcm_observation(payload: dict[str, Any]) -> None:
    if payload.get("contract_version") != CRDCM_OBSERVATION_CONTRACT_VERSION:
        raise ValueError("unsupported CRDCM observation contract")
    if payload.get("oracle_future_fields_actor_visible") is not False:
        raise ValueError("CRDCM actor observation must not expose oracle future fields")
    if payload.get("outcome_fields_present") is not False:
        raise ValueError("CRDCM actor observation must not expose outcome labels")
    if list(payload.get("feature_names", [])) != list(CRDCM_FEATURE_NAMES):
        raise ValueError("CRDCM feature name/order mismatch")
    values = list(payload.get("feature_vector", []))
    if len(values) != len(CRDCM_FEATURE_NAMES):
        raise ValueError("CRDCM feature vector length mismatch")
    for value in values:
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise ValueError("CRDCM normalized features must be in [0, 1]")


def build_crdcm_feature_tensor(
    semantic_state: dict[str, Any],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    payload = semantic_state.get("crdcm_observation")
    if not isinstance(payload, dict):
        raise ValueError("crdcm_decision_v1 requires semantic_state['crdcm_observation']")
    validate_crdcm_observation(payload)
    return torch.as_tensor(payload["feature_vector"], dtype=dtype, device=device)


__all__ = [
    "CRDCM_DECISION_CONTRACT_VERSION",
    "CRDCM_FEATURE_NAMES",
    "CRDCM_OBSERVATION_CONTRACT_VERSION",
    "build_crdcm_feature_tensor",
    "build_crdcm_feature_values",
    "finalize_crdcm_observation",
    "validate_crdcm_observation",
]
