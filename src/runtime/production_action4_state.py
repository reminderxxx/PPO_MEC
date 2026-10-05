"""Explicit opt-in state transfer used by production action 4."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any, Mapping

from src.runtime.workflow_suffix_recovery import (
    read_production_state_package,
    seal_production_state_payload,
    write_state_package,
)


class ProductionAction4StateError(RuntimeError):
    """Fail-closed action-4 state transfer error."""


def export_action4_state(
    *,
    package_dir: Path,
    workflow: Mapping[str, Any],
    completed_nodes: list[str],
    node_outputs: Mapping[str, Any],
    identity: Mapping[str, Any],
    input_identity: Mapping[str, Any],
    source_rsu_id: str,
    target_rsu_id: str,
) -> dict[str, Any]:
    order = list(workflow.get("execution_order") or [])
    remaining = [node_id for node_id in order if node_id not in completed_nodes]
    if not completed_nodes or not remaining:
        raise ProductionAction4StateError(
            "action 4 export requires a completed prefix and remaining suffix"
        )
    started = time.perf_counter()
    payload = {
        "workflow": deepcopy(dict(workflow)),
        "completed_nodes": list(completed_nodes),
        "remaining_nodes": remaining,
        "next_node": remaining[0],
        "node_outputs": deepcopy(dict(node_outputs)),
        "identity": deepcopy(dict(identity)),
        "input_identity": deepcopy(dict(input_identity)),
        "execution_rights": {
            "source_rsu_id": str(source_rsu_id),
            "target_rsu_id": str(target_rsu_id),
            "status": "pending_target_validation",
        },
    }
    envelope = seal_production_state_payload(payload)
    package = write_state_package(package_dir, envelope)
    return {
        "status": "EXPORTED_PENDING_IMPORT",
        "migration_success": False,
        "execution_right_transferred": False,
        "source_rsu_id": str(source_rsu_id),
        "target_rsu_id": str(target_rsu_id),
        "completed_nodes": list(completed_nodes),
        "next_node": remaining[0],
        "serialize_and_save_seconds": time.perf_counter() - started,
        "package_dir": str(package_dir),
        **package,
    }


def import_action4_state(
    *,
    package_dir: Path,
    expected_workflow: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    expected_input_identity: Mapping[str, Any],
    target_model_ready: bool,
    target_rsu_id: str,
) -> dict[str, Any]:
    """Validate state only after the target model is ready; commit last."""
    started = time.perf_counter()
    if not target_model_ready:
        return {
            "status": "BLOCKED_MISSING_TARGET_MODEL",
            "migration_success": False,
            "execution_right_transferred": False,
            "target_rsu_id": str(target_rsu_id),
            "missing_model_preparation": True,
            "restore_validation_seconds": 0.0,
        }
    try:
        payload, package = read_production_state_package(
            package_dir,
            expected_workflow=expected_workflow,
            expected_identity=expected_identity,
            expected_input_identity=expected_input_identity,
        )
        rights = dict(payload.get("execution_rights") or {})
        if rights.get("target_rsu_id") != str(target_rsu_id):
            raise ProductionAction4StateError("target RSU identity mismatch")
    except BaseException as error:
        return {
            "status": "REJECTED_STATE_VALIDATION",
            "migration_success": False,
            "execution_right_transferred": False,
            "target_rsu_id": str(target_rsu_id),
            "missing_model_preparation": False,
            "restore_validation_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {
        "status": "IMPORTED_COMMITTED",
        "migration_success": True,
        "execution_right_transferred": True,
        "target_rsu_id": str(target_rsu_id),
        "missing_model_preparation": False,
        "completed_nodes": list(payload["completed_nodes"]),
        "remaining_nodes": list(payload["remaining_nodes"]),
        "next_node": payload["next_node"],
        "node_outputs": deepcopy(payload["node_outputs"]),
        "restore_validation_seconds": time.perf_counter() - started,
        **package,
    }
