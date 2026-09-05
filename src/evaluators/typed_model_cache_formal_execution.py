"""Executable contracts for typed model-cache protocols v1.1 and v1.2.

This module is intentionally outcome-blind.  It validates frozen settings,
expands commands, binds support runs to typed provenance, and maintains an
append-only phase ledger.  It has no holdout-opening API.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.evaluators.typed_model_cache_formal_protocol import (
    attach_hashes,
    canonical_sha256,
    semantic_projection,
)
from src.runtime.formal_protocol_capabilities import (
    FORMAL_PROTOCOL_CAPABILITY_ROUTING_CONTRACT_VERSION,
    get_protocol_capabilities,
    require_live_execution_protocol,
)


FORMAL_EXECUTION_PROTOCOL_VERSION = "1.1.0"
FORMAL_EXECUTION_PROTOCOL_ID = "typed_model_cache_formal_protocol_v1_1"
FORMAL_EXECUTION_PROTOCOL_V1_2_VERSION = "1.2.0"
FORMAL_EXECUTION_PROTOCOL_V1_2_ID = "typed_model_cache_formal_protocol_v1_2"
FORMAL_EXECUTION_PROTOCOL_V1_3_VERSION = "1.3.0"
FORMAL_EXECUTION_PROTOCOL_V1_3_ID = "typed_model_cache_formal_protocol_v1_3"
FORMAL_EXECUTION_PROTOCOL_V1_4_VERSION = "1.4.0"
FORMAL_EXECUTION_PROTOCOL_V1_4_ID = "typed_model_cache_formal_protocol_v1_4"
FORMAL_EXECUTION_PROTOCOL_V1_5_VERSION = "1.5.0"
FORMAL_EXECUTION_PROTOCOL_V1_5_ID = "typed_model_cache_formal_protocol_v1_5"
FORMAL_EXECUTION_PROTOCOL_V1_6_VERSION = "1.6.0"
FORMAL_EXECUTION_PROTOCOL_V1_6_ID = "typed_model_cache_formal_protocol_v1_6"
FORMAL_EXECUTION_PROTOCOL_V1_7_VERSION = "1.7.0"
FORMAL_EXECUTION_PROTOCOL_V1_7_ID = "typed_model_cache_formal_protocol_v1_7"
FORMAL_EXECUTION_PROTOCOL_V1_8_VERSION = "1.8.0"
FORMAL_EXECUTION_PROTOCOL_V1_8_ID = "typed_model_cache_formal_protocol_v1_8"
FORMAL_EXECUTION_PROTOCOL_V1_9_VERSION = "1.9.0"
FORMAL_EXECUTION_PROTOCOL_V1_9_ID = "typed_model_cache_formal_protocol_v1_9"
FORMAL_EXECUTION_PROTOCOL_V2_0_VERSION = "2.0.0"
FORMAL_EXECUTION_PROTOCOL_V2_0_ID = "typed_model_cache_formal_protocol_v2_0"
FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION = "2.1.0"
FORMAL_EXECUTION_PROTOCOL_V2_1_ID = "typed_model_cache_formal_protocol_v2_1"
FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION = "2.2.0"
FORMAL_EXECUTION_PROTOCOL_V2_2_ID = "typed_model_cache_formal_protocol_v2_2"
FORMAL_EXECUTION_PROTOCOL_V2_3_VERSION = "2.3.0"
FORMAL_EXECUTION_PROTOCOL_V2_3_ID = "typed_model_cache_formal_protocol_v2_3"
FORMAL_EXECUTION_PROTOCOL_V2_4_VERSION = "2.4.0"
FORMAL_EXECUTION_PROTOCOL_V2_4_ID = "typed_model_cache_formal_protocol_v2_4"
FORMAL_EXECUTION_PROTOCOL_V2_5_VERSION = "2.5.0"
FORMAL_EXECUTION_PROTOCOL_V2_5_ID = "typed_model_cache_formal_protocol_v2_5"
FORMAL_EXECUTION_PROTOCOL_V2_6_VERSION = "2.6.0"
FORMAL_EXECUTION_PROTOCOL_V2_6_ID = "typed_model_cache_formal_protocol_v2_6"
FORMAL_PHASE_RUNNER_VERSION = "2.0.0"
FORMAL_PHASE_LEDGER_SCHEMA_VERSION = "2.0.0"
LEGACY_PRIMARY_ENDPOINT_SCHEMA_VERSION = "1.0.0"
PRIMARY_ENDPOINT_SCHEMA_VERSION = "2.0.0"
SUPPORT_RUNNER_CONTRACT_VERSION = "1.0.0"
READINESS_REVIEW_VERSION = "3.0.0"
READY_VERDICT = "READY_FOR_G14C_V2_CLEAN_TRAIN_AND_FORMAL"
READY_V4_VERDICT = "READY_FOR_G14C_V3_CLEAN_TRAIN_AND_FORMAL"
READY_V5_VERDICT = "READY_FOR_G14C_V4_CLEAN_TRAIN_AND_FORMAL"
READY_V6_VERDICT = "READY_FOR_G14C_V5_CLEAN_TRAIN_AND_FORMAL"
READY_V7_VERDICT = "READY_FOR_G14C_V6_CLEAN_TRAIN_AND_FORMAL"
READY_V8_VERDICT = "READY_FOR_G14C_V7_CLEAN_TRAIN_AND_FORMAL"
READY_V9_VERDICT = "READY_FOR_G14C_V8_CLEAN_TRAIN_AND_FORMAL"
OLD_PROTOCOL_SEMANTIC_SHA256 = (
    "41fbfab4ac10bae96250d7ead816d907fd6551bb9651ae03210e801c9e2478b4"
)
SPLIT_SEMANTIC_SHA256 = (
    "aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a"
)
PHASE_ORDER = (
    "preflight",
    "tests",
    "train",
    "dev_select",
    "checkpoint_freeze",
    "formal_cache_policy",
    "formal_controller",
    "formal_ablation",
    "formal_support",
    "formal_scalability",
    "formal_statistics",
    "formal_gate",
    "complete_without_holdout",
)
FORMAL_PHASES = {
    "formal_cache_policy",
    "formal_controller",
    "formal_ablation",
    "formal_support",
    "formal_scalability",
    "formal_statistics",
    "formal_gate",
}
PRIMARY_ENDPOINTS = (
    "full_service_ready_byte_hit_rate",
    "joint_base_adapter_hit_rate",
    "full_service_ready_request_rate",
    "transfer_mb_per_request",
    "workflow_continuity_rate",
    "end_to_end_workflow_delay",
)
PLACEHOLDER_PATTERN = re.compile(r"\{([a-z][a-z0-9_]*)\}")
HOLDOUT_TERMS = ("sealed_holdout", "holdout_token", "hidden", "--holdout")
FAILURE_CLASSIFICATIONS = (
    "infrastructure_retryable",
    "infrastructure_terminal",
    "protocol_mismatch",
    "implementation_error",
    "data_window_unreachable",
    "test_failure",
    "training_failure",
    "artifact_integrity_failure",
    "user_interruption",
)
TERMINAL_LEDGER_STATUSES = {"completed", "failed"}
LEDGER_WALL_CLOCK_TOLERANCE_SECONDS = 2.0


class FormalExecutionError(ValueError):
    """Raised when an execution contract would be ambiguous or mutable."""


def _reject_non_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FormalExecutionError(f"non-finite JSON value at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise FormalExecutionError(f"non-string JSON key at {path}")
            _reject_non_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_non_finite(child, f"{path}[{index}]")


def stable_setting_identity(family: str, parameters: Mapping[str, Any]) -> str:
    _reject_non_finite(parameters)
    digest = canonical_sha256({"family": family, "parameters": dict(parameters)})[:16]
    return f"{family}-{digest}"


def endpoint_schema() -> dict[str, Any]:
    return {
        "primary_endpoint_schema_version": PRIMARY_ENDPOINT_SCHEMA_VERSION,
        "formal_endpoint_metrics_contract_version": "2.0.0",
        "request_execution_contract_version": "1.0.0",
        "request_exposure_trace_version": "1.0.0",
        "external_request_denominator": (
            "exactly one frozen policy-neutral request exposure per request-level CacheEvent"
        ),
        "outcome_independence": (
            "service/cache/workflow outcomes never add, remove, reorder, or retry exposure rows"
        ),
        "event_eligibility": (
            "CacheEvent event_type=request and model_cache_profile_id="
            "typed_base_adapter_state_v1"
        ),
        "full_service_ready_byte_hit_rate": {
            "formula": (
                "sum(unique requested dependency resident bytes within each eligible request "
                "where full_service_ready=true) / sum(unique requested dependency resident "
                "bytes within every eligible request)"
            ),
            "dependency_types": ["base_model", "adapter"],
            "shared_base_semantics": (
                "one base object is counted once within one request; a distinct request counts "
                "its required base once again; lookup rows and resident inventory never add bytes"
            ),
            "partial_readiness": "zero numerator bytes for the whole request",
            "missing_size": "primary value null with partial availability and explicit coverage",
            "zero_denominator": "null",
            "unit": "ratio",
            "legacy_trace": "unavailable",
        },
        "transfer_mb_per_request": {
            "formula": (
                "(base_model_transfer_mb + adapter_transfer_mb + "
                "workflow_state_migration_transfer_mb) / typed_request_event_count"
            ),
            "components": [
                "base_model_transfer_mb",
                "adapter_transfer_mb",
                "workflow_state_migration_transfer_mb",
            ],
            "other_typed_transfer": "reported separately and excluded from primary",
            "zero_denominator": "null",
            "unit": "decimal MB/request",
            "legacy_trace": "unavailable",
        },
        "workflow_continuity_rate": {
            "formula": "successful request-level service outcomes / external request denominator",
            "predecessor_failure": "later frozen exposures remain in the denominator",
            "right_censoring": "reported explicitly; exposed requests remain identifiable",
            "zero_denominator": "null",
            "unit": "ratio",
        },
        "end_to_end_workflow_delay": {
            "availability": (
                "available only for a complete, uncensored workflow whose every exposed request succeeds"
            ),
            "failed_or_incomplete_workflow": "null/unavailable",
            "selection_rule": "null is never imputed from reward and is ordered after finite values",
            "unit": "environment time units",
        },
        "row_fields": list(PRIMARY_ENDPOINTS),
        "nullable_aggregate": True,
        "raw_event_recomputation_required": True,
    }


def build_support_setting_matrix() -> dict[str, Any]:
    raw: list[dict[str, Any]] = []

    def add(
        family: str,
        parameter: str,
        values: list[Any],
        unit: str,
        baseline: Any,
        status: str,
        runtime_binding: str,
        *,
        primary: bool = False,
        reason: str | None = None,
        resource_budget: str = "same frozen formal evaluation unit budget",
    ) -> None:
        parameters = {"parameter": parameter, "values": values, "baseline": baseline, "unit": unit}
        levels = []
        for value in values:
            level_status = status
            level_reason = reason
            if parameter == "typed_semantics" and value not in {"typed_full", "no_prediction"}:
                level_status = "unavailable_pre_execution"
                level_reason = (
                    "variant lacks a typed fingerprint-preserving runtime transformer under the frozen contract"
                )
            elif parameter == "typed_semantics":
                level_status = "available"
                level_reason = None
            level_parameters = {"parameter": parameter, "value": value, "unit": unit}
            levels.append(
                {
                    "setting_id": stable_setting_identity(family, level_parameters),
                    "value": value,
                    "status": level_status,
                    "unavailable_reason": level_reason,
                    "parameters": level_parameters,
                }
            )
        raw.append(
            {
                "family": family,
                "parameter": parameter,
                "values": values,
                "unit": unit,
                "baseline": baseline,
                "range": [values[0], values[-1]],
                "seed_plan": [7, 13, 29, 43, 71],
                "role": "primary" if primary else "exploratory",
                "status": status,
                "unavailable_reason": reason,
                "runtime_binding": runtime_binding,
                "expected_artifact": f"{family}/{parameter}/<setting_id>",
                "cli_override": "forbidden; select only a frozen setting_id",
                "resource_budget": resource_budget,
                "setting_id": stable_setting_identity(family, parameters),
                "levels": levels,
            }
        )

    add("capacity", "capacity_mb", [288.0, 576.0, 864.0], "decimal MB", 576.0, "available", "runtime config + fairness manifest", primary=True)
    add("sensitivity", "object_size_scale", [0.75, 1.0, 1.25], "multiplier", 1.0, "unavailable_pre_execution", "none", reason="runtime catalog scaling would change catalog/dependency/initial fingerprints; no frozen safe transformer")
    add("sensitivity", "transfer_cost_scale", [0.5, 1.0, 1.5], "multiplier", 1.0, "unavailable_pre_execution", "none", reason="environment exposes transfer bytes but no independent typed transfer-cost coefficient")
    add("sensitivity", "handoff_pressure", ["stable_first", "handoff_pressure"], "vehicle selection rule", "handoff_pressure", "available", "--primary_vehicle_selection mapped before outcomes")
    add("sensitivity", "reuse_opportunity", ["low", "medium", "high"], "predefined request profile", "medium", "unavailable_pre_execution", "none", reason="no outcome-blind request-profile parameter exists; oracle-gap grouping is forbidden")
    add("sensitivity", "base_sharing_degree", [1, 3, 6], "adapters per base", 3, "unavailable_pre_execution", "none", reason="no dependency-preserving typed catalog transformer with fixed object accounting exists")
    add("ablation", "typed_semantics", ["typed_full", "legacy_adapter_only", "no_base_sharing", "no_workflow_state_migration", "fixed_no_eviction", "no_prediction"], "variant identity", "typed_full", "partially_available", "typed_full and no_prediction executable; other variants unavailable per setting")
    add("prediction_boundary", "prediction_condition", ["baseline", "no_prediction", "noise_0.2", "confidence_0.7", "delay_2", "drop_0.3"], "fixed condition", "baseline", "available", "benchmark predictor flags; G12 supervised remains disabled")
    add("oracle_state_limit", "state_limit", [1000, 10000, 100000], "visited states", 10000, "available", "run_future_horizon_cache_oracle.py --state_limit", resource_budget="one exact solve per horizon/evaluation unit; unknown_state_limit retained")
    return {
        "support_setting_matrix_version": "1.0.0",
        "settings": raw,
        "g12_supervised_predictor_enabled": False,
        "kv_enabled": False,
        "hf_metadata_profile_enabled": False,
    }


def build_scalability_setting_matrix() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    definitions = (
        ("rsu_count", [2, 3, 4], "RSUs", 3, "unavailable_pre_execution", "typed initial-state identity is fixed to three RSUs"),
        ("active_vehicle_count", [4, 8, 16], "active vehicles/window", 8, "unavailable_pre_execution", "vehicle subsampling is not encoded by current fairness manifest"),
        ("dag_node_count", [5, 10, 20], "nodes/workflow", 10, "unavailable_pre_execution", "existing buckets do not instantiate exact node counts"),
        ("typed_object_count", [4, 8, 10], "capacity-counting objects", 8, "unavailable_pre_execution", "existing adapter proxy is legacy and breaks typed catalog fingerprints"),
        ("oracle_state_limit", [1000, 10000, 100000], "visited states", 10000, "available", None),
    )
    for parameter, values, unit, baseline, status, reason in definitions:
        params = {"parameter": parameter, "values": values, "baseline": baseline, "unit": unit}
        rows.append(
            {
                **params,
                "range": [values[0], values[-1]],
                "seed_plan": [7, 13, 29, 43, 71],
                "role": "exploratory",
                "status": status,
                "unavailable_reason": reason,
                "setting_id": stable_setting_identity("scalability", params),
                "levels": [
                    {
                        "setting_id": stable_setting_identity(
                            "scalability",
                            {"parameter": parameter, "value": value, "unit": unit},
                        ),
                        "value": value,
                        "status": status,
                        "unavailable_reason": reason,
                        "parameters": {
                            "parameter": parameter,
                            "value": value,
                            "unit": unit,
                        },
                    }
                    for value in values
                ],
                "wall_clock_repetitions": 3,
                "memory_measurement": "Python tracemalloc peak increment bytes; process RSS unavailable and not inferred",
                "expected_artifact": f"scalability/{parameter}/<setting_id>",
                "cli_override": "forbidden; select only a frozen setting_id",
                "resource_budget": "3 repetitions x frozen evaluation units; exact oracle bounded by state_limit",
            }
        )
    return {"scalability_setting_matrix_version": "1.0.0", "settings": rows}


def support_setting_by_id(protocol: Mapping[str, Any], setting_id: str) -> dict[str, Any]:
    containers = (
        protocol.get("ablation_and_support", {}).get("support_setting_matrix", {}),
        protocol.get("ablation_and_support", {}).get("scalability_setting_matrix", {}),
    )
    matches: list[dict[str, Any]] = []
    for container in containers:
        for item in container.get("settings", []):
            if item.get("setting_id") == setting_id:
                matches.append(dict(item))
            for level in item.get("levels", []):
                if level.get("setting_id") == setting_id:
                    matches.append(
                        {
                            **dict(item),
                            **dict(level),
                            "dimension_setting_id": item.get("setting_id"),
                            "family": item.get("family", "scalability"),
                            "parameter": item.get("parameter"),
                            "unit": item.get("unit"),
                        }
                    )
    if len(matches) != 1:
        raise FormalExecutionError(f"unknown or duplicate support setting_id: {setting_id}")
    return matches[0]


def validate_support_binding(
    *,
    protocol: Mapping[str, Any],
    setting_id: str,
    runtime_contract: Mapping[str, Any],
    fairness_manifest: Mapping[str, Any],
    checkpoint_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    setting = support_setting_by_id(protocol, setting_id)
    if setting.get("status") == "unavailable_pre_execution":
        raise FormalExecutionError(
            f"support setting unavailable_pre_execution: {setting.get('unavailable_reason')}"
        )
    if runtime_contract.get("model_cache_profile") != "typed_base_adapter_state_v1":
        raise FormalExecutionError("typed support runner rejects legacy runtime")
    if runtime_contract.get("cache_capacity_profile", {}).get("unit") != "mb":
        raise FormalExecutionError("typed support runner rejects slot capacity")
    identity = fairness_manifest.get("identity", {})
    if identity.get("cache_baseline_fairness_manifest_version") != "1.1.0":
        raise FormalExecutionError("typed support runner requires fairness manifest 1.1.0")
    typed = fairness_manifest.get("cache_contract", {}).get("typed_model_cache")
    if not isinstance(typed, Mapping):
        raise FormalExecutionError("fairness manifest lacks typed binding")
    if typed.get("catalog_fingerprint") != runtime_contract.get("typed_catalog_fingerprint"):
        raise FormalExecutionError("support fairness/runtime catalog mismatch")
    if not checkpoint_provenance and setting.get("parameter") != "oracle_state_limit":
        raise FormalExecutionError("typed learned support requires checkpoint provenance")
    return {
        "typed_support_runner_contract_version": SUPPORT_RUNNER_CONTRACT_VERSION,
        "setting_id": setting_id,
        "support_family": setting.get("family", "scalability"),
        "support_setting_sha256": canonical_sha256(setting),
        "protocol_semantic_sha256": protocol.get("hashes", {}).get("semantic_sha256"),
        "split_semantic_sha256": protocol.get("identity", {}).get("split_semantic_sha256"),
        "runtime_contract_sha256": runtime_contract.get("runtime_contract_sha256"),
        "typed_catalog_fingerprint": runtime_contract.get("typed_catalog_fingerprint"),
        "fairness_manifest_id": identity.get("manifest_id"),
        "fairness_manifest_hash": fairness_manifest.get("hashes", {}).get("full_manifest_sha256"),
        "checkpoint_provenance_sha256": (
            canonical_sha256(checkpoint_provenance)
            if checkpoint_provenance
            else None
        ),
        "append_only": True,
        "g12_supervised_predictor_enabled": False,
        "kv_enabled": False,
        "hf_metadata_profile_enabled": False,
    }


def expand_command_template(
    template: Sequence[str], values: Mapping[str, Any]
) -> list[str]:
    expanded: list[str] = []
    for token in template:
        if not isinstance(token, str):
            raise FormalExecutionError("command template token must be a string")
        names = PLACEHOLDER_PATTERN.findall(token)
        rendered = token
        for name in names:
            if name not in values:
                raise FormalExecutionError(f"unresolved command placeholder: {name}")
            value = values[name]
            if value is None or isinstance(value, (dict, list, tuple)):
                raise FormalExecutionError(f"invalid command placeholder value: {name}")
            rendered = rendered.replace("{" + name + "}", str(value))
        if "{" in rendered or "}" in rendered:
            raise FormalExecutionError(f"unresolved command template token: {rendered}")
        expanded.append(rendered)
    validate_no_holdout_capability(expanded)
    return expanded


def validate_no_holdout_capability(command: Sequence[str]) -> None:
    joined = " ".join(command).lower()
    if any(term in joined for term in HOLDOUT_TERMS):
        raise FormalExecutionError("ordinary formal runner has no holdout capability")


def expand_command_plan(
    spec: Mapping[str, Any], expansion_context: Mapping[str, Any]
) -> dict[str, Any]:
    """Expand every frozen matrix cell for one phase command specification."""

    base_context = dict(expansion_context)
    location_sentinels = {
        "/ABSOLUTE/FORMAL_OUTPUT_ROOT": base_context.get("output_root"),
        "/ABSOLUTE/CLEAN_WORKTREE_ROOT": base_context.get("clean_worktree_root"),
    }

    def resolve_location(value: Any, key: str | None = None) -> Any:
        if not isinstance(value, str):
            return value
        for sentinel, root in location_sentinels.items():
            if root is not None and value.startswith(sentinel):
                value = str(root) + value[len(sentinel):]
                break
        if (
            base_context.get("resolve_relative_paths_against_repository_root") is True
            and key is not None
            and (key.endswith("_path") or key.endswith("_root"))
            and "{" not in value
            and not Path(value).is_absolute()
        ):
            repository_root = base_context.get("repository_root")
            if not repository_root or not Path(str(repository_root)).is_absolute():
                raise FormalExecutionError(
                    "resolved command expansion requires an absolute repository root"
                )
            value = str((Path(str(repository_root)) / value).resolve())
        return value

    base_context = {
        key: resolve_location(value, key) for key, value in base_context.items()
    }
    raw_contexts = spec.get("matrix_contexts", [{}])
    if not isinstance(raw_contexts, list) or not raw_contexts:
        raise FormalExecutionError("command template matrix_contexts must be non-empty")
    commands: list[list[str]] = []
    expected_outputs: list[str] = []
    resolved_contexts: list[dict[str, Any]] = []
    for index, overlay in enumerate(raw_contexts):
        if not isinstance(overlay, Mapping):
            raise FormalExecutionError(
                f"command template matrix context {index} must be an object"
            )
        resolved_overlay = {
            key: resolve_location(value, key) for key, value in dict(overlay).items()
        }
        context = {**base_context, **resolved_overlay}
        commands.append(expand_command_template(spec["argv"], context))
        for pattern in spec["expected_outputs"]:
            rendered = expand_command_template([str(pattern)], context)[0]
            if rendered not in expected_outputs:
                expected_outputs.append(rendered)
        # Coordinates deliberately retain portable/scientific values; resolved
        # host paths live only in argv/expected outputs and the full context.
        resolved_contexts.append(dict(overlay))
    return {
        "commands": commands,
        "expected_outputs": expected_outputs,
        "matrix_cell_count": len(commands),
        "matrix_contexts": resolved_contexts,
    }


def validate_command_templates(
    templates: Mapping[str, Any], expansion_context: Mapping[str, Any]
) -> dict[str, Any]:
    required = set(PHASE_ORDER) - {"complete_without_holdout"}
    missing = required - set(templates)
    if missing:
        raise FormalExecutionError(f"missing command template phase(s): {sorted(missing)}")
    expanded: dict[str, Any] = {}
    ordered_templates = [
        *[phase for phase in PHASE_ORDER if phase in required],
        *sorted(set(templates) - required),
    ]
    for phase in ordered_templates:
        spec = templates[phase]
        if not isinstance(spec, Mapping) or not isinstance(spec.get("argv"), list):
            raise FormalExecutionError(f"invalid command template: {phase}")
        if not spec.get("expected_outputs"):
            raise FormalExecutionError(f"command template lacks expected_outputs: {phase}")
        plan = expand_command_plan(spec, expansion_context)
        resume_phase = spec.get("resume_phase")
        if phase in required and resume_phase != phase:
            raise FormalExecutionError(f"command template resume phase mismatch: {phase}")
        if phase not in required and resume_phase not in PHASE_ORDER:
            raise FormalExecutionError(
                f"auxiliary command template lacks owning resume phase: {phase}"
            )
        retries = spec.get("infrastructure_retries")
        if retries not in {0, 1}:
            raise FormalExecutionError("infrastructure retry must be zero or one")
        expanded[phase] = {
            "argv": plan["commands"][0],
            **plan,
            "timeout_seconds": int(spec.get("timeout_seconds", 0)),
            "infrastructure_retries": retries,
        }
    canonical_expansion = {
        "phase_order": [
            *[phase for phase in PHASE_ORDER if phase in expanded],
            *sorted(set(expanded) - set(PHASE_ORDER)),
        ],
        "phases": expanded,
    }
    return {
        "status": "pass",
        "expanded": expanded,
        "canonical_expansion": canonical_expansion,
        "command_matrix_sha256": canonical_sha256(canonical_expansion),
        "phase_count": len(expanded),
        "command_count": sum(
            int(item["matrix_cell_count"]) for item in expanded.values()
        ),
    }


def validate_protocol_v1_1(protocol: Mapping[str, Any]) -> dict[str, Any]:
    _reject_non_finite(protocol)
    version = protocol.get("typed_model_cache_formal_protocol_version")
    if version == FORMAL_EXECUTION_PROTOCOL_V2_6_VERSION:
        if protocol.get("protocol_id") != FORMAL_EXECUTION_PROTOCOL_V2_6_ID:
            raise FormalExecutionError("formal execution protocol v2.6 ID mismatch")
        capabilities = require_live_execution_protocol(version)
        supersession = protocol.get("supersession", {})
        authorization = supersession.get("g14r15_authorization_boundary", {})
        if (
            supersession.get("supersedes_version") != "2.5.0"
            or supersession.get("old_protocol_status")
            != "historical_audit_only_after_cell_publication_contract_mismatch"
            or authorization.get("status")
            != "PRE-EXECUTION AUTHORIZATION WITHHELD / CELL_ARTIFACT_PUBLICATION_CONTRACT_MISMATCH"
            or authorization.get("g14c_v14_created") is not False
            or authorization.get("formal_training_count") != 0
            or authorization.get("formal_checkpoint_count") != 0
            or authorization.get("formal_performance_count") != 0
            or authorization.get("holdout_opened") is not False
        ):
            raise FormalExecutionError("Protocol v2.6 authorization boundary is incomplete")
        publication = protocol.get("formal_cell_artifact_publication_contract", {})
        generated = protocol.get(
            "formal_generated_checkpoint_resource_identity_contract", {}
        )
        if (
            publication.get("version") != "1.0.0"
            or not isinstance(publication.get("semantic_sha256"), str)
            or len(publication["semantic_sha256"]) != 64
            or generated.get("version") != "1.1.0"
            or not isinstance(generated.get("semantic_sha256"), str)
            or len(generated["semantic_sha256"]) != 64
            or not capabilities.generated_checkpoint_resource_required
            or not capabilities.cell_artifact_publication_required
            or capabilities.holdout_capability
        ):
            raise FormalExecutionError("Protocol v2.6 execution contracts are missing")
        profile = protocol.get("execution_contract", {}).get(
            "nonformal_rehearsal_profile", {}
        )
        if (
            not isinstance(profile.get("path"), str)
            or not isinstance(profile.get("semantic_sha256"), str)
            or profile.get("cannot_override_formal_profile") is not True
        ):
            raise FormalExecutionError("Protocol v2.6 rehearsal profile is missing")
        expected = canonical_sha256(
            semantic_projection(
                {key: value for key, value in protocol.items() if key != "hashes"}
            )
        )
        if protocol.get("hashes", {}).get("semantic_sha256") != expected:
            raise FormalExecutionError("formal protocol v2.6 semantic hash mismatch")
        return {
            "status": "pass",
            "protocol_version": version,
            "semantic_sha256": expected,
            "split_semantic_sha256": SPLIT_SEMANTIC_SHA256,
            "primary_endpoint_count": len(PRIMARY_ENDPOINTS),
            "phase_count": len(PHASE_ORDER),
        }
    if version == FORMAL_EXECUTION_PROTOCOL_V2_5_VERSION:
        if protocol.get("protocol_id") != FORMAL_EXECUTION_PROTOCOL_V2_5_ID:
            raise FormalExecutionError("formal execution protocol v2.5 ID mismatch")
        capabilities = get_protocol_capabilities(version)
        supersession = protocol.get("supersession", {})
        if (
            supersession.get("supersedes_version") != "2.4.0"
            or supersession.get("old_protocol_status")
            != "historical_audit_only_after_generated_resource_binding_gap"
        ):
            raise FormalExecutionError("Protocol v2.5 predecessor status is missing")
        authorization = supersession.get("g14r14_authorization_boundary", {})
        if (
            authorization.get("status")
            != "PRE-EXECUTION AUTHORIZATION WITHHELD / DOWNSTREAM_RESOURCE_BINDING_NOT_CLOSED"
            or authorization.get("g14c_v14_created") is not False
            or authorization.get("formal_training_count") != 0
            or authorization.get("formal_checkpoint_count") != 0
            or authorization.get("formal_performance_count") != 0
            or authorization.get("holdout_capability") is not False
        ):
            raise FormalExecutionError("G14R14 authorization boundary is incomplete")
        generated = protocol.get("formal_generated_checkpoint_resource_identity_contract", {})
        if (
            generated.get("version") != "1.0.0"
            or not isinstance(generated.get("semantic_sha256"), str)
            or len(generated["semantic_sha256"]) != 64
            or not capabilities.generated_checkpoint_resource_required
            or capabilities.holdout_capability
        ):
            raise FormalExecutionError("Protocol v2.5 generated resource contract is missing")
        templates = protocol.get("execution_contract", {}).get("command_templates", {})
        consumers = (
            "formal_cache_policy", "formal_controller", "formal_ablation",
            "formal_support", "formal_scalability", "formal_statistics", "formal_gate",
        )
        for phase in consumers:
            argv = templates.get(phase, {}).get("argv", [])
            if "--generated-checkpoint-registry-path" not in argv:
                raise FormalExecutionError(
                    f"Protocol v2.5 generated registry flag missing: {phase}"
                )
        capacity_rows = [
            row for row in templates.get("formal_support", {}).get("matrix_contexts", [])
            if str(row.get("support_setting_id", "")).startswith("capacity-")
        ]
        observed = {
            (
                row.get("capacity_label"), row.get("runtime_config_resource_id"),
                row.get("checkpoint_manifest_id"), row.get("checkpoint_provenance_id"),
            )
            for row in capacity_rows
        }
        expected_capacity = {
            (label, f"runtime_config.{label}", f"checkpoint_manifest.{label}",
             f"checkpoint_provenance.{label}")
            for label in ("constrained_288mb", "medium_576mb", "relaxed_864mb")
        }
        if observed != expected_capacity:
            raise FormalExecutionError("Protocol v2.5 capacity resource mapping drift")
        expected = canonical_sha256(
            semantic_projection(
                {key: value for key, value in protocol.items() if key != "hashes"}
            )
        )
        if protocol.get("hashes", {}).get("semantic_sha256") != expected:
            raise FormalExecutionError("formal protocol v2.5 semantic hash mismatch")
        return {
            "status": "pass",
            "protocol_version": version,
            "semantic_sha256": expected,
            "split_semantic_sha256": SPLIT_SEMANTIC_SHA256,
            "primary_endpoint_count": len(PRIMARY_ENDPOINTS),
            "phase_count": len(PHASE_ORDER),
        }
    if version == FORMAL_EXECUTION_PROTOCOL_V2_4_VERSION:
        if protocol.get("protocol_id") != FORMAL_EXECUTION_PROTOCOL_V2_4_ID:
            raise FormalExecutionError("formal execution protocol v2.4 ID mismatch")
        capabilities = get_protocol_capabilities(version)
        supersession = protocol.get("supersession", {})
        if (
            supersession.get("supersedes_version") != "2.3.0"
            or supersession.get("old_protocol_status")
            != "audit_only_after_pre_execution_validator_version_dispatch_mismatch"
        ):
            raise FormalExecutionError("Protocol v2.4 predecessor status is missing")
        stop = supersession.get("g14c_v13_pre_execution_stop", {})
        if (
            stop.get("classification")
            != "PRE_EXECUTION_STOP / VALIDATOR_VERSION_DISPATCH_MISMATCH"
            or stop.get("durable_run_root_created") is not False
            or stop.get("phase_or_cell_ledger_created") is not False
            or stop.get("formal_training_count") != 0
            or stop.get("formal_checkpoint_count") != 0
            or stop.get("formal_performance_count") != 0
        ):
            raise FormalExecutionError("G14C v13 pre-execution stop audit is incomplete")
        if any(
            isinstance(row, Mapping) and "g14c_v13" in str(row.get("run_id", ""))
            for row in supersession.get("invalid_execution_runs", [])
        ):
            raise FormalExecutionError("G14C v13 must not be registered as an invalid run")
        routing = protocol.get("formal_protocol_capability_routing_contract", {})
        routing_hash = routing.get("semantic_sha256")
        if (
            routing.get("version")
            != FORMAL_PROTOCOL_CAPABILITY_ROUTING_CONTRACT_VERSION
            or not isinstance(routing_hash, str)
            or len(routing_hash) != 64
            or protocol.get("identity", {}).get(
                "formal_protocol_capability_routing_contract_semantic_sha256"
            )
            != routing_hash
            or not capabilities.persisted_resolved_execution_context_required
            or not capabilities.nullable_metric_contract_required
            or capabilities.holdout_capability
        ):
            raise FormalExecutionError("Protocol v2.4 capability routing contract is missing")
        expected = canonical_sha256(
            semantic_projection(
                {key: value for key, value in protocol.items() if key != "hashes"}
            )
        )
        if protocol.get("hashes", {}).get("semantic_sha256") != expected:
            raise FormalExecutionError("formal protocol v2.4 semantic hash mismatch")
        inherited = deepcopy(dict(protocol))
        inherited["typed_model_cache_formal_protocol_version"] = (
            FORMAL_EXECUTION_PROTOCOL_V2_3_VERSION
        )
        inherited["protocol_id"] = FORMAL_EXECUTION_PROTOCOL_V2_3_ID
        inherited.pop("formal_protocol_capability_routing_contract", None)
        inherited.get("identity", {}).pop(
            "formal_protocol_capability_routing_contract_semantic_sha256", None
        )
        inherited["supersession"].pop("g14c_v13_pre_execution_stop", None)
        inherited["supersession"].update(
            supersedes_version="2.2.0",
            old_protocol_status="invalid_protocol_or_implementation",
        )
        inherited = attach_hashes(inherited)
        validate_protocol_v1_1(inherited)
        return {
            "status": "pass",
            "protocol_version": version,
            "semantic_sha256": expected,
            "split_semantic_sha256": SPLIT_SEMANTIC_SHA256,
            "primary_endpoint_count": len(PRIMARY_ENDPOINTS),
            "phase_count": len(PHASE_ORDER),
        }
    if version == FORMAL_EXECUTION_PROTOCOL_V2_3_VERSION:
        if protocol.get("protocol_id") != FORMAL_EXECUTION_PROTOCOL_V2_3_ID:
            raise FormalExecutionError("formal execution protocol v2.3 ID mismatch")
        supersession = protocol.get("supersession", {})
        if (
            supersession.get("supersedes_version") != "2.2.0"
            or supersession.get("old_protocol_status")
            != "invalid_protocol_or_implementation"
        ):
            raise FormalExecutionError("Protocol v2.3 predecessor status is missing")
        failures = supersession.get("invalid_execution_runs", [])
        v12 = next(
            (
                row
                for row in failures
                if isinstance(row, Mapping)
                and row.get("run_id")
                == "typed_model_cache_formal_20260902_162203_g14c_v12"
            ),
            None,
        )
        if not isinstance(v12, Mapping) or (
            v12.get("failure_boundary")
            != "invalid_during_first_training_cell_after_episode_generation_before_cell_commit"
            or v12.get("failure_audit_sha256")
            != "edb85d74152feefff37b1180d9bc5cb2d04cefa64c226eda831db22539cd39e5"
            or v12.get("training_cells_executed") != 0
            or v12.get("candidate_checkpoint_count") != 0
            or v12.get("dev_performance_count") != 0
            or v12.get("formal_performance_count") != 0
            or v12.get("resume_allowed") is not False
            or v12.get("retry_allowed") is not False
            or v12.get("legacy_phase_finalize_allowed") is not False
            or v12.get("checkpoint_reuse_allowed") is not False
        ):
            raise FormalExecutionError("G14C v12 permanent invalidation is incomplete")
        nullable = protocol.get("formal_nullable_metric_aggregation_contract", {})
        nullable_hash = nullable.get("semantic_sha256")
        if (
            nullable.get("version") != "1.0.0"
            or not isinstance(nullable_hash, str)
            or len(nullable_hash) != 64
            or protocol.get("identity", {}).get(
                "formal_nullable_metric_aggregation_contract_semantic_sha256"
            )
            != nullable_hash
        ):
            raise FormalExecutionError("nullable metric aggregation contract is missing")
        expected = canonical_sha256(
            semantic_projection(
                {key: value for key, value in protocol.items() if key != "hashes"}
            )
        )
        if protocol.get("hashes", {}).get("semantic_sha256") != expected:
            raise FormalExecutionError("formal protocol v2.3 semantic hash mismatch")
        inherited = deepcopy(dict(protocol))
        inherited["typed_model_cache_formal_protocol_version"] = (
            FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION
        )
        inherited["protocol_id"] = FORMAL_EXECUTION_PROTOCOL_V2_2_ID
        inherited["supersession"]["supersedes_version"] = "2.1.0"
        inherited["supersession"]["old_protocol_status"] = (
            "invalid_protocol_or_implementation"
        )
        inherited["supersession"]["invalid_execution_runs"] = [
            row
            for row in inherited["supersession"]["invalid_execution_runs"]
            if row.get("run_id")
            != "typed_model_cache_formal_20260902_162203_g14c_v12"
        ]
        inherited = attach_hashes(inherited)
        validate_protocol_v1_1(inherited)
        return {
            "status": "pass",
            "protocol_version": version,
            "semantic_sha256": expected,
            "split_semantic_sha256": SPLIT_SEMANTIC_SHA256,
            "primary_endpoint_count": len(PRIMARY_ENDPOINTS),
            "phase_count": len(PHASE_ORDER),
        }
    if version not in {
        FORMAL_EXECUTION_PROTOCOL_VERSION,
        FORMAL_EXECUTION_PROTOCOL_V1_2_VERSION,
        FORMAL_EXECUTION_PROTOCOL_V1_3_VERSION,
        FORMAL_EXECUTION_PROTOCOL_V1_4_VERSION,
        FORMAL_EXECUTION_PROTOCOL_V1_5_VERSION,
        FORMAL_EXECUTION_PROTOCOL_V1_6_VERSION,
        FORMAL_EXECUTION_PROTOCOL_V1_7_VERSION,
        FORMAL_EXECUTION_PROTOCOL_V1_8_VERSION,
        FORMAL_EXECUTION_PROTOCOL_V1_9_VERSION,
        FORMAL_EXECUTION_PROTOCOL_V2_0_VERSION,
        FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION,
        FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION,
    }:
        raise FormalExecutionError("unsupported formal execution protocol version")
    expected_protocol_id = {
        FORMAL_EXECUTION_PROTOCOL_VERSION: FORMAL_EXECUTION_PROTOCOL_ID,
        FORMAL_EXECUTION_PROTOCOL_V1_2_VERSION: FORMAL_EXECUTION_PROTOCOL_V1_2_ID,
        FORMAL_EXECUTION_PROTOCOL_V1_3_VERSION: FORMAL_EXECUTION_PROTOCOL_V1_3_ID,
        FORMAL_EXECUTION_PROTOCOL_V1_4_VERSION: FORMAL_EXECUTION_PROTOCOL_V1_4_ID,
        FORMAL_EXECUTION_PROTOCOL_V1_5_VERSION: FORMAL_EXECUTION_PROTOCOL_V1_5_ID,
        FORMAL_EXECUTION_PROTOCOL_V1_6_VERSION: FORMAL_EXECUTION_PROTOCOL_V1_6_ID,
        FORMAL_EXECUTION_PROTOCOL_V1_7_VERSION: FORMAL_EXECUTION_PROTOCOL_V1_7_ID,
        FORMAL_EXECUTION_PROTOCOL_V1_8_VERSION: FORMAL_EXECUTION_PROTOCOL_V1_8_ID,
        FORMAL_EXECUTION_PROTOCOL_V1_9_VERSION: FORMAL_EXECUTION_PROTOCOL_V1_9_ID,
        FORMAL_EXECUTION_PROTOCOL_V2_0_VERSION: FORMAL_EXECUTION_PROTOCOL_V2_0_ID,
        FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION: FORMAL_EXECUTION_PROTOCOL_V2_1_ID,
        FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION: FORMAL_EXECUTION_PROTOCOL_V2_2_ID,
    }[version]
    if protocol.get("protocol_id") != expected_protocol_id:
        raise FormalExecutionError("formal execution protocol ID mismatch")
    supersession = protocol.get("supersession", {})
    if version == FORMAL_EXECUTION_PROTOCOL_VERSION:
        if supersession.get("supersedes_version") != "1.0.0":
            raise FormalExecutionError("protocol v1.1 must supersede v1.0")
        if supersession.get("old_protocol_status") != "invalid_before_execution":
            raise FormalExecutionError("old protocol invalid status is missing")
        if supersession.get("old_protocol_semantic_sha256") != OLD_PROTOCOL_SEMANTIC_SHA256:
            raise FormalExecutionError("old protocol hash mismatch")
    elif version == FORMAL_EXECUTION_PROTOCOL_V1_2_VERSION:
        if supersession.get("supersedes_version") != "1.1.0":
            raise FormalExecutionError("protocol v1.2 must supersede v1.1")
        if supersession.get("old_protocol_status") != "invalid_before_performance_execution":
            raise FormalExecutionError("protocol v1.1 invalid status is missing")
        if not supersession.get("failure_audit_sha256"):
            raise FormalExecutionError("G14C v2 failure audit hash is missing")
        window_contract = protocol.get("execution_contract", {}).get(
            "window_consumption_contract", {}
        )
        if window_contract.get("version") != "1.0.0" or not window_contract.get(
            "semantic_sha256"
        ):
            raise FormalExecutionError("formal window consumption contract is missing")
        if window_contract.get("contract_identity") != window_contract.get(
            "semantic_sha256"
        ):
            raise FormalExecutionError("formal window consumption identity is not hash-bound")
        ledger = protocol.get("execution_contract", {}).get("phase_ledger", {})
        if ledger.get("schema_version") != FORMAL_PHASE_LEDGER_SCHEMA_VERSION:
            raise FormalExecutionError("phase ledger schema version mismatch")
        if tuple(ledger.get("failure_classifications", [])) != FAILURE_CLASSIFICATIONS:
            raise FormalExecutionError("phase failure classification enum mismatch")
    elif version == FORMAL_EXECUTION_PROTOCOL_V1_3_VERSION:
        if supersession.get("supersedes_version") != "1.2.0":
            raise FormalExecutionError("protocol v1.3 must supersede v1.2")
        if supersession.get("old_protocol_status") != "invalid_before_dev_performance_execution":
            raise FormalExecutionError("protocol v1.2 invalid dev status is missing")
        if supersession.get("failure_audit_sha256") != "476cfc3f57312263da7dff388a89c088e4716d43b1949eb121598c86dc5ac3af":
            raise FormalExecutionError("G14C v3 failure audit hash mismatch")
        portable = protocol.get("portable_resource_identity_contract", {})
        if portable.get("version") != "1.0.0" or portable.get("resource_resolver_version") != "1.0.0":
            raise FormalExecutionError("portable resource identity contract is missing")
        if portable.get("scientific_identity_rule") != "scientific identity != host absolute path":
            raise FormalExecutionError("portable scientific identity rule changed")
        if not portable.get("resource_registry_semantic_sha256"):
            raise FormalExecutionError("portable resource registry hash is missing")
        fairness = protocol.get("fairness_portability", {})
        if fairness.get("companion_version") != "1.0.0":
            raise FormalExecutionError("fairness portability companion is missing")
        checkpoint = protocol.get("checkpoint_location_contract", {})
        if checkpoint.get("version") != "1.0.0" or checkpoint.get("absolute_path_is_scientific_identity") is not False:
            raise FormalExecutionError("checkpoint location contract is missing")
        window_contract = protocol.get("execution_contract", {}).get(
            "window_consumption_contract", {}
        )
        if window_contract.get("semantic_sha256") != "ec475799b3fba4a3af3e4372e7c25781c6565a88ec814322b4cd4d447fef2771":
            raise FormalExecutionError("window consumption semantic hash changed")
        ledger = protocol.get("execution_contract", {}).get("phase_ledger", {})
        if ledger.get("schema_version") != FORMAL_PHASE_LEDGER_SCHEMA_VERSION:
            raise FormalExecutionError("phase ledger schema version mismatch")
    else:
        if version == FORMAL_EXECUTION_PROTOCOL_V1_4_VERSION:
            if supersession.get("supersedes_version") != "1.3.0":
                raise FormalExecutionError("protocol v1.4 must supersede v1.3")
            failures = supersession.get("invalid_g14c_v4_runs", [])
            expected_boundaries = {
                "typed_model_cache_formal_20260824_110016_g14c_v4": (
                    "invalid_after_training_before_dev_performance_execution",
                    "aaf5cfa717d543ffec5ea15dc5e4e8e7dac107dea51647cea10a9b1884118117",
                    150,
                    1200,
                ),
                "typed_model_cache_formal_20260824_235839_g14c_v4": (
                    "invalid_before_first_frozen_subcommand",
                    "bff76afccff2ea9485555a0bd20b33f5081e2ccaabebeff932f2ef74e8e6f42d",
                    0,
                    0,
                ),
            }
        elif version == FORMAL_EXECUTION_PROTOCOL_V1_5_VERSION:
            if supersession.get("supersedes_version") != "1.4.0":
                raise FormalExecutionError("protocol v1.5 must supersede v1.4")
            if supersession.get("old_protocol_status") != (
                "invalid_during_first_preflight_child_before_window_reachability"
            ):
                raise FormalExecutionError("G14C v5 invalid status is missing")
            failures = supersession.get("invalid_execution_runs", [])
            expected_boundaries = {
                "typed_model_cache_formal_20260820_g14c_351fdb8_v1": (
                    "invalid_before_execution",
                    "fd04ee5e25737d74ae9f58d0e076d4d620eb913dd55a8aef039a61510c71a0b1",
                    0,
                    0,
                ),
                "typed_model_cache_formal_20260820_164251_g14c_v2": (
                    "invalid_before_performance_execution",
                    "5da5e20395e5c1e48bf2e267ce757248d024246bdc121d4d2b33ca4f8c6c594b",
                    0,
                    0,
                ),
                "typed_model_cache_formal_20260820_203430_g14c_v3": (
                    "invalid_before_dev_performance_execution",
                    "476cfc3f57312263da7dff388a89c088e4716d43b1949eb121598c86dc5ac3af",
                    150,
                    1200,
                ),
                "typed_model_cache_formal_20260824_110016_g14c_v4": (
                    "invalid_after_training_before_dev_performance_execution",
                    "aaf5cfa717d543ffec5ea15dc5e4e8e7dac107dea51647cea10a9b1884118117",
                    150,
                    1200,
                ),
                "typed_model_cache_formal_20260824_235839_g14c_v4": (
                    "invalid_before_first_frozen_subcommand",
                    "bff76afccff2ea9485555a0bd20b33f5081e2ccaabebeff932f2ef74e8e6f42d",
                    0,
                    0,
                ),
                "typed_model_cache_formal_20260825_111625_g14c_v5": (
                    "invalid_during_first_preflight_child_before_window_reachability",
                    "3c0de5bfebb5877e1b5a53f42fea1e07504f4355bd1636ad17ed38145439ff93",
                    0,
                    0,
                ),
            }
        elif version == FORMAL_EXECUTION_PROTOCOL_V1_6_VERSION:
            if supersession.get("supersedes_version") != "1.5.0":
                raise FormalExecutionError("protocol v1.6 must supersede v1.5")
            if supersession.get("old_protocol_status") != (
                "invalid_during_first_training_cell_before_episode_zero"
            ):
                raise FormalExecutionError("G14C v6 invalid status is missing")
            failures = supersession.get("invalid_execution_runs", [])
            expected_boundaries = {
                "typed_model_cache_formal_20260820_g14c_351fdb8_v1": (
                    "invalid_before_execution", "fd04ee5e25737d74ae9f58d0e076d4d620eb913dd55a8aef039a61510c71a0b1", 0, 0,
                ),
                "typed_model_cache_formal_20260820_164251_g14c_v2": (
                    "invalid_before_performance_execution", "5da5e20395e5c1e48bf2e267ce757248d024246bdc121d4d2b33ca4f8c6c594b", 0, 0,
                ),
                "typed_model_cache_formal_20260820_203430_g14c_v3": (
                    "invalid_before_dev_performance_execution", "476cfc3f57312263da7dff388a89c088e4716d43b1949eb121598c86dc5ac3af", 150, 1200,
                ),
                "typed_model_cache_formal_20260824_110016_g14c_v4": (
                    "invalid_after_training_before_dev_performance_execution", "aaf5cfa717d543ffec5ea15dc5e4e8e7dac107dea51647cea10a9b1884118117", 150, 1200,
                ),
                "typed_model_cache_formal_20260824_235839_g14c_v4": (
                    "invalid_before_first_frozen_subcommand", "bff76afccff2ea9485555a0bd20b33f5081e2ccaabebeff932f2ef74e8e6f42d", 0, 0,
                ),
                "typed_model_cache_formal_20260825_111625_g14c_v5": (
                    "invalid_during_first_preflight_child_before_window_reachability", "3c0de5bfebb5877e1b5a53f42fea1e07504f4355bd1636ad17ed38145439ff93", 0, 0,
                ),
                "typed_model_cache_formal_20260825_135122_g14c_v6": (
                    "invalid_during_first_training_cell_before_episode_zero", "2cc81ffcd375323caa71c0966ffce36059c43a8da0aad5e7245078727dd0725a", 0, 0,
                ),
            }
        else:
            expected_supersedes = {
                FORMAL_EXECUTION_PROTOCOL_V1_7_VERSION: "1.6.0",
                FORMAL_EXECUTION_PROTOCOL_V1_8_VERSION: "1.7.0",
                FORMAL_EXECUTION_PROTOCOL_V1_9_VERSION: "1.8.0",
                FORMAL_EXECUTION_PROTOCOL_V2_0_VERSION: "1.9.0",
                FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION: "2.0.0",
                FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION: "2.1.0",
            }[version]
            expected_old_status = {
                FORMAL_EXECUTION_PROTOCOL_V1_7_VERSION: "invalid_after_training_before_dev_performance_execution",
                FORMAL_EXECUTION_PROTOCOL_V1_8_VERSION: "audit_only_active_index_readiness_inconsistent",
                FORMAL_EXECUTION_PROTOCOL_V1_9_VERSION: "invalid_after_training_before_dev_performance_execution",
                FORMAL_EXECUTION_PROTOCOL_V2_0_VERSION: (
                    "invalid_after_training_during_first_dev_candidate_evaluation_before_dev_selection"
                ),
                FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION: (
                    "audit_only_after_pre_execution_identity_mismatch"
                ),
                FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION: "invalid_protocol_or_implementation",
            }[version]
            if supersession.get("supersedes_version") != expected_supersedes:
                raise FormalExecutionError(
                    f"protocol {version} must supersede {expected_supersedes}"
                )
            if supersession.get("old_protocol_status") != expected_old_status:
                raise FormalExecutionError("active predecessor status is missing")
            failures = supersession.get("invalid_execution_runs", [])
            expected_boundaries = {
                "typed_model_cache_formal_20260820_g14c_351fdb8_v1": (
                    "invalid_before_execution", "fd04ee5e25737d74ae9f58d0e076d4d620eb913dd55a8aef039a61510c71a0b1", 0, 0,
                ),
                "typed_model_cache_formal_20260820_164251_g14c_v2": (
                    "invalid_before_performance_execution", "5da5e20395e5c1e48bf2e267ce757248d024246bdc121d4d2b33ca4f8c6c594b", 0, 0,
                ),
                "typed_model_cache_formal_20260820_203430_g14c_v3": (
                    "invalid_before_dev_performance_execution", "476cfc3f57312263da7dff388a89c088e4716d43b1949eb121598c86dc5ac3af", 150, 1200,
                ),
                "typed_model_cache_formal_20260824_110016_g14c_v4": (
                    "invalid_after_training_before_dev_performance_execution", "aaf5cfa717d543ffec5ea15dc5e4e8e7dac107dea51647cea10a9b1884118117", 150, 1200,
                ),
                "typed_model_cache_formal_20260824_235839_g14c_v4": (
                    "invalid_before_first_frozen_subcommand", "bff76afccff2ea9485555a0bd20b33f5081e2ccaabebeff932f2ef74e8e6f42d", 0, 0,
                ),
                "typed_model_cache_formal_20260825_111625_g14c_v5": (
                    "invalid_during_first_preflight_child_before_window_reachability", "3c0de5bfebb5877e1b5a53f42fea1e07504f4355bd1636ad17ed38145439ff93", 0, 0,
                ),
                "typed_model_cache_formal_20260825_135122_g14c_v6": (
                    "invalid_during_first_training_cell_before_episode_zero", "2cc81ffcd375323caa71c0966ffce36059c43a8da0aad5e7245078727dd0725a", 0, 0,
                ),
                "typed_model_cache_formal_20260826_233222_g14c_v7": (
                    "invalid_after_training_before_dev_performance_execution", "7fc3685470c1f536def5c504dfbeab83b14dd070a644caefed08e690e10247ba", 150, 1200,
                ),
            }
            if version in {
                FORMAL_EXECUTION_PROTOCOL_V1_9_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V2_0_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION,
            }:
                expected_boundaries["typed_model_cache_formal_20260828_101804_g14c_v8"] = (
                    "invalid_after_training_before_dev_performance_execution",
                    "2c09cd14028051a012ddedf756bd6b186b4d1680582c5944acc0da986aa40ba5",
                    150,
                    1200,
                )
            if version in {
                FORMAL_EXECUTION_PROTOCOL_V2_0_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION,
            }:
                expected_boundaries[
                    "typed_model_cache_formal_20260830_113339_g14c_v9"
                ] = (
                    "invalid_after_training_during_first_dev_candidate_evaluation_before_dev_selection",
                    "ec6b04fee48c4abda056b62f508f186345f10ab580efd896f9f43979d1d728fe",
                    150,
                    1200,
                )
            if version == FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION:
                expected_boundaries[
                    "typed_model_cache_formal_20260901_155201_g14c_v11"
                ] = (
                    "invalid_during_first_training_cell_before_first_episode_commit",
                    "b5eb0063c0cde2670d298027a8aeea1b4661b77fc404b72f673970642662e362",
                    0,
                    0,
                )
        expected_failures = {
            (run_id, values[1]) for run_id, values in expected_boundaries.items()
        }
        observed_failures = {
            (
                str(item.get("run_id")),
                str(
                    item.get("phase_ledger_sha256")
                    if item.get("run_id")
                    == "typed_model_cache_formal_20260830_113339_g14c_v9"
                    else item.get("failure_audit_sha256")
                ),
            )
            for item in failures
            if isinstance(item, Mapping)
        }
        if observed_failures != expected_failures:
            raise FormalExecutionError("invalid formal run references are incomplete")
        failures_by_run = {
            str(item["run_id"]): item for item in failures if isinstance(item, Mapping)
        }
        for run_id, (
            boundary,
            _failure_hash,
            training_cells,
            candidates,
        ) in expected_boundaries.items():
            reference = failures_by_run[run_id]
            if (
                reference.get("failure_boundary") != boundary
                or reference.get("training_cells_executed") != training_cells
                or reference.get("candidate_checkpoint_count") != candidates
                or reference.get("dev_performance_count") != 0
                or reference.get("formal_performance_count") != 0
                or reference.get("resume_allowed") is not False
                or reference.get("checkpoint_reuse_allowed") is not False
                or reference.get("legacy_phase_finalize_allowed") is not False
            ):
                raise FormalExecutionError("invalid formal run boundary or reuse rule changed")
        environment = protocol.get("formal_execution_environment_contract", {})
        expected_environment_version = {
            FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION: "1.1.0",
            FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION: "1.2.0",
        }.get(version, "1.0.0")
        if environment.get("version") != expected_environment_version:
            raise FormalExecutionError("formal execution environment contract is missing")
        expected_identity_rule = (
            "full_projection_v1_excludes_host_paths_and_binds_observed_commit_and_tree_out_of_band"
            if version in {
                FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION,
            }
            else "environment identity != host-specific Python absolute path"
        )
        if environment.get("scientific_identity", {}).get("identity_rule") != expected_identity_rule:
            raise FormalExecutionError("formal environment identity/path rule changed")
        if not environment.get("scientific_identity", {}).get(
            "environment_fingerprint"
        ):
            raise FormalExecutionError("formal environment fingerprint is missing")
        resolver = environment.get("resolver", {})
        expected_priority = (
            ["explicit_python_executable"]
            if version in {
                FORMAL_EXECUTION_PROTOCOL_V1_5_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V1_6_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V1_7_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V1_8_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V1_9_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V2_0_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION,
            }
            else [
                "explicit_python_executable",
                "execution_environment_manifest",
                "current_runner_sys_executable",
                "protocol_allowed_candidate",
            ]
        )
        expected_resolver_version = {
            FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION: "1.1.0",
            FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION: "1.2.0",
        }.get(version, "1.0.0")
        if resolver.get("version") != expected_resolver_version or resolver.get("priority") != expected_priority:
            raise FormalExecutionError("formal Python resolver priority changed")
        execution = protocol.get("execution_contract", {})
        ledger = execution.get("phase_ledger", {})
        if (
            ledger.get("schema_version") != "3.0.0"
            or ledger.get("duration_authority") != "monotonic_clock"
            or ledger.get("absolute_sanity_seconds") != 259200
            or ledger.get("clock_adjustment_is_audited_not_terminal") is not True
        ):
            raise FormalExecutionError("phase ledger v3 contract is missing")
        if execution.get("cell_ledger", {}).get("schema_version") != "1.0.0":
            raise FormalExecutionError("cell ledger v1 contract is missing")
        if execution.get("phase_completion_transaction", {}).get("version") != "1.0.0":
            raise FormalExecutionError("phase completion transaction is missing")
        resume = execution.get("same_run_resume", {})
        if resume.get("version") != "1.0.0" or resume.get("cross_run_import_allowed") is not False:
            raise FormalExecutionError("same-run resume contract is missing")
        templates = execution.get("command_templates", {})
        cell_phases = set(execution.get("cell_ledger", {}).get("phases", []))
        for phase, spec in templates.items():
            argv = spec.get("argv", []) if isinstance(spec, Mapping) else []
            if not argv or argv[0] != "{python_executable}":
                raise FormalExecutionError(
                    f"formal command lacks Python placeholder: {phase}"
                )
            if any(".venv/bin/python" in str(token) for token in argv):
                raise FormalExecutionError("formal command hard-codes .venv Python")
            if phase in cell_phases and spec.get("cell_transaction") is not True:
                raise FormalExecutionError(f"formal cell transaction missing: {phase}")
        if version in {
            FORMAL_EXECUTION_PROTOCOL_V1_5_VERSION,
            FORMAL_EXECUTION_PROTOCOL_V1_6_VERSION,
            FORMAL_EXECUTION_PROTOCOL_V1_7_VERSION,
            FORMAL_EXECUTION_PROTOCOL_V1_8_VERSION,
            FORMAL_EXECUTION_PROTOCOL_V1_9_VERSION,
            FORMAL_EXECUTION_PROTOCOL_V2_0_VERSION,
            FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION,
            FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION,
        }:
            resolved = protocol.get("resolved_formal_execution_context_contract", {})
            if (
                resolved.get("version") != (
                    "2.0.0"
                    if version in {
                        FORMAL_EXECUTION_PROTOCOL_V1_6_VERSION,
                        FORMAL_EXECUTION_PROTOCOL_V1_7_VERSION,
                        FORMAL_EXECUTION_PROTOCOL_V1_8_VERSION,
                        FORMAL_EXECUTION_PROTOCOL_V1_9_VERSION,
                        FORMAL_EXECUTION_PROTOCOL_V2_0_VERSION,
                        FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION,
                        FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION,
                    }
                    else "1.0.0"
                )
                or resolved.get("outer_runner_is_unique_producer") is not True
                or resolved.get("atomic_create_only") is not True
                or resolved.get("context_in_phase_input_hash") is not True
                or resolved.get("resume_revalidates_context_hash") is not True
                or resolved.get("implicit_runtime_fallback_allowed") is not False
            ):
                raise FormalExecutionError(
                    "resolved formal execution context contract is incomplete"
                )
            resume_bindings = set(resume.get("bindings", []))
            if "resolved_execution_context_sha256" not in resume_bindings:
                raise FormalExecutionError(
                    "same-run resume does not bind resolved execution context"
                )
            required_context_consumers = {
                "preflight",
                "dev_select",
                "formal_ablation",
                "formal_support",
                "formal_scalability",
                "formal_statistics",
            }
            for phase in required_context_consumers:
                argv = templates[phase]["argv"]
                if "--resolved-execution-context-path" not in argv:
                    raise FormalExecutionError(
                        f"nested formal consumer lacks resolved context: {phase}"
                    )
            if version in {
                FORMAL_EXECUTION_PROTOCOL_V1_6_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V1_7_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V1_8_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V1_9_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V2_0_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION,
                FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION,
            }:
                scientific = protocol.get(
                    "agent_training_scientific_config_contract", {}
                )
                binding = protocol.get("formal_training_execution_binding_contract", {})
                if (
                    scientific.get("version") != "2.0.0"
                    or not scientific.get("config_semantic_sha256")
                    or binding.get("version") != "1.0.0"
                    or binding.get("protocol_binds_schema_not_runtime_instance") is not True
                    or binding.get("runtime_instance_created_after_protocol_hash") is not True
                    or binding.get("binding_hash_enters_resolved_context") is not True
                ):
                    raise FormalExecutionError(
                        "formal scientific config/execution binding contract is incomplete"
                    )
                train_argv = templates.get("train", {}).get("argv", [])
                for flag in (
                    "--agent_scientific_config_path",
                    "--formal_training_execution_binding_path",
                    "--resolved_execution_context_path",
                ):
                    if flag not in train_argv:
                        raise FormalExecutionError(
                            f"Protocol v1.6 train command lacks {flag}"
                        )
                if "--agent_config_path" in train_argv:
                    raise FormalExecutionError(
                        "Protocol v1.6 train command retains legacy companion"
                    )
                if version in {
                    FORMAL_EXECUTION_PROTOCOL_V1_7_VERSION,
                    FORMAL_EXECUTION_PROTOCOL_V1_8_VERSION,
                    FORMAL_EXECUTION_PROTOCOL_V1_9_VERSION,
                    FORMAL_EXECUTION_PROTOCOL_V2_0_VERSION,
                    FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION,
                    FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION,
                }:
                    from src.runtime.formal_agent_order import (
                        FormalAgentOrderError,
                        resolve_formal_agent_order,
                    )

                    try:
                        order_audit = resolve_formal_agent_order(protocol=protocol)
                    except FormalAgentOrderError as exc:
                        raise FormalExecutionError(str(exc)) from exc
                    if protocol.get("identity", {}).get(
                        "formal_agent_order_contract_semantic_sha256"
                    ) != order_audit["semantic_sha256"]:
                        raise FormalExecutionError("Protocol identity/order contract hash mismatch")
                    for phase in (
                        "dev_select",
                        "formal_cache_policy",
                        "formal_controller",
                        "formal_statistics",
                    ):
                        if "--formal-agent-order-contract-path" not in templates[phase]["argv"]:
                            raise FormalExecutionError(
                                f"Protocol v1.7+ {phase} lacks formal agent order contract"
                            )
                if version in {
                    FORMAL_EXECUTION_PROTOCOL_V1_9_VERSION,
                    FORMAL_EXECUTION_PROTOCOL_V2_0_VERSION,
                    FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION,
                    FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION,
                }:
                    resource_contract = protocol.get(
                        "active_bundle_resource_resolution_contract", {}
                    )
                    if (
                        resource_contract.get("version") != "1.0.0"
                        or resource_contract.get("resource_catalog")
                        != "active_bundle_resources"
                        or resource_contract.get("validated_bundle_required") is not True
                        or resource_contract.get("raw_index_layout_is_consumer_api") is not False
                    ):
                        raise FormalExecutionError(
                            "active bundle resource resolution contract is incomplete"
                        )
    if protocol.get("identity", {}).get("split_semantic_sha256") != SPLIT_SEMANTIC_SHA256:
        raise FormalExecutionError("split semantic hash changed")
    if tuple(protocol.get("endpoints", {}).get("primary", [])) != PRIMARY_ENDPOINTS:
        raise FormalExecutionError("primary endpoint order or identity mismatch")
    expected_endpoint_schema_version = (
        PRIMARY_ENDPOINT_SCHEMA_VERSION
        if version in {
            FORMAL_EXECUTION_PROTOCOL_V2_0_VERSION,
            FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION,
            FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION,
        }
        else LEGACY_PRIMARY_ENDPOINT_SCHEMA_VERSION
    )
    if protocol.get("endpoint_schema", {}).get("primary_endpoint_schema_version") != expected_endpoint_schema_version:
        raise FormalExecutionError("primary endpoint schema missing")
    if version in {
        FORMAL_EXECUTION_PROTOCOL_V2_0_VERSION,
        FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION,
        FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION,
    }:
        request_contract = protocol.get("formal_exogenous_request_execution_contract", {})
        expected_request_contract = (
            ("1.1.0", "2.0.0", "1.0.0")
            if version == FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION
            else ("1.0.0", "1.0.0", None)
        )
        if (
            request_contract.get("version") != expected_request_contract[0]
            or request_contract.get("request_exposure_trace_version")
            != expected_request_contract[1]
            or request_contract.get("request_subject_lifecycle_contract_version")
            != expected_request_contract[2]
            or request_contract.get("default_enabled") is not False
            or request_contract.get("formal_explicit_enable_required") is not True
        ):
            raise FormalExecutionError(
                "Protocol v2.0 formal exogenous request execution contract is incomplete"
            )
    budget = protocol.get("training_budget", {})
    if budget.get("checkpoint_frequency_updates") != 4:
        raise FormalExecutionError("checkpoint frequency must equal four updates")
    agent_configs = budget.get("agent_configs", {})
    if agent_configs.get("sa_ghmappo", {}).get("auxiliary_coef") != 0.06:
        raise FormalExecutionError("SA auxiliary coefficient must equal 0.06")
    if any(
        "auxiliary_coef" in config
        for agent, config in agent_configs.items()
        if agent != "sa_ghmappo"
    ):
        raise FormalExecutionError("SA-only auxiliary coefficient leaked to another agent")
    if protocol.get("execution_contract", {}).get("phase_order") != list(PHASE_ORDER):
        raise FormalExecutionError("formal phase order mismatch")
    holdout = protocol.get("holdout_execution_contract", {})
    if holdout.get("sealed") is not True or holdout.get("opened") is not False:
        raise FormalExecutionError("holdout must remain sealed and unopened")
    if holdout.get("consumed_permanently") is not False:
        raise FormalExecutionError("holdout consumed state changed")
    expected = canonical_sha256(
        semantic_projection({key: value for key, value in protocol.items() if key != "hashes"})
    )
    observed = protocol.get("hashes", {}).get("semantic_sha256")
    if observed != expected:
        raise FormalExecutionError("formal protocol semantic hash mismatch")
    return {
        "status": "pass",
        "protocol_version": version,
        "semantic_sha256": observed,
        "split_semantic_sha256": SPLIT_SEMANTIC_SHA256,
        "primary_endpoint_count": len(PRIMARY_ENDPOINTS),
        "phase_count": len(PHASE_ORDER),
    }


def reconcile_primary_endpoint_row(
    summary: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    from src.metrics.cache_efficiency_metrics import cache_efficiency_row_fields

    recomputed = cache_efficiency_row_fields(summary)
    mismatches = [
        field
        for field in PRIMARY_ENDPOINTS[:4]
        if row.get(field) != recomputed.get(field)
    ]
    for field in PRIMARY_ENDPOINTS[4:]:
        expected = summary.get("system_metrics", {}).get(field)
        if row.get(field) != expected:
            mismatches.append(field)
    if mismatches:
        raise FormalExecutionError(
            f"row/raw reducer primary endpoint mismatch: {sorted(mismatches)}"
        )
    return {"status": "pass", "fields": list(PRIMARY_ENDPOINTS)}


def readiness_v3(checks: Mapping[str, bool]) -> str:
    required = {
        "protocol_fields_have_runtime_consumers",
        "agent_commands_expand",
        "checkpoint_frequency_consistent",
        "sa_auxiliary_consistent",
        "primary_endpoint_producer_exists",
        "primary_endpoint_reconciliation",
        "support_values_concrete_or_unavailable",
        "typed_support_provenance",
        "phase_runner_dry_run",
        "fairness_manifests_persisted",
        "runtime_configs_persisted",
        "command_templates_persisted",
        "output_schema_exists",
        "clean_worktree_execution_plan",
        "holdout_sealed",
    }
    if set(checks) != required:
        missing = required - set(checks)
        extra = set(checks) - required
        raise FormalExecutionError(
            f"readiness v3 check set mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return READY_VERDICT if all(checks.values()) else "BLOCKED_G14R_READINESS_V3"


def readiness_v4(checks: Mapping[str, bool]) -> str:
    required = {
        "window_reachability_60_of_60",
        "frame_time_fingerprint_identity",
        "training_commands_150_of_150",
        "formal_commands_resolved",
        "support_commands_resolved_or_unavailable",
        "no_implicit_mobility_row_default",
        "window_consumption_contract",
        "ledger_schema_complete",
        "ledger_append_chain",
        "failure_classification",
        "rehearsal",
        "holdout_sealed",
        "no_formal_training_or_results",
    }
    if set(checks) != required:
        raise FormalExecutionError(
            "readiness v4 check set mismatch: "
            f"missing={sorted(required - set(checks))}, "
            f"extra={sorted(set(checks) - required)}"
        )
    return (
        READY_V4_VERDICT
        if all(checks.values())
        else "BLOCKED_G14R2_READINESS_V4"
    )


def readiness_v5(checks: Mapping[str, bool]) -> str:
    required = {
        "external_resource_matrix_complete",
        "all_resources_content_addressed",
        "no_cwd_path_guessing",
        "main_clean_scientific_identity_parity",
        "training_commands_150_of_150",
        "dev_selector_complete",
        "checkpoint_freeze_complete",
        "formal_support_resolution_complete",
        "exact_non_formal_phase_chain_complete",
        "invalid_g14c_v3_checkpoints_not_reused",
        "holdout_sealed",
        "no_formal_performance_results",
    }
    if set(checks) != required:
        raise FormalExecutionError(
            "readiness v5 check set mismatch: "
            f"missing={sorted(required - set(checks))}, "
            f"extra={sorted(set(checks) - required)}"
        )
    return READY_V5_VERDICT if all(checks.values()) else "BLOCKED_G14R3_READINESS_V5"


def readiness_v6(checks: Mapping[str, bool]) -> str:
    required = {
        "two_v4_failures_registered",
        "clean_worktree_without_local_venv",
        "all_commands_use_resolved_interpreter",
        "clean_import_origin",
        "environment_fingerprint",
        "long_phase_and_clock_jump",
        "phase_transaction_and_finalize_only",
        "cell_ledger_and_atomic_commit",
        "same_run_resume",
        "interruption_75_of_150",
        "dev_formal_committed_only",
        "old_runs_hard_rejected",
        "holdout_sealed",
        "no_formal_performance_results",
    }
    if set(checks) != required:
        raise FormalExecutionError(
            "readiness v6 check set mismatch: "
            f"missing={sorted(required - set(checks))}, "
            f"extra={sorted(set(checks) - required)}"
        )
    return READY_V6_VERDICT if all(checks.values()) else "BLOCKED_G14R4_READINESS_V6"


def readiness_v7(checks: Mapping[str, bool]) -> str:
    required = {
        "g14c_v5_failure_registered",
        "producer_consumer_matrix_complete",
        "resolved_context_contract_frozen",
        "outer_nested_expansion_equal",
        "context_negative_cases_pass",
        "legacy_invalid_runs_hard_rejected",
        "clean_worktree_without_local_venv",
        "clean_import_origin",
        "window_reachability_60_of_60",
        "real_preflight_completed",
        "real_tests_phase_completed",
        "phase_and_cell_transactions_regression",
        "portable_fairness_checkpoint_regression",
        "full_pytest_and_smoke_pass",
        "holdout_sealed",
        "no_formal_training_or_performance",
    }
    if set(checks) != required:
        raise FormalExecutionError(
            "readiness v7 check set mismatch: "
            f"missing={sorted(required - set(checks))}, "
            f"extra={sorted(set(checks) - required)}"
        )
    return READY_V7_VERDICT if all(checks.values()) else "BLOCKED_G14R5_READINESS_V7"


def readiness_v8(checks: Mapping[str, bool]) -> str:
    required = {
        "g14c_v6_failure_registered",
        "producer_consumer_matrix_complete",
        "scientific_config_contract_frozen",
        "execution_binding_contract_frozen",
        "ten_agent_config_parity",
        "training_commands_150_bound",
        "ten_agent_entrypoint_rehearsal",
        "negative_validation_complete",
        "checkpoint_provenance_consumers_bound",
        "outer_nested_expansion_equal",
        "clean_worktree_without_local_venv",
        "clean_import_origin",
        "window_reachability_60_of_60",
        "real_preflight_completed",
        "real_tests_phase_completed",
        "phase_cell_resume_finalize_regression",
        "full_pytest_and_smoke_pass",
        "holdout_sealed",
        "no_formal_training_checkpoint_or_performance",
    }
    if set(checks) != required:
        raise FormalExecutionError(
            "readiness v8 check set mismatch: "
            f"missing={sorted(required - set(checks))}, "
            f"extra={sorted(set(checks) - required)}"
        )
    return READY_V8_VERDICT if all(checks.values()) else "BLOCKED_G14R6_READINESS_V8"


def readiness_v9(checks: Mapping[str, bool]) -> str:
    required = {
        "g14c_v7_failure_registered",
        "formal_agent_order_contract_frozen",
        "producer_consumer_matrix_complete",
        "scientific_config_hash_unchanged",
        "protocol_fairness_command_order_reconciled",
        "training_commands_150_order_audited",
        "all_dev_commands_order_audited",
        "formal_support_scalability_order_audited",
        "full_15_agent_nonformal_rehearsal",
        "checkpoint_selection_freeze_order_stable",
        "statistics_order_invariant",
        "negative_validation_complete",
        "binding_context_order_hash_bound",
        "outer_nested_expansion_equal",
        "clean_worktree_without_local_venv",
        "clean_import_origin",
        "window_reachability_60_of_60",
        "real_preflight_completed",
        "real_tests_phase_completed",
        "phase_cell_resume_finalize_regression",
        "full_pytest_and_smoke_pass",
        "holdout_sealed",
        "no_formal_training_checkpoint_or_performance",
    }
    if set(checks) != required:
        raise FormalExecutionError(
            "readiness v9 check set mismatch: "
            f"missing={sorted(required - set(checks))}, "
            f"extra={sorted(set(checks) - required)}"
        )
    return READY_V9_VERDICT if all(checks.values()) else "BLOCKED_G14R7_READINESS_V9"


def protocol_hash_changes_on_mutation(
    protocol: Mapping[str, Any], dotted_path: str, value: Any
) -> bool:
    mutated = deepcopy(dict(protocol))
    target: Any = mutated
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    if isinstance(target, list):
        target[int(parts[-1])] = value
    else:
        target[parts[-1]] = value
    return canonical_sha256(semantic_projection(protocol)) != canonical_sha256(
        semantic_projection(mutated)
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


def _parse_ledger_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise FormalExecutionError(f"phase ledger {field} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise FormalExecutionError(f"invalid phase ledger timestamp: {field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FormalExecutionError(f"phase ledger timestamp lacks timezone: {field}")
    return parsed


def _ledger_record_hash(record: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in record.items() if key != "current_record_hash"}
    )


def validate_phase_ledger(
    records: Sequence[Mapping[str, Any]],
    *,
    wall_clock_tolerance_seconds: float = LEDGER_WALL_CLOCK_TOLERANCE_SECONDS,
) -> dict[str, Any]:
    required = {
        "phase_ledger_schema_version",
        "phase",
        "sequence_number",
        "status",
        "started_at",
        "completed_at",
        "wall_clock_seconds",
        "input_hash",
        "output_hash",
        "expanded_command",
        "command_identity",
        "return_code",
        "retry_count",
        "failure_classification",
        "failure_message_reference",
        "previous_record_hash",
        "current_record_hash",
    }
    previous_hash: str | None = None
    terminal_by_phase: dict[str, str] = {}
    for index, raw_record in enumerate(records, start=1):
        record = dict(raw_record)
        missing = required.difference(record)
        if missing:
            raise FormalExecutionError(
                f"phase ledger record {index} missing fields: {sorted(missing)}"
            )
        if record["phase_ledger_schema_version"] != FORMAL_PHASE_LEDGER_SCHEMA_VERSION:
            raise FormalExecutionError("phase ledger schema version mismatch")
        if int(record["sequence_number"]) != index:
            raise FormalExecutionError("phase ledger sequence number mismatch")
        if record["previous_record_hash"] != previous_hash:
            raise FormalExecutionError("phase ledger previous hash mismatch")
        expected_hash = _ledger_record_hash(record)
        if record["current_record_hash"] != expected_hash:
            raise FormalExecutionError("phase ledger current hash mismatch")
        previous_hash = expected_hash
        phase = str(record["phase"])
        status = str(record["status"])
        if phase not in PHASE_ORDER or status not in {"running", *TERMINAL_LEDGER_STATUSES}:
            raise FormalExecutionError("invalid phase ledger phase/status")
        if phase in terminal_by_phase:
            raise FormalExecutionError("terminal phase ledger record is immutable")
        started = _parse_ledger_timestamp(record["started_at"], "started_at")
        classification = record["failure_classification"]
        return_code = record["return_code"]
        if status == "running":
            if record["completed_at"] is not None or record["wall_clock_seconds"] is not None:
                raise FormalExecutionError("running phase record cannot have completion time")
            if return_code is not None or classification is not None:
                raise FormalExecutionError("running phase record cannot have terminal outcome")
        else:
            completed = _parse_ledger_timestamp(record["completed_at"], "completed_at")
            if completed < started:
                raise FormalExecutionError("phase ledger system time moved backwards")
            wall_clock = float(record["wall_clock_seconds"])
            if not math.isfinite(wall_clock) or wall_clock < 0:
                raise FormalExecutionError("invalid phase ledger wall clock")
            timestamp_delta = (completed - started).total_seconds()
            if abs(timestamp_delta - wall_clock) > float(wall_clock_tolerance_seconds):
                raise FormalExecutionError("phase ledger wall clock/timestamp mismatch")
            terminal_by_phase[phase] = status
            if status == "completed":
                if return_code != 0 or classification is not None:
                    raise FormalExecutionError("completed phase ledger outcome is invalid")
            else:
                if classification not in FAILURE_CLASSIFICATIONS:
                    raise FormalExecutionError("missing or invalid failure classification")
                if return_code == 75 and classification != "infrastructure_retryable":
                    raise FormalExecutionError("return code 75 must be infrastructure_retryable")
                if classification == "infrastructure_retryable" and return_code != 75:
                    raise FormalExecutionError("infrastructure_retryable requires return code 75")
        if int(record["retry_count"]) < 0:
            raise FormalExecutionError("phase ledger retry count is invalid")
    return {
        "status": "pass",
        "record_count": len(records),
        "terminal_phase_count": len(terminal_by_phase),
        "last_record_hash": previous_hash,
    }


def classify_phase_failure(
    *, phase: str, return_code: int | None, message: str
) -> str:
    lowered = str(message or "").lower()
    if return_code == 75:
        return "infrastructure_retryable"
    if any(
        token in lowered
        for token in (
            "frame_offset",
            "window unreachable",
            "window_unreachable",
            "data_window_unreachable",
            "fingerprint mismatch",
            "source range",
        )
    ):
        return "data_window_unreachable"
    if any(token in lowered for token in ("protocol mismatch", "contract mismatch", "override rejected")):
        return "protocol_mismatch"
    if phase == "tests":
        return "test_failure"
    if phase == "train":
        return "training_failure"
    if phase in {"formal_gate", "checkpoint_freeze"}:
        return "artifact_integrity_failure"
    if return_code in {130, -2}:
        return "user_interruption"
    if return_code is None:
        return "implementation_error"
    return "infrastructure_terminal" if return_code >= 64 else "implementation_error"


class AppendOnlyPhaseRunner:
    """Hash-chained phase ledger with immutable terminal records."""

    def __init__(
        self,
        *,
        protocol: Mapping[str, Any],
        output_root: str | Path,
        resume: bool = False,
        clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        validate_protocol_v1_1(protocol)
        self.protocol = dict(protocol)
        self.output_root = Path(output_root)
        self.ledger_path = self.output_root / "phase_state.jsonl"
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._monotonic_clock = monotonic_clock or time.monotonic
        if resume:
            if not self.ledger_path.is_file():
                raise FormalExecutionError("resume requires an existing phase ledger")
            self.events()
        else:
            if self.output_root.exists() and any(self.output_root.iterdir()):
                raise FormalExecutionError("output root conflict: non-empty path already exists")
            self.output_root.mkdir(parents=True, exist_ok=True)
            if self.ledger_path.exists():
                raise FormalExecutionError("phase ledger already exists")

    def events(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.ledger_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FormalExecutionError(
                    f"invalid append-only phase ledger line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise FormalExecutionError("phase ledger event must be an object")
            records.append(record)
        validate_phase_ledger(records)
        return records

    def _append_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        existing = self.events()
        previous_hash = existing[-1]["current_record_hash"] if existing else None
        payload = {
            "phase_ledger_schema_version": FORMAL_PHASE_LEDGER_SCHEMA_VERSION,
            **dict(record),
            "sequence_number": len(existing) + 1,
            "previous_record_hash": previous_hash,
        }
        payload["current_record_hash"] = _ledger_record_hash(payload)
        _reject_non_finite(payload)
        validate_phase_ledger([*existing, payload])
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, allow_nan=False
        ) + "\n"
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return payload

    def _completed(self) -> dict[str, dict[str, Any]]:
        return {
            str(record["phase"]): record
            for record in self.events()
            if record.get("status") == "completed"
        }

    def _resolve_output_patterns(self, patterns: Sequence[str]) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for pattern in patterns:
            candidate = Path(pattern)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise FormalExecutionError("expected output pattern must stay within output_root")
            files = [path for path in sorted(self.output_root.glob(pattern)) if path.is_file()]
            if not files:
                raise FormalExecutionError(f"phase expected output missing: {pattern}")
            for path in files:
                resolved[path.relative_to(self.output_root).as_posix()] = file_sha256(path)
        return resolved

    def _base_record(
        self,
        *,
        phase: str,
        status: str,
        started_at: str,
        completed_at: str | None,
        wall_clock_seconds: float | None,
        input_hash: str,
        output_hash: str | None,
        commands: list[list[str]],
        return_code: int | None,
        retry_count: int,
        failure_classification: str | None,
        failure_message_reference: str | None,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "phase": phase,
            "status": status,
            "started_at": started_at,
            "completed_at": completed_at,
            "wall_clock_seconds": wall_clock_seconds,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "expanded_command": commands,
            "commands": commands,
            "command_identity": canonical_sha256(commands),
            "return_code": return_code,
            "returncode": return_code,
            "retry_count": int(retry_count),
            "failure_classification": failure_classification,
            "failure_message_reference": failure_message_reference,
            **extra,
        }

    def run_phase(
        self,
        phase: str,
        *,
        command: Sequence[str] | Sequence[Sequence[str]],
        input_hash: str,
        expected_outputs: Sequence[str],
        executor: Callable[[Sequence[str]], CommandResult] | None = None,
        infrastructure_retries: int = 1,
    ) -> dict[str, Any]:
        if phase not in PHASE_ORDER:
            raise FormalExecutionError(f"unknown phase: {phase}")
        if command and isinstance(command[0], (list, tuple)):  # type: ignore[index]
            commands = [list(item) for item in command]  # type: ignore[arg-type]
        elif command:
            commands = [list(command)]  # type: ignore[list-item]
        else:
            commands = []
        for item in commands:
            validate_no_holdout_capability(item)
        if infrastructure_retries not in {0, 1}:
            raise FormalExecutionError("infrastructure retry may be at most one")
        records = self.events()
        if any(record.get("status") == "failed" for record in records):
            raise FormalExecutionError("failed phase is terminal and cannot be overwritten")
        completed = self._completed()
        if phase == "train" and any(item in completed for item in FORMAL_PHASES):
            raise FormalExecutionError("training is forbidden after formal execution starts")
        if phase in completed:
            prior = completed[phase]
            if prior.get("input_hash") != input_hash:
                raise FormalExecutionError("completed phase input hash mismatch")
            current_hashes = self._resolve_output_patterns(expected_outputs)
            if canonical_sha256(current_hashes) != prior.get("output_hash"):
                raise FormalExecutionError("completed phase output hash mismatch")
            return {"status": "skipped_completed_hash_match", "phase": phase}
        phase_index = PHASE_ORDER.index(phase)
        for predecessor in PHASE_ORDER[:phase_index]:
            if predecessor not in completed:
                raise FormalExecutionError(
                    f"phase order violation: {predecessor} must complete before {phase}"
                )
        if phase == "complete_without_holdout" and commands:
            raise FormalExecutionError("complete_without_holdout is an internal zero-command phase")

        start_dt = self._clock()
        if start_dt.tzinfo is None or start_dt.utcoffset() is None:
            raise FormalExecutionError("phase runner clock must be timezone-aware")
        started_at = start_dt.isoformat()
        monotonic_start = float(self._monotonic_clock())
        prior_running = sum(
            record.get("phase") == phase and record.get("status") == "running"
            for record in records
        )
        self._append_record(
            self._base_record(
                phase=phase,
                status="running",
                started_at=started_at,
                completed_at=None,
                wall_clock_seconds=None,
                input_hash=input_hash,
                output_hash=None,
                commands=commands,
                return_code=None,
                retry_count=prior_running,
                failure_classification=None,
                failure_message_reference=None,
            )
        )
        execute = executor or self._subprocess_executor
        command_attempts: list[int] = []
        result = CommandResult(returncode=0)
        total_retries = prior_running
        try:
            for command_index, current_command in enumerate(commands):
                attempts = 0
                while True:
                    attempts += 1
                    try:
                        result = execute(current_command)
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:  # noqa: BLE001 - ledger must capture executor failure.
                        completed_at, wall_clock = self._terminal_time(
                            start_dt, monotonic_start
                        )
                        classification = classify_phase_failure(
                            phase=phase, return_code=None, message=str(exc)
                        )
                        self._append_record(
                            self._base_record(
                                phase=phase,
                                status="failed",
                                started_at=started_at,
                                completed_at=completed_at,
                                wall_clock_seconds=wall_clock,
                                input_hash=input_hash,
                                output_hash=None,
                                commands=commands,
                                return_code=None,
                                retry_count=total_retries,
                                failure_classification=classification,
                                failure_message_reference=f"{type(exc).__name__}: {exc}"[-4000:],
                                failed_command_index=command_index,
                                attempts=attempts,
                            )
                        )
                        raise FormalExecutionError(f"phase failed: {phase}") from exc
                    if result.returncode == 0:
                        break
                    if result.returncode == 75 and attempts <= infrastructure_retries:
                        total_retries += 1
                        continue
                    completed_at, wall_clock = self._terminal_time(
                        start_dt, monotonic_start
                    )
                    message = result.stderr or result.stdout
                    classification = classify_phase_failure(
                        phase=phase,
                        return_code=result.returncode,
                        message=message,
                    )
                    self._append_record(
                        self._base_record(
                            phase=phase,
                            status="failed",
                            started_at=started_at,
                            completed_at=completed_at,
                            wall_clock_seconds=wall_clock,
                            input_hash=input_hash,
                            output_hash=None,
                            commands=commands,
                            return_code=result.returncode,
                            retry_count=total_retries,
                            failure_classification=classification,
                            failure_message_reference=message[-4000:] or None,
                            failed_command_index=command_index,
                            attempts=attempts,
                        )
                    )
                    raise FormalExecutionError(f"phase failed: {phase}")
                command_attempts.append(attempts)
            try:
                output_hashes = self._resolve_output_patterns(expected_outputs)
            except FormalExecutionError as exc:
                completed_at, wall_clock = self._terminal_time(start_dt, monotonic_start)
                self._append_record(
                    self._base_record(
                        phase=phase,
                        status="failed",
                        started_at=started_at,
                        completed_at=completed_at,
                        wall_clock_seconds=wall_clock,
                        input_hash=input_hash,
                        output_hash=None,
                        commands=commands,
                        return_code=0,
                        retry_count=total_retries,
                        failure_classification="artifact_integrity_failure",
                        failure_message_reference=str(exc),
                        missing_outputs=list(expected_outputs),
                        attempts=sum(command_attempts),
                    )
                )
                raise
        except KeyboardInterrupt as exc:
            completed_at, wall_clock = self._terminal_time(start_dt, monotonic_start)
            self._append_record(
                self._base_record(
                    phase=phase,
                    status="failed",
                    started_at=started_at,
                    completed_at=completed_at,
                    wall_clock_seconds=wall_clock,
                    input_hash=input_hash,
                    output_hash=None,
                    commands=commands,
                    return_code=130,
                    retry_count=total_retries,
                    failure_classification="user_interruption",
                    failure_message_reference="KeyboardInterrupt",
                    attempts=sum(command_attempts),
                )
            )
            raise FormalExecutionError(f"phase interrupted: {phase}") from exc

        completed_at, wall_clock = self._terminal_time(start_dt, monotonic_start)
        output_hash = canonical_sha256(output_hashes)
        event = self._append_record(
            self._base_record(
                phase=phase,
                status="completed",
                started_at=started_at,
                completed_at=completed_at,
                wall_clock_seconds=wall_clock,
                input_hash=input_hash,
                output_hash=output_hash,
                commands=commands,
                return_code=0,
                retry_count=total_retries,
                failure_classification=None,
                failure_message_reference=None,
                command_attempts=command_attempts,
                attempts=sum(command_attempts),
                output_files=output_hashes,
            )
        )
        return event

    def _terminal_time(
        self, started: datetime, monotonic_started: float
    ) -> tuple[str, float]:
        completed = self._clock()
        if completed.tzinfo is None or completed.utcoffset() is None:
            raise FormalExecutionError("phase runner clock must be timezone-aware")
        if completed < started:
            raise FormalExecutionError("phase runner system time moved backwards")
        monotonic_delta = float(self._monotonic_clock()) - monotonic_started
        if not math.isfinite(monotonic_delta) or monotonic_delta < 0:
            raise FormalExecutionError("phase runner monotonic clock is invalid")
        timestamp_delta = (completed - started).total_seconds()
        if abs(timestamp_delta - monotonic_delta) > LEDGER_WALL_CLOCK_TOLERANCE_SECONDS:
            raise FormalExecutionError("phase runner wall clock diverged from system time")
        return completed.isoformat(), round(monotonic_delta, 9)

    @staticmethod
    def _subprocess_executor(command: Sequence[str]) -> CommandResult:
        completed = subprocess.run(
            list(command), text=True, capture_output=True, check=False
        )
        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


__all__ = [
    "AppendOnlyPhaseRunner",
    "CommandResult",
    "FORMAL_EXECUTION_PROTOCOL_ID",
    "FORMAL_EXECUTION_PROTOCOL_VERSION",
    "FORMAL_EXECUTION_PROTOCOL_V1_2_ID",
    "FORMAL_EXECUTION_PROTOCOL_V1_2_VERSION",
    "FORMAL_EXECUTION_PROTOCOL_V1_3_ID",
    "FORMAL_EXECUTION_PROTOCOL_V1_3_VERSION",
    "FORMAL_EXECUTION_PROTOCOL_V1_4_ID",
    "FORMAL_EXECUTION_PROTOCOL_V1_4_VERSION",
    "FORMAL_EXECUTION_PROTOCOL_V1_5_ID",
    "FORMAL_EXECUTION_PROTOCOL_V1_5_VERSION",
    "FORMAL_EXECUTION_PROTOCOL_V1_6_ID",
    "FORMAL_EXECUTION_PROTOCOL_V1_6_VERSION",
    "FORMAL_EXECUTION_PROTOCOL_V1_7_ID",
    "FORMAL_EXECUTION_PROTOCOL_V1_7_VERSION",
    "FORMAL_EXECUTION_PROTOCOL_V1_8_ID",
    "FORMAL_EXECUTION_PROTOCOL_V1_8_VERSION",
    "FORMAL_EXECUTION_PROTOCOL_V1_9_ID",
    "FORMAL_EXECUTION_PROTOCOL_V1_9_VERSION",
    "FORMAL_EXECUTION_PROTOCOL_V2_0_ID",
    "FORMAL_EXECUTION_PROTOCOL_V2_0_VERSION",
    "FORMAL_EXECUTION_PROTOCOL_V2_1_ID",
    "FORMAL_EXECUTION_PROTOCOL_V2_1_VERSION",
    "FORMAL_EXECUTION_PROTOCOL_V2_2_ID",
    "FORMAL_EXECUTION_PROTOCOL_V2_2_VERSION",
    "FORMAL_EXECUTION_PROTOCOL_V2_3_ID",
    "FORMAL_EXECUTION_PROTOCOL_V2_3_VERSION",
    "FORMAL_EXECUTION_PROTOCOL_V2_6_ID",
    "FORMAL_EXECUTION_PROTOCOL_V2_6_VERSION",
    "FORMAL_PHASE_LEDGER_SCHEMA_VERSION",
    "FORMAL_PHASE_RUNNER_VERSION",
    "FAILURE_CLASSIFICATIONS",
    "FormalExecutionError",
    "PHASE_ORDER",
    "PRIMARY_ENDPOINTS",
    "READY_VERDICT",
    "READY_V4_VERDICT",
    "READY_V5_VERDICT",
    "READY_V6_VERDICT",
    "READY_V7_VERDICT",
    "READY_V8_VERDICT",
    "READY_V9_VERDICT",
    "READINESS_REVIEW_VERSION",
    "build_scalability_setting_matrix",
    "build_support_setting_matrix",
    "classify_phase_failure",
    "endpoint_schema",
    "expand_command_template",
    "protocol_hash_changes_on_mutation",
    "readiness_v3",
    "readiness_v4",
    "readiness_v5",
    "readiness_v6",
    "readiness_v7",
    "readiness_v8",
    "readiness_v9",
    "reconcile_primary_endpoint_row",
    "stable_setting_identity",
    "support_setting_by_id",
    "validate_command_templates",
    "validate_no_holdout_capability",
    "validate_phase_ledger",
    "validate_protocol_v1_1",
    "validate_support_binding",
]
