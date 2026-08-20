"""G14R executable contracts for typed model-cache protocol v1.1.

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
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.evaluators.typed_model_cache_formal_protocol import (
    canonical_sha256,
    semantic_projection,
)


FORMAL_EXECUTION_PROTOCOL_VERSION = "1.1.0"
FORMAL_EXECUTION_PROTOCOL_ID = "typed_model_cache_formal_protocol_v1_1"
FORMAL_PHASE_RUNNER_VERSION = "1.0.0"
PRIMARY_ENDPOINT_SCHEMA_VERSION = "1.0.0"
SUPPORT_RUNNER_CONTRACT_VERSION = "1.0.0"
READINESS_REVIEW_VERSION = "3.0.0"
READY_VERDICT = "READY_FOR_G14C_V2_CLEAN_TRAIN_AND_FORMAL"
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
        "cache_efficiency_metrics_contract_version": "1.2.0",
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
        context = {**expansion_context, **dict(overlay)}
        commands.append(expand_command_template(spec["argv"], context))
        for pattern in spec["expected_outputs"]:
            rendered = expand_command_template([str(pattern)], context)[0]
            if rendered not in expected_outputs:
                expected_outputs.append(rendered)
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
    return {"status": "pass", "expanded": expanded}


def validate_protocol_v1_1(protocol: Mapping[str, Any]) -> dict[str, Any]:
    _reject_non_finite(protocol)
    if protocol.get("typed_model_cache_formal_protocol_version") != FORMAL_EXECUTION_PROTOCOL_VERSION:
        raise FormalExecutionError("unsupported formal execution protocol version")
    if protocol.get("protocol_id") != FORMAL_EXECUTION_PROTOCOL_ID:
        raise FormalExecutionError("formal execution protocol ID mismatch")
    supersession = protocol.get("supersession", {})
    if supersession.get("supersedes_version") != "1.0.0":
        raise FormalExecutionError("protocol v1.1 must supersede v1.0")
    if supersession.get("old_protocol_status") != "invalid_before_execution":
        raise FormalExecutionError("old protocol invalid status is missing")
    if supersession.get("old_protocol_semantic_sha256") != OLD_PROTOCOL_SEMANTIC_SHA256:
        raise FormalExecutionError("old protocol hash mismatch")
    if protocol.get("identity", {}).get("split_semantic_sha256") != SPLIT_SEMANTIC_SHA256:
        raise FormalExecutionError("split semantic hash changed")
    if tuple(protocol.get("endpoints", {}).get("primary", [])) != PRIMARY_ENDPOINTS:
        raise FormalExecutionError("primary endpoint order or identity mismatch")
    if protocol.get("endpoint_schema", {}).get("primary_endpoint_schema_version") != PRIMARY_ENDPOINT_SCHEMA_VERSION:
        raise FormalExecutionError("primary endpoint schema missing")
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


class AppendOnlyPhaseRunner:
    """Append phase events without rewriting or deleting earlier failures."""

    def __init__(
        self,
        *,
        protocol: Mapping[str, Any],
        output_root: str | Path,
        resume: bool = False,
    ) -> None:
        validate_protocol_v1_1(protocol)
        self.protocol = dict(protocol)
        self.output_root = Path(output_root)
        self.ledger_path = self.output_root / "phase_state.jsonl"
        if resume:
            if not self.ledger_path.is_file():
                raise FormalExecutionError("resume requires an existing phase ledger")
        else:
            if self.output_root.exists() and any(self.output_root.iterdir()):
                raise FormalExecutionError("output root conflict: non-empty path already exists")
            self.output_root.mkdir(parents=True, exist_ok=True)
            if self.ledger_path.exists():
                raise FormalExecutionError("phase ledger already exists")

    def events(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.ledger_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FormalExecutionError(
                    f"invalid append-only phase ledger line {line_number}"
                ) from exc
            if not isinstance(event, dict):
                raise FormalExecutionError("phase ledger event must be an object")
            events.append(event)
        return events

    def _append(self, event: Mapping[str, Any]) -> None:
        _reject_non_finite(event)
        encoded = json.dumps(
            dict(event), ensure_ascii=False, sort_keys=True, allow_nan=False
        ) + "\n"
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def _completed(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for event in self.events():
            if event.get("status") == "completed":
                result[str(event["phase"])] = event
        return result

    def _resolve_output_patterns(self, patterns: Sequence[str]) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for pattern in patterns:
            candidate = Path(pattern)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise FormalExecutionError("expected output pattern must stay within output_root")
            matches = sorted(self.output_root.glob(pattern))
            files = [path for path in matches if path.is_file()]
            if not files:
                raise FormalExecutionError(f"phase expected output missing: {pattern}")
            for path in files:
                relative = path.relative_to(self.output_root).as_posix()
                resolved[relative] = file_sha256(path)
        return resolved

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
        events = self.events()
        failed = [event for event in events if event.get("status") == "failed"]
        if failed:
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

        execute = executor or self._subprocess_executor
        command_attempts: list[int] = []
        result = CommandResult(returncode=0)
        for command_index, current_command in enumerate(commands):
            attempts = 0
            while True:
                attempts += 1
                result = execute(current_command)
                if result.returncode == 0:
                    break
                # Exit 75 is the only frozen infrastructure-temporary code eligible for retry.
                if result.returncode != 75 or attempts > infrastructure_retries:
                    self._append(
                        {
                            "phase": phase,
                            "status": "failed",
                            "input_hash": input_hash,
                            "commands": commands,
                            "failed_command_index": command_index,
                            "returncode": result.returncode,
                            "attempts": attempts,
                            "stderr": result.stderr[-4000:],
                        }
                    )
                    raise FormalExecutionError(f"phase failed: {phase}")
            command_attempts.append(attempts)
        try:
            output_hashes = self._resolve_output_patterns(expected_outputs)
        except FormalExecutionError as exc:
            self._append(
                {
                    "phase": phase,
                    "status": "failed",
                    "input_hash": input_hash,
                    "commands": commands,
                    "returncode": 0,
                    "attempts": attempts,
                    "missing_outputs": list(expected_outputs),
                }
            )
            raise exc
        event = {
            "phase": phase,
            "status": "completed",
            "input_hash": input_hash,
            "commands": commands,
            "returncode": result.returncode,
            "command_attempts": command_attempts,
            "attempts": sum(command_attempts),
            "output_files": output_hashes,
            "output_hash": canonical_sha256(output_hashes),
        }
        self._append(event)
        return event

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
    "FORMAL_PHASE_RUNNER_VERSION",
    "FormalExecutionError",
    "PHASE_ORDER",
    "PRIMARY_ENDPOINTS",
    "READY_VERDICT",
    "READINESS_REVIEW_VERSION",
    "build_scalability_setting_matrix",
    "build_support_setting_matrix",
    "endpoint_schema",
    "expand_command_template",
    "protocol_hash_changes_on_mutation",
    "readiness_v3",
    "reconcile_primary_endpoint_row",
    "stable_setting_identity",
    "support_setting_by_id",
    "validate_command_templates",
    "validate_no_holdout_capability",
    "validate_protocol_v1_1",
    "validate_support_binding",
]
