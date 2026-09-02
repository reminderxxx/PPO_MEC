"""Freeze Protocol 2.2 with one persistent formal request subject lifecycle."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.typed_model_cache_formal_protocol import attach_hashes
from src.runtime.active_formal_bundle import (
    ACTIVE_BUNDLE_RESOURCE_RESOLUTION_CONTRACT_VERSION,
    ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION,
    ACTIVE_PROTOCOL_ID,
    ACTIVE_PROTOCOL_VERSION,
    READINESS_VERSION,
    READY_STATUS,
    active_bundle_core_projection,
    build_resource_row,
    canonical_sha256,
    ready_index_projection,
    sha256_file,
)
from src.runtime.formal_execution_environment import (
    ENVIRONMENT_IDENTITY_RULE,
    EXECUTION_COMMIT_IDENTITY_RULE,
    FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION,
    FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION,
    PROTOCOL_BOUND_EXTENSION_FIELDS,
    RUNTIME_OBSERVABLE_IDENTITY_FIELDS,
    SOURCE_ROOT_IDENTITY_RULE,
    build_environment_identity_projection,
)
from src.runtime.formal_exogenous_request_execution import (
    FORMAL_EXOGENOUS_REQUEST_EXECUTION_CONTRACT_VERSION,
    FORMAL_REQUEST_EXPOSURE_TRACE_VERSION,
    FORMAL_REQUEST_PHYSICAL_CONTINUITY_RULE_VERSION,
    FORMAL_REQUEST_SUBJECT_LIFECYCLE_CONTRACT_VERSION,
    FORMAL_REQUEST_SUBJECT_SELECTION_VERSION,
    REQUIRED_REQUEST_FIELDS,
    REQUIRED_SUBJECT_LIFECYCLE_FIELDS,
)


SOURCE = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_1_20260831"
TARGET = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_2_20260901"
ARTIFACT = ROOT / (
    "artifacts/analysis/"
    "typed_model_cache_formal_request_subject_repair_20260901_g14r11_v1"
)
OLD_DIR = SOURCE.name
NEW_DIR = TARGET.name


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8-sig"),
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant in {path}: {item}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def semantic(value: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(value)
    payload.pop("semantic_sha256", None)
    payload["semantic_sha256"] = canonical_sha256(payload)
    return payload


def replace_paths(value: Any) -> Any:
    if isinstance(value, str):
        return (
            value.replace(OLD_DIR, NEW_DIR)
            .replace("protocol_v2_1_manifest.json", "protocol_v2_2_manifest.json")
            .replace("readiness_v13.json", "readiness_v14.json")
            .replace("G14R10", "G14R11")
        )
    if isinstance(value, list):
        return [replace_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: replace_paths(item) for key, item in value.items()}
    return value


def current(
    logical_id: str,
    role: str,
    filename: str,
    semantic_sha256: str | None = None,
) -> dict[str, Any]:
    return build_resource_row(
        root=ROOT,
        logical_id=logical_id,
        role=role,
        relative_path=(TARGET / filename).relative_to(ROOT).as_posix(),
        version_scope="current_protocol_version",
        semantic_sha256=semantic_sha256,
    )


def lifecycle_contract() -> dict[str, Any]:
    return semantic(
        {
            "version": FORMAL_REQUEST_SUBJECT_LIFECYCLE_CONTRACT_VERSION,
            "mode": "one_workflow_one_persistent_vehicle",
            "request_count_rule": (
                "min(DAG_execution_order_length,max_steps,mobility_frame_count_minus_one)"
            ),
            "eligibility": {
                "presence_frames": "0..request_count inclusive",
                "same_vehicle_id_required": True,
                "physical_continuity_rule_version": (
                    FORMAL_REQUEST_PHYSICAL_CONTINUITY_RULE_VERSION
                ),
                "vehicle_id_reuse_teleport_or_segment_drift_allowed": False,
                "outcome_inputs_allowed": False,
            },
            "selection": {
                "version": FORMAL_REQUEST_SUBJECT_SELECTION_VERSION,
                "mode": "existing_primary_vehicle_selection_after_eligibility_filter",
                "handoff_pressure_definition_changed": False,
                "deterministic_tie_break_required": True,
            },
            "evidence_fields": sorted(REQUIRED_SUBJECT_LIFECYCLE_FIELDS),
            "rsu_time_semantics": {
                "request_rsu_frame": "step_index_minus_one",
                "current_service_rsu_frame": "step_index",
                "time_index_frame": "step_index",
                "same_selected_vehicle_and_mapper_required": True,
                "null_association_preserved": True,
            },
            "runtime": {
                "reselection_policy": "forbidden_during_formal_episode",
                "missing_or_nonphysical_subject": "fail_fast",
                "runtime_may_rewrite_exposure": False,
                "legacy_nonformal_reselection_unchanged": True,
            },
            "visibility": {
                "selection_evidence_actor_visible": False,
                "selection_evidence_controller_visible": False,
            },
            "no_eligible_subject_verdict": (
                "BLOCKED_BY_FORMAL_REQUEST_SUBJECT_ELIGIBILITY"
            ),
        }
    )


def request_contract(lifecycle: dict[str, Any]) -> dict[str, Any]:
    return semantic(
        {
            "version": FORMAL_EXOGENOUS_REQUEST_EXECUTION_CONTRACT_VERSION,
            "request_exposure_trace_version": FORMAL_REQUEST_EXPOSURE_TRACE_VERSION,
            "request_subject_lifecycle_contract_version": lifecycle["version"],
            "request_subject_lifecycle_contract_semantic_sha256": lifecycle[
                "semantic_sha256"
            ],
            "producer_identity": "formal_request_exposure_producer_v2.0.0",
            "producer_entrypoint": (
                "src.evaluators.main_results_support.build_episode_formal_request_exposure"
            ),
            "execution_entrypoint": (
                "src.envs.core.vec_workflow_core_env.VecWorkflowCoreEnv"
            ),
            "default_enabled": False,
            "formal_explicit_enable_required": True,
            "phase_modes": {
                "train": "replay_driven_exogenous_request_exposure",
                "dev": "replay_driven_exogenous_request_exposure",
                "formal": "replay_driven_exogenous_request_exposure",
                "legacy_nonformal": "legacy_endogenous_progression",
            },
            "action_before_lookup": True,
            "service_failure_changes_future_exposure": False,
            "request_fingerprint_is_outcome_fingerprint": False,
            "request_fingerprint_excludes_execution_specific_provenance": True,
            "actor_or_controller_may_observe_oracle_future_topology": False,
            "actor_or_controller_may_observe_subject_selection_evidence": False,
            "cache_event_alignment": (
                "exactly_one_request_level_CacheEvent_per_exposure"
            ),
            "g08_replay_role": (
                "outcome_blind_analytical_replay_reuses_formal_lifecycle_producer"
            ),
            "g09_opportunity_role": (
                "outcome_blind_analysis_bound_to_exposure_provenance"
            ),
            "fail_fast": [
                "missing_replay",
                "duplicate_missing_extra_or_out_of_order_request",
                "cross_agent_or_capacity_fingerprint_drift",
                "request_event_identity_drift",
                "catalog_dependency_size_or_evaluation_unit_drift",
                "subject_missing_or_nonphysical",
                "subject_candidate_count_or_fingerprint_drift",
                "subject_or_rsu_runtime_drift",
                "outcome_pollution",
                "endogenous_fallback",
                "future_topology_leakage",
                "historical_bundle_v9_or_v11_reference",
            ],
        }
    )


def request_schema(lifecycle: dict[str, Any]) -> dict[str, Any]:
    return semantic(
        {
            "formal_request_exposure_schema_version": "2.0.0",
            "contract_version": FORMAL_EXOGENOUS_REQUEST_EXECUTION_CONTRACT_VERSION,
            "subject_lifecycle_contract_version": lifecycle["version"],
            "required_subject_lifecycle_fields": sorted(
                REQUIRED_SUBJECT_LIFECYCLE_FIELDS
            ),
            "required_request_fields": sorted(REQUIRED_REQUEST_FIELDS),
            "fingerprint": (
                "SHA-256 over the trace excluding fingerprint, validation, and "
                "execution-specific source_provenance"
            ),
            "canonical_serialization": (
                "UTF-8 sorted-key compact JSON; NaN/Infinity rejected"
            ),
            "outcome_fields_forbidden": True,
            "oracle_future_actor_visible": False,
            "subject_selection_evidence_actor_visible": False,
            "subject_selection_evidence_controller_visible": False,
            "unknown_missing_or_extra_lifecycle_fields_rejected": True,
        }
    )


def projection_contract() -> dict[str, Any]:
    return semantic(
        {
            "version": FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION,
            "formal_execution_environment_contract_version": (
                FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION
            ),
            "runtime_observable_fields": list(RUNTIME_OBSERVABLE_IDENTITY_FIELDS),
            "protocol_bound_scientific_extension_fields": list(
                PROTOCOL_BOUND_EXTENSION_FIELDS
            ),
            "environment_fingerprint_field": "environment_fingerprint",
            "fingerprint_rule": (
                "SHA-256(UTF-8 sorted-key compact canonical JSON of the normalized "
                "full scientific identity excluding only environment_fingerprint)"
            ),
            "reject": [
                "missing_fields",
                "unknown_fields",
                "duplicate_or_alias_fields",
                "wrong_types",
                "unsupported_contract_major",
                "NaN_or_Infinity",
                "protocol_extension_version_drift",
            ],
            "host_fields_excluded": [
                "python_absolute_path",
                "clean_worktree_absolute_path",
                "cwd",
                "site_packages_paths",
                "sys_path",
                "virtual_environment_root",
            ],
            "execution_commit_identity_rule": EXECUTION_COMMIT_IDENTITY_RULE,
            "source_root_identity_rule": dict(SOURCE_ROOT_IDENTITY_RULE),
            "identity_rule": ENVIRONMENT_IDENTITY_RULE,
            "producer": (
                "src.runtime.formal_execution_environment."
                "build_environment_identity_projection"
            ),
        }
    )


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    old_protocol = read_json(SOURCE / "protocol_v2_1_manifest.json")
    old_index = read_json(SOURCE / "protocol_index.json")
    lifecycle = lifecycle_contract()
    request = request_contract(lifecycle)
    schema = request_schema(lifecycle)
    projection = projection_contract()

    copied = (
        "agent_training_scientific_config.json",
        "formal_agent_order_contract.json",
        "formal_training_execution_binding_contract.json",
        "resolved_execution_context_contract.json",
        "active_bundle_resource_resolution_contract.json",
        "nonformal_rehearsal_window_plan.json",
        "nonformal_rehearsal_fairness_constrained_288mb.json",
        "nonformal_rehearsal_fairness_medium_576mb.json",
        "nonformal_rehearsal_fairness_relaxed_864mb.json",
    )
    for filename in copied:
        write_json(TARGET / filename, replace_paths(read_json(SOURCE / filename)))
    write_json(TARGET / "formal_request_subject_lifecycle_contract.json", lifecycle)
    write_json(TARGET / "formal_exogenous_request_execution_contract.json", request)
    write_json(TARGET / "formal_request_exposure_schema.json", schema)
    write_json(TARGET / "environment_identity_projection_contract.json", projection)

    binding_schema = read_json(TARGET / "formal_training_execution_binding_contract.json")
    binding_schema.update(
        environment_projection_contract_version=(
            FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION
        ),
        request_subject_lifecycle_hash_enters_binding_context_and_checkpoint=True,
    )
    write_json(TARGET / "formal_training_execution_binding_contract.json", binding_schema)
    context_schema = read_json(TARGET / "resolved_execution_context_contract.json")
    context_schema.update(
        environment_projection_contract_version=(
            FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION
        ),
        request_subject_lifecycle_hash_in_context=True,
    )
    write_json(TARGET / "resolved_execution_context_contract.json", context_schema)

    old_identity = old_protocol["formal_execution_environment_contract"][
        "scientific_identity"
    ]
    runtime_observable = {
        "formal_execution_environment_contract_version": (
            FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION
        ),
        **{
            field: deepcopy(old_identity[field])
            for field in RUNTIME_OBSERVABLE_IDENTITY_FIELDS
            if field != "formal_execution_environment_contract_version"
        },
    }
    extensions = {
        "formal_endpoint_metrics_contract_version": "2.0.0",
        "formal_exogenous_request_execution_contract_version": request["version"],
        "formal_request_exposure_trace_version": request[
            "request_exposure_trace_version"
        ],
        "formal_request_subject_lifecycle_contract_version": lifecycle["version"],
    }
    environment_identity = build_environment_identity_projection(
        runtime_observable, extensions
    )
    environment = replace_paths(read_json(SOURCE / "execution_environment_manifest.json"))
    environment.update(
        formal_execution_environment_contract_version=(
            FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION
        ),
        projection_contract_version=(
            FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION
        ),
        scientific_identity=environment_identity,
    )
    write_json(TARGET / "execution_environment_manifest.json", environment)

    protocol = replace_paths(deepcopy(old_protocol))
    protocol.update(
        typed_model_cache_formal_protocol_version=ACTIVE_PROTOCOL_VERSION,
        protocol_id=ACTIVE_PROTOCOL_ID,
        status="frozen_pre_execution_formal_request_subject_lifecycle",
        formal_exogenous_request_execution_contract=deepcopy(request),
        formal_request_exposure_schema=deepcopy(schema),
        formal_request_subject_lifecycle_contract=deepcopy(lifecycle),
    )
    protocol["identity"][
        "formal_request_subject_lifecycle_contract_semantic_sha256"
    ] = lifecycle["semantic_sha256"]
    protocol["supersession"].update(
        supersedes_version="2.1.0",
        old_protocol_status="invalid_protocol_or_implementation",
        old_protocol_semantic_sha256=old_protocol["hashes"]["semantic_sha256"],
        scientific_fields_changed=True,
        formal_performance_observed=False,
        repair_scope=[
            "filter primary candidates by horizon-long presence and physical continuity",
            "freeze one vehicle for all request/current RSU associations",
            "forbid runtime primary reselection during formal episodes",
            "bind lifecycle evidence into exposure and active Protocol identity",
        ],
    )
    invalid_runs = protocol["supersession"]["invalid_execution_runs"]
    invalid_runs.append(
        {
            "run_id": "typed_model_cache_formal_20260901_155201_g14c_v11",
            "run_root": (
                "artifacts/experiments/typed_model_cache_formal/"
                "typed_model_cache_formal_20260901_155201_g14c_v11"
            ),
            "status": "INVALID_PROTOCOL_OR_IMPLEMENTATION",
            "failure_boundary": (
                "invalid_during_first_training_cell_before_first_episode_commit"
            ),
            "failure_audit_sha256": (
                "b5eb0063c0cde2670d298027a8aeea1b4661b77fc404b72f673970642662e362"
            ),
            "failure_integrity_sha256": (
                "847f3b0d8c8381814c04ba99c643068c3e0c36dd6fe1f10ac2b5542175ba2b72"
            ),
            "training_cells_executed": 0,
            "candidate_checkpoint_count": 0,
            "dev_performance_count": 0,
            "formal_performance_count": 0,
            "resume_allowed": False,
            "retry_allowed": False,
            "legacy_phase_finalize_allowed": False,
            "checkpoint_reuse_allowed": False,
            "candidate_reuse_allowed": False,
            "partial_dev_input_reuse_allowed": False,
            "immutable_old_run": True,
        }
    )
    formal_environment = protocol["formal_execution_environment_contract"]
    formal_environment.update(
        version=FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION,
        identity_projection_contract_version=(
            FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION
        ),
        scientific_identity=deepcopy(environment_identity),
    )
    formal_environment["resolver"]["version"] = "1.2.0"
    protocol["formal_training_execution_binding_contract"].update(
        environment_projection_contract_version=(
            FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION
        ),
        request_subject_lifecycle_hash_enters_binding_context_and_checkpoint=True,
    )
    protocol["resolved_formal_execution_context_contract"].update(
        environment_projection_contract_version=(
            FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION
        ),
        request_subject_lifecycle_hash_in_context=True,
    )
    protocol["active_formal_bundle_contract"]["unique_active_index"] = (
        TARGET / "protocol_index.json"
    ).relative_to(ROOT).as_posix()
    protocol["active_formal_bundle_contract"]["hash_graph"] = [
        "resource content hashes -> active_bundle_core_sha256",
        "core plus acceptance evidence -> Readiness v14 content hash",
        "ready index plus Readiness content hash -> active_formal_bundle_sha256",
    ]
    context = protocol["execution_contract"]["default_expansion_context"]
    context["active_protocol_index_path"] = (
        TARGET / "protocol_index.json"
    ).relative_to(ROOT).as_posix()
    context["protocol_path"] = (
        TARGET / "protocol_v2_2_manifest.json"
    ).relative_to(ROOT).as_posix()
    protocol["paper_claim_boundary"] = (
        "G14R11 is outcome-blind execution-contract evidence only; formal training, "
        "checkpoint, performance, holdout, G14C v12, G14D, and G15 remain unexecuted."
    )
    protocol = attach_hashes(protocol)
    write_json(TARGET / "protocol_v2_2_manifest.json", protocol)

    scientific = read_json(TARGET / "agent_training_scientific_config.json")
    order = read_json(TARGET / "formal_agent_order_contract.json")
    resolution = read_json(TARGET / "active_bundle_resource_resolution_contract.json")
    current_rows = [
        current("protocol_manifest", "active Protocol manifest", "protocol_v2_2_manifest.json", protocol["hashes"]["semantic_sha256"]),
        current("execution_environment_manifest", "active execution environment identity", "execution_environment_manifest.json"),
        current("environment_identity_projection_contract", "formal environment identity projection contract", "environment_identity_projection_contract.json", projection["semantic_sha256"]),
        current("agent_training_scientific_config", "Scientific Config 2.0.0", "agent_training_scientific_config.json", scientific["config_semantic_sha256"]),
        current("formal_agent_order_contract", "Formal Agent Order Contract 1.0.0", "formal_agent_order_contract.json", order["semantic_sha256"]),
        current("formal_training_execution_binding_schema", "formal execution binding schema", "formal_training_execution_binding_contract.json"),
        current("resolved_execution_context_schema", "resolved context schema", "resolved_execution_context_contract.json"),
        current("active_bundle_resource_resolution_contract", "active bundle resource resolver", "active_bundle_resource_resolution_contract.json", resolution["semantic_sha256"]),
        current("formal_exogenous_request_execution_contract", "formal exogenous request execution contract", "formal_exogenous_request_execution_contract.json", request["semantic_sha256"]),
        current("formal_request_exposure_schema", "formal request exposure schema", "formal_request_exposure_schema.json", schema["semantic_sha256"]),
        current("formal_request_subject_lifecycle_contract", "formal request subject lifecycle contract", "formal_request_subject_lifecycle_contract.json", lifecycle["semantic_sha256"]),
        current("rehearsal_window_plan", "nonformal rehearsal window plan", "nonformal_rehearsal_window_plan.json"),
        current("rehearsal_fairness_manifests.constrained_288mb", "nonformal rehearsal fairness manifest", "nonformal_rehearsal_fairness_constrained_288mb.json"),
        current("rehearsal_fairness_manifests.medium_576mb", "nonformal rehearsal fairness manifest", "nonformal_rehearsal_fairness_medium_576mb.json"),
        current("rehearsal_fairness_manifests.relaxed_864mb", "nonformal rehearsal fairness manifest", "nonformal_rehearsal_fairness_relaxed_864mb.json"),
    ]
    shared_rows = [
        deepcopy(row)
        for row in old_index["active_bundle_resources"]
        if row.get("version_scope") == "shared_historical_stable"
    ]
    index: dict[str, Any] = {
        "active_formal_bundle_contract_version": ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION,
        "active_bundle_resource_resolution_contract_version": ACTIVE_BUNDLE_RESOURCE_RESOLUTION_CONTRACT_VERSION,
        "protocol_index_version": ACTIVE_PROTOCOL_VERSION,
        "status": "NOT_READY_PENDING_G14R11_ACCEPTANCE",
        "protocol_identity": {
            "protocol_id": ACTIVE_PROTOCOL_ID,
            "protocol_version": ACTIVE_PROTOCOL_VERSION,
            "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
            "protocol_full_sha256": protocol["hashes"]["full_sha256"],
        },
        "execution_commit_binding": deepcopy(old_index["execution_commit_binding"]),
        "environment_identity": {
            "projection_contract_version": (
                FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION
            ),
            "environment_fingerprint": environment_identity[
                "environment_fingerprint"
            ],
            "dependency_fingerprint": environment_identity["dependency_fingerprint"],
        },
        "command_matrix_identity": {
            "command_templates_sha256": canonical_sha256(
                protocol["execution_contract"]["command_templates"]
            ),
            "outer_nested_expansion_equality_required": True,
        },
        "holdout_seal": deepcopy(protocol["holdout_execution_contract"]),
        "active_bundle_resources": [*current_rows, *shared_rows],
    }
    index["active_bundle_core_sha256"] = canonical_sha256(
        active_bundle_core_projection(index)
    )
    evidence_path = ARTIFACT / "acceptance_evidence_manifest.json"
    if evidence_path.is_file():
        evidence = read_json(evidence_path)
        required_passes = {
            "root_cause_audit": "pass",
            "lifecycle_contract": "pass",
            "producer_consumer_matrix": "pass",
            "negative_validation": "pass",
            "exact_failure_unit_rehearsal": "pass",
            "exposure_eligibility_audit": "pass",
            "cross_agent_capacity_exposure_parity": "pass",
            "clean_detached_candidate": "pass",
            "full_repository_pytest": "pass",
            "smoke_test": "pass",
            "json_round_trip_and_inventory": "pass",
            "protected_files": "pass",
        }
        if (
            evidence.get("status") == "pass"
            and evidence.get("active_bundle_core_sha256")
            == index["active_bundle_core_sha256"]
            and evidence.get("checks") == required_passes
            and evidence.get("formal_training_count") == 0
            and evidence.get("formal_checkpoint_count") == 0
            and evidence.get("formal_performance_count") == 0
            and evidence.get("holdout_sealed_unopened") is True
        ):
            readiness = {
                "readiness_review_version": READINESS_VERSION,
                "status": "ready",
                "verdict": READY_STATUS,
                "active_bundle_core_sha256": index["active_bundle_core_sha256"],
                "evidence_manifest_path": evidence_path.relative_to(ROOT).as_posix(),
                "evidence_manifest_sha256": sha256_file(evidence_path),
                "formal_training_count": 0,
                "formal_checkpoint_count": 0,
                "formal_performance_count": 0,
                "holdout_sealed_unopened": True,
            }
            write_json(TARGET / "readiness_v14.json", readiness)
            readiness_row = current(
                "readiness_companion",
                "Readiness v14 evidence companion",
                "readiness_v14.json",
            )
            index["active_bundle_resources"].append(readiness_row)
            index["readiness_companion"] = {
                "logical_path": readiness_row["logical_path"],
                "content_sha256": readiness_row["content_sha256"],
            }
            index["status"] = READY_STATUS
    index["active_formal_bundle_sha256"] = canonical_sha256(
        ready_index_projection(index)
    )
    write_json(TARGET / "protocol_index.json", index)
    print(
        json.dumps(
            {
                "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
                "protocol_full_sha256": protocol["hashes"]["full_sha256"],
                "active_bundle_core_sha256": index["active_bundle_core_sha256"],
                "active_formal_bundle_sha256": index[
                    "active_formal_bundle_sha256"
                ],
                "environment_fingerprint": environment_identity[
                    "environment_fingerprint"
                ],
                "lifecycle_semantic_sha256": lifecycle["semantic_sha256"],
                "status": index["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
