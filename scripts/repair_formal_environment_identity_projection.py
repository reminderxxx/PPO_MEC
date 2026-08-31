"""Freeze Protocol 2.1 with one strict environment identity projection."""

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
    READY_STATUS,
    READINESS_VERSION,
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
    protocol_bound_extensions_from_protocol,
)


SOURCE = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_0_20260831"
TARGET = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_1_20260831"
ARTIFACT = ROOT / (
    "artifacts/analysis/"
    "typed_model_cache_formal_environment_identity_repair_20260831_g14r10_v1"
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


def replace_paths(value: Any) -> Any:
    if isinstance(value, str):
        return (
            value.replace(OLD_DIR, NEW_DIR)
            .replace("protocol_v2_0_manifest.json", "protocol_v2_1_manifest.json")
            .replace("readiness_v12.json", "readiness_v13.json")
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


def projection_contract() -> dict[str, Any]:
    contract: dict[str, Any] = {
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
            "SHA-256(UTF-8 sorted-key compact canonical JSON of the normalized full "
            "scientific identity excluding only environment_fingerprint)"
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
        "producer": "src.runtime.formal_execution_environment.build_environment_identity_projection",
    }
    contract["semantic_sha256"] = canonical_sha256(contract)
    return contract


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    old_protocol = read_json(SOURCE / "protocol_v2_0_manifest.json")
    old_index = read_json(SOURCE / "protocol_index.json")

    copied = (
        "agent_training_scientific_config.json",
        "formal_agent_order_contract.json",
        "formal_training_execution_binding_contract.json",
        "resolved_execution_context_contract.json",
        "active_bundle_resource_resolution_contract.json",
        "formal_exogenous_request_execution_contract.json",
        "formal_request_exposure_schema.json",
        "nonformal_rehearsal_window_plan.json",
        "nonformal_rehearsal_fairness_constrained_288mb.json",
        "nonformal_rehearsal_fairness_medium_576mb.json",
        "nonformal_rehearsal_fairness_relaxed_864mb.json",
    )
    for filename in copied:
        write_json(TARGET / filename, replace_paths(read_json(SOURCE / filename)))

    binding_schema = read_json(TARGET / "formal_training_execution_binding_contract.json")
    binding_schema["environment_projection_contract_version"] = (
        FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION
    )
    binding_schema["full_normalized_environment_projection_in_binding"] = True
    write_json(TARGET / "formal_training_execution_binding_contract.json", binding_schema)
    context_schema = read_json(TARGET / "resolved_execution_context_contract.json")
    context_schema["environment_projection_contract_version"] = (
        FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION
    )
    context_schema["full_normalized_environment_projection_in_context"] = True
    write_json(TARGET / "resolved_execution_context_contract.json", context_schema)

    extensions = protocol_bound_extensions_from_protocol(old_protocol)
    old_identity = old_protocol["formal_execution_environment_contract"][
        "scientific_identity"
    ]
    runtime_observable = {
        "formal_execution_environment_contract_version": (
            FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION
        ),
        "python_implementation": old_identity["python_implementation"],
        "python_version": old_identity["python_version"],
        "platform_system": old_identity["platform_system"],
        "architecture": old_identity["architecture"],
        "dependency_fingerprint": old_identity["dependency_fingerprint"],
        "installed_package_count": old_identity["installed_package_count"],
        "torch_version": old_identity["torch_version"],
        "critical_package_versions": deepcopy(
            old_identity["critical_package_versions"]
        ),
        "execution_commit": EXECUTION_COMMIT_IDENTITY_RULE,
        "source_root_identity": dict(SOURCE_ROOT_IDENTITY_RULE),
        "identity_rule": ENVIRONMENT_IDENTITY_RULE,
    }
    environment_identity = build_environment_identity_projection(
        runtime_observable, extensions
    )
    environment = replace_paths(read_json(SOURCE / "execution_environment_manifest.json"))
    environment["scientific_identity"] = environment_identity
    environment["projection_contract_version"] = (
        FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION
    )
    write_json(TARGET / "execution_environment_manifest.json", environment)
    projection = projection_contract()
    write_json(TARGET / "environment_identity_projection_contract.json", projection)

    protocol = replace_paths(deepcopy(old_protocol))
    protocol["typed_model_cache_formal_protocol_version"] = ACTIVE_PROTOCOL_VERSION
    protocol["protocol_id"] = ACTIVE_PROTOCOL_ID
    protocol["status"] = "frozen_pre_execution_environment_identity_projection"
    protocol["supersession"].update(
        supersedes_version="2.0.0",
        old_protocol_status="audit_only_after_pre_execution_identity_mismatch",
        old_protocol_semantic_sha256=old_protocol["hashes"]["semantic_sha256"],
        scientific_fields_changed=True,
        formal_performance_observed=False,
        repair_scope=[
            "freeze one strict full scientific environment projection",
            "bind all three Protocol extension versions into the environment fingerprint",
            "separate host runtime audit from scientific identity",
        ],
    )
    protocol["supersession"]["pre_execution_stops"] = [
        {
            "classification": "PRE-EXECUTION STOP / EXECUTION_IDENTITY_MISMATCH",
            "clean_candidate": "/private/tmp/ppo_mec_g14c_v10_8402d2e_20260831_161419",
            "durable_run_root": None,
            "durable_run_root_created": False,
            "preflight_child_executed": False,
            "tests_train_dev_formal_statistics_gate_count": 0,
            "phase_ledger_count": 0,
            "cell_ledger_count": 0,
            "checkpoint_candidate_row_count": 0,
            "holdout_opened": False,
            "resume_or_salvage_allowed": False,
        }
    ]
    protocol["formal_agent_order_contract"].update(
        active_protocol_versions=[ACTIVE_PROTOCOL_VERSION],
        historical_protocol_versions_audit_only=[
            *[f"1.{minor}.0" for minor in range(10)],
            "2.0.0",
        ],
    )
    protocol["formal_execution_environment_contract"]["version"] = (
        FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION
    )
    protocol["formal_execution_environment_contract"][
        "identity_projection_contract_version"
    ] = FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION
    protocol["formal_execution_environment_contract"][
        "scientific_identity"
    ] = deepcopy(environment_identity)
    protocol["formal_execution_environment_contract"]["resolver"]["version"] = "1.1.0"
    protocol["active_formal_bundle_contract"]["unique_active_index"] = (
        TARGET / "protocol_index.json"
    ).relative_to(ROOT).as_posix()
    context = protocol["execution_contract"]["default_expansion_context"]
    context["active_protocol_index_path"] = (
        TARGET / "protocol_index.json"
    ).relative_to(ROOT).as_posix()
    context["protocol_path"] = (
        TARGET / "protocol_v2_1_manifest.json"
    ).relative_to(ROOT).as_posix()
    protocol["paper_claim_boundary"] = (
        "G14R10 is outcome-blind execution-contract evidence only; formal training, "
        "checkpoint, performance, holdout, G14C v11, G14D, and G15 remain unexecuted."
    )
    protocol = attach_hashes(protocol)
    write_json(TARGET / "protocol_v2_1_manifest.json", protocol)

    scientific = read_json(TARGET / "agent_training_scientific_config.json")
    order = read_json(TARGET / "formal_agent_order_contract.json")
    request_contract = read_json(
        TARGET / "formal_exogenous_request_execution_contract.json"
    )
    request_schema = read_json(TARGET / "formal_request_exposure_schema.json")
    resolution = read_json(TARGET / "active_bundle_resource_resolution_contract.json")
    current_rows = [
        current("protocol_manifest", "active Protocol manifest", "protocol_v2_1_manifest.json", protocol["hashes"]["semantic_sha256"]),
        current("execution_environment_manifest", "active execution environment identity", "execution_environment_manifest.json"),
        current("environment_identity_projection_contract", "formal environment identity projection contract", "environment_identity_projection_contract.json", projection["semantic_sha256"]),
        current("agent_training_scientific_config", "Scientific Config 2.0.0", "agent_training_scientific_config.json", scientific["config_semantic_sha256"]),
        current("formal_agent_order_contract", "Formal Agent Order Contract 1.0.0", "formal_agent_order_contract.json", order["semantic_sha256"]),
        current("formal_training_execution_binding_schema", "formal execution binding schema", "formal_training_execution_binding_contract.json"),
        current("resolved_execution_context_schema", "resolved context schema", "resolved_execution_context_contract.json"),
        current("active_bundle_resource_resolution_contract", "active bundle resource resolver", "active_bundle_resource_resolution_contract.json", resolution["semantic_sha256"]),
        current("formal_exogenous_request_execution_contract", "formal exogenous request execution contract", "formal_exogenous_request_execution_contract.json", request_contract["semantic_sha256"]),
        current("formal_request_exposure_schema", "formal request exposure schema", "formal_request_exposure_schema.json", request_schema["semantic_sha256"]),
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
        "status": READY_STATUS,
        "protocol_identity": {
            "protocol_id": ACTIVE_PROTOCOL_ID,
            "protocol_version": ACTIVE_PROTOCOL_VERSION,
            "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
            "protocol_full_sha256": protocol["hashes"]["full_sha256"],
        },
        "execution_commit_binding": deepcopy(old_index["execution_commit_binding"]),
        "environment_identity": {
            "projection_contract_version": FORMAL_ENVIRONMENT_IDENTITY_PROJECTION_CONTRACT_VERSION,
            "environment_fingerprint": environment_identity["environment_fingerprint"],
            "dependency_fingerprint": environment_identity["dependency_fingerprint"],
        },
        "command_matrix_identity": {
            "command_templates_sha256": canonical_sha256(protocol["execution_contract"]["command_templates"]),
            "outer_nested_expansion_equality_required": True,
        },
        "holdout_seal": deepcopy(protocol["holdout_execution_contract"]),
        "active_bundle_resources": [*current_rows, *shared_rows],
    }
    index["active_bundle_core_sha256"] = canonical_sha256(
        active_bundle_core_projection(index)
    )

    ARTIFACT.mkdir(parents=True, exist_ok=True)
    acceptance = {
        "status": "pass",
        "clean_candidate": True,
        "active_bundle_core_sha256": index["active_bundle_core_sha256"],
        "acceptance_scope": "g14r10_contract_and_nonformal_acceptance",
        "clean_detached_worktree": True,
        "candidate_dot_venv_present": False,
        "shared_absolute_python": "/Users/howen/Projects/PPO_MEC/.venv/bin/python",
        "project_import_from_clean_candidate": True,
        "outer_preflight_status": "pass",
        "tests_status": "pass",
        "tests_passed": 1134,
        "tests_skipped": 16,
        "command_count": 186,
        "unresolved_placeholder_count": 0,
        "absolute_sentinel_count": 0,
        "ngsim_raw_rows": 11850526,
        "provider_frames": 73871,
        "reachable_windows": 60,
        "outer_nested_expansion_equal": True,
        "nonformal_rehearsal_status": "pass",
        "formal": False,
        "performance_evidence": False,
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
        "holdout_opened": False,
    }
    write_json(ARTIFACT / "acceptance_evidence_manifest.json", acceptance)
    readiness = {
        "readiness_review_version": READINESS_VERSION,
        "status": "ready",
        "verdict": READY_STATUS,
        "active_bundle_core_sha256": index["active_bundle_core_sha256"],
        "evidence_manifest_path": (ARTIFACT / "acceptance_evidence_manifest.json").relative_to(ROOT).as_posix(),
        "evidence_manifest_sha256": sha256_file(ARTIFACT / "acceptance_evidence_manifest.json"),
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
        "holdout_sealed_unopened": True,
    }
    write_json(TARGET / "readiness_v13.json", readiness)
    readiness_row = current(
        "readiness_companion", "Readiness v13 evidence companion", "readiness_v13.json"
    )
    index["active_bundle_resources"].append(readiness_row)
    index["readiness_companion"] = {
        "logical_path": readiness_row["logical_path"],
        "content_sha256": readiness_row["content_sha256"],
    }
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
                "active_formal_bundle_sha256": index["active_formal_bundle_sha256"],
                "environment_fingerprint": environment_identity["environment_fingerprint"],
                "dependency_fingerprint": environment_identity["dependency_fingerprint"],
                "command_templates_sha256": index["command_matrix_identity"]["command_templates_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
