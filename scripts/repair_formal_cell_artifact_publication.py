"""Freeze Protocol 2.6 cell publication, recovery, and completion semantics."""

from __future__ import annotations

import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.cache_baseline_fairness import (
    full_manifest_sha256,
    semantic_protocol_sha256,
)
from src.evaluators.formal_cell_transaction import (
    CELL_ARTIFACT_PUBLICATION_CONTRACT_VERSION,
    CELL_CHILD_OUTPUT_DESCRIPTOR_VERSION,
    FORMAL_CELL_LEDGER_VERSION,
)
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
from src.runtime.formal_protocol_capabilities import (
    FORMAL_PROTOCOL_CAPABILITY_ROUTING_CONTRACT_VERSION,
    protocol_capability_matrix,
)
from src.runtime.generated_checkpoint_resources import (
    FORMAL_GENERATED_CHECKPOINT_RESOURCE_IDENTITY_CONTRACT_VERSION,
)


SOURCE = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_5_20260905"
TARGET = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_6_20260905"
ARTIFACT = ROOT / (
    "artifacts/analysis/"
    "typed_model_cache_formal_cell_publication_repair_20260905_g14r15_v1"
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8-sig"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant in {path}: {token}")
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


def replace(value: Any) -> Any:
    if isinstance(value, str):
        return (
            value.replace(SOURCE.name, TARGET.name)
            .replace("protocol_v2_5_manifest.json", "protocol_v2_6_manifest.json")
            .replace("readiness_v17.json", "readiness_v18.json")
            .replace("Protocol 2.5", "Protocol 2.6")
        )
    if isinstance(value, list):
        return [replace(item) for item in value]
    if isinstance(value, dict):
        return {key: replace(item) for key, item in value.items()}
    return value


def semantic(value: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(value)
    payload.pop("semantic_sha256", None)
    payload["semantic_sha256"] = canonical_sha256(payload)
    return payload


def routing_contract() -> dict[str, Any]:
    return semantic(
        {
            "version": FORMAL_PROTOCOL_CAPABILITY_ROUTING_CONTRACT_VERSION,
            "capability_matrix": protocol_capability_matrix(),
            "authority": "src.runtime.formal_protocol_capabilities",
            "unknown_versions_fail_closed": True,
            "historical_live_execution_forbidden": True,
            "holdout_capability": False,
        }
    )


def generated_contract() -> dict[str, Any]:
    return semantic(
        {
            "version": FORMAL_GENERATED_CHECKPOINT_RESOURCE_IDENTITY_CONTRACT_VERSION,
            "publication": {
                "source": "one immutable checkpoint_freeze completed terminal",
                "whole_growing_phase_ledger_hash_is_identity": False,
                "create_only_or_exact_identity_match": True,
                "rebuild_from_legal_freeze_only": True,
            },
            "recovery": {
                "freeze_completed_registry_missing": "build_and_create_once",
                "registry_already_published": "validate_exact_and_idempotently_succeed",
                "same_name_different_content": "reject",
                "cross_run_or_missing_freeze": "reject",
            },
            "historical_invalid_run_recovery_allowed": False,
        }
    )


def publication_contract() -> dict[str, Any]:
    return semantic(
        {
            "version": CELL_ARTIFACT_PUBLICATION_CONTRACT_VERSION,
            "cell_ledger_version": FORMAL_CELL_LEDGER_VERSION,
            "child_output_descriptor_version": CELL_CHILD_OUTPUT_DESCRIPTOR_VERSION,
            "logical_cell_identity": "phase plus frozen matrix coordinates",
            "attempt_staging_root": ".staging/<phase>/<cell_id>/attempt_<n>",
            "child_output_location": "attempt staging child_output or typed producer root",
            "validated_artifact_root": "exact structured descriptor or one immediate child",
            "committed_destination": "durable phase/cell logical path",
            "required_payload": "phase-specific exact files plus complete inventory",
            "publication_order": [
                "child_return_zero",
                "resolve_exact_child_output",
                "validate_identity_required_payload_inventory_and_no_symlink",
                "rebase_internal_paths_to_committed_destination",
                "write_committed_marker_in_staging",
                "atomic_rename",
                "append_committed_cell_terminal",
                "phase_completion_candidate_then_terminal",
            ],
            "fail_closed": [
                "missing_or_duplicate_child_output",
                "wrong_setting_or_cross_cell",
                "path_escape_or_symlink",
                "payload_missing_or_hash_drift",
                "staging_or_failed_attempt_consumption",
                "duplicate_commit",
            ],
            "recovery": {
                "before_publish": "new bounded attempt; prior attempt remains non-consumable",
                "after_publish_before_terminal": "revalidate marker and payload then append terminal",
                "already_committed": "revalidate and skip without rerun",
            },
        }
    )


def rehearsal_profile() -> dict[str, Any]:
    return semantic(
        {
            "version": "1.0.0",
            "profile_id": "g14r15_cell_transaction_nonformal_v1",
            "formal": False,
            "performance_evidence": False,
            "holdout_capability": False,
            "training": {
                "seeds": [7],
                "episodes": 4,
                "updates": 4,
                "checkpoint_cadence": 4,
                "max_steps": 1,
                "max_workflows": 1,
                "max_mobility_rows": 1500,
            },
            "matrix": {
                "learned_agents": 10,
                "capacities": 3,
                "train_cells": 30,
                "cache_policy_cells": 3,
                "controller_cells": 3,
                "ablation_settings": 2,
                "support_settings": 11,
                "oracle_scalability_settings": 3,
                "phases": 13,
            },
            "uses_formal_cell_executor": True,
            "formal_profile_override_allowed": False,
            "future_formal_checkpoint_reuse_allowed": False,
        }
    )


def resource_row(logical_id: str, role: str, filename: str, semantic_hash=None):
    return build_resource_row(
        root=ROOT,
        logical_id=logical_id,
        role=role,
        relative_path=(TARGET / filename).relative_to(ROOT).as_posix(),
        version_scope="current_protocol_version",
        semantic_sha256=semantic_hash,
    )


def main() -> None:
    old_protocol = read_json(SOURCE / "protocol_v2_5_manifest.json")
    old_index = read_json(SOURCE / "protocol_index.json")
    TARGET.mkdir(parents=True, exist_ok=True)
    excluded = {
        "protocol_index.json",
        "protocol_v2_5_manifest.json",
        "readiness_v17.json",
        "formal_protocol_capability_routing_contract.json",
        "formal_generated_checkpoint_resource_identity_contract.json",
    }
    for source in sorted(SOURCE.glob("*.json")):
        if source.name not in excluded:
            write_json(TARGET / source.name, replace(read_json(source)))
    for path in TARGET.glob("nonformal_rehearsal_fairness_*.json"):
        payload = read_json(path)
        digest = semantic_protocol_sha256(payload)
        payload["identity"]["manifest_id"] = f"cbfm-{digest[:16]}"
        payload["hashes"]["semantic_protocol_sha256"] = digest
        payload["hashes"]["full_manifest_sha256"] = full_manifest_sha256(payload)
        write_json(path, payload)

    routing = routing_contract()
    generated = generated_contract()
    publication = publication_contract()
    rehearsal = rehearsal_profile()
    write_json(TARGET / "formal_protocol_capability_routing_contract.json", routing)
    write_json(TARGET / "formal_generated_checkpoint_resource_identity_contract.json", generated)
    write_json(TARGET / "formal_cell_artifact_publication_contract.json", publication)
    write_json(TARGET / "nonformal_cell_transaction_rehearsal_profile.json", rehearsal)

    protocol = replace(deepcopy(old_protocol))
    protocol.update(
        typed_model_cache_formal_protocol_version=ACTIVE_PROTOCOL_VERSION,
        protocol_id=ACTIVE_PROTOCOL_ID,
        status="frozen_pre_execution_cell_artifact_publication_closure",
        formal_protocol_capability_routing_contract={
            "version": routing["version"],
            "semantic_sha256": routing["semantic_sha256"],
            "authoritative_module": "src.runtime.formal_protocol_capabilities",
        },
        formal_generated_checkpoint_resource_identity_contract={
            "version": generated["version"],
            "semantic_sha256": generated["semantic_sha256"],
            "authoritative_module": "src.runtime.generated_checkpoint_resources",
        },
        formal_cell_artifact_publication_contract={
            "version": publication["version"],
            "semantic_sha256": publication["semantic_sha256"],
            "authoritative_module": "src.evaluators.formal_cell_transaction",
        },
    )
    protocol["identity"].update(
        formal_protocol_capability_routing_contract_semantic_sha256=routing["semantic_sha256"],
        formal_generated_checkpoint_resource_identity_contract_semantic_sha256=generated["semantic_sha256"],
        formal_cell_artifact_publication_contract_semantic_sha256=publication["semantic_sha256"],
    )
    protocol["execution_contract"].update(
        cell_artifact_publication={
            "version": publication["version"],
            "semantic_sha256": publication["semantic_sha256"],
            "formal_and_rehearsal_shared_executor": True,
        },
        nonformal_rehearsal_profile={
            "path": (TARGET / "nonformal_cell_transaction_rehearsal_profile.json").relative_to(ROOT).as_posix(),
            "semantic_sha256": rehearsal["semantic_sha256"],
            "cannot_override_formal_profile": True,
        },
    )
    protocol["supersession"].update(
        supersedes_version="2.5.0",
        old_protocol_status="historical_audit_only_after_cell_publication_contract_mismatch",
        old_protocol_semantic_sha256=old_protocol["hashes"]["semantic_sha256"],
        scientific_fields_changed=False,
        execution_contract_fields_changed=True,
        formal_performance_observed=False,
        repair_scope=[
            "cell artifact publication path and inventory",
            "checkpoint registry publication recovery",
            "gate failure propagation and completion authorization",
        ],
        g14r15_authorization_boundary={
            "status": "PRE-EXECUTION AUTHORIZATION WITHHELD / CELL_ARTIFACT_PUBLICATION_CONTRACT_MISMATCH",
            "g14c_v14_created": False,
            "g14c_v14_number_consumed": False,
            "formal_training_count": 0,
            "formal_checkpoint_count": 0,
            "formal_performance_count": 0,
            "holdout_opened": False,
        },
    )
    protocol["active_formal_bundle_contract"]["unique_active_index"] = (
        TARGET / "protocol_index.json"
    ).relative_to(ROOT).as_posix()
    protocol["paper_claim_boundary"] = (
        "G14R15 is non-formal execution-contract evidence only; formal training, "
        "checkpoint, performance, holdout, G14C v14, G14D, and G15 are unexecuted."
    )
    protocol = attach_hashes(protocol)
    write_json(TARGET / "protocol_v2_6_manifest.json", protocol)

    skip = {
        "readiness_companion",
        "formal_protocol_capability_routing_contract",
        "formal_generated_checkpoint_resource_identity_contract",
    }
    current = []
    for row in old_index["active_bundle_resources"]:
        if row.get("version_scope") != "current_protocol_version":
            continue
        logical_id = str(row["logical_id"])
        if logical_id in skip:
            continue
        filename = Path(str(row["logical_path"])).name
        semantic_hash = row.get("semantic_sha256")
        if logical_id == "protocol_manifest":
            filename = "protocol_v2_6_manifest.json"
            semantic_hash = protocol["hashes"]["semantic_sha256"]
        current.append(resource_row(logical_id, str(row["role"]), filename, semantic_hash))
    current += [
        resource_row("formal_protocol_capability_routing_contract", "fail-closed Formal Protocol capability routing contract", "formal_protocol_capability_routing_contract.json", routing["semantic_sha256"]),
        resource_row("formal_generated_checkpoint_resource_identity_contract", "run-generated checkpoint resource identity contract", "formal_generated_checkpoint_resource_identity_contract.json", generated["semantic_sha256"]),
        resource_row("formal_cell_artifact_publication_contract", "cell artifact publication transaction contract", "formal_cell_artifact_publication_contract.json", publication["semantic_sha256"]),
        resource_row("nonformal_cell_transaction_rehearsal_profile", "non-formal transaction rehearsal profile", "nonformal_cell_transaction_rehearsal_profile.json", rehearsal["semantic_sha256"]),
    ]
    shared = [
        deepcopy(row) for row in old_index["active_bundle_resources"]
        if row.get("version_scope") == "shared_historical_stable"
    ]
    index: dict[str, Any] = {
        "active_formal_bundle_contract_version": ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION,
        "active_bundle_resource_resolution_contract_version": ACTIVE_BUNDLE_RESOURCE_RESOLUTION_CONTRACT_VERSION,
        "protocol_index_version": ACTIVE_PROTOCOL_VERSION,
        "status": "NOT_READY_PENDING_G14R15_ACCEPTANCE",
        "protocol_identity": {
            "protocol_id": ACTIVE_PROTOCOL_ID,
            "protocol_version": ACTIVE_PROTOCOL_VERSION,
            "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
            "protocol_full_sha256": protocol["hashes"]["full_sha256"],
        },
        "execution_commit_binding": deepcopy(old_index["execution_commit_binding"]),
        "environment_identity": deepcopy(old_index["environment_identity"]),
        "command_matrix_identity": {
            "command_templates_sha256": canonical_sha256(protocol["execution_contract"]["command_templates"]),
            "outer_nested_expansion_equality_required": True,
        },
        "holdout_seal": deepcopy(protocol["holdout_execution_contract"]),
        "active_bundle_resources": [*current, *shared],
    }
    index["active_bundle_core_sha256"] = canonical_sha256(active_bundle_core_projection(index))
    evidence_path = ARTIFACT / "acceptance_evidence_manifest.json"
    if evidence_path.is_file():
        evidence = read_json(evidence_path)
        if (
            evidence.get("status") == "pass"
            and evidence.get("active_bundle_core_sha256") == index["active_bundle_core_sha256"]
            and all(evidence.get(key) == 0 for key in (
                "formal_training_count", "formal_checkpoint_count", "formal_performance_count"
            ))
            and evidence.get("holdout_sealed_unopened_unconsumed") is True
            and all(value == "pass" for value in evidence.get("checks", {}).values())
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
                "holdout_capability": False,
                "holdout_sealed_unopened_unconsumed": True,
            }
            write_json(TARGET / "readiness_v18.json", readiness)
            row = resource_row("readiness_companion", "Readiness v18 evidence companion", "readiness_v18.json")
            index["active_bundle_resources"].append(row)
            index["readiness_companion"] = {
                "logical_path": row["logical_path"],
                "content_sha256": row["content_sha256"],
            }
            index["status"] = READY_STATUS
    index["active_formal_bundle_sha256"] = canonical_sha256(ready_index_projection(index))
    write_json(TARGET / "protocol_index.json", index)
    print(json.dumps({
        "status": index["status"],
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "protocol_full_sha256": protocol["hashes"]["full_sha256"],
        "active_bundle_core_sha256": index["active_bundle_core_sha256"],
        "active_formal_bundle_sha256": index["active_formal_bundle_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
