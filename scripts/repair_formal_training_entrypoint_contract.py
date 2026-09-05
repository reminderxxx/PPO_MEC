"""Freeze Protocol 2.7 after the G14C v14 training-contract repair."""

from __future__ import annotations

import json
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


SOURCE = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_6_20260905"
TARGET = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_7_20260905"
ARTIFACT = ROOT / (
    "artifacts/analysis/"
    "typed_model_cache_formal_training_entrypoint_repair_20260905_g14r16_v1"
)
FAILURE_AUDIT = ROOT / (
    "artifacts/experiments/typed_model_cache_formal/"
    "typed_model_cache_formal_20260905_185105_g14c_v14/audit/failure_audit.json"
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def replace(value: Any) -> Any:
    if isinstance(value, str):
        return (
            value.replace(SOURCE.name, TARGET.name)
            .replace("protocol_v2_6_manifest.json", "protocol_v2_7_manifest.json")
            .replace("readiness_v18.json", "readiness_v19.json")
            .replace("Protocol 2.6", "Protocol 2.7")
        )
    if isinstance(value, list):
        return [replace(item) for item in value]
    if isinstance(value, dict):
        return {key: replace(item) for key, item in value.items()}
    return value


def semantic(payload: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(payload)
    result.pop("semantic_sha256", None)
    result["semantic_sha256"] = canonical_sha256(result)
    return result


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


def resource_row(
    logical_id: str, role: str, filename: str, semantic_hash: str | None = None
) -> dict[str, Any]:
    return build_resource_row(
        root=ROOT,
        logical_id=logical_id,
        role=role,
        relative_path=(TARGET / filename).relative_to(ROOT).as_posix(),
        version_scope="current_protocol_version",
        semantic_sha256=semantic_hash,
    )


def main() -> None:
    old_protocol = read_json(SOURCE / "protocol_v2_6_manifest.json")
    old_index = read_json(SOURCE / "protocol_index.json")
    failure = read_json(FAILURE_AUDIT)
    if sha256_file(FAILURE_AUDIT) != (
        "d323c122230795585bbadb16f8650f5e395716b145935a4a41cf5fafe21e2608"
    ):
        raise ValueError("G14C v14 failure audit SHA-256 drift")
    TARGET.mkdir(parents=True, exist_ok=True)
    excluded = {
        "protocol_index.json",
        "protocol_v2_6_manifest.json",
        "readiness_v18.json",
        "formal_protocol_capability_routing_contract.json",
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
    write_json(TARGET / "formal_protocol_capability_routing_contract.json", routing)
    protocol = replace(deepcopy(old_protocol))
    protocol.update(
        typed_model_cache_formal_protocol_version=ACTIVE_PROTOCOL_VERSION,
        protocol_id=ACTIVE_PROTOCOL_ID,
        status="frozen_pre_execution_training_entrypoint_contract_repaired",
        formal_protocol_capability_routing_contract={
            "version": routing["version"],
            "semantic_sha256": routing["semantic_sha256"],
            "authoritative_module": "src.runtime.formal_protocol_capabilities",
        },
    )
    protocol["identity"][
        "formal_protocol_capability_routing_contract_semantic_sha256"
    ] = routing["semantic_sha256"]
    invalid = {
        "run_id": failure["artifact_run_id"],
        "run_root": (
            "artifacts/experiments/typed_model_cache_formal/"
            f"{failure['artifact_run_id']}"
        ),
        "status": "INVALID_PROTOCOL_OR_IMPLEMENTATION",
        "failure_boundary": "invalid_during_first_training_cell_before_episode_zero",
        "failure_audit_path": FAILURE_AUDIT.relative_to(ROOT).as_posix(),
        "failure_audit_sha256": sha256_file(FAILURE_AUDIT),
        "training_cells_executed": 0,
        "episode_count": 0,
        "environment_interaction_count": 0,
        "update_count": 0,
        "candidate_checkpoint_count": 0,
        "checkpoint_reuse_allowed": False,
        "checkpoint_salvage_allowed": False,
        "dev_performance_count": 0,
        "formal_performance_count": 0,
        "immutable_old_run": True,
        "resume_allowed": False,
        "retry_allowed": False,
        "salvage_allowed": False,
        "legacy_phase_finalize_allowed": False,
        "ledger_or_marker_reuse_allowed": False,
    }
    invalid_runs = protocol["supersession"]["invalid_execution_runs"]
    if not any(row.get("run_id") == invalid["run_id"] for row in invalid_runs):
        invalid_runs.append(invalid)
    protocol["supersession"].update(
        supersedes_version="2.6.0",
        old_protocol_status="historical_audit_only_after_training_contract_name_error",
        old_protocol_semantic_sha256=old_protocol["hashes"]["semantic_sha256"],
        scientific_fields_changed=False,
        execution_contract_fields_changed=True,
        formal_performance_observed=False,
        repair_scope=[
            "formal training resolver nullable-identity NameError",
            "active 150-cell production training-entrypoint initialization acceptance",
            "readiness consumption of commit- and bundle-bound entrypoint evidence",
        ],
        g14r16_authorization_boundary={
            "g14c_v15_created": False,
            "formal_training_count": 0,
            "formal_checkpoint_count": 0,
            "formal_performance_count": 0,
            "holdout_opened": False,
            "status": "EXECUTION_CONTRACT_REPAIR_ONLY",
        },
    )
    protocol["active_formal_bundle_contract"]["unique_active_index"] = (
        TARGET / "protocol_index.json"
    ).relative_to(ROOT).as_posix()
    protocol["paper_claim_boundary"] = (
        "G14R16 validates formal training initialization only; no formal episode, "
        "checkpoint, performance, holdout, G14C v15, G14D, or G15 was executed."
    )
    protocol = attach_hashes(protocol)
    write_json(TARGET / "protocol_v2_7_manifest.json", protocol)

    skip = {"readiness_companion", "formal_protocol_capability_routing_contract"}
    current: list[dict[str, Any]] = []
    for row in old_index["active_bundle_resources"]:
        if row.get("version_scope") != "current_protocol_version":
            continue
        logical_id = str(row["logical_id"])
        if logical_id in skip:
            continue
        filename = Path(str(row["logical_path"])).name
        semantic_hash = row.get("semantic_sha256")
        if logical_id == "protocol_manifest":
            filename = "protocol_v2_7_manifest.json"
            semantic_hash = protocol["hashes"]["semantic_sha256"]
        current.append(
            resource_row(logical_id, str(row["role"]), filename, semantic_hash)
        )
    current.append(
        resource_row(
            "formal_protocol_capability_routing_contract",
            "fail-closed Formal Protocol capability routing contract",
            "formal_protocol_capability_routing_contract.json",
            routing["semantic_sha256"],
        )
    )
    shared = [
        deepcopy(row)
        for row in old_index["active_bundle_resources"]
        if row.get("version_scope") == "shared_historical_stable"
    ]
    index: dict[str, Any] = {
        "active_formal_bundle_contract_version": ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION,
        "active_bundle_resource_resolution_contract_version": (
            ACTIVE_BUNDLE_RESOURCE_RESOLUTION_CONTRACT_VERSION
        ),
        "protocol_index_version": ACTIVE_PROTOCOL_VERSION,
        "status": "NOT_READY_PENDING_G14R16_ACCEPTANCE",
        "protocol_identity": {
            "protocol_id": ACTIVE_PROTOCOL_ID,
            "protocol_version": ACTIVE_PROTOCOL_VERSION,
            "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
            "protocol_full_sha256": protocol["hashes"]["full_sha256"],
        },
        "execution_commit_binding": deepcopy(old_index["execution_commit_binding"]),
        "environment_identity": deepcopy(old_index["environment_identity"]),
        "command_matrix_identity": {
            "command_templates_sha256": canonical_sha256(
                protocol["execution_contract"]["command_templates"]
            ),
            "outer_nested_expansion_equality_required": True,
        },
        "holdout_seal": deepcopy(protocol["holdout_execution_contract"]),
        "active_bundle_resources": [*current, *shared],
    }
    index["active_bundle_core_sha256"] = canonical_sha256(
        active_bundle_core_projection(index)
    )
    evidence_path = ARTIFACT / "acceptance_evidence_manifest.json"
    if evidence_path.is_file():
        evidence = read_json(evidence_path)
        required = (
            evidence.get("status") == "pass"
            and evidence.get("active_bundle_core_sha256")
            == index["active_bundle_core_sha256"]
            and evidence.get("formal_training_entrypoint_acceptance_status") == "pass"
            and all(
                evidence.get(key) == 0
                for key in (
                    "formal_training_count",
                    "formal_checkpoint_count",
                    "formal_performance_count",
                )
            )
            and evidence.get("holdout_sealed_unopened_unconsumed") is True
            and all(value == "pass" for value in evidence.get("checks", {}).values())
        )
        if required:
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
            write_json(TARGET / "readiness_v19.json", readiness)
            row = resource_row(
                "readiness_companion",
                "Readiness v19 evidence companion",
                "readiness_v19.json",
            )
            index["active_bundle_resources"].append(row)
            index["readiness_companion"] = {
                "logical_path": row["logical_path"],
                "content_sha256": row["content_sha256"],
            }
            index["status"] = READY_STATUS
    index["active_formal_bundle_sha256"] = canonical_sha256(
        ready_index_projection(index)
    )
    write_json(TARGET / "protocol_index.json", index)
    print(
        json.dumps(
            {
                "status": index["status"],
                "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
                "protocol_full_sha256": protocol["hashes"]["full_sha256"],
                "active_bundle_core_sha256": index["active_bundle_core_sha256"],
                "active_formal_bundle_sha256": index["active_formal_bundle_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
