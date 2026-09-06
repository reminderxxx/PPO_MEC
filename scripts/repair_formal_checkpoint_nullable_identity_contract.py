"""Freeze Protocol 2.8 after G14C v15 checkpoint nullable-identity repair."""

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


SOURCE = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_7_20260905"
TARGET = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_8_20260906"
ARTIFACT = ROOT / (
    "artifacts/analysis/"
    "typed_model_cache_formal_checkpoint_identity_repair_20260906_g14r17_v1"
)
OLD_RUN = ROOT / (
    "artifacts/experiments/typed_model_cache_formal/"
    "typed_model_cache_formal_20260905_213344_g14c_v15"
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
            .replace("protocol_v2_7_manifest.json", "protocol_v2_8_manifest.json")
            .replace("readiness_v19.json", "readiness_v20.json")
            .replace("Protocol 2.7", "Protocol 2.8")
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


def main() -> None:
    old_protocol = read_json(SOURCE / "protocol_v2_7_manifest.json")
    old_index = read_json(SOURCE / "protocol_index.json")
    excluded = {
        "protocol_v2_7_manifest.json",
        "protocol_index.json",
        "readiness_v19.json",
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
        status="frozen_pre_execution_checkpoint_nullable_identity_repaired",
        formal_protocol_capability_routing_contract={
            "version": routing["version"],
            "semantic_sha256": routing["semantic_sha256"],
            "authoritative_module": "src.runtime.formal_protocol_capabilities",
        },
    )
    protocol["identity"][
        "formal_protocol_capability_routing_contract_semantic_sha256"
    ] = routing["semantic_sha256"]
    failure_audit = ARTIFACT / "g14c_v15_failure_audit.json"
    invalid = {
        "run_id": OLD_RUN.name,
        "run_root": OLD_RUN.relative_to(ROOT).as_posix(),
        "status": "INVALID_PROTOCOL_OR_IMPLEMENTATION",
        "failure_boundary": "invalid_after_training_and_dev_evaluation_before_selection_publication",
        "failure_audit_path": failure_audit.relative_to(ROOT).as_posix(),
        "failure_audit_sha256": sha256_file(failure_audit) if failure_audit.is_file() else None,
        "phase_ledger_sha256": "7630c3ec3ef71cc306eeef806f0697f68fd13080124711d43b7dec02ade7b3ab",
        "cell_ledger_sha256": "a70b9e80e4024eb622b6766704a644a7290e79a3238db13ad5e9e8f45ca095b9",
        "checkpoint_candidates_sha256": "fe309ba793ace56a9c9ae9ae41b804fb311618ab1222c850d03f138ce10448a3",
        "training_cells_committed": 150,
        "dev_cells_committed": 24,
        "candidate_checkpoint_count": 1200,
        "dev_selection_published": False,
        "checkpoint_freeze_published": False,
        "formal_performance_count": 0,
        "holdout_opened": False,
        "immutable_old_run": True,
        "resume_allowed": False,
        "retry_allowed": False,
        "salvage_allowed": False,
        "checkpoint_reuse_allowed": False,
        "candidate_reuse_allowed": False,
        "partial_dev_reuse_allowed": False,
    }
    invalid_runs = protocol["supersession"]["invalid_execution_runs"]
    if not any(row.get("run_id") == invalid["run_id"] for row in invalid_runs):
        invalid_runs.append(invalid)
    protocol["supersession"].update(
        supersedes_version="2.7.0",
        old_protocol_status="historical_audit_only_after_checkpoint_nullable_identity_failure",
        old_protocol_semantic_sha256=old_protocol["hashes"]["semantic_sha256"],
        scientific_fields_changed=False,
        execution_contract_fields_changed=True,
        formal_performance_observed=False,
        repair_scope=[
            "checkpoint nullable identity producer projection",
            "post-serialization checkpoint identity validation",
            "pre-benchmark, pre-sort, freeze, and provenance strict consumption",
            "test-only 1200-coordinate and real save/read-back acceptance",
        ],
        g14r17_authorization_boundary={
            "g14c_v16_created": False,
            "formal_training_count": 0,
            "formal_performance_count": 0,
            "holdout_opened": False,
            "status": "EXECUTION_CONTRACT_REPAIR_ONLY",
        },
    )
    protocol["active_formal_bundle_contract"]["unique_active_index"] = (
        TARGET / "protocol_index.json"
    ).relative_to(ROOT).as_posix()
    protocol["paper_claim_boundary"] = (
        "G14R17 validates checkpoint identity serialization and strict consumption only; "
        "no G14C v16, formal training, formal performance, holdout, G14D, or G15 was executed."
    )
    protocol = attach_hashes(protocol)
    write_json(TARGET / "protocol_v2_8_manifest.json", protocol)

    skip = {"readiness_companion", "formal_protocol_capability_routing_contract"}
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
            filename = "protocol_v2_8_manifest.json"
            semantic_hash = protocol["hashes"]["semantic_sha256"]
        current.append(resource_row(logical_id, str(row["role"]), filename, semantic_hash))
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
    index = {
        "active_formal_bundle_contract_version": ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION,
        "active_bundle_resource_resolution_contract_version": ACTIVE_BUNDLE_RESOURCE_RESOLUTION_CONTRACT_VERSION,
        "protocol_index_version": ACTIVE_PROTOCOL_VERSION,
        "status": "NOT_READY_PENDING_G14R17_ACCEPTANCE",
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
            and evidence.get("checkpoint_identity_acceptance_status") == "pass"
            and evidence.get("formal_training_count") == 0
            and evidence.get("formal_performance_count") == 0
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
                "formal_performance_count": 0,
                "holdout_capability": False,
                "holdout_sealed_unopened_unconsumed": True,
            }
            write_json(TARGET / "readiness_v20.json", readiness)
            row = resource_row(
                "readiness_companion", "Readiness v20 evidence companion", "readiness_v20.json"
            )
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
