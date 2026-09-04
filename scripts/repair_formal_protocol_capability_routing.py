"""Freeze Protocol 2.4 and its fail-closed capability-routing contract."""

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


SOURCE = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_3_20260903"
TARGET = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_4_20260905"
ARTIFACT = ROOT / (
    "artifacts/analysis/"
    "typed_model_cache_formal_preflight_validator_dispatch_repair_20260905_g14r13_v1"
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
            .replace("protocol_v2_3_manifest.json", "protocol_v2_4_manifest.json")
            .replace("readiness_v15.json", "readiness_v16.json")
            .replace("G14R12", "G14R13")
            .replace("G14C v13", "G14C v14")
        )
    if isinstance(value, list):
        return [replace_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: replace_paths(item) for key, item in value.items()}
    return value


def semantic(value: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(value)
    payload.pop("semantic_sha256", None)
    payload["semantic_sha256"] = canonical_sha256(payload)
    return payload


def capability_contract() -> dict[str, Any]:
    return semantic(
        {
            "version": FORMAL_PROTOCOL_CAPABILITY_ROUTING_CONTRACT_VERSION,
            "capability_matrix": protocol_capability_matrix(),
            "authority": "src.runtime.formal_protocol_capabilities",
            "routing_rules": {
                "explicit_per_version_registration": True,
                "unknown_versions_fail_closed": True,
                "major_or_lexical_version_inference_forbidden": True,
                "active_default_expansion_context_fallback_allowed": False,
                "historical_audit_only_live_execution_allowed": False,
                "implicit_python_or_cwd_discovery_allowed": False,
                "holdout_capability": False,
            },
        }
    )


def current_row(
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


def main() -> None:
    old_protocol = read_json(SOURCE / "protocol_v2_3_manifest.json")
    old_index = read_json(SOURCE / "protocol_index.json")
    TARGET.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(SOURCE.glob("*.json")):
        if source_path.name in {
            "protocol_index.json",
            "protocol_v2_3_manifest.json",
            "readiness_v15.json",
        }:
            continue
        write_json(TARGET / source_path.name, replace_paths(read_json(source_path)))
    for fairness_path in sorted(TARGET.glob("nonformal_rehearsal_fairness_*.json")):
        fairness = read_json(fairness_path)
        semantic_hash = semantic_protocol_sha256(fairness)
        fairness["identity"]["manifest_id"] = f"cbfm-{semantic_hash[:16]}"
        fairness["hashes"]["semantic_protocol_sha256"] = semantic_hash
        fairness["hashes"]["full_manifest_sha256"] = full_manifest_sha256(fairness)
        write_json(fairness_path, fairness)

    routing = capability_contract()
    write_json(TARGET / "formal_protocol_capability_routing_contract.json", routing)

    protocol = replace_paths(deepcopy(old_protocol))
    protocol.update(
        typed_model_cache_formal_protocol_version=ACTIVE_PROTOCOL_VERSION,
        protocol_id=ACTIVE_PROTOCOL_ID,
        status="frozen_pre_execution_capability_routing_contract",
        formal_protocol_capability_routing_contract={
            "version": routing["version"],
            "semantic_sha256": routing["semantic_sha256"],
            "authoritative_module": "src.runtime.formal_protocol_capabilities",
            "active_default_context_fallback_allowed": False,
        },
    )
    protocol["identity"][
        "formal_protocol_capability_routing_contract_semantic_sha256"
    ] = routing["semantic_sha256"]
    protocol["supersession"].update(
        supersedes_version="2.3.0",
        old_protocol_status=(
            "audit_only_after_pre_execution_validator_version_dispatch_mismatch"
        ),
        old_protocol_semantic_sha256=old_protocol["hashes"]["semantic_sha256"],
        scientific_fields_changed=False,
        execution_contract_fields_changed=True,
        formal_performance_observed=False,
        repair_scope=[
            "centralize explicit fail-closed Protocol capability routing",
            "force active nested preflight to consume persisted resolved context",
            "supersede Protocol 2.3 and Readiness v15 as audit-only",
        ],
        g14c_v13_pre_execution_stop={
            "classification": (
                "PRE_EXECUTION_STOP / VALIDATOR_VERSION_DISPATCH_MISMATCH"
            ),
            "durable_run_root_created": False,
            "clean_execution_worktree_created": False,
            "phase_or_cell_ledger_created": False,
            "preflight_child_executed": False,
            "tests_executed": False,
            "formal_training_count": 0,
            "formal_checkpoint_count": 0,
            "formal_performance_count": 0,
            "holdout_opened": False,
            "invalid_run_record_created": False,
            "checkpoint_denylist_entry_created": False,
        },
    )
    protocol["active_formal_bundle_contract"]["unique_active_index"] = (
        TARGET / "protocol_index.json"
    ).relative_to(ROOT).as_posix()
    protocol["active_formal_bundle_contract"]["hash_graph"] = [
        "resource content hashes -> active_bundle_core_sha256",
        "core plus G14R13 acceptance evidence -> Readiness v16 content hash",
        "ready index plus Readiness content hash -> active_formal_bundle_sha256",
    ]
    context = protocol["execution_contract"]["default_expansion_context"]
    context["active_protocol_index_path"] = (
        TARGET / "protocol_index.json"
    ).relative_to(ROOT).as_posix()
    context["protocol_path"] = (
        TARGET / "protocol_v2_4_manifest.json"
    ).relative_to(ROOT).as_posix()
    protocol["paper_claim_boundary"] = (
        "G14R13 is outcome-blind execution-contract evidence only; formal training, "
        "checkpoint, performance, holdout, G14C v14, G14D, and G15 remain unexecuted."
    )
    protocol = attach_hashes(protocol)
    write_json(TARGET / "protocol_v2_4_manifest.json", protocol)

    current_rows: list[dict[str, Any]] = []
    for old_row in old_index["active_bundle_resources"]:
        if old_row.get("version_scope") != "current_protocol_version":
            continue
        logical_id = str(old_row["logical_id"])
        if logical_id == "readiness_companion":
            continue
        filename = Path(str(old_row["logical_path"])).name
        if logical_id == "protocol_manifest":
            filename = "protocol_v2_4_manifest.json"
            semantic_hash = protocol["hashes"]["semantic_sha256"]
        else:
            semantic_hash = old_row.get("semantic_sha256")
        current_rows.append(
            current_row(logical_id, str(old_row["role"]), filename, semantic_hash)
        )
    current_rows.insert(
        1,
        current_row(
            "formal_protocol_capability_routing_contract",
            "fail-closed Formal Protocol capability routing contract",
            "formal_protocol_capability_routing_contract.json",
            routing["semantic_sha256"],
        ),
    )
    shared_rows = [
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
        "status": "NOT_READY_PENDING_G14R13_ACCEPTANCE",
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
        "active_bundle_resources": [*current_rows, *shared_rows],
    }
    index["active_bundle_core_sha256"] = canonical_sha256(
        active_bundle_core_projection(index)
    )
    evidence_path = ARTIFACT / "acceptance_evidence_manifest.json"
    if evidence_path.is_file():
        evidence = read_json(evidence_path)
        required = {
            "root_cause_analysis",
            "g14c_v13_pre_execution_stop",
            "protocol_capability_matrix",
            "producer_consumer_matrix",
            "protocol_v2_4_diff_audit",
            "exact_wrapper_reproduction_before_fix",
            "exact_wrapper_preflight_after_fix",
            "negative_validation",
            "clean_candidate_validation",
            "phase_chain_rehearsal",
            "full_repository_pytest",
            "smoke_test",
            "strict_serialization_and_integrity",
            "protected_files",
        }
        if (
            evidence.get("status") == "pass"
            and evidence.get("active_bundle_core_sha256")
            == index["active_bundle_core_sha256"]
            and set(evidence.get("checks", {})) == required
            and all(value == "pass" for value in evidence["checks"].values())
            and evidence.get("formal_training_count") == 0
            and evidence.get("formal_checkpoint_count") == 0
            and evidence.get("formal_performance_count") == 0
            and evidence.get("holdout_capability") is False
            and evidence.get("holdout_sealed_unopened_unconsumed") is True
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
            write_json(TARGET / "readiness_v16.json", readiness)
            readiness_row = current_row(
                "readiness_companion",
                "Readiness v16 evidence companion",
                "readiness_v16.json",
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
                "status": index["status"],
                "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
                "protocol_full_sha256": protocol["hashes"]["full_sha256"],
                "capability_contract_semantic_sha256": routing["semantic_sha256"],
                "active_bundle_core_sha256": index["active_bundle_core_sha256"],
                "active_formal_bundle_sha256": index["active_formal_bundle_sha256"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
