"""Freeze Protocol 2.5 generated-checkpoint identity and consumer closure."""

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
from src.runtime.generated_checkpoint_resources import (
    CAPACITY_MB,
    FORMAL_GENERATED_CHECKPOINT_RESOURCE_IDENTITY_CONTRACT_VERSION,
)


SOURCE = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_4_20260905"
TARGET = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_5_20260905"
ARTIFACT = ROOT / (
    "artifacts/analysis/"
    "typed_model_cache_formal_generated_resource_closure_20260905_g14r14_v1"
)
OLD_DIR = SOURCE.name
NEW_DIR = TARGET.name


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


def replace_paths(value: Any) -> Any:
    if isinstance(value, str):
        return (
            value.replace(OLD_DIR, NEW_DIR)
            .replace("protocol_v2_4_manifest.json", "protocol_v2_5_manifest.json")
            .replace("readiness_v16.json", "readiness_v17.json")
            .replace("G14R13", "G14R14")
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


def generated_contract() -> dict[str, Any]:
    return semantic(
        {
            "version": FORMAL_GENERATED_CHECKPOINT_RESOURCE_IDENTITY_CONTRACT_VERSION,
            "resource_layers": {
                "static_registry": "pre-run immutable inputs only",
                "generated_checkpoint_registry": (
                    "current-run checkpoint-freeze outputs only"
                ),
            },
            "publication": {
                "producer_phase": "checkpoint_freeze",
                "after_committed_terminal_only": True,
                "create_only": True,
                "atomic": True,
                "canonical_json": True,
                "finite_json": True,
                "static_registry_mutation_forbidden": True,
            },
            "identity_fields": [
                "logical_resource_id", "resource_role", "schema_version",
                "capacity_label", "capacity_mb", "durable_run_root_relative_path",
                "size_bytes", "content_sha256", "protocol_semantic_sha256",
                "protocol_full_sha256", "active_formal_bundle_sha256",
                "execution_commit", "resolved_execution_context_sha256",
                "agent_scientific_config_semantic_sha256",
                "formal_training_execution_binding_sha256", "dev_selection_sha256",
                "checkpoint_freeze_sha256", "source_phase_committed_ledger_identity",
                "current_run_id", "registry_canonical_sha256",
            ],
            "fail_closed": [
                "missing_or_duplicate_id", "static_generated_id_collision",
                "role_schema_capacity_drift", "content_hash_or_size_drift",
                "explicit_path_identity_conflict", "symlink_or_path_escape",
                "cross_run_or_staging_reference", "uncommitted_or_failed_phase",
                "stale_protocol_bundle_context_selection_or_freeze",
                "historical_g14c_v1_v13_reference", "post_publication_rewrite",
                "outer_nested_resource_argument_divergence",
            ],
            "checkpoint_coverage": {
                "every_manifest_entry_has_provenance": True,
                "every_provenance_entry_has_manifest_entry": True,
                "duplicates_missing_cross_capacity_and_escape_rejected": True,
            },
        }
    )


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


def _capacity_from_path(value: str) -> str:
    matches = [label for label in CAPACITY_MB if label in value]
    if len(matches) == 1:
        return matches[0]
    return "medium_576mb"


def _add_pair(argv: list[str], flag: str, value: str, *, before: str | None = None) -> None:
    if flag in argv:
        index = argv.index(flag)
        argv[index + 1] = value
        return
    index = argv.index(before) if before and before in argv else len(argv)
    argv[index:index] = [flag, value]


def _remove_portable_tail(argv: list[str]) -> None:
    if "--resource-registry-path" in argv:
        del argv[argv.index("--resource-registry-path"):]


def close_command_templates(protocol: dict[str, Any]) -> None:
    execution = protocol["execution_contract"]
    context = execution["default_expansion_context"]
    context.update(
        generated_checkpoint_registry_path=(
            "/ABSOLUTE/FORMAL_OUTPUT_ROOT/generated_checkpoint_resource_registry.json"
        ),
        generated_checkpoint_resource_contract_path=(
            TARGET / "formal_generated_checkpoint_resource_identity_contract.json"
        ).relative_to(ROOT).as_posix(),
        checkpoint_provenance_id="checkpoint_provenance.medium_576mb",
    )
    templates = execution["command_templates"]
    _remove_portable_tail(templates["checkpoint_freeze"]["argv"])
    for phase, spec in templates.items():
        for row in spec.get("matrix_contexts", []):
            path_source = str(
                row.get("runtime_config_path")
                or row.get("seed_checkpoint_manifest_path")
                or ""
            )
            capacity = _capacity_from_path(path_source)
            row["capacity_label"] = capacity
            row["runtime_config_resource_id"] = f"runtime_config.{capacity}"
            row["checkpoint_manifest_id"] = f"checkpoint_manifest.{capacity}"
            row["checkpoint_provenance_id"] = f"checkpoint_provenance.{capacity}"
    consumers = (
        "formal_controller", "formal_ablation", "formal_support",
        "formal_scalability", "formal_statistics",
    )
    for phase in consumers:
        argv = templates[phase]["argv"]
        _add_pair(argv, "--generated-checkpoint-registry-path", "{generated_checkpoint_registry_path}")
        _add_pair(argv, "--checkpoint-provenance-id", "{checkpoint_provenance_id}")
    for phase in ("formal_controller", "formal_ablation", "formal_support", "formal_scalability"):
        argv = templates[phase]["argv"]
        _add_pair(argv, "--protocol-path", "{protocol_path}")
        _add_pair(argv, "--resolved-execution-context-path", "{resolved_execution_context_path}")
    controller = templates["formal_controller"]["argv"]
    _add_pair(controller, "--formal-training-execution-binding-path", "{formal_training_execution_binding_path}")

    cache = templates["formal_cache_policy"]["argv"]
    marker = cache.index("--command")
    outer, child = cache[:marker], cache[marker + 1:]
    for flag, value in (
        ("--seed-checkpoint-manifest-path", "{seed_checkpoint_manifest_path}"),
        ("--checkpoint-provenance-manifest-path", "{checkpoint_provenance_manifest_path}"),
        ("--generated-checkpoint-registry-path", "{generated_checkpoint_registry_path}"),
        ("--checkpoint-provenance-id", "{checkpoint_provenance_id}"),
        ("--resolved-execution-context-path", "{resolved_execution_context_path}"),
        ("--formal-training-execution-binding-path", "{formal_training_execution_binding_path}"),
    ):
        _add_pair(outer, flag, value)
    child_pairs = (
        ("--resource-registry-path", "{resource_registry_path}"),
        ("--repository-root", "{repository_root}"),
        ("--data-root", "{data_root}"),
        ("--protocol-artifact-root", "{protocol_artifact_root}"),
        ("--checkpoint-root", "{checkpoint_root}"),
        ("--mobility-resource-id", "dataset.mobility.ngsim.vehicle_trajectories"),
        ("--workflow-resource-id", "dataset.workflow.alibaba2018.batch_task"),
        ("--window-plan-resource-id", "window_plan.typed_model_cache.formal"),
        ("--runtime-config-resource-id", "{runtime_config_resource_id}"),
        ("--fairness-manifest-resource-id", "{fairness_manifest_resource_id}"),
        ("--generated-checkpoint-registry-path", "{generated_checkpoint_registry_path}"),
        ("--checkpoint-manifest-id", "{checkpoint_manifest_id}"),
        ("--checkpoint-provenance-id", "{checkpoint_provenance_id}"),
        ("--protocol-path", "{protocol_path}"),
        ("--resolved-execution-context-path", "{resolved_execution_context_path}"),
        ("--formal-training-execution-binding-path", "{formal_training_execution_binding_path}"),
    )
    for flag, value in child_pairs:
        _add_pair(child, flag, value)
    templates["formal_cache_policy"]["argv"] = [*outer, "--command", *child]

    for phase in ("formal_gate", "integrity"):
        argv = templates[phase]["argv"]
        _add_pair(argv, "--resource-registry-path", "{resource_registry_path}")
        _add_pair(argv, "--generated-checkpoint-registry-path", "{generated_checkpoint_registry_path}")


def current_row(
    logical_id: str, role: str, filename: str, semantic_sha256: str | None = None
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
    old_protocol = read_json(SOURCE / "protocol_v2_4_manifest.json")
    old_index = read_json(SOURCE / "protocol_index.json")
    TARGET.mkdir(parents=True, exist_ok=True)
    for source in sorted(SOURCE.glob("*.json")):
        if source.name in {
            "protocol_index.json", "protocol_v2_4_manifest.json", "readiness_v16.json",
            "formal_protocol_capability_routing_contract.json",
        }:
            continue
        write_json(TARGET / source.name, replace_paths(read_json(source)))
    for fairness_path in sorted(TARGET.glob("nonformal_rehearsal_fairness_*.json")):
        fairness = read_json(fairness_path)
        semantic_hash = semantic_protocol_sha256(fairness)
        fairness["identity"]["manifest_id"] = f"cbfm-{semantic_hash[:16]}"
        fairness["hashes"]["semantic_protocol_sha256"] = semantic_hash
        fairness["hashes"]["full_manifest_sha256"] = full_manifest_sha256(fairness)
        write_json(fairness_path, fairness)
    routing = capability_contract()
    generated = generated_contract()
    write_json(TARGET / "formal_protocol_capability_routing_contract.json", routing)
    write_json(TARGET / "formal_generated_checkpoint_resource_identity_contract.json", generated)

    protocol = replace_paths(deepcopy(old_protocol))
    protocol.update(
        typed_model_cache_formal_protocol_version=ACTIVE_PROTOCOL_VERSION,
        protocol_id=ACTIVE_PROTOCOL_ID,
        status="frozen_pre_execution_generated_checkpoint_resource_closure",
        formal_protocol_capability_routing_contract={
            "version": routing["version"],
            "semantic_sha256": routing["semantic_sha256"],
            "authoritative_module": "src.runtime.formal_protocol_capabilities",
            "active_default_context_fallback_allowed": False,
        },
        formal_generated_checkpoint_resource_identity_contract={
            "version": generated["version"],
            "semantic_sha256": generated["semantic_sha256"],
            "authoritative_module": "src.runtime.generated_checkpoint_resources",
        },
    )
    protocol["identity"]["formal_protocol_capability_routing_contract_semantic_sha256"] = routing["semantic_sha256"]
    protocol["identity"]["formal_generated_checkpoint_resource_identity_contract_semantic_sha256"] = generated["semantic_sha256"]
    protocol["supersession"].update(
        supersedes_version="2.4.0",
        old_protocol_status="historical_audit_only_after_generated_resource_binding_gap",
        old_protocol_semantic_sha256=old_protocol["hashes"]["semantic_sha256"],
        scientific_fields_changed=False,
        execution_contract_fields_changed=True,
        formal_performance_observed=False,
        repair_scope=[
            "separate static and current-run generated checkpoint registries",
            "bind every downstream checkpoint consumer before file access",
            "repair capacity resource mapping at command generator level",
            "enforce exact execution counts and claim-evidence states",
        ],
        g14r14_authorization_boundary={
            "status": "PRE-EXECUTION AUTHORIZATION WITHHELD / DOWNSTREAM_RESOURCE_BINDING_NOT_CLOSED",
            "g14c_v14_created": False,
            "g14c_v14_number_consumed": False,
            "durable_formal_run_root_created": False,
            "formal_training_count": 0,
            "formal_checkpoint_count": 0,
            "formal_dev_count": 0,
            "formal_performance_count": 0,
            "holdout_capability": False,
            "holdout_opened": False,
        },
    )
    protocol["active_formal_bundle_contract"]["unique_active_index"] = (
        TARGET / "protocol_index.json"
    ).relative_to(ROOT).as_posix()
    protocol["active_formal_bundle_contract"]["hash_graph"] = [
        "resource content hashes -> active_bundle_core_sha256",
        "core plus G14R14 acceptance evidence -> Readiness v17 content hash",
        "ready index plus Readiness content hash -> active_formal_bundle_sha256",
    ]
    protocol["execution_contract"]["generated_checkpoint_registry_binding"] = {
        "version": generated["version"],
        "registry_canonical_hash_in_downstream_phase_and_cell_input": True,
        "checkpoint_read_before_registry_validation_forbidden": True,
    }
    close_command_templates(protocol)
    protocol["paper_claim_boundary"] = (
        "G14R14 is outcome-blind execution-contract evidence only; formal training, "
        "checkpoint, performance, holdout, G14C v14, G14D, and G15 remain unexecuted."
    )
    protocol = attach_hashes(protocol)
    write_json(TARGET / "protocol_v2_5_manifest.json", protocol)

    current_rows: list[dict[str, Any]] = []
    skip = {
        "readiness_companion", "formal_protocol_capability_routing_contract",
    }
    for old_row in old_index["active_bundle_resources"]:
        if old_row.get("version_scope") != "current_protocol_version":
            continue
        logical_id = str(old_row["logical_id"])
        if logical_id in skip:
            continue
        filename = Path(str(old_row["logical_path"])).name
        semantic_hash = old_row.get("semantic_sha256")
        if logical_id == "protocol_manifest":
            filename = "protocol_v2_5_manifest.json"
            semantic_hash = protocol["hashes"]["semantic_sha256"]
        current_rows.append(current_row(logical_id, str(old_row["role"]), filename, semantic_hash))
    current_rows.extend(
        [
            current_row(
                "formal_protocol_capability_routing_contract",
                "fail-closed Formal Protocol capability routing contract",
                "formal_protocol_capability_routing_contract.json",
                routing["semantic_sha256"],
            ),
            current_row(
                "formal_generated_checkpoint_resource_identity_contract",
                "run-generated checkpoint resource identity contract",
                "formal_generated_checkpoint_resource_identity_contract.json",
                generated["semantic_sha256"],
            ),
        ]
    )
    shared_rows = [
        deepcopy(row) for row in old_index["active_bundle_resources"]
        if row.get("version_scope") == "shared_historical_stable"
    ]
    index: dict[str, Any] = {
        "active_formal_bundle_contract_version": ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION,
        "active_bundle_resource_resolution_contract_version": ACTIVE_BUNDLE_RESOURCE_RESOLUTION_CONTRACT_VERSION,
        "protocol_index_version": ACTIVE_PROTOCOL_VERSION,
        "status": "NOT_READY_PENDING_G14R14_ACCEPTANCE",
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
        "active_bundle_resources": [*current_rows, *shared_rows],
    }
    index["active_bundle_core_sha256"] = canonical_sha256(active_bundle_core_projection(index))
    evidence_path = ARTIFACT / "acceptance_evidence_manifest.json"
    if evidence_path.is_file():
        evidence = read_json(evidence_path)
        if (
            evidence.get("status") == "pass"
            and evidence.get("active_bundle_core_sha256") == index["active_bundle_core_sha256"]
            and evidence.get("formal_training_count") == 0
            and evidence.get("formal_checkpoint_count") == 0
            and evidence.get("formal_performance_count") == 0
            and evidence.get("holdout_capability") is False
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
            write_json(TARGET / "readiness_v17.json", readiness)
            row = current_row(
                "readiness_companion", "Readiness v17 evidence companion", "readiness_v17.json"
            )
            index["active_bundle_resources"].append(row)
            index["readiness_companion"] = {
                "logical_path": row["logical_path"], "content_sha256": row["content_sha256"]
            }
            index["status"] = READY_STATUS
    index["active_formal_bundle_sha256"] = canonical_sha256(ready_index_projection(index))
    write_json(TARGET / "protocol_index.json", index)
    print(json.dumps({
        "status": index["status"],
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "protocol_full_sha256": protocol["hashes"]["full_sha256"],
        "generated_contract_semantic_sha256": generated["semantic_sha256"],
        "active_bundle_core_sha256": index["active_bundle_core_sha256"],
        "active_formal_bundle_sha256": index["active_formal_bundle_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
