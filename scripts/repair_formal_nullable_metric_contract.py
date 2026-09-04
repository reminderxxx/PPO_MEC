"""Freeze Protocol 2.3 around Formal Nullable Metric Aggregation Contract 1.0.0."""

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
from src.evaluators.cache_baseline_fairness import (
    full_manifest_sha256,
    semantic_protocol_sha256,
)
from src.metrics.formal_nullable_metrics import (
    FORMAL_NULLABLE_METRIC_AGGREGATION_CONTRACT_VERSION,
)
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


SOURCE = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_2_20260901"
TARGET = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_3_20260903"
ARTIFACT = ROOT / (
    "artifacts/analysis/"
    "typed_model_cache_formal_nullable_metric_repair_20260903_g14r12_v1"
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
            .replace("protocol_v2_2_manifest.json", "protocol_v2_3_manifest.json")
            .replace("readiness_v14.json", "readiness_v15.json")
            .replace("G14R11", "G14R12")
            .replace("G14C v12", "G14C v13")
        )
    if isinstance(value, list):
        return [replace_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: replace_paths(item) for key, item in value.items()}
    return value


def nullable_contract() -> dict[str, Any]:
    return semantic(
        {
            "version": FORMAL_NULLABLE_METRIC_AGGREGATION_CONTRACT_VERSION,
            "producer_consumer_scope": [
                "train",
                "eval",
                "dev_checkpoint_selection",
                "checkpoint_freeze",
                "formal_benchmark",
                "formal_statistics",
                "formal_gate_and_claim_map",
            ],
            "value_semantics": {
                "available": "finite int or float; true zero is available",
                "unavailable": "explicit JSON null; never imputed as zero",
                "required_field_missing": "fail_fast_on_active_formal_path",
                "explicit_null": "preserve_and_count_unavailable",
                "invalid": [
                    "bool",
                    "unparseable_string",
                    "NaN",
                    "positive_Infinity",
                    "negative_Infinity",
                ],
                "invalid_action": "fail_fast",
            },
            "aggregation": {
                "available_values_only": True,
                "denominator_is_available_count": True,
                "examples": [
                    {"input": [None], "mean": None, "available_count": 0, "unavailable_count": 1},
                    {"input": [0.0], "mean": 0.0, "available_count": 1, "unavailable_count": 0},
                    {"input": [None, 6.0], "mean": 6.0, "available_count": 1, "unavailable_count": 1},
                    {"input": [], "mean": None, "available_count": 0, "unavailable_count": 0, "availability_status": "unavailable_no_rows"},
                ],
                "mean_metrics_compatibility": "top-level metric remains float_or_null",
                "availability_companion_fields": [
                    "total_count",
                    "available_count",
                    "unavailable_count",
                    "availability_status",
                ],
            },
            "endpoint_consistency": {
                "available_completed_workflow": "finite end_to_end_workflow_delay",
                "failed_incomplete_or_right_censored_workflow": "null end_to_end_workflow_delay",
            },
            "serialization": {
                "csv_empty_cell": "round_trips_as_unavailable_not_zero",
                "json_allow_nan": False,
                "canonical_serialization": "UTF-8 sorted-key compact JSON",
            },
            "dev_selection": {
                "frozen_metric_order_unchanged": True,
                "finite_before_unavailable_per_metric": True,
                "both_unavailable": "skip_to_next_dimension",
                "all_candidates_unavailable": "dimension_not_used_and_recorded",
                "forbidden_null_sentinels": [0, "Infinity", "-Infinity", "reward"],
            },
            "statistics": {
                "paired_availability": "candidate_and_baseline_both_finite",
                "zero_available_pairs": "all_effect_CI_and_p_values_null",
                "holm_inputs": "available_finite_p_values_only",
                "zero_pair_gate_and_claim_status": "UNAVAILABLE_not_tie_pass_or_fail",
                "lower_is_better": [
                    "transfer_mb_per_request",
                    "end_to_end_workflow_delay",
                ],
            },
            "pure_reducer": "src.metrics.formal_nullable_metrics",
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
    old_protocol = read_json(SOURCE / "protocol_v2_2_manifest.json")
    old_index = read_json(SOURCE / "protocol_index.json")
    TARGET.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(SOURCE.glob("*.json")):
        if source_path.name in {
            "protocol_index.json",
            "protocol_v2_2_manifest.json",
            "readiness_v14.json",
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

    nullable = nullable_contract()
    write_json(TARGET / "formal_nullable_metric_aggregation_contract.json", nullable)

    binding_schema = read_json(TARGET / "formal_training_execution_binding_contract.json")
    binding_schema["nullable_metric_contract_hash_enters_binding_context_commands_and_checkpoint"] = True
    write_json(TARGET / "formal_training_execution_binding_contract.json", binding_schema)
    context_schema = read_json(TARGET / "resolved_execution_context_contract.json")
    context_schema["nullable_metric_contract_hash_in_context"] = True
    write_json(TARGET / "resolved_execution_context_contract.json", context_schema)

    protocol = replace_paths(deepcopy(old_protocol))
    protocol.update(
        typed_model_cache_formal_protocol_version=ACTIVE_PROTOCOL_VERSION,
        protocol_id=ACTIVE_PROTOCOL_ID,
        status="frozen_pre_execution_nullable_metric_contract",
        formal_nullable_metric_aggregation_contract=deepcopy(nullable),
    )
    protocol["identity"][
        "formal_nullable_metric_aggregation_contract_semantic_sha256"
    ] = nullable["semantic_sha256"]
    protocol["formal_training_execution_binding_contract"][
        "nullable_metric_contract_hash_enters_binding_context_commands_and_checkpoint"
    ] = True
    protocol["resolved_formal_execution_context_contract"][
        "nullable_metric_contract_hash_in_context"
    ] = True
    protocol["execution_contract"]["same_run_resume"]["bindings"].append(
        "formal_nullable_metric_aggregation_contract_semantic_sha256"
    )
    protocol["execution_contract"]["nullable_metric_binding"] = {
        "contract_version": nullable["version"],
        "semantic_sha256": nullable["semantic_sha256"],
        "phase_input_hash": True,
        "cell_input_hash": True,
        "checkpoint_provenance": True,
        "dev_selection": True,
        "formal_statistics": True,
        "integrity": True,
    }
    protocol["training_budget"]["checkpoint_selection"].update(
        nullable_ordering=(
            "finite before unavailable at each metric; both unavailable skips "
            "dimension; no numeric or reward sentinel"
        ),
        per_metric_candidate_availability_counts_required=True,
    )
    protocol["statistics"].update(
        paired_availability_rule="candidate_and_baseline_both_finite",
        zero_available_pairs="effect_CI_sign_test_and_Holm_are_null",
        holm_input_rule="available_finite_p_values_only",
        nullable_drop_counts=[
            "candidate_only_available",
            "baseline_only_available",
            "both_unavailable",
        ],
        lower_is_better=[
            "transfer_mb_per_request",
            "end_to_end_workflow_delay",
        ],
    )
    protocol["claim_evidence_map"]["zero_pair_endpoint_status"] = (
        "unavailable_not_tie_pass_or_fail"
    )
    protocol["supersession"].update(
        supersedes_version="2.2.0",
        old_protocol_status="invalid_protocol_or_implementation",
        old_protocol_semantic_sha256=old_protocol["hashes"]["semantic_sha256"],
        scientific_fields_changed=False,
        execution_contract_fields_changed=True,
        formal_performance_observed=False,
        repair_scope=[
            "freeze shared strict nullable metric aggregation semantics",
            "close train/eval/dev selection/checkpoint/statistics/gate consumers",
            "permanently reject G14C v12 run and staging checkpoints",
        ],
    )
    protocol["supersession"]["invalid_execution_runs"].append(
        {
            "run_id": "typed_model_cache_formal_20260902_162203_g14c_v12",
            "run_root": (
                "artifacts/experiments/typed_model_cache_formal/"
                "typed_model_cache_formal_20260902_162203_g14c_v12"
            ),
            "status": "INVALID_PROTOCOL_OR_IMPLEMENTATION",
            "failure_boundary": (
                "invalid_during_first_training_cell_after_episode_generation_"
                "before_cell_commit"
            ),
            "failure_audit_sha256": (
                "edb85d74152feefff37b1180d9bc5cb2d04cefa64c226eda831db22539cd39e5"
            ),
            "failure_integrity_sha256": (
                "cff0fedde3b0a58fdd3cb9100fb00a84108c6eddf31685125c9f8bfa98eb51f6"
            ),
            "training_cells_executed": 0,
            "candidate_checkpoint_count": 0,
            "staging_candidate_checkpoint_count_non_evidence": 8,
            "dev_performance_count": 0,
            "formal_performance_count": 0,
            "resume_allowed": False,
            "retry_allowed": False,
            "legacy_phase_finalize_allowed": False,
            "salvage_allowed": False,
            "checkpoint_reuse_allowed": False,
            "candidate_reuse_allowed": False,
            "partial_dev_input_reuse_allowed": False,
            "immutable_old_run": True,
        }
    )
    protocol["active_formal_bundle_contract"]["unique_active_index"] = (
        TARGET / "protocol_index.json"
    ).relative_to(ROOT).as_posix()
    protocol["active_formal_bundle_contract"]["hash_graph"] = [
        "resource content hashes -> active_bundle_core_sha256",
        "core plus G14R12 acceptance evidence -> Readiness v15 content hash",
        "ready index plus Readiness content hash -> active_formal_bundle_sha256",
    ]
    context = protocol["execution_contract"]["default_expansion_context"]
    context["active_protocol_index_path"] = (
        TARGET / "protocol_index.json"
    ).relative_to(ROOT).as_posix()
    context["protocol_path"] = (
        TARGET / "protocol_v2_3_manifest.json"
    ).relative_to(ROOT).as_posix()
    protocol["paper_claim_boundary"] = (
        "G14R12 is outcome-blind execution-contract evidence only; formal training, "
        "checkpoint, performance, holdout, G14C v13, G14D, and G15 remain unexecuted."
    )
    protocol = attach_hashes(protocol)
    write_json(TARGET / "protocol_v2_3_manifest.json", protocol)

    current_rows: list[dict[str, Any]] = []
    for old_row in old_index["active_bundle_resources"]:
        if old_row.get("version_scope") != "current_protocol_version":
            continue
        logical_id = str(old_row["logical_id"])
        if logical_id == "readiness_companion":
            continue
        filename = Path(str(old_row["logical_path"])).name
        if logical_id == "protocol_manifest":
            filename = "protocol_v2_3_manifest.json"
            semantic_hash = protocol["hashes"]["semantic_sha256"]
        else:
            semantic_hash = old_row.get("semantic_sha256")
        current_rows.append(
            current_row(logical_id, str(old_row["role"]), filename, semantic_hash)
        )
    current_rows.insert(
        1,
        current_row(
            "formal_nullable_metric_aggregation_contract",
            "strict nullable metric aggregation, selection, and statistics contract",
            "formal_nullable_metric_aggregation_contract.json",
            nullable["semantic_sha256"],
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
        "status": "NOT_READY_PENDING_G14R12_ACCEPTANCE",
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
        required_passes = {
            "root_cause_audit": "pass",
            "nullable_contract": "pass",
            "producer_consumer_matrix": "pass",
            "v12_permanent_invalidation": "pass",
            "synthetic_validation": "pass",
            "exact_v12_failure_unit_rehearsal": "pass",
            "dev_selection_nullable_validation": "pass",
            "statistics_nullable_validation": "pass",
            "phase_chain_rehearsal": "pass",
            "clean_detached_candidate": "pass",
            "full_repository_pytest": "pass",
            "smoke_test": "pass",
            "strict_serialization_and_integrity": "pass",
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
            and evidence.get("holdout_capability") is False
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
                "holdout_capability": False,
                "holdout_sealed_unopened": True,
            }
            write_json(TARGET / "readiness_v15.json", readiness)
            readiness_row = current_row(
                "readiness_companion",
                "Readiness v15 evidence companion",
                "readiness_v15.json",
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
                "nullable_contract_semantic_sha256": nullable["semantic_sha256"],
                "nullable_contract_content_sha256": sha256_file(
                    TARGET / "formal_nullable_metric_aggregation_contract.json"
                ),
                "active_bundle_core_sha256": index["active_bundle_core_sha256"],
                "active_formal_bundle_sha256": index["active_formal_bundle_sha256"],
                "status": index["status"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
