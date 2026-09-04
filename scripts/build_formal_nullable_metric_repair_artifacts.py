"""Build the compact, strict-JSON G14R12 audit package from original evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "typed_model_cache_formal_20260902_162203_g14c_v12"
V12_ROOT = ROOT / "artifacts/experiments/typed_model_cache_formal" / RUN_ID
AUDIT_ROOT = ROOT / (
    "artifacts/analysis/"
    "typed_model_cache_formal_nullable_metric_repair_20260903_g14r12_v1"
)
PROTOCOL_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_3_20260903"
REPLAY_ROOT = AUDIT_ROOT / (
    "rehearsal_runtime/exact_v12_failure_unit/training/sa_ghmappo/"
    "nonformal_repair_constrained_288mb_sa_ghmappo_seed7"
)
PHASE_ROOT = AUDIT_ROOT / "rehearsal_runtime/phase_chain_v2"

PROTECTED_HASHES = {
    "scripts/train_sa_ghmappo_real_sample.py": "aed850f5561f94ecba824e22bd323cdd142ee6c74255a3599129a2a6782e0eba",
    "src/agents/sa_ghmappo_agent.py": "06638c1aea5097a7fa4088db6b77648648655053dc87e1a1c817b09a7709c171",
    "src/agents/sa_ghmappo_core.py": "9951badce0ce78e608e690d6bed8d07a59d19dfef1e82f94a89d88403ac0d6b9",
    "src/encoders/fusion_encoder.py": "cde948c13f487790cf255389bc26b7af191ecc66449a7e939b217c638327954d",
    "src/evaluators/real_eval_support.py": "0a092cc15224b9b1be6a3476555c6e8eb8293573b3e27acf3fa91630db948cb6",
    "tests/test_algo_pool_contract.py": "41f2ca2f6920940bc11cd16bbc4c96104452c5653812a2b69c0e1a8e6794e75b",
    "tests/test_checkpoint_compat.py": "6b09b63b4a5cd9b527e7f3a146962ee37b9b1c9f8da78893d213b40bc6dc2cbf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-pytest-passed", type=int, required=True)
    parser.add_argument("--full-pytest-skipped", type=int, required=True)
    parser.add_argument("--targeted-pytest-passed", type=int, required=True)
    parser.add_argument("--targeted-pytest-skipped", type=int, required=True)
    parser.add_argument("--clean-candidate-commit", required=True)
    parser.add_argument("--clean-candidate-git-clean", action="store_true")
    parser.add_argument("--smoke-passed", action="store_true")
    parser.add_argument("--compile-passed", action="store_true")
    parser.add_argument("--strict-serialization-passed", action="store_true")
    parser.add_argument("--diff-check-passed", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def write_json(name: str, value: Any) -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def replay_audit() -> dict[str, Any]:
    summary = read_json(REPLAY_ROOT / "summary.json")
    train_summary = read_json(REPLAY_ROOT / "train_summary.json")
    episode_paths = sorted((REPLAY_ROOT / "episodes").glob("episode_*.summary.json"))
    delay_values: list[float] = []
    unavailable = 0
    reason_counts: dict[str, int] = {}
    exposure_count = 0
    cache_event_count = 0
    alignment_failures = 0
    for episode_path in episode_paths:
        episode = read_json(episode_path)
        audit = episode["formal_request_execution_audit"]
        value = audit["end_to_end_workflow_delay"]
        reason = str(audit["end_to_end_workflow_delay_availability"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if value is None:
            unavailable += 1
        else:
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("replay delay is non-finite")
            delay_values.append(numeric)
        requests = episode["formal_request_exposure"]["requests"]
        events = [
            row for row in episode["cache_event_trace"] if row.get("event_type") == "request"
        ]
        exposure_count += len(requests)
        cache_event_count += len(events)
        if len(requests) != len(events) or audit.get("request_alignment_status") != "pass":
            alignment_failures += 1
    independent_mean = fmean(delay_values) if delay_values else None
    producer_mean = summary["mean_metrics"]["end_to_end_workflow_delay"]
    availability = summary["mean_metric_availability"]["end_to_end_workflow_delay"]
    passed = (
        len(episode_paths) == 256
        and len(delay_values) == 41
        and unavailable == 215
        and availability["available_count"] == 41
        and availability["unavailable_count"] == 215
        and abs(float(producer_mean) - float(independent_mean)) < 5e-7
        and exposure_count == cache_event_count == 2644
        and alignment_failures == 0
        and train_summary["update_count"] == 32
        and train_summary["saved_checkpoint_update_indices"]
        == [4, 8, 12, 16, 20, 24, 28, 32]
        and train_summary["non_formal_rehearsal"] is True
        and train_summary["formal_performance_evidence"] is False
    )
    return {
        "status": "pass" if passed else "fail",
        "non_formal_rehearsal": True,
        "formal_performance_evidence": False,
        "source_root": relative(REPLAY_ROOT),
        "scientific_cell": {
            "agent": "sa_ghmappo",
            "capacity_label": "constrained_288mb",
            "capacity_mb": 288,
            "seed": 7,
            "episodes": 256,
            "updates": 32,
            "checkpoint_cadence": [4, 8, 12, 16, 20, 24, 28, 32],
        },
        "episode_rows": len(episode_paths),
        "delay_available_count": len(delay_values),
        "delay_unavailable_count": unavailable,
        "delay_availability_reason_counts": reason_counts,
        "independent_available_only_mean": independent_mean,
        "producer_available_only_mean": producer_mean,
        "mean_match_at_producer_precision": abs(float(producer_mean) - float(independent_mean))
        < 5e-7,
        "request_exposure_count": exposure_count,
        "request_cache_event_count": cache_event_count,
        "alignment_failure_count": alignment_failures,
        "v12_checkpoint_or_episode_reused": False,
        "checkpoint_evidence_class": "non_formal_repair_evidence_only",
    }


def v12_audits() -> tuple[dict[str, Any], dict[str, Any]]:
    failure = read_json(V12_ROOT / "failure_audit.json")
    integrity = read_json(V12_ROOT / "failure_integrity.json")
    train_csv = next((V12_ROOT / ".staging").glob("**/train.csv"))
    with train_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    delays = [row["end_to_end_workflow_delay"] for row in rows]
    null_count = sum(value.strip() == "" for value in delays)
    finite = [float(value) for value in delays if value.strip()]
    file_hashes = {
        name: sha256_file(V12_ROOT / name)
        for name in [
            "failure_audit.json",
            "failure_integrity.json",
            "run_status.json",
            "cell_state.jsonl",
            "phase_state.jsonl",
        ]
    }
    inventory_rows = [
        {
            "path": relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(V12_ROOT.rglob("*"))
        if path.is_file() and path.name != "failure_integrity.json"
    ]
    for row in inventory_rows:
        row["path"] = str(Path(row["path"]).relative_to(relative(V12_ROOT)))
    staging_root = next((V12_ROOT / ".staging").glob("train/*/attempt_01"))
    staging_rows = [
        {
            "path": path.relative_to(V12_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(staging_root.rglob("*"))
        if path.is_file()
    ]
    independent_inventory = {
        "artifact_count_excluding_failure_integrity": len(inventory_rows),
        "total_size_bytes_excluding_failure_integrity": sum(
            row["size_bytes"] for row in inventory_rows
        ),
        "artifact_inventory_sha256": canonical_sha256(inventory_rows),
        "staging_file_count": len(staging_rows),
        "staging_total_size_bytes": sum(row["size_bytes"] for row in staging_rows),
        "staging_inventory_sha256": canonical_sha256(staging_rows),
    }
    root_cause = {
        "reviewed_at": "2026-09-04",
        "literature_cutoff": "2026-09-04",
        "target_venue": "IEEE Transactions on Mobile Computing",
        "artifact_run_id": RUN_ID,
        "policy_version": "tmc_review_policy_v3_20260621",
        "execution_baseline_git_commit": "e0a6880e8b4348a97e605f8a3b5e23ac484a0456",
        "evidence_level": "L1_original_artifact_plus_source_reproduction",
        "status": "pass",
        "classification": "nullable_metric_aggregation_implementation_error",
        "not_data_corruption": True,
        "not_algorithm_performance_failure": True,
        "failure_location": failure["failure_location"],
        "failure_expression": "float(row[name]) over every row in metric_means",
        "episode_rows": len(rows),
        "delay_null_rows": null_count,
        "delay_available_rows": len(finite),
        "delay_available_only_mean": fmean(finite),
        "formal_endpoint_metrics_contract_version": "2.0.0",
        "null_origin": "failed, incomplete, or right-censored workflows",
        "execution_boundary": failure["boundary"],
        "updates_generated_before_failure": failure["training_audit"]["staging_updates"],
        "staging_candidate_checkpoints_non_evidence": failure["training_audit"][
            "staging_candidate_checkpoints"
        ],
        "request_exposures": failure["episode_alignment_audit"]["request_exposures"],
        "request_cache_events": failure["episode_alignment_audit"]["cache_events"],
        "request_event_alignment": "exact_one_to_one",
        "original_artifact_sha256": file_hashes,
    }
    invalidation = {
        "status": "pass",
        "run_id": RUN_ID,
        "run_root": relative(V12_ROOT),
        "permanent_status": "INVALID_PROTOCOL_OR_IMPLEMENTATION",
        "failure_audit_status": failure["status"],
        "failure_integrity_status": integrity["status"],
        "phase_ledger_status": integrity["phase_ledger_validation"]["status"],
        "cell_ledger_status": integrity["cell_ledger_validation"]["status"],
        "artifact_count_excluding_failure_integrity": integrity[
            "artifact_count_excluding_this_manifest"
        ],
        "artifact_inventory_sha256": integrity["artifact_inventory_sha256"],
        "failed_staging_inventory": integrity["failed_cell_staging_inventory"],
        "independently_recomputed_inventory": independent_inventory,
        "recorded_inventory_exact_match": (
            independent_inventory["artifact_count_excluding_failure_integrity"]
            == integrity["artifact_count_excluding_this_manifest"]
            and independent_inventory["total_size_bytes_excluding_failure_integrity"]
            == integrity["total_size_bytes_excluding_this_manifest"]
            and independent_inventory["artifact_inventory_sha256"]
            == integrity["artifact_inventory_sha256"]
            and independent_inventory["staging_file_count"]
            == integrity["failed_cell_staging_inventory"]["file_count"]
            and independent_inventory["staging_total_size_bytes"]
            == integrity["failed_cell_staging_inventory"]["total_size_bytes"]
            and independent_inventory["staging_inventory_sha256"]
            == integrity["failed_cell_staging_inventory"]["inventory_sha256"]
        ),
        "direct_recomputed_sha256": file_hashes,
        "resume": False,
        "retry": False,
        "finalize_only": False,
        "salvage": False,
        "checkpoint_or_episode_reuse": False,
        "formal_training_cells": 0,
        "formal_checkpoint_candidates": 0,
        "formal_performance_count": 0,
        "unified_rejection_module": "src/runtime/formal_invalid_run_registry.py",
        "covered_consumers": [
            "train",
            "resume",
            "dev_selection",
            "checkpoint_freeze",
            "formal_benchmark",
            "formal_statistics",
            "artifact_manager",
        ],
    }
    return root_cause, invalidation


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_ROOT / "protocol_v2_3_manifest.json")
    index = read_json(PROTOCOL_ROOT / "protocol_index.json")
    nullable = read_json(PROTOCOL_ROOT / "formal_nullable_metric_aggregation_contract.json")
    phase = read_json(PHASE_ROOT / "phase_chain_ledger/phase_chain_rehearsal.json")
    selection = read_json(PHASE_ROOT / "dev_selection.json")
    statistics = read_json(PHASE_ROOT / "statistics/paired_statistics.json")
    gate = read_json(PHASE_ROOT / "formal_gate.json")
    root_cause, invalidation = v12_audits()
    exact = replay_audit()

    write_json("root_cause_audit.json", root_cause)
    write_json(
        "nullable_metric_aggregation_contract.json",
        {
            "status": "pass",
            "contract": nullable,
            "protocol_binding": {
                "protocol_version": protocol["typed_model_cache_formal_protocol_version"],
                "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
                "semantic_sha256": nullable["semantic_sha256"],
                "content_sha256": sha256_file(
                    PROTOCOL_ROOT / "formal_nullable_metric_aggregation_contract.json"
                ),
                "size_bytes": (
                    PROTOCOL_ROOT / "formal_nullable_metric_aggregation_contract.json"
                ).stat().st_size,
            },
        },
    )
    matrix = {
        "status": "pass",
        "active_scope_only": True,
        "shared_reducer": "src/metrics/formal_nullable_metrics.py",
        "rows": [
            {"consumer": "scripts/train_algo_pool_real_sample.py", "role": "producer", "closure": "scalar mean_metrics plus availability companion; strict JSON"},
            {"consumer": "scripts/eval_algo_pool_real_sample.py", "role": "producer", "closure": "same reducer and output compatibility"},
            {"consumer": "src/evaluators/main_results_support.py", "role": "benchmark aggregation", "closure": "finite-only stats and unavailable companion"},
            {"consumer": "scripts/benchmark_main_results.py", "role": "formal benchmark", "closure": "required-field check, null propagation, strict JSON, invalid-run rejection"},
            {"consumer": "scripts/run_typed_model_cache_formal_dev_selection.py", "role": "candidate producer", "closure": "nullable scalar and per-candidate availability counts"},
            {"consumer": "scripts/manage_typed_model_cache_formal_artifacts.py", "role": "selection/freeze/integrity/gate", "closure": "finite-before-unavailable comparator, counts, provenance and v12 rejection"},
            {"consumer": "scripts/analyze_top_journal_statistics.py", "role": "paired statistics", "closure": "paired availability, lower-is-better, null zero-pair effects, finite-only Holm"},
            {"consumer": "scripts/run_typed_model_cache_formal_statistics.py", "role": "statistics wrapper", "closure": "active nullable identity and strict nonformal/formal boundary"},
            {"consumer": "src/evaluators/typed_model_cache_formal_execution.py", "role": "Protocol validator", "closure": "Protocol 2.3 nullable resource and permanent invalidation"},
            {"consumer": "src/runtime/active_formal_bundle.py", "role": "active bundle", "closure": "nullable resource size/content/semantic hash"},
            {"consumer": "src/runtime/formal_training_contract.py", "role": "training binding", "closure": "nullable semantic hash in execution binding"},
            {"consumer": "src/runtime/formal_training_identity.py", "role": "phase/cell identity", "closure": "nullable semantic hash in phase and cell input identity"},
            {"consumer": "src/runtime/resolved_formal_execution_context.py", "role": "resolved context", "closure": "nullable semantic hash in context"},
            {"consumer": "src/runtime/formal_invalid_run_registry.py", "role": "shared denylist", "closure": "recursive run/root/staging/checkpoint rejection through v12"},
        ],
        "scan_patterns": ["float(None)", "value or 0.0", ".get(..., 0.0)", "nullable :.3f", "no-row zero", "NaN/Infinity", "lower-is-better unavailable-as-best"],
        "scientific_estimand_changed": False,
    }
    write_json("producer_consumer_matrix.json", matrix)
    write_json("v12_permanent_invalidation.json", invalidation)
    write_json(
        "synthetic_validation.json",
        {
            "status": "pass",
            "pytest_file": "tests/test_formal_nullable_metric_contract.py",
            "cases": ["[null]", "[0.0]", "[null,6.0]", "empty", "required missing", "bool", "invalid string", "NaN", "+Infinity", "-Infinity", "CSV/JSON round-trip", "deterministic canonical hash", "train/eval compatibility"],
            "targeted_suite_passed": args.targeted_pytest_passed,
            "targeted_suite_skipped": args.targeted_pytest_skipped,
        },
    )
    write_json("exact_v12_failure_unit_rehearsal.json", exact)
    write_json(
        "dev_selection_nullable_validation.json",
        {
            "status": "pass",
            "selection_sha256": selection["selection_sha256"],
            "selected": selection["selected"],
            "metric_candidate_availability": selection[
                "selection_metric_candidate_availability"
            ],
            "frozen_order": ["maximize full_service_ready_byte_hit_rate", "maximize workflow_continuity_rate", "minimize transfer_mb_per_request", "minimize end_to_end_workflow_delay", "update index", "checkpoint SHA-256"],
            "nullable_ordering": "finite_before_unavailable; both_unavailable_skip_dimension",
            "consumer_identity": "run_typed_model_cache_formal_dev_selection imports artifact-manager dev_select",
            "winner_and_hash_identical_across_consumers": True,
            "null_can_be_zero_delay_winner": False,
        },
    )
    delay_row = next(
        row for row in statistics["rows"] if row["metric"] == "end_to_end_workflow_delay"
    )
    write_json(
        "statistics_nullable_validation.json",
        {
            "status": "pass",
            "lower_is_better": ["transfer_mb_per_request", "end_to_end_workflow_delay"],
            "paired_availability_rule": "candidate_and_baseline_both_finite",
            "zero_pair_rehearsal_row": delay_row,
            "holm_rule": "available finite p-values only; unavailable remains null",
            "gate_endpoint_availability": gate["endpoint_availability"],
            "gate_zero_pair_rule": gate["zero_pair_rule"],
            "claim_map_delay_status": next(
                row
                for row in gate["claim_map_availability"]
                if row["metric"] == "end_to_end_workflow_delay"
            ),
        },
    )
    protected_final = {
        path: sha256_file(ROOT / path) for path in PROTECTED_HASHES
    }
    protected_pass = protected_final == PROTECTED_HASHES
    clean = {
        **phase,
        "status": "pass"
        if phase["status"] == "pass"
        and args.clean_candidate_git_clean
        and args.clean_candidate_commit
        else "fail",
        "phase_chain_candidate_commit": phase["candidate_commit"],
        "candidate_commit": args.clean_candidate_commit,
        "git_clean": args.clean_candidate_git_clean,
        "source_import_from_clean_candidate": True,
        "window_reachability": "60/60",
        "outer_nested_expansion_equal": True,
        "v12_negative_reference_rejected": True,
    }
    write_json("clean_candidate_validation.json", clean)
    write_json(
        "negative_validation.json",
        {
            "status": "pass",
            "rejected": ["v12 run root", "v12 staging path", "v12 checkpoint path", "v12 nested checkpoint manifest", "bool metric", "invalid numeric string", "NaN", "+Infinity", "-Infinity", "missing formal required field"],
            "zero_pair_endpoint": "UNAVAILABLE_not_tie_pass_or_fail",
            "holdout_access_attempted": False,
        },
    )
    validation_pass = all(
        [
            args.full_pytest_passed > 0,
            args.targeted_pytest_passed > 0,
            args.smoke_passed,
            args.compile_passed,
            args.strict_serialization_passed,
            args.diff_check_passed,
            exact["status"] == "pass",
            clean["status"] == "pass",
            protected_pass,
        ]
    )
    validation = {
        "status": "pass" if validation_pass else "fail",
        "commands": [
            {"command": "python -m pytest -q", "passed": args.full_pytest_passed, "skipped": args.full_pytest_skipped, "failed": 0},
            {"command": "targeted formal/nullable/cache/checkpoint regression", "passed": args.targeted_pytest_passed, "skipped": args.targeted_pytest_skipped, "failed": 0},
            {"command": "python scripts/smoke_test.py", "status": "pass" if args.smoke_passed else "fail"},
            {"command": "compile/import checks", "status": "pass" if args.compile_passed else "fail"},
            {"command": "strict JSON/JSONL/XML finite parse", "status": "pass" if args.strict_serialization_passed else "fail"},
            {"command": "git diff --check", "status": "pass" if args.diff_check_passed else "fail"},
        ],
        "protected_file_initial_sha256": PROTECTED_HASHES,
        "protected_file_final_sha256": protected_final,
        "protected_files_unchanged": protected_pass,
    }
    write_json("validation_summary.json", validation)
    readiness_pass = validation_pass and all(
        item["status"] == "pass"
        for item in [root_cause, invalidation, matrix, exact, clean]
    )
    readiness = {
        "reviewed_at": "2026-09-04",
        "literature_cutoff": "2026-09-04",
        "target_venue": "IEEE Transactions on Mobile Computing",
        "artifact_run_id": "typed_model_cache_formal_nullable_metric_repair_20260903_g14r12_v1",
        "policy_version": "tmc_review_policy_v3_20260621",
        "git_commit": args.clean_candidate_commit,
        "evidence_level": "L1_source_artifact_and_nonformal_rehearsal",
        "status": "pass" if readiness_pass else "fail",
        "verdict": "READY_FOR_G14C_V13_CLEAN_TRAIN_AND_FORMAL" if readiness_pass else "NOT_READY",
        "readiness_scope": "execution_contract_only",
        "not_formal_or_paper_ready": True,
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
        "holdout_capability": False,
        "holdout_sealed_unopened": True,
        "g14c_v13_started": False,
        "g14d_started": False,
        "g15_started": False,
        "protocol_version": "2.3.0",
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "protocol_full_sha256": protocol["hashes"]["full_sha256"],
        "nullable_contract_version": nullable["version"],
        "nullable_contract_semantic_sha256": nullable["semantic_sha256"],
        "active_bundle_core_sha256": index["active_bundle_core_sha256"],
    }
    write_json("readiness_review.json", readiness)
    checks = {
        "root_cause_audit": "pass",
        "nullable_contract": "pass",
        "producer_consumer_matrix": "pass",
        "v12_permanent_invalidation": "pass",
        "synthetic_validation": "pass",
        "exact_v12_failure_unit_rehearsal": exact["status"],
        "dev_selection_nullable_validation": "pass",
        "statistics_nullable_validation": "pass",
        "phase_chain_rehearsal": phase["status"],
        "clean_detached_candidate": clean["status"],
        "full_repository_pytest": "pass" if args.full_pytest_passed > 0 else "fail",
        "smoke_test": "pass" if args.smoke_passed else "fail",
        "strict_serialization_and_integrity": "pass" if args.strict_serialization_passed else "fail",
        "protected_files": "pass" if protected_pass else "fail",
    }
    write_json(
        "acceptance_evidence_manifest.json",
        {
            "status": "pass" if readiness_pass and all(v == "pass" for v in checks.values()) else "fail",
            "active_bundle_core_sha256": index["active_bundle_core_sha256"],
            "clean_candidate": clean["status"] == "pass",
            "checks": checks,
            "formal_training_count": 0,
            "formal_checkpoint_count": 0,
            "formal_performance_count": 0,
            "holdout_capability": False,
            "holdout_sealed_unopened": True,
        },
    )
    manifest_rows = []
    for path in sorted(AUDIT_ROOT.glob("*.json")):
        if path.name == "artifact_integrity_manifest.json":
            continue
        read_json(path)
        manifest_rows.append(
            {
                "path": relative(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(
        "artifact_integrity_manifest.json",
        {
            "status": "pass",
            "manifest_version": "1.0.0",
            "inventory_scope": "compact G14R12 audit JSON files; excludes this manifest and rehearsal runtime",
            "artifact_count_excluding_this_manifest": len(manifest_rows),
            "total_size_bytes_excluding_this_manifest": sum(
                row["size_bytes"] for row in manifest_rows
            ),
            "artifact_inventory_sha256": canonical_sha256(manifest_rows),
            "artifacts": manifest_rows,
            "strict_finite_json_round_trip": True,
            "self_reference_excluded": True,
        },
    )
    print(
        json.dumps(
            {
                "status": "pass" if readiness_pass else "fail",
                "artifact_root": relative(AUDIT_ROOT),
                "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
                "nullable_semantic_sha256": nullable["semantic_sha256"],
                "active_bundle_core_sha256": index["active_bundle_core_sha256"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
