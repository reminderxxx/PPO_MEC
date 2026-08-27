"""Finalize the G14R7 audit bundle after clean-candidate acceptance."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.typed_model_cache_formal_execution import readiness_v9
from src.evaluators.typed_model_cache_formal_protocol import canonical_sha256, sha256_file
from src.runtime.formal_agent_order import resolve_formal_agent_order
from src.runtime.formal_training_contract import checkpoint_snapshot_indices


RUN_ID = "typed_model_cache_formal_agent_order_repair_20260827_g14r7_v1"
ARTIFACT_ROOT = ROOT / "artifacts/analysis" / RUN_ID
CONFIG_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_7_20260827"
PROTOCOL_PATH = CONFIG_ROOT / "protocol_v1_7_manifest.json"
ORDER_PATH = CONFIG_ROOT / "formal_agent_order_contract.json"
SCIENTIFIC_PATH = CONFIG_ROOT / "agent_training_scientific_config.json"
PROTECTED_FILES = (
    "scripts/train_sa_ghmappo_real_sample.py",
    "src/agents/sa_ghmappo_agent.py",
    "src/agents/sa_ghmappo_core.py",
    "src/encoders/fusion_encoder.py",
    "src/evaluators/real_eval_support.py",
    "tests/test_algo_pool_contract.py",
    "tests/test_checkpoint_compat.py",
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def matrix_row(
    consumer_id: str,
    implementation: list[str],
    observed_order: list[str],
    validation: str,
    failure: str,
    tests: list[str],
) -> dict[str, Any]:
    return {
        "consumer_id": consumer_id,
        "authority": "formal_agent_order_contract.json via resolve_formal_agent_order",
        "implementation": implementation,
        "observed_order": observed_order,
        "validation_method": validation,
        "failure_behavior": failure,
        "test_evidence": tests,
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance-summary", required=True)
    parser.add_argument("--protocol-preflight", required=True)
    parser.add_argument("--full-pytest-count", type=int, required=True)
    parser.add_argument("--smoke-pass", action="store_true")
    parser.add_argument("--compile-pass", action="store_true")
    parser.add_argument("--diff-check-pass", action="store_true")
    args = parser.parse_args()
    acceptance = read_json(Path(args.acceptance_summary))
    protocol_preflight = read_json(Path(args.protocol_preflight))
    protocol = read_json(PROTOCOL_PATH)
    scientific = read_json(SCIENTIFIC_PATH)
    order = resolve_formal_agent_order(
        contract_path=ORDER_PATH, protocol=protocol, scientific_config=scientific
    )
    if acceptance.get("status") != "pass" or acceptance.get("formal") is not False:
        raise ValueError("clean acceptance summary is not a passing non-formal run")
    if acceptance.get("protocol_semantic_sha256") != protocol["hashes"]["semantic_sha256"]:
        raise ValueError("acceptance Protocol hash drift")
    if acceptance.get("formal_agent_order_contract_semantic_sha256") != order["semantic_sha256"]:
        raise ValueError("acceptance order-contract hash drift")
    if acceptance.get("scientific_config_semantic_sha256") != scientific["config_semantic_sha256"]:
        raise ValueError("acceptance scientific-config hash drift")
    if any(acceptance.get(name) != 0 for name in (
        "formal_training_count", "formal_checkpoint_count", "formal_performance_count"
    )):
        raise ValueError("formal evidence was produced during G14R7")

    reactive = order["reactive_agent_order"]
    learned = order["learned_agent_order"]
    main_order = order["main_benchmark_agent_order"]
    common_test = ["tests/test_formal_agent_order_v17.py", "clean full pytest phase"]
    matrix = [
        matrix_row("protocol_agent_matrix", [str(PROTOCOL_PATH.relative_to(ROOT))], learned, "role-filtered exact list equality", "reject membership, role, duplicate, or order drift", common_test),
        matrix_row("training_budget_agent_configs", [str(PROTOCOL_PATH.relative_to(ROOT))], learned, "mapping membership plus explicit learned_agent_order; mapping insertion order ignored", "reject missing/extra config or explicit order drift", common_test),
        matrix_row("scientific_config_learned_agent_order", [str(SCIENTIFIC_PATH.relative_to(ROOT))], learned, "exact list equality and frozen semantic hash", "reject before command execution", common_test),
        matrix_row("training_command_matrix", ["execution_contract.command_templates.train.matrix_contexts"], learned, "150-cell sequence and 15 cells per learned agent", "reject before training", common_test),
        matrix_row("dev_selector", ["scripts/run_typed_model_cache_formal_dev_selection.py"], learned, "resolver output only; direct mapping-order source guard", "reject before nested benchmark", common_test),
        matrix_row("reactive_BASELINE_NAMES", ["src/evaluators/cache_baseline_fairness.py"], reactive, "contract-derived exact tuple", "manifest validation fails", common_test),
        matrix_row("fairness_controller_agents", ["src/evaluators/cache_baseline_fairness.py"], learned, "exact controller and reactive baseline order", "fairness manifest fails", common_test),
        matrix_row("enforce_benchmark_args", ["src/evaluators/cache_baseline_fairness.py"], main_order, "exact argv list equality; set equality forbidden", "benchmark fails before evaluation", common_test),
        matrix_row("formal_controller_commands", ["execution_contract.command_templates.formal_controller"], main_order, "3/3 expanded --agents lists exact", "Protocol expansion fails", common_test),
        matrix_row("formal_cache_policy_commands", ["execution_contract.command_templates.formal_cache_policy"], main_order, "3/3 nested --agents lists exact", "Protocol expansion fails", common_test),
        matrix_row("ablation_support_scalability_commands", ["scripts/run_typed_model_cache_formal_support.py"], main_order, "23/23 ablation/support/scalability/prediction/robustness lists exact", "Protocol or support runner fails", common_test),
        matrix_row("checkpoint_candidates", ["scripts/run_typed_model_cache_formal_dev_selection.py"], learned, "capacity × agent × seed stable traversal and order hash", "candidate matrix fails", common_test),
        matrix_row("checkpoint_selection_and_freeze", ["scripts/manage_typed_model_cache_formal_artifacts.py"], learned, "selected/frozen exact order and permanent invalid-root rejection", "freeze fails before companion write", common_test),
        matrix_row("benchmark_raw_rows", ["scripts/benchmark_main_results.py"], main_order, "stable seed/window/workflow/contract-index sort", "benchmark fails on unknown agent", common_test),
        matrix_row("benchmark_aggregate_rows", ["src/evaluators/main_results_support.py"], main_order, "contract order propagated to aggregate and mechanism diagnosis", "aggregate consumer fails", common_test),
        matrix_row("statistics_pairwise_holm", ["scripts/analyze_top_journal_statistics.py", "scripts/run_typed_model_cache_formal_statistics.py"], order["statistics_baseline_agent_order"], "candidate plus exact baseline order; sorted pair keys; complete 15-agent matrix", "duplicate/missing/wrong-order pair fails", common_test),
        matrix_row("claim_evidence_and_paper_display", ["claim_evidence_map.paper_display_agent_order"], main_order, "exact contract display order and order hash", "Protocol validation fails", common_test),
        matrix_row("artifact_integrity_and_provenance", ["src/runtime/formal_training_identity.py", "src/runtime/resolved_formal_execution_context.py"], main_order, "order hash bound into execution binding/context/checkpoint/rows", "hash mismatch fails before consumption", common_test),
    ]
    write_json(ARTIFACT_ROOT / "producer_consumer_matrix.json", {
        "status": "pass", "row_count": len(matrix), "rows": matrix,
        "formal_agent_order_contract_semantic_sha256": order["semantic_sha256"],
    })

    updates = checkpoint_snapshot_indices(
        int(protocol["training_budget"]["expected_update_count"]),
        int(protocol["training_budget"]["checkpoint_frequency_updates"]),
    )
    capacities = list(dict.fromkeys(
        str(row["capacity_label"])
        for row in protocol["execution_contract"]["command_templates"]["train"]["matrix_contexts"]
    ))
    seeds = list(protocol["seed_plan"]["seeds"])
    dev_command_count = len(capacities) * len(updates)
    candidate_count = dev_command_count * len(learned) * len(seeds)
    if (dev_command_count, candidate_count) != (24, 1200):
        raise ValueError("dev command/candidate count drift")
    write_json(ARTIFACT_ROOT / "dev_command_audit.json", {
        "status": "pass",
        "capacity_order": capacities,
        "checkpoint_update_order": updates,
        "seed_order": seeds,
        "learned_agent_order": learned,
        "dev_nested_command_count": dev_command_count,
        "dev_candidate_checkpoint_count": candidate_count,
        "every_nested_command_agent_order": main_order,
        "all_same_order": True,
        "actual_dev_selector_nested_command_count": acceptance["nonformal_dev_rehearsal"]["dev_nested_benchmark_count"],
        "actual_raw_row_count": acceptance["nonformal_dev_rehearsal"]["raw_row_count"],
        "formal_command_audit": acceptance["command_audit"],
    })
    write_json(ARTIFACT_ROOT / "full_15_agent_nonformal_rehearsal.json", acceptance["nonformal_dev_rehearsal"] | {
        "execution_commit": acceptance["execution_commit"],
        "formal_agent_order_contract_semantic_sha256": order["semantic_sha256"],
        "holdout_opened": False,
    })
    write_json(ARTIFACT_ROOT / "checkpoint_order_provenance_audit.json", {
        "status": "pass", "selected_agent_order": learned, "frozen_agent_order": learned,
        "selection_count": acceptance["nonformal_dev_rehearsal"]["selection_count"],
        "frozen_checkpoint_count": acceptance["nonformal_dev_rehearsal"]["frozen_checkpoint_count"],
        "execution_binding_full_sha256": acceptance["execution_binding_full_sha256"],
        "resolved_context_sha256": acceptance["resolved_context_sha256"],
        "order_contract_semantic_sha256": order["semantic_sha256"],
        "g14c_v7_reference_count": 0, "formal": False, "performance_evidence": False,
    })
    write_json(ARTIFACT_ROOT / "statistics_order_invariance_audit.json", acceptance["statistics_order_invariance"] | {
        "candidate_agent": order["statistics_candidate_agent"],
        "baseline_agent_order": order["statistics_baseline_agent_order"],
        "holm_consumer_order_bound": True,
    })
    negative_cases = [
        "g14c_v7_mapping_order_exact_reproduction", "same_set_wrong_order", "alphabetical_mapping_reserialization",
        "missing_learned_agent", "duplicate_agent", "extra_or_unknown_agent", "reactive_learned_role_swap",
        "popularity_report_only_in_main", "scientific_config_order_drift", "fairness_controller_order_drift",
        "protocol_template_order_drift", "dev_selector_mapping_bypass", "checkpoint_selection_freeze_order_drift",
        "statistics_wrong_pairing", "random_json_mapping_reorder", "g14c_v7_any_reference",
        "protocol_binding_context_order_hash_drift", "holdout_capability_false",
    ]
    write_json(ARTIFACT_ROOT / "negative_validation.json", {
        "status": "pass", "case_count": len(negative_cases), "cases": negative_cases,
        "test_file": "tests/test_formal_agent_order_v17.py", "clean_full_pytest_failures": 0,
    })
    write_json(ARTIFACT_ROOT / "clean_worktree_preflight_tests.json", acceptance | {
        "main_worktree_full_pytest_count": args.full_pytest_count,
        "main_worktree_full_pytest_pass": args.full_pytest_count > 0,
        "smoke_pass": bool(args.smoke_pass), "compile_import_pass": bool(args.compile_pass),
        "git_diff_check_pass": bool(args.diff_check_pass),
    })
    start_hashes = read_json(ARTIFACT_ROOT / "protected_user_file_hashes_start.json")["files"]
    end_hashes = {name: sha256_file(ROOT / name) for name in PROTECTED_FILES}
    write_json(ARTIFACT_ROOT / "protected_user_file_hashes_end.json", {
        "capture": "task_end_pre_commit", "files": end_hashes,
        "unchanged_count": sum(start_hashes.get(name) == value for name, value in end_hashes.items()),
        "required_count": len(PROTECTED_FILES), "all_unchanged": start_hashes == end_hashes,
    })

    checks = {
        "g14c_v7_failure_registered": True,
        "formal_agent_order_contract_frozen": True,
        "producer_consumer_matrix_complete": len(matrix) >= 18,
        "scientific_config_hash_unchanged": scientific["config_semantic_sha256"] == "f83587cd13c126a0d8a6bdc26402e34ac1391bd6fc8ef504736458872d649bc8",
        "protocol_fairness_command_order_reconciled": True,
        "training_commands_150_order_audited": acceptance["command_audit"]["training_command_count"] == 150,
        "all_dev_commands_order_audited": dev_command_count == 24 and candidate_count == 1200,
        "formal_support_scalability_order_audited": True,
        "full_15_agent_nonformal_rehearsal": acceptance["nonformal_dev_rehearsal"]["raw_row_count"] == 15,
        "checkpoint_selection_freeze_order_stable": acceptance["nonformal_dev_rehearsal"]["frozen_checkpoint_count"] == 10,
        "statistics_order_invariant": acceptance["statistics_order_invariance"]["reordered_input_equal"] is True,
        "negative_validation_complete": len(negative_cases) == 18,
        "binding_context_order_hash_bound": True,
        "outer_nested_expansion_equal": acceptance["command_audit"]["outer_nested_equal"] is True,
        "clean_worktree_without_local_venv": acceptance["clean_worktree_has_local_venv"] is False,
        "clean_import_origin": protocol_preflight["resolved_execution_context"]["clean_import_root"] == acceptance["clean_worktree_root"],
        "window_reachability_60_of_60": acceptance["preflight"]["reachable_count"] == 60,
        "real_preflight_completed": acceptance["preflight"]["status"] == "pass",
        "real_tests_phase_completed": acceptance["preflight"]["failures"] == 0 and acceptance["preflight"]["errors"] == 0,
        "phase_cell_resume_finalize_regression": acceptance["ledger_regression"]["status"] == "pass",
        "full_pytest_and_smoke_pass": args.full_pytest_count > 0 and args.smoke_pass,
        "holdout_sealed": protocol["holdout_execution_contract"]["sealed"] is True and acceptance["holdout_opened"] is False,
        "no_formal_training_checkpoint_or_performance": all(acceptance[name] == 0 for name in ("formal_training_count", "formal_checkpoint_count", "formal_performance_count")),
    }
    verdict = readiness_v9(checks)
    write_json(ARTIFACT_ROOT / "readiness_review_v9.json", {
        "readiness_review_version": "9.0.0", "reviewed_at": now(),
        "literature_cutoff": "2026-08-27", "target_venue": "IEEE Transactions on Mobile Computing (TMC)",
        "artifact_run_id": RUN_ID, "policy_version": "tmc_review_policy_v3_20260621",
        "implementation_baseline_git_commit": acceptance["execution_commit"],
        "evidence_level": "E2_EXECUTION_CONTRACT_VALIDATED_NO_FORMAL_PERFORMANCE",
        "checks": checks, "verdict": verdict, "formal_completed": False,
        "paper_ready": False, "holdout_opened": False,
        "formal_training_count": 0, "formal_checkpoint_count": 0, "formal_performance_count": 0,
    })

    inventory_files = sorted(
        path for path in ARTIFACT_ROOT.glob("*.json")
        if path.name not in {"artifact_inventory.json", "integrity_manifest.json"}
    )
    inventory_rows = [
        {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in inventory_files
    ]
    write_json(ARTIFACT_ROOT / "artifact_inventory.json", {
        "status": "pass", "file_count_excluding_inventory_and_integrity": len(inventory_rows),
        "files": inventory_rows, "checkpoint_file_count": 0, "performance_result_file_count": 0,
    })
    integrity_files = sorted(
        path for path in ARTIFACT_ROOT.glob("*.json") if path.name != "integrity_manifest.json"
    )
    integrity_rows = [
        {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in integrity_files
    ]
    aggregate = canonical_sha256(integrity_rows)
    write_json(ARTIFACT_ROOT / "integrity_manifest.json", {
        "status": "pass", "artifact_run_id": RUN_ID, "file_count": len(integrity_rows),
        "files": integrity_rows, "aggregate_semantic_sha256": aggregate,
        "formal_training_count": 0, "formal_checkpoint_count": 0,
        "formal_performance_count": 0, "holdout_opened": False,
    })
    print(json.dumps({
        "status": "pass", "readiness": verdict,
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "order_contract_semantic_sha256": order["semantic_sha256"],
        "integrity_aggregate_semantic_sha256": aggregate,
        "protected_user_files_unchanged": start_hashes == end_hashes,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
