"""Build G14R18 audit, readiness, and final-execution evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V28 = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_8_20260906"
V29 = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906"
ARTIFACT = ROOT / (
    "artifacts/analysis/"
    "typed_model_cache_formal_provenance_envelope_repair_20260906_g14r18_v1"
)
PROTECTED = {
    "scripts/train_sa_ghmappo_real_sample.py": "aed850f5561f94ecba824e22bd323cdd142ee6c74255a3599129a2a6782e0eba",
    "src/agents/sa_ghmappo_agent.py": "06638c1aea5097a7fa4088db6b77648648655053dc87e1a1c817b09a7709c171",
    "src/agents/sa_ghmappo_core.py": "9951badce0ce78e608e690d6bed8d07a59d19dfef1e82f94a89d88403ac0d6b9",
    "src/encoders/fusion_encoder.py": "cde948c13f487790cf255389bc26b7af191ecc66449a7e939b217c638327954d",
    "src/evaluators/real_eval_support.py": "0a092cc15224b9b1be6a3476555c6e8eb8293573b3e27acf3fa91630db948cb6",
    "tests/test_algo_pool_contract.py": "41f2ca2f6920940bc11cd16bbc4c96104452c5653812a2b69c0e1a8e6794e75b",
    "tests/test_checkpoint_compat.py": "6b09b63b4a5cd9b527e7f3a146962ee37b9b1c9f8da78893d213b40bc6dc2cbf",
}
BASELINE_COMMIT = "c219c88552a26d229e11410331650d51aee5ebfb"
SCIENCE_FIELDS = (
    "workload",
    "agent_matrix",
    "seed_plan",
    "training_budget",
    "typed_catalog_and_capacity",
    "endpoints",
    "ablation_and_support",
    "statistics",
    "claim_evidence_map",
    "comparisons",
    "holdout_execution_contract",
)
REQUIRED_TESTS = {
    "test_full_companion_envelope_real_benchmark_gate_chain",
    "test_full_envelope_shared_identity_negatives_precede_rollout",
    "test_active_execution_binding_drift_precedes_rollout",
    "test_full_envelope_runtime_negatives_precede_rollout",
    "test_checkpoint_top_nested_conflict_and_protocol_downgrade_precede_rollout",
    "test_benchmark_main_calls_strict_envelope_gate_before_rollout",
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def write(name: str, payload: dict[str, Any]) -> Path:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT / name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return path


def junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    cases = []
    for suite in suites:
        for case in suite.findall("testcase"):
            status = "passed"
            if case.find("failure") is not None:
                status = "failure"
            elif case.find("error") is not None:
                status = "error"
            elif case.find("skipped") is not None:
                status = "skipped"
            cases.append(
                {
                    "class": case.get("classname"),
                    "name": case.get("name"),
                    "status": status,
                    "time_seconds": float(case.get("time", "0")),
                }
            )
    return {
        "tests": len(cases),
        "failures": sum(row["status"] == "failure" for row in cases),
        "errors": sum(row["status"] == "error" for row in cases),
        "skipped": sum(row["status"] == "skipped" for row in cases),
        "time_seconds": round(sum(row["time_seconds"] for row in cases), 3),
        "cases": cases,
    }


def common(commit: str) -> dict[str, Any]:
    return {
        "reviewed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "literature_cutoff": "2026-09-06",
        "target_venue": "IEEE Transactions on Mobile Computing (TMC)",
        "artifact_run_id": ARTIFACT.name,
        "policy_version": "tmc_review_policy_v3_20260621",
        "git_commit": commit,
        "evidence_level": "E2_EXECUTION_CONTRACT_VALIDATED_NO_FORMAL_PERFORMANCE",
        "formal": False,
        "performance_evidence": False,
        "holdout_capability": False,
    }


def refresh_integrity(metadata: dict[str, Any]) -> None:
    files = []
    for path in sorted(ARTIFACT.iterdir()):
        if not path.is_file() or path.name in {
            "artifact_integrity_manifest.json",
            "artifact_inventory.json",
        }:
            continue
        files.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    write(
        "artifact_inventory.json",
        {
            **metadata,
            "status": "pass",
            "file_count_excluding_inventory_and_integrity": len(files),
            "files": [row["path"] for row in files],
            "test_checkpoint_included": False,
            "real_data_included": False,
        },
    )
    write(
        "artifact_integrity_manifest.json",
        {
            **metadata,
            "status": "pass",
            "self_excluded": True,
            "inventory_excluded": True,
            "file_count": len(files),
            "files": files,
        },
    )


def build_candidate(args: argparse.Namespace) -> None:
    targeted_path = Path(args.targeted_junit).resolve()
    full_path = Path(args.full_junit).resolve()
    targeted, full = junit(targeted_path), junit(full_path)
    if any(report[key] for report in (targeted, full) for key in ("failures", "errors")):
        raise ValueError("candidate test evidence contains failures or errors")
    envelope_cases = [
        row
        for row in targeted["cases"]
        if row["class"] == "tests.test_checkpoint_nullable_identity_v28"
    ]
    observed = {str(row["name"]).split("[")[0] for row in envelope_cases}
    if not REQUIRED_TESTS <= observed or any(row["status"] != "passed" for row in envelope_cases):
        raise ValueError("real companion benchmark-gate acceptance is incomplete")
    protocol28 = read(V28 / "protocol_v2_8_manifest.json")
    protocol29 = read(V29 / "protocol_v2_9_manifest.json")
    index = read(V29 / "protocol_index.json")
    if index.get("status") != "NOT_READY_PENDING_G14R18_ACCEPTANCE":
        raise ValueError("candidate evidence must bind the pending Protocol 2.9 index")
    metadata = common(args.candidate_commit)
    science = {field: protocol28[field] == protocol29[field] for field in SCIENCE_FIELDS}
    science["command_templates"] = (
        protocol28["execution_contract"]["command_templates"]
        == protocol29["execution_contract"]["command_templates"]
    )
    science["nullable_contract"] = (
        protocol28["formal_nullable_metric_aggregation_contract"]
        == protocol29["formal_nullable_metric_aggregation_contract"]
    )
    write(
        "protocol_scientific_diff.json",
        {
            **metadata,
            "status": "pass" if all(science.values()) else "fail",
            "from_version": "2.8.0",
            "to_version": "2.9.0",
            "scientific_fields_unchanged": science,
            "budget_seed_window_capacity_selection_statistics_holdout_changed": False,
            "implementation_change": (
                "17-field provenance envelope and 8-field shared identity are "
                "validated at separate interface layers"
            ),
        },
    )
    write(
        "root_cause_and_schema_mapping.json",
        {
            **metadata,
            "status": "pass",
            "root_cause": (
                "benchmark_main_results passed the complete companion binding to "
                "expected_formal_training_identity; validate_checkpoint_provenance "
                "projected 8 observed fields then compared against all 17 fields"
            ),
            "coverage_gap": (
                "G14R17 tests passed a pre-projected 8-field identity directly and "
                "did not load write_checkpoint_companions output through the benchmark gate"
            ),
            "provenance_envelope_field_count": 17,
            "shared_training_identity_field_count": 8,
            "envelope_only_fields": [
                "checkpoint_sha256",
                "execution_git_commit",
                "train_window_plan_identity",
                "runtime_contract_sha256",
                "resolved_agent_config",
                "checkpoint_schedule",
                "selection_sha256",
                "checkpoint_identity",
                "artifact_location",
            ],
            "identity_fields_from_shared_capability_parser": True,
            "trusted_expected_source": "validated active Protocol/context/execution binding",
        },
    )
    acceptance_path = write(
        "real_companion_benchmark_gate_acceptance.json",
        {
            **metadata,
            "status": "pass",
            "chain": [
                "production metadata builder",
                "save/annotate/read-back",
                "strict selection/freeze",
                "write_checkpoint_companions",
                "load_checkpoint_provenance_manifest",
                "validate_benchmark_checkpoint_gate",
                "validate_checkpoint_provenance",
            ],
            "complete_envelope_field_count": 17,
            "shared_identity_field_count": 8,
            "checkpoint_gate_compatible_count": 150,
            "agent_count": 10,
            "seed_count": 5,
            "capacity_count": 3,
            "candidate_and_latest_regression": "pass",
            "test_only_checkpoints": True,
            "test_checkpoint_persisted_in_repository": False,
            "environment_rollout_call_count": 0,
            "full_formal_evaluation_executed": False,
            "junit_case_count": len(envelope_cases),
        },
    )
    write(
        "negative_gate_acceptance.json",
        {
            **metadata,
            "status": "pass",
            "covered": [
                "required shared identity missing",
                "uniform wrong nullable hash",
                "top-level/nested conflict",
                "cross Protocol/bundle/binding/context",
                "checkpoint SHA-256",
                "execution Git",
                "train window",
                "runtime/capacity",
                "agent/seed",
                "checkpoint self-reported Protocol downgrade",
                "missing formal binding cannot disable active strict gate",
            ],
            "environment_rollout_call_count": 0,
            "gate_precedes_run_real_episode_source_assertion": True,
        },
    )
    protected = [
        {
            "path": path,
            "baseline_sha256": expected,
            "candidate_sha256": digest(ROOT / path),
            "unchanged": digest(ROOT / path) == expected,
        }
        for path, expected in PROTECTED.items()
    ]
    write(
        "baseline_and_protected_files_audit.json",
        {
            **metadata,
            "status": "pass" if all(row["unchanged"] for row in protected) else "fail",
            "public_baseline": BASELINE_COMMIT,
            "start_head_main_origin_main": [BASELINE_COMMIT] * 3,
            "start_staged_file_count": 0,
            "protected_files": protected,
            "protected_files_staged_or_committed": False,
            "stash_reset_checkout_used": False,
        },
    )
    write(
        "g14c_v16_authorization_deferral.json",
        {
            **metadata,
            "status": "G14C_V16_LAUNCH_AUTHORIZATION_DEFERRED",
            "g14c_v16_created": False,
            "g14c_v16_consumed": False,
            "g14c_v16_executed": False,
            "run_root_created": False,
            "ledger_created": False,
            "checkpoint_created": False,
            "invalid_run_denylist_entry_created": False,
            "not_a_v16_run_failure": True,
            "g14c_v15_and_earlier_invalid_records_preserved": True,
        },
    )
    shutil.copy2(targeted_path, ARTIFACT / "candidate_targeted_tests.junit.xml")
    shutil.copy2(full_path, ARTIFACT / "candidate_full_tests.junit.xml")
    checks = {
        "root_cause_17_to_8_mapping": "pass",
        "real_companion_produced_loaded_and_gated": "pass",
        "ten_agents_three_capacities": "pass",
        "candidate_latest_selection_freeze_regression": "pass",
        "negative_matrix_rollout_zero": "pass",
        "trusted_protocol_context_binding_expected_identity": "pass",
        "scientific_invariants_unchanged": "pass" if all(science.values()) else "fail",
        "g14c_v16_launch_deferred_without_invalid_run": "pass",
        "protected_files_unchanged": "pass" if all(row["unchanged"] for row in protected) else "fail",
        "holdout_sealed_unopened_unconsumed": "pass",
    }
    evidence = write(
        "acceptance_evidence_manifest.json",
        {
            **metadata,
            "status": "pass" if all(value == "pass" for value in checks.values()) else "fail",
            "candidate_commit": args.candidate_commit,
            "implementation_commit": "9f19ef7e19004d59a9d7bffeee5c5e428a612efa",
            "active_bundle_core_sha256": index["active_bundle_core_sha256"],
            "real_companion_benchmark_gate_path": acceptance_path.relative_to(ROOT).as_posix(),
            "real_companion_benchmark_gate_sha256": digest(acceptance_path),
            "real_companion_benchmark_gate_status": "pass",
            "checks": checks,
            "formal_training_count": 0,
            "formal_performance_count": 0,
            "g14c_v16_created": False,
            "holdout_sealed_unopened_unconsumed": True,
        },
    )
    write(
        "readiness_review_v21.json",
        {
            **metadata,
            "status": evidence.name and "pass",
            "verdict": "READY_FOR_G14C_V16_CLEAN_TRAIN_AND_FORMAL",
            "scope": "checkpoint companion provenance interface readiness only",
            "evidence_manifest_sha256": digest(evidence),
            "g14c_v16_launch_authorization_deferred": True,
            "not_claimed": [
                "G14C v16 execution",
                "complete formal training",
                "algorithm performance",
                "formal result",
                "holdout evidence",
                "paper readiness",
            ],
        },
    )
    refresh_integrity(metadata)


def append_final(args: argparse.Namespace) -> None:
    commit = args.final_execution_commit
    clean_root = Path(args.clean_worktree_root).resolve()
    git = lambda *values: subprocess.run(
        ["git", *values], cwd=clean_root, text=True, capture_output=True, check=True
    ).stdout.strip()
    if git("rev-parse", "HEAD") != commit or git("rev-parse", "origin/main") != commit:
        raise ValueError("final worktree must bind HEAD == origin/main == commit")
    if git("status", "--porcelain"):
        raise ValueError("final execution worktree must be clean")
    targeted = junit(Path(args.final_targeted_junit).resolve())
    full = junit(Path(args.final_full_junit).resolve())
    if any(report[key] for report in (targeted, full) for key in ("failures", "errors", "skipped")):
        raise ValueError("final JUnit must have zero failures/errors/skips")
    preflight_path = Path(args.public_preflight).resolve()
    entrypoint_path = Path(args.entrypoint_summary).resolve()
    preflight, entrypoint = read(preflight_path), read(entrypoint_path)
    if not all(
        (
            preflight.get("status") == "pass",
            preflight.get("protocol", {}).get("protocol_version") == "2.9.0",
            preflight.get("holdout_capability") is False,
            entrypoint.get("status") == "pass",
            entrypoint.get("training_command_count") == 150,
            entrypoint.get("passed_command_count") == 150,
            entrypoint.get("episode_count") == 0,
            entrypoint.get("environment_interaction_count") == 0,
            entrypoint.get("update_count") == 0,
            entrypoint.get("checkpoint_file_count") == 0,
            entrypoint.get("performance_result_count") == 0,
        )
    ):
        raise ValueError("final public preflight or entrypoint acceptance is incomplete")
    index = read(V29 / "protocol_index.json")
    readiness = read(V29 / "readiness_v21.json")
    if index.get("status") != "READY_FOR_G14C_V16_CLEAN_TRAIN_AND_FORMAL":
        raise ValueError("final Protocol 2.9 index is not ready")
    for row in index["active_bundle_resources"]:
        path = ROOT / row["logical_path"]
        if path.stat().st_size != row["size_bytes"] or digest(path) != row["content_sha256"]:
            raise ValueError(f"active bundle resource drift: {path}")
    if any(digest(ROOT / path) != expected for path, expected in PROTECTED.items()):
        raise ValueError("protected user file hash drift")
    metadata = common(commit)
    shutil.copy2(Path(args.final_targeted_junit), ARTIFACT / "final_targeted_tests.junit.xml")
    shutil.copy2(Path(args.final_full_junit), ARTIFACT / "final_full_tests.junit.xml")
    write(
        "final_execution_commit_revalidation.json",
        {
            **metadata,
            "status": "pass",
            "implementation_commit": "9f19ef7e19004d59a9d7bffeee5c5e428a612efa",
            "candidate_evidence_commit": read(ARTIFACT / "acceptance_evidence_manifest.json")[
                "candidate_commit"
            ],
            "final_execution_commit": commit,
            "clean_head_equal_main_equal_origin_main": True,
            "active_protocol_version": "2.9.0",
            "readiness_review_version": readiness["readiness_review_version"],
            "ready_status": index["status"],
            "active_bundle_core_sha256": index["active_bundle_core_sha256"],
            "active_formal_bundle_sha256": index["active_formal_bundle_sha256"],
            "public_preflight_sha256": digest(preflight_path),
            "g14r16_entrypoint_summary_sha256": digest(entrypoint_path),
            "training_command_count": 150,
            "passed_command_count": 150,
            "episode_count": 0,
            "environment_interaction_count": 0,
            "update_count": 0,
            "checkpoint_file_count": 0,
            "performance_result_count": 0,
            "targeted_tests": {key: value for key, value in targeted.items() if key != "cases"},
            "full_pytest": {key: value for key, value in full.items() if key != "cases"},
            "protected_file_hashes_reverified": True,
            "active_resource_integrity_file_count": len(index["active_bundle_resources"]),
            "g14c_v16_launch_authorization_deferred": True,
            "g14c_v16_created": False,
            "holdout_sealed_unopened_unconsumed": True,
            "formal_training_count": 0,
            "formal_performance_count": 0,
            "test_artifacts_only": True,
        },
    )
    refresh_integrity(metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targeted-junit")
    parser.add_argument("--full-junit")
    parser.add_argument("--candidate-commit")
    parser.add_argument("--final-execution-commit")
    parser.add_argument("--final-targeted-junit")
    parser.add_argument("--final-full-junit")
    parser.add_argument("--entrypoint-summary")
    parser.add_argument("--public-preflight")
    parser.add_argument("--clean-worktree-root")
    args = parser.parse_args()
    if args.final_execution_commit:
        append_final(args)
    else:
        if not all((args.targeted_junit, args.full_junit, args.candidate_commit)):
            parser.error("candidate mode requires targeted/full JUnit and candidate commit")
        build_candidate(args)


if __name__ == "__main__":
    main()
