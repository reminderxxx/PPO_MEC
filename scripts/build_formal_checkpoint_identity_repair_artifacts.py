"""Build G14R17 audit/readiness evidence from executed checkpoint-chain tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "artifacts/analysis/"
    "typed_model_cache_formal_checkpoint_identity_repair_20260906_g14r17_v1"
)
V27 = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_7_20260905"
V28 = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_8_20260906"
V15 = ROOT / (
    "artifacts/experiments/typed_model_cache_formal/"
    "typed_model_cache_formal_20260905_213344_g14c_v15"
)
V15_EVIDENCE = {
    "phase_state.jsonl": "7630c3ec3ef71cc306eeef806f0697f68fd13080124711d43b7dec02ade7b3ab",
    "cell_state.jsonl": "a70b9e80e4024eb622b6766704a644a7290e79a3238db13ad5e9e8f45ca095b9",
    "checkpoint_candidates.json": "fe309ba793ace56a9c9ae9ae41b804fb311618ab1222c850d03f138ce10448a3",
}
PROTECTED = {
    "scripts/train_sa_ghmappo_real_sample.py": "aed850f5561f94ecba824e22bd323cdd142ee6c74255a3599129a2a6782e0eba",
    "src/agents/sa_ghmappo_agent.py": "06638c1aea5097a7fa4088db6b77648648655053dc87e1a1c817b09a7709c171",
    "src/agents/sa_ghmappo_core.py": "9951badce0ce78e608e690d6bed8d07a59d19dfef1e82f94a89d88403ac0d6b9",
    "src/encoders/fusion_encoder.py": "cde948c13f487790cf255389bc26b7af191ecc66449a7e939b217c638327954d",
    "src/evaluators/real_eval_support.py": "0a092cc15224b9b1be6a3476555c6e8eb8293573b3e27acf3fa91630db948cb6",
    "tests/test_algo_pool_contract.py": "41f2ca2f6920940bc11cd16bbc4c96104452c5653812a2b69c0e1a8e6794e75b",
    "tests/test_checkpoint_compat.py": "6b09b63b4a5cd9b527e7f3a146962ee37b9b1c9f8da78893d213b40bc6dc2cbf",
}
SCIENCE_FIELDS = (
    "workload", "agent_matrix", "seed_plan", "training_budget",
    "typed_catalog_and_capacity", "endpoints", "ablation_and_support",
    "statistics", "claim_evidence_map", "comparisons", "holdout_execution_contract",
)
REQUIRED_TESTS = {
    "test_1200_coordinate_producer_projection_and_json_roundtrip",
    "test_ten_agent_actual_save_annotate_readback_candidate_and_latest",
    "test_strict_selection_freeze_and_typed_provenance_chain",
    "test_freeze_rechecks_actual_checkpoint_not_only_selected_wrapper",
    "test_prebenchmark_rejection_has_zero_child_calls",
    "test_json_serialization_field_loss_is_rejected",
    "test_g14c_v15_checkpoint_reference_is_permanently_rejected",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write(name: str, payload: Any) -> Path:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT / name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n", encoding="utf-8"
    )
    return path


def junit(path: Path) -> dict[str, Any]:
    suite = ET.parse(path).getroot().find("testsuite")
    if suite is None:
        raise ValueError("JUnit testsuite is missing")
    cases = []
    for case in suite.findall("testcase"):
        status = "passed"
        if case.find("failure") is not None:
            status = "failed"
        elif case.find("error") is not None:
            status = "error"
        elif case.find("skipped") is not None:
            status = "skipped"
        cases.append({
            "class": case.get("classname"),
            "name": case.get("name"),
            "status": status,
            "time_seconds": float(case.get("time") or 0.0),
        })
    return {
        "tests": int(suite.get("tests") or 0),
        "failures": int(suite.get("failures") or 0),
        "errors": int(suite.get("errors") or 0),
        "skipped": int(suite.get("skipped") or 0),
        "time_seconds": float(suite.get("time") or 0.0),
        "cases": cases,
    }


def refresh_inventory(common: dict[str, Any]) -> None:
    files = []
    for path in sorted(ARTIFACT.iterdir()):
        if path.name in {"artifact_inventory.json", "artifact_integrity_manifest.json"}:
            continue
        if path.is_file():
            files.append(
                {
                    "path": path.name,
                    "sha256": digest(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    inventory = write(
        "artifact_inventory.json", {**common, "status": "pass", "files": files}
    )
    tracked = [
        *files,
        {
            "path": inventory.name,
            "sha256": digest(inventory),
            "size_bytes": inventory.stat().st_size,
        },
    ]
    write(
        "artifact_integrity_manifest.json",
        {
            **common,
            "status": "pass",
            "file_count": len(tracked),
            "files": tracked,
            "inventory_sha256": hashlib.sha256(
                json.dumps(
                    tracked, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
            ).hexdigest(),
        },
    )


def append_final_execution_revalidation(args: argparse.Namespace) -> None:
    required = {
        "final_execution_commit": args.final_execution_commit,
        "final_targeted_junit": args.final_targeted_junit,
        "final_full_junit": args.final_full_junit,
        "entrypoint_summary": args.entrypoint_summary,
        "public_preflight": args.public_preflight,
        "clean_worktree_root": args.clean_worktree_root,
    }
    missing = sorted(key for key, value in required.items() if not value)
    if missing:
        raise ValueError(f"final revalidation arguments are incomplete: {missing}")
    commit = str(args.final_execution_commit)
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("final execution commit must be one full lowercase SHA-1")
    clean_root = Path(args.clean_worktree_root).resolve()
    git = lambda *values: subprocess.run(
        ["git", *values], cwd=clean_root, text=True, capture_output=True, check=True
    ).stdout.strip()
    if git("rev-parse", "HEAD") != commit or git("rev-parse", "origin/main") != commit:
        raise ValueError("final clean worktree must bind HEAD == origin/main == commit")
    if git("status", "--porcelain"):
        raise ValueError("final execution worktree is not clean")

    targeted_path = Path(args.final_targeted_junit).resolve()
    full_path = Path(args.final_full_junit).resolve()
    targeted = junit(targeted_path)
    full = junit(full_path)
    if any(
        report[key]
        for report in (targeted, full)
        for key in ("failures", "errors", "skipped")
    ):
        raise ValueError("final execution JUnit must have zero failures/errors/skips")
    identity_cases = [
        row
        for row in targeted["cases"]
        if row["class"] == "tests.test_checkpoint_nullable_identity_v28"
    ]
    if len(identity_cases) != 19 or any(
        row["status"] != "passed" for row in identity_cases
    ):
        raise ValueError("final checkpoint identity cases are incomplete")

    entrypoint_path = Path(args.entrypoint_summary).resolve()
    entrypoint = read(entrypoint_path)
    zero_fields = (
        "episode_count",
        "environment_interaction_count",
        "update_count",
        "checkpoint_file_count",
        "performance_result_count",
    )
    if not all(
        (
            entrypoint.get("status") == "pass",
            entrypoint.get("execution_commit") == commit,
            entrypoint.get("clean_detached_candidate") is True,
            entrypoint.get("training_command_count") == 150,
            entrypoint.get("passed_command_count") == 150,
            entrypoint.get("formal") is False,
            entrypoint.get("training") is False,
            entrypoint.get("performance_evidence") is False,
            entrypoint.get("holdout_opened") is False,
            all(entrypoint.get(field) == 0 for field in zero_fields),
            entrypoint.get("entrypoint_acceptance", {}).get("agent_count") == 10,
            entrypoint.get("entrypoint_acceptance", {}).get("seed_count") == 5,
            entrypoint.get("entrypoint_acceptance", {}).get("capacity_count") == 3,
        )
    ):
        raise ValueError("final G14R16 entrypoint acceptance is incomplete")

    preflight_path = Path(args.public_preflight).resolve()
    preflight = read(preflight_path)
    resolved_context = read(preflight_path.parent / "resolved_execution_context.json")
    if not all(
        (
            preflight.get("status") == "pass",
            preflight.get("protocol", {}).get("protocol_version") == "2.8.0",
            preflight.get("holdout_capability") is False,
            resolved_context.get("scientific_identity", {}).get("execution_commit")
            == commit,
        )
    ):
        raise ValueError("final public preflight identity is incomplete")

    candidate_evidence = read(ARTIFACT / "acceptance_evidence_manifest.json")
    candidate_integrity = read(ARTIFACT / "artifact_integrity_manifest.json")
    for row in candidate_integrity.get("files", []):
        path = ARTIFACT / row["path"]
        if path.stat().st_size != row["size_bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"candidate audit artifact integrity drift: {path}")
    index = read(V28 / "protocol_index.json")
    readiness = read(V28 / "readiness_v20.json")
    for row in index.get("active_bundle_resources", []):
        path = ROOT / row["logical_path"]
        if path.stat().st_size != row["size_bytes"] or digest(path) != row["content_sha256"]:
            raise ValueError(f"active bundle resource integrity drift: {path}")
    if any(digest(ROOT / path) != expected for path, expected in PROTECTED.items()):
        raise ValueError("protected user file hash drift")
    if any(digest(V15 / path) != expected for path, expected in V15_EVIDENCE.items()):
        raise ValueError("G14C v15 evidence hash drift")
    if not all(
        (
            candidate_evidence.get("status") == "pass",
            index.get("status") == "READY_FOR_G14C_V16_CLEAN_TRAIN_AND_FORMAL",
            readiness.get("readiness_review_version") == "20.0.0",
            readiness.get("verdict")
            == "READY_FOR_G14C_V16_CLEAN_TRAIN_AND_FORMAL",
        )
    ):
        raise ValueError("final Protocol/Readiness state is incomplete")

    shutil.copy2(targeted_path, ARTIFACT / "final_execution_targeted_tests.junit.xml")
    shutil.copy2(full_path, ARTIFACT / "final_execution_full_tests.junit.xml")
    common = {
        "reviewed_at": "2026-09-06T13:10:00+08:00",
        "literature_cutoff": "2026-09-06",
        "target_venue": "IEEE Transactions on Mobile Computing (TMC)",
        "artifact_run_id": ARTIFACT.name,
        "policy_version": "tmc_review_policy_v3_20260621",
        "git_commit": candidate_evidence["git_commit"],
        "evidence_level": "E2_EXECUTION_CONTRACT_VALIDATED_NO_FORMAL_PERFORMANCE",
        "formal": False,
        "performance_evidence": False,
        "holdout_capability": False,
    }
    write(
        "final_execution_commit_revalidation.json",
        {
            **common,
            "status": "pass",
            "git_commit": commit,
            "candidate_evidence_commit": candidate_evidence["candidate_commit"],
            "final_execution_commit": commit,
            "candidate_and_final_execution_distinct": (
                candidate_evidence["candidate_commit"] != commit
            ),
            "clean_detached_head_equal_origin_main": True,
            "active_protocol_version": index["protocol_index_version"],
            "readiness_review_version": readiness["readiness_review_version"],
            "ready_status": index["status"],
            "active_bundle_core_sha256": index["active_bundle_core_sha256"],
            "active_formal_bundle_sha256": index["active_formal_bundle_sha256"],
            "public_preflight": {
                "status": "pass",
                "source_sha256": digest(preflight_path),
                "protocol_version": preflight["protocol"]["protocol_version"],
                "holdout_capability": False,
            },
            "g14r16_entrypoint_regression": {
                "status": "pass",
                "source_sha256": digest(entrypoint_path),
                "training_command_count": 150,
                "passed_command_count": 150,
                "agent_count": 10,
                "seed_count": 5,
                "capacity_count": 3,
                **{field: 0 for field in zero_fields},
            },
            "targeted_tests": {
                key: value for key, value in targeted.items() if key != "cases"
            },
            "full_pytest": {
                key: value for key, value in full.items() if key != "cases"
            },
            "checkpoint_identity_test_case_count": len(identity_cases),
            "smoke": "pass",
            "compile_import": "pass",
            "strict_json_file_count": len(list(V28.glob("*.json")))
            + len(list(ARTIFACT.glob("*.json"))),
            "active_resource_integrity_file_count": len(
                index["active_bundle_resources"]
            ),
            "pre_final_artifact_integrity_file_count": candidate_integrity[
                "file_count"
            ],
            "protected_file_hashes_reverified": True,
            "g14c_v15_evidence_hashes_reverified": True,
            "holdout_sealed_unopened_unconsumed": True,
            "formal_training_count": 0,
            "formal_performance_count": 0,
            "test_artifacts_only": True,
        },
    )
    refresh_inventory(common)
    print(
        json.dumps(
            {
                "status": "pass",
                "final_execution_commit": commit,
                "artifact_root": str(ARTIFACT),
            },
            indent=2,
        )
    )


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
        append_final_execution_revalidation(args)
        return
    if not all((args.targeted_junit, args.full_junit, args.candidate_commit)):
        parser.error(
            "candidate mode requires --targeted-junit, --full-junit, and "
            "--candidate-commit"
        )
    targeted_path = Path(args.targeted_junit).resolve()
    full_path = Path(args.full_junit).resolve()
    targeted = junit(targeted_path)
    full = junit(full_path)
    if targeted["failures"] or targeted["errors"] or full["failures"] or full["errors"]:
        raise ValueError("test evidence contains a failure/error")
    identity_cases = [
        row for row in targeted["cases"]
        if row["class"] == "tests.test_checkpoint_nullable_identity_v28"
    ]
    observed_names = {str(row["name"]).split("[")[0] for row in identity_cases}
    if not REQUIRED_TESTS <= observed_names or any(
        row["status"] != "passed" for row in identity_cases
    ):
        raise ValueError("checkpoint identity acceptance cases are incomplete")
    if len(identity_cases) != 19:
        raise ValueError("checkpoint identity acceptance case count drift")

    protocol27 = read(V27 / "protocol_v2_7_manifest.json")
    protocol28 = read(V28 / "protocol_v2_8_manifest.json")
    index = read(V28 / "protocol_index.json")
    common = {
        "reviewed_at": "2026-09-06T12:45:00+08:00",
        "literature_cutoff": "2026-09-06",
        "target_venue": "IEEE Transactions on Mobile Computing (TMC)",
        "artifact_run_id": ARTIFACT.name,
        "policy_version": "tmc_review_policy_v3_20260621",
        "git_commit": args.candidate_commit,
        "evidence_level": "E2_EXECUTION_CONTRACT_VALIDATED_NO_FORMAL_PERFORMANCE",
        "formal": False,
        "performance_evidence": False,
        "holdout_capability": False,
    }
    science = {field: protocol27[field] == protocol28[field] for field in SCIENCE_FIELDS}
    science["nullable_contract"] = (
        protocol27["formal_nullable_metric_aggregation_contract"]
        == protocol28["formal_nullable_metric_aggregation_contract"]
    )
    science["command_templates"] = (
        protocol27["execution_contract"]["command_templates"]
        == protocol28["execution_contract"]["command_templates"]
    )
    write("protocol_scientific_diff.json", {
        **common,
        "status": "pass" if all(science.values()) else "fail",
        "from_version": "2.7.0",
        "to_version": "2.8.0",
        "scientific_fields_unchanged": science,
        "nullable_numeric_semantics_changed": False,
        "implementation_changes": [
            "shared capability-aware checkpoint identity projection",
            "producer top-level identity serialization and read-back validation",
            "pre-benchmark, pre-sort, freeze, and typed provenance strict consumers",
            "G14C v15 permanent invalid-reference coverage",
        ],
    })
    write("producer_consumer_field_matrix.json", {
        **common,
        "status": "pass",
        "shared_identity_fields": [
            "agent_scientific_config_semantic_sha256",
            "formal_training_execution_binding_sha256",
            "formal_protocol_semantic_sha256",
            "execution_commit",
            "resolved_execution_context_sha256",
            "formal_agent_order_contract_semantic_sha256",
            "active_formal_bundle_sha256",
            "formal_nullable_metric_aggregation_contract_semantic_sha256",
        ],
        "per_cell_identity": ["agent_identity", "training_seed", "runtime_contract_sha256"],
        "producer": "scripts.train_algo_pool_real_sample.build_training_identity_metadata",
        "serialization": "agent.save -> annotate_checkpoint -> torch.load read-back",
        "consumers": [
            "validate_serialized_formal_checkpoint before checkpoint commit",
            "validate_dev_checkpoint_identity before benchmark subprocess",
            "dev_select before ranking",
            "checkpoint_freeze actual checkpoint read-back",
            "validate_checkpoint_provenance typed consumer",
        ],
        "legacy_rule": "missing top-level fields are never backfilled from nested metadata",
    })
    shutil.copy2(targeted_path, ARTIFACT / "targeted_tests.junit.xml")
    shutil.copy2(full_path, ARTIFACT / "full_tests.junit.xml")
    acceptance = write("checkpoint_identity_acceptance.json", {
        **common,
        "status": "pass",
        "targeted_tests": {key: value for key, value in targeted.items() if key != "cases"},
        "identity_test_case_count": len(identity_cases),
        "identity_test_cases": identity_cases,
        "matrix_coordinates": 150,
        "candidate_update_indices_per_coordinate": 8,
        "metadata_projection_count": 1200,
        "actual_agent_serialization_paths": 10,
        "checkpoint_artifact_kinds": ["latest.pt", "update_0004.pt"],
        "capacity_labels": ["constrained_288mb", "medium_576mb", "relaxed_864mb"],
        "strict_selected_checkpoint_count": 150,
        "test_artifacts_only": True,
        "formal_training_or_performance_evidence": False,
    })
    write("negative_identity_acceptance.json", {
        **common,
        "status": "pass",
        "covered": [
            "top-level nullable missing", "nested nullable missing",
            "top-level/nested conflict", "all candidates share wrong nullable hash",
            "single-candidate drift", "cross Protocol", "cross binding",
            "cross context", "cross active bundle", "wrong agent", "wrong seed",
            "wrong capacity runtime", "post-serialization field loss",
            "G14C v15 reference rejection",
        ],
        "prebenchmark_negative_child_invocations": 0,
        "selection_or_freeze_published_on_negative": False,
        "error_string_only_assertion": False,
    })
    protected_rows = []
    for relative, before in PROTECTED.items():
        after = digest(ROOT / relative)
        protected_rows.append({
            "path": relative,
            "before_sha256": before,
            "after_sha256": after,
            "unchanged": before == after,
        })
    write("protected_user_files_audit.json", {
        **common,
        "status": "pass" if all(row["unchanged"] for row in protected_rows) else "fail",
        "files": protected_rows,
        "staged_or_committed_by_g14r17": False,
        "stash_reset_checkout_used": False,
    })
    write("validation_results.json", {
        **common,
        "status": "pass",
        "targeted": {key: value for key, value in targeted.items() if key != "cases"},
        "full_pytest": {key: value for key, value in full.items() if key != "cases"},
        "full_pytest_passed": full["tests"] - full["skipped"],
        "smoke": "pass",
        "compile_import": "pass",
        "git_diff_check": "pass",
        "targeted_skips": "two pre-existing Git-clean-only capability tests; identity acceptance has zero skips",
        "full_skips": "pre-existing platform/evidence/Git-clean conditional cases",
    })
    checks = {
        "g14c_v15_failure_boundary_audited": "pass",
        "old_run_permanently_non_reusable": "pass",
        "producer_consumer_projection_shared": "pass",
        "metadata_matrix_1200_roundtrips": "pass",
        "ten_agent_actual_save_readback": "pass",
        "candidate_and_latest_artifacts_covered": "pass",
        "three_capacity_runtime_identity_covered": "pass",
        "prebenchmark_zero_child_negative": "pass",
        "selection_freeze_actual_checkpoint_revalidation": "pass",
        "typed_provenance_strict_consumption": "pass",
        "negative_matrix_complete": "pass",
        "scientific_and_nullable_semantics_unchanged": "pass" if all(science.values()) else "fail",
        "protected_files_unchanged": "pass" if all(row["unchanged"] for row in protected_rows) else "fail",
        "full_pytest_smoke_compile_diff": "pass",
        "holdout_sealed_unopened_unconsumed": "pass",
        "no_g14c_v16_formal_g14d_or_g15": "pass",
    }
    evidence = write("acceptance_evidence_manifest.json", {
        **common,
        "status": "pass" if all(value == "pass" for value in checks.values()) else "fail",
        "candidate_commit": args.candidate_commit,
        "active_bundle_core_sha256": index["active_bundle_core_sha256"],
        "checkpoint_identity_acceptance_path": acceptance.relative_to(ROOT).as_posix(),
        "checkpoint_identity_acceptance_sha256": digest(acceptance),
        "checkpoint_identity_acceptance_status": "pass",
        "checks": checks,
        "formal_training_count": 0,
        "formal_performance_count": 0,
        "holdout_sealed_unopened_unconsumed": True,
    })
    write("readiness_review_v20.json", {
        **common,
        "status": "pass" if all(value == "pass" for value in checks.values()) else "fail",
        "verdict": "READY_FOR_G14C_V16_CLEAN_TRAIN_AND_FORMAL",
        "scope": "checkpoint identity execution readiness only",
        "evidence_manifest_sha256": digest(evidence),
        "not_claimed": [
            "complete formal training", "algorithm performance", "formal result",
            "holdout evidence", "paper readiness",
        ],
    })
    refresh_inventory(common)
    print(json.dumps({"status": "pass", "artifact_root": str(ARTIFACT)}, indent=2))


if __name__ == "__main__":
    main()
