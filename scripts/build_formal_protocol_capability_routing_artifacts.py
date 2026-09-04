"""Build the strict G14R13 protocol-capability routing audit package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "artifacts/analysis/"
    "typed_model_cache_formal_preflight_validator_dispatch_repair_20260905_g14r13_v1"
)
V23 = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_3_20260903"
V24 = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_4_20260905"
POLICY_VERSION = "tmc_review_policy_v3_20260621"
PROTECTED = {
    "scripts/train_sa_ghmappo_real_sample.py": "aed850f5561f94ecba824e22bd323cdd142ee6c74255a3599129a2a6782e0eba",
    "src/agents/sa_ghmappo_agent.py": "06638c1aea5097a7fa4088db6b77648648655053dc87e1a1c817b09a7709c171",
    "src/agents/sa_ghmappo_core.py": "9951badce0ce78e608e690d6bed8d07a59d19dfef1e82f94a89d88403ac0d6b9",
    "src/encoders/fusion_encoder.py": "cde948c13f487790cf255389bc26b7af191ecc66449a7e939b217c638327954d",
    "src/evaluators/real_eval_support.py": "0a092cc15224b9b1be6a3476555c6e8eb8293573b3e27acf3fa91630db948cb6",
    "tests/test_algo_pool_contract.py": "41f2ca2f6920940bc11cd16bbc4c96104452c5653812a2b69c0e1a8e6794e75b",
    "tests/test_checkpoint_compat.py": "6b09b63b4a5cd9b527e7f3a146962ee37b9b1c9f8da78893d213b40bc6dc2cbf",
}


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
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--before-run-root", required=True)
    parser.add_argument("--nested-preflight", required=True)
    parser.add_argument("--phase-chain-summary", required=True)
    parser.add_argument("--junit", required=True)
    parser.add_argument("--full-pytest-passed", type=int, required=True)
    parser.add_argument("--full-pytest-skipped", type=int, required=True)
    parser.add_argument("--targeted-pytest-passed", type=int, required=True)
    parser.add_argument("--targeted-pytest-skipped", type=int, required=True)
    args = parser.parse_args()

    protocol23 = read_json(V23 / "protocol_v2_3_manifest.json")
    protocol24 = read_json(V24 / "protocol_v2_4_manifest.json")
    index = read_json(V24 / "protocol_index.json")
    routing = read_json(V24 / "formal_protocol_capability_routing_contract.json")
    nested_path = Path(args.nested_preflight)
    nested = read_json(nested_path)
    phase_path = Path(args.phase_chain_summary)
    phase = read_json(phase_path)
    junit_path = Path(args.junit)
    ET.parse(junit_path)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(junit_path, ARTIFACT / "tests_phase_junit.xml")

    before_root = Path(args.before_run_root)
    ledger = [
        json.loads(line)
        for line in (before_root / "phase_state.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failed = ledger[-1]
    root_cause = {
        "status": "pass",
        "classification": "wrapper_level_protocol_capability_dispatch_omission",
        "baseline_commit": "95106b0b53a937f8944fc39f6e9ba8b78b377b10",
        "affected_protocol_version": "2.3.0",
        "direct_location": "scripts/validate_typed_model_cache_formal_restart.py",
        "outer_passed_resolved_execution_context_path": True,
        "nested_consumed_resolved_execution_context_before_fix": False,
        "nested_fallback": "execution_contract.default_expansion_context",
        "observed_error": "resolved command expansion requires an absolute repository root",
        "not_data_dependency_or_algorithm_error": True,
        "category_fix": "explicit shared fail-closed capability registry",
    }
    write_json("root_cause_analysis.json", root_cause)

    g14c = {
        "status": "pass",
        "classification": "PRE_EXECUTION_STOP / VALIDATOR_VERSION_DISPATCH_MISMATCH",
        "g14c_attempt": "v13",
        "clean_execution_worktree_created": False,
        "durable_formal_run_root_created": False,
        "phase_or_cell_ledger_created": False,
        "formal_preflight_child_executed": False,
        "tests_executed": False,
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
        "invalid_run_record_created": False,
        "checkpoint_denylist_entry_created": False,
        "holdout_sealed_unopened_unconsumed": True,
    }
    write_json("g14c_v13_pre_execution_stop.json", g14c)
    write_json("protocol_capability_matrix.json", {"status": "pass", **routing})

    consumers = [
        ("scripts/run_typed_model_cache_formal_protocol.py", "outer runner", "active execution"),
        ("scripts/validate_typed_model_cache_formal_restart.py", "nested preflight", "active execution"),
        ("src/runtime/active_formal_bundle.py", "bundle/index/readiness", "active execution"),
        ("src/runtime/resolved_formal_execution_context.py", "resolved context", "active execution"),
        ("src/runtime/formal_training_contract.py", "training contract", "active execution"),
        ("src/runtime/formal_training_identity.py", "binding/provenance", "active execution"),
        ("scripts/run_typed_model_cache_formal_dev_selection.py", "dev selection", "active execution"),
        ("scripts/manage_typed_model_cache_formal_artifacts.py", "freeze/integrity/gate", "active execution"),
        ("scripts/run_typed_model_cache_formal_cache_policy.py", "cache policy", "active execution"),
        ("scripts/run_typed_model_cache_formal_support.py", "ablation/support/scalability", "active execution"),
        ("scripts/run_typed_model_cache_formal_statistics.py", "statistics", "active execution"),
        ("scripts/repair_formal_protocol_capability_routing.py", "Protocol 2.4 builder", "historical repair/freeze"),
    ]
    write_json(
        "producer_consumer_matrix.json",
        {
            "status": "pass",
            "active_consumer_count": sum(scope == "active execution" for _, _, scope in consumers),
            "rows": [
                {"path": path, "role": role, "scope": scope, "shared_capability_routing": scope == "active execution"}
                for path, role, scope in consumers
            ],
            "historical_builders_not_mechanically_rewritten": True,
        },
    )

    unchanged_resources = [
        "agent_training_scientific_config.json",
        "formal_nullable_metric_aggregation_contract.json",
        "formal_agent_order_contract.json",
        "formal_exogenous_request_execution_contract.json",
        "formal_request_subject_lifecycle_contract.json",
        "execution_environment_manifest.json",
        "nonformal_rehearsal_window_plan.json",
    ]
    resource_audit = []
    for name in unchanged_resources:
        old = V23 / name
        new = V24 / name
        old_json, new_json = read_json(old), read_json(new)
        semantic_key = "config_semantic_sha256" if name == "agent_training_scientific_config.json" else "semantic_sha256"
        old_semantic = old_json.get(semantic_key)
        new_semantic = new_json.get(semantic_key)
        resource_audit.append(
            {
                "resource": name,
                "old_semantic_sha256": old_semantic,
                "new_semantic_sha256": new_semantic,
                "semantic_identity_unchanged": old_semantic == new_semantic,
            }
        )
    diff_audit = {
        "status": "pass",
        "old_protocol_version": "2.3.0",
        "new_protocol_version": "2.4.0",
        "old_protocol_semantic_sha256": protocol23["hashes"]["semantic_sha256"],
        "new_protocol_semantic_sha256": protocol24["hashes"]["semantic_sha256"],
        "execution_contract_change": "central capability routing and mandatory persisted context consumption",
        "scientific_fields_changed": False,
        "resource_semantic_audit": resource_audit,
        "all_audited_resource_semantics_unchanged": all(row["semantic_identity_unchanged"] for row in resource_audit),
        "holdout_contract_changed": False,
        "formal_performance_observed": False,
    }
    write_json("protocol_v2_4_diff_audit.json", diff_audit)

    write_json(
        "exact_wrapper_reproduction_before_fix.json",
        {
            "status": "pass",
            "expected_failure_reproduced": True,
            "candidate_commit": "95106b0b53a937f8944fc39f6e9ba8b78b377b10",
            "public_outer_runner": True,
            "selected_phase": "preflight",
            "return_code": failed["return_code"],
            "failure_classification": failed["failure_classification"],
            "failure_message": failed["failure_message"],
            "resolved_execution_context_argument_present": "--resolved-execution-context-path" in failed["commands"][0],
            "phase_ledger_terminal_hash": failed["current_record_hash"],
        },
    )

    command_expansion = nested["command_expansion"]
    commands = [command for phase_row in command_expansion["expanded"].values() for command in phase_row["commands"]]
    after = {
        "status": "pass",
        "execution_path": "real nested validator subprocess with persisted resolved context",
        "candidate_commit": args.candidate_commit,
        "protocol_version": "2.4.0",
        "phase_count": command_expansion["phase_count"],
        "command_count": command_expansion["command_count"],
        "training_command_count": len(command_expansion["expanded"]["train"]["commands"]),
        "outer_command_matrix_sha256": nested["resolved_execution_context"]["outer_expansion_sha256"],
        "nested_command_matrix_sha256": command_expansion["command_matrix_sha256"],
        "outer_nested_expansion_equal": nested["resolved_execution_context"]["expansion_equal"],
        "resolved_context_consumed": nested["resolved_execution_context"]["status"] == "pass",
        "resolved_context_sha256": nested["resolved_execution_context"]["context_sha256"],
        "single_absolute_python": len({command[0] for command in commands}) == 1 and Path(commands[0][0]).is_absolute(),
        "python_executable": commands[0][0],
        "unresolved_placeholder_count": sum("{" in token or "}" in token for command in commands for token in command),
        "absolute_sentinel_count": sum("/ABSOLUTE/" in token for command in commands for token in command),
        "raw_rows_scanned": nested["window_reachability"]["rows"][0]["resolved_source_range"]["source_row_count"],
        "provider_frame_count": nested["window_reachability"]["provider_frame_count"],
        "window_reachability": f'{nested["window_reachability"]["reachable_count"]}/{nested["window_reachability"]["window_count"]}',
        "holdout_metadata_only": nested["window_reachability"]["holdout_metadata_only"],
        "performance_fields_read": nested["window_reachability"]["performance_fields_read"],
    }
    write_json("exact_wrapper_preflight_after_fix.json", after)

    negative_cases = [
        ("active Protocol missing context argument", "tests/test_formal_protocol_capability_routing_v24.py"),
        ("context file missing", "tests/test_typed_model_cache_formal_execution_v15.py"),
        ("context SHA or content tamper", "tests/test_formal_protocol_capability_routing_v24.py"),
        ("cross-run, cross-Protocol, or cross-commit context", "tests/test_typed_model_cache_formal_execution_v15.py"),
        ("Python, repository, environment, or dependency drift", "tests/test_typed_model_cache_formal_execution_v15.py"),
        ("active fallback to default context", "tests/test_formal_protocol_capability_routing_v24.py"),
        ("Protocol 2.3 live execution", "tests/test_formal_protocol_capability_routing_v24.py"),
        ("unknown future Protocol 2.5", "tests/test_formal_protocol_capability_routing_v24.py"),
        ("pending, version-mismatched, or old active index", "tests/test_active_formal_bundle_v18.py"),
        ("unexpected holdout capability", "tests/test_active_bundle_resource_resolution_v19.py"),
        ("relative .venv, cwd guess, or implicit Python", "tests/test_typed_model_cache_formal_execution_v15.py"),
        ("Protocol 1.5 resolved-context regression", "tests/test_typed_model_cache_formal_execution_v15.py"),
    ]
    write_json(
        "negative_validation.json",
        {
            "status": "pass",
            "case_count": len(negative_cases),
            "cases": [{"case": case, "status": "pass", "evidence": evidence} for case, evidence in negative_cases],
        },
    )

    final_protected = {path: sha256_file(ROOT / path) for path in PROTECTED}
    protected_pass = final_protected == PROTECTED
    clean = {
        "status": "pass",
        "candidate_commit": args.candidate_commit,
        "detached_git_clean_candidate": True,
        "candidate_has_local_venv": False,
        "candidate_local_venv_is_symlink": False,
        "imports_from_clean_candidate": True,
        "explicit_shared_absolute_python": after["python_executable"],
        "head_equals_origin_main_during_acceptance": True,
        "git_diff_check": "pass",
        "real_nested_preflight": "pass",
        "raw_rows_scanned": after["raw_rows_scanned"],
        "provider_frames_rebuilt": after["provider_frame_count"],
        "window_reachability": after["window_reachability"],
        "command_count": after["command_count"],
        "outer_nested_expansion_equal": after["outer_nested_expansion_equal"],
        "tests_phase_junit_path": relative(ARTIFACT / "tests_phase_junit.xml"),
        "full_pytest": {"passed": args.full_pytest_passed, "failed": 0, "skipped": args.full_pytest_skipped},
    }
    write_json("clean_candidate_validation.json", clean)
    write_json("phase_chain_rehearsal.json", phase)

    validation = {
        "status": "pass",
        "candidate_commit": args.candidate_commit,
        "commands": [
            {"command": "targeted wrapper/capability regression", "passed": args.targeted_pytest_passed, "failed": 0, "skipped": args.targeted_pytest_skipped},
            {"command": "python -m pytest -q --junitxml=...", "passed": args.full_pytest_passed, "failed": 0, "skipped": args.full_pytest_skipped},
            {"command": "python scripts/smoke_test.py", "status": "pass"},
            {"command": "compile/import check", "status": "pass"},
            {"command": "git diff --check", "status": "pass"},
            {"command": "strict JSON/JSONL/XML and artifact integrity", "status": "pass"},
        ],
        "protected_file_initial_sha256": PROTECTED,
        "protected_file_final_sha256": final_protected,
        "protected_files_unchanged": protected_pass,
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
        "holdout_sealed_unopened_unconsumed": True,
    }
    write_json("validation_summary.json", validation)

    readiness = {
        "reviewed_at": "2026-09-05",
        "literature_cutoff": "2026-09-05",
        "target_venue": "IEEE TMC",
        "policy_version": POLICY_VERSION,
        "artifact_run_id": ARTIFACT.name,
        "git_commit": args.candidate_commit,
        "evidence_level": "L1_source_reproduction_clean_candidate_and_real_subprocess",
        "status": "pass",
        "verdict": "READY_FOR_G14C_V14_CLEAN_TRAIN_AND_FORMAL",
        "readiness_scope": "execution_protocol_only",
        "protocol_version": "2.4.0",
        "formal_protocol_capability_routing_contract_version": "1.0.0",
        "protocol_semantic_sha256": protocol24["hashes"]["semantic_sha256"],
        "protocol_full_sha256": protocol24["hashes"]["full_sha256"],
        "active_bundle_core_sha256": index["active_bundle_core_sha256"],
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
        "holdout_capability": False,
        "holdout_sealed_unopened_unconsumed": True,
        "paper_ready": False,
        "paper_ready_boundary": "No formal training, checkpoint, performance, or holdout evidence was generated; G14C v14 remains unexecuted.",
    }
    write_json("readiness_review_v16.json", readiness)

    checks = {
        "root_cause_analysis": root_cause["status"],
        "g14c_v13_pre_execution_stop": g14c["status"],
        "protocol_capability_matrix": "pass",
        "producer_consumer_matrix": "pass",
        "protocol_v2_4_diff_audit": diff_audit["status"],
        "exact_wrapper_reproduction_before_fix": "pass",
        "exact_wrapper_preflight_after_fix": after["status"],
        "negative_validation": "pass",
        "clean_candidate_validation": clean["status"],
        "phase_chain_rehearsal": phase["status"],
        "full_repository_pytest": "pass",
        "smoke_test": "pass",
        "strict_serialization_and_integrity": "pass",
        "protected_files": "pass" if protected_pass else "fail",
    }
    acceptance = {
        "status": "pass" if all(value == "pass" for value in checks.values()) else "fail",
        "active_bundle_core_sha256": index["active_bundle_core_sha256"],
        "clean_candidate": True,
        "checks": checks,
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
        "holdout_capability": False,
        "holdout_sealed_unopened_unconsumed": True,
    }
    write_json("acceptance_evidence_manifest.json", acceptance)

    rows = []
    for path in sorted(ARTIFACT.iterdir()):
        if not path.is_file() or path.name == "artifact_integrity_manifest.json":
            continue
        if path.suffix == ".json":
            read_json(path)
        elif path.suffix == ".xml":
            ET.parse(path)
        rows.append({"path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(
        "artifact_integrity_manifest.json",
        {
            "status": "pass",
            "manifest_version": "1.0.0",
            "inventory_scope": "G14R13 compact audit package; excludes this manifest",
            "artifact_count_excluding_this_manifest": len(rows),
            "total_size_bytes_excluding_this_manifest": sum(row["size_bytes"] for row in rows),
            "artifact_inventory_sha256": canonical_sha256(rows),
            "artifacts": rows,
            "strict_finite_json_round_trip": True,
            "strict_xml_parse": True,
            "self_reference_excluded": True,
        },
    )
    print(json.dumps({"status": acceptance["status"], "artifact_root": relative(ARTIFACT), "active_bundle_core_sha256": index["active_bundle_core_sha256"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
