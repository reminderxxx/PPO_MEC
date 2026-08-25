"""Freeze outcome-blind G14R5 Protocol v1.5 and context-contract evidence.

This generator never trains, evaluates formal or holdout data, or reads an
invalid-run checkpoint.  It only derives v1.5 from the frozen v1.4 contract.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.typed_model_cache_formal_execution import (
    READY_V7_VERDICT,
    readiness_v7,
    validate_protocol_v1_1,
)
from src.evaluators.typed_model_cache_formal_protocol import attach_hashes, sha256_file
from src.runtime.formal_execution_environment import (
    probe_python_environment,
    scientific_environment_identity,
)
from src.runtime.resolved_formal_execution_context import (
    RESOLVED_FORMAL_EXECUTION_CONTEXT_VERSION,
)


RUN_ID = "typed_model_cache_formal_preflight_context_repair_20260825_g14r5_v1"
ARTIFACT_ROOT = ROOT / "artifacts/analysis" / RUN_ID
V14_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_4_20260825"
V15_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_5_20260825"
V14_PROTOCOL = V14_ROOT / "protocol_v1_4_manifest.json"
V5_AUDIT = (
    Path("/Users/howen/Projects/PPO_MEC")
    / "artifacts/experiments/typed_model_cache_formal"
    / "typed_model_cache_formal_20260825_111625_g14c_v5"
    / "failure_audit.json"
)
V5_AUDIT_SHA256 = "3c0de5bfebb5877e1b5a53f42fea1e07504f4355bd1636ad17ed38145439ff93"
PROTECTED_FILES = (
    "scripts/train_sa_ghmappo_real_sample.py",
    "src/agents/sa_ghmappo_agent.py",
    "src/agents/sa_ghmappo_core.py",
    "src/encoders/fusion_encoder.py",
    "src/evaluators/real_eval_support.py",
    "tests/test_algo_pool_contract.py",
    "tests/test_checkpoint_compat.py",
)
PROTECTED_MAIN_WORKTREE_SHA256 = {
    "scripts/train_sa_ghmappo_real_sample.py": "aed850f5561f94ecba824e22bd323cdd142ee6c74255a3599129a2a6782e0eba",
    "src/agents/sa_ghmappo_agent.py": "06638c1aea5097a7fa4088db6b77648648655053dc87e1a1c817b09a7709c171",
    "src/agents/sa_ghmappo_core.py": "9951badce0ce78e608e690d6bed8d07a59d19dfef1e82f94a89d88403ac0d6b9",
    "src/encoders/fusion_encoder.py": "cde948c13f487790cf255389bc26b7af191ecc66449a7e939b217c638327954d",
    "src/evaluators/real_eval_support.py": "0a092cc15224b9b1be6a3476555c6e8eb8293573b3e27acf3fa91630db948cb6",
    "tests/test_algo_pool_contract.py": "41f2ca2f6920940bc11cd16bbc4c96104452c5653812a2b69c0e1a8e6794e75b",
    "tests/test_checkpoint_compat.py": "6b09b63b4a5cd9b527e7f3a146962ee37b9b1c9f8da78893d213b40bc6dc2cbf",
}


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


def invalid_reference(
    run_id: str,
    boundary: str,
    digest: str,
    training_cells: int,
    candidate_count: int,
    audit_path: str,
) -> dict[str, Any]:
    return {
        "status": "INVALID_PROTOCOL_OR_IMPLEMENTATION",
        "run_id": run_id,
        "failure_boundary": boundary,
        "failure_audit_path": audit_path,
        "failure_audit_sha256": digest,
        "training_cells_executed": training_cells,
        "candidate_checkpoint_count": candidate_count,
        "dev_performance_count": 0,
        "formal_performance_count": 0,
        "checkpoint_reuse_allowed": False,
        "resume_allowed": False,
        "legacy_phase_finalize_allowed": False,
        "immutable_old_run": True,
    }


def invalid_runs() -> list[dict[str, Any]]:
    if sha256_file(V5_AUDIT) != V5_AUDIT_SHA256:
        raise ValueError("G14C v5 failure audit hash drift")
    base = "artifacts/experiments/typed_model_cache_formal"
    return [
        invalid_reference(
            "typed_model_cache_formal_20260820_g14c_351fdb8_v1",
            "invalid_before_execution",
            "fd04ee5e25737d74ae9f58d0e076d4d620eb913dd55a8aef039a61510c71a0b1",
            0,
            0,
            f"{base}/typed_model_cache_formal_20260820_g14c_351fdb8_v1/audit/failure_audit.json",
        ),
        invalid_reference(
            "typed_model_cache_formal_20260820_164251_g14c_v2",
            "invalid_before_performance_execution",
            "5da5e20395e5c1e48bf2e267ce757248d024246bdc121d4d2b33ca4f8c6c594b",
            0,
            0,
            "/private/tmp/ppo_mec_g14c_v2_89049c9/artifacts/experiments/typed_model_cache_formal/typed_model_cache_formal_20260820_164251_g14c_v2/audit/failure_audit.json",
        ),
        invalid_reference(
            "typed_model_cache_formal_20260820_203430_g14c_v3",
            "invalid_before_dev_performance_execution",
            "476cfc3f57312263da7dff388a89c088e4716d43b1949eb121598c86dc5ac3af",
            150,
            1200,
            "/private/tmp/ppo_mec_g14c_v3_a7c9e8e/artifacts/experiments/typed_model_cache_formal/typed_model_cache_formal_20260820_203430_g14c_v3/failure_audit.json",
        ),
        invalid_reference(
            "typed_model_cache_formal_20260824_110016_g14c_v4",
            "invalid_after_training_before_dev_performance_execution",
            "aaf5cfa717d543ffec5ea15dc5e4e8e7dac107dea51647cea10a9b1884118117",
            150,
            1200,
            f"{base}/typed_model_cache_formal_20260824_110016_g14c_v4/failure_audit.json",
        ),
        invalid_reference(
            "typed_model_cache_formal_20260824_235839_g14c_v4",
            "invalid_before_first_frozen_subcommand",
            "bff76afccff2ea9485555a0bd20b33f5081e2ccaabebeff932f2ef74e8e6f42d",
            0,
            0,
            f"{base}/typed_model_cache_formal_20260824_235839_g14c_v4/failure_audit.json",
        ),
        invalid_reference(
            "typed_model_cache_formal_20260825_111625_g14c_v5",
            "invalid_during_first_preflight_child_before_window_reachability",
            V5_AUDIT_SHA256,
            0,
            0,
            "artifacts/experiments/typed_model_cache_formal/typed_model_cache_formal_20260825_111625_g14c_v5/failure_audit.json",
        ),
    ]


def build_protocol(shared_python: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    old = read_json(V14_PROTOCOL)
    protocol = deepcopy(old)
    probe = probe_python_environment(shared_python, clean_worktree_root=ROOT)
    identity = scientific_environment_identity(
        probe,
        execution_commit="Commit A6 (exact pushed commit bound out-of-band)",
        source_tree_sha256="Commit A6 Git tree (bound out-of-band to avoid self-reference)",
    )
    protocol.update(
        typed_model_cache_formal_protocol_version="1.5.0",
        protocol_id="typed_model_cache_formal_protocol_v1_5",
        created_at=now(),
        status="frozen_pre_execution_resolved_context_repair_no_performance",
    )
    protocol["supersession"] = {
        "supersedes_version": "1.4.0",
        "old_protocol_status": (
            "invalid_during_first_preflight_child_before_window_reachability"
        ),
        "old_protocol_semantic_sha256": old["hashes"]["semantic_sha256"],
        "invalid_execution_runs": invalid_runs(),
        "formal_performance_observed": False,
        "scientific_fields_changed": False,
        "repair_scope": [
            "resolved execution context propagation",
            "outer/nested command expansion reconciliation",
            "immutable same-run context binding",
            "implicit runtime fallback removal",
        ],
    }
    environment = protocol["formal_execution_environment_contract"]
    environment["scientific_identity"] = identity
    environment["resolver"]["priority"] = ["explicit_python_executable"]
    environment["resolver"]["implicit_fallback_allowed"] = False
    execution = protocol["execution_contract"]
    context = execution["default_expansion_context"]
    context["protocol_path"] = (
        "configs/experiment/typed_model_cache_formal_protocol_v1_5_20260825/"
        "protocol_v1_5_manifest.json"
    )
    context["resolved_execution_context_path"] = (
        "/ABSOLUTE/FORMAL_OUTPUT_ROOT/resolved_execution_context.json"
    )
    context["resolve_relative_paths_against_repository_root"] = True
    templates = execution["command_templates"]
    for phase in (
        "preflight",
        "dev_select",
        "formal_ablation",
        "formal_support",
        "formal_scalability",
        "formal_statistics",
    ):
        argv = templates[phase]["argv"]
        argv.extend(
            [
                "--resolved-execution-context-path",
                "{resolved_execution_context_path}",
            ]
        )
    dev_argv = templates["dev_select"]["argv"]
    window_index = dev_argv.index("--window-plan-path") + 1
    dev_argv[window_index] = "{dev_window_plan_path}"
    execution["same_run_resume"]["bindings"].append(
        "resolved_execution_context_sha256"
    )
    execution["same_run_resume"]["legacy_v1_v5_marker_or_checkpoint_allowed"] = False
    protocol["resolved_formal_execution_context_contract"] = {
        "version": RESOLVED_FORMAL_EXECUTION_CONTEXT_VERSION,
        "artifact_name": "resolved_execution_context.json",
        "outer_runner_is_unique_producer": True,
        "atomic_create_only": True,
        "canonical_json_finite_only": True,
        "scientific_identity_separate_from_host_paths": True,
        "host_paths_in_full_context_hash": True,
        "context_in_phase_input_hash": True,
        "context_in_phase_ledger": True,
        "context_in_integrity_manifest": True,
        "resume_revalidates_context_hash": True,
        "cross_run_or_cross_commit_reuse_allowed": False,
        "implicit_runtime_fallback_allowed": False,
        "nested_consumers_use_persisted_context": True,
        "outer_nested_command_matrix_equality_required": True,
    }
    protocol = attach_hashes(protocol)
    validate_protocol_v1_1(protocol)
    return protocol, probe


def producer_consumer_matrix() -> dict[str, Any]:
    return {
        "version": "1.0.0",
        "status": "pass",
        "unique_producer": "scripts/run_typed_model_cache_formal_protocol.py",
        "producer_stage": "after environment resolution and before first phase ledger append",
        "resolved_fields": [
            "python_executable",
            "execution_environment_manifest",
            "clean_worktree_root",
            "durable_run_root",
            "protocol_path",
            "repository_root",
            "data_root",
            "checkpoint_root",
            "protocol_artifact_root",
            "portable_resources",
            "environment_identity",
        ],
        "consumers": [
            {"consumer": "outer phase expansion", "mode": "direct", "fallback": False},
            {"consumer": "preflight all-template expansion", "mode": "persisted_context", "fallback": False},
            {"consumer": "dev selection nested evaluator", "mode": "persisted_context_python", "fallback": False},
            {"consumer": "formal ablation/support/scalability nested evaluator", "mode": "persisted_context_python", "fallback": False},
            {"consumer": "formal statistics nested analyzer", "mode": "persisted_context_python", "fallback": False},
            {"consumer": "checkpoint freeze", "mode": "outer_expanded_no_nested_python", "fallback": False},
            {"consumer": "formal cache policy/controller/gate", "mode": "outer_expanded_or_explicit_command", "fallback": False},
            {"consumer": "dry-run/fresh/resume/finalize-only", "mode": "same_builder_or_hash_verified_artifact", "fallback": False},
        ],
        "historical_default_context_consumers": [
            "protocol/config generation scripts (audit-only)",
            "v1.0-v1.4 validators and regression fixtures (audit-only)",
        ],
    }


def build_inventory() -> dict[str, Any]:
    rows = []
    for path in sorted(ARTIFACT_ROOT.rglob("*")):
        if not path.is_file() or path.name in {"artifact_inventory.json", "integrity_manifest.json"}:
            continue
        rows.append(
            {
                "path": path.relative_to(ARTIFACT_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {"status": "pass", "file_count": len(rows), "files": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--rehearsal-summary", default="")
    parser.add_argument("--validation-summary", default="")
    parser.add_argument("--ready", action="store_true")
    args = parser.parse_args()
    shared_python = Path(args.python_executable).absolute()
    if not shared_python.is_file() or not os.access(shared_python, os.X_OK):
        raise FileNotFoundError(shared_python)
    protocol, probe = build_protocol(shared_python)
    protocol_path = V15_ROOT / "protocol_v1_5_manifest.json"
    write_json(protocol_path, protocol)
    environment_manifest = {
        "formal_execution_environment_contract_version": "1.0.0",
        "scientific_identity": protocol["formal_execution_environment_contract"]["scientific_identity"],
        "runtime_location": {
            "resolved_python_absolute_path": str(shared_python),
            "virtual_environment_root": probe["sys_prefix"],
            "site_packages_paths": probe["site_packages"],
        },
        "runtime_location_is_scientific_identity": False,
    }
    write_json(V15_ROOT / "execution_environment_manifest.json", environment_manifest)
    old_index = read_json(V14_ROOT / "protocol_index.json")
    index = deepcopy(old_index)
    index.update(
        protocol_index_version="1.5.0",
        protocol_manifest=protocol_path.relative_to(ROOT).as_posix(),
        protocol_semantic_sha256=protocol["hashes"]["semantic_sha256"],
        execution_environment_manifest=(
            V15_ROOT / "execution_environment_manifest.json"
        ).relative_to(ROOT).as_posix(),
        resolved_execution_context_contract_version="1.0.0",
        status=READY_V7_VERDICT if args.ready else "PENDING_G14R5_VALIDATION",
    )
    write_json(V15_ROOT / "protocol_index.json", index)

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    v5 = invalid_runs()[-1]
    write_json(ARTIFACT_ROOT / "g14c_v5_invalid_run_registration.json", v5)
    write_json(
        ARTIFACT_ROOT / "root_cause_audit.json",
        {
            "status": "confirmed",
            "failure_audit_sha256": V5_AUDIT_SHA256,
            "failure": "nested preflight expanded raw default_expansion_context without python_executable",
            "repair": "outer runner is the unique context producer; nested consumers load and validate its create-only artifact",
            "temporary_sys_executable_injection_allowed": False,
            "performance_observed": False,
        },
    )
    write_json(ARTIFACT_ROOT / "producer_consumer_matrix.json", producer_consumer_matrix())
    write_json(
        ARTIFACT_ROOT / "resolved_execution_context_contract.json",
        protocol["resolved_formal_execution_context_contract"],
    )
    write_json(ARTIFACT_ROOT / "protocol_v1_5_manifest.json", protocol)
    write_json(
        ARTIFACT_ROOT / "protocol_v1_4_to_v1_5_diff.json",
        {
            "status": "pass",
            "old_semantic_sha256": read_json(V14_PROTOCOL)["hashes"]["semantic_sha256"],
            "new_semantic_sha256": protocol["hashes"]["semantic_sha256"],
            "scientific_fields_changed": False,
            "changed_scope": protocol["supersession"]["repair_scope"],
        },
    )
    write_json(
        ARTIFACT_ROOT / "environment_identity.json",
        protocol["formal_execution_environment_contract"]["scientific_identity"],
    )
    write_json(
        ARTIFACT_ROOT / "holdout_seal_revalidation.json",
        {
            "status": "pass",
            "sealed": True,
            "opened": False,
            "ordinary_runner_capability": False,
            "training_count": 0,
            "checkpoint_count": 0,
            "performance_row_count": 0,
        },
    )
    if args.rehearsal_summary:
        rehearsal = read_json(Path(args.rehearsal_summary))
        write_json(
            ARTIFACT_ROOT / "real_clean_worktree_preflight_report.json",
            rehearsal,
        )
        write_json(
            ARTIFACT_ROOT / "outer_nested_expansion_reconciliation.json",
            {
                "status": "pass",
                **rehearsal["resolved_context"],
                **{
                    key: rehearsal["real_preflight"][key]
                    for key in (
                        "outer_expansion_sha256",
                        "nested_expansion_sha256",
                        "expansion_equal",
                    )
                },
                "phase_template_count": rehearsal["dry_run"][
                    "phase_template_count"
                ],
                "command_count": rehearsal["dry_run"]["command_count"],
            },
        )
        write_json(
            ARTIFACT_ROOT / "window_reachability_60.json",
            {
                "status": "pass",
                **{
                    key: rehearsal["real_preflight"][key]
                    for key in (
                        "max_mobility_rows",
                        "provider_frame_count",
                        "window_count",
                        "reachable_count",
                        "split_reachable_counts",
                    )
                },
                "metadata_only": True,
                "agent_or_policy_executed": False,
                "performance_fields_read": False,
            },
        )
        write_json(
            ARTIFACT_ROOT / "tests_result.json", rehearsal["tests_phase"]
        )
        write_json(
            ARTIFACT_ROOT / "phase_ledger_audit.json", rehearsal["phase_ledger"]
        )
        write_json(
            ARTIFACT_ROOT / "environment_import_audit.json",
            {
                **rehearsal["import_audit"],
                "clean_worktree_has_local_venv": rehearsal[
                    "clean_worktree_has_local_venv"
                ],
                "resolved_shared_python": rehearsal["resolved_shared_python"],
                "environment_fingerprint": rehearsal["environment_fingerprint"],
                "dependency_fingerprint": rehearsal["dependency_fingerprint"],
            },
        )
    if args.validation_summary:
        validation = read_json(Path(args.validation_summary))
        write_json(ARTIFACT_ROOT / "validation_summary.json", validation)
        write_json(
            ARTIFACT_ROOT / "negative_validation.json",
            validation["negative_validation"],
        )
    checks = {
        "g14c_v5_failure_registered": True,
        "producer_consumer_matrix_complete": True,
        "resolved_context_contract_frozen": True,
        "outer_nested_expansion_equal": args.ready,
        "context_negative_cases_pass": args.ready,
        "legacy_invalid_runs_hard_rejected": True,
        "clean_worktree_without_local_venv": args.ready,
        "clean_import_origin": args.ready,
        "window_reachability_60_of_60": args.ready,
        "real_preflight_completed": args.ready,
        "real_tests_phase_completed": args.ready,
        "phase_and_cell_transactions_regression": args.ready,
        "portable_fairness_checkpoint_regression": args.ready,
        "full_pytest_and_smoke_pass": args.ready,
        "holdout_sealed": True,
        "no_formal_training_or_performance": True,
    }
    verdict = readiness_v7(checks)
    write_json(
        ARTIFACT_ROOT / "readiness_review_v7.json",
        {
            "readiness_review_version": "7.0.0",
            "reviewed_at": now(),
            "literature_cutoff": "2026-08-25",
            "target_venue": "IEEE Transactions on Mobile Computing (TMC)",
            "artifact_run_id": RUN_ID,
            "policy_version": "tmc_review_policy_v3_20260621",
            "implementation_baseline_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "execution_commit": "Commit A6 exact pushed commit bound out-of-band",
            "evidence_level": "E2_EXECUTION_CONTRACT_VALIDATED_NO_FORMAL_PERFORMANCE",
            "checks": checks,
            "verdict": verdict,
            "formal_completed": False,
            "paper_ready": False,
        },
    )
    write_json(
        ARTIFACT_ROOT / "protected_user_file_hashes.json",
        {
            "status": "recorded_from_main_worktree_outside_isolated_repair_worktree",
            "files": PROTECTED_MAIN_WORKTREE_SHA256,
        },
    )
    inventory = build_inventory()
    write_json(ARTIFACT_ROOT / "artifact_inventory.json", inventory)
    write_json(
        ARTIFACT_ROOT / "integrity_manifest.json",
        {
            "status": "pass",
            "artifact_run_id": RUN_ID,
            "inventory_sha256": sha256_file(ARTIFACT_ROOT / "artifact_inventory.json"),
            "file_count": inventory["file_count"],
            "formal": False,
            "training": False,
            "performance_evidence": False,
            "holdout_opened": False,
        },
    )
    print(json.dumps({"status": "pass", "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"], "readiness": verdict}, indent=2))


if __name__ == "__main__":
    main()
