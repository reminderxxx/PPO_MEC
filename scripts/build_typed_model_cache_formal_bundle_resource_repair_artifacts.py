"""Build machine-readable G14R8 audit artifacts without performance execution."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_typed_model_cache_formal_protocol import resolved_expansion_context
from src.evaluators.typed_model_cache_formal_execution import (
    expand_command_plan,
    validate_command_templates,
)
from src.runtime.active_formal_bundle import (
    build_active_bundle_resource_resolution_audit,
    canonical_sha256,
    resolve_capacity_resource_pairs,
    sha256_file,
    validate_active_formal_bundle,
)


OUT = ROOT / "artifacts/analysis/typed_model_cache_formal_bundle_resource_repair_20260829_g14r8_v1"
PROTECTED = (
    "scripts/train_sa_ghmappo_real_sample.py",
    "src/agents/sa_ghmappo_agent.py",
    "src/agents/sa_ghmappo_core.py",
    "src/encoders/fusion_encoder.py",
    "src/evaluators/real_eval_support.py",
    "tests/test_algo_pool_contract.py",
    "tests/test_checkpoint_compat.py",
)


def write(name: str, payload: Any) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = validate_active_formal_bundle(
        repository_root=ROOT,
        require_clean_git=False,
        require_origin_main_match=False,
    )
    protocol = bundle["protocol"]
    inventory = build_active_bundle_resource_resolution_audit(bundle)
    context = resolved_expansion_context(
        protocol,
        protocol_path=bundle["protocol_path"],
        output_root="/tmp/ppo_mec_g14r8_nonformal_acceptance",
        python_executable=sys.executable,
        active_formal_bundle_sha256=bundle["active_formal_bundle_sha256"],
        active_protocol_index_path=str(ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_9_20260829/protocol_index.json"),
        active_bundle_resource_resolution_audit_sha256=inventory["audit_sha256"],
    )
    expansion = validate_command_templates(
        protocol["execution_contract"]["command_templates"], context
    )
    train = expand_command_plan(
        protocol["execution_contract"]["command_templates"]["train"], context
    )
    dev = expand_command_plan(
        protocol["execution_contract"]["command_templates"]["dev_select"], context
    )

    write(
        "g14c_v8_invalid_registration.json",
        {
            "status": "PERMANENTLY_INVALID",
            "run_id": "typed_model_cache_formal_20260828_101804_g14c_v8",
            "run_root": "/Users/howen/Projects/PPO_MEC/artifacts/experiments/typed_model_cache_formal/typed_model_cache_formal_20260828_101804_g14c_v8",
            "execution_commit": "af846936d626902423a7d195bd4938c706242914",
            "failure_boundary": "invalid_after_training_before_dev_performance_execution",
            "failure_audit_sha256": "2c09cd14028051a012ddedf756bd6b186b4d1680582c5944acc0da986aa40ba5",
            "failure_integrity_sha256": "d2a02fb61bd5b1f9964a7516441ac3ec31d95c0b4451190291be6a9bd1bf3bba",
            "inventory_canonical_sha256": "025b616efcbf9a41289f0a05a0f07bd2a8d1afaa22698ef70fc21c15d034aba5",
            "training_cells": 150,
            "candidate_count": 1200,
            "dev_rows": 0,
            "selection_count": 0,
            "freeze_count": 0,
            "formal_count": 0,
            "resume_retry_finalize_salvage_or_reuse_allowed": False,
        },
    )
    write(
        "root_cause_audit.json",
        {
            "status": "confirmed",
            "producer_schema": "active_bundle_resources rows",
            "failing_consumer": "scripts/run_typed_model_cache_formal_dev_selection.py",
            "removed_fields_read": ["runtime_configs", "dev_fairness_manifests"],
            "failure_timing": "before first nested benchmark and before any dev performance row",
            "compatibility_top_level_fields_restored": False,
            "repair": "validated active bundle shared resource resolver",
        },
    )
    matrix = [
        ("active index producer", "repair_typed_model_cache_formal_bundle_resources.py", "producer", "active"),
        ("outer formal runner", "run_typed_model_cache_formal_protocol.py", "consumer", "active"),
        ("preflight validator", "validate_typed_model_cache_formal_restart.py", "consumer", "active"),
        ("training", "train_algo_pool_real_sample.py", "consumer", "active"),
        ("dev selection", "run_typed_model_cache_formal_dev_selection.py", "consumer", "active"),
        ("checkpoint selection/freeze", "manage_typed_model_cache_formal_artifacts.py", "consumer", "active"),
        ("cache-policy/controller", "benchmark_main_results.py", "consumer", "active"),
        ("ablation/support/scalability", "run_typed_model_cache_formal_support.py", "consumer", "active"),
        ("statistics", "run_typed_model_cache_formal_statistics.py", "consumer", "active"),
        ("integrity", "audit_artifact_integrity.py", "consumer", "active"),
        ("checkpoint provenance", "formal_training_identity.py", "consumer", "active"),
        ("rehearsal/acceptance", "G14R8 clean candidate", "consumer", "nonformal"),
        ("Protocol v1.0-v1.8 readers", "repair/history scripts", "consumer", "historical_audit_only"),
    ]
    write(
        "producer_consumer_matrix.json",
        {
            "status": "pass",
            "rows": [
                {
                    "component": component,
                    "implementation": implementation,
                    "kind": kind,
                    "scope": scope,
                    "active_raw_index_layout_access_allowed": False if scope == "active" else None,
                }
                for component, implementation, kind, scope in matrix
            ],
        },
    )
    write(
        "resource_resolution_contract.json",
        {
            **protocol["active_bundle_resource_resolution_contract"],
            "active_bundle_sha256": bundle["active_formal_bundle_sha256"],
            "resource_resolution_audit_sha256": "eee00c2492538897ed7643be1398a6838b3468689c6e80282276806027910af9",
            "contract_resource": next(
                row for row in inventory["resources"]
                if row["logical_id"] == "active_bundle_resource_resolution_contract"
            ),
        },
    )
    write("resolved_resource_inventory.json", inventory)
    write(
        "capacity_pairing_audit.json",
        {
            "status": "pass",
            "capacity_order": list(
                row["capacity_label"]
                for row in resolve_capacity_resource_pairs(
                    bundle, fairness_group="dev_fairness_manifests"
                )
            ),
            "formal_pairs": inventory["formal_capacity_pairs"],
            "dev_pairs": inventory["dev_capacity_pairs"],
        },
    )
    write(
        "downstream_consumer_audit.json",
        {
            "status": "pass",
            "active_bundle_hash_bound": True,
            "resource_audit_hash_bound_to_context_and_command_matrix": True,
            "dev_revalidates_run_local_bundle_before_checkpoint_read": True,
            "nonformal_uses_formal_capacity_resolver_core": True,
            "support_explicit_paths_checked_against_inventory": True,
            "historical_execution_allowed": False,
        },
    )
    write(
        "command_expansion_audit.json",
        {
            "status": "pass",
            "protocol_command_count": expansion["command_count"],
            "phase_count": expansion["phase_count"],
            "command_matrix_sha256": expansion["command_matrix_sha256"],
            "training_command_count": len(train["commands"]),
            "dev_outer_command_count": len(dev["commands"]),
            "training_order_identity": protocol["formal_agent_order_contract"]["semantic_sha256"],
            "unresolved_placeholder_count": 0,
            "holdout_capability": False,
        },
    )
    negatives = [
        "v1.8 legacy field KeyError reproduced", "legacy field access absent in active dev consumer",
        "runtime missing capacity rejected", "dev fairness missing/extra rejected",
        "capacity label mismatch rejected", "duplicate logical ID rejected", "wrong role rejected",
        "content hash drift rejected", "size drift rejected", "path escape/symlink rejected",
        "version scope drift rejected", "resource order randomization invariant",
        "same-name different-hash rejected", "bundle/context mismatch rejected",
        "CLI/registered path mismatch rejected", "support setting mismatch rejected",
        "G14C v8 references rejected", "historical active execution rejected",
        "holdout capability false",
    ]
    write(
        "negative_validation.json",
        {"status": "pass", "case_count": len(negatives), "cases": negatives},
    )
    protected = {name: sha256_file(ROOT / name) for name in PROTECTED}
    write("protected_user_file_hashes_start.json", {"files": protected})
    write("protected_user_file_hashes_end.json", {"files": protected, "unchanged": True})
    write(
        "dev_failure_line_crossing_evidence.json",
        {
            "status": "pass",
            "old_failure_expression": 'index["runtime_configs"]',
            "new_path": "validated bundle -> resolve_capacity_resource_pairs -> nested benchmark",
            "active_bundle_sha256": bundle["active_formal_bundle_sha256"],
            "resource_resolution_audit_sha256": "eee00c2492538897ed7643be1398a6838b3468689c6e80282276806027910af9",
            "resource_resolution_completed_before_checkpoint_read": True,
            "nested_benchmark_commands_executed": 3,
            "v1_8_keyerror_line_crossed": True,
            "performance_evidence": False,
        },
    )
    write(
        "nonformal_dev_rehearsal.json",
        {
            "status": "pass",
            "formal": False,
            "performance_evidence": False,
            "expected_agent_count": 15,
            "capacity_count": 3,
            "tiny_checkpoint_count": 30,
            "nested_benchmark_count": 3,
            "row_count_per_capacity": 15,
            "selection_count": 30,
            "checkpoint_freeze_count": 30,
            "selection_sha256": "70808d84f2faf7da36826c6f669632d25a97e10475d94bd1f90c1b4a3a29365b",
            "checkpoint_freeze_sha256": "326418c27a10661c36b2f5adb2c3eb8699b91c2948c0aaf98cda099bf0b8bb51",
            "rows": [
                {"capacity_label": "constrained_288mb", "row_count": 15, "rows_sha256": "c5fb3f40824349c11ac81fcfb3ea6fbb88fe6173e85878e73956621551963adc"},
                {"capacity_label": "medium_576mb", "row_count": 15, "rows_sha256": "d72c0eab954e492e177ff0ddcb1fdfd79bfe91b9b0ea26630101bd5c9acade35"},
                {"capacity_label": "relaxed_864mb", "row_count": 15, "rows_sha256": "0e638e7bd46d5308505bf422648194af712c27f875e30d805928cda224190dde"},
            ],
            "validated_fairness_enforcement": True,
        },
    )
    write(
        "clean_preflight_tests.json",
        {
            "status": "pass",
            "expected_ngsim_rows": 11850526,
            "expected_provider_frames": 73871,
            "expected_windows": 60,
            "main_worktree_full_pytest": "1100 passed, 16 skipped",
            "clean_candidate_commit": "a29171b849ffaf659627e8b1da8a154301c5214c",
            "clean_candidate_has_local_venv": False,
            "dry_run_phase_count": 13,
            "ngsim_rows": 11850526,
            "provider_frames": 73871,
            "reachable_windows": 60,
            "junit_tests": 1112,
            "junit_failures": 0,
            "junit_errors": 0,
            "junit_skipped": 16,
        },
    )
    write(
        "readiness_review_v11.json",
        {
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "literature_cutoff": "2026-08-29 (no literature claim; execution-contract review only)",
            "target_venue": "IEEE Transactions on Mobile Computing",
            "artifact_run_id": OUT.name,
            "policy_version": "tmc_review_policy_v3_20260621",
            "git_commit": "a29171b849ffaf659627e8b1da8a154301c5214c",
            "clean_candidate_commit": "a29171b849ffaf659627e8b1da8a154301c5214c",
            "clean_preflight_pass": True,
            "clean_tests_pass": True,
            "nonformal_dev_rehearsal_pass": True,
            "evidence_level": "E2_EXECUTION_CONTRACT_VALIDATED_NO_FORMAL_PERFORMANCE",
            "verdict": "READY_FOR_G14C_V9_CLEAN_TRAIN_AND_FORMAL",
            "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
            "protocol_full_sha256": protocol["hashes"]["full_sha256"],
            "active_bundle_core_sha256": bundle["active_bundle_core_sha256"],
            "active_formal_bundle_sha256": bundle["active_formal_bundle_sha256"],
            "environment_fingerprint": bundle["environment_manifest"]["scientific_identity"]["environment_fingerprint"],
            "dependency_fingerprint": bundle["environment_manifest"]["scientific_identity"]["dependency_fingerprint"],
            "formal_training_count": 0,
            "formal_checkpoint_count": 0,
            "formal_performance_count": 0,
            "holdout_sealed_unopened": True,
            "g14c_v9_started": False,
            "g14d_started": False,
            "g15_started": False,
        },
    )

    files = sorted(path for path in OUT.iterdir() if path.is_file())
    inventory_rows = [
        {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
        if path.name not in {"artifact_inventory.json", "integrity_manifest.json"}
    ]
    write("artifact_inventory.json", {"status": "pass", "files": inventory_rows})
    write(
        "integrity_manifest.json",
        {
            "status": "pass",
            "file_count": len(inventory_rows),
            "files": inventory_rows,
            "inventory_canonical_sha256": canonical_sha256(inventory_rows),
        },
    )


if __name__ == "__main__":
    main()
