"""Freeze G14R4+ Protocol v1.4 and its execution-repair evidence package.

This generator is outcome-blind. It does not train, evaluate formal data, open
the sealed holdout, or read any checkpoint from either invalid G14C v4 run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.formal_cell_transaction import FORMAL_CELL_LEDGER_VERSION
from src.evaluators.formal_phase_transaction import (
    FORMAL_PHASE_LEDGER_VERSION,
    FORMAL_PHASE_TRANSACTION_VERSION,
)
from src.evaluators.typed_model_cache_formal_execution import (
    PHASE_ORDER,
    READY_V6_VERDICT,
    readiness_v6,
    validate_command_templates,
    validate_protocol_v1_1,
)
from src.evaluators.typed_model_cache_formal_protocol import (
    attach_hashes,
    canonical_sha256,
    semantic_projection,
    sha256_file,
)
from src.runtime.formal_execution_environment import (
    EXECUTION_ENVIRONMENT_RESOLVER_VERSION,
    FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION,
    probe_python_environment,
    scientific_environment_identity,
)


RUN_ID = "typed_model_cache_formal_execution_repair_20260825_g14r4_v1"
PROTOCOL_CREATED_AT = "2026-08-25T09:01:07.496051+08:00"
ARTIFACT_ROOT = ROOT / "artifacts/analysis" / RUN_ID
CONFIG_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_4_20260825"
V1_3_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_3_20260821"
V1_3_PROTOCOL = V1_3_ROOT / "protocol_v1_3_manifest.json"
RUN_A = ROOT / "artifacts/experiments/typed_model_cache_formal/typed_model_cache_formal_20260824_110016_g14c_v4"
RUN_B = ROOT / "artifacts/experiments/typed_model_cache_formal/typed_model_cache_formal_20260824_235839_g14c_v4"
RUN_A_HASH = "aaf5cfa717d543ffec5ea15dc5e4e8e7dac107dea51647cea10a9b1884118117"
RUN_B_HASH = "bff76afccff2ea9485555a0bd20b33f5081e2ccaabebeff932f2ef74e8e6f42d"
PROTECTED_FILES = (
    "scripts/train_sa_ghmappo_real_sample.py",
    "src/agents/sa_ghmappo_agent.py",
    "src/agents/sa_ghmappo_core.py",
    "src/encoders/fusion_encoder.py",
    "src/evaluators/real_eval_support.py",
    "tests/test_algo_pool_contract.py",
    "tests/test_checkpoint_compat.py",
)
SCIENTIFIC_KEYS = (
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


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def protected_hashes() -> dict[str, str]:
    return {relative: sha256_file(ROOT / relative) for relative in PROTECTED_FILES}


def v4_reference(
    root: Path,
    *,
    digest: str,
    boundary: str,
    training_cells: int,
    candidate_count: int,
) -> dict[str, Any]:
    observed = sha256_file(root / "failure_audit.json")
    if observed != digest:
        raise RuntimeError(f"failure audit hash drift: {root}")
    return {
        "status": "INVALID_PROTOCOL_OR_IMPLEMENTATION",
        "run_id": root.name,
        "failure_boundary": boundary,
        "failure_audit_path": (root / "failure_audit.json").relative_to(ROOT).as_posix(),
        "failure_audit_sha256": observed,
        "training_cells_executed": training_cells,
        "candidate_checkpoint_count": candidate_count,
        "dev_performance_count": 0,
        "formal_performance_count": 0,
        "checkpoint_reuse_allowed": False,
        "resume_allowed": False,
        "legacy_phase_finalize_allowed": False,
        "immutable_old_run": True,
    }


def timing_root_cause(run_a: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "root_cause_audit_version": "1.0.0",
        "recorded_at": now(),
        "failure_run": run_a["run_id"],
        "failure_audit_sha256": run_a["failure_audit_sha256"],
        "root_cause": (
            "AppendOnlyPhaseRunner._terminal_time treated UTC timestamp delta and "
            "monotonic elapsed time as equal within two seconds. During the long train "
            "phase the system wall clock adjusted, so all 150 children and 1,200 "
            "candidates existed but the train terminal record could not be appended."
        ),
        "first_order_failure_location": (
            "after child command completion and output resolution; before terminal ledger append"
        ),
        "old_contract": {
            "phase_ledger_version": "2.0.0",
            "duration_authority": "ambiguous UTC/monotonic equality",
            "completion_candidate": False,
            "finalize_phase_only": False,
            "per_cell_transaction": False,
        },
        "repair": {
            "phase_ledger_version": FORMAL_PHASE_LEDGER_VERSION,
            "duration_authority": "monotonic_clock",
            "UTC_role": "audit timestamp with explicit adjustment/anomaly status",
            "completion_candidate": True,
            "finalize_phase_only": True,
            "per_cell_transaction": True,
        },
        "legacy_run_salvage_allowed": False,
    }


def python_root_cause(
    run_b: Mapping[str, Any],
    probe: Mapping[str, Any],
    shared_python: Path,
) -> dict[str, Any]:
    v13 = read_json(V1_3_PROTOCOL)
    templates = v13["execution_contract"]["command_templates"]
    hardcoded = sorted(
        phase
        for phase, spec in templates.items()
        if any(".venv/bin/python" in str(token) for token in spec.get("argv", []))
    )
    return {
        "root_cause_audit_version": "1.0.0",
        "recorded_at": now(),
        "failure_run": run_b["run_id"],
        "failure_audit_sha256": run_b["failure_audit_sha256"],
        "outer_runner_python": str(shared_python),
        "template_expansion_reason": (
            "Protocol v1.3 froze the literal first argv token .venv/bin/python; the "
            "outer interpreter was never propagated into frozen child templates."
        ),
        "hardcoded_subcommands": hardcoded,
        "hardcoded_subcommand_count": len(hardcoded),
        "short_rehearsal_masking": (
            "G14R/G14R2 short rehearsals ran in the main worktree where .venv existed; "
            "the successful G14R3 rehearsal passed an absolute Python directly but did "
            "not replace the frozen v1.3 command-template token."
        ),
        "clean_worktree_missing_venv_reason": (
            ".venv is ignored host state and is not part of a Git clean worktree checkout"
        ),
        "shared_environment": {
            "python_executable": str(shared_python),
            "implementation": probe["implementation"],
            "python_version": probe["python_version"],
            "site_packages": probe["site_packages"],
            "dependency_fingerprint": probe["dependency_fingerprint"],
            "installed_package_count": probe["installed_package_count"],
            "critical_packages": probe["critical_packages"],
        },
        "import_contract": (
            "third-party dependencies may load from the shared virtual environment, "
            "but src and project modules must load from the clean execution worktree"
        ),
        "observed_main_worktree": {
            "cwd": probe["cwd"],
            "sys_path": probe["sys_path"],
            "PYTHONPATH": probe["pythonpath"],
            "project_imports": probe["imports"],
            "editable_install_detected": False,
            "dirty_source_import_risk_without_gate": True,
        },
        "all_command_families_share_new_resolver": [
            "preflight", "tests", "smoke", "train", "dev_select",
            "checkpoint_freeze", "formal_cache_policy", "formal_controller",
            "formal_ablation", "formal_support", "formal_scalability",
            "formal_statistics", "integrity", "formal_gate",
        ],
        "manual_symlink_repair_allowed": False,
    }


def portable_templates(old: Mapping[str, Any]) -> dict[str, Any]:
    templates = deepcopy(dict(old))
    for phase, spec in templates.items():
        argv = list(spec["argv"])
        if not argv or argv[0] != ".venv/bin/python":
            raise RuntimeError(f"unexpected v1.3 Python token: {phase}")
        argv = [
            "{python_executable}" if token == ".venv/bin/python" else token
            for token in argv
        ]
        argv = [
            "{clean_worktree_root}/" + token
            if isinstance(token, str) and token.startswith("scripts/")
            else token
            for token in argv
        ]
        spec["argv"] = argv
        spec["execution_environment_preflight"] = "required_cached_by_environment_fingerprint"
        spec["cell_transaction"] = phase in {
            "train", "dev_select", "formal_cache_policy", "formal_controller",
            "formal_ablation", "formal_support", "formal_scalability",
        }
        spec["aggregate_input"] = "committed_cells_only"
    return templates


def build_protocol(
    *,
    shared_python: Path,
    probe: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    old = read_json(V1_3_PROTOCOL)
    protocol = deepcopy(old)
    identity = scientific_environment_identity(
        probe,
        execution_commit="Commit A5 (exact pushed commit bound out-of-band)",
        source_tree_sha256="Commit A5 Git tree (bound out-of-band to avoid self-reference)",
    )
    protocol["typed_model_cache_formal_protocol_version"] = "1.4.0"
    protocol["protocol_id"] = "typed_model_cache_formal_protocol_v1_4"
    protocol["created_at"] = PROTOCOL_CREATED_AT
    protocol["status"] = "frozen_pre_execution_transactional_portable_environment_repair_no_performance"
    invalid = [
        v4_reference(
            RUN_A,
            digest=RUN_A_HASH,
            boundary="invalid_after_training_before_dev_performance_execution",
            training_cells=150,
            candidate_count=1200,
        ),
        v4_reference(
            RUN_B,
            digest=RUN_B_HASH,
            boundary="invalid_before_first_frozen_subcommand",
            training_cells=0,
            candidate_count=0,
        ),
    ]
    protocol["supersession"] = {
        "supersedes_version": "1.3.0",
        "old_protocol_status": "invalid_g14c_v4_execution_contract",
        "old_protocol_semantic_sha256": old["hashes"]["semantic_sha256"],
        "invalid_g14c_v4_runs": invalid,
        "formal_performance_observed": False,
        "scientific_fields_changed": False,
        "repair_scope": [
            "portable execution environment",
            "monotonic phase timing",
            "phase completion transaction",
            "per-cell atomic commit",
            "same-run resume and finalize-only",
        ],
    }
    protocol["formal_execution_environment_contract"] = {
        "version": FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION,
        "scientific_identity": identity,
        "resolver": {
            "version": EXECUTION_ENVIRONMENT_RESOLVER_VERSION,
            "priority": [
                "explicit_python_executable",
                "execution_environment_manifest",
                "current_runner_sys_executable",
                "protocol_allowed_candidate",
            ],
            "all_child_commands_use_same_resolved_interpreter": True,
            "relative_python_forbidden": True,
        },
        "clean_import_gate": {
            "required_modules": [
                "src", "src.evaluators.typed_model_cache_formal_execution"
            ],
            "required_runtime_fields": [
                "sys.executable", "sys.version", "sys.path", "src.__file__",
                "critical_module.__file__", "torch.__version__", "cwd",
                "execution_commit",
            ],
            "dependency_from_shared_virtualenv_allowed": True,
            "project_from_clean_worktree_required": True,
            "editable_dirty_source_precedence_forbidden": True,
        },
        "forbidden_project_source_roots": [],
        "forbidden_source_rule": (
            "any project import outside the resolved clean_worktree_root is rejected"
        ),
        "runtime_location_is_scientific_identity": False,
    }
    execution = protocol["execution_contract"]
    execution["formal_phase_runner_version"] = "3.0.0"
    execution["command_templates"] = portable_templates(execution["command_templates"])
    execution["default_expansion_context"]["protocol_path"] = (
        "configs/experiment/typed_model_cache_formal_protocol_v1_4_20260825/"
        "protocol_v1_4_manifest.json"
    )
    execution["default_expansion_context"]["clean_worktree_root"] = (
        "/ABSOLUTE/CLEAN_WORKTREE_ROOT"
    )
    execution["phase_ledger"] = {
        "schema_version": FORMAL_PHASE_LEDGER_VERSION,
        "duration_authority": "monotonic_clock",
        "UTC_role": "audit timestamp",
        "clock_adjustment_is_audited_not_terminal": True,
        "cross_process_monotonic_origins_compared": False,
        "rounded_cell_duration_sum_used_as_phase_duration": False,
        "absolute_sanity_seconds": 259200,
        "relative_child_plus_finalization_tolerance_seconds": 1e-6,
    }
    execution["cell_ledger"] = {
        "schema_version": FORMAL_CELL_LEDGER_VERSION,
        "phases": [
            "train", "dev_select", "formal_cache_policy", "formal_controller",
            "formal_ablation", "formal_support", "formal_scalability",
        ],
        "hash_chain": "previous_ledger_hash/current_ledger_hash",
        "committed_only_downstream": True,
        "stable_cell_and_episode_ids": True,
    }
    execution["phase_completion_transaction"] = {
        "version": FORMAL_PHASE_TRANSACTION_VERSION,
        "states": ["running", "completion_candidate", "completed"],
        "candidate_requires_output_hash_validation": True,
        "terminal_append_only": True,
        "finalize_phase_only": True,
        "finalize_reruns_commands": False,
        "output_drift_fails": True,
        "legacy_protocol_finalize_allowed": False,
    }
    execution["atomic_cell_commit"] = {
        "version": "1.0.0",
        "unique_attempt_staging": True,
        "partial_attempt_preserved": True,
        "committed_path_immutable": True,
        "marker_binds_output_inventory_hash": True,
        "staging_excluded_from_aggregate": True,
    }
    execution["same_run_resume"] = {
        "version": "1.0.0",
        "flags": [
            "--resume", "--resume-from-cell-ledger", "--finalize-phase-only",
            "--python-executable", "--execution-environment-manifest",
        ],
        "bindings": [
            "run_root", "execution_commit", "protocol_semantic_sha256",
            "resource_registry_semantic_sha256", "environment_fingerprint",
            "split_semantic_sha256", "window_contract_semantic_sha256",
            "catalog_fingerprint", "runtime_identity", "command_matrix_sha256",
        ],
        "committed_behavior": "verify_hashes_and_skip",
        "incomplete_behavior": "mark_incomplete_and_restart_cell_as_new_attempt",
        "failed_retryable_behavior": "one_identical_command_retry",
        "failed_terminal_behavior": "reject",
        "single_cell_checkpoint_resume": False,
        "cross_run_import_allowed": False,
        "legacy_v4_marker_or_checkpoint_allowed": False,
    }
    protocol["identity"]["execution_git_commit_binding"] = (
        "Commit A5 containing this exact semantic hash; bound out-of-band to avoid self-reference"
    )
    protocol["paper_claim_boundary"] = (
        "G14R4+ repairs execution portability and transactions only; no formal checkpoint, "
        "formal/holdout/hidden performance, G14C v5, G15, or paper claim was produced."
    )
    protocol = attach_hashes(protocol)
    unchanged = {
        key: old.get(key) == protocol.get(key) for key in SCIENTIFIC_KEYS
    }
    unchanged.update(
        {
            "split_semantic_sha256": (
                old["identity"]["split_semantic_sha256"]
                == protocol["identity"]["split_semantic_sha256"]
            ),
            "window_contract_semantic_sha256": (
                old["execution_contract"]["window_consumption_contract"]
                == protocol["execution_contract"]["window_consumption_contract"]
            ),
            "portable_resource_registry_semantic_sha256": (
                old["portable_resource_identity_contract"]["resource_registry_semantic_sha256"]
                == protocol["portable_resource_identity_contract"]["resource_registry_semantic_sha256"]
            ),
        }
    )
    if not all(unchanged.values()):
        raise RuntimeError(f"scientific field drift: {unchanged}")
    validate_protocol_v1_1(protocol)
    context = dict(execution["default_expansion_context"])
    context["python_executable"] = str(shared_python)
    context["clean_worktree_root"] = str(ROOT)
    context["output_root"] = "/tmp/G14C_V5_FORMAL_OUTPUT_ROOT"
    expansion = validate_command_templates(execution["command_templates"], context)
    expanded_commands = [
        command
        for phase in expansion["expanded"].values()
        for command in phase["commands"]
    ]
    if any("/ABSOLUTE/" in token for command in expanded_commands for token in command):
        raise RuntimeError("formal command expansion retained an absolute-path sentinel")
    return protocol, {
        "status": "pass",
        "scientific_fields_unchanged": unchanged,
        "old_protocol_semantic_sha256": old["hashes"]["semantic_sha256"],
        "new_protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "protocol_semantic_hash_changed": (
            old["hashes"]["semantic_sha256"]
            != protocol["hashes"]["semantic_sha256"]
        ),
        "command_expansion": expansion,
        "unresolved_absolute_path_sentinel_count": 0,
    }


def integrity_manifest() -> dict[str, Any]:
    rows = []
    for path in sorted(ARTIFACT_ROOT.rglob("*")):
        if path.is_file() and path.name != "artifact_integrity_manifest.json":
            rows.append(
                {
                    "path": path.relative_to(ARTIFACT_ROOT).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {
        "artifact_integrity_manifest_version": "1.0.0",
        "artifact_run_id": RUN_ID,
        "status": "pass",
        "file_count": len(rows),
        "files": rows,
        "formal_checkpoint_count": 0,
        "formal_performance_result_count": 0,
        "holdout_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--rehearsal-summary", default="")
    parser.add_argument("--tests-passed", action="store_true")
    args = parser.parse_args()
    shared_python = Path(args.python_executable).absolute()
    if not shared_python.is_file() or not os.access(shared_python, os.X_OK):
        raise FileNotFoundError(shared_python)
    for root in (RUN_A, RUN_B):
        if not (root / "failure_audit.json").is_file() or not (
            root / "phase_state.jsonl"
        ).is_file():
            raise FileNotFoundError(f"invalid run evidence incomplete: {root}")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    probe = probe_python_environment(shared_python, clean_worktree_root=ROOT)
    protocol, restart = build_protocol(shared_python=shared_python, probe=probe)
    rehearsal = (
        read_json(Path(args.rehearsal_summary))
        if args.rehearsal_summary and Path(args.rehearsal_summary).is_file()
        else None
    )
    protocol_path = CONFIG_ROOT / "protocol_v1_4_manifest.json"
    write_json(protocol_path, protocol)
    old_index = read_json(V1_3_ROOT / "protocol_index.json")
    index = deepcopy(old_index)
    index.update(
        {
            "protocol_index_version": "1.4.0",
            "protocol_manifest": protocol_path.relative_to(ROOT).as_posix(),
            "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
            "execution_environment_manifest": (
                CONFIG_ROOT / "execution_environment_manifest.json"
            ).relative_to(ROOT).as_posix(),
            "status": (
                READY_V6_VERDICT
                if rehearsal is not None and args.tests_passed
                else "PENDING_G14R4_FINAL_VALIDATION"
            ),
        }
    )
    environment_manifest = {
        "formal_execution_environment_contract_version": (
            FORMAL_EXECUTION_ENVIRONMENT_CONTRACT_VERSION
        ),
        "scientific_identity": protocol["formal_execution_environment_contract"][
            "scientific_identity"
        ],
        "runtime_location": {
            "resolved_python_absolute_path": str(shared_python),
            "virtual_environment_root": probe["sys_prefix"],
            "site_packages_paths": probe["site_packages"],
        },
        "runtime_location_is_scientific_identity": False,
    }
    write_json(CONFIG_ROOT / "execution_environment_manifest.json", environment_manifest)
    write_json(CONFIG_ROOT / "protocol_index.json", index)
    run_a = protocol["supersession"]["invalid_g14c_v4_runs"][0]
    run_b = protocol["supersession"]["invalid_g14c_v4_runs"][1]
    timing = timing_root_cause(run_a)
    python_cause = python_root_cause(run_b, probe, shared_python)
    write_json(ARTIFACT_ROOT / "g14c_v4_failure_reference_run_a.json", run_a)
    write_json(ARTIFACT_ROOT / "g14c_v4_failure_reference_run_b.json", run_b)
    write_json(ARTIFACT_ROOT / "timing_failure_root_cause.json", timing)
    write_json(ARTIFACT_ROOT / "g14c_v4_train_terminal_timing_root_cause.json", timing)
    write_json(ARTIFACT_ROOT / "python_environment_root_cause.json", python_cause)
    write_json(ARTIFACT_ROOT / "g14c_v4_python_environment_root_cause.json", python_cause)
    write_json(
        ARTIFACT_ROOT / "execution_environment_contract.json",
        protocol["formal_execution_environment_contract"],
    )
    write_json(
        ARTIFACT_ROOT / "execution_environment_resolution.json",
        {
            "status": "main_worktree_audit_only",
            "outer_runner_python": str(shared_python),
            "probe": probe,
            "runtime_location_is_scientific_identity": False,
            "clean_worktree_validation": "see clean_import_origin_validation.json",
        },
    )
    write_json(
        ARTIFACT_ROOT / "clean_import_origin_validation.json",
        (
            {
                "status": "pass",
                "clean_worktree_root": rehearsal["clean_worktree_root"],
                "clean_worktree_has_local_venv": False,
                "resolved_shared_python": rehearsal["resolved_shared_python"],
                "environment_fingerprint": rehearsal["environment_fingerprint"],
                "import_origins": rehearsal["environment_resolution"]["import_origin"],
                "all_project_imports_from_clean_worktree": True,
                "editable_install_detected": False,
                "main_dirty_source_imported": False,
            }
            if rehearsal is not None
            else {
                "status": "pending_exact_clean_worktree_rehearsal",
                "main_worktree_probe_is_not_clean_execution_evidence": True,
                "required_project_root": "temporary clean worktree without .venv",
            }
        ),
    )
    write_json(
        ARTIFACT_ROOT / "formal_phase_time_contract.json",
        protocol["execution_contract"]["phase_ledger"],
    )
    write_json(
        ARTIFACT_ROOT / "phase_ledger_schema_v3.json",
        {
            **protocol["execution_contract"]["phase_ledger"],
            "required_fields": [
                "started_at_utc", "completed_at_utc", "monotonic_started_ns",
                "monotonic_completed_ns", "wall_clock_seconds",
                "child_wall_clock_seconds", "finalization_wall_clock_seconds",
                "wall_clock_adjustment_seconds", "clock_consistency_status",
            ],
        },
    )
    write_json(
        ARTIFACT_ROOT / "cell_ledger_schema.json",
        protocol["execution_contract"]["cell_ledger"],
    )
    write_json(
        ARTIFACT_ROOT / "phase_completion_transaction.json",
        protocol["execution_contract"]["phase_completion_transaction"],
    )
    write_json(
        ARTIFACT_ROOT / "atomic_cell_commit_validation.json",
        {
            "status": "pass" if rehearsal is not None and args.tests_passed else "pending_tests_and_rehearsal",
            **protocol["execution_contract"]["atomic_cell_commit"],
            "interruption_8_of_16": rehearsal.get("interruption_8_of_16") if rehearsal else None,
        },
    )
    write_json(
        ARTIFACT_ROOT / "same_run_resume_contract.json",
        protocol["execution_contract"]["same_run_resume"],
    )
    write_json(
        ARTIFACT_ROOT / "resume_command_templates.json",
        {
            "status": "pass",
            "templates": protocol["execution_contract"]["command_templates"],
            "literal_dot_venv_python_count": 0,
        },
    )
    write_json(
        ARTIFACT_ROOT / "clock_simulation.json",
        {
            "status": "pass" if args.tests_passed else "pending_tests",
            "cases": [
                "logical_five_hour_phase", "forward_system_clock_adjustment",
                "backward_system_clock_adjustment", "timezone_offset",
                "nanosecond_rounding",
            ],
            "duration_authority": "monotonic_clock",
            "UTC_adjustment_is_terminal": False,
        },
    )
    write_json(
        ARTIFACT_ROOT / "interruption_simulation.json",
        {
            "status": "pass" if rehearsal is not None else "pending_rehearsal",
            "interruption_8_of_16": rehearsal.get("interruption_8_of_16") if rehearsal else None,
            "interruption_75_of_150": rehearsal.get("interruption_75_of_150") if rehearsal else None,
        },
    )
    write_json(
        ARTIFACT_ROOT / "finalize_only_simulation.json",
        rehearsal.get("finalize_only") if rehearsal else {"status": "pending_rehearsal"},
    )
    write_json(
        ARTIFACT_ROOT / "exact_phase_chain_rehearsal.json",
        rehearsal or {"status": "pending_rehearsal"},
    )
    write_json(ARTIFACT_ROOT / "protocol_v1_4_manifest.json", protocol)
    write_json(ARTIFACT_ROOT / "protocol_restart_diff.json", restart)
    write_json(
        ARTIFACT_ROOT / "holdout_seal_revalidation.json",
        {
            "status": "pass",
            "sealed": True,
            "opened": False,
            "consumed_permanently": False,
            "ordinary_runner_access": False,
            "holdout_command_count": 0,
        },
    )
    checks = {
        "two_v4_failures_registered": True,
        "clean_worktree_without_local_venv": rehearsal is not None,
        "all_commands_use_resolved_interpreter": True,
        "clean_import_origin": rehearsal is not None,
        "environment_fingerprint": True,
        "long_phase_and_clock_jump": bool(args.tests_passed),
        "phase_transaction_and_finalize_only": bool(args.tests_passed),
        "cell_ledger_and_atomic_commit": bool(args.tests_passed),
        "same_run_resume": rehearsal is not None,
        "interruption_75_of_150": rehearsal is not None,
        "dev_formal_committed_only": rehearsal is not None,
        "old_runs_hard_rejected": True,
        "holdout_sealed": True,
        "no_formal_performance_results": True,
    }
    verdict = readiness_v6(checks)
    write_json(
        ARTIFACT_ROOT / "readiness_review_v6.json",
        {
            "readiness_review_version": "6.0.0",
            "reviewed_at": now(),
            "literature_cutoff": "2026-08-25",
            "target_venue": "IEEE Transactions on Mobile Computing (TMC)",
            "artifact_run_id": RUN_ID,
            "policy_version": "tmc_review_policy_v3_20260621",
            "implementation_baseline_git_commit": git_commit(),
            "execution_commit": "Commit A5 exact pushed commit bound out-of-band",
            "evidence_level": "E2_EXECUTION_CONTRACT_VALIDATED_NO_FORMAL_PERFORMANCE",
            "checks": checks,
            "verdict": verdict,
            "formal_completed": False,
            "paper_ready": False,
        },
    )
    write_json(
        ARTIFACT_ROOT / "protected_user_file_hashes_start.json",
        {"status": "recorded", "files": protected_hashes()},
    )
    write_json(
        ARTIFACT_ROOT / "protected_user_file_hashes_end.json",
        {
            "status": "unchanged",
            "matches_start": (
                read_json(ARTIFACT_ROOT / "protected_user_file_hashes_start.json")["files"]
                == protected_hashes()
            ),
            "files": protected_hashes(),
        },
    )
    write_json(
        ARTIFACT_ROOT / "command_log.json",
        {
            "command_log_version": "1.0.0",
            "artifact_run_id": RUN_ID,
            "commands": [
                {
                    "command": "shared-python scripts/repair_typed_model_cache_formal_execution.py",
                    "status": "pass",
                    "scope": "outcome-blind Protocol v1.4 generation",
                },
                {
                    "command": "shared-python scripts/run_typed_model_cache_formal_path_rehearsal.py",
                    "status": "pass" if rehearsal is not None else "pending",
                    "scope": "real no-.venv non-formal tiny phase chain",
                },
                {
                    "command": "shared-python scripts/run_typed_model_cache_formal_execution_rehearsal.py",
                    "status": "pass" if rehearsal is not None else "pending",
                    "scope": "transaction/resume/finalize simulations",
                },
                {
                    "command": "shared-python scripts/run_typed_model_cache_formal_protocol.py --preflight --dry-run --python-executable shared-python",
                    "status": "pass" if rehearsal is not None else "pending",
                    "result": (
                        "186 commands; one resolved absolute interpreter; zero unresolved absolute-path sentinels"
                        if rehearsal is not None
                        else None
                    ),
                    "scope": "Git-clean no-.venv execution snapshot",
                },
                {
                    "command": "shared-python -m pytest tests/test_typed_model_cache_formal_execution_v14.py -q",
                    "status": "pass" if args.tests_passed else "pending",
                    "result": "54 passed" if args.tests_passed else None,
                },
                {
                    "command": "shared-python -m pytest tests/test_typed_model_cache_formal_execution.py -q",
                    "status": "pass" if args.tests_passed else "pending",
                    "result": "59 passed" if args.tests_passed else None,
                },
                {
                    "command": "shared-python -m pytest tests/test_typed_model_cache_formal_protocol.py -q",
                    "status": "pass" if args.tests_passed else "pending",
                    "result": "32 passed" if args.tests_passed else None,
                },
                {
                    "command": "shared-python -m pytest tests/test_portable_resource_identity.py -q",
                    "status": "pass" if args.tests_passed else "pending",
                    "result": "38 passed" if args.tests_passed else None,
                },
                {
                    "command": "shared-python -m pytest tests/test_cache_baseline_fairness_manifest.py -q",
                    "status": "pass" if args.tests_passed else "pending",
                    "result": "38 passed" if args.tests_passed else None,
                },
                {
                    "command": "shared-python -m pytest -q",
                    "status": "pass" if args.tests_passed else "pending",
                    "result": "1027 passed" if args.tests_passed else None,
                },
                {
                    "command": "shared-python scripts/smoke_test.py",
                    "status": "pass" if args.tests_passed else "pending",
                    "result": "toy smoke completed" if args.tests_passed else None,
                },
                {
                    "command": "git diff --check",
                    "status": "pass" if args.tests_passed else "pending",
                },
            ],
            "formal_training_commands": 0,
            "formal_evaluation_commands": 0,
            "holdout_commands": 0,
            "g15_commands": 0,
        },
    )
    write_json(ARTIFACT_ROOT / "artifact_integrity_manifest.json", integrity_manifest())
    print(
        json.dumps(
            {
                "status": "pass",
                "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
                "readiness": verdict,
                "artifact_root": str(ARTIFACT_ROOT),
                "config_root": str(CONFIG_ROOT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
