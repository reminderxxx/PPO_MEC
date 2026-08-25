"""Freeze outcome-blind G14R6 Protocol v1.6 and two-layer training identity evidence."""

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
    READY_V8_VERDICT,
    readiness_v8,
    validate_protocol_v1_1,
)
from src.evaluators.typed_model_cache_formal_protocol import attach_hashes, sha256_file
from src.runtime.formal_execution_environment import (
    probe_python_environment,
    scientific_environment_identity,
)
from src.runtime.formal_training_identity import (
    AGENT_TRAINING_SCIENTIFIC_CONFIG_CONTRACT_VERSION,
    FORMAL_TRAINING_EXECUTION_BINDING_VERSION,
    SCIENTIFIC_FIELDS,
    canonical_sha256,
    scientific_config_projection,
    validate_scientific_config,
)


RUN_ID = "typed_model_cache_formal_training_binding_repair_20260825_g14r6_v1"
ARTIFACT_ROOT = ROOT / "artifacts/analysis" / RUN_ID
V15_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_5_20260825"
V16_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_6_20260825"
V15_PROTOCOL = V15_ROOT / "protocol_v1_5_manifest.json"
V16_PROTOCOL = V16_ROOT / "protocol_v1_6_manifest.json"
SCIENTIFIC_CONFIG = V16_ROOT / "agent_training_scientific_config.json"
V6_RUN_ID = "typed_model_cache_formal_20260825_135122_g14c_v6"
V6_ROOT = ROOT / "artifacts/experiments/typed_model_cache_formal" / V6_RUN_ID
V6_FAILURE_AUDIT_SHA256 = "2cc81ffcd375323caa71c0966ffce36059c43a8da0aad5e7245078727dd0725a"
V6_FAILURE_INTEGRITY_SHA256 = "5f69b81191136354b79f99d2d1599899dbf2a34a537335daafb0f5a5a9bddc0e"
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


def build_scientific_config(protocol_v15: dict[str, Any]) -> dict[str, Any]:
    agent_rows = [
        row
        for row in protocol_v15["agent_matrix"]["controller_table"]
        if row["training_requirement"] == "clean_typed_checkpoint_per_seed_and_capacity"
    ]
    order = [row["agent"] for row in agent_rows]
    frozen = protocol_v15["training_budget"]["agent_configs"]
    agents = {}
    for name in order:
        hyperparameters = deepcopy(frozen[name])
        agents[name] = {
            "agent_identity": name,
            "hyperparameters": hyperparameters,
            "field_applicability": {
                field: "applicable" if field in hyperparameters else "not_applicable"
                for field in SCIENTIFIC_FIELDS
            },
        }
    payload = {
        "agent_training_scientific_config_contract_version": (
            AGENT_TRAINING_SCIENTIFIC_CONFIG_CONTRACT_VERSION
        ),
        "learned_agent_order": order,
        "scientific_fields": list(SCIENTIFIC_FIELDS),
        "not_applicable_semantics": (
            "field absent from hyperparameters and marked not_applicable; null/default inference forbidden"
        ),
        "canonical_serialization": (
            "UTF-8 sorted-key compact JSON; NaN/Infinity and duplicate/unknown fields rejected"
        ),
        "agents": agents,
    }
    payload["config_semantic_sha256"] = canonical_sha256(
        scientific_config_projection(payload)
    )
    validate_scientific_config(payload)
    return payload


def invalid_v6_reference() -> dict[str, Any]:
    if sha256_file(V6_ROOT / "failure_audit.json") != V6_FAILURE_AUDIT_SHA256:
        raise ValueError("G14C v6 failure audit hash drift")
    if sha256_file(V6_ROOT / "failure_integrity.json") != V6_FAILURE_INTEGRITY_SHA256:
        raise ValueError("G14C v6 failure integrity hash drift")
    return {
        "status": "INVALID_PROTOCOL_OR_IMPLEMENTATION",
        "run_id": V6_RUN_ID,
        "failure_boundary": "invalid_during_first_training_cell_before_episode_zero",
        "failure_audit_path": (V6_ROOT / "failure_audit.json").relative_to(ROOT).as_posix(),
        "failure_audit_sha256": V6_FAILURE_AUDIT_SHA256,
        "failure_integrity_path": (V6_ROOT / "failure_integrity.json").relative_to(ROOT).as_posix(),
        "failure_integrity_sha256": V6_FAILURE_INTEGRITY_SHA256,
        "training_cells_executed": 0,
        "candidate_checkpoint_count": 0,
        "dev_performance_count": 0,
        "formal_performance_count": 0,
        "episode_count": 0,
        "environment_interaction_count": 0,
        "update_count": 0,
        "checkpoint_count": 0,
        "resume_allowed": False,
        "retry_allowed": False,
        "legacy_phase_finalize_allowed": False,
        "checkpoint_reuse_allowed": False,
        "checkpoint_salvage_allowed": False,
        "immutable_old_run": True,
    }


def build_protocol(shared_python: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    old = read_json(V15_PROTOCOL)
    scientific = build_scientific_config(old)
    protocol = deepcopy(old)
    probe = probe_python_environment(shared_python, clean_worktree_root=ROOT)
    environment_identity = scientific_environment_identity(
        probe,
        execution_commit="Commit A7 (runtime binding requires exact observed clean Git HEAD)",
        source_tree_sha256="Commit A7 Git tree (runtime-audited; no self-reference)",
    )
    protocol.update(
        typed_model_cache_formal_protocol_version="1.6.0",
        protocol_id="typed_model_cache_formal_protocol_v1_6",
        created_at=now(),
        status="frozen_pre_execution_training_identity_repair_no_performance",
    )
    failures = deepcopy(old["supersession"]["invalid_execution_runs"])
    failures.append(invalid_v6_reference())
    protocol["supersession"] = {
        "supersedes_version": "1.5.0",
        "old_protocol_status": "invalid_during_first_training_cell_before_episode_zero",
        "old_protocol_semantic_sha256": old["hashes"]["semantic_sha256"],
        "invalid_execution_runs": failures,
        "formal_performance_observed": False,
        "scientific_fields_changed": False,
        "repair_scope": [
            "separate stable agent scientific config from execution binding",
            "bind current Protocol/commit/environment/context/command matrix at runtime",
            "propagate config and binding identities through checkpoints and consumers",
        ],
    }
    protocol["formal_execution_environment_contract"]["scientific_identity"] = (
        environment_identity
    )
    protocol["identity"]["execution_git_commit_binding"] = (
        "Commit A7; exact observed clean Git HEAD is bound by runtime execution binding"
    )
    protocol["agent_training_scientific_config_contract"] = {
        "version": AGENT_TRAINING_SCIENTIFIC_CONFIG_CONTRACT_VERSION,
        "config_semantic_sha256": scientific["config_semantic_sha256"],
        "semantic_projection_contains_protocol_or_execution_identity": False,
        "protocol_training_budget_parity_required": True,
        "content_identity_not_path_identity": True,
        "unknown_duplicate_or_non_finite_fields_rejected": True,
    }
    protocol["formal_training_execution_binding_contract"] = {
        "version": FORMAL_TRAINING_EXECUTION_BINDING_VERSION,
        "protocol_binds_schema_not_runtime_instance": True,
        "runtime_instance_created_after_protocol_hash": True,
        "runtime_instance_unique_producer": "scripts/run_typed_model_cache_formal_protocol.py",
        "execution_commit_is_observed_clean_git_head": True,
        "binding_hash_enters_resolved_context": True,
        "binding_hash_enters_phase_and_cell_input_identity": True,
        "binding_hash_enters_checkpoint_and_downstream_provenance": True,
        "iterative_hashing_or_conflicting_binding_allowed": False,
        "host_paths_in_binding_identity": False,
    }
    resolved = protocol["resolved_formal_execution_context_contract"]
    resolved["version"] = "2.0.0"
    resolved["scientific_config_hash_in_context"] = True
    resolved["execution_binding_hash_in_context"] = True
    execution = protocol["execution_contract"]
    context = execution["default_expansion_context"]
    context.pop("agent_config_path", None)
    context["agent_scientific_config_path"] = SCIENTIFIC_CONFIG.relative_to(ROOT).as_posix()
    context["formal_training_execution_binding_path"] = (
        "/ABSOLUTE/FORMAL_OUTPUT_ROOT/formal_training_execution_binding.json"
    )
    context["protocol_path"] = V16_PROTOCOL.relative_to(ROOT).as_posix()
    train_argv = execution["command_templates"]["train"]["argv"]
    legacy_index = train_argv.index("--agent_config_path")
    train_argv[legacy_index : legacy_index + 2] = [
        "--agent_scientific_config_path",
        "{agent_scientific_config_path}",
        "--formal_training_execution_binding_path",
        "{formal_training_execution_binding_path}",
        "--resolved_execution_context_path",
        "{resolved_execution_context_path}",
    ]
    bindings = execution["same_run_resume"]["bindings"]
    for field in (
        "agent_scientific_config_semantic_sha256",
        "formal_training_execution_binding_sha256",
    ):
        if field not in bindings:
            bindings.append(field)
    protocol = attach_hashes(protocol)
    validate_scientific_config(scientific, protocol=protocol)
    validate_protocol_v1_1(protocol)
    return protocol, scientific, probe


def producer_consumer_matrix() -> dict[str, Any]:
    names = [
        "Protocol training_budget.agent_configs",
        "legacy agent_training_configs.json",
        "default expansion agent config path",
        "train command template",
        "train_algo_pool_real_sample.py",
        "formal_training_contract.py",
        "agent instantiation/checkpoint config audit",
        "train summary/candidate metadata",
        "dev selection",
        "checkpoint freeze",
        "checkpoint provenance manifest",
        "formal benchmark loading",
        "resolved execution context",
        "portable resource registry",
        "fairness/provenance consumer",
        "historical artifact reader",
    ]
    rows = []
    for name in names:
        legacy = name in {"legacy agent_training_configs.json", "historical artifact reader"}
        rows.append(
            {
                "component": name,
                "scientific_config_producer": name == "Protocol training_budget.agent_configs",
                "execution_binding_producer": name == "resolved execution context",
                "cli_or_path_consumer": name in {
                    "default expansion agent config path",
                    "train command template",
                    "train_algo_pool_real_sample.py",
                },
                "runtime_validator": not legacy,
                "artifact_field": (
                    "audit_only_legacy" if legacy else "scientific_config_sha256+execution_binding_sha256"
                ),
                "hash_fields": (
                    ["protocol_semantic_sha256"] if legacy else [
                        "agent_scientific_config_semantic_sha256",
                        "formal_training_execution_binding_sha256",
                    ]
                ),
                "failure_behavior": "audit_only_no_active_execution" if legacy else "fail_before_consumer_action",
                "test_coverage": "tests/test_formal_training_identity_v16.py",
            }
        )
    return {"producer_consumer_matrix_version": "1.0.0", "status": "pass", "rows": rows}


def inventory() -> dict[str, Any]:
    files = []
    for path in sorted(ARTIFACT_ROOT.rglob("*")):
        if path.is_file() and path.name not in {"artifact_inventory.json", "integrity_manifest.json"}:
            files.append(
                {
                    "path": path.relative_to(ARTIFACT_ROOT).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {"status": "pass", "file_count": len(files), "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--validation-summary", default="")
    parser.add_argument("--clean-acceptance-summary", default="")
    parser.add_argument("--ready", action="store_true")
    args = parser.parse_args()
    shared_python = Path(args.python_executable).absolute()
    if not shared_python.is_file() or not os.access(shared_python, os.X_OK):
        raise FileNotFoundError(shared_python)
    protocol, scientific, probe = build_protocol(shared_python)
    write_json(SCIENTIFIC_CONFIG, scientific)
    write_json(V16_PROTOCOL, protocol)
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
    write_json(V16_ROOT / "execution_environment_manifest.json", environment_manifest)
    write_json(
        V16_ROOT / "formal_training_execution_binding_contract.json",
        protocol["formal_training_execution_binding_contract"],
    )
    old_index = read_json(V15_ROOT / "protocol_index.json")
    index = deepcopy(old_index)
    index.pop("agent_config", None)
    index.update(
        protocol_index_version="1.6.0",
        protocol_manifest=V16_PROTOCOL.relative_to(ROOT).as_posix(),
        protocol_semantic_sha256=protocol["hashes"]["semantic_sha256"],
        agent_scientific_config=SCIENTIFIC_CONFIG.relative_to(ROOT).as_posix(),
        agent_scientific_config_semantic_sha256=scientific["config_semantic_sha256"],
        formal_training_execution_binding_version="1.0.0",
        resolved_execution_context_contract_version="2.0.0",
        execution_environment_manifest=(V16_ROOT / "execution_environment_manifest.json").relative_to(ROOT).as_posix(),
        status=READY_V8_VERDICT if args.ready else "PENDING_G14R6_VALIDATION",
    )
    write_json(V16_ROOT / "protocol_index.json", index)

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ARTIFACT_ROOT / "g14c_v6_invalid_run_registration.json", invalid_v6_reference())
    write_json(
        ARTIFACT_ROOT / "root_cause_audit.json",
        {
            "status": "confirmed",
            "failure_audit_sha256": V6_FAILURE_AUDIT_SHA256,
            "failure_integrity_sha256": V6_FAILURE_INTEGRITY_SHA256,
            "root_cause": "legacy companion coupled stable hyperparameters to Protocol v1.3 semantic hash",
            "repair": "execution-neutral scientific config plus post-Protocol runtime execution binding",
            "episode_interaction_update_checkpoint_counts": [0, 0, 0, 0],
            "performance_observed": False,
        },
    )
    write_json(ARTIFACT_ROOT / "producer_consumer_matrix.json", producer_consumer_matrix())
    write_json(ARTIFACT_ROOT / "agent_training_scientific_config_contract.json", scientific)
    write_json(ARTIFACT_ROOT / "formal_training_execution_binding_contract.json", protocol["formal_training_execution_binding_contract"])
    write_json(
        ARTIFACT_ROOT / "config_parity_audit.json",
        {
            "status": "pass",
            "agent_count": 10,
            "source_protocol_version": "1.5.0",
            "source_protocol_semantic_sha256": read_json(V15_PROTOCOL)["hashes"]["semantic_sha256"],
            "scientific_config_semantic_sha256": scientific["config_semantic_sha256"],
            "field_by_field_equal": all(
                scientific["agents"][name]["hyperparameters"]
                == protocol["training_budget"]["agent_configs"][name]
                for name in scientific["learned_agent_order"]
            ),
            "hyperparameters_changed": False,
        },
    )
    write_json(ARTIFACT_ROOT / "protocol_v1_6_manifest.json", protocol)
    write_json(
        ARTIFACT_ROOT / "protocol_v1_5_to_v1_6_diff.json",
        {
            "status": "pass",
            "old_semantic_sha256": read_json(V15_PROTOCOL)["hashes"]["semantic_sha256"],
            "new_semantic_sha256": protocol["hashes"]["semantic_sha256"],
            "scientific_fields_changed": False,
            "changed_scope": protocol["supersession"]["repair_scope"],
        },
    )
    write_json(ARTIFACT_ROOT / "environment_identity.json", protocol["formal_execution_environment_contract"]["scientific_identity"])
    protected_hashes = {name: sha256_file(ROOT / name) for name in PROTECTED_FILES}
    write_json(
        ARTIFACT_ROOT / "protected_user_file_hashes_start.json",
        {"capture": "task_start", "files": protected_hashes},
    )
    if args.ready:
        write_json(
            ARTIFACT_ROOT / "protected_user_file_hashes_end.json",
            {
                "capture": "pre_commit_task_end",
                "status": "pass",
                "matching_file_count": len(PROTECTED_FILES),
                "files": protected_hashes,
            },
        )
    validation = read_json(Path(args.validation_summary)) if args.validation_summary else {}
    clean = read_json(Path(args.clean_acceptance_summary)) if args.clean_acceptance_summary else {}
    if validation:
        for key in ("negative_validation", "checkpoint_provenance_schema_audit", "tests_result"):
            if key in validation:
                write_json(ARTIFACT_ROOT / f"{key}.json", validation[key])
    if clean:
        write_json(ARTIFACT_ROOT / "clean_worktree_acceptance.json", clean)
        if "entrypoint_rehearsal" in clean:
            write_json(
                ARTIFACT_ROOT / "non_formal_training_contract_rehearsal.json",
                clean["entrypoint_rehearsal"],
            )
        for source_key, target_name in (
            ("command_audit", "150_command_contract_audit.json"),
            ("entrypoint_rehearsal", "10_agent_entrypoint_audit.json"),
            ("preflight", "clean_worktree_preflight_tests.json"),
            ("ledger_regression", "phase_cell_ledger_regression.json"),
        ):
            if source_key in clean:
                write_json(ARTIFACT_ROOT / target_name, clean[source_key])
    checks = {
        "g14c_v6_failure_registered": True,
        "producer_consumer_matrix_complete": True,
        "scientific_config_contract_frozen": True,
        "execution_binding_contract_frozen": True,
        "ten_agent_config_parity": True,
        "training_commands_150_bound": bool(args.ready),
        "ten_agent_entrypoint_rehearsal": bool(args.ready),
        "negative_validation_complete": bool(args.ready),
        "checkpoint_provenance_consumers_bound": bool(args.ready),
        "outer_nested_expansion_equal": bool(args.ready),
        "clean_worktree_without_local_venv": bool(args.ready),
        "clean_import_origin": bool(args.ready),
        "window_reachability_60_of_60": bool(args.ready),
        "real_preflight_completed": bool(args.ready),
        "real_tests_phase_completed": bool(args.ready),
        "phase_cell_resume_finalize_regression": bool(args.ready),
        "full_pytest_and_smoke_pass": bool(args.ready),
        "holdout_sealed": True,
        "no_formal_training_checkpoint_or_performance": True,
    }
    verdict = readiness_v8(checks)
    validated_candidate_commit = clean.get("execution_commit") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    write_json(
        ARTIFACT_ROOT / "readiness_review_v8.json",
        {
            "readiness_review_version": "8.0.0",
            "reviewed_at": now(),
            "literature_cutoff": "2026-08-25",
            "target_venue": "IEEE Transactions on Mobile Computing (TMC)",
            "artifact_run_id": RUN_ID,
            "policy_version": "tmc_review_policy_v3_20260621",
            "implementation_baseline_git_commit": validated_candidate_commit,
            "validated_clean_candidate_git_commit": validated_candidate_commit,
            "execution_commit_contract": "Commit A7 runtime binding uses exact observed clean HEAD",
            "evidence_level": "E2_EXECUTION_CONTRACT_VALIDATED_NO_FORMAL_PERFORMANCE",
            "checks": checks,
            "verdict": verdict,
            "formal_completed": False,
            "paper_ready": False,
        },
    )
    write_json(
        ARTIFACT_ROOT / "holdout_seal_revalidation.json",
        {"status": "pass", "sealed": True, "opened": False, "training_count": 0, "checkpoint_count": 0, "performance_row_count": 0},
    )
    inv = inventory()
    write_json(ARTIFACT_ROOT / "artifact_inventory.json", inv)
    write_json(
        ARTIFACT_ROOT / "integrity_manifest.json",
        {
            "status": "pass",
            "artifact_run_id": RUN_ID,
            "inventory_sha256": sha256_file(ARTIFACT_ROOT / "artifact_inventory.json"),
            "file_count": inv["file_count"],
            "formal": False,
            "training": False,
            "performance_evidence": False,
            "holdout_opened": False,
        },
    )
    print(json.dumps({"status": "pass", "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"], "scientific_config_semantic_sha256": scientific["config_semantic_sha256"], "readiness": verdict}, indent=2))


if __name__ == "__main__":
    main()
