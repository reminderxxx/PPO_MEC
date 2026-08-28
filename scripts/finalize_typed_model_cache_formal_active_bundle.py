"""Evidence-gated, atomic finalizer for the G14R7A v1.8 active bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.active_formal_bundle import (
    ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION,
    READINESS_VERSION,
    READY_STATUS,
    active_bundle_core_projection,
    canonical_sha256,
    ready_index_projection,
    sha256_file,
    validate_active_formal_bundle,
)


RUN_ID = "typed_model_cache_formal_active_bundle_closure_20260827_g14r7a_v1"
ARTIFACT_ROOT = ROOT / "artifacts/analysis" / RUN_ID
CONFIG_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_8_20260827"
INDEX_PATH = CONFIG_ROOT / "protocol_index.json"
READINESS_PATH = CONFIG_ROOT / "readiness_v10.json"
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


def encoded(payload: Any) -> str:
    return json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def write_create_only(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded(payload))
        handle.flush()
        os.fsync(handle.fileno())


def atomic_replace_pending_index(payload: Any) -> None:
    temporary = INDEX_PATH.parent / (
        f".{INDEX_PATH.name}.finalize-{os.getpid()}-{time.monotonic_ns()}"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, INDEX_PATH)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_acceptance(acceptance: dict[str, Any], core_sha256: str) -> None:
    required_true = (
        "clean_candidate",
        "without_local_venv",
        "active_bundle_core_validation_pass",
        "all_protocol_commands_dry_run_pass",
        "outer_nested_expansion_equal",
        "real_preflight_pass",
        "real_tests_phase_pass",
        "training_command_order_audit_pass",
        "dev_fairness_probe_pass",
        "full_pytest_pass",
        "smoke_pass",
        "compile_import_pass",
        "git_diff_check_pass",
        "protected_user_files_unchanged",
        "holdout_sealed_unopened",
    )
    if acceptance.get("status") != "pass" or any(
        acceptance.get(field) is not True for field in required_true
    ):
        raise ValueError("clean acceptance evidence is incomplete")
    if acceptance.get("active_bundle_core_sha256") != core_sha256:
        raise ValueError("clean acceptance active bundle core drift")
    exact = {
        "ngsim_raw_rows": 11850526,
        "provider_frames": 73871,
        "reachable_windows": 60,
        "expected_windows": 60,
        "training_command_count": 150,
        "dev_agent_count": 15,
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
    }
    for field, expected in exact.items():
        if acceptance.get(field) != expected:
            raise ValueError(f"clean acceptance count drift: {field}")
    if acceptance.get("unresolved_placeholder_count") != 0:
        raise ValueError("clean acceptance contains unresolved placeholders")
    if acceptance.get("absolute_command_sentinel_count") != 0:
        raise ValueError("clean acceptance contains /ABSOLUTE/ command sentinel")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance-summary", required=True)
    args = parser.parse_args()
    acceptance_path = Path(args.acceptance_summary).resolve()
    acceptance = read_json(acceptance_path)
    index = read_json(INDEX_PATH)
    if index.get("active_formal_bundle_contract_version") != (
        ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION
    ):
        raise ValueError("active bundle contract version drift")
    core_sha256 = canonical_sha256(active_bundle_core_projection(index))
    if index.get("active_bundle_core_sha256") != core_sha256:
        raise ValueError("pending active bundle core drift")

    if index.get("status") == READY_STATUS:
        readiness = read_json(READINESS_PATH)
        verify_acceptance(acceptance, core_sha256)
        # Idempotency is allowed only when the complete frozen graph still
        # validates.  In particular, a drifted readiness/evidence file must
        # not be reported as an already-finalized success merely because the
        # pending acceptance summary still passes.
        validate_active_formal_bundle(
            repository_root=ROOT,
            require_clean_git=False,
            require_origin_main_match=False,
        )
        if readiness.get("active_bundle_core_sha256") != core_sha256:
            raise ValueError("ready finalizer re-entry found bundle drift")
        if index.get("active_formal_bundle_sha256") != canonical_sha256(
            ready_index_projection(index)
        ):
            raise ValueError("ready finalizer re-entry found index drift")
        print(
            json.dumps(
                {
                    "status": "already_finalized_idempotent",
                    "readiness": READY_STATUS,
                    "active_formal_bundle_sha256": index[
                        "active_formal_bundle_sha256"
                    ],
                },
                indent=2,
            )
        )
        return
    if index.get("status") != "PENDING_G14R7A_CLEAN_ACCEPTANCE":
        raise ValueError("finalizer requires the unique pending G14R7A index")
    if READINESS_PATH.exists():
        raise ValueError("Readiness companion already exists before finalization")
    verify_acceptance(acceptance, core_sha256)

    evidence_manifest = {
        "status": "pass",
        "created_at": now(),
        "clean_candidate": True,
        "active_bundle_core_sha256": core_sha256,
        "acceptance_summary_path": acceptance_path.relative_to(ROOT).as_posix(),
        "acceptance_summary_sha256": sha256_file(acceptance_path),
        "acceptance_summary": acceptance,
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
        "holdout_opened": False,
    }
    evidence_path = ARTIFACT_ROOT / "readiness_evidence_manifest_v10.json"
    if evidence_path.exists():
        raise ValueError("Readiness evidence manifest already exists")
    write_create_only(evidence_path, evidence_manifest)
    readiness = {
        "readiness_review_version": READINESS_VERSION,
        "status": "ready",
        "verdict": READY_STATUS,
        "reviewed_at": now(),
        "literature_cutoff": "2026-08-27",
        "target_venue": "IEEE Transactions on Mobile Computing (TMC)",
        "artifact_run_id": RUN_ID,
        "policy_version": "tmc_review_policy_v3_20260621",
        "git_commit_binding": (
            "exact observed clean 40-hex HEAD == origin/main at execution"
        ),
        "evidence_level": "E2_EXECUTION_CONTRACT_VALIDATED_NO_FORMAL_PERFORMANCE",
        "active_bundle_core_sha256": core_sha256,
        "evidence_manifest_path": evidence_path.relative_to(ROOT).as_posix(),
        "evidence_manifest_sha256": sha256_file(evidence_path),
        "formal_completed": False,
        "paper_ready": False,
        "holdout_opened": False,
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
    }
    write_create_only(READINESS_PATH, readiness)
    readiness_row = {
        "logical_id": "readiness_companion",
        "role": "Readiness v10 evidence companion",
        "logical_path": READINESS_PATH.relative_to(ROOT).as_posix(),
        "content_sha256": sha256_file(READINESS_PATH),
        "size_bytes": READINESS_PATH.stat().st_size,
        "version_scope": "current_protocol_version",
    }
    index["active_bundle_resources"].append(readiness_row)
    index["status"] = READY_STATUS
    index["readiness_companion"] = {
        "version": READINESS_VERSION,
        "logical_path": READINESS_PATH.relative_to(ROOT).as_posix(),
        "content_sha256": readiness_row["content_sha256"],
        "evidence_manifest_sha256": readiness["evidence_manifest_sha256"],
    }
    index["active_formal_bundle_sha256"] = canonical_sha256(
        ready_index_projection(index)
    )
    atomic_replace_pending_index(index)

    producer_consumer = {
        "status": "pass",
        "rows": [
            {
                "producer": "repair_typed_model_cache_formal_agent_order.py",
                "consumer": "v1.7 protocol_index.json",
                "root_cause": "deep-copied v1.6 index without updating execution_environment_manifest",
                "old_behavior": "pending index plus separately ready Readiness v9",
                "v1_8_control": "ordinary generator can only write pending; finalizer atomically freezes ready",
            },
            {
                "producer": "finalize_typed_model_cache_formal_agent_order_repair.py",
                "consumer": "Readiness v9 artifact only",
                "root_cause": "did not audit or update active index status/path/content identity",
                "old_behavior": "readiness could diverge from active index",
                "v1_8_control": "Readiness content hash and evidence hash enter the ready index hash",
            },
            {
                "producer": "run_typed_model_cache_formal_protocol.py CLI",
                "consumer": "Protocol and environment chosen by caller",
                "root_cause": "no active-index authority existed",
                "old_behavior": "correct CLI environment could bypass an incorrect index",
                "v1_8_control": "complete active bundle validates before output-root creation in dry-run and real execution",
            },
            {
                "producer": "active Protocol v1.8 bundle",
                "consumer": "binding/context/phase/cell/checkpoint/integrity provenance",
                "root_cause": "bundle identity was absent downstream",
                "old_behavior": "only separately supplied Protocol/environment identities propagated",
                "v1_8_control": "active_formal_bundle_sha256 is a required downstream identity",
            },
        ],
    }
    write_create_only(ARTIFACT_ROOT / "producer_consumer_matrix.json", producer_consumer)
    write_create_only(
        ARTIFACT_ROOT / "active_formal_bundle_contract.json",
        {
            "active_formal_bundle_contract_version": ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION,
            "active_bundle_core_sha256": core_sha256,
            "active_formal_bundle_sha256": index["active_formal_bundle_sha256"],
            "hash_graph_is_acyclic": True,
            "outer_runner_gate_before_writes": True,
            "manual_cli_override_allowed": False,
            "holdout_capability": False,
        },
    )
    write_create_only(
        ARTIFACT_ROOT / "readiness_finalization_audit.json",
        {
            "status": "pass",
            "transition": ["PENDING_G14R7A_CLEAN_ACCEPTANCE", READY_STATUS],
            "create_only_readiness": True,
            "atomic_index_replace": True,
            "ordinary_generator_ready_overwrite_refused": True,
            "finalizer_reentry": "idempotent only if all identities are unchanged",
            "evidence_manifest_sha256": readiness["evidence_manifest_sha256"],
            "readiness_content_sha256": readiness_row["content_sha256"],
            "active_formal_bundle_sha256": index["active_formal_bundle_sha256"],
        },
    )
    write_create_only(
        ARTIFACT_ROOT / "protected_user_file_hashes_end.json",
        {
            "files": {name: sha256_file(ROOT / name) for name in PROTECTED_FILES},
            "required_count": len(PROTECTED_FILES),
        },
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "readiness": READY_STATUS,
                "active_bundle_core_sha256": core_sha256,
                "active_formal_bundle_sha256": index[
                    "active_formal_bundle_sha256"
                ],
                "readiness_evidence_sha256": readiness[
                    "evidence_manifest_sha256"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
