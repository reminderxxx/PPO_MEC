"""G14R20-I5 bounded recovery identity, order, and public acceptance."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts.run_typed_model_cache_restricted_recovery import (
    RestrictedRecoverySingleWriter,
    verify_recovery_grant,
)
from scripts.continuation_executor.locking import writer_lock_path
from src.evaluators.formal_cell_transaction import CellExecutionIdentity, CellTransactionError
from src.runtime.evaluation_only_execution import canonical_sha256, file_sha256
from src.runtime.restricted_recovery import (
    ALLOWED_CELL_IDS,
    FAILED_CELL_ID,
    ORIGINAL_PROJECT_GRANT_PATH,
    ORIGINAL_REQUEST_PATH,
    ORIGINAL_RUN_ROOT,
    PHASE,
    RestrictedRecoveryCellLedger,
    RestrictedRecoveryError,
    UNSTARTED_CELL_ID,
    audit_original_recovery_source,
    restricted_cell_layout,
    validate_restricted_recovery_request,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2] if ROOT.parent.name == "execution_checkouts" else ROOT
PYTHON = str(PROJECT / ".venv/bin/python")
PREPARE = ROOT / "scripts/prepare_typed_model_cache_restricted_recovery.py"
DRIVER = ROOT / "tests/restricted_recovery_public_driver.py"

CONTEXTS = [
    {
        "ablation_setting_id": "ablation-f09814754dc92882",
        "capacity_label": "medium_576mb",
        "checkpoint_manifest_id": "checkpoint_manifest.medium_576mb",
        "checkpoint_provenance_id": "checkpoint_provenance.medium_576mb",
        "fairness_manifest_path": "configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820/fairness_support_ablation-f09814754dc92882.json",
        "fairness_manifest_resource_id": "fairness_manifest.support.ablation-f09814754dc92882",
        "primary_vehicle_selection": "handoff_pressure",
        "runtime_config_resource_id": "runtime_config.medium_576mb",
    },
    {
        "ablation_setting_id": "ablation-8aa367acf3bdf7a9",
        "capacity_label": "medium_576mb",
        "checkpoint_manifest_id": "checkpoint_manifest.medium_576mb",
        "checkpoint_provenance_id": "checkpoint_provenance.medium_576mb",
        "fairness_manifest_path": "configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820/fairness_support_ablation-8aa367acf3bdf7a9.json",
        "fairness_manifest_resource_id": "fairness_manifest.support.ablation-8aa367acf3bdf7a9",
        "primary_vehicle_selection": "handoff_pressure",
        "runtime_config_resource_id": "runtime_config.medium_576mb",
    },
]


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def inventory(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def minimal_request(tmp_path: Path) -> dict:
    root = tmp_path / "synthetic_recovery_run"
    commands = [
        [
            "/absolute/python",
            "/executor/scripts/run_typed_model_cache_formal_support.py",
            "--output-root",
            str(root / PHASE),
            "--formal-window-split",
            "formal",
        ]
        for _ in range(2)
    ]
    plan = {"commands": commands, "matrix_contexts": deepcopy(CONTEXTS), "expected_outputs": []}
    environment_identity = {
        "dependency_fingerprint": "d" * 64,
        "environment_fingerprint": "n" * 64,
    }
    python_binding = {
        "restricted_recovery_python_binding_version": "1.0.0",
        "launch_path": "/absolute/python",
        "binary_realpath_audit_only": "/system/python",
        "observed_sys_executable": "/absolute/python",
        "observed_sys_prefix": "/absolute",
        "observed_sys_base_prefix": "/system",
        "virtual_environment_active": True,
        "environment_identity": environment_identity,
        "runtime_audit": {"resolved_python_absolute_path": "/absolute/python"},
    }
    python_binding["binding_sha256"] = canonical_sha256(python_binding)
    context = {
        "context_sha256": "",
        "runtime_location": {
            "resolved_python_absolute_path": "/absolute/python",
            "python_binary_realpath_audit_only": "/system/python",
            "python_environment_binding_sha256": python_binding["binding_sha256"],
        },
        "resolved_expansion_context": {"python_executable": "/absolute/python"},
        "scientific_identity": {
            "dependency_fingerprint": "d" * 64,
            "environment_fingerprint": "n" * 64,
            "full_normalized_environment_projection": deepcopy(environment_identity),
        },
        "evaluation_execution_identity": {
            "executor_commit": "e" * 40,
            "executor_git_tree": "f" * 40,
            "python_environment_binding_sha256": python_binding["binding_sha256"],
        },
    }
    context["context_sha256"] = canonical_sha256(
        {key: value for key, value in context.items() if key != "context_sha256"}
    )
    execution = {
        "restricted_recovery_contract_version": "1.0.0",
        "recovery_execution_id": root.name,
        "recovery_root": str(root),
        "original_run_id": "typed_model_cache_evaluation_only_20260913_g14r20_i3_pending",
        "executor_checkout": "/executor",
        "executor_commit": "e" * 40,
        "executor_git_tree": "f" * 40,
        "python_executable": "/absolute/python",
        "python_environment_binding": python_binding,
        "model_source_reference_sha256": "s" * 64,
        "allowed_phase": PHASE,
        "allowed_cell_ids": list(ALLOWED_CELL_IDS),
        "cell_specs": [
            {"cell_id": FAILED_CELL_ID, "source_attempt": 1, "recovery_attempt": 2, "execution_kind": "recovery_attempt", "must_commit_before_next_cell": True},
            {"cell_id": UNSTARTED_CELL_ID, "source_attempt_count": 0, "recovery_attempt": 1, "execution_kind": "first_execution", "dispatch_after_cell_id": FAILED_CELL_ID},
        ],
        "command_plan": plan,
        "command_plan_sha256": canonical_sha256(plan),
        "resolved_execution_context": context,
        "resolved_execution_context_sha256": context["context_sha256"],
        "phase_ledger_path": str(root / "recovery_phase_state.jsonl"),
        "cell_ledger_path": str(root / "cell_state.jsonl"),
        "staging_root": str(root / ".staging"),
        "lock_path": str(root.parent / ".locks/x"),
        "external_results_copied": False,
        "later_phases_authorized": False,
        "training_authorized": False,
        "dev_selection_authorized": False,
        "checkpoint_freeze_authorized": False,
        "holdout_capability": False,
        "automatic_retry_count": 0,
    }
    execution["recovery_execution_identity_sha256"] = canonical_sha256(execution)
    request = {
        "restricted_recovery_authorization_request_version": "1.0.0",
        "status": "READY_FOR_RESTRICTED_RECOVERY_AUTHORIZATION",
        "authorization_kind": "restricted_formal_ablation_recovery_only",
        "created_at_utc": "2026-09-14T00:00:00+08:00",
        "original_identity": {"request": {"path": str(ORIGINAL_REQUEST_PATH)}, "project_grant": {"path": str(ORIGINAL_PROJECT_GRANT_PATH)}},
        "immutable_source_audit": {},
        "external_committed_cells": [{"cell_id": f"external-{index}"} for index in range(6)],
        "source_models": [{"agent": f"model-{index}"} for index in range(150)],
        "recovery_execution": execution,
        "scientific_invariance": {"scientific_parameters_unchanged": True},
        "source_mapping_and_deduplication": {
            "external_original_count": 6,
            "recovery_execution_count_after_completion": 2,
            "expected_unique_cell_count_after_completion": 8,
            "duplicate_logical_cells_allowed": False,
            "external_cells_may_be_dispatched_or_rewritten": False,
        },
        "later_stage_boundary": {},
        "blocking_preconditions_for_real_execution": [],
        "recovery_grant_issued": False,
        "real_recovery_started": False,
        "holdout_opened": False,
    }
    request["authorization_request_sha256"] = canonical_sha256(request)
    return request


def rehash_python_request(request: dict) -> None:
    execution = request["recovery_execution"]
    binding = execution["python_environment_binding"]
    binding["binding_sha256"] = canonical_sha256(
        {key: value for key, value in binding.items() if key != "binding_sha256"}
    )
    context = execution["resolved_execution_context"]
    context["context_sha256"] = canonical_sha256(
        {key: value for key, value in context.items() if key != "context_sha256"}
    )
    execution["resolved_execution_context_sha256"] = context["context_sha256"]
    execution["command_plan_sha256"] = canonical_sha256(execution["command_plan"])
    execution["recovery_execution_identity_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in execution.items()
            if key != "recovery_execution_identity_sha256"
        }
    )
    request["authorization_request_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in request.items()
            if key != "authorization_request_sha256"
        }
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda request: request.update(holdout_opened=True), "forbidden authority"),
        (lambda request: request["recovery_execution"].update(allowed_phase="formal_support"), "identity hash"),
        (lambda request: request["source_mapping_and_deduplication"].update(expected_unique_cell_count_after_completion=7), "deduplication"),
        (lambda request: request["source_models"].pop(), "membership"),
    ],
)
def test_request_scope_mutations_fail_closed(tmp_path: Path, mutation, message: str) -> None:
    request = minimal_request(tmp_path)
    mutation(request)
    request["authorization_request_sha256"] = canonical_sha256(
        {key: value for key, value in request.items() if key != "authorization_request_sha256"}
    )
    with pytest.raises(RestrictedRecoveryError, match=message):
        validate_restricted_recovery_request(request, check_live=False)


def test_cell_builder_preserves_request_launch_path(tmp_path: Path) -> None:
    request = minimal_request(tmp_path)
    for cell_id in ALLOWED_CELL_IDS:
        command, _, _, builder, _ = restricted_cell_layout(request, cell_id)
        staged = builder(tmp_path / "not-created", cell_id)
        assert command[0] == "/absolute/python"
        assert staged[0] == command[0]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda execution: execution["command_plan"]["commands"][0].__setitem__(
                0, "/system/python"
            ),
            "command Python launch path drift",
        ),
        (
            lambda execution: execution["resolved_execution_context"][
                "runtime_location"
            ].update(resolved_python_absolute_path="/system/python"),
            "request/context/child Python identity drift",
        ),
        (
            lambda execution: execution["python_environment_binding"][
                "environment_identity"
            ].update(dependency_fingerprint="x" * 64),
            "request/context/child Python identity drift",
        ),
        (
            lambda execution: execution["python_environment_binding"].update(
                observed_sys_prefix="/system"
            ),
            "request/context/child Python identity drift",
        ),
    ],
)
def test_python_binding_drift_is_rejected_before_recovery_write(
    tmp_path: Path, mutation, message: str
) -> None:
    request = minimal_request(tmp_path)
    recovery_root = Path(request["recovery_execution"]["recovery_root"])
    mutation(request["recovery_execution"])
    rehash_python_request(request)
    with pytest.raises(RestrictedRecoveryError, match=message):
        validate_restricted_recovery_request(request, check_live=False)
    assert not recovery_root.exists()


def test_recovery_ledger_uses_attempt_two_then_first_attempt(tmp_path: Path) -> None:
    identity = CellExecutionIdentity(
        run_id="synthetic_recovery_run",
        execution_commit="e" * 40,
        protocol_semantic_sha256="p" * 64,
        resource_registry_semantic_sha256="r" * 64,
        environment_fingerprint="n" * 64,
        split_semantic_sha256="s" * 64,
        window_contract_semantic_sha256="w" * 64,
        catalog_fingerprint="c" * 64,
        runtime_identity="t" * 64,
        command_matrix_sha256="m" * 64,
    )
    root = tmp_path / "synthetic_recovery_run"
    ledger = RestrictedRecoveryCellLedger(run_root=root, identity=identity)
    with pytest.raises(CellTransactionError, match="must commit"):
        ledger.begin_cell(
            phase=PHASE,
            coordinates=CONTEXTS[1],
            command=["second"],
            input_hash="input",
            committed_path=root / PHASE / "second",
        )
    first = ledger.begin_cell(
        phase=PHASE,
        coordinates=CONTEXTS[0],
        command=["first"],
        input_hash="input",
        committed_path=root / PHASE / "first",
    )
    assert first["record"]["attempt"] == 2
    staging = Path(first["record"]["staging_path"])
    (staging / "payload.json").write_text("{}\n")
    ledger.commit_cell(first["cell_id"], required_paths=["payload.json"])
    second = ledger.begin_cell(
        phase=PHASE,
        coordinates=CONTEXTS[1],
        command=["second"],
        input_hash="input",
        committed_path=root / PHASE / "second",
    )
    assert second["record"]["attempt"] == 1
    with pytest.raises(CellTransactionError, match="failure/interruption"):
        ledger.begin_cell(
            phase=PHASE,
            coordinates=CONTEXTS[1],
            command=["second"],
            input_hash="input",
            committed_path=root / PHASE / "second",
        )


def test_grant_is_distinct_and_holdout_false(tmp_path: Path) -> None:
    request = minimal_request(tmp_path)
    execution = request["recovery_execution"]
    review = {
        "status": "pass",
        "reviewer_id": "independent",
        "implementation_agent_id": "implementation",
        "authorization_request_sha256": request["authorization_request_sha256"],
        "recovery_execution_identity_sha256": execution["recovery_execution_identity_sha256"],
        "holdout_capability": False,
        "synthetic_only": True,
    }
    review_path = tmp_path / "review.json"
    dump(review_path, review)
    now = datetime.now(timezone.utc)
    grant = {
        "version": "1.0.0",
        "status": "AUTHORIZED_FOR_SYNTHETIC_RESTRICTED_RECOVERY_ACCEPTANCE",
        "authorization_request_sha256": request["authorization_request_sha256"],
        "recovery_execution_id": execution["recovery_execution_id"],
        "original_run_id": execution["original_run_id"],
        "executor_commit": execution["executor_commit"],
        "executor_git_tree": execution["executor_git_tree"],
        "allowed_phase": PHASE,
        "allowed_cell_ids": list(ALLOWED_CELL_IDS),
        "holdout_capability": False,
        "later_phases_authorized": False,
        "independent_review": {"path": str(review_path), "sha256": file_sha256(review_path), "size_bytes": review_path.stat().st_size},
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
    }
    assert verify_recovery_grant(request, grant, synthetic_only=True)["approval_verified"]
    changed = deepcopy(grant)
    changed["holdout_capability"] = True
    with pytest.raises(RestrictedRecoveryError, match="scope drift"):
        verify_recovery_grant(request, changed, synthetic_only=True)
    old_grant = json.loads(ORIGINAL_PROJECT_GRANT_PATH.read_text())
    with pytest.raises(RestrictedRecoveryError):
        verify_recovery_grant(request, old_grant, synthetic_only=True)


def test_recovery_writer_retains_held_state_and_forbids_takeover(tmp_path: Path) -> None:
    root = tmp_path / "new_recovery"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        with RestrictedRecoverySingleWriter(root, "executor-identity", lambda: None):
            raise RuntimeError("synthetic interruption")
    lock = writer_lock_path(root)
    retained = json.loads(lock.read_text())
    assert retained["state"] == "held"
    assert retained["process"]["identity_method"] == "pid_without_liveness_inference"
    with pytest.raises(RestrictedRecoveryError, match="automatic takeover is forbidden"):
        with RestrictedRecoverySingleWriter(root, "executor-identity", lambda: None):
            pass


PUBLIC = os.environ.get("G14R20_I5_PUBLIC_ACCEPTANCE") == "1"


@pytest.mark.skipif(not PUBLIC, reason="run from final clean commit with G14R20_I5_PUBLIC_ACCEPTANCE=1")
def test_public_two_process_handoff_and_zero_write_rejections(tmp_path: Path) -> None:
    scope = tmp_path / "synthetic_restricted_recovery_public"
    scope.mkdir()
    recovery_root = scope / "synthetic_recovery_run"
    request_path = scope / "request.json"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    prepare = [
        PYTHON,
        str(PREPARE),
        "--action",
        "prepare",
        "--original-request-path",
        str(ORIGINAL_REQUEST_PATH),
        "--original-project-grant-path",
        str(ORIGINAL_PROJECT_GRANT_PATH),
        "--original-run-root",
        str(ORIGINAL_RUN_ROOT),
        "--recovery-execution-id",
        recovery_root.name,
        "--recovery-root",
        str(recovery_root),
        "--executor-checkout",
        str(ROOT),
        "--executor-commit",
        head,
        "--python-executable",
        PYTHON,
        "--output-path",
        str(request_path),
    ]
    environment = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    prepared = subprocess.run(prepare, cwd=ROOT, env=environment, text=True, capture_output=True)
    assert prepared.returncode == 0, prepared.stderr
    request = json.loads(request_path.read_text())
    dump(
        scope / "synthetic_binding.json",
        {"test_only": True, "scope": str(scope), "request_sha256": request["authorization_request_sha256"]},
    )
    execution = request["recovery_execution"]
    review = {
        "status": "pass",
        "reviewer_id": "synthetic-independent-review",
        "implementation_agent_id": "synthetic-implementation",
        "authorization_request_sha256": request["authorization_request_sha256"],
        "recovery_execution_identity_sha256": execution["recovery_execution_identity_sha256"],
        "holdout_capability": False,
        "synthetic_only": True,
    }
    review_path = scope / "review.json"
    dump(review_path, review)
    now = datetime.now(timezone.utc)
    grant = {
        "version": "1.0.0",
        "status": "AUTHORIZED_FOR_SYNTHETIC_RESTRICTED_RECOVERY_ACCEPTANCE",
        "authorization_request_sha256": request["authorization_request_sha256"],
        "recovery_execution_id": execution["recovery_execution_id"],
        "original_run_id": execution["original_run_id"],
        "executor_commit": execution["executor_commit"],
        "executor_git_tree": execution["executor_git_tree"],
        "allowed_phase": PHASE,
        "allowed_cell_ids": list(ALLOWED_CELL_IDS),
        "holdout_capability": False,
        "later_phases_authorized": False,
        "independent_review": {"path": str(review_path), "sha256": file_sha256(review_path), "size_bytes": review_path.stat().st_size},
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(hours=2)).isoformat(),
    }
    grant_path = scope / "grant.json"
    dump(grant_path, grant)
    old_snapshot = {
        name: file_sha256(ORIGINAL_RUN_ROOT / name)
        for name in ("phase_state.jsonl", "cell_state.jsonl")
    }
    old_snapshot["held_lock"] = request["immutable_source_audit"]["held_lock"]["sha256"]

    def command(cell_id: str, *, request_file=request_path, grant_file=grant_path):
        return [
            PYTHON,
            str(DRIVER),
            "host",
            str(scope),
            "--authorization-request-path",
            str(request_file),
            "--recovery-grant-path",
            str(grant_file),
            "--cell-id",
            cell_id,
            "--check",
            "execute",
        ]

    def authorization_for(changed_request: dict, label: str) -> Path:
        changed_execution = changed_request["recovery_execution"]
        changed_review = {
            "status": "pass",
            "reviewer_id": f"synthetic-independent-review-{label}",
            "implementation_agent_id": "synthetic-implementation",
            "authorization_request_sha256": changed_request["authorization_request_sha256"],
            "recovery_execution_identity_sha256": changed_execution[
                "recovery_execution_identity_sha256"
            ],
            "holdout_capability": False,
            "synthetic_only": True,
        }
        changed_review_path = scope / f"review_{label}.json"
        dump(changed_review_path, changed_review)
        changed_grant = {
            **grant,
            "authorization_request_sha256": changed_request[
                "authorization_request_sha256"
            ],
            "recovery_execution_id": changed_execution["recovery_execution_id"],
            "original_run_id": changed_execution["original_run_id"],
            "executor_commit": changed_execution["executor_commit"],
            "executor_git_tree": changed_execution["executor_git_tree"],
            "independent_review": {
                "path": str(changed_review_path),
                "sha256": file_sha256(changed_review_path),
                "size_bytes": changed_review_path.stat().st_size,
            },
        }
        changed_grant_path = scope / f"grant_{label}.json"
        dump(changed_grant_path, changed_grant)
        dump(
            scope / "synthetic_binding.json",
            {
                "test_only": True,
                "scope": str(scope),
                "request_sha256": changed_request["authorization_request_sha256"],
            },
        )
        return changed_grant_path

    def rejected(argv):
        dispatch_before = len(rows(scope / "dispatch.jsonl"))
        lock_path = writer_lock_path(recovery_root)
        write_before = {
            "root": inventory(recovery_root),
            "lock": file_sha256(lock_path) if lock_path.is_file() else None,
        }
        result = subprocess.run(argv, cwd=ROOT, env=environment, text=True, capture_output=True)
        assert result.returncode != 0
        assert len(rows(scope / "dispatch.jsonl")) == dispatch_before
        assert {
            "root": inventory(recovery_root),
            "lock": file_sha256(lock_path) if lock_path.is_file() else None,
        } == write_before

    rejected(command(UNSTARTED_CELL_ID))
    rejected(command(FAILED_CELL_ID, grant_file=scope / "missing_grant.json"))
    rejected(command("formal_support-not-authorized"))
    bad_grant = deepcopy(grant)
    bad_grant["holdout_capability"] = True
    bad_grant_path = scope / "bad_grant.json"
    dump(bad_grant_path, bad_grant)
    rejected(command(FAILED_CELL_ID, grant_file=bad_grant_path))
    symlink_request = scope / "request_symlink.json"
    symlink_request.symlink_to(request_path)
    rejected(command(FAILED_CELL_ID, request_file=symlink_request))
    corrupt_hash = deepcopy(request)
    corrupt_hash["authorization_request_sha256"] = "0" * 64
    corrupt_hash_path = scope / "corrupt_request_hash.json"
    dump(corrupt_hash_path, corrupt_hash)
    rejected(command(FAILED_CELL_ID, request_file=corrupt_hash_path))
    for index, mutate in enumerate(
        (
            lambda item: item["original_identity"].update(source_reference_sha256="0" * 64),
            lambda item: item["immutable_source_audit"]["phase_ledger_prefix"].update(prefix_sha256="0" * 64),
            lambda item: item["immutable_source_audit"]["cell_ledger_prefix"].update(prefix_sha256="0" * 64),
            lambda item: item["external_committed_cells"][0].update(transaction_inventory_sha256="0" * 64),
            lambda item: item["source_models"][0].update(agent="wrong-agent"),
            lambda item: item["recovery_execution"].update(executor_commit="0" * 40),
            lambda item: item["recovery_execution"].update(recovery_root=str(ORIGINAL_RUN_ROOT)),
        )
    ):
        changed = deepcopy(request)
        mutate(changed)
        if index in {5, 6}:
            changed_execution = changed["recovery_execution"]
            changed_execution["recovery_execution_identity_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in changed_execution.items()
                    if key != "recovery_execution_identity_sha256"
                }
            )
        changed["authorization_request_sha256"] = canonical_sha256(
            {key: value for key, value in changed.items() if key != "authorization_request_sha256"}
        )
        changed_path = scope / f"changed_{index}.json"
        dump(changed_path, changed)
        changed_grant_path = authorization_for(changed, str(index))
        rejected(
            command(
                FAILED_CELL_ID,
                request_file=changed_path,
                grant_file=changed_grant_path,
            )
        )

    linked_parent = scope / "linked_output_parent"
    linked_parent.symlink_to(scope, target_is_directory=True)
    linked = deepcopy(request)
    linked_execution = linked["recovery_execution"]
    linked_execution["recovery_root"] = str(
        linked_parent / linked_execution["recovery_execution_id"]
    )
    linked_execution["recovery_execution_identity_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in linked_execution.items()
            if key != "recovery_execution_identity_sha256"
        }
    )
    linked["authorization_request_sha256"] = canonical_sha256(
        {key: value for key, value in linked.items() if key != "authorization_request_sha256"}
    )
    linked_path = scope / "linked_output_request.json"
    dump(linked_path, linked)
    linked_grant = authorization_for(linked, "linked")
    rejected(command(FAILED_CELL_ID, request_file=linked_path, grant_file=linked_grant))
    linked_parent.unlink()

    dump(
        scope / "synthetic_binding.json",
        {"test_only": True, "scope": str(scope), "request_sha256": request["authorization_request_sha256"]},
    )

    first = subprocess.run(command(FAILED_CELL_ID), cwd=ROOT, env=environment, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    first_inventory = inventory(recovery_root)
    assert len(rows(scope / "dispatch.jsonl")) == 1
    assert [row["attempt"] for row in rows(recovery_root / "cell_state.jsonl") if row["status"] == "committed"] == [2]
    second = subprocess.run(command(UNSTARTED_CELL_ID), cwd=ROOT, env=environment, text=True, capture_output=True)
    assert second.returncode == 0, second.stderr
    assert len(rows(scope / "dispatch.jsonl")) == 2
    committed = [row for row in rows(recovery_root / "cell_state.jsonl") if row["status"] == "committed"]
    assert [(row["cell_id"], row["attempt"]) for row in committed] == [
        (FAILED_CELL_ID, 2),
        (UNSTARTED_CELL_ID, 1),
    ]
    assert all(path in inventory(recovery_root) for path in first_inventory)
    handoff = json.loads((recovery_root / "ablation_recovery_handoff.json").read_text())
    assert handoff["source_partitions"] == {"external_original_run": 6, "new_recovery_execution": 2}
    assert handoff["unique_logical_cell_count"] == 8
    assert handoff["next_stage_authorized"] is False
    assert handoff["statistics_consumer_compatibility"]["formal_controller_row_count"] == 3
    assert (recovery_root / "unsigned_followup_request.json").is_file()
    rejected(command(FAILED_CELL_ID))
    rejected(command(UNSTARTED_CELL_ID))
    contract_copy = recovery_root / "restricted_recovery_execution_contract.json"
    changed_contract = json.loads(contract_copy.read_text())
    changed_contract["allowed_phase"] = "formal_support"
    dump(contract_copy, changed_contract)
    rejected(command(UNSTARTED_CELL_ID))
    assert old_snapshot["phase_state.jsonl"] == file_sha256(ORIGINAL_RUN_ROOT / "phase_state.jsonl")
    assert old_snapshot["cell_state.jsonl"] == file_sha256(ORIGINAL_RUN_ROOT / "cell_state.jsonl")
    assert old_snapshot["held_lock"] == audit_original_recovery_source()["held_lock"]["sha256"]
    assert all(row["real_model_open_count"] == 0 for row in rows(scope / "dispatch.jsonl"))
    evidence = os.environ.get("G14R20_I5_EVIDENCE_ROOT")
    if evidence:
        evidence_path = Path(evidence)
        evidence_path.mkdir(parents=True, exist_ok=True)
        dump(
            evidence_path / "public_recovery_acceptance.json",
            {
                "status": "pass",
                "public_process_count": len(list(scope.glob("host_*.json"))),
                "successful_cell_process_count": len(
                    {row["pid"] for row in rows(scope / "dispatch.jsonl")}
                ),
                "synthetic_dispatch_count": len(rows(scope / "dispatch.jsonl")),
                "real_model_open_count": 0,
                "scientific_rollout_count": 0,
                "recovery_committed_attempts": [2, 1],
                "negative_case_count": 17,
                "negative_dispatch_count": 0,
                "negative_recovery_write_count": 0,
                "external_original_cell_count": 6,
                "new_recovery_cell_count": 2,
                "unique_handoff_cell_count": 8,
                "old_phase_ledger_unchanged": True,
                "old_cell_ledger_unchanged": True,
                "old_held_lock_unchanged": True,
                "next_stage_authorized": False,
                "recovery_grant_issued": False,
                "real_recovery_started": False,
                "holdout_opened": False,
            },
        )
