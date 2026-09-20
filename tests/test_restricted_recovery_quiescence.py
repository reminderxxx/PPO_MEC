"""I5-D stable request identity and independent live quiescence boundaries."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.continuation_executor.identity import digest
from src.runtime.evaluation_only_execution import canonical_sha256, file_sha256
from src.runtime import restricted_recovery as recovery
from scripts import run_typed_model_cache_restricted_recovery as runner
from tests.test_restricted_recovery import minimal_request


def _rehash(request):
    request["authorization_request_sha256"] = canonical_sha256(
        {key: value for key, value in request.items() if key != "authorization_request_sha256"}
    )


def _live_validation(monkeypatch, frozen, rebuilt):
    monkeypatch.setattr(recovery, "build_restricted_recovery_request", lambda **_: rebuilt)
    return recovery.validate_restricted_recovery_request(
        frozen, check_live=True,
        _test_recovery_parent=Path(frozen["recovery_execution"]["recovery_root"]).parent,
    )


def _request_with_observation(tmp_path):
    frozen = minimal_request(tmp_path)
    frozen["immutable_source_audit"] = {
        "held_lock": {
            "path": "/old/lock", "device": 1, "inode": 2,
            "size_bytes": 3, "sha256": "a" * 64,
            "owner_canonical_sha256": "b" * 64, "owner_state": "held",
            "pid_observation": {
                "pid": 42, "ps_return_code": 0,
                "live_start_time": "Sep 1", "probe_error": None,
                "quiescence_proven": False, "lock_cleanup_authorized": False,
            },
        },
        "phase_ledger_prefix": {"prefix_sha256": "c" * 64},
        "cell_ledger_prefix": {"prefix_sha256": "d" * 64},
        "failed_cell": {"staging_files": [{"sha256": "e" * 64}]},
        "i4_review_inputs": {"declared_audit_bundle": {
            "path": "/old/audit", "present": True, "used_as_evidence": False,
        }},
    }
    _rehash(frozen)
    return frozen


def test_complete_frozen_hash_checked_before_projection(tmp_path, monkeypatch):
    frozen = _request_with_observation(tmp_path)
    frozen["immutable_source_audit"]["held_lock"]["pid_observation"]["ps_return_code"] = 1
    with pytest.raises(recovery.RestrictedRecoveryError, match="request hash mismatch"):
        _live_validation(monkeypatch, frozen, deepcopy(frozen))


@pytest.mark.parametrize("change", [
    lambda r: r["source_models"][0].update(agent="drift"),
    lambda r: r["external_committed_cells"][0].update(cell_id="drift"),
    lambda r: r["immutable_source_audit"]["phase_ledger_prefix"].update(prefix_sha256="0" * 64),
    lambda r: r["immutable_source_audit"]["cell_ledger_prefix"].update(prefix_sha256="0" * 64),
    lambda r: r["immutable_source_audit"]["held_lock"].update(sha256="0" * 64),
    lambda r: r["immutable_source_audit"]["held_lock"].update(inode=99),
    lambda r: r["immutable_source_audit"]["held_lock"].update(owner_canonical_sha256="0" * 64),
    lambda r: r["immutable_source_audit"]["failed_cell"]["staging_files"][0].update(sha256="0" * 64),
    lambda r: r["recovery_execution"].update(executor_commit="0" * 40),
    lambda r: r["recovery_execution"]["command_plan"].update(commands=[]),
    lambda r: r["immutable_source_audit"].update(unknown_new_field="drift"),
])
def test_scientific_and_unknown_live_drift_rejected(tmp_path, monkeypatch, change):
    frozen = _request_with_observation(tmp_path)
    rebuilt = deepcopy(frozen)
    change(rebuilt)
    with pytest.raises(recovery.RestrictedRecoveryError, match="differs from live immutable sources"):
        _live_validation(monkeypatch, frozen, rebuilt)
    assert not Path(frozen["recovery_execution"]["recovery_root"]).exists()


@pytest.mark.parametrize("observation", [
    {"ps_return_code": 1, "live_start_time": None},
    {"probe_error": "PermissionError: denied"},
    {"pid": 42, "ps_return_code": 1, "live_start_time": None},
    {"i4_bundle_present": False},
])
def test_only_live_pid_observation_changes_identity_neither_hash_nor_sources(
    tmp_path, monkeypatch, observation,
):
    frozen = _request_with_observation(tmp_path)
    rebuilt = deepcopy(frozen)
    if "i4_bundle_present" in observation:
        rebuilt["immutable_source_audit"]["i4_review_inputs"]["declared_audit_bundle"]["present"] = observation["i4_bundle_present"]
    else:
        rebuilt["immutable_source_audit"]["held_lock"]["pid_observation"].update(observation)
    _rehash(rebuilt)
    assert _live_validation(monkeypatch, frozen, rebuilt)["status"] == "pass"
    assert recovery.stable_recovery_request_projection(frozen) == recovery.stable_recovery_request_projection(rebuilt)


def _lock_request(tmp_path):
    lock_path = tmp_path / "old_held.lock"
    owner = {"version": "1.0.0", "state": "held", "process": {
        "host": "test", "pid": 999999, "started": "Mon Sep  1 00:00:00 2025",
    }}
    lock_path.write_text(json.dumps(owner), encoding="utf-8")
    row = lock_path.stat()
    request = _request_with_observation(tmp_path)
    request["immutable_source_audit"]["held_lock"] = {
        "path": str(lock_path), "device": row.st_dev, "inode": row.st_ino,
        "size_bytes": row.st_size, "sha256": file_sha256(lock_path),
        "owner_canonical_sha256": digest(owner), "owner_state": "held",
    }
    _rehash(request)
    return request, lock_path


def _ps_absent(_pid):
    return subprocess.CompletedProcess([], 1, "", "")


def test_ps_permission_unavailable_only_fails_qualification(tmp_path, monkeypatch):
    request, lock_path = _lock_request(tmp_path)
    assert _live_validation(monkeypatch, request, deepcopy(request))["status"] == "pass"

    def denied(_pid):
        raise PermissionError("ps denied")

    evidence = recovery.observe_restricted_recovery_quiescence(request, process_probe=denied)
    assert evidence["status"] == "fail"
    assert evidence["failure_codes"] == ["process_permission_failure"]
    assert evidence["process_identity_probe"]["ps_permission_available"] is False
    assert evidence["lock_bytes_unchanged"] is True
    assert file_sha256(lock_path) == request["immutable_source_audit"]["held_lock"]["sha256"]


def test_kernel_lock_occupied_fails_without_writing(tmp_path):
    request, lock_path = _lock_request(tmp_path)
    child = subprocess.Popen([sys.executable, "-c", "import fcntl,os,sys,time; f=os.open(sys.argv[1],os.O_RDONLY); fcntl.flock(f,fcntl.LOCK_EX); print('locked',flush=True); time.sleep(15)", str(lock_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "locked"
        evidence = recovery.observe_restricted_recovery_quiescence(request, process_probe=_ps_absent)
        assert evidence["status"] == "fail"
        assert "kernel_lock_failure" in evidence["failure_codes"]
        assert evidence["kernel_lock_probe"]["result"] == "occupied"
        assert evidence["lock_bytes_unchanged"] is True
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_free_kernel_lock_passes_and_evidence_expires(tmp_path):
    request, lock_path = _lock_request(tmp_path)
    before = (lock_path.stat().st_ino, file_sha256(lock_path))
    now = datetime.now(timezone.utc)
    evidence = recovery.observe_restricted_recovery_quiescence(request, now=now, process_probe=_ps_absent)
    assert evidence["status"] == "pass"
    recovery.validate_restricted_recovery_quiescence_evidence(evidence, request, now=now)
    with pytest.raises(recovery.RestrictedRecoveryError, match="expired"):
        recovery.validate_restricted_recovery_quiescence_evidence(
            evidence, request, now=now + timedelta(seconds=301),
        )
    assert (lock_path.stat().st_ino, file_sha256(lock_path)) == before
    assert evidence["lock_cleanup_authorized"] is False
    assert evidence["holdout_capability"] is False


@pytest.mark.parametrize("change", ["bytes", "inode"])
def test_changed_lock_fails_both_layers(tmp_path, monkeypatch, change):
    request, lock_path = _lock_request(tmp_path)
    if change == "bytes":
        lock_path.write_text(lock_path.read_text() + " ", encoding="utf-8")
    else:
        replacement = tmp_path / "replacement"
        replacement.write_bytes(lock_path.read_bytes())
        os.replace(replacement, lock_path)
    rebuilt = deepcopy(request)
    row = lock_path.stat()
    rebuilt["immutable_source_audit"]["held_lock"].update(
        inode=row.st_ino, sha256=file_sha256(lock_path), size_bytes=row.st_size,
    )
    _rehash(rebuilt)
    with pytest.raises(recovery.RestrictedRecoveryError, match="differs from live immutable sources"):
        _live_validation(monkeypatch, request, rebuilt)
    evidence = recovery.observe_restricted_recovery_quiescence(request, process_probe=_ps_absent)
    assert evidence["status"] == "fail"
    assert "immutable_lock_changed" in evidence["failure_codes"]


def test_projection_is_shared_at_public_validate_qualify_execute_sites():
    from pathlib import Path
    source = (Path(recovery.__file__).parents[2] / "scripts/run_typed_model_cache_restricted_recovery.py").read_text()
    assert "stable_recovery_request_projection(live_source_request)" in source
    assert "validate_recovery_executor_live(\n        request" in source
    assert "validate_restricted_recovery_request(\n        request" in source
    assert "stable_recovery_request_projection(request) != stable_recovery_request_projection(rebuilt)" in Path(recovery.__file__).read_text()


def test_review_and_grant_bind_fresh_evidence_bytes(tmp_path, monkeypatch):
    request, _ = _lock_request(tmp_path)
    now = datetime.now(timezone.utc)
    evidence = recovery.observe_restricted_recovery_quiescence(request, now=now, process_probe=_ps_absent)
    evidence_path = tmp_path / "qualification.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    ref = {"path": str(evidence_path), "size_bytes": evidence_path.stat().st_size,
           "sha256": file_sha256(evidence_path)}
    execution = request["recovery_execution"]
    review = {
        "status": "pass", "reviewer_id": "independent", "implementation_agent_id": "implementer",
        "authorization_request_sha256": request["authorization_request_sha256"],
        "recovery_execution_identity_sha256": execution["recovery_execution_identity_sha256"],
        "holdout_capability": False, "synthetic_only": False,
        "pregrant_qualification": ref,
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    grant = {
        "version": "1.0.0", "status": "AUTHORIZED_FOR_RESTRICTED_RECOVERY",
        "authorization_request_sha256": request["authorization_request_sha256"],
        "recovery_execution_id": execution["recovery_execution_id"],
        "original_run_id": execution["original_run_id"],
        "executor_commit": execution["executor_commit"],
        "executor_git_tree": execution["executor_git_tree"],
        "allowed_phase": recovery.PHASE, "allowed_cell_ids": list(recovery.ALLOWED_CELL_IDS),
        "holdout_capability": False, "later_phases_authorized": False,
        "pregrant_qualification": ref,
        "independent_review": {"path": str(review_path), "size_bytes": review_path.stat().st_size,
                               "sha256": file_sha256(review_path)},
        "issued_at": (now - timedelta(seconds=1)).isoformat(),
        "expires_at": (now + timedelta(minutes=1)).isoformat(),
    }
    monkeypatch.setattr(runner, "observe_restricted_recovery_quiescence", lambda *a, **k: evidence)
    assert runner.verify_recovery_grant(request, grant, now=now)["approval_verified"]
    wrong = deepcopy(grant)
    wrong["pregrant_qualification"] = {**ref, "sha256": "0" * 64}
    with pytest.raises(recovery.RestrictedRecoveryError, match="binding drift"):
        runner.verify_recovery_grant(request, wrong, now=now)
    evidence_path.write_text(evidence_path.read_text() + " ")
    with pytest.raises(recovery.RestrictedRecoveryError, match="bytes drift"):
        runner.verify_recovery_grant(request, grant, now=now)
