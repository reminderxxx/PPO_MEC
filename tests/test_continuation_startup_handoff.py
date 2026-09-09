"""Startup host/custodian boundaries; all trust is synthetic and test-only."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path

import pytest

from scripts.continuation_executor import PHASES
from scripts.continuation_executor.authorization import verify_approval
from scripts.continuation_executor.identity import ContinuationError, digest, read_json
from scripts.continuation_executor.production_trust import TrustContext
from scripts.continuation_executor.startup import (
    StartupCoordinator, StartupTimeout, atomic_publish_receipt,
    run_startup_operation, verify_handoff_for_custody,
)
from scripts.continuation_executor.test_trust_fixture import TestTrustFixture, signed


@pytest.fixture
def trust(tmp_path):
    run = tmp_path/"synthetic_startup"; run.mkdir()
    proposal = {"run_id": run.name, "run_root": str(run), "ledgers": []}
    contract = {"version":"1.0.0", "domain":"synthetic",
        "proposal_sha256":digest(proposal), "proposal_file_sha256":digest(proposal),
        "executor_identity_sha256":digest({}), "run_id":run.name, "run_root":str(run),
        "phases":list(PHASES), "holdout_capability":False, "prefixes":[],
        "immutable_files":[], "fixture_root":str(tmp_path),
        "expires_at":(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
        "revocation_id":"startup-test", "recovery_owner_sha256":None,
        "recovery_quiescence":None, "coordination_root":str(tmp_path/".continuation_locks"),
        "command_plan_sha256":digest({})}
    return TestTrustFixture(contract)


def _receipt(trust, context, checkpoint, **changes):
    message = {**context.startup_request(), "authority":"revoker",
        "checkpoint_sha256":checkpoint, "issued_at":trust.now.isoformat(),
        "expires_at":(trust.now+timedelta(hours=1)).isoformat()}
    message.update(changes)
    return signed(message, "revoker", trust.keys["revoker"])


def test_same_context_wait_qualification_execute_and_public_custody(trust):
    previous = deepcopy(trust.initial)
    context = TrustContext(trust.pin, test_only=True)
    requests = []
    callback_contexts = []

    def emit(row):
        if row["event"] == "challenge":
            requests.append(row["request"])
        if row["event"] == "waiting":
            trust.publish_receipt(requests[0], digest(previous))

    def verify(now):
        callback_contexts.append(context.startup_request())
        return verify_approval(trust.contract, trust.approval,
                               test_trust_context=context, now=now)

    def operate(grant):
        assert grant["approval_verified"]
        return verify(datetime.now(timezone.utc))

    authorization, result, material = run_startup_operation(
        context, verify, operate, timeout_seconds=.5, emit=emit)
    assert authorization["approval_verified"] and result["approval_verified"]
    assert callback_contexts == [requests[0], requests[0]]
    custody = verify_handoff_for_custody(trust.pin, previous, requests[0], material)
    assert custody["status"] == "custody_verified"
    assert custody["checkpoint_sha256"] == digest(custody["state"])


def test_missing_receipt_times_out_without_continuity_write(trust):
    before = Path(trust.pin["continuity_path"]).read_bytes()
    context = TrustContext(trust.pin, test_only=True)
    with pytest.raises(StartupTimeout):
        run_startup_operation(context, lambda now: None, lambda grant: None,
                              timeout_seconds=.05)
    assert Path(trust.pin["continuity_path"]).read_bytes() == before
    assert context.expected is None


@pytest.mark.parametrize("case", ["schema", "signature", "authority", "expired",
    "nonce", "pid", "checkpoint"])
def test_one_new_invalid_receipt_rejects_without_retry_or_new_context(trust, case):
    context = TrustContext(trust.pin, test_only=True)
    request = context.startup_request()

    def emit(row):
        if row["event"] != "waiting":
            return
        receipt = _receipt(trust, context, digest(trust.initial))
        if case == "schema": receipt["message"]["extra"] = True
        if case == "signature": receipt["signature"] = "00"*64
        if case == "authority": receipt["message"]["authority"] = "other"
        if case == "expired": receipt["message"]["expires_at"] = trust.now.isoformat()
        if case == "nonce": receipt["message"]["session_nonce"] = "wrong"
        if case == "pid": receipt["message"]["process_id"] = os.getpid()+1
        if case == "checkpoint": receipt["message"]["checkpoint_sha256"] = "0"*64
        if case not in {"signature"}:
            receipt = signed(receipt["message"], "revoker", trust.keys["revoker"])
        atomic_publish_receipt(trust.pin["startup_receipt_path"], receipt)

    with pytest.raises((ContinuationError, OSError)):
        run_startup_operation(context,
            lambda now: verify_approval(trust.contract, trust.approval,
                test_trust_context=context, now=now), lambda grant: None,
            timeout_seconds=.5, emit=emit)
    assert context.startup_request() == request


def test_fixed_startup_lock_is_noncreating_noninterfering(trust):
    inode = Path(trust.pin["startup_lock_path"]).stat().st_ino
    first = TrustContext(trust.pin, test_only=True)
    second = TrustContext(trust.pin, test_only=True)
    with StartupCoordinator(first):
        with pytest.raises(ContinuationError, match="already active"):
            with StartupCoordinator(second):
                pass
    assert Path(trust.pin["startup_lock_path"]).stat().st_ino == inode


@pytest.mark.parametrize("case", ["host_hash", "challenge", "local_state", "previous_state"])
def test_custodian_does_not_promote_host_report_or_rollback(trust, case):
    previous = deepcopy(trust.initial)
    context = TrustContext(trust.pin, test_only=True)
    request = context.startup_request()

    def emit(row):
        if row["event"] == "waiting":
            trust.publish_receipt(request, digest(previous))

    _, _, material = run_startup_operation(context,
        lambda now: verify_approval(trust.contract, trust.approval,
            test_trust_context=context, now=now), lambda grant: None,
        timeout_seconds=.5, emit=emit)
    if case == "host_hash": material["checkpoint_sha256"] = "f"*64
    if case == "challenge": material["request"]["session_nonce"] = "other"
    if case == "local_state":
        state = read_json(trust.state); state["sequence"] += 1
        trust.state.write_text(__import__('json').dumps(state))
    if case == "previous_state": previous["sequence"] = 10
    with pytest.raises(ContinuationError):
        verify_handoff_for_custody(trust.pin, previous, request, material)
