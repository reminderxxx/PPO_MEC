"""Two-role, real-process acceptance for the public startup parser/main flow."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from . import PHASES
from .authorization import validate_contract
from .identity import canonical, digest, file_hash, git
from .startup import atomic_publish_receipt, verify_handoff_for_custody
from .test_trust_fixture import TestTrustFixture, signed, write


ROLES = ('protocol', 'bundle', 'environment', 'binding', 'context',
         'command_matrix', 'candidate', 'selection', 'freeze', 'registry',
         'checkpoint', 'resource', 'committed_output')


def _ledger(path, kind):
    fingerprint = "1" * 64
    key = "current_record_hash" if kind == "phase" else "current_ledger_hash"
    previous = "previous_record_hash" if kind == "phase" else "previous_ledger_hash"
    body = {"sequence_number": 1, "run_identity_fingerprint": fingerprint,
            "phase": "checkpoint_freeze", "status": "completed" if kind == "phase" else "committed",
            previous: None}
    body[key] = digest(body)
    path.write_bytes(canonical(body) + b"\n")
    raw = path.read_bytes()
    return {"path": str(path), "kind": kind, "record_count": 1,
            "byte_count": len(raw), "prefix_sha256": hashlib.sha256(raw).hexdigest(),
            "terminal_hash": body[key], "run_identity_fingerprint": fingerprint}


def _proposal(root):
    run = root / "synthetic_public_run"
    run.mkdir()
    ledgers = [_ledger(run / "phase_state.jsonl", "phase"),
               _ledger(run / "cell_state.jsonl", "cell")]
    evidence = []
    evidence_root = root / "evidence"
    evidence_root.mkdir()
    for role in ROLES:
        path = evidence_root / (role + ".json")
        write(path, {"version": "1.0.0", "test_only": True, "role": role})
        evidence.append({"path": str(path), "sha256": file_hash(path),
                         "size_bytes": path.stat().st_size, "role": role})
    source = Path.cwd()
    proposal = {"version": "1.0.0", "state": "proposal", "execution_authorized": False,
        "holdout_capability": False, "run_id": run.name, "run_root": str(run),
        "worktree_root": str(source),
        "execution_commit": "a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d",
        "source_tree": {"git_tree": git(source, "rev-parse", "HEAD^{tree}"),
                        "tracked_sources_sha256": "2" * 64},
        "evidence": evidence, "ledgers": ledgers, "allowed_phases": list(PHASES),
        "write_scope": {"root": str(run), "mode": "approved_append_only_successors",
                        "immutable_prefix": True, "currently_writable": False},
        "origin_evidence": {name: {"status": "unavailable", "path": None, "sha256": None,
            "boundary": "not independently verified"} for name in ("launch_approval", "release_attestation")},
        "executor": {"state": "pending", "commit": None, "files": []},
        "approval": {"state": "pending", "source": None,
                     "scope": "implementation_and_read_only_acceptance_only"},
        "revocation_rules": ["expired", "revoked", "identity_drift", "failed_terminal",
                             "prefix_changed", "unauthorized_successor"]}
    return proposal


def prepare(root, identity):
    root.mkdir()
    proposal = _proposal(root)
    proposal_path = root / "proposal.json"
    write(proposal_path, proposal)
    contract = {"version": "1.0.0", "domain": "synthetic",
        "proposal_sha256": digest(proposal), "proposal_file_sha256": file_hash(proposal_path),
        "executor_identity_sha256": digest(identity), "run_id": proposal["run_id"],
        "run_root": proposal["run_root"], "phases": list(PHASES), "holdout_capability": False,
        "prefixes": proposal["ledgers"], "immutable_files": [], "fixture_root": str(root),
        "expires_at": (datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
        "revocation_id": "synthetic-public-startup", "recovery_owner_sha256": None,
        "recovery_quiescence": None, "coordination_root": str(root/".continuation_locks"),
        "command_plan_sha256": digest({})}
    trust = TestTrustFixture(contract)
    validate_contract(contract, proposal, identity)
    for name, value in (("contract.json", contract), ("approval.json", trust.approval),
                        ("executor_identity.json", identity)):
        write(root/name, value)
    return trust, {"proposal": proposal_path, "contract": root/"contract.json",
                   "approval": root/"approval.json", "identity": root/"executor_identity.json",
                   "installation": root/"trust_installation_test_only.json"}


def _command(entry, paths, *, check="execute", wait=5.0):
    return [sys.executable, "-B", str(entry), "--internal-public-startup-host",
        "--executor-identity", str(paths["identity"]), "--proposal", str(paths["proposal"]),
        "--contract", str(paths["contract"]), "--approval", str(paths["approval"]),
        "--installation", str(paths["installation"]), "--phase", PHASES[0],
        "--check", check, "--startup-wait-seconds", str(wait)]


def _host_environment():
    import cryptography
    dependencies = str(Path(cryptography.__file__).resolve().parent.parent)
    return dict(os.environ, PYTHONPATH=dependencies, PYTHONNOUSERSITE="1",
                PYTHONDONTWRITEBYTECODE="1")


def _next_event(process):
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("public host exited before expected event: " + process.stderr.read())
    return json.loads(line)


def start_and_challenge(entry, paths, *, check="execute", wait=5.0):
    process = subprocess.Popen(_command(entry, paths, check=check, wait=wait), cwd=Path.cwd(),
        env=_host_environment(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    events = []
    while True:
        row = _next_event(process)
        events.append(row)
        if row.get("event") == "waiting":
            challenge = next(x for x in events if x.get("event") == "challenge")
            return process, events, challenge["request"]
        if row.get("event") == "terminal":
            return process, events, None


def finish(process, events):
    for line in process.stdout:
        events.append(json.loads(line))
    stderr = process.stderr.read()
    code = process.wait()
    return code, events, stderr


def run_public_startup_acceptance(root, identity, entry):
    trust, paths = prepare(Path(root), identity)
    prior = trust.initial
    custodian_pid = os.getpid()

    host, events, challenge = start_and_challenge(entry, paths)
    trust.publish_receipt(challenge, digest(prior))
    code, events, stderr = finish(host, events)
    if code or stderr or events[-1].get("status") != "completed":
        raise RuntimeError("public startup positive failed: " + stderr + repr(events[-1:]))
    terminal = events[-1]
    custody = verify_handoff_for_custody(trust.pin, prior, challenge,
                                         terminal["handoff_material"])

    negative_receipts = {}
    for case in ("schema", "signature", "authority", "checkpoint", "nonce", "pid", "expired"):
        candidate, candidate_events, candidate_challenge = start_and_challenge(
            entry, paths, check="qualification")
        message = {**candidate_challenge, "authority": "revoker",
            "checkpoint_sha256": custody["checkpoint_sha256"],
            "issued_at": trust.now.isoformat(),
            "expires_at": (trust.now+timedelta(hours=1)).isoformat()}
        if case == "schema": message["extra"] = True
        if case == "authority": message["authority"] = "other"
        if case == "checkpoint": message["checkpoint_sha256"] = "0"*64
        if case == "nonce": message["session_nonce"] = "wrong"
        if case == "pid": message["process_id"] += 1
        if case == "expired": message["expires_at"] = trust.now.isoformat()
        receipt = signed(message, "revoker", trust.keys["revoker"])
        if case == "signature": receipt["signature"] = "00"*64
        before_state = Path(trust.pin["continuity_path"]).read_bytes()
        atomic_publish_receipt(trust.pin["startup_receipt_path"], receipt)
        candidate_code, candidate_events, candidate_stderr = finish(candidate, candidate_events)
        if (candidate_code != 2 or candidate_stderr
                or candidate_events[-1].get("status") != "rejected"
                or Path(trust.pin["continuity_path"]).read_bytes() != before_state):
            raise RuntimeError("public invalid receipt case failed closed: " + case)
        negative_receipts[case] = candidate_events

    # A new process cannot consume the old receipt: it waits on a new challenge.
    stale, stale_events, stale_challenge = start_and_challenge(entry, paths, check="qualification", wait=.15)
    stale_code, stale_events, stale_stderr = finish(stale, stale_events)
    if stale_challenge is None or stale_code != 124 or stale_events[-1].get("reason_code") != "STARTUP_RECEIPT_TIMEOUT":
        raise RuntimeError("independent command reused an old receipt")

    interrupted, interrupted_events, interrupted_challenge = start_and_challenge(
        entry, paths, check="qualification")
    interrupted.terminate()
    interrupted_code, interrupted_events, interrupted_stderr = finish(interrupted, interrupted_events)
    if (interrupted_code != 128 + signal.SIGTERM or interrupted_stderr
            or interrupted_events[-1].get("reason_code") != "STARTUP_INTERRUPTED"):
        raise RuntimeError("SIGTERM did not produce a bounded interruption")

    crashed, crashed_events, crashed_challenge = start_and_challenge(
        entry, paths, check="qualification")
    crashed.kill()
    crashed_code, crashed_events, crashed_stderr = finish(crashed, crashed_events)
    if crashed_code == 0 or crashed_stderr:
        raise RuntimeError("crash boundary was not observed")

    # First host owns the fixed startup lock; a second host fails without touching it.
    first, first_events, first_challenge = start_and_challenge(entry, paths, check="qualification")
    second = subprocess.run(_command(entry, paths, check="qualification", wait=.2), cwd=Path.cwd(),
        env=_host_environment(), text=True, capture_output=True)
    second_events = [json.loads(line) for line in second.stdout.splitlines()]
    if second.returncode != 2 or second_events[-1].get("reason_code") != "STARTUP_HOST_BUSY":
        raise RuntimeError("concurrent startup host did not fail closed")
    trust.publish_receipt(first_challenge, custody["checkpoint_sha256"])
    first_code, first_events, first_stderr = finish(first, first_events)
    if first_code or first_stderr:
        raise RuntimeError("first concurrent host was disturbed")
    custody2 = verify_handoff_for_custody(trust.pin, custody["state"], first_challenge,
                                          first_events[-1]["handoff_material"])

    # After release, a third fresh process gets a fresh receipt and succeeds.
    third, third_events, third_challenge = start_and_challenge(entry, paths, check="qualification")
    trust.publish_receipt(third_challenge, custody2["checkpoint_sha256"])
    third_code, third_events, third_stderr = finish(third, third_events)
    if third_code or third_stderr:
        raise RuntimeError("fresh post-release startup failed")
    custody3 = verify_handoff_for_custody(trust.pin, custody2["state"], third_challenge,
                                          third_events[-1]["handoff_material"])
    requests = [challenge, stale_challenge, interrupted_challenge, crashed_challenge,
                first_challenge, third_challenge]
    if len({row["session_nonce"] for row in requests}) != len(requests):
        raise RuntimeError("startup nonce reused")
    return {"status": "pass", "public_parser_main": True,
        "host_pid": challenge["process_id"], "custodian_pid": custodian_pid,
        "roles_are_separate_processes": challenge["process_id"] != custodian_pid,
        "positive_events": events, "old_receipt_events": stale_events,
        "invalid_receipt_events": negative_receipts,
        "interrupted_events": interrupted_events, "crashed_events": crashed_events,
        "crash_exit_code": crashed_code,
        "concurrent_second_events": second_events, "concurrent_first_events": first_events,
        "post_release_events": third_events,
        "custody_checkpoints": [custody, custody2, custody3],
        "synthetic_dispatch_count": 1, "scientific_rollout_count": 0,
        "real_v16_dispatch_count": 0, "real_v16_write_count": 0,
        "holdout_consumption_count": 0}
