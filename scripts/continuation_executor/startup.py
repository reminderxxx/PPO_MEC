"""Bounded public startup handoff and independent continuity custody."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import math
import os
import signal
import time

from .identity import ContinuationError, absolute_path, canonical, digest, read_json
from .production_trust import require, text, utc, verify_signature

EVENT_VERSION = "1.0.0"
DEFAULT_WAIT_SECONDS = 30.0
MAX_WAIT_SECONDS = 300.0
POLL_SECONDS = 0.05


class StartupTimeout(ContinuationError):
    pass


class StartupInterrupted(ContinuationError):
    def __init__(self, signum):
        super().__init__("startup interrupted by signal " + str(signum))
        self.signum = signum


def validate_wait_seconds(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContinuationError("startup wait must be a finite number") from exc
    if not math.isfinite(result) or not 0.05 <= result <= MAX_WAIT_SECONDS:
        raise ContinuationError("startup wait outside installed finite bounds")
    return result


def event(kind, status, **fields):
    return {"version": EVENT_VERSION, "event": kind, "status": status, **fields}


def rejection_code(exc):
    message = str(exc)
    if isinstance(exc, StartupTimeout):
        return "STARTUP_RECEIPT_TIMEOUT"
    if isinstance(exc, (StartupInterrupted, KeyboardInterrupt)):
        return "STARTUP_INTERRUPTED"
    if message == "startup host already active":
        return "STARTUP_HOST_BUSY"
    if "startup receipt" in message or "startup challenge" in message or "Ed25519" in message:
        return "STARTUP_RECEIPT_REJECTED"
    if "continuity" in message or "revocation" in message or "approval" in message or "signer" in message:
        return "AUTHORIZATION_REJECTED"
    return "STATIC_OR_OPERATION_REJECTED"


def receipt_identity(path):
    path = absolute_path(path)
    try:
        stat = path.stat()
        if not path.is_file():
            return {"state": "non_file"}
        raw = path.read_bytes()
        return {"state": "present", "inode": stat.st_ino, "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()}
    except FileNotFoundError:
        return {"state": "missing"}


class StartupCoordinator:
    """Installation-prepared fixed-inode lock held for the public host lifetime."""
    def __init__(self, context):
        self.context = context
        self.path = absolute_path(context.pin["startup_lock_path"])
        self.fd = None

    def __enter__(self):
        require(self.path.parent == self.context.state_path.parent,
                "startup lock outside coordination root")
        self.fd = os.open(self.path, os.O_RDWR | os.O_NOFOLLOW)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self.fd)
            self.fd = None
            raise ContinuationError("startup host already active") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        os.close(self.fd)
        self.fd = None


@contextmanager
def interrupt_boundary():
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        del frame
        raise StartupInterrupted(signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def wait_for_new_receipt(context, *, timeout_seconds, emit):
    timeout_seconds = validate_wait_seconds(timeout_seconds)
    before = receipt_identity(str(context.receipt_path))
    emit(event("waiting", "legacy_receipt_ignored" if before["state"] == "present" else "receipt_pending",
               receipt_path=str(context.receipt_path), initial_receipt=before,
               timeout_seconds=timeout_seconds, clock="monotonic"))
    deadline = time.monotonic() + timeout_seconds
    while True:
        current = receipt_identity(str(context.receipt_path))
        if current != before:
            if current["state"] != "present":
                raise ContinuationError("new startup receipt is not a regular file")
            return current
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise StartupTimeout("startup receipt wait timed out")
        time.sleep(min(POLL_SECONDS, remaining))


def handoff_material(context, outcome, *, reported_at=None):
    return {"version": EVENT_VERSION, "trust": "unverified_host_report",
            "request": context.startup_request(), "outcome": outcome,
            "checkpoint_sha256": context.expected,
            "reported_at": (reported_at or datetime.now(timezone.utc)).isoformat(),
            "custodian_verification_required": True}


def run_startup_operation(context, verify, operate, *, timeout_seconds=DEFAULT_WAIT_SECONDS,
                          emit=lambda value: None):
    request = context.startup_request()
    with StartupCoordinator(context), interrupt_boundary():
        emit(event("challenge", "published", request=request,
                   receipt_path=str(context.receipt_path)))
        wait_for_new_receipt(context, timeout_seconds=timeout_seconds, emit=emit)
        authorization = verify(datetime.now(timezone.utc))
        emit(event("qualification", "accepted", request=request,
                   authorization=authorization))
        result = operate(authorization)
        material = handoff_material(context, "completed")
        emit(event("handoff", "verification_required", material=material))
        return authorization, result, material


def _validate_state_schema(state, pin):
    require(isinstance(state, dict) and set(state) == {
        "installation_id", "scope_sha256", "sequence", "content_sha256",
        "issued_at", "observed_at", "revoked_ids", "untrusted_signers"},
        "continuity schema/scope")
    require(state["installation_id"] == pin["installation_id"]
            and state["scope_sha256"] == digest(pin["scope"]),
            "continuity schema/scope")
    require(type(state["sequence"]) is int and state["sequence"] >= -1,
            "continuity sequence")
    for name in ("revoked_ids", "untrusted_signers"):
        require(isinstance(state[name], list) and all(text(x) for x in state[name])
                and len(set(state[name])) == len(state[name]), "continuity identifiers")


def verify_handoff_for_custody(pin, previous_state, challenge, material, *, now=None):
    """Independently recompute a checkpoint; never trust the host-reported hash."""
    now = now or datetime.now(timezone.utc)
    require(now.tzinfo is not None, "untrusted naive time")
    require(isinstance(challenge, dict) and set(challenge) == {
        "version", "installation_id", "scope_sha256", "session_nonce", "process_id"},
        "startup challenge schema")
    require(material.get("version") == EVENT_VERSION
            and material.get("trust") == "unverified_host_report"
            and material.get("request") == challenge
            and material.get("custodian_verification_required") is True,
            "handoff material binding")
    require(challenge["installation_id"] == pin["installation_id"]
            and challenge["scope_sha256"] == digest(pin["scope"])
            and text(challenge["session_nonce"])
            and type(challenge["process_id"]) is int and challenge["process_id"] > 0,
            "startup challenge identity")
    _validate_state_schema(previous_state, pin)
    current = read_json(absolute_path(pin["continuity_path"]))
    _validate_state_schema(current, pin)
    revocation = pin["revocation"]
    message = verify_signature(read_json(absolute_path(revocation["path"])),
                               {revocation["authority"]: revocation["public_key"]})
    require(isinstance(message, dict) and set(message) == {
        "version", "installation_id", "authority", "source_id", "scope",
        "sequence", "issued_at", "expires_at", "revoked_ids", "untrusted_signers"},
        "revocation schema")
    require(message["version"] == "2.0.0"
            and message["installation_id"] == pin["installation_id"]
            and message["authority"] == revocation["authority"]
            and message["source_id"] == revocation["source_id"]
            and message["scope"] == pin["scope"], "revocation identity/scope mismatch")
    issued = utc(message["issued_at"])
    require(issued <= now < utc(message["expires_at"])
            and (now-issued).total_seconds() <= revocation["max_age_seconds"],
            "revocation unavailable or stale trusted time")
    require(current["sequence"] == message["sequence"]
            and current["content_sha256"] == digest(message)
            and current["issued_at"] == message["issued_at"]
            and current["revoked_ids"] == message["revoked_ids"]
            and current["untrusted_signers"] == message["untrusted_signers"],
            "continuity does not match authenticated revocation")
    require(current["sequence"] >= previous_state["sequence"], "custody sequence rollback")
    for name in ("revoked_ids", "untrusted_signers"):
        require(set(previous_state[name]) <= set(current[name]), "custody revocation set rollback")
    require(utc(current["issued_at"]) >= utc(previous_state["issued_at"])
            and utc(current["observed_at"]) >= utc(previous_state["observed_at"]),
            "custody trusted time rollback")
    checkpoint = digest(current)
    require(material.get("checkpoint_sha256") == checkpoint, "host checkpoint report mismatch")
    return {"version": EVENT_VERSION, "status": "custody_verified",
            "checkpoint_sha256": checkpoint, "state": current,
            "challenge": challenge, "verified_at": now.isoformat(),
            "source": "independent continuity and signed revocation recomputation"}


def atomic_publish_receipt(path, envelope):
    """Custodian-side atomic publication; it never acquires the host lock."""
    path = absolute_path(path)
    temp = path.with_name(path.name + ".custodian." + os.urandom(8).hex())
    try:
        with temp.open("xb") as stream:
            stream.write(canonical(envelope) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp.unlink(missing_ok=True)
