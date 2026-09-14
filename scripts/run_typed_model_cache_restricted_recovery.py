"""Qualify or execute one ordered G14R20-I5 restricted recovery cell.

Only ``formal_ablation`` and the two contract-bound cell IDs are accepted.
There is no recovery grant issuer, retry, old-run mutation, later-phase, or
holdout action in this entrypoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any, Mapping
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.continuation_executor.locking import writer_lock_path
from src.evaluators.formal_cell_transaction import execute_cell_artifact_transaction
from src.runtime.evaluation_only_execution import canonical_sha256, file_sha256
from src.runtime.restricted_recovery import (
    ALLOWED_CELL_IDS,
    FAILED_CELL_ID,
    ORIGINAL_RUN_ROOT,
    PHASE,
    RestrictedRecoveryCellLedger,
    RestrictedRecoveryError,
    RestrictedRecoveryPhaseLedger,
    UNSTARTED_CELL_ID,
    audit_original_recovery_source,
    build_recovery_handoff_manifest,
    recovery_cell_identity,
    restricted_cell_layout,
    stable_source_audit_projection,
    validate_recovery_handoff_manifest,
    validate_recovery_executor_live,
    validate_restricted_recovery_request,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-request-path", required=True)
    parser.add_argument("--recovery-grant-path", required=True)
    parser.add_argument("--cell-id", choices=ALLOWED_CELL_IDS, required=True)
    parser.add_argument("--check", choices=("qualify", "execute"), default="qualify")
    return parser


def _read(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise RestrictedRecoveryError(f"required recovery authorization object is missing: {target}")
    value = json.loads(target.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RestrictedRecoveryError("recovery authorization object must be a mapping")
    return value


def _encoded(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


class RestrictedRecoverySingleWriter:
    """Create-only recovery lock with no PID liveness or crash takeover path.

    The old continuation lock deliberately requires host ``ps`` evidence for
    crash recovery.  I5 never recovers a held lock: a retained ``held`` owner
    freezes this new recovery execution, while a normally ``released`` owner
    permits the second, ordered process to acquire the same inode.
    """

    def __init__(self, run_root: Path, executor_sha256: str, authorize) -> None:
        self.root = run_root
        self.executor_sha256 = executor_sha256
        self.authorize = authorize
        self.fd: int | None = None
        self.owner: dict[str, Any] | None = None

    def _write(self, value: Mapping[str, Any]) -> None:
        if self.fd is None:
            raise RestrictedRecoveryError("recovery writer lock is not open")
        os.lseek(self.fd, 0, os.SEEK_SET)
        data = _encoded(value)
        if os.write(self.fd, data) != len(data):
            raise RestrictedRecoveryError("short recovery lock record write")
        os.ftruncate(self.fd, len(data))
        os.fsync(self.fd)

    def __enter__(self):
        self.authorize()
        if self.root.is_symlink() or (self.root.exists() and not self.root.is_dir()):
            raise RestrictedRecoveryError("recovery root must be absent or a real directory")
        parent = self.root.parent
        if parent.is_symlink() or not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
            raise RestrictedRecoveryError("recovery parent is not a writable real directory")
        path = writer_lock_path(self.root)
        lock_parent = path.parent
        if lock_parent.exists() and (lock_parent.is_symlink() or not lock_parent.is_dir()):
            raise RestrictedRecoveryError("recovery lock parent must be a real directory")
        lock_parent.mkdir(mode=0o700, exist_ok=True)
        if lock_parent.is_symlink():
            raise RestrictedRecoveryError("recovery lock parent changed to a symlink")
        flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
        self.fd = os.open(path, flags, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.authorize()
            os.lseek(self.fd, 0, os.SEEK_SET)
            previous_bytes = os.read(self.fd, 65537)
            if len(previous_bytes) > 65536:
                raise RestrictedRecoveryError("recovery lock owner record oversized")
            if previous_bytes:
                previous = json.loads(previous_bytes)
                if previous.get("state") != "released":
                    raise RestrictedRecoveryError(
                        "held recovery lock freezes execution; automatic takeover is forbidden"
                    )
            self.owner = {
                "version": "1.0.0",
                "state": "held",
                "nonce": uuid.uuid4().hex,
                "process": {
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "started": None,
                    "identity_method": "pid_without_liveness_inference",
                },
                "executor_sha256": self.executor_sha256,
            }
            self._write(self.owner)
            return self
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None and self.owner is not None:
                self._write({**self.owner, "state": "released"})
        finally:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None


def _create_file(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _utc(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise RestrictedRecoveryError("invalid recovery authorization timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RestrictedRecoveryError("recovery authorization timestamp requires timezone")
    return parsed.astimezone(timezone.utc)


def verify_recovery_grant(
    request: Mapping[str, Any],
    grant: Mapping[str, Any],
    *,
    synthetic_only: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    expected_fields = {
        "version",
        "status",
        "authorization_request_sha256",
        "recovery_execution_id",
        "original_run_id",
        "executor_commit",
        "executor_git_tree",
        "allowed_phase",
        "allowed_cell_ids",
        "holdout_capability",
        "later_phases_authorized",
        "independent_review",
        "issued_at",
        "expires_at",
    }
    if set(grant) != expected_fields or grant.get("version") != "1.0.0":
        raise RestrictedRecoveryError("restricted recovery grant schema mismatch")
    expected_status = (
        "AUTHORIZED_FOR_SYNTHETIC_RESTRICTED_RECOVERY_ACCEPTANCE"
        if synthetic_only
        else "AUTHORIZED_FOR_RESTRICTED_RECOVERY"
    )
    execution = request["recovery_execution"]
    expected = {
        "status": expected_status,
        "authorization_request_sha256": request["authorization_request_sha256"],
        "recovery_execution_id": execution["recovery_execution_id"],
        "original_run_id": execution["original_run_id"],
        "executor_commit": execution["executor_commit"],
        "executor_git_tree": execution["executor_git_tree"],
        "allowed_phase": PHASE,
        "allowed_cell_ids": list(ALLOWED_CELL_IDS),
        "holdout_capability": False,
        "later_phases_authorized": False,
    }
    if any(grant.get(key) != value for key, value in expected.items()):
        raise RestrictedRecoveryError("restricted recovery grant identity/scope drift")
    current = now or datetime.now(timezone.utc)
    if not _utc(grant["issued_at"]) <= current < _utc(grant["expires_at"]):
        raise RestrictedRecoveryError("restricted recovery grant is future or expired")
    review_row = grant["independent_review"]
    if not isinstance(review_row, dict) or set(review_row) != {"path", "sha256", "size_bytes"}:
        raise RestrictedRecoveryError("restricted recovery independent review reference is invalid")
    review_path = Path(str(review_row["path"]))
    if (
        review_path.is_symlink()
        or not review_path.is_file()
        or review_path.stat().st_size != review_row["size_bytes"]
        or file_sha256(review_path) != review_row["sha256"]
    ):
        raise RestrictedRecoveryError("restricted recovery independent review bytes drift")
    review = _read(review_path)
    if (
        review.get("status") != "pass"
        or review.get("reviewer_id") == review.get("implementation_agent_id")
        or review.get("authorization_request_sha256")
        != request["authorization_request_sha256"]
        or review.get("recovery_execution_identity_sha256")
        != execution["recovery_execution_identity_sha256"]
        or review.get("holdout_capability") is not False
        or bool(review.get("synthetic_only", False)) != synthetic_only
    ):
        raise RestrictedRecoveryError("restricted recovery independent review is incomplete")
    return {
        "approval_verified": True,
        "authorization_mode": (
            "synthetic_restricted_recovery_acceptance_v1"
            if synthetic_only
            else "one_run_two_cell_restricted_recovery_v1"
        ),
        "grant_sha256": canonical_sha256(dict(grant)),
        "holdout_capability": False,
        "later_phases_authorized": False,
    }


def _initial_files(request: Mapping[str, Any]) -> dict[str, Any]:
    source = _read(ORIGINAL_RUN_ROOT / "evaluation_model_source_reference.json")
    execution = request["recovery_execution"]
    return {
        "restricted_recovery_request.json": dict(request),
        "evaluation_model_source_reference.json": source,
        "restricted_recovery_execution_contract.json": execution,
        "resolved_execution_context.json": execution["resolved_execution_context"],
        "restricted_recovery_state.json": {
            "recovery_execution_id": execution["recovery_execution_id"],
            "original_run_id": execution["original_run_id"],
            "authorization_request_sha256": request["authorization_request_sha256"],
            "initialization_status": "initialized",
            "allowed_phase": PHASE,
            "allowed_cell_ids": list(ALLOWED_CELL_IDS),
            "external_committed_cell_count": 6,
            "real_recovery_started": False,
            "holdout_opened": False,
            "later_phases_authorized": False,
        },
    }


def _initialize(
    request: Mapping[str, Any],
) -> tuple[Path, RestrictedRecoveryCellLedger, RestrictedRecoveryPhaseLedger]:
    execution = request["recovery_execution"]
    root = Path(execution["recovery_root"])
    root.mkdir(exist_ok=False)
    _create_file(root / "recovery_phase_state.jsonl", b"")
    for name, payload in _initial_files(request).items():
        _create_file(root / name, _encoded(payload))
    cells = RestrictedRecoveryCellLedger(
        run_root=root,
        identity=recovery_cell_identity(request),
    )
    marker = {
        "version": "1.0.0",
        "recovery_root": str(root),
        "authorization_request_sha256": request["authorization_request_sha256"],
        "recovery_execution_identity_sha256": execution[
            "recovery_execution_identity_sha256"
        ],
        "files": {
            name: file_sha256(root / name)
            for name in [*_initial_files(request), "cell_ledger_identity.json"]
        },
        "initial_phase_ledger_sha256": file_sha256(root / "recovery_phase_state.jsonl"),
        "initial_cell_ledger_sha256": file_sha256(root / "cell_state.jsonl"),
    }
    _create_file(root / "restricted_recovery_initialization_complete.json", _encoded(marker))
    phase = RestrictedRecoveryPhaseLedger(
        path=root / "recovery_phase_state.jsonl",
        recovery_identity_sha256=execution["recovery_execution_identity_sha256"],
    )
    return root, cells, phase


def _load(
    request: Mapping[str, Any],
) -> tuple[Path, RestrictedRecoveryCellLedger, RestrictedRecoveryPhaseLedger]:
    execution = request["recovery_execution"]
    root = Path(execution["recovery_root"])
    if not root.exists():
        return _initialize(request)
    marker = _read(root / "restricted_recovery_initialization_complete.json")
    names = [*_initial_files(request), "cell_ledger_identity.json"]
    if (
        marker.get("version") != "1.0.0"
        or marker.get("recovery_root") != str(root)
        or marker.get("authorization_request_sha256")
        != request["authorization_request_sha256"]
        or marker.get("recovery_execution_identity_sha256")
        != execution["recovery_execution_identity_sha256"]
        or set(marker.get("files", {})) != set(names)
        or any(marker["files"][name] != file_sha256(root / name) for name in names)
        or marker.get("initial_phase_ledger_sha256") != hashlib.sha256(b"").hexdigest()
        or marker.get("initial_cell_ledger_sha256") != hashlib.sha256(b"").hexdigest()
    ):
        raise RestrictedRecoveryError("restricted recovery initialization identity/hash drift")
    for name, payload in _initial_files(request).items():
        path = root / name
        if path.is_symlink() or path.read_bytes() != _encoded(payload):
            raise RestrictedRecoveryError("restricted recovery identity/context bytes drift: " + name)
    for name in ("recovery_phase_state.jsonl", "cell_state.jsonl", "cell_ledger_identity.json"):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise RestrictedRecoveryError("restricted recovery initialization is incomplete")
    cells = RestrictedRecoveryCellLedger(
        run_root=root,
        identity=recovery_cell_identity(request),
        resume=True,
    )
    phase = RestrictedRecoveryPhaseLedger(
        path=root / "recovery_phase_state.jsonl",
        recovery_identity_sha256=execution["recovery_execution_identity_sha256"],
    )
    phase.records()
    return root, cells, phase


def _assert_next(
    root: Path,
    cells: RestrictedRecoveryCellLedger,
    phase: RestrictedRecoveryPhaseLedger,
    cell_id: str,
) -> None:
    events = [row["event"] for row in phase.records()]
    records = cells.records()
    if any(row["status"] == "failed_terminal" for row in records) or (
        events and events[-1] == "failed"
    ):
        raise RestrictedRecoveryError("restricted recovery failure is terminal")
    if events and events[-1] == "completed":
        raise RestrictedRecoveryError("restricted recovery is already completed")
    if not events:
        if cell_id != FAILED_CELL_ID:
            raise RestrictedRecoveryError("failed source cell recovery must execute first")
        return
    if events == ["started", "prerequisite_committed"]:
        if cell_id != UNSTARTED_CELL_ID:
            raise RestrictedRecoveryError("unstarted source cell must execute second")
        return
    raise RestrictedRecoveryError("interrupted restricted recovery cannot resume")


def _write_handoff(root: Path, request: Mapping[str, Any], cells: RestrictedRecoveryCellLedger) -> dict[str, Any]:
    handoff = build_recovery_handoff_manifest(request, cells)
    validate_recovery_handoff_manifest(handoff)
    _create_file(root / "ablation_recovery_handoff.json", _encoded(handoff))
    followup: dict[str, Any] = {
        "version": "1.0.0",
        "status": "UNSIGNED_NOT_ISSUED_NOT_EXECUTABLE",
        "source_handoff_path": str(root / "ablation_recovery_handoff.json"),
        "source_handoff_sha256": handoff["handoff_sha256"],
        "requested_review_scope": [
            "formal_support",
            "formal_scalability",
            "formal_statistics",
            "formal_gate",
            "complete_without_holdout",
        ],
        "next_stage_authorized": False,
        "recovery_grant_issued": False,
        "real_recovery_started": False,
        "holdout_capability": False,
        "holdout_opened": False,
    }
    followup["request_sha256"] = canonical_sha256(followup)
    _create_file(root / "unsigned_followup_request.json", _encoded(followup))
    return handoff


def execute_recovery_cell(
    request: Mapping[str, Any],
    grant: Mapping[str, Any],
    cell_id: str,
    *,
    scientific_child_adapter=None,
) -> dict[str, Any]:
    synthetic_only = scientific_child_adapter is not None
    authorization = verify_recovery_grant(
        request,
        grant,
        synthetic_only=synthetic_only,
    )
    execution = request["recovery_execution"]
    root = Path(execution["recovery_root"])
    if cell_id not in ALLOWED_CELL_IDS:
        raise RestrictedRecoveryError("cell is outside restricted recovery scope")
    if not root.exists() and cell_id != FAILED_CELL_ID:
        raise RestrictedRecoveryError("first recovery process must execute the failed source cell")
    if root.exists():
        preflight_root, preflight_cells, preflight_phase = _load(request)
        _assert_next(
            preflight_root,
            preflight_cells,
            preflight_phase,
            cell_id,
        )
    # Every process revalidates the full immutable source before opening the new
    # writer lock. Rejection here produces neither dispatch nor recovery write.
    validate_recovery_executor_live(request)
    audit = audit_original_recovery_source()
    source = _read(ORIGINAL_RUN_ROOT / "evaluation_model_source_reference.json")
    original = request["original_identity"]
    if (
        stable_source_audit_projection(audit)
        != stable_source_audit_projection(request["immutable_source_audit"])
        or audit["external_committed_cells"] != request["external_committed_cells"]
        or source["models"] != request["source_models"]
        or original.get("run_id") != execution["original_run_id"]
        or original.get("request") != audit["original_request"]
        or original.get("project_grant") != audit["original_project_grant"]
        or original.get("supplemental_authorization")
        != audit["original_supplemental_authorization"]
        or original.get("failure_report") != audit["original_failure_report"]
        or original.get("source_reference_sha256") != audit["source_reference_sha256"]
    ):
        raise RestrictedRecoveryError("immutable recovery source/request binding drift")
    if len(audit["external_committed_cells"]) != 6:
        raise RestrictedRecoveryError("external committed source count drift")
    with RestrictedRecoverySingleWriter(
        root,
        execution["recovery_execution_identity_sha256"],
        lambda: verify_recovery_grant(
            request,
            grant,
            synthetic_only=synthetic_only,
        ),
    ):
        root, cells, phase = _load(request)
        _assert_next(root, cells, phase, cell_id)
        command, coordinates, final, builder, resolver = restricted_cell_layout(
            request, cell_id
        )
        if scientific_child_adapter is not None:
            builder, resolver = scientific_child_adapter.adapt(
                request, cell_id, builder, resolver
            )
        if not phase.records():
            phase.append(
                "started",
                cell_id=cell_id,
                detail={
                    "source_attempt": 1,
                    "recovery_attempt": 2,
                    "external_committed_cell_count": 6,
                },
            )
        try:
            result = execute_cell_artifact_transaction(
                cells,
                phase=PHASE,
                coordinates=coordinates,
                command=command,
                input_hash=canonical_sha256(
                    {
                        "command": command,
                        "authorization_request_sha256": request[
                            "authorization_request_sha256"
                        ],
                        "original_cell_terminal": request["immutable_source_audit"][
                            "cell_ledger_prefix"
                        ]["terminal_hash"],
                    }
                ),
                committed_path=final,
                command_builder=builder,
                artifact_resolver=resolver,
                environment=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1"),
                cwd=ROOT,
                command_failure_classification="restricted_recovery_cell_failure",
            )
            if result.get("status") != "committed":
                phase.append(
                    "failed",
                    cell_id=cell_id,
                    detail={"return_code": result.get("return_code"), "automatic_retry": False},
                )
                raise RestrictedRecoveryError("restricted recovery cell failed terminally")
            record = result["record"]
            expected_attempt = 2 if cell_id == FAILED_CELL_ID else 1
            if int(record["attempt"]) != expected_attempt:
                raise RestrictedRecoveryError("restricted recovery attempt number drift")
            if cell_id == FAILED_CELL_ID:
                phase.append(
                    "prerequisite_committed",
                    cell_id=cell_id,
                    detail={"recovery_attempt": 2, "next_cell": UNSTARTED_CELL_ID},
                )
                return {
                    "authorization": authorization,
                    "status": "prerequisite_committed",
                    "cell_id": cell_id,
                    "recovery_attempt": 2,
                    "next_cell_id": UNSTARTED_CELL_ID,
                    "next_stage_authorized": False,
                    "holdout_opened": False,
                }
            handoff = _write_handoff(root, request, cells)
            phase.append(
                "completed",
                cell_id=cell_id,
                detail={
                    "recovery_attempt": 1,
                    "handoff_sha256": handoff["handoff_sha256"],
                    "next_stage_authorized": False,
                },
            )
            return {
                "authorization": authorization,
                "status": "ablation_recovery_completed_handoff_only",
                "cell_id": cell_id,
                "recovery_attempt": 1,
                "handoff_sha256": handoff["handoff_sha256"],
                "next_stage_authorized": False,
                "recovery_grant_issued_for_next_stage": False,
                "holdout_opened": False,
            }
        except BaseException as exc:
            events = phase.records()
            if not events or events[-1]["event"] not in {"completed", "failed"}:
                phase.append(
                    "failed",
                    cell_id=cell_id,
                    detail={
                        "failure_classification": type(exc).__name__,
                        "automatic_retry": False,
                    },
                )
            raise


def main(*, scientific_child_adapter=None) -> None:
    args = build_parser().parse_args()
    request = _read(args.authorization_request_path)
    validate_restricted_recovery_request(
        request, check_live=args.check == "qualify"
    )
    if scientific_child_adapter is not None:
        from tests.restricted_recovery_public_driver import SyntheticRecoveryChild

        if type(scientific_child_adapter) is not SyntheticRecoveryChild:
            raise RestrictedRecoveryError("unknown restricted recovery acceptance adapter")
        scientific_child_adapter.validate(request)
    grant = _read(args.recovery_grant_path)
    authorization = verify_recovery_grant(
        request,
        grant,
        synthetic_only=scientific_child_adapter is not None,
    )
    if args.check == "qualify":
        print(json.dumps({"status": "qualified", "authorization": authorization}, indent=2))
        return
    result = execute_recovery_cell(
        request,
        grant,
        args.cell_id,
        scientific_child_adapter=scientific_child_adapter,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
