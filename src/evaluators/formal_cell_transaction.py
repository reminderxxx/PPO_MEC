"""Per-cell transactional ledger and atomic commit for formal execution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


FORMAL_CELL_LEDGER_VERSION = "1.0.0"
CELL_IDENTITY_VERSION = "1.0.0"
CELL_STATUSES = {
    "running",
    "incomplete",
    "failed_retryable",
    "failed_terminal",
    "committed",
}


class CellTransactionError(ValueError):
    """Raised when cell resume or artifact commit would be ambiguous."""


def _canonical_sha256(value: Any) -> str:
    def reject(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise CellTransactionError("non-finite cell ledger value")
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise CellTransactionError("cell ledger key must be a string")
                reject(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                reject(child)

    reject(value)
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_inventory(root: str | Path) -> list[dict[str, Any]]:
    base = Path(root)
    if not base.is_dir():
        raise CellTransactionError(f"cell artifact directory is missing: {base}")
    rows = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        relative = path.relative_to(base).as_posix()
        if relative == "committed_marker.json":
            continue
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not rows:
        raise CellTransactionError("cell artifact inventory is empty")
    return rows


def stable_cell_id(phase: str, coordinates: Mapping[str, Any]) -> str:
    if not phase or not isinstance(phase, str):
        raise CellTransactionError("cell phase must be a non-empty string")
    digest = _canonical_sha256(
        {
            "cell_identity_version": CELL_IDENTITY_VERSION,
            "phase": phase,
            "coordinates": dict(coordinates),
        }
    )[:24]
    return f"{phase}-{digest}"


def stable_episode_id(coordinates: Mapping[str, Any]) -> str:
    return stable_cell_id("episode", coordinates)


def atomic_write_json_create_only(path: str | Path, payload: Any) -> None:
    """Atomically publish a JSON object without overwriting an existing file."""

    target = Path(path)
    if target.exists():
        raise CellTransactionError(f"atomic JSON target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.staging-{os.getpid()}-{time.monotonic_ns()}"
    encoded = json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise CellTransactionError(
                f"atomic JSON target already exists: {target}"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class CellExecutionIdentity:
    run_id: str
    execution_commit: str
    protocol_semantic_sha256: str
    resource_registry_semantic_sha256: str
    environment_fingerprint: str
    split_semantic_sha256: str
    window_contract_semantic_sha256: str
    catalog_fingerprint: str
    runtime_identity: str
    command_matrix_sha256: str

    def to_dict(self) -> dict[str, str]:
        return dict(self.__dict__)

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())


def _record_hash(record: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in record.items() if key != "current_ledger_hash"}
    )


def validate_cell_ledger(
    records: Sequence[Mapping[str, Any]],
    *,
    identity: CellExecutionIdentity | None = None,
) -> dict[str, Any]:
    previous: str | None = None
    terminal_attempts: set[tuple[str, int]] = set()
    committed_cells: set[str] = set()
    for index, raw in enumerate(records, start=1):
        record = dict(raw)
        required = {
            "formal_cell_ledger_version",
            "sequence_number",
            "run_identity_fingerprint",
            "cell_id",
            "phase",
            "coordinates",
            "command_hash",
            "input_hash",
            "protocol_semantic_sha256",
            "resource_registry_semantic_sha256",
            "environment_fingerprint",
            "attempt",
            "status",
            "started_at_utc",
            "completed_at_utc",
            "duration_seconds",
            "return_code",
            "failure_classification",
            "retry_allowed",
            "staging_path",
            "committed_path",
            "artifact_inventory",
            "artifact_inventory_sha256",
            "previous_ledger_hash",
            "current_ledger_hash",
        }
        missing = required - set(record)
        if missing:
            raise CellTransactionError(
                f"cell ledger record {index} missing fields: {sorted(missing)}"
            )
        if record["formal_cell_ledger_version"] != FORMAL_CELL_LEDGER_VERSION:
            raise CellTransactionError("cell ledger version mismatch")
        if int(record["sequence_number"]) != index:
            raise CellTransactionError("cell ledger sequence mismatch")
        if record["previous_ledger_hash"] != previous:
            raise CellTransactionError("cell ledger previous hash mismatch")
        expected = _record_hash(record)
        if record["current_ledger_hash"] != expected:
            raise CellTransactionError("cell ledger current hash mismatch")
        previous = expected
        status = str(record["status"])
        if status not in CELL_STATUSES:
            raise CellTransactionError("unknown cell status")
        attempt_key = (str(record["cell_id"]), int(record["attempt"]))
        if status != "running":
            if attempt_key in terminal_attempts:
                raise CellTransactionError("cell attempt has duplicate terminal record")
            terminal_attempts.add(attempt_key)
        if status == "committed":
            cell_id = str(record["cell_id"])
            if cell_id in committed_cells:
                raise CellTransactionError("cell has duplicate committed record")
            committed_cells.add(cell_id)
            if not record["artifact_inventory"] or not record["artifact_inventory_sha256"]:
                raise CellTransactionError("committed cell lacks artifact inventory")
        if identity is not None and record["run_identity_fingerprint"] != identity.fingerprint:
            raise CellTransactionError("cell ledger cross-run identity mismatch")
    return {
        "status": "pass",
        "record_count": len(records),
        "committed_cell_count": len(committed_cells),
        "last_ledger_hash": previous,
    }


class FormalCellLedger:
    """Append-only per-cell ledger bound to exactly one run identity."""

    def __init__(
        self,
        *,
        run_root: str | Path,
        identity: CellExecutionIdentity,
        resume: bool = False,
        utc_clock: Callable[[], datetime] | None = None,
        monotonic_ns: Callable[[], int] | None = None,
    ) -> None:
        self.run_root = Path(run_root)
        self.identity = identity
        self.ledger_path = self.run_root / "cell_state.jsonl"
        self.identity_path = self.run_root / "cell_ledger_identity.json"
        self._utc_clock = utc_clock or (lambda: datetime.now(timezone.utc))
        self._monotonic_ns = monotonic_ns or time.monotonic_ns
        if resume:
            if not self.identity_path.is_file() or not self.ledger_path.is_file():
                raise CellTransactionError("cell resume requires existing identity and ledger")
            observed = json.loads(self.identity_path.read_text(encoding="utf-8"))
            if observed.get("identity") != identity.to_dict() or observed.get(
                "run_identity_fingerprint"
            ) != identity.fingerprint:
                raise CellTransactionError("cell resume protocol/environment/run drift")
            self.records()
        else:
            if self.identity_path.exists() or self.ledger_path.exists():
                raise CellTransactionError("cell ledger already exists")
            self.run_root.mkdir(parents=True, exist_ok=True)
            payload = {
                "formal_cell_ledger_version": FORMAL_CELL_LEDGER_VERSION,
                "identity": identity.to_dict(),
                "run_identity_fingerprint": identity.fingerprint,
            }
            self.identity_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            self.ledger_path.touch(exist_ok=False)

    def records(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        result = []
        for line_number, line in enumerate(
            self.ledger_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CellTransactionError(
                    f"invalid cell ledger line {line_number}"
                ) from exc
            if not isinstance(item, dict):
                raise CellTransactionError("cell ledger record must be an object")
            result.append(item)
        validate_cell_ledger(result, identity=self.identity)
        return result

    def _append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        records = self.records()
        payload = {
            "formal_cell_ledger_version": FORMAL_CELL_LEDGER_VERSION,
            **dict(record),
            "sequence_number": len(records) + 1,
            "run_identity_fingerprint": self.identity.fingerprint,
            "previous_ledger_hash": (
                records[-1]["current_ledger_hash"] if records else None
            ),
        }
        payload["current_ledger_hash"] = _record_hash(payload)
        validate_cell_ledger([*records, payload], identity=self.identity)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        return payload

    def _records_for(self, cell_id: str) -> list[dict[str, Any]]:
        return [record for record in self.records() if record["cell_id"] == cell_id]

    @staticmethod
    def _active_running(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        terminal_attempts = {
            int(record["attempt"])
            for record in records
            if record["status"] != "running"
        }
        return [
            dict(record)
            for record in records
            if record["status"] == "running"
            and int(record["attempt"]) not in terminal_attempts
        ]

    def _base(
        self,
        *,
        cell_id: str,
        phase: str,
        coordinates: Mapping[str, Any],
        command_hash: str,
        input_hash: str,
        attempt: int,
        status: str,
        started_at_utc: str,
        completed_at_utc: str | None,
        duration_seconds: float | None,
        return_code: int | None,
        failure_classification: str | None,
        retry_allowed: bool,
        staging_path: Path,
        committed_path: Path,
        inventory: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        rows = [dict(item) for item in (inventory or [])]
        return {
            "cell_id": cell_id,
            "phase": phase,
            "coordinates": dict(coordinates),
            "command_hash": command_hash,
            "input_hash": input_hash,
            "protocol_semantic_sha256": self.identity.protocol_semantic_sha256,
            "resource_registry_semantic_sha256": (
                self.identity.resource_registry_semantic_sha256
            ),
            "environment_fingerprint": self.identity.environment_fingerprint,
            "attempt": attempt,
            "status": status,
            "started_at_utc": started_at_utc,
            "completed_at_utc": completed_at_utc,
            "duration_seconds": duration_seconds,
            "return_code": return_code,
            "failure_classification": failure_classification,
            "retry_allowed": retry_allowed,
            "staging_path": str(staging_path),
            "committed_path": str(committed_path),
            "artifact_inventory": rows,
            "artifact_inventory_sha256": _canonical_sha256(rows) if rows else None,
        }

    def verify_committed(self, cell_id: str) -> dict[str, Any]:
        records = self._records_for(cell_id)
        committed = [record for record in records if record["status"] == "committed"]
        if len(committed) != 1:
            raise CellTransactionError("cell is not uniquely committed")
        record = committed[0]
        path = Path(record["committed_path"])
        marker_path = path / "committed_marker.json"
        if not marker_path.is_file():
            raise CellTransactionError("committed cell marker is missing")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        inventory = artifact_inventory(path)
        inventory_hash = _canonical_sha256(inventory)
        if inventory_hash != record["artifact_inventory_sha256"]:
            raise CellTransactionError("committed cell output drift")
        if marker.get("artifact_inventory_sha256") != inventory_hash:
            raise CellTransactionError("committed marker output hash mismatch")
        if marker.get("run_identity_fingerprint") != self.identity.fingerprint:
            raise CellTransactionError("committed marker cross-run mismatch")
        if marker.get("cell_id") != cell_id:
            raise CellTransactionError("committed marker cell ID mismatch")
        return record

    def _recover_published_cell(self, cell_id: str) -> dict[str, Any] | None:
        """Finish the ledger append after an atomic publish/process interruption."""

        records = self._records_for(cell_id)
        if any(record["status"] == "committed" for record in records):
            return None
        running = self._active_running(records)
        if not running:
            return None
        record = running[-1]
        target = Path(record["committed_path"])
        marker_path = target / "committed_marker.json"
        if not marker_path.is_file():
            return None
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        expected_marker = {
            "cell_id": cell_id,
            "run_identity_fingerprint": self.identity.fingerprint,
            "command_hash": record["command_hash"],
            "input_hash": record["input_hash"],
        }
        if any(marker.get(key) != value for key, value in expected_marker.items()):
            raise CellTransactionError("published cell marker identity mismatch")
        inventory = artifact_inventory(target)
        inventory_hash = _canonical_sha256(inventory)
        if marker.get("artifact_inventory_sha256") != inventory_hash:
            raise CellTransactionError("published cell marker output hash mismatch")
        return self._append(
            self._base(
                cell_id=cell_id,
                phase=record["phase"],
                coordinates=record["coordinates"],
                command_hash=record["command_hash"],
                input_hash=record["input_hash"],
                attempt=int(record["attempt"]),
                status="committed",
                started_at_utc=record["started_at_utc"],
                completed_at_utc=self._utc_clock().isoformat(),
                duration_seconds=None,
                return_code=0,
                failure_classification=None,
                retry_allowed=False,
                staging_path=Path(record["staging_path"]),
                committed_path=target,
                inventory=inventory,
            )
        )

    def begin_cell(
        self,
        *,
        phase: str,
        coordinates: Mapping[str, Any],
        command: Sequence[str],
        input_hash: str,
        committed_path: str | Path | None = None,
    ) -> dict[str, Any]:
        cell_id = stable_cell_id(phase, coordinates)
        command_hash = _canonical_sha256(list(command))
        records = self._records_for(cell_id)
        if not any(record["status"] == "committed" for record in records):
            self._recover_published_cell(cell_id)
            records = self._records_for(cell_id)
        if any(record["status"] == "committed" for record in records):
            prior = self.verify_committed(cell_id)
            if prior["command_hash"] != command_hash or prior["input_hash"] != input_hash:
                raise CellTransactionError("committed cell command/input hash mismatch")
            return {"status": "skipped_committed", "cell_id": cell_id, "record": prior}
        if any(record["status"] == "failed_terminal" for record in records):
            raise CellTransactionError("terminal cell failure forbids resume")
        retryable = [record for record in records if record["status"] == "failed_retryable"]
        if len(retryable) > 1:
            raise CellTransactionError("retryable cell already exhausted one retry")
        running = self._active_running(records)
        if running:
            last = running[-1]
            if last["command_hash"] != command_hash or last["input_hash"] != input_hash:
                raise CellTransactionError("incomplete cell command/input drift")
            now = self._utc_clock()
            self._append(
                {
                    **{key: value for key, value in last.items() if key not in {
                        "formal_cell_ledger_version", "sequence_number", "status",
                        "completed_at_utc", "duration_seconds", "return_code",
                        "failure_classification", "retry_allowed", "previous_ledger_hash",
                        "current_ledger_hash", "run_identity_fingerprint",
                    }},
                    "status": "incomplete",
                    "completed_at_utc": now.isoformat(),
                    "duration_seconds": None,
                    "return_code": None,
                    "failure_classification": "process_terminated_before_cell_commit",
                    "retry_allowed": True,
                }
            )
        attempt = max([int(record["attempt"]) for record in records] or [0]) + 1
        if attempt > 2:
            raise CellTransactionError("cell attempt limit exceeded")
        target = Path(committed_path) if committed_path else (
            self.run_root / "cells" / phase / cell_id
        )
        staging = self.run_root / ".staging" / phase / cell_id / f"attempt_{attempt:02d}"
        if staging.exists():
            raise CellTransactionError("cell staging path already exists")
        staging.mkdir(parents=True, exist_ok=False)
        started = self._utc_clock()
        if started.tzinfo is None or started.utcoffset() is None:
            raise CellTransactionError("cell UTC clock must be timezone-aware")
        record = self._append(
            self._base(
                cell_id=cell_id,
                phase=phase,
                coordinates=coordinates,
                command_hash=command_hash,
                input_hash=input_hash,
                attempt=attempt,
                status="running",
                started_at_utc=started.isoformat(),
                completed_at_utc=None,
                duration_seconds=None,
                return_code=None,
                failure_classification=None,
                retry_allowed=True,
                staging_path=staging,
                committed_path=target,
            )
        )
        return {"status": "execute", "cell_id": cell_id, "record": record}

    def commit_cell(
        self,
        cell_id: str,
        *,
        required_paths: Sequence[str] = (),
        monotonic_started_ns: int | None = None,
        artifact_subpath: str | Path = ".",
    ) -> dict[str, Any]:
        running = self._active_running(self._records_for(cell_id))
        if not running:
            if any(
                record["status"] == "committed" for record in self._records_for(cell_id)
            ):
                self.verify_committed(cell_id)
                return {"status": "already_committed", "cell_id": cell_id}
            raise CellTransactionError("cell has no running attempt to commit")
        record = running[-1]
        staging = Path(record["staging_path"])
        target = Path(record["committed_path"])
        artifact_root = (staging / Path(artifact_subpath)).resolve()
        try:
            artifact_root.relative_to(staging.resolve())
        except ValueError as exc:
            raise CellTransactionError("cell artifact subpath escapes staging") from exc
        for relative in required_paths:
            if not (artifact_root / relative).is_file():
                raise CellTransactionError(f"cell required artifact is missing: {relative}")
        inventory = artifact_inventory(artifact_root)
        inventory_hash = _canonical_sha256(inventory)
        marker = {
            "formal_cell_ledger_version": FORMAL_CELL_LEDGER_VERSION,
            "cell_id": cell_id,
            "run_identity_fingerprint": self.identity.fingerprint,
            "command_hash": record["command_hash"],
            "input_hash": record["input_hash"],
            "artifact_inventory_sha256": inventory_hash,
        }
        (artifact_root / "committed_marker.json").write_text(
            json.dumps(marker, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            raise CellTransactionError("committed cell path already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(artifact_root, target)
        completed = self._utc_clock()
        duration = None
        if monotonic_started_ns is not None:
            elapsed = self._monotonic_ns() - monotonic_started_ns
            if elapsed < 0:
                raise CellTransactionError("cell monotonic clock moved backwards")
            duration = elapsed / 1_000_000_000
        event = self._append(
            self._base(
                cell_id=cell_id,
                phase=record["phase"],
                coordinates=record["coordinates"],
                command_hash=record["command_hash"],
                input_hash=record["input_hash"],
                attempt=int(record["attempt"]),
                status="committed",
                started_at_utc=record["started_at_utc"],
                completed_at_utc=completed.isoformat(),
                duration_seconds=duration,
                return_code=0,
                failure_classification=None,
                retry_allowed=False,
                staging_path=Path(record["staging_path"]),
                committed_path=target,
                inventory=inventory,
            )
        )
        self.verify_committed(cell_id)
        return event

    def fail_cell(
        self,
        cell_id: str,
        *,
        return_code: int | None,
        classification: str,
        retryable: bool,
    ) -> dict[str, Any]:
        running = self._active_running(self._records_for(cell_id))
        if not running:
            raise CellTransactionError("cell has no running attempt to fail")
        record = running[-1]
        if retryable and any(
            item["status"] == "failed_retryable" for item in self._records_for(cell_id)
        ):
            retryable = False
        return self._append(
            self._base(
                cell_id=cell_id,
                phase=record["phase"],
                coordinates=record["coordinates"],
                command_hash=record["command_hash"],
                input_hash=record["input_hash"],
                attempt=int(record["attempt"]),
                status="failed_retryable" if retryable else "failed_terminal",
                started_at_utc=record["started_at_utc"],
                completed_at_utc=self._utc_clock().isoformat(),
                duration_seconds=None,
                return_code=return_code,
                failure_classification=classification,
                retry_allowed=retryable,
                staging_path=Path(record["staging_path"]),
                committed_path=Path(record["committed_path"]),
            )
        )

    def committed_records(self, *, phase: str | None = None) -> list[dict[str, Any]]:
        result = [
            record
            for record in self.records()
            if record["status"] == "committed"
            and (phase is None or record["phase"] == phase)
        ]
        for record in result:
            self.verify_committed(record["cell_id"])
        return result

    def assert_complete_matrix(
        self,
        *,
        phase: str,
        expected_cell_ids: Sequence[str],
    ) -> dict[str, Any]:
        committed = {
            record["cell_id"] for record in self.committed_records(phase=phase)
        }
        expected = set(expected_cell_ids)
        if committed != expected:
            raise CellTransactionError(
                f"partial committed cell matrix: missing={sorted(expected - committed)}, "
                f"extra={sorted(committed - expected)}"
            )
        return {"status": "pass", "committed_cell_count": len(committed)}


def run_transactional_cell(
    ledger: FormalCellLedger,
    *,
    phase: str,
    coordinates: Mapping[str, Any],
    command: Sequence[str],
    input_hash: str,
    required_paths: Sequence[str],
    committed_path: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    command_builder: Callable[[Path], Sequence[str]] | None = None,
) -> dict[str, Any]:
    begun = ledger.begin_cell(
        phase=phase,
        coordinates=coordinates,
        command=command,
        input_hash=input_hash,
        committed_path=committed_path,
    )
    if begun["status"] == "skipped_committed":
        return begun
    record = begun["record"]
    staging = Path(record["staging_path"])
    actual_command = list(command_builder(staging) if command_builder else command)
    started_ns = time.monotonic_ns()
    completed = subprocess.run(
        actual_command,
        cwd=cwd,
        env=dict(environment) if environment is not None else None,
        text=True,
        capture_output=True,
        check=False,
    )
    (staging / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (staging / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        ledger.fail_cell(
            begun["cell_id"],
            return_code=completed.returncode,
            classification=(
                "infrastructure_retryable"
                if completed.returncode == 75
                else "cell_command_failure"
            ),
            retryable=completed.returncode == 75,
        )
        raise CellTransactionError(
            f"cell command failed: {begun['cell_id']} return_code={completed.returncode}"
        )
    return ledger.commit_cell(
        begun["cell_id"],
        required_paths=required_paths,
        monotonic_started_ns=started_ns,
    )


__all__ = [
    "CELL_IDENTITY_VERSION",
    "CELL_STATUSES",
    "CellExecutionIdentity",
    "CellTransactionError",
    "FORMAL_CELL_LEDGER_VERSION",
    "FormalCellLedger",
    "artifact_inventory",
    "atomic_write_json_create_only",
    "run_transactional_cell",
    "stable_cell_id",
    "stable_episode_id",
    "validate_cell_ledger",
]
