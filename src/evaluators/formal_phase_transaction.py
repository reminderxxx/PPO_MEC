"""Transactional phase ledger v3 with monotonic authoritative timing."""

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


FORMAL_PHASE_LEDGER_VERSION = "3.0.0"
FORMAL_PHASE_TRANSACTION_VERSION = "1.0.0"
PHASE_ABSOLUTE_SANITY_SECONDS = 259_200.0
PHASE_STATUSES = {"running", "completion_candidate", "completed", "failed"}


class PhaseTransactionError(ValueError):
    """Raised when phase timing, transaction, or resume evidence is invalid."""


def _canonical_sha256(value: Any) -> str:
    def reject(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise PhaseTransactionError("non-finite phase ledger value")
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise PhaseTransactionError("phase ledger key must be a string")
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


def _record_hash(record: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in record.items() if key != "current_record_hash"}
    )


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise PhaseTransactionError(f"{field} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise PhaseTransactionError(f"invalid ISO timestamp: {field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PhaseTransactionError(f"timestamp lacks timezone: {field}")
    return parsed


def clock_measurement(
    *,
    started_at_utc: datetime,
    completed_at_utc: datetime,
    monotonic_started_ns: int,
    monotonic_completed_ns: int,
    child_wall_clock_seconds: float,
    adjustment_tolerance_seconds: float = 2.0,
) -> dict[str, Any]:
    if started_at_utc.tzinfo is None or started_at_utc.utcoffset() is None:
        raise PhaseTransactionError("started UTC time must be timezone-aware")
    if completed_at_utc.tzinfo is None or completed_at_utc.utcoffset() is None:
        raise PhaseTransactionError("completed UTC time must be timezone-aware")
    elapsed_ns = int(monotonic_completed_ns) - int(monotonic_started_ns)
    if elapsed_ns < 0:
        raise PhaseTransactionError("monotonic clock moved backwards")
    wall = elapsed_ns / 1_000_000_000
    if wall > PHASE_ABSOLUTE_SANITY_SECONDS:
        raise PhaseTransactionError("phase duration exceeds frozen absolute sanity bound")
    child = float(child_wall_clock_seconds)
    if not math.isfinite(child) or child < 0 or child > wall + 1e-6:
        raise PhaseTransactionError("child command duration is invalid")
    utc_delta = (completed_at_utc - started_at_utc).total_seconds()
    adjustment = utc_delta - wall
    if abs(adjustment) <= adjustment_tolerance_seconds:
        status = "consistent_with_tolerance"
    elif adjustment > 0:
        status = "forward_system_clock_adjustment"
    else:
        status = "backward_system_clock_adjustment"
    return {
        "started_at_utc": started_at_utc.astimezone(timezone.utc).isoformat(),
        "completed_at_utc": completed_at_utc.astimezone(timezone.utc).isoformat(),
        "monotonic_started_ns": int(monotonic_started_ns),
        "monotonic_completed_ns": int(monotonic_completed_ns),
        "wall_clock_seconds": wall,
        "child_wall_clock_seconds": child,
        "finalization_wall_clock_seconds": max(0.0, wall - child),
        "wall_clock_adjustment_seconds": adjustment,
        "clock_consistency_status": status,
        "utc_timestamp_ordered": completed_at_utc >= started_at_utc,
        "duration_authority": "monotonic_clock",
    }


def validate_phase_ledger_v3(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    previous: str | None = None
    terminal_by_phase: dict[str, str] = {}
    candidates: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(records, start=1):
        record = dict(raw)
        required = {
            "formal_phase_ledger_version",
            "formal_phase_transaction_version",
            "sequence_number",
            "phase",
            "status",
            "run_identity_fingerprint",
            "input_hash",
            "command_identity",
            "commands",
            "started_at_utc",
            "completed_at_utc",
            "monotonic_started_ns",
            "monotonic_completed_ns",
            "wall_clock_seconds",
            "child_wall_clock_seconds",
            "finalization_wall_clock_seconds",
            "wall_clock_adjustment_seconds",
            "clock_consistency_status",
            "utc_timestamp_ordered",
            "duration_authority",
            "return_code",
            "failure_classification",
            "output_files",
            "output_hash",
            "completion_candidate_hash",
            "previous_record_hash",
            "current_record_hash",
        }
        missing = required - set(record)
        if missing:
            raise PhaseTransactionError(
                f"phase ledger v3 record {index} missing fields: {sorted(missing)}"
            )
        if record["formal_phase_ledger_version"] != FORMAL_PHASE_LEDGER_VERSION:
            raise PhaseTransactionError("phase ledger v3 version mismatch")
        if record["formal_phase_transaction_version"] != FORMAL_PHASE_TRANSACTION_VERSION:
            raise PhaseTransactionError("phase transaction version mismatch")
        if int(record["sequence_number"]) != index:
            raise PhaseTransactionError("phase ledger v3 sequence mismatch")
        if record["previous_record_hash"] != previous:
            raise PhaseTransactionError("phase ledger v3 previous hash mismatch")
        expected_hash = _record_hash(record)
        if record["current_record_hash"] != expected_hash:
            raise PhaseTransactionError("phase ledger v3 current hash mismatch")
        previous = expected_hash
        phase = str(record["phase"])
        status = str(record["status"])
        if status not in PHASE_STATUSES:
            raise PhaseTransactionError("phase ledger v3 status is invalid")
        if phase in terminal_by_phase:
            raise PhaseTransactionError("phase terminal record is immutable")
        _timestamp(record["started_at_utc"], "started_at_utc")
        if status == "running":
            for field in (
                "completed_at_utc",
                "monotonic_completed_ns",
                "wall_clock_seconds",
                "child_wall_clock_seconds",
                "finalization_wall_clock_seconds",
                "wall_clock_adjustment_seconds",
                "clock_consistency_status",
                "utc_timestamp_ordered",
                "duration_authority",
            ):
                if record[field] is not None:
                    raise PhaseTransactionError("running phase contains terminal timing")
        else:
            _timestamp(record["completed_at_utc"], "completed_at_utc")
            wall = float(record["wall_clock_seconds"])
            child = float(record["child_wall_clock_seconds"])
            finalization = float(record["finalization_wall_clock_seconds"])
            adjustment = float(record["wall_clock_adjustment_seconds"])
            if not all(math.isfinite(item) for item in (wall, child, finalization, adjustment)):
                raise PhaseTransactionError("phase ledger v3 timing is non-finite")
            if wall < 0 or child < 0 or finalization < 0:
                raise PhaseTransactionError("phase ledger v3 duration is negative")
            if abs((child + finalization) - wall) > 1e-6:
                raise PhaseTransactionError("phase ledger v3 duration components disagree")
            if int(record["monotonic_completed_ns"]) < int(record["monotonic_started_ns"]):
                raise PhaseTransactionError("phase ledger v3 monotonic order is invalid")
            if record["duration_authority"] != "monotonic_clock":
                raise PhaseTransactionError("phase ledger v3 duration authority changed")
            if status == "completion_candidate":
                if not record["output_hash"] or record["return_code"] != 0:
                    raise PhaseTransactionError("completion candidate is incomplete")
                candidates[phase] = record
            elif status == "completed":
                candidate = candidates.get(phase)
                if candidate is None:
                    raise PhaseTransactionError("completed phase lacks completion candidate")
                if record["completion_candidate_hash"] != candidate["current_record_hash"]:
                    raise PhaseTransactionError("phase completion candidate binding mismatch")
                if record["output_hash"] != candidate["output_hash"]:
                    raise PhaseTransactionError("phase completed output differs from candidate")
                terminal_by_phase[phase] = status
            elif status == "failed":
                if not record["failure_classification"]:
                    raise PhaseTransactionError("failed phase lacks classification")
                terminal_by_phase[phase] = status
    return {
        "status": "pass",
        "record_count": len(records),
        "completion_candidate_count": len(candidates),
        "terminal_phase_count": len(terminal_by_phase),
        "last_record_hash": previous,
    }


@dataclass(frozen=True)
class PhaseCommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class TransactionalPhaseRunner:
    """Append-only running -> candidate -> terminal phase transaction."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        run_identity_fingerprint: str,
        phase_order: Sequence[str],
        resume: bool = False,
        utc_clock: Callable[[], datetime] | None = None,
        monotonic_ns: Callable[[], int] | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.ledger_path = self.output_root / "phase_state.jsonl"
        self.run_identity_fingerprint = run_identity_fingerprint
        self.phase_order = tuple(phase_order)
        self._utc_clock = utc_clock or (lambda: datetime.now(timezone.utc))
        self._monotonic_ns = monotonic_ns or time.monotonic_ns
        if resume:
            if not self.ledger_path.is_file():
                raise PhaseTransactionError("phase resume requires an existing ledger")
            records = self.records()
            if any(
                record["run_identity_fingerprint"] != run_identity_fingerprint
                for record in records
            ):
                raise PhaseTransactionError("phase resume cross-run identity mismatch")
        else:
            if self.output_root.exists() and any(self.output_root.iterdir()):
                raise PhaseTransactionError("phase output root conflict")
            self.output_root.mkdir(parents=True, exist_ok=True)

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
                raise PhaseTransactionError(
                    f"invalid phase ledger v3 line {line_number}"
                ) from exc
            if not isinstance(item, dict):
                raise PhaseTransactionError("phase ledger v3 record must be an object")
            result.append(item)
        validate_phase_ledger_v3(result)
        return result

    def _append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        records = self.records()
        payload = {
            "formal_phase_ledger_version": FORMAL_PHASE_LEDGER_VERSION,
            "formal_phase_transaction_version": FORMAL_PHASE_TRANSACTION_VERSION,
            **dict(record),
            "sequence_number": len(records) + 1,
            "run_identity_fingerprint": self.run_identity_fingerprint,
            "previous_record_hash": (
                records[-1]["current_record_hash"] if records else None
            ),
        }
        payload["current_record_hash"] = _record_hash(payload)
        validate_phase_ledger_v3([*records, payload])
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        return payload

    def _outputs(self, patterns: Sequence[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for pattern in patterns:
            candidate = Path(pattern)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise PhaseTransactionError("phase output pattern escapes run root")
            matches = [
                path for path in sorted(self.output_root.glob(pattern)) if path.is_file()
            ]
            if not matches:
                raise PhaseTransactionError(f"phase output missing: {pattern}")
            for path in matches:
                result[path.relative_to(self.output_root).as_posix()] = _file_sha256(path)
        return result

    @staticmethod
    def _timing_none(started_at: datetime, monotonic_started_ns: int) -> dict[str, Any]:
        return {
            "started_at_utc": started_at.astimezone(timezone.utc).isoformat(),
            "completed_at_utc": None,
            "monotonic_started_ns": int(monotonic_started_ns),
            "monotonic_completed_ns": None,
            "wall_clock_seconds": None,
            "child_wall_clock_seconds": None,
            "finalization_wall_clock_seconds": None,
            "wall_clock_adjustment_seconds": None,
            "clock_consistency_status": None,
            "utc_timestamp_ordered": None,
            "duration_authority": None,
        }

    def _base(
        self,
        *,
        phase: str,
        status: str,
        input_hash: str,
        commands: Sequence[Sequence[str]],
        timing: Mapping[str, Any],
        return_code: int | None,
        failure_classification: str | None,
        output_files: Mapping[str, str] | None,
        output_hash: str | None,
        completion_candidate_hash: str | None,
        **extra: Any,
    ) -> dict[str, Any]:
        command_rows = [list(command) for command in commands]
        return {
            "phase": phase,
            "status": status,
            "input_hash": input_hash,
            "command_identity": _canonical_sha256(command_rows),
            "commands": command_rows,
            **dict(timing),
            "return_code": return_code,
            "failure_classification": failure_classification,
            "output_files": dict(output_files or {}),
            "output_hash": output_hash,
            "completion_candidate_hash": completion_candidate_hash,
            **extra,
        }

    def _check_order(self, phase: str) -> None:
        if phase not in self.phase_order:
            raise PhaseTransactionError(f"unknown phase: {phase}")
        completed = {
            record["phase"]
            for record in self.records()
            if record["status"] == "completed"
        }
        for predecessor in self.phase_order[: self.phase_order.index(phase)]:
            if predecessor not in completed:
                raise PhaseTransactionError(
                    f"phase order violation: {predecessor} before {phase}"
                )

    def run_phase(
        self,
        phase: str,
        *,
        commands: Sequence[Sequence[str]],
        input_hash: str,
        expected_outputs: Sequence[str],
        executor: Callable[[Sequence[str]], PhaseCommandResult] | None = None,
        infrastructure_retries: int = 0,
        stop_after_completion_candidate: bool = False,
        fail_terminal_append: bool = False,
    ) -> dict[str, Any]:
        if infrastructure_retries not in {0, 1}:
            raise PhaseTransactionError("infrastructure retry must be zero or one")
        self._check_order(phase)
        records = self.records()
        terminal = [
            record
            for record in records
            if record["phase"] == phase and record["status"] in {"completed", "failed"}
        ]
        if terminal:
            record = terminal[-1]
            if record["status"] == "failed":
                raise PhaseTransactionError("failed phase is terminal")
            if record["input_hash"] != input_hash:
                raise PhaseTransactionError("completed phase input drift")
            if _canonical_sha256([list(item) for item in commands]) != record[
                "command_identity"
            ]:
                raise PhaseTransactionError("completed phase command drift")
            current = self._outputs(expected_outputs)
            if _canonical_sha256(current) != record["output_hash"]:
                raise PhaseTransactionError("completed phase output drift")
            return {"status": "skipped_completed", "phase": phase}
        candidates = [
            record
            for record in records
            if record["phase"] == phase and record["status"] == "completion_candidate"
        ]
        if candidates:
            raise PhaseTransactionError("phase has completion candidate; use finalize-phase-only")
        started_at = self._utc_clock()
        started_ns = int(self._monotonic_ns())
        command_rows = [list(command) for command in commands]
        self._append(
            self._base(
                phase=phase,
                status="running",
                input_hash=input_hash,
                commands=command_rows,
                timing=self._timing_none(started_at, started_ns),
                return_code=None,
                failure_classification=None,
                output_files=None,
                output_hash=None,
                completion_candidate_hash=None,
            )
        )
        run = executor or self._subprocess
        child_seconds = 0.0
        for command_index, command in enumerate(command_rows):
            retry_index = 0
            while True:
                child_start = int(self._monotonic_ns())
                try:
                    result = run(command)
                except BaseException as exc:  # ledger an executor crash before propagating.
                    child_seconds += max(0, int(self._monotonic_ns()) - child_start) / 1e9
                    completed_at = self._utc_clock()
                    completed_ns = int(self._monotonic_ns())
                    timing = clock_measurement(
                        started_at_utc=started_at,
                        completed_at_utc=completed_at,
                        monotonic_started_ns=started_ns,
                        monotonic_completed_ns=completed_ns,
                        child_wall_clock_seconds=min(
                            child_seconds, max(0, completed_ns - started_ns) / 1e9
                        ),
                    )
                    self._append(
                        self._base(
                            phase=phase,
                            status="failed",
                            input_hash=input_hash,
                            commands=command_rows,
                            timing=timing,
                            return_code=None,
                            failure_classification=f"executor_exception:{type(exc).__name__}",
                            output_files=None,
                            output_hash=None,
                            completion_candidate_hash=None,
                            failed_command_index=command_index,
                        )
                    )
                    raise
                child_seconds += max(0, int(self._monotonic_ns()) - child_start) / 1e9
                if result.returncode == 75 and retry_index < infrastructure_retries:
                    retry_index += 1
                    continue
                break
            if result.returncode != 0:
                completed_at = self._utc_clock()
                completed_ns = int(self._monotonic_ns())
                timing = clock_measurement(
                    started_at_utc=started_at,
                    completed_at_utc=completed_at,
                    monotonic_started_ns=started_ns,
                    monotonic_completed_ns=completed_ns,
                    child_wall_clock_seconds=min(
                        child_seconds, max(0, completed_ns - started_ns) / 1e9
                    ),
                )
                event = self._append(
                    self._base(
                        phase=phase,
                        status="failed",
                        input_hash=input_hash,
                        commands=command_rows,
                        timing=timing,
                        return_code=result.returncode,
                        failure_classification="child_command_failure",
                        output_files=None,
                        output_hash=None,
                        completion_candidate_hash=None,
                        failed_command_index=command_index,
                        failure_message=(result.stderr or result.stdout)[-4000:],
                    )
                )
                raise PhaseTransactionError(f"phase failed: {phase}: {event['current_record_hash']}")
        output_files = self._outputs(expected_outputs) if expected_outputs else {}
        completed_at = self._utc_clock()
        completed_ns = int(self._monotonic_ns())
        timing = clock_measurement(
            started_at_utc=started_at,
            completed_at_utc=completed_at,
            monotonic_started_ns=started_ns,
            monotonic_completed_ns=completed_ns,
            child_wall_clock_seconds=min(
                child_seconds, max(0, completed_ns - started_ns) / 1e9
            ),
        )
        candidate = self._append(
            self._base(
                phase=phase,
                status="completion_candidate",
                input_hash=input_hash,
                commands=command_rows,
                timing=timing,
                return_code=0,
                failure_classification=None,
                output_files=output_files,
                output_hash=_canonical_sha256(output_files),
                completion_candidate_hash=None,
            )
        )
        if stop_after_completion_candidate:
            return {"status": "completion_candidate_created", "phase": phase}
        if fail_terminal_append:
            raise PhaseTransactionError("simulated terminal append failure")
        return self.finalize_phase_only(
            phase,
            commands=command_rows,
            input_hash=input_hash,
            expected_outputs=expected_outputs,
            candidate=candidate,
        )

    def finalize_phase_only(
        self,
        phase: str,
        *,
        commands: Sequence[Sequence[str]],
        input_hash: str,
        expected_outputs: Sequence[str],
        candidate: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        records = self.records()
        completed = [
            record
            for record in records
            if record["phase"] == phase and record["status"] == "completed"
        ]
        if completed:
            current = self._outputs(expected_outputs) if expected_outputs else {}
            if _canonical_sha256(current) != completed[-1]["output_hash"]:
                raise PhaseTransactionError("finalized phase output drift")
            return {"status": "already_finalized", "phase": phase}
        candidates = [
            record
            for record in records
            if record["phase"] == phase and record["status"] == "completion_candidate"
        ]
        selected = dict(candidate or (candidates[-1] if candidates else {}))
        if not selected:
            raise PhaseTransactionError("finalize-phase-only requires a completion candidate")
        command_rows = [list(command) for command in commands]
        if selected["input_hash"] != input_hash:
            raise PhaseTransactionError("finalize-phase-only input drift")
        if selected["command_identity"] != _canonical_sha256(command_rows):
            raise PhaseTransactionError("finalize-phase-only command drift")
        output_files = self._outputs(expected_outputs) if expected_outputs else {}
        output_hash = _canonical_sha256(output_files)
        if output_hash != selected["output_hash"]:
            raise PhaseTransactionError("finalize-phase-only output drift")
        event = self._append(
            self._base(
                phase=phase,
                status="completed",
                input_hash=input_hash,
                commands=command_rows,
                timing={
                    key: selected[key]
                    for key in (
                        "started_at_utc",
                        "completed_at_utc",
                        "monotonic_started_ns",
                        "monotonic_completed_ns",
                        "wall_clock_seconds",
                        "child_wall_clock_seconds",
                        "finalization_wall_clock_seconds",
                        "wall_clock_adjustment_seconds",
                        "clock_consistency_status",
                        "utc_timestamp_ordered",
                        "duration_authority",
                    )
                },
                return_code=0,
                failure_classification=None,
                output_files=output_files,
                output_hash=output_hash,
                completion_candidate_hash=selected["current_record_hash"],
                terminal_commit_mode="finalize_phase_only",
            )
        )
        return event

    @staticmethod
    def _subprocess(command: Sequence[str]) -> PhaseCommandResult:
        completed = subprocess.run(
            list(command), text=True, capture_output=True, check=False
        )
        return PhaseCommandResult(
            completed.returncode, completed.stdout, completed.stderr
        )


__all__ = [
    "FORMAL_PHASE_LEDGER_VERSION",
    "FORMAL_PHASE_TRANSACTION_VERSION",
    "PHASE_ABSOLUTE_SANITY_SECONDS",
    "PHASE_STATUSES",
    "PhaseCommandResult",
    "PhaseTransactionError",
    "TransactionalPhaseRunner",
    "clock_measurement",
    "validate_phase_ledger_v3",
]
