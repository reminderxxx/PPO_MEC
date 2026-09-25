"""Fail-closed transaction primitives for the one-time public holdout runner.

The production entry point is intentionally separate from the ordinary formal
runner.  Authorization is verified before any durable opening.  Once the
opening directory is atomically published, every outcome (success, child
failure, interruption, or partial staging output) is permanently consumed and
cannot be resumed or reopened.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


HOLDOUT_REQUEST_VERSION = "g14r22_unsigned_holdout_request_v2"
HOLDOUT_COMMAND_PACKAGE_VERSION = "g14r22_holdout_command_package_v2"
HOLDOUT_LEDGER_VERSION = "g14r22_one_time_opening_ledger_v1"
HOLDOUT_RECEIPT_VERSION = "g14r22_holdout_execution_receipt_v1"
HOLDOUT_INTEGRITY_VERSION = "g14r22_holdout_integrity_v1"
HOLDOUT_GRANT_VERSION = "g14r22_holdout_grant_v2"
GRANT_VALIDITY_CONTRACT_VERSION = "g14r22_grant_validity_v1"
MAX_GRANT_VALIDITY = timedelta(hours=72)
GRANT_VALIDITY_CONTRACT = {
    "contract_version": GRANT_VALIDITY_CONTRACT_VERSION,
    "maximum_validity_seconds": int(MAX_GRANT_VALIDITY.total_seconds()),
    "timezone_aware_timestamps_required": True,
    "issued_at_must_be_actual_issuance_time": True,
    "backdating_allowed": False,
    "silent_extension_allowed": False,
    "automatic_resign_allowed": False,
    "authorization_check_rule": "issued_at <= checked_at < expires_at",
    "atomic_open_rule": "grant must still be valid immediately before atomic rename",
    "post_open_expiry_rule": (
        "does_not_cancel_or_change_an_already_started_scientific_process; "
        "the opening remains permanently consumed"
    ),
}
LEARNED_AGENTS = (
    "sa_ghmappo", "ppo", "mappo", "dqn", "dueling_dqn", "qmix",
    "controller_mat", "dag_offload_drl", "cache_offload_drl", "dt_handoff_drl",
)
ALL_AGENTS = (
    "reactive_lru", "reactive_fifo", "reactive_lfu", "reactive_aging_lfu",
    "reactive_random", *LEARNED_AGENTS,
)
SEEDS = (7, 13, 29, 43, 71)
CAPACITIES = ("constrained_288mb", "medium_576mb", "relaxed_864mb")
PRIMARY_METRICS = (
    "full_service_ready_byte_hit_rate",
    "joint_base_adapter_hit_rate",
    "full_service_ready_request_rate",
    "transfer_mb_per_request",
    "workflow_continuity_rate",
    "end_to_end_workflow_delay",
)


class HoldoutExecutionError(ValueError):
    """Raised before or after opening when the one-time contract is violated."""


def utc_now() -> str:
    return _utc_now_datetime().isoformat()


def _utc_now_datetime() -> datetime:
    return datetime.now(timezone.utc)


def _aware_timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise HoldoutExecutionError(f"holdout grant {field} is not valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HoldoutExecutionError(f"holdout grant {field} must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def validate_grant_validity(
    grant: Mapping[str, Any], *, request: Mapping[str, Any], checked_at: datetime | None = None,
    boundary: str,
) -> dict[str, Any]:
    """Validate the fixed 72-hour authorization window at a named boundary."""

    now = checked_at if checked_at is not None else _utc_now_datetime()
    if now.tzinfo is None or now.utcoffset() is None:
        raise HoldoutExecutionError("grant validity check time must be timezone-aware")
    now = now.astimezone(timezone.utc)
    issued = _aware_timestamp(grant.get("issued_at"), "issued_at")
    expires = _aware_timestamp(grant.get("expires_at"), "expires_at")
    if expires <= issued:
        raise HoldoutExecutionError("holdout grant expires_at must be after issued_at")
    if expires - issued > MAX_GRANT_VALIDITY:
        raise HoldoutExecutionError("holdout grant validity exceeds the 72-hour maximum")
    requested_at = _aware_timestamp(request.get("created_at"), "request.created_at")
    if issued < requested_at:
        raise HoldoutExecutionError("holdout grant issued_at predates the exact unsigned request")
    if not issued <= now < expires:
        raise HoldoutExecutionError(f"holdout grant is future or expired at {boundary}")
    return {
        "status": "pass",
        "boundary": boundary,
        "checked_at": now.isoformat(),
        "issued_at": issued.isoformat(),
        "expires_at": expires.isoformat(),
        "validity_seconds": int((expires - issued).total_seconds()),
        "maximum_validity_seconds": int(MAX_GRANT_VALIDITY.total_seconds()),
    }


def canonical_sha256(value: Any) -> str:
    def reject(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise HoldoutExecutionError("non-finite contract value")
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise HoldoutExecutionError("contract keys must be strings")
                reject(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                reject(child)

    reject(value)
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8")
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise HoldoutExecutionError(f"required object is missing or a symlink: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HoldoutExecutionError(f"invalid JSON object: {target}") from exc
    if not isinstance(value, dict):
        raise HoldoutExecutionError(f"required object must be a mapping: {target}")
    return value


def _encoded(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def create_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_encoded(value))
        handle.flush()
        os.fsync(handle.fileno())


def _hash_bound(value: Mapping[str, Any], field: str) -> bool:
    payload = dict(value)
    observed = payload.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(payload)


def _validate_identity_bundle(package: Mapping[str, Any], *, required: bool) -> None:
    """Bind the three sibling files consumed by the real checkpoint loader."""

    contract = package.get("evaluation_execution_contract")
    context = package.get("resolved_execution_context")
    reference = package.get("model_source_reference")
    present = [isinstance(value, Mapping) for value in (contract, context, reference)]
    if not any(present):
        if required:
            raise HoldoutExecutionError("evaluation execution identity bundle is missing")
        return
    if not all(present):
        raise HoldoutExecutionError("evaluation execution identity bundle is incomplete")
    if contract.get("evaluation_execution_context") != context:
        raise HoldoutExecutionError("evaluation execution context/contract drift")
    if contract.get("evaluation_run_root") != package.get("output_root"):
        raise HoldoutExecutionError("evaluation execution contract output-root drift")
    if (contract.get("executor_checkout") != package.get("executor_checkout")
            or contract.get("executor_commit") != package.get("executor_commit")):
        raise HoldoutExecutionError("evaluation execution contract executor drift")
    if contract.get("model_source_reference_sha256") != reference.get(
        "source_reference_sha256"
    ):
        raise HoldoutExecutionError("evaluation execution contract source drift")
    if not _hash_bound(contract, "execution_contract_sha256"):
        raise HoldoutExecutionError("evaluation execution contract hash mismatch")
    if not _hash_bound(context, "context_sha256"):
        raise HoldoutExecutionError("resolved execution context hash mismatch")
    if not _hash_bound(reference, "source_reference_sha256"):
        raise HoldoutExecutionError("evaluation model source reference hash mismatch")


def validate_command_package(
    package: Mapping[str, Any], *, acceptance: bool = False, isolated_fixture: bool = False
) -> dict[str, Any]:
    if package.get("command_package_version") != HOLDOUT_COMMAND_PACKAGE_VERSION:
        raise HoldoutExecutionError("holdout command package version mismatch")
    if not _hash_bound(package, "command_package_sha256"):
        raise HoldoutExecutionError("holdout command package hash mismatch")
    matrix = package.get("scientific_matrix")
    if not isinstance(matrix, Mapping):
        raise HoldoutExecutionError("scientific matrix is missing")
    expected = {
        "agents": list(ALL_AGENTS),
        "learned_agents": list(LEARNED_AGENTS),
        "seeds": list(SEEDS),
        "capacities": list(CAPACITIES),
        "outer_windows": 12,
        "workflows": 3,
        "scientific_child_count": 3,
        "rows_per_child": 2700,
        "total_expected_rows": 8100,
        "checkpoint_count": 150,
        "primary_metrics": list(PRIMARY_METRICS),
        "holm_family_size": 84,
    }
    if isolated_fixture:
        if package.get("isolated_fixture_non_scientific") is not True:
            raise HoldoutExecutionError("isolated fixture marker is missing")
        if package.get("fixture_holdout_policy_runs") != 0:
            raise HoldoutExecutionError("isolated fixture must execute zero holdout policies")
        if package.get("execution_mode") != "formal_holdout_fixture":
            raise HoldoutExecutionError("isolated fixture execution mode mismatch")
        output = Path(str(package.get("output_root", "")))
        if not output.is_absolute() or "artifacts/experiments/typed_model_cache_holdout" in str(output):
            raise HoldoutExecutionError("isolated fixture output must not use the formal holdout root")
        serialized = json.dumps(package.get("commands"), sort_keys=True).lower()
        if "sealed_holdout" in serialized or "holdout_policy" in serialized:
            raise HoldoutExecutionError("isolated fixture contains a real holdout reference")
    elif not acceptance and dict(matrix) != expected:
        raise HoldoutExecutionError("production scientific matrix drift")
    if acceptance:
        if package.get("acceptance_non_holdout") is not True:
            raise HoldoutExecutionError("acceptance package must be explicitly non-holdout")
        serialized = json.dumps(package, sort_keys=True).lower()
        if "sealed_holdout" in serialized or "holdout_policy" in serialized:
            raise HoldoutExecutionError("acceptance package contains a holdout reference")
    phases = package.get("phases")
    expected_phases = ["scientific", "statistics", "publication", "integrity"]
    if phases != expected_phases:
        raise HoldoutExecutionError("holdout phase order drift")
    commands = package.get("commands")
    if not isinstance(commands, Mapping):
        raise HoldoutExecutionError("command mapping is missing")
    scientific = commands.get("scientific")
    if not isinstance(scientific, list) or len(scientific) != 3:
        raise HoldoutExecutionError("exactly three capacity-scoped scientific commands are required")
    if not all(isinstance(command, list) and command for command in scientific):
        raise HoldoutExecutionError("scientific command schema mismatch")
    if not isinstance(commands.get("statistics"), list) or not commands["statistics"]:
        raise HoldoutExecutionError("statistics command is missing")
    if not acceptance and not isolated_fixture:
        for index, command in enumerate(scientific):
            if command[command.index("--agents") + 1:command.index("--seeds")] != list(ALL_AGENTS):
                raise HoldoutExecutionError("scientific command agent order drift")
            seeds = command[command.index("--seeds") + 1:command.index("--seed_checkpoint_manifest_path")]
            if seeds != [str(seed) for seed in SEEDS]:
                raise HoldoutExecutionError("scientific command seed order drift")
            required_pairs = {
                "--formal_window_split": "sealed_holdout",
                "--window-plan-resource-id": "window_plan.typed_model_cache.sealed_holdout",
                "--output_root": "{G14R22_CELL_OUTPUT_ROOT}",
                "--dedicated-holdout-opening-receipt": "{G14R22_OPENING_RECEIPT}",
                "--dedicated-holdout-request-sha256": "{G14R22_REQUEST_SHA256}",
                "--dedicated-holdout-command-package-sha256": "{G14R22_COMMAND_PACKAGE_SHA256}",
                "--runtime-config-resource-id": f"runtime_config.{CAPACITIES[index]}",
                "--checkpoint-manifest-id": f"checkpoint_manifest.{CAPACITIES[index]}",
                "--checkpoint-provenance-id": f"checkpoint_provenance.{CAPACITIES[index]}",
            }
            for flag, expected_value in required_pairs.items():
                if command.count(flag) != 1 or command[command.index(flag) + 1] != expected_value:
                    raise HoldoutExecutionError(f"scientific command binding drift: {flag}")
            plan = command[command.index("--window_plan_path") + 1]
            if Path(plan).name != "sealed_holdout_window_plan.json":
                raise HoldoutExecutionError("scientific command window plan drift")
            lowered = " ".join(command).lower()
            if any(token in lowered for token in ("checkpoint_freeze", "dev_select", "train_agent")):
                raise HoldoutExecutionError("scientific command contains forbidden selection/training action")
        statistics = commands["statistics"]
        if statistics[statistics.index("--candidate_agent") + 1] != "sa_ghmappo":
            raise HoldoutExecutionError("statistics candidate drift")
        baselines = statistics[statistics.index("--baseline_agents") + 1:statistics.index("--metrics")]
        if baselines != [agent for agent in ALL_AGENTS if agent != "sa_ghmappo"]:
            raise HoldoutExecutionError("statistics baseline family drift")
        metrics = statistics[statistics.index("--metrics") + 1:statistics.index("--pair_keys")]
        if metrics != list(PRIMARY_METRICS):
            raise HoldoutExecutionError("statistics metric family drift")
        if statistics[statistics.index("--bootstrap_samples") + 1] != "10000":
            raise HoldoutExecutionError("statistics bootstrap budget drift")
    if package.get("automatic_retry_count") != 0:
        raise HoldoutExecutionError("automatic retry must remain zero")
    _validate_identity_bundle(package, required=not acceptance and not isolated_fixture)
    return {
        "status": "pass",
        "command_count": 4,
        "acceptance": acceptance,
        "isolated_fixture": isolated_fixture,
    }


def validate_unsigned_request(
    request: Mapping[str, Any], package: Mapping[str, Any], *, check_files: bool = True,
    isolated_fixture: bool = False,
) -> dict[str, Any]:
    if request.get("request_version") != HOLDOUT_REQUEST_VERSION:
        raise HoldoutExecutionError("unsigned request version mismatch")
    if not _hash_bound(request, "request_sha256"):
        raise HoldoutExecutionError("unsigned request hash mismatch")
    if request.get("status") != "READY_FOR_AUTHORIZATION_REVIEW":
        raise HoldoutExecutionError("unsigned request is not review-ready")
    for field in ("grant_signed", "execution_authorized", "holdout_opened", "holdout_consumed_permanently"):
        if request.get(field) is not False:
            raise HoldoutExecutionError(f"unsigned request falsely asserts {field}")
    if request.get("command_package_sha256") != package.get("command_package_sha256"):
        raise HoldoutExecutionError("request/command package binding mismatch")
    validate_command_package(package, isolated_fixture=isolated_fixture)
    if request.get("grant_validity_contract") != GRANT_VALIDITY_CONTRACT:
        raise HoldoutExecutionError("unsigned request grant-validity contract drift")
    _aware_timestamp(request.get("created_at"), "request.created_at")
    if request.get("failure_boundary") != {
        "before_atomic_open": "not_consumed; no scientific child may start",
        "at_or_after_atomic_open": "permanently_consumed",
        "child_failure": "terminal_failure_no_retry_no_resume_no_reopen",
        "partial_output": "retained_in_staging_and_permanently_consumed",
        "success": "permanently_consumed",
    }:
        raise HoldoutExecutionError("one-time failure boundary drift")
    if check_files:
        for row in request.get("frozen_inputs", []):
            target = Path(str(row.get("path", "")))
            if target.is_symlink() or not target.is_file():
                raise HoldoutExecutionError(f"frozen input missing: {target}")
            if target.stat().st_size != row.get("size_bytes") or file_sha256(target) != row.get("sha256"):
                raise HoldoutExecutionError(f"frozen input bytes drift: {target}")
    return {"status": "pass", "request_sha256": request["request_sha256"]}


def verify_checkpoint_bytes(source_reference_path: str | Path) -> dict[str, Any]:
    source = read_json(source_reference_path)
    models = source.get("models")
    if not isinstance(models, list) or len(models) != 150:
        raise HoldoutExecutionError("checkpoint source must contain exactly 150 models")
    expected_coordinates = {
        (capacity, agent, seed)
        for capacity in CAPACITIES for agent in LEARNED_AGENTS for seed in SEEDS
    }
    observed_coordinates: set[tuple[str, str, int]] = set()
    rows = []
    total_bytes = 0
    started = datetime.now(timezone.utc)
    for model in models:
        coordinate = (str(model.get("capacity_label")), str(model.get("agent")), int(model.get("seed")))
        if coordinate in observed_coordinates:
            raise HoldoutExecutionError(f"duplicate checkpoint coordinate: {coordinate}")
        observed_coordinates.add(coordinate)
        target = Path(str(model.get("checkpoint_path", "")))
        if target.is_symlink() or not target.is_file():
            raise HoldoutExecutionError(f"checkpoint is missing or a symlink: {target}")
        observed_size = target.stat().st_size
        observed_hash = file_sha256(target)
        if observed_size != model.get("size_bytes") or observed_hash != model.get("checkpoint_sha256"):
            raise HoldoutExecutionError(f"checkpoint byte identity drift: {target}")
        total_bytes += observed_size
        rows.append({
            "capacity_label": coordinate[0], "agent": coordinate[1], "seed": coordinate[2],
            "path": str(target), "size_bytes": observed_size, "sha256": observed_hash,
        })
    if observed_coordinates != expected_coordinates:
        raise HoldoutExecutionError("checkpoint coordinate scope drift")
    completed = datetime.now(timezone.utc)
    return {
        "checkpoint_byte_audit_version": "g14r22_opening_checkpoint_bytes_v1",
        "status": "pass",
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "actual_scope": {
            "models": 150, "learned_agents": list(LEARNED_AGENTS), "seeds": list(SEEDS),
            "capacities": list(CAPACITIES), "total_bytes": total_bytes,
        },
        "models": rows,
        "models_canonical_sha256": canonical_sha256(rows),
    }


def verify_grant(
    request: Mapping[str, Any], package: Mapping[str, Any], grant: Mapping[str, Any], token: bytes,
    *, checked_at: datetime | None = None, boundary: str = "authorization_check",
) -> dict[str, Any]:
    required = {
        "grant_version", "status", "grant_signed", "request_sha256",
        "command_package_sha256", "executor_commit", "output_root",
        "one_time_token_sha256", "independent_review", "issued_at", "expires_at",
        "grant_validity_contract",
    }
    if set(grant) != required or grant.get("grant_version") != HOLDOUT_GRANT_VERSION:
        raise HoldoutExecutionError("holdout grant schema mismatch")
    if grant.get("status") != "AUTHORIZED_ONE_TIME_HOLDOUT" or grant.get("grant_signed") is not True:
        raise HoldoutExecutionError("holdout grant is not signed/authorized")
    expected = {
        "request_sha256": request["request_sha256"],
        "command_package_sha256": package["command_package_sha256"],
        "executor_commit": package["executor_commit"],
        "output_root": package["output_root"],
        "one_time_token_sha256": hashlib.sha256(token).hexdigest(),
    }
    if any(grant.get(key) != value for key, value in expected.items()):
        raise HoldoutExecutionError("holdout grant identity/scope drift")
    if grant.get("grant_validity_contract") != GRANT_VALIDITY_CONTRACT:
        raise HoldoutExecutionError("holdout grant validity contract drift")
    validity = validate_grant_validity(
        grant, request=request, checked_at=checked_at, boundary=boundary
    )
    review = grant["independent_review"]
    if not isinstance(review, Mapping) or set(review) != {"path", "sha256", "size_bytes"}:
        raise HoldoutExecutionError("independent review reference invalid")
    review_path = Path(str(review["path"]))
    if (review_path.is_symlink() or not review_path.is_file()
            or review_path.stat().st_size != review["size_bytes"]
            or file_sha256(review_path) != review["sha256"]):
        raise HoldoutExecutionError("independent review bytes drift")
    review_payload = read_json(review_path)
    if (review_payload.get("status") != "pass"
            or review_payload.get("request_sha256") != request["request_sha256"]
            or review_payload.get("command_package_sha256") != package["command_package_sha256"]):
        raise HoldoutExecutionError("independent review does not approve the exact package")
    return {
        "status": "pass",
        "grant_sha256": canonical_sha256(grant),
        "validity": validity,
    }


def validate_opening_receipt(
    receipt_path: str | Path, *, request_sha256: str, command_package_sha256: str
) -> dict[str, Any]:
    receipt = read_json(receipt_path)
    if (receipt.get("ledger_version") != HOLDOUT_LEDGER_VERSION
            or receipt.get("event") != "opened"
            or receipt.get("consumed_permanently") is not True
            or receipt.get("request_sha256") != request_sha256
            or receipt.get("command_package_sha256") != command_package_sha256
            or receipt.get("checkpoint_byte_validation", {}).get("models") != 150
            or receipt.get("checkpoint_byte_validation", {}).get("status") != "pass"):
        raise HoldoutExecutionError("dedicated holdout opening receipt is invalid")
    return receipt


def _append_ledger(path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    previous = None
    sequence = 1
    if path.exists():
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        if lines:
            last = json.loads(lines[-1])
            previous = last.get("record_sha256")
            sequence = int(last.get("sequence", 0)) + 1
    record = {
        "ledger_version": HOLDOUT_LEDGER_VERSION,
        "sequence": sequence,
        "recorded_at": utc_now(),
        "previous_record_sha256": previous,
        **dict(event),
    }
    record["record_sha256"] = canonical_sha256(record)
    with path.open("ab") as handle:
        handle.write((json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    return record


def _replace_tokens(command: Sequence[str], replacements: Mapping[str, str]) -> list[str]:
    resolved = []
    for item in command:
        value = str(item)
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        if "{G14R22_" in value:
            raise HoldoutExecutionError(f"unresolved command token: {value}")
        resolved.append(value)
    return resolved


def _inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for target in sorted(root.rglob("*")):
        if target.is_symlink():
            raise HoldoutExecutionError(f"artifact symlink forbidden: {target}")
        if target.is_file():
            rows.append({
                "path": target.relative_to(root).as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": file_sha256(target),
            })
    return rows


def _verify_producer_manifest(artifact: Path) -> dict[str, Any]:
    manifest = read_json(artifact / "artifact_integrity_manifest.json")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise HoldoutExecutionError("scientific producer integrity manifest is empty")
    observed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "size_bytes", "sha256"}:
            raise HoldoutExecutionError("scientific producer integrity row schema mismatch")
        relative = Path(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise HoldoutExecutionError("scientific producer integrity path escapes artifact")
        target = artifact / relative
        if target.is_symlink() or not target.is_file():
            raise HoldoutExecutionError("scientific producer integrity file is missing")
        actual = {
            "path": relative.as_posix(),
            "size_bytes": target.stat().st_size,
            "sha256": file_sha256(target),
        }
        if actual != dict(row):
            raise HoldoutExecutionError("scientific producer integrity mismatch")
        observed.append(actual)
    if "benchmark_rows.csv" not in {row["path"] for row in observed}:
        raise HoldoutExecutionError("scientific producer manifest omits benchmark rows")
    return {
        "status": "pass",
        "file_count": len(observed),
        "files_canonical_sha256": canonical_sha256(observed),
    }


def _csv_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _run(command: Sequence[str], *, cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        completed = subprocess.run(
            list(command), cwd=cwd,
            env=dict(os.environ, PYTHONPATH=str(cwd), PYTHONNOUSERSITE="1"),
            stdout=stdout, stderr=stderr, check=False,
        )
    return int(completed.returncode)


def _verify_executor_checkout(package: Mapping[str, Any]) -> None:
    checkout = Path(str(package.get("executor_checkout", "")))
    if checkout.is_symlink() or not (checkout / ".git").exists():
        raise HoldoutExecutionError("executor checkout is missing or not a Git worktree")
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True, stderr=subprocess.STDOUT
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=checkout, text=True, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as exc:
        raise HoldoutExecutionError("executor Git identity check failed") from exc
    if head != package.get("executor_commit") or status:
        raise HoldoutExecutionError("executor checkout commit/cleanliness drift")


def execute_package(
    request: Mapping[str, Any], package: Mapping[str, Any], grant: Mapping[str, Any] | None,
    token: bytes | None, *, acceptance: bool = False, isolated_fixture: bool = False,
) -> dict[str, Any]:
    if acceptance and isolated_fixture:
        raise HoldoutExecutionError("acceptance and isolated fixture modes are mutually exclusive")
    validate_command_package(
        package, acceptance=acceptance, isolated_fixture=isolated_fixture
    )
    if not isolated_fixture and any(isinstance(package.get(field), Mapping) for field in (
        "evaluation_execution_contract", "resolved_execution_context", "model_source_reference"
    )):
        from src.runtime.evaluation_only_execution import validate_execution_contract

        validate_execution_contract(
            package["evaluation_execution_contract"],
            model_source_reference=package["model_source_reference"],
        )
    if not acceptance:
        _verify_executor_checkout(package)
        validate_unsigned_request(request, package, isolated_fixture=isolated_fixture)
        if grant is None or token is None:
            raise HoldoutExecutionError("signed grant and one-time token are required")
        grant_audit = verify_grant(
            request, package, grant, token, boundary="authorization_check"
        )
        checkpoint_audit = verify_checkpoint_bytes(request["checkpoint_source_reference"]["path"])
    else:
        grant_audit = {"status": "acceptance_non_holdout_no_grant"}
        checkpoint_audit = dict(package["acceptance_checkpoint_audit"])
    output_root = Path(str(package["output_root"]))
    if output_root.exists() or output_root.is_symlink():
        raise HoldoutExecutionError("one-time output root already exists; reopen/resume is forbidden")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.opening-", dir=output_root.parent))
    try:
        create_json(temporary / "checkpoint_byte_audit.json", checkpoint_audit)
        create_json(temporary / "request_snapshot.json", dict(request))
        create_json(temporary / "command_package_snapshot.json", dict(package))
        if isinstance(package.get("resolved_execution_context"), Mapping):
            create_json(temporary / "resolved_execution_context.json", package["resolved_execution_context"])
        if isinstance(package.get("model_source_reference"), Mapping):
            create_json(temporary / "evaluation_model_source_reference.json", package["model_source_reference"])
        if isinstance(package.get("evaluation_execution_contract"), Mapping):
            create_json(
                temporary / "evaluation_execution_contract.json",
                package["evaluation_execution_contract"],
            )
        atomic_open_grant_audit = grant_audit
        if not acceptance:
            assert grant is not None and token is not None
            atomic_open_grant_audit = verify_grant(
                request, package, grant, token, boundary="atomic_open"
            )
        opening = {
            "ledger_version": HOLDOUT_LEDGER_VERSION,
            "event": "opened",
            "opened_at": utc_now(),
            "request_sha256": request.get("request_sha256", package.get("acceptance_request_sha256")),
            "command_package_sha256": package["command_package_sha256"],
            "executor_commit": package["executor_commit"],
            "output_root": str(output_root),
            "grant_validation_at_authorization": grant_audit,
            "grant_validation_at_atomic_open": atomic_open_grant_audit,
            "grant_expiry_after_open_policy": GRANT_VALIDITY_CONTRACT["post_open_expiry_rule"],
            "checkpoint_byte_validation": {
                "status": checkpoint_audit["status"],
                "models": checkpoint_audit["actual_scope"]["models"],
                "total_bytes": checkpoint_audit["actual_scope"]["total_bytes"],
                "models_canonical_sha256": checkpoint_audit["models_canonical_sha256"],
            },
            "isolated_fixture_non_scientific": bool(isolated_fixture),
            "consumed_permanently": True,
            "retry_allowed": False,
            "resume_allowed": False,
            "reopen_allowed": False,
        }
        create_json(temporary / "opening_receipt.json", opening)
        _append_ledger(temporary / "opening_ledger.jsonl", opening)
        # This is the final expiry guard.  Once rename succeeds, TTL no longer
        # controls the lifetime of the already-started one-time execution.
        if not acceptance:
            assert grant is not None
            validate_grant_validity(
                grant, request=request, boundary="atomic_rename_guard"
            )
        os.replace(temporary, output_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    ledger = output_root / "opening_ledger.jsonl"
    published = output_root / "published"
    staging = output_root / "staging"
    published.mkdir()
    staging.mkdir()
    science_rows: list[str] = []
    terminal_status = "failed_permanently_consumed"
    failure: str | None = None
    try:
        for index, command in enumerate(package["commands"]["scientific"]):
            capacity = CAPACITIES[index]
            cell = staging / f"scientific_{capacity}"
            cell.mkdir()
            replacements = {
                "{G14R22_CELL_OUTPUT_ROOT}": str(cell),
                "{G14R22_OPENING_RECEIPT}": str(output_root / "opening_receipt.json"),
                "{G14R22_REQUEST_SHA256}": opening["request_sha256"],
                "{G14R22_COMMAND_PACKAGE_SHA256}": package["command_package_sha256"],
            }
            resolved = _replace_tokens(command, replacements)
            rc = _run(resolved, cwd=Path(package["executor_checkout"]),
                      stdout_path=cell / "child_stdout.log", stderr_path=cell / "child_stderr.log")
            _append_ledger(ledger, {"event": "scientific_child_terminal", "capacity": capacity,
                                    "return_code": rc, "command_sha256": canonical_sha256(resolved),
                                    "consumed_permanently": True, "retry_allowed": False})
            if rc != 0:
                raise HoldoutExecutionError(f"scientific child failed permanently: {capacity} rc={rc}")
            candidates = [path for path in cell.iterdir() if path.is_dir()]
            if len(candidates) != 1:
                raise HoldoutExecutionError(f"scientific child output is ambiguous: {capacity}")
            artifact = candidates[0]
            rows_path = artifact / "benchmark_rows.csv"
            producer_manifest = artifact / "artifact_integrity_manifest.json"
            if not rows_path.is_file() or not producer_manifest.is_file():
                raise HoldoutExecutionError(f"scientific child payload incomplete: {capacity}")
            producer_audit = _verify_producer_manifest(artifact)
            expected_rows = package.get("scientific_matrix", {}).get("rows_per_child")
            actual_rows = _csv_data_rows(rows_path)
            if expected_rows is not None and actual_rows != int(expected_rows):
                raise HoldoutExecutionError(
                    f"scientific child row count mismatch: {capacity} {actual_rows} != {expected_rows}"
                )
            _append_ledger(
                ledger,
                {
                    "event": "scientific_producer_integrity_verified",
                    "capacity": capacity,
                    "row_count": actual_rows,
                    **producer_audit,
                    "consumed_permanently": True,
                },
            )
            destination = published / "scientific" / capacity
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(artifact, destination)
            science_rows.append(str(destination / "benchmark_rows.csv"))
        stats_stage = staging / "statistics"
        stats_stage.mkdir()
        replacements = {"{G14R22_STATISTICS_OUTPUT_ROOT}": str(stats_stage)}
        for index, rows_path in enumerate(science_rows):
            replacements[f"{{G14R22_ROWS_{index}}}"] = rows_path
        stats_command = _replace_tokens(package["commands"]["statistics"], replacements)
        stats_rc = _run(stats_command, cwd=Path(package["executor_checkout"]),
                        stdout_path=stats_stage / "statistics_stdout.log",
                        stderr_path=stats_stage / "statistics_stderr.log")
        _append_ledger(ledger, {"event": "statistics_terminal", "return_code": stats_rc,
                                "command_sha256": canonical_sha256(stats_command),
                                "consumed_permanently": True, "retry_allowed": False})
        if stats_rc != 0:
            raise HoldoutExecutionError(f"statistics child failed permanently: rc={stats_rc}")
        stats_payload = read_json(stats_stage / "paired_statistics.json")
        rows = stats_payload.get("rows")
        expected_statistics_rows = int(package.get("scientific_matrix", {}).get("holm_family_size", 84))
        if not isinstance(rows, list) or len(rows) != expected_statistics_rows:
            raise HoldoutExecutionError(
                f"statistics output must contain exactly {expected_statistics_rows} Holm-family rows"
            )
        if any(row.get("holm_preregistered_family_size") != expected_statistics_rows for row in rows):
            raise HoldoutExecutionError("statistics Holm family identity drift")
        stats_destination = published / "statistics"
        os.replace(stats_stage, stats_destination)
        _append_ledger(ledger, {"event": "publication_completed", "scientific_children": 3,
                                "statistics_rows": expected_statistics_rows,
                                "consumed_permanently": True})
        inventory = _inventory(published)
        integrity = {
            "integrity_version": HOLDOUT_INTEGRITY_VERSION,
            "status": "pass", "file_count": len(inventory), "files": inventory,
            "files_canonical_sha256": canonical_sha256(inventory),
            "request_sha256": opening["request_sha256"],
            "command_package_sha256": package["command_package_sha256"],
            "consumed_permanently": True,
        }
        create_json(output_root / "artifact_integrity_manifest.json", integrity)
        _append_ledger(ledger, {"event": "integrity_completed",
                                "files_canonical_sha256": integrity["files_canonical_sha256"],
                                "consumed_permanently": True})
        terminal_status = "completed_permanently_consumed"
    except Exception as exc:
        failure = str(exc)
        _append_ledger(ledger, {"event": "terminal_failure", "failure": failure,
                                "consumed_permanently": True, "retry_allowed": False,
                                "resume_allowed": False, "reopen_allowed": False})
    receipt = {
        "receipt_version": HOLDOUT_RECEIPT_VERSION,
        "status": terminal_status,
        "completed_at": utc_now(),
        "request_sha256": opening["request_sha256"],
        "command_package_sha256": package["command_package_sha256"],
        "executor_commit": package["executor_commit"],
        "output_root": str(output_root),
        "failure": failure,
        "partial_output_retained": failure is not None,
        "consumed_permanently": True,
        "retry_allowed": False,
        "resume_allowed": False,
        "reopen_allowed": False,
        "acceptance_non_holdout": bool(acceptance),
        "holdout_opened": False if acceptance else True,
        "isolated_fixture_non_scientific": bool(isolated_fixture),
        "fixture_holdout_policy_runs": 0 if isolated_fixture else None,
    }
    create_json(output_root / "execution_receipt.json", receipt)
    _append_ledger(ledger, {"event": "execution_receipt_published", "status": terminal_status,
                            "receipt_sha256": file_sha256(output_root / "execution_receipt.json"),
                            "consumed_permanently": True})
    if failure is not None:
        raise HoldoutExecutionError(failure)
    return receipt
