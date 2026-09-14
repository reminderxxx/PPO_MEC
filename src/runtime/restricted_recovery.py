"""Bounded G14R20-I5 recovery identity and immutable-source consumer.

This is deliberately not a general recovery framework.  It recognizes one
failed evaluation-only run, one phase, and two ordered cells.  The source run,
its ledgers, committed cells, failed attempt, staging, and lock are read-only.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from scripts.continuation_executor.cells import cell_layout
from scripts.continuation_executor.identity import digest as continuation_digest
from scripts.continuation_executor.locking import writer_lock_path
from src.evaluators.formal_cell_transaction import (
    CellExecutionIdentity,
    CellTransactionError,
    FormalCellLedger,
    artifact_inventory,
    stable_cell_id,
    validate_producer_integrity_manifests,
)
from src.evaluators.formal_phase_transaction import validate_phase_ledger_v3
from src.runtime.evaluation_only_execution import (
    EvaluationOnlyError,
    build_evaluation_execution_contract,
    canonical_sha256,
    file_sha256,
    validate_model_source_reference,
)
from src.runtime.resolved_formal_execution_context import (
    validate_resolved_formal_execution_context,
)


RESTRICTED_RECOVERY_CONTRACT_VERSION = "1.0.0"
RESTRICTED_RECOVERY_REQUEST_VERSION = "1.0.0"
RESTRICTED_RECOVERY_PHASE_LEDGER_VERSION = "1.0.0"
RESTRICTED_RECOVERY_HANDOFF_VERSION = "1.0.0"
ORIGINAL_RUN_ID = "typed_model_cache_evaluation_only_20260913_g14r20_i3_pending"
ORIGINAL_PROJECT_ROOT = Path("/Users/howen/Projects/PPO_MEC")
ORIGINAL_RUN_ROOT = (
    ORIGINAL_PROJECT_ROOT
    / "artifacts/experiments/typed_model_cache_evaluation_only"
    / ORIGINAL_RUN_ID
)
ORIGINAL_REQUEST_PATH = (
    ORIGINAL_PROJECT_ROOT
    / "artifacts/analysis/g14r20_i3_producer_integrity_20260913"
    / "authorization_request_unsigned.json"
)
ORIGINAL_PROJECT_GRANT_PATH = (
    ORIGINAL_PROJECT_ROOT
    / "artifacts/analysis/g14e03_authorization_audit_20260913/project_grant.json"
)
ORIGINAL_SUPPLEMENTAL_AUTHORIZATION_PATH = (
    ORIGINAL_PROJECT_ROOT
    / "artifacts/analysis/g14e03_host_retry_authorization_20260913"
    / "supplemental_authorization.json"
)
ORIGINAL_FAILURE_REPORT_PATH = (
    ORIGINAL_SUPPLEMENTAL_AUTHORIZATION_PATH.parent
    / "formal_ablation_failure_report.json"
)
I4_PLAN_PATH = Path(__file__).resolve().parents[2] / "docs/project/g14r20_i4_restricted_recovery_plan.md"
I4_DRAFT_PATH = Path(__file__).resolve().parents[2] / "docs/project/g14r20_i4_recovery_authorization_draft.json"
CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/experiment/g14r20_i5_restricted_recovery_v1/restricted_recovery_contract.json"
)
DECLARED_I4_AUDIT_BUNDLE = (
    ORIGINAL_PROJECT_ROOT
    / "artifacts/analysis/g14r20_i4_bundle_root_recovery_20260914"
)
PHASE = "formal_ablation"
FAILED_CELL_ID = "formal_ablation-3e9322fac172fcae01f2cc58"
UNSTARTED_CELL_ID = "formal_ablation-e40a9c86c8fdbf3b7962a689"
ALLOWED_CELL_IDS = (FAILED_CELL_ID, UNSTARTED_CELL_ID)
EXPECTED_PREFIXES = {
    "phase": {
        "filename": "phase_state.jsonl",
        "byte_count": 168899,
        "record_count": 8,
        "prefix_sha256": "a12731365111da83422ad4062ab482b08f3ee30c967084f6a32f4369bbce9adb",
        "terminal_hash": "bfd7f4071b58d3a6f9cb61355dcfb6785c39ab490bfa4e90ebe34ac1807bb362",
        "terminal_status": "failed",
    },
    "cell": {
        "filename": "cell_state.jsonl",
        "byte_count": 4090388,
        "record_count": 14,
        "prefix_sha256": "574ca9084db9ce377507edf7d210854dd69f575eed7135f4d567a35895310016",
        "terminal_hash": "012ba1af0174f723352cf1ff7fb5aad262f80c2c6be0371cb34efee4050d149a",
        "terminal_status": "failed_terminal",
    },
}
EXPECTED_SOURCE_REFERENCE_SHA256 = "0ba289caa663d38dd35f61feafa509a3f8dc2eb4269b3220f18d383be81c3ea3"
EXPECTED_LOCK_OWNER_SHA256 = "0cfeecba26816805d6445b9513263f183a3db651c5bdbbb9e204431b9872d11f"
EXPECTED_FAILED_STDERR_SHA256 = "51b2569c1b7c98c950ba2f509122613f66606a2821368b0756007b828e3db414"
EXPECTED_ORIGINAL_EXECUTOR_COMMIT = "a028ea291941a484ae9cd2e316d1b52adde3d1f2"


class RestrictedRecoveryError(ValueError):
    """A bounded recovery identity, source, scope, or ordering check failed."""


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise RestrictedRecoveryError(f"required immutable JSON is missing: {target}")
    try:
        value = json.loads(
            target.read_text(encoding="utf-8-sig"),
            parse_constant=lambda item: (_ for _ in ()).throw(
                RestrictedRecoveryError(f"non-finite JSON constant: {item}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RestrictedRecoveryError(f"invalid immutable JSON: {target}") from exc
    if not isinstance(value, dict):
        raise RestrictedRecoveryError(f"immutable JSON must be an object: {target}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RestrictedRecoveryError(f"required ledger is missing: {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RestrictedRecoveryError(f"invalid ledger line {number}: {path}") from exc
        if not isinstance(value, dict):
            raise RestrictedRecoveryError("ledger record must be an object")
        rows.append(value)
    return rows


def _file_row(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RestrictedRecoveryError(f"immutable file is missing: {path}")
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "sha256": file_sha256(path),
        "inode": stat.st_ino,
    }


def _canonical_reference(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    return {
        **_file_row(path),
        "canonical_sha256": canonical_sha256(value),
    }


def _assert_exact_path(observed: str | Path, expected: Path, label: str) -> Path:
    supplied = Path(observed)
    if not supplied.is_absolute() or supplied.is_symlink() or supplied.resolve() != expected.resolve():
        raise RestrictedRecoveryError(f"{label} path identity drift")
    return supplied.resolve()


def _assert_disjoint_recovery_root(root: Path, recovery_id: str) -> None:
    if not root.is_absolute() or root.name != recovery_id or ".." in root.parts:
        raise RestrictedRecoveryError("recovery run ID/root mismatch")
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise RestrictedRecoveryError("recovery root path may not contain a symlink")
    resolved = root.resolve(strict=False)
    original = ORIGINAL_RUN_ROOT.resolve()
    source = Path(
        "/Users/howen/Projects/PPO_MEC/artifacts/experiments/typed_model_cache_formal/"
        "typed_model_cache_formal_20260906_152847_g14c_v16"
    ).resolve()
    for protected in (original, source):
        if resolved == protected or protected in resolved.parents or resolved in protected.parents:
            raise RestrictedRecoveryError("recovery output overlaps a protected source root")


def _validate_contract_document() -> dict[str, Any]:
    contract = _read_json(CONTRACT_PATH)
    if (
        contract.get("restricted_recovery_contract_version")
        != RESTRICTED_RECOVERY_CONTRACT_VERSION
        or contract.get("original_run_id") != ORIGINAL_RUN_ID
        or contract.get("allowed_phase") != PHASE
        or [row.get("cell_id") for row in contract.get("allowed_cells", [])]
        != list(ALLOWED_CELL_IDS)
        or contract.get("immutable_original_prefixes")
        != {
            kind: {key: value for key, value in expected.items() if key != "filename"}
            for kind, expected in EXPECTED_PREFIXES.items()
        }
        or contract.get("external_committed_cell_count") != 6
        or contract.get("source_model_count") != 150
        or contract.get("source_reference_sha256")
        != EXPECTED_SOURCE_REFERENCE_SHA256
        or contract.get("automatic_retry_count") != 0
        or any(
            contract.get(field) is not False
            for field in (
                "external_results_copied",
                "later_phases_authorized",
                "training_authorized",
                "dev_selection_authorized",
                "checkpoint_freeze_authorized",
                "holdout_capability",
            )
        )
    ):
        raise RestrictedRecoveryError("versioned restricted recovery contract drift")
    return contract


def _prefix_audit(path: Path, kind: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = EXPECTED_PREFIXES[kind]
    data = path.read_bytes()
    rows = _jsonl(path)
    if (
        len(data) != expected["byte_count"]
        or len(rows) != expected["record_count"]
        or hashlib.sha256(data).hexdigest() != expected["prefix_sha256"]
    ):
        raise RestrictedRecoveryError(f"original {kind} ledger full prefix drift")
    terminal_field = "current_record_hash" if kind == "phase" else "current_ledger_hash"
    if (
        rows[-1].get(terminal_field) != expected["terminal_hash"]
        or rows[-1].get("status") != expected["terminal_status"]
    ):
        raise RestrictedRecoveryError(f"original {kind} terminal drift")
    return rows, {
        "path": str(path),
        "byte_count": len(data),
        "record_count": len(rows),
        "prefix_sha256": hashlib.sha256(data).hexdigest(),
        "terminal_hash": rows[-1][terminal_field],
        "terminal_status": rows[-1]["status"],
    }


def _old_cell_ledger(root: Path) -> FormalCellLedger:
    identity_payload = _read_json(root / "cell_ledger_identity.json")
    try:
        identity = CellExecutionIdentity(**identity_payload["identity"])
    except (KeyError, TypeError) as exc:
        raise RestrictedRecoveryError("original cell ledger identity drift") from exc
    return FormalCellLedger(run_root=root, identity=identity, resume=True)


def audit_original_recovery_source(
    *,
    original_request_path: str | Path = ORIGINAL_REQUEST_PATH,
    original_project_grant_path: str | Path = ORIGINAL_PROJECT_GRANT_PATH,
    original_run_root: str | Path = ORIGINAL_RUN_ROOT,
) -> dict[str, Any]:
    """Recompute the exact I3 source boundary without writing any source object."""

    _validate_contract_document()
    request_path = _assert_exact_path(original_request_path, ORIGINAL_REQUEST_PATH, "original request")
    grant_path = _assert_exact_path(original_project_grant_path, ORIGINAL_PROJECT_GRANT_PATH, "original grant")
    root = _assert_exact_path(original_run_root, ORIGINAL_RUN_ROOT, "original run")
    request = _read_json(request_path)
    request_projection = {key: value for key, value in request.items() if key != "authorization_request_sha256"}
    if request.get("authorization_request_sha256") != canonical_sha256(request_projection):
        raise RestrictedRecoveryError("original authorization request hash drift")
    source = _read_json(root / "evaluation_model_source_reference.json")
    if source.get("source_reference_sha256") != EXPECTED_SOURCE_REFERENCE_SHA256:
        raise RestrictedRecoveryError("original model source reference identity drift")
    try:
        validate_model_source_reference(source)
    except EvaluationOnlyError as exc:
        raise RestrictedRecoveryError(str(exc)) from exc
    execution = _read_json(root / "evaluation_execution_contract.json")
    if (
        request.get("model_source_reference") != source
        or request.get("evaluation_execution_contract") != execution
        or execution.get("evaluation_run_id") != ORIGINAL_RUN_ID
        or execution.get("executor_commit") != EXPECTED_ORIGINAL_EXECUTOR_COMMIT
    ):
        raise RestrictedRecoveryError("original request/run/executor identity drift")
    grant = _read_json(grant_path)
    if (
        grant.get("status") != "AUTHORIZED_FOR_EVALUATION_ONLY"
        or grant.get("authorization_request_sha256") != request["authorization_request_sha256"]
        or grant.get("evaluation_run_id") != ORIGINAL_RUN_ID
        or grant.get("executor_commit") != EXPECTED_ORIGINAL_EXECUTOR_COMMIT
        or grant.get("holdout_capability") is not False
    ):
        raise RestrictedRecoveryError("original project grant identity drift")
    supplemental = _read_json(ORIGINAL_SUPPLEMENTAL_AUTHORIZATION_PATH)
    if (
        supplemental.get("evaluation_run_id") != ORIGINAL_RUN_ID
        or supplemental.get("failure_cell_recovery") is not False
        or supplemental.get("failure_phase_recovery") is not False
        or supplemental.get("holdout_capability") is not False
    ):
        raise RestrictedRecoveryError("original supplemental authorization drift")

    phase_rows, phase_prefix = _prefix_audit(root / "phase_state.jsonl", "phase")
    validate_phase_ledger_v3(phase_rows)
    cell_rows, cell_prefix = _prefix_audit(root / "cell_state.jsonl", "cell")
    ledger = _old_cell_ledger(root)
    committed = ledger.committed_records()
    if len(committed) != 6 or {row["cell_id"] for row in committed} != set(
        ALLOWED_ORIGINAL_COMMITTED_CELL_IDS
    ):
        raise RestrictedRecoveryError("original committed-cell membership drift")
    external_cells: list[dict[str, Any]] = []
    for record in committed:
        path = Path(record["committed_path"])
        marker_path = path / "committed_marker.json"
        marker = _read_json(marker_path)
        producer = validate_producer_integrity_manifests(path)
        inventory = artifact_inventory(path)
        if marker.get("artifact_inventory_sha256") != canonical_sha256(inventory):
            raise RestrictedRecoveryError("external committed transaction inventory drift")
        external_cells.append(
            {
                "origin": "external_original_run",
                "original_run_id": ORIGINAL_RUN_ID,
                "phase": record["phase"],
                "cell_id": record["cell_id"],
                "coordinates": record["coordinates"],
                "committed_path": str(path),
                "source_cell_terminal_hash": record["current_ledger_hash"],
                "source_attempt": int(record["attempt"]),
                "transaction_inventory_sha256": record["artifact_inventory_sha256"],
                "transaction_file_count": len(inventory),
                "committed_marker_sha256": file_sha256(marker_path),
                "producer_integrity_manifest_count": producer["manifest_count"],
                "producer_integrity_manifests": producer["manifests"],
                "retention_eligible": True,
                "selection_basis": "identity_and_dual_integrity_only",
                "dispatched_by_recovery": False,
            }
        )
    external_cells.sort(key=lambda row: (row["phase"], row["cell_id"]))

    failed = [row for row in cell_rows if row.get("cell_id") == FAILED_CELL_ID]
    if [row.get("status") for row in failed] != ["running", "failed_terminal"] or any(
        int(row.get("attempt", 0)) != 1 for row in failed
    ):
        raise RestrictedRecoveryError("original failed attempt boundary drift")
    if any(row.get("cell_id") == UNSTARTED_CELL_ID for row in cell_rows):
        raise RestrictedRecoveryError("original unstarted cell unexpectedly has a ledger record")
    staging = Path(failed[-1]["staging_path"])
    staging_files = sorted(path for path in staging.iterdir() if path.is_file())
    if {path.name for path in staging_files} != {"cell_stdout.log", "cell_stderr.log"}:
        raise RestrictedRecoveryError("original failed staging membership drift")
    if (
        file_sha256(staging / "cell_stdout.log") != hashlib.sha256(b"").hexdigest()
        or file_sha256(staging / "cell_stderr.log") != EXPECTED_FAILED_STDERR_SHA256
    ):
        raise RestrictedRecoveryError("original failed staging payload drift")

    lock_path = writer_lock_path(root)
    lock = _read_json(lock_path)
    if lock.get("state") != "held" or continuation_digest(lock) != EXPECTED_LOCK_OWNER_SHA256:
        raise RestrictedRecoveryError("original held-lock owner drift")
    process = lock.get("process", {})
    try:
        process_probe = subprocess.run(
            ["ps", "-p", str(process.get("pid", "")), "-o", "lstart="],
            text=True,
            capture_output=True,
            check=False,
        )
        process_return_code: int | None = process_probe.returncode
        process_start = process_probe.stdout.strip() or None
        process_error = process_probe.stderr.strip() or None
    except OSError as exc:
        # Process visibility is an external authorization precondition.  Its
        # absence must not weaken source integrity or imply quiescence.
        process_return_code = None
        process_start = None
        process_error = f"{type(exc).__name__}: {exc}"
    return {
        "status": "pass",
        "qualification_scope": "retention_integrity_only_no_performance_claim",
        "original_request": _canonical_reference(request_path),
        "original_project_grant": _canonical_reference(grant_path),
        "original_supplemental_authorization": _canonical_reference(
            ORIGINAL_SUPPLEMENTAL_AUTHORIZATION_PATH
        ),
        "original_failure_report": _canonical_reference(ORIGINAL_FAILURE_REPORT_PATH),
        "source_reference": _file_row(root / "evaluation_model_source_reference.json"),
        "source_reference_sha256": source["source_reference_sha256"],
        "source_model_count": len(source["models"]),
        "phase_ledger_prefix": phase_prefix,
        "cell_ledger_prefix": cell_prefix,
        "external_committed_cells": external_cells,
        "failed_cell": {
            "cell_id": FAILED_CELL_ID,
            "source_attempt": 1,
            "source_terminal_hash": failed[-1]["current_ledger_hash"],
            "source_status": "failed_terminal",
            "staging_path": str(staging),
            "staging_files": [_file_row(path) for path in staging_files],
        },
        "unstarted_cell": {
            "cell_id": UNSTARTED_CELL_ID,
            "source_attempt_count": 0,
        },
        "held_lock": {
            **_file_row(lock_path),
            "owner_canonical_sha256": continuation_digest(lock),
            "owner_state": lock["state"],
            "pid_observation": {
                "pid": process.get("pid"),
                "ps_return_code": process_return_code,
                "live_start_time": process_start,
                "probe_error": process_error,
                "quiescence_proven": False,
                "lock_cleanup_authorized": False,
            },
        },
        "i4_review_inputs": {
            "plan": _file_row(I4_PLAN_PATH),
            "unsigned_draft": _file_row(I4_DRAFT_PATH),
            "declared_audit_bundle": {
                "path": str(DECLARED_I4_AUDIT_BUNDLE),
                "present": DECLARED_I4_AUDIT_BUNDLE.is_dir(),
                "used_as_evidence": False,
                "note": "source objects were independently revalidated",
            },
        },
        "recovery_grant_issued": False,
        "real_recovery_started": False,
        "holdout_opened": False,
    }


def stable_source_audit_projection(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the explicitly non-authoritative live PID observation."""

    value = deepcopy(dict(audit))
    held = value.get("held_lock")
    if isinstance(held, dict):
        held.pop("pid_observation", None)
    return value


ALLOWED_ORIGINAL_COMMITTED_CELL_IDS = (
    "formal_cache_policy-458f1a514d3de2a9855adae8",
    "formal_cache_policy-57ea8942f9af2530278a6888",
    "formal_cache_policy-b995253ce76fcb798c98cf9e",
    "formal_controller-ca75f4fd6cba10d56d8210cb",
    "formal_controller-4931a055d167dfcf76cf6ded",
    "formal_controller-d81f46ce6883684144cfe40d",
)


_LOCAL_PATH_FLAGS = {
    "--output-root": "RECOVERY_OUTPUT_ROOT",
    "--repository-root": "EXECUTOR_ROOT",
    "--resolved-execution-context-path": "RECOVERY_CONTEXT",
    "--evaluation-model-source-reference-path": "SOURCE_REFERENCE_COPY",
}
_MIRROR_FLAGS = {
    "--model-cache-runtime-config",
    "--cache-baseline-fairness-manifest-path",
}


def _command_scientific_projection(
    command: Sequence[str], *, executor_root: Path
) -> dict[str, Any]:
    values = list(command)
    if len(values) < 2:
        raise RestrictedRecoveryError("recovery command is incomplete")
    projected: list[Any] = ["PYTHON_EXECUTOR", "FORMAL_SUPPORT_ENTRYPOINT"]
    index = 2
    while index < len(values):
        token = values[index]
        if token in _LOCAL_PATH_FLAGS:
            if index + 1 >= len(values):
                raise RestrictedRecoveryError("recovery command flag lacks a value")
            projected.extend([token, _LOCAL_PATH_FLAGS[token]])
            index += 2
            continue
        if token in _MIRROR_FLAGS:
            if index + 1 >= len(values):
                raise RestrictedRecoveryError("recovery mirror flag lacks a value")
            path = Path(values[index + 1])
            try:
                relative = path.resolve().relative_to(executor_root.resolve()).as_posix()
            except ValueError as exc:
                raise RestrictedRecoveryError("executor mirror path escapes checkout") from exc
            projected.extend(
                [token, {"logical_path": relative, "sha256": file_sha256(path), "size_bytes": path.stat().st_size}]
            )
            index += 2
            continue
        projected.append(token)
        index += 1
    return {"projection": projected, "sha256": canonical_sha256(projected)}


def _build_recovery_execution(
    *,
    source_reference: Mapping[str, Any],
    original_execution: Mapping[str, Any],
    recovery_execution_id: str,
    recovery_root: Path,
    executor_checkout: Path,
    executor_commit: str,
    python_executable: Path,
    created_at_utc: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = build_evaluation_execution_contract(
        source_reference=source_reference,
        evaluation_run_id=recovery_execution_id,
        evaluation_run_root=recovery_root,
        executor_checkout=executor_checkout,
        executor_commit=executor_commit,
        python_executable=python_executable,
    )
    plan = deepcopy(base["command_plans"][PHASE])
    cell_ids = [stable_cell_id(PHASE, row) for row in plan["matrix_contexts"]]
    if cell_ids != list(ALLOWED_CELL_IDS) or len(plan["commands"]) != 2:
        raise RestrictedRecoveryError("frozen ablation cell membership/order drift")
    context = deepcopy(base["evaluation_execution_context"])
    plan_sha = canonical_sha256(plan)
    context["created_at_utc"] = created_at_utc
    context["created_for_run_identity"] = canonical_sha256(
        {
            "recovery_execution_id": recovery_execution_id,
            "original_run_id": ORIGINAL_RUN_ID,
            "executor_commit": executor_commit,
            "source_reference_sha256": source_reference["source_reference_sha256"],
            "command_plan_sha256": plan_sha,
            "allowed_phase": PHASE,
            "allowed_cell_ids": list(ALLOWED_CELL_IDS),
        }
    )
    context["command_expansion"].update(
        outer_expansion_sha256=plan_sha,
        resolved_command_matrix_sha256=plan_sha,
        phase_count=1,
        command_count=2,
    )
    context["evaluation_execution_identity"].update(
        evaluation_run_id=recovery_execution_id,
        executor_commit=executor_commit,
        executor_git_tree=base["executor_git_tree"],
        recovery_contract_version=RESTRICTED_RECOVERY_CONTRACT_VERSION,
        original_run_id=ORIGINAL_RUN_ID,
        allowed_phase=PHASE,
        allowed_cell_ids=list(ALLOWED_CELL_IDS),
    )
    context["context_sha256"] = canonical_sha256(
        {key: value for key, value in context.items() if key != "context_sha256"}
    )
    protocol = _read_json(source_reference["protocol_path"])
    validate_resolved_formal_execution_context(
        context,
        protocol=protocol,
        clean_worktree_root=executor_checkout,
        durable_run_root=recovery_root,
        check_git=True,
    )
    original_plan = original_execution["command_plans"][PHASE]
    old_executor = Path(original_execution["executor_checkout"])
    original_projections = [
        _command_scientific_projection(command, executor_root=old_executor)
        for command in original_plan["commands"]
    ]
    recovery_projections = [
        _command_scientific_projection(command, executor_root=executor_checkout)
        for command in plan["commands"]
    ]
    if original_projections != recovery_projections:
        raise RestrictedRecoveryError("scientific command parameters changed during recovery binding")
    scientific = context["scientific_identity"]
    invariance = {
        "status": "pass",
        "scientific_parameters_unchanged": True,
        "original_command_plan_sha256": canonical_sha256(original_plan),
        "recovery_command_plan_sha256": plan_sha,
        "normalized_original_commands": original_projections,
        "normalized_recovery_commands": recovery_projections,
        "workload_binding": {
            "split_semantic_sha256": scientific["split_semantic_sha256"],
            "window_contract_semantic_sha256": scientific["window_contract_semantic_sha256"],
            "catalog_fingerprint": scientific["catalog_fingerprint"],
        },
        "agent_order_binding": scientific.get("formal_agent_order_contract_semantic_sha256"),
        "seed_order": [7, 13, 29, 43, 71],
        "capacity": "medium_576mb",
        "model_source_reference_sha256": source_reference["source_reference_sha256"],
        "model_count": len(source_reference["models"]),
        "protocol_semantic_sha256": source_reference["protocol_semantic_sha256"],
        "exposure_context_identity_sha256": source_reference["source_context_identity_sha256"],
        "nullable_metric_contract_sha256": scientific.get(
            "formal_nullable_metric_aggregation_contract_semantic_sha256"
        ),
        "statistics_rules_source": "unchanged_frozen_protocol_not_executed_in_this_scope",
        "endpoint_or_matrix_selection_change": False,
        "holdout_opened": False,
    }
    execution = {
        "restricted_recovery_contract_version": RESTRICTED_RECOVERY_CONTRACT_VERSION,
        "recovery_execution_id": recovery_execution_id,
        "recovery_root": str(recovery_root),
        "original_run_id": ORIGINAL_RUN_ID,
        "executor_checkout": str(executor_checkout),
        "executor_commit": executor_commit,
        "executor_git_tree": base["executor_git_tree"],
        "python_executable": str(python_executable),
        "model_source_reference_sha256": source_reference["source_reference_sha256"],
        "allowed_phase": PHASE,
        "allowed_cell_ids": list(ALLOWED_CELL_IDS),
        "cell_specs": [
            {
                "cell_id": FAILED_CELL_ID,
                "source_attempt": 1,
                "recovery_attempt": 2,
                "execution_kind": "recovery_attempt",
                "must_commit_before_next_cell": True,
            },
            {
                "cell_id": UNSTARTED_CELL_ID,
                "source_attempt_count": 0,
                "recovery_attempt": 1,
                "execution_kind": "first_execution",
                "dispatch_after_cell_id": FAILED_CELL_ID,
            },
        ],
        "command_plan": plan,
        "command_plan_sha256": plan_sha,
        "resolved_execution_context": context,
        "resolved_execution_context_sha256": context["context_sha256"],
        "phase_ledger_path": str(recovery_root / "recovery_phase_state.jsonl"),
        "cell_ledger_path": str(recovery_root / "cell_state.jsonl"),
        "staging_root": str(recovery_root / ".staging"),
        "lock_path": str(writer_lock_path(recovery_root)),
        "external_results_copied": False,
        "later_phases_authorized": False,
        "training_authorized": False,
        "dev_selection_authorized": False,
        "checkpoint_freeze_authorized": False,
        "holdout_capability": False,
        "automatic_retry_count": 0,
    }
    execution["recovery_execution_identity_sha256"] = canonical_sha256(execution)
    return execution, invariance


def build_restricted_recovery_request(
    *,
    original_request_path: str | Path,
    original_project_grant_path: str | Path,
    original_run_root: str | Path,
    recovery_execution_id: str,
    recovery_root: str | Path,
    executor_checkout: str | Path,
    executor_commit: str,
    python_executable: str | Path,
    created_at_utc: str | None = None,
    allow_existing_recovery_root: bool = False,
) -> dict[str, Any]:
    source_audit = audit_original_recovery_source(
        original_request_path=original_request_path,
        original_project_grant_path=original_project_grant_path,
        original_run_root=original_run_root,
    )
    root = Path(recovery_root)
    _assert_disjoint_recovery_root(root, recovery_execution_id)
    if root.exists() and not allow_existing_recovery_root:
        raise RestrictedRecoveryError("unsigned recovery request requires a new output root")
    executor = Path(executor_checkout)
    if not executor.is_absolute() or executor.is_symlink() or not executor.is_dir():
        raise RestrictedRecoveryError("recovery executor checkout identity drift")
    head = subprocess.run(
        ["git", "-C", str(executor), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    tree = subprocess.run(
        ["git", "-C", str(executor), "rev-parse", "HEAD^{tree}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if head.returncode or tree.returncode or head.stdout.strip() != executor_commit:
        raise RestrictedRecoveryError("recovery executor commit drift")
    status = subprocess.run(
        [
            "git", "-C", str(executor), "status", "--porcelain",
            "--untracked-files=all", "--", "README.md", "docs", "scripts", "src", "tests", "configs",
        ],
        text=True,
        capture_output=True,
        check=False,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
    )
    if status.returncode or status.stdout.strip():
        raise RestrictedRecoveryError("recovery executor code/config/documentation is not clean")
    python = Path(python_executable)
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise RestrictedRecoveryError("recovery Python executable is missing")
    source = _read_json(ORIGINAL_RUN_ROOT / "evaluation_model_source_reference.json")
    original_execution = _read_json(ORIGINAL_RUN_ROOT / "evaluation_execution_contract.json")
    created = created_at_utc or datetime.now(timezone.utc).isoformat()
    execution, invariance = _build_recovery_execution(
        source_reference=source,
        original_execution=original_execution,
        recovery_execution_id=recovery_execution_id,
        recovery_root=root,
        executor_checkout=executor.resolve(),
        executor_commit=executor_commit,
        python_executable=python.resolve(),
        created_at_utc=created,
    )
    request: dict[str, Any] = {
        "restricted_recovery_authorization_request_version": RESTRICTED_RECOVERY_REQUEST_VERSION,
        "status": "READY_FOR_RESTRICTED_RECOVERY_AUTHORIZATION",
        "authorization_kind": "restricted_formal_ablation_recovery_only",
        "created_at_utc": created,
        "original_identity": {
            "run_id": ORIGINAL_RUN_ID,
            "request": source_audit["original_request"],
            "project_grant": source_audit["original_project_grant"],
            "supplemental_authorization": source_audit[
                "original_supplemental_authorization"
            ],
            "failure_report": source_audit["original_failure_report"],
            "source_reference_sha256": source_audit["source_reference_sha256"],
            "source_executor_commit": EXPECTED_ORIGINAL_EXECUTOR_COMMIT,
        },
        "immutable_source_audit": source_audit,
        "external_committed_cells": source_audit["external_committed_cells"],
        "source_models": deepcopy(source["models"]),
        "recovery_execution": execution,
        "scientific_invariance": invariance,
        "source_mapping_and_deduplication": {
            "origin_field_required": True,
            "logical_identity": ["phase", "cell_id"],
            "external_original_count": 6,
            "recovery_execution_count_after_completion": 2,
            "expected_unique_cell_count_after_completion": 8,
            "duplicate_logical_cells_allowed": False,
            "performance_based_retention_allowed": False,
            "external_cells_may_be_dispatched_or_rewritten": False,
        },
        "later_stage_boundary": {
            "handoff_only_after_two_ablation_commits": True,
            "formal_support_authorized": False,
            "formal_scalability_authorized": False,
            "formal_statistics_authorized": False,
            "formal_gate_authorized": False,
            "completion_authorized": False,
            "automatic_next_phase": False,
        },
        "blocking_preconditions_for_real_execution": [
            "independent acceptance of this final clean executor commit/tree",
            "fresh exact-prefix, six-cell dual-integrity, 150-model and held-lock revalidation",
            "independent quiescence and kernel-lock evidence without deleting the old inode",
            "separate project-owner grant for the exact request hash and two-cell scope",
        ],
        "recovery_grant_issued": False,
        "real_recovery_started": False,
        "holdout_opened": False,
    }
    request["authorization_request_sha256"] = canonical_sha256(request)
    return request


def validate_restricted_recovery_request(
    request: Mapping[str, Any], *, check_live: bool = True
) -> dict[str, Any]:
    _validate_contract_document()
    value = dict(request)
    supplied_hash = value.pop("authorization_request_sha256", None)
    if supplied_hash != canonical_sha256(value):
        raise RestrictedRecoveryError("restricted recovery request hash mismatch")
    if (
        value.get("restricted_recovery_authorization_request_version")
        != RESTRICTED_RECOVERY_REQUEST_VERSION
        or value.get("status") != "READY_FOR_RESTRICTED_RECOVERY_AUTHORIZATION"
        or value.get("authorization_kind")
        != "restricted_formal_ablation_recovery_only"
    ):
        raise RestrictedRecoveryError("restricted recovery request schema/status drift")
    if any(value.get(name) is not False for name in (
        "recovery_grant_issued", "real_recovery_started", "holdout_opened"
    )):
        raise RestrictedRecoveryError("unsigned recovery request claims forbidden authority")
    execution = value.get("recovery_execution")
    if not isinstance(execution, Mapping):
        raise RestrictedRecoveryError("recovery execution identity is missing")
    execution_value = dict(execution)
    identity_hash = execution_value.pop("recovery_execution_identity_sha256", None)
    if identity_hash != canonical_sha256(execution_value):
        raise RestrictedRecoveryError("recovery execution identity hash mismatch")
    if (
        execution.get("original_run_id") != ORIGINAL_RUN_ID
        or execution.get("allowed_phase") != PHASE
        or execution.get("allowed_cell_ids") != list(ALLOWED_CELL_IDS)
        or execution.get("holdout_capability") is not False
        or execution.get("later_phases_authorized") is not False
        or execution.get("automatic_retry_count") != 0
    ):
        raise RestrictedRecoveryError("recovery execution scope drift")
    specs = execution.get("cell_specs")
    if not isinstance(specs, list) or len(specs) != 2:
        raise RestrictedRecoveryError("recovery cell specification drift")
    if (
        specs[0].get("cell_id") != FAILED_CELL_ID
        or specs[0].get("source_attempt") != 1
        or specs[0].get("recovery_attempt") != 2
        or specs[1].get("cell_id") != UNSTARTED_CELL_ID
        or specs[1].get("source_attempt_count") != 0
        or specs[1].get("recovery_attempt") != 1
        or specs[1].get("dispatch_after_cell_id") != FAILED_CELL_ID
    ):
        raise RestrictedRecoveryError("recovery attempt/order identity drift")
    root = Path(str(execution.get("recovery_root", "")))
    _assert_disjoint_recovery_root(root, str(execution.get("recovery_execution_id", "")))
    plan = execution.get("command_plan")
    if not isinstance(plan, Mapping) or execution.get("command_plan_sha256") != canonical_sha256(plan):
        raise RestrictedRecoveryError("recovery command plan drift")
    if [stable_cell_id(PHASE, row) for row in plan.get("matrix_contexts", [])] != list(
        ALLOWED_CELL_IDS
    ) or len(plan.get("commands", [])) != 2:
        raise RestrictedRecoveryError("recovery command/cell membership drift")
    for command in plan["commands"]:
        if "--output-root" not in command:
            raise RestrictedRecoveryError("recovery command lacks output root")
        output = Path(command[command.index("--output-root") + 1])
        if output != root / PHASE or ORIGINAL_RUN_ROOT.resolve() in output.resolve().parents:
            raise RestrictedRecoveryError("recovery command targets a protected output path")
        if "--formal-window-split" not in command or command[
            command.index("--formal-window-split") + 1
        ] != "formal":
            raise RestrictedRecoveryError("recovery command split drift")
        if any("holdout" in token.lower() or "hidden" in token.lower() for token in command):
            raise RestrictedRecoveryError("recovery command opens holdout/hidden capability")
    context = execution.get("resolved_execution_context")
    if (
        not isinstance(context, Mapping)
        or execution.get("resolved_execution_context_sha256") != context.get("context_sha256")
        or context.get("context_sha256")
        != canonical_sha256({key: item for key, item in context.items() if key != "context_sha256"})
    ):
        raise RestrictedRecoveryError("recovery resolved context drift")
    if value.get("scientific_invariance", {}).get("scientific_parameters_unchanged") is not True:
        raise RestrictedRecoveryError("scientific invariance evidence is missing")
    mapping = value.get("source_mapping_and_deduplication", {})
    if (
        mapping.get("external_original_count") != 6
        or mapping.get("recovery_execution_count_after_completion") != 2
        or mapping.get("expected_unique_cell_count_after_completion") != 8
        or mapping.get("duplicate_logical_cells_allowed") is not False
        or mapping.get("external_cells_may_be_dispatched_or_rewritten") is not False
    ):
        raise RestrictedRecoveryError("source mapping/deduplication policy drift")
    if len(value.get("external_committed_cells", [])) != 6 or len(value.get("source_models", [])) != 150:
        raise RestrictedRecoveryError("external cell/model membership drift")
    if check_live:
        rebuilt = build_restricted_recovery_request(
            original_request_path=value["original_identity"]["request"]["path"],
            original_project_grant_path=value["original_identity"]["project_grant"]["path"],
            original_run_root=ORIGINAL_RUN_ROOT,
            recovery_execution_id=execution["recovery_execution_id"],
            recovery_root=root,
            executor_checkout=execution["executor_checkout"],
            executor_commit=execution["executor_commit"],
            python_executable=execution["python_executable"],
            created_at_utc=value["created_at_utc"],
            allow_existing_recovery_root=True,
        )
        if dict(request) != rebuilt:
            raise RestrictedRecoveryError("recovery request differs from live immutable sources")
    return {
        "status": "pass",
        "authorization_request_sha256": supplied_hash,
        "external_committed_cell_count": 6,
        "source_model_count": 150,
        "allowed_cell_count": 2,
    }


def validate_recovery_executor_live(request: Mapping[str, Any]) -> dict[str, Any]:
    """Recheck the new executor/context/command side without consuming sources."""

    validate_restricted_recovery_request(request, check_live=False)
    execution = request["recovery_execution"]
    executor = Path(execution["executor_checkout"])
    if not executor.is_absolute() or executor.is_symlink() or not executor.is_dir():
        raise RestrictedRecoveryError("live recovery executor checkout drift")
    head = subprocess.run(
        ["git", "-C", str(executor), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    tree = subprocess.run(
        ["git", "-C", str(executor), "rev-parse", "HEAD^{tree}"],
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        [
            "git", "-C", str(executor), "status", "--porcelain",
            "--untracked-files=all", "--", "README.md", "docs", "scripts", "src", "tests", "configs",
        ],
        text=True,
        capture_output=True,
        check=False,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
    )
    if (
        head.returncode
        or tree.returncode
        or status.returncode
        or head.stdout.strip() != execution["executor_commit"]
        or tree.stdout.strip() != execution["executor_git_tree"]
        or status.stdout.strip()
    ):
        raise RestrictedRecoveryError("live recovery executor commit/tree/clean-scope drift")
    python = Path(execution["python_executable"])
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise RestrictedRecoveryError("live recovery Python executable drift")
    source = _read_json(ORIGINAL_RUN_ROOT / "evaluation_model_source_reference.json")
    protocol = _read_json(source["protocol_path"])
    validate_resolved_formal_execution_context(
        execution["resolved_execution_context"],
        protocol=protocol,
        clean_worktree_root=executor,
        durable_run_root=execution["recovery_root"],
        check_git=True,
    )
    from scripts import run_typed_model_cache_formal_support

    projections = []
    for command in execution["command_plan"]["commands"]:
        if (
            len(command) < 2
            or command[0] != str(python)
            or command[1]
            != str(executor / "scripts/run_typed_model_cache_formal_support.py")
        ):
            raise RestrictedRecoveryError("live recovery command entrypoint drift")
        run_typed_model_cache_formal_support.build_parser().parse_args(command[2:])
        projections.append(_command_scientific_projection(command, executor_root=executor))
    if projections != request["scientific_invariance"]["normalized_recovery_commands"]:
        raise RestrictedRecoveryError("live recovery scientific projection drift")
    return {
        "status": "pass",
        "executor_commit": execution["executor_commit"],
        "executor_git_tree": execution["executor_git_tree"],
        "parsed_command_count": 2,
    }


def recovery_cell_identity(request: Mapping[str, Any]) -> CellExecutionIdentity:
    execution = request["recovery_execution"]
    source = request["immutable_source_audit"]
    original_identity = _read_json(ORIGINAL_RUN_ROOT / "cell_ledger_identity.json")["identity"]
    return CellExecutionIdentity(
        run_id=execution["recovery_execution_id"],
        execution_commit=execution["executor_commit"],
        protocol_semantic_sha256=original_identity["protocol_semantic_sha256"],
        resource_registry_semantic_sha256=original_identity[
            "resource_registry_semantic_sha256"
        ],
        environment_fingerprint=hashlib.sha256(
            execution["python_executable"].encode("utf-8")
        ).hexdigest(),
        split_semantic_sha256=original_identity["split_semantic_sha256"],
        window_contract_semantic_sha256=original_identity[
            "window_contract_semantic_sha256"
        ],
        catalog_fingerprint=original_identity["catalog_fingerprint"],
        runtime_identity=original_identity["runtime_identity"],
        command_matrix_sha256=execution["command_plan_sha256"],
    )


class RestrictedRecoveryCellLedger(FormalCellLedger):
    """The normal transaction ledger with exactly attempts 2 then 1."""

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
        if phase != PHASE or cell_id not in ALLOWED_CELL_IDS:
            raise CellTransactionError("cell is outside restricted recovery scope")
        records = self.records()
        if records:
            terminal_failures = [row for row in records if row["status"] == "failed_terminal"]
            active = self._active_running(records)
            if terminal_failures or active:
                raise CellTransactionError("restricted recovery failure/interruption is terminal")
        existing = self._records_for(cell_id)
        if existing:
            raise CellTransactionError("restricted recovery cell cannot be redispatched")
        if cell_id == UNSTARTED_CELL_ID:
            first = [
                row for row in records
                if row["cell_id"] == FAILED_CELL_ID and row["status"] == "committed"
            ]
            if len(first) != 1:
                raise CellTransactionError("failed source cell must commit before the unstarted cell")
        elif any(row["cell_id"] == UNSTARTED_CELL_ID for row in records):
            raise CellTransactionError("recovery cell order drift")
        attempt = 2 if cell_id == FAILED_CELL_ID else 1
        command_hash = canonical_sha256(list(command))
        target = Path(committed_path) if committed_path else (
            self.run_root / "cells" / phase / cell_id
        )
        try:
            target.resolve(strict=False).relative_to(self.run_root.resolve())
        except ValueError as exc:
            raise CellTransactionError("restricted recovery committed path escapes recovery root") from exc
        staging = self.run_root / ".staging" / phase / cell_id / f"attempt_{attempt:02d}"
        if staging.exists() or target.exists():
            raise CellTransactionError("restricted recovery staging/output path conflict")
        staging.mkdir(parents=True, exist_ok=False)
        started = self._utc_clock()
        return {
            "status": "execute",
            "cell_id": cell_id,
            "record": self._append(
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
                    retry_allowed=False,
                    staging_path=staging,
                    committed_path=target,
                )
            ),
        }

    def fail_cell(
        self,
        cell_id: str,
        *,
        return_code: int | None,
        classification: str,
        retryable: bool,
    ) -> dict[str, Any]:
        return super().fail_cell(
            cell_id,
            return_code=return_code,
            classification=classification,
            retryable=False,
        )


def _recovery_phase_hash(record: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in record.items() if key != "current_record_hash"}
    )


@dataclass
class RestrictedRecoveryPhaseLedger:
    path: Path
    recovery_identity_sha256: str

    def records(self) -> list[dict[str, Any]]:
        rows = _jsonl(self.path) if self.path.exists() else []
        previous = None
        events: list[str] = []
        for index, row in enumerate(rows, 1):
            if (
                row.get("restricted_recovery_phase_ledger_version")
                != RESTRICTED_RECOVERY_PHASE_LEDGER_VERSION
                or row.get("sequence_number") != index
                or row.get("recovery_identity_sha256") != self.recovery_identity_sha256
                or row.get("phase") != PHASE
                or row.get("previous_record_hash") != previous
                or row.get("current_record_hash") != _recovery_phase_hash(row)
            ):
                raise RestrictedRecoveryError("restricted recovery phase ledger drift")
            previous = row["current_record_hash"]
            events.append(str(row.get("event")))
        allowed = (
            [],
            ["started"],
            ["started", "prerequisite_committed"],
            ["started", "prerequisite_committed", "completed"],
            ["started", "failed"],
            ["started", "prerequisite_committed", "failed"],
        )
        if events not in allowed:
            raise RestrictedRecoveryError("restricted recovery phase event order drift")
        return rows

    def append(self, event: str, *, cell_id: str | None, detail: Mapping[str, Any] | None = None) -> dict[str, Any]:
        rows = self.records()
        if rows and rows[-1]["event"] in {"completed", "failed"}:
            raise RestrictedRecoveryError("restricted recovery phase terminal is immutable")
        payload = {
            "restricted_recovery_phase_ledger_version": RESTRICTED_RECOVERY_PHASE_LEDGER_VERSION,
            "sequence_number": len(rows) + 1,
            "recovery_identity_sha256": self.recovery_identity_sha256,
            "phase": PHASE,
            "event": event,
            "cell_id": cell_id,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "detail": dict(detail or {}),
            "previous_record_hash": rows[-1]["current_record_hash"] if rows else None,
        }
        payload["current_record_hash"] = _recovery_phase_hash(payload)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records()
        return payload


def build_recovery_handoff_manifest(
    request: Mapping[str, Any], ledger: RestrictedRecoveryCellLedger
) -> dict[str, Any]:
    validate_restricted_recovery_request(request, check_live=False)
    recovered = ledger.committed_records(phase=PHASE)
    if [row["cell_id"] for row in recovered] != list(ALLOWED_CELL_IDS):
        raise RestrictedRecoveryError("recovery handoff requires both ordered ablation commits")
    external = audit_original_recovery_source()["external_committed_cells"]
    rows: list[dict[str, Any]] = [deepcopy(row) for row in external]
    for record in recovered:
        path = Path(record["committed_path"])
        rows.append(
            {
                "origin": "new_recovery_execution",
                "recovery_execution_id": request["recovery_execution"]["recovery_execution_id"],
                "phase": record["phase"],
                "cell_id": record["cell_id"],
                "coordinates": record["coordinates"],
                "committed_path": str(path),
                "recovery_attempt": int(record["attempt"]),
                "transaction_inventory_sha256": record["artifact_inventory_sha256"],
                "transaction_file_count": len(record["artifact_inventory"]),
                "committed_marker_sha256": file_sha256(path / "committed_marker.json"),
                "producer_integrity": validate_producer_integrity_manifests(path),
                "dispatched_by_recovery": True,
            }
        )
    identities = [(row["phase"], row["cell_id"]) for row in rows]
    if len(rows) != 8 or len(set(identities)) != 8:
        raise RestrictedRecoveryError("recovery handoff contains duplicate logical results")
    controller_rows: list[str] = []
    for row in rows:
        if row["phase"] != "formal_controller":
            continue
        matches = list(Path(row["committed_path"]).rglob("benchmark_rows.csv"))
        if len(matches) != 1:
            raise RestrictedRecoveryError("external controller cell rows compatibility drift")
        controller_rows.append(str(matches[0].resolve()))
    manifest: dict[str, Any] = {
        "restricted_recovery_handoff_version": RESTRICTED_RECOVERY_HANDOFF_VERSION,
        "status": "ABLATION_HANDOFF_READY_REQUIRES_SEPARATE_AUTHORIZATION",
        "authorization_request_sha256": request["authorization_request_sha256"],
        "recovery_execution_identity_sha256": request["recovery_execution"][
            "recovery_execution_identity_sha256"
        ],
        "source_partitions": {
            "external_original_run": 6,
            "new_recovery_execution": 2,
        },
        "unique_logical_cell_count": 8,
        "cells": rows,
        "statistics_consumer_compatibility": {
            "status": "pass_explicit_path_resolution_only",
            "formal_controller_row_paths": sorted(controller_rows),
            "formal_controller_row_count": 3,
            "legacy_single_root_recursive_discovery_used": False,
            "formal_statistics_executed": False,
            "separate_statistics_authorization_required": True,
        },
        "next_stage_authorized": False,
        "recovery_grant_issued_for_next_stage": False,
        "real_recovery_started_for_next_stage": False,
        "holdout_opened": False,
    }
    manifest["handoff_sha256"] = canonical_sha256(manifest)
    return manifest


def validate_recovery_handoff_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(manifest)
    supplied = value.pop("handoff_sha256", None)
    if supplied != canonical_sha256(value):
        raise RestrictedRecoveryError("recovery handoff hash mismatch")
    rows = value.get("cells")
    if not isinstance(rows, list) or len(rows) != 8:
        raise RestrictedRecoveryError("recovery handoff cell count drift")
    identities = [(row.get("phase"), row.get("cell_id")) for row in rows]
    if len(set(identities)) != 8:
        raise RestrictedRecoveryError("recovery handoff duplicates a logical result")
    if value.get("source_partitions") != {
        "external_original_run": 6,
        "new_recovery_execution": 2,
    }:
        raise RestrictedRecoveryError("recovery handoff source partition drift")
    if value.get("next_stage_authorized") is not False or value.get("holdout_opened") is not False:
        raise RestrictedRecoveryError("recovery handoff claims unauthorized next-stage scope")
    return {"status": "pass", "handoff_sha256": supplied, "unique_cell_count": 8}


def restricted_cell_layout(request: Mapping[str, Any], cell_id: str):
    execution = request["recovery_execution"]
    plan = execution["command_plan"]
    index = list(ALLOWED_CELL_IDS).index(cell_id)
    command = plan["commands"][index]
    coordinates = plan["matrix_contexts"][index]
    native = type(
        "Native",
        (),
        {
            "stable_cell_id": staticmethod(stable_cell_id),
            "single_child_directory": staticmethod(
                __import__(
                    "src.evaluators.formal_cell_transaction",
                    fromlist=["single_child_directory"],
                ).single_child_directory
            ),
            "resolve_child_output_descriptor": staticmethod(
                __import__(
                    "src.evaluators.formal_cell_transaction",
                    fromlist=["resolve_child_output_descriptor"],
                ).resolve_child_output_descriptor
            ),
        },
    )
    final, builder, resolver = cell_layout(PHASE, coordinates, command, native)
    return command, coordinates, final, builder, resolver


__all__ = [
    "ALLOWED_CELL_IDS",
    "ALLOWED_ORIGINAL_COMMITTED_CELL_IDS",
    "FAILED_CELL_ID",
    "ORIGINAL_PROJECT_GRANT_PATH",
    "ORIGINAL_REQUEST_PATH",
    "ORIGINAL_RUN_ID",
    "ORIGINAL_RUN_ROOT",
    "PHASE",
    "RESTRICTED_RECOVERY_CONTRACT_VERSION",
    "RestrictedRecoveryCellLedger",
    "RestrictedRecoveryError",
    "RestrictedRecoveryPhaseLedger",
    "UNSTARTED_CELL_ID",
    "audit_original_recovery_source",
    "build_recovery_handoff_manifest",
    "build_restricted_recovery_request",
    "recovery_cell_identity",
    "restricted_cell_layout",
    "stable_source_audit_projection",
    "validate_recovery_handoff_manifest",
    "validate_recovery_executor_live",
    "validate_restricted_recovery_request",
]
