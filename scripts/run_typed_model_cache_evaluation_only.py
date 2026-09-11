"""Execute one authorized G14R20-I evaluation-only phase.

No run state is created until an exact, externally issued project grant and its
independent review have both passed.  This entry has no training, selection,
checkpoint-freeze, recovery, or holdout action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.continuation_executor.cells import cell_layout
from scripts.continuation_executor.locking import SingleWriter
from src.evaluators.formal_cell_transaction import (
    CellExecutionIdentity,
    FormalCellLedger,
    execute_cell_artifact_transaction,
    stable_cell_id,
)
from src.evaluators.formal_phase_transaction import (
    PhaseCommandResult,
    TransactionalPhaseRunner,
)
from src.runtime.evaluation_only_execution import (
    PHASES,
    EvaluationOnlyError,
    canonical_sha256,
    file_sha256,
    validate_command_matrix_parsers,
    validate_execution_contract,
    validate_model_source_reference,
)
from scripts.run_typed_model_cache_formal_protocol import (
    validate_complete_without_holdout_gate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-request-path", required=True)
    parser.add_argument("--project-grant-path", required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--check", choices=("qualify", "execute"), default="qualify")
    return parser


def _read(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise EvaluationOnlyError(f"required authorization object is missing: {target}")
    value = json.loads(target.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise EvaluationOnlyError("authorization object must be a mapping")
    return value


def _utc(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise EvaluationOnlyError("invalid authorization timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EvaluationOnlyError("authorization timestamp requires timezone")
    return parsed.astimezone(timezone.utc)


def verify_project_grant(
    package: dict[str, Any], grant: dict[str, Any], *, now: datetime | None = None
) -> dict[str, Any]:
    """Validate the exact future owner grant; hashes provide integrity, not signatures."""

    expected_fields = {
        "version",
        "status",
        "authorization_request_sha256",
        "evaluation_run_id",
        "model_source_reference_sha256",
        "execution_contract_sha256",
        "executor_commit",
        "phases",
        "holdout_capability",
        "independent_review",
        "issued_at",
        "expires_at",
    }
    if set(grant) != expected_fields or grant.get("version") != "1.0.0":
        raise EvaluationOnlyError("evaluation-only project grant schema mismatch")
    execution = package["evaluation_execution_contract"]
    expected = {
        "status": "AUTHORIZED_FOR_EVALUATION_ONLY",
        "authorization_request_sha256": package["authorization_request_sha256"],
        "evaluation_run_id": execution["evaluation_run_id"],
        "model_source_reference_sha256": execution["model_source_reference_sha256"],
        "execution_contract_sha256": execution["execution_contract_sha256"],
        "executor_commit": execution["executor_commit"],
        "phases": list(PHASES),
        "holdout_capability": False,
    }
    if any(grant.get(key) != value for key, value in expected.items()):
        raise EvaluationOnlyError("evaluation-only project grant identity/scope drift")
    current = now or datetime.now(timezone.utc)
    if not _utc(grant["issued_at"]) <= current < _utc(grant["expires_at"]):
        raise EvaluationOnlyError("evaluation-only project grant is future or expired")
    review_row = grant["independent_review"]
    if not isinstance(review_row, dict) or set(review_row) != {
        "path", "sha256", "size_bytes"
    }:
        raise EvaluationOnlyError("independent review reference is invalid")
    review_path = Path(review_row["path"])
    if (
        review_path.is_symlink()
        or not review_path.is_file()
        or file_sha256(review_path) != review_row["sha256"]
        or review_path.stat().st_size != review_row["size_bytes"]
    ):
        raise EvaluationOnlyError("independent review bytes drift")
    review = _read(review_path)
    if (
        review.get("status") != "pass"
        or review.get("reviewer_id") == review.get("implementation_agent_id")
        or review.get("execution_contract_sha256") != execution["execution_contract_sha256"]
        or review.get("source_reference_sha256") != execution["model_source_reference_sha256"]
        or review.get("holdout_capability") is not False
    ):
        raise EvaluationOnlyError("independent evaluation-only review is incomplete")
    return {
        "approval_verified": True,
        "authorization_mode": "one_run_evaluation_only_project_grant_v1",
        "formal_execution_authorized": True,
        "holdout_capability": False,
        "grant_sha256": canonical_sha256(grant),
    }


def _encoded(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _create_file(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _cell_identity(package: dict[str, Any]) -> CellExecutionIdentity:
    execution = package["evaluation_execution_contract"]
    source = package["model_source_reference"]
    return CellExecutionIdentity(
        run_id=execution["evaluation_run_id"],
        execution_commit=execution["executor_commit"],
        protocol_semantic_sha256=source["protocol_semantic_sha256"],
        resource_registry_semantic_sha256=_read(source["protocol_path"])[
            "portable_resource_identity_contract"
        ]["resource_registry_semantic_sha256"],
        environment_fingerprint=hashlib.sha256(
            execution["python_executable"].encode("utf-8")
        ).hexdigest(),
        split_semantic_sha256=_read(source["protocol_path"])["identity"][
            "split_semantic_sha256"
        ],
        window_contract_semantic_sha256=_read(source["protocol_path"])[
            "execution_contract"
        ]["window_consumption_contract"]["semantic_sha256"],
        catalog_fingerprint=_read(source["protocol_path"])["identity"]["catalog_fingerprint"],
        runtime_identity=canonical_sha256(
            _read(source["protocol_path"])["identity"][
                "typed_runtime_contract_hashes_by_capacity"
            ]
        ),
        command_matrix_sha256=execution["command_plan_sha256"],
    )


def _phase_runner(package: dict[str, Any], *, resume: bool) -> TransactionalPhaseRunner:
    execution = package["evaluation_execution_contract"]
    return TransactionalPhaseRunner(
        output_root=execution["evaluation_run_root"],
        run_identity_fingerprint=canonical_sha256({
            "evaluation_execution_contract": execution["execution_contract_sha256"],
            "model_source_reference": package["model_source_reference"]["source_reference_sha256"],
        }),
        phase_order=PHASES, resume=resume,
        resolved_execution_context_sha256=execution["evaluation_execution_context_sha256"],
        resolved_execution_context_file_sha256=hashlib.sha256(
            _encoded(execution["evaluation_execution_context"])
        ).hexdigest(),
    )


def _initial_files(package: dict[str, Any]) -> dict[str, Any]:
    execution = package["evaluation_execution_contract"]
    return {
        "evaluation_model_source_reference.json": package["model_source_reference"],
        "evaluation_execution_contract.json": execution,
        "resolved_execution_context.json": execution["evaluation_execution_context"],
        "evaluation_only_state.json": {
            "evaluation_run_id": execution["evaluation_run_id"],
            "model_source_reference_sha256": execution["model_source_reference_sha256"],
            "executor_commit": execution["executor_commit"],
            "initialization_status": "initialized",
            "formal_execution_authorized": True,
            "formal_execution_started": False,
            "scientific_dispatch_evidence": "phase_state.jsonl and cell_state.jsonl",
            "holdout_opened": False,
            "training_executed": False,
            "dev_selection_executed": False,
            "checkpoint_freeze_executed": False,
        },
    }


def _initialize_run(package: dict[str, Any]) -> tuple[Path, FormalCellLedger, TransactionalPhaseRunner]:
    # Called only while holding the external writer lock. No recovery of remnants.
    root = Path(package["evaluation_execution_contract"]["evaluation_run_root"])
    root.mkdir(exist_ok=False)
    runner = _phase_runner(package, resume=False)  # still requires an empty root
    _create_file(root / "phase_state.jsonl", b"")
    for name, payload in _initial_files(package).items():
        _create_file(root / name, _encoded(payload))
    cells = FormalCellLedger(run_root=root, identity=_cell_identity(package))
    marker = {
        "version": "1.0.0",
        "run_root": str(root),
        "files": {name: file_sha256(root / name) for name in
                  [*_initial_files(package), "cell_ledger_identity.json"]},
        "initial_phase_ledger_sha256": file_sha256(root / "phase_state.jsonl"),
        "initial_cell_ledger_sha256": file_sha256(root / "cell_state.jsonl"),
    }
    _create_file(root / "evaluation_initialization_complete.json", _encoded(marker))
    return root, cells, runner


def _load_run(package: dict[str, Any]) -> tuple[Path, FormalCellLedger, TransactionalPhaseRunner]:
    root = Path(package["evaluation_execution_contract"]["evaluation_run_root"])
    if not root.exists():
        return _initialize_run(package)
    marker = _read(root / "evaluation_initialization_complete.json")
    names = [*_initial_files(package), "cell_ledger_identity.json"]
    if (marker.get("version") != "1.0.0" or marker.get("run_root") != str(root)
            or set(marker.get("files", {})) != set(names)
            or any(marker["files"][name] != file_sha256(root / name) for name in names)
            or any(marker.get(key) != hashlib.sha256(b"").hexdigest() for key in
                   ("initial_phase_ledger_sha256", "initial_cell_ledger_sha256"))):
        raise EvaluationOnlyError("evaluation initialization identity/hash drift")
    for name, payload in _initial_files(package).items():
        if (root / name).is_symlink() or (root / name).read_bytes() != _encoded(payload):
            raise EvaluationOnlyError("evaluation run identity/context bytes drift: " + name)
    for name in ("phase_state.jsonl", "cell_state.jsonl", "cell_ledger_identity.json"):
        if (root / name).is_symlink() or not (root / name).is_file():
            raise EvaluationOnlyError("evaluation initialization is incomplete")
    cells = FormalCellLedger(run_root=root, identity=_cell_identity(package), resume=True)
    return root, cells, _phase_runner(package, resume=True)


def execute_phase(package: dict[str, Any], grant: dict[str, Any], phase: str, *,
                  scientific_child_adapter=None) -> Any:
    authorization = verify_project_grant(package, grant)
    execution = package["evaluation_execution_contract"]
    root = Path(execution["evaluation_run_root"])
    if phase not in PHASES:
        raise EvaluationOnlyError("unknown evaluation phase")
    with SingleWriter(root, canonical_sha256(execution),
                      lambda: verify_project_grant(package, grant), allow_missing_root=True):
        if not root.exists() and phase != PHASES[0]:
            raise EvaluationOnlyError("first initialization requires first phase")
        root, cells, runner = _load_run(package)
        plan = execution["command_plans"][phase]
        records = runner.records()
        expected_context_file_hash = hashlib.sha256(_encoded(execution["evaluation_execution_context"])).hexdigest()
        if any(row.get("resolved_execution_context_sha256") != execution["evaluation_execution_context_sha256"]
               or row.get("resolved_execution_context_file_sha256") != expected_context_file_hash
               for row in records):
            raise EvaluationOnlyError("phase ledger context binding drift")
        if any(row.get("status") == "failed" for row in records):
            raise EvaluationOnlyError("failed evaluation phase is terminal")
        completed_phases = {
            row["phase"] for row in runner.records() if row.get("status") == "completed"
        }
        remaining = [name for name in PHASES if name not in completed_phases]
        if not remaining or phase != remaining[0]:
            raise EvaluationOnlyError("phase must be the next unstarted evaluation-only phase")
        if any(row["phase"] not in completed_phases for row in records):
            raise EvaluationOnlyError("interrupted evaluation phase cannot be resumed")
        for previous in completed_phases:
            prior = execution["command_plans"][previous]
            runner.run_phase(previous, commands=prior["commands"],
                             input_hash=canonical_sha256({"phase": previous, "contract": execution["execution_contract_sha256"]}),
                             expected_outputs=prior["expected_outputs"])
            if previous in PHASES[:5]:
                cells.assert_complete_matrix(phase=previous, expected_cell_ids=[
                    stable_cell_id(previous, row) for row in prior["matrix_contexts"]])
        command_coordinates = {
            canonical_sha256(command): coordinates
            for command, coordinates in zip(plan["commands"], plan["matrix_contexts"])
        }

        def dispatch(command: list[str]) -> PhaseCommandResult:
            verify_project_grant(package, grant)
            if phase in PHASES[:5]:
                coordinates = command_coordinates[canonical_sha256(command)]
                final, builder, resolver = cell_layout(
                    phase,
                    coordinates,
                    command,
                    SimpleNamespace(
                        stable_cell_id=stable_cell_id,
                        single_child_directory=__import__(
                            "src.evaluators.formal_cell_transaction", fromlist=["single_child_directory"]
                        ).single_child_directory,
                        resolve_child_output_descriptor=__import__(
                            "src.evaluators.formal_cell_transaction", fromlist=["resolve_child_output_descriptor"]
                        ).resolve_child_output_descriptor,
                    ),
                )
                if scientific_child_adapter is not None:
                    builder, resolver = scientific_child_adapter.adapt(
                        package, phase, builder, resolver
                    )
                result = execute_cell_artifact_transaction(
                    cells,
                    phase=phase,
                    coordinates=coordinates,
                    command=command,
                    input_hash=canonical_sha256(
                        {"command": command, "source": execution["model_source_reference_sha256"]}
                    ),
                    committed_path=final,
                    command_builder=builder,
                    artifact_resolver=resolver,
                    environment=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1"),
                    cwd=ROOT,
                    command_failure_classification="evaluation_only_cell_failure",
                )
                return PhaseCommandResult(
                    int(result.get("return_code", 0)),
                    str(result.get("stdout", "")),
                    str(result.get("stderr", "")),
                )
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1"),
                text=True,
                capture_output=True,
                check=False,
            )
            return PhaseCommandResult(completed.returncode, completed.stdout, completed.stderr)

        if phase == "complete_without_holdout":
            validate_complete_without_holdout_gate(root, _read(package["model_source_reference"]["protocol_path"]))
        result = runner.run_phase(
            phase,
            commands=plan["commands"],
            input_hash=canonical_sha256(
                {"phase": phase, "contract": execution["execution_contract_sha256"]}
            ),
            expected_outputs=plan["expected_outputs"],
            executor=dispatch,
            infrastructure_retries=0,
        )
        if phase in PHASES[:5]:
            cells.assert_complete_matrix(
                phase=phase,
                expected_cell_ids=[stable_cell_id(phase, row) for row in plan["matrix_contexts"]],
            )
    return {"authorization": authorization, "result": result}


def main(*, scientific_child_adapter=None) -> None:
    args = build_parser().parse_args()
    package = _read(args.authorization_request_path)
    if package.get("authorization_request_sha256") != canonical_sha256(
        {key: value for key, value in package.items() if key != "authorization_request_sha256"}
    ):
        raise EvaluationOnlyError("authorization request hash mismatch")
    validate_model_source_reference(package["model_source_reference"])
    validate_execution_contract(
        package["evaluation_execution_contract"],
        model_source_reference=package["model_source_reference"],
    )
    validate_command_matrix_parsers(
        package["evaluation_execution_contract"],
        model_source_reference=package["model_source_reference"],
    )
    if scientific_child_adapter is not None:
        # No CLI/env/package switch enables this. The isolated acceptance driver
        # supplies a source-controlled adapter for scientific child work only.
        from tests.evaluation_only_public_driver import SyntheticScientificChild
        if type(scientific_child_adapter) is not SyntheticScientificChild:
            raise EvaluationOnlyError("unknown scientific acceptance adapter")
        scientific_child_adapter.validate(package)
    grant = _read(args.project_grant_path)
    authorization = verify_project_grant(package, grant)
    if args.check == "qualify":
        print(json.dumps({"status": "qualified", "authorization": authorization}, indent=2))
        return
    result = execute_phase(package, grant, args.phase,
                           scientific_child_adapter=scientific_child_adapter)
    print(json.dumps({"status": "completed", "phase": args.phase, **result}, default=str, indent=2))


if __name__ == "__main__":
    main()
