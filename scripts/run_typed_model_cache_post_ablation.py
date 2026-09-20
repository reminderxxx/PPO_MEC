"""Run one separately granted post-ablation phase in a new create-only root."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.continuation_executor.cells import cell_layout
from scripts.continuation_executor.locking import SingleWriter
from scripts.prepare_typed_model_cache_post_ablation import validate_package
from scripts.run_typed_model_cache_evaluation_only import _load_run, _read, _encoded
from scripts.run_typed_model_cache_formal_protocol import validate_complete_without_holdout_gate
from src.evaluators.formal_cell_transaction import (
    execute_cell_artifact_transaction, stable_cell_id,
)
from src.evaluators.formal_phase_transaction import PhaseCommandResult, TransactionalPhaseRunner
from src.runtime.evaluation_only_execution import canonical_sha256, file_sha256
from src.runtime.post_ablation_execution import POST_PHASES, audit_handoff

def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--authorization-request-path", required=True)
    value.add_argument("--project-grant-path", required=True)
    value.add_argument("--phase", choices=POST_PHASES, required=True)
    value.add_argument("--check", choices=("qualify", "execute"), default="qualify")
    return value


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("project grant timestamp requires timezone")
    return parsed.astimezone(timezone.utc)


def verify_grant(package: dict, grant: dict, *, now: datetime | None = None) -> dict:
    if set(grant) != {
        "version", "status", "authorization_request_sha256", "evaluation_run_id",
        "model_source_reference_sha256", "execution_contract_sha256",
        "external_handoff_sha256", "executor_commit", "phases",
        "holdout_capability", "independent_review", "issued_at", "expires_at",
    } or grant.get("version") != "1.0.0":
        raise ValueError("post-ablation project grant schema drift")
    execution = package["evaluation_execution_contract"]
    expected = {
        "status": "AUTHORIZED_FOR_POST_ABLATION_EVALUATION",
        "authorization_request_sha256": package["authorization_request_sha256"],
        "evaluation_run_id": execution["evaluation_run_id"],
        "model_source_reference_sha256": execution["model_source_reference_sha256"],
        "execution_contract_sha256": execution["execution_contract_sha256"],
        "external_handoff_sha256": package["post_ablation_handoff_reference"]["sha256"],
        "executor_commit": execution["executor_commit"],
        "phases": list(POST_PHASES),
        "holdout_capability": False,
    }
    if any(grant.get(key) != value for key, value in expected.items()):
        raise ValueError("post-ablation grant identity or scope drift")
    current = now or datetime.now(timezone.utc)
    if not _utc(grant["issued_at"]) <= current < _utc(grant["expires_at"]):
        raise ValueError("post-ablation project grant is future or expired")
    row = grant["independent_review"]
    if not isinstance(row, dict) or set(row) != {"path", "sha256", "size_bytes"}:
        raise ValueError("independent review reference schema drift")
    path = Path(row["path"])
    if (path.is_symlink() or not path.is_file() or file_sha256(path) != row["sha256"]
            or path.stat().st_size != row["size_bytes"]):
        raise ValueError("independent review bytes drift")
    review = _read(path)
    if (review.get("status") != "pass"
            or review.get("reviewer_id") == review.get("implementation_agent_id")
            or review.get("authorization_request_sha256") != package["authorization_request_sha256"]
            or review.get("execution_contract_sha256") != execution["execution_contract_sha256"]
            or review.get("handoff_sha256") != expected["external_handoff_sha256"]
            or review.get("holdout_capability") is not False):
        raise ValueError("independent post-ablation review is incomplete")
    return {"status": "pass", "grant_sha256": canonical_sha256(grant)}


def _runner(package: dict) -> TransactionalPhaseRunner:
    execution = package["evaluation_execution_contract"]
    return TransactionalPhaseRunner(
        output_root=execution["evaluation_run_root"],
        run_identity_fingerprint=canonical_sha256({
            "evaluation_execution_contract": execution["execution_contract_sha256"],
            "model_source_reference": package["model_source_reference"]["source_reference_sha256"],
            "external_handoff": package["post_ablation_handoff_reference"]["sha256"],
        }),
        phase_order=POST_PHASES, resume=True,
        resolved_execution_context_sha256=execution["evaluation_execution_context_sha256"],
        resolved_execution_context_file_sha256=hashlib.sha256(
            _encoded(execution["evaluation_execution_context"])
        ).hexdigest(),
    )


def execute_phase(package: dict, grant: dict, phase: str) -> dict:
    verify_grant(package, grant)
    execution = package["evaluation_execution_contract"]
    root = Path(execution["evaluation_run_root"])
    if phase not in POST_PHASES:
        raise ValueError("phase outside post-ablation authority")
    with SingleWriter(root, canonical_sha256(execution),
                      lambda: verify_grant(package, grant), allow_missing_root=True):
        if not root.exists() and phase != POST_PHASES[0]:
            raise ValueError("first initialization requires formal_support")
        audit_handoff(package["post_ablation_handoff_reference"]["path"],
                      expected_sha256=package["post_ablation_handoff_reference"]["sha256"])
        root, cells, _ = _load_run(package)
        runner = _runner(package)
        records = runner.records()
        expected_context_file_hash = hashlib.sha256(
            _encoded(execution["evaluation_execution_context"])
        ).hexdigest()
        if any(row.get("resolved_execution_context_sha256") != execution["evaluation_execution_context_sha256"]
               or row.get("resolved_execution_context_file_sha256") != expected_context_file_hash
               for row in records):
            raise ValueError("phase ledger context binding drift")
        if any(row.get("status") == "failed" for row in records):
            raise ValueError("failed post-ablation phase is terminal")
        completed = {row["phase"] for row in records if row.get("status") == "completed"}
        remaining = [name for name in POST_PHASES if name not in completed]
        if not remaining or phase != remaining[0]:
            raise ValueError("phase is not the next unstarted phase")
        if any(row["phase"] not in completed for row in records):
            raise ValueError("interrupted phase cannot be resumed")
        for prior in completed:
            plan = execution["command_plans"][prior]
            runner.run_phase(prior, commands=plan["commands"],
                             input_hash=canonical_sha256({"phase": prior,
                                                         "contract": execution["execution_contract_sha256"]}),
                             expected_outputs=plan["expected_outputs"])
            if prior in POST_PHASES[:2]:
                cells.assert_complete_matrix(phase=prior, expected_cell_ids=[
                    stable_cell_id(prior, row) for row in plan["matrix_contexts"]
                ])
        plan = execution["command_plans"][phase]
        coordinates_by_command = {
            canonical_sha256(command): coordinates for command, coordinates in zip(
                plan["commands"], plan["matrix_contexts"]
            )
        }

        def dispatch(command: list[str]) -> PhaseCommandResult:
            verify_grant(package, grant)
            if phase in POST_PHASES[:2]:
                coordinates = coordinates_by_command[canonical_sha256(command)]
                native = SimpleNamespace(
                    stable_cell_id=stable_cell_id,
                    single_child_directory=__import__(
                        "src.evaluators.formal_cell_transaction", fromlist=["single_child_directory"]
                    ).single_child_directory,
                    resolve_child_output_descriptor=__import__(
                        "src.evaluators.formal_cell_transaction", fromlist=["resolve_child_output_descriptor"]
                    ).resolve_child_output_descriptor,
                )
                final, builder, resolver = cell_layout(phase, coordinates, command, native)
                result = execute_cell_artifact_transaction(
                    cells, phase=phase, coordinates=coordinates, command=command,
                    input_hash=canonical_sha256({"command": command,
                                                 "source": execution["model_source_reference_sha256"]}),
                    committed_path=final, command_builder=builder, artifact_resolver=resolver,
                    environment=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1"),
                    cwd=ROOT, command_failure_classification="post_ablation_cell_failure",
                )
                return PhaseCommandResult(int(result.get("return_code", 0)),
                                          str(result.get("stdout", "")),
                                          str(result.get("stderr", "")))
            result = subprocess.run(command, cwd=ROOT,
                                    env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1"),
                                    text=True, capture_output=True, check=False)
            return PhaseCommandResult(result.returncode, result.stdout, result.stderr)

        if phase == "complete_without_holdout":
            validate_complete_without_holdout_gate(
                root, _read(package["model_source_reference"]["protocol_path"])
            )
        result = runner.run_phase(
            phase, commands=plan["commands"],
            input_hash=canonical_sha256({"phase": phase,
                                         "contract": execution["execution_contract_sha256"]}),
            expected_outputs=plan["expected_outputs"], executor=dispatch,
            infrastructure_retries=0,
        )
        if phase in POST_PHASES[:2]:
            cells.assert_complete_matrix(phase=phase, expected_cell_ids=[
                stable_cell_id(phase, row) for row in plan["matrix_contexts"]
            ])
        return {"status": "completed", "phase": phase, "result": result}


def main() -> None:
    args = parser().parse_args()
    package = _read(args.authorization_request_path)
    validate_package(package, require_absent_root=False)
    grant = _read(args.project_grant_path)
    authorization = verify_grant(package, grant)
    if args.check == "qualify":
        print(json.dumps({"status": "qualified", "authorization": authorization}, indent=2))
    else:
        print(json.dumps(execute_phase(package, grant, args.phase), default=str, indent=2))


if __name__ == "__main__":
    main()
