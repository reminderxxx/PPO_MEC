"""Fixed continuation orchestration; native transactions retain all IO semantics."""
from __future__ import annotations

import subprocess

from . import PHASES
from .cells import cell_layout
from .identity import ContinuationError, digest, verify_executor, git
from pathlib import Path
from .locking import SingleWriter
from .planning import cell_input_hash
from .reconciliation import reconcile


def execute_phase(run, phase, authorize, executor_identity, *, finalize_only=False, scope=None):
    if phase not in PHASES:
        raise ContinuationError("unauthorized continuation phase")
    # A phase is the admitted transaction: expiry/revocation prevents admission
    # of the next phase. Native finalization of an admitted phase remains valid.
    # This includes retries belonging to that exact frozen command list.
    grant = authorize()
    if not grant.get("approval_verified") or grant["domain"] != run.contract["domain"]:
        raise ContinuationError("execution requires a verified independent grant")
    if (scope is None) != (grant["domain"] == "production"):
        raise ContinuationError("synthetic scope/production domain mismatch")
    if run.contract["command_plan_sha256"] != digest(run.plans):
        raise ContinuationError("approved command plan drift")
    def fixed_sources():
        # Public CLIs always supply the complete fixed-commit identity. Small
        # internal fault fixtures bind their test producer file separately.
        if "commit" in executor_identity:
            verify_executor(executor_identity,Path(__file__).resolve().parents[2])
        elif grant["domain"]=="production":
            raise ContinuationError("production requires a complete fixed executor identity")
        if git(run.root,"rev-parse","HEAD")!="a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d" or git(run.root,"status","--porcelain","--untracked-files=all"):
            raise ContinuationError("scientific source changed before dispatch")
    fixed_sources()
    n, plan = run.native, run.plans[phase]
    if scope:
        scope.tree()
        for argv in plan["commands"]:
            scope.argv(argv)
    reconcile(run.contract, n["phase"], run.cell_ledger, run.plans, protocol=run.protocol, context=run.context, registry_sha256=run.registry_sha256)
    with SingleWriter(str(run.run_root), digest(executor_identity), authorize,
                      run.contract["recovery_owner_sha256"]):
        reconcile(run.contract, n["phase"], run.cell_ledger, run.plans, protocol=run.protocol, context=run.context, registry_sha256=run.registry_sha256)
        if phase == "complete_without_holdout":
            n["public_checks"].validate_complete_without_holdout_gate(run.run_root, run.protocol)
        coordinates = dict(zip(plan["command_hashes"], plan["matrix_contexts"]))

        def execute(argv):
            fixed_sources()
            original = list(argv)
            if phase in PHASES[:5]:
                coord = coordinates[digest(original)]
                final, build, resolve = cell_layout(phase, coord, original, n["cell"])
                if scope:
                    scope.path(str(final))
                    original_build, original_resolve = build, resolve

                    def build(staging, cell_id):
                        scope.path(str(staging))
                        command = original_build(staging, cell_id)
                        scope.argv(command)
                        return command

                    def resolve(staging, cell_id, completed):
                        scope.tree(staging)
                        result = original_resolve(staging, cell_id, completed)
                        scope.path(str(result[0]))
                        scope.path(str(result[2]))
                        return result

                result = n["cell"].execute_cell_artifact_transaction(
                    run.cell_ledger, phase=phase, coordinates=coord, command=original,
                    input_hash=cell_input_hash(phase, coord, original, run.protocol, run.context, run.registry_sha256),
                    committed_path=final, command_builder=build, artifact_resolver=resolve,
                    environment=run.environment.child_environment, cwd=run.root,
                    command_failure_classification="formal_cell_failure")
                return n["phase"].PhaseCommandResult(int(result.get("return_code", 0)),
                            str(result.get("stdout", "")), str(result.get("stderr", "")))
            completed = subprocess.run(original, cwd=run.root, env=run.environment.child_environment,
                                       text=True, capture_output=True, check=False)
            return n["phase"].PhaseCommandResult(completed.returncode, completed.stdout, completed.stderr)

        if finalize_only:
            result = run.phase_runner.finalize_phase_only(
                phase, commands=plan["commands"], input_hash=plan["input_hash"], expected_outputs=plan["expected_outputs"])
        else:
            result = run.phase_runner.run_phase(
                phase, commands=plan["commands"], input_hash=plan["input_hash"],
                expected_outputs=plan["expected_outputs"], executor=execute,
                infrastructure_retries=plan["infrastructure_retries"])
        if phase in PHASES[:5]:
            run.cell_ledger.assert_complete_matrix(phase=phase, expected_cell_ids=[
                n["cell"].stable_cell_id(phase, coord) for coord in plan["matrix_contexts"]])
        if scope:
            scope.tree()
        reconcile(run.contract, n["phase"], run.cell_ledger, run.plans, protocol=run.protocol, context=run.context, registry_sha256=run.registry_sha256)
        return result
