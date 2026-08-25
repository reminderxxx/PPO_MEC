"""Run the append-only typed model-cache formal protocol v1.1.

This ordinary runner deliberately exposes no holdout or hidden-data option.
Use ``--dry-run`` during G14R to validate expansion without creating results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.typed_model_cache_formal_execution import (
    AppendOnlyPhaseRunner,
    FormalExecutionError,
    PHASE_ORDER,
    expand_command_plan,
    validate_command_templates,
    validate_protocol_v1_1,
)
from src.evaluators.typed_model_cache_formal_protocol import canonical_sha256
from src.evaluators.formal_phase_transaction import (
    PhaseCommandResult,
    TransactionalPhaseRunner,
)
from src.evaluators.formal_cell_transaction import (
    CellExecutionIdentity,
    FormalCellLedger,
    stable_cell_id,
)
from src.runtime.formal_execution_environment import resolve_execution_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--phase", choices=PHASE_ORDER)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from-cell-ledger", action="store_true")
    parser.add_argument("--finalize-phase-only", action="store_true")
    parser.add_argument("--python-executable", default="")
    parser.add_argument("--execution-environment-manifest", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.preflight == bool(args.phase):
        parser.error("select exactly one of --preflight or --phase")
    if args.resume_from_cell_ledger and not args.resume:
        parser.error("--resume-from-cell-ledger requires --resume")
    return args


def load_protocol(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise FormalExecutionError("protocol manifest must be an object")
    validate_protocol_v1_1(payload)
    return payload


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol_path)
    protocol_version = protocol["typed_model_cache_formal_protocol_version"]
    phase = "preflight" if args.preflight else args.phase
    templates = protocol["execution_contract"]["command_templates"]
    context = dict(protocol["execution_contract"]["default_expansion_context"])
    requested_output_root = str(Path(args.output_root).resolve())
    for key, value in list(context.items()):
        if isinstance(value, str) and value.startswith("/ABSOLUTE/FORMAL_OUTPUT_ROOT"):
            context[key] = requested_output_root + value[len("/ABSOLUTE/FORMAL_OUTPUT_ROOT"):]
    context.update(
        protocol_path=str(Path(args.protocol_path).resolve()),
        output_root=requested_output_root,
    )
    environment_resolution = None
    if protocol_version == "1.4.0":
        environment_manifest = None
        if args.execution_environment_manifest:
            environment_manifest = json.loads(
                Path(args.execution_environment_manifest).read_text(encoding="utf-8-sig")
            )
        environment_contract = protocol["formal_execution_environment_contract"]
        environment_resolution = resolve_execution_environment(
            clean_worktree_root=ROOT,
            execution_commit=environment_contract["scientific_identity"][
                "execution_commit"
            ],
            python_executable=args.python_executable or None,
            environment_manifest=environment_manifest,
            expected_identity=environment_contract["scientific_identity"],
            forbidden_source_roots=environment_contract.get(
                "forbidden_project_source_roots", []
            ),
            require_clean_git_worktree=True,
        )
        context["python_executable"] = environment_resolution.python_executable
        context["clean_worktree_root"] = str(ROOT)
        legacy_roots = []
        project_roots = [
            ROOT,
            *[
                Path(item)
                for item in environment_resolution.runtime_audit.get(
                    "forbidden_project_source_roots", []
                )
            ],
        ]
        for reference in protocol["supersession"]["invalid_g14c_v4_runs"]:
            relative_run = Path(reference["failure_audit_path"]).parent
            legacy_roots.extend(root / relative_run for root in project_roots)
        requested_root_path = Path(requested_output_root)
        for legacy_root in legacy_roots:
            try:
                requested_root_path.relative_to(legacy_root.resolve())
            except ValueError:
                continue
            raise FormalExecutionError(
                f"legacy G14C v4 run root is permanently invalid: {legacy_root}"
            )
    if args.dry_run:
        validation = validate_command_templates(templates, context)
        print(
            json.dumps(
                {
                    "status": "dry_run_pass",
                    "selected_phase": phase,
                    "writes_performed": False,
                    "holdout_capability": False,
                    "phase_order": list(PHASE_ORDER),
                    "command_expansion": validation,
                    "execution_environment": (
                        environment_resolution.runtime_audit
                        if environment_resolution is not None
                        else None
                    ),
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return

    if phase == "complete_without_holdout":
        command: list[str] | list[list[str]] = []
        expected_outputs: list[str] = []
        retries = 0
        matrix_contexts: list[dict] = []
    else:
        spec = templates[phase]
        plan = expand_command_plan(spec, context)
        command = plan["commands"]
        expected_outputs = plan["expected_outputs"]
        matrix_contexts = plan["matrix_contexts"]
        retries = int(spec.get("infrastructure_retries", 1))
    input_hash = canonical_sha256(
        {
            "protocol": protocol["hashes"]["semantic_sha256"],
            "phase": phase,
            "commands": command,
        }
    )
    if protocol_version == "1.4.0":
        if environment_resolution is None:
            raise FormalExecutionError("protocol v1.4 environment was not resolved")
        ledger_resume_phases = {
            "train",
            "dev_select",
            "formal_cache_policy",
            "formal_controller",
            "formal_ablation",
            "formal_support",
            "formal_scalability",
        }
        if (
            args.resume
            and phase in ledger_resume_phases
            and (Path(args.output_root) / "cell_ledger_identity.json").is_file()
            and not args.resume_from_cell_ledger
            and not args.finalize_phase_only
        ):
            raise FormalExecutionError(
                f"transactional {phase} resume requires --resume-from-cell-ledger"
            )
        if args.resume_from_cell_ledger and phase not in ledger_resume_phases:
            raise FormalExecutionError(
                "--resume-from-cell-ledger is only valid for a cell-transaction phase"
            )
        run_identity = canonical_sha256(
            {
                "output_root": requested_output_root,
                "protocol": protocol["hashes"]["semantic_sha256"],
                "resource_registry": protocol["portable_resource_identity_contract"][
                    "resource_registry_semantic_sha256"
                ],
                "environment": environment_resolution.environment_identity[
                    "environment_fingerprint"
                ],
                "execution_commit": environment_resolution.runtime_audit[
                    "observed_execution_commit"
                ],
            }
        )
        runner_v3 = TransactionalPhaseRunner(
            output_root=args.output_root,
            run_identity_fingerprint=run_identity,
            phase_order=PHASE_ORDER,
            resume=args.resume or args.finalize_phase_only,
        )

        cell_ledger = None
        cell_transaction_phases = {
            "train",
            "formal_cache_policy",
            "formal_controller",
            "formal_ablation",
            "formal_support",
            "formal_scalability",
        }
        if phase in cell_transaction_phases:
            cell_identity = CellExecutionIdentity(
                run_id=Path(args.output_root).resolve().name,
                execution_commit=environment_resolution.runtime_audit[
                    "observed_execution_commit"
                ],
                protocol_semantic_sha256=protocol["hashes"]["semantic_sha256"],
                resource_registry_semantic_sha256=protocol[
                    "portable_resource_identity_contract"
                ]["resource_registry_semantic_sha256"],
                environment_fingerprint=environment_resolution.environment_identity[
                    "environment_fingerprint"
                ],
                split_semantic_sha256=protocol["identity"]["split_semantic_sha256"],
                window_contract_semantic_sha256=protocol["execution_contract"][
                    "window_consumption_contract"
                ]["semantic_sha256"],
                catalog_fingerprint=protocol["identity"]["catalog_fingerprint"],
                runtime_identity=canonical_sha256(
                    protocol["identity"]["typed_runtime_contract_hashes_by_capacity"]
                ),
                command_matrix_sha256=canonical_sha256(
                    protocol["execution_contract"]["command_templates"]
                ),
            )
            cell_identity_path = Path(args.output_root) / "cell_ledger_identity.json"
            if args.resume_from_cell_ledger and not cell_identity_path.is_file():
                raise FormalExecutionError(
                    "--resume-from-cell-ledger requires an existing cell ledger identity"
                )
            if phase != "train" and not cell_identity_path.is_file():
                raise FormalExecutionError(
                    f"transactional {phase} requires the training cell ledger identity"
                )
            cell_ledger = FormalCellLedger(
                run_root=args.output_root,
                identity=cell_identity,
                resume=cell_identity_path.is_file(),
            )

        command_rows = command if command and isinstance(command[0], list) else ([command] if command else [])
        coordinate_by_command_hash = {
            canonical_sha256(list(argv)): dict(matrix_contexts[index])
            for index, argv in enumerate(command_rows)
        }
        if len(coordinate_by_command_hash) != len(command_rows):
            raise FormalExecutionError("transactional command matrix contains duplicate commands")

        def execute(argv):
            if phase == "train" and cell_ledger is not None:
                original = list(argv)
                coordinates = coordinate_by_command_hash[canonical_sha256(original)]
                cell_input_hash = canonical_sha256(
                    {
                        "protocol": protocol["hashes"]["semantic_sha256"],
                        "phase": phase,
                        "coordinates": coordinates,
                        "command": original,
                    }
                )
                output_flag = "--output_root"
                if output_flag not in original or "--run_id" not in original or "--agent_name" not in original:
                    raise FormalExecutionError("transactional train command lacks output/identity flags")
                final_output_root = Path(original[original.index(output_flag) + 1])
                run_id = original[original.index("--run_id") + 1]
                agent = original[original.index("--agent_name") + 1]
                final_path = final_output_root / agent / run_id
                begun = cell_ledger.begin_cell(
                    phase=phase,
                    coordinates=coordinates,
                    command=original,
                    input_hash=cell_input_hash,
                    committed_path=final_path,
                )
                if begun["status"] == "skipped_committed":
                    return PhaseCommandResult(0, "committed cell hash-verified and skipped", "")
                staging = Path(begun["record"]["staging_path"])
                staged = list(original)
                staged[staged.index(output_flag) + 1] = str(staging)
                started_ns = time.monotonic_ns()
                completed = subprocess.run(
                    staged,
                    cwd=ROOT,
                    env=environment_resolution.child_environment,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                artifact_root = staging / agent / run_id
                artifact_root.mkdir(parents=True, exist_ok=True)
                (artifact_root / "cell_stdout.log").write_text(
                    completed.stdout, encoding="utf-8"
                )
                (artifact_root / "cell_stderr.log").write_text(
                    completed.stderr, encoding="utf-8"
                )
                if completed.returncode != 0:
                    cell_ledger.fail_cell(
                        begun["cell_id"],
                        return_code=completed.returncode,
                        classification=(
                            "infrastructure_retryable"
                            if completed.returncode == 75
                            else "training_cell_failure"
                        ),
                        retryable=completed.returncode == 75,
                    )
                    return PhaseCommandResult(
                        completed.returncode, completed.stdout, completed.stderr
                    )
                cell_ledger.commit_cell(
                    begun["cell_id"],
                    required_paths=[
                        "train_summary.json", "cell_stdout.log", "cell_stderr.log"
                    ],
                    monotonic_started_ns=started_ns,
                    artifact_subpath=Path(agent) / run_id,
                )
                return PhaseCommandResult(0, completed.stdout, completed.stderr)
            if phase in cell_transaction_phases and cell_ledger is not None:
                original = list(argv)
                coordinates = coordinate_by_command_hash[canonical_sha256(original)]
                cell_input_hash = canonical_sha256(
                    {
                        "protocol": protocol["hashes"]["semantic_sha256"],
                        "phase": phase,
                        "coordinates": coordinates,
                        "command": original,
                    }
                )
                output_flag = "--output_root" if "--output_root" in original else "--output-root"
                if output_flag not in original:
                    raise FormalExecutionError(
                        f"transactional {phase} command lacks an output-root flag"
                    )
                final_output_root = Path(original[original.index(output_flag) + 1])
                cell_id = stable_cell_id(phase, coordinates)
                if phase == "formal_cache_policy":
                    capacity = str(coordinates["capacity_label"])
                    final_path = final_output_root / capacity
                elif phase == "formal_controller":
                    final_path = final_output_root / cell_id
                else:
                    setting_key = (
                        "ablation_setting_id"
                        if phase == "formal_ablation"
                        else "support_setting_id"
                        if phase == "formal_support"
                        else "scalability_setting_id"
                    )
                    final_path = final_output_root / str(coordinates[setting_key])
                begun = cell_ledger.begin_cell(
                    phase=phase,
                    coordinates=coordinates,
                    command=original,
                    input_hash=cell_input_hash,
                    committed_path=final_path,
                )
                if begun["status"] == "skipped_committed":
                    return PhaseCommandResult(
                        0, "committed formal cell hash-verified and skipped", ""
                    )
                staging = Path(begun["record"]["staging_path"])
                staged = list(original)
                if phase == "formal_cache_policy":
                    artifact_root = staging / "artifact"
                    staged[staged.index(output_flag) + 1] = str(
                        artifact_root / "benchmark"
                    )
                    replay_flag = "--request-replay-path"
                    if replay_flag not in staged:
                        raise FormalExecutionError(
                            "cache-policy transaction lacks request replay path"
                        )
                    staged[staged.index(replay_flag) + 1] = str(
                        artifact_root / "request_replay.json"
                    )
                    artifact_subpath = Path("artifact")
                elif phase == "formal_controller":
                    artifact_root = staging / "artifact"
                    staged[staged.index(output_flag) + 1] = str(artifact_root)
                    artifact_subpath = Path("artifact")
                else:
                    staged[staged.index(output_flag) + 1] = str(staging)
                    artifact_root = staging / final_path.name
                    artifact_subpath = Path(final_path.name)
                started_ns = time.monotonic_ns()
                completed = subprocess.run(
                    staged,
                    cwd=ROOT,
                    env=environment_resolution.child_environment,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                artifact_root.mkdir(parents=True, exist_ok=True)
                (artifact_root / "cell_stdout.log").write_text(
                    completed.stdout, encoding="utf-8"
                )
                (artifact_root / "cell_stderr.log").write_text(
                    completed.stderr, encoding="utf-8"
                )
                if completed.returncode != 0:
                    cell_ledger.fail_cell(
                        begun["cell_id"],
                        return_code=completed.returncode,
                        classification=(
                            "infrastructure_retryable"
                            if completed.returncode == 75
                            else "formal_cell_failure"
                        ),
                        retryable=completed.returncode == 75,
                    )
                    return PhaseCommandResult(
                        completed.returncode, completed.stdout, completed.stderr
                    )
                if phase == "formal_cache_policy":
                    if not (artifact_root / "request_replay.json").is_file():
                        raise FormalExecutionError("cache-policy replay was not staged")
                    required_name = "aggregate_summary.json"
                elif phase == "formal_controller":
                    required_name = "aggregate_summary.json"
                else:
                    required_name = "support_provenance.json"
                if not any(artifact_root.rglob(required_name)):
                    raise FormalExecutionError(
                        f"transactional {phase} artifact lacks {required_name}"
                    )
                cell_ledger.commit_cell(
                    begun["cell_id"],
                    required_paths=["cell_stdout.log", "cell_stderr.log"],
                    monotonic_started_ns=started_ns,
                    artifact_subpath=artifact_subpath,
                )
                return PhaseCommandResult(0, completed.stdout, completed.stderr)
            completed = subprocess.run(
                list(argv),
                cwd=ROOT,
                env=environment_resolution.child_environment,
                text=True,
                capture_output=True,
                check=False,
            )
            return PhaseCommandResult(
                completed.returncode, completed.stdout, completed.stderr
            )

        if args.finalize_phase_only:
            result = runner_v3.finalize_phase_only(
                phase,
                commands=command_rows,
                input_hash=input_hash,
                expected_outputs=expected_outputs,
            )
        else:
            result = runner_v3.run_phase(
                phase,
                commands=command_rows,
                input_hash=input_hash,
                expected_outputs=expected_outputs,
                executor=execute,
                infrastructure_retries=retries,
            )
        if cell_ledger is not None:
            cell_ledger.assert_complete_matrix(
                phase=phase,
                expected_cell_ids=[
                    stable_cell_id(phase, context_row)
                    for context_row in matrix_contexts
                ],
            )
    else:
        if args.finalize_phase_only or args.resume_from_cell_ledger:
            raise FormalExecutionError(
                "transactional resume/finalize is unavailable for legacy protocols"
            )
        runner = AppendOnlyPhaseRunner(
            protocol=protocol,
            output_root=args.output_root,
            resume=args.resume,
        )
        result = runner.run_phase(
            phase,
            command=command,
            input_hash=input_hash,
            expected_outputs=expected_outputs,
            infrastructure_retries=retries,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
