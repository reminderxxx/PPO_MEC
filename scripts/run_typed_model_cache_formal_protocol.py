"""Run the append-only typed model-cache formal protocol v1.1.

This ordinary runner deliberately exposes no holdout or hidden-data option.
Use ``--dry-run`` during G14R to validate expansion without creating results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
from src.runtime.resolved_formal_execution_context import (
    RESOLVED_FORMAL_EXECUTION_CONTEXT_FILENAME,
    atomic_create_resolved_formal_execution_context,
    build_resolved_formal_execution_context,
    load_resolved_formal_execution_context,
)


PROTOCOL_V14 = "1.4.0"
PROTOCOL_V15 = "1.5.0"


def _absolute_project_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (ROOT / path).resolve())


def resolved_expansion_context(
    protocol: dict,
    *,
    protocol_path: str,
    output_root: str,
    python_executable: str,
) -> dict:
    """Resolve every host/run location once at the outermost runner."""

    context = dict(protocol["execution_contract"]["default_expansion_context"])
    requested_output_root = str(Path(output_root).resolve())
    for key, value in list(context.items()):
        if isinstance(value, str) and value.startswith("/ABSOLUTE/FORMAL_OUTPUT_ROOT"):
            context[key] = requested_output_root + value[
                len("/ABSOLUTE/FORMAL_OUTPUT_ROOT"):
            ]
    context.update(
        protocol_path=str(Path(protocol_path).resolve()),
        output_root=requested_output_root,
        python_executable=str(Path(python_executable).absolute()),
        clean_worktree_root=str(ROOT),
        repository_root=str(ROOT),
        data_root=str((ROOT / "data").resolve()),
        checkpoint_root=requested_output_root,
        resolved_execution_context_path=str(
            Path(requested_output_root)
            / RESOLVED_FORMAL_EXECUTION_CONTEXT_FILENAME
        ),
        resolve_relative_paths_against_repository_root=True,
    )
    for _ in range(4):
        changed = False
        for key, value in list(context.items()):
            if not isinstance(value, str):
                continue
            rendered = value
            for name in re.findall(r"\{([a-z][a-z0-9_]*)\}", value):
                replacement = context.get(name)
                if replacement is not None and not isinstance(
                    replacement, (dict, list, tuple)
                ):
                    rendered = rendered.replace("{" + name + "}", str(replacement))
            if rendered != value:
                context[key] = rendered
                changed = True
        if not changed:
            break
    for key in (
        "protocol_artifact_root",
        "resource_registry_path",
        "agent_config_path",
        "dev_window_plan_path",
        "formal_window_plan_path",
        "train_window_plan_path",
        "window_consumption_contract_path",
    ):
        value = context.get(key)
        if isinstance(value, str) and value and "{" not in value:
            context[key] = _absolute_project_path(value)
    unresolved = {
        key: value
        for key, value in context.items()
        if isinstance(value, str) and ("{" in value or "}" in value)
    }
    if unresolved:
        raise FormalExecutionError(
            f"resolved execution context contains placeholders: {sorted(unresolved)}"
        )
    return context


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


def reject_invalid_run_root(protocol: dict, output_root: str | Path) -> None:
    version = protocol.get("typed_model_cache_formal_protocol_version")
    supersession = protocol.get("supersession", {})
    references = (
        supersession.get("invalid_execution_runs", [])
        if version == PROTOCOL_V15
        else supersession.get("invalid_g14c_v4_runs", [])
        if version == PROTOCOL_V14
        else []
    )
    invalid_run_ids = {
        str(item.get("run_id"))
        for item in references
        if isinstance(item, dict) and item.get("run_id")
    }
    if any(part in invalid_run_ids for part in Path(output_root).resolve().parts):
        raise FormalExecutionError(
            "legacy invalid G14C run root is permanently rejected"
        )


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
    requested_output_root = str(Path(args.output_root).resolve())
    reject_invalid_run_root(protocol, requested_output_root)
    environment_resolution = None
    environment_manifest = None
    if protocol_version == PROTOCOL_V15:
        if not args.python_executable or not args.execution_environment_manifest:
            raise FormalExecutionError(
                "protocol v1.5 requires explicit Python and execution environment manifest"
            )
        if not Path(args.python_executable).is_absolute():
            raise FormalExecutionError(
                "protocol v1.5 forbids relative Python or .venv fallback"
            )
    if protocol_version in {PROTOCOL_V14, PROTOCOL_V15}:
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
        if protocol_version == PROTOCOL_V15 and environment_resolution.runtime_audit.get(
            "resolution_source"
        ) != "explicit_python_executable":
            raise FormalExecutionError(
                "protocol v1.5 Python must come from explicit outer resolution"
            )
    resolved_python = (
        environment_resolution.python_executable
        if environment_resolution is not None
        else args.python_executable or sys.executable
    )
    context = resolved_expansion_context(
        protocol,
        protocol_path=args.protocol_path,
        output_root=requested_output_root,
        python_executable=resolved_python,
    )
    outer_validation = validate_command_templates(templates, context)
    resolved_context_payload = None
    resolved_context_report = None
    resolved_context_file_sha256 = None
    if protocol_version == PROTOCOL_V15:
        if environment_resolution is None or not args.execution_environment_manifest:
            raise FormalExecutionError("protocol v1.5 environment was not resolved")
        context_path = Path(context["resolved_execution_context_path"])
        if args.resume or args.finalize_phase_only:
            resolved_context_payload, resolved_context_report = (
                load_resolved_formal_execution_context(
                    context_path,
                    protocol=protocol,
                    clean_worktree_root=ROOT,
                    durable_run_root=requested_output_root,
                    environment_identity=environment_resolution.environment_identity,
                    runtime_audit=environment_resolution.runtime_audit,
                    check_git=True,
                )
            )
            if resolved_context_payload["resolved_expansion_context"] != context:
                raise FormalExecutionError(
                    "same-run resolved expansion context drift"
                )
            if resolved_context_payload["command_expansion"][
                "outer_expansion_sha256"
            ] != outer_validation["command_matrix_sha256"]:
                raise FormalExecutionError(
                    "same-run resolved command matrix drift"
                )
            resolved_context_file_sha256 = resolved_context_report["file_sha256"]
        else:
            resolved_context_payload = build_resolved_formal_execution_context(
                protocol=protocol,
                expansion_context=context,
                environment_identity=environment_resolution.environment_identity,
                runtime_audit=environment_resolution.runtime_audit,
                environment_manifest_path=args.execution_environment_manifest,
                outer_expansion_sha256=outer_validation["command_matrix_sha256"],
                phase_count=outer_validation["phase_count"],
                command_count=outer_validation["command_count"],
            )
            encoded_context = (
                json.dumps(
                    resolved_context_payload,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            resolved_context_file_sha256 = hashlib.sha256(
                encoded_context
            ).hexdigest()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_pass",
                    "selected_phase": phase,
                    "writes_performed": False,
                    "holdout_capability": False,
                    "phase_order": list(PHASE_ORDER),
                    "command_expansion": outer_validation,
                    "resolved_execution_context": (
                        {
                            "context_sha256": resolved_context_payload[
                                "context_sha256"
                            ],
                            "file_sha256": resolved_context_file_sha256,
                            "persisted": False,
                            "outer_expansion_sha256": outer_validation[
                                "command_matrix_sha256"
                            ],
                        }
                        if resolved_context_payload is not None
                        else None
                    ),
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

    if protocol_version != PROTOCOL_V15:
        raise FormalExecutionError(
            "formal Protocol v1.0-v1.4 is audit-only; new execution requires v1.5"
        )

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
            "resolved_execution_context_sha256": resolved_context_payload[
                "context_sha256"
            ],
            "resolved_execution_context_file_sha256": (
                resolved_context_file_sha256
            ),
        }
    )
    if protocol_version == PROTOCOL_V15:
        if environment_resolution is None or resolved_context_payload is None:
            raise FormalExecutionError("protocol v1.5 context was not resolved")
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
                "resolved_execution_context_sha256": resolved_context_payload[
                    "context_sha256"
                ],
                "resolved_execution_context_file_sha256": (
                    resolved_context_file_sha256
                ),
            }
        )
        runner_v3 = TransactionalPhaseRunner(
            output_root=args.output_root,
            run_identity_fingerprint=run_identity,
            phase_order=PHASE_ORDER,
            resume=args.resume or args.finalize_phase_only,
            resolved_execution_context_sha256=resolved_context_payload[
                "context_sha256"
            ],
            resolved_execution_context_file_sha256=(
                resolved_context_file_sha256
            ),
        )
        if not (args.resume or args.finalize_phase_only):
            persisted = atomic_create_resolved_formal_execution_context(
                context["resolved_execution_context_path"],
                resolved_context_payload,
            )
            if persisted["file_sha256"] != resolved_context_file_sha256:
                raise FormalExecutionError(
                    "resolved execution context artifact hash drift"
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
                    {
                        "command_templates": protocol["execution_contract"][
                            "command_templates"
                        ],
                        "resolved_execution_context_sha256": (
                            resolved_context_payload["context_sha256"]
                        ),
                    }
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
