"""G14R20-I evaluation-only source and execution identity boundary.

This module is intentionally specific to the reviewed v16 frozen-model source.  It
does not create a run, ledger, lock, grant, or staging directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluators.typed_model_cache_formal_execution import expand_command_plan
from src.runtime.generated_checkpoint_resources import load_generated_checkpoint_registry


EVALUATION_ONLY_SOURCE_CONTRACT_VERSION = "1.0.0"
EVALUATION_ONLY_EXECUTION_CONTRACT_VERSION = "1.0.0"
SOURCE_SCIENTIFIC_COMMIT = "a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d"
SOURCE_SCIENTIFIC_TREE = "eb8e83c6e4532bc45c22fb7a816388e576753e30"
SOURCE_RUN_ID = "typed_model_cache_formal_20260906_152847_g14c_v16"
SOURCE_RUN_ROOT = Path(
    "/Users/howen/Projects/PPO_MEC/artifacts/experiments/typed_model_cache_formal/"
    "typed_model_cache_formal_20260906_152847_g14c_v16"
)
DISPOSITION_ELIGIBILITY_PATH = Path(
    "/Users/howen/Projects/PPO_MEC/artifacts/analysis/"
    "g14r20_h_v16_failed_run_disposition_20260911/"
    "checkpoint_and_partial_result_eligibility.json"
)
DISPOSITION_ELIGIBILITY_SHA256 = (
    "de39608762c86074b8c7e0528d2e3b0262150e2f8ac7929e18bda42cb45d0d34"
)
DISPOSITION_INTEGRITY_MANIFEST_PATH = DISPOSITION_ELIGIBILITY_PATH.parent / "integrity_manifest.json"
DISPOSITION_INTEGRITY_MANIFEST_SHA256 = (
    "89b4f3f62a768e06fd0d8c662f9b7a38ad0ad7142c0b63b6f751dacb42fe67e1"
)
PHASES = (
    "formal_cache_policy",
    "formal_controller",
    "formal_ablation",
    "formal_support",
    "formal_scalability",
    "formal_statistics",
    "formal_gate",
    "complete_without_holdout",
)
SOURCE_RUN_FILES = (
    "checkpoint_candidates.json",
    "dev_selection.json",
    "checkpoint_freeze.json",
    "generated_checkpoint_resource_registry.json",
    "resolved_execution_context.json",
    "formal_training_execution_binding.json",
)
SOURCE_ONLY_EXPANSION_KEYS = {
    "checkpoint_root",
    "checkpoint_freeze_output_path",
    "generated_checkpoint_registry_path",
    "seed_checkpoint_manifest_path",
    "checkpoint_provenance_manifest_path",
    "formal_training_execution_binding_path",
    "training_output_root",
    "dev_input_root",
    "dev_selection_output_path",
}


class EvaluationOnlyError(ValueError):
    """Raised when model-source and evaluation-execution identities are mixed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise EvaluationOnlyError(f"required immutable object is missing: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationOnlyError(f"invalid immutable JSON object: {target}") from exc
    if not isinstance(value, dict):
        raise EvaluationOnlyError(f"immutable JSON object must be a mapping: {target}")
    return value


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise EvaluationOnlyError(completed.stderr.strip() or "git identity check failed")
    return completed.stdout.strip()


def _require_clean_code(root: Path) -> None:
    scope = ("README.md", "docs", "scripts", "src", "tests", "configs")
    tracked = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", *scope],
        text=True,
        capture_output=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise EvaluationOnlyError("checkout tracked code/config/documentation is not clean")
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "--", *scope)
    if untracked:
        raise EvaluationOnlyError("checkout contains untracked code/config/documentation")


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def build_model_source_reference(
    *,
    disposition_eligibility_path: str | Path,
    source_run_root: str | Path,
    scientific_checkout: str | Path,
) -> dict[str, Any]:
    """Build a strict reference without changing any source run artifact."""

    eligibility_path = Path(disposition_eligibility_path).resolve()
    if (
        Path(disposition_eligibility_path).is_symlink()
        or eligibility_path != DISPOSITION_ELIGIBILITY_PATH
        or file_sha256(eligibility_path) != DISPOSITION_ELIGIBILITY_SHA256
        or DISPOSITION_INTEGRITY_MANIFEST_PATH.is_symlink()
        or file_sha256(DISPOSITION_INTEGRITY_MANIFEST_PATH)
        != DISPOSITION_INTEGRITY_MANIFEST_SHA256
    ):
        raise EvaluationOnlyError("H disposition trust root drift")
    eligibility = _read_object(eligibility_path)
    source = Path(source_run_root).resolve()
    scientific = Path(scientific_checkout).resolve()
    if (
        Path(source_run_root).is_symlink()
        or source != SOURCE_RUN_ROOT
        or source.name != SOURCE_RUN_ID
        or eligibility.get("artifact_run_id") != SOURCE_RUN_ID
    ):
        raise EvaluationOnlyError("unreviewed model source run")
    if eligibility.get("formal_execution_authorized") is not False:
        raise EvaluationOnlyError("disposition authorization boundary drift")
    checkpoints = eligibility.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 150:
        raise EvaluationOnlyError("source eligibility must contain exactly 150 checkpoints")
    if any(row.get("status") != "verified" for row in checkpoints if isinstance(row, dict)):
        raise EvaluationOnlyError("source eligibility contains an unverified checkpoint")
    if _git(scientific, "rev-parse", "HEAD") != SOURCE_SCIENTIFIC_COMMIT:
        raise EvaluationOnlyError("scientific checkout commit drift")
    _require_clean_code(scientific)
    scientific_tree = _git(scientific, "rev-parse", "HEAD^{tree}")
    if scientific_tree != SOURCE_SCIENTIFIC_TREE:
        raise EvaluationOnlyError("scientific checkout tree drift")
    context_path = source / "resolved_execution_context.json"
    binding_path = source / "formal_training_execution_binding.json"
    context = _read_object(context_path)
    binding = _read_object(binding_path)
    protocol_path = (
        scientific
        / "configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906"
        / "protocol_v2_9_manifest.json"
    )
    protocol = _read_object(protocol_path)
    if protocol.get("typed_model_cache_formal_protocol_version") != "2.9.0":
        raise EvaluationOnlyError("source Protocol is not 2.9.0")
    upstream_expected = {
        Path(row["path"]).name: row
        for row in eligibility.get("upstream_chain_files", [])
        if isinstance(row, dict)
    }
    immutable_files: list[dict[str, Any]] = []
    for name in SOURCE_RUN_FILES:
        path = source / name
        row = upstream_expected.get(name)
        observed = file_sha256(path)
        if row is not None and (
            row.get("status") != "verified"
            or row.get("observed_sha256") != observed
            or row.get("size_bytes") != path.stat().st_size
        ):
            raise EvaluationOnlyError(f"reviewed source chain drift: {name}")
        immutable_files.append(
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": observed}
        )
    registry_path = source / "generated_checkpoint_resource_registry.json"
    static_registry_path = (
        scientific
        / "configs/experiment/typed_model_cache_formal_protocol_v1_3_20260821"
        / "portable_resource_registry.json"
    )
    static_registry = _read_object(static_registry_path)
    _, registry_audit = load_generated_checkpoint_registry(
        registry_path,
        run_root=source,
        expected_run_id=SOURCE_RUN_ID,
        static_registry_semantic_sha256=static_registry["hashes"]["semantic_sha256"],
        protocol_semantic_sha256=protocol["hashes"]["semantic_sha256"],
        protocol_full_sha256=protocol["hashes"]["full_sha256"],
        active_formal_bundle_sha256=context["scientific_identity"]["active_formal_bundle_sha256"],
        execution_commit=SOURCE_SCIENTIFIC_COMMIT,
        resolved_execution_context_sha256=context["context_sha256"],
        formal_training_execution_binding_sha256=binding["binding_full_sha256"],
    )
    freeze = _read_object(source / "checkpoint_freeze.json")
    frozen = freeze.get("frozen_checkpoints")
    if not isinstance(frozen, list) or len(frozen) != 150:
        raise EvaluationOnlyError("checkpoint freeze does not contain 150 models")
    expected_coordinates = {
        (row["agent"], int(row["seed"]), row["capacity_label"])
        for row in protocol["execution_contract"]["command_templates"]["train"]["matrix_contexts"]
    }
    observed_coordinates = {
        (row.get("agent_name"), int(row.get("seed")), row.get("capacity_label"))
        for row in frozen
    }
    if observed_coordinates != expected_coordinates or len(observed_coordinates) != 150:
        raise EvaluationOnlyError("frozen agent/seed/capacity mapping drift")
    reviewed_by_coordinate = {
        (row["agent"], int(row["seed"]), row["capacity_label"]): row
        for row in checkpoints
    }
    model_rows = []
    for row in frozen:
        coordinate = (row["agent_name"], int(row["seed"]), row["capacity_label"])
        reviewed = reviewed_by_coordinate.get(coordinate)
        path = Path(row["checkpoint_path"])
        if path.is_symlink() or not path.is_file() or not _within(path, source):
            raise EvaluationOnlyError(f"reviewed frozen model path drift: {coordinate}")
        digest = file_sha256(path)
        if (
            reviewed is None
            or reviewed.get("expected_sha256") != digest
            or row.get("checkpoint_sha256") != digest
        ):
            raise EvaluationOnlyError(f"reviewed frozen model drift: {coordinate}")
        model_rows.append(
            {
                "agent": coordinate[0],
                "seed": coordinate[1],
                "capacity_label": coordinate[2],
                "checkpoint_path": str(path.resolve()),
                "checkpoint_sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    model_rows.sort(key=lambda row: (row["capacity_label"], row["agent"], row["seed"]))
    reference: dict[str, Any] = {
        "evaluation_only_model_source_contract_version": EVALUATION_ONLY_SOURCE_CONTRACT_VERSION,
        "disposition_eligibility_path": str(eligibility_path),
        "disposition_eligibility_sha256": file_sha256(eligibility_path),
        "disposition_integrity_manifest_path": str(DISPOSITION_INTEGRITY_MANIFEST_PATH),
        "disposition_integrity_manifest_sha256": file_sha256(
            DISPOSITION_INTEGRITY_MANIFEST_PATH
        ),
        "source_run_id": SOURCE_RUN_ID,
        "source_run_root": str(source),
        "scientific_commit": SOURCE_SCIENTIFIC_COMMIT,
        "scientific_git_tree": scientific_tree,
        "scientific_checkout": str(scientific),
        "protocol_version": "2.9.0",
        "protocol_path": str(protocol_path),
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "source_context_path": str(context_path),
        "source_context_sha256": file_sha256(context_path),
        "source_context_identity_sha256": context["context_sha256"],
        "source_binding_path": str(binding_path),
        "source_binding_sha256": file_sha256(binding_path),
        "source_binding_identity_sha256": binding["binding_full_sha256"],
        "source_generated_registry_path": str(registry_path),
        "source_generated_registry_file_sha256": file_sha256(registry_path),
        "source_generated_registry_canonical_sha256": registry_audit[
            "registry_canonical_sha256"
        ],
        "selection_sha256": freeze["selection_sha256"],
        "checkpoint_freeze_sha256": freeze["freeze_sha256"],
        "immutable_source_files": immutable_files,
        "models": model_rows,
        "model_count": 150,
        "training_claimed_by_evaluation_run": False,
        "dev_selection_claimed_by_evaluation_run": False,
        "checkpoint_freeze_claimed_by_evaluation_run": False,
        "formal_source_exposure": "seen_from_failed_v16_partial_288mb",
        "legacy_result_exclusions": [
            str(source / "formal_cache_policy" / "constrained_288mb"),
            str(source / ".staging" / "formal_cache_policy"),
        ],
        "legacy_results_enter_new_statistics": False,
        "holdout_opened": False,
    }
    reference["source_reference_sha256"] = canonical_sha256(reference)
    return reference


def validate_model_source_reference(reference: Mapping[str, Any]) -> dict[str, Any]:
    supplied = dict(reference)
    digest = supplied.pop("source_reference_sha256", None)
    if digest != canonical_sha256(supplied):
        raise EvaluationOnlyError("model source reference canonical hash mismatch")
    rebuilt = build_model_source_reference(
        disposition_eligibility_path=supplied["disposition_eligibility_path"],
        source_run_root=supplied["source_run_root"],
        scientific_checkout=supplied["scientific_checkout"],
    )
    if dict(reference) != rebuilt:
        raise EvaluationOnlyError("model source reference differs from current immutable source")
    return {"status": "pass", "model_count": 150, "source_reference_sha256": digest}


def build_evaluation_execution_contract(
    *,
    source_reference: Mapping[str, Any],
    evaluation_run_id: str,
    evaluation_run_root: str | Path,
    executor_checkout: str | Path,
    executor_commit: str,
    python_executable: str | Path,
) -> dict[str, Any]:
    """Build the new execution identity and exact eight-phase command matrix."""

    source = dict(source_reference)
    source_hash = source.get("source_reference_sha256")
    if source_hash != canonical_sha256(
        {key: value for key, value in source.items() if key != "source_reference_sha256"}
    ):
        raise EvaluationOnlyError("model source reference hash mismatch")
    if not evaluation_run_id or evaluation_run_id == SOURCE_RUN_ID:
        raise EvaluationOnlyError("evaluation run must have a new identity")
    run_root = Path(evaluation_run_root).resolve()
    source_root = Path(source["source_run_root"]).resolve()
    if _within(run_root, source_root) or _within(source_root, run_root):
        raise EvaluationOnlyError("evaluation and model-source run roots must be disjoint")
    if run_root.name != evaluation_run_id:
        raise EvaluationOnlyError("evaluation run ID/root mismatch")
    executor = Path(executor_checkout).resolve()
    if _git(executor, "rev-parse", "HEAD") != executor_commit:
        raise EvaluationOnlyError("executor checkout commit drift")
    _require_clean_code(executor)
    python = Path(python_executable)
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise EvaluationOnlyError("absolute evaluation interpreter is missing")
    protocol = _read_object(source["protocol_path"])
    source_context = _read_object(source["source_context_path"])
    static_registry_path = (
        Path(source["scientific_checkout"])
        / "configs/experiment/typed_model_cache_formal_protocol_v1_3_20260821"
        / "portable_resource_registry.json"
    )
    old_scientific = Path(
        source_context["resolved_expansion_context"]["clean_worktree_root"]
    )
    scientific = Path(source["scientific_checkout"])
    expansion = deepcopy(source_context["resolved_expansion_context"])
    for key, value in list(expansion.items()):
        if not isinstance(value, str):
            continue
        if key not in SOURCE_ONLY_EXPANSION_KEYS and value.startswith(str(source_root)):
            expansion[key] = str(run_root) + value[len(str(source_root)) :]
        if value.startswith(str(old_scientific / "data")):
            expansion[key] = str(Path(source["scientific_checkout"]).parents[2] / "data") + value[
                len(str(old_scientific / "data")) :
            ]
        elif value.startswith(str(old_scientific)):
            expansion[key] = str(scientific) + value[len(str(old_scientific)) :]
    expansion.update(
        clean_worktree_root=str(executor),
        repository_root=str(executor),
        protocol_path=source["protocol_path"],
        python_executable=str(python),
        output_root=str(run_root),
        resolved_execution_context_path=str(run_root / "resolved_execution_context.json"),
        formal_training_execution_binding_path=source["source_binding_path"],
        generated_checkpoint_registry_path=source["source_generated_registry_path"],
        checkpoint_root=str(source_root),
        checkpoint_freeze_output_path=str(source_root / "checkpoint_freeze.json"),
    )
    plans: dict[str, Any] = {}
    for phase in PHASES:
        if phase == "complete_without_holdout":
            plan = {"commands": [], "expected_outputs": [], "matrix_contexts": []}
        else:
            plan = expand_command_plan(
                protocol["execution_contract"]["command_templates"][phase], expansion
            )
        commands = plan["commands"]
        reference_flag = [
            "--evaluation-model-source-reference-path",
            str(run_root / "evaluation_model_source_reference.json"),
        ]
        for command in commands:
            if "--command" in command:
                boundary = command.index("--command")
                command[boundary:boundary] = reference_flag
                command.extend(reference_flag)
            else:
                command.extend(reference_flag)
        serialized = "\n".join("\0".join(command) for command in commands).lower()
        forbidden = ("checkpoint_freeze", "dev_select", "/train_", "sealed_holdout", "--holdout")
        if any(token in serialized for token in forbidden):
            raise EvaluationOnlyError(f"unauthorized command in evaluation plan: {phase}")
        plans[phase] = plan
    contract: dict[str, Any] = {
        "evaluation_only_execution_contract_version": EVALUATION_ONLY_EXECUTION_CONTRACT_VERSION,
        "evaluation_run_id": evaluation_run_id,
        "evaluation_run_root": str(run_root),
        "executor_checkout": str(executor),
        "executor_commit": executor_commit,
        "executor_git_tree": _git(executor, "rev-parse", "HEAD^{tree}"),
        "python_executable": str(python),
        "model_source_reference_sha256": source_hash,
        "model_source_run_id": SOURCE_RUN_ID,
        "phases": list(PHASES),
        "command_plans": plans,
        "command_plan_sha256": canonical_sha256(plans),
        "ledger_path": str(run_root / "phase_state.jsonl"),
        "cell_ledger_path": str(run_root / "cell_state.jsonl"),
        "lock_root": str(run_root.parent / ".evaluation_only_locks"),
        "staging_root": str(run_root / ".staging"),
        "legacy_result_exclusions": list(source["legacy_result_exclusions"]),
        "legacy_results_enter_new_statistics": False,
        "formal_execution_authorized": False,
        "formal_execution_started": False,
        "holdout_capability": False,
        "holdout_opened": False,
        "training_commands": 0,
        "dev_selection_commands": 0,
        "checkpoint_freeze_commands": 0,
        "holdout_commands": 0,
    }
    evaluation_context = deepcopy(source_context)
    evaluation_context["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    evaluation_context["created_for_run_identity"] = canonical_sha256(
        {
            "evaluation_run_id": evaluation_run_id,
            "executor_commit": executor_commit,
            "model_source_reference_sha256": source_hash,
            "command_plan_sha256": contract["command_plan_sha256"],
        }
    )
    evaluation_context["scientific_identity"]["execution_commit"] = executor_commit
    evaluation_context["runtime_location"].update(
        resolved_python_absolute_path=str(python),
        python_resolution_source="explicit_evaluation_only_python_executable",
        clean_worktree_root=str(executor),
        durable_run_root=str(run_root),
        protocol_path=source["protocol_path"],
        repository_root=str(executor),
        data_root=str(Path(source["scientific_checkout"]).parents[2] / "data"),
        checkpoint_root=str(source_root),
        protocol_artifact_root=str(
            Path(source["scientific_checkout"])
            / "configs/experiment/typed_model_cache_formal_protocol_v1_3_20260821"
        ),
        resource_registry_path=str(static_registry_path),
        execution_environment_manifest_path=str(
            Path(source["scientific_checkout"])
            / "configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906"
            / "execution_environment_manifest.json"
        ),
        resolved_execution_context_path=str(run_root / "resolved_execution_context.json"),
        formal_training_execution_binding_path=source["source_binding_path"],
    )
    evaluation_context["runtime_location"]["execution_environment_manifest_sha256"] = (
        file_sha256(evaluation_context["runtime_location"]["execution_environment_manifest_path"])
    )
    evaluation_context["command_expansion"].update(
        outer_expansion_sha256=contract["command_plan_sha256"],
        resolved_command_matrix_sha256=contract["command_plan_sha256"],
        phase_count=len(PHASES),
        command_count=sum(len(plan["commands"]) for plan in plans.values()),
    )
    evaluation_context["resolved_expansion_context"] = expansion
    evaluation_context["evaluation_execution_identity"] = {
        "evaluation_run_id": evaluation_run_id,
        "executor_commit": executor_commit,
        "executor_git_tree": contract["executor_git_tree"],
        "model_source_reference_sha256": source_hash,
        "source_run_id": source["source_run_id"],
        "source_context_identity_sha256": source["source_context_identity_sha256"],
        "source_context_file_sha256": source["source_context_sha256"],
        "source_binding_identity_sha256": source["source_binding_identity_sha256"],
        "host_paths_are_scientific_identity": False,
    }
    evaluation_context["context_sha256"] = canonical_sha256(
        {key: item for key, item in evaluation_context.items() if key != "context_sha256"}
    )
    from src.runtime.resolved_formal_execution_context import (
        validate_resolved_formal_execution_context,
    )

    validate_resolved_formal_execution_context(
        evaluation_context,
        protocol=protocol,
        clean_worktree_root=executor,
        durable_run_root=run_root,
        check_git=True,
    )
    contract["evaluation_execution_context"] = evaluation_context
    contract["evaluation_execution_context_sha256"] = evaluation_context["context_sha256"]
    contract["execution_contract_sha256"] = canonical_sha256(contract)
    return contract


def validate_execution_contract(
    contract: Mapping[str, Any],
    *,
    model_source_reference: Mapping[str, Any] | None = None,
    check_live: bool = True,
) -> dict[str, Any]:
    value = dict(contract)
    digest = value.pop("execution_contract_sha256", None)
    if digest != canonical_sha256(value):
        raise EvaluationOnlyError("evaluation execution contract hash mismatch")
    if tuple(value.get("phases", ())) != PHASES:
        raise EvaluationOnlyError("evaluation phase authority drift")
    if any(value.get(field) != 0 for field in (
        "training_commands", "dev_selection_commands", "checkpoint_freeze_commands", "holdout_commands"
    )):
        raise EvaluationOnlyError("evaluation-only contract contains forbidden commands")
    if value.get("holdout_capability") is not False or value.get("holdout_opened") is not False:
        raise EvaluationOnlyError("evaluation-only contract cannot open holdout")
    if value.get("formal_execution_authorized") is not False or value.get(
        "formal_execution_started"
    ) is not False:
        raise EvaluationOnlyError("unsigned preparation cannot claim execution authority")
    if set(value.get("command_plans", {})) != set(PHASES):
        raise EvaluationOnlyError("evaluation command matrix membership drift")
    plans = value["command_plans"]
    if value.get("command_plan_sha256") != canonical_sha256(plans):
        raise EvaluationOnlyError("evaluation command plan hash mismatch")
    expected_counts = {
        "formal_cache_policy": 3,
        "formal_controller": 3,
        "formal_ablation": 2,
        "formal_support": 11,
        "formal_scalability": 3,
        "formal_statistics": 1,
        "formal_gate": 1,
        "complete_without_holdout": 0,
    }
    for phase, expected_count in expected_counts.items():
        plan = plans[phase]
        commands = plan.get("commands")
        contexts = plan.get("matrix_contexts")
        if not isinstance(commands, list) or len(commands) != expected_count:
            raise EvaluationOnlyError(f"evaluation command count drift: {phase}")
        if not isinstance(contexts, list) or len(contexts) != expected_count:
            raise EvaluationOnlyError(f"evaluation matrix context count drift: {phase}")
        if plan.get("matrix_cell_count", expected_count) != expected_count:
            raise EvaluationOnlyError(f"evaluation matrix cell count drift: {phase}")
    if model_source_reference is not None:
        source = dict(model_source_reference)
        source_digest = source.get("source_reference_sha256")
        if source_digest != canonical_sha256(
            {key: item for key, item in source.items() if key != "source_reference_sha256"}
        ):
            raise EvaluationOnlyError("model source reference hash mismatch")
        if value.get("model_source_reference_sha256") != source_digest:
            raise EvaluationOnlyError("execution/model-source reference mismatch")
        if value.get("model_source_run_id") != source.get("source_run_id"):
            raise EvaluationOnlyError("execution/model-source run mismatch")
        if value.get("legacy_result_exclusions") != source.get("legacy_result_exclusions"):
            raise EvaluationOnlyError("execution/model-source legacy exclusion mismatch")
    context = value.get("evaluation_execution_context")
    if not isinstance(context, Mapping) or value.get(
        "evaluation_execution_context_sha256"
    ) != context.get("context_sha256"):
        raise EvaluationOnlyError("evaluation execution context binding is missing")
    if context.get("evaluation_execution_identity", {}).get(
        "model_source_reference_sha256"
    ) != value.get("model_source_reference_sha256"):
        raise EvaluationOnlyError("evaluation context/model-source binding drift")
    if context.get("evaluation_execution_identity", {}).get(
        "evaluation_run_id"
    ) != value.get("evaluation_run_id"):
        raise EvaluationOnlyError("evaluation context/run binding drift")
    if context.get("context_sha256") != canonical_sha256(
        {key: item for key, item in context.items() if key != "context_sha256"}
    ):
        raise EvaluationOnlyError("evaluation execution context hash mismatch")
    if check_live:
        executor = Path(str(value.get("executor_checkout", "")))
        python = Path(str(value.get("python_executable", "")))
        if (
            not executor.is_absolute()
            or executor.is_symlink()
            or not executor.is_dir()
            or _git(executor, "rev-parse", "HEAD") != value.get("executor_commit")
            or _git(executor, "rev-parse", "HEAD^{tree}")
            != value.get("executor_git_tree")
        ):
            raise EvaluationOnlyError("live executor checkout identity drift")
        _require_clean_code(executor)
        if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
            raise EvaluationOnlyError("live evaluation interpreter drift")
        if context.get("runtime_location", {}).get(
            "resolved_python_absolute_path"
        ) != str(python):
            raise EvaluationOnlyError("evaluation context/interpreter drift")
        protocol_path = (model_source_reference or {}).get(
            "protocol_path"
        ) or context.get("runtime_location", {}).get("protocol_path")
        protocol = _read_object(protocol_path)
        from src.runtime.resolved_formal_execution_context import (
            validate_resolved_formal_execution_context,
        )

        validate_resolved_formal_execution_context(
            context,
            protocol=protocol,
            clean_worktree_root=executor,
            durable_run_root=value["evaluation_run_root"],
            check_git=True,
        )
    return {
        "status": "pass",
        "phase_count": len(PHASES),
        "command_count": sum(
            len(plan["commands"]) for plan in value["command_plans"].values()
        ),
        "execution_contract_sha256": digest,
    }


def validate_command_matrix_parsers(
    contract: Mapping[str, Any],
    *,
    model_source_reference: Mapping[str, Any] | None = None,
    check_live: bool = True,
) -> dict[str, Any]:
    """Exercise the actual phase parsers, including cache-policy's nested benchmark."""

    validate_execution_contract(
        contract,
        model_source_reference=model_source_reference,
        check_live=check_live,
    )
    from scripts import benchmark_main_results
    from scripts import manage_typed_model_cache_formal_artifacts
    from scripts import run_typed_model_cache_formal_cache_policy
    from scripts import run_typed_model_cache_formal_statistics
    from scripts import run_typed_model_cache_formal_support

    executor = Path(contract["executor_checkout"]).resolve()
    python = str(Path(contract["python_executable"]))
    scripts = {
        "formal_cache_policy": executor / "scripts/run_typed_model_cache_formal_cache_policy.py",
        "formal_controller": executor / "scripts/benchmark_main_results.py",
        "formal_ablation": executor / "scripts/run_typed_model_cache_formal_support.py",
        "formal_support": executor / "scripts/run_typed_model_cache_formal_support.py",
        "formal_scalability": executor / "scripts/run_typed_model_cache_formal_support.py",
        "formal_statistics": executor / "scripts/run_typed_model_cache_formal_statistics.py",
        "formal_gate": executor / "scripts/manage_typed_model_cache_formal_artifacts.py",
    }
    parsed = 0
    nested = 0
    for phase, plan in contract["command_plans"].items():
        if phase == "complete_without_holdout":
            if plan["commands"]:
                raise EvaluationOnlyError("completion phase unexpectedly contains a command")
            continue
        for command in plan["commands"]:
            if len(command) < 2 or command[0] != python or command[1] != str(scripts[phase]):
                raise EvaluationOnlyError(f"evaluation command entrypoint drift: {phase}")
            if phase == "formal_cache_policy":
                args = run_typed_model_cache_formal_cache_policy.build_parser().parse_args(
                    command[2:]
                )
                nested_command = list(args.command)
                if nested_command and nested_command[0] == "--":
                    nested_command = nested_command[1:]
                expected_nested = str(executor / "scripts/benchmark_main_results.py")
                if (
                    len(nested_command) < 2
                    or nested_command[0] != python
                    or nested_command[1] != expected_nested
                ):
                    raise EvaluationOnlyError("nested benchmark entrypoint drift")
                _parse_with_existing_parse_args(
                    benchmark_main_results.parse_args, nested_command[2:]
                )
                nested += 1
            else:
                _parse_with_existing_parse_args(
                    {
                        "formal_controller": benchmark_main_results.parse_args,
                        "formal_ablation": run_typed_model_cache_formal_support.parse_args,
                        "formal_support": run_typed_model_cache_formal_support.parse_args,
                        "formal_scalability": run_typed_model_cache_formal_support.parse_args,
                        "formal_statistics": run_typed_model_cache_formal_statistics.parse_args,
                        "formal_gate": manage_typed_model_cache_formal_artifacts.parse_args,
                    }[phase],
                    command[2:],
                )
            parsed += 1
    return {"status": "pass", "parsed_command_count": parsed, "nested_command_count": nested}


def _parse_with_existing_parse_args(parse_args: Any, argv: Sequence[str]) -> Any:
    previous = sys.argv
    try:
        sys.argv = [previous[0], *argv]
        return parse_args()
    except SystemExit as exc:
        raise EvaluationOnlyError(f"actual command parser rejected matrix: {argv}") from exc
    finally:
        sys.argv = previous


def reject_legacy_result_path(path: str | Path, contract: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    for excluded in contract.get("legacy_result_exclusions", []):
        if _within(target, Path(excluded)):
            raise EvaluationOnlyError("legacy v16 result/staging is excluded from new statistics")


__all__ = [
    "EVALUATION_ONLY_EXECUTION_CONTRACT_VERSION",
    "EVALUATION_ONLY_SOURCE_CONTRACT_VERSION",
    "EvaluationOnlyError",
    "PHASES",
    "SOURCE_RUN_ID",
    "SOURCE_SCIENTIFIC_COMMIT",
    "build_evaluation_execution_contract",
    "build_model_source_reference",
    "canonical_sha256",
    "reject_legacy_result_path",
    "validate_execution_contract",
    "validate_command_matrix_parsers",
    "validate_model_source_reference",
]
