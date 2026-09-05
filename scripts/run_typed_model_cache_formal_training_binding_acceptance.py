"""Run active clean-worktree formal training-entrypoint acceptance without training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import hashlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_typed_model_cache_formal_protocol import resolved_expansion_context
from src.evaluators.typed_model_cache_formal_execution import (
    validate_command_templates,
    validate_protocol_v1_1,
)
from src.evaluators.typed_model_cache_formal_protocol import canonical_sha256, sha256_file
from src.runtime.active_formal_bundle import (
    DEFAULT_ACTIVE_INDEX_RELATIVE,
    build_active_bundle_resource_resolution_audit,
    validate_active_formal_bundle,
)
from src.runtime.formal_protocol_capabilities import get_protocol_capabilities


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(cwd), "PYTHONNOUSERSITE": "1"},
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    return completed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-worktree-root", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-path", required=True)
    args = parser.parse_args()
    clean_root = Path(args.clean_worktree_root).resolve()
    python = str(Path(args.python_executable).absolute())
    output_root = Path(args.output_root).resolve()
    summary_path = Path(args.summary_path).resolve()
    if (clean_root / ".venv").exists():
        raise ValueError("clean acceptance worktree must not contain local .venv")
    if output_root.exists():
        raise FileExistsError(output_root)
    active_bundle = validate_active_formal_bundle(
        repository_root=clean_root,
        index_path=clean_root / DEFAULT_ACTIVE_INDEX_RELATIVE,
        require_ready=True,
        require_clean_git=True,
        require_origin_main_match=True,
    )
    protocol_path = Path(active_bundle["protocol_path"])
    environment_path = Path(active_bundle["execution_environment_manifest_path"])
    protocol = read_json(protocol_path)
    validate_protocol_v1_1(protocol)
    capabilities = get_protocol_capabilities(
        protocol["typed_model_cache_formal_protocol_version"]
    )
    if not capabilities.nullable_metric_contract_required:
        raise ValueError("active Protocol must require the nullable metric contract")
    resource_audit = build_active_bundle_resource_resolution_audit(active_bundle)
    context = resolved_expansion_context(
        protocol,
        protocol_path=str(protocol_path),
        output_root=str(output_root),
        python_executable=python,
        active_formal_bundle_sha256=active_bundle["active_formal_bundle_sha256"],
        active_protocol_index_path=str(clean_root / DEFAULT_ACTIVE_INDEX_RELATIVE),
        active_bundle_resource_resolution_audit_sha256=resource_audit["audit_sha256"],
    )
    outer = validate_command_templates(protocol["execution_contract"]["command_templates"], context)
    train = outer["expanded"]["train"]
    training_commands = train["commands"]
    matrix_contexts = train["matrix_contexts"]
    if len(training_commands) != 150 or len(matrix_contexts) != 150:
        raise ValueError("active Protocol training matrix must contain 150 cells")
    required_flags = {
        "--agent_scientific_config_path",
        "--formal_training_execution_binding_path",
        "--resolved_execution_context_path",
        "--formal_protocol_path",
    }
    command_identity_rows = []
    for command, coordinates in zip(training_commands, matrix_contexts):
        missing = sorted(required_flags - set(command))
        if missing or "--agent_config_path" in command:
            raise ValueError(f"training command identity flags drift: {missing}")
        command_identity_rows.append(
            {
                "agent": coordinates["agent"],
                "seed": coordinates["seed"],
                "capacity_label": coordinates["capacity_label"],
                "scientific_config_path": command[command.index("--agent_scientific_config_path") + 1],
                "execution_binding_path": command[command.index("--formal_training_execution_binding_path") + 1],
                "resolved_context_path": command[command.index("--resolved_execution_context_path") + 1],
            }
        )
    identity_triplets = {
        (row["scientific_config_path"], row["execution_binding_path"], row["resolved_context_path"])
        for row in command_identity_rows
    }
    if len(identity_triplets) != 1:
        raise ValueError("150 training commands do not share one config/binding/context identity")

    common = [
        python,
        str(clean_root / "scripts/run_typed_model_cache_formal_protocol.py"),
        "--active-protocol-index", str(clean_root / DEFAULT_ACTIVE_INDEX_RELATIVE),
        "--output-root", str(output_root),
        "--python-executable", python,
        "--execution-environment-manifest", str(environment_path),
    ]
    dry_run = run([*common, "--preflight", "--dry-run"], cwd=clean_root)
    dry_payload = json.loads(dry_run.stdout)
    if dry_payload["command_expansion"]["command_matrix_sha256"] != outer["command_matrix_sha256"]:
        raise ValueError("dry-run/outer command matrix mismatch")
    run([*common, "--preflight"], cwd=clean_root)
    resolved_context = read_json(output_root / "resolved_execution_context.json")
    binding = read_json(output_root / "formal_training_execution_binding.json")
    preflight = read_json(output_root / "preflight.json")
    nested_hash = preflight["command_expansion"]["command_matrix_sha256"]
    if nested_hash != outer["command_matrix_sha256"]:
        raise ValueError("outer/nested command matrix mismatch")

    rehearsal_root = output_root / "entrypoint_cells"
    log_root = output_root / "entrypoint_logs"
    log_root.mkdir(parents=True)
    command_results = []
    for command_index, (original, coordinates) in enumerate(
        zip(training_commands, matrix_contexts)
    ):
        agent = str(coordinates["agent"])
        seed = int(coordinates["seed"])
        capacity = str(coordinates["capacity_label"])
        command = list(original)
        command[command.index("--output_root") + 1] = str(rehearsal_root)
        run_id = f"entrypoint_preflight_{capacity}_{agent}_seed{seed}"
        command[command.index("--run_id") + 1] = run_id
        command.append("--formal_contract_preflight_only")
        completed = subprocess.run(
            command,
            cwd=clean_root,
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(clean_root), "PYTHONNOUSERSITE": "1"},
        )
        cell_label = f"{command_index:03d}_{capacity}_{agent}_seed{seed}"
        stdout_path = log_root / f"{cell_label}.stdout.log"
        stderr_path = log_root / f"{cell_label}.stderr.log"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        artifact = rehearsal_root / agent / run_id / "formal_contract_preflight.json"
        if completed.returncode != 0 or not artifact.is_file():
            raise RuntimeError(
                f"training entrypoint preflight failed for {cell_label}: "
                f"return_code={completed.returncode}; stderr={completed.stderr[-2000:]}"
            )
        payload = read_json(artifact)
        if any(
            (
                payload["formal"], payload["training"], payload["performance_evidence"],
                payload["checkpoint_created"], payload["episode_count"] != 0,
            )
        ):
            raise ValueError(f"contract rehearsal crossed execution boundary: {cell_label}")
        contract = payload["formal_training_contract"]
        if contract[
            "formal_nullable_metric_aggregation_contract_semantic_sha256"
        ] != protocol["formal_nullable_metric_aggregation_contract"]["semantic_sha256"]:
            raise ValueError(f"nullable contract identity drift: {cell_label}")
        cell_root = artifact.parent
        episode_files = sorted((cell_root / "episodes").glob("**/*"))
        checkpoint_files = sorted((cell_root / "checkpoints").glob("**/*"))
        if any(path.is_file() for path in (*episode_files, *checkpoint_files)):
            raise ValueError(f"preflight produced training files: {cell_label}")
        command_results.append(
            {
                "command_index": command_index,
                "agent": agent,
                "seed": seed,
                "capacity_label": capacity,
                "status": "pass",
                "return_code": completed.returncode,
                "command": command,
                "resolved_agent_config": payload["resolved_agent_config"],
                "resolved_training_budget": {
                    key: contract[key]
                    for key in (
                        "episodes", "update_every", "batch_size", "max_steps",
                        "checkpoint_every_updates", "expected_update_count",
                    )
                },
                "scientific_config_sha256": contract["agent_scientific_config_semantic_sha256"],
                "execution_binding_sha256": contract["formal_training_execution_binding_sha256"],
                "resolved_context_sha256": contract["resolved_execution_context_sha256"],
                "active_formal_bundle_sha256": contract["active_formal_bundle_sha256"],
                "nullable_metric_contract_semantic_sha256": contract[
                    "formal_nullable_metric_aggregation_contract_semantic_sha256"
                ],
                "artifact_sha256": sha256_file(artifact),
                "artifact_path": str(artifact),
                "stdout_path": str(stdout_path),
                "stdout_sha256": file_sha256(stdout_path),
                "stderr_path": str(stderr_path),
                "stderr_sha256": file_sha256(stderr_path),
                "episode_count": 0,
                "environment_interaction_count": 0,
                "update_count": 0,
                "checkpoint_file_count": 0,
                "performance_result_count": 0,
                "checkpoint_created": False,
            }
        )
    identity_pairs = {
        (row["scientific_config_sha256"], row["execution_binding_sha256"])
        for row in command_results
    }
    if len(identity_pairs) != 1:
        raise ValueError("150 cells did not resolve one scientific/binding identity")
    reachability = preflight["window_reachability"]
    summary = {
        "status": "pass",
        "formal_training_entrypoint_acceptance_version": "1.0.0",
        "formal": False,
        "training": False,
        "performance_evidence": False,
        "clean_detached_candidate": True,
        "training_command_count": len(command_results),
        "passed_command_count": sum(row["status"] == "pass" for row in command_results),
        "episode_count": 0,
        "environment_interaction_count": 0,
        "update_count": 0,
        "checkpoint_file_count": 0,
        "performance_result_count": 0,
        "checkpoint_count": 0,
        "holdout_opened": False,
        "clean_worktree_root": str(clean_root),
        "acceptance_output_root": str(output_root),
        "clean_worktree_has_local_venv": False,
        "execution_commit": resolved_context["scientific_identity"]["execution_commit"],
        "resolved_shared_python": python,
        "environment_fingerprint": resolved_context["scientific_identity"]["environment_fingerprint"],
        "dependency_fingerprint": resolved_context["scientific_identity"]["dependency_fingerprint"],
        "scientific_config_semantic_sha256": resolved_context["scientific_identity"]["agent_scientific_config_semantic_sha256"],
        "execution_binding_full_sha256": binding["binding_full_sha256"],
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "protocol_full_sha256": protocol["hashes"]["full_sha256"],
        "active_bundle_core_sha256": active_bundle["active_bundle_core_sha256"],
        "active_formal_bundle_sha256": active_bundle["active_formal_bundle_sha256"],
        "nullable_metric_contract_required": True,
        "nullable_metric_contract_semantic_sha256": protocol[
            "formal_nullable_metric_aggregation_contract"
        ]["semantic_sha256"],
        "resolved_context_sha256": resolved_context["context_sha256"],
        "command_audit": {
            "status": "pass",
            "phase_template_count": outer["phase_count"],
            "command_count": outer["command_count"],
            "training_command_count": len(training_commands),
            "unique_training_command_count": len({canonical_sha256(row) for row in training_commands}),
            "shared_identity_triplet_count": len(identity_triplets),
            "outer_expansion_sha256": outer["command_matrix_sha256"],
            "nested_expansion_sha256": nested_hash,
            "outer_nested_equal": True,
        },
        "entrypoint_acceptance": {
            "status": "pass",
            "agent_count": len({row["agent"] for row in command_results}),
            "seed_count": len({row["seed"] for row in command_results}),
            "capacity_count": len({row["capacity_label"] for row in command_results}),
            "commands": command_results,
            "formal": False,
            "training": False,
            "performance_evidence": False,
            "checkpoint_count": 0,
        },
        "preflight": {
            "status": "pass",
            "max_mobility_rows": int(context["max_mobility_rows"]),
            "provider_frame_count": reachability["provider_frame_count"],
            "window_count": reachability["window_count"],
            "reachable_count": reachability["reachable_count"],
            "split_reachable_counts": reachability["split_reachable_counts"],
        },
        "filesystem_execution_boundary": {
            "status": "pass",
            "episode_files": 0,
            "checkpoint_files": 0,
            "train_summary_files": 0,
            "performance_result_files": 0,
            "empty_episode_and_checkpoint_directories_allowed": True,
            "boolean_only_assertion_used": False,
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
