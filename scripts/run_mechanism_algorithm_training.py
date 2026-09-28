#!/usr/bin/env python3
"""Launch a frozen, non-formal matched retraining pair in the background."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNNER_VERSION = "mechanism_algorithm_training_background_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML mapping required: {path}")
    return value


def _resolve_data_path(data_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else data_root / path).resolve()


def _validate_source(path: Path, expected: dict[str, Any], label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != int(expected["size_bytes"]):
        raise ValueError(f"{label} size mismatch: {size}")
    sha256 = _file_sha256(path)
    if sha256 != str(expected["sha256"]):
        raise ValueError(f"{label} SHA-256 mismatch: {sha256}")
    return {"path": str(path), "size_bytes": size, "sha256": sha256}


def _git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT_DIR, text=True
    ).strip()


def _build_manifest(config_path: Path, data_root: Path, output_root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    python = Path(str(config["python_executable"])).expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    data = dict(config["data"])
    expected = dict(config["expected_sources"])
    mobility = _resolve_data_path(data_root, str(data["mobility_csv_path"]))
    workflow = _resolve_data_path(data_root, str(data["workflow_csv_path"]))
    sources = {
        "mobility": _validate_source(mobility, dict(expected["mobility"]), "mobility"),
        "workflow": _validate_source(workflow, dict(expected["workflow"]), "workflow"),
    }
    runtime_config = (ROOT_DIR / str(config["runtime_config"])).resolve()
    window_plan = (ROOT_DIR / str(config["window_plan"])).resolve()
    for path in (runtime_config, window_plan):
        if not path.is_file():
            raise FileNotFoundError(path)

    budget = dict(config["budget"])
    commands: list[dict[str, Any]] = []
    for agent_name in list(config["agents"]):
        run_id = f"{config['study_version']}_{agent_name}_seed{budget['seed']}"
        command = [
            str(python),
            str(ROOT_DIR / "scripts/train_algo_pool_real_sample.py"),
            "--agent_name",
            str(agent_name),
            "--profile",
            "baseline_safe",
            "--episodes",
            str(budget["episodes_per_agent"]),
            "--update_every",
            str(budget["update_every"]),
            "--batch_size",
            str(budget["batch_size"]),
            "--max_steps",
            str(budget["max_steps"]),
            "--checkpoint_every_updates",
            str(budget["checkpoint_every_updates"]),
            "--random_seed",
            str(budget["seed"]),
            "--model_cache_runtime_config",
            str(runtime_config),
            "--mechanism_factorial_arm",
            "sharing_on_migration_on",
            "--adapter_assignment_profile",
            "semantic_ai_service",
            "--mobility_source",
            "ngsim",
            "--mobility_csv_path",
            str(mobility),
            "--workflow_csv_path",
            str(workflow),
            "--max_mobility_rows",
            str(data["max_mobility_rows"]),
            "--max_workflows",
            str(data["max_workflows"]),
            "--workflow_selector",
            "ordered",
            "--min_tasks",
            str(data["min_tasks"]),
            "--max_tasks",
            str(data["max_tasks"]),
            "--window_plan_path",
            str(window_plan),
            "--primary_vehicle_selection",
            "handoff_pressure",
            "--reward_positive_offset",
            "0",
            "--formal-exogenous-request-execution",
            "--non-formal-rehearsal",
            "--output_root",
            str(output_root / "runs"),
            "--run_id",
            run_id,
        ]
        commands.append(
            {
                "agent_name": str(agent_name),
                "run_id": run_id,
                "command": command,
                "stdout_path": str(output_root / f"{agent_name}.stdout.log"),
                "stderr_path": str(output_root / f"{agent_name}.stderr.log"),
            }
        )
    return {
        "runner_version": RUNNER_VERSION,
        "study_version": str(config["study_version"]),
        "created_at": _utc_now(),
        "evidence_scope": "observed_data_controlled_training_supplement_not_holdout",
        "formal": False,
        "independent_test": False,
        "algorithm_superiority_claim_allowed": False,
        "cwd": str(ROOT_DIR),
        "output_root": str(output_root),
        "python_executable": str(python),
        "config_path": str(config_path),
        "config_sha256": _file_sha256(config_path),
        "runtime_config": {"path": str(runtime_config), "sha256": _file_sha256(runtime_config)},
        "window_plan": {"path": str(window_plan), "sha256": _file_sha256(window_plan)},
        "sources": sources,
        "budget": budget,
        "agents": list(config["agents"]),
        "commands": commands,
        "automatic_retry_count": 0,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "claim_boundary": str(config["claim_boundary"]),
    }


def launch(config_path: Path, data_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse background output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    manifest = _build_manifest(config_path, data_root, output_root)
    manifest_path = output_root / "command_manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha256 = _file_sha256(manifest_path)
    stdout_path = output_root / "outer.stdout.log"
    stderr_path = output_root / "outer.stderr.log"
    command = [
        str(Path(manifest["python_executable"])),
        str(Path(__file__).resolve()),
        "run",
        "--manifest",
        str(manifest_path),
        "--manifest-sha256",
        manifest_sha256,
    ]
    with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open(
        "ab", buffering=0
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    receipt = {
        "runner_version": RUNNER_VERSION,
        "status": "RUNNING",
        "launched_at": _utc_now(),
        "pid": process.pid,
        "fixed_command": command,
        "command_manifest_path": str(manifest_path),
        "command_manifest_sha256": manifest_sha256,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "state_path": str(output_root / "state.json"),
        "exit_receipt_path": str(output_root / "completion_receipt.json"),
        "automatic_retry_count": 0,
    }
    _write_json(output_root / "launch_receipt.json", receipt)
    return receipt


def run(manifest_path: Path, expected_sha256: str) -> int:
    if _file_sha256(manifest_path) != expected_sha256:
        raise ValueError("command manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_root = Path(str(manifest["output_root"])).resolve()
    state_path = output_root / "state.json"
    started_at = _utc_now()
    _write_json(
        state_path,
        {
            "runner_version": RUNNER_VERSION,
            "status": "RUNNING",
            "pid": os.getpid(),
            "started_at": started_at,
            "command_manifest_path": str(manifest_path),
            "command_manifest_sha256": expected_sha256,
        },
    )
    child_receipts: list[dict[str, Any]] = []
    final_code = 0
    for row in list(manifest["commands"]):
        child_started = _utc_now()
        stdout_path = Path(str(row["stdout_path"]))
        stderr_path = Path(str(row["stderr_path"]))
        with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open(
            "ab", buffering=0
        ) as stderr:
            completed = subprocess.run(
                list(row["command"]),
                cwd=str(manifest["cwd"]),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        child_receipts.append(
            {
                "agent_name": row["agent_name"],
                "run_id": row["run_id"],
                "started_at": child_started,
                "completed_at": _utc_now(),
                "return_code": completed.returncode,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
        )
        if completed.returncode != 0:
            final_code = completed.returncode
            break
    completion = {
        "runner_version": RUNNER_VERSION,
        "status": "SUCCEEDED" if final_code == 0 else "FAILED",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "return_code": final_code,
        "command_manifest_path": str(manifest_path),
        "command_manifest_sha256": expected_sha256,
        "child_receipts": child_receipts,
        "expected_child_count": len(manifest["commands"]),
        "completed_child_count": len(child_receipts),
        "automatic_retry_count": 0,
        "evidence_scope": manifest["evidence_scope"],
        "algorithm_superiority_claim_allowed": False,
    }
    _write_json(output_root / "completion_receipt.json", completion)
    _write_json(state_path, completion)
    return final_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--config", type=Path, required=True)
    launch_parser.add_argument("--data-root", type=Path, required=True)
    launch_parser.add_argument("--output-root", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--manifest-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "launch":
        receipt = launch(
            args.config.resolve(), args.data_root.resolve(), args.output_root.resolve()
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    return run(args.manifest.resolve(), str(args.manifest_sha256))


if __name__ == "__main__":
    raise SystemExit(main())
