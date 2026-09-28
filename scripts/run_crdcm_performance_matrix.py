#!/usr/bin/env python3
"""Preflight, smoke, or launch the frozen CRDCM v2 development matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNNER_VERSION = "crdcm_performance_matrix_runner_v2"
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_integrity_manifest(output_root: Path) -> None:
    target = output_root / "artifact_integrity_manifest.json"
    files = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path == target:
            continue
        files.append(
            {
                "path": str(path.relative_to(output_root)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_json(
        target,
        {
            "artifact_integrity_manifest_version": "1.0.0",
            "self_excluding": True,
            "status": "pass",
            "file_count": len(files),
            "files": files,
        },
    )


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("CRDCM matrix config must be a YAML mapping")
    return value


def _git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _configured_python(value: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(value)))


def _python_preflight(python: Path) -> dict[str, Any]:
    if not python.is_file():
        raise FileNotFoundError(python)
    probe = (
        "import json,sys,torch,yaml;"
        "print(json.dumps({'sys_executable':sys.executable,'sys_prefix':sys.prefix,"
        "'torch_version':torch.__version__,'yaml_version':yaml.__version__}))"
    )
    completed = subprocess.run(
        [str(python), "-c", probe],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "configured Python dependency preflight failed: "
            f"return_code={completed.returncode}; stderr={completed.stderr.strip()}"
        )
    identity = json.loads(completed.stdout.strip())
    expected_prefix = python.parent.parent
    observed_prefix = Path(str(identity["sys_prefix"]))
    if expected_prefix.name == ".venv" and observed_prefix.resolve() != expected_prefix.resolve():
        raise RuntimeError(
            "configured virtualenv entrypoint did not preserve sys.prefix: "
            f"{observed_prefix} != {expected_prefix}"
        )
    entrypoints = [
        ROOT / "scripts/train_algo_pool_real_sample.py",
        ROOT / "scripts/evaluate_crdcm_frozen_checkpoints.py",
    ]
    for entrypoint in entrypoints:
        help_probe = subprocess.run(
            [str(python), str(entrypoint), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if help_probe.returncode != 0:
            raise RuntimeError(
                f"entrypoint import preflight failed for {entrypoint.name}: "
                f"return_code={help_probe.returncode}; stderr={help_probe.stderr.strip()}"
            )
    return {
        "configured_path": str(python),
        "configured_realpath": os.path.realpath(python),
        "observed_sys_executable": str(identity["sys_executable"]),
        "observed_sys_prefix": str(identity["sys_prefix"]),
        "torch_version": str(identity["torch_version"]),
        "yaml_version": str(identity["yaml_version"]),
        "entrypoint_import_preflight": "passed",
    }


def _resolve_data_path(data_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else data_root / path).resolve()


def _validate_source(path: Path, expected: dict[str, Any], label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        prefix = handle.read(len(LFS_HEADER))
    if prefix == LFS_HEADER:
        raise ValueError(
            f"{label} resolves to a Git LFS pointer, not dataset content: {path}"
        )
    size = path.stat().st_size
    if size != int(expected["size_bytes"]):
        raise ValueError(f"{label} size mismatch: {size}")
    digest = _sha256(path)
    if digest != str(expected["sha256"]):
        raise ValueError(f"{label} SHA-256 mismatch: {digest}")
    return {"path": str(path), "size_bytes": size, "sha256": digest}


def _validate_window_intervals(window_plan_path: Path) -> dict[str, Any]:
    payload = json.loads(window_plan_path.read_text(encoding="utf-8"))
    windows = [dict(item) for item in payload["selected_window_plan"]]
    comparisons: list[dict[str, Any]] = []
    safe = True
    for left_index, left in enumerate(windows):
        for right in windows[left_index + 1 :]:
            frame_overlap = not (
                int(left["segment_frame_end"]) < int(right["segment_frame_start"])
                or int(right["segment_frame_end"]) < int(left["segment_frame_start"])
            )
            time_overlap = not (
                int(left["time_index_end"]) < int(right["time_index_start"])
                or int(right["time_index_end"]) < int(left["time_index_start"])
            )
            pair_safe = not frame_overlap and not time_overlap
            safe = safe and pair_safe
            comparisons.append(
                {
                    "left": left["window_id"],
                    "right": right["window_id"],
                    "frame_overlap": frame_overlap,
                    "time_overlap": time_overlap,
                    "safe": pair_safe,
                }
            )
    if not safe:
        raise ValueError("CRDCM development windows overlap in raw frame/time intervals")
    return {
        "window_count": len(windows),
        "pair_count": len(comparisons),
        "all_pairwise_nonoverlap": safe,
        "comparisons": comparisons,
    }


def _validate_scientific_config(config: dict[str, Any]) -> None:
    if config.get("study_version") != "crdcm_performance_matrix_v2":
        raise ValueError("unexpected CRDCM matrix version")
    seeds = [int(item) for item in config["seeds"]]
    seed_rule = dict(config["seed_rule"])
    expected_seeds = [
        int(seed_rule["base_seed"]) + int(offset) for offset in seed_rule["offsets"]
    ]
    if seeds != expected_seeds or len(set(seeds)) != 3:
        raise ValueError("seed list does not match the frozen pre-outcome seed rule")
    conditions = [dict(item) for item in config["conditions"]]
    if len(conditions) != 4 or len({item["condition_id"] for item in conditions}) != 4:
        raise ValueError("CRDCM v2 requires exactly four unique learned conditions")
    expected_conditions = {
        ("crdcm_full_sa", "crdcm_sa_ghmappo", "full"),
        ("crdcm_signal_off_sa", "crdcm_sa_ghmappo", "signal_off"),
        ("crdcm_full_mappo", "crdcm_mappo", "full"),
        ("crdcm_full_ppo", "crdcm_ppo", "full"),
    }
    observed_conditions = {
        (str(item["condition_id"]), str(item["agent_name"]), str(item["crdcm_feature_mode"]))
        for item in conditions
    }
    if observed_conditions != expected_conditions:
        raise ValueError("condition-to-agent/feature mapping drift")
    hyper = dict(config["hyperparameters"])
    budget = dict(config["budget"])
    training_episodes = len(conditions) * len(seeds) * int(
        hyper["episodes_per_condition_seed"]
    )
    if training_episodes != int(budget["training_episodes"]):
        raise ValueError("training episode budget mismatch")
    if training_episodes * int(hyper["max_steps"]) != int(
        budget["training_environment_step_cap"]
    ):
        raise ValueError("training step cap mismatch")
    evaluation = dict(config["evaluation"])
    checkpoint_count = len(conditions) * len(seeds)
    paired_units = len(evaluation["pairs"]) * len(evaluation["scenarios"])
    learned_eval = checkpoint_count * paired_units
    heuristic_eval = paired_units
    if checkpoint_count != int(budget["learned_checkpoint_count"]):
        raise ValueError("learned checkpoint count mismatch")
    if learned_eval != int(budget["learned_evaluation_episodes"]):
        raise ValueError("learned evaluation episode budget mismatch")
    if heuristic_eval != int(budget["heuristic_evaluation_episodes"]):
        raise ValueError("heuristic evaluation episode budget mismatch")
    if training_episodes + learned_eval + heuristic_eval != int(
        budget["full_execution_episode_cap"]
    ):
        raise ValueError("full execution episode cap mismatch")
    if int(budget["full_execution_episode_cap"]) > int(
        budget["prior_proposal_training_episode_upper_bound"]
    ):
        raise ValueError("combined matrix exceeds the previously discussed episode cap")
    if int(budget["full_execution_environment_step_cap"]) > int(
        budget["prior_proposal_training_environment_step_upper_bound"]
    ):
        raise ValueError("combined matrix exceeds the previously discussed step cap")


def _build_manifest(
    config_path: Path,
    data_root: Path,
    output_root: Path,
    *,
    smoke: bool,
) -> dict[str, Any]:
    config = _load_config(config_path)
    _validate_scientific_config(config)
    python = _configured_python(str(config["python_executable"]))
    python_identity = _python_preflight(python)
    data = dict(config["training_data"])
    expected = dict(config["expected_sources"])
    mobility = _resolve_data_path(data_root, str(data["mobility_csv_path"]))
    workflow = _resolve_data_path(data_root, str(data["workflow_csv_path"]))
    sources = {
        "mobility": _validate_source(mobility, dict(expected["mobility"]), "mobility"),
        "workflow": _validate_source(workflow, dict(expected["workflow"]), "workflow"),
    }
    runtime_config = (ROOT / str(config["runtime_config"])).resolve()
    window_plan = (ROOT / str(config["training_window_plan"])).resolve()
    for path in (runtime_config, window_plan):
        if not path.is_file():
            raise FileNotFoundError(path)
    interval_audit = _validate_window_intervals(window_plan)
    hyper = dict(config["hyperparameters"])
    smoke_budget = dict(config["smoke_budget"])
    seeds = [int(config["seeds"][0])] if smoke else [int(item) for item in config["seeds"]]
    episodes = (
        int(smoke_budget["training_episodes_per_condition"])
        if smoke
        else int(hyper["episodes_per_condition_seed"])
    )
    commands: list[dict[str, Any]] = []
    for condition in config["conditions"]:
        condition = dict(condition)
        for seed in seeds:
            run_id = f"{config['study_version']}_{condition['condition_id']}_seed{seed}"
            if smoke:
                run_id += "_implementation_smoke"
            command = [
                str(python),
                str(ROOT / "scripts/train_algo_pool_real_sample.py"),
                "--agent_name", str(condition["agent_name"]),
                "--profile", "baseline_safe",
                "--episodes", str(episodes),
                "--update_every", str(hyper["update_every"]),
                "--batch_size", str(hyper["batch_size"]),
                "--max_steps", str(hyper["max_steps"]),
                "--checkpoint_every_updates", str(hyper["checkpoint_every_updates"]),
                "--learning_rate", str(hyper["learning_rate"]),
                "--clip_ratio", str(hyper["clip_ratio"]),
                "--entropy_coef", str(hyper["entropy_coef"]),
                "--value_coef", str(hyper["value_coef"]),
                "--gamma", str(hyper["gamma"]),
                "--gae_lambda", str(hyper["gae_lambda"]),
                "--prediction_horizon", str(hyper["prediction_horizon"]),
                "--random_seed", str(seed),
                "--crdcm_condition_id", str(condition["condition_id"]),
                "--crdcm_feature_mode", str(condition["crdcm_feature_mode"]),
                "--model_cache_runtime_config", str(runtime_config),
                "--mechanism_factorial_arm", str(data["mechanism_factorial_arm"]),
                "--adapter_assignment_profile", str(data["adapter_assignment_profile"]),
                "--mobility_source", "ngsim",
                "--mobility_csv_path", str(mobility),
                "--workflow_csv_path", str(workflow),
                "--max_mobility_rows", str(data["max_mobility_rows"]),
                "--max_workflows", str(data["max_workflows"]),
                "--workflow_selector", str(data["workflow_selector"]),
                "--min_tasks", str(data["min_tasks"]),
                "--max_tasks", str(data["max_tasks"]),
                "--window_plan_path", str(window_plan),
                "--primary_vehicle_selection", str(data["primary_vehicle_selection"]),
                "--reward_positive_offset", str(hyper["reward_positive_offset"]),
                "--formal-exogenous-request-execution",
                "--non-formal-rehearsal",
                "--output_root", str(output_root / "runs"),
                "--run_id", run_id,
            ]
            run_dir = output_root / "runs" / str(condition["agent_name"]) / run_id
            commands.append(
                {
                    "condition_id": str(condition["condition_id"]),
                    "agent_name": str(condition["agent_name"]),
                    "crdcm_feature_mode": str(condition["crdcm_feature_mode"]),
                    "seed": seed,
                    "episodes": episodes,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "train_summary_path": str(run_dir / "train_summary.json"),
                    "command": command,
                    "stdout_path": str(output_root / "logs" / f"{run_id}.stdout.log"),
                    "stderr_path": str(output_root / "logs" / f"{run_id}.stderr.log"),
                }
            )
    return {
        "runner_version": RUNNER_VERSION,
        "study_version": config["study_version"],
        "execution_scope": "implementation_smoke" if smoke else "full_observed_development_matrix",
        "smoke": smoke,
        "created_at": _utc_now(),
        "cwd": str(ROOT),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "python_executable": str(python),
        "python_identity": python_identity,
        "runtime_config": {"path": str(runtime_config), "sha256": _sha256(runtime_config)},
        "window_plan": {"path": str(window_plan), "sha256": _sha256(window_plan)},
        "window_interval_audit": interval_audit,
        "sources": sources,
        "hf_manifest_required": False,
        "hf_manifest_reason": "CRDCM uses the repository typed controlled catalog plus NGSIM and Alibaba; HF size metadata is not consumed",
        "conditions": config["conditions"],
        "seeds": seeds,
        "hyperparameters": hyper,
        "budget": smoke_budget if smoke else config["budget"],
        "commands": commands,
        "automatic_retry_count": 0,
        "formal": False,
        "holdout": False,
        "performance_result_eligible": not smoke,
        "algorithm_superiority_claim_allowed": False,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "claim_boundary": config["claim_boundary"],
    }


def _checkpoint_entry(row: dict[str, Any]) -> dict[str, Any]:
    summary_path = Path(str(row["train_summary_path"]))
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["episodes"]) != int(row["episodes"]):
        raise ValueError("training summary episode count mismatch")
    if summary.get("crdcm_condition_id") != row["condition_id"]:
        raise ValueError("training condition identity mismatch")
    if summary.get("crdcm_feature_mode") != row["crdcm_feature_mode"]:
        raise ValueError("training feature mode mismatch")
    parameter_identity = dict(summary.get("parameter_identity", {}))
    before = dict(parameter_identity.get("before_training", {}))
    after = dict(parameter_identity.get("after_training", {}))
    if not parameter_identity.get("changed") or before.get("sha256") == after.get("sha256"):
        raise RuntimeError("training did not change model parameters")
    if not before.get("all_finite") or not after.get("all_finite"):
        raise RuntimeError("non-finite model parameter detected")
    checkpoint = Path(str(summary["latest_checkpoint_path"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return {
        "condition_id": row["condition_id"],
        "agent_name": row["agent_name"],
        "crdcm_feature_mode": row["crdcm_feature_mode"],
        "seed": int(row["seed"]),
        "episodes": int(summary["episodes"]),
        "update_count": int(summary["update_count"]),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": _sha256(checkpoint),
        "parameter_identity": parameter_identity,
        "train_summary_path": str(summary_path),
        "train_summary_sha256": _sha256(summary_path),
    }


def run_manifest(manifest_path: Path, expected_sha256: str) -> int:
    output_root = manifest_path.parent.resolve()
    started_at = _utc_now()
    manifest: dict[str, Any] | None = None
    child_receipts: list[dict[str, Any]] = []
    evaluation_receipt: dict[str, Any] | None = None
    final_code = 1
    failure: str | None = None
    try:
        if _sha256(manifest_path) != expected_sha256:
            raise ValueError("command manifest SHA-256 mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if Path(str(manifest["output_root"])).resolve() != output_root:
            raise ValueError("manifest output root mismatch")
        expected_prefix = Path(str(manifest["python_identity"]["observed_sys_prefix"]))
        if expected_prefix.resolve() != Path(sys.prefix).resolve():
            raise RuntimeError("matrix runner Python environment differs from preflight")
        _write_json(
            output_root / "state.json",
            {
                "runner_version": RUNNER_VERSION,
                "status": "RUNNING",
                "pid": os.getpid(),
                "started_at": started_at,
                "manifest_sha256": expected_sha256,
                "execution_scope": manifest["execution_scope"],
            },
        )
        checkpoint_entries: list[dict[str, Any]] = []
        for row in manifest["commands"]:
            child_started = _utc_now()
            stdout_path = Path(str(row["stdout_path"]))
            stderr_path = Path(str(row["stderr_path"]))
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open(
                "ab", buffering=0
            ) as stderr:
                completed = subprocess.run(
                    list(row["command"]),
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            receipt = {
                "condition_id": row["condition_id"],
                "seed": row["seed"],
                "run_id": row["run_id"],
                "started_at": child_started,
                "completed_at": _utc_now(),
                "return_code": completed.returncode,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
            child_receipts.append(receipt)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"training child {row['condition_id']}/seed_{row['seed']} exited {completed.returncode}"
                )
            checkpoint_entries.append(_checkpoint_entry(row))
        checkpoint_manifest = {
            "checkpoint_manifest_version": "crdcm_checkpoint_manifest_v2",
            "study_version": manifest["study_version"],
            "execution_scope": manifest["execution_scope"],
            "created_at": _utc_now(),
            "fixed_endpoint": (
                "latest.pt after implementation smoke"
                if manifest["smoke"]
                else "latest.pt after episode 64 / update 16"
            ),
            "metric_based_selection": False,
            "entries": checkpoint_entries,
        }
        checkpoint_manifest_path = output_root / "checkpoint_manifest.json"
        _write_json(checkpoint_manifest_path, checkpoint_manifest)
        evaluation_command = [
            str(manifest["python_executable"]),
            str(ROOT / "scripts/evaluate_crdcm_frozen_checkpoints.py"),
            "--config", str(manifest["config_path"]),
            "--data-root", str(manifest["data_root"]),
            "--checkpoint-manifest", str(checkpoint_manifest_path),
            "--output-dir", str(output_root / "evaluation"),
        ]
        if manifest["smoke"]:
            evaluation_command.append("--smoke")
        evaluation_stdout = output_root / "logs" / "evaluation.stdout.log"
        evaluation_stderr = output_root / "logs" / "evaluation.stderr.log"
        with evaluation_stdout.open("ab", buffering=0) as stdout, evaluation_stderr.open(
            "ab", buffering=0
        ) as stderr:
            evaluated = subprocess.run(
                evaluation_command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        evaluation_receipt = {
            "command": evaluation_command,
            "return_code": evaluated.returncode,
            "stdout_path": str(evaluation_stdout),
            "stderr_path": str(evaluation_stderr),
            "result_path": str(output_root / "evaluation" / "evaluation_receipt.json"),
        }
        if evaluated.returncode != 0:
            raise RuntimeError(f"paired evaluation exited {evaluated.returncode}")
        final_code = 0
    except BaseException as exc:
        final_code = 1
        failure = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    completion = {
        "runner_version": RUNNER_VERSION,
        "status": "SUCCEEDED" if final_code == 0 else "FAILED",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "return_code": final_code,
        "failure": failure,
        "command_manifest_path": str(manifest_path),
        "command_manifest_sha256": expected_sha256,
        "child_receipts": child_receipts,
        "expected_child_count": len(manifest.get("commands", [])) if manifest else 0,
        "completed_child_count": len(child_receipts),
        "evaluation_receipt": evaluation_receipt,
        "automatic_retry_count": 0,
        "performance_result_eligible": bool(manifest and not manifest.get("smoke")),
        "algorithm_superiority_claim_allowed": False,
    }
    _write_json(output_root / "completion_receipt.json", completion)
    _write_json(output_root / "state.json", completion)
    _write_integrity_manifest(output_root)
    return final_code


def _prepare_output(
    config_path: Path,
    data_root: Path,
    output_root: Path,
    *,
    smoke: bool,
) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse CRDCM output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        manifest = _build_manifest(config_path, data_root, output_root, smoke=smoke)
        manifest_path = output_root / "command_manifest.json"
        _write_json(manifest_path, manifest)
        return manifest, manifest_path, _sha256(manifest_path)
    except BaseException as exc:
        failure = {
            "runner_version": RUNNER_VERSION,
            "status": "FAILED",
            "failed_at": _utc_now(),
            "reason": f"{type(exc).__name__}: {exc}",
            "phase": "preflight_before_training",
            "training_episode_count": 0,
            "automatic_retry_count": 0,
        }
        _write_json(output_root / "startup_failure_receipt.json", failure)
        _write_json(
            output_root / "launch_receipt.json",
            {
                **failure,
                "startup_confirmed": False,
                "startup_failure_receipt_path": str(
                    output_root / "startup_failure_receipt.json"
                ),
            },
        )
        return None, None, None


def launch(config_path: Path, data_root: Path, output_root: Path) -> dict[str, Any]:
    manifest, manifest_path, manifest_sha256 = _prepare_output(
        config_path, data_root, output_root, smoke=False
    )
    if manifest is None or manifest_path is None or manifest_sha256 is None:
        return json.loads((output_root / "launch_receipt.json").read_text(encoding="utf-8"))
    outer_stdout = output_root / "outer.stdout.log"
    outer_stderr = output_root / "outer.stderr.log"
    command = [
        str(manifest["python_executable"]),
        str(Path(__file__).resolve()),
        "run",
        "--manifest", str(manifest_path),
        "--manifest-sha256", manifest_sha256,
    ]
    with outer_stdout.open("ab", buffering=0) as stdout, outer_stderr.open(
        "ab", buffering=0
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    receipt = {
        "runner_version": RUNNER_VERSION,
        "status": "STARTING",
        "launched_at": _utc_now(),
        "pid": process.pid,
        "fixed_command": command,
        "command_manifest_path": str(manifest_path),
        "command_manifest_sha256": manifest_sha256,
        "stdout_path": str(outer_stdout),
        "stderr_path": str(outer_stderr),
        "state_path": str(output_root / "state.json"),
        "exit_receipt_path": str(output_root / "completion_receipt.json"),
        "automatic_retry_count": 0,
        "startup_confirmed": False,
    }
    _write_json(output_root / "launch_receipt.json", receipt)
    for _ in range(250):
        state_path = output_root / "state.json"
        completion_path = output_root / "completion_receipt.json"
        if completion_path.is_file():
            terminal = json.loads(completion_path.read_text(encoding="utf-8"))
            receipt.update(
                status=str(terminal.get("status", "FAILED")),
                startup_failure="run reached terminal state before startup confirmation",
            )
            _write_json(output_root / "launch_receipt.json", receipt)
            return receipt
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") == "RUNNING" and process.poll() is None:
                receipt.update(
                    status="RUNNING",
                    startup_confirmed=True,
                    confirmed_child_pid=int(state["pid"]),
                )
                _write_json(output_root / "launch_receipt.json", receipt)
                return receipt
        if process.poll() is not None:
            break
        time.sleep(0.02)
    failure = {
        "runner_version": RUNNER_VERSION,
        "status": "FAILED",
        "failed_at": _utc_now(),
        "reason": "background process exited or failed to publish RUNNING within five seconds",
        "pid": process.pid,
        "return_code": process.poll(),
        "automatic_retry_count": 0,
    }
    _write_json(output_root / "startup_failure_receipt.json", failure)
    receipt.update(
        status="FAILED",
        startup_confirmed=False,
        startup_failure=failure["reason"],
        startup_failure_receipt_path=str(output_root / "startup_failure_receipt.json"),
    )
    _write_json(output_root / "launch_receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for name in ("preflight", "smoke", "launch"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--data-root", type=Path, required=True)
        if name != "preflight":
            child.add_argument("--output-root", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--manifest-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "run":
        return run_manifest(args.manifest.resolve(), str(args.manifest_sha256))
    config_path = args.config.resolve()
    data_root = args.data_root.resolve()
    if args.mode == "preflight":
        temporary_root = ROOT / "artifacts" / ".crdcm_preflight_not_created"
        manifest = _build_manifest(config_path, data_root, temporary_root, smoke=False)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    output_root = args.output_root.resolve()
    if args.mode == "launch":
        receipt = launch(config_path, data_root, output_root)
        print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if receipt.get("status") == "RUNNING" else 1
    manifest, manifest_path, manifest_sha256 = _prepare_output(
        config_path, data_root, output_root, smoke=True
    )
    if manifest is None or manifest_path is None or manifest_sha256 is None:
        return 1
    _write_json(
        output_root / "smoke_launch_receipt.json",
        {
            "runner_version": RUNNER_VERSION,
            "status": "RUNNING_FOREGROUND_SMOKE",
            "started_at": _utc_now(),
            "command_manifest_path": str(manifest_path),
            "command_manifest_sha256": manifest_sha256,
            "performance_result_eligible": False,
        },
    )
    return run_manifest(manifest_path, manifest_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
