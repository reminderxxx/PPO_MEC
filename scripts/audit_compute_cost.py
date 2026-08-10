"""Build an auditable compute table from checkpoints and timed benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit training and inference compute costs.")
    parser.add_argument("--seed_checkpoint_manifest_path", type=str, required=True)
    parser.add_argument("--training_command_log_path", type=str, required=True)
    parser.add_argument("--benchmark_run_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT_DIR / path


def tensor_count(payload: Any) -> int:
    if torch.is_tensor(payload):
        return int(payload.numel())
    if isinstance(payload, dict):
        return sum(tensor_count(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return sum(tensor_count(value) for value in payload)
    return 0


def checkpoint_metrics(checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint must contain a mapping: {checkpoint_path}")
    transition_state = payload.get("learned_transition_model_state")
    transition_parameters = 0
    if isinstance(transition_state, dict):
        transition_parameters = tensor_count(transition_state.get("models", []))
    policy_parameters = tensor_count(payload.get("network_state_dict", {}))
    target_parameters = tensor_count(payload.get("target_network_state_dict", {}))
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "policy_parameter_count": policy_parameters,
        "target_parameter_count_training_only": target_parameters,
        "learned_dynamics_parameter_count": transition_parameters,
        "deployed_parameter_count": policy_parameters + transition_parameters,
        "training_parameter_footprint_count": policy_parameters + target_parameters + transition_parameters,
        "update_count": int(payload.get("update_count", 0) or 0),
    }


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"status": "unavailable", "sample_count": 0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered) + 0.999999) - 1))
    return {
        "status": "measured",
        "sample_count": len(values),
        "mean": round(statistics.fmean(values), 9),
        "median": round(statistics.median(values), 9),
        "min": round(min(values), 9),
        "max": round(max(values), 9),
        "p95": round(ordered[p95_index], 9),
        "population_std": round(statistics.pstdev(values), 9) if len(values) > 1 else 0.0,
    }


def load_training_times(command_log_path: Path) -> dict[tuple[str, str], float]:
    payload = load_json(command_log_path)
    commands = payload.get("commands", []) if isinstance(payload, dict) else payload
    output: dict[tuple[str, str], float] = {}
    for item in commands if isinstance(commands, list) else []:
        label = str(item.get("label", ""))
        if not label.startswith("train_") or item.get("returncode") != 0:
            continue
        suffix = label.removeprefix("train_")
        if "_seed_" not in suffix:
            continue
        agent_name, seed = suffix.rsplit("_seed_", 1)
        output[(agent_name, seed)] = float(item["elapsed_sec"])
    return output


def load_evaluation_metrics(benchmark_root: Path) -> dict[str, dict[str, list[float]]]:
    by_agent: dict[str, dict[str, list[float]]] = {}
    for summary_path in benchmark_root.rglob("seed_*.summary.json"):
        payload = load_json(summary_path)
        run_info = payload.get("run_info", {})
        agent_name = str(run_info.get("agent_name", ""))
        if not agent_name:
            continue
        values = by_agent.setdefault(
            agent_name,
            {
                "wall_clock_sec": [],
                "wall_clock_sec_per_step": [],
                "python_peak_increment_bytes": [],
                "model_query_count": [],
                "model_transition_count": [],
            },
        )
        compute = payload.get("compute_audit", {})
        if compute:
            values["wall_clock_sec"].append(float(compute.get("wall_clock_sec", 0.0) or 0.0))
            values["wall_clock_sec_per_step"].append(
                float(compute.get("wall_clock_sec_per_step", 0.0) or 0.0)
            )
            values["python_peak_increment_bytes"].append(
                float(compute.get("python_peak_increment_bytes", 0.0) or 0.0)
            )
        trainer_info = payload.get("trainer_info", {})
        values["model_query_count"].append(
            float(trainer_info.get("counterfactual_model_query_count", 0.0) or 0.0)
        )
        values["model_transition_count"].append(
            float(trainer_info.get("counterfactual_model_transition_count", 0.0) or 0.0)
        )
    return by_agent


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def main() -> None:
    args = parse_args()
    manifest_path = resolve_path(args.seed_checkpoint_manifest_path)
    command_log_path = resolve_path(args.training_command_log_path)
    benchmark_root = resolve_path(args.benchmark_run_root)
    output_dir = resolve_path(args.output_dir)
    manifest = load_json(manifest_path)
    training_times = load_training_times(command_log_path)
    evaluation = load_evaluation_metrics(benchmark_root)

    rows: list[dict[str, Any]] = []
    by_agent: dict[str, Any] = {}
    for agent_name in sorted(set(manifest) | set(evaluation)):
        seed_map = manifest.get(agent_name, {})
        checkpoints = []
        elapsed_values = []
        for seed, path_text in sorted(seed_map.items(), key=lambda item: int(item[0])):
            checkpoint = checkpoint_metrics(resolve_path(str(path_text)))
            checkpoint["seed"] = int(seed)
            checkpoints.append(checkpoint)
            elapsed = training_times.get((agent_name, str(seed)))
            if elapsed is not None:
                elapsed_values.append(elapsed)
        eval_values = evaluation.get(agent_name, {})
        representative = checkpoints[0] if checkpoints else {
            "policy_parameter_count": 0,
            "target_parameter_count_training_only": 0,
            "learned_dynamics_parameter_count": 0,
            "deployed_parameter_count": 0,
            "training_parameter_footprint_count": 0,
        }
        agent_summary = {
            "agent_kind": "learned" if checkpoints else "checkpoint_free",
            "checkpoint_count": len(checkpoints),
            "checkpoint_metrics_by_seed": checkpoints,
            "policy_parameter_count": representative["policy_parameter_count"],
            "target_parameter_count_training_only": representative["target_parameter_count_training_only"],
            "learned_dynamics_parameter_count": representative["learned_dynamics_parameter_count"],
            "deployed_parameter_count": representative["deployed_parameter_count"],
            "training_parameter_footprint_count": representative["training_parameter_footprint_count"],
            "training_wall_clock_sec": summarize(elapsed_values),
            "inference_wall_clock_sec": summarize(eval_values.get("wall_clock_sec", [])),
            "inference_wall_clock_sec_per_step": summarize(eval_values.get("wall_clock_sec_per_step", [])),
            "python_peak_increment_bytes": summarize(eval_values.get("python_peak_increment_bytes", [])),
            "model_query_count_per_episode": summarize(eval_values.get("model_query_count", [])),
            "model_transition_count_per_episode": summarize(eval_values.get("model_transition_count", [])),
        }
        by_agent[agent_name] = agent_summary
        rows.append(
            {
                "agent_name": agent_name,
                "policy_parameter_count": agent_summary["policy_parameter_count"],
                "learned_dynamics_parameter_count": agent_summary["learned_dynamics_parameter_count"],
                "deployed_parameter_count": agent_summary["deployed_parameter_count"],
                "training_wall_clock_sec_mean": agent_summary["training_wall_clock_sec"].get("mean", ""),
                "inference_wall_clock_sec_per_step_mean": agent_summary["inference_wall_clock_sec_per_step"].get("mean", ""),
                "python_peak_increment_bytes_mean": agent_summary["python_peak_increment_bytes"].get("mean", ""),
                "model_query_count_per_episode_mean": agent_summary["model_query_count_per_episode"].get("mean", ""),
                "model_transition_count_per_episode_mean": agent_summary["model_transition_count_per_episode"].get("mean", ""),
            }
        )

    payload = {
        "audit_version": "compute_audit_v1",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "seed_checkpoint_manifest_path": str(manifest_path),
        "training_command_log_path": str(command_log_path),
        "benchmark_run_root": str(benchmark_root),
        "memory_scope": "tracemalloc_python_allocations_only; not total process RSS",
        "training_time_scope": "subprocess wall-clock from the original equal-budget training command log",
        "inference_time_scope": "run_real_episode wall-clock from --audit_runtime benchmark summaries",
        "agents": by_agent,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "compute_audit.json"
    csv_path = output_dir / "compute_audit.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"compute_audit_json: {json_path}")
    print(f"compute_audit_csv: {csv_path}")


if __name__ == "__main__":
    main()
