"""Train and evaluate the matched calibrated continuous-workflow pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.registry import build_agent
from src.envs.core.calibrated_continuous_workflow_env import (
    CalibratedContinuousWorkflowEnv,
    TwoStepCostRule,
)
from src.trainers.ppo_buffer import PPORolloutBuffer


PRIMARY_METRICS = (
    "workflow_completion_rate",
    "modeled_completion_seconds",
    "total_transfer_mb",
    "recompute_seconds",
    "handoff_failure_rate",
    "reward",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _collect_training_episode(
    agent: Any,
    config: dict[str, Any],
    instance: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = CalibratedContinuousWorkflowEnv(config, instance)
    observation, info = env.reset()
    buffer = PPORolloutBuffer()
    terminated = truncated = False
    while not terminated and not truncated:
        decision_info = dict(info)
        action, action_info = agent.act(observation, decision_info)
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        value = float(action_info.get("value", agent.evaluate_value(observation, decision_info)))
        buffer.add_step(
            observation=observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            log_prob=float(action_info.get("log_prob", 0.0)),
            value=value,
            next_observation=next_observation,
            action_info=action_info,
            decision_info=decision_info,
            env_info=next_info,
        )
        observation, info = next_observation, next_info
    last_value = 0.0 if terminated else float(agent.evaluate_value(observation, info))
    training = config["training"]
    buffer.finalize(
        last_value=last_value,
        gamma=float(training["gamma"]),
        gae_lambda=float(training["gae_lambda"]),
    )
    return buffer.to_training_rows(), env.summary()


def _evaluation_row(
    *,
    method: str,
    seed: int | None,
    env: CalibratedContinuousWorkflowEnv,
    action_counts: Counter[int],
) -> dict[str, Any]:
    summary = env.summary()
    handoffs = int(summary["handoff_count"])
    return {
        "method": method,
        "seed": "rule" if seed is None else seed,
        "design_id": summary["design_id"],
        "window_id": summary["window_id"],
        "workflow_id": summary["workflow_id"],
        "sharing": env.instance["factors"]["sharing"],
        "capacity": env.instance["factors"]["capacity"],
        "state_scale": env.instance["factors"]["state_scale"],
        "initial_target": env.instance["factors"]["initial_target"],
        "handoff_pressure": env.instance["trace_features"]["handoff_pressure"],
        "topology_class": env.instance["workflow_features"]["topology_class"],
        "node_count": summary["node_count"],
        "workflow_completion_rate": float(summary["workflow_completed"]),
        "modeled_completion_seconds": float(summary["modeled_completion_seconds"]),
        "total_transfer_mb": float(summary["total_transfer_bytes"]) / 1_000_000.0,
        "recompute_seconds": float(summary["recompute_seconds"]),
        "handoff_failure_rate": float(summary["handoff_failures"]) / max(handoffs, 1),
        "deadline_violation_rate": float(summary["deadline_violations"]),
        "service_failures": int(summary["service_failures"]),
        "migration_attempts": int(summary["migration_attempts"]),
        "migration_successes": int(summary["migration_successes"]),
        "cache_evictions": int(summary["cache_evictions"]),
        "reward": float(summary["reward"]),
        "action_0": int(action_counts[0]),
        "action_1": int(action_counts[1]),
        "action_2": int(action_counts[2]),
        "action_3": int(action_counts[3]),
        "action_4": int(action_counts[4]),
    }


def _evaluate_agent(
    agent: Any,
    method: str,
    seed: int,
    config: dict[str, Any],
    instances: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    agent._deterministic_action = True
    rows = []
    for instance in instances:
        env = CalibratedContinuousWorkflowEnv(config, instance)
        observation, info = env.reset()
        counts: Counter[int] = Counter()
        terminated = truncated = False
        while not terminated and not truncated:
            action, _ = agent.act(observation, info)
            counts[int(action)] += 1
            observation, _, terminated, truncated, info = env.step(action)
        rows.append(_evaluation_row(method=method, seed=seed, env=env, action_counts=counts))
    return rows


def _evaluate_rule(
    config: dict[str, Any],
    instances: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rule = TwoStepCostRule()
    rows = []
    for instance in instances:
        env = CalibratedContinuousWorkflowEnv(config, instance)
        env.reset()
        counts: Counter[int] = Counter()
        while not env.terminated and env.step_index < int(instance["max_steps"]):
            action = rule.select_action(env)
            counts[int(action)] += 1
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        rows.append(_evaluation_row(method=rule.method_name, seed=None, env=env, action_counts=counts))
    return rows


def _window_level(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric in PRIMARY_METRICS:
            grouped[(str(row["method"]), str(row["window_id"]))][metric].append(float(row[metric]))
    result: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (method, _window), metrics in grouped.items():
        for metric, values in metrics.items():
            result[method][metric].append(float(np.mean(values)))
    return result


def _bootstrap_ci(values: list[float], rng: np.random.Generator, draws: int = 5000) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(data), size=(draws, len(data)))
    means = data[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    windows = _window_level(rows)
    rng = np.random.default_rng(20261006)
    aggregate: dict[str, Any] = {}
    for method in sorted(windows):
        aggregate[method] = {"outer_unit": "evaluation_window", "outer_n": 0, "metrics": {}}
        for metric in PRIMARY_METRICS:
            values = windows[method][metric]
            low, high = _bootstrap_ci(values, rng)
            aggregate[method]["outer_n"] = len(values)
            aggregate[method]["metrics"][metric] = {
                "mean": float(np.mean(values)),
                "ci95_low": low,
                "ci95_high": high,
            }
    rule = windows["two_step_cost_rule"]
    paired: dict[str, Any] = {}
    for method in sorted(set(windows) - {"two_step_cost_rule"}):
        paired[method] = {}
        for metric in PRIMARY_METRICS:
            delta = np.asarray(windows[method][metric]) - np.asarray(rule[metric])
            low, high = _bootstrap_ci(delta.tolist(), rng)
            paired[method][metric] = {
                "mean_delta_vs_rule": float(delta.mean()),
                "ci95_low": low,
                "ci95_high": high,
            }
    return {"methods": aggregate, "paired_vs_two_step_rule": paired, "bootstrap_draws": 5000}


def _format_cell(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f} [{metric['ci95_low']:.3f}, {metric['ci95_high']:.3f}]"


def _write_paper_table(path: Path, aggregate: dict[str, Any]) -> None:
    labels = {
        "sa_ghmappo": "SA-GHMAPPO",
        "ppo": "PPO",
        "mappo": "Controller-level MAPPO",
        "two_step_cost_rule": "Two-step cost rule",
    }
    columns = list(PRIMARY_METRICS)
    lines = [
        "| Method | " + " | ".join(columns) + " |",
        "|---|" + "|".join("---:" for _ in columns) + "|",
    ]
    for method in ("sa_ghmappo", "ppo", "mappo", "two_step_cost_rule"):
        metrics = aggregate["methods"][method]["metrics"]
        lines.append(
            f"| {labels[method]} | " + " | ".join(_format_cell(metrics[column]) for column in columns) + " |"
        )
    lines.extend(
        [
            "",
            "Values are evaluation-window means with percentile 95% bootstrap CIs (12 outer windows). ",
            "Learned methods are averaged over seeds within each window before inference; the deterministic rule has one run per window. ",
            "This is a calibrated non-formal pilot, not formal/holdout evidence or a real-RSU deployment result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/calibrated_continuous_workflow_pilot_v1.json")
    parser.add_argument("--manifest", default="configs/experiment/calibrated_continuous_workflow_pilot_v1_manifest.json")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--agents", nargs="*", choices=["sa_ghmappo", "ppo", "mappo"])
    args = parser.parse_args()

    config_path = ROOT_DIR / args.config
    manifest_path = ROOT_DIR / args.manifest
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output already exists: {output_root}")
    output_root.mkdir(parents=True)
    (output_root / "checkpoints").mkdir()

    config = _load_json(config_path)
    manifest = _load_json(manifest_path)
    if manifest["config_sha256"] != _sha256(config_path):
        raise RuntimeError("frozen manifest config hash mismatch")
    training = config["training"]
    episodes = int(args.episodes or training["episodes"])
    seeds = list(args.seeds or training["seeds"])
    agents = list(args.agents or training["agents"])
    train_instances = [row for row in manifest["instances"] if row["split"] == "train"]
    evaluation_instances = [row for row in manifest["instances"] if row["split"] == "evaluation"]
    evaluation_rows: list[dict[str, Any]] = []
    training_logs = []

    for method in agents:
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            agent = build_agent(
                method,
                random_seed=seed,
                learning_rate=float(training["learning_rate"]),
                clip_ratio=float(training["clip_ratio"]),
                entropy_coef=float(training["entropy_coef"]),
                value_coef=float(training["value_coef"]),
                batch_size=32,
                train_epochs=4,
                deterministic_action=False,
            )
            rng = random.Random(seed)
            pending: list[dict[str, Any]] = []
            summaries = []
            updates = []
            order = list(range(len(train_instances)))
            for episode in range(episodes):
                if episode % len(order) == 0:
                    rng.shuffle(order)
                instance = train_instances[order[episode % len(order)]]
                rollout, summary = _collect_training_episode(agent, config, instance)
                pending.extend(rollout)
                summaries.append(summary)
                if (episode + 1) % int(training["update_every_episodes"]) == 0:
                    updates.append(_json_safe(agent.learn(pending)))
                    pending = []
            if pending:
                updates.append(_json_safe(agent.learn(pending)))
            checkpoint = output_root / "checkpoints" / f"{method}_seed{seed}.pt"
            agent.save(str(checkpoint))
            evaluation_rows.extend(_evaluate_agent(agent, method, seed, config, evaluation_instances))
            training_logs.append(
                {
                    "method": method,
                    "seed": seed,
                    "episodes": episodes,
                    "updates": len(updates),
                    "training_workflow_completion_rate": float(
                        np.mean([item["workflow_completed"] for item in summaries])
                    ),
                    "last_update": updates[-1] if updates else None,
                    "checkpoint": str(checkpoint.relative_to(output_root)),
                    "checkpoint_sha256": _sha256(checkpoint),
                }
            )
            print(json.dumps({"completed": method, "seed": seed, "updates": len(updates)}), flush=True)

    evaluation_rows.extend(_evaluate_rule(config, evaluation_instances))
    aggregate = _aggregate(evaluation_rows)
    _write_csv(output_root / "evaluation_rows.csv", evaluation_rows)
    _write_json(output_root / "evaluation_rows.json", evaluation_rows)
    _write_json(output_root / "aggregate.json", aggregate)
    _write_json(output_root / "training_summary.json", training_logs)
    _write_paper_table(output_root / "paper_table.md", aggregate)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, check=True, capture_output=True, text=True
    ).stdout.strip()
    run_manifest = {
        "schema_version": "calibrated_continuous_workflow_pilot_run_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "config": {"path": args.config, "sha256": _sha256(config_path)},
        "workload_manifest": {"path": args.manifest, "sha256": _sha256(manifest_path)},
        "episodes_per_method_seed": episodes,
        "agents": agents,
        "seeds": seeds,
        "rule": "two_step_cost_rule",
        "evaluation_outer_unit": "evaluation_window",
        "evaluation_outer_n": len(evaluation_instances),
        "claim_boundary": config["claim_boundary"],
        "command": " ".join(sys.argv),
    }
    _write_json(output_root / "run_manifest.json", run_manifest)
    _write_json(
        output_root / "completion_receipt.json",
        {
            "status": "complete",
            "methods": agents + ["two_step_cost_rule"],
            "evaluation_rows": len(evaluation_rows),
            "paper_table": "paper_table.md",
        },
    )
    artifact_files = sorted(
        path for path in output_root.rglob("*") if path.is_file() and path.name != "artifact_integrity.json"
    )
    _write_json(
        output_root / "artifact_integrity.json",
        {
            "schema_version": "artifact_integrity_v1",
            "files": [
                {"path": str(path.relative_to(output_root)), "sha256": _sha256(path), "bytes": path.stat().st_size}
                for path in artifact_files
            ],
        },
    )
    print(json.dumps({"output_root": str(output_root), "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
