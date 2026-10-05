"""Execute the frozen, information-matched continuous-workflow pilot v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
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
    ImmediateCostRule,
    TwoStepCostRule,
)
from src.trainers.ppo_buffer import PPORolloutBuffer


PRIMARY_METRICS = (
    "workflow_completion_rate",
    "node_coverage_rate",
    "unfinished_rate",
    "deadline_violation_rate",
    "service_failure_rate",
    "handoff_failure_rate",
    "modeled_completion_seconds",
    "total_transfer_mb",
    "model_prepare_mb",
    "state_transfer_mb",
    "recompute_seconds",
    "decision_overhead_ms",
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def _build(method: str, seed: int, training: dict[str, Any]) -> Any:
    registry_name = "sa_ghmappo" if method == "sa_ghmappo_no_dependency" else method
    kwargs: dict[str, Any] = {
        "random_seed": seed,
        "learning_rate": float(training["learning_rate"]),
        "clip_ratio": float(training["clip_ratio"]),
        "entropy_coef": float(training["entropy_coef"]),
        "value_coef": float(training["value_coef"]),
        "batch_size": 32,
        "train_epochs": 4,
        "deterministic_action": False,
    }
    if method == "sa_ghmappo_no_dependency":
        kwargs["use_dependency_aware"] = False
    return build_agent(registry_name, **kwargs)


def _collect_training_episode(
    agent: Any,
    config: dict[str, Any],
    instance: dict[str, Any],
    step_cap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = CalibratedContinuousWorkflowEnv(config, instance)
    observation, info = env.reset()
    buffer = PPORolloutBuffer()
    terminated = truncated = False
    while not terminated and not truncated and env.step_index < step_cap:
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
    if not terminated and not truncated and env.step_index >= step_cap:
        truncated = True
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
    decision_ns: int,
) -> dict[str, Any]:
    summary = env.summary()
    handoffs = int(summary["handoff_count"])
    nodes = int(summary["node_count"])
    completed = int(summary["completed_nodes"])
    total_bytes = int(summary["total_transfer_bytes"])
    factors = env.instance["factors"]
    return {
        "method": method,
        "seed": "rule" if seed is None else seed,
        "design_id": summary["design_id"],
        "window_id": summary["window_id"],
        "workflow_id": summary["workflow_id"],
        "source_segment_id": env.instance["source_interval"]["source_segment_id"],
        "sharing": factors["sharing"],
        "capacity": factors["capacity"],
        "state_scale": factors["state_scale"],
        "initial_target": factors["initial_target"],
        "link_error_class": env.instance.get("link_profile", {}).get("error_class", "matched"),
        "handoff_pressure": env.instance["trace_features"]["handoff_pressure"],
        "topology_class": env.instance["workflow_features"]["topology_class"],
        "node_count": nodes,
        "completed_nodes": completed,
        "workflow_completion_rate": float(summary["workflow_completed"]),
        "node_coverage_rate": float(completed) / max(nodes, 1),
        "unfinished_rate": float(not bool(summary["workflow_completed"])),
        "truncated_rate": float(not bool(summary["terminated"])),
        "deadline_violation_rate": float(summary["deadline_violations"]),
        "service_failure_rate": float(summary["service_failures"] > 0),
        "service_failures": int(summary["service_failures"]),
        "handoff_failure_rate": float(summary["handoff_failures"]) / max(handoffs, 1),
        "handoff_failures": int(summary["handoff_failures"]),
        "modeled_completion_seconds": float(summary["modeled_completion_seconds"]),
        "total_transfer_mb": float(total_bytes) / 1_000_000.0,
        "model_prepare_mb": float(summary["model_transfer_bytes"]) / 1_000_000.0,
        "state_transfer_mb": float(summary["state_transfer_bytes"]) / 1_000_000.0,
        "input_transfer_mb": float(summary["input_transfer_bytes"]) / 1_000_000.0,
        "model_load_seconds": float(summary["model_load_seconds"]),
        "state_restore_seconds": float(summary["state_restore_seconds"]),
        "recompute_seconds": float(summary["recompute_seconds"]),
        "migration_attempts": int(summary["migration_attempts"]),
        "migration_successes": int(summary["migration_successes"]),
        "cache_evictions": int(summary["cache_evictions"]),
        "decision_overhead_ms": float(decision_ns) / 1_000_000.0,
        "reward": float(summary["reward"]),
        "reward_node_completion": float(summary["reward_node_completion"]),
        "reward_workflow_completion": float(summary["reward_workflow_completion"]),
        "reward_time_penalty": float(summary["reward_time_penalty"]),
        "reward_transfer_penalty": float(summary["reward_transfer_penalty"]),
        "reward_failure_penalty": float(summary["reward_failure_penalty"]),
        "reward_deadline_penalty": float(summary["reward_deadline_penalty"]),
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
    step_cap: int,
) -> list[dict[str, Any]]:
    old_deterministic = bool(getattr(agent, "_deterministic_action", False))
    agent._deterministic_action = True
    rows = []
    for instance in instances:
        env = CalibratedContinuousWorkflowEnv(config, instance)
        observation, info = env.reset()
        counts: Counter[int] = Counter()
        decision_ns = 0
        while not env.terminated and env.step_index < step_cap:
            started = time.perf_counter_ns()
            action, _ = agent.act(observation, info)
            decision_ns += time.perf_counter_ns() - started
            counts[int(action)] += 1
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        rows.append(
            _evaluation_row(
                method=method,
                seed=seed,
                env=env,
                action_counts=counts,
                decision_ns=decision_ns,
            )
        )
    agent._deterministic_action = old_deterministic
    return rows


def _evaluate_rule(
    rule: ImmediateCostRule | TwoStepCostRule,
    config: dict[str, Any],
    instances: list[dict[str, Any]],
    step_cap: int,
) -> list[dict[str, Any]]:
    rows = []
    for instance in instances:
        env = CalibratedContinuousWorkflowEnv(config, instance)
        env.reset()
        counts: Counter[int] = Counter()
        decision_ns = 0
        while not env.terminated and env.step_index < step_cap:
            started = time.perf_counter_ns()
            action = rule.select_action(env)
            decision_ns += time.perf_counter_ns() - started
            counts[int(action)] += 1
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        rows.append(
            _evaluation_row(
                method=rule.method_name,
                seed=None,
                env=env,
                action_counts=counts,
                decision_ns=decision_ns,
            )
        )
    return rows


def _selection_score(rows: list[dict[str, Any]]) -> tuple[float, ...]:
    means = {key: float(np.mean([float(row[key]) for row in rows])) for key in PRIMARY_METRICS}
    return (
        means["workflow_completion_rate"],
        -means["deadline_violation_rate"],
        -means["service_failure_rate"],
        -means["handoff_failure_rate"],
        -means["modeled_completion_seconds"],
        -means["total_transfer_mb"],
        means["reward"],
    )


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


def _stratified(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("sharing", "capacity", "link_error_class", "state_scale", "topology_class", "handoff_pressure")
    result: list[dict[str, Any]] = []
    for field in fields:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row["method"]), str(row[field]))].append(row)
        for (method, value), group in sorted(grouped.items()):
            result.append(
                {
                    "stratum": field,
                    "value": value,
                    "method": method,
                    "row_n": len(group),
                    "independent_window_n": len({str(item["window_id"]) for item in group}),
                    **{
                        metric: float(np.mean([float(item[metric]) for item in group]))
                        for metric in PRIMARY_METRICS
                    },
                }
            )
    return result


def _format_cell(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f} [{metric['ci95_low']:.3f}, {metric['ci95_high']:.3f}]"


def _write_paper_table(path: Path, aggregate: dict[str, Any]) -> None:
    labels = {
        "sa_ghmappo": "SA-GHMAPPO",
        "sa_ghmappo_no_dependency": "SA-GHMAPPO without dependency messages",
        "ppo": "PPO",
        "mappo": "Controller-level MAPPO",
        "immediate_cost_rule": "Immediate cost rule",
        "two_step_cost_rule": "Two-step cost rule",
    }
    columns = (
        "workflow_completion_rate",
        "node_coverage_rate",
        "deadline_violation_rate",
        "modeled_completion_seconds",
        "total_transfer_mb",
        "recompute_seconds",
        "decision_overhead_ms",
    )
    order = (
        "sa_ghmappo",
        "sa_ghmappo_no_dependency",
        "mappo",
        "ppo",
        "immediate_cost_rule",
        "two_step_cost_rule",
    )
    lines = [
        "| Method | " + " | ".join(columns) + " |",
        "|---|" + "|".join("---:" for _ in columns) + "|",
    ]
    for method in order:
        metrics = aggregate["methods"][method]["metrics"]
        lines.append(
            f"| {labels[method]} | " + " | ".join(_format_cell(metrics[column]) for column in columns) + " |"
        )
    lines.extend(
        [
            "",
            "Values are evaluation-window means with percentile 95% bootstrap CIs over 12 frozen windows. ",
            "Seeds are averaged within each window before resampling; repeated seed rows are not independent samples. ",
            "This is a measurement-calibrated synthetic pilot, not formal/holdout evidence or a real-RSU deployment result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_curve_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    colors = {
        "sa_ghmappo": "#1367d1",
        "sa_ghmappo_no_dependency": "#56a0e8",
        "mappo": "#d16900",
        "ppo": "#2b8a3e",
    }
    width, height = 960, 520
    left, right, top, bottom = 70, 30, 45, 70
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["method"])][int(row["episode"])].append(float(row["workflow_completed"]))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="480" y="26" text-anchor="middle" font-family="sans-serif" font-size="18">Training workflow completion: cumulative mean</text>',
    ]
    x0, y0 = left, height - bottom
    x1, y1 = width - right, top
    svg.extend(
        [
            f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333"/>',
        ]
    )
    for tick in range(6):
        value = tick / 5
        y = y0 - value * (y0 - y1)
        svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#ddd"/>')
        svg.append(f'<text x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.1f}</text>')
    max_episode = max(int(row["episode"]) for row in rows)
    for method, episodes in sorted(grouped.items()):
        cumulative: list[float] = []
        points = []
        for episode in sorted(episodes):
            cumulative.extend(episodes[episode])
            value = float(np.mean(cumulative))
            x = x0 + episode / max_episode * (x1 - x0)
            y = y0 - value * (y0 - y1)
            points.append(f"{x:.1f},{y:.1f}")
        svg.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[method]}" stroke-width="2"/>'
        )
    legend_x = 100
    for method in colors:
        svg.append(f'<line x1="{legend_x}" y1="{height - 28}" x2="{legend_x + 24}" y2="{height - 28}" stroke="{colors[method]}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x + 30}" y="{height - 23}" font-family="sans-serif" font-size="12">{method}</text>')
        legend_x += 205
    svg.append(f'<text x="{(x0 + x1) / 2}" y="{height - 45}" text-anchor="middle" font-family="sans-serif" font-size="13">episode</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/calibrated_continuous_workflow_pilot_v2.json")
    parser.add_argument("--manifest", default="configs/experiment/calibrated_continuous_workflow_pilot_v2_manifest.json")
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    config_path = ROOT_DIR / args.config
    manifest_path = ROOT_DIR / args.manifest
    diagnostic_path = Path(args.diagnostic).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output already exists: {output_root}")
    diagnostic = _load_json(diagnostic_path)
    if diagnostic.get("status") != "pass":
        raise RuntimeError("non-degeneracy diagnostic did not pass; training is blocked")
    output_root.mkdir(parents=True)
    (output_root / "checkpoints").mkdir()

    config = _load_json(config_path)
    manifest = _load_json(manifest_path)
    if manifest["config_sha256"] != _sha256(config_path):
        raise RuntimeError("frozen manifest config hash mismatch")
    training = config["training"]
    seeds = [int(seed) for seed in training["seeds"]]
    episodes = int(training["episodes"])
    main_methods = list(training["agents"])
    ablation = str(training["ablation"])
    methods = main_methods + [ablation]
    candidates = {int(value) for value in training["checkpoint_episode_candidates"]}
    budget = training["budget"]
    step_cap = int(budget["episode_max_steps"])
    if episodes > 128 or max(candidates) != episodes:
        raise RuntimeError("frozen episode/checkpoint budget is invalid")
    main_ceiling = len(main_methods) * len(seeds) * episodes * step_cap
    ablation_ceiling = len(seeds) * episodes * step_cap
    total_ceiling = main_ceiling + ablation_ceiling
    if main_ceiling > int(budget["main_training_step_cap"]):
        raise RuntimeError("main training budget exceeds frozen cap")
    if ablation_ceiling > int(budget["ablation_step_cap"]):
        raise RuntimeError("ablation training budget exceeds frozen cap")
    if total_ceiling > int(budget["total_training_step_cap"]):
        raise RuntimeError("total training budget exceeds frozen cap")
    if any(int(row["max_steps"]) > step_cap for row in manifest["instances"]):
        raise RuntimeError("a frozen instance exceeds the per-episode step cap")

    splits = {
        split: [row for row in manifest["instances"] if row["split"] == split]
        for split in ("train", "dev", "evaluation")
    }
    evaluation_rows: list[dict[str, Any]] = []
    training_curves: list[dict[str, Any]] = []
    checkpoint_selection: list[dict[str, Any]] = []
    training_summary: list[dict[str, Any]] = []
    actual_steps = {"main": 0, "ablation": 0}

    for method in methods:
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            agent = _build(method, seed, training)
            rng = random.Random(seed)
            pending: list[dict[str, Any]] = []
            updates: list[dict[str, Any]] = []
            order = list(range(len(splits["train"])))
            candidate_records: list[dict[str, Any]] = []
            for episode_index in range(episodes):
                if episode_index % len(order) == 0:
                    rng.shuffle(order)
                instance = splits["train"][order[episode_index % len(order)]]
                rollout, summary = _collect_training_episode(agent, config, instance, step_cap)
                pending.extend(rollout)
                bucket = "ablation" if method == ablation else "main"
                actual_steps[bucket] += len(rollout)
                training_curves.append(
                    {
                        "method": method,
                        "seed": seed,
                        "episode": episode_index + 1,
                        "design_id": summary["design_id"],
                        "steps": summary["steps"],
                        "workflow_completed": summary["workflow_completed"],
                        "node_coverage_rate": float(summary["completed_nodes"]) / max(int(summary["node_count"]), 1),
                        "deadline_violation": summary["deadline_violations"],
                        "modeled_completion_seconds": summary["modeled_completion_seconds"],
                        "total_transfer_mb": float(summary["total_transfer_bytes"]) / 1_000_000.0,
                        "reward": summary["reward"],
                    }
                )
                if (episode_index + 1) % int(training["update_every_episodes"]) == 0:
                    updates.append(_json_safe(agent.learn(pending)))
                    pending = []
                if episode_index + 1 in candidates:
                    checkpoint = output_root / "checkpoints" / f"{method}_seed{seed}_episode{episode_index + 1}.pt"
                    agent.save(str(checkpoint))
                    dev_rows = _evaluate_agent(agent, method, seed, config, splits["dev"], step_cap)
                    score = _selection_score(dev_rows)
                    candidate_records.append(
                        {
                            "method": method,
                            "seed": seed,
                            "episode": episode_index + 1,
                            "score": list(score),
                            "checkpoint": str(checkpoint.relative_to(output_root)),
                            "checkpoint_sha256": _sha256(checkpoint),
                            "dev_rows": dev_rows,
                        }
                    )
            if pending:
                updates.append(_json_safe(agent.learn(pending)))
            best = max(candidate_records, key=lambda row: (tuple(row["score"]), -int(row["episode"])))
            best_path = output_root / best["checkpoint"]
            agent.load(str(best_path))
            selected = output_root / "checkpoints" / f"{method}_seed{seed}_selected.pt"
            agent.save(str(selected))
            evaluation_rows.extend(_evaluate_agent(agent, method, seed, config, splits["evaluation"], step_cap))
            checkpoint_selection.extend(candidate_records)
            training_summary.append(
                {
                    "method": method,
                    "seed": seed,
                    "episodes": episodes,
                    "updates": len(updates),
                    "selected_episode": best["episode"],
                    "selected_checkpoint": str(selected.relative_to(output_root)),
                    "selected_checkpoint_sha256": _sha256(selected),
                    "last_update": updates[-1] if updates else None,
                }
            )
            print(json.dumps({"completed": method, "seed": seed, "selected_episode": best["episode"]}), flush=True)

    if actual_steps["main"] > int(budget["main_training_step_cap"]):
        raise RuntimeError("observed main training steps exceeded cap")
    if actual_steps["ablation"] > int(budget["ablation_step_cap"]):
        raise RuntimeError("observed ablation training steps exceeded cap")
    if sum(actual_steps.values()) > int(budget["total_training_step_cap"]):
        raise RuntimeError("observed total training steps exceeded cap")

    rules: list[ImmediateCostRule | TwoStepCostRule] = [ImmediateCostRule(), TwoStepCostRule()]
    for rule in rules:
        evaluation_rows.extend(_evaluate_rule(rule, config, splits["evaluation"], step_cap))

    aggregate = _aggregate(evaluation_rows)
    strata = _stratified(evaluation_rows)
    _write_csv(output_root / "evaluation_rows.csv", evaluation_rows)
    _write_json(output_root / "evaluation_rows.json", evaluation_rows)
    _write_json(output_root / "aggregate.json", aggregate)
    _write_csv(output_root / "stratified_results.csv", strata)
    _write_json(output_root / "stratified_results.json", strata)
    _write_csv(output_root / "training_curves.csv", training_curves)
    _write_json(output_root / "training_curves.json", training_curves)
    _write_curve_svg(output_root / "training_completion_curve.svg", training_curves)
    _write_json(output_root / "training_summary.json", training_summary)
    _write_json(output_root / "checkpoint_selection.json", checkpoint_selection)
    _write_paper_table(output_root / "paper_table.md", aggregate)

    run_manifest = {
        "schema_version": "calibrated_continuous_workflow_pilot_run_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": {"path": args.config, "sha256": _sha256(config_path)},
        "workload_manifest": {"path": args.manifest, "sha256": _sha256(manifest_path)},
        "non_degeneracy_diagnostic": {"path": str(diagnostic_path), "sha256": _sha256(diagnostic_path)},
        "episodes_per_method_seed": episodes,
        "methods": methods,
        "rules": [rule.method_name for rule in rules],
        "seeds": seeds,
        "checkpoint_selection": training["checkpoint_selection"],
        "budget_ceiling_steps": {"main": main_ceiling, "ablation": ablation_ceiling, "total": total_ceiling},
        "actual_training_steps": {**actual_steps, "total": sum(actual_steps.values())},
        "evaluation_outer_unit": "evaluation_window",
        "evaluation_outer_n": len(splits["evaluation"]),
        "claim_boundary": config["claim_boundary"],
        "real_model_generate_calls": 0,
        "downloads": 0,
        "old_holdout_calls": 0,
        "command": " ".join(sys.argv),
    }
    _write_json(output_root / "run_manifest.json", run_manifest)
    _write_json(
        output_root / "completion_receipt.json",
        {
            "status": "complete",
            "methods": methods + [rule.method_name for rule in rules],
            "evaluation_rows": len(evaluation_rows),
            "actual_training_steps": run_manifest["actual_training_steps"],
            "paper_table": "paper_table.md",
            "training_curve": "training_completion_curve.svg",
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
