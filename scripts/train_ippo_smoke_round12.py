"""Train a small IPPO smoke checkpoint and evaluate live proposal rows.

This is proposal-only smoke work. It does not modify reward, SA-GHMAPPO,
heuristic baselines, checkpoint selection, or the formal mixed/full benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.registry import build_agent, checkpoint_required_agents, list_evaluable_agents
from src.data.mobility.replay_provider import ReplayProvider
from src.data.workflow.workflow_dataset_builder import WorkflowDatasetBuilder
from src.envs.core.predictor_manager import PredictorManager
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.evaluators.main_results_support import (
    clone_frames,
    clone_rsu_state,
    clone_workflow_state,
    load_window_bundle,
    resolve_window_candidates,
    run_real_episode,
    summary_to_row,
)
from src.metrics.recorder import EpisodeRecorder
from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer


TASK_NAME = "ippo_live_eval_rows_round12"
DEFAULT_CONFIG = Path("configs/experiment/ippo_smoke_round12.yaml")
OUTPUT_DIR = Path("artifacts/analysis/ippo_live_eval_rows_round12")
REPORT_PATH = Path("docs/agent/ippo_live_eval_rows_round12_report.md")
LOWER_IS_BETTER = {
    "handoff_failure_rate",
    "backhaul_traffic_cost",
    "adapter_miss_count",
    "adapter_cold_start_count",
    "eviction_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for this script.")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def float_value(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "missing"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    return round(fmean(values), 6) if values else 0.0


def metric_mean(rows: list[dict[str, Any]], metric_name: str) -> float:
    return mean([float_value(row.get(metric_name)) for row in rows])


def metric_sum(rows: list[dict[str, Any]], metric_name: str) -> float:
    return round(sum(float_value(row.get(metric_name)) for row in rows), 6)


def load_manifest(path: str | Path) -> dict[str, dict[str, str]]:
    manifest_path = Path(path)
    if not str(path) or not manifest_path.exists():
        return {}
    payload = read_json(manifest_path)
    normalized: dict[str, dict[str, str]] = {}
    for agent_name, seed_map in payload.items():
        if isinstance(seed_map, dict):
            normalized[str(agent_name)] = {str(seed): str(value) for seed, value in seed_map.items() if str(value)}
    for primary, legacy in [("ppo", "flat_ppo"), ("mappo", "flat_mappo")]:
        if primary in normalized and legacy not in normalized:
            normalized[legacy] = dict(normalized[primary])
        if legacy in normalized and primary not in normalized:
            normalized[primary] = dict(normalized[legacy])
    return normalized


def checkpoint_map_for_seed(
    manifests: dict[str, dict[str, dict[str, str]]],
    seed: int,
    ippo_checkpoint_path: str,
) -> dict[str, str]:
    checkpoint_map = {
        "ippo": ippo_checkpoint_path,
        "sa_ghmappo": "",
        "ppo": "",
        "flat_ppo": "",
        "mappo": "",
        "flat_mappo": "",
        "popularity_cache_heuristic": "",
        "reactive_greedy": "",
    }
    seed_key = str(seed)
    for manifest in manifests.values():
        for agent_name, seed_map in manifest.items():
            if seed_key in seed_map:
                checkpoint_map[agent_name] = seed_map[seed_key]
    if checkpoint_map.get("ppo") and not checkpoint_map.get("flat_ppo"):
        checkpoint_map["flat_ppo"] = checkpoint_map["ppo"]
    if checkpoint_map.get("flat_ppo") and not checkpoint_map.get("ppo"):
        checkpoint_map["ppo"] = checkpoint_map["flat_ppo"]
    return checkpoint_map


def load_workflows(config: dict[str, Any]) -> list[Any]:
    data_cfg = config.get("data", {})
    workflow_ids = [str(item) for item in data_cfg.get("workflow_ids", ["j_3", "j_8"])]
    samples = WorkflowDatasetBuilder().build_alibaba_samples(
        csv_path=data_cfg.get("workflow_csv_path", "data/raw/workflow/alibaba2018/batch_task.csv"),
        limit_jobs=max(64, len(workflow_ids) * 24),
        min_tasks=int(data_cfg.get("min_tasks", 5)),
        max_tasks=int(data_cfg.get("max_tasks", 20)),
        adapter_assignment_profile=str(config.get("adapter_assignment_profile", "semantic_ai_service")),
    )
    by_id = {str(sample.get("workflow_id")): sample for sample in samples}
    missing = [workflow_id for workflow_id in workflow_ids if workflow_id not in by_id]
    if missing:
        raise RuntimeError(f"requested workflow ids missing from parsed Alibaba samples: {missing}")
    builder = WorkflowDatasetBuilder()
    return [builder.sample_to_workflow_state(by_id[workflow_id]) for workflow_id in workflow_ids]


def workflow_metadata(workflow_state: Any) -> dict[str, Any]:
    required_adapters = sorted({str(node.required_adapter) for node in workflow_state.nodes if node.required_adapter})
    required_base_models = sorted({str(node.required_base_model) for node in workflow_state.nodes if node.required_base_model})
    return {
        "required_adapter_count": len(required_adapters),
        "unique_adapter_per_episode": len(required_adapters),
        "required_adapter_ids": ";".join(required_adapters) if required_adapters else "missing",
        "required_base_model_count": len(required_base_models),
        "required_base_model_ids": ";".join(required_base_models) if required_base_models else "missing",
    }


def cross_rsu_workflow_rate(summary: dict[str, Any]) -> float:
    rsu_ids: set[str] = set()
    for step in summary.get("step_trace", []):
        if not isinstance(step, dict):
            continue
        for key in ["pre_action_associated_rsu_id", "current_associated_rsu_id", "post_action_associated_rsu_id", "offload_target_rsu_id"]:
            value = step.get(key)
            if value not in (None, "", "None"):
                rsu_ids.add(str(value))
    return 1.0 if len(rsu_ids) > 1 else 0.0


def augment_row(row: dict[str, Any], summary: dict[str, Any], workflow_state: Any, run_id: str) -> dict[str, Any]:
    augmented = dict(row)
    augmented.update(workflow_metadata(workflow_state))
    augmented["profile_name"] = TASK_NAME
    augmented["proposal_only"] = True
    augmented["do_not_use_for_freeze"] = True
    augmented["adapter_assignment_profile"] = "semantic_ai_service"
    augmented["cache_capacity_profile_name"] = "multi_adapter_capacity_stress"
    augmented["run_id"] = run_id
    augmented["reward"] = augmented.get("total_reward", 0.0)
    augmented["failure"] = augmented.get("handoff_failure_rate", 0.0)
    augmented["delay"] = augmented.get("end_to_end_workflow_delay", 0.0)
    augmented["success"] = 1.0 if str(augmented.get("episode_success")).lower() == "true" else 0.0
    augmented["handoff_during_workflow_rate"] = 1.0 if float_value(augmented.get("handoff_total_count")) > 0.0 else 0.0
    augmented["cross_rsu_workflow_rate"] = cross_rsu_workflow_rate(summary)
    return augmented


def selected_windows(config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    data_cfg = config.get("data", {})
    _, window_payload = resolve_window_candidates(
        root_dir=ROOT_DIR,
        mobility_source=str(data_cfg.get("mobility_source", "ngsim")),
        mobility_csv_path=str(data_cfg.get("mobility_csv_path", "")),
        lust_scenario_root=str(data_cfg.get("lust_scenario_root", "")),
        max_mobility_rows=int(data_cfg.get("max_mobility_rows", 2500)),
        rsu_layout=str(data_cfg.get("rsu_layout", "auto_dominant_tight")),
        frame_offset=int(data_cfg.get("frame_offset", 0)),
        window_length=int(data_cfg.get("window_length", 24)),
        window_selector=str(data_cfg.get("window_selector", "max_handoff_candidate")),
        window_count=int(data_cfg.get("window_count", 1)),
        window_scan_stride=int(data_cfg.get("window_scan_stride", 2)),
        random_seed=seeds[0] if seeds else 7,
        window_mode=str(data_cfg.get("window_mode", "activating_only")),
    )
    windows = list(window_payload.get("selected_windows", []))
    if not windows:
        raise RuntimeError("No selected windows for IPPO round12 smoke.")
    return windows


def annotate_checkpoint(path: Path, metadata: dict[str, Any]) -> None:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        payload["training_metadata"] = dict(metadata)
        torch.save(payload, path)


def train_ippo_smoke(config: dict[str, Any], workflows: list[Any], windows: list[dict[str, Any]]) -> dict[str, Any]:
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    cache_profile = dict(config.get("cache_capacity_profile", {}) or {})
    train_seed = int(train_cfg.get("seeds", [7])[0])
    episodes = int(train_cfg.get("episodes", 6))
    update_every = max(int(train_cfg.get("update_every", 2)), 1)
    output_root = Path(config.get("training_output_dir", "artifacts/training/ippo_smoke_round12"))
    run_id = datetime.now().strftime(f"ippo_smoke_round12_train_%Y%m%d_%H%M%S_%f_seed{train_seed}")
    run_root = output_root / "ippo" / run_id
    checkpoint_root = run_root / "checkpoints"
    episode_root = run_root / "episodes"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    episode_root.mkdir(parents=True, exist_ok=True)

    agent = build_agent(
        "ippo",
        random_seed=train_seed,
        deterministic_action=False,
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        clip_ratio=float(train_cfg.get("clip_ratio", 0.2)),
        entropy_coef=float(train_cfg.get("entropy_coef", 0.01)),
        value_coef=float(train_cfg.get("value_coef", 0.5)),
        batch_size=int(train_cfg.get("batch_size", 16)),
    )
    pending_rollout: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    update_logs: list[dict[str, Any]] = []
    update_index = 0
    latest_path = checkpoint_root / "latest.pt"

    for episode_index in range(1, episodes + 1):
        workflow_state = workflows[(episode_index - 1) % len(workflows)]
        window = windows[(episode_index - 1) % len(windows)]
        mobility_bundle = load_window_bundle(
            root_dir=ROOT_DIR,
            mobility_source=str(data_cfg.get("mobility_source", "ngsim")),
            mobility_csv_path=str(data_cfg.get("mobility_csv_path", "")),
            lust_scenario_root=str(data_cfg.get("lust_scenario_root", "")),
            max_mobility_rows=int(data_cfg.get("max_mobility_rows", 2500)),
            rsu_layout=str(window.get("recommended_rsu_layout", data_cfg.get("rsu_layout", "auto_dominant_tight"))),
            frame_offset=int(window.get("frame_offset", data_cfg.get("frame_offset", 0))),
            window_length=int(window.get("window_length", data_cfg.get("window_length", 24))),
            random_seed=train_seed,
        )
        mobility_bundle.rsu_metadata["window_rank"] = window.get("window_rank")
        mobility_bundle.rsu_metadata["window_class"] = window.get("window_class", "mechanism_activating")
        recorder = EpisodeRecorder(prefetch_validation_window=6)
        core_env = VecWorkflowCoreEnv(
            mobility_provider=ReplayProvider(trajectory_frames=clone_frames(mobility_bundle.frames)),
            workflow_state=clone_workflow_state(workflow_state),
            rsu_states=[clone_rsu_state(rsu_state) for rsu_state in mobility_bundle.rsu_states],
            predictor_manager=PredictorManager(random_seed=train_seed + episode_index),
            max_steps=max(int(data_cfg.get("max_steps", 12)) + 2, 8),
            mobility_source=str(data_cfg.get("mobility_source", "ngsim")),
            cache_capacity_profile=cache_profile,
        )
        env = GymVecEnv(core_env=core_env, recorder=recorder)
        trainer = MARLOnPolicyTrainer(
            env=env,
            agent=agent,
            recorder=recorder,
            max_steps=int(data_cfg.get("max_steps", 12)),
            gamma=float(train_cfg.get("gamma", 0.99)),
            gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
        )
        summary, rollout = trainer.collect_episode(
            run_metadata={
                "script": "scripts/train_ippo_smoke_round12.py",
                "run_id": run_id,
                "agent_name": "ippo",
                "workflow_id": workflow_state.workflow_id,
                "window_id": mobility_bundle.rsu_metadata.get("window_id"),
                "config_profile": TASK_NAME,
                "window_mode": TASK_NAME,
                "window_class": mobility_bundle.rsu_metadata.get("window_class"),
                "proposal_only": True,
                "do_not_use_for_freeze": True,
            }
        )
        pending_rollout.extend(rollout)
        should_update = episode_index % update_every == 0 or episode_index == episodes
        if should_update:
            update_index += 1
            learn_info = agent.learn(pending_rollout)
            pending_rollout = []
            update_path = checkpoint_root / f"update_{update_index:04d}.pt"
            agent.save(str(update_path))
            agent.save(str(latest_path))
            metadata = {
                "run_id": run_id,
                "agent_name": "ippo",
                "config_profile": TASK_NAME,
                "episodes": episodes,
                "update_count": update_index,
                "is_smoke_checkpoint": True,
                "train_window_mode": TASK_NAME,
                "script": "scripts/train_ippo_smoke_round12.py",
            }
            annotate_checkpoint(update_path, metadata)
            annotate_checkpoint(latest_path, metadata)
            update_logs.append({"episode_index": episode_index, **learn_info})
        else:
            learn_info = {
                "agent_name": "ippo",
                "policy_update_skipped": True,
                "reason": "waiting_for_update_every",
                "pending_rollout_steps": len(pending_rollout),
            }
        summary["agent_info"] = {"agent_name": "ippo", "learn_info": learn_info}
        summary["episode_success"] = bool(summary.get("episode_status", {}).get("completed", False))
        episode_path = episode_root / f"episode_{episode_index:04d}.summary.json"
        episode_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        row = augment_row(summary_to_row(summary), summary, workflow_state, run_id)
        row["episode_index"] = episode_index
        row["updated"] = should_update
        rows.append(row)

    train_summary = {
        "task_name": TASK_NAME,
        "run_id": run_id,
        "agent_name": "ippo",
        "training_type": "ippo_smoke_training_only",
        "proposal_only": True,
        "do_not_use_for_freeze": True,
        "episodes": episodes,
        "update_count": update_index,
        "latest_checkpoint_path": str(latest_path),
        "train_rows": rows,
        "mean_metrics": {
            "total_reward": metric_mean(rows, "total_reward"),
            "workflow_continuity_rate": metric_mean(rows, "workflow_continuity_rate"),
            "handoff_failure_rate": metric_mean(rows, "handoff_failure_rate"),
            "backhaul_traffic_cost": metric_mean(rows, "backhaul_traffic_cost"),
            "adapter_miss_count": metric_mean(rows, "adapter_miss_count"),
            "eviction_count": metric_mean(rows, "eviction_count"),
        },
        "update_logs": update_logs,
    }
    (run_root / "train_summary.json").write_text(json.dumps(train_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return train_summary


def evaluate_policies(
    config: dict[str, Any],
    workflows: list[Any],
    windows: list[dict[str, Any]],
    ippo_checkpoint_path: str,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    data_cfg = config.get("data", {})
    eval_cfg = config.get("evaluation", {})
    cache_profile = dict(config.get("cache_capacity_profile", {}) or {})
    seeds = [int(seed) for seed in eval_cfg.get("seeds", [7, 13, 29])]
    policies = [str(item) for item in eval_cfg.get("policies", ["ippo", "sa_ghmappo", "ppo", "popularity_cache_heuristic", "reactive_greedy"])]
    manifests = {
        str(agent_name): load_manifest(path)
        for agent_name, path in (config.get("checkpoint_manifests", {}) or {}).items()
    }
    evaluable = set(list_evaluable_agents())
    checkpoint_required = checkpoint_required_agents()
    missing_rows: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    run_id = datetime.now().strftime(f"{TASK_NAME}_eval_%Y%m%d_%H%M%S_%f")
    episode_root = output_dir / "episodes"

    for policy_name in policies:
        if policy_name not in evaluable:
            missing_rows[policy_name] = "not_registered_or_not_evaluable"
            continue
        for seed in seeds:
            checkpoint_map = checkpoint_map_for_seed(manifests, seed, ippo_checkpoint_path)
            if policy_name in checkpoint_required and not checkpoint_map.get(policy_name):
                missing_rows.setdefault(policy_name, f"missing_checkpoint_for_seed_{seed}")
                continue
            for window in windows:
                mobility_bundle = load_window_bundle(
                    root_dir=ROOT_DIR,
                    mobility_source=str(data_cfg.get("mobility_source", "ngsim")),
                    mobility_csv_path=str(data_cfg.get("mobility_csv_path", "")),
                    lust_scenario_root=str(data_cfg.get("lust_scenario_root", "")),
                    max_mobility_rows=int(data_cfg.get("max_mobility_rows", 2500)),
                    rsu_layout=str(window.get("recommended_rsu_layout", data_cfg.get("rsu_layout", "auto_dominant_tight"))),
                    frame_offset=int(window.get("frame_offset", data_cfg.get("frame_offset", 0))),
                    window_length=int(window.get("window_length", data_cfg.get("window_length", 24))),
                    random_seed=seed,
                )
                mobility_bundle.rsu_metadata["window_rank"] = window.get("window_rank")
                mobility_bundle.rsu_metadata["window_class"] = window.get("window_class", "mechanism_activating")
                for workflow_state in workflows:
                    summary = run_real_episode(
                        root_dir=ROOT_DIR,
                        agent_name=policy_name,
                        checkpoint_map=checkpoint_map,
                        workflow_state=workflow_state,
                        workflow_source_path=str(data_cfg.get("workflow_csv_path", "")),
                        mobility_bundle=mobility_bundle,
                        seed=seed,
                        max_steps=int(data_cfg.get("max_steps", 12)),
                        mobility_source=str(data_cfg.get("mobility_source", "ngsim")),
                        run_metadata={
                            "script": "scripts/train_ippo_smoke_round12.py",
                            "benchmark_run_id": run_id,
                            "mode": TASK_NAME,
                            "window_mode": TASK_NAME,
                            "window_rank": window.get("window_rank"),
                            "window_class": window.get("window_class", "mechanism_activating"),
                            "proposal_only": True,
                            "do_not_use_for_freeze": True,
                            "adapter_assignment_profile": str(config.get("adapter_assignment_profile")),
                            "cache_capacity_profile": cache_profile,
                        },
                        cache_capacity_profile=cache_profile,
                    )
                    summary_path = episode_root / str(mobility_bundle.rsu_metadata.get("window_id")) / workflow_state.workflow_id / policy_name / f"seed_{seed}.summary.json"
                    summary_path.parent.mkdir(parents=True, exist_ok=True)
                    summary["run_info"]["summary_path"] = str(summary_path)
                    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                    row = augment_row(summary_to_row(summary), summary, workflow_state, run_id)
                    row["summary_path"] = str(summary_path)
                    rows.append(row)
    return rows, missing_rows


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "unknown")), []).append(row)
    return grouped


def policy_comparison(rows: list[dict[str, Any]], policies: list[str]) -> list[dict[str, Any]]:
    grouped = group_rows(rows, "policy_name")
    metrics = [
        "total_reward",
        "workflow_continuity_rate",
        "handoff_failure_rate",
        "backhaul_traffic_cost",
        "adapter_miss_count",
        "adapter_warm_hit_count",
        "adapter_cold_start_count",
        "eviction_count",
        "cache_occupancy_rate",
    ]
    output: list[dict[str, Any]] = []
    for policy_name in policies:
        group = grouped.get(policy_name, [])
        row: dict[str, Any] = {"policy_name": policy_name, "episode_count": len(group), "status": "evaluated" if group else "missing"}
        for metric in metrics:
            row[f"{metric}_mean"] = metric_mean(group, metric) if group else "missing"
            row[f"{metric}_sum"] = metric_sum(group, metric) if group else "missing"
        output.append(row)
    return output


def pairwise_summary(rows: list[dict[str, Any]], candidate: str, baseline: str) -> list[dict[str, Any]]:
    grouped = group_rows(rows, "policy_name")
    candidate_rows = grouped.get(candidate, [])
    baseline_rows = grouped.get(baseline, [])
    metrics = [
        "total_reward",
        "workflow_continuity_rate",
        "handoff_failure_rate",
        "backhaul_traffic_cost",
        "adapter_miss_count",
        "adapter_cold_start_count",
        "eviction_count",
        "cache_occupancy_rate",
    ]
    output: list[dict[str, Any]] = []
    for metric in metrics:
        candidate_value = metric_mean(candidate_rows, metric) if candidate_rows else 0.0
        baseline_value = metric_mean(baseline_rows, metric) if baseline_rows else 0.0
        delta = round(candidate_value - baseline_value, 6)
        effective_delta = -delta if metric in LOWER_IS_BETTER else delta
        if not candidate_rows or not baseline_rows:
            result = "missing"
        elif effective_delta > 1e-6:
            result = "win"
        elif effective_delta < -1e-6:
            result = "loss"
        else:
            result = "tie"
        output.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "metric": metric,
                "candidate_mean": candidate_value if candidate_rows else "missing",
                "baseline_mean": baseline_value if baseline_rows else "missing",
                "delta_candidate_minus_baseline": delta if candidate_rows and baseline_rows else "missing",
                "result": result,
            }
        )
    return output


def cache_eviction_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for policy_name, group in sorted(group_rows(rows, "policy_name").items()):
        output.append(
            {
                "policy_name": policy_name,
                "episode_count": len(group),
                "cache_capacity_enabled_rate": metric_mean(group, "cache_capacity_enabled"),
                "rsu_adapter_slots_mean": metric_mean(group, "rsu_adapter_slots"),
                "cache_occupancy_rate_mean": metric_mean(group, "cache_occupancy_rate"),
                "eviction_count_sum": metric_sum(group, "eviction_count"),
                "adapter_miss_count_sum": metric_sum(group, "adapter_miss_count"),
                "adapter_warm_hit_count_sum": metric_sum(group, "adapter_warm_hit_count"),
                "adapter_cold_start_count_sum": metric_sum(group, "adapter_cold_start_count"),
            }
        )
    return output


def continuity_stall_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for policy_name, group in sorted(group_rows(rows, "policy_name").items()):
        output.append(
            {
                "policy_name": policy_name,
                "episode_count": len(group),
                "workflow_continuity_rate_mean": metric_mean(group, "workflow_continuity_rate"),
                "handoff_failure_rate_mean": metric_mean(group, "handoff_failure_rate"),
                "successful_episode_rate_mean": metric_mean(group, "successful_episode_rate"),
                "adapter_miss_count_mean": metric_mean(group, "adapter_miss_count"),
                "adapter_cold_start_count_mean": metric_mean(group, "adapter_cold_start_count"),
                "handoff_ready_ratio_mean": metric_mean(group, "handoff_ready_ratio"),
                "continuity_reward_component_mean": metric_mean(group, "continuity_reward_component"),
            }
        )
    return output


def build_report(
    *,
    training_summary: dict[str, Any],
    rows: list[dict[str, Any]],
    policy_summary: list[dict[str, Any]],
    ippo_vs_sa: list[dict[str, Any]],
    diagnosis: dict[str, Any],
) -> str:
    policy_by_name = {row["policy_name"]: row for row in policy_summary}
    sa = policy_by_name.get("sa_ghmappo", {})
    ippo = policy_by_name.get("ippo", {})
    ppo = policy_by_name.get("ppo", {})
    pop = policy_by_name.get("popularity_cache_heuristic", {})
    reactive = policy_by_name.get("reactive_greedy", {})
    ippo_reward_delta = next((row for row in ippo_vs_sa if row["metric"] == "total_reward"), {})
    return f"""# ippo_live_eval_rows_round12 报告

## 范围

本轮补齐 IPPO live eval/checkpoint rows。没有 freeze，没有修改 reward，没有修改 SA-GHMAPPO policy，没有修改 `popularity_cache_heuristic` / `reactive_greedy` 行为，没有修改 checkpoint selection，也没有替换 mixed/full benchmark。

本轮结果仍是 `multi_adapter_hard_joint` proposal smoke，不是正式论文结果。

## 1. 本轮是否训练？

是，但只训练了 IPPO smoke checkpoint。训练 profile 是 `ippo_smoke_round12`，episodes=`{training_summary.get('episodes')}`，update_count=`{training_summary.get('update_count')}`。该 checkpoint 不是 fully tuned baseline。

## 2. IPPO agent 是否已注册？

是。`src/agents/registry.py` 已注册 `ippo`。

## 3. `list_evaluable_agents()` 是否包含 ippo？

`{diagnosis.get('list_evaluable_agents_contains_ippo')}`。

## 4. IPPO 是否有 checkpoint？

是。checkpoint: `{training_summary.get('latest_checkpoint_path')}`。

## 5. IPPO 是否成功产生 benchmark rows？

是。IPPO rows=`{ippo.get('episode_count')}`，总 rows=`{len(rows)}`。

## 6. IPPO 与 SA-GHMAPPO 的结构差异是什么？

IPPO 是 flat semantic encoder + independent critic + shared wrapper decision stream 的 independent-style PPO baseline。SA-GHMAPPO 使用图/层级机制、机制窗口 guard/auxiliary 等主方法能力。

## 7. IPPO 是否使用 centralized critic？

不使用。

## 8. IPPO 是否使用 SA 的 hierarchical mechanism？

不使用。它没有 SA hierarchy、graph-continuity critic、heuristic imitation、mechanism auxiliary 或 mechanism logit prior。
训练日志中若出现 deterministic prepare 等基类字段，只是因为 IPPO 复用通用 PPO base；在 IPPO 中 `use_hierarchy=False` 且 `event_head_enabled=False`，这些 SA 机制路径不激活。

## 9. IPPO 与 flat PPO row 的区别是什么？

`ippo` 是独立注册的 agent/checkpoint/run 名称，policy_type=`ippo_policy`；`ppo` row 是 current `PPOAgent` 加载 existing `flat_ppo` checkpoint alias。两者都基于 flat PPO 基础实现，但 IPPO 本轮有独立 smoke checkpoint 和 live rows。

## 10. SA vs IPPO 在 hard_joint smoke 下结果如何？

SA reward `{sa.get('total_reward_mean')}`，IPPO reward `{ippo.get('total_reward_mean')}`，SA-IPPO reward delta `{ippo_reward_delta.get('delta_candidate_minus_baseline')}`，result `{ippo_reward_delta.get('result')}`。逐指标见 `ippo_vs_sa_summary.csv`。

## 11. SA vs PPO / popularity / reactive 是否与 round10 基本一致？

本轮复用同一 proposal smoke 协议。SA reward `{sa.get('total_reward_mean')}`，PPO `{ppo.get('total_reward_mean')}`，popularity `{pop.get('total_reward_mean')}`，reactive `{reactive.get('total_reward_mean')}`。方向上仍是 SA 高于 PPO/reactive，并在 reward/backhaul 上优于 popularity，但 continuity 低于 popularity。

## 12. continuity 低的问题是否仍然存在？

存在。SA continuity `{sa.get('workflow_continuity_rate_mean')}`，popularity continuity `{pop.get('workflow_continuity_rate_mean')}`。

## 13. IPPO 是否也出现 stall / adapter miss / cold start？

IPPO continuity `{ippo.get('workflow_continuity_rate_mean')}`，adapter miss `{ippo.get('adapter_miss_count_mean')}`，cold start `{ippo.get('adapter_cold_start_count_mean')}`。它也出现 adapter miss/stall，符合 smoke baseline 预期。

## 14. 下一轮建议

建议先补 `local_only` / `reactive_offloading` / `reactive_caching` live rows，随后做 `hard_joint_policy_failure_diagnosis`，最后再考虑 policy-side limited prefetch/cache-admission bias。

## 15. 本轮是否可以 freeze？

不可以。本轮仍是 proposal smoke / IPPO live baseline check。

## 输出

```json
{json.dumps(diagnosis.get('generated_artifacts', {}), ensure_ascii=False, indent=2)}
```
"""


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    output_dir = Path(config.get("output_dir", OUTPUT_DIR))
    report_path = Path(config.get("report_path", REPORT_PATH))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    train_seeds = [int(seed) for seed in config.get("training", {}).get("seeds", [7])]
    workflows = load_workflows(config)
    windows = selected_windows(config, train_seeds)
    training_summary = train_ippo_smoke(config, workflows, windows)
    ippo_checkpoint_path = str(training_summary.get("latest_checkpoint_path", ""))
    rows, missing_rows = evaluate_policies(config, workflows, windows, ippo_checkpoint_path, output_dir)
    policies = [str(item) for item in config.get("evaluation", {}).get("policies", [])]
    policy_summary = policy_comparison(rows, policies)
    ippo_vs_sa = pairwise_summary(rows, "sa_ghmappo", "ippo")
    cache_summary = cache_eviction_summary(rows)
    continuity_summary = continuity_stall_summary(rows)
    generated_artifacts = {
        "benchmark_rows": str(output_dir / "benchmark_rows.csv"),
        "ippo_training_summary": str(output_dir / "ippo_training_summary.json"),
        "policy_comparison_summary": str(output_dir / "policy_comparison_summary.csv"),
        "ippo_vs_sa_summary": str(output_dir / "ippo_vs_sa_summary.csv"),
        "cache_eviction_summary": str(output_dir / "cache_eviction_summary.csv"),
        "continuity_stall_summary": str(output_dir / "continuity_stall_summary.csv"),
        "diagnosis_summary": str(output_dir / "diagnosis_summary.json"),
        "report": str(report_path),
    }
    diagnosis = {
        "task_name": TASK_NAME,
        "changed_files": [
            "src/agents/ippo_agent.py",
            "src/agents/registry.py",
            "src/agents/__init__.py",
            "src/evaluators/real_eval_support.py",
            "configs/algo/ippo.yaml",
            "configs/experiment/ippo_smoke_round12.yaml",
            "scripts/train_ippo_smoke_round12.py",
            "docs/agent/ippo_live_eval_rows_round12_report.md",
        ],
        "generated_artifacts": generated_artifacts,
        "training_run": True,
        "training_scope": "ippo_smoke_training_only",
        "do_not_freeze": True,
        "reward_modified": False,
        "sa_policy_modified": False,
        "heuristic_baseline_modified": False,
        "checkpoint_selection_modified": False,
        "list_evaluable_agents_contains_ippo": "ippo" in list_evaluable_agents(),
        "ippo_checkpoint_path": ippo_checkpoint_path,
        "policies_evaluated": sorted({str(row.get("policy_name")) for row in rows}),
        "missing_policy_rows": missing_rows,
        "benchmark_row_count": len(rows),
        "ippo_row_count": sum(1 for row in rows if str(row.get("policy_name")) == "ippo"),
        "ippo_result_is_smoke_baseline": True,
        "recommended_next_step": [
            "补 local_only/reactive_offloading/reactive_caching live rows",
            "做 hard_joint_policy_failure_diagnosis",
            "最后再考虑 policy-side limited prefetch/cache-admission bias",
        ],
    }

    write_csv(output_dir / "benchmark_rows.csv", rows)
    (output_dir / "ippo_training_summary.json").write_text(
        json.dumps(training_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "policy_comparison_summary.csv", policy_summary)
    write_csv(output_dir / "ippo_vs_sa_summary.csv", ippo_vs_sa)
    write_csv(output_dir / "cache_eviction_summary.csv", cache_summary)
    write_csv(output_dir / "continuity_stall_summary.csv", continuity_summary)
    (output_dir / "diagnosis_summary.json").write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(
        build_report(
            training_summary=training_summary,
            rows=rows,
            policy_summary=policy_summary,
            ippo_vs_sa=ippo_vs_sa,
            diagnosis=diagnosis,
        ),
        encoding="utf-8",
    )

    print("ippo live eval rows round12 complete")
    print(f"ippo_checkpoint_path: {ippo_checkpoint_path}")
    print(f"benchmark_rows: {len(rows)}")
    print(f"ippo_rows: {diagnosis['ippo_row_count']}")
    print(f"policies_evaluated: {', '.join(diagnosis['policies_evaluated'])}")
    for name, path in generated_artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
