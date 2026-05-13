"""Run a local HF model-cache profile experiment aligned with Transactions work.

This script does not download Hugging Face files. It projects the audited Hub
file-size metadata into PPO_MEC cache objects, then runs a small local
NGSIM+Alibaba experiment with cache/offload/migration metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.registry import build_agent, checkpoint_required_agents, get_algo_spec, list_evaluable_agents, list_trainable_agents
from src.data.mobility.replay_provider import ReplayProvider
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.data.workflow.workflow_dataset_builder import WorkflowDatasetBuilder
from src.envs.core.predictor_manager import PredictorManager
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.evaluators.main_results_support import (
    MAIN_RESULT_METRICS,
    aggregate_rows,
    build_win_tie_loss_summary,
    clone_frames,
    clone_rsu_state,
    clone_workflow_state,
    load_checkpoint_metadata,
    load_window_bundle,
    resolve_window_candidates,
    run_real_episode,
    summary_to_row,
    write_rows_csv,
)
from src.evaluators.real_eval_support import build_inference_agent
from src.metrics.recorder import EpisodeRecorder
from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer


HF_TO_ADAPTER_PROJECTION = [
    {
        "dataset_id": "ClemSummer/qwen-model-cache",
        "adapter_id": "adapter_perception",
        "projection_role": "source_or_input_perception_adapter",
    },
    {
        "dataset_id": "ClemSummer/cbow-model-cache",
        "adapter_id": "adapter_tracking",
        "projection_role": "middle_tracking_adapter",
    },
    {
        "dataset_id": "Efficient-Large-Model/imagenet-llamagen-cache",
        "adapter_id": "adapter_fusion",
        "projection_role": "multi_parent_fusion_adapter",
    },
    {
        "dataset_id": "Kuperberg/bert-model-cache",
        "adapter_id": "adapter_intent",
        "projection_role": "late_intent_adapter",
    },
    {
        "dataset_id": "ClemSummer/qwen-model-cache",
        "adapter_id": "adapter_control",
        "projection_role": "sink_or_output_control_adapter",
    },
]

TRANSACTION_ALIGNED_METRICS = [
    "total_reward",
    "workflow_continuity_rate",
    "successful_episode_rate",
    "end_to_end_workflow_delay",
    "handoff_failure_rate",
    "handoff_ready_ratio",
    "adapter_warm_hit_ratio",
    "adapter_hit_count",
    "adapter_miss_count",
    "adapter_cold_start_count",
    "cross_rsu_cold_start_frequency",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
    "cache_eviction_count",
    "eviction_count",
    "cache_admission_count",
    "cache_occupancy_rate",
    "rsu_adapter_slots",
    "prefetch_action_count",
    "migration_action_count",
    "mechanism_realization_rate",
]

LOWER_IS_BETTER = {
    "end_to_end_workflow_delay",
    "handoff_failure_rate",
    "adapter_miss_count",
    "adapter_cold_start_count",
    "cross_rsu_cold_start_frequency",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
    "cache_eviction_count",
    "eviction_count",
}

TRAIN_SUMMARY_METRICS = [
    "total_reward",
    "workflow_continuity_rate",
    "handoff_failure_rate",
    "handoff_ready_ratio",
    "adapter_warm_hit_ratio",
    "cross_rsu_cold_start_frequency",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
    "mechanism_realization_rate",
]

CHECKPOINT_REQUIRED_AGENTS = checkpoint_required_agents()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HF model-cache transaction-aligned local experiment")
    parser.add_argument("--hf_manifest_path", type=str, default=str(ROOT_DIR / "data" / "raw" / "model_cache" / "huggingface_model_cache_sources.json"))
    parser.add_argument("--base_catalog_path", type=str, default=str(ROOT_DIR / "src" / "data" / "model_catalog" / "sample_model_catalog.json"))
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "artifacts" / "experiments" / "hf_model_cache_transaction_round1"))
    parser.add_argument("--agents", nargs="+", default=["sa_ghmappo", "ppo", "dqn", "reactive_greedy", "popularity_cache_heuristic"], choices=list_evaluable_agents())
    parser.add_argument("--train_agents", nargs="*", default=["sa_ghmappo", "ppo", "dqn"], choices=list_trainable_agents())
    parser.add_argument("--sa_checkpoint_path", type=str, default="")
    parser.add_argument("--ppo_checkpoint_path", type=str, default="")
    parser.add_argument("--mappo_checkpoint_path", type=str, default="")
    parser.add_argument("--ippo_checkpoint_path", type=str, default="")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--update_every", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--clip_ratio", type=float, default=0.1)
    parser.add_argument("--entropy_coef", type=float, default=0.003)
    parser.add_argument("--value_coef", type=float, default=0.7)
    parser.add_argument("--auxiliary_coef", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--max_mobility_rows", type=int, default=1500)
    parser.add_argument("--max_workflows", type=int, default=1)
    parser.add_argument("--workflow_csv_path", type=str, default=str(ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"))
    parser.add_argument("--workflow_selector", type=str, default="ordered")
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--mobility_source", choices=["ngsim", "lust"], default="ngsim")
    parser.add_argument("--mobility_csv_path", type=str, default="")
    parser.add_argument(
        "--lust_scenario_root",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "mobility" / "LuSTScenario" / "LuSTScenario-master" / "scenario"),
    )
    parser.add_argument("--rsu_layout", type=str, default="auto_dominant_tight")
    parser.add_argument("--frame_offset", type=int, default=0)
    parser.add_argument("--window_length", type=int, default=24)
    parser.add_argument("--window_count", type=int, default=2)
    parser.add_argument("--window_scan_stride", type=int, default=2)
    parser.add_argument("--window_selector", type=str, default="max_handoff_candidate", choices=["ordered", "random", "max_handoff_candidate", "max_axis_crossing"])
    parser.add_argument("--window_mode", type=str, default="mixed_informative", choices=["activating_only", "mixed", "full", "mixed_informative", "full_stratified"])
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--rsu_adapter_slots", type=int, default=2)
    parser.add_argument("--skip_training", action="store_true")
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_metric(rows: list[dict[str, Any]], metric_name: str) -> float:
    values = [float_value(row.get(metric_name), 0.0) for row in rows]
    return round(fmean(values), 6) if values else 0.0


def build_hf_adapter_catalog(
    *,
    base_catalog_path: str | Path,
    hf_manifest_path: str | Path,
) -> tuple[AdapterCatalog, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    base_payload = read_json(base_catalog_path)
    manifest = read_json(hf_manifest_path)
    source_map = {str(item.get("dataset_id")): item for item in manifest.get("sources", [])}
    cache_objects: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []

    for projection in HF_TO_ADAPTER_PROJECTION:
        dataset_id = projection["dataset_id"]
        source = source_map.get(dataset_id)
        if source is None:
            raise KeyError(f"HF dataset missing from manifest: {dataset_id}")
        size_mb = float_value(source.get("file_summary", {}).get("total_size_mb"), 0.0)
        if size_mb <= 0:
            raise ValueError(f"HF dataset has no usable total_size_mb: {dataset_id}")
        adapter_id = projection["adapter_id"]
        representative_files = source.get("file_summary", {}).get("representative_files", [])
        cache_objects.append(
            {
                "object_id": f"hf_{adapter_id}",
                "adapter_id": adapter_id,
                "size_mb": round(size_mb, 6),
                "source": f"hf_dataset:{dataset_id}:file_summary.total_size_mb",
            }
        )
        projection_rows.append(
            {
                "adapter_id": adapter_id,
                "dataset_id": dataset_id,
                "dataset_name": source.get("dataset_name", ""),
                "projected_size_mb": round(size_mb, 6),
                "projection_role": projection["projection_role"],
                "representative_files": ";".join(str(item) for item in representative_files),
                "safe_scope": source.get("fit_assessment", {}).get("safe_integration_scope", ""),
                "claim_boundary": "file_size_profile_only_not_vec_cache_trace",
            }
        )

    payload = {
        "vehicle_base_models": list(base_payload.get("vehicle_base_models", [])),
        "rsu_adapter_caches": [
            {"rsu_id": "rsu_a", "cached_adapter_ids": ["adapter_perception", "adapter_tracking"]},
            {"rsu_id": "rsu_b", "cached_adapter_ids": ["adapter_tracking", "adapter_fusion"]},
            {"rsu_id": "rsu_c", "cached_adapter_ids": ["adapter_intent", "adapter_control"]},
        ],
        "adapter_state_bundles": [
            {
                "bundle_id": f"hf_bundle_{projection['adapter_id']}",
                "adapter_id": projection["adapter_id"],
                "state_version": "hf_profile_v1",
                "continuity_token": f"hf_token_{projection['adapter_id']}",
                "serialized_state_ref": f"hf_profile://state/{projection['adapter_id']}",
            }
            for projection in HF_TO_ADAPTER_PROJECTION
        ],
        "cache_objects": cache_objects,
        "model_cache_datasets": list(base_payload.get("model_cache_datasets", [])),
    }
    diagnosis = {
        "profile_name": "hf_file_size_transaction_profile_v1",
        "source_boundary": manifest.get("source_boundary", "audited_metadata_and_file_size_no_automatic_download"),
        "hf_manifest_path": str(hf_manifest_path),
        "adapter_projection_count": len(projection_rows),
        "projected_total_size_mb": round(sum(float_value(row["projected_size_mb"]) for row in projection_rows), 6),
        "cannot_claim": [
            "real HF cache hit/miss trace",
            "real HF RSU locality trace",
            "real HF handoff demand trace",
            "real HF adapter state migration trace",
        ],
        "transaction_alignment": [
            "model-cache object size and backhaul cost",
            "cache hit/miss and eviction",
            "offloading target and workflow delay",
            "handoff continuity and adapter-state migration",
            "multi-timescale cache/offload/migration actions",
        ],
    }
    return AdapterCatalog.from_dict(payload), payload, projection_rows, diagnosis


def build_workflow_states(args: argparse.Namespace, seed: int) -> list[Any]:
    builder = WorkflowDatasetBuilder()
    return builder.build_selected_alibaba_workflow_states(
        csv_path=args.workflow_csv_path,
        max_workflows=args.max_workflows,
        workflow_selector=args.workflow_selector,
        min_tasks=args.min_tasks,
        max_tasks=args.max_tasks,
        random_seed=seed,
        adapter_assignment_profile="semantic_ai_service",
    )


def train_row_from_summary(summary: dict[str, Any], *, episode_index: int, updated: bool) -> dict[str, Any]:
    metrics = summary.get("system_metrics", {})
    handoff = summary.get("handoff_summary", {})
    prefetch = summary.get("prefetch_summary", {})
    validation = summary.get("prefetch_validation_summary", {})
    mechanism_realized = float(
        validation.get("validated_predictive_prefetch_count", 0) > 0
        or handoff.get("handoff_ready_count", 0) > 0
        or handoff.get("migration_during_handoff_count", 0) > 0
    )
    run_info = summary.get("run_info", {})
    return {
        "episode_index": episode_index,
        "agent_name": run_info.get("agent_name"),
        "workflow_id": run_info.get("workflow_id"),
        "window_id": run_info.get("window_id"),
        "window_class": run_info.get("window_class"),
        "updated": bool(updated),
        "episode_success": bool(summary.get("episode_success", False)),
        "total_reward": float_value(summary.get("reward_breakdown", {}).get("total", {}).get("sum"), 0.0),
        "workflow_continuity_rate": float_value(metrics.get("workflow_continuity_rate"), 0.0),
        "handoff_failure_rate": float_value(metrics.get("handoff_failure_rate"), 0.0),
        "handoff_ready_ratio": float_value(metrics.get("handoff_ready_ratio"), 0.0),
        "adapter_warm_hit_ratio": float_value(metrics.get("adapter_warm_hit_ratio"), 0.0),
        "cross_rsu_cold_start_frequency": float_value(metrics.get("cross_rsu_cold_start_frequency"), 0.0),
        "backhaul_traffic_cost": float_value(metrics.get("backhaul_traffic_cost"), 0.0),
        "adapter_state_migration_overhead": float_value(metrics.get("adapter_state_migration_overhead"), 0.0),
        "predictive_prefetch_request_count": int(prefetch.get("true_predictive_prefetch_count", 0) or 0),
        "validated_predictive_prefetch_count": int(validation.get("validated_predictive_prefetch_count", 0) or 0),
        "migration_prepare_count": int(handoff.get("migration_prepare_count", 0) or 0),
        "migration_during_handoff_count": int(handoff.get("migration_during_handoff_count", 0) or 0),
        "handoff_ready_count": int(handoff.get("handoff_ready_count", 0) or 0),
        "handoff_total_count": int(handoff.get("handoff_total_count", 0) or 0),
        "mechanism_realization_rate": mechanism_realized,
    }


def agent_kwargs(agent_name: str, args: argparse.Namespace, seed: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "random_seed": seed,
        "learning_rate": args.learning_rate,
        "clip_ratio": args.clip_ratio,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "batch_size": args.batch_size,
        "deterministic_action": False,
    }
    if agent_name == "sa_ghmappo":
        kwargs.update(
            {
                "auxiliary_coef": args.auxiliary_coef,
                "continuity_guard_enabled": True,
                "handoff_target_alignment_guard_enabled": True,
                "continuity_guard_hard_override_enabled": True,
                "mechanism_aux_coef": 0.05,
                "mechanism_window_weight": 1.25,
                "prepare_action_prior_weight": 0.5,
            }
        )
    return kwargs


def annotate_checkpoint(path: Path, metadata: dict[str, Any]) -> None:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        payload["training_metadata"] = dict(metadata)
        torch.save(payload, path)


def save_agent_checkpoint(agent: Any, path: Path, metadata: dict[str, Any]) -> None:
    agent.save(str(path))
    annotate_checkpoint(path, metadata)


def train_one_agent(
    *,
    agent_name: str,
    seed: int,
    args: argparse.Namespace,
    adapter_catalog: AdapterCatalog,
    cache_capacity_profile: dict[str, Any],
    selected_window_plan: list[dict[str, Any]],
    workflow_states: list[Any],
    output_root: Path,
    run_id: str,
) -> dict[str, Any]:
    train_root = output_root / "training" / agent_name / f"{agent_name}_hf_train_seed{seed}"
    episode_root = train_root / "episodes"
    checkpoint_root = train_root / "checkpoints"
    episode_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    agent = build_agent(agent_name, **agent_kwargs(agent_name, args, seed))

    rows: list[dict[str, Any]] = []
    update_logs: list[dict[str, Any]] = []
    pending_rollout: list[dict[str, Any]] = []
    update_index = 0
    latest_checkpoint_path = checkpoint_root / "latest.pt"
    for episode_index in range(1, args.episodes + 1):
        workflow_state = workflow_states[(episode_index - 1) % len(workflow_states)]
        window_candidate = selected_window_plan[(episode_index - 1) % len(selected_window_plan)]
        mobility_bundle = load_window_bundle(
            root_dir=ROOT_DIR,
            mobility_source=args.mobility_source,
            mobility_csv_path=args.mobility_csv_path,
            lust_scenario_root=args.lust_scenario_root,
            max_mobility_rows=args.max_mobility_rows,
            rsu_layout=str(window_candidate.get("recommended_rsu_layout", args.rsu_layout)),
            frame_offset=int(window_candidate.get("frame_offset", args.frame_offset)),
            window_length=int(window_candidate.get("window_length", args.window_length)),
            random_seed=seed,
        )
        mobility_bundle.rsu_metadata["window_rank"] = window_candidate.get("window_rank")
        mobility_bundle.rsu_metadata["window_class"] = window_candidate.get("window_class")
        recorder = EpisodeRecorder(prefetch_validation_window=6)
        core_env = VecWorkflowCoreEnv(
            mobility_provider=ReplayProvider(trajectory_frames=clone_frames(mobility_bundle.frames)),
            workflow_state=clone_workflow_state(workflow_state),
            adapter_catalog=AdapterCatalog.from_dict(adapter_catalog.to_dict()),
            rsu_states=[clone_rsu_state(rsu_state) for rsu_state in mobility_bundle.rsu_states],
            predictor_manager=PredictorManager(random_seed=seed + episode_index),
            max_steps=max(args.max_steps + 2, 8),
            mobility_source=args.mobility_source,
            cache_capacity_profile=cache_capacity_profile,
        )
        env = GymVecEnv(core_env=core_env, recorder=recorder)
        trainer = MARLOnPolicyTrainer(
            env=env,
            agent=agent,
            recorder=recorder,
            max_steps=args.max_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        summary, rollout = trainer.collect_episode(
            run_metadata={
                "script": "scripts/run_hf_model_cache_transaction_experiment.py",
                "run_id": run_id,
                "agent_name": agent_name,
                "workflow_id": workflow_state.workflow_id,
                "window_id": mobility_bundle.rsu_metadata.get("window_id"),
                "window_mode": args.window_mode,
                "window_class": mobility_bundle.rsu_metadata.get("window_class"),
                "model_cache_profile": "hf_file_size_transaction_profile_v1",
                "adapter_assignment_profile": "semantic_ai_service",
            }
        )
        summary["episode_success"] = bool(summary.get("episode_status", {}).get("completed", False))
        pending_rollout.extend(rollout)
        should_update = episode_index % max(args.update_every, 1) == 0 or episode_index == args.episodes
        if should_update:
            update_index += 1
            learn_info = agent.learn(pending_rollout)
            pending_rollout = []
            metadata = {
                "run_id": run_id,
                "agent_name": agent_name,
                "config_profile": "hf_model_cache_transaction_round1",
                "train_window_mode": args.window_mode,
                "episodes": args.episodes,
                "update_count": update_index,
                "is_smoke_checkpoint": False,
                "script": "scripts/run_hf_model_cache_transaction_experiment.py",
                "model_cache_profile": "hf_file_size_transaction_profile_v1",
            }
            save_agent_checkpoint(agent, checkpoint_root / f"update_{update_index:04d}.pt", metadata)
            save_agent_checkpoint(agent, latest_checkpoint_path, metadata)
            update_logs.append({"episode_index": episode_index, **learn_info})
        rows.append(train_row_from_summary(summary, episode_index=episode_index, updated=should_update))
        write_json(episode_root / f"episode_{episode_index:04d}.summary.json", summary)

    convergence = build_convergence_summary(agent_name=agent_name, seed=seed, rows=rows, update_logs=update_logs)
    train_summary = {
        "run_id": run_id,
        "agent_name": agent_name,
        "algo_spec": get_algo_spec(agent_name),
        "config_profile": "hf_model_cache_transaction_round1",
        "model_cache_profile": "hf_file_size_transaction_profile_v1",
        "adapter_assignment_profile": "semantic_ai_service",
        "episodes": args.episodes,
        "update_every": args.update_every,
        "update_count": update_index,
        "latest_checkpoint_path": str(latest_checkpoint_path),
        "mean_metrics": {metric: mean_metric(rows, metric) for metric in TRAIN_SUMMARY_METRICS},
        "convergence_summary": convergence,
        "rows": rows,
        "update_logs": update_logs,
    }
    write_csv(train_root / "train.csv", rows)
    write_json(train_root / "train_summary.json", train_summary)
    write_json(train_root / "summary.json", train_summary)
    return {
        "agent_name": agent_name,
        "seed": seed,
        "latest_checkpoint_path": str(latest_checkpoint_path),
        "train_csv_path": str(train_root / "train.csv"),
        "train_summary_path": str(train_root / "train_summary.json"),
        "convergence_summary": convergence,
        "rows": rows,
        "update_logs": update_logs,
    }


def build_convergence_summary(
    *,
    agent_name: str,
    seed: int,
    rows: list[dict[str, Any]],
    update_logs: list[dict[str, Any]],
) -> dict[str, Any]:
    first_count = max(1, min(2, len(rows)))
    last_count = max(1, min(2, len(rows)))
    first_reward = round(fmean(float_value(row["total_reward"]) for row in rows[:first_count]), 6) if rows else 0.0
    last_reward = round(fmean(float_value(row["total_reward"]) for row in rows[-last_count:]), 6) if rows else 0.0
    best_reward = round(max((float_value(row["total_reward"]) for row in rows), default=0.0), 6)
    return {
        "agent_name": agent_name,
        "seed": seed,
        "episode_count": len(rows),
        "update_count": len(update_logs),
        "first_reward_mean": first_reward,
        "last_reward_mean": last_reward,
        "reward_delta_last_minus_first": round(last_reward - first_reward, 6),
        "best_episode_reward": best_reward,
        "final_continuity_mean": round(fmean(float_value(row["workflow_continuity_rate"]) for row in rows[-last_count:]), 6) if rows else 0.0,
        "final_handoff_ready_ratio_mean": round(fmean(float_value(row["handoff_ready_ratio"]) for row in rows[-last_count:]), 6) if rows else 0.0,
    }


def checkpoint_arg_map(args: argparse.Namespace) -> dict[str, str]:
    return {
        "sa_ghmappo": args.sa_checkpoint_path,
        "ppo": args.ppo_checkpoint_path,
        "mappo": args.mappo_checkpoint_path,
        "ippo": args.ippo_checkpoint_path,
    }


def build_checkpoint_map(
    *,
    args: argparse.Namespace,
    training_results: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, dict[str, str]]:
    explicit = checkpoint_arg_map(args)
    checkpoint_map: dict[str, dict[str, str]] = {}
    for agent_name in args.agents:
        if agent_name not in CHECKPOINT_REQUIRED_AGENTS:
            checkpoint_map[agent_name] = {str(seed): "" for seed in args.seeds}
            continue
        per_seed: dict[str, str] = {}
        for seed in args.seeds:
            trained_path = training_results.get(agent_name, {}).get(seed, {}).get("latest_checkpoint_path", "")
            per_seed[str(seed)] = trained_path or explicit.get(agent_name, "")
        checkpoint_map[agent_name] = per_seed
    return checkpoint_map


def flat_checkpoint_map_for_seed(checkpoint_map: dict[str, dict[str, str]], seed: int) -> dict[str, str]:
    seed_key = str(seed)
    return {
        agent_name: per_seed.get(seed_key, "")
        for agent_name, per_seed in checkpoint_map.items()
    }


def validate_checkpoint_map(args: argparse.Namespace, checkpoint_map: dict[str, dict[str, str]]) -> None:
    missing: list[str] = []
    for agent_name in args.agents:
        if agent_name not in CHECKPOINT_REQUIRED_AGENTS:
            continue
        for seed in args.seeds:
            checkpoint_path = checkpoint_map.get(agent_name, {}).get(str(seed), "")
            if not checkpoint_path:
                missing.append(f"{agent_name}:seed{seed}")
            elif not Path(checkpoint_path).exists():
                missing.append(f"{agent_name}:seed{seed}:{checkpoint_path}")
    if missing:
        raise FileNotFoundError("Missing checkpoint(s) for evaluation: " + ", ".join(missing))


def run_benchmark(
    *,
    args: argparse.Namespace,
    adapter_catalog: AdapterCatalog,
    cache_capacity_profile: dict[str, Any],
    selected_window_plan: list[dict[str, Any]],
    checkpoint_map: dict[str, dict[str, str]],
    output_root: Path,
    run_id: str,
    catalog_path: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    selected_workflow_ids_by_seed: dict[str, list[str]] = {}
    for seed in args.seeds:
        workflow_states = build_workflow_states(args, seed)
        selected_workflow_ids_by_seed[str(seed)] = [workflow.workflow_id for workflow in workflow_states]
        seed_checkpoint_map = flat_checkpoint_map_for_seed(checkpoint_map, seed)
        for window_candidate in selected_window_plan:
            mobility_bundle = load_window_bundle(
                root_dir=ROOT_DIR,
                mobility_source=args.mobility_source,
                mobility_csv_path=args.mobility_csv_path,
                lust_scenario_root=args.lust_scenario_root,
                max_mobility_rows=args.max_mobility_rows,
                rsu_layout=str(window_candidate.get("recommended_rsu_layout", args.rsu_layout)),
                frame_offset=int(window_candidate.get("frame_offset", args.frame_offset)),
                window_length=int(window_candidate.get("window_length", args.window_length)),
                random_seed=seed,
            )
            mobility_bundle.rsu_metadata["window_rank"] = window_candidate.get("window_rank")
            mobility_bundle.rsu_metadata["window_class"] = window_candidate.get("window_class")
            for workflow_state in workflow_states:
                for agent_name in args.agents:
                    summary = run_real_episode(
                        root_dir=ROOT_DIR,
                        agent_name=agent_name,
                        checkpoint_map=seed_checkpoint_map,
                        workflow_state=workflow_state,
                        workflow_source_path=args.workflow_csv_path,
                        mobility_bundle=mobility_bundle,
                        seed=seed,
                        max_steps=args.max_steps,
                        mobility_source=args.mobility_source,
                        run_metadata={
                            "script": "scripts/run_hf_model_cache_transaction_experiment.py",
                            "run_id": run_id,
                            "mainline": "NGSIM + Alibaba + HF file-size model-cache profile",
                            "model_cache_profile": "hf_file_size_transaction_profile_v1",
                            "adapter_assignment_profile": "semantic_ai_service",
                            "hf_projection_catalog_path": str(catalog_path),
                            "window_rank": window_candidate.get("window_rank"),
                            "window_class": window_candidate.get("window_class"),
                            "window_mode": args.window_mode,
                        },
                        adapter_catalog_override=AdapterCatalog.from_dict(adapter_catalog.to_dict()),
                        cache_capacity_profile=cache_capacity_profile,
                    )
                    row = summary_to_row(summary)
                    row["model_cache_profile"] = "hf_file_size_transaction_profile_v1"
                    row["adapter_assignment_profile"] = "semantic_ai_service"
                    rows.append(row)

    aggregate_by_agent = aggregate_rows(rows, group_keys=["agent_name"], metrics=MAIN_RESULT_METRICS)
    aggregate_by_window_and_agent = aggregate_rows(rows, group_keys=["window_id", "agent_name"], metrics=MAIN_RESULT_METRICS)
    aggregate_by_workflow_and_agent = aggregate_rows(rows, group_keys=["workflow_id", "agent_name"], metrics=MAIN_RESULT_METRICS)
    transaction_comparison = build_transaction_comparison(aggregate_by_agent)
    summary = {
        "run_id": run_id,
        "experiment_scope": "local_adaptation_not_paper_claim",
        "model_cache_profile": "hf_file_size_transaction_profile_v1",
        "adapter_assignment_profile": "semantic_ai_service",
        "agents": args.agents,
        "train_agents": args.train_agents,
        "seeds": args.seeds,
        "episode_count": len(rows),
        "selected_workflow_ids_by_seed": selected_workflow_ids_by_seed,
        "selected_window_plan": selected_window_plan,
        "cache_capacity_profile": cache_capacity_profile,
        "checkpoint_map": checkpoint_map,
        "checkpoint_metadata": build_checkpoint_metadata_table(checkpoint_map),
        "aggregate_by_agent": aggregate_by_agent,
        "transaction_aligned_comparison": transaction_comparison,
        "win_tie_loss_summary": build_win_tie_loss_summary(
            aggregate_by_window_and_agent=aggregate_by_window_and_agent,
            aggregate_by_workflow_and_agent=aggregate_by_workflow_and_agent,
            metrics=TRANSACTION_ALIGNED_METRICS,
        ),
        "rows": rows,
    }
    write_rows_csv(output_root / "benchmark_rows.csv", rows)
    write_json(output_root / "aggregate_summary.json", summary)
    write_csv(output_root / "algorithm_comparison.csv", transaction_comparison["rows"])
    return summary


def build_checkpoint_metadata_table(checkpoint_map: dict[str, dict[str, str]]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for agent_name, seed_map in checkpoint_map.items():
        metadata[agent_name] = {}
        for seed, checkpoint_path in seed_map.items():
            metadata[agent_name][seed] = load_checkpoint_metadata(checkpoint_path) if checkpoint_path else {
                "checkpoint_path": "",
                "config_profile": "non_checkpoint_agent",
                "run_id": agent_name,
                "episodes": 0,
                "update_count": 0,
            }
    return metadata


def build_transaction_comparison(aggregate_by_agent: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for agent_name, payload in aggregate_by_agent.items():
        metric_payload = payload.get("metrics", {})
        row = {"agent_name": agent_name, "episode_count": payload.get("episode_count", 0)}
        for metric in TRANSACTION_ALIGNED_METRICS:
            row[metric] = metric_payload.get(metric, {}).get("mean", 0.0)
        rows.append(row)

    best_by_metric: dict[str, Any] = {}
    for metric in TRANSACTION_ALIGNED_METRICS:
        candidate_rows = [row for row in rows if metric in row]
        if not candidate_rows:
            continue
        reverse = metric not in LOWER_IS_BETTER
        best_row = sorted(candidate_rows, key=lambda row: float_value(row.get(metric)), reverse=reverse)[0]
        best_by_metric[metric] = {
            "best_agent": best_row["agent_name"],
            "best_value": best_row.get(metric, 0.0),
            "higher_is_better": reverse,
        }

    sa_advantage: dict[str, Any] = {"available": "sa_ghmappo" in aggregate_by_agent}
    if sa_advantage["available"]:
        sa_row = next(row for row in rows if row["agent_name"] == "sa_ghmappo")
        wins: list[str] = []
        losses: list[str] = []
        ties: list[str] = []
        for metric, best in best_by_metric.items():
            best_value = float_value(best["best_value"])
            sa_value = float_value(sa_row.get(metric))
            if abs(sa_value - best_value) <= 1e-6:
                ties.append(metric)
            elif metric in LOWER_IS_BETTER:
                (wins if sa_value < best_value else losses).append(metric)
            else:
                (wins if sa_value > best_value else losses).append(metric)
        sa_advantage.update(
            {
                "best_metric_count": len(ties),
                "metric_ties_for_best": ties,
                "metric_losses_to_best": losses,
                "metric_wins_over_best_reference": wins,
                "advantage_claim_supported_in_this_round": len(ties) >= max(4, len(TRANSACTION_ALIGNED_METRICS) // 3)
                and "total_reward" in ties,
            }
        )
    return {
        "metrics": TRANSACTION_ALIGNED_METRICS,
        "lower_is_better": sorted(LOWER_IS_BETTER),
        "rows": rows,
        "best_by_metric": best_by_metric,
        "sa_advantage": sa_advantage,
    }


def write_report(
    *,
    output_root: Path,
    run_id: str,
    profile_diagnosis: dict[str, Any],
    convergence_rows: list[dict[str, Any]],
    benchmark_summary: dict[str, Any],
) -> None:
    comparison_rows = benchmark_summary["transaction_aligned_comparison"]["rows"]
    sa_advantage = benchmark_summary["transaction_aligned_comparison"]["sa_advantage"]
    lines = [
        "# HF model-cache transaction round1",
        "",
        "## Scope",
        "",
        "This is a local adaptation experiment. HF datasets are used only as audited file-size/cache-volume profiles, not as VEC cache request traces.",
        "",
        "## HF profile",
        "",
        f"- profile_name: `{profile_diagnosis['profile_name']}`",
        f"- projected_total_size_mb: `{profile_diagnosis['projected_total_size_mb']}`",
        "- claim_boundary: `file_size_profile_only_not_vec_cache_trace`",
        "",
        "## Convergence",
        "",
        "| agent | seed | episodes | first_reward | last_reward | delta | best_reward |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in convergence_rows:
        lines.append(
            "| {agent_name} | {seed} | {episode_count} | {first_reward_mean} | {last_reward_mean} | {reward_delta_last_minus_first} | {best_episode_reward} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Algorithm comparison",
            "",
            "| agent | reward | continuity | handoff_fail | warm_hit | cold_start | backhaul | migration_overhead | eviction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        lines.append(
            "| {agent_name} | {total_reward} | {workflow_continuity_rate} | {handoff_failure_rate} | {adapter_warm_hit_ratio} | {cross_rsu_cold_start_frequency} | {backhaul_traffic_cost} | {adapter_state_migration_overhead} | {eviction_count} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## SA advantage",
            "",
            f"- advantage_claim_supported_in_this_round: `{sa_advantage.get('advantage_claim_supported_in_this_round')}`",
            f"- best_metric_count: `{sa_advantage.get('best_metric_count')}`",
            f"- metric_ties_for_best: `{';'.join(sa_advantage.get('metric_ties_for_best', []))}`",
            f"- metric_losses_to_best: `{';'.join(sa_advantage.get('metric_losses_to_best', []))}`",
            "",
            "## Artifacts",
            "",
            f"- aggregate_summary: `{output_root / 'aggregate_summary.json'}`",
            f"- algorithm_comparison: `{output_root / 'algorithm_comparison.csv'}`",
            f"- convergence_rewards: `{output_root / 'convergence_rewards.csv'}`",
            f"- hf_projection_mapping: `{output_root / 'hf_projection_mapping.csv'}`",
            f"- run_manifest: `{output_root / 'run_manifest.json'}`",
        ]
    )
    (output_root / "hf_model_cache_transaction_round1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_id = datetime.now().strftime("hf_model_cache_transaction_round1_%Y%m%d_%H%M%S_%f")
    output_root = Path(args.output_root) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    adapter_catalog, catalog_payload, projection_rows, profile_diagnosis = build_hf_adapter_catalog(
        base_catalog_path=args.base_catalog_path,
        hf_manifest_path=args.hf_manifest_path,
    )
    generated_catalog_path = output_root / "hf_model_cache_adapter_catalog.json"
    write_json(generated_catalog_path, catalog_payload)
    write_csv(output_root / "hf_projection_mapping.csv", projection_rows)
    write_json(output_root / "hf_profile_diagnosis.json", profile_diagnosis)

    cache_capacity_profile = {
        "enabled": True,
        "unit": "adapter_slots",
        "profile_name": "hf_transaction_adapter_slot_stress",
        "rsu_adapter_slots": int(args.rsu_adapter_slots),
        "eviction_policy": "lru",
        "telemetry_enabled": True,
        "controlled_cache_stress_setting": True,
    }
    _, window_payload = resolve_window_candidates(
        root_dir=ROOT_DIR,
        mobility_source=args.mobility_source,
        mobility_csv_path=args.mobility_csv_path,
        lust_scenario_root=args.lust_scenario_root,
        max_mobility_rows=args.max_mobility_rows,
        rsu_layout=args.rsu_layout,
        frame_offset=args.frame_offset,
        window_length=args.window_length,
        window_selector=args.window_selector,
        window_count=args.window_count,
        window_scan_stride=args.window_scan_stride,
        random_seed=args.seeds[0] if args.seeds else 7,
        window_mode=args.window_mode,
    )
    selected_window_plan = list(window_payload.get("selected_windows", []))
    if not selected_window_plan:
        raise RuntimeError("No selected windows for HF transaction experiment.")

    training_results: dict[str, dict[int, dict[str, Any]]] = {}
    convergence_rows: list[dict[str, Any]] = []
    if not args.skip_training:
        for seed in args.seeds:
            workflow_states = build_workflow_states(args, seed)
            for agent_name in args.train_agents:
                if agent_name not in args.agents:
                    continue
                result = train_one_agent(
                    agent_name=agent_name,
                    seed=seed,
                    args=args,
                    adapter_catalog=adapter_catalog,
                    cache_capacity_profile=cache_capacity_profile,
                    selected_window_plan=selected_window_plan,
                    workflow_states=workflow_states,
                    output_root=output_root,
                    run_id=run_id,
                )
                training_results.setdefault(agent_name, {})[seed] = result
                convergence_rows.append(result["convergence_summary"])

    checkpoint_map = build_checkpoint_map(args=args, training_results=training_results)
    validate_checkpoint_map(args, checkpoint_map)
    benchmark_summary = run_benchmark(
        args=args,
        adapter_catalog=adapter_catalog,
        cache_capacity_profile=cache_capacity_profile,
        selected_window_plan=selected_window_plan,
        checkpoint_map=checkpoint_map,
        output_root=output_root,
        run_id=run_id,
        catalog_path=generated_catalog_path,
    )

    write_csv(output_root / "convergence_rewards.csv", convergence_rows)
    run_manifest = {
        "run_id": run_id,
        "script": "scripts/run_hf_model_cache_transaction_experiment.py",
        "output_root": str(output_root),
        "generated_catalog_path": str(generated_catalog_path),
        "hf_manifest_path": args.hf_manifest_path,
        "base_catalog_path": args.base_catalog_path,
        "cache_capacity_profile": cache_capacity_profile,
        "args": vars(args),
        "profile_diagnosis": profile_diagnosis,
        "checkpoint_map": checkpoint_map,
        "training_results": {
            agent: {str(seed): {key: value for key, value in result.items() if key != "rows"} for seed, result in seed_map.items()}
            for agent, seed_map in training_results.items()
        },
        "outputs": {
            "aggregate_summary": str(output_root / "aggregate_summary.json"),
            "algorithm_comparison": str(output_root / "algorithm_comparison.csv"),
            "convergence_rewards": str(output_root / "convergence_rewards.csv"),
            "hf_projection_mapping": str(output_root / "hf_projection_mapping.csv"),
            "report": str(output_root / "hf_model_cache_transaction_round1_report.md"),
        },
    }
    write_json(output_root / "run_manifest.json", run_manifest)
    write_report(
        output_root=output_root,
        run_id=run_id,
        profile_diagnosis=profile_diagnosis,
        convergence_rows=convergence_rows,
        benchmark_summary=benchmark_summary,
    )

    print("HF model-cache transaction experiment complete")
    print(f"run_id: {run_id}")
    print(f"output_root: {output_root}")
    print(f"aggregate_summary_path: {output_root / 'aggregate_summary.json'}")
    print(f"algorithm_comparison_path: {output_root / 'algorithm_comparison.csv'}")
    print(f"convergence_rewards_path: {output_root / 'convergence_rewards.csv'}")
    sa_advantage = benchmark_summary["transaction_aligned_comparison"]["sa_advantage"]
    print(f"sa_advantage_supported: {sa_advantage.get('advantage_claim_supported_in_this_round')}")
    for row in benchmark_summary["transaction_aligned_comparison"]["rows"]:
        print(
            "[{agent}] reward={reward:.3f} continuity={continuity:.3f} backhaul={backhaul:.3f} cold_start={cold:.3f}".format(
                agent=row["agent_name"],
                reward=float_value(row.get("total_reward")),
                continuity=float_value(row.get("workflow_continuity_rate")),
                backhaul=float_value(row.get("backhaul_traffic_cost")),
                cold=float_value(row.get("cross_rsu_cold_start_frequency")),
            )
        )


if __name__ == "__main__":
    main()
