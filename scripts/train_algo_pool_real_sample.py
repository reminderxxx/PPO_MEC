"""Train direction-matched baseline agents on the existing real-sample path."""

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

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.registry import build_agent, get_algo_spec, list_trainable_agents
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.data.mobility.replay_provider import ReplayProvider
from src.envs.core.predictor_manager import PredictorManager
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.evaluators.main_results_support import (
    build_selected_workflow_states,
    clone_frames,
    clone_rsu_state,
    clone_workflow_state,
    load_window_bundle,
    resolve_window_candidates,
)
from src.metrics.recorder import EpisodeRecorder
from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer


TRAINABLE_BASELINES = [agent for agent in list_trainable_agents() if agent != "sa_ghmappo"]
REPLAY_BASELINES = {"dqn", "ddqn", "dueling_dqn", "dueling_ddqn", "qmix"}
PROFILE_DEFAULTS = {
    "smoke": {"episodes": 2, "update_every": 1, "max_steps": 6, "batch_size": 8},
    "baseline_safe": {"episodes": 12, "update_every": 3, "max_steps": 12, "batch_size": 24},
}
SUMMARY_METRICS = [
    "total_reward",
    "end_to_end_workflow_delay",
    "workflow_continuity_rate",
    "handoff_failure_rate",
    "handoff_ready_ratio",
    "adapter_warm_hit_ratio",
    "cross_rsu_cold_start_frequency",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
    "predictive_prefetch_precision",
    "predictive_prefetch_request_count",
    "validated_predictive_prefetch_count",
    "migration_prepare_count",
    "migration_during_handoff_count",
    "handoff_ready_count",
    "handoff_total_count",
    "mechanism_realization_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train direction-matched baseline agents")
    parser.add_argument("--agent_name", choices=TRAINABLE_BASELINES, default="ppo")
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="smoke")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--update_every", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--random_seed", type=int, default=7)
    parser.add_argument("--mobility_source", choices=["ngsim", "lust"], default="ngsim")
    parser.add_argument("--primary_vehicle_selection", choices=["stable_first", "handoff_pressure"], default="stable_first")
    parser.add_argument("--mobility_csv_path", type=str, default="")
    parser.add_argument(
        "--lust_scenario_root",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "mobility" / "LuSTScenario" / "LuSTScenario-master" / "scenario"),
    )
    parser.add_argument("--workflow_csv_path", type=str, default=str(ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"))
    parser.add_argument("--max_mobility_rows", type=int, default=1500)
    parser.add_argument("--max_workflows", type=int, default=2)
    parser.add_argument("--workflow_selector", type=str, default="ordered")
    parser.add_argument("--rsu_layout", type=str, default="auto_dominant_tight")
    parser.add_argument("--frame_offset", type=int, default=0)
    parser.add_argument("--window_length", type=int, default=24)
    parser.add_argument("--window_selector", type=str, default="max_handoff_candidate", choices=["ordered", "random", "max_handoff_candidate", "max_axis_crossing"])
    parser.add_argument("--window_count", type=int, default=1)
    parser.add_argument("--window_scan_stride", type=int, default=2)
    parser.add_argument("--window_mode", type=str, default="activating_only", choices=["activating_only", "mixed", "full", "mixed_informative", "full_stratified"])
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "artifacts" / "training" / "algo_pool"))
    args = parser.parse_args()
    defaults = PROFILE_DEFAULTS[args.profile]
    for field_name in ["episodes", "update_every", "batch_size", "max_steps"]:
        if getattr(args, field_name) is None:
            setattr(args, field_name, defaults[field_name])
    return args


def build_summary_row(summary: dict[str, Any], *, episode_index: int, updated: bool) -> dict[str, Any]:
    metrics = summary["system_metrics"]
    handoff = summary["handoff_summary"]
    prefetch = summary["prefetch_summary"]
    validation = summary["prefetch_validation_summary"]
    mechanism_realized = int(
        validation.get("validated_predictive_prefetch_count", 0) > 0
        or handoff.get("handoff_ready_count", 0) > 0
        or handoff.get("migration_during_handoff_count", 0) > 0
    )
    return {
        "episode_index": episode_index,
        "agent_name": summary["run_info"].get("agent_name"),
        "workflow_id": summary["run_info"].get("workflow_id"),
        "window_id": summary["run_info"].get("window_id"),
        "primary_vehicle_selection": summary["run_info"].get("primary_vehicle_selection", "stable_first"),
        "updated": bool(updated),
        "episode_success": bool(summary.get("episode_success", False)),
        "total_reward": float(summary["reward_breakdown"]["total"]["sum"]),
        "end_to_end_workflow_delay": metrics["end_to_end_workflow_delay"],
        "workflow_continuity_rate": metrics["workflow_continuity_rate"],
        "handoff_failure_rate": metrics["handoff_failure_rate"],
        "handoff_ready_ratio": metrics["handoff_ready_ratio"],
        "adapter_warm_hit_ratio": metrics["adapter_warm_hit_ratio"],
        "cross_rsu_cold_start_frequency": metrics["cross_rsu_cold_start_frequency"],
        "backhaul_traffic_cost": metrics["backhaul_traffic_cost"],
        "adapter_state_migration_overhead": metrics["adapter_state_migration_overhead"],
        "predictive_prefetch_precision": metrics["predictive_prefetch_precision"],
        "handoff_total_count": handoff["handoff_total_count"],
        "handoff_ready_count": handoff["handoff_ready_count"],
        "migration_prepare_count": handoff["migration_prepare_count"],
        "migration_during_handoff_count": handoff["migration_during_handoff_count"],
        "predictive_prefetch_request_count": prefetch["true_predictive_prefetch_count"],
        "validated_predictive_prefetch_count": validation["validated_predictive_prefetch_count"],
        "mechanism_realization_rate": float(mechanism_realized),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        name: round(fmean(float(row[name]) for row in rows), 6) if rows else 0.0
        for name in SUMMARY_METRICS
    }


def annotate_checkpoint(path: Path, metadata: dict[str, Any]) -> None:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        payload["training_metadata"] = dict(metadata)
        torch.save(payload, path)


def main() -> None:
    args = parse_args()
    run_id = datetime.now().strftime(f"{args.agent_name}_train_%Y%m%d_%H%M%S_%f_seed{args.random_seed}")
    output_root = Path(args.output_root) / args.agent_name / run_id
    episode_root = output_root / "episodes"
    checkpoint_root = output_root / "checkpoints"
    episode_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    workflow_states = build_selected_workflow_states(
        workflow_csv_path=args.workflow_csv_path,
        max_workflows=args.max_workflows,
        workflow_selector=args.workflow_selector,
        min_tasks=args.min_tasks,
        max_tasks=args.max_tasks,
        random_seed=args.random_seed,
    )
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
        random_seed=args.random_seed,
        window_mode=args.window_mode,
    )
    selected_window_plan = list(window_payload.get("selected_windows", []))
    if not selected_window_plan:
        selected_window_plan = [
            {
                "frame_offset": args.frame_offset,
                "window_length": args.window_length,
                "recommended_rsu_layout": args.rsu_layout,
                "window_id": f"window_off{args.frame_offset}_len{args.window_length}",
                "window_class": "manual",
            }
        ]
    adapter_catalog = AdapterCatalog.from_json(ROOT_DIR / "src" / "data" / "model_catalog" / "sample_model_catalog.json")
    agent_kwargs = {
        "random_seed": args.random_seed,
        "learning_rate": args.learning_rate,
        "clip_ratio": args.clip_ratio,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "batch_size": args.batch_size,
        "deterministic_action": False,
    }
    if args.profile == "smoke" and args.agent_name in REPLAY_BASELINES:
        smoke_rollout_capacity = max(int(args.max_steps) * max(int(args.update_every), 1), 1)
        agent_kwargs["min_replay_size"] = max(1, min(int(args.batch_size), smoke_rollout_capacity))
    agent = build_agent(args.agent_name, **agent_kwargs)

    pending_rollout: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    update_logs: list[dict[str, Any]] = []
    latest_checkpoint_path = ""
    update_index = 0
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
            random_seed=args.random_seed,
        )
        mobility_bundle.rsu_metadata["window_rank"] = window_candidate.get("window_rank")
        mobility_bundle.rsu_metadata["window_class"] = window_candidate.get("window_class")
        recorder = EpisodeRecorder(prefetch_validation_window=6)
        core_env = VecWorkflowCoreEnv(
            mobility_provider=ReplayProvider(trajectory_frames=clone_frames(mobility_bundle.frames)),
            workflow_state=clone_workflow_state(workflow_state),
            adapter_catalog=adapter_catalog,
            rsu_states=[clone_rsu_state(rsu_state) for rsu_state in mobility_bundle.rsu_states],
            predictor_manager=PredictorManager(random_seed=args.random_seed + episode_index),
            max_steps=max(args.max_steps + 2, 8),
            mobility_source=args.mobility_source,
            primary_vehicle_selection=args.primary_vehicle_selection,
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
                "script": "scripts/train_algo_pool_real_sample.py",
                "run_id": run_id,
                "agent_name": args.agent_name,
                "workflow_id": workflow_state.workflow_id,
                "window_id": mobility_bundle.rsu_metadata.get("window_id"),
                "config_profile": args.profile,
                "window_mode": args.window_mode,
                "window_class": mobility_bundle.rsu_metadata.get("window_class"),
                "primary_vehicle_selection": args.primary_vehicle_selection,
            }
        )
        summary["episode_success"] = bool(summary.get("episode_status", {}).get("completed", False))
        pending_rollout.extend(rollout)
        should_update = episode_index % max(args.update_every, 1) == 0 or episode_index == args.episodes
        if should_update:
            update_index += 1
            learn_info = agent.learn(pending_rollout)
            pending_rollout = []
            checkpoint_path = checkpoint_root / f"update_{update_index:04d}.pt"
            latest_path = checkpoint_root / "latest.pt"
            agent.save(str(checkpoint_path))
            agent.save(str(latest_path))
            latest_checkpoint_path = str(latest_path)
            checkpoint_metadata = {
                "run_id": run_id,
                "agent_name": args.agent_name,
                "config_profile": args.profile,
                "primary_vehicle_selection": args.primary_vehicle_selection,
                "episodes": args.episodes,
                "update_count": update_index,
                "is_smoke_checkpoint": args.profile == "smoke",
                "script": "scripts/train_algo_pool_real_sample.py",
            }
            annotate_checkpoint(checkpoint_path, checkpoint_metadata)
            annotate_checkpoint(latest_path, checkpoint_metadata)
            update_logs.append({"episode_index": episode_index, **learn_info})
        else:
            learn_info = {
                "agent_name": args.agent_name,
                "policy_update_skipped": True,
                "reason": "waiting_for_update_every",
                "pending_rollout_steps": len(pending_rollout),
            }
        summary["agent_info"] = {"agent_name": args.agent_name, "learn_info": learn_info}
        episode_path = episode_root / f"episode_{episode_index:04d}.summary.json"
        episode_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(build_summary_row(summary, episode_index=episode_index, updated=should_update))

    train_csv_path = output_root / "train.csv"
    summary_path = output_root / "summary.json"
    train_summary_path = output_root / "train_summary.json"
    write_csv(train_csv_path, rows)
    summary_payload = {
        "run_id": run_id,
        "agent_name": args.agent_name,
        "algo_spec": get_algo_spec(args.agent_name),
        "profile": args.profile,
        "config_profile": args.profile,
        "episodes": args.episodes,
        "update_every": args.update_every,
        "update_count": update_index,
        "latest_checkpoint_path": latest_checkpoint_path,
        "output_dir": str(output_root),
        "train_csv_path": str(train_csv_path),
        "summary_json_path": str(summary_path),
        "workflow_ids": [workflow_state.workflow_id for workflow_state in workflow_states],
        "selected_window_plan": selected_window_plan,
        "window_mode": args.window_mode,
        "primary_vehicle_selection": args.primary_vehicle_selection,
        "window_selector": args.window_selector,
        "window_count": args.window_count,
        "window_scan_stride": args.window_scan_stride,
        "agent_protocol": getattr(agent, "baseline_config", {}),
        "mean_metrics": metric_means(rows),
        "rows": rows,
        "update_logs": update_logs,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    train_summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("algo pool training complete")
    print(f"run_id: {run_id}")
    print(f"output_dir: {output_root}")
    print(f"latest_checkpoint_path: {latest_checkpoint_path}")
    print(f"train_csv_path: {train_csv_path}")
    print(f"summary_path: {summary_path}")


if __name__ == "__main__":
    main()
