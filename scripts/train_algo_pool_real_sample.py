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
from src.data.mobility.replay_provider import ReplayProvider
from src.envs.core.predictor_manager import PredictorManager
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.evaluators.main_results_support import (
    apply_frozen_window_plan,
    build_selected_workflow_states,
    build_episode_formal_request_exposure,
    clone_frames,
    clone_rsu_state,
    clone_workflow_state,
    load_window_bundle,
    resolve_window_candidates,
)
from src.evaluators.formal_window_consumption import (
    FormalWindowConsumptionError,
    load_contract as load_window_consumption_contract,
    validate_window_plan_binding,
)
from src.metrics.recorder import EpisodeRecorder
from src.metrics.cache_efficiency_metrics import reduce_cache_efficiency_summary
from src.runtime.typed_model_cache_runtime import (
    build_checkpoint_provenance,
    load_runtime_catalog,
    resolve_model_cache_runtime,
    sha256_value,
)
from src.runtime.formal_training_contract import (
    FormalTrainingContractError,
    audited_agent_config,
    checkpoint_schedule_metadata,
    load_json_mapping,
    resolve_training_contract,
    should_save_checkpoint,
    validate_resume_checkpoint_schedule,
)
from src.runtime.formal_training_identity import (
    FormalTrainingIdentityError,
    load_strict_json_mapping,
    validate_checkpoint_training_identity,
)
from src.runtime.resolved_formal_execution_context import (
    load_resolved_formal_execution_context,
    resolved_python_for_nested_consumer,
)
from src.runtime.portable_resource_identity import (
    add_portable_resource_arguments,
    resolve_argument_resources,
)
from src.runtime.formal_exogenous_request_execution import compute_formal_endpoint_metrics
from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer


TRAINABLE_BASELINES = list_trainable_agents()
REPLAY_BASELINES = {"dqn", "ddqn", "dueling_dqn", "dueling_ddqn", "qmix"}
PROFILE_DEFAULTS = {
    "smoke": {"episodes": 2, "update_every": 1, "max_steps": 6, "batch_size": 8},
    "baseline_safe": {"episodes": 12, "update_every": 3, "max_steps": 12, "batch_size": 24},
    "mappo_strong_audit": {"episodes": 96, "update_every": 6, "max_steps": 16, "batch_size": 32},
}
SUMMARY_METRICS = [
    "total_reward",
    "offset_adjusted_total_reward",
    "reward_positive_offset_component",
    "episode_step_count",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train direction-matched baseline agents")
    parser.add_argument("--agent_name", choices=TRAINABLE_BASELINES, default="ppo")
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="smoke")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--update_every", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=None)
    parser.add_argument("--value_coef", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--random_seed", type=int, default=7)
    parser.add_argument(
        "--model_cache_runtime_config",
        type=str,
        default="",
        help="Shared legacy/typed runtime YAML; typed fields may not be overridden on the CLI.",
    )
    parser.add_argument(
        "--formal_protocol_path",
        type=str,
        default="",
        help="Optional typed formal protocol v1.1 manifest; semantic CLI overrides are rejected.",
    )
    parser.add_argument(
        "--formal-exogenous-request-execution",
        action="store_true",
        help="Explicitly enable replay-driven request exposure (required by Protocol 2.x).",
    )
    parser.add_argument(
        "--agent_config_path",
        type=str,
        default="",
        help="Versioned per-agent training config companion; required with --formal_protocol_path.",
    )
    parser.add_argument(
        "--agent_scientific_config_path",
        type=str,
        default="",
        help="Protocol v1.6 scientific hyperparameter identity; execution-neutral.",
    )
    parser.add_argument(
        "--formal_training_execution_binding_path",
        type=str,
        default="",
        help="Protocol v1.6 run/commit/environment/command binding artifact.",
    )
    parser.add_argument(
        "--resolved_execution_context_path",
        type=str,
        default="",
        help="Protocol v1.6 immutable resolved execution context artifact.",
    )
    parser.add_argument(
        "--checkpoint_every_updates",
        type=int,
        default=None,
        help="Snapshot cadence. Legacy omission preserves one snapshot per update.",
    )
    parser.add_argument("--resume_checkpoint_path", type=str, default="")
    parser.add_argument("--resume_completed_episodes", type=int, default=0)
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
    parser.add_argument("--reward_positive_offset", type=float, default=5.0)
    parser.add_argument("--workflow_selector", type=str, default="ordered")
    parser.add_argument("--rsu_layout", type=str, default="auto_dominant_tight")
    parser.add_argument("--frame_offset", type=int, default=0)
    parser.add_argument("--window_length", type=int, default=24)
    parser.add_argument("--window_selector", type=str, default="max_handoff_candidate", choices=["ordered", "random", "max_handoff_candidate", "max_axis_crossing"])
    parser.add_argument("--window_count", type=int, default=1)
    parser.add_argument("--window_scan_stride", type=int, default=2)
    parser.add_argument("--window_mode", type=str, default="activating_only", choices=["activating_only", "mixed", "full", "mixed_informative", "full_stratified"])
    parser.add_argument("--window_plan_path", type=str, default="")
    parser.add_argument(
        "--formal_window_consumption_contract_path",
        type=str,
        default="",
        help="Frozen raw/provider window identity contract; required for formal training.",
    )
    parser.add_argument(
        "--formal_window_split",
        choices=["train", "dev", "formal", "sealed_holdout"],
        default="",
    )
    parser.add_argument(
        "--window_consumption_mode",
        choices=["formal", "rehearsal"],
        default="formal",
    )
    parser.add_argument("--prediction_horizon", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "artifacts" / "training" / "algo_pool"))
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help="Optional create-only stable run identity for protocol orchestration.",
    )
    parser.add_argument(
        "--formal_contract_preflight_only",
        action="store_true",
        help="Resolve Protocol v1.6 and instantiate/audit one agent, then exit before episode 0.",
    )
    add_portable_resource_arguments(parser)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def agent_profile_kwargs(agent_name: str, profile: str) -> dict[str, Any]:
    if agent_name != "mappo" or profile != "mappo_strong_audit":
        return {}
    return {
        "learning_rate": 2e-4,
        "clip_ratio": 0.16,
        "entropy_coef": 0.015,
        "value_coef": 0.65,
        "train_epochs": 8,
        "target_kl": 0.018,
        "kl_early_stop_enabled": True,
        "max_grad_norm": 0.7,
        "head_credit_enabled": True,
        "head_credit_protocol": "aggregation_reason_weighted_controller_ppo_v3",
        "slow_policy_credit_floor": 0.25,
        "fast_policy_credit_floor": 0.10,
        "event_policy_credit_floor": 0.12,
        "slow_entropy_coef_scale": 1.25,
        "fast_entropy_coef_scale": 1.00,
        "event_entropy_coef_scale": 1.35,
        "slow_entropy_credit_floor": 0.20,
        "fast_entropy_credit_floor": 0.08,
        "event_entropy_credit_floor": 0.12,
        "event_advantage_blend": 0.85,
    }


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
    step_trace = [step for step in summary.get("step_trace", []) if isinstance(step, dict)]
    reward_positive_offset = float(summary["run_info"].get("reward_positive_offset", 5.0) or 0.0)
    reward_positive_offset_component = sum(
        float(step.get("reward_dict", {}).get("positive_offset", 0.0) or 0.0)
        for step in step_trace
    )
    if reward_positive_offset_component <= 0.0 and reward_positive_offset > 0.0:
        reward_positive_offset_component = reward_positive_offset * float(len(step_trace))
    total_reward = float(summary["reward_breakdown"]["total"]["sum"])
    return {
        "episode_index": episode_index,
        "agent_name": summary["run_info"].get("agent_name"),
        "workflow_id": summary["run_info"].get("workflow_id"),
        "window_id": summary["run_info"].get("window_id"),
        "request_exposure_fingerprint": summary["run_info"].get(
            "request_exposure_fingerprint"
        ),
        "outcome_fingerprint": summary.get("formal_request_execution_audit", {}).get(
            "outcome_fingerprint"
        ),
        "primary_vehicle_selection": summary["run_info"].get("primary_vehicle_selection", "stable_first"),
        "reward_positive_offset": reward_positive_offset,
        "updated": bool(updated),
        "episode_success": bool(summary.get("episode_success", False)),
        "total_reward": total_reward,
        "offset_adjusted_total_reward": round(float(total_reward - reward_positive_offset_component), 6),
        "reward_positive_offset_component": round(float(reward_positive_offset_component), 6),
        "episode_step_count": len(step_trace),
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


def load_checkpoint_training_metadata(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise FormalTrainingContractError("resume checkpoint payload must be an object")
    metadata = payload.get("training_metadata") or payload.get("checkpoint_metadata")
    if not isinstance(metadata, dict):
        return {}
    return dict(metadata)


def main() -> None:
    args = parse_args()
    if args.resource_registry_path:
        resolve_argument_resources(
            args,
            bindings=(
                ("mobility_resource_id", "mobility_csv_path", "mobility_dataset"),
                ("workflow_resource_id", "workflow_csv_path", "workflow_dataset"),
                ("window_plan_resource_id", "window_plan_path", "window_plan"),
                ("runtime_config_resource_id", "model_cache_runtime_config", "runtime_config"),
            ),
        )
    formal_protocol = (
        load_json_mapping(args.formal_protocol_path, "formal protocol")
        if args.formal_protocol_path
        else None
    )
    if formal_protocol is not None:
        formal_version = str(
            formal_protocol.get("typed_model_cache_formal_protocol_version", "")
        )
        if formal_version.startswith("2.") and not args.formal_exogenous_request_execution:
            raise FormalTrainingContractError(
                "Protocol 2.x forbids endogenous request progression"
            )
    agent_config_companion = (
        load_json_mapping(args.agent_config_path, "agent config companion")
        if args.agent_config_path
        else None
    )
    scientific_config = (
        load_strict_json_mapping(
            args.agent_scientific_config_path, "agent scientific config"
        )
        if args.agent_scientific_config_path
        else None
    )
    execution_binding = (
        load_strict_json_mapping(
            args.formal_training_execution_binding_path,
            "formal training execution binding",
        )
        if args.formal_training_execution_binding_path
        else None
    )
    resolved_execution_context = None
    if args.resolved_execution_context_path:
        if formal_protocol is None:
            raise FormalTrainingContractError(
                "resolved execution context requires a formal protocol"
            )
        resolved_execution_context, _ = load_resolved_formal_execution_context(
            args.resolved_execution_context_path,
            protocol=formal_protocol,
            clean_worktree_root=ROOT_DIR,
            durable_run_root=Path(args.resolved_execution_context_path).resolve().parent,
            check_git=True,
        )
        resolved_python_for_nested_consumer(
            resolved_execution_context, observed_sys_executable=sys.executable
        )
    resolved_training = resolve_training_contract(
        agent_name=args.agent_name,
        profile_defaults=PROFILE_DEFAULTS[args.profile],
        cli_values={
            "episodes": args.episodes,
            "update_every": args.update_every,
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "checkpoint_every_updates": args.checkpoint_every_updates,
        },
        formal_protocol=formal_protocol,
        agent_config_companion=agent_config_companion,
        scientific_config=scientific_config,
        execution_binding=execution_binding,
        resolved_execution_context=resolved_execution_context,
    )
    args.episodes = resolved_training.episodes
    args.update_every = resolved_training.update_every
    args.batch_size = resolved_training.batch_size
    args.max_steps = resolved_training.max_steps
    args.checkpoint_every_updates = resolved_training.checkpoint_every_updates
    runtime_contract = resolve_model_cache_runtime(
        args.model_cache_runtime_config or None,
        root=ROOT_DIR,
    )
    window_consumption_contract: dict[str, Any] | None = None
    window_binding: dict[str, Any] | None = None
    if formal_protocol is not None and not args.formal_window_consumption_contract_path:
        raise FormalWindowConsumptionError(
            "formal training requires --formal_window_consumption_contract_path"
        )
    if args.formal_window_consumption_contract_path:
        if not args.window_plan_path or not args.formal_window_split:
            raise FormalWindowConsumptionError(
                "frozen-window consumption requires window plan and split"
            )
        if formal_protocol is not None and args.formal_window_split != "train":
            raise FormalWindowConsumptionError("formal training may consume only the train split")
        window_consumption_contract = load_window_consumption_contract(
            args.formal_window_consumption_contract_path
        )
        expected_contract_hash = (
            formal_protocol.get("execution_contract", {})
            .get("window_consumption_contract", {})
            .get("semantic_sha256")
            if formal_protocol is not None
            else None
        )
        if expected_contract_hash and expected_contract_hash != window_consumption_contract["hashes"]["semantic_sha256"]:
            raise FormalWindowConsumptionError(
                "formal protocol/window consumption contract hash mismatch"
            )
        window_binding = validate_window_plan_binding(
            contract=window_consumption_contract,
            plan_path=args.window_plan_path,
            split=args.formal_window_split,
            max_mobility_rows=args.max_mobility_rows,
            mobility_csv_path=args.mobility_csv_path,
            window_selector=args.window_selector,
            window_length=args.window_length,
            rsu_layout=args.rsu_layout,
            primary_vehicle_selection=args.primary_vehicle_selection,
            mode=("formal" if formal_protocol is not None else args.window_consumption_mode),
        )
    run_id = args.run_id or datetime.now().strftime(
        f"{args.agent_name}_train_%Y%m%d_%H%M%S_%f_seed{args.random_seed}"
    )
    output_root = Path(args.output_root) / args.agent_name / run_id
    episode_root = output_root / "episodes"
    checkpoint_root = output_root / "checkpoints"
    if output_root.exists() and any(output_root.iterdir()) and not args.resume_checkpoint_path:
        raise FileExistsError(f"refusing to overwrite non-empty training run: {output_root}")
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
    if args.window_plan_path:
        # A frozen protocol plan is already outcome-blind; avoid rescanning the raw trace.
        window_payload = apply_frozen_window_plan({}, args.window_plan_path)
    else:
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
    adapter_catalog = load_runtime_catalog(runtime_contract, root=ROOT_DIR)
    cache_capacity_profile = dict(runtime_contract["cache_capacity_profile"])
    if not cache_capacity_profile.get("enabled"):
        cache_capacity_profile = None
    window_plan_identity = {
        "path": str(window_payload.get("frozen_window_plan_path") or args.window_plan_path or "runtime_selected"),
        "protocol_version": str(window_payload.get("frozen_window_plan_protocol_version") or "runtime_selected_v1"),
        "split": str(window_payload.get("frozen_window_plan_split") or "non_formal_runtime_selected"),
        "selected_window_plan_sha256": sha256_value(selected_window_plan),
        "formal_window_consumption_contract_sha256": (
            window_consumption_contract["hashes"]["semantic_sha256"]
            if window_consumption_contract is not None
            else None
        ),
        "formal_window_split": args.formal_window_split or None,
    }
    agent_kwargs = {
        "random_seed": args.random_seed,
        "learning_rate": 3e-4 if args.learning_rate is None else args.learning_rate,
        "clip_ratio": args.clip_ratio,
        "entropy_coef": 0.01 if args.entropy_coef is None else args.entropy_coef,
        "value_coef": 0.5 if args.value_coef is None else args.value_coef,
        "batch_size": args.batch_size,
        "deterministic_action": False,
    }
    agent_kwargs.update(agent_profile_kwargs(args.agent_name, args.profile))
    for field_name, frozen_value in resolved_training.agent_config.items():
        supplied = getattr(args, field_name, None)
        if supplied is not None and supplied != frozen_value:
            raise FormalTrainingContractError(
                f"formal CLI/runtime mismatch for {field_name}: supplied={supplied}, frozen={frozen_value}"
            )
        agent_kwargs[field_name] = frozen_value
    if args.profile == "smoke" and args.agent_name in REPLAY_BASELINES:
        smoke_rollout_capacity = max(int(args.max_steps) * max(int(args.update_every), 1), 1)
        agent_kwargs["min_replay_size"] = max(1, min(int(args.batch_size), smoke_rollout_capacity))
    agent = build_agent(args.agent_name, **agent_kwargs)
    resolved_agent_config = audited_agent_config(
        agent, resolved_training.agent_config
    )

    if args.formal_contract_preflight_only:
        if resolved_training.formal_protocol_version not in {"1.6.0", "1.7.0", "1.8.0", "1.9.0", "2.0.0", "2.1.0"}:
            raise FormalTrainingContractError(
                "formal contract preflight is restricted to Protocol v1.6/v1.7"
            )
        payload = {
            "formal_training_contract_preflight_version": "1.0.0",
            "formal": False,
            "training": False,
            "performance_evidence": False,
            "checkpoint_created": False,
            "episode_count": 0,
            "agent_name": args.agent_name,
            "seed": args.random_seed,
            "resolved_agent_config": resolved_agent_config,
            "formal_training_contract": resolved_training.to_dict(),
        }
        target = output_root / "formal_contract_preflight.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        return

    resume_metadata: dict[str, Any] = {}
    if args.resume_checkpoint_path:
        resume_metadata = load_checkpoint_training_metadata(args.resume_checkpoint_path)
        validate_resume_checkpoint_schedule(
            resume_metadata,
            checkpoint_every_updates=args.checkpoint_every_updates,
        )
        if int(resume_metadata.get("update_count", 0) or 0) < 0:
            raise FormalTrainingContractError("resume checkpoint update_count is invalid")
        if resolved_training.formal_training_execution_binding_sha256:
            try:
                validate_checkpoint_training_identity(
                    resume_metadata,
                    scientific_config_sha256=str(
                        resolved_training.agent_scientific_config_semantic_sha256
                    ),
                    binding_sha256=str(
                        resolved_training.formal_training_execution_binding_sha256
                    ),
                    protocol_semantic_sha256=str(
                        resolved_training.formal_protocol_semantic_sha256
                    ),
                    execution_commit=str(resolved_training.execution_commit),
                    resolved_context_sha256=str(
                        resolved_training.resolved_execution_context_sha256
                    ),
                    formal_agent_order_contract_semantic_sha256=(
                        resolved_training.formal_agent_order_contract_semantic_sha256
                    ),
                    active_formal_bundle_sha256=(
                        resolved_training.active_formal_bundle_sha256
                    ),
                )
            except FormalTrainingIdentityError as exc:
                raise FormalTrainingContractError(str(exc)) from exc
        agent.load(str(Path(args.resume_checkpoint_path)))
    if args.resume_completed_episodes < 0 or args.resume_completed_episodes > args.episodes:
        raise FormalTrainingContractError(
            "resume_completed_episodes must be in [0, episodes]"
        )

    pending_rollout: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    update_logs: list[dict[str, Any]] = []
    latest_checkpoint_path = ""
    checkpoint_paths: list[str] = []
    saved_checkpoint_update_indices: list[int] = []
    update_index = int(resume_metadata.get("update_count", 0) or 0)
    for episode_index in range(args.resume_completed_episodes + 1, args.episodes + 1):
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
            formal_window_consumption_contract_path=args.formal_window_consumption_contract_path,
            formal_window_split=args.formal_window_split,
            expected_window_id=str(window_candidate.get("window_id", "")),
        )
        mobility_bundle.rsu_metadata["window_rank"] = window_candidate.get("window_rank")
        mobility_bundle.rsu_metadata["window_class"] = window_candidate.get("window_class")
        formal_request_exposure_trace = None
        if args.formal_exogenous_request_execution:
            evaluation_unit_id = (
                f"seed_{args.random_seed}/{mobility_bundle.rsu_metadata.get('window_id')}/"
                f"{workflow_state.workflow_id}"
            )
            formal_request_exposure_trace = build_episode_formal_request_exposure(
                workflow_state=workflow_state,
                mobility_bundle=mobility_bundle,
                adapter_catalog=adapter_catalog,
                max_steps=args.max_steps,
                mobility_source=args.mobility_source,
                primary_vehicle_selection=args.primary_vehicle_selection,
                cache_capacity_profile=cache_capacity_profile,
                evaluation_unit={
                    "evaluation_unit_id": evaluation_unit_id,
                    "benchmark_run_seed": args.random_seed,
                    "window_id": str(mobility_bundle.rsu_metadata.get("window_id")),
                    "workflow_id": workflow_state.workflow_id,
                    "raw_frame_interval": {
                        "start": int(window_candidate.get("frame_offset", args.frame_offset)),
                        "end": int(window_candidate.get("frame_offset", args.frame_offset))
                        + int(window_candidate.get("window_length", args.window_length))
                        - 1,
                    },
                },
                source_provenance={
                    "producer_consumer": "train_algo_pool_real_sample_pre_agent",
                    "phase": "train",
                    "runtime_contract_sha256": runtime_contract["runtime_contract_sha256"],
                    "window_plan_identity": window_plan_identity,
                    "formal_protocol_semantic_sha256": (
                        resolved_training.formal_protocol_semantic_sha256
                    ),
                },
            )
        recorder = EpisodeRecorder(prefetch_validation_window=6)
        core_env = VecWorkflowCoreEnv(
            mobility_provider=ReplayProvider(trajectory_frames=clone_frames(mobility_bundle.frames)),
            workflow_state=clone_workflow_state(workflow_state),
            adapter_catalog=adapter_catalog,
            rsu_states=[clone_rsu_state(rsu_state) for rsu_state in mobility_bundle.rsu_states],
            predictor_manager=PredictorManager(
                random_seed=args.random_seed + episode_index,
                horizon=args.prediction_horizon,
            ),
            max_steps=max(args.max_steps + 2, 8),
            mobility_source=args.mobility_source,
            primary_vehicle_selection=args.primary_vehicle_selection,
            reward_positive_offset=args.reward_positive_offset,
            cache_capacity_profile=cache_capacity_profile,
            formal_request_exposure_trace=formal_request_exposure_trace,
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
                "reward_positive_offset": args.reward_positive_offset,
                "prediction_horizon": args.prediction_horizon,
                "model_cache_profile": runtime_contract["model_cache_profile"],
                "runtime_contract_sha256": runtime_contract["runtime_contract_sha256"],
                "cache_event_schema_version": runtime_contract["cache_event_schema_version"],
                "cache_efficiency_metrics_contract_version": runtime_contract[
                    "cache_efficiency_metrics_contract_version"
                ],
                "evaluation_unit_id": (
                    formal_request_exposure_trace["evaluation_unit"]["evaluation_unit_id"]
                    if formal_request_exposure_trace is not None
                    else None
                ),
                "formal_exogenous_request_execution_contract_version": (
                    "1.0.0" if formal_request_exposure_trace is not None else None
                ),
                "request_exposure_fingerprint": (
                    formal_request_exposure_trace["request_exposure_fingerprint"]
                    if formal_request_exposure_trace is not None
                    else None
                ),
            }
        )
        summary["episode_success"] = bool(summary.get("episode_status", {}).get("completed", False))
        summary["cache_efficiency_metrics"] = reduce_cache_efficiency_summary(summary).to_dict()
        if formal_request_exposure_trace is not None:
            endpoint_metrics = compute_formal_endpoint_metrics(
                summary["cache_event_trace"],
                formal_request_exposure_trace,
                truncated=bool(summary.get("episode_status", {}).get("truncated", False)),
            )
            summary["formal_request_exposure"] = formal_request_exposure_trace
            summary["formal_request_execution_audit"] = endpoint_metrics
            summary["system_metrics"].update(
                {
                    key: endpoint_metrics[key]
                    for key in (
                        "full_service_ready_byte_hit_rate",
                        "joint_base_adapter_hit_rate",
                        "full_service_ready_request_rate",
                        "transfer_mb_per_request",
                        "workflow_continuity_rate",
                        "end_to_end_workflow_delay",
                    )
                }
            )
            summary["episode_success"] = bool(
                endpoint_metrics["workflow_completed_under_exogenous_execution"]
            )
        pending_rollout.extend(rollout)
        should_update = episode_index % max(args.update_every, 1) == 0 or episode_index == args.episodes
        if should_update:
            update_index += 1
            learn_info = agent.learn(pending_rollout)
            pending_rollout = []
            checkpoint_path = checkpoint_root / f"update_{update_index:04d}.pt"
            latest_path = checkpoint_root / "latest.pt"
            agent.save(str(latest_path))
            latest_checkpoint_path = str(latest_path)
            schedule_metadata = checkpoint_schedule_metadata(
                checkpoint_every_updates=args.checkpoint_every_updates,
                expected_update_count=resolved_training.expected_update_count,
            )
            checkpoint_metadata = {
                "run_id": run_id,
                "agent_name": args.agent_name,
                "config_profile": args.profile,
                "primary_vehicle_selection": args.primary_vehicle_selection,
                "reward_positive_offset": args.reward_positive_offset,
                "prediction_horizon": args.prediction_horizon,
                "episodes": args.episodes,
                "update_count": update_index,
                "checkpoint_schedule": schedule_metadata,
                "resolved_agent_config": resolved_agent_config,
                "formal_training_contract": resolved_training.to_dict(),
                "agent_scientific_config_semantic_sha256": (
                    resolved_training.agent_scientific_config_semantic_sha256
                ),
                "formal_training_execution_binding_sha256": (
                    resolved_training.formal_training_execution_binding_sha256
                ),
                "formal_protocol_semantic_sha256": (
                    resolved_training.formal_protocol_semantic_sha256
                ),
                "execution_commit": resolved_training.execution_commit,
                "resolved_execution_context_sha256": (
                    resolved_training.resolved_execution_context_sha256
                ),
                "environment_fingerprint": resolved_training.environment_fingerprint,
                "dependency_fingerprint": resolved_training.dependency_fingerprint,
                "environment_identity_projection_contract_version": (
                    resolved_training.environment_identity_projection_contract_version
                ),
                "full_normalized_environment_projection": (
                    resolved_training.full_normalized_environment_projection
                ),
                "formal_agent_order_contract_semantic_sha256": (
                    resolved_training.formal_agent_order_contract_semantic_sha256
                ),
                "active_formal_bundle_sha256": (
                    resolved_training.active_formal_bundle_sha256
                ),
                "formal_exogenous_request_execution_contract_version": (
                    "1.0.0" if args.formal_exogenous_request_execution else None
                ),
                "formal_request_exposure_trace_version": (
                    "1.0.0" if args.formal_exogenous_request_execution else None
                ),
                "request_execution_mode": (
                    "replay_driven_exogenous_request_exposure"
                    if args.formal_exogenous_request_execution
                    else "legacy_endogenous_progression"
                ),
                "is_smoke_checkpoint": args.profile == "smoke",
                "script": "scripts/train_algo_pool_real_sample.py",
                "typed_runtime_provenance": build_checkpoint_provenance(
                    root=ROOT_DIR,
                    agent_name=args.agent_name,
                    training_seed=args.random_seed,
                    runtime_contract=runtime_contract,
                    reward_positive_offset=args.reward_positive_offset,
                    train_window_plan_identity=window_plan_identity,
                ),
            }
            annotate_checkpoint(latest_path, checkpoint_metadata)
            if should_save_checkpoint(update_index, args.checkpoint_every_updates):
                agent.save(str(checkpoint_path))
                annotate_checkpoint(checkpoint_path, checkpoint_metadata)
                checkpoint_paths.append(str(checkpoint_path))
                saved_checkpoint_update_indices.append(update_index)
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
        "checkpoint_every_updates": args.checkpoint_every_updates,
        "checkpoint_schedule": checkpoint_schedule_metadata(
            checkpoint_every_updates=args.checkpoint_every_updates,
            expected_update_count=resolved_training.expected_update_count,
        ),
        "checkpoint_paths": checkpoint_paths,
        "saved_checkpoint_update_indices": saved_checkpoint_update_indices,
        "latest_checkpoint_path": latest_checkpoint_path,
        "output_dir": str(output_root),
        "train_csv_path": str(train_csv_path),
        "summary_json_path": str(summary_path),
        "workflow_ids": [workflow_state.workflow_id for workflow_state in workflow_states],
        "selected_window_plan": selected_window_plan,
        "frozen_window_plan_path": window_payload.get("frozen_window_plan_path", ""),
        "frozen_window_plan_protocol_version": window_payload.get("frozen_window_plan_protocol_version", ""),
        "frozen_window_plan_split": window_payload.get("frozen_window_plan_split", ""),
        "outcome_blind_window_selection": window_payload.get("outcome_blind_selection", False),
        "window_mode": args.window_mode,
        "primary_vehicle_selection": args.primary_vehicle_selection,
        "reward_protocol": {
            "reward_positive_offset": float(args.reward_positive_offset),
            "offset_free": abs(float(args.reward_positive_offset)) <= 1e-12,
            "offset_adjusted_total_reward_reported": True,
        },
        "window_selector": args.window_selector,
        "window_count": args.window_count,
        "window_scan_stride": args.window_scan_stride,
        "prediction_horizon": args.prediction_horizon,
        "agent_protocol": getattr(agent, "baseline_config", {}),
        "resolved_agent_config": resolved_agent_config,
        "formal_training_contract": resolved_training.to_dict(),
        "agent_scientific_config_semantic_sha256": (
            resolved_training.agent_scientific_config_semantic_sha256
        ),
        "formal_training_execution_binding_sha256": (
            resolved_training.formal_training_execution_binding_sha256
        ),
        "formal_protocol_semantic_sha256": (
            resolved_training.formal_protocol_semantic_sha256
        ),
        "execution_commit": resolved_training.execution_commit,
        "resolved_execution_context_sha256": (
            resolved_training.resolved_execution_context_sha256
        ),
        "environment_fingerprint": resolved_training.environment_fingerprint,
        "dependency_fingerprint": resolved_training.dependency_fingerprint,
        "environment_identity_projection_contract_version": (
            resolved_training.environment_identity_projection_contract_version
        ),
        "full_normalized_environment_projection": (
            resolved_training.full_normalized_environment_projection
        ),
        "formal_agent_order_contract_semantic_sha256": (
            resolved_training.formal_agent_order_contract_semantic_sha256
        ),
        "active_formal_bundle_sha256": (
            resolved_training.active_formal_bundle_sha256
        ),
        "formal_exogenous_request_execution_contract_version": (
            "1.0.0" if args.formal_exogenous_request_execution else None
        ),
        "formal_request_exposure_trace_version": (
            "1.0.0" if args.formal_exogenous_request_execution else None
        ),
        "request_execution_mode": (
            "replay_driven_exogenous_request_exposure"
            if args.formal_exogenous_request_execution
            else "legacy_endogenous_progression"
        ),
        "request_exposure_fingerprints": sorted(
            {
                str(row.get("request_exposure_fingerprint"))
                for row in rows
                if row.get("request_exposure_fingerprint")
            }
        ),
        "resolved_model_cache_runtime": runtime_contract,
        "runtime_contract_sha256": runtime_contract["runtime_contract_sha256"],
        "cache_capacity_profile": runtime_contract["cache_capacity_profile"],
        "train_window_plan_identity": window_plan_identity,
        "formal_window_consumption_binding": window_binding,
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
