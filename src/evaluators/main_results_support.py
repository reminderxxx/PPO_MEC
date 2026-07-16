"""Shared helpers for real-sample benchmark and evaluation."""

from __future__ import annotations

import csv
import json
import random
from copy import deepcopy
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import torch

from src.agents.registry import checkpoint_required_agents
from src.data.mobility.replay_provider import ReplayProvider
from src.data.mobility.rsu_mapper import RSUMapper
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.data.workflow.workflow_dataset_builder import WorkflowDatasetBuilder
from src.envs.core.predictor_manager import PredictorManager
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs import RSUState, VehicleState, WorkflowGraphState, WorkflowNode
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.evaluators.real_eval_support import build_inference_agent, ensure_agent_checkpoint_path
from src.evaluators.real_sample_support import RealMobilityBundle, load_real_mobility_bundle, load_real_source_frames, scan_mobility_windows
from src.metrics.recorder import EpisodeRecorder
from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer


PAPER_PROTOCOL_VERSION = "paper_protocol_v1_20260409"
PAPER_PROTOCOL_FROZEN = True
SA_GHMAPPO_V11_REWARD_PROFILE = "top_journal_mechanism_v11_mappo_reward"

MAIN_RESULT_METRICS = [
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
    "validated_predictive_prefetch_count",
    "migration_during_handoff_count",
    "handoff_ready_count",
    "handoff_total_count",
    "successful_episode_rate",
    "mechanism_realization_rate",
    "continuity_guard_trigger_count",
    "continuity_guard_trigger_rate",
    "target_mismatch_guard_count",
    "guard_prefetch_to_prepare_count",
    "guard_hard_override_count",
    "action_projection_count",
    "action_projection_rate",
    "invalid_action_attempt_count",
    "invalid_action_attempt_rate",
    "guard_action_delta_count",
    "guard_action_delta_rate",
    "dag_frontier_size_mean",
    "dag_critical_path_pressure_mean",
    "dag_current_node_dependency_pressure_mean",
    "dag_remaining_nodes_mean",
    "backhaul_guard_count",
    "backhaul_guard_rate",
    "cache_warm_start_guard_count",
    "cache_warm_start_guard_rate",
    "predictive_prefetch_admission_guard_count",
    "predictive_prefetch_admission_guard_rate",
]
ACTIONMIX_DIAGNOSTIC_METRICS = [
    "service_success_count",
    "service_delay_sum",
    "service_wait_sum",
    "service_restart_count",
    "workflow_completed_count",
    "workflow_unfinished_count",
    "adapter_hit_count",
    "adapter_miss_count",
    "adapter_warm_hit_count",
    "adapter_cold_start_count",
    "cache_eviction_count",
    "cache_admission_count",
    "cache_admission_added_new_adapter_count",
    "cache_noop_count",
    "cache_capacity_enabled",
    "rsu_adapter_slots",
    "cache_capacity",
    "cache_used_size",
    "cache_remaining_size",
    "cache_occupancy_rate",
    "eviction_count",
    "evicted_adapter_count",
    "local_exec_count",
    "current_rsu_exec_count",
    "next_rsu_exec_count",
    "neighbor_rsu_exec_count",
    "cloud_exec_count",
    "prefetch_action_count",
    "migration_action_count",
    "no_op_action_count",
    "prefetch_attempt_count",
    "prefetch_success_count",
    "prefetch_failed_count",
    "migration_attempt_count",
    "migration_success_count",
    "migration_failed_count",
    "mechanism_attempt_count",
    "mechanism_validated_success_count",
    "mechanism_pending_success_count",
    "mechanism_success_rate",
    "env_invalid_action_count",
    "migration_overhead_sum",
    "delay_reward_component",
    "cache_reward_component",
    "handoff_reward_component",
    "mechanism_shaping_reward_component",
    "backhaul_reward_component",
    "failure_reward_component",
    "continuity_reward_component",
    "service_reward_component",
    "mechanism_exploration_reward_component",
    "constraint_penalty_sum",
    "migration_cost_sum",
    "cache_miss_penalty_sum",
    "delay_penalty_sum",
]
MAIN_RESULT_METRICS.extend(ACTIONMIX_DIAGNOSTIC_METRICS)
MECHANISM_DIAG_FIELDS = [
    "predictive_prefetch_request_count",
    "prefetch_validated_hit_count",
    "prefetch_expired_miss_count",
    "migration_prepare_count",
    "migration_during_handoff_count",
    "handoff_ready_count",
]
LOWER_IS_BETTER_METRICS = {
    "end_to_end_workflow_delay",
    "handoff_failure_rate",
    "cross_rsu_cold_start_frequency",
    "backhaul_traffic_cost",
    "adapter_state_migration_overhead",
    "service_delay_sum",
    "service_wait_sum",
    "service_restart_count",
    "workflow_unfinished_count",
    "adapter_miss_count",
    "adapter_cold_start_count",
    "cache_eviction_count",
    "eviction_count",
    "evicted_adapter_count",
    "cache_noop_count",
    "prefetch_failed_count",
    "migration_failed_count",
    "migration_overhead_sum",
}
CHECKPOINT_REQUIRED_AGENTS = checkpoint_required_agents()



def classify_experiment_scale(config_profile: str, episodes: int, update_count: int) -> dict[str, Any]:
    profile = str(config_profile or "unknown")
    episodes = int(episodes or 0)
    update_count = int(update_count or 0)
    if profile == "smoke":
        return {
            "experiment_run_type": "debug",
            "paper_claim_ready": False,
            "scope_note": "smoke/debug run; only validates the execution path.",
        }
    if profile in {"formal_main", "formal_main_stable", "formal_main_baseline", "formal_main_stable_baseline"}:
        scope_note = "formal_main profile; still verify seeds, windows, and checkpoint audit before paper use."
        if profile == "formal_main_stable":
            scope_note = "formal_main_stable profile with mechanism anti-collapse controls."
        if profile == "formal_main_baseline":
            scope_note = "formal_main_baseline profile for PPO/flat baseline comparison."
        if profile == "formal_main_stable_baseline":
            scope_note = "formal_main_stable_baseline profile for stable PPO baseline comparison."
        return {
            "experiment_run_type": "formal",
            "paper_claim_ready": True,
            "scope_note": scope_note,
        }
    if profile in {"sa_advantage_round1", "sa_mechanism_policy_round2", "sa_mechanism_retention_round3", "top_journal_mechanism_v1", "sa_reward_tiebreak_round4"}:
        return {
            "experiment_run_type": "formal",
            "paper_claim_ready": False,
            "scope_note": f"{profile} formal experiment profile; benchmark outputs decide freeze status.",
        }
    if profile == "baseline_safe":
        return {
            "experiment_run_type": "baseline",
            "paper_claim_ready": False,
            "scope_note": "non-smoke baseline run; requires full audit before paper claims.",
        }
    if episodes <= 4 or update_count <= 1:
        return {
            "experiment_run_type": "debug",
            "paper_claim_ready": False,
            "scope_note": "short run with limited episodes or updates; not paper-claim ready.",
        }
    return {
        "experiment_run_type": "baseline",
        "paper_claim_ready": False,
        "scope_note": "custom baseline profile; validate protocol before paper use.",
    }
ABLATION_CONTRIBUTION_MAP = {
    "sa_ghmappo_full": {
        "removed_module": "none",
        "paper_contribution": "??????surrogate-assisted prediction + DAG-aware graph encoder + hierarchical multi-timescale control",
    },
    "no_prediction": {
        "removed_module": "surrogate-assisted prediction",
        "paper_contribution": "?? surrogate / prediction usage???????????????????????",
    },
    "no_graph_encoder": {
        "removed_module": "DAG-aware graph encoding",
        "paper_contribution": "???????????? DAG ????????????????",
    },
    "no_hierarchy": {
        "removed_module": "multi-timescale hierarchy",
        "paper_contribution": "?? slow/fast/event ??????????????????",
    },
    "no_event_agent": {
        "removed_module": "handoff-aware event controller",
        "paper_contribution": "?????? handoff migration controller??? continuity protection ???",
    },
    "no_adapter_prefetch": {
        "removed_module": "proactive adapter prefetch",
        "paper_contribution": "?? proactive cache placement??? predictive prefetch ? warm hit ????????",
    },
    "no_dag_dependency_aware": {
        "removed_module": "dependency-aware DAG modeling",
        "paper_contribution": "?? DAG ??????? dependency-aware workflow modeling ????",
    },
    "no_uncertainty_signal": {
        "removed_module": "prediction confidence / uncertainty gating",
        "paper_contribution": "?? prediction confidence/uncertainty??? surrogate reliability gating ???",
    },
}


def expand_checkpoint_aliases(checkpoint_map: dict[str, str]) -> dict[str, str]:
    """Keep current PPO/MAPPO names compatible with historical flat_* keys."""
    expanded = dict(checkpoint_map)
    for primary_name, legacy_name in [("ppo", "flat_ppo"), ("mappo", "flat_mappo")]:
        primary_path = expanded.get(primary_name, "")
        legacy_path = expanded.get(legacy_name, "")
        if primary_path and not legacy_path:
            expanded[legacy_name] = primary_path
        elif legacy_path and not primary_path:
            expanded[primary_name] = legacy_path
    return expanded


def load_seed_checkpoint_manifest(path: str | Path) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"seed checkpoint manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"seed checkpoint manifest must be a mapping: {manifest_path}")
    normalized: dict[str, dict[str, str]] = {}
    for agent_name, seed_map in payload.items():
        if not isinstance(seed_map, dict):
            continue
        normalized[str(agent_name)] = {
            str(seed): str(checkpoint_path)
            for seed, checkpoint_path in seed_map.items()
            if str(checkpoint_path)
        }
    for primary_name, legacy_name in [("ppo", "flat_ppo"), ("mappo", "flat_mappo")]:
        if primary_name in normalized and legacy_name not in normalized:
            normalized[legacy_name] = dict(normalized[primary_name])
        elif legacy_name in normalized and primary_name not in normalized:
            normalized[primary_name] = dict(normalized[legacy_name])
    return normalized


def checkpoint_map_for_seed(
    base_checkpoint_map: dict[str, str],
    seed_checkpoint_manifest: dict[str, dict[str, str]],
    seed: int,
) -> dict[str, str]:
    checkpoint_map = dict(base_checkpoint_map)
    seed_key = str(seed)
    for agent_name, seed_map in seed_checkpoint_manifest.items():
        if seed_key in seed_map:
            checkpoint_map[agent_name] = seed_map[seed_key]
    return expand_checkpoint_aliases(checkpoint_map)


def representative_checkpoint_map(
    base_checkpoint_map: dict[str, str],
    seed_checkpoint_manifest: dict[str, dict[str, str]],
    seeds: list[int],
) -> dict[str, str]:
    checkpoint_map = dict(base_checkpoint_map)
    for agent_name, seed_map in seed_checkpoint_manifest.items():
        for seed in seeds:
            checkpoint_path = seed_map.get(str(seed), "")
            if checkpoint_path:
                checkpoint_map[agent_name] = checkpoint_path
                break
    return expand_checkpoint_aliases(checkpoint_map)


def build_selected_workflow_states(
    workflow_csv_path: str | Path,
    max_workflows: int,
    workflow_selector: str,
    min_tasks: int,
    max_tasks: int,
    random_seed: int,
) -> list[Any]:
    workflow_csv_path = Path(workflow_csv_path)
    if not workflow_csv_path.exists():
        raise FileNotFoundError(
            f"Alibaba batch_task.csv ???: {workflow_csv_path}???? data/raw/workflow/alibaba2018/ ?????"
        )
    builder = WorkflowDatasetBuilder()
    return builder.build_selected_alibaba_workflow_states(
        csv_path=workflow_csv_path,
        max_workflows=max_workflows,
        workflow_selector=workflow_selector,
        min_tasks=min_tasks,
        max_tasks=max_tasks,
        random_seed=random_seed,
    )


def clone_vehicle(vehicle: VehicleState) -> VehicleState:
    return VehicleState(**vehicle.to_dict())


def clone_rsu_state(rsu_state: RSUState) -> RSUState:
    return RSUState(**rsu_state.to_dict())


def clone_workflow_state(workflow_state: WorkflowGraphState) -> WorkflowGraphState:
    return WorkflowGraphState(
        workflow_id=workflow_state.workflow_id,
        nodes=[WorkflowNode(**node.to_dict()) for node in workflow_state.nodes],
        edges=[tuple(edge) for edge in workflow_state.edges],
        execution_order=list(workflow_state.execution_order),
        completed_node_ids=list(workflow_state.completed_node_ids),
        current_node_id=workflow_state.current_node_id,
        is_completed=bool(workflow_state.is_completed),
    )


def clone_adapter_catalog(adapter_catalog: AdapterCatalog) -> AdapterCatalog:
    return AdapterCatalog.from_dict(adapter_catalog.to_dict())


def clone_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cloned: list[dict[str, Any]] = []
    for frame in frames:
        cloned.append(
            {
                "time_index": int(frame["time_index"]),
                "vehicles": [clone_vehicle(vehicle) for vehicle in frame.get("vehicles", [])],
            }
        )
    return cloned


def clone_mobility_bundle(bundle: RealMobilityBundle) -> RealMobilityBundle:
    frames = clone_frames(bundle.frames)
    rsu_states = [clone_rsu_state(rsu_state) for rsu_state in bundle.rsu_states]
    return RealMobilityBundle(
        provider=ReplayProvider(trajectory_frames=frames),
        frames=frames,
        rsu_states=rsu_states,
        rsu_metadata=dict(bundle.rsu_metadata),
        source_path=bundle.source_path,
    )


def compute_workflow_width(workflow_state: WorkflowGraphState) -> int:
    node_ids = [node.node_id for node in workflow_state.nodes]
    indegree = {node_id: 0 for node_id in node_ids}
    children: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for src, dst in workflow_state.edges:
        if src in indegree and dst in indegree:
            indegree[dst] += 1
            children[src].append(dst)
    frontier = sorted([node_id for node_id, degree in indegree.items() if degree == 0])
    if not frontier:
        return max(1, len(node_ids))
    max_width = len(frontier)
    remaining_indegree = dict(indegree)
    while frontier:
        next_frontier: list[str] = []
        for node_id in frontier:
            for child_id in children.get(node_id, []):
                remaining_indegree[child_id] -= 1
                if remaining_indegree[child_id] == 0:
                    next_frontier.append(child_id)
        frontier = sorted(next_frontier)
        if frontier:
            max_width = max(max_width, len(frontier))
    return max_width


def compute_critical_path_length(workflow_state: WorkflowGraphState) -> int:
    node_ids = [node.node_id for node in workflow_state.nodes]
    parents: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    children: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for src, dst in workflow_state.edges:
        if src in indegree and dst in indegree:
            indegree[dst] += 1
            parents[dst].append(src)
            children[src].append(dst)
    frontier = [node_id for node_id, degree in indegree.items() if degree == 0]
    if not frontier:
        return max(1, len(node_ids))
    topo_order: list[str] = []
    remaining_indegree = dict(indegree)
    while frontier:
        node_id = frontier.pop(0)
        topo_order.append(node_id)
        for child_id in children.get(node_id, []):
            remaining_indegree[child_id] -= 1
            if remaining_indegree[child_id] == 0:
                frontier.append(child_id)
    longest = {node_id: 1 for node_id in node_ids}
    for node_id in topo_order:
        parent_lengths = [longest[parent_id] for parent_id in parents.get(node_id, [])]
        if parent_lengths:
            longest[node_id] = max(parent_lengths) + 1
    return max(longest.values()) if longest else 1


def _bucket_match(value: int, bucket: str, dimension: str = "generic") -> bool:
    normalized = (bucket or "all").strip().lower()
    if normalized in {"", "all", "full"}:
        return True
    if normalized.startswith("gte:"):
        return value >= int(normalized.split(":", 1)[1])
    if normalized.startswith("lte:"):
        return value <= int(normalized.split(":", 1)[1])
    if "-" in normalized and normalized.replace("-", "").isdigit():
        lower_str, upper_str = normalized.split("-", 1)
        return int(lower_str) <= value <= int(upper_str)
    if dimension == "width":
        presets = {
            "narrow": (0, 2),
            "medium": (3, 4),
            "wide": (5, 9999),
        }
    else:
        presets = {
            "small": (0, 8),
            "medium": (9, 16),
            "large": (17, 9999),
        }
    if normalized in presets:
        lower, upper = presets[normalized]
        return lower <= value <= upper
    raise ValueError(f"?? bucket ??: {bucket}")


def filter_workflow_states_by_buckets(
    workflow_states: list[WorkflowGraphState],
    workflow_node_count_bucket: str = "all",
    critical_path_bucket: str = "all",
    workflow_width_bucket: str = "all",
) -> list[WorkflowGraphState]:
    filtered: list[WorkflowGraphState] = []
    for workflow_state in workflow_states:
        node_count = len(workflow_state.nodes)
        critical_path = compute_critical_path_length(workflow_state)
        workflow_width = compute_workflow_width(workflow_state)
        if not _bucket_match(node_count, workflow_node_count_bucket, dimension="node_count"):
            continue
        if not _bucket_match(critical_path, critical_path_bucket, dimension="critical_path"):
            continue
        if not _bucket_match(workflow_width, workflow_width_bucket, dimension="width"):
            continue
        filtered.append(workflow_state)
    return filtered



def apply_dependency_simplification(
    workflow_state: WorkflowGraphState,
    drop_rate: float,
    random_seed: int,
) -> WorkflowGraphState:
    if drop_rate <= 0.0 or not workflow_state.edges:
        return clone_workflow_state(workflow_state)
    rng = random.Random(random_seed)
    cloned = clone_workflow_state(workflow_state)
    kept_edges: list[tuple[str, str]] = []
    incoming_count: dict[str, int] = {}
    for _, dst in cloned.edges:
        incoming_count[dst] = incoming_count.get(dst, 0) + 1
    for src, dst in cloned.edges:
        if incoming_count.get(dst, 0) <= 1:
            kept_edges.append((src, dst))
            continue
        if rng.random() < drop_rate:
            incoming_count[dst] -= 1
            continue
        kept_edges.append((src, dst))
    node_map = {node.node_id: node for node in cloned.nodes}
    for node in cloned.nodes:
        node.predecessors = []
        node.successors = []
    for src, dst in kept_edges:
        if src in node_map and dst in node_map:
            node_map[src].successors.append(dst)
            node_map[dst].predecessors.append(src)
    cloned.edges = kept_edges
    return cloned


def apply_adapter_capacity_scale(
    adapter_catalog: AdapterCatalog,
    rsu_states: list[RSUState],
    adapter_capacity_scale: float,
) -> tuple[AdapterCatalog, list[RSUState], dict[str, Any]]:
    if adapter_capacity_scale <= 0.0:
        adapter_capacity_scale = 0.1
    catalog = clone_adapter_catalog(adapter_catalog)
    rsu_clones = [clone_rsu_state(rsu_state) for rsu_state in rsu_states]
    all_adapters = sorted(
        {
            adapter_id
            for cache_profile in catalog.rsu_adapter_caches
            for adapter_id in cache_profile.cached_adapter_ids
        }
    )
    cache_plan: dict[str, list[str]] = {}
    for profile in catalog.rsu_adapter_caches:
        original_ids = list(profile.cached_adapter_ids)
        if adapter_capacity_scale >= 1.0:
            target_count = min(len(all_adapters), max(len(original_ids), int(round(len(original_ids) * adapter_capacity_scale))))
            expanded = list(dict.fromkeys(original_ids + [adapter_id for adapter_id in all_adapters if adapter_id not in original_ids]))
            cache_plan[profile.rsu_id] = expanded[:target_count]
        else:
            target_count = max(1, int(round(max(len(original_ids), 1) * adapter_capacity_scale)))
            cache_plan[profile.rsu_id] = original_ids[:target_count]
    scaled_catalog = catalog.clone_with_cache_plan(cache_plan)
    for rsu_state in rsu_clones:
        rsu_state.cached_adapter_ids = list(cache_plan.get(rsu_state.rsu_id, []))
    return scaled_catalog, rsu_clones, {
        "adapter_capacity_scale": round(float(adapter_capacity_scale), 6),
        "proxy_scaling": True,
        "proxy_note": "adapter capacity scale ??????????? cached adapters ? proxy scaling?",
    }


def apply_adapter_type_proxy(
    workflow_state: WorkflowGraphState,
    adapter_catalog: AdapterCatalog,
    adapter_type_count: int,
) -> tuple[WorkflowGraphState, AdapterCatalog, dict[str, Any]]:
    cloned_workflow = clone_workflow_state(workflow_state)
    if adapter_type_count <= 0:
        return cloned_workflow, clone_adapter_catalog(adapter_catalog), {
            "adapter_type_count": 0,
            "proxy_scaling": False,
            "proxy_note": "adapter_type_count ????",
        }

    catalog_dict = clone_adapter_catalog(adapter_catalog).to_dict()
    base_adapter_ids = sorted(
        {
            node.required_adapter for node in cloned_workflow.nodes
        }
        | {
            cache_object["adapter_id"] for cache_object in catalog_dict.get("cache_objects", [])
        }
        | {
            bundle["adapter_id"] for bundle in catalog_dict.get("adapter_state_bundles", [])
        }
        | {
            adapter_id
            for cache_profile in catalog_dict.get("rsu_adapter_caches", [])
            for adapter_id in cache_profile.get("cached_adapter_ids", [])
        }
    )
    if not base_adapter_ids:
        base_adapter_ids = ["adapter_proxy_seed"]

    target_adapter_ids: list[str] = []
    for index in range(adapter_type_count):
        if index < len(base_adapter_ids):
            target_adapter_ids.append(base_adapter_ids[index])
        else:
            target_adapter_ids.append(f"adapter_proxy_{index + 1:02d}")

    for node_index, node in enumerate(cloned_workflow.nodes):
        node.required_adapter = target_adapter_ids[node_index % len(target_adapter_ids)]

    rsu_profiles = catalog_dict.get("rsu_adapter_caches", [])
    if not rsu_profiles:
        rsu_profiles = [{"rsu_id": "rsu_a", "cached_adapter_ids": []}]
    for index, cache_profile in enumerate(rsu_profiles):
        cache_profile["cached_adapter_ids"] = [
            adapter_id
            for adapter_position, adapter_id in enumerate(target_adapter_ids)
            if adapter_position % len(rsu_profiles) == index % len(rsu_profiles)
        ]

    template_object = catalog_dict.get("cache_objects", [])[:1] or [{
        "object_id": "cache_obj_template",
        "adapter_id": "adapter_template",
        "size_mb": 96.0,
        "source": "proxy_cache",
    }]
    template_bundle = catalog_dict.get("adapter_state_bundles", [])[:1] or [{
        "bundle_id": "bundle_template",
        "adapter_id": "adapter_template",
        "state_version": "v1",
        "continuity_token": "token_template",
        "serialized_state_ref": "cache://bundle_template",
    }]
    catalog_dict["cache_objects"] = [
        {
            **deepcopy(template_object[0]),
            "object_id": f"cache_obj_{adapter_id}",
            "adapter_id": adapter_id,
        }
        for adapter_id in target_adapter_ids
    ]
    catalog_dict["adapter_state_bundles"] = [
        {
            **deepcopy(template_bundle[0]),
            "bundle_id": f"bundle_{adapter_id}",
            "adapter_id": adapter_id,
            "continuity_token": f"token_{adapter_id}",
            "serialized_state_ref": f"cache://bundle_{adapter_id}",
        }
        for adapter_id in target_adapter_ids
    ]

    return cloned_workflow, AdapterCatalog.from_dict(catalog_dict), {
        "adapter_type_count": len(target_adapter_ids),
        "proxy_scaling": True,
        "proxy_note": "adapter_type_count ???? workflow adapter remap + catalog clone ? proxy scaling?",
    }


def subsample_mobility_bundle_by_vehicle_count(
    mobility_bundle: RealMobilityBundle,
    vehicle_sample_count: int,
) -> tuple[RealMobilityBundle, dict[str, Any]]:
    if vehicle_sample_count <= 0:
        return clone_mobility_bundle(mobility_bundle), {
            "vehicle_sample_count": 0,
            "proxy_scaling": False,
            "proxy_note": "vehicle_sample_count ????",
        }
    frequency: dict[str, int] = {}
    for frame in mobility_bundle.frames:
        for vehicle in frame.get("vehicles", []):
            frequency[vehicle.vehicle_id] = frequency.get(vehicle.vehicle_id, 0) + 1
    selected_vehicle_ids = {
        vehicle_id
        for vehicle_id, _ in sorted(frequency.items(), key=lambda item: (-item[1], item[0]))[:vehicle_sample_count]
    }
    sampled_frames: list[dict[str, Any]] = []
    for frame in mobility_bundle.frames:
        sampled_frames.append(
            {
                "time_index": int(frame["time_index"]),
                "vehicles": [clone_vehicle(vehicle) for vehicle in frame.get("vehicles", []) if vehicle.vehicle_id in selected_vehicle_ids],
            }
        )
    sampled_bundle = RealMobilityBundle(
        provider=ReplayProvider(trajectory_frames=sampled_frames),
        frames=sampled_frames,
        rsu_states=[clone_rsu_state(rsu_state) for rsu_state in mobility_bundle.rsu_states],
        rsu_metadata={**mobility_bundle.rsu_metadata, "vehicle_sample_count": len(selected_vehicle_ids)},
        source_path=mobility_bundle.source_path,
    )
    return sampled_bundle, {
        "vehicle_sample_count": len(selected_vehicle_ids),
        "proxy_scaling": True,
        "proxy_note": "vehicle_sample_count ?????????????????? proxy scaling?",
    }


def _estimate_prediction_activity(window_frames: list[dict[str, Any]], rsu_states: list[RSUState]) -> dict[str, float]:
    if not window_frames:
        return {
            "predicted_next_rsu_non_null_ratio": 0.0,
            "predicted_handoff_target_non_null_ratio": 0.0,
        }
    predictor = PredictorManager()
    predictor.reset()
    mapper = RSUMapper(rsu_states)
    total_vehicle_observations = 0
    predicted_next_non_null = 0
    predicted_handoff_non_null = 0
    for frame in window_frames:
        vehicles = frame.get("vehicles", [])
        current_associations = mapper.associate(vehicles)
        sequences = predictor.predict_next_rsu_sequence(vehicles, rsu_states)
        for vehicle in vehicles:
            total_vehicle_observations += 1
            sequence = list(sequences.get(vehicle.vehicle_id, []))
            if sequence and sequence[0] is not None:
                predicted_next_non_null += 1
            predicted_handoff_target = predictor._extract_first_handoff_rsu(
                current_rsu_id=current_associations.get(vehicle.vehicle_id),
                sequence=sequence,
            )
            if predicted_handoff_target is not None:
                predicted_handoff_non_null += 1
        predictor._update_last_positions(vehicles)
    if total_vehicle_observations <= 0:
        return {
            "predicted_next_rsu_non_null_ratio": 0.0,
            "predicted_handoff_target_non_null_ratio": 0.0,
        }
    return {
        "predicted_next_rsu_non_null_ratio": round(predicted_next_non_null / total_vehicle_observations, 6),
        "predicted_handoff_target_non_null_ratio": round(predicted_handoff_non_null / total_vehicle_observations, 6),
    }


def _build_window_rsus(window_frames: list[dict[str, Any]], recommended_layout: str) -> tuple[list[RSUState], dict[str, Any]]:
    from src.evaluators.real_sample_support import build_sample_rsus

    return build_sample_rsus(frames=window_frames, rsu_layout=recommended_layout)


def resolve_window_candidates(
    root_dir: Path,
    mobility_csv_path: str,
    max_mobility_rows: int,
    rsu_layout: str,
    frame_offset: int,
    window_length: int,
    window_selector: str,
    window_count: int,
    window_scan_stride: int,
    random_seed: int,
    mobility_source: str = "ngsim",
    lust_scenario_root: str = "",
    window_mode: str = "mixed",
    activating_handoff_threshold: int = 2,
    activating_vehicle_threshold: float = 2.0,
    activating_predicted_next_ratio_threshold: float = 0.3,
    activating_handoff_prediction_ratio_threshold: float = 0.15,
    non_mechanism_handoff_max: int = 0,
    non_mechanism_prediction_ratio_max: float = 0.05,
    active_non_mechanism_vehicle_threshold: float = 2.0,
    active_non_mechanism_association_change_min: int = 1,
    active_non_mechanism_handoff_max: int = 1,
    active_non_mechanism_predicted_next_ratio_max: float = 0.2,
    active_non_mechanism_handoff_prediction_ratio_max: float = 0.1,
    idle_or_sparse_vehicle_max: float = 1.5,
    idle_or_sparse_association_change_max: int = 0,
    window_rank_offset: int = 0,
    excluded_window_intervals: list[tuple[int, int]] | None = None,
    holdout_min_gap_frames: int = 0,
    enforce_non_overlapping_selection: bool = False,
) -> tuple[str, dict[str, Any]]:
    raw_frames, source_path = load_real_source_frames(
        root_dir=root_dir,
        mobility_source=mobility_source,
        mobility_csv_path=mobility_csv_path,
        lust_scenario_root=lust_scenario_root,
        max_mobility_rows=max_mobility_rows,
    )
    ranking_mode = "max_handoff_candidate" if window_selector in {"ordered", "random"} else window_selector
    scan_results = scan_mobility_windows(
        frames=raw_frames,
        layout_candidates=[rsu_layout],
        frame_offset=frame_offset,
        window_length=window_length,
        stride=max(1, window_scan_stride),
        ranking_mode=ranking_mode,
    )
    if not scan_results:
        raise RuntimeError("??? benchmark ??? mobility windows?")

    enriched_windows: list[dict[str, Any]] = []
    effective_activating_handoff_threshold = int(activating_handoff_threshold)
    effective_activating_vehicle_threshold = float(activating_vehicle_threshold)
    effective_activating_predicted_next_ratio_threshold = float(activating_predicted_next_ratio_threshold)
    effective_activating_handoff_prediction_ratio_threshold = float(activating_handoff_prediction_ratio_threshold)
    if mobility_source == "lust":
        effective_activating_handoff_threshold = 1
        effective_activating_vehicle_threshold = min(effective_activating_vehicle_threshold, 1.0)
        effective_activating_predicted_next_ratio_threshold = 0.0
        effective_activating_handoff_prediction_ratio_threshold = 0.0
    for index, item in enumerate(scan_results):
        window_frames = raw_frames[int(item["frame_offset"]): int(item["frame_offset"]) + int(item["window_length"])]
        rsu_states, _ = _build_window_rsus(window_frames=window_frames, recommended_layout=str(item["recommended_rsu_layout"]))
        prediction_activity = _estimate_prediction_activity(window_frames=window_frames, rsu_states=rsu_states)
        estimated_handoff_count = int(item["estimated_handoff_count"])
        estimated_association_change_count = int(item["estimated_association_change_count"])
        active_vehicle_count_mean = float(item["active_vehicle_count_mean"])
        predicted_next_ratio = float(prediction_activity["predicted_next_rsu_non_null_ratio"])
        predicted_handoff_ratio = float(prediction_activity["predicted_handoff_target_non_null_ratio"])

        is_activating = (
            estimated_handoff_count >= effective_activating_handoff_threshold
            and active_vehicle_count_mean >= effective_activating_vehicle_threshold
            and predicted_next_ratio >= effective_activating_predicted_next_ratio_threshold
            and predicted_handoff_ratio >= effective_activating_handoff_prediction_ratio_threshold
        )
        is_active_non_mechanism = (
            not is_activating
            and active_vehicle_count_mean >= float(active_non_mechanism_vehicle_threshold)
            and estimated_association_change_count >= int(active_non_mechanism_association_change_min)
            and estimated_handoff_count <= int(active_non_mechanism_handoff_max)
            and predicted_handoff_ratio <= float(active_non_mechanism_handoff_prediction_ratio_max)
        )
        is_idle_or_sparse = (
            active_vehicle_count_mean <= float(idle_or_sparse_vehicle_max)
            or (
                estimated_association_change_count <= int(idle_or_sparse_association_change_max)
                and estimated_handoff_count <= int(non_mechanism_handoff_max)
                and predicted_handoff_ratio <= float(non_mechanism_prediction_ratio_max)
            )
        )
        if is_activating:
            window_class = "mechanism_activating"
        elif is_active_non_mechanism:
            window_class = "active_non_mechanism"
        elif is_idle_or_sparse:
            window_class = "idle_or_sparse"
        elif (
            active_vehicle_count_mean > float(idle_or_sparse_vehicle_max)
            and estimated_association_change_count > int(idle_or_sparse_association_change_max)
        ):
            window_class = "mechanism_activating"
        else:
            window_class = "idle_or_sparse"

        mechanism_score = round(
            2.0 * estimated_handoff_count
            + 0.5 * active_vehicle_count_mean
            + 8.0 * predicted_handoff_ratio
            + 4.0 * predicted_next_ratio,
            6,
        )
        enriched_windows.append(
            {
                "window_rank": index + 1,
                "window_id": item["window_id"],
                "frame_offset": item["frame_offset"],
                "window_length": item["window_length"],
                "time_index_start": item["time_index_start"],
                "time_index_end": item["time_index_end"],
                "dominant_axis": item["dominant_axis"],
                "recommended_rsu_layout": item["recommended_rsu_layout"],
                "chosen_rsu_axis": item["chosen_rsu_axis"],
                "coverage_radius": item["coverage_radius"],
                "spacing": item["spacing"],
                "estimated_association_change_count": estimated_association_change_count,
                "estimated_handoff_count": estimated_handoff_count,
                "active_vehicle_count_mean": active_vehicle_count_mean,
                "active_vehicle_count_max": item["active_vehicle_count_max"],
                "unique_vehicle_count": item["unique_vehicle_count"],
                "predicted_next_rsu_non_null_ratio": predicted_next_ratio,
                "predicted_handoff_target_non_null_ratio": predicted_handoff_ratio,
                "mechanism_activation_score": mechanism_score,
                "window_class": window_class,
            }
        )

    excluded_intervals = list(excluded_window_intervals or [])
    gap = max(0, int(holdout_min_gap_frames))
    excluded_windows = [
        item
        for item in enriched_windows
        if any(
            int(item["frame_offset"]) <= int(excluded_end) + gap
            and int(item["frame_offset"]) + int(item["window_length"]) - 1 >= int(excluded_start) - gap
            for excluded_start, excluded_end in excluded_intervals
        )
    ]
    if excluded_intervals:
        excluded_ids = {str(item["window_id"]) for item in excluded_windows}
        enriched_windows = [item for item in enriched_windows if str(item["window_id"]) not in excluded_ids]

    mechanism_activating_windows = [item for item in enriched_windows if item["window_class"] == "mechanism_activating"]
    active_non_mechanism_windows = [item for item in enriched_windows if item["window_class"] == "active_non_mechanism"]
    idle_or_sparse_windows = [item for item in enriched_windows if item["window_class"] == "idle_or_sparse"]
    selection_offset = max(0, int(window_rank_offset))
    mechanism_selection_pool = mechanism_activating_windows[selection_offset:]
    active_non_mechanism_selection_pool = active_non_mechanism_windows[selection_offset:]
    idle_or_sparse_selection_pool = idle_or_sparse_windows[selection_offset:]

    occupied_intervals: list[tuple[int, int]] = []

    def select_from_pool(pool: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for item in pool:
            if len(selected) >= count:
                break
            start = int(item["frame_offset"])
            end = start + int(item["window_length"]) - 1
            if enforce_non_overlapping_selection and any(
                start <= occupied_end + gap and end >= occupied_start - gap
                for occupied_start, occupied_end in occupied_intervals
            ):
                continue
            selected.append(item)
            occupied_intervals.append((start, end))
        return selected

    if window_mode == "activating_only":
        selected_windows = select_from_pool(mechanism_selection_pool, max(1, window_count))
        selected_window_plan_by_strata = {
            "mechanism_activating": list(selected_windows),
            "active_non_mechanism": [],
            "idle_or_sparse": [],
        }
        if not selected_windows:
            # raise RuntimeError("window_mode=activating_only ???????? mechanism_activating windows?")
            print("Warning: no activating windows found, but proceeding with empty set.")
    elif window_mode in {"mixed", "mixed_informative"}:
        target_count = max(1, window_count)
        activating_target = max(1, (target_count + 1) // 2)
        active_non_mechanism_target = max(1, target_count - activating_target)
        selected_windows = select_from_pool(mechanism_selection_pool, activating_target)
        selected_windows += select_from_pool(active_non_mechanism_selection_pool, active_non_mechanism_target)
        if len(selected_windows) < target_count:
            selected_windows += select_from_pool(
                mechanism_selection_pool[activating_target:]
                + active_non_mechanism_selection_pool[active_non_mechanism_target:],
                target_count - len(selected_windows),
            )
        if not selected_windows:
            raise RuntimeError("window_mode=mixed_informative ?????? mechanism_activating / active_non_mechanism windows?")
        selected_window_plan_by_strata = {
            "mechanism_activating": [item for item in selected_windows if item["window_class"] == "mechanism_activating"],
            "active_non_mechanism": [item for item in selected_windows if item["window_class"] == "active_non_mechanism"],
            "idle_or_sparse": [],
        }
    elif window_mode in {"full", "full_stratified"}:
        selected_window_plan_by_strata = {
            "mechanism_activating": select_from_pool(mechanism_selection_pool, max(1, window_count)),
            "active_non_mechanism": select_from_pool(active_non_mechanism_selection_pool, max(1, window_count)),
            "idle_or_sparse": select_from_pool(idle_or_sparse_selection_pool, max(1, window_count)),
        }
        selected_windows = (
            list(selected_window_plan_by_strata["mechanism_activating"])
            + list(selected_window_plan_by_strata["active_non_mechanism"])
            + list(selected_window_plan_by_strata["idle_or_sparse"])
        )
        if not selected_windows:
            raise RuntimeError("window_mode=full_stratified ????????")
    else:
        raise ValueError("window_mode ??? activating_only, mixed_informative, full_stratified??????? mixed/full")

    return source_path, {
        "window_mode": window_mode,
        "window_rank_offset": selection_offset,
        "excluded_window_intervals": [list(interval) for interval in excluded_intervals],
        "holdout_min_gap_frames": gap,
        "enforce_non_overlapping_selection": bool(enforce_non_overlapping_selection),
        "excluded_window_count": len(excluded_windows),
        "excluded_window_ids": [str(item["window_id"]) for item in excluded_windows],
        "selected_windows": selected_windows,
        "selected_window_plan_by_strata": selected_window_plan_by_strata,
        "mechanism_activating_windows": mechanism_activating_windows,
        "active_non_mechanism_windows": active_non_mechanism_windows,
        "idle_or_sparse_windows": idle_or_sparse_windows,
        "selection_pool_by_strata": {
            "mechanism_activating": mechanism_selection_pool,
            "active_non_mechanism": active_non_mechanism_selection_pool,
            "idle_or_sparse": idle_or_sparse_selection_pool,
        },
        "non_mechanism_windows": active_non_mechanism_windows,
        "activation_thresholds": {
            "activating_handoff_threshold": effective_activating_handoff_threshold,
            "activating_vehicle_threshold": effective_activating_vehicle_threshold,
            "activating_predicted_next_ratio_threshold": effective_activating_predicted_next_ratio_threshold,
            "activating_handoff_prediction_ratio_threshold": effective_activating_handoff_prediction_ratio_threshold,
            "non_mechanism_handoff_max": non_mechanism_handoff_max,
            "non_mechanism_prediction_ratio_max": non_mechanism_prediction_ratio_max,
            "active_non_mechanism_vehicle_threshold": active_non_mechanism_vehicle_threshold,
            "active_non_mechanism_association_change_min": active_non_mechanism_association_change_min,
            "active_non_mechanism_handoff_max": active_non_mechanism_handoff_max,
            "active_non_mechanism_predicted_next_ratio_max": active_non_mechanism_predicted_next_ratio_max,
            "active_non_mechanism_handoff_prediction_ratio_max": active_non_mechanism_handoff_prediction_ratio_max,
            "idle_or_sparse_vehicle_max": idle_or_sparse_vehicle_max,
            "idle_or_sparse_association_change_max": idle_or_sparse_association_change_max,
            "lust_relaxed_activation": mobility_source == "lust",
        },
    }


def apply_frozen_window_plan(
    window_payload: dict[str, Any],
    plan_path: str | Path,
) -> dict[str, Any]:
    """Replace outcome-derived selection with an explicitly frozen plan."""
    resolved_path = Path(plan_path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8-sig"))
    metadata = payload if isinstance(payload, dict) else {}
    plan = metadata.get("selected_window_plan") if metadata else payload
    if not isinstance(plan, list) or not plan:
        raise ValueError(f"selected_window_plan missing or empty: {resolved_path}")
    required_fields = {"window_id", "frame_offset", "window_length", "window_class"}
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(plan):
        if not isinstance(item, dict):
            raise ValueError(f"window plan item {index} is not an object: {resolved_path}")
        missing = required_fields.difference(item)
        if missing:
            raise ValueError(f"window plan item {index} missing {sorted(missing)}: {resolved_path}")
        window_id = str(item["window_id"])
        if window_id in seen_ids:
            raise ValueError(f"duplicate window_id {window_id}: {resolved_path}")
        seen_ids.add(window_id)
        normalized.append(dict(item))
    by_strata = {
        label: [item for item in normalized if str(item.get("window_class")) == label]
        for label in ("mechanism_activating", "active_non_mechanism", "idle_or_sparse")
    }
    updated = dict(window_payload)
    updated.update(
        {
            "selected_windows": normalized,
            "selected_window_plan_by_strata": by_strata,
            "mechanism_activating_windows": by_strata["mechanism_activating"],
            "active_non_mechanism_windows": by_strata["active_non_mechanism"],
            "idle_or_sparse_windows": by_strata["idle_or_sparse"],
            "non_mechanism_windows": by_strata["active_non_mechanism"],
            "frozen_window_plan_path": str(resolved_path.resolve()),
            "frozen_window_plan_protocol_version": str(metadata.get("protocol_version", "unknown")),
            "frozen_window_plan_split": str(metadata.get("split", "unknown")),
            "outcome_blind_selection": bool(metadata.get("outcome_blind_selection", False)),
        }
    )
    return updated


def resolve_agent_checkpoint(agent_name: str, checkpoint_map: dict[str, str]) -> str:
    checkpoint_path = checkpoint_map.get(agent_name, "")
    if agent_name in CHECKPOINT_REQUIRED_AGENTS:
        ensure_agent_checkpoint_path(agent_name, checkpoint_path)
    return checkpoint_path


def _load_train_summary_metadata(checkpoint_path: Path) -> dict[str, Any]:
    train_summary_path = checkpoint_path.parent.parent / "train_summary.json"
    if not train_summary_path.exists():
        return {
            "checkpoint_path": str(checkpoint_path),
            "train_summary_path": str(train_summary_path),
            "metadata_source": "missing",
            "exists": checkpoint_path.exists(),
            "run_id": "unknown",
            "config_profile": "unknown",
            "train_window_mode": "unknown",
            "episodes": 0,
            "run_update_count": 0,
            "checkpoint_source_update_index": 0,
            "is_smoke_checkpoint": False,
            "smoke_warning": False,
        }
    payload = json.loads(train_summary_path.read_text(encoding="utf-8"))
    episodes = int(payload.get("episodes", 0) or 0)
    run_update_count = int(payload.get("update_count", 0) or len(payload.get("update_logs", [])) or payload.get("training_effectiveness_audit", {}).get("update_count", 0) or 0)
    config_profile = str(payload.get("config_profile", "unknown"))
    is_smoke = bool(config_profile == "smoke" or episodes <= 4)
    return {
        "checkpoint_path": str(checkpoint_path),
        "train_summary_path": str(train_summary_path),
        "metadata_source": "train_summary",
        "exists": checkpoint_path.exists(),
        "run_id": str(payload.get("run_id", "unknown")),
        "config_profile": config_profile,
        "train_window_mode": str(payload.get("train_window_mode", "unknown")),
        "episodes": episodes,
        "run_update_count": run_update_count,
        "checkpoint_source_update_index": 0,
        "is_smoke_checkpoint": is_smoke,
        "smoke_warning": is_smoke,
    }


def load_checkpoint_metadata(checkpoint_path: str) -> dict[str, Any]:
    if not checkpoint_path:
        metadata = {
            "checkpoint_path": "",
            "exists": False,
            "metadata_source": "none",
            "run_id": "none",
            "config_profile": "none",
            "train_window_mode": "none",
            "episodes": 0,
            "run_update_count": 0,
            "checkpoint_source_update_index": 0,
            "is_smoke_checkpoint": False,
            "smoke_warning": False,
        }
        metadata.update(classify_experiment_scale("none", 0, 0))
        metadata["update_count"] = metadata["run_update_count"]
        return metadata
    resolved = Path(checkpoint_path)
    if not resolved.exists():
        raise FileNotFoundError(f"checkpoint ???: {resolved}")
    metadata = _load_train_summary_metadata(resolved)
    try:
        payload = torch.load(resolved, map_location="cpu")
    except Exception:
        metadata.update(classify_experiment_scale(metadata.get("config_profile", "unknown"), metadata.get("episodes", 0), metadata.get("run_update_count", 0)))
        metadata["update_count"] = metadata["run_update_count"]
        return metadata
    if isinstance(payload, dict):
        training_metadata = payload.get("training_metadata") or payload.get("checkpoint_metadata")
        if isinstance(training_metadata, dict):
            episodes = int(training_metadata.get("episodes", metadata.get("episodes", 0)) or 0)
            config_profile = str(training_metadata.get("config_profile", metadata.get("config_profile", "unknown")))
            checkpoint_source_update_index = int(training_metadata.get("update_count", metadata.get("checkpoint_source_update_index", 0)) or 0)
            is_smoke = bool(training_metadata.get("is_smoke_checkpoint", config_profile == "smoke" or episodes <= 4))
            metadata.update(
                {
                    "metadata_source": "checkpoint_metadata",
                    "run_id": str(training_metadata.get("run_id", metadata.get("run_id", "unknown"))),
                    "config_profile": config_profile,
                    "train_window_mode": str(training_metadata.get("train_window_mode", metadata.get("train_window_mode", "unknown"))),
                    "episodes": episodes,
                    "checkpoint_source_update_index": checkpoint_source_update_index,
                    "is_smoke_checkpoint": is_smoke,
                    "smoke_warning": is_smoke,
                }
            )
    metadata.update(classify_experiment_scale(metadata.get("config_profile", "unknown"), metadata.get("episodes", 0), metadata.get("run_update_count", 0)))
    metadata["update_count"] = metadata["run_update_count"]
    return metadata


def audit_checkpoint_map(checkpoint_map: dict[str, str], agents: list[str]) -> dict[str, Any]:
    audits: dict[str, Any] = {}
    warnings: list[str] = []
    for agent_name in agents:
        if agent_name not in CHECKPOINT_REQUIRED_AGENTS:
            audits[agent_name] = {
                "checkpoint_path": checkpoint_map.get(agent_name, ""),
                "requires_checkpoint": False,
                "config_profile": "non_checkpoint_agent",
                "run_id": agent_name,
                "episodes": 0,
                "run_update_count": 0,
                "checkpoint_source_update_index": 0,
                "update_count": 0,
                "is_smoke_checkpoint": False,
                "smoke_warning": False,
            }
            continue
        checkpoint_path = checkpoint_map.get(agent_name, "")
        metadata = load_checkpoint_metadata(checkpoint_path)
        metadata["requires_checkpoint"] = True
        audits[agent_name] = metadata
        if metadata.get("smoke_warning"):
            warnings.append(f"agent={agent_name} uses a smoke checkpoint: {checkpoint_path}")
    return {"checkpoint_audit": audits, "warnings": warnings}


def infer_benchmark_config_profile(checkpoint_audit: dict[str, Any], agents: list[str]) -> str:
    profiles = {
        str(checkpoint_audit.get(agent_name, {}).get("config_profile", "unknown"))
        for agent_name in agents
    }
    profiles.discard("non_checkpoint_agent")
    if not profiles:
        return "non_checkpoint_only"
    if len(profiles) == 1:
        return sorted(profiles)[0]
    return "mixed"


def build_window_context_agent_overrides(
    *,
    agent_name: str,
    checkpoint_profile: str,
    run_metadata: dict[str, Any],
    base_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = dict(base_overrides or {})
    if (
        agent_name == "sa_ghmappo"
        and checkpoint_profile == SA_GHMAPPO_V11_REWARD_PROFILE
        and str(run_metadata.get("window_class", "")) == "idle_or_sparse"
    ):
        overrides.setdefault("idle_popularity_no_rsu_local_fallback_enabled", True)
        overrides.setdefault("idle_popularity_no_rsu_local_requires_low_context", False)
    return overrides


def run_real_episode(
    *,
    root_dir: Path,
    agent_name: str,
    checkpoint_map: dict[str, str],
    workflow_state: Any,
    workflow_source_path: str,
    mobility_bundle: Any,
    seed: int,
    max_steps: int,
    mobility_source: str = "ngsim",
    primary_vehicle_selection: str = "stable_first",
    run_metadata: dict[str, Any],
    predictor_kwargs: dict[str, Any] | None = None,
    agent_config_overrides: dict[str, Any] | None = None,
    adapter_catalog_override: AdapterCatalog | None = None,
    workflow_state_override: WorkflowGraphState | None = None,
    rsu_states_override: list[RSUState] | None = None,
    mobility_frames_override: list[dict[str, Any]] | None = None,
    cache_capacity_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recorder = EpisodeRecorder(prefetch_validation_window=6)
    adapter_catalog = adapter_catalog_override or AdapterCatalog.from_json(root_dir / "src" / "data" / "model_catalog" / "sample_model_catalog.json")
    rsu_states = [clone_rsu_state(rsu_state) for rsu_state in (rsu_states_override or mobility_bundle.rsu_states)]
    workflow_state_runtime = clone_workflow_state(workflow_state_override or workflow_state)
    trajectory_frames = clone_frames(mobility_frames_override or mobility_bundle.frames)
    checkpoint_path = resolve_agent_checkpoint(agent_name, checkpoint_map)
    checkpoint_metadata = load_checkpoint_metadata(checkpoint_path) if checkpoint_path else {
        "checkpoint_path": "",
        "config_profile": "non_checkpoint_agent",
        "run_id": agent_name,
        "episodes": 0,
        "update_count": 0,
        "is_smoke_checkpoint": False,
        "smoke_warning": False,
    }
    runtime_predictor_kwargs = dict(predictor_kwargs or {})
    if runtime_predictor_kwargs.get("oracle_prediction_enabled"):
        runtime_predictor_kwargs.setdefault("oracle_future_frames", trajectory_frames)
        runtime_predictor_kwargs.setdefault("oracle_rsu_states", rsu_states)
    core_env = VecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(trajectory_frames=trajectory_frames),
        workflow_state=workflow_state_runtime,
        adapter_catalog=adapter_catalog,
        rsu_states=rsu_states,
        predictor_manager=PredictorManager(**runtime_predictor_kwargs),
        max_steps=max(max_steps + 2, 8),
        mobility_source=mobility_source,
        primary_vehicle_selection=primary_vehicle_selection,
        cache_capacity_profile=cache_capacity_profile,
    )
    env = GymVecEnv(core_env=core_env, recorder=recorder)
    runtime_agent_config_overrides = build_window_context_agent_overrides(
        agent_name=agent_name,
        checkpoint_profile=str(checkpoint_metadata.get("config_profile", "")),
        run_metadata=run_metadata,
        base_overrides=agent_config_overrides,
    )
    agent = build_inference_agent(
        agent_name=agent_name,
        random_seed=seed,
        checkpoint_path=checkpoint_path,
        deterministic_action=True,
        agent_config_overrides=runtime_agent_config_overrides,
    )
    trainer = MARLOnPolicyTrainer(env=env, agent=agent, recorder=recorder, max_steps=max_steps)
    summary = trainer.run_episode(
        run_metadata={
            **run_metadata,
            "workflow_id": workflow_state_runtime.workflow_id,
            "workflow_source_path": workflow_source_path,
            "agent_name": agent_name,
            "seed": seed,
            "window_id": mobility_bundle.rsu_metadata.get("window_id"),
            "rsu_layout": mobility_bundle.rsu_metadata.get("effective_rsu_layout"),
            "primary_vehicle_selection": primary_vehicle_selection,
            "checkpoint_run_id": checkpoint_metadata.get("run_id"),
            "checkpoint_profile": checkpoint_metadata.get("config_profile"),
            "checkpoint_is_smoke": checkpoint_metadata.get("is_smoke_checkpoint", False),
        },
        learn=False,
    )
    summary["episode_success"] = bool(summary.get("episode_status", {}).get("completed", False))
    summary["run_info"]["rsu_metadata"] = mobility_bundle.rsu_metadata
    summary["run_info"]["checkpoint_metadata"] = checkpoint_metadata
    return summary


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _float_value(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _reward_sum(step_trace: list[dict[str, Any]], reward_key: str) -> float:
    total = 0.0
    for step in step_trace:
        reward_dict = step.get("reward_dict") or {}
        total += _float_value(reward_dict.get(reward_key), 0.0)
    return total


def _control_block(step: dict[str, Any], block_name: str) -> dict[str, Any]:
    control = step.get("control_action") or {}
    block = control.get(block_name) or {}
    return block if isinstance(block, dict) else {}


def _offload_mode(step: dict[str, Any]) -> str:
    mode = _control_block(step, "offload_action").get("mode")
    return str(mode or "").lower()


def _is_executable_step(step: dict[str, Any]) -> bool:
    return step.get("current_node_id") is not None


def _is_current_rsu_execution(step: dict[str, Any]) -> bool:
    if _offload_mode(step) != "rsu":
        return False
    action_name = str(step.get("action_name") or "")
    target_rsu_id = step.get("offload_target_rsu_id")
    current_candidates = {
        step.get("pre_action_associated_rsu_id"),
        step.get("current_associated_rsu_id"),
        step.get("post_action_associated_rsu_id"),
    }
    if target_rsu_id is not None and target_rsu_id in current_candidates:
        return True
    return action_name in {"current_rsu_cache_fill", "current_rsu_steady_offload", "handoff_migration_prepare"}


def _is_next_rsu_execution(step: dict[str, Any]) -> bool:
    if _offload_mode(step) != "rsu":
        return False
    target_rsu_id = step.get("offload_target_rsu_id")
    predicted_next_rsu_id = step.get("predicted_next_rsu_id")
    current_rsu_id = step.get("pre_action_associated_rsu_id") or step.get("current_associated_rsu_id")
    return bool(target_rsu_id and predicted_next_rsu_id and target_rsu_id == predicted_next_rsu_id and target_rsu_id != current_rsu_id)


def _is_neighbor_rsu_execution(step: dict[str, Any]) -> bool:
    if _offload_mode(step) != "rsu":
        return False
    if _is_current_rsu_execution(step) or _is_next_rsu_execution(step):
        return False
    return step.get("offload_target_rsu_id") is not None


def _build_actionmix_diagnostics(summary: dict[str, Any]) -> dict[str, float]:
    step_trace = [step for step in summary.get("step_trace", []) if isinstance(step, dict)]
    executable_steps = [step for step in step_trace if _is_executable_step(step)]
    success_steps = [step for step in executable_steps if not _bool_value(step.get("stall_occurred", False))]
    stall_steps = [step for step in executable_steps if _bool_value(step.get("stall_occurred", False))]
    cache_hits = [step for step in executable_steps if _bool_value(step.get("cache_hit", False))]
    cache_misses = [step for step in executable_steps if not _bool_value(step.get("cache_hit", False))]
    prefetch_successes = [
        step
        for step in step_trace
        if _bool_value(step.get("prefetch_validated_hit", False))
        or _bool_value(step.get("predictive_prefetch_correct", False))
    ]
    migration_attempts = [
        step
        for step in step_trace
        if _bool_value(step.get("migration_prepare_requested", False))
        or str(step.get("migration_mode") or "").lower() in {"prepare", "migrate"}
    ]
    migration_successes = [
        step
        for step in step_trace
        if _bool_value(step.get("migration_prepare_realized", False))
        or _bool_value(step.get("migration_during_handoff", False))
    ]
    mechanism_attempts = [
        step
        for step in step_trace
        if _bool_value(step.get("mechanism_attempt_selected", False))
        or _bool_value(step.get("predictive_prefetch_requested", False))
        or _bool_value(step.get("migration_prepare_requested", False))
    ]
    mechanism_strict_successes = [
        step
        for step in step_trace
        if _bool_value(step.get("mechanism_success_strict", False))
    ]
    service_reward = _reward_sum(step_trace, "service_reward")
    delay_penalty = _reward_sum(step_trace, "delay_penalty")
    cache_miss_penalty = _reward_sum(step_trace, "cache_miss_penalty")
    migration_cost = _reward_sum(step_trace, "migration_cost")
    continuity_bonus = _reward_sum(step_trace, "continuity_bonus")
    mechanism_bonus = _reward_sum(step_trace, "mechanism_exploration_bonus")
    constraint_penalty = _reward_sum(step_trace, "constraint_penalty")
    migration_failed_count = max(float(len(migration_attempts) - len(migration_successes)), 0.0)
    capacity_enabled_steps = [step for step in step_trace if _bool_value(step.get("cache_capacity_enabled", False))]
    cache_used_values = [_float_value(step.get("cache_used_size"), 0.0) for step in capacity_enabled_steps if step.get("cache_used_size") is not None]
    cache_remaining_values = [
        _float_value(step.get("cache_remaining_size"), 0.0)
        for step in capacity_enabled_steps
        if step.get("cache_remaining_size") is not None
    ]
    cache_occupancy_values = [
        _float_value(step.get("cache_occupancy_rate"), 0.0)
        for step in capacity_enabled_steps
        if step.get("cache_occupancy_rate") is not None
    ]
    cache_capacity_values = [
        _float_value(step.get("cache_capacity"), 0.0)
        for step in capacity_enabled_steps
        if step.get("cache_capacity") is not None
    ]
    rsu_slot_values = [
        _float_value(step.get("rsu_adapter_slots"), 0.0)
        for step in step_trace
        if step.get("rsu_adapter_slots") not in (None, "")
    ]
    cache_capacity_unit = next(
        (str(step.get("cache_capacity_unit")) for step in step_trace if step.get("cache_capacity_unit")),
        "adapter_slots",
    )
    diagnostics: dict[str, float] = {
        "service_success_count": float(len(success_steps)),
        "service_delay_sum": delay_penalty,
        "service_wait_sum": float(len(stall_steps)),
        "service_restart_count": float(sum(1 for step in step_trace if _bool_value(step.get("service_restart", False)))),
        "workflow_completed_count": 1.0 if summary.get("episode_success", False) else 0.0,
        "workflow_unfinished_count": 0.0 if summary.get("episode_success", False) else 1.0,
        "adapter_hit_count": float(len(cache_hits)),
        "adapter_miss_count": float(len(cache_misses)),
        "adapter_warm_hit_count": float(sum(1 for step in executable_steps if _bool_value(step.get("warm_hit", False)))),
        "adapter_cold_start_count": float(sum(1 for step in executable_steps if _bool_value(step.get("cross_rsu_cold_start", False)))),
        "cache_eviction_count": float(sum(1 for step in step_trace if _bool_value(step.get("cache_eviction", False)))),
        "cache_admission_count": float(sum(1 for step in step_trace if _bool_value(step.get("cache_applied", False)))),
        "cache_admission_added_new_adapter_count": float(
            sum(1 for step in step_trace if _bool_value(step.get("cache_admission_added_new_adapter", False)))
        ),
        "cache_capacity_enabled": 1.0 if capacity_enabled_steps else 0.0,
        "rsu_adapter_slots": max(rsu_slot_values) if rsu_slot_values else 0.0,
        "cache_capacity": max(cache_capacity_values) if cache_capacity_values else 0.0,
        "cache_used_size": round(fmean(cache_used_values), 6) if cache_used_values else 0.0,
        "cache_remaining_size": round(fmean(cache_remaining_values), 6) if cache_remaining_values else 0.0,
        "cache_occupancy_rate": round(fmean(cache_occupancy_values), 6) if cache_occupancy_values else 0.0,
        "eviction_count": float(sum(_float_value(step.get("eviction_count"), 0.0) for step in step_trace)),
        "evicted_adapter_count": float(sum(_float_value(step.get("evicted_adapter_count"), 0.0) for step in step_trace)),
        "cache_noop_count": float(
            sum(
                1
                for step in step_trace
                if not _bool_value(step.get("cache_applied", False))
                and str(step.get("cache_strategy") or "none").lower() in {"", "none"}
            )
        ),
        "local_exec_count": float(sum(1 for step in executable_steps if _offload_mode(step) == "vehicle")),
        "current_rsu_exec_count": float(sum(1 for step in executable_steps if _is_current_rsu_execution(step))),
        "next_rsu_exec_count": float(sum(1 for step in executable_steps if _is_next_rsu_execution(step))),
        "neighbor_rsu_exec_count": float(sum(1 for step in executable_steps if _is_neighbor_rsu_execution(step))),
        "cloud_exec_count": float(sum(1 for step in executable_steps if _offload_mode(step) == "cloud")),
        "prefetch_action_count": float(
            sum(
                1
                for step in step_trace
                if str(step.get("action_name") or "") == "predictive_next_rsu_prefetch"
                or _bool_value(step.get("predictive_prefetch_requested", False))
            )
        ),
        "migration_action_count": float(len(migration_attempts)),
        "no_op_action_count": float(sum(1 for step in step_trace if str(step.get("action_name") or "") in {"no_op", "unknown_action"})),
        "prefetch_attempt_count": float(sum(1 for step in step_trace if _bool_value(step.get("predictive_prefetch_requested", False)))),
        "prefetch_success_count": float(len(prefetch_successes)),
        "prefetch_failed_count": float(sum(1 for step in step_trace if _bool_value(step.get("prefetch_expired_miss", False)))),
        "migration_attempt_count": float(len(migration_attempts)),
        "migration_success_count": float(len(migration_successes)),
        "migration_failed_count": migration_failed_count,
        "mechanism_attempt_count": float(len(mechanism_attempts)),
        "mechanism_validated_success_count": float(len(prefetch_successes) + len(mechanism_strict_successes)),
        "mechanism_pending_success_count": float(
            sum(1 for step in step_trace if _bool_value(step.get("mechanism_success_gate_pending", False)))
        ),
        "mechanism_success_rate": (
            float(len(prefetch_successes) + len(mechanism_strict_successes))
            / float(max(len(mechanism_attempts), 1))
        ),
        "env_invalid_action_count": float(sum(1 for step in step_trace if _bool_value(step.get("action_invalid", False)))),
        "migration_overhead_sum": sum(_float_value(step.get("adapter_state_migration_overhead"), 0.0) for step in step_trace),
        "delay_reward_component": -delay_penalty,
        "cache_reward_component": -cache_miss_penalty,
        "handoff_reward_component": mechanism_bonus,
        "mechanism_shaping_reward_component": mechanism_bonus,
        "backhaul_reward_component": -migration_cost,
        "failure_reward_component": -constraint_penalty,
        "continuity_reward_component": continuity_bonus,
        "service_reward_component": service_reward,
        "mechanism_exploration_reward_component": mechanism_bonus,
        "constraint_penalty_sum": constraint_penalty,
        "migration_cost_sum": migration_cost,
        "cache_miss_penalty_sum": cache_miss_penalty,
        "delay_penalty_sum": delay_penalty,
    }
    rounded = {key: round(float(value), 6) for key, value in diagnostics.items()}
    rounded["cache_capacity_unit"] = cache_capacity_unit
    return rounded


def summary_to_row(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["system_metrics"]
    handoff_summary = summary["handoff_summary"]
    prefetch_summary = summary["prefetch_summary"]
    validation_summary = summary["prefetch_validation_summary"]
    run_info = summary["run_info"]
    checkpoint_metadata = run_info.get("checkpoint_metadata", {})
    agent_action_diagnostics = summary.get("agent_action_diagnostics", {})
    mechanism_realized = int(
        validation_summary.get("validated_predictive_prefetch_count", 0) > 0
        or handoff_summary.get("handoff_ready_count", 0) > 0
        or handoff_summary.get("migration_during_handoff_count", 0) > 0
    )
    actionmix_diagnostics = _build_actionmix_diagnostics(summary)
    return {
        "window_id": run_info.get("window_id"),
        "scenario_id": run_info.get("scenario_id", run_info.get("window_id")),
        "mode": run_info.get("window_mode", run_info.get("mode", "unknown")),
        "window_rank": run_info.get("window_rank"),
        "window_class": run_info.get("window_class", run_info.get("rsu_metadata", {}).get("window_class", "unknown")),
        "window_tag": run_info.get("window_class", run_info.get("rsu_metadata", {}).get("window_class", "unknown")),
        "workflow_id": run_info.get("workflow_id"),
        "agent_name": run_info.get("agent_name"),
        "policy_name": run_info.get("agent_name"),
        "seed": run_info.get("seed"),
        "primary_vehicle_selection": run_info.get("primary_vehicle_selection", "stable_first"),
        "episode_success": summary.get("episode_success", False),
        "successful_episode_rate": 1.0 if summary.get("episode_success", False) else 0.0,
        "mechanism_realization_rate": float(mechanism_realized),
        "continuity_guard_trigger_count": int(agent_action_diagnostics.get("continuity_guard_trigger_count", 0) or 0),
        "continuity_guard_trigger_rate": float(agent_action_diagnostics.get("continuity_guard_trigger_rate", 0.0) or 0.0),
        "target_mismatch_guard_count": int(agent_action_diagnostics.get("target_mismatch_guard_count", 0) or 0),
        "guard_prefetch_to_prepare_count": int(agent_action_diagnostics.get("guard_prefetch_to_prepare_count", 0) or 0),
        "guard_hard_override_count": int(agent_action_diagnostics.get("guard_hard_override_count", 0) or 0),
        "action_projection_count": int(agent_action_diagnostics.get("action_projection_count", 0) or 0),
        "action_projection_rate": float(agent_action_diagnostics.get("action_projection_rate", 0.0) or 0.0),
        "invalid_action_attempt_count": int(agent_action_diagnostics.get("invalid_action_attempt_count", 0) or 0),
        "invalid_action_attempt_rate": float(agent_action_diagnostics.get("invalid_action_attempt_rate", 0.0) or 0.0),
        "guard_action_delta_count": int(agent_action_diagnostics.get("guard_action_delta_count", 0) or 0),
        "guard_action_delta_rate": float(agent_action_diagnostics.get("guard_action_delta_rate", 0.0) or 0.0),
        "dag_frontier_size_mean": float(agent_action_diagnostics.get("dag_frontier_size_mean", 0.0) or 0.0),
        "dag_critical_path_pressure_mean": float(agent_action_diagnostics.get("dag_critical_path_pressure_mean", 0.0) or 0.0),
        "dag_current_node_dependency_pressure_mean": float(agent_action_diagnostics.get("dag_current_node_dependency_pressure_mean", 0.0) or 0.0),
        "dag_remaining_nodes_mean": float(agent_action_diagnostics.get("dag_remaining_nodes_mean", 0.0) or 0.0),
        "backhaul_guard_count": int(agent_action_diagnostics.get("backhaul_guard_count", 0) or 0),
        "backhaul_guard_rate": float(agent_action_diagnostics.get("backhaul_guard_rate", 0.0) or 0.0),
        "cache_warm_start_guard_count": int(agent_action_diagnostics.get("cache_warm_start_guard_count", 0) or 0),
        "cache_warm_start_guard_rate": float(agent_action_diagnostics.get("cache_warm_start_guard_rate", 0.0) or 0.0),
        "predictive_prefetch_admission_guard_count": int(
            agent_action_diagnostics.get("predictive_prefetch_admission_guard_count", 0) or 0
        ),
        "predictive_prefetch_admission_guard_rate": float(
            agent_action_diagnostics.get("predictive_prefetch_admission_guard_rate", 0.0) or 0.0
        ),
        "total_reward": summary["reward_breakdown"]["total"]["sum"],
        "end_to_end_workflow_delay": metrics["end_to_end_workflow_delay"],
        "workflow_continuity_rate": metrics["workflow_continuity_rate"],
        "handoff_failure_rate": metrics["handoff_failure_rate"],
        "handoff_ready_ratio": metrics["handoff_ready_ratio"],
        "adapter_warm_hit_ratio": metrics["adapter_warm_hit_ratio"],
        "cross_rsu_cold_start_frequency": metrics["cross_rsu_cold_start_frequency"],
        "backhaul_traffic_cost": metrics["backhaul_traffic_cost"],
        "adapter_state_migration_overhead": metrics["adapter_state_migration_overhead"],
        "predictive_prefetch_precision": metrics["predictive_prefetch_precision"],
        "validated_predictive_prefetch_count": validation_summary["validated_predictive_prefetch_count"],
        "migration_during_handoff_count": handoff_summary["migration_during_handoff_count"],
        "handoff_ready_count": handoff_summary["handoff_ready_count"],
        "handoff_total_count": handoff_summary["handoff_total_count"],
        "predictive_prefetch_request_count": prefetch_summary["true_predictive_prefetch_count"],
        "prefetch_validated_hit_count": validation_summary["prefetch_validated_hit_count"],
        "prefetch_expired_miss_count": validation_summary["prefetch_expired_miss_count"],
        "migration_prepare_count": handoff_summary["migration_prepare_count"],
        "checkpoint_run_id": checkpoint_metadata.get("run_id", "none"),
        "checkpoint_profile": checkpoint_metadata.get("config_profile", "none"),
        "checkpoint_run_update_count": checkpoint_metadata.get("run_update_count", checkpoint_metadata.get("update_count", 0)),
        "checkpoint_source_update_index": checkpoint_metadata.get("checkpoint_source_update_index", 0),
        "checkpoint_episode_count": checkpoint_metadata.get("episodes", 0),
        "checkpoint_is_smoke": checkpoint_metadata.get("is_smoke_checkpoint", False),
        **actionmix_diagnostics,
    }



def metric_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(fmean(values), 6),
        "std": round(pstdev(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str], metrics: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = "|".join(str(row[group_key]) for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    aggregate: dict[str, Any] = {}
    for key, group_rows in grouped.items():
        aggregate[key] = {
            "group": {group_key: group_rows[0][group_key] for group_key in group_keys},
            "episode_count": len(group_rows),
            "metrics": {
                metric_name: metric_stats([float(item[metric_name]) for item in group_rows])
                for metric_name in metrics
            },
        }
    return aggregate


def build_pairwise_comparison(
    aggregate_by_agent: dict[str, Any],
    baseline_agent: str,
    metrics: list[str],
) -> dict[str, Any]:
    if baseline_agent not in aggregate_by_agent:
        return {}
    baseline_metrics = aggregate_by_agent[baseline_agent]["metrics"]
    comparison: dict[str, Any] = {}
    for agent_name, payload in aggregate_by_agent.items():
        if agent_name == baseline_agent:
            continue
        delta: dict[str, float] = {}
        result: dict[str, str] = {}
        for metric_name in metrics:
            delta_value = round(
                float(payload["metrics"][metric_name]["mean"]) - float(baseline_metrics[metric_name]["mean"]),
                6,
            )
            delta[metric_name] = delta_value
            effective_delta = -delta_value if metric_name in LOWER_IS_BETTER_METRICS else delta_value
            if effective_delta > 1e-6:
                result[metric_name] = "win"
            elif effective_delta < -1e-6:
                result[metric_name] = "loss"
            else:
                result[metric_name] = "tie"
        comparison[agent_name] = {
            "baseline_agent": baseline_agent,
            "delta_vs_baseline": delta,
            "result_by_metric": result,
        }
    return comparison


def _collect_group_scores(
    aggregate_table: dict[str, Any],
    baseline_agent: str,
    group_dimension: str,
    metrics: list[str],
) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for payload in aggregate_table.values():
        group_value = str(payload["group"].get(group_dimension))
        agent_name = str(payload["group"].get("agent_name"))
        grouped.setdefault(group_value, {})[agent_name] = payload["metrics"]
    counts: dict[str, dict[str, dict[str, int]]] = {}
    overall_counts: dict[str, dict[str, int]] = {}
    for group_value, agent_payloads in grouped.items():
        baseline_metrics = agent_payloads.get(baseline_agent)
        if baseline_metrics is None:
            continue
        for agent_name, agent_metrics in agent_payloads.items():
            if agent_name == baseline_agent:
                continue
            counts.setdefault(agent_name, {metric_name: {"win": 0, "tie": 0, "loss": 0} for metric_name in metrics})
            group_outcome = "tie"
            for metric_name in metrics:
                delta_value = float(agent_metrics[metric_name]["mean"]) - float(baseline_metrics[metric_name]["mean"])
                effective_delta = -delta_value if metric_name in LOWER_IS_BETTER_METRICS else delta_value
                if effective_delta > 1e-6:
                    outcome = "win"
                elif effective_delta < -1e-6:
                    outcome = "loss"
                else:
                    outcome = "tie"
                counts[agent_name][metric_name][outcome] += 1
                if outcome == "loss":
                    group_outcome = "loss"
                elif outcome == "win" and group_outcome != "loss":
                    group_outcome = "win"
            overall_counts.setdefault(agent_name, {"win": 0, "tie": 0, "loss": 0})
            overall_counts[agent_name][group_outcome] += 1
    return {
        "baseline_agent": baseline_agent,
        "group_dimension": group_dimension,
        "per_agent_metric_counts": counts,
        "per_agent_group_counts": overall_counts,
    }


def _build_named_win_tie_loss_summary(
    aggregate_by_window_and_agent: dict[str, Any],
    aggregate_by_workflow_and_agent: dict[str, Any],
    metrics: list[str],
    baseline_agent: str,
) -> dict[str, Any]:
    return {
        "window_level": _collect_group_scores(
            aggregate_by_window_and_agent,
            baseline_agent=baseline_agent,
            group_dimension="window_id",
            metrics=metrics,
        ),
        "workflow_level": _collect_group_scores(
            aggregate_by_workflow_and_agent,
            baseline_agent=baseline_agent,
            group_dimension="workflow_id",
            metrics=metrics,
        ),
    }


def build_win_tie_loss_summary(
    aggregate_by_window_and_agent: dict[str, Any],
    aggregate_by_workflow_and_agent: dict[str, Any],
    metrics: list[str],
) -> dict[str, Any]:
    return _build_named_win_tie_loss_summary(
        aggregate_by_window_and_agent=aggregate_by_window_and_agent,
        aggregate_by_workflow_and_agent=aggregate_by_workflow_and_agent,
        metrics=metrics,
        baseline_agent="sa_ghmappo",
    )


def build_mechanism_diagnosis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    triggered = {
        "predictive_prefetch_request_count": int(sum(float(row["predictive_prefetch_request_count"]) for row in rows)),
        "migration_prepare_count": int(sum(float(row["migration_prepare_count"]) for row in rows)),
        "handoff_total_count": int(sum(float(row["handoff_total_count"]) for row in rows)),
    }
    realized = {
        "prefetch_validated_hit_count": int(sum(float(row["prefetch_validated_hit_count"]) for row in rows)),
        "handoff_ready_count": int(sum(float(row["handoff_ready_count"]) for row in rows)),
        "migration_during_handoff_count": int(sum(float(row["migration_during_handoff_count"]) for row in rows)),
    }
    per_agent: dict[str, Any] = {}
    for agent_name in sorted({str(row["agent_name"]) for row in rows}):
        agent_rows = [row for row in rows if str(row["agent_name"]) == agent_name]
        per_agent[agent_name] = {
            "successful_episode_rate": round(fmean(float(row["successful_episode_rate"]) for row in agent_rows), 6) if agent_rows else 0.0,
            "mechanism_realization_rate": round(fmean(float(row["mechanism_realization_rate"]) for row in agent_rows), 6) if agent_rows else 0.0,
            "triggered": {
                metric_name: int(sum(float(row.get(metric_name, 0.0)) for row in agent_rows))
                for metric_name in ["predictive_prefetch_request_count", "migration_prepare_count", "handoff_total_count"]
            },
            "realized": {
                metric_name: int(sum(float(row.get(metric_name, 0.0)) for row in agent_rows))
                for metric_name in ["prefetch_validated_hit_count", "handoff_ready_count", "migration_during_handoff_count"]
            },
        }
    return {
        "triggered": triggered,
        "realized": realized,
        "successful_episode_rate": round(fmean(float(row["successful_episode_rate"]) for row in rows), 6) if rows else 0.0,
        "mechanism_realization_rate": round(fmean(float(row["mechanism_realization_rate"]) for row in rows), 6) if rows else 0.0,
        "per_agent": per_agent,
    }


def write_rows_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_window_bundle(
    *,
    root_dir: Path,
    mobility_csv_path: str,
    max_mobility_rows: int,
    rsu_layout: str,
    frame_offset: int,
    window_length: int,
    random_seed: int,
    mobility_source: str = "ngsim",
    lust_scenario_root: str = "",
) -> Any:
    return load_real_mobility_bundle(
        root_dir=root_dir,
        mobility_source=mobility_source,
        mobility_csv_path=mobility_csv_path,
        lust_scenario_root=lust_scenario_root,
        max_mobility_rows=max_mobility_rows,
        rsu_layout=rsu_layout,
        frame_offset=frame_offset,
        window_length=window_length,
        window_selector="ordered",
        random_seed=random_seed,
    )


def build_rsu_layout_proxy(base_bundle: RealMobilityBundle, rsu_count: int, rsu_coverage_radius: float) -> tuple[str, dict[str, Any]]:
    dominant_axis = str(base_bundle.rsu_metadata.get("dominant_axis", base_bundle.rsu_metadata.get("chosen_rsu_axis", "x")))
    coverage = max(float(rsu_coverage_radius), 1.0)
    rsu_count = max(1, int(rsu_count))
    return (
        f"custom:axis={dominant_axis},count={rsu_count},coverage={coverage}",
        {
            "proxy_scaling": True,
            "proxy_note": "rsu_count / rsu_coverage_radius ???? custom layout ? proxy scaling?",
            "rsu_count": rsu_count,
            "rsu_coverage_radius": coverage,
        },
    )
