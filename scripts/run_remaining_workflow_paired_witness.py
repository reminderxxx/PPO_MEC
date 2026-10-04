"""Run the pre-frozen bounded remaining-workflow paired witness.

Only normal ``GymVecEnv`` actions are used.  The script does not mutate cache
residency, workflow progress, or migration state after reset.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_native_typed_cache_probe import build_audit_catalog  # noqa: E402
from src.data.mobility.replay_provider import ReplayProvider  # noqa: E402
from src.data.model_catalog.adapter_catalog import RSUTypedCacheProfile  # noqa: E402
from src.envs.core.cache_eviction import TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU  # noqa: E402
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv  # noqa: E402
from src.envs.specs import RSUState, WorkflowGraphState, WorkflowNode  # noqa: E402
from src.envs.wrappers.gym_vec_env import GymVecEnv  # noqa: E402


MIB = 1024 * 1024
RULE_IDS = (
    "remaining_reuse_information_ablated",
    "native_two_step_lookahead",
    "remaining_workflow_reuse",
    "remaining_workflow_handoff_prepare",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workflow_for(instance_id: str) -> WorkflowGraphState:
    adapters_by_instance = {
        "tail_length__short": ["b0.a0"],
        "tail_length__long": ["b0.a0", "b0.a0", "b0.a0"],
        "tail_adapter_structure__short": ["b0.a0", "b1.a0"],
        "tail_adapter_structure__long": ["b0.a0", "b0.a0", "b1.a0"],
    }
    adapters = adapters_by_instance[instance_id]
    nodes = []
    edges = []
    for index, adapter_id in enumerate(adapters):
        node_id = f"n{index}"
        previous = f"n{index - 1}" if index else None
        following = f"n{index + 1}" if index + 1 < len(adapters) else None
        base_id = adapter_id.split(".", 1)[0]
        nodes.append(
            WorkflowNode(
                node_id=node_id,
                node_name=f"bounded paired witness {adapter_id}",
                required_base_model=base_id,
                required_adapter=adapter_id,
                input_size=1,
                output_size=1,
                predecessors=[previous] if previous else [],
                successors=[following] if following else [],
            )
        )
        if previous:
            edges.append((previous, node_id))
    return WorkflowGraphState(
        workflow_id=f"remaining_workflow_witness::{instance_id}",
        nodes=nodes,
        edges=edges,
        execution_order=[node.node_id for node in nodes],
        current_node_id="n0",
    )


def build_env(instance_id: str) -> GymVecEnv:
    catalog = build_audit_catalog(sharing_enabled=True)
    catalog.rsu_typed_cache_profiles = [
        RSUTypedCacheProfile(rsu_id="rsu_a", resident_object_ids=[]),
        RSUTypedCacheProfile(rsu_id="rsu_b", resident_object_ids=[]),
    ]
    frames = [
        {
            "time_index": index,
            "vehicles": [
                {
                    "vehicle_id": "veh_1",
                    "position_x": position,
                    "position_y": 0.0,
                    "speed": 40.0,
                    "base_model_id": "b0",
                    "active_workflow_id": f"remaining_workflow_witness::{instance_id}",
                }
            ],
        }
        for index, position in enumerate((0.0, 40.0, 80.0, 100.0, 100.0, 100.0))
    ]
    core = VecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(trajectory_frames=frames),
        workflow_state=workflow_for(instance_id),
        adapter_catalog=catalog,
        rsu_states=[
            RSUState("rsu_a", 0.0, 0.0, 55.0),
            RSUState("rsu_b", 100.0, 0.0, 55.0),
        ],
        max_steps=4,
        cache_capacity_profile={
            "model_cache_profile_id": "typed_base_adapter_state_v1",
            "enabled": True,
            "unit": "mb",
            "capacity_mb": 136.0,
            "count_base_model_separately": True,
            "eviction_policy": "lru",
            "eviction_policy_seed": None,
            "typed_eviction_semantics": TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU,
            "telemetry_enabled": True,
        },
    )
    return GymVecEnv(core)


def remaining_adapters(state: dict[str, Any]) -> tuple[str | None, list[str]]:
    workflow = state["workflow"]
    node_by_id = {node["node_id"]: node for node in workflow["nodes"]}
    current_id = workflow.get("current_node_id")
    order = list(workflow["execution_order"])
    if current_id not in order:
        return None, []
    index = order.index(current_id)
    current = node_by_id[current_id]["required_adapter"]
    tail = [node_by_id[node_id]["required_adapter"] for node_id in order[index + 1 :]]
    return str(current), [str(item) for item in tail]


def choose_action(rule_id: str, state: dict[str, Any], action_mask: list[bool]) -> tuple[int, dict[str, Any]]:
    current, tail = remaining_adapters(state)
    if current is None:
        return 0, {"reason": "missing_current_node", "tail_visible": []}
    if rule_id == "remaining_reuse_information_ablated":
        proposed = 0
        visible_tail: list[str] = []
        reason = "remaining_tail_hidden"
    elif rule_id == "native_two_step_lookahead":
        visible_tail = tail[:1]
        proposed = 1 if visible_tail and visible_tail[0] == current else 0
        reason = "next_node_reuses_current_adapter" if proposed == 1 else "next_node_does_not_reuse"
    elif rule_id == "remaining_workflow_reuse":
        visible_tail = tail
        proposed = 1 if current in visible_tail else 0
        reason = "later_tail_reuses_current_adapter" if proposed == 1 else "no_later_reuse"
    elif rule_id == "remaining_workflow_handoff_prepare":
        visible_tail = tail
        proposed = 4 if current in visible_tail else 0
        reason = "later_tail_reuses_current_adapter_prepare_state" if proposed == 4 else "no_later_reuse"
    else:
        raise ValueError(f"unknown rule: {rule_id}")
    selected = proposed if proposed < len(action_mask) and action_mask[proposed] else 0
    return selected, {
        "current_adapter": current,
        "tail_visible": visible_tail,
        "proposed_action": proposed,
        "selected_action": selected,
        "proposed_action_allowed": bool(proposed < len(action_mask) and action_mask[proposed]),
        "fallback_to_action0": selected != proposed,
        "reason": reason,
    }


def run_episode(instance_id: str, rule_id: str) -> dict[str, Any]:
    env = build_env(instance_id)
    _, reset_info = env.reset(seed=7)
    rows = []
    state = deepcopy(reset_info["semantic_state"])
    action_mask = list(reset_info["action_mask"])
    terminated = False
    truncated = False
    for step_index in range(4):
        action, rationale = choose_action(rule_id, state, action_mask)
        workflow_before = deepcopy(state["workflow"])
        _, reward, terminated, truncated, info = env.step(action)
        state = deepcopy(info["semantic_state"])
        action_mask = list(info["action_mask"])
        event = deepcopy(info["cache_event"])
        completed_now = [
            node_id
            for node_id in state["workflow"]["completed_node_ids"]
            if node_id not in workflow_before["completed_node_ids"]
        ]
        rows.append(
            {
                "step_index": step_index + 1,
                "decision_rationale": rationale,
                "action_id": action,
                "action_name": info["action_name"],
                "action_invalid": info["action_invalid"],
                "control_action": info["control_action"],
                "pre_action_workflow": workflow_before,
                "post_action_workflow": deepcopy(state["workflow"]),
                "completed_node_ids_this_step": completed_now,
                "reward": reward,
                "cache_event": event,
                "migration_prepare_realized": info["metrics_protocol"].get("migration_prepare_realized"),
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        if terminated or truncated:
            break

    prepared = []
    for row in rows:
        if row["action_id"] not in {1, 4}:
            continue
        event = row["cache_event"]
        target = event.get("cache_target_rsu_id")
        adapter = event.get("adapter_id")
        later_use = any(
            later["step_index"] > row["step_index"]
            and later["cache_event"].get("served_rsu_id") == target
            and later["cache_event"].get("adapter_id") == adapter
            and later["cache_event"].get("service_success") is True
            for later in rows
        )
        state_realized = any(
            later["step_index"] >= row["step_index"]
            and later["cache_event"].get("migration_realized") is True
            for later in rows
        )
        prepared.append(
            {
                "step_index": row["step_index"],
                "action_id": row["action_id"],
                "target_rsu_id": target,
                "adapter_id": adapter,
                "later_model_service_use": later_use,
                "state_prepare_realized": state_realized,
                "unused_model_preparation": not later_use,
            }
        )

    transfer_mib: dict[str, float] = {}
    evicted_mib: dict[str, float] = {}
    for row in rows:
        event = row["cache_event"]
        for object_type, value in (event.get("transfer_mb_by_type") or {}).items():
            transfer_mib[object_type] = transfer_mib.get(object_type, 0.0) + float(value)
        for object_type, value in (event.get("evicted_mb_by_type") or {}).items():
            evicted_mib[object_type] = evicted_mib.get(object_type, 0.0) + float(value)
    workflow = state["workflow"]
    return {
        "instance_id": instance_id,
        "rule_id": rule_id,
        "seed": 7,
        "fixed_step_cap": 4,
        "initial_action_mask": reset_info["action_mask"],
        "initial_semantic_state": reset_info["semantic_state"],
        "steps": rows,
        "summary": {
            "step_count": len(rows),
            "node_completion_count": len(workflow["completed_node_ids"]),
            "node_total": len(workflow["execution_order"]),
            "workflow_completed": bool(workflow["is_completed"]),
            "service_success_count": sum(row["cache_event"]["service_success"] for row in rows),
            "service_failure_count": sum(not row["cache_event"]["service_success"] for row in rows),
            "transfer_mib_by_type": dict(sorted(transfer_mib.items())),
            "transfer_bytes_by_type": {
                key: int(round(value * MIB)) for key, value in sorted(transfer_mib.items())
            },
            "evicted_mib_by_type": dict(sorted(evicted_mib.items())),
            "unused_model_preparation_count": sum(item["unused_model_preparation"] for item in prepared),
            "preparation_records": prepared,
            "invalid_action_count": sum(row["action_invalid"] for row in rows),
            "state_transfer_interpretation": "native simulated fixed-size workflow-state field, not local measured application-state bytes",
        },
    }


def build_output(design_path: Path) -> dict[str, Any]:
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if design["budget"]["planned_episodes"] != 16:
        raise ValueError("paired witness design is not the frozen v1.1 16-episode plan")
    instances = (
        "tail_length__short",
        "tail_length__long",
        "tail_adapter_structure__short",
        "tail_adapter_structure__long",
    )
    episodes = [run_episode(instance, rule) for instance in instances for rule in RULE_IDS]
    if len(episodes) > int(design["budget"]["maximum_episodes"]):
        raise AssertionError("episode budget exceeded")
    by_instance_rule = {(row["instance_id"], row["rule_id"]): row for row in episodes}
    decision_changes = []
    for instance in instances:
        ablated = by_instance_rule[(instance, "remaining_reuse_information_ablated")]
        ablated_action = ablated["steps"][0]["action_id"]
        for rule in RULE_IDS[1:]:
            episode = by_instance_rule[(instance, rule)]
            action = episode["steps"][0]["action_id"]
            decision_changes.append(
                {
                    "instance_id": instance,
                    "comparison": f"{rule}_vs_information_ablated",
                    "ablated_action": ablated_action,
                    "candidate_action": action,
                    "decision_changed": action != ablated_action,
                    "candidate_information": episode["steps"][0]["decision_rationale"]["tail_visible"],
                    "completed_nodes_delta": episode["summary"]["node_completion_count"] - ablated["summary"]["node_completion_count"],
                    "service_failures_delta": episode["summary"]["service_failure_count"] - ablated["summary"]["service_failure_count"],
                    "base_transfer_bytes_delta": episode["summary"]["transfer_bytes_by_type"].get("base_model", 0) - ablated["summary"]["transfer_bytes_by_type"].get("base_model", 0),
                    "adapter_transfer_bytes_delta": episode["summary"]["transfer_bytes_by_type"].get("adapter", 0) - ablated["summary"]["transfer_bytes_by_type"].get("adapter", 0),
                    "workflow_state_transfer_bytes_delta": episode["summary"]["transfer_bytes_by_type"].get("workflow_state", 0) - ablated["summary"]["transfer_bytes_by_type"].get("workflow_state", 0),
                }
            )
    two_step_matches_full = all(
        by_instance_rule[(instance, "native_two_step_lookahead")]["steps"][0]["action_id"]
        == by_instance_rule[(instance, "remaining_workflow_reuse")]["steps"][0]["action_id"]
        for instance in instances
    )
    return {
        "artifact_run_id": design["artifact_run_id"],
        "evidence_type": "bounded_native_paired_witness",
        "design_path": str(design_path),
        "design_sha256": sha256_file(design_path),
        "episode_count": len(episodes),
        "episode_budget": design["budget"],
        "episodes": episodes,
        "decision_changes": decision_changes,
        "two_step_matches_full_remaining_on_all_designed_instances": two_step_matches_full,
        "statistical_claim": "none_descriptive_only",
        "future_random_outcome_leakage": False,
        "manual_state_or_cache_mutation_during_episode": False,
        "migration_and_forwarding_limit": "explicit migrate and continue-old-RSU/result-forward actions are unreachable; only native prepare is exercised",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = build_output(args.design)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "episode_count": output["episode_count"],
        "decision_change_count": sum(row["decision_changed"] for row in output["decision_changes"]),
        "two_step_matches_full": output["two_step_matches_full_remaining_on_all_designed_instances"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
