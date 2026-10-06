"""Measurement-calibrated small continuous-workflow simulation.

This module deliberately stays separate from the production environment.  It
reuses the five-action semantic contract and dependency-safe typed-cache
semantics, but it never invokes a real model during training.  Every episode is
created from a frozen manifest row so learned agents and the cost rule see the
same cache, mobility, DAG, and action capabilities.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.envs.specs.action_schema import ActionMaskBuilder


ACTION_NAMES = {
    0: "current_rsu_cache_fill",
    1: "predictive_next_rsu_prefetch",
    2: "vehicle_fallback",
    3: "current_rsu_steady_offload",
    4: "handoff_migration_prepare",
}

ORIGINAL_REWARD_PROFILE = "original_reward_v1"
SERVICE_ALIGNED_REWARD_PROFILE = "service_aligned_v1"


def _network_seconds(byte_count: int, mbps: float, fixed_seconds: float) -> float:
    if byte_count <= 0:
        return 0.0
    return float(byte_count) * 8.0 / (float(mbps) * 1_000_000.0) + float(fixed_seconds)


@dataclass
class _RSUCache:
    capacity_bytes: int
    residents: list[str] = field(default_factory=list)
    last_used: dict[str, int] = field(default_factory=dict)


class CalibratedContinuousWorkflowEnv:
    """A compact event-driven environment for matched pilot training."""

    def __init__(self, config: dict[str, Any], instance: dict[str, Any]) -> None:
        self.config = deepcopy(config)
        self.instance = deepcopy(instance)
        self._mask_builder = ActionMaskBuilder()
        self._object_catalog = deepcopy(config["object_catalog"])
        self._adapter_to_bundle = deepcopy(config["adapter_to_bundle"])
        self._rng = np.random.default_rng(int(instance.get("episode_seed", 0)))
        self.reset()

    def clone(self) -> "CalibratedContinuousWorkflowEnv":
        return deepcopy(self)

    def clone_for_decision_model(self) -> "CalibratedContinuousWorkflowEnv":
        """Clone state while forcing previews to use decision-time estimates."""
        clone = deepcopy(self)
        clone._decision_model_mode = True
        return clone

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self._decision_model_mode = False
        self.nodes = deepcopy(self.instance["nodes"])
        self.node_map = {str(node["node_id"]): node for node in self.nodes}
        self.execution_order = [str(node_id) for node_id in self.instance["execution_order"]]
        self.completed: list[str] = []
        self.node_index = 0
        self.step_index = 0
        self.clock_seconds = 0.0
        self.last_execution_rsu = str(self.instance["rsu_sequence"][0])
        self.prepared_state: dict[str, dict[str, Any]] = {}
        self.metrics = {
            "reward": 0.0,
            "completed_nodes": 0,
            "workflow_completed": 0,
            "deadline_violations": 0,
            "deadline_missed": 0,
            "on_time_workflow_completed": 0,
            "late_workflow_completed": 0,
            "unfinished_after_deadline": 0,
            "service_failures": 0,
            "failed_service_attempt_seconds_proxy": 0.0,
            "handoff_count": 0,
            "handoff_failures": 0,
            "migration_attempts": 0,
            "migration_successes": 0,
            "model_transfer_bytes": 0,
            "state_transfer_bytes": 0,
            "input_transfer_bytes": 0,
            "recompute_seconds": 0.0,
            "model_load_seconds": 0.0,
            "state_restore_seconds": 0.0,
            "transfer_seconds": 0.0,
            "modeled_completion_seconds": 0.0,
            "cache_evictions": 0,
            "cache_rejections": 0,
            "vehicle_fallback_count": 0,
            "invalid_action_count": 0,
            "reward_node_completion": 0.0,
            "reward_workflow_completion": 0.0,
            "reward_time_penalty": 0.0,
            "reward_transfer_penalty": 0.0,
            "reward_failure_penalty": 0.0,
            "reward_deadline_penalty": 0.0,
            "reward_service_operation_time_penalty": 0.0,
            "reward_recompute_penalty": 0.0,
            "reward_on_time_completion": 0.0,
        }
        self._deadline_miss_recorded = False
        self._reward_totals_by_profile = {
            profile: 0.0
            for profile in (ORIGINAL_REWARD_PROFILE, SERVICE_ALIGNED_REWARD_PROFILE)
            if self._reward_profile_available(profile)
        }
        capacity = int(self.instance["cache_capacity_bytes"])
        self.caches = {
            rsu_id: _RSUCache(capacity_bytes=capacity)
            for rsu_id in self.instance["rsu_ids"]
        }
        for rsu_id, object_ids in self.instance["initial_residents"].items():
            cache = self.caches[str(rsu_id)]
            for object_id in object_ids:
                if object_id not in cache.residents:
                    cache.residents.append(str(object_id))
                    cache.last_used[str(object_id)] = 0
        self._validate_all_caches()
        return self._observation(), self._info()

    def _mobility_index(self) -> int:
        progression = str(
            self.config.get("mobility_progression", "completed_node_index")
        )
        if progression == "decision_step_index":
            return int(self.step_index)
        return int(self.node_index)

    @property
    def terminated(self) -> bool:
        return self.node_index >= len(self.execution_order)

    def valid_actions(self) -> list[int]:
        info = self._info()
        return [index for index, allowed in enumerate(info["action_mask"]) if allowed]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.terminated:
            return self._observation(), 0.0, True, False, self._info()

        semantic_state = self._semantic_state()
        mask = self._mask_builder.build_mask(semantic_state)
        action = int(action)
        if action < 0 or action >= len(mask) or not mask[action]:
            self.metrics["invalid_action_count"] += 1
            action = 3 if mask[3] else next(index for index, valid in enumerate(mask) if valid)

        node = self._current_node()
        current_rsu = self._current_rsu_id()
        next_rsu = self._predicted_handoff_target()
        previous_rsu = self.last_execution_rsu
        handoff = bool(previous_rsu and current_rsu != previous_rsu)
        if handoff:
            self.metrics["handoff_count"] += 1

        event: dict[str, Any] = {
            "step_index": self.step_index,
            "node_id": node["node_id"],
            "action": action,
            "action_name": ACTION_NAMES[action],
            "current_rsu_id": current_rsu,
            "predicted_handoff_target_rsu_id": next_rsu,
            "handoff": handoff,
            "cache_events": [],
            "state_transfer": None,
        }
        step_cost = 0.0
        model_bytes = 0
        state_bytes = 0
        input_bytes = 0
        recompute_seconds = 0.0
        service_operation_seconds = 0.0
        model_load_seconds = 0.0
        state_restore_seconds = 0.0
        failure = False
        migration_success = False
        staged_migration: dict[str, Any] | None = None

        if action == 1 and next_rsu:
            cache_event = self._admit_bundle(next_rsu, str(node["required_adapter"]))
            event["cache_events"].append(cache_event)
            if cache_event["committed"]:
                transfer_seconds = self._model_transfer_seconds(cache_event)
                if transfer_seconds <= self._contact_budget_seconds():
                    model_bytes += int(cache_event["transfer_bytes"])
                    model_load_seconds += float(cache_event["load_seconds"])
                else:
                    self._rollback_cache_event(cache_event)
                    cache_event["committed"] = False
                    cache_event["reason"] = "contact_budget_exceeded"

        if action == 4 and next_rsu:
            self.metrics["migration_attempts"] += 1
            cache_event = self._admit_bundle(next_rsu, str(node["required_adapter"]))
            event["cache_events"].append(cache_event)
            package_bytes = self._state_package_bytes(node)
            model_prepare_seconds = self._model_transfer_seconds(cache_event)
            state_prepare_seconds = _network_seconds(
                package_bytes,
                self._effective_link_mbps(),
                float(self.config["link"]["fixed_seconds"]),
            ) + float(self.config["measured_time_seconds"]["state_restore_overhead"])
            if cache_event["committed"] and model_prepare_seconds + state_prepare_seconds <= self._contact_budget_seconds():
                model_bytes += int(cache_event["transfer_bytes"])
                model_load_seconds += float(cache_event["load_seconds"])
                staged_migration = {
                    "target_rsu_id": str(next_rsu),
                    "bytes": package_bytes,
                }
                event["state_transfer"] = {
                    "status": "staged_pending_current_node_completion",
                    **staged_migration,
                }
            else:
                if cache_event["committed"]:
                    self._rollback_cache_event(cache_event)
                    cache_event["committed"] = False
                    cache_event["reason"] = "contact_budget_exceeded"
                event["state_transfer"] = {
                    "status": "not_ready_before_handoff",
                    "target_rsu_id": next_rsu,
                    "bytes": package_bytes,
                }

        service_rsu: str | None = None if action == 2 else current_rsu
        if action == 2:
            self.metrics["vehicle_fallback_count"] += 1
            input_bytes += int(node["input_bytes"])
            service_operation_seconds += float(self.config["vehicle"]["fallback_seconds"])
            step_cost += service_operation_seconds
        else:
            bundle_ready = self._bundle_ready(current_rsu, str(node["required_adapter"]))
            if action == 0 and not bundle_ready:
                cache_event = self._admit_bundle(current_rsu, str(node["required_adapter"]))
                event["cache_events"].append(cache_event)
                if cache_event["committed"]:
                    model_bytes += int(cache_event["transfer_bytes"])
                    model_load_seconds += float(cache_event["load_seconds"])
                    bundle_ready = True
            if not bundle_ready:
                failure = True

        state_ready = True
        if handoff and action != 2:
            prepared = self.prepared_state.get(current_rsu)
            state_ready = bool(
                prepared
                and prepared.get("workflow_id") == self.instance["workflow_id"]
                and set(self.completed).issubset(set(prepared.get("completed_node_ids", [])))
            )
            if not state_ready and not failure:
                recompute_seconds = self._recompute_seconds(node)
                input_bytes += int(node["input_bytes"])
                self.metrics["handoff_failures"] += 1

        if failure:
            self.metrics["service_failures"] += 1
            step_cost += float(self.config["objective"]["failed_service_seconds"])
        else:
            compute_seconds = float(node["compute_seconds"])
            service_operation_seconds += compute_seconds
            step_cost += compute_seconds + recompute_seconds
            self.completed.append(str(node["node_id"]))
            self.node_index += 1
            self.metrics["completed_nodes"] += 1
            self.last_execution_rsu = current_rsu if service_rsu else self.last_execution_rsu
            self._touch_bundle(current_rsu, str(node["required_adapter"])) if service_rsu else None
            if staged_migration is not None:
                migration_success = True
                state_bytes += int(staged_migration["bytes"])
                state_restore_seconds += float(
                    self.config["measured_time_seconds"]["state_restore_overhead"]
                )
                target_rsu = str(staged_migration["target_rsu_id"])
                self.prepared_state[target_rsu] = {
                    "completed_node_ids": list(self.completed),
                    "workflow_id": self.instance["workflow_id"],
                    "package_bytes": int(staged_migration["bytes"]),
                }
                self.metrics["migration_successes"] += 1
                event["state_transfer"] = {
                    "status": "prepared_after_current_node_completion",
                    **staged_migration,
                }

        transfer_seconds = sum(
            _network_seconds(
                byte_count,
                self._effective_link_mbps(),
                float(self.config["link"]["fixed_seconds"]),
            )
            for byte_count in (model_bytes, state_bytes, input_bytes)
        )
        step_cost += transfer_seconds + model_load_seconds + state_restore_seconds
        service_operation_seconds += model_load_seconds + state_restore_seconds
        self.clock_seconds += step_cost
        self.step_index += 1

        self.metrics["model_transfer_bytes"] += model_bytes
        self.metrics["state_transfer_bytes"] += state_bytes
        self.metrics["input_transfer_bytes"] += input_bytes
        self.metrics["recompute_seconds"] += recompute_seconds
        self.metrics["model_load_seconds"] += model_load_seconds
        self.metrics["state_restore_seconds"] += state_restore_seconds
        self.metrics["transfer_seconds"] += transfer_seconds
        self.metrics["modeled_completion_seconds"] = self.clock_seconds
        deadline_miss_event = bool(
            not self._deadline_miss_recorded
            and self.clock_seconds > float(self.instance["deadline_seconds"])
        )
        if deadline_miss_event:
            self._deadline_miss_recorded = True
            self.metrics["deadline_missed"] = 1
            self.metrics["deadline_violations"] = 1
        if self.terminated:
            self.metrics["workflow_completed"] = 1
            if self._deadline_miss_recorded:
                self.metrics["late_workflow_completed"] = 1
            else:
                self.metrics["on_time_workflow_completed"] = 1
        elif self._deadline_miss_recorded:
            self.metrics["unfinished_after_deadline"] = 1
        self.metrics["failed_service_attempt_seconds_proxy"] = float(
            self.metrics["service_failures"]
        ) * float(self.config["objective"]["failed_service_seconds"])

        reward_components_by_profile = {
            profile: self._reward_components(
                profile=profile,
                completed=not failure,
                step_cost=step_cost,
                transfer_bytes=model_bytes + state_bytes + input_bytes,
                failure=failure,
                terminal=self.terminated,
                deadline_miss_event=deadline_miss_event,
                service_operation_seconds=service_operation_seconds,
                recompute_seconds=recompute_seconds,
            )
            for profile in (ORIGINAL_REWARD_PROFILE, SERVICE_ALIGNED_REWARD_PROFILE)
            if self._reward_profile_available(profile)
        }
        active_reward_profile = str(
            self.config.get("reward_profile", ORIGINAL_REWARD_PROFILE)
        )
        if active_reward_profile not in reward_components_by_profile:
            raise ValueError(f"unknown reward profile: {active_reward_profile}")
        reward_components = reward_components_by_profile[active_reward_profile]
        reward = float(sum(reward_components.values()))
        for profile, components in reward_components_by_profile.items():
            self._reward_totals_by_profile[profile] += float(sum(components.values()))
        legacy_reward_components = {
            key: float(value)
            for key, value in reward_components.items()
            if key in self.metrics
        }
        for key in self.metrics:
            if key.startswith("reward_") and key not in legacy_reward_components:
                legacy_reward_components[key] = 0.0
        reward_components = legacy_reward_components
        self.metrics["reward"] += reward
        for key, value in reward_components.items():
            self.metrics[key] += value
        event.update(
            {
                "service_completed": not failure,
                "state_ready": state_ready,
                "migration_success": migration_success,
                "model_transfer_bytes": model_bytes,
                "state_transfer_bytes": state_bytes,
                "input_transfer_bytes": input_bytes,
                "recompute_seconds": recompute_seconds,
                "model_load_seconds": model_load_seconds,
                "state_restore_seconds": state_restore_seconds,
                "transfer_seconds": transfer_seconds,
                "step_cost_seconds": step_cost,
                "service_operation_seconds": service_operation_seconds,
                "clock_seconds_after": self.clock_seconds,
                "deadline_seconds": float(self.instance["deadline_seconds"]),
                "deadline_miss_event": deadline_miss_event,
                "active_reward_profile": active_reward_profile,
                "reward": reward,
                "reward_components": reward_components,
                "reward_components_by_profile": reward_components_by_profile,
            }
        )
        info = self._info()
        info["transition"] = event
        info["episode_metrics"] = deepcopy(self.metrics)
        max_steps = int(self.instance["max_steps"])
        truncated = bool(self.step_index >= max_steps and not self.terminated)
        return self._observation(), reward, self.terminated, truncated, info

    def summary(self) -> dict[str, Any]:
        result = deepcopy(self.metrics)
        result.update(
            {
                "workflow_id": self.instance["workflow_id"],
                "window_id": self.instance["window_id"],
                "design_id": self.instance["design_id"],
                "split": self.instance["split"],
                "node_count": len(self.nodes),
                "total_transfer_bytes": int(
                    self.metrics["model_transfer_bytes"]
                    + self.metrics["state_transfer_bytes"]
                    + self.metrics["input_transfer_bytes"]
                ),
                "terminated": self.terminated,
                "steps": self.step_index,
                "deadline_seconds": float(self.instance["deadline_seconds"]),
                "reward_profile": str(
                    self.config.get("reward_profile", ORIGINAL_REWARD_PROFILE)
                ),
                "reward_totals_by_profile": deepcopy(
                    self._reward_totals_by_profile
                ),
            }
        )
        return result

    def _current_node(self) -> dict[str, Any]:
        return self.node_map[self.execution_order[self.node_index]]

    def _current_rsu_id(self) -> str:
        sequence = self.instance["rsu_sequence"]
        return str(sequence[min(self._mobility_index(), len(sequence) - 1)])

    def _predicted_sequence(self) -> list[str]:
        sequence = self.instance.get("predicted_rsu_sequence", self.instance["rsu_sequence"])
        horizon = int(self.config["prediction_horizon"])
        start = min(self._mobility_index(), len(sequence) - 1)
        values = [str(item) for item in sequence[start + 1 : start + 1 + horizon]]
        if not values:
            values = [str(sequence[-1])]
        while len(values) < horizon:
            values.append(values[-1])
        return values

    def _predicted_handoff_target(self) -> str | None:
        current = self._current_rsu_id()
        for candidate in self._predicted_sequence():
            if candidate != current:
                return candidate
        return None

    def _contact_budget_seconds(self) -> float:
        current = self._current_rsu_id()
        countdown = 1
        for index, candidate in enumerate(self._predicted_sequence(), start=1):
            if candidate != current:
                countdown = index
                break
        return float(countdown) * float(self.config["mobility_abstraction"]["decision_step_seconds"])

    def _state_package_bytes(self, node: dict[str, Any]) -> int:
        return int(node["state_bytes"])

    def _recompute_seconds(self, node: dict[str, Any]) -> float:
        required: set[str] = set()

        def visit(node_id: str) -> None:
            for predecessor in self.node_map[node_id].get("predecessors", []):
                predecessor = str(predecessor)
                if predecessor in self.completed and predecessor not in required:
                    required.add(predecessor)
                    visit(predecessor)

        visit(str(node["node_id"]))
        if not required and self.completed:
            required.add(self.completed[-1])
        relative_prefix_work = sum(
            float(self.node_map[item]["compute_seconds"]) for item in required
        ) / max(float(self.config["measured_time_seconds"]["node_compute_reference"]), 1e-9)
        return float(self.config["measured_time_seconds"]["prefix_recompute_reference"]) * relative_prefix_work

    def _bundle_ids(self, adapter_id: str) -> list[str]:
        return [str(item) for item in self._adapter_to_bundle[adapter_id]]

    def _bundle_ready(self, rsu_id: str, adapter_id: str) -> bool:
        residents = set(self.caches[str(rsu_id)].residents)
        return set(self._bundle_ids(adapter_id)).issubset(residents)

    def _touch_bundle(self, rsu_id: str, adapter_id: str) -> None:
        cache = self.caches[str(rsu_id)]
        for object_id in self._bundle_ids(adapter_id):
            if object_id in cache.residents:
                cache.last_used[object_id] = self.step_index

    def _resident_bytes(self, cache: _RSUCache) -> int:
        return sum(int(self._object_catalog[item]["resident_bytes"]) for item in cache.residents)

    def _dependency_safe(self, object_id: str, residents: set[str]) -> bool:
        row = self._object_catalog[object_id]
        if row["object_type"] == "adapter":
            return True
        return not any(
            object_id in self._object_catalog[candidate].get("dependency_ids", [])
            for candidate in residents
            if candidate != object_id
        )

    def _admit_bundle(self, rsu_id: str, adapter_id: str) -> dict[str, Any]:
        cache = self.caches[str(rsu_id)]
        before = list(cache.residents)
        before_last_used = dict(cache.last_used)
        bundle = self._bundle_ids(adapter_id)
        missing = [item for item in bundle if item not in cache.residents]
        if not missing:
            self._touch_bundle(rsu_id, adapter_id)
            return {
                "rsu_id": rsu_id,
                "adapter_id": adapter_id,
                "committed": True,
                "reason": "noop_all_resident",
                "victims": [],
                "admitted": [],
                "transfer_bytes": 0,
                "load_seconds": 0.0,
                "before_residents": before,
                "after_residents": list(cache.residents),
                "before_last_used": before_last_used,
            }
        required_bytes = sum(int(self._object_catalog[item]["resident_bytes"]) for item in missing)
        if sum(int(self._object_catalog[item]["resident_bytes"]) for item in bundle) > cache.capacity_bytes:
            self.metrics["cache_rejections"] += 1
            return {
                "rsu_id": rsu_id,
                "adapter_id": adapter_id,
                "committed": False,
                "reason": "dependency_bundle_exceeds_total_capacity",
                "victims": [],
                "admitted": [],
                "transfer_bytes": 0,
                "load_seconds": 0.0,
                "before_residents": before,
                "after_residents": before,
                "before_last_used": before_last_used,
            }
        shadow = list(cache.residents)
        victims: list[str] = []
        while sum(int(self._object_catalog[item]["resident_bytes"]) for item in shadow) + required_bytes > cache.capacity_bytes:
            shadow_set = set(shadow)
            candidates = [
                item
                for item in shadow
                if item not in bundle and self._dependency_safe(item, shadow_set)
            ]
            if not candidates:
                self.metrics["cache_rejections"] += 1
                return {
                    "rsu_id": rsu_id,
                    "adapter_id": adapter_id,
                    "committed": False,
                    "reason": "insufficient_dependency_safe_evictable_capacity",
                    "victims": victims,
                    "admitted": [],
                    "transfer_bytes": 0,
                    "load_seconds": 0.0,
                    "before_residents": before,
                    "after_residents": before,
                    "before_last_used": before_last_used,
                }
            victim = min(candidates, key=lambda item: (cache.last_used.get(item, -1), item))
            shadow.remove(victim)
            victims.append(victim)
        cache.residents = shadow + missing
        for victim in victims:
            cache.last_used.pop(victim, None)
        for item in missing:
            cache.last_used[item] = self.step_index
        self.metrics["cache_evictions"] += len(victims)
        transfer_bytes = sum(int(self._object_catalog[item]["transfer_bytes"]) for item in missing)
        load_seconds = sum(float(self._object_catalog[item]["load_seconds"]) for item in missing)
        self._validate_cache(cache)
        return {
            "rsu_id": rsu_id,
            "adapter_id": adapter_id,
            "committed": True,
            "reason": "committed",
            "victims": victims,
            "admitted": missing,
            "transfer_bytes": transfer_bytes,
            "load_seconds": load_seconds,
            "before_residents": before,
            "after_residents": list(cache.residents),
            "before_last_used": before_last_used,
        }

    def _rollback_cache_event(self, event: dict[str, Any]) -> None:
        cache = self.caches[str(event["rsu_id"])]
        cache.residents = list(event["before_residents"])
        cache.last_used = dict(event["before_last_used"])
        self.metrics["cache_evictions"] -= len(event.get("victims", []))

    def _model_transfer_seconds(self, event: dict[str, Any]) -> float:
        if not event.get("committed"):
            return float("inf")
        return _network_seconds(
            int(event["transfer_bytes"]),
            self._effective_link_mbps(),
            float(self.config["link"]["fixed_seconds"]),
        ) + float(event["load_seconds"])

    def _effective_link_mbps(self) -> float:
        profile = self.instance.get("link_profile", {})
        if self._decision_model_mode:
            return float(profile.get("estimated_mbps", self.config["link"]["mbps"]))
        return float(profile.get("actual_mbps", self.config["link"]["mbps"]))

    def _validate_cache(self, cache: _RSUCache) -> None:
        residents = set(cache.residents)
        if len(residents) != len(cache.residents):
            raise RuntimeError("duplicate typed cache resident")
        if self._resident_bytes(cache) > cache.capacity_bytes:
            raise RuntimeError("typed cache capacity exceeded")
        for object_id in residents:
            dependencies = set(self._object_catalog[object_id].get("dependency_ids", []))
            if not dependencies.issubset(residents):
                raise RuntimeError("typed cache orphan dependency")

    def _validate_all_caches(self) -> None:
        for cache in self.caches.values():
            self._validate_cache(cache)

    def _reward_profile_available(self, profile: str) -> bool:
        if profile == ORIGINAL_REWARD_PROFILE:
            return True
        return profile in self.config["objective"].get("reward_profiles", {})

    def _reward_components(
        self,
        *,
        profile: str,
        completed: bool,
        step_cost: float,
        transfer_bytes: int,
        failure: bool,
        terminal: bool,
        deadline_miss_event: bool,
        service_operation_seconds: float,
        recompute_seconds: float,
    ) -> dict[str, float]:
        objective = self.config["objective"]
        if profile == ORIGINAL_REWARD_PROFILE:
            return {
                "reward_node_completion": float(objective["node_completion_reward"])
                if completed
                else 0.0,
                "reward_workflow_completion": float(
                    objective["workflow_completion_reward"]
                )
                if terminal
                else 0.0,
                "reward_time_penalty": -float(objective["time_weight"])
                * float(step_cost),
                "reward_transfer_penalty": -float(objective["transfer_gib_weight"])
                * float(transfer_bytes)
                / float(1024**3),
                "reward_failure_penalty": -float(objective["failure_penalty"])
                if failure
                else 0.0,
                "reward_deadline_penalty": (
                    -float(objective["deadline_penalty"])
                    if terminal
                    and self.clock_seconds > float(self.instance["deadline_seconds"])
                    else 0.0
                ),
            }
        if profile != SERVICE_ALIGNED_REWARD_PROFILE:
            raise ValueError(f"unknown reward profile: {profile}")
        aligned = objective["reward_profiles"][profile]
        return {
            "reward_node_completion": float(aligned["node_completion_reward"])
            if completed
            else 0.0,
            "reward_workflow_completion": float(
                aligned["workflow_completion_reward"]
            )
            if terminal
            else 0.0,
            "reward_time_penalty": 0.0,
            "reward_service_operation_time_penalty": -float(
                aligned["service_operation_second_weight"]
            )
            * float(service_operation_seconds),
            "reward_transfer_penalty": -float(aligned["transfer_gib_weight"])
            * float(transfer_bytes)
            / float(1024**3),
            "reward_recompute_penalty": -float(
                aligned["recompute_second_weight"]
            )
            * float(recompute_seconds),
            "reward_failure_penalty": -float(
                aligned["failed_service_attempt_penalty"]
            )
            if failure
            else 0.0,
            "reward_deadline_penalty": -float(aligned["deadline_miss_penalty"])
            if deadline_miss_event
            else 0.0,
            "reward_on_time_completion": 0.0,
        }

    def _observation(self) -> np.ndarray:
        if self.terminated:
            return np.zeros(9, dtype=np.float32)
        node = self._current_node()
        current = self._current_rsu_id()
        progress = float(len(self.completed)) / max(float(len(self.execution_order)), 1.0)
        return np.asarray(
            [
                progress,
                float(node["input_bytes"]) / 1_000_000.0,
                float(node["state_bytes"]) / 1_000_000.0,
                float(node["compute_seconds"]) / 10.0,
                float(self._contact_budget_seconds()) / 20.0,
                1.0 if self._bundle_ready(current, str(node["required_adapter"])) else 0.0,
                float(self._resident_bytes(self.caches[current])) / max(float(self.caches[current].capacity_bytes), 1.0),
                1.0 if self._predicted_handoff_target() else 0.0,
                float(len(node.get("successors", []))) / 4.0,
            ],
            dtype=np.float32,
        )

    def _semantic_state(self) -> dict[str, Any]:
        current_rsu = self._current_rsu_id()
        predicted_sequence = self._predicted_sequence()
        target = self._predicted_handoff_target()
        node = None if self.terminated else deepcopy(self._current_node())
        rsus = []
        for index, rsu_id in enumerate(self.instance["rsu_ids"]):
            cache = self.caches[str(rsu_id)]
            adapters = [
                self._object_catalog[item].get("adapter_id")
                for item in cache.residents
                if self._object_catalog[item]["object_type"] == "adapter"
            ]
            rsus.append(
                {
                    "rsu_id": str(rsu_id),
                    "position_x": float(index * 100),
                    "position_y": 0.0,
                    "coverage_radius": 100.0,
                    "cached_adapter_ids": [item for item in adapters if item],
                    "active_vehicle_ids": ["veh_pilot"] if str(rsu_id) == current_rsu else [],
                    "cache_capacity": float(cache.capacity_bytes),
                    "cache_used_bytes": self._resident_bytes(cache),
                    "typed_resident_object_ids": list(cache.residents),
                }
            )
        predictions = {
            "next_rsu_sequence": {"veh_pilot": predicted_sequence},
            "predicted_next_rsu_by_vehicle": {"veh_pilot": predicted_sequence[0]},
            "predicted_first_handoff_rsu_by_vehicle": {"veh_pilot": target},
            "predicted_handoff_target_rsu_id_by_vehicle": {"veh_pilot": target},
            "predicted_handoff_vehicle_ids": ["veh_pilot"] if target else [],
            "prediction_confidence_by_vehicle": {
                "veh_pilot": float(self.instance.get("prediction_profile", {}).get("confidence", 0.9))
            },
            "prediction_uncertainty_by_vehicle": {
                "veh_pilot": float(self.instance.get("prediction_profile", {}).get("uncertainty", 0.1))
            },
            "dwell_time": {"veh_pilot": self._contact_budget_seconds()},
            "future_load": {
                str(rsu_id): float(self._resident_bytes(self.caches[str(rsu_id)]))
                / max(float(self.caches[str(rsu_id)].capacity_bytes), 1.0)
                for rsu_id in self.instance["rsu_ids"]
            },
            "cache_demand": {"demand_score_by_rsu": {}},
        }
        interface_profile = str(
            self.config.get("interface_profile", "legacy_calibrated_workflow_v1")
        )
        current_bundle_ready = bool(
            node and self._bundle_ready(current_rsu, str(node["required_adapter"]))
        )
        target_bundle_ready = bool(
            node and target and self._bundle_ready(target, str(node["required_adapter"]))
        )
        bundle_ids = self._bundle_ids(str(node["required_adapter"])) if node else []
        bundle_resident_bytes = sum(
            int(self._object_catalog[item]["resident_bytes"]) for item in bundle_ids
        )
        return {
            "interface_profile": interface_profile,
            "mobility_progression": str(
                self.config.get("mobility_progression", "completed_node_index")
            ),
            "time_index": self.step_index,
            "primary_vehicle_id": "veh_pilot",
            "vehicles": [
                {
                    "vehicle_id": "veh_pilot",
                    "position_x": float(self._mobility_index() * 10),
                    "position_y": 0.0,
                    "speed": float(self.instance["trace_features"]["mean_speed_proxy"]),
                    "base_model_id": node.get("required_base_model") if node else "none",
                    "associated_rsu_id": current_rsu,
                    "active_workflow_id": self.instance["workflow_id"],
                }
            ],
            "rsus": rsus,
            "workflow": {
                "workflow_id": self.instance["workflow_id"],
                "nodes": deepcopy(self.nodes),
                "edges": [list(edge) for edge in self.instance["edges"]],
                "execution_order": list(self.execution_order),
                "completed_node_ids": list(self.completed),
                "current_node_id": node.get("node_id") if node else None,
                "is_completed": self.terminated,
            },
            "current_workflow_node": node,
            "current_node_service_steps_remaining": 1 if node else 0,
            "predictions": predictions,
            "handoff_events": [],
            "calibrated_context": {
                "contact_budget_seconds": self._contact_budget_seconds(),
                "cache_capacity_bytes": self.instance["cache_capacity_bytes"],
                "state_bytes": node.get("state_bytes", 0) if node else 0,
                "input_bytes": node.get("input_bytes", 0) if node else 0,
                "compute_seconds": node.get("compute_seconds", 0.0) if node else 0.0,
                "required_bundle_ids": bundle_ids,
                "required_bundle_resident_bytes": bundle_resident_bytes,
                "current_bundle_ready": current_bundle_ready,
                "target_bundle_ready": target_bundle_ready,
                "cache_capacity_unit": "bytes",
                "link": {
                    "estimated_mbps": float(
                        self.instance.get("link_profile", {}).get(
                            "estimated_mbps", self.config["link"]["mbps"]
                        )
                    ),
                    "fixed_seconds": float(self.config["link"]["fixed_seconds"]),
                    "source_type": self.config["link"]["source_type"],
                },
                "object_catalog": deepcopy(self._object_catalog),
                "adapter_to_bundle": deepcopy(self._adapter_to_bundle),
                "measured_time_seconds": deepcopy(self.config["measured_time_seconds"]),
                "source_classes": deepcopy(self.instance["source_classes"]),
            },
        }

    def _info(self) -> dict[str, Any]:
        semantic_state = self._semantic_state()
        return {
            "semantic_state": semantic_state,
            "action_mask": self._mask_builder.build_mask(semantic_state),
            "deterministic_policy": False,
            "run_metadata": {"policy_evaluation_mode": "raw_policy"},
        }


class TwoStepCostRule:
    """Information-matched deterministic two-step cost rule."""

    method_name = "two_step_cost_rule"

    def select_action(self, env: CalibratedContinuousWorkflowEnv) -> int:
        candidates: list[tuple[tuple[Any, ...], int]] = []
        for first_action in env.valid_actions():
            first_env = env.clone_for_decision_model()
            _, _, terminated, truncated, _ = first_env.step(first_action)
            if terminated or truncated:
                candidates.append((self._objective(first_env), first_action))
                continue
            second_scores = []
            for second_action in first_env.valid_actions():
                second_env = first_env.clone()
                second_env.step(second_action)
                second_scores.append(self._objective(second_env))
            candidates.append((min(second_scores), first_action))
        return min(candidates, key=lambda item: (item[0], item[1]))[1]

    @staticmethod
    def _objective(env: CalibratedContinuousWorkflowEnv) -> tuple[Any, ...]:
        metrics = env.metrics
        return (
            -int(metrics["completed_nodes"]),
            int(metrics["service_failures"]),
            int(metrics["deadline_violations"]),
            float(metrics["modeled_completion_seconds"]),
            int(metrics["model_transfer_bytes"] + metrics["state_transfer_bytes"] + metrics["input_transfer_bytes"]),
        )


class ImmediateCostRule(TwoStepCostRule):
    """Information-matched one-step rule using the estimated decision model."""

    method_name = "immediate_cost_rule"

    def select_action(self, env: CalibratedContinuousWorkflowEnv) -> int:
        candidates: list[tuple[tuple[Any, ...], int]] = []
        for action in env.valid_actions():
            preview = env.clone_for_decision_model()
            preview.step(action)
            candidates.append((self._objective(preview), action))
        return min(candidates, key=lambda item: (item[0], item[1]))[1]
