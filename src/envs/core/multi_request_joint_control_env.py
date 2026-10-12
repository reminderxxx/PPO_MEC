"""Eight-request controller experiment with the native five semantic actions.

The four RSUs are simulated. Every active request receives a controller action;
cache admission and handoff preparation happen only when that action asks for it.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from copy import deepcopy
from dataclasses import asdict
from typing import Any
from math import ceil, isfinite

from src.envs.core.multi_request_replay_env import MultiRequestReplayEnv
from src.envs.specs import ActionAdapter, ActionMaskBuilder, VehicleState


class MultiRequestJointControlEnv(MultiRequestReplayEnv):
    """Shared-resource, per-request execution of ActionSchema's five actions."""

    action_contract = "semantic_discrete_5_per_request_v1"
    vehicle_compute_frames = 4

    def __init__(self, *, environment_revision: str = "v1",
                 rsu_work_units_per_frame: float = 1.0,
                 vehicle_work_units_per_frame: float = 0.25,
                 workload_bytes_per_unit: float = 1024.0,
                 workflow_profile_unit_bytes: int = 1,
                 state_bytes_per_link_frame: int = 1024,
                 model_cache_config: dict[str, Any] | None = None,
                 vehicle_adapter_policy: str = "declared_only",
                 vehicle_adapter_ids: dict[str, list[str]] | list[str] | None = None,
                 prediction_enabled: bool = True,
                 prediction_horizon_frames: int = 3,
                 shared_state_enabled: bool = True,
                 shared_state_delay_frames: int = 1,
                 shared_state_interval_frames: int = 1,
                 prediction_provider: Any = None,
                 shared_state_channel: Any = None, **kwargs: Any) -> None:
        if environment_revision not in {"v1", "v2"}:
            raise ValueError("unknown joint environment revision")
        for value in (rsu_work_units_per_frame, vehicle_work_units_per_frame,
                      workload_bytes_per_unit, state_bytes_per_link_frame, workflow_profile_unit_bytes):
            if not isfinite(value) or value <= 0:
                raise ValueError("work and transfer rates must be finite and positive")
        self.environment_revision = environment_revision
        self.observation_contract = f"joint_request_observation_{environment_revision}"
        self.rsu_work_units_per_frame = float(rsu_work_units_per_frame)
        self.vehicle_work_units_per_frame = float(vehicle_work_units_per_frame)
        self.workload_bytes_per_unit = float(workload_bytes_per_unit)
        if int(workflow_profile_unit_bytes) != workflow_profile_unit_bytes:
            raise ValueError("workflow_profile_unit_bytes must be an integer byte count")
        self.workflow_profile_unit_bytes = int(workflow_profile_unit_bytes)
        self.state_bytes_per_link_frame = int(state_bytes_per_link_frame)
        self.model_cache_config = dict(model_cache_config or {})
        if vehicle_adapter_policy not in {"declared_only", "assumed_available"}:
            raise ValueError("unknown vehicle_adapter_policy")
        if vehicle_adapter_ids is not None and not isinstance(vehicle_adapter_ids, (dict, list)):
            raise ValueError("vehicle_adapter_ids must be an adapter list or map vehicle IDs to adapter lists")
        self.vehicle_adapter_policy = vehicle_adapter_policy
        self._declared_vehicle_adapter_ids = deepcopy(vehicle_adapter_ids) if vehicle_adapter_ids is not None else []
        if int(state_bytes_per_link_frame) != state_bytes_per_link_frame:
            raise ValueError("state_bytes_per_link_frame must be an integer byte count")
        self.prediction_horizon_frames = prediction_horizon_frames
        self.prediction_enabled = bool(prediction_enabled)
        self.shared_state_enabled = bool(shared_state_enabled)
        self.shared_state_delay_frames = shared_state_delay_frames
        self.shared_state_interval_frames = shared_state_interval_frames
        self._provided_prediction_provider = prediction_provider
        self._provided_shared_state_channel = shared_state_channel
        self.prediction_provider = prediction_provider
        self.shared_state_channel = shared_state_channel
        super().__init__(**kwargs)

    def reset(self) -> dict[str, Any]:
        observation = super().reset()
        self.prepared_state: dict[str, set[str]] = {vehicle_id: set() for vehicle_id in self.workflows}
        self.state_transfer_remaining: dict[tuple[str, str], int] = {}
        self.cold_restart_done: set[tuple[str, str]] = set()
        self.previous_positions: dict[str, tuple[float, float]] = {}
        self.previous_associations: dict[str, str | None] = {}
        self.vehicle_compute_progress: dict[str, int] = {vehicle_id: 0 for vehicle_id in self.workflows}
        if self.environment_revision == "v2":
            self.state_version = {vehicle_id: 0 for vehicle_id in self.workflows}
            self.state_payload_bytes = {vehicle_id: 0 for vehicle_id in self.workflows}
            self.execution_location = {vehicle_id: None for vehicle_id in self.workflows}
            self.work_progress = {vehicle_id: 0.0 for vehicle_id in self.workflows}
            self.state_copies: dict[tuple[str, str], dict[str, Any]] = {}
            self.state_transfers: dict[tuple[str, str], dict[str, Any]] = {}
            from src.envs.core.joint_causal_observation import CausalMobilityPredictor, SharedStateChannel
            self.prediction_provider = deepcopy(self._provided_prediction_provider) or CausalMobilityPredictor(horizon=self.prediction_horizon_frames)
            self.shared_state_channel = deepcopy(self._provided_shared_state_channel) or SharedStateChannel(
                interval_frames=self.shared_state_interval_frames,
                delay_frames=self.shared_state_delay_frames, enabled=self.shared_state_enabled)
            self.sharing_bytes_charged = 0
            self._frame_shared_state_bytes = 0
            self.model_link_frames_charged = 0
            if self.prediction_provider is not None and hasattr(self.prediction_provider, "reset"):
                self.prediction_provider.reset()
            if self.shared_state_channel is not None and hasattr(self.shared_state_channel, "reset"):
                self.shared_state_channel.reset()
            self._initialize_model_cache()
            self._initialize_vehicle_adapters()
        return observation

    def _causal_predictions(self, vehicles: list[VehicleState], associations: dict[str, str | None]) -> dict:
        if self.environment_revision == "v2" and not self.prediction_enabled:
            return {}
        if self.environment_revision == "v2" and self.prediction_provider is not None:
            return self.prediction_provider.predict(
                vehicles, associations, self.rsus, self.mapper,
                self.frame_index)
        predicted: dict[str, str | None] = {}
        confidence: dict[str, float] = {}
        dwell: dict[str, float] = {}
        for vehicle in vehicles:
            previous = self.previous_positions.get(vehicle.vehicle_id)
            current = associations[vehicle.vehicle_id]
            if previous is None:
                predicted[vehicle.vehicle_id] = None
                confidence[vehicle.vehicle_id] = 0.0
                dwell[vehicle.vehicle_id] = 0.0
                continue
            dx = vehicle.position_x - previous[0]
            dy = vehicle.position_y - previous[1]
            projected = deepcopy(vehicle)
            projected.position_x += 3.0 * dx
            projected.position_y += 3.0 * dy
            target = self.mapper.associate([projected])[vehicle.vehicle_id]
            predicted[vehicle.vehicle_id] = target if target != current else None
            confidence[vehicle.vehicle_id] = 0.5 if dx or dy else 0.0
            dwell[vehicle.vehicle_id] = 3.0 if target != current else 0.0
        counts = Counter(rsu_id for rsu_id in associations.values() if rsu_id is not None)
        return {
            "predicted_next_rsu_by_vehicle": predicted,
            "predicted_first_handoff_rsu_by_vehicle": predicted,
            "predicted_handoff_target_rsu_id_by_vehicle": predicted,
            "next_rsu_sequence": {vehicle_id: [target] if target else []
                                  for vehicle_id, target in predicted.items()},
            "predicted_handoff_vehicle_ids": [vehicle_id for vehicle_id, target in predicted.items() if target],
            "prediction_confidence_by_vehicle": confidence,
            "prediction_uncertainty_by_vehicle": {vehicle_id: 1.0 - value
                                                   for vehicle_id, value in confidence.items()},
            "dwell_time": dwell,
            "future_load": {rsu.rsu_id: float(counts[rsu.rsu_id]) for rsu in self.rsus},
        }

    def decision_states(self) -> dict[str, dict[str, Any]]:
        """Build native-agent semantic states from current and previous frames only."""
        if self.frame_index >= len(self.frames):
            return {}
        vehicles = list(self.frames[self.frame_index]["vehicles"])
        associations = self.mapper.associate(vehicles)
        predictions = self._causal_predictions(vehicles, associations)
        counts = Counter(rsu_id for rsu_id in associations.values() if rsu_id is not None)
        rsus = [{**rsu.to_dict(),
                 "cached_adapter_ids": list(self.cache[rsu.rsu_id]),
                 "loading_adapter_ids": list(self.loading[rsu.rsu_id]),
                 "active_vehicle_ids": [vehicle.vehicle_id for vehicle in vehicles
                                        if associations[vehicle.vehicle_id] == rsu.rsu_id],
                 "adapter_cache_capacity": self.costs.adapter_slots_per_rsu,
                 "associated_vehicle_count": counts[rsu.rsu_id]}
                for rsu in self.rsus]
        shared_snapshot = None
        if self.environment_revision == "v2":
            rsus = [{**row, **self.model_cache.snapshot(row["rsu_id"])} for row in rsus]
            shared_snapshot = self.shared_state_channel.snapshot(rsus, self.frame_index)
            self._frame_shared_state_bytes = int(shared_snapshot["metadata"]["transmitted_bytes"])
        result = {}
        for vehicle in vehicles:
            workflow = self.workflows[vehicle.vehicle_id]
            if workflow.is_completed:
                continue
            node = workflow.current_node()
            if node is None:
                raise RuntimeError("active workflow has no node")
            vehicle_rows = [{**item.to_dict(),
                             "associated_rsu_id": associations[item.vehicle_id],
                             "active_workflow_id": self.workflows[item.vehicle_id].workflow_id}
                            for item in vehicles]
            vehicle_rows.sort(key=lambda item: (item["vehicle_id"] != vehicle.vehicle_id,
                                                item["vehicle_id"]))
            state = {
                "time_index": int(self.frames[self.frame_index]["time_index"]),
                "vehicles": vehicle_rows,
                "primary_vehicle_id": vehicle.vehicle_id,
                "primary_vehicle_present": True,
                "rsus": rsus,
                "associations": associations,
                "workflow": workflow.to_dict(),
                "current_workflow_node": node.to_dict(),
                "current_node_service_steps_required": 1,
                "current_node_service_steps_remaining": 1,
                "predictions": predictions,
                "handoff_events": [],
            }
            if self.environment_revision == "v2":
                self._augment_execution_state(state, vehicle.vehicle_id)
                state["rsus"] = self.shared_state_channel.actor_rsus(
                    rsus, associations[vehicle.vehicle_id], shared_snapshot)
                # Current wire length reveals unpublished remote cache contents.
                # Keep byte accounting in frame receipts and environment totals.
                state["shared_state"] = {
                    key: deepcopy(value) for key, value in shared_snapshot["metadata"].items()
                    if key not in {"transmitted_bytes", "total_transmitted_bytes"}
                }
            result[vehicle.vehicle_id] = {
                "semantic_state": state,
                "action_mask": ActionMaskBuilder().build_mask(state),
                "action_contract": self.action_contract,
            }
            if self.environment_revision == "v2":
                result[vehicle.vehicle_id]["action_mask"][2] = (
                    vehicle.base_model_id == node.required_base_model and state["vehicle_adapter_ready"])
                result[vehicle.vehicle_id]["centralized_critic_state"] = {
                    "joint_workflows": state.pop("joint_workflows"),
                    "rsus": deepcopy(rsus), "associations": deepcopy(associations)}
        return result

    def _request_adapter(self, rsu_id: str, adapter_id: str,
                         link_left: dict[str, int]) -> str:
        cache = self.cache[rsu_id]
        loading = self.loading[rsu_id]
        if adapter_id in cache:
            return "already_cached"
        if adapter_id in loading:
            return "already_loading"
        if link_left[rsu_id] < 1:
            return "link_queue"
        if len(cache) + len(loading) >= self.costs.adapter_slots_per_rsu and not cache:
            return "cache_queue"
        link_left[rsu_id] -= 1
        while len(cache) + len(loading) >= self.costs.adapter_slots_per_rsu:
            cache.popitem(last=False)
        if self.costs.adapter_fetch_frames:
            loading[adapter_id] = self.costs.adapter_fetch_frames
        else:
            cache[adapter_id] = None
        return "adapter_fetch"

    def _request_state(self, vehicle_id: str, target: str,
                       link_left: dict[str, int]) -> str:
        if target in self.prepared_state[vehicle_id]:
            return "already_prepared"
        if self.costs.handoff_state_frames == 0:
            self.prepared_state[vehicle_id].add(target)
            return "state_prepared"
        if link_left[target] < 1:
            return "link_queue"
        link_left[target] -= 1
        key = (vehicle_id, target)
        remaining = self.state_transfer_remaining.get(key, self.costs.handoff_state_frames)
        remaining = max(remaining - 1, 0)
        if remaining:
            self.state_transfer_remaining[key] = remaining
            return "state_transfer"
        self.state_transfer_remaining.pop(key, None)
        self.prepared_state[vehicle_id].add(target)
        return "state_prepared"

    def step_actions(self, actions: dict[str, int]) -> tuple[dict, float, bool, dict]:
        """Apply one legal native semantic action per active request and frame."""
        if self.environment_revision == "v2":
            return self._step_actions_v2(actions)
        if self.frame_index >= len(self.frames):
            raise RuntimeError("episode ended")
        decision = self.decision_states()
        if set(actions) != set(decision):
            raise ValueError("one action is required for every active request")
        for vehicle_id, action in actions.items():
            if not isinstance(action, int) or action not in range(5) or not decision[vehicle_id]["action_mask"][action]:
                raise ValueError(f"illegal semantic action for {vehicle_id}: {action}")
        vehicles = {vehicle.vehicle_id: vehicle for vehicle in self.frames[self.frame_index]["vehicles"]}
        compute_left = {rsu.rsu_id: self.costs.compute_slots_per_rsu for rsu in self.rsus}
        link_left = {rsu.rsu_id: self.costs.link_slots_per_rsu - len(self.loading[rsu.rsu_id])
                     for rsu in self.rsus}
        frame_events = []
        completed_now = 0
        ids = sorted(decision)
        shift = self.frame_index % len(ids)
        for vehicle_id in ids[shift:] + ids[:shift]:
            state = decision[vehicle_id]["semantic_state"]
            control = ActionAdapter().decode(actions[vehicle_id], state)
            workflow = self.workflows[vehicle_id]
            node = workflow.current_node()
            current_rsu = state["associations"][vehicle_id]
            association_handoff = (
                vehicle_id in self.previous_associations
                and self.previous_associations[vehicle_id] != current_rsu
            )
            prepared_on_arrival = bool(
                association_handoff and current_rsu is not None
                and current_rsu in self.prepared_state[vehicle_id]
            )
            cache_status = "none"
            state_status = "none"
            cache_action = control.cache_action
            if cache_action and cache_action.get("rsu_id") in link_left:
                cache_status = self._request_adapter(str(cache_action["rsu_id"]),
                                                     node.required_adapter, link_left)
            migration = control.migration_action
            if migration.get("mode") == "prepare":
                target = migration.get("expected_target_rsu_id")
                if target in link_left:
                    state_status = self._request_state(vehicle_id, str(target), link_left)
            if actions[vehicle_id] == 2:
                if vehicles[vehicle_id].base_model_id != node.required_base_model:
                    status = "base_model_mismatch"
                else:
                    self.vehicle_compute_progress[vehicle_id] += 1
                    status = "vehicle_compute"
                    if self.vehicle_compute_progress[vehicle_id] >= self.vehicle_compute_frames:
                        workflow.mark_current_completed()
                        self.last_service_rsu[vehicle_id] = None
                        self.vehicle_compute_progress[vehicle_id] = 0
                        completed_now += 1
                        status = "vehicle_fallback_completed"
            elif current_rsu is None:
                self.vehicle_compute_progress[vehicle_id] = 0
                status = "no_coverage"
            elif vehicles[vehicle_id].base_model_id != node.required_base_model:
                self.vehicle_compute_progress[vehicle_id] = 0
                status = "base_model_mismatch"
            elif (self.last_service_rsu[vehicle_id] not in (None, current_rsu)
                  and current_rsu not in self.prepared_state[vehicle_id]
                  and (vehicle_id, current_rsu) not in self.cold_restart_done):
                self.vehicle_compute_progress[vehicle_id] = 0
                self.cold_restart_done.add((vehicle_id, current_rsu))
                status = "cold_restart"
            elif node.required_adapter in self.loading[current_rsu]:
                self.vehicle_compute_progress[vehicle_id] = 0
                status = "adapter_loading"
            elif node.required_adapter not in self.cache[current_rsu]:
                self.vehicle_compute_progress[vehicle_id] = 0
                status = "adapter_not_ready"
            elif compute_left[current_rsu] < 1:
                self.vehicle_compute_progress[vehicle_id] = 0
                status = "compute_queue"
            else:
                self.vehicle_compute_progress[vehicle_id] = 0
                compute_left[current_rsu] -= 1
                self.cache[current_rsu].move_to_end(node.required_adapter)
                workflow.mark_current_completed()
                self.last_service_rsu[vehicle_id] = current_rsu
                completed_now += 1
                status = "node_completed"
            frame_events.append({
                "frame_index": self.frame_index,
                "time_index": int(self.frames[self.frame_index]["time_index"]),
                "vehicle_id": vehicle_id,
                "request_id": f"{vehicle_id}/{workflow.workflow_id}",
                "workflow_id": workflow.workflow_id,
                "node_id": node.node_id,
                "associated_rsu_id": current_rsu,
                "target_rsu_id": current_rsu,
                "action_id": actions[vehicle_id],
                "cache_target_rsu_id": cache_action.get("rsu_id"),
                "cache_action_status": cache_status,
                "state_action_status": state_status,
                "association_handoff": association_handoff,
                "prepared_on_arrival": prepared_on_arrival,
                "handoff_failed": association_handoff and status == "cold_restart",
                "status": status,
                "workflow_completed": workflow.is_completed,
            })
        self.events.extend(frame_events)
        self.previous_positions = {vehicle_id: (float(vehicle.position_x), float(vehicle.position_y))
                                   for vehicle_id, vehicle in vehicles.items()}
        self.previous_associations = {
            vehicle_id: decision[vehicle_id]["semantic_state"]["associations"][vehicle_id]
            for vehicle_id in decision
        }
        self._finish_loading()
        self.frame_index += 1
        done = self.frame_index >= len(self.frames) or all(w.is_completed for w in self.workflows.values())
        observation = self.observe() if not done else {"frame_index": self.frame_index, "vehicles": {}, "rsus": {}}
        reward = (
            float(completed_now)
            - 0.01 * sum(not workflow.is_completed for workflow in self.workflows.values())
            - 0.02 * sum(event["status"].startswith("vehicle_") for event in frame_events)
            - 0.01 * sum(event["cache_action_status"] == "adapter_fetch" for event in frame_events)
            - 0.01 * sum(event["state_action_status"] == "state_transfer" for event in frame_events)
        )
        return observation, reward, done, {"events": frame_events, "completed_nodes": completed_now}

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        result["cache_fetch_count"] = sum(event["cache_action_status"] == "adapter_fetch" for event in self.events)
        result["state_prepared_count"] = sum(event["state_action_status"] == "state_prepared" for event in self.events)
        result["cold_restart_count"] = sum(event["status"] == "cold_restart" for event in self.events)
        result["vehicle_fallback_count"] = sum(event["status"] == "vehicle_fallback_completed" for event in self.events)
        result["vehicle_compute_frames"] = sum(event["status"].startswith("vehicle_") for event in self.events)
        result["association_handoff_count"] = sum(event["association_handoff"] for event in self.events)
        result["prepared_on_arrival_count"] = sum(event["prepared_on_arrival"] for event in self.events)
        result["handoff_failed_count"] = sum(event["handoff_failed"] for event in self.events)
        if self.environment_revision == "v2":
            result.update({
                "environment_revision": "v2", "observation_contract": self.observation_contract,
                "state_transfer_bytes": sum(event["state_transfer_bytes"] for event in self.events),
                "state_transfer_link_frames": sum(event["state_transfer_link_frames"] for event in self.events),
                "model_transfer_link_frames": self.model_link_frames_charged,
                "shared_state_transmitted_bytes": self.sharing_bytes_charged,
                "state_restore_frames": sum(event["status"] == "state_restore" for event in self.events),
                "stale_state_copy_count": sum(copy["version"] != self.state_version[vehicle_id]
                    for (vehicle_id, _), copy in self.state_copies.items()),
                "pending_state_transfer_bytes": sum(t["transferred_bytes"] for t in self.state_transfers.values()),
                "pending_state_transfer_count": len(self.state_transfers),
                "state_prepared_count": sum(t["status"] == "state_prepared" for event in self.events
                                            for t in event["state_transfers"]),
                "model_cache": {r.rsu_id: self.model_cache.snapshot(r.rsu_id) for r in self.rsus},
            })
        return result

    def _initialize_vehicle_adapters(self) -> None:
        declarations = self._declared_vehicle_adapter_ids
        if isinstance(declarations, list):
            declarations = {vehicle_id: declarations for vehicle_id in self.workflows}
        unknown_vehicles = set(declarations) - set(self.workflows)
        if unknown_vehicles:
            raise ValueError(f"unknown vehicle_adapter_ids vehicles: {sorted(unknown_vehicles)}")
        known_adapters = {node.required_adapter for workflow in self.workflows.values() for node in workflow.nodes}
        for vehicle_id, adapters in declarations.items():
            if not isinstance(adapters, list) or any(not isinstance(adapter, str) for adapter in adapters):
                raise ValueError(f"vehicle_adapter_ids[{vehicle_id}] must be a list of adapter IDs")
            unknown = set(adapters) - known_adapters
            if unknown:
                raise ValueError(f"unknown vehicle adapter IDs for {vehicle_id}: {sorted(unknown)}")
        self.vehicle_adapter_ids = {
            vehicle_id: sorted(set(declarations.get(vehicle_id, []))) for vehicle_id in self.workflows}

    def _vehicle_adapter_ready(self, vehicle_id: str, adapter_id: str) -> bool:
        return (self.vehicle_adapter_policy == "assumed_available"
                or adapter_id in self.vehicle_adapter_ids[vehicle_id])

    def _initialize_model_cache(self) -> None:
        from src.envs.core.joint_model_cache import JointModelCache, controlled_catalog_from_workflows
        defaults = {"capacity_bytes": 4096, "base_size_bytes": 2048,
                    "adapter_size_bytes": 512, "transfer_bytes_per_frame": 1024}
        unknown = set(self.model_cache_config) - set(defaults)
        if unknown:
            raise ValueError(f"unknown model_cache_config fields: {sorted(unknown)}")
        defaults.update(self.model_cache_config)
        self.model_cache_config = defaults
        self.model_cache = JointModelCache(
            rsu_ids=[rsu.rsu_id for rsu in self.rsus],
            capacity_bytes=defaults["capacity_bytes"],
            objects=controlled_catalog_from_workflows(
                self.workflows, base_size_bytes=defaults["base_size_bytes"],
                adapter_size_bytes=defaults["adapter_size_bytes"]),
            link_slots=self.costs.link_slots_per_rsu,
            transfer_bytes_per_frame=defaults["transfer_bytes_per_frame"])
        self._sync_cache_view()

    def _sync_cache_view(self) -> None:
        for rsu in self.rsus:
            snapshot = self.model_cache.snapshot(rsu.rsu_id)
            self.cache[rsu.rsu_id] = OrderedDict.fromkeys(snapshot["cached_adapter_ids"])
            self.loading[rsu.rsu_id] = dict.fromkeys(snapshot["loading_adapter_ids"], 1)

    def _work_required(self, node: Any) -> float:
        # Legacy workflow sizes are synthetic profile units, not measured bytes.
        synthetic_bytes = max(0, node.input_size) * self.workflow_profile_unit_bytes
        return max(1.0, synthetic_bytes / self.workload_bytes_per_unit)

    def _parents_ready(self, vehicle_id: str) -> bool:
        workflow = self.workflows[vehicle_id]
        node = workflow.current_node()
        if node is None:
            return False
        parents = set(node.predecessors)
        parents.update(source for source, target in workflow.edges if target == node.node_id)
        return parents.issubset(workflow.completed_node_ids)

    def _state_ready(self, vehicle_id: str, target: str) -> bool:
        if self.state_version[vehicle_id] == 0 or self.execution_location[vehicle_id] == target:
            return True
        copy = self.state_copies.get((vehicle_id, target), {})
        return copy.get("version") == self.state_version[vehicle_id]

    def _augment_execution_state(self, state: dict, vehicle_id: str) -> None:
        node = self.workflows[vehicle_id].current_node()
        current = state["associations"][vehicle_id]
        required = self._work_required(node)
        remaining = max(0.0, required - self.work_progress[vehicle_id])
        joint = {}
        for item, workflow in self.workflows.items():
            item_node = workflow.current_node()
            joint[item] = {
                "workflow_id": workflow.workflow_id,
                "current_node_id": workflow.current_node_id,
                "completed_node_ids": list(workflow.completed_node_ids),
                "remaining_nodes": len(workflow.nodes) - len(workflow.completed_node_ids),
                "required_adapter": item_node.required_adapter if item_node else None,
                "work_progress": self.work_progress[item],
                "remaining_work_units": max(0.0, self._work_required(item_node) - self.work_progress[item])
                if item_node else 0.0,
                "state_version": self.state_version[item],
                "execution_location": self.execution_location[item],
                "parents_ready": self._parents_ready(item),
            }
        state.update({
            "environment_revision": "v2", "observation_contract": self.observation_contract,
            "current_node_work_units_required": required,
            "current_node_work_units_progress": self.work_progress[vehicle_id],
            "current_node_service_steps_required": ceil(required / self.rsu_work_units_per_frame),
            "current_node_service_steps_remaining": ceil(remaining / self.rsu_work_units_per_frame),
            "vehicle_service_steps_remaining": ceil(remaining / self.vehicle_work_units_per_frame),
            "current_node_parents_ready": self._parents_ready(vehicle_id),
            "vehicle_adapter_ready": self._vehicle_adapter_ready(vehicle_id, node.required_adapter),
            "vehicle_adapter_ids": list(self.vehicle_adapter_ids[vehicle_id]),
            "vehicle_adapter_policy": self.vehicle_adapter_policy,
            "current_adapter_ready": bool(current and self.model_cache.is_ready(
                current, node.required_adapter, node.required_base_model)),
            "current_state_ready": bool(current and self._state_ready(vehicle_id, current)),
            "execution_state_version": self.state_version[vehicle_id],
            "execution_state_payload_bytes": self.state_payload_bytes[vehicle_id],
            "execution_location": self.execution_location[vehicle_id],
            "state_copies_by_rsu": {target: deepcopy(copy) for (owner, target), copy
                                    in self.state_copies.items() if owner == vehicle_id},
            "state_transfers_by_rsu": {target: deepcopy(transfer) for (owner, target), transfer
                                        in self.state_transfers.items() if owner == vehicle_id},
            "joint_workflows": joint,
            "cost_model": "synthetic_input_byte_work_and_append_only_state_v2",
        })

    def _transfer_state_v2(self, vehicle_id: str, target: str,
                           link_left: dict[str, int]) -> dict[str, Any]:
        result = {"status": "already_prepared", "bytes": 0, "link_frames": 0,
                  "target": target, "version": self.state_version[vehicle_id]}
        if self._state_ready(vehicle_id, target):
            return result
        key = (vehicle_id, target)
        if key not in self.state_transfers:
            copied = self.state_copies.get(key, {}).get("payload_bytes", 0)
            payload = max(1, self.state_payload_bytes[vehicle_id] - copied)
            self.state_transfers[key] = {
                "version": self.state_version[vehicle_id],
                "payload_bytes": self.state_payload_bytes[vehicle_id],
                "remaining_bytes": payload, "transferred_bytes": 0,
                "source": self.execution_location[vehicle_id],
                "remaining_frames": max(self.costs.handoff_state_frames,
                                        ceil(payload / self.state_bytes_per_link_frame)),
                "completed_node_ids": list(self.workflows[vehicle_id].completed_node_ids),
                "work_progress": self.work_progress[vehicle_id],
            }
        transfer = self.state_transfers[key]
        result["version"] = transfer["version"]
        if link_left.get(target, 0) < 1:
            result["status"] = "link_queue"
            return result
        link_left[target] -= 1
        moved = min(transfer["remaining_bytes"], self.state_bytes_per_link_frame)
        transfer["remaining_bytes"] -= moved
        transfer["transferred_bytes"] += moved
        transfer["remaining_frames"] -= 1
        result.update(bytes=moved, link_frames=1, status="state_transfer")
        if transfer["remaining_frames"] <= 0 and transfer["remaining_bytes"] == 0:
            self.state_copies[key] = {field: deepcopy(transfer[field]) for field in
                                      ("version", "payload_bytes", "completed_node_ids", "work_progress")}
            self.state_transfers.pop(key)
            result["status"] = "state_prepared"
        return result

    def _advance_work(self, vehicle_id: str, location: str, rate: float) -> bool:
        workflow = self.workflows[vehicle_id]
        node = workflow.current_node()
        self.work_progress[vehicle_id] += rate
        self.execution_location[vehicle_id] = location
        self.last_service_rsu[vehicle_id] = None if location.startswith("vehicle:") else location
        self.state_version[vehicle_id] += 1
        # Each update adds a synthetic one-byte progress delta; completed outputs
        # are append-only, so an old prepared prefix remains useful but not ready.
        self.state_payload_bytes[vehicle_id] += 1
        completed = self.work_progress[vehicle_id] + 1e-12 >= self._work_required(node)
        if completed:
            self.state_payload_bytes[vehicle_id] += max(1, node.output_size * self.workflow_profile_unit_bytes)
            workflow.mark_current_completed()
            self.work_progress[vehicle_id] = 0.0
        self.state_copies[(vehicle_id, location)] = {
            "version": self.state_version[vehicle_id],
            "payload_bytes": self.state_payload_bytes[vehicle_id],
            "completed_node_ids": list(workflow.completed_node_ids),
            "work_progress": self.work_progress[vehicle_id],
        }
        return completed

    def _step_actions_v2(self, actions: dict[str, int]) -> tuple[dict, float, bool, dict]:
        if self.frame_index >= len(self.frames) or all(w.is_completed for w in self.workflows.values()):
            raise RuntimeError("episode ended")
        decision = self.decision_states()
        if set(actions) != set(decision):
            raise ValueError("one action is required for every active request")
        for vehicle_id, action in actions.items():
            if type(action) is not int or action not in range(5) or not decision[vehicle_id]["action_mask"][action]:
                raise ValueError(f"illegal semantic action for {vehicle_id}: {action}")
        vehicles = {v.vehicle_id: v for v in self.frames[self.frame_index]["vehicles"]}
        compute_left = {r.rsu_id: self.costs.compute_slots_per_rsu for r in self.rsus}
        occupied = self.model_cache.begin_frame()
        link_left = {r.rsu_id: self.costs.link_slots_per_rsu - occupied[r.rsu_id] for r in self.rsus}
        frame_events, completed_now = [], 0
        ids = sorted(decision)
        shift = self.frame_index % len(ids)
        for vehicle_id in ids[shift:] + ids[:shift]:
            state = decision[vehicle_id]["semantic_state"]
            action = actions[vehicle_id]
            control = ActionAdapter().decode(action, state)
            workflow = self.workflows[vehicle_id]
            node = workflow.current_node()
            current = state["associations"][vehicle_id]
            handoff = vehicle_id in self.previous_associations and self.previous_associations[vehicle_id] != current
            prepared_on_arrival = bool(handoff and current and self.state_version[vehicle_id] > 0
                                       and self._state_ready(vehicle_id, current))
            before_version = self.state_version[vehicle_id]
            transfers = []
            cache_result = {"status": "none", "link_slots_consumed": 0}
            cache_target = control.cache_action.get("rsu_id")
            if cache_target in link_left:
                cache_result = self.model_cache.request_adapter(
                    cache_target, node.required_adapter, node.required_base_model,
                    available_link_slots=link_left[cache_target])
                link_left[cache_target] -= cache_result["link_slots_consumed"]
            migration_target = control.migration_action.get("expected_target_rsu_id")
            if control.migration_action.get("mode") == "prepare" and migration_target in link_left:
                transfers.append(self._transfer_state_v2(vehicle_id, migration_target, link_left))
            status, completed = "no_coverage", False
            if not self._parents_ready(vehicle_id):
                status = "parents_not_ready"
            elif action == 2:
                location = f"vehicle:{vehicle_id}"
                if vehicles[vehicle_id].base_model_id != node.required_base_model:
                    status = "base_model_mismatch"
                elif not self._vehicle_adapter_ready(vehicle_id, node.required_adapter):
                    status = "vehicle_adapter_not_ready"
                elif not self._state_ready(vehicle_id, location):
                    # Vehicle reception consumes the current RSU link. No free
                    # teleport when switching from edge execution to fallback.
                    if current is None:
                        status = "state_unavailable_no_coverage"
                    else:
                        local_link = {location: link_left[current]}
                        transfer = self._transfer_state_v2(vehicle_id, location, local_link)
                        link_left[current] = local_link[location]
                        transfers.append(transfer)
                        status = "state_restore" if transfer["link_frames"] else "state_link_queue"
                else:
                    completed = self._advance_work(vehicle_id, location, self.vehicle_work_units_per_frame)
                    status = "vehicle_fallback_completed" if completed else "vehicle_compute"
            elif current is not None:
                if not self._state_ready(vehicle_id, current):
                    transfer = self._transfer_state_v2(vehicle_id, current, link_left)
                    transfers.append(transfer)
                    status = "state_restore" if transfer["link_frames"] else "state_link_queue"
                elif not self.model_cache.is_ready(current, node.required_adapter, node.required_base_model):
                    status = "adapter_not_ready"
                elif compute_left[current] < 1:
                    status = "compute_queue"
                else:
                    compute_left[current] -= 1
                    self.model_cache.touch(current, node.required_adapter, node.required_base_model)
                    completed = self._advance_work(vehicle_id, current, self.rsu_work_units_per_frame)
                    status = "node_completed" if completed else "rsu_compute"
            completed_now += int(completed)
            frame_events.append({
                "frame_index": self.frame_index, "time_index": int(self.frames[self.frame_index]["time_index"]),
                "vehicle_id": vehicle_id, "request_id": f"{vehicle_id}/{workflow.workflow_id}",
                "workflow_id": workflow.workflow_id, "node_id": node.node_id,
                "associated_rsu_id": current, "target_rsu_id": current, "action_id": action,
                "cache_target_rsu_id": cache_target, "cache_action_status": cache_result["status"],
                "cache_request": cache_result,
                "state_action_status": transfers[-1]["status"] if transfers else "none",
                "state_transfers": transfers,
                "state_transfer_bytes": sum(t["bytes"] for t in transfers),
                "state_transfer_link_frames": sum(t["link_frames"] for t in transfers),
                "state_version_before": before_version, "state_version_after": self.state_version[vehicle_id],
                "state_payload_bytes": self.state_payload_bytes[vehicle_id],
                "work_progress": self.work_progress[vehicle_id], "work_units_required": self._work_required(node),
                "association_handoff": handoff, "prepared_on_arrival": prepared_on_arrival,
                "handoff_failed": bool(handoff and status in {"state_restore", "state_link_queue"}),
                "status": status, "workflow_completed": workflow.is_completed,
                "environment_revision": "v2",
            })
        cache_events = self.model_cache.advance()
        self._sync_cache_view()
        self.events.extend(frame_events)
        self.previous_positions = {v.vehicle_id: (float(v.position_x), float(v.position_y)) for v in vehicles.values()}
        self.previous_associations = {v: decision[v]["semantic_state"]["associations"][v] for v in decision}
        self.frame_index += 1
        done = self.frame_index >= len(self.frames) or all(w.is_completed for w in self.workflows.values())
        observation = self.observe() if not done else {"frame_index": self.frame_index, "vehicles": {}, "rsus": {}}
        # Every state link frame is charged, including the completing frame.
        cache_link_frames = sum(occupied.values()) + sum(e["cache_request"]["link_slots_consumed"] for e in frame_events)
        sharing_bytes = self._frame_shared_state_bytes
        self.sharing_bytes_charged += sharing_bytes
        self.model_link_frames_charged += cache_link_frames
        components = {
            "shared_state": -0.001 * sharing_bytes / 1024.0,
            "completed_nodes": float(completed_now),
            "unfinished_wait": -0.01 * sum(not w.is_completed for w in self.workflows.values()),
            "vehicle_compute": -0.02 * sum(e["status"].startswith("vehicle_") for e in frame_events),
            "model_transfer": -0.01 * cache_link_frames,
            "state_transfer": -0.01 * sum(e["state_transfer_link_frames"] for e in frame_events),
        }
        # The launcher persists request events. Attach exactly one independent
        # receipt per frame so a reader can reconstruct costs without replaying
        # the environment or counting a global cost once for each request.
        frame_events[0]["frame_cost_receipt"] = {
            "contract_version": "joint_frame_cost_receipt_v1",
            "frame_index": frame_events[0]["frame_index"],
            "cache_events": deepcopy(cache_events),
            "reward_components": deepcopy(components),
            "model_transfer_bytes": sum(event["transferred_bytes"] for event in cache_events),
            "model_transfer_link_frames": cache_link_frames,
            "shared_state_transmitted_bytes": sharing_bytes,
            "state_transfer_bytes": sum(event["state_transfer_bytes"] for event in frame_events),
            "state_transfer_link_frames": sum(event["state_transfer_link_frames"] for event in frame_events),
        }
        return observation, sum(components.values()), done, {
            "events": frame_events, "completed_nodes": completed_now,
            "cache_events": cache_events, "reward_components": components,
            "model_transfer_link_frames": cache_link_frames,
        }

    def configuration(self) -> dict[str, Any]:
        """Return effective protocol parameters, including synthetic cost sources."""
        config = {
            "environment_revision": self.environment_revision,
            "action_contract": self.action_contract,
            "observation_contract": self.observation_contract,
            "replay_costs": asdict(self.costs),
        }
        if self.environment_revision == "v1":
            config["vehicle_compute_frames"] = self.vehicle_compute_frames
            return config
        config.update({
            "rsu_work_units_per_frame": self.rsu_work_units_per_frame,
            "vehicle_work_units_per_frame": self.vehicle_work_units_per_frame,
            "workload_bytes_per_unit": self.workload_bytes_per_unit,
            "workflow_size_source": "synthetic_profile_units_not_measured_bytes",
            "workflow_profile_unit_bytes": self.workflow_profile_unit_bytes,
            "workload_formula": "max(1,input_size*workflow_profile_unit_bytes/workload_bytes_per_unit)",
            "state_bytes_per_link_frame": self.state_bytes_per_link_frame,
            "state_payload_formula": "append_only_completed_output_size_times_profile_unit_bytes_plus_one_byte_per_compute_update",
            "state_minimum_transfer_frames": self.costs.handoff_state_frames,
            "model_cache": deepcopy(self.model_cache_config),
            "model_cache_contract": self.model_cache.contract_version,
            "model_cache_eviction_policy": self.model_cache.eviction_policy,
            "prediction_enabled": self.prediction_enabled,
            "prediction_horizon_frames": self.prediction_horizon_frames,
            "prediction_source": "causal_constant_velocity_v1",
            "shared_state_enabled": self.shared_state_enabled,
            "shared_state_delay_frames": self.shared_state_delay_frames,
            "shared_state_interval_frames": self.shared_state_interval_frames,
            "shared_state_cost_source": "serialized_json_proxy_not_measured_radio",
            "reward_coefficients": {
                "completed_node": 1.0, "unfinished_request_frame": -0.01,
                "vehicle_compute_frame": -0.02, "model_transfer_link_frame": -0.01,
                "state_transfer_link_frame": -0.01, "shared_state_per_1024_bytes": -0.001},
            "cost_calibration": "controlled_synthetic_not_measured",
            "event_receipt_contract": "one_joint_frame_cost_receipt_v1_per_executed_frame",
            "vehicle_adapter_policy": self.vehicle_adapter_policy,
            "vehicle_adapter_ids": deepcopy(self.vehicle_adapter_ids),
            "vehicle_adapter_declaration": deepcopy(self._declared_vehicle_adapter_ids),
            "vehicle_adapter_preload_contract": "static_declared_objects_preloaded_without_runtime_transfer_or_capacity_charge",
            "fallback_adapter_contract": ("base_match_and_declared_adapter_required"
                                          if self.vehicle_adapter_policy == "declared_only"
                                          else "explicit_legacy_adapter_available_assumption"),
            "workflow_arrival_contract": "one_workflow_per_persistent_vehicle_at_frame_zero",
            "dag_execution_contract": "serial_topological_order_with_parent_readiness_check",
            "state_storage_contract": "separate_unbounded_append_only_prefix_storage",
            "shared_state_transport_contract": "separate_proxy_control_plane_cost_not_data_link_contention",
        })
        return config
