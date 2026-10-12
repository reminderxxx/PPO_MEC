"""Eight-request controller experiment with the native five semantic actions.

The four RSUs are simulated. Every active request receives a controller action;
cache admission and handoff preparation happen only when that action asks for it.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from src.envs.core.multi_request_replay_env import MultiRequestReplayEnv
from src.envs.specs import ActionAdapter, ActionMaskBuilder, VehicleState


class MultiRequestJointControlEnv(MultiRequestReplayEnv):
    """Shared-resource, per-request execution of ActionSchema's five actions."""

    action_contract = "semantic_discrete_5_per_request_v1"
    vehicle_compute_frames = 4

    def reset(self) -> dict[str, Any]:
        observation = super().reset()
        self.prepared_state: dict[str, set[str]] = {vehicle_id: set() for vehicle_id in self.workflows}
        self.state_transfer_remaining: dict[tuple[str, str], int] = {}
        self.cold_restart_done: set[tuple[str, str]] = set()
        self.previous_positions: dict[str, tuple[float, float]] = {}
        self.previous_associations: dict[str, str | None] = {}
        self.vehicle_compute_progress: dict[str, int] = {vehicle_id: 0 for vehicle_id in self.workflows}
        return observation

    def _causal_predictions(self, vehicles: list[VehicleState], associations: dict[str, str | None]) -> dict:
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
            result[vehicle.vehicle_id] = {
                "semantic_state": state,
                "action_mask": ActionMaskBuilder().build_mask(state),
                "action_contract": self.action_contract,
            }
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
        return result
