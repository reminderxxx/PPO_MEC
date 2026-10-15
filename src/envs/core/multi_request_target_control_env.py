"""Explicit placement/cache/state control for engineering validation.

This is a new parameterized contract, not an adapter for five-action policies.
All sizes and rates remain declared frame-based simulation assumptions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from contextlib import contextmanager
from math import isfinite
from typing import Any

from src.envs.core.multi_request_joint_control_env import MultiRequestJointControlEnv


@dataclass(frozen=True)
class JointRequestControl:
    execution_target: str | None = None  # RSU ID, "vehicle", or wait.
    cache_target: str | None = None
    cache_node_id: str | None = None
    cache_kind: str = "adapter"  # Adapter admission includes its base dependency.
    prepare_target: str | None = None
    evict_object_ids: tuple[str, ...] = ()
    priority: int = 0
    link_reservations: tuple[tuple[str, str], ...] = ()


class MultiRequestTargetControlEnv(MultiRequestJointControlEnv):
    action_contract = "joint_parameterized_control_v3"

    def __init__(self, *, input_bytes_per_link_frame: int = 1024,
                 output_bytes_per_link_frame: int = 1024, coordination_intent_enabled: bool = False, link_reservation_enabled: bool = False, **kwargs: Any) -> None:
        if any(not isfinite(value) or value < 1 or int(value) != value for value in
               (input_bytes_per_link_frame, output_bytes_per_link_frame)):
            raise ValueError("input/output rates must be positive integer bytes")
        if "environment_revision" in kwargs or "strict_cold_cache_mask" in kwargs:
            raise ValueError("target control uses its own v3 contract")
        self.coordination_intent_enabled = bool(coordination_intent_enabled)
        self.link_reservation_enabled = bool(link_reservation_enabled)
        self.input_bytes_per_link_frame = int(input_bytes_per_link_frame)
        self.output_bytes_per_link_frame = int(output_bytes_per_link_frame)
        super().__init__(environment_revision="v2", **kwargs)
        if self.link_reservation_enabled and self.costs.link_slots_per_rsu!=1:
            raise ValueError("scheduled link contract currently requires one slot per RSU")
        self.environment_revision = "v3"
        self.observation_contract = "joint_request_observation_v3"

    def reset(self) -> dict[str, Any]:
        observation = super().reset()
        self.input_remaining: dict[tuple[str, str, str], int] = {}
        self.input_received: set[tuple[str, str, str]] = set()
        self.output_pending: dict[str, dict[str, Any]] = {}
        self.previous_service_targets: dict[str, str | None] = {}
        self.io_totals: Counter = Counter()
        self.control_intent_bytes = 0
        return observation

    def step_actions(self, actions: dict[str, int]):
        raise ValueError("v3 requires JointRequestControl via step_controls; five-action policies are incompatible")

    def decision_states(self) -> dict[str, dict[str, Any]]:
        states = super().decision_states()
        vehicles = {vehicle.vehicle_id: vehicle for vehicle in self.frames[self.frame_index]["vehicles"]}
        for vid, info in states.items():
            state = info["semantic_state"]
            state.update(environment_revision="v3", observation_contract=self.observation_contract,
                         reachable_execution_rsu_ids=sorted(self._covered_rsus(vehicles[vid])),
                         known_cache_node_ids=[node.node_id for node in self.workflows[vid].nodes],
                         input_received_targets=sorted(target for owner, node, target in self.input_received
                                                       if owner == vid and node == self.workflows[vid].current_node_id),
                         output_pending=dict(self.output_pending.get(vid, {})))
            if self.link_reservation_enabled:
                state['input_remaining_bytes_by_rsu']={target:amount for (owner,node,target),amount in self.input_remaining.items() if owner==vid and node==self.workflows[vid].current_node_id}
            state['episode_frames_remaining']=len(self.frames)-self.frame_index
            state['control_cost_parameters'] = dict(rsu_work_units_per_frame=self.rsu_work_units_per_frame,
                profile_unit_bytes=self.workflow_profile_unit_bytes,workload_bytes_per_unit=self.workload_bytes_per_unit,
                state_bytes_per_frame=self.state_bytes_per_link_frame,minimum_state_frames=self.costs.handoff_state_frames,
                input_bytes_per_frame=self.input_bytes_per_link_frame,output_bytes_per_frame=self.output_bytes_per_link_frame,
                model_bytes_per_frame=self.model_cache_config.get('transfer_bytes_per_frame',1024),
                base_size_bytes=self.model_cache_config.get('base_size_bytes',2048),adapter_size_bytes=self.model_cache_config.get('adapter_size_bytes',512))
            info.pop("action_mask", None)
            info["action_contract"] = self.action_contract
            info["control_space"] = {
                "execution_targets": [None, *state["reachable_execution_rsu_ids"]],
                "vehicle_execution_available": bool(state["vehicle_adapter_ready"] and
                    vehicles[vid].base_model_id == state["current_workflow_node"]["required_base_model"]),
                "cache_and_prepare_targets": sorted(r.rsu_id for r in self.rsus),
                "cache_kinds": ["base_model", "adapter"], "priority_values": [0, 1, 2, 3],
            }
        return states

    def _validate_controls(self, controls: dict[str, JointRequestControl], decisions: dict) -> dict[str, list[str]]:
        if set(controls) != set(decisions):
            raise ValueError("one control is required for every active request")
        rsus = {r.rsu_id for r in self.rsus}
        evictions: dict[str, list[str]] = defaultdict(list)
        for vid, control in controls.items():
            if not isinstance(control, JointRequestControl):
                raise ValueError("controls must be JointRequestControl objects")
            if type(control.priority) is not int or control.priority not in range(4):
                raise ValueError("priority must be an integer in 0..3")
            if control.link_reservations and not self.link_reservation_enabled:
                raise ValueError("link reservations require scheduled environment")
            if len({rid for rid,kind in control.link_reservations})!=len(control.link_reservations):
                raise ValueError("one reservation per request and destination")
            if any(rid not in rsus or kind not in {'model','input','output','state'} for rid,kind in control.link_reservations):
                raise ValueError("invalid link reservation")
            space = decisions[vid]["control_space"]
            if control.execution_target == "vehicle":
                if not space["vehicle_execution_available"]:
                    raise ValueError("vehicle execution requires declared compatible base and adapter")
            elif control.execution_target not in space["execution_targets"]:
                raise ValueError("execution target must be a currently reachable RSU")
            if control.cache_kind not in {"adapter", "base_model"}:
                raise ValueError("unknown cache kind")
            if control.prepare_target is not None and control.prepare_target not in rsus:
                raise ValueError("unknown state preparation target")
            if control.cache_target is None:
                if control.cache_node_id is not None or control.evict_object_ids:
                    raise ValueError("cache node or eviction requires a cache target")
            else:
                if control.cache_target not in rsus:
                    raise ValueError("unknown cache target")
                if control.cache_node_id is not None and control.cache_node_id not in {
                    node.node_id for node in self.workflows[vid].nodes
                }:
                    raise ValueError("cache node must belong to the known DAG")
                evictions[control.cache_target].extend(control.evict_object_ids)
        # Validate the joint ordered list before mutation, including two requests
        # trying to evict the same object or a base with a remaining dependent.
        for target, victims in evictions.items():
            self.model_cache.validate_eviction(target, victims)
        return evictions

    def _input(self, vid: str, node: Any, target: str, link_left: dict[str, int]) -> dict:
        key = (vid, node.node_id, target)
        if key in self.input_received:
            return {"status": "input_available", "bytes": 0, "link_frames": 0}
        if link_left[target] < 1:
            return {"status": "input_link_queue", "bytes": 0, "link_frames": 0}
        remaining = self.input_remaining.get(key, max(1, node.input_size * self.workflow_profile_unit_bytes))
        moved = min(remaining, self.input_bytes_per_link_frame)
        link_left[target] -= 1
        self.input_remaining[key] = remaining - moved
        if self.input_remaining[key] == 0:
            self.input_received.add(key)
            self.input_remaining.pop(key)
        return {"status": "input_ready" if key in self.input_received else "input_transfer",
                "bytes": moved, "link_frames": 1}

    def _snapshot_execution(self, vid: str) -> None:
        self.state_copies[(vid, self.execution_location[vid])] = {
            "version": self.state_version[vid], "payload_bytes": self.state_payload_bytes[vid],
            "completed_node_ids": list(self.workflows[vid].completed_node_ids),
            "work_progress": self.work_progress[vid],
            "pending_output_node_id": self.output_pending.get(vid, {}).get("node_id"),
        }

    def _transfer_state_v2(self, vehicle_id: str, target: str, link_left: dict[str, int]) -> dict[str, Any]:
        result = super()._transfer_state_v2(vehicle_id, target, link_left)
        copy = self.state_copies.get((vehicle_id, target))
        if copy is not None and copy["version"] == self.state_version[vehicle_id]:
            copy["pending_output_node_id"] = self.output_pending.get(vehicle_id, {}).get("node_id")
        return result

    def _compute(self, vid: str, location: str, rate: float) -> bool:
        node = self.workflows[vid].current_node()
        self.work_progress[vid] += rate
        self.execution_location[vid] = location
        self.last_service_rsu[vid] = None if location.startswith("vehicle:") else location
        self.state_version[vid] += 1
        self.state_payload_bytes[vid] += 1
        completed = self.work_progress[vid] + 1e-12 >= self._work_required(node)
        if completed:
            output = max(1, node.output_size * self.workflow_profile_unit_bytes)
            self.state_payload_bytes[vid] += output
            if location.startswith("vehicle:"):
                self.workflows[vid].mark_current_completed()
                self.work_progress[vid] = 0.0
            else:
                self.output_pending[vid] = {"node_id": node.node_id, "rsu_id": location,
                                            "remaining_bytes": output}
                completed = False  # The node is delivered only after downlink.
        self._snapshot_execution(vid)
        return completed

    def _output(self, vid: str, reachable: list[str], link_left: dict[str, int],
                selected_target: str | None) -> dict:
        pending = self.output_pending[vid]
        target = pending["rsu_id"]
        if selected_target in reachable and selected_target != target:
            copy = self.state_copies.get((vid, selected_target), {})
            if not self._state_ready(vid, selected_target) or copy.get("pending_output_node_id") != pending["node_id"]:
                transfer = self._transfer_state_v2(vid, selected_target, link_left)
                return {"status": "state_restore" if transfer["link_frames"] else "state_link_queue",
                        "bytes": 0, "link_frames": 0, "completed": False, "state_transfer": transfer}
            target = selected_target
            pending["rsu_id"] = target
            self.execution_location[vid] = target
            self.last_service_rsu[vid] = target
        if target not in reachable:
            return {"status": "output_out_of_coverage", "bytes": 0, "link_frames": 0, "completed": False}
        if link_left[target] < 1:
            return {"status": "output_link_queue", "bytes": 0, "link_frames": 0, "completed": False}
        moved = min(pending["remaining_bytes"], self.output_bytes_per_link_frame)
        pending["remaining_bytes"] -= moved
        link_left[target] -= 1
        completed = pending["remaining_bytes"] == 0
        if completed:
            self.workflows[vid].mark_current_completed()
            self.work_progress[vid] = 0.0
            self.state_version[vid] += 1
            self.state_payload_bytes[vid] += 1
            self.output_pending.pop(vid)
            self._snapshot_execution(vid)
        return {"status": "node_completed" if completed else "output_transfer",
                "bytes": moved, "link_frames": 1, "completed": completed}

    @contextmanager
    def _link_scope(self, vid, kind, link_left):
        blocked={rid:link_left[rid] for rid,grant in getattr(self,'_link_grants',{}).items() if grant!=(vid,kind)}
        for rid in blocked:link_left[rid]=0
        try:yield
        finally:
            for rid,amount in blocked.items():link_left[rid]=amount

    def _control_record(self, control):
        record=asdict(control)
        if not self.link_reservation_enabled:record.pop('link_reservations')
        return record

    def step_controls(self, controls: dict[str, JointRequestControl]) -> tuple[dict, float, bool, dict]:
        if self.frame_index >= len(self.frames) or all(w.is_completed for w in self.workflows.values()):
            raise RuntimeError("episode ended")
        decisions = self.decision_states()
        evictions = self._validate_controls(controls, decisions)
        intent_bytes = 0
        if self.coordination_intent_enabled:
            import json
            from src.envs.core.joint_control_planning import workflow_service_frames
            intents = [{'vehicle_id':vid,'execution_target':control.execution_target,
                        'service_frames':workflow_service_frames(decisions[vid]['semantic_state'])}
                       for vid,control in controls.items()]
            if self.link_reservation_enabled:
                for intent in intents:
                    state=decisions[intent['vehicle_id']]['semantic_state']
                    from src.envs.core.joint_link_reservation import execution_link_need
                    intent.update(link_need=execution_link_need(state,controls[intent['vehicle_id']].execution_target),
                                  output_release_service_frames=state['current_node_service_steps_remaining'],
                                  node_output_bytes=state['current_workflow_node']['output_size']*self.workflow_profile_unit_bytes,link_reservations=controls[intent['vehicle_id']].link_reservations,
                                  output_pending=state['output_pending'],input_received_targets=state['input_received_targets'],
                                  input_remaining_bytes_by_rsu=state.get('input_remaining_bytes_by_rsu',{}),
                                  state_payload_bytes=state['execution_state_payload_bytes'],state_version=state['execution_state_version'])
            intent_bytes = len(json.dumps({'frame':self.frame_index,'intents':intents},sort_keys=True,separators=(',',':')).encode())
        occupied = self.model_cache.begin_frame()
        link_left = {r.rsu_id: self.costs.link_slots_per_rsu - occupied[r.rsu_id] for r in self.rsus}
        compute_left = {r.rsu_id: self.costs.compute_slots_per_rsu for r in self.rsus}
        eviction_receipts = {target: self.model_cache.evict_objects(target, victims)
                             for target, victims in evictions.items() if victims}
        ids = sorted(decisions)
        shift = self.frame_index % len(ids)
        tie_order = ids[shift:] + ids[:shift]
        order = sorted(tie_order, key=lambda vid: -controls[vid].priority)
        self._link_grants={}
        if self.link_reservation_enabled:
            for owner in order:
                for rid,kind in controls[owner].link_reservations:
                    self._link_grants.setdefault(rid,(owner,kind))
        events, completed_now = [], 0
        for vid in order:
            control = controls[vid]
            state = decisions[vid]["semantic_state"]
            node = self.workflows[vid].current_node()
            target = control.execution_target
            association = state["associations"][vid]
            handoff = vid in self.previous_associations and self.previous_associations[vid] != association
            changed = vid in self.previous_service_targets and self.previous_service_targets[vid] != target
            prepared = bool(target not in {None, "vehicle"} and changed and
                            self.state_version[vid] > 0 and self._state_ready(vid, target))
            version_before = self.state_version[vid]
            cache = {"status": "none", "link_slots_consumed": 0, "evicted_object_ids": []}
            if control.cache_node_id is not None:
                cache_node = next(n for n in self.workflows[vid].nodes if n.node_id == control.cache_node_id)
                with self._link_scope(vid,"model",link_left):
                    if control.cache_kind == "base_model":
                        cache = self.model_cache.request_base_model(control.cache_target,
                            cache_node.required_base_model, available_link_slots=link_left[control.cache_target])
                    else:
                        cache = self.model_cache.request_adapter(control.cache_target, cache_node.required_adapter,
                            cache_node.required_base_model, available_link_slots=link_left[control.cache_target])
                    link_left[control.cache_target] -= cache["link_slots_consumed"]
            transfers = []
            if control.prepare_target is not None:
                with self._link_scope(vid,"state",link_left):
                    transfers.append(self._transfer_state_v2(vid, control.prepare_target, link_left))
            input_result = {"bytes": 0, "link_frames": 0}
            output_result = {"bytes": 0, "link_frames": 0, "completed": False}
            output_source = None
            completed, status = False, "waiting"
            if vid in self.output_pending:
                phase="state" if target in state['reachable_execution_rsu_ids'] and target!=self.output_pending[vid]['rsu_id'] and not self._state_ready(vid,target) else "output"
                with self._link_scope(vid,phase,link_left):
                    output_result = self._output(vid, state["reachable_execution_rsu_ids"], link_left, target)
                output_source = self.output_pending.get(vid, {}).get("rsu_id", self.execution_location[vid])
                if "state_transfer" in output_result:
                    transfers.append(output_result["state_transfer"])
                completed, status = output_result["completed"], output_result["status"]
            elif not self._parents_ready(vid):
                status = "parents_not_ready"
            elif target == "vehicle":
                location = f"vehicle:{vid}"
                if not self._state_ready(vid, location):
                    if association is None:
                        status = "state_unavailable_no_coverage"
                    else:
                        with self._link_scope(vid,"state",link_left):
                            local_link = {location: link_left[association]}
                            transfer = self._transfer_state_v2(vid, location, local_link)
                            link_left[association] = local_link[location]
                        transfers.append(transfer)
                        status = "state_restore" if transfer["link_frames"] else "state_link_queue"
                else:
                    completed = self._compute(vid, location, self.vehicle_work_units_per_frame)
                    status = "vehicle_fallback_completed" if completed else "vehicle_compute"
            elif target is not None:
                if not self._state_ready(vid, target):
                    with self._link_scope(vid,"state",link_left):
                        transfer = self._transfer_state_v2(vid, target, link_left)
                    transfers.append(transfer)
                    status = "state_restore" if transfer["link_frames"] else "state_link_queue"
                elif not self.model_cache.is_ready(target, node.required_adapter, node.required_base_model):
                    status = "adapter_not_ready"
                elif (vid, node.node_id, target) not in self.input_received:
                    with self._link_scope(vid,"input",link_left):
                        input_result = self._input(vid, node, target, link_left)
                    status = input_result["status"]
                elif compute_left[target] < 1:
                    status = "compute_queue"
                else:
                    compute_left[target] -= 1
                    self.model_cache.touch(target, node.required_adapter, node.required_base_model)
                    self._compute(vid, target, self.rsu_work_units_per_frame)
                    status = "output_ready" if vid in self.output_pending else "rsu_compute"
            completed_now += int(completed)
            events.append({
                "frame_index": self.frame_index, "time_index": int(self.frames[self.frame_index]["time_index"]),
                "vehicle_id": vid, "request_id": f"{vid}/{self.workflows[vid].workflow_id}",
                "workflow_id": self.workflows[vid].workflow_id, "node_id": node.node_id,
                "associated_rsu_id": association, "target_rsu_id": target,
                "control": self._control_record(control), "cache_target_rsu_id": control.cache_target,
                "cache_action_status": cache["status"], "cache_request": cache,
                "explicit_evicted_object_ids": list(control.evict_object_ids),
                "state_action_status": transfers[-1]["status"] if transfers else "none",
                "state_transfers": transfers,
                "state_transfer_bytes": sum(t["bytes"] for t in transfers),
                "state_transfer_link_frames": sum(t["link_frames"] for t in transfers),
                "input_transfer_bytes": input_result["bytes"], "input_transfer_link_frames": input_result["link_frames"],
                "output_transfer_bytes": output_result["bytes"], "output_transfer_link_frames": output_result["link_frames"],
                "output_source_rsu_id": output_source,
                "state_version_before": version_before, "state_version_after": self.state_version[vid],
                "state_payload_bytes": self.state_payload_bytes[vid], "work_progress": self.work_progress[vid],
                "work_units_required": self._work_required(node), "association_handoff": handoff,
                "service_target_changed": changed, "prepared_on_arrival": prepared,
                "handoff_failed": bool(changed and status in {"state_restore", "state_link_queue"}),
                "status": status, "workflow_completed": self.workflows[vid].is_completed,
                "environment_revision": "v3",
            })
            self.previous_service_targets[vid] = target
        if self.link_reservation_enabled:
            for event in events:
                event['link_reservation_grants']={rid:dict(vehicle_id=owner,kind=kind,background_occupied=occupied[rid]) for rid,(owner,kind) in self._link_grants.items()}
        cache_events = self.model_cache.advance()
        link_usage = {target: self.costs.link_slots_per_rsu - left for target, left in link_left.items()}
        if any(not 0 <= used <= self.costs.link_slots_per_rsu for used in link_usage.values()):
            raise RuntimeError("shared data link overallocated")
        self._sync_cache_view()
        self.events.extend(events)
        self.previous_associations = {vid: decisions[vid]["semantic_state"]["associations"][vid] for vid in decisions}
        self.frame_index += 1
        done = self.frame_index >= len(self.frames) or all(w.is_completed for w in self.workflows.values())
        model_frames = sum(occupied.values()) + sum(e["cache_request"]["link_slots_consumed"] for e in events)
        sharing = self._frame_shared_state_bytes
        sharing += intent_bytes
        self.control_intent_bytes += intent_bytes
        self.model_link_frames_charged += model_frames
        self.sharing_bytes_charged += sharing
        components = {
            "completed_nodes": float(completed_now),
            "unfinished_wait": -0.01 * sum(not w.is_completed for w in self.workflows.values()),
            "vehicle_compute": -0.02 * sum(e["status"].startswith("vehicle_") for e in events),
            "model_transfer": -0.01 * model_frames,
            "state_transfer": -0.01 * sum(e["state_transfer_link_frames"] for e in events),
            "input_transfer": -0.01 * sum(e["input_transfer_link_frames"] for e in events),
            "output_transfer": -0.01 * sum(e["output_transfer_link_frames"] for e in events),
            "shared_state": -0.001 * sharing / 1024.0,
        }
        totals = {field: sum(e[field] for e in events) for field in (
            "input_transfer_bytes", "input_transfer_link_frames", "output_transfer_bytes", "output_transfer_link_frames")}
        self.io_totals.update(totals)
        from src.envs.core.joint_control_planning import request_reward_ledger
        credit = request_reward_ledger(events,components)
        events[0]["frame_cost_receipt"] = {
            **({"request_rewards": credit, "control_intent_transmitted_bytes": intent_bytes} if self.coordination_intent_enabled else {}),
            "contract_version": "joint_frame_cost_receipt_v3", "frame_index": events[0]["frame_index"],
            "link_usage_by_rsu": link_usage,
            "explicit_evictions_by_rsu": eviction_receipts,
            "reward_components": components, "cache_events": cache_events,
            "model_transfer_bytes": sum(e["transferred_bytes"] for e in cache_events),
            "model_transfer_link_frames": model_frames, "shared_state_transmitted_bytes": sharing,
            "state_transfer_bytes": sum(e["state_transfer_bytes"] for e in events),
            "state_transfer_link_frames": sum(e["state_transfer_link_frames"] for e in events), **totals,
        }
        observation = self.observe() if not done else {"frame_index": self.frame_index}
        return observation, sum(components.values()), done, {
            "events": events, "completed_nodes": completed_now, "reward_components": components,
            "cache_events": cache_events,
        }

    def summary(self) -> dict[str, Any]:
        summary = super().summary()
        summary.update(environment_revision="v3", observation_contract=self.observation_contract,
                       action_contract=self.action_contract, **self.io_totals)
        summary["pending_output_count"] = len(self.output_pending)
        summary["control_intent_transmitted_bytes"] = self.control_intent_bytes
        summary["service_target_change_count"] = sum(e["service_target_changed"] for e in self.events)
        summary["base_model_fetch_count"] = sum(e["cache_action_status"] == "base_model_fetch" for e in self.events)
        summary["explicit_eviction_count"] = sum(len(e["explicit_evicted_object_ids"]) for e in self.events)
        return summary

    def configuration(self) -> dict[str, Any]:
        return {**super().configuration(), "environment_revision": "v3",
                "input_bytes_per_link_frame": self.input_bytes_per_link_frame,
                "output_bytes_per_link_frame": self.output_bytes_per_link_frame,
                "completion_contract": "node_counts_only_after_output_delivery",
                "coordination_intent_enabled":self.coordination_intent_enabled,
                "communication_contract": "input_output_model_state_share_destination_rsu_link_slots",
                "control_channel": "shared_state_json_fee_only_no_data_link_reservation",
                "execution_contract": "currently_covered_rsu_or_declared_vehicle_or_wait",
                "cache_control": "explicit_base_or_known_dag_adapter_admission_and_dependency_safe_eviction",
                "reward_coefficients": {**super().configuration()["reward_coefficients"],
                    "input_transfer_link_frame": -0.01, "output_transfer_link_frame": -0.01}}
