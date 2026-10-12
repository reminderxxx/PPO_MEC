"""Opt-in, event-driven VEC workflow environment.

This environment uses controlled typed-cache objects and synthetic execution
parameters unless a measured cost profile is supplied. It never performs model
inference; output identities are deterministic stand-ins released only after
compute completion. Historical VecWorkflowCoreEnv remains independent.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Mapping

from src.data.mobility.handoff_builder import HandoffBuilder
from src.data.mobility.replay_provider import ReplayProvider
from src.data.mobility.rsu_mapper import RSUMapper
from src.data.model_catalog.adapter_catalog import AdapterCatalog, TYPED_MODEL_CACHE_PROFILE_ID
from src.envs.core.dag_execution import DagExecutionRuntime, TransferRequest
from src.envs.core.physical_cache_service import PhysicalTypedCache
from src.envs.core.physical_runtime import PhysicalRuntime
from src.envs.core.predictor_manager import PredictorManager
from src.envs.specs import ControlAction, RSUState, RewardBreakdown, WorkflowGraphState, WorkflowNode
from src.metrics.physical_event_metrics import PhysicalEvent, reduce_physical_events
from src.runtime.physical_cost_profile import (
    PhysicalCostProfile, ReplayIdentity, ReplayOutputGate, canonical_identity,
)


_MIB = 1024 * 1024


def _default_graph() -> WorkflowGraphState:
    adapters = ("adapter_perception", "adapter_tracking", "adapter_fusion")
    nodes = [
        WorkflowNode(
            node_id=f"n{index + 1}", node_name=f"physical_task_{index + 1}",
            required_base_model="veh_base_v1", required_adapter=adapter,
            input_size=1024, output_size=1024,
            predecessors=[] if index == 0 else [f"n{index}"],
            successors=[] if index == 2 else [f"n{index + 2}"],
        )
        for index, adapter in enumerate(adapters)
    ]
    return WorkflowGraphState(
        workflow_id="wf_physical_default", nodes=nodes,
        edges=[("n1", "n2"), ("n2", "n3")],
        execution_order=["n1", "n2", "n3"], current_node_id="n1",
    )


def _default_rsus() -> list[RSUState]:
    return [
        RSUState("rsu_a", 0.0, 0.0, 45.0),
        RSUState("rsu_b", 65.0, 0.0, 45.0),
        RSUState("rsu_c", 130.0, 0.0, 45.0),
    ]


class PhysicalVecWorkflowCoreEnv:
    """One primary-vehicle DAG with shared links, cache and compute resources."""

    physical_mode = True
    metrics_protocol_version = "physical_events_v1"

    def __init__(
        self,
        mobility_provider: ReplayProvider | None = None,
        workflow_state: WorkflowGraphState | None = None,
        adapter_catalog: AdapterCatalog | None = None,
        rsu_states: list[RSUState] | None = None,
        predictor_manager: PredictorManager | None = None,
        max_steps: int = 32,
        control_step_s: float = 1.0,
        backhaul_bytes_per_s: float = 100_000_000.0,
        wireless_bytes_per_s: float = 10_000_000.0,
        compute_slots_per_rsu: int | Mapping[str, int] = 1,
        compute_rate_work_units_per_s: float | Mapping[str, float] = 100.0,
        cache_capacity_bytes_per_rsu: float | Mapping[str, float] = 1_000_000_000.0,
        cost_profile: PhysicalCostProfile | None = None,
        replay_input_value: Any | None = None,
        generation_config: Mapping[str, Any] | None = None,
        replay_output: Any | None = None,
        root_input_payload_identity: str = "synthetic-root",
        background_jobs: list[dict[str, Any]] | None = None,
        mobility_source: str = "ngsim",
        primary_vehicle_selection: str = "stable_first",
        reward_positive_offset: float = 0.0,
        cache_capacity_profile: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be positive")
        for name, value in (
            ("control_step_s", control_step_s),
            ("backhaul_bytes_per_s", backhaul_bytes_per_s),
            ("wireless_bytes_per_s", wireless_bytes_per_s),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if primary_vehicle_selection not in {"stable_first", "handoff_pressure"}:
            raise ValueError("unsupported primary_vehicle_selection")
        if mobility_source not in {"ngsim", "alibaba", "toy", "lust", "highd"}:
            raise ValueError("unsupported mobility_source")
        if reward_positive_offset < 0 or not math.isfinite(reward_positive_offset):
            raise ValueError("reward_positive_offset must be finite and nonnegative")
        if cost_profile is not None and not isinstance(cost_profile, PhysicalCostProfile):
            raise ValueError("cost_profile must be PhysicalCostProfile")
        replay_supplied = any(value is not None for value in (replay_input_value, generation_config, replay_output))
        if cost_profile is None and replay_supplied:
            raise ValueError("replay inputs require a physical cost profile")
        if cost_profile is not None:
            if replay_input_value is None or generation_config is None or replay_output is None:
                raise ValueError("cost profile requires replay input, generation config and output")
            if len((workflow_state or _default_graph()).nodes) != 1:
                raise ValueError("single cost profile is only valid for a one-node workflow")
        if not root_input_payload_identity:
            raise ValueError("root_input_payload_identity is required")
        self._mobility_provider = mobility_provider or ReplayProvider()
        self._workflow_template = deepcopy(workflow_state or _default_graph())
        default_catalog_path = Path(__file__).resolve().parents[2] / "data/model_catalog/typed_model_cache_controlled.json"
        self._catalog_template = deepcopy(adapter_catalog or AdapterCatalog.from_json(default_catalog_path))
        if self._catalog_template.model_cache_profile_id != TYPED_MODEL_CACHE_PROFILE_ID:
            raise ValueError("physical environment requires a typed model-cache catalog")
        for node in self._workflow_template.nodes:
            adapter = self._catalog_template.get_typed_adapter(node.required_adapter)
            base = self._catalog_template.get_typed_base(node.required_base_model)
            if adapter.required_base_model_id != node.required_base_model or adapter.base_model_family != base.base_model_family:
                raise ValueError(f"workflow node {node.node_id} has incompatible typed model bundle")
        self._rsu_template = deepcopy(rsu_states or _default_rsus())
        if not self._rsu_template or len({item.rsu_id for item in self._rsu_template}) != len(self._rsu_template):
            raise ValueError("RSU IDs must be unique and nonempty")
        self._predictor = predictor_manager or PredictorManager()
        self._max_steps = max_steps
        self._control_step_s = float(control_step_s)
        self._backhaul_rate = float(backhaul_bytes_per_s)
        self._wireless_rate = float(wireless_bytes_per_s)
        self._compute_slots_config = compute_slots_per_rsu
        self._compute_rate_config = compute_rate_work_units_per_s
        self._cache_capacity_config = cache_capacity_bytes_per_rsu
        self._cost_profile = cost_profile
        self._replay_output = deepcopy(replay_output)
        self._replay_identity: ReplayIdentity | None = None
        self._replay_input_bytes: int | None = None
        self._replay_output_identity: str | None = None
        self._replay_output_bytes: int | None = None
        if cost_profile is not None:
            replay_node = self._workflow_template.nodes[0]
            self._replay_identity = ReplayIdentity.from_components(
                model_identity=f"{replay_node.required_base_model}:{replay_node.required_adapter}",
                input_value=replay_input_value, generation_config=generation_config,
            )
            cost_profile.validate_replay_identity(self._replay_identity)
            self._replay_input_bytes = len(json.dumps(
                replay_input_value, sort_keys=True, ensure_ascii=False, allow_nan=False,
            ).encode("utf-8"))
            self._replay_output_identity = canonical_identity(replay_output)
            self._replay_output_bytes = len(json.dumps(
                replay_output, sort_keys=True, ensure_ascii=False, allow_nan=False,
            ).encode("utf-8"))
        self._root_identity = root_input_payload_identity
        self._background_jobs = sorted(deepcopy(background_jobs or []), key=lambda item: float(item["time_s"]))
        self._mobility_source = mobility_source
        self._primary_selection = primary_vehicle_selection
        self._reward_positive_offset = float(reward_positive_offset)
        self._cache_capacity_profile = deepcopy(cache_capacity_profile)
        if cache_capacity_profile is not None:
            if cache_capacity_profile.get("model_cache_profile_id", TYPED_MODEL_CACHE_PROFILE_ID) != TYPED_MODEL_CACHE_PROFILE_ID:
                raise ValueError("cache profile must name typed model cache")
            if cache_capacity_profile.get("enabled") is False:
                raise ValueError("physical typed cache capacity cannot be disabled")
            if cache_capacity_profile.get("unit") != "mb":
                raise ValueError("physical cache_capacity_profile requires unit='mb'")
            capacity_mb = cache_capacity_profile.get("capacity_mb")
            if capacity_mb is None or float(capacity_mb) <= 0:
                raise ValueError("physical cache_capacity_profile requires positive capacity_mb")
            if cache_capacity_profile.get("eviction_policy", "lru") != "lru":
                raise ValueError("physical typed cache currently supports LRU only")
        self._mapper = RSUMapper(deepcopy(self._rsu_template))
        self._handoff_builder = HandoffBuilder()
        self.reset()

    @property
    def reward_positive_offset(self) -> float:
        return self._reward_positive_offset

    def _resource_values(self, value: Any, rsu_ids: list[str], name: str) -> dict[str, Any]:
        result = dict(value) if isinstance(value, Mapping) else {rsu_id: value for rsu_id in rsu_ids}
        if set(result) != set(rsu_ids):
            raise ValueError(f"{name} must name every RSU exactly once")
        return result

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if options:
            raise ValueError("physical reset options are not supported")
        self._episode_steps = 0
        self._background_index = 0
        self.workflow_state = deepcopy(self._workflow_template)
        self.adapter_catalog = deepcopy(self._catalog_template)
        self.rsu_states = deepcopy(self._rsu_template)
        self._predictor.reset()
        self._mapper.update_rsus(self.rsu_states)
        self.vehicles = self._mobility_provider.reset()
        self._associations = self._mapper.associate(self.vehicles)
        self._apply_associations()
        self._primary_vehicle_id = self._choose_primary_vehicle()
        primary = self._primary_vehicle()
        self._request_id = f"{self.workflow_state.workflow_id}:{self._primary_vehicle_id or 'none'}"
        self._events: list[PhysicalEvent] = []
        self._event_number = 0
        self._emit("request_created", 0.0)
        rsu_ids = [item.rsu_id for item in self.rsu_states]
        self._runtime = PhysicalRuntime(
            now_s=0.0,
            link_capacity_bytes_per_s={
                "backhaul": self._backhaul_rate,
                **{f"wireless:{rsu_id}": self._wireless_rate for rsu_id in rsu_ids},
            },
            compute_slots={
                **self._resource_values(self._compute_slots_config, rsu_ids, "compute_slots_per_rsu"),
                **({f"vehicle:{self._primary_vehicle_id}": 1} if primary else {}),
            },
            compute_rate_work_units_per_s={
                **self._resource_values(self._compute_rate_config, rsu_ids, "compute_rate_work_units_per_s"),
                **({f"vehicle:{self._primary_vehicle_id}": min(self._resource_values(self._compute_rate_config, rsu_ids, "compute_rate_work_units_per_s").values()) / 2} if primary else {}),
            },
        )
        initial = {rsu_id: self.adapter_catalog.get_initial_typed_residents(rsu_id) for rsu_id in rsu_ids}
        self._cache = PhysicalTypedCache(
            self.adapter_catalog, rsu_ids,
            self._resource_values(self._cache_capacity_config, rsu_ids, "cache_capacity_bytes_per_rsu"),
            initial_residents=initial,
        )
        self._dag = DagExecutionRuntime(self.workflow_state, primary.associated_rsu_id if primary else None)
        self._replay_gate = (
            ReplayOutputGate(profile=self._cost_profile, output=deepcopy(self._replay_output), earliest_completion_s=0.0)
            if self._cost_profile is not None else None
        )
        self._job_context: dict[str, dict[str, Any]] = {}
        self._node_jobs: dict[str, str] = {}
        self._input_jobs: set[tuple[str, str, str]] = set()
        self._forward_jobs: set[str] = set()
        self._delivery_job: str | None = None
        self._delivered_sink_ids: set[str] = set()
        self._waits: dict[str, tuple[str, str]] = {}
        if primary:
            for node in self.workflow_state.nodes:
                if not node.predecessors:
                    self._dag.register_root_input(
                        node.node_id, f"{self.workflow_state.workflow_id}:{node.node_id}:root",
                        self._replay_input_bytes if self._replay_input_bytes is not None else int(node.input_size),
                        self._replay_identity.input_identity if self._replay_identity else self._root_identity,
                        f"vehicle:{primary.vehicle_id}", 0.0,
                    )
        self._handoff_events: list[dict[str, Any]] = []
        self._finished = False
        self._failed = False
        self._last_state = self._state()
        return deepcopy(self._last_state), self._info(0)

    def _choose_primary_vehicle(self) -> str | None:
        if not self.vehicles:
            return None
        if self._primary_selection == "stable_first":
            return min(item.vehicle_id for item in self.vehicles)
        # Current-frame boundary distance is a causal handoff-pressure proxy.
        rsu_map = {item.rsu_id: item for item in self.rsu_states}
        def pressure(vehicle: Any) -> tuple[float, str]:
            rsu = rsu_map.get(vehicle.associated_rsu_id)
            if rsu is None:
                return (float("inf"), vehicle.vehicle_id)
            distance = math.dist((vehicle.position_x, vehicle.position_y), (rsu.position_x, rsu.position_y))
            return ((rsu.coverage_radius - distance) / max(vehicle.speed, 1.0), vehicle.vehicle_id)
        return min(self.vehicles, key=pressure).vehicle_id

    def _primary_vehicle(self) -> Any | None:
        return next((item for item in self.vehicles if item.vehicle_id == self._primary_vehicle_id), None)

    def _apply_associations(self) -> None:
        for rsu in self.rsu_states:
            rsu.active_vehicle_ids = []
        rsu_map = {item.rsu_id: item for item in self.rsu_states}
        for vehicle in self.vehicles:
            vehicle.associated_rsu_id = self._associations.get(vehicle.vehicle_id)
            if vehicle.associated_rsu_id in rsu_map:
                rsu_map[vehicle.associated_rsu_id].active_vehicle_ids.append(vehicle.vehicle_id)

    def _emit(
        self, event_type: str, time_s: float,
        request_id_override: str | None = None, **fields: Any,
    ) -> PhysicalEvent:
        self._event_number += 1
        event = PhysicalEvent(
            event_id=f"{self._request_id}:event:{self._event_number}",
            event_type=event_type, time_s=float(time_s),
            request_id=request_id_override or self._request_id,
            workflow_id=self.workflow_state.workflow_id, **fields,
        )
        self._events.append(event)
        return event

    def _current_node_id(self) -> str | None:
        return next((node_id for node_id in self.workflow_state.execution_order
                     if self._dag.state(node_id).status != "completed"), None)

    def _prediction(self) -> dict[str, Any]:
        sequences = self._predictor.predict_next_rsu_sequence(self.vehicles, self.rsu_states)
        first = {vehicle_id: next((item for item in sequence if item is not None), None)
                 for vehicle_id, sequence in sequences.items()}
        current = {vehicle.vehicle_id: vehicle.associated_rsu_id for vehicle in self.vehicles}
        handoff_target = {
            vehicle_id: next((item for item in sequence if item is not None and item != current.get(vehicle_id)), None)
            for vehicle_id, sequence in sequences.items()
        }
        return {
            "next_rsu_sequence": sequences,
            "predicted_next_rsu_by_vehicle": first,
            "predicted_first_handoff_rsu_by_vehicle": handoff_target,
            "predicted_handoff_vehicle_ids": [vehicle_id for vehicle_id, item in handoff_target.items() if item],
            "future_load": {rsu.rsu_id: float(len(rsu.active_vehicle_ids)) for rsu in self.rsu_states},
            "dwell_time_by_vehicle": self._predictor.predict_dwell_time(self.vehicles, self.rsu_states),
            "prediction_evidence": "current_and_past_frames_only",
        }

    def _state(self) -> dict[str, Any]:
        current_id = self._current_node_id()
        graph = self.workflow_state.to_dict()
        graph["completed_node_ids"] = [item for item in self.workflow_state.execution_order
                                        if self._dag.state(item).status == "completed"]
        graph["current_node_id"] = current_id
        graph["is_completed"] = self._dag.is_completed
        snapshot = self._runtime.snapshot()
        cache = self._cache.snapshot()
        rsu_rows = []
        for rsu in self.rsu_states:
            rsu_cache = cache["rsus"][rsu.rsu_id]
            objects = rsu_cache["objects"]
            loading = [item for item in objects if item["state"] in {"loading", "loaded_waiting_dependency"}]
            adapter_ids = sorted({
                self.adapter_catalog.get_typed_object(item["object_id"]).adapter_id
                for item in loading
                if self.adapter_catalog.get_typed_object(item["object_id"]).adapter_id
            })
            rsu_rows.append({
                "rsu_id": rsu.rsu_id,
                "cache_used_bytes": rsu_cache["used_bytes"],
                "cache_reserved_bytes": sum(float(item["resident_bytes"]) for item in loading),
                "model_loading_count": len(loading),
                "loading_adapter_ids": adapter_ids,
                "compute_running_count": sum(1 for item in snapshot["computes"] if item["resource_id"] == rsu.rsu_id and item["state"] == "running"),
                "compute_queued_count": sum(1 for item in snapshot["computes"] if item["resource_id"] == rsu.rsu_id and item["state"] == "queued"),
            })
            rsu.cached_adapter_ids = sorted({
                self.adapter_catalog.get_typed_object(item["object_id"]).adapter_id
                for item in objects
                if self.adapter_catalog.get_typed_object(item["object_id"]).adapter_id
                and item["state"] == "resident"
            })
        links = [{
            "link_id": link_id, "capacity_bytes_per_s": capacity,
            "active_transfer_count": sum(1 for item in snapshot["transfers"] if item["resource_id"] == link_id),
        } for link_id, capacity in snapshot["link_capacity_bytes_per_s"].items()]
        physical_state = {
            "time_s": self._runtime.now_s,
            "environment_version": "physical_vec_env_v1",
            "cost_evidence": "synthetic_physics_with_profiled_hot_call" if self._cost_profile else "synthetic",
            "payload_size_evidence": "serialized_replay_payload" if self._replay_gate else "synthetic_workflow_size_mapping",
            "rsus": rsu_rows, "links": links,
            "nodes": [asdict(self._dag.state(item)) for item in self.workflow_state.execution_order],
            "handoff_receipts": [asdict(item) for item in self._dag.handoff_receipts],
            "in_flight": snapshot,
        }
        return {
            "time_index": self._mobility_provider.get_time(),
            "primary_vehicle_id": self._primary_vehicle_id,
            "vehicles": [item.to_dict() for item in self.vehicles],
            "rsus": [item.to_dict() for item in self.rsu_states],
            "workflow": graph,
            "current_workflow_node": self._dag.nodes[current_id].to_dict() if current_id else None,
            "predictions": self._prediction(),
            "handoff_events": deepcopy(self._handoff_events),
            "physical_state": physical_state,
        }

    def _info(self, start_index: int) -> dict[str, Any]:
        trace = self.export_physical_events()
        step_events = trace[start_index:]
        primary = self._primary_vehicle()
        current = self._current_node_id()
        current_execution = self._dag.state(current) if current else None
        current_adapter = self._dag.nodes[current].required_adapter if current else None
        warm_hit = bool(
            primary and primary.associated_rsu_id and current_adapter
            and self._cache.is_ready(primary.associated_rsu_id, current_adapter)
        )
        backhaul_bytes = sum(
            int(item.get("bytes_delta") or 0) for item in step_events
            if item["event_type"] in {"transfer_progress", "transfer_completed"}
            and item.get("resource_id") == "backhaul"
        )
        state_bytes = sum(
            int(item.get("bytes_delta") or 0) for item in step_events
            if item["event_type"] in {"transfer_progress", "transfer_completed"}
            and item.get("transfer_kind") == "transfer_state"
        )
        compat = {
            "contract_version": self.metrics_protocol_version,
            "handoff_event_count": sum(item["event_type"] == "handoff" for item in self._handoff_events),
            "stall_occurred": bool(current_execution and current_execution.status.startswith("waiting_")),
            "handoff_failed": bool(self._handoff_events and primary and primary.associated_rsu_id is None),
            "handoff_ready": bool(self._handoff_events and warm_hit),
            "warm_hit": warm_hit,
            "cross_rsu_cold_start": bool(self._handoff_events and not warm_hit),
            "backhaul_traffic_cost": float(backhaul_bytes),
            "adapter_state_migration_overhead": float(state_bytes),
            "predictive_prefetch_requested": False,
            "predictive_prefetch_validated": False,
            "predictive_prefetch_correct": False,
        }
        physical_summary = reduce_physical_events(self._events).to_dict()
        workflow_row = next(
            (item for item in physical_summary["requests"] if item["request_id"] == self._request_id),
            None,
        )
        return {
            "metrics_protocol": compat,
            "physical_events": step_events,
            "physical_event_trace": trace,
            "physical_event_summary": physical_summary,
            "workflow_request_id": self._request_id,
            "workflow_physical_metrics": workflow_row,
            "physical_cost_profile_id": self._cost_profile.profile_id if self._cost_profile else "synthetic_unmeasured",
            "physical_cost_evidence": "synthetic_physics_with_profiled_hot_call" if self._cost_profile else "synthetic",
            "hot_call_profile_evidence": self._cost_profile.hot_call.evidence if self._cost_profile else "synthetic",
            "payload_size_evidence": "serialized_replay_payload" if self._replay_gate else "synthetic_workflow_size_mapping",
            "environment_version": "physical_vec_env_v1",
            "cache_event": None,
            **compat,
        }

    def export_physical_events(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._events]

    def export_workflow_physical_metrics(self) -> dict[str, Any]:
        """Return the primary workflow row, excluding still-running preparations."""
        summary = reduce_physical_events(self._events)
        for row in summary.requests:
            if row.request_id == self._request_id:
                return row.to_dict()
        raise RuntimeError("primary workflow request is missing from physical ledger")

    def read_replay_output(self, node_id: str, time_s: float, identity: ReplayIdentity) -> Any:
        """Expose an exact-identity replay output only after compute completion."""
        if self._replay_gate is None or node_id != self.workflow_state.nodes[0].node_id:
            raise ValueError("no replay output configured for node")
        return self._replay_gate.read(time_s=time_s, identity=identity)

    def export_cache_trace_snapshot(self) -> dict[str, Any]:
        cache = self._cache.snapshot()
        rows = []
        for rsu_id, data in cache["rsus"].items():
            rows.append({
                "rsu_id": rsu_id, "capacity_enabled": True, "capacity_unit": "byte",
                "capacity": cache["capacity_bytes_per_rsu"][rsu_id],
                "residents": [dict(item) for item in data["objects"] if item["state"] == "resident"],
            })
        return {"snapshot_step_index": self._episode_steps, "rsus": rows}

    def _start_transfer(
        self, *, size_bytes: float, kind: str, object_id: str,
        node_id: str | None, link_id: str, context: dict[str, Any],
    ) -> str:
        total = int(round(size_bytes))
        job_id = self._runtime.submit_transfer(link_id, total, kind, context)
        ledger_request_id = (
            f"{self._request_id}:{context.get('detached_kind', 'prepare')}:{job_id}"
            if context.get("detached_prepare") else self._request_id
        )
        delivery_operation_id = f"prepare_delivery:{job_id}" if context.get("detached_prepare") else None
        self._job_context[job_id] = {
            **context, "kind": kind, "node_id": node_id, "object_id": object_id,
            "total": total, "emitted": 0,
            "ledger_request_id": ledger_request_id,
            "prepare_delivery_operation_id": delivery_operation_id,
        }
        if delivery_operation_id is not None:
            self._emit("request_created", self._runtime.now_s, request_id_override=ledger_request_id)
            self._emit(
                "delivery_started", self._runtime.now_s,
                request_id_override=ledger_request_id, operation_id=delivery_operation_id,
                node_id=node_id, object_id=object_id, resource_id=link_id,
            )
        self._emit(
            "transfer_started", self._runtime.now_s,
            request_id_override=ledger_request_id, operation_id=job_id,
            node_id=node_id, object_id=object_id, resource_id=link_id,
            transfer_kind=kind, total_bytes=total, bytes_delta=0,
        )
        return job_id

    def _link_for(self, source: str, target: str) -> str:
        if source.startswith("vehicle:"):
            return f"wireless:{target}"
        if target.startswith("vehicle:"):
            return f"wireless:{source}"
        return "backhaul"

    def _schedule_dag_transfer(self, request: TransferRequest, *, kind_override: str | None = None) -> str:
        kind = kind_override or request.kind
        link = self._link_for(request.source_rsu_id, request.target_rsu_id)
        return self._start_transfer(
            size_bytes=request.size_bytes, kind=kind, object_id=request.object_id,
            node_id=request.node_id, link_id=link,
            context={"dag_request_id": request.request_id, "source_rsu_id": request.source_rsu_id,
                     "target_rsu_id": request.target_rsu_id},
        )

    def _submit_cache(self, rsu_id: str, adapter_id: str, *, detached: bool = False) -> dict[str, Any]:
        return self._cache.request(
            rsu_id, adapter_id, self._runtime.now_s,
            submit_transfer=lambda size, metadata: self._start_transfer(
                size_bytes=size, kind="model_load", object_id=str(metadata["object_id"]),
                node_id=self._current_node_id(), link_id="backhaul",
                context={**metadata, "detached_prepare": detached},
            ),
        )

    def _apply_action(self, control: ControlAction) -> dict[str, Any]:
        node_id = self._current_node_id()
        primary = self._primary_vehicle()
        if node_id is None or primary is None:
            return {"status": "no_current_node_or_vehicle"}
        node = self._dag.nodes[node_id]
        action_id = control.metadata.get("action_id")
        if control.metadata.get("invalid_action"):
            return {"status": "invalid", "reason": control.metadata.get("invalid_reason")}
        if action_id is None:
            strategy = control.cache_action.get("strategy")
            action_id = {"reactive_cache_fill": 0, "predictive_prefetch": 1,
                         "handoff_prepare_prefetch": 4}.get(strategy, 3)
        if action_id not in {0, 1, 2, 3, 4}:
            return {"status": "invalid", "reason": "unknown_action_id"}
        current_rsu = primary.associated_rsu_id
        if action_id == 2:
            target = f"vehicle:{primary.vehicle_id}"
            execution = self._dag.state(node_id)
            if execution.status not in {"running", "completed"}:
                self._dag.deploy_node(node_id, target)
                self._dag.set_model_ready(node_id, target)
            return {"status": "vehicle_execution", "target_rsu_id": target}
        if current_rsu is None:
            return {"status": "uncovered", "reason": "no_associated_rsu"}
        execution = self._dag.state(node_id)
        if execution.deployment_rsu_id is None:
            self._dag.deploy_node(node_id, current_rsu)
        elif execution.status not in {"running", "completed"} and execution.deployment_rsu_id != current_rsu:
            self._dag.deploy_node(node_id, current_rsu)
        if action_id in {1, 4}:
            prediction = self._last_state.get("predictions", {})
            key = "predicted_next_rsu_by_vehicle" if action_id == 1 else "predicted_first_handoff_rsu_by_vehicle"
            target = prediction.get(key, {}).get(primary.vehicle_id)
            if target is None or target == current_rsu:
                return {"status": "invalid", "reason": "missing_distinct_predicted_target"}
        else:
            target = current_rsu
        if action_id in {0, 1, 4}:
            result = self._submit_cache(target, node.required_adapter, detached=action_id in {1, 4})
            if action_id == 4:
                predecessors = node.predecessors
                for predecessor in predecessors:
                    previous = self._dag.state(predecessor)
                    if previous.status == "completed" and previous.output_object_id is not None:
                        request = self._dag.request_transfer_state(predecessor, target, self._runtime.now_s)
                        self._schedule_dag_transfer(self._dag.pending_transfers[request.request_id])
            return {**result, "action_id": action_id}
        return {"status": "steady_offload", "target_rsu_id": current_rsu}

    def _update_waits(self) -> None:
        for node_id in self.workflow_state.execution_order:
            status = self._dag.state(node_id).status
            reason = status if status.startswith("waiting_") or status == "queued" else None
            old = self._waits.get(node_id)
            if old is not None and old[0] != reason:
                self._emit("wait_ended", self._runtime.now_s, node_id=node_id,
                           operation_id=old[1], wait_reason=old[0])
                del self._waits[node_id]
            if reason is not None and node_id not in self._waits:
                operation_id = f"wait:{node_id}:{self._event_number + 1}"
                self._waits[node_id] = (reason, operation_id)
                self._emit("wait_started", self._runtime.now_s, node_id=node_id,
                           operation_id=operation_id, wait_reason=reason)

    def _drive(self) -> None:
        primary = self._primary_vehicle()
        for node_id in self.workflow_state.execution_order:
            execution = self._dag.state(node_id)
            if execution.status in {"completed", "running"}:
                continue
            if execution.deployment_rsu_id is None and primary and primary.associated_rsu_id:
                if not self._dag.nodes[node_id].predecessors or all(
                    self._dag.state(pred).status == "completed" for pred in self._dag.nodes[node_id].predecessors
                ):
                    self._dag.deploy_node(node_id, primary.associated_rsu_id)
            target = execution.deployment_rsu_id
            if target is None:
                continue
            if target.startswith("vehicle:"):
                if not execution.model_ready:
                    self._dag.set_model_ready(node_id, target)
            elif self._cache.is_ready(target, self._dag.nodes[node_id].required_adapter) and not execution.model_ready:
                self._dag.set_model_ready(node_id, target)
            if execution.status == "waiting_input":
                for predecessor in self._dag.nodes[node_id].predecessors or [None]:
                    object_id = (
                        self._dag.state(predecessor).output_object_id if predecessor is not None
                        else self._dag._root_inputs.get(node_id)
                    )
                    if object_id is None or target in self._dag.object_locations[object_id]:
                        continue
                    key = (node_id, object_id, target)
                    if key not in self._input_jobs:
                        request = self._dag.request_input_transfer(node_id, predecessor, self._runtime.now_s)
                        self._schedule_dag_transfer(request)
                        self._input_jobs.add(key)
            if execution.status == "queued" and node_id not in self._node_jobs:
                work_units = self._work_units(node_id, target)
                job_id = self._runtime.submit_compute(target, work_units, {"node_id": node_id})
                self._node_jobs[node_id] = job_id
                self._job_context[job_id] = {
                    "kind": "compute", "node_id": node_id, "total": work_units,
                    "emitted": 0.0, "started": False,
                    "ledger_request_id": self._request_id,
                }
        self._process_compute_starts()
        if self._dag.is_completed and not self._finished and not self._forward_jobs and self._delivery_job is None:
            for node_id in self.workflow_state.execution_order:
                if any(source == node_id for source, _ in self.workflow_state.edges):
                    continue
                if node_id in self._delivered_sink_ids:
                    continue
                output = self._dag.state(node_id).output_object_id
                if output is None:
                    continue
                if primary and primary.associated_rsu_id:
                    request = self._dag.request_forward_result(node_id, primary.associated_rsu_id, self._runtime.now_s)
                    self._schedule_dag_transfer(self._dag.pending_transfers[request.request_id], kind_override="result_forward")
                    self._forward_jobs.add(output)
                    break
        self._update_waits()

    def _work_units(self, node_id: str, target: str) -> float:
        if self._cost_profile is not None:
            rate = self._runtime.compute_rate_work_units_per_s[target]
            return max(self._cost_profile.hot_call.seconds * rate, 1e-9)
        node = self._dag.nodes[node_id]
        return float(max(node.input_size + node.output_size, 1))

    def _process_compute_starts(self) -> None:
        for job in self._runtime.snapshot()["computes"]:
            context = self._job_context.get(job["job_id"])
            if context is None or context.get("kind") not in {"compute", "background_compute"} or context.get("started"):
                continue
            if job["state"] != "running":
                continue
            node_id = context["node_id"]
            if context["kind"] == "compute":
                self._dag.on_compute_start(node_id, float(job["started_at_s"]))
            context["started"] = True
            self._emit(
                "compute_started", float(job["started_at_s"]),
                request_id_override=context["ledger_request_id"], node_id=node_id,
                operation_id=job["job_id"], resource_id=job["resource_id"],
                total_work_units=context["total"], work_units_delta=0.0,
            )
            if context["kind"] == "compute" and not job["resource_id"].startswith("vehicle:"):
                self._cache.pin(job["resource_id"], self._dag.nodes[node_id].required_adapter)

    def _flush_progress(self) -> None:
        snapshot = self._runtime.snapshot()
        for group in ("transfers", "computes"):
            for job in snapshot[group]:
                context = self._job_context.get(job["job_id"])
                if context is None:
                    continue
                if group == "transfers":
                    observed = min(context["total"], int(math.floor(job["completed"] + 1e-7)))
                    delta = observed - context["emitted"]
                    if delta > 0:
                        context["emitted"] = observed
                        self._emit(
                            "transfer_progress", self._runtime.now_s,
                            request_id_override=context["ledger_request_id"],
                            node_id=context.get("node_id"), operation_id=job["job_id"],
                            object_id=context["object_id"], resource_id=job["resource_id"],
                            transfer_kind=context["kind"], total_bytes=context["total"], bytes_delta=delta,
                        )
                elif context.get("started"):
                    observed = min(context["total"], float(job["completed"]))
                    delta = observed - context["emitted"]
                    if delta > 1e-9:
                        context["emitted"] = observed
                        self._emit(
                            "compute_progress", self._runtime.now_s,
                            request_id_override=context["ledger_request_id"],
                            node_id=context["node_id"], operation_id=job["job_id"],
                            resource_id=job["resource_id"],
                            total_work_units=context["total"], work_units_delta=delta,
                        )

    def _handle_completion(self, event: dict[str, Any]) -> None:
        job_id = event["job_id"]
        context = self._job_context.pop(job_id, None)
        if context is None:
            return  # explicit background work has no workflow ledger operation
        time_s = float(event["time_s"])
        if event["resource_type"] == "transfer":
            self._emit(
                "transfer_completed", time_s,
                request_id_override=context["ledger_request_id"],
                node_id=context.get("node_id"),
                operation_id=job_id, object_id=context["object_id"],
                resource_id=event["resource_id"], transfer_kind=context["kind"],
                total_bytes=context["total"], bytes_delta=context["total"] - context["emitted"],
            )
            if context["kind"] == "model_load":
                self._cache.on_transfer_complete(job_id, time_s)
            if context["prepare_delivery_operation_id"] is not None:
                self._emit(
                    "delivery_completed", time_s,
                    request_id_override=context["ledger_request_id"],
                    operation_id=context["prepare_delivery_operation_id"],
                    node_id=context["node_id"], object_id=context["object_id"],
                    resource_id=event["resource_id"],
                )
                self._emit("request_completed", time_s, request_id_override=context["ledger_request_id"])
            elif context["kind"] in {"input", "transfer_state", "result_forward"}:
                dag_request_id = context["dag_request_id"]
                request = self._dag.pending_transfers[dag_request_id]
                receipt = self._dag.on_transfer_complete(dag_request_id, time_s)
                if context["kind"] == "input":
                    self._input_jobs.discard((request.node_id, request.object_id, request.target_rsu_id))
                elif context["kind"] == "result_forward":
                    if receipt is not None and receipt.delivered_to_current_access:
                        self._start_wireless_delivery(request.node_id, request.object_id)
                    else:
                        self._forward_jobs.discard(request.object_id)
            elif context["kind"] == "wireless_delivery":
                primary = self._primary_vehicle()
                delivered = bool(primary and primary.associated_rsu_id == context["target_access_rsu_id"])
                self._emit(
                    "delivery_completed" if delivered else "delivery_failed", time_s, node_id=context["node_id"],
                    operation_id=context["delivery_operation_id"], object_id=context["object_id"],
                    resource_id=event["resource_id"],
                )
                self._forward_jobs.discard(context["object_id"])
                self._delivery_job = None
                if not delivered:
                    self._emit("request_failed", time_s)
                    self._failed = True
                    self._finished = True
                    return
                self._delivered_sink_ids.add(context["node_id"])
                sink_ids = {
                    item for item in self.workflow_state.execution_order
                    if not any(source == item for source, _ in self.workflow_state.edges)
                }
                if self._delivered_sink_ids == sink_ids:
                    self._emit("request_completed", time_s)
                    self._finished = True
        else:
            node_id = context["node_id"]
            self._emit(
                "compute_completed", time_s,
                request_id_override=context["ledger_request_id"],
                node_id=node_id, operation_id=job_id,
                resource_id=event["resource_id"], total_work_units=context["total"],
                work_units_delta=context["total"] - context["emitted"],
            )
            if context["kind"] == "background_compute":
                self._emit(
                    "delivery_completed", time_s,
                    request_id_override=context["ledger_request_id"],
                    operation_id=context["prepare_delivery_operation_id"],
                    node_id=node_id, resource_id=event["resource_id"],
                )
                self._emit("request_completed", time_s, request_id_override=context["ledger_request_id"])
                return
            if not event["resource_id"].startswith("vehicle:"):
                self._cache.unpin(event["resource_id"], self._dag.nodes[node_id].required_adapter)
            input_ids = self._dag.state(node_id).input_object_ids
            identity = canonical_identity({
                "workflow": self.workflow_state.workflow_id, "node": node_id,
                "inputs": [self._dag.objects[item].payload_identity for item in input_ids],
                "execution_count": self._dag.state(node_id).execution_count + 1,
            })
            output_size = int(self._dag.nodes[node_id].output_size)
            if self._replay_gate is not None:
                assert self._replay_identity is not None
                self._replay_gate.mark_completed(time_s=time_s, identity=self._replay_identity)
                actual_output = self._replay_gate.read(time_s=time_s, identity=self._replay_identity)
                assert self._replay_output_identity is not None and self._replay_output_bytes is not None
                if canonical_identity(actual_output) != self._replay_output_identity:
                    raise ValueError("replay output changed after profile validation")
                identity = self._replay_output_identity
                output_size = self._replay_output_bytes
            self._dag.on_compute_complete(
                node_id, f"{self.workflow_state.workflow_id}:{node_id}:output:{self._dag.state(node_id).execution_count + 1}",
                identity, time_s, size_bytes=output_size,
            )
            self._node_jobs.pop(node_id, None)

    def _start_wireless_delivery(self, node_id: str, object_id: str) -> None:
        primary = self._primary_vehicle()
        if primary is None or primary.associated_rsu_id is None or self._delivery_job is not None:
            return
        link = f"wireless:{primary.associated_rsu_id}"
        self._emit(
            "delivery_started", self._runtime.now_s, node_id=node_id,
            operation_id=f"delivery:pending:{self._event_number + 1}",
            object_id=object_id, resource_id=link,
        )
        operation_id = self._events[-1].operation_id
        job_id = self._start_transfer(
            size_bytes=self._dag.objects[object_id].size_bytes,
            kind="wireless_delivery", object_id=object_id, node_id=node_id,
            link_id=link, context={"delivery_operation_id": operation_id},
        )
        self._job_context[job_id]["delivery_operation_id"] = operation_id
        self._job_context[job_id]["target_access_rsu_id"] = primary.associated_rsu_id
        self._delivery_job = job_id

    def _submit_background_jobs(self, until_s: float) -> None:
        while self._background_index < len(self._background_jobs):
            row = self._background_jobs[self._background_index]
            if float(row["time_s"]) > until_s:
                break
            if float(row["time_s"]) != self._runtime.now_s:
                break
            self._background_index += 1
            if row["kind"] == "transfer":
                self._start_transfer(
                    size_bytes=row["size_bytes"], kind="background_transfer",
                    object_id=f"background:{self._background_index}", node_id=None,
                    link_id=row["resource_id"],
                    context={"vehicle_id": row.get("vehicle_id"),
                             "detached_prepare": True, "detached_kind": "background"},
                )
            elif row["kind"] == "compute":
                node_id = f"background:{self._background_index}"
                work = float(row["work_units"])
                job_id = self._runtime.submit_compute(
                    row["resource_id"], work, {"vehicle_id": row.get("vehicle_id")},
                )
                ledger_id = f"{self._request_id}:background:{job_id}"
                delivery_id = f"background_delivery:{job_id}"
                self._job_context[job_id] = {
                    "kind": "background_compute", "node_id": node_id,
                    "total": work, "emitted": 0.0, "started": False,
                    "ledger_request_id": ledger_id,
                    "prepare_delivery_operation_id": delivery_id,
                }
                self._emit("request_created", self._runtime.now_s, request_id_override=ledger_id)
                self._emit(
                    "delivery_started", self._runtime.now_s,
                    request_id_override=ledger_id, operation_id=delivery_id,
                    node_id=node_id, resource_id=row["resource_id"],
                )
            else:
                raise ValueError("background kind must be transfer or compute")

    def _advance_until(self, target_s: float) -> None:
        guard = 0
        while (self._runtime.now_s < target_s or self._runtime.next_completion_time() == self._runtime.now_s) and not self._finished:
            guard += 1
            if guard > 10000:
                raise RuntimeError("physical event loop did not converge")
            self._drive()
            next_completion = self._runtime.next_completion_time()
            next_background = (
                float(self._background_jobs[self._background_index]["time_s"])
                if self._background_index < len(self._background_jobs) else None
            )
            candidates = [target_s]
            if next_completion is not None:
                candidates.append(next_completion)
            if next_background is not None and next_background >= self._runtime.now_s:
                candidates.append(next_background)
            boundary = min(candidates)
            completions = self._runtime.advance_to(boundary)
            self._flush_progress()
            for event in completions:
                self._handle_completion(event)
            self._submit_background_jobs(boundary)
            self._process_compute_starts()
            if boundary == target_s and not completions:
                break
        if not self._finished:
            self._drive()

    def step(self, control: ControlAction) -> tuple[dict[str, Any], RewardBreakdown, bool, bool, dict[str, Any]]:
        if self._finished or self._episode_steps >= self._max_steps:
            raise RuntimeError("episode already ended")
        event_start = len(self._events)
        self._episode_steps += 1
        action_result = self._apply_action(control)
        target_s = self._episode_steps * self._control_step_s
        self._advance_until(target_s)
        self._handoff_events = []
        if not self._finished:
            previous = dict(self._associations)
            self.vehicles = self._mobility_provider.step()
            self._associations = self._mapper.associate(self.vehicles)
            self._apply_associations()
            self._handoff_events = [item.to_dict() for item in self._handoff_builder.build_events(
                previous, self._associations, self._mobility_provider.get_time(),
            )]
            primary = self._primary_vehicle()
            if primary is not None:
                self._dag.on_access_handoff(primary.associated_rsu_id)
            for event in self._handoff_events:
                if event["vehicle_id"] == self._primary_vehicle_id:
                    operation_id = f"handoff:{self._event_number + 1}"
                    self._emit("wait_started", target_s, operation_id=operation_id,
                               wait_reason="handoff", node_id=self._current_node_id())
                    self._emit("wait_ended", target_s, operation_id=operation_id,
                               wait_reason="handoff", node_id=self._current_node_id())
        self._last_state = self._state()
        reward = RewardBreakdown(
            total=self._reward_positive_offset + (1.0 if self._finished and not self._failed else 0.0) - self._control_step_s,
            positive_offset=self._reward_positive_offset,
            service_reward=1.0 if self._finished and not self._failed else 0.0,
            delay_penalty=self._control_step_s,
            cache_miss_penalty=0.0, migration_cost=0.0,
            continuity_bonus=0.0, mechanism_exploration_bonus=0.0, constraint_penalty=0.0,
        )
        truncated = self._failed or (self._episode_steps >= self._max_steps and not self._finished)
        info = self._info(event_start)
        info["cache_action_result"] = action_result
        action_id = control.metadata.get("action_id")
        info["predictive_prefetch_requested"] = action_id in {1, 4} and action_result.get("status") in {"ready", "loading"}
        info["predictive_prefetch_validated"] = bool(info["predictive_prefetch_requested"] and self._handoff_events)
        info["predictive_prefetch_correct"] = bool(
            info["predictive_prefetch_validated"] and self._primary_vehicle()
            and action_result.get("rsu_id") == self._primary_vehicle().associated_rsu_id
        )
        for key in ("predictive_prefetch_requested", "predictive_prefetch_validated", "predictive_prefetch_correct"):
            info["metrics_protocol"][key] = info[key]
        info["workflow_completion_time_s"] = self._runtime.now_s if self._finished and not self._failed else None
        info["service_success"] = self._finished and not self._failed
        info["failure_reason"] = "access_changed_during_wireless_delivery" if self._failed else None
        info["metrics_protocol"]["service_success"] = info["service_success"]
        info["metrics_protocol"]["failure_reason"] = info["failure_reason"]
        info["handoff_events"] = deepcopy(self._handoff_events)
        return deepcopy(self._last_state), reward, self._finished and not self._failed, truncated, info
