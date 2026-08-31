"""AI-driven VEC 工作流核心环境最小骨架。"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

from src.data.mobility.handoff_builder import HandoffBuilder
from src.data.mobility.replay_provider import ReplayProvider
from src.data.mobility.rsu_mapper import RSUMapper
from src.data.model_catalog.adapter_catalog import (
    AdapterCatalog,
    LEGACY_MODEL_CACHE_PROFILE_ID,
    TYPED_MODEL_CACHE_CONTRACT_VERSION,
    TYPED_MODEL_CACHE_PROFILE_ID,
)
from src.data.workflow.toy_workflow_generator import ToyWorkflowGenerator
from src.envs.core.cache_eviction import EvictionPlan, build_eviction_policy
from src.envs.core.predictor_manager import PredictorManager
from src.envs.specs import (
    CACHE_EVENT_SCHEMA_VERSION,
    CacheEvent,
    ControlAction,
    RSUState,
    RewardBreakdown,
    VehicleState,
    WorkflowGraphState,
)
from src.runtime.formal_exogenous_request_execution import (
    FormalRequestExposureError,
    align_cache_event_to_request,
    validate_formal_request_exposure_trace,
)


PRIMARY_VEHICLE_SELECTION_CHOICES = {"stable_first", "handoff_pressure"}
CACHE_CAPACITY_UNITS = {"adapter_slots", "mb"}
CACHE_CAPACITY_EPSILON = 1.0e-9
TYPED_MAX_DEPENDENCY_BUNDLE_OBJECTS = 2


class VecWorkflowCoreEnv:
    """面向跨 RSU 连续 DAG workflow 的最小语义环境。"""

    def __init__(
        self,
        mobility_provider: ReplayProvider | None = None,
        workflow_state: WorkflowGraphState | None = None,
        adapter_catalog: AdapterCatalog | None = None,
        rsu_states: list[RSUState] | None = None,
        predictor_manager: PredictorManager | None = None,
        max_steps: int = 8,
        handoff_prepare_window: int = 6,
        reward_positive_offset: float = 5.0,
        mobility_source: str = "ngsim",
        cache_capacity_profile: dict[str, Any] | None = None,
        primary_vehicle_selection: str = "stable_first",
        formal_request_exposure_trace: dict[str, Any] | None = None,
    ) -> None:
        self._mobility_provider = mobility_provider or ReplayProvider()
        self._mobility_source = str(mobility_source or "ngsim").strip().lower()
        self._primary_vehicle_selection = self._normalize_primary_vehicle_selection(
            primary_vehicle_selection
        )
        self._lust_workflow_size_scale = 2.0
        self._lust_rsu_compute_scale = 0.5
        self._lust_service_step_divisor = 64.0
        self._workflow_template = self._prepare_workflow_template(
            workflow_state or ToyWorkflowGenerator().generate()
        )
        self._catalog_template = deepcopy(adapter_catalog or self._load_default_catalog())
        self._rsu_template = deepcopy(rsu_states or self._build_default_rsus())
        self._predictor_manager = predictor_manager or PredictorManager()
        self._max_steps = max_steps
        self._handoff_prepare_window = max(1, int(handoff_prepare_window))
        self._reward_positive_offset = max(float(reward_positive_offset), 0.0)
        self._cache_capacity_profile = self._normalize_cache_capacity_profile(cache_capacity_profile)
        self._formal_request_exposure_trace = (
            deepcopy(formal_request_exposure_trace)
            if formal_request_exposure_trace is not None
            else None
        )
        if self._formal_request_exposure_trace is not None:
            validate_formal_request_exposure_trace(self._formal_request_exposure_trace)
        self._eviction_policy = build_eviction_policy(
            self._cache_capacity_profile["eviction_policy"],
            seed=self._cache_capacity_profile.get("eviction_policy_seed"),
            **dict(self._cache_capacity_profile.get("eviction_policy_config", {})),
        )

        self._handoff_builder = HandoffBuilder()
        self._mapper = RSUMapper(deepcopy(self._rsu_template))

        self.workflow_state: WorkflowGraphState = deepcopy(self._workflow_template)
        self.adapter_catalog: AdapterCatalog = deepcopy(self._catalog_template)
        self.rsu_states: list[RSUState] = deepcopy(self._rsu_template)
        self._last_associations: dict[str, str | None] = {}
        self._episode_steps = 0
        self._last_state: dict[str, Any] = {}
        self._prepare_history: list[dict[str, Any]] = []
        self._last_eviction_plan: dict[str, Any] | None = None
        self._primary_vehicle_id: str | None = None
        self._node_service_steps: dict[str, int] = {}
        self._node_remaining_service_steps: dict[str, int] = {}
        self._typed_resident_object_ids: dict[str, list[str]] = {}
        self._typed_workflow_state_ready: dict[str, set[str]] = {}
        self._formal_request_exposure_index = 0

    @property
    def reward_positive_offset(self) -> float:
        """Return the per-step positive reward offset used by this env."""
        return self._reward_positive_offset

    def export_cache_eviction_policy_state(self) -> dict[str, Any]:
        """Return a detached policy snapshot for debugging and artifact audit."""
        return self._eviction_policy.export_state()

    def export_last_eviction_plan(self) -> dict[str, Any] | None:
        """Return the last planned victim selection without exposing live state."""
        return deepcopy(self._last_eviction_plan)

    def export_cache_trace_snapshot(self) -> dict[str, Any]:
        """Return a detached, JSON-safe per-RSU residency snapshot for metrics."""
        capacity_enabled = self._cache_capacity_enabled()
        capacity = self._cache_capacity_value()
        rsus = []
        for rsu in sorted(self.rsu_states, key=lambda item: item.rsu_id):
            residents = []
            if self._typed_mode_enabled():
                for object_id in self._typed_resident_object_ids.get(rsu.rsu_id, []):
                    item = self.adapter_catalog.get_typed_object(object_id)
                    residents.append({
                        "object_id": item.object_id,
                        "object_type": item.object_type,
                        "adapter_id": item.adapter_id,
                        "base_model_id": item.base_model_id,
                        "required_base_model_id": item.required_base_model_id,
                        "size_mb": float(item.resident_size_mb),
                        "evictability": item.evictability,
                    })
            else:
                for adapter_id in rsu.cached_adapter_ids:
                    resolution = self.adapter_catalog.resolve_adapter_resident_size_mb(adapter_id)
                    cache_object = next(
                        (item for item in self.adapter_catalog.cache_objects if item.adapter_id == adapter_id),
                        None,
                    )
                    residents.append({
                        "object_id": cache_object.object_id if cache_object else f"adapter:{adapter_id}",
                        "adapter_id": adapter_id,
                        "size_mb": float(resolution.size_mb),
                    })
            rsus.append({
                "rsu_id": rsu.rsu_id,
                "residents": residents,
                "capacity_enabled": capacity_enabled,
                "capacity_unit": self._cache_capacity_profile.get("unit", "adapter_slots"),
                "capacity": capacity,
                **(
                    {
                        "model_cache_profile_id": self.adapter_catalog.model_cache_profile_id,
                        "typed_model_cache_contract_version": TYPED_MODEL_CACHE_CONTRACT_VERSION,
                    }
                    if self._typed_mode_enabled()
                    else {}
                ),
            })
        return {"snapshot_step_index": int(self._episode_steps), "rsus": rsus}

    @property
    def formal_request_exposure_trace(self) -> dict[str, Any] | None:
        return deepcopy(self._formal_request_exposure_trace)

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """重置环境并返回语义状态字典。"""
        self._episode_steps = 0
        self.workflow_state = deepcopy(self._workflow_template)
        self.adapter_catalog = deepcopy(self._catalog_template)
        self.rsu_states = deepcopy(self._rsu_template)
        self._predictor_manager.reset()
        self._prepare_history = []
        self._last_eviction_plan = None
        self._primary_vehicle_id = None
        self._node_service_steps = self._build_node_service_step_plan(self.workflow_state)
        self._node_remaining_service_steps = dict(self._node_service_steps)
        for rsu in self.rsu_states:
            rsu.cached_adapter_ids = self.adapter_catalog.get_initial_cached_adapters(rsu.rsu_id)
            rsu.active_vehicle_ids = []
        self._typed_resident_object_ids = {
            rsu.rsu_id: self.adapter_catalog.get_initial_typed_residents(rsu.rsu_id)
            for rsu in self.rsu_states
        }
        self._typed_workflow_state_ready = {rsu.rsu_id: set() for rsu in self.rsu_states}
        if self._typed_mode_enabled():
            self._sync_legacy_adapter_views_from_typed()
        self._initialize_cache_capacity_metadata()

        self._mapper.update_rsus(self.rsu_states)
        vehicles = self._mobility_provider.reset()
        self._initialize_primary_vehicle_id(vehicles)
        self._formal_request_exposure_index = 0
        if self._formal_request_exposure_trace is not None:
            self._initialize_formal_request_exposure()
        associations = self._mapper.associate(vehicles)
        self._apply_associations(vehicles, associations)
        predictions = self._build_predictions(vehicles, associations)
        handoff_events: list[dict[str, Any]] = []
        self._last_associations = associations
        self._last_state = self._build_state_dict(
            vehicles=vehicles,
            associations=associations,
            predictions=predictions,
            handoff_events=handoff_events,
        )
        info = self._build_info(
            current_node=self.workflow_state.current_node(),
            primary_vehicle=self._select_primary_vehicle(vehicles),
            handoff_events=handoff_events,
            cache_hit=False,
            offload_target_rsu_id=None,
            stall_occurred=False,
            reward=RewardBreakdown(
                total=0.0,
                positive_offset=0.0,
                service_reward=0.0,
                delay_penalty=0.0,
                cache_miss_penalty=0.0,
                migration_cost=0.0,
                continuity_bonus=0.0,
                mechanism_exploration_bonus=0.0,
                constraint_penalty=0.0,
            ),
            control=ControlAction(),
            cache_result=self._default_cache_result(),
            handoff_count=0,
            raw_handoff_count=0,
            gap_transfer_count=0,
            pre_action_associated_rsu_id=None,
            pre_action_prediction_snapshot={},
            realized_prepare=self._default_prepare_realization(),
        )
        return deepcopy(self._last_state), info

    def step(
        self,
        control: ControlAction,
    ) -> tuple[dict[str, Any], RewardBreakdown, bool, bool, dict[str, Any]]:
        """推进环境一个时间步。"""
        self._episode_steps += 1
        self._prune_prepare_history()

        pre_action_vehicle = self._extract_primary_vehicle_from_state(self._last_state)
        pre_action_vehicle_id = pre_action_vehicle.get("vehicle_id") or self._primary_vehicle_id
        pre_action_associated_rsu_id = pre_action_vehicle.get("associated_rsu_id")
        pre_action_prediction_snapshot = self._extract_prediction_snapshot(
            state=self._last_state,
            vehicle_id=pre_action_vehicle_id,
        )

        vehicles = self._mobility_provider.step()
        self._ensure_primary_vehicle_id(vehicles)
        associations = self._mapper.associate(vehicles)
        handoff_events = [
            event.to_dict()
            for event in self._handoff_builder.build_events(
                previous_associations=self._last_associations,
                current_associations=associations,
                time_index=self._mobility_provider.get_time(),
            )
        ]
        self._apply_associations(vehicles, associations)

        primary_vehicle = self._select_primary_vehicle(vehicles)
        current_node = self.workflow_state.current_node()
        if self._formal_request_exposure_trace is not None:
            self._validate_formal_request_runtime_identity(
                current_node=current_node,
                primary_vehicle=primary_vehicle,
                request_rsu_id=pre_action_associated_rsu_id,
                current_service_rsu_id=(
                    primary_vehicle.associated_rsu_id if primary_vehicle else None
                ),
            )
        current_required_adapter = current_node.required_adapter if current_node else None
        tracked_vehicle_id = primary_vehicle.vehicle_id if primary_vehicle else pre_action_vehicle_id
        offload_target_rsu_id = self._resolve_target_rsu_id(primary_vehicle, control)
        if self._typed_mode_enabled() and current_node is not None:
            pre_execution_cache_hit = self._typed_service_readiness(
                current_node=current_node,
                primary_vehicle=primary_vehicle,
                offload_mode=str(control.offload_action.get("mode", "rsu")),
                service_rsu_id=offload_target_rsu_id,
                state_required=False,
                state_ready=True,
            )["joint_base_adapter_hit"]
        else:
            pre_execution_cache_hit = self._check_rsu_has_required_adapter(
                rsu_id=offload_target_rsu_id,
                required_adapter=current_required_adapter,
            )
        cache_result = self._apply_cache_action(
            control=control,
            primary_vehicle=primary_vehicle,
            current_node_id=current_node.node_id if current_node else None,
            required_adapter=current_required_adapter,
        )
        if (
            self._typed_mode_enabled()
            and current_required_adapter is not None
            and cache_result.get("dependency_bundle") is None
        ):
            request_rsu_id = offload_target_rsu_id
            request_residents = list(
                self._typed_resident_object_ids.get(str(request_rsu_id), [])
            )
            request_placement = self.adapter_catalog.resolve_typed_placement_plan(
                adapter_id=current_required_adapter,
                resident_object_ids=request_residents,
            )
            cache_result["dependency_bundle"] = request_placement.to_dict()
            cache_result["requested_typed_objects"] = [
                self._typed_object_row(object_id)
                for object_id in request_placement.ordered_object_ids
            ]
        handoff_count = sum(
            1
            for event in handoff_events
            if event["vehicle_id"] == tracked_vehicle_id and event["event_type"] == "handoff"
        )
        migration_mode = control.migration_action.get("mode", "keep")
        prepare_action_context = self._build_prepare_action_context(
            control=control,
            vehicle_id=tracked_vehicle_id,
            required_adapter=current_required_adapter,
            prediction_snapshot=pre_action_prediction_snapshot,
        )
        gap_transfer_count = int(
            self._is_gap_transfer_entry(
                pre_action_associated_rsu_id=pre_action_associated_rsu_id,
                post_action_associated_rsu_id=primary_vehicle.associated_rsu_id if primary_vehicle else None,
                prediction_snapshot=pre_action_prediction_snapshot,
                prepare_action_context=prepare_action_context,
            )
        )
        mobility_transfer_count = max(handoff_count, gap_transfer_count)
        realized_prepare = self._consume_realized_prepare(
            vehicle_id=tracked_vehicle_id,
            actual_target_rsu_id=primary_vehicle.associated_rsu_id if primary_vehicle else None,
            required_adapter=current_required_adapter,
            handoff_count=mobility_transfer_count,
            current_prepare_action=prepare_action_context,
        )
        offload_mode = str(control.offload_action.get("mode", "rsu"))
        typed_readiness: dict[str, Any] | None = None
        if self._typed_mode_enabled() and current_node is not None:
            continuity_identity = f"{tracked_vehicle_id}/{self.workflow_state.workflow_id}"
            state_required = mobility_transfer_count > 0
            migration_realized_now = bool(
                realized_prepare.get("realized", False)
                or (mobility_transfer_count > 0 and migration_mode == "migrate")
            )
            if migration_realized_now and offload_target_rsu_id:
                self._typed_workflow_state_ready.setdefault(
                    offload_target_rsu_id, set()
                ).add(continuity_identity)
            state_ready = bool(
                not state_required
                or continuity_identity
                in self._typed_workflow_state_ready.get(str(offload_target_rsu_id), set())
            )
            typed_readiness = self._typed_service_readiness(
                current_node=current_node,
                primary_vehicle=primary_vehicle,
                offload_mode=offload_mode,
                service_rsu_id=offload_target_rsu_id,
                state_required=state_required,
                state_ready=state_ready,
            )
            cache_result["service_readiness"] = typed_readiness

        cache_hit = False
        base_model_ok = False
        service_reward = 0.15
        delay_penalty = 0.15
        cache_miss_penalty = 0.0
        migration_cost = 0.0
        continuity_bonus = 0.0
        mechanism_exploration_bonus = 0.0
        constraint_penalty = 0.0
        if current_node is None:
            reward = RewardBreakdown(
                total=0.0,
                positive_offset=0.0,
                service_reward=0.0,
                delay_penalty=0.0,
                cache_miss_penalty=0.0,
                migration_cost=0.0,
                continuity_bonus=0.0,
                mechanism_exploration_bonus=0.0,
                constraint_penalty=0.0,
            )
            predictions = self._build_predictions(vehicles, associations)
            self._last_associations = associations
            self._last_state = self._build_state_dict(
                vehicles=vehicles,
                associations=associations,
                predictions=predictions,
                handoff_events=handoff_events,
            )
            self._register_prepare_action(prepare_action_context, realized_prepare)
            info = self._build_info(
                current_node=None,
                primary_vehicle=primary_vehicle,
                handoff_events=handoff_events,
                cache_hit=False,
                offload_target_rsu_id=offload_target_rsu_id,
                stall_occurred=False,
                reward=reward,
                control=control,
                cache_result=cache_result,
                handoff_count=0,
                raw_handoff_count=0,
                gap_transfer_count=0,
                pre_action_associated_rsu_id=pre_action_associated_rsu_id,
                pre_action_prediction_snapshot=pre_action_prediction_snapshot,
                realized_prepare=realized_prepare,
            )
            return deepcopy(self._last_state), reward, True, False, info

        if primary_vehicle is None:
            constraint_penalty += 1.0
        elif typed_readiness is not None:
            service_reward += 0.15
            base_model_ok = bool(typed_readiness["base_ready"])
            if not base_model_ok:
                constraint_penalty += 1.0
            else:
                service_reward += 0.2
        else:
            service_reward += 0.15
            base_model_ok = primary_vehicle.base_model_id == current_node.required_base_model
            if not base_model_ok:
                constraint_penalty += 1.0
            else:
                service_reward += 0.2

        if offload_target_rsu_id is None and (
            typed_readiness is None or offload_mode == "rsu"
        ):
            constraint_penalty += 0.7
        elif typed_readiness is not None:
            service_reward += 0.1
            cache_hit = bool(typed_readiness["full_service_ready"])
            if cache_hit:
                service_reward += 0.45
        else:
            service_reward += 0.1
            target_rsu = self._get_rsu_map().get(offload_target_rsu_id)
            cache_hit = bool(
                target_rsu
                and current_node.required_adapter in target_rsu.cached_adapter_ids
            )
            if cache_hit:
                service_reward += 0.45

        if offload_mode == "vehicle":
            delay_penalty += 0.65
        elif offload_mode == "rsu":
            delay_penalty += 0.75
        else:
            delay_penalty += 0.7

        warm_ready = bool(
            cache_hit
            or pre_execution_cache_hit
            or cache_result.get("was_cached_before", False)
        )
        prepared_handoff_realized = bool(realized_prepare.get("realized", False))
        predicted_handoff_signal = self._has_predicted_handoff_signal(
            prediction_snapshot=pre_action_prediction_snapshot,
            current_rsu_id=pre_action_associated_rsu_id,
        )
        mechanism_exploration_action = self._is_mechanism_exploration_action(control)
        if predicted_handoff_signal and mechanism_exploration_action:
            mechanism_exploration_bonus = 1.0
        if mobility_transfer_count > 0:
            delay_penalty += 0.25 * mobility_transfer_count
            if migration_mode == "migrate":
                migration_cost = 0.35 * mobility_transfer_count
                continuity_bonus = 1.45 if warm_ready else 0.25
            elif migration_mode == "prepare" or prepared_handoff_realized:
                migration_cost = 0.18 * mobility_transfer_count
                continuity_bonus = 8.0 if warm_ready else 0.35
            else:
                migration_cost = 1.0 * mobility_transfer_count
                continuity_bonus = 0.1 if cache_hit else 0.0
        else:
            continuity_bonus = 0.35 if cache_hit else 0.05

        if not cache_hit:
            cache_miss_penalty = 1.2

        stall_occurred = not (primary_vehicle and base_model_ok and cache_hit and offload_target_rsu_id)
        if stall_occurred:
            delay_penalty += 0.8
        else:
            if self._formal_request_exposure_trace is not None:
                service_completed = True
            else:
                service_completed = self._advance_current_node_service(current_node)
            if service_completed:
                service_reward += 1.15

        if cache_result["added_new_adapter"] and cache_hit:
            continuity_bonus += 0.15
        if mobility_transfer_count > 0 and warm_ready:
            service_reward += 2.0
        if prepared_handoff_realized and warm_ready:
            service_reward += 2.0

        total_reward = (
            self._reward_positive_offset
            + service_reward
            + continuity_bonus
            + mechanism_exploration_bonus
            - delay_penalty
            - cache_miss_penalty
            - migration_cost
            - constraint_penalty
        )
        reward = RewardBreakdown(
            total=total_reward,
            positive_offset=self._reward_positive_offset,
            service_reward=service_reward,
            delay_penalty=delay_penalty,
            cache_miss_penalty=cache_miss_penalty,
            migration_cost=migration_cost,
            continuity_bonus=continuity_bonus,
            mechanism_exploration_bonus=mechanism_exploration_bonus,
            constraint_penalty=constraint_penalty,
        )

        if self._formal_request_exposure_trace is not None:
            info = self._build_info(
                current_node=current_node,
                primary_vehicle=primary_vehicle,
                handoff_events=handoff_events,
                cache_hit=cache_hit,
                offload_target_rsu_id=offload_target_rsu_id,
                stall_occurred=stall_occurred,
                reward=reward,
                control=control,
                cache_result=cache_result,
                handoff_count=mobility_transfer_count,
                raw_handoff_count=handoff_count,
                gap_transfer_count=gap_transfer_count,
                pre_action_associated_rsu_id=pre_action_associated_rsu_id,
                pre_action_prediction_snapshot=pre_action_prediction_snapshot,
                realized_prepare=realized_prepare,
                pre_execution_cache_hit=pre_execution_cache_hit,
            )
            request = self._current_formal_request()
            info["cache_event"]["requested_typed_objects"] = deepcopy(
                request["requested_typed_objects"]
            )
            info["cache_event"]["dependency_bundle"] = deepcopy(
                request["dependency_bundle"]
            )
            aligned_event = align_cache_event_to_request(
                info["cache_event"],
                request,
                trace_fingerprint=str(
                    self._formal_request_exposure_trace["request_exposure_fingerprint"]
                ),
            )
            info["cache_event"] = aligned_event
            info["metrics_protocol"]["cache_event"] = deepcopy(aligned_event)
            info["metrics_protocol"].update(
                formal_request_id=request["request_id"],
                formal_request_order=request["request_order"],
                request_exposure_fingerprint=self._formal_request_exposure_trace[
                    "request_exposure_fingerprint"
                ],
                request_alignment_status="matched_exactly_once",
            )
            exposure_exhausted = self._advance_formal_request_exposure(
                service_success=not stall_occurred
            )
            predictions = self._build_predictions(vehicles, associations)
            self._last_associations = associations
            self._last_state = self._build_state_dict(
                vehicles=vehicles,
                associations=associations,
                predictions=predictions,
                handoff_events=handoff_events,
            )
            self._register_prepare_action(prepare_action_context, realized_prepare)
            truncated = bool(
                self._episode_steps >= self._max_steps and not exposure_exhausted
            )
            return deepcopy(self._last_state), reward, exposure_exhausted, truncated, info

        predictions = self._build_predictions(vehicles, associations)
        self._last_associations = associations
        self._last_state = self._build_state_dict(
            vehicles=vehicles,
            associations=associations,
            predictions=predictions,
            handoff_events=handoff_events,
        )
        self._register_prepare_action(prepare_action_context, realized_prepare)

        terminated = self.workflow_state.is_completed
        truncated = self._episode_steps >= self._max_steps and not terminated
        info = self._build_info(
            current_node=current_node,
            primary_vehicle=primary_vehicle,
            handoff_events=handoff_events,
            cache_hit=cache_hit,
            offload_target_rsu_id=offload_target_rsu_id,
            stall_occurred=stall_occurred,
            reward=reward,
            control=control,
            cache_result=cache_result,
            handoff_count=mobility_transfer_count,
            raw_handoff_count=handoff_count,
            gap_transfer_count=gap_transfer_count,
            pre_action_associated_rsu_id=pre_action_associated_rsu_id,
            pre_action_prediction_snapshot=pre_action_prediction_snapshot,
            realized_prepare=realized_prepare,
            pre_execution_cache_hit=pre_execution_cache_hit,
        )
        return deepcopy(self._last_state), reward, terminated, truncated, info

    def _initialize_formal_request_exposure(self) -> None:
        trace = self._formal_request_exposure_trace
        if trace is None:
            raise FormalRequestExposureError("formal request exposure is missing")
        validate_formal_request_exposure_trace(trace)
        unit = trace.get("evaluation_unit") or {}
        if str(unit.get("workflow_id")) != str(self.workflow_state.workflow_id):
            raise FormalRequestExposureError("workflow identity drift in request exposure")
        semantics = trace.get("execution_semantics") or {}
        if semantics.get("primary_vehicle_selection") != self._primary_vehicle_selection:
            raise FormalRequestExposureError("primary vehicle selection drift")
        if str(trace["requests"][0]["vehicle_id"]) != str(self._primary_vehicle_id):
            raise FormalRequestExposureError("primary vehicle identity drift")
        catalog_fingerprints = {
            str(request.get("catalog_fingerprint")) for request in trace["requests"]
        }
        if catalog_fingerprints != {self.adapter_catalog.canonical_fingerprint()}:
            raise FormalRequestExposureError("catalog identity drift in request exposure")
        self.workflow_state.completed_node_ids = []
        self.workflow_state.is_completed = False
        self.workflow_state.current_node_id = str(trace["requests"][0]["node_id"])

    def _current_formal_request(self) -> dict[str, Any]:
        if self._formal_request_exposure_trace is None:
            raise FormalRequestExposureError("formal request exposure is not enabled")
        requests = self._formal_request_exposure_trace["requests"]
        if self._formal_request_exposure_index >= len(requests):
            raise FormalRequestExposureError("extra environment request after exposure exhaustion")
        return requests[self._formal_request_exposure_index]

    def _validate_formal_request_runtime_identity(
        self,
        *,
        current_node: Any,
        primary_vehicle: VehicleState | None,
        request_rsu_id: str | None,
        current_service_rsu_id: str | None,
    ) -> None:
        request = self._current_formal_request()
        checks = {
            "step_index": (self._episode_steps, request["step_index"]),
            "time_index": (self._mobility_provider.get_time(), request["time_index"]),
            "vehicle_id": (
                primary_vehicle.vehicle_id if primary_vehicle else self._primary_vehicle_id,
                request["vehicle_id"],
            ),
            "workflow_id": (self.workflow_state.workflow_id, request["workflow_id"]),
            "node_id": (current_node.node_id if current_node else None, request["node_id"]),
            "required_base_model": (
                current_node.required_base_model if current_node else None,
                request["required_base_model"],
            ),
            "adapter_id": (
                current_node.required_adapter if current_node else None,
                request["adapter_id"],
            ),
            "request_rsu_id": (request_rsu_id, request["request_rsu_id"]),
            "current_service_rsu_id": (
                current_service_rsu_id,
                request["current_service_rsu_id"],
            ),
        }
        drift = [name for name, values in checks.items() if values[0] != values[1]]
        if drift:
            raise FormalRequestExposureError(
                "formal request runtime identity drift: " + ", ".join(drift)
            )

    def _advance_formal_request_exposure(self, *, service_success: bool) -> bool:
        request = self._current_formal_request()
        if service_success and request["node_id"] not in self.workflow_state.completed_node_ids:
            self.workflow_state.completed_node_ids.append(str(request["node_id"]))
        self._formal_request_exposure_index += 1
        assert self._formal_request_exposure_trace is not None
        requests = self._formal_request_exposure_trace["requests"]
        if self._formal_request_exposure_index >= len(requests):
            self.workflow_state.current_node_id = None
            self.workflow_state.is_completed = len(self.workflow_state.completed_node_ids) == len(
                requests
            )
            return True
        self.workflow_state.current_node_id = str(
            requests[self._formal_request_exposure_index]["node_id"]
        )
        self.workflow_state.is_completed = False
        return False

    def _build_state_dict(
        self,
        vehicles: list[VehicleState],
        associations: dict[str, str | None],
        predictions: dict[str, Any],
        handoff_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        current_node = self.workflow_state.current_node()
        ordered_vehicles, primary_vehicle_present, primary_vehicle_reordered_to_front = self._order_vehicles_for_primary(
            vehicles
        )
        state = {
            "time_index": self._mobility_provider.get_time(),
            "vehicles": [vehicle.to_dict() for vehicle in ordered_vehicles],
            "primary_vehicle_id": self._primary_vehicle_id,
            "primary_vehicle_selection": self._primary_vehicle_selection,
            "primary_vehicle_handoff_pressure_enabled": self._uses_handoff_pressure_primary_selection(),
            "primary_vehicle_present": bool(primary_vehicle_present),
            "primary_vehicle_reordered_to_front": bool(primary_vehicle_reordered_to_front),
            "rsus": [rsu.to_dict() for rsu in self.rsu_states],
            "associations": dict(associations),
            "workflow": self.workflow_state.to_dict(),
            "current_workflow_node": current_node.to_dict() if current_node else None,
            "current_node_service_steps_required": self._get_current_node_service_steps_required(current_node),
            "current_node_service_steps_remaining": self._get_current_node_service_steps_remaining(current_node),
            "predictions": deepcopy(predictions),
            "handoff_events": handoff_events,
        }
        if self._formal_request_exposure_trace is not None:
            state["formal_request_exposure"] = {
                "contract_version": "1.0.0",
                "request_exposure_fingerprint": self._formal_request_exposure_trace[
                    "request_exposure_fingerprint"
                ],
                "current_request_index": int(self._formal_request_exposure_index),
                "request_count": len(self._formal_request_exposure_trace["requests"]),
                "oracle_future_fields_actor_visible": False,
                "outcome_fields_present": False,
            }
        return state

    def _order_vehicles_for_primary(
        self,
        vehicles: list[VehicleState],
    ) -> tuple[list[VehicleState], bool, bool]:
        if not vehicles:
            return [], False, False
        ordered_vehicles = list(vehicles)
        if not self._primary_vehicle_id:
            return ordered_vehicles, False, False
        for index, vehicle in enumerate(ordered_vehicles):
            if vehicle.vehicle_id != self._primary_vehicle_id:
                continue
            if index == 0:
                return ordered_vehicles, True, False
            primary_vehicle = ordered_vehicles[index]
            remaining_vehicles = ordered_vehicles[:index] + ordered_vehicles[index + 1 :]
            return [primary_vehicle, *remaining_vehicles], True, True
        return ordered_vehicles, False, False

    def _build_predictions(
        self,
        vehicles: list[VehicleState],
        associations: dict[str, str | None],
    ) -> dict[str, Any]:
        return self._predictor_manager.build_predictions(
            time_index=self._mobility_provider.get_time(),
            vehicles=vehicles,
            rsu_states=self.rsu_states,
            workflow_state=self.workflow_state,
            adapter_catalog=self.adapter_catalog,
            current_associations=associations,
        )

    def _apply_associations(
        self,
        vehicles: list[VehicleState],
        associations: dict[str, str | None],
    ) -> None:
        rsu_map = self._get_rsu_map()
        for rsu in self.rsu_states:
            rsu.active_vehicle_ids = []
        for vehicle in vehicles:
            vehicle.associated_rsu_id = associations.get(vehicle.vehicle_id)
            if vehicle.associated_rsu_id and vehicle.associated_rsu_id in rsu_map:
                rsu_map[vehicle.associated_rsu_id].active_vehicle_ids.append(vehicle.vehicle_id)

    def _initialize_primary_vehicle_id(self, vehicles: list[VehicleState]) -> None:
        if not vehicles:
            self._primary_vehicle_id = None
            return
        if self._uses_handoff_pressure_primary_selection():
            candidate_vehicle_ids = {vehicle.vehicle_id for vehicle in vehicles}
            high_pressure_vehicle_id = self._select_high_pressure_vehicle_id(
                candidate_vehicle_ids=candidate_vehicle_ids
            )
            if high_pressure_vehicle_id is not None:
                self._primary_vehicle_id = high_pressure_vehicle_id
                return
        self._primary_vehicle_id = sorted(vehicle.vehicle_id for vehicle in vehicles)[0]

    def _ensure_primary_vehicle_id(self, vehicles: list[VehicleState]) -> None:
        if self._primary_vehicle_id and any(vehicle.vehicle_id == self._primary_vehicle_id for vehicle in vehicles):
            return
        if self._mobility_source == "lust" and self._primary_vehicle_id:
            return
        self._initialize_primary_vehicle_id(vehicles)

    def _select_primary_vehicle(
        self,
        vehicles: list[VehicleState],
    ) -> VehicleState | None:
        if not vehicles:
            self._primary_vehicle_id = None
            return None
        self._ensure_primary_vehicle_id(vehicles)
        for vehicle in vehicles:
            if vehicle.vehicle_id == self._primary_vehicle_id:
                return vehicle
        if self._mobility_source == "lust" and self._primary_vehicle_id:
            return None
        sorted_vehicles = sorted(vehicles, key=lambda item: item.vehicle_id)
        self._primary_vehicle_id = sorted_vehicles[0].vehicle_id
        return sorted_vehicles[0]

    def _normalize_primary_vehicle_selection(self, value: str) -> str:
        selection = str(value or "stable_first").strip().lower()
        if selection not in PRIMARY_VEHICLE_SELECTION_CHOICES:
            choices = ", ".join(sorted(PRIMARY_VEHICLE_SELECTION_CHOICES))
            raise ValueError(f"unsupported primary_vehicle_selection={value!r}; choices: {choices}")
        return selection

    def _uses_handoff_pressure_primary_selection(self) -> bool:
        return self._primary_vehicle_selection == "handoff_pressure" or self._mobility_source == "lust"

    def _resolve_target_rsu_id(
        self,
        primary_vehicle: VehicleState | None,
        control: ControlAction,
    ) -> str | None:
        return control.offload_action.get("target_rsu_id") or (
            primary_vehicle.associated_rsu_id if primary_vehicle else None
        )

    def _normalize_cache_capacity_profile(self, profile: dict[str, Any] | None) -> dict[str, Any]:
        merged = {
            "model_cache_profile_id": LEGACY_MODEL_CACHE_PROFILE_ID,
            "enabled": False,
            "unit": "adapter_slots",
            "rsu_adapter_slots": 0,
            "capacity_mb": None,
            "count_base_model_separately": False,
            "eviction_policy": "lru",
            "eviction_policy_seed": None,
            "eviction_policy_config": {},
            "telemetry_enabled": True,
        }
        if profile:
            merged.update(dict(profile))
        merged["enabled"] = bool(merged.get("enabled", False))
        merged["model_cache_profile_id"] = str(
            merged.get("model_cache_profile_id") or LEGACY_MODEL_CACHE_PROFILE_ID
        )
        catalog_profile = str(
            getattr(self._catalog_template, "model_cache_profile_id", LEGACY_MODEL_CACHE_PROFILE_ID)
        )
        if merged["model_cache_profile_id"] != catalog_profile:
            raise ValueError(
                "cache capacity model_cache_profile_id must match catalog profile"
            )
        merged["unit"] = str(merged.get("unit") or "adapter_slots").strip().lower()
        if merged["unit"] not in CACHE_CAPACITY_UNITS:
            raise ValueError(f"unsupported cache capacity unit: {merged['unit']}")
        raw_slots = merged.get("rsu_adapter_slots", 0)
        merged["rsu_adapter_slots"] = int(raw_slots or 0)
        if merged["rsu_adapter_slots"] < 0:
            raise ValueError("rsu_adapter_slots must be non-negative")
        if merged["enabled"] and merged["unit"] == "mb":
            if merged.get("capacity_mb") is None:
                raise ValueError("capacity_mb is required when cache capacity unit=mb")
            capacity_mb = float(merged["capacity_mb"])
            if not math.isfinite(capacity_mb) or capacity_mb <= 0.0:
                raise ValueError("capacity_mb must be a finite positive number")
            merged["capacity_mb"] = capacity_mb
        merged["count_base_model_separately"] = bool(merged.get("count_base_model_separately", False))
        merged["eviction_policy"] = str(merged.get("eviction_policy") or "lru").lower()
        if merged.get("eviction_policy_seed") is not None:
            merged["eviction_policy_seed"] = int(merged["eviction_policy_seed"])
        if not isinstance(merged.get("eviction_policy_config"), dict):
            raise ValueError("eviction_policy_config must be a mapping")
        merged["eviction_policy_config"] = dict(merged["eviction_policy_config"])
        merged["telemetry_enabled"] = bool(merged.get("telemetry_enabled", True))
        if merged["model_cache_profile_id"] == TYPED_MODEL_CACHE_PROFILE_ID:
            if not merged["enabled"] or merged["unit"] != "mb":
                raise ValueError("typed model cache requires enabled MB capacity")
            if merged.get("count_base_model_separately") is False:
                merged["count_base_model_separately"] = True
        return merged

    def _typed_mode_enabled(self) -> bool:
        return (
            self.adapter_catalog.model_cache_profile_id == TYPED_MODEL_CACHE_PROFILE_ID
            and self._cache_capacity_profile.get("model_cache_profile_id")
            == TYPED_MODEL_CACHE_PROFILE_ID
        )

    def _cache_capacity_enabled(self) -> bool:
        if not self._cache_capacity_profile.get("enabled", False):
            return False
        if self._cache_capacity_profile.get("unit") == "mb":
            return True
        return int(self._cache_capacity_profile.get("rsu_adapter_slots", 0) or 0) > 0

    def _cache_capacity_value(self) -> float | int | None:
        if not self._cache_capacity_enabled():
            return None
        if self._cache_capacity_profile["unit"] == "mb":
            return float(self._cache_capacity_profile["capacity_mb"])
        return int(self._cache_capacity_profile["rsu_adapter_slots"])

    def _adapter_resident_size_mb(self, adapter_id: str) -> float:
        return self.adapter_catalog.resolve_adapter_resident_size_mb(adapter_id).size_mb

    def _cache_used_value(self, rsu: RSUState) -> float | int:
        if self._typed_mode_enabled():
            return float(
                sum(
                    self.adapter_catalog.get_typed_object(object_id).resident_size_mb
                    for object_id in self._typed_resident_object_ids.get(rsu.rsu_id, [])
                    if self.adapter_catalog.get_typed_object(object_id).counts_toward_capacity
                )
            )
        unique_adapters = list(dict.fromkeys(rsu.cached_adapter_ids))
        if self._cache_capacity_profile["unit"] == "mb":
            return float(sum(self._adapter_resident_size_mb(item) for item in unique_adapters))
        return len(unique_adapters)

    def _initialize_cache_capacity_metadata(self) -> None:
        self._eviction_policy.reset()
        for rsu in self.rsu_states:
            initial_residents = (
                list(self._typed_resident_object_ids.get(rsu.rsu_id, []))
                if self._typed_mode_enabled()
                else list(rsu.cached_adapter_ids)
            )
            self._eviction_policy.reset(
                rsu_id=rsu.rsu_id,
                initial_resident_ids=initial_residents,
                current_step=self._episode_steps,
            )
            if self._cache_capacity_enabled():
                if self._typed_mode_enabled():
                    self._validate_typed_resident_invariants(rsu.rsu_id)
                    if float(self._cache_used_value(rsu)) > float(self._cache_capacity_value() or 0):
                        raise ValueError(
                            "typed initial cache must fit capacity without policy-specific trimming"
                        )
                else:
                    self._enforce_initial_cache_capacity(rsu)

    def _sync_legacy_adapter_views_from_typed(self) -> None:
        for rsu in self.rsu_states:
            adapter_ids = []
            for object_id in self._typed_resident_object_ids.get(rsu.rsu_id, []):
                item = self.adapter_catalog.get_typed_object(object_id)
                if item.object_type == "adapter" and item.adapter_id:
                    adapter_ids.append(item.adapter_id)
            rsu.cached_adapter_ids = adapter_ids

    def _validate_typed_resident_invariants(self, rsu_id: str) -> None:
        residents = self._typed_resident_object_ids.get(rsu_id, [])
        if len(residents) != len(set(residents)):
            raise RuntimeError(f"duplicate typed resident at {rsu_id}")
        resident_set = set(residents)
        for object_id in residents:
            item = self.adapter_catalog.get_typed_object(object_id)
            if not item.counts_toward_capacity or item.object_type == "kv_prefix":
                raise RuntimeError(f"invalid typed resident at {rsu_id}: {object_id}")
            if not set(item.dependency_ids).issubset(resident_set):
                raise RuntimeError(f"orphan typed resident at {rsu_id}: {object_id}")

    def _typed_used_mb_by_type(self, rsu_id: str) -> dict[str, float]:
        totals: dict[str, float] = {}
        for object_id in self._typed_resident_object_ids.get(rsu_id, []):
            item = self.adapter_catalog.get_typed_object(object_id)
            totals[item.object_type] = totals.get(item.object_type, 0.0) + float(
                item.resident_size_mb
            )
        return {key: round(value, 6) for key, value in sorted(totals.items())}

    def _enforce_initial_cache_capacity(self, rsu: RSUState) -> None:
        capacity = self._cache_capacity_value()
        if capacity is None:
            return
        used = float(self._cache_used_value(rsu))
        required = max(used - float(capacity), 0.0)
        if required <= CACHE_CAPACITY_EPSILON:
            return
        plan = self._eviction_policy.plan_victims(
            rsu_id=rsu.rsu_id,
            resident_ids=list(rsu.cached_adapter_ids),
            resident_sizes=self._resident_sizes_for_policy(rsu),
            required_free_capacity=required,
            protected_object_id=None,
            capacity_unit=self._cache_capacity_profile["unit"],
            current_step=self._episode_steps,
        )
        self._validate_eviction_plan(
            plan=plan,
            rsu=rsu,
            required_free_capacity=required,
            protected_object_id=None,
        )
        if not plan.sufficient:
            raise RuntimeError("initial cache cannot be trimmed to configured capacity")
        self._last_eviction_plan = plan.to_dict()
        for evicted_adapter_id in plan.ordered_victim_ids:
            if evicted_adapter_id not in rsu.cached_adapter_ids:
                raise RuntimeError("eviction policy selected a non-resident initial-cache victim")
            rsu.cached_adapter_ids.remove(evicted_adapter_id)
            self._eviction_policy.on_eviction(
                rsu_id=rsu.rsu_id,
                object_id=evicted_adapter_id,
                current_step=self._episode_steps,
            )
            self._remove_catalog_cached_adapter(rsu.rsu_id, evicted_adapter_id)

    def _touch_cached_adapter(self, rsu_id: str | None, adapter_id: str | None) -> None:
        if not rsu_id or not adapter_id:
            return
        self._eviction_policy.on_hit(
            rsu_id=rsu_id,
            object_id=adapter_id,
            current_step=self._episode_steps,
        )

    def _resident_sizes_for_policy(self, rsu: RSUState) -> dict[str, float]:
        if self._typed_mode_enabled():
            return {
                object_id: float(
                    self.adapter_catalog.get_typed_object(object_id).resident_size_mb
                )
                for object_id in self._typed_resident_object_ids.get(rsu.rsu_id, [])
            }
        if self._cache_capacity_profile["unit"] == "mb":
            return {
                adapter_id: self._adapter_resident_size_mb(adapter_id)
                for adapter_id in dict.fromkeys(rsu.cached_adapter_ids)
            }
        return {adapter_id: 1.0 for adapter_id in dict.fromkeys(rsu.cached_adapter_ids)}

    def _validate_eviction_plan(
        self,
        *,
        plan: EvictionPlan,
        rsu: RSUState,
        required_free_capacity: float,
        protected_object_id: str | None,
    ) -> None:
        residents = list(
            dict.fromkeys(
                self._typed_resident_object_ids.get(rsu.rsu_id, [])
                if self._typed_mode_enabled()
                else rsu.cached_adapter_ids
            )
        )
        victims = list(plan.ordered_victim_ids)
        if plan.rsu_id != rsu.rsu_id:
            raise RuntimeError("eviction plan RSU does not match admission target")
        if (
            plan.policy_name != self._eviction_policy.policy_name
            or plan.policy_version != self._eviction_policy.policy_version
        ):
            raise RuntimeError("eviction plan identity does not match configured policy")
        if plan.capacity_unit != self._cache_capacity_profile["unit"]:
            raise RuntimeError("eviction plan capacity unit does not match environment")
        if (
            abs(float(plan.required_free_capacity) - float(required_free_capacity))
            > CACHE_CAPACITY_EPSILON
        ):
            raise RuntimeError("eviction plan required capacity does not match environment")
        if len(victims) != len(set(victims)):
            raise RuntimeError("eviction policy selected duplicate victims")
        if any(victim not in residents for victim in victims):
            raise RuntimeError("eviction policy selected a non-resident victim")
        if protected_object_id is not None and protected_object_id in victims:
            raise RuntimeError("eviction policy selected the protected incoming object")
        sizes = self._resident_sizes_for_policy(rsu)
        actual_freed = sum(sizes[victim] for victim in victims)
        if abs(actual_freed - float(plan.cumulative_freed_capacity)) > CACHE_CAPACITY_EPSILON:
            raise RuntimeError("eviction plan freed capacity does not match resident sizes")
        if bool(actual_freed + CACHE_CAPACITY_EPSILON >= required_free_capacity) != plan.sufficient:
            raise RuntimeError("eviction plan sufficient flag is inconsistent")

    def _remove_catalog_cached_adapter(self, rsu_id: str, adapter_id: str) -> None:
        for profile in self.adapter_catalog.rsu_adapter_caches:
            if profile.rsu_id == rsu_id and adapter_id in profile.cached_adapter_ids:
                profile.cached_adapter_ids.remove(adapter_id)
                return

    def _cache_capacity_snapshot(self, rsu_id: str | None) -> dict[str, Any]:
        capacity_enabled = self._cache_capacity_enabled()
        capacity = self._cache_capacity_value()
        used_size = None
        remaining_size = None
        occupancy_rate = None
        if rsu_id is not None:
            rsu = self._get_rsu_map().get(rsu_id)
            if rsu is not None and capacity_enabled and capacity is not None:
                used_size = self._cache_used_value(rsu)
                remaining_size = max(float(capacity) - float(used_size), 0.0)
                occupancy_rate = round(float(used_size) / float(capacity), 6)
        return {
            "model_cache_profile_id": self._cache_capacity_profile.get(
                "model_cache_profile_id", LEGACY_MODEL_CACHE_PROFILE_ID
            ),
            "cache_capacity_enabled": capacity_enabled,
            "cache_capacity_unit": self._cache_capacity_profile.get("unit", "adapter_slots"),
            "eviction_policy": self._eviction_policy.policy_name,
            "eviction_policy_version": self._eviction_policy.policy_version,
            "eviction_policy_seed": self._cache_capacity_profile.get("eviction_policy_seed"),
            "rsu_adapter_slots": int(self._cache_capacity_profile.get("rsu_adapter_slots", 0) or 0),
            "cache_capacity": capacity,
            "cache_used_size": used_size,
            "cache_remaining_size": remaining_size,
            "cache_occupancy_rate": occupancy_rate,
            "cache_used_mb_by_object_type": (
                self._typed_used_mb_by_type(rsu_id)
                if rsu_id is not None and self._typed_mode_enabled()
                else None
            ),
        }

    def _apply_cache_action(
        self,
        control: ControlAction,
        primary_vehicle: VehicleState | None,
        current_node_id: str | None,
        required_adapter: str | None,
    ) -> dict[str, Any]:
        if self._typed_mode_enabled():
            return self._apply_typed_cache_action(
                control=control,
                primary_vehicle=primary_vehicle,
                current_node_id=current_node_id,
                required_adapter=required_adapter,
            )
        if current_node_id is None or required_adapter is None:
            return self._default_cache_result()
        if not control.cache_action:
            return self._default_cache_result()
        operation = control.cache_action.get("operation", "cache")
        if operation == "noop":
            return self._default_cache_result()

        strategy = control.cache_action.get("strategy", "manual_cache")
        prediction_driven = bool(control.cache_action.get("prediction_driven", False))
        decision_target_rsu_id = control.cache_action.get("rsu_id")
        adapter_id = control.cache_action.get("adapter_id") or required_adapter
        current_associated_rsu_id = primary_vehicle.associated_rsu_id if primary_vehicle else None
        execution_target_rsu_id = decision_target_rsu_id or current_associated_rsu_id
        cache_target_corrected_by_handoff = False

        if (
            strategy == "reactive_cache_fill"
            and current_associated_rsu_id is not None
            and execution_target_rsu_id != current_associated_rsu_id
        ):
            execution_target_rsu_id = current_associated_rsu_id
            cache_target_corrected_by_handoff = True

        if execution_target_rsu_id is None:
            return {
                **self._default_cache_result(),
                "requested": True,
                "decision_target_rsu_id": decision_target_rsu_id,
                "target_rsu_id": execution_target_rsu_id,
                "adapter_id": adapter_id,
                "strategy": strategy,
                "prediction_driven": prediction_driven,
                "cache_target_corrected_by_handoff": cache_target_corrected_by_handoff,
            }

        rsu = self._get_rsu_map().get(execution_target_rsu_id)
        if rsu is None:
            return {
                **self._default_cache_result(),
                "requested": True,
                "decision_target_rsu_id": decision_target_rsu_id,
                "target_rsu_id": execution_target_rsu_id,
                "adapter_id": adapter_id,
                "strategy": strategy,
                "prediction_driven": prediction_driven,
                "cache_target_corrected_by_handoff": cache_target_corrected_by_handoff,
            }

        was_cached_before = adapter_id in rsu.cached_adapter_ids
        size_resolution = self.adapter_catalog.resolve_adapter_resident_size_mb(adapter_id)
        capacity_before = self._cache_capacity_snapshot(execution_target_rsu_id)
        added_new_adapter = False
        evicted_adapter_ids: list[str] = []
        eviction_plan: EvictionPlan | None = None
        rejection_reason = None
        if adapter_id not in rsu.cached_adapter_ids:
            if self._cache_capacity_enabled():
                capacity = float(self._cache_capacity_value() or 0.0)
                object_size = size_resolution.size_mb if self._cache_capacity_profile["unit"] == "mb" else 1.0
                if object_size > capacity + CACHE_CAPACITY_EPSILON:
                    rejection_reason = "object_exceeds_total_capacity"
                else:
                    used = float(self._cache_used_value(rsu))
                    required = max(used + object_size - capacity, 0.0)
                    if required > CACHE_CAPACITY_EPSILON:
                        eviction_plan = self._eviction_policy.plan_victims(
                            rsu_id=rsu.rsu_id,
                            resident_ids=list(rsu.cached_adapter_ids),
                            resident_sizes=self._resident_sizes_for_policy(rsu),
                            required_free_capacity=required,
                            protected_object_id=adapter_id,
                            capacity_unit=self._cache_capacity_profile["unit"],
                            current_step=self._episode_steps,
                        )
                        self._validate_eviction_plan(
                            plan=eviction_plan,
                            rsu=rsu,
                            required_free_capacity=required,
                            protected_object_id=adapter_id,
                        )
                        self._last_eviction_plan = eviction_plan.to_dict()
                        evicted_adapter_ids = list(eviction_plan.ordered_victim_ids)
                    if eviction_plan is not None and not eviction_plan.sufficient:
                        rejection_reason = "insufficient_evictable_capacity"
                        evicted_adapter_ids = []
            if rejection_reason is None:
                if any(victim not in rsu.cached_adapter_ids for victim in evicted_adapter_ids):
                    raise RuntimeError("eviction policy selected a non-resident victim")
                for victim in evicted_adapter_ids:
                    rsu.cached_adapter_ids.remove(victim)
                    self._eviction_policy.on_eviction(
                        rsu_id=rsu.rsu_id,
                        object_id=victim,
                        current_step=self._episode_steps,
                    )
                    self._remove_catalog_cached_adapter(rsu.rsu_id, victim)
                rsu.cached_adapter_ids.append(adapter_id)
                self.adapter_catalog.ensure_cached_adapter(execution_target_rsu_id, adapter_id)
                self._eviction_policy.on_admission(
                    rsu_id=execution_target_rsu_id,
                    object_id=adapter_id,
                    current_step=self._episode_steps,
                )
                added_new_adapter = True
        if was_cached_before:
            self._touch_cached_adapter(execution_target_rsu_id, adapter_id)
        evicted_size_mb_sum = sum(self._adapter_resident_size_mb(item) for item in evicted_adapter_ids)
        return {
            "requested": True,
            "decision_target_rsu_id": decision_target_rsu_id,
            "target_rsu_id": execution_target_rsu_id,
            "adapter_id": adapter_id,
            "was_cached_before": was_cached_before,
            "added_new_adapter": added_new_adapter,
            "cache_admission_added_new_adapter": added_new_adapter,
            "cache_eviction": bool(evicted_adapter_ids),
            "eviction_count": len(evicted_adapter_ids),
            "evicted_adapter_count": len(evicted_adapter_ids),
            "evicted_adapter_id": evicted_adapter_ids[0] if evicted_adapter_ids else None,
            "evicted_adapter_ids": evicted_adapter_ids,
            "evicted_size_mb_sum": evicted_size_mb_sum,
            "requested_object_size_mb": size_resolution.size_mb,
            "resident_size_source": size_resolution.source,
            "capacity_rejection_reason": rejection_reason,
            "eviction_plan": eviction_plan.to_dict() if eviction_plan is not None else None,
            "strategy": strategy,
            "prediction_driven": prediction_driven,
            "cache_target_corrected_by_handoff": cache_target_corrected_by_handoff,
            "capacity_before": capacity_before,
            **self._cache_capacity_snapshot(execution_target_rsu_id),
        }

    def _typed_evictable_residents(self, rsu_id: str) -> list[str]:
        residents = list(self._typed_resident_object_ids.get(rsu_id, []))
        resident_set = set(residents)
        result = []
        for object_id in residents:
            item = self.adapter_catalog.get_typed_object(object_id)
            if item.evictability != "evictable":
                continue
            if item.object_type == "base_model" and any(
                object_id in self.adapter_catalog.get_typed_object(candidate).dependency_ids
                for candidate in resident_set
                if candidate != object_id
            ):
                continue
            result.append(object_id)
        return result

    @staticmethod
    def _typed_rows_by_type(rows: list[dict[str, Any]]) -> dict[str, float]:
        totals: dict[str, float] = {}
        for row in rows:
            object_type = str(row["object_type"])
            totals[object_type] = totals.get(object_type, 0.0) + float(row["resident_size_mb"])
        return {key: round(value, 6) for key, value in sorted(totals.items())}

    def _typed_object_row(self, object_id: str) -> dict[str, Any]:
        item = self.adapter_catalog.get_typed_object(object_id)
        return {
            "object_id": item.object_id,
            "object_type": item.object_type,
            "version": item.version,
            "resident_size_mb": float(item.resident_size_mb),
            "transfer_size_mb": float(item.transfer_size_mb),
            "adapter_id": item.adapter_id,
            "base_model_id": item.base_model_id,
            "required_base_model_id": item.required_base_model_id,
            "dependency_ids": list(item.dependency_ids),
            "evictability": item.evictability,
        }

    def _apply_typed_cache_action(
        self,
        *,
        control: ControlAction,
        primary_vehicle: VehicleState | None,
        current_node_id: str | None,
        required_adapter: str | None,
    ) -> dict[str, Any]:
        if current_node_id is None or required_adapter is None or not control.cache_action:
            return self._default_cache_result()
        if control.cache_action.get("operation", "cache") == "noop":
            return self._default_cache_result()
        strategy = str(control.cache_action.get("strategy", "manual_cache"))
        prediction_driven = bool(control.cache_action.get("prediction_driven", False))
        decision_target_rsu_id = control.cache_action.get("rsu_id")
        current_rsu_id = primary_vehicle.associated_rsu_id if primary_vehicle else None
        target_rsu_id = decision_target_rsu_id or current_rsu_id
        corrected = False
        if strategy == "reactive_cache_fill" and current_rsu_id is not None and target_rsu_id != current_rsu_id:
            target_rsu_id = current_rsu_id
            corrected = True
        base_result = {
            **self._default_cache_result(),
            "requested": True,
            "decision_target_rsu_id": decision_target_rsu_id,
            "target_rsu_id": target_rsu_id,
            "adapter_id": required_adapter,
            "strategy": strategy,
            "prediction_driven": prediction_driven,
            "cache_target_corrected_by_handoff": corrected,
        }
        rsu = self._get_rsu_map().get(target_rsu_id) if target_rsu_id else None
        if rsu is None:
            return {**base_result, "atomic_transaction_status": "rejected_missing_target_rsu"}
        residents = list(self._typed_resident_object_ids.get(rsu.rsu_id, []))
        placement = self.adapter_catalog.resolve_typed_placement_plan(
            adapter_id=required_adapter,
            resident_object_ids=residents,
        )
        if len(placement.ordered_object_ids) > TYPED_MAX_DEPENDENCY_BUNDLE_OBJECTS:
            raise RuntimeError("typed dependency bundle exceeds frozen object limit")
        requested_rows = [self._typed_object_row(item) for item in placement.ordered_object_ids]
        missing_rows = [self._typed_object_row(item) for item in placement.missing_object_ids]
        capacity_before = self._cache_capacity_snapshot(rsu.rsu_id)
        adapter = self.adapter_catalog.get_typed_adapter(required_adapter)
        adapter_was_resident = adapter.object_id in residents
        for object_id in placement.already_resident_object_ids:
            self._eviction_policy.on_hit(
                rsu_id=rsu.rsu_id, object_id=object_id, current_step=self._episode_steps
            )
        if not placement.missing_object_ids:
            return {
                **base_result,
                "was_cached_before": True,
                "dependency_bundle": placement.to_dict(),
                "requested_typed_objects": requested_rows,
                "atomic_transaction_status": "noop_all_resident",
                "orphan_count": 0,
                "capacity_before": capacity_before,
                **self._cache_capacity_snapshot(rsu.rsu_id),
            }
        capacity = float(self._cache_capacity_value() or 0.0)
        requested_mb = float(placement.requested_bundle_mb)
        rejection_reason: str | None = None
        if any(float(row["resident_size_mb"]) > capacity + CACHE_CAPACITY_EPSILON for row in missing_rows):
            rejection_reason = "object_exceeds_total_capacity"
        elif requested_mb > capacity + CACHE_CAPACITY_EPSILON:
            rejection_reason = "dependency_bundle_exceeds_total_capacity"
        used = float(self._cache_used_value(rsu))
        required_free = max(used + requested_mb - capacity, 0.0)
        eviction_plan: EvictionPlan | None = None
        victim_ids: list[str] = []
        if rejection_reason is None and required_free > CACHE_CAPACITY_EPSILON:
            eligible = self._typed_evictable_residents(rsu.rsu_id)
            eviction_plan = self._eviction_policy.plan_victims(
                rsu_id=rsu.rsu_id,
                resident_ids=eligible,
                resident_sizes=self._resident_sizes_for_policy(rsu),
                required_free_capacity=required_free,
                protected_object_id=None,
                capacity_unit="mb",
                current_step=self._episode_steps,
            )
            self._validate_eviction_plan(
                plan=eviction_plan,
                rsu=rsu,
                required_free_capacity=required_free,
                protected_object_id=None,
            )
            self._last_eviction_plan = eviction_plan.to_dict()
            if not eviction_plan.sufficient:
                rejection_reason = "insufficient_dependency_safe_evictable_capacity"
            else:
                victim_ids = list(eviction_plan.ordered_victim_ids)
        if rejection_reason is not None:
            return {
                **base_result,
                "was_cached_before": adapter_was_resident,
                "dependency_bundle": placement.to_dict(),
                "requested_typed_objects": requested_rows,
                "requested_object_size_mb": requested_mb,
                "capacity_rejection_reason": rejection_reason,
                "atomic_transaction_status": "rolled_back_no_mutation",
                "orphan_count": 0,
                "eviction_plan": eviction_plan.to_dict() if eviction_plan else None,
                "capacity_before": capacity_before,
                **self._cache_capacity_snapshot(rsu.rsu_id),
            }
        evicted_rows = [self._typed_object_row(item) for item in victim_ids]
        admitted_rows = [self._typed_object_row(item) for item in placement.missing_object_ids]
        next_residents = [item for item in residents if item not in set(victim_ids)]
        next_residents.extend(placement.missing_object_ids)
        # All checks precede this commit point; callbacks and resident mutation now form one transaction.
        for victim_id in victim_ids:
            self._eviction_policy.on_eviction(
                rsu_id=rsu.rsu_id, object_id=victim_id, current_step=self._episode_steps
            )
        for object_id in placement.missing_object_ids:
            self._eviction_policy.on_admission(
                rsu_id=rsu.rsu_id, object_id=object_id, current_step=self._episode_steps
            )
        self._typed_resident_object_ids[rsu.rsu_id] = next_residents
        self._validate_typed_resident_invariants(rsu.rsu_id)
        if float(self._cache_used_value(rsu)) > capacity + CACHE_CAPACITY_EPSILON:
            raise RuntimeError("typed transaction violated MB capacity")
        self._sync_legacy_adapter_views_from_typed()
        adapter_admitted = adapter.object_id in placement.missing_object_ids
        evicted_adapter_ids = [
            str(row["adapter_id"]) for row in evicted_rows if row.get("adapter_id")
        ]
        return {
            **base_result,
            "was_cached_before": adapter_was_resident,
            "added_new_adapter": adapter_admitted,
            "cache_admission_added_new_adapter": adapter_admitted,
            "cache_eviction": bool(victim_ids),
            "eviction_count": len(victim_ids),
            "evicted_adapter_count": len(evicted_adapter_ids),
            "evicted_adapter_id": evicted_adapter_ids[0] if evicted_adapter_ids else None,
            "evicted_adapter_ids": evicted_adapter_ids,
            "evicted_object_ids": victim_ids,
            "evicted_size_mb_sum": sum(float(row["resident_size_mb"]) for row in evicted_rows),
            "requested_object_size_mb": requested_mb,
            "resident_size_source": "typed_catalog_dependency_bundle",
            "dependency_bundle": placement.to_dict(),
            "requested_typed_objects": requested_rows,
            "admitted_typed_objects": admitted_rows,
            "evicted_typed_objects": evicted_rows,
            "admitted_mb_by_type": self._typed_rows_by_type(admitted_rows),
            "evicted_mb_by_type": self._typed_rows_by_type(evicted_rows),
            "transfer_mb_by_type": dict(placement.transfer_mb_by_type),
            "atomic_transaction_status": "committed",
            "orphan_count": 0,
            "eviction_plan": eviction_plan.to_dict() if eviction_plan else None,
            "capacity_before": capacity_before,
            **self._cache_capacity_snapshot(rsu.rsu_id),
        }

    def _check_rsu_has_required_adapter(
        self,
        rsu_id: str | None,
        required_adapter: str | None,
    ) -> bool:
        if rsu_id is None or required_adapter is None:
            return False
        rsu = self._get_rsu_map().get(rsu_id)
        if rsu is None:
            return False
        cache_hit = required_adapter in rsu.cached_adapter_ids
        if cache_hit:
            self._touch_cached_adapter(rsu_id, required_adapter)
        return cache_hit

    def _typed_service_readiness(
        self,
        *,
        current_node: Any,
        primary_vehicle: VehicleState | None,
        offload_mode: str,
        service_rsu_id: str | None,
        state_required: bool,
        state_ready: bool,
    ) -> dict[str, Any]:
        if current_node is None:
            return {
                "base_ready": False,
                "adapter_ready": False,
                "joint_base_adapter_hit": False,
                "state_required": False,
                "state_ready": False,
                "full_service_ready": False,
                "missing_object_types": [],
                "incompatibility_reason": "not_applicable",
                "service_scope": "not_applicable",
                "per_object_lookup_results": [],
            }
        adapter = self.adapter_catalog.get_typed_adapter(current_node.required_adapter)
        base = self.adapter_catalog.get_typed_base(current_node.required_base_model)
        compatible = (
            adapter.required_base_model_id == current_node.required_base_model
            and adapter.base_model_family == base.base_model_family
            and current_node.required_adapter
            in self.adapter_catalog.compatibility_map.get(current_node.required_base_model, [])
        )
        lookup_rows: list[dict[str, Any]] = []
        if offload_mode == "vehicle":
            base_ready = bool(
                primary_vehicle
                and primary_vehicle.base_model_id == current_node.required_base_model
            )
            adapter_ready = False
            service_scope = "vehicle_local"
            lookup_rows = [
                {"object_id": base.object_id, "object_type": "base_model", "resident": base_ready, "evidence": "vehicle_capability"},
                {"object_id": adapter.object_id, "object_type": "adapter", "resident": False, "evidence": "vehicle_adapter_residency_disabled"},
            ]
        else:
            residents = set(self._typed_resident_object_ids.get(str(service_rsu_id), []))
            base_ready = base.object_id in residents
            adapter_ready = adapter.object_id in residents
            service_scope = "rsu"
            lookup_rows = [
                {"object_id": base.object_id, "object_type": "base_model", "resident": base_ready, "evidence": "rsu_resident_state"},
                {"object_id": adapter.object_id, "object_type": "adapter", "resident": adapter_ready, "evidence": "rsu_resident_state"},
            ]
        effective_state_ready = bool(not state_required or state_ready)
        missing = []
        if not base_ready:
            missing.append("base_model")
        if not adapter_ready:
            missing.append("adapter")
        if state_required and not effective_state_ready:
            missing.append("workflow_state")
        if not compatible:
            reason = "base_adapter_family_or_version_incompatible"
        elif service_rsu_id is None and offload_mode == "rsu":
            reason = "illegal_or_missing_service_target"
        elif missing:
            reason = "missing:" + ",".join(missing)
        else:
            reason = None
        full_ready = bool(
            compatible
            and base_ready
            and adapter_ready
            and effective_state_ready
            and (offload_mode != "rsu" or service_rsu_id in self._get_rsu_map())
        )
        return {
            "base_ready": base_ready,
            "adapter_ready": adapter_ready,
            "joint_base_adapter_hit": bool(base_ready and adapter_ready and compatible),
            "state_required": bool(state_required),
            "state_ready": effective_state_ready,
            "full_service_ready": full_ready,
            "missing_object_types": missing,
            "incompatibility_reason": reason,
            "compatibility_result": "compatible" if compatible else "incompatible",
            "service_scope": service_scope,
            "per_object_lookup_results": lookup_rows,
        }

    def _get_rsu_map(self) -> dict[str, RSUState]:
        return {rsu.rsu_id: rsu for rsu in self.rsu_states}

    def _extract_prediction_snapshot(
        self,
        state: dict[str, Any],
        vehicle_id: str | None,
    ) -> dict[str, Any]:
        predictions = state.get("predictions", {})
        if not vehicle_id:
            return {
                "predicted_next_rsu_id": None,
                "predicted_handoff_target_rsu_id": None,
                "prediction_confidence": 0.0,
                "has_predicted_handoff_target": False,
                "next_rsu_sequence": [],
                "predictor_name": predictions.get("predictor_name"),
                "predictor_kind": predictions.get("predictor_kind"),
                "surrogate_claim_boundary": predictions.get("surrogate_claim_boundary"),
                "prediction_quality_audit": dict(predictions.get("prediction_quality_audit", {})),
                "causal_snapshot_id": None,
                "causal_snapshot_contract_version": predictions.get("causal_predictor_snapshot_contract_version"),
                "causal_snapshot_availability_mask": 0,
                "causal_snapshot_generated_at_step": None,
                "causal_snapshot_observation_as_of_step": None,
                "causal_snapshot_age_steps": None,
            }
        next_sequence = list(predictions.get("next_rsu_sequence", {}).get(vehicle_id, []))
        predicted_handoff_target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
        causal_snapshot = dict(
            predictions.get("causal_predictor_snapshots_by_vehicle", {}).get(vehicle_id, {})
        )
        causal_identity = dict(causal_snapshot.get("identity", {}))
        causal_time = dict(causal_snapshot.get("causal_time", {}))
        return {
            "predicted_next_rsu_id": predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id),
            "predicted_handoff_target_rsu_id": predicted_handoff_target_rsu_id,
            "prediction_confidence": float(
                predictions.get("prediction_confidence_by_vehicle", {}).get(vehicle_id, 0.0) or 0.0
            ),
            "has_predicted_handoff_target": bool(predicted_handoff_target_rsu_id is not None),
            "next_rsu_sequence": next_sequence,
            "predictor_name": predictions.get("predictor_name"),
            "predictor_kind": predictions.get("predictor_kind"),
            "surrogate_claim_boundary": predictions.get("surrogate_claim_boundary"),
            "prediction_quality_audit": dict(predictions.get("prediction_quality_audit", {})),
            "causal_snapshot_id": causal_identity.get("snapshot_id"),
            "causal_snapshot_contract_version": causal_identity.get("contract_version"),
            "causal_snapshot_availability_mask": predictions.get(
                "causal_snapshot_availability_by_vehicle", {}
            ).get(vehicle_id, 0),
            "causal_snapshot_generated_at_step": causal_time.get("generated_at_step"),
            "causal_snapshot_observation_as_of_step": causal_time.get("observation_as_of_step"),
            "causal_snapshot_age_steps": causal_time.get("age_steps"),
        }

    def _has_predicted_handoff_signal(
        self,
        prediction_snapshot: dict[str, Any],
        current_rsu_id: str | None,
    ) -> bool:
        has_predicted_handoff_target = bool(prediction_snapshot.get("has_predicted_handoff_target", False))
        if has_predicted_handoff_target:
            return True
        prediction_confidence = float(prediction_snapshot.get("prediction_confidence", 0.0) or 0.0)
        predicted_next_rsu_id = prediction_snapshot.get("predicted_next_rsu_id")
        return bool(
            prediction_confidence >= 0.7
            and predicted_next_rsu_id is not None
            and predicted_next_rsu_id != current_rsu_id
        )

    def _is_gap_transfer_entry(
        self,
        *,
        pre_action_associated_rsu_id: str | None,
        post_action_associated_rsu_id: str | None,
        prediction_snapshot: dict[str, Any],
        prepare_action_context: dict[str, Any] | None,
    ) -> bool:
        if pre_action_associated_rsu_id is not None or post_action_associated_rsu_id is None:
            return False
        candidate_targets = {
            prediction_snapshot.get("predicted_handoff_target_rsu_id"),
            prediction_snapshot.get("predicted_next_rsu_id"),
        }
        if prepare_action_context is not None:
            candidate_targets.add(prepare_action_context.get("target_rsu_id"))
        for prepare_entry in self._prepare_history:
            candidate_targets.add(prepare_entry.get("target_rsu_id"))
        return post_action_associated_rsu_id in candidate_targets

    def _is_mechanism_exploration_action(self, control: ControlAction) -> bool:
        return bool(
            control.migration_action.get("mode") == "prepare"
            or control.cache_action.get("strategy") == "predictive_prefetch"
        )

    def _build_prepare_action_context(
        self,
        control: ControlAction,
        vehicle_id: str | None,
        required_adapter: str | None,
        prediction_snapshot: dict[str, Any],
    ) -> dict[str, Any] | None:
        if control.migration_action.get("mode") != "prepare":
            return None
        target_rsu_id = control.migration_action.get("expected_target_rsu_id") or prediction_snapshot.get(
            "predicted_handoff_target_rsu_id"
        )
        if not vehicle_id or not required_adapter or target_rsu_id is None:
            return None
        return {
            "vehicle_id": vehicle_id,
            "target_rsu_id": target_rsu_id,
            "required_adapter": required_adapter,
            "prepared_at_step": self._episode_steps,
            "prepared_at_time_index": self._mobility_provider.get_time(),
        }

    def _register_prepare_action(
        self,
        prepare_action_context: dict[str, Any] | None,
        realized_prepare: dict[str, Any],
    ) -> None:
        if prepare_action_context is None:
            self._prune_prepare_history()
            return
        if realized_prepare.get("realized", False) and realized_prepare.get("source") == "same_step":
            self._prune_prepare_history()
            return
        for item in self._prepare_history:
            if (
                item["vehicle_id"] == prepare_action_context["vehicle_id"]
                and item["target_rsu_id"] == prepare_action_context["target_rsu_id"]
                and item["required_adapter"] == prepare_action_context["required_adapter"]
            ):
                item.update(prepare_action_context)
                self._prune_prepare_history()
                return
        self._prepare_history.append(prepare_action_context)
        self._prune_prepare_history()

    def _consume_realized_prepare(
        self,
        vehicle_id: str | None,
        actual_target_rsu_id: str | None,
        required_adapter: str | None,
        handoff_count: int,
        current_prepare_action: dict[str, Any] | None,
    ) -> dict[str, Any]:
        self._prune_prepare_history()
        if handoff_count <= 0 or not vehicle_id or not actual_target_rsu_id or not required_adapter:
            return self._default_prepare_realization()

        if current_prepare_action is not None:
            if (
                current_prepare_action["vehicle_id"] == vehicle_id
                and current_prepare_action["target_rsu_id"] == actual_target_rsu_id
                and current_prepare_action["required_adapter"] == required_adapter
            ):
                return {
                    "realized": True,
                    "source": "same_step",
                    "vehicle_id": vehicle_id,
                    "target_rsu_id": actual_target_rsu_id,
                    "required_adapter": required_adapter,
                    "prepared_at_step": self._episode_steps,
                    "prepare_age": 0,
                }

        matched_index: int | None = None
        matched_entry: dict[str, Any] | None = None
        for index, entry in enumerate(self._prepare_history):
            if entry["vehicle_id"] != vehicle_id:
                continue
            if entry["target_rsu_id"] != actual_target_rsu_id:
                continue
            prepare_age = self._episode_steps - int(entry["prepared_at_step"])
            if prepare_age < 1 or prepare_age > self._handoff_prepare_window:
                continue
            if matched_entry is None or int(entry["prepared_at_step"]) > int(matched_entry["prepared_at_step"]):
                matched_index = index
                matched_entry = entry

        if matched_entry is None or matched_index is None:
            return self._default_prepare_realization()

        self._prepare_history.pop(matched_index)
        return {
            "realized": True,
            "source": "history",
            "vehicle_id": vehicle_id,
            "target_rsu_id": actual_target_rsu_id,
            "required_adapter": required_adapter,
            "prepared_required_adapter": matched_entry.get("required_adapter"),
            "required_adapter_match": bool(matched_entry.get("required_adapter") == required_adapter),
            "prepared_at_step": matched_entry["prepared_at_step"],
            "prepare_age": self._episode_steps - int(matched_entry["prepared_at_step"]),
        }

    def _prune_prepare_history(self) -> None:
        self._prepare_history = [
            entry
            for entry in self._prepare_history
            if self._episode_steps - int(entry["prepared_at_step"]) <= self._handoff_prepare_window
        ]

    def _default_prepare_realization(self) -> dict[str, Any]:
        return {
            "realized": False,
            "source": None,
            "vehicle_id": None,
            "target_rsu_id": None,
            "required_adapter": None,
            "prepared_required_adapter": None,
            "required_adapter_match": False,
            "prepared_at_step": None,
            "prepare_age": None,
        }

    def _build_dag_evidence_metrics(self, current_node: Any) -> dict[str, float | int | str | None]:
        completed = {str(node_id) for node_id in self.workflow_state.completed_node_ids}
        node_map = {str(node.node_id): node for node in self.workflow_state.nodes}
        remaining_nodes = [
            node for node in self.workflow_state.nodes if str(node.node_id) not in completed
        ]
        remaining_ids = {str(node.node_id) for node in remaining_nodes}
        frontier_nodes = [
            node
            for node in remaining_nodes
            if all(str(predecessor) in completed for predecessor in node.predecessors)
        ]

        def critical_path_from(node_id: str, visiting: set[str] | None = None) -> int:
            visiting = set(visiting or set())
            if node_id in visiting:
                return 0
            visiting.add(node_id)
            node = node_map.get(node_id)
            if node is None or node_id not in remaining_ids:
                return 0
            child_lengths = [
                critical_path_from(str(successor), visiting)
                for successor in node.successors
                if str(successor) in remaining_ids
            ]
            return 1 + (max(child_lengths) if child_lengths else 0)

        critical_path_length = max(
            [critical_path_from(str(node.node_id)) for node in frontier_nodes],
            default=0,
        )
        current_node_id = str(current_node.node_id) if current_node else None
        current_predecessors = list(getattr(current_node, "predecessors", []) or [])
        unmet_predecessors = [
            predecessor
            for predecessor in current_predecessors
            if str(predecessor) not in completed
        ]
        current_successors = list(getattr(current_node, "successors", []) or [])
        remaining_count = len(remaining_nodes)
        predecessor_count = len(current_predecessors)
        return {
            "dag_current_node_id": current_node_id,
            "dag_frontier_size": len(frontier_nodes),
            "dag_frontier_width_ratio": round(
                float(len(frontier_nodes)) / float(max(remaining_count, 1)),
                6,
            ),
            "dag_remaining_nodes": remaining_count,
            "dag_remaining_nodes_ratio": round(
                float(remaining_count) / float(max(len(self.workflow_state.nodes), 1)),
                6,
            ),
            "dag_current_node_predecessor_count": predecessor_count,
            "dag_current_node_successor_count": len(current_successors),
            "dag_current_node_dependency_pressure": round(
                float(len(unmet_predecessors)) / float(max(predecessor_count, 1)),
                6,
            ),
            "dag_critical_path_length": critical_path_length,
            "dag_critical_path_pressure": round(
                float(critical_path_length) / float(max(remaining_count, 1)),
                6,
            ),
        }

    def _build_info(
        self,
        current_node: Any,
        primary_vehicle: VehicleState | None,
        handoff_events: list[dict[str, Any]],
        cache_hit: bool,
        offload_target_rsu_id: str | None,
        stall_occurred: bool,
        reward: RewardBreakdown,
        control: ControlAction,
        cache_result: dict[str, Any],
        handoff_count: int,
        pre_action_associated_rsu_id: str | None,
        pre_action_prediction_snapshot: dict[str, Any],
        realized_prepare: dict[str, Any],
        pre_execution_cache_hit: bool = False,
        raw_handoff_count: int = 0,
        gap_transfer_count: int = 0,
    ) -> dict[str, Any]:
        post_action_associated_rsu_id = primary_vehicle.associated_rsu_id if primary_vehicle else None
        control_metadata = dict(getattr(control, "metadata", {}) or {})
        migration_mode = control.migration_action.get("mode", "keep")
        predicted_next_rsu_id = pre_action_prediction_snapshot.get("predicted_next_rsu_id")
        predicted_handoff_target_rsu_id = (
            control.migration_action.get("expected_target_rsu_id")
            or pre_action_prediction_snapshot.get("predicted_handoff_target_rsu_id")
        )
        prediction_quality_audit = dict(pre_action_prediction_snapshot.get("prediction_quality_audit", {}))
        prediction_confidence = float(pre_action_prediction_snapshot.get("prediction_confidence", 0.0) or 0.0)
        has_predicted_handoff_target = bool(pre_action_prediction_snapshot.get("has_predicted_handoff_target", False))
        predicted_handoff_signal = self._has_predicted_handoff_signal(
            prediction_snapshot=pre_action_prediction_snapshot,
            current_rsu_id=pre_action_associated_rsu_id,
        )
        mechanism_exploration_action = self._is_mechanism_exploration_action(control)
        migration_prepare_requested = bool(migration_mode == "prepare")
        migration_prepare_realized = bool(realized_prepare.get("realized", False))
        migration_prepare_target_rsu_id = (
            realized_prepare.get("target_rsu_id")
            or control.migration_action.get("expected_target_rsu_id")
            or pre_action_prediction_snapshot.get("predicted_handoff_target_rsu_id")
        )
        warm_ready = bool(
            cache_hit
            or pre_execution_cache_hit
            or cache_result.get("was_cached_before", False)
        )
        prepared_target_aligned = True
        if migration_prepare_realized:
            prepared_target_aligned = realized_prepare.get("target_rsu_id") == post_action_associated_rsu_id
        elif migration_prepare_requested and migration_prepare_target_rsu_id is not None:
            prepared_target_aligned = migration_prepare_target_rsu_id == post_action_associated_rsu_id
        migration_during_handoff = bool(
            handoff_count > 0 and (migration_mode in {"prepare", "migrate"} or migration_prepare_realized)
        )
        reactive_cache_fill = bool(
            cache_result.get("requested", False)
            and cache_result.get("strategy") == "reactive_cache_fill"
        )
        predictive_prefetch_requested = bool(
            cache_result.get("requested", False)
            and cache_result.get("strategy") in {"predictive_prefetch", "handoff_prepare_prefetch"}
            and cache_result.get("prediction_driven", False)
            and cache_result.get("target_rsu_id") is not None
            and cache_result.get("target_rsu_id") != pre_action_associated_rsu_id
        )
        handoff_ready = bool(
            handoff_count > 0
            and migration_during_handoff
            and prepared_target_aligned
            and warm_ready
        )
        handoff_failed = bool(handoff_count > 0 and stall_occurred)
        warm_hit = bool(cache_hit and warm_ready)
        cross_rsu_cold_start = bool(
            handoff_count > 0
            and cache_result.get("added_new_adapter", False)
            and cache_result.get("target_rsu_id") != pre_action_associated_rsu_id
        )
        backhaul_traffic_cost = self._estimate_backhaul_traffic_cost(
            adapter_id=current_node.required_adapter if current_node else None,
            cache_result=cache_result,
            handoff_count=handoff_count,
            migration_mode=migration_mode,
            realized_prepare=realized_prepare,
        )
        cache_target_alignment_mismatch = bool(
            cache_result.get("requested", False)
            and cache_result.get("target_rsu_id") is not None
            and cache_result.get("target_rsu_id") != post_action_associated_rsu_id
            and not predictive_prefetch_requested
        )
        mechanism_attempt_selected = bool(
            predictive_prefetch_requested
            or migration_prepare_requested
            or mechanism_exploration_action
        )
        mechanism_success_strict = bool(
            migration_prepare_realized
            or handoff_ready
        )
        mechanism_success_gate_pending = bool(
            predictive_prefetch_requested and not mechanism_success_strict
        )
        dag_evidence_metrics = self._build_dag_evidence_metrics(current_node)
        cache_event = self._build_cache_event(
            current_node=current_node,
            primary_vehicle=primary_vehicle,
            request_rsu_id=pre_action_associated_rsu_id,
            selected_target_rsu_id=offload_target_rsu_id,
            predicted_next_rsu_id=predicted_next_rsu_id,
            predicted_handoff_target_rsu_id=predicted_handoff_target_rsu_id,
            cache_hit=cache_hit,
            stall_occurred=stall_occurred,
            control=control,
            cache_result=cache_result,
            handoff_count=handoff_count,
            migration_prepare_requested=migration_prepare_requested,
            migration_prepare_realized=migration_prepare_realized,
        )

        metrics_protocol = {
            "time_index": self._mobility_provider.get_time(),
            "required_adapter": current_node.required_adapter if current_node else None,
            "required_base_model": current_node.required_base_model if current_node else None,
            "current_node_id": current_node.node_id if current_node else None,
            "pre_action_associated_rsu_id": pre_action_associated_rsu_id,
            "post_action_associated_rsu_id": post_action_associated_rsu_id,
            "current_associated_rsu_id": post_action_associated_rsu_id,
            "decision_cache_target_rsu_id": cache_result.get("decision_target_rsu_id"),
            "cache_target_rsu_id": cache_result.get("target_rsu_id"),
            "cache_target_corrected_by_handoff": cache_result.get("cache_target_corrected_by_handoff", False),
            "cache_target_alignment_mismatch": cache_target_alignment_mismatch,
            "cache_strategy": cache_result.get("strategy", "none"),
            "reactive_cache_fill": reactive_cache_fill,
            "predictive_prefetch_requested": predictive_prefetch_requested,
            "predictive_prefetch_correct": False,
            "predictive_prefetch_validated": False,
            "predictive_prefetch_pending": predictive_prefetch_requested,
            "predictive_prefetch_validation_state": "pending" if predictive_prefetch_requested else "not_applicable",
            "prefetch_target_rsu_match": False,
            "prefetch_validated_hit": False,
            "prefetch_expired_miss": False,
            "predicted_next_rsu_id": predicted_next_rsu_id,
            "predicted_handoff_target_rsu_id": predicted_handoff_target_rsu_id,
            "predictor_name": pre_action_prediction_snapshot.get("predictor_name"),
            "predictor_kind": pre_action_prediction_snapshot.get("predictor_kind"),
            "surrogate_claim_boundary": pre_action_prediction_snapshot.get("surrogate_claim_boundary"),
            "predictor_handoff_target_precision_proxy": prediction_quality_audit.get("handoff_target_precision_proxy"),
            "predictor_handoff_target_recall_proxy": prediction_quality_audit.get("handoff_target_recall_proxy"),
            "predictor_brier_score_proxy": prediction_quality_audit.get("brier_score_proxy"),
            "predictor_confidence_calibration_error_proxy": prediction_quality_audit.get("confidence_calibration_error_proxy"),
            "predictor_prediction_delay_steps": prediction_quality_audit.get("prediction_delay_steps"),
            "predictor_drop_handoff_prediction_prob": prediction_quality_audit.get("drop_handoff_prediction_prob"),
            "prediction_confidence": round(prediction_confidence, 6),
            "causal_predictor_snapshot_id": pre_action_prediction_snapshot.get("causal_snapshot_id"),
            "causal_predictor_snapshot_contract_version": pre_action_prediction_snapshot.get(
                "causal_snapshot_contract_version"
            ),
            "causal_predictor_snapshot_availability_mask": pre_action_prediction_snapshot.get(
                "causal_snapshot_availability_mask", 0
            ),
            "causal_predictor_snapshot_generated_at_step": pre_action_prediction_snapshot.get(
                "causal_snapshot_generated_at_step"
            ),
            "causal_predictor_snapshot_observation_as_of_step": pre_action_prediction_snapshot.get(
                "causal_snapshot_observation_as_of_step"
            ),
            "causal_predictor_snapshot_age_steps": pre_action_prediction_snapshot.get(
                "causal_snapshot_age_steps"
            ),
            "has_predicted_handoff_target": has_predicted_handoff_target,
            "predicted_handoff_signal": predicted_handoff_signal,
            "handoff_event_count": int(handoff_count),
            "raw_handoff_event_count": int(raw_handoff_count),
            "gap_transfer_event_count": int(gap_transfer_count),
            "handoff_ready": handoff_ready,
            "handoff_ready_from_prepare": bool(handoff_ready and migration_prepare_realized),
            "handoff_failed": handoff_failed,
            "warm_hit": warm_hit,
            "cross_rsu_cold_start": cross_rsu_cold_start,
            "backhaul_traffic_cost": round(backhaul_traffic_cost, 6),
            "adapter_state_migration_overhead": round(reward.migration_cost, 6),
            "migration_mode": migration_mode,
            "migration_prepare_requested": migration_prepare_requested,
            "migration_prepare_target_rsu_id": migration_prepare_target_rsu_id,
            "migration_prepare_realized": migration_prepare_realized,
            "migration_prepare_realized_source": realized_prepare.get("source"),
            "migration_prepare_source_step": realized_prepare.get("prepared_at_step"),
            "migration_prepare_age": realized_prepare.get("prepare_age"),
            "migration_prepare_required_adapter_match": bool(realized_prepare.get("required_adapter_match", False)),
            "migration_prepare_prepared_required_adapter": realized_prepare.get("prepared_required_adapter"),
            "migration_prepare_window": self._handoff_prepare_window,
            "migration_during_handoff": migration_during_handoff,
            "mechanism_exploration_action_selected": mechanism_exploration_action,
            "mechanism_exploration_bonus_awarded": bool(reward.mechanism_exploration_bonus > 0.0),
            "mechanism_exploration_bonus": round(reward.mechanism_exploration_bonus, 6),
            "mechanism_exploration_bonus_role": "shaping_diagnostic",
            "mechanism_attempt_selected": mechanism_attempt_selected,
            "mechanism_success_strict": mechanism_success_strict,
            "mechanism_success_gate_pending": mechanism_success_gate_pending,
            "mechanism_success_gate_source": (
                "migration_prepare_realized"
                if migration_prepare_realized
                else "handoff_ready"
                if handoff_ready
                else "pending_prefetch_validation"
                if mechanism_success_gate_pending
                else "none"
            ),
            "action_invalid": bool(control_metadata.get("invalid_action", False)),
            "action_invalid_reason": str(control_metadata.get("invalid_reason", "none")),
            "action_precondition_valid": not bool(control_metadata.get("invalid_action", False)),
            "stall_occurred": bool(stall_occurred),
            "cache_hit": bool(cache_hit),
            "model_cache_profile_id": self.adapter_catalog.model_cache_profile_id,
            "typed_model_cache_contract_version": (
                TYPED_MODEL_CACHE_CONTRACT_VERSION if self._typed_mode_enabled() else None
            ),
            "base_model_hit": (
                cache_event.base_model_hit if self._typed_mode_enabled() else None
            ),
            "adapter_hit": (
                cache_event.adapter_hit if self._typed_mode_enabled() else None
            ),
            "joint_model_hit": (
                cache_event.joint_model_hit if self._typed_mode_enabled() else None
            ),
            "workflow_state_ready": (
                cache_event.workflow_state_ready if self._typed_mode_enabled() else None
            ),
            "full_service_ready": (
                cache_event.full_service_ready if self._typed_mode_enabled() else None
            ),
            "cache_applied": bool(cache_result.get("requested", False)),
            "cache_admission_count": int(bool(cache_result.get("requested", False))),
            "cache_admission_added_new_adapter": bool(cache_result.get("cache_admission_added_new_adapter", False)),
            "cache_capacity_enabled": bool(cache_result.get("cache_capacity_enabled", False)),
            "cache_capacity_unit": cache_result.get("cache_capacity_unit", "adapter_slots"),
            "rsu_adapter_slots": cache_result.get("rsu_adapter_slots", 0),
            "cache_capacity": cache_result.get("cache_capacity"),
            "cache_used_size": cache_result.get("cache_used_size"),
            "cache_remaining_size": cache_result.get("cache_remaining_size"),
            "cache_occupancy_rate": cache_result.get("cache_occupancy_rate"),
            "cache_eviction": bool(cache_result.get("cache_eviction", False)),
            "eviction_count": int(cache_result.get("eviction_count", 0) or 0),
            "evicted_adapter_count": int(cache_result.get("evicted_adapter_count", 0) or 0),
            "evicted_adapter_id": cache_result.get("evicted_adapter_id"),
            "offload_target_rsu_id": offload_target_rsu_id,
            "cache_event": cache_event.to_dict(),
            **dag_evidence_metrics,
        }
        return {
            "handoff_events": handoff_events,
            "cache_hit": cache_hit,
            "offload_target_rsu_id": offload_target_rsu_id,
            "cache_target_corrected_by_handoff": cache_result.get("cache_target_corrected_by_handoff", False),
            "stall_occurred": stall_occurred,
            "control_hierarchy": {
                "cache_action": "慢时间尺度",
                "offload_action": "快时间尺度",
                "migration_action": "事件触发时间尺度",
            },
            "reward_dict": reward.to_dict(),
            "cache_applied": bool(cache_result.get("requested", False)),
            "metrics_protocol": metrics_protocol,
            "cache_event": cache_event.to_dict(),
        }

    def _build_cache_event(
        self,
        *,
        current_node: Any,
        primary_vehicle: VehicleState | None,
        request_rsu_id: str | None,
        selected_target_rsu_id: str | None,
        predicted_next_rsu_id: str | None,
        predicted_handoff_target_rsu_id: str | None,
        cache_hit: bool,
        stall_occurred: bool,
        control: ControlAction,
        cache_result: dict[str, Any],
        handoff_count: int,
        migration_prepare_requested: bool,
        migration_prepare_realized: bool,
    ) -> CacheEvent:
        adapter_id = current_node.required_adapter if current_node else None
        typed_mode = self._typed_mode_enabled()
        if typed_mode and adapter_id:
            typed_adapter = self.adapter_catalog.get_typed_adapter(adapter_id)
            object_id = typed_adapter.object_id
            size_mb = float(typed_adapter.resident_size_mb)
            transfer_source = "typed_catalog_dependency_bundle"
            size_resolution = None
        else:
            size_resolution = (
                self.adapter_catalog.resolve_adapter_resident_size_mb(adapter_id)
                if adapter_id else None
            )
            object_id = (
                size_resolution.object_id or f"adapter:{adapter_id}"
                if size_resolution else None
            )
            size_mb = size_resolution.size_mb if size_resolution else None
            transfer_source = size_resolution.source if size_resolution else "not_applicable"
        offload_mode = str(control.offload_action.get("mode", "rsu"))
        if current_node is None:
            hit_source = "not_applicable"
        elif offload_mode == "vehicle":
            hit_source = "vehicle_local" if cache_hit else "unserved"
        elif offload_mode == "cloud":
            hit_source = "cloud"
        elif not cache_hit:
            hit_source = "unserved"
        elif selected_target_rsu_id == request_rsu_id:
            hit_source = "current_rsu"
        elif selected_target_rsu_id in {predicted_next_rsu_id, predicted_handoff_target_rsu_id}:
            hit_source = "target_rsu"
        else:
            hit_source = "neighbor_rsu"

        admission_requested = bool(cache_result.get("requested", False))
        admission_added = bool(cache_result.get("added_new_adapter", False))
        if not admission_requested:
            admission_reason = "not_requested"
        elif not cache_result.get("target_rsu_id"):
            admission_reason = "missing_target_rsu"
        elif cache_result.get("was_cached_before", False):
            admission_reason = "already_cached"
        elif cache_result.get("capacity_rejection_reason"):
            admission_reason = str(cache_result["capacity_rejection_reason"])
        elif admission_added:
            admission_reason = str(cache_result.get("strategy") or "manual_cache")
        else:
            admission_reason = "not_added"

        eviction_occurred = bool(cache_result.get("cache_eviction", False))
        evicted_adapter_id = cache_result.get("evicted_adapter_id")
        evicted_adapter_ids = list(cache_result.get("evicted_adapter_ids") or [])
        if typed_mode:
            evicted_object_ids = list(cache_result.get("evicted_object_ids") or [])
            evicted_object_id = evicted_object_ids[0] if evicted_object_ids else None
        else:
            evicted_object_ids = []
            for victim_id in evicted_adapter_ids:
                victim_object = next(
                    (item for item in self.adapter_catalog.cache_objects if item.adapter_id == victim_id),
                    None,
                )
                evicted_object_ids.append(victim_object.object_id if victim_object else f"adapter:{victim_id}")
            evicted_object = next(
                (item for item in self.adapter_catalog.cache_objects if item.adapter_id == evicted_adapter_id),
                None,
            )
            evicted_object_id = (
                evicted_object.object_id
                if evicted_object
                else (f"adapter:{evicted_adapter_id}" if evicted_adapter_id else None)
            )
        capacity_before = dict(cache_result.get("capacity_before") or {})
        capacity_enabled = bool(cache_result.get("cache_capacity_enabled", False))
        adapter_transfer_size = (
            float(cache_result.get("transfer_mb_by_type", {}).get("adapter", 0.0))
            if typed_mode
            else size_mb if admission_added and size_mb is not None else 0.0
        )
        migration_requested_flag = bool(
            migration_prepare_requested
            or migration_prepare_realized
            or control.migration_action.get("mode") == "migrate"
        )
        if typed_mode:
            state_object = self.adapter_catalog.resolve_workflow_state_object()
            migration_transfer_size = (
                float(state_object.transfer_size_mb)
                if state_object and migration_requested_flag
                else 0.0
            )
        else:
            migration_transfer_size = (
                float(self.adapter_catalog.estimate_bundle_transfer_size_mb(adapter_id))
                if adapter_id and migration_requested_flag
                else 0.0
            )
        readiness = dict(cache_result.get("service_readiness") or {})
        typed_transfer_by_type = dict(cache_result.get("transfer_mb_by_type") or {})
        if typed_mode and migration_transfer_size > 0:
            typed_transfer_by_type["workflow_state"] = round(migration_transfer_size, 6)
        return CacheEvent(
            event_id=f"cache-event-{self._episode_steps:06d}",
            event_schema_version=CACHE_EVENT_SCHEMA_VERSION,
            event_type="request" if current_node else "not_applicable",
            time_index=int(self._mobility_provider.get_time()),
            episode_step_index=int(self._episode_steps),
            vehicle_id=primary_vehicle.vehicle_id if primary_vehicle else self._primary_vehicle_id,
            workflow_id=self.workflow_state.workflow_id,
            node_id=current_node.node_id if current_node else None,
            object_id=object_id,
            adapter_id=adapter_id,
            object_type="adapter" if adapter_id else "not_applicable",
            size_mb=size_mb,
            request_rsu_id=request_rsu_id,
            selected_target_rsu_id=selected_target_rsu_id,
            served_rsu_id=(selected_target_rsu_id if current_node and offload_mode == "rsu" and not stall_occurred else None),
            predicted_next_rsu_id=predicted_next_rsu_id,
            predicted_handoff_target_rsu_id=predicted_handoff_target_rsu_id,
            hit_source=hit_source,
            cache_lookup_performed=bool(current_node and offload_mode == "rsu" and selected_target_rsu_id),
            cache_hit=bool(cache_hit),
            was_cached_before=bool(cache_result.get("was_cached_before", False)),
            admission_requested=admission_requested,
            admission_added=admission_added,
            admission_reason=admission_reason,
            cache_target_rsu_id=cache_result.get("target_rsu_id"),
            eviction_occurred=eviction_occurred,
            eviction_policy=(str(self._cache_capacity_profile.get("eviction_policy", "lru")) if capacity_enabled else "not_applicable"),
            evicted_object_id=evicted_object_id,
            evicted_adapter_id=evicted_adapter_id,
            eviction_reason="capacity_limit" if eviction_occurred else "not_occurred",
            adapter_transfer_size_mb=float(adapter_transfer_size),
            state_migration_size_mb=float(migration_transfer_size),
            transfer_source=transfer_source,
            migration_requested=migration_requested_flag,
            migration_realized=bool(migration_prepare_realized or (handoff_count > 0 and control.migration_action.get("mode") == "migrate")),
            cache_capacity_enabled=capacity_enabled,
            cache_capacity_unit=str(cache_result.get("cache_capacity_unit", "adapter_slots")),
            cache_capacity_before=capacity_before.get("cache_capacity") if capacity_enabled else None,
            cache_used_before=capacity_before.get("cache_used_size") if capacity_enabled else None,
            cache_remaining_before=capacity_before.get("cache_remaining_size") if capacity_enabled else None,
            cache_capacity_after=cache_result.get("cache_capacity") if capacity_enabled else None,
            cache_used_after=cache_result.get("cache_used_size") if capacity_enabled else None,
            cache_remaining_after=cache_result.get("cache_remaining_size") if capacity_enabled else None,
            action_id=control.metadata.get("action_id"),
            action_name=control.metadata.get("action_name"),
            cache_strategy=str(cache_result.get("strategy", "none")),
            offload_mode=offload_mode,
            service_success=bool(current_node and not stall_occurred),
            stall_occurred=bool(stall_occurred),
            handoff_event_count=int(handoff_count),
            eviction_count=(len(evicted_object_ids) if typed_mode else len(evicted_adapter_ids)),
            evicted_object_ids=evicted_object_ids,
            evicted_adapter_ids=evicted_adapter_ids,
            evicted_size_mb_sum=float(cache_result.get("evicted_size_mb_sum", 0.0) or 0.0),
            requested_object_size_mb=(size_mb if typed_mode else cache_result.get("requested_object_size_mb", size_mb)),
            capacity_rejection_reason=cache_result.get("capacity_rejection_reason"),
            admitted_object_id=(
                (
                    self.adapter_catalog.get_typed_adapter(
                        str(cache_result.get("adapter_id"))
                    ).object_id
                    if typed_mode
                    else next(
                        (
                            item.object_id
                            for item in self.adapter_catalog.cache_objects
                            if item.adapter_id == cache_result.get("adapter_id")
                        ),
                        f"adapter:{cache_result.get('adapter_id')}",
                    )
                )
                if admission_added and cache_result.get("adapter_id")
                else None
            ),
            admitted_adapter_id=(cache_result.get("adapter_id") if admission_added else None),
            admitted_size_mb=(
                float(size_mb)
                if admission_added and size_mb is not None
                else None
            ),
            evicted_sizes_mb=(
                [
                    float(self.adapter_catalog.get_typed_object(item).resident_size_mb)
                    for item in evicted_object_ids
                ]
                if typed_mode
                else [float(self._adapter_resident_size_mb(item)) for item in evicted_adapter_ids]
            ),
            typed_model_cache_contract_version=(
                TYPED_MODEL_CACHE_CONTRACT_VERSION if typed_mode else None
            ),
            model_cache_profile_id=self.adapter_catalog.model_cache_profile_id,
            requested_typed_objects=list(cache_result.get("requested_typed_objects") or []),
            dependency_bundle=cache_result.get("dependency_bundle"),
            per_object_lookup_results=list(readiness.get("per_object_lookup_results") or []),
            base_model_hit=(bool(readiness.get("base_ready")) if typed_mode else None),
            adapter_hit=(bool(readiness.get("adapter_ready")) if typed_mode else None),
            joint_model_hit=(bool(readiness.get("joint_base_adapter_hit")) if typed_mode else None),
            workflow_state_ready=(bool(readiness.get("state_ready")) if typed_mode else None),
            full_service_ready=(bool(readiness.get("full_service_ready")) if typed_mode else None),
            missing_object_types=list(readiness.get("missing_object_types") or []),
            incompatibility_reason=readiness.get("incompatibility_reason"),
            compatibility_result=readiness.get("compatibility_result"),
            admitted_typed_objects=list(cache_result.get("admitted_typed_objects") or []),
            evicted_typed_objects=list(cache_result.get("evicted_typed_objects") or []),
            admitted_mb_by_type=dict(cache_result.get("admitted_mb_by_type") or {}),
            evicted_mb_by_type=dict(cache_result.get("evicted_mb_by_type") or {}),
            transfer_mb_by_type=typed_transfer_by_type,
            typed_capacity_snapshot=(
                {
                    "before": capacity_before,
                    "after": {
                        "capacity_mb": cache_result.get("cache_capacity"),
                        "used_mb": cache_result.get("cache_used_size"),
                        "remaining_mb": cache_result.get("cache_remaining_size"),
                        "used_mb_by_type": cache_result.get("cache_used_mb_by_object_type"),
                    },
                    "requested_dependency_bundle_mb": (
                        (cache_result.get("dependency_bundle") or {}).get("requested_bundle_mb")
                    ),
                }
                if typed_mode
                else None
            ),
            atomic_transaction_status=(
                cache_result.get("atomic_transaction_status") if typed_mode else None
            ),
            orphan_count=(int(cache_result.get("orphan_count", 0)) if typed_mode else None),
        )

    def _estimate_backhaul_traffic_cost(
        self,
        adapter_id: str | None,
        cache_result: dict[str, Any],
        handoff_count: int,
        migration_mode: str,
        realized_prepare: dict[str, Any],
    ) -> float:
        cache_cost = 0.0
        if self._typed_mode_enabled():
            cache_cost = sum(
                float(value)
                for object_type, value in dict(
                    cache_result.get("transfer_mb_by_type") or {}
                ).items()
                if object_type in {"base_model", "adapter"}
            )
        elif cache_result.get("added_new_adapter", False):
            cache_cost = self.adapter_catalog.estimate_adapter_transfer_size_mb(
                cache_result.get("adapter_id") or adapter_id
            )
        migration_cost = 0.0
        if handoff_count > 0 and (migration_mode in {"prepare", "migrate"} or realized_prepare.get("realized", False)):
            if self._typed_mode_enabled():
                state_object = self.adapter_catalog.resolve_workflow_state_object()
                migration_cost = float(state_object.transfer_size_mb) if state_object else 0.0
            else:
                migration_cost = self.adapter_catalog.estimate_bundle_transfer_size_mb(adapter_id)
        return cache_cost + migration_cost

    def _extract_primary_vehicle_from_state(self, state: dict[str, Any]) -> dict[str, Any]:
        vehicles = state.get("vehicles", [])
        preferred_vehicle_id = state.get("primary_vehicle_id") or self._primary_vehicle_id
        if preferred_vehicle_id:
            for vehicle in vehicles:
                if vehicle.get("vehicle_id") == preferred_vehicle_id:
                    return vehicle
        return vehicles[0] if vehicles else {}

    def _default_cache_result(self) -> dict[str, Any]:
        return {
            "requested": False,
            "decision_target_rsu_id": None,
            "target_rsu_id": None,
            "adapter_id": None,
            "was_cached_before": False,
            "added_new_adapter": False,
            "strategy": "none",
            "prediction_driven": False,
            "cache_target_corrected_by_handoff": False,
            "cache_admission_added_new_adapter": False,
            "cache_eviction": False,
            "eviction_count": 0,
            "evicted_adapter_count": 0,
            "evicted_adapter_id": None,
            "evicted_adapter_ids": [],
            "evicted_object_ids": [],
            "dependency_bundle": None,
            "requested_typed_objects": [],
            "admitted_typed_objects": [],
            "evicted_typed_objects": [],
            "admitted_mb_by_type": {},
            "evicted_mb_by_type": {},
            "transfer_mb_by_type": {},
            "atomic_transaction_status": "not_requested",
            "orphan_count": 0,
            **self._cache_capacity_snapshot(None),
        }

    def _select_high_pressure_vehicle_id(
        self,
        candidate_vehicle_ids: set[str] | None = None,
    ) -> str | None:
        trajectory_frames = getattr(self._mobility_provider, "_trajectory_frames", [])
        if len(trajectory_frames) < 2:
            return None
        candidate_ids = {str(vehicle_id) for vehicle_id in candidate_vehicle_ids} if candidate_vehicle_ids else None
        physical_transfer_scores = self._score_physical_transfer_pressure(
            trajectory_frames=trajectory_frames,
            candidate_ids=candidate_ids,
        )
        if physical_transfer_scores:
            best_vehicle_id, best_score = max(sorted(physical_transfer_scores.items()), key=lambda item: item[1])
            if best_score > 0:
                return best_vehicle_id

        previous_associations = self._mapper.associate(self._frame_to_vehicle_states(trajectory_frames[0]))
        if candidate_ids is not None:
            previous_associations = {
                vehicle_id: rsu_id
                for vehicle_id, rsu_id in previous_associations.items()
                if vehicle_id in candidate_ids
            }
        handoff_counts = {
            vehicle_id: 0
            for vehicle_id in (candidate_ids if candidate_ids is not None else previous_associations)
        }
        for frame in trajectory_frames[1:]:
            current_associations = self._mapper.associate(self._frame_to_vehicle_states(frame))
            if candidate_ids is not None:
                current_associations = {
                    vehicle_id: rsu_id
                    for vehicle_id, rsu_id in current_associations.items()
                    if vehicle_id in candidate_ids
                }
            events = self._handoff_builder.build_events(
                previous_associations=previous_associations,
                current_associations=current_associations,
                time_index=int(frame.get("time_index", 0)),
            )
            for event in events:
                if candidate_ids is not None and event.vehicle_id not in candidate_ids:
                    continue
                handoff_counts.setdefault(event.vehicle_id, 0)
                if event.event_type == "handoff":
                    handoff_counts[event.vehicle_id] += 1
            previous_associations = current_associations
        if not handoff_counts:
            return None
        best_vehicle_id, best_count = max(sorted(handoff_counts.items()), key=lambda item: item[1])
        if best_count <= 0:
            return None
        return best_vehicle_id

    def _score_physical_transfer_pressure(
        self,
        *,
        trajectory_frames: list[dict[str, Any]],
        candidate_ids: set[str] | None,
        horizon: int = 16,
    ) -> dict[str, int]:
        vehicle_maps = [
            {vehicle.vehicle_id: vehicle for vehicle in self._frame_to_vehicle_states(frame)}
            for frame in trajectory_frames
        ]
        association_maps = [
            self._mapper.associate(list(vehicle_map.values()))
            for vehicle_map in vehicle_maps
        ]
        rsu_order_index = self._rsu_order_index()
        scores: dict[str, int] = {}
        max_horizon = max(1, int(horizon))
        for frame_index, vehicle_map in enumerate(vehicle_maps[:-1]):
            for vehicle_id, vehicle in vehicle_map.items():
                if candidate_ids is not None and vehicle_id not in candidate_ids:
                    continue
                current_rsu_id = association_maps[frame_index].get(vehicle_id)
                if current_rsu_id is None:
                    continue
                last_vehicle = vehicle
                for future_index in range(frame_index + 1, min(len(vehicle_maps), frame_index + max_horizon + 1)):
                    future_vehicle = vehicle_maps[future_index].get(vehicle_id)
                    if future_vehicle is None:
                        break
                    if not self._vehicle_step_is_physical(last_vehicle, future_vehicle):
                        break
                    future_rsu_id = association_maps[future_index].get(vehicle_id)
                    if future_rsu_id is not None and future_rsu_id != current_rsu_id:
                        if self._rsu_transition_is_adjacent(current_rsu_id, future_rsu_id, rsu_order_index):
                            scores[vehicle_id] = scores.get(vehicle_id, 0) + 1
                        break
                    last_vehicle = future_vehicle
        return scores

    def _rsu_order_index(self) -> dict[str, int]:
        if not self.rsu_states:
            return {}
        x_values = [float(rsu.position_x) for rsu in self.rsu_states]
        y_values = [float(rsu.position_y) for rsu in self.rsu_states]
        axis = "y" if (max(y_values) - min(y_values)) > (max(x_values) - min(x_values)) else "x"
        ordered_rsus = sorted(
            self.rsu_states,
            key=lambda rsu: (float(rsu.position_y), float(rsu.position_x))
            if axis == "y"
            else (float(rsu.position_x), float(rsu.position_y)),
        )
        return {rsu.rsu_id: index for index, rsu in enumerate(ordered_rsus)}

    def _rsu_transition_is_adjacent(
        self,
        current_rsu_id: str,
        future_rsu_id: str,
        rsu_order_index: dict[str, int],
    ) -> bool:
        if current_rsu_id not in rsu_order_index or future_rsu_id not in rsu_order_index:
            return False
        return abs(int(rsu_order_index[current_rsu_id]) - int(rsu_order_index[future_rsu_id])) <= 1

    def _vehicle_step_is_physical(
        self,
        previous_vehicle: VehicleState,
        current_vehicle: VehicleState,
    ) -> bool:
        displacement = math.dist(
            (float(previous_vehicle.position_x), float(previous_vehicle.position_y)),
            (float(current_vehicle.position_x), float(current_vehicle.position_y)),
        )
        max_speed = max(float(previous_vehicle.speed or 0.0), float(current_vehicle.speed or 0.0))
        return bool(displacement <= max(25.0, 0.5 * max_speed))

    def _frame_to_vehicle_states(self, frame: dict[str, Any]) -> list[VehicleState]:
        vehicles: list[VehicleState] = []
        for item in frame.get("vehicles", []):
            if isinstance(item, VehicleState):
                vehicles.append(
                    VehicleState(
                        vehicle_id=item.vehicle_id,
                        position_x=float(item.position_x),
                        position_y=float(item.position_y),
                        speed=float(item.speed),
                        base_model_id=item.base_model_id,
                        active_workflow_id=item.active_workflow_id,
                    )
                )
            else:
                vehicles.append(
                    VehicleState(
                        vehicle_id=str(item["vehicle_id"]),
                        position_x=float(item["position_x"]),
                        position_y=float(item["position_y"]),
                        speed=float(item["speed"]),
                        base_model_id=str(item["base_model_id"]),
                        active_workflow_id=item.get("active_workflow_id"),
                    )
                )
        return vehicles

    def _prepare_workflow_template(self, workflow_state: WorkflowGraphState) -> WorkflowGraphState:
        template = deepcopy(workflow_state)
        if self._mobility_source != "lust":
            return template
        for node in template.nodes:
            node.input_size = max(1, int(round(float(node.input_size) * self._lust_workflow_size_scale)))
            node.output_size = max(1, int(round(float(node.output_size) * self._lust_workflow_size_scale)))
        return template

    def _build_node_service_step_plan(self, workflow_state: WorkflowGraphState) -> dict[str, int]:
        if self._mobility_source != "lust":
            return {node.node_id: 1 for node in workflow_state.nodes}
        plan: dict[str, int] = {}
        for node in workflow_state.nodes:
            workload_units = max(int(node.input_size) + int(node.output_size), 1)
            base_steps = math.ceil(workload_units / self._lust_service_step_divisor)
            plan[node.node_id] = max(1, int(math.ceil(base_steps / max(self._lust_rsu_compute_scale, 1e-6))))
        return plan

    def _advance_current_node_service(self, current_node: WorkflowGraphState | Any) -> bool:
        if current_node is None:
            return False
        if self._mobility_source != "lust":
            self.workflow_state.mark_current_completed()
            return True
        node_id = current_node.node_id
        remaining_before = self._node_remaining_service_steps.get(
            node_id,
            self._node_service_steps.get(node_id, 1),
        )
        remaining_after = max(0, int(remaining_before) - 1)
        self._node_remaining_service_steps[node_id] = remaining_after
        if remaining_after == 0:
            self.workflow_state.mark_current_completed()
            return True
        return False

    def _get_current_node_service_steps_required(self, current_node: WorkflowGraphState | Any) -> int:
        if current_node is None:
            return 0
        return int(self._node_service_steps.get(current_node.node_id, 1))

    def _get_current_node_service_steps_remaining(self, current_node: WorkflowGraphState | Any) -> int:
        if current_node is None:
            return 0
        return int(
            self._node_remaining_service_steps.get(
                current_node.node_id,
                self._node_service_steps.get(current_node.node_id, 1),
            )
        )

    def _build_default_rsus(self) -> list[RSUState]:
        return [
            RSUState(rsu_id="rsu_a", position_x=0.0, position_y=0.0, coverage_radius=40.0),
            RSUState(rsu_id="rsu_b", position_x=60.0, position_y=0.0, coverage_radius=40.0),
            RSUState(rsu_id="rsu_c", position_x=110.0, position_y=0.0, coverage_radius=35.0),
        ]

    def _load_default_catalog(self) -> AdapterCatalog:
        catalog_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "model_catalog"
            / "sample_model_catalog.json"
        )
        return AdapterCatalog.from_json(catalog_path)


def make_toy_vec_env() -> VecWorkflowCoreEnv:
    """构造默认 toy 环境。"""
    return VecWorkflowCoreEnv()
