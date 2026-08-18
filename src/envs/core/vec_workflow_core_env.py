"""AI-driven VEC 工作流核心环境最小骨架。"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

from src.data.mobility.handoff_builder import HandoffBuilder
from src.data.mobility.replay_provider import ReplayProvider
from src.data.mobility.rsu_mapper import RSUMapper
from src.data.model_catalog.adapter_catalog import AdapterCatalog
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


PRIMARY_VEHICLE_SELECTION_CHOICES = {"stable_first", "handoff_pressure"}
CACHE_CAPACITY_UNITS = {"adapter_slots", "mb"}
CACHE_CAPACITY_EPSILON = 1.0e-9


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
            })
        return {"snapshot_step_index": int(self._episode_steps), "rsus": rsus}

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
        self._initialize_cache_capacity_metadata()

        self._mapper.update_rsus(self.rsu_states)
        vehicles = self._mobility_provider.reset()
        self._initialize_primary_vehicle_id(vehicles)
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
        current_required_adapter = current_node.required_adapter if current_node else None
        tracked_vehicle_id = primary_vehicle.vehicle_id if primary_vehicle else pre_action_vehicle_id
        offload_target_rsu_id = self._resolve_target_rsu_id(primary_vehicle, control)
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
        else:
            service_reward += 0.15
            base_model_ok = primary_vehicle.base_model_id == current_node.required_base_model
            if not base_model_ok:
                constraint_penalty += 1.0
            else:
                service_reward += 0.2

        if offload_target_rsu_id is None:
            constraint_penalty += 0.7
        else:
            service_reward += 0.1
            target_rsu = self._get_rsu_map().get(offload_target_rsu_id)
            cache_hit = bool(
                target_rsu
                and current_node.required_adapter in target_rsu.cached_adapter_ids
            )
            if cache_hit:
                service_reward += 0.45

        offload_mode = control.offload_action.get("mode", "rsu")
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
            if self._advance_current_node_service(current_node):
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
        return {
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
        return merged

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
        unique_adapters = list(dict.fromkeys(rsu.cached_adapter_ids))
        if self._cache_capacity_profile["unit"] == "mb":
            return float(sum(self._adapter_resident_size_mb(item) for item in unique_adapters))
        return len(unique_adapters)

    def _initialize_cache_capacity_metadata(self) -> None:
        self._eviction_policy.reset()
        for rsu in self.rsu_states:
            self._eviction_policy.reset(
                rsu_id=rsu.rsu_id,
                initial_resident_ids=list(rsu.cached_adapter_ids),
                current_step=self._episode_steps,
            )
            if self._cache_capacity_enabled():
                self._enforce_initial_cache_capacity(rsu)

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
        residents = list(dict.fromkeys(rsu.cached_adapter_ids))
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
        }

    def _apply_cache_action(
        self,
        control: ControlAction,
        primary_vehicle: VehicleState | None,
        current_node_id: str | None,
        required_adapter: str | None,
    ) -> dict[str, Any]:
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
            }
        next_sequence = list(predictions.get("next_rsu_sequence", {}).get(vehicle_id, []))
        predicted_handoff_target_rsu_id = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
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
        size_resolution = (
            self.adapter_catalog.resolve_adapter_resident_size_mb(adapter_id)
            if adapter_id else None
        )
        object_id = (
            size_resolution.object_id or f"adapter:{adapter_id}"
            if size_resolution else None
        )
        size_mb = size_resolution.size_mb if size_resolution else None
        offload_mode = str(control.offload_action.get("mode", "rsu"))
        if current_node is None:
            hit_source = "not_applicable"
        elif offload_mode == "vehicle":
            hit_source = "vehicle_local"
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
        capacity_before = dict(cache_result.get("capacity_before") or {})
        capacity_enabled = bool(cache_result.get("cache_capacity_enabled", False))
        adapter_transfer_size = size_mb if admission_added and size_mb is not None else 0.0
        migration_transfer_size = (
            float(self.adapter_catalog.estimate_bundle_transfer_size_mb(adapter_id))
            if adapter_id and (migration_prepare_requested or migration_prepare_realized)
            else 0.0
        )
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
            evicted_object_id=(evicted_object.object_id if evicted_object else (f"adapter:{evicted_adapter_id}" if evicted_adapter_id else None)),
            evicted_adapter_id=evicted_adapter_id,
            eviction_reason="capacity_limit" if eviction_occurred else "not_occurred",
            adapter_transfer_size_mb=float(adapter_transfer_size),
            state_migration_size_mb=float(migration_transfer_size),
            transfer_source=(size_resolution.source if size_resolution else "not_applicable"),
            migration_requested=bool(migration_prepare_requested or control.migration_action.get("mode") == "migrate"),
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
            eviction_count=len(evicted_adapter_ids),
            evicted_object_ids=evicted_object_ids,
            evicted_adapter_ids=evicted_adapter_ids,
            evicted_size_mb_sum=float(cache_result.get("evicted_size_mb_sum", 0.0) or 0.0),
            requested_object_size_mb=cache_result.get("requested_object_size_mb", size_mb),
            capacity_rejection_reason=cache_result.get("capacity_rejection_reason"),
            admitted_object_id=(
                next(
                    (
                        item.object_id
                        for item in self.adapter_catalog.cache_objects
                        if item.adapter_id == cache_result.get("adapter_id")
                    ),
                    f"adapter:{cache_result.get('adapter_id')}",
                )
                if admission_added and cache_result.get("adapter_id")
                else None
            ),
            admitted_adapter_id=(cache_result.get("adapter_id") if admission_added else None),
            admitted_size_mb=(
                float(cache_result.get("requested_object_size_mb"))
                if admission_added and cache_result.get("requested_object_size_mb") is not None
                else None
            ),
            evicted_sizes_mb=[float(self._adapter_resident_size_mb(item)) for item in evicted_adapter_ids],
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
        if cache_result.get("added_new_adapter", False):
            cache_cost = self.adapter_catalog.estimate_adapter_transfer_size_mb(
                cache_result.get("adapter_id") or adapter_id
            )
        migration_cost = 0.0
        if handoff_count > 0 and (migration_mode in {"prepare", "migrate"} or realized_prepare.get("realized", False)):
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
