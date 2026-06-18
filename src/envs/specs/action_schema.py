"""Action schema and adapter boundary for VEC control agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.envs.specs.semantic_objects import ControlAction


@dataclass(frozen=True)
class DiscreteActionSpec:
    """One semantic discrete action exposed by the wrapper."""

    action_id: int
    name: str
    description: str


class ActionSchema:
    """Stable action contract shared by wrappers, agents, and benchmarks."""

    def __init__(self, actions: list[DiscreteActionSpec]) -> None:
        self._actions = tuple(actions)
        self._by_id = {action.action_id: action for action in self._actions}

    @classmethod
    def default_vec_workflow_schema(cls) -> "ActionSchema":
        return cls(
            [
                DiscreteActionSpec(
                    0,
                    "current_rsu_cache_fill",
                    "Cache the required adapter at the currently associated RSU.",
                ),
                DiscreteActionSpec(
                    1,
                    "predictive_next_rsu_prefetch",
                    "Prefetch the required adapter at the predicted next RSU.",
                ),
                DiscreteActionSpec(
                    2,
                    "vehicle_fallback",
                    "Run the current workflow node on the vehicle.",
                ),
                DiscreteActionSpec(
                    3,
                    "current_rsu_steady_offload",
                    "Offload to the currently associated RSU without cache mutation.",
                ),
                DiscreteActionSpec(
                    4,
                    "handoff_migration_prepare",
                    "Prepare adapter-state migration for the predicted handoff target.",
                ),
            ]
        )

    @property
    def discrete_action_count(self) -> int:
        return len(self._actions)

    @property
    def supports_continuous_control(self) -> bool:
        return False

    @property
    def continuous_control_blocker(self) -> str:
        return (
            "GymVecEnv currently exposes a five-way semantic discrete action contract; "
            "there is no natural low-level continuous or parameterized-continuous control "
            "surface for TD3/SAC/MADDPG without changing the action problem definition."
        )

    def action_name(self, action_id: int) -> str:
        return self._by_id.get(int(action_id), DiscreteActionSpec(-1, "unknown_action", "")).name

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "semantic_discrete",
            "discrete_action_count": self.discrete_action_count,
            "supports_continuous_control": self.supports_continuous_control,
            "continuous_control_blocker": self.continuous_control_blocker,
            "actions": [
                {
                    "action_id": action.action_id,
                    "name": action.name,
                    "description": action.description,
                }
                for action in self._actions
            ],
        }


class ActionMaskBuilder:
    """Builds masks without writing algorithm-specific tensor logic into env."""

    def __init__(self, schema: ActionSchema | None = None) -> None:
        self._schema = schema or ActionSchema.default_vec_workflow_schema()

    def build_mask_info(self, semantic_state: dict[str, Any] | None) -> dict[str, Any]:
        state = semantic_state or {}
        preconditions = _build_action_preconditions(state)
        mask = [False for _ in range(self._schema.discrete_action_count)]
        invalid_reasons: dict[str, str] = {}
        if not preconditions["has_workflow_node"]:
            return {
                "mask": mask,
                "invalid_reasons": {
                    str(action_id): "missing_current_workflow_node"
                    for action_id in range(self._schema.discrete_action_count)
                },
                "valid_action_count": 0,
                "semantic_preconditions": preconditions,
            }

        mask[0] = True
        mask[2] = True
        mask[3] = True
        if preconditions["distinct_predicted_next_rsu"]:
            if preconditions["target_adapter_ready"]:
                invalid_reasons["1"] = "target_adapter_already_ready"
            else:
                mask[1] = True
        else:
            invalid_reasons["1"] = "missing_distinct_predicted_next_rsu"

        if preconditions["distinct_handoff_target"]:
            mask[4] = True
        else:
            invalid_reasons["4"] = "missing_distinct_handoff_target"

        return {
            "mask": mask,
            "invalid_reasons": invalid_reasons,
            "valid_action_count": int(sum(1 for item in mask if item)),
            "semantic_preconditions": preconditions,
        }

    def build_mask(self, semantic_state: dict[str, Any] | None) -> list[bool]:
        return list(self.build_mask_info(semantic_state)["mask"])


class ActionAdapter:
    """Converts schema-level action ids into core ``ControlAction`` objects."""

    def __init__(self, schema: ActionSchema | None = None) -> None:
        self._schema = schema or ActionSchema.default_vec_workflow_schema()

    @property
    def schema(self) -> ActionSchema:
        return self._schema

    def decode(self, action: int, semantic_state: dict[str, Any] | None) -> ControlAction:
        state = semantic_state or {}
        current_node = state.get("current_workflow_node")
        action_id = int(action)
        metadata = self._base_metadata(action_id)
        if not current_node:
            return ControlAction(
                metadata={
                    **metadata,
                    "invalid_action": True,
                    "invalid_reason": "missing_current_workflow_node",
                }
            )

        preconditions = _build_action_preconditions(state)
        current_vehicle = preconditions["primary_vehicle"]
        vehicle_id = current_vehicle.get("vehicle_id")
        current_rsu_id = preconditions["current_rsu_id"]
        predicted_next_rsu_id = preconditions["predicted_next_rsu_id"]
        predicted_handoff_target_rsu_id = preconditions["predicted_handoff_target_rsu_id"]
        required_adapter = preconditions["required_adapter"]
        metadata = {
            **metadata,
            "invalid_action": False,
            "invalid_reason": "none",
            "primary_vehicle_id": vehicle_id,
            "current_rsu_id": current_rsu_id,
            "required_adapter": required_adapter,
        }

        if action_id == 0:
            return ControlAction(
                cache_action={
                    "operation": "cache",
                    "rsu_id": current_rsu_id,
                    "adapter_id": required_adapter,
                    "strategy": "reactive_cache_fill",
                    "prediction_driven": False,
                },
                offload_action={"mode": "rsu", "strategy": "current_rsu_cache_fill"},
                migration_action={"mode": "keep", "strategy": "none"},
                metadata=metadata,
            )

        if action_id == 1:
            if not preconditions["distinct_predicted_next_rsu"]:
                return self._invalid_noop_control(
                    metadata=metadata,
                    invalid_reason="missing_distinct_predicted_next_rsu",
                    offload_strategy="invalid_predictive_prefetch_noop",
                )
            if preconditions["target_adapter_ready"]:
                return self._invalid_noop_control(
                    metadata=metadata,
                    invalid_reason="target_adapter_already_ready",
                    offload_strategy="invalid_predictive_prefetch_noop",
                )
            return ControlAction(
                cache_action={
                    "operation": "cache",
                    "rsu_id": predicted_next_rsu_id,
                    "adapter_id": required_adapter,
                    "strategy": "predictive_prefetch",
                    "prediction_driven": True,
                    "prediction_source": "next_rsu_sequence:first_hop",
                },
                offload_action={"mode": "rsu", "strategy": "steady_current_execution"},
                migration_action={"mode": "keep", "strategy": "none"},
                metadata=metadata,
            )

        if action_id == 2:
            return ControlAction(
                cache_action={},
                offload_action={"mode": "vehicle", "strategy": "vehicle_fallback"},
                migration_action={"mode": "keep", "strategy": "none"},
                metadata=metadata,
            )

        if action_id == 3:
            return ControlAction(
                cache_action={},
                offload_action={"mode": "rsu", "strategy": "current_rsu_steady_offload"},
                migration_action={"mode": "keep", "strategy": "none"},
                metadata=metadata,
            )

        if action_id == 4:
            if not preconditions["distinct_handoff_target"]:
                return self._invalid_noop_control(
                    metadata=metadata,
                    invalid_reason="missing_distinct_handoff_target",
                    offload_strategy="invalid_handoff_prepare_noop",
                    migration_strategy="invalid_handoff_prepare",
                )
            return ControlAction(
                cache_action={},
                offload_action={"mode": "rsu", "strategy": "current_rsu_steady_offload"},
                migration_action={
                    "mode": "prepare",
                    "strategy": "handoff_migration_prepare",
                    "expected_target_rsu_id": predicted_handoff_target_rsu_id,
                },
                metadata=metadata,
            )

        return ControlAction(
            metadata={
                **metadata,
                "invalid_action": True,
                "invalid_reason": "unknown_action_id",
            }
        )

    def _base_metadata(self, action_id: int) -> dict[str, Any]:
        return {
            "action_id": int(action_id),
            "action_name": self._schema.action_name(int(action_id)),
            "action_contract": "semantic_discrete_5",
        }

    def _invalid_noop_control(
        self,
        *,
        metadata: dict[str, Any],
        invalid_reason: str,
        offload_strategy: str,
        migration_strategy: str = "none",
    ) -> ControlAction:
        return ControlAction(
            cache_action={},
            offload_action={"mode": "rsu", "strategy": offload_strategy},
            migration_action={"mode": "keep", "strategy": migration_strategy},
            metadata={
                **metadata,
                "invalid_action": True,
                "invalid_reason": invalid_reason,
            },
        )


def _primary_vehicle_from_state(state: dict[str, Any]) -> dict[str, Any]:
    vehicles = state.get("vehicles") or []
    primary_vehicle_id = state.get("primary_vehicle_id")
    if primary_vehicle_id is not None:
        for vehicle in vehicles:
            if str(vehicle.get("vehicle_id")) == str(primary_vehicle_id):
                return dict(vehicle)
    return dict(vehicles[0]) if vehicles else {}


def _build_action_preconditions(state: dict[str, Any]) -> dict[str, Any]:
    current_node = state.get("current_workflow_node") or {}
    primary_vehicle = _primary_vehicle_from_state(state)
    vehicle_id = primary_vehicle.get("vehicle_id")
    current_rsu_id = primary_vehicle.get("associated_rsu_id")
    required_adapter = current_node.get("required_adapter")
    predictions = state.get("predictions") or {}
    next_rsu_sequence = list(predictions.get("next_rsu_sequence", {}).get(vehicle_id, []))
    predicted_next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
    if predicted_next_rsu_id is None and next_rsu_sequence:
        predicted_next_rsu_id = next_rsu_sequence[0]
    if predicted_next_rsu_id is None or str(predicted_next_rsu_id) == str(current_rsu_id):
        for candidate_rsu_id in next_rsu_sequence:
            if candidate_rsu_id is not None and str(candidate_rsu_id) != str(current_rsu_id):
                predicted_next_rsu_id = candidate_rsu_id
                break
    predicted_handoff_target_rsu_id = (
        predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
        or predictions.get("predicted_handoff_target_rsu_id_by_vehicle", {}).get(vehicle_id)
    )
    distinct_predicted_next_rsu = bool(
        predicted_next_rsu_id is not None and str(predicted_next_rsu_id) != str(current_rsu_id)
    )
    distinct_handoff_target = bool(
        predicted_handoff_target_rsu_id is not None
        and str(predicted_handoff_target_rsu_id) != str(current_rsu_id)
    )
    target_adapter_ready = _rsu_has_adapter(
        state=state,
        rsu_id=predicted_next_rsu_id,
        adapter_id=required_adapter,
    )
    handoff_target_adapter_ready = _rsu_has_adapter(
        state=state,
        rsu_id=predicted_handoff_target_rsu_id,
        adapter_id=required_adapter,
    )
    return {
        "has_workflow_node": bool(state.get("current_workflow_node")),
        "primary_vehicle": primary_vehicle,
        "primary_vehicle_id": vehicle_id,
        "current_rsu_id": current_rsu_id,
        "required_adapter": required_adapter,
        "predicted_next_rsu_id": predicted_next_rsu_id,
        "predicted_handoff_target_rsu_id": predicted_handoff_target_rsu_id,
        "distinct_predicted_next_rsu": distinct_predicted_next_rsu,
        "distinct_handoff_target": distinct_handoff_target,
        "target_adapter_ready": target_adapter_ready,
        "handoff_target_adapter_ready": handoff_target_adapter_ready,
        "next_rsu_sequence_horizon": len(next_rsu_sequence),
    }


def _rsu_has_adapter(
    *,
    state: dict[str, Any],
    rsu_id: str | None,
    adapter_id: str | None,
) -> bool:
    if rsu_id is None or adapter_id is None:
        return False
    for rsu in state.get("rsus") or []:
        if str(rsu.get("rsu_id")) != str(rsu_id):
            continue
        cached = {str(item) for item in rsu.get("cached_adapter_ids", [])}
        return str(adapter_id) in cached
    return False
