"""Non-learning handoff-first rule used only for completion feasibility checks."""

from __future__ import annotations

from typing import Any

from src.agents.popularity_cache_heuristic_agent import PopularityCacheHeuristicAgent


class HandoffFirstFeasibilityAgent(PopularityCacheHeuristicAgent):
    """Prefer target adapter readiness, then state prepare, using visible predictions.

    This is a diagnostic fixed rule, not a paper baseline.  It consumes the same
    semantic state and action mask as the existing popularity heuristic and has
    no future-trace or outcome access.
    """

    support_level = "diagnostic"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.agent_name = "handoff_first_feasibility"

    def act(
        self,
        observation: Any,
        info: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        del observation
        semantic_state = self._semantic_state(info)
        current_node = semantic_state.get("current_workflow_node") or {}
        if not current_node:
            action = self._select_allowed([3, 2, 0], info)
            return action, self._action_info(action, "no_current_workflow_node")

        vehicle = self._primary_vehicle(semantic_state)
        vehicle_id = str(vehicle.get("vehicle_id", "")) if vehicle else None
        current_rsu_id = vehicle.get("associated_rsu_id")
        required_adapter = current_node.get("required_adapter")
        self._remember_adapter(required_adapter)
        predicted_next, predicted_handoff = self._prediction_targets(
            semantic_state, vehicle_id
        )
        target_rsu_id = predicted_handoff or predicted_next

        if current_rsu_id is None:
            action = self._select_allowed([2, 3, 0], info)
            return action, self._action_info(
                action, "no_associated_rsu_vehicle_fallback"
            )
        if target_rsu_id and target_rsu_id != current_rsu_id:
            if required_adapter and not self._adapter_cached(
                semantic_state, target_rsu_id, required_adapter
            ):
                action = self._select_allowed([1, 4, 3], info)
                return action, self._action_info(
                    action,
                    "handoff_target_adapter_first",
                    {"predicted_handoff_target": target_rsu_id},
                )
            action = self._select_allowed([4, 1, 3], info)
            return action, self._action_info(
                action,
                "handoff_state_prepare_after_target_ready",
                {"predicted_handoff_target": target_rsu_id},
            )
        if required_adapter and not self._adapter_cached(
            semantic_state, current_rsu_id, required_adapter
        ):
            action = self._select_allowed([0, 3, 2], info)
            return action, self._action_info(
                action, "reactive_current_rsu_cache_fill"
            )
        action = self._select_allowed([3, 0, 2], info)
        return action, self._action_info(action, "current_rsu_steady_offload")
