"""Popularity-aware cache heuristic over the shared semantic action contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agents.base_agent import BaseAgent


class PopularityCacheHeuristicAgent(BaseAgent):
    """Popularity-aware cache heuristic with simple prediction-aware offloading."""

    support_level = "heuristic"
    observation_contract = "semantic_state_info_v1"
    action_contract = "semantic_discrete_5"

    def __init__(self, popularity_prefetch_threshold: int = 2, **kwargs: Any) -> None:
        super().__init__(agent_name="popularity_cache_heuristic")
        self.config = {"popularity_prefetch_threshold": popularity_prefetch_threshold, **dict(kwargs)}
        self._adapter_counts: dict[str, int] = {}
        self._popularity_prefetch_threshold = max(int(popularity_prefetch_threshold), 1)

    def act(self, observation: Any, info: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
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
        adapter_seen_count = self._remember_adapter(required_adapter)
        predicted_next_rsu_id, predicted_handoff_target = self._prediction_targets(semantic_state, vehicle_id)

        if current_rsu_id is None:
            action = self._select_allowed([2, 3, 0], info)
            return action, self._action_info(action, "no_associated_rsu_vehicle_fallback")

        if required_adapter and not self._adapter_cached(semantic_state, current_rsu_id, required_adapter):
            action = self._select_allowed([0, 3, 2], info)
            return action, self._action_info(
                action,
                "popular_adapter_reactive_cache_fill",
                {"adapter_seen_count": adapter_seen_count},
            )

        if (
            adapter_seen_count >= self._popularity_prefetch_threshold
            and predicted_next_rsu_id
            and predicted_next_rsu_id != current_rsu_id
            and not self._adapter_cached(semantic_state, predicted_next_rsu_id, required_adapter)
        ):
            action = self._select_allowed([1, 3, 4], info)
            return action, self._action_info(
                action,
                "popular_adapter_predictive_prefetch",
                {"adapter_seen_count": adapter_seen_count, "predicted_next_rsu_id": predicted_next_rsu_id},
            )

        if predicted_handoff_target and predicted_handoff_target != current_rsu_id:
            action = self._select_allowed([4, 3, 1], info)
            return action, self._action_info(
                action,
                "predicted_handoff_migration_prepare",
                {"adapter_seen_count": adapter_seen_count, "predicted_handoff_target": predicted_handoff_target},
            )

        action = self._select_allowed([3, 0, 2], info)
        return action, self._action_info(
            action,
            "popularity_steady_offload",
            {"adapter_seen_count": adapter_seen_count},
        )

    def learn(self, rollout: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "policy_update_skipped": True,
            "support_level": self.support_level,
            "reason": "non_learning_heuristic",
            "collected_steps": len(rollout),
        }

    def save(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "agent_name": self.agent_name,
                    "support_level": self.support_level,
                    "config": self.config,
                    "adapter_counts": self._adapter_counts,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, path: str) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.config.update(dict(payload.get("config", {})))
        self._adapter_counts = {
            str(adapter_id): int(count)
            for adapter_id, count in dict(payload.get("adapter_counts", {})).items()
        }
        threshold = self.config.get("popularity_prefetch_threshold", self._popularity_prefetch_threshold)
        self._popularity_prefetch_threshold = max(int(threshold), 1)

    def evaluate_value(self, observation: Any, info: dict[str, Any] | None = None) -> float:
        del observation, info
        return 0.0

    def _semantic_state(self, info: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(info, dict):
            return {}
        semantic_state = info.get("semantic_state")
        return dict(semantic_state) if isinstance(semantic_state, dict) else {}

    def _action_mask(self, info: dict[str, Any] | None) -> list[bool]:
        if not isinstance(info, dict):
            return [True, True, True, True, True]
        mask = info.get("action_mask")
        if isinstance(mask, list) and len(mask) >= 5:
            return [bool(item) for item in mask[:5]]
        return [True, True, True, True, True]

    def _select_allowed(self, preferred_actions: list[int], info: dict[str, Any] | None) -> int:
        mask = self._action_mask(info)
        for action_id in preferred_actions:
            if 0 <= int(action_id) < len(mask) and mask[int(action_id)]:
                return int(action_id)
        for action_id, allowed in enumerate(mask):
            if allowed:
                return int(action_id)
        return 3

    def _primary_vehicle(self, semantic_state: dict[str, Any]) -> dict[str, Any]:
        vehicles = list(semantic_state.get("vehicles", []))
        if not vehicles:
            return {}
        primary_vehicle_id = semantic_state.get("primary_vehicle_id")
        if primary_vehicle_id:
            for vehicle in vehicles:
                if str(vehicle.get("vehicle_id", "")) == str(primary_vehicle_id):
                    return dict(vehicle)
        return dict(vehicles[0])

    def _rsu_by_id(self, semantic_state: dict[str, Any], rsu_id: str | None) -> dict[str, Any]:
        if rsu_id is None:
            return {}
        for rsu in semantic_state.get("rsus", []):
            if str(rsu.get("rsu_id", "")) == str(rsu_id):
                return dict(rsu)
        return {}

    def _adapter_cached(self, semantic_state: dict[str, Any], rsu_id: str | None, adapter_id: str | None) -> bool:
        if not rsu_id or not adapter_id:
            return False
        rsu = self._rsu_by_id(semantic_state, rsu_id)
        return str(adapter_id) in {str(item) for item in rsu.get("cached_adapter_ids", [])}

    def _prediction_targets(
        self,
        semantic_state: dict[str, Any],
        vehicle_id: str | None,
    ) -> tuple[str | None, str | None]:
        predictions = semantic_state.get("predictions", {})
        if not vehicle_id or not isinstance(predictions, dict):
            return None, None
        next_rsu_id = predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id)
        next_sequence = predictions.get("next_rsu_sequence", {}).get(vehicle_id, [])
        if next_rsu_id is None and next_sequence:
            next_rsu_id = next_sequence[0]
        handoff_target = predictions.get("predicted_first_handoff_rsu_by_vehicle", {}).get(vehicle_id)
        return next_rsu_id, handoff_target

    def _remember_adapter(self, adapter_id: str | None) -> int:
        if not adapter_id:
            return 0
        adapter_key = str(adapter_id)
        self._adapter_counts[adapter_key] = self._adapter_counts.get(adapter_key, 0) + 1
        return self._adapter_counts[adapter_key]

    def _action_info(self, action: int, reason: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "value": 0.0,
            "log_prob": 0.0,
            "policy_type": self.agent_name,
            "heuristic_reason": reason,
            "support_level": self.support_level,
            **dict(extra or {}),
        }
