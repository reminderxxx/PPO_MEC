"""Reactive greedy heuristic over the shared semantic action contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agents.base_agent import BaseAgent


class ReactiveGreedyAgent(BaseAgent):
    """Reactive cache-fill and offload heuristic without predictive prefetch."""

    support_level = "heuristic"
    observation_contract = "semantic_state_info_v1"
    action_contract = "semantic_discrete_5"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(agent_name="reactive_greedy")
        self.config = dict(kwargs)
        self._adapter_counts: dict[str, int] = {}

    def act(self, observation: Any, info: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        del observation
        semantic_state = self._semantic_state(info)
        current_node = semantic_state.get("current_workflow_node") or {}
        if not current_node:
            action = self._select_allowed([3, 2, 0], info)
            return action, self._action_info(action, "no_current_workflow_node")

        vehicle = self._primary_vehicle(semantic_state)
        current_rsu_id = vehicle.get("associated_rsu_id")
        required_adapter = current_node.get("required_adapter")
        self._remember_adapter(required_adapter)

        if current_rsu_id is None:
            action = self._select_allowed([2, 3, 0], info)
            return action, self._action_info(action, "no_associated_rsu_vehicle_fallback")
        if required_adapter and not self._adapter_cached(semantic_state, current_rsu_id, required_adapter):
            action = self._select_allowed([0, 3, 2], info)
            return action, self._action_info(action, "reactive_current_rsu_cache_fill")
        action = self._select_allowed([3, 0, 2], info)
        return action, self._action_info(action, "current_rsu_steady_offload")

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

    def _remember_adapter(self, adapter_id: str | None) -> int:
        if not adapter_id:
            return 0
        adapter_key = str(adapter_id)
        self._adapter_counts[adapter_key] = self._adapter_counts.get(adapter_key, 0) + 1
        return self._adapter_counts[adapter_key]

    def _action_info(self, action: int, reason: str) -> dict[str, Any]:
        return {
            "value": 0.0,
            "log_prob": 0.0,
            "policy_type": self.agent_name,
            "heuristic_reason": reason,
            "support_level": self.support_level,
        }
