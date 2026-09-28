"""Fair dependency/critical-path/reuse-aware CRDCM heuristic baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agents.base_agent import BaseAgent
from src.encoders.crdcm_observation import (
    CRDCM_DECISION_CONTRACT_VERSION,
    CRDCM_OBSERVATION_CONTRACT_VERSION,
    validate_crdcm_observation,
)


class CRDCMCriticalPathHeuristicAgent(BaseAgent):
    """Strong transparent controller over the same actor-visible CRDCM state."""

    observation_contract = CRDCM_OBSERVATION_CONTRACT_VERSION
    action_contract = "semantic_discrete_5_crdcm_decision_v1"
    support_level = "heuristic"

    def __init__(self, **kwargs: Any) -> None:
        del kwargs
        super().__init__(agent_name="crdcm_critical_path_heuristic")

    def act(
        self,
        observation: Any,
        info: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        del observation
        semantic_state = dict((info or {}).get("semantic_state", {}) or {})
        payload = dict(semantic_state.get("crdcm_observation", {}) or {})
        validate_crdcm_observation(payload)
        mask = list((info or {}).get("action_mask", [True] * 5))
        current = dict(payload.get("current_rsu", {}) or {})
        target = dict(payload.get("predicted_target_rsu", {}) or {})
        current_bundle = dict(current.get("required_bundle", {}) or {})
        target_bundle = dict(target.get("required_bundle", {}) or {})
        dag = dict(payload.get("remaining_dag", {}) or {})
        prediction = dict(payload.get("handoff_prediction", {}) or {})
        migration = dict(payload.get("migration", {}) or {})

        current_ready = bool(
            current_bundle.get("base_ready", False)
            and current_bundle.get("adapter_ready", False)
        )
        target_ready = bool(
            target_bundle.get("base_ready", False)
            and target_bundle.get("adapter_ready", False)
        )
        target_available = bool(prediction.get("target_available", False))
        confidence = float(prediction.get("confidence", 0.0) or 0.0)
        reuse_count = int(dag.get("current_adapter_remaining_reuse_count", 0) or 0)
        critical_path = int(dag.get("critical_path_length", 0) or 0)
        near_handoff = int(prediction.get("eta_steps", 999) or 999) <= 4
        capacity_conflict = bool(migration.get("capacity_conflict", False))
        state_missing = bool(
            migration.get("state_required", False)
            and not migration.get("state_ready", False)
        )

        if not current_ready and mask[0]:
            action, reason = 0, "current_required_bundle_missing"
        elif (
            target_available
            and not target_ready
            and confidence >= 0.55
            and near_handoff
            and not capacity_conflict
            and (reuse_count >= 2 or critical_path >= 4)
        ):
            if state_missing and bool(migration.get("enabled", False)) and mask[4]:
                action, reason = 4, "cross_rsu_bundle_and_state_prepare"
            elif mask[1]:
                action, reason = 1, "critical_path_reuse_prefetch"
            else:
                action, reason = 3, "prefetch_masked"
        elif (
            target_available
            and state_missing
            and confidence >= 0.7
            and near_handoff
            and bool(migration.get("enabled", False))
            and not capacity_conflict
            and (reuse_count >= 2 or critical_path >= 4)
            and mask[4]
        ):
            action, reason = 4, "state_migration_readiness_prepare"
        else:
            action, reason = 3, "steady_current_rsu_negative_control"

        if action >= len(mask) or not mask[action]:
            valid = [index for index, allowed in enumerate(mask[:5]) if allowed]
            action = 3 if 3 in valid else valid[0]
            reason = f"mask_fallback:{reason}"
        decision_contract = {
            "contract_version": CRDCM_DECISION_CONTRACT_VERSION,
            "observation_contract_version": CRDCM_OBSERVATION_CONTRACT_VERSION,
            "raw_selected_action": action,
            "aggregated_policy_action": action,
            "pre_override_action": action,
            "post_override_action": action,
            "actual_executed_action": action,
            "override_applied": False,
            "override_reasons": [],
            "raw_policy_log_prob": None,
            "executed_action_log_prob_under_policy": None,
            "actor_credit_weight": 0.0,
            "actor_credit_source": "non_learning_heuristic",
            "sampled_from_raw_policy_distribution": False,
            "heuristic_reason": reason,
        }
        return action, {
            "policy_type": "crdcm_critical_path_reuse_heuristic_v1",
            "policy_mode": "deterministic",
            "action_mask": mask,
            "raw_env_action": action,
            "projected_env_action": action,
            "final_env_action": action,
            "aggregation_reason": reason,
            "log_prob": 0.0,
            "value": 0.0,
            "crdcm_decision_contract": decision_contract,
            "decision_contract_version": CRDCM_DECISION_CONTRACT_VERSION,
        }

    def learn(self, rollout: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "policy_update_skipped": True,
            "reason": "non_learning_heuristic",
            "rollout_steps": len(rollout),
        }

    def save(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "agent_name": self.agent_name,
                    "observation_contract": self.observation_contract,
                    "action_contract": self.action_contract,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, path: str) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("observation_contract") != self.observation_contract:
            raise ValueError("CRDCM heuristic observation contract mismatch")

    def evaluate_value(self, observation: Any, info: dict[str, Any] | None = None) -> float:
        del observation, info
        return 0.0


__all__ = ["CRDCMCriticalPathHeuristicAgent"]
