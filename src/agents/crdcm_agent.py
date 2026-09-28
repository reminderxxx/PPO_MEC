"""Version-separated CRDCM policy adapters for SA-GHMAPPO, MAPPO, and PPO."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.agents.mappo_agent import MAPPOAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.sa_ghmappo_agent import SAGHMAPPOAgent
from src.encoders.crdcm_observation import (
    CRDCM_DECISION_CONTRACT_VERSION,
    CRDCM_FEATURE_NAMES,
    CRDCM_OBSERVATION_CONTRACT_VERSION,
    build_crdcm_feature_tensor,
)


CRDCM_CHECKPOINT_FORMAT_VERSION = "crdcm_checkpoint_v1"
CRDCM_CREDIT_CONTRACT_VERSION = "mask_external_override_actor_credit_v1"


def _tensor_values(value: Any) -> list[float]:
    if not isinstance(value, torch.Tensor):
        return []
    return [round(float(item), 6) for item in value.detach().flatten().tolist()]


def _softmax_values(value: Any) -> list[float]:
    if not isinstance(value, torch.Tensor):
        return []
    return [
        round(float(item), 6)
        for item in torch.softmax(value.detach(), dim=-1).flatten().tolist()
    ]


class _CRDCMPolicyMixin:
    """Add a trainable CRDCM residual without modifying protected legacy agents."""

    crdcm_agent_name = "crdcm_policy"
    crdcm_policy_type = "crdcm_policy_v1"
    observation_contract = CRDCM_OBSERVATION_CONTRACT_VERSION
    action_contract = "semantic_discrete_5_crdcm_decision_v1"
    support_level = "diagnostic_trainable"

    def __init__(
        self,
        *,
        crdcm_residual_scale: float = 0.35,
        crdcm_hidden_dim: int = 64,
        **kwargs: Any,
    ) -> None:
        random_seed = int(kwargs.get("random_seed", 7))
        super().__init__(**kwargs)
        self.agent_name = self.crdcm_agent_name
        self.policy_type = self.crdcm_policy_type
        self._crdcm_residual_scale = max(float(crdcm_residual_scale), 0.0)
        self._crdcm_hidden_dim = max(int(crdcm_hidden_dim), 8)
        output_dim = 6 if not self._use_hierarchy else 8
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(random_seed + 17041)
            self._crdcm_residual = nn.Sequential(
                nn.Linear(len(CRDCM_FEATURE_NAMES), self._crdcm_hidden_dim),
                nn.Tanh(),
                nn.Linear(self._crdcm_hidden_dim, output_dim),
            ).to(self._device)
        self._optimizer.add_param_group(
            {
                "params": list(self._crdcm_residual.parameters()),
                "lr": self._learning_rate,
            }
        )
        self._crdcm_last_forward_trace: dict[str, Any] = {}

    def _forward_policy(
        self,
        semantic_state: dict[str, Any],
        run_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        output = dict(super()._forward_policy(semantic_state, run_metadata=run_metadata))
        features = build_crdcm_feature_tensor(
            semantic_state,
            device=self._device,
            dtype=next(self._crdcm_residual.parameters()).dtype,
        )
        residual = self._crdcm_residual(features) * self._crdcm_residual_scale
        pre_logits: dict[str, list[float]] = {}
        post_logits: dict[str, list[float]] = {}
        post_probabilities: dict[str, list[float]] = {}
        if self._use_hierarchy:
            offset = 0
            for head_name, width in (("slow", 3), ("fast", 2), ("event", 2)):
                key = f"{head_name}_logits"
                pre_logits[head_name] = _tensor_values(output[key])
                output[key] = output[key] + residual[offset : offset + width]
                post_logits[head_name] = _tensor_values(output[key])
                post_probabilities[head_name] = _softmax_values(output[key])
                offset += width
            value_residual = residual[7]
        else:
            pre_logits["flat"] = _tensor_values(output["flat_logits"])
            output["flat_logits"] = output["flat_logits"] + residual[:5]
            post_logits["flat"] = _tensor_values(output["flat_logits"])
            post_probabilities["flat"] = _softmax_values(output["flat_logits"])
            value_residual = residual[5]
        output["value"] = output["value"] + value_residual
        output["crdcm_feature_vector"] = features
        output["crdcm_residual"] = residual
        self._crdcm_last_forward_trace = {
            "feature_vector": _tensor_values(features),
            "pre_crdcm_head_logits": pre_logits,
            "post_crdcm_head_logits": post_logits,
            "post_crdcm_head_probabilities": post_probabilities,
            "residual": _tensor_values(residual),
        }
        return output

    @staticmethod
    def _override_reasons(action_info: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        for key in (
            "continuity_guard",
            "cache_warm_start_guard",
            "predictive_prefetch_admission_guard",
            "backhaul_guard",
            "coverage_recovery_guard",
            "coverage_recovery_final_guard",
            "idle_popularity_fallback",
            "option_gate",
        ):
            detail = dict(action_info.get(key, {}) or {})
            if any(
                bool(detail.get(flag, False))
                for flag in ("guarded", "applied", "hard_override_applied", "override_triggered")
            ):
                reasons.append(f"{key}:{detail.get('reason', 'applied')}")
        if bool(action_info.get("guard_action_delta", False)) and not reasons:
            reasons.append("legacy_guard_chain:action_delta")
        return reasons

    def act(
        self,
        observation: Any,
        info: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        action, legacy_info = super().act(observation, info)
        action_info = dict(legacy_info)
        raw_action = int(action_info.get("raw_env_action", action))
        aggregated_action = int(action_info.get("projected_env_action", action))
        executed_action = int(action_info.get("final_env_action", action))
        override_reasons = self._override_reasons(action_info)
        override_applied = bool(
            action_info.get("guard_action_delta", False)
            or aggregated_action != executed_action
        )
        actor_credit_weight = 0.0 if override_applied else 1.0
        env_probs = list(action_info.get("env_action_probs", []) or [])
        raw_log_prob = None
        if 0 <= aggregated_action < len(env_probs) and env_probs[aggregated_action] > 0.0:
            raw_log_prob = round(math.log(float(env_probs[aggregated_action])), 6)
        decision_contract = {
            "contract_version": CRDCM_DECISION_CONTRACT_VERSION,
            "credit_contract_version": CRDCM_CREDIT_CONTRACT_VERSION,
            "observation_contract_version": CRDCM_OBSERVATION_CONTRACT_VERSION,
            "raw_selected_action": raw_action,
            "aggregated_policy_action": aggregated_action,
            "pre_override_action": aggregated_action,
            "post_override_action": executed_action,
            "actual_executed_action": executed_action,
            "override_applied": override_applied,
            "override_reasons": override_reasons,
            "raw_policy_log_prob": raw_log_prob,
            "executed_action_log_prob_under_policy": action_info.get("env_action_log_prob"),
            "actor_credit_weight": actor_credit_weight,
            "actor_credit_source": (
                "external_override_masked" if override_applied else "raw_policy_sample"
            ),
            "sampled_from_raw_policy_distribution": not override_applied,
            "structure_specific_encoder": self._encoder_kind,
            **dict(self._crdcm_last_forward_trace),
        }
        action_info["crdcm_decision_contract"] = decision_contract
        action_info["decision_contract_version"] = CRDCM_DECISION_CONTRACT_VERSION
        if override_applied and self._use_hierarchy:
            action_info["head_credit_weights"] = {
                head_name: 0.0 for head_name in ("slow", "fast", "event")
            }
        return action, action_info

    def learn(self, rollout: list[dict[str, Any]]) -> dict[str, Any]:
        eligible = [
            row
            for row in rollout
            if float(
                row.get("action_info", {})
                .get("crdcm_decision_contract", {})
                .get("actor_credit_weight", 1.0)
            )
            > 0.0
        ]
        masked_count = len(rollout) - len(eligible)
        if not eligible:
            return {
                "agent_name": self.agent_name,
                "policy_type": self.policy_type,
                "policy_update_skipped": True,
                "reason": "all_actions_externally_overridden",
                "credit_contract_version": CRDCM_CREDIT_CONTRACT_VERSION,
                "external_override_masked_count": masked_count,
                "update_count": self._update_count,
            }
        stats = dict(super().learn(eligible))
        stats["credit_contract_version"] = CRDCM_CREDIT_CONTRACT_VERSION
        stats["external_override_masked_count"] = masked_count
        stats["actor_credit_eligible_count"] = len(eligible)
        return stats

    def _checkpoint_config(self) -> dict[str, Any]:
        config = dict(super()._checkpoint_config())
        config.update(
            {
                "agent_name": self.agent_name,
                "policy_type": self.policy_type,
                "crdcm_checkpoint_format_version": CRDCM_CHECKPOINT_FORMAT_VERSION,
                "crdcm_observation_contract_version": CRDCM_OBSERVATION_CONTRACT_VERSION,
                "crdcm_decision_contract_version": CRDCM_DECISION_CONTRACT_VERSION,
                "crdcm_credit_contract_version": CRDCM_CREDIT_CONTRACT_VERSION,
                "crdcm_residual_scale": self._crdcm_residual_scale,
                "crdcm_hidden_dim": self._crdcm_hidden_dim,
            }
        )
        return config

    def save(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "agent_name": self.agent_name,
                "policy_type": self.policy_type,
                "update_count": self._update_count,
                "config": self._checkpoint_config(),
                "network_state_dict": self._network.state_dict(),
                "crdcm_residual_state_dict": self._crdcm_residual.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "learned_transition_model_state": (
                    self._learned_transition_model.state_dict()
                    if self._learned_transition_model is not None
                    else None
                ),
            },
            output_path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(Path(path), map_location=self._device)
        config = dict(checkpoint.get("config", {}) or {})
        if config.get("crdcm_checkpoint_format_version") != CRDCM_CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                "legacy/non-CRDCM checkpoint is incompatible with crdcm_decision_v1; "
                "training must start as a new version"
            )
        if config.get("crdcm_decision_contract_version") != CRDCM_DECISION_CONTRACT_VERSION:
            raise ValueError("CRDCM decision contract mismatch")
        self._network.load_state_dict(checkpoint["network_state_dict"], strict=True)
        self._crdcm_residual.load_state_dict(
            checkpoint["crdcm_residual_state_dict"], strict=True
        )
        if checkpoint.get("optimizer_state_dict") is not None:
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        learned_state = checkpoint.get("learned_transition_model_state")
        if self._learned_transition_model is not None and isinstance(learned_state, dict):
            self._learned_transition_model.load_state_dict(learned_state)
        self._update_count = int(checkpoint.get("update_count", 0))


class CRDCMPPOAgent(_CRDCMPolicyMixin, PPOAgent):
    crdcm_agent_name = "crdcm_ppo"
    crdcm_policy_type = "crdcm_ppo_policy_v1"


class CRDCMMAPPOAgent(_CRDCMPolicyMixin, MAPPOAgent):
    crdcm_agent_name = "crdcm_mappo"
    crdcm_policy_type = "crdcm_mappo_policy_v1"


class CRDCMSAGHMAPPOAgent(_CRDCMPolicyMixin, SAGHMAPPOAgent):
    crdcm_agent_name = "crdcm_sa_ghmappo"
    crdcm_policy_type = "crdcm_sa_ghmappo_policy_v1"


__all__ = [
    "CRDCM_CHECKPOINT_FORMAT_VERSION",
    "CRDCM_CREDIT_CONTRACT_VERSION",
    "CRDCMMAPPOAgent",
    "CRDCMPPOAgent",
    "CRDCMSAGHMAPPOAgent",
]
