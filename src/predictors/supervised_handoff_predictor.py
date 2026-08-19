"""Thin supervised handoff predictor for short-horizon RSU anticipation."""

from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.envs.specs import RSUState, VehicleState, WorkflowGraphState


FEATURE_SCHEMA_VERSION = "supervised_handoff_feature_v1"
CHECKPOINT_SCHEMA_VERSION = "supervised_handoff_predictor_checkpoint_v1"
NORMALIZATION_VERSION = "supervised_handoff_fixed_clamp_v1"


def feature_names_for_rsus(rsu_ids: list[str]) -> list[str]:
    """Return the semantic feature order frozen by feature schema v1."""
    names = [
        "vehicle.position_x_scaled",
        "vehicle.position_y_scaled",
        "vehicle.speed_scaled",
        "vehicle.delta_x_scaled",
        "vehicle.delta_y_scaled",
        "vehicle.current_rsu_available",
        "workflow.current_node_available",
        "workflow.progress",
        "current_rsu.required_adapter_ready",
        "workflow.remaining_nodes_scaled",
    ]
    for rsu_id in rsu_ids:
        names.extend(
            [
                f"rsu.{rsu_id}.relative_x_scaled",
                f"rsu.{rsu_id}.relative_y_scaled",
                f"rsu.{rsu_id}.distance_scaled",
                f"rsu.{rsu_id}.is_current",
                f"rsu.{rsu_id}.active_vehicle_count_scaled",
                f"rsu.{rsu_id}.coverage_radius_scaled",
                f"rsu.{rsu_id}.required_adapter_ready",
            ]
        )
    return names


def feature_order_sha256(feature_names: list[str]) -> str:
    return hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(float(value), upper))


def _current_node(workflow_state: WorkflowGraphState | None) -> Any:
    if workflow_state is None:
        return None
    if hasattr(workflow_state, "current_node"):
        return workflow_state.current_node()
    return None


def build_feature_vector(
    *,
    vehicle: VehicleState,
    rsu_states: list[RSUState],
    workflow_state: WorkflowGraphState | None,
    current_associations: dict[str, str | None],
    rsu_ids: list[str],
    last_vehicle_positions: dict[str, tuple[float, float]] | None = None,
) -> list[float]:
    """Build the frozen v1 feature vector for one vehicle.

    The feature schema intentionally uses only mobility, RSU, workflow, and cache
    state available before the controller acts. It does not consume reward,
    action, checkpoint outcome, or benchmark labels.
    """
    last_positions = last_vehicle_positions or {}
    current_rsu_id = current_associations.get(vehicle.vehicle_id)
    current_node = _current_node(workflow_state)
    required_adapter = getattr(current_node, "required_adapter", None) if current_node else None
    execution_order = list(getattr(workflow_state, "execution_order", []) or []) if workflow_state else []
    completed_node_ids = list(getattr(workflow_state, "completed_node_ids", []) or []) if workflow_state else []
    progress = float(len(completed_node_ids)) / float(max(len(execution_order), 1))
    remaining_nodes = max(len(execution_order) - len(completed_node_ids), 0)
    previous_position = last_positions.get(vehicle.vehicle_id)
    if previous_position is None:
        delta_x = 0.0
        delta_y = 0.0
    else:
        delta_x = float(vehicle.position_x) - float(previous_position[0])
        delta_y = float(vehicle.position_y) - float(previous_position[1])

    rsu_map = {str(rsu.rsu_id): rsu for rsu in rsu_states}
    current_rsu = rsu_map.get(str(current_rsu_id)) if current_rsu_id is not None else None
    current_cache_ready = bool(
        current_rsu
        and required_adapter
        and required_adapter in list(getattr(current_rsu, "cached_adapter_ids", []) or [])
    )
    features = [
        _clamp(float(vehicle.position_x) / 1000.0, -10.0, 10.0),
        _clamp(float(vehicle.position_y) / 1000.0, -10.0, 10.0),
        _clamp(float(vehicle.speed) / 50.0, 0.0, 5.0),
        _clamp(delta_x / 100.0, -5.0, 5.0),
        _clamp(delta_y / 100.0, -5.0, 5.0),
        1.0 if current_rsu_id is not None else 0.0,
        1.0 if current_node is not None else 0.0,
        _clamp(progress, 0.0, 1.0),
        1.0 if current_cache_ready else 0.0,
        _clamp(float(remaining_nodes) / 50.0, 0.0, 5.0),
    ]
    for rsu_id in rsu_ids:
        rsu = rsu_map.get(str(rsu_id))
        if rsu is None:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        dx = float(rsu.position_x) - float(vehicle.position_x)
        dy = float(rsu.position_y) - float(vehicle.position_y)
        distance = math.sqrt(dx * dx + dy * dy)
        cached_adapter_ids = list(getattr(rsu, "cached_adapter_ids", []) or [])
        active_vehicle_ids = list(getattr(rsu, "active_vehicle_ids", []) or [])
        features.extend(
            [
                _clamp(dx / 1000.0, -10.0, 10.0),
                _clamp(dy / 1000.0, -10.0, 10.0),
                _clamp(distance / 1000.0, 0.0, 10.0),
                1.0 if current_rsu_id is not None and str(current_rsu_id) == str(rsu_id) else 0.0,
                _clamp(float(len(active_vehicle_ids)) / 20.0, 0.0, 5.0),
                _clamp(float(rsu.coverage_radius) / 1000.0, 0.0, 10.0),
                1.0 if required_adapter and required_adapter in cached_adapter_ids else 0.0,
            ]
        )
    return features


class SupervisedHandoffPredictorNetwork(nn.Module):
    """Small MLP with classification and ETA heads."""

    def __init__(self, input_dim: int, rsu_class_count: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.rsu_class_count = int(rsu_class_count)
        self.hidden_dim = int(hidden_dim)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.next_rsu_head = nn.Linear(self.hidden_dim, self.rsu_class_count)
        self.handoff_target_head = nn.Linear(self.hidden_dim, self.rsu_class_count)
        self.handoff_logit_head = nn.Linear(self.hidden_dim, 1)
        self.eta_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(features)
        return {
            "next_rsu_logits": self.next_rsu_head(hidden),
            "handoff_target_logits": self.handoff_target_head(hidden),
            "handoff_logit": self.handoff_logit_head(hidden).squeeze(-1),
            "eta_steps": torch.nn.functional.softplus(self.eta_head(hidden).squeeze(-1)) + 1.0,
        }


class SupervisedHandoffPredictorRuntime:
    """Validated runtime wrapper around a supervised handoff predictor checkpoint."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"supervised predictor checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"invalid supervised predictor checkpoint payload: {path}")
        schema_version = str(payload.get("checkpoint_schema_version", ""))
        if schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported supervised predictor checkpoint schema {schema_version}; "
                f"expected {CHECKPOINT_SCHEMA_VERSION}"
            )
        feature_schema = dict(payload.get("feature_schema", {}))
        if feature_schema.get("schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported supervised predictor feature schema {feature_schema.get('schema_version')}; "
                f"expected {FEATURE_SCHEMA_VERSION}"
            )
        rsu_label_map = dict(payload.get("rsu_label_map", {}))
        rsu_ids = [str(item) for item in rsu_label_map.get("rsu_ids", [])]
        none_index = int(rsu_label_map.get("none_index", len(rsu_ids)))
        feature_names = list(feature_schema.get("feature_names", []))
        input_dim = int(payload.get("input_dim", len(feature_names)))
        expected_dim = 10 + 7 * len(rsu_ids)
        if input_dim != expected_dim:
            raise ValueError(f"checkpoint input_dim={input_dim} does not match expected feature dim {expected_dim}")
        if len(feature_names) != input_dim:
            raise ValueError("checkpoint feature_names length does not match input_dim")
        computed_feature_order_sha256 = feature_order_sha256(feature_names)
        declared_feature_order_sha256 = feature_schema.get("feature_order_sha256")
        if (
            declared_feature_order_sha256 is not None
            and str(declared_feature_order_sha256) != computed_feature_order_sha256
        ):
            raise ValueError("checkpoint feature_order_sha256 does not match feature_names")
        normalization_version = str(
            feature_schema.get("normalization_version", NORMALIZATION_VERSION)
        )
        if normalization_version != NORMALIZATION_VERSION:
            raise ValueError(
                f"unsupported predictor normalization version: {normalization_version}"
            )
        hidden_dim = int(payload.get("hidden_dim", 64))
        rsu_class_count = len(rsu_ids) + 1
        network = SupervisedHandoffPredictorNetwork(
            input_dim=input_dim,
            rsu_class_count=rsu_class_count,
            hidden_dim=hidden_dim,
        )
        network.load_state_dict(payload["model_state_dict"])
        network.eval()
        self.checkpoint_path = str(path)
        self.payload_metadata = {
            "checkpoint_path": str(path),
            "checkpoint_schema_version": schema_version,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "horizon": int(payload.get("horizon", 3)),
            "run_id": str(payload.get("run_id", "unknown")),
            "calibration": dict(payload.get("calibration", {})),
            "metrics": dict(payload.get("metrics", {})),
            "feature_names": feature_names,
            "feature_order_sha256": computed_feature_order_sha256,
            "normalization_version": normalization_version,
            "rsu_ids": list(rsu_ids),
        }
        self._network = network
        self._rsu_ids = rsu_ids
        self._none_index = none_index
        self._horizon = max(int(payload.get("horizon", 3)), 1)
        calibration = dict(payload.get("calibration", {}))
        self._target_selection_mode = str(
            calibration.get("target_selection_mode", "hard")
        ).strip().lower()
        if self._target_selection_mode not in {"hard", "soft"}:
            raise ValueError(
                "unsupported supervised predictor target_selection_mode "
                f"{self._target_selection_mode!r}; expected 'hard' or 'soft'"
            )
        self._handoff_threshold = _clamp(
            float(calibration.get("handoff_decision_threshold", 0.5)),
            0.0,
            1.0,
        )

    @property
    def rsu_ids(self) -> list[str]:
        return list(self._rsu_ids)

    @property
    def horizon(self) -> int:
        return self._horizon

    def _index_to_rsu(self, index: int) -> str | None:
        if int(index) == self._none_index:
            return None
        if 0 <= int(index) < len(self._rsu_ids):
            return self._rsu_ids[int(index)]
        return None

    def predict(
        self,
        *,
        vehicles: list[VehicleState],
        rsu_states: list[RSUState],
        workflow_state: WorkflowGraphState | None,
        current_associations: dict[str, str | None],
        last_vehicle_positions: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        runtime_rsu_ids = [str(rsu.rsu_id) for rsu in rsu_states]
        checkpoint_rsu_ids = set(self._rsu_ids)
        runtime_rsu_id_set = set(runtime_rsu_ids)
        if (
            len(runtime_rsu_ids) != len(runtime_rsu_id_set)
            or not runtime_rsu_id_set.issubset(checkpoint_rsu_ids)
        ):
            raise ValueError(
                f"runtime RSU map {runtime_rsu_ids} is not compatible with supervised predictor "
                f"checkpoint slot map {self._rsu_ids}"
            )
        sequences: dict[str, list[str | None]] = {}
        predicted_next: dict[str, str | None] = {}
        predicted_target: dict[str, str | None] = {}
        confidence: dict[str, float] = {}
        uncertainty: dict[str, float] = {}
        eta_by_vehicle: dict[str, float] = {}
        raw_scores: dict[str, dict[str, Any]] = {}
        prediction_details: dict[str, dict[str, Any]] = {}
        class_ids: list[str | None] = [*self._rsu_ids, None]
        for vehicle in vehicles:
            feature_vector = build_feature_vector(
                vehicle=vehicle,
                rsu_states=rsu_states,
                workflow_state=workflow_state,
                current_associations=current_associations,
                rsu_ids=self._rsu_ids,
                last_vehicle_positions=last_vehicle_positions,
            )
            feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                output = self._network(feature_tensor)
                next_probs = torch.softmax(output["next_rsu_logits"], dim=-1).squeeze(0)
                target_probs = torch.softmax(output["handoff_target_logits"], dim=-1).squeeze(0)
                handoff_prob = torch.sigmoid(output["handoff_logit"]).squeeze(0)
                eta_steps = float(output["eta_steps"].squeeze(0).item())
            next_index = int(torch.argmax(next_probs).item())
            target_index = int(torch.argmax(target_probs).item())
            next_rsu_id = self._index_to_rsu(next_index)
            target_rsu_id = self._index_to_rsu(target_index)
            if next_rsu_id is not None and str(next_rsu_id) not in runtime_rsu_id_set:
                next_rsu_id = None
            if target_rsu_id is not None and str(target_rsu_id) not in runtime_rsu_id_set:
                target_rsu_id = None
            current_rsu_id = current_associations.get(vehicle.vehicle_id)
            if target_rsu_id is not None and str(target_rsu_id) == str(current_rsu_id):
                # A handoff target is defined as the first RSU different from
                # the current association; retaining the current RSU here
                # would mask a valid next-RSU prediction in the sequence head.
                target_rsu_id = None
            handoff_confidence = _clamp(float(handoff_prob.item()), 0.0, 1.0)
            if (
                self._target_selection_mode == "hard"
                and (target_rsu_id is None or handoff_confidence < self._handoff_threshold)
            ):
                target_rsu_id = None
            sequence = self._build_sequence(
                current_rsu_id=current_rsu_id,
                next_rsu_id=next_rsu_id,
                target_rsu_id=target_rsu_id,
                eta_steps=eta_steps,
            )
            sequences[vehicle.vehicle_id] = sequence
            predicted_next[vehicle.vehicle_id] = sequence[0] if sequence else next_rsu_id
            predicted_target[vehicle.vehicle_id] = target_rsu_id
            confidence[vehicle.vehicle_id] = round(handoff_confidence, 6)
            uncertainty[vehicle.vehicle_id] = round(1.0 - handoff_confidence, 6)
            eta_by_vehicle[vehicle.vehicle_id] = round(eta_steps, 6)
            raw_scores[vehicle.vehicle_id] = {
                "next_rsu_probability": round(float(next_probs[next_index].item()), 6),
                "target_rsu_probability": round(float(target_probs[target_index].item()), 6),
                "handoff_probability": round(handoff_confidence, 6),
                "handoff_decision_threshold": round(self._handoff_threshold, 6),
                "target_selection_mode": self._target_selection_mode,
                "eta_steps": round(eta_steps, 6),
            }
            prediction_details[vehicle.vehicle_id] = {
                "class_ids": list(class_ids),
                "next_rsu_logits": [float(value) for value in output["next_rsu_logits"].squeeze(0).tolist()],
                "next_rsu_probabilities": [float(value) for value in next_probs.tolist()],
                "handoff_target_logits": [
                    float(value) for value in output["handoff_target_logits"].squeeze(0).tolist()
                ],
                "handoff_target_probabilities": [float(value) for value in target_probs.tolist()],
                "handoff_logit": float(output["handoff_logit"].squeeze(0).item()),
                "handoff_probability": handoff_confidence,
                "eta_point_estimate_steps": eta_steps,
                "eta_interval_steps": None,
                "eta_uncertainty": None,
                "feature_availability_mask": {
                    "current_observation": True,
                    "previous_vehicle_position": vehicle.vehicle_id in (last_vehicle_positions or {}),
                    "workflow_state": workflow_state is not None,
                    "current_rsu_association": current_rsu_id is not None,
                },
                "normalization_version": NORMALIZATION_VERSION,
            }
        return {
            "next_rsu_sequence": sequences,
            "predicted_next_rsu_by_vehicle": predicted_next,
            "predicted_first_handoff_rsu_by_vehicle": predicted_target,
            "predicted_handoff_target_rsu_id_by_vehicle": predicted_target,
            "predicted_handoff_vehicle_ids": [
                vehicle_id for vehicle_id, target_rsu_id in predicted_target.items() if target_rsu_id is not None
            ],
            "prediction_confidence_by_vehicle": confidence,
            "prediction_uncertainty_by_vehicle": uncertainty,
            "predicted_handoff_eta_steps_by_vehicle": eta_by_vehicle,
            "supervised_predictor_scores_by_vehicle": raw_scores,
            "supervised_prediction_details_by_vehicle": prediction_details,
        }

    def _build_sequence(
        self,
        *,
        current_rsu_id: str | None,
        next_rsu_id: str | None,
        target_rsu_id: str | None,
        eta_steps: float,
    ) -> list[str | None]:
        eta_index = max(1, min(int(round(float(eta_steps))), self._horizon))
        sequence: list[str | None] = []
        for step_index in range(1, self._horizon + 1):
            if target_rsu_id is not None and step_index >= eta_index:
                sequence.append(target_rsu_id)
            elif next_rsu_id is not None:
                sequence.append(next_rsu_id)
            else:
                sequence.append(current_rsu_id)
        return sequence
