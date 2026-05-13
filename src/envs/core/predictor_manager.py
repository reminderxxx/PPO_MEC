"""预测层管理器与首轮 baseline predictor。"""

from __future__ import annotations

from collections import Counter
import random
from math import dist, sqrt
from typing import Any

from src.data.mobility.rsu_mapper import RSUMapper
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.envs.specs import RSUState, VehicleState, WorkflowGraphState


class PredictorManager:
    """管理短时预测接口。

    当前实现是第一轮 baseline predictor，只用于打通协议层。
    后续可替换为数字孪生、surrogate、seq2seq 或 learned predictor。
    """

    def __init__(
        self,
        horizon: int = 3,
        history_window: int = 5,
        prediction_noise_std: float = 0.0,
        prediction_confidence_scale: float = 1.0,
        prediction_delay_steps: int = 0,
        drop_handoff_prediction_prob: float = 0.0,
        oracle_prediction_enabled: bool = False,
        disable_prediction_output: bool = False,
        oracle_future_frames: list[dict[str, Any]] | None = None,
        oracle_rsu_states: list[RSUState] | None = None,
        random_seed: int = 7,
        predictor_kind: str = "baseline",
    ) -> None:
        self._horizon = max(int(horizon), 1)
        self._history_window = max(int(history_window), 1)
        self._prediction_noise_std = max(float(prediction_noise_std), 0.0)
        self._prediction_confidence_scale = max(float(prediction_confidence_scale), 0.0)
        self._prediction_delay_steps = max(int(prediction_delay_steps), 0)
        self._drop_handoff_prediction_prob = min(max(float(drop_handoff_prediction_prob), 0.0), 1.0)
        self._oracle_prediction_enabled = bool(oracle_prediction_enabled)
        self._disable_prediction_output = bool(disable_prediction_output)
        self._oracle_future_frames = [
            {
                "time_index": int(frame.get("time_index", 0)),
                "vehicles": list(frame.get("vehicles", [])),
            }
            for frame in (oracle_future_frames or [])
        ]
        self._oracle_rsu_states = list(oracle_rsu_states or [])
        self._requested_predictor_kind = self._normalize_predictor_kind(predictor_kind)
        self._oracle_time_to_index = {
            int(frame.get("time_index", 0)): index
            for index, frame in enumerate(self._oracle_future_frames)
        }
        self._rng = random.Random(random_seed)
        self._load_history: list[dict[str, int]] = []
        self._adapter_heat: Counter[str] = Counter()
        self._last_vehicle_positions: dict[str, tuple[float, float]] = {}
        self._prediction_history: list[dict[str, Any]] = []

    def reset(self) -> None:
        """重置内部统计状态。"""
        self._load_history = []
        self._adapter_heat = Counter()
        self._last_vehicle_positions = {}
        self._prediction_history = []

    def predict_next_rsu_sequence(
        self,
        vehicles: list[VehicleState],
        rsu_states: list[RSUState],
    ) -> dict[str, list[str | None]]:
        """基于最近一帧位移方向做短时位置外推并预测未来 RSU 序列。"""
        return {
            vehicle.vehicle_id: self._predict_vehicle_rsu_sequence(vehicle, rsu_states)
            for vehicle in vehicles
        }

    def predict_dwell_time(
        self,
        vehicles: list[VehicleState],
        rsu_states: list[RSUState],
    ) -> dict[str, float]:
        """估计车辆在当前关联 RSU 内的停留时间。"""
        rsu_map = {rsu.rsu_id: rsu for rsu in rsu_states}
        dwell_time: dict[str, float] = {}
        for vehicle in vehicles:
            if vehicle.associated_rsu_id is None:
                dwell_time[vehicle.vehicle_id] = 0.0
                continue
            rsu = rsu_map.get(vehicle.associated_rsu_id)
            if rsu is None:
                dwell_time[vehicle.vehicle_id] = 0.0
                continue
            center_distance = dist((vehicle.position_x, vehicle.position_y), (rsu.position_x, rsu.position_y))
            remaining_distance = max(rsu.coverage_radius - center_distance, 0.0)
            dwell_time[vehicle.vehicle_id] = round(
                remaining_distance / max(vehicle.speed, 1.0),
                3,
            )
        return dwell_time

    def predict_future_load(
        self,
        vehicles: list[VehicleState],
        rsu_states: list[RSUState],
        predicted_sequence: dict[str, list[str | None]] | None = None,
    ) -> dict[str, float]:
        """基于历史统计与一步前瞻估计未来 RSU 负载。"""
        current_load = {
            rsu.rsu_id: len(rsu.active_vehicle_ids) for rsu in rsu_states
        }
        self._load_history.append(current_load)
        self._load_history = self._load_history[-self._history_window :]

        historical_average: dict[str, float] = {}
        for rsu in rsu_states:
            values = [frame.get(rsu.rsu_id, 0) for frame in self._load_history]
            historical_average[rsu.rsu_id] = sum(values) / max(len(values), 1)

        predicted_sequence = predicted_sequence or self.predict_next_rsu_sequence(
            vehicles,
            rsu_states,
        )
        predicted_next_load = {rsu.rsu_id: 0.0 for rsu in rsu_states}
        for sequence in predicted_sequence.values():
            if not sequence:
                continue
            next_rsu_id = sequence[0]
            if next_rsu_id in predicted_next_load:
                predicted_next_load[next_rsu_id] += 1.0

        return {
            rsu.rsu_id: round(
                0.6 * historical_average[rsu.rsu_id] + 0.4 * predicted_next_load[rsu.rsu_id],
                3,
            )
            for rsu in rsu_states
        }

    def predict_cache_demand(
        self,
        workflow_state: WorkflowGraphState,
        rsu_states: list[RSUState],
        vehicles: list[VehicleState],
        adapter_catalog: AdapterCatalog,
        predicted_sequence: dict[str, list[str | None]] | None = None,
    ) -> dict[str, Any]:
        """结合后续 workflow 节点与 adapter 热度估计 cache demand。"""
        remaining_nodes = [
            node
            for node in workflow_state.nodes
            if node.node_id not in workflow_state.completed_node_ids
        ]
        for node in remaining_nodes:
            self._adapter_heat[node.required_adapter] += 1

        predicted_sequence = predicted_sequence or self.predict_next_rsu_sequence(
            vehicles,
            rsu_states,
        )
        demand_by_rsu: dict[str, dict[str, float]] = {rsu.rsu_id: {} for rsu in rsu_states}
        ranked_future_adapters = [
            node.required_adapter for node in remaining_nodes[: self._horizon + 1]
        ]

        for vehicle in vehicles:
            sequence = predicted_sequence.get(vehicle.vehicle_id, [])
            for step_index, predicted_rsu_id in enumerate(sequence):
                if predicted_rsu_id is None:
                    continue
                for adapter_rank, adapter_id in enumerate(ranked_future_adapters):
                    rank_weight = 1.0 / float(adapter_rank + 1)
                    time_weight = 1.0 / float(step_index + 1)
                    heat_weight = 1.0 + 0.1 * float(self._adapter_heat.get(adapter_id, 0))
                    current_score = demand_by_rsu[predicted_rsu_id].get(adapter_id, 0.0)
                    demand_by_rsu[predicted_rsu_id][adapter_id] = round(
                        current_score + rank_weight * time_weight * heat_weight,
                        3,
                    )

        top_adapter_by_rsu: dict[str, str | None] = {}
        for rsu in rsu_states:
            rsu_scores = demand_by_rsu[rsu.rsu_id]
            if not rsu_scores:
                top_adapter_by_rsu[rsu.rsu_id] = None
                continue
            top_adapter_by_rsu[rsu.rsu_id] = max(
                rsu_scores.items(),
                key=lambda item: item[1],
            )[0]

        return {
            "remaining_workflow_adapters": ranked_future_adapters,
            "demand_score_by_rsu": demand_by_rsu,
            "top_adapter_by_rsu": top_adapter_by_rsu,
            "vehicle_base_models": adapter_catalog.get_vehicle_base_model_ids(),
        }

    def build_predictions(
        self,
        time_index: int,
        vehicles: list[VehicleState],
        rsu_states: list[RSUState],
        workflow_state: WorkflowGraphState,
        adapter_catalog: AdapterCatalog,
        current_associations: dict[str, str | None],
    ) -> dict[str, Any]:
        """构造环境状态中的统一 predictions 字段。"""
        next_rsu_sequence = self.predict_next_rsu_sequence(vehicles, rsu_states)
        predicted_next_rsu_by_vehicle = {
            vehicle_id: sequence[0] if sequence else current_associations.get(vehicle_id)
            for vehicle_id, sequence in next_rsu_sequence.items()
        }
        predicted_first_handoff_rsu_by_vehicle = {
            vehicle_id: self._extract_first_handoff_rsu(
                current_rsu_id=current_associations.get(vehicle_id),
                sequence=sequence,
            )
            for vehicle_id, sequence in next_rsu_sequence.items()
        }
        predicted_handoff_vehicle_ids = [
            vehicle_id
            for vehicle_id, target_rsu_id in predicted_first_handoff_rsu_by_vehicle.items()
            if target_rsu_id is not None
        ]
        dwell_time = self.predict_dwell_time(vehicles, rsu_states)
        future_load = self.predict_future_load(
            vehicles,
            rsu_states,
            predicted_sequence=next_rsu_sequence,
        )
        cache_demand = self.predict_cache_demand(
            workflow_state=workflow_state,
            rsu_states=rsu_states,
            vehicles=vehicles,
            adapter_catalog=adapter_catalog,
            predicted_sequence=next_rsu_sequence,
        )
        confidence_by_vehicle, uncertainty_by_vehicle = self.predict_confidence(
            vehicles=vehicles,
            current_associations=current_associations,
            next_rsu_sequence=next_rsu_sequence,
            dwell_time=dwell_time,
        )
        surrogate_delay_by_vehicle = {
            vehicle.vehicle_id: round(
                0.8 + 0.2 * len(next_rsu_sequence.get(vehicle.vehicle_id, [])),
                3,
            )
            for vehicle in vehicles
        }
        predictions = {
            "predictor_name": "baseline_predictor_v2",
            "predictor_note": "?? baseline predictor??????? DT / surrogate / seq2seq / learned predictor?",
            "prediction_time": time_index,
            "next_rsu_sequence": next_rsu_sequence,
            "predicted_next_rsu_by_vehicle": predicted_next_rsu_by_vehicle,
            "predicted_first_handoff_rsu_by_vehicle": predicted_first_handoff_rsu_by_vehicle,
            "predicted_handoff_vehicle_ids": predicted_handoff_vehicle_ids,
            "dwell_time": dwell_time,
            "future_load": future_load,
            "cache_demand": cache_demand,
            "surrogate_delay_by_vehicle": surrogate_delay_by_vehicle,
            "prediction_confidence_by_vehicle": confidence_by_vehicle,
            "prediction_uncertainty_by_vehicle": uncertainty_by_vehicle,
            "predicted_handoff_target_rsu_id_by_vehicle": predicted_first_handoff_rsu_by_vehicle,
            "_current_associations": dict(current_associations),
        }
        if self._oracle_prediction_enabled:
            predictions = self._apply_oracle_predictions(
                predictions=predictions,
                time_index=time_index,
                vehicles=vehicles,
                rsu_states=rsu_states,
            )
        if self._disable_prediction_output:
            predictions = self._apply_no_prediction_mode(
                predictions=predictions,
                vehicles=vehicles,
            )
        self._update_last_positions(vehicles)
        perturbed_predictions = self._apply_prediction_perturbations(predictions)
        return self._attach_prediction_contract_metadata(
            predictions=perturbed_predictions,
            vehicles=vehicles,
            current_associations=current_associations,
        )

    def _normalize_predictor_kind(self, predictor_kind: str) -> str:
        normalized = str(predictor_kind or "baseline").strip().lower()
        if normalized in {"learned", "calibrated", "learned_calibrated", "learned-or-calibrated"}:
            return "learned_or_calibrated"
        if normalized in {"baseline", "oracle", "learned_or_calibrated", "no_prediction"}:
            return normalized
        return "baseline"

    def _effective_predictor_kind(self, predictions: dict[str, Any]) -> str:
        predictor_name = str(predictions.get("predictor_name", ""))
        if self._disable_prediction_output or predictor_name == "no_prediction_mode":
            return "no_prediction"
        if predictor_name.startswith("oracle_"):
            return "oracle"
        if self._requested_predictor_kind == "oracle":
            return "baseline"
        return self._requested_predictor_kind

    def _attach_prediction_contract_metadata(
        self,
        *,
        predictions: dict[str, Any],
        vehicles: list[VehicleState],
        current_associations: dict[str, str | None],
    ) -> dict[str, Any]:
        predictor_kind = self._effective_predictor_kind(predictions)
        claim_boundary_by_kind = {
            "baseline": "prediction_aware_surrogate_feature_assisted",
            "oracle": "oracle_diagnostic_not_learned_policy_claim",
            "learned_or_calibrated": "calibrated_surrogate_candidate_not_learned_model",
            "no_prediction": "no_prediction_diagnostic",
        }
        if predictor_kind == "learned_or_calibrated":
            predictions["predictor_name"] = "calibrated_baseline_surrogate_v1"
            predictions["predictor_note"] = (
                "calibrated baseline surrogate interface; no learned predictor checkpoint attached"
            )
        elif predictor_kind == "baseline":
            predictions["predictor_name"] = "baseline_predictor_v2"
            predictions["predictor_note"] = (
                "baseline short-horizon predictor; use prediction-aware/surrogate-feature-assisted wording"
            )
        predictions["predictor_kind"] = predictor_kind
        predictions["predictor_interfaces_available"] = [
            "baseline",
            "oracle",
            "learned_or_calibrated",
        ]
        predictions["learned_predictor_attached"] = False
        predictions["surrogate_claim_boundary"] = claim_boundary_by_kind.get(
            predictor_kind,
            "prediction_aware_surrogate_feature_assisted",
        )
        predictions["prediction_quality_audit"] = self._build_prediction_quality_audit(
            predictions=predictions,
            vehicles=vehicles,
            current_associations=current_associations,
        )
        return predictions

    def _build_prediction_quality_audit(
        self,
        *,
        predictions: dict[str, Any],
        vehicles: list[VehicleState],
        current_associations: dict[str, str | None],
    ) -> dict[str, Any]:
        predicted_targets = dict(predictions.get("predicted_first_handoff_rsu_by_vehicle", {}))
        next_sequences = {
            vehicle_id: list(sequence)
            for vehicle_id, sequence in predictions.get("next_rsu_sequence", {}).items()
        }
        confidence_by_vehicle = dict(predictions.get("prediction_confidence_by_vehicle", {}))
        true_positive = 0
        false_positive = 0
        false_negative = 0
        brier_terms: list[float] = []
        confidence_values: list[float] = []
        label_values: list[float] = []
        for vehicle in vehicles:
            vehicle_id = vehicle.vehicle_id
            current_rsu_id = current_associations.get(vehicle_id)
            sequence_target = self._extract_first_handoff_rsu(
                current_rsu_id=current_rsu_id,
                sequence=next_sequences.get(vehicle_id, []),
            )
            predicted_target = predicted_targets.get(vehicle_id)
            if predicted_target is not None and predicted_target == sequence_target:
                true_positive += 1
            elif predicted_target is not None:
                false_positive += 1
            elif sequence_target is not None:
                false_negative += 1
            confidence = min(max(float(confidence_by_vehicle.get(vehicle_id, 0.0) or 0.0), 0.0), 1.0)
            label = 1.0 if sequence_target is not None else 0.0
            confidence_values.append(confidence)
            label_values.append(label)
            brier_terms.append((confidence - label) * (confidence - label))

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        mean_confidence = sum(confidence_values) / max(len(confidence_values), 1)
        mean_label = sum(label_values) / max(len(label_values), 1)
        brier_score = sum(brier_terms) / max(len(brier_terms), 1)
        return {
            "audit_schema": "predictor_quality_audit_v1",
            "audit_scope": "online_proxy_from_current_prediction_sequence",
            "handoff_target_true_positive_proxy": true_positive,
            "handoff_target_false_positive_proxy": false_positive,
            "handoff_target_false_negative_proxy": false_negative,
            "handoff_target_precision_proxy": round(
                float(true_positive) / float(max(precision_denominator, 1)),
                6,
            ),
            "handoff_target_recall_proxy": round(
                float(true_positive) / float(max(recall_denominator, 1)),
                6,
            ),
            "brier_score_proxy": round(brier_score, 6),
            "confidence_mean": round(mean_confidence, 6),
            "handoff_signal_rate_proxy": round(mean_label, 6),
            "confidence_calibration_error_proxy": round(abs(mean_confidence - mean_label), 6),
            "prediction_delay_steps": self._prediction_delay_steps,
            "drop_handoff_prediction_prob": round(self._drop_handoff_prediction_prob, 6),
            "prediction_noise_std": round(self._prediction_noise_std, 6),
            "prediction_confidence_scale": round(self._prediction_confidence_scale, 6),
            "delay_drop_sensitivity_enabled": bool(
                self._prediction_delay_steps > 0
                or self._drop_handoff_prediction_prob > 0.0
                or self._prediction_noise_std > 0.0
            ),
            "claim_note": (
                "proxy audit for runtime diagnostics; paper-grade predictor quality needs realized future labels"
            ),
        }

    def _apply_oracle_predictions(
        self,
        predictions: dict[str, Any],
        time_index: int,
        vehicles: list[VehicleState],
        rsu_states: list[RSUState],
    ) -> dict[str, Any]:
        if not self._oracle_future_frames:
            return predictions
        current_index = self._oracle_time_to_index.get(int(time_index))
        if current_index is None:
            return predictions
        mapper = RSUMapper(self._oracle_rsu_states or rsu_states)
        current_associations = dict(predictions.get("_current_associations", {}))
        next_rsu_sequence: dict[str, list[str | None]] = {}
        for vehicle in vehicles:
            sequence: list[str | None] = []
            for step_offset in range(1, self._horizon + 1):
                future_index = current_index + step_offset
                if future_index >= len(self._oracle_future_frames):
                    sequence.append(None)
                    continue
                future_frame = self._oracle_future_frames[future_index]
                future_vehicle = next(
                    (candidate for candidate in future_frame.get("vehicles", []) if getattr(candidate, "vehicle_id", None) == vehicle.vehicle_id),
                    None,
                )
                if future_vehicle is None:
                    sequence.append(None)
                    continue
                future_association = mapper.associate([future_vehicle]).get(vehicle.vehicle_id)
                sequence.append(future_association)
            next_rsu_sequence[vehicle.vehicle_id] = sequence
        predicted_next_rsu_by_vehicle = {
            vehicle_id: sequence[0] if sequence else current_associations.get(vehicle_id)
            for vehicle_id, sequence in next_rsu_sequence.items()
        }
        predicted_first_handoff_rsu_by_vehicle = {
            vehicle_id: self._extract_first_handoff_rsu(
                current_rsu_id=current_associations.get(vehicle_id),
                sequence=sequence,
            )
            for vehicle_id, sequence in next_rsu_sequence.items()
        }
        predictions["predictor_name"] = "oracle_predictor_v1"
        predictions["predictor_note"] = "oracle predictor?????? mobility frame ?? next_rsu / handoff target????????? robustness ???"
        predictions["next_rsu_sequence"] = next_rsu_sequence
        predictions["predicted_next_rsu_by_vehicle"] = predicted_next_rsu_by_vehicle
        predictions["predicted_first_handoff_rsu_by_vehicle"] = predicted_first_handoff_rsu_by_vehicle
        predictions["predicted_handoff_target_rsu_id_by_vehicle"] = predicted_first_handoff_rsu_by_vehicle
        predictions["predicted_handoff_vehicle_ids"] = [
            vehicle_id for vehicle_id, target_rsu_id in predicted_first_handoff_rsu_by_vehicle.items() if target_rsu_id is not None
        ]
        predictions["prediction_confidence_by_vehicle"] = {vehicle.vehicle_id: 0.98 for vehicle in vehicles}
        predictions["prediction_uncertainty_by_vehicle"] = {vehicle.vehicle_id: 0.02 for vehicle in vehicles}
        return predictions

    def _apply_no_prediction_mode(
        self,
        predictions: dict[str, Any],
        vehicles: list[VehicleState],
    ) -> dict[str, Any]:
        empty_sequence = {vehicle.vehicle_id: [None] * self._horizon for vehicle in vehicles}
        predictions["predictor_name"] = "no_prediction_mode"
        predictions["predictor_note"] = "no_prediction ??????? predictor ???????????????"
        predictions["next_rsu_sequence"] = empty_sequence
        predictions["predicted_next_rsu_by_vehicle"] = {vehicle.vehicle_id: None for vehicle in vehicles}
        predictions["predicted_first_handoff_rsu_by_vehicle"] = {vehicle.vehicle_id: None for vehicle in vehicles}
        predictions["predicted_handoff_target_rsu_id_by_vehicle"] = {vehicle.vehicle_id: None for vehicle in vehicles}
        predictions["predicted_handoff_vehicle_ids"] = []
        predictions["prediction_confidence_by_vehicle"] = {vehicle.vehicle_id: 0.0 for vehicle in vehicles}
        predictions["prediction_uncertainty_by_vehicle"] = {vehicle.vehicle_id: 1.0 for vehicle in vehicles}
        predictions["future_load"] = {rsu_id: 0.0 for rsu_id in predictions.get("future_load", {}).keys()}
        cache_demand = dict(predictions.get("cache_demand", {}))
        cache_demand["demand_score_by_rsu"] = {rsu_id: {} for rsu_id in cache_demand.get("demand_score_by_rsu", {}).keys()}
        cache_demand["top_adapter_by_rsu"] = {rsu_id: None for rsu_id in cache_demand.get("top_adapter_by_rsu", {}).keys()}
        cache_demand["remaining_workflow_adapters"] = []
        predictions["cache_demand"] = cache_demand
        return predictions

    def predict_confidence(
        self,
        vehicles: list[VehicleState],
        current_associations: dict[str, str | None],
        next_rsu_sequence: dict[str, list[str | None]],
        dwell_time: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """输出预测置信度与不确定性，供 surrogate-assisted 策略显式消费。"""
        confidence_by_vehicle: dict[str, float] = {}
        uncertainty_by_vehicle: dict[str, float] = {}
        for vehicle in vehicles:
            vehicle_id = vehicle.vehicle_id
            current_rsu_id = current_associations.get(vehicle_id)
            sequence = list(next_rsu_sequence.get(vehicle_id, []))
            non_null_sequence = [rsu_id for rsu_id in sequence if rsu_id is not None]
            handoff_target = self._extract_first_handoff_rsu(current_rsu_id=current_rsu_id, sequence=sequence)
            dwell_score = min(max(float(dwell_time.get(vehicle_id, 0.0)) / 10.0, 0.0), 1.0)
            stability_score = 1.0
            if non_null_sequence:
                stability_score = 1.0 / float(len(set(non_null_sequence)))
            handoff_score = 0.75 if handoff_target is not None else 0.45
            confidence = 0.45 * stability_score + 0.35 * dwell_score + 0.20 * handoff_score
            confidence = max(0.05, min(confidence, 0.95))
            confidence_by_vehicle[vehicle_id] = round(confidence, 3)
            uncertainty_by_vehicle[vehicle_id] = round(1.0 - confidence, 3)
        return confidence_by_vehicle, uncertainty_by_vehicle

    def _apply_prediction_perturbations(self, predictions: dict[str, Any]) -> dict[str, Any]:
        perturbed = {
            key: (value.copy() if isinstance(value, dict) else list(value) if isinstance(value, list) else value)
            for key, value in predictions.items()
        }
        next_rsu_sequence = {
            vehicle_id: list(sequence)
            for vehicle_id, sequence in predictions.get("next_rsu_sequence", {}).items()
        }
        current_associations = dict(predictions.get("_current_associations", {}))
        if self._prediction_noise_std > 0.0:
            next_rsu_sequence = self._inject_sequence_noise(next_rsu_sequence)
        current_predicted_next = {}
        current_handoff_targets = {}
        for vehicle_id, sequence in next_rsu_sequence.items():
            current_predicted_next[vehicle_id] = sequence[0] if sequence else None
            current_handoff_targets[vehicle_id] = self._extract_first_handoff_rsu(
                current_rsu_id=current_associations.get(vehicle_id),
                sequence=sequence,
            )
        if self._drop_handoff_prediction_prob > 0.0:
            for vehicle_id, target_rsu_id in list(current_handoff_targets.items()):
                if target_rsu_id is None:
                    continue
                if self._rng.random() < self._drop_handoff_prediction_prob:
                    current_handoff_targets[vehicle_id] = None
        perturbed["next_rsu_sequence"] = next_rsu_sequence
        perturbed["predicted_next_rsu_by_vehicle"] = current_predicted_next
        perturbed["predicted_first_handoff_rsu_by_vehicle"] = current_handoff_targets
        perturbed["predicted_handoff_target_rsu_id_by_vehicle"] = current_handoff_targets
        perturbed["predicted_handoff_vehicle_ids"] = [
            vehicle_id for vehicle_id, target_rsu_id in current_handoff_targets.items() if target_rsu_id is not None
        ]
        confidence_by_vehicle = {
            vehicle_id: round(min(max(float(confidence) * self._prediction_confidence_scale, 0.0), 1.0), 3)
            for vehicle_id, confidence in predictions.get("prediction_confidence_by_vehicle", {}).items()
        }
        uncertainty_by_vehicle = {
            vehicle_id: round(1.0 - confidence, 3)
            for vehicle_id, confidence in confidence_by_vehicle.items()
        }
        perturbed["prediction_confidence_by_vehicle"] = confidence_by_vehicle
        perturbed["prediction_uncertainty_by_vehicle"] = uncertainty_by_vehicle
        perturbed["prediction_perturbation"] = {
            "prediction_noise_std": self._prediction_noise_std,
            "prediction_confidence_scale": self._prediction_confidence_scale,
            "prediction_delay_steps": self._prediction_delay_steps,
            "drop_handoff_prediction_prob": self._drop_handoff_prediction_prob,
        }
        perturbed.pop("_current_associations", None)
        self._prediction_history.append(perturbed)
        if self._prediction_delay_steps > 0 and len(self._prediction_history) > self._prediction_delay_steps:
            delayed = self._prediction_history[-(self._prediction_delay_steps + 1)]
            return {
                key: (value.copy() if isinstance(value, dict) else list(value) if isinstance(value, list) else value)
                for key, value in delayed.items()
            }
        return perturbed

    def _inject_sequence_noise(self, next_rsu_sequence: dict[str, list[str | None]]) -> dict[str, list[str | None]]:
        perturbed: dict[str, list[str | None]] = {}
        for vehicle_id, sequence in next_rsu_sequence.items():
            new_sequence = list(sequence)
            for index, rsu_id in enumerate(new_sequence):
                if rsu_id is None:
                    continue
                if self._rng.random() < min(self._prediction_noise_std, 1.0):
                    if index + 1 < len(new_sequence):
                        new_sequence[index] = new_sequence[index + 1]
                    else:
                        new_sequence[index] = None
            perturbed[vehicle_id] = new_sequence
        return perturbed

    def _predict_vehicle_rsu_sequence(
        self,
        vehicle: VehicleState,
        rsu_states: list[RSUState],
    ) -> list[str | None]:
        direction_x, direction_y = self._estimate_motion_direction(vehicle)
        sequence: list[str | None] = []
        for step_index in range(1, self._horizon + 1):
            predicted_x = vehicle.position_x + direction_x * vehicle.speed * float(step_index)
            predicted_y = vehicle.position_y + direction_y * vehicle.speed * float(step_index)
            sequence.append(self._find_best_rsu(predicted_x, predicted_y, rsu_states))
        return sequence

    def _find_best_rsu(
        self,
        position_x: float,
        position_y: float,
        rsu_states: list[RSUState],
    ) -> str | None:
        best_rsu_id: str | None = None
        best_distance: float | None = None
        for rsu in rsu_states:
            current_distance = dist(
                (position_x, position_y),
                (rsu.position_x, rsu.position_y),
            )
            if current_distance > rsu.coverage_radius:
                continue
            if best_distance is None or current_distance < best_distance:
                best_distance = current_distance
                best_rsu_id = rsu.rsu_id
        return best_rsu_id

    def _extract_first_handoff_rsu(
        self,
        current_rsu_id: str | None,
        sequence: list[str | None],
    ) -> str | None:
        for rsu_id in sequence:
            if rsu_id is not None and rsu_id != current_rsu_id:
                return rsu_id
        return None

    def _estimate_motion_direction(self, vehicle: VehicleState) -> tuple[float, float]:
        previous_position = self._last_vehicle_positions.get(vehicle.vehicle_id)
        if previous_position is None:
            return 1.0, 0.0
        delta_x = float(vehicle.position_x) - float(previous_position[0])
        delta_y = float(vehicle.position_y) - float(previous_position[1])
        norm = sqrt(delta_x * delta_x + delta_y * delta_y)
        if norm <= 1e-6:
            return 1.0, 0.0
        return delta_x / norm, delta_y / norm

    def _update_last_positions(self, vehicles: list[VehicleState]) -> None:
        self._last_vehicle_positions = {
            vehicle.vehicle_id: (float(vehicle.position_x), float(vehicle.position_y))
            for vehicle in vehicles
        }
