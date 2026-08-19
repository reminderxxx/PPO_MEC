"""Versioned causal calibrated predictor snapshot contract for G12."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.predictors.calibration import (
    apply_binary_temperature,
    apply_multiclass_temperature,
    canonical_sha256,
    validate_probability_simplex,
)


CAUSAL_PREDICTOR_SNAPSHOT_CONTRACT_VERSION = "1.0.0"
CALIBRATION_ARTIFACT_CONTRACT_VERSION = "causal_predictor_calibration_artifact_v1.0.0"
SNAPSHOT_VALIDATOR_VERSION = "causal_predictor_snapshot_validator_v1.0.0"
ABSTENTION_REASONS = frozenset(
    {
        "confidence_below_threshold",
        "calibration_unavailable",
        "snapshot_stale",
        "prediction_expired",
        "unseen_rsu_or_class",
        "invalid_probability_simplex",
        "insufficient_history",
        "target_not_distinct",
        "predictor_unavailable",
        "handoff_target_unavailable",
        "oracle_not_allowed_for_supervised_consumer",
    }
)


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _optional_finite(value: Any, field: str) -> float | None:
    return None if value is None else _finite(value, field)


def _assert_json_safe(value: Any, path: str = "snapshot") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains NaN or infinity")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string JSON key")
            _assert_json_safe(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_json_safe(item, f"{path}[{index}]")
        return
    raise ValueError(f"{path} contains non-JSON-safe type {type(value).__name__}")


def load_calibration_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"calibration artifact not found: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8-sig"))
    if payload.get("calibration_artifact_contract_version") != CALIBRATION_ARTIFACT_CONTRACT_VERSION:
        raise ValueError("unsupported calibration artifact contract version")
    if payload.get("fit_split") != "calibration":
        raise ValueError("calibrator must be fit on the calibration split")
    if payload.get("evaluation_labels_used_for_fit") is not False:
        raise ValueError("calibration artifact must state evaluation_labels_used_for_fit=false")
    if payload.get("rl_reward_used_for_selection") is not False:
        raise ValueError("calibration artifact must state rl_reward_used_for_selection=false")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("calibration artifact parameters must be an object")
    for task in ("handoff", "next_rsu", "handoff_target"):
        task_parameters = parameters.get(task)
        if not isinstance(task_parameters, dict):
            raise ValueError(f"calibration artifact missing {task} parameters")
        method = task_parameters.get("method")
        if method not in {"identity", "binary_temperature_scaling", "multiclass_temperature_scaling"}:
            raise ValueError(f"unsupported {task} calibration method: {method}")
        temperature = _finite(task_parameters.get("temperature", 1.0), f"{task}.temperature")
        if temperature <= 0.0:
            raise ValueError(f"{task}.temperature must be positive")
    threshold = payload.get("abstention", {}).get("selected_threshold")
    if threshold is None or not 0.0 <= _finite(threshold, "abstention.selected_threshold") <= 1.0:
        raise ValueError("calibration artifact requires a selected abstention threshold")
    _assert_json_safe(payload, "calibration_artifact")
    payload["_artifact_path"] = str(artifact_path.resolve())
    payload["_artifact_sha256"] = sha256_file(artifact_path)
    return payload


def calibration_artifact_round_trip(payload: dict[str, Any]) -> dict[str, Any]:
    _assert_json_safe(payload, "calibration_artifact")
    return json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _distribution(class_ids: Sequence[str | None], probabilities: Sequence[float]) -> dict[str, float]:
    if len(class_ids) != len(probabilities):
        raise ValueError("class IDs and probabilities must have equal length")
    return {
        "__none__" if class_id is None else str(class_id): float(probability)
        for class_id, probability in zip(class_ids, probabilities)
    }


def _predicted_class(class_ids: Sequence[str | None], probabilities: Sequence[float]) -> str | None:
    if not probabilities:
        return None
    class_id = class_ids[max(range(len(probabilities)), key=lambda index: probabilities[index])]
    return None if class_id is None else str(class_id)


def build_causal_predictor_snapshot(
    *,
    vehicle_id: str,
    predictor_kind: str,
    model_identity: str,
    checkpoint_identity: dict[str, Any] | None,
    predictor_config_hash: str,
    source_dataset_identity: dict[str, Any],
    source_window_plan_identity: dict[str, Any] | None,
    git_commit: str,
    generated_at_step: int,
    generated_at_time: int | float,
    observation_as_of_step: int,
    observation_as_of_time: int | float,
    consumed_at_step: int,
    consumed_at_time: int | float,
    label_horizon: int,
    valid_for_steps: int,
    update_interval_steps: int,
    current_rsu_id: str | None,
    class_ids: Sequence[str | None],
    runtime_rsu_ids: Sequence[str],
    raw_next_rsu_logits: Sequence[float] | None,
    raw_handoff_logit: float | None,
    raw_handoff_target_logits: Sequence[float] | None,
    eta_point_estimate: float | None,
    calibration_artifact: dict[str, Any] | None,
    feature_availability_mask: dict[str, bool],
    normalization_version: str,
    history_start_step: int | None,
    history_end_step: int | None,
    history_start_time: int | float | None,
    history_end_time: int | float | None,
    source_frame_interval: Sequence[int] | None = None,
    source_time_interval: Sequence[int | float] | None = None,
    causal_cutoff_step: int | None = None,
    causal_cutoff_time: int | float | None = None,
    insufficient_history: bool = False,
    predictor_available: bool = True,
    fallback_behavior: str = "mask_only",
    oracle: bool = False,
    allow_oracle_consumer: bool = False,
    demand_or_arrival_belief: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generated_step = int(generated_at_step)
    as_of_step = int(observation_as_of_step)
    consumed_step = int(consumed_at_step)
    valid_from_step = generated_step
    valid_until_step = generated_step + max(int(valid_for_steps), 0)
    age_steps = consumed_step - generated_step
    reasons: list[str] = []
    calibrated_next: list[float] | None = None
    calibrated_target: list[float] | None = None
    calibrated_handoff: float | None = None
    raw_scores: dict[str, Any] = {
        "next_rsu_logits": list(raw_next_rsu_logits) if raw_next_rsu_logits is not None else None,
        "handoff_logit": raw_handoff_logit,
        "handoff_target_logits": list(raw_handoff_target_logits) if raw_handoff_target_logits is not None else None,
        "eta_model_output": eta_point_estimate,
    }
    if not predictor_available or raw_next_rsu_logits is None or raw_handoff_logit is None or raw_handoff_target_logits is None:
        reasons.append("predictor_unavailable")
    elif calibration_artifact is None:
        reasons.append("calibration_unavailable")
    else:
        parameters = calibration_artifact["parameters"]
        handoff_temperature = float(parameters["handoff"].get("temperature", 1.0))
        next_temperature = float(parameters["next_rsu"].get("temperature", 1.0))
        target_temperature = float(parameters["handoff_target"].get("temperature", 1.0))
        calibrated_handoff = apply_binary_temperature([float(raw_handoff_logit)], handoff_temperature)[0]
        calibrated_next = apply_multiclass_temperature([list(raw_next_rsu_logits)], next_temperature)[0]
        calibrated_target = apply_multiclass_temperature([list(raw_handoff_target_logits)], target_temperature)[0]
    if insufficient_history:
        reasons.append("insufficient_history")
    if oracle and not allow_oracle_consumer:
        reasons.append("oracle_not_allowed_for_supervised_consumer")
    if consumed_step > valid_until_step:
        reasons.extend(["snapshot_stale", "prediction_expired"])

    predicted_next = _predicted_class(class_ids, calibrated_next or [])
    candidate_target = _predicted_class(class_ids, calibrated_target or [])
    handoff_predicted = calibrated_handoff is not None and calibrated_handoff >= 0.5
    predicted_target = candidate_target if handoff_predicted else None
    runtime_set = {str(item) for item in runtime_rsu_ids}
    if predicted_next is not None and predicted_next not in runtime_set:
        reasons.append("unseen_rsu_or_class")
    if predicted_target is not None and predicted_target not in runtime_set:
        reasons.append("unseen_rsu_or_class")
    if calibrated_next is not None and not validate_probability_simplex(calibrated_next)["valid"]:
        reasons.append("invalid_probability_simplex")
    if calibrated_target is not None and not validate_probability_simplex(calibrated_target)["valid"]:
        reasons.append("invalid_probability_simplex")
    if handoff_predicted and predicted_target is None:
        reasons.append("handoff_target_unavailable")
    if handoff_predicted and predicted_target is not None and current_rsu_id is not None and predicted_target == str(current_rsu_id):
        reasons.append("target_not_distinct")

    confidence = max(calibrated_handoff, 1.0 - calibrated_handoff) if calibrated_handoff is not None else None
    uncertainty = 1.0 - confidence if confidence is not None else None
    confidence_threshold = (
        float(calibration_artifact["abstention"]["selected_threshold"])
        if calibration_artifact is not None
        else None
    )
    if confidence is not None and confidence_threshold is not None and confidence < confidence_threshold:
        reasons.append("confidence_below_threshold")
    reasons = sorted(set(reasons))
    available = not reasons
    calibration_identity = {
        "method": (
            {
                task: calibration_artifact["parameters"][task]["method"]
                for task in ("handoff", "next_rsu", "handoff_target")
            }
            if calibration_artifact is not None
            else None
        ),
        "version": calibration_artifact.get("calibration_method_version") if calibration_artifact else None,
        "artifact_hash": calibration_artifact.get("_artifact_sha256") if calibration_artifact else None,
    }
    immutable_identity = {
        "vehicle_id": str(vehicle_id),
        "predictor_kind": str(predictor_kind),
        "model_identity": str(model_identity),
        "checkpoint_identity": checkpoint_identity,
        "predictor_config_hash": str(predictor_config_hash),
        "calibration": calibration_identity,
        "generated_at_step": generated_step,
        "observation_as_of_step": as_of_step,
        "source_window_plan_identity": source_window_plan_identity,
        "git_commit": str(git_commit),
        "oracle": bool(oracle),
        "prediction_input_digest": canonical_sha256(
            {
                "raw_scores": raw_scores,
                "current_rsu_id": current_rsu_id,
                "runtime_rsu_ids": list(runtime_rsu_ids),
                "feature_availability_mask": feature_availability_mask,
                "history_start_step": history_start_step,
                "history_end_step": history_end_step,
            }
        ),
    }
    snapshot_id = f"causal-predictor-{canonical_sha256(immutable_identity)[:24]}"
    snapshot = {
        "identity": {
            "snapshot_id": snapshot_id,
            "contract_version": CAUSAL_PREDICTOR_SNAPSHOT_CONTRACT_VERSION,
            "predictor_kind": str(predictor_kind),
            "predictor_model_identity": str(model_identity),
            "predictor_checkpoint_identity": checkpoint_identity,
            "predictor_config_hash": str(predictor_config_hash),
            "calibration_method": calibration_identity["method"],
            "calibration_version": calibration_identity["version"],
            "calibration_artifact_hash": calibration_identity["artifact_hash"],
            "source_dataset_identity": deepcopy(source_dataset_identity),
            "source_window_plan_identity": deepcopy(source_window_plan_identity),
            "git_commit": str(git_commit),
            "vehicle_id": str(vehicle_id),
            "oracle": bool(oracle),
        },
        "causal_time": {
            "generated_at_step": generated_step,
            "generated_at_time": _finite(generated_at_time, "generated_at_time"),
            "observation_as_of_step": as_of_step,
            "observation_as_of_time": _finite(observation_as_of_time, "observation_as_of_time"),
            "label_horizon_steps": int(label_horizon),
            "valid_from_step": valid_from_step,
            "valid_until_step": valid_until_step,
            "consumed_at_step": consumed_step,
            "consumed_at_time": _finite(consumed_at_time, "consumed_at_time"),
            "age_steps": age_steps,
            "staleness_steps": age_steps,
            "update_interval_steps": int(update_interval_steps),
            "source_frame_interval": list(source_frame_interval) if source_frame_interval is not None else None,
            "source_time_interval": list(source_time_interval) if source_time_interval is not None else None,
            "history_start_step": history_start_step,
            "history_end_step": history_end_step,
            "history_start_time": _optional_finite(history_start_time, "history_start_time"),
            "history_end_time": _optional_finite(history_end_time, "history_end_time"),
            "causal_cutoff_step": as_of_step if causal_cutoff_step is None else int(causal_cutoff_step),
            "causal_cutoff_time": (
                _finite(observation_as_of_time, "observation_as_of_time")
                if causal_cutoff_time is None
                else _finite(causal_cutoff_time, "causal_cutoff_time")
            ),
        },
        "predictions": {
            "current_rsu_id": current_rsu_id,
            "next_rsu_probability_distribution": _distribution(class_ids, calibrated_next) if calibrated_next is not None else None,
            "predicted_next_rsu_id": predicted_next,
            "handoff_probability": calibrated_handoff,
            "handoff_predicted": handoff_predicted if calibrated_handoff is not None else None,
            "handoff_target_probability_distribution": _distribution(class_ids, calibrated_target) if calibrated_target is not None else None,
            "predicted_target_rsu_id": predicted_target,
            "eta_point_estimate_steps": _optional_finite(eta_point_estimate, "eta_point_estimate"),
            "eta_interval_steps": None,
            "eta_uncertainty": None,
            "demand_or_arrival_belief": deepcopy(demand_or_arrival_belief),
            "raw_scores": raw_scores,
            "calibrated_probabilities": {
                "next_rsu": _distribution(class_ids, calibrated_next) if calibrated_next is not None else None,
                "handoff": calibrated_handoff,
                "handoff_target": _distribution(class_ids, calibrated_target) if calibrated_target is not None else None,
            },
            "confidence": confidence,
            "confidence_definition": "max_calibrated_binary_handoff_class_probability",
            "uncertainty": uncertainty,
            "uncertainty_definition": "one_minus_confidence_not_ETA_uncertainty",
            "abstained": not available,
            "abstention_reasons": reasons,
            "availability": available,
            "availability_mask": 1 if available else 0,
        },
        "audit": {
            "feature_availability_mask": {str(key): bool(value) for key, value in feature_availability_mask.items()},
            "unseen_rsu_or_slot_handling": "abstain_and_mask",
            "normalization_version": str(normalization_version),
            "snapshot_validation_status": "pending",
            "snapshot_validator_version": SNAPSHOT_VALIDATOR_VERSION,
            "oracle_flag": bool(oracle),
            "leakage_audit": {
                "runtime_feature_cutoff_enforced": True,
                "future_label_in_runtime_snapshot": False,
                "reward_or_service_outcome_in_runtime_snapshot": False,
                "history_reset_required_per_window_or_episode": True,
            },
            "fallback_behavior": str(fallback_behavior),
            "unknown_values_are_json_null": True,
        },
    }
    validation = validate_causal_predictor_snapshot(snapshot)
    snapshot["audit"]["snapshot_validation_status"] = validation["status"]
    if validation["status"] != "pass":
        raise ValueError(f"causal predictor snapshot validation failed: {validation['errors']}")
    _assert_json_safe(snapshot)
    return snapshot


def consume_snapshot(snapshot: dict[str, Any], *, consumed_at_step: int, consumed_at_time: int | float) -> dict[str, Any]:
    consumed = deepcopy(snapshot)
    timing = consumed["causal_time"]
    timing["consumed_at_step"] = int(consumed_at_step)
    timing["consumed_at_time"] = _finite(consumed_at_time, "consumed_at_time")
    timing["age_steps"] = int(consumed_at_step) - int(timing["generated_at_step"])
    timing["staleness_steps"] = timing["age_steps"]
    reasons = set(consumed["predictions"].get("abstention_reasons", []))
    if int(consumed_at_step) > int(timing["valid_until_step"]):
        reasons.update({"snapshot_stale", "prediction_expired"})
    consumed["predictions"]["abstention_reasons"] = sorted(reasons)
    consumed["predictions"]["abstained"] = bool(reasons)
    consumed["predictions"]["availability"] = not reasons
    consumed["predictions"]["availability_mask"] = 0 if reasons else 1
    consumed["audit"]["snapshot_validation_status"] = "pending"
    validation = validate_causal_predictor_snapshot(consumed)
    consumed["audit"]["snapshot_validation_status"] = validation["status"]
    if validation["status"] != "pass":
        raise ValueError(f"consumed snapshot validation failed: {validation['errors']}")
    return consumed


def validate_causal_predictor_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    try:
        _assert_json_safe(snapshot)
    except ValueError as error:
        errors.append(str(error))
    required_sections = {"identity", "causal_time", "predictions", "audit"}
    if not isinstance(snapshot, dict) or not required_sections.issubset(snapshot):
        return {"status": "fail", "validator_version": SNAPSHOT_VALIDATOR_VERSION, "errors": ["snapshot sections are incomplete"]}
    identity = snapshot.get("identity", {})
    timing = snapshot.get("causal_time", {})
    predictions = snapshot.get("predictions", {})
    audit = snapshot.get("audit", {})
    if identity.get("contract_version") != CAUSAL_PREDICTOR_SNAPSHOT_CONTRACT_VERSION:
        errors.append("unsupported causal predictor snapshot contract version")
    oracle = bool(identity.get("oracle"))
    if oracle != bool(audit.get("oracle_flag")):
        errors.append("oracle identity and audit flags conflict")
    if oracle != (str(identity.get("predictor_kind")) == "oracle"):
        errors.append("oracle predictor must use a distinct oracle identity")
    try:
        generated_step = int(timing["generated_at_step"])
        as_of_step = int(timing["observation_as_of_step"])
        consumed_step = int(timing["consumed_at_step"])
        generated_time = _finite(timing["generated_at_time"], "generated_at_time")
        as_of_time = _finite(timing["observation_as_of_time"], "observation_as_of_time")
        consumed_time = _finite(timing["consumed_at_time"], "consumed_at_time")
        valid_from = int(timing["valid_from_step"])
        valid_until = int(timing["valid_until_step"])
        age = int(timing["age_steps"])
        staleness = int(timing["staleness_steps"])
        cutoff_step = int(timing["causal_cutoff_step"])
        if as_of_step > generated_step:
            errors.append("observation_as_of_step cannot be in the future of generation")
        if generated_step > consumed_step:
            errors.append("generated_at_step cannot be later than consumed_at_step")
        if as_of_time > generated_time:
            errors.append("observation_as_of_time cannot be in the future of generation")
        if generated_time > consumed_time:
            errors.append("generated_at_time cannot be later than consumed_at_time")
        if valid_from < generated_step or valid_until < valid_from:
            errors.append("snapshot validity interval is inconsistent")
        if age != consumed_step - generated_step or staleness != age:
            errors.append("snapshot age/staleness is not derived from generation and consumption")
        if cutoff_step > as_of_step:
            errors.append("causal cutoff cannot exceed observation as-of")
        history_end = timing.get("history_end_step")
        if history_end is not None and int(history_end) > as_of_step:
            errors.append("history cannot extend beyond observation as-of")
        history_end_time = timing.get("history_end_time")
        if history_end_time is not None and _finite(history_end_time, "history_end_time") > as_of_time:
            errors.append("history time cannot extend beyond observation as-of time")
        source_frame_interval = timing.get("source_frame_interval")
        if source_frame_interval is not None:
            if len(source_frame_interval) != 2 or int(source_frame_interval[0]) > int(source_frame_interval[1]):
                errors.append("source frame interval is invalid")
            elif int(source_frame_interval[1]) > as_of_step:
                errors.append("source frame interval cannot extend beyond observation as-of")
        source_time_interval = timing.get("source_time_interval")
        if source_time_interval is not None:
            if len(source_time_interval) != 2 or _finite(source_time_interval[0], "source time start") > _finite(source_time_interval[1], "source time end"):
                errors.append("source time interval is invalid")
            elif _finite(source_time_interval[1], "source time end") > as_of_time:
                errors.append("source time interval cannot extend beyond observation as-of time")
        if int(timing.get("label_horizon_steps", 0)) < 1:
            errors.append("label horizon must be positive")
        expired = consumed_step > valid_until
        reasons = set(predictions.get("abstention_reasons", []))
        if expired and not {"snapshot_stale", "prediction_expired"}.issubset(reasons):
            errors.append("expired snapshot must be explicitly stale and expired")
        if not expired and "prediction_expired" in reasons:
            errors.append("fresh snapshot cannot be marked expired")
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"causal time fields are invalid: {error}")
    reasons = predictions.get("abstention_reasons", [])
    if not isinstance(reasons, list) or any(reason not in ABSTENTION_REASONS for reason in reasons):
        errors.append("snapshot contains an unsupported abstention reason")
    if bool(predictions.get("availability")) == bool(predictions.get("abstained")):
        errors.append("availability and abstention must be logical opposites")
    if int(predictions.get("availability_mask", -1)) != int(bool(predictions.get("availability"))):
        errors.append("availability mask conflicts with availability")
    if "predictor_unavailable" in reasons and predictions.get("handoff_probability") is not None:
        errors.append("unavailable predictor must expose null rather than a synthetic zero probability")
    for field in ("next_rsu_probability_distribution", "handoff_target_probability_distribution"):
        distribution = predictions.get(field)
        if distribution is not None:
            try:
                if not validate_probability_simplex(list(distribution.values()))["valid"]:
                    errors.append(f"{field} is not a probability simplex")
            except (TypeError, ValueError) as error:
                errors.append(f"{field} is invalid: {error}")
    handoff_probability = predictions.get("handoff_probability")
    if handoff_probability is not None:
        try:
            probability = _finite(handoff_probability, "handoff_probability")
            if probability < 0.0 or probability > 1.0:
                errors.append("handoff probability must be in [0, 1]")
        except ValueError as error:
            errors.append(str(error))
    if predictions.get("predicted_target_rsu_id") is not None and predictions.get("predicted_target_rsu_id") == predictions.get("current_rsu_id"):
        if "target_not_distinct" not in reasons:
            errors.append("non-distinct handoff target must trigger abstention")
    forbidden_runtime_fields = {"label", "future_label", "reward", "service_result", "oracle_action"}

    def walk(value: Any, path: str = "snapshot") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in forbidden_runtime_fields:
                    errors.append(f"forbidden runtime leakage field: {path}.{key}")
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(snapshot)
    return {
        "status": "pass" if not errors else "fail",
        "validator_version": SNAPSHOT_VALIDATOR_VERSION,
        "errors": errors,
        "snapshot_id": identity.get("snapshot_id"),
    }


def staleness_diagnostics(
    snapshots: Sequence[dict[str, Any]],
    *,
    update_intervals: Sequence[int] = (1, 3, 6, 12),
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for update_interval in update_intervals:
        interval = max(int(update_interval), 1)
        ages = [step % interval for step in range(max(len(snapshots), interval * 2))]
        rows.append(
            {
                "update_interval_steps": interval,
                "simulated_step_count": len(ages),
                "refresh_count": sum(age == 0 for age in ages),
                "snapshot_reuse_count": sum(age > 0 for age in ages),
                "maximum_age_steps": max(ages) if ages else None,
                "mean_age_steps": sum(ages) / len(ages) if ages else None,
                "prediction_drift": None,
                "calibration_by_staleness_bucket": None,
                "handoff_error_by_staleness": None,
                "target_error_by_staleness": None,
                "eta_error_by_staleness": None,
                "metric_availability_note": "requires realized future labels aligned to each historical snapshot",
            }
        )
    observed_ages = [snapshot.get("causal_time", {}).get("age_steps") for snapshot in snapshots]
    observed_ages = [int(age) for age in observed_ages if age is not None]
    return {
        "contract_version": CAUSAL_PREDICTOR_SNAPSHOT_CONTRACT_VERSION,
        "prediction_delay_definition": "consumed_at_step_minus_historical_generated_at_step",
        "delayed_snapshot_is_recomputed_with_future_information": False,
        "observed_snapshot_count": len(snapshots),
        "observed_age_min": min(observed_ages) if observed_ages else None,
        "observed_age_max": max(observed_ages) if observed_ages else None,
        "update_interval_rows": rows,
        "is_rl_performance_comparison": False,
    }
