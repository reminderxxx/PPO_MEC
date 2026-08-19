"""Pure calibration reducers and deterministic fit helpers for G12."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from statistics import median
from typing import Any, Iterable, Sequence


CALIBRATION_AUDIT_VERSION = "1.0.0"
RELIABILITY_BIN_CONTRACT_VERSION = "equal_width_10_v1.0.0"
RELIABILITY_BIN_EDGES = tuple(index / 10.0 for index in range(11))
TEMPERATURE_SCALING_VERSION = "deterministic_log_temperature_golden_v1.0.0"
SELECTIVE_PREDICTION_VERSION = "calibration_only_risk_coverage_v1.0.0"


def _finite_float(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _clip_probability(value: Any, field: str = "probability") -> float:
    probability = _finite_float(value, field)
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return probability


def validate_probability_simplex(
    probabilities: Sequence[Any],
    *,
    tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    values = [_clip_probability(value, f"probabilities[{index}]") for index, value in enumerate(probabilities)]
    total = sum(values)
    valid = bool(values) and math.isclose(total, 1.0, rel_tol=0.0, abs_tol=tolerance)
    return {
        "valid": valid,
        "class_count": len(values),
        "sum": total,
        "tolerance": tolerance,
    }


def reliability_bins(
    labels: Sequence[int | float],
    confidences: Sequence[float],
    *,
    correctness: Sequence[int | float] | None = None,
    small_bin_threshold: int = 5,
) -> dict[str, Any]:
    if len(labels) != len(confidences):
        raise ValueError("labels and confidences must have equal length")
    outcomes = list(correctness if correctness is not None else labels)
    if len(outcomes) != len(confidences):
        raise ValueError("correctness and confidences must have equal length")
    rows: list[dict[str, Any]] = []
    weighted_gap = 0.0
    maximum_gap: float | None = None
    for index, (lower, upper) in enumerate(zip(RELIABILITY_BIN_EDGES, RELIABILITY_BIN_EDGES[1:])):
        member_indices = [
            item_index
            for item_index, raw_confidence in enumerate(confidences)
            if (
                _clip_probability(raw_confidence, f"confidences[{item_index}]") >= lower
                and (
                    float(raw_confidence) < upper
                    or (index == len(RELIABILITY_BIN_EDGES) - 2 and float(raw_confidence) <= upper)
                )
            )
        ]
        if member_indices:
            mean_confidence = sum(float(confidences[item]) for item in member_indices) / len(member_indices)
            empirical = sum(float(outcomes[item]) for item in member_indices) / len(member_indices)
            gap = abs(mean_confidence - empirical)
            weighted_gap += len(member_indices) * gap
            maximum_gap = gap if maximum_gap is None else max(maximum_gap, gap)
        else:
            mean_confidence = None
            empirical = None
            gap = None
        rows.append(
            {
                "lower": lower,
                "upper": upper,
                "upper_inclusive": index == len(RELIABILITY_BIN_EDGES) - 2,
                "count": len(member_indices),
                "mean_confidence": mean_confidence,
                "empirical_accuracy_or_frequency": empirical,
                "absolute_gap": gap,
                "empty_bin": not member_indices,
                "small_bin_warning": bool(member_indices and len(member_indices) < small_bin_threshold),
            }
        )
    total = len(confidences)
    return {
        "contract_version": RELIABILITY_BIN_CONTRACT_VERSION,
        "bin_edges": list(RELIABILITY_BIN_EDGES),
        "rows": rows,
        "sample_count": total,
        "coverage": 1.0 if total else None,
        "ece_weighting": "sample_count_weighted_absolute_gap",
        "ece": weighted_gap / total if total else None,
        "maximum_calibration_error": maximum_gap,
        "empty_bin_count": sum(row["empty_bin"] for row in rows),
        "small_bin_count": sum(row["small_bin_warning"] for row in rows),
    }


def _binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positive_count = sum(label == 1 for label in labels)
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label == 1 for _, label in ordered[index:end])
        index = end
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def binary_calibration_metrics(
    labels: Sequence[int | float],
    probabilities: Sequence[float],
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have equal length")
    normalized_labels: list[int] = []
    normalized_probabilities: list[float] = []
    for index, (label, probability) in enumerate(zip(labels, probabilities)):
        label_value = int(label)
        if label_value not in {0, 1} or float(label) != float(label_value):
            raise ValueError(f"labels[{index}] must be binary")
        normalized_labels.append(label_value)
        normalized_probabilities.append(_clip_probability(probability, f"probabilities[{index}]"))
    count = len(normalized_labels)
    positive = sum(normalized_labels)
    negative = count - positive
    epsilon = 1.0e-15
    brier = (
        sum((probability - label) ** 2 for label, probability in zip(normalized_labels, normalized_probabilities)) / count
        if count
        else None
    )
    nll = (
        -sum(
            label * math.log(max(probability, epsilon))
            + (1 - label) * math.log(max(1.0 - probability, epsilon))
            for label, probability in zip(normalized_labels, normalized_probabilities)
        )
        / count
        if count
        else None
    )
    reliability = reliability_bins(normalized_labels, normalized_probabilities)
    threshold_metrics: dict[str, Any] | None = None
    if threshold is not None:
        threshold_value = _clip_probability(threshold, "threshold")
        predicted = [int(probability >= threshold_value) for probability in normalized_probabilities]
        true_positive = sum(prediction == 1 and label == 1 for prediction, label in zip(predicted, normalized_labels))
        false_positive = sum(prediction == 1 and label == 0 for prediction, label in zip(predicted, normalized_labels))
        false_negative = sum(prediction == 0 and label == 1 for prediction, label in zip(predicted, normalized_labels))
        true_negative = sum(prediction == 0 and label == 0 for prediction, label in zip(predicted, normalized_labels))
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        f1 = _ratio(2.0 * precision * recall, precision + recall) if precision is not None and recall is not None else None
        threshold_metrics = {
            "threshold": threshold_value,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "accuracy": _ratio(true_positive + true_negative, count),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "availability": "available" if count else "unavailable",
        "sample_count": count,
        "positive_count": positive,
        "negative_count": negative,
        "prevalence": _ratio(positive, count),
        "threshold_independent": {
            "brier_score": brier,
            "negative_log_likelihood": nll,
            "ece": reliability["ece"],
            "maximum_calibration_error": reliability["maximum_calibration_error"],
            "auroc": _binary_auc(normalized_labels, normalized_probabilities),
        },
        "threshold_dependent": threshold_metrics,
        "reliability": reliability,
    }


def _f1_by_class(labels: Sequence[int], predictions: Sequence[int], class_count: int) -> tuple[float | None, float | None]:
    f1_values: list[float] = []
    weighted_sum = 0.0
    total = len(labels)
    for class_index in range(class_count):
        true_positive = sum(label == class_index and prediction == class_index for label, prediction in zip(labels, predictions))
        false_positive = sum(label != class_index and prediction == class_index for label, prediction in zip(labels, predictions))
        false_negative = sum(label == class_index and prediction != class_index for label, prediction in zip(labels, predictions))
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        class_f1 = (
            _ratio(2.0 * precision * recall, precision + recall)
            if precision is not None and recall is not None and precision + recall > 0.0
            else 0.0
        )
        f1_values.append(float(class_f1))
        weighted_sum += float(class_f1) * sum(label == class_index for label in labels)
    return (
        sum(f1_values) / class_count if class_count else None,
        weighted_sum / total if total else None,
    )


def multiclass_calibration_metrics(
    labels: Sequence[int],
    probability_rows: Sequence[Sequence[float]],
    *,
    class_names: Sequence[str] | None = None,
    known_class_mask: Sequence[bool] | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    if len(labels) != len(probability_rows):
        raise ValueError("labels and probability rows must have equal length")
    class_count = len(probability_rows[0]) if probability_rows else len(class_names or [])
    names = list(class_names or [str(index) for index in range(class_count)])
    if len(names) != class_count:
        raise ValueError("class_names length must match probability dimension")
    normalized_rows: list[list[float]] = []
    simplex_failures: list[int] = []
    for row_index, row in enumerate(probability_rows):
        if len(row) != class_count:
            raise ValueError("all probability rows must have equal class count")
        values = [_clip_probability(value, f"probability_rows[{row_index}][{index}]") for index, value in enumerate(row)]
        if not validate_probability_simplex(values)["valid"]:
            simplex_failures.append(row_index)
        normalized_rows.append(values)
    for index, label in enumerate(labels):
        if int(label) < 0 or int(label) >= class_count:
            raise ValueError(f"labels[{index}] is an unknown class")
    predictions = [max(range(class_count), key=lambda index: row[index]) for row in normalized_rows] if class_count else []
    confidence = [row[prediction] for row, prediction in zip(normalized_rows, predictions)]
    correctness = [int(prediction == int(label)) for prediction, label in zip(predictions, labels)]
    count = len(labels)
    epsilon = 1.0e-15
    brier = (
        sum(
            sum((probability - float(class_index == int(label))) ** 2 for class_index, probability in enumerate(row))
            for label, row in zip(labels, normalized_rows)
        )
        / count
        if count
        else None
    )
    nll = (
        -sum(math.log(max(row[int(label)], epsilon)) for label, row in zip(labels, normalized_rows)) / count
        if count
        else None
    )
    macro_f1, weighted_f1 = _f1_by_class([int(label) for label in labels], predictions, class_count)
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for label, prediction in zip(labels, predictions):
        confusion[int(label)][prediction] += 1
    top_label_reliability = reliability_bins(labels, confidence, correctness=correctness)
    classwise: dict[str, Any] = {}
    for class_index, name in enumerate(names):
        class_labels = [int(int(label) == class_index) for label in labels]
        class_probabilities = [row[class_index] for row in normalized_rows]
        classwise[name] = reliability_bins(class_labels, class_probabilities)
    mask = list(known_class_mask or [True] * count)
    if len(mask) != count:
        raise ValueError("known_class_mask length must match labels")
    top_k_value = min(max(int(top_k), 1), max(class_count, 1))
    top_k_correct = sum(
        int(label) in sorted(range(class_count), key=lambda index: row[index], reverse=True)[:top_k_value]
        for label, row in zip(labels, normalized_rows)
    )
    return {
        "availability": "available" if count else "unavailable",
        "sample_count": count,
        "class_count": class_count,
        "top_1_accuracy": _ratio(sum(correctness), count),
        "top_k": top_k_value,
        "top_k_accuracy": _ratio(top_k_correct, count),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "multiclass_brier_score": brier,
        "negative_log_likelihood": nll,
        "top_label_ece": top_label_reliability["ece"],
        "top_label_reliability": top_label_reliability,
        "classwise_calibration": classwise,
        "confusion_matrix": {"class_names": names, "rows": confusion},
        "unseen_class_coverage": _ratio(sum(bool(value) for value in mask), count),
        "probability_simplex_validation": {
            "passed": not simplex_failures,
            "failure_count": len(simplex_failures),
            "failure_indices": simplex_failures,
        },
    }


def eta_regression_metrics(
    labels: Sequence[float | None],
    predictions: Sequence[float | None],
    *,
    intervals: Sequence[Sequence[float] | None] | None = None,
) -> dict[str, Any]:
    if len(labels) != len(predictions):
        raise ValueError("ETA labels and predictions must have equal length")
    if intervals is not None and len(intervals) != len(labels):
        raise ValueError("ETA intervals must have equal length")
    pairs: list[tuple[float, float, Sequence[float] | None]] = []
    for index, (label, prediction) in enumerate(zip(labels, predictions)):
        if label is None or prediction is None:
            continue
        label_value = _finite_float(label, f"labels[{index}]")
        prediction_value = _finite_float(prediction, f"predictions[{index}]")
        interval = intervals[index] if intervals is not None else None
        pairs.append((label_value, prediction_value, interval))
    errors = [prediction - label for label, prediction, _ in pairs]
    absolute = [abs(error) for error in errors]
    bucket_edges = (0.0, 2.0, 4.0, 7.0, 13.0, math.inf)
    bucket_rows: list[dict[str, Any]] = []
    for lower, upper in zip(bucket_edges, bucket_edges[1:]):
        members = [abs(prediction - label) for label, prediction, _ in pairs if label >= lower and label < upper]
        bucket_rows.append(
            {
                "lower": lower,
                "upper": None if math.isinf(upper) else upper,
                "count": len(members),
                "mae": sum(members) / len(members) if members else None,
            }
        )
    interval_members = [(label, interval) for label, _, interval in pairs if interval is not None]
    interval_coverage: float | None = None
    if interval_members:
        valid_count = 0
        for index, (label, interval) in enumerate(interval_members):
            if interval is None or len(interval) != 2:
                raise ValueError(f"intervals[{index}] must contain lower and upper")
            lower = _finite_float(interval[0], "interval lower")
            upper = _finite_float(interval[1], "interval upper")
            if lower > upper:
                raise ValueError("ETA interval lower cannot exceed upper")
            valid_count += int(lower <= label <= upper)
        interval_coverage = valid_count / len(interval_members)
    return {
        "availability": "available" if pairs else "unavailable",
        "sample_count": len(pairs),
        "missing_count": len(labels) - len(pairs),
        "mae": sum(absolute) / len(absolute) if absolute else None,
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else None,
        "median_absolute_error": median(absolute) if absolute else None,
        "interval_coverage": interval_coverage,
        "interval_is_classification_confidence": False,
        "eta_buckets": bucket_rows,
    }


def sigmoid(value: float) -> float:
    value = _finite_float(value, "logit")
    if value >= 0.0:
        factor = math.exp(-value)
        return 1.0 / (1.0 + factor)
    factor = math.exp(value)
    return factor / (1.0 + factor)


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    values = [_finite_float(value, f"logits[{index}]") / temperature for index, value in enumerate(logits)]
    if not values:
        raise ValueError("logits cannot be empty")
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def apply_binary_temperature(logits: Sequence[float], temperature: float) -> list[float]:
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    return [sigmoid(_finite_float(logit, f"logits[{index}]") / temperature) for index, logit in enumerate(logits)]


def apply_multiclass_temperature(logit_rows: Sequence[Sequence[float]], temperature: float) -> list[list[float]]:
    return [softmax(row, temperature) for row in logit_rows]


def _golden_section_minimize(objective: Any, lower: float = -4.0, upper: float = 4.0, iterations: int = 96) -> tuple[float, float]:
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value = objective(left)
    right_value = objective(right)
    for _ in range(iterations):
        if left_value <= right_value:
            upper = right
            right = left
            right_value = left_value
            left = upper - ratio * (upper - lower)
            left_value = objective(left)
        else:
            lower = left
            left = right
            left_value = right_value
            right = lower + ratio * (upper - lower)
            right_value = objective(right)
    point = (lower + upper) / 2.0
    return point, objective(point)


def fit_binary_temperature(labels: Sequence[int], logits: Sequence[float], *, seed: int = 0) -> dict[str, Any]:
    if len(labels) != len(logits) or not labels:
        raise ValueError("non-empty equal-length labels and logits are required")
    normalized_labels = [int(label) for label in labels]
    if any(label not in {0, 1} for label in normalized_labels):
        raise ValueError("binary temperature labels must be 0 or 1")

    def objective(log_temperature: float) -> float:
        probabilities = apply_binary_temperature(logits, math.exp(log_temperature))
        return float(binary_calibration_metrics(normalized_labels, probabilities)["threshold_independent"]["negative_log_likelihood"])

    log_temperature, objective_value = _golden_section_minimize(objective)
    temperature = math.exp(log_temperature)
    return {
        "method": "binary_temperature_scaling",
        "method_version": TEMPERATURE_SCALING_VERSION,
        "temperature": temperature,
        "objective": "negative_log_likelihood",
        "objective_value": objective_value,
        "converged": True,
        "seed": int(seed),
        "fit_sample_count": len(labels),
        "fit_split": "calibration",
    }


def fit_multiclass_temperature(labels: Sequence[int], logit_rows: Sequence[Sequence[float]], *, seed: int = 0) -> dict[str, Any]:
    if len(labels) != len(logit_rows) or not labels:
        raise ValueError("non-empty equal-length labels and logits are required")

    def objective(log_temperature: float) -> float:
        probabilities = apply_multiclass_temperature(logit_rows, math.exp(log_temperature))
        return float(multiclass_calibration_metrics(labels, probabilities)["negative_log_likelihood"])

    log_temperature, objective_value = _golden_section_minimize(objective)
    temperature = math.exp(log_temperature)
    return {
        "method": "multiclass_temperature_scaling",
        "method_version": TEMPERATURE_SCALING_VERSION,
        "temperature": temperature,
        "objective": "negative_log_likelihood",
        "objective_value": objective_value,
        "converged": True,
        "seed": int(seed),
        "fit_sample_count": len(labels),
        "fit_split": "calibration",
    }


def identity_calibration(task: str, sample_count: int) -> dict[str, Any]:
    return {
        "method": "identity",
        "method_version": CALIBRATION_AUDIT_VERSION,
        "task": str(task),
        "temperature": 1.0,
        "objective": None,
        "objective_value": None,
        "converged": True,
        "seed": 0,
        "fit_sample_count": int(sample_count),
        "fit_split": "calibration",
    }


def select_calibration_method(candidates: Sequence[dict[str, Any]], *, tolerance: float = 1.0e-12) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one calibration candidate is required")

    def key(candidate: dict[str, Any]) -> tuple[float, float, int, str]:
        metrics = candidate.get("metrics", {})
        nll = _finite_float(metrics.get("negative_log_likelihood"), "candidate NLL")
        brier = _finite_float(metrics.get("brier_score", metrics.get("multiclass_brier_score")), "candidate Brier")
        complexity = 0 if candidate.get("method") == "identity" else 1
        return (round(nll / tolerance) * tolerance, round(brier / tolerance) * tolerance, complexity, str(candidate.get("method")))

    selected = min(candidates, key=key)
    return {
        "selection_rule": "minimum_calibration_NLL_then_Brier_then_identity",
        "uses_rl_reward": False,
        "candidate_count": len(candidates),
        "selected_method": selected.get("method"),
        "selected": selected,
    }


def risk_coverage_curve(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    confidence: Sequence[float] | None = None,
    candidate_thresholds: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have equal length")
    confidences = list(confidence if confidence is not None else [max(probability, 1.0 - probability) for probability in probabilities])
    if len(confidences) != len(labels):
        raise ValueError("confidence must have equal length")
    thresholds = list(candidate_thresholds or [index / 20.0 for index in range(21)])
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        threshold_value = _clip_probability(threshold, "candidate threshold")
        accepted = [index for index, value in enumerate(confidences) if _clip_probability(value) >= threshold_value]
        accepted_labels = [labels[index] for index in accepted]
        accepted_probabilities = [probabilities[index] for index in accepted]
        accepted_metrics = binary_calibration_metrics(accepted_labels, accepted_probabilities, threshold=0.5)
        rows.append(
            {
                "confidence_threshold": threshold_value,
                "accepted_count": len(accepted),
                "accepted_coverage": _ratio(len(accepted), len(labels)),
                "abstention_rate": _ratio(len(labels) - len(accepted), len(labels)),
                "accepted_accuracy": (
                    accepted_metrics["threshold_dependent"]["accuracy"] if accepted_metrics["threshold_dependent"] else None
                ),
                "accepted_brier": accepted_metrics["threshold_independent"]["brier_score"],
                "selective_ece": accepted_metrics["threshold_independent"]["ece"],
            }
        )
    return rows


def select_abstention_threshold(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    confidence: Sequence[float] | None = None,
    minimum_coverage: float = 0.5,
    candidate_thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    minimum = _clip_probability(minimum_coverage, "minimum_coverage")
    rows = risk_coverage_curve(
        labels,
        probabilities,
        confidence=confidence,
        candidate_thresholds=candidate_thresholds,
    )
    eligible = [row for row in rows if row["accepted_coverage"] is not None and row["accepted_coverage"] >= minimum and row["accepted_brier"] is not None]
    selected = min(
        eligible,
        key=lambda row: (row["accepted_brier"], -row["accepted_coverage"], row["confidence_threshold"]),
    ) if eligible else None
    return {
        "selection_rule": "calibration_only_minimum_accepted_Brier_subject_to_minimum_coverage",
        "selection_version": SELECTIVE_PREDICTION_VERSION,
        "uses_rl_reward": False,
        "minimum_coverage": minimum,
        "candidate_grid": [row["confidence_threshold"] for row in rows],
        "selected_threshold": selected["confidence_threshold"] if selected else None,
        "calibration_only_result": selected,
        "risk_coverage_rows": rows,
        "small_sample_warning": len(labels) < 100,
    }


def canonical_sha256(payload: Any) -> str:
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _window_intervals(window: dict[str, Any]) -> dict[str, tuple[int, int]]:
    intervals: dict[str, tuple[int, int]] = {}
    if window.get("frame_offset") is not None and window.get("window_length") is not None:
        start = int(window["frame_offset"])
        intervals["frame_interval"] = (start, start + int(window["window_length"]) - 1)
    if window.get("time_index_start") is not None and window.get("time_index_end") is not None:
        intervals["time_interval"] = (int(window["time_index_start"]), int(window["time_index_end"]))
    if window.get("segment_frame_start") is not None and window.get("segment_frame_end") is not None:
        intervals["segment_frame_interval"] = (int(window["segment_frame_start"]), int(window["segment_frame_end"]))
    return intervals


def audit_predictor_splits(split_windows: dict[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    required = {"predictor_train", "calibration", "evaluation_dev"}
    if set(split_windows) != required:
        raise ValueError(f"predictor splits must be exactly {sorted(required)}")
    for split_name, windows in split_windows.items():
        if any(token in str(split_name).lower() for token in ("formal", "hidden", "holdout")):
            raise ValueError("formal/hidden/holdout predictor splits are forbidden")
        if not windows:
            raise ValueError(f"{split_name} must contain at least one window")
        for window in windows:
            declared_role = " ".join(
                str(window.get(field, ""))
                for field in ("split", "dataset_split", "source_split", "split_role")
            ).lower()
            if any(token in declared_role for token in ("formal", "hidden", "holdout")):
                raise ValueError("formal/hidden/holdout windows are forbidden")
    conflicts: list[dict[str, Any]] = []
    flattened = [(split_name, dict(window)) for split_name, windows in split_windows.items() for window in windows]
    for left_index, (left_split, left) in enumerate(flattened):
        for right_split, right in flattened[left_index + 1 :]:
            if left_split == right_split:
                continue
            left_segment = str(left.get("source_segment_id") or "")
            right_segment = str(right.get("source_segment_id") or "")
            if left_segment and right_segment and left_segment != right_segment:
                continue
            for interval_kind in sorted(set(_window_intervals(left)) & set(_window_intervals(right))):
                left_interval = _window_intervals(left)[interval_kind]
                right_interval = _window_intervals(right)[interval_kind]
                overlaps = left_interval[0] <= right_interval[1] and right_interval[0] <= left_interval[1]
                if overlaps:
                    conflicts.append(
                        {
                            "interval_kind": interval_kind,
                            "left_split": left_split,
                            "left_window_id": left.get("window_id"),
                            "left_interval": list(left_interval),
                            "right_split": right_split,
                            "right_window_id": right.get("window_id"),
                            "right_interval": list(right_interval),
                            "same_window_id": left.get("window_id") == right.get("window_id"),
                        }
                    )
    manifest = {
        "split_contract_version": "causal_predictor_three_way_split_v1.0.0",
        "splits": {name: [dict(window) for window in windows] for name, windows in sorted(split_windows.items())},
    }
    return {
        "passed": not conflicts,
        "checked_interval_kinds": ["frame_interval", "time_interval", "segment_frame_interval"],
        "window_count_by_split": {name: len(windows) for name, windows in split_windows.items()},
        "overlap_conflicts": conflicts,
        "different_window_id_is_independence_evidence": False,
        "split_manifest_sha256": canonical_sha256(manifest),
        "split_manifest": manifest,
    }


def audit_vehicle_group_overlap(rows_by_split: dict[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    vehicle_sets = {
        split_name: {str(row.get("vehicle_id")) for row in rows if row.get("vehicle_id") is not None}
        for split_name, rows in rows_by_split.items()
    }
    pairwise: list[dict[str, Any]] = []
    names = sorted(vehicle_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(vehicle_sets[left] & vehicle_sets[right])
            pairwise.append(
                {
                    "left_split": left,
                    "right_split": right,
                    "overlap_count": len(overlap),
                    "overlap_examples": overlap[:20],
                    "risk": "adjacent_trajectory_group_dependence" if overlap else "none_observed",
                }
            )
    return {
        "group_key": "vehicle_id",
        "pairwise": pairwise,
        "requires_interval_isolation_even_when_vehicle_repeats": True,
    }


def abstention_reason_counts(snapshots: Iterable[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for snapshot in snapshots:
        for reason in snapshot.get("predictions", {}).get("abstention_reasons", []):
            counter[str(reason)] += 1
    return dict(sorted(counter.items()))
