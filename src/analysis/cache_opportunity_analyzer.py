"""G09 request/window cache opportunity analysis over strictly matched raw artifacts."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Iterable

from src.evaluators.cache_baseline_fairness import sha256_value
from src.oracles.cache_request_replay import validate_request_replay
from src.oracles.future_horizon_cache_oracle import (
    ALLOWED_HORIZONS,
    OBJECTIVE_IDENTITY,
    _capacity_value,
    _problem_from_manifest,
    _state_fingerprint,
)


CACHE_OPPORTUNITY_ANALYZER_CONTRACT_VERSION = "1.0.0"
ANALYZER_IDENTITY = "cache_opportunity_analyzer_v1.0.0"
PRIMARY_REASON_PRIORITY = (
    "unavailable_or_incomparable",
    "initial_cache_hit",
    "captured",
    "baseline_hit_oracle_miss",
    "right_censored",
    "oversized_infeasible",
    "topology_not_eligible",
    "compulsory_first_request",
    "no_reuse_within_horizon",
    "wrong_cache_target",
    "eviction_choice",
    "insufficient_free_capacity",
    "transfer_tradeoff",
    "admission_not_selected",
    "capacity_not_binding",
)
DEFAULT_CONFIG = {
    "reuse_horizons": [1, 3, 6, 12],
    "object_size_mb_boundaries": [32.0, 64.0, 128.0],
    "request_frequency_boundaries": [1, 3, 7],
    "reuse_distance_step_boundaries": [1, 3, 6, 12],
    "capacity_pressure_boundaries": [0.5, 0.85],
    "opportunity_density_strata": {"low_upper_exclusive": 0.25, "medium_upper_exclusive": 0.75},
    "concentration_top_k": [1, 3, 5],
    "small_sample_min_entities": 5,
}
FORBIDDEN_BASELINE_SOURCES = {"reward", "legacy_aggregate", "g06_future_reuse_proxy"}


class CacheOpportunityAnalyzerError(ValueError):
    """Raised when G09 inputs are invalid, incomplete, or incomparable."""


def _finite_nonnegative(value: Any, path: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CacheOpportunityAnalyzerError(f"{path} must be numeric") from exc
    if not math.isfinite(number) or number < 0 or (positive and number <= 0):
        qualifier = "positive" if positive else "nonnegative"
        raise CacheOpportunityAnalyzerError(f"{path} must be finite and {qualifier}")
    return number


def _safe_rate(numerator: float | int, denominator: float | int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _bucket(value: float | int | None, boundaries: list[float], labels: list[str]) -> str:
    if value is None:
        return "none"
    for boundary, label in zip(boundaries, labels):
        if float(value) <= boundary:
            return label
    return labels[-1]


def _size_bucket(value: float, config: dict[str, Any]) -> str:
    return _bucket(value, config["object_size_mb_boundaries"], ["le_32", "gt_32_le_64", "gt_64_le_128", "gt_128"])


def _frequency_bucket(value: int, config: dict[str, Any]) -> str:
    return _bucket(value, config["request_frequency_boundaries"], ["one", "two_to_three", "four_to_seven", "eight_plus"])


def _reuse_bucket(value: int | None, config: dict[str, Any]) -> str:
    return _bucket(value, config["reuse_distance_step_boundaries"], ["one", "two_to_three", "four_to_six", "seven_to_twelve", "thirteen_plus"])


def _pressure_bucket(value: float | None, config: dict[str, Any]) -> str:
    if value is None:
        return "unavailable"
    low, medium = config["capacity_pressure_boundaries"]
    return "low" if value < low else ("medium" if value < medium else "high")


def _gini(values: Iterable[float]) -> float | None:
    ordered = sorted(float(item) for item in values if float(item) >= 0)
    if not ordered or sum(ordered) == 0:
        return None
    n = len(ordered)
    return sum((2 * index - n - 1) * value for index, value in enumerate(ordered, 1)) / (n * sum(ordered))


def _manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_id": manifest["identity"]["manifest_id"],
        "full_manifest_sha256": manifest["hashes"]["full_manifest_sha256"],
        "semantic_protocol_sha256": manifest["hashes"]["semantic_protocol_sha256"],
        "git_commit": manifest["identity"]["git_commit"],
    }


def validate_analyzer_inputs(
    *,
    manifest: dict[str, Any],
    replay: dict[str, Any],
    oracle_results: dict[str, Any],
    oracle_action_traces: dict[str, Any],
    baseline_outcomes: dict[str, Any],
    horizons: Iterable[int],
) -> dict[str, Any]:
    errors: list[str] = []
    replay_report = validate_request_replay(replay, source_manifest=manifest)
    errors.extend(replay_report["errors"])
    horizon_values = sorted(set(int(item) for item in horizons))
    if not horizon_values or any(item not in ALLOWED_HORIZONS for item in horizon_values):
        errors.append(f"horizons must be a non-empty subset of {ALLOWED_HORIZONS}")
    try:
        state, _, capacity = _problem_from_manifest(replay, manifest)
        capacity_value = _capacity_value(capacity)
        initial_fp = _state_fingerprint(state)
    except Exception as exc:  # normalized into a single input report
        errors.append(str(exc))
        capacity = {}
        capacity_value = None
        initial_fp = None
    request_ids = [str(row.get("request_id")) for row in replay.get("requests", [])]
    if len(request_ids) != len(set(request_ids)):
        errors.append("duplicate request_id in replay")
    for key in (f"h_{item}" for item in horizon_values):
        result = oracle_results.get(key)
        trace = oracle_action_traces.get(key)
        if not isinstance(result, dict) or not isinstance(trace, list):
            errors.append(f"missing oracle result/action trace: {key}")
            continue
        identity = result.get("identity") or {}
        checks = {
            "request replay fingerprint": (identity.get("request_replay_fingerprint"), replay.get("request_replay_fingerprint")),
            "manifest semantic hash": (identity.get("g07_manifest_semantic_sha256"), manifest["hashes"]["semantic_protocol_sha256"]),
            "capacity unit": (identity.get("capacity_unit"), capacity.get("unit")),
            "capacity value": (identity.get("capacity_value"), capacity_value),
            "initial state fingerprint": (identity.get("initial_state_fingerprint"), initial_fp),
            "horizon": (identity.get("horizon"), int(key[2:])),
            "objective": (identity.get("objective_identity"), OBJECTIVE_IDENTITY),
            "optimality": (identity.get("optimality_status"), "optimal"),
        }
        for label, (actual, expected) in checks.items():
            if actual != expected:
                errors.append(f"{key} {label} mismatch: {actual!r} != {expected!r}")
        trace_ids = [str(row.get("request_id")) for row in trace]
        if trace_ids != request_ids:
            errors.append(f"{key} action trace cannot align exactly to replay")
    for baseline, outcome in sorted(baseline_outcomes.items()):
        if outcome.get("baseline_identity") != baseline:
            errors.append(f"baseline identity key mismatch: {baseline}")
        if outcome.get("metric_source") in FORBIDDEN_BASELINE_SOURCES:
            errors.append(f"{baseline} uses forbidden comparison source")
        checks = {
            "request replay fingerprint": (outcome.get("request_replay_fingerprint"), replay.get("request_replay_fingerprint")),
            "manifest ID": (outcome.get("g07_manifest_id"), manifest["identity"]["manifest_id"]),
            "manifest full hash": (outcome.get("g07_manifest_full_sha256"), manifest["hashes"]["full_manifest_sha256"]),
            "manifest semantic hash": (outcome.get("g07_manifest_semantic_sha256"), manifest["hashes"]["semantic_protocol_sha256"]),
            "capacity unit": (outcome.get("capacity_unit"), capacity.get("unit")),
            "capacity value": (outcome.get("capacity_value"), capacity_value),
            "initial state fingerprint": (outcome.get("initial_state_fingerprint"), initial_fp),
        }
        for label, (actual, expected) in checks.items():
            if actual != expected:
                errors.append(f"{baseline} {label} mismatch")
        rows = outcome.get("request_outcomes")
        if not isinstance(rows, list):
            errors.append(f"{baseline} missing raw request_outcomes")
            continue
        ids = [str(row.get("request_id")) for row in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"{baseline} duplicate request outcome ID")
        if ids != request_ids:
            errors.append(f"{baseline} request outcomes cannot align exactly to replay")
        event_ids = [row.get("event_id") for row in rows]
        if any(item is None for item in event_ids):
            errors.append(f"{baseline} missing raw event_id")
        if len(event_ids) != len(set(event_ids)):
            errors.append(f"{baseline} duplicate raw event_id")
        for index, row in enumerate(rows):
            missing = row.get("source_missing_fields") or []
            if missing:
                errors.append(f"{baseline}.request_outcomes[{index}] missing raw fields: {missing}")
            for field in ("adapter_transfer_size_mb", "state_migration_size_mb", "admitted_size_mb", "evicted_size_mb_sum"):
                try:
                    _finite_nonnegative(row.get(field), f"{baseline}.request_outcomes[{index}].{field}")
                except CacheOpportunityAnalyzerError as exc:
                    errors.append(str(exc))
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "required_artifacts": ["G07 fairness manifest", "G08 external request replay", "G08 exact oracle result and action trace", "matched baseline raw CacheEvent outcomes", "initial cache and capacity contract"],
        "request_count": len(request_ids),
        "baseline_count": len(baseline_outcomes),
        "horizons": horizon_values,
        "request_replay_fingerprint": replay.get("request_replay_fingerprint"),
        "manifest_semantic_sha256": manifest["hashes"]["semantic_protocol_sha256"],
    }


def _demand_rows(replay: dict[str, Any], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requests = replay["requests"]
    positions: dict[str, list[int]] = defaultdict(list)
    for index, request in enumerate(requests):
        size = _finite_nonnegative(request.get("object_size_mb"), f"requests[{index}].object_size_mb", positive=True)
        request["object_size_mb"] = size
        positions[str(request["object_id"])].append(index)
    frequency = Counter(str(row["object_id"]) for row in requests)
    last_step = int(requests[-1]["step_index"]) if requests else 0
    rows: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        object_id = str(request["object_id"])
        sequence = positions[object_id]
        sequence_index = sequence.index(index)
        previous_index = sequence[sequence_index - 1] if sequence_index > 0 else None
        next_index = sequence[sequence_index + 1] if sequence_index + 1 < len(sequence) else None
        previous = requests[previous_index] if previous_index is not None else None
        following = requests[next_index] if next_index is not None else None
        reuse_distance = int(request["step_index"]) - int(previous["step_index"]) if previous else None
        next_use = int(following["step_index"]) - int(request["step_index"]) if following else None
        current_rsu = request.get("current_service_rsu_id")
        prior_rsu = previous.get("current_service_rsu_id") if previous else None
        local_reuse = previous is not None and current_rsu == prior_rsu
        cross_reuse = previous is not None and current_rsu != prior_rsu
        reusable_topology = bool(previous) and bool(set(previous["eligible_cache_target_rsu_ids"]) & set(request["eligible_service_rsu_ids"]))
        handoff = bool(
            request.get("actual_handoff_target_rsu_id")
            or (request.get("request_rsu_id") is not None and request.get("request_rsu_id") != current_rsu)
            or (request.get("previous_rsu_id") is not None and request.get("previous_rsu_id") != current_rsu)
        )
        horizon_flags = {}
        for horizon in config["reuse_horizons"]:
            future_object_ids = [
                str(candidate["object_id"])
                for candidate in requests[index + 1 :]
                if int(candidate["step_index"]) - int(request["step_index"]) <= horizon
            ]
            horizon_flags[str(horizon)] = {
                "reuse_within_horizon": next_use is not None and next_use <= horizon,
                "right_censored": next_use is None and last_step - int(request["step_index"]) < horizon,
                "future_object_ids": future_object_ids,
            }
        rows.append({
            "request_id": request["request_id"],
            "evaluation_unit_id": request["evaluation_unit_id"],
            "workflow_id": request["workflow_id"],
            "object_id": object_id,
            "adapter_id": request["adapter_id"],
            "object_size_mb": request["object_size_mb"],
            "step_index": request["step_index"],
            "time_index": request["time_index"],
            "current_rsu_id": current_rsu,
            "request_rsu_id": request.get("request_rsu_id"),
            "actual_next_rsu_id": request.get("actual_next_rsu_id"),
            "eligible_service_rsu_ids": deepcopy(request["eligible_service_rsu_ids"]),
            "eligible_cache_target_rsu_ids": deepcopy(request["eligible_cache_target_rsu_ids"]),
            "eligible_service_rsu_breadth": len(request["eligible_service_rsu_ids"]),
            "eligible_cache_target_rsu_breadth": len(request["eligible_cache_target_rsu_ids"]),
            "first_request": previous is None,
            "repeated_request": previous is not None,
            "compulsory_cold_request": previous is None,
            "reuse_distance_steps": reuse_distance,
            "next_use_distance_steps": next_use,
            "rsu_local_reuse": local_reuse,
            "cross_rsu_reuse": cross_reuse,
            "cross_rsu_reuse_directly_usable": cross_reuse and reusable_topology,
            "topology_ineligible_reuse": cross_reuse and not reusable_topology,
            "handoff_adjacent": handoff,
            "handoff_adjacent_reuse": bool(previous) and handoff,
            "object_request_frequency": frequency[object_id],
            "object_size_bucket": _size_bucket(float(request["object_size_mb"]), config),
            "request_frequency_bucket": _frequency_bucket(frequency[object_id], config),
            "reuse_distance_bucket": _reuse_bucket(reuse_distance, config),
            "reuse_horizons": horizon_flags,
        })
    requested_mb = sum(float(row["object_size_mb"]) for row in rows)
    object_sizes = {row["object_id"]: float(row["object_size_mb"]) for row in rows}
    per_object = []
    for object_id in sorted(positions):
        object_rows = [row for row in rows if row["object_id"] == object_id]
        per_object.append({"object_id": object_id, "request_count": len(object_rows), "requested_mb": sum(row["object_size_mb"] for row in object_rows), "object_size_mb": object_sizes[object_id]})
    reuse = {}
    for horizon in config["reuse_horizons"]:
        available = [row for row in rows if not row["reuse_horizons"][str(horizon)]["right_censored"]]
        hits = [row for row in available if row["reuse_horizons"][str(horizon)]["reuse_within_horizon"]]
        reuse[str(horizon)] = {
            "availability": "available" if available else "unavailable",
            "unavailable_reason": None if available else "all_requests_right_censored",
            "required_fields": ["object_id", "step_index"],
            "available_request_count": len(available),
            "unavailable_request_count": len(rows) - len(available),
            "coverage": _safe_rate(len(available), len(rows)),
            "reuse_count": len(hits),
            "reuse_mb": sum(row["object_size_mb"] for row in hits),
            "reuse_rate": _safe_rate(len(hits), len(available)),
            "reuse_byte_rate": _safe_rate(sum(row["object_size_mb"] for row in hits), sum(row["object_size_mb"] for row in available)),
        }
    summary = {
        "availability": "available",
        "unavailable_reason": None,
        "required_fields": ["request_id", "object_id", "object_size_mb", "step_index", "current_service_rsu_id", "eligible_service_rsu_ids", "eligible_cache_target_rsu_ids"],
        "available_request_count": len(rows),
        "unavailable_request_count": 0,
        "coverage": 1.0 if rows else None,
        "request_count": len(rows),
        "requested_mb": requested_mb,
        "unique_object_count": len(object_sizes),
        "unique_object_mb": sum(object_sizes.values()),
        "first_request_count": sum(row["first_request"] for row in rows),
        "first_request_mb": sum(row["object_size_mb"] for row in rows if row["first_request"]),
        "repeated_request_count": sum(row["repeated_request"] for row in rows),
        "repeated_request_mb": sum(row["object_size_mb"] for row in rows if row["repeated_request"]),
        "compulsory_cold_request_count": sum(row["compulsory_cold_request"] for row in rows),
        "compulsory_cold_request_mb": sum(row["object_size_mb"] for row in rows if row["compulsory_cold_request"]),
        "rsu_local_reuse_count": sum(row["rsu_local_reuse"] for row in rows),
        "cross_rsu_reuse_count": sum(row["cross_rsu_reuse"] for row in rows),
        "handoff_adjacent_reuse_count": sum(row["handoff_adjacent_reuse"] for row in rows),
        "topology_ineligible_reuse_count": sum(row["topology_ineligible_reuse"] for row in rows),
        "reuse_within_horizon": reuse,
        "per_object": per_object,
        "object_size_concentration_gini_requested_mb": _gini([row["requested_mb"] for row in per_object]),
        "interpretation": "exogenous demand reuse; not necessarily a feasible cache hit",
    }
    return rows, summary


def _initial_objects(manifest: dict[str, Any]) -> set[str]:
    cache = manifest["cache_contract"]
    mapping = {str(row["adapter_id"]): str(row.get("object_id") or f"adapter:{row['adapter_id']}") for row in cache["resident_sizes"]}
    return {mapping.get(str(adapter), f"adapter:{adapter}") for row in cache["initial_per_rsu_cache_contents"] for adapter in row["cached_adapter_ids"]}


def _secondary_evidence(demand: dict[str, Any], oracle: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for condition, tag in (
        (demand["repeated_request"], "demand_reuse_observed"),
        (demand["rsu_local_reuse"], "rsu_local_reuse"),
        (demand["cross_rsu_reuse"], "cross_rsu_reuse"),
        (demand["handoff_adjacent"], "handoff_adjacent"),
        (demand["topology_ineligible_reuse"], "topology_ineligible_reuse"),
        (bool(oracle.get("evicted_object_ids")), "oracle_victim_replacement"),
        (len(oracle.get("evicted_object_ids") or []) > 1, "oracle_multi_victim"),
        (float(oracle.get("adapter_transfer_mb") or 0) > 0, "oracle_transfer_required"),
        (bool(baseline.get("eviction_occurred")), "baseline_eviction_observed"),
        (bool(baseline.get("admission_requested")), "baseline_admission_requested"),
    ):
        if condition:
            tags.append(tag)
    return sorted(tags)


def _primary_reason(
    *, demand: dict[str, Any], oracle: dict[str, Any], baseline: dict[str, Any], initial_objects: set[str], horizon: int
) -> str:
    oracle_hit = bool(oracle["post_action_hit"])
    baseline_hit = bool(baseline["cache_hit"])
    if oracle_hit and oracle.get("pre_action_hit") and demand["object_id"] in initial_objects and oracle.get("admitted_object_id") is None:
        return "initial_cache_hit"
    if oracle_hit and baseline_hit:
        return "captured"
    if baseline_hit and not oracle_hit:
        return "baseline_hit_oracle_miss"
    right_censored = demand["reuse_horizons"][str(horizon)]["right_censored"]
    if not oracle_hit and right_censored:
        return "right_censored"
    if oracle.get("rejection_reason") == "object_exceeds_total_capacity" or baseline.get("capacity_rejection_reason") == "object_exceeds_total_capacity":
        return "oversized_infeasible"
    if not demand["eligible_service_rsu_ids"] or not demand["eligible_cache_target_rsu_ids"]:
        return "topology_not_eligible"
    if not oracle_hit and demand["first_request"]:
        return "compulsory_first_request"
    if not oracle_hit and not demand["reuse_horizons"][str(horizon)]["reuse_within_horizon"]:
        return "no_reuse_within_horizon"
    if oracle_hit and not baseline_hit:
        oracle_target = oracle.get("cache_target_rsu_id")
        baseline_target = baseline.get("cache_target_rsu_id")
        legal = set(demand["eligible_cache_target_rsu_ids"])
        if oracle_target in legal and baseline_target in legal and baseline_target != oracle_target:
            return "wrong_cache_target"
        oracle_victims = set(oracle.get("evicted_object_ids") or [])
        baseline_victims = set(baseline.get("evicted_object_ids") or [])
        # This evidence is deliberately strict: different victims plus a later request for
        # a baseline victim inside H. It remains an explanatory diagnostic, not causal regret.
        if oracle_victims != baseline_victims and bool(
            baseline_victims & set(demand["reuse_horizons"][str(horizon)]["future_object_ids"])
        ):
            return "eviction_choice"
        if oracle_victims:
            return "insufficient_free_capacity"
        if float(baseline.get("adapter_transfer_size_mb") or 0) > float(oracle.get("adapter_transfer_mb") or 0):
            return "transfer_tradeoff"
        if not baseline.get("admission_requested") or not baseline.get("admission_added"):
            return "admission_not_selected"
        return "capacity_not_binding"
    return "unavailable_or_incomparable"


def _information_labels(row: dict[str, Any]) -> list[dict[str, str]]:
    labels = [
        ("current_object_identity_and_size", "decision-time observable"),
        ("current_rsu", "decision-time observable"),
        ("current_cache_contents", "decision-time observable"),
        ("remaining_capacity", "decision-time observable"),
    ]
    if row["demand"]["repeated_request"]:
        labels.append(("object_recency_frequency", "history-derived"))
    if row["demand"]["next_use_distance_steps"] is not None:
        labels.extend([("future_reuse_estimate", "predictor-required"), ("exact_future_reuse", "oracle-only future information")])
    if row["demand"]["handoff_adjacent"]:
        labels.append(("next_rsu_handoff_estimate", "predictor-required"))
    if row["demand"]["cross_rsu_reuse"]:
        labels.extend([("cross_rsu_cache_state", "currently absent/unknown"), ("multi_agent_coordination_information", "currently absent/unknown")])
    if row["oracle"]["adapter_transfer_mb"] > 0:
        labels.append(("transfer_cost", "decision-time observable"))
    labels.append(("dag_workflow_future_demand", "oracle-only future information"))
    return [{"information": name, "availability_class": category} for name, category in sorted(set(labels))]


def _aggregate(rows: list[dict[str, Any]], key_name: str | tuple[str, ...], key_fn) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    output = []
    for key in sorted(groups, key=lambda item: str(item)):
        values = groups[key]
        request_mb = sum(row["object_size_mb"] for row in values)
        oracle_hits = sum(row["oracle_hit"] for row in values)
        baseline_hits = sum(row["baseline_hit"] for row in values)
        oracle_mb = sum(row["object_size_mb"] for row in values if row["oracle_hit"])
        baseline_mb = sum(row["object_size_mb"] for row in values if row["baseline_hit"])
        dimensions = (
            dict(zip(key_name, key))
            if isinstance(key_name, tuple)
            else {key_name: key}
        )
        output.append({
            **dimensions,
            "request_count": len(values), "requested_mb": request_mb,
            "oracle_hit_count": oracle_hits, "oracle_hit_mb": oracle_mb,
            "baseline_hit_count": baseline_hits, "baseline_hit_mb": baseline_mb,
            "absolute_object_gap": oracle_hits - baseline_hits,
            "absolute_byte_gap_mb": oracle_mb - baseline_mb,
            "normalized_capture_rate": _safe_rate(sum(row["captured_opportunity"] for row in values), oracle_hits),
            "byte_opportunity_capture_rate": _safe_rate(
                sum(row["object_size_mb"] for row in values if row["captured_opportunity"]), oracle_mb
            ),
            "captured_opportunity_count": sum(row["captured_opportunity"] for row in values),
            "captured_opportunity_mb": sum(row["object_size_mb"] for row in values if row["captured_opportunity"]),
            "missed_opportunity_count": sum(row["missed_opportunity"] for row in values),
            "missed_opportunity_mb": sum(row["object_size_mb"] for row in values if row["missed_opportunity"]),
            "baseline_hit_and_oracle_hit_count": sum(row["baseline_hit_and_oracle_hit"] for row in values),
            "baseline_miss_and_oracle_hit_count": sum(row["baseline_miss_and_oracle_hit"] for row in values),
            "baseline_hit_and_oracle_miss_count": sum(row["baseline_hit_and_oracle_miss"] for row in values),
            "baseline_miss_and_oracle_miss_count": sum(row["baseline_miss_and_oracle_miss"] for row in values),
            "transfer_mb": sum(row["baseline_transfer_mb"] for row in values),
            "churn_mb": sum(row["baseline_churn_mb"] for row in values),
            "transfer_excess_baseline_minus_oracle_mb": sum(row["baseline_transfer_mb"] - row["oracle_transfer_mb"] for row in values),
            "churn_excess_baseline_minus_oracle_mb": sum(row["baseline_churn_mb"] - row["oracle_churn_mb"] for row in values),
            "availability": "available", "coverage": 1.0,
            "right_censored_count": sum(row["right_censored"] for row in values),
        })
    return output


def _concentration(group_rows: list[dict[str, Any]], key: str, top_k: list[int], minimum: int) -> dict[str, Any]:
    gaps = [max(float(row["absolute_object_gap"]), 0.0) for row in group_rows]
    total = sum(gaps)
    shares = {str(k): _safe_rate(sum(sorted(gaps, reverse=True)[:k]), total) for k in top_k}
    return {
        "entity": key,
        "entity_count": len(group_rows),
        "positive_gap_total": total,
        "top_k_gap_share": shares,
        "opportunity_gini": _gini(gaps),
        "coverage": 1.0 if group_rows else None,
        "warning": "small_sample_concentration_unstable" if len(group_rows) < minimum else None,
    }


def analyze_cache_opportunities(
    *, manifest: dict[str, Any], replay: dict[str, Any], oracle_results: dict[str, Any],
    oracle_action_traces: dict[str, Any], baseline_outcomes: dict[str, Any],
    horizons: Iterable[int] = ALLOWED_HORIZONS, config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = deepcopy(DEFAULT_CONFIG)
    if config:
        unknown = sorted(set(config) - set(resolved))
        if unknown:
            raise CacheOpportunityAnalyzerError(f"unknown analyzer config fields: {unknown}")
        resolved.update(deepcopy(config))
    horizon_values = sorted(set(int(item) for item in horizons))
    validation = validate_analyzer_inputs(manifest=manifest, replay=replay, oracle_results=oracle_results, oracle_action_traces=oracle_action_traces, baseline_outcomes=baseline_outcomes, horizons=horizon_values)
    if validation["status"] != "pass":
        raise CacheOpportunityAnalyzerError("; ".join(validation["errors"]))
    demand_rows, demand_summary = _demand_rows(deepcopy(replay), resolved)
    initial_objects = _initial_objects(manifest)
    request_rows: list[dict[str, Any]] = []
    oracle_by_horizon: dict[str, Any] = {}
    for horizon in horizon_values:
        key = f"h_{horizon}"
        trace = oracle_action_traces[key]
        result = oracle_results[key]
        capacity_unit = result["identity"]["capacity_unit"]
        capacity_value = float(result["identity"]["capacity_value"])
        oversized_flags = [
            capacity_unit == "mb" and float(row["object_size_mb"]) > capacity_value
            for row in demand_rows
        ]
        oracle_by_horizon[key] = {
            "availability": "available", "unavailable_reason": None,
            "required_fields": ["pre_action_hit", "post_action_hit", "action", "cache_target_rsu_id", "evicted_object_ids", "adapter_transfer_mb", "capacity_occupancy"],
            "available_request_count": len(trace), "unavailable_request_count": 0,
            "coverage": _safe_rate(len(trace), len(demand_rows)),
            "oracle_achievable_hit_count": sum(row["post_action_hit"] for row in trace),
            "oracle_achievable_hit_mb": sum(demand_rows[i]["object_size_mb"] for i, row in enumerate(trace) if row["post_action_hit"]),
            "initial_cache_natural_hit_count": sum(row["pre_action_hit"] and demand_rows[i]["object_id"] in initial_objects and row.get("admitted_object_id") is None for i, row in enumerate(trace)),
            "same_step_admission_hit_count": sum((not row["pre_action_hit"]) and row["post_action_hit"] and row.get("admitted_object_id") is not None for row in trace),
            "oracle_admission_opportunity_count": sum(row.get("admitted_object_id") is not None for row in trace),
            "oracle_eviction_required_opportunity_count": sum(bool(row.get("evicted_object_ids")) for row in trace),
            "oracle_noop_no_benefit_count": sum(row.get("action") == "noop" and not row["post_action_hit"] for row in trace),
            "capacity_binding_opportunity_count": sum(bool(row.get("evicted_object_ids")) for row in trace),
            "oversized_infeasible_request_count": sum(oversized_flags),
            "oversized_infeasible_request_mb": sum(demand_rows[i]["object_size_mb"] for i, flag in enumerate(oversized_flags) if flag),
            "transfer_required_opportunity_count": sum(float(row.get("adapter_transfer_mb") or 0) > 0 for row in trace),
            "multi_victim_required_opportunity_count": sum(len(row.get("evicted_object_ids") or []) > 1 for row in trace),
            "oracle_opportunity_density": _safe_rate(sum(row["post_action_hit"] for row in trace), len(trace)),
            "oracle_byte_opportunity_density": _safe_rate(sum(demand_rows[i]["object_size_mb"] for i, row in enumerate(trace) if row["post_action_hit"]), sum(row["object_size_mb"] for row in demand_rows)),
            "per_rsu_oracle_opportunity": [
                {
                    "rsu_id": rsu,
                    "request_count": sum(demand["current_rsu_id"] == rsu for demand in demand_rows),
                    "oracle_hit_count": sum(demand_rows[i]["current_rsu_id"] == rsu and row["post_action_hit"] for i, row in enumerate(trace)),
                    "oracle_hit_mb": sum(demand_rows[i]["object_size_mb"] for i, row in enumerate(trace) if demand_rows[i]["current_rsu_id"] == rsu and row["post_action_hit"]),
                }
                for rsu in sorted({str(row["current_rsu_id"]) for row in demand_rows})
            ],
            "latency_saved": deepcopy(result["performance"]["latency_saved"]),
        }
        for baseline_name, outcome in sorted(baseline_outcomes.items()):
            for index, (demand, oracle, baseline) in enumerate(zip(demand_rows, trace, outcome["request_outcomes"])):
                oracle_for_reason = deepcopy(oracle)
                if oversized_flags[index]:
                    oracle_for_reason["rejection_reason"] = "object_exceeds_total_capacity"
                reason = _primary_reason(demand=demand, oracle=oracle_for_reason, baseline=baseline, initial_objects=initial_objects, horizon=horizon)
                if reason not in PRIMARY_REASON_PRIORITY:
                    raise AssertionError(f"unfrozen primary reason: {reason}")
                occupancy = [float(row["occupancy_rate"]) for row in oracle.get("capacity_occupancy", {}).values()]
                oracle_hit, baseline_hit = bool(oracle["post_action_hit"]), bool(baseline["cache_hit"])
                row = {
                    "request_id": demand["request_id"], "baseline_identity": baseline_name,
                    "horizon": horizon, "capacity_unit": result["identity"]["capacity_unit"],
                    "capacity_value": result["identity"]["capacity_value"],
                    "evaluation_unit_id": demand["evaluation_unit_id"], "workflow_id": demand["workflow_id"],
                    "object_id": demand["object_id"], "adapter_id": demand["adapter_id"],
                    "object_size_mb": demand["object_size_mb"], "object_size_bucket": demand["object_size_bucket"],
                    "request_frequency_bucket": demand["request_frequency_bucket"], "reuse_distance_bucket": demand["reuse_distance_bucket"],
                    "current_rsu_id": demand["current_rsu_id"], "request_rsu_id": demand["request_rsu_id"],
                    "actual_next_rsu_id": demand["actual_next_rsu_id"], "handoff_adjacent": demand["handoff_adjacent"],
                    "capacity_pressure": max(occupancy) if occupancy else None,
                    "capacity_pressure_bucket": _pressure_bucket(max(occupancy) if occupancy else None, resolved),
                    "oracle_action_type": oracle["action"], "oracle_hit": oracle_hit, "baseline_hit": baseline_hit,
                    "baseline_hit_and_oracle_hit": baseline_hit and oracle_hit,
                    "baseline_miss_and_oracle_hit": (not baseline_hit) and oracle_hit,
                    "baseline_hit_and_oracle_miss": baseline_hit and not oracle_hit,
                    "baseline_miss_and_oracle_miss": (not baseline_hit) and not oracle_hit,
                    "captured_opportunity": baseline_hit and oracle_hit,
                    "missed_opportunity": (not baseline_hit) and oracle_hit,
                    "primary_opportunity_reason": reason,
                    "secondary_evidence": _secondary_evidence(demand, oracle, baseline),
                    "right_censored": demand["reuse_horizons"][str(horizon)]["right_censored"],
                    "oracle_transfer_mb": float(oracle.get("adapter_transfer_mb") or 0),
                    "oracle_churn_mb": float(oracle.get("cache_churn_mb") or 0),
                    "baseline_transfer_mb": float(baseline.get("adapter_transfer_size_mb") or 0) + float(baseline.get("state_migration_size_mb") or 0),
                    "baseline_churn_mb": float(baseline.get("admitted_size_mb") or 0) + float(baseline.get("evicted_size_mb_sum") or 0),
                    "demand": demand,
                    "oracle": {name: deepcopy(oracle.get(name)) for name in ("pre_action_hit", "post_action_hit", "action", "cache_target_rsu_id", "admitted_object_id", "evicted_object_ids", "rejection_reason", "adapter_transfer_mb")},
                    "baseline": deepcopy(baseline),
                    "availability": "available", "coverage": 1.0,
                }
                row["information_requirement_labels"] = _information_labels(row)
                request_rows.append(row)
    by_baseline = _aggregate(request_rows, ("baseline_identity", "horizon"), lambda row: (row["baseline_identity"], row["horizon"]))
    by_horizon = _aggregate(request_rows, "horizon", lambda row: row["horizon"])
    by_reason = _aggregate(request_rows, ("baseline_identity", "horizon", "primary_opportunity_reason"), lambda row: (row["baseline_identity"], row["horizon"], row["primary_opportunity_reason"]))
    by_object = _aggregate(request_rows, ("baseline_identity", "horizon", "object_id"), lambda row: (row["baseline_identity"], row["horizon"], row["object_id"]))
    by_window = _aggregate(request_rows, ("baseline_identity", "horizon", "evaluation_unit_id"), lambda row: (row["baseline_identity"], row["horizon"], row["evaluation_unit_id"]))
    by_rsu = _aggregate(request_rows, ("baseline_identity", "horizon", "rsu_id"), lambda row: (row["baseline_identity"], row["horizon"], row["current_rsu_id"]))
    concentration_by_object = _aggregate(request_rows, "object_id", lambda row: row["object_id"])
    concentration_by_window = _aggregate(request_rows, "evaluation_unit_id", lambda row: row["evaluation_unit_id"])
    concentration_by_rsu = _aggregate(request_rows, "rsu_id", lambda row: row["current_rsu_id"])
    taxonomy = Counter(row["primary_opportunity_reason"] for row in request_rows)
    expected = len(demand_rows) * len(horizon_values) * len(baseline_outcomes)
    reconciliation = {
        "status": "pass" if sum(taxonomy.values()) == expected and len(request_rows) == expected else "fail",
        "request_denominator": len(demand_rows), "horizon_count": len(horizon_values), "baseline_count": len(baseline_outcomes),
        "expected_request_analysis_rows": expected, "actual_request_analysis_rows": len(request_rows),
        "primary_reason_count_sum": sum(taxonomy.values()), "primary_reason_counts": dict(sorted(taxonomy.items())),
        "each_row_has_exactly_one_primary_reason": all(row["primary_opportunity_reason"] in PRIMARY_REASON_PRIORITY for row in request_rows),
    }
    if reconciliation["status"] != "pass":
        raise CacheOpportunityAnalyzerError("taxonomy denominator reconciliation failed")
    info_counter: Counter[tuple[str, str]] = Counter()
    for row in request_rows:
        info_counter.update((item["information"], item["availability_class"]) for item in row["information_requirement_labels"])
    information = [{"information": key[0], "availability_class": key[1], "request_analysis_row_count": count} for key, count in sorted(info_counter.items())]
    density = {key: value["oracle_opportunity_density"] for key, value in oracle_by_horizon.items()}
    thresholds = resolved["opportunity_density_strata"]
    strata = {key: ("low" if value is not None and value < thresholds["low_upper_exclusive"] else "medium" if value is not None and value < thresholds["medium_upper_exclusive"] else "high" if value is not None else "unavailable") for key, value in density.items()}
    summary = {
        "identity": {
            "cache_opportunity_analyzer_contract_version": CACHE_OPPORTUNITY_ANALYZER_CONTRACT_VERSION,
            "analyzer_identity": ANALYZER_IDENTITY,
            "analysis_fingerprint": None,
            "request_replay_fingerprint": replay["request_replay_fingerprint"],
            "g07_manifest": _manifest_identity(manifest),
            "oracle_contract_version": next(iter(oracle_results.values()))["identity"]["oracle_contract_version"],
            "oracle_objective_identity": OBJECTIVE_IDENTITY,
            "horizons": horizon_values,
        },
        "resolved_config": resolved,
        "primary_reason_priority": list(PRIMARY_REASON_PRIORITY),
        "gap_decomposition_dimensions": [
            "baseline_identity", "horizon", "capacity_unit", "capacity_value",
            "evaluation_unit_id", "workflow_id", "object_id", "adapter_id",
            "object_size_bucket", "request_frequency_bucket", "reuse_distance_bucket",
            "current_rsu_id", "request_rsu_id", "actual_next_rsu_id",
            "eligible_service_rsu_ids", "eligible_cache_target_rsu_ids",
            "handoff_adjacent", "capacity_pressure_bucket", "oracle_action_type",
            "primary_opportunity_reason",
        ],
        "demand_opportunity": demand_summary,
        "feasible_oracle_opportunity_by_horizon": oracle_by_horizon,
        "baseline_capture_loss": by_baseline,
        "opportunity_concentration": {
            "objects": _concentration(concentration_by_object, "object", resolved["concentration_top_k"], resolved["small_sample_min_entities"]),
            "windows": _concentration(concentration_by_window, "window", resolved["concentration_top_k"], resolved["small_sample_min_entities"]),
            "rsus": _concentration(concentration_by_rsu, "rsu", resolved["concentration_top_k"], resolved["small_sample_min_entities"]),
            "zero_opportunity_window_ratio": _safe_rate(sum(row["oracle_hit_count"] == 0 for row in concentration_by_window), len(concentration_by_window)),
            "fixed_opportunity_density_strata": strata,
        },
        "latency_saved": {"availability": "unavailable", "value": None, "unavailable_reason": "request-aligned observed and counterfactual latency components are absent", "required_fields": ["observed_service_latency", "cold_or_cloud_counterfactual_latency", "transfer_latency", "stall_restart_latency"]},
        "claim_boundary": "diagnostic placement opportunity association; not causal regret, latency gain, MARL necessity, or algorithm superiority",
    }
    fingerprint_payload = {"summary": summary, "request_rows": request_rows, "by_reason": by_reason, "by_object": by_object, "by_window": by_window, "by_rsu": by_rsu, "information": information, "reconciliation": reconciliation}
    summary["identity"]["analysis_fingerprint"] = sha256_value(fingerprint_payload)
    result_bundle = {
        "opportunity_summary": summary,
        "request_opportunity_rows": request_rows,
        "opportunity_by_baseline": by_baseline,
        "opportunity_by_horizon": by_horizon,
        "opportunity_by_reason": by_reason,
        "opportunity_by_object": by_object,
        "opportunity_by_window": by_window,
        "opportunity_by_rsu": by_rsu,
        "information_requirement_summary": {"availability": "available", "labels_are_not_information_sufficiency_conclusions": True, "rows": information},
        "input_validation_report": validation,
        "reconciliation_report": reconciliation,
    }
    json.loads(json.dumps(result_bundle, ensure_ascii=False, allow_nan=False))
    return result_bundle
