"""Exact rolling finite-horizon cache placement/admission/eviction oracle."""

from __future__ import annotations

import itertools
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from src.evaluators.cache_baseline_fairness import sha256_value
from src.oracles.cache_request_replay import validate_request_replay


CACHE_ORACLE_CONTRACT_VERSION = "future_horizon_cache_oracle_contract_v1.0.0"
SOLVER_IDENTITY = "exact_rolling_enumeration_v1.0.0"
OBJECTIVE_IDENTITY = "lex_hit_mb_hit_count_transfer_evicted_churn_v1.0.0"
ALLOWED_HORIZONS = (1, 3, 6, 12)
EPSILON = 1e-9


class CacheOracleError(ValueError):
    """Raised for an invalid or incomparable oracle problem."""


class _StateLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class _Action:
    action: str
    rsu_id: str | None
    object_id: str | None
    evicted_object_ids: tuple[str, ...] = ()
    rejection_reason: str | None = None

    @property
    def tie_key(self) -> tuple[str, str, str, tuple[str, ...]]:
        return (
            self.rsu_id or "",
            self.object_id or "",
            self.action,
            self.evicted_object_ids,
        )


def _capacity_value(capacity: dict[str, Any]) -> float:
    if capacity.get("enabled") is not True:
        raise CacheOracleError("capacity-disabled oracle is not applicable")
    unit = capacity.get("unit")
    if unit == "adapter_slots":
        value = capacity.get("rsu_adapter_slots")
    elif unit == "mb":
        value = capacity.get("capacity_mb")
    else:
        raise CacheOracleError(f"unsupported capacity unit: {unit!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise CacheOracleError("capacity value must be numeric") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise CacheOracleError("capacity value must be finite and positive")
    return numeric


def _state_key(state: dict[str, set[str]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple((rsu, tuple(sorted(objects))) for rsu, objects in sorted(state.items()))


def _state_fingerprint(state: dict[str, set[str]]) -> str:
    return sha256_value({rsu: sorted(objects) for rsu, objects in sorted(state.items())})


def _size(object_id: str, sizes: dict[str, float], unit: str) -> float:
    if object_id not in sizes:
        raise CacheOracleError(f"resident size unavailable for object: {object_id}")
    return 1.0 if unit == "adapter_slots" else sizes[object_id]


def _used(residents: set[str], sizes: dict[str, float], unit: str) -> float:
    return sum(_size(item, sizes, unit) for item in residents)


def _validate_state(
    state: dict[str, set[str]], sizes: dict[str, float], unit: str, capacity_value: float
) -> None:
    for rsu_id, residents in state.items():
        used = _used(residents, sizes, unit)
        if used > capacity_value + EPSILON:
            raise CacheOracleError(f"capacity invariant violated for {rsu_id}: {used} > {capacity_value}")


def _candidate_actions(
    request: dict[str, Any],
    state: dict[str, set[str]],
    sizes: dict[str, float],
    unit: str,
    capacity_value: float,
) -> list[_Action]:
    object_id = str(request["object_id"])
    actions = [_Action("noop", None, None)]
    object_size = _size(object_id, sizes, unit)
    for rsu_id in request["eligible_cache_target_rsu_ids"]:
        residents = state.setdefault(str(rsu_id), set())
        if object_id in residents:
            actions.append(_Action("already_resident", str(rsu_id), object_id))
            continue
        if object_size > capacity_value + EPSILON:
            actions.append(
                _Action(
                    "reject_oversized",
                    str(rsu_id),
                    object_id,
                    rejection_reason="object_exceeds_total_capacity",
                )
            )
            continue
        required = max(_used(residents, sizes, unit) + object_size - capacity_value, 0.0)
        if required <= EPSILON:
            actions.append(_Action("admit", str(rsu_id), object_id))
            continue
        ordered = sorted(residents)
        for count in range(1, len(ordered) + 1):
            for victims in itertools.combinations(ordered, count):
                freed = sum(_size(item, sizes, unit) for item in victims)
                if freed + EPSILON < required:
                    continue
                # Reject supersets with a still-sufficient proper subset. This matches atomic
                # minimum-sufficient planning while leaving victim identity to the oracle.
                if any(freed - _size(item, sizes, unit) + EPSILON >= required for item in victims):
                    continue
                actions.append(_Action("admit", str(rsu_id), object_id, tuple(victims)))
    return sorted(set(actions), key=lambda item: item.tie_key)


def _apply_action(
    action: _Action,
    state: dict[str, set[str]],
    sizes: dict[str, float],
    unit: str,
    capacity_value: float,
) -> tuple[dict[str, set[str]], float, float, bool]:
    next_state = {rsu: set(objects) for rsu, objects in state.items()}
    transfer_mb = 0.0
    evicted_mb = 0.0
    admitted = False
    if action.action == "admit" and action.rsu_id and action.object_id:
        residents = next_state.setdefault(action.rsu_id, set())
        for victim in action.evicted_object_ids:
            residents.remove(victim)
            evicted_mb += sizes[victim]
        residents.add(action.object_id)
        transfer_mb = sizes[action.object_id]
        admitted = True
    _validate_state(next_state, sizes, unit, capacity_value)
    return next_state, transfer_mb, evicted_mb, admitted


def _hit(request: dict[str, Any], state: dict[str, set[str]]) -> bool:
    object_id = str(request["object_id"])
    return any(object_id in state.get(str(rsu_id), set()) for rsu_id in request["eligible_service_rsu_ids"])


def _better(
    left: tuple[float, int, float, float, tuple[Any, ...]],
    right: tuple[float, int, float, float, tuple[Any, ...]] | None,
) -> bool:
    if right is None:
        return True
    left_primary = (round(left[0], 12), left[1], round(-left[2], 12), round(-left[3], 12))
    right_primary = (round(right[0], 12), right[1], round(-right[2], 12), round(-right[3], 12))
    if left_primary != right_primary:
        return left_primary > right_primary
    return left[4] < right[4]


def _plan_exact(
    requests: list[dict[str, Any]],
    state: dict[str, set[str]],
    sizes: dict[str, float],
    unit: str,
    capacity_value: float,
    state_limit: int,
) -> tuple[list[_Action], tuple[float, int, float, float, tuple[Any, ...]], int]:
    visited = 0
    memo: dict[tuple[int, tuple[Any, ...]], tuple[list[_Action], tuple[float, int, float, float, tuple[Any, ...]]]] = {}

    def recurse(index: int, current: dict[str, set[str]]) -> tuple[list[_Action], tuple[float, int, float, float, tuple[Any, ...]]]:
        nonlocal visited
        key = (index, _state_key(current))
        if key in memo:
            return memo[key]
        visited += 1
        if visited > state_limit:
            raise _StateLimit
        if index >= len(requests):
            return [], (0.0, 0, 0.0, 0.0, ())
        request = requests[index]
        best_actions: list[_Action] | None = None
        best_score: tuple[float, int, float, float, tuple[Any, ...]] | None = None
        for action in _candidate_actions(request, current, sizes, unit, capacity_value):
            next_state, transfer_mb, evicted_mb, _ = _apply_action(
                action, current, sizes, unit, capacity_value
            )
            hit = _hit(request, next_state)
            future_actions, future = recurse(index + 1, next_state)
            score = (
                (float(request["object_size_mb"]) if hit else 0.0) + future[0],
                int(hit) + future[1],
                transfer_mb + future[2],
                evicted_mb + future[3],
                (action.tie_key,) + future[4],
            )
            if _better(score, best_score):
                best_actions = [action, *future_actions]
                best_score = score
        assert best_actions is not None and best_score is not None
        memo[key] = (best_actions, best_score)
        return memo[key]

    actions, score = recurse(0, state)
    return actions, score, visited


def _problem_from_manifest(
    replay: dict[str, Any], manifest: dict[str, Any]
) -> tuple[dict[str, set[str]], dict[str, float], dict[str, Any]]:
    replay_report = validate_request_replay(replay, source_manifest=manifest)
    if replay_report["status"] != "pass":
        raise CacheOracleError("; ".join(replay_report["errors"]))
    cache = manifest["cache_contract"]
    capacity = deepcopy(cache["capacity"])
    _capacity_value(capacity)
    sizes: dict[str, float] = {}
    adapter_to_object: dict[str, str] = {}
    for row in cache["resident_sizes"]:
        adapter = str(row["adapter_id"])
        object_id = str(row.get("object_id") or f"adapter:{adapter}")
        size = float(row["resident_size_mb"])
        if not math.isfinite(size) or size <= 0:
            raise CacheOracleError(f"invalid resident size for {object_id}")
        sizes[object_id] = size
        adapter_to_object[adapter] = object_id
    for request in replay["requests"]:
        object_id = str(request["object_id"])
        size = float(request["object_size_mb"])
        if object_id in sizes and abs(sizes[object_id] - size) > EPSILON:
            raise CacheOracleError(f"request/catalog size mismatch for {object_id}")
        sizes[object_id] = size
    state: dict[str, set[str]] = {}
    for row in cache["initial_per_rsu_cache_contents"]:
        state[str(row["rsu_id"])] = {
            adapter_to_object.get(str(adapter), f"adapter:{adapter}")
            for adapter in row["cached_adapter_ids"]
        }
    for request in replay["requests"]:
        for rsu_id in request["eligible_service_rsu_ids"] + request["eligible_cache_target_rsu_ids"]:
            state.setdefault(str(rsu_id), set())
    _validate_state(state, sizes, str(capacity["unit"]), _capacity_value(capacity))
    initial_fp = sha256_value({rsu: sorted(objects) for rsu, objects in sorted(state.items())})
    if initial_fp != sha256_value({rsu: sorted(objects) for rsu, objects in sorted(state.items())}):
        raise AssertionError("unreachable fingerprint instability")
    return state, sizes, capacity


def solve_future_horizon_cache_oracle(
    *,
    replay: dict[str, Any],
    manifest: dict[str, Any],
    horizon: int,
    state_limit: int = 200_000,
    full_trace_diagnostic: bool = False,
) -> dict[str, Any]:
    if not full_trace_diagnostic and horizon not in ALLOWED_HORIZONS:
        raise CacheOracleError(f"horizon must be one of {ALLOWED_HORIZONS}")
    if state_limit <= 0:
        raise CacheOracleError("state_limit must be positive")
    if (
        replay.get("request_semantics", {}).get("model_cache_profile_id")
        == "typed_base_adapter_state_v1"
    ):
        replay_report = validate_request_replay(replay, source_manifest=manifest)
        if replay_report["status"] != "pass":
            raise CacheOracleError("; ".join(replay_report["errors"]))
        from src.oracles.typed_model_cache_oracle import solve_typed_model_cache_oracle

        return solve_typed_model_cache_oracle(
            replay=replay,
            manifest=manifest,
            horizon=horizon,
            state_limit=state_limit,
            full_trace_diagnostic=full_trace_diagnostic,
            contract_version=CACHE_ORACLE_CONTRACT_VERSION,
            solver_identity=SOLVER_IDENTITY,
            objective_identity=OBJECTIVE_IDENTITY,
        )
    state, sizes, capacity = _problem_from_manifest(replay, manifest)
    unit = str(capacity["unit"])
    cap_value = _capacity_value(capacity)
    initial_fingerprint = _state_fingerprint(state)
    requests = replay["requests"]
    trace: list[dict[str, Any]] = []
    total_visited = 0
    status = "optimal"
    for index, request in enumerate(requests):
        visible_end = len(requests) if full_trace_diagnostic else min(len(requests), index + horizon)
        visible = requests[index:visible_end]
        pre_state = {rsu: set(objects) for rsu, objects in state.items()}
        pre_hit = _hit(request, pre_state)
        try:
            actions, score, visited = _plan_exact(
                visible, pre_state, sizes, unit, cap_value, state_limit
            )
        except _StateLimit:
            status = "unknown_state_limit"
            break
        total_visited += visited
        action = actions[0]
        state, transfer_mb, evicted_mb, admitted = _apply_action(
            action, pre_state, sizes, unit, cap_value
        )
        post_hit = _hit(request, state)
        occupancy = {
            rsu: {
                "used": _used(objects, sizes, unit),
                "capacity": cap_value,
                "occupancy_rate": _used(objects, sizes, unit) / cap_value,
            }
            for rsu, objects in sorted(state.items())
        }
        trace.append(
            {
                "request_id": request["request_id"],
                "step_index": request["step_index"],
                "request_order": request["request_order"],
                "visible_request_ids": [item["request_id"] for item in visible],
                "visible_start_index": index,
                "visible_end_index_exclusive": visible_end,
                "pre_action_hit": pre_hit,
                "post_action_hit": post_hit,
                "action": action.action,
                "cache_target_rsu_id": action.rsu_id,
                "admitted_object_id": action.object_id if admitted else None,
                "evicted_object_ids": list(action.evicted_object_ids),
                "rejection_reason": action.rejection_reason,
                "adapter_transfer_mb": transfer_mb,
                "evicted_mb": evicted_mb,
                "cache_churn_mb": transfer_mb + evicted_mb,
                "rolling_plan_objective": {
                    "hit_mb": score[0],
                    "hit_count": score[1],
                    "transfer_mb": score[2],
                    "evicted_mb": score[3],
                },
                "post_state": {rsu: sorted(objects) for rsu, objects in sorted(state.items())},
                "post_state_fingerprint": _state_fingerprint(state),
                "capacity_occupancy": occupancy,
            }
        )
    hit_count = sum(int(item["post_action_hit"]) for item in trace)
    requested_mb = sum(float(item["object_size_mb"]) for item in requests[: len(trace)])
    hit_mb = sum(
        float(requests[index]["object_size_mb"])
        for index, item in enumerate(trace)
        if item["post_action_hit"]
    )
    occupancy_values = [
        row["occupancy_rate"]
        for item in trace
        for row in item["capacity_occupancy"].values()
    ]
    result = {
        "identity": {
            "oracle_contract_version": CACHE_ORACLE_CONTRACT_VERSION,
            "oracle_identity": (
                "full_trace_exact_diagnostic_v1.0.0"
                if full_trace_diagnostic
                else "rolling_finite_horizon_exact_v1.0.0"
            ),
            "solver_identity": SOLVER_IDENTITY,
            "objective_identity": OBJECTIVE_IDENTITY,
            "objective_lexicographic": [
                "maximize_future_cache_hit_mb",
                "maximize_future_cache_hit_count",
                "minimize_adapter_transfer_mb",
                "minimize_evicted_mb_and_cache_churn",
                "canonical_(rsu_id,object_id,action,victims)_tie_break",
            ],
            "horizon": "full_trace" if full_trace_diagnostic else horizon,
            "horizon_includes_current_request": True,
            "rolling_horizon": not full_trace_diagnostic,
            "admission_timing": "action_before_service_lookup_same_step_admission_can_hit",
            "optimality_status": status,
            "request_replay_fingerprint": replay["request_replay_fingerprint"],
            "g07_manifest_id": manifest["identity"]["manifest_id"],
            "g07_manifest_full_sha256": manifest["hashes"]["full_manifest_sha256"],
            "g07_manifest_semantic_sha256": manifest["hashes"]["semantic_protocol_sha256"],
            "git_commit": manifest["identity"]["git_commit"],
            "capacity_unit": unit,
            "capacity_value": cap_value,
            "initial_state_fingerprint": initial_fingerprint,
            "state_limit": state_limit,
            "states_visited": total_visited,
        },
        "performance": {
            "request_count": len(trace),
            "total_replay_request_count": len(requests),
            "object_hit_count": hit_count,
            "object_hit_rate": hit_count / len(trace) if trace else None,
            "requested_mb": requested_mb,
            "hit_mb": hit_mb,
            "byte_hit_rate": hit_mb / requested_mb if requested_mb > 0 else None,
            "admission_count": sum(item["admitted_object_id"] is not None for item in trace),
            "admission_mb": sum(item["adapter_transfer_mb"] for item in trace),
            "eviction_event_count": sum(bool(item["evicted_object_ids"]) for item in trace),
            "eviction_victim_count": sum(len(item["evicted_object_ids"]) for item in trace),
            "evicted_mb": sum(item["evicted_mb"] for item in trace),
            "transfer_mb": sum(item["adapter_transfer_mb"] for item in trace),
            "churn_mb": sum(item["cache_churn_mb"] for item in trace),
            "mean_capacity_occupancy": (
                sum(occupancy_values) / len(occupancy_values) if occupancy_values else None
            ),
            "peak_capacity_occupancy": max(occupancy_values) if occupancy_values else None,
            "oversized_rejection_count": sum(
                _size(str(request["object_id"]), sizes, unit) > cap_value + EPSILON
                for request in requests[: len(trace)]
            ),
            "latency_saved": {
                "availability": "unavailable",
                "value": None,
                "reason": "request-aligned observed and counterfactual latency components are absent",
            },
        },
        "action_trace": trace,
        "capacity_invariant_audit": {
            "status": "pass" if status == "optimal" else "incomplete",
            "checked_step_count": len(trace),
            "capacity_never_exceeded": all(
                row["used"] <= row["capacity"] + EPSILON
                for item in trace
                for row in item["capacity_occupancy"].values()
            ),
            "per_rsu_isolation": True,
        },
        "horizon_information_audit": {
            "status": "pass" if status == "optimal" else "incomplete",
            "full_trace_diagnostic": full_trace_diagnostic,
            "max_visible_request_count": max((len(item["visible_request_ids"]) for item in trace), default=0),
            "outside_horizon_read": False,
        },
    }
    return result


def compare_baseline_to_oracle(
    *, oracle_result: dict[str, Any], observed_baseline: dict[str, Any]
) -> dict[str, Any]:
    identity = oracle_result["identity"]
    reasons: list[str] = []
    if observed_baseline.get("request_replay_fingerprint") != identity["request_replay_fingerprint"]:
        reasons.append("request_replay_fingerprint_mismatch")
    if observed_baseline.get("capacity_unit") != identity["capacity_unit"]:
        reasons.append("capacity_unit_mismatch")
    if float(observed_baseline.get("capacity_value", math.nan)) != float(identity["capacity_value"]):
        reasons.append("capacity_value_mismatch")
    if observed_baseline.get("initial_state_fingerprint") != identity["initial_state_fingerprint"]:
        reasons.append("initial_state_fingerprint_mismatch")
    if observed_baseline.get("metric_source") in {"g06_future_reuse_proxy", "legacy_aggregate", "reward"}:
        reasons.append("forbidden_non_oracle_comparison_source")
    if reasons:
        return {
            "comparable_status": "incompatible",
            "incompatibility_reasons": reasons,
            "observed_baseline_identity": observed_baseline.get("baseline_identity"),
            "request_replay_fingerprint": identity["request_replay_fingerprint"],
            "gaps": None,
            "latency_gap": {"availability": "unavailable", "value": None},
        }
    oracle = oracle_result["performance"]
    baseline_hits = int(observed_baseline["object_hit_count"])
    baseline_hit_mb = float(observed_baseline["hit_mb"])
    baseline_transfer = float(observed_baseline["transfer_mb"])
    baseline_churn = float(observed_baseline["churn_mb"])
    hit_gap = int(oracle["object_hit_count"]) - baseline_hits
    byte_gap = float(oracle["hit_mb"]) - baseline_hit_mb
    return {
        "comparable_status": "matched",
        "incompatibility_reasons": [],
        "observed_baseline_identity": observed_baseline["baseline_identity"],
        "observed_request_fingerprint": observed_baseline["request_replay_fingerprint"],
        "oracle_identity": identity["oracle_identity"],
        "horizon": identity["horizon"],
        "gaps": {
            "object_hit_gap_oracle_minus_baseline": hit_gap,
            "hit_mb_gap_oracle_minus_baseline": byte_gap,
            "transfer_gap_baseline_minus_oracle_mb": baseline_transfer - float(oracle["transfer_mb"]),
            "churn_gap_baseline_minus_oracle_mb": baseline_churn - float(oracle["churn_mb"]),
            "normalized_object_hit_gap": (
                hit_gap / int(oracle["object_hit_count"])
                if int(oracle["object_hit_count"]) > 0
                else None
            ),
            "normalized_hit_mb_gap": byte_gap / float(oracle["hit_mb"]) if float(oracle["hit_mb"]) > 0 else None,
        },
        "interpretation": "matched placement opportunity gap; not causal regret or end-to-end latency gain",
        "g06_future_reuse_proxy": "separate_not_used",
        "latency_gap": {
            "availability": "unavailable",
            "value": None,
            "reason": "request-aligned observed and counterfactual latency components are absent",
        },
    }


def build_observed_baseline_outcome(
    *, replay: dict[str, Any], manifest: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    """Align a raw observed CacheEvent outcome to an already external replay."""
    typed_mode = (
        replay.get("request_semantics", {}).get("model_cache_profile_id")
        == "typed_base_adapter_state_v1"
    )
    if typed_mode:
        typed_contract = manifest["cache_contract"].get("typed_model_cache") or {}
        if typed_contract.get("catalog_fingerprint") != replay["requests"][0].get("catalog_fingerprint"):
            raise CacheOracleError("typed observed outcome catalog fingerprint mismatch")
        state = {
            str(item["rsu_id"]): set(map(str, item["resident_object_ids"]))
            for item in typed_contract.get("initial_typed_cache_contents", [])
        }
        capacity = deepcopy(manifest["cache_contract"]["capacity"])
    else:
        state, _, capacity = _problem_from_manifest(replay, manifest)
    run_info = summary.get("run_info") or {}
    if run_info.get("fairness_manifest_id") != manifest["identity"]["manifest_id"]:
        raise CacheOracleError("observed summary G07 manifest ID mismatch")
    if run_info.get("fairness_manifest_hash") != manifest["hashes"]["full_manifest_sha256"]:
        raise CacheOracleError("observed summary G07 manifest full hash mismatch")
    if run_info.get("fairness_semantic_protocol_hash") != manifest["hashes"]["semantic_protocol_sha256"]:
        raise CacheOracleError("observed summary G07 manifest semantic hash mismatch")
    if run_info.get("evaluation_unit_id") != replay["evaluation_unit"]["evaluation_unit_id"]:
        raise CacheOracleError("observed summary evaluation unit mismatch")
    events = [item for item in summary.get("cache_event_trace", []) if item.get("event_type") == "request"]
    requests = replay["requests"]
    if len(events) != len(requests):
        raise CacheOracleError("observed request count diverges from policy-neutral replay")
    for index, (request, event) in enumerate(zip(requests, events)):
        expected = (
            int(request["step_index"]),
            int(request["time_index"]),
            str(request["vehicle_id"]),
            str(request["workflow_id"]),
            str(request["node_id"]),
            str(request["object_id"]),
            str(request["adapter_id"]),
            round(float(request["object_size_mb"]), 9),
        )
        observed = (
            int(event.get("episode_step_index")),
            int(event.get("time_index")),
            str(event.get("vehicle_id")),
            str(event.get("workflow_id")),
            str(event.get("node_id")),
            str(event.get("object_id")),
            str(event.get("adapter_id")),
            round(float(event.get("size_mb")), 9),
        )
        if expected != observed:
            raise CacheOracleError(f"observed request divergence at index {index}")
        if event.get("served_rsu_id") is not None and event["served_rsu_id"] not in request["eligible_service_rsu_ids"]:
            raise CacheOracleError(f"observed service RSU outside replay feasibility at index {index}")
        if event.get("cache_target_rsu_id") is not None and event["cache_target_rsu_id"] not in request["eligible_cache_target_rsu_ids"]:
            raise CacheOracleError(f"observed cache target outside replay feasibility at index {index}")
    capacity_value = _capacity_value(capacity)
    return {
        "observed_outcome_contract_version": "observed_cache_baseline_outcome_v1.0.0",
        "baseline_identity": f"{run_info.get('agent_name')}/{run_info.get('eviction_policy')}",
        "metric_source": "observed_request_outcome_v1",
        "request_replay_fingerprint": replay["request_replay_fingerprint"],
        "g07_manifest_id": manifest["identity"]["manifest_id"],
        "g07_manifest_full_sha256": manifest["hashes"]["full_manifest_sha256"],
        "g07_manifest_semantic_sha256": manifest["hashes"]["semantic_protocol_sha256"],
        "capacity_unit": capacity["unit"],
        "capacity_value": capacity_value,
        "initial_state_fingerprint": _state_fingerprint(state),
        "object_hit_count": sum(
            bool(item.get("full_service_ready") if typed_mode else item.get("cache_hit"))
            for item in events
        ),
        "hit_mb": sum(
            float(item["size_mb"])
            for item in events
            if bool(item.get("full_service_ready") if typed_mode else item.get("cache_hit"))
        ),
        "transfer_mb": sum(
            (
                sum(float(value) for value in (item.get("transfer_mb_by_type") or {}).values())
                if typed_mode
                else float(item.get("adapter_transfer_size_mb") or 0.0)
                + float(item.get("state_migration_size_mb") or 0.0)
            )
            for item in events
        ),
        "churn_mb": sum(
            (
                sum(float(value) for value in (item.get("admitted_mb_by_type") or {}).values())
                + sum(float(value) for value in (item.get("evicted_mb_by_type") or {}).values())
                if typed_mode
                else float(item.get("admitted_size_mb") or 0.0)
                + float(item.get("evicted_size_mb_sum") or 0.0)
            )
            for item in events
        ),
        "model_cache_profile_id": (
            "typed_base_adapter_state_v1" if typed_mode else "legacy_adapter_only_v1"
        ),
        "request_alignment_status": "matched_external_replay",
        "legacy_observed_request_stream_fingerprint": run_info.get("observed_request_stream_fingerprint"),
        "reward_consumed": False,
        "aggregate_consumed": False,
        "request_outcomes": [
            {
                "request_id": request["request_id"],
                "event_id": event.get("event_id"),
                "cache_hit": bool(event.get("cache_hit")),
                "full_service_ready": (
                    bool(event.get("full_service_ready")) if typed_mode else None
                ),
                "hit_source": event.get("hit_source"),
                "cache_target_rsu_id": event.get("cache_target_rsu_id"),
                "served_rsu_id": event.get("served_rsu_id"),
                "admission_requested": bool(event.get("admission_requested")),
                "admission_added": bool(event.get("admission_added")),
                "admission_reason": event.get("admission_reason"),
                "capacity_rejection_reason": event.get("capacity_rejection_reason"),
                "eviction_occurred": bool(event.get("eviction_occurred")),
                "evicted_object_ids": list(event.get("evicted_object_ids") or []),
                "adapter_transfer_size_mb": (
                    float(event.get("adapter_transfer_size_mb") or 0.0)
                    if "adapter_transfer_size_mb" in event else None
                ),
                "state_migration_size_mb": (
                    float(event.get("state_migration_size_mb") or 0.0)
                    if "state_migration_size_mb" in event else None
                ),
                "transfer_mb_by_type": (
                    dict(event.get("transfer_mb_by_type") or {}) if typed_mode else None
                ),
                "admitted_mb_by_type": (
                    dict(event.get("admitted_mb_by_type") or {}) if typed_mode else None
                ),
                "evicted_mb_by_type": (
                    dict(event.get("evicted_mb_by_type") or {}) if typed_mode else None
                ),
                "admitted_size_mb": (
                    float(event.get("admitted_size_mb") or 0.0)
                    if "admitted_size_mb" in event else None
                ),
                "evicted_size_mb_sum": (
                    float(event.get("evicted_size_mb_sum") or 0.0)
                    if "evicted_size_mb_sum" in event else None
                ),
                "source_missing_fields": sorted(
                    field for field in (
                        "event_id", "cache_hit", "admission_requested", "admission_added",
                        "eviction_occurred", "evicted_object_ids", "adapter_transfer_size_mb",
                        "state_migration_size_mb", "admitted_size_mb", "evicted_size_mb_sum",
                        "transfer_mb_by_type", "admitted_mb_by_type", "evicted_mb_by_type",
                    ) if field not in event
                ),
            }
            for request, event in zip(requests, events)
        ],
    }
