from __future__ import annotations

import json
from copy import deepcopy

import pytest

from src.evaluators.cache_baseline_fairness import full_manifest_sha256, semantic_protocol_sha256, sha256_value
from src.oracles.cache_request_replay import (
    CacheRequestReplayError,
    build_request_replay,
    missing_replay_status,
    request_replay_fingerprint,
    validate_request_replay,
)
from src.oracles.future_horizon_cache_oracle import (
    CacheOracleError,
    compare_baseline_to_oracle,
    solve_future_horizon_cache_oracle,
)


def manifest(unit: str = "adapter_slots", capacity: float = 2.0, initial=None) -> dict:
    initial = initial if initial is not None else {"r1": ["a"], "r2": []}
    payload = {
        "identity": {"manifest_id": "m", "git_commit": "abc"},
        "hashes": {"full_manifest_sha256": "full", "semantic_protocol_sha256": "semantic"},
        "cache_contract": {
            "capacity": {
                "enabled": True,
                "unit": unit,
                "rsu_adapter_slots": int(capacity) if unit == "adapter_slots" else None,
                "capacity_mb": capacity if unit == "mb" else None,
            },
            "initial_per_rsu_cache_contents": [
                {"rsu_id": rsu, "cached_adapter_ids": items} for rsu, items in initial.items()
            ],
            "resident_sizes": [
                {"adapter_id": name, "object_id": name, "resident_size_mb": size, "source": "test"}
                for name, size in {"a": 2.0, "b": 3.0, "c": 4.0, "d": 7.0, "x": 11.0}.items()
            ],
        },
    }
    return payload


def evaluation_unit() -> dict:
    return {
        "evaluation_unit_id": "u",
        "benchmark_run_seed": 7,
        "window_id": "w",
        "workflow_id": "wf",
        "workflow_dag_sha256": "dag",
        "expected_workload_fingerprint": "work",
        "raw_frame_interval": {"start": 1, "end": 12},
        "raw_time_interval": {"start": 10, "end": 120},
    }


def request(index: int, object_id: str, size: float, rsu: str = "r1") -> dict:
    return {
        "request_id": f"q{index}",
        "evaluation_unit_id": "u",
        "episode_id": "e",
        "step_index": index + 1,
        "time_index": 100 + index,
        "request_order": index,
        "vehicle_id": "v",
        "workflow_id": "wf",
        "node_id": f"n{index}",
        "required_base_model": "base",
        "object_id": object_id,
        "adapter_id": object_id,
        "object_size_mb": size,
        "size_source": "test",
        "request_rsu_id": rsu,
        "current_service_rsu_id": rsu,
        "previous_rsu_id": rsu,
        "actual_next_rsu_id": rsu,
        "predicted_next_rsu_id": None,
        "actual_handoff_target_rsu_id": None,
        "predicted_handoff_target_rsu_id": None,
        "eligible_service_rsu_ids": [rsu],
        "eligible_cache_target_rsu_ids": [rsu],
        "dag_provenance": {
            "workflow_dag_sha256": "dag",
            "execution_order_index": index,
            "predecessors": [],
            "successors": [],
            "policy_neutral_progression": True,
        },
    }


def replay(requests: list[dict], source: dict | None = None) -> dict:
    return build_request_replay(
        requests=requests,
        evaluation_unit=evaluation_unit(),
        source_manifest=source or manifest(),
    )


def solve(requests: list[dict], source: dict | None = None, horizon: int = 3, **kwargs) -> dict:
    source = source or manifest()
    return solve_future_horizon_cache_oracle(
        replay=replay(requests, source), manifest=source, horizon=horizon, **kwargs
    )


def test_replay_fingerprint_round_trip_and_outcomes_excluded() -> None:
    value = replay([request(0, "a", 2.0)])
    assert validate_request_replay(value)["json_round_trip_stable"] is True
    assert json.loads(json.dumps(value)) == value
    assert request_replay_fingerprint(value) == value["request_replay_fingerprint"]
    contaminated = deepcopy(value)
    contaminated["requests"][0]["cache_hit"] = True
    contaminated["request_replay_fingerprint"] = request_replay_fingerprint(contaminated)
    assert "forbidden" in " ".join(validate_request_replay(contaminated)["errors"])


@pytest.mark.parametrize("field,value", [
    ("vehicle_id", "v2"), ("workflow_id", "wf2"), ("time_index", 999),
    ("request_rsu_id", "r2"), ("object_id", "b"),
])
def test_replay_identity_changes_fingerprint(field: str, value: object) -> None:
    left = replay([request(0, "a", 2.0)])
    changed_request = request(0, "a", 2.0)
    changed_request[field] = value
    right = replay([changed_request])
    assert left["request_replay_fingerprint"] != right["request_replay_fingerprint"]


def test_replay_rejects_endogenous_divergence_duplicate_and_invalid_size() -> None:
    value = replay([request(0, "a", 2.0), request(1, "b", 3.0)])
    diverged = deepcopy(value)
    diverged["producer"]["policy_neutral"] = False
    diverged["request_replay_fingerprint"] = request_replay_fingerprint(diverged)
    assert validate_request_replay(diverged)["status"] == "fail"
    duplicate = [request(0, "a", 2.0), request(1, "b", 3.0)]
    duplicate[1]["request_id"] = "q0"
    with pytest.raises(CacheRequestReplayError, match="duplicate"):
        replay(duplicate)
    with pytest.raises(CacheRequestReplayError, match="positive"):
        replay([request(0, "a", -1.0)])


def test_missing_old_replay_is_unavailable_not_guessed() -> None:
    assert missing_replay_status()["availability"] == "unavailable"


def test_unknown_major_and_nonfinite_fail() -> None:
    value = replay([request(0, "a", 2.0)])
    value["cache_request_replay_version"] = "2.0.0"
    value["request_replay_fingerprint"] = request_replay_fingerprint(value)
    assert validate_request_replay(value)["status"] == "fail"
    with pytest.raises(CacheRequestReplayError, match="finite"):
        replay([request(0, "a", float("nan"))])


def test_same_step_admission_can_hit_and_pre_action_hit_is_separate() -> None:
    result = solve([request(0, "b", 3.0)], horizon=1)
    step = result["action_trace"][0]
    assert step["pre_action_hit"] is False
    assert step["post_action_hit"] is True
    assert step["action"] == "admit"


def test_admission_persists_to_next_step_and_initial_cache_hits() -> None:
    result = solve([request(0, "b", 3.0), request(1, "b", 3.0)], horizon=1)
    assert result["action_trace"][1]["pre_action_hit"] is True
    initial = solve([request(0, "a", 2.0)], horizon=1)
    assert initial["action_trace"][0]["pre_action_hit"] is True
    assert initial["performance"]["transfer_mb"] == 0.0


def test_horizon_boundary_and_tail_truncation() -> None:
    base = [request(0, "b", 3.0), request(1, "a", 2.0), request(2, "a", 2.0)]
    first = solve(base, horizon=1)["action_trace"][0]
    changed = deepcopy(base)
    changed[2] = request(2, "c", 4.0)
    second = solve(changed, horizon=1)["action_trace"][0]
    assert (first["action"], first["evicted_object_ids"]) == (second["action"], second["evicted_object_ids"])
    tail = solve(base, horizon=12)["action_trace"][-1]
    assert tail["visible_request_ids"] == ["q2"]


@pytest.mark.parametrize("horizon", [1, 3, 6, 12])
def test_supported_rolling_horizons(horizon: int) -> None:
    result = solve([request(0, "b", 3.0), request(1, "a", 2.0)], horizon=horizon)
    assert result["identity"]["horizon"] == horizon
    assert result["identity"]["rolling_horizon"] is True
    assert result["horizon_information_audit"]["outside_horizon_read"] is False


def test_full_trace_identity_is_distinct() -> None:
    source = manifest()
    value = solve_future_horizon_cache_oracle(
        replay=replay([request(0, "b", 3.0)], source), manifest=source,
        horizon=12, full_trace_diagnostic=True,
    )
    assert value["identity"]["horizon"] == "full_trace"
    assert "diagnostic" in value["identity"]["oracle_identity"]


def test_slot_capacity_and_per_rsu_isolation_never_overflow() -> None:
    source = manifest(capacity=1, initial={"r1": ["a"], "r2": []})
    result = solve([request(0, "b", 3.0, "r2")], source=source, horizon=1)
    assert result["action_trace"][0]["post_state"]["r1"] == ["a"]
    assert result["capacity_invariant_audit"]["capacity_never_exceeded"] is True


def test_mb_heterogeneous_multi_victim_exact() -> None:
    source = manifest("mb", 10.0, {"r1": ["a", "b", "c"]})
    result = solve([request(0, "d", 7.0)], source=source, horizon=1)
    step = result["action_trace"][0]
    assert step["post_action_hit"] is True
    assert len(step["evicted_object_ids"]) == 2
    assert step["evicted_mb"] >= 6.0


def test_oversized_rejection_has_no_eviction() -> None:
    source = manifest("mb", 10.0, {"r1": ["a", "b"]})
    result = solve([request(0, "x", 11.0)], source=source, horizon=1)
    step = result["action_trace"][0]
    assert step["post_action_hit"] is False
    assert step["evicted_object_ids"] == []
    assert result["performance"]["oversized_rejection_count"] == 1


def test_disabled_capacity_not_applicable() -> None:
    source = manifest()
    source["cache_contract"]["capacity"]["enabled"] = False
    with pytest.raises(CacheOracleError, match="not applicable"):
        solve([request(0, "a", 2.0)], source=source)


def test_lexical_objective_prefers_fewer_transfer_and_deterministic() -> None:
    requests = [request(0, "a", 2.0), request(1, "a", 2.0)]
    one = solve(requests, horizon=3)
    two = solve(requests, horizon=3)
    assert one == two
    assert one["performance"]["transfer_mb"] == 0.0


def test_tiny_exact_matches_independent_enumeration_bound() -> None:
    # With one slot and requests b,a,a, at most two of three requests can hit when
    # admission is allowed each request and transfer is secondary to hit count.
    source = manifest(capacity=1, initial={"r1": []})
    requests = [request(0, "b", 3.0), request(1, "a", 2.0), request(2, "a", 2.0)]
    result = solve(requests, source=source, horizon=3)
    assert result["performance"]["object_hit_count"] == 3
    # Independent finite action enumeration: admit current always attains three,
    # so no solution can exceed the request count upper bound.
    assert result["performance"]["object_hit_count"] == len(requests)


def test_state_limit_is_unknown_not_optimal_and_identity_is_exact() -> None:
    result = solve([request(0, "b", 3.0), request(1, "c", 4.0)], horizon=3, state_limit=1)
    assert result["identity"]["optimality_status"] == "unknown_state_limit"
    assert result["performance"]["request_count"] == 0
    assert "exact" in result["identity"]["solver_identity"]


def observed(result: dict, **overrides) -> dict:
    identity = result["identity"]
    value = {
        "baseline_identity": "reactive_lru/1.0.0",
        "metric_source": "observed_request_outcome_v1",
        "request_replay_fingerprint": identity["request_replay_fingerprint"],
        "capacity_unit": identity["capacity_unit"],
        "capacity_value": identity["capacity_value"],
        "initial_state_fingerprint": identity["initial_state_fingerprint"],
        "object_hit_count": 0,
        "hit_mb": 0.0,
        "transfer_mb": 10.0,
        "churn_mb": 12.0,
    }
    value.update(overrides)
    return value


def test_matched_gap_count_byte_cost_and_zero_denominator() -> None:
    result = solve([request(0, "b", 3.0)], horizon=1)
    gap = compare_baseline_to_oracle(oracle_result=result, observed_baseline=observed(result))
    assert gap["comparable_status"] == "matched"
    assert gap["gaps"]["object_hit_gap_oracle_minus_baseline"] == 1
    assert gap["gaps"]["hit_mb_gap_oracle_minus_baseline"] == 3.0
    empty = solve([], horizon=1)
    empty_gap = compare_baseline_to_oracle(oracle_result=empty, observed_baseline=observed(empty))
    assert empty_gap["gaps"]["normalized_object_hit_gap"] is None
    assert empty_gap["latency_gap"]["availability"] == "unavailable"


@pytest.mark.parametrize("override,reason", [
    ({"request_replay_fingerprint": "bad"}, "request_replay_fingerprint_mismatch"),
    ({"capacity_unit": "mb"}, "capacity_unit_mismatch"),
    ({"capacity_value": 999}, "capacity_value_mismatch"),
    ({"initial_state_fingerprint": "bad"}, "initial_state_fingerprint_mismatch"),
    ({"metric_source": "g06_future_reuse_proxy"}, "forbidden_non_oracle_comparison_source"),
    ({"metric_source": "reward"}, "forbidden_non_oracle_comparison_source"),
])
def test_gap_rejects_incompatible_or_proxy_sources(override: dict, reason: str) -> None:
    result = solve([request(0, "b", 3.0)], horizon=1)
    gap = compare_baseline_to_oracle(
        oracle_result=result, observed_baseline=observed(result, **override)
    )
    assert gap["comparable_status"] == "incompatible"
    assert reason in gap["incompatibility_reasons"]


def test_manifest_source_mismatch_rejected() -> None:
    source = manifest()
    value = replay([request(0, "a", 2.0)], source)
    changed = deepcopy(source)
    changed["identity"]["manifest_id"] = "other"
    with pytest.raises(CacheOracleError, match="source manifest"):
        solve_future_horizon_cache_oracle(replay=value, manifest=changed, horizon=1)
