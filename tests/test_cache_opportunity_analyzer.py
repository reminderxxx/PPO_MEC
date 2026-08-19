from __future__ import annotations

import json
from copy import deepcopy

import pytest

from src.analysis.cache_opportunity_analyzer import (
    CACHE_OPPORTUNITY_ANALYZER_CONTRACT_VERSION,
    DEFAULT_CONFIG,
    PRIMARY_REASON_PRIORITY,
    CacheOpportunityAnalyzerError,
    _primary_reason,
    analyze_cache_opportunities,
)
from src.oracles.cache_request_replay import build_request_replay, request_replay_fingerprint
from src.oracles.future_horizon_cache_oracle import (
    build_observed_baseline_outcome,
    solve_future_horizon_cache_oracle,
)


def manifest(unit="adapter_slots", capacity=2.0, initial=None):
    initial = initial if initial is not None else {"r1": ["a"], "r2": []}
    return {
        "identity": {"manifest_id": "m", "git_commit": "abc"},
        "hashes": {"full_manifest_sha256": "full", "semantic_protocol_sha256": "semantic"},
        "cache_contract": {
            "capacity": {"enabled": True, "unit": unit, "rsu_adapter_slots": int(capacity) if unit == "adapter_slots" else None, "capacity_mb": capacity if unit == "mb" else None},
            "initial_per_rsu_cache_contents": [{"rsu_id": rsu, "cached_adapter_ids": values} for rsu, values in initial.items()],
            "resident_sizes": [{"adapter_id": name, "object_id": name, "resident_size_mb": size, "source": "test"} for name, size in {"a": 2.0, "b": 3.0, "c": 4.0, "d": 7.0, "x": 11.0}.items()],
        },
    }


def unit():
    return {"evaluation_unit_id": "u", "benchmark_run_seed": 7, "window_id": "w", "workflow_id": "wf", "workflow_dag_sha256": "dag", "expected_workload_fingerprint": "work", "raw_frame_interval": {"start": 1, "end": 20}, "raw_time_interval": {"start": 10, "end": 200}}


def request(index, object_id, size, rsu="r1", *, service=None, target=None, handoff=False):
    service = service if service is not None else [rsu]
    target = target if target is not None else [rsu]
    return {
        "request_id": f"q{index}", "evaluation_unit_id": "u", "episode_id": "e",
        "step_index": index + 1, "time_index": 100 + index, "request_order": index,
        "vehicle_id": "v", "workflow_id": "wf", "node_id": f"n{index}",
        "required_base_model": "base", "object_id": object_id, "adapter_id": object_id,
        "object_size_mb": size, "size_source": "test", "request_rsu_id": "r1" if handoff else rsu,
        "current_service_rsu_id": rsu, "previous_rsu_id": "r1", "actual_next_rsu_id": "r2" if handoff else rsu,
        "predicted_next_rsu_id": None, "actual_handoff_target_rsu_id": rsu if handoff else None,
        "predicted_handoff_target_rsu_id": None, "eligible_service_rsu_ids": service,
        "eligible_cache_target_rsu_ids": target,
        "dag_provenance": {"workflow_dag_sha256": "dag", "execution_order_index": index, "predecessors": [], "successors": [], "policy_neutral_progression": True},
    }


def replay(rows, source):
    return build_request_replay(requests=rows, evaluation_unit=unit(), source_manifest=source)


def outcome_for(value, result, name="reactive_lru/lru", hits=None, targets=None, victims=None):
    hits = hits if hits is not None else [row["post_action_hit"] for row in result["action_trace"]]
    targets = targets or [row["cache_target_rsu_id"] for row in result["action_trace"]]
    victims = victims or [[] for _ in hits]
    identity = result["identity"]
    request_outcomes = []
    for index, hit in enumerate(hits):
        request_outcomes.append({
            "request_id": value["requests"][index]["request_id"], "event_id": f"e{index}",
            "cache_hit": hit, "hit_source": "current_rsu" if hit else "cloud",
            "cache_target_rsu_id": targets[index], "served_rsu_id": value["requests"][index]["current_service_rsu_id"],
            "admission_requested": bool(targets[index]), "admission_added": bool(targets[index]),
            "admission_reason": None, "capacity_rejection_reason": None,
            "eviction_occurred": bool(victims[index]), "evicted_object_ids": victims[index],
            "adapter_transfer_size_mb": 0.0, "state_migration_size_mb": 0.0,
            "admitted_size_mb": 0.0, "evicted_size_mb_sum": 0.0,
        })
    return {
        "observed_outcome_contract_version": "observed_cache_baseline_outcome_v1.0.0",
        "baseline_identity": name, "metric_source": "observed_request_outcome_v1",
        "request_replay_fingerprint": identity["request_replay_fingerprint"],
        "g07_manifest_id": "m", "g07_manifest_full_sha256": "full", "g07_manifest_semantic_sha256": "semantic",
        "capacity_unit": identity["capacity_unit"], "capacity_value": identity["capacity_value"],
        "initial_state_fingerprint": identity["initial_state_fingerprint"],
        "object_hit_count": sum(hits), "hit_mb": sum(row["object_size_mb"] for row, hit in zip(value["requests"], hits) if hit),
        "transfer_mb": 0.0, "churn_mb": 0.0, "request_alignment_status": "matched_external_replay",
        "reward_consumed": False, "aggregate_consumed": False, "request_outcomes": request_outcomes,
    }


def bundle(rows, source=None, baselines=1, horizons=(1, 3, 6, 12), hit_overrides=None):
    source = source or manifest()
    value = replay(rows, source)
    results, traces = {}, {}
    for horizon in horizons:
        solved = solve_future_horizon_cache_oracle(replay=value, manifest=source, horizon=horizon)
        results[f"h_{horizon}"] = {"identity": solved["identity"], "performance": solved["performance"]}
        traces[f"h_{horizon}"] = solved["action_trace"]
    first = solve_future_horizon_cache_oracle(replay=value, manifest=source, horizon=horizons[0])
    outcomes = {}
    names = ["reactive_lru/lru", "reactive_fifo/fifo", "reactive_lfu/lfu", "reactive_aging_lfu/aging_lfu", "reactive_random/random"]
    for index in range(baselines):
        hits = hit_overrides[index] if hit_overrides else None
        outcomes[names[index]] = outcome_for(value, first, names[index], hits=hits)
    return source, value, results, traces, outcomes


def analyze(rows, **kwargs):
    source, value, results, traces, outcomes = bundle(rows, **kwargs)
    return analyze_cache_opportunities(manifest=source, replay=value, oracle_results=results, oracle_action_traces=traces, baseline_outcomes=outcomes)


def test_demand_first_repeat_distance_horizons_and_tail_censoring():
    result = analyze([request(0, "b", 3), request(1, "c", 4), request(2, "b", 3)])
    demand = result["opportunity_summary"]["demand_opportunity"]
    assert demand["first_request_count"] == 2
    assert demand["repeated_request_count"] == 1
    rows = result["request_opportunity_rows"]
    q2 = next(row for row in rows if row["request_id"] == "q2" and row["horizon"] == 3)
    assert q2["demand"]["reuse_distance_steps"] == 2
    assert demand["reuse_within_horizon"]["1"]["reuse_count"] == 0
    assert demand["reuse_within_horizon"]["3"]["reuse_count"] == 1
    assert any(row["right_censored"] for row in rows)


def test_heterogeneous_bytes_local_cross_handoff_and_topology_ineligible():
    rows = [
        request(0, "b", 3, "r1", target=["r1"]),
        request(1, "b", 3, "r1", target=["r1"]),
        request(2, "b", 3, "r2", service=["r2"], target=["r2"], handoff=True),
        request(3, "c", 4, "r2"),
    ]
    result = analyze(rows)
    demand = result["opportunity_summary"]["demand_opportunity"]
    assert demand["requested_mb"] == 13
    assert demand["rsu_local_reuse_count"] == 1
    assert demand["cross_rsu_reuse_count"] == 1
    assert demand["handoff_adjacent_reuse_count"] >= 1
    assert demand["topology_ineligible_reuse_count"] == 1


def test_oracle_natural_same_step_eviction_multivictim_oversized_and_noop():
    natural = analyze([request(0, "a", 2)])
    assert natural["opportunity_summary"]["feasible_oracle_opportunity_by_horizon"]["h_1"]["initial_cache_natural_hit_count"] == 1
    same = analyze([request(0, "b", 3)])
    assert same["opportunity_summary"]["feasible_oracle_opportunity_by_horizon"]["h_1"]["same_step_admission_hit_count"] == 1
    mb = manifest("mb", 10, {"r1": ["a", "b", "c"], "r2": []})
    multi = analyze([request(0, "d", 7)], source=mb)
    assert multi["opportunity_summary"]["feasible_oracle_opportunity_by_horizon"]["h_1"]["multi_victim_required_opportunity_count"] == 1
    oversized = analyze([request(0, "x", 11)], source=manifest("mb", 10, {"r1": [], "r2": []}))
    oracle = oversized["opportunity_summary"]["feasible_oracle_opportunity_by_horizon"]["h_1"]
    assert oracle["oversized_infeasible_request_count"] == 1
    assert oracle["oracle_noop_no_benefit_count"] in (0, 1)


def test_per_rsu_isolation_and_fixed_buckets():
    result = analyze([request(0, "b", 3, "r2"), request(1, "a", 2, "r1")])
    per_rsu = result["opportunity_summary"]["feasible_oracle_opportunity_by_horizon"]["h_1"]["per_rsu_oracle_opportunity"]
    assert {row["rsu_id"] for row in per_rsu} == {"r1", "r2"}
    assert result["opportunity_summary"]["resolved_config"]["object_size_mb_boundaries"] == [32.0, 64.0, 128.0]


def test_baseline_quadrants_object_byte_cost_and_five_schema():
    rows = [request(0, "a", 2), request(1, "b", 3), request(2, "c", 4)]
    source, value, results, traces, outcomes = bundle(rows, baselines=5, hit_overrides=[[True, False, False], [False, True, False], [True, True, False], [False, False, False], [True, False, True]])
    result = analyze_cache_opportunities(manifest=source, replay=value, oracle_results=results, oracle_action_traces=traces, baseline_outcomes=outcomes)
    schemas = [{key for key in row if key != "baseline_identity"} for row in result["opportunity_by_baseline"]]
    assert len(result["opportunity_by_baseline"]) == 20
    assert len({row["baseline_identity"] for row in result["opportunity_by_baseline"]}) == 5
    assert all(item == schemas[0] for item in schemas)
    assert any(row["absolute_object_gap"] != row["absolute_byte_gap_mb"] for row in result["opportunity_by_baseline"])
    assert all("transfer_excess_baseline_minus_oracle_mb" in row and "churn_excess_baseline_minus_oracle_mb" in row for row in result["opportunity_by_baseline"])


def _taxonomy_inputs(reason):
    demand = {
        "object_id": "b", "first_request": False, "eligible_service_rsu_ids": ["r1"], "eligible_cache_target_rsu_ids": ["r1", "r2"],
        "next_use_distance_steps": 1, "reuse_horizons": {"3": {"right_censored": False, "reuse_within_horizon": True, "future_object_ids": ["a"]}},
    }
    oracle = {"post_action_hit": False, "pre_action_hit": False, "admitted_object_id": None, "rejection_reason": None, "cache_target_rsu_id": "r1", "evicted_object_ids": [], "adapter_transfer_mb": 0}
    baseline = {"cache_hit": False, "capacity_rejection_reason": None, "cache_target_rsu_id": None, "evicted_object_ids": [], "adapter_transfer_size_mb": 0, "admission_requested": True, "admission_added": True}
    initial = set()
    if reason == "initial_cache_hit": oracle.update(post_action_hit=True, pre_action_hit=True); initial.add("b")
    elif reason == "captured": oracle["post_action_hit"] = True; baseline["cache_hit"] = True
    elif reason == "baseline_hit_oracle_miss": baseline["cache_hit"] = True
    elif reason == "right_censored": demand["reuse_horizons"]["3"]["right_censored"] = True
    elif reason == "oversized_infeasible": oracle["rejection_reason"] = "object_exceeds_total_capacity"
    elif reason == "topology_not_eligible": demand["eligible_service_rsu_ids"] = []
    elif reason == "compulsory_first_request": demand["first_request"] = True
    elif reason == "no_reuse_within_horizon": demand["reuse_horizons"]["3"]["reuse_within_horizon"] = False
    elif reason == "wrong_cache_target": oracle["post_action_hit"] = True; baseline["cache_target_rsu_id"] = "r2"
    elif reason == "eviction_choice": oracle.update(post_action_hit=True, evicted_object_ids=["c"]); baseline["evicted_object_ids"] = ["a"]
    elif reason == "insufficient_free_capacity": oracle.update(post_action_hit=True, evicted_object_ids=["c"])
    elif reason == "transfer_tradeoff": oracle.update(post_action_hit=True, adapter_transfer_mb=1); baseline["adapter_transfer_size_mb"] = 2
    elif reason == "admission_not_selected": oracle["post_action_hit"] = True; baseline["admission_requested"] = False
    elif reason == "capacity_not_binding": oracle["post_action_hit"] = True
    elif reason == "unavailable_or_incomparable": demand["reuse_horizons"]["3"]["reuse_within_horizon"] = True
    return demand, oracle, baseline, initial


@pytest.mark.parametrize("reason", PRIMARY_REASON_PRIORITY)
def test_every_primary_reason_is_deterministic_and_exclusive(reason):
    demand, oracle, baseline, initial = _taxonomy_inputs(reason)
    assert _primary_reason(demand=demand, oracle=oracle, baseline=baseline, initial_objects=initial, horizon=3) == reason


def test_secondary_multilabel_information_boundary_reconciliation_and_json_roundtrip():
    result = analyze([request(0, "b", 3), request(1, "b", 3)])
    assert any(len(row["secondary_evidence"]) > 1 for row in result["request_opportunity_rows"])
    assert result["reconciliation_report"]["status"] == "pass"
    assert result["information_requirement_summary"]["labels_are_not_information_sufficiency_conclusions"] is True
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["opportunity_summary"]["latency_saved"]["availability"] == "unavailable"


def test_concentration_zero_window_strata_and_small_sample_warning():
    result = analyze([request(0, "x", 11)], source=manifest("mb", 10, {"r1": [], "r2": []}))
    concentration = result["opportunity_summary"]["opportunity_concentration"]
    assert concentration["objects"]["warning"] == "small_sample_concentration_unstable"
    assert concentration["zero_opportunity_window_ratio"] == 1.0
    assert set(concentration["fixed_opportunity_density_strata"].values()) <= {"low", "medium", "high", "unavailable"}


@pytest.mark.parametrize("kind", ["replay", "manifest", "capacity", "initial", "horizon", "objective", "duplicate", "missing_outcome", "missing_size", "nonfinite"])
def test_integrity_fail_fast(kind):
    source, value, results, traces, outcomes = bundle([request(0, "b", 3)], horizons=(1,))
    if kind == "replay": results["h_1"]["identity"]["request_replay_fingerprint"] = "bad"
    elif kind == "manifest": results["h_1"]["identity"]["g07_manifest_semantic_sha256"] = "bad"
    elif kind == "capacity": outcomes["reactive_lru/lru"]["capacity_value"] = 99
    elif kind == "initial": outcomes["reactive_lru/lru"]["initial_state_fingerprint"] = "bad"
    elif kind == "horizon": results["h_1"]["identity"]["horizon"] = 3
    elif kind == "objective": results["h_1"]["identity"]["objective_identity"] = "bad"
    elif kind == "duplicate": outcomes["reactive_lru/lru"]["request_outcomes"].append(deepcopy(outcomes["reactive_lru/lru"]["request_outcomes"][0]))
    elif kind == "missing_outcome": outcomes["reactive_lru/lru"].pop("request_outcomes")
    elif kind == "missing_size": value["requests"][0]["object_size_mb"] = None; value["request_replay_fingerprint"] = request_replay_fingerprint(value)
    elif kind == "nonfinite": outcomes["reactive_lru/lru"]["request_outcomes"][0]["admitted_size_mb"] = float("inf")
    with pytest.raises(CacheOpportunityAnalyzerError):
        analyze_cache_opportunities(manifest=source, replay=value, oracle_results=results, oracle_action_traces=traces, baseline_outcomes=outcomes, horizons=[1])


def test_canonical_deterministic_repeated_result_and_contract_version():
    one = analyze([request(0, "b", 3), request(1, "a", 2)])
    two = analyze([request(0, "b", 3), request(1, "a", 2)])
    assert one == two
    assert one["opportunity_summary"]["identity"]["cache_opportunity_analyzer_contract_version"] == CACHE_OPPORTUNITY_ANALYZER_CONTRACT_VERSION


def test_observed_raw_builder_exports_request_rows():
    source = manifest()
    value = replay([request(0, "b", 3)], source)
    solved = solve_future_horizon_cache_oracle(replay=value, manifest=source, horizon=1)
    event = {"event_type": "request", "event_id": "e", "episode_step_index": 1, "time_index": 100, "vehicle_id": "v", "workflow_id": "wf", "node_id": "n0", "object_id": "b", "adapter_id": "b", "size_mb": 3.0, "served_rsu_id": "r1", "cache_target_rsu_id": "r1", "cache_hit": True, "hit_source": "current_rsu", "admission_requested": True, "admission_added": True, "eviction_occurred": False}
    summary = {"run_info": {"fairness_manifest_id": "m", "fairness_manifest_hash": "full", "fairness_semantic_protocol_hash": "semantic", "evaluation_unit_id": "u", "agent_name": "reactive_lru", "eviction_policy": "lru"}, "cache_event_trace": [event]}
    observed = build_observed_baseline_outcome(replay=value, manifest=source, summary=summary)
    assert observed["request_outcomes"][0]["request_id"] == "q0"
    assert observed["reward_consumed"] is False
