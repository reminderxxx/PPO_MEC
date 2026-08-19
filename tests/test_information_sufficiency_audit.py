from __future__ import annotations

import json
from copy import deepcopy

import pytest

from src.analysis.information_sufficiency_audit import (
    DECISION_OBSERVATION_TRACE_VERSION,
    INFORMATION_SUFFICIENCY_AUDIT_CONTRACT_VERSION,
    DEFAULT_AUDIT_CONFIG,
    InformationSufficiencyAuditError,
    analyze_aliasing_and_information,
    analyze_recoverability,
    audit_information_sufficiency,
    build_architecture_audit,
    build_observation_field_map,
    build_synthetic_validation_report,
    empirical_information_statistics,
    evaluate_marl_necessity,
    summarize_schema_coverage,
    validate_audit_inputs,
    validate_decision_observation_trace,
)


def manifest(hidden=False):
    return {
        "identity": {"manifest_id": "g07", "git_commit": "abc"},
        "hashes": {"full_manifest_sha256": "full", "semantic_protocol_sha256": "semantic"},
        "window_workload_plan": {"hidden": hidden},
    }


def request(index, unit="u1", size=64.0):
    return {
        "request_id": f"q{index}", "evaluation_unit_id": unit, "step_index": index + 1,
        "time_index": 100 + index, "object_size_mb": size,
    }


def replay(count=4, units=("u1", "u2")):
    return {
        "cache_request_replay_version": "1.0.0",
        "requests": [request(index, units[index % len(units)]) for index in range(count)],
        "request_replay_fingerprint": "replay-fp",
    }


def opportunity_payload(value, *, cross=False):
    rows = []
    for req in value["requests"]:
        rows.append({
            "request_id": req["request_id"], "evaluation_unit_id": req["evaluation_unit_id"],
            "object_size_mb": req["object_size_mb"], "capacity_value": 3.0,
            "missed_opportunity": True, "primary_opportunity_reason": "wrong_cache_target" if cross else "admission_not_selected",
            "demand": {"cross_rsu_reuse": cross, "handoff_adjacent_reuse": cross},
            "information_requirement_labels": [
                {"information": "current_object_identity_and_size", "availability_class": "decision-time observable"},
                {"information": "current_rsu", "availability_class": "decision-time observable"},
                {"information": "current_cache_contents", "availability_class": "decision-time observable"},
                {"information": "remaining_capacity", "availability_class": "decision-time observable"},
                {"information": "next_rsu_handoff_estimate", "availability_class": "predictor-required"},
                {"information": "transfer_cost", "availability_class": "decision-time observable"},
                {"information": "dag_workflow_future_demand", "availability_class": "oracle-only future information"},
            ],
        })
    return {
        "provenance": {
            "cache_opportunity_analyzer_contract_version": "1.0.0",
            "analysis_fingerprint": "g09-fp", "request_replay_fingerprint": "replay-fp",
            "oracle_contract_version": "future_horizon_cache_oracle_contract_v1.0.0",
            "g07_manifest": {"manifest_id": "g07", "full_manifest_sha256": "full", "semantic_protocol_sha256": "semantic"},
        },
        "rows": rows,
    }


def oracle_trace(value, actions=None):
    actions = actions or ["admit" if index % 2 else "noop" for index in range(len(value["requests"]))]
    return {"h_1": [
        {"request_id": req["request_id"], "action": actions[index], "cache_target_rsu_id": "r2" if actions[index] == "admit" else None, "evicted_object_ids": []}
        for index, req in enumerate(value["requests"])
    ]}


def trace(value, local_values=None, global_values=None):
    local_values = local_values or [0, 0, 1, 1]
    global_values = global_values or list(range(len(value["requests"])))
    records = []
    for index, req in enumerate(value["requests"]):
        records.append({
            "request_id": req["request_id"], "step_index": req["step_index"], "time_index": req["time_index"],
            "captured_phase": "pre_action", "controller_identity": "controller",
            "observation_contract_version": "synthetic_v1", "raw_semantic_fields": {"current_rsu": "r1"},
            "flattened_observation": [float(local_values[index]), float(global_values[index])],
            "flattened_dimension": 2, "feature_name_to_index": {"local": 0, "global": 1},
            "flattened_feature_values": {"local": float(local_values[index]), "global": float(global_values[index])},
            "feature_availability": {"local": True, "global": True},
            "normalization": {"kind": "identity", "version": "1"},
            "information_scopes": {"current_local_information": ["local"], "cross_rsu_global_information": ["global"], "predictor_outputs": [], "history_derived_information": []},
            "action_mask": [True, True], "eligible_actions": [0, 1], "actual_selected_action": index % 2,
            "observation_projections": {
                "actor_local": {"local": local_values[index]},
                "controller_global": {"local": local_values[index], "cross_rsu": global_values[index]},
                "critic_only": {"global": global_values[index]},
                "predictor_augmented": {"local": local_values[index], "cross_rsu": global_values[index], "handoff_prediction": index % 2},
            },
            "recoverability_truth": {"current_rsu": "r1", "remaining_capacity": 2.0, "object_size": 64.0},
            "observation_derived_values": {"current_rsu": "r1", "remaining_capacity": 2.0, "object_size": 32.0},
            "recoverability_resolution": {"object_size": "lossy"},
        })
    return {"decision_observation_trace_version": DECISION_OBSERVATION_TRACE_VERSION, "provenance": {"request_replay_fingerprint": "replay-fp", "g07_manifest_id": "g07", "g07_manifest_semantic_sha256": "semantic", "g08_oracle_contract_version": "future_horizon_cache_oracle_contract_v1.0.0", "g09_analysis_fingerprint": "g09-fp"}, "records": records}


@pytest.mark.parametrize("agent,classification,count", [
    ("ppo", "single_controller", 1), ("mappo", "controller_level_ctde", 3), ("sa_ghmappo", "controller_level_ctde", 3),
])
def test_architecture_classification_comes_from_real_contract(agent, classification, count):
    report = build_architecture_audit(agent)
    assert report["current_architecture_classification"] == classification
    assert report["logical_actor_count"] == count
    assert report["actors_are_vehicle_or_rsu_entities"] is False
    assert report["joint_action_is_centrally_selected_at_execution"] is True
    assert report["parameter_sharing_is_entity_level_multi_agent"] is False


def test_multihead_and_central_critic_do_not_create_entity_observability():
    report = build_architecture_audit("mappo")
    assert report["factorized_controller"] is True
    assert report["independent_entity_local_observation_isolation"] is False
    assert "centralized" in report["centralized_critic"]


def test_field_map_distinguishes_environment_observation_and_encoder_consumption():
    rows = opportunity_payload(replay())["rows"]
    mapped = {row["information_item"]: row for row in build_observation_field_map("sa_ghmappo", rows)}
    assert mapped["object_size"]["environment_contains"] is True
    assert mapped["object_size"]["observation_contains"] is False
    assert mapped["remaining_capacity"]["encoder_consumption"] is None
    assert mapped["current_cache_contents"]["feature_index"]
    assert mapped["current_cache_contents"]["value_resolution"] == "lossy"
    assert "aggregated" in mapped["current_cache_contents"]["information_loss"]


def test_schema_coverage_reports_required_rates_and_weighting_without_equating_sufficiency():
    rows = opportunity_payload(replay())["rows"]
    report = summarize_schema_coverage(build_observation_field_map("sa_ghmappo", rows), rows)
    assert report["required_information_item_count"] == 15
    assert 0 <= report["actor_visible_rate"] <= 1
    assert report["coverage_is_sufficiency"] is False
    assert report["missed_opportunity_bytes_weighted_actor_coverage"] is not None


def test_trace_contract_maps_features_records_normalization_and_pre_action():
    value = replay()
    report = validate_decision_observation_trace(trace(value), value)
    assert report["status"] == "pass"
    assert report["record_count"] == 4


def test_trace_rejects_post_action_future_outcome_leakage_and_raw_flat_mismatch():
    value = replay()
    bad = trace(value)
    bad["records"][0]["captured_phase"] = "post_action"
    bad["records"][0]["service_result"] = "hit"
    bad["records"][1]["flattened_feature_values"]["local"] = 99
    report = validate_decision_observation_trace(bad, value)
    assert report["status"] == "fail"
    assert any("before action" in item for item in report["errors"])
    assert any("forbidden" in item for item in report["errors"])
    assert any("raw/flattened" in item for item in report["errors"])


def test_old_artifact_without_trace_is_unavailable_not_reconstructed():
    report = validate_decision_observation_trace(None, replay())
    assert report["status"] == "unavailable"
    assert "not reconstructed" in report["reason"]


def test_recoverability_exact_lossy_absent_and_inconsistent():
    value = replay(1, units=("u1",))
    one = trace(value, [0], [0])
    one["records"][0]["recoverability_truth"]["next_rsu"] = "r2"
    one["records"][0]["observation_derived_values"]["remaining_capacity"] = 3.0
    report = analyze_recoverability(one, DEFAULT_AUDIT_CONFIG)
    statuses = {row["field"]: row["status"] for row in report["checks"]}
    assert statuses == {"current_rsu": "exact", "next_rsu": "absent", "object_size": "lossy", "remaining_capacity": "inconsistent"}


def test_identical_local_observation_can_conflict_while_global_disambiguates():
    value = replay()
    obs = trace(value, local_values=[0, 0, 0, 0], global_values=[0, 1, 0, 1])
    aliasing, _ = analyze_aliasing_and_information(obs, value, oracle_trace(value), DEFAULT_AUDIT_CONFIG)
    assert aliasing["scopes"]["actor_local"]["coarsened"]["conflicting_oracle_action_group_count"] == 1
    assert aliasing["scopes"]["controller_global"]["exact"]["conflicting_oracle_action_group_count"] == 0


def test_global_observation_can_remain_aliased_and_same_action_alias_is_not_conflict():
    value = replay()
    same = trace(value, local_values=[0, 0, 0, 0], global_values=[0, 0, 0, 0])
    conflicting, _ = analyze_aliasing_and_information(same, value, oracle_trace(value), DEFAULT_AUDIT_CONFIG)
    assert conflicting["scopes"]["controller_global"]["exact"]["conflicting_oracle_action_group_count"] == 1
    nonconflicting, _ = analyze_aliasing_and_information(same, value, oracle_trace(value, ["admit"] * 4), DEFAULT_AUDIT_CONFIG)
    assert nonconflicting["scopes"]["controller_global"]["exact"]["alias_group_count"] == 1
    assert nonconflicting["scopes"]["controller_global"]["exact"]["conflicting_oracle_action_group_count"] == 0


def test_fixed_coarsening_and_feature_removal_projections_are_reported():
    value = replay()
    aliasing, _ = analyze_aliasing_and_information(trace(value), value, oracle_trace(value), DEFAULT_AUDIT_CONFIG)
    assert aliasing["coarsening_version"].startswith("fixed_")
    assert set(aliasing["feature_removal_projections"]) == {"without_cross_rsu", "without_handoff_prediction", "without_cache_state"}
    assert aliasing["continuous_no_exact_duplicate_proves_sufficiency"] is False


def test_entropy_nmi_cmi_zero_entropy_and_sparse_warning_are_hand_checkable():
    cfg = {**DEFAULT_AUDIT_CONFIG, "minimum_information_samples": 4, "minimum_independent_evaluation_units": 2}
    perfect = empirical_information_statistics([0, 0, 1, 1], [0, 0, 1, 1], independent_unit_count=2, config=cfg)
    assert perfect["label_entropy_bits"] == pytest.approx(1.0)
    assert perfect["conditional_entropy_bits"] == pytest.approx(0.0)
    assert perfect["normalized_mutual_information"] == pytest.approx(1.0)
    conditional = empirical_information_statistics([0, 1, 0, 1], [0, 1, 0, 1], conditioning=[0, 0, 1, 1], independent_unit_count=2, config=cfg)
    assert conditional["conditional_mutual_information_bits"] == pytest.approx(1.0)
    zero = empirical_information_statistics([0, 0, 0, 0], [0, 1, 0, 1], independent_unit_count=2, config=cfg)
    assert zero["normalized_mutual_information"] == 0.0
    assert perfect["sparse_cell_warning"] is True


def test_small_sample_and_replicated_horizons_are_unverifiable():
    report = empirical_information_statistics([0] * 20, [0] * 20, independent_unit_count=1, config=DEFAULT_AUDIT_CONFIG)
    assert report["availability"] == "unverifiable"
    value = replay(1, units=("u1",))
    result = audit_information_sufficiency(manifest=manifest(), replay=value, oracle_action_trace=oracle_trace(value), opportunity_rows_payload=opportunity_payload(value), agent_identity="mappo")
    assert result["marl_necessity_verdict"]["overall_verdict"] == "UNVERIFIABLE"
    assert result["input_validation_report"]["replicated_rows_are_independent_samples"] is False


def test_marl_gate_distinguishes_centralized_and_factorized_benefit_from_entity_necessity():
    architecture = build_architecture_audit("mappo")
    aliasing = {"availability": "available", "scopes": {"actor_local": {"coarsened": {"conflicting_oracle_action_group_count": 2}}, "controller_global": {"coarsened": {"conflicting_oracle_action_group_count": 0}}}}
    verdict = evaluate_marl_necessity(architecture=architecture, aliasing=aliasing, opportunity_rows=opportunity_payload(replay(), cross=True)["rows"], independent_evaluation_unit_count=2)
    assert verdict["centralized_information_beneficial"] == "supported"
    assert verdict["factorized_decision_beneficial"] == "partially_supported"
    assert verdict["entity_level_marl_evidence"] == "not_supported"


def test_synthetic_all_entity_level_conditions_can_validate_supported_gate_only():
    evidence = {key: True for key in (
        "two_or_more_real_decision_entities", "independent_local_observations", "entity_owned_actions",
        "concurrent_or_coupled_actions", "irreducible_local_information_limit", "stable_cross_entity_incremental_information",
        "nontrivial_cross_rsu_opportunity", "multiple_independent_evaluation_units", "centralized_controller_lacks_required_information",
    )}
    verdict = evaluate_marl_necessity(
        architecture=build_architecture_audit("mappo"), aliasing={"availability": "available", "scopes": {"actor_local": {"coarsened": {"conflicting_oracle_action_group_count": 1}}, "controller_global": {"coarsened": {"conflicting_oracle_action_group_count": 0}}}},
        opportunity_rows=opportunity_payload(replay(), cross=True)["rows"], independent_evaluation_unit_count=2, synthetic_evidence=evidence,
    )
    assert verdict["entity_level_marl_evidence"] == "supported"
    assert verdict["all_entity_level_conditions_met"] is True


@pytest.mark.parametrize("mutation,error", [
    ("fingerprint", "fingerprint"), ("alignment", "align"), ("duplicate", "duplicate"), ("hidden", "hidden"), ("nan", "finite"),
])
def test_integrity_failures_are_rejected(mutation, error):
    value = replay()
    source = manifest()
    opportunities = opportunity_payload(value)
    oracle = oracle_trace(value)
    obs = trace(value)
    if mutation == "fingerprint": opportunities["provenance"]["request_replay_fingerprint"] = "bad"
    elif mutation == "alignment": oracle["h_1"].pop()
    elif mutation == "duplicate": value["requests"][1]["request_id"] = value["requests"][0]["request_id"]
    elif mutation == "hidden": source["window_workload_plan"]["hidden"] = True
    elif mutation == "nan": opportunities["rows"][0]["object_size_mb"] = float("nan")
    if mutation == "nan":
        with pytest.raises(InformationSufficiencyAuditError, match=error):
            validate_audit_inputs(manifest=source, replay=value, oracle_action_trace=oracle, opportunity_rows_payload=opportunities, observation_trace=obs)
    else:
        report = validate_audit_inputs(manifest=source, replay=value, oracle_action_trace=oracle, opportunity_rows_payload=opportunities, observation_trace=obs)
        assert report["status"] == "fail"
        assert any(error in item for item in report["errors"])


def test_reward_and_oracle_future_leakage_are_rejected():
    value = replay()
    obs = trace(value)
    obs["records"][0]["raw_semantic_fields"]["reward"] = 1.0
    obs["records"][1]["raw_semantic_fields"]["oracle_future_demand"] = ["x"]
    report = validate_decision_observation_trace(obs, value)
    assert report["status"] == "fail"
    assert sum("forbidden" in item for item in report["errors"]) >= 2


def test_bundle_is_json_roundtrip_deterministic_and_provenance_bound():
    value = replay()
    kwargs = dict(manifest=manifest(), replay=value, oracle_action_trace=oracle_trace(value), opportunity_rows_payload=opportunity_payload(value), observation_trace=trace(value), agent_identity="sa_ghmappo")
    first = audit_information_sufficiency(**kwargs)
    second = audit_information_sufficiency(**kwargs)
    assert first == second
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    assert first["identity"]["information_sufficiency_audit_contract_version"] == INFORMATION_SUFFICIENCY_AUDIT_CONTRACT_VERSION
    assert first["identity"]["request_replay_fingerprint"] == "replay-fp"
    assert first["marl_necessity_verdict"]["gnn_gat_necessity"] in {"not_supported", "unverifiable"}
    assert first["marl_necessity_verdict"]["conclusion_matrix"]["entity_level_marl_necessity"] in {"NOT_SUPPORTED", "UNVERIFIABLE"}
    assert first["marl_necessity_verdict"]["required_next_evidence"]


def test_synthetic_validation_report_covers_required_positive_and_negative_cases():
    report = build_synthetic_validation_report()
    assert report["local_alias_global_disambiguation_case"]["passed"] is True
    assert report["global_observation_still_insufficient_case"]["passed"] is True
    assert report["entity_level_marl_all_conditions_case"]["passed"] is True
    assert report["hand_checkable_information_case"]["result"]["normalized_mutual_information"] == pytest.approx(1.0)
