"""G10 read-only observation sufficiency and entity-level MARL necessity audit."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Iterable


INFORMATION_SUFFICIENCY_AUDIT_CONTRACT_VERSION = "1.0.0"
DECISION_OBSERVATION_TRACE_VERSION = "1.0.0"
AUDITOR_IDENTITY = "cache_information_sufficiency_marl_auditor_v1.0.0"
EMPIRICAL_ESTIMATOR = "empirical_plugin_discrete_v1.0.0"
COARSENING_VERSION = "fixed_information_projection_buckets_v1.0.0"

DEFAULT_AUDIT_CONFIG: dict[str, Any] = {
    "continuous_absolute_tolerance": 1e-9,
    "continuous_relative_tolerance": 1e-9,
    "minimum_information_samples": 8,
    "minimum_independent_evaluation_units": 2,
    "sparse_cell_minimum_mean_count": 5.0,
    "float_buckets": [-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
    "feature_removal_projections": {
        "without_cross_rsu": ["cross_rsu", "global_cache", "other_rsu"],
        "without_handoff_prediction": ["handoff", "next_rsu", "prediction"],
        "without_cache_state": ["cache", "capacity", "resident"],
    },
}

FORBIDDEN_INPUT_KEYS = {
    "reward", "reward_total", "reward_breakdown", "legacy_aggregate", "aggregate_summary",
    "oracle_future", "oracle_visible_requests", "future_ground_truth", "service_result",
    "post_action_state", "post_state", "cache_hit", "stall_occurred", "hidden_state",
    "learned_hidden_state", "formal_hidden", "hidden_holdout",
}

REQUIRED_INFORMATION_ITEMS = (
    "current_object_identity", "object_size", "current_request_rsu", "current_cache_contents",
    "remaining_capacity", "capacity_unit_value", "object_recency", "object_frequency",
    "future_reuse_estimate", "next_rsu_handoff_estimate", "cross_rsu_cache_state",
    "transfer_cost", "multi_victim_capacity_pressure", "coordination_information",
    "dag_workflow_future_demand",
)

SOURCE_CITATIONS = {
    "wrapper_single_action": "src/envs/wrappers/gym_vec_env.py:53-81,144-181",
    "semantic_state": "src/envs/core/vec_workflow_core_env.py:439-466",
    "action_schema": "src/envs/specs/action_schema.py:20-92,144-280",
    "action_preconditions": "src/envs/specs/action_schema.py:312-377",
    "head_aggregation": "src/agents/sa_ghmappo_core.py:661-681,2822-2958",
    "ppo": "src/agents/ppo_agent.py:10-55",
    "mappo": "src/agents/mappo_agent.py:10-20,41-135",
    "policy_heads": "src/agents/sa_ghmappo_core.py:224-334,596-639",
    "flat_encoder": "src/encoders/fusion_encoder.py:263-402",
    "graph_encoder": "src/encoders/dag_graph_encoder.py:37-147",
    "rsu_encoder": "src/encoders/rsu_state_encoder.py:22-108",
    "fusion_encoder": "src/encoders/fusion_encoder.py:405-671",
    "sa_agent": "src/agents/sa_ghmappo_agent.py:8-16,18-136",
    "registry": "src/agents/registry.py:31-49,162-175",
    "capacity_internal": "src/envs/core/vec_workflow_core_env.py:571-764",
    "semantic_objects": "src/envs/specs/semantic_objects.py:25-73",
}


class InformationSufficiencyAuditError(ValueError):
    """Raised when G10 input identity, trace timing, or leakage checks fail."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _finite(value: Any, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InformationSufficiencyAuditError(f"{path} must be numeric") from exc
    if not math.isfinite(result):
        raise InformationSufficiencyAuditError(f"{path} must be finite")
    return result


def _walk_forbidden(value: Any, path: str = "input") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_INPUT_KEYS or normalized.startswith("oracle_future"):
                errors.append(f"forbidden leakage/source key at {path}.{key}")
            errors.extend(_walk_forbidden(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_walk_forbidden(child, f"{path}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(f"non-finite value at {path}")
    return errors


def build_architecture_audit(agent_identity: str) -> dict[str, Any]:
    """Return source-grounded decision-subject facts; names never imply entities."""
    agent = str(agent_identity).strip().lower()
    if agent not in {"ppo", "mappo", "sa_ghmappo"}:
        raise InformationSufficiencyAuditError(f"unsupported audited agent: {agent_identity}")
    if agent == "ppo":
        logical_actors = [{"actor_id": "flat_actor", "role": "joint semantic controller", "entity_binding": None}]
        classification = "single_controller"
        critic = "independent critic over the same flat encoder context"
        scopes = ["flat semantic aggregate"]
    else:
        logical_actors = [
            {"actor_id": "slow_actor", "role": "cache controller head", "entity_binding": None},
            {"actor_id": "fast_actor", "role": "execution/offload controller head", "entity_binding": None},
            {"actor_id": "event_actor", "role": "handoff-event controller head", "entity_binding": None},
        ]
        classification = "controller_level_ctde"
        critic = "one centralized controller-level critic"
        scopes = (["shared flat semantic embedding"] * 3 if agent == "mappo" else
                  ["cache/RSU/prediction role projection or shared fusion", "DAG/current-RSU/vehicle role projection or shared fusion", "vehicle/current-target-RSU/prediction role projection or shared fusion"])
    return {
        "information_sufficiency_audit_contract_version": INFORMATION_SUFFICIENCY_AUDIT_CONTRACT_VERSION,
        "agent_identity": agent,
        "current_architecture_classification": classification,
        "execution_topology": "one GymVecEnv wrapper and one controller call produce one semantic_discrete_5 action",
        "final_cache_offload_action_producer": "the selected agent controller; hierarchical heads are centrally aggregated before ActionAdapter decoding" if agent != "ppo" else "the single PPO flat actor before ActionAdapter decoding",
        "logical_actor_count": len(logical_actors),
        "logical_actors": logical_actors,
        "actor_input_scopes": scopes,
        "actors_are_vehicle_or_rsu_entities": False,
        "independent_entity_local_observation_isolation": False,
        "independent_entity_action_ownership": False,
        "centralized_critic": critic,
        "joint_action_is_centrally_selected_at_execution": True,
        "parameter_sharing_is_entity_level_multi_agent": False,
        "factorized_controller": agent != "ppo",
        "entity_level_marl": False,
        "naming_risk": (
            "MAPPO/SA-GHMAPPO and 'decentralized controller actors' can be misread as vehicle/RSU-level MARL; "
            "the code implements controller-role heads without entity binding."
        ),
        "source_evidence": SOURCE_CITATIONS,
    }


def _field_spec(agent: str, item: str) -> dict[str, Any]:
    is_flat = agent in {"ppo", "mappo"}
    specs: dict[str, dict[str, Any]] = {
        "current_object_identity": dict(source="current_workflow_node.required_adapter", in_environment=True, in_observation=True, actor="lossy" if not is_flat else "absent", critic="lossy" if not is_flat else "absent", controller=True, availability="decision-time observable", feature="DAG per-node required-adapter membership flags" if not is_flat else None, loss="identity is not embedded; only readiness membership survives"),
        "object_size": dict(source="AdapterCatalog CacheObject.size_mb", in_environment=True, in_observation=False, actor="absent", critic="absent", controller=False, availability="currently absent/unknown", feature=None, loss="catalog resident size is not in semantic state"),
        "current_request_rsu": dict(source="vehicles[].associated_rsu_id / associations", in_environment=True, in_observation=True, actor="lossy", critic="lossy", controller=True, availability="decision-time observable", feature="current-RSU selector/association indicator", loss="RSU identifier is used for selection but not retained as an exact categorical feature"),
        "current_cache_contents": dict(source="rsus[].cached_adapter_ids", in_environment=True, in_observation=True, actor="lossy" if not is_flat else "absent", critic="lossy", controller=True, availability="decision-time observable", feature="required-adapter-ready flags and cache counts/occupancy aggregates", loss="full resident identities are aggregated to membership/count/set mean"),
        "remaining_capacity": dict(source="VecWorkflowCoreEnv._cache_capacity_snapshot", in_environment=True, in_observation=False, actor="absent", critic="absent", controller=False, availability="currently absent/unknown", feature=None, loss="capacity snapshot is outcome telemetry/internal state, not semantic observation"),
        "capacity_unit_value": dict(source="VecWorkflowCoreEnv._cache_capacity_profile", in_environment=True, in_observation=False, actor="absent", critic="absent", controller=False, availability="currently absent/unknown", feature=None, loss="capacity unit/value are not RSUState fields"),
        "object_recency": dict(source="eviction policy internal state", in_environment=True, in_observation=False, actor="absent", critic="absent", controller=False, availability="history-derived", feature=None, loss="not exported to semantic state"),
        "object_frequency": dict(source="predictor cache_demand.demand_score_by_rsu", in_environment=True, in_observation=True, actor="lossy" if not is_flat else "absent", critic="lossy" if not is_flat else "absent", controller=True, availability="history-derived", feature="required-adapter demand score" if not is_flat else None, loss="score is not an exact frequency counter"),
        "future_reuse_estimate": dict(source="no dedicated per-object reuse probability", in_environment=False, in_observation=False, actor="absent", critic="absent", controller=False, availability="predictor-required", feature=None, loss="future_load/cache_demand are aggregate surrogates, not object reuse"),
        "next_rsu_handoff_estimate": dict(source="predictions next_rsu_sequence/predicted_first_handoff_rsu_by_vehicle", in_environment=True, in_observation=True, actor="lossy", critic="lossy", controller=True, availability="predictor-required", feature="presence, selected target embedding, countdown/confidence/uncertainty", loss="exact ID/sequence is selected or aggregated rather than preserved"),
        "cross_rsu_cache_state": dict(source="rsus[].cached_adapter_ids", in_environment=True, in_observation=True, actor="lossy" if not is_flat else "absent", critic="lossy", controller=True, availability="decision-time observable", feature="RSU set mean and required-adapter membership per RSU" if not is_flat else "mean/max cache occupancy for critic only", loss="set pooling discards exact joint cache configuration"),
        "transfer_cost": dict(source="catalog size and environment cost model", in_environment=True, in_observation=False, actor="absent", critic="absent", controller=False, availability="decision-time observable", feature=None, loss="no resident-size/transfer-cost scalar in semantic actor input"),
        "multi_victim_capacity_pressure": dict(source="capacity profile plus victim planner", in_environment=True, in_observation=False, actor="absent", critic="absent", controller=False, availability="decision-time observable", feature=None, loss="required free space and victim count are computed after action inside environment"),
        "coordination_information": dict(source="global RSU set/predicted target", in_environment=True, in_observation=True, actor="lossy" if not is_flat else "absent", critic="lossy", controller=True, availability="decision-time observable", feature="global pooled RSU embedding/aggregates", loss="no independently owned entity messages or actions"),
        "dag_workflow_future_demand": dict(source="workflow graph/execution_order", in_environment=True, in_observation=True, actor="lossy" if not is_flat else "absent", critic="lossy", controller=True, availability="oracle-only future information", feature="DAG graph/frontier embeddings" if not is_flat else "progress/frontier aggregates", loss="observable DAG structure is pooled, while future realized demand remains oracle-only"),
    }
    value = deepcopy(specs[item])
    if agent == "mappo" and item in {"current_cache_contents", "cross_rsu_cache_state", "coordination_information", "dag_workflow_future_demand"}:
        value["critic_only"] = value["actor"] == "absent" and value["critic"] != "absent"
    else:
        value["critic_only"] = False
    return value


def build_observation_field_map(agent_identity: str, opportunity_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    agent = str(agent_identity).lower()
    if agent not in {"ppo", "mappo", "sa_ghmappo"}:
        raise InformationSufficiencyAuditError(f"unsupported audited agent: {agent_identity}")
    reasons: dict[str, set[str]] = defaultdict(set)
    for row in opportunity_rows:
        for label in row.get("information_requirement_labels", []):
            raw = str(label.get("information", ""))
            mapped = {
                "current_object_identity_and_size": ("current_object_identity", "object_size"),
                "current_rsu": ("current_request_rsu",),
                "current_cache_contents": ("current_cache_contents",),
                "remaining_capacity": ("remaining_capacity", "capacity_unit_value", "multi_victim_capacity_pressure"),
                "next_rsu_handoff_estimate": ("next_rsu_handoff_estimate",),
                "transfer_cost": ("transfer_cost",),
                "dag_workflow_future_demand": ("dag_workflow_future_demand",),
            }.get(raw, (raw,))
            for item in mapped:
                reasons[item].add(str(row.get("primary_opportunity_reason", "unknown")))
    output = []
    graph_indices = {
        "current_object_identity": ["DAG node features[7:10]"],
        "current_request_rsu": ["RSU features[3]", "selected current_rsu_embedding"],
        "current_cache_contents": ["RSU features[2,9]", "DAG node features[7:10]"],
        "object_frequency": ["RSU features[8]"],
        "next_rsu_handoff_estimate": ["RSU features[4,5]", "prediction features[0:5,8:13]"],
        "cross_rsu_cache_state": ["RSU features[2,9] before set mean"],
        "coordination_information": ["RSU set_embedding", "target_rsu_embedding"],
        "dag_workflow_future_demand": ["DAG node features[0:7] before graph/frontier pooling"],
    }
    flat_indices = {
        "current_request_rsu": ["actor flat features[10]"],
        "next_rsu_handoff_estimate": ["actor flat features[12,14:18]"],
        "current_cache_contents": ["critic flat features[10,11]"],
        "cross_rsu_cache_state": ["critic flat features[10,11]"],
        "coordination_information": ["critic flat features[8:16]"],
        "dag_workflow_future_demand": ["actor flat features[3,6:10]", "critic flat features[3:6,16:18]"],
    }
    for item in REQUIRED_INFORMATION_ITEMS:
        spec = _field_spec(agent, item)
        output.append({
            "information_item": item,
            "required_by_opportunity_types": sorted(reasons.get(item, set())),
            "source_semantic_field": spec["source"],
            "environment_contains": spec["in_environment"],
            "observation_contains": spec["in_observation"],
            "feature_index": (flat_indices if agent in {"ppo", "mappo"} else graph_indices).get(item),
            "value_resolution": spec["actor"],
            "local_actor_visible": spec["actor"] != "absent",
            "centralized_critic_visible": spec["critic"] != "absent",
            "controller_visible": spec["controller"],
            "availability_class": spec["availability"],
            "encoder_consumption": spec["feature"],
            "information_loss": spec["loss"],
            "critic_only": spec["critic_only"],
            "source_evidence": SOURCE_CITATIONS,
        })
    return output


def summarize_schema_coverage(field_map: list[dict[str, Any]], opportunity_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(field_map)
    counts = {
        "required_information_item_count": total,
        "actor_visible_count": sum(bool(row["local_actor_visible"]) for row in field_map),
        "controller_visible_count": sum(bool(row["controller_visible"]) for row in field_map),
        "critic_only_count": sum(bool(row["critic_only"]) for row in field_map),
        "predictor_dependent_count": sum(row["availability_class"] == "predictor-required" for row in field_map),
        "oracle_only_count": sum(row["availability_class"] == "oracle-only future information" for row in field_map),
        "absent_count": sum(row["value_resolution"] == "absent" for row in field_map),
    }
    for name in tuple(counts):
        if name.endswith("_count") and name != "required_information_item_count":
            counts[name.replace("_count", "_rate")] = counts[name] / total if total else None
    required_lookup = {row["information_item"]: row for row in field_map}
    reason_weight = Counter()
    byte_weight = Counter()
    reason_covered = Counter()
    byte_covered = Counter()
    for row in opportunity_rows:
        weight = float(row.get("object_size_mb", 0.0) or 0.0) if row.get("missed_opportunity") else 0.0
        reason = str(row.get("primary_opportunity_reason", "unknown"))
        labels = []
        for label in row.get("information_requirement_labels", []):
            labels.extend({
                "current_object_identity_and_size": ["current_object_identity", "object_size"],
                "current_rsu": ["current_request_rsu"],
                "remaining_capacity": ["remaining_capacity"],
            }.get(label.get("information"), [label.get("information")]))
        labels = [item for item in labels if item in required_lookup]
        for item in labels:
            reason_weight[(reason, item)] += 1
            byte_weight[(reason, item)] += weight
            if required_lookup[item]["local_actor_visible"]:
                reason_covered[(reason, item)] += 1
                byte_covered[(reason, item)] += weight
    denom = sum(reason_weight.values())
    byte_denom = sum(byte_weight.values())
    return {
        **counts,
        "opportunity_reason_weighted_actor_coverage": sum(reason_covered.values()) / denom if denom else None,
        "missed_opportunity_bytes_weighted_actor_coverage": sum(byte_covered.values()) / byte_denom if byte_denom else None,
        "coverage_is_sufficiency": False,
    }


def validate_decision_observation_trace(trace: dict[str, Any] | None, replay: dict[str, Any]) -> dict[str, Any]:
    if trace is None:
        return {"status": "unavailable", "reason": "decision-time observation trace was not supplied; old artifacts are not reconstructed"}
    errors: list[str] = []
    if trace.get("decision_observation_trace_version") != DECISION_OBSERVATION_TRACE_VERSION:
        errors.append("unsupported decision observation trace version")
    provenance = trace.get("provenance", {})
    required_provenance = {
        "request_replay_fingerprint", "g07_manifest_id", "g07_manifest_semantic_sha256",
        "g08_oracle_contract_version", "g09_analysis_fingerprint",
    }
    missing_provenance = sorted(required_provenance - set(provenance))
    if missing_provenance:
        errors.append(f"decision trace missing provenance fields: {missing_provenance}")
    if provenance.get("request_replay_fingerprint") != replay.get("request_replay_fingerprint"):
        errors.append("decision trace request replay fingerprint mismatch")
    records = trace.get("records")
    if not isinstance(records, list):
        errors.append("decision trace records must be a list")
        records = []
    ids = [str(row.get("request_id")) for row in records]
    replay_ids = [str(row.get("request_id")) for row in replay.get("requests", [])]
    if len(ids) != len(set(ids)):
        errors.append("duplicate request ID in decision trace")
    if ids != replay_ids:
        errors.append("decision trace cannot align request-by-request with replay")
    for index, record in enumerate(records):
        required = {
            "request_id", "step_index", "time_index", "captured_phase", "controller_identity",
            "observation_contract_version", "raw_semantic_fields", "flattened_observation",
            "feature_name_to_index", "feature_availability", "normalization", "information_scopes",
            "action_mask", "eligible_actions", "actual_selected_action", "observation_projections",
        }
        missing = sorted(required - set(record))
        if missing:
            errors.append(f"records[{index}] missing fields: {missing}")
            continue
        if record["captured_phase"] != "pre_action":
            errors.append(f"records[{index}] was not captured before action selection")
        flattened = record["flattened_observation"]
        mapping = record["feature_name_to_index"]
        if not isinstance(flattened, list) or not isinstance(mapping, dict):
            errors.append(f"records[{index}] flattened observation/mapping malformed")
            continue
        for feature, position in mapping.items():
            if not isinstance(position, int) or position < 0 or position >= len(flattened):
                errors.append(f"records[{index}] invalid feature index: {feature}")
            elif feature in record.get("flattened_feature_values", {}):
                expected = _finite(record["flattened_feature_values"][feature], f"records[{index}].flattened_feature_values.{feature}")
                actual = _finite(flattened[position], f"records[{index}].flattened_observation[{position}]")
                if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                    errors.append(f"records[{index}] raw/flattened feature mismatch: {feature}")
        errors.extend(_walk_forbidden(record, f"records[{index}]"))
    return {"status": "pass" if not errors else "fail", "errors": errors, "record_count": len(records)}


def validate_audit_inputs(*, manifest: dict[str, Any], replay: dict[str, Any], oracle_action_trace: dict[str, Any], opportunity_rows_payload: dict[str, Any], observation_trace: dict[str, Any] | None) -> dict[str, Any]:
    errors: list[str] = []
    if manifest.get("window_workload_plan", {}).get("hidden"):
        errors.append("hidden manifest input is forbidden")
    replay_fp = replay.get("request_replay_fingerprint")
    provenance = opportunity_rows_payload.get("provenance", {})
    if provenance.get("request_replay_fingerprint") != replay_fp:
        errors.append("G09 request replay fingerprint mismatch")
    if provenance.get("cache_opportunity_analyzer_contract_version") != "1.0.0":
        errors.append("unsupported G09 contract version")
    g07 = provenance.get("g07_manifest", {})
    checks = {
        "manifest ID": (g07.get("manifest_id"), manifest.get("identity", {}).get("manifest_id")),
        "manifest full hash": (g07.get("full_manifest_sha256"), manifest.get("hashes", {}).get("full_manifest_sha256")),
        "manifest semantic hash": (g07.get("semantic_protocol_sha256"), manifest.get("hashes", {}).get("semantic_protocol_sha256")),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"G09 {label} mismatch")
    replay_ids = [str(row.get("request_id")) for row in replay.get("requests", [])]
    if len(replay_ids) != len(set(replay_ids)):
        errors.append("duplicate request ID in replay")
    rows = opportunity_rows_payload.get("rows")
    if not isinstance(rows, list):
        errors.append("G09 opportunity rows must be a list")
        rows = []
    row_ids = {str(row.get("request_id")) for row in rows}
    if row_ids != set(replay_ids):
        errors.append("G09 opportunity rows do not align to replay request IDs")
    for row in rows:
        if any(str(key).lower().startswith(("reward", "aggregate")) for key in row):
            errors.append("G09 opportunity rows contain forbidden reward/aggregate dependency")
        for field in ("object_size_mb", "capacity_value"):
            _finite(row.get(field), f"opportunity row {field}")
    oracle_ids: set[str] = set()
    for key, trace_rows in oracle_action_trace.items():
        if not (str(key).startswith("h_") and isinstance(trace_rows, list)):
            continue
        ids = [str(row.get("request_id")) for row in trace_rows]
        if ids != replay_ids:
            errors.append(f"{key} oracle trace cannot align to replay")
        oracle_ids.update(ids)
    if oracle_ids != set(replay_ids):
        errors.append("oracle action trace request coverage mismatch")
    trace_report = validate_decision_observation_trace(observation_trace, replay)
    if trace_report["status"] == "fail":
        errors.extend(trace_report["errors"])
    if observation_trace is not None:
        trace_provenance = observation_trace.get("provenance", {})
        trace_checks = {
            "G07 manifest ID": (trace_provenance.get("g07_manifest_id"), manifest.get("identity", {}).get("manifest_id")),
            "G07 semantic hash": (trace_provenance.get("g07_manifest_semantic_sha256"), manifest.get("hashes", {}).get("semantic_protocol_sha256")),
            "G08 oracle contract": (trace_provenance.get("g08_oracle_contract_version"), provenance.get("oracle_contract_version")),
            "G09 analysis fingerprint": (trace_provenance.get("g09_analysis_fingerprint"), provenance.get("analysis_fingerprint")),
        }
        for label, (actual, expected) in trace_checks.items():
            if actual != expected:
                errors.append(f"decision trace {label} mismatch")
    errors.extend(_walk_forbidden({"observation_trace": observation_trace} if observation_trace else {}))
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "decision_observation_trace": trace_report,
        "request_count": len(replay_ids),
        "g09_row_count": len(rows),
        "replicated_rows_are_independent_samples": False,
        "request_replay_fingerprint": replay_fp,
    }


def analyze_recoverability(observation_trace: dict[str, Any] | None, config: dict[str, Any]) -> dict[str, Any]:
    if observation_trace is None:
        return {"availability": "unavailable", "reason": "no matched pre-action observation trace", "checks": []}
    checks = []
    abs_tol = float(config["continuous_absolute_tolerance"])
    rel_tol = float(config["continuous_relative_tolerance"])
    for record in observation_trace.get("records", []):
        truth = record.get("recoverability_truth", {})
        derived = record.get("observation_derived_values", {})
        resolutions = record.get("recoverability_resolution", {})
        for field in sorted(set(truth) | set(derived)):
            if field not in truth or field not in derived:
                status = "absent"
            elif resolutions.get(field) == "lossy":
                status = "lossy"
            elif isinstance(truth[field], (int, float)) and isinstance(derived[field], (int, float)):
                status = "exact" if math.isclose(float(truth[field]), float(derived[field]), rel_tol=rel_tol, abs_tol=abs_tol) else "inconsistent"
            else:
                status = "exact" if truth[field] == derived[field] else "inconsistent"
            checks.append({"request_id": record["request_id"], "field": field, "truth": truth.get(field), "derived": derived.get(field), "status": status})
    counts = Counter(row["status"] for row in checks)
    return {"availability": "available", "tolerance": {"absolute": abs_tol, "relative": rel_tol}, "status_counts": dict(sorted(counts.items())), "checks": checks}


def _coarsen(value: Any, boundaries: list[float]) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        number = _finite(value, "coarsened value")
        for boundary in boundaries:
            if number <= boundary:
                return f"le_{boundary:g}"
        return f"gt_{boundaries[-1]:g}"
    if isinstance(value, list):
        return [_coarsen(item, boundaries) for item in value]
    if isinstance(value, dict):
        return {key: _coarsen(value[key], boundaries) for key in sorted(value)}
    return str(value)


def _entropy(labels: Iterable[Any]) -> float:
    values = list(labels)
    if not values:
        return 0.0
    counts = Counter(_canonical(item) for item in values)
    return -sum((count / len(values)) * math.log2(count / len(values)) for count in counts.values())


def _conditional_entropy(labels: list[Any], conditions: list[Any]) -> float:
    groups: dict[str, list[Any]] = defaultdict(list)
    for label, condition in zip(labels, conditions):
        groups[_canonical(condition)].append(label)
    total = len(labels)
    return sum((len(group) / total) * _entropy(group) for group in groups.values()) if total else 0.0


def empirical_information_statistics(labels: list[Any], conditions: list[Any], *, independent_unit_count: int, config: dict[str, Any], conditioning: list[Any] | None = None) -> dict[str, Any]:
    if len(labels) != len(conditions) or (conditioning is not None and len(labels) != len(conditioning)):
        raise InformationSufficiencyAuditError("information statistic inputs must have equal lengths")
    n = len(labels)
    min_n = int(config["minimum_information_samples"])
    min_units = int(config["minimum_independent_evaluation_units"])
    cells = len({_canonical(item) for item in conditions})
    sparse = bool(cells and n / cells < float(config["sparse_cell_minimum_mean_count"]))
    base = {
        "estimator": EMPIRICAL_ESTIMATOR, "sample_count": n, "independent_evaluation_unit_count": independent_unit_count,
        "effective_condition_cells": cells, "mean_samples_per_cell": n / cells if cells else None,
        "sparse_cell_warning": sparse, "mutual_information_is_causal": False,
    }
    if n < min_n or independent_unit_count < min_units:
        return {**base, "availability": "unverifiable", "reason": "sample or independent evaluation-unit minimum not met"}
    h_y = _entropy(labels)
    h_y_x = _conditional_entropy(labels, conditions)
    mi = max(h_y - h_y_x, 0.0)
    h_x = _entropy(conditions)
    nmi = mi / math.sqrt(h_y * h_x) if h_y > 0 and h_x > 0 else 0.0
    result = {**base, "availability": "available", "label_entropy_bits": h_y, "conditional_entropy_bits": h_y_x, "information_gain_bits": mi, "normalized_mutual_information": nmi}
    if conditioning is not None:
        combined = list(zip(conditions, conditioning))
        h_y_z = _conditional_entropy(labels, conditioning)
        h_y_xz = _conditional_entropy(labels, combined)
        result["conditional_mutual_information_bits"] = max(h_y_z - h_y_xz, 0.0)
    return result


def _group_aliases(rows: list[dict[str, Any]], key_name: str, label_name: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_canonical(row[key_name])].append(row)
    aliases = [group for group in groups.values() if len(group) > 1]
    conflicts = [group for group in aliases if len({_canonical(row[label_name]) for row in group}) > 1]
    conflict_ids = {row["request_id"] for group in conflicts for row in group}
    conflict_mb = sum(float(row.get("object_size_mb", 0.0)) for group in conflicts for row in group)
    return {
        "group_count": len(groups), "alias_group_count": len(aliases), "conflicting_oracle_action_group_count": len(conflicts),
        "conditional_label_entropy_bits": _conditional_entropy([row[label_name] for row in rows], [row[key_name] for row in rows]),
        "conflict_request_count": len(conflict_ids), "conflict_request_coverage": len(conflict_ids) / len({row['request_id'] for row in rows}) if rows else None,
        "conflict_mb": conflict_mb,
        "examples": [[{"request_id": row["request_id"], "oracle_action": row[label_name]} for row in group[:4]] for group in conflicts[:3]],
    }


def analyze_aliasing_and_information(observation_trace: dict[str, Any] | None, replay: dict[str, Any], oracle_action_trace: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if observation_trace is None:
        unavailable = {"availability": "unavailable", "reason": "no matched pre-action observation trace; exact duplicates cannot be used to claim sufficiency"}
        return unavailable, unavailable
    request_lookup = {str(row["request_id"]): row for row in replay.get("requests", [])}
    oracle_by_horizon = {
        key: {str(row["request_id"]): row for row in values}
        for key, values in oracle_action_trace.items() if str(key).startswith("h_") and isinstance(values, list)
    }
    joined = []
    for record in observation_trace.get("records", []):
        rid = str(record["request_id"])
        projections = record["observation_projections"]
        for horizon, oracle_rows in sorted(oracle_by_horizon.items()):
            if rid not in oracle_rows:
                continue
            oracle = oracle_rows[rid]
            label = {"action": oracle.get("action"), "target": oracle.get("cache_target_rsu_id"), "victims": oracle.get("evicted_object_ids", [])}
            row = {
                "request_id": rid, "evaluation_unit_id": request_lookup[rid].get("evaluation_unit_id"), "horizon": horizon,
                "object_size_mb": request_lookup[rid].get("object_size_mb", 0.0), "oracle_action": label,
                "opportunity_label": oracle.get("rejection_reason") or oracle.get("action"),
                "flattened_hash": _sha256_value(record["flattened_observation"]),
            }
            for name in ("actor_local", "controller_global", "critic_only", "predictor_augmented"):
                row[name] = projections.get(name, {})
                row[f"{name}_coarsened"] = _coarsen(row[name], config["float_buckets"])
            joined.append(row)
    scopes = {}
    for scope in ("actor_local", "controller_global", "critic_only", "predictor_augmented"):
        scopes[scope] = {
            "exact": _group_aliases(joined, scope, "oracle_action"),
            "coarsened": _group_aliases(joined, f"{scope}_coarsened", "oracle_action"),
            "conflicting_opportunity_label_exact": _group_aliases(joined, scope, "opportunity_label")["conflicting_oracle_action_group_count"],
            "conflicting_opportunity_label_coarsened": _group_aliases(joined, f"{scope}_coarsened", "opportunity_label")["conflicting_oracle_action_group_count"],
        }
    removals = {}
    for projection_name, tokens in config["feature_removal_projections"].items():
        projected_rows = []
        for row in joined:
            value = row["controller_global"]
            if isinstance(value, dict):
                value = {key: child for key, child in value.items() if not any(token in key.lower() for token in tokens)}
            projected_rows.append({**row, "masked": _coarsen(value, config["float_buckets"])})
        removals[projection_name] = _group_aliases(projected_rows, "masked", "oracle_action")
    exact_flat = _group_aliases(joined, "flattened_hash", "oracle_action")
    independent_units = len({row["evaluation_unit_id"] for row in joined})
    unique_requests = len({row["request_id"] for row in joined})
    labels = [row["oracle_action"] for row in joined]
    stats = {
        scope: empirical_information_statistics(labels, [row[f"{scope}_coarsened"] for row in joined], independent_unit_count=independent_units, config=config)
        for scope in ("actor_local", "controller_global", "critic_only", "predictor_augmented")
    }
    stats["cross_rsu_increment_given_local"] = empirical_information_statistics(
        labels, [row["controller_global_coarsened"] for row in joined], conditioning=[row["actor_local_coarsened"] for row in joined], independent_unit_count=independent_units, config=config,
    )
    return ({
        "availability": "available", "coarsening_version": COARSENING_VERSION, "fixed_float_buckets": config["float_buckets"],
        "joined_row_count": len(joined), "unique_request_count": unique_requests, "independent_evaluation_unit_count": independent_units,
        "horizon_rows_are_independent_samples": False, "flattened_observation_hash": exact_flat,
        "scopes": scopes, "feature_removal_projections": removals,
        "continuous_no_exact_duplicate_proves_sufficiency": False,
    }, {"availability": "available", "statistics": stats, "sample_independence_warning": "horizon/baseline replicas are not independent requests"})


def analyze_opportunity_identifiability(field_map: list[dict[str, Any]], opportunity_rows: list[dict[str, Any]], trace_available: bool) -> dict[str, Any]:
    visible = {row["information_item"] for row in field_map if row["local_actor_visible"]}
    requirements = {
        "admission_opportunity": {"current_object_identity", "current_request_rsu", "current_cache_contents", "remaining_capacity"},
        "wrong_cache_target": {"current_object_identity", "current_request_rsu", "cross_rsu_cache_state", "next_rsu_handoff_estimate"},
        "eviction_choice": {"current_object_identity", "object_size", "current_cache_contents", "remaining_capacity", "object_recency", "object_frequency", "future_reuse_estimate"},
        "insufficient_capacity": {"object_size", "remaining_capacity", "capacity_unit_value"},
        "transfer_tradeoff": {"object_size", "transfer_cost", "future_reuse_estimate"},
        "topology_eligibility": {"current_request_rsu", "next_rsu_handoff_estimate"},
        "handoff_adjacent_reuse": {"next_rsu_handoff_estimate", "future_reuse_estimate", "cross_rsu_cache_state"},
        "multi_victim_requirement": {"object_size", "current_cache_contents", "remaining_capacity", "multi_victim_capacity_pressure"},
    }
    observed_reasons = {str(row.get("primary_opportunity_reason")) for row in opportunity_rows}
    results = []
    for opportunity, required in requirements.items():
        overlap = visible & required
        if not trace_available:
            verdict = "unavailable"
        elif not overlap:
            verdict = "not_identifiable"
        elif overlap == required:
            verdict = "identifiable"
        else:
            verdict = "partially_identifiable"
        results.append({"opportunity_type": opportunity, "required_information": sorted(required), "actor_visible_information": sorted(overlap), "verdict": verdict, "observed_primary_reason": opportunity.replace("_opportunity", "") in observed_reasons})
    return {"trace_available": trace_available, "rows": results, "oracle_future_treated_as_observable": False}


def evaluate_marl_necessity(*, architecture: dict[str, Any], aliasing: dict[str, Any], opportunity_rows: list[dict[str, Any]], independent_evaluation_unit_count: int, synthetic_evidence: dict[str, bool] | None = None) -> dict[str, Any]:
    synthetic = synthetic_evidence or {}
    cross_count = len({row.get("request_id") for row in opportunity_rows if row.get("demand", {}).get("cross_rsu_reuse") or row.get("demand", {}).get("handoff_adjacent_reuse")})
    conditions = {
        "two_or_more_real_decision_entities": bool(synthetic.get("two_or_more_real_decision_entities", architecture.get("actors_are_vehicle_or_rsu_entities") and architecture.get("logical_actor_count", 0) >= 2)),
        "independent_local_observations": bool(synthetic.get("independent_local_observations", architecture.get("independent_entity_local_observation_isolation"))),
        "entity_owned_actions": bool(synthetic.get("entity_owned_actions", architecture.get("independent_entity_action_ownership"))),
        "concurrent_or_coupled_actions": bool(synthetic.get("concurrent_or_coupled_actions", False)),
        "irreducible_local_information_limit": bool(synthetic.get("irreducible_local_information_limit", False)),
        "stable_cross_entity_incremental_information": bool(synthetic.get("stable_cross_entity_incremental_information", False)),
        "nontrivial_cross_rsu_opportunity": bool(synthetic.get("nontrivial_cross_rsu_opportunity", cross_count > 1)),
        "multiple_independent_evaluation_units": bool(synthetic.get("multiple_independent_evaluation_units", independent_evaluation_unit_count >= 2)),
        "centralized_controller_lacks_required_information": bool(synthetic.get("centralized_controller_lacks_required_information", False)),
    }
    trace_available = aliasing.get("availability") == "available"
    if all(conditions.values()):
        entity_verdict = "supported"
    elif not trace_available or independent_evaluation_unit_count < 2:
        entity_verdict = "unverifiable"
    elif not conditions["two_or_more_real_decision_entities"] or not conditions["independent_local_observations"] or not conditions["entity_owned_actions"]:
        entity_verdict = "not_supported"
    else:
        entity_verdict = "partially_supported"
    centralized_benefit = "unverifiable"
    if trace_available:
        local_conflicts = aliasing["scopes"]["actor_local"]["coarsened"]["conflicting_oracle_action_group_count"]
        global_conflicts = aliasing["scopes"]["controller_global"]["coarsened"]["conflicting_oracle_action_group_count"]
        centralized_benefit = "supported" if local_conflicts > global_conflicts else "not_supported"
    factorized = "partially_supported" if architecture.get("factorized_controller") else "not_supported"
    overall = "UNVERIFIABLE" if entity_verdict == "unverifiable" else ("SUPPORTED" if entity_verdict == "supported" else ("PARTIALLY_SUPPORTED" if entity_verdict == "partially_supported" else "NOT_SUPPORTED"))
    return {
        "overall_verdict": overall,
        "single_controller_sufficient_in_principle": "partially_supported" if architecture.get("joint_action_is_centrally_selected_at_execution") else "unverifiable",
        "centralized_information_beneficial": centralized_benefit,
        "factorized_decision_beneficial": factorized,
        "entity_level_marl_evidence": entity_verdict,
        "gnn_gat_necessity": "unverifiable" if not trace_available else "not_supported",
        "necessary_condition_gate": conditions,
        "all_entity_level_conditions_met": all(conditions.values()),
        "cross_rsu_unique_request_count": cross_count,
        "independent_evaluation_unit_count": independent_evaluation_unit_count,
        "blockers": [name for name, met in conditions.items() if not met],
        "prohibited_claims": [
            "MAPPO naming, multiple heads, parameter sharing, a centralized critic, or graph structure proves entity-level MARL necessity",
            "centralized information benefit is equivalent to MARL necessity",
            "mutual information or conditional mutual information is causal",
            "replicated horizon/baseline rows are independent samples",
            "GNN/GAT is necessary without matched feature-value evidence",
        ],
    }


def build_synthetic_validation_report() -> dict[str, Any]:
    """Return deterministic hand-checkable cases for the G10 diagnostic gates."""
    local_global_rows = [
        {"request_id": "q1", "object_size_mb": 1.0, "local": {"x": 0}, "global": {"x": 0, "cross_rsu": 0}, "oracle_action": "noop"},
        {"request_id": "q2", "object_size_mb": 1.0, "local": {"x": 0}, "global": {"x": 0, "cross_rsu": 1}, "oracle_action": "admit"},
    ]
    local_alias = _group_aliases(local_global_rows, "local", "oracle_action")
    global_disambiguation = _group_aliases(local_global_rows, "global", "oracle_action")
    global_conflict_rows = deepcopy(local_global_rows)
    global_conflict_rows[1]["global"] = deepcopy(global_conflict_rows[0]["global"])
    global_alias = _group_aliases(global_conflict_rows, "global", "oracle_action")
    stat_config = {
        **DEFAULT_AUDIT_CONFIG,
        "minimum_information_samples": 4,
        "minimum_independent_evaluation_units": 2,
    }
    information = empirical_information_statistics(
        [0, 0, 1, 1], [0, 0, 1, 1], independent_unit_count=2, config=stat_config,
    )
    all_conditions = {
        key: True for key in (
            "two_or_more_real_decision_entities", "independent_local_observations", "entity_owned_actions",
            "concurrent_or_coupled_actions", "irreducible_local_information_limit",
            "stable_cross_entity_incremental_information", "nontrivial_cross_rsu_opportunity",
            "multiple_independent_evaluation_units", "centralized_controller_lacks_required_information",
        )
    }
    synthetic_marl = evaluate_marl_necessity(
        architecture=build_architecture_audit("mappo"),
        aliasing={"availability": "available", "scopes": {"actor_local": {"coarsened": {"conflicting_oracle_action_group_count": 1}}, "controller_global": {"coarsened": {"conflicting_oracle_action_group_count": 0}}}},
        opportunity_rows=[], independent_evaluation_unit_count=2, synthetic_evidence=all_conditions,
    )
    return {
        "synthetic_validation_version": "1.0.0",
        "diagnostic_only_not_policy_training": True,
        "single_controller_sufficient_case": {"verdict": "supported", "reason": "one controller observes the complete synthetic state and emits the only joint action"},
        "local_alias_global_disambiguation_case": {"local": local_alias, "global": global_disambiguation, "passed": local_alias["conflicting_oracle_action_group_count"] == 1 and global_disambiguation["conflicting_oracle_action_group_count"] == 0},
        "global_observation_still_insufficient_case": {"global": global_alias, "passed": global_alias["conflicting_oracle_action_group_count"] == 1},
        "entity_level_marl_all_conditions_case": {"verdict": synthetic_marl["entity_level_marl_evidence"], "gate": synthetic_marl["necessary_condition_gate"], "passed": synthetic_marl["entity_level_marl_evidence"] == "supported"},
        "hand_checkable_information_case": {"labels": [0, 0, 1, 1], "projection": [0, 0, 1, 1], "expected_entropy_bits": 1.0, "expected_nmi": 1.0, "result": information},
        "feature_removal_projection_names": sorted(DEFAULT_AUDIT_CONFIG["feature_removal_projections"]),
    }


def audit_information_sufficiency(*, manifest: dict[str, Any], replay: dict[str, Any], oracle_action_trace: dict[str, Any], opportunity_rows_payload: dict[str, Any], agent_identity: str, observation_trace: dict[str, Any] | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved_config = deepcopy(DEFAULT_AUDIT_CONFIG)
    if config:
        resolved_config.update(deepcopy(config))
    validation = validate_audit_inputs(manifest=manifest, replay=replay, oracle_action_trace=oracle_action_trace, opportunity_rows_payload=opportunity_rows_payload, observation_trace=observation_trace)
    if validation["status"] != "pass":
        raise InformationSufficiencyAuditError("; ".join(validation["errors"]))
    rows = deepcopy(opportunity_rows_payload["rows"])
    architecture = build_architecture_audit(agent_identity)
    field_map = build_observation_field_map(agent_identity, rows)
    coverage = summarize_schema_coverage(field_map, rows)
    recoverability = analyze_recoverability(observation_trace, resolved_config)
    aliasing, information_gain = analyze_aliasing_and_information(observation_trace, replay, oracle_action_trace, resolved_config)
    identifiability = analyze_opportunity_identifiability(field_map, rows, observation_trace is not None)
    independent_units = len({str(row.get("evaluation_unit_id")) for row in replay.get("requests", [])})
    marl = evaluate_marl_necessity(architecture=architecture, aliasing=aliasing, opportunity_rows=rows, independent_evaluation_unit_count=independent_units)
    evidence_level = "E2_ARTIFACT_AUDITED" if observation_trace is not None and aliasing.get("availability") == "available" else "E1_DOCUMENTED_CONTRACT_PLUS_MATCHED_G07_G09_WITHOUT_DECISION_TRACE"
    identity = {
        "information_sufficiency_audit_contract_version": INFORMATION_SUFFICIENCY_AUDIT_CONTRACT_VERSION,
        "decision_observation_trace_version": DECISION_OBSERVATION_TRACE_VERSION if observation_trace is not None else "unavailable",
        "auditor_identity": AUDITOR_IDENTITY,
        "agent_identity": agent_identity,
        "request_replay_fingerprint": replay.get("request_replay_fingerprint"),
        "g07_manifest_id": manifest.get("identity", {}).get("manifest_id"),
        "g09_analysis_fingerprint": opportunity_rows_payload.get("provenance", {}).get("analysis_fingerprint"),
        "reviewed_at": "2026-08-19",
        "literature_cutoff": "2026-08-14",
        "target_venue": "IEEE TMC",
        "review_policy_version": "tmc_review_policy_v3_20260621",
        "g07_manifest_git_commit": manifest.get("identity", {}).get("git_commit"),
    }
    identity["audit_fingerprint"] = _sha256_value({"identity": identity, "config": resolved_config, "architecture": architecture, "field_map": field_map, "recoverability": recoverability, "aliasing": aliasing, "information_gain": information_gain, "identifiability": identifiability, "marl": marl})
    return {
        "identity": identity,
        "resolved_config": resolved_config,
        "architecture_audit": architecture,
        "observation_field_map": {"schema_coverage": coverage, "rows": field_map},
        "observation_recoverability": recoverability,
        "observation_aliasing": aliasing,
        "opportunity_identifiability": identifiability,
        "information_gain": information_gain,
        "marl_necessity_verdict": {
            **marl,
            "evidence_level": evidence_level,
            "architecture_classification": architecture["current_architecture_classification"],
            "conclusion_matrix": {
                "observation_schema_coverage": "PARTIALLY_SUPPORTED" if coverage["actor_visible_count"] else "NOT_SUPPORTED",
                "actor_local_sufficiency": "UNVERIFIABLE" if aliasing.get("availability") != "available" else "PARTIALLY_SUPPORTED",
                "controller_level_sufficiency": "UNVERIFIABLE" if aliasing.get("availability") != "available" else "PARTIALLY_SUPPORTED",
                "predictor_sufficiency": "UNVERIFIABLE" if aliasing.get("availability") != "available" else "PARTIALLY_SUPPORTED",
                "cross_rsu_information_value": "UNVERIFIABLE" if information_gain.get("availability") != "available" else marl["centralized_information_beneficial"].upper(),
                "centralized_information_benefit": marl["centralized_information_beneficial"].upper(),
                "factorized_decision_benefit": marl["factorized_decision_beneficial"].upper(),
                "entity_level_marl_necessity": marl["entity_level_marl_evidence"].upper(),
                "gnn_gat_necessity": marl["gnn_gat_necessity"].upper(),
            },
            "required_next_evidence": [
                "matched pre-action decision_observation_trace_v1 on the exact G08 replay",
                "multiple independent non-overlapping evaluation units with nontrivial cross-RSU opportunities",
                "real entity-bound local observations and entity-owned coupled actions before an entity-level MARL claim",
                "pre-registered graph/cross-RSU/handoff/cache-state projections followed by independent algorithm evaluation",
            ],
        },
        "input_validation_report": validation,
    }
