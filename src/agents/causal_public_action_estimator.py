"""Causal, side-effect-free action estimates for calibrated workflows.

The estimator consumes only the public semantic state exposed to every method.
It deliberately returns ``unknown`` instead of filling missing online fields from
an environment instance, exact transition clone, or realized future route.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from src.envs.specs.action_schema import ActionMaskBuilder


YES = "yes"
NO = "no"
UNKNOWN = "unknown"
NOT_APPLICABLE = "not_applicable"
SCHEMA_VERSION = "causal_public_action_estimator_v2"
EVENT_LABEL_SCHEMA_VERSION = "causal_public_prepare_advantage_v2"


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _nonnegative(value: Any) -> float | None:
    parsed = _finite(value)
    return parsed if parsed is not None and parsed >= 0.0 else None


def _network_seconds(byte_count: int | None, mbps: float | None, fixed: float | None) -> float | None:
    if byte_count is None or mbps is None or fixed is None or mbps <= 0.0:
        return None
    if byte_count <= 0:
        return 0.0
    return float(byte_count) * 8.0 / (mbps * 1_000_000.0) + fixed


def _primary_vehicle(state: dict[str, Any]) -> dict[str, Any]:
    vehicles = [dict(row) for row in state.get("vehicles", []) if isinstance(row, dict)]
    primary = state.get("primary_vehicle_id")
    if primary is not None:
        for vehicle in vehicles:
            if str(vehicle.get("vehicle_id", "")) == str(primary):
                return vehicle
    return vehicles[0] if vehicles else {}


def _rsu(state: dict[str, Any], rsu_id: Any) -> dict[str, Any]:
    if rsu_id is None:
        return {}
    return next(
        (
            dict(row)
            for row in state.get("rsus", [])
            if isinstance(row, dict) and str(row.get("rsu_id")) == str(rsu_id)
        ),
        {},
    )


def _prediction_target(state: dict[str, Any], vehicle_id: str) -> str | None:
    predictions = dict(state.get("predictions", {}) or {})
    for field in (
        "predicted_first_handoff_rsu_by_vehicle",
        "predicted_handoff_target_rsu_id_by_vehicle",
        "predicted_next_rsu_by_vehicle",
    ):
        value = dict(predictions.get(field, {}) or {}).get(vehicle_id)
        if value is not None:
            return str(value)
    return None


def _bundle(
    state: dict[str, Any],
    rsu_id: str | None,
) -> dict[str, Any]:
    context = dict(state.get("calibrated_context", {}) or {})
    required = [str(item) for item in context.get("required_bundle_ids", [])]
    catalog = dict(context.get("object_catalog", {}) or {})
    rsu = _rsu(state, rsu_id)
    if rsu_id is None or not rsu or not required or not catalog:
        return {
            "readiness": UNKNOWN,
            "admission": UNKNOWN,
            "missing_ids": [],
            "resident_bytes": None,
            "transfer_bytes": None,
            "load_seconds": None,
            "reason": "missing_public_bundle_catalog_or_rsu",
        }
    residents = {str(item) for item in rsu.get("typed_resident_object_ids", [])}
    missing = [item for item in required if item not in residents]
    try:
        resident_bytes = sum(int(catalog[item]["resident_bytes"]) for item in missing)
        transfer_bytes = sum(int(catalog[item]["transfer_bytes"]) for item in missing)
        load_seconds = sum(float(catalog[item]["load_seconds"]) for item in missing)
    except (KeyError, TypeError, ValueError):
        return {
            "readiness": UNKNOWN,
            "admission": UNKNOWN,
            "missing_ids": missing,
            "resident_bytes": None,
            "transfer_bytes": None,
            "load_seconds": None,
            "reason": "incomplete_public_object_catalog",
        }
    if not missing:
        admission = YES
        reason = "bundle_already_ready"
    else:
        capacity = _nonnegative(rsu.get("cache_capacity"))
        used = _nonnegative(rsu.get("cache_used_bytes"))
        if capacity is None or used is None:
            admission = UNKNOWN
            reason = "missing_public_cache_capacity"
        elif resident_bytes <= max(capacity - used, 0.0) + 1e-6:
            admission = YES
            reason = "fits_without_eviction"
        else:
            admission = UNKNOWN
            reason = "requires_private_eviction_order"
    return {
        "readiness": YES if not missing else NO,
        "admission": admission,
        "missing_ids": missing,
        "resident_bytes": resident_bytes,
        "transfer_bytes": transfer_bytes,
        "load_seconds": load_seconds,
        "reason": reason,
    }


def _prefix_status(state: dict[str, Any], key: str) -> dict[str, Any]:
    context = dict(state.get("calibrated_context", {}) or {})
    prepared = dict(context.get("prepared_state_prefix", {}) or {})
    row = prepared.get(key)
    if not isinstance(row, dict) or not bool(row.get("known", False)):
        return {"status": UNKNOWN, "valid": None, "missing_fraction": None}
    if bool(row.get("valid", False)):
        status = "valid"
    elif bool(row.get("exists", False)):
        status = "stale"
    else:
        status = "missing"
    return {
        "status": status,
        "valid": bool(row.get("valid", False)),
        "missing_fraction": _nonnegative(row.get("missing_completed_fraction")),
    }


def _known_recompute_seconds(state: dict[str, Any]) -> tuple[float | None, str]:
    workflow = dict(state.get("workflow", {}) or {})
    completed = list(workflow.get("completed_node_ids", []) or [])
    if not completed:
        return 0.0, "no_completed_prefix"
    current = _prefix_status(state, "current")
    if current["valid"] is True:
        return 0.0, "current_prefix_valid"
    return None, "handoff_and_missing_predecessor_identity_unknown"


def _sum_known(parts: list[float | None]) -> float | None:
    return None if any(item is None for item in parts) else float(sum(item for item in parts if item is not None))


def _tri_leq(left: float | None, right: float | None) -> str:
    if left is None or right is None:
        return UNKNOWN
    return YES if left <= right + 1e-9 else NO


def _service_status(action: int, current_bundle: dict[str, Any]) -> str:
    if action == 2:
        return YES
    if action == 0:
        if current_bundle["readiness"] == YES or current_bundle["admission"] == YES:
            return YES
        return UNKNOWN
    if current_bundle["readiness"] == YES:
        return YES
    if current_bundle["readiness"] == NO:
        return NO
    return UNKNOWN


def estimate_public_actions(
    semantic_state: dict[str, Any],
    action_mask: list[bool] | None = None,
) -> dict[str, Any]:
    """Estimate all five actions without mutating or consulting an environment."""

    state = semantic_state
    context = dict(state.get("calibrated_context", {}) or {})
    node = dict(state.get("current_workflow_node", {}) or {})
    vehicle = _primary_vehicle(state)
    vehicle_id = str(vehicle.get("vehicle_id", ""))
    current_rsu_id = vehicle.get("associated_rsu_id")
    target_rsu_id = _prediction_target(state, vehicle_id)
    if target_rsu_id is not None and str(target_rsu_id) == str(current_rsu_id):
        target_rsu_id = None
    mask = list(action_mask) if action_mask is not None else ActionMaskBuilder().build_mask(state)
    if len(mask) != 5:
        raise ValueError("public action mask must contain five entries")

    current_bundle = _bundle(state, str(current_rsu_id) if current_rsu_id is not None else None)
    target_bundle = _bundle(state, target_rsu_id)
    current_prefix = _prefix_status(state, "current")
    target_prefix = _prefix_status(state, "predicted_target")
    link = dict(context.get("link", {}) or {})
    mbps = _nonnegative(link.get("estimated_mbps"))
    fixed = _nonnegative(link.get("fixed_seconds"))
    contact = _nonnegative(context.get("contact_budget_seconds"))
    compute = _nonnegative(context.get("compute_seconds", node.get("compute_seconds")))
    state_bytes = context.get("state_bytes", node.get("state_bytes"))
    input_bytes = context.get("input_bytes", node.get("input_bytes"))
    try:
        state_bytes = int(state_bytes) if int(state_bytes) >= 0 else None
    except (TypeError, ValueError):
        state_bytes = None
    try:
        input_bytes = int(input_bytes) if int(input_bytes) >= 0 else None
    except (TypeError, ValueError):
        input_bytes = None
    measured = dict(context.get("measured_time_seconds", {}) or {})
    restore = _nonnegative(measured.get("state_restore_overhead"))
    fallback = _nonnegative(context.get("vehicle_fallback_seconds"))
    failure = _nonnegative(context.get("failed_service_seconds"))
    time_contract = dict(context.get("time_contract", {}) or {})
    remaining_deadline = _nonnegative(time_contract.get("remaining_deadline_seconds"))
    recompute, recompute_reason = _known_recompute_seconds(state)
    raw_full_step_contact_contract = (
        str(state.get("interface_profile", "")) == "raw_ngsim_event_time_v1"
    )

    current_model_network = _network_seconds(current_bundle["transfer_bytes"], mbps, fixed)
    current_model_prepare = _sum_known([current_model_network, current_bundle["load_seconds"]])
    target_model_network = _network_seconds(target_bundle["transfer_bytes"], mbps, fixed)
    target_model_prepare = _sum_known([target_model_network, target_bundle["load_seconds"]])
    state_network = _network_seconds(state_bytes, mbps, fixed)
    target_state_prepare = _sum_known([target_model_prepare, state_network, restore])
    input_network = _network_seconds(input_bytes, mbps, fixed)

    actions: dict[str, dict[str, Any]] = {}
    for action in range(5):
        legal = bool(mask[action])
        service = _service_status(action, current_bundle) if legal else NO
        target_prepare = "not_applicable"
        target_state_commit = "not_applicable"
        model_bytes: int | None = 0
        transfer_state_bytes: int | None = 0
        transfer_input_bytes: int | None = 0
        known_parts: list[float | None] = []
        unknown_reasons: list[str] = []
        phase_status = {
            "current_model_prepare": NOT_APPLICABLE,
            "target_model_prepare": NOT_APPLICABLE,
            "current_service": NOT_APPLICABLE,
            "target_state_commit": NOT_APPLICABLE,
        }
        phase_seconds: dict[str, float | None] = {
            key: None for key in phase_status
        }

        if not legal:
            unknown_reasons.append("action_masked")
            total = None
        else:
            if action == 0 and current_bundle["readiness"] == NO:
                if current_bundle["admission"] == YES:
                    known_parts.append(current_model_prepare)
                    model_bytes = int(current_bundle["transfer_bytes"] or 0)
                    phase_status["current_model_prepare"] = "known_estimate"
                    phase_seconds["current_model_prepare"] = current_model_prepare
                else:
                    unknown_reasons.append(str(current_bundle["reason"]))
                    known_parts.append(None)
                    model_bytes = None
                    phase_status["current_model_prepare"] = "unknown"
            if action in {1, 4}:
                prepare_seconds = target_model_prepare if action == 1 else target_state_prepare
                if target_rsu_id is None:
                    target_prepare = NO
                    phase_status["target_model_prepare"] = "skipped_no_target"
                elif target_bundle["admission"] != YES:
                    target_prepare = UNKNOWN
                    unknown_reasons.append(str(target_bundle["reason"]))
                    known_parts.append(None)
                    model_bytes = None
                    phase_status["target_model_prepare"] = "unknown"
                else:
                    target_prepare = _tri_leq(prepare_seconds, contact)
                    if target_prepare == YES:
                        # Native action 4 only commits model staging before the
                        # current service. State transfer/restore is charged
                        # after that service succeeds.
                        known_parts.append(target_model_prepare)
                        model_bytes = int(target_bundle["transfer_bytes"] or 0)
                        phase_status["target_model_prepare"] = "known_estimate"
                        phase_seconds["target_model_prepare"] = target_model_prepare
                    elif target_prepare == NO:
                        phase_status["target_model_prepare"] = "rolled_back_contact"
                    elif target_prepare == UNKNOWN:
                        unknown_reasons.append("missing_public_contact_or_prepare_time")
                        known_parts.append(None)
                        model_bytes = None
                        phase_status["target_model_prepare"] = "unknown"
                if action == 4:
                    if service == YES and target_prepare == YES:
                        target_state_commit = YES
                    elif service == NO or target_prepare == NO:
                        target_state_commit = NO
                    else:
                        target_state_commit = UNKNOWN

            if service == NO:
                known_parts.append(failure)
                phase_seconds["current_service"] = failure
                if failure is None:
                    unknown_reasons.append("missing_public_failed_service_seconds")
                    phase_status["current_service"] = "unknown"
                else:
                    phase_status["current_service"] = "known_failure_estimate"
            elif service == YES:
                if action == 2:
                    known_parts.extend([fallback, compute, input_network])
                    phase_seconds["current_service"] = _sum_known(
                        [fallback, compute, input_network]
                    )
                    transfer_input_bytes = (
                        int(input_bytes) if input_bytes is not None else None
                    )
                    if fallback is None:
                        unknown_reasons.append("missing_public_vehicle_fallback_seconds")
                    if compute is None:
                        unknown_reasons.append("missing_public_compute_seconds")
                    if input_network is None:
                        unknown_reasons.append("missing_public_input_transfer_time")
                else:
                    known_parts.extend([compute, recompute])
                    phase_seconds["current_service"] = _sum_known(
                        [compute, recompute]
                    )
                    if compute is None:
                        unknown_reasons.append("missing_public_compute_seconds")
                    if recompute is None:
                        unknown_reasons.append(recompute_reason)
                phase_status["current_service"] = (
                    "known_estimate"
                    if phase_seconds["current_service"] is not None
                    else "unknown"
                )
            else:
                unknown_reasons.append("current_service_feasibility_unknown")
                known_parts.append(None)
                phase_status["current_service"] = "unknown"

            if action == 4:
                if target_prepare == YES and service == YES:
                    known_parts.extend([state_network, restore])
                    phase_seconds["target_state_commit"] = _sum_known(
                        [state_network, restore]
                    )
                    phase_status["target_state_commit"] = (
                        "known_estimate"
                        if phase_seconds["target_state_commit"] is not None
                        else "unknown"
                    )
                    if phase_seconds["target_state_commit"] is None:
                        unknown_reasons.append(
                            "missing_public_state_commit_time"
                        )
                    transfer_state_bytes = (
                        int(state_bytes) if state_bytes is not None else None
                    )
                elif target_prepare == YES and service == NO:
                    phase_status["target_state_commit"] = (
                        "skipped_current_service_failed"
                    )
                    transfer_state_bytes = 0
                elif target_prepare == YES:
                    known_parts.append(None)
                    phase_status["target_state_commit"] = "conditional_unknown"
                    transfer_state_bytes = None
                elif target_prepare == UNKNOWN:
                    phase_status["target_state_commit"] = "conditional_unknown"
                    transfer_state_bytes = None
                else:
                    phase_status["target_state_commit"] = (
                        "skipped_target_prepare"
                    )
                    transfer_state_bytes = 0
            total = _sum_known(known_parts)

        if legal and raw_full_step_contact_contract and action != 2:
            raw_full_step_contact_fit = _tri_leq(total, contact)
        else:
            raw_full_step_contact_fit = NOT_APPLICABLE
        if legal and raw_full_step_contact_contract:
            raw_trace_fit = UNKNOWN
            raw_execution_fit = (
                NO if raw_full_step_contact_fit == NO else UNKNOWN
            )
        else:
            raw_trace_fit = NOT_APPLICABLE
            raw_execution_fit = NOT_APPLICABLE
        cost_status = (
            "not_executed"
            if not legal
            else (
                "known_conditional_estimate"
                if total is not None
                else "unknown_required_phase"
            )
        )

        readiness_gain = "none"
        if legal and action == 0 and service == YES and current_bundle["readiness"] == NO:
            readiness_gain = "current_bundle"
        elif legal and action == 1 and target_prepare == YES and target_bundle["readiness"] == NO:
            readiness_gain = "target_bundle"
        elif legal and action == 4 and target_prepare == YES and service == YES:
            if target_bundle["readiness"] == NO or target_prefix["valid"] is False:
                readiness_gain = "target_bundle_and_state"

        actions[str(action)] = {
            "legal": legal,
            "current_service": service,
            "target_prepare": target_prepare,
            "target_state_commit": target_state_commit,
            "future_readiness_gain": readiness_gain,
            "cost_status": cost_status,
            "cost_basis": "public_conditional_estimate_not_realized_cost",
            "estimated_conditional_seconds": total,
            "estimated_total_seconds": total,
            "actual_executed_seconds": None,
            "actual_execution_cost_status": "unavailable_online",
            "deadline_fit": _tri_leq(total, remaining_deadline),
            "target_prepare_contact_fit": target_prepare,
            "raw_full_step_contact_fit": raw_full_step_contact_fit,
            "raw_trace_fit": raw_trace_fit,
            "raw_execution_fit": raw_execution_fit,
            "phase_status": phase_status,
            "estimated_phase_seconds": phase_seconds,
            "model_transfer_bytes": model_bytes,
            "state_transfer_bytes": transfer_state_bytes,
            "input_transfer_bytes": transfer_input_bytes,
            "unknown_reasons": sorted(set(unknown_reasons)),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "public_only": True,
        "current_rsu_id": current_rsu_id,
        "predicted_target_rsu_id": target_rsu_id,
        "current_bundle": current_bundle,
        "target_bundle": target_bundle,
        "current_prefix": current_prefix,
        "target_prefix": target_prefix,
        "time_contract_schema": time_contract.get("schema_version"),
        "remaining_deadline_seconds": remaining_deadline,
        "contact_budget_seconds": contact,
        "recompute_seconds": recompute,
        "recompute_reason": recompute_reason,
        "actions": actions,
        "consumed_fields": [
            "primary_vehicle_id/vehicles.associated_rsu_id",
            "current_workflow_node/workflow",
            "predictions causal target/sequence",
            "rsus typed residents/capacity/used bytes",
            "calibrated_context required bundle/object catalog/prepared prefix",
            "calibrated_context estimated link/contact/node costs",
            "calibrated_context time_contract deadline fields",
            "calibrated_context vehicle_fallback_seconds/failed_service_seconds",
        ],
        "forbidden_fields": [
            "instance.rsu_sequence",
            "actual link",
            "realized future position/contact",
            "environment clone/step",
            "reward/outcome label",
        ],
    }


def public_prepare_advantage_label(
    semantic_state: dict[str, Any],
    action_mask: list[bool] | None = None,
) -> dict[str, Any]:
    """Return the frozen prepare/serve/abstain event-supervision decision."""

    estimate = estimate_public_actions(semantic_state, action_mask)
    action4 = estimate["actions"]["4"]
    service_actions = [
        estimate["actions"][str(action)]
        for action in (0, 2, 3)
        if estimate["actions"][str(action)]["legal"]
    ]
    service_fit = any(
        row["current_service"] == YES and row["deadline_fit"] == YES
        for row in service_actions
    )
    if not action4["legal"]:
        decision, reason = "abstain", "action4_not_legal"
    elif action4["current_service"] == NO:
        decision, reason = "serve", "action4_fails_current_service"
    elif action4["target_prepare"] == NO:
        decision, reason = "serve", "target_prepare_infeasible"
    elif action4["raw_full_step_contact_fit"] == NO:
        decision, reason = "abstain", "raw_full_step_contact_insufficient"
    elif action4["deadline_fit"] == NO and service_fit:
        decision, reason = "serve", "prepare_misses_deadline_while_service_fits"
    elif (
        action4["current_service"] == YES
        and action4["target_prepare"] == YES
        and action4["deadline_fit"] == YES
        and action4["future_readiness_gain"] == "target_bundle_and_state"
        and any(row["current_service"] == YES for row in service_actions)
    ):
        decision, reason = "prepare", "service_safe_feasible_future_readiness_gain"
    else:
        decision, reason = "abstain", "decisive_public_comparison_unavailable"
    return {
        "schema_version": EVENT_LABEL_SCHEMA_VERSION,
        "decision": decision,
        "event_target": 1 if decision == "prepare" else 0,
        "event_soft_target": 1.0 if decision == "prepare" else 0.0,
        "supervision_weight": 0.0 if decision == "abstain" else 1.0,
        "reason": reason,
        "estimate": estimate,
    }


def _score(row: dict[str, Any]) -> tuple[Any, ...]:
    service_rank = {NO: 0, UNKNOWN: 1, YES: 2}[str(row["current_service"])]
    deadline_rank = {NO: 0, UNKNOWN: 1, YES: 2}[str(row["deadline_fit"])]
    gain_rank = {
        "none": 0,
        "current_bundle": 1,
        "target_bundle": 1,
        "target_bundle_and_state": 2,
    }.get(str(row["future_readiness_gain"]), 0)
    known = row["estimated_total_seconds"] is not None
    total = float(row["estimated_total_seconds"] or 0.0)
    byte_fields = (
        row["model_transfer_bytes"],
        row["state_transfer_bytes"],
        row["input_transfer_bytes"],
    )
    bytes_total = sum(int(value) for value in byte_fields if value is not None)
    return service_rank, deadline_rank, int(known), gain_rank, -total, -bytes_total


class CausalPublicImmediateRule:
    """Myopic rule over the public estimator; never previews an environment."""

    method_name = "causal_public_immediate_rule"

    def select_action_from_info(self, info: dict[str, Any]) -> int:
        estimate = estimate_public_actions(
            dict(info.get("semantic_state", {}) or {}),
            list(info.get("action_mask", []) or []),
        )
        candidates = [
            (action, row)
            for action, row in ((int(key), value) for key, value in estimate["actions"].items())
            if row["legal"]
        ]
        if not candidates:
            raise RuntimeError("public action estimator received no legal action")
        return max(candidates, key=lambda item: (_score(item[1]), -item[0]))[0]

    def select_action(self, source: Any) -> int:
        info = source if isinstance(source, dict) else source._info()
        return self.select_action_from_info(info)


def project_public_state(
    semantic_state: dict[str, Any],
    estimate: dict[str, Any],
    action: int,
) -> tuple[dict[str, Any] | None, str]:
    """Project only effects that are identifiable from the public estimate."""

    row = estimate["actions"][str(int(action))]
    if row["current_service"] != YES or row["estimated_total_seconds"] is None:
        return None, "first_step_service_or_time_unknown"
    state = deepcopy(semantic_state)
    workflow = dict(state.get("workflow", {}) or {})
    order = [str(item) for item in workflow.get("execution_order", [])]
    current_node = dict(state.get("current_workflow_node", {}) or {})
    current_node_id = str(current_node.get("node_id", ""))
    if not current_node_id or current_node_id not in order:
        return None, "public_workflow_order_unknown"
    next_index = order.index(current_node_id) + 1
    if next_index >= len(order):
        return None, "workflow_terminates_after_first_step"
    nodes = {
        str(item.get("node_id")): deepcopy(item)
        for item in workflow.get("nodes", [])
        if isinstance(item, dict)
    }
    if order[next_index] not in nodes:
        return None, "next_public_node_unknown"

    context = dict(state.get("calibrated_context", {}) or {})
    required = [str(item) for item in context.get("required_bundle_ids", [])]
    current_id = estimate.get("current_rsu_id")
    target_id = estimate.get("predicted_target_rsu_id")
    if action == 0 and row["future_readiness_gain"] == "current_bundle":
        rsu = _rsu(state, current_id)
        rsu["typed_resident_object_ids"] = sorted(set(rsu.get("typed_resident_object_ids", [])) | set(required))
        for index, original in enumerate(state.get("rsus", [])):
            if str(original.get("rsu_id")) == str(current_id):
                state["rsus"][index] = rsu
                break
    if action in {1, 4} and row["target_prepare"] == YES and target_id is not None:
        rsu = _rsu(state, target_id)
        rsu["typed_resident_object_ids"] = sorted(set(rsu.get("typed_resident_object_ids", [])) | set(required))
        for index, original in enumerate(state.get("rsus", [])):
            if str(original.get("rsu_id")) == str(target_id):
                state["rsus"][index] = rsu
                break

    completed = [str(item) for item in workflow.get("completed_node_ids", [])]
    if current_node_id not in completed:
        completed.append(current_node_id)
    workflow["completed_node_ids"] = completed
    workflow["current_node_id"] = order[next_index]
    next_node = nodes[order[next_index]]
    state["workflow"] = workflow
    state["current_workflow_node"] = next_node

    predictions = dict(state.get("predictions", {}) or {})
    vehicle = _primary_vehicle(state)
    vehicle_id = str(vehicle.get("vehicle_id", ""))
    sequence = list(dict(predictions.get("next_rsu_sequence", {}) or {}).get(vehicle_id, []) or [])
    if not sequence:
        return None, "causal_next_rsu_sequence_unknown"
    next_rsu = str(sequence[0])
    for item in state.get("vehicles", []):
        if str(item.get("vehicle_id", "")) == vehicle_id:
            item["associated_rsu_id"] = next_rsu
    predictions.setdefault("next_rsu_sequence", {})[vehicle_id] = sequence[1:]
    predictions.setdefault("predicted_next_rsu_by_vehicle", {})[vehicle_id] = (
        sequence[1] if len(sequence) > 1 else None
    )
    predictions.setdefault("predicted_first_handoff_rsu_by_vehicle", {})[vehicle_id] = next(
        (item for item in sequence[1:] if str(item) != next_rsu),
        None,
    )
    state["predictions"] = predictions

    prepared = dict(context.get("prepared_state_prefix", {}) or {})
    if action == 4 and row["target_state_commit"] == YES and str(target_id) == next_rsu:
        prepared["current"] = {
            "known": True,
            "exists": True,
            "valid": True,
            "missing_completed_count": 0,
            "missing_completed_fraction": 0.0,
        }
    else:
        prepared["current"] = prepared.get("predicted_target", {"known": False})
    prepared["predicted_target"] = {"known": False, "exists": False, "valid": False}
    context["prepared_state_prefix"] = prepared
    adapter = str(next_node.get("required_adapter", ""))
    bundle_map = dict(context.get("adapter_to_bundle", {}) or {})
    next_bundle = bundle_map.get(adapter)
    if not isinstance(next_bundle, list) or not next_bundle:
        return None, "next_public_required_bundle_unknown"
    context["required_bundle_ids"] = [str(item) for item in next_bundle]
    context["state_bytes"] = next_node.get("state_bytes")
    context["input_bytes"] = next_node.get("input_bytes")
    context["compute_seconds"] = next_node.get("compute_seconds")
    context["contact_budget_seconds"] = None
    time_contract = dict(context.get("time_contract", {}) or {})
    elapsed = float(row["estimated_total_seconds"])
    if _finite(time_contract.get("clock_seconds")) is not None:
        time_contract["clock_seconds"] = float(time_contract["clock_seconds"]) + elapsed
    if _finite(time_contract.get("remaining_deadline_seconds")) is not None:
        time_contract["remaining_deadline_seconds"] = max(
            float(time_contract["remaining_deadline_seconds"]) - elapsed,
            0.0,
        )
    context["time_contract"] = time_contract
    state["calibrated_context"] = context
    return state, "causal_public_first_step_projection"


class CausalPublicTwoStepRule(CausalPublicImmediateRule):
    """Two-step public rule with explicit fallback when projection is unknown."""

    method_name = "causal_public_two_step_rule"

    def select_action_from_info(self, info: dict[str, Any]) -> int:
        state = dict(info.get("semantic_state", {}) or {})
        mask = list(info.get("action_mask", []) or [])
        first = estimate_public_actions(state, mask)
        candidates: list[tuple[tuple[Any, ...], int]] = []
        for action, row in ((int(key), value) for key, value in first["actions"].items()):
            if not row["legal"]:
                continue
            projected, reason = project_public_state(state, first, action)
            if projected is None:
                score = (_score(row), (0, 0, 0, 0, 0.0, 0))
            else:
                second = estimate_public_actions(projected)
                legal_second = [value for value in second["actions"].values() if value["legal"]]
                second_score = max((_score(value) for value in legal_second), default=(0, 0, 0, 0, 0.0, 0))
                score = (_score(row), second_score)
            candidates.append((score + ((1 if reason == "causal_public_first_step_projection" else 0),), action))
        if not candidates:
            raise RuntimeError("public two-step estimator received no legal action")
        return max(candidates, key=lambda item: (item[0], -item[1]))[1]
