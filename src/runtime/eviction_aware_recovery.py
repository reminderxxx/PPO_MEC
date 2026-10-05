"""Small opt-in recovery rules for the bounded cache-coupling witness.

The functions in this module are pure.  They consume a read-only preview from
the existing typed-cache transaction and never inspect realized future events.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence


RERUN_ACTION_ID = 0
RECOVER_ACTION_ID = 4


@dataclass(frozen=True)
class RecoveryDecisionInputs:
    """Decision-time fields shared by the eviction-aware online rules."""

    action4_legal: bool
    recovery_preview_feasible: bool
    current_missing_object_ids: tuple[str, ...]
    legal_victim_ids: tuple[str, ...]
    near_term_dependency_ids: tuple[str, ...]
    near_term_missing_before_ids: tuple[str, ...]
    near_term_missing_after_ids: tuple[str, ...]
    transfer_bytes_by_object: Mapping[str, int]
    state_package_bytes: int | None
    rerun_input_bytes: int | None
    effective_link_mbps: float | None
    positive_transfer_fixed_latency_seconds: float | None
    state_restore_overhead_seconds: float | None
    prefix_recompute_seconds: float | None
    future_reload_cost_scale: float = 1.0


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _network_seconds(
    byte_count: int,
    *,
    effective_link_mbps: float,
    fixed_latency_seconds: float,
) -> float:
    if byte_count <= 0:
        return 0.0
    return byte_count * 8.0 / (effective_link_mbps * 1_000_000.0) + fixed_latency_seconds


def _fallback(reason: str, *, method: str, opt_in_enabled: bool) -> dict[str, object]:
    return {
        "method": method,
        "selected_first_action": RERUN_ACTION_ID,
        "decision": "rerun",
        "opt_in_enabled": opt_in_enabled,
        "conservative_fallback": True,
        "fallback_reason": reason,
    }


def _validated_cost_context(inputs: RecoveryDecisionInputs) -> dict[str, object] | None:
    numeric = {
        "effective_link_mbps": inputs.effective_link_mbps,
        "positive_transfer_fixed_latency_seconds": inputs.positive_transfer_fixed_latency_seconds,
        "state_restore_overhead_seconds": inputs.state_restore_overhead_seconds,
        "prefix_recompute_seconds": inputs.prefix_recompute_seconds,
        "future_reload_cost_scale": inputs.future_reload_cost_scale,
    }
    if inputs.state_package_bytes is None or inputs.rerun_input_bytes is None:
        return None
    if inputs.state_package_bytes < 0 or inputs.rerun_input_bytes < 0:
        return None
    for key, value in numeric.items():
        if value is None or not math.isfinite(float(value)):
            return None
        if key == "effective_link_mbps" and float(value) <= 0.0:
            return None
        if key != "effective_link_mbps" and float(value) < 0.0:
            return None
    object_ids = set(_unique(inputs.current_missing_object_ids))
    object_ids.update(_unique(inputs.near_term_missing_before_ids))
    object_ids.update(_unique(inputs.near_term_missing_after_ids))
    for object_id in object_ids:
        value = inputs.transfer_bytes_by_object.get(object_id)
        if value is None or int(value) < 0:
            return None
    return numeric


def _cost_of_objects(
    object_ids: Sequence[str],
    *,
    inputs: RecoveryDecisionInputs,
) -> tuple[int, float]:
    unique_ids = _unique(object_ids)
    byte_count = sum(int(inputs.transfer_bytes_by_object[object_id]) for object_id in unique_ids)
    return byte_count, _network_seconds(
        byte_count,
        effective_link_mbps=float(inputs.effective_link_mbps),
        fixed_latency_seconds=float(inputs.positive_transfer_fixed_latency_seconds),
    )


def original_simple_threshold_decision(inputs: RecoveryDecisionInputs) -> dict[str, object]:
    """Preserved workload-v0.1 threshold; model preparation is treated as common."""

    context = _validated_cost_context(inputs)
    if not inputs.action4_legal:
        return _fallback("action4_illegal", method="original_simple_threshold", opt_in_enabled=False)
    if context is None:
        return _fallback("missing_or_invalid_cost_input", method="original_simple_threshold", opt_in_enabled=False)
    restart_seconds = float(inputs.prefix_recompute_seconds) + _network_seconds(
        int(inputs.rerun_input_bytes),
        effective_link_mbps=float(inputs.effective_link_mbps),
        fixed_latency_seconds=float(inputs.positive_transfer_fixed_latency_seconds),
    )
    recovery_seconds = float(inputs.state_restore_overhead_seconds) + _network_seconds(
        int(inputs.state_package_bytes),
        effective_link_mbps=float(inputs.effective_link_mbps),
        fixed_latency_seconds=float(inputs.positive_transfer_fixed_latency_seconds),
    )
    selected = RECOVER_ACTION_ID if recovery_seconds < restart_seconds else RERUN_ACTION_ID
    return {
        "method": "original_simple_threshold",
        "selected_first_action": selected,
        "decision": "recover" if selected == RECOVER_ACTION_ID else "rerun",
        "opt_in_enabled": False,
        "conservative_fallback": False,
        "restart_incremental_seconds": restart_seconds,
        "recovery_incremental_seconds": recovery_seconds,
        "excluded_model_prepare_reason": (
            "preserved baseline treats target-model preparation as common to both paths"
        ),
    }


def _full_incremental_costs(inputs: RecoveryDecisionInputs) -> dict[str, object] | None:
    if _validated_cost_context(inputs) is None:
        return None
    current_missing = _unique(inputs.current_missing_object_ids)
    victims = _unique(inputs.legal_victim_ids)
    near_dependencies = _unique(inputs.near_term_dependency_ids)
    missing_before = _unique(inputs.near_term_missing_before_ids)
    missing_after = _unique(inputs.near_term_missing_after_ids)
    induced_reload = tuple(sorted(set(missing_after) - set(missing_before)))
    avoided_future_load = tuple(sorted(set(missing_before) - set(missing_after)))
    if not set(induced_reload).issubset(set(victims)):
        return None
    if (victims or induced_reload) and not near_dependencies:
        return None

    current_bytes, current_seconds = _cost_of_objects(current_missing, inputs=inputs)
    before_bytes, before_seconds = _cost_of_objects(missing_before, inputs=inputs)
    after_bytes, after_seconds_unscaled = _cost_of_objects(missing_after, inputs=inputs)
    induced_bytes, induced_seconds_unscaled = _cost_of_objects(induced_reload, inputs=inputs)
    state_seconds = _network_seconds(
        int(inputs.state_package_bytes),
        effective_link_mbps=float(inputs.effective_link_mbps),
        fixed_latency_seconds=float(inputs.positive_transfer_fixed_latency_seconds),
    )
    input_seconds = _network_seconds(
        int(inputs.rerun_input_bytes),
        effective_link_mbps=float(inputs.effective_link_mbps),
        fixed_latency_seconds=float(inputs.positive_transfer_fixed_latency_seconds),
    )
    scale = float(inputs.future_reload_cost_scale)
    after_seconds = after_seconds_unscaled * scale
    future_delta_seconds = after_seconds - before_seconds
    restart_seconds = float(inputs.prefix_recompute_seconds) + input_seconds + before_seconds
    recovery_seconds = (
        current_seconds
        + state_seconds
        + float(inputs.state_restore_overhead_seconds)
        + after_seconds
    )
    common_ids = tuple(sorted(set(missing_before) & set(missing_after)))
    return {
        "restart_incremental_seconds": restart_seconds,
        "recovery_incremental_seconds": recovery_seconds,
        "current_prepare_object_ids": list(current_missing),
        "current_prepare_bytes": current_bytes,
        "legal_victim_ids": list(victims),
        "near_term_dependency_ids": list(near_dependencies),
        "near_term_missing_before_ids": list(missing_before),
        "near_term_missing_after_ids": list(missing_after),
        "induced_reload_object_ids": list(induced_reload),
        "induced_reload_bytes": induced_bytes,
        "induced_reload_seconds_unscaled": induced_seconds_unscaled,
        "avoided_future_load_object_ids": list(avoided_future_load),
        "future_model_cost_delta_seconds": future_delta_seconds,
        "future_reload_cost_scale": scale,
        "state_transfer_seconds": state_seconds,
        "rerun_input_transfer_seconds": input_seconds,
        "future_prepare_before_bytes": before_bytes,
        "future_prepare_after_bytes": after_bytes,
        "common_future_object_ids": list(common_ids),
        "common_cost_exclusion_reason": (
            "objects missing on both paths and successful service time are identical; "
            "they are retained in both full totals and cancel in the comparison"
        ),
    }


def eviction_aware_recovery_decision(
    inputs: RecoveryDecisionInputs,
    *,
    opt_in_enabled: bool,
) -> dict[str, object]:
    """Choose recovery only with complete inputs and a legal native preview."""

    if not opt_in_enabled:
        baseline = original_simple_threshold_decision(inputs)
        return {**baseline, "method": "eviction_aware_recovery", "opt_in_enabled": False}
    if not inputs.action4_legal:
        return _fallback("action4_illegal", method="eviction_aware_recovery", opt_in_enabled=True)
    if not inputs.recovery_preview_feasible:
        return _fallback(
            "native_recovery_preview_infeasible",
            method="eviction_aware_recovery",
            opt_in_enabled=True,
        )
    costs = _full_incremental_costs(inputs)
    if costs is None:
        return _fallback(
            "missing_invalid_or_unexplained_reload_input",
            method="eviction_aware_recovery",
            opt_in_enabled=True,
        )
    selected = (
        RECOVER_ACTION_ID
        if float(costs["recovery_incremental_seconds"])
        < float(costs["restart_incremental_seconds"])
        else RERUN_ACTION_ID
    )
    return {
        "method": "eviction_aware_recovery",
        "selected_first_action": selected,
        "decision": "recover" if selected == RECOVER_ACTION_ID else "rerun",
        "opt_in_enabled": True,
        "conservative_fallback": False,
        "formula": "recover iff J_recovery < J_rerun",
        **costs,
    }


def two_step_lookahead_decision(inputs: RecoveryDecisionInputs) -> dict[str, object]:
    """Explicitly compare the two legal two-step paths with the same information."""

    if not inputs.action4_legal:
        return _fallback("action4_illegal", method="two_step_lookahead", opt_in_enabled=True)
    if not inputs.recovery_preview_feasible:
        return _fallback(
            "native_recovery_preview_infeasible",
            method="two_step_lookahead",
            opt_in_enabled=True,
        )
    costs = _full_incremental_costs(inputs)
    if costs is None:
        return _fallback(
            "missing_invalid_or_unexplained_reload_input",
            method="two_step_lookahead",
            opt_in_enabled=True,
        )
    selected = (
        RECOVER_ACTION_ID
        if float(costs["recovery_incremental_seconds"])
        < float(costs["restart_incremental_seconds"])
        else RERUN_ACTION_ID
    )
    return {
        "method": "two_step_lookahead",
        "selected_first_action": selected,
        "decision": "recover" if selected == RECOVER_ACTION_ID else "rerun",
        "opt_in_enabled": True,
        "conservative_fallback": False,
        "candidate_paths": {
            "rerun_then_declared_next": costs["restart_incremental_seconds"],
            "recover_then_declared_next": costs["recovery_incremental_seconds"],
        },
        **costs,
    }


def decision_inputs_to_dict(inputs: RecoveryDecisionInputs) -> dict[str, object]:
    """Return a detached JSON-compatible record for artifact provenance."""

    payload = asdict(inputs)
    payload["transfer_bytes_by_object"] = dict(inputs.transfer_bytes_by_object)
    return payload
