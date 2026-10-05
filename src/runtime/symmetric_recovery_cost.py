"""Pure symmetric lifecycle accounting for restart/recovery decisions.

The caller supplies detached per-path cache transition ledgers.  Online rules
consume only decision-time previews and estimates; realized branch scores are
kept outside this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping


RERUN_ACTION_ID = 0
RECOVER_ACTION_ID = 4


@dataclass(frozen=True)
class LifecycleEventEstimate:
    event_id: str
    required_adapter_id: str
    pre_resident_object_ids: tuple[str, ...]
    victim_object_ids: tuple[str, ...]
    admitted_object_ids: tuple[str, ...]
    post_resident_object_ids: tuple[str, ...]
    transfer_bytes_by_object: Mapping[str, int]
    transaction_status: str
    dependency_safe: bool


@dataclass(frozen=True)
class PathLifecycleEstimate:
    action_id: int
    events: tuple[LifecycleEventEstimate, ...]
    dynamic_transfer_bytes: int
    state_overhead_seconds: float
    recompute_seconds: float
    successful_service_seconds: float


@dataclass(frozen=True)
class SymmetricRecoveryDecisionInputs:
    action4_legal: bool
    restart_path: PathLifecycleEstimate
    recovery_path: PathLifecycleEstimate
    effective_link_mbps: float | None
    positive_transfer_fixed_latency_seconds: float | None
    future_reload_cost_scale: float = 1.0


def _network_seconds(
    byte_count: int,
    *,
    effective_link_mbps: float,
    fixed_latency_seconds: float,
) -> float:
    if byte_count <= 0:
        return 0.0
    return byte_count * 8.0 / (effective_link_mbps * 1_000_000.0) + fixed_latency_seconds


def _valid(inputs: SymmetricRecoveryDecisionInputs) -> bool:
    if inputs.effective_link_mbps is None or not math.isfinite(inputs.effective_link_mbps):
        return False
    if inputs.effective_link_mbps <= 0.0:
        return False
    fixed = inputs.positive_transfer_fixed_latency_seconds
    if fixed is None or not math.isfinite(fixed) or fixed < 0.0:
        return False
    if not math.isfinite(inputs.future_reload_cost_scale) or inputs.future_reload_cost_scale < 0.0:
        return False
    for path in (inputs.restart_path, inputs.recovery_path):
        if path.dynamic_transfer_bytes < 0:
            return False
        for value in (
            path.state_overhead_seconds,
            path.recompute_seconds,
            path.successful_service_seconds,
        ):
            if not math.isfinite(value) or value < 0.0:
                return False
        for event in path.events:
            if not event.dependency_safe:
                return False
            if event.transaction_status not in {"committed", "noop_all_resident"}:
                return False
            if any(int(value) < 0 for value in event.transfer_bytes_by_object.values()):
                return False
    return True


def score_estimated_path(
    path: PathLifecycleEstimate,
    *,
    effective_link_mbps: float,
    fixed_latency_seconds: float,
    future_reload_cost_scale: float,
) -> dict[str, object]:
    event_rows: list[dict[str, object]] = []
    model_seconds = 0.0
    model_bytes = 0
    for index, event in enumerate(path.events):
        event_bytes = sum(int(value) for value in event.transfer_bytes_by_object.values())
        unscaled = _network_seconds(
            event_bytes,
            effective_link_mbps=effective_link_mbps,
            fixed_latency_seconds=fixed_latency_seconds,
        )
        scale = future_reload_cost_scale if index > 0 else 1.0
        scaled = unscaled * scale
        model_bytes += event_bytes
        model_seconds += scaled
        event_rows.append(
            {
                "event_id": event.event_id,
                "required_adapter_id": event.required_adapter_id,
                "victim_object_ids": list(event.victim_object_ids),
                "admitted_object_ids": list(event.admitted_object_ids),
                "pre_resident_object_ids": list(event.pre_resident_object_ids),
                "post_resident_object_ids": list(event.post_resident_object_ids),
                "transfer_bytes": event_bytes,
                "network_seconds_unscaled": unscaled,
                "estimate_scale": scale,
                "network_seconds_scored": scaled,
            }
        )
    dynamic_seconds = _network_seconds(
        path.dynamic_transfer_bytes,
        effective_link_mbps=effective_link_mbps,
        fixed_latency_seconds=fixed_latency_seconds,
    )
    total = (
        model_seconds
        + dynamic_seconds
        + path.state_overhead_seconds
        + path.recompute_seconds
        + path.successful_service_seconds
    )
    return {
        "action_id": path.action_id,
        "event_costs": event_rows,
        "model_transfer_bytes": model_bytes,
        "model_transfer_seconds": model_seconds,
        "dynamic_transfer_bytes": path.dynamic_transfer_bytes,
        "dynamic_transfer_seconds": dynamic_seconds,
        "state_overhead_seconds": path.state_overhead_seconds,
        "recompute_seconds": path.recompute_seconds,
        "successful_service_seconds": path.successful_service_seconds,
        "estimated_total_seconds": total,
        "terminal_resident_object_ids": (
            list(path.events[-1].post_resident_object_ids) if path.events else []
        ),
    }


def _fallback(method: str, reason: str) -> dict[str, object]:
    return {
        "method": method,
        "selected_first_action": RERUN_ACTION_ID,
        "decision": "rerun",
        "conservative_fallback": True,
        "fallback_reason": reason,
    }


def original_simple_threshold_decision(
    inputs: SymmetricRecoveryDecisionInputs,
) -> dict[str, object]:
    """Simple differential rule; lifecycle terms are deliberately ignored."""

    if not inputs.action4_legal:
        return _fallback("original_simple_threshold", "action4_illegal")
    if not _valid(inputs):
        return _fallback("original_simple_threshold", "missing_or_invalid_cost_input")
    restart = score_estimated_path(
        PathLifecycleEstimate(
            action_id=RERUN_ACTION_ID,
            events=(),
            dynamic_transfer_bytes=inputs.restart_path.dynamic_transfer_bytes,
            state_overhead_seconds=inputs.restart_path.state_overhead_seconds,
            recompute_seconds=inputs.restart_path.recompute_seconds,
            successful_service_seconds=0.0,
        ),
        effective_link_mbps=float(inputs.effective_link_mbps),
        fixed_latency_seconds=float(inputs.positive_transfer_fixed_latency_seconds),
        future_reload_cost_scale=1.0,
    )
    recovery = score_estimated_path(
        PathLifecycleEstimate(
            action_id=RECOVER_ACTION_ID,
            events=(),
            dynamic_transfer_bytes=inputs.recovery_path.dynamic_transfer_bytes,
            state_overhead_seconds=inputs.recovery_path.state_overhead_seconds,
            recompute_seconds=inputs.recovery_path.recompute_seconds,
            successful_service_seconds=0.0,
        ),
        effective_link_mbps=float(inputs.effective_link_mbps),
        fixed_latency_seconds=float(inputs.positive_transfer_fixed_latency_seconds),
        future_reload_cost_scale=1.0,
    )
    selected = (
        RECOVER_ACTION_ID
        if recovery["estimated_total_seconds"] < restart["estimated_total_seconds"]
        else RERUN_ACTION_ID
    )
    return {
        "method": "original_simple_threshold",
        "selected_first_action": selected,
        "decision": "recover" if selected == RECOVER_ACTION_ID else "rerun",
        "conservative_fallback": False,
        "restart_differential_seconds": restart["estimated_total_seconds"],
        "recovery_differential_seconds": recovery["estimated_total_seconds"],
        "lifecycle_scope": "ignored; valid cancellation only when separately audited equal",
    }


def _full_decision(
    inputs: SymmetricRecoveryDecisionInputs,
    *,
    method: str,
) -> dict[str, object]:
    if not inputs.action4_legal:
        return _fallback(method, "action4_illegal")
    if not _valid(inputs):
        return _fallback(method, "missing_invalid_or_illegal_path_input")
    restart = score_estimated_path(
        inputs.restart_path,
        effective_link_mbps=float(inputs.effective_link_mbps),
        fixed_latency_seconds=float(inputs.positive_transfer_fixed_latency_seconds),
        future_reload_cost_scale=float(inputs.future_reload_cost_scale),
    )
    recovery = score_estimated_path(
        inputs.recovery_path,
        effective_link_mbps=float(inputs.effective_link_mbps),
        fixed_latency_seconds=float(inputs.positive_transfer_fixed_latency_seconds),
        future_reload_cost_scale=float(inputs.future_reload_cost_scale),
    )
    selected = (
        RECOVER_ACTION_ID
        if recovery["estimated_total_seconds"] < restart["estimated_total_seconds"]
        else RERUN_ACTION_ID
    )
    return {
        "method": method,
        "selected_first_action": selected,
        "decision": "recover" if selected == RECOVER_ACTION_ID else "rerun",
        "conservative_fallback": False,
        "restart_path_estimate": restart,
        "recovery_path_estimate": recovery,
        "information_scope": (
            "initial residents, declared current/next dependencies, legal native previews, "
            "and ex-ante cost estimates only"
        ),
        "realized_score_used": False,
    }


def eviction_aware_recovery_decision(
    inputs: SymmetricRecoveryDecisionInputs,
) -> dict[str, object]:
    return _full_decision(inputs, method="eviction_aware_recovery")


def two_step_lookahead_decision(
    inputs: SymmetricRecoveryDecisionInputs,
) -> dict[str, object]:
    result = _full_decision(inputs, method="two_step_lookahead")
    if result.get("conservative_fallback"):
        return result
    return {
        **result,
        "candidate_paths": ["restart_current_then_declared_next", "recover_current_then_declared_next"],
        "future_truth_permission": False,
    }


def decision_inputs_to_dict(inputs: SymmetricRecoveryDecisionInputs) -> dict[str, object]:
    return asdict(inputs)
