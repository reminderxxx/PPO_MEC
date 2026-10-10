"""Tests for the preregistered dual-view verdict."""

from scripts.finalize_calibrated_workflow_prepared_state_visibility_matched import _gate_verdict


def _row(view: str, method: str, on_time: float, failure: float) -> dict:
    return {
        "checkpoint_view": view,
        "method": method,
        "split": "frozen_check",
        "mean_delta_on_time_workflow_completion_rate": on_time,
        "mean_delta_service_failure_rate": failure,
    }


def test_dual_view_gate_rejects_one_view_only_and_failure_worsening() -> None:
    rows = []
    for method in ("sa_ghmappo", "mappo", "ppo", "dt_handoff_drl"):
        rows.append(_row("selected", method, 0.1 if method == "sa_ghmappo" else 0.0,
                         0.1 if method == "sa_ghmappo" else 0.0))
        rows.append(_row("update96", method, 0.0, 0.0))
    verdict = _gate_verdict(rows)
    assert verdict["performance_candidate"] == "rejected_by_preregistered_dual_view_gate"
    assert verdict["reasons"] == {
        "sa_selected_frozen_failure_worsened": True,
        "sa_on_time_improvement_not_present_in_both_views": True,
        "any_learned_method_frozen_failure_worsened": True,
    }
