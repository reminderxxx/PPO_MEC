"""Contracts for the bounded state-conditioned action evidence analysis."""

from __future__ import annotations

from scripts.analyze_cscwd_state_conditioned_actions import (
    _bundle_at_rsu,
    _dominates,
    _prepared_category,
    _service_dominates,
    _source_category,
)


def _outcome(
    action: int,
    *,
    completed: bool = True,
    on_time: bool = True,
    failures: int = 0,
    elapsed: float = 10.0,
    model: int = 0,
    state: int = 0,
    input_bytes: int = 0,
    recompute: float = 0.0,
    first_service: bool = True,
    candidate: int = 2,
) -> dict:
    return {
        "candidate_action": candidate,
        "branch_action": action,
        "workflow_completed": completed,
        "on_time_workflow_completed": on_time,
        "service_failures": failures,
        "elapsed_seconds": elapsed,
        "model_transfer_bytes": model,
        "state_transfer_bytes": state,
        "input_transfer_bytes": input_bytes,
        "recompute_seconds": recompute,
        "first_service_completed": first_service,
    }


def test_pareto_dominance_requires_no_service_or_resource_regression() -> None:
    better = _outcome(3, elapsed=5.0)
    worse = _outcome(2, on_time=False, elapsed=10.0, input_bytes=10)
    assert _dominates(better, worse)
    assert _service_dominates(better, worse)

    bytes_tradeoff = _outcome(0, elapsed=4.0, model=100)
    assert not _dominates(bytes_tradeoff, better)
    assert not _dominates(better, bytes_tradeoff)


def test_selected_avoidable_and_current_service_tradeoff_are_separate() -> None:
    factual = _outcome(2, on_time=False, elapsed=12.0, input_bytes=10)
    dominant = _outcome(3, elapsed=4.0)
    category, _ = _source_category({}, [factual, dominant])
    assert category == "strictly_dominated_avoidable"

    failed_prepare = _outcome(
        4,
        failures=1,
        elapsed=4.0,
        first_service=False,
        candidate=4,
    )
    recovery = _outcome(
        2,
        on_time=False,
        elapsed=12.0,
        input_bytes=10,
        candidate=4,
    )
    category, _ = _source_category({}, [failed_prepare, recovery])
    assert category == "current_service_recovery_tradeoff"


def test_public_bundle_and_prefix_categories_do_not_invent_thresholds() -> None:
    semantic = {
        "rsus": [
            {
                "rsu_id": "rsu_0",
                "typed_resident_object_ids": ["base:a"],
            }
        ],
        "calibrated_context": {
            "required_bundle_ids": ["base:a", "adapter:x"],
            "object_catalog": {
                "base:a": {
                    "resident_bytes": 100,
                    "transfer_bytes": 90,
                    "load_seconds": 0.1,
                },
                "adapter:x": {
                    "resident_bytes": 20,
                    "transfer_bytes": 15,
                    "load_seconds": 0.2,
                },
            },
        },
    }
    bundle = _bundle_at_rsu(semantic, "rsu_0")
    assert bundle == {
        "status": "missing",
        "missing_ids": ["adapter:x"],
        "missing_resident_bytes": 20,
        "missing_transfer_bytes": 15,
        "missing_load_seconds": 0.2,
    }
    assert _prepared_category({"known": True, "exists": True, "valid": False}) == "exists_but_stale"
    assert _prepared_category({"known": True, "exists": False, "valid": False}) == "missing"
    assert _prepared_category({"known": False}) == "unknown"
