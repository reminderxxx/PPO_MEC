"""Pure decision tests for the training-source exposure gate."""

from scripts.qualify_cscwd_training_source_positive_prepare import _gate_decision


def test_gate_fails_without_actual_positive_prepare_intervention() -> None:
    result = _gate_decision(
        base_positive_prepare_count=0,
        candidate_prepare_to_abstain_count=0,
        consumer_mismatch_count=0,
        minimum_base_positive_prepare_count=1,
        minimum_candidate_prepare_to_abstain_count=1,
    )
    assert result["gate_passed"] is False
    assert result["training_authorized"] is False
    assert result["failure_reasons"] == [
        "nonzero_base_positive_prepare_requirement_failed",
        "nonzero_candidate_intervention_requirement_failed",
    ]


def test_gate_pass_still_requires_separate_training_authorization() -> None:
    result = _gate_decision(
        base_positive_prepare_count=1,
        candidate_prepare_to_abstain_count=1,
        consumer_mismatch_count=0,
        minimum_base_positive_prepare_count=1,
        minimum_candidate_prepare_to_abstain_count=1,
    )
    assert result["gate_passed"] is True
    assert result["training_authorized"] is False
    assert result["failure_reasons"] == []
    assert result["training_authorization_reason"] == (
        "separate_central_authorization_required_after_gate_pass"
    )
