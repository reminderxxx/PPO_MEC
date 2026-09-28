import pytest
from scripts.audit_research_problem_evidence import aggregate, episode_record


def row(done=0, cost=10, unit='u'):
    return dict(condition='x', seed=1, unit=unit, window='w', completed=done,
                requests=2, successful_requests=done, total_transfer_mb=cost)


def test_failed_costs_are_retained():
    result = aggregate([row(1, 10, 'a'), row(0, 30, 'b')])['x']
    assert result['completion_rate'] == .5
    assert result['transfer_per_completed_workflow_including_failure_cost'] == 40


def test_zero_completion_is_null_not_zero_cost():
    assert aggregate([row()])['x']['transfer_per_completed_workflow_including_failure_cost'] is None


def test_duplicate_units_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        aggregate([row(), row()])


def test_missing_request_events_rejected():
    with pytest.raises(ValueError, match='count mismatch'):
        episode_record(dict(run_info={}, cache_event_trace=[],
                            formal_request_execution_audit={'external_request_denominator': 2}))
