"""Pure subject binding and explicit opt-in runtime failure contracts.

Boundary rows remain failures in the original exposure. A rejected attribution
must never be used to drop rows or rescore legacy rewards as candidate rewards.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

CONTRACT = 'diagnostic_subject_prediction_binding_v1'


@dataclass(frozen=True)
class SubjectBinding:
    episode_vehicle_id: str | None
    decision_vehicle_id: str | None
    prediction_vehicle_id: str | None
    execution_vehicle_id: str | None
    decision_time_index: int | None
    prediction_time_index: int | None
    execution_time_index: int | None
    prediction_origin: str
    bound: bool
    reason: str
    contract: str = CONTRACT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bind_subject_transition(
    *,
    episode_vehicle_id: str | None,
    decision_vehicle_id: str | None,
    prediction_vehicle_id: str | None,
    execution_vehicle_id: str | None,
    decision_time_index: int | None,
    prediction_time_index: int | None,
    execution_time_index: int | None,
    prediction_origin: str,
    decision_subject_present: bool,
    execution_subject_present: bool,
    boundary_already_recorded: bool = False,
) -> SubjectBinding:
    """Check causal attribution; no action selection, stepping, reward or filtering.

    The episode subject stays fixed until an externally recorded reset. Once a
    boundary is recorded the remainder stays outside the bound episode, rather
    than silently becoming a successful new subject's continuation.
    """
    if boundary_already_recorded:
        reason = 'episode_boundary_already_recorded'
    elif not episode_vehicle_id:
        reason = 'missing_episode_subject'
    elif not decision_vehicle_id or not decision_subject_present:
        reason = 'decision_subject_absent'
    elif decision_vehicle_id != episode_vehicle_id:
        reason = 'decision_subject_changed'
    elif not prediction_vehicle_id:
        reason = 'missing_prediction_subject'
    elif prediction_vehicle_id != decision_vehicle_id:
        reason = 'prediction_subject_mismatch'
    elif not prediction_origin:
        reason = 'missing_prediction_origin'
    elif decision_time_index is None or prediction_time_index is None:
        reason = 'missing_snapshot_time'
    elif prediction_time_index > decision_time_index:
        reason = 'future_prediction_snapshot'
    elif prediction_time_index < decision_time_index:
        reason = 'stale_prediction_snapshot'
    elif not execution_vehicle_id or not execution_subject_present:
        reason = 'execution_subject_absent'
    elif execution_vehicle_id != episode_vehicle_id:
        reason = 'execution_subject_changed'
    elif execution_time_index is None:
        reason = 'missing_execution_time'
    elif execution_time_index < decision_time_index:
        reason = 'execution_precedes_decision'
    else:
        reason = 'bound'
    return SubjectBinding(
        episode_vehicle_id, decision_vehicle_id, prediction_vehicle_id,
        execution_vehicle_id, decision_time_index, prediction_time_index,
        execution_time_index, prediction_origin, reason == 'bound', reason,
    )


def guard_prepare_context(
    context: dict[str, Any] | None, binding: SubjectBinding,
) -> tuple[dict[str, Any] | None, str]:
    """Return a private ledger preview only; never modify an original context."""
    if context is None:
        return None, 'no_prepare_context'
    if not binding.bound:
        return None, binding.reason
    if context.get('vehicle_id') != binding.execution_vehicle_id:
        return None, 'prepare_subject_mismatch'
    if not context.get('target_rsu_id') or not context.get('required_adapter'):
        return None, 'incomplete_prepare_context'
    return deepcopy(context), 'bound'


LEGACY_SUBJECT_CONTRACT = 'legacy_unbound_v1'
STRICT_SUBJECT_CONTRACT = 'strict_episode_subject_v1'
RUNTIME_SNAPSHOT_ORIGIN = 'runtime_per_vehicle_prediction_lookup_v1'


class SubjectBindingBoundaryError(RuntimeError):
    """An attempted exposure failed before applying control; no reward exists."""
    def __init__(self, record):
        self.record = deepcopy(record)
        super().__init__('Subject binding boundary: ' + str(record['binding']['reason']))


def write_subject_boundary_receipt(path, error, *, exposure_id, action, decision_trace):
    """Create a failure receipt; identical retries are idempotent, never overwrite."""
    import json
    from pathlib import Path
    if not exposure_id:
        raise ValueError('A frozen exposure identity is required')
    payload = dict(schema='subject_binding_failure_v1', exposure_id=exposure_id, status='EXPOSURE_FAILED',
                   action=action, reward=None, reward_available=False,
                   control_applied=False, exposure_in_denominator=True,
                   failure=deepcopy(error.record), decision_trace=deepcopy(decision_trace))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open('x') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except FileExistsError:
        if json.loads(target.read_text()) != payload:
            raise ValueError('Conflicting subject boundary receipt; refusing overwrite')
    return payload


def summarize_subject_exposures(exposures, *, expected_exposure_ids):
    """Failures stay in the denominator and prohibit publishing a filtered mean."""
    rows = list(exposures)
    if not rows or any(r.get('status') not in {'COMPLETE','EXPOSURE_FAILED'} for r in rows):
        raise ValueError('Every exposure needs an explicit complete/failure status')
    expected = list(expected_exposure_ids)
    actual = [r.get('exposure_id') for r in rows]
    if (not expected or len(expected) != len(set(expected)) or
            len(actual) != len(set(actual)) or set(actual) != set(expected)):
        raise ValueError('Every frozen exposure must appear exactly once; failures cannot be omitted')
    failures = sum(r['status'] == 'EXPOSURE_FAILED' for r in rows)
    return dict(exposures=len(rows), failed=failures,
                status='INCOMPLETE_WITH_FAILURES' if failures else 'COMPLETE',
                mean_reward=None if failures else sum(float(r['reward']) for r in rows)/len(rows))
