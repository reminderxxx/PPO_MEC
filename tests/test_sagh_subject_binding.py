"""Zero-interaction contracts; no environment constructor/reset/step is allowed."""
from copy import deepcopy
import pytest

from scripts.sagh_subject_binding import bind_subject_transition, guard_prepare_context
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs.action_schema import ActionAdapter
from src.data.mobility.replay_provider import ReplayProvider


def forbidden(*args, **kwargs):
    raise AssertionError('Zero-interaction contract: environment execution forbidden')


@pytest.fixture(autouse=True)
def tripwires(monkeypatch):
    for name in ('__init__', 'reset', 'step'):
        monkeypatch.setattr(VecWorkflowCoreEnv, name, forbidden)
    monkeypatch.setattr(ReplayProvider, 'step', forbidden)


def binding(**kwargs):
    args = dict(episode_vehicle_id='v1', decision_vehicle_id='v1',
                prediction_vehicle_id='v1', execution_vehicle_id='v1',
                decision_time_index=100, prediction_time_index=100,
                execution_time_index=200, prediction_origin='recorded_per_vehicle',
                decision_subject_present=True, execution_subject_present=True)
    args.update(kwargs)
    return bind_subject_transition(**args)


@pytest.mark.parametrize('changes,reason', [
    ({}, 'bound'),
    ({'episode_vehicle_id': None}, 'missing_episode_subject'),
    ({'decision_subject_present': False}, 'decision_subject_absent'),
    ({'decision_vehicle_id': 'v2'}, 'decision_subject_changed'),
    ({'prediction_vehicle_id': None}, 'missing_prediction_subject'),
    ({'prediction_vehicle_id': 'v2'}, 'prediction_subject_mismatch'),
    ({'prediction_origin': ''}, 'missing_prediction_origin'),
    ({'prediction_time_index': None}, 'missing_snapshot_time'),
    ({'decision_time_index': None}, 'missing_snapshot_time'),
    ({'prediction_time_index': 101}, 'future_prediction_snapshot'),
    ({'prediction_time_index': 99}, 'stale_prediction_snapshot'),
    ({'execution_subject_present': False}, 'execution_subject_absent'),
    ({'execution_vehicle_id': None}, 'execution_subject_absent'),
    ({'execution_vehicle_id': 'v2'}, 'execution_subject_changed'),
    ({'execution_time_index': None}, 'missing_execution_time'),
    ({'execution_time_index': 99}, 'execution_precedes_decision'),
    ({'boundary_already_recorded': True}, 'episode_boundary_already_recorded'),
])
def test_binding_failure_modes(changes, reason):
    result = binding(**changes)
    assert result.reason == reason
    assert result.bound == (reason == 'bound')
    assert result.to_dict()['prediction_vehicle_id'] == changes.get('prediction_vehicle_id', 'v1')


def test_boundary_is_not_rebased_to_new_subject():
    first = binding(execution_vehicle_id='v2')
    second = binding(decision_vehicle_id='v2', prediction_vehicle_id='v2',
                     execution_vehicle_id='v2', boundary_already_recorded=not first.bound)
    assert not first.bound and not second.bound
    assert second.episode_vehicle_id == 'v1'


def context(**changes):
    row = dict(vehicle_id='v1', target_rsu_id='rsu_b', required_adapter='adapter_a',
               prepared_at_step=1, prepared_at_time_index=100, extra={'unchanged': [0]})
    row.update(changes)
    return row


@pytest.mark.parametrize('changes,reason', [
    ({'vehicle_id': 'v2'}, 'prepare_subject_mismatch'),
    ({'target_rsu_id': None}, 'incomplete_prepare_context'),
    ({'required_adapter': None}, 'incomplete_prepare_context'),
])
def test_prepare_context_identity(changes, reason):
    assert guard_prepare_context(context(**changes), binding()) == (None, reason)


def test_guard_preserves_and_copies_bound_context():
    original = context()
    result, reason = guard_prepare_context(original, binding())
    assert reason == 'bound' and result == original and result is not original
    result['extra']['unchanged'].append(1)
    assert original == context()


def test_changed_subject_cannot_register_old_prediction_for_new_vehicle():
    raw = context(vehicle_id='v2')  # old runtime's reassigned context
    assert guard_prepare_context(raw, binding(execution_vehicle_id='v2')) == (None, 'execution_subject_changed')
    assert raw['vehicle_id'] == 'v2'


def ledger(step=1):
    env = object.__new__(VecWorkflowCoreEnv)
    env._prepare_history = []
    env._episode_steps = step
    env._handoff_prepare_window = 6
    return env


@pytest.mark.parametrize('mode', ['same_step', 'history', 'gap'])
def test_bound_prepare_matches_frozen_ledger(mode):
    before = ledger()
    after = ledger()
    raw = context()
    guarded, _ = guard_prepare_context(raw, binding())
    if mode == 'history':
        before._prepare_history = [deepcopy(raw)]
        after._prepare_history = [deepcopy(guarded)]
        before._episode_steps = after._episode_steps = 3
        raw = guarded = None
    if mode == 'gap':
        kw = dict(pre_action_associated_rsu_id=None, post_action_associated_rsu_id='rsu_b',
                  prediction_snapshot={'predicted_next_rsu_id': 'rsu_b'}, prepare_action_context=guarded)
        assert before._is_gap_transfer_entry(**kw) == after._is_gap_transfer_entry(**kw) is True
    a = before._consume_realized_prepare('v1', 'rsu_b', 'adapter_a', 1, raw)
    b = after._consume_realized_prepare('v1', 'rsu_b', 'adapter_a', 1, guarded)
    assert a == b and a['realized']
    assert a['source'] == ('history' if mode == 'history' else 'same_step')


def state(rsu):
    return {'primary_vehicle_id': 'v1', 'vehicles': [{'vehicle_id': 'v1', 'associated_rsu_id': rsu}],
            'rsus': [{'rsu_id': 'rsu_a', 'cached_adapter_ids': []}],
            'current_workflow_node': {'required_adapter': 'adapter_a'}, 'predictions': {}}


@pytest.mark.parametrize('rsu', [None, 'rsu_a'])
def test_action_zero_is_preserved_and_not_converted_to_prepare(rsu):
    decoded = ActionAdapter().decode(0, state(rsu))
    assert decoded.metadata['action_id'] == 0
    assert decoded.cache_action['strategy'] == 'reactive_cache_fill'
    assert decoded.cache_action['rsu_id'] == rsu
    assert decoded.migration_action['mode'] == 'keep'
    assert guard_prepare_context(None, binding()) == (None, 'no_prepare_context')


def test_history_adapter_mismatch_semantics_are_not_silently_fixed():
    env = ledger(step=2)
    env._prepare_history = [context(required_adapter='old_adapter')]
    result = env._consume_realized_prepare('v1', 'rsu_b', 'adapter_a', 1, None)
    assert result['realized'] and not result['required_adapter_match']


def test_trace_consumer_retains_failure_and_tail_without_reward_rescoring(tmp_path):
    import json
    import csv
    from scripts.check_sagh_subject_binding import check
    source = tmp_path/'input'
    cell = source/'variants/test/cells/sa_ghmappo/seed0'
    cell.mkdir(parents=True)
    rows = []
    for step, (pre, post) in enumerate([('v1','v1'), ('v1','v2'), ('v2','v2')]):
        s = state('rsu_a')
        s.update(primary_vehicle_id=pre, time_index=100+step*100,
                 vehicles=[dict(vehicle_id=pre, associated_rsu_id='rsu_a')],
                 predictions={'prediction_time':100+step*100})
        reward = {k:0. for k in ['service_reward','continuity_bonus','mechanism_exploration_bonus','positive_offset','delay_penalty','cache_miss_penalty','migration_cost','constraint_penalty']}
        metrics = dict(cache_event={'vehicle_id':post}, time_index=200+step*100,
                       required_adapter='adapter_a', predicted_handoff_target_rsu_id=None,
                       post_action_associated_rsu_id='rsu_a', migration_prepare_window=6,
                       handoff_event_count=0, migration_prepare_realized=False,
                       migration_prepare_realized_source=None, migration_prepare_age=None,
                       migration_prepare_required_adapter_match=False)
        rows.append(dict(semantic_state=s, metadata=dict(step=step, phase='validation', unit=0, window_id='w', workflow_id='wf'),
                         metrics_protocol=metrics, actual_action=0, native_reward=0., reward_breakdown=reward))
    path=cell/'steps.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    original=path.read_bytes()
    out=tmp_path/'output'
    check(source,out)
    result=json.loads((out/'result.json').read_text())
    phase=next(iter(result['cells'].values()))
    assert result['rows_retained']==3 and phase['first_boundaries']==1
    assert phase['bound']==1 and phase['after_boundary_rows']==1
    assert phase['action_zero_rows']==3 and phase['original_native_reward_sum']==0
    assert result['candidate_service_rewards_computed'] is False
    assert result['production_environment_integrated'] is False
    assert path.read_bytes()==original
    assert len(list(csv.DictReader((out/'all_transition_bindings.csv').open())))==3
    with pytest.raises(FileExistsError):
        check(source,out)
