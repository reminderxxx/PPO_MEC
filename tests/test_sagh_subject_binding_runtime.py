"""Production step and wrapper integration on synthetic frames; no real rollout."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import types
import pytest

from src.data.mobility.replay_provider import ReplayProvider
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs import ActionAdapter, ControlAction, RSUState
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.runtime.subject_binding import (
    STRICT_SUBJECT_CONTRACT, SubjectBindingBoundaryError,
    summarize_subject_exposures, write_subject_boundary_receipt,
)


class SyntheticProvider(ReplayProvider):
    """Only test-defined frames; ReplayProvider.step itself is forbidden."""
    def __init__(self, frames):
        super().__init__(deepcopy(frames))
        self.synthetic_steps = 0
    def step(self):
        self.synthetic_steps += 1
        self._frame_index = min(self._frame_index+1, len(self._trajectory_frames)-1)
        self._active_vehicles = self._frame_to_vehicle_states(self._trajectory_frames[self._frame_index])
        return self.get_active_vehicles()


@pytest.fixture(autouse=True)
def no_real_provider_step(monkeypatch):
    def forbidden(*a, **kw):raise AssertionError('Real provider interaction forbidden')
    monkeypatch.setattr(ReplayProvider, 'step', forbidden)


def frame(t, subject='v1', x=0.):
    return dict(time_index=t, vehicles=[] if subject is None else [dict(
        vehicle_id=subject, position_x=x, position_y=0., speed=0., base_model_id='veh_base_v1')])


def env(cls=VecWorkflowCoreEnv, strict=False, frames=None):
    args=dict(mobility_provider=SyntheticProvider(frames or [frame(0),frame(1),frame(2),frame(3)]),
              rsu_states=[RSUState(rsu_id='a',position_x=0.,position_y=0.,coverage_radius=20.),
                          RSUState(rsu_id='b',position_x=200.,position_y=0.,coverage_radius=20.)],
              reward_positive_offset=0., max_steps=20)
    if strict:args['subject_binding_contract']=STRICT_SUBJECT_CONTRACT
    return cls(**args)


def strip_witness(value):
    if isinstance(value,dict):return {k:strip_witness(v) for k,v in value.items() if k!='subject_binding_snapshot'}
    if isinstance(value,list):return [strip_witness(v) for v in value]
    if isinstance(value,tuple):return tuple(strip_witness(v) for v in value)
    return value


@pytest.fixture(scope='module')
def baseline_class():
    root=Path(__file__).resolve().parents[1]
    code=subprocess.check_output(['git','show','f17707ff0adb24ebc271995c3b3392f6261e1a6a:src/envs/core/vec_workflow_core_env.py'],cwd=root).decode('utf-8-sig')
    module=types.ModuleType('frozen_subject_baseline')
    module.__file__=str(root/'src/envs/core/vec_workflow_core_env.py')
    exec(compile(code,'frozen_subject_baseline.py','exec'),module.__dict__)
    return module.VecWorkflowCoreEnv


@pytest.mark.parametrize('action',range(5))
@pytest.mark.parametrize('frames',[
    [frame(0),frame(1),frame(2),frame(3)],
    [frame(0),frame(1,x=1000.),frame(2,x=200.),frame(3,x=200.)],
    [frame(0),frame(1,'v2'),frame(2,'v2'),frame(3,'v2')],
])
def test_legacy_default_matches_frozen_full_step(baseline_class,action,frames):
    old=env(baseline_class,frames=frames);new=env(frames=frames)
    assert old.reset()==new.reset()
    for _ in range(3):
        a=ActionAdapter().decode(action,old._last_state);b=ActionAdapter().decode(action,new._last_state)
        assert old.step(a)==new.step(b)


@pytest.mark.parametrize('action',range(5))
@pytest.mark.parametrize('frames',[
    [frame(0),frame(1),frame(2),frame(3)],
    [frame(0),frame(1,x=1000.),frame(2,x=200.),frame(3,x=200.)],
])
def test_strict_same_subject_matches_legacy_except_identity_witness(action,frames):
    old=env(frames=frames);strict=env(strict=True,frames=frames)
    assert old.reset()==strip_witness(strict.reset())
    for _ in range(3):
        a=ActionAdapter().decode(action,old._last_state);b=ActionAdapter().decode(action,strict._last_state)
        assert old.step(a)==strip_witness(strict.step(b))


@pytest.mark.parametrize('action',range(5))
@pytest.mark.parametrize('next_subject',['v2',None])
def test_boundary_prevents_every_control_before_cache_offload_reward(monkeypatch,action,next_subject):
    e=env(strict=True,frames=[frame(0),frame(1,next_subject)])
    state,_=e.reset();before=(deepcopy(e.rsu_states),deepcopy(e.workflow_state),deepcopy(e._prepare_history))
    def forbidden(*a,**kw):raise AssertionError('Control reached execution after subject loss')
    for name in ['_apply_cache_action','_resolve_target_rsu_id','_advance_current_node_service','_consume_realized_prepare']:
        monkeypatch.setattr(e,name,forbidden)
    control=ActionAdapter().decode(action,state)
    with pytest.raises(SubjectBindingBoundaryError) as caught:e.step(control)
    assert caught.value.record['binding']['reason']=='execution_subject_absent'
    assert caught.value.record['stage']=='after_mobility_before_control'
    assert not caught.value.record['control_applied'] and not caught.value.record['reward_available']
    assert (e.rsu_states,e.workflow_state,e._prepare_history)==before
    assert e._primary_vehicle_id=='v1'
    assert e._mobility_provider.synthetic_steps==1
    with pytest.raises(SubjectBindingBoundaryError):e.step(control)
    assert e._mobility_provider.synthetic_steps==1


@pytest.mark.parametrize('fault',['missing','wrong_subject','stale','future','wrong_origin'])
def test_bad_native_witness_fails_before_mobility(fault):
    e=env(strict=True);e.reset();s=e._last_state['predictions']['subject_binding_snapshot']
    if fault=='missing':e._last_state['predictions'].pop('subject_binding_snapshot')
    if fault=='wrong_subject':s['prediction_vehicle_id']='v2'
    if fault=='stale':s['prediction_time_index']=-1
    if fault=='future':s['prediction_time_index']=1
    if fault=='wrong_origin':s['origin']='reconstructed_legacy'
    with pytest.raises(SubjectBindingBoundaryError):e.step(ControlAction())
    assert e._mobility_provider.synthetic_steps==0


@pytest.mark.parametrize('action',[0,4])
def test_wrapper_records_failure_without_fake_reward_and_refuses_implicit_retry(tmp_path,action):
    core=env(strict=True,frames=[frame(0),frame(1,'v2')]);path=tmp_path/'failure.json'
    wrapped=GymVecEnv(core,subject_boundary_receipt_path=str(path),subject_boundary_exposure_id='exposure_0');wrapped.reset()
    with pytest.raises(SubjectBindingBoundaryError):wrapped.step(action)
    data=json.loads(path.read_text());assert data['status']=='EXPOSURE_FAILED'
    assert data['action']==action and data['reward'] is None and data['exposure_in_denominator']
    assert wrapped._episode_step_index==0 and core._mobility_provider.synthetic_steps==1
    with pytest.raises(SubjectBindingBoundaryError):wrapped.step(action)
    assert json.loads(path.read_text())==data and core._mobility_provider.synthetic_steps==1
    with pytest.raises(FileExistsError):wrapped.reset()


def test_strict_wrapper_requires_failure_sink():
    with pytest.raises(ValueError):GymVecEnv(env(strict=True))


def test_failure_summary_keeps_denominator_and_cannot_publish_filtered_mean():
    rows=[dict(exposure_id='a',status='COMPLETE',reward=10.),dict(exposure_id='b',status='EXPOSURE_FAILED',reward=None)]
    assert summarize_subject_exposures(rows,expected_exposure_ids=['a','b'])==dict(exposures=2,failed=1,status='INCOMPLETE_WITH_FAILURES',mean_reward=None)
    assert summarize_subject_exposures([dict(exposure_id='a',status='COMPLETE',reward=2.),dict(exposure_id='b',status='COMPLETE',reward=4.)],expected_exposure_ids=['a','b'])['mean_reward']==3.
    with pytest.raises(ValueError):summarize_subject_exposures([{}],expected_exposure_ids=['a'])


@pytest.mark.parametrize('actual',[
    [dict(exposure_id='a',status='COMPLETE',reward=5.)],
    [dict(exposure_id='a',status='COMPLETE',reward=5.),dict(exposure_id='a',status='COMPLETE',reward=5.)],
    [dict(exposure_id='a',status='COMPLETE',reward=5.),dict(exposure_id='c',status='COMPLETE',reward=5.)],
])
def test_failure_consumer_rejects_omitted_duplicate_or_wrong_exposure(actual):
    with pytest.raises(ValueError):summarize_subject_exposures(actual,expected_exposure_ids=['a','b'])


def test_conflicting_failure_receipt_is_never_overwritten(tmp_path):
    error=SubjectBindingBoundaryError({'binding':{'reason':'x'}});p=tmp_path/'receipt.json'
    a=write_subject_boundary_receipt(p,error,exposure_id='e',action=0,decision_trace={})
    with pytest.raises(ValueError):write_subject_boundary_receipt(p,error,exposure_id='e',action=4,decision_trace={})
    assert json.loads(p.read_text())==a


def test_runtime_reset_clears_core_latch_for_explicit_new_exposure():
    e=env(strict=True,frames=[frame(0),frame(1,'v2')]);e.reset()
    with pytest.raises(SubjectBindingBoundaryError):e.step(ControlAction())
    e.reset();assert e._subject_binding_failure is None and e._subject_binding_episode_vehicle_id=='v1'


def test_unknown_mode_is_rejected_before_provider_use():
    with pytest.raises(ValueError):VecWorkflowCoreEnv(subject_binding_contract='unknown')


def test_strict_contract_cannot_be_mixed_into_formal_protocol():
    with pytest.raises(ValueError, match='not a formal'):
        VecWorkflowCoreEnv(subject_binding_contract=STRICT_SUBJECT_CONTRACT, formal_request_exposure_trace={})


def test_valid_prepare_and_cache_bundle_cannot_reach_replacement_subject(monkeypatch):
    e=env(strict=True,frames=[frame(0),frame(1,'v2')]);e.reset()
    control=ControlAction(cache_action={'operation':'cache','rsu_id':'b','adapter_id':'adapter_perception','strategy':'handoff_prepare_prefetch'},
                          offload_action={'mode':'rsu'},migration_action={'mode':'prepare','expected_target_rsu_id':'b'},metadata={'action_id':4,'primary_vehicle_id':'v1'})
    e._prepare_history=[{'vehicle_id':'v1','target_rsu_id':'b','required_adapter':'adapter_perception','prepared_at_step':0}]
    before=deepcopy(e._prepare_history)
    def forbidden(*a,**kw):raise AssertionError('Old prepare/cache control reached replacement subject')
    monkeypatch.setattr(e,'_apply_cache_action',forbidden)
    with pytest.raises(SubjectBindingBoundaryError):e.step(control)
    assert e._prepare_history==before


@pytest.mark.parametrize('control_subject',[None,'v2'])
def test_control_itself_must_belong_to_bound_decision_subject(control_subject):
    e=env(strict=True);e.reset()
    control=ControlAction(metadata={'primary_vehicle_id':control_subject})
    with pytest.raises(SubjectBindingBoundaryError) as caught:e.step(control)
    assert caught.value.record['binding']['reason']=='control_subject_mismatch'
    assert e._mobility_provider.synthetic_steps==0
