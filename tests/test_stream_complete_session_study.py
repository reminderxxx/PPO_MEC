"""Full terminal MC targets, idle/late exposure, likelihoods and audit rejection."""
import json
import numpy as np
import pytest
import torch

import scripts.run_stream_complete_session_study as runner
from scripts.audit_stream_complete_session_study import verify_complete_training
from tests.test_stream_control_learning import env


def make_unit(monkeypatch,tmp_path,variant='intent'):
    torch.set_num_threads(1)
    e=env();monkeypatch.setattr(runner,'make_env',lambda *args:e)
    agent=runner.Main(**runner.VARIANTS[variant])
    with (tmp_path/'training_trace.jsonl').open('w') as trace,(tmp_path/'training_targets.jsonl').open('w') as targets:
        row,rolls=runner.complete_train_unit({},None,None,agent,480,trace,targets,
            dict(seed=7,variant=variant,case_id='fixture'))
    (tmp_path/'training.jsonl').write_text(json.dumps(dict(case_id='fixture',**row))+'\n')
    case=dict(arrivals=e.arrivals,catalog=e.catalog_objects,
        options=dict(model_cache_config=e.model_cache_config),arrival_mode='stream')
    plan=dict(training_cases=['fixture'],maximum_frames_per_episode=480)
    return e,agent,row,rolls,case,plan


@pytest.mark.parametrize('variant',list(runner.VARIANTS))
def test_complete_mc_covers_late_release_and_preserves_behavior(monkeypatch,tmp_path,variant):
    e,agent,row,rolls,case,plan=make_unit(monkeypatch,tmp_path,variant)
    frames=[json.loads(s) for s in (tmp_path/'training_trace.jsonl').read_text().splitlines()]
    assert 45<row['actual_frames']<=70
    assert len(frames)==row['actual_frames'] and row['physical_terminal'] and row['bootstrap_team_tail']==0
    assert all('v/late' not in f['released_request_ids_before'] for f in frames[:45])
    assert 'v/late' in frames[45]['released_request_ids_before']
    for r in rolls:
        expected=sum(f['reward'] for f in frames[r['frame_index']:])/10
        assert torch.allclose(r['returns'],torch.full_like(r['returns'],expected),atol=1e-6)
        assert torch.allclose(r['credit_targets'],r['returns']/len(e.workflows),atol=1e-6)
        assert torch.allclose(agent._evaluate(r['data'],actions=r['actions'])[1],r['head_logs'],atol=1e-6)
    verified=verify_complete_training(tmp_path,{'fixture':case},{'fixture':70},plan,(7,variant))
    assert verified['episodes']==1 and verified['all_late_releases_observed']
    assert verified['frames']==row['actual_frames'] and verified['cut_bootstrap_count']==0
    assert np.isfinite(agent.update(rolls,epochs=1)['policy_loss'])


def test_full_unit_includes_idle_receipts_and_never_pads_terminal(monkeypatch,tmp_path):
    from tests.test_multi_request_stream_env import make,wf
    from src.envs.core.multi_request_stream_env import MultiRequestStreamEnv,WorkflowArrival
    from src.envs.specs import RSUState
    old=make([WorkflowArrival(8,'v',wf('late'))],horizon=70)
    e=MultiRequestStreamEnv(frames=old.frames,arrivals=old.arrivals,catalog=old.catalog_objects,
        rsu_states=[RSUState(s,0,0,10) for s in 'abcd'],costs=old.costs,workload_bytes_per_unit=1,
        shared_state_enabled=False,coordination_intent_enabled=True,link_reservation_enabled=True,
        model_cache_config={'capacity_bytes':8,'transfer_bytes_per_frame':4})
    monkeypatch.setattr(runner,'make_env',lambda *args:e)
    with (tmp_path/'trace').open('w') as trace,(tmp_path/'targets').open('w') as targets:
        row,rolls=runner.complete_train_unit({},None,None,runner.Main(),480,trace,targets,{})
    frames=[json.loads(s) for s in (tmp_path/'trace').read_text().splitlines()]
    assert all(f['events']==[] for f in frames[:8])
    assert rolls[0]['frame_index']==8
    assert row['actual_frames']<70 and len(frames)==e.frame_index
    assert e._control_episode_done()


def test_cap_must_fit_complete_session_before_any_step(monkeypatch,tmp_path):
    e=env();monkeypatch.setattr(runner,'make_env',lambda *args:e)
    with (tmp_path/'trace').open('w') as trace,(tmp_path/'targets').open('w') as targets:
        with pytest.raises(ValueError,match='fit the declared frame cap'):
            runner.complete_train_unit({},None,None,runner.Main(),20,trace,targets,{})
    assert e.frame_index==0 and not (tmp_path/'trace').read_text()


def test_audit_rejects_changed_mc_tail(monkeypatch,tmp_path):
    e,agent,row,rolls,case,plan=make_unit(monkeypatch,tmp_path)
    f=tmp_path/'training_targets.jsonl';rows=[json.loads(s) for s in f.read_text().splitlines()]
    rows[0]['returns'][0]+=1
    f.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(AssertionError):verify_complete_training(tmp_path,{'fixture':case},{'fixture':70},plan,(7,'intent'))


def test_audit_rejects_falsely_terminal_prefix_before_late_release(monkeypatch,tmp_path):
    e,agent,row,rolls,case,plan=make_unit(monkeypatch,tmp_path)
    f=tmp_path/'training_trace.jsonl';f.write_text('\n'.join(f.read_text().splitlines()[:20])+'\n')
    row['actual_frames']=20;row['summary']['steps']=20
    (tmp_path/'training.jsonl').write_text(json.dumps(dict(case_id='fixture',**row))+'\n')
    with pytest.raises(AssertionError):verify_complete_training(tmp_path,{'fixture':case},{'fixture':70},plan,(7,'intent'))
