"""Finite-horizon credit, queue-neutral surrogate and train-only scenario coverage."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
import torch
from scripts.train_joint_parameterized_control import episode
from src.agents.registry import build_agent
from src.envs.core.joint_training_scenarios import coverage_counts,mobile_cohort,tight_sites
from tests.test_scheduled_control_learning import env


def policy(name='sa_ghmappo',**kwargs):
    torch.set_num_threads(1)
    return build_agent(name,control_contract='joint_parameterized_control_v3',tail_controls=True,**kwargs)


@pytest.mark.parametrize('name',['sa_ghmappo','mappo','ppo','dqn','cache_offload_drl','dag_offload_drl','dt_handoff_drl'])
def test_tail_algorithms_update_reload_credit_and_cache_closure(name,tmp_path):
    p=policy(name,**(dict(terminal_team_tail=True) if name=='dqn' else {}));e=env()
    initial={k:v.clone() for k,v in p.state_dict().items()};row,rolls=episode(e,p)
    assert p.return_gamma==1.
    assert float(rolls[0]['returns'][0])*10==pytest.approx(row['reward'],abs=1e-5)
    assert float(rolls[0]['cost_targets'][:,0].max())==pytest.approx(.2)
    if name=='dqn':
        while sum(len(r['actions']) for r in rolls)<128:rolls+=episode(env(),p)[1]
    saved=[r['returns'].clone() for r in rolls]
    loss=p.update(rolls,epochs=1)
    assert all(np.isfinite(x) for x in loss.values())
    assert any(not torch.equal(v,p.state_dict()[k]) for k,v in initial.items() if k.startswith('heads.'))
    assert all(torch.equal(r['returns'],v) for r,v in zip(rolls,saved))
    p.save_checkpoint(tmp_path/'tail.pt');q=type(p).from_checkpoint(tmp_path/'tail.pt')
    assert episode(env(),p,deterministic=True)[0]==episode(env(),q,deterministic=True)[0]
    if name=='dqn':assert p.replay and loss['policy_loss']==0


def test_surrogate_cannot_reorder_noop_or_same_target_priorities():
    p=policy();prediction=torch.randn(2,20,3);mask=torch.ones(2,20,dtype=torch.bool)
    f=torch.zeros(2,20,32);f[:,4:,0]=1;f[:,4:,24]=.1
    adjustment=p._surrogate_logit_adjustment(2,prediction,mask,{'f2':f})
    assert not adjustment[:,:4].any()
    assert torch.allclose(adjustment.reshape(2,5,4),adjustment.reshape(2,5,4)[:,:,:1].expand(-1,-1,4))
    assert not p._surrogate_logit_adjustment(0,prediction,mask,{}).any()
    f[:,:,24]=0
    assert not p._surrogate_logit_adjustment(2,prediction,mask,{'f2':f}).any()


def test_finite_horizon_preserves_late_delivery_objective():
    assert .99**119*2<1
    assert policy().return_gamma**119*2>1


def test_observed_execution_features_are_prediction_invariant_and_output_slack_corrected():
    e=env();p=policy();q=policy(prediction_enabled=False)
    decision=e.decision_states()['v0'];a,*_=p._coordinated_features(decision,0,{})
    b,*_=q._coordinated_features(decision,0,{})
    assert np.array_equal(a['own'],b['own']) and np.array_equal(a['f0'],b['f0'])
    assert not a['f0'][:,24:27].any()


def test_grouped_predictor_validation_keeps_same_interval_cohorts_together():
    from src.envs.core.joint_learned_prediction import fit_predictor
    e=env();windows=[]
    for i in range(5):
        for j in range(2):windows.append(dict(frame_offset=i,source_segment_id='train',time_start=i*100,time_end=i*100+24,frames=e.frames[:25]))
    b=fit_predictor(windows,horizon=8,epochs=1,group_by_interval=True)
    assert set(b['fit_offsets'])=={0,1,2,3} and set(b['validation_offsets'])=={4}


def test_mobile_cohort_and_tight_sites_do_not_mutate_data():
    e=env();sites=deepcopy(e.rsus);small=tight_sites(sites)
    assert all(a.coverage_radius>b.coverage_radius for a,b in zip(sites,small))
    frames,ids=mobile_cohort(e.frames,sites,count=2)
    assert len(ids)==2 and all(len(f['vehicles'])==2 for f in frames)
    assert sites==e.rsus and len(e.frames[0]['vehicles'])==2
    assert coverage_counts(frames,sites)['uncovered_vehicle_frames']==0
