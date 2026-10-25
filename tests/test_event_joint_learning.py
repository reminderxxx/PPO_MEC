from copy import deepcopy
import pytest
import torch
from tests.test_joint_workflow_control import joint_fixture
from src.agents.ppo_agent import EventJointPPOAgent,ReadyWorkflowPPOAgent
from src.encoders.event_joint_encoder import event_features,batch_event_features
from scripts.run_event_joint_learning import event_available


def features():
    env,s=joint_fixture();s,_,_,_,_=env.step_joint('r0:a',None)
    return env,s,event_features(s)


def test_events_are_causal_and_suppressed_while_loading():
    env,s,f=features();assert event_available(s)
    assert f['mask'].tolist()==[False,False,True,False,False,True]
    s,_,_,_,_=env.step_joint('r0:b','adapter_fusion')
    assert not event_available(s)
    assert not f['mask'][0] and f['ids'][2]=='adapter_fusion'


def test_matching_graph_set_initial_policy_and_permutation_invariance():
    _,s,f=features();a=EventJointPPOAgent(7,True);b=EventJointPPOAgent(7,False)
    assert all(torch.equal(v,b.network.state_dict()[k]) for k,v in a.network.state_dict().items())
    batch=batch_event_features([f]);x,v=a.network(batch);y,w=b.network(batch)
    assert torch.equal(x,y) and torch.equal(v,w)
    t=deepcopy(s);t['workflow']['nodes'].reverse();ff=event_features(t)
    z,u=a.network(batch_event_features([ff]))
    assert torch.allclose(x,z,atol=1e-6) and torch.allclose(v,u,atol=1e-6)
    a.network.message_gate.data.fill_(1.);xx,_=a.network(batch)
    assert not torch.equal(xx,x)


def test_ppo_update_checkpoint_replay_and_contract_rejection(tmp_path):
    _,s,f=features();a=EventJointPPOAgent(13,True);rows=[]
    for advantage in [-1.,1.,2.,-2.]:
        act,lp,v=a.act(f);rows.append(dict(features=f,action=act,log_prob=lp,value=v,advantage=advantage,**{'return':advantage}))
    old={k:v.clone() for k,v in a.network.state_dict().items()};result=a.learn(rows)
    assert result['rows']==4 and any(not torch.equal(v,old[k]) for k,v in a.network.state_dict().items())
    path=tmp_path/'event.pt';a.save(path,{'test':True});b,p=EventJointPPOAgent.load(path)
    assert a.act(f,True)==b.act(f,True) and p['provenance']['test']
    with pytest.raises(FileExistsError):a.save(path,{})
    with pytest.raises(ValueError,match='contract'):ReadyWorkflowPPOAgent.load(path)
    p['features']='wrong';torch.save(p,tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='contract'):EventJointPPOAgent.load(tmp_path/'bad.pt')


def test_critic_gradients_do_not_change_actor():
    _,_,f=features();a=EventJointPPOAgent(7,True)
    logits,value=a.network(batch_event_features([f]));value.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in a.network.critic.parameters())
    assert all(p.grad is None for p in a.network.actor_parameters())


def test_eviction_observation_is_readonly_and_matches_native_transaction():
    from src.envs.wrappers.joint_workflow_control import observe_joint_effects
    env,_=joint_fixture();env._cache_capacity_profile['capacity_mb']=320.;s,_=env.reset()
    s,_,_,_,_=env.step_joint('r0:a',None)
    previous=deepcopy(env._eviction_policy.export_state());old=deepcopy(s)
    observed=observe_joint_effects(env,s)
    assert s==old and previous==env._eviction_policy.export_state()
    predicted=observed['joint_cache_observation']['candidate_effects']['adapter_fusion']
    assert predicted['victims']==['adapter:perception']
    f=event_features(observed,True)
    assert not f['mask'][2] and f['aux'][2,-1]==1 and f['mask'][5]
    _,_,_,_,info=env.step_joint('r0:b','adapter_fusion')
    actual=[x['object_id'] for x in info['persistent_stream']['cache_event']['evicted_typed_objects']]
    assert actual==predicted['victims']
    assert not info['persistent_stream']['cache_event']['service_success']


def test_eviction_checkpoint_v2_and_v1_are_both_explicitly_loadable(tmp_path):
    from src.envs.wrappers.joint_workflow_control import observe_joint_effects
    env,s,_=features();f=event_features(observe_joint_effects(env,s),True)
    with pytest.raises(ValueError,match='observable'):event_features(s,True)
    for aware in [True,False]:
        a=EventJointPPOAgent(7,True,aware);p=tmp_path/f'aware_{aware}.pt';a.save(p,{});b,payload=EventJointPPOAgent.load(p)
        assert b.eviction_aware==aware
        ff=f if aware else event_features(s)
        assert a.act(ff,True)==b.act(ff,True)


def test_physical_contract_cannot_silently_change(tmp_path):
    a=EventJointPPOAgent(7,True,True);p=tmp_path/'v2.pt';a.save(p,{})
    data=torch.load(p);data['physical_contract']='legacy_admission_recency_v1';torch.save(data,p)
    with pytest.raises(ValueError,match='physical contract'):EventJointPPOAgent.load(p)
