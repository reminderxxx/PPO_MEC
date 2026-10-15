"""Stream behavior identity, queue/cost features and physical idle/tail credit."""
from copy import deepcopy
import json
from dataclasses import replace
import pytest
import torch
from src.agents.registry import build_agent
from src.envs.core.multi_request_stream_env import MultiRequestStreamEnv,WorkflowArrival
from src.envs.core.joint_model_cache import ModelCacheObject
from src.envs.specs import RSUState
from tests.test_multi_request_stream_env import make,wf
from scripts.train_joint_stream_control import stream_episode


def env():
    old=make([WorkflowArrival(0,'v',wf('one',3)),WorkflowArrival(0,'v',wf('queued',2)),WorkflowArrival(45,'v',wf('late'))],horizon=70)
    return MultiRequestStreamEnv(frames=old.frames,arrivals=old.arrivals,catalog=old.catalog_objects,
        rsu_states=[RSUState(r,0,0,10) for r in 'abcd'],costs=old.costs,workload_bytes_per_unit=1,
        shared_state_enabled=False,coordination_intent_enabled=True,link_reservation_enabled=True,
        model_cache_config={'capacity_bytes':8,'transfer_bytes_per_frame':4})


def policy(name='sa_ghmappo',**kw):
    torch.set_num_threads(1)
    return build_agent(name,control_contract='joint_stream_control_v1',return_gamma=1.,**kw)


@pytest.mark.parametrize('name',['sa_ghmappo','mappo','ppo','dqn','cache_offload_drl','dag_offload_drl','dt_handoff_drl'])
def test_all_stream_algorithms_probabilities_update_checkpoint_replay(name,tmp_path):
    p=policy(name);e=env();_,r=p.act_controls(e.decision_states(),0)
    assert torch.allclose(p._evaluate(r['data'],actions=r['actions'])[1],r['head_logs'],atol=1e-6)
    assert r['data']['graph'].shape[1:]==(64,4)
    initial={k:v.clone() for k,v in p.state_dict().items()};row,rs=stream_episode(env(),p)
    assert rs[-1]['episode_end'] and all(torch.allclose(r['team_credit_targets'],r['returns']/r['initial_request_count']) for r in rs)
    if name=='dqn':
        while sum(len(r['actions']) for r in rs)<128:rs+=stream_episode(env(),p)[1]
    loss=p.update(rs,epochs=1)
    assert all(torch.isfinite(torch.tensor(v)) for v in loss.values())
    assert any(not torch.equal(v,p.state_dict()[k]) for k,v in initial.items() if k.startswith('heads.'))
    path=tmp_path/'p.pt';p.save_checkpoint(path);q=type(p).from_checkpoint(path)
    a,b=env(),env();x,_=stream_episode(a,p,deterministic=True);y,_=stream_episode(b,q,deterministic=True)
    assert x==y and json.dumps(a.events,sort_keys=True)==json.dumps(b.events,sort_keys=True)


def test_queue_candidate_identity_and_unreleased_graph_hidden():
    e=env();p=policy();d=e.decision_states()['v'];data,candidates,*_=p._coordinated_features(d,0,{})
    queued=[k for k,(site,nid,kind) in enumerate(candidates[1]) if nid and nid.startswith('queued')]
    legal_queued=[k for k in queued if data['m1'][k]]
    assert queued and legal_queued and (data['f1'][legal_queued,27]>0).all()
    assert not any(nid and nid.startswith('late') for _,nid,_ in candidates[1])
    before=data['f1'].copy();d2=deepcopy(d)
    d2['semantic_state']['control_cost_parameters']['model_catalog']['adapter:x']['resident_size_bytes']=4
    after=p._coordinated_features(d2,0,{})[0]['f1'];assert not (before==after).all()


def test_request_change_discards_old_target_memory_even_with_same_node_name():
    e=env();p=policy();ds=e.decision_states();p.request_memory['v']='old';p.target_memory['v']=('one_n0','d',-999)
    p.act_controls(ds,0,deterministic=True)
    assert p.request_memory['v']=='v/one' and p.target_memory['v'][2]==0


def test_empty_frame_and_future_team_returns_are_accounted():
    e=env();p=policy();row,rs=stream_episode(e,p,deterministic=True)
    assert len(e.frame_receipts)==row['steps']
    for r in rs:
        expected=sum(sum(q['reward_components'].values()) for q in e.frame_receipts[r['frame_index']:])/10
        assert float(r['returns'][0])==pytest.approx(expected,abs=1e-6)
    assert all('late' not in str(r['data']) for r in rs)  # tensor features contain no identity strings
    assert p.act_controls({},0)==({},None)


def test_legacy_checkpoint_cannot_be_loaded_as_stream(tmp_path):
    legacy=build_agent('mappo',control_contract='joint_parameterized_control_v3',delivery_controls=True)
    p=tmp_path/'legacy.pt';legacy.save_checkpoint(p)
    with pytest.raises(ValueError,match='incompatible'):type(policy('mappo')).from_checkpoint(p)


def test_dqn_gap_rewards_and_terminal_tails_conserve_observed_return():
    p=policy('dqn');_,rs=stream_episode(env(),p,deterministic=True)
    p.update(rs,epochs=0)
    # All transitions of the physical vehicle telescope through request/idle boundaries.
    assert sum(x[3] for x in p.replay)==pytest.approx(float(rs[0]['returns'][0]),abs=1e-6)
    assert sum(x[4] for x in p.replay)==1
    assert p.replay[0][0]['f1'].shape[1]==513


def test_two_long_released_graphs_have_distinct_visible_candidates_beyond_old_dqn_bound():
    base=env();e=MultiRequestStreamEnv(frames=base.frames,arrivals=[WorkflowArrival(0,'v',wf('long0',30)),WorkflowArrival(0,'v',wf('long1',30))],
        catalog=base.catalog_objects,rsu_states=base.rsus,costs=base.costs,workload_bytes_per_unit=1,
        coordination_intent_enabled=True,link_reservation_enabled=True,model_cache_config={'capacity_bytes':8,'transfer_bytes_per_frame':4})
    p=policy('dqn');_,r=p.act_controls(e.decision_states(),0)
    assert r['data']['m1'].shape[1]==481 and r['data']['graph_mask'].sum()==60
    padded=p._replay_data(r['data'],0)
    assert padded['m1'].shape[1]==513 and torch.equal(padded['m1'][0,:481],r['data']['m1'][0])
    assert torch.allclose(p._evaluate(r['data'],actions=r['actions'])[1],r['head_logs'],atol=1e-6)


def test_current_migration_value_is_not_assigned_to_queued_candidate():
    p=policy();d=env().decision_states()['v'];p._reserved_opportunities=lambda s,c: {'b':{'net_reward_proxy':1.,'prepare_feasible':False}}
    data,cs,*_=p._coordinated_features(d,0,{})
    queued=[k for k,(site,nid,kind) in enumerate(cs[1]) if site=='b' and nid and nid.startswith('queued')]
    assert queued and not data['f1'][queued,24].any()


def test_unknown_remote_capacity_uses_public_byte_scale_without_enabling_admission():
    p=policy();d=env().decision_states()['v'];data,cs,*_=p._coordinated_features(d,0,{})
    unknown=[r['rsu_id'] for r in d['semantic_state']['rsus'] if not r.get('capacity_bytes')]
    assert unknown
    for site in unknown:
        k=cs[0].index(site);assert (data['f0'][k,20:22]<=1).all() and (data['f0'][k,20:22]>0).all()
        assert not any(data['m1'][i] for i,c in enumerate(cs[1]) if c[0]==site)
