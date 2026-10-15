"""Team delivery contracts: probability identity, actual cold-admission protection and credit."""
from copy import deepcopy
import json
import pytest
import torch
from src.agents.registry import build_agent
from src.envs.core.joint_delivery_planning import reactive_link_calendar,delivery_migration_opportunities,book_delivery_control,node_delivery_utility
from src.envs.core.joint_link_reservation import reserved_migration_opportunities
from src.envs.core.multi_request_target_control_env import JointRequestControl
from scripts.train_joint_parameterized_control import episode
from tests.test_scheduled_control_learning import env
from tests.test_coordinated_control_learning import forecast_state


def policy(name='sa_ghmappo',**kw):
    torch.set_num_threads(1)
    return build_agent(name,control_contract='joint_parameterized_control_v3',delivery_controls=True,**kw)


@pytest.mark.parametrize('name',['sa_ghmappo','mappo','ppo','dqn','cache_offload_drl','dag_offload_drl','dt_handoff_drl'])
def test_two_pass_probabilities_learn_reload_and_replay(name,tmp_path):
    e=env();p=policy(name);controls,roll=p.act_controls(e.decision_states(),0)
    result=p._evaluate(roll['data'],actions=roll['actions'],return_details=True)
    assert torch.allclose(result[1],roll['head_logs'],atol=1e-6)
    p.reset_controller();_,rd=p.act_controls(env().decision_states(),0,deterministic=True)
    assert torch.allclose(p._evaluate(rd['data'],actions=rd['actions'],deterministic=True)[1],rd['head_logs'],atol=1e-6)
    initial={k:v.clone() for k,v in p.state_dict().items()};row,rs=episode(env(),p)
    if name=='dqn':
        while sum(len(r['actions']) for r in rs)<128:rs+=episode(env(),p)[1]
    loss=p.update(rs,epochs=1)
    assert all(torch.isfinite(torch.tensor(v)) for v in loss.values())
    assert any(not torch.equal(v,p.state_dict()[k]) for k,v in initial.items() if k.startswith('heads.'))
    path=tmp_path/'policy.pt';p.save_checkpoint(path);q=type(p).from_checkpoint(path)
    assert p.return_gamma==q.return_gamma==1 and p.credit_mode==q.credit_mode=='team'
    a,b=env(),env();x,_=episode(a,p,deterministic=True);y,_=episode(b,q,deterministic=True)
    assert x==y and json.loads(json.dumps(a.events))==json.loads(json.dumps(b.events))


def test_full_initial_policy_matches_cache_actor_and_keeps_dag_learnable():
    full=policy(seed=7);cache=policy('cache_offload_drl',seed=7);d=env().decision_states()
    a,ra=full.act_controls(d,0,deterministic=True);b,rb=cache.act_controls(d,0,deterministic=True)
    assert a==b and torch.equal(ra['head_logs'],rb['head_logs'])
    assert full.graph_residual_gate.requires_grad
    assert full.surrogate_enabled and not cache.surrogate_enabled


def test_cold_adapter_transaction_is_protected_before_preparation():
    s=forecast_state();s['control_cost_parameters'].update(base_size_bytes=2048,adapter_size_bytes=512,model_bytes_per_frame=1024)
    s['execution_state_payload_bytes']=6425
    other=deepcopy(s);other['primary_vehicle_id']='v1';other['execution_location']='b';other['execution_state_version']=0
    for row in other['rsus']:
        if row['rsu_id']=='b':row['cached_adapter_ids']=[]
    decisions={'v0':{'semantic_state':s},'v1':{'semantic_state':other}}
    calendar=reactive_link_calendar(decisions,{'v0':'a','v1':'b'},8)
    assert calendar['b'][0]==('v1','model')
    assert 'b' not in delivery_migration_opportunities(s,calendar,decisions=decisions,targets={'v0':'a','v1':'b'})
    e=env();states=e.decision_states();p=policy();controls,_=p.act_controls(states,0)
    for vid,c in controls.items():
        if c.prepare_target:
            assert p.last_diagnostics['by_vehicle'][vid]['link_calendar'][c.prepare_target][0] is None


def test_node_delivery_prefix_is_valued_without_full_workflow_finish():
    s=forecast_state();s['episode_frames_remaining']=13;s['execution_state_payload_bytes']=6425
    s['control_cost_parameters'].update(base_size_bytes=2048,adapter_size_bytes=512,model_bytes_per_frame=1024)
    calendar={r['rsu_id']:[None]*8 for r in s['rsus']}
    assert not reserved_migration_opportunities(s,calendar)
    plans=delivery_migration_opportunities(s,calendar)
    assert plans and all(x['team_node_gain_state']>0 and x['prepare_feasible'] for x in plans.values())
    assert node_delivery_utility([12,30],0,13)>node_delivery_utility([14,32],0,13)


def test_book_idle_prefix_does_not_erase_own_or_other_reactive_transactions():
    s=forecast_state();c={r['rsu_id']:[None]*8 for r in s['rsus']}
    c['a'][0]=('v0','input');c['a'][4]=('v0','output');c['b'][2]=('v1','model')
    before=deepcopy(c);book_delivery_control(c,'v0',[('a','input'),('b','state')],s,JointRequestControl('a',prepare_target='b'))
    assert c['a']==before['a'] and c['b'][2]==('v1','model')


def test_ppo_and_dqn_credit_share_team_return_including_early_terminal():
    p=policy();row,rs=episode(env(),p)
    targets=[r['credit_targets'].clone() for r in rs];captured=[]
    from src.agents.ppo_agent import TailControlPPO
    original=TailControlPPO.update
    def inspect(self,rollouts,**kw):
        for r in rollouts:assert torch.equal(r['credit_targets'],r['team_credit_targets'])
        captured.append(True);return dict()
    try:
        TailControlPPO.update=inspect;p.update(rs)
    finally:TailControlPPO.update=original
    assert captured and all(torch.equal(r['credit_targets'],t) for r,t in zip(rs,targets))
    q=policy('dqn');r={'initial_request_count':8,'request_rewards':{'a':-1.,'b':2.},'returns':torch.tensor([.7])}
    assert q._transition_reward(r,0,'a',False)==pytest.approx(1/80)
    assert q._transition_reward(r,0,'a',True)==pytest.approx(.7/8)


def test_surrogate_loss_scale_does_not_clip_actor_gradient_or_reorder_actions():
    left=policy();right=policy();d=env().decision_states();_,r=left.act_controls(d,0,deterministic=True)
    before=left._evaluate(r['data'],deterministic=True)[0]
    for p in left.surrogates.parameters():p.data.add_(100)
    assert torch.equal(before,left._evaluate(r['data'],deterministic=True)[0])
    left=policy()
    for agent,scale in [(left,1.),(right,10000.)]:
        out=agent._evaluate(r['data'],actions=r['actions'])
        loss=out[1].sum()+scale*out[4].square().mean()
        agent._apply_control_gradients(loss)
    assert all(torch.equal(p,q) for p,q in zip(left._actor_parameters,right._actor_parameters))


def test_async_future_model_must_fit_entire_idle_prefix_before_output():
    e=env();p=policy();s=e.decision_states()['v0'];state=s['semantic_state']
    p._calendar={r['rsu_id']:[None]*8 for r in state['rsus']}
    # Both toy objects need one slot at their declared byte rate.
    p._calendar['b'][1]=('v1','output')
    data,candidates,*_=p._coordinated_features(s,0,{})
    for k,(rid,nid,kind) in enumerate(candidates[1]):
        if rid=='b' and kind=='adapter':assert not data['m1'][k]
    assert p._calendar['b'][0] is None  # one free slot is insufficient


def test_future_prefetch_does_not_evict_another_current_adapter():
    from src.envs.core.joint_delivery_planning import preserves_current_model_demands
    s=forecast_state();other=deepcopy(s);other['primary_vehicle_id']='v1'
    decisions={'v0':{'semantic_state':s},'v1':{'semantic_state':other}};targets={'v0':'a','v1':'b'}
    n=dict(s['current_workflow_node'],node_id='future',required_adapter='new')
    s['control_cost_parameters'].update(base_size_bytes=8,adapter_size_bytes=2)
    row=dict(rsu_id='b',resident_object_ids=['base:base','adapter:adapter0'],loading_object_ids=[],occupied_bytes=10,capacity_bytes=10)
    catalog={'adapter:adapter0':'base:base','adapter:unused':'base:base'}
    assert not preserves_current_model_demands(s,row,n,'adapter',decisions,targets,catalog)
    row.update(resident_object_ids=['base:base','adapter:unused','adapter:adapter0'],occupied_bytes=12,capacity_bytes=12)
    assert preserves_current_model_demands(s,row,n,'adapter',decisions,targets,catalog)


def test_current_node_migration_proxy_is_not_copied_to_a_different_adapter():
    p=policy();d=env().decision_states()['v0'];d['semantic_state']=forecast_state()
    p._calendar={r['rsu_id']:[None]*8 for r in d['semantic_state']['rsus']}
    p._reserved_opportunities=lambda state,calendar:dict(b=dict(cache_net_reward_proxy=1.,eta_frames=3,prepare_budget_frames=2,prepare_feasible=True))
    data,candidates,_,node,_=p._coordinated_features(d,0,{})
    future=[k for k,(rid,nid,kind) in enumerate(candidates[1]) if nid is not None and nid!=node]
    assert future and all(not data['f1'][k,24:27].any() for k in future)


def test_raw_audit_rejects_optional_eviction_of_a_team_current_dependency():
    from scripts.audit_joint_delivery_resources import audit_optional_cache_evictions
    e=env();workflows=e.workflows
    peer=dict(vehicle_id='v1',node_id='n0',control=dict(execution_target='b'),output_source_rsu_id=None)
    event=dict(vehicle_id='v0',node_id='n0',control=dict(execution_target='a',cache_target='b',cache_node_id='n1'),output_source_rsu_id=None,cache_request=dict(evicted_object_ids=['adapter:adapter0']))
    with pytest.raises(AssertionError,match='current team model'):audit_optional_cache_evictions(event,[event,peer],workflows)
    event['cache_request']['evicted_object_ids']=['adapter:unused']
    audit_optional_cache_evictions(event,[event,peer],workflows)


def test_request_finishing_before_departure_never_prepares():
    s=forecast_state();node=s['workflow']['nodes'][-1]
    s['workflow'].update(completed_node_ids=['n0','n1'],current_node_id=node['node_id']);s['current_workflow_node']=node
    s['predictions']['predicted_positions_by_vehicle']['v0']=[[1,0],[2,0],[3,0],[4,0],[5,0],[6,0]]
    s['control_cost_parameters'].update(base_size_bytes=2048,adapter_size_bytes=512,model_bytes_per_frame=1024);s['execution_state_payload_bytes']=6425
    from src.envs.core.joint_delivery_planning import node_delivery_times
    assert node_delivery_times(s,'a')==[4.]
    assert not delivery_migration_opportunities(s,{r['rsu_id']:[None]*8 for r in s['rsus']})
    s['predictions']['predicted_positions_by_vehicle']['v0']=[[1,0],[6,0]]
    assert not delivery_migration_opportunities(s,{r['rsu_id']:[None]*8 for r in s['rsus']})  # exactly zero net, not +8e-18
    s['predictions']['predicted_positions_by_vehicle']['v0']=[[1,0],[2,0],[3,0],[4,0],[5,0],[6,0]]
    other=deepcopy(s);other['primary_vehicle_id']='v1';other['current_node_work_units_required']=20
    decisions={'v0':{'semantic_state':s},'v1':{'semantic_state':other}}
    assert delivery_migration_opportunities(s,{r['rsu_id']:[None]*8 for r in s['rsus']},decisions=decisions,targets={'v0':'a','v1':'a'})


def test_departure_delay_never_recharges_nodes_already_delivered():
    # The first node arrives before departure and must keep its reward and time.
    assert node_delivery_utility([4.,10.,20.],5.,12.,1)==pytest.approx(1-.12)
    assert node_delivery_utility([4.,10.,20.],1.,12.,1)==pytest.approx(2-.12)
    # Once the entire request finishes, an irrelevant future migration delay has zero value.
    assert node_delivery_utility([4.],7.,50.,1)==node_delivery_utility([4.],2.,50.,1)


def test_v2_checkpoint_refuses_v1_contract(tmp_path):
    p=policy();path=tmp_path/'v1.pt';p.save_checkpoint(path);saved=torch.load(path,weights_only=False)
    saved['contract']='delivery_parameterized_ppo_v1';torch.save(saved,path)
    with pytest.raises(ValueError,match='incompatible'):type(p).from_checkpoint(path)


def test_reserved_reactive_miss_can_load_off_the_proactive_cache_clock():
    e=env();p=policy();states=e.decision_states();p._delivery_decisions=states;p._team_targets={'v0':'b','v1':'a'}
    p._calendar=reactive_link_calendar(states,p._team_targets,8)
    data,candidates,_,node,_=p._coordinated_features(states['v0'],1,{})
    k=candidates[1].index(('b',node,'adapter'))
    assert p._calendar['b'][0]==('v0','model') and data['m1'][k]
    assert not any(data['m1'][j] for j,(rid,nid,kind) in enumerate(candidates[1]) if nid is not None and nid!=node)
