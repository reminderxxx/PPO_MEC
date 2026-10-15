"""Scheduled contracts: actual slot protection, scale, action credit and DQN TD."""
from dataclasses import replace
from copy import deepcopy
import numpy as np
import pytest
import torch
from tests.test_parameterized_control_learning import four_site_env
from tests.test_coordinated_control_learning import forecast_state
from scripts.train_joint_parameterized_control import episode
from src.agents.registry import build_agent
from src.envs.core.multi_request_target_control_env import JointRequestControl
from src.envs.core.joint_link_reservation import public_link_calendar,reserved_migration_opportunities,book_control


def env():
    e=four_site_env();e.coordination_intent_enabled=e.link_reservation_enabled=True
    return e


def policy(name='sa_ghmappo',**kwargs):
    torch.set_num_threads(1)
    return build_agent(name,control_contract='joint_parameterized_control_v3',scheduled_controls=True,**kwargs)


@pytest.mark.parametrize('name',['sa_ghmappo','mappo','ppo','dqn','cache_offload_drl','dag_offload_drl','dt_handoff_drl'])
def test_all_algorithms_exact_probabilities_learn_reload_and_replay(name,tmp_path):
    e=env();p=policy(name);initial={k:v.clone() for k,v in p.state_dict().items()}
    controls,r=p.act_controls(e.decision_states(),0)
    assert torch.allclose(p.distributions(r['data'],actions=r['actions'])[1],r['head_logs'].sum(1),atol=1e-6)
    assert all(len({rid for rid,kind in c.link_reservations})==len(c.link_reservations) for c in controls.values())
    _,rolls=episode(e,p)
    if name=='dqn':
        while sum(len(r['actions']) for r in rolls)<128:rolls+=episode(env(),p)[1]
    losses=p.update(rolls,epochs=1)
    assert all(np.isfinite(v) for v in losses.values())
    assert any(not torch.equal(v,p.state_dict()[k]) for k,v in initial.items() if k.startswith('heads.'))
    if name=='dqn':
        assert p.replay and losses['gradient_steps']>0 and losses['policy_loss']==0
        assert any(not torch.equal(v,p.target_heads.state_dict()[k]) for k,v in p.heads.state_dict().items())
    path=tmp_path/f'{name}.pt';p.save_checkpoint(path);q=type(p).from_checkpoint(path)
    a,ar=p.act_controls(env().decision_states(),0,deterministic=True)
    b,br=q.act_controls(env().decision_states(),0,deterministic=True)
    # Reset target intent before exact event comparison.
    row,_=episode(env(),p,deterministic=True);other,_=episode(env(),q,deterministic=True)
    assert row==other
    assert torch.equal(p.state_dict()['completed_updates'],q.state_dict()['completed_updates'])


def test_reserved_output_cannot_be_stolen_by_an_earlier_cache_request():
    e=env();e.output_pending['v1']=dict(node_id='n0',rsu_id='a',remaining_bytes=2)
    e.execution_location['v1']='a';e.state_version['v1']=1;e.state_payload_bytes['v1']=2
    controls={'v0':JointRequestControl(None,'a','n0',priority=3),
              'v1':JointRequestControl('a',link_reservations=(('a','output'),))}
    _,_,_,info=e.step_controls(controls);events={x['vehicle_id']:x for x in info['events']}
    assert events['v0']['cache_request']['link_slots_consumed']==0
    assert events['v1']['status']=='node_completed'
    assert events['v1']['output_transfer_link_frames']==1
    assert info['events'][0]['frame_cost_receipt']['link_usage_by_rsu']['a']==1


def test_link_claims_are_opt_in_and_duplicate_claims_rejected():
    e=four_site_env();controls={v:JointRequestControl(None) for v in e.decision_states()}
    controls['v0']=JointRequestControl(None,link_reservations=(('a','output'),))
    with pytest.raises(ValueError,match='scheduled environment'):e.step_controls(controls)
    e=env();controls['v0']=JointRequestControl(None,link_reservations=(('a','state'),('a','model')))
    with pytest.raises(ValueError,match='one reservation'):e.step_controls(controls)


def test_public_calendar_protects_current_io_and_nonpreemptive_model_backlog():
    e=env();s=e.decision_states()
    s['v0']['semantic_state']['output_pending']=dict(rsu_id='a',remaining_bytes=20)
    for d in s.values():
        for r in d['semantic_state']['rsus']:
            if r['rsu_id']=='b':r['loading_remaining_frames']={'base:x':2}
    c=public_link_calendar(s,{'v0':'a','v1':'c'})
    assert c['a'][0]==('v0','output') and c['b'][:2]==[('background','model')]*2
    assert all(len(slots)==3 for slots in c.values())


def test_preparation_budget_respects_free_slots_and_rejects_zero_saved_delay():
    s=forecast_state();s['control_cost_parameters'].update(base_size_bytes=2048,adapter_size_bytes=512,model_bytes_per_frame=1024)
    s['execution_state_payload_bytes']=6425
    free={r['rsu_id']:[None]*3 for r in s['rsus']}
    full=reserved_migration_opportunities(s,free)
    assert full and all(v['state_net_reward_proxy']>0 for v in full.values())
    busy={r['rsu_id']:[('other','output')]*3 for r in s['rsus']}
    assert not reserved_migration_opportunities(s,busy)
    one={r['rsu_id']:[None,('other','output'),('other','output')] for r in s['rsus']}
    assert not any(v['prepare_feasible'] for v in reserved_migration_opportunities(s,one).values())


def test_scaled_graph_and_local_ppo_critic_are_bounded_and_have_distinct_contract():
    e=env();p=policy();decision=e.decision_states()['v0']
    for n in decision['semantic_state']['workflow']['nodes']:n['output_size']=1600
    data,*_=p._coordinated_features(decision,0,{})
    assert data['graph'][:,1].max()<1
    assert data['f1'][:,31].max()<1
    latent=p.graph_encoder(torch.tensor(data['graph']))
    assert not bool((latent.abs()>.99).any())
    local,*_=policy('ppo')._coordinated_features(decision,0,{})
    assert not local['critic'][20:].any()
    assert p.checkpoint_contract!='coordinated_parameterized_ppo_v1'


def test_same_target_cache_and_prepare_are_not_jointly_sampled_and_baseline_is_action_independent():
    e=env();p=policy();controls,r=p.act_controls(e.decision_states(),0)
    result=p._evaluate(r['data'],actions=r['actions'],return_details=True)
    assert torch.allclose(result[7],r['counterfactual_baselines'],atol=1e-6)
    alternative=r['actions'].clone();alternative[:,2]=(alternative[:,2]+1)%4
    changed=p._evaluate(r['data'],actions=alternative,return_details=True)
    assert torch.allclose(result[7][:,2],changed[7][:,2],atol=1e-6)
    for c in controls.values():assert not c.prepare_target or c.prepare_target not in (c.execution_target,c.cache_target)
    assert p.surrogate_mse==1


def test_book_control_updates_future_prefix_without_stealing_other_reservations():
    s=forecast_state();c={r['rsu_id']:[None]*3 for r in s['rsus']};c['b'][1]=('other','output')
    book_control(c,'v0',(('b','state'),),s,JointRequestControl('a',prepare_target='b'))
    assert c['b']==[('v0','state'),('other','output'),('v0','state')]


def test_eight_frame_predictor_fit_split_shape_and_causal_reload():
    from src.envs.core.joint_learned_prediction import fit_predictor,LearnedMobilityPredictor
    e=env();windows=[dict(frame_offset=0,frames=e.frames[:25]),dict(frame_offset=25,frames=e.frames[25:])]
    bundle=fit_predictor(windows,horizon=8,epochs=1)
    assert bundle['horizon']==8 and bundle['fit_offsets']==[0] and bundle['validation_offsets']==[25]
    predictor=LearnedMobilityPredictor(bundle)
    v=e.frames[0]['vehicles'];result=predictor.predict(v,e.mapper.associate(v),e.rsus,e.mapper,0)
    assert all(len(points)==8 for points in result['predicted_positions_by_vehicle'].values())


def test_cost_auxiliary_counts_each_admitted_object_minimum_slot_not_fractional_bytes():
    e=env();row,rolls=episode(e,policy(),deterministic=True)
    # Cold closure has an 8-byte base and 2-byte adapter, each needing a distinct
    # slot at 8 bytes/frame: two future slots, not just the one admitted now.
    assert torch.isclose(rolls[0]['cost_targets'][:,0].max(),torch.tensor(.2))
