"""Joint reservation, physical forecast opportunity, credit and probability witnesses."""
from collections import Counter
from copy import deepcopy
import numpy as np
import pytest
import torch
from tests.test_parameterized_control_learning import four_site_env
from src.agents.registry import build_agent
from src.envs.core.joint_control_planning import migration_opportunities, request_reward_ledger
from scripts.train_joint_parameterized_control import episode


def agent(name='sa_ghmappo',**kwargs):
    torch.set_num_threads(1)
    return build_agent(name,control_contract='joint_parameterized_control_v3',coordinated_controls=True,**kwargs)


@pytest.mark.parametrize('name',['sa_ghmappo','mappo','cache_offload_drl'])
def test_coordinated_probability_target_binding_update_and_checkpoint(name,tmp_path):
    env=four_site_env();env.coordination_intent_enabled=True
    policy=agent(name);before={k:v.clone() for k,v in policy.state_dict().items()}
    commands,roll=policy.act_controls(env.decision_states(),0)
    _,logs,_,_,surrogate=policy.distributions(roll['data'],actions=roll['actions'])
    assert torch.allclose(logs,roll['head_logs'].sum(1),atol=1e-6)
    assert torch.allclose(logs,roll['logs'],atol=1e-6)
    if policy.surrogate_enabled:assert bool((surrogate[...,1:]>=0).all())
    for vid,c in commands.items():
        assert c.execution_target==c.cache_target and c.cache_node_id==env.workflows[vid].current_node_id
        assert c.prepare_target is None
    row,rolls=episode(env,policy)
    assert all('credit_targets' in r and 'prefix_values' in r for r in rolls)
    policy.update(rolls,epochs=1)
    assert any(not torch.equal(v,policy.state_dict()[k]) for k,v in before.items())
    path=tmp_path/'coordinated.pt';policy.save_checkpoint(path);restored=type(policy).from_checkpoint(path)
    assert restored.checkpoint_contract=='coordinated_parameterized_ppo_v1'
    env=four_site_env();policy.reset_controller();restored.reset_controller()
    a,ar=policy.act_controls(env.decision_states(),0,deterministic=True)
    b,br=restored.act_controls(env.decision_states(),0,deterministic=True)
    assert a==b and torch.equal(ar['logs'],br['logs'])


def test_same_frame_intents_change_load_features_and_remove_synthetic_target_aliasing():
    env=four_site_env();p=agent();d=env.decision_states()['v0']
    a,c,_,_,_=p._coordinated_features(d,0,{})
    b,_,_,_,_=p._coordinated_features(d,0,{'v1':('b',60.)})
    assert b['prior0'][2] < a['prior0'][2]-1
    assert b['f0'][2,21]==.5
    n1=next(i for i,item in enumerate(c[1]) if item==('b','n1','adapter'))
    n2=next(i for i,item in enumerate(c[1]) if item==('b','n2','adapter'))
    assert not np.array_equal(a['f1'][n1],a['f1'][n2])
    _,roll=p.act_controls(env.decision_states(),0,deterministic=True)
    assert sorted(p.last_diagnostics['request_order'])==sorted(env.decision_states())
    assert any(r['b']>0 or r['a']>0 or r['c']>0 or r['d']>0
        for r in [v['reserved_service_frames'] for v in p.last_diagnostics['by_vehicle'].values()])


def forecast_state():
    env=four_site_env();s=deepcopy(env.decision_states()['v0']['semantic_state'])
    s.update(execution_location='a',execution_state_version=4,execution_state_payload_bytes=3072)
    for r in s['rsus']:
        r['position_x']=0 if r['rsu_id']=='a' else 10
        r['position_y']=0;r['coverage_radius']=5 if r['rsu_id']=='a' else 10
        r['service_request_count']=0
        r['ready_base_model_ids']=['base'];r['cached_adapter_ids']=['adapter0']
    s['predictions']=dict(predicted_positions_by_vehicle={'v0':[[3,0],[4,0],[6,0]]},
                         prediction_confidence_by_vehicle={'v0':1.},position_uncertainty_margin=0.)
    return s


def test_migration_requires_execution_coverage_loss_feasible_transfer_and_positive_net():
    s=forecast_state();op=migration_opportunities(s)
    assert op and all(v['net_reward_proxy']>0 and v['prepare_frames']<=v['eta_frames'] for v in op.values())
    assert not migration_opportunities(s,prediction_enabled=False)
    overlap=deepcopy(s);overlap['rsus'][0]['coverage_radius']=20
    assert not migration_opportunities(overlap)
    outside=deepcopy(s);outside['vehicles'][0]['position_x']=100
    assert not migration_opportunities(outside)
    too_late=deepcopy(s);too_late['execution_state_payload_bytes']=4096
    assert not migration_opportunities(too_late)
    expensive=deepcopy(s)
    for r in expensive['rsus']:r['service_request_count']=100
    assert not migration_opportunities(expensive)
    ready=deepcopy(s);ready['state_copies_by_rsu']={rid:dict(version=4,payload_bytes=3072) for rid in ('b','c','d')}
    assert not migration_opportunities(ready)


def test_ledger_and_charged_intents_conserve_reward_and_local_credit_differs():
    env=four_site_env();env.coordination_intent_enabled=True;policy=agent()
    for node in env.workflows['v1'].nodes:node.input_size=16
    row,rolls=episode(env,policy,deterministic=True)
    for e in env.events:
        if 'frame_cost_receipt' not in e:continue
        r=e['frame_cost_receipt']
        assert abs(sum(r['request_rewards'].values())-sum(r['reward_components'].values()))<1e-9
        assert r['control_intent_transmitted_bytes']>0
        assert r['shared_state_transmitted_bytes']>=r['control_intent_transmitted_bytes']
    assert row['control_intent_transmitted_bytes']>0
    assert any(not torch.allclose(r['credit_targets'],r['returns']/2) for r in rolls)


def test_no_opportunity_disables_only_prepare_and_preserves_predictor_branch():
    env=four_site_env();full=agent(seed=7);without=agent(seed=7,prediction_enabled=False)
    a,ar=full.act_controls(env.decision_states(),0,deterministic=True)
    b,br=without.act_controls(env.decision_states(),0,deterministic=True)
    assert a==b and torch.equal(ar['logs'],br['logs'])
    assert full.prediction_enabled and all(c.prepare_target is None for c in a.values())


@pytest.mark.parametrize('options',[dict(credit_mode='team'),dict(coordination_enabled=False),dict(migration_gate_enabled=False)])
def test_ablation_options_update_and_restore(options,tmp_path):
    env=four_site_env();env.coordination_intent_enabled=True;p=agent(**options)
    _,rolls=episode(env,p);loss=p.update(rolls,epochs=1)
    assert all(np.isfinite(v) for v in loss.values())
    path=tmp_path/'ablation.pt';p.save_checkpoint(path);restored=type(p).from_checkpoint(path)
    assert all(getattr(restored,k)==v for k,v in options.items())


def test_inflight_certified_prefix_can_finish_and_late_episode_is_noop():
    s=forecast_state();s['execution_state_payload_bytes']=3073
    s['state_transfers_by_rsu']={'b':dict(remaining_frames=1,remaining_bytes=1024,payload_bytes=3072)}
    assert migration_opportunities(s)['b']['prepare_feasible']
    s['episode_frames_remaining']=2
    assert not migration_opportunities(s)
