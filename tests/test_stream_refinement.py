"""Failure-derived gates: balanced coverage, forecast authority and cold costs."""
from copy import deepcopy
import pytest
import torch
from src.envs.core.joint_stream_refinement import (balanced_training_schedule,
    state_restore_frames, optional_cache_value)
from tests.test_stream_control_learning import env, policy


def test_balanced_schedule_has_all_conditions_and_sizes_in_both_halves():
    cases=[dict(case_id=f'train_{i:03d}') for i in range(72)]
    schedule=balanced_training_schedule(cases,64)
    assert len(schedule)==64
    for half in [schedule[:32],schedule[32:]]:
        conditions=[int(x.split('_')[1])%24 for x in half]
        assert set(conditions)==set(range(24))
        counts=[sum(c//8==size for c in conditions) for size in range(3)]
        assert max(counts)-min(counts)<=1
    assert {int(x.split('_')[1])//24 for x in schedule}=={0,1,2}
    with pytest.raises(ValueError):balanced_training_schedule(cases[:24],64)


def test_real_minimum_state_slots_and_exact_version_copy():
    s=env().decision_states()['v']['semantic_state'];s['execution_state_version']=2
    s['execution_location']='a';s['execution_state_payload_bytes']=1
    minimum=s['control_cost_parameters']['minimum_state_frames']
    assert state_restore_frames(s,'b')==minimum
    s['state_copies_by_rsu']['b']={'version':2,'payload_bytes':1}
    assert state_restore_frames(s,'b')==0
    s['state_copies_by_rsu']['b']['version']=1
    assert state_restore_frames(s,'b')>=minimum


def test_optional_preparation_requires_use_in_horizon_and_positive_survival_gain():
    s=env().decision_states()['v']['semantic_state'];row=next(r for r in s['rsus'] if r['rsu_id']=='a')
    n=s['workflow']['nodes'][1];s['predictions']={'predicted_positions_by_vehicle':{'v':[(0,0)]*8},'prediction_confidence_by_vehicle':{'v':1.}}
    value=optional_cache_value(s,row,n,1,'a',6,prediction_enabled=True,horizon=8)
    assert value['eligible'] and value['net_delivery_proxy']>0
    s['predictions']['predicted_positions_by_vehicle']['v']=[(100,100)]*8
    assert not optional_cache_value(s,row,n,1,'a',6,prediction_enabled=True,horizon=8)['eligible']
    queued=s['queued_workflows'][0]['workflow']['nodes'][0]
    assert not optional_cache_value(s,row,queued,1,'a',6,prediction_enabled=False,horizon=8)['eligible']


def test_remote_current_preparation_can_use_causal_migration_eta():
    s=env().decision_states()['v']['semantic_state'];row=next(r for r in s['rsus'] if r['rsu_id']=='a');n=s['current_workflow_node']
    s['predictions']={'predicted_positions_by_vehicle':{'v':[(100,100),(0,0),(0,0)]},'prediction_confidence_by_vehicle':{'v':1.}}
    assert optional_cache_value(s,row,n,1,'b',6,prediction_enabled=True,horizon=8,use_after_override=2)['eligible']
    assert not optional_cache_value(s,row,n,1,'b',6,prediction_enabled=False,horizon=8,use_after_override=2)['eligible']


def test_DT_context_responds_to_prediction_and_DAG_context_does_not():
    d=env().decision_states()['v'];p=policy('dt_handoff_drl');q=policy('dag_offload_drl')
    data=p._coordinated_features(d,0,{})[0];batch={k:torch.as_tensor(v).unsqueeze(0) for k,v in data.items()}
    changed=deepcopy(batch);changed['f0'][...,10]=1.;changed['f0'][...,11]=.5
    latent=torch.zeros(1,64,16)
    assert not torch.equal(p._policy_context(batch,latent),p._policy_context(changed,latent))
    assert torch.equal(q._policy_context(batch,latent),q._policy_context(changed,latent))


def test_cold_cache_prior_prices_loading_and_old_v1_checkpoint_is_rejected(tmp_path):
    d=env().decision_states()['v'];p=policy();before=p._coordinated_features(d,0,{})[0]
    warm=deepcopy(d);row=next(r for r in warm['semantic_state']['rsus'] if r['rsu_id']=='a')
    row['ready_base_model_ids']=['base'];row['cached_adapter_ids']=['x']
    after=p._coordinated_features(warm,0,{})[0]
    assert before['prior0'][1]<after['prior0'][1]
    path=tmp_path/'v1.pt';p.save_checkpoint(path);saved=torch.load(path,weights_only=False)
    saved.update(contract='stream_parameterized_ppo_v1',action_contract='joint_stream_control_v1');torch.save(saved,path)
    with pytest.raises(ValueError,match='incompatible'):type(p).from_checkpoint(path)
