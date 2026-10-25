from copy import deepcopy
from pathlib import Path
import pytest
import torch
from scripts.run_workflow_recovery import agent_for, save_agent, load_agent, expert
from src.envs.wrappers.persistent_workflow_control import ReleasedPreparationEnv,PreparationGymEnv,candidates,CONTROL_CONTRACT
from src.encoders.persistent_workflow_encoder import persistent_features
from src.data.mobility.replay_provider import ReplayProvider
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.data.workflow.toy_workflow_generator import ToyWorkflowGenerator
from src.envs.specs import RSUState
from src.metrics.recorder import EpisodeRecorder
from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer

ROOT=Path(__file__).resolve().parents[1]

def fixture_env(tmp_path,horizon=20):
    frames=[dict(time_index=i*100,vehicles=[dict(vehicle_id='fixed',position_x=0.,position_y=0.,speed=0.,base_model_id='veh_base_v1')]) for i in range(30)]
    core=ReleasedPreparationEnv(workflows=[ToyWorkflowGenerator().generate()],subject_id='fixed',request_count=2,
        mobility_provider=ReplayProvider(frames),rsu_states=[RSUState('rsu_a',0.,0.,100.)],
        adapter_catalog=AdapterCatalog.from_json(ROOT/'src/data/model_catalog/typed_model_cache_controlled.json'),max_steps=horizon,
        cache_capacity_profile=dict(model_cache_profile_id='typed_base_adapter_state_v1',enabled=True,unit='mb',capacity_mb=384.,count_base_model_separately=True,eviction_policy='lru',typed_eviction_semantics='sequential_dependency_recompute_lru_v1'))
    for i,n in enumerate(core._workflow_template.nodes):n.required_adapter=['adapter_control','adapter_fusion','adapter_intent'][min(i,2)]
    return PreparationGymEnv(core,EpisodeRecorder(),subject_boundary_receipt_path=str(tmp_path/'boundary.json'),subject_boundary_exposure_id='test')


def test_future_prepare_changes_requested_object_without_changing_service_node(tmp_path):
    e=fixture_env(tmp_path);_,info=e.reset(seed=7)
    before=info['semantic_state']['current_workflow_node']
    opts=candidates(info['semantic_state']);assert opts[1]['adapter_id']=='adapter_fusion'
    _,_,_,_,info=e.step(1)
    event=info['persistent_stream']['cache_event']
    assert info['control_action']['cache_action']['adapter_id']=='adapter_fusion'
    assert event['node_id']==before['node_id']
    assert info['semantic_state']['current_workflow_node']['node_id']==before['node_id']
    assert any(x['object_id']=='adapter:fusion' for x in event['admitted_typed_objects'])
    assert event['transfer_mb_by_type']['adapter']>0


def test_redundant_admission_is_masked_and_invalid_action_never_remapped(tmp_path):
    e=fixture_env(tmp_path);_,info=e.reset(seed=7)
    _,_,_,_,info=e.step(0)
    assert not info['action_mask'][0]
    with pytest.raises(ValueError,match='no fallback'):e.step(0)


@pytest.mark.parametrize('field',['loading','horizon','pending','membership'])
def test_new_fields_reach_encoder_and_raw_logits(tmp_path,field):
    e=fixture_env(tmp_path);_,info=e.reset(seed=7);s=info['semantic_state'];b=deepcopy(s)
    if field=='loading':b['model_loading']=[dict(rsu_id='rsu_a',object_id='adapter:control',remaining_seconds=5.)]
    if field=='horizon':b['request_stream']['remaining_horizon_steps']=1
    if field=='pending':b['request_stream']['pending_request_count']=1
    if field=='membership':b['request_stream']['requests']=b['request_stream']['requests'][:1]
    assert not torch.equal(persistent_features(s),persistent_features(b))
    a=agent_for('graph_ppo',7)
    with torch.no_grad():
        x=a._forward_policy(s,run_metadata={'policy_evaluation_mode':'raw_policy'})
        y=a._forward_policy(b,run_metadata={'policy_evaluation_mode':'raw_policy'})
    assert not torch.equal(x['flat_logits'],y['flat_logits'])


@pytest.mark.parametrize('method',['graph_ppo','set_ppo','mappo','qmix'])
def test_real_two_update_chain_and_checkpoint_contract(tmp_path,method):
    a=agent_for(method,7)
    for j in range(2):
        d=tmp_path/str(j);d.mkdir();e=fixture_env(d)
        _,rows=MARLOnPolicyTrainer(e,a,e._recorder,max_steps=12,gamma=1.).collect_episode(run_metadata={'policy_evaluation_mode':'raw_policy'},collect_model_targets=False)
        assert all(x['decision_info']['action_mask'][x['action']] for x in rows)
        before={k:v.clone() for k,v in a._network.state_dict().items()}
        a.learn(rows)
        if method!='qmix' or j==1:assert any(not torch.equal(v,before[k]) for k,v in a._network.state_dict().items())
    path=tmp_path/'checkpoint.pt';save_agent(a,path);b=load_agent(path,method,7)
    assert all(torch.equal(v,b._network.state_dict()[k]) for k,v in a._network.state_dict().items())
    with pytest.raises(ValueError,match='contract'):load_agent(path,'set_ppo' if method!='set_ppo' else 'graph_ppo',7)


def test_graph_and_set_have_same_parameters_and_raw_information(tmp_path):
    g=agent_for('graph_ppo',7);s=agent_for('set_ppo',7)
    assert all(torch.equal(v,s._network.state_dict()[k]) for k,v in g._network.state_dict().items())
    assert g._network.encoder.base._dag_encoder._use_dependency_aware
    assert not s._network.encoder.base._dag_encoder._use_dependency_aware


def test_reward_capacity_and_true_terminal_are_preserved(tmp_path):
    e=fixture_env(tmp_path,horizon=8);_,info=e.reset(seed=7)
    for _ in range(8):
        action=expert(info)
        _,reward,done,truncated,info=e.step(action)
        row=info['persistent_stream']
        assert reward==pytest.approx(sum(row['reward_terms'].values()))
        assert all(c['resident_mb']<=c['capacity_mb']+1e-8 for c in info['semantic_state']['cache_runtime'])
        assert info['action_schema']['actions'][action]['name']==info['action_name']
        if done:break
    assert done and not truncated and row['terminal_reason']
    assert info['control_action']['metadata']['action_contract']==CONTROL_CONTRACT
