from copy import deepcopy
from pathlib import Path
import pytest
from src.envs.core.persistent_workflow_env import PersistentWorkflowEnv,build_batch
from src.data.mobility.replay_provider import ReplayProvider
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.data.workflow.toy_workflow_generator import ToyWorkflowGenerator
from src.envs.specs import RSUState,ActionAdapter
from src.runtime.subject_binding import SubjectBindingBoundaryError

ROOT=Path(__file__).resolve().parents[1]

def env(horizon=20):
    frames=[dict(time_index=i*100,vehicles=[dict(vehicle_id='fixed',position_x=0.,position_y=0.,speed=0.,base_model_id='veh_base_v1')]) for i in range(30)]
    return PersistentWorkflowEnv(workflows=[ToyWorkflowGenerator().generate()],subject_id='fixed',request_count=2,
        mobility_provider=ReplayProvider(frames),rsu_states=[RSUState('rsu_a',0.,0.,100.)],
        adapter_catalog=AdapterCatalog.from_json(ROOT/'src/data/model_catalog/typed_model_cache_controlled.json'),
        max_steps=horizon,cache_capacity_profile=dict(model_cache_profile_id='typed_base_adapter_state_v1',enabled=True,
            unit='mb',capacity_mb=288.,count_base_model_separately=True,eviction_policy='lru',typed_eviction_semantics='sequential_dependency_recompute_lru_v1'))


def test_batch_preserves_edges_and_distinct_request_identity():
    wf=ToyWorkflowGenerator().generate();g,requests=build_batch([wf],3)
    assert len(g.nodes)==3*len(wf.nodes)
    assert len(set(g.execution_order))==len(g.execution_order)
    assert len(g.edges)==3*len(wf.edges)
    assert all(r['release_step']==0 for r in requests)


def test_loading_blocks_service_and_repeated_admission_does_not_recharge():
    e=env();state,_=e.reset()
    node=e.workflow_state.current_node();node.required_adapter='adapter_control'
    e._last_state['current_workflow_node']=node.to_dict()
    results=[]
    for _ in range(12):
        state,reward,done,truncated,info=e.step(ActionAdapter().decode(0,e._last_state));results.append(info['persistent_stream'])
        if done:break
    assert not results[0]['completed_requests']
    assert results[0]['transfer_mb']>0
    assert results[1]['transfer_mb']==0
    assert e.workflow_state.completed_node_ids
    assert all(r['subject_id']=='fixed' for r in results)


def test_horizon_is_terminal_with_all_unfinished_in_denominator():
    e=env(1);state,_=e.reset();_,reward,done,truncated,info=e.step(ActionAdapter().decode(3,state))
    row=info['persistent_stream']
    assert done and not truncated and row['pending_requests']==2
    assert row['reward_terms']['unfinished']==-10
    assert row['pending_area_seconds']==pytest.approx(.2)
    assert reward.total==pytest.approx(sum(row['reward_terms'].values()))
    with pytest.raises(RuntimeError):e.step(ActionAdapter().decode(3,state))


def test_subject_loss_is_not_silently_replaced():
    e=env();state,_=e.reset();e._mobility_provider._trajectory_frames[1]['vehicles'][0]['vehicle_id']='replacement'
    before=deepcopy(e.workflow_state)
    with pytest.raises(SubjectBindingBoundaryError):e.step(ActionAdapter().decode(0,state))
    assert e.workflow_state==before


def test_no_reset_between_requests():
    e=env(29);state,_=e.reset();identity=id(e.rsu_states)
    for _ in range(20):
        state,_,done,_,_=e.step(ActionAdapter().decode(0,e._last_state))
        assert id(e.rsu_states)==identity
        if done:break


def test_minimal_graph_policy_has_no_hierarchy_or_auxiliary():
    from src.agents.sa_ghmappo_agent import DirectWorkflowGraphPPOAgent
    a=DirectWorkflowGraphPPOAgent(random_seed=7)
    assert not a._use_hierarchy and a._auxiliary_coef==0
    e=env();s,i=e.reset()
    from src.envs.specs import ActionMaskBuilder
    action,details=a.act(None,dict(semantic_state=s,action_mask=ActionMaskBuilder().build_mask(s),run_metadata=dict(policy_evaluation_mode='raw_policy')))
    assert action==details['sampled_base_action']


def test_delayed_use_has_strictly_earlier_admission_and_no_instant_loading():
    e=env(29);s,_=e.reset();node=e.workflow_state.current_node();node.required_adapter='adapter_control'
    e._last_state['current_workflow_node']=node.to_dict()
    for _ in range(20):
        s,_,done,_,info=e.step(ActionAdapter().decode(0,e._last_state))
        for use in info['persistent_stream']['later_load_use']:
            assert use['step']<use['execution_step']
            assert use['ready_at_seconds']<=e._seconds()+1e-9
        if done:break
    assert any(x['later_load_use'] for x in e.stream_rows)


def test_reward_has_no_native_exploration_or_positive_offset():
    e=env(2);s,_=e.reset()
    s,reward,_,_,info=e.step(ActionAdapter().decode(0,s))
    assert reward.positive_offset==reward.mechanism_exploration_bonus==0.
    assert reward.total==pytest.approx(sum(info['persistent_stream']['reward_terms'].values()))
    assert info['persistent_stream']['request_count']==2


@pytest.mark.parametrize('graph',[True,False])
def test_direct_policy_learns_from_new_reward_and_checkpoint_roundtrips(tmp_path,graph):
    import torch
    from src.agents.sa_ghmappo_agent import DirectWorkflowGraphPPOAgent
    from src.envs.wrappers.gym_vec_env import GymVecEnv
    from src.metrics.recorder import EpisodeRecorder
    from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer
    a=DirectWorkflowGraphPPOAgent(use_graph=graph,random_seed=7)
    recorder=EpisodeRecorder();wrapped=GymVecEnv(env(),recorder,
        subject_boundary_receipt_path=str(tmp_path/'boundary.json'),subject_boundary_exposure_id='test')
    _,rollout=MARLOnPolicyTrainer(wrapped,a,recorder,max_steps=8).collect_episode(
        run_metadata=dict(policy_evaluation_mode='raw_policy'),collect_model_targets=False)
    before={k:v.clone() for k,v in a._network.state_dict().items()}
    a.learn(rollout)
    assert any(not torch.equal(v,before[k]) for k,v in a._network.state_dict().items())
    path=tmp_path/'new.pt';a.save(str(path));b=DirectWorkflowGraphPPOAgent(use_graph=graph,random_seed=7);b.load(str(path))
    assert all(torch.equal(v,b._network.state_dict()[k]) for k,v in a._network.state_dict().items())
