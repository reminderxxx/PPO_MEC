"""Probability of sampled paths, numerical guards, and shared native classes."""
import copy
import math
import sys
from pathlib import Path

import pytest
import torch
from torch.distributions import Categorical

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.agents.registry import build_agent
from src.envs.core.vec_workflow_core_env import make_toy_vec_env
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.evaluators.main_results_support import classify_window_activity


def output():
    return {"slow_logits":torch.tensor([.2,-.4,.1],requires_grad=True),
            "fast_logits":torch.tensor([.3,-.2],requires_grad=True),
            "event_logits":torch.tensor([.8,-3.],requires_grad=True),
            "option_logits":torch.tensor([.1,.3,-.1,.5],requires_grad=True)}


def test_many_to_one_mapping_requires_sampled_joint_likelihood():
    agent=build_agent('sa_ghmappo',device='cpu',option_gate_enabled=True)
    out=output();mask=[True]*5
    base=Categorical(logits=agent._hierarchical_env_action_scores(out))
    option=Categorical(logits=agent._masked_option_logits(out['option_logits'],[True,True,False,True],prior_target=3))
    rows=[]
    for a in range(5):
        for o in (0,1,3):
            rows.append({'action': a if o==0 else 4,'action_info':{'probability_contract':'sampled_base_and_option_v1',
                'sampled_base_action':a,'sampled_base_probs':base.probs.detach().tolist(),
                'option_gate':{'enabled':True,'option_action':o,'option_mask':[True,True,False,True],
                               'prior_target':3,'option_probs':option.probs.detach().tolist()}}})
    lp,_,_=agent._latent_policy_statistics([out]*len(rows),rows,[mask]*len(rows))
    assert lp.exp().sum().item()==pytest.approx(1.,abs=1e-6)
    final4=sum(lp[i].exp().item() for i,r in enumerate(rows) if r['action']==4)
    assert final4 > base.probs[4].item()
    i=3*3+2
    assert lp[i].item()==pytest.approx((base.log_prob(torch.tensor(3))+option.log_prob(torch.tensor(3))).item())
    assert lp[i].item()!=pytest.approx(base.log_prob(torch.tensor(4)).item())
    (-lp[i]).backward()
    assert out['option_logits'].grad is not None
    assert out['fast_logits'].grad is not None
    assert agent._sampled_policy_kl([out]*len(rows),rows,[mask]*len(rows)).max().item()<1e-6


def collected(agent):
    env=GymVecEnv(core_env=make_toy_vec_env())
    observation,info=env.reset();info=dict(info);info['run_metadata']={'window_class':'idle_or_sparse'}
    _,a=agent.act(observation,info)
    row={'action_info':a,'decision_info':info}
    state=agent._extract_semantic_state(info);out=agent._forward_policy(state,run_metadata=info['run_metadata'])
    return row,state,out,agent._extract_action_mask(info)


@pytest.mark.parametrize('name',['sa_ghmappo','ppo','mappo','controller_mat','dag_offload_drl','cache_offload_drl','dt_handoff_drl'])
def test_old_new_probability_ratio_is_one_before_any_update(name):
    agent=build_agent(name,device='cpu')
    row,state,out,mask=collected(agent)
    assert row['action_info']['probability_contract']=='sampled_base_and_option_v1'
    lp,_,_=agent._latent_policy_statistics([out],[row],[mask])
    assert lp.item()==pytest.approx(row['action_info']['log_prob'],abs=1e-6)
    assert row['action_info']['env_action_log_prob_action']==row['action_info']['sampled_base_action']
    assert row['action_info']['env_action_probability_scope']=='pre_mapping_proposal_diagnostic_only'


def test_rejected_optimizer_step_restores_weights_and_adam_state():
    agent=build_agent('ppo',device='cpu',learning_rate=100.,target_kl=1e-12)
    row,state,out,mask=collected(agent);before=copy.deepcopy(agent._network.state_dict())
    result=agent._guarded_policy_step(-out['flat_logits'][3]*100.,[row],[state],[row['decision_info']['run_metadata']],[mask])
    assert not result['accepted']
    assert not agent._optimizer.state
    for k,v in before.items():assert torch.equal(v,agent._network.state_dict()[k])


def test_accepted_optimizer_step_is_finite_and_within_policy_kl():
    agent=build_agent('ppo',device='cpu',learning_rate=1e-5,target_kl=.0014)
    row,state,out,mask=collected(agent)
    result=agent._guarded_policy_step(-out['flat_logits'][3],[row],[state],[row['decision_info']['run_metadata']],[mask])
    assert result['accepted']
    assert 0<=result['kl']<=agent._target_kl
    assert all(torch.isfinite(v).all() for v in agent._network.state_dict().values())


def test_nonfinite_loss_never_enters_optimizer():
    agent=build_agent('ppo',device='cpu');before=copy.deepcopy(agent._network.state_dict())
    with pytest.raises(FloatingPointError):agent._guarded_policy_step(torch.tensor(float('nan'),requires_grad=True),[],[],[],[])
    assert not agent._optimizer.state
    for k,v in before.items():assert torch.equal(v,agent._network.state_dict()[k])


@pytest.mark.parametrize('activity,expected',[
    ({'mechanism_signal_count':3,'active_vehicle_count_mean':3.,'predicted_next_ratio':.5,'predicted_handoff_ratio':.2,'estimated_association_change_count':3},'mechanism_activating'),
    ({'mechanism_signal_count':0,'active_vehicle_count_mean':3.,'predicted_next_ratio':.1,'predicted_handoff_ratio':.0,'estimated_association_change_count':1},'active_non_mechanism'),
    ({'mechanism_signal_count':0,'active_vehicle_count_mean':1.,'predicted_next_ratio':.0,'predicted_handoff_ratio':.0,'estimated_association_change_count':0},'idle_or_sparse')])
def test_shared_native_window_classification_without_rewards(activity,expected):
    assert classify_window_activity(**activity)==expected


def test_rejection_restores_existing_adam_moments_and_step_counters():
    agent=build_agent('ppo',device='cpu',learning_rate=1e-5,target_kl=.0014)
    row,state,out,mask=collected(agent)
    assert agent._guarded_policy_step(-out['flat_logits'][3],[row],[state],[row['decision_info']['run_metadata']],[mask])['accepted']
    assert agent._optimizer.state
    before=copy.deepcopy(agent._network.state_dict())
    agent._optimizer.param_groups[0]['lr']=100.
    optimizer_before=copy.deepcopy(agent._optimizer.state_dict())
    agent._target_kl=1e-12
    row,state,out,mask=collected(agent)
    assert not agent._guarded_policy_step(-out['flat_logits'][3]*100.,[row],[state],[row['decision_info']['run_metadata']],[mask])['accepted']
    after=agent._optimizer.state_dict()
    assert after['param_groups']==optimizer_before['param_groups']
    for key,state_before in optimizer_before['state'].items():
        for field,value in state_before.items():
            if torch.is_tensor(value):assert torch.equal(value,after['state'][key][field])
            else:assert value==after['state'][key][field]
    for key,value in before.items():assert torch.equal(value,agent._network.state_dict()[key])
