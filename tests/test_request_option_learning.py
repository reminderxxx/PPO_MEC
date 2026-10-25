from copy import deepcopy
import pytest
import torch
from tests.test_ready_workflow import fixture
from src.encoders.request_option_encoder import option_features,batch_option_features
from src.agents.ppo_agent import RequestOptionPPOAgent,ReadyWorkflowPPOAgent
from scripts.run_request_option_learning import execute_request


def test_option_holding_time_and_reward_are_exact_native_sum():
    env,state=fixture();state,reward,done,trace,reason=execute_request(env,state,'r0')
    assert done and reason=='terminal' and len(trace)==8
    assert reward==pytest.approx(sum(r['stream']['reward'] for r in trace))
    assert all(r['node_id'].startswith('r0:') and r['node_id'] in r['state']['ready_node_ids'] for r in trace)


def test_option_features_and_policy_ignore_request_list_order():
    _,state=fixture(count=2);changed=deepcopy(state);changed['request_stream']['requests'].reverse()
    a=option_features(state);b=option_features(changed)
    agent=RequestOptionPPOAgent(7);x,v=agent.network(batch_option_features([a]));y,w=agent.network(batch_option_features([b]))
    assert a['x'].shape==(2,17) and a['ids']==list(reversed(b['ids']))
    assert torch.allclose(x,y.flip(-1),atol=1e-6) and torch.allclose(v,w,atol=1e-6)
    agent.network.zero_grad();v.square().mean().backward()
    assert all(p.grad is None for p in agent.network.actor_parameters())


def test_option_ppo_update_and_checkpoint_cannot_use_node_contract(tmp_path):
    _,state=fixture(count=2)
    state['workflow']['nodes'][-1]['required_adapter']='adapter_control'
    features=option_features(state);agent=RequestOptionPPOAgent(7)
    _,lp,v=agent.act(features);before={k:p.clone() for k,p in agent.network.state_dict().items()}
    logits,_=agent.network(batch_option_features([features]));dist=torch.distributions.Categorical(logits=logits[0])
    result=agent.learn([dict(features=features,action=i,log_prob=float(dist.log_prob(torch.tensor(i)).detach()),advantage=1. if i==0 else -1.,value=v,**{'return':1.}) for i in range(2)])
    assert result['rows']==2 and any(not torch.equal(p,before[k]) for k,p in agent.network.state_dict().items() if k.startswith('actor'))
    path=tmp_path/'option.pt';agent.save(path,{'test':True});restored,p=RequestOptionPPOAgent.load(path)
    assert all(torch.equal(v,restored.network.state_dict()[k]) for k,v in agent.network.state_dict().items())
    with pytest.raises(ValueError,match='contract'):ReadyWorkflowPPOAgent.load(path)
