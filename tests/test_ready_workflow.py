from copy import deepcopy
from pathlib import Path
import pytest
import torch
from scripts.run_ready_workflow_learning import make_env, rule_action
from src.envs.specs import WorkflowGraphState,WorkflowNode
from src.encoders.ready_workflow_encoder import ready_features,batch_ready_features,ReadyWorkflowActorCritic
from src.agents.ppo_agent import ReadyWorkflowPPOAgent


def fixture(swapped=False,count=1):
    edges=[('b' if swapped else 'a','c'),('a' if swapped else 'b','d'),('d','e')]
    nodes=[WorkflowNode(n,n,'veh_base_v1','adapter_perception',1,1,[a for a,b in edges if b==n],[b for a,b in edges if a==n]) for n in 'abcde']
    workflow=WorkflowGraphState('edge_test',nodes,edges,list('abcde'),current_node_id='a')
    unit=dict(subject_id='fixed',frames=[dict(time_index=i*100,vehicles=[dict(vehicle_id='fixed',position_x=0.,position_y=0.,speed=0.,base_model_id='veh_base_v1')]) for i in range(10)],rsus=[dict(rsu_id='rsu_a',position_x=0.,position_y=0.,coverage_radius=100.)])
    env=make_env(unit,[workflow],count=count,horizon=8)
    for n in env._workflow_template.nodes:n.required_adapter={'a':'adapter_perception','b':'adapter_perception','c':'adapter_fusion','d':'adapter_tracking','e':'adapter_tracking'}[n.node_id.split(':')[1]]
    state,_=env.reset();return env,state


def test_dependencies_are_enforced_without_silent_action_substitution():
    env,state=fixture();assert state['ready_node_ids']==['r0:a','r0:b']
    with pytest.raises(ValueError,match='dependency-ready'):env.step_node('r0:c')
    state,_,_,_,info=env.step_node('r0:a')
    assert 'r0:c' in state['ready_node_ids'] and 'r0:a' not in state['ready_node_ids']
    assert info['ready_scheduling']['selected_node_id']==info['persistent_stream']['cache_event']['node_id']=='r0:a'
    with pytest.raises(ValueError):env.step_node('r0:a')


def test_loading_can_overlap_other_ready_service_and_original_reward_is_preserved():
    env,state=fixture();returns=0.
    for node in ['a','c','b','d','e','c','c','c']:
        state,reward,done,_,info=env.step_node('r0:'+node);returns+=reward.total
        assert reward.total==pytest.approx(sum(info['persistent_stream']['reward_terms'].values()))
        assert all(r['resident_mb']<=r['capacity_mb']+1e-8 for r in state['cache_runtime'])
    assert done and info['persistent_stream']['completed_requests']==1
    assert returns==pytest.approx(4.184375)
    with pytest.raises(RuntimeError,match='Terminal'):env.step_node('r0:c')


def test_edge_swap_preserves_local_features_but_changes_directed_messages():
    _,a=fixture();_,b=fixture(True);fa,fb=ready_features(a),ready_features(b)
    assert torch.equal(fa['x'],fb['x']) and torch.equal(fa['mask'],fb['mask'])
    assert not torch.equal(fa['succ'],fb['succ'])
    torch.manual_seed(7);graph=ReadyWorkflowActorCritic(True)
    with torch.no_grad():graph.message_gate.fill_(1.)
    flat=deepcopy(graph);flat.use_graph=False
    ga,_=graph(batch_ready_features([fa]));gb,_=graph(batch_ready_features([fb]))
    sa,_=flat(batch_ready_features([fa]));sb,_=flat(batch_ready_features([fb]))
    assert torch.equal(sa,sb) and sa[0,0]==sa[0,1]
    assert not torch.equal(ga,gb) and float(ga[0,0].detach())==pytest.approx(float(gb[0,1].detach()),abs=1e-6)


def test_candidate_policy_is_equivariant_to_node_storage_order():
    _,s=fixture();t=deepcopy(s);t['workflow']['nodes']=list(reversed(t['workflow']['nodes']))
    a,b=ready_features(s),ready_features(t);net=ReadyWorkflowActorCritic()
    x,v=net(batch_ready_features([a]));y,w=net(batch_ready_features([b]))
    assert torch.allclose(x,y.flip(-1),atol=1e-6) and torch.allclose(v,w,atol=1e-6)


@pytest.mark.parametrize('graph',[True,False])
def test_actual_ppo_update_and_checkpoint_roundtrip(tmp_path,graph):
    a=ReadyWorkflowPPOAgent(7,graph);e,s=fixture();rows=[]
    while True:
        f=ready_features(s);act,lp,v=a.act(f);s,r,done,_,_=e.step_node(f['ids'][act])
        rows.append(dict(features=f,action=act,log_prob=lp,value=v,reward=r.total/20.,advantage=r.total/20.-v,return_=r.total/20.))
        if done:break
    for row in rows:row['return']=row.pop('return_')
    before={k:v.clone() for k,v in a.network.state_dict().items()};result=a.learn(rows)
    assert result['rows']==8 and any(not torch.equal(v,before[k]) for k,v in a.network.state_dict().items())
    path=tmp_path/'checkpoint.pt';a.save(path,{'test':True});b,p=ReadyWorkflowPPOAgent.load(path)
    assert all(torch.equal(v,b.network.state_dict()[k]) for k,v in a.network.state_dict().items())
    assert p['provenance']['test'] and b.updates==1
    with pytest.raises(FileExistsError):a.save(path,{})
    p['contract']='released_fifo_prepare_5_v1';torch.save(p,tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='contract'):ReadyWorkflowPPOAgent.load(tmp_path/'bad.pt')


def test_graph_set_initial_parameters_and_information_match():
    graph=ReadyWorkflowPPOAgent(13,True);flat=ReadyWorkflowPPOAgent(13,False)
    assert all(torch.equal(v,flat.network.state_dict()[k]) for k,v in graph.network.state_dict().items())


def test_zero_gate_starts_graph_and_set_at_exactly_same_policy():
    _,state=fixture();features=batch_ready_features([ready_features(state)])
    graph=ReadyWorkflowPPOAgent(7,True);flat=ReadyWorkflowPPOAgent(7,False)
    g,v=graph.network(features);s,w=flat.network(features)
    assert torch.equal(g,s) and torch.equal(v,w)


def test_legacy_pilot_checkpoint_remains_loadable(tmp_path):
    a=ReadyWorkflowPPOAgent(7,True,architecture='ready_propagated_actor_v1')
    p=tmp_path/'v1.pt';a.save(p,{})
    payload=torch.load(p);payload.pop('architecture');torch.save(payload,p)
    b,_=ReadyWorkflowPPOAgent.load(p)
    assert b.network.architecture=='ready_propagated_actor_v1'
    assert all(torch.equal(v,b.network.state_dict()[k]) for k,v in a.network.state_dict().items())


def test_value_loss_cannot_modify_actor_representation_in_detached_variant():
    from src.encoders.ready_workflow_encoder import MODEL_VERSION, RESIDUAL_MODEL_VERSION
    _,s=fixture();features=batch_ready_features([ready_features(s)])
    for architecture,should_receive in [(MODEL_VERSION,False),(RESIDUAL_MODEL_VERSION,True)]:
        a=ReadyWorkflowPPOAgent(7,True,architecture)
        logits,value=a.network(features);value.square().mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in a.network.critic.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in a.network.node.parameters())==should_receive
        a.network.zero_grad();logits,_=a.network(features)
        loss=-torch.distributions.Categorical(logits=logits).log_prob(torch.tensor([0])).mean();loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in a.network.node.parameters())


def test_v2_boundary_weights_upgrade_without_policy_change():
    from src.encoders.ready_workflow_encoder import MODEL_VERSION, RESIDUAL_MODEL_VERSION
    _,s=fixture();f=batch_ready_features([ready_features(s)])
    a=ReadyWorkflowPPOAgent(7,True,RESIDUAL_MODEL_VERSION);x,v=a.network(f)
    a.network.architecture=MODEL_VERSION;y,w=a.network(f)
    assert torch.equal(x,y) and torch.equal(v,w)
