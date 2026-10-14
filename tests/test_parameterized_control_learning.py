"""Probability, information boundary, checkpoint and causal predictor contracts."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from tests.test_multi_request_target_control_env import make_env
from src.envs.specs import RSUState
from src.agents.registry import build_agent
from src.envs.core.joint_learned_prediction import LearnedMobilityPredictor, ResidualMobilityGRU, prediction_samples


def four_site_env():
    env = make_env(count=2, nodes=3)
    # Rebuild with the same geometry plus two reachable sites.
    from src.envs.core.multi_request_target_control_env import MultiRequestTargetControlEnv
    return MultiRequestTargetControlEnv(frames=env.frames,vehicle_workflows=env.workflows,
        rsu_states=[*env.rsus, RSUState('c',0,10,20), RSUState('d',10,10,20)],
        model_cache_config={'capacity_bytes':20,'base_size_bytes':8,'adapter_size_bytes':2,'transfer_bytes_per_frame':8},
        workload_bytes_per_unit=2)


@pytest.mark.parametrize('name',['sa_ghmappo','mappo','cache_offload_drl'])
def test_joint_probability_update_and_exact_checkpoint(name,tmp_path):
    torch.set_num_threads(1)
    env=four_site_env(); agent=build_agent(name,control_contract=env.action_contract,seed=7)
    controls,roll=agent.act_controls(env.decision_states(),0)
    _,logs,entropy,values,_=agent.distributions(roll['data'],actions=roll['actions'])
    assert torch.allclose(logs,roll['logs'],atol=1e-6) and torch.isfinite(entropy).all()
    env.step_controls(controls)
    before={k:v.clone() for k,v in agent.state_dict().items()}
    roll['returns']=roll['values']+torch.tensor([1.,-1.])
    agent.update([roll],epochs=1)
    assert any(not torch.equal(v,agent.state_dict()[k]) for k,v in before.items())
    path=tmp_path/'checkpoint.pt'; agent.save_checkpoint(path)
    restored=type(agent).from_checkpoint(path)
    agent.reset_controller();restored.reset_controller()
    a,ar=agent.act_controls(env.decision_states(),1,deterministic=True)
    b,br=restored.act_controls(env.decision_states(),1,deterministic=True)
    assert a==b and torch.equal(ar['logs'],br['logs'])
    for _ in range(12):
        commands,_=agent.act_controls(env.decision_states(),env.frame_index)
        _,_,done,_=env.step_controls(commands)
        if done: break


def test_actor_does_not_read_centralized_truth_and_rejects_mask_violation():
    env=four_site_env(); agent=build_agent('sa_ghmappo',control_contract=env.action_contract)
    d=env.decision_states()['v0']; changed=deepcopy(d)
    for r in changed['centralized_critic_state']['rsus']:
        r['occupied_bytes']=123456
    a=agent._features(d,0)[0]; b=agent._features(changed,0)[0]
    assert all(np.array_equal(a[k],b[k]) for k in a if k!='critic')
    assert not np.array_equal(a['critic'],b['critic'])
    _,roll=agent.act_controls(env.decision_states(),0)
    bad=roll['actions'].clone();bad[:,0]=5
    with pytest.raises(ValueError,match='behavior mask'):
        agent.distributions(roll['data'],actions=bad)


def test_prediction_is_future_invariant_and_samples_stay_within_window():
    env=four_site_env(); model=ResidualMobilityGRU()
    bundle=dict(state_dict=model.state_dict(),scale=1.,horizon=3,history=8)
    x=LearnedMobilityPredictor(bundle); y=LearnedMobilityPredictor(bundle)
    left=deepcopy(env.frames);right=deepcopy(left)
    for frame in right[5:]:
        for vehicle in frame['vehicles']:vehicle.position_x+=100000
    for t in range(5):
        a=x.predict(left[t]['vehicles'],env.mapper.associate(left[t]['vehicles']),env.rsus,env.mapper,t)
        b=y.predict(right[t]['vehicles'],env.mapper.associate(right[t]['vehicles']),env.rsus,env.mapper,t)
        assert a==b
    old=deepcopy(x.last_result); assert x.predict(left[4]['vehicles'],{},env.rsus,env.mapper,4)==old
    x.reset(); assert not x.previous and not x.pending and x.last_step==-1
    samples,targets=prediction_samples([dict(frames=left[:12])])
    assert len(samples)==2 and samples.shape==(2,8,2) and targets.shape==(2,3,2)


def test_unknown_contract_and_legacy_checkpoint_rejected(tmp_path):
    with pytest.raises(ValueError,match='control contract'):
        build_agent('mappo',control_contract='invented')
    with pytest.raises(ValueError,match='no parameterized'):
        build_agent('dqn',control_contract='joint_parameterized_control_v3')
    agent=build_agent('mappo',control_contract='joint_parameterized_control_v3')
    path=tmp_path/'legacy.pt';torch.save({'state_dict':agent.state_dict()},path)
    with pytest.raises(ValueError,match='incompatible'):
        type(agent).from_checkpoint(path)


def test_registry_import_in_fresh_process():
    import subprocess, sys
    result = subprocess.run([sys.executable, '-c', 'from src.agents.registry import build_agent; build_agent("mappo", control_contract="joint_parameterized_control_v3")'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
