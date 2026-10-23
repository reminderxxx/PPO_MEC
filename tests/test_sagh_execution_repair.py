"""Zero-action semantics, common execution mapping, and passive diagnostics."""
import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from src.agents.registry import build_agent
from src.agents.sa_ghmappo_core import _action_id
from src.envs.core.vec_workflow_core_env import make_toy_vec_env
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.metrics.recorder import EpisodeRecorder
from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer
from sagh_execution_diagnostics import enumerate_executed_distribution, install_gate_observer, step_record


def agent(**kw):
    return build_agent('sa_ghmappo', device='cpu', execution_mapping_contract='shared_train_eval_v1', **kw)


def state():
    env=GymVecEnv(core_env=make_toy_vec_env())
    _, info=env.reset();info['run_metadata']={'window_class':'idle_or_sparse'}
    return info


@pytest.mark.parametrize('value', [0,1,2,3,4,None])
@pytest.mark.parametrize('default', [0,3,4])
def test_only_none_uses_action_default(value,default):
    assert _action_id(value,default)==(default if value is None else value)


def test_zero_behavior_target_survives_annotation_and_loss(monkeypatch):
    a=agent(advantage_weighted_behavior_regularization_enabled=True,advantage_weighted_behavior_coef=.24)
    info=state();row={'action':0,'action_info':{'final_env_action':0},'decision_info':info}
    monkeypatch.setattr('src.agents.sa_ghmappo_core.PopularityCacheHeuristicAgent.act',lambda *args,**kw:(3,{}))
    stats=a._annotate_advantage_weighted_behavior_targets([row],advantage_values=np.array([2.]))
    assert stats['positive_count']==1
    assert row['advantage_weighted_behavior_student_action']==0
    assert row['advantage_weighted_behavior_target_action']==0
    output=a._forward_policy(info['semantic_state'],run_metadata=info['run_metadata'])
    logits=a._masked_flat_logits(a._hierarchical_env_action_scores(output),info['action_mask'])
    loss=a._compute_advantage_weighted_behavior_loss(batch_outputs=[output],batch_action_masks=[info['action_mask']],batch_rows=[row])
    expected=torch.nn.functional.cross_entropy(logits.unsqueeze(0),torch.tensor([0]))*row['advantage_weighted_behavior_weight']
    assert loss.item()==pytest.approx(expected.item())


def test_counterfactual_current_zero_is_not_replaced_by_best(monkeypatch):
    a=agent(env_action_counterfactual_margin_enabled=True,env_action_counterfactual_margin_coef=.5,
            counterfactual_teacher_prd_enabled=True)
    calls=[]
    def teacher(row,action):calls.append(action);return 1. if action==4 else .1
    monkeypatch.setattr(a,'_counterfactual_teacher_action_credit',teacher)
    monkeypatch.setattr(a,'_row_sparse_recovery_context',lambda *x,**y:False)
    monkeypatch.setattr(a,'_env_action_recovery_risk_score',lambda *x,**y:0.)
    info=state();o=a._forward_policy(info['semantic_state'],run_metadata=info['run_metadata'])
    a._compute_env_action_counterfactual_margin_loss(batch_outputs=[o],batch_action_masks=[[True]*5],batch_rows=[{'action':0,'action_info':{'final_env_action':0}}])
    assert calls[-1]==0


@pytest.mark.parametrize('base',[0,1,2,3,4])
def test_fallback_mapping_equal_for_stochastic_and_greedy(monkeypatch,base):
    a=agent(idle_popularity_fallback_enabled=True)
    monkeypatch.setattr(a,'_idle_popularity_candidate_action',lambda *args:(4,'predicted_handoff_migration_prepare',{}))
    args=dict(semantic_state={},action_mask=[True]*5,original_env_action=base)
    assert a._maybe_apply_idle_popularity_fallback(**args,deterministic=False)==a._maybe_apply_idle_popularity_fallback(**args,deterministic=True)


def candidate():
    return dict(option_mask=[True]*4,prior_target=1,option_actions={0:3,1:4,2:3,3:4},
        popularity_reason='test',mechanism_reason='test',no_rsu_available=False,mechanism_available=True)


def test_option_post_selection_remap_keeps_sampled_log_probability(monkeypatch):
    a=agent(option_gate_enabled=True,option_gate_context_prior_enabled=True,option_gate_mechanism_preserve_enabled=False)
    monkeypatch.setattr(a,'_build_option_gate_candidates',lambda **kw:candidate())
    monkeypatch.setattr(a,'_dag_aware_option_termination_reason',lambda **kw:'fixture_forced_termination')
    monkeypatch.setattr(a,'_sample_shared_option',lambda d,det:torch.tensor(2))
    output={'option_logits':torch.tensor([.2,.1,.8,-.4],requires_grad=True)}
    args=dict(semantic_state={},action_mask=[True]*5,policy_output=output,base_env_action=3,
              run_metadata={'window_class':'active_non_mechanism'})
    x=a._maybe_apply_option_gate(**args,deterministic=False)
    y=a._maybe_apply_option_gate(**args,deterministic=True)
    assert x['option_action']==y['option_action']==2
    assert x['executed_option_action']==y['executed_option_action']==1
    assert x['option_env_action']==y['option_env_action']==4
    assert x['option_log_prob']==pytest.approx(math.log(x['option_probs'][2]))
    assert x['option_remapping']==y['option_remapping']
    assert x['execution_option_actions']=={'0':4,'1':4,'2':4,'3':4}


def test_context_margin_is_a_shared_mapping_not_a_second_sampler(monkeypatch):
    a=agent(option_gate_context_prior_enabled=True,option_gate_deterministic_prior_margin=.22)
    monkeypatch.setattr(a,'_should_apply_net_utility_option_termination',lambda **kw:False)
    monkeypatch.setattr(a,'_dag_aware_option_termination_reason',lambda **kw:None)
    mapping,_=a._shared_option_remapping(semantic_state={},candidate_info=candidate(),base_env_action=3,
        window_class='idle_or_sparse',option_probs=torch.tensor([.4,.35,.15,.1]),top_action=0)
    assert mapping=={0:1,1:1,2:2,3:3}


@pytest.mark.parametrize('setting',[{'deterministic_temporal_smoothing_enabled':True},
    {'option_counterfactual_policy_improvement_enabled':True}])
def test_unmatched_optional_features_fail_closed(setting):
    with pytest.raises(ValueError,match='Unsupported shared'):
        agent(**setting).act(None,state())


def test_empty_legal_mask_and_raw_bypass_refused():
    a=agent();info=state();info['action_mask']=[False]*5
    with pytest.raises(ValueError,match='nonempty'):a.act(None,info)
    info=state();info['run_metadata']['policy_evaluation_mode']='raw_policy'
    with pytest.raises(ValueError,match='Raw-policy'):a.act(None,info)


def test_exact_execution_marginal_and_observer_preserve_rng_and_parameters():
    a=install_gate_observer(agent(option_gate_enabled=True,idle_popularity_fallback_enabled=True))
    info=state();before=copy.deepcopy(a._network.state_dict())
    rng=torch.random.get_rng_state().clone();nprng=copy.deepcopy(np.random.get_state());pyrng=random.getstate()
    diagnostic=enumerate_executed_distribution(a,info)
    a._deterministic_action=True
    greedy=enumerate_executed_distribution(a,info)
    assert diagnostic['executed_action_probs']==pytest.approx(greedy['executed_action_probs'],abs=1e-7)
    assert sum(diagnostic['executed_action_probs'])==pytest.approx(1.,abs=2e-5)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert np.array_equal(nprng[1],np.random.get_state()[1]) and nprng[2:]==np.random.get_state()[2:]
    assert pyrng==random.getstate() and not a._optimizer.state
    assert all(torch.equal(v,a._network.state_dict()[k]) for k,v in before.items())
    assert '_sample_actions' not in a.__dict__ and '_sample_shared_option' not in a.__dict__


def test_native_reward_components_recorded_without_rewriting():
    recorder=EpisodeRecorder();a=install_gate_observer(agent())
    env=GymVecEnv(core_env=make_toy_vec_env(),recorder=recorder)
    _,rr=MARLOnPolicyTrainer(env=env,agent=a,recorder=recorder,max_steps=3).collect_episode(run_metadata={'window_class':'idle_or_sparse'})
    record=step_record(rr[0],{'phase':'test'})
    json.dumps(record,allow_nan=False)
    assert record['native_reward']==rr[0]['reward']
    assert record['stages'][0]['stage']=='network'
    bad=copy.deepcopy(rr[0]);bad['reward']+=1
    with pytest.raises(ValueError,match='ledger'):step_record(bad,{})


def assert_same_state(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_same_state(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_same_state(x,y)
    else:assert a==b


def test_gradient_diagnostics_do_not_change_update_or_adam_or_rng():
    torch.set_num_threads(1)
    settings=dict(train_epochs=1,batch_size=8,random_seed=13,auxiliary_coef=.22,
        env_action_ppo_enabled=True,env_action_ppo_coef=1.08,net_utility_prd_enabled=True,
        net_utility_cost_dual_enabled=True,net_utility_cost_dual_lr=.07,
        mechanism_credit_prd_enabled=True,mechanism_credit_policy_coef=.5,
        advantage_weighted_behavior_regularization_enabled=True,advantage_weighted_behavior_coef=.24,
        option_gate_enabled=True,idle_popularity_fallback_enabled=True)
    a=agent(**settings)
    b=agent(**settings,objective_diagnostics_enabled=True)
    recorder=EpisodeRecorder();env=GymVecEnv(core_env=make_toy_vec_env(),recorder=recorder)
    _,rollout=MARLOnPolicyTrainer(env=env,agent=a,recorder=recorder,max_steps=6).collect_episode(run_metadata={'window_class':'idle_or_sparse'})
    rng=copy.deepcopy(np.random.get_state());trng=torch.random.get_rng_state();prng=random.getstate()
    one=a.learn(copy.deepcopy(rollout));after=(np.random.get_state(),torch.random.get_rng_state(),random.getstate())
    np.random.set_state(rng);torch.random.set_rng_state(trng);random.setstate(prng)
    two=b.learn(copy.deepcopy(rollout))
    assert_same_state(a._network.state_dict(),b._network.state_dict())
    assert_same_state(a._optimizer.state_dict(),b._optimizer.state_dict())
    assert np.array_equal(after[0][1],np.random.get_state()[1])
    assert torch.equal(after[1],torch.random.get_rng_state()) and after[2]==random.getstate()
    assert one['actor_loss']==two['actor_loss']
    d=two['objective_diagnostics'];assert len(d['gradient_batches'])==1 and len(d['credit_rows'])==len(rollout)
    g=d['gradient_batches'][0];assert g['weighted_term_sum']==pytest.approx(g['total_loss'],rel=1e-5,abs=1e-5)
    assert 'encoder' in g['terms']['native_return_actor_probe']['gradient_norm']


def test_new_contract_checkpoint_roundtrip(tmp_path):
    a=agent(objective_diagnostics_enabled=True);p=tmp_path/'new.pt';a.save(str(p))
    payload=torch.load(p,map_location='cpu',weights_only=False)
    cfg=payload['config'];assert cfg['execution_mapping_contract']=='shared_train_eval_v1'
    b=build_agent('sa_ghmappo',**{k:v for k,v in cfg.items() if k not in {
        'agent_name','policy_type','encoder_kind','centralized_critic','hierarchical_conditioning','event_head_enabled','adapter_prefetch_enabled'}})
    b.load(str(p));assert b._execution_mapping_contract==a._execution_mapping_contract
    from src.evaluators.real_eval_support import _filter_checkpoint_config,build_inference_agent
    assert _filter_checkpoint_config('sa_ghmappo',cfg)['execution_mapping_contract']=='shared_train_eval_v1'
    inferred=build_inference_agent('sa_ghmappo',checkpoint_path=p,random_seed=7)
    assert inferred._execution_mapping_contract=='shared_train_eval_v1'
    inferred.act(None,state())


def test_legacy_checkpoint_contract_keeps_its_explicit_boundary(monkeypatch):
    a=build_agent('sa_ghmappo',idle_popularity_fallback_enabled=True)
    monkeypatch.setattr(a,'_idle_popularity_candidate_action',lambda *args:(4,'fixture',{}))
    r=a._maybe_apply_idle_popularity_fallback(semantic_state={},action_mask=[True]*5,original_env_action=2,deterministic=False)
    assert r['reason']=='non_deterministic_policy'


def test_forced_option_never_bypasses_mask(monkeypatch):
    a=agent();c=candidate();c['option_mask'][1]=False
    with pytest.raises(ValueError,match='masked'):
        a._shared_option_remapping(semantic_state={},candidate_info=c,base_env_action=3,window_class='idle_or_sparse',
            option_probs=torch.tensor([.4,0,.5,.1]),top_action=2,forced_selection_reason='forced')


def test_preflight_has_fixed_budget_and_refuses_existing_outputs(tmp_path):
    import preflight_sagh_execution_repair as preflight
    assert preflight.STEPS_PER_CELL==768
    assert len(preflight.AGENTS)*len(preflight.SEEDS)*preflight.STEPS_PER_CELL==6144
    sentinel=tmp_path/'original.txt';sentinel.write_text('unchanged')
    with pytest.raises(FileExistsError):preflight.execute(tmp_path,tmp_path)
    assert sentinel.read_text()=='unchanged' and not (tmp_path/'failure.json').exists()


@pytest.mark.parametrize('target,expected',[(0.,.01),(-1.,.01),(.0014,.0014)])
def test_preflight_checker_uses_unchanged_native_kl_limit(target,expected):
    from preflight_sagh_execution_repair import policy_kl_limit
    assert policy_kl_limit(target)==expected


def test_preflight_inheritance_refuses_changed_runtime_or_incomplete_cells(tmp_path):
    from preflight_sagh_execution_repair import reuse_completed_sagh
    (tmp_path/'execution_receipt.json').write_text(json.dumps({'parent_plan_sha256':'p','hard_total_interaction_cap':6144,
        'source_sha256':{'src/agents/sa_ghmappo_core.py':'changed'}}))
    with pytest.raises(ValueError,match='runtime changed'):
        reuse_completed_sagh(tmp_path,{'plan_sha256':'p'})
