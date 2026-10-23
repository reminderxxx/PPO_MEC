"""Matched witnesses, including actual zero-label gradients and option aliases."""
import copy
import json
import math
import random
import numpy as np
import pytest
import torch
from tests.test_sagh_execution_repair import agent, state, candidate, assert_same_state
from sagh_execution_diagnostics import enumerate_executed_distribution
from src.agents.registry import build_agent
from src.envs.core.vec_workflow_core_env import make_toy_vec_env
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.metrics.recorder import EpisodeRecorder
from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer


def aligned(**kw):
    return agent(auxiliary_action_contract='executed_marginal_v1', **kw)


def row_at(a, info, action=0):
    return {'action': action, 'decision_info': copy.deepcopy(info), 'action_info': {
        'final_env_action': action, 'execution_mapping_state': {
            'idle_popularity_adapter_counts': dict(a._idle_popularity_adapter_counts)}}}


def test_positive_zero_annotation_loss_gradient_and_adam(monkeypatch):
    a=aligned(advantage_weighted_behavior_regularization_enabled=True,advantage_weighted_behavior_coef=.24)
    info=state();r=row_at(a,info)
    monkeypatch.setattr('src.agents.sa_ghmappo_core.PopularityCacheHeuristicAgent.act',lambda *x,**k:(3,{}))
    a._annotate_advantage_weighted_behavior_targets([r],advantage_values=np.array([2.]))
    assert r['advantage_weighted_behavior_applied'] and r['advantage_weighted_behavior_target_action']==0
    o=a._forward_policy(info['semantic_state'],run_metadata=info['run_metadata'])
    loss=a._compute_advantage_weighted_behavior_loss(batch_outputs=[o],batch_action_masks=[info['action_mask']],batch_rows=[r])
    expected=-a._executed_auxiliary_statistics(o,info['action_mask'],r)['log_probs'][0]*r['advantage_weighted_behavior_weight']
    assert loss.item()==pytest.approx(expected.item()) and loss.item()>0
    before=copy.deepcopy(a._network.state_dict());a._optimizer.zero_grad();loss.backward()
    assert sum(float(p.grad.square().sum()) for p in a._network.parameters() if p.grad is not None)>0
    a._optimizer.step()
    assert any(not torch.equal(v,a._network.state_dict()[k]) for k,v in before.items())
    assert r['auxiliary_execution_diagnostics']['behavior']['target_action']==0


def test_zero_target_in_real_learner_witness(monkeypatch):
    """Constructed toy teacher witness, separate from untouched real-data preflight."""
    a=aligned(train_epochs=1,batch_size=8,objective_diagnostics_enabled=True,
        advantage_weighted_behavior_regularization_enabled=True,advantage_weighted_behavior_coef=.24)
    def zero(output, deterministic, action_mask):
        selected=a._head_targets_for_env_action(0)
        lp,en,pr=a._selected_action_statistics(policy_output=output,selected_actions=selected,action_mask=action_mask)
        return selected,lp,en,pr,{'raw_env_action':0,'raw_head_actions':selected,'projection_applied':False}
    monkeypatch.setattr(a,'_sample_actions',zero)
    monkeypatch.setattr('src.agents.sa_ghmappo_core.PopularityCacheHeuristicAgent.act',lambda *x,**k:(3,{}))
    rec=EpisodeRecorder();env=GymVecEnv(core_env=make_toy_vec_env(),recorder=rec)
    _,rr=MARLOnPolicyTrainer(env=env,agent=a,recorder=rec,max_steps=6).collect_episode(run_metadata={'window_class':'idle_or_sparse'})
    result=a.learn(rr)
    rows=result['objective_diagnostics']['credit_rows']
    hits=[r for r in rows if r['behavior_supervision'].get('advantage_weighted_behavior_applied') and r['behavior_supervision'].get('advantage_weighted_behavior_target_action')==0]
    assert hits and result['accepted_optimizer_steps']>0
    grad=result['objective_diagnostics']['gradient_batches'][0]
    assert grad['terms']['behavior_regularizer']['gradient_norm']['encoder']>0


def install_nontrivial(a,monkeypatch):
    def candidates(**kw):
        c=candidate();c['prior_target']=1
        c['option_actions']={0:kw['base_env_action'],1:0,2:4,3:3}
        return c
    monkeypatch.setattr(a,'_build_option_gate_candidates',candidates)
    monkeypatch.setattr(a,'_should_apply_net_utility_option_termination',lambda **k:False)
    monkeypatch.setattr(a,'_dag_aware_option_termination_reason',lambda **k:None)
    info=state();info['action_mask']=[True]*5
    o=a._forward_policy(info['semantic_state'],run_metadata=info['run_metadata'])
    o['option_logits']=torch.tensor([.1,.35,.4,.05],requires_grad=True)
    return info,o


def test_nontrivial_remap_alias_mass_and_gradient_match_actual_act(monkeypatch):
    a=aligned(option_gate_enabled=True,option_gate_context_prior_enabled=True,
        option_gate_mechanism_preserve_enabled=False,option_gate_prior_logit_bias=0.,option_gate_deterministic_prior_margin=.22)
    info,o=install_nontrivial(a,monkeypatch);r=row_at(a,info)
    stats=a._executed_auxiliary_statistics(o,info['action_mask'],r)
    assert any(p['sampled_option']==2 and p['executed_option']==1 and p['action']==0 for p in stats['paths'])
    monkeypatch.setattr(a,'_forward_policy',lambda *x,**k:o)
    independent=enumerate_executed_distribution(a,info)
    assert stats['log_probs'].exp().detach().tolist()==pytest.approx(independent['executed_action_probs'],abs=2e-6)
    base=torch.softmax(a._hierarchical_env_action_scores(o),-1);option=torch.softmax(o['option_logits'],-1)
    expected=base[0]*option[0]+option[1]+option[2]
    assert stats['log_probs'][0].exp().item()==pytest.approx(expected.item(),abs=1e-6)
    actual=torch.autograd.grad(-stats['log_probs'][0],o['option_logits'],retain_graph=True)[0]
    expected_grad=torch.autograd.grad(-expected.log(),o['option_logits'])[0]
    assert torch.allclose(actual,expected_grad,atol=2e-6)
    assert actual[1]<0 and actual[2]<0  # Both latent preimages get credit.


def test_unreachable_behavior_has_zero_gradient_and_keeps_denominator(monkeypatch):
    a=aligned(advantage_weighted_behavior_regularization_enabled=True,advantage_weighted_behavior_coef=.24,
        idle_popularity_fallback_enabled=True)
    monkeypatch.setattr(a,'_idle_popularity_candidate_action',lambda *x:(4,'fixture',{}))
    info=state();info['action_mask']=[True]*5
    rows=[row_at(a,info,2),row_at(a,info,4)]
    for r,target in zip(rows,[2,4]):r.update(advantage_weighted_behavior_applied=True,advantage_weighted_behavior_target_action=target,advantage_weighted_behavior_weight=1.)
    outputs=[a._forward_policy(info['semantic_state'],run_metadata=info['run_metadata']) for _ in rows]
    loss=a._compute_advantage_weighted_behavior_loss(batch_outputs=outputs,batch_action_masks=[info['action_mask']]*2,batch_rows=rows)
    assert not rows[0]['auxiliary_execution_diagnostics']['behavior']['target_reachable']
    expected=-a._executed_auxiliary_statistics(outputs[1],info['action_mask'],rows[1])['log_probs'][4]/2
    assert loss.item()==pytest.approx(expected.item())
    zero=a._auxiliary_action_nll(outputs[0],info['action_mask'],rows[0],2,'behavior')
    grads=torch.autograd.grad(zero,[p for p in a._network.parameters() if p.requires_grad],allow_unused=True)
    assert all(g is None or torch.count_nonzero(g)==0 for g in grads)


def test_counterfactual_ranks_only_executable_targets_and_preserves_current_zero(monkeypatch):
    a=aligned(idle_popularity_fallback_enabled=True,env_action_counterfactual_margin_enabled=True,
        env_action_counterfactual_margin_coef=.5,counterfactual_teacher_prd_enabled=True)
    monkeypatch.setattr(a,'_idle_popularity_candidate_action',lambda *x:(4,'fixture',{}))
    called=[]
    def credit(row,action):called.append(action);return {2:10.,4:2.}.get(action,.1)
    monkeypatch.setattr(a,'_counterfactual_teacher_action_credit',credit)
    monkeypatch.setattr(a,'_row_sparse_recovery_context',lambda *x,**k:False)
    monkeypatch.setattr(a,'_env_action_recovery_risk_score',lambda *x,**k:0.)
    info=state();info['action_mask']=[True]*5;r=row_at(a,info,0)
    o=a._forward_policy(info['semantic_state'],run_metadata=info['run_metadata'])
    loss=a._compute_env_action_counterfactual_margin_loss(batch_outputs=[o],batch_action_masks=[info['action_mask']],batch_rows=[r])
    assert 2 not in called and called[-1]==0 and loss.item()>0
    d=r['auxiliary_execution_diagnostics'];assert d['counterfactual']['target_action']==4
    assert d['counterfactual_candidates']['current_action']==0


def test_final_guard_collapse_is_zero_auxiliary_gradient(monkeypatch):
    a=aligned()
    monkeypatch.setattr(a,'_apply_coverage_recovery_final_guard_to_env_action',lambda **k:{'enabled':True,'guarded':True,'guarded_action':3,'reason':'fixture'})
    info=state();r=row_at(a,info)
    o=a._forward_policy(info['semantic_state'],run_metadata=info['run_metadata'])
    stats=a._executed_auxiliary_statistics(o,info['action_mask'],r)
    assert stats['reachable']==[False,False,False,True,False]
    loss=a._auxiliary_action_nll(o,info['action_mask'],r,3,'behavior')
    assert abs(loss.item())<2e-6
    grads=torch.autograd.grad(loss,[p for p in a._network.parameters() if p.requires_grad],allow_unused=True)
    assert max([float(g.abs().max()) for g in grads if g is not None])<2e-6


def test_enumeration_is_passive_for_all_mapper_memory_and_rng():
    a=aligned(option_gate_enabled=True,idle_popularity_fallback_enabled=True)
    a._idle_popularity_adapter_counts={'adapter_batch_type_1':5}
    info=state();r=row_at(a,info);o=a._forward_policy(info['semantic_state'],run_metadata=info['run_metadata'])
    memory=copy.deepcopy(a._idle_popularity_adapter_counts);rng=torch.random.get_rng_state().clone();npstate=copy.deepcopy(np.random.get_state());py=random.getstate()
    a._executed_auxiliary_statistics(o,info['action_mask'],r)
    enumerate_executed_distribution(a,info)
    assert memory==a._idle_popularity_adapter_counts
    assert torch.equal(rng,torch.random.get_rng_state()) and py==random.getstate()
    assert np.array_equal(npstate[1],np.random.get_state()[1])


def test_missing_mapper_state_and_wrong_contract_fail_closed():
    with pytest.raises(ValueError,match='requires shared'):build_agent('sa_ghmappo',auxiliary_action_contract='executed_marginal_v1')
    a=aligned();info=state();o=a._forward_policy(info['semantic_state'],run_metadata=info['run_metadata'])
    with pytest.raises(ValueError,match='pre-decision'):a._executed_auxiliary_statistics(o,info['action_mask'],{'decision_info':info})


def test_execution_contract_checkpoint_and_inference_roundtrip(tmp_path):
    from src.evaluators.real_eval_support import build_inference_agent
    a=aligned();p=tmp_path/'aligned.pt';a.save(str(p))
    b=build_inference_agent('sa_ghmappo',checkpoint_path=p,random_seed=7)
    assert b._auxiliary_action_contract=='executed_marginal_v1'
    assert b._execution_mapping_contract=='shared_train_eval_v1'
    b.act(None,state())


@pytest.mark.parametrize('deterministic',[False,True])
def test_new_supervision_switch_does_not_change_act_before_training(deterministic):
    settings=dict(option_gate_enabled=True,idle_popularity_fallback_enabled=True)
    a=agent(**settings);b=aligned(**settings)
    info=state();info['deterministic_policy']=deterministic
    rng=torch.random.get_rng_state().clone()
    action,one=a.act(None,copy.deepcopy(info));torch.random.set_rng_state(rng)
    other,two=b.act(None,copy.deepcopy(info))
    assert action==other and one['log_prob']==two['log_prob']
    assert one['option_gate']==two['option_gate']
    assert_same_state(a._network.state_dict(),b._network.state_dict())


@pytest.mark.parametrize('deterministic',[False,True])
def test_production_refactor_matches_frozen_pre_alignment_act(deterministic):
    import subprocess,textwrap,types
    import src.agents.sa_ghmappo_core as core
    original=subprocess.check_output(['git','show','495da56:src/agents/sa_ghmappo_core.py'],text=True)
    start=original.index('    def act(',original.index('class 分层PPO基类'))
    end=original.index('\n    def ',start+10)
    namespace={};exec("from __future__ import annotations\n"+textwrap.dedent(original[start:end]),core.__dict__,namespace)
    settings=dict(option_gate_enabled=True,idle_popularity_fallback_enabled=True)
    a=agent(**settings);b=agent(**settings);b.act=types.MethodType(namespace['act'],b)
    info=state();info['deterministic_policy']=deterministic
    for step in range(3):
        rng=torch.random.get_rng_state().clone()
        actual,one=a.act(None,copy.deepcopy(info));torch.random.set_rng_state(rng)
        expected,two=b.act(None,copy.deepcopy(info))
        assert actual==expected
        for key in two:assert_same_state(one[key],two[key])
        assert a._idle_popularity_adapter_counts==b._idle_popularity_adapter_counts


def test_auxiliary_comparison_hard_caps_and_create_only(tmp_path):
    import preflight_sagh_auxiliary_alignment as pre
    assert pre.TRAIN_CAP==len(pre.VARIANTS)*len(pre.SEEDS)*pre.STEPS_PER_CELL==3072
    assert pre.VALIDATION_CAP==2112
    with pytest.raises(FileExistsError):pre.execute(tmp_path,tmp_path)
