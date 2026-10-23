"""DT density contract: constructed/recorded witnesses, no real-data rollout."""
import copy,io,types,subprocess,textwrap
import pytest
import torch
from tests.test_sagh_auxiliary_execution import aligned,row_at,install_nontrivial
from tests.test_sagh_execution_repair import state
from tests.test_sagh_value_encoder_routing import gradients

E='executed_marginal_v1'
def agent(**kw):
 return aligned(digital_twin_env_supervision_contract=E,digital_twin_policy_prior_enabled=True,
  digital_twin_policy_prior_distill_coef=.4,digital_twin_policy_prior_env_action_bias_enabled=True,
  objective_diagnostics_enabled=True,**kw)
def annotation(target=2):
 return dict(apply=True,strength=.8,event_target=0,slow_target=0,fast_target=1,pacing_target=True,env_target=target)
def loss(a,info,outputs,rows,annotations):
 return a._compute_digital_twin_policy_prior_loss(batch_outputs=outputs,batch_annotations=annotations,
  batch_advantage=torch.ones(len(rows)),batch_rows=rows,batch_action_masks=[info['action_mask']]*len(rows))

def test_unreachable_env_zero_retains_original_denominator_and_all_heads(monkeypatch):
 a=agent(idle_popularity_fallback_enabled=True)
 monkeypatch.setattr(a,'_idle_popularity_candidate_action',lambda *x:(4,'fixture',{}))
 info=state();info['action_mask']=[True]*5;rows=[row_at(a,info) for _ in range(2)]
 outputs=[a._forward_policy(info['semantic_state'],run_metadata=info['run_metadata']) for _ in rows]
 annotations=[annotation(2),annotation(4)];v=loss(a,info,outputs,rows,annotations);parts=a._digital_twin_loss_components
 assert rows[0]['auxiliary_execution_diagnostics']['digital_twin_env']['original_eligible_denominator']==2
 assert not rows[0]['auxiliary_execution_diagnostics']['digital_twin_env']['active']
 weight=rows[1]['auxiliary_execution_diagnostics']['digital_twin_env']['original_weight']
 expected=-a._executed_auxiliary_statistics(outputs[1],info['action_mask'],rows[1])['log_probs'][4]*a._digital_twin_policy_prior_continuation_prepare_scale*weight/2
 assert torch.allclose(parts['environment'],expected)
 zz=a._auxiliary_action_nll(outputs[0],info['action_mask'],rows[0],2,'witness')
 assert all(not g.count_nonzero() for g in gradients(a,zz).values())
 assert v.item()==pytest.approx((parts['environment']+parts['heads']).item())
 gh=gradients(a,parts['heads']);a._digital_twin_env_supervision_contract='proposal_v1'
 old=loss(a,info,outputs,rows,annotations);oldparts=a._digital_twin_loss_components
 assert all(torch.equal(gh[k],g) for k,g in gradients(a,oldparts['heads']).items())
 assert torch.equal(parts['fast_head'],oldparts['fast_head'])
 assert old>v

def test_zero_target_nontrivial_remap_and_probability_gradient(monkeypatch):
 a=agent(option_gate_enabled=True,option_gate_context_prior_enabled=True,
  option_gate_mechanism_preserve_enabled=False,option_gate_prior_logit_bias=0.,option_gate_deterministic_prior_margin=.22)
 info,o=install_nontrivial(a,monkeypatch);r=row_at(a,info)
 loss(a,info,[o],[r],[annotation(0)]);part=a._digital_twin_loss_components['environment']
 stat=a._executed_auxiliary_statistics(o,info['action_mask'],r)
 assert any(p['sampled_option']==2 and p['executed_option']==1 and p['action']==0 for p in stat['paths'])
 expected=-stat['log_probs'][0]*r['auxiliary_execution_diagnostics']['digital_twin_env']['original_weight']
 assert torch.equal(part,expected) and part.item()>0
 assert torch.allclose(torch.autograd.grad(part,o['option_logits'],retain_graph=True)[0],torch.autograd.grad(expected,o['option_logits'])[0])
 assert r['auxiliary_execution_diagnostics']['digital_twin_env']['target_action']==0

def test_alias_collapse_reachable_is_not_nonzero_gradient(monkeypatch):
 a=agent();monkeypatch.setattr(a,'_apply_coverage_recovery_final_guard_to_env_action',lambda **k:dict(enabled=True,guarded=True,guarded_action=3,reason='fixture'))
 info=state();r=row_at(a,info);o=a._forward_policy(info['semantic_state'],run_metadata=info['run_metadata'])
 loss(a,info,[o],[r],[annotation(3)])
 assert r['auxiliary_execution_diagnostics']['digital_twin_env']['active']
 assert abs(a._digital_twin_loss_components['environment'].item())<2e-6
 assert max(float(v.abs().max()) for v in gradients(a,a._digital_twin_loss_components['environment']).values())<2e-6

def test_proposal_arm_matches_frozen_original_every_parameter():
 import src.agents.sa_ghmappo_core as core
 a=agent();info=state();r=row_at(a,info);o=a._forward_policy(info['semantic_state'],run_metadata=info['run_metadata'])
 a._digital_twin_env_supervision_contract='proposal_v1';actual=loss(a,info,[o],[r],[annotation(2)])
 src=subprocess.check_output(['git','show','c12a27b:src/agents/sa_ghmappo_core.py'],text=True);start=src.index('    def _compute_digital_twin_policy_prior_loss(');end=src.index('\n    def ',start+10);ns={}
 exec('from __future__ import annotations\n'+textwrap.dedent(src[start:end]),core.__dict__,ns)
 original=types.MethodType(ns['_compute_digital_twin_policy_prior_loss'],a)
 expected=original(batch_outputs=[o],batch_annotations=[annotation(2)],batch_advantage=torch.ones(1))
 assert torch.equal(actual,expected)
 ga=gradients(a,actual);ge=gradients(a,expected);assert all(torch.equal(ga[k],v) for k,v in ge.items())

def test_contract_roundtrip_and_fail_closed(tmp_path):
 from src.evaluators.real_eval_support import build_inference_agent
 from src.agents.registry import build_agent
 a=agent();p=tmp_path/'dt_density.pt';a.save(str(p));b=build_inference_agent('sa_ghmappo',random_seed=7,checkpoint_path=p)
 assert b._digital_twin_env_supervision_contract==E
 with pytest.raises(ValueError,match='Unknown digital twin'):aligned(digital_twin_env_supervision_contract='typo')
 with pytest.raises(ValueError,match='requires shared'):build_agent('sa_ghmappo',digital_twin_env_supervision_contract=E)
 info=state();o=a._forward_policy(info['semantic_state'],run_metadata=info['run_metadata'])
 with pytest.raises(ValueError,match='matched rows'):a._compute_digital_twin_policy_prior_loss(batch_outputs=[o],batch_annotations=[annotation()],batch_advantage=torch.ones(1))

@pytest.mark.parametrize('deterministic',[False,True])
def test_density_switch_has_identical_inference_and_rng(deterministic):
 a=agent();b=agent();b._digital_twin_env_supervision_contract='proposal_v1';info=state();info['deterministic_policy']=deterministic
 rng=torch.random.get_rng_state().clone();one,oi=a.act(None,copy.deepcopy(info));torch.random.set_rng_state(rng);two,ti=b.act(None,copy.deepcopy(info))
 assert one==two and oi==ti

def test_hard_global_and_cell_budget_and_create_only(tmp_path):
 from preflight_sagh_dt_executed_density import InteractionBudget,execute,CONTRACTS
 assert CONTRACTS=={'D':'proposal_v1','E':E}
 b=InteractionBudget(io.StringIO())
 for cell in ['D/seed7','D/seed13','E/seed7','E/seed13']:
  b.cell=cell
  for phase,n in [('train',768),('validation',528)]:
   b.phase=phase
   for _ in range(n):b.charge()
 assert b.total==5184
 with pytest.raises(RuntimeError,match='hard cap'):b.charge()
 assert b.total==5184
 with pytest.raises(FileExistsError):execute(tmp_path,tmp_path)
