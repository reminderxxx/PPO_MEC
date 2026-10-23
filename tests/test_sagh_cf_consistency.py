"""Fixed tensor witnesses; no extra real-data rollouts for matching."""
import copy,math,subprocess,textwrap,io
import numpy as np
import pytest
import torch
from tests.test_sagh_auxiliary_execution import aligned,row_at
from tests.test_sagh_execution_repair import state,assert_same_state
import src.agents.sa_ghmappo_core as core

T='native_positive_executed_action_v1'
def fixture(monkeypatch,contract='none_v1'):
 a=aligned(counterfactual_consistency_contract=contract,env_action_counterfactual_margin_enabled=True,
     env_action_counterfactual_margin_coef=.5,counterfactual_teacher_prd_enabled=True,
     env_action_counterfactual_margin_max_weight=4.2,env_action_counterfactual_margin_min_gap=.1,
     env_action_counterfactual_margin_advantage_gate=0.)
 monkeypatch.setattr(a,'_counterfactual_teacher_action_credit',lambda row,action:2. if action==row['fixture_target'] else .25)
 monkeypatch.setattr(a,'_row_sparse_recovery_context',lambda *x:False)
 monkeypatch.setattr(a,'_env_action_recovery_risk_score',lambda *x:0.)
 info=state();info['action_mask']=[True]*5
 return a,info
def rows_and_outputs(a,info,spec):
 rows=[];outputs=[]
 for target,current,adv in spec:
  r=row_at(a,info,current);r['fixture_target']=target;r['cf_native_normalized_advantage']=adv;rows.append(r)
  outputs.append(a._forward_policy(info['semantic_state'],run_metadata=info['run_metadata']))
 return rows,outputs
def loss(a,info,rows,outputs):
 return a._compute_env_action_counterfactual_margin_loss(batch_outputs=outputs,batch_action_masks=[info['action_mask']]*len(rows),batch_rows=rows,
     base_advantage=torch.ones(len(rows)),event_advantage=torch.ones(len(rows)))
def gradients(a,l):
 ps=[p for p in a._network.parameters() if p.requires_grad];g=torch.autograd.grad(l,ps,retain_graph=True,allow_unused=True)
 return [torch.zeros_like(p) if v is None else v for p,v in zip(ps,g)]

@pytest.mark.parametrize('adv,current,target,paused',[(1.,4,3,True),(0.,4,3,False),(-1.,4,3,False),(1.,3,3,False),(1.,3,0,True),(1.,0,0,False)])
def test_gate_uses_native_sign_and_zero_preserving_final_action(monkeypatch,adv,current,target,paused):
 a,info=fixture(monkeypatch,T);rows,outs=rows_and_outputs(a,info,[(target,current,adv)]);l=loss(a,info,rows,outs)
 d=rows[0]['auxiliary_execution_diagnostics']['counterfactual_consistency'];assert d['paused']==paused
 assert d['target_action']==target and d['current_executed_action']==current
 assert d['original_eligible_denominator']==1
 gs=gradients(a,l);assert (max(float(g.abs().max()) for g in gs)==0)==paused
 assert a._env_action_counterfactual_margin_coef==.5 and a._env_action_counterfactual_margin_max_weight==4.2

def test_keeps_original_denominator_and_gradient_including_paused(monkeypatch):
 a,info=fixture(monkeypatch,T);rows,outs=rows_and_outputs(a,info,[(3,4,1.),(0,0,-1.)]);l=loss(a,info,rows,outs)
 active=loss(a,info,[rows[1]],[outs[1]])
 assert torch.allclose(l,active/2)
 for actual,want in zip(gradients(a,l),gradients(a,active)):assert torch.allclose(actual,want/2,atol=1e-7)
 assert rows[0]['cf_supervision_coverage']['paused_evaluations']==1
 assert rows[0]['cf_supervision_coverage']['active_weight_sum']==0

def test_disabled_control_matches_frozen_loss_and_all_gradients(monkeypatch):
 a,info=fixture(monkeypatch);rows,outs=rows_and_outputs(a,info,[(3,4,1.),(0,0,-1.)]);new=loss(a,info,rows,outs)
 original=subprocess.check_output(['git','show','9cb156a:src/agents/sa_ghmappo_core.py'],text=True)
 st=original.index('    def _compute_env_action_counterfactual_margin_loss(');en=original.index('\n    def ',st+8)
 namespace={};exec('from __future__ import annotations\n'+textwrap.dedent(original[st:en]),core.__dict__,namespace)
 old=namespace['_compute_env_action_counterfactual_margin_loss'](a,batch_outputs=outs,batch_action_masks=[info['action_mask']]*2,batch_rows=rows,base_advantage=torch.ones(2),event_advantage=torch.ones(2))
 assert torch.equal(new,old)
 for g,h in zip(gradients(a,new),gradients(a,old)):assert torch.equal(g,h)

def test_missing_native_advantage_fails_closed(monkeypatch):
 a,info=fixture(monkeypatch,T);rows,outs=rows_and_outputs(a,info,[(3,4,1.)]);del rows[0]['cf_native_normalized_advantage']
 with pytest.raises(ValueError,match='finite frozen native'):loss(a,info,rows,outs)

def test_native_advantage_not_credited_advantage(monkeypatch):
 a,info=fixture(monkeypatch,T);rows,outs=rows_and_outputs(a,info,[(3,4,-1.)]);l=loss(a,info,rows,outs)
 assert l.item()>0 and not rows[0]['auxiliary_execution_diagnostics']['counterfactual_consistency']['paused']

def test_checkpoint_inference_roundtrip_and_no_act_change(monkeypatch,tmp_path):
 from src.evaluators.real_eval_support import build_inference_agent
 a,info=fixture(monkeypatch);b,_=fixture(monkeypatch,T);assert_same_state(a._network.state_dict(),b._network.state_dict())
 before=torch.random.get_rng_state().clone();one,_=a.act(None,copy.deepcopy(info));torch.random.set_rng_state(before);two,_=b.act(None,copy.deepcopy(info));assert one==two
 p=tmp_path/'t.pt';b.save(str(p));c=build_inference_agent('sa_ghmappo',random_seed=7,checkpoint_path=p)
 assert c._counterfactual_consistency_contract==T

def test_contract_validation_and_exact_pre_step_budget(monkeypatch,tmp_path):
 from src.agents.registry import build_agent
 from preflight_sagh_cf_consistency import InteractionBudget,TOTAL_CAP,TRAIN_CAP,VALIDATION_CAP,execute
 with pytest.raises(ValueError,match='Unknown counterfactual consistency'):aligned(counterfactual_consistency_contract='typo')
 with pytest.raises(ValueError,match='requires executed'):build_agent('sa_ghmappo',counterfactual_consistency_contract=T)
 assert TOTAL_CAP==5184 and TRAIN_CAP==3072 and VALIDATION_CAP==2112
 budget=InteractionBudget(io.StringIO());budget.cell='C/seed7';budget.phase='train';budget.counts['train']=3072
 with pytest.raises(RuntimeError,match='hard cap'):budget.charge()
 assert budget.total==0
 with pytest.raises(FileExistsError):execute(tmp_path,tmp_path)

def test_budget_accepts_exact_5184_calls_and_rejects_5185th():
 from preflight_sagh_cf_consistency import InteractionBudget
 b=InteractionBudget(io.StringIO())
 for cell in ['C/seed7','C/seed13','T/seed7','T/seed13']:
  b.cell=cell
  for phase,limit in [('train',768),('validation',528)]:
   b.phase=phase
   for _ in range(limit):b.charge()
 assert b.total==5184 and b.counts=={'train':3072,'validation':2112}
 with pytest.raises(RuntimeError,match='hard cap'):b.charge()
 assert b.total==5184

def test_treatment_learner_freezes_native_before_credits(monkeypatch):
 from src.envs.core.vec_workflow_core_env import make_toy_vec_env
 from src.envs.wrappers.gym_vec_env import GymVecEnv
 from src.metrics.recorder import EpisodeRecorder
 from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer
 a=aligned(counterfactual_consistency_contract=T,objective_diagnostics_enabled=True,train_epochs=1,batch_size=8,
     counterfactual_teacher_prd_enabled=True,env_action_counterfactual_margin_enabled=True,env_action_counterfactual_margin_coef=.5)
 monkeypatch.setattr(a,'_counterfactual_teacher_action_credit',lambda row,action:2. if action==3 else .25)
 rec=EpisodeRecorder();env=GymVecEnv(core_env=make_toy_vec_env(),recorder=rec)
 _,rows=MARLOnPolicyTrainer(env=env,agent=a,recorder=rec,max_steps=6).collect_episode(run_metadata={'window_class':'idle_or_sparse'})
 result=a.learn(rows);diag=result['objective_diagnostics'];assert diag['counterfactual_consistency_contract']==T
 for r,c in zip(rows,diag['credit_rows']):assert r['cf_native_normalized_advantage']==c['native_normalized_advantage']
 checked=[x['diagnostics']['counterfactual_consistency'] for x in diag['auxiliary_execution_last_seen_rows'] if 'counterfactual_consistency' in x['diagnostics']]
 assert checked and all(math.isfinite(x['native_normalized_advantage']) for x in checked)
