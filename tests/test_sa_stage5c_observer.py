"""Synthetic callbacks and source/probability fixtures. No real agent, network,
environment, backward or optimizer is constructed or executed.
"""
import ast,copy,json,math,subprocess,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import scripts.sa_stage5c_credit_observer as O
import scripts.sa_stage5c_credit_audit as A
ROOT=Path(__file__).resolve().parents[1]
BASE='5c73ace0230c39798b332d122be3b5156972b455'

def zero_delta():return {g:dict(l2=0.,max_abs=0.,parameter_count=0,changed_parameter_count=0) for g in ('policy','value','surrogate')}
def sink(path):return A.AuditSink(path,enable=True,header=dict(run_uid='fixture',source_commit=BASE,provenance=dict(source_run_id='SYNTHETIC_NO_TRAINING'),evidence_kind='synthetic_fixture'))
class SyntheticStep:
 def __init__(self):self.calls=0;self.fail=False;self.param_groups=[{'lr':.001}]
 def step(self):
  self.calls+=1
  if self.fail:raise LookupError('synthetic original callback failed after counter increment')
  return 'synthetic_result'
class ProtocolFixture:
 action_contract='joint_stream_control_v2';method='sa_ghmappo';surrogate_enabled=True
 def __init__(self):self.optimizer=SyntheticStep();self.surrogate_optimizer=SyntheticStep();self.completed_updates=0
 def act_controls(self,*a,**k):return {},None
 def _coordinated_features(self,*a,**k):return None
 def _evaluate(self,*a,**k):raise AssertionError('No evaluation is permitted in this fixture')
 def _choice_distribution(self,*a,**k):raise AssertionError('No categorical construction/sampling permitted')
 def update(self,*a,**k):raise AssertionError('No real/fake learning entry point is called')
 def named_parameters(self):return iter(())

def behavior_row(ob,frame=0):
 data=dict(own=[[0.]],f1=[[[1.]]],m1=[[True]],node1=[[0]])
 logs=[math.log(.5),0.,math.log(.5)];masks=[[True,True],[True],[True,True]];candidates=[['a','b'],[None],[[None,0],[None,1]]]
 f=dict(vehicle_id='v',node_id='n',decision=dict(semantic_state={'request_id':'r'}),candidates=candidates,context=dict(calendar={},reservations={},target_memory={},commitments={},request_memory={}))
 first=dict(feature=copy.deepcopy(f),input=copy.deepcopy(data),kwargs={'execution_only':True},result=[[[0]],[[logs[0]]]],distributions=[{'actual_mask':[masks[0]]}])
 second=dict(feature=copy.deepcopy(f),input=copy.deepcopy(data),kwargs={},result=[],distributions=[{'actual_mask':[x]} for x in masks])
 ob._calls=[first,second]
 r=dict(vehicle_ids=['v'],request_ids=['r'],actions=[[0,0,0]],head_logs=[logs],counterfactual_baselines=[[.1,.2,.3]],values=[.4],data=data,frame_index=frame,returns=[8.],initial_request_count=8,cost_targets=[[0.,0.]])
 with ob.session(dict(seed=17,session_id='s')):
  return ob._emit_behavior(r,{'v':dict(semantic_state={'request_id':'r'})},frame,{'v':{}})

def credit(ob,r):
 ob._begin_update([r]);ob.credit_inputs([r],r['actions'],r['head_logs'],[1.],[1.],[[True,False,True]],r['counterfactual_baselines'],[[.9,.8,.7]])
 ob.credit_normalized([[.9,.8,.7]]);ob.epoch_start();ob.minibatch_start([0],r['data']);ob.minibatch_scores([[0.,0.,0.]],[[0.,0.,0.]],[.4],[[[0.,0.,0.]]],[[0.,0.,0.]],[[True,False,True]],[[.1,.2,.3]],[[.1,.2,.3]],[[1.,1.,1.]],0.,[[1.,0.,1.]],2.)

class ObserverTests(unittest.TestCase):
 def patches(self):return mock.patch.multiple(O,parameter_snapshot=mock.Mock(return_value={}),parameter_delta=mock.Mock(side_effect=lambda *a:zero_delta()),postclip_gradients=mock.Mock(return_value={}),rng_digest=mock.Mock(return_value='0'*64))
 def test_disabled_never_reads_agent(self):
  class Poison:
   def __getattribute__(self,n):raise AssertionError(n)
  with O.credit_audit(Poison(),SimpleNamespace(enabled=False)) as observer:self.assertIsNone(observer)
 def test_failed_entry_never_leaks_private_binding(self):
  f=SimpleNamespace(action_contract='joint_stream_control_v2',method='sa_ghmappo')
  with tempfile.TemporaryDirectory() as d:
   s=sink(Path(d)/'a')
   with self.assertRaises(AttributeError):O.CreditAuditObserver(f,s).__enter__()
   self.assertFalse(hasattr(f,'_credit_audit'));s.finish()
 def test_context_restores_instance_lookup_on_success_and_error(self):
  for fail in (False,True):
   f=ProtocolFixture();before=dict(f.__dict__)
   with tempfile.TemporaryDirectory() as d:
    s=sink(Path(d)/'a')
    try:
     with O.credit_audit(f,s):
      self.assertTrue(hasattr(f,'_credit_audit'))
      if fail:raise LookupError('fixture')
    except LookupError:pass
    self.assertEqual(set(before),set(f.__dict__));self.assertNotIn('step',f.optimizer.__dict__)
    if not s.closed:s.finish()
 def test_greedy_and_missing_session_rejected_before_original_call(self):
  f=ProtocolFixture()
  with tempfile.TemporaryDirectory() as d:
   s=sink(Path(d)/'a')
   with O.credit_audit(f,s) as ob:
    with self.assertRaises(ValueError):f.act_controls({},0)
    with ob.session(dict(seed=17,session_id='s')):
     with self.assertRaises(ValueError):f.act_controls({},0,deterministic=True)
   s.finish()
 def test_behavior_head0_counted_once_and_detached_input(self):
  with tempfile.TemporaryDirectory() as d:
   s=sink(Path(d)/'a');ob=O.CreditAuditObserver(ProtocolFixture(),s);r=behavior_row(ob)
   self.assertEqual(len(r['credit_audit_uids']),1);self.assertEqual(s.joins.counts['behavior'],1)
   b=next(iter(s.joins.behaviors.values()));self.assertEqual(b['head_passes'],A.DEFAULT_HEAD_PASSES)
   self.assertEqual(b['masks'],[[True,True],[True],[True,True]]);s.finish()
 def test_identity_and_baseline_mutation_rejected(self):
  for field,value in [('frame_index',1),('vehicle_ids',['other']),('request_ids',['other']),('counterfactual_baselines',[[9.,9.,9.]])]:
   with self.subTest(field=field),tempfile.TemporaryDirectory() as d,self.patches():
    s=sink(Path(d)/'a');ob=O.CreditAuditObserver(ProtocolFixture(),s);r=behavior_row(ob);r[field]=value
    with self.assertRaises(ValueError):ob._begin_update([r])
    s.finish()
 def test_behavior_input_mutation_rejected(self):
  with tempfile.TemporaryDirectory() as d,self.patches():
   s=sink(Path(d)/'a');ob=O.CreditAuditObserver(ProtocolFixture(),s);r=behavior_row(ob);r['data']['own'][0][0]=7.
   with self.assertRaises(ValueError):ob._begin_update([r])
   s.finish()
 def test_kl_rejected_chain_has_no_step_and_retains_raw_forced_credit(self):
  with tempfile.TemporaryDirectory() as d,self.patches():
   s=sink(Path(d)/'a');f=ProtocolFixture();ob=O.CreditAuditObserver(f,s);r=behavior_row(ob);credit(ob,r);ob.minibatch_stop(.1);f.completed_updates=1;ob._finish_update({'gradient_steps':0});s.finish(complete=True)
   v=A.validate(s.path);self.assertEqual(v['observed_main_optimizer_steps'],0);self.assertEqual(v['observed_total_adam_calls'],0)
   cr=next(iter(s.joins.credits.values()));self.assertEqual(cr['raw_adv'],[.9,.8,.7]);self.assertEqual(cr['active_heads'],[True,False,True]);self.assertEqual(cr['T_team'],cr['global_value_target']);self.assertEqual(cr['metadata']['cost_targets'],[0.,0.])
 def test_each_synthetic_step_callback_once_no_actual_optimizer(self):
  with tempfile.TemporaryDirectory() as d,self.patches():
   s=sink(Path(d)/'a');f=ProtocolFixture()
   with O.credit_audit(f,s) as ob:
    r=behavior_row(ob);credit(ob,r);ob.before_step(f,1.,2.,3.,4.)
    self.assertEqual(f.optimizer.step(),'synthetic_result');f.surrogate_optimizer.step();ob.after_step(f,0.)
    f.completed_updates=1;ob._finish_update({'gradient_steps':1})
   s.finish(complete=True);v=A.validate(s.path)
   self.assertEqual((f.optimizer.calls,f.surrogate_optimizer.calls),(1,1));self.assertEqual(v['observed_total_adam_calls'],2);self.assertFalse(v['actual_training_verified'])
 def test_step_return_then_observation_failure_is_persisted(self):
  with tempfile.TemporaryDirectory() as d,self.patches():
   s=sink(Path(d)/'a');f=ProtocolFixture()
   with self.assertRaisesRegex(RuntimeError,'observation failed'):
    with O.credit_audit(f,s) as ob:
     r=behavior_row(ob);credit(ob,r);ob.before_step(f,0.,0.,0.,0.)
     with mock.patch.object(O,'parameter_delta',side_effect=RuntimeError('observation failed')):f.optimizer.step()
   receipt=json.loads((s.root/'failure.json').read_text());self.assertEqual(f.optimizer.calls,1);self.assertTrue(receipt['recent_optimizer_attempts'][0]['original_returned']);self.assertEqual(receipt['optimizer_attempt_counts']['observation_failed'],1)
   self.assertEqual(A.validate(s.path)['status'],'failed');self.assertNotIn('_credit_audit',f.__dict__)
 def test_original_callback_raises_may_have_partially_executed(self):
  with tempfile.TemporaryDirectory() as d,self.patches():
   s=sink(Path(d)/'a');f=ProtocolFixture();f.optimizer.fail=True
   with self.assertRaises(LookupError):
    with O.credit_audit(f,s) as ob:
     r=behavior_row(ob);credit(ob,r);ob.before_step(f,0.,0.,0.,0.);f.optimizer.step()
   a=json.loads((s.root/'failure.json').read_text())['recent_optimizer_attempts'][0];self.assertEqual(f.optimizer.calls,1);self.assertTrue(a['possibly_partial']);self.assertFalse(a['original_returned'])
 def test_original_error_survives_failure_receipt_and_terminal_IO_errors(self):
  with tempfile.TemporaryDirectory() as d,self.patches():
   s=sink(Path(d)/'a');f=ProtocolFixture();f.optimizer.fail=True
   with self.assertRaises(LookupError):
    with O.credit_audit(f,s) as ob:
     r=behavior_row(ob);credit(ob,r);ob.before_step(f,0.,0.,0.,0.)
     with mock.patch.object(Path,'open',side_effect=OSError('disk failed')),mock.patch.object(s,'__exit__',side_effect=OSError('footer failed')),mock.patch.object(O.sys.stderr,'write',side_effect=OSError('stderr failed')):f.optimizer.step()
   self.assertNotIn('_credit_audit',f.__dict__)
   if not s.closed:s.finish()
 def test_closed_sink_rejects_before_any_original_call(self):
  with tempfile.TemporaryDirectory() as d:
   s=sink(Path(d)/'a');f=ProtocolFixture();s.finish()
   with O.credit_audit(f,s) as ob:
    with ob.session(dict(seed=17,session_id='s')):
     with self.assertRaises(ValueError):f.act_controls({},0)
    with self.assertRaises(ValueError):ob._begin_update([])
 def test_stochastic_frame_RNG_probes_observe_only_original_call(self):
  with tempfile.TemporaryDirectory() as d,mock.patch.object(O,'rng_digest',side_effect=['a'*64,'b'*64]) as probe:
   s=sink(Path(d)/'a');f=ProtocolFixture()
   with O.credit_audit(f,s) as ob:
    with ob.session(dict(seed=17,session_id='s')):self.assertEqual(f.act_controls({},0),({},None))
    self.assertEqual((ob._act_rng_before,ob._act_rng_after),('a'*64,'b'*64));self.assertEqual(ob.idle_frames,1)
   self.assertEqual(probe.call_count,2);s.finish()
 def test_full_off_algorithm_AST_identical_to_frozen_baseline(self):
  class Strip(ast.NodeTransformer):
   def visit_Assign(self,n):
    if any(isinstance(t,ast.Name) and t.id=='audit' for t in n.targets):return None
    return self.generic_visit(n)
   def visit_If(self,n):
    if isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='audit':return None
    return self.generic_visit(n)
  old=subprocess.check_output(['git','show',BASE+':src/agents/ppo_agent.py'],cwd=ROOT,text=True);new=(ROOT/'src/agents/ppo_agent.py').read_text()
  self.assertEqual(ast.dump(ast.parse(old),include_attributes=False),ast.dump(Strip().visit(ast.parse(new)),include_attributes=False))
 def test_adapters_have_no_forward_backward_or_optimizer_constructors(self):
  tree=ast.parse((ROOT/'scripts/sa_stage5c_credit_observer.py').read_text())
  for n in ast.walk(tree):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):self.assertNotIn(n.func.attr,('backward','sample','randperm','manual_seed','Adam','load_state_dict'))

class ProbabilityContractTests(unittest.TestCase):
 def test_future_condition_is_not_joint_posterior(self):
  joint={(r,p):.25 for r in (0,1) for p in (0,1)};posterior=sum(v for (r,p),v in joint.items() if r==1)/.5
  self.assertEqual(posterior,1.);self.assertEqual(sum(joint[(1,p)] for p in (0,1)),.5)
 def test_future_baseline_bias_for_prepare_score(self):
  self.assertEqual(sum(.25*r*(r-.5) for r in (0,1) for p in (0,1)),.25)
  self.assertEqual(sum(.25*r*(p-.5) for r in (0,1) for p in (0,1)),0.)
 def test_class_ratio_is_old_conditional_weighted_raw_ratio(self):
  old=[.1,.2,.7];new=[.3,.4,.3];c=[0,1];ratio=sum(new[i] for i in c)/sum(old[i] for i in c)
  self.assertAlmostEqual(ratio,sum(old[i]/sum(old[k] for k in c)*new[i]/old[i] for i in c));self.assertNotEqual(ratio,sum(new[i]/old[i] for i in c)/2)
 def test_raw_clipped_and_class_clipped_are_different_objectives(self):
  old=[.5,.5];new=[.9,.1];raw=sum(p*min(q/p,max(.8,min(1.2,q/p))) for p,q in zip(old,new))
  self.assertAlmostEqual(raw,.7);self.assertEqual(min(sum(new)/sum(old),1.2),1.)
 def test_baseline_regrouping_needs_probability_weighting(self):
  p=[.1,.2,.7];v=[1.,5.,2.];raw=sum(a*b for a,b in zip(p,v));group=.3*((.1*1+.2*5)/.3)+.7*2
  self.assertAlmostEqual(raw,group);self.assertNotAlmostEqual(raw,.3*((1+5)/2)+.7*2)
 def test_separate_head_clipping_is_not_joint_clipping(self):
  ratios=[2.,.5];head=sum(min(x,1.2) for x in ratios)/2;joint=min(math.prod(ratios),1.2)
  self.assertNotEqual(head,joint)

if __name__=='__main__':unittest.main()
