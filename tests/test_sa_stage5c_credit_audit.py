"""Pure synthetic fixtures only. Zero env/model/forward/backward/Adam calls."""
import copy,importlib.util,json,math,tempfile,threading,unittest
from pathlib import Path
from unittest import mock
S=importlib.util.spec_from_file_location('credit_audit',Path(__file__).resolve().parents[1]/'scripts/sa_stage5c_credit_audit.py');M=importlib.util.module_from_spec(S);S.loader.exec_module(M)

def header(kind='synthetic_fixture'):
 return dict(run_uid='fixture_run',source_commit='0'*40,provenance=dict(source_run_id='SYNTHETIC_NO_TRAINING'),evidence_kind=kind)
def behavior(frame=0):
 return dict(seed=17,session_id='fixture_session',frame=frame,vehicle_id='v0',pass_name='two_pass_behavior',request_id='r0',node_id='n0',decision_uid='d'+str(frame),candidates=[['a','b'],[None],[[None,i] for i in range(4)]],masks=[[True,True],[True],[True]*4],action_indices=[0,0,1],old_log_probs=[-.693147,0.,-1.386294],causal_prefix=dict(behavior_uids=[],decision_sha256='a'*64),head_passes=M.DEFAULT_HEAD_PASSES[:])
def credit(b,u):
 return dict(seed=17,update_index=1,update_uid=u,behavior_uid=b,T=8.,T_team=1.,fleet=8,B=[.25,.5,.75],raw_adv=[.75,.5,.25],normalized_adv=[1.,0.,-1.],active_heads=[True,False,True],global_value_target=1.)
def minibatch(b,c,u,epoch=0,stepped=True):
 return dict(seed=17,update_index=1,update_uid=u,epoch=epoch,minibatch_index=0,behavior_uids=[b],credit_uids=[c],rng_before_sha256='b'*64,rng_after_sha256='c'*64,kl=.001,step_observed=stepped,step_uid='observed_main_step_'+str(epoch) if stepped else None,kl_early_stop=not stepped)
def update(u,steps):
 n=len(steps)
 return dict(seed=17,update_index=1,update_uid=u,completed_updates_before=0,completed_updates_after=1,observed_adam_steps=n,step_uids=steps,delta_observation='producer_supplied',adam_delta_summary={g:dict(l2=.25 if n else 0.,max_abs=.25 if n else 0.,parameter_count=1,changed_parameter_count=1 if n else 0) for g in ('policy','value','surrogate')},metadata=dict(optimizer_step_counts=dict(main=n,surrogate=n),total_adam_calls=2*n))
def emit_chain(s,epochs=1,stepped=True):
 b=s.emit('behavior',behavior());u=M.update_uid('fixture_run',17,1);c=s.emit('credit',credit(b,u));steps=[]
 for ep in range(epochs):
  p=minibatch(b,c,u,ep,stepped);s.emit('minibatch',p)
  if p['step_uid'] is not None:steps.append(p['step_uid'])
 s.emit('update',update(u,steps));return b,c,u

class AuditTests(unittest.TestCase):
 def sink(self,p,**kw):return M.AuditSink(p,enable=True,header=header(),**kw)
 def test_disabled_no_path_payload_hash_serialize_callback_or_finish(self):
  class Poison:
   def __fspath__(self):raise AssertionError('path touched')
   def __iter__(self):raise AssertionError('payload touched')
   def __deepcopy__(self,m):raise AssertionError('copied')
  def callback():raise AssertionError('callback called')
  with mock.patch.object(M,'Path',side_effect=AssertionError('Path called')),mock.patch.object(M,'canonical',side_effect=AssertionError('serialized')):
   s=M.AuditSink(Poison(),enable=False,header=Poison());self.assertIsNone(s.emit(Poison(),Poison(),payload_factory=callback));self.assertIsNone(s.finish(complete=Poison()))
 def test_enabled_fresh_directory_and_no_overwrite(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'fresh';s=self.sink(p);s.finish();before=(p/'events.jsonl').read_bytes()
   with self.assertRaises(FileExistsError):self.sink(p)
   self.assertEqual(before,(p/'events.jsonl').read_bytes())
 def test_complete_fixture_and_observed_two_optimizers(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';s=self.sink(p);emit_chain(s);self.assertEqual(s.finish(complete=True),'complete_fixture');r=M.validate(p)
   self.assertTrue(r['structural_complete']);self.assertFalse(r['actual_training_verified']);self.assertEqual(r['unique_behavior_rows'],1);self.assertEqual(r['observed_main_optimizer_steps'],1);self.assertEqual(r['observed_surrogate_optimizer_steps'],1);self.assertEqual(r['observed_total_adam_calls'],2)
 def test_unverified_chain_never_certifies_actual(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';s=M.AuditSink(p,enable=True,header=header('unverified_actual_chain'));emit_chain(s);s.finish(complete=True);r=M.validate(p)
   self.assertEqual(r['status'],'complete_unverified');self.assertFalse(r['actual_training_verified']);self.assertEqual(r['source_stochastic_origin'],'UNVERIFIED')
 def test_kl_early_stop_zero_steps(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';s=self.sink(p);emit_chain(s,stepped=False);s.finish(complete=True);r=M.validate(p)
   self.assertTrue(r['structural_complete']);self.assertEqual(r['observed_total_adam_calls'],0)
 def test_across_epoch_reuse_not_new_behavior(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';s=self.sink(p);emit_chain(s,epochs=4);s.finish(complete=True);r=M.validate(p)
   self.assertEqual(r['unique_behavior_rows'],1);self.assertEqual(r['sample_occurrences'],4);self.assertEqual(r['observed_total_adam_calls'],8)
 def test_uid_stable_identity_not_actions(self):
  b=behavior();u=M.behavior_uid('fixture_run',b);b['action_indices'][0]=1;self.assertEqual(M.behavior_uid('fixture_run',b),u);b['frame']=1;self.assertNotEqual(M.behavior_uid('fixture_run',b),u)
 def test_joint_three_heads_no_duplicate_execution_pass(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';s=self.sink(p);b=behavior();b['pass_name']='second_cache_event_pass'
   with self.assertRaises(M.AuditError):s.emit('behavior',b)
   self.assertEqual(M.validate(p)['status'],'failed')
 def test_empty_on_cannot_complete(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';s=self.sink(p)
   with self.assertRaises(M.AuditError):s.finish(complete=True)
   self.assertFalse(M.validate(p)['structural_complete'])
 def test_partial_and_pending_update_are_not_complete(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';s=self.sink(p);b=s.emit('behavior',behavior());s.emit('credit',credit(b,M.update_uid('fixture_run',17,1)));s.finish();r=M.validate(p)
   self.assertEqual(r['status'],'partial');self.assertEqual(len(r['pending_updates']),1)
 def test_missing_or_truncated_footer_partial(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';s=self.sink(p);emit_chain(s);s.finish(complete=True);f=p/'events.jsonl';data=f.read_bytes();footer=data.splitlines(keepends=True)[-1];f.write_bytes(data[:-len(footer)]);self.assertEqual(M.validate(p)['status'],'partial');f.write_bytes(data[:-3]);self.assertEqual(M.validate(p)['status'],'partial')
 def test_reject_nonfinite_custom_and_ragged_payloads(self):
  bad=[]
  for v in (math.nan,math.inf):b=behavior();b['old_log_probs'][0]=v;bad.append(b)
  b=behavior();b['candidates'][1].append('x');bad.append(b)
  b=behavior();b['masks'][0]=[False,False];bad.append(b)
  b=behavior();b['action_indices'][0]=2;bad.append(b)
  b=behavior();b['seed']=True;bad.append(b)
  b=behavior();b['metadata']=object();bad.append(b)
  for b in bad:
   with self.subTest(b=repr(b)[:90]),tempfile.TemporaryDirectory() as d:
    p=Path(d)/'x';s=self.sink(p)
    with self.assertRaises(M.AuditError):s.emit('behavior',b)
    r=M.validate(p);self.assertEqual(r['event_counts'],{});self.assertEqual(r['status'],'failed')
 def test_failed_row_limit_preserves_only_valid_prefix(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';s=self.sink(p,max_rows=3);s.emit('behavior',behavior())
   with self.assertRaises(M.AuditLimitError):s.emit('behavior',behavior(1))
   r=M.validate(p);self.assertEqual(r['unique_behavior_rows'],1);self.assertLessEqual(r['rows'],3);self.assertEqual(r['status'],'failed')
 def test_byte_limit_failclosed_and_no_sampling(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';s=self.sink(p,max_bytes=3000);b=behavior();b['metadata']={'big':'x'*3000}
   with self.assertRaises(M.AuditLimitError):s.emit('behavior',b)
   r=M.validate(p);self.assertEqual(r['unique_behavior_rows'],0);self.assertLessEqual((p/'events.jsonl').stat().st_size,3000);self.assertEqual(r['status'],'failed')
 def test_noncommit_preflight_and_payload_mutation_isolation(self):
  j=M._Joins(M._header(header(),100,100000));b=behavior();p,u=j.event('behavior',b,commit=False);self.assertEqual(j.behaviors,{})
  j.event('behavior',b);b['masks'][0][0]=False;self.assertTrue(j.behaviors[u]['masks'][0][0])
 def test_kl_reject_before_all_eligible_samples_is_complete_with_gap(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';s=self.sink(p);b=s.emit('behavior',behavior());b2=s.emit('behavior',behavior(1));u=M.update_uid('fixture_run',17,1);c=s.emit('credit',credit(b,u));s.emit('credit',credit(b2,u));s.emit('minibatch',minibatch(b,c,u,stepped=False));up=update(u,[]);up['metadata'].update(eligible_behavior_uids=[b,b2],eligible_behavior_rows=2);s.emit('update',up);s.finish(complete=True);r=M.validate(p)
   self.assertTrue(r['structural_complete']);self.assertEqual(r['never_sampled_behavior_uids'],[b2]);self.assertEqual(r['sampled_unique_behavior_rows'],1);self.assertEqual(r['kl_early_stop_minibatches'],1)
 def test_update_without_any_minibatch_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';s=self.sink(p);b=s.emit('behavior',behavior());u=M.update_uid('fixture_run',17,1);s.emit('credit',credit(b,u))
   with self.assertRaises(M.AuditError):s.emit('update',update(u,[]))
 def test_declared_bound_applies_to_truncated_tail(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';s=self.sink(p,max_bytes=3000);s.finish();f=p/'events.jsonl';head=f.read_bytes().splitlines(keepends=True)[0];f.write_bytes(head+b'x'*3000);self.assertEqual(M.validate(p)['status'],'invalid')
 def test_huge_json_number_fails_closed(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';s=self.sink(p);s.emit('behavior',behavior());s.finish();f=p/'events.jsonl';lines=f.read_bytes().splitlines(keepends=True);x=json.loads(lines[1]);x['payload']['old_log_probs'][0]=10**400;lines[1]=M.canonical(x);f.write_bytes(b''.join(lines));self.assertEqual(M.validate(p)['status'],'invalid')
 def test_sink_never_clones_prior_join_history(self):
  with tempfile.TemporaryDirectory() as d:
   s=self.sink(Path(d)/'x');s.emit('behavior',behavior())
   original=copy.deepcopy
   with mock.patch('copy.deepcopy',wraps=original) as copier:s.emit('behavior',behavior(1))
   self.assertTrue(copier.call_count>=2)
   self.assertTrue(all(not isinstance(call.args[0],M._Joins) for call in copier.call_args_list))
   s.finish()
 def test_boolean_footer_counts_are_invalid(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';s=self.sink(p);emit_chain(s);s.finish(complete=True);f=p/'events.jsonl';lines=f.read_bytes().splitlines(keepends=True);x=json.loads(lines[-1]);x['payload']['event_counts']['behavior']=True;lines[-1]=M.canonical(x);f.write_bytes(b''.join(lines));self.assertEqual(M.validate(p)['status'],'invalid')
 def test_duplicate_behavior_and_unknown_credit_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';s=self.sink(p);s.emit('behavior',behavior())
   with self.assertRaises(M.AuditError):s.emit('behavior',behavior())
   self.assertEqual(M.validate(p)['unique_behavior_rows'],1)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';s=self.sink(p)
   with self.assertRaises(M.AuditError):s.emit('credit',credit('unknown',M.update_uid('fixture_run',17,1)))
 def test_credit_active_and_cross_seed_join_rejected(self):
  for change in ('active','seed'):
   with tempfile.TemporaryDirectory() as d:
    s=self.sink(Path(d)/'x');b=s.emit('behavior',behavior());u=M.update_uid('fixture_run',17,1);c=credit(b,u)
    if change=='active':c['active_heads'][1]=True
    else:c['seed']=7;c['update_uid']=M.update_uid('fixture_run',7,1)
    with self.assertRaises(M.AuditError):s.emit('credit',c)
 def test_update_observed_count_and_delta_finite(self):
  for change in ('main','surrogate','delta'):
   with tempfile.TemporaryDirectory() as d:
    s=self.sink(Path(d)/'x');b=s.emit('behavior',behavior());u=M.update_uid('fixture_run',17,1);c=s.emit('credit',credit(b,u));mb=minibatch(b,c,u);s.emit('minibatch',mb);p=update(u,[mb['step_uid']])
    if change=='main':p['observed_adam_steps']=2
    elif change=='surrogate':p['metadata']['total_adam_calls']=3
    else:p['adam_delta_summary']['policy']['l2']=float('inf')
    with self.assertRaises(M.AuditError):s.emit('update',p)
 def test_single_thread_and_recursive_factory_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   s=self.sink(Path(d)/'x');errors=[]
   def worker():
    try:s.emit('behavior',behavior())
    except M.AuditError as e:errors.append(e)
   t=threading.Thread(target=worker);t.start();t.join();self.assertEqual(len(errors),1)
   with self.assertRaises(M.AuditError):s.emit('behavior',payload_factory=lambda:s.emit('behavior',behavior()))
 def test_invalid_header_and_greedy_origin_not_actual(self):
  for h in [dict(header(),actual_training_verified=True),dict(header(),evidence_kind='actual_stochastic_training'),dict(header(),source_commit='bad')]:
   with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'x'
    with self.assertRaises(M.AuditError):M.AuditSink(p,enable=True,header=h)
    self.assertFalse(p.exists())
  with tempfile.TemporaryDirectory() as d:
   s=self.sink(Path(d)/'x');b=behavior();b['behavior_origin']='greedy_replay'
   with self.assertRaises(M.AuditError):s.emit('behavior',b)
 def test_footer_tamper_trailing_record_sequence_and_duplicate_json_key(self):
  for mode in ('footer','trailing','seq','duplicate_key','nan','exp_overflow'):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
    p=Path(d)/'x';s=self.sink(p);emit_chain(s);s.finish(complete=True);f=p/'events.jsonl';rows=f.read_bytes().splitlines(keepends=True)
    if mode=='footer':x=json.loads(rows[-1]);x['payload']['data_sha256']='f'*64;rows[-1]=M.canonical(x)
    elif mode=='trailing':rows.append(rows[1])
    elif mode=='seq':x=json.loads(rows[1]);x['seq']=123;rows[1]=M.canonical(x)
    elif mode=='duplicate_key':rows[1]=rows[1].replace(b'"seed":17',b'"seed":17,"seed":17')
    elif mode=='nan':rows[1]=rows[1].replace(b'"frame":0',b'"frame":NaN')
    else:rows[1]=rows[1].replace(b'"frame":0',b'"frame":1e999')
    f.write_bytes(b''.join(rows));self.assertEqual(M.validate(p)['status'],'invalid')
class ConsumerCLITests(unittest.TestCase):
 def test_readonly_CLI_fixture_and_incomplete_exit_codes(self):
  import io
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';s=M.AuditSink(p,enable=True,header=header());emit_chain(s);s.finish(complete=True)
   with mock.patch('sys.stdout',new_callable=io.StringIO) as out:self.assertEqual(M.main([str(p)]),0)
   r=json.loads(out.getvalue());self.assertFalse(r['actual_training_verified']);self.assertEqual(r['status'],'complete_fixture')
   f=p/'events.jsonl';f.write_bytes(f.read_bytes().rsplit(b'\n',2)[0]+b'\n')
   with mock.patch('sys.stdout',new_callable=io.StringIO):self.assertEqual(M.main([str(p)]),1)

if __name__=='__main__':unittest.main()
