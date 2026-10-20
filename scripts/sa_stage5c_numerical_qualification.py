"""Bounded private numerical observer qualification; never owns an environment.

bind/status are pure stdlib. run is an exclusive, separately authorized software
qualification: original stochastic calls on frozen observations, synthetic targets.
No historical training replay, environment trajectory or performance claim.
"""
from __future__ import annotations
import argparse, copy, dataclasses, datetime, hashlib, json, math, os, pathlib, random, struct, subprocess, sys, traceback
from contextlib import contextmanager, nullcontext
from types import MethodType

SCHEMA='sa_stage5c_numerical_qualification_v1'
LIMITS=dict(model_instances=14,original_evaluate_calls=1024,module_forward_calls=100000,
 categorical_sample_calls=2048,backward_calls=32,adam_calls=64,synthetic_update_calls=14,
 environment_calls=0,autograd_grad_calls=0,scientific_training_sessions=0)
CONTROLLER_FIELDS=('request_memory','target_memory','commitments','_team_targets','_calendar',
 '_delivery_decisions','_feature_opportunities','_cache_values','_model_catalog','last_diagnostics')
EXCLUDED_OBSERVER_FIELDS=('credit_audit_uids',)

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read_json(path):return json.loads(pathlib.Path(path).read_text())
def write_json(path,data,exclusive=False):
 path=pathlib.Path(path);wire=json.dumps(data,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)+'\n'
 if exclusive:
  with path.open('x') as f:f.write(wire);f.flush();os.fsync(f.fileno())
 else:
  temporary=path.with_name(path.name+'.tmp')
  with temporary.open('w') as f:f.write(wire);f.flush();os.fsync(f.fileno())
  os.replace(temporary,path)
def verify_bindings(bindings):
 for item in bindings:
  if sha_file(item['path'])!=item['sha256']:raise ValueError('Frozen SHA mismatch: '+str(item['path']))
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()

def compare_exact(left,right,path='root'):
 """No tolerance, no NaN equality. Runtime tensors/arrays are duck typed."""
 if dataclasses.is_dataclass(left) or dataclasses.is_dataclass(right):
  if type(left)!=type(right):raise AssertionError(path+': dataclass type')
  return compare_exact(dataclasses.asdict(left),dataclasses.asdict(right),path)
 if hasattr(left,'detach') or hasattr(right,'detach'):
  if type(left)!=type(right) or left.dtype!=right.dtype or left.shape!=right.shape or left.device!=right.device:raise AssertionError(path+': tensor schema')
  if not bool(left.isfinite().all()) or not bool(right.isfinite().all()):raise AssertionError(path+': nonfinite tensor')
  import torch
  lb=left.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
  rb=right.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
  if lb!=rb:raise AssertionError(path+': tensor bits')
  return
 if type(left).__module__.startswith('numpy') or type(right).__module__.startswith('numpy'):
  if type(left)!=type(right) or left.dtype!=right.dtype or left.shape!=right.shape:raise AssertionError(path+': array schema')
  compare_exact(left.tolist(),right.tolist(),path+'.array_finite')
  if left.tobytes(order='C')!=right.tobytes(order='C'):raise AssertionError(path+': array bits')
  return
 if type(left)!=type(right):raise AssertionError(path+': type '+str(type(left))+' != '+str(type(right)))
 if isinstance(left,dict):
  compare_exact(list(left.keys()),list(right.keys()),path+'.key_order')
  for key in left:compare_exact(left[key],right[key],path+'.'+str(key))
 elif isinstance(left,(tuple,list)):
  if len(left)!=len(right):raise AssertionError(path+': sequence length')
  for i,(a,b) in enumerate(zip(left,right)):compare_exact(a,b,path+'['+str(i)+']')
 elif isinstance(left,(set,frozenset)):
  for value in left:compare_exact(value,value,path+'.set_finite')
  for value in right:compare_exact(value,value,path+'.set_finite')
  if len(left)!=len(right):raise AssertionError(path+': set size')
  for value in left:
   matches=[v for v in right if type(v) is type(value) and v==value]
   if len(matches)!=1:raise AssertionError(path+': set value/type')
   compare_exact(value,matches[0],path+'.set_value')
 elif isinstance(left,float):
  if not math.isfinite(left) or not math.isfinite(right) or struct.pack('!d',left)!=struct.pack('!d',right):raise AssertionError(path+': nonfinite/different float bits')
 elif left!=right:raise AssertionError(path+': value mismatch')

class Budget:
 def __init__(self,limits,output_root):
  self.limits=dict(limits);self.counts={k:0 for k in limits};self.denied=[];self.output_root=pathlib.Path(output_root);self.context={};self.flush()
 def flush(self):write_json(self.output_root/'progress.json',dict(schema=SCHEMA,status=self.context.get('phase') if self.context.get('phase') in ('complete','failed') else 'running',updated_at=utc(),counts=self.counts,limits=self.limits,denied=self.denied,context=self.context))
 def charge(self,name,n=1):
  if type(n) is not int or n<0 or name not in self.limits:raise ValueError('Invalid budget charge')
  if self.counts[name]+n>self.limits[name]:
   self.denied.append(dict(name=name,n=n,before=self.counts[name],context=dict(self.context)));self.flush();raise RuntimeError('Budget exceeded before original: '+name)
  self.counts[name]+=n;self.flush()

class Patches:
 def __init__(self):self.originals=[]
 def set(self,obj,name,value):
  own=name in vars(obj);self.originals.append((obj,name,own,vars(obj).get(name)));setattr(obj,name,value)
 def close(self):
  for obj,name,own,old in reversed(self.originals):
   if own:setattr(obj,name,old)
   else:delattr(obj,name)
  self.originals=[]
 def __enter__(self):return self
 def __exit__(self,*exc):self.close();return False

def contract_check(c):
 if c.get('schema')!=SCHEMA or c.get('limits')!=LIMITS:raise ValueError('Exact frozen schema/limits required')
 expected=[(s,a,'unmodified_model') for s in (7,17,27) for a in ('sa_legacy_clip','sa_group_clip')]+[(7,'sa_legacy_clip','synthetic_update_logit_stress')]
 if [(r['seed'],r['arm'],r['mode']) for r in c['matrix']]!=expected or any(r['epochs']!=2 for r in c['matrix']):raise ValueError('Exact seven-pair matrix required')
 bound={x['path'] for x in c['bindings']}
 if any(r[k] not in bound for r in c['matrix'] for k in ('checkpoint','snapshot')):raise ValueError('Matrix assets must be SHA bound')

def bind(args):
 root=args.repo_root.resolve();out=args.output_root.resolve()
 if out!=args.contract.resolve().parent:raise ValueError('Frozen budget root must equal contract parent')
 out.mkdir(parents=True,exist_ok=True)
 if (out/'run_started.json').exists():raise RuntimeError('Already claimed; no rebind')
 c=read_json(args.contract);contract_check(c);verify_bindings(c['bindings'])
 if git(root,'status','--porcelain'):raise RuntimeError('Clean runtime checkout required')
 subprocess.check_call(['git','merge-base','--is-ancestor',c['source_commit'],'HEAD'],cwd=root)
 b_root=pathlib.Path(c['matrix'][0]['snapshot']).parent.parent
 extra=[]
 for filename in ('contract.json','plan.json','result.json'):
  p=b_root/filename;extra.append(dict(path=str(p),sha256=sha_file(p)))
 old=read_json(b_root/'contract.json');done=read_json(b_root/'result.json')
 if done['status']!='complete' or not done['source_before_after_exact']:raise ValueError('B must be complete with original sources intact')
 if done['contract_sha256']!=sha_file(b_root/'contract.json'):raise ValueError('B receipt contract SHA')
 cp=[]
 for byseed in old['checkpoints'].values():
  for item in byseed.values():cp.append(dict(path=item['path'],sha256=item['sha256']))
 if len(cp)!=18:raise ValueError('Expected eighteen original B checkpoint bindings')
 verify_bindings(cp)
 sources=[]
 for rel in git(root,'ls-files','src').splitlines():
  if rel.endswith('.py'):sources.append(dict(path=str(root/rel),sha256=sha_file(root/rel)))
 for rel in ('scripts/sa_stage5c_credit_observer.py','scripts/sa_stage5c_credit_audit.py','scripts/sa_stage5c_numerical_qualification.py'):
  sources.append(dict(path=str(root/rel),sha256=sha_file(root/rel)))
 if pathlib.Path(__file__).resolve()!=root/'scripts/sa_stage5c_numerical_qualification.py':raise ValueError('Bind installed runner only')
 payload=dict(schema=SCHEMA+'_binding',bound_at=utc(),contract=dict(path=str(args.contract.resolve()),sha256=sha_file(args.contract)),
  repo_root=str(root),output_root=str(out),source_commit=c['source_commit'],runtime_commit=git(root,'rev-parse','HEAD'),
  runner=dict(path=str(pathlib.Path(__file__).resolve()),sha256=sha_file(__file__)),source_fingerprints=sources,
  b_metadata=extra,b_checkpoints=cp,limits=LIMITS,private_only=True,environment_calls=0)
 write_json(out/'runner_binding.json',payload,exclusive=True);print(json.dumps(dict(status='bound',binding_sha256=sha_file(out/'runner_binding.json'))))

def verify_run(binding,contract):
 root=pathlib.Path(binding['repo_root']);contract_check(contract)
 if sha_file(binding['contract']['path'])!=binding['contract']['sha256']:raise ValueError('Contract changed')
 if sha_file(__file__)!=binding['runner']['sha256']:raise ValueError('Runner changed')
 if git(root,'rev-parse','HEAD')!=binding['runtime_commit'] or git(root,'status','--porcelain'):raise ValueError('Runtime HEAD/clean status changed')
 verify_bindings(binding['source_fingerprints']);verify_bindings(binding['b_metadata']);verify_bindings(binding['b_checkpoints']);verify_bindings(contract['bindings'])

def rng_state(torch,np):return dict(python=copy.deepcopy(random.getstate()),numpy=copy.deepcopy(np.random.get_state()),torch_cpu=torch.get_rng_state().clone())
def restore_rng(state,torch,np):random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'])
def clone_tree(x):
 if hasattr(x,'detach'):return x.detach().cpu().clone()
 if dataclasses.is_dataclass(x):return copy.deepcopy(x)
 if isinstance(x,dict):return {k:clone_tree(v) for k,v in x.items()}
 if isinstance(x,list):return [clone_tree(v) for v in x]
 if isinstance(x,tuple):return tuple(clone_tree(v) for v in x)
 return copy.deepcopy(x)
def controller(agent):return {k:clone_tree(getattr(agent,k)) for k in CONTROLLER_FIELDS if hasattr(agent,k)}
def parameter_gradients(agent):return {n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in agent.named_parameters()}
def learned_state(agent):
 return dict(state_dict=clone_tree(agent.state_dict()),optimizer=clone_tree(agent.optimizer.state_dict()),
  surrogate_optimizer=clone_tree(agent.surrogate_optimizer.state_dict()),gradients=parameter_gradients(agent),
  modes={n:bool(m.training) for n,m in agent.named_modules()},options=clone_tree(agent.options),completed_updates=int(agent.completed_updates))
def extra_agent_state(agent):
 # Exclude registered PyTorch modules/optimizers and observer instance wrappers;
 # all controller caches, scalar flags and gradient summary lists remain visible.
 omit={'_parameters','_buffers','_modules','optimizer','surrogate_optimizer','_actor_parameters','_separate_policy_parameters','_separate_value_parameters','_credit_audit'}
 out={}
 for k,v in agent.__dict__.items():
  if k in omit or callable(v):continue
  if k.startswith('_') and 'hook' in k:continue
  out[k]=clone_tree(v)
 return out

class RuntimeGuards:
 def __init__(self,torch,budget,Agent,Scheduled):self.torch=torch;self.budget=budget;self.Agent=Agent;self.Scheduled=Scheduled;self.patches=Patches();self.trace=None;self.forward_trace=None;self.backward_trace=None;self.agent=None;self.backward_depth=0
 def __enter__(self):
  try:return self._install()
  except BaseException:self.patches.close();raise
 def _install(self):
  t=self.torch;b=self.budget;p=self.patches
  if t.cuda.is_initialized():raise RuntimeError('CPU qualification refuses initialized CUDA')
  def forbidden_cuda(*a,**k):raise RuntimeError('CPU qualification forbids CUDA initialization/migration')
  p.set(t.cuda,'_lazy_init',forbidden_cuda);p.set(t.Tensor,'cuda',forbidden_cuda);p.set(t.nn.Module,'cuda',forbidden_cuda)
  original_init=self.Agent.__init__
  def init(this,*a,**k):b.charge('model_instances');return original_init(this,*a,**k)
  p.set(self.Agent,'__init__',init)
  original_evaluate=self.Scheduled._evaluate
  def evaluate(this,*a,**k):
   b.charge('original_evaluate_calls')
   if self.forward_trace is None or this is not self.agent:raise RuntimeError('Evaluate outside a private side')
   row=dict(phase='update' if k.get('actions') is not None else 'first_execution' if k.get('execution_only') else 'second_cache_event',
    model_completed_updates=int(this.completed_updates),preceding_adam_calls=len(self.trace),
    args=clone_tree(a),kwargs=clone_tree(k),rng_before=rng_state(t,self.np),status='attempted')
   self.forward_trace.append(row)
   result=original_evaluate(this,*a,**k)
   row.update(result=clone_tree(result),rng_after=rng_state(t,self.np),status='returned')
   return result
  p.set(self.Scheduled,'_evaluate',evaluate)
  original_module=t.nn.Module._call_impl
  def module(this,*a,**k):b.charge('module_forward_calls');return original_module(this,*a,**k)
  p.set(t.nn.Module,'_call_impl',module)
  original_sample=t.distributions.Categorical.sample
  def sample(this,*a,**k):b.charge('categorical_sample_calls');return original_sample(this,*a,**k)
  p.set(t.distributions.Categorical,'sample',sample)
  original_backward=t.Tensor.backward
  def backward(this,*a,**k):
   b.charge('backward_calls')
   if self.backward_trace is None or self.agent is None:raise RuntimeError('Backward outside a private side')
   row=dict(loss=clone_tree(this),preceding_adam_calls=len(self.trace),rng_before=rng_state(t,self.np),status='attempted')
   self.backward_trace.append(row);self.backward_depth+=1
   try:
    result=original_backward(this,*a,**k)
    row.update(status='returned',raw_preclip_gradients=parameter_gradients(self.agent),rng_after=rng_state(t,self.np))
    return result
   finally:self.backward_depth-=1
  p.set(t.Tensor,'backward',backward)
  original_autograd=t.autograd.backward
  def autograd(*a,**k):
   if not self.backward_depth:raise RuntimeError('Direct autograd.backward forbidden; counted Tensor.backward only')
   return original_autograd(*a,**k)
  p.set(t.autograd,'backward',autograd)
  def forbidden_grad(*a,**k):b.charge('autograd_grad_calls');raise AssertionError('unreachable')
  p.set(t.autograd,'grad',forbidden_grad)
  original_step=t.optim.Adam.step
  def step(opt,*a,**k):
   b.charge('adam_calls')
   if self.trace is None or self.agent is None:raise RuntimeError('Adam outside a private observed side')
   which='main' if opt is self.agent.optimizer else 'surrogate' if opt is self.agent.surrogate_optimizer else None
   if which is None:raise RuntimeError('Unknown optimizer')
   row=dict(optimizer=which,before=learned_state(self.agent),rng_before=rng_state(t,self.np),status='attempted')
   self.trace.append(row)
   result=original_step(opt,*a,**k)
   row.update(status='returned',after=learned_state(self.agent),rng_after=rng_state(t,self.np))
   return result
  p.set(t.optim.Adam,'step',step)
  # Import required class definitions but never instantiate/reset/step an env.
  from src.envs.core.multi_request_stream_env import MultiRequestStreamEnv
  seen=set()
  for modname,mod in list(sys.modules.items()):
   if not modname.startswith('src.envs.') or mod is None:continue
   for value in list(vars(mod).values()):
    if not isinstance(value,type) or not value.__name__.endswith('Env') or value in seen:continue
    seen.add(value)
    for name in ('__init__','reset','step','step_controls','step_actions','decision_states'):
     if name not in value.__dict__:continue
     def blocked(*a,**k):b.charge('environment_calls');raise AssertionError('unreachable')
     p.set(value,name,blocked)
  return self
 def __exit__(self,*exc):self.patches.close();return False

@contextmanager
def update_stress(agent,enabled,torch):
 if not enabled:yield;return
 original=agent._choice_distribution;own='_choice_distribution' in agent.__dict__;old=agent.__dict__.get('_choice_distribution')
 def choice(this,logits,mask,deterministic):
  shifted=logits+torch.arange(logits.shape[-1],dtype=logits.dtype,device=logits.device)[None,:]*4
  return original(shifted,mask,deterministic)
 agent._choice_distribution=MethodType(choice,agent)
 try:yield
 finally:
  if own:agent._choice_distribution=old
  else:del agent._choice_distribution

def prepare_inputs(capsule):
 frames=capsule['selected_frames']
 if len(frames)!=9 or len(set(frames))!=9:raise ValueError('Exact nine frozen quantile frames required')
 selected=[]
 for index in (0,4,8):
  frame=frames[index];team=capsule['team_frames'][str(frame)];decisions=team['decisions'];shots=[s for s in capsule['snapshots'] if s['frame']==frame]
  if bool(decisions)!=team['scored'] or len(shots)!=2*len(decisions):raise ValueError('Full-frame snapshot cardinality')
  first=[s for s in shots if s['phase']=='first_execution_pass']
  if decisions:
   ids=sorted(decisions);shift=frame%len(ids);order=ids[shift:]+ids[:shift]
   if [s['feature']['vehicle_id'] for s in first]!=order:raise ValueError('First-pass owner order')
   for shot in first:compare_exact(shot['feature']['full_decision'],decisions[shot['feature']['vehicle_id']],'frozen_decision')
   context=copy.deepcopy(first[0]['feature']['controller'])
  else:context=None
  selected.append(dict(frame=frame,quantile_index=index,decisions=copy.deepcopy(decisions),controller=context,
   restoration='effective post-request-cleanup/pre-first-sample context; not historical RNG' if decisions else 'idle reset_controller; no historical controller-restoration claim'))
 return selected

def attach_targets(roll,frame,start,torch):
 n=len(roll['vehicle_ids']);values=[8*(.5+(i%7)/4) for i in range(n)]
 roll.update(frame_index=frame,initial_request_count=8,episode_end=False,
  returns=torch.tensor(values,dtype=roll['values'].dtype,device=roll['values'].device),
  cost_targets=torch.tensor([[.02,.03] for _ in range(n)],dtype=roll['values'].dtype,device=roll['values'].device))
 roll['team_credit_targets']=roll['returns']/8;roll['credit_targets']=roll['team_credit_targets'].clone()
 return start+n

def _run_side_core(agent,inputs,item,side,home,guards,budget,torch,np,AuditSink,credit_audit,validate,partial):
 enabled=side=='on';trace=[];forward_trace=[];backward_trace=[];guards.trace=trace;guards.forward_trace=forward_trace;guards.backward_trace=backward_trace;guards.agent=agent
 sink=AuditSink(home/'audit',enable=enabled,header=dict(run_uid=home.parent.name+'_'+home.name,source_commit=budget.binding['runtime_commit'],
  evidence_kind='synthetic_fixture',provenance=dict(source_run_id='frozen_B_inputs_plus_synthetic_labels',checkpoint=item['checkpoint'],snapshot=item['snapshot'])),
  max_rows=10000,max_bytes=67108864)
 acts=[];rolls=[];rowindex=0;before=learned_state(agent);start_rng=rng_state(torch,np)
 partial.update(acts=acts,rolls=rolls,step_trace=trace,forward_trace=forward_trace,backward_trace=backward_trace,before=before,rng_before=start_rng)
 with sink:
  with credit_audit(agent,sink) as observer:
   for inp in inputs:
    agent.reset_controller()
    if inp['controller'] is not None:
     for key,value in inp['controller'].items():setattr(agent,key,copy.deepcopy(value))
    decisions=copy.deepcopy(inp['decisions']);before_decisions=copy.deepcopy(decisions)
    identity=dict(seed=item['seed'],session_id=home.parent.name+'_'+home.name+'_frame'+str(inp['frame']),arm=item['arm'],phase='synthetic_frozen_input',case_id='dev_046',frame=inp['frame'])
    with observer.session(identity) if enabled else nullcontext():
     pre_rng=rng_state(torch,np);controls,roll=agent.act_controls(decisions,inp['frame'],deterministic=False);post_rng=rng_state(torch,np)
    compare_exact(decisions,before_decisions,'act input mutation')
    if roll is not None:rowindex=attach_targets(roll,inp['frame'],rowindex,torch);rolls.append(roll)
    acts.append(dict(frame=inp['frame'],controls=clone_tree(controls),roll=clone_tree(roll),controller=controller(agent),rng_before=pre_rng,rng_after=post_rng))
   if not rolls:raise ValueError('Frozen sample has no active row')
   rolls[-1]['episode_end']=True
   targets_before=[{k:clone_tree(r[k]) for k in ('returns','credit_targets','team_credit_targets','cost_targets')} for r in rolls]
   budget.charge('synthetic_update_calls')
   with update_stress(agent,item['mode']=='synthetic_update_logit_stress',torch):stats=agent.update(rolls,epochs=item['epochs'])
   for i,r in enumerate(rolls):compare_exact({k:r[k] for k in targets_before[i]},targets_before[i],'MRO targets restored')
  if enabled:sink.finish(complete=True)
 # Context has ended: no observer attributes or instance methods may survive.
 if hasattr(agent,'_credit_audit') or any(k in agent.__dict__ for k in ('act_controls','_coordinated_features','_evaluate','_choice_distribution','update')):raise ValueError('Observer/stress wrapper not restored')
 audit=validate(home/'audit') if enabled else None
 if enabled and (audit['status']!='complete_fixture' or not audit['structural_complete'] or audit.get('actual_training_verified') is True):raise ValueError('Synthetic audit did not complete structurally')
 result=dict(before=before,after=learned_state(agent),act_outputs=acts,rolls=clone_tree(rolls),update_stats=clone_tree(stats),step_trace=trace,forward_trace=forward_trace,backward_trace=backward_trace,
  rng_before=start_rng,rng_after=rng_state(torch,np),controller=controller(agent),extra_agent_state=extra_agent_state(agent),audit=audit)
 agent.save_checkpoint(home/'private_serialized.pt')
 result['serialized_payload']=torch.load(home/'private_serialized.pt',map_location='cpu',weights_only=False)
 result['rng_after_serialization']=rng_state(torch,np)
 compare_exact(result['rng_after'],result['rng_after_serialization'],'serialization RNG')
 torch.save(result,home/'evidence.pt')
 guards.trace=None;guards.forward_trace=None;guards.backward_trace=None;guards.agent=None
 return result

def run_side(agent,inputs,item,side,home,guards,budget,torch,np,AuditSink,credit_audit,validate):
 partial={}
 try:return _run_side_core(agent,inputs,item,side,home,guards,budget,torch,np,AuditSink,credit_audit,validate,partial)
 except BaseException as exc:
  try:
   payload=dict(partial=clone_tree(partial),after_failure=learned_state(agent),controller=controller(agent),
    extra_agent_state=extra_agent_state(agent),rng_after_failure=rng_state(torch,np),counts=dict(budget.counts),
    exception_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),possibly_partial_update=True)
   torch.save(payload,home/'partial_evidence.pt')
  except BaseException:pass
  raise

def without_audit_uids(value):
 if isinstance(value,dict):return {k:without_audit_uids(v) for k,v in value.items() if k not in EXCLUDED_OBSERVER_FIELDS}
 if isinstance(value,list):return [without_audit_uids(v) for v in value]
 if isinstance(value,tuple):return tuple(without_audit_uids(v) for v in value)
 return value

def execute(args):
 out=args.output_root.resolve();binding=read_json(out/'runner_binding.json');c=read_json(binding['contract']['path']);verify_run(binding,c)
 if str(out)!=binding['output_root']:raise ValueError('Output root differs from binding')
 write_json(out/'run_started.json',dict(schema=SCHEMA,started_at=utc(),pid=os.getpid(),runner_sha256=sha_file(__file__),binding_sha256=sha_file(out/'runner_binding.json'),contract_sha256=binding['contract']['sha256']),exclusive=True)
 budget=Budget(c['limits'],out);budget.binding=binding;completed=[];failure=None
 try:
  sys.path.insert(0,binding['repo_root'])
  import numpy as np
  import torch
  from src.agents.sa_ghmappo_agent import StreamSAGHMAPPOAgent as Agent
  from src.agents.ppo_agent import ScheduledControlPPO as Scheduled
  from scripts.sa_stage5c_credit_audit import AuditSink,validate
  from scripts.sa_stage5c_credit_observer import credit_audit
  torch.set_num_threads(1)
  if torch.cuda.is_initialized():raise RuntimeError('CPU only; CUDA already initialized')
  with RuntimeGuards(torch,budget,Agent,Scheduled) as guards:
   guards.np=np
   for index,item in enumerate(c['matrix']):
    budget.context=dict(pair_index=index,seed=item['seed'],arm=item['arm'],mode=item['mode'],phase='input_load');budget.flush()
    pair=out/('pair_%02d_%s_seed%d_%s'%(index,item['arm'],item['seed'],item['mode']));pair.mkdir(exist_ok=False)
    capsule=torch.load(item['snapshot'],map_location='cpu',weights_only=False)
    witness=capsule['witness']
    if witness['seed']!=item['seed'] or witness['arm']!=item['arm'] or witness['episode']!=64 or witness['case_id']!='dev_046':raise ValueError('Frozen capsule witness mismatch')
    inputs=prepare_inputs(capsule);del capsule
    torch.save(dict(inputs=inputs,source=dict(checkpoint=item['checkpoint'],snapshot=item['snapshot']),synthetic_target_rule=c['synthetic_targets']),pair/'frozen_inputs.pt')
    models=[Agent.from_checkpoint(item['checkpoint']),Agent.from_checkpoint(item['checkpoint'])]
    for model in models:
     if model.separate_policy_value_clip!=(item['arm']=='sa_group_clip') or int(model.completed_updates)!=16:raise ValueError('Frozen checkpoint flag/update mismatch')
     if any(p.device.type!='cpu' for p in model.parameters()):raise ValueError('CPU parameters required')
    compare_exact(learned_state(models[0]),learned_state(models[1]),'loaded private copies')
    # This fixture RNG is explicit; it is not the archived greedy or historic training RNG.
    random.seed(20261020+index);np.random.seed(20261020+index);torch.manual_seed(20261020+index)
    pair_rng=rng_state(torch,np);results=[];deltas=[]
    for side,model in zip(('off','on'),models):
     restore_rng(pair_rng,torch,np);home=pair/side;home.mkdir(exist_ok=False)
     budget.context.update(phase=side);before_counts=dict(budget.counts);budget.flush()
     result=run_side(model,inputs,item,side,home,guards,budget,torch,np,AuditSink,credit_audit,validate)
     results.append(result);deltas.append({k:budget.counts[k]-before_counts[k] for k in budget.counts})
    fields=('before','after','act_outputs','rolls','update_stats','step_trace','forward_trace','backward_trace','rng_before','rng_after','controller','extra_agent_state','serialized_payload','rng_after_serialization')
    for field in fields:compare_exact(without_audit_uids(results[0][field]),without_audit_uids(results[1][field]),'pair.'+field)
    compare_exact(deltas[0],deltas[1],'on/off original operation counts')
    steps=results[0]['update_stats']['gradient_steps'];stopped=results[0]['update_stats']['kl_early_stop']
    if item['mode']=='synthetic_update_logit_stress':
     if steps!=0 or stopped!=1 or deltas[0]['backward_calls'] or deltas[0]['adam_calls']:raise ValueError('Stress must stop at KL before any backward/Adam')
    elif steps<1:raise ValueError('Unmodified pair must exercise at least one original gradient step')
    verify_run(binding,c)
    receipt=dict(status='pass',pair_index=index,seed=item['seed'],arm=item['arm'],mode=item['mode'],counts_per_side=deltas,gradient_steps=steps,
     frames=[x['frame'] for x in inputs],rows=sum(len(x['decisions']) for x in inputs),compared_fields=list(fields),excluded_fields=list(EXCLUDED_OBSERVER_FIELDS),
     evidence_sha256={str(p.relative_to(pair)):sha_file(p) for p in (pair/'frozen_inputs.pt',pair/'off/evidence.pt',pair/'on/evidence.pt',pair/'off/private_serialized.pt',pair/'on/private_serialized.pt')})
    write_json(pair/'result.json',receipt,exclusive=True);completed.append(receipt);write_json(out/'pairs.json',completed)
    print(json.dumps(dict(pair=index,status='pass',mode=item['mode'],gradient_steps=steps,counts=budget.counts)),flush=True)
    del models,results,inputs
  verify_run(binding,c)
  result=dict(schema=SCHEMA,status='pass',completed_at=utc(),binding_sha256=sha_file(out/'runner_binding.json'),counts=budget.counts,limits=budget.limits,pairs=completed,
   environment_calls=0,scientific_training_sessions=0,device_scope='single-thread CPU; GPU untested',
   boundary='Exact observer on/off private numerical software qualification on frozen B observations plus synthetic targets. No original stochastic training reconstruction, no performance evidence; historical pre-hook source was AST-compared, not separately model-executed.')
  write_json(out/'result.json',result,exclusive=True);budget.context={'phase':'complete'};budget.flush();return 0
 except BaseException as exc:
  failure=dict(schema=SCHEMA,status='failed',failed_at=utc(),exception_type=type(exc).__name__,message=str(exc),context=budget.context,counts=budget.counts,limits=budget.limits,denied=budget.denied,completed_pairs=len(completed),no_retry=True,possibly_partial_private_update=True,traceback=traceback.format_exc())
  try:
   write_json(out/'failure.json',failure,exclusive=True);budget.context.update(phase='failed');budget.flush()
  except BaseException:pass
  raise

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 b=sub.add_parser('bind');b.add_argument('--contract',type=pathlib.Path,required=True);b.add_argument('--repo_root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]);b.add_argument('--output_root',type=pathlib.Path,required=True)
 for mode in ('run','status'):
  s=sub.add_parser(mode);s.add_argument('--output_root',type=pathlib.Path,required=True)
 a=p.parse_args(argv)
 if a.command=='bind':bind(a);return 0
 if a.command=='run':return execute(a)
 for name in ('failure.json','result.json','progress.json','runner_binding.json'):
  q=a.output_root/name
  if q.exists():print(json.dumps(read_json(q),ensure_ascii=False));return 0
 raise FileNotFoundError('No qualification status')
if __name__=='__main__':raise SystemExit(main())
