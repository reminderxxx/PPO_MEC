"""Opt-in observations of existing SA stream calls; never starts a run or an update.

No registry/CLI activation and no new forward/backward/optimizer call. The caller
must independently authorize actual stochastic sessions. Default disabled binding
returns before touching the agent, filesystem, tensors or random state.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict,is_dataclass
from types import MethodType
import hashlib,json,math,pickle,random,sys
from pathlib import Path


def detached(value):
    """Copy observations to JSON primitives immediately; retain no graph/tensor."""
    if value is None or type(value) in (str,int,float,bool):return value
    if hasattr(value,'detach') and hasattr(value,'cpu'):return detached(value.detach().cpu().tolist())
    if is_dataclass(value):return detached(asdict(value))
    if isinstance(value,dict):return {str(k):detached(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [detached(v) for v in value]
    if type(value).__module__.startswith('numpy'):
        return detached(value.tolist() if hasattr(value,'tolist') else value.item())
    raise TypeError('Unsupported observation type: '+type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(detached(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def rng_digest():
    # These reads do not seed/sample/restore a generator. No CUDA initialization.
    import numpy as np
    import torch
    h=hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()),protocol=4))
    h.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_initialized():
        for state in torch.cuda.get_rng_state_all():h.update(state.cpu().numpy().tobytes())
    return h.hexdigest()


def group_name(name):
    if name.startswith('surrogates.'):return 'surrogate'
    if name.startswith(('critic.','prefix_values.','action_values.')):return 'value'
    return 'policy'


def parameter_snapshot(agent):
    return {n:p.detach().cpu().clone() for n,p in agent.named_parameters()}


def parameter_delta(agent,before):
    groups={g:dict(squared=0.,max_abs=0.,parameter_count=0,changed_parameter_count=0) for g in ('policy','value','surrogate')}
    pairs=dict(agent.named_parameters())
    if set(pairs)!=set(before):raise RuntimeError('Observed parameter topology changed')
    for name,p in pairs.items():
        delta=p.detach().cpu().double()-before[name].double();g=groups[group_name(name)]
        if not bool(delta.isfinite().all()):raise FloatingPointError('Nonfinite observed parameter delta')
        g['squared']+=float(delta.square().sum());g['max_abs']=max(g['max_abs'],float(delta.abs().max()) if delta.numel() else 0.)
        g['parameter_count']+=delta.numel();g['changed_parameter_count']+=int((delta!=0).sum())
    return {g:dict(l2=math.sqrt(v.pop('squared')),**v) for g,v in groups.items()}


def postclip_gradients(agent):
    groups={g:dict(squared=0.,present_tensors=0,elements=0) for g in ('policy','value','surrogate')}
    for n,p in agent.named_parameters():
        if p.grad is None:continue
        z=p.grad.detach().cpu().double();g=groups[group_name(n)]
        if not bool(z.isfinite().all()):raise FloatingPointError('Nonfinite observed postclip gradient')
        g['squared']+=float(z.square().sum());g['present_tensors']+=1;g['elements']+=z.numel()
    return {g:dict(l2=math.sqrt(v.pop('squared')),**v) for g,v in groups.items()}


class CreditAuditObserver:
    """Passive adapter. Only explicitly bound agents receive the private callback."""
    def __init__(self,agent,sink):
        self.agent=agent;self.sink=sink;self.enabled=sink.enabled;self._patches=[]
        self._session=None;self._collect=False;self._updating=False;self._behaviors={};self._calls=[]
        self._epoch=-1;self._mb=None;self._update=None;self.idle_frames=0;self._attempt_counts={};self._recent_attempts=[]

    def _patch(self,obj,name,call):
        own=name in obj.__dict__;previous=obj.__dict__.get(name)
        self._patches.append((obj,name,own,previous));setattr(obj,name,MethodType(call,obj))

    def __enter__(self):
        if not self.enabled:return self
        a=self.agent
        if a.action_contract!='joint_stream_control_v2' or a.method!='sa_ghmappo':raise ValueError('Only SA stream audit is supported')
        if hasattr(a,'_credit_audit'):raise ValueError('Nested/conflicting audit binding')
        act=a.act_controls;features=a._coordinated_features;evaluate=a._evaluate;choice=a._choice_distribution;update=a.update
        observer=self
        def observe_features(this,decision,frame,reservations):
            result=features(decision,frame,reservations)
            if observer._collect:
                observer._feature=dict(vehicle_id=result[2],node_id=result[3],candidates=detached(result[1]),decision=detached(decision),
                    capture_phase='after_features_before_evaluate',context=detached(dict(capture_phase='after_features_before_evaluate',calendar=this._calendar,reservations=reservations,target_memory=this.target_memory,commitments=this.commitments,request_memory=this.request_memory)))
            return result
        def observe_choice(this,logits,mask,deterministic):
            result=choice(logits,mask,deterministic)
            if observer._collect:
                observer._distributions.append(dict(logits=detached(result.logits),probabilities=detached(result.probs),actual_mask=detached(mask)))
            return result
        def observe_evaluate(this,data,**kwargs):
            if not observer._collect:return evaluate(data,**kwargs)
            before=dict(feature=observer._feature,input=detached(data),kwargs=detached(kwargs),input_schema={k:dict(shape=list(getattr(v,'shape',[])),dtype=str(getattr(v,'dtype','primitive'))) for k,v in data.items()});before['input_json_sha256']=digest(before['input']);observer._distributions=[]
            result=evaluate(data,**kwargs)
            before.update(distributions=observer._distributions,result=detached(result));observer._calls.append(before)
            return result
        def observe_act(this,decisions,frame,*,deterministic=False):
            if observer.sink.closed:raise ValueError('Closed audit sink; original behavior not called')
            if observer._session is None:raise ValueError('Explicit session identity required before behavior')
            if deterministic:raise ValueError('Greedy reference data is not stochastic behavior credit')
            if observer._collect or observer._updating:raise ValueError('Nested behavior/update')
            observer._act_rng_before=rng_digest();observer._collect=True;observer._calls=[]
            try:controls,roll=act(decisions,frame,deterministic=deterministic)
            finally:observer._collect=False
            observer._act_rng_after=rng_digest()
            if roll is None:
                if decisions or observer._calls:raise RuntimeError('Unexpected idle observation')
                observer.idle_frames+=1;return controls,roll
            return controls,observer._emit_behavior(roll,decisions,frame,controls)
        def observe_update(this,rollouts,*,epochs=4):
            observer._begin_update(rollouts)
            try:
                result=update(rollouts,epochs=epochs)
                observer._finish_update(result)
                return result
            finally:observer._updating=False
        try:
            self._patches.append((a,'_credit_audit',False,None));a._credit_audit=self
            for name,call in (('act_controls',observe_act),('_coordinated_features',observe_features),('_evaluate',observe_evaluate),('_choice_distribution',observe_choice),('update',observe_update)):
                self._patch(a,name,call)
            for name in ('optimizer','surrogate_optimizer'):
                opt=getattr(a,name);original=opt.step;key='main' if name=='optimizer' else 'surrogate'
                def observed_step(this,*args,_original=original,_key=key,**kwargs):
                    if not observer._updating or observer._mb is None or observer._mb.get('before_parameters') is None:raise RuntimeError('Optimizer outside observed minibatch')
                    before=parameter_snapshot(a);grad=postclip_gradients(a);rng=rng_digest()
                    attempt=dict(optimizer=_key,status='attempted_execution_unknown',original_returned=False,possibly_partial=True,epoch=observer._epoch,minibatch_index=observer._mb_index)
                    observer._recent_attempts.append(attempt);observer._recent_attempts=observer._recent_attempts[-2:]
                    observer._attempt_counts['attempted']=observer._attempt_counts.get('attempted',0)+1
                    try:
                        result=_original(*args,**kwargs)
                    except BaseException:
                        attempt['status']='original_raised_possibly_partial';raise
                    attempt.update(status='original_returned_observation_pending',original_returned=True)
                    observer._attempt_counts['original_returned']=observer._attempt_counts.get('original_returned',0)+1
                    try:
                        if rng_digest()!=rng:raise RuntimeError('Unexpected RNG consumption in observed original Adam step')
                        change=parameter_delta(a,before)
                    except BaseException:
                        attempt['status']='observation_failed_after_original_returned'
                        observer._attempt_counts['observation_failed']=observer._attempt_counts.get('observation_failed',0)+1
                        raise
                    observer._mb['optimizer_calls'].append(dict(optimizer=_key,parameter_delta=change,postclip_gradients=grad))
                    attempt.update(status='completed_and_observed',possibly_partial=False)
                    return result
                self._patch(opt,'step',observed_step)
        except BaseException:
            self._restore();raise
        return self

    def _restore(self):
        for obj,name,own,old in reversed(self._patches):
            if own:setattr(obj,name,old)
            elif name in obj.__dict__:delattr(obj,name)
        self._patches=[]

    def __exit__(self,typ,value,tb):
        if not self.enabled:return False
        try:self._restore()
        except BaseException:
            if typ is None:raise
        if typ:
            try:self._failure_receipt(typ,value)
            except BaseException:pass
            try:
                if not self.sink.closed:self.sink.__exit__(typ,value,tb)
            except BaseException:pass
        return False

    def _failure_receipt(self,typ,value):
        # Separate bounded receipt: a failed observer can follow an already
        # applied/partially applied optimizer step. This never promises rollback.
        data=dict(schema='sa_credit_audit_failure_v1',run_uid=self.sink.header['run_uid'],source_commit=self.sink.header['source_commit'],
            status='failed',exception_type=typ.__name__,message=str(value)[:512],optimizer_attempt_counts=self._attempt_counts,
            recent_optimizer_attempts=self._recent_attempts,scope='Last two attempts only; normal successful steps are in events.jsonl. Not a complete step ledger.',
            execution_boundary='Original step may have completed or partially executed; no rollback, resume or retry is authorized.')
        try:
            wire=(json.dumps(data,sort_keys=True,allow_nan=False)+'\n').encode()
            if len(wire)>8192:raise ValueError('Failure receipt bound')
            with (self.sink.root/'failure.json').open('xb') as f:f.write(wire)
        except BaseException:
            try:sys.stderr.write('Credit audit failure receipt unavailable; optimizer execution may already be partial/completed. Original error preserved.\n')
            except BaseException:pass

    @contextmanager
    def session(self,identity):
        if not self.enabled:yield self;return
        if self._session is not None:raise ValueError('Nested session')
        if not isinstance(identity,dict) or type(identity.get('seed')) is not int or not identity.get('session_id'):raise ValueError('Session requires seed and globally unique session_id')
        self._session=detached(identity)
        try:yield self
        finally:self._session=None

    def _emit_behavior(self,roll,decisions,frame,controls):
        uids=[];calls={}
        for call in self._calls:
            key=(call['feature']['vehicle_id'],bool(call['kwargs'].get('execution_only',False)))
            if key in calls:raise RuntimeError('Extra behavior evaluate call')
            calls[key]=call
        if len(calls)!=2*len(roll['vehicle_ids']):raise RuntimeError('Two original calls per owner required')
        for i,vid in enumerate(roll['vehicle_ids']):
            first=calls[(vid,True)];second=calls[(vid,False)];actions=detached(roll['actions'])[i];logs=detached(roll['head_logs'])[i]
            if first['result'][0][0][0]!=actions[0] or first['result'][1][0][0]!=logs[0]:raise RuntimeError('First/second execution provenance differs')
            if len(first['distributions'])!=1 or len(second['distributions'])!=3:raise RuntimeError('Unexpected sampled/teacher-forced head counts')
            if first['input']!= {k:v for k,v in second['input'].items() if k in first['input']}:
                # Cache/event fields are intentionally rebuilt. Execution input
                # identity uses only its consumed fields, verified in5B and source.
                for key in ('own','critic','graph','graph_mask','adjacency','f0','m0','node0','prior0'):
                    if first['input'][key]!=second['input'][key]:raise RuntimeError('Execution input changed between passes: '+key)
            state=decisions[vid]['semantic_state'];f=second['feature'];masks=[first['distributions'][0]['actual_mask'][0],*[z['actual_mask'][0] for z in second['distributions'][1:]]]
            payload=dict(seed=self._session['seed'],session_id=str(self._session['session_id']),frame=int(frame),vehicle_id=str(vid),pass_name='two_pass_behavior',
                request_id=str(state['request_id']),node_id=str(f['node_id']),decision_uid=digest(f['decision']),candidates=f['candidates'],masks=masks,action_indices=actions,old_log_probs=logs,
                head_passes=['first_execution','second_cache','second_priority'],causal_prefix=dict(first_execution=dict(owner_position=roll['vehicle_ids'].index(vid),**first['feature']['context']),second_cache_priority=dict(owner_position=roll['vehicle_ids'].index(vid),**f['context']),
                    head_passes=['first_execution','second_cache','second_priority']),
                metadata=dict(session=self._session,act_rng_before_sha256=getattr(self,'_act_rng_before',None),act_rng_after_sha256=getattr(self,'_act_rng_after',None),rng_scope='Whole frame original stochastic act; after-state is not actor-visible prefix',first=first,second=second,selected_control=detached(controls[vid]),stored_baseline=detached(roll['counterfactual_baselines'])[i],
                    stored_global_value=detached(roll['values'])[i],head0_second_pass='teacher-forced; no second sample',categorical_logprob='Actual stochastic conditional head logs; not full-team marginal probability'))
            uid=self.sink.emit('behavior',payload);uids.append(uid);self._behaviors[uid]=payload
        out=dict(roll);out['credit_audit_uids']=uids;return out

    def _begin_update(self,rollouts):
        from scripts.sa_stage5c_credit_audit import update_uid
        if self.sink.closed:raise ValueError('Closed audit sink; original update not called')
        if self._updating or self._collect:raise ValueError('Nested update')
        uids=[u for r in rollouts for u in r.get('credit_audit_uids',[])]
        if len(uids)!=sum(len(r['vehicle_ids']) for r in rollouts) or not uids or len(set(uids))!=len(uids) or any(u not in self._behaviors for u in uids):raise ValueError('Missing/duplicate real behavior UIDs')
        for r in rollouts:
            if len(r.get('request_ids',[]))!=len(r['vehicle_ids']):raise ValueError('Request row identities missing')
            for i,uid in enumerate(r['credit_audit_uids']):
                b=self._behaviors[uid]
                if b['frame']!=r.get('frame_index') or b['vehicle_id']!=r['vehicle_ids'][i] or b['request_id']!=r['request_ids'][i]:raise ValueError('Behavior frame/vehicle/request row differs')
                if b['metadata']['stored_baseline']!=detached(r['counterfactual_baselines'])[i]:raise ValueError('Behavior stored baseline differs')
                expected=b['metadata']['second']['input'];observed=detached({k:v[i:i+1] for k,v in r['data'].items()})
                if set(expected)!=set(observed):raise ValueError('Behavior input keys differ')
                for key,value in expected.items():
                    got=observed[key]
                    if key in ('f1','m1','node1'):got=[got[0][:len(value[0])]]
                    if got!=value:raise ValueError('Behavior training input differs: '+key)
                if any(observed['m1'][0][len(expected['m1'][0]):]):raise ValueError('Padded cache support changed')
        seeds={self._behaviors[u]['seed'] for u in uids}
        if len(seeds)!=1:raise ValueError('Mixed seeds in update')
        before=int(self.agent.completed_updates);seed=next(iter(seeds));index=before+1
        self._update=dict(seed=seed,update_index=index,update_uid=update_uid(self.sink.header['run_uid'],seed,index),completed_updates_before=before,uids=uids,
            T=[z for r in rollouts for z in detached(r['returns'])],cost_targets=[z for r in rollouts for z in detached(r['cost_targets'])],fleet=[int(r['initial_request_count']) for r in rollouts for _ in r['vehicle_ids']],before_parameters=parameter_snapshot(self.agent),steps=[],optimizer_counts={'main':0,'surrogate':0})
        self._updating=True;self._epoch=-1;self._credit_rows=None

    def credit_inputs(self,rollouts,actions,old,targets,team,active,baseline,raw_adv):
        if not self._updating:raise RuntimeError('Missing update envelope')
        u=self._update
        if len(detached(actions))!=len(u['uids']):raise RuntimeError('Scheduled row order differs')
        self._credit_rows=dict(actions=detached(actions),old=detached(old),targets=detached(targets),team=detached(team),active=detached(active),B=detached(baseline),raw=detached(raw_adv),raw_dtype=str(getattr(raw_adv,'dtype','synthetic_primitive')))
        for i,uid in enumerate(u['uids']):
            b=self._behaviors[uid]
            if b['action_indices']!=self._credit_rows['actions'][i] or b['old_log_probs']!=self._credit_rows['old'][i]:raise RuntimeError('Stored behavior data mismatch')

    def credit_normalized(self,advantages):
        from scripts.sa_stage5c_credit_audit import credit_uid
        u=self._update;r=self._credit_rows;normal=detached(advantages);u['credit_uids']=[]
        for i,uid in enumerate(u['uids']):
            payload=dict(seed=u['seed'],update_index=u['update_index'],update_uid=u['update_uid'],behavior_uid=uid,T=u['T'][i],T_team=r['targets'][i],fleet=u['fleet'][i],B=r['B'][i],raw_adv=r['raw'][i],normalized_adv=normal[i],active_heads=r['active'][i],global_value_target=r['team'][i],
                metadata=dict(cost_targets=u['cost_targets'][i],cost_units='Newly admitted full model closure nominal slots/10; actual event state slots/10, not shared per-owner model fee',raw_adv_dtype=r['raw_dtype'],returns_outside_MRO='physical_MC/10',scheduled_team_units='physical_MC/10/fleet',normalization='Original active-head sample.mean/sample.std with default correction1 and clamp1e-6; forced rows retained but actor weight0'))
            if abs(payload['T_team']-payload['T']/payload['fleet'])>2e-6*max(1,abs(payload['T_team'])) or payload['global_value_target']!=payload['T_team']:raise RuntimeError('Unexpected MC/fleet units in actual consumer')
            u['credit_uids'].append(self.sink.emit('credit',payload))

    def epoch_start(self):
        self._epoch+=1;self._mb_index=-1;self._epoch_rng=rng_digest()

    def minibatch_start(self,index,batch):
        self._mb_index+=1;ids=detached(index)
        self._mb=dict(indices=ids,epoch=self._epoch,minibatch_index=self._mb_index,rng_before_sha256=rng_digest(),optimizer_calls=[],input_sha256=digest(batch),input_shapes_dtypes={k:dict(shape=list(getattr(v,'shape',[])),dtype=str(getattr(v,'dtype','primitive'))) for k,v in batch.items()})

    def minibatch_scores(self,logs,entropy,value,surrogate,prefix,enabled,cf,q,ratio,kl,weight,den):
        self._mb['scores']=detached(dict(logs=logs,normalized_entropy=entropy,global_value=value,surrogate=surrogate,prefix_values=prefix,active_heads=enabled,current_baseline=cf,selected_q=q,ratio=ratio,kl=kl,weight=weight,den=den))

    def before_step(self,agent,policy,vf,aux,loss):
        self._mb['losses']=detached(dict(policy=policy,value=vf,surrogate=aux,total=loss));self._mb['before_parameters']=parameter_snapshot(agent)

    def after_step(self,agent,kl):
        calls=self._mb['optimizer_calls'];main=sum(c['optimizer']=='main' for c in calls);sur=sum(c['optimizer']=='surrogate' for c in calls)
        if main!=1 or sur!=int(agent.surrogate_enabled):raise RuntimeError('Original optimizer call counts differ')
        self._mb['compound_parameter_delta']=parameter_delta(agent,self._mb.pop('before_parameters'))
        for c in calls:self._update['optimizer_counts'][c['optimizer']]+=1
        step=digest(dict(update_uid=self._update['update_uid'],epoch=self._epoch,minibatch=self._mb_index))
        self._update['steps'].append(step);self._emit_minibatch(kl,step)

    def minibatch_stop(self,kl):self._emit_minibatch(kl,None)

    def _emit_minibatch(self,kl,step):
        u=self._update;m=self._mb;ids=m['indices']
        self.sink.emit('minibatch',dict(seed=u['seed'],update_index=u['update_index'],update_uid=u['update_uid'],epoch=m['epoch'],minibatch_index=m['minibatch_index'],behavior_uids=[u['uids'][i] for i in ids],credit_uids=[u['credit_uids'][i] for i in ids],rng_before_sha256=m['rng_before_sha256'],rng_after_sha256=rng_digest(),kl=detached(kl),step_observed=step is not None,step_uid=step,kl_early_stop=step is None,
            metadata=dict(indices=ids,input_sha256=m['input_sha256'],input_shapes_dtypes=m['input_shapes_dtypes'],epoch_rng_before_randperm=self._epoch_rng,rng_scope='minibatch before original evaluate to after original step or KL rejection',scores=m.get('scores'),losses=m.get('losses'),optimizer_calls=m['optimizer_calls'],compound_parameter_delta=m.get('compound_parameter_delta'),learning_rate=[g['lr'] for g in self.agent.optimizer.param_groups])))
        self._mb=None

    def _finish_update(self,result):
        u=self._update
        if self._mb is not None or self._credit_rows is None:raise RuntimeError('Unclosed scheduled minibatch/credit hooks')
        if int(self.agent.completed_updates)!=u['completed_updates_before']+1 or int(result['gradient_steps'])!=len(u['steps']):raise RuntimeError('Original update counters differ')
        self.sink.emit('update',dict(seed=u['seed'],update_index=u['update_index'],update_uid=u['update_uid'],completed_updates_before=u['completed_updates_before'],completed_updates_after=int(self.agent.completed_updates),observed_adam_steps=u['optimizer_counts']['main'],step_uids=u['steps'],adam_delta_summary=parameter_delta(self.agent,u['before_parameters']),delta_observation='producer_supplied',
            metadata=dict(optimizer_step_counts=u['optimizer_counts'],total_adam_calls=sum(u['optimizer_counts'].values()),eligible_behavior_uids=u['uids'],eligible_behavior_rows=len(u['uids']),losses=detached(result),boundary='Observed full original update, not an extra optimization; main/surrogate calls counted separately')))
        for uid in u['uids']:del self._behaviors[uid]
        self._update=None


@contextmanager
def credit_audit(agent,sink):
    """Default-off API. No runtime training entry point or automatic activation."""
    if not sink.enabled:
        yield None
        return
    with CreditAuditObserver(agent,sink) as observer:yield observer
