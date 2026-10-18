# -*- coding: utf-8 -*-
"""Pure scoring/capture instrumentation for frozen stream v2 controllers.

This module has NO environment constructor, environment step, rollout runner,
optimizer update, or training entry point. CaptureTap can be installed around an
independently authorized final64 reference rollout. score_snapshot consumes only
captured tensors and teacher-forces the reference prefix for all checkpoints.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path
from types import MethodType
from typing import Any, Mapping
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_ROOT=ROOT
DEFAULT_RUN=DEFAULT_ROOT/'artifacts/joint_stream_learning/20261016T162100Z_confirmation'
HEAD_NAMES=('execution','cache','prepare_priority')


def tensor_digest(data: Mapping[str,torch.Tensor]) -> str:
    h=hashlib.sha256()
    for k,v in sorted(data.items()):
        x=v.detach().cpu().contiguous()
        h.update(k.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()


def module_digest(agent) -> str:
    return tensor_digest(agent.state_dict())


def rng_digest() -> str:
    h=hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()),protocol=4))
    h.update(torch.get_rng_state().numpy().tobytes())
    return h.hexdigest()


@contextmanager
def preserved_rng():
    py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state().clone()
    try:yield
    finally:random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu)


def jsonable(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None:return x
    raise TypeError(f'Non-serializable captured context: {type(x).__name__}')


def json_digest(x) -> str:
    return hashlib.sha256(json.dumps(jsonable(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _clone_batch(batch):
    if not all(isinstance(v,torch.Tensor) for v in batch.values()):raise TypeError('Expected tensor-only _evaluate batch')
    return {k:v.detach().cpu().clone() for k,v in batch.items()}


@contextmanager
def patched_method(obj,name,call):
    # Restore original instance lookup, not a permanent copy of a class method.
    had=name in obj.__dict__;old=obj.__dict__.get(name)
    setattr(obj,name,MethodType(call,obj))
    try:yield
    finally:
        if had:setattr(obj,name,old)
        else:delattr(obj,name)


def _head_summary(logits,mask,learned,original_learned,prior,reference_action):
    logits=logits.detach().cpu();mask=mask.detach().cpu().bool()
    if logits.ndim!=2 or logits.shape[0]!=1:raise ValueError('Capture/scoring API uses one vehicle row per call')
    ids=torch.where(mask[0])[0]
    if not len(ids):raise ValueError('No valid candidate')
    if not bool(torch.isfinite(logits[mask]).all()):raise ValueError('Non-finite valid logits')
    # Explicit deterministic tie break equals torch.argmax: lowest candidate index.
    ranking=sorted(ids.tolist(),key=lambda i:(-float(logits[0,i]),i))
    winner=ranking[0];runner=ranking[1] if len(ranking)>1 else None
    prior_winner=min(ids.tolist(),key=lambda i:(-float(prior[0,i]),i))
    learned_winner=min(ids.tolist(),key=lambda i:(-float(learned[0,i]),i))
    residual=logits-learned-prior
    return dict(valid_candidate_ids=ids.tolist(),valid_candidate_count=len(ids),
        actual_mask=mask[0].tolist(),valid_total_logits=[float(logits[0,i]) for i in ids],
        valid_learned_logits=[float(learned[0,i]) for i in ids],
        valid_original_head_logits=[float(original_learned[0,i]) for i in ids],
        valid_public_prior=[float(prior[0,i]) for i in ids],
        valid_other_adjustment=[float(residual[0,i]) for i in ids],
        max_abs_other_adjustment=float(residual[mask].abs().max()),
        greedy_action=winner,runner_up_action=runner,
        top2_margin=float(logits[0,winner]-logits[0,runner]) if runner is not None else None,
        reference_action=int(reference_action),reference_is_valid=bool(mask[0,int(reference_action)]),
        reference_action_probability=float(torch.softmax(logits.masked_fill(~mask,-1e9),-1)[0,int(reference_action)]),
        public_prior_argmax=prior_winner,learned_only_argmax=learned_winner,
        overrides_prior=winner!=prior_winner,prior_changes_learned_winner=winner!=learned_winner,
        tie_break='lowest valid candidate index; exact software control, not a new trained/heuristic baseline')


def trace_evaluate(agent,batch,kwargs,*,zero_learned=False,call=None):
    """Trace existing forward only. No RNG, optimizer, controller memory or tensor mutation.

    kwargs must always be deterministic. For same-state scoring actions=reference
    is mandatory. Reference capture may instead use its actual greedy path.
    """
    if not kwargs.get('deterministic',False):raise ValueError('Only deterministic scoring/capture authorized')
    if kwargs.get('target',False):raise ValueError('Diagnostic compares online heads, never DQN target heads')
    batch_before=tensor_digest(batch);weights_before=module_digest(agent);rng_before=rng_digest()
    raw_heads={};effective_heads={};distributions=[];handles=[]
    original_choice=agent._choice_distribution
    original_evaluate=call or agent._evaluate
    for j,head in enumerate(agent.heads):
        def hook(module,inputs,output,j=j):
            raw_heads[j]=output.detach().cpu().reshape(-1).clone()
            effective=torch.zeros_like(output) if zero_learned else output
            effective_heads[j]=effective.detach().cpu().reshape(-1).clone()
            return effective if zero_learned else None
        handles.append(head.register_forward_hook(hook))
    def choice(this,logits,mask,deterministic):
        if not deterministic:raise ValueError('DQN epsilon/stochastic path forbidden')
        j=len(distributions)
        if j not in effective_heads:raise RuntimeError('Head output not captured before distribution')
        lm=torch.zeros_like(logits.detach().cpu());orig=torch.zeros_like(lm)
        cm=mask.detach().cpu().bool()
        if int(cm.sum())!=len(effective_heads[j]):raise RuntimeError('Head/mask flattened cardinality mismatch')
        lm[cm]=effective_heads[j];orig[cm]=raw_heads[j]
        prior=batch['prior0'].detach().cpu().clone() if j==0 else torch.zeros_like(lm)
        distributions.append(dict(logits=logits.detach().cpu().clone(),mask=cm.clone(),learned=lm,original_learned=orig,prior=prior))
        return original_choice(logits,mask,deterministic)
    try:
        with patched_method(agent,'_choice_distribution',choice),torch.no_grad():
            result=original_evaluate(batch,**kwargs)
    finally:
        for handle in handles:handle.remove()
    actions=result[0].detach().cpu().clone()
    for j,z in enumerate(distributions):
        z['summary']=_head_summary(**z,reference_action=actions[0,j])
    guards=dict(batch_unchanged=tensor_digest(batch)==batch_before,
        weights_unchanged=module_digest(agent)==weights_before,rng_unchanged=rng_digest()==rng_before)
    if not all(guards.values()):raise RuntimeError(f'Purity guard failed: {guards}')
    return result,dict(heads=[z['summary'] for z in distributions],actions=actions.tolist(),guards=guards,
        head_count=len(distributions),weights_sha256=weights_before,batch_sha256=batch_before)


def _controller_context(agent):
    return {name:deepcopy(getattr(agent,name,None)) for name in ('request_memory','target_memory','commitments','_team_targets','_calendar')}


class CaptureTap:
    """Attach only around authorized reference64 agent; never owns/steps an env.

    Records frame0 plus earliest max_extra_frames whole physical decision frames
    with >=2 actual valid choices in any first/second-pass head. Both passes and
    all reference vehicles in each selected frame are retained. The reference
    policy sees the original forward function, RNG, weights, actions and booking.
    """
    def __init__(self,agent,witness,*,max_extra_frames=8):
        self.agent=agent;self.witness=jsonable(witness);self.max_extra_frames=int(max_extra_frames)
        self.snapshots=[];self.selected_frames=[];self._contexts=[];self._frame=None;self._calls=[];self._feature=None;self._tracing=False

    def __enter__(self):
        agent=self.agent;original_act=agent.act_controls;original_features=agent._coordinated_features;original_evaluate=agent._evaluate
        def features(this,decision,frame,reservations):
            answer=original_features(decision,frame,reservations)
            if self._tracing:
                data,candidates,vid,node,work=answer;s=decision['semantic_state']
                self._feature=dict(frame=int(frame),vehicle_id=vid,node_id=node,work=float(work),candidates=deepcopy(candidates),
                    semantic_identity={k:deepcopy(s.get(k)) for k in ('request_id','execution_location','execution_state_version','execution_state_payload_bytes','current_node_work_units_progress','current_node_work_units_required')},
                    decision_sha256=json_digest(decision),full_decision=deepcopy(decision),reservations=deepcopy(reservations),controller=_controller_context(this))
            return answer
        def evaluate(this,data,**kwargs):
            if not self._tracing:return original_evaluate(data,**kwargs)
            if self._feature is None:raise RuntimeError('No matching feature/candidate capture')
            feature=deepcopy(self._feature)
            if feature['frame']!=self._frame:raise RuntimeError('Feature frame mismatch')
            frozen=_clone_batch(data)
            result,trace=trace_evaluate(this,data,kwargs,call=original_evaluate)
            phase='first_execution_pass' if kwargs.get('execution_only',False) else 'second_cache_event_pass'
            actions=result[0].detach().cpu().clone()
            capture=dict(witness=self.witness,frame=self._frame,call_index=len(self._calls),phase=phase,
                feature=feature,batch=frozen,reference_actions=actions,reference_trace=trace)
            if phase=='second_cache_event_pass':
                capture['forward_rollout']={name:result[i].detach().cpu().clone() for name,i in [('actions',0),('head_logs',1),('entropy',2),('values',3),('prefix_values',5),('active_heads',6),('counterfactual_baselines',7),('selected_q',8)]}
            capture['state_identity']=dict(batch_sha256=tensor_digest(frozen),candidates_sha256=json_digest(feature['candidates']),
                context_sha256=json_digest(feature['controller']),decision_sha256=feature['decision_sha256'],
                reference_prefix_sha256=tensor_digest({'actions':actions}))
            self._calls.append(capture)
            return result
        def act(this,decisions,frame,*,deterministic=False):
            if not deterministic:raise ValueError('Reference capture requires fixed greedy final64')
            if self._tracing:raise RuntimeError('Nested act_controls unsupported')
            self._frame=int(frame);self._calls=[];self._feature=None
            extra=sum(f!=0 for f in self.selected_frames)
            self._tracing=(int(frame)==0 or extra<self.max_extra_frames)
            try:result=original_act(decisions,frame,deterministic=True)
            finally:self._tracing=False
            eligible=any(any(h['valid_candidate_count']>=2 for h in c['reference_trace']['heads']) for c in self._calls)
            if self._calls and (int(frame)==0 or eligible):
                self.selected_frames.append(int(frame));self.snapshots.extend(self._calls)
            self._calls=[]
            return result
        for name,fn in [('_coordinated_features',features),('_evaluate',evaluate),('act_controls',act)]:
            cm=patched_method(agent,name,fn);cm.__enter__();self._contexts.append(cm)
        return self

    def __exit__(self,typ,value,tb):
        for cm in reversed(self._contexts):cm.__exit__(typ,value,tb)
        self._contexts=[]

    def bind_episode_labels(self,rollouts,*,fleet=8):
        by_frame={int(r['frame_index']):r for r in rollouts}
        for snapshot in self.snapshots:
            if snapshot['phase']!='second_cache_event_pass':continue
            r=by_frame[snapshot['frame']];vid=snapshot['feature']['vehicle_id'];i=r['vehicle_ids'].index(vid)
            if int(r['initial_request_count'])!=fleet:raise ValueError('Fleet target normalization mismatch')
            data={k:v[i:i+1].detach().cpu().clone() for k,v in r['data'].items()}
            # The controller pads per-vehicle cache arrays at the end of each frame.
            n=snapshot['batch']['m1'].shape[1]
            for key in ('f1','m1','node1'):data[key]=data[key][:,:n]
            if tensor_digest(data)!=snapshot['state_identity']['batch_sha256']:raise ValueError('Episode rollout/captured batch mismatch')
            for name in ('actions','head_logs','active_heads','counterfactual_baselines'):
                v=r[name][i:i+1].detach().cpu()
                if not torch.equal(v,snapshot['forward_rollout'][name]):raise ValueError('Episode behavior/capture mismatch: '+name)
            snapshot['episode_labels']={name:r[name][i:i+1].detach().cpu().clone() for name in ('returns','team_credit_targets','credit_targets','cost_targets')}
            snapshot['episode_labels'].update(initial_request_count=fleet,episode_end=bool(r['episode_end']))
        return self

    def save(self,path):
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        torch.save(dict(schema='same_state_reference_capture_v1',witness=self.witness,selected_frames=self.selected_frames,
            selection='frame0 + earliest8physicalframes with any actualhead validchoices>=2; allvehicles/bothpasses kept',
            snapshots=self.snapshots),path)


def load_exact_models(root,source_run,method,seed):
    """Reconstruct exact0 and load32/64 without changing any caller RNG state."""
    root=Path(root);source_run=Path(source_run)
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from scripts.train_joint_stream_control import make_agent,actor_hash
    from scripts.train_joint_tail_control import tensor_hash
    manifest=json.loads((source_run/method/'manifest.json').read_text())
    protocol=json.loads((source_run/'protocol/manifest.json').read_text());budget=protocol['episodes']
    identities=[x for x in manifest['checkpoints'] if x['seed']==seed]
    if len(identities)!=2:raise ValueError('Expected frozen32/64checkpoint pair')
    before=rng_digest();models={};provenance={}
    with preserved_rng():
        initial=make_agent(method,seed,budget)
        if tensor_hash(initial)!=identities[0]['initial_tensor_sha256'] or actor_hash(initial)!=identities[0]['initial_actor_sha256']:
            raise ValueError('Exact initial reconstruction identity mismatch')
        models['0']=initial;provenance['0']=dict(full_tensor_sha256=tensor_hash(initial),heads_sha256=actor_hash(initial),origin='exactseed/optionsreconstruction')
        for ep in [32,64]:
            meta=next(x for x in identities if x['episode']==ep);p=source_run/method/meta['path']
            if hashlib.sha256(p.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Checkpoint file hash mismatch')
            model=type(initial).from_checkpoint(p)
            if tensor_hash(model)!=meta['trained_tensor_sha256']:raise ValueError('Checkpoint tensor hash mismatch')
            models[str(ep)]=model;provenance[str(ep)]=dict(path=str(p),file_sha256=meta['sha256'],full_tensor_sha256=tensor_hash(model),heads_sha256=actor_hash(model))
    if before!=rng_digest():raise RuntimeError('Model loading changed caller RNG')
    return models,provenance


def score_snapshot(snapshot,models,*,include_prior_only=True,require_reference64_parity=True):
    """Same tensor batch/reference-prefix, masked online-head scoring only.

    In second pass, reference actions teacher-force all previous heads. Reported
    greedy_action is the hypothetical argmax under that fixed reference prefix,
    not a free-running alternative policy path. No _coordinated_features call.
    """
    batch=_clone_batch(snapshot['batch']);actions=snapshot['reference_actions'].detach().cpu().clone()
    expected=snapshot['state_identity']
    if tensor_digest(batch)!=expected['batch_sha256']:raise ValueError('Captured input modified')
    if json_digest(snapshot['feature']['candidates'])!=expected['candidates_sha256']:raise ValueError('Candidates modified')
    if json_digest(snapshot['feature']['controller'])!=expected['context_sha256']:raise ValueError('Controller context modified')
    kwargs=dict(deterministic=True,actions=actions)
    first=snapshot['phase']=='first_execution_pass'
    if first:kwargs['execution_only']=True
    else:kwargs['return_details']=True
    variants={};mask_by_head=None
    for label,model in [*[(str(ep),models[str(ep)]) for ep in [0,32,64]],*([('prior_only_64',models['64'])] if include_prior_only else [])]:
        _,trace=trace_evaluate(model,batch,kwargs,zero_learned=label=='prior_only_64')
        masks=[h['actual_mask'] for h in trace['heads']]
        if mask_by_head is None:mask_by_head=masks
        if masks!=mask_by_head:raise RuntimeError('Variants have different actual head masks despite reference prefix')
        variants[label]=trace
    if require_reference64_parity:
        ref=snapshot['reference_trace'];now=variants['64']
        for j,(a,b) in enumerate(zip(ref['heads'],now['heads'])):
            for f in ['actual_mask','valid_total_logits','valid_learned_logits','valid_public_prior','greedy_action']:
                if a[f]!=b[f]:raise RuntimeError(f'Reference64 forward parity failed head{j}/{f}')
        if len(ref['heads'])!=len(now['heads']):raise RuntimeError('Reference64 headcount mismatch')
    comparison={}
    for a,b in [('0','32'),('32','64'),('0','64'),('prior_only_64','64')]:
        if a not in variants:continue
        comparison[f'{a}_to_{b}']=[dict(head=HEAD_NAMES[j],action_changed=x['greedy_action']!=y['greedy_action'],
            from_action=x['greedy_action'],to_action=y['greedy_action'],from_margin=x['top2_margin'],to_margin=y['top2_margin'],
            max_abs_valid_logit_change=max(abs(u-v) for u,v in zip(x['valid_total_logits'],y['valid_total_logits'])))
            for j,(x,y) in enumerate(zip(variants[a]['heads'],variants[b]['heads']))]
    return dict(schema='same_state_scores_v1',witness=snapshot['witness'],frame=snapshot['frame'],phase=snapshot['phase'],
        vehicle_id=snapshot['feature']['vehicle_id'],node_id=snapshot['feature']['node_id'],state_identity=expected,
        reference_actions=actions.tolist(),teacher_forcing='all previous heads fixed to reference64 chosen action; no alternate trajectory',
        candidates=jsonable(snapshot['feature']['candidates']),variants=variants,comparisons=comparison,
        guards=dict(reference64_parity=require_reference64_parity,actual_masks_identical=True,environment_steps=0,optimizer_steps=0))


def design():
    return dict(protocol='same_state_scoring_design_v1',status='tool_only_not_executed_on_environment',
        api=['models,provenance=load_exact_models(root,source_run,method,seed)',
             'with CaptureTap(models["64"],witness,max_extra_frames=8) as tap: authorized_reference_runner(models["64"])',
             'tap.bind_episode_labels(rollouts,fleet=8); tap.save(new_snapshot_path)',
             'scores=[score_snapshot(snapshot,models) for snapshot in tap.snapshots]'],
        reference_budget='rootcontract:15predeclaredfinal64reference sessions≤7200physicalframes; scorer doesnotown runner/budget',
        state_identity='tensor batch+candidates+controllercontext+referenceprefix SHA; allsame acrossscoringvariants',
        first_pass='execution_only input exactlycaptured before two-pass teamtargets settled',
        second_pass='updatedcache/state feature batch plus referencefirstpass chosenexecution; teacherforcecacheprefix too for eventheadcomparability',
        prior_control='Zerolearnedscoreviaforwardhook; unchangedmasks/publicprior;lowestvalidindex tie break.Cache/event no numericprior => thisissoftwareinvariancecontrol,notclaimedstrongbaseline.',
        dqn='UseactualStreamDQNAgent MRO and online heads,target=False,deterministic=True; noepsilon sampling; replay/optimizer untouched.',
        software_selfchecks=['py_compile/import withoutenvconstruction',
            'Capturedreference64forward logits/masks/argmax bitexactwhen rescored on samebatch',
            'Beforeafterbatch/weights/RNG digests equal pertrace',
            'Model load caller Python/NumPy/TorchRNG identical',
            'Dynamicmasks identical across0/32/64/prior under referenceprefix',
            'Explicitlyforced firstpass vssecondpass checked; headcount1vs3',
            'Monkeypatch/hooks restored evenonexception',
            'Zeroheadscontrol validlearnedlogits allzero,validtotallogits=publicprior withinfloatrounding',
            'Hookoff/on reference controls andfullphysicalrows mustmatch independentlyauthorizedqualification; toolalonehasnotrun it'],
        limitations=['No isolatedlearningcausality fromactionflipalone','Allstates reference64 on-policy; comparefrozenconditionalscoringnot otherpoliciesown distributions',
            'Actual head masks use referenceautoregressiveprefix; freevariantprefixwouldnotbe samecondition',
            'No costs/returns computedbyscorer; linkscorechanges toreferenceexistingevents separately',
            'No training/weightsupdate/rewardchange/checkpointselection/newfullpanel'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design-output',type=Path,required=True,help='write design only; never runs an environment')
    a=p.parse_args();a.design_output.write_text(json.dumps(design(),ensure_ascii=False,indent=2)+'\n')
    print(f'Wrote design only: {a.design_output}')
