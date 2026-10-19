"""Frozen Stage5 B reference-only diagnostic; no training or alternate trajectories.

Workflow: contract -> qualify (cached software only) -> prepare -> run -> status.
All commands write only a new output root. A review receipt is required at prepare.
Qualification: <=16 old snapshots / <=112 conditional forward calls, zero env.
Run: exactly144 existing reference cells, <=69120 frames, <=20736 snapshots;
7 conditional variants per snapshot. Failure consumes the claim; no resume/retry.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
ARMS = {'sa_legacy_clip': False, 'sa_group_clip': True}
SEEDS = (7, 17, 27)
EPISODES = (0, 32, 64)
CASES = tuple('dev_' + x for x in ('000','018','021','022','038','043','045','046'))
CAPS = dict(reference_sessions=144, reference_frames=69120, frames_per_session=480,
            saved_calls=20736, second_pass_calls=10368, scoring_calls=145152,
            qualification_snapshots=16, qualification_scoring_calls=112,
            training_sessions=0, backward_calls=0, optimizer_steps=0)
SCHEMA = 'sa_stage5_same_state_v1'
IDENTITY = {'arm','agent','rule','seed','episode','case_id'}
HELPERS = ('scripts/joint_stream_same_state_scoring.py', 'scripts/diagnose_sa_midlate.py',
           'scripts/diagnose_joint_stream_learning.py', 'scripts/train_sa_group_clip_pair.py',
           'scripts/train_joint_stream_control.py', 'scripts/train_joint_tail_control.py')
BOUNDARY = ('Post-selected development diagnostic. Six models share the captured inputs and '
            'teacher-forced reference prefix; this is not an alternate trajectory, causal '
            'effect, new reward comparison, convergence estimate, or paper promotion. '
            'None-prepare priority choices cannot be merged without checking team order.')


def utc(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode() + b'\n'
def clean_payload(x): return {k:v for k,v in x.items() if k not in IDENTITY}
def digest(x): return hashlib.sha256(canonical(x)).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def write(path, value, *, exclusive=True):
    with Path(path).open('x' if exclusive else 'w', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
def bound(path):
    p=Path(path).resolve(); return dict(path=str(p), sha256=sha(p))
def check_bound(item):
    if sha(item['path']) != item['sha256']: raise ValueError('Bound file changed: '+item['path'])
def rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def cell_key(x): return (x['seed'], x['episode'], x['case_id'])
def model_key(arm, episode): return arm + ':' + str(episode)
def quantile_frames(n):
    if type(n) is not int or not 1 <= n <= 480: raise ValueError('Invalid original steps')
    return sorted({i*(n-1)//8 for i in range(9)})
def git(root, *args):
    return subprocess.check_output(['git',*args], cwd=root, text=True).strip()
def clean_head(root):
    if git(root,'status','--porcelain'): raise ValueError('Clean Git worktree required')
    return git(root,'rev-parse','HEAD')
def unique_index(items, key):
    result={}
    for x in items:
        k=key(x)
        if k in result: raise ValueError('Duplicate evidence key: '+repr(k))
        result[k]=x
    return result


class Budget:
    """Charge before the operation; a failed operation does not refund its charge."""
    def __init__(self, caps): self.caps=dict(caps); self.used={k:0 for k in caps}
    def charge(self, key, amount=1):
        if type(amount) is not int or amount < 0 or self.used[key]+amount > self.caps[key]:
            raise RuntimeError('Frozen budget exceeded before '+key)
        self.used[key]+=amount


def verify_shape(plan):
    if plan['schema'] != SCHEMA or plan['budgets'] != CAPS: raise ValueError('Unapproved schema/budget')
    if plan['arms'] != ARMS or plan['scoring_conditions'] != [model_key(a,e) for a in ARMS for e in EPISODES]+['reference_prior_only']:
        raise ValueError('Scoring model set changed')
    expected={(a,s,e,c) for s in SEEDS for a in ARMS for e in EPISODES for c in CASES}
    actual={(x['arm'],x['seed'],x['episode'],x['case_id']) for x in plan['sessions']}
    if len(plan['sessions']) != 144 or actual != expected: raise ValueError('Frozen session list changed')
    for x in plan['sessions']:
        if x['frames'] != quantile_frames(x['expected_summary']['steps']): raise ValueError('Quantile plan changed')
        if x['expected_summary']['steps'] != x['expected_ledger']['receipt_frames']: raise ValueError('Original frame ledger mismatch')
        if x['expected_summary_sha256'] != digest(x['expected_summary']): raise ValueError('Original row changed')
        if x['identity'] != dict(arm=x['arm'],agent='sa_ghmappo',rule=f"{x['arm']}_seed{x['seed']}",seed=x['seed'],episode=x['episode'],case_id=x['case_id']):
            raise ValueError('Reference identity changed')
    if set(plan['checkpoints']) != {str(s) for s in SEEDS}: raise ValueError('Checkpoint seeds changed')
    for by_model in plan['checkpoints'].values():
        if set(by_model) != {model_key(a,e) for a in ARMS for e in EPISODES}: raise ValueError('Checkpoint set changed')
        for label, cp in by_model.items():
            arm, ep=label.split(':'); ep=int(ep)
            if cp['completed_updates'] != ep//4 or cp['expected_updates'] != ep//4 or cp['options_flag'] is not ARMS[arm]:
                raise ValueError('Checkpoint option/update identity mismatch')


def build_plan(args):
    root=args.repo_root.resolve(); source=args.source_run.resolve(); out=args.output_root.resolve()
    head=clean_head(root)
    if not Path(__file__).resolve().is_relative_to(root): raise ValueError('Install reviewed runner before freezing')
    result=read(source/'result.json'); audit=read(source/'audit/status.json'); old=read(source/'contract.json')
    if result['status']!='complete' or audit['status']!='pass' or not audit['integrity_passed']: raise ValueError('Stage4 incomplete/unverified')
    if result['contract_sha256']!=sha(source/'contract.json') or audit['contract_sha256']!=result['contract_sha256']:
        raise ValueError('Stage4 contract binding changed')
    if result['audit_status_sha256']!=sha(source/'audit/status.json'): raise ValueError('Stage4 audit changed')
    if result['source_commit']!=old['source_commit'] or audit['source_commit']!=old['source_commit']: raise ValueError('Stage4 source identity differs')
    if len(old['source_fingerprints'])!=130: raise ValueError('Expected original130 source files')
    for name,h in old['source_fingerprints'].items():
        if sha(root/name)!=h: raise ValueError('Original source changed: '+name)
    bindings=[bound(source/'result.json'),bound(source/'audit/status.json'),bound(source/'contract.json')]
    sessions=[]; checkpoints={str(s):{} for s in SEEDS}
    for arm in ARMS:
        home=source/'arms'/arm; man=read(home/'manifest.json')
        if man['status']!='complete' or man['contract_sha256']!=result['contract_sha256'] or man['source_commit']!=old['source_commit']:
            raise ValueError('Arm provenance invalid')
        if man['separate_policy_value_clip'] is not ARMS[arm] or man['git_dirty']: raise ValueError('Arm option/provenance mismatch')
        if sha(home/'dev.jsonl')!=man['output_sha256']['dev.jsonl']: raise ValueError('Original dev rows changed')
        ledger_path=source/'audit'/f'{arm}_ledger.jsonl'
        if sha(ledger_path)!=audit['output_sha256'][ledger_path.name]: raise ValueError('Audit ledger changed')
        bindings.extend([bound(home/'manifest.json'),bound(home/'dev.jsonl'),bound(ledger_path)])
        summaries=unique_index(rows(home/'dev.jsonl'),cell_key); ledger=unique_index(rows(ledger_path),cell_key)
        cps=unique_index(man['checkpoints'],lambda x:(x['seed'],x['episode']))
        for seed in SEEDS:
            for ep in EPISODES:
                cp=deepcopy(cps[seed,ep]); check_bound(dict(path=str(home/cp['path']),sha256=cp['sha256']))
                cp['path']=str((home/cp['path']).resolve()); checkpoints[str(seed)][model_key(arm,ep)]=cp
                for cid in CASES:
                    row=summaries[seed,ep,cid]; led=ledger[seed,ep,cid]
                    sessions.append(dict(arm=arm,seed=seed,episode=ep,case_id=cid,
                        identity={k:row[k] for k in IDENTITY}, frames=quantile_frames(row['steps']),
                        expected_summary=row,expected_summary_sha256=digest(row),expected_ledger=led))
    # Stable seed-major schedule; never sort by outcomes.
    sessions.sort(key=lambda x:(SEEDS.index(x['seed']),list(ARMS).index(x['arm']),EPISODES.index(x['episode']),CASES.index(x['case_id'])))
    protocol=dict(path=old['source_protocol'],sha256=old['source_protocol_sha256'])
    check_bound(protocol); check_bound(old['protocol_data'])
    bindings.extend([protocol,old['protocol_data']])
    for name,h in read(protocol['path'])['assets'].items():
        asset=dict(path=str(Path(protocol['path']).parent/name),sha256=h); check_bound(asset); bindings.append(asset)
    scripts={name:sha(root/name) for name in HELPERS}
    scripts[str(Path(__file__).resolve().relative_to(root))]=sha(__file__)
    plan=dict(schema=SCHEMA,status='planned',created_at=utc(),repo_root=str(root),output_root=str(out),
        source_run=str(source),source_commit=head,stage4_source_commit=old['source_commit'],
        stage4_source_fingerprints=old['source_fingerprints'],script_sha256=scripts,bindings=bindings,
        source_protocol=protocol,protocol_data=old['protocol_data'],arms=ARMS,budgets=CAPS,
        scoring_conditions=[model_key(a,e) for a in ARMS for e in EPISODES]+['reference_prior_only'],
        sessions=sessions,checkpoints=checkpoints,
        qualification_capsule=bound(args.qualification_capsule),qualification_seed=7,
        qualification_selection='first8 first_execution_pass and first8 second_cache_event_pass in the SHA-bound seed7 SA64 capsule',
        boundary=BOUNDARY,reference_sampling='sorted({i*(N-1)//8 for i in range(9)}), N=original arm/seed/episode/case actual steps',
        capture_policy='all live vehicles, both passes; full team controls once per frame; no extra anchors',
        forbidden=['training','update','learn','backward','autograd.grad','optimizer.step','alternate-policy rollout','resume','automatic rerun'])
    capsule=Path(args.qualification_capsule).resolve()
    capsule_receipt=capsule.parent/'result.json'
    proof=read(capsule_receipt)
    if (proof['status']!='pass' or proof['witness']['seed']!=7 or proof['witness']['method']!='sa_ghmappo'
            or proof['witness']['case_id']!='dev_046' or proof['output_sha256']['snapshots.pt']!=sha(capsule)
            or proof['provenance']['64']['full_tensor_sha256']!=checkpoints['7'][model_key('sa_legacy_clip',64)]['trained_tensor_sha256']):
        raise ValueError('Qualification capsule is not the audited original seed7/dev046/final64 reference')
    plan['qualification_provenance']=bound(capsule_receipt)
    plan['qualification_reference_tensor_sha256']=proof['provenance']['64']['full_tensor_sha256']
    plan['bindings'].append(plan['qualification_provenance'])
    plan['run_command']=[sys.executable,str(Path(__file__).resolve()),'run','--output_root',str(out)]
    verify_shape(plan); out.mkdir(parents=True,exist_ok=False); write(out/'plan.json',plan)
    return dict(status='planned',plan=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'))


def verify(plan, out):
    verify_shape(plan)
    root=Path(plan['repo_root'])
    if Path(plan['output_root']).resolve()!=out.resolve(): raise ValueError('Output root changed')
    if clean_head(root)!=plan['source_commit']: raise ValueError('Frozen HEAD changed')
    for name,h in {**plan['stage4_source_fingerprints'],**plan['script_sha256']}.items():
        if sha(root/name)!=h: raise ValueError('Frozen source changed: '+name)
    for item in plan['bindings']: check_bound(item)
    check_bound(plan['qualification_capsule'])
    for by in plan['checkpoints'].values():
        for cp in by.values(): check_bound(cp)
    # Embedded summaries and ledgers remain bound to authoritative small files.
    source=Path(plan['source_run'])
    for arm in ARMS:
        summaries=unique_index(rows(source/'arms'/arm/'dev.jsonl'),cell_key)
        ledgers=unique_index(rows(source/'audit'/f'{arm}_ledger.jsonl'),cell_key)
        for s in (x for x in plan['sessions'] if x['arm']==arm):
            if s['expected_summary']!=summaries[cell_key(s)] or s['expected_ledger']!=ledgers[cell_key(s)]:
                raise ValueError('Embedded source evidence differs')


def runtime(root):
    sys.path.insert(0,str(root))
    import torch
    from scripts import joint_stream_same_state_scoring as score
    from scripts import train_sa_group_clip_pair as train
    from scripts.diagnose_sa_midlate import FixedFrameCapture
    from scripts.train_joint_stream_control import actor_hash
    from scripts.train_joint_tail_control import tensor_hash
    from src.agents.sa_ghmappo_agent import StreamSAGHMAPPOAgent
    return dict(torch=torch,score=score,train=train,Capture=FixedFrameCapture,actor_hash=actor_hash,
                tensor_hash=tensor_hash,Agent=StreamSAGHMAPPOAgent,base_trace=score.trace_evaluate)


def forbidden(*args,**kwargs): raise RuntimeError('Stage5 forbids training/update/learn/backward/grad/optimizer.step')


@contextmanager
def no_learning(rt,models):
    """Guard all live optimizers, gradient entry points, and update/learn methods."""
    torch=rt['torch']; score=rt['score']; originals=[]
    with ExitStack() as stack:
        for m in models.values():
            for name in ('update','learn'):
                stack.enter_context(score.patched_method(m,name,forbidden))
            for optname in ('optimizer','surrogate_optimizer'):
                stack.enter_context(score.patched_method(getattr(m,optname),'step',forbidden))
        for owner,name in [(torch.Tensor,'backward'),(torch.autograd,'backward'),(torch.autograd,'grad'),
                           (torch.optim.Optimizer,'step'),(torch.optim.Adam,'step'),(torch.optim.AdamW,'step')]:
            originals.append((owner,name,getattr(owner,name))); setattr(owner,name,forbidden)
        try:
            with torch.no_grad(): yield
        finally:
            for owner,name,value in reversed(originals): setattr(owner,name,value)


def learning_hash(rt,agent):
    return rt['train'].object_hash(dict(weights=agent.state_dict(),optimizer=agent.optimizer.state_dict(),
        surrogate_optimizer=agent.surrogate_optimizer.state_dict(),gradients={n:p.grad for n,p in agent.named_parameters()},
        options=agent.options,train_mode=agent.training))


def load_models(rt,plan,seed):
    score=rt['score']; models={}; before=score.rng_digest()
    with score.preserved_rng(),no_learning(rt,{}):
        for label,cp in plan['checkpoints'][str(seed)].items():
            m=rt['Agent'].from_checkpoint(Path(cp['path'])); arm,ep=label.split(':'); ep=int(ep)
            if m.separate_policy_value_clip is not ARMS[arm] or int(m.completed_updates)!=ep//4:
                raise ValueError('Loaded options/update mismatch')
            if rt['tensor_hash'](m)!=cp['trained_tensor_sha256'] or rt['actor_hash'](m)!=cp['trained_actor_sha256']:
                raise ValueError('Loaded checkpoint tensors differ')
            models[label]=m
    if score.rng_digest()!=before: raise ValueError('Checkpoint loading changed caller RNG')
    if score.module_digest(models[model_key('sa_legacy_clip',0)])!=score.module_digest(models[model_key('sa_group_clip',0)]):
        raise ValueError('Initial arms have different tensors')
    return models


def check_reference_parity(reference, actual):
    fields=('actual_mask','valid_total_logits','valid_learned_logits','valid_public_prior','greedy_action')
    if len(reference['heads'])!=len(actual['heads']): raise ValueError('Reference head count changed')
    for a,b in zip(reference['heads'],actual['heads']):
        for field in fields:
            if a[field]!=b[field]: raise ValueError('Reference conditional parity failed: '+field)
        for field in ('distribution','valid_candidate_surrogate_raw','valid_candidate_surrogate_estimates','valid_candidate_q','prefix_value'):
            if field in a and a[field]!=b.get(field): raise ValueError('Rich reference conditional parity failed: '+field)
    if 'forward_outputs' in reference and reference['forward_outputs']!=actual.get('forward_outputs'):
        raise ValueError('Reference value/baseline/entropy outputs differ')


def score_seven(rt,snapshot,models,reference_label,budget,*,qualification=False):
    score=rt['score']; train=rt['train']; batch=score._clone_batch(snapshot['batch'])
    frozen=train.object_hash(snapshot); state=snapshot['state_identity']
    if score.tensor_digest(batch)!=state['batch_sha256']: raise ValueError('Snapshot batch changed')
    for field,key in [('candidates','candidates_sha256'),('controller','context_sha256')]:
        if score.json_digest(snapshot['feature'][field])!=state[key]: raise ValueError('Snapshot context changed')
    actions=snapshot['reference_actions'].detach().cpu().clone()
    if score.tensor_digest({'actions':actions})!=state['reference_prefix_sha256']: raise ValueError('Reference prefix changed')
    kwargs=dict(deterministic=True,actions=actions)
    kwargs['execution_only' if snapshot['phase']=='first_execution_pass' else 'return_details']=True
    variants={}; masks=None; guards=[]
    for label in [model_key(a,e) for a in ARMS for e in EPISODES]+['reference_prior_only']:
        m=models[reference_label if label=='reference_prior_only' else label]
        before=train.training_state_hash(m); rng=score.rng_digest(); batch_before=score.tensor_digest(batch)
        budget.charge('qualification_scoring_calls' if qualification else 'scoring_calls')
        _,trace=rich_trace(rt,m,batch,kwargs,zero_learned=label=='reference_prior_only')
        after=train.training_state_hash(m)
        if before!=after or rng!=score.rng_digest() or batch_before!=score.tensor_digest(batch):
            raise ValueError('Scoring changed state/controller/optimizer/RNG/input')
        current=[h['actual_mask'] for h in trace['heads']]
        if masks is not None and current!=masks: raise ValueError('Teacher-forced actual masks differ')
        masks=current; variants[label]=trace
        if label=='reference_prior_only':
            for head in trace['heads']:
                if any(v!=0. for v in head['valid_learned_logits']) or head['valid_total_logits']!=head['valid_public_prior']:
                    raise ValueError('Prior-only hook did not remove learned logits exactly')
        guards.append(dict(label=label,state_before=before,state_after=after,rng_sha256=rng,batch_sha256=batch_before))
    check_reference_parity(snapshot['reference_trace'],variants[reference_label])
    check_reference_parity(variants[model_key('sa_legacy_clip',0)],variants[model_key('sa_group_clip',0)])
    if train.object_hash(snapshot)!=frozen: raise ValueError('Snapshot mutated during conditional scoring')
    return dict(schema=SCHEMA+'_scores',witness=snapshot['witness'],frame=snapshot['frame'],phase=snapshot['phase'],
        vehicle_id=snapshot['feature']['vehicle_id'],node_id=snapshot['feature']['node_id'],
        state_identity=state,reference_model=reference_label,reference_actions=actions.tolist(),
        candidates=score.jsonable(snapshot['feature']['candidates']),variants=variants,
        diagnostic_metadata=score.jsonable(snapshot.get('diagnostic_metadata',{})),
        episode_labels=score.jsonable(snapshot.get('episode_labels',{})),
        team_frame_key=snapshot.get('team_frame_key'),team_frame_sha256=snapshot.get('team_frame_sha256'),
        guards=guards,reference_parity=True,initial_arm_parity=True,boundary=BOUNDARY,
        reference_policy_mode='deterministic_greedy',
        probabilities_role='conditional categorical scores; NOT actual sampling behavior probabilities',
        episode_labels_role='factual greedy reference MC teamtail/10 and teamcredit/8; NOT original stochastic training labels')


def rich_trace(rt,agent,batch,kwargs,**options):
    """Observe exactly one existing forward; never evaluate missing first-pass heads."""
    torch=rt['torch']; score=rt['score']; captured={}; distributions=[]; handles=[]
    original_choice=agent._choice_distribution
    def hook_for(name):
        def hook(module,inputs,output): captured[name]=output.detach().cpu().clone()
        return hook
    for group in ('surrogates','action_values','prefix_values'):
        for j,module in enumerate(getattr(agent,group)):
            handles.append(module.register_forward_hook(hook_for(f'{group}.{j}')))
    handles.append(agent.critic.register_forward_hook(hook_for('critic')))
    def choice(this,logits,mask,deterministic):
        dist=original_choice(logits,mask,deterministic)
        distributions.append(dict(probabilities=dist.probs.detach().cpu().clone(),
            log_probabilities=dist.logits.detach().cpu().clone(),entropy=dist.entropy().detach().cpu().clone(),
            actual_mask=mask.detach().cpu().bool().clone()))
        return dist
    try:
        with score.patched_method(agent,'_choice_distribution',choice):
            result,trace=rt['base_trace'](agent,batch,kwargs,**options)
    finally:
        for handle in handles: handle.remove()
    first=bool(kwargs.get('execution_only',False))
    for j,head in enumerate(trace['heads']):
        d=distributions[j]; action=result[0][:,j].detach().cpu().long()
        head['distribution']=score.jsonable(dict(probabilities=d['probabilities'],
            log_probabilities=d['log_probabilities'],entropy=d['entropy'],
            normalized_entropy=d['entropy']/d['actual_mask'].sum(1).float().log().clamp(min=1),
            reference_log_probability=d['log_probabilities'].gather(1,action[:,None]).squeeze(1)))
        raw=captured.get(f'surrogates.{j}')
        estimates=None if raw is None else torch.cat((raw[:,:1],torch.nn.functional.softplus(raw[:,1:])),1)
        head['valid_candidate_surrogate_raw']=score.jsonable(raw)
        head['valid_candidate_surrogate_estimates']=score.jsonable(estimates)
        head['surrogate_axes']=['return','model_link_cost','state_link_cost']
        head['valid_candidate_q']=score.jsonable(captured.get(f'action_values.{j}'))
        head['prefix_value']=score.jsonable(captured.get(f'prefix_values.{j}'))
        head['availability']=dict(surrogate='original existing forward' if raw is not None else 'disabled',
            q='not computed: execution_only early return' if first else 'original existing forward',
            prefix_value='not computed: execution_only early return' if first else 'original existing forward',
            critic_value='not computed: execution_only early return' if first else 'original existing forward')
    trace['forward_outputs']={'reference_head_log_probs':score.jsonable(result[1]),
                             'critic_value':score.jsonable(captured.get('critic'))}
    if not first:
        trace['forward_outputs'].update({name:score.jsonable(result[i]) for name,i in
            [('normalized_entropies',2),('value',3),('selected_surrogate',4),('prefix_values',5),
             ('active_heads',6),('counterfactual_baselines',7),('selected_q',8),('maximum_logits',9)]})
    trace['diagnostic_forward_count']=1
    return result,trace


def stable_sets(value):
    """Preserve dict/tuple keys while giving sets a cross-process stable encoding."""
    if isinstance(value,(set,frozenset)):
        return {'__set_type__':type(value).__name__,'items':[stable_sets(x) for x in sorted(value,key=repr)]}
    if isinstance(value,dict): return {k:stable_sets(v) for k,v in value.items()}
    if isinstance(value,tuple): return tuple(stable_sets(v) for v in value)
    if isinstance(value,list): return [stable_sets(v) for v in value]
    return value


def physical_state(env):
    """Read-only causal internal ledger; not an input to any alternative model."""
    names=('state_version','state_payload_bytes','execution_location','work_progress','last_service_rsu',
           'state_copies','state_transfers','state_transfer_remaining','input_remaining','input_received',
           'output_pending','previous_service_targets','vehicle_compute_progress','handoff_wait','handoff_target',
           'prepared_state','cold_restart_done','_released','_finished','_started')
    state={name:deepcopy(getattr(env,name,None)) for name in names}
    state.update(frame=int(env.frame_index),time_index=int(env.frames[env.frame_index]['time_index']),
        active={vid:dict(request_id=a.request_id,arrival_frame=a.frame,workflow_id=a.workflow.workflow_id) for vid,a in env._active.items()},
        released_queues={vid:[dict(request_id=a.request_id,arrival_frame=a.frame,workflow=a.workflow.to_dict()) for a in queue] for vid,queue in env._queues.items()},
        workflows={vid:w.to_dict() for vid,w in env.workflows.items()},
        rsu_cache={r.rsu_id:deepcopy(env.model_cache.snapshot(r.rsu_id)) for r in env.rsus},
        resource_capacities=dict(link_slots_per_rsu=env.costs.link_slots_per_rsu,compute_slots_per_rsu=env.costs.compute_slots_per_rsu),
        costs=asdict(env.costs),
        limitations='Actual link/compute allocation occurs inside step_controls. Before-step ledger is causal; realized grants/conflicts are recorded in reference events/receipts. No future arrivals included. This diagnostic ledger is not a complete restorable environment snapshot.')
    return stable_sets(state)


def physical_priority_order(controls,frame):
    ids=sorted(controls); shift=frame%len(ids) if ids else 0
    tie=ids[shift:]+ids[:shift]
    return sorted(tie,key=lambda vid:-controls[vid].priority)


def collector_class(rt):
    base=rt['Capture']; score=rt['score']; torch=rt['torch']
    class TeamCapture(base):
        def __init__(self,agent,witness,frames,budget,env):
            super().__init__(agent,witness,frames); self.team_frames={}; self.budget=budget; self.env=env
        def __enter__(self):
            # Temporarily route capture tracing through enriched passive hooks.
            old_trace=score.trace_evaluate
            score.trace_evaluate=lambda *args,**kwargs: rich_trace(rt,*args,**kwargs)
            class RestoreTrace:
                def __exit__(self,*exc): score.trace_evaluate=old_trace
            self._contexts.append(RestoreTrace())
            super().__enter__(); original=self.agent.act_controls
            def act(this,decisions,frame,*,deterministic=False):
                selected=int(frame) in self.frames
                if selected:
                    # Charge known all-live-vehicle two-pass capture before any traced forward.
                    self.budget.charge('saved_calls',2*len(decisions)); self.budget.charge('second_pass_calls',len(decisions))
                    frozen_decisions=deepcopy(decisions); pre=physical_state(self.env); start=len(self.snapshots)
                result=original(decisions,frame,deterministic=deterministic)
                if selected:
                    controls,roll=result
                    payload=dict(frame=int(frame),physical_pre_action=pre,decision_vehicle_order=list(decisions),
                        physical_priority_order=physical_priority_order(controls,int(frame)),
                        scored=bool(decisions),idle_no_active_decision=not bool(decisions),
                        decisions=frozen_decisions,
                        ordered_team_controls=[dict(vehicle_id=v,control=asdict(c) if is_dataclass(c) else deepcopy(c)) for v,c in controls.items()],
                        rollout_vehicle_order=list(roll['vehicle_ids']) if roll else [],
                        controller_after=score._controller_context(this),
                        last_diagnostics=deepcopy(this.last_diagnostics),
                        interpretation='Both controller insertion order and frozen-source priority/rotating-tie physical order are recorded. Confirm actual event/receipt admissions before claiming physical equivalence.')
                    key=str(frame); h=rt['train'].object_hash(payload); payload['pre_action_payload_sha256']=h; self.team_frames[key]=payload
                    for snap in self.snapshots[start:]: snap.update(team_frame_key=key,team_frame_sha256=h)
                    if len(self.snapshots)-start!=2*len(decisions): raise ValueError('Captured live team call count differs')
                return result
            try:
                cm=score.patched_method(self.agent,'act_controls',act); cm.__enter__(); self._contexts.append(cm)
            except BaseException:
                self.__exit__(*sys.exc_info()); raise
            return self
        def save(self,path):
            if Path(path).exists(): raise FileExistsError(path)
            torch.save(dict(schema=SCHEMA+'_capture',witness=self.witness,selected_frames=self.selected_frames,
                snapshots=self.snapshots,team_frames=self.team_frames,
                selection='Frozen original-step nine quantile frames; every live vehicle and both passes'),path)
    return TeamCapture


def trace_digest(path):
    h=hashlib.sha256(); n=0
    with gzip.open(path,'rt',encoding='utf8') as f:
        for line in f: h.update(canonical(clean_payload(json.loads(line)))); n+=1
    return dict(sha256=h.hexdigest(),rows=n)


def claim(path,payload): write(path,dict(started_at=utc(),pid=os.getpid(),**payload))

def qualify(args):
    out=args.output_root.resolve(); plan=read(out/'plan.json'); verify(plan,out)
    claim(out/'qualification_started.json',dict(plan_sha256=sha(out/'plan.json')))
    budget=Budget(CAPS)
    try:
        rt=runtime(Path(plan['repo_root'])); rt['torch'].set_num_threads(1)
        capsule=rt['torch'].load(plan['qualification_capsule']['path'],map_location='cpu',weights_only=False)
        witness=capsule['witness']
        if witness['seed']!=7 or witness.get('method',witness.get('agent'))!='sa_ghmappo': raise ValueError('Qualification requires seed7 SA64 old capture')
        snapshots=capsule['snapshots']; selected=[]
        for phase in ('first_execution_pass','second_cache_event_pass'):
            subset=[i for i,s in enumerate(snapshots) if s['phase']==phase][:8]
            if len(subset)!=8: raise ValueError('Need8 cached snapshots of each pass')
            selected.extend(subset)
        models=load_models(rt,plan,7); states={k:rt['train'].training_state_hash(v) for k,v in models.items()}
        rng=rt['score'].rng_digest()
        with no_learning(rt,models),gzip.open(out/'qualification_scores.jsonl.gz','xt',encoding='utf8') as f:
            for i in selected:
                budget.charge('qualification_snapshots')
                scored=score_seven(rt,snapshots[i],models,model_key('sa_legacy_clip',64),budget,qualification=True)
                f.write(json.dumps(dict(capsule_index=i,**scored),ensure_ascii=False,allow_nan=False)+'\n')
        if states!={k:rt['train'].training_state_hash(v) for k,v in models.items()} or rng!=rt['score'].rng_digest():
            raise ValueError('Qualification changed model/controller/optimizer/RNG')
        receipt=dict(status='pass',completed_at=utc(),plan_sha256=sha(out/'plan.json'),selected_indices=selected,
            counts=budget.used,environment_frames=0,source_commit=plan['source_commit'],
            scores=bound(out/'qualification_scores.jsonl.gz'),guards='exact reference64 and initial-arm parity; state/input/RNG unchanged')
        write(out/'qualification.json',receipt); return receipt
    except BaseException as exc:
        write(out/'qualification_failure.json',dict(status='paused_failed',error=repr(exc),traceback=traceback.format_exc(),counts=budget.used)); raise


def prepare(args):
    out=args.output_root.resolve(); plan=read(out/'plan.json'); verify(plan,out)
    q=read(out/'qualification.json')
    if q['status']!='pass' or q['plan_sha256']!=sha(out/'plan.json') or q['source_commit']!=plan['source_commit']:
        raise ValueError('Successful exact-plan software qualification required')
    if q['counts']['qualification_snapshots']!=16 or q['counts']['qualification_scoring_calls']!=112 or q['environment_frames']!=0:
        raise ValueError('Qualification budget/coverage mismatch')
    if any(q['counts'][k] for k in CAPS if k not in ('qualification_snapshots','qualification_scoring_calls')):
        raise ValueError('Qualification used unauthorized operations')
    check_bound(q['scores']); a=read(args.a_receipt)
    if (a.get('schema')!='sa_stage5_existing_evidence_review_v1' or a.get('status')!='complete'
            or a.get('source_integrity_pass') is not True):
        raise ValueError('Completed source-verified Stage5 A receipt required')
    contract=dict(plan,status='frozen',prepared_at=utc(),plan=bound(out/'plan.json'),
                  qualification=bound(out/'qualification.json'),a_review=bound(args.a_receipt))
    write(out/'contract.json',contract); return dict(status='frozen',contract=bound(out/'contract.json'))


def verify_frozen_receipts(c,out):
    check_bound(c['plan']); check_bound(c['qualification']); check_bound(c['a_review'])
    plan=read(c['plan']['path'])
    for key,value in plan.items():
        if key!='status' and c.get(key)!=value: raise ValueError('Frozen contract differs from qualified plan: '+key)
    q=read(c['qualification']['path'])
    if q['status']!='pass' or q['plan_sha256']!=c['plan']['sha256'] or q['source_commit']!=c['source_commit']:
        raise ValueError('Qualification binding invalid')
    expected={k:0 for k in CAPS}; expected.update(qualification_snapshots=16,qualification_scoring_calls=112)
    if q['counts']!=expected or q['environment_frames']!=0: raise ValueError('Qualification operations differ')
    check_bound(q['scores']); a=read(c['a_review']['path'])
    if a.get('schema')!='sa_stage5_existing_evidence_review_v1' or a.get('status')!='complete' or a.get('source_integrity_pass') is not True:
        raise ValueError('A review no longer valid')


def run(args):
    out=args.output_root.resolve(); path=out/'contract.json'; c=read(path)
    # One path-bound claim for the whole frozen plan, before model/env construction.
    if c['status']!='frozen': raise ValueError('Frozen contract required')
    claim(out/'run_started.json',dict(contract_sha256=sha(path),policy='exclusive; no resume or retry'))
    budget=Budget(CAPS); completed=[]; current=None; env=None; tap=None; home=None; rt=None
    try:
        verify(c,out); verify_frozen_receipts(c,out)
        rt=runtime(Path(c['repo_root'])); torch=rt['torch']; score=rt['score']; torch.set_num_threads(1)
        from scripts.train_joint_stream_control import make_env,stream_episode
        protocol=read(c['source_protocol']['path']); data=torch.load(c['protocol_data']['path'],map_location='cpu',weights_only=False)
        cases=unique_index(data['cases']['dev'],lambda x:x['case_id'])
        Collector=collector_class(rt); models=None; current_seed=None
        write(out/'command.json',dict(argv=sys.argv,cwd=os.getcwd(),python=sys.executable))
        for index,item in enumerate(c['sessions']):
            current=dict(index=index,**{k:item[k] for k in ('arm','seed','episode','case_id')}); tap=None
            if item['seed']!=current_seed: models=load_models(rt,c,item['seed']); current_seed=item['seed']
            label=model_key(item['arm'],item['episode']); agent=models[label]
            home=out/f"{index:03d}_{item['arm']}_seed{item['seed']}_ep{item['episode']}_{item['case_id']}"; home.mkdir(exist_ok=False)
            learning={k:learning_hash(rt,m) for k,m in models.items()}; inactive={k:rt['train'].training_state_hash(m) for k,m in models.items() if k!=label}
            # Reference rollout may change only its controller; software scoring may change nothing.
            with score.preserved_rng(),no_learning(rt,models):
                budget.charge('reference_sessions'); env=make_env(cases[item['case_id']],data,protocol)
                original_step=env.step_controls; session={'frames':0}
                def bounded_step(this,controls):
                    if session['frames']>=item['expected_summary']['steps'] or session['frames']>=480:
                        raise RuntimeError('Reference exceeds frozen original length before step')
                    budget.charge('reference_frames'); session['frames']+=1
                    # The unchanged episode loop calls act_controls even for an empty team.
                    # Preserve every frozen quantile frame, including idle/no-score frames.
                    if int(this.frame_index) in tap.frames:
                        frame=tap.team_frames.get(str(this.frame_index))
                        if frame is None: raise ValueError('Missing pre-action frame ledger')
                        frame['physical_pre_step']=physical_state(this)
                        frame['physical_pre_step_sha256']=rt['train'].object_hash(frame['physical_pre_step'])
                    return original_step(controls)
                with score.patched_method(env,'step_controls',bounded_step):
                    with gzip.open(home/'events.jsonl.gz','xt',encoding='utf8') as ef,gzip.open(home/'receipts.jsonl.gz','xt',encoding='utf8') as rf:
                        with Collector(agent,item['identity'],item['frames'],budget,env) as tap:
                            row,rolls=stream_episode(env,agent,deterministic=True,event_file=ef,receipt_file=rf,identity=item['identity'])
                tap.bind_episode_labels(rolls,fleet=8); tap.save(home/'snapshots.pt')
                actual=dict(item['identity'],**row,tensor_sha256=rt['tensor_hash'](agent),actor_sha256=rt['actor_hash'](agent),completed_updates=int(agent.completed_updates))
                write(home/'reference_row.json',actual)
                if clean_payload(actual)!=clean_payload(item['expected_summary']) or row['steps']!=session['frames']:
                    raise ValueError('Full original summary parity failed')
                traces={name:trace_digest(home/(name+'.jsonl.gz')) for name in ('events','receipts')}
                led=item['expected_ledger']
                if traces!={'events':dict(sha256=led['event_sha256'],rows=led['events']), 'receipts':dict(sha256=led['receipt_sha256'],rows=led['receipt_frames'])}:
                    raise ValueError('Canonical original event/receipt ledger parity failed')
                if tap.selected_frames!=item['frames']: raise ValueError('Frozen quantile frames not exactly captured')
                torch.save(dict(schema=SCHEMA+'_labels',identity=item['identity'],rollouts=[r for r in rolls if int(r['frame_index']) in tap.frames],
                    initial_request_count=8,episode_steps=row['steps'],episode_decision_rows=row['decision_rows'],
                    reference_policy_mode='deterministic_greedy',
                    MC_contract='factual greedy reference physical teamtail/10; teamcredit/8; NOT original stochastic training labels; receipt request_rewards is not the PPO label'),home/'rollouts.pt')
                scoring_states={k:rt['train'].training_state_hash(m) for k,m in models.items()}; rng=score.rng_digest()
                with gzip.open(home/'scoring.jsonl.gz','xt',encoding='utf8') as f:
                    for snapshot in tap.snapshots:
                        scored=score_seven(rt,snapshot,models,label,budget)
                        f.write(json.dumps(scored,ensure_ascii=False,allow_nan=False)+'\n')
                if scoring_states!={k:rt['train'].training_state_hash(m) for k,m in models.items()} or rng!=score.rng_digest():
                    raise ValueError('Conditional scoring changed full model states/RNG')
            if learning!={k:learning_hash(rt,m) for k,m in models.items()} or inactive!={k:rt['train'].training_state_hash(models[k]) for k in inactive}:
                raise ValueError('Replay modified learning state or another controller')
            entry=dict(status='pass',identity=item['identity'],reference_frames=session['frames'],selected_frames=tap.selected_frames,
                saved_calls=len(tap.snapshots),second_pass_calls=sum(s['phase']=='second_cache_event_pass' for s in tap.snapshots),
                summary_exact=True,traces_exact=traces,team_frames=len(tap.team_frames),
                outputs={name:bound(home/name) for name in ('reference_row.json','events.jsonl.gz','receipts.jsonl.gz','snapshots.pt','rollouts.pt','scoring.jsonl.gz')})
            write(home/'result.json',entry); completed.append(entry)
            progress=dict(status='running',completed_sessions=len(completed),counts=budget.used,current=current)
            write(out/'progress.json',progress,exclusive=False); print(json.dumps(progress),flush=True)
        verify(c,out)
        if len(completed)!=144 or budget.used['scoring_calls']!=7*budget.used['saved_calls'] or budget.used['saved_calls']!=2*budget.used['second_pass_calls']:
            raise ValueError('Final accounting mismatch')
        result=dict(status='complete',completed_at=utc(),contract_sha256=sha(path),counts=budget.used,results=completed,boundary=BOUNDARY,
                    source_before_after_exact=True,training_sessions=0,backward_calls=0,optimizer_steps=0)
        write(out/'result.json',result); write(out/'progress.json',dict(status='complete',counts=budget.used),exclusive=False); return result
    except BaseException as exc:
        # Preserve a partially captured failing session without implying parity/label completion.
        if tap is not None and home is not None and rt is not None and not (home/'snapshots.pt').exists():
            try: tap.save(home/'partial_snapshots.pt')
            except BaseException: pass
        failure=dict(status='paused_failed',failed_at=utc(),current=current,error=repr(exc),traceback=traceback.format_exc(),counts=budget.used,
                     completed_sessions=len(completed),current_env_frame=getattr(env,'frame_index',None),policy='Preserve evidence/claim. No retry/resume or source repair then continuation.')
        write(out/'failure.json',failure); write(out/'progress.json',failure,exclusive=False); raise


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    for name in ('contract','qualify','prepare','run','status'):
        q=sub.add_parser(name); q.add_argument('--output_root',type=Path,required=True)
        if name=='contract':
            q.add_argument('--repo_root',type=Path,default=ROOT); q.add_argument('--source_run',type=Path,required=True)
            q.add_argument('--qualification_capsule',type=Path,required=True)
        if name=='prepare': q.add_argument('--a_receipt',type=Path,required=True)
    a=p.parse_args()
    if a.command=='contract': result=build_plan(a)
    elif a.command=='qualify': result=qualify(a)
    elif a.command=='prepare': result=prepare(a)
    elif a.command=='run': result=run(a)
    else:
        out=a.output_root.resolve(); result=next((read(out/name) for name in ('failure.json','result.json','qualification_failure.json','progress.json','qualification.json','contract.json','plan.json') if (out/name).exists()),dict(status='absent'))
    print(json.dumps({k:v for k,v in result.items() if k not in ('results','sessions','checkpoints','bindings','stage4_source_fingerprints')},ensure_ascii=False,allow_nan=False))


if __name__=='__main__': main()
