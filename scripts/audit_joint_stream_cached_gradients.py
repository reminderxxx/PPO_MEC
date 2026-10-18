"""Pure cached-reference PPO-form gradient diagnostic, never a training entrypoint.

Reference trajectories are greedy final64. Their categorical scores are NOT
sampling probabilities of a stochastic behavior policy. Ratios here only compare
fixed conditional categorical scores; no causal/off-policy-learning claim follows.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib, importlib.util, json, math, sys
from pathlib import Path
from types import MethodType
import torch

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/joint_stream_learning/20261016T162100Z_confirmation'
SCORER=ROOT/'scripts/joint_stream_same_state_scoring.py'
TASKS=ROOT/'artifacts/joint_stream_learning/learning_effectiveness_stage1_20261017/recovery/stage2_task_points.json'
HEADS=('execution','cache','prepare_priority')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_scorer(path=SCORER):
    spec=importlib.util.spec_from_file_location('stage2_frozen_scorer',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def object_digest(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous();h.update(str((t.dtype,tuple(t.shape))).encode());h.update(t.numpy().tobytes())
        elif isinstance(x,dict):
            for k,v in sorted(x.items(),key=lambda kv:str(kv[0])):h.update(repr(k).encode());visit(v)
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def model_full_digest(model):
    return object_digest({'weights':model.state_dict(),'optimizer':model.optimizer.state_dict(),
        'surrogate_optimizer':model.surrogate_optimizer.state_dict() if hasattr(model,'surrogate_optimizer') else None,
        'parameter_gradients':{n:p.grad for n,p in model.named_parameters()}})


def forbidden(*args,**kwargs):raise RuntimeError('cached gradient diagnostic forbids update, optimizer.step and action/environment execution')


@contextmanager
def zero_update_guard(model):
    old_step=torch.optim.Adam.step;torch.optim.Adam.step=forbidden
    methods=['update','act_controls','_coordinated_features','_apply_control_gradients']
    old={name:(name in model.__dict__,model.__dict__.get(name)) for name in methods}
    try:
        for name in methods:setattr(model,name,MethodType(forbidden,model))
        yield
    finally:
        torch.optim.Adam.step=old_step
        for name,(had,value) in old.items():
            if had:setattr(model,name,value)
            else:delattr(model,name)


def gradients(loss,named_parameters):
    names,params=zip(*named_parameters)
    gs=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
    squares={}; counts={}
    for name,g in zip(names,gs):
        if g is None:continue
        group=name.split('.')[0]
        squares[group]=squares.get(group,0.)+float(g.detach().double().square().sum())
        counts[group]=counts.get(group,0)+g.numel()
    return {'module_norms':{k:math.sqrt(v) for k,v in squares.items()},
        'module_gradient_elements':counts,'total_norm':math.sqrt(sum(squares.values()))}


def finite_summary(x):
    x=x.detach().cpu().double().flatten()
    if not len(x):return {'count':0,'min':None,'max':None,'mean':None}
    if not bool(torch.isfinite(x).all()):raise ValueError('non-finite diagnostic values')
    return {'count':len(x),'min':float(x.min()),'max':float(x.max()),'mean':float(x.mean())}


def pack_rows(snapshots,scorer):
    rows=[r for r in snapshots if r['phase']=='second_cache_event_pass']
    if not rows:raise ValueError('No captured second-pass rows')
    batches=[];fields={k:[] for k in ['actions','head_logs','active_heads','counterfactual_baselines','team_credit_targets','cost_targets']}
    ids=[]
    for row in rows:
        b=row['batch'];identity=row['state_identity'];feature=row['feature']
        if b['own'].shape[0]!=1:raise ValueError('Expected one vehicle per captured evaluate call')
        if scorer.tensor_digest(b)!=identity['batch_sha256']:raise ValueError('Batch identity mismatch')
        if scorer.json_digest(feature['candidates'])!=identity['candidates_sha256']:raise ValueError('Candidate identity mismatch')
        if scorer.json_digest(feature['controller'])!=identity['context_sha256']:raise ValueError('Controller identity mismatch')
        if scorer.tensor_digest({'actions':row['reference_actions']})!=identity['reference_prefix_sha256']:raise ValueError('Reference prefix mismatch')
        forward=row['forward_rollout'];labels=row['episode_labels'];fleet=int(labels['initial_request_count'])
        if fleet!=8:raise ValueError('Frozen diagnosis requires physical fleet8')
        if not torch.equal(forward['actions'],row['reference_actions']):raise ValueError('Captured reference actions mismatch')
        if not torch.allclose(labels['returns']/fleet,labels['team_credit_targets'],atol=1e-7,rtol=1e-7):raise ValueError('MC/fleet units mismatch')
        if not torch.equal(labels['credit_targets'],labels['team_credit_targets']):raise ValueError('Team target mismatch')
        for key in fields:
            src=labels if key in ('team_credit_targets','cost_targets') else forward
            fields[key].append(src[key].detach().cpu().clone())
        batches.append({k:v.detach().cpu().clone() for k,v in b.items()})
        ids.append({'frame':int(row['frame']),'vehicle_id':feature['vehicle_id'],'state_identity':identity})
    count=max(b['m1'].shape[1] for b in batches)
    if count>513:raise ValueError('Cache candidate bound exceeded')
    for b in batches:
        missing=count-b['m1'].shape[1]
        for key in ['f1','m1','node1']:
            b[key]=torch.nn.functional.pad(b[key],(0,0,0,missing) if key=='f1' else (0,missing),value=-1 if key=='node1' else 0)
    data={k:torch.cat([b[k] for b in batches]) for k in batches[0]}
    labels={k:torch.cat(v) for k,v in fields.items()}
    return rows,data,labels,ids


def diagnose_model(model,snapshot_label,data,labels,ids,scorer):
    if model.method=='dqn':raise ValueError('PPO-form gradients do not apply to DQN')
    state_before=model_full_digest(model);rng_before=scorer.rng_digest();input_before=object_digest((data,labels))
    actions=labels['actions'];old=labels['head_logs'];active=labels['active_heads'];target=labels['team_credit_targets']
    # Exactly frozen head-specific normalization, on this bounded captured tuple.
    advantage=target[:,None]-labels['counterfactual_baselines'];normalization=[]
    for j in range(3):
        sample=advantage[active[:,j],j]
        mu=float(sample.mean()) if len(sample) else None;sd=float(sample.std()) if len(sample)>1 else None
        if len(sample)>1:advantage[:,j]=(advantage[:,j]-sample.mean())/sample.std().clamp(min=1e-6)
        normalization.append({'head':HEADS[j],'active_samples':len(sample),'raw_mean':mu,'raw_std':sd,
            'normalization_applied':len(sample)>1})
    trace=[];original_choice=model._choice_distribution
    def choice(this,logits,mask,deterministic):
        j=len(trace);counts=mask.sum(1);masked=logits.detach().masked_fill(~mask,-torch.inf)
        top=masked.topk(min(2,masked.shape[1]),dim=1).values
        margin=top[:,0]-top[:,1] if top.shape[1]>1 else logits.new_full((len(logits),),float('nan'))
        winners=masked.argmax(1)
        trace.append({'head':HEADS[j],'valid_candidate_counts':counts.tolist(),
            'forced_count':int((counts==1).sum()),'optional_count':int((counts>1).sum()),
            'top2_margin_on_optional':finite_summary(margin[counts>1]),
            'greedy_reference_action_agreement':int((winners==actions[:,j]).sum()),
            'greedy_indices':winners.tolist(),'reference_indices':actions[:,j].tolist()})
        return original_choice(logits,mask,deterministic)
    with zero_update_guard(model),scorer.patched_method(model,'_choice_distribution',choice):
        _,logs,entropy,value,surrogate,prefix,enabled,baseline,q,_=model._evaluate(
            data,actions=actions,deterministic=True,return_details=True)
        if not torch.equal(enabled,active):raise ValueError('Dynamic active masks differ from reference')
        if str(snapshot_label)=='64' and not torch.allclose(logs,old,atol=1e-5,rtol=1e-5):
            raise ValueError('Reference64 conditional log-prob parity failed')
        logratio=logs-old;ratio=logratio.exp();weight=enabled.float();den=weight.sum().clamp(min=1)
        objective=torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)
        policy=-(objective*weight).sum()/den
        vf=(prefix-target[:,None]).square().mean()+(value-target).square().mean()+(q-target[:,None]).square().mean()
        expected=torch.cat((target[:,None],labels['cost_targets']),-1)[:,None]
        aux=(surrogate-expected).square().mean() if model.surrogate_enabled else vf*0
        progress=min(1.,float(model.completed_updates)*4/max(1,model.training_budget-4))
        entropy_coefficient=model._entropy_coefficient(progress)
        entropy_loss=-entropy_coefficient*(entropy*weight).sum()/den
        terms={'policy':policy,'value_half':.5*vf,'entropy':entropy_loss,'surrogate_point2':.2*aux}
        terms['total']=sum(terms.values())
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        components={name:{'loss':float(loss.detach()),**gradients(loss,named)} for name,loss in terms.items()}
        policy_by_head={}
        for j in range(3):
            loss=-(objective[:,j]*weight[:,j]).sum()/den
            policy_by_head[HEADS[j]]={'loss':float(loss.detach()),**gradients(loss,named)}
    groups=components['total']['module_norms']
    non_surrogate=math.sqrt(sum(v*v for k,v in groups.items() if k!='surrogates'))
    surrogate_norm=groups.get('surrogates',0.)
    clip=min(1.,.5/(non_surrogate+1e-6));surrogate_clip=min(1.,.5/(surrogate_norm+1e-6))
    guards={'weights_optimizer_and_grad_buffers_unchanged':state_before==model_full_digest(model),
        'input_unchanged':input_before==object_digest((data,labels)),'rng_unchanged':rng_before==scorer.rng_digest(),
        'environment_steps':0,'update_calls':0,'optimizer_steps':0}
    if not all(guards[k] for k in ['weights_optimizer_and_grad_buffers_unchanged','input_unchanged','rng_unchanged']):
        raise RuntimeError(f'Diagnostic purity failed: {guards}')
    return {'snapshot':str(snapshot_label),'rows':len(actions),'identities':ids,
        'normalization_scope':'only the fixed selected second-pass rows in this witness tuple; not original training batches',
        'normalization':normalization,'return_target_summary':finite_summary(target),
        'cost_target_units':'model/state link frames divided by10; distinct physical auxiliary outputs',
        'entropy_coefficient':entropy_coefficient,'completed_updates':int(model.completed_updates),
        'schedule_point':'next-update schedule at this checkpoint; no update executed',
        'reference_categorical_ratio':finite_summary(ratio[enabled]),
        'reference_logprob_max_abs_difference':float(logratio.detach().abs().max()),
        'reference_categorical_kl_estimator':float((((ratio-1)-logratio)*weight).sum().detach()/den),
        'reference_categorical_clip_fraction':float((((ratio-1).abs()>.2)*weight).sum().detach()/den),
        'would_cross_frozen_kl_stop_threshold':bool(model.stability_enabled and float((((ratio-1)-logratio)*weight).sum().detach()/den)>.02),
        'kl_stop_note':'If threshold exceeded, frozen training would skip; diagnostic still reports hypothetical loss gradients without stepping.',
        'coverage_and_margins':trace,'components':components,'policy_components_by_head':policy_by_head,
        'clip':{'non_surrogate_norm':non_surrogate,'non_surrogate_coefficient':clip,
                'surrogate_norm':surrogate_norm,'surrogate_coefficient':surrogate_clip,
                'heads_norm_before':groups.get('heads',0.),'heads_norm_after_hypothetical_clip':groups.get('heads',0.)*clip},
        'guards':guards}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--capture',nargs='+',type=Path)
    inputs.add_argument('--stage2_root','--stage2-root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-run',type=Path);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--scorer',type=Path)
    p.add_argument('--task-points',type=Path)
    args=p.parse_args()
    args.source_run=args.source_run or args.root/'artifacts/joint_stream_learning/20261016T162100Z_confirmation'
    args.scorer=args.scorer or args.root/'scripts/joint_stream_same_state_scoring.py'
    args.task_points=args.task_points or args.root/'artifacts/joint_stream_learning/learning_effectiveness_stage1_20261017/recovery/stage2_task_points.json'
    capture_manifest=None;capture_expectations={}
    if args.stage2_root is not None:
        capture_manifest=args.stage2_root/'result.json'
        completed=json.loads(capture_manifest.read_text())
        if completed.get('status')!='complete' or completed.get('reference_sessions')!=15:
            raise ValueError('Reference capture must be complete with all15 frozen witnesses')
        if completed.get('training_sessions')!=0 or completed.get('optimizer_steps')!=0:
            raise ValueError('Reference capture violates zero-training contract')
        if not 0<int(completed['reference_frames'])<=7200:raise ValueError('Reference frame budget exceeded')
        entries=completed['captures']
        if len(entries)!=15:raise ValueError('Incomplete capture manifest')
        args.capture=[]
        for entry in entries:
            path=Path(entry['path'])
            if not path.is_absolute():path=args.stage2_root/path
            path=path.resolve()
            if not path.is_relative_to(args.stage2_root.resolve()):raise ValueError('Capture outside stage2 root')
            if str(path) in capture_expectations:raise ValueError('Duplicate capture path')
            if sha(path)!=entry['sha256']:raise ValueError(f'Capture hash mismatch: {path}')
            capture_expectations[str(path)]=entry
            args.capture.append(path)
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    scorer=load_scorer(args.scorer)
    allowed={(r['method'],int(r['seed']),r['case_id']) for r in json.loads(args.task_points.read_text())['fixed_witnesses']['cases']}
    protocol=json.loads((args.source_run/'protocol/manifest.json').read_text())
    checks={name:sha(args.root/name)==expected for name,expected in protocol['source_fingerprints'].items()}
    if not all(checks.values()):raise ValueError('Frozen algorithm fingerprints mismatch')
    captures=[];seen=set();total_rows=0
    for path in args.capture:
        cap=torch.load(path,map_location='cpu',weights_only=False)
        if cap['schema']!='same_state_reference_capture_v1':raise ValueError('Unsupported capture schema')
        w=cap['witness'];key=(w['method'],int(w['seed']),w['case_id'])
        if capture_expectations:
            expected_w=capture_expectations[str(path)]['witness']
            if (w['method'],int(w['seed']),w['case_id'])!=(expected_w['method'],int(expected_w['seed']),expected_w['case_id']):
                raise ValueError('Capture witness differs from completed manifest')
        if key not in allowed or key in seen:raise ValueError('Unfrozen/duplicate witness')
        seen.add(key)
        frames=cap['selected_frames']
        if len(frames)>9 or len(set(frames))!=len(frames):raise ValueError('Frame sample bound exceeded')
        rows,data,labels,ids=pack_rows(cap['snapshots'],scorer);total_rows+=len(rows)
        perframe={f:sum(r['frame']==f for r in rows) for f in frames}
        if any(r['frame'] not in frames for r in rows):raise ValueError('Row outside selected frames')
        rowkeys={(r['frame'],r['feature']['vehicle_id']) for r in rows}
        if len(rowkeys)!=len(rows):raise ValueError('Duplicate vehicle/frame row')
        if len(rows)>72 or max(perframe.values(),default=0)>8:raise ValueError('Witness row bound exceeded')
        captures.append((path,cap,data,labels,ids))
    if len(captures)>15 or total_rows>1080:raise ValueError('Frozen capture budget exceeded')
    results=[];models_cache={}
    for path,cap,data,labels,ids in captures:
        w=cap['witness'];method=w['method'];seed=int(w['seed'])
        if method=='dqn':
            results.append({'witness':w,'capture_path':str(path),'capture_sha256':sha(path),
                'rows':len(ids),'skipped':'DQN uses Bellman/Huber target-head loss; not PPO-form gradients'})
            continue
        key=(method,seed)
        if key not in models_cache:models_cache[key]=scorer.load_exact_models(args.root,args.source_run,method,seed)
        models,provenance=models_cache[key]
        analyses=[diagnose_model(models[str(ep)],str(ep),data,labels,ids,scorer) for ep in [0,32,64]]
        results.append({'witness':w,'capture_path':str(path),'capture_sha256':sha(path),'provenance':provenance,'analyses':analyses})
    output={'schema':'stage2_cached_reference_gradients_v1','reviewed_at':datetime.now(timezone.utc).isoformat(),
        'source_commit':'35975cb3d73c84ddbabf01f6749bbc2d60c826de','source_fingerprints_verified':len(checks),
        'script_sha256':sha(__file__),'scorer_sha256':sha(args.scorer),'task_points_sha256':sha(args.task_points),
        'completed_capture_manifest':str(capture_manifest) if capture_manifest else None,
        'completed_capture_manifest_sha256':sha(capture_manifest) if capture_manifest else None,
        'captures':len(captures),'second_pass_rows_per_snapshot':total_rows,'results':results,
        'environment_steps':0,'update_calls':0,'optimizer_steps':0,
        'interpretation':'Development reference-state hypothetical PPO-form gradients. Greedy final64 categorical scores are not stochastic behavior probabilities. Neither original training gradients nor causal/off-policy estimates. No optimizer/adaptive-state simulation; clipping alone cannot prove harmful learning.',
        'input_boundary':'Only bound selected captures; first-pass states not used for gradients; no additional frame/vehicle selection.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'output':str(args.output),'captures':len(captures),'second_pass_rows':total_rows,'optimizer_steps':0,'environment_steps':0}))


if __name__=='__main__':main()
