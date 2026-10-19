"""Offline Stage5B conditional-credit/effective-class summary.

This module constructs no agent/env and never performs neural forward/backward or
optimizer operations. `qualification` reads JSON only. `summarize` accepts only a
completed frozen Stage5B result and SHA-bound, trusted tensor capsules; torch is
imported lazily solely for deserialization. No alternate action is executed.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

SCHEMA = 'sa_stage5_same_state_v1'
ARMS = ('sa_legacy_clip', 'sa_group_clip')
SEEDS = (7, 17, 27)
EPISODES = (0, 32, 64)
CASES = tuple('dev_' + c for c in ('000','018','021','022','038','043','045','046'))
LABELS = tuple(f'{a}:{e}' for a in ARMS for e in EPISODES) + ('reference_prior_only',)
PHASES = ('first_execution_pass', 'second_cache_event_pass')
HEADS = ('execution', 'cache', 'prepare_priority')
IDENTITY = ('arm','agent','rule','seed','episode','case_id')
BOUNDARIES = [
    'Post-selected development diagnostic; no formal/holdout or paper promotion.',
    'All six models are teacher-forced on the reference input and prefix in the same pass. Rankings are relative conditional scores, not alternative action sequences or returns.',
    'Reference is deterministic greedy. Categorical probabilities/log-probabilities are not actual behavior sampling probabilities.',
    'Factual calibration uses only the exact reference model and its second-pass greedy team MC/80 target. It is not the original stochastic-training loss, advantage, or credit.',
    'Prior-only retains reference value/Q/surrogate networks; it is a software score control, not an independently trained baseline, and receives no factual-error label.',
    'Same REFERENCE prepare + same complete rotated-priority order, holding co-controls and original own claim fields fixed, is a source-algebra sufficient one-frame physical-effect class, excluding descriptive priority. Other prepare targets retain singleton UNVERIFIED buckets: original reference reservations do not prove their decoded alternate claims. Different order/bucket does not prove different physical effects.',
    'Class probabilities are conditional categorical score mass only. Class contraction is descriptive and does not identify gradient waste, effective entropy in training, avoidable cost, or causal reward improvement.',
    'Nine frozen time quantiles can miss rare mechanisms. In particular, omission of known dev046 singleton frames220/221 does not establish absence. No extra anchors are added.',
    'The pre-step ledger is causal but is not a complete restorable environment; future arrivals and alternate physical trajectories are not reconstructed.',
    'Positive seeds7/27 and negative seed17/cells are retained without outcome filtering. Frames within sessions and cases are correlated; no independent-cluster inference is performed.',
]


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode() + b'\n'


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def check_bound(item):
    if sha(item['path']) != item['sha256']: raise ValueError('SHA mismatch: ' + str(item['path']))


def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def dump(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def emit(handle, row): handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def plain(value):
    if type(value).__module__.startswith('torch') and hasattr(value, 'tolist'):
        return value.detach().cpu().tolist()
    if isinstance(value, dict): return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [plain(v) for v in value]
    return value


def flat(value):
    value = plain(value)
    if isinstance(value, (tuple,list)):
        return [z for part in value for z in flat(part)]
    if value is None: raise ValueError('Unexpected unavailable scalar')
    if isinstance(value, bool): return [value]
    x = float(value)
    if not math.isfinite(x): raise ValueError('Nonfinite numeric evidence')
    return [x]


def scalar(value):
    z = flat(value)
    if len(z) != 1: raise ValueError('Expected one scalar')
    return z[0]


def close(actual, expected, *, atol=2e-6, rtol=2e-5):
    # Float32 producer reductions vs offline Python float64 algebra; NOT an
    # experiment effect-size or acceptance tolerance.
    if not math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol):
        raise ValueError(f'Offline algebra mismatch: {actual!r} != {expected!r}')


def categorical_roundoff_atol(logits):
    """Float32 logsumexp subtraction can lose absolute precision at large offsets.

    Four unit-roundoff-scale operations on the ORIGINAL logit magnitude are a
    conservative algebra check bound. This never alters predictions, effect
    estimates, fitted targets, or any scientific acceptance threshold.
    """
    return max(2e-6,4*(2**-23)*max(1.,max(abs(float(v)) for v in logits)))


def log_probability_roundoff_atol(logits, probability):
    bound=categorical_roundoff_atol(logits)
    if 0 < probability < 2**-126:
        # Quantized nonzero float32 subnormals have HALF-ULP absolute error,
        # not bounded relative error. Transform that quantization interval to
        # log space. Zero probabilities have no finite log(p) identity check.
        half_ulp=2**-150
        bound+=max(math.log(probability/(probability-half_ulp)),math.log((probability+half_ulp)/probability))
    return bound


def object_hash(value):
    """Exact producer type/shape hash; no project imports or model operations."""
    h = hashlib.sha256()
    def visit(x):
        module = type(x).__module__
        if module.startswith('torch') and hasattr(x, 'detach'):
            z=x.detach().cpu().contiguous(); h.update(b'tensor')
            h.update(str((z.dtype, tuple(z.shape))).encode()); h.update(z.numpy().tobytes())
        elif module.startswith('numpy') and type(x).__name__ in ('ndarray','matrix','memmap'):
            # Match producer ndarray handling; numpy scalar objects instead use
            # its type+repr fallback below. Capsules contain standard ndarrays.
            h.update(b'numpy'); h.update(str((x.dtype, x.shape)).encode()); h.update(x.copy(order='C').tobytes())
        elif isinstance(x, dict):
            h.update(b'dict')
            for k,v in sorted(x.items(), key=lambda item: repr(item[0])): visit(k); visit(v)
        elif isinstance(x, (list,tuple)):
            h.update(type(x).__name__.encode()); h.update(str(len(x)).encode())
            for v in x: visit(v)
        else:
            h.update(type(x).__name__.encode()); h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def ranking(ids, values):
    if len(ids) != len(values): raise ValueError('Candidate-vector length mismatch')
    return sorted(ids, key=lambda i: (-float(values[ids.index(i)]), i))


def ranks(ids, values): return {cid: i+1 for i,cid in enumerate(ranking(ids,values))}


def physical_order(controls, frame):
    ids = sorted(controls)
    shift = frame % len(ids) if ids else 0
    tie = ids[shift:] + ids[:shift]
    return sorted(tie, key=lambda v: -controls[v]['priority'])


def priority_classes(candidates, valid_ids, owner, team, frame):
    """Conservative source-level proof, not an environment counterfactual."""
    controls = {x['vehicle_id']: deepcopy(x['control']) for x in team['ordered_team_controls']}
    if len(controls) != len(team['ordered_team_controls']) or owner not in controls:
        raise ValueError('Invalid/duplicate team controls')
    order = physical_order(controls, frame)
    if order != team['physical_priority_order']: raise ValueError('Reference global order differs')
    co = {k:v for k,v in controls.items() if k != owner}
    fixed_owner = {k:v for k,v in controls[owner].items() if k not in ('prepare_target','priority')}
    reference_prepare=controls[owner]['prepare_target']
    groups = {}
    for cid in valid_ids:
        prepare, priority = candidates[cid]
        if type(priority) is not int or priority not in range(4): raise ValueError('Priority outside frozen schema')
        alt = deepcopy(controls); alt[owner]['prepare_target'] = prepare; alt[owner]['priority'] = priority
        if {k:v for k,v in alt.items() if k!=owner} != co:
            raise ValueError('Co-controls changed')
        if {k:v for k,v in alt[owner].items() if k not in ('prepare_target','priority')} != fixed_owner:
            raise ValueError('Other owner control fields changed')
        qualified=prepare==reference_prepare
        # Nonreference prepare may require different reservations/claims. Keep
        # every such candidate separate; never infer decoder-level equivalence.
        key = json.dumps([prepare, physical_order(alt,frame),None if qualified else cid], separators=(',',':'), sort_keys=True)
        groups.setdefault(key, dict(prepare_target=prepare, physical_priority_order=physical_order(alt,frame), candidate_ids=[],
            equivalence_status='source_sufficient_fixed_reference_claims' if qualified else 'nonreference_prepare_UNVERIFIED_singleton'))['candidate_ids'].append(cid)
    result = sorted(groups.values(), key=lambda g:min(g['candidate_ids']))
    lookup = {cid:k for k,g in enumerate(result) for cid in g['candidate_ids']}
    return dict(classes=result, candidate_class=lookup, valid_candidate_count=len(valid_ids),
        sufficient_class_count=len(result), nontrivial_class_count=sum(len(g['candidate_ids'])>1 for g in result),
        class_redundancy_lower_bound=len(valid_ids)-len(result), single_active_control=len(controls)==1,
        reference_prepare_target=reference_prepare,
        contraction_role='Reference-prepare conditional index contraction only; NOT original-training redundancy',
        co_controls_fixed=True, other_owner_fields_fixed=True,
        co_controls_sha256=object_hash(co), owner_fixed_fields_sha256=object_hash(fixed_owner),
        reference_global_order=order,
        proof_scope='Only reference-prepare groups with fixed original own claims/co-controls; nonreference prepare singleton buckets UNVERIFIED; different buckets need not differ physically; no alternative step executed')


def analyze_head(row, label, j):
    trace=row['variants'][label]; head=trace['heads'][j]; first=row['phase']==PHASES[0]
    ids=head['valid_candidate_ids']; mask=head['actual_mask']; n=len(ids)
    if not n or len(set(ids))!=n or ids!=[i for i,v in enumerate(mask) if v]: raise ValueError('Invalid valid-id/mask mapping')
    if head['valid_candidate_count']!=n: raise ValueError('Valid count differs')
    ref=int(head['reference_action']); action=int(head['greedy_action'])
    if ref not in ids or action not in ids or ref != int(row['reference_actions'][0][j]): raise ValueError('Reference/action mapping differs')
    vector={name:flat(head[name]) for name in ('valid_total_logits','valid_learned_logits','valid_public_prior','valid_other_adjustment')}
    if any(len(v)!=n for v in vector.values()): raise ValueError('Invalid head vector size')
    for k in range(n): close(vector['valid_total_logits'][k],vector['valid_learned_logits'][k]+vector['valid_public_prior'][k]+vector['valid_other_adjustment'][k])
    if label=='reference_prior_only' and (any(x!=0 for x in vector['valid_learned_logits']) or vector['valid_total_logits']!=vector['valid_public_prior']): raise ValueError('Prior-only software intervention differs')
    orders={k:ranking(ids,v) for k,v in vector.items()}
    if action!=orders['valid_total_logits'][0] or head['public_prior_argmax']!=orders['valid_public_prior'][0] or head['learned_only_argmax']!=orders['valid_learned_logits'][0]: raise ValueError('Recorded rank winner differs')
    if head['overrides_prior']!=(action!=head['public_prior_argmax']) or head['prior_changes_learned_winner']!=(action!=head['learned_only_argmax']): raise ValueError('Prior-override flag mismatch')
    probs=flat(head['distribution']['probabilities']); logs=flat(head['distribution']['log_probabilities'])
    categorical_atol=categorical_roundoff_atol(vector['valid_total_logits'])
    if len(probs)!=len(mask) or len(logs)!=len(mask) or any(p<0 for p in probs): raise ValueError('Distribution size/sign differs')
    close(sum(probs[i] for i in ids),1)
    if any(probs[i]!=0 for i,v in enumerate(mask) if not v): raise ValueError('Invalid candidate has probability mass')
    for i in ids:
        if probs[i]>0: close(math.log(probs[i]),logs[i],atol=log_probability_roundoff_atol(vector['valid_total_logits'],probs[i]))
    peak=max(vector['valid_total_logits']); norm=sum(math.exp(x-peak) for x in vector['valid_total_logits'])
    for k,i in enumerate(ids): close(probs[i],math.exp(vector['valid_total_logits'][k]-peak)/norm)
    close(probs[ref], head['reference_action_probability'])
    close(logs[ref],scalar(head['distribution']['reference_log_probability']))
    close(logs[ref],flat(trace['forward_outputs']['reference_head_log_probs'])[j])
    entropy=-sum(probs[i]*math.log(probs[i]) for i in ids if probs[i]>0)
    close(entropy, scalar(head['distribution']['entropy']),atol=categorical_atol)
    close(entropy/max(math.log(n),1),scalar(head['distribution']['normalized_entropy']),atol=categorical_atol/max(math.log(n),1))
    sur=head.get('valid_candidate_surrogate_estimates')
    if sur is None or len(sur)!=n or any(len(x)!=3 for x in sur): raise ValueError('Frozen SA surrogate schema missing')
    sur=[[float(x) for x in v] for v in sur]
    q=None if head.get('valid_candidate_q') is None else flat(head['valid_candidate_q'])
    out=trace['forward_outputs']; factual=None
    raw_first_names=('selected_q','prefix_values','value','counterfactual_baselines','selected_surrogate','active_heads')
    if first:
        if q is not None or head.get('prefix_value') is not None or out.get('critic_value') is not None or any(k in out for k in raw_first_names): raise ValueError('First-pass uncomputed value was fabricated')
        baseline=prefix=global_v=selected_q=None
    else:
        if q is None or len(q)!=n: raise ValueError('Second-pass Q missing/misaligned')
        baseline=sum(probs[i]*.5*(q[k]+sur[k][0]) for k,i in enumerate(ids))
        selected_q=q[ids.index(ref)]; prefix=scalar(head['prefix_value']); global_v=scalar(out['critic_value'])
        close(baseline,flat(out['counterfactual_baselines'])[j]); close(selected_q,flat(out['selected_q'])[j])
        close(prefix,flat(out['prefix_values'])[j]); close(global_v,scalar(out['value']))
        close(entropy/max(math.log(n),1),flat(out['normalized_entropies'])[j],atol=categorical_atol/max(math.log(n),1))
        selected_sur=out['selected_surrogate'][0][j]
        for k in range(3): close(sur[ids.index(ref)][k],selected_sur[k])
        if bool(flat(out['active_heads'])[j])!=(n>1): raise ValueError('Forced/active mismatch')
        if label==row['reference_model']:
            labels=row['episode_labels']; target=scalar(labels['team_credit_targets'])
            close(target,scalar(labels['returns'])/8)
            if 'credit_targets' in labels: close(target,scalar(labels['credit_targets']))
            values=dict(selected_q=selected_q,selected_surrogate_return=sur[ids.index(ref)][0],counterfactual_baseline=baseline,prefix_value=prefix,global_value=global_v)
            factual=dict(role='Exact-reference second-pass greedy calibration; not original training credit',
                target_team_mc_over_80=target, values=values, signed_errors={k:v-target for k,v in values.items()},
                factual_team_mc_over_10=scalar(labels['returns']))
    rank_vectors=dict(total_logits=vector['valid_total_logits'],learned_logits=vector['valid_learned_logits'],public_prior=vector['valid_public_prior'],surrogate_return=[x[0] for x in sur])
    if q is not None: rank_vectors.update(q=q,mean_q_surrogate_return=[.5*(q[k]+sur[k][0]) for k in range(n)])
    rankmaps={k:ranks(ids,v) for k,v in rank_vectors.items()}
    prior_range=max(vector['valid_public_prior'])-min(vector['valid_public_prior'])
    candidates=[]
    for k,i in enumerate(ids):
        candidates.append(dict(candidate_id=i,candidate=row['candidates'][j][i],conditional_probability=probs[i],
            conditional_log_probability=logs[i],total_logit=vector['valid_total_logits'][k],learned_logit=vector['valid_learned_logits'][k],
            log_probability_algebra_absolute_roundoff_bound=None if probs[i]==0 else log_probability_roundoff_atol(vector['valid_total_logits'],probs[i]),
            public_prior=vector['valid_public_prior'][k],other_adjustment=vector['valid_other_adjustment'][k],
            surrogate_return=sur[k][0],surrogate_model_slots_over_10=sur[k][1],surrogate_state_slots_over_10=sur[k][2],
            q=None if q is None else q[k],ranks={name:m[i] for name,m in rankmaps.items()}))
    witness=row['witness']
    # Qualification intentionally reuses an older capsule whose witness has
    # method/seed/case rather than the later B session arm/episode schema.
    # reference_model is explicit in every scored row; do not invent identity.
    reference_arm, reference_episode=row['reference_model'].rsplit(':',1)
    reference_episode=int(reference_episode)
    if witness.get('arm',reference_arm)!=reference_arm or witness.get('episode',reference_episode)!=reference_episode:
        raise ValueError('Reference model/witness identity mismatch')
    return dict(seed=witness['seed'],reference_arm=reference_arm,reference_episode=reference_episode,case_id=witness['case_id'],
        pass_name=row['phase'],head=j,head_name=HEADS[j],model=label,reference_model=row['reference_model'],
        frame=row['frame'],vehicle_id=row['vehicle_id'],node_id=row['node_id'],state_identity=row['state_identity'],
        forced=n==1,valid_candidates=n,reference_action=ref,conditional_greedy_action=action,
        reference_action_conditional_rank=rankmaps['total_logits'][ref],greedy_equals_reference=action==ref,
        reference_action_conditional_probability=probs[ref],conditional_entropy=entropy,
        categorical_algebra_absolute_roundoff_bound=categorical_atol,
        conditional_normalized_entropy=entropy/max(math.log(n),1),overrides_prior=head['overrides_prior'],
        public_prior_range=prior_range,public_prior_nonconstant=prior_range>0,
        overrides_nonconstant_prior=head['overrides_prior'] if prior_range>0 else None,
        prior_changes_learned_winner=head['prior_changes_learned_winner'],prior_argmax=head['public_prior_argmax'],
        learned_argmax=head['learned_only_argmax'],top2_margin=head['top2_margin'],
        selected_q=selected_q,counterfactual_baseline=baseline,prefix_value=prefix,global_value=global_v,
        factual_reference_calibration=factual, candidates=candidates,
        conditioning='Same input, same pass, teacher-forced complete reference prefix; no alternate rollout')


def analyze_score(row):
    if row['schema']!=SCHEMA+'_scores' or row['phase'] not in PHASES: raise ValueError('Score schema mismatch')
    if set(row['variants'])!=set(LABELS) or row['reference_model'] not in LABELS[:-1]: raise ValueError('Model set differs')
    if not row['reference_parity'] or not row['initial_arm_parity']: raise ValueError('Producer parity failed')
    if {g['label'] for g in row['guards']}!=set(LABELS) or any(g['state_before']!=g['state_after'] for g in row['guards']): raise ValueError('State/purity guards failed')
    count=1 if row['phase']==PHASES[0] else 3
    reference_masks=None
    for label in LABELS:
        trace=row['variants'][label]
        if trace['head_count']!=count or len(trace['heads'])!=count or trace['diagnostic_forward_count']!=1 or not all(trace['guards'].values()): raise ValueError('Original forward coverage/purity differs')
        masks=[h['actual_mask'] for h in trace['heads']]
        if reference_masks is not None and masks!=reference_masks: raise ValueError('Same-prefix masks differ')
        reference_masks=masks
    return [analyze_head(row,label,j) for label in LABELS for j in range(count)]


class Aggregate:
    def __init__(self): self.values=defaultdict(lambda:dict(n=0,sums=defaultdict(float),errors={}))
    def add(self,row):
        key=tuple(row[k] for k in ('seed','reference_arm','reference_episode','case_id','pass_name','head','model'))
        d=self.values[key]; d['n']+=1
        for name in ('forced','valid_candidates','greedy_equals_reference','reference_action_conditional_rank','reference_action_conditional_probability','conditional_entropy','conditional_normalized_entropy','overrides_prior','prior_changes_learned_winner','public_prior_nonconstant'):
            d['sums'][name]+=row[name]
        if row['overrides_nonconstant_prior'] is not None:
            d.setdefault('informative_prior',dict(n=0,overrides=0))['n']+=1
            d['informative_prior']['overrides']+=row['overrides_nonconstant_prior']
        for name in ('raw_index_flip_from_reference','sufficient_class_flip_from_reference','class_count','class_redundancy_lower_bound','conditional_sufficient_class_entropy'):
            if name in row: d['sums'][name]+=row[name]
        factual=row['factual_reference_calibration']
        if factual:
            for name,error in factual['signed_errors'].items():
                e=d['errors'].setdefault(name,dict(n=0,signed_sum=0.,absolute_sum=0.,squared_sum=0.)); e['n']+=1
                e['signed_sum']+=error; e['absolute_sum']+=abs(error); e['squared_sum']+=error*error
    def finish(self):
        out=[]
        for key,d in sorted(self.values.items()):
            names=('seed','reference_arm','reference_episode','case_id','pass_name','head','model')
            errors={name:dict(n=e['n'],bias=e['signed_sum']/e['n'],mae=e['absolute_sum']/e['n'],rmse=math.sqrt(e['squared_sum']/e['n'])) for name,e in d['errors'].items()}
            out.append(dict(zip(names,key),observations=d['n'],means={k:v/d['n'] for k,v in d['sums'].items()},factual_reference_calibration=errors or None,
                informative_prior=d.get('informative_prior'),
                aggregation='Descriptive captured-row means; correlated frames; no inferential cluster treatment'))
        return out


def score_key(x): return (x['frame'],x['phase'],x['vehicle_id'])


def validate_team(team, frame):
    if team['frame']!=frame: raise ValueError('Team frame identity differs')
    original={k:v for k,v in team.items() if k not in ('pre_action_payload_sha256','physical_pre_step','physical_pre_step_sha256')}
    if object_hash(original)!=team['pre_action_payload_sha256']: raise ValueError('Team original-scope hash differs')
    if object_hash(team['physical_pre_step'])!=team['physical_pre_step_sha256']: raise ValueError('Physical pre-step hash differs')
    controls={c['vehicle_id']:c['control'] for c in team['ordered_team_controls']}
    if physical_order(controls,frame)!=team['physical_priority_order']: raise ValueError('Frozen physical order differs')
    if bool(controls)!=bool(team['scored']) or bool(controls)==bool(team['idle_no_active_decision']): raise ValueError('Idle/scored boundary mismatch')


def verify_complete(root):
    c=read(root/'contract.json'); result=read(root/'result.json')
    if (root/'failure.json').exists() or result['status']!='complete' or c['status']!='frozen': raise ValueError('Require complete, nonfailed frozen run; no partial summary')
    if result['contract_sha256']!=sha(root/'contract.json') or not result['source_before_after_exact']: raise ValueError('Completion contract/source binding failed')
    if any(result[k]!=0 for k in ('training_sessions','backward_calls','optimizer_steps')): raise ValueError('Forbidden operation reported')
    if len(result['results'])!=144 or result['counts']['reference_sessions']!=144: raise ValueError('Incomplete reference panel')
    expected={(a,s,e,k) for a in ARMS for s in SEEDS for e in EPISODES for k in CASES}
    actual={(x['identity']['arm'],x['identity']['seed'],x['identity']['episode'],x['identity']['case_id']) for x in result['results']}
    if len(actual)!=144 or actual!=expected: raise ValueError('Reference panel duplicates/omissions')
    if len(c['sessions'])!=144 or {(x['arm'],x['seed'],x['episode'],x['case_id']) for x in c['sessions']}!=expected: raise ValueError('Frozen plan panel differs')
    for key,value in result['counts'].items():
        if value>c['budgets'][key] or value<0: raise ValueError('Producer budget exceeded')
    for item in c['bindings']: check_bound(item)
    for key in ('plan','qualification','a_review'): check_bound(c[key])
    q=read(c['qualification']['path']); check_bound(q['scores'])
    if q['status']!='pass' or q['environment_frames']!=0 or q['counts']['qualification_snapshots']!=16 or q['counts']['qualification_scoring_calls']!=112: raise ValueError('Qualification differs')
    repo=Path(c['repo_root'])
    for name,h in {**c['stage4_source_fingerprints'],**c['script_sha256']}.items():
        if sha(repo/name)!=h: raise ValueError('Frozen proof source changed: '+name)
    for cps in c['checkpoints'].values():
        for cp in cps.values(): check_bound(cp)
    return c,result


def summarize(args):
    if not args.trusted_tensor_capsules: raise ValueError('Explicit --trusted_tensor_capsules required for bound project .pt deserialization')
    root=args.root.resolve(); c,result=verify_complete(root)
    # No project imports, model construction, environment, forward, or optimization.
    import torch
    out=args.output_root.resolve()
    if out==root or out.is_relative_to(root): raise ValueError('Summary must not modify the original result tree')
    out.mkdir(parents=True,exist_ok=False)
    aggregate=Aggregate(); totals=dict(sessions=0,reference_frames=0,snapshots=0,second_pass_snapshots=0,conditional_scores=0,head_records=0,team_frames=0,idle_quantile_frames=0,nontrivial_priority_class_snapshots=0)
    source_assets=[]; plans={(x['arm'],x['seed'],x['episode'],x['case_id']):x for x in c['sessions']}
    with gzip.open(out/'heads.jsonl.gz','xt',encoding='utf8') as hf,gzip.open(out/'priority_classes.jsonl.gz','xt',encoding='utf8') as cf:
        for entry in result['results']:
            if entry['status']!='pass' or not entry['summary_exact']: raise ValueError('Session not exact/pass')
            for name,binding in entry['outputs'].items(): check_bound(binding)
            assets=entry['outputs']; source_assets.append(assets)
            capsule=torch.load(assets['snapshots.pt']['path'],map_location='cpu',weights_only=False)
            if capsule['schema']!=SCHEMA+'_capture' or capsule['witness']!=entry['identity']: raise ValueError('Capsule identity mismatch')
            identity=entry['identity']; key=(identity['arm'],identity['seed'],identity['episode'],identity['case_id']); plan=plans[key]
            if capsule['selected_frames']!=plan['frames'] or entry['selected_frames']!=plan['frames']: raise ValueError('Frozen quantile selection differs')
            actual=read(assets['reference_row.json']['path'])
            if {k:v for k,v in actual.items() if k not in IDENTITY}!={k:v for k,v in plan['expected_summary'].items() if k not in IDENTITY}: raise ValueError('Reference summary changed')
            teams=capsule['team_frames']
            if sorted(map(int,teams))!=plan['frames'] or entry['team_frames']!=len(teams): raise ValueError('Missing/extra quantile team ledger')
            for f,team in teams.items():
                validate_team(team,int(f)); totals['team_frames']+=1; totals['idle_quantile_frames']+=team['idle_no_active_decision']
            snaps={}
            for snap in capsule['snapshots']:
                sk=(snap['frame'],snap['phase'],snap['feature']['vehicle_id'])
                if sk in snaps: raise ValueError('Duplicate snapshot key')
                snaps[sk]=snap
            seen=set(); second=0
            for row in rows(assets['scoring.jsonl.gz']['path']):
                sk=score_key(row)
                if sk in seen or sk not in snaps: raise ValueError('Duplicate/unmatched score')
                seen.add(sk); snap=snaps[sk]
                if row['witness']!=identity or row['reference_model']!=f"{identity['arm']}:{identity['episode']}" or row['state_identity']!=snap['state_identity']: raise ValueError('Score identity/state binding differs')
                if row['reference_actions']!=plain(snap['reference_actions']) or row['candidates']!=plain(snap['feature']['candidates']): raise ValueError('Saved prefix/candidates changed')
                if row['episode_labels']!=plain(snap.get('episode_labels',{})): raise ValueError('Factual label changed')
                for field in ('heads','forward_outputs'):
                    if row['variants'][row['reference_model']][field]!=plain(snap['reference_trace'][field]): raise ValueError('Reference scored/captured trace changed')
                team=teams[str(row['frame'])]
                if row['team_frame_key']!=str(row['frame']) or row['team_frame_sha256']!=team['pre_action_payload_sha256'] or snap['team_frame_sha256']!=row['team_frame_sha256']: raise ValueError('Score/team linkage differs')
                output=analyze_score(row); effect=None
                if row['phase']==PHASES[1]:
                    second+=1; ids=row['variants'][row['reference_model']]['heads'][2]['valid_candidate_ids']
                    effect=priority_classes(row['candidates'][2],ids,row['vehicle_id'],team,row['frame'])
                    ref=int(row['reference_actions'][0][2]); ref_class=effect['candidate_class'][ref]
                    owner_control=next(x['control'] for x in team['ordered_team_controls'] if x['vehicle_id']==row['vehicle_id'])
                    if list(row['candidates'][2][ref])!=[owner_control['prepare_target'],owner_control['priority']]: raise ValueError('Head2 selected action/control differs')
                    totals['nontrivial_priority_class_snapshots']+=effect['nontrivial_class_count']>0
                    emit(cf,dict(identity=identity,frame=row['frame'],vehicle_id=row['vehicle_id'],reference_action=ref,
                        reference_class=ref_class,team_frame_sha256=row['team_frame_sha256'],physical_pre_step_sha256=team['physical_pre_step_sha256'],**effect))
                for h in output:
                    if h['head']==2:
                        lookup=effect['candidate_class']; probs={x['candidate_id']:x['conditional_probability'] for x in h['candidates']}
                        masses=[sum(probs[i] for i in group['candidate_ids']) for group in effect['classes']]
                        ent=-sum(p*math.log(p) for p in masses if p>0)
                        h.update(raw_index_flip_from_reference=h['conditional_greedy_action']!=h['reference_action'],
                            sufficient_class_flip_from_reference=lookup[h['conditional_greedy_action']]!=lookup[h['reference_action']],
                            class_count=effect['sufficient_class_count'],class_redundancy_lower_bound=effect['class_redundancy_lower_bound'],
                            conditional_sufficient_class_masses=masses,conditional_sufficient_class_entropy=ent,
                            sufficient_class_scope=effect['proof_scope'])
                        h['class_flip_interpretation']='Different conservative bucket/order signature is UNQUALIFIED, not proof of different physical effect; nonreference prepare never merged'
                        for candidate in h['candidates']: candidate['sufficient_class']=lookup[candidate['candidate_id']]
                    aggregate.add(h); emit(hf,h); totals['head_records']+=1
                totals['snapshots']+=1; totals['conditional_scores']+=len(row['variants'])
            if seen!=set(snaps) or len(seen)!=entry['saved_calls'] or second!=entry['second_pass_calls']: raise ValueError('Snapshot/scoring coverage differs')
            totals['second_pass_snapshots']+=second; totals['reference_frames']+=entry['reference_frames']; totals['sessions']+=1
    for local,producer in (('sessions','reference_sessions'),('reference_frames','reference_frames'),('snapshots','saved_calls'),('second_pass_snapshots','second_pass_calls'),('conditional_scores','scoring_calls')):
        if totals[local]!=result['counts'][producer]: raise ValueError('Independent summary count mismatch: '+local)
    if totals['snapshots']!=2*totals['second_pass_snapshots']: raise ValueError('Two-pass capture mismatch')
    with (out/'by_reference_condition_head.jsonl').open('x',encoding='utf8') as f:
        for x in aggregate.finish(): emit(f,x)
    summary=dict(status='complete',schema='sa_stage5_offline_credit_summary_v1',reviewed_at=datetime.now(timezone.utc).isoformat(),
        literature_cutoff='2026-10-17; inherited, no novelty refresh',target_venue='IEEE TMC internal diagnostic/no readiness claim',
        artifact_run_id=root.parent.name+'/'+root.name,policy_version='tmc_review_policy_v3_20260621',git_commit=c['source_commit'],evidence_level='E0_UNAVAILABLE for formal/holdout; complete development saved-data diagnostic',
        script_status='Saved-data audit only; no algorithm or scientific claim promotion',
        command=sys.argv,script_sha256=sha(__file__),contract=dict(path=str(root/'contract.json'),sha256=sha(root/'contract.json')),
        producer_result=dict(path=str(root/'result.json'),sha256=sha(root/'result.json')),source_assets=source_assets,counts=totals,
        helper_operations=dict(environment_frames=0,model_construction=0,neural_forward=0,backward=0,optimizer_steps=0),
        retained_seed_roles=dict(positive_reference_seeds=[7,27],negative_reference_seed=17,selection='All frozen cells/conditions retained; roles are historical descriptive labels, not a result filter'),
        calibration_units='physical reward suffix /10 = returns; returns /8 = factual team MC/80 target. Model/state cost surrogates are slots/10, not MC/80.',
        zero_witness_interpretation='No captured witness is UNWITNESSED, not absent; frozen quantiles can omit220/221.',
        causal_effect='UNVERIFIED',original_training_credit='NOT_RECONSTRUCTED',boundaries=BOUNDARIES,
        outputs={name:dict(path=str(out/name),sha256=sha(out/name)) for name in ('heads.jsonl.gz','priority_classes.jsonl.gz','by_reference_condition_head.jsonl')})
    dump(out/'summary.json',summary)
    return dict(status='complete',summary=str(out/'summary.json'),counts=totals)


def qualification(args):
    """Offline arithmetic audit of the already generated16/112 qualification."""
    out=args.output_root.resolve(); counts=dict(snapshots=0,variants=0,head_records=0,first_pass=0,second_pass=0)
    for row in rows(args.scores):
        records=analyze_score(row); counts['snapshots']+=1; counts['variants']+=len(row['variants']); counts['head_records']+=len(records)
        counts['first_pass' if row['phase']==PHASES[0] else 'second_pass']+=1
    if counts!=dict(snapshots=16,variants=112,head_records=224,first_pass=8,second_pass=8): raise ValueError('Qualification16/112 shape differs')
    out.mkdir(parents=True,exist_ok=False)
    value=dict(status='pass',scope='Offline algebra/shape/purity evidence only; no new model calls',
        scores=dict(path=str(args.scores.resolve()),sha256=sha(args.scores)),script_sha256=sha(__file__),counts=counts,
        helper_operations=dict(environment_frames=0,model_construction=0,neural_forward=0,backward=0,optimizer_steps=0),boundaries=BOUNDARIES)
    dump(out/'qualification_algebra.json',value); return value


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('qualification'); q.add_argument('--scores',type=Path,required=True); q.add_argument('--output_root',type=Path,required=True)
    s=sub.add_parser('summarize'); s.add_argument('--root',type=Path,required=True); s.add_argument('--output_root',type=Path,required=True)
    s.add_argument('--trusted_tensor_capsules',action='store_true')
    args=p.parse_args(); result=qualification(args) if args.command=='qualification' else summarize(args)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__': main()
