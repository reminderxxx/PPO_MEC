"""Read-only stage4 paired-SA artifact audit. No environment, policy scoring or updates.

All numerical comparisons use the predeclared full development panel and fixed
final64. Passing integrity is distinct from satisfying scientific effect gates.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

ARMS=('sa_legacy_clip','sa_group_clip')
SEEDS=(7,17,27)
POINTS=tuple(range(0,65,8))
TOL=1e-8
STRATA=('size','arrival_mode','families','capacity','session')
GRAD_GROUPS=('policy','value','surrogate')
ROOT=Path(__file__).resolve().parents[1]


def require(condition,message):
    if not condition: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def rows(path):
    path=Path(path); opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as f:
        for number,line in enumerate(f,1):
            if line.strip():
                try:yield json.loads(line)
                except Exception as exc:raise ValueError(f'{path}:{number}: invalid JSON') from exc


def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def finite(value,label='numeric value'):
    require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value),f'{label} must be finite numeric')
    return float(value)


def close(a,b,label,tol=TOL):
    require(abs(finite(a,label)-finite(b,label))<=tol,f'{label}: {a} != {b}')


def identity(row):return (int(row['seed']),int(row['episode']),row['case_id'])


def bound_path(base,path):
    p=Path(path);p=(p if p.is_absolute() else base/p).resolve()
    require(p.is_relative_to(base.resolve()),f'path escapes artifact root: {p}')
    return p


def dump(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def ratio(a,b):return a/b if b else None


def aggregate(selected):
    require(bool(selected),'cannot aggregate empty panel')
    n=len(selected);req=sum(r['request_count'] for r in selected);done=sum(r['completed_requests'] for r in selected)
    model=sum(sum(v['transferred_bytes'] for v in r['model_cache'].values()) for r in selected)/1024
    state=sum(r['state_transfer_bytes'] for r in selected)/1024
    result={'cells':n,'reward':statistics.mean(r['reward'] for r in selected),
      'request_completion_percent':100*done/req,'completed_requests':done/n,
      'completed_nodes':statistics.mean(r['completed_nodes'] for r in selected),
      'model_transfer_kib':model/n,'model_kib_per_completed_request':ratio(model,done),
      'state_transfer_kib':state/n,'state_kib_per_completed_request':ratio(state,done)}
    for name in ('model_transfer_link_frames','state_transfer_link_frames','service_target_change_count','steps','decision_rows'):
        result[name]=statistics.mean(r[name] for r in selected)
    return result


def nonregression(candidate,reference,tol=TOL):
    specs={'reward':'min','request_completion_percent':'min','model_transfer_kib':'max','model_kib_per_completed_request':'max'}
    checks={}
    for name,direction in specs.items():
        a,b=candidate[name],reference[name]
        checks[name]=dict(candidate=a,reference=b,direction=direction,
            passed=a is not None and b is not None and (a>=b-tol if direction=='min' else a<=b+tol))
    return dict(passed=all(v['passed'] for v in checks.values()),checks=checks)


def preservation_gates(candidate,plan):
    gates=plan['promotion_gates'];tol=gates['floating_tolerance_only']
    require(tol==TOL,'preservation tolerance differs from frozen1e-8')
    results={}
    for scope,key in [('overall','fixed_final_aggregate'),('seed7','seed7'),('seed27','seed27')]:
        stats=candidate[scope];checks={}
        for gate,threshold in gates[key].items():
            name,direction=gate.rsplit('_',1);value=stats[name]
            checks[gate]={'value':value,'threshold':threshold,'passed':value is not None and
                (value>=threshold-tol if direction=='min' else value<=threshold+tol)}
        results[scope]={'checks':checks,'passed':all(x['passed'] for x in checks.values())}
    return dict(passed=all(x['passed'] for x in results.values()),scopes=results)


def tail_stability(curves,thresholds):
    require(thresholds['episodes']==[48,56,64],'tail points changed')
    require(thresholds['reward_relative_range']==.01 and thresholds['completion_percentage_point_range']==1. and
        thresholds['model_cost_relative_range']==.02,'tail thresholds changed')
    result={}
    for arm in ARMS:
        result[arm]={}
        for scope in ('overall','seed7','seed17','seed27'):
            tail=[curves[arm][str(ep)][scope] for ep in (48,56,64)]
            checks={}
            for name,fraction,absolute in [('reward',.01,False),('request_completion_percent',1.,True),('model_kib_per_completed_request',.02,False)]:
                values=[x[name] for x in tail]
                if any(v is None for v in values):checks[name]={'values':values,'passed':False,'reason':'undefined cost per completion'};continue
                spread=max(values)-min(values);limit=fraction if absolute else fraction*abs(statistics.mean(values))
                checks[name]={'values':values,'range':spread,'limit':limit,'passed':spread<=limit+TOL}
            result[arm][scope]={'checks':checks,'passed':all(c['passed'] for c in checks.values())}
    return result


def summarize(all_rows,cases,plan,tail_thresholds):
    curves={};strata=[];indexed={}
    for arm,data in all_rows.items():
        indexed[arm]={identity(r):r for r in data}
        curves[arm]={}
        for point in POINTS:
            selected=[r for r in data if r['episode']==point]
            curves[arm][str(point)]={'overall':aggregate(selected),**{f'seed{s}':aggregate([r for r in selected if r['seed']==s]) for s in SEEDS}}
            for axis in STRATA:
                for value in sorted({cases[r['case_id']][axis] for r in selected},key=str):
                    subset=[r for r in selected if cases[r['case_id']][axis]==value]
                    strata.append(dict(arm=arm,episode=point,axis=axis,stratum=value,scope='overall',**aggregate(subset)))
                    for seed in SEEDS:strata.append(dict(arm=arm,episode=point,axis=axis,stratum=value,scope=f'seed{seed}',**aggregate([r for r in subset if r['seed']==seed])))
    paired=[]
    for key,legacy in indexed[ARMS[0]].items():
        candidate=indexed[ARMS[1]][key]
        item={'seed':key[0],'episode':key[1],'case_id':key[2],**{k:cases[key[2]][k] for k in STRATA}}
        a,b=aggregate([candidate]),aggregate([legacy])
        for name in a:
            if name!='cells':item[name+'_delta']=None if a[name] is None or b[name] is None else a[name]-b[name]
        item['negative_reward']=item['reward_delta'] < -TOL
        paired.append(item)
    final={arm:curves[arm]['64'] for arm in ARMS}
    preservation=preservation_gates(final[ARMS[1]],plan)
    same_run={scope:nonregression(final[ARMS[1]][scope],final[ARMS[0]][scope]) for scope in ('overall','seed7','seed17','seed27')}
    # Historical preservation protects overall/7/27; show17 separately without letting it compensate others.
    same_protected=all(same_run[s]['passed'] for s in ('overall','seed7','seed27'))
    improvements={}
    for arm in ARMS:
        improvements[arm]={scope:{name:(None if final[arm][scope][name] is None or curves[arm]['0'][scope][name] is None
            else final[arm][scope][name]-curves[arm]['0'][scope][name]) for name in ('reward','request_completion_percent','model_kib_per_completed_request')}
            for scope in ('overall','seed7','seed17','seed27')}
    recovery={name:improvements[ARMS[1]]['seed17'][name]>=-TOL for name in ('reward','request_completion_percent')}
    increment=final[ARMS[1]]['overall']['reward']>final[ARMS[0]]['overall']['reward']+TOL
    stability=tail_stability(curves,tail_thresholds)
    return dict(curves=curves,strata=strata,paired_cells=paired,paired_negative_count=sum(r['negative_reward'] for r in paired),
      historical_preservation=preservation,same_run_final64_nonregression=same_run,
      fixed_final64_incremental_utility_supported=preservation['passed'] and same_protected and increment,
      strict_overall_reward_increment=increment,final64_minus_own0=improvements,
      seed17_own0_recovery={'checks':recovery,'passed':all(recovery.values()),'resolution_claim_allowed':False,
         'reason':'Reward/completion recovery is separate from unquantified continuity/mechanism improvement; no automatic resolved claim.'},
      tail_stability=stability,candidate_bounded_tail_stability_with_gates=preservation['passed'] and same_protected and all(x['passed'] for x in stability[ARMS[1]].values()))


def tensor_hash(state,actor=False):
    h=hashlib.sha256()
    for name,value in state.items():
        if not actor or name.startswith('heads.'):
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def audit_checkpoints(home,manifest,arm,contract,historical,torch):
    records=manifest['checkpoints'];require(len(records)==27,f'{arm}: need27 checkpoints including0')
    require({(r['seed'],r['episode']) for r in records}=={(s,p) for s in SEEDS for p in POINTS},f'{arm}: checkpoint grid')
    out={}
    for c in records:
        s,p=int(c['seed']),int(c['episode']);path=bound_path(home,c['path'])
        require(path.name==f'{arm}_seed{s}_ep{p}.pt','checkpoint naming mismatch')
        require(sha(path)==c['sha256'],'checkpoint file hash mismatch')
        saved=torch.load(path,map_location='cpu',weights_only=False);state=saved['state_dict'];options=saved['options']
        require(saved['contract']=='stream_parameterized_ppo_v2' and saved['action_contract']=='joint_stream_control_v2','checkpoint contract mismatch')
        flag=options.get('separate_policy_value_clip',False);expected=arm==ARMS[1]
        require(type(flag) is bool and flag is expected,'checkpoint candidate flag mismatch')
        expected_options=dict(historical['_options_by_seed'][s])
        if expected:expected_options['separate_policy_value_clip']=True
        require(options==expected_options,'checkpoint options differ beyond the single clip flag')
        require((saved.get('training_gradient_contract')=='sa_policy_value_clip_v1') if expected else ('training_gradient_contract' not in saved),'checkpoint gradient tag mismatch')
        require(saved.get('training_resume_supported') is False,'checkpoint must not claim complete training resume')
        require(int(state['completed_updates'])==p//4,'checkpoint completed update count mismatch')
        for n,t in state.items():require(bool(torch.isfinite(t).all()),f'nonfinite checkpoint tensor {n}')
        hashes={'tensor_sha256':tensor_hash(state),'actor_sha256':tensor_hash(state,True)}
        require(hashes['tensor_sha256']==c['trained_tensor_sha256'] and hashes['actor_sha256']==c['trained_actor_sha256'],'checkpoint tensor metadata mismatch')
        initial=contract['expected_initial'][str(s)]
        require(c['initial_tensor_sha256']==initial['tensor_sha256'] and c['initial_actor_sha256']==initial['actor_sha256'],'checkpoint initial hash metadata mismatch')
        if p==0:require(hashes==initial,'exact initial checkpoint mismatch')
        if arm==ARMS[0] and p in (32,64):
            old=next(x for x in historical['checkpoints'] if x['seed']==s and x['episode']==p)
            require(hashes['tensor_sha256']==old['trained_tensor_sha256'] and hashes['actor_sha256']==old['trained_actor_sha256'],'legacy training diverged from historical exact tensors')
        require(len(saved['optimizer']['param_groups'])==1 and len(saved['optimizer']['param_groups'][0]['params'])==49,'mainAdam group changed')
        require(len(saved['surrogate_optimizer']['param_groups'])==1 and len(saved['surrogate_optimizer']['param_groups'][0]['params'])==12,'surrogateAdam group changed')
        if p==0:require(not saved['optimizer']['state'] and not saved['surrogate_optimizer']['state'],'initial optimizer not empty')
        for optimizer_name in ('optimizer','surrogate_optimizer'):
            for entry in saved[optimizer_name]['state'].values():
                for name,value in entry.items():
                    if isinstance(value,torch.Tensor):require(bool(torch.isfinite(value).all()),'nonfinite optimizer state')
        require(c['expected_updates']==p//4 and c['options_flag'] is expected and c['gradient_tag']==saved.get('training_gradient_contract'),'checkpoint provenance fields mismatch')
        hashes['_optimizer_steps']={}
        for label,name in [('main','optimizer'),('surrogate','surrogate_optimizer')]:
            opt=saved[name];ids=opt['param_groups'][0]['params']
            hashes['_optimizer_steps'][label]={str(i):float(opt['state'].get(pid,{}).get('step',0)) for i,pid in enumerate(ids)}
        out[s,p]=hashes
    return out


def clean_payload(row):return {k:v for k,v in row.items() if k not in ('arm','agent','rule','seed','episode','case_id')}


def audit_ledgers(event_path,receipt_path,dev):
    """Stream matching frame groups. Memory bound is oneframe plus2592 small cell summaries."""
    result={identity(r):r for r in dev};ordered={key:i for i,key in enumerate(result)}
    event_groups=iter(groupby(rows(event_path),key=lambda e:(identity(e),int(e['frame_index']))))
    pending=next(event_groups,None);summary={};frame_counts=Counter();last_order=-1;active_key=None
    for receipt in rows(receipt_path):
        key=identity(receipt);require(key in result,'receipt outside frozen panel');frame=int(receipt['frame_index'])
        require(frame==frame_counts[key],'receipt frame duplicated, missing or nonzero start')
        require(ordered[key]>=last_order,'receipt cells not in evaluation row order');last_order=ordered[key]
        if key!=active_key:
            active_key=key;admitted=Counter();node_targets={}
        events=[]
        if pending is not None and pending[0]==(key,frame):
            events=list(pending[1]);pending=next(event_groups,None)
        elif pending is not None:
            next_key,next_frame=pending[0]
            require(next_key in ordered and (ordered[next_key],next_frame)>(ordered[key],frame),'event frames out of order or missing receipt')
        cell=summary.setdefault(key,{'receipt_frames':0,'events':0,'reward':0.,'fees':Counter(),'totals':Counter(),
            'event_sha':hashlib.sha256(),'receipt_sha':hashlib.sha256(),'completed_requests':{},'mechanisms':Counter()})
        components=receipt['reward_components'];require(set(components)=={'completed_nodes','unfinished_wait','vehicle_compute','model_transfer','state_transfer','input_transfer','output_transfer','shared_state'},'reward component schema changed')
        reward=sum(finite(v,'receipt reward component') for v in components.values())
        close(sum(receipt['request_rewards'].values())+receipt['unassigned_team_reward'],reward,'request/team ledger')
        unfinished=sum(r['arrival_frame']<=frame and (not r['completed'] or r['completion_frame_index']>frame) for r in result[key]['per_request'])
        expected=dict(completed_nodes=sum(e['status']=='node_completed' for e in events),unfinished_wait=-.01*unfinished,
            vehicle_compute=-.02*sum(e['status'].startswith('vehicle_') for e in events),
            model_transfer=-.01*receipt['model_transfer_link_frames'],state_transfer=-.01*receipt['state_transfer_link_frames'],
            input_transfer=-.01*receipt['input_transfer_link_frames'],output_transfer=-.01*receipt['output_transfer_link_frames'],
            shared_state=-.001*receipt['shared_state_transmitted_bytes']/1024.)
        for name,value in expected.items():close(components[name],value,f'reward formula {name}')
        require(all(0<=finite(v)<=1 for v in receipt['link_usage_by_rsu'].values()),'link slot bound violated')
        for name in ('state_transfer_bytes','state_transfer_link_frames','input_transfer_bytes','input_transfer_link_frames','output_transfer_bytes','output_transfer_link_frames'):
            close(receipt[name],sum(e[name] for e in events),'event/receipt '+name)
        close(receipt['model_transfer_bytes'],sum(e['transferred_bytes'] for e in receipt['cache_events']),'cache transfer receipt')
        for event in events:
            require(event['vehicle_id'] not in {e['vehicle_id'] for e in events if e is not event},'duplicate vehicle event in frame')
            cell['event_sha'].update(canonical(clean_payload(event)))
            if event.get('frame_cost_receipt') is not None:require(event['frame_cost_receipt']==clean_payload(receipt),'embedded receipt mismatch')
            if event['workflow_completed']:
                rid=event['request_id'];require(rid not in cell['completed_requests'],'duplicate completion event');cell['completed_requests'][rid]=frame
            q=event['cache_request'];target=event['target_rsu_id']
            for obj in q.get('admitted_object_ids',[]):
                site=event['control'].get('cache_target') or target
                if admitted[site,obj]:cell['mechanisms']['same_site_object_readmissions']+=1
                if admitted[site,obj] and obj.startswith('base:'):cell['mechanisms']['same_site_base_readmissions']+=1
                admitted[site,obj]+=1
            cell['mechanisms']['state_link_queue_events']+=event['status']=='state_link_queue'
            node=(event['request_id'],event['node_id'])
            if node in node_targets and node_targets[node]!=target:cell['mechanisms']['same_node_target_changes']+=1
            node_targets[node]=target
        for name in ('model_transfer_bytes','model_transfer_link_frames','state_transfer_bytes','state_transfer_link_frames','input_transfer_bytes','input_transfer_link_frames','output_transfer_bytes','output_transfer_link_frames','shared_state_transmitted_bytes'):
            cell['totals'][name]+=finite(receipt[name],name)
        cell['fees'].update(components);cell['reward']+=reward;cell['events']+=len(events);cell['receipt_frames']+=1
        cell['receipt_sha'].update(canonical(clean_payload(receipt)));frame_counts[key]+=1
    require(pending is None,'unconsumed events after receiptEOF');require(summary.keys()==result.keys(),'missing trace cells')
    output=[]
    for key,c in summary.items():
        row=result[key];close(c['reward'],row['reward'],'session reward ledger')
        require(c['receipt_frames']==row['steps']==row['frame_receipt_count'],'receipt/session step count')
        close(c['fees']['completed_nodes'],row['completed_nodes'],'node delivery ledger')
        require(len(c['completed_requests'])==row['completed_requests']==sum(r['completed'] for r in row['per_request']),'request delivery ledger')
        for req in row['per_request']:require(req['completion_frame_index']==c['completed_requests'].get(req['request_id']),'request completion frame mismatch')
        for name,value in c['totals'].items():
            reference=sum(v['transferred_bytes'] for v in row['model_cache'].values()) if name=='model_transfer_bytes' else row[name]
            close(value,reference,'session fee ledger '+name)
        output.append(dict(seed=key[0],episode=key[1],case_id=key[2],events=c['events'],receipt_frames=c['receipt_frames'],
          reward=c['reward'],fees=dict(c['fees']),totals=dict(c['totals']),event_sha256=c['event_sha'].hexdigest(),
          receipt_sha256=c['receipt_sha'].hexdigest(),mechanisms=dict(c['mechanisms'])))
    return output


def audit_panel(rows_,arm,checkpoints,cases,training_schedule,training=False):
    seen=set()
    for row in rows_:
        key=identity(row);require(key not in seen,'duplicate panel cell');seen.add(key)
        require(row['arm']==arm and row['agent']=='sa_ghmappo','row arm/agent identity mismatch')
        for name in ('reward','steps','request_count','completed_requests','completed_nodes','state_transfer_bytes','model_transfer_link_frames','decision_rows'):
            finite(row[name],name)
        require(0<row['steps']<=480 and row['steps']==int(row['steps']),'physical frame bound')
        require(0<=row['completed_requests']<=row['request_count'],'request count bounds')
        if training:
            require(key[2]==training_schedule[str(key[0])][key[1]-1],'training order/schedule mismatch')
            require(row['completed_updates']==key[1]//4,'training completed update counter')
            losses=row['losses']
            if key[1]%4:require(not losses,'update on non-fourth session')
            else:
                require(bool(losses),'missing fourth-session update record')
                for name,value in losses.items():finite(value,'loss '+name)
        else:
            require(key[2] in cases,'unfrozen evaluationcase')
            for name in ('tensor_sha256','actor_sha256'):require(row[name]==checkpoints[key[0],key[1]][name],'eval/checkpoint hash mismatch')
    expected=({(s,ep,training_schedule[str(s)][ep-1]) for s in SEEDS for ep in range(1,65)} if training else
        {(s,ep,c) for s in SEEDS for ep in POINTS for c in cases})
    require(seen==expected and len(rows_)==len(expected),'incomplete/extra panel grid')
    return {'sessions':len(rows_),'frames':sum(r['steps'] for r in rows_),'decision_rows':sum(r['decision_rows'] for r in rows_)}


def validate_frozen_inputs(home,repo):
    contract=read_json(home/'contract.json')
    require(contract['schema']=='sa_group_clip_pair_v1' and contract['status']=='frozen','contract not frozen')
    require(Path(contract['repo_root']).resolve()==repo.resolve() and Path(contract['output_root']).resolve()==home.resolve(),'frozen root mismatch')
    require(contract['primary_checkpoint']==64 and contract['numeric_tolerance']==TOL,'fixed final/tolerance changed')
    limits=dict(max_training_sessions=384,max_training_frames=184320,max_evaluation_sessions=2592,max_evaluation_frames=1244160,
        per_arm_max_training_frames=92160,per_arm_max_evaluation_frames=622080,per_arm_training_sessions=192,per_arm_evaluation_sessions=1296,max_frames_per_session=480)
    require(all(contract[k]==v for k,v in limits.items()),'frozen budget mismatch')
    qualification=contract['qualification'];require(qualification['max_sessions']==2 and qualification['max_frames']==960,'qualification budget mismatch')
    require(sha(qualification['receipt'])==qualification['receipt_sha256'],'qualification receipt hash mismatch')
    qr=read_json(qualification['receipt']);require(qr['status']=='pass' and qr['sessions']==2 and 0<qr['frames']<=960,'qualification incomplete or overbudget')
    require(qr['source_commit']==contract['source_commit'] and qr['training_sessions']==qr['update_calls']==qr['optimizer_steps']==0,'qualification provenance/update failure')
    require(sha(home/'qualification_contract.json')==qr['contract_sha256'],'qualification contract binding mismatch')
    require(read_json(home/'freeze_receipt.json')['contract_sha256']==sha(home/'contract.json'),'main freeze receipt mismatch')
    require(contract['arms']==dict(zip(ARMS,(False,True))),'two-arm intervention mismatch')
    require(contract['seeds']==list(SEEDS) and contract['episodes']==64 and contract['checkpoint_episodes']==list(POINTS),'training schedule bounds changed')
    require(len(contract['eval_panel_ids'])==48 and len(set(contract['eval_panel_ids']))==48,'frozen dev panel must have48 unique ids')
    git=lambda *args:subprocess.check_output(['git',*args],cwd=repo,text=True).strip()
    require(git('rev-parse','HEAD')==contract['source_commit'],'working source HEAD differs from frozenrun')
    require(not git('status','--porcelain'),'audit must use clean frozen source checkout')
    for name,expected in contract['source_fingerprints'].items():require(sha(repo/name)==expected,'frozen source hash mismatch '+name)
    source_protocol=Path(contract['source_protocol'])
    require(sha(source_protocol)==contract['source_protocol_sha256'],'source protocol hash mismatch')
    protocol=read_json(source_protocol)
    require(protocol['status']=='frozen' and protocol['protocol']=='joint_stream_learning_v2' and not protocol['git_dirty'],'historical protocol not clean/frozen')
    require(contract['training_schedule']==protocol['training_schedule'],'two-arm training order differs from historical protocol')
    for name,expected in protocol['source_fingerprints'].items():
        if name!='src/agents/sa_ghmappo_agent.py':require(sha(repo/name)==expected,'unauthorized change from original algorithm '+name)
    for name,expected in protocol['assets'].items():require(sha(source_protocol.parent/name)==expected,'protocol asset hash mismatch '+name)
    import torch
    for spec in contract['source_assets'].values():require(sha(spec['path'])==spec['sha256'],'frozen protocol asset mismatch')
    require(sha(contract['protocol_data']['path'])==contract['protocol_data']['sha256']==protocol['assets']['protocol.pt'],'protocol data binding mismatch')
    data=torch.load(contract['protocol_data']['path'],map_location='cpu',weights_only=False)
    cases={c['case_id']:c for c in data['cases']['dev']}
    require(set(cases)==set(contract['eval_panel_ids']),'dev case ids differ from original fixed48')
    require(len({tuple(c[k] for k in STRATA) for c in cases.values()})==48,'dev factorial duplicate strata')
    plan_ref=contract['preservation_plan'];plan_path=Path(plan_ref['path'])
    require(sha(plan_path)==plan_ref['sha256'],'preservation plan hash mismatch')
    plan=read_json(plan_path);require(sha(Path(plan['baseline_metrics']))==plan['baseline_metrics_sha256'],'historical gate evidence hash mismatch')
    source_run=Path(contract['source_run']);old_ref=contract['original_sa_manifest']
    require(Path(old_ref['path']).resolve()==(source_run/'sa_ghmappo/manifest.json').resolve() and sha(old_ref['path'])==old_ref['sha256'],'historical manifest binding mismatch')
    historical=read_json(old_ref['path'])
    for cp in historical['checkpoints']:require(sha(source_run/'sa_ghmappo'/cp['path'])==cp['sha256'],'historical checkpoint changed')
    historical['_options_by_seed']={}
    for seed in SEEDS:
        cp=next(c for c in historical['checkpoints'] if c['seed']==seed)
        historical['_options_by_seed'][seed]=torch.load(source_run/'sa_ghmappo'/cp['path'],map_location='cpu',weights_only=False)['options']
    require(historical['status']=='complete' and not historical['git_dirty'],'historicalSA run not complete/clean')
    for seed in SEEDS:
        previous=next(c for c in historical['checkpoints'] if c['seed']==seed)
        expected={'tensor_sha256':previous['initial_tensor_sha256'],'actor_sha256':previous['initial_actor_sha256']}
        require(contract['expected_initial'][str(seed)]==expected,'historical initialization binding mismatch')
    return contract,protocol,cases,plan,historical,torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--repo-root',type=Path,default=ROOT)
    args=parser.parse_args();home=args.root.resolve();repo=args.repo_root.resolve()
    if str(repo) not in sys.path:sys.path.insert(0,str(repo))
    out=home/'audit';out.mkdir(exist_ok=False)
    status={'status':'running','reviewed_at':datetime.now(timezone.utc).isoformat(),'artifact_run_id':str(home),
       'literature_cutoff':'not updated: numerical software audit only','target_venue':'IEEE TMC internal policy; no submission-readiness claim',
       'policy_version':'tmc_review_policy_v3_20260621','paper_evidence_level':'E0_UNAVAILABLE','paper_verdict':'Unverifiable',
       'paper_claim_ready':False,'canonical_changed':False,'formal_holdout_opened':False,
       'environment_steps':0,'optimizer_steps':0,'policy_scoring_calls':0,'audit_script_sha256':sha(__file__)}
    try:
        contract,protocol,cases,plan,historical,torch=validate_frozen_inputs(home,repo)
        status['source_commit']=contract['source_commit'];status['contract_sha256']=sha(home/'contract.json')
        all_rows={};arm_results={};traces={}
        for arm in ARMS:
            arm_home=bound_path(home,contract['arm_output_roots'][arm]);manifest=read_json(arm_home/'manifest.json')
            require(manifest['status']=='complete' and manifest['arm']==arm,'arm not complete')
            require(manifest['source_commit']==contract['source_commit'] and manifest['git_dirty'] is False,'arm source provenance mismatch')
            require(manifest['contract_sha256']==status['contract_sha256'],'arm contract binding mismatch')
            require(manifest['separate_policy_value_clip'] is contract['arms'][arm],'arm flag mismatch')
            require(bool(manifest['command']),'missing execution command')
            completed=read_json(arm_home/'result.json')
            require(completed['status']=='complete' and completed['manifest_sha256']==sha(arm_home/'manifest.json'),'arm completion receipt mismatch')
            require(manifest['no_peak_selection'] is True and manifest['final_episode']==64 and manifest['additional_replay_sessions']==0 and manifest['original_sa_manifest_and_checkpoints_unchanged'] is True and manifest['source_run_write_operations']==0,'selection/replay/source boundary violation')
            require(manifest['updates_by_seed']=={str(s):16 for s in SEEDS},'perseed update counts')
            require(manifest['guarded_training_step_attempts']==manifest['training_frames'] and manifest['guarded_evaluation_step_attempts']==manifest['evaluation_frames'],'frame precharge accounting mismatch')
            required_outputs={'training.jsonl','dev.jsonl','gradient_minibatches.jsonl','gradient_forwards.jsonl','gradient_updates.jsonl','rng_isolation.jsonl','events.jsonl.gz','receipts.jsonl.gz','command.json','run_started.json'}
            require(required_outputs<=set(manifest['output_sha256']),'required output file hashes missing')
            for name,expected in manifest['output_sha256'].items():require(sha(bound_path(arm_home,name))==expected,'output file hash mismatch '+name)
            cp=audit_checkpoints(arm_home,manifest,arm,contract,historical,torch)
            train=list(rows(arm_home/'training.jsonl'));dev=list(rows(arm_home/'dev.jsonl'))
            train_count=audit_panel(train,arm,cp,cases,contract['training_schedule'],training=True)
            eval_count=audit_panel(dev,arm,cp,cases,contract['training_schedule'])
            require(train_count['sessions']==192 and train_count['frames']<=92160,'arm training budget')
            require(eval_count['sessions']==1296 and eval_count['frames']<=622080,'arm evaluation budget')
            for field,expected in [('training_sessions',192),('training_frames',train_count['frames']),('training_decisions',train_count['decision_rows']),
                ('eval_cells',1296),('evaluation_frames',eval_count['frames']),('actual_updates',48)]:require(manifest[field]==expected,'manifest count mismatch '+field)
            gradient=audit_gradients(arm_home,arm,manifest,train)
            counters=gradient.pop('parameter_steps_at_updates')
            for (seed,point),checkpoint in cp.items():
                observed=counters[f'{seed}/{point}'] if point else {'main':{str(i):0. for i in range(49)},'surrogate':{str(i):0. for i in range(12)}}
                require(checkpoint['_optimizer_steps']==observed,'checkpoint Adam counters differ from actual observed steps')
            rng=audit_rng(arm_home,manifest,cp)
            ledger=audit_ledgers(arm_home/'events.jsonl.gz',arm_home/'receipts.jsonl.gz',dev)
            traces[arm]={ (r['seed'],r['episode'],r['case_id']):r for r in ledger }
            with (out/(arm+'_ledger.jsonl')).open('x') as f:
                for row in ledger:f.write(json.dumps(row,allow_nan=False)+'\n')
            all_rows[arm]=dev;arm_results[arm]={'training':train_count,'evaluation':eval_count,'checkpoints':len(cp),'gradients':gradient,'rng_isolation':rng}
        for seed in SEEDS:
            for case in cases:
                key=seed,0,case
                for field in ('event_sha256','receipt_sha256'):
                    require(traces[ARMS[0]][key][field]==traces[ARMS[1]][key][field],'initial two-arm physicaltrace mismatch')
        summary=summarize(all_rows,cases,plan,contract['tail_stability'])
        with (out/'paired_cells.jsonl').open('x') as f:
            for row in summary.pop('paired_cells'):f.write(json.dumps(row,allow_nan=False)+'\n')
        mechanisms=[]
        for key,a in traces[ARMS[0]].items():
            b=traces[ARMS[1]][key]
            mechanisms.append({'seed':key[0],'episode':key[1],'case_id':key[2],
                'legacy':a['mechanisms'],'candidate':b['mechanisms'],
                'delta':{k:b['mechanisms'].get(k,0)-a['mechanisms'].get(k,0) for k in set(a['mechanisms'])|set(b['mechanisms'])}})
        with (out/'mechanism_pairs.jsonl').open('x') as f:
            for row in mechanisms:f.write(json.dumps(row,allow_nan=False)+'\n')
        summary.update(arm_integrity=arm_results,primary_checkpoint=64,automatic_promotion=False,
          unchanged_limits=['consumed development panel','three seeds are not independent mobility windows','same two source time sessions','synthetic physical costs','no full baseline comparison','no mechanism ablation or causal effect proof'],
          mechanism_interpretation='same-site object/base readmissions, state_link_queue event count and same-node target changes are descriptive, not avoidable loss or causal attribution',
          mechanism_coverage_pending=['paid-before-compute departures require separate offline visit analysis; not automatically covered here','stage3 prior-versus-learned-margin counterexamples remain valid and are not negated by newclip results'])
        dump(out/'summary.json',summary)
        status.update(status='pass',integrity_passed=True,scientific_gate_passed=summary['fixed_final64_incremental_utility_supported'],
            bounded_tail_stability=summary['candidate_bounded_tail_stability_with_gates'],
            conclusion='Fixed-final development candidate gates only; no automatic promotion or convergence theorem.',
            output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    except Exception as exc:
        status.update(status='failed',integrity_passed=False,error=repr(exc),traceback=traceback.format_exc())
        dump(out/'status.json',status)
        raise
    dump(out/'status.json',status)
    print(json.dumps({'audit':str(out),'status':status['status'],'scientific_gate_passed':status['scientific_gate_passed']}))


# Observer/RNG validators are defined below before invoking main.


def numerical_close(a,b,label):
    a,b=finite(a,label),finite(b,label)
    require(abs(a-b)<=2e-7+2e-5*max(abs(a),abs(b)),f'gradient numerical mismatch {label}: {a} != {b}')


def audit_rng(home,manifest,checkpoints):
    logged=list(rows(home/'rng_isolation.jsonl'))
    require(len(logged)==27,'RNG isolation missing checkpoints')
    cp={(r['seed'],r['episode']):r for r in manifest['checkpoints']}
    require({(r['seed'],r['episode']) for r in logged}==set(cp),'RNG isolation grid mismatch')
    for row in logged:
        key=row['seed'],row['episode']
        require(row==cp[key],'RNG log differs from checkpoint provenance')
        require(row['rng_unchanged'] is True and row['rng_before']==row['rng_after'],'evaluation perturbed RNG')
        require(row['training_agent_unchanged'] is True and row['training_state_before']==row['training_state_after'],'evaluation perturbed training state/optimizer/controller')
        require(row['trained_tensor_sha256']==checkpoints[key]['tensor_sha256'],'RNG record tensor mismatch')
    return {'points_checked':27,'all_rng_and_training_state_unchanged':True}


def audit_gradients(home,arm,manifest,training):
    updates=list(rows(home/'gradient_updates.jsonl'))
    require(len(updates)==48,'missing real update observations')
    index={(r['seed'],r['episode']):r for r in updates}
    expected={(s,p) for s in SEEDS for p in range(4,65,4)}
    require(set(index)==expected,'gradient update schedule grid mismatch')
    minibatches=defaultdict(list);forwards=defaultdict(list)
    for r in rows(home/'gradient_minibatches.jsonl'):minibatches[r['seed'],r['episode']].append(r)
    for r in rows(home/'gradient_forwards.jsonl'):forwards[r['seed'],r['episode']].append(r)
    require(set(minibatches)<=expected and set(forwards)<=expected,'gradient outside actual update grid')
    training_index={(r['seed'],r['episode']):r for r in training if r['losses']}
    totals=Counter();coverage=Counter();delta_sums=Counter();norms=defaultdict(list);by_seed_episode={}
    previous_steps={};checkpoint_steps={}
    for key in sorted(index):
        record=index[key];seed,ep=key;mini=minibatches[key];fw=forwards[key]
        require(record['arm']==arm and record['update']==ep//4,'gradient update identity mismatch')
        require(record['sample_sessions']==list(range(ep-3,ep+1)),'gradient update session grouping mismatch')
        require(record['default_epochs']==4,'PPO epochs changed')
        require(record['completed_updates_before']==ep//4-1 and record['completed_updates_after']==ep//4,'update counter increment mismatch')
        require(record['observer_only'] is True and record['additional_backward_calls']==0 and record['additional_optimizer_steps']==0,'observer added learning calls')
        require(record['group_partition_disjoint_exhaustive'] is True,'parameter partition invalid')
        groups=record['group_parameter_names'];flat=[x for v in groups.values() for x in v]
        require(set(groups)==set(GRAD_GROUPS) and len(flat)==len(set(flat))==61,'group name overlap/count')
        require([len(groups[g]) for g in GRAD_GROUPS]==[21,28,12],'group parameter sizes changed')
        for g,names in groups.items():
            for name in names:
                expected_g='surrogate' if name.startswith('surrogates.') else ('value' if name.startswith(('critic.','prefix_values.','action_values.')) else 'policy')
                require(g==expected_g,'gradient parameter in wrong group')
        require(record['losses']==training_index[key]['losses'],'observer losses differ from actual training log')
        require(record['actual_backwards']==len(mini)==record['losses']['gradient_steps'],'backward count mismatch')
        require(record['actual_forward_calls']==len(fw),'forward count mismatch')
        require(record['actual_adam_steps'].get('main',0)==len(mini)==record['actual_adam_steps'].get('surrogate',0),'unexpected optimizer calls')
        require(record['all_backward_rng_unchanged'] is True,'backward observer changed RNG')
        fwindex={r['forward_id']:r for r in fw}
        require(len(fwindex)==len(fw) and sorted(fwindex)==list(range(1,len(fw)+1)),'forward ID coverage')
        for f in fw:
            require(f['arm']==arm and f['update']==ep//4,'forward identity')
            require(0<f['total_rows']<=512,'training minibatch row bound')
            require(len(f['choice_coverage'])==3 and len(f['active_head_rows'])==3,'head coverage missing')
            for j,c in enumerate(f['choice_coverage']):
                require(c['head']==j and c['rows']==f['total_rows'],'head coverage dimensions')
                require(c['forced_rows']+c['multichoice_rows']==c['rows'] and c['multichoice_rows']==f['active_head_rows'][j],'forced/active head partition')
                require(1<=c['valid_candidate_min']<=c['valid_candidate_max'],'invalid candidate coverage')
                coverage[f'head{j}_forced']+=c['forced_rows'];coverage[f'head{j}_multichoice']+=c['multichoice_rows']
        require([r['minibatch'] for r in mini]==list(range(1,len(mini)+1)),'minibatch ID gap')
        linked=set()
        for b in mini:
            require(b['forward_id'] in fwindex and b['forward_id'] not in linked,'backward without unique actualforward');linked.add(b['forward_id'])
            require(b['arm']==arm and b['update']==ep//4,'minibatch identity mismatch')
            require(b['finite'] is True and b['rng_unchanged'] is True and b['rng_before']==b['rng_after'],'backward purity/finite failure')
            finite(b['loss'],'real training loss')
            pre,post=b['preclip'],b['postclip']
            for group,total in zip(GRAD_GROUPS,(21,28,12)):
                for phase in (pre,post):
                    value=phase[group];require(finite(value['norm'])>=0,'negative gradient norm')
                    require(value['parameters_total']==total and value['parameters_with_grad']+value['none_grad_parameters']==total,'gradient availability counts')
                    require(value['elements']>=0,'negative gradient element count')
                require({k:v for k,v in pre[group].items() if k!='norm'}=={k:v for k,v in post[group].items() if k!='norm'},'clipping changed gradient availability')
                norms[group+'_pre'].append(pre[group]['norm']);norms[group+'_post'].append(post[group]['norm'])
            calls=b['clip_calls'];wanted=['policy','value','surrogate'] if arm==ARMS[1] else ['policy_value','surrogate']
            require([c['group'] for c in calls]==wanted,'clip partition/order mismatch')
            current={g:pre[g]['norm'] for g in GRAD_GROUPS}
            for c in calls:
                group=c['group'];target=('policy','value') if group=='policy_value' else (group,)
                original=math.sqrt(sum(current[g]**2 for g in target))
                numerical_close(c['returned_pre_norm'],original,'returned clip pre-norm')
                require(c['max_norm']==.5,'clip threshold changed')
                expected_coef=min(1.,.5/(c['returned_pre_norm']+1e-6));numerical_close(c['coefficient'],expected_coef,'clip coefficient')
                for g in target:current[g]*=expected_coef
                for g in GRAD_GROUPS:numerical_close(c['group_norms_after'][g]['norm'],current[g],'intermediate group norm')
            for g in GRAD_GROUPS:numerical_close(post[g]['norm'],current[g],'postclip group norm')
            if arm==ARMS[0]:require(math.hypot(post['policy']['norm'],post['value']['norm'])<=.50001,'legacy combined norm bound')
            else:require(post['policy']['norm']<=.50001 and post['value']['norm']<=.50001,'candidate separate norm bound')
            require(post['surrogate']['norm']<=.50001,'surrogate norm bound')
            require([a['optimizer'] for a in b['adam_steps']]==['main','surrogate'],'Adam count/order mismatch')
            for a in b['adam_steps']:
                opt=a['optimizer'];require(a['rng_unchanged'] is True,'Adam observation RNG changed')
                before,after=a['parameter_steps_before'],a['parameter_steps_after']
                require(set(before)==set(after) and len(before)==(49 if opt=='main' else 12),'Adam parameter index layout mismatch')
                if (seed,opt) in previous_steps:require(before==previous_steps[seed,opt],'Adam step counters discontinuous')
                else:require(all(v==0 for v in before.values()),'Adam moments not fresh atfirststep')
                changes=[after[k]-before[k] for k in before]
                require(all(v in (0,1) for v in changes),'extra/missing per-parameter Adam increments')
                expected_changes=(pre['policy']['parameters_with_grad']+pre['value']['parameters_with_grad']) if opt=='main' else pre['surrogate']['parameters_with_grad']
                require(sum(changes)==expected_changes,'Adam step coverage differs from available gradients')
                previous_steps[seed,opt]=after
                for g in GRAD_GROUPS:
                    delta=a['parameter_delta'][g]
                    require(all(finite(delta[n])>=0 for n in ('l2','changed_elements','max_abs')),'invalid parameter delta')
                    require(delta['changed_elements']==int(delta['changed_elements']),'noninteger changed parameter count')
                    if (opt=='main' and g=='surrogate') or (opt=='surrogate' and g!='surrogate'):
                        require(delta['l2']==0 and delta['changed_elements']==0 and delta['max_abs']==0,'optimizer changed othergroup')
                    delta_sums[g]+=delta['l2']
                totals[opt]+=1
            totals['backwards']+=1
        checkpoint_steps[f'{seed}/{ep}']={name:dict(previous_steps.get((seed,name),{str(i):0. for i in range(49 if name=='main' else 12)})) for name in ('main','surrogate')}
        by_seed_episode[f'{seed}/{ep}']={'backwards':len(mini),'forward_calls':len(fw),'kl_stop_without_step_forwards':len(fw)-len(mini)}
        require(len(fw)-len(mini) in (0,1),'unexpected forward/backward difference')
    require(dict(manifest['observer_adam_steps'])=={k:totals[k] for k in ('main','surrogate')},'manifest Adam totals mismatch')
    return {'actual_backwards':totals['backwards'],'actual_adam_steps':{k:totals[k] for k in ('main','surrogate')},
      'updates':by_seed_episode,'parameter_steps_at_updates':checkpoint_steps,'head_coverage':dict(coverage),'sum_actual_step_delta_l2_by_group':dict(delta_sums),
      'norm_summary':{k:{'count':len(v),'min':min(v) if v else None,'max':max(v) if v else None,'mean':statistics.mean(v) if v else None} for k,v in norms.items()},
      'observer_no_extra_calls':True,'gradient_checks_numeric_tolerance':'2e-7+2e-5 relative; scientific gates remain1e-8',
      'interpretation':'actual training gradients and actual Adam deltas, descriptive; no causal inference from clipping alone'}


if __name__=='__main__':main()
