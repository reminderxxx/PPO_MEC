"""Reconstruct complete MC labels, late-wave exposure and native cache receipts."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_stream_main_ablation import verify_evaluation
from scripts.audit_stream_cache_receipts import CacheLedger
from scripts.probe_stream_main_opportunity import file_hash
from scripts.train_joint_tail_control import tensor_hash
from scripts.run_stream_complete_session_study import CONTRACT
from src.agents.sa_ghmappo_agent import IntentAwareStreamSAGHMAPPOAgent as Main


def verify_complete_training(home,cases,horizons,plan,identity):
    """Independently derive suffix targets and physical terminal/exposure facts."""
    train=[json.loads(s) for s in (home/'training.jsonl').read_text().splitlines()]
    assert [r['case_id'] for r in train]==plan['training_cases']
    frames=defaultdict(list);ledgers={};completed=defaultdict(set);order=[]
    for line in (home/'training_trace.jsonl').open():
        x=json.loads(line);cid=x['case_id'];case=cases[cid]
        assert (x['seed'],x['variant'])==identity
        if not order or cid!=order[-1]:
            assert cid not in order;order.append(cid)
        frame=len(frames[cid]);assert x['unit_frame']==x['frame_index']==frame
        assert frame<min(horizons[cid],plan['maximum_frames_per_episode'])
        assert set(x['released_request_ids_before'])=={a.request_id for a in case['arrivals'] if a.frame<=frame}
        assert abs(x['reward']-sum(x['receipt']['reward_components'].values()))<1e-8
        if cid not in ledgers:ledgers[cid]=CacheLedger(case)
        ledgers[cid].consume(x);frames[cid].append(x)
        for event in x['events']:
            if event['workflow_completed']:
                assert event['request_id'] not in completed[cid]
                completed[cid].add(event['request_id'])
        if len(completed[cid])==len(case['arrivals']):
            assert frame==next(r['actual_frames'] for r in train if r['case_id']==cid)-1
    assert order==plan['training_cases']
    tails={};counts=defaultdict(int);ends=defaultdict(list)
    for r in train:
        cid=r['case_id'];n=len(frames[cid]);arrivals=cases[cid]['arrivals']
        assert n==r['actual_frames']==r['summary']['steps']
        assert 0<n<=horizons[cid]<=plan['maximum_frames_per_episode']
        assert r['physical_terminal'] is True and r['bootstrap_team_tail']==0
        assert r['last_scheduled_release_frame']==max(a.frame for a in arrivals)<n
        assert set(r['released_request_ids'])=={a.request_id for a in arrivals}
        assert r['summary']['completed_requests']==len(completed[cid])
        assert n==horizons[cid] or len(completed[cid])==len(arrivals)
        assert abs(sum(x['reward'] for x in frames[cid])-r['reward'])<1e-7
        tail=0.;tails[cid]={}
        for x in reversed(frames[cid]):
            tail+=x['reward']/10;tails[cid][x['frame_index']]=tail
    seen=set();last_frame={}
    for line in (home/'training_targets.jsonl').open():
        x=json.loads(line);cid=x['case_id'];frame=x['frame_index']
        assert (x['seed'],x['variant'])==identity
        assert (cid,frame) not in seen;seen.add((cid,frame))
        assert frame>last_frame.get(cid,-1);last_frame[cid]=frame
        assert frame in tails[cid]
        fleet=len({a.vehicle_id for a in cases[cid]['arrivals']})
        n=len(x['vehicle_ids']);assert n==len(x['request_ids'])==len(x['returns'])
        assert set(x['request_ids'])<=set(frames[cid][frame]['released_request_ids_before'])
        for key in ['values','returns','credit_targets','team_credit_targets','head_logs',
                    'cost_targets','prefix_values','counterfactual_baselines']:
            assert np.isfinite(np.asarray(x[key])).all(),key
        assert np.allclose(x['returns'],tails[cid][frame],rtol=1e-6,atol=1e-6)
        assert np.allclose(x['credit_targets'],tails[cid][frame]/fleet,rtol=1e-6,atol=1e-6)
        assert x['team_credit_targets']==x['credit_targets']
        counts[cid]+=n;ends[cid].append((frame,x['episode_end']))
    for r in train:
        cid=r['case_id'];assert counts[cid]==r['decision_rows']
        assert ends[cid] and sum(bool(v) for _,v in ends[cid])==1 and ends[cid][-1][1] is True
    return dict(episodes=len(train),frames=sum(len(x) for x in frames.values()),
        decision_rows=sum(counts.values()),stream_episodes=sum(cases[c]['arrival_mode']=='stream' for c in frames),
        all_late_releases_observed=True,cut_bootstrap_count=0,
        cache_receipts={c:ledger.summary() for c,ledger in ledgers.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--opportunity',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args();torch.set_num_threads(1)
    rows=[];cache=[];sources=[];plans=[];identities=set();initials=defaultdict(set)
    verification=dict(cells=0,checkpoints=0,training_frames=0,training_episodes=0,
        stream_training_episodes=0,evaluation_frames=0,evaluation_events=0,evaluation_rows=0,gradient_steps={})
    for run in a.runs:
        plan=json.loads((run/'protocol.json').read_text());plans.append(plan)
        assert plan['contract']==CONTRACT and plan['preflight']==a.preflight
        assert not plan['git_dirty'] or a.preflight
        assert file_hash(plan['diagnosis'])==plan['diagnosis_sha256']
        assert file_hash(plan['parent_protocol'])==plan['parent_protocol_sha256']
        for field,name in [('main_source_sha256','src/agents/sa_ghmappo_agent.py'),
            ('runner_source_sha256','scripts/run_stream_complete_session_study.py'),
            ('environment_source_sha256','src/envs/core/multi_request_stream_env.py'),
            ('baseline_source_sha256','src/envs/core/joint_stream_rules.py')]:assert file_hash(ROOT/name)==plan[field]
        if not a.preflight:
            campaign=Path(plan['campaign']);assert file_hash(campaign)==plan['campaign_sha256']
            for name,h in json.loads(campaign.read_text())['source_fingerprints'].items():
                assert file_hash(ROOT/name)==h,name
            claim=json.loads((campaign.parent/f'seed{plan["seed"]}_execution_claim.json').read_text())
            assert claim['campaign_sha256']==plan['campaign_sha256']
        for cp in plan['parents'].values():assert file_hash(cp['path'])==cp['sha256']
        source_path=Path(plan['parent_protocol']);source=json.loads(source_path.read_text())
        assert file_hash(source_path.parent/'protocol.pt')==source['assets']['protocol.pt']
        for name,h in source['source_fingerprints'].items():
            if name.startswith('src/') and name!='src/agents/sa_ghmappo_agent.py':assert file_hash(ROOT/name)==h
        data=torch.load(source_path.parent/'protocol.pt',weights_only=False)
        cases={c['case_id']:c for c in data['cases']['train']}
        horizons={cid:len(data['sessions'][c['session']]['frames']) for cid,c in cases.items()}
        complete=json.loads((run/'complete.json').read_text());assert file_hash(run/'cells.jsonl')==complete['cells_sha256']
        cells=[json.loads(s) for s in (run/'cells.jsonl').read_text().splitlines()]
        assert len(cells)==complete['cells']==len(plan['variants'])
        assert {x['variant'] for x in cells}==set(plan['variants'])
        for cell in cells:
            home=run/cell['path'];meta=json.loads((home/'manifest.json').read_text())
            identity=(meta['seed'],meta['variant']);assert identity not in identities;identities.add(identity)
            assert meta['status']=='complete' and meta['seed']==plan['seed']
            assert meta['training_episodes']==plan['training_episodes_per_cell']
            assert meta['actual_frames']<=plan['maximum_training_frames_per_cell']
            assert len(meta['updates'])==plan['update_calls_per_cell'] and meta['parameter_count']==73940
            initials[meta['seed']].add(meta['initial_tensor_sha256'])
            for name,h in meta['output_sha256'].items():assert file_hash(home/name)==h
            for cp in meta['checkpoints']:
                path=home/cp['path'];assert file_hash(path)==cp['sha256'];q=Main.from_checkpoint(path)
                assert q.cache_intent_enabled==(meta['variant']!='control')
                assert q.prediction_enabled==(meta['variant']!='no_prediction')
                assert q.optional_cache_enabled==(meta['variant']!='no_optional_cache')
                assert tensor_hash(q)==meta['initial_tensor_sha256' if cp['phase']=='initial' else 'final_tensor_sha256']
                verification['checkpoints']+=1
            train=verify_complete_training(home,cases,horizons,plan,identity)
            assert train['frames']==meta['actual_frames']
            verification['training_frames']+=train['frames'];verification['training_episodes']+=train['episodes']
            verification['stream_training_episodes']+=train['stream_episodes']
            verification['gradient_steps'][f'{identity[1]}/{identity[0]}']=sum(x['gradient_steps'] for x in meta['updates'])
            ev=[json.loads(s) for s in (home/'evaluation.jsonl').read_text().splitlines()]
            expected={('initial',c) for c in plan['initial_evaluation_cases']}|{('final',c) for c in plan['final_evaluation_cases']}
            assert len(ev)==len(expected) and {(r['phase'],r['case_id']) for r in ev}==expected
            for r in ev:
                n,m=verify_evaluation(home,r);assert n<=480
                verification['evaluation_frames']+=n;verification['evaluation_events']+=m
                rows.append(dict(seed=meta['seed'],variant=meta['variant'],**r))
                ledger=CacheLedger(cases[r['case_id']])
                for line in (home/f'{r["phase"]}_{r["case_id"]}.jsonl').open():ledger.consume(json.loads(line))
                cache.append(dict(seed=meta['seed'],variant=meta['variant'],phase=r['phase'],case_id=r['case_id'],**ledger.summary()))
        sources.append(dict(path=str(run.resolve()),protocol_sha256=file_hash(run/'protocol.json'),cells_sha256=complete['cells_sha256']))
    assert all(len(x)==1 for x in initials.values())
    assert set(initials)==({7} if a.preflight else {7,17,27})
    assert len({p['source_commit'] for p in plans})==len({p['runner_source_sha256'] for p in plans})==1
    verification.update(cells=len(identities),evaluation_rows=len(rows),cut_bootstrap_count=0,all_late_releases_observed=True)
    if not a.preflight:
        assert verification['cells']==12 and verification['training_episodes']==144
        assert verification['training_frames']<=69120 and verification['evaluation_rows']==180
        assert verification['evaluation_frames']<=86400
    idx={(r['variant'],r['seed'],r['phase'],r['case_id']):r for r in rows};assert len(idx)==len(rows)
    metrics={}
    for v in plans[0]['variants']:
        finals=[r for r in rows if r['variant']==v and r['phase']=='final']
        cr=[r for r in cache if r['variant']==v and r['phase']=='final']
        initial=[r for r in rows if r['variant']==v and r['phase']=='initial']
        metrics[v]=dict(reward=mean(r['reward'] for r in finals),completed_requests=mean(r['completed_requests'] for r in finals),
            target_changes=mean(r['counts']['service_target_changed'] for r in finals),
            learning_increment_initial_panel=mean(idx[v,r['seed'],'final',r['case_id']]['reward']-r['reward'] for r in initial),
            **{k:sum(o[k] for r in cr for o in r['objects'].values()) for k in ['optional_transfer_bytes',
                'optional_unused_transfer_bytes','optional_admissions','optional_observed_used']})
    pairs=[dict(seed=s,case_id=c,reward_delta=idx['intent',s,'final',c]['reward']-idx['control',s,'final',c]['reward'],
        completed_delta=idx['intent',s,'final',c]['completed_requests']-idx['control',s,'final',c]['completed_requests'])
        for s in sorted(initials) for c in plans[0]['final_evaluation_cases']]
    per_seed={str(s):dict(reward_delta=mean(r['reward_delta'] for r in pairs if r['seed']==s),
        completed_delta=mean(r['completed_delta'] for r in pairs if r['seed']==s)) for s in sorted(initials)}
    reward=metrics['intent']['reward']-metrics['control']['reward'];completion=metrics['intent']['completed_requests']-metrics['control']['completed_requests']
    unused=metrics['control']['optional_unused_transfer_bytes']
    reduction=(unused-metrics['intent']['optional_unused_transfer_bytes'])/unused if unused else None
    gate=dict(mean_reward_nonnegative=reward>=0,mean_completed_nonnegative=completion>=0,
        waste_reduction_at_least_20_percent=reduction is not None and reduction>=.20,
        no_seed_loses_over_quarter_request=all(r['completed_delta']>=-.25 for r in per_seed.values()))
    old=[json.loads(s) for s in (a.opportunity/'results.jsonl').read_text().splitlines()]
    rule={r['case_id']:r for r in old if r['method']=='gain_prefetch'}
    rule_mean=mean(rule[c]['reward'] for c in plans[0]['final_evaluation_cases'])
    report=dict(reviewed_at=datetime.now(timezone.utc).isoformat(),literature_cutoff='2026-10-26',
        target_venue='CSCWD2027; TMC internal evidence policy',artifact_run_id=a.output.name,
        policy_version='tmc_review_policy_v3_20260621',git_commit=plans[0]['source_commit'],
        consumer_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        evidence_level='HASH_VERIFIED_COMPLETE_SESSION_MC_AND_NATIVE_TRAIN_RECEIPTS_ONLY',
        verdict='Unverifiable for full-paper readiness',score='N/S',preflight=a.preflight,verification=verification,
        metrics=metrics,pairs=pairs,per_seed=per_seed,reward_delta=reward,completed_delta=completion,
        unused_bytes_reduction=reduction,internal_gate=gate,internal_development_pass=all(gate.values()) and not a.preflight,
        frozen_compatible_rule_mean=rule_mean,intent_minus_rule=metrics['intent']['reward']-rule_mean,
        module_full_minus_deleted={v:metrics['intent']['reward']-metrics[v]['reward'] for v in metrics if v not in ['intent','control']},
        source_runs=sources,cache_receipts=cache,limits=plans[0]['limits']+[
            'No whole stochastic-training or gradient numerical reproduction',
            'Three consumed train time units, not formal independent statistical evidence',
            'Initial intervention and learning separated only in the initial evaluation panel',
            'Object use is not causal benefit; no realistic RSU/network/task-quality calibration'])
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    text=['# 完整会话MC确认审查','',
        '| Variant | Reward | Completed /16 | Optional used/admitted | Unused optional bytes |',
        '| --- | ---: | ---: | ---: | ---: |']
    for v,r in metrics.items():text.append(f'| {v} | {r["reward"]:.6f} | {r["completed_requests"]:.4f} | {r["optional_observed_used"]}/{r["optional_admissions"]} | {r["optional_unused_transfer_bytes"]} |')
    text+=['',f'候选减control reward={reward:.6f}，完成={completion:.6f}；开发门通过={report["internal_development_pass"]}。',
        '同外生完整会话及预算上限，不同真实提前终止长度；不是严格等实际帧比较。',
        '训练集局部warm-start，不证明泛化、收敛或完整结构必要性。formal/holdout/support缺失，整篇仍Unverifiable。']
    (a.output/'review.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:report[k] for k in ['verification','internal_development_pass']}))


if __name__=='__main__':main()
