"""Independent budget/checkpoint/native/cache-ledger audit of the intent study."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_stream_main_ablation import verify_evaluation
from scripts.audit_stream_cache_receipts import CacheLedger
from scripts.probe_stream_main_opportunity import file_hash
from scripts.train_joint_tail_control import tensor_hash
from src.agents.sa_ghmappo_agent import IntentAwareStreamSAGHMAPPOAgent as Main


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--opportunity',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args();torch.set_num_threads(1)
    rows=[];cache=[];initials=defaultdict(set);identities=set();sources=[]
    frames=events=train_frames=checkpoints=0;gradient_steps={};plans=[]
    for run in a.runs:
        plan=json.loads((run/'protocol.json').read_text());plans.append(plan)
        assert plan['preflight']==a.preflight
        assert not plan['git_dirty'] or a.preflight
        assert file_hash(plan['diagnosis'])==plan['diagnosis_sha256']
        assert file_hash(plan['parent_protocol'])==plan['parent_protocol_sha256']
        for cp in plan['parents'].values():assert file_hash(cp['path'])==cp['sha256']
        for field, name in [('main_source_sha256','src/agents/sa_ghmappo_agent.py'),
            ('training_unit_source_sha256','scripts/run_stream_main_ablation.py'),
            ('environment_source_sha256','src/envs/core/multi_request_stream_env.py'),
            ('baseline_source_sha256','src/envs/core/joint_stream_rules.py')]:
            assert file_hash(ROOT/name)==plan[field]
        source_path=Path(plan['parent_protocol']);source=json.loads(source_path.read_text())
        assert file_hash(source_path.parent/'protocol.pt')==source['assets']['protocol.pt']
        for name,expected in source['source_fingerprints'].items():
            if name.startswith('src/') and name!='src/agents/sa_ghmappo_agent.py':
                assert file_hash(ROOT/name)==expected,name
        data=torch.load(source_path.parent/'protocol.pt',weights_only=False)
        cases={c['case_id']:c for c in data['cases']['train']}
        complete=json.loads((run/'complete.json').read_text())
        assert file_hash(run/'cells.jsonl')==complete['cells_sha256']
        cells=[json.loads(s) for s in (run/'cells.jsonl').read_text().splitlines()]
        assert len(cells)==complete['cells']==len(plan['variants'])
        assert {x['variant'] for x in cells}==set(plan['variants'])
        for cell in cells:
            home=run/cell['path'];meta=json.loads((home/'manifest.json').read_text())
            assert meta['status']=='complete' and meta['seed']==plan['seed']
            identity=(meta['variant'],meta['seed']);assert identity not in identities;identities.add(identity)
            assert meta['actual_frames']==plan['actual_training_frames_per_cell']
            assert meta['parameter_count']==plan['parameter_count']
            assert len(meta['updates'])==plan['update_calls_per_cell']
            initials[meta['seed']].add(meta['initial_tensor_sha256'])
            for f,h in meta['output_sha256'].items():assert file_hash(home/f)==h
            for cp in meta['checkpoints']:
                path=home/cp['path'];assert file_hash(path)==cp['sha256']
                q=Main.from_checkpoint(path)
                assert q.cache_intent_enabled==(meta['variant']!='control')
                assert q.prediction_enabled==(meta['variant']!='no_prediction')
                assert q.optional_cache_enabled==(meta['variant']!='no_optional_cache')
                expected=meta['initial_tensor_sha256'] if cp['phase']=='initial' else meta['final_tensor_sha256']
                assert tensor_hash(q)==expected;checkpoints+=1
            train=[json.loads(s) for s in (home/'training.jsonl').read_text().splitlines()]
            assert [x['case_id'] for x in train]==plan['training_cases']
            counts=defaultdict(int);returns=defaultdict(float);ledgers={}
            for line in (home/'training_trace.jsonl').open():
                x=json.loads(line);cid=x['case_id']
                assert x['unit_frame']==counts[cid]
                assert x['variant']==meta['variant'] and x['seed']==meta['seed']
                assert abs(x['reward']-sum(x['receipt']['reward_components'].values()))<1e-8
                if cid not in ledgers:ledgers[cid]=CacheLedger(cases[cid])
                ledgers[cid].consume(x)
                counts[cid]+=1;train_frames+=1;returns[cid]+=x['reward']
            assert all(counts[x['case_id']]==x['actual_frames']==plan['frames_per_training_case']
                and abs(returns[x['case_id']]-x['reward'])<1e-7 for x in train)
            gradient_steps[f'{meta["variant"]}/{meta["seed"]}']=sum(x['gradient_steps'] for x in meta['updates'])
            evaluations=[json.loads(s) for s in (home/'evaluation.jsonl').read_text().splitlines()]
            expected={('initial',c) for c in plan['initial_evaluation_cases']}|{('final',c) for c in plan['final_evaluation_cases']}
            assert len(evaluations)==len(expected)
            assert {(x['phase'],x['case_id']) for x in evaluations}==expected
            for row in evaluations:
                n,m=verify_evaluation(home,row);frames+=n;events+=m
                rows.append(dict(variant=meta['variant'],seed=meta['seed'],**row))
                ledger=CacheLedger(cases[row['case_id']])
                for line in (home/f'{row["phase"]}_{row["case_id"]}.jsonl').open():ledger.consume(json.loads(line))
                cache.append(dict(variant=meta['variant'],seed=meta['seed'],phase=row['phase'],
                    case_id=row['case_id'],**ledger.summary()))
        sources.append(dict(path=str(run.resolve()),protocol_sha256=file_hash(run/'protocol.json'),
            cells_sha256=complete['cells_sha256']))
    assert all(len(x)==1 for x in initials.values())
    assert set(initials)==({7} if a.preflight else {7,17,27})
    assert len({p['source_commit'] for p in plans})==1
    assert len({p['main_source_sha256'] for p in plans})==1
    idx={(r['variant'],r['seed'],r['phase'],r['case_id']):r for r in rows}
    assert len(idx)==len(rows)
    final=[r for r in rows if r['phase']=='final'];metrics={}
    for v in plans[0]['variants']:
        selected=[r for r in final if r['variant']==v]
        cr=[r for r in cache if r['variant']==v and r['phase']=='final']
        inc=[idx[v,r['seed'],'final',r['case_id']]['reward']-r['reward']
             for r in rows if r['variant']==v and r['phase']=='initial']
        metrics[v]=dict(reward=mean(r['reward'] for r in selected),
            completed_requests=mean(r['completed_requests'] for r in selected),
            target_changes=mean(r['counts']['service_target_changed'] for r in selected),
            optional_transfer_bytes=sum(o['optional_transfer_bytes'] for r in cr for o in r['objects'].values()),
            unused_optional_transfer_bytes=sum(o['optional_unused_transfer_bytes'] for r in cr for o in r['objects'].values()),
            optional_admissions=sum(o['optional_admissions'] for r in cr for o in r['objects'].values()),
            optional_observed_used=sum(o['optional_observed_used'] for r in cr for o in r['objects'].values()),
            learning_increment_three_case_panel=mean(inc),
            reward_components={k:mean(r['components'][k] for r in selected) for k in selected[0]['components']})
    per_seed={str(s):dict(reward_delta=mean(idx['intent',s,'final',cid]['reward']-idx['control',s,'final',cid]['reward']
            for cid in plans[0]['final_evaluation_cases']),
        completed_delta=mean(idx['intent',s,'final',cid]['completed_requests']-idx['control',s,'final',cid]['completed_requests']
            for cid in plans[0]['final_evaluation_cases'])) for s in sorted(initials)}
    reward_delta=metrics['intent']['reward']-metrics['control']['reward']
    completed_delta=metrics['intent']['completed_requests']-metrics['control']['completed_requests']
    unused=metrics['control']['unused_optional_transfer_bytes']
    reduction=(unused-metrics['intent']['unused_optional_transfer_bytes'])/unused if unused else None
    gate=dict(mean_reward_nonnegative=reward_delta>=0,mean_completed_nonnegative=completed_delta>=0,
        waste_reduction_at_least_20_percent=reduction is not None and reduction>=.20,
        no_seed_loses_over_quarter_request=all(r['completed_delta']>=-.25 for r in per_seed.values()))
    opportunity=[json.loads(s) for s in (a.opportunity/'results.jsonl').read_text().splitlines()]
    rule={r['case_id']:r for r in opportunity if r['method']=='gain_prefetch'}
    rule_mean=mean(rule[c]['reward'] for c in plans[0]['final_evaluation_cases'])
    pairs=[dict(seed=r['seed'],case_id=r['case_id'],intent_minus_control=r['reward']-idx['control',r['seed'],'final',r['case_id']]['reward'],
        intent_minus_rule=r['reward']-rule[r['case_id']]['reward']) for r in final if r['variant']=='intent']
    report=dict(reviewed_at=datetime.now(timezone.utc).isoformat(),literature_cutoff='2026-10-26',
        target_venue='IEEE TMC methodology / bounded train-only study',artifact_run_id=a.output.name,
        policy_version='tmc_review_policy_v3_20260621',git_commit=plans[0]['source_commit'],
        consumer_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        evidence_level='HASH_VERIFIED_MATCHED_TRAIN_BUDGET_AND_NATIVE_CACHE_RECEIPTS',
        verdict='Unverifiable for full-system formal readiness',score='N/S',preflight=a.preflight,
        verification=dict(cells=len(identities),checkpoints=checkpoints,training_frames=train_frames,
            evaluation_frames=frames,evaluation_events=events,evaluation_rows=len(rows),gradient_steps=gradient_steps),
        metrics=metrics,per_seed=per_seed,reward_delta=reward_delta,completed_delta=completed_delta,
        unused_bytes_reduction=reduction,internal_gate=gate,
        internal_development_pass=all(gate.values()) and not a.preflight,
        frozen_compatible_rule_mean=rule_mean,intent_minus_rule=metrics['intent']['reward']-rule_mean,
        pairs=pairs,module_full_minus_deleted={v:metrics['intent']['reward']-metrics[v]['reward']
            for v in metrics if v not in ['intent','control']},source_runs=sources,
        cache_receipts=cache,limits=plans[0]['limits']+[
            'Three consumed train time clusters; no formal significance, convergence or whole-architecture claim',
            'Optional object use is not counterfactual net benefit; proxy AI bytes and modeled latencies',
            'Only initial three-case panel permits separating initial restriction from later learning'])
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 准备意图候选独立审查','',
        '| Variant | Native reward | Completed / 16 | Optional used/admitted | Unused optional bytes |',
        '| --- | ---: | ---: | ---: | ---: |']
    for v,r in metrics.items():
        lines.append(f'| {v} | {r["reward"]:.6f} | {r["completed_requests"]:.4f} | {r["optional_observed_used"]}/{r["optional_admissions"]} | {r["unused_optional_transfer_bytes"]} |')
    lines += ['',f'候选减等预算对照 reward={reward_delta:.6f}；完成请求={completed_delta:.6f}。',
        f'候选减冻结兼容规则={report["intent_minus_rule"]:.6f}；内部开发门通过={report["internal_development_pass"]}。',
        '三训练时段、代理成本、局部 warm-start、预测同时影响掩码等边界继续保留。',
        '所有seed、条件、负例、训练费用与固定最终模型均保留。正式论文证据仍 Unverifiable。']
    (a.output/'review.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ['verification','reward_delta','completed_delta','unused_bytes_reduction','internal_development_pass']}))


if __name__=='__main__':main()
