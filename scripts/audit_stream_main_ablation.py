"""Independently reconstruct fixed-budget, main-only warm-start evidence.

Three train-time evaluation cells are descriptive. No formal significance or
full-system E2/E3 readiness is inferred from this small development package.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.probe_stream_main_opportunity import file_hash
from scripts.train_joint_tail_control import tensor_hash
from src.agents.sa_ghmappo_agent import RelocationAwareStreamSAGHMAPPOAgent as Main


def verify_evaluation(home,row):
    f=home/f'{row["phase"]}_{row["case_id"]}.jsonl'
    assert file_hash(f)==row['trace_sha256']
    total=0.;costs=defaultdict(float);events=0;frames=0
    with f.open() as stream:
        for i,line in enumerate(stream):
            frame=json.loads(line)
            assert frame['frame_index']==i
            assert abs(frame['reward']-sum(frame['receipt']['reward_components'].values()))<1e-8
            total+=frame['reward'];frames+=1;events+=len(frame['events'])
            for k,v in frame['receipt']['reward_components'].items():costs[k]+=v
    assert frames==row['steps'] and abs(total-row['reward'])<1e-7
    assert all(abs(v-row['components'][k])<1e-7 for k,v in costs.items())
    return frames,events


def summarize(rows,rule_rows):
    index={(r['variant'],r['seed'],r['phase'],r['case_id']):r for r in rows}
    if len(index)!=len(rows):raise ValueError('duplicate evaluation identity')
    seeds=sorted({r['seed'] for r in rows});variants=list(dict.fromkeys(r['variant'] for r in rows))
    case_ids=sorted({r['case_id'] for r in rows})
    result={};paired=[]
    for variant in variants:
        initial=[index[variant,s,'initial',cid] for s in seeds for cid in case_ids]
        final=[index[variant,s,'final',cid] for s in seeds for cid in case_ids]
        result[variant]=dict(initial_reward=statistics.mean(x['reward'] for x in initial),
            final_reward=statistics.mean(x['reward'] for x in final),
            learned_increment=statistics.mean(b['reward']-a['reward'] for a,b in zip(initial,final)),
            final_completed_requests=statistics.mean(x['completed_requests'] for x in final),
            final_target_changes=statistics.mean(x['counts']['service_target_changed'] for x in final),
            final_state_restore=statistics.mean(x['statuses'].get('state_restore',0) for x in final),
            final_components={k:statistics.mean(x['components'].get(k,0) for x in final) for k in final[0]['components']},
            final_per_seed={str(s):statistics.mean(index[variant,s,'final',cid]['reward'] for cid in case_ids) for s in seeds})
    for s in seeds:
        for cid in case_ids:
            old=index['original',s,'final',cid];new=index['relocation',s,'final',cid]
            rule=rule_rows[cid]
            paired.append(dict(seed=s,case_id=cid,optimized_minus_original=new['reward']-old['reward'],
                optimized_minus_rule=new['reward']-rule['reward'],
                completed_minus_original=new['completed_requests']-old['completed_requests'],
                initial_optimized_minus_original=index['relocation',s,'initial',cid]['reward']-index['original',s,'initial',cid]['reward']))
    effects={v:statistics.mean(index['relocation',s,'final',cid]['reward']-index[v,s,'final',cid]['reward']
        for s in seeds for cid in case_ids) for v in variants if v not in ['original','relocation']}
    return dict(methods=result,paired=paired,module_full_minus_deleted=effects,
        strongest_compatible_rule_mean=statistics.mean(rule_rows[cid]['reward'] for cid in case_ids),
        optimization_effect=statistics.mean(x['optimized_minus_original'] for x in paired),
        optimized_minus_rule=statistics.mean(x['optimized_minus_rule'] for x in paired))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--opportunity',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    plan=json.loads((a.run/'protocol.json').read_text());assert not plan['preflight']
    complete=json.loads((a.run/'complete.json').read_text())
    assert file_hash(a.run/'cells.jsonl')==complete['cells_sha256']
    cells=[json.loads(s) for s in (a.run/'cells.jsonl').read_text().splitlines()]
    assert len(cells)==complete['cells']==len(plan['variants'])*len(plan['seeds'])
    assert len({(x['variant'],x['seed']) for x in cells})==len(cells)
    parents=defaultdict(set);actors=defaultdict(set);evaluations=[];frames=events=training_frames=0;checkpoints=0
    gradient_steps={}
    for cell in cells:
        home=a.run/cell['path'];meta=json.loads((home/'manifest.json').read_text())
        assert meta['status']=='complete' and meta['actual_frames']==plan['actual_training_frames_per_cell']
        assert meta['parameter_count']==plan['parameter_count']
        assert len(meta['updates'])==plan['update_calls_per_cell']
        parents[cell['seed']].add(meta['initial_tensor_sha256']);actors[cell['seed']].add(meta['initial_actor_sha256'])
        for f,sha in meta['output_sha256'].items():assert file_hash(home/f)==sha
        for cp in meta['checkpoints']:
            f=home/cp['path'];assert file_hash(f)==cp['sha256'];q=Main.from_checkpoint(f)
            for flag,value in plan['variants'][cell['variant']].items():
                assert q.options[flag]==value,(cell['variant'],flag)
            expected=meta['initial_tensor_sha256'] if cp['phase']=='initial' else meta['final_tensor_sha256']
            assert tensor_hash(q)==expected;checkpoints+=1
        train=[json.loads(s) for s in (home/'training.jsonl').read_text().splitlines()]
        assert [r['case_id'] for r in train]==plan['training_cases']
        assert all(r['actual_frames']==plan['frames_per_training_case'] for r in train)
        frame_counts=defaultdict(int);returns=defaultdict(float)
        with (home/'training_trace.jsonl').open() as stream:
            for line in stream:
                x=json.loads(line);cid=x['case_id']
                assert x['unit_frame']==frame_counts[cid]
                assert abs(x['reward']-sum(x['receipt']['reward_components'].values()))<1e-8
                frame_counts[cid]+=1;training_frames+=1;returns[cid]+=x['reward']
        assert all(frame_counts[r['case_id']]==r['actual_frames'] and abs(returns[r['case_id']]-r['reward'])<1e-7 for r in train)
        gradient_steps[home.name]=sum(x['gradient_steps'] for x in meta['updates'])
        ev=[json.loads(s) for s in (home/'evaluation.jsonl').read_text().splitlines()]
        assert {(r['phase'],r['case_id']) for r in ev}=={(ph,c) for ph in ['initial','final'] for c in plan['evaluation_cases']}
        assert len(ev)==2*len(plan['evaluation_cases'])
        for row in ev:
            n,m=verify_evaluation(home,row);frames+=n;events+=m
            evaluations.append(dict(variant=cell['variant'],seed=cell['seed'],**row))
    assert all(len(v)==1 for v in parents.values()) and all(len(v)==1 for v in actors.values())
    source_manifest=json.loads(Path(plan['parent_protocol']).read_text())
    for name,expected in source_manifest['source_fingerprints'].items():
        if name.startswith('src/') and name!='src/agents/sa_ghmappo_agent.py':assert file_hash(ROOT/name)==expected,name
    for key,path in [('baseline_source_sha256','src/envs/core/joint_stream_rules.py'),
        ('environment_source_sha256','src/envs/core/multi_request_stream_env.py'),('actor_source_sha256','src/agents/ppo_agent.py'),
        ('main_source_sha256','src/agents/sa_ghmappo_agent.py')]:assert file_hash(ROOT/path)==plan[key]
    original=[json.loads(s) for s in (a.opportunity/'results.jsonl').read_text().splitlines()]
    old={(r['method'],r['case_id']):r for r in original}
    for row in evaluations:
        if row['variant']=='original' and row['phase']=='initial':
            ref=old[f'sa_seed{row["seed"]}',row['case_id']]
            assert abs(row['reward']-ref['reward'])<1e-8
            assert row['statuses']==ref['statuses'] and row['counts']==ref['counts']
    rule_rows={r['case_id']:r for r in original if r['method']=='gain_prefetch'}
    summary=summarize(evaluations,rule_rows)
    report=dict(reviewed_at=datetime.now(timezone.utc).isoformat(),literature_cutoff='2026-10-26',
        target_venue='IEEE TMC methodology / bounded development only',artifact_run_id=a.run.name,
        policy_version='tmc_review_policy_v3_20260621',git_commit=plan['source_commit'],
        consumer_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        evidence_level='HASH_VERIFIED_MATCHED_BUDGET_WARM_START_AND_NATIVE_TRACE_AUDIT_ONLY',
        verdict='Unverifiable for full-system formal readiness',score='N/S',
        verification=dict(cells=len(cells),checkpoints=checkpoints,training_frames=training_frames,
            evaluation_frames=frames,evaluation_events=events,initial_parent_identity='equal across variants per seed',
            originals_reproduce_frozen_diagnosis=True,other_algorithm_environment_sources='unchanged',gradient_steps=gradient_steps),
        **summary,
        limits=['Three training-time evaluation cells; no independent formal inference or sealed holdout',
            'All candidates warm-start the same full-method parents; not from-scratch module necessity evidence',
            'A new common fixed-fragment/critic-bootstrap training protocol, not old full-episode MC exact continuation',
            'Equal actual environment frames and update calls; realized active decisions and KL-stopped gradient counts can differ',
            'DAG ablation deletes adjacency only; other public DAG-derived features remain',
            'Prediction ablation also changes anticipation feasibility gates; not accuracy-only intervention',
            'Compatible existing rule pool only; proxy model costs, no measured real AI inference',
            'No all-update independent numerical reproduction, convergence or novelty claim'])
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 主方法等帧消融结果（开发证据）','',
        '| Variant | Initial reward | Final reward | RL increment | Completed / 16 | Target changes |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,r in summary['methods'].items():
        lines.append(f'| {name} | {r["initial_reward"]:.6f} | {r["final_reward"]:.6f} | {r["learned_increment"]:.6f} | {r["final_completed_requests"]:.4f} | {r["final_target_changes"]:.3f} |')
    lines += ['',f'冻结 gain_prefetch 在相同三个评价条件的均值：{summary["strongest_compatible_rule_mean"]:.6f}。',
        f'优化候选减等预算原方法：{summary["optimization_effect"]:.6f}；候选减规则：{summary["optimized_minus_rule"]:.6f}。',
        '', '组件删除结果、所有正负配对与完整口径见 analysis.json。',
        '初始 prior 干预收益与训练增量分别列出；不得把前者归为 RL 学会的收益。',
        '这是三训练时段上的有界 warm-start 消融，不支持正式独立优势、模块普遍必要或稳定收敛。']
    (a.output/'review.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(verification=report['verification'],optimization_effect=summary['optimization_effect'],optimized_minus_rule=summary['optimized_minus_rule'])))


if __name__=='__main__':main()
