"""Audit every final candidate/control result on the complete train cohort."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.probe_stream_main_opportunity import file_hash
from scripts.audit_stream_main_ablation import verify_evaluation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--opportunity',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=json.loads((a.opportunity/'protocol.json').read_text())
    attrs={c['case_id']:c for c in source['cases']};all_rows=[];frames=events=0;plans=[];training_gits=set()
    for run in a.runs:
        plan=json.loads((run/'protocol.json').read_text());done=json.loads((run/'complete.json').read_text())
        training_gits.add(json.loads((Path(plan['study'])/'protocol.json').read_text())['source_commit'])
        assert file_hash(run/'results.jsonl')==done['results_sha256']
        assert set(plan['cases'])==set(attrs) and set(plan['variants'])=={'original','relocation'}
        for cp in plan['checkpoints'].values():assert file_hash(cp['path'])==cp['sha256']
        rows=[json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
        assert len(rows)==done['rows']==24
        assert {(r['variant'],r['case_id']) for r in rows}=={(v,c) for v in plan['variants'] for c in plan['cases']}
        assert all(r['seed']==plan['seed'] for r in rows)
        for row in rows:
            n,m=verify_evaluation(run,dict(row,phase=row['variant']));frames+=n;events+=m
        all_rows+=rows;plans.append(dict(path=str(run.resolve()),protocol_sha256=file_hash(run/'protocol.json')))
    assert len({r['seed'] for r in all_rows})==3 and len(all_rows)==72
    idx={(r['variant'],r['seed'],r['case_id']):r for r in all_rows};assert len(idx)==72
    frozen=[json.loads(s) for s in (a.opportunity/'results.jsonl').read_text().splitlines()]
    rule={r['case_id']:r for r in frozen if r['method']=='gain_prefetch'}
    parent={(int(r['method'].removeprefix('sa_seed')),r['case_id']):r for r in frozen if r['method'].startswith('sa_seed')}
    pairs=[]
    for seed in [7,17,27]:
        for cid in attrs:
            old=idx['original',seed,cid];new=idx['relocation',seed,cid];ref=rule[cid]
            pairs.append(dict(seed=seed,case_id=cid,session=attrs[cid]['session'],size=attrs[cid]['size'],
                optimized_minus_control=new['reward']-old['reward'],optimized_minus_rule=new['reward']-ref['reward'],
                optimized_minus_frozen_parent=new['reward']-parent[seed,cid]['reward'],
                completed_minus_control=new['completed_requests']-old['completed_requests'],
                completed_minus_rule=new['completed_requests']-ref['completed_requests'],
                component_deltas={k:new['components'][k]-old['components'][k] for k in new['components']}))
    methods={}
    for v in ['original','relocation']:
        rows=[r for r in all_rows if r['variant']==v]
        methods[v]=dict(mean_reward=statistics.mean(r['reward'] for r in rows),
            mean_completed_requests=statistics.mean(r['completed_requests'] for r in rows),
            mean_target_changes=statistics.mean(r['counts']['service_target_changed'] for r in rows),
            mean_state_restore=statistics.mean(r['statuses'].get('state_restore',0) for r in rows),
            reward_components={k:statistics.mean(r['components'][k] for r in rows) for k in rows[0]['components']})
    def grouped(field):
        return {str(key):dict(pairs=len(rows),optimized_minus_control=statistics.mean(x['optimized_minus_control'] for x in rows),
            optimized_minus_rule=statistics.mean(x['optimized_minus_rule'] for x in rows))
            for key in sorted({r[field] for r in pairs})
            for rows in [[x for x in pairs if x[field]==key]]}
    report=dict(reviewed_at=datetime.now(timezone.utc).isoformat(),literature_cutoff='2026-10-26',
        target_venue='IEEE TMC methodology / train-only support',artifact_run_id=a.output.name,
        policy_version='tmc_review_policy_v3_20260621',git_commit=sorted(training_gits),
        consumer_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        evidence_level='HASH_VERIFIED_ALL_TRAIN_CONDITIONS_FIXED_FINAL_NATIVE_TRACE_ONLY',
        verdict='Unverifiable for full-system formal readiness',score='N/S',source_runs=plans,
        verified_evaluation_rows=len(all_rows),verified_frames=frames,verified_events=events,
        methods=methods,rule_mean=statistics.mean(r['reward'] for r in rule.values()),
        frozen_parent_mean=statistics.mean(r['reward'] for r in parent.values()),
        mean_optimized_minus_control=statistics.mean(x['optimized_minus_control'] for x in pairs),
        mean_optimized_minus_rule=statistics.mean(x['optimized_minus_rule'] for x in pairs),
        positive_against_rule=sum(x['optimized_minus_rule']>1e-8 for x in pairs),
        negative_against_rule=sum(x['optimized_minus_rule']< -1e-8 for x in pairs),
        positive_against_control=sum(x['optimized_minus_control']>1e-8 for x in pairs),
        negative_against_control=sum(x['optimized_minus_control']< -1e-8 for x in pairs),
        per_seed=grouped('seed'),per_session=grouped('session'),per_size=grouped('size'),pairs=pairs,
        limitations=['Three train-time outer intervals, low statistical power; no formal CI/promotion',
            'Same twelve predeclared input conditions, not held-out or reward-selected',
            'Initial prior intervention and subsequent RL learning are separated only in the fixed three-cell ablation panel',
            'Existing compatible rule pool; modeled AI costs; no independent real deployment evidence'])
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    text=['# 全训练条件支持复核','', '| Method | Native return | Completed / 16 | Target changes | State restore |',
        '| --- | ---: | ---: | ---: | ---: |',
        f'| frozen gain_prefetch | {report["rule_mean"]:.6f} | — | — | — |',
        f'| frozen parent | {report["frozen_parent_mean"]:.6f} | — | — | — |']
    for v,r in methods.items():text.append(f'| {v} | {r["mean_reward"]:.6f} | {r["mean_completed_requests"]:.4f} | {r["mean_target_changes"]:.3f} | {r["mean_state_restore"]:.3f} |')
    text += ['',f'候选减等预算原方法：{report["mean_optimized_minus_control"]:.6f}；减冻结规则：{report["mean_optimized_minus_rule"]:.6f}。',
        f'对规则正/负配对：{report["positive_against_rule"]}/{report["negative_against_rule"]}（共36）。',
        '', '仅训练分区完整覆盖；所有正、零、负结果保留，不作正式统计优势或 holdout 结论。']
    (a.output/'review.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:report[k] for k in ['verified_evaluation_rows','mean_optimized_minus_control','mean_optimized_minus_rule','positive_against_rule','negative_against_rule']}))


if __name__=='__main__':main()
