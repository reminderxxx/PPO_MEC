"""Rebuild a train-only matched-rule diagnosis from complete physical traces."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.probe_stream_main_opportunity import file_hash


def compare(rows):
    grouped=defaultdict(dict)
    for row in rows:
        if row['case_id'] in grouped[row['method']]:
            raise ValueError('duplicate method/case')
        grouped[row['method']][row['case_id']]=row
    rules=['reactive','naive_prefetch','gain_prefetch']
    cases=set(grouped[rules[0]])
    if not cases or any(set(v)!=cases for v in grouped.values()):
        raise ValueError('unmatched cases')
    means={m:statistics.mean(r['reward'] for r in cells.values()) for m,cells in grouped.items()}
    strongest=max(rules,key=lambda m:means[m])
    pairs=[]
    for method,cells in grouped.items():
        if not method.startswith('sa_'):continue
        for cid,row in cells.items():
            ref=grouped[strongest][cid]
            pairs.append(dict(method=method,case_id=cid,rule=strongest,
                reward_delta=row['reward']-ref['reward'],
                completed_delta=row['completed_requests']-ref['completed_requests'],
                component_deltas={k:row['components'].get(k,0)-ref['components'].get(k,0)
                    for k in set(row['components'])|set(ref['components'])},
                status_deltas={k:row['statuses'].get(k,0)-ref['statuses'].get(k,0)
                    for k in set(row['statuses'])|set(ref['statuses'])}))
    return dict(strongest_compatible_rule=strongest,means=means,pairs=pairs,
        mean_main_minus_rule=statistics.mean(x['reward_delta'] for x in pairs),
        negative_pairs=sum(x['reward_delta'] < -1e-8 for x in pairs),
        positive_pairs=sum(x['reward_delta'] > 1e-8 for x in pairs),
        total_pairs=len(pairs))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    complete=json.loads((a.run/'complete.json').read_text())
    assert complete['results_sha256']==file_hash(a.run/'results.jsonl')
    protocol=json.loads((a.run/'protocol.json').read_text())
    assert not protocol['preflight']
    rows=[json.loads(s) for s in (a.run/'results.jsonl').read_text().splitlines()]
    assert len(rows)==complete['rows']==len(protocol['cases'])*len(protocol['methods'])
    frames=events=0
    for row in rows:
        f=a.run/f'{row["method"]}_{row["case_id"]}.jsonl'
        assert file_hash(f)==row['trace_sha256']
        native=0.;components=defaultdict(float)
        with f.open() as stream:
            for frame,line in enumerate(stream):
                x=json.loads(line)
                assert x['frame_index']==frame
                assert abs(x['reward']-sum(x['receipt']['reward_components'].values()))<1e-8
                native+=x['reward'];frames+=1;events+=len(x['events'])
                for k,v in x['receipt']['reward_components'].items():components[k]+=v
        assert abs(native-row['reward'])<1e-7
        assert frame+1==row['steps']
        assert all(abs(v-row['components'][k])<1e-7 for k,v in components.items())
    report=dict(reviewed_at=datetime.now(timezone.utc).isoformat(),literature_cutoff='2026-10-26',
        target_venue='IEEE TMC methodology / training-only development',
        artifact_run_id=a.run.name,policy_version='tmc_review_policy_v3_20260621',
        git_commit=protocol['source_commit'],evidence_level='HASH_VERIFIED_TRAINING_DIAGNOSIS_ONLY',
        verdict='Unverifiable for full-system formal readiness',score='N/S',
        verified_frames=frames,verified_events=events,**compare(rows),
        limits=['Three training time sessions, not formal independent clusters',
            'Existing compatible rule pool only; single-vehicle eviction-cost policy has a different contract',
            'Native-cost accounting differences are descriptive, not intervention-based causal attribution',
            'Checkpoint parents have prior training exposure; no holdout is read'])
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    text=['# Train-only opportunity diagnosis', '',
        f"Strongest existing compatible rule: {report['strongest_compatible_rule']}",
        f"Mean full minus rule: {report['mean_main_minus_rule']:.6f}",
        f"Positive/negative matched cells: {report['positive_pairs']}/{report['negative_pairs']} of {report['total_pairs']}", '',
        '| Method | Native return |','| --- | ---: |']
    text += [f'| {m} | {v:.6f} |' for m,v in report['means'].items()]
    text += ['', 'All traces and native reward components verified. No algorithm changed by this audit.',
        'These training diagnostics do not establish independent superiority or causal module benefit.']
    (a.output/'review.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:report[k] for k in ['strongest_compatible_rule','mean_main_minus_rule','negative_pairs','positive_pairs','verified_frames']}))


if __name__=='__main__':main()
