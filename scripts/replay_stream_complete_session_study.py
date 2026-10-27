"""Nine prospectively selected fixed-final complete-session checkpoint replays."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.probe_stream_main_opportunity import file_hash,run_episode
from scripts.train_joint_stream_control import make_env
from scripts.run_stream_complete_session_study import CONTRACT
from src.agents.sa_ghmappo_agent import IntentAwareStreamSAGHMAPPOAgent as Main


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs=3,required=True)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    campaign=json.loads(a.campaign.read_text());assert campaign['contract']==CONTRACT
    for name,h in campaign['source_fingerprints'].items():assert file_hash(ROOT/name)==h,name
    replay=campaign['replay'];assert replay['variant']=='intent' and replay['maximum_episodes']==9
    assert replay['case_ids']==['train_000','train_038','train_067']
    assert replay['maximum_physical_frames']==4320
    plans={};run_by_seed={}
    for run in a.runs:
        plan=json.loads((run/'protocol.json').read_text());done=json.loads((run/'complete.json').read_text())
        assert not plan['preflight'] and file_hash(run/'cells.jsonl')==done['cells_sha256']
        assert plan['campaign_sha256']==file_hash(a.campaign) and plan['source_commit']==campaign['source_commit']
        assert plan['seed'] not in plans;plans[plan['seed']]=plan;run_by_seed[plan['seed']]=run
    assert set(plans)=={7,17,27}
    a.output.mkdir(parents=True,exist_ok=False)
    verified=[];parents=[];physical_frames=0
    for seed in campaign['seeds']:
        run=run_by_seed[seed];plan=plans[seed];source=Path(plan['parent_protocol']);manifest=json.loads(source.read_text())
        assert file_hash(source.parent/'protocol.pt')==manifest['assets']['protocol.pt']
        data=torch.load(source.parent/'protocol.pt',weights_only=False);cases={c['case_id']:c for c in data['cases']['train']}
        parent_root=Path(plan['parents'][str(seed)]['source_run'])
        prior={(r['variant'],r['case_id']):r for r in map(json.loads,(parent_root/'results.jsonl').read_text().splitlines())}
        for r in map(json.loads,(run/f'control_seed{seed}/evaluation.jsonl').read_text().splitlines()):
            if r['phase']=='initial':
                assert r['trace_sha256']==prior['relocation',r['case_id']]['trace_sha256']
                parents.append(dict(seed=seed,case_id=r['case_id'],trace_sha256=r['trace_sha256']))
        home=run/f'intent_seed{seed}';meta=json.loads((home/'manifest.json').read_text())
        cp=next(c for c in meta['checkpoints'] if c['phase']=='final');path=home/cp['path']
        assert file_hash(path)==cp['sha256'];agent=Main.from_checkpoint(path)
        old={(r['phase'],r['case_id']):r for r in map(json.loads,(home/'evaluation.jsonl').read_text().splitlines())}
        for cid in replay['case_ids']:
            row=run_episode(make_env(cases[cid],data,manifest),agent,a.output/f'seed{seed}_{cid}.jsonl',learned=True)
            for k,v in row.items():
                if k!='wall_seconds':assert v==old['final',cid][k],(seed,cid,k)
            physical_frames+=row['steps']
            verified.append(dict(seed=seed,case_id=cid,checkpoint_sha256=cp['sha256'],trace_sha256=row['trace_sha256'],frames=row['steps']))
            print(json.dumps(dict(seed=seed,case_id=cid,status='exact_replay')),flush=True)
    assert len(verified)==len(parents)==9 and physical_frames<=replay['maximum_physical_frames']==4320
    report=dict(reviewed_at=datetime.now(timezone.utc).isoformat(),literature_cutoff='2026-10-26',
        target_venue='CSCWD2027; TMC internal evidence policy',artifact_run_id=a.output.name,
        policy_version='tmc_review_policy_v3_20260621',git_commit=campaign['source_commit'],
        consumer_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        evidence_level='E3_REPRODUCED_NINE_PREDECLARED_TRAIN_EPISODES_ONLY',
        verdict='Unverifiable for full-paper readiness',campaign_sha256=file_hash(a.campaign),
        verified=verified,parent_initial_matches=parents,physical_frames=physical_frames,new_training_frames=0,
        limits=['Nine final episodes only, not full stochastic training/gradient reproduction',
            'Consumed train time intervals, no independent formal or holdout result'])
    (a.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(episodes=9,physical_frames=physical_frames,status='pass')))


if __name__=='__main__':main()
