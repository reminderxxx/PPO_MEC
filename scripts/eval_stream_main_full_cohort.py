"""Fixed-final train-only support over all original opportunity conditions."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.probe_stream_main_opportunity import file_hash,select_train_cases,run_episode
from scripts.train_joint_stream_control import make_env
from src.agents.sa_ghmappo_agent import RelocationAwareStreamSAGHMAPPOAgent as Main


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[7,17,27],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait',action='store_true')
    a=p.parse_args();torch.set_num_threads(1)
    prospective=json.loads(a.plan.read_text())
    assert prospective['no_candidate_final_result_available_at_freeze']
    study=json.loads((a.study/'protocol.json').read_text())
    for key,name in [('main_source_sha256','src/agents/sa_ghmappo_agent.py'),
        ('actor_source_sha256','src/agents/ppo_agent.py'),('environment_source_sha256','src/envs/core/multi_request_stream_env.py')]:
        assert file_hash(ROOT/name)==study[key],name
    source=json.loads(Path(study['parent_protocol']).read_text())
    source_home=Path(study['parent_protocol']).parent
    assert file_hash(source_home/'protocol.pt')==source['assets']['protocol.pt']
    data=torch.load(source_home/'protocol.pt',weights_only=False)
    cases=select_train_cases(data)
    a.output.mkdir(parents=True,exist_ok=False)
    output_plan=dict(created_at=datetime.now(timezone.utc).isoformat(),seed=a.seed,
        command=sys.argv,consumer_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        study=str(a.study.resolve()),study_protocol_sha256=file_hash(a.study/'protocol.json'),
        prospective_plan=str(a.plan.resolve()),prospective_plan_sha256=file_hash(a.plan),
        cases=[c['case_id'] for c in cases],variants=prospective['variants'],
        scope='All 12 original training conditions; final checkpoint only; no new training',checkpoints={})
    (a.output/'protocol.json').write_text(json.dumps(output_plan,indent=2)+'\n')
    for _ in range(7200 if a.wait else 1):
        if all((a.study/f'{v}_seed{a.seed}'/'manifest.json').exists() for v in prospective['variants']):break
        time.sleep(1)
    else:raise ValueError('Both fixed-final scientific cells must finish before support evaluation')
    with (a.output/'results.jsonl').open('x') as output:
        for variant in prospective['variants']:
            home=a.study/f'{variant}_seed{a.seed}'
            manifest=json.loads((home/'manifest.json').read_text())
            assert manifest['status']=='complete' and manifest['actual_frames']==3072
            cp=next(c for c in manifest['checkpoints'] if c['phase']=='final')
            checkpoint=home/cp['path'];assert file_hash(checkpoint)==cp['sha256']
            output_plan['checkpoints'][variant]=dict(path=str(checkpoint.resolve()),sha256=cp['sha256'])
            for c in cases:
                q=Main.from_checkpoint(checkpoint)
                trace=a.output/f'{variant}_{c["case_id"]}.jsonl'
                row=run_episode(make_env(c,data,source),q,trace,learned=True)
                output.write(json.dumps(dict(variant=variant,seed=a.seed,case_id=c['case_id'],**row))+'\n');output.flush()
                print(json.dumps(dict(seed=a.seed,variant=variant,case_id=c['case_id'],phase='full_train_cohort_evaluation')),flush=True)
    (a.output/'protocol.json').write_text(json.dumps(output_plan,indent=2)+'\n')
    (a.output/'complete.json').write_text(json.dumps(dict(status='complete',rows=24,
        results_sha256=file_hash(a.output/'results.jsonl')),indent=2)+'\n')


if __name__=='__main__':main()
