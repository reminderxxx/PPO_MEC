"""Three prospective query observations, bounded payload, no retry or science."""
import argparse
import csv
import json
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

try:
    from scripts.prepare_tgsim_input import Budget, HTTPStatusFailure, digest, fetch, save_new
    from scripts.prepare_tgsim_paged_input import ROOT, NoRedirect, disk_bytes, system_id_key
except ModuleNotFoundError:
    from prepare_tgsim_input import Budget, HTTPStatusFailure, digest, fetch, save_new
    from prepare_tgsim_paged_input import ROOT, NoRedirect, disk_bytes, system_id_key


def execute(plan_path, plan_sha):
    if digest(plan_path)!=plan_sha:raise ValueError('frozen diagnostic plan hash differs')
    plan=json.loads(plan_path.read_text());destination=Path(plan['output'])
    if destination.exists() or not destination.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('fresh worktree diagnostic output required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('freeze clean source before diagnostic observations')
    if (len(plan['requests'])!=3 or plan['maximum_requests']!=3 or plan['max_download_bytes']!=262144
            or plan['max_wall_clock_seconds']!=120 or plan['training_frames']!=0 or plan['evaluation_frames']!=0):
        raise ValueError('diagnostic-only caps differ')
    destination.mkdir(parents=True,exist_ok=False)
    budget=Budget(plan,[Path(p) for p in plan['disk_roots']])
    record=dict(status='running',plan_sha256=plan_sha,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                training_frames=0,evaluation_frames=0,hidden_openings=0,observations=[])
    save_new(destination/'execution_claim.json',dict(record,plan=plan))
    def deadline(signum,frame):raise TimeoutError('two-minute diagnostic deadline')
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,120)
    try:
        opener=urllib.request.build_opener(NoRedirect())
        for index,item in enumerate(plan['requests']):
            path=destination/f'observation_{index}.body';receipt=destination/f'observation_{index}_receipt.json'
            try:
                info=fetch(item['url'],path,budget,receipt,maximum_bytes=65536,opener=opener)
            except HTTPStatusFailure:
                info=json.loads(receipt.read_text())
                if index!=0 or info['status_code']!=400 or not info.get('diagnostic_entity_complete'):
                    raise
                body=json.loads(path.read_text())
                message=str(body.get('message',''))
                record['observations'].append(dict(index=index,status_code=400,error=body))
                if not any(word in message.lower() for word in ('parse','syntax','unexpected')):
                    raise ValueError('400 is not an identified syntax rejection; stop without query shopping')
                continue
            if index==0:
                raise ValueError('original failed query no longer rejects; stop without guessing')
            with path.open(encoding='utf-8-sig',newline='') as data:rows=list(csv.reader(data))
            if len(rows)!=2 or rows[0]!=['source_row_id']+item['expected_native_fields'] or len(rows[1])!=len(rows[0]):
                raise ValueError('one-row explicit-field capability schema differs')
            system_id_key(rows[1][0])
            record['observations'].append(dict(index=index,status_code=200,rows=1,header=rows[0],source_system_id=rows[1][0],sha256=info['sha256']))
        record['status']='syntax_diagnosed_and_two_explicit_native_probes_pass'
    except BaseException as error:
        record.update(status='stopped_preserved',error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        record.update(returned_body_bytes=budget.download_bytes,interrupted_read_bound=budget.interrupted_body_upper_bound,
                      elapsed_seconds=budget.clock()-budget.started,cumulative_disk_bytes=disk_bytes(plan['disk_roots']))
        save_new(destination/'completion.json',record)
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan_sha',required=True)
    args=parser.parse_args();execute(args.plan,args.plan_sha)
