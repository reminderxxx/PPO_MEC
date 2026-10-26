"""One frozen main-only, equal-frame preparation study on consumed train inputs.

Control, intent, intent/no-prediction and intent/no-optional-cache share relocated
parent tensors, fresh optimizers, actual frame budgets and update settings.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.probe_stream_main_opportunity import file_hash, select_train_cases, run_episode
from scripts.run_stream_main_ablation import train_unit
from scripts.train_joint_stream_control import make_env, actor_hash
from scripts.train_joint_tail_control import tensor_hash
from src.agents.sa_ghmappo_agent import IntentAwareStreamSAGHMAPPOAgent as Main

VARIANTS = dict(control=dict(cache_intent_enabled=False), intent=dict(),
    no_prediction=dict(prediction_enabled=False), no_optional_cache=dict(optional_cache_enabled=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent_runs',type=Path,nargs=3,required=True)
    p.add_argument('--diagnosis',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[7,17,27],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight',action='store_true')
    a = p.parse_args(); torch.set_num_threads(1)
    dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    if dirty and not a.preflight:
        raise ValueError('clean committed source required')
    parents = {}; source_path = None
    for run in a.parent_runs:
        plan = json.loads((run/'protocol.json').read_text())
        assert (run/'complete.json').exists()
        study = json.loads((Path(plan['study'])/'protocol.json').read_text())
        path = Path(study['parent_protocol'])
        assert source_path is None or path == source_path
        source_path = path
        cp = plan['checkpoints']['relocation']
        assert file_hash(cp['path']) == cp['sha256']
        parents[plan['seed']] = dict(**cp, source_run=str(run.resolve()),
            source_run_protocol_sha256=file_hash(run/'protocol.json'))
    assert set(parents) == {7,17,27}
    source = json.loads(source_path.read_text())
    assert file_hash(source_path.parent/'protocol.pt') == source['assets']['protocol.pt']
    for name, expected in source['source_fingerprints'].items():
        if name.startswith('src/') and name != 'src/agents/sa_ghmappo_agent.py':
            assert file_hash(ROOT/name) == expected, name
    diagnosis = json.loads(a.diagnosis.read_text())
    assert len(diagnosis['evidence']) == 72
    group = diagnosis['groups']['relocation']
    assert group['adapter/unannounced_target']['unused_bytes'] > group['adapter/own_target']['unused_bytes']
    data = torch.load(source_path.parent/'protocol.pt',weights_only=False)
    cases = select_train_cases(data)
    initial_cases = [next(c for c in cases if c['case_id']==cid) for cid in ['train_000','train_038','train_067']]
    budget = 128; epochs = 4; variants = VARIANTS
    if a.preflight:
        cases = cases[:1]; initial_cases = initial_cases[:1]
        budget = 8; epochs = 1; variants = {k:VARIANTS[k] for k in ['control','intent']}
    a.output.mkdir(parents=True,exist_ok=False)
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_dirty=dirty, phase='bounded_main_only_intent_warm_start', command=sys.argv,
        parent_protocol=str(source_path), parent_protocol_sha256=file_hash(source_path),
        diagnosis=str(a.diagnosis.resolve()), diagnosis_sha256=file_hash(a.diagnosis),
        parents=parents, seed=a.seed, variants=variants, preflight=a.preflight,
        training_cases=[c['case_id'] for c in cases], frames_per_training_case=budget,
        actual_training_frames_per_cell=budget*len(cases), update_group_size=4,
        update_calls_per_cell=(len(cases)+3)//4, epochs=epochs, parameter_count=73940,
        initial_evaluation_cases=[c['case_id'] for c in initial_cases],
        final_evaluation_cases=[c['case_id'] for c in cases],
        main_source_sha256=file_hash(ROOT/'src/agents/sa_ghmappo_agent.py'),
        training_unit_source_sha256=file_hash(ROOT/'scripts/run_stream_main_ablation.py'),
        environment_source_sha256=file_hash(ROOT/'src/envs/core/multi_request_stream_env.py'),
        baseline_source_sha256=file_hash(ROOT/'src/envs/core/joint_stream_rules.py'),
        evaluation='greedy complete physical episode, fixed final only, all outcomes retained',
        warm_start='same relocated parent per seed; fresh Adam and reset schedule for every variant',
        return_contract='unchanged native reward/10 plus critic bootstrap only at fixed-frame cuts, credit/8',
        internal_gate=dict(mean_reward_delta_min=0,mean_completed_delta_min=0,
            unused_optional_bytes_reduction_min=.20,per_seed_mean_completed_delta_min=-.25),
        limits=['Consumed training sessions, no formal/holdout',
            'Prediction deletion changes planner masks too, not predictor-accuracy-only',
            'Inherited full-model parents: local retrained intervention, not from-scratch module necessity',
            'Intent restricts optional policy actions; physical legality and comparator actions unchanged'],
        paper_claim_ready=False)
    (a.output/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    with (a.output/'cells.jsonl').open('x') as cells:
        for variant, options in variants.items():
            home = a.output/f'{variant}_seed{a.seed}'; home.mkdir()
            random.seed(a.seed); np.random.seed(a.seed)
            agent = Main.from_relocation_checkpoint(parents[a.seed]['path'],training_budget=len(cases),**options)
            assert sum(p.numel() for p in agent.parameters()) == plan['parameter_count']
            assert not agent.optimizer.state and not agent.surrogate_optimizer.state
            meta = dict(variant=variant,seed=a.seed,actual_frames=0,updates=[],status='running',
                initial_tensor_sha256=tensor_hash(agent),initial_actor_sha256=actor_hash(agent),
                parameter_count=plan['parameter_count'],checkpoints=[])
            agent.save_checkpoint(home/'initial.pt')
            meta['checkpoints'].append(dict(phase='initial',path='initial.pt',sha256=file_hash(home/'initial.pt')))
            with (home/'evaluation.jsonl').open('x') as ev:
                for c in initial_cases:
                    row = run_episode(make_env(c,data,source),agent,home/f'initial_{c["case_id"]}.jsonl',learned=True)
                    ev.write(json.dumps(dict(phase='initial',case_id=c['case_id'],**row))+'\n'); ev.flush()
                torch.manual_seed(a.seed+20000); buffer=[]
                with (home/'training.jsonl').open('x') as log, (home/'training_trace.jsonl').open('x') as trace:
                    for i,c in enumerate(cases):
                        started=time.monotonic()
                        row, rolls = train_unit(c,data,source,agent,budget,trace,
                            dict(seed=a.seed,variant=variant,case_id=c['case_id']))
                        buffer.extend(rolls); meta['actual_frames'] += row['actual_frames']; update=None
                        if (i+1)%4==0 or i==len(cases)-1:
                            update=agent.update(buffer,epochs=epochs); buffer=[]; meta['updates'].append(update)
                        log.write(json.dumps(dict(case_id=c['case_id'],**row,update=update))+'\n'); log.flush(); trace.flush()
                        (home/'progress.json').write_text(json.dumps(dict(case_index=i+1,**meta),indent=2)+'\n')
                        print(json.dumps(dict(seed=a.seed,variant=variant,phase='train',case=c['case_id'],
                            frames=meta['actual_frames'],seconds=time.monotonic()-started)),flush=True)
                assert meta['actual_frames'] == plan['actual_training_frames_per_cell']
                assert len(meta['updates']) == plan['update_calls_per_cell']
                agent.save_checkpoint(home/'final.pt')
                meta['checkpoints'].append(dict(phase='final',path='final.pt',sha256=file_hash(home/'final.pt')))
                evaluated=Main.from_checkpoint(home/'final.pt')
                assert tensor_hash(agent)==tensor_hash(evaluated)
                for c in cases:
                    row=run_episode(make_env(c,data,source),evaluated,home/f'final_{c["case_id"]}.jsonl',learned=True)
                    ev.write(json.dumps(dict(phase='final',case_id=c['case_id'],**row))+'\n'); ev.flush()
            meta.update(status='complete',final_tensor_sha256=tensor_hash(agent),final_actor_sha256=actor_hash(agent),
                output_sha256={f:file_hash(home/f) for f in ['training.jsonl','training_trace.jsonl','evaluation.jsonl']})
            (home/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
            cells.write(json.dumps(dict(path=home.name,**meta))+'\n'); cells.flush()
            print(json.dumps(dict(seed=a.seed,variant=variant,phase='cell_complete')),flush=True)
    (a.output/'complete.json').write_text(json.dumps(dict(status='complete',cells=len(variants),
        cells_sha256=file_hash(a.output/'cells.jsonl')),indent=2)+'\n')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        if '--output' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--output')+1])
            if output.is_dir():
                (output/'failure.json').write_text(json.dumps(dict(
                    failed_at=datetime.now(timezone.utc).isoformat(),traceback=traceback.format_exc()),indent=2)+'\n')
        raise
