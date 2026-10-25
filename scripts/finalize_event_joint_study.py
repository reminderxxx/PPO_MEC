"""Prospectively frozen confirmation, paired analysis and figures; no training."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.run_persistent_workflow_learning import dump,sha
from src.agents.ppo_agent import EventJointPPOAgent

CONSUMERS=['scripts/evaluate_event_joint.py','scripts/analyze_event_joint_study.py','scripts/plot_event_joint_study.py','scripts/finalize_event_joint_study.py']
TRAINING=['src/agents/ppo_agent.py','src/encoders/event_joint_encoder.py','src/envs/wrappers/joint_workflow_control.py','src/envs/wrappers/ready_workflow_control.py','scripts/run_event_joint_learning.py','scripts/run_ready_workflow_learning.py','scripts/run_joint_workflow_probe.py']


def timestamp():return datetime.now(timezone.utc).isoformat()


def waiting_cells(root,plan):
    return [(m,s) for m in plan['methods'] for s in plan['seeds'] if not (root/'study_v1'/m/f'seed_{s}/completion.json').exists()]


def parameter_hash(state):
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):digest.update(key.encode());digest.update(str(tuple(value.shape)).encode());digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def freeze_candidates(root,plan_path):
    torch.set_num_threads(1);plan=json.loads(plan_path.read_text())
    if waiting_cells(root,plan):raise ValueError('Cannot freeze incomplete candidate cells')
    original=json.loads((root/'study_v1/plan_freeze.json').read_text())
    if original['plan_sha256']!=sha(plan_path):raise ValueError('Training plan drift')
    science_git=original['source_git'];source_hashes={}
    for name in TRAINING:
        expected=hashlib.sha256(subprocess.check_output(['git','show',f'{science_git}:{name}'],cwd=ROOT)).hexdigest()
        if sha(ROOT/name)!=expected:raise ValueError('Frozen training source drift')
        source_hashes[name]=expected
    records=[];initial={}
    for method,spec in plan['methods'].items():
        for seed in plan['seeds']:
            cell=root/'study_v1'/method/f'seed_{seed}';receipt=json.loads((cell/'completion.json').read_text());rows=json.loads((cell/'training.json').read_text())
            if receipt['status']!='complete' or receipt['episodes']!=256 or receipt['updates']!=64 or len(rows)!=256:raise ValueError('Incomplete candidate budget')
            if sum(x['steps'] for x in rows)!=receipt['native_steps'] or receipt['native_steps']>256*95:raise ValueError('Interaction count drift')
            if any(r['window']!=i%16 for i,r in enumerate(rows)):raise ValueError('Input exposure order differs')
            checkpoint=cell/'checkpoint_256.pt';agent,payload=EventJointPPOAgent.load(checkpoint)
            if agent.seed!=seed or agent.controller!=spec['controller'] or agent.use_graph!=spec['graph'] or not agent.eviction_aware:raise ValueError('Candidate contract mismatch')
            if payload['provenance']['source_git']!=science_git:raise ValueError('Training source provenance drift')
            if payload['provenance']['input_sha256']!=original['train_dev_inputs_sha256']:raise ValueError('Training input provenance drift')
            initial_payload=torch.load(cell/'checkpoint_0.pt',map_location='cpu');h=parameter_hash(initial_payload['network']);initial.setdefault(str(seed),set()).add(h)
            records.append(dict(method=method,seed=seed,path=str(checkpoint),sha256=sha(checkpoint),native_steps=receipt['native_steps']))
    if any(len(values)!=1 for values in initial.values()):raise ValueError('Unmatched initial network parameters')
    packages=[root/'fresh_inputs_v4/confirmation_inputs.json',root/'supporting_workload_matrix_v1/inputs.json']
    freeze=dict(frozen_at=timestamp(),training_source_git=science_git,consumer_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes=source_hashes,consumer_hashes={n:sha(ROOT/n) for n in CONSUMERS},plan_sha256=sha(plan_path),input_sha256s=[sha(p) for p in packages],
        checkpoint_sha256=[x['sha256'] for x in records],checkpoints=records,rules=plan['causal_rules'],primary_method=plan['primary_method'],checkpoint_episode=256,
        matched_initial_parameters={k:list(v)[0] for k,v in initial.items()},input_diversity_review_sha256=sha(root/'input_diversity_readonly_review.json'),
        prospective_support_amendment_sha256=sha(root/'supporting_workload_matrix_v1/prospective_amendment.json'),
        claim_boundary='Controlled joint scheduling/cache experiment; no original SA-GHMAPPO, handoff, real AI inference, canonical or formal superiority claim',confirmation_outcomes_previously_opened=False)
    target=root/'confirmation_freeze.json'
    if target.exists():raise FileExistsError(target)
    dump(target,freeze);return freeze


def child(command,logs,label,env):
    with (logs/f'{label}.stdout.log').open('x') as stdout,(logs/f'{label}.stderr.log').open('x') as stderr:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
    return dict(label=label,argv=command,returncode=result.returncode)


def finalize(args):
    root=Path(args.root);plan_path=Path(args.plan);plan=json.loads(plan_path.read_text());state=root/'finalization_state.json'
    while waiting_cells(root,plan):
        dump(state,dict(phase='waiting_for_fixed_training',at=timestamp(),remaining=waiting_cells(root,plan),confirmation_opened=False))
        if not args.wait:raise ValueError('Fixed training is incomplete')
        time.sleep(15)
    if (root/'confirmation_v1').exists() or (root/'supporting_confirmation_v1').exists():raise FileExistsError('Confirmation output already exists; no automatic resume or rerun')
    freeze=freeze_candidates(root,plan_path);freeze_path=root/'confirmation_freeze.json';logs=root/'finalization_command_logs';logs.mkdir()
    dump(root/'confirmation_open_record.json',dict(opened_at=timestamp(),freeze_sha256=sha(freeze_path),source='local fixed-candidate executor',purpose='Predeclared primary and complete supporting matrix; no retuning after opening',formal_sealed_holdout_opened=False))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');results=[]
    packages=[('confirmation_v1',root/'fresh_inputs_v4/confirmation_inputs.json'),('supporting_confirmation_v1',root/'supporting_workload_matrix_v1/inputs.json')]
    for cohort,inputs in packages:
        out=root/cohort;out.mkdir();jobs=[]
        for candidate in freeze['checkpoints']:
            label=f'{candidate["method"]}_s{candidate["seed"]}'
            command=[sys.executable,'-B',str(ROOT/'scripts/evaluate_event_joint.py'),'--inputs',str(inputs),'--output',str(out/label),'--split','confirmation','--checkpoint',candidate['path'],'--freeze',str(freeze_path)]
            jobs.append((cohort+'_'+label,command))
        for rule in plan['causal_rules']:
            command=[sys.executable,'-B',str(ROOT/'scripts/evaluate_event_joint.py'),'--inputs',str(inputs),'--output',str(out/rule),'--split','confirmation','--rule',rule,'--freeze',str(freeze_path)]
            jobs.append((cohort+'_'+rule,command))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending={pool.submit(child,command,logs,label,env):label for label,command in jobs}
            for future in as_completed(pending):
                result=future.result();results.append(result);dump(state,dict(phase=cohort,at=timestamp(),completed_jobs=len(results),results=results,confirmation_opened=True));print(json.dumps(dict(label=result['label'],returncode=result['returncode'])),flush=True)
        if any(x['returncode'] for x in results):raise RuntimeError('Evaluation failed; raw output retained, no automatic retry')
    for name,expected in freeze['consumer_hashes'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen consumer changed during confirmation')
    if sha(root/'input_diversity_readonly_review.json')!=freeze['input_diversity_review_sha256']:raise ValueError('Input-equivalence review drift')
    if sha(root/'supporting_workload_matrix_v1/prospective_amendment.json')!=freeze['prospective_support_amendment_sha256']:raise ValueError('Prospective supporting protocol drift')
    command=[sys.executable,'-B',str(ROOT/'scripts/analyze_event_joint_study.py'),'--root',str(root),'--output',str(root/'report_v1'),'--plan',str(plan_path)]
    result=child(command,logs,'analysis',env);results.append(result)
    if result['returncode']:raise RuntimeError('Analysis failed; no automatic reinterpretation')
    plot_env=dict(env,PYTHONPATH=args.plot_dependencies,MPLCONFIGDIR='/private/tmp/ppo_mec_matplotlib')
    command=[sys.executable,'-B',str(ROOT/'scripts/plot_event_joint_study.py'),'--analysis',str(root/'report_v1'),'--output',str(root/'figures_v1')]
    result=child(command,logs,'figures',plot_env);results.append(result)
    if result['returncode']:raise RuntimeError('Figure rendering failed')
    dump(root/'finalization_completion.json',dict(status='complete',completed_at=timestamp(),freeze_sha256=sha(freeze_path),jobs=results,
        analysis_sha256=sha(root/'report_v1/analysis.json'),figures_sha256=sha(root/'figures_v1/figure_manifest.json'),paper_ready=False,requires_independent_final_audit=True))
    dump(state,dict(phase='complete_pending_readonly_audit',at=timestamp(),confirmation_opened=True));print('FIXED CONFIRMATION / ANALYSIS / FIGURES COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--plan',default=str(ROOT/'configs/experiment/event_joint_learning_v1.json'));p.add_argument('--workers',type=int,default=8);p.add_argument('--wait',action='store_true');p.add_argument('--plot-dependencies',default='/private/tmp/ppo_mec_reward_plot_deps');a=p.parse_args()
    try:finalize(a)
    except Exception as error:
        dump(Path(a.root)/'finalization_failure.json',dict(at=timestamp(),error_type=type(error).__name__,message=str(error)));raise
