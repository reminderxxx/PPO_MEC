"""Freeze, qualify and coordinate one bounded paired SA learning experiment.

Authoring/importing this script does not execute environments or training.
qualification-plan -> qualify -> prepare -> run is an exclusive, non-resumable
sequence. status only reads compact receipts. A failed run is never retried.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARMS = {'sa_legacy_clip': False, 'sa_group_clip': True}
SEEDS = [7, 17, 27]
POINTS = list(range(0, 65, 8))
TRAINER = 'scripts/train_sa_group_clip_pair.py'
AUDITOR = 'scripts/audit_sa_group_clip_pair.py'
DEFAULT_RUN = ROOT / 'artifacts/joint_stream_learning/20261016T162100Z_confirmation'
DEFAULT_GATES = ROOT / 'artifacts/joint_stream_learning/next_stage_preservation_plan_20261017.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value, exclusive=True):
    path = Path(path)
    if exclusive:
        with path.open('x', encoding='utf8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
    else:
        temp = path.with_name(path.name + '.tmp')
        with temp.open('w', encoding='utf8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        temp.replace(path)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def clean_head(expected=None):
    require(not git('status', '--porcelain', '--untracked-files=normal'), 'Clean Git checkout required')
    head = git('rev-parse', 'HEAD')
    require(expected is None or head == expected, 'Frozen HEAD changed')
    return head


def remote_state():
    try:
        upstream = git('rev-parse', '@{upstream}')
        return dict(local_commit=git('rev-parse', 'HEAD'), upstream_commit=upstream,
                    status='locally_matches_tracking_ref' if upstream == git('rev-parse','HEAD') else 'pending_or_diverged',
                    note='Local tracking ref only; no network/push was attempted by coordinator')
    except subprocess.CalledProcessError:
        return dict(local_commit=git('rev-parse','HEAD'), status='pending_no_local_upstream',
                    note='Local freeze permitted; remote synchronization must be recorded separately')


def assert_files(fingerprints):
    for name, expected in fingerprints.items():
        require(sha(ROOT / name) == expected, 'Frozen source changed: ' + name)


def protocol_info(source):
    p = source / 'protocol/manifest.json'
    protocol = read(p)
    require(protocol['episodes'] == 64 and protocol['seeds'] == SEEDS, 'Wrong original budget/seeds')
    require(protocol['action_contract'] == 'joint_stream_control_v2', 'Wrong original action contract')
    require(len(protocol['source_fingerprints']) == 123, 'Expected123 historical source paths')
    assets = {name: dict(path=str(source/'protocol'/name), sha256=value)
              for name,value in protocol['assets'].items()}
    for value in assets.values():
        require(sha(value['path']) == value['sha256'], 'Original protocol asset changed')
    manpath = source / 'sa_ghmappo/manifest.json'
    manifest = read(manpath)
    require(manifest['status'] == 'complete' and not manifest['git_dirty'], 'Original SA run not complete/clean')
    require(manifest['protocol_sha256'] == sha(p) and manifest['source_commit'] == protocol['source_commit'], 'Original SA/protocol mismatch')
    return protocol, manifest, dict(source_run=str(source), source_protocol=str(p),
        source_protocol_sha256=sha(p), source_assets=assets,
        protocol_data=assets['protocol.pt'], original_sa_manifest=dict(path=str(manpath),sha256=sha(manpath)))


def qualification_plan(args):
    out = args.output_root.resolve()
    source = args.source_run.resolve()
    require(not out.exists(), 'Use a new stage4 root; no overwrite')
    require(out != source and not out.is_relative_to(source), 'Cannot write within original scientific run')
    head = clean_head()
    protocol, manifest, bound = protocol_info(source)
    for name, expected in protocol['source_fingerprints'].items():
        if name != 'src/agents/sa_ghmappo_agent.py':
            require(sha(ROOT / name) == expected, 'Unapproved historical source change: ' + name)
    # Bind current candidate source, NOT an invented35975 live identity.
    names = set(protocol['source_fingerprints']) | {str(Path(__file__).resolve().relative_to(ROOT)), TRAINER, AUDITOR,
        'scripts/joint_stream_same_state_scoring.py', 'scripts/diagnose_joint_stream_learning.py'} | set(args.additional_source)
    tracked = set(git('ls-files').splitlines())
    require(names <= tracked, 'Every frozen source must be tracked in the clean commit')
    fingerprints = {name:sha(ROOT/name) for name in sorted(names)}
    checkpoints = [x for x in manifest['checkpoints'] if x['seed'] == 7 and x['episode'] == 64]
    require(len(checkpoints) == 1, 'Expected one seed7/final64 checkpoint')
    checkpoint = checkpoints[0]
    path = source/'sa_ghmappo'/checkpoint['path']
    require(sha(path) == checkpoint['sha256'], 'Qualification checkpoint changed')
    out.mkdir(parents=True, exist_ok=False)
    c = dict(schema='sa_pair_qualification_v1', status='frozen_before_qualification', frozen_at=now(),
        repo_root=str(ROOT), output_root=str(out), source_commit=head, source_fingerprints=fingerprints,
        **bound, qualification=dict(max_sessions=2,max_frames=960,per_session_max_frames=480,
            seed=7,episode=64,case_id='dev_000',arms=ARMS,checkpoint=dict(path=str(path),sha256=sha(path))),
        no_training=True, no_optimizer_steps=True, remote_sync=remote_state())
    write(out/'qualification_contract.json',c)
    write(out/'qualification_plan_command.json',dict(argv=sys.argv,executable=sys.executable,cwd=os.getcwd()))
    print(json.dumps(dict(status='qualification_budget_frozen',root=str(out),contract_sha256=sha(out/'qualification_contract.json'))))


def validate_qualification_contract(out):
    c=read(out/'qualification_contract.json')
    require(c['schema']=='sa_pair_qualification_v1' and c['output_root']==str(out), 'Wrong qualification contract')
    clean_head(c['source_commit']); assert_files(c['source_fingerprints'])
    require(sha(c['source_protocol'])==c['source_protocol_sha256'],'Original protocol changed')
    require(sha(c['original_sa_manifest']['path'])==c['original_sa_manifest']['sha256'],'Original SA manifest changed')
    for item in c['source_assets'].values():require(sha(item['path'])==item['sha256'],'Original asset changed')
    q=c['qualification']
    require(q['max_sessions']==2 and q['max_frames']==960 and q['per_session_max_frames']==480,'Qualification budget expansion')
    require(q['seed']==7 and q['episode']==64 and q['case_id']=='dev_000' and q['arms']==ARMS,'Qualification witness changed')
    require(sha(q['checkpoint']['path'])==q['checkpoint']['sha256'],'Qualification checkpoint changed')
    return c


def canonical(row):
    return json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\n'


def qualification_original_traces(source, manifest):
    """One read per original SA raw file; verify whole SHA while selecting fixed case."""
    output={}
    for name in ('events.jsonl','receipts.jsonl'):
        whole=hashlib.sha256();selected=hashlib.sha256();count=0
        with (source/'sa_ghmappo'/name).open('rb') as f:
            for line in f:
                whole.update(line)
                # Producer writes spaced JSON. Parse only possible fixed-case rows.
                if b'"case_id": "dev_000"' not in line or b'"seed": 7,' not in line:continue
                row=json.loads(line)
                if row.get('agent')=='sa_ghmappo' and row.get('seed')==7 and row.get('episode')==64 and row.get('case_id')=='dev_000':
                    selected.update(canonical(row));count+=1
        require(whole.hexdigest()==manifest['output_sha256'][name],'Original raw SHA mismatch: '+name)
        require(count>0,'Original fixed qualification trace missing')
        output[name]=dict(canonical_sha256=selected.hexdigest(),rows=count)
    return output


def compressed_trace(path):
    h=hashlib.sha256();count=0
    with gzip.open(path,'rt') as f:
        for line in f:h.update(canonical(json.loads(line)));count+=1
    return dict(canonical_sha256=h.hexdigest(),rows=count)


def qualify(args):
    out=args.output_root.resolve()
    write(out/'qualification_started.json',dict(started_at=now(),pid=os.getpid(),policy='exclusive; no retry'))
    frames=0;completed=[];env=None
    try:
        c=validate_qualification_contract(out);source=Path(c['source_run']);protocol=read(c['source_protocol'])
        manifest=read(c['original_sa_manifest']['path'])
        path=source/'sa_ghmappo/dev.jsonl'
        require(sha(path)==manifest['output_sha256']['dev.jsonl'],'Original dev changed')
        original=[r for r in map(json.loads,path.read_text().splitlines()) if r['seed']==7 and r['episode']==64 and r['case_id']=='dev_000']
        require(len(original)==1,'Qualification original row missing')
        print(json.dumps(dict(phase='verify_original_qualification_traces')),flush=True)
        reference=qualification_original_traces(source,manifest)
        import numpy as np
        import torch
        from scripts.train_joint_stream_control import make_env, stream_episode
        from scripts import joint_stream_same_state_scoring as scoring
        from scripts.diagnose_joint_stream_learning import no_training
        from src.agents.sa_ghmappo_agent import StreamSAGHMAPPOAgent
        torch.set_num_threads(1)
        data=torch.load(c['protocol_data']['path'],map_location='cpu',weights_only=False)
        case=next(x for x in data['cases']['dev'] if x['case_id']=='dev_000')
        require(len(data['sessions'][case['session']]['frames'])<=480,'Qualification case frame cap')
        with scoring.preserved_rng():
            legacy=StreamSAGHMAPPOAgent.from_checkpoint(c['qualification']['checkpoint']['path'])
            options=deepcopy(legacy.options);options.pop('separate_policy_value_clip',None)
            candidate=StreamSAGHMAPPOAgent(separate_policy_value_clip=True,**options)
            candidate.load_state_dict(deepcopy(legacy.state_dict()))
        models={'sa_legacy_clip':legacy,'sa_group_clip':candidate}
        require(scoring.module_digest(legacy)==scoring.module_digest(candidate),'Qualification input weights differ')
        directory=out/'qualification';directory.mkdir(exist_ok=False)
        identity=dict(agent='sa_ghmappo',rule='sa_ghmappo_seed7',seed=7,episode=64,case_id='dev_000')
        for arm,model in models.items():
            group=directory/arm;group.mkdir()
            count={'frames':0};before=scoring.module_digest(model)
            random.seed(7);np.random.seed(7);torch.manual_seed(7)
            with no_training(models):
                env=make_env(case,data,protocol);original_step=env.step_controls
                def step(this,controls):
                    nonlocal frames
                    require(count['frames']<480 and frames<960,'Qualification budget exceeded before step')
                    count['frames']+=1;frames+=1
                    return original_step(controls)
                with scoring.patched_method(env,'step_controls',step):
                    with gzip.open(group/'events.jsonl.gz','xt') as ef,gzip.open(group/'receipts.jsonl.gz','xt') as rf:
                        row,_=stream_episode(env,model,deterministic=True,event_file=ef,receipt_file=rf,identity=identity)
            actual=dict(identity,**row)
            require(actual==original[0],'Qualification full original summary mismatch')
            require(row['steps']==count['frames'],'Qualification frame accounting mismatch')
            traces={name:compressed_trace(group/(name+'.gz')) for name in reference}
            require(traces==reference,'Qualification original full events/receipts mismatch')
            require(scoring.module_digest(model)==before,'Qualification model weights changed')
            write(group/'row.json',actual)
            completed.append(dict(arm=arm,frames=count['frames'],summary_exact=True,traces_exact=True,
                trace_digests=traces,weights_sha256=before,source_checkpoint_sha256=c['qualification']['checkpoint']['sha256'],
                outputs={name:sha(group/name) for name in ('row.json','events.jsonl.gz','receipts.jsonl.gz')}))
        clean_head(c['source_commit']);assert_files(c['source_fingerprints'])
        write(out/'qualification_result.json',dict(status='pass',completed_at=now(),
            contract_sha256=sha(out/'qualification_contract.json'),source_commit=c['source_commit'],
            sessions=2,frames=frames,results=completed,training_sessions=0,update_calls=0,optimizer_steps=0))
        print(json.dumps(dict(status='qualification_pass',sessions=2,frames=frames)))
    except BaseException as exc:
        write(out/'qualification_failure.json',dict(status='failed',failed_at=now(),error=repr(exc),
            traceback=traceback.format_exc(),completed_sessions=len(completed),charged_frames=frames,
            current_env_frame=getattr(env,'frame_index',None),policy='no rerun; preserve all evidence'))
        raise


def prepare(args):
    out=args.output_root.resolve();c=validate_qualification_contract(out)
    require(not (out/'contract.json').exists(),'Main contract already frozen')
    receipt=read(out/'qualification_result.json')
    require(receipt['status']=='pass' and receipt['contract_sha256']==sha(out/'qualification_contract.json') and receipt['source_commit']==c['source_commit'],'Passing current qualification receipt required')
    require(receipt['sessions']==2 and receipt['frames']<=960 and receipt['optimizer_steps']==0,'Qualification receipt budget mismatch')
    for result in receipt['results']:
        for name,value in result['outputs'].items():require(sha(out/'qualification'/result['arm']/name)==value,'Qualification output changed')
    source=Path(c['source_run']);protocol=read(c['source_protocol']);manifest=read(c['original_sa_manifest']['path'])
    import numpy as np
    import torch
    from scripts.train_joint_stream_control import actor_hash
    from scripts.train_joint_tail_control import tensor_hash
    from scripts.joint_stream_same_state_scoring import preserved_rng, rng_digest
    from src.agents.registry import build_agent
    torch.set_num_threads(1)
    data=torch.load(c['protocol_data']['path'],map_location='cpu',weights_only=False)
    panel=[x['case_id'] for x in data['cases']['dev']]
    require(len(panel)==48 and len(set(panel))==48,'Expected complete fixed48 development panel')
    train_ids={x['case_id'] for x in data['cases']['train']}
    schedule=protocol['training_schedule']
    require(set(schedule)==set(map(str,SEEDS)) and all(len(schedule[str(s)])==64 and set(schedule[str(s)])<=train_ids for s in SEEDS),'Invalid original64-session schedules')
    expected={};initial=[]
    for seed in SEEDS:
        old=[x for x in manifest['checkpoints'] if x['seed']==seed]
        require(len(old)==2 and len({(x['initial_tensor_sha256'],x['initial_actor_sha256']) for x in old})==1,'Historical initial identity conflict')
        expected[str(seed)]=dict(tensor_sha256=old[0]['initial_tensor_sha256'],actor_sha256=old[0]['initial_actor_sha256'])
        for arm,flag in ARMS.items():
            with preserved_rng():
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                before=rng_digest()
                model=build_agent('sa_ghmappo',control_contract='joint_stream_control_v2',seed=seed,
                    training_budget=64,prediction_enabled=True,return_gamma=1.,separate_policy_value_clip=flag)
                actual=dict(tensor_sha256=tensor_hash(model),actor_sha256=actor_hash(model))
                require(actual==expected[str(seed)],'Fresh initial model does not match historical identity')
                require(int(model.completed_updates)==0,'Initial model update count nonzero')
                initial.append(dict(arm=arm,seed=seed,**actual,rng_before_sha256=before,
                    rng_after_sha256=rng_digest(),options=deepcopy(model.options),completed_updates=0))
        pair=initial[-2:]
        require(pair[0]['rng_before_sha256']==pair[1]['rng_before_sha256'] and pair[0]['rng_after_sha256']==pair[1]['rng_after_sha256'],'Paired initialization RNG history differs')
    frozen=out/'frozen';frozen.mkdir(exist_ok=False)
    gate_source=args.preservation_plan.resolve();gates=read(gate_source)
    gate_source_sha256=sha(gate_source)
    require(gates['promotion_gates']['floating_tolerance_only']==1e-8,'Preservation tolerance changed')
    gatepath=frozen/'preservation_plan.json'
    write(gatepath,{name:deepcopy(gates[name]) for name in ('promotion_gates','baseline_metrics','baseline_metrics_sha256')})
    (frozen/'source_protocol.json').write_bytes(Path(c['source_protocol']).read_bytes())
    head=clean_head(c['source_commit']);assert_files(c['source_fingerprints'])
    contract=dict(schema='sa_group_clip_pair_v1',status='frozen',frozen_at=now(),repo_root=str(ROOT),output_root=str(out),
        source_commit=head,source_fingerprints=c['source_fingerprints'],
        **{name:c[name] for name in ('source_run','source_protocol','source_protocol_sha256','source_assets','protocol_data','original_sa_manifest')},
        arms=ARMS,arm_output_roots={arm:str(out/'arms'/arm) for arm in ARMS},seeds=SEEDS,episodes=64,
        checkpoint_episodes=POINTS,primary_checkpoint=64,eval_panel_ids=panel,training_schedule=schedule,
        expected_initial=expected,initial_current=initial,
        preservation_plan=dict(path=str(gatepath),sha256=sha(gatepath)),preservation_source_at_freeze=dict(path=str(gate_source),sha256=gate_source_sha256),
        qualification=dict(max_sessions=2,max_frames=960,receipt=str(out/'qualification_result.json'),receipt_sha256=sha(out/'qualification_result.json')),
        tail_stability=dict(episodes=[48,56,64],reward_relative_range=.01,completion_percentage_point_range=1.,model_cost_relative_range=.02),
        numeric_tolerance=1e-8,max_training_sessions=384,max_training_frames=184320,
        max_evaluation_sessions=2592,max_evaluation_frames=1244160,
        per_arm_max_training_frames=92160,per_arm_max_evaluation_frames=622080,
        per_arm_training_sessions=192,per_arm_evaluation_sessions=1296,max_sessions_per_seed=64,
        max_frames_per_session=480,max_parallel_arms=2,torch_threads_per_arm=1,
        original_predictor_sha256=protocol['predictor_sha256'],predictor_asset=c['source_assets']['predictor.pt'],
        scientific_budget='one newly authorized paired run only; qualification is separately bounded; no retry/extension',
        selection='fixed64 only; all snapshots and all48conditions retained; no peak/seed selection',
        formal_holdout='sealed; not part of this experiment',canonical_changed=False,
        remote_sync=remote_state(),trainer=TRAINER,auditor=AUDITOR)
    write(out/'contract.json',contract)
    write(out/'prepare_command.json',dict(argv=sys.argv,executable=sys.executable,cwd=os.getcwd()))
    write(out/'freeze_receipt.json',dict(frozen_at=now(),contract_sha256=sha(out/'contract.json'),source_commit=head,remote_sync=contract['remote_sync']))
    print(json.dumps(dict(status='prepared',root=str(out),source_commit=head,contract_sha256=sha(out/'contract.json'))))


def validate_run(out):
    c=read(out/'contract.json');require(c['schema']=='sa_group_clip_pair_v1' and c['status']=='frozen','Frozen paired contract required')
    require(c['output_root']==str(out) and c['repo_root']==str(ROOT),'Frozen root mismatch')
    require(c['arms']==ARMS and c['seeds']==SEEDS and c['episodes']==64 and c['checkpoint_episodes']==POINTS,'Fixed paired experiment identity changed')
    limits=dict(max_training_sessions=384,max_training_frames=184320,max_evaluation_sessions=2592,
        max_evaluation_frames=1244160,per_arm_max_training_frames=92160,per_arm_max_evaluation_frames=622080,
        max_parallel_arms=2,torch_threads_per_arm=1,max_frames_per_session=480)
    require(all(c[name]==value for name,value in limits.items()),'Scientific budget changed')
    require(sha(out/'contract.json')==read(out/'freeze_receipt.json')['contract_sha256'],'Frozen main contract changed')
    clean_head(c['source_commit']);assert_files(c['source_fingerprints'])
    require(sha(c['source_protocol'])==c['source_protocol_sha256'],'Original protocol changed')
    for entry in c['source_assets'].values():require(sha(entry['path'])==entry['sha256'],'Original asset changed')
    for name in ('preservation_plan','original_sa_manifest'):
        entry=c[name];require(sha(entry['path'])==entry['sha256'],name+' changed')
    q=c['qualification'];require(sha(q['receipt'])==q['receipt_sha256'] and read(q['receipt'])['status']=='pass','Qualification receipt invalid')
    require(c['trainer']==TRAINER and c['auditor']==AUDITOR,'Frozen executables changed')
    return c


def stop_children(children):
    for arm,p in children.items():
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    deadline=time.monotonic()+5
    for arm,p in children.items():
        if p.poll() is None:
            try:p.wait(timeout=max(.01,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait()


def compact_status(out):
    c=read(out/'contract.json') if (out/'contract.json').exists() else {}
    arms={}
    for arm,path in c.get('arm_output_roots',{}).items():
        p=Path(path)/'manifest.json'
        if not p.exists():arms[arm]=dict(status='not_started');continue
        try:
            m=read(p)
            arms[arm]={k:m.get(k) for k in ('status','training_sessions','training_episodes','training_frames','evaluation_sessions','eval_cells','evaluation_frames','current_seed','current_episode') if k in m}
        except (OSError,json.JSONDecodeError):arms[arm]=dict(status='manifest_write_in_progress')
    return dict(observed_at=now(),root=str(out),coordinator=read(out/'status.json') if (out/'status.json').exists() else None,
                qualification=read(out/'qualification_result.json').get('status') if (out/'qualification_result.json').exists() else 'not_complete',arms=arms)


def run(args):
    out=args.output_root.resolve()
    # Contract-global lock is in the same frozen root, independent of arm output paths.
    write(out/'run_started.json',dict(started_at=now(),pid=os.getpid(),policy='one execution; no resume/retry'))
    children={};logs={};phase='verification'
    previous_term_handler=signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise RuntimeError('Coordinator terminated by signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        c=validate_run(out)
        (out/'arms').mkdir(exist_ok=False)
        (out/'logs').mkdir(exist_ok=False)
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                 NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
        commands={arm:[sys.executable,str(ROOT/TRAINER),'--contract',str(out/'contract.json'),
                       '--arm',arm,'--output_root',c['arm_output_roots'][arm]] for arm in ARMS}
        audit_command=[sys.executable,str(ROOT/AUDITOR),'--root',str(out)]
        write(out/'run_command.json',dict(argv=sys.argv,commands=commands,auditor=audit_command,
            source_commit=c['source_commit'],contract_sha256=sha(out/'contract.json'),threads=1))
        phase='training_and_evaluation'
        for arm,command in commands.items():
            logs[arm]=(out/'logs'/(arm+'.log')).open('x')
            children[arm]=subprocess.Popen(command,cwd=ROOT,env=env,stdout=logs[arm],stderr=subprocess.STDOUT,start_new_session=True)
        last_progress=0.;last_notice=0.
        while True:
            codes={arm:p.poll() for arm,p in children.items()}
            failed={arm:code for arm,code in codes.items() if code is not None and code!=0}
            if failed:raise RuntimeError('Paired arm failed: '+repr(failed))
            if all(code is not None for code in codes.values()):break
            if time.monotonic()-last_progress>=60:
                clean_head(c['source_commit']);assert_files(c['source_fingerprints'])
                write(out/'status.json',dict(status='running',phase=phase,updated_at=now(),pids={arm:p.pid for arm,p in children.items()},exit_codes=codes),exclusive=False)
                # Compact, infrequent status; no reward-driven action or early selection.
                if time.monotonic()-last_notice>=1200:
                    print(json.dumps(compact_status(out)),flush=True)
                    last_notice=time.monotonic()
                last_progress=time.monotonic()
            time.sleep(1)
        phase='final_source_verification';validate_run(out)
        phase='audit'
        with (out/'logs/audit.log').open('x') as f:
            auditor=subprocess.Popen(audit_command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            children['auditor']=auditor
            code=auditor.wait()
            require(code==0,'Paired auditor failed; see audit.log')
        require((out/'audit/status.json').is_file() and (out/'audit/summary.json').is_file(),'Auditor receipts missing')
        require(read(out/'audit/status.json').get('status')=='pass','Auditor did not explicitly pass integrity/completeness checks')
        write(out/'result.json',dict(status='complete',completed_at=now(),source_commit=c['source_commit'],
            contract_sha256=sha(out/'contract.json'),audit_status=str(out/'audit/status.json'),
            audit_status_sha256=sha(out/'audit/status.json'),audit_summary_sha256=sha(out/'audit/summary.json'),
            remote_sync=remote_state(),selection='no automatic model replacement; auditor reports scientific gates'))
        write(out/'status.json',dict(status='complete',completed_at=now(),phase='complete'),exclusive=False)
    except BaseException as exc:
        stop_children(children)
        failure=dict(status='failed',failed_at=now(),phase=phase,error=repr(exc),traceback=traceback.format_exc(),
            exit_codes={arm:p.poll() for arm,p in children.items()},policy='preserve all artifacts; do not resume, retry or expand budget')
        if not (out/'failure.json').exists():write(out/'failure.json',failure)
        write(out/'status.json',failure,exclusive=False)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term_handler)
        for log in logs.values():log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('qualification-plan');p.add_argument('--output_root',type=Path,required=True)
    p.add_argument('--source_run',type=Path,default=DEFAULT_RUN)
    p.add_argument('--additional_source',action='append',default=[])
    p=subs.add_parser('qualify');p.add_argument('--output_root',type=Path,required=True)
    p=subs.add_parser('prepare');p.add_argument('--output_root',type=Path,required=True)
    p.add_argument('--preservation_plan',type=Path,default=DEFAULT_GATES)
    p=subs.add_parser('run');p.add_argument('--output_root',type=Path,required=True)
    p=subs.add_parser('status');p.add_argument('--output_root',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='qualification-plan':qualification_plan(args)
    elif args.command=='qualify':qualify(args)
    elif args.command=='prepare':prepare(args)
    elif args.command=='run':run(args)
    else:print(json.dumps(compact_status(args.output_root.resolve()),ensure_ascii=False))


if __name__=='__main__':main()
