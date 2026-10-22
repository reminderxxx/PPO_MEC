"""Prepare/check an independent bounded stream protocol; never auto-launch.

Run requires a separately hashed root diagnostic-recovery decision receipt.
Learners start from scratch. Existing train/dev assets and predictor weights are
read-only; no raw-data/provider reconstruction, predictor fitting or holdout use.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time
import traceback
import zlib

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'joint_stream_long_v1'
DECISION_SCHEMA = 'stream_long_execution_decision_v1'
BASELINE_COMMIT = '8c836f9edbaedb42e99dd3026ea6935796533ba0'
METHODS = ['sa_ghmappo', 'mappo', 'ppo', 'dag_offload_drl',
    'sa_no_prediction', 'cache_offload_drl', 'dt_handoff_drl']
SEEDS = [7, 17, 27]
EPISODES = 504
PROBE_POINTS = list(range(0, 505, 24))
FULL_POINTS = list(range(0, 505, 72))
TAIL_STABILITY = dict(episodes=[360,432,504],panel='full48',
    scopes=['overall','seed7','seed17','seed27'],
    reward_relative_range=.01,completion_percentage_point_range=1.,model_cost_relative_range=.02,
    relative_spread='(max(values)-min(values))/abs(mean(values))',zero_mean='undefined_fail',
    zero_completed_requests='undefined_cost_fail',missing_or_incomplete_panel='fail',
    aggregation=dict(reward='arithmetic mean of all unique full48 rows in the scope',
        request_completion_percent='100*sum(completed_requests)/sum(request_count)',
        model_kib_per_completed_request='sum(sum(model_cache[*].transferred_bytes))/1024/sum(completed_requests)'),
    required_reports=['tail metrics/checks per method and scope','final504_minus_initial0 reward per method and scope',
        'final504 absolute reward/completion/cost and negative results'],
    interpretation='Development descriptive bounded-tail stability only; a low-performance plateau is not learning success, mathematical convergence or independent generalization.')
GIB = 1024 ** 3
MIB = 1024 ** 2
OUTPUT_BYTE_COUNT_SCOPE = ('Conservative cumulative write charges: checkpoint writes charged full16MiB upperbound '
    'before saving with no refund, with cumulative checkpoint writes bounded by that reservation; '
    'streams/metadata actual cumulative bytes, including replaced progress writes. '
    'Not measured resident directory size.')
LIMITS = dict(learner_models=21, evaluation_models=462, predictor_models=22176,
    training_sessions=10584, evaluation_sessions=11592, environments=22176,
    training_frames=5080320, evaluation_frames=5564160, frames_per_session=480,
    learner_updates=2646, backward_calls=317520, main_optimizer_steps=317520,
    surrogate_optimizer_steps=90720, categorical_sample_calls=121927680,
    evaluate_calls=170629200, checkpoints=462, max_wall_seconds=7*24*3600,
    max_rss_bytes=8*GIB, minimum_free_disk_bytes=30*GIB,
    output_bytes=20*GIB, failure_receipt_bytes=65536, checkpoint_bytes=16*MIB,
    training_jsonl_bytes_per_cell=16*MIB, dev_jsonl_bytes_per_cell=32*MIB,
    updates_jsonl_bytes_per_cell=4*MIB, isolation_jsonl_bytes_per_cell=8*MIB,
    final_events_bytes_per_cell=256*MIB, final_events_raw_bytes_per_cell=2*GIB,
    final_receipts_bytes_per_cell=64*MIB, final_receipts_raw_bytes_per_cell=256*MIB)


def utc(): return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1048576), b''): digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def protocol_identity(plan):
    unsigned={key:value for key,value in plan.items() if key!='protocol_id'}
    return SCHEMA+'_'+hashlib.sha256(canonical(unsigned)).hexdigest()[:20]


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('Duplicate JSON key: '+key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write_json(path, value, *, exclusive=True, budget=None):
    path = Path(path)
    wire=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if budget is not None:budget.charge('output_bytes',len(wire.encode()))
    if exclusive:
        with path.open('x') as handle:
            handle.write(wire)
    else:
        temporary = path.with_name(path.name+'.tmp')
        with temporary.open('w') as handle:
            handle.write(wire)
        temporary.replace(path)


def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def clean_source(expected=None):
    head = git('rev-parse', 'HEAD')
    if expected is not None and head != expected: raise ValueError('Frozen runtime HEAD differs')
    if git('status', '--porcelain'): raise ValueError('Clean source required before preparation/run')
    subprocess.check_call(['git', 'diff', '--quiet', BASELINE_COMMIT, 'HEAD', '--', 'src'], cwd=ROOT)
    return head


def source_bindings():
    paths = [*(ROOT/'src').rglob('*.py'), *(ROOT/'scripts').glob('*.py')]
    return {str(path.relative_to(ROOT)):sha_file(path) for path in sorted(paths)}


def check_bindings(bindings, root=None):
    for name, expected in bindings.items():
        path = Path(name)
        if root is not None:
            if path.is_absolute() or '..' in path.parts: raise ValueError('Source path escapes repository')
            path = root/path
        if sha_file(path) != expected: raise ValueError('Frozen SHA mismatch: '+str(path))


def source_protocol(path):
    path = Path(path).resolve(); original = read_json(path)
    if (original.get('protocol') != 'joint_stream_learning_v2'
        or original.get('action_contract') != 'joint_stream_control_v2'
        or original.get('smoke') is not False or original.get('episodes') != 64
        or original.get('seeds') != SEEDS or original.get('train_case_count') != 72
        or original.get('dev_case_count') != 48):
        raise ValueError('Original non-smoke 72-train/48-dev stream v2 asset required')
    if set(original['assets']) != {'protocol.pt','predictor.pt'}:
        raise ValueError('Only original protocol/predictor assets are eligible')
    assets = {}
    for name, expected in original['assets'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Asset path escapes original protocol')
        asset = path.parent/relative
        if sha_file(asset) != expected: raise ValueError('Original asset SHA mismatch: '+name)
        assets[str(asset.resolve())] = expected
    return original, dict(path=str(path), sha256=sha_file(path)), assets


def panel_ids(episode, plan):
    if type(episode) is not int or episode not in PROBE_POINTS: raise ValueError('Unfrozen checkpoint point')
    return plan['dev_case_ids'] if episode in FULL_POINTS else plan['probe_case_ids']


def budget_counts():
    cells = len(METHODS)*len(SEEDS)
    unique_eval = len(FULL_POINTS)*48+(len(PROBE_POINTS)-len(FULL_POINTS))*12
    return dict(cells=cells, training_sessions=cells*EPISODES,
        unique_evaluation_sessions=cells*unique_eval,
        logical_full_panel_rows=cells*len(FULL_POINTS)*48,
        logical_probe_panel_rows=cells*len(PROBE_POINTS)*12,
        reused_probe_rows_at_full_points=cells*len(FULL_POINTS)*12,
        learner_updates=cells*(EPISODES//4), checkpoints=cells*len(PROBE_POINTS),
        predictor_models=cells*(EPISODES+unique_eval), learner_models=cells,
        evaluation_models=cells*len(PROBE_POINTS))


def build_plan(original, metadata, schedule_builder):
    """Validate metadata and balanced cycles without a model/env/forward call."""
    if set(metadata['cases']) != {'train','dev'}: raise ValueError('Holdout/unknown cases forbidden')
    train = metadata['cases']['train']; dev = metadata['cases']['dev']
    train_ids = [case['case_id'] for case in train]; dev_ids = [case['case_id'] for case in dev]
    if len(train_ids) != 72 or len(set(train_ids)) != 72 or len(dev_ids) != 48 or len(set(dev_ids)) != 48:
        raise ValueError('Exact unique 72/48 cases required')
    if set(train_ids)&set(dev_ids): raise ValueError('Train/dev case IDs overlap')
    probes = original['curve_probe_ids']
    if len(probes) != 12 or len(set(probes)) != 12 or not set(probes)<=set(dev_ids):
        raise ValueError('Original fixed twelve probes required')
    by_id = {case['case_id']:case for case in dev}
    if any(by_id[key]['families'] != 1 or by_id[key]['capacity'] != 4096 for key in probes):
        raise ValueError('Original probe restriction differs; cannot claim narrow panel')
    for split,cases in (('train',train),('dev',dev)):
        for case in cases:
            session = metadata['sessions'][case['session']]
            if session['split'] != split or len(session['frames']) != 480:
                raise ValueError('Original case split/session span differs')
            if len(session['vehicle_ids']) != 8 or len(session['rsus']) != 4:
                raise ValueError('Eight vehicles/four RSUs required')
    schedule = list(schedule_builder(train,EPISODES))
    if len(schedule) != EPISODES: raise ValueError('Balanced schedule length differs')
    for start in range(0,EPISODES,72):
        if Counter(schedule[start:start+72]) != Counter(train_ids):
            raise ValueError('Every 72-session block must contain every train case once')
        if schedule[start:start+72] != schedule[:72]:
            raise ValueError('Seven complete identical balanced cycles required')
    return dict(methods=METHODS, seeds=SEEDS, episodes=EPISODES, training_budget=EPISODES,
        epochs=4, update_every=4, training_case_ids=train_ids, dev_case_ids=dev_ids,
        probe_case_ids=probes, training_schedule={str(seed):list(schedule) for seed in SEEDS},
        full_points=FULL_POINTS, probe_points=PROBE_POINTS, checkpoint_points=PROBE_POINTS,
        primary_checkpoint=504, schedule_contract='seven_complete_balanced_72_case_cycles',
        probe_scope='Original 12 fixed dev cases: families=1 and capacity=4096 only; not full-panel convergence evidence.',
        evaluation_contract='full48 at multiples72; probe12 at multiples24; full-point probe rows reused, no duplicate environment',
        counts=budget_counts(), limits=LIMITS, output_byte_count_scope=OUTPUT_BYTE_COUNT_SCOPE, predictor_fit_calls=0,
        tail_stability=TAIL_STABILITY,
        action_contract='joint_stream_control_v2', formal_holdout='sealed_not_read',
        selection='fixed_final504_no_peak_selection_no_resume_no_retry', paper_claim_ready=False)


def prepare(args):
    head = clean_source(); original, spec, assets = source_protocol(args.source_protocol)
    output = args.output_root.resolve()
    if output.exists(): raise FileExistsError('Fresh preparation root required; do not overwrite/retry')
    if not output.parent.is_dir(): raise ValueError('Preparation parent must exist')
    # Read-only CPU deserialization of original assets, never original prepare/fit.
    if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
    import torch
    from src.envs.core.joint_stream_refinement import balanced_training_schedule
    data = torch.load(Path(spec['path']).parent/'protocol.pt',map_location='cpu',weights_only=False)
    plan = build_plan(original,data,balanced_training_schedule); del data
    plan.update(schema=SCHEMA,status='frozen',created_at=utc(),repo_root=str(ROOT),
        source_commit=head,baseline_source_commit=BASELINE_COMMIT,source_fingerprints=source_bindings(),
        source_protocol=spec,source_assets=assets,output_root=str(output),run_root=str(output/'run'),
        execution_policy='Requires separately hashed root diagnostic-recovery decision; preparation is not launch authorization')
    plan['protocol_id'] = protocol_identity(plan)
    clean_source(head); check_bindings(assets)
    if sha_file(spec['path']) != spec['sha256']: raise ValueError('Original manifest changed during preparation')
    output.mkdir(exist_ok=False)
    write_json(output/'manifest.json',plan)
    write_json(output/'prepare_command.json',dict(command=sys.argv,created_at=utc(),training_sessions=0,predictor_fit_calls=0))
    print(json.dumps(dict(status='prepared_not_authorized_to_run',protocol=str(output/'manifest.json'),
        protocol_sha256=sha_file(output/'manifest.json'),protocol_id=plan['protocol_id'],counts=plan['counts'])))
    return 0


def validate_plan(plan):
    if plan.get('schema') != SCHEMA or plan.get('status') != 'frozen': raise ValueError('Independent frozen long protocol required')
    expected = dict(methods=METHODS,seeds=SEEDS,episodes=EPISODES,training_budget=EPISODES,
        epochs=4,update_every=4,full_points=FULL_POINTS,probe_points=PROBE_POINTS,
        checkpoint_points=PROBE_POINTS,primary_checkpoint=504,counts=budget_counts(),limits=LIMITS,
        output_byte_count_scope=OUTPUT_BYTE_COUNT_SCOPE,
        tail_stability=TAIL_STABILITY,
        predictor_fit_calls=0,action_contract='joint_stream_control_v2',formal_holdout='sealed_not_read',
        baseline_source_commit=BASELINE_COMMIT)
    for key,value in expected.items():
        if canonical(plan.get(key)) != canonical(value): raise ValueError('Frozen scope differs: '+key)
    if Path(plan['repo_root']).resolve()!=ROOT: raise ValueError('Protocol belongs to another repository')
    if Path(plan['run_root']).resolve()!=Path(plan['output_root']).resolve()/'run': raise ValueError('Run root differs')
    if len(plan['training_case_ids'])!=72 or len(set(plan['training_case_ids']))!=72 or len(plan['dev_case_ids'])!=48 or len(set(plan['dev_case_ids']))!=48:
        raise ValueError('Case identity counts differ')
    if len(plan['probe_case_ids'])!=12 or len(set(plan['probe_case_ids']))!=12 or not set(plan['probe_case_ids'])<=set(plan['dev_case_ids']):
        raise ValueError('Probe identity differs')
    for seed in SEEDS:
        schedule=plan['training_schedule'][str(seed)]
        if len(schedule)!=504 or any(Counter(schedule[start:start+72])!=Counter(plan['training_case_ids']) for start in range(0,504,72)):
            raise ValueError('Balanced full cycles differ')
        if schedule!=plan['training_schedule']['7']: raise ValueError('Methods/seeds must share the same fixed schedule')
        if any(schedule[start:start+72]!=schedule[:72] for start in range(0,504,72)):
            raise ValueError('Seven identical complete cycles required')


def scheduled_cell(schedule,session,update,evaluate,training_row,update_row,progress):
    """Pure orchestration: four original sessions/update, fixed points, no selection."""
    if len(schedule)!=504:raise ValueError('Exactly504 original scheduled sessions required')
    buffer=[];checkpoints=[evaluate(0)]
    for episode,case_id in enumerate(schedule,1):
        progress('training',episode=episode,case_id=case_id)
        row,rolls=session(case_id);buffer.extend(rolls);del rolls;loss=None
        if episode%4==0:
            progress('update')
            if not buffer:raise RuntimeError('Original update has no stochastic behavior rows')
            loss=update(buffer);buffer.clear();update_row(episode,loss)
        training_row(episode,case_id,row,loss)
        if episode in PROBE_POINTS:checkpoints.append(evaluate(episode))
    if buffer or len(checkpoints)!=22:raise RuntimeError('Incomplete fixed-final504 orchestration')
    return checkpoints


def check_protocol(path, *, require_unclaimed=True):
    path=Path(path).resolve(); plan=read_json(path); validate_plan(plan)
    if plan.get('protocol_id')!=protocol_identity(plan):raise ValueError('Independent protocol semantic identity differs')
    if path!=Path(plan['output_root']).resolve()/'manifest.json': raise ValueError('Manifest path/root identity differs')
    clean_source(plan['source_commit'])
    if source_bindings()!=plan['source_fingerprints']: raise ValueError('Runtime source file inventory/SHA differs')
    check_bindings(plan['source_fingerprints'],ROOT); check_bindings(plan['source_assets'])
    original,spec,assets=source_protocol(plan['source_protocol']['path'])
    if spec!=plan['source_protocol'] or assets!=plan['source_assets']: raise ValueError('Original source identity differs')
    if require_unclaimed and Path(plan['run_root']).exists(): raise FileExistsError('Run already claimed; no resume/retry')
    return plan,path,original


def check_decision(path, protocol_path, plan):
    """A local execution gate, not a self-authenticating scientific review."""
    receipt=read_json(path)
    expected=dict(schema=DECISION_SCHEMA,execution_authorized=True,
        decision='authorize_frozen_long_run',reviewer_role='root_independent_diagnostic_recovery',
        protocol_sha256=sha_file(protocol_path),protocol_id=plan['protocol_id'],source_commit=plan['source_commit'])
    for key,value in expected.items():
        if type(receipt.get(key)) is not type(value) or receipt[key]!=value:
            raise ValueError('Missing/mismatched independent execution decision: '+key)
    kind=receipt.get('intervention_type')
    if kind not in ('bounded_exposure_budget_test','verified_nonalgorithmic_protocol_fix'):
        raise ValueError('Concrete bounded-budget/nonalgorithmic intervention required')
    if receipt.get('root_cause_status') not in ('UNIDENTIFIED','IDENTIFIED'):
        raise ValueError('Explicit root-cause evidence status required')
    if type(receipt.get('rationale')) is not str or not receipt['rationale'].strip():
        raise ValueError('Nonempty independent diagnostic rationale required')
    if receipt.get('algorithm_stability_fix_verified') is not False:
        raise ValueError('Unmodified algorithm cannot self-certify a stability fix')
    if kind=='verified_nonalgorithmic_protocol_fix' and (
        receipt['root_cause_status']!='IDENTIFIED' or receipt.get('nonalgorithmic_fix_scope') not in
        ('evaluation_panel_and_schedule','statistical_reporting','evaluation_panel_schedule_and_reporting')):
        raise ValueError('Nonalgorithmic fix must identify statistical/panel/schedule scope')
    for key in ('diagnostic_result','independent_review'):
        spec=receipt[key]
        if sha_file(spec['path'])!=spec['sha256']: raise ValueError('Decision evidence SHA differs: '+key)
    diagnostic=read_json(receipt['diagnostic_result']['path'])
    if diagnostic.get('status')!='complete_exact_observer_parity': raise ValueError('Diagnostic run did not complete required parity')
    if not receipt.get('reviewed_at') or not receipt.get('diagnostic_run_id'):
        raise ValueError('Independent diagnostic recovery identity/date required')
    evidence=receipt.get('validation_evidence')
    if type(evidence) is not list or not evidence:raise ValueError('Concrete nonempty validation evidence required')
    for spec in evidence:
        if type(spec) is not dict or sha_file(spec['path'])!=spec['sha256']:
            raise ValueError('Diagnostic validation evidence SHA differs')
    return dict(path=str(Path(path).resolve()),sha256=sha_file(path)),receipt


class Budget:
    def __init__(self,root,*,clock=time.monotonic,rss=None,disk=shutil.disk_usage):
        self.root=Path(root); self.clock=clock; self.started=clock(); self.disk=disk
        self.rss=rss or self.current_rss; self.peak_rss=0; self.context={}
        self.counts={key:0 for key in ('learner_models','evaluation_models','predictor_models',
            'training_sessions','evaluation_sessions','environments','training_frames','evaluation_frames',
            'learner_updates','backward_calls','main_optimizer_steps','surrogate_optimizer_steps',
            'categorical_sample_calls','evaluate_calls','checkpoints','output_bytes')}
    @staticmethod
    def current_rss():
        value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform=='darwin': return int(value)
        if sys.platform.startswith('linux'): return int(value)*1024
        raise RuntimeError('Unsupported RSS platform')
    def check(self,*,disk=False):
        value=self.rss();self.peak_rss=max(self.peak_rss,value)
        if value>LIMITS['max_rss_bytes']: raise RuntimeError('RSS hard limit; no expansion/retry')
        if self.clock()-self.started>LIMITS['max_wall_seconds']: raise RuntimeError('Wall-time hard limit; no extension/retry')
        if disk and self.disk(self.root.parent).free<LIMITS['minimum_free_disk_bytes']:
            raise RuntimeError('At least30GiB free disk required')
    def charge(self,key,n=1):
        if type(n) is not int or n<0 or key not in self.counts: raise ValueError('Invalid operation charge')
        if self.counts[key]+n>LIMITS[key]: raise RuntimeError('Call/output cap before original: '+key)
        self.counts[key]+=n;self.check()
    def progress(self,phase,**identity):
        self.context.update(identity,phase=phase);self.check()
        write_json(self.root/'progress.json',dict(schema=SCHEMA,updated_at=utc(),context=self.context,
            counts=self.counts,peak_rss_bytes=self.peak_rss,elapsed_seconds=self.clock()-self.started),exclusive=False,budget=self)


class BoundedText:
    """Streaming JSONL text sink with pre-write row/raw/compressed/output caps."""
    def __init__(self,path,budget,*,rows,raw_bytes,file_bytes,gzip=False):
        self.path=Path(path);self.budget=budget;self.row_cap=rows;self.raw_cap=raw_bytes;self.file_cap=file_bytes
        self.rows=0;self.raw_bytes=0;self.file_bytes=0;self.closed=False
        self.compressor=zlib.compressobj(level=1,wbits=31) if gzip else None
        self.file=self.path.open('xb')
    def _output(self,wire,*,reserve=0):
        if self.file_bytes+len(wire)+reserve>self.file_cap: raise RuntimeError('File byte cap: '+self.path.name)
        self.budget.charge('output_bytes',len(wire));self.file.write(wire);self.file.flush();self.file_bytes+=len(wire)
    def write(self,text):
        if self.closed or type(text) is not str or not text.endswith('\n') or text.count('\n')!=1:
            raise ValueError('One complete JSONL record required')
        wire=text.encode()
        if self.rows+1>self.row_cap or self.raw_bytes+len(wire)>self.raw_cap: raise RuntimeError('Record/raw cap: '+self.path.name)
        self.budget.check()
        data=self.compressor.compress(wire) if self.compressor else wire
        self._output(data,reserve=64 if self.compressor else 0)
        self.rows+=1;self.raw_bytes+=len(wire)
        return len(text)
    def row(self,value): return self.write(canonical(value).decode()+'\n')
    def flush(self): self.file.flush()
    def close(self):
        if self.closed:return
        try:
            if self.compressor:self._output(self.compressor.flush())
        finally:self.file.close();self.closed=True
    def __enter__(self):return self
    def __exit__(self,typ,value,tb):
        try:self.close()
        except BaseException:
            if typ is None:raise
        return False


@contextmanager
def preserve_rng(torch,np):
    state=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    try:yield state
    finally:random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])


class Runtime:
    def __init__(self,budget,torch):self.b=budget;self.t=torch;self.learner=None;self.phase='setup'
    def __enter__(self):
        from scripts.sa_stage5c_numerical_qualification import Patches
        from src.agents.ppo_agent import ScheduledControlPPO
        self.patches=Patches();self.patches.__enter__();p=self.patches;t=self.t;b=self.b
        try:
            if t.cuda.is_initialized():raise RuntimeError('CPU-only run refuses initialized CUDA')
            def forbidden(*a,**k):raise RuntimeError('Operation forbidden in frozen CPU stream run')
            p.set(t.cuda,'_lazy_init',forbidden);p.set(t.Tensor,'cuda',forbidden);p.set(t.nn.Module,'cuda',forbidden)
            p.set(t.autograd,'grad',forbidden)
            original_eval=ScheduledControlPPO._evaluate
            def evaluate(this,*a,**k):
                b.charge('evaluate_calls');answer=original_eval(this,*a,**k);b.check();return answer
            p.set(ScheduledControlPPO,'_evaluate',evaluate)
            original_backward=t.Tensor.backward
            def backward(tensor,*a,**k):
                if self.phase!='train':raise RuntimeError('Evaluation/setup backward forbidden')
                b.charge('backward_calls');answer=original_backward(tensor,*a,**k);b.check();return answer
            p.set(t.Tensor,'backward',backward)
            original_step=t.optim.Adam.step
            def step(optimizer,*a,**k):
                if self.phase!='train' or self.learner is None:raise RuntimeError('Non-learner Adam forbidden')
                key='main_optimizer_steps' if optimizer is self.learner.optimizer else 'surrogate_optimizer_steps' if optimizer is self.learner.surrogate_optimizer else None
                if key is None:raise RuntimeError('Unexpected optimizer/predictor fitting forbidden')
                b.charge(key);answer=original_step(optimizer,*a,**k);b.check();return answer
            p.set(t.optim.Adam,'step',step)
            original_sample=t.distributions.Categorical.sample
            def sample(distribution,*a,**k):
                if self.phase!='train':raise RuntimeError('Greedy evaluation must not sample')
                b.charge('categorical_sample_calls');return original_sample(distribution,*a,**k)
            p.set(t.distributions.Categorical,'sample',sample)
            return self
        except BaseException:self.patches.close();raise
    def __exit__(self,*exc):self.patches.close();return False


def run_session(case,data,original,agent,runtime,make_env,stream_episode,*,evaluate=False,event_file=None,receipt_file=None,identity=None):
    b=runtime.b;phase='evaluate' if evaluate else 'train';runtime.phase=phase
    b.charge('evaluation_sessions' if evaluate else 'training_sessions');b.charge('environments');b.charge('predictor_models')
    env=make_env(case,data,original);original_step=env.step_controls;frames=[0]
    def step(controls):
        if frames[0]>=480:raise RuntimeError('Frame cap before original environment step')
        b.charge('evaluation_frames' if evaluate else 'training_frames');frames[0]+=1
        result=original_step(controls);b.check();return result
    env.step_controls=step
    row,rolls=stream_episode(env,agent,deterministic=evaluate,event_file=event_file,receipt_file=receipt_file,identity=identity)
    if row['steps']!=frames[0] or not 0<frames[0]<=480:raise RuntimeError('Session frame ledger mismatch')
    return row,rolls


def learner_digest(agent):
    from scripts.diagnose_stream_stability import exact_digest
    from scripts.sa_stage5c_numerical_qualification import learned_state,controller,extra_agent_state
    return exact_digest(dict(learned=learned_state(agent),controller=controller(agent),extra=extra_agent_state(agent)))


class BoundedCheckpointFile:
    def __init__(self, handle, cap, state):self.handle=handle;self.cap=cap;self.state=state

    def write(self, value):
        size=memoryview(value).nbytes
        if self.handle.tell()+size>self.cap:raise RuntimeError('Checkpoint byte cap exceeded before write')
        if self.state['write_bytes']+size>self.cap:raise RuntimeError('Cumulative checkpoint reservation exceeded before write')
        self.state['write_bytes']+=size
        return self.handle.write(value)

    def fileno(self):raise io.UnsupportedOperation('Bounded checkpoint writes require the checked file-like path')

    def __getattr__(self, name):return getattr(self.handle,name)


@contextmanager
def checkpoint_save_boundary(torch, checkpoint, cap):
    target=Path(checkpoint).resolve();original_save=torch.save;state=dict(write_bytes=0,save_calls=0)
    def bounded_save(value, f, *args, **kwargs):
        if isinstance(f,(str,os.PathLike)) and Path(f).resolve()==target:
            state['save_calls']+=1
            with target.open('wb') as handle:
                return original_save(value,BoundedCheckpointFile(handle,cap,state),*args,**kwargs)
        return original_save(value,f,*args,**kwargs)
    torch.save=bounded_save
    try:yield state
    finally:torch.save=original_save


def save_checkpoint_bounded(agent, checkpoint, budget, torch):
    checkpoint=Path(checkpoint)
    if checkpoint.exists():raise FileExistsError('Checkpoint overwrite forbidden')
    budget.charge('checkpoints')
    # Precharge the whole allowed write before the original saver can touch disk.
    # Attempts consume their reservation even if saving fails or produces fewer bytes.
    budget.charge('output_bytes',LIMITS['checkpoint_bytes'])
    with checkpoint_save_boundary(torch,checkpoint,LIMITS['checkpoint_bytes']) as writes:
        agent.save_checkpoint(checkpoint)
    if writes['save_calls']==0:raise RuntimeError('Checkpoint saver bypassed bounded torch.save')
    actual_bytes=checkpoint.stat().st_size
    if actual_bytes>LIMITS['checkpoint_bytes']:raise RuntimeError('Checkpoint byte cap exceeded')
    return dict(actual_bytes=actual_bytes,write_bytes=writes['write_bytes'],save_calls=writes['save_calls'])


def evaluate_point(agent,method,seed,episode,home,plan,protocol_sha,data,original,runtime,files,make_env,stream_episode,torch,np):
    from scripts.diagnose_stream_stability import exact_digest
    from scripts.sa_stage5c_numerical_qualification import Patches,rng_state
    from scripts.train_joint_stream_control import actor_hash
    from scripts.train_joint_tail_control import tensor_hash
    b=runtime.b;b.progress('checkpoint_evaluation',method=method,seed=seed,episode=episode)
    if int(agent.completed_updates)!=episode//4:raise RuntimeError('Checkpoint update index differs')
    before=learner_digest(agent);rng_before=exact_digest(rng_state(torch,np));checkpoint=home/('ep%03d.pt'%episode)
    with preserve_rng(torch,np):
        saved=save_checkpoint_bounded(agent,checkpoint,b,torch);cp_sha=sha_file(checkpoint)
        identity=dict(protocol=SCHEMA,protocol_id=plan['protocol_id'],protocol_sha256=protocol_sha,
            source_commit=plan['source_commit'],method=method,seed=seed,episode=episode,
            checkpoint_sha256=cp_sha,checkpoint_actual_bytes=saved['actual_bytes'],checkpoint_write_bytes=saved['write_bytes'],
            checkpoint_save_calls=saved['save_calls'],checkpoint_charged_bytes=LIMITS['checkpoint_bytes'],
            output_byte_count_scope=OUTPUT_BYTE_COUNT_SCOPE,training_budget=504,completed_updates=episode//4,
            resume_supported=False,primary=episode==504)
        write_json(checkpoint.with_suffix('.identity.json'),identity,budget=b)
        b.charge('evaluation_models');evaluated=type(agent).from_checkpoint(checkpoint)
        if tensor_hash(evaluated)!=tensor_hash(agent) or actor_hash(evaluated)!=actor_hash(agent):raise RuntimeError('Evaluation copy differs')
        if evaluated.training_budget!=504 or evaluated.prediction_enabled!=agent.prediction_enabled:raise RuntimeError('Evaluation options differ')
        def forbidden(*a,**k):raise RuntimeError('Independent evaluation agent cannot update')
        with Patches() as patches:
            patches.set(evaluated,'update',forbidden);patches.set(evaluated.optimizer,'step',forbidden);patches.set(evaluated.surrogate_optimizer,'step',forbidden)
            by_id={case['case_id']:case for case in data['cases']['dev']}
            for case_id in panel_ids(episode,plan):
                case=by_id[case_id];panels=['full48'] if episode in FULL_POINTS else []
                if case_id in plan['probe_case_ids']:panels.append('probe12')
                observed_identity=dict(protocol_id=plan['protocol_id'],method=method,seed=seed,episode=episode,case_id=case_id)
                row,rolls=run_session(case,data,original,evaluated,runtime,make_env,stream_episode,evaluate=True,
                    event_file=files['events'] if episode==504 else None,receipt_file=files['receipts'] if episode==504 else None,identity=observed_identity)
                files['dev'].row(dict(observed_identity,panels=panels,checkpoint_sha256=cp_sha,**row));del rolls
        if tensor_hash(evaluated)!=tensor_hash(agent) or actor_hash(evaluated)!=actor_hash(agent):raise RuntimeError('Evaluation changed model tensors')
        del evaluated;gc.collect()
    after=learner_digest(agent);rng_after=exact_digest(rng_state(torch,np))
    if before!=after or rng_before!=rng_after:raise RuntimeError('Evaluation disturbed learner/controller/optimizer/RNG')
    files['isolation'].row(dict(identity,learner_before=before,learner_after=after,rng_before=rng_before,rng_after=rng_after,
        rng_and_learner_unchanged=True,unique_evaluation_cases=len(panel_ids(episode,plan))))
    return identity


def cell_files(home,budget,stack):
    result={}
    for key,name,rows,cap in (('train','training.jsonl',504,LIMITS['training_jsonl_bytes_per_cell']),
        ('dev','dev.jsonl',552,LIMITS['dev_jsonl_bytes_per_cell']),('updates','updates.jsonl',126,LIMITS['updates_jsonl_bytes_per_cell']),
        ('isolation','rng_isolation.jsonl',22,LIMITS['isolation_jsonl_bytes_per_cell'])):
        result[key]=stack.enter_context(BoundedText(home/name,budget,rows=rows,raw_bytes=cap,file_bytes=cap))
    result['events']=stack.enter_context(BoundedText(home/'final_events.jsonl.gz',budget,rows=48*480*8,
        raw_bytes=LIMITS['final_events_raw_bytes_per_cell'],file_bytes=LIMITS['final_events_bytes_per_cell'],gzip=True))
    result['receipts']=stack.enter_context(BoundedText(home/'final_receipts.jsonl.gz',budget,rows=48*480,
        raw_bytes=LIMITS['final_receipts_raw_bytes_per_cell'],file_bytes=LIMITS['final_receipts_bytes_per_cell'],gzip=True))
    return result


def execute(args):
    plan,path,original=check_protocol(args.protocol)
    decision_spec,decision=check_decision(args.decision_receipt,path,plan)
    protocol_sha=sha_file(path);root=Path(plan['run_root']);b=Budget(root);b.check(disk=True)
    root.mkdir(exist_ok=False)
    write_json(root/'run_started.json',dict(schema=SCHEMA,started_at=utc(),pid=os.getpid(),protocol_sha256=protocol_sha,
        decision=decision_spec,command=sys.argv,source_commit=plan['source_commit'],no_resume=True,no_retry=True),budget=b)
    cells=[]
    try:
        b.progress('numerical_imports')
        if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
        import numpy as np
        import torch
        from scripts.train_joint_stream_control import make_agent,make_env,stream_episode
        from src.envs.core.joint_stream_refinement import balanced_training_schedule
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        data=torch.load(Path(plan['source_protocol']['path']).parent/'protocol.pt',map_location='cpu',weights_only=False)
        rebuilt=build_plan(original,data,balanced_training_schedule)
        for key,value in rebuilt.items():
            if canonical(plan.get(key))!=canonical(value):raise RuntimeError('Derived frozen plan mismatch: '+key)
        train={case['case_id']:case for case in data['cases']['train']}
        with Runtime(b,torch) as runtime:
            for method in METHODS:
                for seed in SEEDS:
                    b.check(disk=True);home=root/method/('seed'+str(seed));home.mkdir(parents=True,exist_ok=False)
                    b.progress('initialize',method=method,seed=seed,episode=0)
                    before_counts=dict(b.counts)
                    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);b.charge('learner_models')
                    agent=make_agent(method,seed,504);runtime.learner=agent
                    if agent.training_budget!=504 or int(agent.completed_updates)!=0 or agent.optimizer.state or agent.surrogate_optimizer.state:
                        raise RuntimeError('Fresh learner/empty Adam/504-budget required')
                    if any(param.device.type!='cpu' for param in agent.parameters()):raise RuntimeError('CPU learner required')
                    with ExitStack() as stack:
                        files=cell_files(home,b,stack)
                        def session(case_id):return run_session(train[case_id],data,original,agent,runtime,make_env,stream_episode)
                        def update(buffer):
                            b.charge('learner_updates');runtime.phase='train'
                            return agent.update(buffer,epochs=4)
                        def evaluate(episode):return evaluate_point(agent,method,seed,episode,home,plan,protocol_sha,data,original,runtime,files,make_env,stream_episode,torch,np)
                        def training_row(episode,case_id,row,loss):files['train'].row(dict(method=method,seed=seed,episode=episode,case_id=case_id,losses=loss,**row))
                        def update_row(episode,loss):
                            if int(agent.completed_updates)!=episode//4:raise RuntimeError('Actual update count mismatch')
                            files['updates'].row(dict(method=method,seed=seed,episode=episode,update=episode//4,losses=loss))
                        checkpoints=scheduled_cell(plan['training_schedule'][str(seed)],session,update,evaluate,training_row,update_row,b.progress)
                        if int(agent.completed_updates)!=126:raise RuntimeError('Incomplete fixed-final learner')
                        if (files['train'].rows,files['dev'].rows,files['updates'].rows,files['isolation'].rows)!=(504,552,126,22):
                            raise RuntimeError('Cell JSONL row accounting differs')
                    counts={key:b.counts[key]-before_counts[key] for key in b.counts}
                    if counts['training_sessions']!=504 or counts['evaluation_sessions']!=552 or counts['learner_updates']!=126:
                        raise RuntimeError('Cell session/update accounting differs')
                    files_out={str(p.relative_to(home)):sha_file(p) for p in home.iterdir() if p.is_file()}
                    receipt=dict(schema=SCHEMA,status='complete_fixed504',method=method,seed=seed,counts=counts,
                        checkpoints=checkpoints,output_sha256=files_out,primary_checkpoint='ep504.pt',
                        predictor_fit_calls=0,formal_holdout='not_read',no_peak_selection=True)
                    write_json(home/'result.json',receipt,budget=b);cells.append(receipt);runtime.learner=None
                    del agent,files,checkpoints;gc.collect();b.progress('cell_complete')
        check_protocol(path,require_unclaimed=False)
        if sha_file(path)!=protocol_sha or sha_file(decision_spec['path'])!=decision_spec['sha256']:
            raise RuntimeError('Protocol/decision changed during run')
        check_decision(decision_spec['path'],path,plan)
        for key in ('learner_models','evaluation_models','predictor_models','training_sessions','evaluation_sessions','environments','learner_updates','checkpoints'):
            if b.counts[key]!=LIMITS[key]:raise RuntimeError('Incomplete frozen total: '+key)
        write_json(root/'result.json',dict(schema=SCHEMA,status='complete_fixed504',protocol_sha256=protocol_sha,
            protocol_id=plan['protocol_id'],decision=decision_spec,counts=b.counts,limits=LIMITS,cells=cells,
            elapsed_seconds=b.clock()-b.started,peak_rss_bytes=b.peak_rss,predictor_fit_calls=0,
            tail_stability=dict(criteria=plan['tail_stability'],audit_status='pending_independent_consumer'),
            output_byte_count_scope=OUTPUT_BYTE_COUNT_SCOPE+' Counts precede this final receipt.',
            paper_claim_ready=False,formal_holdout='not_read',
            evidence_boundary='Bounded longer development training; not a demonstrated stability fix, convergence guarantee or formal promotion.'),budget=b)
        b.progress('complete');return 0
    except BaseException as exc:
        try:
            failure=dict(schema=SCHEMA,status='failed',failed_at=utc(),exception_type=type(exc).__name__,
                message=str(exc)[:2048],traceback=traceback.format_exc()[:32768],context=b.context,counts=b.counts,completed_cells=len(cells),
                no_resume=True,no_retry=True,possibly_partial_optimizer_step=True,
                receipt_scope='Separate bounded64KiB failure reserve outside cumulative guarded output quota; may be unavailable on I/O failure.')
            if len((json.dumps(failure,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())>LIMITS['failure_receipt_bytes']:
                raise RuntimeError('Failure receipt reserve exceeded')
            write_json(root/'failure.json',failure)
        except BaseException:pass
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--source_protocol',type=Path,required=True);p.add_argument('--output_root',type=Path,required=True)
    for mode in ('check','run'):
        p=sub.add_parser(mode);p.add_argument('--protocol',type=Path,required=True)
        p.add_argument('--decision_receipt',type=Path,required=mode=='run')
    args=parser.parse_args(argv)
    if args.command=='prepare':return prepare(args)
    if args.command=='run':return execute(args)
    plan,path,_=check_protocol(args.protocol)
    decision=None
    if args.decision_receipt:decision,_=check_decision(args.decision_receipt,path,plan)
    Budget(Path(plan['run_root'])).check(disk=True)
    print(json.dumps(dict(status='checked_not_started',protocol_id=plan['protocol_id'],counts=plan['counts'],
        execution_decision_present=decision is not None,decision=decision)))
    return 0


if __name__=='__main__':raise SystemExit(main())
