"""Synthetic-only acceptance for the separately frozen missing-initial task.

The parent package is an explicitly fabricated temporary software fixture; its
hash constants are monkeypatched only inside tests. No real policy evaluation,
source data, hidden results, training or predictor fitting is consumed.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import gzip
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch

import scripts.complete_stream_initial_evidence as runner
import scripts.audit_stream_initial_evidence as consumer
from scripts.run_stream_reward_curve import build_policy, checkpoint_policy
from tests.test_stream_control_learning import env
from src.envs.core.multi_request_stream_env import MultiRequestStreamEnv


def write(path, value):
    Path(path).write_text(runner.json_text(value))


def tiny_env():
    original = env()
    arrivals = [replace(a, frame=4 if a.frame else 0) for a in original.arrivals]
    return MultiRequestStreamEnv(frames=original.frames[:6], arrivals=arrivals,
        catalog=original.catalog_objects, rsu_states=original.rsus, costs=original.costs,
        workload_bytes_per_unit=1, shared_state_enabled=False, coordination_intent_enabled=True,
        link_reservation_enabled=True, model_cache_config={'capacity_bytes':8, 'transfer_bytes_per_frame':4})


def fixture_plan(root):
    limits=dict(maximum_total_physical_frames=23040, maximum_wall_seconds=10800,
                maximum_output_bytes=536870912, maximum_frames_per_episode=480)
    return dict(contract=runner.CONTRACT, seeds=runner.SEEDS, training_case_ids=runner.CASES,
        probe_case_ids=runner.PROBES, initial_replay_episodes=runner.REPLAYS,
        source_commit='synthetic_fixture', source_fingerprints={},
        parent_campaign=dict(path=str(root/'parent/campaign.json'), sha256=runner.PARENT_CAMPAIGN_SHA),
        parent_audit=dict(path=str(root/'parent/collection/audit.json'), sha256=runner.PARENT_AUDIT_SHA),
        output=str(root/'new'), audit_output=str(root/'new/audit'), budgets=limits,
        training_frames=0, fit_calls=0, downloads=0, hidden_episodes=0)


def make_parent(root):
    home=root/'parent'; (home/'collection').mkdir(parents=True)
    source=home/'source.json'; write(source, {'synthetic':True})
    e=tiny_env()
    cases=[dict(case_id=c, session=0, arrivals=deepcopy(e.arrivals)) for c in runner.CASES]
    data=dict(cases={'train':cases}, sessions=[dict(split='train', frames=e.frames)])
    pt=home/'protocol.pt'; torch.save(data,pt)
    parent=dict(source_commit=runner.PARENT_SOURCE, seeds=runner.SEEDS,
        training_case_ids=runner.CASES, probe_case_ids=runner.PROBES,
        source_fingerprints={}, outputs={'main':{}},
        protocol=dict(path=str(source), sha256=runner.file_hash(source),
                      protocol_pt=str(pt), protocol_pt_sha256=runner.file_hash(pt)))
    cells=[]
    for method in ['main','mappo','ppo','dqn']:
        for seed in runner.SEEDS:
            if method!='main':
                cells.append(dict(method=method,seed=seed));continue
            cell=home/f'main_seed{seed}'; (cell/'checkpoints').mkdir(parents=True)
            parent['outputs']['main'][str(seed)]=str(cell)
            policy=build_policy('main',seed)
            budget=runner.budget_for(cell, fixture_plan(root))
            cp=checkpoint_policy(policy,cell/'checkpoints/ep000.pt',
                dict(method='main',seed=seed,episode=0,cumulative_training_frames=0,cumulative_updates=0),budget)
            cp96=deepcopy(cp);cp96.update(episode=96,sha256='f'*64)
            evaluation=[]
            for episode,ids in [(0,runner.PROBES),(96,runner.CASES)]:
                for cid in ids:
                    row={name:0 for name in runner.METRICS}
                    row.update(method='main',seed=seed,case_id=cid,episode=episode,checkpoint_episode=episode,
                        checkpoint_sha256=cp['sha256'] if episode==0 else cp96['sha256'],
                        reward=1. if episode==0 else 2.,physical_terminal=True,
                        is_curve_probe=cid in runner.PROBES,is_fixed_final=episode==96)
                    evaluation.append(row)
            cells.append(dict(method='main',seed=seed,checkpoints=[cp,cp96],evaluation=evaluation))
    write(home/'campaign.json',parent)
    rules=[dict(case_id=c,**{name:0 for name in runner.METRICS}) for c in runner.CASES]
    audit=dict(status='curve_receipts_and_checkpoints_audited',source_commit=runner.PARENT_SOURCE,
        campaign_sha256=runner.file_hash(home/'campaign.json'),cells=cells,rules=rules,
        final_selection='episode96_fixed_no_peak',training_case_ids=runner.CASES,probe_case_ids=runner.PROBES)
    write(home/'collection/audit.json',audit)
    return parent,audit


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    torch.set_num_threads(1)
    root=tmp_path_factory.mktemp('initial_evidence_synthetic')
    parent,audit=make_parent(root)
    prospective=fixture_plan(root)
    prospective['parent_campaign']['sha256']=runner.file_hash(root/'parent/campaign.json')
    prospective['parent_audit']['sha256']=runner.file_hash(root/'parent/collection/audit.json')
    path=root/'plan.json';write(path,prospective)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner,'PARENT_CAMPAIGN_SHA',prospective['parent_campaign']['sha256'])
        patch.setattr(runner,'PARENT_AUDIT_SHA',prospective['parent_audit']['sha256'])
        patch.setattr(runner,'make_env',lambda *args:tiny_env())
        patch.setattr(runner.IntentAwareStreamSAGHMAPPOAgent,'update',
                      lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('no training allowed')))
        result=runner.run(path,runner.file_hash(path),verify=False)
    assert result['status']=='initial_attribution_evidence_complete'
    output=root/'new'
    return SimpleNamespace(root=root,plan=prospective,path=path,output=output,
        original={p:p.read_bytes() for p in output.rglob('*') if p.is_file()})


@contextmanager
def edited(package):
    """Restore only the synthetic package, never production or scientific files."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner,'PARENT_CAMPAIGN_SHA',package.plan['parent_campaign']['sha256'])
        patch.setattr(runner,'PARENT_AUDIT_SHA',package.plan['parent_audit']['sha256'])
        try:
            # The audit is a create-only consumer; each independent tamper case
            # receives a fresh synthetic consumer directory after saved output removal.
            for p in sorted((package.output/'audit').rglob('*'),reverse=True):
                if p.is_file():p.unlink()
                elif p.is_dir():p.rmdir()
            (package.output/'audit').rmdir()
            yield patch
        finally:
            for p in package.output.rglob('*'):
                if p.is_file() and p not in package.original:p.unlink()
            for p,b in package.original.items():
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(b)


def read_frames(package):
    with gzip.open(package.output/'frames.jsonl.gz','rt') as f:return [json.loads(x) for x in f]


def change_frames(package,rows):
    with gzip.open(package.output/'frames.jsonl.gz','wt') as f:
        for r in rows:f.write(runner.json_text(r))
    refresh(package)


def refresh(package):
    p=package.output/'complete.json';d=json.loads(p.read_text())
    for name in d['output_sha256']:d['output_sha256'][name]=runner.file_hash(package.output/name)
    write(p,d)


def consume(package):
    return consumer.consume(package.path,runner.file_hash(package.path),verify=False)


def test_complete_new_matrix_reuse_and_descriptive_report(completed):
    report=json.loads((completed.output/'audit/audit.json').read_text())
    summary=json.loads((completed.output/'audit/summary.json').read_text())
    assert report['new_initial_episodes']==45 and report['reused_initial_probes']==15
    assert report['reused_final_episodes']==60 and report['reused_rule_physical_records']==12
    assert report['new_initial_replays']==3 and report['replay_digests_match']
    assert report['actual_physical_frames']==48*6
    assert report['training_frames']==report['fit_calls']==0
    assert len(summary['per_seed'])==5 and len(summary['per_condition'])==12
    assert summary['inferential_confidence_intervals_computed'] is False
    assert summary['paper_goal_complete'] is False


@pytest.mark.parametrize('mutation',[
    lambda p:p.update(seeds=[7]),lambda p:p.update(training_case_ids=runner.CASES[:-1]),
    lambda p:p.update(probe_case_ids=[]),lambda p:p.update(initial_replay_episodes=[]),
    lambda p:p.update(training_frames=1),lambda p:p.update(fit_calls=1),lambda p:p.update(downloads=1),
    lambda p:p['budgets'].update(maximum_total_physical_frames=23041),
    lambda p:p['budgets'].update(maximum_wall_seconds=10801),
    lambda p:p['budgets'].update(maximum_output_bytes=536870913),
    lambda p:p['budgets'].update(maximum_frames_per_episode=481),
    lambda p:p.update(audit_output='/tmp/outside-total-disk-budget'),
    lambda p:p['parent_audit'].update(sha256='0'*64),
])
def test_plan_rejects_scope_or_budget_drift(tmp_path,mutation):
    p=fixture_plan(tmp_path);mutation(p)
    with pytest.raises(ValueError):runner.validate_plan(p,verify_files=False)


def test_create_only_executor_and_consumer(completed):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner,'PARENT_CAMPAIGN_SHA',completed.plan['parent_campaign']['sha256'])
        patch.setattr(runner,'PARENT_AUDIT_SHA',completed.plan['parent_audit']['sha256'])
        with pytest.raises(FileExistsError):runner.run(completed.path,runner.file_hash(completed.path),verify=False)
        with pytest.raises(FileExistsError):consume(completed)


@pytest.mark.parametrize('kind',['reward','receipt','frame_gap','post_terminal','commands','native_event',
                                 'handoff_event','released','terminal_flag','extra','omit','replay_literal'])
def test_native_tamper_rejected_even_after_hash_refresh(completed,kind):
    with edited(completed):
        rows=read_frames(completed)
        if kind=='reward':rows[0]['reward']+=1
        elif kind=='receipt':rows[0]['receipt']['reward_components']['made_up']=1
        elif kind=='frame_gap':rows[0]['frame_index']=1
        elif kind=='post_terminal':rows[0]['terminal_after_step']=rows[0]['done_returned']=True
        elif kind=='commands':rows[0]['commands']={}
        elif kind=='native_event':rows[0]['events'][0]['status']='tampered'
        elif kind=='handoff_event':
            rows[0]['events'][0]['handoff_failed']=True;rows[0]['native_events'][0]['handoff_failed']=True
        elif kind=='released':rows[0]['released_request_ids']=[]
        elif kind=='terminal_flag':rows[-1]['terminal_after_step']=False
        elif kind=='extra':rows.insert(0,deepcopy(rows[0]))
        elif kind=='omit':rows.pop(0)
        elif kind=='replay_literal':
            row=next(r for r in rows if r['phase']=='replay')
            row['commands'][next(iter(row['commands']))]['extra']='changed'
        change_frames(completed,rows)
        with pytest.raises(ValueError):consume(completed)


@pytest.mark.parametrize('kind',['missing','duplicated','seed','checkpoint','steps','completed_requests',
                                'state_transfer_bytes','handoff_failed_count','initial_reused_in_new'])
def test_episode_matrix_metric_and_checkpoint_tamper_rejected(completed,kind):
    with edited(completed):
        p=completed.output/'episodes.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()]
        if kind=='missing':rows.pop(0)
        elif kind=='duplicated':rows[1]=deepcopy(rows[0])
        elif kind=='seed':rows[0]['seed']=17
        elif kind=='checkpoint':rows[0]['checkpoint_sha256']='0'*64
        elif kind=='initial_reused_in_new':rows[0]['case_id']=runner.PROBES[0]
        else:rows[0][kind]+=1
        p.write_text(''.join(runner.json_text(r) for r in rows));refresh(completed)
        with pytest.raises(ValueError):consume(completed)


@pytest.mark.parametrize('kind',['reuse','final','rule','checkpoint','protocol'])
def test_old_reference_selection_or_sha_tamper_rejected(completed,kind):
    with edited(completed):
        p=completed.output/'source_references.json';d=json.loads(p.read_text())
        if kind in ['reuse','final','rule']:
            name={'reuse':'reused_initial','final':'final_reference','rule':'gain_reference'}[kind]
            d[name][0]['record']['reward']+=1
        elif kind=='checkpoint':d['initial_checkpoints'][0]['tensor_sha256']='0'*64
        else:d['protocol']['sha256']='0'*64
        write(p,d);refresh(completed)
        with pytest.raises(ValueError):consume(completed)


def test_hard_frame_wall_disk_and_postterminal_guards(tmp_path):
    p=fixture_plan(tmp_path);tmp_path.mkdir(exist_ok=True)
    b=runner.budget_for(tmp_path,p)
    b.frames['evaluation']=23040
    with pytest.raises(RuntimeError):b.before_step('evaluation')
    b.frames['evaluation']=0;b.started=time.monotonic()-10800
    with pytest.raises(RuntimeError):b.before_step('evaluation')
    b.started=time.monotonic();b.limits['maximum_cell_disk_bytes']=65536
    with pytest.raises(RuntimeError):b.check(1)
    e=tiny_env();e.frame_index=len(e.frames)
    with pytest.raises(ValueError):runner.GuardedEnv(e,b,'evaluation')


def test_frame_budget_stops_before_third_physical_interaction(tmp_path):
    plan=fixture_plan(tmp_path);budget=runner.budget_for(tmp_path,plan)
    budget.limits['maximum_evaluation_frames_per_cell']=2
    writer=runner.GzipRecords(tmp_path/'partial.gz',budget);physical=tiny_env()
    try:
        with pytest.raises(RuntimeError,match='frame budget'):
            runner.episode(physical,build_policy('main',7),budget,
                dict(method='main',seed=7,case_id='fixture',phase='initial',checkpoint_episode=0,
                     checkpoint_sha256='synthetic'),writer)
    finally:writer.close()
    assert physical.frame_index==budget.frames['evaluation']==2
    with gzip.open(tmp_path/'partial.gz','rt') as stream:assert len(list(stream))==2


def test_claim_future_clock_is_not_a_wall_budget_reset(completed):
    with edited(completed):
        p=completed.output/'launch_claim.json';claim=json.loads(p.read_text());claim['started_unix']=time.time()+10000
        write(p,claim);refresh(completed)
        with pytest.raises(ValueError,match='clock'):consume(completed)


def test_full_native_idle_late_tail_is_observed(tmp_path):
    p=fixture_plan(tmp_path);b=runner.budget_for(tmp_path,p)
    writer=runner.GzipRecords(tmp_path/'native.gz',b);a=build_policy('main',7)
    row=runner.episode(env(),a,b,dict(method='main',seed=7,case_id='software',phase='initial',
        checkpoint_episode=0,checkpoint_sha256='synthetic'),writer);writer.close()
    with gzip.open(tmp_path/'native.gz','rt') as f:frames=[json.loads(x) for x in f]
    assert len(frames)==row['steps'] and row['steps']>45
    assert row['last_scheduled_release_frame']==45 and len(row['released_request_ids'])==3
    assert any(not f['commands'] for f in frames)
    assert frames[-1]['terminal_after_step'] and all(not f['terminal_after_step'] for f in frames[:-1])


def test_changed_checkpoint_bytes_rejected_without_policy_call(completed):
    with edited(completed):
        refs=json.loads((completed.output/'source_references.json').read_text())
        cp=Path(refs['initial_checkpoints'][0]['path']);original=cp.read_bytes()
        try:
            cp.write_bytes(original+b'changed')
            with pytest.raises(ValueError,match='checkpoint hash'):consume(completed)
        finally:cp.write_bytes(original)


def test_failed_terminal_and_resource_bill_rejected(completed):
    for field,value in [('status','failed'),('actual_physical_frames',23041),('elapsed_seconds',10800)]:
        with edited(completed):
            path=completed.output/'complete.json';row=json.loads(path.read_text());row[field]=value;write(path,row)
            with pytest.raises(ValueError):consume(completed)


def test_failure_preserved_without_restart_or_partial_completion(tmp_path,monkeypatch):
    make_parent(tmp_path)
    p=fixture_plan(tmp_path)
    p['parent_campaign']['sha256']=runner.file_hash(tmp_path/'parent/campaign.json')
    p['parent_audit']['sha256']=runner.file_hash(tmp_path/'parent/collection/audit.json')
    monkeypatch.setattr(runner,'PARENT_CAMPAIGN_SHA',p['parent_campaign']['sha256'])
    monkeypatch.setattr(runner,'PARENT_AUDIT_SHA',p['parent_audit']['sha256'])
    calls=[]
    def fail(*args):
        calls.append(True);raise RuntimeError('synthetic env construction failure')
    monkeypatch.setattr(runner,'make_env',fail)
    plan=tmp_path/'plan.json';write(plan,p)
    with pytest.raises(RuntimeError,match='synthetic'):runner.run(plan,runner.file_hash(plan),verify=False)
    output=tmp_path/'new'
    assert (output/'failure.json').is_file() and (output/'launch_claim.json').is_file()
    assert not (output/'complete.json').exists() and len(calls)==1
    with pytest.raises(FileExistsError):runner.run(plan,runner.file_hash(plan),verify=False)
    assert len(calls)==1


def compatibility_fixture(tmp_path):
    paths=runner.COMPATIBLE_SOURCE_PATHS+['src/other.py']
    old={}
    for path in paths:
        f=tmp_path/path;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('old');old[path]=runner.file_hash(f)
    new={}
    for path in runner.COMPATIBLE_SOURCE_PATHS:
        (tmp_path/path).write_text('new');new[path]=runner.file_hash(tmp_path/path)
    proof=dict(contract='stream_initial_legacy_source_compatibility_v1',status='passed',
        parent_source_commit=runner.PARENT_SOURCE,
        changes={p:dict(old_sha256=old[p],new_sha256=new[p]) for p in new},
        assertions={p:True for p in runner.COMPATIBILITY_ASSERTIONS},
        validation=dict(command='synthetic exact default bitwise tests',passed=7,failures=0))
    qualification=tmp_path/'qualification.json';write(qualification,proof)
    plan=dict(source_fingerprints=new,source_compatibility=dict(approved_changed_paths=runner.COMPATIBLE_SOURCE_PATHS,
        qualification=dict(path=str(qualification),sha256=runner.file_hash(qualification))))
    return plan,dict(source_fingerprints=old),proof,qualification


def test_exact_two_path_source_compatibility_preserves_double_source(tmp_path):
    plan,parent,proof,path=compatibility_fixture(tmp_path)
    result=runner.source_compatibility(plan,parent,root=tmp_path)
    assert result['changes']==proof['changes']
    assert result['qualification']['sha256']==runner.file_hash(path)
    assert result['status']=='passed'


@pytest.mark.parametrize('kind',['third_path','no_approval','wrong_allowlist','failed','no_assertion',
                                 'false_assertion','old_sha','new_sha','missing_change','zero_tests',
                                 'failed_tests','wrong_parent','unfrozen_new_sha','changed_qualification'])
def test_source_compatibility_never_generalizes_authorization(tmp_path,kind):
    plan,parent,proof,path=compatibility_fixture(tmp_path)
    if kind=='third_path':(tmp_path/'src/other.py').write_text('changed')
    elif kind=='no_approval':plan.pop('source_compatibility')
    elif kind=='wrong_allowlist':plan['source_compatibility']['approved_changed_paths']=['src/other.py']
    elif kind=='failed':proof['status']='failed'
    elif kind=='no_assertion':proof['assertions'].pop(next(iter(proof['assertions'])))
    elif kind=='false_assertion':proof['assertions'][next(iter(proof['assertions']))]=False
    elif kind in ['old_sha','new_sha']:proof['changes'][runner.COMPATIBLE_SOURCE_PATHS[0]][kind+'256']='0'*64
    elif kind=='missing_change':proof['changes'].pop(runner.COMPATIBLE_SOURCE_PATHS[0])
    elif kind=='zero_tests':proof['validation']['passed']=0
    elif kind=='failed_tests':proof['validation']['failures']=1
    elif kind=='wrong_parent':proof['parent_source_commit']='other'
    elif kind=='unfrozen_new_sha':plan['source_fingerprints'][runner.COMPATIBLE_SOURCE_PATHS[0]]='0'*64
    write(path,proof)
    if kind!='changed_qualification':plan.get('source_compatibility',{}).get('qualification',{})['sha256']=runner.file_hash(path)
    else:path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError):runner.source_compatibility(plan,parent,root=tmp_path)

