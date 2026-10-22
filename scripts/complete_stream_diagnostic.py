"""Complete only the two never-run seeds of a failed observer diagnostic.

This is a new exclusive profile, not resume/retry or completion of the old root.
``check`` is standard-library only; ``run`` requires a frozen parent failure and
an independently recovered seed7 receipt. No launch authority is emitted here.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import diagnose_stream_stability as original

SCHEMA = 'unexecuted_stream_diagnostic_v1'
SUCCESS = 'complete_unexecuted_seed_observer_parity'
PARTIAL_STATUS = 'RECOVERED_SEED7_ONLY_OBSERVER_PARITY'
SEEDS = [17, 27]
SIDES = ['off', 'on']
LIMITS = dict(training_sessions=16, training_updates=4,
    frames_per_session=480, training_frames=7680, models=4, environments=16,
    audit_rows_per_seed=100000, audit_bytes_per_seed=6*original.GIB,
    minimum_free_disk_bytes=30*original.GIB, max_rss_bytes=8*original.GIB,
    max_wall_seconds=7200, backward_calls=480, adam_calls=960,
    categorical_sample_calls=184320, evaluate_calls=123360)
REQUIRED_SCRIPTS = (*original.REQUIRED_SCRIPTS, 'scripts/complete_stream_diagnostic.py')
read_json = original.read_json
write_json = original.write_json
exact_digest = original.exact_digest


def sha_file(path,guard=None):
    digest=hashlib.sha256()
    if guard is not None:guard.check()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1048576),b''):
            if guard is not None:guard.check()
            digest.update(block)
    if guard is not None:guard.check()
    return digest.hexdigest()


def same_contract(left, right):
    """JSON object order is immaterial; scalar/list types remain exact."""
    if type(left) is not type(right):return False
    if type(right) is dict:
        return set(left)==set(right) and all(same_contract(left[key],right[key]) for key in right)
    if type(right) is list:
        return len(left)==len(right) and all(same_contract(a,b) for a,b in zip(left,right))
    return exact_digest(left)==exact_digest(right)


def bound_json(spec,guard=None):
    if type(spec) is not dict or set(spec) != {'path', 'sha256'}:
        raise ValueError('Exact path/SHA binding required')
    path = Path(spec['path']).resolve()
    if str(path) != spec['path'] or sha_file(path,guard) != spec['sha256']:
        raise ValueError('Bound JSON source SHA/path differs')
    return read_json(path), path


def check_shape(profile):
    expected = dict(schema=SCHEMA, status='frozen', seeds=SEEDS, sides=SIDES,
        sessions_per_seed=4, training_budget=64, epochs=4, limits=LIMITS,
        baseline_source_commit=original.BASELINE_SOURCE_COMMIT,
        original_scientific_budget=dict(training_sessions=24, training_updates=6),
        already_executed=dict(seeds=[7], training_sessions=8, training_updates=2),
        compact_joins=True, no_resume=True, no_retry=True)
    for key, value in expected.items():
        if not same_contract(profile.get(key),value):
            raise ValueError('Independent remaining scope differs: '+key)
    if Path(profile['repo_root']).resolve() != ROOT:
        raise ValueError('Profile belongs to another repository')
    if type(profile.get('source_fingerprints')) is not dict or not profile['source_fingerprints']:
        raise ValueError('Frozen runtime source bindings required')


def parent_required_files(old_root):
    names = ['run_started.json', 'progress.json', 'seed7/off/result.json', 'seed7/on/audit/events.jsonl']
    names += [f'seed7/{side}/episode{episode}.json' for side in SIDES for episode in range(1,5)]
    names += [f'seed7/{side}/{name}' for side in SIDES for name in ('checkpoint.pt','final_evidence.pt')]
    return {str((old_root/name).resolve()) for name in names}


def recovery_input_paths(old_root,old_contract,protocol):
    paths=dict(original_contract=old_contract,failure=old_root/'failure.json',progress=old_root/'progress.json',
        run_started=old_root/'run_started.json',off_result=old_root/'seed7/off/result.json',
        audit=old_root/'seed7/on/audit/events.jsonl',protocol_manifest=Path(protocol['path']).resolve())
    for side in SIDES:
        for filename in ['checkpoint.pt','final_evidence.pt']:
            paths[side+'_'+filename.replace('.','_')]=old_root/'seed7'/side/filename
        for episode in range(1,5):paths[f'{side}_episode{episode}']=old_root/'seed7'/side/f'episode{episode}.json'
    return {name:str(path.resolve()) for name,path in paths.items()}


def check_parent(profile, protocol,guard=None):
    old, old_path = bound_json(profile['parent_contract'],guard)
    failure, failure_path = bound_json(profile['parent_failure'],guard)
    recovery, recovery_path = bound_json(profile['partial_seed7_recovery'],guard)
    if (old.get('schema') != original.SCHEMA or old.get('status') != 'frozen'
        or not same_contract(old.get('seeds'),[7,17,27])
        or not same_contract(old.get('sides'),SIDES)
        or type(old.get('sessions_per_seed')) is not int or old['sessions_per_seed'] != 4
        or type(old.get('training_budget')) is not int or old['training_budget'] != 64
        or type(old.get('epochs')) is not int or old['epochs'] != 4
        or not same_contract(old.get('limits'),original.FIXED_LIMITS)
        or old.get('baseline_source_commit') != original.BASELINE_SOURCE_COMMIT):
        raise ValueError('Original frozen 24-session diagnostic required')
    if not same_contract(old['protocol'],profile['protocol']):
        raise ValueError('Remaining profile must use the original protocol')
    old_root = Path(old['output_root']).resolve()
    if failure_path != old_root/'failure.json':
        raise ValueError('Parent failure belongs to another root')
    if (failure.get('schema') != original.SCHEMA or failure.get('status') != 'failed'
        or failure.get('no_retry') is not True or not same_contract(failure.get('completed_seeds'),0)
        or not same_contract({key:failure.get('counts',{}).get(key) for key in ['training_sessions','training_updates','models','environments']},
            dict(training_sessions=8,training_updates=2,models=2,environments=8))
        or failure.get('context',{}).get('seed') != 7
        or failure.get('context',{}).get('side') != 'on'
        or failure.get('context',{}).get('phase') != 'independent_consumer'):
        raise ValueError('Expected seed7 consumer failure after exactly8/2 operations')
    if (recovery.get('status') != PARTIAL_STATUS or recovery.get('original_run_status') != 'failed'
        or not same_contract(recovery.get('recovered_seeds'),[7])
        or not same_contract(recovery.get('not_run_seeds'),SEEDS)):
        raise ValueError('Independently recovered seed7-only parity receipt required')
    zero_counts=recovery.get('counts')
    if (type(zero_counts) is not dict or not {'training_sessions','training_updates'}<=set(zero_counts)
        or any(type(value) is not int or value!=0 for value in zero_counts.values())):
        raise ValueError('Partial recovery must declare zero new scientific operations')
    if recovery_path.is_relative_to(old_root):
        raise ValueError('Partial recovery must remain outside the failed original root')
    for relative in ['result.json','seed7/result.json','seed7/on/result.json']:
        if (old_root/relative).exists():
            raise ValueError('Failed original root must not be backfilled as success')
    for seed in SEEDS:
        if (old_root/f'seed{seed}').exists():
            raise ValueError('A requested seed already has an original claimed directory')
    output = Path(profile['output_root']).resolve()
    if output == old_root or output.is_relative_to(old_root) or old_root.is_relative_to(output):
        raise ValueError('New output must be separate from the failed root')
    bindings = profile.get('parent_bindings')
    if type(bindings) is not dict or not parent_required_files(old_root).issubset(bindings):
        raise ValueError('All eight old episode and paired final source bindings required')
    for name, expected in bindings.items():
        path = Path(name).resolve()
        if str(path) != name or not path.is_relative_to(old_root) or sha_file(path,guard) != expected:
            raise ValueError('Parent raw receipt SHA/path differs')
    if recovery.get('qualification_root')!=str(old_root) or recovery.get('original_source_commit')!=old['source_commit']:
        raise ValueError('Partial recovery root/source identity differs')
    if recovery.get('saved_final_files_byte_equal') is not True or recovery.get('four_episode_exact_digest_parity') is not True:
        raise ValueError('Partial recovery saved/episode parity evidence missing')
    recovered_inputs=recovery.get('input_bindings')
    expected_paths=recovery_input_paths(old_root,old_path,profile['protocol'])
    if type(recovered_inputs) is not dict or set(recovered_inputs)!=set(expected_paths):
        raise ValueError('Exact partial recovery input binding inventory required')
    recovered_map={}
    for label,spec in recovered_inputs.items():
        if type(spec) is not dict or set(spec)!={'path','sha256','size_bytes'}:raise ValueError('Partial recovery binding shape differs')
        name=spec['path'];bound=Path(name).resolve()
        if name!=expected_paths[label] or str(bound)!=name or name in recovered_map or sha_file(bound,guard)!=spec['sha256']:
            raise ValueError('Partial recovery source SHA/path/uniqueness differs')
        if 'size_bytes' in spec and (type(spec['size_bytes']) is not int or bound.stat().st_size!=spec['size_bytes']):
            raise ValueError('Partial recovery source size differs')
        recovered_map[name]=spec['sha256']
    needed={**{name:bindings[name] for name in parent_required_files(old_root)},
        str(old_path):profile['parent_contract']['sha256'],str(failure_path):profile['parent_failure']['sha256']}
    if any(recovered_map.get(name)!=expected for name,expected in needed.items()):
        raise ValueError('Partial recovery does not bind this failed contract and all paired originals')
    start = read_json(old_root/'run_started.json')
    if start.get('contract_sha256') != sha_file(old_path,guard) or start.get('source_commit') != old['source_commit']:
        raise ValueError('Parent execution did not bind the declared frozen contract/source')
    for side in SIDES:
        for episode in range(1,5):
            row = read_json(old_root/f'seed7/{side}/episode{episode}.json')
            if type(row.get('episode')) is not int or row['episode'] != episode or row.get('case_id') != protocol['training_schedule']['7'][episode-1]:
                raise ValueError('Old eight episode receipts differ from the original schedule')
    off = read_json(old_root/'seed7/off/result.json')
    if (off.get('seed') != 7 or off.get('side') != 'off' or off.get('episode_count') != 4
        or off.get('case_ids') != protocol['training_schedule']['7'][:4]
        or off.get('original_call_counts',{}).get('training_sessions') != 4
        or off.get('original_call_counts',{}).get('training_updates') != 1):
        raise ValueError('Old completed off-side identity differs')
    # Old source files stay bound in their own repository; the new consumer is
    # deliberately allowed to differ, while the scientific src remains frozen.
    old_repo = Path(old['repo_root']).resolve()
    if original.git(old_repo,'rev-parse','HEAD') != old['source_commit']:
        raise ValueError('Original failed-run source HEAD changed')
    old_required=set(original.REQUIRED_SCRIPTS)|{str(p.relative_to(old_repo)) for p in (old_repo/'src').rglob('*.py')}
    if not old_required.issubset(old['source_fingerprints']):
        raise ValueError('Original failed-run source binding inventory incomplete')
    for name, expected in old['source_fingerprints'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha_file(old_repo/relative,guard) != expected:
            raise ValueError('Original failed-run source binding changed')
    return old_root


def check_contract(path, *, require_unclaimed=True,guard=None):
    if guard is not None:guard.check()
    path = Path(path).resolve(); profile = read_json(path); check_shape(profile)
    output = Path(profile['output_root']).resolve()
    if require_unclaimed and output.exists():
        raise FileExistsError('New remaining-seed profile already claimed; no resume/retry')
    if not output.parent.is_dir():
        raise ValueError('New output parent must exist')
    if original.git(ROOT,'rev-parse','HEAD') != profile['source_commit'] or original.git(ROOT,'status','--porcelain'):
        raise ValueError('Exact clean new frozen source required')
    original.subprocess.check_call(['git','diff','--quiet',original.BASELINE_SOURCE_COMMIT,'HEAD','--','src'],cwd=ROOT)
    required = set(REQUIRED_SCRIPTS) | {str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.py')}
    if not required.issubset(profile['source_fingerprints']):
        raise ValueError('Required new runner/compact consumer/source bindings missing')
    for name, expected in profile['source_fingerprints'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha_file(ROOT/relative,guard) != expected:
            raise ValueError('New source SHA/path differs')
    protocol, protocol_path = bound_json(profile['protocol'],guard)
    if (protocol.get('protocol') != 'joint_stream_learning_v2' or protocol.get('action_contract') != 'joint_stream_control_v2'
        or protocol.get('smoke') is not False or type(protocol.get('episodes')) is not int or protocol['episodes'] != 64
        or not same_contract(protocol.get('seeds'),[7,17,27])
        or set(protocol.get('assets',{})) != {'protocol.pt','predictor.pt'}):
        raise ValueError('Original non-smoke stream64 assets required')
    for name, expected in protocol['assets'].items():
        if sha_file(protocol_path.parent/name,guard) != expected:
            raise ValueError('Original protocol asset SHA differs')
    for seed in [7,17,27]:
        schedule = protocol.get('training_schedule',{}).get(str(seed))
        if type(schedule) is not list or len(schedule) != 64:
            raise ValueError('Original complete64 schedule required')
    check_parent(profile,protocol,guard)
    if guard is not None:guard.check()
    return profile, protocol, path, protocol_path


class Resources(original.Resources):
    """Independent remaining allowance; never mutate old FIXED_LIMITS."""
    def check(self, *, disk=False):
        used=self.rss(); self.peak_rss_bytes=max(self.peak_rss_bytes,used)
        if used>LIMITS['max_rss_bytes']:
            raise RuntimeError('New profile RSS cap; no automatic expansion/retry')
        if self.clock()-self.started>LIMITS['max_wall_seconds']:
            raise RuntimeError('New profile wall cap; no automatic extension/retry')
        if disk and self.disk(self.output.parent).free<LIMITS['minimum_free_disk_bytes']:
            raise RuntimeError('At least30GiB free disk required')

    def charge(self,name,amount=1):
        if type(amount) is not int or amount<0 or name not in self.counts:
            raise ValueError('Invalid remaining-profile charge')
        if self.counts[name]+amount>LIMITS[name]:
            raise RuntimeError('New remaining cap before original: '+name)
        self.counts[name]+=amount;self.check()

    def progress(self,phase,**identity):
        self.context.update(identity,phase=phase);self.check()
        value=dict(schema=SCHEMA,updated_at=datetime.now(timezone.utc).isoformat(),context=self.context,
            counts=self.counts,peak_rss_bytes=self.peak_rss_bytes,elapsed_seconds=self.clock()-self.started)
        temporary=self.output/'progress.json.tmp'
        temporary.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
        temporary.replace(self.output/'progress.json')


def execute(contract_path):
    # Charge wall/RSS from before streaming preflight of the old multi-GB log.
    initial_path=Path(contract_path).resolve()
    initial_profile=read_json(initial_path);check_shape(initial_profile)
    out=Path(initial_profile['output_root']).resolve();budget=Resources(out);budget.check(disk=True)
    initial_contract_sha=sha_file(initial_path,budget)
    profile,protocol,path,protocol_path=check_contract(initial_path,guard=budget)
    contract_sha=sha_file(path,budget)
    if Path(path).resolve()!=initial_path or not same_contract(profile,initial_profile) or contract_sha!=initial_contract_sha:
        raise RuntimeError('Remaining profile changed during preclaim validation')
    original.claim_output(out,dict(schema=SCHEMA,contract_sha256=contract_sha,
        source_commit=profile['source_commit'],command=sys.argv,no_retry=True,no_resume=True,
        seed_scope=SEEDS,started_at=datetime.now(timezone.utc).isoformat()))
    completed=[]
    try:
        budget.progress('numerical_imports')
        import numpy as np
        import torch
        from scripts.train_joint_stream_control import make_agent,make_env,stream_episode
        from scripts.sa_stage5c_credit_audit import AuditSink,validate
        from scripts.sa_stage5c_credit_observer import credit_audit,detached
        from scripts.sa_stage5c_numerical_qualification import (compare_exact,learned_state,
            controller,extra_agent_state,rng_state,restore_rng)
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():raise RuntimeError('CPU-only remaining profile refuses CUDA')
        data=torch.load(protocol_path.parent/'protocol.pt',map_location='cpu',weights_only=False)
        if set(data['cases'])!={'train','dev'}:raise ValueError('No holdout/unknown split permitted')
        training={case['case_id']:case for case in data['cases']['train']}
        if len(training)!=len(data['cases']['train']):raise ValueError('Duplicate training case')
        class GuardedSink(AuditSink):
            def emit(self,*args,**kwargs):
                budget.check();answer=super().emit(*args,**kwargs);budget.check();return answer
        for seed in SEEDS:
            budget.check(disk=True);seed_home=out/f'seed{seed}';seed_home.mkdir()
            cases=protocol['training_schedule'][str(seed)][:4]
            if any(case not in training for case in cases):raise ValueError('Original first-four case missing')
            pair=[];common_rng=None
            for side in SIDES:
                home=seed_home/side;home.mkdir();budget.context=dict(seed=seed,side=side,phase='initialize')
                budget.progress('initialize');random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                budget.charge('models');agent=make_agent('sa_ghmappo',seed,64)
                if any(p.device.type!='cpu' for p in agent.parameters()):raise RuntimeError('CPU learner required')
                if int(agent.completed_updates)!=0 or agent.optimizer.state or agent.surrogate_optimizer.state:
                    raise RuntimeError('Fresh zero-update empty-Adam learner required')
                initial=exact_digest(learned_state(agent));start_rng=rng_state(torch,np)
                if side=='off':common_rng=start_rng;initial_hash=initial
                else:
                    if initial!=initial_hash:raise AssertionError('Fresh on/off initial states differ')
                    compare_exact(start_rng,common_rng,'fresh RNG')
                restore_rng(common_rng,torch,np);enabled=side=='on'
                sink=GuardedSink(home/'audit',enable=enabled,compact_joins=True,
                    header=dict(run_uid=out.name+f'_seed{seed}',source_commit=profile['source_commit'],
                        evidence_kind='unverified_actual_chain',provenance=dict(source_run_id=out.name,
                            protocol_path=str(protocol_path),protocol_sha256=profile['protocol']['sha256'],
                            contract_sha256=contract_sha,seed=seed,side=side,
                            source='fresh_original_unexecuted_seed_stochastic_stream_sessions')),
                    max_rows=LIMITS['audit_rows_per_seed'],max_bytes=LIMITS['audit_bytes_per_seed'])
                rows=[];buffer=[];before_counts=dict(budget.counts)
                with sink:
                    with original.guarded_runtime(agent,budget,torch) as observed:
                        with credit_audit(agent,sink) as observer:
                            for episode,case_id in enumerate(cases,1):
                                budget.context=dict(seed=seed,side=side,episode=episode,case_id=case_id,phase='rollout')
                                budget.progress('rollout');budget.charge('training_sessions');budget.charge('environments')
                                env=make_env(training[case_id],data,protocol);original_step=env.step_controls;frames=[0]
                                def step(controls,_original=original_step,_frames=frames):
                                    if _frames[0]>=LIMITS['frames_per_session']:raise RuntimeError('Session frame cap before original')
                                    budget.charge('training_frames');_frames[0]+=1
                                    answer=_original(controls);budget.check();return answer
                                env.step_controls=step
                                identity=dict(seed=seed,session_id=out.name+f'_seed{seed}_ep{episode}',
                                    episode=episode,case_id=case_id,phase='original_stochastic_train')
                                with observer.session(identity) if enabled else nullcontext():
                                    summary,rolls=stream_episode(env,agent,deterministic=False)
                                if summary['steps']!=frames[0] or not 0<frames[0]<=480:raise RuntimeError('Frame ledger differs')
                                row=dict(episode=episode,case_id=case_id,summary=detached(summary),
                                    summary_sha256=exact_digest(summary),rolls_sha256=exact_digest(rolls,exclude_audit_uids=True),
                                    receipts_sha256=exact_digest(env.frame_receipts),rng_after_sha256=exact_digest(rng_state(torch,np)))
                                rows.append(row);write_json(home/f'episode{episode}.json',row)
                                if enabled:original.compare_episode_receipt(row,read_json(seed_home/'off'/f'episode{episode}.json'),compare_exact)
                                buffer.extend(rolls);del env,rolls;budget.progress('episode_complete')
                            budget.progress('update');stats=agent.update(buffer,epochs=4)
                            if int(agent.completed_updates)!=1:raise RuntimeError('Exactly one original update required per side')
                            buffer.clear();budget.progress('update_complete')
                        if enabled:sink.finish(complete=True)
                final=dict(learned_state=learned_state(agent),controller=controller(agent),
                    extra_agent_state=extra_agent_state(agent),rng=rng_state(torch,np))
                checkpoint=home/'checkpoint.pt';agent.save_checkpoint(checkpoint)
                compare_exact(final['rng'],rng_state(torch,np),'checkpoint RNG unchanged')
                final['checkpoint_payload']=torch.load(checkpoint,map_location='cpu',weights_only=False)
                torch.save(final,home/'final_evidence.pt')
                del observer,sink;gc.collect();budget.check();budget.progress('independent_consumer')
                with original.guarded_consumer(budget):
                    audit=validate(home/'audit',max_file_bytes=LIMITS['audit_bytes_per_seed'],compact_joins=True) if enabled else None
                if enabled and (audit['status']!='complete_unverified' or not audit['structural_complete'] or audit['actual_training_verified']):
                    raise RuntimeError('Compact observed chain structural completion failed')
                side_counts={key:budget.counts[key]-before_counts[key] for key in before_counts}
                receipt=dict(side=side,seed=seed,initial_learned_state_sha256=initial,case_ids=cases,
                    original_call_counts=side_counts,coverage=detached(observed),update_stats=detached(stats),audit=audit,
                    final_evidence_sha256=sha_file(home/'final_evidence.pt'),checkpoint_sha256=sha_file(checkpoint),episode_count=len(rows))
                write_json(home/'result.json',receipt);pair.append(receipt)
                if enabled:
                    off=torch.load(seed_home/'off/final_evidence.pt',map_location='cpu',weights_only=False)
                    for field in final:compare_exact(final[field],off[field],'final.'+field)
                    compare_exact(pair[0]['original_call_counts'],side_counts,'original operation counts')
                    compare_exact(pair[0]['coverage'],detached(observed),'coverage/update statistics');del off
                del agent,final,checkpoint,observed,rows;gc.collect();budget.check()
            result=dict(seed=seed,status='exact_parity',sides=pair,only_excluded_rollout_field=original.EXCLUDED_UID_KEY,
                coverage_scope='actual observed paths only; absent free-cache/idle remains unverified')
            write_json(seed_home/'result.json',result);completed.append(result);budget.progress('seed_complete')
        check_contract(path,require_unclaimed=False,guard=budget)
        if sha_file(path,budget)!=contract_sha:raise RuntimeError('Remaining profile changed during execution')
        if budget.counts['training_sessions']!=16 or budget.counts['training_updates']!=4 or budget.counts['models']!=4 or budget.counts['environments']!=16:
            raise RuntimeError('Incomplete exact remaining scope')
        result=dict(schema=SCHEMA,status=SUCCESS,counts=budget.counts,limits=LIMITS,seeds=completed,
            source_commit=profile['source_commit'],contract_sha256=contract_sha,
            parent_contract=profile['parent_contract'],parent_failure=profile['parent_failure'],
            partial_seed7_recovery=profile['partial_seed7_recovery'],executed_seeds=SEEDS,
            original_run_status='failed',original_root_completed=False,composite_recovery='not_created_here',
            peak_rss_bytes=budget.peak_rss_bytes,elapsed_seconds=budget.clock()-budget.started,
            evidence_boundary='Only previously unexecuted seeds17/27; original root remains failed. Observer parity is not a stability repair, convergence or launch authorization.',
            formal_holdout='not_read',no_resume=True,no_retry=True)
        write_json(out/'result.json',result);budget.progress('complete');return 0
    except BaseException as exc:
        try:write_json(out/'failure.json',dict(schema=SCHEMA,status='failed',exception_type=type(exc).__name__,message=str(exc),
            traceback=traceback.format_exc(),counts=budget.counts,context=budget.context,completed_seeds=len(completed),
            no_resume=True,no_retry=True,possibly_partial_update=True,original_run_status='failed'))
        except BaseException:pass
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('check','run'))
    p.add_argument('--contract',type=Path,required=True);args=p.parse_args(argv)
    if args.command=='run':return execute(args.contract)
    profile,protocol,path,_=check_contract(args.contract)
    Resources(Path(profile['output_root'])).check(disk=True)
    print(json.dumps(dict(schema=SCHEMA,status='preflight_pass_no_numerical_imports',contract_sha256=sha_file(path),
        seeds=SEEDS,limits=LIMITS,first_four={str(seed):protocol['training_schedule'][str(seed)][:4] for seed in SEEDS})))
    return 0


if __name__=='__main__':raise SystemExit(main())
