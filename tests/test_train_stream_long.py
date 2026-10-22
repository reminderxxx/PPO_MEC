"""Pure metadata/orchestration/resource mocks. No model, env or optimizer work."""
import ast
import copy
import gzip
import importlib.util
import itertools
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/train_stream_long.py'
SPEC=importlib.util.spec_from_file_location('stream_long_pure_helpers',SCRIPT)
M=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=M;SPEC.loader.exec_module(M)


def original_balancer():
    # Execute only the pure original function, not its environment imports.
    tree=ast.parse((ROOT/'src/envs/core/joint_stream_refinement.py').read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='balanced_training_schedule')
    scope={};exec(compile(ast.Module(body=[function],type_ignores=[]),'original_balanced_function','exec'),scope)
    return scope['balanced_training_schedule']


def metadata():
    sessions=[dict(split='train' if i<3 else 'dev',frames=[None]*480,
        vehicle_ids=list(range(8)),rsus=list(range(4))) for i in range(5)]
    cases={'train':[],'dev':[]}
    grid=list(itertools.product(('short','medium','long'),('queued_batch','stream'),(1,2),(4096,8192)))
    for session in range(5):
        split=sessions[session]['split']
        for size,arrival_mode,families,capacity in grid:
            cases[split].append(dict(case_id=split+'_%03d'%len(cases[split]),session=session,
                size=size,arrival_mode=arrival_mode,families=families,capacity=capacity))
    probes=[case['case_id'] for i,case in enumerate(cases['dev']) if i%4==0]
    return dict(curve_probe_ids=probes),dict(sessions=sessions,cases=cases)


class ProtocolTests(unittest.TestCase):
    def plan(self):
        original,data=metadata();return M.build_plan(original,data,original_balancer())

    def test_module_top_level_is_stdlib_only(self):
        names=[]
        for node in ast.parse(SCRIPT.read_text()).body:
            if isinstance(node,ast.Import):names.extend(value.name for value in node.names)
            elif isinstance(node,ast.ImportFrom):names.append(node.module or '')
        self.assertFalse(any(name.split('.')[0] in ('torch','numpy','src','scripts') for name in names))

    def test_seven_complete_balanced_cycles_and_common_seed_schedule(self):
        plan=self.plan();schedule=plan['training_schedule']['7']
        self.assertEqual(len(schedule),504)
        for start in range(0,504,72):
            self.assertEqual(set(schedule[start:start+72]),set(plan['training_case_ids']))
            self.assertEqual(len(set(schedule[start:start+72])),72)
            self.assertEqual(schedule[start:start+72],schedule[:72])
        self.assertEqual(schedule,plan['training_schedule']['17'])
        self.assertEqual(schedule,plan['training_schedule']['27'])
        self.assertEqual(plan['training_budget'],504)

    def test_panels_deduplicate_full_point_probes_and_keep_scope(self):
        plan=self.plan()
        self.assertEqual(len(M.PROBE_POINTS),22);self.assertEqual(len(M.FULL_POINTS),8)
        self.assertEqual(sum(len(M.panel_ids(point,plan)) for point in M.PROBE_POINTS),552)
        self.assertEqual(len(M.panel_ids(0,plan)),48)
        self.assertEqual(len(M.panel_ids(24,plan)),12)
        self.assertEqual(len(M.panel_ids(504,plan)),48)
        self.assertIn('families=1',plan['probe_scope']);self.assertIn('capacity=4096',plan['probe_scope'])
        with self.assertRaises(ValueError):M.panel_ids(25,plan)

    def test_counts_distinguish_learner_eval_predictor_and_reused_rows(self):
        count=M.budget_counts()
        self.assertEqual(M.SCHEMA,'joint_stream_long_four_v1')
        self.assertEqual(M.METHODS,['sa_ghmappo','mappo','ppo','dag_offload_drl'])
        self.assertEqual(count['training_sessions'],6048)
        self.assertEqual(count['unique_evaluation_sessions'],6624)
        self.assertEqual(count['learner_models'],12)
        self.assertEqual(count['evaluation_models'],264)
        self.assertEqual(count['predictor_models'],12672)
        self.assertEqual(count['logical_full_panel_rows']+count['logical_probe_panel_rows']-count['reused_probe_rows_at_full_points'],6624)
        self.assertEqual(count['learner_updates'],1512)
        self.assertEqual(M.LIMITS['training_frames'],6048*480)
        self.assertEqual(M.LIMITS['evaluation_frames'],6624*480)
        self.assertEqual(M.LIMITS['backward_calls'],12*126*120)
        self.assertEqual(M.LIMITS['main_optimizer_steps'],12*126*120)
        self.assertEqual(M.LIMITS['surrogate_optimizer_steps'],3*126*120)

    def test_protocol_identity_survives_json_key_order_but_binds_budget(self):
        plan=self.plan();identity=M.protocol_identity(plan);plan['protocol_id']=identity
        archived=json.loads(json.dumps(plan,sort_keys=True))
        self.assertEqual(M.protocol_identity(archived),identity)
        archived['training_budget']=64
        self.assertNotEqual(M.protocol_identity(archived),identity)

    def test_missing_cases_biased_schedule_changed_probe_or_holdout_rejected(self):
        for kind in ('missing','schedule','probe','holdout'):
            original,data=metadata();builder=original_balancer()
            if kind=='missing':data['cases']['train'].pop()
            elif kind=='schedule':builder=lambda cases,episodes:[cases[0]['case_id']]*episodes
            elif kind=='probe':data['cases']['dev'][0]['families']=2
            else:data['cases']['holdout']=[]
            with self.subTest(kind=kind),self.assertRaises(ValueError):M.build_plan(original,data,builder)

    def test_frozen_scope_rejects_short_budget_and_dqn(self):
        with tempfile.TemporaryDirectory() as root:
            base=self.plan();base.update(schema=M.SCHEMA,status='frozen',repo_root=str(M.ROOT),
                output_root=root,run_root=str(Path(root)/'run'),baseline_source_commit=M.BASELINE_COMMIT)
            M.validate_plan(base)
            for field,value in (('training_budget',64),('primary_checkpoint',480),('epochs',True),('methods',M.METHODS+['dqn'])):
                candidate=copy.deepcopy(base);candidate[field]=value
                with self.subTest(field=field),self.assertRaises(ValueError):M.validate_plan(candidate)

    def test_tail_requires_fixed_full_panels_all_scopes_and_defined_cost(self):
        plan=self.plan();tail=plan['tail_stability']
        self.assertEqual(tail['episodes'],[360,432,504])
        self.assertTrue(set(tail['episodes'])<=set(M.FULL_POINTS))
        self.assertEqual(tail['scopes'],['overall','seed7','seed17','seed27'])
        self.assertEqual(tail['zero_completed_requests'],'undefined_cost_fail')
        self.assertEqual(tail['zero_mean'],'undefined_fail')
        with tempfile.TemporaryDirectory() as root:
            plan.update(schema=M.SCHEMA,status='frozen',repo_root=str(M.ROOT),output_root=root,
                run_root=str(Path(root)/'run'),baseline_source_commit=M.BASELINE_COMMIT)
            plan['tail_stability']=copy.deepcopy(tail);plan['tail_stability']['reward_relative_range']=.1
            with self.assertRaises(ValueError):M.validate_plan(plan)


class ScheduleExecutionTests(unittest.TestCase):
    def test_full504_worker_orchestration_never_selects_peak_or_repeats_eval_point(self):
        original,data=metadata();plan=M.build_plan(original,data,original_balancer())
        state=dict(sessions=[],updates=[],points=[],training_rows=[],update_rows=[])
        def session(case):
            state['sessions'].append(case);return {'fake_summary':case},[case]
        def update(buffer):
            self.assertEqual(len(buffer),4);state['updates'].append(list(buffer));return {'fake_step':len(state['updates'])}
        def evaluate(episode):
            self.assertEqual(len(state['updates']),episode//4)
            state['points'].append(episode);return {'episode':episode}
        def training_row(episode,case,row,loss):
            state['training_rows'].append(episode)
            self.assertEqual(loss is not None,episode%4==0)
        checkpoints=M.scheduled_cell(plan['training_schedule']['7'],session,update,evaluate,training_row,
            lambda episode,loss:state['update_rows'].append(episode),lambda *a,**k:None)
        self.assertEqual(state['sessions'],plan['training_schedule']['7'])
        self.assertEqual(len(state['updates']),126)
        self.assertEqual(len(state['training_rows']),504)
        self.assertEqual(state['points'],M.PROBE_POINTS)
        self.assertEqual(state['update_rows'],list(range(4,505,4)))
        self.assertEqual(checkpoints[-1],{'episode':504})

    def test_failure_stops_without_replaying_or_adding_budget(self):
        calls=[]
        def session(case):
            calls.append(case)
            if len(calls)==5:raise RuntimeError('fixture failure')
            return {},[case]
        with self.assertRaises(RuntimeError):
            M.scheduled_cell(['x']*504,session,lambda rows:{},lambda ep:ep,
                lambda *a:None,lambda *a:None,lambda *a,**k:None)
        self.assertEqual(len(calls),5)


class DecisionTests(unittest.TestCase):
    def fixture(self,root):
        root=Path(root);protocol=root/'manifest.json';M.write_json(protocol,{'frozen':'protocol'})
        diagnostic=root/'diagnostic.json';M.write_json(diagnostic,{'status':'complete_exact_observer_parity'})
        review=root/'independent_review.md';review.write_text('Independent source and credit recovery.\n')
        plan={'protocol_id':'protocol_fixture','source_commit':'a'*40}
        receipt=dict(schema=M.DECISION_SCHEMA,execution_authorized=True,decision='authorize_frozen_long_run',
            reviewer_role='root_independent_diagnostic_recovery',protocol_sha256=M.sha_file(protocol),
            protocol_id=plan['protocol_id'],source_commit=plan['source_commit'],reviewed_at='2026-10-22',diagnostic_run_id='fixture',
            diagnostic_result={'path':str(diagnostic),'sha256':M.sha_file(diagnostic)},
            independent_review={'path':str(review),'sha256':M.sha_file(review)},
            intervention_type='bounded_exposure_budget_test',root_cause_status='UNIDENTIFIED',
            rationale='Fixture authorizes exposure test without claiming a scientific repair.',algorithm_stability_fix_verified=False,
            validation_evidence=[{'path':str(review),'sha256':M.sha_file(review)}])
        path=root/'decision.json';M.write_json(path,receipt)
        return protocol,plan,path,receipt

    def test_independent_decision_binds_protocol_source_result_and_review(self):
        with tempfile.TemporaryDirectory() as root:
            protocol,plan,path,receipt=self.fixture(root)
            spec,_=M.check_decision(path,protocol,plan);self.assertEqual(spec['sha256'],M.sha_file(path))
            Path(receipt['independent_review']['path']).write_text('Changed')
            with self.assertRaises(ValueError):M.check_decision(path,protocol,plan)

    def test_denied_bool_substitution_or_wrong_protocol_never_authorizes(self):
        for field,value in (('execution_authorized',False),('execution_authorized',1),
                            ('protocol_sha256','b'*64),('reviewer_role','runner_self_review')):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as root:
                protocol,plan,path,receipt=self.fixture(root);receipt[field]=value;path.write_text(json.dumps(receipt))
                with self.assertRaises(ValueError):M.check_decision(path,protocol,plan)

    def test_missing_decision_rejected_before_run_claim(self):
        with tempfile.TemporaryDirectory() as root:
            plan={'run_root':str(Path(root)/'run')};args=types.SimpleNamespace(protocol='mock',decision_receipt=Path(root)/'missing.json')
            with patch.object(M,'check_protocol',return_value=(plan,Path(root)/'manifest.json',{})):
                with self.assertRaises(FileNotFoundError):M.execute(args)
            self.assertFalse(Path(plan['run_root']).exists())

    def test_incomplete_diagnostic_result_is_not_success(self):
        with tempfile.TemporaryDirectory() as root:
            protocol,plan,path,receipt=self.fixture(root)
            diagnostic=Path(receipt['diagnostic_result']['path']);diagnostic.write_text(json.dumps({'status':'failed'}))
            receipt['diagnostic_result']['sha256']=M.sha_file(diagnostic);path.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):M.check_decision(path,protocol,plan)

    def test_parity_alone_does_not_supply_independent_root_cause_or_validation(self):
        for field,value in (('validation_evidence',[]),('rationale',''),('root_cause_status',''),
            ('algorithm_stability_fix_verified',True),('intervention_type','algorithm_fix')):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as root:
                protocol,plan,path,receipt=self.fixture(root);receipt[field]=value;path.write_text(json.dumps(receipt))
                with self.assertRaises(ValueError):M.check_decision(path,protocol,plan)


class CompositeDecisionTests(DecisionTests):
    """Pure on-disk receipts and opaque byte fixtures; no tensor/model imports."""
    def composite_fixture(self, directory):
        root=Path(directory).resolve();protocol,plan,path,receipt=self.fixture(root)
        old_root=root/'failed';old_root.mkdir();new_root=root/'remaining';new_root.mkdir()
        recovered=root/'recovered';recovered.mkdir()
        scientific=root/'science';(scientific/'src').mkdir(parents=True)
        source=scientific/'src/example.py';source.write_text('# opaque frozen scientific source\n')
        fingerprints={'src/example.py':M.sha_file(source)}
        def put(where,value):
            where=Path(where);where.parent.mkdir(parents=True,exist_ok=True);M.write_json(where,value);return where
        def spec(where):return dict(path=str(Path(where)),sha256=M.sha_file(where))
        def binding(where):return dict(spec(where),size_bytes=Path(where).stat().st_size)
        cases=['train_%03d'%i for i in range(4)]
        original_protocol=put(root/'original_protocol.json',dict(training_schedule={str(seed):cases*16 for seed in M.SEEDS}))
        plan.update(source_protocol=spec(original_protocol),source_fingerprints=fingerprints)
        old=dict(schema='stream_stability_diagnostic_v1',status='frozen',seeds=M.SEEDS,sides=['off','on'],
            training_budget=64,sessions_per_seed=4,epochs=4,output_root=str(old_root),
            repo_root=str(scientific),source_commit='a'*40,source_fingerprints=fingerprints,protocol=spec(original_protocol))
        old_path=put(root/'original_contract.json',old)
        side_counts=dict(training_sessions=4,training_updates=1,training_frames=20,environments=4,
            adam_calls=2,backward_calls=1,categorical_sample_calls=8,evaluate_calls=10)
        fail_counts={key:value*2 for key,value in side_counts.items()}
        failure=put(old_root/'failure.json',dict(status='failed',no_retry=True,completed_seeds=0,
            context=dict(seed=7,side='on',phase='independent_consumer'),counts=fail_counts))
        put(old_root/'run_started.json',dict(contract_sha256=M.sha_file(old_path),source_commit=old['source_commit']))
        audits={};all_sides={}
        for seed in M.SEEDS:
            home=(old_root if seed==7 else new_root)/('seed%d'%seed);sides=[]
            audit=dict(status='complete_unverified',structural_complete=True,
                actual_training_verified=False,synthetic_fixture=False)
            audits[seed]=audit
            for side in ('off','on'):
                folder=home/side;folder.mkdir(parents=True)
                for name in ('checkpoint.pt','final_evidence.pt'):
                    (folder/name).write_bytes(('%s_%d'%(name,seed)).encode())
                for episode,case in enumerate(cases,1):
                    put(folder/('episode%d.json'%episode),dict(episode=episode,case_id=case,
                        summary_sha256='summary'+str(episode),rolls_sha256='rolls'+str(episode),
                        receipts_sha256='receipts'+str(episode),rng_after_sha256='rng'+str(episode)))
                row=dict(seed=seed,side=side,episode_count=4,case_ids=cases,
                    original_call_counts=side_counts.copy(),coverage={},initial_learned_state_sha256='initial',
                    checkpoint_sha256=M.sha_file(folder/'checkpoint.pt'),
                    final_evidence_sha256=M.sha_file(folder/'final_evidence.pt'))
                if side=='on':row['audit']=audit
                if seed!=7 or side=='off':put(folder/'result.json',row)
                sides.append(row)
            audit_path=home/'on/audit/events.jsonl';audit_path.parent.mkdir()
            audit_path.write_text('opaque raw audit seed%d\n'%seed)
            all_sides[seed]=sides
        input_bindings={str(p):binding(p) for p in [old_path,failure,original_protocol,*old_root.rglob('*')] if p.is_file()}
        recovery=dict(schema='stream_seed7_partial_recovery_v1',status='frozen',seed=7,recovered_seeds=[7],
            not_run_seeds=[17,27],counts=dict(training_sessions=0,training_updates=0,models=0,environments=0),
            output_root=str(recovered),original_source_commit=old['source_commit'],
            input_bindings=input_bindings,recovery_sources={'pure_schema':binding(source)})
        recovery_path=put(root/'recovery_contract.json',recovery)
        partial=dict(status='RECOVERED_SEED7_ONLY_OBSERVER_PARITY',original_run_status='failed',original_run_failed=True,
            full_matrix_incomplete=True,recovered_seeds=[7],not_run_seeds=[17,27],
            counts=dict(training_sessions=0,training_updates=0),saved_final_files_byte_equal=True,
            four_episode_exact_digest_parity=True,actual_training_verified=False,formal_holdout='not_read',
            recovery_contract_sha256=M.sha_file(recovery_path),qualification_root=str(old_root),
            original_run_counts=fail_counts,original_source_commit=old['source_commit'],
            input_bindings=input_bindings,recovery_sources=recovery['recovery_sources'],compact_schema_validation=audits[7])
        partial_path=put(recovered/'result.json',partial)
        limits=dict(training_sessions=16,training_updates=4,models=4,environments=16)
        remaining_counts={key:value*4 for key,value in side_counts.items()};remaining_counts['models']=4
        profile=dict(schema='unexecuted_stream_diagnostic_v1',status='frozen',seeds=[17,27],sides=['off','on'],
            training_budget=64,sessions_per_seed=4,epochs=4,compact_joins=True,no_resume=True,no_retry=True,
            already_executed=dict(seeds=[7],training_sessions=8,training_updates=2),
            original_scientific_budget=dict(training_sessions=24,training_updates=6),limits=limits,
            parent_contract=spec(old_path),parent_failure=spec(failure),partial_seed7_recovery=spec(partial_path),
            output_root=str(new_root),repo_root=str(scientific),source_commit='b'*40,
            source_fingerprints=fingerprints,protocol=spec(original_protocol))
        profile_path=put(root/'remaining_contract.json',profile)
        put(new_root/'run_started.json',dict(contract_sha256=M.sha_file(profile_path),source_commit=profile['source_commit']))
        remaining=dict(schema=profile['schema'],status='complete_unexecuted_seed_observer_parity',
            executed_seeds=[17,27],original_run_status='failed',original_root_completed=False,formal_holdout='not_read',
            no_resume=True,no_retry=True,source_commit=profile['source_commit'],contract_sha256=M.sha_file(profile_path),
            parent_contract=profile['parent_contract'],parent_failure=profile['parent_failure'],
            partial_seed7_recovery=profile['partial_seed7_recovery'],limits=limits,counts=remaining_counts,
            seeds=[dict(seed=seed,status='exact_parity',sides=all_sides[seed]) for seed in (17,27)])
        remaining_path=put(new_root/'result.json',remaining)
        diagnostic=dict(schema=M.COMPOSITE_SCHEMA,status=M.COMPOSITE_STATUS,
            counts=dict(training_sessions=0,training_updates=0,models=0,environments=0),
            original_run_status='failed',original_root_completed=False,original_run_failed=True,
            original_matrix_remains_incomplete=True,composite_observer_matrix_complete=True,
            recovered_seed7_scientific_counts=dict(training_sessions=0,training_updates=0),
            previously_executed_counts=dict(training_sessions=8,training_updates=2),
            remaining_executed_counts=dict(training_sessions=16,training_updates=4),
            combined_observed_counts=dict(training_sessions=24,training_updates=6),observed_seeds=M.SEEDS,
            actual_training_verified=False,formal_holdout='not_read',convergence='UNVERIFIED',
            causal_root_cause='UNIDENTIFIED_by_descriptive_statistics',algorithm_fix='NOT_CLAIMED',
            scientific_sources={str(seed):old['source_commit'] if seed==7 else profile['source_commit'] for seed in M.SEEDS},
            seeds=[dict(seed=seed,independent_schema_validation=audits[seed],counts=dict(updates=1),
                scientific_source_commit=old['source_commit'] if seed==7 else profile['source_commit'],
                file_sha256=M.sha_file((old_root if seed==7 else new_root)/('seed%d/on/audit/events.jsonl'%seed)),
                bytes=((old_root if seed==7 else new_root)/('seed%d/on/audit/events.jsonl'%seed)).stat().st_size) for seed in M.SEEDS])
        diagnostic['input_bindings']={str(p):binding(p) for p in root.rglob('*') if p.is_file() and p!=path}
        diagnostic_path=root/'composite.json';put(diagnostic_path,diagnostic)
        receipt['diagnostic_result']=spec(diagnostic_path)
        receipt['diagnostic_components']={name:spec(p) for name,p in dict(original_contract=old_path,
            original_failure=failure,seed7_recovery_contract=recovery_path,seed7_recovery_result=partial_path,
            remaining_contract=profile_path,remaining_result=remaining_path).items()}
        receipt.update(decision_origin='root_conditional_authorization_after_complete_diagnostic',
            conditional_authorization=dict(path=str(root/'root_authorization.json'),sha256='c'*64))
        path.write_text(json.dumps(receipt))
        return protocol,plan,path,receipt,diagnostic,old_root,new_root

    def save_summary(self,path,receipt,diagnostic):
        where=Path(receipt['diagnostic_result']['path']);where.write_text(json.dumps(diagnostic))
        receipt['diagnostic_result']['sha256']=M.sha_file(where);path.write_text(json.dumps(receipt))

    def test_complete_composite_binds_failed_original_components_and_exact_files(self):
        with tempfile.TemporaryDirectory() as root:
            protocol,plan,path,receipt,diagnostic,old_root,new_root=self.composite_fixture(root)
            M.check_decision(path,protocol,plan)
            self.assertFalse((old_root/'result.json').exists())

    def test_status_alone_or_wrong_scope_never_authorizes(self):
        changes=[('observed_seeds',[7,17]),('combined_observed_counts',dict(training_sessions=24,training_updates=7)),
            ('original_run_status','complete'),('original_root_completed',True),
            ('composite_observer_matrix_complete',False),('actual_training_verified',True),
            ('algorithm_fix','VERIFIED'),('input_bindings',{})]
        for field,value in changes:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as root:
                protocol,plan,path,receipt,diagnostic,*_=self.composite_fixture(root)
                diagnostic[field]=value;self.save_summary(path,receipt,diagnostic)
                with self.assertRaises(ValueError):M.check_decision(path,protocol,plan)

    def test_component_missing_mismatch_raw_tamper_and_backfilled_failure_rejected(self):
        for mode in ('missing','sha','raw','backfill','remaining_failure','source','duplicate_seed','audit'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as root:
                protocol,plan,path,receipt,diagnostic,old_root,new_root=self.composite_fixture(root)
                if mode=='missing':receipt['diagnostic_components'].pop('original_failure')
                elif mode=='sha':receipt['diagnostic_components']['remaining_contract']['sha256']='d'*64
                elif mode=='raw':(old_root/'seed7/on/checkpoint.pt').write_bytes(b'tampered')
                elif mode=='backfill':(old_root/'result.json').write_text('{}')
                elif mode=='remaining_failure':(new_root/'failure.json').write_text('{}')
                elif mode=='source':plan['source_fingerprints']['src/example.py']='e'*64
                elif mode=='duplicate_seed':diagnostic['seeds'][2]['seed']=17
                else:diagnostic['seeds'][1]['independent_schema_validation']['structural_complete']=False
                self.save_summary(path,receipt,diagnostic)
                with self.assertRaises(ValueError):M.check_decision(path,protocol,plan)

    def test_composite_cannot_be_relabelled_as_identified_stability_fix(self):
        for field,value in (('root_cause_status','IDENTIFIED'),('intervention_type','verified_nonalgorithmic_protocol_fix'),
                ('algorithm_stability_fix_verified',True)):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as root:
                protocol,plan,path,receipt,diagnostic,*_=self.composite_fixture(root)
                receipt[field]=value;receipt['nonalgorithmic_fix_scope']='statistical_reporting'
                self.save_summary(path,receipt,diagnostic)
                with self.assertRaises(ValueError):M.check_decision(path,protocol,plan)


class ResourceAndSinkTests(unittest.TestCase):
    def budget(self,root):return M.Budget(root,rss=lambda:1,disk=lambda _:types.SimpleNamespace(free=40*M.GIB))

    def test_checkpoint_reserves_full_bound_before_save_and_never_refunds(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'checkpoint.pt';calls=[]
            serializer=types.SimpleNamespace(save=lambda value,target:target.write(value))
            original_save=serializer.save
            def save(target):calls.append(target);serializer.save(b'abc',target)
            agent=types.SimpleNamespace(save_checkpoint=save)
            budget=self.budget(root);budget.counts['output_bytes']=M.LIMITS['output_bytes']-1
            with self.assertRaises(RuntimeError):M.save_checkpoint_bounded(agent,path,budget,serializer)
            self.assertEqual(calls,[]);self.assertFalse(path.exists())
            budget=self.budget(root)
            self.assertEqual(M.save_checkpoint_bounded(agent,path,budget,serializer),dict(actual_bytes=3,write_bytes=3,save_calls=1))
            self.assertEqual(calls,[path]);self.assertEqual(budget.counts['output_bytes'],M.LIMITS['checkpoint_bytes'])
            self.assertEqual(budget.counts['checkpoints'],1)
            self.assertIs(serializer.save,original_save)
            path.unlink();budget=self.budget(root)
            with patch.dict(M.LIMITS,checkpoint_bytes=2):
                with self.assertRaisesRegex(RuntimeError,'Checkpoint byte cap'):M.save_checkpoint_bounded(agent,path,budget,serializer)
            self.assertEqual(budget.counts['output_bytes'],2)
            self.assertEqual(path.read_bytes(),b'');self.assertIs(serializer.save,original_save)

    def test_checkpoint_rewrites_share_one_reservation_and_other_paths_are_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'checkpoint.pt';other=Path(root)/'other.pt';calls=[]
            def original_save(value,target):
                calls.append(target)
                if isinstance(target,Path):target.write_bytes(value)
                else:target.write(value)
            serializer=types.SimpleNamespace(save=original_save)
            def save(target):
                serializer.save(b'abc',target);serializer.save(b'ab',target);serializer.save(b'not bounded',other)
            agent=types.SimpleNamespace(save_checkpoint=save);budget=self.budget(root)
            with patch.dict(M.LIMITS,checkpoint_bytes=5):
                self.assertEqual(M.save_checkpoint_bounded(agent,path,budget,serializer),dict(actual_bytes=2,write_bytes=5,save_calls=2))
            self.assertEqual(budget.counts['output_bytes'],5);self.assertEqual(other.read_bytes(),b'not bounded')
            self.assertIs(calls[-1],other);self.assertIs(serializer.save,original_save)
            path.unlink();budget=self.budget(root)
            with patch.dict(M.LIMITS,checkpoint_bytes=4):
                with self.assertRaisesRegex(RuntimeError,'Cumulative checkpoint'):M.save_checkpoint_bounded(agent,path,budget,serializer)
            self.assertEqual(budget.counts['output_bytes'],4);self.assertIs(serializer.save,original_save)

    def test_checkpoint_file_denies_direct_descriptor_writes(self):
        with tempfile.TemporaryFile() as handle:
            state=dict(write_bytes=0);bounded=M.BoundedCheckpointFile(handle,4,state)
            with self.assertRaises(M.io.UnsupportedOperation):bounded.fileno()
            bounded.write(b'abcd')
            with self.assertRaisesRegex(RuntimeError,'before write'):bounded.write(b'e')
            handle.seek(0);self.assertEqual(handle.read(),b'abcd');self.assertEqual(state['write_bytes'],4)

    def test_record_raw_file_and_global_caps_fail_before_original_write(self):
        for cap_kind in ('row','raw','file','global'):
            with self.subTest(cap_kind=cap_kind),tempfile.TemporaryDirectory() as root:
                budget=self.budget(root)
                if cap_kind=='global':budget.counts['output_bytes']=M.LIMITS['output_bytes']
                sink=M.BoundedText(Path(root)/'x.jsonl',budget,rows=0 if cap_kind=='row' else 1,
                    raw_bytes=1 if cap_kind=='raw' else 100,file_bytes=1 if cap_kind=='file' else 100)
                with self.assertRaises(RuntimeError):sink.row({'a':1})
                sink.close();self.assertEqual((Path(root)/'x.jsonl').read_bytes(),b'')
                self.assertEqual(sink.rows,0)

    def test_gzip_stream_roundtrip_and_exclusive_path(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'x.gz';budget=self.budget(root)
            with M.BoundedText(path,budget,rows=3,raw_bytes=1000,file_bytes=1000,gzip=True) as sink:
                sink.row({'a':1});sink.row({'a':2})
            self.assertEqual([json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()],[{'a':1},{'a':2}])
            with self.assertRaises(FileExistsError):M.BoundedText(path,budget,rows=3,raw_bytes=1000,file_bytes=1000)

    def test_operation_and_resource_caps_do_not_expand(self):
        with tempfile.TemporaryDirectory() as root:
            budget=self.budget(root);budget.counts['learner_updates']=2646
            with self.assertRaises(RuntimeError):budget.charge('learner_updates')
            self.assertEqual(budget.counts['learner_updates'],2646)
            with self.assertRaises(ValueError):budget.charge('learner_updates',True)
            for kwargs in ({'rss':lambda:8*M.GIB+1}, {'disk':lambda _:types.SimpleNamespace(free=30*M.GIB-1)}):
                bad=M.Budget(root,**kwargs)
                with self.assertRaises(RuntimeError):bad.check(disk='disk' in kwargs)

    def test_wall_cap_and_progress_have_no_reward_or_resume(self):
        with tempfile.TemporaryDirectory() as root:
            now=[0];budget=M.Budget(root,clock=lambda:now[0],rss=lambda:1)
            budget.progress('checkpoint_evaluation',method='sa_ghmappo',seed=7,episode=0)
            value=json.loads((Path(root)/'progress.json').read_text())
            self.assertNotIn('reward',json.dumps(value))
            now[0]=M.LIMITS['max_wall_seconds']+1
            with self.assertRaises(RuntimeError):budget.check()

    def test_all_three_rng_domains_restore_after_evaluation_exception(self):
        class State:
            def __init__(self,value):self.value=value
            def clone(self):return State(self.value)
        class Torch:
            value=13
            def get_rng_state(self):return State(self.value)
            def set_rng_state(self,state):self.value=state.value
        class NumPy:
            value=17
            class Random:
                def __init__(self,outer):self.outer=outer
                def get_state(self):return self.outer.value
                def set_state(self,value):self.outer.value=value
            def __init__(self):self.random=self.Random(self)
        torch=Torch();np=NumPy();before=random.getstate()
        with self.assertRaises(RuntimeError):
            with M.preserve_rng(torch,np):
                random.random();torch.value=1;np.value=2;raise RuntimeError('mock evaluation failure')
        self.assertEqual(random.getstate(),before);self.assertEqual(torch.value,13);self.assertEqual(np.value,17)


if __name__=='__main__':unittest.main()
