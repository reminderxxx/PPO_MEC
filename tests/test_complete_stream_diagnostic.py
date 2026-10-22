"""Pure/mock tests: never construct models, assets, environments or optimizers."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'scripts/complete_stream_diagnostic.py'
SPEC=importlib.util.spec_from_file_location('complete_stream_pure',PATH)
M=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=M;SPEC.loader.exec_module(M)


class Fixture(unittest.TestCase):
    def fixture(self, root):
        root=Path(root).resolve(); repo=root/'new_repo'; old_repo=root/'old_repo';old_root=root/'failed';home=root/'assets'
        repo.mkdir();old_repo.mkdir();old_root.mkdir();home.mkdir()
        source={}
        for name in (*M.REQUIRED_SCRIPTS,'src/agents/dummy.py'):
            f=repo/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('# bound\n');source[name]=M.sha_file(f)
        old_source={}
        for name in (*M.original.REQUIRED_SCRIPTS,'src/agents/dummy.py'):
            f=old_repo/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('# original\n');old_source[name]=M.sha_file(f)
        for name in ['protocol.pt','predictor.pt']:(home/name).write_bytes(b'opaque never loaded')
        schedules={str(seed):[f'train_{i:03d}' for i in range(64)] for seed in [7,17,27]}
        protocol=dict(protocol='joint_stream_learning_v2',action_contract='joint_stream_control_v2',
            smoke=False,episodes=64,seeds=[7,17,27],training_schedule=schedules,
            assets={name:M.sha_file(home/name) for name in ['protocol.pt','predictor.pt']})
        M.write_json(home/'manifest.json',protocol)
        spec=dict(path=str(home/'manifest.json'),sha256=M.sha_file(home/'manifest.json'))
        old=dict(schema=M.original.SCHEMA,status='frozen',seeds=[7,17,27],sides=['off','on'],
            sessions_per_seed=4,training_budget=64,epochs=4,limits=copy.deepcopy(M.original.FIXED_LIMITS),
            baseline_source_commit=M.original.BASELINE_SOURCE_COMMIT,protocol=spec,
            output_root=str(old_root),repo_root=str(old_repo),source_commit='b'*40,
            source_fingerprints=old_source)
        M.write_json(root/'old_contract.json',old)
        failure=dict(schema=M.original.SCHEMA,status='failed',no_retry=True,completed_seeds=0,
            counts=dict(training_sessions=8,training_updates=2,models=2,environments=8),
            context=dict(seed=7,side='on',phase='independent_consumer'))
        M.write_json(old_root/'failure.json',failure)
        M.write_json(old_root/'run_started.json',dict(contract_sha256=M.sha_file(root/'old_contract.json'),source_commit='b'*40))
        M.write_json(old_root/'progress.json',dict(schema=M.original.SCHEMA,counts=failure['counts'],context=failure['context']))
        for side in ['off','on']:
            h=old_root/'seed7'/side;h.mkdir(parents=True)
            for ep in range(1,5):M.write_json(h/f'episode{ep}.json',dict(episode=ep,case_id=schedules['7'][ep-1]))
            for name in ['checkpoint.pt','final_evidence.pt']:(h/name).write_bytes(b'opaque matched evidence')
        audit=old_root/'seed7/on/audit';audit.mkdir();(audit/'events.jsonl').write_text('opaque fixture audit never evaluated\n')
        M.write_json(old_root/'seed7/off/result.json',dict(seed=7,side='off',episode_count=4,
            case_ids=schedules['7'][:4],original_call_counts=dict(training_sessions=4,training_updates=1)))
        recovery=root/'recovery_seed7';recovery.mkdir()
        recovered=dict(status=M.PARTIAL_STATUS,original_run_status='failed',recovered_seeds=[7],not_run_seeds=[17,27],
            counts=dict(training_sessions=0,training_updates=0),
            qualification_root=str(old_root),original_source_commit='b'*40,
            saved_final_files_byte_equal=True,four_episode_exact_digest_parity=True,
            input_bindings={name:dict(path=str(p),sha256=M.sha_file(p),size_bytes=Path(p).stat().st_size)
                for name,p in M.recovery_input_paths(old_root,root/'old_contract.json',spec).items()})
        M.write_json(recovery/'result.json',recovered)
        bind=lambda p:dict(path=str(p.resolve()),sha256=M.sha_file(p))
        profile=dict(schema=M.SCHEMA,status='frozen',seeds=[17,27],sides=['off','on'],sessions_per_seed=4,
            training_budget=64,epochs=4,limits=copy.deepcopy(M.LIMITS),repo_root=str(repo),output_root=str(root/'remaining'),
            baseline_source_commit=M.original.BASELINE_SOURCE_COMMIT,source_commit='a'*40,source_fingerprints=source,
            original_scientific_budget=dict(training_sessions=24,training_updates=6),
            already_executed=dict(seeds=[7],training_sessions=8,training_updates=2),compact_joins=True,no_resume=True,no_retry=True,
            protocol=spec,parent_contract=bind(root/'old_contract.json'),parent_failure=bind(old_root/'failure.json'),
            partial_seed7_recovery=bind(recovery/'result.json'),
            parent_bindings={name:M.sha_file(name) for name in M.parent_required_files(old_root)})
        M.write_json(root/'profile.json',profile)
        return profile,protocol

    def checked(self, root, *, require_unclaimed=True):
        root=Path(root)
        with patch.object(M,'ROOT',(root/'new_repo').resolve()), \
             patch.object(M.original,'git',side_effect=lambda _root,*args:('b'*40 if Path(_root).name=='old_repo' else 'a'*40) if args==('rev-parse','HEAD') else ''), \
             patch.object(M.original.subprocess,'check_call'):
            return M.check_contract(root/'profile.json',require_unclaimed=require_unclaimed)

    def save(self, root, profile):
        (Path(root)/'profile.json').write_text(json.dumps(profile))


class ProfileTests(Fixture):
    def test_check_is_standard_library_only_and_never_deserializes_opaque_assets(self):
        names=[]
        for node in ast.parse(PATH.read_text()).body:
            if isinstance(node,ast.Import):names.extend(v.name for v in node.names)
            if isinstance(node,ast.ImportFrom):names.append(node.module or '')
        self.assertFalse(any(v.split('.')[0] in ['torch','numpy','src'] for v in names))
        with tempfile.TemporaryDirectory() as root:
            profile,_=self.fixture(root);self.checked(root)
            self.assertFalse(Path(profile['output_root']).exists())

    def test_remaining_scope_is_exactly_16sessions_4updates_and_never_seed7(self):
        with tempfile.TemporaryDirectory() as root:
            profile,_=self.fixture(root)
            for key,value in [('seeds',[7,17,27]),('training_budget',504),('epochs',True),
                ('sessions_per_seed',4.),('compact_joins',False),('no_resume',False),
                ('limits',dict(M.LIMITS,training_sessions=24))]:
                p=copy.deepcopy(profile);p[key]=value;self.save(root,p)
                with self.subTest(key=key),self.assertRaises(ValueError):self.checked(root)

    def test_sorted_json_roundtrip_preserves_contract_but_types_remain_exact(self):
        with tempfile.TemporaryDirectory() as root:
            p,_=self.fixture(root)
            # Fixture uses the real sorted writer for both old/new contracts.
            actual,_,_,_=self.checked(root)
            self.assertTrue(M.same_contract(actual,p))
            for left,right in [(True,1),(0,False),(4.,4),([17,27],[27,17])]:
                self.assertFalse(M.same_contract(left,right))
            self.assertTrue(M.same_contract({'b':2,'a':1},{'a':1,'b':2}))

    def test_partial_receipt_cannot_bind_other_original_or_omit_audit(self):
        for kind in ['audit_missing','different_failure','wrong_root','wrong_source','nonzero_operation']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                p,_=self.fixture(root);f=Path(p['partial_seed7_recovery']['path']);v=M.read_json(f)
                if kind=='audit_missing':v['input_bindings'].pop('audit')
                elif kind=='different_failure':v['input_bindings']['failure']=v['input_bindings']['progress']
                elif kind=='wrong_root':v['qualification_root']=str(Path(root)/'different')
                elif kind=='wrong_source':v['original_source_commit']='c'*40
                else:v['counts']['forward_calls']=1
                f.write_text(json.dumps(v));p['partial_seed7_recovery']['sha256']=M.sha_file(f);self.save(root,p)
                with self.assertRaises(ValueError):self.checked(root)

    def test_original_failure_must_be_exactly_seed7_consumer_after8_and2(self):
        for key,value in [('status','complete_exact_observer_parity'),('completed_seeds',1),
            ('counts',dict(training_sessions=16,training_updates=4,models=4,environments=16)),
            ('context',dict(seed=17,side='on',phase='independent_consumer'))]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as root:
                p,_=self.fixture(root);f=Path(p['parent_failure']['path']);v=M.read_json(f);v[key]=value
                f.write_text(json.dumps(v));p['parent_failure']['sha256']=M.sha_file(f);self.save(root,p)
                with self.assertRaises(ValueError):self.checked(root)

    def test_existing_parent_seed17_or27_directory_is_not_run_eligibility_failure(self):
        for seed in [17,27]:
            with self.subTest(seed=seed),tempfile.TemporaryDirectory() as root:
                self.fixture(root);(Path(root)/'failed'/f'seed{seed}').mkdir()
                with self.assertRaises(ValueError):self.checked(root)

    def test_parent_success_backfill_is_rejected(self):
        for relative in ['result.json','seed7/result.json','seed7/on/result.json']:
            with self.subTest(relative=relative),tempfile.TemporaryDirectory() as root:
                self.fixture(root);M.write_json(Path(root)/'failed'/relative,{'status':'complete'})
                with self.assertRaises(ValueError):self.checked(root)

    def test_partial_receipt_requires_seed7_only_zero_new_training(self):
        for key,value in [('status','complete_exact_observer_parity'),('recovered_seeds',[7,17]),
            ('not_run_seeds',[27]),('counts',dict(training_sessions=8,training_updates=2)),
            ('original_run_status','complete')]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as root:
                p,_=self.fixture(root);f=Path(p['partial_seed7_recovery']['path']);v=M.read_json(f);v[key]=value
                f.write_text(json.dumps(v));p['partial_seed7_recovery']['sha256']=M.sha_file(f);self.save(root,p)
                with self.assertRaises(ValueError):self.checked(root)

    def test_source_asset_receipt_and_parent_case_changes_rejected(self):
        for kind in ['source','old_source','asset','episode','failure','recovery']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                p,_=self.fixture(root)
                f={'source':Path(root)/'new_repo/src/agents/dummy.py',
                    'old_source':Path(root)/'old_repo/src/agents/dummy.py','asset':Path(root)/'assets/protocol.pt',
                    'episode':Path(root)/'failed/seed7/on/episode4.json','failure':Path(p['parent_failure']['path']),
                    'recovery':Path(p['partial_seed7_recovery']['path'])}[kind]
                f.write_bytes(b'changed')
                with self.assertRaises(ValueError):self.checked(root)

    def test_missing_eighth_parent_receipt_and_path_traversal_rejected(self):
        for kind in ['missing','traversal']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                p,_=self.fixture(root)
                if kind=='missing':p['parent_bindings'].pop(str((Path(root)/'failed/seed7/on/episode4.json').resolve()))
                else:p['source_fingerprints']['../outside.py']='0'*64
                self.save(root,p)
                with self.assertRaises(ValueError):self.checked(root)

    def test_new_claim_is_exclusive_and_cannot_be_inside_or_above_old_root(self):
        for kind in ['exists','inside','ancestor']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                p,_=self.fixture(root)
                if kind=='exists':Path(p['output_root']).mkdir()
                elif kind=='inside':p['output_root']=str(Path(root)/'failed'/'child')
                else:p['output_root']=str(Path(root))
                self.save(root,p)
                with self.assertRaises((ValueError,FileExistsError)):self.checked(root)

    def test_final_check_rehashes_sources_but_does_not_allow_a_new_claim(self):
        with tempfile.TemporaryDirectory() as root:
            p,_=self.fixture(root);Path(p['output_root']).mkdir()
            self.checked(root,require_unclaimed=False)
            with self.assertRaises(FileExistsError):self.checked(root)

    def test_missing_partial_receipt_fails_before_output_claim(self):
        with tempfile.TemporaryDirectory() as root:
            p,_=self.fixture(root);Path(p['partial_seed7_recovery']['path']).unlink()
            with patch.object(M,'ROOT',(Path(root)/'new_repo').resolve()), \
                 patch.object(M.original,'git',side_effect=lambda _root,*args:('b'*40 if Path(_root).name=='old_repo' else 'a'*40) if args==('rev-parse','HEAD') else ''), \
                 patch.object(M.original.subprocess,'check_call'):
                with self.assertRaises(FileNotFoundError):M.execute(Path(root)/'profile.json')
            self.assertFalse(Path(p['output_root']).exists())

    def test_preclaim_contract_race_cannot_claim_initial_or_replacement_root(self):
        with tempfile.TemporaryDirectory() as root:
            p,protocol=self.fixture(root);p2=copy.deepcopy(p);p2['output_root']=str(Path(root)/'other')
            b=types.SimpleNamespace(check=lambda **kwargs:None)
            with patch.object(M,'ROOT',(Path(root)/'new_repo').resolve()),patch.object(M,'Resources',return_value=b), \
                 patch.object(M,'check_contract',return_value=(p2,protocol,Path(root)/'profile.json',Path(root)/'assets/manifest.json')):
                with self.assertRaisesRegex(RuntimeError,'preclaim'):M.execute(Path(root)/'profile.json')
            self.assertFalse(Path(p['output_root']).exists());self.assertFalse(Path(p2['output_root']).exists())


class ResourceTests(unittest.TestCase):
    def budget(self,root,**kwargs):
        return M.Resources(root,rss=kwargs.get('rss',lambda:1),clock=kwargs.get('clock',lambda:0),
            disk=lambda _:types.SimpleNamespace(free=40*M.original.GIB))

    def test_new_caps_do_not_modify_original_limits(self):
        before=copy.deepcopy(M.original.FIXED_LIMITS)
        with tempfile.TemporaryDirectory() as root:
            b=self.budget(root);b.counts['training_sessions']=16
            with self.assertRaises(RuntimeError):b.charge('training_sessions')
            self.assertEqual(b.counts['training_sessions'],16)
            with self.assertRaises(ValueError):b.charge('training_updates',True)
        self.assertEqual(M.original.FIXED_LIMITS,before);self.assertEqual(before['training_sessions'],24)

    def test_hash_guard_checks_before_and_between_megabyte_chunks(self):
        with tempfile.TemporaryDirectory() as root:
            f=Path(root)/'opaque';f.write_bytes(b'x'*(2*1048576+1));calls=[]
            guard=types.SimpleNamespace(check=lambda:calls.append('check'))
            self.assertEqual(M.sha_file(f,guard),M.original.sha_file(f))
            self.assertGreaterEqual(len(calls),5)
            def stop():raise RuntimeError('wall before hash')
            with self.assertRaises(RuntimeError):M.sha_file(f,types.SimpleNamespace(check=stop))

    def test_480backward_960Adam_4models_7680frames_hard_caps(self):
        with tempfile.TemporaryDirectory() as root:
            for key,value in [('backward_calls',480),('adam_calls',960),('models',4),('training_frames',7680),
                ('categorical_sample_calls',184320),('evaluate_calls',123360),('training_updates',4)]:
                b=self.budget(root);b.counts[key]=value
                with self.subTest(key=key),self.assertRaises(RuntimeError):b.charge(key)

    def test_rss_wall_and_progress_use_new_profile_identity(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError):self.budget(root,rss=lambda:8*M.original.GIB+1).check()
            now=[0];b=self.budget(root,clock=lambda:now[0]);b.progress('rollout',seed=17,side='off')
            self.assertEqual(M.read_json(Path(root)/'progress.json')['schema'],M.SCHEMA)
            now[0]=7201
            with self.assertRaises(RuntimeError):b.check()

    def test_execution_source_explicitly_uses_compact_api_without_old_constant_patch(self):
        source=PATH.read_text()
        self.assertEqual(source.count('compact_joins=True'),3)  # expected shape + producer + consumer
        self.assertNotIn('original.FIXED_LIMITS =',source)
        self.assertIn("status=SUCCESS",source)
        self.assertEqual(M.SUCCESS,'complete_unexecuted_seed_observer_parity')
        self.assertNotEqual(M.SUCCESS,'complete_exact_observer_parity')


if __name__=='__main__':unittest.main()
