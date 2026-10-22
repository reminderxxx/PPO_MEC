"""Pure-software safety and parity-hash tests; no model/environment/optimizer."""
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

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'scripts/diagnose_stream_stability.py'
SPEC = importlib.util.spec_from_file_location('stream_stability_pure_helpers', PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class DigestTests(unittest.TestCase):
    def test_top_level_has_no_numerical_or_agent_imports(self):
        names = []
        for node in ast.parse(PATH.read_text()).body:
            if isinstance(node, ast.Import):
                names.extend(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or '')
        self.assertFalse(any(name.split('.')[0] in ('torch', 'numpy', 'src', 'scripts') for name in names))

    def test_exact_types_bits_container_and_key_order(self):
        pairs = [(True, 1), (1, 1.), (0., -0.), ('a', b'a'), ([1], (1,)),
            ({'a': 1, 'b': 2}, {'b': 2, 'a': 1}), ([1.], [1. + 1e-12])]
        for left, right in pairs:
            with self.subTest(left=left, right=right):
                self.assertNotEqual(M.exact_digest(left), M.exact_digest(right))

    def test_stable_exact_digest_does_not_mutate_input(self):
        value = {'a': [1, 2., None], 'b': (b'bytes', True), 'c': {1, 2, 3}}
        before = copy.deepcopy(value)
        self.assertEqual(M.exact_digest(value), M.exact_digest(before))
        self.assertEqual(value, before)

    def test_only_observer_uid_field_is_optional(self):
        plain = {'roll': {'actions': [0, 1, 2], 'log_probs': [0., -.2, -.3]}}
        observed = copy.deepcopy(plain); observed['roll']['credit_audit_uids'] = ['uid']
        self.assertNotEqual(M.exact_digest(plain), M.exact_digest(observed))
        self.assertEqual(M.exact_digest(plain, exclude_audit_uids=True),
                         M.exact_digest(observed, exclude_audit_uids=True))
        observed['roll']['log_probs'][1] += 1e-12
        self.assertNotEqual(M.exact_digest(plain, exclude_audit_uids=True),
                            M.exact_digest(observed, exclude_audit_uids=True))

    def test_nonfinite_cycles_and_unknown_types_fail_closed(self):
        cycle = []; cycle.append(cycle)
        for value in (float('nan'), float('inf'), [float('-inf')], cycle, object()):
            with self.subTest(value=type(value).__name__), self.assertRaises((ValueError, TypeError)):
                M.exact_digest(value)

    def test_json_sorted_keys_do_not_create_false_episode_parity_failure(self):
        observed = dict(episode=1, case_id='train_000', summary={'z':1.,'a':2.},
            summary_sha256='s', rolls_sha256='r', receipts_sha256='c', rng_after_sha256='g')
        archived = json.loads(json.dumps(observed, sort_keys=True))
        self.assertNotEqual(list(observed), list(archived))
        self.assertNotEqual(list(observed['summary']), list(archived['summary']))
        def strict(left, right, where):
            if M.exact_digest(left) != M.exact_digest(right): raise AssertionError(where)
        M.compare_episode_receipt(observed, archived, strict)
        archived['rolls_sha256'] = 'tampered'
        with self.assertRaises(AssertionError): M.compare_episode_receipt(observed, archived, strict)


class ResourceTests(unittest.TestCase):
    def budget(self, root, *, elapsed=0, rss=1, free=None):
        now = [0]
        budget = M.Resources(root, clock=lambda: now[0], rss=lambda: rss,
            disk=lambda _: types.SimpleNamespace(free=40*M.GIB if free is None else free))
        now[0] = elapsed
        return budget, now

    def test_platform_rss_units(self):
        self.assertEqual(M.rss_bytes('darwin', 123), 123)
        self.assertEqual(M.rss_bytes('linux', 123), 123*1024)
        with self.assertRaises(RuntimeError): M.rss_bytes('other', 123)

    def test_charge_stops_before_original_without_refund_or_expansion(self):
        with tempfile.TemporaryDirectory() as root:
            budget, _ = self.budget(root); calls = []
            budget.counts['training_updates'] = M.FIXED_LIMITS['training_updates']
            def guarded():
                budget.charge('training_updates'); calls.append('original')
            with self.assertRaises(RuntimeError): guarded()
            self.assertEqual(calls, [])
            self.assertEqual(budget.counts['training_updates'], 6)
            with self.assertRaises(ValueError): budget.charge('models', True)

    def test_rss_time_and_disk_have_independent_hard_limits(self):
        with tempfile.TemporaryDirectory() as root:
            for kwargs, disk in (({'rss': 8*M.GIB+1}, False),
                                 ({'elapsed': 7201}, False),
                                 ({'free': 30*M.GIB-1}, True)):
                budget, _ = self.budget(root, **kwargs)
                with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError): budget.check(disk=disk)

    def test_progress_is_small_atomic_and_contains_no_reward(self):
        with tempfile.TemporaryDirectory() as root:
            budget, _ = self.budget(root)
            budget.progress('rollout', seed=7, side='off', episode=1)
            budget.progress('episode_complete')
            value = json.loads((Path(root)/'progress.json').read_text())
            self.assertEqual(value['context']['phase'], 'episode_complete')
            self.assertNotIn('reward', json.dumps(value))
            self.assertFalse((Path(root)/'progress.json.tmp').exists())

    def test_output_claim_is_exclusive_even_after_failure_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)/'claim'; M.claim_output(output, {'no_retry': True})
            M.write_json(output/'failure.json', {'status': 'failed'})
            with self.assertRaises(FileExistsError): M.claim_output(output, {'no_retry': True})
            with self.assertRaises(FileExistsError): M.write_json(output/'failure.json', {})


class ContractTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        for name in (*M.REQUIRED_SCRIPTS, 'src/agents/dummy.py'):
            target = root/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_text('# bound\n')
        home = root/'protocol'; home.mkdir()
        for name in ('protocol.pt', 'predictor.pt'): (home/name).write_bytes(b'frozen opaque asset')
        protocol = dict(protocol='joint_stream_learning_v2', action_contract='joint_stream_control_v2',
            smoke=False, episodes=64, seeds=[7,17,27],
            assets={name:M.sha_file(home/name) for name in ('protocol.pt','predictor.pt')},
            training_schedule={str(seed):['train_000']*64 for seed in (7,17,27)})
        M.write_json(home/'manifest.json', protocol)
        contract = dict(schema=M.SCHEMA, status='frozen', repo_root=str(root), output_root=str(root/'output'),
            source_commit='a'*40, baseline_source_commit=M.BASELINE_SOURCE_COMMIT,
            source_fingerprints={name:M.sha_file(root/name) for name in (*M.REQUIRED_SCRIPTS,'src/agents/dummy.py')},
            protocol=dict(path=str(home/'manifest.json'),sha256=M.sha_file(home/'manifest.json')),
            seeds=[7,17,27], sides=['off','on'], sessions_per_seed=4, training_budget=64, epochs=4,
            limits=copy.deepcopy(M.FIXED_LIMITS))
        M.write_json(root/'contract.json', contract)
        return contract

    def checked(self, root):
        def git(_root, *args): return 'a'*40 if args == ('rev-parse','HEAD') else ''
        with patch.object(M,'ROOT',Path(root).resolve()), patch.object(M,'git',side_effect=git), \
             patch.object(M.subprocess,'check_call') as diff:
            result = M.check_contract(Path(root)/'contract.json')
            diff.assert_called_once()
            return result

    def test_preflight_hashes_opaque_assets_without_loading_them(self):
        with tempfile.TemporaryDirectory() as root:
            contract = self.fixture(root)
            actual, protocol, _, _ = self.checked(root)
            self.assertEqual(actual['training_budget'],64)
            self.assertEqual(len(protocol['training_schedule']['7']),64)
            self.assertFalse(Path(contract['output_root']).exists())

    def test_source_and_asset_changes_rejected_before_claim(self):
        for relative in ('src/agents/dummy.py','protocol/protocol.pt'):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as root:
                self.fixture(root); (Path(root)/relative).write_bytes(b'changed')
                with self.assertRaises(ValueError): self.checked(root)
                self.assertFalse((Path(root)/'output').exists())

    def test_existing_output_prevents_retry(self):
        with tempfile.TemporaryDirectory() as root:
            self.fixture(root); (Path(root)/'output').mkdir()
            with self.assertRaises(FileExistsError): self.checked(root)

    def test_no_short_budget_or_bool_count_substitution(self):
        with tempfile.TemporaryDirectory() as root:
            original = self.fixture(root)
            for field, value in (('training_budget',4),('sessions_per_seed',4.),('epochs',True),('sides',['on','off'])):
                candidate=copy.deepcopy(original); candidate[field]=value
                with patch.object(M,'ROOT',Path(root).resolve()), self.subTest(field=field), self.assertRaises(ValueError):
                    M.contract_shape(candidate)

    def test_missing_binding_and_traversal_rejected(self):
        for mutation in ('missing','traversal'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                value=self.fixture(root)
                if mutation=='missing': value['source_fingerprints'].pop(M.REQUIRED_SCRIPTS[0])
                else: value['source_fingerprints']['../outside.py']='0'*64
                (Path(root)/'contract.json').write_text(json.dumps(value))
                with self.assertRaises(ValueError): self.checked(root)


class RuntimeWrapperTests(unittest.TestCase):
    def test_passive_wrappers_count_single_original_calls_and_restore(self):
        # Explicit fake callbacks only; no torch import, model or environment.
        events = []
        class Tensor:
            def backward(self): events.append('backward')
        class Categorical:
            def sample(self): events.append('sample'); return 0
        class Optimizer:
            def __init__(self): self.state = {}
            def step(self): events.append('adam'); self.state['populated'] = True
        class Agent:
            def __init__(self):
                self.optimizer = Optimizer(); self.surrogate_optimizer = Optimizer(); self.completed_updates = 0
            def _evaluate(self, **kwargs): events.append('evaluate'); return 'evaluated'
            def act_controls(self, decisions, frame, *, deterministic=False):
                events.append('act'); Categorical().sample(); self._evaluate(); return {}, None
            def update(self, rollouts, *, epochs=4):
                events.append('update'); self._evaluate(actions=[0]); Tensor().backward()
                self.optimizer.step(); self.surrogate_optimizer.step(); self.completed_updates += 1
                return {'gradient_steps':1}
        torch = types.SimpleNamespace(Tensor=Tensor, distributions=types.SimpleNamespace(Categorical=Categorical))
        agent = Agent(); original_backward = Tensor.backward
        with tempfile.TemporaryDirectory() as root:
            budget = M.Resources(root, rss=lambda:1)
            with M.guarded_runtime(agent,budget,torch) as counts:
                self.assertEqual(agent.act_controls({},0),({},None))
                self.assertEqual(agent.update([]),{'gradient_steps':1})
            self.assertEqual(events,['act','sample','evaluate','update','evaluate','backward','adam','adam'])
            self.assertEqual(counts['idle_frames'],1)
            self.assertTrue(counts['updates'][0]['before']['main_empty'])
            self.assertFalse(counts['updates'][0]['after']['main_empty'])
            self.assertEqual(budget.counts['training_updates'],1)
            self.assertEqual(budget.counts['evaluate_calls'],2)
            self.assertEqual(budget.counts['backward_calls'],1)
            self.assertEqual(budget.counts['adam_calls'],2)
            self.assertEqual(budget.counts['categorical_sample_calls'],1)
            self.assertIs(Tensor.backward,original_backward)
            self.assertFalse(any(name in vars(agent) for name in ('act_controls','update','_evaluate')))
            self.assertNotIn('step',vars(agent.optimizer))

    def test_consumer_checks_each_event_and_restores_class(self):
        from scripts.sa_stage5c_credit_audit import _Joins
        events=[]
        class Budget:
            def check(self): events.append('check')
        def original(this,*args,**kwargs): events.append('original'); return 'row'
        with patch.object(_Joins,'event',original):
            with M.guarded_consumer(Budget()):
                self.assertEqual(_Joins.event(object(),'behavior',{}),'row')
            self.assertIs(_Joins.event,original)
        self.assertEqual(events,['check','original','check'])


if __name__ == '__main__':
    unittest.main()
