"""Pure-stdlib numerical qualification guard/comparator fixtures.

Never invokes runner.run/runtime, constructs a model, or executes any numeric
network/gradient/optimizer work. The authoritative runner budget is separate.
"""
import ast,copy,hashlib,importlib.util,json,math,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts'/'sa_stage5c_numerical_qualification.py'
# Only the uninstalled /tmp draft uses its sibling; copied repo tests bind ROOT/scripts.
if not SCRIPT.is_file() and Path(__file__).resolve().parent==Path('/tmp').resolve():
 SCRIPT=Path(__file__).resolve().with_name('sa_stage5c_numerical_qualification.py')

def load_helpers():
 if not SCRIPT.is_file():raise unittest.SkipTest('Runner draft not delivered; tests designed but not executed')
 spec=importlib.util.spec_from_file_location('stage5c_numerical_pure_helpers',SCRIPT)
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 return module

class NumericalGuardTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.M=load_helpers()
 def budget(self,d,cap=2):
  root=Path(d)/'budget';root.mkdir();return self.M.Budget({'original_evaluate_calls':cap,'environment_calls':0,'backward_calls':0,'adam_calls':0},root)
 def test_plain_helper_import_has_no_top_level_runtime_imports(self):
  tree=ast.parse(SCRIPT.read_text())
  imports=[]
  for node in tree.body:
   if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
   elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
  self.assertFalse(any(n.split('.')[0] in ('torch','numpy','src','scripts') for n in imports),imports)
 def test_budget_exact_cap_then_reject_before_fake_original(self):
  with tempfile.TemporaryDirectory() as d:
   b=self.budget(d);calls=[]
   def fake_original():b.charge('original_evaluate_calls');calls.append('called')
   fake_original();fake_original();self.assertEqual(len(calls),2)
   with self.assertRaises(Exception):fake_original()
   self.assertEqual(len(calls),2);self.assertEqual(b.counts['original_evaluate_calls'],2)
 def test_zero_environment_backward_adam_block_before_callback(self):
  for key in ('environment_calls','backward_calls','adam_calls'):
   with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
    b=self.budget(d);calls=[]
    def guarded():b.charge(key);calls.append('called')
    with self.assertRaises(Exception):guarded()
    self.assertEqual(calls,[]);self.assertEqual(b.counts[key],0)
 def test_negative_boolean_fractional_and_unknown_budget_never_charge(self):
  with tempfile.TemporaryDirectory() as d:
   b=self.budget(d);before=copy.deepcopy(b.counts)
   for amount in (-1,True,.5,'1'):
    with self.subTest(amount=amount),self.assertRaises(Exception):b.charge('original_evaluate_calls',amount)
    self.assertEqual(b.counts,before)
   with self.assertRaises(Exception):b.charge('unfrozen_operation')
   self.assertEqual(b.counts,before)
 def test_multi_charge_upper_bound_is_atomic(self):
  with tempfile.TemporaryDirectory() as d:
   b=self.budget(d,cap=3);b.charge('original_evaluate_calls',2)
   with self.assertRaises(Exception):b.charge('original_evaluate_calls',2)
   self.assertEqual(b.counts['original_evaluate_calls'],2)
 def test_counts_persist_before_failure_and_denial_does_not_reset(self):
  with tempfile.TemporaryDirectory() as d:
   b=self.budget(d);b.charge('original_evaluate_calls');root=Path(d)/'budget';p=root/'progress.json'
   self.assertTrue(p.exists());saved=json.loads(p.read_text());self.assertEqual(saved['counts']['original_evaluate_calls'],1)
   try:b.charge('environment_calls')
   except Exception:pass
   self.assertEqual(b.counts['original_evaluate_calls'],1);self.assertEqual(b.counts['environment_calls'],0)
   saved=json.loads(p.read_text());self.assertEqual(saved['counts']['original_evaluate_calls'],1)
   self.assertTrue(b.denied)
 def test_budget_does_not_borrow_mutated_external_limits(self):
  with tempfile.TemporaryDirectory() as d:
   limits={'environment_calls':0};root=Path(d)/'b';root.mkdir();b=self.M.Budget(limits,root);limits['environment_calls']=999
   with self.assertRaises(Exception):b.charge('environment_calls')
   self.assertEqual(b.counts['environment_calls'],0)
 def test_exact_container_values_and_small_difference(self):
  x={'source':'frozen','tuple':(1,None,True),'rng':b'\x00\x02','v':[1.,-.25]}
  self.M.compare_exact(x,copy.deepcopy(x));y=copy.deepcopy(x);y['v'][0]+=1e-12
  with self.assertRaises(Exception):self.M.compare_exact(x,y)
 def test_exact_type_shape_keys_and_bytes(self):
  for a,b in [(True,1),(1,1.),([1],(1,)),({'a':1},{'b':1}),([1],[1,2]),(b'abc',b'abd')]:
   with self.subTest(a=a,b=b),self.assertRaises(Exception):self.M.compare_exact(a,b)
 def test_nan_and_infinity_rejected_even_identical_alias(self):
  for n in (float('nan'),float('inf'),float('-inf')):
   for a,b in [(n,n),([n],[n]),({'same_alias':[n]},None)]:
    if b is None:b=a
    with self.subTest(value=repr(n)),self.assertRaises(Exception):self.M.compare_exact(a,b)
 def test_signed_zero_bits_must_not_be_conflated(self):
  self.M.compare_exact(0.0,0.0);self.M.compare_exact(-0.0,-0.0)
  for a,b in [(0.0,-0.0),([0.0],[-0.0]),({'value':0.0},{'value':-0.0})]:
   with self.subTest(a=a,b=b),self.assertRaises(Exception):self.M.compare_exact(a,b)
 def test_nonfinite_set_members_and_dict_keys_rejected(self):
  for n in (float('nan'),float('inf'),float('-inf')):
   for x in ({n},frozenset([n]),{n:'value'}):
    with self.subTest(container=type(x).__name__,value=repr(n)),self.assertRaises(Exception):self.M.compare_exact(x,x)
 def test_rng_python_numpy_and_cpu_domains_all_required(self):
  rng={'python':(3,[1,2,3],None),'numpy':('MT19937',[1,2,3],624,0,0.),'torch_cpu':bytes([1,2,3]),'cuda':None}
  self.M.compare_exact(rng,copy.deepcopy(rng))
  for domain in ('python','numpy','torch_cpu'):
   y=copy.deepcopy(rng)
   if domain=='python':y[domain]=(3,[1,2,4],None)
   elif domain=='numpy':y[domain]=('MT19937',[1,2,4],624,0,0.)
   else:y[domain]=bytes([1,2,4])
   with self.subTest(domain=domain),self.assertRaises(Exception):self.M.compare_exact(rng,y)
 def test_bound_old_asset_changed_and_missing_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'old.asset';p.write_bytes(b'frozen input');h=hashlib.sha256(p.read_bytes()).hexdigest();binding=[dict(path=str(p),sha256=h)]
   self.assertEqual(self.M.sha_file(p),h);self.M.verify_bindings(binding);p.write_bytes(b'changed input')
   with self.assertRaises(Exception):self.M.verify_bindings(binding)
   p.unlink()
   with self.assertRaises(Exception):self.M.verify_bindings(binding)
 def test_json_failure_receipt_keeps_counts_and_claim(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);claim=p/'run_started.json';self.M.write_json(claim,{'claim':1},exclusive=True)
   b=self.budget(d);b.charge('original_evaluate_calls');failure={'status':'failed','counts':dict(b.counts),'denied':b.denied,'no_retry':True};self.M.write_json(p/'failure.json',failure,exclusive=True)
   self.assertEqual(json.loads((p/'failure.json').read_text())['counts']['original_evaluate_calls'],1)
   with self.assertRaises(Exception):self.M.write_json(claim,{'claim':2},exclusive=True)
   self.assertEqual(json.loads(claim.read_text()),{'claim':1})

if __name__=='__main__':unittest.main()
