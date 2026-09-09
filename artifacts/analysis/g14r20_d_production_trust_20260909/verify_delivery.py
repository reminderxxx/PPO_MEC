"""Independent standard-library verifier for the compact D evidence package.

Optional --synthetic validates the retained large fixture by its content inventory.
No admission, signing, rollout, real run writes, or trust installation occurs.
"""
import argparse,hashlib,json,pathlib,subprocess,xml.etree.ElementTree as ET

def h(path):
 s=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):s.update(b)
 return s.hexdigest()
def strict(raw):
 def pairs(rows):
  d={}
  for k,v in rows:
   if k in d:raise ValueError('duplicate key '+k)
   d[k]=v
  return d
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite '+x)))
def load(path):return strict(path.read_text(encoding='utf-8-sig'))
def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('artifact_root');a.add_argument('--repository',required=True);a.add_argument('--synthetic',action='store_true');args=a.parse_args()
 root=pathlib.Path(args.artifact_root);repo=pathlib.Path(args.repository)
 manifest=load(root/'artifact_integrity.json');listed=set()
 for row in manifest['files']:
  rel=pathlib.Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
  p=root/rel;assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['size_bytes'] and h(p)==row['sha256'],str(p)
  assert row['path'] not in listed;listed.add(row['path'])
 actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='artifact_integrity.json'}
 assert listed==actual,(listed-actual,actual-listed)
 counts=dict(json=0,jsonl=0,jsonl_records=0,xml=0)
 for p in root.rglob('*'):
  if p.suffix=='.json':load(p);counts['json']+=1
  elif p.suffix=='.jsonl':
   for line in p.read_text().splitlines():strict(line);counts['jsonl_records']+=1
   counts['jsonl']+=1
  elif p.suffix=='.xml':ET.parse(p);counts['xml']+=1
 i=load(root/'executor_identity.json')
 def git(*cmd):return subprocess.check_output(['git','-C',str(repo),*cmd])
 assert git('rev-parse',i['commit']+'^{tree}').decode().strip()==i['git_tree']
 expected={p for p in git('ls-tree','-r','--name-only',i['commit'],'scripts/continuation_executor').decode().splitlines() if p.endswith('.py')}
 expected|={'scripts/execute_fixed_commit_continuation.py','scripts/run_fixed_commit_continuation_acceptance.py','src/runtime/fixed_commit_continuation.py'}
 assert {r['path'] for r in i['files']}==expected and len(i['files'])==len(expected)
 for r in i['files']:assert hashlib.sha256(git('show',i['commit']+':'+r['path'])).hexdigest()==r['sha256']
 s=load(root/'acceptance_summary.json');assert digest(i)==s['executor_identity_sha256']
 assert s['real_execution_authorized'] is False and s['isolated_acceptance']=='pass'
 assert digest(load(root/'unsigned_execution_contract.json'))==s['unsigned_contract_sha256']
 assert digest(load(root/'frozen_command_plans.json'))==s['command_plan_sha256']
 for name,d in s['commands'].items():
  assert d['exit_code']==0 and d['executor_commit_before']==d['executor_commit_after']==i['commit']
  assert not d['executor_status_before'] and not d['executor_status_after']
 tests=list(ET.parse(root/'full_pytest.xml').getroot().iter('testsuite'))
 assert all(int(t.get(k,0))==0 for t in tests for k in ('failures','errors','skipped'))
 assert sum(int(t.get('tests',0)) for t in tests)==s['suites']['tests']
 protected=[load(root/('protected_'+n+'.json')) for n in ('before','after')]
 for k in ('main','proposal_sha256','user_files','ledgers'):assert protected[0][k]==protected[1][k]
 assert all(not d['mismatches'] and not d['additions'] for d in protected)
 checked=0
 if args.synthetic:
  v=load(root/'synthetic_file_inventory.json');f=pathlib.Path(v['root'])
  for r in v['files']:
   p=f/r['path'];assert p.is_file() and p.stat().st_size==r['size_bytes'] and h(p)==r['sha256'];checked+=1
 print(json.dumps(dict(status='pass',executor_commit=i['commit'],executor_files=len(i['files']),evidence_files=len(listed),strict_formats=counts,synthetic_files_checked=checked,real_execution_authorized=False)))
if __name__=='__main__':main()
