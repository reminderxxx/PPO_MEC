import pathlib,json,gzip,hashlib,subprocess,datetime,xml.etree.ElementTree as ET
R=pathlib.Path('/Users/howen/Projects/PPO_MEC'); B=R/'artifacts/analysis/g14r20_b_continuation_20260908/e6a73357d08220d34ddc2d6537d140131c60520f'; O=pathlib.Path(__file__).parent
def H(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(R),*a],text=True).strip()
def check(rows,root=None):
 bad=[];size=0
 for x in rows:
  p=(root/x['path']) if root else pathlib.Path(x['path'])
  if 'symlink_target' in x:
   ok=p.is_symlink() and str(p.readlink())==x['symlink_target']
  else:
   ok=p.is_file() and p.stat().st_size==x['size_bytes'] and H(p)==x['sha256'];size+=x.get('size_bytes',0)
  if not ok:bad.append(str(p))
 return dict(count=len(rows),bytes=size,mismatches=bad)
d={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for n in ['versioned_evidence_manifest.json','artifact_integrity.json']:
 v=json.loads((B/n).read_text());d[n]=check(v['files'],B)
print('B inventories',d,flush=True)
i=json.loads((B/'executor_identity.json').read_text());d['executor_tree']=git('rev-parse',i['commit']+'^{tree}');d['executor_files']=[]
for x in i['files']:
 raw=subprocess.check_output(['git','-C',str(R),'show',i['commit']+':'+x['path']]);d['executor_files'].append(dict(path=x['path'],matches=hashlib.sha256(raw).hexdigest()==x['sha256']))
d['evidence_parent']=git('show','-s','--format=%P','33bc736');d['main']=git('rev-parse','main');d['evidence_diff']=git('diff','--name-only','e6a7335','33bc736').splitlines()
rel=str(B.relative_to(R))
d['versioned_commit_comparison']=[]
for x in json.loads((B/'versioned_evidence_manifest.json').read_text())['files']:
 raw=subprocess.check_output(['git','-C',str(R),'show','33bc736:'+rel+'/'+x['path']]);d['versioned_commit_comparison'].append(dict(path=x['path'],matches=hashlib.sha256(raw).hexdigest()==x['sha256']))
p=json.loads((R/'artifacts/analysis/g14r20_a_continuation_20260908/v16_continuation_proposal.json').read_text());d['proposal_keys']=list(p);d['ledger_checks']=[]
for x in p['ledgers']:
 raw=pathlib.Path(x['path']).read_bytes();rows=[json.loads(l) for l in raw.splitlines()];d['ledger_checks'].append(dict(kind=x['kind'],records=len(rows),bytes=len(raw),prefix_matches=hashlib.sha256(raw[:x['byte_count']]).hexdigest()==x['prefix_sha256'],no_successors=len(raw)==x['byte_count'],states=[{k:r[k] for k in ['phase','status','event','state'] if k in r} for r in rows] if x['kind']=='phase' else None))
d['user_files']={p:H(p) for p in json.loads((B/'protected_final.json').read_text())['user_files_end']};d['user_files_match_b']=d['user_files']==json.loads((B/'protected_final.json').read_text())['user_files_end']
for n in ['full_pytest.xml','targeted_pytest.xml']:
 root=ET.parse(B/n).getroot();d[n]=[s.attrib for s in root.iter('testsuite')]
v=json.load(gzip.open(R/'artifacts/analysis/g14r20_a_continuation_20260908/protected_before.json.gz','rt'));d['protected_inventory']=check(v['files']);print('protected done',d['protected_inventory'],flush=True)
paths={x['path'] for x in v['files']};d['additions']={}
for root in [pathlib.Path(p['run_root']),pathlib.Path('/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847')]:
 d['additions'][str(root)]=[str(x) for x in root.rglob('*') if (x.is_file() or x.is_symlink()) and str(x) not in paths]
(O/'identity_integrity_recomputed.json').write_text(json.dumps(d,indent=2)+'\n')
print('DONE',flush=True)
