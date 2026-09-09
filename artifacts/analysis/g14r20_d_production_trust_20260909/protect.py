import pathlib,json,gzip,hashlib,subprocess,sys,datetime
R=pathlib.Path('/Users/howen/Projects/PPO_MEC'); O=pathlib.Path('/private/tmp/g14r20_d_evidence')
def h(p):
 s=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(4194304),b''):s.update(c)
 return s.hexdigest()
v=json.load(gzip.open(R/'artifacts/analysis/g14r20_a_continuation_20260908/protected_before.json.gz','rt'));bad=[]
for x in v['files']:
 p=pathlib.Path(x['path']); ok=(p.is_symlink() and str(p.readlink())==x['symlink_target']) if 'symlink_target' in x else (p.is_file() and p.stat().st_size==x['size_bytes'] and h(p)==x['sha256'])
 if not ok:bad.append(str(p))
p=json.loads((R/'artifacts/analysis/g14r20_a_continuation_20260908/v16_continuation_proposal.json').read_text()); known={x['path'] for x in v['files']}
roots=[pathlib.Path(p['run_root']),pathlib.Path('/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847')]
user=['scripts/train_sa_ghmappo_real_sample.py','src/agents/sa_ghmappo_agent.py','src/agents/sa_ghmappo_core.py','src/encoders/fusion_encoder.py','src/evaluators/real_eval_support.py','tests/test_algo_pool_contract.py','tests/test_checkpoint_compat.py']
d=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),count=len(v['files']),mismatches=bad,additions=[str(x) for r in roots for x in r.rglob('*') if (x.is_file() or x.is_symlink()) and str(x) not in known],user_files={f:h(R/f) for f in user},main=subprocess.check_output(['git','-C',str(R),'rev-parse','main'],text=True).strip(),proposal_sha256=h(R/'artifacts/analysis/g14r20_a_continuation_20260908/v16_continuation_proposal.json'),ledgers=[dict(path=x['path'],sha256=h(x['path']),size=pathlib.Path(x['path']).stat().st_size) for x in p['ledgers']])
(O/('protected_'+sys.argv[1]+'.json')).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d));assert not bad and not d['additions']
