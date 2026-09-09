import json,pathlib,sys,subprocess,os,hashlib,datetime,collections
O=pathlib.Path(__file__).parent; E=pathlib.Path('/private/tmp/ppo_mec_g14r20_b_62f432c'); old=pathlib.Path('/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847'); B=pathlib.Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_b_continuation_20260908/e6a73357d08220d34ddc2d6537d140131c60520f')
sys.path.insert(0,str(E/'scripts'))
from continuation_executor.authorization import verify_approval,PRODUCTION_SIGNERS,approval_message
from continuation_executor.identity import verify_executor,read_json,canonical,file_hash
out={'reviewed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'executor':verify_executor(read_json(B/'executor_identity.json'),E),'production_signers':len(PRODUCTION_SIGNERS)}
# An in-memory negative input, no approval file or key creation.
c={'domain':'production','expires_at':'2099-01-01T00:00:00+00:00'}
a={'message':approval_message(c,{k:{'state':'independently_verified','reference_sha256':'0'*64} for k in ['launch_approval','release_attestation','continuation_approval']}),'signer_id':'uninstalled','signature':'00'}
try:verify_approval(c,a);raise AssertionError('empty trust accepted')
except ValueError as exc:out['empty_production_trust_refusal']=str(exc)
s=read_json(B/'synthetic_acceptance_report.json');m=s['monitor'];children=m['child_monitors'];out['b_monitor_recomputed']={'children':len(children),'reported_dispatch_count':m['synthetic_child_dispatch_count'],'all_zero':all(m[k]==0 and all(ch[k]==0 for ch in children) for k in ['scientific_rollout_count','real_v16_dispatch_count','real_v16_write_count'])}
all_dispatch=m['dispatches']+[a for ch in children for a in ch['dispatches']]
out['b_monitor_recomputed']['dispatch_entry_bijection']=collections.Counter(map(tuple,all_dispatch))==collections.Counter(tuple(ch['entry_argv']) for ch in children)
bad=[];origins=[]
for origins_map in [s['origins'],*[ch['origins'] for ch in children]]:
 for name,row in origins_map.items():
  p=pathlib.Path(row['path']);ok=old in p.parents and file_hash(p)==row['sha256'];origins.append((name,str(p),ok))
  if not ok:bad.append(row)
out['b_origins_recomputed']={'observations':len(origins),'mismatches':bad}
out['consumer_evidence']=[{'case':x['case'],'boundary':x['boundary'],'actual_calls':x['actual_calls']} for x in s['consumer_cases']]
out['b_phase_statuses']=[{'phase':x['phase'],'status':x['status'],'return_code':x.get('return_code')} for x in s['phase_results']]
out['command_log_consistency']=[]
for p in B.glob('*_job.json'):
 q=p.with_name(p.name.replace('_job.json','_command_result.json'))
 if not q.exists():continue
 j=read_json(p);r=read_json(q);out['command_log_consistency'].append({'name':p.name,'argv_equal':j['argv']==r['argv'],'cwd_equal':j['cwd']==r['cwd'],'exit_code':r['exit_code'],'duration_seconds':r.get('duration_seconds'),'fixed_commit_before':r.get('executor_commit_before'),'fixed_commit_after':r.get('executor_commit_after')})
for case in ['gate_missing','gate_false']:
 root=pathlib.Path('/private/tmp/synthetic_g14r20_c_direct_'+case);assert not root.exists();root.mkdir()
 argv=[sys.executable,'-I','-B',str(E/'tests/continuation_native_driver.py'),str(root),case]
 r=subprocess.run(argv,cwd=old,env=dict(os.environ,PYTHONPATH=str(old),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
 out[case]={'argv':argv,'cwd':str(old),'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
(O/'independent_probes.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['consumer_evidence','gate_false','gate_missing']},indent=2))
