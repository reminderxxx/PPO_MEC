"""Read-only B qualification inspector. This CLI cannot create locks or dispatch phases."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal',required=True)
    parser.add_argument('--contract',required=True)
    args=parser.parse_args()
    def external(name):
        spec=importlib.util.spec_from_file_location('external_'+name,ROOT/'scripts'/(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    security=external('continuation_executor_security')
    contract=security.validate_execution_contract(security.strict_json(args.contract))
    counters={'scientific_calls':0,'scientific_rollout_count':0,'real_v16_dispatch_count':0,
              'real_v16_write_count':0,'blocked_attempts':0,'readonly_subprocesses':[]}
    def audit(event,values):
        if event=='open':
            path,mode,flags=values
            write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if write and path!='/dev/null':
                counters['blocked_attempts']+=1
                raise ValueError('read-only qualification blocked write')
        if event in {'os.mkdir','os.remove','os.rename','os.rmdir','os.link','os.symlink','os.truncate','os.chmod','os.utime'}:
            counters['blocked_attempts']+=1
            raise ValueError('read-only qualification blocked mutation: '+event)
        if event=='subprocess.Popen':
            executable,argv,cwd,env=values
            if not isinstance(argv,(tuple,list)):
                raise ValueError('read-only inspector forbids shell')
            git=Path(str(executable)).name=='git' and any(x in argv for x in ('rev-parse','status','diff','ls-files','show'))
            probe=os.path.abspath(str(executable))==os.path.abspath(sys.executable) and '-c' in argv
            if not (git or probe):
                counters['blocked_attempts']+=1
                raise ValueError('read-only inspector blocks scientific dispatch')
            counters['readonly_subprocesses'].append({'argv':list(argv),'cwd':str(cwd)})
    def profile(frame,event,arg):
        if event!='call':return
        module=frame.f_globals.get('__name__','') or '';name=frame.f_code.co_name
        if not module.startswith(('src.','scripts.')):return
        counters['scientific_calls']+=1
        forbidden=('rollout' in name or name in {'run_real_episode','train','learn'}) or (
            module.startswith('src.envs.') and name in {'step','reset'}) or (module.startswith('scripts.') and name=='main')
        if forbidden:
            counters['blocked_attempts']+=1
            raise ValueError('read-only inspector blocks scientific function '+module+'.'+name)
    sys.addaudithook(audit);sys.setprofile(profile)
    try:
        qualified=external('continuation_qualification').qualify(args.proposal,contract)
        report={'status':'read_only_checks_passed','execution_authorized':False,
            'origin_evidence':contract['origin_evidence'],'report':qualified['report'],
            'plans':qualified['plans'],'counters':counters}
        code=0
    except Exception as exc:
        report={'status':'rejected','execution_authorized':False,'error':str(exc),'counters':counters};code=2
    finally:sys.setprofile(None)
    print(json.dumps(report,indent=2,allow_nan=False,default=str))
    return code

if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'rejected','execution_authorized':False,'error':str(exc)}))
        raise SystemExit(2)
