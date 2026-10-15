"""One bounded background pipeline: frozen preparation, all methods, raw audit."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output_root',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    home=a.output_root.resolve();home.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',phase='preparing_protocol',root=str(home),pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),smoke=a.smoke,completed_methods=[])
    def publish():
        tmp=home/'progress.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(home/'progress.json')
    def run(stage,script,args):
        state['phase']=stage;publish()
        command=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)]
        with (home/f'{stage}_command.json').open('w') as f:json.dump(command,f,indent=2)
        with (home/f'{stage}.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        run('prepare','prepare_joint_stream_learning.py',['--output',home/'protocol',*(['--smoke'] if a.smoke else [])])
        manifest=json.loads((home/'protocol/manifest.json').read_text())
        for name in manifest['methods']:
            run(name,'train_joint_stream_control.py',['--protocol',home/'protocol/manifest.json','--output_root',home,'--agent',name])
            state['completed_methods'].append(name);publish()
        run('audit','audit_joint_stream_learning.py',[home]);state.update(status='complete',phase='audited_results_ready')
    except Exception as error:
        state.update(status='failed',error=repr(error));publish();(home/'launcher_result.json').write_text(json.dumps(state,indent=2)+'\n');raise
    publish();(home/'launcher_result.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)


if __name__=='__main__':main()
