"""Execute one fixed method group; no hidden evaluation or automatic extension."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_event_joint_learning import train
from scripts.run_persistent_workflow_learning import dump,sha


def run(args):
    plan=json.loads(Path(args.plan).read_text());p=json.loads(Path(args.inputs).read_text());p['_sha']=sha(args.inputs)
    method=plan['methods'][args.method]
    if plan['episodes_per_cell']!=256 or plan['checkpoint_cadence']!=16:raise ValueError('Unexpected frozen budget')
    if any(u['split']=='confirmation' for u in p['units']):raise ValueError('Training input contains confirmation')
    out=Path(args.output);out.mkdir(parents=True)
    dump(out/'command.json',dict(argv=sys.argv,source_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        plan_sha256=sha(args.plan),inputs_sha256=p['_sha'],method=args.method,python=platform.python_version(),platform=platform.platform()))
    for seed in plan['seeds']:
        train(out/f'seed_{seed}',p,seed,method['graph'],plan['episodes_per_cell'],plan['checkpoint_cadence'],True,method['controller'])
    dump(out/'completion.json',dict(status='complete',method=args.method,seeds=plan['seeds'],episodes_per_cell=256,plan_sha256=sha(args.plan)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--method',required=True,choices=['event_set_ppo','event_graph_ppo','primitive_ppo']);run(p.parse_args())
