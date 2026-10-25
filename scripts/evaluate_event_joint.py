"""Matched causal baselines and frozen-checkpoint evaluation for joint control."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.run_event_joint_learning import episode
from scripts.run_workflow_recovery import write_trace
from scripts.run_persistent_workflow_learning import dump,sha
from src.agents.ppo_agent import EventJointPPOAgent

RULES=('no_prefetch','shortest_frontier','ready_demand','min_transfer','eviction_cost')


class JointRulePolicy:
    eviction_aware=True
    def __init__(self,rule):
        if rule not in RULES:raise ValueError('Unknown causal baseline')
        self.rule=rule

    def act(self,f,deterministic=True):
        candidates=[i for i in range(5) if f['mask'][i]]
        if self.rule=='no_prefetch' or not candidates:return 5,0.,0.
        g=f['graph']
        def rank(i):
            return min((j for j in range(len(g['ids'])) if g['mask'][j] and g['x'][j,i]>.5),default=len(g['ids']))
        def key(i):
            a=f['aux'][i]
            if self.rule=='shortest_frontier':return (float(a[2]),rank(i))
            if self.rule=='ready_demand':return (-float(a[1]),float(a[2]),rank(i))
            if self.rule=='min_transfer':return (float(a[6]),float(a[2]),rank(i))
            # Outstanding shortest request work + actual count of ready tasks
            # whose warm objects would be evicted. Fixed transparent units.
            return (float(a[2])*20.+float(a[9])*8.,float(a[6]),rank(i))
        return min(candidates,key=key),0.,0.


def evaluate(args):
    out=Path(args.output);out.mkdir(parents=True);torch.set_num_threads(1);p=json.loads(Path(args.inputs).read_text())
    units=[u for u in p['units'] if args.split=='all' or u['split']==args.split]
    if not units:raise ValueError('Empty evaluation split')
    if any(u['split']=='confirmation' for u in units):
        if not args.freeze:raise ValueError('Confirmation requires a frozen manifest')
        freeze=json.loads(Path(args.freeze).read_text())
        if sha(args.inputs) not in freeze.get('input_sha256s',[freeze.get('inputs_sha256')]):raise ValueError('Confirmation input freeze mismatch')
        if args.checkpoint and sha(args.checkpoint) not in freeze['checkpoint_sha256']:raise ValueError('Unfrozen checkpoint')
        if not args.checkpoint and args.rule not in freeze['rules']:raise ValueError('Unfrozen baseline')
    if args.checkpoint:
        agent,payload=EventJointPPOAgent.load(args.checkpoint)
        if not agent.eviction_aware:raise ValueError('Evaluation requires the shared v2 physical contract')
        name='primitive_ppo' if agent.controller=='primitive' else 'event_graph_ppo' if agent.use_graph else 'event_set_ppo';seed=agent.seed
    else:agent=JointRulePolicy(args.rule);name=args.rule;seed=0
    provenance=dict(source_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),argv=sys.argv,input_sha256=sha(args.inputs),
        checkpoint_sha256=sha(args.checkpoint) if args.checkpoint else None,freeze_sha256=sha(args.freeze) if args.freeze else None,
        name=name,seed=seed,split=args.split,physical_contract='joint_service_hit_lru_v1',feature_contract='event_joint_eviction_features_v2',formal=False)
    dump(out/'command.json',provenance);rows=[];start=time.monotonic()
    for unit in units:
        result,decisions,trace=episode(unit,p,agent,90000+unit['index'],True)
        name_trace=f'w{unit["index"]}.json.gz';write_trace(out/name_trace,trace)
        rows.append(dict(method=name,seed=seed,window=unit['index'],outer_window=unit.get('outer_window_index',unit['index']),case_index=unit.get('case_index',unit['index']),workflow_ids=[w['workflow_id'] for w in unit.get('workflows',p['workflows'])],split=unit['split'],decisions=len(decisions),trace=name_trace,trace_sha256=sha(out/name_trace),**result))
        dump(out/'rows.json',rows)
        print(json.dumps(dict(method=name,window=unit['index'],reward=result['reward'],completed=result['completed_requests'])),flush=True)
    dump(out/'completion.json',dict(status='complete',rows=len(rows),native_steps=sum(x['steps'] for x in rows),wall_seconds=time.monotonic()-start,provenance=provenance))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--split',default='dev',choices=['train','dev','confirmation','all']);p.add_argument('--rule',choices=RULES);p.add_argument('--checkpoint');p.add_argument('--freeze');a=p.parse_args()
    if bool(a.rule)==bool(a.checkpoint):p.error('Select exactly one baseline or checkpoint')
    evaluate(a)
