"""Prerecorded, bounded opportunity probe; no policy training or hidden holdout."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_ready_workflow_learning import make_env, rule_action, node_runtime, request_remaining
from scripts.run_workflow_recovery import workflows, result_from, write_trace
from scripts.run_persistent_workflow_learning import dump,sha
from src.envs.wrappers.joint_workflow_control import JointWorkflowEnv,JOINT_CONTRACT

RULES=('cache_first','shortest_request','joint_ready','joint_successor','joint_demand','joint_request')


def joint_env(unit,wfs,count=8,seed=7,horizon=95,capacity=384.):
    # Match the tested physical factory exactly; only expose another control method.
    return make_env(unit,wfs,count,seed,horizon,capacity,env_class=JointWorkflowEnv)


def joint_action(state,rule):
    node_id=rule_action(state,'cache_first' if rule!='joint_request' else 'shortest_request')
    nodes=state['workflow']['nodes'];by_id={n['node_id']:n for n in nodes};node=by_id[node_id]
    vehicle=next(v for v in state['vehicles'] if v['vehicle_id']==state['primary_vehicle_id'])
    rsu=vehicle.get('associated_rsu_id');rt=node_runtime(state,node)
    if not rsu:return node_id,None
    # Never overwrite an in-flight admission merely because it cannot serve yet.
    if not rt['present']:return node_id,node['required_adapter']
    if any(x['rsu_id']==rsu and x['remaining_seconds']>0 for x in state['model_loading']):return node_id,None
    done=set(state['workflow']['completed_node_ids']);remaining=request_remaining(state)
    ready=set(state['ready_node_ids']);rank={n:i for i,n in enumerate(state['workflow']['execution_order'])}
    pending=[n for n in nodes if n['node_id'] not in done and not node_runtime(state,n)['present']]
    if rule=='joint_ready':pending=[n for n in pending if n['node_id'] in ready]
    if not pending:return node_id,None
    def key(n):
        immediate=n['node_id'] in node['successors'] and set(n['predecessors'])-done=={node_id}
        request=n['node_id'].split(':')[0]
        distance=len(set(n['predecessors'])-done)
        demand=sum(x['required_adapter']==n['required_adapter'] and (x['node_id'] in ready or set(x['predecessors'])-done=={node_id}) for x in pending)
        if rule=='joint_demand':return (-demand,distance,remaining[request],rank[n['node_id']])
        if rule=='joint_successor':return (not immediate,remaining[request],distance,rank[n['node_id']])
        return (remaining[request],distance,rank[n['node_id']])
    target=min(pending,key=key)['required_adapter']
    return node_id,target


def rollout(unit,wfs,rule):
    env=joint_env(unit,wfs);state,_=env.reset();trace=[]
    while True:
        before=state
        if rule in ('cache_first','shortest_request'):
            node=rule_action(state,rule);prep=None
            state,r,done,trunc,info=env.step_node(node)
        else:
            node,prep=joint_action(state,rule)
            state,r,done,trunc,info=env.step_joint(node,prep)
        trace.append(dict(state=before,node_id=node,prepare_adapter=prep,reward=r.total,stream=info['persistent_stream']))
        if done or trunc:break
    return result_from(type('Envelope',(),{'core_env':env})()),trace


def main():
    a=argparse.ArgumentParser();a.add_argument('--inputs',required=True);a.add_argument('--output',required=True);args=a.parse_args()
    out=Path(args.output);out.mkdir();p=json.loads(Path(args.inputs).read_text());rows=[]
    protocol=dict(contract=JOINT_CONTRACT,rules=RULES,inputs_sha256=sha(args.inputs),source_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        hypothesis='Independent service and future loading can improve the unchanged business return',
        gate='At least one TRAIN window has positive completion or >1 reward gain vs strong global-cache-first; all dev rows are exposed diagnostics',
        formal=False,holdout=False,argv=sys.argv)
    dump(out/'protocol.json',protocol)
    for unit in p['units']:
        for rule in RULES:
            result,trace=rollout(unit,workflows(p['workflows']),rule)
            name=f'w{unit["index"]}_{rule}.json.gz';write_trace(out/name,trace)
            rows.append(dict(window=unit['index'],split=unit['split'],rule=rule,trace=name,trace_sha256=sha(out/name),**result))
        dump(out/'rows.json',rows)
        print(json.dumps(dict(window=unit['index'],results=[{k:x[k] for k in ['rule','reward','completed_requests','transfer_mb']} for x in rows[-len(RULES):]])),flush=True)
    pairs=[]
    for unit in p['units']:
        r={x['rule']:x for x in rows if x['window']==unit['index']};base=r['cache_first']
        for rule in RULES[2:]:pairs.append(dict(window=unit['index'],split=unit['split'],rule=rule,reward_gain=r[rule]['reward']-base['reward'],completion_gain=r[rule]['completed_requests']-base['completed_requests']))
    passed=any(x['split']=='train' and (x['completion_gain']>0 or x['reward_gain']>1) for x in pairs)
    dump(out/'results.json',dict(rows=rows,pairs=pairs,opportunity_gate_passed=passed,native_steps=sum(x['steps'] for x in rows),formal=False,holdout=False))

if __name__=='__main__':main()
