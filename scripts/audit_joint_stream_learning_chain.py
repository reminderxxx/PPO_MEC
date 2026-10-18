"""Reproduce the frozen stream learning-chain audit without training or rollout.

Reads the confirmation protocol fingerprints and archived training summaries;
executes fixed synthetic forward/autograd checks and only the AST-extracted
replay-construction prefix. No full update, environment or optimizer step is
called. Output JSON names retain the independently reviewed receipt names.

Install under scripts/ so the default root resolves to the repository root.
Pass --root explicitly when running a copy outside that directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

OUTPUT_NAMES = (
    "ppo_mec_stage2_learning_chain_audit.json",
    "ppo_mec_stage2_dqn_replay_order.json",
)

def audit_learning_chain(root: Path, output_root: Path) -> None:
    import ast, copy, hashlib, json, math, pathlib, sys
    from datetime import datetime, timezone
    import numpy as np
    import torch
    ROOT=pathlib.Path(root)
    sys.path.insert(0,str(ROOT)); torch.set_num_threads(1);torch.manual_seed(432)
    from scripts.train_joint_stream_control import make_agent
    from src.agents.dqn_agent import StreamDQNAgent

    def forbidden(*a,**k): raise RuntimeError('audit forbids optimizer.step')
    torch.optim.Adam.step=forbidden

    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def tensorhash(p):return hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in p.state_dict().values())).hexdigest()
    def data(B=12):
        g=torch.Generator().manual_seed(521)
        d={'own':torch.rand(B,20,generator=g)*.3,'critic':torch.rand(B,36,generator=g)*.3,
           'graph':torch.rand(B,64,4,generator=g)*.3,'adjacency':torch.eye(64).repeat(B,1,1),
           'graph_mask':torch.arange(64).repeat(B,1)<6,'domain':torch.rand(B,4,generator=g)*.3,
           'prior0':torch.tensor([0.,-.5,-1.,-.7,-.2,0.]).repeat(B,1),'opportunity_count':torch.full((B,),2)}
        for j,n in enumerate((6,9,20)):
            d[f'f{j}']=torch.rand(B,n,32,generator=g)*.3
            d[f'm{j}']=torch.ones(B,n,dtype=torch.bool)
            d[f'node{j}']=torch.arange(n).remainder(6).repeat(B,1)
            f=d[f'f{j}'];f[...,0]=0;f[...,16:20]=0;f[...,12]=1;f[...,14]=0
            for k in range(n):
                site=(k-1)%4 if j!=2 else k//4-1
                if ((j!=2 and k>0 and k!=5) or (j==2 and site>=0)):
                    f[:,k,0]=1;f[:,k,16+site]=1
            f[...,4]=1;f[...,6]=1;f[...,5]=0
        d['m0'][:,[0,5]]=False
        # Some one-choice execution states and cold execution closures.
        d['m0'][::3]=False;d['m0'][::3,1]=True
        d['f0'][1::3,:,4]=0;d['f0'][1::3,:,6]=0;d['own'][:,9]=0
        d['f2'][:,:4,0]=0;d['f2'][:,:4,16:20]=0
        return d

    def grads(loss,p):
        names,params=zip(*[(n,t) for n,t in p.named_parameters() if t.requires_grad])
        gs=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
        groups={}
        for n,g in zip(names,gs):
            if g is not None:groups[n.split('.')[0]]=groups.get(n.split('.')[0],0.)+float(g.square().sum())
        return {k:math.sqrt(v) for k,v in groups.items()}, math.sqrt(sum(groups.values()))

    run=ROOT/'artifacts/joint_stream_learning/20261016T162100Z_confirmation'
    manifest=json.loads((run/'protocol/manifest.json').read_text())
    identity={f:sha(ROOT/f)==h for f,h in manifest['source_fingerprints'].items()}
    assert all(identity.values())
    out={'reviewed_at':datetime.now(timezone.utc).isoformat(),'source_commit':'35975cb3d73c84ddbabf01f6749bbc2d60c826de',
         'review_commit':'4ee7fd0504c46da0fe12b00039026a3200f47df9','source_fingerprints_checked':len(identity),
         'environment_steps':0,'optimizer_steps':0,'full_update_calls':0,'synthetic_only':True,'probability':[],'gradients':[]}
    for name in ['sa_ghmappo','sa_no_prediction','mappo','ppo','cache_offload_drl','dag_offload_drl','dt_handoff_drl','dqn']:
        p=make_agent(name,17,64);before=tensorhash(p);d=data()
        with torch.no_grad():
            first_a,first_log=p._evaluate(d,execution_only=True)
            # All second-pass modifications are confined to cache/prepare tensors.
            second={k:v.clone() for k,v in d.items()};second['f1'][...,7]+=.1
            behavior=p._evaluate(second,execution_actions=first_a[:,0],return_details=True)
            replay=p._evaluate(second,actions=behavior[0],return_details=True)
        diff=(behavior[1]-replay[1]).abs().max().item();prefix=(first_log[:,0]-behavior[1][:,0]).abs().max().item()
        assert diff<1e-6 and prefix<1e-6
        out['probability'].append({'method':name,'old_current_logprob_max_error':diff,'first_second_execution_logprob_error':prefix,
            'ratio_max_abs_error':(torch.exp(replay[1]-behavior[1])-1).abs().max().item(),
            'active_head_counts':behavior[6].sum(0).tolist()})
        if name!='dqn':
            now=p._evaluate(second,actions=behavior[0],return_details=True)
            a,logs,ent,value,surr,prefix,enabled,cf,q,_=now
            target=torch.linspace(.5,4.0,len(a));advantages=target[:,None]-behavior[7]
            for j in range(3):
                sample=advantages[enabled[:,j],j]
                if len(sample)>1:advantages[:,j]=(advantages[:,j]-sample.mean())/sample.std().clamp(min=1e-6)
            ratio=(logs-behavior[1]).exp();weight=enabled.float();den=weight.sum().clamp(min=1)
            policy=-(torch.minimum(ratio*advantages,ratio.clamp(.8,1.2)*advantages)*weight).sum()/den
            vf=(prefix-target[:,None]).square().mean()+(value-target).square().mean()+(q-target[:,None]).square().mean()
            aux=(surr-torch.cat((target[:,None],torch.zeros(len(a),2)),-1)[:,None]).square().mean() if p.surrogate_enabled else vf*0
            entropy=-p._entropy_coefficient(0)*(ent*weight).sum()/den
            full=policy+.5*vf+entropy+.2*aux
            result={ 'method':name,'target_range':[.5,4.0], 'components':{} }
            for label,loss in [('policy',policy),('value_half',.5*vf),('entropy',entropy),('surrogate_point2',.2*aux),('total',full)]:
                group,norm=grads(loss,p);result['components'][label]={'groups':group,'norm':norm}
            groups=result['components']['total']['groups'];nongrad=math.sqrt(sum(v*v for k,v in groups.items() if k!='surrogates'))
            result['non_surrogate_clip_coefficient']=min(1.,.5/(nongrad+1e-6))
            result['head_norm_before_clip']=groups.get('heads',0.)
            result['head_norm_after_clip']=groups.get('heads',0.)*result['non_surrogate_clip_coefficient']
            out['gradients'].append(result)
        assert tensorhash(p)==before

    # Run ONLY the actual replay-construction prefix of source update; never call update.
    tree=ast.parse((ROOT/'src/agents/dqn_agent.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StreamDQNAgent')
    update=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='update')
    body=[]
    for n in update.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='losses' for t in n.targets):break
        body.append(n)
    fn=ast.FunctionDef(name='source_replay_construction_only',args=update.args,body=body,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));scope={};exec(compile(module,'<exact_source_replay_prefix>','exec'),scope)
    p=make_agent('dqn',17,64);before=tensorhash(p)
    def row(frame,remaining,request,*,end=False,vid='v'):
        d=data(1);d['own'][0,0]=frame/480
        return {'data':d,'vehicle_ids':[vid],'request_ids':[request], 'actions':torch.tensor([[1,0,0]]),
                'returns':torch.tensor([remaining*8]),'initial_request_count':8,'episode_end':end}
    rows=[row(0,3.,'req1'),row(5,2.5,'req2'),row(9,.75,'req2',end=True),row(0,4.,'req3'),row(7,1.,'req3',end=True)]
    scope['source_replay_construction_only'](p,rows)
    transitions=[{'frame':round(float(x[0]['own'][0,0])*480),'next_frame':round(float(x[2]['own'][0,0])*480),'reward':x[3],'terminal':x[4]} for x in p.replay]
    assert [x['reward'] for x in transitions]==[1.,3.,.75,1.75,.5]
    assert sum(x['terminal'] for x in transitions)==2
    assert abs(sum(x['reward'] for x in transitions)-7.)<1e-6
    assert tensorhash(p)==before
    out['dqn_replay']=transitions
    # Existing raw training logs: no stored gradient decomposition or behavior tensors.
    out['training_log_summary']={}
    for name in ['sa_ghmappo','mappo','ppo','cache_offload_drl','dag_offload_drl','dt_handoff_drl','dqn']:
        rows=[json.loads(l) for l in (run/name/'training.jsonl').read_text().splitlines()]
        losses=[r['losses'] for r in rows if r.get('losses')]
        out['training_log_summary'][name]={k:{'min':min(float(x[k]) for x in losses if k in x),'max':max(float(x[k]) for x in losses if k in x),'mean':float(np.mean([float(x[k]) for x in losses if k in x]))}
            for k in ['value_loss','actor_gradient_norm','surrogate_gradient_norm','approx_kl','gradient_steps'] if any(k in x for x in losses)}
    path=pathlib.Path(output_root)/'ppo_mec_stage2_learning_chain_audit.json';path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'output':str(path),'probability':out['probability'],'clip_coefficients':[(r['method'],r['non_surrogate_clip_coefficient']) for r in out['gradients']],'dqn_replay':transitions},indent=2))


def audit_dqn_replay_order(root: Path, output_root: Path) -> None:
    import ast, hashlib, json, pathlib, sys
    from collections import deque
    from datetime import datetime, timezone
    import torch
    ROOT=pathlib.Path(root)
    path=ROOT/'src/agents/dqn_agent.py'
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StreamDQNAgent')
    update=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='update')
    body=[]
    for n in update.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='losses' for t in n.targets):break
        body.append(n)
    fn=ast.FunctionDef(name='source_replay_construction_only',args=update.args,body=body,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));scope={};exec(compile(module,'<frozen_replay_construction_prefix>','exec'),scope)
    class Probe:
        def __init__(self):self.replay=deque(maxlen=4096)
        def _replay_data(self,data,index):return {'physical_decision_index':data['physical_decision_index']}
    p=Probe()
    # Sequence 0..4096, with actual source's terminal reset and MC difference logic.
    rows=[{'data':{'physical_decision_index':i},'vehicle_ids':['v'], 'actions':torch.zeros(1,3,dtype=torch.long),
           'returns':torch.tensor([(4097-i)*8.]),'initial_request_count':8,'episode_end':i==4096} for i in range(4097)]
    scope['source_replay_construction_only'](p,rows)
    retained=[r[0]['physical_decision_index'] for r in p.replay]
    assert set(retained)==set(range(4096)) and 4096 not in retained
    assert not any(r[4] for r in p.replay)
    # Reconstruct actual episode-level inclusion exactly from decision-row counts.
    run=ROOT/'artifacts/joint_stream_learning/20261016T162100Z_confirmation/dqn'
    training_path=run/'training.jsonl'
    training_hash=hashlib.sha256(training_path.read_bytes()).hexdigest()
    training_manifest=json.loads((run/'manifest.json').read_text())
    assert training_hash==training_manifest['output_sha256']['training.jsonl']
    protocol=json.loads((run/'../protocol/manifest.json').resolve().read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol['source_fingerprints']['src/agents/dqn_agent.py']
    training=[json.loads(l) for l in training_path.read_text().splitlines()]
    by={seed:sorted([r for r in training if r['seed']==seed],key=lambda r:r['episode']) for seed in (7,17,27)}
    updates=[]
    for seed,episodes in by.items():
        for offset in range(0,len(episodes),4):
            pack=episodes[offset:offset+4]
            counts=[int(r['decision_rows']) for r in pack]
            capacity=4096;kept=[]
            for count in counts:
                n=min(count,capacity);kept.append(n);capacity-=n
            updates.append({'seed':seed,'update_episode':pack[-1]['episode'],
                'episodes':[r['episode'] for r in pack], 'decision_rows':counts,
                'total_new_decisions':sum(counts),'overflow':max(0,sum(counts)-4096),
                'retained_from_new_packet':kept,'discarded_from_new_packet':[n-k for n,k in zip(counts,kept)],
                'fully_discarded_episodes':[r['episode'] for r,n in zip(pack,kept) if n==0]})
    out={'reviewed_at':datetime.now(timezone.utc).isoformat(),'source_commit':'35975cb3d73c84ddbabf01f6749bbc2d60c826de',
        'evidence':'verified frozen-source FIFO insertion-order defect plus raw-log decision-count reconstruction',
        'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'training_log_sha256':training_hash,'training_manifest_hash_verified':True,'source_protocol_hash_verified':True,
        'environment_steps':0,'optimizer_steps':0,'update_calls':0,
        'synthetic_default_capacity':{'capacity':4096,'input_decisions':4097,'retained_first_storage_index':retained[0],
            'retained_last_storage_index':retained[-1],'earliest_retained':min(retained),'latest_retained':max(retained),
            'newest_dropped':4096 not in retained,'terminal_transitions_retained':sum(r[4] for r in p.replay)},
        'actual_updates':updates,'summary':{'updates':len(updates),'overflow_updates':sum(r['overflow']>0 for r in updates),
            'new_decisions':sum(r['total_new_decisions'] for r in updates),'discarded_before_sampling':sum(r['overflow'] for r in updates),
            'fully_discarded_episodes':sum(len(r['fully_discarded_episodes']) for r in updates),
            'final_episode64_kept_by_seed':{r['seed']:r['retained_from_new_packet'][-1] for r in updates if r['update_episode']==64},
            'final_episode64_decisions_by_seed':{r['seed']:r['decision_rows'][-1] for r in updates if r['update_episode']==64}}}
    output=pathlib.Path(output_root)/'ppo_mec_stage2_dqn_replay_order.json';output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'summary':out['summary'],'synthetic':out['synthetic_default_capacity'],'first_update':updates[0]},indent=2))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_root", type=Path, required=True,
        help="Directory for the two new JSON receipts; existing receipts are never overwritten.",
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1],
        help="Repository root containing the frozen confirmation artifacts.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = arguments(argv)
    root = args.root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not (root / "src/agents/dqn_agent.py").is_file():
        raise FileNotFoundError(f"Not a PPO_MEC repository root: {root}")
    for name in OUTPUT_NAMES:
        if (output_root / name).exists():
            raise FileExistsError(f"Refusing to overwrite an existing audit receipt: {output_root / name}")
    output_root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    import torch

    # The original diagnostic blocks Adam.step. Restore the process-global
    # method even on audit failure, so importing/calling this CLI is contained.
    original_adam_step = torch.optim.Adam.step
    try:
        audit_learning_chain(root, output_root)
        audit_dqn_replay_order(root, output_root)
    finally:
        torch.optim.Adam.step = original_adam_step


if __name__ == "__main__":
    main()
