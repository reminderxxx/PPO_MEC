"""Replay every stochastic action and PPO update from saved states, no simulator."""
import argparse
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.agents.ppo_agent import EventJointPPOAgent
from src.encoders.event_joint_encoder import event_features
from scripts.run_event_joint_learning import event_available
from scripts.run_ready_workflow_learning import rule_action,node_runtime
from scripts.run_persistent_workflow_learning import dump,sha


def replay(cell,out):
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);receipt=json.loads((cell/'completion.json').read_text());training=json.loads((cell/'training.json').read_text())
    if receipt['status']!='complete':raise ValueError('Cell is incomplete')
    a,payload=EventJointPPOAgent.load(cell/'checkpoint_0.pt');expected_updates=json.loads((cell/'updates.json').read_text())
    pending=[];native_rows=0;decisions=0;checks=[];max_reward_error=0.;max_metric_error=0.;global_reward=0.
    for ep in range(1,receipt['episodes']+1):
        torch.manual_seed(a.seed*100000+ep-1);trace=json.load(gzip.open(cell/f'train_{ep}.json.gz','rt'));rows=[];episode_reward=0.
        for item in trace:
            s=item['state'];primitive=a.controller=='primitive';event=primitive or event_available(s,a.eviction_aware)
            if event!=item['event']:raise AssertionError('Decision-boundary replay mismatch')
            node=rule_action(s,'cache_first')
            if node!=item['node_id'] or node not in s['ready_node_ids']:raise AssertionError('Service selection mismatch')
            prep=None
            if event:
                f=event_features(s,a.eviction_aware);action,lp,value=a.act(f)
                if action!=item['action']:raise AssertionError(f'Stochastic action replay mismatch at ep{ep}')
                prep=f['ids'][action];decisions+=1
                rows.append(dict(features=f,action=action,log_prob=lp,value=value,reward=0.,raw_reward=0.,holding_steps=0))
            else:
                vehicle=next(v for v in s['vehicles'] if v['vehicle_id']==s['primary_vehicle_id']);rsu=vehicle.get('associated_rsu_id')
                loading=any(x['rsu_id']==rsu and x['remaining_seconds']>0 for x in s['model_loading']);by_id={n['node_id']:n for n in s['workflow']['nodes']}
                if rsu and not loading and not node_runtime(s,by_id[node])['present']:prep=by_id[node]['required_adapter']
            if prep!=item['prepare_adapter']:raise AssertionError('Executed preparation mismatch')
            raw=item['reward'];stream=item['stream'];terms=stream['reward_terms'];error=abs(raw-sum(terms.values()));max_reward_error=max(max_reward_error,error)
            if error>1e-8 or abs(raw-stream['reward'])>1e-8:raise AssertionError('Raw reward accounting mismatch')
            transfer=sum(stream['cache_event'].get('transfer_mb_by_type',{}).values())
            if abs(transfer-stream['transfer_mb'])>1e-8 or abs(terms['model_transfer']+.25*transfer/1024.)>1e-8:raise AssertionError('Transfer charge mismatch')
            if abs(terms['pending_time']+stream['pending_area_seconds'])>1e-8:raise AssertionError('Pending cost mismatch')
            episode_reward+=raw;native_rows+=1
            if rows:rows[-1]['reward']+=raw/20.;rows[-1]['raw_reward']+=raw;rows[-1]['holding_steps']+=1
        if abs(episode_reward-training[ep-1]['reward'])>1e-8:raise AssertionError('Episode total mismatch')
        value=0.
        for row in reversed(rows):value+=row['reward'];row['return']=value;row['advantage']=value-row['value']
        pending+=rows;global_reward+=episode_reward
        if ep%4==0 and pending:
            metrics=a.learn(pending);pending=[];expected=expected_updates[a.updates-1]
            for key in ['policy_loss','value_loss','entropy']:
                delta=abs(metrics[key]-expected[key]);max_metric_error=max(max_metric_error,delta)
                if delta>1e-7:raise AssertionError(f'PPO update metric drift: {key} {delta}')
        checkpoint=cell/f'checkpoint_{ep}.pt'
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location='cpu')
            max_delta=max(float((v-saved['network'][k]).abs().max()) for k,v in a.network.state_dict().items())
            if max_delta>0.:raise AssertionError(f'Checkpoint weights differ at episode{ep}: {max_delta}')
            if a.updates!=saved['updates']:raise AssertionError('Update count mismatch')
            checks.append(dict(episode=ep,checkpoint_sha256=sha(checkpoint),max_parameter_error=max_delta,updates=a.updates))
    if native_rows!=receipt['native_steps']:raise AssertionError('Recorded physical-step count mismatch')
    result=dict(status='PASS',cell=str(cell),initial_checkpoint_sha256=sha(cell/'checkpoint_0.pt'),recorded_native_rows=native_rows,
        new_environment_interactions=0,stochastic_decisions_replayed=decisions,ppo_updates_replayed=a.updates,checkpoints=checks,
        max_reward_accounting_error=max_reward_error,max_update_metric_error=max_metric_error,global_training_reward=global_reward,
        scope='Saved-state stochastic action and exact parameter/update reproduction; no claim of independent physical rollout')
    out.parent.mkdir(parents=True,exist_ok=True);dump(out,result);print(json.dumps({k:v for k,v in result.items() if k!='checkpoints'}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--output',required=True);a=p.parse_args();replay(Path(a.cell),Path(a.output))
