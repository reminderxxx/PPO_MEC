"""Observable request summaries for the opt-in temporal decision experiment.

This is an engineered request encoder, not a learned graph contribution.
"""
import torch
from torch import nn
from src.encoders.ready_workflow_encoder import ready_features
from src.envs.core.persistent_workflow_env import ADAPTERS

OPTION_CONTRACT='released_request_option_v1'
OPTION_FEATURE_VERSION='request_option_features_v1'
OPTION_DIM=17


def option_features(state):
    # Only decision-boundary state is encoded; no hidden active-option aliasing.
    context=ready_features(state)['context'];nodes={n['node_id']:n for n in state['workflow']['nodes']}
    done=set(state['workflow']['completed_node_ids']);ready=set(state['ready_node_ids'])
    vehicle=next(v for v in state['vehicles'] if v['vehicle_id']==state['primary_vehicle_id']);rsu_id=vehicle.get('associated_rsu_id')
    rsu=next((r for r in state['rsus'] if r['rsu_id']==rsu_id),{});cached=set(rsu.get('cached_adapter_ids',[]))
    ids=[];rows=[]
    def depth(n):return 0 if n in done else 1+max((depth(s) for s in nodes[n]['successors']),default=0)
    for request in state['request_stream']['requests']:
        unfinished=[nodes[n] for n in request['node_ids'] if n not in done]
        if not unfinished:continue
        frontier=[n for n in unfinished if n['node_id'] in ready]
        def loading(n):return max((x['remaining_seconds'] for x in state['model_loading'] if x['rsu_id']==rsu_id and x['object_id'] in ['base:veh_base_v1','adapter:'+n['required_adapter'].removeprefix('adapter_')]),default=0.)
        k=len(unfinished);ids.append(request['request_id'])
        rows.append([sum(n['required_adapter']==a for n in unfinished)/k for a in ADAPTERS]+[
            k/20.,len(request['node_ids'])/20.,max(depth(n['node_id']) for n in unfinished)/20.,len(frontier)/8.,
            sum(n['required_adapter'] in cached and loading(n)<=1e-9 for n in frontier)/8.,max((loading(n) for n in frontier),default=0.)/10.,
            sum(n['required_adapter'] in cached for n in unfinished)/k,len({n['required_adapter'] for n in unfinished}-cached)/5.,
            sum(n['input_size'] for n in unfinished)/k/20.,sum(n['output_size'] for n in unfinished)/k/2000.,
            float(bool(rsu_id)),sum(len(n['predecessors']) for n in unfinished)/k/4.])
    return dict(x=torch.tensor(rows,dtype=torch.float32),context=context,mask=torch.ones(len(ids),dtype=torch.bool),ids=ids)


def batch_option_features(items):
    n=max(len(d['ids']) for d in items);b=len(items)
    result=dict(x=torch.zeros(b,n,OPTION_DIM),context=torch.stack([d['context'] for d in items]),mask=torch.zeros(b,n,dtype=torch.bool),present=torch.zeros(b,n,dtype=torch.bool))
    for i,d in enumerate(items):
        k=len(d['ids']);result['x'][i,:k]=d['x'];result['mask'][i,:k]=d['mask'];result['present'][i,:k]=True
    return result


class RequestOptionActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_node=nn.Sequential(nn.Linear(OPTION_DIM,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        self.actor=nn.Sequential(nn.Linear(72,32),nn.Tanh(),nn.Linear(32,1))
        # Independent value representation; value loss cannot rewrite the actor.
        self.critic=nn.Sequential(nn.Linear(25,32),nn.Tanh(),nn.Linear(32,1))

    def forward(self,batch):
        h=self.actor_node(batch['x']);weights=batch['present'].unsqueeze(-1)
        pooled=(h*weights).sum(1)/weights.sum(1).clamp_min(1)
        context=torch.cat([pooled,batch['context']],-1)
        logits=self.actor(torch.cat([h,context.unsqueeze(1).expand(-1,h.shape[1],-1)],-1)).squeeze(-1).masked_fill(~batch['mask'],-1e9)
        raw_mean=(batch['x']*weights).sum(1)/weights.sum(1).clamp_min(1)
        return logits,self.critic(torch.cat([raw_mean,batch['context']],-1)).squeeze(-1)

    def actor_parameters(self):return list(self.actor_node.parameters())+list(self.actor.parameters())
