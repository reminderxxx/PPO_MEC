"""Causal adapter candidates for interruptible joint-control PPO.

Graph/set variants share the exact candidate mask, local inputs and parameters.
"""
import torch
from torch import nn
from src.encoders.ready_workflow_encoder import ready_features,batch_ready_features
from src.envs.core.persistent_workflow_env import ADAPTERS

EVENT_FEATURE_VERSION='event_joint_adapter_features_v1'
EVICTION_FEATURE_VERSION='event_joint_eviction_features_v2'


def event_features(state,eviction_aware=False):
    graph=ready_features(state);nodes=state['workflow']['nodes'];done=set(state['workflow']['completed_node_ids']);ready=set(state['ready_node_ids'])
    remaining={r['request_id']:len(set(r['node_ids'])-done) for r in state['request_stream']['requests']}
    vehicle=next(v for v in state['vehicles'] if v['vehicle_id']==state['primary_vehicle_id'])
    rsu=next((r for r in state['rsus'] if r['rsu_id']==vehicle.get('associated_rsu_id')),{});cached=set(rsu.get('cached_adapter_ids',[]))
    pool=torch.zeros(6,len(nodes));aux=[];mask=[]
    for i,adapter in enumerate(ADAPTERS):
        unfinished=[n for n in nodes if n['node_id'] not in done and n['required_adapter']==adapter]
        frontier=[n for n in unfinished if n['node_id'] in ready]
        for j,n in enumerate(nodes):pool[i,j]=float(n in unfinished)
        aux.append([len(unfinished)/20.,len(frontier)/8.,min((remaining[n['node_id'].split(':')[0]] for n in frontier),default=20)/20.,
            min((len(set(n['predecessors'])-done) for n in unfinished),default=8)/8.,float(adapter in cached),float(not unfinished)])
        mask.append(bool(frontier and adapter not in cached and vehicle.get('associated_rsu_id')))
    if eviction_aware:
        observation=state.get('joint_cache_observation',{})
        if observation.get('version')!='observable_lru_effects_v1':raise ValueError('Missing observable eviction contract')
        by_id={n['node_id']:n for n in nodes}
        rank={n:i for i,n in enumerate(state['workflow']['execution_order'])}
        def service_key(n):
            loading=max((x['remaining_seconds'] for x in state['model_loading'] if x['rsu_id']==vehicle.get('associated_rsu_id') and x['object_id'] in ['base:veh_base_v1','adapter:'+n['required_adapter'].removeprefix('adapter_')]),default=0.)
            present=n['required_adapter'] in cached
            return (not(present and loading<=1e-9),not present,loading,remaining[n['node_id'].split(':')[0]],rank[n['node_id']])
        selected=min([by_id[n] for n in ready],key=service_key)
        protected='adapter:'+selected['required_adapter'].removeprefix('adapter_')
        for i,adapter in enumerate(ADAPTERS):
            effect=observation['candidate_effects'].get(adapter,dict(victims=[],sufficient=False,transfer_mb=0.,required_free_mb=0.))
            victims=set(effect['victims'])
            affected=[n for n in nodes if n['node_id'] not in done and 'adapter:'+n['required_adapter'].removeprefix('adapter_') in victims]
            destroys_current=protected in victims
            aux[i]+=[effect['transfer_mb']/100.,effect['required_free_mb']/100.,len(victims)/5.,
                sum(n['node_id'] in ready for n in affected)/8.,len(affected)/20.,float(destroys_current)]
            # Explicit shared proposal constraint, never a silent action rewrite.
            mask[i]=mask[i] and effect['sufficient'] and not destroys_current
    aux.append([0.,0.,1.,1.,0.,1.]+([0.]*6 if eviction_aware else []));mask.append(True)
    return dict(graph=graph,pool=pool,aux=torch.tensor(aux,dtype=torch.float32),mask=torch.tensor(mask,dtype=torch.bool),ids=list(ADAPTERS)+[None])


def batch_event_features(items):
    graph=batch_ready_features([d['graph'] for d in items]);n=graph['x'].shape[1]
    pool=torch.zeros(len(items),6,n)
    for i,d in enumerate(items):pool[i,:,:d['pool'].shape[1]]=d['pool']
    return dict(graph=graph,pool=pool,aux=torch.stack([d['aux'] for d in items]),mask=torch.stack([d['mask'] for d in items]))


class EventJointActorCritic(nn.Module):
    def __init__(self,use_graph=True,aux_dim=6):
        super().__init__();self.use_graph=bool(use_graph)
        self.node=nn.Sequential(nn.Linear(17,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        self.succ=nn.Linear(32,32,bias=False);self.pred=nn.Linear(32,32,bias=False)
        self.message_gate=nn.Parameter(torch.zeros(()))
        self.actor=nn.Sequential(nn.Linear(72+aux_dim,32),nn.Tanh(),nn.Linear(32,1))
        self.critic=nn.Sequential(nn.Linear(25,32),nn.Tanh(),nn.Linear(32,1))

    def forward(self,b):
        g=b['graph'];h=self.node(g['x'])
        if self.use_graph:
            for _ in range(2):h=h+self.message_gate.tanh()*torch.tanh(self.succ(torch.bmm(g['succ'],h))+self.pred(torch.bmm(g['pred'],h)))
        weights=g['present'].unsqueeze(-1);pooled=(h*weights).sum(1)/weights.sum(1).clamp_min(1)
        candidates=torch.bmm(b['pool'],h)/b['pool'].sum(-1,keepdim=True).clamp_min(1)
        context=torch.cat([pooled,g['context']],-1).unsqueeze(1).expand(-1,6,-1)
        logits=self.actor(torch.cat([candidates,context,b['aux']],-1)).squeeze(-1).masked_fill(~b['mask'],-1e9)
        raw=(g['x']*weights).sum(1)/weights.sum(1).clamp_min(1)
        return logits,self.critic(torch.cat([raw,g['context']],-1)).squeeze(-1)

    def actor_parameters(self):
        return [p for n,p in self.named_parameters() if not n.startswith('critic.')]
