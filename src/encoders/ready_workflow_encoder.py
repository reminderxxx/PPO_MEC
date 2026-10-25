"""Action-conditioned node features and directed propagation for ready scheduling."""
import torch
from torch import nn
from src.envs.core.persistent_workflow_env import ADAPTERS
from src.envs.wrappers.ready_workflow_control import READY_CONTRACT

FEATURE_VERSION = 'ready_node_features_v1'
MODEL_VERSION = 'ready_detached_actor_v3'
RESIDUAL_MODEL_VERSION = 'ready_residual_actor_v2'
LEGACY_MODEL_VERSION = 'ready_propagated_actor_v1'
NODE_DIM = 17
GLOBAL_DIM = 8


def ready_features(state):
    if state.get('ready_scheduling_contract') != READY_CONTRACT:
        raise ValueError('Ready-node observation contract mismatch')
    workflow = state['workflow']; nodes = workflow['nodes']; ids = [n['node_id'] for n in nodes]
    index = {n:i for i,n in enumerate(ids)}; done = set(workflow['completed_node_ids']); ready = set(state['ready_node_ids'])
    request_by_node = {n:r for r in state['request_stream']['requests'] for n in r['node_ids']}
    v = next(v for v in state['vehicles'] if v['vehicle_id'] == state['primary_vehicle_id']); rsu_id = v.get('associated_rsu_id')
    rsu = next((r for r in state['rsus'] if r['rsu_id'] == rsu_id), {})
    runtime = next((r for r in state['cache_runtime'] if r['rsu_id'] == rsu_id), {})
    rows=[]
    for n in nodes:
        req = request_by_node[n['node_id']]
        loading=max((x['remaining_seconds'] for x in state['model_loading'] if x['rsu_id']==rsu_id and x['object_id'] in ['base:veh_base_v1','adapter:'+n['required_adapter'].removeprefix('adapter_')]), default=0.)
        rows.append([float(n['required_adapter']==a) for a in ADAPTERS]+[
            n['input_size']/max(max(x['input_size'] for x in nodes),1),
            n['output_size']/max(max(x['output_size'] for x in nodes),1),
            float(n['node_id'] in done),float(n['node_id'] in ready),
            float(n['required_adapter'] in rsu.get('cached_adapter_ids',[])),loading/10.,
            len(set(req['node_ids'])-done)/20.,len(n['predecessors'])/4.,len(n['successors'])/4.,
            len(req['node_ids'])/20.,float(loading>0),float(bool(rsu_id))])
    stream=state['request_stream']
    context=[stream['remaining_horizon_steps']/max(stream['horizon_steps'],1),stream['pending_request_count']/8.,
        runtime.get('queue_seconds',0.)/10.,float(bool(rsu_id)),
        sum(x['remaining_seconds'] for x in state['model_loading'])/10.,
        runtime.get('capacity_mb',0.)/512.,runtime.get('resident_mb',0.)/512.,len(done)/max(len(nodes),1)]
    adj=torch.zeros(len(nodes),len(nodes))
    for a,b in workflow['edges']:adj[index[a],index[b]]=1.
    membership=torch.tensor([[float(request_by_node[a]['request_id']==request_by_node[b]['request_id']) for b in ids] for a in ids])
    return dict(x=torch.tensor(rows,dtype=torch.float32),context=torch.tensor(context,dtype=torch.float32),
        succ=adj/adj.sum(-1,keepdim=True).clamp_min(1),pred=adj.T/adj.T.sum(-1,keepdim=True).clamp_min(1),
        request=membership/membership.sum(-1,keepdim=True).clamp_min(1),mask=torch.tensor([i in ready for i in ids]),ids=ids)


def batch_ready_features(items):
    n=max(len(d['ids']) for d in items); batch=len(items)
    result=dict(x=torch.zeros(batch,n,NODE_DIM),context=torch.stack([d['context'] for d in items]),
        succ=torch.zeros(batch,n,n),pred=torch.zeros(batch,n,n),request=torch.zeros(batch,n,n),
        mask=torch.zeros(batch,n,dtype=torch.bool),present=torch.zeros(batch,n,dtype=torch.bool))
    for i,d in enumerate(items):
        k=len(d['ids']);result['x'][i,:k]=d['x'];result['mask'][i,:k]=d['mask'];result['present'][i,:k]=True
        for name in ['succ','pred','request']:result[name][i,:k,:k]=d[name]
    return result


class ReadyWorkflowActorCritic(nn.Module):
    def __init__(self,use_graph=True,hidden_dim=48,architecture=MODEL_VERSION):
        super().__init__();self.use_graph=bool(use_graph);self.architecture=architecture
        if architecture not in [MODEL_VERSION,RESIDUAL_MODEL_VERSION,LEGACY_MODEL_VERSION]:raise ValueError('Unknown ready-node architecture')
        if architecture!=LEGACY_MODEL_VERSION:self.message_gate=nn.Parameter(torch.zeros(()))
        self.node=nn.Sequential(nn.Linear(NODE_DIM,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,hidden_dim),nn.Tanh())
        self.self_update=nn.Linear(hidden_dim,hidden_dim)
        self.succ_update=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.pred_update=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.actor=nn.Sequential(nn.Linear(hidden_dim*(4 if architecture!=LEGACY_MODEL_VERSION else 3)+GLOBAL_DIM,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,1))
        self.critic=nn.Sequential(nn.Linear(hidden_dim+GLOBAL_DIM,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,1))

    def forward(self,batch):
        local=self.node(batch['x']);h=local
        for _ in range(2):
            update=self.self_update(h)
            if self.use_graph:
                message=self.succ_update(torch.bmm(batch['succ'],h))+self.pred_update(torch.bmm(batch['pred'],h))
                update=update+(self.message_gate.tanh()*message if self.architecture!=LEGACY_MODEL_VERSION else message)
            h=torch.tanh(h+update)
        weights=batch['present'].unsqueeze(-1)
        pooled=(h*weights).sum(1)/weights.sum(1).clamp_min(1)
        global_context=torch.cat([pooled,batch['context']],-1)
        request_h=torch.bmm(batch['request'],h)
        parts=[h,request_h,global_context.unsqueeze(1).expand(-1,h.shape[1],-1)]
        if self.architecture!=LEGACY_MODEL_VERSION:parts.insert(0,local)
        candidates=torch.cat(parts,-1)
        logits=self.actor(candidates).squeeze(-1).masked_fill(~batch['mask'],-1e9)
        critic_input=global_context.detach() if self.architecture==MODEL_VERSION else global_context
        return logits,self.critic(critic_input).squeeze(-1)
