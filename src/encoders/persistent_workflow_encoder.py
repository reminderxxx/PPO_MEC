"""Shared causal features for the opt-in preparation comparison."""
import torch
from torch import nn
from src.encoders.fusion_encoder import SurrogateFusionEncoder
from src.envs.wrappers.persistent_workflow_control import candidates, CONTROL_CONTRACT
from src.envs.core.persistent_workflow_env import ADAPTERS

OBSERVATION_CONTRACT="persistent_preparation_features_v1"
FEATURE_DIM=6+5*13


def persistent_features(state):
    stream=state["request_stream"]
    count=max(len(stream["requests"]),1)
    completed=set(state["workflow"].get("completed_node_ids",[]))
    total=max(len(state["workflow"].get("nodes",[])),1)
    result=[stream["pending_request_count"]/8.,stream["remaining_horizon_steps"]/max(stream["horizon_steps"],1),
            len(stream["requests"])/8.,len(completed)/total,
            len(state.get("model_loading",[]))/10.,sum(x["remaining_seconds"] for x in state.get("model_loading",[]))/10.]
    rsus={r["rsu_id"]:r for r in state.get("rsus",[])}
    runtime={r["rsu_id"]:r for r in state.get("cache_runtime",[])}
    for c in candidates(state):
        r=rsus.get(c["rsu_id"],{}); cache=runtime.get(c["rsu_id"],{})
        loading=[v["remaining_seconds"] for v in state.get("model_loading",[]) if v["rsu_id"]==c["rsu_id"]]
        result.extend([float(c["adapter_id"]==a) for a in ADAPTERS])
        result.extend([float(c["rsu_id"] is not None),float(c["adapter_id"] in r.get("cached_adapter_ids",[])),
            max(loading,default=0.)/10.,cache.get("queue_seconds",0.)/10.,cache.get("resident_mb",0.)/512.,
            cache.get("capacity_mb",0.)/512.,float(c["action_id"]==3),float(bool(c["adapter_id"]))])
    assert len(result)==FEATURE_DIM
    return torch.tensor(result,dtype=torch.float32)


class PersistentWorkflowEncoder(nn.Module):
    def __init__(self,hidden_dim=64,use_graph=True):
        super().__init__()
        self.base=SurrogateFusionEncoder(hidden_dim=hidden_dim,use_uncertainty_signal=False,use_dependency_aware=use_graph)
        self.context=nn.Sequential(nn.Linear(FEATURE_DIM,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,hidden_dim))
        self.fusion=nn.Linear(hidden_dim*2,hidden_dim)

    def forward(self,state):
        if state.get("persistent_control_contract")!=CONTROL_CONTRACT:raise ValueError("Preparation observation contract mismatch")
        out=self.base(state)
        features=persistent_features(state).to(next(self.parameters()).device)
        extra=self.context(features)
        for name in ["shared_embedding","slow_context","fast_context","event_context","critic_context"]:
            out[name]=torch.tanh(self.fusion(torch.cat([out[name],extra])))
        out["centralized_critic_context"]=out["critic_context"]
        return out
