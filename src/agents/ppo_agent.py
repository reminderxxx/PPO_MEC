"""Basic PPO baseline over the shared semantic action contract."""

from __future__ import annotations

from typing import Any

from src.agents.sa_ghmappo_core import 分层PPO基类 as PPOBaseAgent


class PPOAgent(PPOBaseAgent):
    """Single-agent PPO with a flat semantic encoder and independent critic."""

    observation_contract = "flat_semantic_encoder_v1"
    action_contract = "semantic_discrete_5"
    support_level = "trainable"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            agent_name="ppo",
            policy_type="ppo_policy",
            encoder_kind="flat",
            centralized_critic=False,
            hierarchical_conditioning=False,
            use_hierarchy=False,
            use_prediction_features=False,
            use_uncertainty_signal=False,
            use_dependency_aware=False,
            graph_continuity_critic_enabled=False,
            uncertainty_aware_event_scaling_enabled=False,
            uncertainty_aware_critic_enabled=False,
            event_head_enabled=False,
            adapter_prefetch_enabled=True,
            auxiliary_coef=0.0,
            event_entropy_coef_scale=1.0,
            event_entropy_credit_floor=0.0,
            event_policy_credit_floor=0.0,
            event_advantage_blend=1.0,
            event_logit_temperature=1.0,
            event_logit_temperature_final=1.0,
            event_temperature_decay_updates=0,
            event_logit_sharpening_final_scale=1.0,
            event_logit_sharpening_timing_gain=0.0,
            event_actor_loss_extra_gain=1.0,
            event_prepare_margin_boost=0.0,
            temporal_consistency_coef=0.0,
            **kwargs,
        )
        self.baseline_config = {
            "family": "ppo",
            "flat_policy": True,
            "graph_encoder": False,
            "hierarchy": False,
            "surrogate_enhanced_head": False,
            "centralized_critic": False,
        }


class ReadyWorkflowPPOAgent:
    """Experimental variable-candidate PPO; deliberately outside the live registry.

    No expert, action guard, shaping or graph-specific reward. The set ablation
    has identical parameters, features and PPO updates; it skips edge messages.
    """
    def __init__(self, random_seed=7, use_graph=True, architecture=None):
        import torch
        from src.encoders.ready_workflow_encoder import ReadyWorkflowActorCritic, MODEL_VERSION, batch_ready_features
        self.seed=int(random_seed);self.use_graph=bool(use_graph)
        torch.manual_seed(self.seed)
        self.network=ReadyWorkflowActorCritic(use_graph=self.use_graph,architecture=architecture or MODEL_VERSION)
        self.optimizer=torch.optim.Adam(self.network.parameters(),lr=3e-4)
        self.updates=0
        self._batch_features=batch_ready_features

    def act(self,features,deterministic=False):
        import torch
        from src.encoders.ready_workflow_encoder import batch_ready_features
        with torch.no_grad():
            logits,value=self.network(self._batch_features([features]))
            distribution=torch.distributions.Categorical(logits=logits[0])
            action=logits[0].argmax() if deterministic else distribution.sample()
        return int(action),float(distribution.log_prob(action)),float(value[0])

    def learn(self,rows):
        import torch
        from src.encoders.ready_workflow_encoder import batch_ready_features
        adv=torch.tensor([r['advantage'] for r in rows]);adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        returns=torch.tensor([r['return'] for r in rows]);old=torch.tensor([r['log_prob'] for r in rows])
        actions=torch.tensor([r['action'] for r in rows]);metrics=[]
        for _ in range(4):
            for idx in torch.randperm(len(rows)).split(128):
                logits,values=self.network(self._batch_features([rows[int(i)]['features'] for i in idx]))
                distribution=torch.distributions.Categorical(logits=logits)
                logp=distribution.log_prob(actions[idx]);ratio=(logp-old[idx]).exp()
                surrogate=torch.minimum(ratio*adv[idx],ratio.clamp(.8,1.2)*adv[idx])
                choices=torch.tensor([int(rows[int(i)]['features']['mask'].sum())>1 for i in idx])
                policy=-surrogate[choices].mean() if choices.any() else surrogate.sum()*0.
                entropy=distribution.entropy()[choices].mean() if choices.any() else policy*0.
                value_loss=(values-returns[idx]).square().mean()
                loss=policy+.5*value_loss-.02*entropy
                if not torch.isfinite(loss):raise ValueError('Non-finite PPO objective')
                self.optimizer.zero_grad();loss.backward()
                if getattr(self,'_separate_value_clip',False):
                    torch.nn.utils.clip_grad_norm_(self.network.actor_parameters(),.5)
                    torch.nn.utils.clip_grad_norm_(self.network.critic.parameters(),.5)
                else:torch.nn.utils.clip_grad_norm_(self.network.parameters(),.5)
                self.optimizer.step()
                metrics.append((float(policy.detach()),float(value_loss.detach()),float(entropy.detach())))
        self.updates+=1
        return dict(update=self.updates,rows=len(rows),policy_loss=sum(x[0] for x in metrics)/len(metrics),value_loss=sum(x[1] for x in metrics)/len(metrics),entropy=sum(x[2] for x in metrics)/len(metrics))

    def save(self,path,provenance):
        import torch
        from pathlib import Path
        from src.envs.wrappers.ready_workflow_control import READY_CONTRACT
        from src.encoders.ready_workflow_encoder import FEATURE_VERSION
        if Path(path).exists():raise FileExistsError(path)
        torch.save(dict(contract=READY_CONTRACT,features=FEATURE_VERSION,architecture=self.network.architecture,seed=self.seed,use_graph=self.use_graph,
            network=self.network.state_dict(),optimizer=self.optimizer.state_dict(),updates=self.updates,
            rng=torch.random.get_rng_state(),provenance=provenance),path)

    @classmethod
    def load(cls,path):
        import torch
        from src.envs.wrappers.ready_workflow_control import READY_CONTRACT
        from src.encoders.ready_workflow_encoder import FEATURE_VERSION
        p=torch.load(path,map_location='cpu')
        if p.get('contract')!=READY_CONTRACT or p.get('features')!=FEATURE_VERSION:
            raise ValueError('Ready-node checkpoint contract mismatch')
        a=cls(p['seed'],p['use_graph'],p.get('architecture','ready_propagated_actor_v1'));a.network.load_state_dict(p['network']);a.optimizer.load_state_dict(p['optimizer']);a.updates=p['updates']
        return a,p


class RequestOptionPPOAgent(ReadyWorkflowPPOAgent):
    """Experimental request-boundary PPO with a fixed causal fast executor."""
    def __init__(self,random_seed=7):
        import torch
        from src.encoders.request_option_encoder import RequestOptionActorCritic,batch_option_features
        self.seed=int(random_seed);torch.manual_seed(self.seed)
        self.network=RequestOptionActorCritic()
        self.optimizer=torch.optim.Adam(self.network.parameters(),lr=1e-3)
        self.updates=0;self._batch_features=batch_option_features;self._separate_value_clip=True

    def save(self,path,provenance):
        import torch
        from pathlib import Path
        from src.encoders.request_option_encoder import OPTION_CONTRACT,OPTION_FEATURE_VERSION
        if Path(path).exists():raise FileExistsError(path)
        torch.save(dict(contract=OPTION_CONTRACT,features=OPTION_FEATURE_VERSION,seed=self.seed,
            network=self.network.state_dict(),optimizer=self.optimizer.state_dict(),updates=self.updates,
            rng=torch.random.get_rng_state(),provenance=provenance),path)

    @classmethod
    def load(cls,path):
        import torch
        from src.encoders.request_option_encoder import OPTION_CONTRACT,OPTION_FEATURE_VERSION
        p=torch.load(path,map_location='cpu')
        if p.get('contract')!=OPTION_CONTRACT or p.get('features')!=OPTION_FEATURE_VERSION:raise ValueError('Request-option checkpoint contract mismatch')
        a=cls(p['seed']);a.network.load_state_dict(p['network']);a.optimizer.load_state_dict(p['optimizer']);a.updates=p['updates']
        return a,p


class EventJointPPOAgent(ReadyWorkflowPPOAgent):
    """Opt-in event-driven adapter preparation; separate from original SA-GHMAPPO."""
    def __init__(self,random_seed=7,use_graph=True,eviction_aware=False):
        import torch
        from src.encoders.event_joint_encoder import EventJointActorCritic,batch_event_features
        self.seed=int(random_seed);self.use_graph=bool(use_graph);torch.manual_seed(self.seed)
        self.eviction_aware=bool(eviction_aware)
        self.network=EventJointActorCritic(self.use_graph,12 if self.eviction_aware else 6)
        self.optimizer=torch.optim.Adam(self.network.parameters(),lr=1e-3)
        self.updates=0;self._batch_features=batch_event_features;self._separate_value_clip=True

    def save(self,path,provenance):
        import torch
        from pathlib import Path
        from src.encoders.event_joint_encoder import EVENT_FEATURE_VERSION,EVICTION_FEATURE_VERSION
        from src.envs.wrappers.joint_workflow_control import JOINT_CONTRACT
        if Path(path).exists():raise FileExistsError(path)
        torch.save(dict(contract=JOINT_CONTRACT,features=EVICTION_FEATURE_VERSION if self.eviction_aware else EVENT_FEATURE_VERSION,physical_contract='joint_service_hit_lru_v1' if self.eviction_aware else 'legacy_admission_recency_v1',seed=self.seed,use_graph=self.use_graph,
            network=self.network.state_dict(),optimizer=self.optimizer.state_dict(),updates=self.updates,
            rng=torch.random.get_rng_state(),provenance=provenance),path)

    @classmethod
    def load(cls,path):
        import torch
        from src.encoders.event_joint_encoder import EVENT_FEATURE_VERSION,EVICTION_FEATURE_VERSION
        from src.envs.wrappers.joint_workflow_control import JOINT_CONTRACT
        p=torch.load(path,map_location='cpu')
        if p.get('contract')!=JOINT_CONTRACT or p.get('features') not in [EVENT_FEATURE_VERSION,EVICTION_FEATURE_VERSION]:raise ValueError('Event-joint checkpoint contract mismatch')
        expected_physical='joint_service_hit_lru_v1' if p['features']==EVICTION_FEATURE_VERSION else 'legacy_admission_recency_v1'
        if p.get('physical_contract','legacy_admission_recency_v1')!=expected_physical:raise ValueError('Event-joint physical contract mismatch')
        a=cls(p['seed'],p['use_graph'],p['features']==EVICTION_FEATURE_VERSION);a.network.load_state_dict(p['network']);a.optimizer.load_state_dict(p['optimizer']);a.updates=p['updates']
        return a,p
