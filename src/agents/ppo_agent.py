"""Basic PPO baseline over the shared semantic action contract."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

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


class TraceDrivenMultiRequestPPO(nn.Module):
    """Exploratory shared actor for per-vehicle RSU choices in the replay env.

    This has a separate observation/action contract from PPOAgent and is not
    registered as a formal PPO_MEC baseline.
    """

    feature_names = (
        "distance_over_radius",
        "nearest_association",
        "adapter_cached",
        "adapter_loading",
        "associated_demand_over_eight",
        "cache_occupancy",
        "remaining_nodes_over_twenty",
        "last_service_here",
        "frame_fraction",
        "unfinished_fraction",
    )

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        width = len(self.feature_names)
        self.actor = nn.Sequential(
            nn.Linear(width, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(width, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    @classmethod
    def encode(
        cls, observation: dict[str, Any], *, window_length: int
    ) -> tuple[list[str], list[str], torch.Tensor, torch.Tensor]:
        """Encode only the current frame; mask uncovered RSU choices."""
        from math import dist

        rsu_ids = sorted(observation["rsus"])
        active_ids = [
            vehicle_id for vehicle_id, item in sorted(observation["vehicles"].items())
            if not item["completed"] and item["covered_rsu_ids"]
        ]
        unfinished = sum(not item["completed"] for item in observation["vehicles"].values())
        rows: list[list[list[float]]] = []
        masks: list[list[bool]] = []
        for vehicle_id in active_ids:
            vehicle = observation["vehicles"][vehicle_id]
            row = []
            mask = []
            for rsu_id in rsu_ids:
                rsu = observation["rsus"][rsu_id]
                covered = rsu_id in vehicle["covered_rsu_ids"]
                mask.append(covered)
                adapter_id = vehicle["required_adapter"]
                radius = max(float(rsu["coverage_radius"]), 1e-6)
                row.append([
                    min(dist(
                        (vehicle["position_x"], vehicle["position_y"]),
                        (rsu["position_x"], rsu["position_y"]),
                    ) / radius, 2.0),
                    float(vehicle["associated_rsu_id"] == rsu_id),
                    float(adapter_id in rsu["cached_adapter_ids"]),
                    float(adapter_id in rsu["loading_adapter_ids"]),
                    float(rsu["associated_vehicle_count"]) / 8.0,
                    float(len(rsu["cached_adapter_ids"]) + len(rsu["loading_adapter_ids"]))
                    / max(float(rsu["adapter_slots"]), 1.0),
                    min(float(vehicle["remaining_nodes"]) / 20.0, 1.0),
                    float(vehicle["last_service_rsu_id"] == rsu_id),
                    float(observation["frame_index"]) / max(window_length - 1, 1),
                    float(unfinished) / max(len(observation["vehicles"]), 1),
                ])
            rows.append(row)
            masks.append(mask)
        if not active_ids:
            return active_ids, rsu_ids, torch.empty((0, len(rsu_ids), len(cls.feature_names))), torch.empty((0, len(rsu_ids)), dtype=torch.bool)
        return (
            active_ids,
            rsu_ids,
            torch.tensor(rows, dtype=torch.float32),
            torch.tensor(masks, dtype=torch.bool),
        )

    def distribution_and_value(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        if features.ndim != 3 or mask.shape != features.shape[:2] or not bool(mask.any(dim=1).all()):
            raise ValueError("features/mask must contain at least one legal RSU per request")
        logits = self.actor(features).squeeze(-1).masked_fill(~mask, -1.0e9)
        weights = mask.unsqueeze(-1).float()
        pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        values = self.critic(pooled).squeeze(-1)
        return Categorical(logits=logits), values

    @torch.no_grad()
    def act(
        self, features: torch.Tensor, mask: torch.Tensor, *, deterministic: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self.distribution_and_value(features, mask)
        actions = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions), values

# v3 is opt-in and has separate checkpoint/probability contracts from legacy PPO.
import numpy as np
from pathlib import Path


class ParameterizedControlPPO(nn.Module):
    """Exact autoregressive joint PPO over execution, cache, and state/priority.

    Target commitment is a public controller option, released on node change or
    coverage loss. Cache decisions use a shared four-frame period. Typed LRU is
    the common fallback; explicit victim selection is not learned in this version.
    """
    action_contract = 'joint_parameterized_control_v3'
    checkpoint_contract = 'parameterized_control_ppo_v1'
    max_nodes, max_rsus = 32, 4

    def __init__(self, *, method='mappo', surrogate_enabled=True, prediction_enabled=True,
                 learning_rate=3e-4, seed=7, productive_controls=False):
        super().__init__()
        torch.manual_seed(seed)
        self.method, self.surrogate_enabled = method, bool(surrogate_enabled and method == 'sa_ghmappo')
        self.prediction_enabled = prediction_enabled
        self.productive_controls = bool(productive_controls)
        feature_width = 20 if self.productive_controls else 16
        own_width = 20 if self.productive_controls else 16
        input_width = own_width+16+2*feature_width
        self.options = dict(method=method, surrogate_enabled=surrogate_enabled,
                            prediction_enabled=prediction_enabled, learning_rate=learning_rate, seed=seed, productive_controls=productive_controls)
        self.graph_encoder = nn.Sequential(nn.Linear(4, 16), nn.Tanh())
        self.graph_message = nn.Sequential(nn.Linear(32, 16), nn.Tanh())
        self.cache_encoder = nn.Sequential(nn.Linear(feature_width, 16), nn.Tanh())
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(input_width, 64), nn.Tanh(), nn.Linear(64, 1)) for _ in range(3)])
        self.surrogates = nn.ModuleList([nn.Sequential(nn.Linear(input_width, 32), nn.Tanh(), nn.Linear(32, 3)) for _ in range(3)])
        self.critic = nn.Sequential(nn.Linear(own_width+16, 64), nn.Tanh(), nn.Linear(64, 1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.reset_controller()

    def reset_controller(self):
        self.commitments = {}

    def _features(self, decision, frame):
        if decision.get('action_contract') != self.action_contract:
            raise ValueError('parameterized agent requires v3 decision contract')
        s = decision['semantic_state']; vid = s['primary_vehicle_id']; current = s['current_workflow_node']
        workflow = s['workflow']; nodes = workflow['nodes']; done = set(workflow['completed_node_ids'])
        if len(nodes) > self.max_nodes or len(s['rsus']) != self.max_rsus:
            raise ValueError('v3 learning contract supports <=32 nodes and exactly four RSUs')
        ids = sorted(r['rsu_id'] for r in s['rsus']); rsus = {r['rsu_id']: r for r in s['rsus']}
        pred = s['predictions'] if self.prediction_enabled else {}
        future = pred.get('predicted_next_rsu_by_vehicle', {}).get(vid)
        confidence = pred.get('prediction_confidence_by_vehicle', {}).get(vid, 0.)
        def feature(rid=None, node=None, kind='adapter', priority=0):
            r = rsus.get(rid, {}); node = node or current
            base = node['required_base_model']; adapter = node['required_adapter']
            cached = adapter in r.get('cached_adapter_ids', [])
            ready_base = base in r.get('ready_base_model_ids', [])
            features = [float(rid is not None), float(rid in s['reachable_execution_rsu_ids']),
                float(rid == s['execution_location']), float(rid == s['associations'][vid]),
                float(cached), float(adapter in r.get('loading_adapter_ids', [])), float(ready_base),
                float(r.get('occupied_bytes', 0))/max(r.get('capacity_bytes', 1), 1),
                float(r.get('service_request_count', r.get('associated_vehicle_count', 0)))/8, float(rid in s['input_received_targets']),
                float(rid == future)*confidence, float(pred.get('future_load', {}).get(rid, 0))/8,
                float(node['node_id'] == current['node_id']), float(node['input_size'])/32,
                float(kind == 'base_model'), priority/3]
            if self.productive_controls:
                features += [float(rid == site) for site in ids]
            return features
        own = [frame/120, len(done)/max(len(nodes), 1), (len(nodes)-len(done))/32,
            s['current_node_work_units_progress']/32, s['current_node_work_units_required']/32,
            s['execution_state_payload_bytes']/4096, s['execution_state_version']/120,
            float(s['current_node_parents_ready']), float(s['vehicle_adapter_ready']),
            float(bool(s['output_pending'])), len(s['input_received_targets'])/4,
            confidence, float(pred.get('dwell_time', {}).get(vid, 0))/3,
            float(s['shared_state'].get('available', False)),
            float(s['shared_state'].get('age_frames') or 0)/8, float(len(s['reachable_execution_rsu_ids']))/4]
        if self.productive_controls:
            vehicle = next(v for v in s['vehicles'] if v['vehicle_id']==vid)
            reference = rsus[ids[0]]; radius = max(reference['coverage_radius'], 1.)
            own += [sorted(v['vehicle_id'] for v in s['vehicles']).index(vid)/7,
                (vehicle['position_x']-reference['position_x'])/radius,
                (vehicle['position_y']-reference['position_y'])/radius,
                sum(r.get('service_request_count',0) for r in rsus.values())/8]
        graph = np.zeros((32, 4), np.float32); adjacency = np.zeros((32, 32), np.float32)
        index = {n['node_id']: j for j, n in enumerate(nodes)}
        for j, n in enumerate(nodes):
            graph[j] = [n['input_size']/32, n['output_size']/32, float(n['node_id'] in done), float(n['node_id']==current['node_id'])]
        for a, b in workflow['edges']:
            adjacency[index[b], index[a]] = 1
        adjacency /= np.maximum(adjacency.sum(1, keepdims=True), 1)
        targets = [None, *ids, 'vehicle']; target_features = [feature(r) for r in targets]
        target_features[-1] = [0., 0., float(s['execution_location']==f'vehicle:{vid}'), 0.,
                                  float(s['vehicle_adapter_ready']), 0., 1., 0., 0., 1., 0., 0., 1., current['input_size']/32, 0., 0.]
        if self.productive_controls:
            target_features[-1] += [0.]*4
        mask = [r in decision['control_space']['execution_targets'] or
                (r == 'vehicle' and decision['control_space']['vehicle_execution_available']) for r in targets]
        if self.productive_controls and any(mask[1:]):
            mask[0] = False
        old = self.commitments.get(vid)
        if old and old[0] == current['node_id'] and old[1] in targets and mask[targets.index(old[1])]:
            mask = [r == old[1] for r in targets]
        # Cache noop + every uncompleted known DAG base/adapter at every RSU.
        cache = [(None, None, 'adapter')]; cache_features = [feature()]; cache_mask = [True]
        for n in nodes:
            for rid in ids:
                for kind in ('base_model', 'adapter'):
                    r = rsus[rid]
                    ready = n['required_base_model'] in r.get('ready_base_model_ids', []) if kind=='base_model' else (
                        n['required_adapter'] in r.get('cached_adapter_ids', []) or n['required_adapter'] in r.get('loading_adapter_ids', []))
                    cache.append((rid, n['node_id'], kind)); cache_features.append(feature(rid, n, kind))
                    cache_mask.append(frame % 4 == 0 and n['node_id'] not in done and not ready)
        event = [(rid, priority) for rid in [None, *ids] for priority in range(4)]
        event_features = [feature(rid, priority=priority) for rid, priority in event]
        event_mask = [rid is None or (s['execution_state_version'] > 0 and
            s['state_copies_by_rsu'].get(rid, {}).get('version') != s['execution_state_version'] and
            rid != s['execution_location']) for rid, priority in event]
        critic = list(own)
        c = decision['centralized_critic_state']
        for r in sorted(c['rsus'], key=lambda r: r['rsu_id']):
            critic += [r.get('occupied_bytes', 0)/max(r.get('capacity_bytes', 1), 1),
                       r.get('associated_vehicle_count', 0)/8, len(r.get('loading_adapter_ids', []))/8,
                       sum(w['execution_location']==r['rsu_id'] for w in c['joint_workflows'].values())/8]
        arrays = dict(own=np.array(own, np.float32), graph=graph, adjacency=adjacency,
                      graph_mask=np.array([j<len(nodes) for j in range(32)]), critic=np.array(critic, np.float32))
        for j, (f, m) in enumerate(zip((target_features, cache_features, event_features), (mask, cache_mask, event_mask))):
            count = (6, len(cache), 20)[j]
            arrays[f'f{j}'] = np.zeros((count, len(own)), np.float32); arrays[f'f{j}'][:len(f)] = f
            arrays[f'm{j}'] = np.zeros(count, bool); arrays[f'm{j}'][:len(m)] = m
        return arrays, (targets, cache, event), vid, current['node_id']

    def distributions(self, data, *, actions=None, deterministic=False):
        own = data['own']; batch = len(own)
        if self.method == 'sa_ghmappo':
            encoded = self.graph_encoder(data['graph'])
            messages = torch.bmm(data['adjacency'], encoded)
            encoded = self.graph_message(torch.cat((encoded, messages), -1))
            weight = data['graph_mask'].float().unsqueeze(-1)
            context = (encoded*weight).sum(1)/weight.sum(1).clamp(min=1)
        elif self.method == 'cache_offload_drl':
            context = self.cache_encoder(data['f0'][:, 1:5]).mean(1)
        else:
            context = own.new_zeros(batch, 16)
        condition = own.new_zeros(batch, data['f0'].shape[-1]); selected, logs, entropies, surrogates = [], [], [], []
        for j in range(3):
            f = data[f'f{j}']; n = f.shape[1]
            x = torch.cat((own[:, None].expand(-1,n,-1), context[:,None].expand(-1,n,-1), f,
                           condition[:,None].expand(-1,n,-1)), -1)
            logits = self.heads[j](x).squeeze(-1)
            predicted = self.surrogates[j](x.detach()) if self.surrogate_enabled else logits.new_zeros(batch, n, 3)
            if self.surrogate_enabled:
                logits = logits + .3*(predicted[..., 0]-predicted[..., 1]-predicted[..., 2]).detach().tanh()
            mask = data[f'm{j}']
            if self.productive_controls and j == 1:
                # Resource binding is part of the conditional distribution, not a post-action override.
                matching = (f[...,16:] == condition[:,None,16:]).all(-1) & (f[...,12] == 1) & (f[...,14] == 0)
                eligible = matching & mask
                cold = (condition[:,0] == 1) & ~((condition[:,4] == 1) & (condition[:,6] == 1)) & (condition[:,5] == 0)
                gate = cold & eligible.any(1)
                mask = torch.where(gate[:,None], eligible, mask)
            dist = Categorical(logits=logits.masked_fill(~mask, -1e9))
            action = actions[:,j] if actions is not None else (dist.logits.argmax(-1) if deterministic else dist.sample())
            if not bool(mask.gather(1,action[:,None]).all()):
                raise ValueError('action outside stored behavior mask')
            selected.append(action); logs.append(dist.log_prob(action)); entropies.append(dist.entropy())
            surrogates.append(predicted[torch.arange(batch), action])
            condition = f[torch.arange(batch), action]
        return torch.stack(selected,1), torch.stack(logs,1).sum(1), torch.stack(entropies,1).sum(1), self.critic(data['critic']).squeeze(-1), torch.stack(surrogates,1)

    @torch.no_grad()
    def act_controls(self, decisions, frame, *, deterministic=False):
        from src.envs.core.multi_request_target_control_env import JointRequestControl
        items = [self._features(decision, frame) for decision in decisions.values()]
        cache_count = max(len(item[0]['m1']) for item in items)
        for item in items:
            missing = cache_count-len(item[0]['m1'])
            item[0]['f1'] = np.pad(item[0]['f1'], ((0,missing),(0,0)))
            item[0]['m1'] = np.pad(item[0]['m1'], (0,missing))
        data = {k: torch.as_tensor(np.stack([item[0][k] for item in items])) for k in items[0][0]}
        actions, logs, _, values, _ = self.distributions(data, deterministic=deterministic)
        controls = {}
        for item, action in zip(items, actions.tolist()):
            _, candidates, vid, node = item
            target = candidates[0][action[0]]; cache_target, cache_node, kind = candidates[1][action[1]]
            prepare, priority = candidates[2][action[2]]
            controls[vid] = JointRequestControl(target, cache_target, cache_node, kind, prepare, priority=priority)
            # Waiting remains a one-frame decision; active service stays until node/coverage changes.
            self.commitments[vid] = (node, target) if target is not None else None
        return controls, dict(data=data, actions=actions, logs=logs, values=values, vehicle_ids=list(decisions))

    def update(self, rollouts, *, epochs=4):
        cache_count = max(r['data']['m1'].shape[1] for r in rollouts)
        for r in rollouts:
            missing = cache_count-r['data']['m1'].shape[1]
            r['data']['f1'] = torch.nn.functional.pad(r['data']['f1'], (0,0,0,missing))
            r['data']['m1'] = torch.nn.functional.pad(r['data']['m1'], (0,missing))
        data = {k: torch.cat([r['data'][k] for r in rollouts]) for k in rollouts[0]['data']}
        actions = torch.cat([r['actions'] for r in rollouts]); old = torch.cat([r['logs'] for r in rollouts])
        targets = torch.cat([r['returns'] for r in rollouts]); costs = torch.cat([r.get('cost_targets', torch.zeros(len(r['returns']), 2)) for r in rollouts]); old_values = torch.cat([r['values'] for r in rollouts])
        advantage = targets-old_values; advantage = (advantage-advantage.mean())/advantage.std().clamp(min=1e-6)
        losses = []
        for _ in range(epochs):
            for index in torch.randperm(len(actions)).split(512):
                batch = {k: v[index] for k,v in data.items()}
                _, logs, entropy, value, surrogate = self.distributions(batch, actions=actions[index])
                ratio = (logs-old[index]).exp()
                policy = -torch.minimum(ratio*advantage[index], ratio.clamp(.8,1.2)*advantage[index]).mean()
                value_loss = (value-targets[index]).square().mean()
                surrogate_targets = torch.cat((targets[index,None], costs[index]), -1)
                auxiliary = (surrogate-surrogate_targets[:,None]).square().mean() if self.surrogate_enabled else value_loss*0
                loss = policy + .5*value_loss - .01*entropy.mean() + .2*auxiliary
                self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.parameters(), .5); self.optimizer.step()
                losses.append([float(policy.detach()),float(value_loss.detach()),float(auxiliary.detach())])
        return dict(policy_loss=float(np.mean(losses,0)[0]),value_loss=float(np.mean(losses,0)[1]),
                    surrogate_loss=float(np.mean(losses,0)[2]), samples=len(actions))

    def save_checkpoint(self, path):
        torch.save(dict(contract=self.checkpoint_contract, action_contract=self.action_contract,
                        options=self.options, state_dict=self.state_dict(), optimizer=self.optimizer.state_dict()), path)

    @classmethod
    def from_checkpoint(cls, path):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved.get('contract') != cls.checkpoint_contract or saved.get('action_contract') != cls.action_contract:
            raise ValueError('incompatible v3 checkpoint')
        obj = cls(**saved['options']); obj.load_state_dict(saved['state_dict']); obj.optimizer.load_state_dict(saved['optimizer'])
        return obj


class CoordinatedControlPPO(ParameterizedControlPPO):
    """Request-autoregressive control with charged intent reservations and prefix credit."""
    checkpoint_contract = 'coordinated_parameterized_ppo_v1'

    def __init__(self, *, credit_mode='request_team_blend', migration_gate_enabled=True, coordination_enabled=True, partial_preparation_enabled=False, output_aware_cache=False, **kwargs):
        kwargs.pop('productive_controls',None)
        super().__init__(productive_controls=True, **kwargs)
        if credit_mode not in ('request_team_blend','team'):
            raise ValueError('unknown coordinated credit mode')
        self.credit_mode, self.migration_gate_enabled = credit_mode, bool(migration_gate_enabled)
        self.coordination_enabled = bool(coordination_enabled)
        self.partial_preparation_enabled = bool(partial_preparation_enabled)
        self.output_aware_cache = bool(output_aware_cache)
        self.options.update(credit_mode=credit_mode,migration_gate_enabled=migration_gate_enabled,coordination_enabled=coordination_enabled,partial_preparation_enabled=partial_preparation_enabled,output_aware_cache=output_aware_cache)
        width = 20+16+32+32+16
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(width,64),nn.Tanh(),nn.Linear(64,1)) for _ in range(3)])
        self.surrogates = nn.ModuleList([nn.Sequential(nn.Linear(width,32),nn.Tanh(),nn.Linear(32,3)) for _ in range(3)])
        self.prefix_values = nn.ModuleList([nn.Sequential(nn.Linear(36+32,64),nn.Tanh(),nn.Linear(64,1)) for _ in range(3)])
        self.cache_encoder = nn.Sequential(nn.Linear(32,16),nn.Tanh())
        self.optimizer = torch.optim.Adam(self.parameters(),lr=self.options['learning_rate'])
        self.reset_controller()

    def reset_controller(self):
        super().reset_controller()
        self.target_memory = {}
        self.last_diagnostics = {}

    def _coordinated_features(self, decision, frame, reservations):
        from src.envs.core.joint_control_planning import migration_opportunities, workflow_service_frames
        # Common base features are built without the legacy whole-node commitment mask.
        data,candidates,vid,node_id = super()._features(decision,frame)
        state = decision['semantic_state']; nodes = state['workflow']['nodes']
        ids = sorted(r['rsu_id'] for r in state['rsus']); rsus={r['rsu_id']:r for r in state['rsus']}
        index={n['node_id']:i for i,n in enumerate(nodes)}
        order={n:i for i,n in enumerate(state['workflow']['execution_order'])}
        opportunities = migration_opportunities(state,prediction_enabled=self.prediction_enabled,allow_partial=self.partial_preparation_enabled,model_progress_aware=self.output_aware_cache)
        # A forecast affects control only when it yields a physical, positive-net preparation opportunity.
        if self.migration_gate_enabled and not opportunities:
            data['own'][11:13] = 0.
            data['critic'][11:13] = 0.
            for j in range(3):data[f'f{j}'][:,10:12]=0.
        own_work = workflow_service_frames(state)
        queue = {rid:sum(work for owner,(target,work) in reservations.items() if owner!=vid and target==rid) for rid in ids}
        counts={rid:sum(owner!=vid and target==rid for owner,(target,_) in reservations.items()) for rid in ids}
        plans=[[(target,node_id) for target in candidates[0]],
               [(target,nid or node_id) for target,nid,kind in candidates[1]],
               [(target,node_id) for target,priority in candidates[2]]]
        for j in range(3):
            extra=np.zeros((len(plans[j]),12),np.float32); node_indices=[]
            for k,(rid,nid) in enumerate(plans[j]):
                r=rsus.get(rid,{}); n=nodes[index[nid]]; opportunity=opportunities.get(rid,{})
                relocation = 0.
                if rid in ids and state['execution_location'] and rid!=state['execution_location']:
                    copied=state['state_copies_by_rsu'].get(rid,{}).get('payload_bytes',0)
                    relocation=max(1.,(state['execution_state_payload_bytes']-copied)/state['control_cost_parameters']['state_bytes_per_frame'])
                    if self.partial_preparation_enabled:
                        rate=state['control_cost_parameters']['state_bytes_per_frame']
                        transfer=state['state_transfers_by_rsu'].get(rid)
                        ready=state['state_copies_by_rsu'].get(rid,{}).get('version')==state['execution_state_version']
                        if ready:relocation=0.
                        elif transfer is not None:
                            relocation=max(transfer['remaining_frames'],np.ceil(transfer['remaining_bytes']/rate))
                            if transfer['version']!=state['execution_state_version']:
                                relocation+=max(state['control_cost_parameters']['minimum_state_frames'],np.ceil(max(1,state['execution_state_payload_bytes']-transfer['payload_bytes'])/rate))
                        else:relocation=max(state['control_cost_parameters']['minimum_state_frames'],np.ceil(max(1,state['execution_state_payload_bytes']-copied)/rate))
                net=opportunity.get('net_reward_proxy',0)
                if self.partial_preparation_enabled and j==1:net=opportunity.get('cache_net_reward_proxy',0) if nid==node_id else 0.
                if self.partial_preparation_enabled and j==2:net=opportunity.get('state_net_reward_proxy',0)
                extra[k]=[r.get('service_work_units',0)/120,queue.get(rid,0)/120,counts.get(rid,0)/8,
                          relocation/120,net,opportunity.get('eta_frames',0)/3,
                          opportunity.get('prepare_budget_frames',opportunity.get('prepare_frames',0))/3,(order[nid]-order[node_id])/32,
                          len(n.get('predecessors',[]))/8,sum(a==nid for a,b in state['workflow']['edges'])/8,
                          own_work/120,n['output_size']/32]
                node_indices.append(index[nid] if rid is not None else -1)
            data[f'f{j}']=np.concatenate((data[f'f{j}'],extra),1)
            data[f'node{j}']=np.asarray(node_indices,np.int64)
        data['prior0']=np.zeros(6,np.float32)
        horizon=max(4.,state['episode_frames_remaining'])
        for k,rid in enumerate(candidates[0]):
            if rid in ids:
                # Declared service-frame cost, shared equally by every method. No hidden action override.
                rates=state['control_cost_parameters']
                model_delay=(np.ceil(rates['base_size_bytes']/rates['model_bytes_per_frame'])*(1-data['f0'][k,6])+
                             np.ceil(rates['adapter_size_bytes']/rates['model_bytes_per_frame'])*(1-data['f0'][k,4]))
                if self.output_aware_cache:
                    from src.envs.core.joint_control_planning import remaining_model_frames
                    model_delay=remaining_model_frames(state,rsus[rid])
                data['prior0'][k]=-(queue[rid]+120*data['f0'][k,23]+model_delay)/(horizon/4)
        old=self.target_memory.get(vid)
        if old and old[0]==node_id and old[1] in state['reachable_execution_rsu_ids']:
            location=state['execution_location']
            # In-flight work/output stays in place; break only for coverage loss.
            if state['current_node_work_units_progress']>0 or state['output_pending'] or frame-old[2]<8:
                data['m0'][:]=False;data['m0'][candidates[0].index(old[1])]=True
        if self.migration_gate_enabled:
            for k,(rid,priority) in enumerate(candidates[2]):
                data['m2'][k] &= rid is None or opportunities.get(rid,{}).get('prepare_feasible',False)
        data['opportunity_count']=np.asarray(len(opportunities),np.int64)
        return data,candidates,vid,node_id,own_work

    def _evaluate(self,data,*,actions=None,deterministic=False):
        own=data['own'];batch=len(own)
        latent=self.graph_encoder(data['graph'])
        latent=self.graph_message(torch.cat((latent,torch.bmm(data['adjacency'],latent)),-1))
        if self.method=='sa_ghmappo':
            weight=data['graph_mask'].float().unsqueeze(-1)
            context=(latent*weight).sum(1)/weight.sum(1).clamp(min=1)
        elif self.method=='cache_offload_drl':context=self.cache_encoder(data['f0'][:,1:5]).mean(1)
        else:context=own.new_zeros(batch,16)
        condition=own.new_zeros(batch,32);chosen=[];logs=[];entropies=[];surrogate=[];values=[];active=[]
        for j in range(3):
            f=data[f'f{j}'];mask=data[f'm{j}'];n=f.shape[1]
            if j==1:
                match=(f[...,16:20]==condition[:,None,16:20]).all(-1)&(f[...,12]==1)&(f[...,14]==0)
                eligible=match&mask
                cold=(condition[:,0]==1)&~((condition[:,4]==1)&(condition[:,6]==1))&(condition[:,5]==0)
                if self.output_aware_cache:cold=cold&~own[:,9].bool()
                mask=torch.where((cold&eligible.any(1))[:,None],eligible,mask)
            node_indices=data[f'node{j}']; graph=latent.gather(1,node_indices.clamp(min=0)[...,None].expand(-1,-1,16))
            graph=graph*(node_indices>=0)[...,None] if self.method=='sa_ghmappo' else torch.zeros_like(graph)
            x=torch.cat((own[:,None].expand(-1,n,-1),context[:,None].expand(-1,n,-1),f,
                         condition[:,None].expand(-1,n,-1),graph),-1)
            # Evaluate only legal rows: avoids scoring hundreds of masked cache options on noop frames.
            logits=own.new_full((batch,n),-1e9); predictions=own.new_zeros(batch,n,3)
            valid=x[mask]; logits[mask]=self.heads[j](valid).squeeze(-1)
            if self.surrogate_enabled:
                raw=self.surrogates[j](valid.detach())
                estimates=torch.cat((raw[:,:1],torch.nn.functional.softplus(raw[:,1:])),1)
                predictions[mask]=estimates
                logits[mask]=logits[mask]+.3*(estimates[:,0]-estimates[:,1]-estimates[:,2]).detach().tanh()
            if j==0:logits=logits+data['prior0']
            dist=Categorical(logits=logits)
            action=actions[:,j] if actions is not None else (dist.logits.argmax(-1) if deterministic else dist.sample())
            if not bool(mask.gather(1,action[:,None]).all()):raise ValueError('action outside stored coordinated behavior mask')
            chosen.append(action);logs.append(dist.log_prob(action));entropies.append(dist.entropy())
            values.append(self.prefix_values[j](torch.cat((data['critic'],condition.detach()),-1)).squeeze(-1))
            surrogate.append(predictions[torch.arange(batch),action]);active.append(mask.sum(1)>1)
            condition=f[torch.arange(batch),action]
        return (torch.stack(chosen,1),torch.stack(logs,1),torch.stack(entropies,1),
                self.critic(data['critic']).squeeze(-1),torch.stack(surrogate,1),torch.stack(values,1),torch.stack(active,1))

    def distributions(self,data,*,actions=None,deterministic=False):
        a,log,entropy,value,surrogate,_,_=self._evaluate(data,actions=actions,deterministic=deterministic)
        return a,log.sum(1),entropy.sum(1),value,surrogate

    @torch.no_grad()
    def act_controls(self,decisions,frame,*,deterministic=False):
        from src.envs.core.multi_request_target_control_env import JointRequestControl
        from src.envs.core.joint_control_planning import workflow_service_frames
        ids=sorted(decisions);shift=frame%len(ids);ids=ids[shift:]+ids[:shift]
        reservations={vid:(old[1],workflow_service_frames(decisions[vid]['semantic_state']))
                      for vid,old in self.target_memory.items() if vid in decisions and
                      old[1] in decisions[vid]['semantic_state']['reachable_execution_rsu_ids']}
        controls={};rolls=[];opportunity_count=0;diagnostics={}
        for vid in ids:
            data,candidates,_,node,work=self._coordinated_features(decisions[vid],frame,reservations if self.coordination_enabled else {})
            batch={k:torch.as_tensor(v).unsqueeze(0) for k,v in data.items()}
            a,logs,entropy,value,surrogate,prefix,active=self._evaluate(batch,deterministic=deterministic)
            target=candidates[0][int(a[0,0])];cache_target,cache_node,kind=candidates[1][int(a[0,1])]
            prepare,priority=candidates[2][int(a[0,2])]
            controls[vid]=JointRequestControl(target,cache_target,cache_node,kind,prepare,priority=priority)
            old=self.target_memory.get(vid)
            since=old[2] if old and old[0]==node and old[1]==target else frame
            self.target_memory[vid]=(node,target,since)
            reservations[vid]=(target,work)
            opportunity_count+=int(data['opportunity_count'])
            from src.envs.core.joint_control_planning import migration_opportunities
            diagnostics[vid]=dict(migration_candidates=migration_opportunities(decisions[vid]['semantic_state'],prediction_enabled=self.prediction_enabled,allow_partial=self.partial_preparation_enabled,model_progress_aware=self.output_aware_cache),
                reserved_service_frames={rid:float(data['f0'][i,21]*120) for i,rid in enumerate(candidates[0]) if rid and rid!='vehicle'})
            rolls.append(dict(data=batch,actions=a,logs=logs.sum(1),head_logs=logs,values=value,prefix_values=prefix,active_heads=active))
        cache_count=max(r['data']['m1'].shape[1] for r in rolls)
        self._pad_cache(rolls,cache_count)
        keys=rolls[0]
        result={k:({name:torch.cat([r[k][name] for r in rolls]) for name in rolls[0][k]} if k=='data' else torch.cat([r[k] for r in rolls])) for k in keys}
        result['vehicle_ids']=ids
        self.last_diagnostics=dict(migration_opportunities=opportunity_count,request_order=ids,by_vehicle=diagnostics)
        return controls,result

    @staticmethod
    def _pad_cache(rollouts,count):
        for r in rollouts:
            missing=count-r['data']['m1'].shape[1]
            r['data']['f1']=torch.nn.functional.pad(r['data']['f1'],(0,0,0,missing))
            r['data']['m1']=torch.nn.functional.pad(r['data']['m1'],(0,missing))
            r['data']['node1']=torch.nn.functional.pad(r['data']['node1'],(0,missing),value=-1)

    def update(self,rollouts,*,epochs=4):
        self._pad_cache(rollouts,max(r['data']['m1'].shape[1] for r in rollouts))
        data={k:torch.cat([r['data'][k] for r in rollouts]) for k in rollouts[0]['data']}
        actions=torch.cat([r['actions'] for r in rollouts]);old=torch.cat([r['head_logs'] for r in rollouts])
        team=torch.cat([r['returns'] for r in rollouts]);targets=torch.cat([r['credit_targets'] for r in rollouts])
        if self.credit_mode=='team':targets=torch.cat([r['team_credit_targets'] for r in rollouts])
        values=torch.cat([r['prefix_values'] for r in rollouts]);active=torch.cat([r['active_heads'] for r in rollouts])
        advantages=targets[:,None]-values
        for j in range(3):
            sample=advantages[active[:,j],j]
            if len(sample)>1:advantages[:,j]=(advantages[:,j]-sample.mean())/sample.std().clamp(min=1e-6)
        costs=torch.cat([r['cost_targets'] for r in rollouts]);losses=[]
        for _ in range(epochs):
            for index in torch.randperm(len(actions)).split(512):
                batch={k:v[index] for k,v in data.items()}
                _,logs,entropy,value,surrogate,prefix,enabled=self._evaluate(batch,actions=actions[index])
                ratio=(logs-old[index]).exp()
                objective=torch.minimum(ratio*advantages[index],ratio.clamp(.8,1.2)*advantages[index])
                weight=enabled.float();denominator=weight.sum().clamp(min=1)
                policy=-(objective*weight).sum()/denominator
                vf=(prefix-targets[index,None]).square().mean()+ (value-team[index]).square().mean()
                aux=(surrogate-torch.cat((targets[index,None],costs[index]),-1)[:,None]).square().mean() if self.surrogate_enabled else vf*0
                loss=policy+.5*vf-.01*(entropy*weight).sum()/denominator+.2*aux
                self.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.parameters(),.5);self.optimizer.step()
                losses.append([float(policy.detach()),float(vf.detach()),float(aux.detach())])
        means=np.mean(losses,0)
        return dict(policy_loss=float(means[0]),value_loss=float(means[1]),surrogate_loss=float(means[2]),samples=len(actions))


class ScheduledControlPPO(CoordinatedControlPPO):
    """Versioned link-calendar, service-scaled, action-conditioned control learner."""
    checkpoint_contract='scheduled_parameterized_ppo_v1'

    def __init__(self, *, stability_enabled=True, training_budget=128, planning_horizon=8, **kwargs):
        kwargs.update(partial_preparation_enabled=True,output_aware_cache=True)
        super().__init__(**kwargs)
        self.stability_enabled=bool(stability_enabled);self.training_budget=int(training_budget);self.planning_horizon=int(planning_horizon)
        self.options.update(stability_enabled=stability_enabled,training_budget=training_budget,planning_horizon=planning_horizon)
        self.action_values=nn.ModuleList([nn.Sequential(nn.Linear(116,64),nn.Tanh(),nn.Linear(64,1)) for _ in range(3)])
        self.domain_encoder=nn.Sequential(nn.Linear(4,16),nn.Tanh())
        self.register_buffer('completed_updates',torch.zeros((),dtype=torch.int64))
        self.register_buffer('surrogate_mse',torch.ones(()))
        self.optimizer=torch.optim.Adam(self.parameters(),lr=self.options['learning_rate'])

    def _coordinated_features(self,decision,frame,reservations):
        from src.envs.core.joint_link_reservation import reserved_migration_opportunities,service_scale
        data,candidates,vid,node,work=super()._coordinated_features(decision,frame,reservations)
        s=decision['semantic_state'];rates=s['control_cost_parameters'];nodes=s['workflow']['nodes'];lookup={n['node_id']:n for n in nodes}
        calendar=getattr(self,'_calendar',{r['rsu_id']:[None]*self.planning_horizon for r in s['rsus']})
        opportunities=self._reserved_opportunities(s,calendar)
        # Rebuild forecast fields before applying the v4 opportunity gate (v3 may have cleared them).
        raw=super(CoordinatedControlPPO,self)._features(decision,frame)[0]
        data['m2']=raw['m2'].copy()
        data['own'][11:13]=raw['own'][11:13]
        for j in range(3):data[f'f{j}'][:,10:12]=raw[f'f{j}'][:,10:12]
        if self.method!='dt_handoff_drl' and not opportunities:
            data['own'][11:13]=0
            for j in range(3):data[f'f{j}'][:,10:12]=0
        data['own'][12]*=3/self.planning_horizon
        horizon=s['episode_frames_remaining'];den=max(1.,s['current_node_work_units_required'])
        data['own'][3]=min(1.,s['current_node_work_units_progress']/den)
        data['own'][4]=service_scale(s['current_node_work_units_required']/rates['rsu_work_units_per_frame'])
        data['own'][5]=service_scale(s['execution_state_payload_bytes']/rates['state_bytes_per_frame'])
        data['critic'][:20]=data['own']
        if self.method=='ppo':data['critic'][20:]=0  # Local critic, same public actor observations.
        for k,n in enumerate(nodes):
            data['graph'][k,0]=service_scale(n['input_size']*rates['profile_unit_bytes']/rates['workload_bytes_per_unit']/rates['rsu_work_units_per_frame'])
            data['graph'][k,1]=service_scale(n['output_size']*rates['profile_unit_bytes']/rates['output_bytes_per_frame'])
        for j in range(3):
            for k,f in enumerate(data[f'f{j}']):
                rid=(candidates[j][k] if j==0 else candidates[j][k][0]);nid=(candidates[1][k][1] or node) if j==1 else node
                n=lookup[nid];f[13]=service_scale(n['input_size']*rates['profile_unit_bytes']/rates['input_bytes_per_frame'])
                f[31]=service_scale(n['output_size']*rates['profile_unit_bytes']/rates['output_bytes_per_frame'])
                f[30]=np.clip((horizon-work+float(data['prior0'][k])*max(4.,horizon)/4)/max(1.,horizon),-1,1) if j==0 else service_scale(work);op=opportunities.get(rid,{})
                f[24]=op.get('cache_net_reward_proxy' if j==1 else 'state_net_reward_proxy',0.) if j else op.get('net_reward_proxy',0.)
                f[25]=op.get('eta_frames',0)/self.planning_horizon
                f[26]=op.get('prepare_budget_frames',0)/self.planning_horizon
                if rid in calendar:
                    grant=calendar[rid][0];available=grant is None or grant[0]==vid
                    if j==1:data['m1'][k]&=available and (grant is None or grant[1]=='model')
                    if j==2 and rid is not None:data['m2'][k]&=available and (grant is None or grant[1]=='state') and op.get('prepare_feasible',False)
            data[f'f{j}'][:,20:22]=np.log1p(np.maximum(0,data[f'f{j}'][:,20:22]*120))/np.log1p(120)
        data['opportunity_count']=np.asarray(len(opportunities),np.int64)
        data['domain']=np.array([len(nodes)/32,len(s['workflow']['edges'])/64,service_scale(work),len(s['workflow']['completed_node_ids'])/max(1,len(nodes))],np.float32)
        if self.method=='dt_handoff_drl':data['domain']=np.array([data['own'][11],data['own'][12],data['f0'][:,10].max(),data['f0'][:,11].mean()],np.float32)
        self._feature_opportunities=opportunities
        return data,candidates,vid,node,work

    def _reserved_opportunities(self,state,calendar):
        from src.envs.core.joint_link_reservation import reserved_migration_opportunities
        return reserved_migration_opportunities(state,calendar,prediction_enabled=self.prediction_enabled)

    def _policy_context(self,data,latent):
        own=data['own']
        if self.method=='sa_ghmappo':
            weight=data['graph_mask'].float().unsqueeze(-1)
            return (latent*weight).sum(1)/weight.sum(1).clamp(min=1)
        if self.method=='cache_offload_drl':return self.cache_encoder(data['f0'][:,1:5]).mean(1)
        if self.method in ('dt_handoff_drl','dag_offload_drl'):return self.domain_encoder(data['domain'])
        return own.new_zeros(len(own),16)

    def _candidate_graph(self,graph,idx):
        return graph*(idx>=0)[...,None] if self.method=='sa_ghmappo' else torch.zeros_like(graph)

    def _control_variate(self,probs,q,predictions):
        return (probs.detach()*q).sum(1)

    def _entropy_coefficient(self,progress):return .01

    def _apply_control_gradients(self,loss):
        self.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.parameters(),.5);self.optimizer.step()

    def _reduce_active_policy_terms(self, objective, entropy, weight, *, batch_indices=None,
                                    entropy_coefficient=1.):
        """Historical active-row reduction; opt-in subclasses may balance heads.

        Keep the default operations unchanged so existing policies and controls
        retain their original update semantics.
        """
        denominator = weight.sum().clamp(min=1)
        return (-(objective*weight).sum()/denominator,
                entropy_coefficient*(entropy*weight).sum()/denominator)

    def _evaluate(self,data,*,actions=None,deterministic=False,return_details=False,target=False,execution_actions=None,execution_only=False):
        own=data['own'];batch=len(own)
        latent=self.graph_encoder(data['graph']);latent=self.graph_message(torch.cat((latent,torch.bmm(data['adjacency'],latent)),-1))
        context=self._policy_context(data,latent)
        condition=own.new_zeros(batch,32);execution=None;chosen=[];logs=[];entropies=[];surrogate=[];values=[];active=[];baselines=[];selected_q=[];maxima=[]
        heads=self.target_heads if target else self.heads
        for j in range(3):
            f=data[f'f{j}'];mask=data[f'm{j}'].clone();n=f.shape[1]
            same=(f[...,16:20]==condition[:,None,16:20]).all(-1)
            if j==1:
                eligible=same&(f[...,12]==1)&(f[...,14]==0)&mask
                cold=(condition[:,0]==1)&~((condition[:,4]==1)&(condition[:,6]==1))&(condition[:,5]==0)&~own[:,9].bool()
                mask=torch.where((cold&eligible.any(1))[:,None],eligible,mask)
            if j==2:
                cache_conflict=(condition[:,0]==1)[:,None]&same
                exec_conflict=(f[...,16:20]==execution[:,None,16:20]).all(-1)&(execution[:,0]==1)[:,None]
                mask&=~((cache_conflict|exec_conflict)&(f[...,0]==1))
            idx=data[f'node{j}'];graph=latent.gather(1,idx.clamp(min=0)[...,None].expand(-1,-1,16))
            graph=self._candidate_graph(graph,idx)
            x=torch.cat((own[:,None].expand(-1,n,-1),context[:,None].expand(-1,n,-1),f,condition[:,None].expand(-1,n,-1),graph),-1)
            logits=own.new_full((batch,n),-1e9);predictions=own.new_zeros(batch,n,3);q=own.new_zeros(batch,n)
            valid=x[mask];logits[mask]=heads[j](valid).squeeze(-1)
            if self.surrogate_enabled:
                raw=self.surrogates[j](valid.detach());est=torch.cat((raw[:,:1],torch.nn.functional.softplus(raw[:,1:])),1);predictions[mask]=est
                # Return targets already include physical costs. Centering cancels common return drift;
                # subtracting the cost estimate again would double-charge an action.
                logits=logits+self._surrogate_logit_adjustment(j,predictions,mask,data)
            if j==0:logits=logits+data['prior0']
            dist=self._choice_distribution(logits,mask,deterministic)
            action=actions[:,j] if actions is not None else (execution_actions if j==0 and execution_actions is not None else (dist.logits.argmax(-1) if deterministic else dist.sample()))
            if not bool(mask.gather(1,action[:,None]).all()):raise ValueError('action outside stored scheduled behavior mask')
            if execution_only:return action[:,None],dist.log_prob(action)[:,None]
            q[mask]=self.action_values[j](valid.detach()).squeeze(-1)
            if self.method=='dqn':q=logits
            baselines.append(self._control_variate(dist.probs,q,predictions));selected_q.append(q.gather(1,action[:,None]).squeeze(1));maxima.append(logits.masked_fill(~mask,-1e9).max(1).values)
            chosen.append(action);logs.append(dist.log_prob(action));entropies.append(dist.entropy()/mask.sum(1).float().log().clamp(min=1))
            values.append(self.prefix_values[j](torch.cat((data['critic'],condition.detach()),-1)).squeeze(-1));surrogate.append(predictions[torch.arange(batch),action]);active.append(mask.sum(1)>1)
            condition=f[torch.arange(batch),action]
            if j==0:execution=condition
        result=(torch.stack(chosen,1),torch.stack(logs,1),torch.stack(entropies,1),self.critic(data['critic']).squeeze(-1),torch.stack(surrogate,1),torch.stack(values,1),torch.stack(active,1))
        if return_details:return (*result,torch.stack(baselines,1),torch.stack(selected_q,1),torch.stack(maxima,1))
        return result

    def _choice_distribution(self,logits,mask,deterministic):
        return Categorical(logits=logits.masked_fill(~mask,-1e9))

    def _surrogate_logit_adjustment(self,j,predictions,mask,data):
        center=(predictions[...,0]*mask).sum(1)/mask.sum(1).clamp(min=1)
        relative=predictions[...,0]-center[:,None]
        if j:relative=predictions[...,0]-predictions[:,0:1,0]
        reliability=1/(1+10*self.surrogate_mse.sqrt())
        return .3*reliability*relative.detach().tanh()

    @torch.no_grad()
    def act_controls(self,decisions,frame,*,deterministic=False):
        from dataclasses import replace
        from copy import deepcopy
        from src.envs.core.multi_request_target_control_env import JointRequestControl
        from src.envs.core.joint_link_reservation import public_link_calendar,control_link_claims,book_control
        from src.envs.core.joint_control_planning import workflow_service_frames
        ids=sorted(decisions);shift=frame%len(ids);ids=ids[shift:]+ids[:shift]
        targets={v:old[1] for v,old in self.target_memory.items() if v in decisions}
        self._calendar=public_link_calendar(decisions,targets,horizon=self.planning_horizon)
        reservations={v:(old[1],workflow_service_frames(decisions[v]['semantic_state'])) for v,old in self.target_memory.items() if v in decisions and old[1] in decisions[v]['semantic_state']['reachable_execution_rsu_ids']}
        controls={};rolls=[];diagnostics={};opportunities=0
        for vid in ids:
            data,candidates,_,node,work=self._coordinated_features(decisions[vid],frame,reservations)
            plans=deepcopy(self._feature_opportunities);calendar=deepcopy(self._calendar)
            batch={k:torch.as_tensor(v).unsqueeze(0) for k,v in data.items()}
            a,logs,entropy,value,surrogate,prefix,enabled,baseline,q,_=self._evaluate(batch,deterministic=deterministic,return_details=True)
            target=candidates[0][int(a[0,0])];cache,nid,kind=candidates[1][int(a[0,1])];prepare,priority=candidates[2][int(a[0,2])]
            control=JointRequestControl(target,cache,nid,kind,prepare,priority=priority)
            claims=control_link_claims(decisions[vid]['semantic_state'],control,self._calendar)
            controls[vid]=replace(control,link_reservations=claims);book_control(self._calendar,vid,claims,decisions[vid]['semantic_state'],control)
            old=self.target_memory.get(vid);since=old[2] if old and old[0]==node and old[1]==target else frame
            self.target_memory[vid]=(node,target,since);reservations[vid]=(target,work);opportunities+=len(plans)
            diagnostics[vid]=dict(migration_candidates=plans,reserved_service_frames={rid:sum(w for v,(r,w) in reservations.items() if v!=vid and r==rid) for rid in self._calendar},link_calendar=calendar)
            rolls.append(dict(data=batch,actions=a,logs=logs.sum(1),head_logs=logs,values=value,prefix_values=prefix,active_heads=enabled,counterfactual_baselines=baseline))
        count=max(r['data']['m1'].shape[1] for r in rolls);self._pad_cache(rolls,count)
        result={k:torch.cat([r[k] for r in rolls]) for k in rolls[0] if k!='data'}
        result['data']={k:torch.cat([r['data'][k] for r in rolls]) for k in rolls[0]['data']};result['vehicle_ids']=ids
        self.last_diagnostics=dict(migration_opportunities=opportunities,request_order=ids,by_vehicle=diagnostics)
        return controls,result

    def update(self,rollouts,*,epochs=4):
        self._pad_cache(rollouts,max(r['data']['m1'].shape[1] for r in rollouts))
        data={k:torch.cat([r['data'][k] for r in rollouts]) for k in rollouts[0]['data']}
        actions=torch.cat([r['actions'] for r in rollouts]);old=torch.cat([r['head_logs'] for r in rollouts])
        targets=torch.cat([r['credit_targets'] for r in rollouts]);team=torch.cat([r['returns'] for r in rollouts]);active=torch.cat([r['active_heads'] for r in rollouts])
        baseline=torch.cat([r['counterfactual_baselines'] for r in rollouts]);advantages=targets[:,None]-baseline
        for j in range(3):
            sample=advantages[active[:,j],j]
            if len(sample)>1:advantages[:,j]=(advantages[:,j]-sample.mean())/sample.std().clamp(min=1e-6)
        costs=torch.cat([r['cost_targets'] for r in rollouts]);losses=[];kls=[];clips=[];entropies=[];qerrors=[];stopped=False
        progress=min(1.,float(self.completed_updates)*4/max(1,self.training_budget-4))
        lr=self.options['learning_rate']*(1-.9*progress) if self.stability_enabled else self.options['learning_rate']
        for group in self.optimizer.param_groups:group['lr']=lr
        for _ in range(epochs):
            for index in torch.randperm(len(actions)).split(512):
                batch={k:v[index] for k,v in data.items()}
                _,logs,entropy,value,surrogate,prefix,enabled,cf,q,_=self._evaluate(batch,actions=actions[index],return_details=True)
                logratio=logs-old[index];ratio=logratio.exp();weight=enabled.float();den=weight.sum().clamp(min=1)
                kl=(((ratio-1)-logratio)*weight).sum()/den
                if self.stability_enabled and float(kl.detach())>.02:stopped=True;break
                objective=torch.minimum(ratio*advantages[index],ratio.clamp(.8,1.2)*advantages[index])
                policy,entropy_penalty=self._reduce_active_policy_terms(
                    objective,entropy,weight,batch_indices=index,
                    entropy_coefficient=self._entropy_coefficient(progress))
                vf=(prefix-targets[index,None]).square().mean()+(value-team[index]).square().mean()+(q-targets[index,None]).square().mean()
                expected=torch.cat((targets[index,None],costs[index]),-1)[:,None]
                aux=(surrogate-expected).square().mean() if self.surrogate_enabled else vf*0
                loss=policy+.5*vf-entropy_penalty+.2*aux
                self._apply_control_gradients(loss)
                if self.surrogate_enabled:self.surrogate_mse.mul_(.95).add_(.05*(surrogate[...,0].detach()-targets[index,None]).square().mean())
                losses.append([float(policy.detach()),float(vf.detach()),float(aux.detach())]);kls.append(float(kl.detach()));clips.append(float((((ratio-1).abs()>.2)*weight).sum().detach()/den));entropies.append(float((entropy*weight).sum().detach()/den));qerrors.append(float((q.detach()-targets[index,None]).square().mean()))
            if stopped:break
        self.completed_updates+=1
        means=np.mean(losses,0) if losses else [0.,0.,0.]
        return dict(policy_loss=float(means[0]),value_loss=float(means[1]),surrogate_loss=float(means[2]),samples=len(actions),gradient_steps=len(losses),learning_rate=lr,approx_kl=float(np.mean(kls)) if kls else 0.,clip_fraction=float(np.mean(clips)) if clips else 0.,normalized_entropy=float(np.mean(entropies)) if entropies else 0.,q_mse=float(np.mean(qerrors)) if qerrors else 0.,kl_early_stop=int(stopped))


class TailControlPPO(ScheduledControlPPO):
    """Finite-horizon delivery credit and preparation-scoped surrogate assistance."""
    checkpoint_contract='tail_parameterized_ppo_v1'

    def __init__(self, *, return_gamma=1., **kwargs):
        super().__init__(**kwargs)
        if not 0<return_gamma<=1:raise ValueError('return_gamma must be in (0,1]')
        self.return_gamma=float(return_gamma);self.options['return_gamma']=self.return_gamma

    def _surrogate_logit_adjustment(self,j,predictions,mask,data):
        if j==0:return torch.zeros_like(predictions[...,0])
        values=predictions[...,0]
        if j==2:
            # Four priorities belong to each preparation target. Auxiliary return
            # can value preparing a target, but cannot reorder that target's queue.
            grouped=values.reshape(len(values),5,4)
            weights=mask.reshape(len(values),5,4).float()
            means=(grouped*weights).sum(-1)/weights.sum(-1).clamp(min=1)
            values=means.repeat_interleave(4,dim=1)
        relative=values-values[:,0:1]
        f=data[f'f{j}']
        eligible=mask & (f[...,0]>0) & (f[...,24]>0) & mask[:,0:1]
        reliability=1/(1+10*self.surrogate_mse.sqrt())
        return .3*reliability*relative.detach().tanh()*eligible

    def _coordinated_features(self,decision,frame,reservations):
        data,candidates,vid,node,work=super()._coordinated_features(decision,frame,reservations)
        s=decision['semantic_state'];rates=s['control_cost_parameters']
        pending=s['output_pending'];current=s['current_workflow_node']
        # The current input has already been paid while computing or returning output.
        current_compute=max(0.,s['current_node_work_units_required']-s['current_node_work_units_progress'])/rates['rsu_work_units_per_frame']
        current_input=np.ceil(max(1,current['input_size']*rates['profile_unit_bytes'])/rates['input_bytes_per_frame'])
        current_output=np.ceil(max(1,current['output_size']*rates['profile_unit_bytes'])/rates['output_bytes_per_frame'])
        correction=current_input if s['execution_location'] in s['input_received_targets'] else 0.
        if pending:correction=current_input+current_compute+current_output-np.ceil(pending['remaining_bytes']/rates['output_bytes_per_frame'])
        remaining=max(0.,work-correction)
        data['own'][2]=(len(s['workflow']['nodes'])-len(s['workflow']['completed_node_ids']))/max(1,len(s['workflow']['nodes']))
        # Forecast belongs to preparation. Execution/priority use observed progress,
        # load and deadline; DT's dedicated raw forecast context remains available.
        data['own'][11:13]=0
        data['critic'][:20]=data['own']
        data['f0'][:,10:12]=0;data['f0'][:,24:27]=0
        horizon=max(1,s['episode_frames_remaining'])
        for k,f in enumerate(data['f0']):
            queue=np.expm1(max(0.,f[21])*np.log1p(120))
            relocation=f[23]*120
            f[30]=np.clip((horizon-remaining-queue-relocation)/horizon,-1,1)
        return data,candidates,vid,node,work

    def update(self,rollouts,*,epochs=4):
        # Global value and request/action values now share reward units per initial
        # request, avoiding a larger team regression dominating global grad clipping.
        original=[r['returns'] for r in rollouts]
        try:
            for r in rollouts:r['returns']=r['returns']/r['initial_request_count']
            return super().update(rollouts,epochs=epochs)
        finally:
            for r,value in zip(rollouts,original):r['returns']=value


class DeliveryControlPPO(TailControlPPO):
    """Team node delivery with two-pass reactive protection and cache-residual DAG context."""
    checkpoint_contract='delivery_parameterized_ppo_v2'

    def __init__(self, *, entropy_anneal=True, **kwargs):
        kwargs['credit_mode']='team'
        super().__init__(**kwargs)
        self.entropy_anneal=bool(entropy_anneal);self.options['entropy_anneal']=self.entropy_anneal
        self.graph_residual_gate=nn.Parameter(torch.zeros(()))
        self._actor_parameters=[p for n,p in self.named_parameters() if not n.startswith('surrogates.')]
        self.optimizer=torch.optim.Adam(self._actor_parameters,lr=self.options['learning_rate'])
        self.surrogate_optimizer=torch.optim.Adam(self.surrogates.parameters(),lr=self.options['learning_rate'])
        self._gradient_stats=[]

    def _policy_context(self,data,latent):
        if self.method!='sa_ghmappo':return super()._policy_context(data,latent)
        weight=data['graph_mask'].float().unsqueeze(-1)
        graph=(latent*weight).sum(1)/weight.sum(1).clamp(min=1)
        return self.cache_encoder(data['f0'][:,1:5]).mean(1)+.25*self.graph_residual_gate.tanh()*graph

    def _candidate_graph(self,graph,idx):
        value=super()._candidate_graph(graph,idx)
        return .25*self.graph_residual_gate.tanh()*value if self.method=='sa_ghmappo' else value

    def _surrogate_logit_adjustment(self,j,predictions,mask,data):
        # Observational return predicts a control variate, not a causal action bonus.
        return torch.zeros_like(predictions[...,0])

    def _control_variate(self,probs,q,predictions):
        estimate=.5*(q+predictions[...,0]) if self.surrogate_enabled else q
        return (probs.detach()*estimate).sum(1)

    def _entropy_coefficient(self,progress):return .01*(1-.9*progress) if self.entropy_anneal else .01

    def _apply_control_gradients(self,loss):
        self.optimizer.zero_grad();self.surrogate_optimizer.zero_grad();loss.backward()
        actor_norm=torch.nn.utils.clip_grad_norm_(self._actor_parameters,.5)
        surrogate_norm=torch.nn.utils.clip_grad_norm_(self.surrogates.parameters(),.5)
        self.optimizer.step()
        if self.surrogate_enabled:self.surrogate_optimizer.step()
        self._gradient_stats.append((float(actor_norm),float(surrogate_norm)))

    def _coordinated_features(self,decision,frame,reservations):
        from src.envs.core.joint_delivery_planning import fresh_model_span,preserves_current_model_demands
        data,candidates,vid,node,work=super()._coordinated_features(decision,frame,reservations)
        state=decision['semantic_state'];rows={r['rsu_id']:r for r in state['rsus']}
        nodes={n['node_id']:n for n in state['workflow']['nodes']};calendar=getattr(self,'_calendar',{})
        for k,(rid,nid,kind) in enumerate(candidates[1]):
            if nid is not None and nid!=node:data['f1'][k,24:27]=0
            if rid in calendar and nid==node and getattr(self,'_team_targets',{}).get(vid)==rid and not state['output_pending']:
                row=rows[rid];n=nodes[nid];grant=calendar[rid][0]
                ready=(n['required_base_model'] in row.get('ready_base_model_ids',[]) if kind=='base_model' else
                       n['required_adapter'] in row.get('cached_adapter_ids',[]) or n['required_adapter'] in row.get('loading_adapter_ids',[]))
                # Reactive misses run immediately; only proactive cache runs on the 4-frame clock.
                data['m1'][k]=not ready and (grant is None or grant==(vid,'model'))
            if rid is None or rid not in calendar or not data['m1'][k]:continue
            span=fresh_model_span(state,rows[rid],nodes[nid],kind)
            # Async model transfer cannot yield the next frame: protect every
            # already reserved phase in the entire newly admitted closure.
            if span and (span>len(calendar[rid]) or any(x is not None and x!=(vid,'model') for x in calendar[rid][:span])):
                data['m1'][k]=False
            if not preserves_current_model_demands(state,rows[rid],nodes[nid],kind,getattr(self,'_delivery_decisions',{}),getattr(self,'_team_targets',{}),getattr(self,'_model_catalog',{})):
                data['m1'][k]=False
        return data,candidates,vid,node,work

    def _reserved_opportunities(self,state,calendar):
        from src.envs.core.joint_delivery_planning import delivery_migration_opportunities
        return delivery_migration_opportunities(state,calendar,decisions=getattr(self,'_delivery_decisions',None),
            targets=getattr(self,'_team_targets',None),prediction_enabled=self.prediction_enabled)

    @torch.no_grad()
    def act_controls(self,decisions,frame,*,deterministic=False):
        from copy import deepcopy
        from dataclasses import replace
        from src.envs.core.joint_control_planning import workflow_service_frames
        from src.envs.core.joint_link_reservation import control_link_claims
        from src.envs.core.joint_delivery_planning import reactive_link_calendar,book_delivery_control,fresh_model_span
        from src.envs.core.multi_request_target_control_env import JointRequestControl
        ids=sorted(decisions);shift=frame%len(ids);ids=ids[shift:]+ids[:shift]
        self._delivery_decisions=decisions
        self._model_catalog={f"adapter:{n['required_adapter']}":f"base:{n['required_base_model']}" for d in decisions.values() for n in d['semantic_state']['workflow']['nodes']}
        targets={v:(self.target_memory[v][1] if v in self.target_memory else (d['semantic_state']['execution_location'] or d['semantic_state']['associations'][v])) for v,d in decisions.items()}
        self._team_targets=targets;self._calendar=reactive_link_calendar(decisions,targets,self.planning_horizon)
        reservations={v:(old[1],workflow_service_frames(decisions[v]['semantic_state'])) for v,old in self.target_memory.items()
                      if v in decisions and old[1] in decisions[v]['semantic_state']['reachable_execution_rsu_ids']}
        first={};chosen={};work_by_vehicle={};nodes={};candidates_by_vehicle={}
        # Freeze all execution actions first. Their exact behavior features are
        # retained while cache/event masks are rebuilt from the common team plan.
        for vid in ids:
            data,candidates,_,node,work=self._coordinated_features(decisions[vid],frame,reservations)
            batch={k:torch.as_tensor(v).unsqueeze(0) for k,v in data.items()}
            a,*_=self._evaluate(batch,deterministic=deterministic,execution_only=True)
            first[vid]=data;chosen[vid]=a[:,0];targets[vid]=candidates[0][int(a[0,0])]
            reservations[vid]=(targets[vid],work);work_by_vehicle[vid]=work;nodes[vid]=node;candidates_by_vehicle[vid]=candidates
        self._calendar=reactive_link_calendar(decisions,targets,self.planning_horizon)
        controls={};rolls=[];diagnostics={};opportunities=0
        for vid in ids:
            fresh,candidates,_,node,work=self._coordinated_features(decisions[vid],frame,reservations)
            plans=deepcopy(self._feature_opportunities);calendar=deepcopy(self._calendar)
            data=dict(first[vid])
            for key in ('f1','m1','node1','f2','m2','node2','opportunity_count'):data[key]=fresh[key]
            batch={k:torch.as_tensor(v).unsqueeze(0) for k,v in data.items()}
            a,logs,entropy,value,surrogate,prefix,enabled,baseline,q,_=self._evaluate(batch,
                deterministic=deterministic,execution_actions=chosen[vid],return_details=True)
            cache,nid,kind=candidates[1][int(a[0,1])];prepare,priority=candidates[2][int(a[0,2])]
            control=JointRequestControl(targets[vid],cache,nid,kind,prepare,priority=priority)
            claims=control_link_claims(decisions[vid]['semantic_state'],control,self._calendar)
            controls[vid]=replace(control,link_reservations=claims)
            book_delivery_control(self._calendar,vid,claims,decisions[vid]['semantic_state'],control)
            old=self.target_memory.get(vid);since=old[2] if old and old[0]==node and old[1]==targets[vid] else frame
            self.target_memory[vid]=(node,targets[vid],since);opportunities+=len(plans)
            cache_span=0
            if cache is not None:
                st=decisions[vid]['semantic_state'];row=next(r for r in st['rsus'] if r['rsu_id']==cache)
                cn=next(n for n in st['workflow']['nodes'] if n['node_id']==nid)
                cache_span=fresh_model_span(st,row,cn,kind)
            diagnostics[vid]=dict(migration_candidates=plans,link_calendar=calendar,two_pass_execution_targets=dict(targets),cache_admission_span=cache_span,
                reserved_service_frames={rid:sum(w for v,(target,w) in reservations.items() if v!=vid and target==rid) for rid in calendar})
            rolls.append(dict(data=batch,actions=a,logs=logs.sum(1),head_logs=logs,values=value,prefix_values=prefix,
                active_heads=enabled,counterfactual_baselines=baseline))
        self._pad_cache(rolls,max(r['data']['m1'].shape[1] for r in rolls))
        result={k:torch.cat([r[k] for r in rolls]) for k in rolls[0] if k!='data'}
        result['data']={k:torch.cat([r['data'][k] for r in rolls]) for k in rolls[0]['data']};result['vehicle_ids']=ids
        self.last_diagnostics=dict(migration_opportunities=opportunities,request_order=ids,by_vehicle=diagnostics)
        return controls,result

    def update(self,rollouts,*,epochs=4):
        old=[r['credit_targets'] for r in rollouts];self._gradient_stats=[]
        try:
            for r in rollouts:r['credit_targets']=r['team_credit_targets']
            result=super().update(rollouts,epochs=epochs)
        finally:
            for r,value in zip(rollouts,old):r['credit_targets']=value
        result.update(actor_gradient_norm=float(np.mean([x[0] for x in self._gradient_stats])) if self._gradient_stats else 0.,
            surrogate_gradient_norm=float(np.mean([x[1] for x in self._gradient_stats])) if self._gradient_stats else 0.,
            graph_residual_gate=float(.25*self.graph_residual_gate.detach().tanh()),
            entropy_coefficient=self._entropy_coefficient(min(1.,max(0.,float(self.completed_updates-1))*4/max(1,self.training_budget-4))))
        return result

    def save_checkpoint(self,path):
        super().save_checkpoint(path)
        saved=torch.load(path,weights_only=False);saved['surrogate_optimizer']=self.surrogate_optimizer.state_dict()
        saved['training_resume_supported']=False;torch.save(saved,path)

    @classmethod
    def from_checkpoint(cls,path):
        obj=super().from_checkpoint(path);saved=torch.load(path,weights_only=False)
        obj.surrogate_optimizer.load_state_dict(saved['surrogate_optimizer']);return obj


class StreamControlPPO(DeliveryControlPPO):
    """Released-request graph controls; session team returns, no arrival oracle.

    Feature widths match the delivery network, but field meanings and graph size
    are new and checkpoint-incompatible. Identity is used for binding only.
    """
    action_contract = 'joint_stream_control_v2'
    checkpoint_contract = 'stream_parameterized_ppo_v2'
    max_released_nodes = 64
    max_cache_candidates = 513

    def reset_controller(self):
        super().reset_controller()
        self.request_memory = {}

    def _coordinated_features(self, decision, frame, reservations):
        from math import ceil
        from src.envs.core.joint_link_reservation import service_scale
        from src.envs.core.joint_delivery_planning import fresh_model_span, node_delivery_times
        from src.envs.core.joint_stream_rules import protects_current_models
        from src.envs.core.joint_stream_refinement import state_restore_frames, optional_cache_value
        if decision.get('action_contract') != self.action_contract:
            raise ValueError('stream learner requires request-bound stream contract')
        s = decision['semantic_state']; vid = s['primary_vehicle_id']; rid = s['request_id']
        current = s['current_workflow_node']; done = set(s['workflow']['completed_node_ids'])
        graphs = [(rid, s['workflow']), *[(q['request_id'], q['workflow']) for q in s['queued_workflows']]]
        nodes = [(request, n) for request, w in graphs for n in w['nodes']]
        if len(nodes) > self.max_released_nodes or len(s['rsus']) != 4:
            raise ValueError('stream learning supports <=64 released nodes and four RSUs')
        lookup = {n['node_id']: (request, n) for request, n in nodes}
        if len(lookup) != len(nodes):raise ValueError('released node identities must be unique')
        index = {n['node_id']: i for i, (_, n) in enumerate(nodes)}
        rows = {r['rsu_id']: r for r in s['rsus']}; ids = sorted(rows)
        rates = s['control_cost_parameters']; catalog = rates['model_catalog']; horizon = s['episode_frames_remaining']
        model_byte_scale=max(o['resident_size_bytes']+sum(catalog[k]['resident_size_bytes'] for k in o['dependency_ids']) for o in catalog.values())
        scale = lambda x: service_scale(x, 480)
        work = sum(ceil(max(1., n['input_size']*rates['profile_unit_bytes']/rates['workload_bytes_per_unit'])/rates['rsu_work_units_per_frame']) for request,n in nodes if request != rid or n['node_id'] not in done)
        queued_work = sum(ceil(max(1., n['input_size']*rates['profile_unit_bytes']/rates['workload_bytes_per_unit'])/rates['rsu_work_units_per_frame']) for request,n in nodes if request != rid)
        pred = s['predictions'] if self.prediction_enabled else {}
        future = pred.get('predicted_next_rsu_by_vehicle', {}).get(vid)
        confidence = pred.get('prediction_confidence_by_vehicle', {}).get(vid, 0.)
        calendar = getattr(self, '_calendar', {r:[None]*self.planning_horizon for r in ids})
        opportunities = self._reserved_opportunities(s, calendar)
        self._feature_opportunities = opportunities
        self._cache_values = {}
        # Public intent demand includes already released FIFO queues, never unseen jobs.
        from collections import Counter
        demands = {r:set() for r in ids};frequencies={r:Counter() for r in ids}
        delivery_offsets={};elapsed=0.
        for request,w in graphs:
            for nid in w['execution_order']:
                if request==rid and nid in done:continue
                n=lookup[nid][1];delivery_offsets[nid]=elapsed
                elapsed+=ceil(n['input_size']*rates['profile_unit_bytes']/rates['input_bytes_per_frame'])+ceil(max(1.,n['input_size']*rates['profile_unit_bytes']/rates['workload_bytes_per_unit'])/rates['rsu_work_units_per_frame'])+ceil(n['output_size']*rates['profile_unit_bytes']/rates['output_bytes_per_frame'])
        for owner,d in getattr(self, '_delivery_decisions', {}).items():
            target = getattr(self, '_team_targets', {}).get(owner)
            if target not in demands:continue
            st = d['semantic_state']; ws = [st['workflow'], *[q['workflow'] for q in st['queued_workflows']]]
            for w in ws:
                for n in w['nodes']:
                    if n['node_id'] not in w['completed_node_ids']:
                        frequencies[target][f"adapter:{n['required_adapter']}"]+=1
                        demands[target].add(f"adapter:{n['required_adapter']}")
                        demands[target].add(f"base:{n['required_base_model']}")
        own = np.array([frame/480, len(done)/max(1,len(s['workflow']['nodes'])),
            (len(nodes)-len(done))/64, s['current_node_work_units_progress']/max(1.,s['current_node_work_units_required']),
            scale(s['current_node_work_units_required']/rates['rsu_work_units_per_frame']),
            scale(s['execution_state_payload_bytes']/rates['state_bytes_per_frame']),s['execution_state_version']/480,
            s['current_node_parents_ready'],s['vehicle_adapter_ready'],bool(s['output_pending']),
            len(s['input_received_targets'])/4,confidence if opportunities else 0.,
            pred.get('dwell_time',{}).get(vid,0)/self.planning_horizon if opportunities else 0.,
            s['shared_state'].get('available',False),(s['shared_state'].get('age_frames') or 0)/8,
            len(s['reachable_execution_rsu_ids'])/4,len(graphs)-1,len(graphs)/2,
            min(1.,work/max(1,horizon)),scale(queued_work)],np.float32)
        graph = np.zeros((64,4),np.float32); adjacency = np.zeros((64,64),np.float32)
        for i,(request,n) in enumerate(nodes):
            graph[i] = [scale(n['input_size']*rates['profile_unit_bytes']/rates['workload_bytes_per_unit']/rates['rsu_work_units_per_frame']),
                scale(n['output_size']*rates['profile_unit_bytes']/rates['output_bytes_per_frame']),
                request == rid and n['node_id'] in done,n['node_id']==current['node_id']]
        for _,w in graphs:
            for a,b in w['edges']:adjacency[index[b],index[a]]=1
        adjacency /= np.maximum(1,adjacency.sum(1,keepdims=True))
        queue = {r:sum(w for v,(target,w) in reservations.items() if v!=vid and target==r) for r in ids}
        def feature(site=None,n=None,kind='adapter',priority=0):
            n = n or current; r = rows.get(site,{}); request = lookup[n['node_id']][0]
            cap = max(1,r.get('capacity_bytes') or model_byte_scale); base = catalog[f"base:{n['required_base_model']}"]; adapter = catalog[f"adapter:{n['required_adapter']}"]
            closure = {f"base:{n['required_base_model']}",f"adapter:{n['required_adapter']}"}
            demand = demands.get(site,set()) | closure
            working_bytes = sum(catalog[k]['resident_size_bytes'] for k in demand)
            alternate = dict(s,control_cost_parameters=dict(rates,base_size_bytes=base['transfer_size_bytes'],adapter_size_bytes=adapter['transfer_size_bytes']))
            span = fresh_model_span(alternate,r,n,kind) if site in rows else 0
            relocation = 0.
            if site in rows and s['execution_state_version'] and site!=s['execution_location']:
                relocation=state_restore_frames(s,site)
            op=opportunities.get(site,{}) if n['node_id']==current['node_id'] else {}
            # Risk is a feature, not a label claimed to be causal net reward.
            risk=max(0.,working_bytes/cap-1) if site in rows else 0.
            return [float(site in rows),float(site in s['reachable_execution_rsu_ids']),float(site==s['execution_location']),float(site==s['associations'][vid]),
                float(n['required_adapter'] in r.get('cached_adapter_ids',[])),float(n['required_adapter'] in r.get('loading_adapter_ids',[])),
                float(n['required_base_model'] in r.get('ready_base_model_ids',[])),r.get('occupied_bytes',0)/cap,
                r.get('service_request_count',0)/8,float(site in s['input_received_targets']),
                float(site==future)*confidence,float(pred.get('future_load',{}).get(site,0))/8,
                float(n['node_id']==current['node_id']),scale(n['input_size']*rates['profile_unit_bytes']/rates['input_bytes_per_frame']),float(kind=='base_model'),priority/3,
                *[float(site==r) for r in ids],base['resident_size_bytes']/cap if site in rows else 0.,adapter['resident_size_bytes']/cap if site in rows else 0.,
                scale(queue.get(site,0)),scale(relocation),np.tanh(op.get('net_reward_proxy',0.)),
                min(1.,frequencies.get(site,{}).get(f"adapter:{n['required_adapter']}",0)/8),min(4.,risk),scale(delivery_offsets.get(n['node_id'],0.)),
                (len(nodes)-len(done))/64,scale(span),np.clip((horizon-work-queue.get(site,0)-relocation)/max(1,horizon),-1,1),
                scale(n['output_size']*rates['profile_unit_bytes']/rates['output_bytes_per_frame'])]
        targets=[None,*ids,'vehicle']; cache=[(None,None,'adapter')]; event=[(r,p) for r in [None,*ids] for p in range(4)]
        f0=[feature(r) for r in targets]; mask=[r in decision['control_space']['execution_targets'] or (r=='vehicle' and decision['control_space']['vehicle_execution_available']) for r in targets]
        if any(mask[1:]):mask[0]=False
        old=self.target_memory.get(vid)
        if old and old[0]==current['node_id'] and old[1] in s['reachable_execution_rsu_ids'] and (s['current_node_work_units_progress']>0 or s['output_pending'] or frame-old[2]<8):mask=[r==old[1] for r in targets]
        f1=[feature()]; m1=[True]; ni1=[-1]
        for request,n in nodes:
            for site in ids:
                row=rows[site]
                for kind in ('base_model','adapter'):
                    cache.append((site,n['node_id'],kind));ni1.append(index[n['node_id']])
                    ready=n['required_base_model'] in row.get('ready_base_model_ids',[]) if kind=='base_model' else n['required_adapter'] in row.get('cached_adapter_ids',[]) or n['required_adapter'] in row.get('loading_adapter_ids',[])
                    reactive=request==rid and n['node_id']==current['node_id'] and getattr(self,'_team_targets',{}).get(vid)==site and not s['output_pending']
                    grant=calendar[site][0]
                    valid=not ready and (request!=rid or n['node_id'] not in done) and (reactive or frame%4==0)
                    valid &= grant is None or (reactive and grant==(vid,'model'))
                    base=catalog[f"base:{n['required_base_model']}"];adapter=catalog[f"adapter:{n['required_adapter']}"]
                    alt=dict(s,control_cost_parameters=dict(rates,base_size_bytes=base['transfer_size_bytes'],adapter_size_bytes=adapter['transfer_size_bytes']))
                    span=fresh_model_span(alt,row,n,kind)
                    valid &= not span or (span<=len(calendar[site]) and all(x is None or (reactive and x==(vid,'model')) for x in calendar[site][:span]))
                    value = None
                    if not reactive:
                        valid &= protects_current_models(alt,row,n,getattr(self,'_delivery_decisions',{}),getattr(self,'_team_targets',{}),catalog)
                        # A completed current node awaiting output is not future model demand.
                        valid &= not (request==rid and n['node_id']==current['node_id'] and bool(s['output_pending']))
                        if valid:
                            demand=demands.get(site,set()) | {f"base:{n['required_base_model']}",f"adapter:{n['required_adapter']}"}
                            value=optional_cache_value(s,row,n,span,getattr(self,'_team_targets',{}).get(vid),
                                sum(catalog[k]['resident_size_bytes'] for k in demand),
                                prediction_enabled=self.prediction_enabled,horizon=self.planning_horizon,
                                use_after_override=opportunities[site].get('eta_frames') if request==rid and n['node_id']==current['node_id'] and site in opportunities and opportunities[site].get('cache_net_reward_proxy',0)>0 else None)
                            valid &= value['eligible']
                    if value is not None:self._cache_values[site,n['node_id'],kind]=value
                    m1.append(valid)
                    f1.append(feature(site,n,kind) if valid else [0.]*32)
        m2=[r is None or opportunities.get(r,{}).get('prepare_feasible',False) for r,p in event]
        critic=list(own)
        c=decision['centralized_critic_state']
        for row in sorted(c['rsus'],key=lambda r:r['rsu_id']):
            critic += [row.get('occupied_bytes',0)/max(1,row.get('capacity_bytes',1)),row.get('associated_vehicle_count',0)/8,
                len(row.get('loading_adapter_ids',[]))/16,sum(w['execution_location']==row['rsu_id'] for w in c['joint_workflows'].values())/8]
        if self.method=='ppo':critic[20:]=[0.]*16
        data=dict(own=own,critic=np.asarray(critic,np.float32),graph=graph,adjacency=adjacency,graph_mask=np.arange(64)<len(nodes),
            prior0=np.asarray([-(queue.get(r,0)+np.expm1(float(f0[i][23])*np.log1p(480)))/(max(4,horizon)/4) if r in rows else 0. for i,r in enumerate(targets)],np.float32),
            domain=np.asarray([len(nodes)/64,sum(len(w['edges']) for _,w in graphs)/128,scale(work),len(done)/max(1,len(nodes))],np.float32),opportunity_count=np.asarray(len(opportunities),np.int64))
        # Public physical cost prior shared by SA and every comparator.  Cold
        # model closures and current family conflicts used to be omitted here.
        for i,site in enumerate(targets):
            if site not in rows:continue
            row=rows[site];base=catalog[f"base:{current['required_base_model']}"];adapter=catalog[f"adapter:{current['required_adapter']}"]
            alt=dict(s,control_cost_parameters=dict(rates,base_size_bytes=base['transfer_size_bytes'],adapter_size_bytes=adapter['transfer_size_bytes']))
            loading=row.get('loading_remaining_frames',{})
            cold=fresh_model_span(alt,row,current)+sum(loading.get(k,0) for k in (f"base:{current['required_base_model']}",f"adapter:{current['required_adapter']}"))
            closure={f"base:{current['required_base_model']}",f"adapter:{current['required_adapter']}"}
            peer=set()
            for owner,d in getattr(self,'_delivery_decisions',{}).items():
                if owner!=vid and getattr(self,'_team_targets',{}).get(owner)==site and not d['semantic_state']['output_pending']:
                    n=d['semantic_state']['current_workflow_node'];peer.update((f"base:{n['required_base_model']}",f"adapter:{n['required_adapter']}"))
            overflow=max(0.,sum(catalog[k]['resident_size_bytes'] for k in closure|peer)/max(1,row.get('capacity_bytes') or model_byte_scale)-1)
            conflict=overflow*sum(ceil(catalog[k]['transfer_size_bytes']/rates['model_bytes_per_frame']) for k in peer-closure)
            data['prior0'][i]-=(cold+conflict)/max(1.,max(4,horizon)/4)
        for j,(features,masks,indices) in enumerate([(f0,mask,[index[current['node_id']]]*6),(f1,m1,ni1),([feature(r,priority=p) for r,p in event],m2,[index[current['node_id']]]*20)]):
            data[f'f{j}']=np.asarray(features,np.float32);data[f'm{j}']=np.asarray(masks,bool);data[f'node{j}']=np.asarray(indices,np.int64)
        return data,(targets,cache,event),vid,current['node_id'],work

    @torch.no_grad()
    def act_controls(self,decisions,frame,*,deterministic=False):
        from copy import deepcopy
        from dataclasses import replace
        from src.envs.core.multi_request_stream_env import StreamingRequestControl
        from src.envs.core.joint_delivery_planning import reactive_link_calendar,book_delivery_control
        from src.envs.core.joint_link_reservation import control_link_claims
        if not decisions:
            self.last_diagnostics=dict(migration_opportunities=0,by_vehicle={});return {},None
        for vid,d in decisions.items():
            rid=d['semantic_state']['request_id']
            if self.request_memory.get(vid)!=rid:self.target_memory.pop(vid,None);self.commitments.pop(vid,None)
            self.request_memory[vid]=rid
        self._delivery_decisions=decisions
        targets={v:self.target_memory[v][1] if v in self.target_memory else d['semantic_state']['execution_location'] or d['semantic_state']['associations'][v] for v,d in decisions.items()}
        self._team_targets=targets;self._calendar=reactive_link_calendar(decisions,targets,self.planning_horizon)
        ids=sorted(decisions);shift=frame%len(ids);ids=ids[shift:]+ids[:shift]
        reservations={};first={};chosen={}
        for vid in ids:
            data,candidates,_,node,work=self._coordinated_features(decisions[vid],frame,reservations)
            batch={k:torch.as_tensor(v).unsqueeze(0) for k,v in data.items()}
            actions,*_=self._evaluate(batch,deterministic=deterministic,execution_only=True)
            chosen[vid]=actions[:,0];targets[vid]=candidates[0][int(actions[0,0])];first[vid]=data
            reservations[vid]=(targets[vid],work)
        self._calendar=reactive_link_calendar(decisions,targets,self.planning_horizon)
        controls={};rolls=[];diagnostics={};opportunities=0
        for vid in ids:
            fresh,candidates,_,node,work=self._coordinated_features(decisions[vid],frame,reservations)
            data=dict(first[vid])
            for key in ('f1','m1','node1','f2','m2','node2','opportunity_count'):data[key]=fresh[key]
            batch={k:torch.as_tensor(v).unsqueeze(0) for k,v in data.items()}
            a,logs,entropy,value,surrogate,prefix,enabled,baseline,q,_=self._evaluate(batch,deterministic=deterministic,execution_actions=chosen[vid],return_details=True)
            cache,nid,kind=candidates[1][int(a[0,1])];prepare,priority=candidates[2][int(a[0,2])]
            state=decisions[vid]['semantic_state'];known=[(state['request_id'],state['workflow']),*[(z['request_id'],z['workflow']) for z in state['queued_workflows']]]
            cache_request=next((request for request,w in known if any(n['node_id']==nid for n in w['nodes'])),None)
            control=StreamingRequestControl(targets[vid],cache,nid,kind,prepare,priority=priority,request_id=state['request_id'],cache_request_id=cache_request)
            # Booking consumes the selected object's heterogeneous size, including queued DAG nodes.
            planning=deepcopy(state);planning['workflow']['nodes']=[n for _,w in known for n in w['nodes']]
            if nid is not None:
                n=next(n for n in planning['workflow']['nodes'] if n['node_id']==nid);cat=state['control_cost_parameters']['model_catalog']
                planning['control_cost_parameters'].update(base_size_bytes=cat[f"base:{n['required_base_model']}"]['transfer_size_bytes'],adapter_size_bytes=cat[f"adapter:{n['required_adapter']}"]['transfer_size_bytes'])
            claims=control_link_claims(planning,control,self._calendar);controls[vid]=replace(control,link_reservations=claims)
            from src.envs.core.joint_delivery_planning import fresh_model_span
            span=fresh_model_span(planning,next(r for r in state['rsus'] if r['rsu_id']==cache),n,kind) if cache is not None else 0
            diagnostics[vid]=dict(migration_candidates=deepcopy(self._feature_opportunities),link_calendar=deepcopy(self._calendar),two_pass_execution_targets=dict(targets),request_id=state['request_id'],cache_admission_span=span,
                cache_value=deepcopy(self._cache_values.get((cache,nid,kind))))
            book_delivery_control(self._calendar,vid,claims,planning,control)
            old=self.target_memory.get(vid);since=old[2] if old and old[:2]==(node,targets[vid]) else frame
            self.target_memory[vid]=(node,targets[vid],since);opportunities+=len(self._feature_opportunities)
            rolls.append(dict(data=batch,actions=a,logs=logs.sum(1),head_logs=logs,values=value,prefix_values=prefix,active_heads=enabled,counterfactual_baselines=baseline))
        self._pad_cache(rolls,max(r['data']['m1'].shape[1] for r in rolls))
        result={k:torch.cat([r[k] for r in rolls]) for k in rolls[0] if k!='data'}
        result['data']={k:torch.cat([r['data'][k] for r in rolls]) for k in rolls[0]['data']};result['vehicle_ids']=ids
        result['request_ids']=[decisions[v]['semantic_state']['request_id'] for v in ids]
        self.last_diagnostics=dict(migration_opportunities=opportunities,by_vehicle=diagnostics)
        return controls,result
