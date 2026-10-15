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

    def __init__(self, *, credit_mode='request_team_blend', migration_gate_enabled=True, coordination_enabled=True, partial_preparation_enabled=False, **kwargs):
        kwargs.pop('productive_controls',None)
        super().__init__(productive_controls=True, **kwargs)
        if credit_mode not in ('request_team_blend','team'):
            raise ValueError('unknown coordinated credit mode')
        self.credit_mode, self.migration_gate_enabled = credit_mode, bool(migration_gate_enabled)
        self.coordination_enabled = bool(coordination_enabled)
        self.partial_preparation_enabled = bool(partial_preparation_enabled)
        self.options.update(credit_mode=credit_mode,migration_gate_enabled=migration_gate_enabled,coordination_enabled=coordination_enabled,partial_preparation_enabled=partial_preparation_enabled)
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
        opportunities = migration_opportunities(state,prediction_enabled=self.prediction_enabled,allow_partial=self.partial_preparation_enabled)
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
            diagnostics[vid]=dict(migration_candidates=migration_opportunities(decisions[vid]['semantic_state'],prediction_enabled=self.prediction_enabled,allow_partial=self.partial_preparation_enabled),
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
