"""Opt-in FIFO request arrivals; caches persist within one causal mobility session."""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Any
from src.envs.specs import WorkflowGraphState
from src.envs.core.joint_model_cache import JointModelCache, ModelCacheObject
from src.envs.core.multi_request_target_control_env import MultiRequestTargetControlEnv, JointRequestControl


@dataclass(frozen=True)
class WorkflowArrival:
    frame: int
    vehicle_id: str
    workflow: WorkflowGraphState

    @property
    def request_id(self):
        return f'{self.vehicle_id}/{self.workflow.workflow_id}'


@dataclass(frozen=True)
class StreamingRequestControl(JointRequestControl):
    request_id: str = ''
    cache_request_id: str | None = None


class MultiRequestStreamEnv(MultiRequestTargetControlEnv):
    action_contract = 'joint_stream_control_v1'
    stream_contract = 'joint_fifo_workload_stream_v2'

    def __init__(self, *, frames, arrivals, catalog, **kwargs):
        self.arrivals = sorted(deepcopy(list(arrivals)), key=lambda a: (a.frame, a.vehicle_id))
        self.catalog_objects = list(catalog)
        ids = [v.vehicle_id for v in frames[0]['vehicles']]
        if not self.arrivals or len({a.request_id for a in self.arrivals}) != len(self.arrivals):
            raise ValueError('nonempty unique request identities required')
        catalog_map = {o.object_id: o for o in self.catalog_objects}
        for a in self.arrivals:
            w = a.workflow
            if type(a.frame) is not int or not 0 <= a.frame < len(frames) or a.vehicle_id not in ids:
                raise ValueError('arrival must be inside session for a present vehicle')
            if not w.nodes or w.is_completed or w.completed_node_ids or w.current_node_id != w.execution_order[0]:
                raise ValueError('arriving workflows must be fresh')
            for n in w.nodes:
                obj = catalog_map.get(f'adapter:{n.required_adapter}')
                if obj is None or obj.dependency_ids != (f'base:{n.required_base_model}',):
                    raise ValueError('workflow model is absent or incompatible with catalog')
        dummy = {v: WorkflowGraphState(f'idle:{v}', [], [], [], is_completed=True) for v in ids}
        super().__init__(frames=frames, vehicle_workflows=dummy, **kwargs)
        self.observation_contract = 'joint_stream_observation_v1'

    def _initialize_model_cache(self):
        config = dict(capacity_bytes=4096, base_size_bytes=2048, adapter_size_bytes=512, transfer_bytes_per_frame=1024)
        if set(self.model_cache_config)-set(config):
            raise ValueError('unknown model cache options')
        config.update(self.model_cache_config); self.model_cache_config = config
        self.model_cache = JointModelCache([r.rsu_id for r in self.rsus], capacity_bytes=config['capacity_bytes'],
            objects=self.catalog_objects, link_slots=self.costs.link_slots_per_rsu,
            transfer_bytes_per_frame=config['transfer_bytes_per_frame'])
        for o in self.catalog_objects:
            closure = o.resident_size_bytes + sum(self.model_cache.objects[d].resident_size_bytes for d in o.dependency_ids)
            if any(closure > n for n in self.model_cache.capacity_bytes.values()):
                raise ValueError('a complete model closure cannot fit')
        self._sync_cache_view()

    def reset(self):
        self._stream_ready = False
        super().reset()
        self._released = set(); self._active = {}; self._queues = defaultdict(list)
        self._finished = {}; self._started = {}; self.frame_receipts = []
        self.observation_contract = 'joint_stream_observation_v1'
        self._stream_ready = True
        self._release_and_start()
        return self.observe()

    def _clear_request_execution(self, vid):
        for field, value in [('state_version', 0), ('state_payload_bytes', 0), ('execution_location', None),
                             ('work_progress', 0.), ('last_service_rsu', None), ('handoff_wait', 0),
                             ('handoff_target', None), ('vehicle_compute_progress', 0)]:
            getattr(self, field)[vid] = value
        self.previous_service_targets.pop(vid, None)
        for field in ['state_copies', 'state_transfers', 'state_transfer_remaining', 'input_remaining']:
            mapping = getattr(self, field)
            for key in list(mapping):
                if key[0] == vid: mapping.pop(key)
        self.input_received = {key for key in self.input_received if key[0] != vid}
        self.prepared_state[vid] = set()
        self.cold_restart_done = {key for key in self.cold_restart_done if key[0] != vid}
        self.output_pending.pop(vid, None)

    def _release_and_start(self):
        if self.frame_index >= len(self.frames): return
        for a in self.arrivals:
            if a.frame <= self.frame_index and a.request_id not in self._released:
                self._released.add(a.request_id); self._queues[a.vehicle_id].append(a)
        for vid in self.workflows:
            if vid not in self._active and self._queues[vid]:
                a = self._queues[vid].pop(0)
                self._clear_request_execution(vid)
                self.workflows[vid] = deepcopy(a.workflow)
                self._active[vid] = a; self._started[a.request_id] = self.frame_index

    def _control_episode_done(self):
        if self.frame_index >= len(self.frames): return True
        if not getattr(self, '_stream_ready', False): return False
        complete = set(self._finished) | {a.request_id for v,a in self._active.items() if self.workflows[v].is_completed}
        return len(complete) == len(self.arrivals)

    def _waiting_request_count(self):
        return sum(not self.workflows[v].is_completed for v in self._active) + sum(len(q) for q in self._queues.values())

    def observe(self):
        obs = super().observe()
        if getattr(self, '_stream_ready', False):
            obs.update(workload_contract=self.stream_contract, action_contract=self.action_contract,
                active_request_ids={v:a.request_id for v,a in self._active.items()},
                released_queue_lengths={v:len(self._queues[v]) for v in self.workflows})
        return obs

    def decision_states(self):
        states = super().decision_states()
        for vid,d in states.items():
            state=d['semantic_state']; state['request_id']=self._active[vid].request_id
            state['workload_contract']=self.stream_contract
            state['queued_workflows']=[{'request_id':a.request_id,'arrival_frame':a.frame,'workflow':a.workflow.to_dict()} for a in self._queues[vid]]
            state['control_cost_parameters']['model_catalog']={o.object_id:asdict(o) for o in self.catalog_objects}
            node=state['current_workflow_node']; objects=self.model_cache.objects
            state['control_cost_parameters']['base_size_bytes']=objects[f"base:{node['required_base_model']}"].transfer_size_bytes
            state['control_cost_parameters']['adapter_size_bytes']=objects[f"adapter:{node['required_adapter']}"].transfer_size_bytes
            d['action_contract']=self.action_contract
        return states

    def _cache_node_for_control(self, vid, control):
        rid = control.cache_request_id or control.request_id
        visible = [self._active[vid], *self._queues[vid]]
        arrival = next((a for a in visible if a.request_id == rid), None)
        if arrival is None:
            raise ValueError('cache request must be a released active or queued request')
        node = next((n for n in arrival.workflow.nodes if n.node_id == control.cache_node_id), None)
        if node is None:
            raise ValueError('cache node must belong to the known DAG')
        return node

    def _validate_controls(self, controls, decisions):
        for vid,c in controls.items():
            if not isinstance(c, StreamingRequestControl):
                raise ValueError('stream contract requires request-bound controls; legacy policies are incompatible')
            if c.cache_request_id is not None and c.cache_node_id is None:
                raise ValueError('cache request identity requires a node')
            if vid not in self._active or c.request_id != self._active[vid].request_id:
                raise ValueError('stale or unknown request control')
        return super()._validate_controls(controls, decisions)

    def _request_credit_ledger(self, events, components):
        exposed = [a.request_id for a in self._active.values()] + [a.request_id for q in self._queues.values() for a in q]
        return {rid:sum(components.values())/len(exposed) for rid in exposed}

    def step_controls(self, controls):
        exposed = [a.request_id for a in self._active.values()] + [a.request_id for q in self._queues.values() for a in q]
        _, reward, _, info = super().step_controls(controls)
        receipt=info['frame_cost_receipt']; receipt['workload_contract']=self.stream_contract
        # Explicit team shares only: not counterfactual/action-specific attribution.
        receipt['credit_contract']='uniform_released_pending_team_share_not_causal'
        receipt['request_rewards']={rid:reward/len(exposed) for rid in exposed}
        receipt['unassigned_team_reward']=reward if not exposed else 0.
        self.frame_receipts.append(deepcopy(receipt))
        for vid,a in list(self._active.items()):
            if self.workflows[vid].is_completed:
                self._finished[a.request_id]={'completion_frame':self.frame_index-1,'completed_nodes':len(self.workflows[vid].completed_node_ids)}
                self._active.pop(vid)
        self._release_and_start()
        done=self._control_episode_done()
        return self.observe() if not done else {'frame_index':self.frame_index}, reward, done, info

    def summary(self):
        result=super().summary(); rows=[]
        for a in self.arrivals:
            active=self._active.get(a.vehicle_id)
            completed=a.request_id in self._finished
            count=(self._finished[a.request_id]['completed_nodes'] if completed else len(self.workflows[a.vehicle_id].completed_node_ids) if active and active.request_id==a.request_id else 0)
            rows.append(dict(vehicle_id=a.vehicle_id,request_id=a.request_id,workflow_id=a.workflow.workflow_id,
                arrival_frame=a.frame,start_frame=self._started.get(a.request_id),released=a.request_id in self._released,
                completed=completed,completed_nodes=count,total_nodes=len(a.workflow.nodes),completion_frame_index=self._finished.get(a.request_id,{}).get('completion_frame')))
        result.update(workload_contract=self.stream_contract,action_contract=self.action_contract,
            request_count=len(rows),released_requests=len(self._released),completed_requests=len(self._finished),
            completed_nodes=sum(r['completed_nodes'] for r in rows),per_request=rows,
            queued_requests=sum(len(q) for q in self._queues.values()),frame_receipt_count=len(self.frame_receipts))
        return result

    def configuration(self):
        return {**super().configuration(),'workload_contract':self.stream_contract,'action_contract':self.action_contract,
            'observation_contract':self.observation_contract,'arrival_contract':'exogenous_frame_release_per_vehicle_fifo',
            'same_frame_fifo_tie':'published_schedule_list_order',
            'cache_lifetime':'persistent_within_session_cold_on_reset_no_cross_split',
            'future_visibility':'unreleased_requests_hidden','catalog':[asdict(o) for o in self.catalog_objects]}
