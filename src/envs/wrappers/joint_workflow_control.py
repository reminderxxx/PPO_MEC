"""Explicit independent service/preparation contract over native cache physics."""
from copy import deepcopy
from src.envs.wrappers.ready_workflow_control import ReadyWorkflowEnv, ready_node_ids
from src.envs.wrappers.persistent_workflow_control import CONTROL_CONTRACT
from src.envs.specs import ControlAction

JOINT_CONTRACT = 'released_joint_ready_prepare_v1'


class JointWorkflowEnv(ReadyWorkflowEnv):
    """A ready task plus optional released adapter at the observed current RSU.

    No auto-admission, fallback, cache shielding or future trajectory access.
    The inherited step_node contract remains available for matched baselines.
    """
    def __init__(self, service_hit_lru=False, **kwargs):
        self.service_hit_lru=bool(service_hit_lru)
        super().__init__(**kwargs)

    def _after_selected_step(self, state, reward, terminated, truncated, info):
        event=info['persistent_stream']['cache_event']
        if self.service_hit_lru and event.get('service_success') and event.get('served_rsu_id'):
            for row in event.get('per_object_lookup_results',[]):
                if row.get('resident') and row['object_type'] in ('base_model','adapter'):
                    self._eviction_policy.on_hit(rsu_id=event['served_rsu_id'],object_id=row['object_id'],current_step=self._episode_steps)
        return state, reward, terminated, truncated, info

    def step_joint(self, node_id, prepare_adapter=None):
        if self._stream_done:
            raise RuntimeError('Terminal stream requires reset')
        before = deepcopy(self._last_state)
        if node_id not in ready_node_ids(before):
            raise ValueError('Selected task is not dependency-ready')
        allowed = {n.required_adapter for n in self.workflow_state.nodes}
        if prepare_adapter is not None and prepare_adapter not in allowed:
            raise ValueError('Unreleased/unknown preparation adapter')
        vehicle = next(v for v in before['vehicles'] if v['vehicle_id'] == self.subject_id)
        rsu = vehicle.get('associated_rsu_id')
        if prepare_adapter is not None and rsu is None:
            raise ValueError('Preparation requires an observed current RSU')
        node = self.workflow_state.node_map()[node_id]
        self.workflow_state.current_node_id = node_id
        self._last_state['workflow'] = self.workflow_state.to_dict()
        self._last_state['current_workflow_node'] = node.to_dict()
        cache = {} if prepare_adapter is None else dict(operation='cache',
            adapter_id=prepare_adapter, rsu_id=rsu, strategy='independent_released_preparation',
            prediction_driven=False)
        control = ControlAction(cache_action=cache,
            offload_action=dict(mode='rsu', strategy='joint_selected_ready_node'),
            migration_action=dict(mode='keep', strategy='none'),
            metadata=dict(primary_vehicle_id=self.subject_id,
                persistent_control_contract=CONTROL_CONTRACT, action_contract=JOINT_CONTRACT,
                selected_node_id=node_id, prepare_adapter=prepare_adapter,
                invalid_action=False, invalid_reason='none'))
        state, reward, terminated, truncated, info = super(ReadyWorkflowEnv, self).step(control)
        info['joint_control'] = dict(contract=JOINT_CONTRACT, selected_node_id=node_id,
            prepare_adapter=prepare_adapter, observed_rsu_id=rsu)
        return self._after_selected_step(state, reward, terminated, truncated, info)


def observe_joint_effects(env, state):
    """Read-only causal cache transaction forecasts; no mobility/service rollout.

    All policies may consume this same view. Forecasts describe the observed RSU
    and native LRU state, not a guarantee about the next mobility transition.
    """
    from src.envs.core.persistent_workflow_env import ADAPTERS
    state = deepcopy(state)
    snapshot = env.export_cache_trace_snapshot()
    policy_before = env._eviction_policy.export_state()
    vehicle = next(v for v in state['vehicles'] if v['vehicle_id'] == env.subject_id)
    rsu_id = vehicle.get('associated_rsu_id')
    rsu = next((r for r in env.rsu_states if r.rsu_id == rsu_id), None)
    effects = {}
    if rsu is not None:
        residents = list(env._typed_resident_object_ids.get(rsu_id, []))
        used = sum(env.adapter_catalog.get_typed_object(o).resident_size_mb for o in residents)
        for adapter_id in ADAPTERS:
            adapter = env.adapter_catalog.get_typed_adapter(adapter_id)
            incoming = [*adapter.dependency_ids, adapter.object_id]
            missing = [o for o in incoming if o not in residents]
            need = max(used + sum(env.adapter_catalog.get_typed_object(o).resident_size_mb for o in missing) - env._cache_capacity_value(), 0.)
            if need > 1e-9:
                plan, _ = env._plan_sequential_typed_lru_evictions(rsu=rsu, residents=residents,
                    required_free_capacity=need, protected_object_ids=set(incoming))
                victims = plan.ordered_victim_ids
                sufficient = plan.sufficient
            else:
                victims = []; sufficient = True
            effects[adapter_id] = dict(victims=list(victims), sufficient=sufficient,
                transfer_mb=sum(env.adapter_catalog.get_typed_object(o).transfer_size_mb for o in missing),
                required_free_mb=need)
    if policy_before != env._eviction_policy.export_state():
        raise RuntimeError('Cache forecast mutated native eviction state')
    state['joint_cache_observation'] = dict(version='observable_lru_effects_v1', service_hit_lru=env.service_hit_lru,
        snapshot=snapshot, policy=policy_before, candidate_effects=effects)
    return state
