"""Opt-in ready-node scheduling over the unchanged persistent physical model.

This contract changes the decision space, not the original FIFO experiment.
Only released, dependency-ready tasks may be selected. Loading, capacity,
transfer charges and the original finite-horizon business reward are inherited.
"""
from copy import deepcopy
from src.envs.wrappers.persistent_workflow_control import ReleasedPreparationEnv, CONTROL_CONTRACT
from src.envs.specs import ControlAction

READY_CONTRACT = 'released_ready_node_v1'


def ready_node_ids(state):
    workflow = state['workflow']
    done = set(workflow['completed_node_ids'])
    return [n['node_id'] for n in workflow['nodes']
            if n['node_id'] not in done and set(n['predecessors']) <= done]


class ReadyWorkflowEnv(ReleasedPreparationEnv):
    """Explicit scheduling; no policy-dependent remapping or expert fallback."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        nodes = self._workflow_template.node_map()
        edges = set(self._workflow_template.edges)
        if len(nodes) != len(self._workflow_template.nodes):
            raise ValueError('Duplicate scheduling node identity')
        predecessors = {(p, n.node_id) for n in nodes.values() for p in n.predecessors}
        successors = {(n.node_id, s) for n in nodes.values() for s in n.successors}
        if edges != predecessors or edges != successors or any(a not in nodes or b not in nodes for a,b in edges):
            raise ValueError('Inconsistent scheduling dependency graph')
        order = self._workflow_template.execution_order
        rank = {n:i for i,n in enumerate(order)}
        if len(order) != len(nodes) or set(order) != set(nodes) or any(rank[a] >= rank[b] for a,b in edges):
            raise ValueError('Scheduling graph must have a valid topological order')

    def _annotate(self, state):
        state = super()._annotate(state)
        state['ready_scheduling_contract'] = READY_CONTRACT
        state['ready_node_ids'] = ready_node_ids(state)
        return state

    def step_node(self, node_id):
        if self._stream_done:
            raise RuntimeError('Terminal stream requires reset')
        before = deepcopy(self._last_state)
        if node_id not in ready_node_ids(before):
            raise ValueError('Selected task is not dependency-ready')
        node = self.workflow_state.node_map()[node_id]
        self.workflow_state.current_node_id = node_id
        self._last_state['workflow'] = self.workflow_state.to_dict()
        self._last_state['current_workflow_node'] = node.to_dict()
        vehicle = next(v for v in before['vehicles'] if v['vehicle_id'] == self.subject_id)
        rsu = vehicle.get('associated_rsu_id')
        resident = next((r for r in before['rsus'] if r['rsu_id'] == rsu), {})
        admission = bool(rsu and node.required_adapter not in resident.get('cached_adapter_ids', []))
        cache = dict(operation='cache', adapter_id=node.required_adapter, rsu_id=rsu,
                     strategy='ready_node_selected_admission', prediction_driven=False) if admission else {}
        control = ControlAction(cache_action=cache,
            offload_action=dict(mode='rsu', strategy='selected_ready_node'),
            migration_action=dict(mode='keep', strategy='none'),
            metadata=dict(primary_vehicle_id=self.subject_id, persistent_control_contract=CONTROL_CONTRACT,
                          ready_scheduling_contract=READY_CONTRACT, selected_node_id=node_id,
                          invalid_action=False, invalid_reason='none'))
        state, reward, terminated, truncated, info = super().step(control)
        info['ready_scheduling'] = dict(contract=READY_CONTRACT, selected_node_id=node_id,
            ready_before=before['ready_node_ids'], admission_requested=admission)
        return state, reward, terminated, truncated, info
