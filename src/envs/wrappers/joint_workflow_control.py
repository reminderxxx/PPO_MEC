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
        return state, reward, terminated, truncated, info
