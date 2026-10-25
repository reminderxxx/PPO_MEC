"""Opt-in causal five-way preparation contract; legacy GymVecEnv is unchanged."""
from copy import deepcopy
from src.envs.core.persistent_workflow_env import PersistentWorkflowEnv
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.envs.specs import ActionSchema, DiscreteActionSpec, ControlAction

CONTROL_CONTRACT = "released_fifo_prepare_5_v1"
ACTION_NAMES = ("fill_current", "prepare_first_future", "prepare_second_future", "steady", "prepare_predicted_rsu")


def candidates(state):
    node = state.get("current_workflow_node") or {}
    vehicles = state.get("vehicles", [])
    vehicle = next((v for v in vehicles if v["vehicle_id"] == state.get("primary_vehicle_id")), {})
    current = vehicle.get("associated_rsu_id")
    predicted = state.get("predictions", {}).get("predicted_next_rsu_by_vehicle", {}).get(vehicle.get("vehicle_id"))
    wf = state.get("workflow", {})
    by_id = {n["node_id"]: n for n in wf.get("nodes", [])}
    done = set(wf.get("completed_node_ids", []))
    future = []
    for node_id in wf.get("execution_order", []):
        if node_id in done or node_id == node.get("node_id"):
            continue
        adapter = by_id[node_id]["required_adapter"]
        if adapter != node.get("required_adapter") and adapter not in future:
            future.append(adapter)
    first, second = (future + [None, None])[:2]
    return [dict(action_id=i, name=ACTION_NAMES[i], adapter_id=a, rsu_id=r)
            for i, (a,r) in enumerate([(node.get("required_adapter"),current),(first,current),(second,current),(None,current),(node.get("required_adapter"),predicted if predicted != current else None)])]


class ReleasedPreparationEnv(PersistentWorkflowEnv):
    """Future placement is explicit and limited to already released FIFO nodes."""
    def _apply_cache_action(self, **kwargs):
        control = kwargs["control"]
        if control.metadata.get("persistent_control_contract") == CONTROL_CONTRACT and control.cache_action:
            adapter = control.cache_action["adapter_id"]
            allowed = {n.required_adapter for n in self.workflow_state.nodes}
            if adapter not in allowed:
                raise ValueError("Unreleased/unknown preparation adapter")
            kwargs["required_adapter"] = adapter
        return super()._apply_cache_action(**kwargs)

    def _annotate(self, state):
        state = super()._annotate(state)
        state["persistent_control_contract"] = CONTROL_CONTRACT
        state["cache_runtime"] = [dict(rsu_id=r.rsu_id,
            resident_mb=sum(self.adapter_catalog.get_typed_object(o).resident_size_mb for o in self._typed_resident_object_ids.get(r.rsu_id, [])),
            capacity_mb=float(self._cache_capacity_value()),
            queue_seconds=max(self._link_free_at.get(r.rsu_id,self._seconds())-self._seconds(),0.)) for r in self.rsu_states]
        return state


class PreparationMask:
    def build_mask_info(self,state):
        options = candidates(state)
        rsus = {r["rsu_id"]:r for r in state.get("rsus", [])}
        node = bool(state.get("current_workflow_node"))
        mask=[]
        for c in options:
            valid = node and (c["action_id"] == 3 or bool(c["adapter_id"] and c["rsu_id"]))
            if c["action_id"] != 3 and c["adapter_id"] in rsus.get(c["rsu_id"],{}).get("cached_adapter_ids",[]):
                valid=False  # repeated admission is exactly redundant with steady execution
            mask.append(bool(valid))
        if node and not any(mask):mask[3]=True
        return dict(mask=mask,valid_action_count=sum(mask),invalid_reasons={},semantic_preconditions={"contract":CONTROL_CONTRACT})

    def build_mask(self,state):
        return self.build_mask_info(state)["mask"]


class PreparationAdapter:
    def __init__(self,schema):self.schema=schema

    def decode(self,action,state):
        if action not in range(5):raise ValueError("Unknown preparation action")
        c=candidates(state)[action]
        mask=PreparationMask().build_mask(state)
        if not mask[action]:raise ValueError("Invalid preparation action; no fallback/rewrite permitted")
        cache={} if action==3 else dict(operation="cache",adapter_id=c["adapter_id"],rsu_id=c["rsu_id"],strategy="released_fifo_prepare",prediction_driven=action==4)
        return ControlAction(cache_action=cache,offload_action={"mode":"rsu","strategy":"current_rsu_steady_offload"},migration_action={"mode":"keep","strategy":"none"},metadata=dict(primary_vehicle_id=state.get("primary_vehicle_id"),action_id=action,action_name=ACTION_NAMES[action],action_contract=CONTROL_CONTRACT,persistent_control_contract=CONTROL_CONTRACT,invalid_action=False,invalid_reason="none"))


class PreparationGymEnv(GymVecEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._action_schema=ActionSchema([DiscreteActionSpec(i,n,n) for i,n in enumerate(ACTION_NAMES)])
        self._action_mask_builder=PreparationMask()
        self._action_adapter=PreparationAdapter(self._action_schema)

    def _build_decision_observation_trace_record(self,action):
        trace=super()._build_decision_observation_trace_record(action)
        trace["action_contract_version"]=CONTROL_CONTRACT
        trace["preparation_candidates"]=candidates(self._last_state)
        return trace

    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        info["action_name"]=ACTION_NAMES[action]
        return obs,reward,done,truncated,info
