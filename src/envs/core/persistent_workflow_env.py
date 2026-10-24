"""Opt-in finite-horizon FIFO batch. Costs/model assignments are simulated."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import math
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs import WorkflowGraphState, WorkflowNode, RewardBreakdown
from src.runtime.subject_binding import STRICT_SUBJECT_CONTRACT

CONTRACT = "persistent_fifo_batch_v1"
ADAPTERS = ("adapter_perception", "adapter_tracking", "adapter_fusion", "adapter_intent", "adapter_control")


def build_batch(workflows, request_count=8):
    if not workflows or request_count < 1:
        raise ValueError("A nonempty workflow batch is required")
    nodes, edges, order, requests = [], [], [], []
    for i in range(request_count):
        wf = workflows[i % len(workflows)]
        ids = {n.node_id: f"r{i}:{n.node_id}" for n in wf.nodes}
        for n in wf.nodes:
            data = n.to_dict()
            data.update(node_id=ids[n.node_id], predecessors=[ids[x] for x in n.predecessors],
                        successors=[ids[x] for x in n.successors], required_base_model="veh_base_v1")
            digest = hashlib.sha256(f"{wf.workflow_id}/{n.node_id}".encode()).digest()
            data["required_adapter"] = ADAPTERS[int.from_bytes(digest[:4], "big") % len(ADAPTERS)]
            nodes.append(WorkflowNode(**data))
        edges.extend((ids[x], ids[y]) for x, y in wf.edges)
        current_order = [ids[x] for x in wf.execution_order]
        order.extend(current_order)
        requests.append(dict(request_id=f"r{i}", source_workflow_id=wf.workflow_id,
                             node_ids=current_order, release_step=0))
    graph = WorkflowGraphState(workflow_id="fifo_batch", nodes=nodes, edges=edges,
                               execution_order=order, current_node_id=order[0])
    return graph, requests


class PersistentWorkflowEnv(VecWorkflowCoreEnv):
    """One fixed subject, persistent cache, released FIFO requests, finite H.

    Admission reserves capacity immediately. Serial per-RSU modeled loading
    delays service readiness. Eviction cancels readiness, but consumed bandwidth
    and admitted-byte costs are retained. This is not a wireless measurement.
    """
    def __init__(self, *, workflows, subject_id, request_count=8,
                 transfer_mb_per_second=128., load_seconds=.05, **kwargs):
        graph, self.requests = build_batch(workflows, request_count)
        self.subject_id = str(subject_id)
        self.transfer_rate, self.load_seconds = float(transfer_mb_per_second), float(load_seconds)
        if not math.isfinite(self.transfer_rate) or self.transfer_rate <= 0 or not math.isfinite(self.load_seconds) or self.load_seconds < 0:
            raise ValueError("Invalid modeled loading parameters")
        self._ready_at, self._link_free_at, self._admissions = {}, {}, {}
        self._stream_done = False
        self._completed_requests = set()
        super().__init__(workflow_state=graph, subject_binding_contract=STRICT_SUBJECT_CONTRACT,
                         reward_positive_offset=0., **kwargs)
        if not self._typed_mode_enabled() or not self._cache_capacity_enabled():
            raise ValueError("Persistent batch requires enabled typed MB capacity")

    def _initialize_primary_vehicle_id(self, vehicles):
        if not any(v.vehicle_id == self.subject_id for v in vehicles):
            raise ValueError("Frozen subject absent at reset")
        self._primary_vehicle_id = self.subject_id

    def _seconds(self):
        return float(self._mobility_provider.get_time()) / 1000.

    def reset(self):
        self._ready_at, self._link_free_at, self._admissions = {}, {}, {}
        self._completed_requests = set()
        self._stream_done = False
        state, info = super().reset()
        self.stream_rows, self.pending_area_seconds, self.stream_return = [], 0., 0.
        self._last_state = self._annotate(state)
        return deepcopy(self._last_state), info

    def _apply_cache_action(self, **kwargs):
        result = super()._apply_cache_action(**kwargs)
        rsu = result.get("target_rsu_id")
        for row in result.get("evicted_typed_objects", []):
            self._ready_at.pop((rsu,row["object_id"]),None)
            self._admissions.pop((rsu,row["object_id"]),None)
        admitted = result.get("admitted_typed_objects", [])
        if admitted:
            finish = max(self._seconds(), self._link_free_at.get(rsu,self._seconds()))
            for row in admitted:
                finish += float(row["transfer_size_mb"]) / self.transfer_rate + self.load_seconds
                key = (rsu,row["object_id"])
                self._ready_at[key] = finish
                self._admissions[key] = dict(step=self._episode_steps,strategy=result.get("strategy"),ready_at_seconds=finish)
            self._link_free_at[rsu] = finish
        return result

    def _typed_service_readiness(self, **kwargs):
        result = super()._typed_service_readiness(**kwargs)
        node,rsu = kwargs.get("current_node"),kwargs.get("service_rsu_id")
        if node is None or rsu is None or kwargs.get("offload_mode")=="vehicle":
            return result
        adapter = self.adapter_catalog.get_typed_adapter(node.required_adapter)
        loading = [x for x in [adapter.object_id,*adapter.dependency_ids]
                   if self._ready_at.get((rsu,x),-math.inf)>self._seconds()+1e-9]
        if loading:
            for object_id in loading:
                kind = self.adapter_catalog.get_typed_object(object_id).object_type
                if kind == "base_model": result["base_ready"] = False
                if kind == "adapter": result["adapter_ready"] = False
                if kind not in result["missing_object_types"]: result["missing_object_types"].append(kind)
            result.update(joint_base_adapter_hit=bool(result["base_ready"] and result["adapter_ready"]),
                          full_service_ready=False, loading_object_ids=loading,
                          incompatibility_reason="model_loading_pending")
        return result

    def _annotate(self,state):
        state["request_stream"] = dict(contract=CONTRACT,requests=deepcopy(self.requests),
            pending_request_count=len(self.requests)-len(self._completed_requests),horizon_steps=self._max_steps,
            remaining_horizon_steps=max(self._max_steps-self._episode_steps,0),all_requests_released=True)
        state["model_loading"] = [dict(rsu_id=k[0],object_id=k[1],remaining_seconds=max(t-self._seconds(),0.))
                                  for k,t in sorted(self._ready_at.items()) if t>self._seconds()]
        return state

    def step(self,control):
        if self._stream_done:
            raise RuntimeError("Terminal stream requires explicit reset")
        before,t = len(self.requests)-len(self._completed_requests),self._seconds()
        node = self.workflow_state.current_node()
        state,native,terminated,truncated,info = super().step(control)
        dt = self._seconds()-t
        if not math.isfinite(dt) or dt<=0:
            raise ValueError("Non-increasing physical timestamps")
        completed = set(self.workflow_state.completed_node_ids)
        now_done = {r["request_id"] for r in self.requests if set(r["node_ids"])<=completed}
        newly_done = len(now_done-self._completed_requests)
        self._completed_requests = now_done
        pending,area = len(self.requests)-len(now_done),before*dt
        event = info.get("cache_event",{})
        transfer_mb = sum(float(v) for v in event.get("transfer_mb_by_type",{}).values())
        finite_end = terminated or truncated or self._episode_steps>=self._max_steps
        terms = dict(completion=5.*newly_done,pending_time=-area,model_transfer=-.25*transfer_mb/1024.,
                     unfinished=-5.*pending if finite_end else 0.)
        total = sum(terms.values())
        reward = RewardBreakdown(total=total,positive_offset=0.,service_reward=terms["completion"],delay_penalty=area,
            cache_miss_penalty=0.,migration_cost=-terms["model_transfer"],continuity_bonus=0.,
            mechanism_exploration_bonus=0.,constraint_penalty=-terms["unfinished"])
        later_use = []
        if node and node.node_id in completed:
            rsu = event.get("served_rsu_id")
            adapter = self.adapter_catalog.get_typed_adapter(node.required_adapter)
            admission = self._admissions.get((rsu,adapter.object_id))
            if admission and admission["step"]<self._episode_steps:
                later_use = [dict(admission,execution_step=self._episode_steps,rsu_id=rsu)]
        row = dict(step=self._episode_steps,subject_id=self.subject_id,request_count=len(self.requests),
            completed_requests=len(now_done),newly_completed_requests=newly_done,pending_requests=pending,
            pending_area_seconds=area,transfer_mb=transfer_mb,native_reward=native.total,reward=total,
            reward_terms=terms,cache_event=deepcopy(event),later_load_use=later_use,
            terminal_reason="all_completed" if terminated else "finite_horizon_unfinished" if finite_end else None)
        self.stream_rows.append(row)
        self.pending_area_seconds += area
        self.stream_return += total
        info.update(persistent_stream=row,native_reward_breakdown=native.to_dict(),persistent_reward_contract=CONTRACT)
        self._stream_done = finite_end
        self._last_state = self._annotate(state)
        # Finite H is a task boundary: no bootstrap into nonexistent continuation.
        return deepcopy(self._last_state),reward,bool(finite_end),False,info
