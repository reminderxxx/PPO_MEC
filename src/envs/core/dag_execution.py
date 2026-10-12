"""Event-driven DAG execution and handoff state, without simulated inference.

The caller owns time, link scheduling, cache readiness and compute slots. This
runtime accepts their completion events and refuses to expose an output before
its producing computation has completed or its transfer has arrived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.envs.specs import WorkflowGraphState, WorkflowNode


NodeStatus = Literal[
    "waiting_predecessor", "waiting_input", "waiting_model", "queued", "running", "completed"
]
HandoffKind = Literal["transfer_state", "reexecute", "forward_result"]


@dataclass(frozen=True)
class DataObject:
    """Immutable identity for a source input, predecessor output or saved state."""

    object_id: str
    producer_node_id: str | None
    size_bytes: int
    payload_identity: str
    created_at: float


@dataclass(frozen=True)
class TransferRequest:
    request_id: str
    kind: Literal["input", "transfer_state", "forward_result"]
    object_id: str
    source_rsu_id: str
    target_rsu_id: str
    size_bytes: int
    node_id: str
    requested_at: float


@dataclass(frozen=True)
class HandoffRequest:
    request_id: str
    kind: HandoffKind
    node_id: str
    source_rsu_id: str
    target_rsu_id: str
    object_id: str | None
    size_bytes: int
    requested_at: float


@dataclass(frozen=True)
class HandoffReceipt:
    request_id: str
    kind: HandoffKind
    node_id: str
    source_rsu_id: str
    target_rsu_id: str
    object_id: str | None
    size_bytes: int
    completed_at: float
    delivered_to_current_access: bool = False


@dataclass
class NodeExecution:
    node_id: str
    deployment_rsu_id: str | None = None
    status: NodeStatus = "waiting_predecessor"
    model_ready: bool = False
    input_object_ids: tuple[str, ...] = ()
    output_object_id: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    execution_count: int = 0


class DagExecutionRuntime:
    """Strict event consumer for one workflow; no duration or bandwidth model."""

    def __init__(self, graph: WorkflowGraphState, access_rsu_id: str | None = None) -> None:
        self.graph = graph
        self.nodes = self._validated_nodes(graph)
        self.access_rsu_id = access_rsu_id
        self.executions = {node_id: NodeExecution(node_id) for node_id in self.nodes}
        self.objects: dict[str, DataObject] = {}
        self.object_locations: dict[str, set[str]] = {}
        self.pending_transfers: dict[str, TransferRequest] = {}
        self.pending_reexecutions: dict[str, HandoffRequest] = {}
        self.handoff_receipts: list[HandoffReceipt] = []
        self.delivered_object_ids: set[str] = set()
        self._root_inputs: dict[str, str] = {}
        self._request_number = 0

    @staticmethod
    def _validated_nodes(graph: WorkflowGraphState) -> dict[str, WorkflowNode]:
        nodes = graph.node_map()
        if not nodes or len(nodes) != len(graph.nodes):
            raise ValueError("DAG nodes must be nonempty with unique IDs")
        order = list(graph.execution_order)
        if len(order) != len(nodes) or set(order) != set(nodes):
            raise ValueError("execution_order must contain each DAG node exactly once")
        edges = set(graph.edges)
        if any(source not in nodes or target not in nodes or source == target for source, target in edges):
            raise ValueError("DAG contains an invalid edge")
        position = {node_id: index for index, node_id in enumerate(order)}
        if any(position[source] >= position[target] for source, target in edges):
            raise ValueError("execution_order is not topological")
        for node_id, node in nodes.items():
            expected = {source for source, target in edges if target == node_id}
            if set(node.predecessors) != expected:
                raise ValueError(f"predecessor/edge mismatch for {node_id}")
        return nodes

    def state(self, node_id: str) -> NodeExecution:
        return self.executions[node_id]

    @property
    def is_completed(self) -> bool:
        return all(execution.status == "completed" for execution in self.executions.values())

    def deploy_node(self, node_id: str, rsu_id: str) -> NodeStatus:
        execution = self.state(node_id)
        if not rsu_id:
            raise ValueError("deployment RSU is required")
        if execution.status == "running":
            raise ValueError("running node cannot move with vehicle access")
        if execution.status == "completed":
            raise ValueError("completed node requires explicit reexecution")
        if execution.deployment_rsu_id != rsu_id:
            execution.model_ready = False
        execution.deployment_rsu_id = rsu_id
        return self._refresh(node_id)

    def on_access_handoff(self, new_access_rsu_id: str | None) -> None:
        """Access changes do not move any running or deployed task."""
        self.access_rsu_id = new_access_rsu_id

    def register_root_input(
        self, node_id: str, object_id: str, size_bytes: int,
        payload_identity: str, rsu_id: str, created_at: float,
    ) -> DataObject:
        if self.nodes[node_id].predecessors:
            raise ValueError("root input can only be attached to a source node")
        if node_id in self._root_inputs or object_id in self.objects:
            raise ValueError("input object identity already registered")
        data = self._register_object(object_id, None, size_bytes, payload_identity, rsu_id, created_at)
        self._root_inputs[node_id] = object_id
        self._refresh(node_id)
        return data

    def set_model_ready(self, node_id: str, rsu_id: str, ready: bool = True) -> NodeStatus:
        execution = self.state(node_id)
        if execution.deployment_rsu_id != rsu_id:
            raise ValueError("model event must match node deployment")
        if execution.status in {"running", "completed"}:
            raise ValueError("cannot change model readiness during or after execution")
        execution.model_ready = ready
        return self._refresh(node_id)

    def request_input_transfer(
        self, node_id: str, predecessor_node_id: str | None, requested_at: float,
    ) -> TransferRequest:
        execution = self.state(node_id)
        if execution.deployment_rsu_id is None:
            raise ValueError("deploy node before requesting input")
        if execution.status in {"running", "completed"}:
            raise ValueError("node no longer accepts input transfers")
        if predecessor_node_id is None:
            object_id = self._root_inputs[node_id]
        else:
            if predecessor_node_id not in self.nodes[node_id].predecessors:
                raise ValueError("requested producer is not a predecessor")
            object_id = self.state(predecessor_node_id).output_object_id
            if object_id is None:
                raise ValueError("predecessor output is not complete")
        data = self.objects[object_id]
        source = self._source_for(object_id, execution.deployment_rsu_id)
        return self._new_transfer("input", data, source, execution.deployment_rsu_id, node_id, requested_at)

    def on_transfer_complete(self, request_id: str, completed_at: float) -> HandoffReceipt | None:
        request = self.pending_transfers.pop(request_id)
        if completed_at < request.requested_at:
            self.pending_transfers[request_id] = request
            raise ValueError("transfer completion precedes request")
        self.object_locations[request.object_id].add(request.target_rsu_id)
        if request.kind == "input":
            self._refresh(request.node_id)
            return None
        delivered = (
            request.kind == "forward_result"
            and self.state(request.node_id).status == "completed"
            and self.state(request.node_id).output_object_id == request.object_id
            and request.target_rsu_id == self.access_rsu_id
        )
        receipt = HandoffReceipt(
            request_id, request.kind, request.node_id, request.source_rsu_id,
            request.target_rsu_id, request.object_id, request.size_bytes, completed_at,
            delivered_to_current_access=delivered,
        )
        self.handoff_receipts.append(receipt)
        if delivered:
            self.delivered_object_ids.add(request.object_id)
        self._refresh_all()
        return receipt

    def on_compute_start(self, node_id: str, started_at: float) -> tuple[DataObject, ...]:
        execution = self.state(node_id)
        self._refresh(node_id)
        if execution.status != "queued":
            raise ValueError(f"node {node_id} cannot start from {execution.status}")
        ids = self._required_inputs(node_id)
        execution.input_object_ids = ids
        execution.status = "running"
        execution.started_at = started_at
        return tuple(self.objects[object_id] for object_id in ids)

    def on_compute_complete(
        self, node_id: str, object_id: str, payload_identity: str, completed_at: float,
        size_bytes: int | None = None,
    ) -> DataObject:
        execution = self.state(node_id)
        if execution.status != "running" or execution.deployment_rsu_id is None:
            raise ValueError("node is not running")
        if execution.started_at is None or completed_at < execution.started_at:
            raise ValueError("compute completion precedes start")
        data = self._register_object(
            object_id, node_id,
            self.nodes[node_id].output_size if size_bytes is None else size_bytes,
            payload_identity, execution.deployment_rsu_id, completed_at,
        )
        execution.output_object_id = object_id
        execution.completed_at = completed_at
        execution.execution_count += 1
        execution.status = "completed"
        for request_id, request in list(self.pending_reexecutions.items()):
            if request.node_id == node_id:
                self.handoff_receipts.append(HandoffReceipt(
                    request_id, "reexecute", node_id, request.source_rsu_id,
                    request.target_rsu_id, object_id, data.size_bytes, completed_at,
                ))
                del self.pending_reexecutions[request_id]
        self._refresh_all()
        return data

    def request_transfer_state(
        self, completed_node_id: str, target_rsu_id: str, requested_at: float,
    ) -> HandoffRequest:
        """Transfer an actual completed output as boundary state; restore is external."""
        execution = self.state(completed_node_id)
        if execution.status != "completed" or execution.output_object_id is None:
            raise ValueError("boundary state requires a completed output")
        data = self.objects[execution.output_object_id]
        source = self._source_for(data.object_id, target_rsu_id)
        request = self._new_transfer(
            "transfer_state", data, source, target_rsu_id, completed_node_id, requested_at,
        )
        return HandoffRequest(
            request.request_id, "transfer_state", completed_node_id, source,
            target_rsu_id, data.object_id, data.size_bytes, requested_at,
        )

    def request_forward_result(
        self, node_id: str, target_rsu_id: str, requested_at: float,
    ) -> HandoffRequest:
        if any(source == node_id for source, _ in self.graph.edges):
            raise ValueError("only a sink output is a final result")
        execution = self.state(node_id)
        if execution.status != "completed" or execution.output_object_id is None:
            raise ValueError("result does not exist")
        data = self.objects[execution.output_object_id]
        source = self._source_for(data.object_id, target_rsu_id)
        request = self._new_transfer(
            "forward_result", data, source, target_rsu_id, node_id, requested_at,
        )
        return HandoffRequest(
            request.request_id, "forward_result", node_id, source,
            target_rsu_id, data.object_id, data.size_bytes, requested_at,
        )

    def request_reexecute(self, node_id: str, target_rsu_id: str, requested_at: float) -> HandoffRequest:
        execution = self.state(node_id)
        if execution.status != "completed" or execution.output_object_id is None:
            raise ValueError("reexecution requires a completed node")
        if any(self.state(successor).status in {"running", "completed"}
               for successor in self._successors(node_id)):
            raise ValueError("cannot reexecute an ancestor of started or completed work")
        old_output = execution.output_object_id
        source = execution.deployment_rsu_id
        assert source is not None
        request_id = self._next_request_id()
        request = HandoffRequest(
            request_id, "reexecute", node_id, source, target_rsu_id,
            old_output, 0, requested_at,
        )
        self.pending_reexecutions[request_id] = request
        self.delivered_object_ids.discard(old_output)
        execution.deployment_rsu_id = target_rsu_id
        execution.model_ready = False
        execution.output_object_id = None
        execution.input_object_ids = ()
        execution.started_at = None
        execution.completed_at = None
        execution.status = "waiting_input"
        self._refresh_all()
        return request

    def _successors(self, node_id: str) -> set[str]:
        seen: set[str] = set()
        frontier = [node_id]
        while frontier:
            parent = frontier.pop()
            for source, target in self.graph.edges:
                if source == parent and target not in seen:
                    seen.add(target)
                    frontier.append(target)
        return seen

    def _register_object(
        self, object_id: str, producer: str | None, size_bytes: int,
        payload_identity: str, rsu_id: str, created_at: float,
    ) -> DataObject:
        if not object_id or object_id in self.objects:
            raise ValueError("object ID must be new and nonempty")
        if size_bytes < 0 or not payload_identity or not rsu_id:
            raise ValueError("object requires nonnegative bytes, payload identity and RSU")
        data = DataObject(object_id, producer, size_bytes, payload_identity, created_at)
        self.objects[object_id] = data
        self.object_locations[object_id] = {rsu_id}
        return data

    def _required_inputs(self, node_id: str) -> tuple[str, ...]:
        predecessors = self.nodes[node_id].predecessors
        if not predecessors:
            root = self._root_inputs.get(node_id)
            return (root,) if root else ()
        return tuple(self.state(predecessor).output_object_id or "" for predecessor in predecessors)

    def _refresh(self, node_id: str) -> NodeStatus:
        execution = self.state(node_id)
        if execution.status in {"running", "completed"}:
            return execution.status
        if any(self.state(predecessor).status != "completed"
               for predecessor in self.nodes[node_id].predecessors):
            execution.status = "waiting_predecessor"
            return execution.status
        required = self._required_inputs(node_id)
        if not required or execution.deployment_rsu_id is None or any(
            not object_id or execution.deployment_rsu_id not in self.object_locations[object_id]
            for object_id in required
        ):
            execution.status = "waiting_input"
        elif not execution.model_ready:
            execution.status = "waiting_model"
        else:
            execution.status = "queued"
        return execution.status

    def _refresh_all(self) -> None:
        for node_id in self.graph.execution_order:
            self._refresh(node_id)

    def _source_for(self, object_id: str, target: str) -> str:
        locations = self.object_locations[object_id]
        return target if target in locations else min(locations)

    def _next_request_id(self) -> str:
        self._request_number += 1
        return f"{self.graph.workflow_id}:transfer:{self._request_number}"

    def _new_transfer(
        self, kind: Literal["input", "transfer_state", "forward_result"],
        data: DataObject, source: str, target: str, node_id: str, requested_at: float,
    ) -> TransferRequest:
        if not target:
            raise ValueError("transfer target is required")
        request = TransferRequest(
            self._next_request_id(), kind, data.object_id, source, target,
            0 if source == target else data.size_bytes, node_id, requested_at,
        )
        self.pending_transfers[request.request_id] = request
        return request
