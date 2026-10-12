"""DAG dependence, object arrival and handoff behavior at event boundaries."""

import pytest

from src.data.workflow.toy_workflow_generator import ToyWorkflowGenerator
from src.envs.core.dag_execution import DagExecutionRuntime


def _runtime() -> DagExecutionRuntime:
    graph = ToyWorkflowGenerator().generate(workflow_id="fork_join", node_count=5)
    runtime = DagExecutionRuntime(graph, access_rsu_id="a")
    runtime.register_root_input("n1", "source", 16, "source-sha", "a", 0.0)
    for node_id, rsu_id in (
        ("n1", "a"), ("n2", "a"), ("n3", "b"), ("n4", "b"), ("n5", "b"),
    ):
        runtime.deploy_node(node_id, rsu_id)
        runtime.set_model_ready(node_id, rsu_id)
    return runtime


def _complete(runtime: DagExecutionRuntime, node_id: str, at: float) -> None:
    runtime.on_compute_start(node_id, at)
    runtime.on_compute_complete(node_id, f"{node_id}-out", f"{node_id}-sha", at + 1)


def test_fork_join_waits_for_real_output_arrival_and_model() -> None:
    runtime = _runtime()
    assert runtime.state("n1").status == "queued"
    assert runtime.state("n4").status == "waiting_predecessor"
    _complete(runtime, "n1", 1.0)
    assert runtime.state("n2").status == "queued"
    assert runtime.state("n3").status == "waiting_input"
    with pytest.raises(ValueError, match="cannot start"):
        runtime.on_compute_start("n3", 2.0)
    transfer = runtime.request_input_transfer("n3", "n1", 2.0)
    assert transfer.size_bytes == runtime.objects["n1-out"].size_bytes
    runtime.on_transfer_complete(transfer.request_id, 3.0)
    assert runtime.state("n3").status == "queued"
    _complete(runtime, "n2", 3.0)
    assert runtime.state("n4").status == "waiting_predecessor"
    _complete(runtime, "n3", 4.0)
    assert runtime.state("n4").status == "waiting_input"
    for predecessor in ("n2", "n3"):
        transfer = runtime.request_input_transfer("n4", predecessor, 5.0)
        runtime.on_transfer_complete(transfer.request_id, 6.0)
    assert runtime.state("n4").status == "queued"
    inputs = runtime.on_compute_start("n4", 7.0)
    assert {item.object_id for item in inputs} == {"n2-out", "n3-out"}
    runtime.on_compute_complete("n4", "n4-out", "n4-sha", 8.0)
    assert runtime.state("n5").status == "queued"
    _complete(runtime, "n5", 9.0)
    assert runtime.is_completed


def test_handoff_leaves_running_node_in_place_and_result_requires_delivery() -> None:
    runtime = _runtime()
    runtime.on_compute_start("n1", 1.0)
    runtime.on_access_handoff("b")
    assert runtime.access_rsu_id == "b"
    assert runtime.state("n1").deployment_rsu_id == "a"
    with pytest.raises(ValueError, match="running node cannot move"):
        runtime.deploy_node("n1", "b")
    runtime.on_compute_complete("n1", "n1-out", "n1-sha", 2.0)
    assert runtime.object_locations["n1-out"] == {"a"}
    with pytest.raises(ValueError, match="only a sink"):
        runtime.request_forward_result("n1", "b", 2.0)
    transfer = runtime.request_transfer_state("n1", "b", 2.0)
    assert transfer.kind == "transfer_state" and transfer.size_bytes > 0
    assert runtime.state("n3").status == "waiting_input"
    receipt = runtime.on_transfer_complete(transfer.request_id, 3.0)
    assert receipt is not None and receipt.kind == "transfer_state"
    assert runtime.state("n3").status == "queued"
    assert "n1-out" in runtime.object_locations and "b" in runtime.object_locations["n1-out"]


def test_reexecution_is_explicit_and_stale_result_is_not_delivered() -> None:
    runtime = _runtime()
    _complete(runtime, "n1", 1.0)
    _complete(runtime, "n2", 2.0)
    request = runtime.request_reexecute("n2", "b", 4.0)
    assert request.kind == "reexecute" and runtime.state("n2").status == "waiting_input"
    assert runtime.state("n4").status == "waiting_predecessor"
    transfer = runtime.request_input_transfer("n2", "n1", 4.0)
    runtime.on_transfer_complete(transfer.request_id, 5.0)
    assert runtime.state("n2").status == "waiting_model"
    runtime.set_model_ready("n2", "b")
    runtime.on_compute_start("n2", 6.0)
    runtime.on_compute_complete("n2", "n2-out-v2", "n2-sha-v2", 7.0)
    receipt = runtime.handoff_receipts[-1]
    assert receipt.kind == "reexecute" and receipt.object_id == "n2-out-v2"
    assert runtime.state("n2").execution_count == 2
    assert runtime.state("n4").status == "waiting_predecessor"


def test_invalid_event_order_and_missing_predecessor_output_fail_closed() -> None:
    runtime = _runtime()
    with pytest.raises(ValueError, match="predecessor output is not complete"):
        runtime.request_input_transfer("n3", "n1", 1.0)
    with pytest.raises(ValueError, match="cannot start"):
        runtime.on_compute_start("n4", 1.0)
    _complete(runtime, "n1", 1.0)
    transfer = runtime.request_input_transfer("n3", "n1", 2.0)
    with pytest.raises(ValueError, match="precedes request"):
        runtime.on_transfer_complete(transfer.request_id, 1.0)
    assert transfer.request_id in runtime.pending_transfers
    runtime.on_transfer_complete(transfer.request_id, 3.0)
    with pytest.raises(KeyError):
        runtime.on_transfer_complete(transfer.request_id, 4.0)


def test_final_result_forward_receipt_follows_transfer_completion() -> None:
    graph = ToyWorkflowGenerator().generate(workflow_id="deliver", node_count=5)
    runtime = DagExecutionRuntime(graph, access_rsu_id="a")
    runtime.register_root_input("n1", "source", 1, "sha", "a", 0.0)
    for node_id in graph.execution_order:
        runtime.deploy_node(node_id, "a")
        runtime.set_model_ready(node_id, "a")
    for index, node_id in enumerate(graph.execution_order):
        _complete(runtime, node_id, float(index + 1))
    runtime.on_access_handoff("b")
    request = runtime.request_forward_result("n5", "b", 8.0)
    assert request.object_id not in runtime.delivered_object_ids
    receipt = runtime.on_transfer_complete(request.request_id, 9.0)
    assert receipt is not None and receipt.kind == "forward_result"
    assert receipt.delivered_to_current_access
    assert request.object_id in runtime.delivered_object_ids


def test_result_sent_to_old_access_is_recorded_but_not_delivered() -> None:
    graph = ToyWorkflowGenerator().generate(workflow_id="late_handoff", node_count=5)
    runtime = DagExecutionRuntime(graph, access_rsu_id="a")
    runtime.register_root_input("n1", "source", 1, "sha", "a", 0.0)
    for node_id in graph.execution_order:
        runtime.deploy_node(node_id, "a")
        runtime.set_model_ready(node_id, "a")
        _complete(runtime, node_id, 1.0)
    request = runtime.request_forward_result("n5", "a", 3.0)
    runtime.on_access_handoff("b")
    receipt = runtime.on_transfer_complete(request.request_id, 4.0)
    assert receipt is not None and not receipt.delivered_to_current_access
    assert receipt.object_id not in runtime.delivered_object_ids
