from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import socket
import struct

import pytest

from scripts.run_handoff_state_runtime import build_env, causal_target, controls, workflow
from src.envs.specs import ControlAction, RSUState
from src.runtime.handoff_state_runtime import (
    HandoffStateRuntime, decode_files, encode_files, receive_packet, send_packet,
)


def case():
    def point(x, time):
        return {"time_index":time,"vehicle":{"vehicle_id":"v","position_x":x,
            "position_y":0.,"speed":10.,"base_model_id":"veh_base_v1","active_workflow_id":None}}
    history=[point(30.,0),point(40.,100)]
    return {"case_id":"test","source_rsu":"rsu_a","target_rsu":"rsu_b",
        "predicted_target":"rsu_b","history":history,
        "frames":[{"time_index":p["time_index"],"vehicles":[p["vehicle"]]}
            for p in [history[-1],point(49.,200),point(60.,300)]]}


def rsus():
    return [RSUState("rsu_a",0,0,60),RSUState("rsu_b",100,0,60)]


class RecordingBackend:
    def __init__(self, fail=None):
        self.calls=[]
        self.fail=fail

    def reset(self, **kwargs):self.calls.append(("reset",kwargs))
    def prepare(self, **kwargs):self.calls.append(("prepare",{}))
    def on_handoff(self, event):
        self.calls.append(("handoff",event))
        if self.fail=="handoff":raise RuntimeError("transport failed")
    def execute_node(self, **kwargs):
        self.calls.append(("execute",kwargs))
        if self.fail==kwargs["node_id"]:raise RuntimeError("model failed")
    def snapshot(self):return {"calls":deepcopy(self.calls)}


def test_native_handoff_executes_backend_before_suffix_and_preserves_default_reward():
    c=case()
    observer=RecordingBackend()
    measured=build_env(c,rsus(),observer)
    default=build_env(c,rsus())
    measured.reset();default.reset()
    for action in controls(c,"prepare_recover"):
        actual=measured.step(action)
        expected=default.step(action)
        assert actual[:4]==expected[:4]
        assert actual[4]["measured_execution"]
        without_measurement={k:v for k,v in actual[4].items() if k!="measured_execution"}
        assert without_measurement==expected[4]
    actions=[kind for kind,_ in observer.calls]
    assert actions==["reset","prepare","execute","prepare","handoff","execute"]
    assert observer.calls[2][1]=={"node_id":"n0","rsu_id":"rsu_a"}
    assert observer.calls[-1][1]=={"node_id":"n1","rsu_id":"rsu_b"}


@pytest.mark.parametrize("failure",["n0","handoff","n1"])
def test_backend_failure_does_not_commit_unexecuted_node(failure):
    c=case();backend=RecordingBackend(failure)
    env=build_env(c,rsus(),backend);env.reset()
    if failure=="n0":
        with pytest.raises(RuntimeError):env.step(controls(c,"recover")[0])
        assert env.workflow_state.completed_node_ids==[]
    else:
        env.step(controls(c,"recover")[0])
        with pytest.raises(RuntimeError):env.step(controls(c,"recover")[1])
        assert env.workflow_state.completed_node_ids==["n0"]


def test_causal_predictor_only_reads_supplied_past_and_has_correct_target():
    c=case()
    assert causal_target(c["history"],rsus())=="rsu_b"
    c["frames"][-1]["vehicles"][0]["position_x"]=-10000
    assert causal_target(c["history"],rsus())=="rsu_b"
    with pytest.raises(ValueError):causal_target([c["history"][0]]*2,rsus())


@pytest.mark.parametrize("kind,files",[
    ("state",{"state.json":b'{"a":1}',"manifest.json":b"manifest"}),
    ("input",{"input.png":b"not decoded in this transport test"}),
])
def test_real_tcp_transfer_conserves_bytes_hash_and_applies_final_chunk_pacing(kind,files):
    packet=encode_files(kind,files)
    listener=socket.socket();listener.bind(("127.0.0.1",0));listener.listen(1)
    with ThreadPoolExecutor() as pool:
        pending=pool.submit(receive_packet,listener)
        sent=send_packet(listener.getsockname()[1],packet,mbps=.1,latency_seconds=.01)
        received,receipt=pending.result(timeout=5)
    assert decode_files(received)==(kind,files)
    assert sent["packet_sha256"]==receipt["packet_sha256"]
    assert sent["forward_bytes"]==len(packet)+8
    assert sent["ack_bytes"]==64
    assert sent["sender_wall_seconds"]>=.01+(len(packet)+8)*8/1e5


def test_receiver_rejects_truncated_packet():
    listener=socket.socket();listener.bind(("127.0.0.1",0));listener.listen(1)
    with ThreadPoolExecutor() as pool:
        pending=pool.submit(receive_packet,listener)
        with socket.create_connection(listener.getsockname()):
            pass
        with pytest.raises(ConnectionError,match="truncated"):pending.result(timeout=5)


@pytest.mark.parametrize("packet",[
    b'{"kind":"state","files":{"../state.json":"AA==","manifest.json":"AA=="}}',
    b'{"kind":"input","files":{"input.png":"%%%"}}',
    b'{"kind":"other","files":{}}',
])
def test_packet_allowlist_and_encoding_fail_closed(packet):
    with pytest.raises(ValueError):decode_files(packet)


class FakeWorker:
    def __init__(self,pid):self.pid=pid;self.operations=[]
    def call(self,op,**kwargs):
        self.operations.append(op)
        result={"status":"PASS"}
        if op=="listen":result["port"]=1
        if op in {"send","receive"}:result["transport"]={"packet_sha256":"identical"}
        return result
    def close(self):pass


@pytest.mark.parametrize("arm,operation",[("restart","restart"),("recover","suffix"),("prepare_recover","suffix")])
def test_runtime_transfers_exactly_once_and_uses_matched_target(arm,operation):
    workers={"rsu_a":FakeWorker(10),"rsu_b":FakeWorker(20)}
    runtime=HandoffStateRuntime(arm=arm,workers=workers,source_rsu="rsu_a",mbps=10,latency_seconds=0)
    try:
        runtime.reset(workflow=workflow().to_dict(),vehicle_id="v",rsu_id="rsu_a")
        runtime.prepare(control=controls(case(),arm)[0],state={})
        runtime.execute_node(node_id="n0",rsu_id="rsu_a")
        event={"vehicle_id":"v","event_type":"handoff","previous_rsu_id":"rsu_a","current_rsu_id":"rsu_b"}
        runtime.on_handoff({**event,"vehicle_id":"other"})
        assert "send" not in workers["rsu_a"].operations
        runtime.on_handoff(event)
        with pytest.raises(ValueError):runtime.on_handoff(event)
        runtime.execute_node(node_id="n1",rsu_id="rsu_b")
        assert workers["rsu_a"].operations==["n0","send"]
        assert workers["rsu_b"].operations==["prepare","listen","receive",operation]
        assert runtime.snapshot()["completed_nodes"]==["n0","n1"]
    finally:runtime.close()


def test_runtime_rejects_wrong_workflow_owner_and_execution_order():
    runtime=HandoffStateRuntime(arm="recover",workers={"rsu_a":FakeWorker(1)},source_rsu="rsu_a",mbps=10,latency_seconds=0)
    try:
        changed=workflow().to_dict();changed["workflow_id"]="wrong"
        with pytest.raises(ValueError):runtime.reset(workflow=changed,vehicle_id="v",rsu_id="rsu_a")
        runtime.reset(workflow=workflow().to_dict(),vehicle_id="v",rsu_id="rsu_a")
        with pytest.raises(ValueError):runtime.execute_node(node_id="n1",rsu_id="rsu_a")
        with pytest.raises(ValueError):runtime.execute_node(node_id="n0",rsu_id="rsu_b")
        with pytest.raises(ValueError):runtime.reset(workflow=workflow().to_dict(),vehicle_id="v",rsu_id="rsu_a")
    finally:runtime.close()


@pytest.mark.parametrize("field,value",[("completed_node_ids",["n0"]),
    ("current_node_id","n1"),("execution_order",["n1","n0"])])
def test_runtime_rejects_partially_consumed_or_reordered_workflow(field,value):
    runtime=HandoffStateRuntime(arm="recover",workers={},source_rsu="rsu_a",mbps=10,latency_seconds=0)
    try:
        changed=workflow().to_dict();changed[field]=value
        with pytest.raises(ValueError):runtime.reset(workflow=changed,vehicle_id="v",rsu_id="rsu_a")
    finally:runtime.close()


def test_runtime_rejects_model_mapping_mismatch():
    runtime=HandoffStateRuntime(arm="recover",workers={},source_rsu="rsu_a",mbps=10,latency_seconds=0)
    try:
        changed=workflow().to_dict();changed["nodes"][0]["required_adapter"]="another"
        with pytest.raises(ValueError):runtime.reset(workflow=changed,vehicle_id="v",rsu_id="rsu_a")
    finally:runtime.close()
