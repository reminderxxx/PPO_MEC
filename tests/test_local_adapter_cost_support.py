"""Twelve finite, tiny synthetic families; no models, sockets or original inputs.

Qualification requires an explicit round directory and logs every materialization
and independent-consumer invocation. The prospective parent caps are two rounds,
24 instances and 48 consumer calls; repeating fixtures is not new evidence.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import struct
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/"scripts"/filename)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


producer=module("local_cost_producer_test","run_local_adapter_cost_support.py")
consumer=module("local_cost_consumer_test","audit_local_adapter_cost_support.py")
FAMILIES=("valid_complete","matrix_and_qualifier_order","create_only_and_failstop","hardware_offline_and_identity","old_reference_generation_contract","active_adapter_and_raw_tokens","state_integrity_and_answer_injection","transfer_frame_ack_and_partial_bill","nested_partition_clock_residual","deadline_and_disk_caps","summary_tamper_and_manifest_refresh","missing_duplicate_and_extra_artifacts")


@pytest.fixture(scope="session")
def qualification_round():
    root=os.environ.get("LOCAL_COST_QUALIFICATION_ROOT")
    number=os.environ.get("LOCAL_COST_QUALIFICATION_ROUND")
    if not root or number not in ("round_01","round_02"):
        pytest.skip("bounded qualification requires explicit LOCAL_COST_QUALIFICATION_ROOT and ROUND")
    base=Path(root)
    assert (base/"synthetic_subplan.json").exists()
    previous=list(base.glob("round_*/instances/*/instance.json"))
    assert len(previous)<=24
    directory=base/number
    directory.mkdir(exist_ok=False)
    producer.create_json(directory/"round.json",{"round":number,"started_ns":time.perf_counter_ns(),"families":list(FAMILIES),"actual_models":0,"actual_network":0})
    return directory


def fixture_plan(directory):
    sources=directory/"sources";sources.mkdir()
    resources={}
    for name,data in (("base",b"synthetic-base-not-a-model"),("A",b"A"*37),("B",b"B"*19)):
        path=sources/name;path.mkdir();filename="model.safetensors" if name=="base" else "adapter_model.safetensors"
        (path/filename).write_bytes(data)
        resources[name]={"path":str(path),"weight_file":filename,"weight_bytes":len(data),"weight_sha256":producer.sha(data),"files":[{"name":filename,"bytes":len(data),"sha256":producer.sha(data)}]}
    identity={"base_weight_sha256":resources["base"]["weight_sha256"],"adapter_name":"adapter_a","adapter_weight_sha256":resources["A"]["weight_sha256"],"processor_sha256":"a"*64}
    generation=copy.deepcopy(producer.GENERATION)
    payload={"workflow_id":"technical_vision_description_suffix_v1","nodes":["n0","n1"],"edges":[["n0","n1"]],"completed_nodes":["n0"],"remaining_nodes":["n1"],"next_node":"n1","n0_output":{"raw_text":"synthetic upstream","token_ids":[11]},"suffix_input_material":"synthetic upstream","generation":{"n0":generation,"n1":generation},"identity":identity,"rng_state":{"required":False,"reason":"greedy explicit node"},"tensor_state":{"required":False,"reason":"explicit text not token continuation"}}
    state={"schema_version":"ppo_mec.two_node_workflow_state.v1","payload":payload,"payload_sha256":producer.sha(producer.canonical(payload))}
    state_bytes=json.dumps(state,indent=2,ensure_ascii=False,sort_keys=True).encode()+b"\n"
    (sources/"state.json").write_bytes(state_bytes)
    manifest={"schema_version":"ppo_mec.workflow_state_package.v1","allowed_files":["state.json","manifest.json"],"state":{"path":"state.json","bytes":len(state_bytes),"sha256":producer.sha(state_bytes)},"contains_model_weights":False,"contains_final_output":False}
    manifest_bytes=json.dumps(manifest,indent=2,sort_keys=True).encode()+b"\n";(sources/"manifest.json").write_bytes(manifest_bytes)
    old_plan={"generation":{"n1":generation},"workflow":{"n1_prompt_template":"Upstream: {n0_output}\nSuffix only."}}
    producer.create_json(sources/"old_plan.json",old_plan)
    inp={"prompt":"Upstream: synthetic upstream\nSuffix only.","rendered_prompt":"frozen rendered", "input_ids":[10,11],"source_image_access_count":0}
    reference={"runtime":copy.deepcopy(producer.VERSIONS),"n1_input":inp,"n1_output":{"token_ids":[1,2],"decoded_text":"reference A"}}
    producer.create_json(sources/"reference.json",reference)
    hardware={key:"synthetic-"+key for key in producer.IDENTITY_FIELDS};hardware["AC_power_state"]="AC"
    def spec(name):
        path=sources/name;return {"path":str(path),"sha256":producer.file_sha(path)}
    transfer={"state_package":{"bytes":len(state_bytes)+len(manifest_bytes),"sha256":producer.sha(state_bytes+manifest_bytes)},"adapter_A_weight":{"bytes":37,"sha256":resources["A"]["weight_sha256"]},"adapter_B_weight":{"bytes":19,"sha256":resources["B"]["weight_sha256"]}}
    caps=copy.deepcopy(producer.HARD);caps.update(disk_bytes=4194304,log_bytes=1048576,payload_bytes=4*sum(v["bytes"] for v in transfer.values()),application_bytes=4*sum(v["bytes"] for v in transfer.values())+12*112)
    return {"contract":producer.SYNTHETIC_CONTRACT,"generation":generation,"matrix":producer.matrix(),"runtime":{"device":"cpu","dtype":"float32","batch_size":1,"intra_threads":1,"inter_threads":1,"local_files_only":True,"trust_remote_code":False,"automatic_retry":False,"versions":copy.deepcopy(producer.VERSIONS)},"hardware_expected":hardware,"source_image_access_allowed":False,"prefix_calls":0,"old_reference":spec("reference.json"),"old_plan":spec("old_plan.json"),"old_runtime_identity":copy.deepcopy(producer.VERSIONS),"reference_input":inp,"reference_A_tokens":[1,2],"state_identity":identity,"state_file":spec("state.json"),"state_manifest":spec("manifest.json"),"resources":resources,"transfer_objects":transfer,"caps":caps,"source_commit":"synthetic-source-not-a-scientific-commit"}


class FakeBackend:
    """Only materializes tiny synthetic native receipts; never loads or connects."""
    def __init__(self,fail_first=False,partial=False):
        self.fail_first,self.partial=fail_first,partial
        self.invocations=[]

    def __call__(self,plan,row,directory,remaining):
        self.invocations.append(row["id"])
        ledger=producer.NativeLedger(directory/"events.jsonl")
        try:
            if row["phase"] in ("T1","T2"):
                with ledger.span("processor_load"):pass
                with ledger.span("base_load"):pass
                for active in row["adapters"]:
                    with ledger.span("adapter_load",adapter=active):pass
                if row["phase"]=="T1":
                    with ledger.span("activate",adapter=row["adapters"][0]):pass
                with ledger.span("input_rebuild"):
                    producer.create_json(directory/"input.json",plan["reference_input"])
                sequence=[row["adapters"][0]]*2 if row["phase"]=="T1" else ["A","B","A"]
                for i,active in enumerate(sequence):
                    if row["phase"]=="T2":
                        with ledger.span("activate",adapter=active):pass
                    with ledger.span("generate",adapter=active,repeat=i if row["phase"]=="T1" else 0) as parent:
                        with ledger.span("generate_compute",parent=parent):pass
                        with ledger.span("output_copy_decode",parent=parent):pass
                    item={"adapter":active,"repeat":i,"token_ids":[99] if self.fail_first and row["id"]=="T1_A0" else [1,2] if active=="A" else [3,4],"decoded_text":"reference "+active,"status":{"active_adapters":[active],"available_adapters":row["adapters"]},"loaded_parameters":[{"key":"layer.lora_A."+active+".weight","shape":[1],"dtype":"torch.float32","nonzero":1,"sha256":active.lower()*64}],"adapter_weight_sha256":plan["resources"][active]["weight_sha256"],"generation":plan["generation"]}
                    producer.create_json(directory/("output_%02d.json"%i),item)
                producer.create_json(directory/"child_terminal.json",{"status":"PASS","pid":100+producer.matrix().index(row),"outputs":len(sequence),"versions":producer.VERSIONS,"threads":[1,1],"device":"cpu","dtype":"float32","module_origins":{},"synthetic_only":True})
                (directory/"stdout.log").write_bytes(b"");(directory/"stderr.log").write_bytes(b"")
            elif row["phase"]=="T3":
                with ledger.span("encode"):
                    state=Path(plan["state_file"]["path"]).read_bytes();manifest=Path(plan["state_manifest"]["path"]).read_bytes()
                with ledger.span("durable_save") as parent:
                    package=directory/"package";package.mkdir()
                    for filename,data in (("state.json",state),("manifest.json",manifest)):
                        with ledger.span("file_write",parent=parent,filename=filename,bytes=len(data)):
                            (package/filename).write_bytes(data)
                        for kind in ("file_fsync","file_rename","directory_fsync"):
                            with ledger.span(kind,parent=parent,filename=filename):pass
                with ledger.span("restore"):pass
                with ledger.span("prompt_rebuild"):
                    producer.create_json(directory/"restored_input.json",{"prompt":plan["reference_input"]["prompt"],"identity":plan["state_identity"],"source_image_access_count":0})
            else:
                obj=row["object"];spec=plan["transfer_objects"][obj]
                data=(Path(plan["state_file"]["path"]).read_bytes()+Path(plan["state_manifest"]["path"]).read_bytes()) if obj=="state_package" else (Path(plan["resources"]["A" if obj=="adapter_A_weight" else "B"]["path"])/"adapter_model.safetensors").read_bytes()
                wire=producer.NativeLedger(directory/"wire.jsonl")
                partial=self.partial and row["id"]=="T4_state_package0"
                try:
                    with ledger.span("transfer",object=obj,requested_payload_bytes=len(data)):
                        header=producer.HEADER.pack(producer.MAGIC,len(data),bytes.fromhex(spec["sha256"]),obj.encode().ljust(16,b"\0"))
                        with wire.span("send_header",requested_bytes=64):
                            (directory/"header.bin").write_bytes(header)
                        with wire.span("send_payload",requested_bytes=len(data),chunk_sha256=producer.sha(data)):
                            (directory/("payload.partial" if partial else "payload.bin")).write_bytes(data[:7] if partial else data)
                            if partial:raise EOFError("synthetic disconnected attempt; no real socket")
                        with wire.span("receive_ack",requested_bytes=48):
                            (directory/"ack.bin").write_bytes(producer.ACK.pack(producer.ACK_MAGIC,len(data),bytes.fromhex(spec["sha256"])))
                finally:
                    wire.close()
                    producer.create_json(directory/"transfer_native.json",{"header_upper_bytes":64,"payload_complete_bytes":0 if partial else len(data),"payload_uncertain_upper_bytes":len(data) if partial else 0,"ack_received_bytes":0 if partial else 48,"receiver_bytes":7 if partial else len(data),"receiver_recv_calls":1,"sender_send_calls":1 if partial else 2,"application_charge_upper_bytes":64+len(data)+(0 if partial else 48)})
        finally:
            ledger.close()


def refresh(root):
    (root/"inventory.json").unlink()
    producer.create_json(root/"inventory.json",{"files":producer.inventory(root)})
    closure=producer.read_json(root/"execution_finish.json")
    closure.update(inventory_sha256=producer.file_sha(root/"inventory.json"),terminal_sha256=producer.file_sha(root/"terminal_receipt.json"))
    (root/"execution_finish.json").write_text(json.dumps(closure,sort_keys=True,indent=2)+"\n")


def change_json(path,mutator,fixture):
    value=producer.read_json(path);before=copy.deepcopy(value);mutator(value)
    actions=list(fixture.glob("tamper_*.json"))
    producer.create_json(fixture/("tamper_%02d.json"%len(actions)),{"path":str(path),"before":before,"after":value})
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n")


def checked_audit(root,plan,fixture):
    logs=list(fixture.glob("consumer_call_*.json"))
    assert len(logs)<2
    start=time.perf_counter_ns();target=fixture/("consumer_call_%02d.json"%len(logs))
    try:
        result=consumer.audit(root,producer.sha(producer.canonical(plan)),allow_synthetic=True)
    except Exception as exc:
        producer.create_json(target,{"status":"REJECTED","error":repr(exc),"started_ns":start,"finished_ns":time.perf_counter_ns(),"actual_model_calls":0,"actual_network_calls":0})
        raise
    producer.create_json(target,{"status":"ACCEPTED_SYNTHETIC_ONLY","native_terminal":result["native_terminal"],"started_ns":start,"finished_ns":time.perf_counter_ns(),"actual_model_calls":0,"actual_network_calls":0})
    return result


@pytest.mark.parametrize("family",FAMILIES)
def test_contract_family(qualification_round,family):
    directory=qualification_round/"instances"/family;directory.mkdir(parents=True)
    producer.create_json(directory/"instance.json",{"family":family,"actual_model_loads":0,"actual_generate_calls":0,"actual_socket_calls":0,"copies":0})
    base=qualification_round.parent
    assert len(list(base.glob("round_*/instances/*/instance.json")))<=24
    assert producer.disk_bytes(base)<=134217728
    plan=fixture_plan(directory)
    backend=FakeBackend(fail_first=family=="create_only_and_failstop",partial=family=="transfer_frame_ack_and_partial_bill")
    preflight={"synthetic_only":True,"hardware":plan["hardware_expected"],"versions":producer.VERSIONS,"source_commit":plan["source_commit"]}
    root=directory/"run"
    terminal=producer.run(plan,root,backend,synthetic=True,preflight_record=preflight)
    if family=="valid_complete":
        result=checked_audit(root,plan,directory)
        assert result["native_terminal"]=="PASS" and result["raw_calls"]==consumer.EXPECTED
        assert len(result["samples"])==28 and all(x["n"]==3 for x in result["aggregates"].values())
        with pytest.raises(ValueError,match="execution/synthetic"):
            producer.validate_plan(plan)
    elif family=="matrix_and_qualifier_order":
        change_json(root/"attempts/T1_A0/started.json",lambda v:v.update(row=producer.matrix()[2]),directory);refresh(root)
        with pytest.raises(ValueError,match="attempt plan/identity"):checked_audit(root,plan,directory)
        invalid=copy.deepcopy(plan);invalid["matrix"][0],invalid["matrix"][2]=invalid["matrix"][2],invalid["matrix"][0]
        with pytest.raises(ValueError,match="matrix/order"):producer.validate_plan(invalid,synthetic=True)
    elif family=="create_only_and_failstop":
        assert terminal["status"]=="FAIL" and backend.invocations==["T1_A0"]
        result=checked_audit(root,plan,directory);assert result["native_terminal"]=="FAIL" and result["raw_calls"]["generate"]==2
        with pytest.raises(FileExistsError):producer.run(plan,root,FakeBackend(),synthetic=True,preflight_record=preflight)
    elif family=="hardware_offline_and_identity":
        change_json(root/"preflight.json",lambda v:v["hardware"].update(AC_power_state="BATTERY"),directory);refresh(root)
        with pytest.raises(ValueError,match="hardware gate"):checked_audit(root,plan,directory)
        invalid=copy.deepcopy(plan);invalid["runtime"]["local_files_only"]=False
        with pytest.raises(ValueError,match="runtime"):producer.validate_plan(invalid,synthetic=True)
        rows=[{"pid":1,"command":"/usr/bin/python3 -B /w/scripts/run_stream_reward_curve.py --cell main"},{"pid":2,"command":"/bin/zsh -c 'echo run_stream_reward_curve.py'"},{"pid":3,"command":"/usr/bin/python3 /w/scripts/run_local_adapter_cost_support.py --child T1_A0"},{"pid":4,"command":"/usr/bin/python3 -m scripts.launch_stream_reward_curve"},{"pid":5,"command":"/usr/bin/python3 /w/scripts/run_local_adapter_cost_support.py --plan p.json"}]
        assert [x["pid"] for x in producer.active_scientific_python_jobs(rows,{3})]==[1,4]
        assert producer.active_scientific_python_jobs([{"pid":6,"command":"python -c 'print(1)'"}],set())[0]["python_program"]=="UNVERIFIED_PYTHON_C"
        with pytest.raises(ValueError,match="execution grant"):producer.preflight(plan,{},"not-a-plan")
    elif family=="old_reference_generation_contract":
        change_json(root/"attempts/T1_A0/output_00.json",lambda v:v["generation"].update(max_new_tokens=64),directory);refresh(root)
        with pytest.raises(ValueError,match="generation identity"):checked_audit(root,plan,directory)
        invalid=copy.deepcopy(plan);invalid["reference_A_tokens"]=[99]
        with pytest.raises(ValueError,match="reference mismatch"):producer.validate_plan(invalid,synthetic=True)
    elif family=="active_adapter_and_raw_tokens":
        target=root/"attempts/T1_A0/output_00.json"
        change_json(target,lambda v:v.update(token_ids=[99]),directory);refresh(root)
        with pytest.raises(ValueError,match="token reference"):checked_audit(root,plan,directory)
        def wrong_active(v):v.update(token_ids=[1,2]);v["status"]["active_adapters"]=["B"]
        change_json(target,wrong_active,directory);refresh(root)
        with pytest.raises(ValueError,match="active adapter"):checked_audit(root,plan,directory)
    elif family=="state_integrity_and_answer_injection":
        def inject(v):v["payload"]["n1_output"]="forged answer";v["payload_sha256"]=producer.sha(producer.canonical(v["payload"]))
        change_json(root/"attempts/T3_state0/package/state.json",inject,directory);refresh(root)
        with pytest.raises(ValueError,match="injected final answer"):checked_audit(root,plan,directory)
        invalid=producer.read_json(Path(plan["state_file"]["path"]));invalid["payload"]["next_node"]="n0";invalid["payload_sha256"]=producer.sha(producer.canonical(invalid["payload"]))
        with pytest.raises(ValueError,match="boundary"):producer.validate_state(invalid,plan["state_identity"])
    elif family=="transfer_frame_ack_and_partial_bill":
        result=checked_audit(root,plan,directory)
        assert terminal["status"]=="FAIL" and result["raw_calls"]["transfer"]==1 and result["payload_charge_upper_bytes"]==plan["transfer_objects"]["state_package"]["bytes"]
        change_json(root/"attempts/T4_state_package0/transfer_native.json",lambda v:v.update(payload_uncertain_upper_bytes=0,application_charge_upper_bytes=64),directory);refresh(root)
        with pytest.raises(ValueError,match="writeahead native"):checked_audit(root,plan,directory)
        header=producer.HEADER.pack(producer.MAGIC,9,bytes(32),b"state_package".ljust(16,b"\0"));assert len(header)==64 and producer.ACK.size==48
    elif family=="nested_partition_clock_residual":
        target=root/"attempts/T1_A0/events.jsonl";events=[json.loads(x) for x in target.read_text().splitlines()]
        events[1]["ns"]=events[3]["ns"]+1;producer.create_json(directory/"original_events.json",events);target.write_text("\n".join(json.dumps(x) for x in events)+"\n");refresh(root)
        with pytest.raises(ValueError,match="overlapping partition"):checked_audit(root,plan,directory)
    elif family=="deadline_and_disk_caps":
        change_json(root/"terminal_receipt.json",lambda v:v.update(end_ns=v["begin_ns"]+5401*10**9),directory);refresh(root)
        with pytest.raises(ValueError,match="wall cap"):checked_audit(root,plan,directory)
        guard=producer.Guard(root,{"wall_seconds":5400,"disk_bytes":1,"log_bytes":1})
        with pytest.raises(RuntimeError,match="disk cap"):guard.check()
    elif family=="summary_tamper_and_manifest_refresh":
        change_json(root/"terminal_receipt.json",lambda v:v["summary"]["raw_calls"].update(generate=0),directory);refresh(root)
        with pytest.raises(ValueError,match="summary differs"):checked_audit(root,plan,directory)
    else:
        (root/"extra.bin").write_bytes(b"unplanned");refresh(root)
        with pytest.raises(ValueError,match="unexpected root"):checked_audit(root,plan,directory)
        producer.create_json(directory/"removed_extra.json",{"original_hex":(root/"extra.bin").read_bytes().hex()});(root/"extra.bin").unlink()
        ack=root/"attempts/T4_state_package0/ack.bin";old=ack.read_bytes()
        producer.create_json(directory/"tampered_ack.json",{"before_hex":old.hex(),"after_hex":(old[:-1]+bytes([old[-1]^1])).hex()})
        ack.write_bytes(old[:-1]+bytes([old[-1]^1]));refresh(root)
        with pytest.raises(ValueError,match="native ACK"):checked_audit(root,plan,directory)
        invalid=copy.deepcopy(plan);invalid["hardware_expected"].pop("CPU_model")
        with pytest.raises(ValueError,match="hardware identity"):producer.validate_plan(invalid,synthetic=True)
    assert producer.disk_bytes(base)<=134217728
