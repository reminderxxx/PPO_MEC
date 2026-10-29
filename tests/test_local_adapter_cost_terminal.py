"""Exactly two new synthetic terminal fixtures; no old family reruns or models."""
from __future__ import annotations

import ast
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import sys
import time

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("local_cost_prior_software_helpers",ROOT/"tests/test_local_adapter_cost_support.py")
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
producer=helpers.producer
consumer=helpers.consumer


def real_t1_event_order():
    """Static witness binds the real model_child, rather than trusting its fake."""
    tree=ast.parse(inspect.getsource(producer.model_child))
    rows=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.With):continue
        call=node.items[0].context_expr
        if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute) or call.func.attr!="span" or not call.args or not isinstance(call.args[0],ast.Constant):continue
        kind=call.args[0].value
        if any(k.arg=="parent" for k in call.keywords):continue
        if kind not in {"processor_load","base_load","adapter_load","activate","input_rebuild","generate"}:continue
        if kind=="adapter_load" and any(k.arg=="adapter" and isinstance(k.value,ast.Constant) and k.value.value=="B" for k in call.keywords):continue
        rows.append((node.lineno,kind))
    result=[]
    for _,kind in sorted(rows):result.extend([kind]*2 if kind=="generate" else [kind])
    repeat=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=="repeat")
    assert isinstance(repeat.iter,ast.Call) and repeat.iter.func.id=="range" and isinstance(repeat.iter.args[0],ast.IfExp)
    assert ast.literal_eval(repeat.iter.args[0].body)==2 and ast.literal_eval(repeat.iter.args[0].orelse)==1
    audit_tree=ast.parse(inspect.getsource(consumer.audit))
    assignment=next(n for n in ast.walk(audit_tree) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=="expected_ops" for x in n.targets))
    expected=ast.literal_eval(assignment.value.value)["T1"]
    assert result==expected,"real model_child T1 AST order differs from independent consumer"
    return result


class ASTAlignedBackend(helpers.FakeBackend):
    def __init__(self,fail_after_transfer=False):
        super().__init__();self.fail_after_transfer=fail_after_transfer

    def __call__(self,plan,row,directory,remaining):
        if row["phase"]!="T1":
            super().__call__(plan,row,directory,remaining)
            if self.fail_after_transfer and row["id"]=="T4_state_package0":
                raise TimeoutError("synthetic gate failure AFTER full payload.bin was committed")
            return
        self.invocations.append(row["id"])
        active=row["adapters"][0];ledger=producer.NativeLedger(directory/"events.jsonl");index=0
        try:
            for kind in real_t1_event_order():
                with ledger.span(kind,**({"adapter":active} if kind in ("adapter_load","activate","generate") else {})) as parent:
                    if kind=="input_rebuild":producer.create_json(directory/"input.json",plan["reference_input"])
                    if kind=="generate":
                        with ledger.span("generate_compute",parent=parent):pass
                        with ledger.span("output_copy_decode",parent=parent):pass
                if kind=="generate":
                    producer.create_json(directory/("output_%02d.json"%index),{"adapter":active,"repeat":index,"token_ids":[1,2] if active=="A" else [3,4],"decoded_text":"reference "+active,"status":{"active_adapters":[active],"available_adapters":[active]},"loaded_parameters":[{"key":"layer.lora_A."+active+".weight","shape":[1],"dtype":"torch.float32","nonzero":1,"sha256":active.lower()*64}],"adapter_weight_sha256":plan["resources"][active]["weight_sha256"],"generation":plan["generation"]});index+=1
            producer.create_json(directory/"child_terminal.json",{"status":"PASS","pid":100+producer.matrix().index(row),"outputs":2,"versions":producer.VERSIONS,"threads":[1,1],"device":"cpu","dtype":"float32","module_origins":{},"synthetic_only":True})
            (directory/"stdout.log").write_bytes(b"");(directory/"stderr.log").write_bytes(b"")
        finally:ledger.close()


@pytest.fixture(scope="session")
def terminal_round():
    value=os.environ.get("LOCAL_COST_TERMINAL_QUALIFICATION_ROOT")
    if not value:pytest.skip("explicit terminal qualification root required")
    base=Path(value)
    assert (base/"qualification_source_freeze.json").exists()
    directory=base/"qualification_once";directory.mkdir(exist_ok=False)
    producer.create_json(directory/"round.json",{"started_ns":time.perf_counter_ns(),"scenarios":["producer_pass_delivery","producer_fail_prefix_delivery"],"old_families_rerun":False,"actual_models_network":0})
    producer.qualify_shared_clock("/Users/howen/Projects/PPO_MEC/artifacts/environments/adapter_state_acceptance_py39_v1/bin/python",base/"clock_qualification.json")
    return directory


@pytest.mark.parametrize("failed",[False,True],ids=["producer_pass_delivery","producer_fail_prefix_delivery"])
def test_terminal_pipeline(terminal_round,failed,monkeypatch):
    directory=terminal_round/("producer_fail_prefix_delivery" if failed else "producer_pass_delivery");directory.mkdir()
    producer.create_json(directory/"fixture_claim.json",{"actual_model_loads":0,"actual_generates":0,"actual_sockets":0,"original_resource_reads":0})
    plan=helpers.fixture_plan(directory)
    clock_path=terminal_round.parent/"clock_qualification.json"
    plan["clock_contract"]=producer.CLOCK_CONTRACT
    plan["clock_qualification"]={"path":str(clock_path),"sha256":producer.file_sha(clock_path)}
    plan["utility_processes_cap"]=16
    output=directory/"raw";audit=directory/"independent_audit.json";delivery=directory/"delivery.json"
    plan["terminal_pipeline"]={"producer_seconds":4800,"consumer_seconds":600,"combined_seconds":5400}
    plan["outputs"]={"raw_root":str(output.resolve()),"audit":str(audit.resolve()),"delivery":str(delivery.resolve())}
    source_plan=directory/"prospective_plan.json";producer.create_json(source_plan,plan)
    backend=ASTAlignedBackend(fail_after_transfer=failed)
    preflight={"synthetic_only":True,"hardware":plan["hardware_expected"],"versions":producer.VERSIONS,"source_commit":plan["source_commit"],"clock_contract":producer.CLOCK_CONTRACT,"clock_qualification_sha256":plan["clock_qualification"]["sha256"]}
    result=producer.terminal_pipeline(plan,output,audit,delivery,backend,synthetic=True,preflight_record=preflight,plan_bytes=source_plan.read_bytes())
    assert result["status"]==("FAIL" if failed else "PASS")
    assert result["audit_invocations"]==1 and result["audit_exit_code"]==0 and result["audit_status"]=="SOFTWARE_SYNTHETIC_ONLY"
    assert result["raw_inventory_sha256_before"]==result["raw_inventory_sha256_after"]
    checked=producer.read_json(audit)
    assert checked["native_terminal"]==("FAIL" if failed else "PASS") and checked["synthetic_only"] and checked["raw_calls"]["generate"]==28
    assert result["combined_retained_bytes"]==sum(p.stat().st_size for p in output.rglob("*") if p.is_file())+sum(p.stat().st_size for p in (audit,delivery,directory/"delivery.execution_claim.json",directory/"independent_audit.stdout.log",directory/"independent_audit.stderr.log"))
    assert result["combined_seconds"]<5400 and result["consumer_seconds"]<600
    assert checked["activation_scope"].startswith("activation_with_integrity")
    with pytest.raises(ValueError,match="phase mismatch"):producer.dispatch_worker(plan,producer.matrix()[12],directory,"model")
    with pytest.raises(ValueError,match="phase mismatch"):producer.dispatch_worker(plan,producer.matrix()[0],directory,"helper")
    with pytest.raises(ValueError,match="execution/synthetic"):producer.validate_plan(plan)
    if failed:
        transfer_dir=output/"attempts/T4_state_package0"
        start=producer.read_json(transfer_dir/"attempt_terminal.json")
        row=next(x for x in producer.matrix() if x["id"]=="T4_state_package0")
        original_exists=Path.exists
        with monkeypatch.context() as patch:
            patch.setattr(Path,"exists",lambda p:False if p==transfer_dir/"transfer_native.json" else original_exists(p))
            reconstructed=consumer.audit_transfer(transfer_dir,row,plan,True,start["begin_ns"],start["end_ns"])
            assert reconstructed==(plan["transfer_objects"]["state_package"]["bytes"],plan["transfer_objects"]["state_package"]["bytes"]+112)
        with monkeypatch.context() as patch:
            patch.setattr(Path,"exists",lambda p:True if p==transfer_dir/"payload.partial" else original_exists(p))
            with pytest.raises(ValueError,match="ambiguous dual"):consumer.audit_transfer(transfer_dir,row,plan,True,start["begin_ns"],start["end_ns"])
        terminal=producer.read_json(output/"terminal_receipt.json")
        for stage,count in (("preflight",0),("before_attempt",16),("finalization",28)):
            view=dict(terminal,failure_stage=stage,current_attempt=None,completed=[x["id"] for x in producer.matrix()[:count]])
            assert consumer.validate_terminal_prefix(view,[(x,"PASS") for x in view["completed"]])==view["completed"]
    else:
        # An owned helper timeout must kill/reap its session, not the producer.
        records=[];kills=[];base=producer.clock_ns();ticks=iter((base,base+9*10**9,base+9*10**9))
        class MockChild:
            pid=43210;returncode=None
            def poll(self):return None
            def wait(self,timeout):self.returncode=-9;return -9
        actual_open=Path.open
        with monkeypatch.context() as patch:
            patch.setattr(Path,"open",lambda p,*a,**k:io.BytesIO() if p.name in ("stdout.log","stderr.log") else actual_open(p,*a,**k))
            patch.setattr(producer,"create_json",lambda p,v:records.append((p.name,v)))
            patch.setattr(producer,"clock_ns",lambda:next(ticks))
            patch.setattr(producer.subprocess,"Popen",lambda *a,**k:MockChild())
            patch.setattr(producer.os,"killpg",lambda pid,sig:kills.append((pid,sig)))
            with pytest.raises(TimeoutError):producer.RealBackend(source_plan)(plan,producer.matrix()[12],output/"attempts/T3_state0",10)
        assert kills and kills[0][0]==43210 and records[-1][1]["reaped"] and records[-1][1]["kill_requested"]
        assert records[0][1]["limit_seconds"]==10 and records[-1][1]["end_ns"]-records[-1][1]["begin_ns"]<=10*10**9
    before=list(backend.invocations)
    with pytest.raises(FileExistsError):producer.terminal_pipeline(plan,output,audit,delivery,backend,synthetic=True,preflight_record=preflight)
    assert backend.invocations==before
    # Check dispatch on this same frozen metadata: child must never call pipeline.
    with monkeypatch.context() as patch:
        calls=[]
        patch.setattr(producer,"validate_plan",lambda *a,**k:None)
        patch.setattr(producer,"model_child",lambda *a,**k:calls.append("child_only"))
        patch.setattr(producer,"terminal_pipeline",lambda *a,**k:pytest.fail("child called terminal pipeline"))
        patch.setenv("PPO_LOCAL_COST_GRANTED_PLAN_SHA",producer.file_sha(source_plan))
        patch.setattr(sys,"argv",["run_local_adapter_cost_support.py","--plan",str(source_plan),"--output",str(output/"attempts/T1_A0"),"--child","T1_A0"])
        producer.main();assert calls==["child_only"]
    producer.create_json(directory/"scenario_result.json",{"status":"PASS","producer_native":checked["native_terminal"],"independent_audits":1,"delivery":result["status"],"AST_order":real_t1_event_order(),"actual_models_generates_sockets_transfers":0,"clock_contract":producer.CLOCK_CONTRACT,"same_data_views_not_new_fixtures":True})
