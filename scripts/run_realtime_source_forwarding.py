"""Diagnostic alternate deployment contract: keep source, forward final result.

Mobility handoff does not imply source-RSU failure. This separately frozen
development counterfactual tests that assumption, outside the legacy five-action
contract. It is not an algorithm baseline ranking or independent confirmation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from scripts.run_handoff_state_runtime import write_json
from src.runtime.handoff_state_runtime import ModelWorker, receive_packet, send_packet
from src.runtime.workflow_suffix_recovery import canonical_json_bytes, sha256_file


def receiver(directory):
    listener=socket.socket();listener.bind(("127.0.0.1",0));listener.listen(1)
    # Source service may outlast the generic packet receiver's 120-second wait.
    write_json(directory/"port.partial.json",{"port":listener.getsockname()[1],"pid":__import__('os').getpid()})
    (directory/"port.partial.json").replace(directory/"port.json")
    packet,transport=receive_packet(listener)
    result=json.loads(packet)
    if result.get("kind")!="final_result":raise ValueError("unexpected forwarding payload")
    write_json(directory/"received.json",{"result":result,"transport":transport})


def execute(args):
    from src.data.mobility.rsu_mapper import RSUMapper
    from src.data.mobility.handoff_builder import HandoffBuilder
    from src.envs.specs import RSUState,VehicleState
    base=Path(args.base_study);root=Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    plan=json.loads((base/"plan.json").read_text());inputs=json.loads((base/"inputs.json").read_text())
    reference=json.loads((base/"reference.json").read_text())
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True)
    if commit!=args.expected_commit or dirty:raise ValueError("clean frozen source required")
    write_json(root/"execution_freeze.json",{"git_commit":commit,"base_plan_sha256":sha256_file(base/"plan.json"),
        "base_inputs_sha256":sha256_file(base/"inputs.json"),"opened_at_unix":time.time(),"argv":sys.argv,
        "scope":"postulated keep-source/forward deployment contract, development diagnostic only",
        "generation_attempt_budget":4,"cases":2,"training_calls":0,
        "assumptions":["source RSU remains available after vehicle handoff","backhaul result forwarding is permitted",
                       "one workflow, no competing source capacity demand"]})
    rows=[]
    for case in inputs["cases"]:
        cell=root/case["case_id"];cell.mkdir()
        source=ModelWorker(python=plan["python"],script=ROOT/"scripts/run_handoff_state_runtime.py",
            plan_path=base/"plan.json",directory=cell/"source",rsu_id=case["source_rsu"],source=True)
        target_dir=cell/"result_target";target_dir.mkdir()
        # Start the lightweight receiver after observing handoff, to avoid
        # including an idle TCP timeout in the source-execution contract.
        receiver_process=None;receiver_log=None
        pool=ThreadPoolExecutor(max_workers=1)
        try:
            warm=source.call("prepare")
            mapper=RSUMapper([RSUState(**r) for r in inputs["rsus"]]);builder=HandoffBuilder()
            previous=mapper.associate([VehicleState(**case["frames"][0]["vehicle"])])
            started=time.monotonic();release_index=case["frames"][1]["time_index"]
            prefix_future=pool.submit(source.call,"n0");suffix_future=None;prefix=None;output=None
            completed_at=None;handoff_at=None;handoff_event=None;replay=[]
            for frame in case["frames"][1:]:
                scheduled=(frame["time_index"]-release_index)/1000
                time.sleep(max(0.,started+scheduled-time.monotonic()))
                if prefix is None and prefix_future.done():
                    prefix=prefix_future.result();suffix_future=pool.submit(source.call,"suffix")
                if suffix_future is not None and suffix_future.done() and output is None:
                    output=suffix_future.result();completed_at=time.monotonic()
                association=mapper.associate([VehicleState(**frame["vehicle"])])
                events=[e.to_dict() for e in builder.build_events(previous,association,frame["time_index"])]
                replay.append({"time_index":frame["time_index"],"elapsed_seconds":time.monotonic()-started,
                    "scheduled_seconds":scheduled,"events":events})
                actual=[e for e in events if e["event_type"]=="handoff"]
                if actual:
                    if len(actual)!=1 or actual[0]["time_index"]!=case["handoff_time"]:
                        raise ValueError("forwarding native handoff mismatch")
                    handoff_event=actual[0];handoff_at=time.monotonic();break
                previous=association
            if handoff_at is None:raise ValueError("no forwarding handoff")
            if output is None:
                if prefix is None:prefix=prefix_future.result();suffix_future=pool.submit(source.call,"suffix")
                output=suffix_future.result()
                receiver_log=(target_dir/"stderr.log").open("w")
                receiver_process=subprocess.Popen([sys.executable,str(Path(__file__)),"--receiver",str(target_dir)],
                    stdout=receiver_log,stderr=receiver_log)
                deadline=time.monotonic()+10
                while not (target_dir/"port.json").exists():
                    if time.monotonic()>deadline:raise TimeoutError("forward receiver startup")
                    time.sleep(.01)
                port=json.loads((target_dir/"port.json").read_text())["port"]
                packet=canonical_json_bytes({"kind":"final_result","workflow_id":plan["workflow"]["workflow_id"],
                    "vehicle_id":case["vehicle_id"],"n1_input_sha256":output["n1_input"]["content_sha256"],
                    "token_ids":output["n1_output"]["token_ids"],"decoded_text":output["n1_output"]["decoded_text"]})
                sent=send_packet(port,packet,mbps=plan["link"]["mbps"],latency_seconds=plan["link"]["one_way_latency_seconds"])
                receiver_process.wait(timeout=10)
                if receiver_process.returncode:raise RuntimeError("forward receiver failed")
                received=json.loads((target_dir/"received.json").read_text())
                if received["result"]!=json.loads(packet):
                    raise ValueError("forward result mismatch")
                completed_at=time.monotonic();mode="source_continued_result_forwarded"
            else:
                sent=received=None;mode="source_completed_before_handoff"
            correct=(prefix["n0_output"]["token_ids"]==reference["prefix"]["n0_output"]["token_ids"] and
                output["n1_input"]==reference["suffix"]["n1_input"] and
                output["n1_output"]["token_ids"]==reference["suffix"]["n1_output"]["token_ids"])
            row={"case_id":case["case_id"],"arm":"continue_forward","status":"PASS" if correct else "FAIL",
                "workflow_wall_seconds":completed_at-started,"handoff_to_completion_seconds":max(0.,completed_at-handoff_at),
                "actual_handoff_elapsed_seconds":handoff_at-started,"mode":mode,
                "completed_before_target_leave":completed_at-handoff_at<case["target_dwell_ms"]/1000,
                "target_dwell_seconds":case["target_dwell_ms"]/1000,
                "forward_bytes":sent["forward_bytes"] if sent else 0,
                "ack_bytes":sent["ack_bytes"] if sent else 0,
                "target_inference_calls":0,"source_n0_calls":1,"source_n1_calls":1,
                "source_preserved_after_handoff":True}
            write_json(cell/"receipt.json",{"summary":row,"prefix":prefix,"suffix":output,"sender":sent,
                "receiver":received,"handoff_event":handoff_event,"replay":replay,"source_prepare_outside_request":warm})
            rows.append(row)
            if not correct:raise ValueError("forward source output fidelity mismatch")
            print(json.dumps(row),flush=True)
        except Exception as error:
            write_json(cell/"failure.json",{"error":str(error)});raise
        finally:
            pool.shutdown(wait=True);source.close()
            if receiver_process and receiver_process.poll() is None:receiver_process.terminate();receiver_process.wait(timeout=10)
            if receiver_log:receiver_log.close()
            write_json(cell/"commands.json",{"source_argv":source.command,"source_pid":source.pid,
                "source_commands":source.commands})
    write_json(root/"rows.json",rows)
    write_json(root/"completion.json",{"status":"PASS","cells":len(rows),"generate_calls":4,
        "git_commit":commit,"training_calls":0,"formal_calls":0,"holdout_calls":0,
        "claim_boundary":"alternative deployment feasibility only, outside legacy five-action comparison"})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receiver",type=Path)
    parser.add_argument("--base-study",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--expected-commit")
    args=parser.parse_args()
    if args.receiver:receiver(args.receiver)
    else:execute(args)
