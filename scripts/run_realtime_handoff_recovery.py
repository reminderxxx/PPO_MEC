"""Wall-paced NGSIM interruption test, separate from accelerated acceptance.

The source really runs until the observed handoff, is terminated if still busy,
and the target consumes only a committed node-boundary checkpoint or raw input.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from scripts.run_handoff_state_runtime import write_json
from src.runtime.handoff_state_runtime import ARMS, ModelWorker, encode_files, send_packet
from src.runtime.workflow_suffix_recovery import seal_state_payload, sha256_file, write_state_package


def prepare(args):
    from src.data.mobility.ngsim_provider import NGSIMProvider
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    base=json.loads(Path(args.base_study,"plan.json").read_text())
    qualification=json.loads(Path(args.eligibility).read_text())
    frames=NGSIMProvider(base["mobility"]["csv_path"],max_rows=base["mobility"]["max_rows"]).get_loaded_frames()
    cases=[]
    # Use every event meeting the input-only dwell rule. No performance access.
    for event in qualification["qualified"]:
        h=event["handoff_index"];start=h-240
        if start<event["source_start_index"]:raise ValueError("24-second release does not fit source dwell")
        selected=[]
        for frame in frames[start-1:h+1]:
            vehicle=next(v for v in frame["vehicles"] if v.vehicle_id==event["vehicle_id"])
            selected.append({"time_index":frame["time_index"],"vehicle":vehicle.to_dict()})
        if any(b["time_index"]-a["time_index"]!=100 for a,b in zip(selected,selected[1:])):
            raise ValueError("realtime subject trajectory is not contiguous")
        cases.append({**event,"case_id":f"realtime_{len(cases):02d}","frames":selected,
            "release_lead_seconds":24.0})
    if len(cases)!=2:raise ValueError("this fixed development test expects exactly two eligible events")
    base.update(plan_id="realtime_handoff_recovery_20261026_v1",
        classification="wall-paced local technical workflow interruption; not wireless or Alibaba execution",
        causal_prepare_horizon_seconds=6.0,repetitions=1,case_count=len(cases),
        scientific_budget={"cells":6,"generate_attempt_ceiling":22,"automatic_retry":False,"training_calls":0})
    base["generation"]["n1"]["max_new_tokens"]=32
    base["budget"]={"process_timeout_seconds":180,"scientific_total_timeout_seconds":1200}
    base["fairness"]["clock"]="raw 100-ms mobility paced against monotonic wall clock; actual unfinished source node is cancelled"
    write_json(root/"plan.json",base)
    write_json(root/"inputs.json",{"cases":cases,"rsus":base["mobility"]["rsus"],
        "eligibility_sha256":sha256_file(Path(args.eligibility)),
        "selection":"all two events satisfying source>=20s and target>=25s; each also fits fixed 24s release",
        "independence":"overlapping intervals; a single development time cluster; no inferential CI",
        "model_calls":0})
    print(json.dumps({"status":"INPUTS_FROZEN","cases":len(cases),"cells":6}))


def forecast_target(previous,current,rsus,horizon):
    from src.data.mobility.rsu_mapper import RSUMapper
    from src.envs.specs import VehicleState
    dt=(current["time_index"]-previous["time_index"])/1000
    if dt<=0:raise ValueError("invalid causal history")
    point=deepcopy(current["vehicle"])
    for axis in ("position_x","position_y"):
        point[axis]+=horizon*(current["vehicle"][axis]-previous["vehicle"][axis])/dt
    return RSUMapper(rsus).associate([VehicleState(**point)])[point["vehicle_id"]]


def execute(args):
    from src.data.mobility.handoff_builder import HandoffBuilder
    from src.data.mobility.rsu_mapper import RSUMapper
    from src.envs.specs import RSUState, VehicleState
    from scripts.run_two_node_workflow_suffix_recovery import make_state_payload
    root=Path(args.output);plan_path=root/"plan.json"
    plan=json.loads(plan_path.read_text());inputs=json.loads((root/"inputs.json").read_text())
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True)
    if dirty or commit!=args.expected_commit:raise ValueError("clean frozen source required")
    write_json(root/"execution_freeze.json",{"git_commit":commit,"plan_sha256":sha256_file(plan_path),
        "inputs_sha256":sha256_file(root/"inputs.json"),"opened_at_unix":time.time(),"argv":sys.argv,
        "formal_or_holdout":False})
    worker_script=ROOT/"scripts/run_handoff_state_runtime.py"
    reference=ModelWorker(python=plan["python"],script=worker_script,plan_path=plan_path,
        directory=root/"reference",rsu_id="reference",source=True)
    try:
        reference.call("prepare");ref_prefix=reference.call("n0");ref_suffix=reference.call("suffix")
        write_json(root/"reference.json",{"prefix":ref_prefix,"suffix":ref_suffix,"commands":reference.commands})
    finally:reference.close()
    all_rows=[];attempts=2;cohort_started=time.monotonic()
    for case_index,case in enumerate(inputs["cases"]):
        for arm in ARMS[case_index:]+ARMS[:case_index]:
            if time.monotonic()-cohort_started>plan["budget"]["scientific_total_timeout_seconds"]:
                raise TimeoutError("fixed real-time cohort budget exhausted")
            cell=root/"cells"/f"{case['case_id']}_{arm}"
            cell.mkdir(parents=True,exist_ok=False)
            rsus=[RSUState(**r) for r in inputs["rsus"]]
            workers={r.rsu_id:ModelWorker(python=plan["python"],script=worker_script,plan_path=plan_path,
                directory=cell/r.rsu_id,rsu_id=r.rsu_id,source=r.rsu_id==case["source_rsu"]) for r in rsus}
            source=workers[case["source_rsu"]];target=workers[case["target_rsu"]]
            pool=ThreadPoolExecutor(max_workers=5)
            try:
                source_ready=source.call("prepare")
                source_prefix=None;source_suffix=None;source_complete_time=None
                prepared={};preparation_events=[];checkpoint=None
                mapper=RSUMapper(rsus);builder=HandoffBuilder()
                initial=case["frames"][0]
                previous_assoc=mapper.associate([VehicleState(**initial["vehicle"])])
                release_time_index=case["frames"][1]["time_index"]
                started=time.monotonic()
                pending_prefix=pool.submit(source.call,"n0");attempts+=1
                pending_suffix=None
                source_stage="n0";handoff_time=None;event=None
                replay=[]
                for index,frame in enumerate(case["frames"][1:],start=1):
                    scheduled=(frame["time_index"]-release_time_index)/1000
                    time.sleep(max(0.,started+scheduled-time.monotonic()))
                    now=time.monotonic()
                    if source_prefix is None and pending_prefix.done():
                        source_prefix=pending_prefix.result()
                        if source_prefix["n0_output"]["token_ids"]!=ref_prefix["n0_output"]["token_ids"]:
                            raise ValueError("source prefix differs from continuous reference")
                        mark=time.monotonic()
                        envelope=seal_state_payload(make_state_payload(plan,source_prefix["n0_output"]))
                        checkpoint=write_state_package(cell/"source_checkpoint",envelope)
                        checkpoint["export_and_persist_seconds"]=time.monotonic()-mark
                        pending_suffix=pool.submit(source.call,"suffix");attempts+=1
                        source_stage="n1"
                    if pending_suffix is not None and pending_suffix.done() and source_suffix is None:
                        source_suffix=pending_suffix.result();source_complete_time=time.monotonic()
                        source_stage="completed"
                    associations=mapper.associate([VehicleState(**frame["vehicle"])])
                    events=[e.to_dict() for e in builder.build_events(previous_assoc,associations,frame["time_index"])]
                    replay.append({"time_index":frame["time_index"],"elapsed_seconds":now-started,
                        "scheduled_seconds":scheduled,"lag_seconds":now-started-scheduled,
                        "association":associations[case["vehicle_id"]],"events":events})
                    actual=[e for e in events if e["event_type"]=="handoff"]
                    if actual:
                        if len(actual)!=1 or actual[0]["time_index"]!=case["handoff_time"]:
                            raise ValueError("unexpected real-time handoff")
                        event=actual[0];handoff_time=time.monotonic();break
                    if arm=="prepare_recover" and source_stage!="completed":
                        predicted=forecast_target(case["frames"][index-1],frame,rsus,plan["causal_prepare_horizon_seconds"])
                        current=associations[case["vehicle_id"]]
                        if predicted and predicted!=current and predicted not in prepared:
                            prepared[predicted]=pool.submit(workers[predicted].call,"prepare")
                            preparation_events.append({"target":predicted,"issued_elapsed_seconds":time.monotonic()-started,
                                "history_max_time_index":frame["time_index"],"forecast_horizon_seconds":plan["causal_prepare_horizon_seconds"]})
                    previous_assoc=associations
                if event is None:raise ValueError("raw handoff was not reached")
                if source_suffix is not None:
                    output=source_suffix;completed_at=source_complete_time
                    mode="source_completed_before_handoff";sender=receiver=None
                else:
                    # No speculative completed node is manufactured. A running
                    # n0 has no application checkpoint and falls back to restart.
                    source.process.terminate();source.process.wait(timeout=10)
                    inflight=pending_suffix if pending_suffix is not None else pending_prefix
                    try:inflight.result(timeout=10)
                    except RuntimeError:pass
                    if target is source:raise ValueError("handoff did not change RSU")
                    if case["target_rsu"] in prepared:prepared[case["target_rsu"]].result(timeout=180)
                    else:target.call("prepare")
                    use_state=bool(arm!="restart" and checkpoint is not None)
                    if use_state:
                        packet=encode_files("state",{n:(cell/"source_checkpoint"/n).read_bytes() for n in ("state.json","manifest.json")})
                        mode="suffix_recovery"
                    else:
                        packet=encode_files("input",{"input.png":Path(plan["source_input_provenance"]["path"]).read_bytes()})
                        mode="restart" if arm=="restart" else "restart_no_committed_prefix"
                    port=target.call("listen")["port"]
                    receiving=pool.submit(target.call,"receive")
                    sender=send_packet(port,packet,mbps=plan["link"]["mbps"],
                        latency_seconds=plan["link"]["one_way_latency_seconds"])
                    receiver=receiving.result(timeout=180)
                    operation="suffix" if use_state else "restart"
                    output=target.call(operation);attempts+=1 if use_state else 2
                    completed_at=time.monotonic()
                correct=(output["n1_input"]==ref_suffix["n1_input"] and
                    output["n1_output"]["token_ids"]==ref_suffix["n1_output"]["token_ids"])
                row={"cell_id":cell.name,"arm":arm,"case_id":case["case_id"],"status":"PASS" if correct else "FAIL",
                    "workflow_wall_seconds":completed_at-started,"actual_handoff_elapsed_seconds":handoff_time-started,
                    "handoff_to_completion_seconds":max(0.,completed_at-handoff_time),"mode":mode,
                    "source_stage_at_handoff":source_stage,"checkpoint_committed":checkpoint is not None,
                    "completed_before_target_leave":completed_at-handoff_time<case["target_dwell_ms"]/1000,
                    "target_dwell_seconds":case["target_dwell_ms"]/1000,
                    "replay_max_lag_seconds":max(r["lag_seconds"] for r in replay),
                    "prefix":source_prefix,"suffix":output,"checkpoint":checkpoint,"handoff_event":event,
                    "sender":sender,"receiver":receiver,"preparation_events":preparation_events,
                    "source_prepare_outside_request":source_ready,"replay":replay,
                    "source_n1_cancelled":source_stage=="n1" and source_suffix is None}
                write_json(cell/"receipt.json",row)
                all_rows.append({k:v for k,v in row.items() if k not in {"prefix","suffix","checkpoint","handoff_event",
                    "sender","receiver","preparation_events","source_prepare_outside_request","replay"}})
                if not correct:raise ValueError("realtime suffix fidelity failed")
                print(json.dumps(all_rows[-1]),flush=True)
            except Exception as error:
                write_json(cell/"failure.json",{"error":str(error)});raise
            finally:
                pool.shutdown(wait=True)
                for worker in workers.values():worker.close()
                write_json(cell/"commands.json",{key:{"pid":w.pid,"argv":w.command,"commands":w.commands} for key,w in workers.items()})
    if attempts>plan["scientific_budget"]["generate_attempt_ceiling"]:raise ValueError("generation budget exceeded")
    write_json(root/"rows.json",all_rows)
    write_json(root/"completion.json",{"status":"PASS","cells":len(all_rows),"generate_attempts":attempts,
        "training_calls":0,"formal_calls":0,"holdout_calls":0,"git_commit":commit,
        "independence":"one overlapping development time cluster, descriptive only"})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-inputs",action="store_true")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--base-study")
    parser.add_argument("--eligibility")
    parser.add_argument("--output",required=True)
    parser.add_argument("--expected-commit")
    args=parser.parse_args()
    if args.prepare_inputs:prepare(args)
    elif args.execute:execute(args)
    else:parser.error("select preparation or execution")
