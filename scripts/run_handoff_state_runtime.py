"""Native handoff -> model state TCP transfer -> suffix execution acceptance.

Use --prepare-inputs first, commit/freeze the producer, then --execute. Neither
mode trains an agent or opens formal/holdout inputs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.handoff_state_runtime import (  # noqa: E402
    ARMS, HandoffStateRuntime, ModelWorker, decode_files, encode_files,
    receive_packet, send_packet,
)
from src.runtime.workflow_suffix_recovery import (  # noqa: E402
    EXPECTED_WORKFLOW_ID, read_state_package, seal_state_payload,
    sha256_file, write_state_package,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def worker_main(args: argparse.Namespace) -> None:
    # Model/library messages cannot corrupt the JSON control channel.
    protocol_output = sys.stdout
    sys.stdout = sys.stderr
    from scripts.run_two_node_workflow_suffix_recovery import (
        generate, load_runtime, make_state_payload, prepare_n0, prepare_n1,
        state_identity, verify_static_resources,
    )
    plan = json.loads(Path(args.plan).read_text())
    directory = Path(args.directory)
    original_image = str(Path(plan["source_input_provenance"]["path"]).resolve())
    if not args.source:
        def deny_source_image(event: str, arguments: tuple[Any, ...]) -> None:
            if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
                if str(Path(os.fsdecode(arguments[0])).resolve()) == original_image:
                    raise PermissionError("recovery target may not open original source image")
        sys.addaudithook(deny_source_image)
    processor = model = status = n0 = listener = None
    generations = 0
    checked = False
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        started = time.monotonic()
        try:
            command = json.loads(line)
            operation = command["operation"]
            response: dict[str, Any] = {"status": "PASS", "pid": os.getpid(),
                                       "rsu_id": args.rsu_id, "operation": operation}
            if operation == "prepare":
                if model is not None:
                    raise ValueError("duplicate model preparation")
                import torch
                torch.set_num_threads(int(plan["runtime"]["torch_threads"]))
                if not checked:
                    verify_static_resources(plan, include_source_image=args.source)
                    checked = True
                processor, model, timings, status = load_runtime(plan)
                response.update(model_load_timings=timings, adapter_status=status,
                                identity=state_identity(plan))
            elif operation == "n0":
                if not args.source or model is None or n0 is not None:
                    raise ValueError("source prefix is unavailable or already executed")
                tensors, record = prepare_n0(plan, processor)
                n0 = generate(model, processor, tensors, plan["generation"]["n0"])
                generations += 1
                response.update(n0_input=record, n0_output=n0)
            elif operation == "listen":
                if listener is not None:
                    raise ValueError("duplicate transport listener")
                listener = socket.socket()
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
                response["port"] = listener.getsockname()[1]
            elif operation == "send":
                if not args.source or n0 is None:
                    raise ValueError("only a committed source can transfer")
                if command["kind"] == "state":
                    mark = time.monotonic()
                    envelope = seal_state_payload(make_state_payload(plan, n0))
                    package = write_state_package(directory / "state_package", envelope)
                    response["state_export_seconds"] = time.monotonic() - mark
                    response["state_package"] = package
                    packet = encode_files("state", {name: (directory / "state_package" / name).read_bytes()
                        for name in ("state.json", "manifest.json")})
                elif command["kind"] == "input":
                    packet = encode_files("input", {"input.png": Path(original_image).read_bytes()})
                else:
                    raise ValueError("unknown transfer kind")
                response["transport"] = send_packet(command["port"], packet,
                    mbps=command["mbps"], latency_seconds=command["latency_seconds"])
            elif operation == "receive":
                if args.source or listener is None:
                    raise ValueError("target listener is required")
                packet, transport = receive_packet(listener)
                listener = None
                mark = time.monotonic()
                kind, files = decode_files(packet)
                incoming = directory / "incoming"
                incoming.mkdir(exist_ok=False)
                for name, data in files.items():
                    (incoming / name).write_bytes(data)
                if kind == "state":
                    payload, receipt = read_state_package(incoming, expected_identity=state_identity(plan))
                    if payload["generation"] != plan["generation"]:
                        raise ValueError("restored generation configuration mismatch")
                    n0 = {"decoded_text": payload["n0_output"]["raw_text"],
                          "token_ids": payload["n0_output"]["token_ids"]}
                    response["state_import"] = receipt
                else:
                    if sha256_file(incoming / "input.png") != plan["source_input_provenance"]["sha256"]:
                        raise ValueError("transferred source input hash mismatch")
                response.update(kind=kind, transport=transport,
                                restore_and_persist_seconds=time.monotonic() - mark)
            elif operation in {"suffix", "restart"}:
                if model is None:
                    raise ValueError("model not prepared")
                if operation == "restart":
                    if args.source or n0 is not None:
                        raise ValueError("restart target must use transferred raw input")
                    local_plan = deepcopy(plan)
                    local_plan["source_input_provenance"]["path"] = str(directory / "incoming" / "input.png")
                    tensors, record = prepare_n0(local_plan, processor)
                    n0 = generate(model, processor, tensors, plan["generation"]["n0"])
                    generations += 1
                    response.update(n0_input=record, n0_output=n0)
                if n0 is None:
                    raise ValueError("prefix not available")
                tensors, record = prepare_n1(plan, processor, n0["decoded_text"])
                output = generate(model, processor, tensors, plan["generation"]["n1"])
                generations += 1
                response.update(n1_input=record, n1_output=output,
                    prefix_tokens=n0["token_ids"], source_image_direct_access_count=0,
                    node_call_counts={"n0": int(operation == "restart"), "n1": 1})
            else:
                raise ValueError("unknown worker operation")
            response.update(wall_seconds=time.monotonic() - started,
                            cumulative_generate_calls=generations)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            response = {"status": "FAIL", "error": str(error),
                        "error_type": type(error).__name__, "pid": os.getpid()}
        protocol_output.write(json.dumps(response) + "\n")
        protocol_output.flush()


def causal_target(history: list[dict[str, Any]], rsus: list[Any]) -> str | None:
    """Constant-position-velocity extrapolation using only two past observations."""
    from src.data.mobility.rsu_mapper import RSUMapper
    from src.envs.specs import VehicleState
    previous, current = history[-2:]
    dt = current["time_index"] - previous["time_index"]
    if dt <= 0:
        raise ValueError("non-increasing predictor history")
    # Predict two observed sampling intervals ahead; no future position is read.
    vehicle = deepcopy(current["vehicle"])
    for axis in ("position_x", "position_y"):
        vehicle[axis] += 2 * (current["vehicle"][axis] - previous["vehicle"][axis])
    return RSUMapper(rsus).associate([VehicleState(**vehicle)])[vehicle["vehicle_id"]]


def prepare_inputs(args: argparse.Namespace) -> None:
    from src.data.mobility.ngsim_provider import NGSIMProvider
    from src.data.mobility.rsu_mapper import RSUMapper
    from src.envs.specs import RSUState
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.plan).read_text())
    csv_path = Path(config["mobility"]["csv_path"])
    provider = NGSIMProvider(csv_path, max_rows=config["mobility"]["max_rows"])
    frames = provider.get_loaded_frames()
    rsus = [RSUState(**item) for item in config["mobility"]["rsus"]]
    mapper = RSUMapper(rsus)
    associations = [mapper.associate(frame["vehicles"]) for frame in frames]
    cases = []
    previous_end = -10**20
    eligible = 0
    for index in range(3, len(frames)):
        if any(frames[j]["time_index"] - frames[j-1]["time_index"] != 100 for j in range(index-2,index+1)):
            continue
        for vehicle_id in sorted(associations[index]):
            ids = [associations[j].get(vehicle_id) for j in range(index-3, index+1)]
            if not (ids[0] and ids[0] == ids[1] == ids[2] and ids[3] and ids[3] != ids[2]):
                continue
            eligible += 1
            if len(cases) >= config["case_count"] or frames[index-3]["time_index"] <= previous_end:
                continue
            selected = []
            for j in range(index-3,index+1):
                vehicle = next(v for v in frames[j]["vehicles"] if v.vehicle_id == vehicle_id)
                value = vehicle.to_dict()
                value.pop("associated_rsu_id",None)
                selected.append({"time_index": frames[j]["time_index"], "vehicle": value})
            prediction = causal_target(selected[:2], rsus)
            cases.append({"case_id": f"case_{len(cases):02d}", "history": selected[:2],
                "frames": [{"time_index": f["time_index"], "vehicles": [f["vehicle"]]} for f in selected[1:]],
                "source_rsu": ids[2], "target_rsu": ids[3], "predicted_target": prediction,
                "raw_time_start": selected[0]["time_index"], "raw_time_end": selected[-1]["time_index"],
                "selection": "first chronological eligible boundary; no model/reward access"})
            previous_end = selected[-1]["time_index"] + config["minimum_gap_ms"]
    if len(cases) != config["case_count"]:
        raise RuntimeError(f"insufficient input-only cases: {len(cases)}")
    write_json(root / "inputs.json", {"cases": cases, "eligible_events": eligible,
        "rsus": config["mobility"]["rsus"], "source_csv": str(csv_path),
        "source_prefix_rows": config["mobility"]["max_rows"],
        "source_csv_bytes": csv_path.stat().st_size,
        "independence": "disjoint selected intervals; not a fresh global holdout",
        "model_calls": 0, "training_calls": 0})
    write_json(root / "plan.json", config)
    print(json.dumps({"status":"INPUTS_FROZEN", "cases":len(cases), "eligible":eligible}))


def workflow() -> Any:
    from src.envs.specs import WorkflowGraphState, WorkflowNode
    return WorkflowGraphState(EXPECTED_WORKFLOW_ID, [
        WorkflowNode("n0","image prefix","veh_base_v1","adapter_a",1,1,[],["n1"]),
        WorkflowNode("n1","text suffix","veh_base_v1","adapter_a",1,1,["n0"],[])],
        [("n0","n1")], ["n0","n1"], current_node_id="n0")


def build_env(case: dict[str,Any], rsus: list[Any], backend: Any = None) -> Any:
    from src.data.mobility.replay_provider import ReplayProvider
    from src.data.model_catalog.adapter_catalog import AdapterCatalog, RSUAdapterCacheProfile
    from src.envs.core.predictor_manager import PredictorManager
    from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
    catalog = AdapterCatalog.from_json(ROOT / "src/data/model_catalog/sample_model_catalog.json")
    catalog.rsu_adapter_caches = [RSUAdapterCacheProfile(r.rsu_id,["adapter_a"]) for r in rsus]
    # Logical residency permits dispatch; actual memory residency/load cost is
    # owned and recorded by the measured backend, never by legacy reward.
    return VecWorkflowCoreEnv(mobility_provider=ReplayProvider(case["frames"]),
        workflow_state=workflow(),adapter_catalog=catalog,rsu_states=rsus,max_steps=2,
        predictor_manager=PredictorManager(disable_prediction_output=True),
        execution_backend=backend)


def controls(case: dict[str,Any], arm: str) -> list[Any]:
    from src.envs.specs import ControlAction
    first = ControlAction(offload_action={"mode":"rsu"},migration_action={"mode":"keep"})
    if arm == "prepare_recover" and case["predicted_target"] != case["source_rsu"]:
        first.migration_action={"mode":"prepare","expected_target_rsu_id":case["predicted_target"]}
    return [first,ControlAction(offload_action={"mode":"rsu"},migration_action={"mode":"migrate"})]


def execute(args: argparse.Namespace) -> None:
    from src.envs.specs import RSUState
    root = Path(args.output)
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text())
    inputs = json.loads((root / "inputs.json").read_text())
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    dirty = subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True)
    if dirty or not args.expected_commit or commit != args.expected_commit:
        raise ValueError("scientific execution requires clean, explicitly frozen commit")
    if (root / "execution_freeze.json").exists():
        raise FileExistsError("scientific cohort already opened; automatic reruns are prohibited")
    digest=hashlib.sha256()
    source_prefix_bytes=0
    with Path(inputs["source_csv"]).open("rb") as handle:
        for _ in range(inputs["source_prefix_rows"]+1):
            line=handle.readline()
            if not line:raise ValueError("NGSIM source prefix unexpectedly truncated")
            digest.update(line)
            source_prefix_bytes+=len(line)
    write_json(root / "execution_freeze.json", {"git_commit":commit,
        "inputs_sha256":sha256_file(root / "inputs.json"),"plan_sha256":sha256_file(plan_path),
        "raw_source_prefix":{"data_rows":inputs["source_prefix_rows"],
            "bytes":source_prefix_bytes,"sha256":digest.hexdigest()},
        "argv":sys.argv,"opened_at_unix":time.time(),"formal_or_holdout":False})
    rsus = [RSUState(**item) for item in inputs["rsus"]]
    # One new same-source continuous reference, independent of the three arms.
    reference_worker = ModelWorker(python=plan["python"], script=Path(__file__), plan_path=plan_path,
        directory=root/"continuous_reference",rsu_id="reference",source=True)
    try:
        reference_worker.call("prepare")
        prefix = reference_worker.call("n0")
        suffix = reference_worker.call("suffix")
        reference={"prefix":prefix,"suffix":suffix,"commands":reference_worker.commands}
        write_json(root / "continuous_reference.json",reference)
    finally:
        reference_worker.close()
    rows=[]
    cohort_started=time.monotonic()
    for repetition in range(plan["repetitions"]):
        for case in inputs["cases"]:
            # Balanced rotation reduces systematic order/thermal confounding.
            rotation=(repetition+int(case["case_id"][-2:])) % 3
            for arm in ARMS[rotation:]+ARMS[:rotation]:
                if time.monotonic()-cohort_started>plan["budget"]["scientific_total_timeout_seconds"]:
                    raise TimeoutError("fixed scientific wall budget exhausted")
                cell_id=f"{case['case_id']}_r{repetition}_{arm}"
                cell=root/"cells"/cell_id
                cell.mkdir(parents=True,exist_ok=False)
                workers={}
                backend=None
                try:
                    # Cold target control processes exist before request release;
                    # disk weights are prestaged equally, memory is not prepared.
                    for rsu in rsus:
                        workers[rsu.rsu_id]=ModelWorker(python=plan["python"],script=Path(__file__),
                            plan_path=plan_path,directory=cell/rsu.rsu_id,rsu_id=rsu.rsu_id,
                            source=rsu.rsu_id==case["source_rsu"])
                    source_ready=workers[case["source_rsu"]].call("prepare")
                    backend=HandoffStateRuntime(arm=arm,workers=workers,source_rsu=case["source_rsu"],
                        mbps=plan["link"]["mbps"],latency_seconds=plan["link"]["one_way_latency_seconds"])
                    env=build_env(case,deepcopy(rsus),backend)
                    env.reset()
                    receipts=[]
                    for control in controls(case,arm):
                        _,_,terminated,truncated,info=env.step(control)
                        receipts.append(info)
                    snapshot=info["measured_execution"]
                    nodes=[e for e in snapshot["events"] if e["kind"]=="node_completed"]
                    handoffs=[e for e in snapshot["events"] if e["kind"]=="handoff_completed"]
                    correct=bool(terminated and not truncated and len(handoffs)==1 and
                        nodes[0]["result"]["n0_output"]["token_ids"]==prefix["n0_output"]["token_ids"] and
                        nodes[1]["result"]["prefix_tokens"]==prefix["n0_output"]["token_ids"] and
                        nodes[1]["result"]["n1_input"]==suffix["n1_input"] and
                        nodes[1]["result"]["n1_output"]["token_ids"]==suffix["n1_output"]["token_ids"])
                    row={"status":"PASS" if correct else "FAIL", "cell_id":cell_id,
                        "case_id":case["case_id"],"repetition":repetition,"arm":arm,
                        "workflow_wall_seconds":snapshot["workflow_wall_seconds"],
                        "handoff_to_completion_seconds":snapshot["handoff_to_completion_seconds"],
                        "forward_bytes":handoffs[0]["sender"]["transport"]["forward_bytes"],
                        "ack_bytes":handoffs[0]["sender"]["transport"]["ack_bytes"],
                        "completed_nodes":snapshot["completed_nodes"],
                        "source_rsu":case["source_rsu"],"target_rsu":case["target_rsu"],
                        "prefix_tokens_equal":nodes[0]["result"]["n0_output"]["token_ids"]==prefix["n0_output"]["token_ids"],
                        "suffix_input_and_tokens_equal":correct,
                        "source_preparation_outside_request":source_ready,
                        "measured_execution":snapshot,"native_infos":receipts}
                    write_json(cell/"receipt.json",row)
                    rows.append({k:v for k,v in row.items() if k not in
                        {"source_preparation_outside_request","measured_execution","native_infos"}})
                except Exception as error:
                    write_json(cell/"failure.json",{"error":str(error),"traceback":traceback.format_exc()})
                    rows.append({"status":"FAIL","cell_id":cell_id,"error":str(error)})
                    raise
                finally:
                    if backend:
                        backend.close()
                    else:
                        for worker in workers.values():worker.close()
                    write_json(cell/"commands.json",{key:{"argv":w.command,"pid":w.pid,"commands":w.commands}
                        for key,w in workers.items()})
                    write_json(cell/"terminal.json",{"finished_at_unix":time.time(),"status":rows[-1]["status"] if rows else "FAIL"})
                print(json.dumps({"cell_id":cell_id,"status":rows[-1]["status"],
                    "seconds":rows[-1].get("workflow_wall_seconds")}),flush=True)
    write_json(root/"rows.json",rows)
    write_json(root/"completion.json",{"status":"PASS" if all(r["status"]=="PASS" for r in rows) else "FAIL",
        "cells":len(rows),"git_commit":commit,"training_calls":0,"formal_calls":0,"holdout_calls":0,
        "scope":"technical node-boundary replay, real inference and paced localhost TCP, not wireless or application accuracy"})


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--prepare-inputs",action="store_true")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--plan")
    parser.add_argument("--output")
    parser.add_argument("--expected-commit")
    parser.add_argument("--directory")
    parser.add_argument("--rsu-id")
    parser.add_argument("--source",action="store_true")
    args=parser.parse_args()
    if args.worker:worker_main(args)
    elif args.prepare_inputs:prepare_inputs(args)
    elif args.execute:execute(args)
    else:parser.error("choose --prepare-inputs or --execute")


if __name__=="__main__":
    main()
