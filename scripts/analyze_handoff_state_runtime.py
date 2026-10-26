"""Read-only reduction of the fixed native handoff acceptance study."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, median
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.runtime.handoff_state_runtime import ARMS
from src.runtime.workflow_suffix_recovery import sha256_file


def analyze(study: Path) -> dict:
    completion=json.loads((study/"completion.json").read_text())
    if completion["status"]!="PASS":raise ValueError("study is not complete")
    inputs=json.loads((study/"inputs.json").read_text())
    plan=json.loads((study/"plan.json").read_text())
    freeze=json.loads((study/"execution_freeze.json").read_text())
    for name in ("inputs","plan"):
        if sha256_file(study/f"{name}.json")!=freeze[f"{name}_sha256"]:
            raise ValueError("frozen input/plan drift")
    reference=json.loads((study/"continuous_reference.json").read_text())
    expected_prefix=reference["prefix"]["n0_output"]["token_ids"]
    expected_suffix=reference["suffix"]["n1_output"]["token_ids"]
    expected_input=reference["suffix"]["n1_input"]
    rows=[]
    generate_calls=reference["suffix"]["cumulative_generate_calls"]
    cases={c["case_id"]:c for c in inputs["cases"]}
    used_pids=set()
    for case_id,case in cases.items():
        for repetition in range(plan["repetitions"]):
            for arm in ARMS:
                cell=study/"cells"/f"{case_id}_r{repetition}_{arm}"
                row=json.loads((cell/"receipt.json").read_text())
                commands=json.loads((cell/"commands.json").read_text())
                terminal=json.loads((cell/"terminal.json").read_text())
                if row["status"]!="PASS" or terminal["status"]!="PASS":
                    raise ValueError("failed cell cannot be discarded")
                execution=row["measured_execution"]
                handoff=[e for e in execution["events"] if e["kind"]=="handoff_completed"]
                nodes=[e for e in execution["events"] if e["kind"]=="node_completed"]
                if len(handoff)!=1 or [e["node_id"] for e in nodes]!=["n0","n1"]:
                    raise ValueError("native event/node count mismatch")
                event=handoff[0]["event"]
                if (event["vehicle_id"]!=case["history"][0]["vehicle"]["vehicle_id"] or
                    event["time_index"]!=case["raw_time_end"] or
                    event["previous_rsu_id"]!=case["source_rsu"] or
                    event["current_rsu_id"]!=case["target_rsu"]):
                    raise ValueError("native event/input mismatch")
                if (nodes[0]["result"]["n0_output"]["token_ids"]!=expected_prefix or
                    nodes[1]["result"]["prefix_tokens"]!=expected_prefix or
                    nodes[1]["result"]["n1_output"]["token_ids"]!=expected_suffix or
                    nodes[1]["result"]["n1_input"]!=expected_input):
                    raise ValueError("continuous/recovery token/input fidelity failed")
                if nodes[0]["worker_pid"]==nodes[1]["worker_pid"]:
                    raise ValueError("source and target must be distinct processes")
                sender=handoff[0]["sender"]
                receiver=handoff[0]["receiver"]
                for key in ("packet_sha256","packet_bytes","forward_bytes","ack_bytes"):
                    if sender["transport"][key]!=receiver["transport"][key]:
                        raise ValueError("transport conservation failed")
                target=commands[case["target_rsu"]]
                source=commands[case["source_rsu"]]
                if target["pid"]==source["pid"]:raise ValueError("process identity collision")
                for worker in commands.values():
                    used_pids.add(worker["pid"])
                    calls=worker["commands"]
                    if calls:
                        generate_calls+=max(c["response"]["cumulative_generate_calls"] for c in calls)
                target_op="restart" if arm=="restart" else "suffix"
                target_prepare=next(c for c in target["commands"] if c["request"]["operation"]=="prepare")
                target_finish=next(c for c in target["commands"] if c["request"]["operation"]==target_op)
                if target_finish["response"]["node_call_counts"]!={"n0":int(arm=="restart"),"n1":1}:
                    raise ValueError("unexpected target prefix reexecution")
                if receiver["kind"]!=("input" if arm=="restart" else "state"):
                    raise ValueError("wrong dynamic input kind")
                if arm!="restart":
                    from scripts.run_two_node_workflow_suffix_recovery import state_identity
                    from src.runtime.workflow_suffix_recovery import read_state_package
                    source_package=cell/case["source_rsu"]/"state_package"
                    target_package=cell/case["target_rsu"]/"incoming"
                    read_state_package(target_package,expected_identity=state_identity(plan))
                    for name in ("state.json","manifest.json"):
                        if (source_package/name).read_bytes()!=(target_package/name).read_bytes():
                            raise ValueError("persisted state package differs after network")
                rows.append({"case_id":case_id,"repetition":repetition,"arm":arm,
                    "workflow_wall_seconds":row["workflow_wall_seconds"],
                    "handoff_to_completion_seconds":row["handoff_to_completion_seconds"],
                    "prefix_wall_seconds":nodes[0]["result"]["wall_seconds"],
                    "target_prepare_work_seconds":target_prepare["response"]["wall_seconds"],
                    "target_prepare_wait_seconds":handoff[0]["preparation_wait_seconds"],
                    "handoff_to_state_ready_seconds":handoff[0]["handoff_to_state_ready_seconds"],
                    "target_execute_seconds":target_finish["response"]["wall_seconds"],
                    "transfer_seconds":sender["transport"]["sender_wall_seconds"],
                    "state_export_seconds":sender.get("state_export_seconds",0.),
                    "restore_and_persist_seconds":receiver["restore_and_persist_seconds"],
                    "target_repeated_prefix_seconds":target_finish["response"].get("n0_output",{}).get("inference_seconds",0.),
                    "dynamic_package_bytes":sender.get("state_package",{}).get("package_bytes"),
                    "forward_bytes":row["forward_bytes"],"ack_bytes":row["ack_bytes"],
                    "prepared_before_handoff":handoff[0]["preparation_issued_before_handoff"],
                    "source_prepare_seconds_outside_request":row["source_preparation_outside_request"]["wall_seconds"]})
    if generate_calls!=plan["scientific_budget"]["hard_generate_ceiling"]:
        raise ValueError(f"generate accounting mismatch: {generate_calls}")
    metrics=("workflow_wall_seconds","handoff_to_completion_seconds","prefix_wall_seconds",
        "target_prepare_work_seconds","target_prepare_wait_seconds","transfer_seconds",
        "state_export_seconds","restore_and_persist_seconds","forward_bytes")
    summary={arm:{metric:mean(row[metric] for row in rows if row["arm"]==arm)
                   for metric in metrics} for arm in ARMS}
    pairs=[]
    for case_id in cases:
        for repetition in range(plan["repetitions"]):
            matched={r["arm"]:r for r in rows if r["case_id"]==case_id and r["repetition"]==repetition}
            for baseline,candidate in [("restart","recover"),("recover","prepare_recover"),("restart","prepare_recover")]:
                pairs.append({"case_id":case_id,"repetition":repetition,"baseline":baseline,"candidate":candidate,
                    "workflow_seconds_saved":matched[baseline]["workflow_wall_seconds"]-matched[candidate]["workflow_wall_seconds"],
                    "blackout_seconds_saved":matched[baseline]["handoff_to_completion_seconds"]-matched[candidate]["handoff_to_completion_seconds"]})
    # Analytic boundary is explicitly separate from measured data. Wire format
    # expansion is included. No unmeasured large-state transfer is called real.
    restore=median(r["state_export_seconds"]+r["restore_and_persist_seconds"] for r in rows if r["arm"]=="recover")
    repeated=median(r["target_repeated_prefix_seconds"] for r in rows if r["arm"]=="restart")
    state_packet=next(r["forward_bytes"] for r in rows if r["arm"]=="recover")
    input_packet=next(r["forward_bytes"] for r in rows if r["arm"]=="restart")
    boundaries=[]
    for rate in (.01,.1,1.,10.,100.):
        boundary=input_packet+(repeated-restore)*rate*1e6/8
        boundaries.append({"mbps_assumption":rate,"break_even_forward_wire_bytes":boundary,
            "observed_state_forward_bytes":state_packet,
            "classification":"analytic sensitivity from measured median CPU costs, not measured links or large states"})
    mechanism={"native_handoff_cases":len(cases),"technical_workflow_node_count":2,
        "technical_workflow_edges":[["n0","n1"]],"distinct_adapter_count":1,
        "concurrent_request_count":1,"capacity_competition_exercised":False,
        "all_causal_targets_match_observed":all(c["predicted_target"]==c["target_rsu"] for c in cases.values()),
        "strong_causal_reference":"prepare_recover is itself the free-target causal preparation rule; recover is the no-prediction recovery rule",
        "learned_candidate_evaluated":False,"achievable_gap_over_strong_rule":"UNVERIFIED",
        "graph_sensitive_decision_bottleneck":"NOT_EXERCISED",
        "training_gate":"DO_NOT_START_TRAINING",
        "reason":"bounded linear single-adapter workflow validates execution; it cannot identify a graph/cache coordination learning advantage",
        "next_required_contract":"released real DAG with multiple adapter requirements, finite RSU memory, competing demand/link resources, model-validated state/cost binding; input-only development opportunity first"}
    return {"status":"PASS","scientific_git_commit":freeze["git_commit"],"rows":rows,
        "summary":summary,"paired_differences":pairs,"analytic_boundaries":boundaries,
        "generate_calls":generate_calls,"cells":len(rows),"unique_worker_pids":len(used_pids),
        "case_count":len(cases),"repetitions":plan["repetitions"],
        "statistics":"descriptive finite-case measurements only; repeated timing trials are not independent mobility/workload samples",
        "stage_2_gate":mechanism,"training_calls":0,"formal_calls":0,"holdout_calls":0,
        "limitations":["localhost emulation, not wireless or two physical RSUs",
            "accelerated blocking mobility replay, not real-time deadlines",
            "two-node technical DAG, not execution of Alibaba task semantics",
            "one fixed image without application ground-truth accuracy",
            "static weights prestaged on disk; source memory preloaded equally",
            "three handoff boundaries from a known source prefix, not sealed confirmation",
            "no algorithm superiority, graph benefit or convergence claim"]}


def export(study: Path, output: Path) -> dict:
    result=analyze(study)
    output.mkdir(parents=True,exist_ok=False)
    (output/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    for name in ("rows","paired_differences","analytic_boundaries"):
        with (output/f"{name}.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(result[name][0]));writer.writeheader();writer.writerows(result[name])
    inventory=[{"path":str(path.relative_to(study)),"bytes":path.stat().st_size,"sha256":sha256_file(path)}
        for path in sorted(study.rglob("*")) if path.is_file()]
    (output/"input_integrity_manifest.json").write_text(json.dumps({"files":inventory},indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=export(args.study,args.output)
    print(json.dumps({"status":result["status"],"cells":result["cells"],"generate_calls":result["generate_calls"],
        "summary":result["summary"],"training_gate":result["stage_2_gate"]["training_gate"]}))
