"""Independent stdlib consumer for the local technical cost experiment.

No producer import, model import, network, inference, retry or benchmark execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import struct
import statistics
import time
from typing import Any

OLD_PLAN_SHA = "74b428386f785e4942cde2ebcaf1642077ea43f4f6f4743a907b0095120c208c"
OLD_REFERENCE_SHA = "f789ba8663495692bac2193c45677430e3ca6ebd89396ffd658d4a29c4b4abee"
GENERATION = {"do_sample": False, "num_beams": 1, "max_new_tokens": 8, "use_cache": True}
VERSIONS = {"python": "3.9.6", "torch": "2.8.0", "transformers": "4.49.0", "peft": "0.14.0", "accelerate": "1.2.1", "safetensors": "0.5.3"}
COUNTERS = ("processor_load", "base_load", "adapter_load", "activate", "generate", "transfer", "encode", "durable_save", "restore", "prompt_rebuild")
EXPECTED = dict(zip(COUNTERS, (12, 12, 16, 20, 28, 12, 4, 4, 4, 4)))
ACTIVE_DEADLINE_NS = None
CLOCK_CONTRACT={"api":"time.clock_gettime_ns","clock":"CLOCK_MONOTONIC_RAW","clock_id":4,"unit":"ns","namespace":"system_kernel_monotonic_raw_same_host","platform":"Darwin","offset_normalization":False,"automatic_fallback":False}


def clock_ns() -> int:
    if platform.system()!="Darwin" or getattr(time,"CLOCK_MONOTONIC_RAW",None)!=4:
        raise RuntimeError("fixed Darwin kernel CLOCK_MONOTONIC_RAW=4 unavailable; no fallback")
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            require(ACTIVE_DEADLINE_NS is None or clock_ns()<=ACTIVE_DEADLINE_NS,"consumer hashing deadline")
            h.update(part)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def require(condition: bool, label: str):
    if not condition:
        raise ValueError(label)


def expected_matrix() -> list[dict]:
    rows = [{"id": "T1_"+label, "phase": "T1", "ordinal": int(label[1]), "adapters": [label[0]]} for label in ("A0", "B0", "A1", "B1", "B2", "A2", "A3", "B3")]
    rows += [{"id": "T2_ABA"+str(i), "phase": "T2", "ordinal": i, "adapters": ["A", "B"]} for i in range(4)]
    rows += [{"id": "T3_state"+str(i), "phase": "T3", "ordinal": i, "adapters": []} for i in range(4)]
    rows += [{"id": "T4_"+obj+str(i), "phase": "T4", "ordinal": i, "object": obj, "adapters": []} for i in range(4) for obj in ("state_package", "adapter_A_weight", "adapter_B_weight")]
    return rows


def bound(spec: dict):
    path = Path(spec["path"])
    require(digest(path) == spec["sha256"], "bound old metadata hash")
    return load(path)


def state_semantics(state: dict, identity: dict):
    p = state.get("payload", {})
    require(state.get("schema_version") == "ppo_mec.two_node_workflow_state.v1" and state.get("payload_sha256") == sha(canonical(p)), "state envelope/hash")
    for key, val in {"workflow_id": "technical_vision_description_suffix_v1", "nodes": ["n0", "n1"], "edges": [["n0", "n1"]], "completed_nodes": ["n0"], "remaining_nodes": ["n1"], "next_node": "n1", "identity": identity}.items():
        require(p.get(key) == val, "state node/identity:"+key)
    require(not any(key in p for key in ("n1_output", "final_output", "answer")), "injected final answer")
    n0 = p.get("n0_output", {})
    require(isinstance(n0.get("raw_text"), str) and bool(n0["raw_text"].strip()) and isinstance(n0.get("token_ids"), list) and bool(n0["token_ids"]) and all(type(x) is int for x in n0["token_ids"]) and p.get("suffix_input_material") == n0["raw_text"], "upstream material")
    require(p.get("generation", {}).get("n1") == GENERATION, "state full generation contract")
    for key in ("rng_state", "tensor_state"):
        require(p.get(key, {}).get("required") is False and bool(p[key].get("reason")), "continuation boundary:"+key)


def check_plan(plan: dict, allow_synthetic: bool):
    synthetic = plan.get("contract") == "local_adapter_cost_synthetic_plan_v1"
    require((synthetic and allow_synthetic) or plan.get("contract") == "local_adapter_cost_execution_plan_v1", "real/synthetic contract")
    require(plan.get("clock_contract")==CLOCK_CONTRACT,"fixed kernel clock contract")
    clocks=bound(plan["clock_qualification"])
    require(clocks.get("status")=="CLOCK_NAMESPACE_QUALIFIED" and clocks.get("clock_contract")==CLOCK_CONTRACT and len(clocks.get("probes",[]))==2,"clock namespace qualification")
    for row in clocks["probes"]:
        require(row["exit_code"]==0 and row["common_epoch_containment"] and row["parent_begin_ns"]<=row["native"]["begin_ns"]<=row["native"]["end_ns"]<=row["parent_end_ns"],"raw clock brackets")
    require(plan["generation"] == GENERATION and plan["matrix"] == expected_matrix(), "generation/matrix")
    runtime = plan["runtime"]
    for key, val in {"device": "cpu", "dtype": "float32", "batch_size": 1, "intra_threads": 1, "inter_threads": 1, "local_files_only": True, "trust_remote_code": False, "automatic_retry": False, "versions": VERSIONS}.items():
        require(runtime.get(key) == val, "fixed runtime:"+key)
    require(plan.get("source_image_access_allowed") is False and plan.get("prefix_calls") == 0, "suffix scope")
    hardware_keys={"host_model","CPU_model","physical_cores","logical_cores","RAM_bytes","OS_build","architecture","disk_model_and_filesystem","AC_power_state","power_mode"}
    require(set(plan.get("hardware_expected",{}))==hardware_keys and all(v not in (None,"","UNVERIFIED") for v in plan["hardware_expected"].values()) and plan["hardware_expected"]["AC_power_state"]=="AC","hardware/power field completeness")
    if not synthetic:
        require(plan["old_plan"]["sha256"] == OLD_PLAN_SHA and plan["old_reference"]["sha256"] == OLD_REFERENCE_SHA, "immutable old reference")
        require(plan["caps"] == {"wall_seconds": 5400, "disk_bytes": 1073741824, "log_bytes": 67108864, "model_processes": 12, "processor_load": 12, "base_load": 12, "adapter_load": 16, "activate": 20, "generate": 28, "transfer": 12, "payload_bytes": 656269664, "application_bytes": 656271008}, "real hard caps")
    old, reference = bound(plan["old_plan"]), bound(plan["old_reference"])
    if not synthetic:
        expected={"base":(1015025832,"d05b567eeaf534e83d375551f068ed57b5f52d37c657197f644af5ef9db091a2"),"A":(154423432,"95103f5d6547c194770b03ee0ac24fe6fc125eb8890e992977be8dcbfea10f83"),"B":(9641944,"e899c9c13b53bf80bcfc86048f5678a727e8c1838dc824156b521d5cdaafdd2d")}
        for name,(size,h) in expected.items():
            r=plan["resources"][name];files={x["name"]:x for x in r["files"]}
            require(len(files)==len(r["files"]) and (r["weight_bytes"],r["weight_sha256"])==(size,h) and (files[r["weight_file"]]["bytes"],files[r["weight_file"]]["sha256"])==(size,h),"resource native weight verification matrix")
            require(all(Path(x["name"]).name==x["name"] for x in r["files"]),"resource path escape")
            if name!="base":
                expected_config="d4e1d0e6c2c77424d4ed76956cb5642a2fb8e64f233b5a4874e5341e9e085216" if name=="A" else "f32a6b2b99479a4d680495593cebb13da0623abc88d099d58a7500a32c40b3a7"
                require(files["adapter_config.json"]["sha256"]==expected_config,"adapter config identity")
        basefiles={x["name"]:x for x in plan["resources"]["base"]["files"]}
        for filename,key in {"processor_config.json":"config_sha256","preprocessor_config.json":"preprocessor_sha256","tokenizer.json":"tokenizer_sha256","chat_template.json":"chat_template_sha256"}.items():
            require(basefiles[filename]["sha256"]==old["identity"]["processor"][key],"processor/tokenizer source identity")
    require(old["generation"]["n1"] == GENERATION and reference["runtime"] == plan["old_runtime_identity"], "old generation/runtime identity")
    require(all(reference["runtime"][k] == VERSIONS[k] for k in VERSIONS), "old package versions")
    require(reference["n1_input"] == plan["reference_input"] and reference["n1_output"]["token_ids"] == plan["reference_A_tokens"], "old input/token reference")
    state = bound(plan["state_file"])
    state_semantics(state, plan["state_identity"])
    manifest=bound(plan["state_manifest"])
    state_bytes=Path(plan["state_file"]["path"]).read_bytes();manifest_bytes=Path(plan["state_manifest"]["path"]).read_bytes()
    require(manifest.get("allowed_files")==["state.json","manifest.json"] and manifest.get("contains_final_output") is False and manifest.get("contains_model_weights") is False and manifest["state"]["sha256"]==sha(state_bytes) and manifest["state"]["bytes"]==len(state_bytes),"bound source state manifest")
    objects={"state_package":{"bytes":len(state_bytes)+len(manifest_bytes),"sha256":sha(state_bytes+manifest_bytes)},"adapter_A_weight":{"bytes":plan["resources"]["A"]["weight_bytes"],"sha256":plan["resources"]["A"]["weight_sha256"]},"adapter_B_weight":{"bytes":plan["resources"]["B"]["weight_bytes"],"sha256":plan["resources"]["B"]["weight_sha256"]}}
    require(plan["transfer_objects"]==objects,"transfer source object projection")
    if not synthetic:
        require(len(state_bytes)==1703 and len(manifest_bytes)==337 and plan["state_file"]["sha256"]=="53c4f94fa8b50025a2ae4e532d31e211996e49eb302859f37561ef93e66409c9" and plan["state_manifest"]["sha256"]=="362123e41c78f9fe53cd7b36faf8337d18133caf3e9040175d4e1f520dd14839","immutable old state identity")
        require(reference["identity"]==plan["state_identity"] and state["payload"]["generation"]==old["generation"],"old complete state identity/generation")
    prompt = old["workflow"]["n1_prompt_template"].replace("{n0_output}", state["payload"]["n0_output"]["raw_text"])
    require(prompt == reference["n1_input"]["prompt"], "state rebuild exact reference prompt")
    return synthetic


def parse_events(path: Path, begin: int, end: int, failed: bool):
    if not path.exists():
        require(failed, "missing native ledger")
        return [], {k: 0 for k in COUNTERS}, end-begin
    starts, ends = {}, {}
    lines=path.read_bytes().splitlines()
    for i,line in enumerate(lines):
        try:row=json.loads(line)
        except ValueError:
            require(failed and i==len(lines)-1,"malformed nonterminal native line")
            break
        key = row["id"]
        target = starts if row["edge"] == "start" else ends
        require(key not in target, "duplicate event edge")
        target[key] = row
    require(set(ends) <= set(starts), "end without start")
    spans = []
    for key, start in starts.items():
        stop = ends.get(key)
        require(stop is not None or failed, "unclosed successful native operation")
        stop_ns = stop["ns"] if stop else end
        require(type(start["ns"]) is int and type(stop_ns) is int and begin <= start["ns"] <= stop_ns <= end, "native timestamp containment")
        require(stop is None or stop["status"] in ("PASS", "FAIL"), "native terminal status")
        if not failed:
            require(stop["status"] == "PASS", "failed operation in successful attempt")
        spans.append({**start, "end_ns": stop_ns,"native_end_ns":stop["ns"] if stop else None,"censored_supervisor_upper_bound":stop is None, "status": stop["status"] if stop else "INTERRUPTED"})
    children = {}
    for span in spans:
        parent = span.get("parent")
        if parent:
            require(parent in starts, "missing nested parent")
            owner = next(x for x in spans if x["id"] == parent)
            require(owner["ns"] <= span["ns"] <= span["end_ns"] <= owner["end_ns"], "nested interval containment")
        children.setdefault(parent, []).append(span)
    for siblings in children.values():
        ordered = sorted(siblings, key=lambda x: x["ns"])
        require(all(a["end_ns"] <= b["ns"] for a, b in zip(ordered, ordered[1:])), "overlapping partition/siblings")
    top = children.get(None, [])
    residual = end-begin-sum(x["end_ns"]-x["ns"] for x in top)
    require(residual >= 0, "negative process residual")
    counts = {k: sum(x["kind"] == k for x in spans) for k in COUNTERS}
    return spans, counts, residual


def validate_terminal_prefix(terminal: dict, statuses: list[tuple[str,str]]) -> list[str]:
    ids=[x["id"] for x in expected_matrix()]
    require([x[0] for x in statuses]==ids[:len(statuses)],"exact observed attempt prefix")
    failed=[i for i,x in enumerate(statuses) if x[1]=="FAIL"]
    require(all(x[1] in ("PASS","FAIL") for x in statuses) and (not failed or failed==[len(statuses)-1]),"failure must terminate prefix")
    completed=[identity for identity,status in statuses if status=="PASS"]
    require(terminal["completed"]==completed,"native versus terminal completed matrix")
    if terminal["status"]=="PASS":
        require(len(statuses)==28 and not failed and terminal.get("failure_stage") is None,"complete successful prefix")
    else:
        stage=terminal.get("failure_stage")
        require(terminal["failure"] is not None,"failure reason")
        if failed:require(stage=="in_attempt" and terminal.get("current_attempt")==statuses[-1][0],"current failed attempt provenance")
        else:require(stage in ("preflight","before_attempt","finalization") and terminal.get("current_attempt") is None and (stage!="preflight" or not statuses),"between-attempt failure without fictitious attempt")
    return completed


def audit_model(directory: Path, row: dict, plan: dict, ref_B: dict):
    child = load(directory / "child_terminal.json")
    require(child["status"] == "PASS" and child["versions"] == VERSIONS and child["threads"] == [1, 1] and child["device"] == "cpu" and child["dtype"] == "float32", "child runtime identity")
    if plan["contract"]=="local_adapter_cost_execution_plan_v1":
        require(child["module_origins"]==plan["old_runtime_identity"]["module_origins"],"runtime module origin drift")
        require(type(child["memory"]["peak_rss_bytes"]) is int and child["memory"]["peak_rss_bytes"]>0 and child["memory"]["current_rss_bytes"] is None and child["memory"]["GPU_memory_bytes"] is None and child["memory"]["observations"]==1,"native process peak memory scope")
    inp = load(directory / "input.json")
    require(inp.get("source_image_access_count") == 0 and all(inp.get(k) == plan["reference_input"][k] for k in ("prompt", "rendered_prompt", "input_ids")), "raw input reference")
    files = sorted(directory.glob("output_*.json"))
    sequence = [row["adapters"][0]]*2 if row["phase"] == "T1" else ["A", "B", "A"]
    require(len(files) == len(sequence) == child["outputs"], "raw generation count")
    for path, active in zip(files, sequence):
        item = load(path)
        require(item["adapter"] == active and item["generation"] == GENERATION and item["adapter_weight_sha256"] == plan["resources"][active]["weight_sha256"], "raw adapter generation identity")
        require(item["status"].get("active_adapters") == [active] and active in item["status"].get("available_adapters", []), "raw active adapter")
        params = item["loaded_parameters"]
        require(params and any(x["nonzero"] > 0 for x in params) and all("."+active+"." in x["key"] and len(x["sha256"]) == 64 and x["dtype"]=="torch.float32" for x in params), "raw loaded parameter identity")
        token_ids = item["token_ids"]
        require(isinstance(token_ids, list) and 0 < len(token_ids) <= 8 and all(type(x) is int for x in token_ids), "raw token format")
        if active == "A":
            require(token_ids == plan["reference_A_tokens"], "old ALPR token reference fidelity")
            require(item["decoded_text"]==ref_B["_A_text"],"old ALPR decoded reference fidelity")
        else:
            ref_B.setdefault("tokens", token_ids)
            require(token_ids == ref_B["tokens"], "Helmet repeat fidelity")
            ref_B.setdefault("text",item["decoded_text"])
            require(item["decoded_text"]==ref_B["text"],"Helmet decoded repeat fidelity")
        key = active+"_parameters"
        ref_B.setdefault(key, params)
        require(params == ref_B[key], "adapter parameters changed across activation/process")
    return child["pid"]


def audit_state(directory: Path, plan: dict):
    package = directory / "package"
    require(sorted(x.name for x in package.iterdir()) == ["manifest.json", "state.json"], "state package file allowlist")
    state = load(package / "state.json")
    state_semantics(state, plan["state_identity"])
    require(digest(package/"state.json") == plan["state_file"]["sha256"] and digest(package/"manifest.json") == plan["state_manifest"]["sha256"], "state package exact bytes")
    manifest = load(package/"manifest.json")
    require(manifest.get("contains_final_output") is False and manifest.get("contains_model_weights") is False and manifest["state"]["bytes"] == (package/"state.json").stat().st_size and manifest["state"]["sha256"] == digest(package/"state.json"), "state manifest accounting")
    inp = load(directory/"restored_input.json")
    require(inp["prompt"] == plan["reference_input"]["prompt"] and inp["identity"] == plan["state_identity"] and inp["source_image_access_count"] == 0, "restored suffix input")


def audit_transfer(directory: Path, row: dict, plan: dict, failed: bool, begin: int, end: int):
    spec=plan["transfer_objects"][row["object"]]
    bins=[p for p in (directory/"payload.bin",directory/"payload.partial") if p.exists()]
    require(len(bins)<=1,"ambiguous dual complete/partial payload files")
    payload=bins[0] if bins else None
    actual=payload.stat().st_size if payload else 0
    wire,_,_=parse_events(directory/"wire.jsonl",begin,end,failed)
    require(all(s["kind"] in ("send_header","send_payload","receive_ack") and 0<s["requested_bytes"]<=65536 for s in wire),"wire operation vocabulary/bound")
    require(sum(s["kind"]=="send_header" for s in wire)<=1 and sum(s["kind"]=="receive_ack" for s in wire)<=1,"wire duplicate header/ACK")
    header=sum(s["requested_bytes"] for s in wire if s["kind"]=="send_header")
    body=[s for s in wire if s["kind"]=="send_payload"]
    upper=sum(s["requested_bytes"] for s in body)
    complete=sum(s["requested_bytes"] for s in body if s["status"]=="PASS")
    uncertain=upper-complete
    require(header in (0,64) and 0<=uncertain<=65536 and 0<=actual<=upper<=spec["bytes"],"partial writeahead/read upper bound")
    ack_upper=sum(s["requested_bytes"] for s in wire if s["kind"]=="receive_ack")
    ack_path=directory/"ack.bin"
    ack_bytes=ack_path.stat().st_size if ack_path.exists() else 0
    charge=header+upper+max(ack_upper,ack_bytes)
    native_path=directory/"transfer_native.json"
    if native_path.exists():
        native=load(native_path)
        require((native["header_upper_bytes"],native["payload_complete_bytes"],native["payload_uncertain_upper_bytes"],native["receiver_bytes"],native["application_charge_upper_bytes"])==(header,complete,uncertain,actual,charge),"writeahead native versus summary partial bill")
    else:
        require(failed,"successful transfer missing native receipt")
    completed_object=bool(payload and payload.name=="payload.bin")
    if completed_object or not failed:
        require(payload is not None and actual==upper==complete==spec["bytes"] and digest(payload)==spec["sha256"],"complete object hash/length independent of enclosing FAIL")
        header_data=(directory/"header.bin").read_bytes();ack=ack_path.read_bytes()
        require(len(header_data)==64 and struct.unpack(">8sQ32s16s",header_data)==(b"PPOCOST1",spec["bytes"],bytes.fromhex(spec["sha256"]),row["object"].encode().ljust(16,b"\0")),"native header")
        require(len(ack)==48 and struct.unpack(">8sQ32s",ack)==(b"PPOACK01",spec["bytes"],bytes.fromhex(spec["sha256"])) and charge==spec["bytes"]+112,"native ACK/application bill")
        with payload.open("rb") as f:
            for span in body:require(sha(f.read(span["requested_bytes"]))==span["chunk_sha256"],"raw chunk/writeahead hash")
    if not failed:require(completed_object,"successful transport not committed")
    return upper,charge


def audit(root: Path, expected_plan_sha: str, *, allow_synthetic: bool = False) -> dict:
    global ACTIVE_DEADLINE_NS
    started = clock_ns()
    ACTIVE_DEADLINE_NS=started+600*10**9
    require(root.is_dir(), "missing run")
    plan = load(root/"plan.json")
    require(digest(root/"plan.json") == expected_plan_sha, "prospective plan binding")
    synthetic = check_plan(plan, allow_synthetic)
    reservation=plan.get("terminal_pipeline")
    if reservation is not None:
        require(reservation=={"producer_seconds":4800,"consumer_seconds":600,"combined_seconds":5400},"exact producer/consumer reservation")
        require(Path(plan["outputs"]["raw_root"]).resolve()==root.resolve(),"frozen raw output path")
    files = list(root.rglob("*"))
    require(not any(p.is_symlink() for p in files), "symlink artifact")
    actual_files = {str(p.relative_to(root)): p for p in files if p.is_file() and p.name not in ("inventory.json","execution_finish.json")}
    require({p.name for p in root.iterdir()}=={"plan.json","preflight.json","terminal_receipt.json","inventory.json","execution_finish.json","attempts"},"unexpected root artifact")
    rows = load(root/"inventory.json")["files"]
    require(len(rows) == len({r["path"] for r in rows}) and set(actual_files) == {r["path"] for r in rows}, "inventory exact file set")
    for row in rows:
        path = actual_files[row["path"]]
        require(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], "raw artifact identity")
    terminal = load(root/"terminal_receipt.json")
    require((terminal["end_ns"]-terminal["begin_ns"])/1e9<=plan["caps"]["wall_seconds"],"executor wall cap")
    closure=load(root/"execution_finish.json")
    require(closure["inventory_sha256"]==digest(root/"inventory.json") and closure["terminal_sha256"]==digest(root/"terminal_receipt.json") and closure["inventory_hashing_included_in_wall"] is True,"execution closure hashes")
    require(closure["begin_ns"]==terminal["begin_ns"] and terminal["end_ns"]<=closure["end_ns"]<=started,"execution closure clock")
    require((closure["end_ns"]-closure["begin_ns"])/1e9<=plan["caps"]["wall_seconds"],"finalization wall cap")
    if reservation is not None:
        require((closure["end_ns"]-closure["begin_ns"])/1e9<=reservation["producer_seconds"],"producer reserved4800s cap")
    if not synthetic:
        ACTIVE_DEADLINE_NS=min(ACTIVE_DEADLINE_NS,closure["begin_ns"]+plan["caps"]["wall_seconds"]*10**9)
    require(terminal["status"] in ("PASS", "FAIL") and terminal["synthetic_only"] is synthetic and terminal["paper_goal_complete"] is False, "terminal scope")
    require(terminal["begin_ns"] <= terminal["end_ns"], "global monotonic clock")
    ids = [x["id"] for x in expected_matrix()]
    observed_dirs = {p.name for p in (root/"attempts").iterdir() if p.is_dir()}
    n = len(observed_dirs)
    require(observed_dirs == set(ids[:n]), "prefix only/no replacement attempt")
    raw_statuses=[(identity,load(root/"attempts"/identity/"attempt_terminal.json")["status"]) for identity in ids[:n]]
    validate_terminal_prefix(terminal,raw_statuses)
    preflight = load(root/"preflight.json")
    require(preflight.get("synthetic_only",False) is synthetic,"synthetic preflight boundary")
    if preflight["status"]=="FAIL":
        require(terminal["status"]=="FAIL" and n==0,"preflight failure must stop before attempts")
    else:
        require(preflight["status"]=="PASS" and preflight["versions"] == VERSIONS and preflight["source_commit"] == plan["source_commit"], "preflight library/source")
        require(preflight.get("clock_contract")==CLOCK_CONTRACT and preflight.get("clock_qualification_sha256")==plan["clock_qualification"]["sha256"],"preflight clock identity")
        require(all(preflight["hardware"][k] == v for k,v in plan["hardware_expected"].items()), "hardware gate")
        if not synthetic:
            require(preflight["verified_resource_files"]==[{"resource":name,**spec} for name,resource in plan["resources"].items() for spec in resource["files"]],"complete native resource verification matrix")
    counts = {k: 0 for k in COUNTERS}; completed_counts={k:0 for k in COUNTERS};ref_B = {"_A_text":bound(plan["old_reference"])["n1_output"]["decoded_text"]}; pids=[]; samples=[]; payload_bill=app_bill=0; groups={};utility_launches=0
    previous_end = terminal["begin_ns"]
    for index, row in enumerate(expected_matrix()[:n]):
        directory = root/"attempts"/row["id"]
        begin = load(directory/"started.json"); stop = load(directory/"attempt_terminal.json")
        require(begin["row"] == row and begin["plan_sha256"] == expected_plan_sha and begin["begin_ns"] == stop["begin_ns"], "attempt plan/identity")
        failed = stop["status"] == "FAIL"
        require(previous_end <= stop["begin_ns"] <= stop["end_ns"] <= terminal["end_ns"], "serial attempt clock")
        previous_end = stop["end_ns"]
        require((stop["end_ns"]-stop["begin_ns"])/1e9 <= (300 if row["phase"] in ("T1","T2") else 60 if row["phase"]=="T4" else 10),"per-attempt deadline")
        common={"started.json","attempt_terminal.json","events.jsonl","stdout.log","stderr.log"}
        if row["phase"] in ("T1","T2"):
            allowed=common|{"child_terminal.json","input.json","stdout.log","stderr.log"}|{"output_%02d.json"%i for i in range(2 if row["phase"]=="T1" else 3)}
        elif row["phase"]=="T3":
            allowed=common|{"package","restored_input.json","utility_launch.json","utility_terminal.json"}
        else:
            allowed=common|{"transfer_native.json","wire.jsonl","payload.bin","payload.partial","header.bin","ack.bin","utility_launch.json","utility_terminal.json"}
        require({p.name for p in directory.iterdir()}<=allowed,"unexpected attempt artifact")
        spans, current, residual = parse_events(directory/"events.jsonl", stop["begin_ns"], stop["end_ns"], failed)
        for key in counts:
            counts[key] += current[key]
            completed_counts[key]+=sum(s["kind"]==key and s["status"]=="PASS" for s in spans)
        if not failed:
            top = [s["kind"] for s in spans if s.get("parent") is None]
            expected_ops = {"T1": ["processor_load","base_load","adapter_load","input_rebuild","activate","generate","generate"], "T2": ["processor_load","base_load","adapter_load","adapter_load","input_rebuild","activate","generate","activate","generate","activate","generate"], "T3": ["encode","durable_save","restore","prompt_rebuild"], "T4": ["transfer"]}[row["phase"]]
            require(top == expected_ops, "exact native operation order/count")
            if row["phase"] in ("T1", "T2"):
                expected_active=[row["adapters"][0]] if row["phase"]=="T1" else ["A","B","A"]
                expected_generated=[row["adapters"][0]]*2 if row["phase"]=="T1" else ["A","B","A"]
                require([s.get("adapter") for s in spans if s["kind"]=="activate"]==expected_active and [s.get("adapter") for s in spans if s["kind"]=="generate"]==expected_generated,"native adapter activation/generation sequence")
                for generate in [s for s in spans if s["kind"]=="generate"]:
                    require([s["kind"] for s in spans if s.get("parent")==generate["id"]]==["generate_compute","output_copy_decode"],"native generate nested partition")
                pids.append(audit_model(directory,row,plan,ref_B))
            elif row["phase"] == "T3":
                for kind in ("file_write","file_fsync","file_rename","directory_fsync"):
                    require(sum(s["kind"]==kind for s in spans)==2,"state native durable calls:"+kind)
                audit_state(directory,plan)
        if row["phase"] in ("T3","T4") and (directory/"utility_launch.json").exists():
            utility_launches+=1
            launch=load(directory/"utility_launch.json");finish=load(directory/"utility_terminal.json")
            require(launch["row"]==row["id"] and launch["role"]=="--helper" and finish["reaped"] and launch["begin_ns"]==finish["begin_ns"] and stop["begin_ns"]<=launch["begin_ns"]<=finish["end_ns"]<=stop["end_ns"],"utility process supervision/cleanup")
            require((finish["end_ns"]-finish["begin_ns"])/1e9<=launch["limit_seconds"]<= (10 if row["phase"]=="T3" else 60),"utility remaining deadline")
        if row["phase"] == "T4":
            body, app = audit_transfer(directory,row,plan,failed,stop["begin_ns"],stop["end_ns"]);payload_bill += body;app_bill += app
        if not failed and row["ordinal"]>0:
            occurrences={}
            for span in spans:
                if span.get("parent") is not None:
                    continue
                context=row.get("object",row["adapters"][0] if row["phase"]=="T1" else "resident" if row["phase"]=="T2" else "state")
                label="|".join((row["phase"],context,span["kind"],span.get("adapter","")))
                occurrence=occurrences.get(label,0);occurrences[label]=occurrence+1
                groups.setdefault(label+"|"+str(occurrence),[]).append((span["end_ns"]-span["ns"])/1e6)
        samples.append({"id": row["id"], "qualification": row["ordinal"] == 0, "status": stop["status"], "process_residual_ns": residual, "partitioned_ns": [{"kind":s["kind"],"ns":s["end_ns"]-s["ns"],"censored_upper_bound":s["censored_supervisor_upper_bound"]} for s in spans if not s.get("parent")], "nested_ns": [{"kind":s["kind"],"ns":s["end_ns"]-s["ns"],"censored_upper_bound":s["censored_supervisor_upper_bound"]} for s in spans if s.get("parent")]})
    require(len(pids) == len(set(pids)), "fresh independent model process IDs")
    for key,val in counts.items():
        require(val <= EXPECTED[key], "attempt hard call cap:"+key)
        if terminal["status"] == "PASS":
            require(val == EXPECTED[key], "complete raw call matrix:"+key)
    require((terminal["end_ns"]-terminal["begin_ns"])/1e9 <= plan["caps"]["wall_seconds"], "executor wall cap")
    retained = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    require(retained <= plan["caps"]["disk_bytes"], "retained disk cap")
    require(sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and p.suffix in (".log",".jsonl")) <= plan["caps"]["log_bytes"], "log cap")
    require(payload_bill <= plan["caps"]["payload_bytes"] and app_bill <= plan["caps"]["application_bytes"], "cumulative payload/application cap")
    require(utility_launches<=16,"utility process launch cap")
    if not synthetic and terminal["status"]=="PASS":require(utility_launches==16 and plan.get("utility_processes_cap")==16,"complete utility launch matrix")
    require(terminal["summary"]=={"raw_calls":counts,"raw_completed_calls":completed_counts,"payload_charge_upper_bytes":payload_bill,"application_charge_upper_bytes":app_bill,"utility_process_launches":utility_launches},"producer summary differs from independent native reconstruction")
    if terminal["status"] == "PASS":
        require(payload_bill == plan["caps"]["payload_bytes"] and app_bill == plan["caps"]["application_bytes"], "exact aggregate transport bill")
    require((clock_ns()-started)/1e9 < 600, "consumer deadline")
    require(synthetic or (clock_ns()-terminal["begin_ns"])/1e9 <= plan["caps"]["wall_seconds"],"executor plus consumer total deadline")
    aggregates={k:{"all_ms":v,"n":len(v),"mean_ms":statistics.mean(v),"sample_sd_ms":statistics.stdev(v) if len(v)>1 else None,"median_ms":statistics.median(v),"min_ms":min(v),"max_ms":max(v),"is_CI":False} for k,v in groups.items()}
    if terminal["status"]=="PASS":
        require(all(len(v)==3 for v in groups.values()),"exact three formal samples per component")
    return {"status": "SOFTWARE_SYNTHETIC_ONLY" if synthetic else ("AUDITED_PASS_TECHNICAL_COMPONENT_COST_ONLY" if terminal["status"]=="PASS" else "AUDITED_FAILURE_PRESERVED"), "native_terminal": terminal["status"], "raw_calls": counts,"raw_completed_calls":completed_counts,"utility_process_launch_attempts":utility_launches, "payload_charge_upper_bytes": payload_bill, "application_charge_upper_bytes": app_bill, "retained_bytes": retained, "samples": samples,"aggregates":aggregates,"activation_scope":"activation_with_integrity_verification; pure set_adapter latency unavailable", "task_quality": "UNAVAILABLE", "wireless_RSU": "NOT_MEASURED", "paper_goal_complete": False, "synthetic_only": synthetic}


def observed_spent(root: Path) -> dict:
    """Diagnostic lower/upper observations on rejected raw data; never a PASS."""
    calls={k:0 for k in COUNTERS};body=app=tail_bytes=0
    for path in root.glob("attempts/*/events.jsonl"):
        lines=path.read_bytes().splitlines()
        for i,line in enumerate(lines):
            try:row=json.loads(line)
            except ValueError:tail_bytes+=len(line);continue
            if row.get("edge")=="start" and row.get("kind") in calls:calls[row["kind"]]+=1
    for path in root.glob("attempts/T4_*/wire.jsonl"):
        ack=0
        for line in path.read_bytes().splitlines():
            try:row=json.loads(line)
            except ValueError:tail_bytes+=len(line);continue
            if row.get("edge")!="start":continue
            amount=row.get("requested_bytes",0)
            if row.get("kind")=="send_payload":body+=amount;app+=amount
            elif row.get("kind")=="send_header":app+=amount
            elif row.get("kind")=="receive_ack":ack+=amount
        ack_path=path.parent/"ack.bin"
        app+=max(ack,ack_path.stat().st_size if ack_path.exists() else 0)
    return {"status":"OBSERVED_DIAGNOSTIC_NOT_INTEGRITY_OR_BUDGET_PASS","raw_operation_starts":calls,"payload_writeahead_upper_bytes":body,"application_writeahead_upper_bytes":app,"retained_file_bytes":sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),"malformed_tail_bytes":tail_bytes,"budget_compliant":False,"unknown_or_live_writer_possible":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--plan_sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--allow_synthetic",action="store_true",help="software-only fixture consumption; real producer CLI never accepts synthetic plans")
    args=parser.parse_args()
    require(not args.output.exists(), "audit is create-only")
    require(not args.output.resolve().is_relative_to(args.root.resolve()),"audit output must be outside immutable raw root")
    saved_plan=load(args.root/"plan.json")
    if saved_plan.get("terminal_pipeline") is not None:
        require(Path(saved_plan["outputs"]["audit"]).resolve()==args.output.resolve(),"frozen external audit output path")
    try:
        result=audit(args.root,args.plan_sha256,allow_synthetic=args.allow_synthetic)
    except Exception as exc:
        with args.output.open("x") as f:
            json.dump({"status":"AUDIT_REJECTED_PRESERVED","error":repr(exc),"root":str(args.root),"expected_plan_sha256":args.plan_sha256,"model_calls":0,"spent_observations":observed_spent(args.root),"paper_goal_complete":False},f,indent=2);f.write("\n")
        raise
    body=json.dumps(result,ensure_ascii=False,indent=2)+"\n"
    require(result["retained_bytes"]+len(body.encode())<=load(args.root/"plan.json")["caps"]["disk_bytes"],"consumer output disk charge")
    with args.output.open("x") as f:
        f.write(body)


if __name__ == "__main__":
    main()
