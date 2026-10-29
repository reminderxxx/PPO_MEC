"""Prospective, fail-closed local component cost experiment (never a quality test).

Importing this module imports no model library, opens no socket and reads no model.
The real CLI requires a separate exact execution grant and a clean source freeze.
Synthetic qualification uses injected backends; it cannot use the real CLI.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import plistlib
import queue
import shlex
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REAL_CONTRACT = "local_adapter_cost_execution_plan_v1"
SYNTHETIC_CONTRACT = "local_adapter_cost_synthetic_plan_v1"
PROPOSAL_SHA = "b0b0b9d362dd05b9a3070f59b5ec99a6eb124c616803ca98bd3c1ac0260e4f8d"
OLD_PLAN_SHA = "74b428386f785e4942cde2ebcaf1642077ea43f4f6f4743a907b0095120c208c"
OLD_REFERENCE_SHA = "f789ba8663495692bac2193c45677430e3ca6ebd89396ffd658d4a29c4b4abee"
VERSIONS = {"python": "3.9.6", "torch": "2.8.0", "transformers": "4.49.0", "peft": "0.14.0", "accelerate": "1.2.1", "safetensors": "0.5.3"}
GENERATION = {"do_sample": False, "num_beams": 1, "max_new_tokens": 8, "use_cache": True}
HEADER = struct.Struct(">8sQ32s16s")
ACK = struct.Struct(">8sQ32s")
MAGIC = b"PPOCOST1"
ACK_MAGIC = b"PPOACK01"
HARD = {"wall_seconds": 5400, "disk_bytes": 1073741824, "log_bytes": 67108864,
        "model_processes": 12, "processor_load": 12, "base_load": 12,
        "adapter_load": 16, "activate": 20, "generate": 28,
        "transfer": 12, "payload_bytes": 656269664, "application_bytes": 656271008}
KINDS = ("processor_load", "base_load", "adapter_load", "activate", "generate", "transfer", "encode", "durable_save", "restore", "prompt_rebuild")
CLOCK_CONTRACT={"api":"time.clock_gettime_ns","clock":"CLOCK_MONOTONIC_RAW","clock_id":4,"unit":"ns","namespace":"system_kernel_monotonic_raw_same_host","platform":"Darwin","offset_normalization":False,"automatic_fallback":False}
IDENTITY_FIELDS = ("host_model", "CPU_model", "physical_cores", "logical_cores", "RAM_bytes", "OS_build", "architecture", "disk_model_and_filesystem", "AC_power_state", "power_mode")


def clock_ns() -> int:
    if platform.system()!="Darwin" or getattr(time,"CLOCK_MONOTONIC_RAW",None)!=4:
        raise RuntimeError("fixed Darwin kernel CLOCK_MONOTONIC_RAW=4 unavailable; no fallback")
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path, check: Any = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            if check:
                check()
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def ledger_rows(path: Path) -> tuple[list[dict], int]:
    """A killed writer may leave one incomplete final line; never invent an end."""
    if not path.exists():return [],0
    lines=path.read_bytes().splitlines(keepends=True);rows=[]
    for i,line in enumerate(lines):
        try:rows.append(json.loads(line))
        except ValueError:
            if i!=len(lines)-1:raise ValueError("malformed nonterminal ledger line")
            return rows,len(line)
    return rows,0


def transfer_spent(directory: Path) -> tuple[int,int]:
    rows,_=ledger_rows(directory/"wire.jsonl")
    starts=[r for r in rows if r["edge"]=="start"]
    body=sum(r["requested_bytes"] for r in starts if r["kind"]=="send_payload")
    header=sum(r["requested_bytes"] for r in starts if r["kind"]=="send_header")
    ack=sum(r["requested_bytes"] for r in starts if r["kind"]=="receive_ack")
    if (directory/"ack.bin").exists():ack=max(ack,(directory/"ack.bin").stat().st_size)
    return body,header+body+ack


def qualify_shared_clock(interpreter: str, output: Path) -> dict:
    """Two exact-runtime, no-model clock probes; never normalize an offset."""
    if output.exists():raise FileExistsError("clock qualification create-only")
    expected="/Users/howen/Projects/PPO_MEC/artifacts/environments/adapter_state_acceptance_py39_v1/bin/python"
    if str(Path(interpreter).absolute())!=expected:raise ValueError("exact frozen interpreter required")
    clock_ns()
    rows=[];good=True
    code="import json,time,os,sys,platform; assert platform.system()=='Darwin' and time.CLOCK_MONOTONIC_RAW==4; a=time.clock_gettime_ns(4); b=time.clock_gettime_ns(4); print(json.dumps({'begin_ns':a,'end_ns':b,'pid':os.getpid(),'python':sys.version,'executable':sys.executable,'clock':{'api':'time.clock_gettime_ns','clock':'CLOCK_MONOTONIC_RAW','clock_id':4,'resolution':time.clock_getres(4),'platform':platform.system()}}))"
    for ordinal in range(2):
        parent_begin=clock_ns()
        try:result=subprocess.run([interpreter,"-c",code],capture_output=True,text=True,timeout=5)
        except Exception as exc:
            rows.append({"ordinal":ordinal,"parent_begin_ns":parent_begin,"parent_end_ns":clock_ns(),"error":repr(exc),"common_epoch_containment":False});good=False;break
        parent_end=clock_ns()
        native=json.loads(result.stdout) if result.returncode==0 else None
        okay=bool(native and parent_begin<=native["begin_ns"]<=native["end_ns"]<=parent_end and native["pid"]!=os.getpid() and native["clock"]["clock_id"]==4 and native["clock"]["platform"]=="Darwin")
        rows.append({"ordinal":ordinal,"parent_begin_ns":parent_begin,"parent_end_ns":parent_end,"exit_code":result.returncode,"stdout":result.stdout,"stderr":result.stderr,"native":native,"common_epoch_containment":okay})
        good=good and okay
        if not okay:break
    good=good and len(rows)==2 and rows[0]["native"]["pid"]!=rows[1]["native"]["pid"]
    record={"status":"CLOCK_NAMESPACE_QUALIFIED" if good else "CLOCK_NAMESPACE_FAILED","interpreter":interpreter,"parent_pid":os.getpid(),"clock_contract":CLOCK_CONTRACT,"parent_clock":{"api":"time.clock_gettime_ns","clock_id":4,"resolution":time.clock_getres(4)},"probes":rows,"offset_normalization":False,"model_calls":0,"network_calls":0,"upstream_old_perf_counter_diagnosis":"https://raw.githubusercontent.com/python/cpython/v3.9.6/Python/pytime.c lines771-821/963-970; no old-perf probe was run"}
    create_json(output,record)
    if not good:raise RuntimeError("clock namespace failed; no normalization or fallback")
    return record


def create_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n")


def disk_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def matrix() -> list[dict]:
    rows = []
    for label in ("A0", "B0", "A1", "B1", "B2", "A2", "A3", "B3"):
        rows.append({"id": "T1_" + label, "phase": "T1", "ordinal": int(label[1]), "adapters": [label[0]]})
    for i in range(4):
        rows.append({"id": "T2_ABA" + str(i), "phase": "T2", "ordinal": i, "adapters": ["A", "B"]})
    for i in range(4):
        rows.append({"id": "T3_state" + str(i), "phase": "T3", "ordinal": i, "adapters": []})
    for i in range(4):
        for obj in ("state_package", "adapter_A_weight", "adapter_B_weight"):
            rows.append({"id": "T4_" + obj + str(i), "phase": "T4", "ordinal": i, "object": obj, "adapters": []})
    return rows


def bound_json(spec: dict) -> dict:
    p = Path(spec["path"])
    if file_sha(p) != spec["sha256"]:
        raise ValueError("bound input SHA mismatch: " + str(p))
    return read_json(p)


def validate_plan(plan: dict, *, synthetic: bool = False) -> None:
    expected_contract = SYNTHETIC_CONTRACT if synthetic else REAL_CONTRACT
    if plan.get("contract") != expected_contract:
        raise ValueError("execution/synthetic contract mismatch")
    if plan.get("clock_contract")!=CLOCK_CONTRACT:
        raise ValueError("fixed cross-process kernel clock contract")
    clock_record=bound_json(plan["clock_qualification"])
    if clock_record.get("status")!="CLOCK_NAMESPACE_QUALIFIED" or clock_record.get("clock_contract")!=CLOCK_CONTRACT or len(clock_record.get("probes",[]))!=2:
        raise ValueError("unqualified clock namespace")
    for probe in clock_record["probes"]:
        if probe["exit_code"]!=0 or not probe["common_epoch_containment"] or not probe["parent_begin_ns"]<=probe["native"]["begin_ns"]<=probe["native"]["end_ns"]<=probe["parent_end_ns"]:
            raise ValueError("clock raw bracket failed")
    if plan.get("generation") != GENERATION:
        raise ValueError("full generation contract mismatch")
    runtime = plan.get("runtime", {})
    for key, val in {"device": "cpu", "dtype": "float32", "batch_size": 1, "intra_threads": 1, "inter_threads": 1, "local_files_only": True, "trust_remote_code": False, "automatic_retry": False}.items():
        if runtime.get(key) != val:
            raise ValueError("fixed runtime mismatch: " + key)
    if runtime.get("versions") != VERSIONS:
        raise ValueError("version contract mismatch")
    if plan.get("matrix") != matrix():
        raise ValueError("exact attempt matrix/order mismatch")
    if set(plan.get("hardware_expected", {})) != set(IDENTITY_FIELDS):
        raise ValueError("incomplete hardware identity freeze")
    if any(plan["hardware_expected"][x] in (None, "", "UNVERIFIED") for x in IDENTITY_FIELDS):
        raise ValueError("unverified hardware identity")
    if plan["hardware_expected"]["AC_power_state"] != "AC":
        raise ValueError("AC power required")
    if plan.get("source_image_access_allowed") is not False or plan.get("prefix_calls") != 0:
        raise ValueError("technical suffix only/no image")
    if not synthetic:
        if plan.get("utility_processes_cap")!=16:raise ValueError("explicit16 utility-process cap required")
        if plan["old_plan"]["sha256"] != OLD_PLAN_SHA or plan["old_reference"]["sha256"] != OLD_REFERENCE_SHA:
            raise ValueError("immutable old reference identity")
        if plan.get("caps") != HARD:
            raise ValueError("hard budget drift")
        if plan["prepared_proposal"]["sha256"] != PROPOSAL_SHA:
            raise ValueError("wrong prepared proposal")
        proposal = bound_json(plan["prepared_proposal"])
        if proposal["status"] != "PREPARATION_COMPLETE_EXECUTION_NOT_STARTED":
            raise ValueError("proposal status")
        expected = {
            "A": (154423432, "95103f5d6547c194770b03ee0ac24fe6fc125eb8890e992977be8dcbfea10f83"),
            "B": (9641944, "e899c9c13b53bf80bcfc86048f5678a727e8c1838dc824156b521d5cdaafdd2d"),
        }
        for key, pair in expected.items():
            resource = plan["resources"][key]
            if (resource["weight_bytes"], resource["weight_sha256"]) != pair:
                raise ValueError("adapter identity drift")
            files={x["name"]:x for x in resource["files"]}
            if len(files)!=len(resource["files"]) or resource["weight_file"]!="adapter_model.safetensors" or "adapter_config.json" not in files or (files[resource["weight_file"]]["bytes"],files[resource["weight_file"]]["sha256"])!=pair:
                raise ValueError("missing adapter weight/config verification")
            expected_config="d4e1d0e6c2c77424d4ed76956cb5642a2fb8e64f233b5a4874e5341e9e085216" if key=="A" else "f32a6b2b99479a4d680495593cebb13da0623abc88d099d58a7500a32c40b3a7"
            if files["adapter_config.json"]["sha256"]!=expected_config:
                raise ValueError("adapter config drift")
        if plan["resources"]["base"]["weight_sha256"] != "d05b567eeaf534e83d375551f068ed57b5f52d37c657197f644af5ef9db091a2":
            raise ValueError("base identity drift")
    reference = bound_json(plan["old_reference"])
    old_plan = bound_json(plan["old_plan"])
    if not synthetic:
        base=plan["resources"]["base"]
        files={x["name"]:x for x in base["files"]}
        if len(files)!=len(base["files"]) or base["weight_file"]!="model.safetensors" or files.get("model.safetensors",{}).get("bytes")!=1015025832 or files.get("model.safetensors",{}).get("sha256")!=base["weight_sha256"]:
            raise ValueError("base weight verification missing")
        for filename,key in {"processor_config.json":"config_sha256","preprocessor_config.json":"preprocessor_sha256","tokenizer.json":"tokenizer_sha256","chat_template.json":"chat_template_sha256"}.items():
            if files.get(filename,{}).get("sha256")!=old_plan["identity"]["processor"][key]:
                raise ValueError("processor/tokenizer identity drift")
        for resource in plan["resources"].values():
            if any(Path(x["name"]).name!=x["name"] for x in resource["files"]):
                raise ValueError("resource file path escape")
    if old_plan["generation"]["n1"] != GENERATION or reference["runtime"] != plan["old_runtime_identity"]:
        raise ValueError("old full generation/library identity mismatch")
    if reference["n1_input"] != plan["reference_input"] or reference["n1_output"]["token_ids"] != plan["reference_A_tokens"]:
        raise ValueError("old prompt/token reference mismatch")
    if any(reference["runtime"][x] != VERSIONS[x] for x in VERSIONS):
        raise ValueError("old runtime library identity mismatch")
    if len(plan["reference_A_tokens"]) > GENERATION["max_new_tokens"]:
        raise ValueError("reference token limit")
    # A saved envelope is bound by bytes AND semantics. Rehashing an invalid state cannot admit it.
    state = bound_json(plan["state_file"])
    validate_state(state, plan["state_identity"])
    state_manifest=bound_json(plan["state_manifest"])
    state_bytes=Path(plan["state_file"]["path"]).read_bytes();manifest_bytes=Path(plan["state_manifest"]["path"]).read_bytes()
    if state_manifest.get("contains_final_output") is not False or state_manifest.get("contains_model_weights") is not False or state_manifest.get("allowed_files")!=["state.json","manifest.json"] or state_manifest.get("state",{}).get("sha256")!=sha(state_bytes) or state_manifest["state"]["bytes"]!=len(state_bytes):
        raise ValueError("bound state manifest contract")
    expected_objects={"state_package":{"bytes":len(state_bytes)+len(manifest_bytes),"sha256":sha(state_bytes+manifest_bytes)},"adapter_A_weight":{"bytes":plan["resources"]["A"]["weight_bytes"],"sha256":plan["resources"]["A"]["weight_sha256"]},"adapter_B_weight":{"bytes":plan["resources"]["B"]["weight_bytes"],"sha256":plan["resources"]["B"]["weight_sha256"]}}
    if plan["transfer_objects"]!=expected_objects:
        raise ValueError("transfer object identity/byte projection")
    if not synthetic:
        if len(state_bytes)!=1703 or len(manifest_bytes)!=337 or plan["state_file"]["sha256"]!="53c4f94fa8b50025a2ae4e532d31e211996e49eb302859f37561ef93e66409c9" or plan["state_manifest"]["sha256"]!="362123e41c78f9fe53cd7b36faf8337d18133caf3e9040175d4e1f520dd14839":
            raise ValueError("immutable old state package")
        if reference["identity"]!=plan["state_identity"] or state["payload"]["generation"]!=old_plan["generation"]:
            raise ValueError("old full state/model/generation identity")
        required_sources={"scripts/run_local_adapter_cost_support.py","scripts/audit_local_adapter_cost_support.py"}
        names=[x["path"] for x in plan["source_files"]]
        if not required_sources<=set(names) or len(names)!=len(set(names)) or any(Path(x).is_absolute() or ".." in Path(x).parts for x in names):
            raise ValueError("source freeze file identity/escape")
    prompt = old_plan["workflow"]["n1_prompt_template"].replace("{n0_output}", state["payload"]["n0_output"]["raw_text"])
    if prompt != plan["reference_input"]["prompt"]:
        raise ValueError("state does not rebuild exact old suffix prompt")
    if state["payload"]["generation"]["n1"] != GENERATION:
        raise ValueError("state generation mismatch")


def validate_state(state: dict, identity: dict) -> None:
    p = state.get("payload", {})
    expected = {"workflow_id": "technical_vision_description_suffix_v1", "nodes": ["n0", "n1"], "edges": [["n0", "n1"]], "completed_nodes": ["n0"], "remaining_nodes": ["n1"], "next_node": "n1", "identity": identity}
    if state.get("schema_version") != "ppo_mec.two_node_workflow_state.v1" or state.get("payload_sha256") != sha(canonical(p)):
        raise ValueError("state envelope/hash")
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError("state identity or node boundary")
    if any(x in p for x in ("final_output", "n1_output", "answer")):
        raise ValueError("state injected suffix answer")
    n0 = p.get("n0_output", {})
    if not n0.get("raw_text", "").strip() or not n0.get("token_ids") or p.get("suffix_input_material") != n0["raw_text"]:
        raise ValueError("state upstream material")
    for key in ("rng_state", "tensor_state"):
        if p.get(key, {}).get("required") is not False or not p[key].get("reason"):
            raise ValueError("undeclared continuation state")


def hardware_snapshot() -> dict:
    if platform.system() != "Darwin":
        raise RuntimeError("frozen local Mac CPU envelope only; no platform fallback")
    def command(*args: str) -> str:
        return subprocess.check_output(args, text=True, timeout=10).strip()
    disk = plistlib.loads(subprocess.check_output(["diskutil", "info", "-plist", "/"], timeout=10))
    battery = command("pmset", "-g", "batt")
    snapshot=command("ps","-axo","pid=,ppid=,pcpu=,command=")
    rows=[]
    for line in snapshot.splitlines():
        fields=line.strip().split(None,3)
        if len(fields)!=4:
            raise ValueError("unparseable process inventory row")
        rows.append({"pid":int(fields[0]),"ppid":int(fields[1]),"cpu_percent":float(fields[2]),"command":fields[3]})
    return {"host_model": command("sysctl", "-n", "hw.model"), "CPU_model": command("sysctl", "-n", "machdep.cpu.brand_string"), "physical_cores": int(command("sysctl", "-n", "hw.physicalcpu")), "logical_cores": int(command("sysctl", "-n", "hw.logicalcpu")), "RAM_bytes": int(command("sysctl", "-n", "hw.memsize")), "OS_build": command("sw_vers", "-buildVersion"), "architecture": platform.machine(), "disk_model_and_filesystem": {"device": disk.get("DeviceIdentifier"), "media": disk.get("MediaName"), "filesystem": disk.get("FilesystemType"), "block_bytes": disk.get("DeviceBlockSize")}, "AC_power_state": "AC" if "AC Power" in battery else "BATTERY", "power_mode": command("pmset", "-g", "custom"), "load_average": list(os.getloadavg()), "thermal_status": command("pmset", "-g", "therm"), "free_disk_bytes": __import__("shutil").disk_usage(ROOT).free, "background_process_snapshot":snapshot,"background_process_rows":rows}


def active_scientific_python_jobs(rows: list[dict], excluded_pids: set[int]) -> list[dict]:
    """Inspect Python's program token, not substrings inside shell/code arguments."""
    found=[]
    for row in rows:
        if row["pid"] in excluded_pids:
            continue
        raw=row["command"]
        executable=Path(raw.split(None,1)[0]).name.lower()
        if not executable.startswith("python"):
            continue
        args=shlex.split(raw)
        program=None;i=1
        while i<len(args):
            token=args[i]
            if token=="-c":
                # An unknown Python -c workload is not certified idle.
                program="UNVERIFIED_PYTHON_C";break
            if token=="-m":
                if i+1>=len(args):raise ValueError("incomplete Python module command")
                program=args[i+1].rsplit(".",1)[-1]+".py";break
            if token in ("-W","-X"):
                i+=2;continue
            if token.startswith("-"):
                i+=1;continue
            program=Path(token).name;break
        if program=="UNVERIFIED_PYTHON_C" or (program and (program.startswith(("launch_stream_","run_stream_","train_predictor")) or program in ("benchmark_main_results.py","run_two_node_workflow_suffix_recovery.py","calibrate_workflow_state_recovery.py") or (program=="run_local_adapter_cost_support.py" and "--child" in args))):
            found.append({**row,"python_program":program,"argv":args})
    return found


def preflight(plan: dict, grant: dict, plan_sha: str) -> dict:
    if grant.get("contract") != "local_adapter_cost_execution_grant_v1" or grant.get("plan_sha256") != plan_sha or grant.get("source_commit") != plan["source_commit"] or grant.get("execute_T") is not True or grant.get("execute_Q") is not False:
        raise ValueError("new exact execution grant required")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,timeout=10).strip() != plan["source_commit"]:
        raise ValueError("source commit mismatch")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True,timeout=10).strip():
        raise ValueError("tracked source is not clean")
    for row in plan["source_files"]:
        if file_sha(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("source file drift")
    versions = {"python": platform.python_version(), **{k: importlib.metadata.version(k) for k in VERSIONS if k != "python"}}
    if versions != VERSIONS:
        raise ValueError("installed package versions mismatch")
    actual = hardware_snapshot()
    if any(actual[k] != plan["hardware_expected"][k] for k in IDENTITY_FIELDS):
        raise ValueError("hardware/power freeze mismatch")
    if actual["free_disk_bytes"] < HARD["disk_bytes"]:
        raise ValueError("insufficient disk")
    jobs=active_scientific_python_jobs(actual["background_process_rows"],{os.getpid()})
    if jobs:
        raise ValueError("concurrent scientific/model workload")
    actual["active_scientific_jobs"] = jobs
    return {"hardware": actual, "versions": versions, "source_commit": plan["source_commit"],"clock_contract":CLOCK_CONTRACT,"clock_qualification_sha256":plan["clock_qualification"]["sha256"]}


class Guard:
    def __init__(self, root: Path, caps: dict, *, external_paths: tuple[Path, ...] = (), begin_ns: int | None = None):
        self.root, self.caps = root, caps
        self.external_paths=external_paths
        self.start_ns = clock_ns() if begin_ns is None else begin_ns

    def check(self) -> None:
        if (clock_ns() - self.start_ns) / 1e9 > self.caps["wall_seconds"]:
            raise TimeoutError("total wall deadline")
        external=sum(p.stat().st_size for p in self.external_paths if p.exists() and p.is_file())
        if disk_bytes(self.root)+external > self.caps["disk_bytes"]-min(1048576,self.caps["disk_bytes"]//4):
            raise RuntimeError("total retained disk cap")
        logs = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file() and p.suffix in (".log", ".jsonl"))
        if logs > self.caps["log_bytes"]:
            raise RuntimeError("log cap")


class NativeLedger:
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("xb")
        self.serial = 0

    def row(self, value: dict) -> None:
        self.handle.write(canonical(value) + b"\n")
        self.handle.flush()

    @contextlib.contextmanager
    def span(self, kind: str, *, parent: str | None = None, **detail: Any):
        self.serial += 1
        key = "e" + str(self.serial)
        self.row({"edge": "start", "id": key, "kind": kind, "parent": parent, "ns": clock_ns(), **detail})
        try:
            yield key
        except BaseException as exc:
            self.row({"edge": "end", "id": key, "ns": clock_ns(), "status": "FAIL", "error_type": type(exc).__name__})
            raise
        else:
            self.row({"edge": "end", "id": key, "ns": clock_ns(), "status": "PASS"})

    def close(self) -> None:
        self.handle.close()


def verify_resource_files(plan: dict, check: Any = None) -> list[dict]:
    rows = []
    for name, resource in plan["resources"].items():
        for spec in resource["files"]:
            p = Path(resource["path"]) / spec["name"]
            if p.is_symlink() or p.stat().st_size != spec["bytes"] or file_sha(p,check) != spec["sha256"]:
                raise ValueError("resource byte/hash mismatch")
            rows.append({"resource": name, **spec})
    return rows


def model_child(plan: dict, row: dict, directory: Path) -> None:
    """Only reached in the granted real subprocess, never in qualification."""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import PeftModel
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ledger = NativeLedger(directory / "events.jsonl")
    outputs = []
    try:
        with ledger.span("processor_load"):
            processor = AutoProcessor.from_pretrained(plan["resources"]["base"]["path"], local_files_only=True, trust_remote_code=False)
        with ledger.span("base_load"):
            base = AutoModelForVision2Seq.from_pretrained(plan["resources"]["base"]["path"], torch_dtype=torch.float32, local_files_only=True, trust_remote_code=False)
            base.to("cpu").eval()
        first = row["adapters"][0]
        with ledger.span("adapter_load", adapter=first):
            model = PeftModel.from_pretrained(base, plan["resources"][first]["path"], adapter_name=first, is_trainable=False, local_files_only=True)
            model.eval()
        if row["phase"] == "T2":
            with ledger.span("adapter_load", adapter="B"):
                model.load_adapter(plan["resources"]["B"]["path"], adapter_name="B", is_trainable=False, local_files_only=True)
        with ledger.span("input_rebuild"):
            prompt = plan["reference_input"]["prompt"]
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            rendered = processor.apply_chat_template(messages, add_generation_prompt=True)
            tensors = dict(processor(text=rendered, return_tensors="pt"))
            inp = {"prompt": prompt, "rendered_prompt": rendered, "input_ids": tensors["input_ids"][0].tolist(), "source_image_access_count": 0}
            for key in ("prompt", "rendered_prompt", "input_ids"):
                if inp[key] != plan["reference_input"][key]:
                    raise ValueError("actual suffix input differs from frozen old reference")
            create_json(directory / "input.json", inp)
        sequence = [first] if row["phase"] == "T1" else ["A", "B", "A"]
        for active in sequence:
            with ledger.span("activate", adapter=active):
                model.set_adapter(active)
                raw_status = model.get_model_status()
                status = dict(vars(raw_status))
                if status.get("active_adapters") != [active] or active not in status.get("available_adapters", []):
                    raise ValueError("actual adapter status")
                params = []
                for key, tensor in model.named_parameters():
                    if "." + active + "." in key:
                        data = tensor.detach().cpu().contiguous()
                        params.append({"key": key, "shape": list(data.shape), "dtype": str(data.dtype), "nonzero": int(torch.count_nonzero(data).item()), "sha256": sha(data.numpy().tobytes())})
                if not params or not any(p["nonzero"] for p in params):
                    raise ValueError("adapter tensors missing/zero")
                status = json.loads(json.dumps(status, default=str))
            for repeat in range(2 if row["phase"] == "T1" else 1):
                with ledger.span("generate", adapter=active, repeat=repeat) as event_id:
                    with ledger.span("generate_compute", parent=event_id):
                        with torch.inference_mode():
                            out = model.generate(**tensors, **plan["generation"])
                    with ledger.span("output_copy_decode", parent=event_id):
                        tokens = out[:, int(tensors["input_ids"].shape[-1]):].cpu()
                        ids = [int(x) for x in tokens[0].tolist()]
                        decoded = processor.batch_decode(tokens, skip_special_tokens=True)[0]
                record = {"adapter": active, "repeat": repeat, "token_ids": ids, "decoded_text": decoded, "status": status, "loaded_parameters": params, "adapter_weight_sha256": plan["resources"][active]["weight_sha256"], "generation": plan["generation"]}
                outputs.append(record)
                create_json(directory / ("output_%02d.json" % (len(outputs)-1)), record)
        import resource
        peak_rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        create_json(directory / "child_terminal.json", {"status": "PASS", "pid": os.getpid(), "outputs": len(outputs), "versions": VERSIONS, "module_origins": {"torch": torch.__file__, "transformers": sys.modules["transformers"].__file__, "peft": sys.modules["peft"].__file__}, "threads": [torch.get_num_threads(), torch.get_num_interop_threads()], "device": "cpu", "dtype": "float32","memory":{"peak_rss_bytes":peak_rss,"source":"macOS getrusage ru_maxrss bytes","current_rss_bytes":None,"GPU_memory_bytes":None,"observations":1}})
    except BaseException as exc:
        create_json(directory / "child_terminal.json", {"status": "FAIL", "pid": os.getpid(), "error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        ledger.close()


def state_trial(plan: dict, directory: Path) -> None:
    ledger = NativeLedger(directory / "events.jsonl")
    try:
        state = read_json(Path(plan["state_file"]["path"]))
        with ledger.span("encode"):
            state_bytes = json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True).encode() + b"\n"
            manifest = {"schema_version": "ppo_mec.workflow_state_package.v1", "allowed_files": ["state.json", "manifest.json"], "state": {"path": "state.json", "bytes": len(state_bytes), "sha256": sha(state_bytes)}, "contains_model_weights": False, "contains_final_output": False}
            manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
        with ledger.span("durable_save") as parent:
            package = directory / "package"
            package.mkdir()
            for name, data in (("state.json", state_bytes), ("manifest.json", manifest_bytes)):
                part = package / (name + ".partial")
                with part.open("xb") as f:
                    with ledger.span("file_write",parent=parent,filename=name,bytes=len(data)):
                        f.write(data); f.flush()
                    with ledger.span("file_fsync",parent=parent,filename=name):
                        os.fsync(f.fileno())
                with ledger.span("file_rename",parent=parent,filename=name):
                    part.rename(package / name)
                fd = os.open(package, os.O_RDONLY)
                try:
                    with ledger.span("directory_fsync",parent=parent,filename=name):
                        os.fsync(fd)
                finally:
                    os.close(fd)
        with ledger.span("restore"):
            restored = read_json(package / "state.json")
            validate_state(restored, plan["state_identity"])
            if file_sha(package / "state.json") != plan["state_file"]["sha256"] or file_sha(package / "manifest.json") != plan["state_manifest"]["sha256"]:
                raise ValueError("state package bytes differ from old reference")
        with ledger.span("prompt_rebuild"):
            old = bound_json(plan["old_plan"])
            prompt = old["workflow"]["n1_prompt_template"].replace("{n0_output}", restored["payload"]["n0_output"]["raw_text"])
            if prompt != plan["reference_input"]["prompt"]:
                raise ValueError("restored suffix prompt mismatch")
            create_json(directory / "restored_input.json", {"prompt": prompt, "identity": restored["payload"]["identity"], "source_image_access_count": 0})
    finally:
        ledger.close()


def receive_exact(sock: socket.socket, amount: int) -> bytes:
    chunks = []
    while amount:
        chunk = sock.recv(min(65536, amount))
        if not chunk:
            raise EOFError("early loopback EOF")
        chunks.append(chunk); amount -= len(chunk)
    return b"".join(chunks)


def loopback_trial(plan: dict, row: dict, directory: Path) -> None:
    obj = row["object"]
    if obj == "state_package":
        sources = [Path(plan["state_file"]["path"]), Path(plan["state_manifest"]["path"])]
    else:
        resource = plan["resources"]["A" if obj == "adapter_A_weight" else "B"]
        sources = [Path(resource["path"]) / resource["weight_file"]]
    expected = plan["transfer_objects"][obj]
    ledger = NativeLedger(directory / "events.jsonl")
    wire = NativeLedger(directory / "wire.jsonl")
    deadline = (clock_ns()/1e9) + 58  # reserve two seconds of the 60s attempt for teardown
    def check_deadline():
        if (clock_ns()/1e9) > deadline:
            raise TimeoutError("loopback attempt deadline")
        if disk_bytes(directory.parents[1]) > plan["caps"]["disk_bytes"]:
            raise RuntimeError("loopback retained disk cap")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.settimeout(10)
    listener.bind(("127.0.0.1", 0)); listener.listen(1)
    errors: queue.Queue = queue.Queue()
    observed = {"header_upper_bytes": 0, "payload_complete_bytes": 0, "payload_uncertain_upper_bytes": 0, "ack_received_bytes": 0, "receiver_bytes": 0, "receiver_recv_calls": 0, "sender_send_calls": 0}
    def receiver():
        try:
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(min(10,max(.001,deadline-(clock_ns()/1e9))))
                header = receive_exact(conn, 64)
                (directory / "header.bin").write_bytes(header)
                magic, size, digest, name = HEADER.unpack(header)
                if magic != MAGIC or size != expected["bytes"] or digest.hex() != expected["sha256"] or name.rstrip(b"\0").decode() != obj:
                    raise ValueError("loopback header identity")
                h = hashlib.sha256(); remaining = size
                with (directory / "payload.partial").open("xb") as f:
                    while remaining:
                        check_deadline()
                        conn.settimeout(min(10,max(.001,deadline-(clock_ns()/1e9))))
                        block = conn.recv(min(65536, remaining)); observed["receiver_recv_calls"] += 1
                        if not block:
                            raise EOFError("partial payload")
                        f.write(block); h.update(block); remaining -= len(block); observed["receiver_bytes"] += len(block)
                    f.flush(); os.fsync(f.fileno())
                if h.hexdigest() != expected["sha256"]:
                    raise ValueError("loopback body hash")
                ack = ACK.pack(ACK_MAGIC, size, h.digest())
                (directory / "ack.bin").write_bytes(ack)
                conn.sendall(ack)
                (directory / "payload.partial").rename(directory / "payload.bin")
        except BaseException as exc:
            errors.put(repr(exc))
    thread = threading.Thread(target=receiver, daemon=True)
    thread.start()
    try:
        with ledger.span("transfer", object=obj, requested_payload_bytes=expected["bytes"]):
            with socket.create_connection(listener.getsockname(), timeout=10) as client:
                client.settimeout(10)
                observed["header_upper_bytes"] = 64
                with wire.span("send_header", requested_bytes=64):
                    client.sendall(HEADER.pack(MAGIC, expected["bytes"], bytes.fromhex(expected["sha256"]), obj.encode().ljust(16, b"\0"))); observed["sender_send_calls"] += 1
                for path in sources:
                    with path.open("rb") as f:
                        for chunk in iter(lambda: f.read(65536), b""):
                            check_deadline()
                            client.settimeout(min(10,max(.001,deadline-(clock_ns()/1e9))))
                            observed["payload_uncertain_upper_bytes"] = len(chunk)
                            with wire.span("send_payload", requested_bytes=len(chunk), chunk_sha256=sha(chunk)):
                                client.sendall(chunk); observed["sender_send_calls"] += 1
                            observed["payload_complete_bytes"] += len(chunk)
                            observed["payload_uncertain_upper_bytes"] = 0
                client.settimeout(min(10,max(.001,deadline-(clock_ns()/1e9))))
                with wire.span("receive_ack", requested_bytes=48):
                    ack = receive_exact(client, 48); observed["ack_received_bytes"] = len(ack)
                if ACK.unpack(ack) != (ACK_MAGIC, expected["bytes"], bytes.fromhex(expected["sha256"])):
                    raise ValueError("loopback ACK")
            thread.join(timeout=max(.001,min(1,deadline-(clock_ns()/1e9))))
            if thread.is_alive() or not errors.empty():
                raise RuntimeError("loopback receiver: " + (errors.get() if not errors.empty() else "timeout"))
    finally:
        listener.close(); thread.join(timeout=1); ledger.close(); wire.close()
        observed["application_charge_upper_bytes"] = observed["header_upper_bytes"] + observed["payload_complete_bytes"] + observed["payload_uncertain_upper_bytes"] + max(observed["ack_received_bytes"], 48 if observed["payload_complete_bytes"] == expected["bytes"] else 0)
        create_json(directory / "transfer_native.json", observed)


class RealBackend:
    def __init__(self, plan_path: Path):
        self.plan_path = plan_path

    def __call__(self, plan: dict, row: dict, directory: Path, remaining: float):
        utility=row["phase"] in ("T3","T4")
        limit=10 if row["phase"]=="T3" else 60 if row["phase"]=="T4" else 300
        if remaining<=2:raise TimeoutError("insufficient remaining budget; no child launched")
        role="--helper" if utility else "--child"
        command=[sys.executable,str(Path(__file__).resolve()),"--plan",str(self.plan_path),role,row["id"],"--output",str(directory)]
        began=clock_ns()
        if utility:create_json(directory/"utility_launch.json",{"row":row["id"],"role":role,"begin_ns":began,"limit_seconds":min(limit,remaining),"command":command})
        child=None;killed=False;reaped=False
        try:
            with (directory/"stdout.log").open("xb") as out,(directory/"stderr.log").open("xb") as err:
                child=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"})
                deadline=began+int((min(limit,remaining)-2)*1e9)
                try:
                    while child.poll() is None:
                        root=directory.parents[1]
                        if clock_ns()>deadline or disk_bytes(root)>plan["caps"]["disk_bytes"] or sum(p.stat().st_size for p in root.rglob("*.log"))>plan["caps"]["log_bytes"]:
                            raise TimeoutError("owned child deadline/disk/log cap")
                        time.sleep(.02)
                    child.wait(timeout=1);reaped=True
                except BaseException:
                    import signal
                    killed=True
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait(timeout=1);reaped=True
                    raise
            if child.returncode:raise RuntimeError("owned child exited "+str(child.returncode))
        finally:
            if utility:create_json(directory/"utility_terminal.json",{"pid":child.pid if child else None,"exit_code":child.returncode if child else None,"launched":child is not None,"reaped":reaped if child else True,"kill_requested":killed,"begin_ns":began,"end_ns":clock_ns()})


def dispatch_worker(plan: dict, row: dict, directory: Path, role: str):
    if role=="model":
        if row["phase"] not in ("T1","T2"):raise ValueError("model child phase mismatch before model import")
        return model_child(plan,row,directory)
    if role!="helper" or row["phase"] not in ("T3","T4"):
        raise ValueError("utility helper phase mismatch before resource access")
    return state_trial(plan,directory) if row["phase"]=="T3" else loopback_trial(plan,row,directory)


def qualify_attempt(plan: dict, row: dict, directory: Path, references: dict) -> None:
    if row["phase"] in ("T1", "T2"):
        outputs = [read_json(p) for p in sorted(directory.glob("output_*.json"))]
        sequence = [row["adapters"][0]] * 2 if row["phase"] == "T1" else ["A", "B", "A"]
        if [x["adapter"] for x in outputs] != sequence:
            raise ValueError("raw output/activation matrix")
        for item in outputs:
            active = item["adapter"]
            if item["generation"] != GENERATION or item["status"].get("active_adapters") != [active] or active not in item["status"].get("available_adapters", []):
                raise ValueError("active adapter/generation mismatch")
            if item["adapter_weight_sha256"] != plan["resources"][active]["weight_sha256"] or not item["loaded_parameters"] or not any(x["nonzero"] > 0 for x in item["loaded_parameters"]):
                raise ValueError("loaded adapter provenance")
            if not isinstance(item["token_ids"], list) or not item["token_ids"] or len(item["token_ids"]) > 8:
                raise ValueError("raw tokens")
            expected = plan["reference_A_tokens"] if active == "A" else references.setdefault("B", item["token_ids"])
            if item["token_ids"] != expected:
                raise ValueError("suffix token fidelity failed")
    elif row["phase"] == "T3":
        validate_state(read_json(directory / "package/state.json"), plan["state_identity"])
    elif not (directory / "payload.bin").exists():
        raise ValueError("local transfer incomplete")


def inventory(root: Path) -> list[dict]:
    return [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": file_sha(p)} for p in sorted(root.rglob("*")) if p.is_file() and p.name not in ("inventory.json","execution_finish.json")]


def run(plan: dict, output: Path, backend: Any, *, synthetic: bool = False, preflight_record: dict | None = None, preflight_callback: Any = None, plan_bytes: bytes | None = None, producer_seconds: int | None = None, external_paths: tuple[Path,...] = (), begin_ns: int | None = None) -> dict:
    validate_plan(plan, synthetic=synthetic)
    output.mkdir(parents=True, exist_ok=False)
    active_caps=dict(plan["caps"])
    if producer_seconds is not None:
        active_caps["wall_seconds"]=producer_seconds
    guard = Guard(output, active_caps,external_paths=external_paths,begin_ns=begin_ns)
    with (output/"plan.json").open("xb") as f:
        f.write(plan_bytes if plan_bytes is not None else canonical(plan))
    plan_digest=file_sha(output/"plan.json")
    (output/"attempts").mkdir()
    completed = []; references = {}; failure = None;failure_stage="preflight";current_attempt=None
    try:
        try:
            snapshot=preflight_callback(guard) if preflight_callback else preflight_record
            if snapshot is None:
                raise ValueError("a real preflight or explicit synthetic snapshot is required")
            create_json(output/"preflight.json",{"status":"PASS",**snapshot})
            guard.check()
        except BaseException as exc:
            if not (output/"preflight.json").exists():
                create_json(output/"preflight.json",{"status":"FAIL","error":repr(exc),"synthetic_only":synthetic})
            raise
        for row in matrix():
            failure_stage="before_attempt";current_attempt=None
            guard.check()
            directory = output / "attempts" / row["id"]
            directory.mkdir(parents=True, exist_ok=False)
            started = clock_ns()
            create_json(directory / "started.json", {"row": row, "begin_ns": started, "plan_sha256": plan_digest})
            failure_stage="in_attempt";current_attempt=row["id"]
            try:
                remaining = guard.caps["wall_seconds"] - (started-guard.start_ns)/1e9
                backend(plan, row, directory, remaining)
                qualify_attempt(plan, row, directory, references)
                guard.check()
            except BaseException as exc:
                create_json(directory / "attempt_terminal.json", {"status": "FAIL", "begin_ns": started, "end_ns": clock_ns(), "error": repr(exc), "traceback": traceback.format_exc()})
                raise
            create_json(directory / "attempt_terminal.json", {"status": "PASS", "begin_ns": started, "end_ns": clock_ns()})
            completed.append(row["id"])
        failure_stage="finalization";current_attempt=None
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "error": repr(exc), "traceback": traceback.format_exc()}
    calls={k:0 for k in KINDS};completed_calls={k:0 for k in KINDS};body_bill=app_bill=0
    for p in output.glob("attempts/*/events.jsonl"):
        rows,_=ledger_rows(p)
        ends={r["id"]:r for r in rows if r["edge"]=="end"}
        for event in rows:
            if event["edge"]=="start" and event["kind"] in calls:
                calls[event["kind"]]+=1
                if ends.get(event["id"],{}).get("status")=="PASS":
                    completed_calls[event["kind"]]+=1
    for p in output.glob("attempts/T4_*"):
        body,app=transfer_spent(p);body_bill+=body;app_bill+=app
    utility_launches=len(list(output.glob("attempts/*/utility_launch.json")))
    terminal = {"status": "FAIL" if failure else "PASS", "completed": completed, "failure": failure,"failure_stage":failure_stage if failure else None,"current_attempt":current_attempt if failure else None, "begin_ns": guard.start_ns, "end_ns": clock_ns(), "synthetic_only": synthetic, "paper_goal_complete": False, "summary":{"raw_calls":calls,"raw_completed_calls":completed_calls,"payload_charge_upper_bytes":body_bill,"application_charge_upper_bytes":app_bill,"utility_process_launches":utility_launches},"activation_scope":"activation_with_integrity_verification; pure set_adapter latency unavailable"}
    create_json(output / "terminal_receipt.json", terminal)
    if any(not read_json(p)["reaped"] for p in output.glob("attempts/*/utility_terminal.json")):
        raise RuntimeError("unreaped utility child; raw inventory not sealed, manual intervention required")
    create_json(output / "inventory.json", {"files": inventory(output)})
    create_json(output / "execution_finish.json", {"begin_ns":guard.start_ns,"end_ns":clock_ns(),"inventory_sha256":file_sha(output/"inventory.json"),"terminal_sha256":file_sha(output/"terminal_receipt.json"),"inventory_hashing_included_in_wall":True})
    return terminal


def combined_log_bytes(root: Path, external: tuple[Path, ...]) -> int:
    """One byte bill for immutable raw logs and the external audit process logs."""
    return sum(p.stat().st_size for p in (*root.rglob("*"), *external)
               if p.exists() and p.is_file() and p.suffix in (".log", ".jsonl"))


def terminal_pipeline(plan: dict, output: Path, audit_output: Path, delivery_output: Path, backend: Any, *, synthetic: bool = False, preflight_record: dict | None = None, preflight_callback: Any = None, plan_bytes: bytes | None = None) -> dict:
    """Exactly one producer and one separate audit; no outer kill of the model parent.

    The producer's existing child-session cleanup receives only the 4800s
    allocation. The 600s audit subprocess runs after either producer terminal.
    """
    validate_plan(plan,synthetic=synthetic)
    limits={"producer_seconds":4800,"consumer_seconds":600,"combined_seconds":5400}
    if plan.get("terminal_pipeline")!=limits:
        raise ValueError("exact terminal reservation required")
    paths={"raw_root":str(output.resolve()),"audit":str(audit_output.resolve()),"delivery":str(delivery_output.resolve())}
    if plan.get("outputs")!=paths or len(set(paths.values()))!=3 or audit_output.resolve().is_relative_to(output.resolve()) or delivery_output.resolve().is_relative_to(output.resolve()):
        raise ValueError("frozen external terminal paths")
    claim=delivery_output.with_name(delivery_output.stem+".execution_claim.json")
    stdout=audit_output.with_name(audit_output.stem+".stdout.log")
    stderr=audit_output.with_name(audit_output.stem+".stderr.log")
    external=(claim,audit_output,delivery_output,stdout,stderr)
    if any(p.exists() for p in (output,*external)):
        raise FileExistsError("single-launch raw/audit/delivery/claim is create-only")
    if not all(p.parent.exists() for p in external):
        raise ValueError("frozen terminal parent directories must already exist")
    begin=clock_ns()
    plan_digest=sha(plan_bytes if plan_bytes is not None else canonical(plan))
    create_json(claim,{"plan_sha256":plan_digest,"source_commit":plan["source_commit"],"outputs":paths,"begin_ns":begin,"limits":limits,"synthetic_only":synthetic,"automatic_retry":False})
    produced=None;producer_error=None
    try:
        produced=run(plan,output,backend,synthetic=synthetic,preflight_record=preflight_record,preflight_callback=preflight_callback,plan_bytes=plan_bytes,producer_seconds=4800,external_paths=external,begin_ns=begin)
    except BaseException as exc:
        producer_error={"type":type(exc).__name__,"error":repr(exc),"traceback":traceback.format_exc()}
    audit_begin=clock_ns();audit_exception=None;audit_exit=None;audit_invocations=0
    inventory_before=file_sha(output/"inventory.json") if (output/"inventory.json").exists() else None
    try:
        remaining=5400-(audit_begin-begin)/1e9
        if remaining<=1:
            raise TimeoutError("no independent consumer budget remains")
        command=[sys.executable,str(ROOT/"scripts/audit_local_adapter_cost_support.py"),"--root",str(output),"--plan_sha256",plan_digest,"--output",str(audit_output)]
        if synthetic:
            command.append("--allow_synthetic")
        with stdout.open("xb") as out,stderr.open("xb") as err:
            audit_invocations=1
            audit_child=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
            deadline=(clock_ns()/1e9)+min(599,remaining-1)
            try:
                while audit_child.poll() is None:
                    retained=disk_bytes(output)+sum(p.stat().st_size for p in external if p.exists() and p.is_file())
                    logs=combined_log_bytes(output,external)
                    if (clock_ns()/1e9)>deadline or retained>plan["caps"]["disk_bytes"]-min(1048576,plan["caps"]["disk_bytes"]//4) or logs>plan["caps"]["log_bytes"]:
                        raise TimeoutError("independent consumer deadline/disk/log cap")
                    time.sleep(.02)
            except BaseException:
                import signal
                os.killpg(audit_child.pid,signal.SIGKILL);audit_child.wait(timeout=1)
                raise
            audit_exit=audit_child.returncode
    except BaseException as exc:
        audit_exception={"type":type(exc).__name__,"error":repr(exc)}
        if not audit_output.exists():
            create_json(audit_output,{"status":"AUDIT_NOT_COMPLETED_PRESERVED","error":audit_exception,"paper_goal_complete":False,"model_calls":0})
    audit_end=clock_ns()
    audit_record=None
    if audit_output.exists():
        try:
            audit_record=read_json(audit_output)
        except (ValueError,OSError) as exc:
            audit_exception={"type":type(exc).__name__,"error":repr(exc)}
    inventory_after=file_sha(output/"inventory.json") if (output/"inventory.json").exists() else None
    expected_status="SOFTWARE_SYNTHETIC_ONLY" if synthetic else "AUDITED_PASS_TECHNICAL_COMPONENT_COST_ONLY"
    passed=bool(produced and produced["status"]=="PASS" and producer_error is None and audit_invocations==1 and audit_exit==0 and audit_exception is None and audit_record and audit_record.get("status")==expected_status and audit_record.get("native_terminal")=="PASS" and inventory_before==inventory_after)
    retained_without_delivery=disk_bytes(output)+sum(p.stat().st_size for p in external if p.exists() and p.is_file())
    elapsed=(audit_end-begin)/1e9
    combined_logs=combined_log_bytes(output,external)
    if combined_logs>plan["caps"]["log_bytes"]:
        passed=False
    if elapsed>5400 or (audit_end-audit_begin)/1e9>600:
        passed=False
    result={"status":"PASS" if passed else "FAIL","classification":"software_pipeline_only" if synthetic else "technical_component_cost_only","synthetic_only":synthetic,"plan_sha256":plan_digest,"outputs":paths,"producer_status":produced["status"] if produced else "EXCEPTION","producer_exception":producer_error,"audit_invocations":audit_invocations,"audit_exit_code":audit_exit,"audit_status":audit_record.get("status") if audit_record else "UNREADABLE","audit_exception":audit_exception,"audit_sha256":file_sha(audit_output) if audit_output.exists() else None,"raw_inventory_sha256_before":inventory_before,"raw_inventory_sha256_after":inventory_after,"begin_ns":begin,"audit_begin_ns":audit_begin,"audit_end_ns":audit_end,"combined_seconds":elapsed,"consumer_seconds":(audit_end-audit_begin)/1e9,"limits":limits,"combined_retained_bytes":0,"task_quality":"UNAVAILABLE","wireless_RSU":"NOT_MEASURED","paper_goal_complete":False}
    result["combined_log_bytes"]=combined_logs
    result["log_cap_exceeded"]=combined_logs>plan["caps"]["log_bytes"]
    # Fixed point includes this external create-only delivery itself in the byte bill.
    for _ in range(8):
        encoded=json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2).encode()+b"\n"
        count=retained_without_delivery+len(encoded)
        if result["combined_retained_bytes"]==count:
            break
        result["combined_retained_bytes"]=count
    if result["combined_retained_bytes"]>plan["caps"]["disk_bytes"]:
        result["status"]="FAIL";result["disk_cap_exceeded"]=True
        for _ in range(8):
            count=retained_without_delivery+len(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2).encode())+1
            if count==result["combined_retained_bytes"]:break
            result["combined_retained_bytes"]=count
    create_json(delivery_output,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan_sha256")
    parser.add_argument("--grant", type=Path)
    parser.add_argument("--audit_output",type=Path)
    parser.add_argument("--delivery_output",type=Path)
    parser.add_argument("--child")
    parser.add_argument("--helper")
    args = parser.parse_args()
    plan = read_json(args.plan)
    validate_plan(plan)
    if args.child and args.helper:raise ValueError("one worker role only")
    if args.child or args.helper:
        # Child inherits a signed launch context, not a stand-alone public model entry.
        if os.environ.get("PPO_LOCAL_COST_GRANTED_PLAN_SHA") != file_sha(args.plan):
            raise ValueError("child has no granted parent context")
        row = next(x for x in matrix() if x["id"] == (args.child or args.helper))
        dispatch_worker(plan,row,args.output,"model" if args.child else "helper")
        return
    if not args.grant or args.plan_sha256 != file_sha(args.plan):
        raise ValueError("explicit plan SHA and separate grant required")
    os.environ["PPO_LOCAL_COST_GRANTED_PLAN_SHA"] = args.plan_sha256
    def checked_preflight(guard):
        guard.check()
        snapshot=preflight(plan,read_json(args.grant),args.plan_sha256)
        snapshot["verified_resource_files"]=verify_resource_files(plan,guard.check)
        guard.check()
        return snapshot
    if not args.audit_output or not args.delivery_output:
        raise ValueError("mandatory external audit and delivery outputs required")
    result = terminal_pipeline(plan,args.output,args.audit_output,args.delivery_output,RealBackend(args.plan),preflight_callback=checked_preflight,plan_bytes=args.plan.read_bytes())
    print(json.dumps({"status": result["status"], "audit_invocations": result["audit_invocations"]}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
