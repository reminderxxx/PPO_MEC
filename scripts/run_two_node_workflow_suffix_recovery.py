from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.workflow_suffix_recovery import (
    EXPECTED_EDGES,
    EXPECTED_NODES,
    EXPECTED_WORKFLOW_ID,
    StateValidationError,
    atomic_write,
    build_n1_prompt,
    canonical_json_bytes,
    input_record,
    loaded_adapter_names,
    read_state_package,
    seal_state_payload,
    serialize_state_envelope,
    sha256_bytes,
    sha256_file,
    validate_state_envelope,
    write_state_package,
)


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(jsonable(value), indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def combined_processor_sha256(plan: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(plan["identity"]["processor"]))


def state_identity(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_weight_sha256": plan["identity"]["base"]["weight_sha256"],
        "adapter_name": plan["identity"]["adapter"]["name"],
        "adapter_weight_sha256": plan["identity"]["adapter"]["weight_sha256"],
        "processor_sha256": combined_processor_sha256(plan),
    }


def verify_static_resources(plan: dict[str, Any], *, include_source_image: bool) -> dict[str, Any]:
    base = Path(plan["identity"]["base"]["path"])
    adapter = Path(plan["identity"]["adapter"]["path"])
    checked = {
        "base_weight_sha256": sha256_file(base / "model.safetensors"),
        "adapter_weight_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "processor_config_sha256": sha256_file(base / "processor_config.json"),
        "preprocessor_config_sha256": sha256_file(base / "preprocessor_config.json"),
        "tokenizer_sha256": sha256_file(base / "tokenizer.json"),
        "chat_template_sha256": sha256_file(base / "chat_template.json"),
    }
    expected = {
        "base_weight_sha256": plan["identity"]["base"]["weight_sha256"],
        "adapter_weight_sha256": plan["identity"]["adapter"]["weight_sha256"],
        "processor_config_sha256": plan["identity"]["processor"]["config_sha256"],
        "preprocessor_config_sha256": plan["identity"]["processor"]["preprocessor_sha256"],
        "tokenizer_sha256": plan["identity"]["processor"]["tokenizer_sha256"],
        "chat_template_sha256": plan["identity"]["processor"]["chat_template_sha256"],
    }
    if checked != expected:
        raise RuntimeError(f"static resource hash drift: expected={expected}, actual={checked}")
    if include_source_image:
        image_hash = sha256_file(Path(plan["source_input_provenance"]["path"]))
        if image_hash != plan["source_input_provenance"]["sha256"]:
            raise RuntimeError("source image hash drift")
        checked["source_image_sha256"] = image_hash
    return checked


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def runtime_metadata() -> dict[str, Any]:
    import accelerate
    import peft
    import psutil
    import safetensors
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "safetensors": safetensors.__version__,
        "psutil": psutil.__version__,
        "module_origins": {
            "torch": torch.__file__,
            "transformers": transformers.__file__,
            "peft": peft.__file__,
        },
    }


def load_runtime(plan: dict[str, Any]) -> tuple[Any, Any, dict[str, float], dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq, AutoProcessor

    base_path = plan["identity"]["base"]["path"]
    adapter_path = plan["identity"]["adapter"]["path"]
    adapter_name = plan["identity"]["adapter"]["name"]
    timings: dict[str, float] = {}
    mark = time.monotonic()
    processor = AutoProcessor.from_pretrained(base_path, local_files_only=True, trust_remote_code=False)
    timings["processor_load_seconds"] = time.monotonic() - mark
    mark = time.monotonic()
    base = AutoModelForVision2Seq.from_pretrained(
        base_path,
        torch_dtype=torch.float32,
        local_files_only=True,
        trust_remote_code=False,
    )
    base.to("cpu")
    base.eval()
    timings["base_load_seconds"] = time.monotonic() - mark
    mark = time.monotonic()
    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        adapter_name=adapter_name,
        is_trainable=False,
        local_files_only=True,
    )
    timings["adapter_load_seconds"] = time.monotonic() - mark
    model.eval()
    model.set_adapter(adapter_name)
    status = jsonable(model.get_model_status())
    if adapter_name not in loaded_adapter_names(status):
        raise RuntimeError("expected adapter is not available after load")
    if status.get("active_adapters") != [adapter_name]:
        raise RuntimeError("expected adapter is not active after load")
    return processor, model, timings, status


def prepare_n0(plan: dict[str, Any], processor: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    from PIL import Image

    image = Image.open(plan["source_input_provenance"]["path"]).convert("RGB")
    prompt = plan["workflow"]["n0_prompt"]
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    rendered = processor.apply_chat_template(messages, add_generation_prompt=True)
    tensors = dict(processor(text=rendered, images=[image], return_tensors="pt"))
    return tensors, {
        "prompt": prompt,
        "rendered_prompt": rendered,
        "input_ids_sha256": sha256_bytes(canonical_json_bytes(tensors["input_ids"][0].tolist())),
        "image_size": list(image.size),
        "image_access_count": 1,
    }


def prepare_n1(plan: dict[str, Any], processor: Any, n0_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = build_n1_prompt(plan["workflow"]["n1_prompt_template"], n0_text)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    rendered = processor.apply_chat_template(messages, add_generation_prompt=True)
    tensors = dict(processor(text=rendered, return_tensors="pt"))
    record = input_record(prompt=prompt, rendered_prompt=rendered, input_ids=tensors["input_ids"][0].tolist())
    record["contains_n0_output"] = n0_text in prompt and n0_text in rendered
    record["source_image_access_count"] = 0
    return tensors, record


def generate(model: Any, processor: Any, tensors: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    import torch

    input_length = int(tensors["input_ids"].shape[-1])
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**tensors, **config)
    elapsed = time.monotonic() - started
    tokens = output[:, input_length:]
    token_ids = [int(item) for item in tokens[0].tolist()]
    return {
        "token_ids": token_ids,
        "decoded_text": processor.batch_decode(tokens, skip_special_tokens=True)[0],
        "new_token_count": len(token_ids),
        "inference_seconds": elapsed,
    }


def make_state_payload(plan: dict[str, Any], n0: dict[str, Any]) -> dict[str, Any]:
    return {
        "workflow_id": EXPECTED_WORKFLOW_ID,
        "nodes": EXPECTED_NODES,
        "edges": EXPECTED_EDGES,
        "completed_nodes": ["n0"],
        "remaining_nodes": ["n1"],
        "next_node": "n1",
        "n0_output": {"raw_text": n0["decoded_text"], "token_ids": n0["token_ids"]},
        "suffix_input_material": n0["decoded_text"],
        "generation": plan["generation"],
        "identity": state_identity(plan),
        "rng_state": {
            "required": False,
            "reason": "do_sample=false for both nodes; n1 starts from explicit text rather than RNG continuation",
        },
        "tensor_state": {
            "required": False,
            "reason": "the application-node boundary passes explicit text; no KV cache or tensor is consumed by n1",
        },
    }


def base_receipt(role: str, started_wall: float, started: float, plan_path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "PASS",
        "role": role,
        "pid": os.getpid(),
        "started_at_unix": started_wall,
        "process_elapsed_seconds": time.monotonic() - started,
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "runtime": runtime_metadata(),
        "identity": state_identity(plan),
        "task_correctness": "unavailable",
        "task_correctness_reason": "technical input has no applicable ALPR/Helmet ground-truth label",
    }


def execute_continuous(plan_path: Path, run_root: Path) -> dict[str, Any]:
    started_wall, started = time.time(), time.monotonic()
    plan = read_json(plan_path)
    checked = verify_static_resources(plan, include_source_image=True)
    processor, model, timings, status = load_runtime(plan)
    n0_inputs, n0_input = prepare_n0(plan, processor)
    n0 = generate(model, processor, n0_inputs, plan["generation"]["n0"])
    n1_inputs, n1_input = prepare_n1(plan, processor, n0["decoded_text"])
    n1 = generate(model, processor, n1_inputs, plan["generation"]["n1"])
    receipt = base_receipt("continuous", started_wall, started, plan_path, plan)
    receipt.update({
        "process_elapsed_seconds": time.monotonic() - started,
        "static_resource_hashes": checked,
        "model_load_timings": timings,
        "adapter_status": status,
        "executed_nodes": ["n0", "n1"],
        "node_call_counts": {"n0": 1, "n1": 1},
        "generate_attempted": 2,
        "generate_completed": 2,
        "n0_input": n0_input,
        "n0_output": n0,
        "n1_input": n1_input,
        "n1_output": n1,
    })
    return receipt


def execute_source(plan_path: Path, run_root: Path) -> dict[str, Any]:
    started_wall, started = time.time(), time.monotonic()
    plan = read_json(plan_path)
    checked = verify_static_resources(plan, include_source_image=True)
    processor, model, timings, status = load_runtime(plan)
    n0_inputs, n0_input = prepare_n0(plan, processor)
    n0 = generate(model, processor, n0_inputs, plan["generation"]["n0"])
    mark = time.monotonic()
    envelope = seal_state_payload(make_state_payload(plan, n0))
    serialize_seconds = time.monotonic() - mark
    serialized_state_bytes = len(serialize_state_envelope(envelope))
    mark = time.monotonic()
    package = write_state_package(run_root / "state_package", envelope)
    save_seconds = time.monotonic() - mark
    receipt = base_receipt("source", started_wall, started, plan_path, plan)
    receipt.update({
        "process_elapsed_seconds": time.monotonic() - started,
        "static_resource_hashes": checked,
        "model_load_timings": timings,
        "adapter_status": status,
        "executed_nodes": ["n0"],
        "node_call_counts": {"n0": 1, "n1": 0},
        "generate_attempted": 1,
        "generate_completed": 1,
        "n0_input": n0_input,
        "n0_output": n0,
        "state": {
            "serialize_seconds": serialize_seconds,
            "durable_save_seconds": save_seconds,
            "serialized_state_bytes": serialized_state_bytes,
            **package,
        },
    })
    return receipt


def execute_target(plan_path: Path, run_root: Path) -> dict[str, Any]:
    started_wall, started = time.time(), time.monotonic()
    plan = read_json(plan_path)
    mark = time.monotonic()
    checked = verify_static_resources(plan, include_source_image=False)
    static_validation_seconds = time.monotonic() - mark
    mark = time.monotonic()
    payload, package = read_state_package(run_root / "state_package", expected_identity=state_identity(plan))
    restore_validation_seconds = time.monotonic() - mark
    processor, model, timings, status = load_runtime(plan)
    mark = time.monotonic()
    n1_inputs, n1_input = prepare_n1(plan, processor, payload["suffix_input_material"])
    input_reconstruction_seconds = time.monotonic() - mark
    n1 = generate(model, processor, n1_inputs, plan["generation"]["n1"])
    receipt = base_receipt("target", started_wall, started, plan_path, plan)
    receipt.update({
        "process_elapsed_seconds": time.monotonic() - started,
        "static_resource_hashes": checked,
        "static_resource_validation_seconds": static_validation_seconds,
        "state_restore_validation_seconds": restore_validation_seconds,
        "input_reconstruction_seconds": input_reconstruction_seconds,
        "model_load_timings": timings,
        "adapter_status": status,
        "executed_nodes": ["n1"],
        "node_call_counts": {"n0": 0, "n1": 1},
        "generate_attempted": 1,
        "generate_completed": 1,
        "restored_state": package,
        "restored_n0_output": payload["n0_output"],
        "n1_input": n1_input,
        "n1_output": n1,
        "target_access": {
            "declared_read_allowlist": plan["target_read_allowlist"],
            "source_image_access_count": 0,
            "source_process_memory_access": False,
            "undeclared_temporary_file_access": False,
        },
    })
    return receipt


def run_negative_checks(plan: dict[str, Any]) -> dict[str, Any]:
    identity = state_identity(plan)

    def sample_payload(description: str) -> dict[str, Any]:
        return {
            "workflow_id": EXPECTED_WORKFLOW_ID,
            "nodes": EXPECTED_NODES,
            "edges": EXPECTED_EDGES,
            "completed_nodes": ["n0"],
            "remaining_nodes": ["n1"],
            "next_node": "n1",
            "n0_output": {"raw_text": description, "token_ids": [11, 22]},
            "suffix_input_material": description,
            "generation": plan["generation"],
            "identity": identity,
            "rng_state": {"required": False, "reason": "greedy explicit boundary"},
            "tensor_state": {"required": False, "reason": "explicit text boundary"},
        }

    checks: list[dict[str, Any]] = []

    def expect_reject(name: str, envelope: dict[str, Any], expected: str) -> None:
        try:
            validate_state_envelope(envelope, expected_identity=identity)
        except StateValidationError as error:
            checks.append({"name": name, "status": "PASS", "rejected": True, "error": str(error)})
            if expected not in str(error):
                raise AssertionError(f"{name} rejected for unexpected reason: {error}")
        else:
            raise AssertionError(f"{name} was not rejected")

    missing = sample_payload("a line graph")
    del missing["n0_output"]
    expect_reject("missing_intermediate", seal_state_payload(missing), "n0_output is required")

    tampered = seal_state_payload(sample_payload("a line graph"))
    tampered["payload"]["n0_output"]["raw_text"] = "tampered bytes"
    expect_reject("tampered_content_stale_hash", tampered, "payload_sha256 mismatch")

    wrong_next = sample_payload("a line graph")
    wrong_next["next_node"] = "n0"
    expect_reject("next_node_conflict", seal_state_payload(wrong_next), "next_node mismatch")

    wrong_completed = sample_payload("a line graph")
    wrong_completed["completed_nodes"] = []
    expect_reject("completed_nodes_conflict", seal_state_payload(wrong_completed), "completed_nodes mismatch")

    wrong_model = sample_payload("a line graph")
    wrong_model["identity"] = {**identity, "base_weight_sha256": "conflicting-base"}
    expect_reject("model_identity_conflict", seal_state_payload(wrong_model), "identity mismatch")

    descriptions = ["a line graph", "a road intersection"]
    prompts = []
    for description in descriptions:
        valid = validate_state_envelope(seal_state_payload(sample_payload(description)), expected_identity=identity)
        prompt = build_n1_prompt(plan["workflow"]["n1_prompt_template"], valid["suffix_input_material"])
        prompts.append({"description": description, "prompt": prompt, "sha256": sha256_bytes(prompt.encode("utf-8"))})
    dependency_pass = prompts[0]["prompt"] != prompts[1]["prompt"] and prompts[0]["sha256"] != prompts[1]["sha256"]
    checks.append({
        "name": "two_valid_intermediates_change_n1_input",
        "status": "PASS" if dependency_pass else "FAIL",
        "inputs": prompts,
    })
    return {
        "status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "model_loaded": False,
        "generate_calls": 0,
        "checks": checks,
    }


def process_command(script: Path, role: str, plan_path: Path, run_root: Path) -> list[str]:
    return [sys.executable, str(script), "--role", role, "--plan", str(plan_path), "--run-root", str(run_root)]


def run_child(script: Path, role: str, plan_path: Path, run_root: Path, timeout: int) -> dict[str, Any]:
    command = process_command(script, role, plan_path, run_root)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    started_wall, started = time.time(), time.monotonic()
    try:
        completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout, check=False)
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as error:
        returncode = None
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
        timed_out = True
    ended_wall = time.time()
    receipt_path = run_root / f"{role}_receipt.json"
    receipt = read_json(receipt_path) if receipt_path.exists() else {"status": "MISSING"}
    child_wall = time.monotonic() - started
    return {
        "role": role,
        "command": command,
        "pid": receipt.get("pid"),
        "started_at_unix": started_wall,
        "ended_at_unix": ended_wall,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": child_wall,
        "process_startup_overhead_seconds": max(0.0, child_wall - float(receipt.get("process_elapsed_seconds", child_wall))),
        "stdout": stdout,
        "stderr": stderr,
        "receipt_status": receipt.get("status"),
    }


def compare_receipts(continuous: dict[str, Any], source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    comparisons = {
        "continuous_source_n0_text_equal": continuous["n0_output"]["decoded_text"] == source["n0_output"]["decoded_text"],
        "continuous_source_n0_token_ids_equal": continuous["n0_output"]["token_ids"] == source["n0_output"]["token_ids"],
        "continuous_target_n1_prompt_equal": continuous["n1_input"]["prompt"] == target["n1_input"]["prompt"],
        "continuous_target_n1_rendered_prompt_equal": continuous["n1_input"]["rendered_prompt"] == target["n1_input"]["rendered_prompt"],
        "continuous_target_n1_input_ids_equal": continuous["n1_input"]["input_ids"] == target["n1_input"]["input_ids"],
        "continuous_target_n1_input_hash_equal": continuous["n1_input"]["content_sha256"] == target["n1_input"]["content_sha256"],
        "continuous_target_n1_token_ids_equal": continuous["n1_output"]["token_ids"] == target["n1_output"]["token_ids"],
        "identity_equal": continuous["identity"] == source["identity"] == target["identity"],
        "active_adapter_equal": continuous["adapter_status"]["active_adapters"] == source["adapter_status"]["active_adapters"] == target["adapter_status"]["active_adapters"],
        "execution_order_exact": continuous["executed_nodes"] == ["n0", "n1"] and source["executed_nodes"] == ["n0"] and target["executed_nodes"] == ["n1"],
        "target_call_counts_exact": target["node_call_counts"] == {"n0": 0, "n1": 1},
        "target_consumed_saved_n0": target["restored_n0_output"] == {
            "raw_text": source["n0_output"]["decoded_text"],
            "token_ids": source["n0_output"]["token_ids"],
        },
        "target_n1_contains_n0": target["n1_input"]["contains_n0_output"] is True,
        "state_hash_equal": target["restored_state"]["payload_sha256"] == source["state"]["payload_sha256"],
        "target_source_image_access_zero": target["target_access"]["source_image_access_count"] == 0,
    }
    return {"status": "PASS" if all(comparisons.values()) else "FAIL", "checks": comparisons}


def inventory_tree(root: Path, *, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = exclude or set()
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative in excluded:
            continue
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def current_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def execute_supervisor(plan_path: Path, run_root: Path, expected_commit: str) -> int:
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite run root: {run_root}")
    actual_commit = current_commit()
    if actual_commit != expected_commit:
        raise RuntimeError(f"execution commit mismatch: expected {expected_commit}, got {actual_commit}")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
    if dirty:
        raise RuntimeError(f"execution worktree is not clean:\n{dirty}")
    run_root.mkdir(parents=True)
    plan = read_json(plan_path)
    started_wall, started = time.time(), time.monotonic()
    negative = run_negative_checks(plan)
    write_json(run_root / "preflight_negative_checks.json", negative)
    if negative["status"] != "PASS":
        raise RuntimeError("preflight negative checks failed")
    base_path = Path(plan["identity"]["base"]["path"])
    adapter_path = Path(plan["identity"]["adapter"]["path"])
    execution = {
        "status": "STARTED",
        "started_at_unix": started_wall,
        "supervisor_pid": os.getpid(),
        "fixed_execution_commit": actual_commit,
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "module_sha256": sha256_file(ROOT / "src/runtime/workflow_suffix_recovery.py"),
        "automatic_retry": False,
        "planned_generate_calls": plan["budget"]["planned_generate_calls"],
        "hard_generate_ceiling": plan["budget"]["hard_generate_ceiling"],
        "static_resources": {
            "target_prestaged": True,
            "base_directory_bytes": directory_bytes(base_path),
            "base_weight_bytes": (base_path / "model.safetensors").stat().st_size,
            "adapter_directory_bytes": directory_bytes(adapter_path),
            "adapter_weight_bytes": (adapter_path / "adapter_model.safetensors").stat().st_size,
            "missing_target_transfer_required_for_measured_path": False,
        },
    }
    write_json(run_root / "execution_started.json", execution)
    script = Path(__file__).resolve()
    phases = []
    terminal: dict[str, Any]
    try:
        for role in ("continuous", "source", "target"):
            if role == "target":
                execution["source_confirmed_exited_before_target_start"] = phases[-1]["ended_at_unix"] <= time.time()
            phase = run_child(script, role, plan_path, run_root, plan["budget"]["process_timeout_seconds"])
            phases.append(phase)
            write_json(run_root / f"{role}_process.json", phase)
            if phase["returncode"] != 0 or phase["timed_out"] or phase["receipt_status"] != "PASS":
                raise RuntimeError(f"{role} failed; no retry: {phase}")
        continuous = read_json(run_root / "continuous_receipt.json")
        source = read_json(run_root / "source_receipt.json")
        target = read_json(run_root / "target_receipt.json")
        comparisons = compare_receipts(continuous, source, target)
        generate_attempted = sum(item["generate_attempted"] for item in (continuous, source, target))
        generate_completed = sum(item["generate_completed"] for item in (continuous, source, target))
        scientific_wall = time.monotonic() - started
        budget_pass = (
            generate_attempted == plan["budget"]["planned_generate_calls"]
            and generate_completed == plan["budget"]["planned_generate_calls"]
            and generate_attempted <= plan["budget"]["hard_generate_ceiling"]
            and scientific_wall <= plan["budget"]["scientific_total_timeout_seconds"]
        )
        process_separation = (
            len({continuous["pid"], source["pid"], target["pid"]}) == 3
            and phases[1]["ended_at_unix"] <= phases[2]["started_at_unix"]
        )
        terminal = {
            "status": "PASS" if comparisons["status"] == "PASS" and budget_pass and process_separation else "FAIL",
            "classification": plan["classification"],
            "fixed_execution_commit": actual_commit,
            "plan_sha256": sha256_file(plan_path),
            "started_at_unix": started_wall,
            "finished_at_unix": time.time(),
            "scientific_wall_seconds": scientific_wall,
            "automatic_retry": False,
            "phases": phases,
            "source_confirmed_exited_before_target_start": process_separation,
            "process_ids_distinct": len({continuous["pid"], source["pid"], target["pid"]}) == 3,
            "generate_attempted": generate_attempted,
            "generate_completed": generate_completed,
            "budget_compliance": "pass" if budget_pass else "fail",
            "comparisons": comparisons,
            "negative_checks": negative,
            "cost_layers": {
                "target_prestaged_static_resources": execution["static_resources"],
                "transfer_only_if_target_missing": {
                    "base_directory_bytes": execution["static_resources"]["base_directory_bytes"],
                    "adapter_directory_bytes": execution["static_resources"]["adapter_directory_bytes"],
                    "measured_transfer_latency": None,
                },
                "per_workflow_dynamic_state": source["state"],
            },
            "task_correctness": "unavailable",
            "task_correctness_reason": "technical input has no applicable ALPR/Helmet ground-truth label",
            "claims": {
                "workflow_suffix_recovery_fidelity": comparisons["status"].lower(),
                "production_action_4_real_migration": "not_evaluated",
                "algorithm_comparison": "not_evaluated",
                "wireless_transfer": "not_measured",
            },
        }
    except BaseException as error:
        terminal = {
            "status": "FAIL",
            "classification": plan["classification"],
            "fixed_execution_commit": actual_commit,
            "plan_sha256": sha256_file(plan_path),
            "started_at_unix": started_wall,
            "finished_at_unix": time.time(),
            "scientific_wall_seconds": time.monotonic() - started,
            "automatic_retry": False,
            "phases": phases,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "negative_checks": negative,
            "task_correctness": "unavailable",
        }
    write_json(run_root / "terminal_receipt.json", terminal)
    files = inventory_tree(run_root, exclude={"integrity_manifest.json"})
    write_json(run_root / "integrity_manifest.json", {
        "schema_version": "ppo_mec.artifact_integrity.v1",
        "run_root": str(run_root),
        "file_count": len(files),
        "total_bytes_excluding_manifest": sum(item["bytes"] for item in files),
        "files": files,
    })
    return 0 if terminal["status"] == "PASS" else 1


def role_main(role: str, plan_path: Path, run_root: Path) -> int:
    import torch

    torch.set_num_threads(1)
    receipt_path = run_root / f"{role}_receipt.json"
    try:
        if role == "continuous":
            receipt = execute_continuous(plan_path, run_root)
        elif role == "source":
            receipt = execute_source(plan_path, run_root)
        elif role == "target":
            receipt = execute_target(plan_path, run_root)
        else:
            raise ValueError(f"unsupported scientific role: {role}")
    except BaseException as error:
        receipt = {
            "status": "FAIL",
            "role": role,
            "pid": os.getpid(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "automatic_retry": False,
            "task_correctness": "unavailable",
        }
    write_json(receipt_path, receipt)
    print(receipt["status"], flush=True)
    return 0 if receipt["status"] == "PASS" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot two-node workflow suffix-recovery acceptance")
    parser.add_argument("--role", choices=["supervisor", "continuous", "source", "target"], required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan_path = args.plan.resolve()
    run_root = args.run_root.resolve()
    if args.role == "supervisor":
        if not args.expected_commit:
            raise SystemExit("--expected-commit is required for supervisor")
        return execute_supervisor(plan_path, run_root, args.expected_commit)
    return role_main(args.role, plan_path, run_root)


if __name__ == "__main__":
    raise SystemExit(main())
