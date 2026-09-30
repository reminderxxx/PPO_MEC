"""Calibrate a bounded base-only workflow save/independent-process restore path.

This consumes a plan frozen before measurement. It never downloads resources,
loads adapters, clears OS caches, trains, or changes the production action
contract. The resulting state is complete only for the declared two-node
calibration workflow, not for an arbitrary model runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any


PROCESSOR_RESOURCE_FILES = (
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_status(text: str) -> str:
    return "".join(character for character in text.lower() if "a" <= character <= "z")


def processor_resource_identity(model_root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name in PROCESSOR_RESOURCE_FILES:
        path = model_root / name
        rows[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return rows


def state_without_integrity(
    plan: dict[str, Any],
    stage1: dict[str, Any],
    resources: dict[str, dict[str, Any]],
    measurement_index: int,
) -> dict[str, Any]:
    workflow = plan["workflow"]
    model = plan["base_model"]
    return {
        "schema": plan["state_contract"]["schema"],
        "workflow_id": workflow["workflow_id"],
        "workflow_version": workflow["workflow_version"],
        "measurement_index": measurement_index,
        "boundary": workflow["stage_boundary"],
        "completed_node_ids": ["normalize_event"],
        "remaining_node_ids": ["publish_event"],
        "dag": {
            "execution_order": ["normalize_event", "publish_event"],
            "edges": workflow["edges"],
        },
        "control_state": {
            "next_node_id": "publish_event",
            "next_node_attempt": 0,
            "terminated": False,
        },
        "node_outputs": {
            "normalize_event": {
                "decoded_text": stage1["decoded_text"],
                "generated_token_ids": stage1["generated_token_ids"],
                "normalized_status": stage1["normalized_status"],
            }
        },
        "generation_contract": {
            "do_sample": False,
            "stage1_max_new_tokens": workflow["nodes"][0]["max_new_tokens"],
            "stage2_max_new_tokens": workflow["nodes"][1]["max_new_tokens"],
            "seed": plan["state_contract"]["random_state"]["seed"],
        },
        "random_state": plan["state_contract"]["random_state"],
        "kv_cache": plan["state_contract"]["kv_cache"],
        "tensor_state": plan["state_contract"]["tensor_state"],
        "base_model_identity": {
            "repository": model["repository"],
            "revision": model["revision"],
            "weight_file": model["weight_file"],
            "weight_bytes": model["weight_bytes"],
            "weight_sha256": model["weight_sha256"],
            "adapter": None,
        },
        "processor_resource_identity": resources,
        "input_identity": {
            "stage1_input_sha256": sha256_bytes(workflow["nodes"][0]["input"].encode("utf-8")),
            "stage2_template_sha256": sha256_bytes(
                workflow["nodes"][1]["input_template"].encode("utf-8")
            ),
        },
        "claim_boundary": plan["state_contract"]["completeness_boundary"],
    }


def add_state_integrity(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["integrity"] = {
        "algorithm": "sha256",
        "payload_without_integrity_sha256": sha256_bytes(canonical_bytes(payload)),
    }
    return result


def validate_state(
    state: dict[str, Any],
    plan: dict[str, Any],
    actual_resources: dict[str, dict[str, Any]],
) -> None:
    integrity = state.get("integrity")
    if not isinstance(integrity, dict):
        raise ValueError("state integrity record missing")
    without_integrity = {key: value for key, value in state.items() if key != "integrity"}
    if integrity.get("algorithm") != "sha256" or integrity.get(
        "payload_without_integrity_sha256"
    ) != sha256_bytes(canonical_bytes(without_integrity)):
        raise ValueError("state content hash mismatch")
    workflow = plan["workflow"]
    expected_exact = {
        "schema": plan["state_contract"]["schema"],
        "workflow_id": workflow["workflow_id"],
        "workflow_version": workflow["workflow_version"],
        "boundary": workflow["stage_boundary"],
        "completed_node_ids": ["normalize_event"],
        "remaining_node_ids": ["publish_event"],
    }
    for key, value in expected_exact.items():
        if state.get(key) != value:
            raise ValueError(f"state {key} mismatch")
    if state.get("dag") != {
        "execution_order": ["normalize_event", "publish_event"],
        "edges": workflow["edges"],
    }:
        raise ValueError("state DAG mismatch")
    if state.get("control_state") != {
        "next_node_id": "publish_event",
        "next_node_attempt": 0,
        "terminated": False,
    }:
        raise ValueError("state control mismatch")
    if state.get("processor_resource_identity") != actual_resources:
        raise ValueError("processor resource identity mismatch")
    model = plan["base_model"]
    expected_model = {
        "repository": model["repository"],
        "revision": model["revision"],
        "weight_file": model["weight_file"],
        "weight_bytes": model["weight_bytes"],
        "weight_sha256": model["weight_sha256"],
        "adapter": None,
    }
    if state.get("base_model_identity") != expected_model:
        raise ValueError("base model identity mismatch")
    output = state.get("node_outputs", {}).get("normalize_event", {})
    if output.get("normalized_status") != workflow["expected_normalized_status"]:
        raise ValueError("upstream normalized output failed frozen correctness criterion")


def durable_create(path: Path, payload: bytes) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite state: {path}")
    staged = path.with_name(path.name + ".tmp")
    if staged.exists():
        raise FileExistsError(f"refusing to overwrite staged state: {staged}")
    write_start = time.perf_counter_ns()
    with staged.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staged, path)
    directory_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {"durable_write_ns": time.perf_counter_ns() - write_start}


def configure_runtime(plan: dict[str, Any]) -> tuple[Any, Any, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    torch.set_num_threads(int(plan["runtime"]["torch_num_threads"]))
    torch.set_num_interop_threads(int(plan["runtime"]["torch_num_interop_threads"]))
    torch.manual_seed(int(plan["state_contract"]["random_state"]["seed"]))
    return torch, AutoModelForVision2Seq, AutoProcessor


def synchronize(torch: Any, device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def memory_snapshot(torch: Any, device: str) -> dict[str, Any]:
    return {
        "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "mps_current_allocated_bytes": (
            int(torch.mps.current_allocated_memory()) if device == "mps" else None
        ),
        "mps_driver_allocated_bytes": (
            int(torch.mps.driver_allocated_memory()) if device == "mps" else None
        ),
    }


def load_model(plan: dict[str, Any]) -> tuple[Any, Any, Any, int]:
    torch, model_class, processor_class = configure_runtime(plan)
    model_path = plan["base_model"]["local_path"]
    device = plan["runtime"]["device"]
    dtype = getattr(torch, plan["runtime"]["dtype"])
    synchronize(torch, device)
    started = time.perf_counter_ns()
    processor = processor_class.from_pretrained(model_path, local_files_only=True)
    model = model_class.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.eval()
    model.to(device)
    synchronize(torch, device)
    return torch, processor, model, time.perf_counter_ns() - started


def generate_stage(
    torch: Any,
    processor: Any,
    model: Any,
    device: str,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    render_started = time.perf_counter_ns()
    chat = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    rendered = processor.apply_chat_template(chat, add_generation_prompt=True)
    render_ns = time.perf_counter_ns() - render_started
    tokenize_started = time.perf_counter_ns()
    inputs = processor(text=[rendered], return_tensors="pt")
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in dict(inputs).items()
    }
    synchronize(torch, device)
    tokenize_ns = time.perf_counter_ns() - tokenize_started
    synchronize(torch, device)
    inference_started = time.perf_counter_ns()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )
    synchronize(torch, device)
    inference_ns = time.perf_counter_ns() - inference_started
    prompt_token_count = int(inputs["input_ids"].shape[-1])
    generated = output[:, prompt_token_count:]
    token_ids = [int(value) for value in generated[0].detach().cpu().tolist()]
    decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return {
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "render_ns": render_ns,
        "tokenize_and_device_transfer_ns": tokenize_ns,
        "inference_ns": inference_ns,
        "prompt_token_count": prompt_token_count,
        "generated_token_ids": token_ids,
        "generated_token_count": len(token_ids),
        "decoded_text": decoded,
        "normalized_status": normalize_status(decoded),
    }


def run_continuous(plan: dict[str, Any]) -> dict[str, Any]:
    total_started = time.perf_counter_ns()
    torch, processor, model, load_ns = load_model(plan)
    workflow = plan["workflow"]
    stage1 = generate_stage(
        torch,
        processor,
        model,
        plan["runtime"]["device"],
        workflow["nodes"][0]["input"],
        workflow["nodes"][0]["max_new_tokens"],
    )
    rebuild_started = time.perf_counter_ns()
    stage2_prompt = workflow["nodes"][1]["input_template"].format(
        normalized_status=stage1["normalized_status"]
    )
    rebuild_ns = time.perf_counter_ns() - rebuild_started
    stage2 = generate_stage(
        torch,
        processor,
        model,
        plan["runtime"]["device"],
        stage2_prompt,
        workflow["nodes"][1]["max_new_tokens"],
    )
    return {
        "status": "pass",
        "mode": "continuous",
        "pid": os.getpid(),
        "model_load_ns": load_ns,
        "stage1": stage1,
        "stage2_prompt_rebuild_ns": rebuild_ns,
        "stage2": stage2,
        "memory": memory_snapshot(torch, plan["runtime"]["device"]),
        "workflow_total_ns": time.perf_counter_ns() - total_started,
    }


def run_source(plan: dict[str, Any], state_path: Path, measurement_index: int) -> dict[str, Any]:
    total_started = time.perf_counter_ns()
    model_root = Path(plan["base_model"]["local_path"])
    resources = processor_resource_identity(model_root)
    torch, processor, model, load_ns = load_model(plan)
    workflow = plan["workflow"]
    stage1 = generate_stage(
        torch,
        processor,
        model,
        plan["runtime"]["device"],
        workflow["nodes"][0]["input"],
        workflow["nodes"][0]["max_new_tokens"],
    )
    serialization_started = time.perf_counter_ns()
    state = add_state_integrity(
        state_without_integrity(plan, stage1, resources, measurement_index)
    )
    encoded = canonical_bytes(state)
    serialization_ns = time.perf_counter_ns() - serialization_started
    save = durable_create(state_path, encoded)
    return {
        "status": "pass",
        "mode": "source",
        "pid": os.getpid(),
        "model_load_ns": load_ns,
        "stage1": stage1,
        "state": {
            "path": str(state_path),
            "bytes": state_path.stat().st_size,
            "sha256": sha256_file(state_path),
            "serialization_ns": serialization_ns,
            **save,
        },
        "memory": memory_snapshot(torch, plan["runtime"]["device"]),
        "workflow_total_ns": time.perf_counter_ns() - total_started,
    }


def run_target(plan: dict[str, Any], state_path: Path) -> dict[str, Any]:
    total_started = time.perf_counter_ns()
    read_started = time.perf_counter_ns()
    encoded = state_path.read_bytes()
    read_ns = time.perf_counter_ns() - read_started
    parse_started = time.perf_counter_ns()
    state = json.loads(encoded)
    parse_ns = time.perf_counter_ns() - parse_started
    resource_started = time.perf_counter_ns()
    resources = processor_resource_identity(Path(plan["base_model"]["local_path"]))
    validate_state(state, plan, resources)
    resource_and_state_validation_ns = time.perf_counter_ns() - resource_started
    torch, processor, model, load_ns = load_model(plan)
    workflow = plan["workflow"]
    rebuild_started = time.perf_counter_ns()
    normalized = state["node_outputs"]["normalize_event"]["normalized_status"]
    stage2_prompt = workflow["nodes"][1]["input_template"].format(
        normalized_status=normalized
    )
    rebuild_ns = time.perf_counter_ns() - rebuild_started
    stage2 = generate_stage(
        torch,
        processor,
        model,
        plan["runtime"]["device"],
        stage2_prompt,
        workflow["nodes"][1]["max_new_tokens"],
    )
    return {
        "status": "pass",
        "mode": "target",
        "pid": os.getpid(),
        "state": {
            "path": str(state_path),
            "bytes": len(encoded),
            "sha256": sha256_bytes(encoded),
            "read_ns": read_ns,
            "parse_ns": parse_ns,
            "resource_and_state_validation_ns": resource_and_state_validation_ns,
            "restore_total_ns": read_ns + parse_ns + resource_and_state_validation_ns,
        },
        "model_load_ns": load_ns,
        "stage2_prompt_rebuild_ns": rebuild_ns,
        "stage2": stage2,
        "memory": memory_snapshot(torch, plan["runtime"]["device"]),
        "workflow_total_ns": time.perf_counter_ns() - total_started,
    }


def child_failure(mode: str, error: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "mode": mode,
        "pid": os.getpid(),
        "error_type": type(error).__name__,
        "error": str(error),
    }


def run_child_command(
    python: str,
    script: Path,
    plan_path: Path,
    mode: str,
    timeout_seconds: int,
    state_path: Path | None = None,
    measurement_index: int | None = None,
) -> dict[str, Any]:
    command = [python, str(script), "--plan", str(plan_path), "--child-mode", mode]
    if state_path is not None:
        command.extend(["--state", str(state_path)])
    if measurement_index is not None:
        command.extend(["--measurement-index", str(measurement_index)])
    started_unix_ns = time.time_ns()
    wall_started = time.perf_counter_ns()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env={
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
    except subprocess.TimeoutExpired as error:
        return {
            "status": "timeout",
            "mode": mode,
            "command": command,
            "started_unix_ns": started_unix_ns,
            "finished_unix_ns": time.time_ns(),
            "process_wall_ns": time.perf_counter_ns() - wall_started,
            "timeout_seconds": timeout_seconds,
            "stdout": error.stdout,
            "stderr": error.stderr,
        }
    row: dict[str, Any] = {
        "mode": mode,
        "command": command,
        "returncode": completed.returncode,
        "started_unix_ns": started_unix_ns,
        "finished_unix_ns": time.time_ns(),
        "process_wall_ns": time.perf_counter_ns() - wall_started,
        "stderr": completed.stderr,
    }
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError:
        row.update(status="failed_invalid_json", stdout=completed.stdout)
        return row
    row.update(parsed)
    if completed.returncode != 0 and row.get("status") == "pass":
        row["status"] = "failed_nonzero_return"
    return row


def pair_correctness(plan: dict[str, Any], rows: dict[str, Any]) -> dict[str, Any]:
    continuous = rows["continuous"]
    source = rows["source"]
    target = rows["target"]
    expected = plan["workflow"]["expected_normalized_status"]
    all_children_pass = all(row.get("status") == "pass" for row in rows.values())
    if not all_children_pass:
        return {"all_children_pass": False, "pair_pass": False}
    checks = {
        "source_and_target_are_distinct_processes": source["pid"] != target["pid"],
        "source_finished_before_target_started": source["finished_unix_ns"] <= target["started_unix_ns"],
        "state_file_sha_matches": source["state"]["sha256"] == target["state"]["sha256"],
        "state_file_bytes_match": source["state"]["bytes"] == target["state"]["bytes"],
        "intermediate_token_ids_exact": (
            continuous["stage1"]["generated_token_ids"]
            == source["stage1"]["generated_token_ids"]
        ),
        "intermediate_decoded_text_exact": (
            continuous["stage1"]["decoded_text"] == source["stage1"]["decoded_text"]
        ),
        "intermediate_normalized_exact": (
            continuous["stage1"]["normalized_status"]
            == source["stage1"]["normalized_status"]
            == expected
        ),
        "final_token_ids_exact": (
            continuous["stage2"]["generated_token_ids"]
            == target["stage2"]["generated_token_ids"]
        ),
        "final_decoded_text_exact": (
            continuous["stage2"]["decoded_text"] == target["stage2"]["decoded_text"]
        ),
        "final_normalized_exact": (
            continuous["stage2"]["normalized_status"]
            == target["stage2"]["normalized_status"]
            == expected
        ),
    }
    return {
        "all_children_pass": True,
        "checks": checks,
        "pair_pass": all(checks.values()),
        "float_rtol": plan["workflow"]["correctness"]["float_rtol"],
        "float_atol": plan["workflow"]["correctness"]["float_atol"],
        "float_comparison_used": False,
    }


def run_pair(
    plan: dict[str, Any],
    plan_path: Path,
    script: Path,
    state_path: Path,
    measurement_index: int,
) -> dict[str, Any]:
    python = plan["runtime"]["python_executable"]
    timeout = int(plan["conditions"]["timeout_seconds_per_process"])
    continuous = run_child_command(
        python, script, plan_path, "continuous", timeout, measurement_index=measurement_index
    )
    if continuous.get("status") != "pass":
        rows = {"continuous": continuous}
        return {"measurement_index": measurement_index, "rows": rows, "correctness": {"pair_pass": False}}
    source = run_child_command(
        python,
        script,
        plan_path,
        "source",
        timeout,
        state_path=state_path,
        measurement_index=measurement_index,
    )
    if source.get("status") != "pass":
        rows = {"continuous": continuous, "source": source}
        return {"measurement_index": measurement_index, "rows": rows, "correctness": {"pair_pass": False}}
    target = run_child_command(
        python,
        script,
        plan_path,
        "target",
        timeout,
        state_path=state_path,
        measurement_index=measurement_index,
    )
    rows = {"continuous": continuous, "source": source, "target": target}
    return {
        "measurement_index": measurement_index,
        "rows": rows,
        "correctness": pair_correctness(plan, rows),
    }


def median_summary(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min_ns": min(values),
        "median_ns": int(statistics.median(values)),
        "max_ns": max(values),
        "mean_ns": int(statistics.mean(values)),
        "raw_ns": values,
    }


def measurement_summary(measurements: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not measurements or not all(row["correctness"].get("pair_pass") for row in measurements):
        return None
    continuous_internal = [row["rows"]["continuous"]["workflow_total_ns"] for row in measurements]
    recovered_internal = [
        row["rows"]["source"]["workflow_total_ns"]
        + row["rows"]["target"]["workflow_total_ns"]
        for row in measurements
    ]
    recovered_minus_continuous = [
        recovered - continuous
        for continuous, recovered in zip(continuous_internal, recovered_internal)
    ]
    return {
        "state_bytes": sorted({row["rows"]["source"]["state"]["bytes"] for row in measurements}),
        "continuous_internal_total": median_summary(continuous_internal),
        "recovered_internal_total_no_transport": median_summary(recovered_internal),
        "recovered_minus_continuous_internal_no_transport": median_summary(
            recovered_minus_continuous
        ),
        "state_serialization": median_summary(
            [row["rows"]["source"]["state"]["serialization_ns"] for row in measurements]
        ),
        "state_durable_save": median_summary(
            [row["rows"]["source"]["state"]["durable_write_ns"] for row in measurements]
        ),
        "state_restore_read_parse_validate": median_summary(
            [row["rows"]["target"]["state"]["restore_total_ns"] for row in measurements]
        ),
        "target_model_load": median_summary(
            [row["rows"]["target"]["model_load_ns"] for row in measurements]
        ),
        "target_prompt_rebuild": median_summary(
            [row["rows"]["target"]["stage2_prompt_rebuild_ns"] for row in measurements]
        ),
        "target_stage2_tokenize": median_summary(
            [
                row["rows"]["target"]["stage2"]["tokenize_and_device_transfer_ns"]
                for row in measurements
            ]
        ),
        "target_stage2_inference": median_summary(
            [row["rows"]["target"]["stage2"]["inference_ns"] for row in measurements]
        ),
        "process_wall_continuous": median_summary(
            [row["rows"]["continuous"]["process_wall_ns"] for row in measurements]
        ),
        "process_wall_source_plus_target": median_summary(
            [
                row["rows"]["source"]["process_wall_ns"]
                + row["rows"]["target"]["process_wall_ns"]
                for row in measurements
            ]
        ),
        "source_process_peak_rss": median_summary(
            [row["rows"]["source"]["memory"]["process_max_rss_bytes"] for row in measurements]
        ),
        "target_process_peak_rss": median_summary(
            [row["rows"]["target"]["memory"]["process_max_rss_bytes"] for row in measurements]
        ),
        "continuous_process_peak_rss": median_summary(
            [
                row["rows"]["continuous"]["memory"]["process_max_rss_bytes"]
                for row in measurements
            ]
        ),
    }


def verify_plan(plan: dict[str, Any]) -> None:
    if plan.get("frozen_before_measurement") is not True:
        raise ValueError("measurement plan is not frozen")
    if plan.get("plan_version") != "1.2.0":
        raise ValueError("unexpected plan version")
    if plan["conditions"] != {
        "warmup_pairs": 1,
        "measurement_pairs": 3,
        "timeout_seconds_per_process": 180,
        "load_label": "process-first load with unspecified OS page-cache state",
        "storage_cold_claim_allowed": False,
        "network_claim_allowed": False,
        "cross_rsu_claim_allowed": False,
        "failure_policy": "retain the first configuration or execution failure; only a documented configuration correction may be rerun and must use a versioned output",
    }:
        raise ValueError("measurement conditions differ from frozen v1.1 plan")
    if plan["adapter_measurement"]["status"] != (
        "unavailable_no_local_pair_and_new_weight_download_not_authorized"
    ):
        raise ValueError("adapter authorization boundary changed")


def verify_base(plan: dict[str, Any]) -> dict[str, Any]:
    model = plan["base_model"]
    weight = Path(model["local_path"]) / model["weight_file"]
    actual = {
        "path": str(weight),
        "bytes": weight.stat().st_size,
        "sha256": sha256_file(weight),
    }
    actual["match"] = bool(
        actual["bytes"] == model["weight_bytes"]
        and actual["sha256"] == model["weight_sha256"]
    )
    if not actual["match"]:
        raise ValueError("base model weight identity mismatch")
    return actual


def orchestrate(plan_path: Path, output_path: Path, state_root: Path) -> dict[str, Any]:
    if output_path.exists() or state_root.exists():
        raise FileExistsError("output and state root must be create-only")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    verify_plan(plan)
    base_identity = verify_base(plan)
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="ppo_mec_state_recovery_warmup_") as temp_root:
        warmup = run_pair(
            plan,
            plan_path.resolve(),
            script,
            Path(temp_root) / "state.json",
            -1,
        )
    measurements: list[dict[str, Any]] = []
    if warmup["correctness"].get("pair_pass"):
        state_root.mkdir(parents=True, exist_ok=False)
        for index in range(1, int(plan["conditions"]["measurement_pairs"]) + 1):
            trial_root = state_root / f"measurement_{index:02d}"
            trial_root.mkdir(exist_ok=False)
            row = run_pair(
                plan,
                plan_path.resolve(),
                script,
                trial_root / "state.json",
                index,
            )
            measurements.append(row)
            if not row["correctness"].get("pair_pass"):
                break
    failures = []
    for phase, pairs in (("warmup", [warmup]), ("measurement", measurements)):
        for pair in pairs:
            for mode, row in pair["rows"].items():
                if row.get("status") != "pass":
                    failures.append(
                        {
                            "phase": phase,
                            "measurement_index": pair["measurement_index"],
                            "mode": mode,
                            "status": row.get("status"),
                            "returncode": row.get("returncode"),
                            "error_type": row.get("error_type"),
                            "error": row.get("error"),
                            "stderr": row.get("stderr"),
                        }
                    )
            if not pair["correctness"].get("pair_pass") and not any(
                row.get("status") != "pass" for row in pair["rows"].values()
            ):
                failures.append(
                    {
                        "phase": phase,
                        "measurement_index": pair["measurement_index"],
                        "mode": "pair_correctness",
                        "status": "failed_correctness",
                        "checks": pair["correctness"].get("checks"),
                    }
                )
    output = {
        "artifact_run_id": plan["artifact_run_id"],
        "evidence_type": "same_host_independent_process_recovery_witness",
        "measured_at_unix_ns": time.time_ns(),
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "runner_path": str(script),
        "runner_sha256": sha256_file(script),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_orchestrator": sys.version,
        },
        "base_weight_identity": base_identity,
        "adapter_execution": {
            "status": "unavailable",
            "reason": plan["adapter_measurement"]["status"],
            "adapter_processes": 0,
            "adapter_model_calls": 0,
            "downloaded_weight_bytes": 0,
        },
        "warmup": warmup,
        "measurements": measurements,
        "measurement_count": len(measurements),
        "all_measurements_pass": bool(
            len(measurements) == int(plan["conditions"]["measurement_pairs"])
            and all(row["correctness"].get("pair_pass") for row in measurements)
        ),
        "summary": measurement_summary(measurements),
        "failure_records": failures,
        "execution_counts": {
            "warmup_pairs": 1,
            "measurement_pairs": len(measurements),
            "model_processes": sum(len(pair["rows"]) for pair in [warmup, *measurements]),
            "maximum_model_processes": plan["resource_budget"]["maximum_model_processes"],
            "model_calls": sum(
                2 if mode == "continuous" else 1
                for pair in [warmup, *measurements]
                for mode, row in pair["rows"].items()
                if row.get("status") == "pass"
            ),
            "maximum_model_calls": plan["resource_budget"]["maximum_new_model_calls"],
        },
        "independence_boundary": {
            "source_exits_before_target_spawn": all(
                row["correctness"].get("checks", {}).get(
                    "source_finished_before_target_started", False
                )
                for row in measurements
            ),
            "target_declared_inputs": [
                "retained state.json",
                "frozen measurement_plan.json",
                "pinned local model/processor files",
                "calibration runner source"
            ],
            "undeclared_source_memory_or_files_used": False,
            "transport_measured": False,
            "cross_rsu_measured": False,
        },
        "claim_boundary": plan["state_contract"]["completeness_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
        handle.write(b"\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--child-mode", choices=("continuous", "source", "target"), default=None
    )
    parser.add_argument("--state", type=Path)
    parser.add_argument("--measurement-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state-root", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.child_mode:
        try:
            if args.child_mode == "continuous":
                result = run_continuous(plan)
            elif args.child_mode == "source":
                if args.state is None:
                    raise ValueError("source requires --state")
                result = run_source(plan, args.state, args.measurement_index)
            else:
                if args.state is None:
                    raise ValueError("target requires --state")
                result = run_target(plan, args.state)
        except Exception as error:
            print(json.dumps(child_failure(args.child_mode, error), sort_keys=True))
            raise SystemExit(1)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return
    if args.output is None or args.state_root is None:
        parser.error("orchestration requires --output and --state-root")
    output = orchestrate(args.plan.resolve(), args.output, args.state_root)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "measurement_count": output["measurement_count"],
                "all_measurements_pass": output["all_measurements_pass"],
                "failure_count": len(output["failure_records"]),
                "adapter_status": output["adapter_execution"]["status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
