"""Bounded local model and explicit application-state cost measurement.

The script consumes a pre-frozen plan.  It does not download models, clear OS
caches, synthesize adapters, or treat a runtime object dump as workflow state.
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


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def application_state_payload(model_identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "explicit_application_workflow_state_v1",
        "workflow_id": "bounded_local_measurement_workflow",
        "workflow_version": "1.0.0",
        "completed_node_ids": ["normalize", "extract"],
        "current_node_id": "summarize",
        "node_outputs": {
            "normalize": {
                "event_text": "lane obstruction reported near interchange",
                "event_time": "2026-09-30T17:20:00+08:00",
            },
            "extract": {
                "event_type": "road_obstruction",
                "location": "interchange",
                "confidence": 0.75,
            },
        },
        "conversation_summary": "A non-safety-critical road-event information request is active.",
        "model_identity": model_identity,
    }


def deterministic_next_node_key(payload: dict[str, Any]) -> str:
    material = {
        "workflow_id": payload["workflow_id"],
        "current_node_id": payload["current_node_id"],
        "completed_node_ids": payload["completed_node_ids"],
        "node_outputs": payload["node_outputs"],
        "model_identity": payload["model_identity"],
    }
    return hashlib.sha256(canonical_bytes(material)).hexdigest()


def measure_state_once(payload: dict[str, Any], directory: Path, index: int) -> dict[str, Any]:
    target = directory / f"state_{index}.json"
    staged = directory / f"state_{index}.json.tmp"
    encoded = canonical_bytes(payload)
    save_start = time.perf_counter_ns()
    with staged.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staged, target)
    save_ns = time.perf_counter_ns() - save_start

    restore_start = time.perf_counter_ns()
    restored = json.loads(target.read_text(encoding="utf-8"))
    restore_ns = time.perf_counter_ns() - restore_start
    expected_key = deterministic_next_node_key(payload)
    restored_key = deterministic_next_node_key(restored)
    return {
        "index": index,
        "serialized_bytes": len(encoded),
        "file_bytes": target.stat().st_size,
        "save_ns": save_ns,
        "restore_ns": restore_ns,
        "canonical_payload_equal": canonical_bytes(restored) == encoded,
        "next_node_key_before": expected_key,
        "next_node_key_after": restored_key,
        "continuation_key_equal": expected_key == restored_key,
    }


def _move_to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def child_model_measurement(plan: dict[str, Any], run_inference: bool) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    model_path = plan["model"]["local_path"]
    device = plan["runtime"]["device"]
    dtype_name = plan["runtime"]["dtype"]
    dtype = getattr(torch, dtype_name)
    load_start = time.perf_counter_ns()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.eval()
    model.to(device)
    if device == "mps":
        torch.mps.synchronize()
    load_ns = time.perf_counter_ns() - load_start

    result: dict[str, Any] = {
        "status": "pass",
        "load_ns": load_ns,
        "run_inference": run_inference,
        "process_max_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mps_current_allocated_bytes_after_load": (
            int(torch.mps.current_allocated_memory()) if device == "mps" else None
        ),
        "inference_warmup_ns": None,
        "inference_measurement_ns": None,
        "generated_token_count": None,
        "decoded_text": None,
    }
    if not run_inference:
        return result

    prompt = plan["fixed_input"]["text"]
    chat = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    rendered = processor.apply_chat_template(chat, add_generation_prompt=True)
    inputs = processor(text=[rendered], return_tensors="pt")
    inputs = _move_to_device(dict(inputs), device)

    def generate() -> Any:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=int(plan["fixed_input"]["max_new_tokens"]),
                do_sample=bool(plan["fixed_input"]["do_sample"]),
            )
        if device == "mps":
            torch.mps.synchronize()
        return output

    warmup_start = time.perf_counter_ns()
    generate()
    result["inference_warmup_ns"] = time.perf_counter_ns() - warmup_start
    measured_start = time.perf_counter_ns()
    output = generate()
    result["inference_measurement_ns"] = time.perf_counter_ns() - measured_start
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    generated = output[:, prompt_tokens:]
    result["generated_token_count"] = int(generated.shape[-1])
    result["decoded_text"] = processor.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0]
    result["process_max_rss_raw"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["mps_current_allocated_bytes_after_inference"] = (
        int(torch.mps.current_allocated_memory()) if device == "mps" else None
    )
    if result["generated_token_count"] < 1:
        result["status"] = "fail_no_generated_token"
    return result


def _summary_ns(rows: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    values = [int(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return None
    return {
        "count": len(values),
        "min_ns": min(values),
        "median_ns": int(statistics.median(values)),
        "max_ns": max(values),
        "mean_ns": int(statistics.mean(values)),
        "raw_ns": values,
    }


def run_child(python: str, script: Path, plan_path: Path, run_inference: bool, timeout: int) -> dict[str, Any]:
    command = [
        python,
        str(script),
        "--plan",
        str(plan_path),
        "--child",
        "--run-inference",
        "yes" if run_inference else "no",
    ]
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
    except subprocess.TimeoutExpired as error:
        return {
            "status": "timeout",
            "command": command,
            "timeout_seconds": timeout,
            "elapsed_seconds": time.time() - started,
            "stdout": error.stdout,
            "stderr": error.stderr,
        }
    row: dict[str, Any] = {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": time.time() - started,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        row.update(status="failed", stdout=completed.stdout)
        return row
    try:
        row.update(json.loads(completed.stdout))
    except json.JSONDecodeError:
        row.update(status="failed_invalid_json", stdout=completed.stdout)
    return row


def orchestrate(plan_path: Path, output_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    model_root = Path(plan["model"]["local_path"])
    weight = model_root / plan["model"]["weight_file"]
    weight_identity = {
        "path": str(weight),
        "expected_bytes": plan["model"]["weight_bytes"],
        "actual_bytes": weight.stat().st_size,
        "expected_sha256": plan["model"]["weight_sha256"],
        "actual_sha256": sha256_file(weight),
    }
    weight_identity["match"] = bool(
        weight_identity["expected_bytes"] == weight_identity["actual_bytes"]
        and weight_identity["expected_sha256"] == weight_identity["actual_sha256"]
    )
    adapter_candidates = sorted(
        str(path) for path in model_root.parent.rglob("adapter_config.json")
    )
    if adapter_candidates:
        raise RuntimeError("measurement plan froze adapter as unavailable but candidates appeared")

    python = plan["runtime"]["python"]
    timeout = int(plan["conditions"]["timeout_seconds_per_process"])
    warmup = run_child(python, Path(__file__), plan_path, False, timeout)
    rows = [
        run_child(python, Path(__file__), plan_path, True, timeout)
        for _ in range(int(plan["conditions"]["load_measurements"]))
    ]

    model_identity = {
        "repository": plan["model"]["repository"],
        "revision": plan["model"]["revision"],
        "weight_sha256": plan["model"]["weight_sha256"],
        "adapter": None,
    }
    state_payload = application_state_payload(model_identity)
    with tempfile.TemporaryDirectory(prefix="ppo_mec_state_measurement_") as temp_dir:
        state_root = Path(temp_dir)
        measure_state_once(state_payload, state_root, -1)
        state_rows = [
            measure_state_once(state_payload, state_root, index)
            for index in range(int(plan["state_measurement"]["measurements"]))
        ]

    output = {
        "artifact_run_id": plan["artifact_run_id"],
        "evidence_type": "local_host_measurement",
        "measured_at_unix": time.time(),
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_orchestrator": sys.version,
        },
        "weight_identity": weight_identity,
        "adapter_inventory": {
            "adapter_config_candidates": adapter_candidates,
            "status": "unavailable_no_locally_identified_compatible_adapter",
            "adapter_load_or_switch_time": None,
        },
        "base_load_warmup": warmup,
        "base_load_measurements": rows,
        "base_load_summary": _summary_ns(rows, "load_ns"),
        "minimal_inference_summary": _summary_ns(rows, "inference_measurement_ns"),
        "minimal_inference_success_count": sum(row.get("status") == "pass" for row in rows),
        "state_payload_schema": state_payload["schema"],
        "state_measurements": state_rows,
        "state_save_summary": _summary_ns(state_rows, "save_ns"),
        "state_restore_summary": _summary_ns(state_rows, "restore_ns"),
        "state_serialized_bytes": sorted({row["serialized_bytes"] for row in state_rows}),
        "state_restore_correct": all(
            row["canonical_payload_equal"] and row["continuation_key_equal"]
            for row in state_rows
        ),
        "measurement_boundaries": {
            "load": "process-first; OS page-cache state uncontrolled and not cleared",
            "memory": "per-process ru_maxrss raw value plus PyTorch MPS current allocation; neither is whole-system peak memory",
            "network": "not measured",
            "deployment": "Apple M5 local host only; not representative of vehicle/RSU/wireless deployment",
            "workflow_state": "explicit application payload only; not KV cache or complete runtime migration state",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--run-inference", choices=["yes", "no"], default="no")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.child:
        try:
            value = child_model_measurement(plan, args.run_inference == "yes")
        except Exception as error:  # keep the bounded failure record in the parent
            print(json.dumps({"status": "failed", "error": repr(error)}))
            raise
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return
    if args.output is None:
        parser.error("--output is required unless --child is used")
    output = orchestrate(args.plan, args.output)
    print(json.dumps({
        "output": str(args.output),
        "load_successes": sum(row.get("status") == "pass" for row in output["base_load_measurements"]),
        "inference_successes": output["minimal_inference_success_count"],
        "state_restore_correct": output["state_restore_correct"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
