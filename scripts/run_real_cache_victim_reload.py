"""Run the frozen 12-call real adapter victim->reload witness.

This is an explicit opt-in technical measurement.  Both restart and recovery
arms use the same native typed-cache transaction path and the same PEFT runtime
lifecycle bridge.  Network time remains a reported assumption only.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_two_node_workflow_suffix_recovery import (  # noqa: E402
    current_commit,
    generate,
    inventory_tree,
    prepare_n0,
    prepare_n1,
    read_json,
    sha256_file,
    write_json,
)
from src.data.mobility.replay_provider import ReplayProvider  # noqa: E402
from src.data.model_catalog.adapter_catalog import AdapterCatalog  # noqa: E402
from src.envs.core.cache_eviction import (  # noqa: E402
    TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU,
)
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv  # noqa: E402
from src.envs.specs import ControlAction, RSUState, WorkflowGraphState, WorkflowNode  # noqa: E402
from src.runtime.peft_adapter_lifecycle import (  # noqa: E402
    PeftAdapterLifecycleError,
    apply_previewed_adapter_transaction,
    loaded_adapter_names,
    runtime_snapshot,
)
from src.runtime.production_action4_state import (  # noqa: E402
    export_action4_state,
    import_action4_state,
)
from src.runtime.workflow_suffix_recovery import input_record  # noqa: E402


MIB = 1_048_576
ARMS = ("restart", "recovery")
BASE_ID = "smolvlm_500m"
BASE_OBJECT = f"base:{BASE_ID}"
CURRENT_ADAPTER = "helmet"
FUTURE_ADAPTER = "alpr"
ADAPTER_OBJECTS = {
    CURRENT_ADAPTER: f"adapter:{CURRENT_ADAPTER}",
    FUTURE_ADAPTER: f"adapter:{FUTURE_ADAPTER}",
}


def _typed_object(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["stable_fingerprint"] = AdapterCatalog.compute_object_fingerprint(result)
    return result


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if plan["scope"] != {
        "explicit_opt_in_required": True,
        "training_forbidden": True,
        "downloads_forbidden": True,
        "old_holdout_read_forbidden": True,
        "automatic_retry": False,
        "os_file_cache_flush_forbidden": True,
        "weight_file_deletion_forbidden": True,
    }:
        raise ValueError("scope freeze does not match the v1 contract")
    conditions = {item["condition_id"]: item for item in plan["conditions"]}
    if set(conditions) != {"adapter_victim_reload", "no_eviction_control"}:
        raise ValueError("exactly one reload condition and one no-eviction control are required")
    if conditions["adapter_victim_reload"]["initial_resident_adapters"] != [FUTURE_ADAPTER]:
        raise ValueError("reload condition must start with only the future adapter")
    if set(conditions["no_eviction_control"]["initial_resident_adapters"]) != {
        CURRENT_ADAPTER,
        FUTURE_ADAPTER,
    }:
        raise ValueError("control must start with both adapters")
    order = list(plan["execution_order"])
    if len(order) != 2 or {row["condition_id"] for row in order} != set(conditions):
        raise ValueError("execution order must contain both conditions once")
    for sequence, row in enumerate(order, start=1):
        if row["sequence"] != sequence or sorted(row["arm_order"]) != sorted(ARMS):
            raise ValueError("execution order or arm order is not frozen")
    calls = plan["budget"]["generate_calls_per_condition"]
    per_condition = sum(int(calls[name]) for name in ("source", *ARMS))
    planned = len(order) * per_condition
    if per_condition != 6 or planned != 12:
        raise ValueError("v1 must freeze 6 calls per condition and 12 calls total")
    if planned > int(plan["budget"]["hard_generate_ceiling"]):
        raise ValueError("planned calls exceed hard ceiling")
    for condition in conditions.values():
        prediction = condition["prediction"]
        expected = min(
            ARMS,
            key=lambda arm: float(prediction[f"{arm}_completion_seconds"]),
        )
        if expected != prediction["predicted_cheaper_action"]:
            raise ValueError("frozen prediction is internally inconsistent")
    return {"planned_generate_calls": planned, "per_condition": per_condition}


def _resource_paths(plan: dict[str, Any]) -> dict[str, Path]:
    return {
        "base": Path(plan["resources"]["base"]["path"]),
        CURRENT_ADAPTER: Path(plan["resources"]["adapters"][CURRENT_ADAPTER]["path"]),
        FUTURE_ADAPTER: Path(plan["resources"]["adapters"][FUTURE_ADAPTER]["path"]),
        "source_input": Path(plan["source_input_provenance"]["path"]),
    }


def _verify_resources(plan: dict[str, Any], *, include_input: bool) -> dict[str, Any]:
    paths = _resource_paths(plan)
    checks = {
        "base_weight": (paths["base"] / "model.safetensors", plan["resources"]["base"]),
        "helmet_weight": (
            paths[CURRENT_ADAPTER] / "adapter_model.safetensors",
            plan["resources"]["adapters"][CURRENT_ADAPTER],
        ),
        "alpr_weight": (
            paths[FUTURE_ADAPTER] / "adapter_model.safetensors",
            plan["resources"]["adapters"][FUTURE_ADAPTER],
        ),
    }
    if include_input:
        checks["source_input"] = (paths["source_input"], plan["source_input_provenance"])
    result = {}
    for label, (path, expected) in checks.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual["bytes"] != int(expected["weight_bytes" if label != "source_input" else "bytes"]):
            raise RuntimeError(f"resource size drift: {label}")
        if actual["sha256"] != expected["weight_sha256" if label != "source_input" else "sha256"]:
            raise RuntimeError(f"resource hash drift: {label}")
        result[label] = actual
    return result


def _load_runtime(plan: dict[str, Any], initial_adapters: list[str]) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if not initial_adapters:
        raise ValueError("at least one initial adapter is required")
    paths = _resource_paths(plan)
    timings: dict[str, Any] = {"adapter_load_calls": []}
    mark = time.monotonic()
    processor = AutoProcessor.from_pretrained(
        str(paths["base"]), local_files_only=True, trust_remote_code=False
    )
    timings["processor_load_seconds"] = time.monotonic() - mark
    mark = time.monotonic()
    base = AutoModelForVision2Seq.from_pretrained(
        str(paths["base"]),
        torch_dtype=torch.float32,
        local_files_only=True,
        trust_remote_code=False,
    )
    base.to("cpu")
    base.eval()
    timings["base_load_seconds"] = time.monotonic() - mark
    first = initial_adapters[0]
    mark = time.monotonic()
    model = PeftModel.from_pretrained(
        base,
        str(paths[first]),
        adapter_name=first,
        is_trainable=False,
        local_files_only=True,
    )
    timings["adapter_load_calls"].append(
        {"adapter_name": first, "phase": "setup", "seconds": time.monotonic() - mark}
    )
    for name in initial_adapters[1:]:
        mark = time.monotonic()
        model.load_adapter(
            str(paths[name]),
            adapter_name=name,
            is_trainable=False,
            local_files_only=True,
        )
        timings["adapter_load_calls"].append(
            {"adapter_name": name, "phase": "setup", "seconds": time.monotonic() - mark}
        )
    model.eval()
    if loaded_adapter_names(model) != sorted(initial_adapters):
        raise RuntimeError("runtime initial adapter set does not match the frozen resident set")
    return processor, model, timings


def _workflow() -> dict[str, Any]:
    return {
        "workflow_id": "real_adapter_victim_reload_v1",
        "nodes": [{"node_id": name} for name in ("n0", "n1", "n2")],
        "edges": [["n0", "n1"], ["n1", "n2"]],
        "execution_order": ["n0", "n1", "n2"],
    }


def _state_identity(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_weight_sha256": plan["resources"]["base"]["weight_sha256"],
        "current_adapter": CURRENT_ADAPTER,
        "current_adapter_weight_sha256": plan["resources"]["adapters"][CURRENT_ADAPTER]["weight_sha256"],
        "future_adapter": FUTURE_ADAPTER,
        "future_adapter_weight_sha256": plan["resources"]["adapters"][FUTURE_ADAPTER]["weight_sha256"],
        "next_node": "n1",
    }


def _input_identity(plan: dict[str, Any]) -> dict[str, Any]:
    source = plan["source_input_provenance"]
    return {"path": source["path"], "bytes": source["bytes"], "sha256": source["sha256"]}


def _prepare_n2(plan: dict[str, Any], processor: Any, n1_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = plan["workflow"]["n2_prompt_template"].format(n1_output=n1_text)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    rendered = processor.apply_chat_template(messages, add_generation_prompt=True)
    tensors = dict(processor(text=rendered, return_tensors="pt"))
    record = input_record(
        prompt=prompt,
        rendered_prompt=rendered,
        input_ids=tensors["input_ids"][0].tolist(),
    )
    record["contains_n1_output"] = n1_text in prompt and n1_text in rendered
    return tensors, record


def _catalog(plan: dict[str, Any], condition: dict[str, Any]) -> AdapterCatalog:
    resources = plan["resources"]
    sizes = {
        BASE_ID: resources["base"]["weight_bytes"] / MIB,
        CURRENT_ADAPTER: resources["adapters"][CURRENT_ADAPTER]["weight_bytes"] / MIB,
        FUTURE_ADAPTER: resources["adapters"][FUTURE_ADAPTER]["weight_bytes"] / MIB,
    }
    family = "smolvlm_500m_local_weights"
    objects = [
        _typed_object(
            {
                "object_id": BASE_OBJECT,
                "object_type": "base_model",
                "version": resources["base"]["revision"],
                "resident_size_mb": sizes[BASE_ID],
                "transfer_size_mb": sizes[BASE_ID],
                "source": "existing_local_model_weight",
                "provenance": {"weight_sha256": resources["base"]["weight_sha256"]},
                "base_model_family": family,
                "base_model_id": BASE_ID,
                "required_base_model_id": None,
                "adapter_id": None,
                "workflow_identity": None,
                "shareability_scope": "all_compatible_adapters_at_rsu",
                "mutability": "immutable",
                "persistence": "measurement_process_resident",
                "evictability": "evictable",
                "migration_semantics": "local_weight_reload_measured_network_unavailable",
                "dependency_ids": [],
                "dataset_profile_source": "real_local_resource",
                "license_status": "declared_public_existing_local_copy",
                "formal_use_status": "non_formal_real_lifecycle_witness",
                "availability": "available",
                "counts_toward_capacity": True,
            }
        )
    ]
    for name in (CURRENT_ADAPTER, FUTURE_ADAPTER):
        item = resources["adapters"][name]
        objects.append(
            _typed_object(
                {
                    "object_id": ADAPTER_OBJECTS[name],
                    "object_type": "adapter",
                    "version": item["revision"],
                    "resident_size_mb": sizes[name],
                    "transfer_size_mb": sizes[name],
                    "source": "existing_local_adapter_weight",
                    "provenance": {"weight_sha256": item["weight_sha256"]},
                    "base_model_family": family,
                    "base_model_id": None,
                    "required_base_model_id": BASE_ID,
                    "adapter_id": name,
                    "workflow_identity": None,
                    "shareability_scope": "shared_base_named_adapter",
                    "mutability": "immutable",
                    "persistence": "measurement_process_resident",
                    "evictability": "evictable",
                    "migration_semantics": "local_weight_reload_measured_network_unavailable",
                    "dependency_ids": [BASE_OBJECT],
                    "dataset_profile_source": "real_local_resource",
                    "license_status": "declared_public_existing_local_copy",
                    "formal_use_status": "non_formal_real_lifecycle_witness",
                    "availability": "available",
                    "counts_toward_capacity": True,
                }
            )
        )
    initial = [BASE_OBJECT] + [ADAPTER_OBJECTS[name] for name in condition["initial_resident_adapters"]]
    raw = {
        "model_cache_profile_id": "typed_base_adapter_state_v1",
        "typed_model_cache_contract_version": "1.0.0",
        "vehicle_base_models": [{"base_model_id": BASE_ID, "family": family, "memory_mb": sizes[BASE_ID]}],
        "rsu_adapter_caches": [{"rsu_id": "rsu_target", "cached_adapter_ids": []}],
        "adapter_state_bundles": [],
        "cache_objects": [
            {
                "object_id": f"legacy:{name}",
                "adapter_id": name,
                "size_mb": sizes[name],
                "source": "real_lifecycle_legacy_view",
            }
            for name in (CURRENT_ADAPTER, FUTURE_ADAPTER)
        ],
        "typed_cache_objects": objects,
        "rsu_typed_cache_profiles": [{"rsu_id": "rsu_target", "resident_object_ids": initial}],
        "compatibility_map": {BASE_ID: [CURRENT_ADAPTER, FUTURE_ADAPTER]},
        "kv_prefix_enabled": False,
        "vehicle_adapter_residency_enabled": False,
        "model_cache_datasets": [],
    }
    return AdapterCatalog.from_dict(raw)


def _build_env(plan: dict[str, Any], condition: dict[str, Any]) -> VecWorkflowCoreEnv:
    bootstrap = WorkflowNode("n0", "prefix", BASE_ID, CURRENT_ADAPTER, 1, 1, [], ["n1"])
    workflow = WorkflowGraphState(
        workflow_id="real_adapter_victim_reload_v1",
        nodes=[bootstrap],
        edges=[],
        execution_order=["n0"],
        current_node_id="n0",
    )
    mobility = ReplayProvider(
        trajectory_frames=[
            {
                "time_index": index,
                "vehicles": [
                    {
                        "vehicle_id": "technical_vehicle",
                        "position_x": 0.0,
                        "position_y": 0.0,
                        "speed": 0.0,
                        "base_model_id": BASE_ID,
                        "active_workflow_id": workflow.workflow_id,
                    }
                ],
            }
            for index in range(3)
        ]
    )
    env = VecWorkflowCoreEnv(
        mobility_provider=mobility,
        workflow_state=workflow,
        adapter_catalog=_catalog(plan, condition),
        rsu_states=[RSUState("rsu_target", 0.0, 0.0, 1000.0)],
        max_steps=3,
        cache_capacity_profile={
            "model_cache_profile_id": "typed_base_adapter_state_v1",
            "enabled": True,
            "unit": "mb",
            "capacity_mb": float(condition["capacity_bytes"]) / MIB,
            "count_base_model_separately": True,
            "eviction_policy": "lru",
            "eviction_policy_seed": None,
            "typed_eviction_semantics": TYPED_EVICTION_SEMANTICS_SEQUENTIAL_LRU,
            "telemetry_enabled": True,
        },
    )
    env.reset()
    return env


def _control() -> ControlAction:
    return ControlAction(
        cache_action={"operation": "cache", "rsu_id": "rsu_target", "strategy": "real_lifecycle_opt_in"},
        offload_action={"mode": "rsu", "target_rsu_id": "rsu_target"},
        migration_action={"mode": "keep"},
        metadata={"action_id": 1, "action_name": "cache_current"},
    )


def _resident(env: VecWorkflowCoreEnv) -> list[str]:
    return list(env._typed_resident_object_ids["rsu_target"])


def _native_request(env: VecWorkflowCoreEnv, adapter_name: str, request_index: int) -> dict[str, Any]:
    env._episode_steps = request_index
    return env._apply_typed_cache_action(
        control=_control(),
        primary_vehicle=None,
        current_node_id=f"request_{request_index}_{adapter_name}",
        required_adapter=adapter_name,
    )


def _preview(plan: dict[str, Any], condition: dict[str, Any], history: list[str], requested: str) -> dict[str, Any]:
    shadow = _build_env(plan, condition)
    for index, prior in enumerate(history):
        replay = _native_request(shadow, prior, index)
        if replay["atomic_transaction_status"] not in {"committed", "noop_all_resident"}:
            raise RuntimeError("committed request could not be replayed in preview")
    return _native_request(shadow, requested, len(history))


def _apply_lifecycle_request(
    *,
    plan: dict[str, Any],
    condition: dict[str, Any],
    env: VecWorkflowCoreEnv,
    model: Any,
    history: list[str],
    requested: str,
) -> dict[str, Any]:
    paths = _resource_paths(plan)
    names = [CURRENT_ADAPTER, FUTURE_ADAPTER]
    before_ledger = _resident(env)
    before_runtime = runtime_snapshot(model, names)
    preview = _preview(plan, condition, history, requested)
    runtime_result = apply_previewed_adapter_transaction(
        model,
        preview=preview,
        adapter_paths={CURRENT_ADAPTER: paths[CURRENT_ADAPTER], FUTURE_ADAPTER: paths[FUTURE_ADAPTER]},
        object_to_adapter={value: key for key, value in ADAPTER_OBJECTS.items()},
        opt_in_enabled=True,
    )
    committed = _native_request(env, requested, len(history))
    for field in ("atomic_transaction_status", "evicted_object_ids", "admitted_typed_objects"):
        if committed.get(field) != preview.get(field):
            raise RuntimeError(f"native preview/commit mismatch for {field}")
    history.append(requested)
    after_ledger = _resident(env)
    after_runtime = runtime_snapshot(model, names)
    expected_runtime = sorted(
        row["adapter_id"]
        for row in (
            env._typed_object_row(object_id) for object_id in after_ledger
        )
        if row.get("adapter_id")
    )
    if after_runtime["registered_adapter_names"] != expected_runtime:
        raise RuntimeError("logical resident adapters diverged from runtime loaded adapters")
    return {
        "requested_adapter": requested,
        "logical_resident_before": before_ledger,
        "runtime_before": before_runtime,
        "legal_native_preview": preview,
        "runtime_application": runtime_result,
        "native_commit": committed,
        "logical_resident_after": after_ledger,
        "runtime_after": after_runtime,
    }


def _source(plan: dict[str, Any], condition: dict[str, Any], condition_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    static = _verify_resources(plan, include_input=True)
    processor, model, setup = _load_runtime(plan, [CURRENT_ADAPTER])
    model.set_adapter(CURRENT_ADAPTER)
    n0_tensors, n0_input = prepare_n0(plan, processor)
    n0 = generate(model, processor, n0_tensors, plan["generation"]["n0"])
    exported = export_action4_state(
        package_dir=condition_root / "production_action4_state_package",
        workflow=_workflow(),
        completed_nodes=["n0"],
        node_outputs={"n0": {"raw_text": n0["decoded_text"], "token_ids": n0["token_ids"]}},
        identity=_state_identity(plan),
        input_identity=_input_identity(plan),
        source_rsu_id="technical_source",
        target_rsu_id="rsu_target",
    )
    return {
        "status": "PASS",
        "role": "source",
        "condition_id": condition["condition_id"],
        "pid": os.getpid(),
        "process_wall_seconds": time.monotonic() - started,
        "static_resources": static,
        "setup": setup,
        "runtime": runtime_snapshot(model, [CURRENT_ADAPTER, FUTURE_ADAPTER]),
        "node_call_counts": {"n0": 1, "n1": 0, "n2": 0},
        "generate_attempted": 1,
        "generate_completed": 1,
        "n0_input": n0_input,
        "n0_output": n0,
        "production_action4_state_export": exported,
    }


def _arm(plan: dict[str, Any], condition: dict[str, Any], condition_root: Path, role: str) -> dict[str, Any]:
    process_started = time.monotonic()
    static_before = _verify_resources(plan, include_input=role == "restart")
    processor, model, setup = _load_runtime(plan, list(condition["initial_resident_adapters"]))
    env = _build_env(plan, condition)
    initial_ledger = _resident(env)
    initial_runtime = runtime_snapshot(model, [CURRENT_ADAPTER, FUTURE_ADAPTER])
    history: list[str] = []
    action_started = time.monotonic()
    current_event = _apply_lifecycle_request(
        plan=plan,
        condition=condition,
        env=env,
        model=model,
        history=history,
        requested=CURRENT_ADAPTER,
    )
    if role == "restart":
        n0_tensors, n0_input = prepare_n0(plan, processor)
        n0 = generate(model, processor, n0_tensors, plan["generation"]["n0"])
        restored = None
        state_restore = None
    else:
        imported = import_action4_state(
            package_dir=condition_root / "production_action4_state_package",
            expected_workflow=_workflow(),
            expected_identity=_state_identity(plan),
            expected_input_identity=_input_identity(plan),
            target_model_ready=(
                CURRENT_ADAPTER in loaded_adapter_names(model)
                and ADAPTER_OBJECTS[CURRENT_ADAPTER] in _resident(env)
            ),
            target_rsu_id="rsu_target",
        )
        if not imported.get("migration_success"):
            raise RuntimeError(f"state import failed: {imported}")
        restored = imported["node_outputs"]["n0"]
        n0_input = None
        n0 = None
        state_restore = imported
    n0_text = n0["decoded_text"] if n0 is not None else restored["raw_text"]
    model.set_adapter(CURRENT_ADAPTER)
    n1_tensors, n1_input = prepare_n1(plan, processor, n0_text)
    n1 = generate(model, processor, n1_tensors, plan["generation"]["n1"])
    future_event = _apply_lifecycle_request(
        plan=plan,
        condition=condition,
        env=env,
        model=model,
        history=history,
        requested=FUTURE_ADAPTER,
    )
    model.set_adapter(FUTURE_ADAPTER)
    n2_tensors, n2_input = _prepare_n2(plan, processor, n1["decoded_text"])
    n2 = generate(model, processor, n2_tensors, plan["generation"]["n2"])
    action_wall = time.monotonic() - action_started
    static_after = _verify_resources(plan, include_input=role == "restart")
    if static_before != static_after:
        raise RuntimeError("weight or input files changed during the lifecycle")
    counts = {"n0": 1 if role == "restart" else 0, "n1": 1, "n2": 1}
    attempted = sum(counts.values())
    return {
        "status": "PASS",
        "role": role,
        "condition_id": condition["condition_id"],
        "pid": os.getpid(),
        "process_wall_seconds": time.monotonic() - process_started,
        "action_wall_seconds": action_wall,
        "setup_excluded_from_action_window": setup,
        "static_resources_before": static_before,
        "static_resources_after": static_after,
        "os_file_cache": "uncontrolled_not_flushed",
        "initial_logical_residents": initial_ledger,
        "initial_runtime": initial_runtime,
        "current_request_event": current_event,
        "future_request_event": future_event,
        "final_logical_residents": _resident(env),
        "final_runtime": runtime_snapshot(model, [CURRENT_ADAPTER, FUTURE_ADAPTER]),
        "node_call_counts": counts,
        "generate_attempted": attempted,
        "generate_completed": attempted,
        "n0_input": n0_input,
        "n0_output": n0,
        "restored_n0_output": restored,
        "state_restore": state_restore,
        "n1_input": n1_input,
        "n1_output": n1,
        "n2_input": n2_input,
        "n2_output": n2,
        "task_correctness": "unavailable_technical_input",
    }


def _negative_checks(plan: dict[str, Any]) -> dict[str, Any]:
    condition = next(item for item in plan["conditions"] if item["condition_id"] == "adapter_victim_reload")
    env = _build_env(plan, condition)
    before = _resident(env)
    preview = _preview(plan, condition, [], CURRENT_ADAPTER)
    checks = []
    try:
        apply_previewed_adapter_transaction(
            object(),
            preview={**preview, "evicted_typed_objects": [{"object_id": BASE_OBJECT, "object_type": "base_model"}]},
            adapter_paths={},
            object_to_adapter={},
            opt_in_enabled=True,
        )
    except PeftAdapterLifecycleError as error:
        checks.append({"name": "dependency_or_base_victim_rejected", "status": "PASS", "error": str(error)})
    else:
        checks.append({"name": "dependency_or_base_victim_rejected", "status": "FAIL"})
    try:
        _native_request(env, "missing_adapter", 0)
    except ValueError as error:
        checks.append({"name": "missing_model_rejected", "status": "PASS", "error": str(error)})
    else:
        checks.append({"name": "missing_model_rejected", "status": "FAIL"})
    checks.append(
        {
            "name": "failed_requests_do_not_commit_logical_state",
            "status": "PASS" if _resident(env) == before else "FAIL",
            "before": before,
            "after": _resident(env),
        }
    )
    return {"status": "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL", "checks": checks}


def _child(role: str, condition_id: str, plan_path: Path, condition_root: Path) -> int:
    import torch

    torch.set_num_threads(1)
    receipt_path = condition_root / f"{role}_receipt.json"
    try:
        plan = read_json(plan_path)
        validate_plan(plan)
        condition = next(item for item in plan["conditions"] if item["condition_id"] == condition_id)
        receipt = _source(plan, condition, condition_root) if role == "source" else _arm(plan, condition, condition_root, role)
    except BaseException as error:
        receipt = {
            "status": "FAIL",
            "role": role,
            "condition_id": condition_id,
            "pid": os.getpid(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "automatic_retry": False,
        }
    write_json(receipt_path, receipt)
    print(receipt["status"], flush=True)
    return 0 if receipt["status"] == "PASS" else 1


def _run_child(role: str, condition_id: str, plan_path: Path, condition_root: Path, timeout: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--role",
        role,
        "--condition-id",
        condition_id,
        "--plan",
        str(plan_path),
        "--run-root",
        str(condition_root),
    ]
    started_wall = time.time()
    started = time.monotonic()
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        returncode, stdout, stderr, timed_out = completed.returncode, completed.stdout, completed.stderr, False
    except subprocess.TimeoutExpired as error:
        returncode = None
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
        timed_out = True
    receipt_path = condition_root / f"{role}_receipt.json"
    receipt = read_json(receipt_path) if receipt_path.exists() else {"status": "MISSING"}
    return {
        "role": role,
        "condition_id": condition_id,
        "command": command,
        "started_at_unix": started_wall,
        "ended_at_unix": time.time(),
        "wall_seconds": time.monotonic() - started,
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "receipt_status": receipt.get("status"),
        "pid": receipt.get("pid"),
    }


def _compare(source: dict[str, Any], restart: dict[str, Any], recovery: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "completion_counts_equal": restart["node_call_counts"] == {"n0": 1, "n1": 1, "n2": 1}
        and recovery["node_call_counts"] == {"n0": 0, "n1": 1, "n2": 1},
        "source_restart_n0_text_equal": source["n0_output"]["decoded_text"] == restart["n0_output"]["decoded_text"],
        "source_restart_n0_tokens_equal": source["n0_output"]["token_ids"] == restart["n0_output"]["token_ids"],
        "recovery_consumed_source_n0": recovery["restored_n0_output"] == {
            "raw_text": source["n0_output"]["decoded_text"],
            "token_ids": source["n0_output"]["token_ids"],
        },
        "n1_input_equal": restart["n1_input"] == recovery["n1_input"],
        "n1_output_tokens_equal": restart["n1_output"]["token_ids"] == recovery["n1_output"]["token_ids"],
        "n2_input_equal": restart["n2_input"] == recovery["n2_input"],
        "n2_output_tokens_equal": restart["n2_output"]["token_ids"] == recovery["n2_output"]["token_ids"],
        "current_native_events_equal": restart["current_request_event"]["native_commit"] == recovery["current_request_event"]["native_commit"],
        "future_native_events_equal": restart["future_request_event"]["native_commit"] == recovery["future_request_event"]["native_commit"],
        "no_weight_file_changes": restart["static_resources_before"] == restart["static_resources_after"]
        and recovery["static_resources_before"] == recovery["static_resources_after"],
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def _event_seconds(receipt: dict[str, Any], event_key: str, operation: str) -> float:
    return sum(
        float(row["seconds"])
        for row in receipt[event_key]["runtime_application"]["events"]
        if row["operation"] == operation
    )


def _model_load_calls(receipt: dict[str, Any]) -> int:
    return sum(
        row["operation"] == "load_adapter"
        for key in ("current_request_event", "future_request_event")
        for row in receipt[key]["runtime_application"]["events"]
    )


def _score(plan: dict[str, Any], condition: dict[str, Any], source: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    fixed = plan["network_assumption"]
    dynamic_bytes = (
        int(plan["source_input_provenance"]["bytes"])
        if arm["role"] == "restart"
        else int(source["production_action4_state_export"]["package_bytes"])
    )
    network = dynamic_bytes * 8.0 / (float(fixed["effective_link_mbps"]) * 1_000_000.0)
    if dynamic_bytes:
        network += float(fixed["positive_transfer_fixed_latency_seconds"])
    state_serialize = (
        float(source["production_action4_state_export"]["serialize_and_save_seconds"])
        if arm["role"] == "recovery"
        else 0.0
    )
    measured = float(arm["action_wall_seconds"]) + state_serialize + network
    predicted = float(condition["prediction"][f"{arm['role']}_completion_seconds"])
    return {
        "predicted_completion_seconds": predicted,
        "measured_action_wall_seconds": float(arm["action_wall_seconds"]),
        "state_serialize_and_save_seconds": state_serialize,
        "dynamic_transfer_bytes": dynamic_bytes,
        "simulated_network_seconds": network,
        "network_measured": False,
        "measured_scored_completion_seconds": measured,
        "signed_prediction_error_seconds": predicted - measured,
        "prefix_recompute_seconds": float(arm["n0_output"]["inference_seconds"]) if arm["n0_output"] else 0.0,
        "suffix_n1_seconds": float(arm["n1_output"]["inference_seconds"]),
        "future_n2_seconds": float(arm["n2_output"]["inference_seconds"]),
        "adapter_unload_seconds": _event_seconds(arm, "current_request_event", "unload_adapter")
        + _event_seconds(arm, "future_request_event", "unload_adapter"),
        "adapter_load_seconds": _event_seconds(arm, "current_request_event", "load_adapter")
        + _event_seconds(arm, "future_request_event", "load_adapter"),
        "post_setup_adapter_load_calls": _model_load_calls(arm),
    }


def execute(plan_path: Path, run_root: Path, expected_commit: str) -> int:
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite run root: {run_root}")
    actual_commit = current_commit()
    if actual_commit != expected_commit:
        raise RuntimeError(f"execution commit mismatch: expected {expected_commit}, got {actual_commit}")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
    if dirty:
        raise RuntimeError(f"execution worktree is not clean:\n{dirty}")
    plan = read_json(plan_path)
    validation = validate_plan(plan)
    conditions = {item["condition_id"]: item for item in plan["conditions"]}
    run_root.mkdir(parents=True)
    write_json(run_root / "frozen_plan.json", plan)
    negative = _negative_checks(plan)
    write_json(run_root / "preflight_negative_checks.json", negative)
    started_wall, started = time.time(), time.monotonic()
    rows = []
    processes = []
    terminal: dict[str, Any]
    try:
        if negative["status"] != "PASS":
            raise RuntimeError("preflight negative checks failed")
        for item in plan["execution_order"]:
            condition_id = item["condition_id"]
            condition_root = run_root / condition_id
            condition_root.mkdir()
            local_processes = []
            for role in ["source", *item["arm_order"]]:
                process = _run_child(
                    role,
                    condition_id,
                    plan_path,
                    condition_root,
                    int(plan["budget"]["process_timeout_seconds"]),
                )
                processes.append(process)
                local_processes.append(process)
                write_json(condition_root / f"{role}_process.json", process)
                if process["returncode"] != 0 or process["timed_out"] or process["receipt_status"] != "PASS":
                    raise RuntimeError(f"{condition_id}/{role} failed; no retry: {process}")
            source = read_json(condition_root / "source_receipt.json")
            restart = read_json(condition_root / "restart_receipt.json")
            recovery = read_json(condition_root / "recovery_receipt.json")
            comparison = _compare(source, restart, recovery)
            if comparison["status"] != "PASS":
                raise RuntimeError(f"{condition_id} fidelity/lifecycle comparison failed")
            scoring = {
                arm: _score(plan, conditions[condition_id], source, receipt)
                for arm, receipt in (("restart", restart), ("recovery", recovery))
            }
            measured_action = min(scoring, key=lambda arm: scoring[arm]["measured_scored_completion_seconds"])
            predicted_action = conditions[condition_id]["prediction"]["predicted_cheaper_action"]
            row = {
                "sequence": item["sequence"],
                "condition_id": condition_id,
                "arm_order": item["arm_order"],
                "processes": local_processes,
                "comparison": comparison,
                "generate_attempted": sum(x["generate_attempted"] for x in (source, restart, recovery)),
                "generate_completed": sum(x["generate_completed"] for x in (source, restart, recovery)),
                "predicted_cheaper_action": predicted_action,
                "measured_cheaper_action": measured_action,
                "decision_match": predicted_action == measured_action,
                "scoring": scoring,
                "lifecycle": {
                    arm: {
                        "initial_logical_residents": receipt["initial_logical_residents"],
                        "initial_runtime": receipt["initial_runtime"],
                        "current_request_event": receipt["current_request_event"],
                        "future_request_event": receipt["future_request_event"],
                        "final_logical_residents": receipt["final_logical_residents"],
                        "final_runtime": receipt["final_runtime"],
                    }
                    for arm, receipt in (("restart", restart), ("recovery", recovery))
                },
            }
            write_json(condition_root / "condition_receipt.json", row)
            rows.append(row)
        attempted = sum(row["generate_attempted"] for row in rows)
        completed = sum(row["generate_completed"] for row in rows)
        wall = time.monotonic() - started
        pids = [int(row["pid"]) for row in processes]
        budget_pass = (
            attempted == validation["planned_generate_calls"]
            and completed == validation["planned_generate_calls"]
            and attempted <= int(plan["budget"]["hard_generate_ceiling"])
            and wall <= float(plan["budget"]["scientific_total_timeout_seconds"])
        )
        write_json(run_root / "all_measurements.json", {"rows": rows})
        terminal = {
            "status": "PASS" if budget_pass and len(pids) == len(set(pids)) else "FAIL",
            "classification": "bounded real adapter cache lifecycle witness",
            "fixed_execution_commit": actual_commit,
            "plan_sha256": sha256_file(plan_path),
            "started_at_unix": started_wall,
            "finished_at_unix": time.time(),
            "scientific_wall_seconds": wall,
            "automatic_retry": False,
            "generate_attempted": attempted,
            "generate_completed": completed,
            "budget_compliance": "pass" if budget_pass else "fail",
            "all_process_ids_distinct": len(pids) == len(set(pids)),
            "negative_checks": negative,
            "rows": rows,
            "network_time_status": "frozen_assumption_only",
            "os_file_cache": "uncontrolled_not_flushed",
            "task_correctness": "unavailable_technical_input",
        }
    except BaseException as error:
        terminal = {
            "status": "FAIL",
            "fixed_execution_commit": actual_commit,
            "plan_sha256": sha256_file(plan_path),
            "started_at_unix": started_wall,
            "finished_at_unix": time.time(),
            "scientific_wall_seconds": time.monotonic() - started,
            "automatic_retry": False,
            "completed_conditions": len(rows),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    write_json(run_root / "terminal_receipt.json", terminal)
    files = inventory_tree(run_root, exclude={"integrity_manifest.json"})
    write_json(
        run_root / "integrity_manifest.json",
        {
            "schema_version": "ppo_mec.artifact_integrity.v1",
            "run_root": str(run_root),
            "file_count": len(files),
            "total_bytes_excluding_manifest": sum(item["bytes"] for item in files),
            "files": files,
        },
    )
    return 0 if terminal["status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["supervisor", "source", *ARMS], required=True)
    parser.add_argument("--condition-id")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    plan_path, run_root = args.plan.resolve(), args.run_root.resolve()
    if args.role == "supervisor":
        if not args.expected_commit:
            raise SystemExit("--expected-commit is required")
        return execute(plan_path, run_root, args.expected_commit)
    if not args.condition_id:
        raise SystemExit("--condition-id is required")
    return _child(args.role, args.condition_id, plan_path, run_root)


if __name__ == "__main__":
    raise SystemExit(main())
