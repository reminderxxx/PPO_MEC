"""Versioned, policy-neutral cache request replay contract."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from src.evaluators.cache_baseline_fairness import (
    FairnessManifestError,
    sha256_value,
    validate_manifest,
    workload_fingerprint,
)


CACHE_REQUEST_REPLAY_VERSION = "1.0.0"
REQUEST_REPLAY_PRODUCER_VERSION = "policy_neutral_dag_mobility_replay_v1"
FORBIDDEN_INPUT_FIELDS = {
    "actual_cache_hit",
    "cache_hit",
    "hit_source",
    "actual_victim",
    "evicted_object_ids",
    "actual_admission",
    "admission_added",
    "actual_policy_state",
    "reward",
    "actual_execution_result",
    "service_success",
    "stall_occurred",
    "baseline_cache_contents",
}
REQUIRED_REQUEST_FIELDS = {
    "request_id",
    "evaluation_unit_id",
    "episode_id",
    "step_index",
    "time_index",
    "request_order",
    "vehicle_id",
    "workflow_id",
    "node_id",
    "required_base_model",
    "object_id",
    "adapter_id",
    "object_size_mb",
    "size_source",
    "request_rsu_id",
    "current_service_rsu_id",
    "previous_rsu_id",
    "actual_next_rsu_id",
    "predicted_next_rsu_id",
    "actual_handoff_target_rsu_id",
    "predicted_handoff_target_rsu_id",
    "eligible_service_rsu_ids",
    "eligible_cache_target_rsu_ids",
    "dag_provenance",
}


class CacheRequestReplayError(ValueError):
    """Raised when request replay is unsafe or incompatible."""


def _version_major(version: Any) -> int:
    try:
        return int(str(version).split(".", 1)[0])
    except (TypeError, ValueError) as exc:
        raise CacheRequestReplayError(f"invalid cache request replay version: {version!r}") from exc


def _validate_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise CacheRequestReplayError(f"non-finite JSON value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_INPUT_FIELDS:
                raise CacheRequestReplayError(f"policy outcome field forbidden in oracle input: {path}.{key}")
            _validate_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")


def request_replay_fingerprint(replay: dict[str, Any]) -> str:
    payload = deepcopy(replay)
    payload.pop("request_replay_fingerprint", None)
    payload.pop("validation", None)
    return sha256_value(payload)


def build_request_replay(
    *,
    requests: Iterable[dict[str, Any]],
    evaluation_unit: dict[str, Any],
    source_manifest: dict[str, Any],
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    unit_id = str(evaluation_unit["evaluation_unit_id"])
    replay = {
        "cache_request_replay_version": CACHE_REQUEST_REPLAY_VERSION,
        "producer": producer
        or {
            "identity": REQUEST_REPLAY_PRODUCER_VERSION,
            "policy_neutral": True,
            "executes_cache_policy": False,
            "executes_service_policy": False,
        },
        "source_manifest": {
            "manifest_id": source_manifest["identity"]["manifest_id"],
            "full_manifest_sha256": source_manifest["hashes"]["full_manifest_sha256"],
            "semantic_protocol_sha256": source_manifest["hashes"]["semantic_protocol_sha256"],
            "git_commit": source_manifest["identity"]["git_commit"],
        },
        "evaluation_unit": {
            "evaluation_unit_id": unit_id,
            "benchmark_run_seed": int(evaluation_unit["benchmark_run_seed"]),
            "window_id": str(evaluation_unit["window_id"]),
            "workflow_id": str(evaluation_unit["workflow_id"]),
            "workflow_dag_sha256": str(evaluation_unit["workflow_dag_sha256"]),
            "expected_workload_fingerprint": str(evaluation_unit["expected_workload_fingerprint"]),
            "raw_frame_interval": deepcopy(evaluation_unit["raw_frame_interval"]),
            "raw_time_interval": deepcopy(evaluation_unit["raw_time_interval"]),
        },
        "request_semantics": {
            "request_source": "policy_neutral_static_dag_execution_order_plus_mobility_topology",
            "outcome_fields_forbidden": sorted(FORBIDDEN_INPUT_FIELDS),
            "admission_timing": "action_before_service_lookup_same_step_admission_can_hit",
            "action_budget": "at_most_one_current_object_admission_per_step",
            "oracle_control_scope": "current_rsu_placement_admission_eviction_or_noop",
        },
        "requests": [deepcopy(item) for item in requests],
    }
    _validate_finite(replay)
    replay["request_replay_fingerprint"] = request_replay_fingerprint(replay)
    report = validate_request_replay(replay, source_manifest=source_manifest)
    if report["status"] != "pass":
        raise CacheRequestReplayError("; ".join(report["errors"]))
    return replay


def validate_request_replay(
    replay: dict[str, Any], *, source_manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if _version_major(replay.get("cache_request_replay_version")) != 1:
            errors.append("unsupported cache request replay major version")
        _validate_finite(replay)
    except CacheRequestReplayError as exc:
        errors.append(str(exc))
    producer = replay.get("producer") or {}
    if producer.get("policy_neutral") is not True or producer.get("executes_cache_policy") is not False:
        errors.append("request replay producer must be policy-neutral and must not execute cache policy")
    requests = replay.get("requests")
    if not isinstance(requests, list):
        errors.append("requests must be a list")
        requests = []
    seen: set[str] = set()
    last_order: tuple[int, int] | None = None
    unit_id = str((replay.get("evaluation_unit") or {}).get("evaluation_unit_id"))
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            errors.append(f"request[{index}] must be an object")
            continue
        missing = sorted(REQUIRED_REQUEST_FIELDS - set(request))
        if missing:
            errors.append(f"request[{index}] missing fields: {missing}")
            continue
        request_id = str(request["request_id"])
        if request_id in seen:
            errors.append(f"duplicate request_id: {request_id}")
        seen.add(request_id)
        if str(request["evaluation_unit_id"]) != unit_id:
            errors.append(f"request[{index}] evaluation unit mismatch")
        try:
            size = float(request["object_size_mb"])
            if not math.isfinite(size) or size <= 0:
                errors.append(f"request[{index}] object_size_mb must be finite and positive")
            order = (int(request["step_index"]), int(request["request_order"]))
            if last_order is not None and order <= last_order:
                errors.append("requests must be in strictly increasing (step_index, request_order) order")
            last_order = order
        except (TypeError, ValueError):
            errors.append(f"request[{index}] has invalid numeric fields")
        for field in ("eligible_service_rsu_ids", "eligible_cache_target_rsu_ids"):
            value = request.get(field)
            if not isinstance(value, list) or value != sorted(set(value)):
                errors.append(f"request[{index}].{field} must be a canonical sorted unique list")
        if request.get("current_service_rsu_id") is not None and request["current_service_rsu_id"] not in request["eligible_service_rsu_ids"]:
            errors.append(f"request[{index}] current_service_rsu_id is not service-eligible")
    expected = replay.get("request_replay_fingerprint")
    if not isinstance(expected, str) or expected != request_replay_fingerprint(replay):
        errors.append("request replay canonical fingerprint mismatch")
    if source_manifest is not None:
        source = replay.get("source_manifest") or {}
        expected_source = {
            "manifest_id": source_manifest["identity"]["manifest_id"],
            "full_manifest_sha256": source_manifest["hashes"]["full_manifest_sha256"],
            "semantic_protocol_sha256": source_manifest["hashes"]["semantic_protocol_sha256"],
            "git_commit": source_manifest["identity"]["git_commit"],
        }
        if source != expected_source:
            errors.append("request replay source manifest identity mismatch")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "request_count": len(requests),
        "request_replay_fingerprint": expected,
        "json_round_trip_stable": (
            json.loads(json.dumps(replay, ensure_ascii=False, allow_nan=False)) == replay
            if not errors
            else False
        ),
    }


def load_and_validate_request_replay(
    path: str | Path, *, source_manifest: dict[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    replay = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    report = validate_request_replay(replay, source_manifest=source_manifest)
    if report["status"] != "pass":
        raise CacheRequestReplayError("; ".join(report["errors"]))
    return replay, report


def missing_replay_status() -> dict[str, Any]:
    return {
        "availability": "unavailable",
        "reason": "cache_request_replay_missing; aggregates and observed outcomes cannot reconstruct it",
        "request_replay_fingerprint": None,
    }


def _input_path(manifest: dict[str, Any], logical_id: str, root: Path) -> Path:
    entry = next(
        (item for item in manifest["dataset_provenance"]["inputs"] if item["logical_dataset_id"] == logical_id),
        None,
    )
    if entry is None:
        raise CacheRequestReplayError(f"manifest input missing: {logical_id}")
    path = Path(entry.get("normalized_absolute_path") or entry["logical_path"])
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def build_policy_neutral_replay_from_manifest(
    *, root: str | Path, manifest: dict[str, Any], evaluation_unit_id: str
) -> dict[str, Any]:
    """Reuse the formal workflow/provider/mapper path without executing a policy."""
    from src.data.mobility.rsu_mapper import RSUMapper
    from src.evaluators.main_results_support import build_selected_workflow_states, load_window_bundle

    root_path = Path(root).resolve()
    manifest_report = validate_manifest(manifest, root=root_path, check_files=True)
    if manifest_report["status"] != "pass":
        raise FairnessManifestError("; ".join(manifest_report["errors"]))
    unit = next(
        (item for item in manifest["window_workload_plan"]["evaluation_units"] if item["evaluation_unit_id"] == evaluation_unit_id),
        None,
    )
    if unit is None:
        raise CacheRequestReplayError(f"unknown G07 evaluation unit: {evaluation_unit_id}")
    filters = manifest["dataset_provenance"]["selection_filter_parameters"]
    workflow_path = _input_path(manifest, "alibaba_cluster_trace_2018_batch_task", root_path)
    workflows = build_selected_workflow_states(
        workflow_csv_path=workflow_path,
        max_workflows=int(filters["max_workflows"]),
        workflow_selector=str(filters["workflow_selector"]),
        min_tasks=int(filters["min_tasks"]),
        max_tasks=int(filters["max_tasks"]),
        random_seed=int(unit["benchmark_run_seed"]),
    )
    workflow = next((item for item in workflows if item.workflow_id == unit["workflow_id"]), None)
    if workflow is None or workload_fingerprint(workflow) != unit["expected_workload_fingerprint"]:
        raise CacheRequestReplayError("workflow reconstruction does not match G07 expected workload fingerprint")
    mobility_path = _input_path(manifest, "ngsim_vehicle_trajectories", root_path)
    frame_interval = unit["raw_frame_interval"]
    bundle = load_window_bundle(
        root_dir=root_path,
        mobility_source="ngsim",
        mobility_csv_path=str(mobility_path),
        lust_scenario_root="",
        max_mobility_rows=int(filters["max_mobility_rows"]),
        rsu_layout=str(unit["rsu_layout"]),
        frame_offset=int(frame_interval["start"]),
        window_length=int(frame_interval["end"]) - int(frame_interval["start"]) + 1,
        random_seed=int(unit["benchmark_run_seed"]),
    )
    frames = bundle.frames
    if not frames:
        raise CacheRequestReplayError("mobility window contains no frames")
    mapper = RSUMapper(bundle.rsu_states)
    association_maps = [mapper.associate(frame.get("vehicles", [])) for frame in frames]
    first_ids = sorted(vehicle.vehicle_id for vehicle in frames[0].get("vehicles", []))
    if not first_ids:
        raise CacheRequestReplayError("mobility window has no primary vehicle candidate")
    vehicle_id = first_ids[0]
    catalog_path = _input_path(manifest, "ppo_mec_sample_adapter_catalog", root_path)
    from src.data.model_catalog.adapter_catalog import AdapterCatalog

    catalog = AdapterCatalog.from_json(catalog_path)
    node_map = {node.node_id: node for node in workflow.nodes}
    max_requests = min(int(unit["max_steps"]), len(workflow.execution_order), max(len(frames) - 1, 0))
    requests: list[dict[str, Any]] = []
    episode_id = f"policy-neutral/{unit['evaluation_unit_id']}"
    for offset, node_id in enumerate(workflow.execution_order[:max_requests], start=1):
        node = node_map[node_id]
        frame_index = offset
        request_rsu = association_maps[frame_index - 1].get(vehicle_id)
        current_rsu = association_maps[frame_index].get(vehicle_id)
        previous_rsu = association_maps[frame_index - 2].get(vehicle_id) if frame_index >= 2 else None
        next_rsu = association_maps[frame_index + 1].get(vehicle_id) if frame_index + 1 < len(frames) else None
        resolution = catalog.resolve_adapter_resident_size_mb(node.required_adapter)
        cache_object = next((item for item in catalog.cache_objects if item.adapter_id == node.required_adapter), None)
        eligible = [current_rsu] if current_rsu is not None else []
        requests.append(
            {
                "request_id": f"{unit['evaluation_unit_id']}/request_{offset:06d}",
                "evaluation_unit_id": unit["evaluation_unit_id"],
                "episode_id": episode_id,
                "step_index": offset,
                "time_index": int(frames[frame_index]["time_index"]),
                "request_order": offset - 1,
                "vehicle_id": vehicle_id,
                "workflow_id": workflow.workflow_id,
                "node_id": node.node_id,
                "required_base_model": node.required_base_model,
                "object_id": cache_object.object_id if cache_object else f"adapter:{node.required_adapter}",
                "adapter_id": node.required_adapter,
                "object_size_mb": float(resolution.size_mb),
                "size_source": resolution.source,
                "request_rsu_id": request_rsu,
                "current_service_rsu_id": current_rsu,
                "previous_rsu_id": previous_rsu,
                "actual_next_rsu_id": next_rsu,
                "predicted_next_rsu_id": None,
                "actual_handoff_target_rsu_id": current_rsu if current_rsu != request_rsu else None,
                "predicted_handoff_target_rsu_id": None,
                "eligible_service_rsu_ids": sorted(eligible),
                "eligible_cache_target_rsu_ids": sorted(eligible),
                "dag_provenance": {
                    "workflow_dag_sha256": unit["workflow_dag_sha256"],
                    "execution_order_index": offset - 1,
                    "predecessors": sorted(node.predecessors),
                    "successors": sorted(node.successors),
                    "policy_neutral_progression": True,
                },
            }
        )
    return build_request_replay(
        requests=requests,
        evaluation_unit=unit,
        source_manifest=manifest,
        producer={
            "identity": REQUEST_REPLAY_PRODUCER_VERSION,
            "policy_neutral": True,
            "executes_cache_policy": False,
            "executes_service_policy": False,
            "workflow_provider": "WorkflowDatasetBuilder.build_selected_alibaba_workflow_states",
            "mobility_provider": "load_window_bundle/NGSIMProvider/ReplayProvider",
            "rsu_mapper": "RSUMapper",
        },
    )
