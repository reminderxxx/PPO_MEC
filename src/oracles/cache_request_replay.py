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
FORMAL_LIFECYCLE_REQUEST_REPLAY_PRODUCER_VERSION = (
    "policy_neutral_formal_lifecycle_replay_v2"
)
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
TYPED_REQUEST_FIELDS = {
    "model_cache_profile_id",
    "typed_model_cache_contract_version",
    "catalog_fingerprint",
    "requested_typed_objects",
    "dependency_bundle",
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
    request_rows = [deepcopy(item) for item in requests]
    typed_mode = bool(request_rows) and all(
        item.get("model_cache_profile_id") == "typed_base_adapter_state_v1"
        for item in request_rows
    )
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
            "model_cache_profile_id": (
                "typed_base_adapter_state_v1" if typed_mode else "legacy_adapter_only_v1"
            ),
            "atomic_dependency_bundle": typed_mode,
            "max_dependency_bundle_objects": 2 if typed_mode else 1,
        },
        "requests": request_rows,
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
    if producer.get("identity") == FORMAL_LIFECYCLE_REQUEST_REPLAY_PRODUCER_VERSION:
        from src.runtime.formal_exogenous_request_execution import (
            FORMAL_REQUEST_EXPOSURE_TRACE_VERSION,
            FORMAL_REQUEST_SUBJECT_LIFECYCLE_CONTRACT_VERSION,
            FORMAL_REQUEST_SUBJECT_SELECTION_VERSION,
            REQUIRED_SUBJECT_LIFECYCLE_FIELDS,
        )

        lifecycle = replay.get("formal_request_subject_lifecycle")
        semantics = replay.get("request_semantics") or {}
        fingerprint = replay.get("formal_request_exposure_fingerprint")
        if not isinstance(lifecycle, dict) or set(lifecycle) != set(
            REQUIRED_SUBJECT_LIFECYCLE_FIELDS
        ):
            errors.append("formal request replay lifecycle evidence fields drift")
        else:
            if lifecycle.get("contract_version") != (
                FORMAL_REQUEST_SUBJECT_LIFECYCLE_CONTRACT_VERSION
            ):
                errors.append("formal request replay lifecycle version mismatch")
            if lifecycle.get("selection_version") != (
                FORMAL_REQUEST_SUBJECT_SELECTION_VERSION
            ):
                errors.append("formal request replay selection version mismatch")
            if lifecycle.get("reselection_policy") != (
                "forbidden_during_formal_episode"
            ):
                errors.append("formal request replay reselection policy drift")
            if lifecycle.get("selection_evidence_actor_visible") is not False or lifecycle.get(
                "selection_evidence_controller_visible"
            ) is not False:
                errors.append("formal request replay lifecycle evidence is actor-visible")
            if lifecycle.get("outcome_independence") is not True:
                errors.append("formal request replay lifecycle is outcome-dependent")
        if semantics.get("formal_request_exposure_trace_version") != (
            FORMAL_REQUEST_EXPOSURE_TRACE_VERSION
        ):
            errors.append("formal request replay exposure trace version mismatch")
        if semantics.get("formal_request_subject_lifecycle_contract_version") != (
            FORMAL_REQUEST_SUBJECT_LIFECYCLE_CONTRACT_VERSION
        ):
            errors.append("formal request replay lifecycle contract version mismatch")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            errors.append("formal request replay exposure fingerprint is missing")
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
        if request.get("model_cache_profile_id") is not None:
            typed_missing = sorted(TYPED_REQUEST_FIELDS - set(request))
            if typed_missing:
                errors.append(f"request[{index}] missing typed fields: {typed_missing}")
            elif request.get("model_cache_profile_id") != "typed_base_adapter_state_v1":
                errors.append(f"request[{index}] unsupported typed profile")
            elif request.get("typed_model_cache_contract_version") != "1.0.0":
                errors.append(f"request[{index}] typed contract version mismatch")
            else:
                bundle = request.get("dependency_bundle") or {}
                rows = request.get("requested_typed_objects") or []
                ordered_ids = list(bundle.get("ordered_object_ids") or [])
                if not isinstance(rows, list) or not 1 <= len(rows) <= 2:
                    errors.append(f"request[{index}] typed object bundle must contain 1-2 objects")
                elif ordered_ids != [row.get("object_id") for row in rows]:
                    errors.append(f"request[{index}] dependency bundle order mismatch")
                elif rows[-1].get("object_type") != "adapter":
                    errors.append(f"request[{index}] dependency bundle must end with adapter")
                elif len(rows) == 2 and rows[0].get("object_type") != "base_model":
                    errors.append(f"request[{index}] dependency bundle must admit base before adapter")
                for row in rows:
                    try:
                        resident_size = float(row.get("resident_size_mb"))
                        transfer_size = float(row.get("transfer_size_mb"))
                        if not math.isfinite(resident_size) or resident_size <= 0 or not math.isfinite(transfer_size) or transfer_size <= 0:
                            raise ValueError
                    except (TypeError, ValueError):
                        errors.append(f"request[{index}] invalid typed object size")
                        break
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
    if producer.get("identity") == FORMAL_LIFECYCLE_REQUEST_REPLAY_PRODUCER_VERSION:
        lifecycle = replay.get("formal_request_subject_lifecycle") or {}
        selected_vehicle_id = lifecycle.get("selected_primary_vehicle_id")
        if lifecycle.get("exposure_horizon") != len(requests):
            errors.append("formal request replay exposure horizon mismatch")
        if any(request.get("vehicle_id") != selected_vehicle_id for request in requests):
            errors.append("formal request replay vehicle differs from frozen subject")
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
    """Reuse the frozen formal exposure producer without executing a policy."""
    from src.evaluators.main_results_support import (
        build_episode_formal_request_exposure,
        build_selected_workflow_states,
        load_window_bundle,
    )

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
    catalog_path = _input_path(manifest, "ppo_mec_sample_adapter_catalog", root_path)
    from src.data.model_catalog.adapter_catalog import AdapterCatalog

    catalog = AdapterCatalog.from_json(catalog_path)
    exposure = build_episode_formal_request_exposure(
        workflow_state=workflow,
        mobility_bundle=bundle,
        adapter_catalog=catalog,
        max_steps=int(unit["max_steps"]),
        mobility_source="ngsim",
        primary_vehicle_selection=str(
            filters.get("primary_vehicle_selection") or unit["vehicle_selection"]
        ),
        cache_capacity_profile={
            "model_cache_profile_id": "typed_base_adapter_state_v1",
            "enabled": True,
            "unit": "mb",
            "capacity_mb": 288.0,
            "count_base_model_separately": True,
            "eviction_policy": "lru",
            "eviction_policy_seed": int(unit["benchmark_run_seed"]),
            "telemetry_enabled": True,
        },
        evaluation_unit=unit,
        source_provenance={
            "phase": "analytical_request_replay",
            "formal": False,
            "training": False,
            "performance_evidence": False,
            "manifest_id": manifest["identity"]["manifest_id"],
        },
    )
    requests: list[dict[str, Any]] = []
    episode_id = f"policy-neutral/{unit['evaluation_unit_id']}"
    for index, formal_request in enumerate(exposure["requests"]):
        current_rsu = formal_request["current_service_rsu_id"]
        request_rsu = formal_request["request_rsu_id"]
        previous_rsu = (
            exposure["requests"][index - 1]["request_rsu_id"]
            if index > 0
            else None
        )
        next_rsu = formal_request["oracle_only_future_topology"][
            "actual_next_rsu_id"
        ]
        requests.append(
            {
                "request_id": formal_request["request_id"],
                "evaluation_unit_id": unit["evaluation_unit_id"],
                "episode_id": episode_id,
                "step_index": formal_request["step_index"],
                "time_index": formal_request["time_index"],
                "request_order": formal_request["request_order"],
                "vehicle_id": formal_request["vehicle_id"],
                "workflow_id": formal_request["workflow_id"],
                "node_id": formal_request["node_id"],
                "required_base_model": formal_request["required_base_model"],
                "object_id": formal_request["object_id"],
                "adapter_id": formal_request["adapter_id"],
                "object_size_mb": formal_request["object_size_mb"],
                "size_source": "typed_catalog",
                "request_rsu_id": request_rsu,
                "current_service_rsu_id": current_rsu,
                "previous_rsu_id": previous_rsu,
                "actual_next_rsu_id": next_rsu,
                "predicted_next_rsu_id": None,
                "actual_handoff_target_rsu_id": current_rsu if current_rsu != request_rsu else None,
                "predicted_handoff_target_rsu_id": None,
                "eligible_service_rsu_ids": formal_request[
                    "eligible_service_rsu_ids"
                ],
                "eligible_cache_target_rsu_ids": formal_request[
                    "eligible_cache_target_rsu_ids"
                ],
                "dag_provenance": {
                    "workflow_dag_sha256": unit["workflow_dag_sha256"],
                    **formal_request["dag_provenance"],
                    "policy_neutral_progression": True,
                },
                "model_cache_profile_id": formal_request[
                    "model_cache_profile_id"
                ],
                "typed_model_cache_contract_version": formal_request[
                    "typed_model_cache_contract_version"
                ],
                "catalog_fingerprint": formal_request["catalog_fingerprint"],
                "requested_typed_objects": formal_request[
                    "requested_typed_objects"
                ],
                "dependency_bundle": formal_request["dependency_bundle"],
            }
        )
    replay = build_request_replay(
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
    replay["formal_request_subject_lifecycle"] = deepcopy(
        exposure["subject_lifecycle"]
    )
    replay["producer"]["identity"] = (
        FORMAL_LIFECYCLE_REQUEST_REPLAY_PRODUCER_VERSION
    )
    replay["formal_request_exposure_fingerprint"] = exposure[
        "request_exposure_fingerprint"
    ]
    replay["request_semantics"].update(
        formal_request_exposure_trace_version=exposure[
            "formal_request_exposure_trace_version"
        ],
        formal_request_subject_lifecycle_contract_version=exposure[
            "formal_request_subject_lifecycle_contract_version"
        ],
        subject_reselection_policy="forbidden_during_formal_episode",
    )
    replay["request_replay_fingerprint"] = request_replay_fingerprint(replay)
    report = validate_request_replay(replay, source_manifest=manifest)
    if report["status"] != "pass":
        raise CacheRequestReplayError("; ".join(report["errors"]))
    return replay
