"""Fail-closed loader for a frozen observed 4-RSU/8-vehicle scenario."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.data.workflow.workflow_dataset_builder import WorkflowDatasetBuilder
from src.envs.core.vec_competition_contract import (
    CONTRACT_VERSION, BackhaulLink, FrozenScenario, RequestSpec, RSUConfig,
    TraceSample,
)
from src.evaluators.vec_competition_scenario import FEET_TO_METRES, is_lfs_pointer, sha256_file


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_path(root: Path, relative: str) -> Path:
    candidate = root / relative
    if candidate.is_file() and not is_lfs_pointer(candidate):
        return candidate
    common = subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=root, text=True).strip()
    primary = Path(common).resolve().parent / relative
    if primary.is_file() and not is_lfs_pointer(primary):
        return primary
    raise FileNotFoundError(f"materialized local source unavailable: {relative}")


def load_frozen_scenario(profile_path: Path, root: Path) -> tuple[FrozenScenario, dict[str, Any]]:
    profile_path = profile_path.resolve()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("unsupported frozen scenario contract version")
    mobility = profile["mobility"]
    workflow_spec = profile["workflow"]
    catalog_spec = profile["model_catalog"]
    sources = {
        "mobility": _source_path(root, mobility["relative_path"]),
        "workflow": _source_path(root, workflow_spec["relative_path"]),
        "catalog": _source_path(root, catalog_spec["relative_path"]),
    }
    for label, spec in (("mobility", mobility), ("workflow", workflow_spec), ("catalog", catalog_spec)):
        actual_hash = sha256_file(sources[label])
        if actual_hash != spec["sha256"]:
            raise ValueError(f"source hash mismatch: {label} expected={spec['sha256']} actual={actual_hash}")
    vehicle_ids = tuple(mobility["vehicle_ids"])
    if len(vehicle_ids) != 8 or len(set(vehicle_ids)) != 8:
        raise ValueError("frozen profile requires eight unique vehicle IDs")
    first_ms = int(mobility["first_time_ms"])
    tick_ms = int(mobility["frame_period_ms"])
    frame_count = int(mobility["frame_count"])
    last_ms = first_ms + (frame_count - 1) * tick_ms
    rows: dict[tuple[str, int], list[dict[str, str]]] = {}
    with sources["mobility"].open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"Vehicle_ID", "Location", "Global_Time", "Local_X", "Local_Y", "v_Vel", "Section_ID", "Lane_ID", "Direction"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"NGSIM source fields missing: {sorted(required - set(reader.fieldnames or []))}")
        for index, row in enumerate(reader):
            if index >= int(mobility["rows_to_scan"]):
                break
            if row["Location"].strip().lower() != mobility["source_segment_id"]:
                continue
            vehicle_id = f"{mobility['source_segment_id']}:{row['Vehicle_ID']}"
            if vehicle_id not in vehicle_ids:
                continue
            time_ms = int(row["Global_Time"].replace(",", ""))
            if first_ms <= time_ms <= last_ms:
                rows.setdefault((vehicle_id, time_ms), []).append(row)
    traces: dict[str, tuple[TraceSample, ...]] = {}
    for vehicle_id in vehicle_ids:
        points: list[TraceSample] = []
        for frame in range(frame_count):
            time_ms = first_ms + frame * tick_ms
            records = rows.get((vehicle_id, time_ms), [])
            if len(records) != 1:
                raise ValueError(f"frozen trace identity violation: {vehicle_id} time_ms={time_ms} row_count={len(records)}")
            row = records[0]
            points.append(TraceSample(
                time_s=round(frame * tick_ms / 1000, 9),
                source_time_ms=time_ms,
                x_m=float(row["Local_X"].replace(",", "")) * FEET_TO_METRES,
                y_m=float(row["Local_Y"].replace(",", "")) * FEET_TO_METRES,
                speed_mps=abs(float(row["v_Vel"].replace(",", ""))) * FEET_TO_METRES,
                road_key=f"section:{row['Section_ID']}|lane:{row['Lane_ID']}|direction:{row['Direction']}",
                direction=row["Direction"],
            ))
        traces[vehicle_id] = tuple(points)
    builder = WorkflowDatasetBuilder()
    workflow_states = builder.build_alibaba_workflow_states(
        sources["workflow"],
        limit_jobs=int(workflow_spec["scan_limit_jobs"]),
        min_tasks=int(workflow_spec["min_tasks"]),
        max_tasks=int(workflow_spec["max_tasks"]),
    )
    by_id = {workflow.workflow_id: workflow for workflow in workflow_states}
    assignments = workflow_spec["assignments"]
    if set(assignments) != set(vehicle_ids) or len(set(assignments.values())) != 8:
        raise ValueError("frozen workflow assignments must be one-to-one for eight vehicles")
    try:
        workflows = {vehicle_id: by_id[assignments[vehicle_id]] for vehicle_id in vehicle_ids}
    except KeyError as error:
        raise ValueError(f"frozen Alibaba job unavailable: {error}") from error
    catalog = AdapterCatalog.from_json(sources["catalog"])
    request_spec = profile["requests"]
    if request_spec.get("count") != 8:
        raise ValueError("request count must be eight")
    requests = {vehicle_id: RequestSpec(vehicle_id, float(request_spec["arrival_offset_s"]), float(request_spec["deadline_offset_s"])) for vehicle_id in vehicle_ids}
    rsus = tuple(RSUConfig(**item) for item in profile["rsus"])
    links = tuple(BackhaulLink(item["from"], item["to"], float(item["bytes_per_s"]), float(item["latency_s"])) for item in profile["backhaul"]["links"])
    scenario = FrozenScenario(
        scenario_id=profile["scenario_id"],
        source_segment_id=mobility["source_segment_id"],
        traces=traces,
        workflows=workflows,
        requests=requests,
        rsus=rsus,
        backhaul_links=links,
        cloud_port_bytes_per_s=float(profile["backhaul"]["cloud_port_bytes_per_s"]),
        allowed_road_edges=frozenset(tuple(edge) for edge in mobility["allowed_road_edges"]),
        catalog=catalog,
        alibaba_unit_bytes=int(workflow_spec["alibaba_unit_bytes"]),
        tick_s=tick_ms / 1000,
    )
    scenario.validate()
    trace_payload = {vehicle_id: [vars(sample) for sample in traces[vehicle_id]] for vehicle_id in vehicle_ids}
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "scenario_id": scenario.scenario_id,
        "profile_path": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "input_paths": {label: str(path.resolve()) for label, path in sources.items()},
        "input_sha256": {label: spec["sha256"] for label, spec in (("mobility", mobility), ("workflow", workflow_spec), ("catalog", catalog_spec))},
        "trace_slice_sha256": _canonical_hash(trace_payload),
        "workflow_graph_sha256": _canonical_hash({vehicle_id: workflows[vehicle_id].to_dict() for vehicle_id in vehicle_ids}),
        "selected_vehicle_ids": list(vehicle_ids),
        "source_location": mobility["source_segment_id"],
        "raw_window": {"first_time_ms": first_ms, "last_time_ms": last_ms, "frame_count": frame_count, "frame_period_ms": tick_ms},
        "road_graph_type": mobility["road_graph_type"],
        "rsus": [vars(rsu) for rsu in rsus],
        "backhaul_links": [vars(link) for link in links],
        "request_specs": {vehicle_id: vars(requests[vehicle_id]) for vehicle_id in vehicle_ids},
        "claim_scope": profile["claim_scope"],
    }
    return scenario, metadata
