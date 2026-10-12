"""Build one reproducible 4-RSU/8-vehicle competition scenario from local traces."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.data.workflow.workflow_dataset_builder import WorkflowDatasetBuilder
from src.envs.core.vec_competition_env import CompetitionScenario, RoadsideUnit, TracePoint


FEET_TO_METRES = 0.3048


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as file:
        return file.read(40).startswith(b"version https://git-lfs.github.com/spec")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _segment(value: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in value.strip()).strip("_") or "unknown"


def _observed_lane_graph(csv_path: Path, *, max_rows: int, place: str, selected: tuple[str, ...], first_ms: int, frame_count: int) -> dict[str, Any]:
    """Observed Section_ID/Lane_ID transitions, not a surveyed road graph."""
    records: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"Section_ID", "Lane_ID", "Direction"}.issubset(reader.fieldnames or []):
            return {"status": "unavailable", "reason": "section/lane/direction columns missing"}
        for index, row in enumerate(reader):
            if index >= max_rows:
                break
            if _segment(row["Location"]) != place:
                continue
            vehicle_id = f"{place}:{row['Vehicle_ID']}"
            if vehicle_id not in selected:
                continue
            time_ms = int(float(row["Global_Time"].replace(",", "")))
            if first_ms <= time_ms < first_ms + frame_count * 100:
                label = f"section:{row['Section_ID']}|lane:{row['Lane_ID']}|direction:{row['Direction']}"
                records[vehicle_id].append((time_ms, label))
    nodes: set[str] = set()
    edges: dict[tuple[str, str], int] = defaultdict(int)
    for ordered in records.values():
        ordered.sort()
        nodes.update(label for _, label in ordered)
        for (_, source), (_, target) in zip(ordered, ordered[1:]):
            if source != target:
                edges[(source, target)] += 1
    return {
        "status": "observed_transition_graph",
        "source_fields": ["Section_ID", "Lane_ID", "Direction"],
        "nodes": sorted(nodes),
        "edges": [{"from": source, "to": target, "count": count} for (source, target), count in sorted(edges.items())],
        "vehicle_count": len(records),
        "scope": "selected vehicles/window only; no geometry, lane connectivity or surveyed infrastructure",
    }


def load_ngsim_window(csv_path: Path, *, max_rows: int = 500_000, frame_count: int = 120, segment: str = "") -> tuple[str, dict[str, tuple[TracePoint, ...]], dict[str, Any]]:
    """Find eight vehicles present throughout one contiguous 10 Hz corridor window.

    NGSIM Local_X/Y and v_Vel are feet and feet/s; Global_Time is milliseconds.
    A source Location is never merged with another, even when vehicle IDs repeat.
    """
    if is_lfs_pointer(csv_path):
        raise ValueError(f"NGSIM is an LFS pointer, supply the materialized CSV: {csv_path}")
    frames: dict[tuple[str, int], dict[str, TracePoint]] = defaultdict(dict)
    duplicate_keys: set[tuple[str, int, str]] = set()
    counts: dict[str, int] = defaultdict(int)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"Vehicle_ID", "Global_Time", "Local_X", "Local_Y", "v_Vel", "Location"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"NGSIM missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for row_index, row in enumerate(reader):
            if row_index >= max_rows:
                break
            place = _segment(row["Location"])
            if segment and place != segment:
                continue
            time_ms = int(float(row["Global_Time"].replace(",", "")))
            vehicle_id = f"{place}:{row['Vehicle_ID']}"
            identity = (place, time_ms, vehicle_id)
            if identity in duplicate_keys:
                continue
            if vehicle_id in frames[(place, time_ms)]:
                frames[(place, time_ms)].pop(vehicle_id)
                duplicate_keys.add(identity)
                continue
            frames[(place, time_ms)][vehicle_id] = TracePoint(
                time_ms,
                float(row["Local_X"].replace(",", "")) * FEET_TO_METRES,
                float(row["Local_Y"].replace(",", "")) * FEET_TO_METRES,
                abs(float(row["v_Vel"].replace(",", ""))) * FEET_TO_METRES,
                f"section:{row.get('Section_ID', '')}|lane:{row.get('Lane_ID', '')}|direction:{row.get('Direction', '')}",
                str(row.get("Direction", "")),
            )
            counts[place] += 1
    if not frames:
        raise ValueError("NGSIM scan found no trace rows")
    by_place: dict[str, list[int]] = defaultdict(list)
    for place, time_ms in frames:
        by_place[place].append(time_ms)
    best: tuple[float, str, int, tuple[str, ...]] | None = None
    for place, unordered_times in by_place.items():
        times = sorted(unordered_times)
        if len(times) < frame_count:
            continue
        for start in range(0, len(times) - frame_count + 1, 10):
            window = times[start:start + frame_count]
            if window[-1] - window[0] != (frame_count - 1) * 100:
                continue
            common = set(frames[(place, window[0])])
            for time_ms in window[1:]:
                common.intersection_update(frames[(place, time_ms)])
                if len(common) < 8:
                    break
            if len(common) < 8:
                continue
            plausible: list[str] = []
            for vid in common:
                points = [frames[(place, time_ms)][vid] for time_ms in window]
                if len({point.direction for point in points}) != 1:
                    continue
                if all(
                    math.hypot(b.x_m - a.x_m, b.y_m - a.y_m)
                    <= max(2.0, 2.5 * max(a.speed_mps, b.speed_mps) * 0.1, 8.0)
                    for a, b in zip(points, points[1:])
                ):
                    plausible.append(vid)
            if len(plausible) < 8:
                continue
            first = frames[(place, window[0])]
            last = frames[(place, window[-1])]
            scored = sorted(plausible, key=lambda vid: (-math.hypot(last[vid].x_m - first[vid].x_m, last[vid].y_m - first[vid].y_m), vid))
            selected = tuple(sorted(scored[:8]))
            displacement = sum(math.hypot(last[v].x_m - first[v].x_m, last[v].y_m - first[v].y_m) for v in selected)
            spread = max(first[v].y_m for v in selected) - min(first[v].y_m for v in selected)
            score = displacement + 0.05 * spread
            candidate = (score, place, window[0], selected)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        raise ValueError(f"no {frame_count}-frame shared window with eight continuous vehicles in first {max_rows} rows")
    _, place, first_ms, selected = best
    traces = {vid: tuple(frames[(place, first_ms + step * 100)][vid] for step in range(frame_count)) for vid in selected}
    metadata = {
        "source_location": place,
        "source_rows_scanned": sum(counts.values()),
        "max_rows": max_rows,
        "frame_count": frame_count,
        "first_time_ms": first_ms,
        "last_time_ms": first_ms + (frame_count - 1) * 100,
        "vehicle_ids": list(selected),
        "duplicate_location_vehicle_time_keys_excluded": len(duplicate_keys),
        "identity_rule": "unique Location+Vehicle_ID+Global_Time in every selected frame; one direction and bounded per-frame displacement",
        "coordinate_system": "NGSIM Local_X/Local_Y, feet converted to metres; no GIS road graph",
        "speed_unit": "NGSIM v_Vel feet/s converted to m/s",
        "selection_rule": f"highest eight-vehicle total displacement among scanned {frame_count}-frame shared windows, stride 10; duplicate raw identities excluded",
        "observed_lane_graph": _observed_lane_graph(csv_path, max_rows=max_rows, place=place, selected=selected, first_ms=first_ms, frame_count=frame_count),
    }
    return place, traces, metadata


def derive_four_rsus(traces: dict[str, tuple[TracePoint, ...]], *, cache_capacity_mb: float = 512.0, compute_slots: int = 1, compute_mb_per_s: float = 40.0, radio_mb_per_s: float = 20.0, backhaul_mb_per_s: float = 40.0) -> tuple[RoadsideUnit, ...]:
    """Explicit inferred geometry in local coordinates, not measured RSU sites."""
    points = [point for trace in traces.values() for point in trace]
    x_span = max(p.x_m for p in points) - min(p.x_m for p in points)
    y_span = max(p.y_m for p in points) - min(p.y_m for p in points)
    dominant_x = x_span >= y_span
    axis = sorted(p.x_m if dominant_x else p.y_m for p in points)
    cross = sorted(p.y_m if dominant_x else p.x_m for p in points)
    median_cross = cross[len(cross) // 2]
    low, high = axis[0], axis[-1]
    spacing = max((high - low) / 4, 1.0)
    perpendicular = max(abs(c - median_cross) for c in cross)
    radius = max(20.0, 0.85 * spacing + perpendicular)
    rsus = []
    for index in range(4):
        position = low + spacing * (index + 0.5)
        x_m, y_m = (position, median_cross) if dominant_x else (median_cross, position)
        rsus.append(RoadsideUnit(f"rsu_{'abcd'[index]}", x_m, y_m, radius, cache_capacity_mb, compute_slots, compute_mb_per_s, radio_mb_per_s, backhaul_mb_per_s))
    return tuple(rsus)


def build_scenario(ngsim_csv: Path, alibaba_csv: Path, catalog_json: Path, *, max_rows: int = 500_000, frame_count: int = 120, segment: str = "", cache_capacity_mb: float = 512.0, compute_slots: int = 1, compute_mb_per_s: float = 40.0, radio_mb_per_s: float = 20.0, backhaul_mb_per_s: float = 40.0, alibaba_size_unit_mb: float = 0.001) -> tuple[CompetitionScenario, dict[str, Any]]:
    for path in (ngsim_csv, alibaba_csv, catalog_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    if is_lfs_pointer(alibaba_csv):
        raise ValueError(f"Alibaba is an LFS pointer, supply the materialized CSV: {alibaba_csv}")
    place, traces, mobility = load_ngsim_window(ngsim_csv, max_rows=max_rows, frame_count=frame_count, segment=segment)
    states = WorkflowDatasetBuilder().build_alibaba_workflow_states(alibaba_csv, limit_jobs=64, min_tasks=5, max_tasks=32)
    if len(states) < 8:
        raise ValueError("Alibaba scan produced fewer than eight distinct 5-32 task DAG jobs")
    workflows = dict(zip(sorted(traces), states[:8]))
    catalog = AdapterCatalog.from_json(catalog_json)
    scenario = CompetitionScenario(place, traces, workflows, derive_four_rsus(traces, cache_capacity_mb=cache_capacity_mb, compute_slots=compute_slots, compute_mb_per_s=compute_mb_per_s, radio_mb_per_s=radio_mb_per_s, backhaul_mb_per_s=backhaul_mb_per_s), catalog, alibaba_size_unit_mb)
    scenario.validate()
    metadata = {
        "mobility": mobility,
        "workflow_assignments": {vid: {"workflow_id": wf.workflow_id, "node_count": len(wf.nodes)} for vid, wf in workflows.items()},
        "rsu_layout": "four inferred points along dominant local-coordinate axis; not surveyed RSU sites",
        "rsus": [vars(r) for r in scenario.rsus],
        "model_catalog_fingerprint": catalog.canonical_fingerprint(),
        "model_catalog_formal_use_status": "controlled_validation_only",
        "input_sha256": {"ngsim": sha256_file(ngsim_csv), "alibaba": sha256_file(alibaba_csv), "catalog": sha256_file(catalog_json)},
        "input_paths": {"ngsim": str(ngsim_csv.resolve()), "alibaba": str(alibaba_csv.resolve()), "catalog": str(catalog_json.resolve())},
    }
    return scenario, metadata
