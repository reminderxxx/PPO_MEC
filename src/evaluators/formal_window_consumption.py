"""Frozen-window loading and reachability checks for typed formal execution.

The formal split stores offsets in the provider's globally sorted frame list,
not raw CSV row offsets.  This module binds those offsets to raw segment/time
identity and loads only the frozen frames while still scanning the complete,
explicit source-row range.  It deliberately has no policy or holdout
performance execution API.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import pandas as pd

from src.data.mobility.replay_provider import ReplayProvider
from src.envs.specs import VehicleState
from src.evaluators.real_sample_support import RealMobilityBundle, build_sample_rsus
from src.evaluators.typed_model_cache_formal_protocol import (
    attach_hashes,
    canonical_sha256,
    semantic_projection,
)


FORMAL_WINDOW_CONSUMPTION_CONTRACT_VERSION = "1.0.0"
FORMAL_WINDOW_LOADER_IDENTITY = "ngsim_frozen_segment_time_loader_v1"
FORMAL_WINDOW_PREPROCESSING_IDENTITY = (
    "NGSIMProvider_v1: dtype=str, comma-stripped numeric conversion, "
    "segment=normalized(Location), frame=(segment,Global_Time), preserve raw rows"
)
FORMAL_WINDOW_VEHICLE_SELECTION_IDENTITY = (
    "full frame vehicle coverage before outcome-blind primary vehicle selection"
)
FORMAL_WINDOW_RSU_MAPPER_IDENTITY = (
    "build_sample_rsus/RSUMapper:auto_dominant_tight_v1"
)
PERFORMANCE_FORBIDDEN_FIELDS = {
    "reward",
    "total_reward",
    "episode_success",
    "performance",
    "cache_hit",
    "oracle_gap",
    "agent_name",
    "checkpoint_path",
}


class FormalWindowConsumptionError(ValueError):
    """Raised when a frozen window cannot be resolved without ambiguity."""


def normalized_segment(value: Any) -> str:
    result = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(value or "unknown")
    ).strip("_")
    return result or "unknown"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalWindowConsumptionError(f"unable to load {label}: {target}") from exc
    if not isinstance(payload, dict):
        raise FormalWindowConsumptionError(f"{label} must be a JSON object")
    return payload


def _reject_non_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FormalWindowConsumptionError(f"non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_non_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_non_finite(child, f"{path}[{index}]")


def load_window_plan(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = Path(path).resolve()
    payload = load_json_object(target, "window plan")
    windows = payload.get("selected_window_plan")
    if not isinstance(windows, list) or not windows:
        raise FormalWindowConsumptionError(f"selected_window_plan missing: {target}")
    required = {
        "window_id",
        "frame_offset",
        "window_length",
        "source_segment_id",
        "source_segment_run_id",
        "raw_frame_start",
        "raw_frame_end",
        "raw_time_start",
        "raw_time_end",
    }
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(windows):
        if not isinstance(item, Mapping):
            raise FormalWindowConsumptionError(f"window {index} is not an object")
        forbidden = PERFORMANCE_FORBIDDEN_FIELDS.intersection(item)
        if forbidden:
            raise FormalWindowConsumptionError(
                f"performance field leaked into frozen window {index}: {sorted(forbidden)}"
            )
        missing = required.difference(item)
        if missing:
            raise FormalWindowConsumptionError(
                f"window {index} missing identity fields: {sorted(missing)}"
            )
        row = dict(item)
        window_id = str(row["window_id"])
        if window_id in seen:
            raise FormalWindowConsumptionError(f"duplicate window_id: {window_id}")
        seen.add(window_id)
        if int(row["window_length"]) != 24:
            raise FormalWindowConsumptionError("formal window length must remain 24")
        normalized.append(row)
    return payload, normalized


def _vehicle_projection(vehicle: VehicleState) -> dict[str, Any]:
    return {
        "vehicle_id": str(vehicle.vehicle_id),
        "position_x": round(float(vehicle.position_x), 9),
        "position_y": round(float(vehicle.position_y), 9),
        "speed": round(float(vehicle.speed), 9),
        "base_model_id": str(vehicle.base_model_id),
    }


def window_fingerprint(frames: Sequence[Mapping[str, Any]]) -> str:
    projection = []
    for frame in frames:
        projection.append(
            {
                "source_segment_id": str(frame.get("source_segment_id") or ""),
                "provider_frame_offset": int(frame.get("provider_frame_offset", -1)),
                "segment_frame_index": int(frame.get("segment_frame_index", -1)),
                "raw_frame": int(frame.get("ngsim_frame_id", -1)),
                "raw_time": int(frame.get("global_time", frame.get("time_index", -1))),
                "vehicles": sorted(
                    (_vehicle_projection(vehicle) for vehicle in frame.get("vehicles", [])),
                    key=lambda row: (
                        row["vehicle_id"],
                        row["position_x"],
                        row["position_y"],
                        row["speed"],
                    ),
                ),
            }
        )
    return canonical_sha256(projection)


def _contract_units(contract: Mapping[str, Any], split: str | None = None) -> list[dict[str, Any]]:
    units = contract.get("evaluation_units")
    if not isinstance(units, list) or not units:
        raise FormalWindowConsumptionError("window consumption contract has no evaluation_units")
    result = [dict(item) for item in units if split is None or item.get("split_name") == split]
    if split is not None and not result:
        raise FormalWindowConsumptionError(f"contract has no split: {split}")
    return result


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    _reject_non_finite(contract)
    if contract.get("formal_window_consumption_contract_version") != FORMAL_WINDOW_CONSUMPTION_CONTRACT_VERSION:
        raise FormalWindowConsumptionError("unsupported window consumption contract version")
    if contract.get("loader_identity") != FORMAL_WINDOW_LOADER_IDENTITY:
        raise FormalWindowConsumptionError("formal window loader identity mismatch")
    source = contract.get("source")
    if not isinstance(source, Mapping):
        raise FormalWindowConsumptionError("contract source identity missing")
    row_count = int(source.get("row_count", 0))
    resolved = contract.get("resolved_source_range")
    if not isinstance(resolved, Mapping):
        raise FormalWindowConsumptionError("resolved source range missing")
    if int(resolved.get("start_row_inclusive", -1)) != 0:
        raise FormalWindowConsumptionError("formal source range must be a prefix from row zero")
    end = int(resolved.get("end_row_exclusive", 0))
    if end <= 0 or end > row_count:
        raise FormalWindowConsumptionError("resolved source range exceeds source length")
    units = _contract_units(contract)
    expected_counts = {"train": 24, "dev": 12, "formal": 12, "sealed_holdout": 12}
    observed_counts = {
        split: sum(item.get("split_name") == split for item in units)
        for split in expected_counts
    }
    if observed_counts != expected_counts:
        raise FormalWindowConsumptionError(
            f"formal split counts changed: {observed_counts}"
        )
    ids = [str(item.get("window_id")) for item in units]
    if len(ids) != len(set(ids)):
        raise FormalWindowConsumptionError("duplicate contract window identity")
    expected_hash = canonical_sha256(semantic_projection(contract))
    if contract.get("hashes", {}).get("semantic_sha256") != expected_hash:
        raise FormalWindowConsumptionError("window consumption contract hash mismatch")
    return {
        "status": "pass",
        "window_count": len(units),
        "split_counts": observed_counts,
        "resolved_source_rows": end,
        "semantic_sha256": expected_hash,
    }


def load_contract(path: str | Path) -> dict[str, Any]:
    contract = load_json_object(path, "window consumption contract")
    validate_contract(contract)
    return contract


def _unit_identity_projection(unit: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "window_id",
        "requested_frame_offset",
        "window_length",
        "source_segment_id",
        "source_segment_run_id",
        "raw_frame_start",
        "raw_frame_end",
        "raw_time_start",
        "raw_time_end",
    )
    return {field: unit.get(field) for field in fields}


def validate_window_plan_binding(
    *,
    contract: Mapping[str, Any],
    plan_path: str | Path,
    split: str,
    max_mobility_rows: int,
    mobility_csv_path: str | Path,
    window_selector: str,
    window_length: int,
    rsu_layout: str,
    primary_vehicle_selection: str,
    mode: str = "formal",
) -> dict[str, Any]:
    """Validate CLI/source/plan fields before any episode or checkpoint write."""

    if mode not in {"formal", "rehearsal", "identity_only"}:
        raise FormalWindowConsumptionError(f"unsupported window consumption mode: {mode}")
    if split == "sealed_holdout" and mode != "identity_only":
        raise FormalWindowConsumptionError("sealed holdout permits identity-only validation")
    if split != "sealed_holdout" and mode == "identity_only":
        raise FormalWindowConsumptionError("identity-only mode is reserved for sealed holdout")
    source = contract["source"]
    resolved = contract["resolved_source_range"]
    if int(max_mobility_rows) != int(resolved["end_row_exclusive"]):
        raise FormalWindowConsumptionError(
            "formal max_mobility_rows must equal the frozen resolved source range"
        )
    source_path = Path(mobility_csv_path).resolve()
    portable_resolution = source.get("runtime_resolution") or {}
    portable = (
        portable_resolution.get("portable_resource_identity_contract_version")
        == "1.0.0"
    )
    if not source_path.is_file():
        raise FormalWindowConsumptionError("formal mobility source is missing")
    if not portable and source_path != Path(source["path"]).resolve():
        raise FormalWindowConsumptionError("formal mobility source path mismatch")
    if source_path.stat().st_size != int(source["size_bytes"]):
        raise FormalWindowConsumptionError("formal mobility source size mismatch")
    if portable and file_sha256(source_path) != source["sha256"]:
        raise FormalWindowConsumptionError("formal mobility source content mismatch")
    if window_selector != "ordered":
        raise FormalWindowConsumptionError("frozen windows require ordered exact-offset loading")
    if int(window_length) != int(contract["window_length"]):
        raise FormalWindowConsumptionError("formal window length override rejected")
    if rsu_layout != contract["rsu_layout"]:
        raise FormalWindowConsumptionError("formal RSU layout override rejected")
    if primary_vehicle_selection not in contract["allowed_vehicle_selection_identities"]:
        raise FormalWindowConsumptionError("formal vehicle selection identity is not frozen")
    payload, windows = load_window_plan(plan_path)
    if str(payload.get("split")) != split:
        raise FormalWindowConsumptionError(
            f"window plan split mismatch: {payload.get('split')} != {split}"
        )
    expected = _contract_units(contract, split)
    expected_by_id = {str(item["window_id"]): item for item in expected}
    observed_projection = []
    for window in windows:
        expected_unit = expected_by_id.get(str(window["window_id"]))
        if expected_unit is None:
            raise FormalWindowConsumptionError(
                f"window is not frozen for split {split}: {window['window_id']}"
            )
        observed_projection.append(
            {
                "window_id": str(window["window_id"]),
                "requested_frame_offset": int(window["frame_offset"]),
                "window_length": int(window["window_length"]),
                "source_segment_id": str(window["source_segment_id"]),
                "source_segment_run_id": str(window["source_segment_run_id"]),
                "raw_frame_start": int(window["raw_frame_start"]),
                "raw_frame_end": int(window["raw_frame_end"]),
                "raw_time_start": int(window["raw_time_start"]),
                "raw_time_end": int(window["raw_time_end"]),
            }
        )
        if observed_projection[-1] != _unit_identity_projection(expected_unit):
            raise FormalWindowConsumptionError(
                f"window plan identity mismatch: {window['window_id']}"
            )
    if mode == "formal" and set(expected_by_id) != {
        str(window["window_id"]) for window in windows
    }:
        raise FormalWindowConsumptionError("formal command must consume the complete frozen split plan")
    plan_sha256 = file_sha256(Path(plan_path))
    expected_plan_hash = contract["window_plans"][split]["file_sha256"]
    if mode == "formal" and plan_sha256 != expected_plan_hash:
        raise FormalWindowConsumptionError("formal window plan file hash mismatch")
    return {
        "status": "pass",
        "mode": mode,
        "split": split,
        "window_count": len(windows),
        "complete_split_bound": len(windows) == len(expected),
        "plan_file_sha256": plan_sha256,
        "resolved_source_rows": int(resolved["end_row_exclusive"]),
        "contract_semantic_sha256": contract["hashes"]["semantic_sha256"],
    }


def _scan_units(
    *,
    source_path: Path,
    max_rows: int,
    units: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[int, int],
    dict[str, Any],
]:
    """Run the NGSIM parser semantics over the frozen source prefix once."""

    by_key: dict[tuple[str, int], list[str]] = defaultdict(list)
    unit_by_id = {str(unit["window_id"]): dict(unit) for unit in units}
    for unit in units:
        sampling = int(unit.get("sampling_interval", 100))
        for index in range(int(unit["window_length"])):
            raw_time = int(unit["raw_time_start"]) + index * sampling
            by_key[(str(unit["source_segment_id"]), raw_time)].append(str(unit["window_id"]))
    target_segments = {segment for segment, _ in by_key}
    target_times = {raw_time for _, raw_time in by_key}
    frames_by_window: dict[str, dict[int, dict[str, Any]]] = {
        window_id: {} for window_id in unit_by_id
    }
    row_ranges: dict[str, dict[str, Any]] = {
        window_id: {
            "first_row_inclusive": None,
            "last_row_inclusive": None,
            "matched_raw_row_count": 0,
        }
        for window_id in unit_by_id
    }
    max_row_by_i80_time: dict[int, int] = {}
    provider_frame_raw: dict[tuple[str, int], int] = {}
    invalid_provider_keys: set[tuple[str, int]] = set()
    columns = [
        "Vehicle_ID",
        "Frame_ID",
        "Local_X",
        "Local_Y",
        "v_Vel",
        "Location",
        "Global_Time",
    ]
    row_start = 0
    for chunk in pd.read_csv(
        source_path,
        usecols=columns,
        dtype=str,
        keep_default_na=False,
        nrows=int(max_rows),
        chunksize=250_000,
    ):
        segments = chunk["Location"].map(normalized_segment)
        times = pd.to_numeric(
            chunk["Global_Time"].str.replace(",", "", regex=False), errors="raise"
        ).astype("int64")
        absolute_rows = pd.Series(
            range(row_start, row_start + len(chunk)), index=chunk.index, dtype="int64"
        )
        raw_frames = pd.to_numeric(
            chunk["Frame_ID"].str.replace(",", "", regex=False), errors="raise"
        ).astype("int64")
        identities = pd.DataFrame(
            {
                "segment": segments,
                "raw_time": times,
                "raw_frame": raw_frames,
            }
        ).drop_duplicates()
        for identity in identities.itertuples(index=False):
            key = (str(identity.segment), int(identity.raw_time))
            raw_frame = int(identity.raw_frame)
            previous_raw_frame = provider_frame_raw.setdefault(key, raw_frame)
            if previous_raw_frame != raw_frame:
                invalid_provider_keys.add(key)
        i80 = segments.eq("i_80")
        for raw_time, maximum in (
            pd.DataFrame(
                {"time": times.loc[i80], "row": absolute_rows.loc[i80]}
            )
            .groupby("time", sort=False)["row"]
            .max()
            .items()
        ):
            max_row_by_i80_time[int(raw_time)] = max(
                max_row_by_i80_time.get(int(raw_time), -1), int(maximum)
            )
        target_mask = segments.isin(target_segments) & times.isin(target_times)
        if target_mask.any():
            selected = chunk.loc[target_mask]
            for index, row in selected.iterrows():
                segment = str(segments.loc[index])
                raw_time = int(times.loc[index])
                absolute_row = int(absolute_rows.loc[index])
                raw_frame = int(raw_frames.loc[index])
                for window_id in by_key[(segment, raw_time)]:
                    unit = unit_by_id[window_id]
                    local_index = int((raw_time - int(unit["raw_time_start"])) // int(unit.get("sampling_interval", 100)))
                    frame = frames_by_window[window_id].setdefault(
                        raw_time,
                        {
                            "time_index": raw_time,
                            "ngsim_frame_id": raw_frame,
                            "global_time": raw_time,
                            "source_location": str(row["Location"]),
                            "source_segment_id": segment,
                            "provider_frame_offset": None,
                            "segment_frame_index": None,
                            "vehicles": [],
                        },
                    )
                    if int(frame["ngsim_frame_id"]) != raw_frame:
                        raise FormalWindowConsumptionError(
                            f"conflicting raw frame identity at {segment}/{raw_time}"
                        )
                    frame["vehicles"].append(
                        VehicleState(
                            vehicle_id=f"{segment}:{row['Vehicle_ID']}",
                            position_x=float(str(row["Local_X"]).replace(",", "")),
                            position_y=float(str(row["Local_Y"]).replace(",", "")),
                            speed=abs(float(str(row["v_Vel"]).replace(",", ""))),
                            base_model_id="veh_base_v1",
                        )
                    )
                    interval = row_ranges[window_id]
                    interval["first_row_inclusive"] = (
                        absolute_row
                        if interval["first_row_inclusive"] is None
                        else min(int(interval["first_row_inclusive"]), absolute_row)
                    )
                    interval["last_row_inclusive"] = (
                        absolute_row
                        if interval["last_row_inclusive"] is None
                        else max(int(interval["last_row_inclusive"]), absolute_row)
                    )
                    interval["matched_raw_row_count"] += 1
        row_start += len(chunk)
    provider_offsets = {
        key: index
        for index, key in enumerate(
            sorted(key for key in provider_frame_raw if key not in invalid_provider_keys)
        )
    }
    run_local_indices: dict[tuple[str, int], int] = {}
    keys_by_segment: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in provider_offsets:
        keys_by_segment[key[0]].append(key)
    for segment, keys in keys_by_segment.items():
        ordered_keys = sorted(keys, key=lambda item: item[1])
        positive_deltas = [
            right[1] - left[1]
            for left, right in zip(ordered_keys, ordered_keys[1:])
            if right[1] > left[1]
        ]
        sampling = int(median(positive_deltas)) if positive_deltas else 1
        local_index = 0
        previous_key: tuple[str, int] | None = None
        for key in ordered_keys:
            if previous_key is not None:
                time_contiguous = key[1] - previous_key[1] == sampling
                frame_contiguous = (
                    provider_frame_raw[key] - provider_frame_raw[previous_key] == 1
                )
                local_index = local_index + 1 if time_contiguous and frame_contiguous else 0
            run_local_indices[key] = local_index
            previous_key = key
    ordered: dict[str, list[dict[str, Any]]] = {}
    for window_id, frames in frames_by_window.items():
        unit = unit_by_id[window_id]
        result = [frames[key] for key in sorted(frames)]
        if len(result) != int(unit["window_length"]):
            raise FormalWindowConsumptionError(
                f"window unreachable: {window_id}, observed_frames={len(result)}"
            )
        for frame in result:
            key = (str(frame["source_segment_id"]), int(frame["global_time"]))
            frame["provider_frame_offset"] = provider_offsets[key]
            frame["segment_frame_index"] = run_local_indices[key]
        ordered[window_id] = result
    return (
        ordered,
        row_ranges,
        max_row_by_i80_time,
        {
            "provider_frame_count": len(provider_offsets),
            "invalid_provider_frame_identity_count": len(invalid_provider_keys),
            "provider_identity_recomputed_from_raw_source": True,
            "run_local_identity_recomputed_from_raw_source": True,
        },
    )


def _copy_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for frame in frames:
        copied = {key: value for key, value in frame.items() if key != "vehicles"}
        copied["vehicles"] = [
            VehicleState(
                vehicle_id=vehicle.vehicle_id,
                position_x=vehicle.position_x,
                position_y=vehicle.position_y,
                speed=vehicle.speed,
                base_model_id=vehicle.base_model_id,
                associated_rsu_id=vehicle.associated_rsu_id,
                active_workflow_id=vehicle.active_workflow_id,
            )
            for vehicle in frame.get("vehicles", [])
        ]
        result.append(copied)
    return result


@lru_cache(maxsize=8)
def _load_split_frames_cached(
    contract_path_raw: str,
    split: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    contract = load_contract(contract_path_raw)
    units = _contract_units(contract, split)
    source_path = Path(contract["source"]["path"])
    frames, ranges, _, _ = _scan_units(
        source_path=source_path,
        max_rows=int(contract["resolved_source_range"]["end_row_exclusive"]),
        units=units,
    )
    return frames, ranges


def load_window_bundle_from_contract(
    *,
    contract_path: str | Path,
    split: str,
    window_id: str,
    rsu_layout: str,
) -> RealMobilityBundle:
    if split == "sealed_holdout":
        raise FormalWindowConsumptionError(
            "sealed holdout cannot be loaded by a training/evaluation bundle consumer"
        )
    contract_path_resolved = str(Path(contract_path).resolve())
    contract = load_contract(contract_path_resolved)
    unit = next(
        (
            item
            for item in _contract_units(contract, split)
            if str(item["window_id"]) == str(window_id)
        ),
        None,
    )
    if unit is None:
        raise FormalWindowConsumptionError(
            f"window is not bound to split {split}: {window_id}"
        )
    frame_map, row_ranges = _load_split_frames_cached(contract_path_resolved, split)
    frames = _copy_frames(frame_map[str(window_id)])
    fingerprint = window_fingerprint(frames)
    if fingerprint != unit["expected_fingerprint"]:
        raise FormalWindowConsumptionError(
            f"observed window fingerprint mismatch: {window_id}"
        )
    rsu_states, rsu_metadata = build_sample_rsus(frames=frames, rsu_layout=rsu_layout)
    metadata = {
        **rsu_metadata,
        "window_id": str(window_id),
        "frame_offset": int(unit["requested_frame_offset"]),
        "window_length": int(unit["window_length"]),
        "window_selector": "ordered",
        "time_index_start": int(unit["raw_time_start"]),
        "time_index_end": int(unit["raw_time_end"]),
        "source_segment_id": str(unit["source_segment_id"]),
        "source_segment_run_id": str(unit["source_segment_run_id"]),
        "source_frame_count": int(contract["source"]["provider_frame_count"]),
        "selected_frame_count": len(frames),
        "formal_window_consumption_contract_version": FORMAL_WINDOW_CONSUMPTION_CONTRACT_VERSION,
        "formal_window_consumption_contract_sha256": contract["hashes"]["semantic_sha256"],
        "observed_window_fingerprint": fingerprint,
        "expected_window_fingerprint": unit["expected_fingerprint"],
        "observed_source_row_interval": row_ranges[str(window_id)],
    }
    return RealMobilityBundle(
        provider=ReplayProvider(trajectory_frames=frames),
        frames=frames,
        rsu_states=rsu_states,
        rsu_metadata=metadata,
        source_path=str(contract["source"]["path"]),
    )


def validate_reachability(
    contract_path: str | Path,
    *,
    splits: Sequence[str] = ("train", "dev", "formal", "sealed_holdout"),
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    requested_units = [
        item for item in _contract_units(contract) if item["split_name"] in set(splits)
    ]
    frames_by_id, row_ranges, _, loader_audit = _scan_units(
        source_path=Path(contract["source"]["path"]),
        max_rows=int(contract["resolved_source_range"]["end_row_exclusive"]),
        units=requested_units,
    )
    rows = []
    for unit in requested_units:
        window_id = str(unit["window_id"])
        frames = frames_by_id[window_id]
        observed_fingerprint = window_fingerprint(frames)
        frame_interval = [
            int(frames[0]["ngsim_frame_id"]),
            int(frames[-1]["ngsim_frame_id"]),
        ]
        time_interval = [int(frames[0]["global_time"]), int(frames[-1]["global_time"])]
        provider_interval = [
            int(frames[0]["provider_frame_offset"]),
            int(frames[-1]["provider_frame_offset"]),
        ]
        run_local_interval = [
            int(frames[0]["segment_frame_index"]),
            int(frames[-1]["segment_frame_index"]),
        ]
        errors = []
        if frame_interval != [int(unit["raw_frame_start"]), int(unit["raw_frame_end"])]:
            errors.append("raw_frame_interval_mismatch")
        if time_interval != [int(unit["raw_time_start"]), int(unit["raw_time_end"])]:
            errors.append("raw_time_interval_mismatch")
        if provider_interval != [
            int(unit["requested_frame_offset"]),
            int(unit["requested_frame_offset"]) + int(unit["window_length"]) - 1,
        ]:
            errors.append("provider_interval_mismatch")
        if run_local_interval != [
            int(unit["provider_run_local_start"]),
            int(unit["provider_run_local_end"]),
        ]:
            errors.append("provider_run_local_interval_mismatch")
        if int(loader_audit["provider_frame_count"]) != int(
            contract["source"]["provider_frame_count"]
        ):
            errors.append("provider_frame_count_mismatch")
        if observed_fingerprint != unit["expected_fingerprint"]:
            errors.append("fingerprint_mismatch")
        vehicle_counts = [len(frame.get("vehicles", [])) for frame in frames]
        if not vehicle_counts or min(vehicle_counts) <= 0:
            errors.append("vehicle_coverage_missing")
        rows.append(
            {
                "split": unit["split_name"],
                "window_id": window_id,
                "metadata_only": unit["split_name"] == "sealed_holdout",
                "reachable": not errors,
                "observed_frame_interval": frame_interval,
                "observed_time_interval": time_interval,
                "observed_provider_interval": provider_interval,
                "observed_provider_run_local_interval": run_local_interval,
                "provider_frame_count": int(loader_audit["provider_frame_count"]),
                "provider_frame_count_match": int(loader_audit["provider_frame_count"])
                == int(contract["source"]["provider_frame_count"]),
                "provider_identity_recomputed_from_raw_source": True,
                "run_local_identity_recomputed_from_raw_source": True,
                "required_source_range": unit["required_source_range"],
                "resolved_source_range": contract["resolved_source_range"],
                "observed_source_interval": row_ranges[window_id],
                "vehicle_coverage": {
                    "minimum": min(vehicle_counts),
                    "maximum": max(vehicle_counts),
                    "mean": round(sum(vehicle_counts) / len(vehicle_counts), 6),
                },
                "expected_fingerprint": unit["expected_fingerprint"],
                "observed_fingerprint": observed_fingerprint,
                "fingerprint_match": observed_fingerprint == unit["expected_fingerprint"],
                "errors": errors,
            }
        )
    split_counts = {
        split: sum(row["split"] == split and row["reachable"] for row in rows)
        for split in splits
    }
    return {
        "status": "pass" if all(row["reachable"] for row in rows) else "fail",
        "formal_window_consumption_contract_version": FORMAL_WINDOW_CONSUMPTION_CONTRACT_VERSION,
        "window_count": len(rows),
        "reachable_count": sum(row["reachable"] for row in rows),
        "split_reachable_counts": split_counts,
        "provider_frame_count": int(loader_audit["provider_frame_count"]),
        "provider_frame_count_match": int(loader_audit["provider_frame_count"])
        == int(contract["source"]["provider_frame_count"]),
        "provider_identity_recomputed_from_raw_source": True,
        "run_local_identity_recomputed_from_raw_source": True,
        "holdout_metadata_only": True,
        "agent_or_policy_executed": False,
        "performance_fields_read": False,
        "rows": rows,
    }


def build_contract(
    *,
    source_path: str | Path,
    plan_paths: Mapping[str, str | Path],
    source_row_count: int,
    source_size_bytes: int,
    source_sha256: str,
    provider_frame_count: int,
    split_semantic_sha256: str,
    historical_registry_semantic_sha256: str,
    inventory_semantic_sha256: str,
) -> dict[str, Any]:
    source = Path(source_path).resolve()
    units: list[dict[str, Any]] = []
    plan_identity: dict[str, Any] = {}
    for split, raw_path in plan_paths.items():
        plan_path = Path(raw_path).resolve()
        payload, windows = load_window_plan(plan_path)
        if str(payload.get("split")) != split:
            raise FormalWindowConsumptionError(f"plan split mismatch: {plan_path}")
        plan_identity[split] = {
            "path": str(plan_path),
            "file_sha256": file_sha256(plan_path),
            "window_count": len(windows),
        }
        for window in windows:
            units.append(
                {
                    "split_name": split,
                    "window_id": str(window["window_id"]),
                    "segment_id": str(window["source_segment_id"]),
                    "source_segment_id": str(window["source_segment_id"]),
                    "source_segment_run_id": str(window["source_segment_run_id"]),
                    "raw_frame_start": int(window["raw_frame_start"]),
                    "raw_frame_end": int(window["raw_frame_end"]),
                    "raw_time_start": int(window["raw_time_start"]),
                    "raw_time_end": int(window["raw_time_end"]),
                    "provider_local_start": int(window.get("provider_segment_frame_start", -1)),
                    "provider_local_end": int(window.get("provider_segment_frame_end", -1)),
                    "provider_run_local_start": int(window.get("provider_segment_frame_start", -1)),
                    "provider_run_local_end": int(window.get("provider_segment_frame_end", -1)),
                    "requested_frame_offset": int(window["frame_offset"]),
                    "window_length": int(window["window_length"]),
                    "sampling_interval": int(window.get("sampling_interval", 100)),
                    "source_file_hash": source_sha256,
                    "loader_identity": FORMAL_WINDOW_LOADER_IDENTITY,
                    "preprocessing_identity": FORMAL_WINDOW_PREPROCESSING_IDENTITY,
                    "vehicle_selection_identity": FORMAL_WINDOW_VEHICLE_SELECTION_IDENTITY,
                    "rsu_mapper_identity": FORMAL_WINDOW_RSU_MAPPER_IDENTITY,
                    "expected_fingerprint": None,
                }
            )
    provisional = {
        "formal_window_consumption_contract_version": FORMAL_WINDOW_CONSUMPTION_CONTRACT_VERSION,
        "loader_identity": FORMAL_WINDOW_LOADER_IDENTITY,
        "preprocessing_identity": FORMAL_WINDOW_PREPROCESSING_IDENTITY,
        "vehicle_selection_identity": FORMAL_WINDOW_VEHICLE_SELECTION_IDENTITY,
        "rsu_mapper_identity": FORMAL_WINDOW_RSU_MAPPER_IDENTITY,
        "split_semantic_sha256": split_semantic_sha256,
        "historical_registry_semantic_sha256": historical_registry_semantic_sha256,
        "available_interval_inventory_semantic_sha256": inventory_semantic_sha256,
        "source": {
            "path": str(source),
            "sha256": source_sha256,
            "size_bytes": int(source_size_bytes),
            "row_count": int(source_row_count),
            "provider_frame_count": int(provider_frame_count),
        },
        "window_plans": plan_identity,
        "window_length": 24,
        "window_selector": "ordered",
        "rsu_layout": "auto_dominant_tight",
        "allowed_vehicle_selection_identities": ["handoff_pressure", "stable_first"],
        "evaluation_units": units,
    }
    frames_by_id, row_ranges, max_row_by_time, loader_audit = _scan_units(
        source_path=source,
        max_rows=int(source_row_count),
        units=units,
    )
    if int(loader_audit["provider_frame_count"]) != int(provider_frame_count):
        raise FormalWindowConsumptionError(
            "provider frame count changed while rebuilding window consumption contract"
        )
    ordered_times = sorted(max_row_by_time)
    cumulative_max = -1
    cumulative_by_time: dict[int, int] = {}
    for raw_time in ordered_times:
        cumulative_max = max(cumulative_max, int(max_row_by_time[raw_time]))
        cumulative_by_time[raw_time] = cumulative_max
    for unit in units:
        end_time = int(unit["raw_time_end"])
        eligible = [time for time in ordered_times if time <= end_time]
        if not eligible:
            raise FormalWindowConsumptionError("unable to derive provider-prefix reachability")
        required_rows = int(cumulative_by_time[eligible[-1]]) + 1
        unit["required_source_range"] = {
            "start_row_inclusive": 0,
            "end_row_exclusive": required_rows,
            "derivation": (
                "max raw row index + 1 over every i_80 provider frame at or before "
                "the frozen window end; preserves provider offset and full target vehicles"
            ),
        }
        unit["observed_source_interval"] = row_ranges[str(unit["window_id"])]
        unit["expected_fingerprint"] = window_fingerprint(
            frames_by_id[str(unit["window_id"])]
        )
        unit["observed_fingerprint_at_freeze"] = unit["expected_fingerprint"]
    resolved_rows = max(
        int(unit["required_source_range"]["end_row_exclusive"]) for unit in units
    )
    if resolved_rows > int(source_row_count):
        raise FormalWindowConsumptionError("derived source range exceeds source row count")
    provisional["resolved_source_range"] = {
        "start_row_inclusive": 0,
        "end_row_exclusive": resolved_rows,
        "source_row_count": int(source_row_count),
        "margin_rows": int(source_row_count) - resolved_rows,
        "exceeds_source_behavior": "reject",
        "derivation": (
            "max(per-window minimum safe prefix); NGSIM rows are vehicle-major and the "
            "last required preceding i_80 frame row occurs at the final raw row"
        ),
    }
    provisional["split_required_source_rows"] = {
        split: max(
            int(unit["required_source_range"]["end_row_exclusive"])
            for unit in units
            if unit["split_name"] == split
        )
        for split in plan_paths
    }
    provisional["training_benchmark_fingerprint_contract"] = {
        "same_loader": True,
        "same_preprocessing": True,
        "same_expected_fingerprint_by_window_id": True,
        "CLI_offset_or_range_override": "reject",
    }
    provisional["holdout_access"] = {
        "sealed": True,
        "identity_interval_reachability_only": True,
        "policy_or_agent_execution": False,
        "performance_fields": False,
    }
    return attach_hashes(provisional)


__all__ = [
    "FORMAL_WINDOW_CONSUMPTION_CONTRACT_VERSION",
    "FORMAL_WINDOW_LOADER_IDENTITY",
    "FormalWindowConsumptionError",
    "build_contract",
    "file_sha256",
    "load_contract",
    "load_window_bundle_from_contract",
    "load_window_plan",
    "validate_contract",
    "validate_reachability",
    "validate_window_plan_binding",
    "window_fingerprint",
]
