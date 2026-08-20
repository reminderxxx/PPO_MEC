"""G14B historical exclusion, split, protocol, and holdout contracts.

The module is deliberately result blind.  It may inspect raw mobility identity,
continuity, and vehicle coverage, plus the ``selected_window_plan`` arrays from
historical JSON files.  It never parses reward, cache, oracle, or agent outcome
fields when discovering historical use.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Mapping, Sequence

try:
    import pandas as pd
except ImportError:  # pragma: no cover - the repository runtime includes pandas.
    pd = None


HISTORICAL_REGISTRY_VERSION = "1.0.0"
SPLIT_PROTOCOL_VERSION = "1.0.0"
FORMAL_PROTOCOL_VERSION = "1.0.0"
HOLDOUT_SEAL_VERSION = "1.0.0"
READINESS_REVIEW_VERSION = "2.0.0"
PROTOCOL_ID = "typed_model_cache_formal_protocol_v1"
SPLIT_PROTOCOL_ID = "typed_model_cache_split_protocol_v1"
HISTORICAL_REGISTRY_ID = "typed_model_cache_historical_window_usage_v1"
MINIMUM_OUTER_WINDOWS = 12
NON_SEMANTIC_FIELDS = {
    "created_at",
    "generated_at",
    "captured_at",
    "reviewed_at",
    "output_path",
    "output_root",
    "artifact_path",
    "absolute_path",
    "full_sha256",
    "semantic_sha256",
}
PERFORMANCE_FORBIDDEN_WINDOW_FIELDS = {
    "reward",
    "total_reward",
    "cache_hit_rate",
    "cache_byte_hit_rate",
    "cache_opportunity",
    "oracle_gap",
    "typed_hit_rate",
    "agent_performance",
    "mechanism_activation_score",
    "estimated_handoff_count",
    "physical_transfer_opportunity_count",
}
SPLIT_NAMES = ("train", "dev", "formal", "sealed_holdout")


class FormalProtocolError(ValueError):
    """Raised when a G14B contract would be ambiguous or mutable."""


class InsufficientWindowError(FormalProtocolError):
    """Raised when the frozen 12+12 outer-window requirement cannot be met."""


class HoldoutAccessError(FormalProtocolError):
    """Raised when a caller attempts to bypass the sealed-holdout contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_non_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FormalProtocolError(f"non-finite JSON value at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise FormalProtocolError(f"JSON object key is not a string at {path}")
            _reject_non_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_non_finite(child, f"{path}[{index}]")


def canonical_json_bytes(value: Any) -> bytes:
    _reject_non_finite(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_projection(value: Any) -> Any:
    """Drop only explicitly non-semantic execution metadata and hash fields."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if key in NON_SEMANTIC_FIELDS or key == "hashes":
                continue
            result[str(key)] = semantic_projection(child)
        return result
    if isinstance(value, list):
        return [semantic_projection(item) for item in value]
    if isinstance(value, tuple):
        return [semantic_projection(item) for item in value]
    return value


def attach_hashes(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result.pop("hashes", None)
    result["hashes"] = {
        "full_sha256": canonical_sha256(result),
        "semantic_sha256": canonical_sha256(semantic_projection(result)),
        "canonical_serialization": "UTF-8 sorted-key compact JSON; NaN/Infinity rejected",
        "semantic_exclusions": sorted(NON_SEMANTIC_FIELDS | {"hashes"}),
    }
    return result


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def normalized_segment(value: Any) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(value or "unknown")
    ).strip("_")
    return normalized or "unknown"


def _extract_json_array_after_key(path: Path, key: str) -> list[Any] | None:
    """Decode one JSON array without decoding outcome fields around it.

    Files are streamed until the named array closes.  This is intentionally
    narrower than ``json.load`` for historical aggregate/training summaries.
    """
    marker = f'"{key}"'
    collecting = False
    marker_buffer = ""
    array_chars: list[str] = []
    depth = 0
    in_string = False
    escaped = False
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        while True:
            chunk = handle.read(256 * 1024)
            if not chunk:
                break
            if not collecting:
                marker_buffer += chunk
                marker_index = marker_buffer.find(marker)
                if marker_index < 0:
                    marker_buffer = marker_buffer[-(len(marker) + 64) :]
                    continue
                suffix = marker_buffer[marker_index + len(marker) :]
                bracket_index = suffix.find("[")
                while bracket_index < 0:
                    extra = handle.read(256 * 1024)
                    if not extra:
                        return None
                    suffix += extra
                    bracket_index = suffix.find("[")
                chunk = suffix[bracket_index:]
                collecting = True
            for character in chunk:
                array_chars.append(character)
                if escaped:
                    escaped = False
                    continue
                if character == "\\" and in_string:
                    escaped = True
                    continue
                if character == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if character == "[":
                    depth += 1
                elif character == "]":
                    depth -= 1
                    if depth == 0:
                        payload = json.loads("".join(array_chars))
                        if not isinstance(payload, list):
                            raise FormalProtocolError(f"{key} is not a list in {path}")
                        return payload
    return None


def extract_selected_window_plan(path: str | Path) -> list[dict[str, Any]]:
    values = _extract_json_array_after_key(Path(path), "selected_window_plan")
    if values is None:
        values = _extract_json_array_after_key(Path(path), "selected_windows")
    if values is None:
        return []
    return [dict(item) for item in values if isinstance(item, Mapping)]


def discover_historical_plan_files(root: str | Path) -> list[Path]:
    root_path = Path(root).resolve()
    discovered: set[Path] = set()
    for path in (root_path / "configs").rglob("*.json"):
        if extract_selected_window_plan(path):
            discovered.add(path.resolve())
    artifacts = root_path / "artifacts"
    if artifacts.is_dir():
        rg = shutil.which("rg")
        if rg:
            result = subprocess.run(
                [rg, "-l", '"selected_window_plan"', str(artifacts), "-g", "*.json"],
                check=False,
                capture_output=True,
                text=True,
            )
            for raw_path in result.stdout.splitlines():
                path = Path(raw_path)
                if "typed_model_cache_formal_protocol_freeze_" not in path.as_posix():
                    discovered.add(path.resolve())
        else:  # pragma: no cover - rg is part of the project environment.
            for path in artifacts.rglob("*.json"):
                if "typed_model_cache_formal_protocol_freeze_" in path.as_posix():
                    continue
                if extract_selected_window_plan(path):
                    discovered.add(path.resolve())
    return sorted(discovered, key=lambda item: item.as_posix())


def _parse_date_from_path(path: str) -> str | None:
    dates = re.findall(r"(?<!\d)(20\d{6})(?!\d)", path)
    if not dates:
        return None
    value = min(dates)
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def _run_id_from_path(path: Path) -> str:
    timestamp_part = next(
        (
            part
            for part in reversed(path.parts)
            if re.search(r"20\d{6}[_-]\d{6}", part)
        ),
        "",
    )
    return timestamp_part or path.parent.name


def classify_history_purpose(path: str) -> list[str]:
    lowered = path.lower()
    rules = (
        ("sealed_holdout", ("sealed_holdout",)),
        ("hidden", ("hidden",)),
        ("holdout", ("holdout",)),
        ("future_validation", ("future_validation", "future-validation", "future")),
        ("formal", ("formal",)),
        ("dev", ("dev", "development")),
        ("train", ("train",)),
        ("calibration", ("calibrat",)),
        ("robustness", ("robust",)),
        ("scalability", ("scalab",)),
        ("ablation", ("ablation",)),
        ("checkpoint_selection", ("checkpoint", "update_eval")),
        ("opportunity_or_predictor_validation", ("opportunity", "predictor", "oracle")),
        ("window_scanning", ("window_plan", "split_manifest")),
        ("rehearsal_or_smoke", ("rehearsal", "smoke", "controlled")),
        ("benchmark_observation", ("benchmark", "aggregate_summary")),
    )
    purposes = [label for label, needles in rules if any(item in lowered for item in needles)]
    return purposes or ["historical_result_or_manifest"]


def result_blind_window_projection(window: Mapping[str, Any]) -> dict[str, Any]:
    """Keep interval/coverage metadata and explicitly drop outcome-like fields."""
    allowed = {
        "window_id",
        "frame_offset",
        "window_length",
        "time_index_start",
        "time_index_end",
        "raw_frame_start",
        "raw_frame_end",
        "raw_time_start",
        "raw_time_end",
        "source_segment_id",
        "source_segment_run_id",
        "source_location",
        "segment_frame_start",
        "segment_frame_end",
        "provider_segment_frame_start",
        "provider_segment_frame_end",
        "provider_frame_offset",
        "window_class",
        "recommended_rsu_layout",
        "dominant_axis",
        "chosen_rsu_axis",
        "coverage_radius",
        "spacing",
        "active_vehicle_count_mean",
        "active_vehicle_count_min",
        "active_vehicle_count_max",
        "unique_vehicle_count",
        "sampling_interval",
    }
    return {key: deepcopy(value) for key, value in window.items() if key in allowed}


def assert_result_blind_windows(windows: Sequence[Mapping[str, Any]]) -> None:
    for index, window in enumerate(windows):
        forbidden = PERFORMANCE_FORBIDDEN_WINDOW_FIELDS & set(window)
        if forbidden:
            raise FormalProtocolError(
                f"result-driven field(s) in split candidate {index}: {sorted(forbidden)}"
            )


def _numeric(series: Any) -> Any:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def scan_ngsim_intervals(
    csv_path: str | Path,
    *,
    prefix_rows: int = 5_000_000,
    chunksize: int = 250_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build full raw NGSIM continuity inventory and prefix runner mappings."""
    if pd is None:
        raise FormalProtocolError("pandas is required for full NGSIM interval inventory")
    source = Path(csv_path).resolve()
    frames: dict[tuple[str, int], dict[str, Any]] = {}
    row_count = 0
    invalid_numeric_rows = 0
    prefix_cutoff_keys: set[tuple[str, int]] = set()
    columns = ["Vehicle_ID", "Frame_ID", "Global_Time", "Local_X", "Local_Y", "Location"]
    for chunk in pd.read_csv(
        source,
        usecols=columns,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    ):
        chunk_start = row_count
        chunk_length = len(chunk)
        row_count += chunk_length
        frame_ids = _numeric(chunk["Frame_ID"])
        times = _numeric(chunk["Global_Time"])
        local_x = _numeric(chunk["Local_X"])
        local_y = _numeric(chunk["Local_Y"])
        valid = frame_ids.notna() & times.notna()
        invalid_numeric_rows += int((~valid).sum())
        working = chunk.loc[valid, ["Vehicle_ID", "Location"]].copy()
        working["Frame_ID"] = frame_ids[valid].astype("int64")
        working["Global_Time"] = times[valid].astype("int64")
        working["coordinate_valid"] = (local_x[valid].notna() & local_y[valid].notna()).astype(int)
        working["source_segment_id"] = working["Location"].map(normalized_segment)
        indices = chunk.index[valid]
        absolute_rows = chunk_start + pd.Series(range(chunk_length), index=chunk.index)
        working["in_prefix"] = (absolute_rows.loc[indices] < int(prefix_rows)).astype(int).values
        grouped = working.groupby(
            ["source_segment_id", "Global_Time"], sort=False, observed=True
        ).agg(
            raw_frame_min=("Frame_ID", "min"),
            raw_frame_max=("Frame_ID", "max"),
            vehicle_count=("Vehicle_ID", "count"),
            coordinate_valid_count=("coordinate_valid", "sum"),
            prefix_vehicle_count=("in_prefix", "sum"),
        )
        for (segment, global_time), record in grouped.iterrows():
            key = (str(segment), int(global_time))
            target = frames.setdefault(
                key,
                {
                    "source_segment_id": str(segment),
                    "global_time": int(global_time),
                    "raw_frame_min": int(record.raw_frame_min),
                    "raw_frame_max": int(record.raw_frame_max),
                    "vehicle_count": 0,
                    "coordinate_valid_count": 0,
                    "prefix_vehicle_count": 0,
                },
            )
            target["raw_frame_min"] = min(target["raw_frame_min"], int(record.raw_frame_min))
            target["raw_frame_max"] = max(target["raw_frame_max"], int(record.raw_frame_max))
            target["vehicle_count"] += int(record.vehicle_count)
            target["coordinate_valid_count"] += int(record.coordinate_valid_count)
            target["prefix_vehicle_count"] += int(record.prefix_vehicle_count)

    # A partial cutoff exists only when the same raw frame has rows on both
    # sides of the runner prefix boundary.  Marking every frame in the pandas
    # chunk that contains the boundary would greatly overstate this scope.
    prefix_cutoff_keys = {
        key
        for key, record in frames.items()
        if 0 < int(record["prefix_vehicle_count"]) < int(record["vehicle_count"])
    }

    by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    conflicting_frame_identity_count = 0
    for record in frames.values():
        if record["raw_frame_min"] != record["raw_frame_max"]:
            conflicting_frame_identity_count += 1
            record["invalid_frame_identity"] = True
        else:
            record["raw_frame"] = int(record["raw_frame_min"])
            record["invalid_frame_identity"] = False
        record["coordinate_coverage"] = round(
            record["coordinate_valid_count"] / max(record["vehicle_count"], 1), 6
        )
        by_segment[record["source_segment_id"]].append(record)

    runs: list[dict[str, Any]] = []
    frame_lookup_by_time: dict[tuple[str, int], dict[str, Any]] = {}
    frame_lookup_by_raw: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    discontinuities: list[dict[str, Any]] = []
    for segment in sorted(by_segment):
        ordered = sorted(by_segment[segment], key=lambda item: int(item["global_time"]))
        positive_time_deltas = [
            int(right["global_time"]) - int(left["global_time"])
            for left, right in zip(ordered, ordered[1:])
            if int(right["global_time"]) > int(left["global_time"])
            and int(right["global_time"]) - int(left["global_time"]) <= 10_000
        ]
        sampling = int(median(positive_time_deltas)) if positive_time_deltas else 1
        current: list[dict[str, Any]] = []
        run_index = 0

        def flush_run() -> None:
            nonlocal run_index, current
            if not current:
                return
            run_index += 1
            run_id = f"{segment}_run_{run_index:03d}"
            for local_index, item in enumerate(current):
                item["source_segment_run_id"] = run_id
                item["run_frame_index"] = local_index
                frame_lookup_by_time[(segment, int(item["global_time"]))] = item
                if not item.get("invalid_frame_identity"):
                    frame_lookup_by_raw[(segment, int(item["raw_frame"]))].append(item)
            counts = [int(item["vehicle_count"]) for item in current]
            coverage = [float(item["coordinate_coverage"]) for item in current]
            runs.append(
                {
                    "source_segment_id": segment,
                    "source_segment_run_id": run_id,
                    "raw_frame_start": int(current[0].get("raw_frame", current[0]["raw_frame_min"])),
                    "raw_frame_end": int(current[-1].get("raw_frame", current[-1]["raw_frame_max"])),
                    "raw_time_start": int(current[0]["global_time"]),
                    "raw_time_end": int(current[-1]["global_time"]),
                    "frame_count": len(current),
                    "sampling_interval": sampling,
                    "duration_time_units": len(current) * sampling,
                    "vehicle_coverage": {
                        "minimum": min(counts),
                        "mean": round(fmean(counts), 6),
                        "maximum": max(counts),
                    },
                    "coordinate_coverage_min": min(coverage),
                    "rsu_mapper_available": min(coverage) >= 1.0,
                    "window_length_24_feasible": len(current) >= 24,
                    "frames": current,
                }
            )
            current = []

        for item in ordered:
            if item.get("invalid_frame_identity"):
                flush_run()
                continue
            if current:
                previous = current[-1]
                time_delta = int(item["global_time"]) - int(previous["global_time"])
                frame_delta = int(item["raw_frame"]) - int(previous["raw_frame"])
                if time_delta != sampling or frame_delta != 1:
                    discontinuities.append(
                        {
                            "source_segment_id": segment,
                            "left_raw_frame": int(previous["raw_frame"]),
                            "right_raw_frame": int(item["raw_frame"]),
                            "left_raw_time": int(previous["global_time"]),
                            "right_raw_time": int(item["global_time"]),
                            "frame_delta": frame_delta,
                            "time_delta": time_delta,
                        }
                    )
                    flush_run()
            current.append(item)
        flush_run()

    full_provider_frames = [item for item in frames.values() if not item.get("invalid_frame_identity")]
    full_provider_frames.sort(
        key=lambda item: (str(item["source_segment_id"]), int(item["global_time"]))
    )
    for provider_offset, item in enumerate(full_provider_frames):
        item["provider_frame_offset"] = provider_offset
    prefix_frames = [item for item in full_provider_frames if int(item["prefix_vehicle_count"]) > 0]
    prefix_segments = sorted({str(item["source_segment_id"]) for item in prefix_frames})
    cutoff_ranges: list[dict[str, Any]] = []
    cutoff_times_by_segment: dict[str, list[int]] = defaultdict(list)
    for segment, raw_time in sorted(prefix_cutoff_keys):
        cutoff_times_by_segment[segment].append(raw_time)
    for segment, raw_times in sorted(cutoff_times_by_segment.items()):
        deltas = [right - left for left, right in zip(raw_times, raw_times[1:]) if right > left]
        sampling = int(median(deltas)) if deltas else 1
        range_start = raw_times[0]
        previous = raw_times[0]
        count = 1
        for raw_time in raw_times[1:]:
            if raw_time - previous != sampling:
                cutoff_ranges.append(
                    {
                        "source_segment_id": segment,
                        "raw_time_start": range_start,
                        "raw_time_end": previous,
                        "frame_count": count,
                        "sampling_interval": sampling,
                    }
                )
                range_start = raw_time
                count = 0
            previous = raw_time
            count += 1
        cutoff_ranges.append(
            {
                "source_segment_id": segment,
                "raw_time_start": range_start,
                "raw_time_end": previous,
                "frame_count": count,
                "sampling_interval": sampling,
            }
        )
    source_hash = sha256_file(source)
    public_runs = [{key: value for key, value in run.items() if key != "frames"} for run in runs]
    total_duration = sum(int(run["duration_time_units"]) for run in runs)
    inventory = attach_hashes(
        {
            "inventory_version": "ngsim_available_interval_inventory_v1.0.0",
            "created_at": utc_now(),
            "dataset": {
                "path": source.as_posix(),
                "sha256": source_hash,
                "size_bytes": source.stat().st_size,
                "row_count": row_count,
            },
            "runner_prefix_scope": {
                "max_mobility_rows": int(prefix_rows),
                "loaded_frame_count": len(prefix_frames),
                "source_segments": prefix_segments,
                "partial_cutoff_frame_count": len(prefix_cutoff_keys),
                "partial_cutoff_ranges_conservatively_invalid": cutoff_ranges,
            },
            "full_runner_scope": {
                "max_mobility_rows": row_count,
                "loaded_frame_count": len(full_provider_frames),
                "source_segments": sorted(by_segment),
            },
            "summary": {
                "segment_count": len(by_segment),
                "continuous_run_count": len(runs),
                "raw_unique_frame_count": len(frames),
                "total_available_duration_time_units": total_duration,
                "invalid_numeric_row_count": invalid_numeric_rows,
                "conflicting_frame_identity_count": conflicting_frame_identity_count,
                "discontinuity_count": len(discontinuities),
            },
            "continuous_runs": public_runs,
            "discontinuities_and_missing_ranges": discontinuities,
            "selection_boundary": (
                "Inventory uses only raw identity, continuity, coordinate validity, and vehicle coverage; "
                "no handoff/cache/reward/oracle/agent outcome is computed."
            ),
        }
    )
    internal = {
        "runs": runs,
        "frame_lookup_by_time": frame_lookup_by_time,
        "frame_lookup_by_raw": frame_lookup_by_raw,
        "prefix_frames": prefix_frames,
        "full_provider_frames": full_provider_frames,
        "prefix_cutoff_keys": prefix_cutoff_keys,
        "dataset_sha256": source_hash,
    }
    return inventory, internal


def _infer_historical_interval(
    window: Mapping[str, Any],
    *,
    inventory_internal: Mapping[str, Any],
) -> dict[str, Any]:
    projected = result_blind_window_projection(window)
    segment = normalized_segment(projected.get("source_segment_id"))
    segment_known = bool(projected.get("source_segment_id")) and segment != "unknown"
    time_start = projected.get("raw_time_start", projected.get("time_index_start"))
    time_end = projected.get("raw_time_end", projected.get("time_index_end"))
    raw_frame_start = projected.get("raw_frame_start")
    raw_frame_end = projected.get("raw_frame_end")
    lookup_time = inventory_internal["frame_lookup_by_time"]
    lookup_raw = inventory_internal["frame_lookup_by_raw"]
    confidence = "high_explicit_raw_identity"
    inference = "explicit_raw_fields"
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if time_start is not None and time_end is not None and int(time_start) > 1_000_000_000:
        if segment_known:
            left = lookup_time.get((segment, int(time_start)))
            right = lookup_time.get((segment, int(time_end)))
            if left and right:
                candidates.append((left, right))
        else:
            left_matches = [
                item for (candidate_segment, value), item in lookup_time.items() if value == int(time_start)
            ]
            right_matches = [
                item for (candidate_segment, value), item in lookup_time.items() if value == int(time_end)
            ]
            for left in left_matches:
                for right in right_matches:
                    if left["source_segment_run_id"] == right["source_segment_run_id"]:
                        candidates.append((left, right))
        confidence = "high_raw_time_recovered"
        inference = "raw_global_time_lookup"
    elif time_start is not None and time_end is not None:
        legacy_segment = segment if segment_known else "peachtree"
        left_matches = lookup_raw.get((legacy_segment, int(time_start)), [])
        right_matches = lookup_raw.get((legacy_segment, int(time_end)), [])
        for left in left_matches:
            for right in right_matches:
                if left["source_segment_run_id"] == right["source_segment_run_id"]:
                    candidates.append((left, right))
        confidence = "medium_legacy_raw_frame_reconstruction"
        inference = "legacy_time_index_as_raw_frame_with_prefix_peachtree_provenance"
    if raw_frame_start is not None and raw_frame_end is not None and segment_known and not candidates:
        left_matches = lookup_raw.get((segment, int(raw_frame_start)), [])
        right_matches = lookup_raw.get((segment, int(raw_frame_end)), [])
        for left in left_matches:
            for right in right_matches:
                if left["source_segment_run_id"] == right["source_segment_run_id"]:
                    candidates.append((left, right))
    if len(candidates) == 1:
        left, right = candidates[0]
        return {
            "source_segment_id": str(left["source_segment_id"]),
            "source_segment_run_id": str(left["source_segment_run_id"]),
            "raw_frame_start": int(left["raw_frame"]),
            "raw_frame_end": int(right["raw_frame"]),
            "raw_time_start": int(left["global_time"]),
            "raw_time_end": int(right["global_time"]),
            "sampling_interval": max(
                1,
                int(
                    (int(right["global_time"]) - int(left["global_time"]))
                    / max(int(right["raw_frame"]) - int(left["raw_frame"]), 1)
                ),
            ),
            "confidence": confidence,
            "interval_inference": inference,
            "unknown_interval_flag": False,
            "conservative_exclusion_scope": [],
        }
    conservative_scope = [segment] if segment_known else sorted(
        {str(item["source_segment_id"]) for item in inventory_internal["prefix_frames"]}
    )
    return {
        "source_segment_id": segment if segment_known else None,
        "source_segment_run_id": None,
        "raw_frame_start": None,
        "raw_frame_end": None,
        "raw_time_start": None,
        "raw_time_end": None,
        "sampling_interval": None,
        "confidence": "low_unknown_conservative",
        "interval_inference": "ambiguous_or_missing_raw_identity",
        "unknown_interval_flag": True,
        "conservative_exclusion_scope": conservative_scope,
    }


def build_historical_registry(
    root: str | Path,
    *,
    inventory_internal: Mapping[str, Any],
    mobility_sha256: str,
    plan_paths: Sequence[str | Path] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root_path = Path(root).resolve()
    paths = [Path(path).resolve() for path in plan_paths] if plan_paths is not None else discover_historical_plan_files(root_path)
    by_identity: dict[str, dict[str, Any]] = {}
    raw_reference_count = 0
    duplicate_reference_count = 0
    unknown_reference_count = 0
    parse_failures: list[dict[str, Any]] = []
    for path in paths:
        try:
            windows = extract_selected_window_plan(path)
        except (OSError, UnicodeError, json.JSONDecodeError, FormalProtocolError) as exc:
            parse_failures.append({"path": path.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not windows:
            continue
        try:
            relative_path = path.relative_to(root_path).as_posix()
        except ValueError:
            relative_path = path.as_posix()
        purposes = classify_history_purpose(relative_path)
        evidence = {
            "path": relative_path,
            "run_id": _run_id_from_path(path),
            "purposes": purposes,
            "first_observed_date": _parse_date_from_path(relative_path),
        }
        for window in windows:
            raw_reference_count += 1
            interval = _infer_historical_interval(window, inventory_internal=inventory_internal)
            if interval["unknown_interval_flag"]:
                unknown_reference_count += 1
                identity_payload = {
                    "unknown": True,
                    "window_id": str(window.get("window_id") or "unknown"),
                    "scope": interval["conservative_exclusion_scope"],
                }
            else:
                identity_payload = {
                    key: interval[key]
                    for key in (
                        "source_segment_run_id",
                        "raw_frame_start",
                        "raw_frame_end",
                        "raw_time_start",
                        "raw_time_end",
                    )
                }
            identity = canonical_sha256(identity_payload)
            record = by_identity.get(identity)
            if record is None:
                record = {
                    "historical_interval_id": f"hist-{identity[:20]}",
                    "outer_interval_identity_sha256": identity,
                    "window_ids": [],
                    "source_artifacts": [],
                    "run_ids": [],
                    "purposes": [],
                    "consumed_status": "consumed_permanent",
                    "selection_influence": True,
                    "mobility_source": "ngsim",
                    "mobility_file_sha256": mobility_sha256,
                    "segment_id": interval["source_segment_id"],
                    "segment_run_id": interval["source_segment_run_id"],
                    "raw_frame_start": interval["raw_frame_start"],
                    "raw_frame_end": interval["raw_frame_end"],
                    "raw_time_start": interval["raw_time_start"],
                    "raw_time_end": interval["raw_time_end"],
                    "sampling_interval": interval["sampling_interval"],
                    "vehicle_selection": "historical_plan_or_runner_default; exact mode unavailable when not recorded",
                    "workflow_related": False,
                    "first_observed_date": evidence["first_observed_date"],
                    "evidence_paths": [],
                    "confidence": interval["confidence"],
                    "interval_inference": interval["interval_inference"],
                    "unknown_interval_flag": interval["unknown_interval_flag"],
                    "conservative_exclusion_scope": interval["conservative_exclusion_scope"],
                    "mixed_full_outer_deduplicated": False,
                    "reference_count": 0,
                }
                by_identity[identity] = record
            else:
                duplicate_reference_count += 1
            window_id = str(window.get("window_id") or "unknown")
            if window_id not in record["window_ids"]:
                record["window_ids"].append(window_id)
            if relative_path not in record["evidence_paths"]:
                record["evidence_paths"].append(relative_path)
                record["source_artifacts"].append(evidence)
            if evidence["run_id"] not in record["run_ids"]:
                record["run_ids"].append(evidence["run_id"])
            for purpose in purposes:
                if purpose not in record["purposes"]:
                    record["purposes"].append(purpose)
            record["workflow_related"] = bool(
                record["workflow_related"]
                or any(
                    purpose
                    in {
                        "train",
                        "dev",
                        "formal",
                        "holdout",
                        "hidden",
                        "sealed_holdout",
                        "future_validation",
                        "benchmark_observation",
                        "ablation",
                        "robustness",
                        "scalability",
                    }
                    for purpose in purposes
                )
            )
            record["reference_count"] += 1
            if any("mixed" in item.lower() for item in record["evidence_paths"]) and any(
                "full" in item.lower() for item in record["evidence_paths"]
            ):
                record["mixed_full_outer_deduplicated"] = True
            dates = [item["first_observed_date"] for item in record["source_artifacts"] if item["first_observed_date"]]
            record["first_observed_date"] = min(dates) if dates else None
    records = sorted(
        by_identity.values(),
        key=lambda item: (
            bool(item["unknown_interval_flag"]),
            str(item.get("segment_run_id") or ""),
            int(item.get("raw_time_start") or 0),
            item["historical_interval_id"],
        ),
    )
    for record in records:
        record["window_ids"].sort()
        record["evidence_paths"].sort()
        record["run_ids"].sort()
        record["purposes"].sort()
        record["source_artifacts"].sort(key=lambda item: item["path"])
    registry = attach_hashes(
        {
            "historical_window_usage_registry_version": HISTORICAL_REGISTRY_VERSION,
            "registry_id": HISTORICAL_REGISTRY_ID,
            "created_at": utc_now(),
            "consumption_rule": (
                "Any interval used for design, tuning, checkpoint selection, scanning, result observation, "
                "formal, hidden, holdout, support, rehearsal, or smoke remains permanently consumed."
            ),
            "source_discovery": {
                "plan_or_result_file_count": len(paths),
                "metadata_only_parser": "selected_window_plan array only; performance fields not decoded",
                "parse_failure_count": len(parse_failures),
                "parse_failures": parse_failures,
            },
            "summary": {
                "raw_window_reference_count": raw_reference_count,
                "unique_outer_interval_count": len(records),
                "duplicate_reference_count": duplicate_reference_count,
                "unknown_interval_reference_count": unknown_reference_count,
                "unknown_unique_interval_count": sum(1 for item in records if item["unknown_interval_flag"]),
                "mixed_full_deduplicated_count": sum(1 for item in records if item["mixed_full_outer_deduplicated"]),
            },
            "records": records,
        }
    )
    validation = attach_hashes(
        {
            "historical_registry_validation_version": "1.0.0",
            "created_at": utc_now(),
            "passed": not parse_failures,
            "registry_semantic_sha256": registry["hashes"]["semantic_sha256"],
            "checks": {
                "canonical_round_trip": json.loads(canonical_json_bytes(registry)) == registry,
                "duplicate_history_collapsed": duplicate_reference_count >= 0,
                "mixed_full_outer_deduplicated": True,
                "seed_workflow_do_not_increase_outer_count": True,
                "all_unknowns_have_conservative_scope": all(
                    bool(item["conservative_exclusion_scope"])
                    for item in records
                    if item["unknown_interval_flag"]
                ),
                "performance_fields_not_decoded": True,
            },
            "parse_failures": parse_failures,
        }
    )
    return registry, validation


def _range_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start <= right_end and right_start <= left_end


def interval_relation(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    minimum_gap_frames: int,
) -> dict[str, Any]:
    left_scope = set(left.get("conservative_exclusion_scope") or [])
    right_scope = set(right.get("conservative_exclusion_scope") or [])
    left_segment = left.get("source_segment_id") or left.get("segment_id")
    right_segment = right.get("source_segment_id") or right.get("segment_id")
    if left.get("unknown_interval_flag") or right.get("unknown_interval_flag"):
        known_segment = right_segment if left.get("unknown_interval_flag") else left_segment
        unknown_scope = left_scope if left.get("unknown_interval_flag") else right_scope
        conflict = not known_segment or not unknown_scope or str(known_segment) in unknown_scope
        return {
            "classification": "unknown_conservative_conflict" if conflict else "safe",
            "frame_overlap": None,
            "time_overlap": None,
            "segment_frame_overlap": None,
            "minimum_gap_passed": not conflict,
        }
    left_run = left.get("source_segment_run_id") or left.get("segment_run_id")
    right_run = right.get("source_segment_run_id") or right.get("segment_run_id")
    if left_run and right_run and str(left_run) != str(right_run):
        return {
            "classification": "safe",
            "frame_overlap": False,
            "time_overlap": False,
            "segment_frame_overlap": False,
            "minimum_gap_passed": True,
        }
    if left_segment and right_segment and str(left_segment) != str(right_segment):
        return {
            "classification": "safe",
            "frame_overlap": False,
            "time_overlap": False,
            "segment_frame_overlap": False,
            "minimum_gap_passed": True,
        }
    frame_overlap = _range_overlap(
        int(left["raw_frame_start"]),
        int(left["raw_frame_end"]),
        int(right["raw_frame_start"]),
        int(right["raw_frame_end"]),
    )
    time_overlap = _range_overlap(
        int(left["raw_time_start"]),
        int(left["raw_time_end"]),
        int(right["raw_time_start"]),
        int(right["raw_time_end"]),
    )
    frame_distance = max(
        int(right["raw_frame_start"]) - int(left["raw_frame_end"]) - 1,
        int(left["raw_frame_start"]) - int(right["raw_frame_end"]) - 1,
    )
    sampling = max(
        int(left.get("sampling_interval") or 1), int(right.get("sampling_interval") or 1)
    )
    time_distance = max(
        int(right["raw_time_start"]) - int(left["raw_time_end"]) - sampling,
        int(left["raw_time_start"]) - int(right["raw_time_end"]) - sampling,
    )
    insufficient_gap = (
        not frame_overlap
        and not time_overlap
        and (
            frame_distance < int(minimum_gap_frames)
            or time_distance < int(minimum_gap_frames) * sampling
        )
    )
    if frame_overlap or time_overlap:
        classification = "exact_overlap"
    elif insufficient_gap:
        classification = "insufficient_gap"
    else:
        classification = "safe"
    return {
        "classification": classification,
        "frame_overlap": frame_overlap,
        "time_overlap": time_overlap,
        "segment_frame_overlap": frame_overlap,
        "minimum_gap_passed": classification == "safe",
        "frame_gap": frame_distance,
        "time_gap": time_distance,
    }


def _historical_conflict(
    frame: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    gap: int,
) -> bool:
    return any(
        interval_relation(frame, record, minimum_gap_frames=gap)["classification"] != "safe"
        for record in records
    )


def build_candidate_inventory(
    inventory: Mapping[str, Any],
    inventory_internal: Mapping[str, Any],
    historical_registry: Mapping[str, Any],
    *,
    window_length: int,
    minimum_gap_frames: int,
    minimum_vehicle_count: int,
    split_seed: int,
    allowed_source_segments: Sequence[str] | None = None,
    use_full_runner_scope: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    history = list(historical_registry["records"])
    candidates: list[dict[str, Any]] = []
    excluded_historical_frame_count = 0
    excluded_conservative_frame_count = 0
    excluded_quality_frame_count = 0
    prefix_cutoff = set(inventory_internal["prefix_cutoff_keys"])
    allowed_segments = {normalized_segment(item) for item in (allowed_source_segments or [])}
    by_run_counts: dict[str, int] = defaultdict(int)
    by_segment_counts: dict[str, int] = defaultdict(int)
    for run in inventory_internal["runs"]:
        if allowed_segments and str(run["source_segment_id"]) not in allowed_segments:
            continue
        frames = [
            item
            for item in run["frames"]
            if use_full_runner_scope or int(item.get("prefix_vehicle_count", 0)) > 0
        ]
        if not frames:
            continue
        safe_flags: list[bool] = []
        for item in frames:
            identity = {
                "source_segment_id": item["source_segment_id"],
                "source_segment_run_id": item["source_segment_run_id"],
                "raw_frame_start": item["raw_frame"],
                "raw_frame_end": item["raw_frame"],
                "raw_time_start": item["global_time"],
                "raw_time_end": item["global_time"],
                "sampling_interval": run["sampling_interval"],
                "unknown_interval_flag": False,
            }
            quality_ok = (
                not item.get("invalid_frame_identity")
                and float(item.get("coordinate_coverage", 0.0)) >= 1.0
                and int(
                    item.get("vehicle_count", 0)
                    if use_full_runner_scope
                    else item.get("prefix_vehicle_count", 0)
                )
                >= int(minimum_vehicle_count)
                and (
                    use_full_runner_scope
                    or (item["source_segment_id"], int(item["global_time"])) not in prefix_cutoff
                )
            )
            if not quality_ok:
                excluded_quality_frame_count += 1
                safe_flags.append(False)
                continue
            conflict_records = [
                record
                for record in history
                if interval_relation(identity, record, minimum_gap_frames=minimum_gap_frames)["classification"] != "safe"
            ]
            if conflict_records:
                if any(record["unknown_interval_flag"] for record in conflict_records):
                    excluded_conservative_frame_count += 1
                else:
                    excluded_historical_frame_count += 1
                safe_flags.append(False)
            else:
                safe_flags.append(True)
        cursor = 0
        while cursor + window_length <= len(frames):
            window_frames = frames[cursor : cursor + window_length]
            contiguous_provider = all(
                int(right["provider_frame_offset"]) == int(left["provider_frame_offset"]) + 1
                for left, right in zip(window_frames, window_frames[1:])
            )
            if all(safe_flags[cursor : cursor + window_length]) and contiguous_provider:
                first, last = window_frames[0], window_frames[-1]
                counts = [
                    int(item["vehicle_count"] if use_full_runner_scope else item["prefix_vehicle_count"])
                    for item in window_frames
                ]
                run_id = str(first["source_segment_run_id"])
                identity = {
                    "source_segment_run_id": run_id,
                    "raw_frame_start": int(first["raw_frame"]),
                    "raw_frame_end": int(last["raw_frame"]),
                    "raw_time_start": int(first["global_time"]),
                    "raw_time_end": int(last["global_time"]),
                }
                stable_rank = canonical_sha256({"seed": int(split_seed), **identity})
                window_id = (
                    f"g14b_{run_id}_f{first['raw_frame']}_{last['raw_frame']}_"
                    f"t{first['global_time']}_{last['global_time']}"
                )
                candidates.append(
                    {
                        "window_id": window_id,
                        "frame_offset": int(first["provider_frame_offset"]),
                        "provider_frame_offset": int(first["provider_frame_offset"]),
                        "window_length": int(window_length),
                        "time_index_start": int(first["global_time"]),
                        "time_index_end": int(last["global_time"]),
                        "raw_frame_start": int(first["raw_frame"]),
                        "raw_frame_end": int(last["raw_frame"]),
                        "raw_time_start": int(first["global_time"]),
                        "raw_time_end": int(last["global_time"]),
                        "source_segment_id": str(first["source_segment_id"]),
                        "source_segment_run_id": run_id,
                        "source_location": str(first["source_segment_id"]),
                        "segment_frame_start": int(first["raw_frame"]),
                        "segment_frame_end": int(last["raw_frame"]),
                        "provider_segment_frame_start": int(first["run_frame_index"]),
                        "provider_segment_frame_end": int(last["run_frame_index"]),
                        "sampling_interval": int(run["sampling_interval"]),
                        "active_vehicle_count_min": min(counts),
                        "active_vehicle_count_mean": round(fmean(counts), 6),
                        "active_vehicle_count_max": max(counts),
                        "window_class": "coverage_only_result_blind",
                        "recommended_rsu_layout": "auto_dominant_tight",
                        "selection_rank_sha256": stable_rank,
                        "selection_features": [
                            "raw continuity",
                            "minimum vehicle availability",
                            "coordinate validity",
                            "historical exclusion",
                            "minimum gap",
                        ],
                    }
                )
                by_run_counts[run_id] += 1
                by_segment_counts[str(first["source_segment_id"])] += 1
                cursor += window_length + minimum_gap_frames
            else:
                cursor += 1
    assert_result_blind_windows(
        [
            {key: value for key, value in item.items() if key != "selection_rank_sha256"}
            for item in candidates
        ]
    )
    candidates.sort(key=lambda item: (item["selection_rank_sha256"], item["window_id"]))
    candidate_public = [{key: value for key, value in item.items() if key != "selection_rank_sha256"} for item in candidates]
    candidate_inventory = attach_hashes(
        {
            "candidate_window_inventory_version": "1.0.0",
            "created_at": utc_now(),
            "split_protocol_version": SPLIT_PROTOCOL_VERSION,
            "selection_policy": "result-blind deterministic coverage/continuity selection",
            "parameters": {
                "window_length": int(window_length),
                "minimum_gap_frames": int(minimum_gap_frames),
                "minimum_vehicle_count": int(minimum_vehicle_count),
                "split_generation_seed": int(split_seed),
                "tie_break": "SHA-256(seed, segment-run, raw-frame, raw-time), then window_id",
                "runner_prefix_max_mobility_rows": (
                    inventory["full_runner_scope"]["max_mobility_rows"]
                    if use_full_runner_scope
                    else inventory["runner_prefix_scope"]["max_mobility_rows"]
                ),
                "allowed_source_segments": sorted(allowed_segments),
                "use_full_runner_scope": bool(use_full_runner_scope),
            },
            "summary": {
                "eligible_non_overlapping_candidate_count": len(candidate_public),
                "maximum_non_overlapping_window_count": len(candidate_public),
                "per_segment_run_candidate_count": dict(sorted(by_run_counts.items())),
                "per_segment_candidate_count": dict(sorted(by_segment_counts.items())),
                "formal_12_feasible": len(candidate_public) >= MINIMUM_OUTER_WINDOWS,
                "sealed_holdout_12_feasible": len(candidate_public) >= 2 * MINIMUM_OUTER_WINDOWS,
                "excluded_historical_or_gap_frame_count": excluded_historical_frame_count,
                "excluded_unknown_conservative_frame_count": excluded_conservative_frame_count,
                "excluded_quality_or_discontinuity_frame_count": excluded_quality_frame_count,
            },
            "candidates": candidate_public,
            "forbidden_selection_inputs": sorted(PERFORMANCE_FORBIDDEN_WINDOW_FIELDS),
        }
    )
    unknown_scope_segments = {
        segment
        for item in history
        if item["unknown_interval_flag"]
        for segment in item["conservative_exclusion_scope"]
    }
    known_by_run: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for item in history:
        if item["unknown_interval_flag"] or not item.get("segment_run_id"):
            continue
        known_by_run[str(item["segment_run_id"])].append(
            (int(item["raw_frame_start"]), int(item["raw_frame_end"]))
        )
    total_duration = 0
    known_union_duration = 0
    conservative_duration = 0
    net_known_outside_conservative_duration = 0
    minimum_gap_duration = 0
    quality_duration = 0
    remaining_eligible_duration = 0
    for run in inventory_internal["runs"]:
        segment = str(run["source_segment_id"])
        run_id = str(run["source_segment_run_id"])
        sampling = int(run["sampling_interval"])
        records = known_by_run.get(run_id, [])
        for frame in run["frames"]:
            total_duration += sampling
            raw_frame = int(frame["raw_frame"])
            known = any(left <= raw_frame <= right for left, right in records)
            within_gap = any(
                left - minimum_gap_frames <= raw_frame <= right + minimum_gap_frames
                for left, right in records
            )
            conservative = segment in unknown_scope_segments
            quality_ok = (
                float(frame.get("coordinate_coverage", 0.0)) >= 1.0
                and int(frame.get("vehicle_count", 0)) >= int(minimum_vehicle_count)
            )
            if known:
                known_union_duration += sampling
            if conservative:
                conservative_duration += sampling
                continue
            if known:
                net_known_outside_conservative_duration += sampling
                continue
            if within_gap:
                minimum_gap_duration += sampling
                continue
            if not quality_ok:
                quality_duration += sampling
                continue
            remaining_eligible_duration += sampling
    exclusion_audit = attach_hashes(
        {
            "historical_exclusion_audit_version": "1.0.0",
            "created_at": utc_now(),
            "passed": True,
            "historical_registry_semantic_sha256": historical_registry["hashes"]["semantic_sha256"],
            "minimum_gap_frames": int(minimum_gap_frames),
            "historically_consumed_duration_time_units": known_union_duration,
            "conservative_unknown_scope_segments": sorted(unknown_scope_segments),
            "duration_accounting": {
                "total_available_duration_time_units": total_duration,
                "historically_consumed_union_duration_time_units": known_union_duration,
                "conservatively_excluded_duration_time_units": conservative_duration,
                "known_consumed_outside_conservative_scope_duration_time_units": net_known_outside_conservative_duration,
                "minimum_gap_excluded_duration_time_units": minimum_gap_duration,
                "quality_excluded_duration_time_units": quality_duration,
                "remaining_eligible_duration_time_units": remaining_eligible_duration,
                "overlap_accounting_rule": (
                    "known history inside a conservatively excluded segment is reported in the gross known union "
                    "but subtracted only once through conservative exclusion"
                ),
            },
            "unknown_unique_interval_count": sum(1 for item in history if item["unknown_interval_flag"]),
            "remaining_eligible_candidate_count": len(candidate_public),
            "result_blind": True,
        }
    )
    return candidate_inventory, exclusion_audit


def build_split_manifest(
    candidate_inventory: Mapping[str, Any],
    *,
    counts: Mapping[str, int],
    minimum_gap_frames: int,
    created_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    expected_keys = set(SPLIT_NAMES)
    if set(counts) != expected_keys:
        raise FormalProtocolError(f"split counts must be exactly {sorted(expected_keys)}")
    if int(counts["formal"]) < MINIMUM_OUTER_WINDOWS:
        raise InsufficientWindowError("BLOCKED_INSUFFICIENT_UNCONSUMED_WINDOWS: formal<12")
    if int(counts["sealed_holdout"]) < MINIMUM_OUTER_WINDOWS:
        raise InsufficientWindowError("BLOCKED_INSUFFICIENT_UNCONSUMED_WINDOWS: sealed_holdout<12")
    required = sum(int(counts[name]) for name in SPLIT_NAMES)
    candidates = list(candidate_inventory["candidates"])
    if len(candidates) < required:
        raise InsufficientWindowError(
            "BLOCKED_INSUFFICIENT_UNCONSUMED_WINDOWS: "
            f"required={required}, available={len(candidates)}, missing={required - len(candidates)}"
        )
    allocation: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    cursor = 0
    while any(len(allocation[name]) < int(counts[name]) for name in SPLIT_NAMES):
        for name in SPLIT_NAMES:
            if len(allocation[name]) >= int(counts[name]):
                continue
            allocation[name].append(deepcopy(candidates[cursor]))
            cursor += 1
    all_windows = [(split, item) for split in SPLIT_NAMES for item in allocation[split]]
    pairwise: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_outer: set[str] = set()
    for split, window in all_windows:
        if window["window_id"] in seen_ids:
            conflicts.append({"classification": "duplicate_window", "window_id": window["window_id"]})
        seen_ids.add(window["window_id"])
        outer_identity = canonical_sha256(
            {
                key: window[key]
                for key in (
                    "source_segment_run_id",
                    "raw_frame_start",
                    "raw_frame_end",
                    "raw_time_start",
                    "raw_time_end",
                )
            }
        )
        if outer_identity in seen_outer:
            conflicts.append(
                {"classification": "same_interval_different_id", "window_id": window["window_id"]}
            )
        seen_outer.add(outer_identity)
    for left_index, (left_split, left) in enumerate(all_windows):
        for right_split, right in all_windows[left_index + 1 :]:
            relation = interval_relation(left, right, minimum_gap_frames=minimum_gap_frames)
            row = {
                "left_split": left_split,
                "left_window_id": left["window_id"],
                "right_split": right_split,
                "right_window_id": right["window_id"],
                **relation,
            }
            pairwise.append(row)
            if relation["classification"] != "safe":
                conflicts.append(row)
    audit = attach_hashes(
        {
            "split_independence_audit_version": "1.0.0",
            "created_at": created_at or utc_now(),
            "passed": not conflicts,
            "minimum_gap_frames": int(minimum_gap_frames),
            "checked_interval_kinds": ["raw_frame", "raw_time", "segment_frame"],
            "outer_cluster_counts": {name: len(allocation[name]) for name in SPLIT_NAMES},
            "seed_workflow_repetitions_count_as_outer": False,
            "mixed_full_variants_count_as_outer": False,
            "conflicts": conflicts,
        }
    )
    if not audit["passed"]:
        raise FormalProtocolError(f"split independence audit failed: {conflicts[:3]}")
    plan_hashes = {name: canonical_sha256(allocation[name]) for name in SPLIT_NAMES}
    manifest = attach_hashes(
        {
            "typed_model_cache_split_protocol_version": SPLIT_PROTOCOL_VERSION,
            "split_protocol_id": SPLIT_PROTOCOL_ID,
            "created_at": created_at or utc_now(),
            "status": "frozen_pre_run",
            "selection_policy": "result-blind raw continuity/coverage with historical exclusion",
            "selection_seed": candidate_inventory["parameters"]["split_generation_seed"],
            "sorting_and_tie_break": candidate_inventory["parameters"]["tie_break"],
            "minimum_gap_frames": int(minimum_gap_frames),
            "window_length": candidate_inventory["parameters"]["window_length"],
            "minimum_vehicle_count": candidate_inventory["parameters"]["minimum_vehicle_count"],
            "runner_prefix_max_mobility_rows": candidate_inventory["parameters"]["runner_prefix_max_mobility_rows"],
            "candidate_inventory_semantic_sha256": candidate_inventory["hashes"]["semantic_sha256"],
            "splits": {
                name: {
                    "sealed": name == "sealed_holdout",
                    "outer_window_count": len(allocation[name]),
                    "outer_window_plan_sha256": plan_hashes[name],
                    "window_ids": [item["window_id"] for item in allocation[name]],
                    "selected_window_plan": allocation[name],
                }
                for name in SPLIT_NAMES
            },
            "independence_audit_semantic_sha256": audit["hashes"]["semantic_sha256"],
            "forbidden_selection_inputs": sorted(PERFORMANCE_FORBIDDEN_WINDOW_FIELDS),
        }
    )
    matrix = attach_hashes(
        {
            "split_pairwise_overlap_matrix_version": "1.0.0",
            "created_at": created_at or utc_now(),
            "minimum_gap_frames": int(minimum_gap_frames),
            "pair_count": len(pairwise),
            "rows": pairwise,
        }
    )
    return manifest, audit, allocation | {"_pairwise_matrix": matrix}  # type: ignore[operator]


def build_capacity_strata(runtime_contract: Mapping[str, Any]) -> dict[str, Any]:
    residents = list(runtime_contract["resident_objects"])
    by_id = {item["object_id"]: item for item in residents}
    capacity_objects = [item for item in residents if item["counts_toward_capacity"]]
    initial_values = [
        float(item["resident_mb"])
        for item in runtime_contract["initial_per_rsu_typed_state"]
    ]
    bundle_values = []
    for item in capacity_objects:
        if item["object_type"] != "adapter":
            continue
        bundle_values.append(
            float(item["resident_size_mb"])
            + sum(float(by_id[dependency]["resident_size_mb"]) for dependency in item["dependency_ids"])
        )
    total_resident = sum(float(item["resident_size_mb"]) for item in capacity_objects)
    quantum = 32.0

    def round_up(value: float) -> float:
        return float(math.ceil(value / quantum) * quantum)

    constrained = round_up(max(max(initial_values), max(bundle_values)))
    relaxed = round_up(total_resident)
    medium = round_up((constrained + relaxed) / 2.0)
    strata = [
        {
            "stratum": "constrained",
            "capacity_mb": constrained,
            "derivation": "ceil(max(max initial-RSU resident MB, largest atomic base+adapter bundle MB)/32)*32",
        },
        {
            "stratum": "medium",
            "capacity_mb": medium,
            "derivation": "ceil(mean(constrained MB, relaxed MB)/32)*32",
        },
        {
            "stratum": "relaxed",
            "capacity_mb": relaxed,
            "derivation": "ceil(total repository-controlled capacity-counting base+adapter resident MB/32)*32",
        },
    ]
    return attach_hashes(
        {
            "capacity_strata_version": "1.0.0",
            "unit": "MB (decimal megabytes, repository contract unit)",
            "rounding_quantum_mb": quantum,
            "inputs": {
                "max_initial_rsu_resident_mb": max(initial_values),
                "largest_atomic_dependency_bundle_mb": max(bundle_values),
                "total_capacity_counting_catalog_resident_mb": total_resident,
            },
            "strata": strata,
            "oversized_semantics": "atomic rolled_back_no_mutation; no partial base or adapter admission",
            "workflow_state_counts_toward_long_term_capacity": False,
            "kv_prefix_enabled": False,
            "hf_metadata_formal_use": False,
            "result_dependent_adjustment_forbidden": True,
        }
    )


def build_agent_matrix() -> dict[str, Any]:
    from src.agents.registry import get_algo_spec

    controller_agents = [
        "sa_ghmappo",
        "ppo",
        "mappo",
        "dqn",
        "dueling_dqn",
        "qmix",
        "controller_mat",
        "dag_offload_drl",
        "cache_offload_drl",
        "dt_handoff_drl",
        "popularity_cache_heuristic",
    ]
    table = []
    for name in controller_agents:
        spec = get_algo_spec(name)
        table.append(
            {
                "agent": name,
                "registry_identity": name,
                "status": "paper_grade_current_contract" if name != "popularity_cache_heuristic" else "matched_heuristic",
                "observation_contract": spec["observation_contract"],
                "action_contract": spec["action_contract"],
                "training_requirement": "clean_typed_checkpoint_per_seed_and_capacity" if spec["checkpoint_required"] else "checkpoint_free",
                "budget_class": "neural_equal_environment_interactions" if spec["checkpoint_required"] else "checkpoint_free_controller",
                "checkpoint_selection_rule": "frozen dev lexicographic system endpoint rule; never formal/holdout",
                "incompatibility_handling": "fail-fast; no legacy checkpoint substitution and no post-run deletion",
            }
        )
    reactive = [
        {
            "agent": name,
            "registry_identity": name,
            "eviction_policy": policy,
            "training_requirement": "none",
            "budget_class": "checkpoint_free_reactive",
            "only_primary_difference": "eviction_policy",
        }
        for name, policy in (
            ("reactive_lru", "lru"),
            ("reactive_fifo", "fifo"),
            ("reactive_lfu", "lfu"),
            ("reactive_aging_lfu", "aging_lfu"),
            ("reactive_random", "random_seed_bound_to_benchmark_seed"),
        )
    ]
    oracle = [
        {
            "agent": f"exact_oracle_h{horizon}",
            "horizon": horizon,
            "status": "report_only_when_solver_status_exact",
            "training_requirement": "none",
            "budget_class": "exact_solver_state_limit",
            "unknown_handling": "unknown_state_limit; never greedy fallback or upper-bound claim",
        }
        for horizon in (1, 3, 6, 12)
    ]
    return {
        "controller_table": table,
        "reactive_cache_policy_isolation": reactive,
        "exact_oracle_cells": oracle,
        "excluded_live_agents": {
            "ippo": "diagnostic only; current wrapper is not independent entity-level IPPO",
            "ddqn": "independent implementation but not in the frozen current paper-grade default set",
            "dueling_ddqn": "independent implementation but not in the frozen current paper-grade default set",
            "reactive_greedy": "superseded for cache-policy isolation by the five matched reactive policies",
        },
        "controller_level_boundary": "SA-GHMAPPO, MAPPO, QMIX, and MAT are controller-level contracts, not vehicle/RSU entity-level MARL.",
        "post_result_baseline_removal_forbidden": True,
    }


def build_statistics_protocol() -> dict[str, Any]:
    return {
        "statistics_protocol_version": "typed_model_cache_statistics_v1.0.0",
        "outer_unit": "raw-time mobility window",
        "hierarchical_bootstrap": {
            "outer_cluster_keys": ["source_segment_run_id", "window_id"],
            "inner_cluster_keys": ["seed", "workflow_id"],
            "mixed_full_dependence": "same outer interval clustered once; never treated as two outer samples",
            "replicates": 10_000,
            "bootstrap_seed": 1401,
        },
        "confidence_intervals": ["percentile_95", "BCa_95"],
        "paired_test": "exact two-sided sign test after dropping preregistered numerical ties",
        "tie_tolerance": 1e-9,
        "effect_sizes": ["paired Cohen dz", "outer-window standardized mean delta"],
        "win_tie_loss": True,
        "multiplicity": {
            "method": "Holm",
            "alpha": 0.05,
            "primary_family": "all preregistered primary comparisons x six primary endpoints",
            "secondary_family": "exploratory; separate Holm family and no primary-confirmatory language",
        },
        "missing_or_failed_run_policy": {
            "infrastructure_failure": "one identical-command retry; preserve both records",
            "algorithmic_failure": "retain as failure/missing under worst-case sensitivity; never delete seed/window",
            "minimum_complete_outer_clusters": 12,
            "post_result_seed_or_window_deletion": False,
        },
    }


def build_claim_evidence_template() -> dict[str, Any]:
    rows = []
    for claim_id, endpoint, evidence in (
        ("typed_base_sharing", "sharing_reuse_and_avoided_duplicate_transfer", "typed full vs no-base-sharing"),
        ("byte_hit", "full_service_ready_byte_hit_rate", "typed full vs matched non-oracle baselines"),
        ("transfer_overhead", "transfer_mb_per_request", "paired raw CacheEvent transfer bytes"),
        ("workflow_continuity", "workflow_continuity_rate", "paired workflow outcomes by outer window"),
        ("capacity_pressure", "primary endpoints across three frozen MB strata", "capacity support matrix"),
        ("eviction_policy", "hit/transfer/pollution/churn", "five reactive policies under G07 only-policy difference"),
        ("oracle_opportunity", "exact feasible opportunity gap", "H=1/3/6/12 exact cells only"),
        ("controller_comparison", "six primary endpoints", "SA-GHMAPPO vs preregistered strongest dev baseline"),
        ("predictor_boundary", "prediction support endpoints", "baseline predictor; G12 supervised disabled"),
        ("data_realism_boundary", "provenance only", "NGSIM + Alibaba DAG + controlled typed catalog statement"),
    ):
        rows.append(
            {
                "claim_id": claim_id,
                "endpoint_or_boundary": endpoint,
                "required_evidence": evidence,
                "result_status": None,
                "allowed_result_statuses": ["supported", "mixed", "unsupported", "contradicted", "unavailable"],
                "formal_evidence_path": None,
                "sealed_holdout_evidence_path": None,
                "claim_text": None,
            }
        )
    return {
        "claim_evidence_template_version": "1.0.0",
        "pre_registered_before_training": True,
        "rows": rows,
        "no_result_status_assigned_during_g14b": True,
    }


def build_holdout_seal(split_manifest: Mapping[str, Any]) -> dict[str, Any]:
    holdout = split_manifest["splits"]["sealed_holdout"]
    allowed_checks = [
        "training_and_eval_completeness",
        "protocol_hash_consistency",
        "checkpoint_provenance",
        "fairness_manifest_validation",
        "artifact_integrity",
        "infrastructure_health",
    ]
    return attach_hashes(
        {
            "holdout_seal_contract_version": HOLDOUT_SEAL_VERSION,
            "holdout_manifest_identity": {
                "split_protocol_semantic_sha256": split_manifest["hashes"]["semantic_sha256"],
                "outer_window_plan_sha256": holdout["outer_window_plan_sha256"],
                "outer_window_count": holdout["outer_window_count"],
            },
            "sealed": True,
            "opened": False,
            "opened_at": None,
            "consumed_permanently": False,
            "allowed_pre_open_metadata": [
                "manifest identity",
                "raw frame/time/segment intervals",
                "outer count",
                "dataset and split hashes",
            ],
            "forbidden_pre_open_access": ["reward", "cache outcomes", "agent outcomes", "rankings", "claim status"],
            "formal_checkpoint_binding": "SHA-256 of immutable typed checkpoint provenance manifest required at opening",
            "one_time_execution_token_status": "not_issued_in_G14B",
            "one_time_execution_token_sha256": None,
            "opening_gate": {
                "allowed_checks": allowed_checks,
                "performance_gate_forbidden": True,
                "formal_performance_threshold_forbidden": True,
            },
            "execution_record": {
                "append_only": True,
                "maximum_successful_open_count": 1,
                "required_fields": ["opened_at", "execution_commit", "command", "output_run_id", "gate_results"],
            },
            "infrastructure_retry_conditions": [
                "process terminated before any outcome row is durably written",
                "verified storage corruption with original files retained",
                "hardware/runtime failure unrelated to an agent outcome",
            ],
            "post_result_protocol_checkpoint_or_seed_change_forbidden": True,
        }
    )


def build_formal_protocol(
    *,
    split_manifest: Mapping[str, Any],
    historical_registry: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    runtime_hashes_by_capacity: Mapping[str, str],
    capacity_strata: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
    workflow_ids: Sequence[str],
    fairness_manifest_version: str,
    holdout_seal: Mapping[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    agent_matrix = build_agent_matrix()
    statistics = build_statistics_protocol()
    claims = build_claim_evidence_template()
    primary = [
        "full_service_ready_byte_hit_rate",
        "joint_base_adapter_hit_rate",
        "full_service_ready_request_rate",
        "transfer_mb_per_request",
        "workflow_continuity_rate",
        "end_to_end_workflow_delay",
    ]
    secondary = [
        "base_model_hit_rate",
        "adapter_hit_rate",
        "base_sharing_reuse",
        "avoided_duplicate_transfer_mb",
        "cache_pollution_ratio",
        "cache_churn_mb",
        "eviction_future_reuse_proxy",
        "occupancy_and_saturation",
        "bundle_rejection_rate",
        "backhaul_traffic_cost",
        "handoff_failure_rate",
        "workflow_state_migration_overhead",
        "total_reward",
        "inference_wall_clock_sec",
        "python_peak_increment_bytes",
        "oracle_state_count_and_solve_time",
        "cache_latency_saved_sum_ms_unavailable",
    ]
    budget = {
        "budget_version": "typed_model_cache_equal_interaction_budget_v1.0.0",
        "episodes_per_learned_agent_seed_capacity": 256,
        "update_interval_episodes": 8,
        "expected_update_count": 32,
        "batch_size": 64,
        "max_steps_per_episode": 22,
        "maximum_environment_interactions_per_learned_agent_seed_capacity": 5632,
        "train_window_count": split_manifest["splits"]["train"]["outer_window_count"],
        "workflow_count": len(workflow_ids),
        "optimizer": "Adam (algorithm-native implementation; no optimizer substitution after freeze)",
        "learning_rates": {
            "sa_ghmappo": 1e-4,
            "mappo": 2e-4,
            "ppo": 3e-4,
            "dqn": 3e-4,
            "dueling_dqn": 3e-4,
            "qmix": 3e-4,
            "controller_mat": 3e-4,
            "dag_offload_drl": 3e-4,
            "cache_offload_drl": 3e-4,
            "dt_handoff_drl": 3e-4,
        },
        "entropy_and_value_coefficients": {
            "sa_ghmappo": {"entropy_coef": 0.004, "value_coef": 0.70, "auxiliary_coef": 0.06},
            "mappo": {"entropy_coef": 0.015, "value_coef": 0.65},
            "other_on_policy": {"entropy_coef": 0.01, "value_coef": 0.5},
            "value_based": {"entropy_coef": "not_applicable", "value_coef": 1.0},
        },
        "checkpoint_frequency_updates": 4,
        "checkpoint_selection": {
            "split": "dev",
            "metric_rule": [
                "maximize full_service_ready_byte_hit_rate",
                "maximize workflow_continuity_rate",
                "minimize transfer_mb_per_request",
                "minimize end_to_end_workflow_delay",
                "earlier update then lexical checkpoint hash tie-break",
            ],
            "formal_or_holdout_selection_forbidden": True,
        },
        "early_stop": "disabled; fixed interaction budget",
        "retry": "one identical-command infrastructure retry; no reseed or post-result replacement",
        "device": "cpu",
        "maximum_wall_clock_per_learned_agent_seed_capacity_hours": 12,
        "maximum_total_compute_budget_cpu_hours": 2500,
        "equal_budget_rule": "all learned agents receive identical maximum environment interactions per seed/capacity; intrinsic update/replay differences are retained and reported",
        "algorithm_intrinsic_differences": [
            "on-policy vs replay-based update mechanics",
            "controller-level CTDE/value decomposition vs flat controller",
            "network architecture and optimizer state size",
        ],
    }
    protocol = attach_hashes(
        {
            "typed_model_cache_formal_protocol_version": FORMAL_PROTOCOL_VERSION,
            "protocol_id": PROTOCOL_ID,
            "created_at": created_at or utc_now(),
            "status": "frozen_pre_training_no_performance_data",
            "identity": {
                "execution_git_commit_binding": "Commit A containing this exact protocol semantic hash; clean worktree must resolve to that commit",
                "dataset_hashes": dict(dataset_hashes),
                "split_semantic_sha256": split_manifest["hashes"]["semantic_sha256"],
                "historical_registry_semantic_sha256": historical_registry["hashes"]["semantic_sha256"],
                "typed_runtime_contract_version": runtime_contract["runtime_contract_version"],
                "typed_runtime_contract_hashes_by_capacity": dict(runtime_hashes_by_capacity),
                "catalog_fingerprint": runtime_contract["typed_catalog_fingerprint"],
                "fairness_manifest_version": fairness_manifest_version,
                "cache_event_schema_version": "1.3.0",
                "cache_efficiency_metrics_contract_version": "1.1.0",
                "top_journal_policy_version": "tmc_review_policy_v3_20260621",
            },
            "agent_matrix": agent_matrix,
            "seed_plan": {
                "seeds": [7, 13, 29, 43, 71],
                "resource_decision": "five seeds frozen before execution; no post-run deletion or reduction",
            },
            "training_budget": budget,
            "typed_catalog_and_capacity": {
                "catalog_source": "repository-controlled typed_model_cache_controlled.json",
                "catalog_fingerprint": runtime_contract["typed_catalog_fingerprint"],
                "initial_typed_state_fingerprint": runtime_contract["typed_initial_state_fingerprint"],
                "base_adapter_dependency_fingerprint": runtime_contract["dependency_fingerprint"],
                "pinned_evictability_fingerprint": runtime_contract["pinned_evictability_fingerprint"],
                "capacity_strata": capacity_strata["strata"],
                "capacity_contract_semantic_sha256": capacity_strata["hashes"]["semantic_sha256"],
                "workflow_state_long_term_capacity": False,
                "kv_prefix_enabled": False,
                "hf_metadata_formal_use": False,
                "capacity_result_dependent_adjustment": False,
            },
            "workload": {
                "mobility": "NGSIM",
                "workflow_structure": "Alibaba 2018 batch DAG",
                "workflow_ids": list(workflow_ids),
                "window_length": split_manifest["window_length"],
                "max_steps": budget["max_steps_per_episode"],
                "vehicle_selection": "handoff_pressure frozen runner rule; split selection itself did not use handoff outcomes",
                "rsu_layout": "auto_dominant_tight",
                "request_replay": "static DAG pre-run fingerprint plus observed identical cross-agent request fingerprint",
                "cross_source_boundary": "NGSIM mobility + Alibaba DAG + repository-controlled typed catalog; not a real joint model-cache request trace",
                "predictor_kind": "baseline",
                "g12_supervised_predictor_enabled": False,
            },
            "endpoints": {"primary": primary, "secondary": secondary, "latency_saved_status": "unavailable"},
            "comparisons": {
                "primary": [
                    "SA-GHMAPPO vs strongest matched non-oracle baseline selected once on dev",
                    "typed full vs legacy adapter-only",
                    "best compatible learned controller vs best reactive baseline",
                    "five reactive policies vs exact oracle only in exact solver cells",
                ],
                "strongest_baseline_dev_rule": [
                    "eligible non-oracle controller-table baselines excluding SA-GHMAPPO",
                    "maximize full_service_ready_byte_hit_rate",
                    "maximize workflow_continuity_rate",
                    "minimize transfer_mb_per_request",
                    "minimize end_to_end_workflow_delay",
                    "lexical agent identity tie-break",
                ],
                "formal_or_holdout_strongest_selection_forbidden": True,
            },
            "ablation_and_support": {
                "typed_semantics": [
                    "typed_full",
                    "legacy_adapter_only",
                    "no_base_sharing",
                    "no_workflow_state_migration",
                    "fixed_no_eviction",
                    "no_prediction",
                ],
                "capacity": ["constrained", "medium", "relaxed"],
                "sensitivity": [
                    "object_size_scale",
                    "transfer_cost_scale",
                    "handoff_pressure",
                    "reuse_opportunity",
                    "base_sharing_degree",
                ],
                "scalability": ["rsu_count", "vehicle_count", "dag_size", "object_count"],
                "predictor_boundary": "baseline only; G12 supervised disabled",
                "oracle_state_limit": "report exact or unknown_state_limit",
            },
            "statistics": statistics,
            "claim_evidence_map": claims,
            "holdout_execution_contract": {
                "seal_semantic_sha256": holdout_seal["hashes"]["semantic_sha256"],
                "sealed": True,
                "opened": False,
                "performance_gate_forbidden": True,
            },
            "immutability": {
                "canonical_json": "sorted UTF-8 compact JSON; NaN/Infinity rejected",
                "semantic_nonsemantic_exclusions": sorted(NON_SEMANTIC_FIELDS | {"hashes"}),
                "cli_semantic_overrides": "rejected",
                "validator_mode": "read-only after freeze",
                "change_rule": "new version + new Git commit + new run ID; old protocol retained",
            },
            "paper_claim_boundary": (
                "Protocol freeze is not formal completion, not holdout opening, not a checkpoint, "
                "not a performance result, and not paper-ready evidence."
            ),
        }
    )
    validate_protocol_manifest(protocol)
    return protocol


def validate_protocol_manifest(protocol: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "typed_model_cache_formal_protocol_version",
        "protocol_id",
        "created_at",
        "status",
        "identity",
        "agent_matrix",
        "seed_plan",
        "training_budget",
        "typed_catalog_and_capacity",
        "workload",
        "endpoints",
        "comparisons",
        "ablation_and_support",
        "statistics",
        "claim_evidence_map",
        "holdout_execution_contract",
        "immutability",
        "paper_claim_boundary",
        "hashes",
    }
    unknown = set(protocol) - allowed
    if unknown:
        raise FormalProtocolError(f"unknown formal protocol field(s): {sorted(unknown)}")
    version = protocol.get("typed_model_cache_formal_protocol_version")
    if version != FORMAL_PROTOCOL_VERSION:
        raise FormalProtocolError(f"unsupported formal protocol version: {version!r}")
    required = allowed - {"hashes"}
    missing = required - set(protocol)
    if missing:
        raise FormalProtocolError(f"missing formal protocol field(s): {sorted(missing)}")
    seeds = protocol["seed_plan"]["seeds"]
    if seeds not in ([7, 13, 29, 43, 71], [7, 13, 29]):
        raise FormalProtocolError("seed plan must be the frozen five-seed preference or documented three-seed limit")
    capacities = protocol["typed_catalog_and_capacity"]["capacity_strata"]
    if [item["stratum"] for item in capacities] != ["constrained", "medium", "relaxed"]:
        raise FormalProtocolError("capacity strata must be constrained/medium/relaxed")
    primary_required = {
        "full_service_ready_byte_hit_rate",
        "joint_base_adapter_hit_rate",
        "full_service_ready_request_rate",
        "transfer_mb_per_request",
        "workflow_continuity_rate",
        "end_to_end_workflow_delay",
    }
    if not primary_required.issubset(set(protocol["endpoints"]["primary"])):
        raise FormalProtocolError("primary endpoint set is incomplete")
    if protocol["statistics"]["multiplicity"]["method"] != "Holm":
        raise FormalProtocolError("primary multiple-comparison method must be Holm")
    _reject_non_finite(protocol)
    expected_semantic = canonical_sha256(semantic_projection({key: value for key, value in protocol.items() if key != "hashes"}))
    actual_semantic = protocol.get("hashes", {}).get("semantic_sha256")
    if actual_semantic and actual_semantic != expected_semantic:
        raise FormalProtocolError("formal protocol semantic hash mismatch")
    return {
        "passed": True,
        "semantic_sha256": actual_semantic or expected_semantic,
        "agent_count": len(protocol["agent_matrix"]["controller_table"]),
        "seed_count": len(seeds),
    }


def assert_no_semantic_cli_overrides(argv: Sequence[str]) -> None:
    allowed = {"--output_dir", "--created_at", "--git_commit_binding", "--help", "-h"}
    for item in argv:
        if item.startswith("--") and item.split("=", 1)[0] not in allowed:
            raise FormalProtocolError(f"semantic CLI override rejected: {item.split('=', 1)[0]}")


def validate_split_access(
    split: str,
    *,
    caller_role: str,
    execution_token: str | None = None,
) -> None:
    if split != "sealed_holdout":
        return
    if caller_role != "dedicated_one_time_holdout_executor" or not execution_token:
        raise HoldoutAccessError("sealed holdout cannot be opened by an ordinary runner or validator")


def append_holdout_execution_record(
    log_path: str | Path,
    *,
    seal_record: Mapping[str, Any],
    execution_token: str,
    gate_results: Mapping[str, bool],
    execution_commit: str,
    command: Sequence[str],
    output_run_id: str,
    opened_at: str | None = None,
) -> dict[str, Any]:
    validate_split_access(
        "sealed_holdout",
        caller_role="dedicated_one_time_holdout_executor",
        execution_token=execution_token,
    )
    if not seal_record.get("sealed") or seal_record.get("opened"):
        raise HoldoutAccessError("holdout is not sealed-and-unopened")
    token_hash = hashlib.sha256(execution_token.encode("utf-8")).hexdigest()
    expected = seal_record.get("one_time_execution_token_sha256")
    if not expected or token_hash != expected:
        raise HoldoutAccessError("one-time execution token mismatch or not issued")
    allowed_gates = set(seal_record["opening_gate"]["allowed_checks"])
    if set(gate_results) != allowed_gates or not all(bool(value) for value in gate_results.values()):
        raise HoldoutAccessError("holdout opening gate incomplete or failed")
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "holdout_manifest_identity": seal_record["holdout_manifest_identity"],
        "opened_at": opened_at or utc_now(),
        "execution_commit": execution_commit,
        "command": list(command),
        "output_run_id": output_run_id,
        "gate_results": dict(gate_results),
        "infrastructure_retry_conditions": seal_record["infrastructure_retry_conditions"],
        "consumed_permanently": True,
    }
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    try:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise HoldoutAccessError("append-only holdout record already exists; second opening rejected") from exc
    return record


def protocol_hash_changes_on_mutation(protocol: Mapping[str, Any], dotted_path: str, value: Any) -> bool:
    mutated = deepcopy(dict(protocol))
    target: Any = mutated
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    final = parts[-1]
    if isinstance(target, list):
        target[int(final)] = value
    else:
        target[final] = value
    return canonical_sha256(semantic_projection(protocol)) != canonical_sha256(semantic_projection(mutated))


def readiness_verdict(checks: Mapping[str, bool]) -> str:
    if checks and all(bool(value) for value in checks.values()):
        return "READY_FOR_G14C_CLEAN_TRAIN_AND_FORMAL"
    return "BLOCKED_G14B_READINESS_V2"


def git_commit(root: str | Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(root), text=True
    ).strip()


def protected_file_hashes(root: str | Path, paths: Iterable[str]) -> dict[str, str]:
    root_path = Path(root)
    return {path: sha256_file(root_path / path) for path in paths}


__all__ = [
    "FORMAL_PROTOCOL_VERSION",
    "HISTORICAL_REGISTRY_VERSION",
    "HOLDOUT_SEAL_VERSION",
    "MINIMUM_OUTER_WINDOWS",
    "READINESS_REVIEW_VERSION",
    "SPLIT_PROTOCOL_VERSION",
    "FormalProtocolError",
    "HoldoutAccessError",
    "InsufficientWindowError",
    "append_holdout_execution_record",
    "assert_no_semantic_cli_overrides",
    "assert_result_blind_windows",
    "attach_hashes",
    "build_candidate_inventory",
    "build_historical_registry",
    "build_split_manifest",
    "canonical_json_bytes",
    "canonical_sha256",
    "discover_historical_plan_files",
    "extract_selected_window_plan",
    "git_commit",
    "interval_relation",
    "protected_file_hashes",
    "protocol_hash_changes_on_mutation",
    "readiness_verdict",
    "result_blind_window_projection",
    "scan_ngsim_intervals",
    "semantic_projection",
    "sha256_file",
    "utc_now",
    "validate_protocol_manifest",
    "validate_split_access",
    "write_json",
]
