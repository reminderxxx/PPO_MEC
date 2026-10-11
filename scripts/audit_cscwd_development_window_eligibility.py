"""Outcome-blind source qualification for frozen CSCWD development intervals."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config


PARENT_MANIFEST = ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json"
PARENT_SHA = "b7efe5d6e50ad17b744230c03a126370656d0196b60c7f7e6b5a86bb702effbc"
SPLIT_MANIFEST = ROOT / "configs/experiment/top_journal_v28_executable_handoff_strict_split_20260719/split_manifest.json"
SPLIT_SHA = "246fcacebb04051e5181a1b2eacf702e942532bbf69e5a4c2e6d1194e98ee322"
RAW_SHA = "ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c"
TARGET_SECONDS = 118.8
TARGET_FRAMES = 1189
WINDOW_ID = re.compile(r"^window_(.+)_off(\d+)_len(\d+)_t(\d+)_(\d+)$")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def segment(location: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in str(location)).strip("_") or "unknown"


def numeric(raw: str) -> int:
    return int(float(str(raw).replace(",", "")))


def parse_window_id(window_id: str) -> tuple[str, int, int, int, int]:
    match = WINDOW_ID.fullmatch(window_id)
    if match is None:
        raise ValueError(f"unparseable window ID: {window_id}")
    return (match.group(1), *(int(match.group(index)) for index in range(2, 6)))


def gap_ms(a: dict, b: dict) -> int | None:
    if a["source_segment_id"] != b["source_segment_id"]:
        return None
    return max(int(a["time_index_start"]) - int(b["time_index_end"]),
               int(b["time_index_start"]) - int(a["time_index_end"]))


def cost_threshold(parent: dict) -> dict:
    config, _ = _load_experiment_config(ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json")
    instances = parent["instances"]
    sums = [sum(float(node["compute_seconds"]) for node in row["nodes"]) for row in instances]
    compute_min, compute_max = min(sums), max(sums)
    slowest_mbps = min(float(row["link_profile"]["actual_mbps"]) for row in instances)
    fixed = float(config["link"]["fixed_seconds"])
    catalog = config["object_catalog"]
    cold_bundle_seconds = max(
        sum(int(catalog[item]["transfer_bytes"]) for item in items) * 8 / (slowest_mbps * 1_000_000)
        + fixed + sum(float(catalog[item]["load_seconds"]) for item in items)
        for items in config["adapter_to_bundle"].values()
    )
    input_seconds = (max(int(node["input_bytes"]) for row in instances for node in row["nodes"])
                     * 8 / (slowest_mbps * 1_000_000) + fixed)
    normal_state_bytes = max(int(node["state_bytes"]) for row in instances
                             if int(row["factors"]["state_scale"]) <= 64 for node in row["nodes"])
    normal_state_seconds = (normal_state_bytes * 8 / (slowest_mbps * 1_000_000) + fixed
                            + float(config["measured_time_seconds"]["state_restore_overhead"]))
    fallback = float(config["vehicle"]["fallback_seconds"])
    target = math.ceil((1.10 * (compute_max + cold_bundle_seconds + input_seconds + normal_state_seconds + fallback) + 0.1) * 10) / 10
    frames = math.ceil(target / 0.1) + 1
    if not math.isclose(target, TARGET_SECONDS) or frames != TARGET_FRAMES:
        raise RuntimeError("frozen engineering target no longer matches parent workload")
    return {"minimum_compute_only_seconds": compute_min,
            "maximum_compute_only_seconds": compute_max,
            "slowest_configured_actual_mbps": slowest_mbps,
            "largest_one_cold_bundle_seconds": cold_bundle_seconds,
            "largest_one_input_seconds": input_seconds,
            "largest_normal_state_network_restore_seconds": normal_state_seconds,
            "one_vehicle_fallback_seconds": fallback,
            "engineering_margin_fraction": 0.10,
            "first_observation_seconds": 0.1,
            "target_source_span_seconds": target,
            "target_source_frame_count": frames,
            "claim": "engineering_source_qualification_not_a_guarantee_of_service_or_migration"}


def load_plan_rows(parent: dict, split: dict) -> list[dict]:
    rows: list[dict] = []
    for role, key in (("train", "train_window_plan"), ("dev", "evaluation_window_plan")):
        source = parent["source_files"][key]
        path = ROOT / source["path"]
        if sha(path) != source["sha256"] or source["sha256"] != split["plans"][role]["sha256"]:
            raise RuntimeError(f"{role} source plan SHA mismatch")
        plan = json.loads(path.read_text())
        if plan.get("split") != role or plan.get("sealed") is not False:
            raise RuntimeError(f"{role} source plan is not open development")
        if len(plan["selected_window_plan"]) != split["plans"][role]["window_count"]:
            raise RuntimeError(f"{role} window count mismatch")
        for index, window in enumerate(plan["selected_window_plan"]):
            parsed = parse_window_id(window["window_id"])
            segment_id, offset, length, start, end = parsed
            if (segment_id != window["source_segment_id"] or offset != int(window["frame_offset"])
                    or length != int(window["window_length"]) or start != int(window["time_index_start"])
                    or end != int(window["time_index_end"])):
                raise RuntimeError(f"{role} source interval differs from window ID")
            if window["window_id"] not in split["plans"][role]["window_ids"]:
                raise RuntimeError(f"{role} source window absent from split manifest")
            rows.append({"plan_role": role, "plan_index": index, "window_id": window["window_id"],
                         "source_segment_id": segment_id, "frame_offset": offset,
                         "window_length": length, "time_index_start": start, "time_index_end": end})
    return sorted(rows, key=lambda row: (row["source_segment_id"], row["time_index_start"], row["window_id"]))


def forbidden_interval_metadata(split: dict) -> list[dict]:
    result = []
    for role in ("formal", "hidden_holdout"):
        for window_id in split["plans"][role]["window_ids"]:
            segment_id, offset, length, start, end = parse_window_id(window_id)
            result.append({"plan_role": role, "source_segment_id": segment_id,
                           "time_index_start": start, "time_index_end": end})
    return result


def allowed_contiguous_spans(rows: list[dict]) -> list[dict]:
    spans = []
    for segment_id in sorted({row["source_segment_id"] for row in rows}):
        sequence = sorted((row for row in rows if row["source_segment_id"] == segment_id),
                          key=lambda row: row["time_index_start"])
        for row in sequence:
            if not spans or spans[-1]["source_segment_id"] != segment_id or row["time_index_start"] > spans[-1]["time_index_end"] + 100:
                spans.append({"source_segment_id": segment_id,
                              "time_index_start": row["time_index_start"],
                              "time_index_end": row["time_index_end"], "window_count": 1})
            else:
                spans[-1]["time_index_end"] = max(spans[-1]["time_index_end"], row["time_index_end"])
                spans[-1]["window_count"] += 1
    return spans


def inspect_raw(path: Path, rows: list[dict]) -> tuple[dict[str, dict[int, list[tuple[int, int]]]], int]:
    target: dict[tuple[str, int], str] = {}
    for row in rows:
        for index in range(int(row["window_length"])):
            timestamp = int(row["time_index_start"]) + 100 * index
            key = (str(row["source_segment_id"]), timestamp)
            if key in target:
                raise RuntimeError("development windows overlap at a raw timestamp")
            target[key] = str(row["window_id"])
    target_times = {str(timestamp) for _, timestamp in target}
    hits: dict[str, dict[int, list[tuple[int, int]]]] = {row["window_id"]: defaultdict(list) for row in rows}
    scanned = 0
    for chunk in pd.read_csv(path, usecols=["Vehicle_ID", "Frame_ID", "Global_Time", "Location"],
                             dtype=str, keep_default_na=False, chunksize=500_000):
        scanned += len(chunk)
        subset = chunk[chunk["Global_Time"].isin(target_times)]
        if subset.empty:
            continue
        for vehicle_id, frame_id, global_time, location in subset.itertuples(index=False, name=None):
            timestamp = numeric(global_time)
            window_id = target.get((segment(location), timestamp))
            if window_id is not None:
                hits[window_id][timestamp].append((numeric(vehicle_id), numeric(frame_id)))
    return hits, scanned


def qualify(row: dict, hits: dict[int, list[tuple[int, int]]], forbidden: list[dict], threshold: dict) -> dict:
    result = dict(row)
    reasons = []
    expected = [int(row["time_index_start"]) + 100 * index for index in range(int(row["window_length"]))]
    result["source_timestamp_count"] = len(hits)
    result["matched_raw_rows"] = sum(len(value) for value in hits.values())
    result["raw_time_span_seconds"] = (expected[-1] - expected[0]) / 1000
    result["decision_time_available_seconds"] = max(0.0, result["raw_time_span_seconds"] - 0.1)
    if expected[-1] != int(row["time_index_end"]) or sorted(hits) != expected:
        reasons.append("source_timestamp_gap_or_interval_mismatch")
    if int(row["time_index_start"]) < 10**12:
        reasons.append("unit_ambiguous_short_global_time")
    first = hits.get(expected[0], [])
    if not first:
        reasons.append("first_frame_missing")
        vehicle_id = None
    else:
        vehicle_id = min(item[0] for item in first)
    result["selected_vehicle_id"] = vehicle_id
    selected = []
    for timestamp in expected:
        matching = [frame for candidate, frame in hits.get(timestamp, []) if candidate == vehicle_id]
        if len(matching) != 1:
            break
        if selected and matching[0] != selected[-1] + 1:
            break
        selected.append(matching[0])
    result["selected_vehicle_contiguous_frames_from_start"] = len(selected)
    result["selected_vehicle_frame_start"] = selected[0] if selected else None
    result["selected_vehicle_frame_end"] = selected[-1] if selected else None
    result["selected_vehicle_decision_available_seconds"] = max(0.0, (len(selected) - 2) * 0.1)
    if len(selected) != len(expected):
        reasons.append("selected_vehicle_not_unique_contiguous_for_full_interval")
    conflicts = [entry["plan_role"] for entry in forbidden if gap_ms(row, entry) is not None and gap_ms(row, entry) <= 0]
    result["formal_hidden_interval_conflict_count"] = len(conflicts)
    if conflicts:
        reasons.append("formal_or_hidden_interval_overlap")
    if result["selected_vehicle_decision_available_seconds"] < float(threshold["minimum_compute_only_seconds"]):
        reasons.append("below_any_workflow_compute_only_lower_bound")
    if len(selected) < TARGET_FRAMES:
        reasons.append("below_frozen_engineering_source_duration")
    result["qualified"] = not reasons
    result["rejection_reasons"] = reasons
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if sha(PARENT_MANIFEST) != PARENT_SHA or sha(SPLIT_MANIFEST) != SPLIT_SHA:
        raise RuntimeError("parent/split manifest identity mismatch")
    parent = json.loads(PARENT_MANIFEST.read_text())
    split = json.loads(SPLIT_MANIFEST.read_text())
    if split["hidden_holdout_policy"].get("sealed") is not True or split["hidden_holdout_policy"].get("opened_at") is not None:
        raise RuntimeError("hidden holdout metadata no longer sealed/unopened")
    if sha(args.raw_csv_path) != RAW_SHA:
        raise RuntimeError("raw NGSIM identity mismatch")
    threshold = cost_threshold(parent)
    rows = load_plan_rows(parent, split)
    if len(rows) != 40:
        raise RuntimeError("expected exactly 40 authorized train/dev windows")
    forbidden = forbidden_interval_metadata(split)
    hits, scanned = inspect_raw(args.raw_csv_path, rows)
    results = [qualify(row, hits[row["window_id"]], forbidden, threshold) for row in rows]
    if any(result["qualified"] for result in results):
        raise RuntimeError("new qualified source requires separately frozen reachability run")
    spans = allowed_contiguous_spans(rows)
    allowed_pairs = [gap_ms(a, b) for index, a in enumerate(rows) for b in rows[index + 1:]]
    forbidden_pairs = [gap_ms(a, b) for a in rows for b in forbidden]
    allowed_gaps = [gap for gap in allowed_pairs if gap is not None]
    forbidden_gaps = [gap for gap in forbidden_pairs if gap is not None]
    if any(gap <= 100 for gap in allowed_gaps) or any(gap <= 0 for gap in forbidden_gaps):
        raise RuntimeError("allowed spans touch/overlap or forbidden split conflicts")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    output.mkdir(parents=True)
    report = {"schema_version": "cscwd_development_window_eligibility_v1",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "parent_manifest_sha256": PARENT_SHA,
              "split_manifest_sha256": SPLIT_SHA,
              "raw_source_sha256": RAW_SHA,
              "raw_rows_scanned": scanned,
              "split_metadata_roles_read": ["train", "dev", "formal_interval_ids_only", "hidden_holdout_interval_ids_only"],
              "hidden_holdout_opened": False,
              "threshold": threshold,
              "candidate_count": len(results),
              "qualified_count": 0,
              "decision": "NO_QUALIFIED_DEVELOPMENT_INTERVAL",
              "maximum_authorized_contiguous_source_span_seconds": max((span["time_index_end"] - span["time_index_start"]) / 1000 for span in spans),
              "allowed_contiguous_span_count": len(spans),
              "minimum_same_segment_allowed_gap_seconds": min(allowed_gaps) / 1000,
              "minimum_allowed_to_formal_hidden_gap_seconds": min(forbidden_gaps) / 1000,
              "formal_hidden_overlap_count": sum(result["formal_hidden_interval_conflict_count"] for result in results),
              "candidates": results}
    (output / "eligibility_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "candidate_count": len(results),
                      "qualified_count": 0, "max_contiguous_seconds": report["maximum_authorized_contiguous_source_span_seconds"],
                      "decision": report["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
