"""Freeze outcome-blind long NGSIM development windows without reading outcomes."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW_SHA = "ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c"
PARENT_SHA = "b7efe5d6e50ad17b744230c03a126370656d0196b60c7f7e6b5a86bb702effbc"
PLAN = ROOT / "docs/project/cscwd_new_development_source_plan_20261011.md"
PLAN_SHA = "1643e3ed4fee1b05c57f2e0aee46f99dd479ecc2fb9c489d4bb629e0626a6ac9"
WINDOW_FRAMES = 1189
EMBARGO_MS = 2400  # inclusive blocked endpoints => at least 2500 ms to the next sample
SOURCE_SEGMENTS = {"us_101", "lankershim"}
STRICT_SPLITS = {
    "top_journal_v27_segmented_strict_split_20260719": "6e724186cfb18032b27e4f1ef44ce949e2b9298a78de24893da293ca50f41cd3",
    "top_journal_v28_executable_handoff_strict_split_20260719": "246fcacebb04051e5181a1b2eacf702e942532bbf69e5a4c2e6d1194e98ee322",
    "top_journal_v71_strict_split_20260730": "2110347990464f12532a73886f6f617e461a18b4e781d940da2172b6ca3eeff5",
}
ANONYMOUS_MANIFESTS = {
    "top_journal_v8_strict_split_20260621/split_manifest.json": "c974a18443d4b6600c765bc71af486308b2fbe0f0fe6be4e094abbd37937f53a",
    "top_journal_v17_future_validation_time_audited_20260717/future_validation_manifest.json": "449ed33ff52e8d760601c91dd7421091a4e99bf472abe91f1a93766503cfe41c",
    "top_journal_v20_formal_time_audited_20260717/future_validation_manifest.json": "f0be3b4d48f9f243a233e891c10965902ed51b98e605d2e40262140d7ac84d3a",
    "top_journal_v20_future_validation_time_audited_20260717/future_validation_manifest.json": "fe15b77e9dd4337ee259aca47ac0369b609ff92b25f5d319064d5e5cea7e34ea",
    "top_journal_v20_hidden_time_audited_20260717/future_validation_manifest.json": "61648d88fa295649025adf26f573ec80db738c7c1e3dd3ccf9dce7f61e37d4db",
}
WINDOW_ID = re.compile(r"^window_(.+)_off(\d+)_len(\d+)_t(\d+)_(\d+)$")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def segment(raw: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in str(raw)).strip("_")


def strict_metadata() -> tuple[list[dict], list[dict], list[dict]]:
    forbidden, exposed, records = [], [], []
    for directory, expected_sha in STRICT_SPLITS.items():
        path = ROOT / "configs/experiment" / directory / "split_manifest.json"
        if sha(path) != expected_sha:
            raise RuntimeError(f"strict manifest changed: {path}")
        data = json.loads(path.read_text())
        if data["source_records"]["mobility"]["sha256"] != RAW_SHA:
            raise RuntimeError("strict manifest points at another mobility source")
        if int(data["parameters"]["minimum_gap_frames"]) != 24:
            raise RuntimeError("strict split embargo changed")
        records.append({"path": str(path.relative_to(ROOT)), "sha256": expected_sha})
        for role in ("train", "dev", "formal", "hidden_holdout"):
            window_ids = data["plans"][role]["window_ids"]
            if len(window_ids) != int(data["plans"][role]["window_count"]):
                raise RuntimeError("strict split ID count mismatch")
            for window_id in window_ids:
                match = WINDOW_ID.fullmatch(window_id)
                if match is None:
                    raise RuntimeError(f"unparseable strict window: {window_id}")
                source, _, length, start, end = match.groups()
                if int(length) != 24 or int(end) - int(start) != 2300:
                    raise RuntimeError("strict raw window duration mismatch")
                if source not in SOURCE_SEGMENTS:
                    continue
                row = {"source_segment_id": source, "time_start": int(start),
                       "time_end": int(end), "role": role, "source_manifest": directory,
                       "window_id": window_id}
                (forbidden if role in {"formal", "hidden_holdout"} else exposed).append(row)
    return forbidden, exposed, records


def anonymous_metadata() -> list[dict]:
    records = []
    for name, expected_sha in ANONYMOUS_MANIFESTS.items():
        path = ROOT / "configs/experiment" / name
        if sha(path) != expected_sha:
            raise RuntimeError(f"anonymous source manifest changed: {path}")
        data = json.loads(path.read_text())
        if data["source_records"]["mobility"]["sha256"] != RAW_SHA:
            raise RuntimeError("anonymous manifest points at another mobility source")
        rows = int(data["parameters"]["max_mobility_rows"])
        if rows > 200_000:
            raise RuntimeError("anonymous historical row scope exceeds frozen quarantine")
        records.append({"path": str(path.relative_to(ROOT)), "sha256": expected_sha,
                        "max_mobility_rows": rows, "identity_status": "anonymous_no_raw_time_mapping"})
    return records


def merge_blocks(blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(blocks):
        if merged and start <= merged[-1][1] + 100:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def allowed_fragments(start: int, end: int, blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return inclusive time fragments outside all merged blocked intervals."""
    result = []
    cursor = start
    for blocked_start, blocked_end in blocks:
        if blocked_end < cursor:
            continue
        if blocked_start > end:
            break
        if cursor < blocked_start:
            result.append((cursor, min(end, blocked_start - 100)))
        cursor = max(cursor, blocked_end + 100)
        if cursor > end:
            break
    if cursor <= end:
        result.append((cursor, end))
    return result


def load_source(path: Path) -> tuple[pd.DataFrame, dict, int, int]:
    pieces = []
    prefix: dict[str, list[int]] = {}
    scanned = 0
    invalid_identity_rows = 0
    for chunk in pd.read_csv(
        path,
        usecols=["Vehicle_ID", "Frame_ID", "Global_Time", "Local_X", "Local_Y", "Location"],
        dtype=str,
        keep_default_na=False,
        chunksize=500_000,
    ):
        chunk["source_segment_id"] = chunk["Location"].map(segment)
        relevant = chunk[chunk["source_segment_id"].isin(SOURCE_SEGMENTS)].copy()
        for column in ("Vehicle_ID", "Frame_ID", "Global_Time", "Local_X", "Local_Y"):
            relevant[column] = pd.to_numeric(
                relevant[column].str.replace(",", "", regex=False), errors="coerce"
            )
        if scanned < 200_000:
            prefix_chunk = chunk.iloc[: max(0, 200_000 - scanned)]
            prefix_rows = prefix_chunk[prefix_chunk["source_segment_id"].isin(SOURCE_SEGMENTS)]
            for source, group in prefix_rows.groupby("source_segment_id"):
                times = pd.to_numeric(group["Global_Time"].str.replace(",", "", regex=False), errors="coerce")
                if times.isna().any():
                    raise RuntimeError("anonymous prefix contains unknown source time")
                bounds = prefix.setdefault(source, [int(times.min()), int(times.max())])
                bounds[0] = min(bounds[0], int(times.min()))
                bounds[1] = max(bounds[1], int(times.max()))
        scanned += len(chunk)
        if len(relevant):
            relevant["source_csv_row"] = np.arange(scanned - len(chunk), scanned, dtype=np.int64)[
                chunk["source_segment_id"].isin(SOURCE_SEGMENTS).to_numpy()
            ]
            missing_identity = relevant[["Vehicle_ID", "Frame_ID", "Global_Time"]].isna().any(axis=1)
            invalid_identity_rows += int(missing_identity.sum())
            relevant = relevant.loc[~missing_identity].copy()
            for column in ("Vehicle_ID", "Frame_ID", "Global_Time"):
                relevant[column] = relevant[column].astype(np.int64)
            pieces.append(relevant.drop(columns="Location"))
    if not pieces:
        raise RuntimeError("no source rows in frozen segments")
    data = pd.concat(pieces, ignore_index=True)
    data.sort_values(["source_segment_id", "Vehicle_ID", "Global_Time", "Frame_ID"],
                     kind="mergesort", inplace=True, ignore_index=True)
    return data, prefix, scanned, invalid_identity_rows


def qualify(data: pd.DataFrame, blocks_by_segment: dict[str, list[tuple[int, int]]],
            exposed: list[dict]) -> tuple[list[dict], list[dict], dict]:
    source = data["source_segment_id"].to_numpy()
    vehicle = data["Vehicle_ID"].to_numpy(dtype=np.int64)
    time = data["Global_Time"].to_numpy(dtype=np.int64)
    frame = data["Frame_ID"].to_numpy(dtype=np.int64)
    finite = np.isfinite(data["Local_X"].to_numpy(dtype=float)) & np.isfinite(data["Local_Y"].to_numpy(dtype=float))
    valid = finite & (time >= 10**12) & (time < 10**13) & (time % 100 == 0)
    duplicates = data.duplicated(["source_segment_id", "Vehicle_ID", "Global_Time"], keep=False).to_numpy()
    valid &= ~duplicates
    boundary = np.ones(len(data), dtype=bool)
    boundary[1:] = (~valid[1:] | ~valid[:-1] | (source[1:] != source[:-1])
                    | (vehicle[1:] != vehicle[:-1]) | (time[1:] - time[:-1] != 100)
                    | (frame[1:] - frame[:-1] != 1))
    starts = np.flatnonzero(boundary)
    ends = np.r_[starts[1:], len(data)]
    candidates: list[dict] = []
    counters = Counter()
    for lo, hi in zip(starts, ends):
        if not valid[lo]:
            counters["invalid_source_rows"] += 1
            continue
        seg = str(source[lo])
        run_start, run_end = int(time[lo]), int(time[hi - 1])
        fragments = allowed_fragments(run_start, run_end, blocks_by_segment.get(seg, []))
        if not fragments:
            counters["fully_protected_runs"] += 1
        for fragment_start, fragment_end in fragments:
            first = lo + max(0, (fragment_start - run_start + 99) // 100)
            last = min(hi - 1, lo + (fragment_end - run_start) // 100)
            count = last - first + 1
            if count <= 0:
                continue
            start = int(time[first])
            end = int(time[last])
            row = {"source_segment_id": seg, "vehicle_id": int(vehicle[first]),
                   "source_time_start": start, "source_time_end": end,
                   "source_frame_start": int(frame[first]), "source_frame_end": int(frame[last]),
                   "contiguous_frame_count": count,
                   "source_csv_first_row": int(data.iloc[first]["source_csv_row"]),
                   "decision": "candidate" if count >= WINDOW_FRAMES else "rejected",
                   "rejection_reasons": [] if count >= WINDOW_FRAMES else ["below_frozen_1189_frame_source_span"]}
            if count >= WINDOW_FRAMES:
                row["source_time_end"] = start + (WINDOW_FRAMES - 1) * 100
                row["source_frame_end"] = int(frame[first]) + WINDOW_FRAMES - 1
                row["contiguous_frame_count"] = WINDOW_FRAMES
                row["available_fragment_frame_count"] = count
                row["historical_train_dev_overlap_count"] = sum(
                    old["source_segment_id"] == seg and old["time_start"] <= row["source_time_end"]
                    and old["time_end"] >= start for old in exposed
                )
                row["window_id"] = f"cscwd_newdev_{seg}_v{int(vehicle[first])}_t{start}_{row['source_time_end']}"
            candidates.append(row)
    candidates.sort(key=lambda row: (row["source_segment_id"], row["source_time_start"], row["vehicle_id"]))
    quotas = {"lankershim": 1, "us_101": 2}
    selected: list[dict] = []
    for row in candidates:
        if row["decision"] != "candidate":
            continue
        seg = row["source_segment_id"]
        prior = [x for x in selected if x["source_segment_id"] == seg]
        if len(prior) >= quotas[seg]:
            reason = "frozen_segment_quota_reached"
        elif any(row["source_time_start"] < x["source_time_end"] + 2500
                 and row["source_time_end"] > x["source_time_start"] - 2500 for x in prior):
            reason = "selected_window_time_embargo"
        else:
            reason = None
        if reason:
            row["decision"] = "rejected"
            row["rejection_reasons"] = [reason]
        else:
            row["decision"] = "selected"
            selected.append(row)
    for row in selected:
        times = np.unique(time[source == row["source_segment_id"]])
        row["segment_time_rank"] = int(np.searchsorted(times, row["source_time_start"]))
    return candidates, selected, {"run_count": len(starts), "invalid_source_rows": counters["invalid_source_rows"],
                                  "fully_protected_runs": counters["fully_protected_runs"],
                                  "candidate_fragment_count": len(candidates)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if sha(PLAN) != PLAN_SHA or sha(ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json") != PARENT_SHA:
        raise RuntimeError("frozen plan or workload manifest identity mismatch")
    if sha(args.raw_csv_path) != RAW_SHA:
        raise RuntimeError("raw NGSIM identity mismatch")
    forbidden, exposed, strict = strict_metadata()
    anonymous = anonymous_metadata()
    data, prefix, scanned, invalid_identity_rows = load_source(args.raw_csv_path)
    blocks: dict[str, list[tuple[int, int]]] = {}
    for seg in SOURCE_SEGMENTS:
        source_blocks = [(row["time_start"] - EMBARGO_MS, row["time_end"] + EMBARGO_MS)
                         for row in forbidden if row["source_segment_id"] == seg]
        if seg in prefix:
            source_blocks.append((prefix[seg][0] - EMBARGO_MS, prefix[seg][1] + EMBARGO_MS))
        blocks[seg] = merge_blocks(source_blocks)
    candidates, selected, counts = qualify(data, blocks, exposed)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    output.mkdir(parents=True)
    ledger = output / "candidate_fragments.jsonl"
    with ledger.open("w") as stream:
        for row in candidates:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {"schema_version": "cscwd_new_development_source_v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "script_sha256": sha(Path(__file__)), "plan_sha256": PLAN_SHA,
                "raw_source_sha256": RAW_SHA, "parent_workload_manifest_sha256": PARENT_SHA,
                "raw_rows_scanned": scanned, "included_source_rows": len(data),
                "unidentifiable_source_rows_excluded": invalid_identity_rows,
                "source_segments": sorted(SOURCE_SEGMENTS), "excluded_source_segments": ["peachtree", "i_80"],
                "anonymous_prefix_rows_quarantined": 200_000,
                "anonymous_prefix_time_bounds": prefix,
                "minimum_endpoint_gap_ms": 2500, "window_frames": WINDOW_FRAMES,
                "source_span_seconds": (WINDOW_FRAMES - 1) / 10,
                "strict_source_manifests": strict, "anonymous_source_manifests": anonymous,
                "forbidden_interval_count": len(forbidden), "historical_train_dev_interval_count": len(exposed),
                "merged_forbidden_intervals": blocks, "counts": counts,
                "selected_count": len(selected), "selected_windows": selected,
                "candidate_ledger": {"path": ledger.name, "sha256": sha(ledger), "row_count": len(candidates)},
                "decision": "READY_FOR_BOUNDED_REACHABILITY" if selected else "NO_QUALIFIED_NEW_DEVELOPMENT_SOURCE",
                "claim_scope": "source_and_time_eligibility_only_not_algorithm_performance"}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "selected_count": len(selected),
                      "decision": manifest["decision"], "candidate_fragments": len(candidates)}, sort_keys=True))


if __name__ == "__main__":
    main()
