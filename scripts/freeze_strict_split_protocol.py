#!/usr/bin/env python3
"""Freeze outcome-blind, temporally separated train/dev/formal/hidden plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluators.main_results_support import resolve_window_candidates


PROTOCOL_VERSION = "strict_split_v3_segmented_executable_handoff_20260719"
SPLIT_NAMES = ("train", "dev", "formal", "hidden_holdout")
STRATUM_KEYS = (
    "active_non_mechanism_windows",
    "mechanism_activating_windows",
    "idle_or_sparse_windows",
)
STRATUM_LABELS = {
    "active_non_mechanism_windows": "active_non_mechanism",
    "mechanism_activating_windows": "mechanism_activating",
    "idle_or_sparse_windows": "idle_or_sparse",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT_DIR / "configs" / "experiment" / "top_journal_v8_strict_split_20260621",
    )
    parser.add_argument("--mobility_source", choices=["ngsim"], default="ngsim")
    parser.add_argument("--mobility_csv_path", default="")
    parser.add_argument(
        "--workflow_csv_path",
        default=str(ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"),
    )
    parser.add_argument("--max_mobility_rows", type=int, default=10000)
    parser.add_argument(
        "--layout_candidates",
        default="auto_dominant_tight,lust_micro,tight_y,tight_x,auto_grid_tight",
        help="Comma-separated outcome-blind RSU layout candidates used during window scanning.",
    )
    parser.add_argument("--window_length", type=int, default=24)
    parser.add_argument("--window_scan_stride", type=int, default=2)
    parser.add_argument("--minimum_gap_frames", type=int, default=24)
    parser.add_argument("--windows_per_split", type=int, default=20)
    parser.add_argument("--mechanism_windows_per_split", type=int, default=6)
    parser.add_argument("--active_non_mechanism_windows_per_split", type=int, default=2)
    parser.add_argument("--random_seed", type=int, default=7)
    return parser.parse_args()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_layout_candidates(raw_value: str) -> list[str]:
    return [
        item.strip()
        for item in str(raw_value or "").split(",")
        if item.strip()
    ]


def interval(window: dict[str, Any]) -> tuple[int, int]:
    start = int(window["frame_offset"])
    return start, start + int(window["window_length"]) - 1


def available_intervals(window: dict[str, Any]) -> dict[str, tuple[int, int]]:
    intervals = {"frame_offset": interval(window)}
    if window.get("time_index_start") is not None and window.get("time_index_end") is not None:
        intervals["time_index"] = (int(window["time_index_start"]), int(window["time_index_end"]))
    if window.get("segment_frame_start") is not None and window.get("segment_frame_end") is not None:
        intervals["segment_frame"] = (int(window["segment_frame_start"]), int(window["segment_frame_end"]))
    return intervals


def intervals_have_gap(
    left_interval: tuple[int, int],
    right_interval: tuple[int, int],
    minimum_gap_frames: int,
) -> bool:
    left_start, left_end = left_interval
    right_start, right_end = right_interval
    gap = max(0, int(minimum_gap_frames))
    return left_end + gap < right_start or right_end + gap < left_start


def same_known_segment(left: dict[str, Any], right: dict[str, Any]) -> bool | None:
    left_segment = str(left.get("source_segment_id") or "").strip()
    right_segment = str(right.get("source_segment_id") or "").strip()
    if not left_segment or not right_segment:
        return None
    return left_segment == right_segment


def intervals_separated(
    left: dict[str, Any],
    right: dict[str, Any],
    minimum_gap_frames: int,
) -> bool:
    if same_known_segment(left, right) is False:
        return True
    left_intervals = available_intervals(left)
    right_intervals = available_intervals(right)
    for interval_kind in set(left_intervals) & set(right_intervals):
        if not intervals_have_gap(
            left_intervals[interval_kind],
            right_intervals[interval_kind],
            minimum_gap_frames,
        ):
            return False
    return True


def select_stratified_windows(
    pools: dict[str, list[dict[str, Any]]],
    *,
    split_names: tuple[str, ...],
    targets_per_split: dict[str, int],
    minimum_gap_frames: int,
) -> dict[str, list[dict[str, Any]]]:
    plans = {split_name: [] for split_name in split_names}
    occupied: list[dict[str, Any]] = []

    for stratum_key in STRATUM_KEYS:
        per_split = int(targets_per_split[STRATUM_LABELS[stratum_key]])
        total_target = per_split * len(split_names)
        selected: list[dict[str, Any]] = []
        for candidate in pools.get(stratum_key, []):
            if all(
                intervals_separated(candidate, existing, minimum_gap_frames)
                for existing in occupied
            ):
                selected.append(dict(candidate))
                occupied.append(dict(candidate))
                if len(selected) == total_target:
                    break
        if len(selected) != total_target:
            raise RuntimeError(
                f"insufficient separated windows for {stratum_key}: "
                f"required={total_target}, selected={len(selected)}"
            )
        for index, candidate in enumerate(selected):
            plans[split_names[index % len(split_names)]].append(candidate)

    for split_name in split_names:
        plans[split_name].sort(key=lambda item: int(item["frame_offset"]))
    return plans


def audit_split_plans(
    plans: dict[str, list[dict[str, Any]]],
    minimum_gap_frames: int,
) -> dict[str, Any]:
    conflicts: list[dict[str, Any]] = []
    flattened = [
        (split_name, window)
        for split_name, windows in plans.items()
        for window in windows
    ]
    for left_index, (left_split, left_window) in enumerate(flattened):
        for right_split, right_window in flattened[left_index + 1 :]:
            if not intervals_separated(left_window, right_window, minimum_gap_frames):
                conflicts.append(
                    {
                        "left_split": left_split,
                        "left_window_id": left_window.get("window_id"),
                        "left_source_segment_id": left_window.get("source_segment_id", ""),
                        "left_interval": list(interval(left_window)),
                        "right_split": right_split,
                        "right_window_id": right_window.get("window_id"),
                        "right_source_segment_id": right_window.get("source_segment_id", ""),
                        "right_interval": list(interval(right_window)),
                    }
                )
    return {
        "passed": not conflicts,
        "minimum_gap_frames": int(minimum_gap_frames),
        "window_count": len(flattened),
        "conflicts": conflicts,
    }


def relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(resolved)


def main() -> int:
    args = parse_args()
    idle_per_split = (
        int(args.windows_per_split)
        - int(args.mechanism_windows_per_split)
        - int(args.active_non_mechanism_windows_per_split)
    )
    if idle_per_split < 0:
        raise ValueError("per-stratum targets exceed windows_per_split")

    source_path, window_payload = resolve_window_candidates(
        root_dir=ROOT_DIR,
        mobility_source=args.mobility_source,
        mobility_csv_path=args.mobility_csv_path,
        lust_scenario_root="",
        max_mobility_rows=args.max_mobility_rows,
        rsu_layout="auto_dominant_tight",
        frame_offset=0,
        window_length=args.window_length,
        window_selector="max_handoff_candidate",
        window_count=1,
        window_scan_stride=args.window_scan_stride,
        random_seed=args.random_seed,
        window_mode="full_stratified",
        layout_candidates=parse_layout_candidates(args.layout_candidates),
    )
    targets = {
        "mechanism_activating": int(args.mechanism_windows_per_split),
        "active_non_mechanism": int(args.active_non_mechanism_windows_per_split),
        "idle_or_sparse": idle_per_split,
    }
    pools = {key: list(window_payload.get(key, [])) for key in STRATUM_KEYS}
    plans = select_stratified_windows(
        pools,
        split_names=SPLIT_NAMES,
        targets_per_split=targets,
        minimum_gap_frames=args.minimum_gap_frames,
    )
    audit = audit_split_plans(plans, args.minimum_gap_frames)
    if not audit["passed"]:
        raise RuntimeError(f"split independence audit failed: {audit['conflicts']}")

    mobility_path = Path(source_path)
    workflow_path = Path(args.workflow_csv_path)
    if not mobility_path.is_absolute():
        mobility_path = ROOT_DIR / mobility_path
    if not workflow_path.is_absolute():
        workflow_path = ROOT_DIR / workflow_path
    source_records = {
        "mobility": {
            "path": relative_or_absolute(mobility_path),
            "sha256": sha256_file(mobility_path),
            "size_bytes": mobility_path.stat().st_size,
        },
        "workflow": {
            "path": relative_or_absolute(workflow_path),
            "sha256": sha256_file(workflow_path),
            "size_bytes": workflow_path.stat().st_size,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_records: dict[str, Any] = {}
    for split_name, windows in plans.items():
        plan_payload = {
            "protocol_version": PROTOCOL_VERSION,
            "split": split_name,
            "sealed": split_name == "hidden_holdout",
            "outcome_blind_selection": True,
            "selected_window_plan": windows,
        }
        plan_path = args.output_dir / f"{split_name}_window_plan.json"
        rendered = (json.dumps(plan_payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        plan_path.write_bytes(rendered)
        plan_records[split_name] = {
            "path": relative_or_absolute(plan_path),
            "sha256": sha256_bytes(rendered),
            "window_count": len(windows),
            "window_ids": [window["window_id"] for window in windows],
            "stratum_counts": {
                label: sum(1 for window in windows if window["window_class"] == label)
                for label in targets
            },
        }

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_policy": "mobility-covariate-only; no reward, checkpoint, or outcome inspected",
        "hidden_holdout_policy": {
            "sealed": True,
            "opened_at": None,
            "open_condition": "candidate frozen and formal strict-full gate passed",
            "maximum_open_count": 1,
        },
        "parameters": {
            "mobility_source": args.mobility_source,
            "max_mobility_rows": args.max_mobility_rows,
            "layout_candidates": parse_layout_candidates(args.layout_candidates),
            "window_length": args.window_length,
            "window_scan_stride": args.window_scan_stride,
            "minimum_gap_frames": args.minimum_gap_frames,
            "windows_per_split": args.windows_per_split,
            "targets_per_split": targets,
            "random_seed": args.random_seed,
        },
        "source_records": source_records,
        "candidate_pool_counts": {
            STRATUM_LABELS[key]: len(pools[key]) for key in STRATUM_KEYS
        },
        "plans": plan_records,
        "independence_audit": audit,
    }
    manifest["manifest_content_sha256"] = sha256_bytes(canonical_json(manifest).encode("utf-8"))
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
