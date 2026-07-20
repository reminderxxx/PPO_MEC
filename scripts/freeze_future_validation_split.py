#!/usr/bin/env python3
"""Freeze an outcome-blind future-validation window plan.

The plan is selected from the same candidate pool as the strict v8 split, while
excluding all previously frozen train/dev/formal/hidden intervals. This gives a
new validation split without reopening the consumed hidden holdout.
"""

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


PROTOCOL_VERSION = "future_validation_split_v4_segmented_executable_handoff_20260719"
SPLIT_NAME = "future_validation"
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
DEFAULT_EXISTING_SPLIT_DIR = ROOT_DIR / "configs" / "experiment" / "top_journal_v8_strict_split_20260621"
DEFAULT_EXCLUDE_PLANS = (
    DEFAULT_EXISTING_SPLIT_DIR / "train_window_plan.json",
    DEFAULT_EXISTING_SPLIT_DIR / "dev_window_plan.json",
    DEFAULT_EXISTING_SPLIT_DIR / "formal_window_plan.json",
    DEFAULT_EXISTING_SPLIT_DIR / "hidden_holdout_window_plan.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT_DIR / "configs" / "experiment" / "top_journal_v17_future_validation_time_audited_20260717",
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
    parser.add_argument("--window_count", type=int, default=20)
    parser.add_argument("--mechanism_windows", type=int, default=6)
    parser.add_argument("--active_non_mechanism_windows", type=int, default=2)
    parser.add_argument("--random_seed", type=int, default=7)
    parser.add_argument(
        "--exclude_plan_path",
        action="append",
        default=[str(path) for path in DEFAULT_EXCLUDE_PLANS],
        help="Existing plan to exclude from future validation; repeatable.",
    )
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


def relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(resolved)


def frame_interval(window: dict[str, Any]) -> tuple[int, int]:
    start = int(window["frame_offset"])
    return start, start + int(window["window_length"]) - 1


def available_intervals(window: dict[str, Any]) -> dict[str, tuple[int, int]]:
    intervals = {"frame_offset": frame_interval(window)}
    if window.get("time_index_start") is not None and window.get("time_index_end") is not None:
        intervals["time_index"] = (int(window["time_index_start"]), int(window["time_index_end"]))
    if window.get("segment_frame_start") is not None and window.get("segment_frame_end") is not None:
        intervals["segment_frame"] = (int(window["segment_frame_start"]), int(window["segment_frame_end"]))
    return intervals


def same_known_segment(left: dict[str, Any], right: dict[str, Any]) -> bool | None:
    left_segment = str(left.get("source_segment_id") or "").strip()
    right_segment = str(right.get("source_segment_id") or "").strip()
    if not left_segment or not right_segment:
        return None
    return left_segment == right_segment


def intervals_have_gap(
    left_interval: tuple[int, int],
    right_interval: tuple[int, int],
    minimum_gap_frames: int,
) -> bool:
    left_start, left_end = left_interval
    right_start, right_end = right_interval
    gap = max(0, int(minimum_gap_frames))
    return left_end + gap < right_start or right_end + gap < left_start


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


def load_plan(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    plan = payload.get("selected_window_plan")
    if not isinstance(plan, list):
        raise ValueError(f"selected_window_plan missing from {path}")
    return plan


def load_excluded_windows(paths: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    excluded_windows: list[dict[str, Any]] = []
    plan_records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT_DIR / path
        windows = load_plan(path)
        excluded_windows.extend(dict(window) for window in windows)
        plan_records.append(
            {
                "path": relative_or_absolute(path),
                "sha256": sha256_file(path),
                "window_count": len(windows),
                "window_ids": [window.get("window_id") for window in windows],
            }
        )
    return excluded_windows, plan_records


def select_windows(
    pools: dict[str, list[dict[str, Any]]],
    *,
    targets: dict[str, int],
    excluded_windows: list[dict[str, Any]],
    minimum_gap_frames: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    occupied = [dict(window) for window in excluded_windows]
    for stratum_key in STRATUM_KEYS:
        target = int(targets[STRATUM_LABELS[stratum_key]])
        stratum_selected: list[dict[str, Any]] = []
        for candidate in pools.get(stratum_key, []):
            if all(
                intervals_separated(candidate, existing, minimum_gap_frames)
                for existing in occupied
            ):
                candidate_record = dict(candidate)
                stratum_selected.append(candidate_record)
                selected.append(candidate_record)
                occupied.append(candidate_record)
                if len(stratum_selected) == target:
                    break
        if len(stratum_selected) != target:
            raise RuntimeError(
                f"insufficient separated windows for {stratum_key}: "
                f"required={target}, selected={len(stratum_selected)}"
            )
    selected.sort(key=lambda item: int(item["frame_offset"]))
    return selected


def audit_future_plan(
    selected_windows: list[dict[str, Any]],
    excluded_windows: list[dict[str, Any]],
    minimum_gap_frames: int,
) -> dict[str, Any]:
    conflicts: list[dict[str, Any]] = []
    for left_index, left_window in enumerate(selected_windows):
        for right_window in selected_windows[left_index + 1 :]:
            if not intervals_separated(left_window, right_window, minimum_gap_frames):
                shared_intervals = set(available_intervals(left_window)) & set(available_intervals(right_window))
                conflict_intervals = {
                    interval_kind: {
                        "left_interval": list(available_intervals(left_window)[interval_kind]),
                        "right_interval": list(available_intervals(right_window)[interval_kind]),
                    }
                    for interval_kind in sorted(shared_intervals)
                    if not intervals_have_gap(
                        available_intervals(left_window)[interval_kind],
                        available_intervals(right_window)[interval_kind],
                        minimum_gap_frames,
                    )
                }
                conflicts.append(
                    {
                        "scope": "within_future_validation",
                        "left_window_id": left_window.get("window_id"),
                        "right_window_id": right_window.get("window_id"),
                        "conflict_intervals": conflict_intervals,
                    }
                )
        for excluded_window in excluded_windows:
            if not intervals_separated(left_window, excluded_window, minimum_gap_frames):
                shared_intervals = set(available_intervals(left_window)) & set(available_intervals(excluded_window))
                conflict_intervals = {
                    interval_kind: {
                        "left_interval": list(available_intervals(left_window)[interval_kind]),
                        "right_interval": list(available_intervals(excluded_window)[interval_kind]),
                    }
                    for interval_kind in sorted(shared_intervals)
                    if not intervals_have_gap(
                        available_intervals(left_window)[interval_kind],
                        available_intervals(excluded_window)[interval_kind],
                        minimum_gap_frames,
                    )
                }
                conflicts.append(
                    {
                        "scope": "future_vs_excluded",
                        "left_window_id": left_window.get("window_id"),
                        "right_window_id": excluded_window.get("window_id"),
                        "conflict_intervals": conflict_intervals,
                    }
                )
    return {
        "passed": not conflicts,
        "minimum_gap_frames": int(minimum_gap_frames),
        "checked_interval_kinds": ["frame_offset", "time_index", "segment_frame"],
        "future_window_count": len(selected_windows),
        "excluded_window_count": len(excluded_windows),
        "conflicts": conflicts,
    }


def main() -> int:
    args = parse_args()
    idle_count = int(args.window_count) - int(args.mechanism_windows) - int(args.active_non_mechanism_windows)
    if idle_count < 0:
        raise ValueError("per-stratum targets exceed window_count")
    targets = {
        "mechanism_activating": int(args.mechanism_windows),
        "active_non_mechanism": int(args.active_non_mechanism_windows),
        "idle_or_sparse": idle_count,
    }
    excluded_windows, exclude_plan_records = load_excluded_windows(list(args.exclude_plan_path))

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
    pools = {key: list(window_payload.get(key, [])) for key in STRATUM_KEYS}
    selected_windows = select_windows(
        pools,
        targets=targets,
        excluded_windows=excluded_windows,
        minimum_gap_frames=args.minimum_gap_frames,
    )
    audit = audit_future_plan(selected_windows, excluded_windows, args.minimum_gap_frames)
    if not audit["passed"]:
        raise RuntimeError(f"future-validation independence audit failed: {audit['conflicts']}")

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
    plan_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "split": SPLIT_NAME,
        "sealed": True,
        "outcome_blind_selection": True,
        "excluded_consumed_hidden_outcomes": True,
        "selected_window_plan": selected_windows,
    }
    plan_path = args.output_dir / "future_validation_window_plan.json"
    rendered_plan = (json.dumps(plan_payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    plan_path.write_bytes(rendered_plan)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_policy": (
            "mobility-covariate-only; excludes prior train/dev/formal/hidden frame and time intervals; "
            "does not inspect reward, checkpoint, or outcome rows"
        ),
        "parameters": {
            "mobility_source": args.mobility_source,
            "max_mobility_rows": args.max_mobility_rows,
            "layout_candidates": parse_layout_candidates(args.layout_candidates),
            "window_length": args.window_length,
            "window_scan_stride": args.window_scan_stride,
            "minimum_gap_frames": args.minimum_gap_frames,
            "window_count": args.window_count,
            "targets": targets,
            "random_seed": args.random_seed,
        },
        "source_records": source_records,
        "exclude_plan_records": exclude_plan_records,
        "candidate_pool_counts": {
            STRATUM_LABELS[key]: len(pools[key]) for key in STRATUM_KEYS
        },
        "future_validation_plan": {
            "path": relative_or_absolute(plan_path),
            "sha256": sha256_bytes(rendered_plan),
            "window_count": len(selected_windows),
            "window_ids": [window["window_id"] for window in selected_windows],
            "stratum_counts": {
                label: sum(1 for window in selected_windows if window["window_class"] == label)
                for label in targets
            },
        },
        "independence_audit": audit,
    }
    manifest["manifest_content_sha256"] = sha256_bytes(canonical_json(manifest).encode("utf-8"))
    manifest_path = args.output_dir / "future_validation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
