#!/usr/bin/env python3
"""Audit temporal independence within and between benchmark window plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_plan(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    plan = payload.get("selected_window_plan")
    if not isinstance(plan, list):
        raise ValueError(f"selected_window_plan missing from {path}")
    return plan


def interval(window: dict[str, Any]) -> tuple[int, int]:
    start = int(window["frame_offset"])
    return start, start + int(window["window_length"]) - 1


def overlaps(left: dict[str, Any], right: dict[str, Any], gap: int = 0) -> bool:
    left_start, left_end = interval(left)
    right_start, right_end = interval(right)
    return left_start <= right_end + gap and right_start <= left_end + gap


def overlap_rows(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    same_plan: bool,
    gap: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_index, left_window in enumerate(left):
        start = left_index + 1 if same_plan else 0
        for right_window in right[start:]:
            if overlaps(left_window, right_window, gap):
                rows.append(
                    {
                        "left_window_id": left_window.get("window_id"),
                        "left_interval": list(interval(left_window)),
                        "right_window_id": right_window.get("window_id"),
                        "right_interval": list(interval(right_window)),
                    }
                )
    return rows


def build_report(formal_path: Path, holdout_path: Path, gap: int = 0) -> dict[str, Any]:
    formal = load_plan(formal_path)
    holdout = load_plan(holdout_path)
    formal_overlaps = overlap_rows(formal, formal, same_plan=True, gap=gap)
    holdout_overlaps = overlap_rows(holdout, holdout, same_plan=True, gap=gap)
    cross_overlaps = overlap_rows(formal, holdout, same_plan=False, gap=gap)
    return {
        "passed": not formal_overlaps and not holdout_overlaps and not cross_overlaps,
        "minimum_gap_frames": gap,
        "formal_summary_path": str(formal_path.resolve()),
        "holdout_summary_path": str(holdout_path.resolve()),
        "formal_window_ids": [item.get("window_id") for item in formal],
        "holdout_window_ids": [item.get("window_id") for item in holdout],
        "within_formal_overlaps": formal_overlaps,
        "within_holdout_overlaps": holdout_overlaps,
        "formal_holdout_overlaps": cross_overlaps,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal_summary", type=Path, required=True)
    parser.add_argument("--holdout_summary", type=Path, required=True)
    parser.add_argument("--minimum_gap_frames", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.formal_summary, args.holdout_summary, args.minimum_gap_frames)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
