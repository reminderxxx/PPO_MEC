#!/usr/bin/env python3
"""Classify request-level workflow failures from preserved episode traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_VERSION = "mechanism_failure_cause_analysis_v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT_DIR, text=True
    ).strip()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classify_episode(summary: dict[str, Any]) -> dict[str, Any]:
    endpoint = dict(summary["formal_request_execution_audit"])
    exposure = dict(summary["formal_request_exposure"])
    events = [
        event
        for event in summary.get("cache_event_trace", [])
        if event.get("event_type") == "request"
    ]
    steps = list(summary.get("step_trace", []))
    if len(events) != len(steps):
        raise ValueError("request event and step trace lengths differ")
    failed = [
        (index, event, steps[index])
        for index, event in enumerate(events)
        if not bool(event.get("service_success"))
    ]
    completed = bool(endpoint["workflow_completed_under_exogenous_execution"])
    right_censored = bool(endpoint["right_censored"])
    handoff_unready = [
        item
        for item in failed
        if int(item[2].get("handoff_event_count", 0) or 0) > 0
        and not bool(item[2].get("handoff_ready"))
    ]
    dependency_miss = [
        item
        for item in failed
        if not bool(item[1].get("base_model_hit", True))
        or not bool(item[1].get("adapter_hit", True))
        or not bool(item[1].get("joint_model_hit", True))
    ]
    state_not_ready = [
        item for item in failed if not bool(item[1].get("workflow_state_ready", True))
    ]
    invalid = [
        item
        for item in failed
        if bool(item[2].get("action_invalid"))
        or not bool(item[2].get("action_precondition_valid", True))
    ]
    stalls = [item for item in failed if bool(item[1].get("stall_occurred"))]
    capacity_rejections = [
        item for item in failed if item[1].get("capacity_rejection_reason")
    ]
    if completed:
        primary = "completed"
    elif right_censored:
        primary = "right_censored"
    elif not failed:
        primary = "incomplete_without_request_failure"
    elif handoff_unready:
        primary = "handoff_unprepared"
    elif dependency_miss:
        primary = "cache_dependency_miss"
    elif state_not_ready:
        primary = "workflow_state_unready"
    elif invalid:
        primary = "action_invalid_or_precondition"
    elif stalls:
        primary = "stall_or_timeout"
    else:
        primary = "other_request_failure"
    first_failure = failed[0][0] if failed else None
    external_denominator = int(endpoint["external_request_denominator"])
    exposure_count = len(exposure["requests"])
    return {
        "completed": completed,
        "right_censored": right_censored,
        "primary_failure_category": primary,
        "request_count": len(events),
        "external_request_denominator": external_denominator,
        "exposure_request_count": exposure_count,
        "denominator_consistent": (
            len(events) == external_denominator == exposure_count
            and endpoint.get("request_alignment_status") == "pass"
        ),
        "request_failure_count": len(failed),
        "cache_dependency_failure_count": len(dependency_miss),
        "adapter_miss_failure_count": sum(
            not bool(event.get("adapter_hit", True)) for _, event, _ in failed
        ),
        "base_miss_failure_count": sum(
            not bool(event.get("base_model_hit", True)) for _, event, _ in failed
        ),
        "workflow_state_not_ready_failure_count": len(state_not_ready),
        "handoff_unprepared_failure_count": len(handoff_unready),
        "capacity_rejection_failure_count": len(capacity_rejections),
        "invalid_or_precondition_failure_count": len(invalid),
        "stall_failure_count": len(stalls),
        "first_failure_request_index": first_failure,
        "requests_after_first_failure": (
            len(events) - int(first_failure) - 1 if first_failure is not None else 0
        ),
        "requests_exposed_after_first_failure": bool(
            first_failure is not None and int(first_failure) + 1 < len(events)
        ),
        "failed_action_counts": dict(
            sorted(
                Counter(str(step.get("action_id")) for _, _, step in failed).items()
            )
        ),
        "action_counts": dict(
            sorted(Counter(str(step.get("action_id")) for step in steps).items())
        ),
        "cache_target_alignment_mismatch_count": sum(
            bool(step.get("cache_target_alignment_mismatch")) for step in steps
        ),
        "action_projection_count": int(
            summary.get("agent_action_diagnostics", {}).get(
                "action_projection_count", 0
            )
            or 0
        ),
        "migration_prepare_count": int(
            summary.get("handoff_summary", {}).get("migration_prepare_count", 0)
            or 0
        ),
        "migration_during_handoff_count": int(
            summary.get("handoff_summary", {}).get(
                "migration_during_handoff_count", 0
            )
            or 0
        ),
        "transfer_mb_per_request": float(endpoint["transfer_mb_per_request"]),
        "workflow_continuity_rate": float(endpoint["workflow_continuity_rate"]),
        "end_to_end_workflow_delay": endpoint.get("end_to_end_workflow_delay"),
        "end_to_end_workflow_delay_available": (
            endpoint.get("end_to_end_workflow_delay") is not None
        ),
        "request_exposure_fingerprint": endpoint["request_exposure_fingerprint"],
    }


def _merge_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for row in rows:
        merged.update(dict(row[field]))
    return dict(sorted(merged.items()))


def _aggregate(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    delay_values = [
        float(row["end_to_end_workflow_delay"])
        for row in rows
        if row["end_to_end_workflow_delay"] is not None
    ]
    categories = Counter(str(row["primary_failure_category"]) for row in rows)
    return {
        "source_label": label,
        "episode_count": len(rows),
        "completed_count": sum(bool(row["completed"]) for row in rows),
        "completed_rate": round(
            fmean(float(bool(row["completed"])) for row in rows), 6
        ),
        "right_censored_count": sum(bool(row["right_censored"]) for row in rows),
        "denominator_inconsistency_count": sum(
            not bool(row["denominator_consistent"]) for row in rows
        ),
        "request_failure_count": sum(int(row["request_failure_count"]) for row in rows),
        "cache_dependency_failure_count": sum(
            int(row["cache_dependency_failure_count"]) for row in rows
        ),
        "adapter_miss_failure_count": sum(
            int(row["adapter_miss_failure_count"]) for row in rows
        ),
        "base_miss_failure_count": sum(
            int(row["base_miss_failure_count"]) for row in rows
        ),
        "workflow_state_not_ready_failure_count": sum(
            int(row["workflow_state_not_ready_failure_count"]) for row in rows
        ),
        "handoff_unprepared_failure_count": sum(
            int(row["handoff_unprepared_failure_count"]) for row in rows
        ),
        "capacity_rejection_failure_count": sum(
            int(row["capacity_rejection_failure_count"]) for row in rows
        ),
        "invalid_or_precondition_failure_count": sum(
            int(row["invalid_or_precondition_failure_count"]) for row in rows
        ),
        "stall_failure_count": sum(int(row["stall_failure_count"]) for row in rows),
        "episodes_exposing_requests_after_failure": sum(
            bool(row["requests_exposed_after_first_failure"]) for row in rows
        ),
        "requests_after_first_failure_mean": round(
            fmean(float(row["requests_after_first_failure"]) for row in rows), 6
        ),
        "primary_failure_categories": dict(sorted(categories.items())),
        "all_action_counts": _merge_counts(rows, "action_counts"),
        "failed_action_counts": _merge_counts(rows, "failed_action_counts"),
        "cache_target_alignment_mismatch_count": sum(
            int(row["cache_target_alignment_mismatch_count"]) for row in rows
        ),
        "action_projection_count": sum(
            int(row["action_projection_count"]) for row in rows
        ),
        "delay_available_count": len(delay_values),
        "delay_coverage_rate": round(len(delay_values) / len(rows), 6),
        "conditional_delay_mean": (
            round(fmean(delay_values), 6) if delay_values else None
        ),
        "continuity_mean": round(
            fmean(float(row["workflow_continuity_rate"]) for row in rows), 6
        ),
        "transfer_mb_per_request_mean": round(
            fmean(float(row["transfer_mb_per_request"]) for row in rows), 6
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode-source",
        action="append",
        required=True,
        help="LABEL=directory containing *.summary.json recursively",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse analysis output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    source_inventory: list[dict[str, Any]] = []
    for source in args.episode_source:
        label, separator, raw_path = source.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"invalid episode source: {source}")
        root = Path(raw_path).resolve()
        files = sorted(root.rglob("*.summary.json"))
        if not files:
            raise ValueError(f"no episode summaries under {root}")
        source_inventory.append(
            {
                "label": label,
                "path": str(root),
                "episode_count": len(files),
                "inventory_sha256": hashlib.sha256(
                    "\n".join(
                        f"{path.relative_to(root)}:{_file_sha256(path)}" for path in files
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
        for path in files:
            summary = json.loads(path.read_text(encoding="utf-8"))
            row = classify_episode(summary)
            row.update(
                source_label=label,
                episode_path=str(path),
                workflow_id=summary.get("workflow_info", {}).get("workflow_id"),
                window_id=summary.get("run_info", {}).get("window_id"),
                episode_index=summary.get("run_info", {}).get("episode_index"),
            )
            rows.append(row)
    aggregate = [
        _aggregate(label, [row for row in rows if row["source_label"] == label])
        for label in sorted({str(row["source_label"]) for row in rows})
    ]
    _write_csv(output_dir / "failure_episode_rows.csv", rows)
    _write_json(output_dir / "failure_aggregate.json", {"rows": aggregate})
    receipt = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "completed",
        "git_commit": _git_value("rev-parse", "HEAD"),
        "source_inventory": source_inventory,
        "aggregate": aggregate,
        "metric_semantics": {
            "workflow_failure": (
                "Any failed exposed request makes workflow_completed_under_exogenous_execution false."
            ),
            "post_failure_exposure": (
                "Exogenous request replay continues after a failed request; later requests remain "
                "in the fixed external denominator and are not additional workflow instances."
            ),
            "delay": "Conditional on completed workflows; coverage is reported separately.",
            "training_sufficiency": (
                "Not identifiable from failure events alone; the fixed 64-episode budget is not convergence evidence."
            ),
        },
        "claim_boundary": (
            "First-order observed-data diagnostic only; categories can overlap at event level "
            "and do not establish generalization or causal algorithm superiority."
        ),
    }
    _write_json(output_dir / "completion_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
