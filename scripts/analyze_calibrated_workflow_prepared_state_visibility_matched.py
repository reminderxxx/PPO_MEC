"""Analyze the frozen prepared-state visibility matched development run."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_calibrated_workflow_interface_repair import _integrity, _sha256, _write_json  # noqa: E402
from scripts.analyze_calibrated_workflow_service_reward_alignment import _write_csv  # noqa: E402


METRICS = (
    "workflow_completion_rate",
    "on_time_workflow_completion_rate",
    "service_failure_rate",
    "completed_sample_elapsed_coverage",
    "recompute_seconds",
    "model_prepare_mb",
    "state_transfer_mb",
    "input_transfer_mb",
    "total_transfer_mb",
    "reward",
)
PAIR_KEYS = ("method", "seed", "split", "design_id", "window_id", "workflow_id", "source_segment_id")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, Any], field: str) -> float:
    value = row.get(field, "")
    return float(value) if value not in {"", None} else math.nan


def _validate_source(source_root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    receipt = json.loads((source_root / "completion_receipt.json").read_text())
    manifest = json.loads((source_root / "run_manifest.json").read_text())
    if receipt.get("status") != "complete" or not receipt.get("scientific_execution_complete"):
        raise RuntimeError("source run is not scientifically complete")
    expected = {
        "learned_cells": 20,
        "learned_environment_steps": 115200,
        "learned_optimizer_steps": 15360,
        "new_evaluation_episodes": 1200,
    }
    if any(int(receipt.get(key, -1)) != value for key, value in expected.items()):
        raise RuntimeError("source completion budget mismatch")
    integrity = json.loads((source_root / "artifact_integrity.json").read_text())
    for item in integrity["files"]:
        path = source_root / item["path"]
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"source integrity mismatch: {item['path']}")
    rows = _read_csv(source_root / "evaluation_rows.csv")
    if len(rows) != 1640:
        raise RuntimeError("source evaluation row count mismatch")
    return manifest, rows


def _summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["method"] in {"popularity_cache_heuristic", "two_step_cost_rule"}:
            continue
        grouped[(row["observation_arm"], row["checkpoint_view"], row["method"], row["split"])].append(row)
        grouped[(row["observation_arm"], row["checkpoint_view"], row["method"], "combined")].append(row)
    result = []
    for key, group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "observation_arm": key[0], "checkpoint_view": key[1],
            "method": key[2], "split": key[3], "episode_n": len(group),
            "seed_n": len({row["seed"] for row in group}),
            "source_window_n": len({row["window_id"] for row in group}),
        }
        for metric in METRICS:
            values = [_number(row, metric) for row in group]
            item[metric] = float(np.nanmean(values))
        completed = [
            _number(row, "completed_sample_elapsed_seconds") for row in group
            if _number(row, "workflow_completion_rate") == 1.0
        ]
        item["completed_sample_elapsed_seconds"] = float(np.mean(completed)) if completed else ""
        item["completed_elapsed_n"] = len(completed)
        result.append(item)
    return result


def _paired(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result: list[dict[str, Any]] = []
    seed_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    learned = [row for row in rows if row["method"] not in {"popularity_cache_heuristic", "two_step_cost_rule"}]
    for view in ("selected", "update96"):
        old = {
            tuple(row[key] for key in PAIR_KEYS): row
            for row in learned if row["checkpoint_view"] == view and row["observation_arm"] == "prepared_state_hidden_v3"
        }
        new = {
            tuple(row[key] for key in PAIR_KEYS): row
            for row in learned if row["checkpoint_view"] == view and row["observation_arm"] == "prepared_state_visible_v4"
        }
        if old.keys() != new.keys() or len(old) != 400:
            raise RuntimeError(f"paired identity mismatch: {view}")
        for key in sorted(old):
            before, after = old[key], new[key]
            item: dict[str, Any] = {name: value for name, value in zip(PAIR_KEYS, key)}
            item["checkpoint_view"] = view
            for metric in METRICS:
                item[f"old_{metric}"] = _number(before, metric)
                item[f"new_{metric}"] = _number(after, metric)
                item[f"delta_{metric}"] = item[f"new_{metric}"] - item[f"old_{metric}"]
            common = _number(before, "workflow_completion_rate") == _number(after, "workflow_completion_rate") == 1.0
            item["common_completed"] = common
            item["old_completed_elapsed_seconds"] = _number(before, "completed_sample_elapsed_seconds") if common else ""
            item["new_completed_elapsed_seconds"] = _number(after, "completed_sample_elapsed_seconds") if common else ""
            item["delta_completed_elapsed_seconds"] = (
                _number(after, "completed_sample_elapsed_seconds") - _number(before, "completed_sample_elapsed_seconds")
                if common else ""
            )
            result.append(item)
            seed_grouped[(view, str(item["method"]), str(item["seed"]))].append(item)
    seed_rows = []
    for (view, method, seed), group in sorted(seed_grouped.items()):
        item: dict[str, Any] = {"checkpoint_view": view, "method": method, "seed": seed, "episode_n": len(group)}
        for metric in ("workflow_completion_rate", "on_time_workflow_completion_rate", "service_failure_rate", "recompute_seconds"):
            item[f"mean_delta_{metric}"] = float(np.mean([row[f"delta_{metric}"] for row in group]))
        common = [row for row in group if row["common_completed"]]
        item["common_completed_n"] = len(common)
        item["mean_delta_common_completed_elapsed_seconds"] = (
            float(np.mean([float(row["delta_completed_elapsed_seconds"]) for row in common])) if common else ""
        )
        seed_rows.append(item)
    return result, seed_rows


def _behavior(source_root: Path) -> list[dict[str, Any]]:
    result = []
    for arm, view, name in (
        ("prepared_state_visible_v4", "selected", "new_selected_behavior_ledger.csv"),
        ("prepared_state_visible_v4", "update96", "new_update96_behavior_ledger.csv"),
        ("prepared_state_hidden_v3", "update96", "historical_update96_behavior_ledger.csv"),
    ):
        rows = _read_csv(source_root / name)
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[(row["method"], row["split"])].append(row)
            grouped[(row["method"], "combined")].append(row)
        for (method, split), group in sorted(grouped.items()):
            handoffs = [row for row in group if int(row["executed_action"]) in {0, 1, 3}]
            known_current = [row for row in group if row.get("current_prepared_known") == "True"]
            known_target = [row for row in group if row.get("target_prepared_known") == "True"]
            result.append({
                "observation_arm": arm, "checkpoint_view": view, "method": method, "split": split,
                "decision_n": len(group), "vehicle_fallback_actions": sum(int(row["executed_action"]) == 2 for row in group),
                "handoff_action_n": len(handoffs), "handoff_state_ready_n": sum(row["state_ready"] == "True" for row in handoffs),
                "handoff_state_invalid_n": sum(row["state_ready"] != "True" for row in handoffs),
                "recompute_seconds_sum": float(sum(_number(row, "recompute_seconds") for row in group)),
                "current_prepared_known_n": len(known_current),
                "current_prepared_exists_n": sum(row.get("current_prepared_exists") == "True" for row in known_current),
                "current_prepared_valid_n": sum(row.get("current_prepared_valid") == "True" for row in known_current),
                "target_prepared_known_n": len(known_target),
                "target_prepared_exists_n": sum(row.get("target_prepared_exists") == "True" for row in known_target),
                "target_prepared_valid_n": sum(row.get("target_prepared_valid") == "True" for row in known_target),
            })
    return result


def analyze(source_root: Path, analysis_root: Path) -> None:
    manifest, rows = _validate_source(source_root)
    if analysis_root.exists():
        raise FileExistsError(f"create-only analysis root exists: {analysis_root}")
    analysis_root.mkdir(parents=True)
    paired_rows, seed_rows = _paired(rows)
    _write_csv(analysis_root / "arm_summary.csv", _summary(rows))
    _write_csv(analysis_root / "paired_episode_deltas.csv", paired_rows)
    _write_csv(analysis_root / "paired_seed_deltas.csv", seed_rows)
    _write_csv(analysis_root / "behavior_breakdown.csv", _behavior(source_root))
    selections = json.loads((source_root / "training_summary.json").read_text())
    _write_csv(analysis_root / "checkpoint_selection_summary.csv", [{
        "method": row["method"], "seed": row["seed"], "selected_update": row["selected_update"],
        "parameter_count": row["parameter_count"], "environment_steps": row["environment_steps"],
        "optimizer_steps": row["optimizer_steps"],
    } for row in selections])
    _write_json(analysis_root / "analysis_manifest.json", {
        "schema_version": "calibrated_workflow_prepared_state_visibility_matched_analysis_v1",
        "source_run_id": source_root.name, "source_run_manifest_sha256": _sha256(source_root / "run_manifest.json"),
        "source_artifact_integrity_sha256": _sha256(source_root / "artifact_integrity.json"),
        "scientific_commit": manifest["git_commit"], "primary_view": "selected", "secondary_view": "update96",
        "pair_keys": list(PAIR_KEYS), "evaluation_rows": len(rows), "paired_rows": len(paired_rows),
        "claim_boundary": manifest["claim_boundary"],
    })
    _integrity(analysis_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--analysis-root", required=True)
    args = parser.parse_args()
    analyze(Path(args.source_root).resolve(), Path(args.analysis_root).resolve())


if __name__ == "__main__":
    main()
