"""Rebuild bounded statistics and mechanism diagnostics from a completed repair run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY = (
    "workflow_completion_rate",
    "node_coverage_rate",
    "deadline_violation_rate",
    "service_failure_rate",
    "modeled_completion_seconds",
    "total_transfer_mb",
    "model_prepare_mb",
    "state_transfer_mb",
    "recompute_seconds",
    "reward",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_integrity(run_root: Path) -> dict[str, Any]:
    manifest = _load_json(run_root / "artifact_integrity.json")
    failures = []
    for item in manifest["files"]:
        path = run_root / item["path"]
        if not path.is_file():
            failures.append({"path": item["path"], "reason": "missing"})
            continue
        observed = _sha256(path)
        if observed != item["sha256"] or path.stat().st_size != int(item["bytes"]):
            failures.append(
                {
                    "path": item["path"],
                    "reason": "hash_or_size_mismatch",
                    "expected_sha256": item["sha256"],
                    "observed_sha256": observed,
                }
            )
    return {
        "verified_file_count": len(manifest["files"]),
        "failure_count": len(failures),
        "failures": failures,
    }


def _bootstrap(values: list[float], rng: np.random.Generator) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    draws = data[rng.integers(0, len(data), size=(5000, len(data)))].mean(axis=1)
    return {
        "mean": float(data.mean()),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "outer_n": len(values),
    }


def _statistical_rebuild(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(20261006)
    result: dict[str, Any] = {}
    for split in ("regression", "frozen_check"):
        split_rows = [row for row in rows if row["split"] == split]
        methods = sorted({str(row["method"]) for row in split_rows})
        by_method_window: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in split_rows:
            by_method_window[str(row["method"])][str(row["window_id"])].append(row)
        method_stats: dict[str, Any] = {}
        for method in methods:
            method_stats[method] = {}
            for metric in PRIMARY:
                values = [
                    float(np.mean([float(row[metric]) for row in window_rows]))
                    for window_rows in by_method_window[method].values()
                ]
                method_stats[method][metric] = _bootstrap(values, rng)
        rule = by_method_window["two_step_cost_rule"]
        paired: dict[str, Any] = {}
        for method in sorted(set(methods) - {"two_step_cost_rule"}):
            paired[method] = {}
            common_windows = sorted(set(by_method_window[method]) & set(rule))
            for metric in PRIMARY:
                deltas = []
                for window in common_windows:
                    learned = float(
                        np.mean(
                            [float(row[metric]) for row in by_method_window[method][window]]
                        )
                    )
                    baseline = float(
                        np.mean([float(row[metric]) for row in rule[window]])
                    )
                    deltas.append(learned - baseline)
                paired[method][metric] = _bootstrap(deltas, rng)
        result[split] = {
            "outer_unit": "source_window",
            "seed_handling": "averaged_within_window_before_bootstrap",
            "bootstrap_draws": 5000,
            "methods": method_stats,
            "paired_delta_vs_two_step_rule": paired,
        }
    return result


def _action_audit(
    evaluation_rows: list[dict[str, Any]],
    ledger: list[dict[str, str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("regression", "frozen_check"):
        split_result: dict[str, Any] = {}
        for method in sorted({row["method"] for row in ledger if row["split"] == split}):
            action_rows = [
                row for row in ledger if row["split"] == split and row["method"] == method
            ]
            action_counts = Counter(int(row["executed_action"]) for row in action_rows)
            missing_rows = [
                row for row in action_rows if row["current_bundle_ready"] == "False"
            ]
            missing_counts = Counter(int(row["executed_action"]) for row in missing_rows)
            action4_rows = [row for row in action_rows if int(row["executed_action"]) == 4]
            policy_rows = [row for row in action_rows if row["seed"] != "rule"]
            logprob_deltas = [
                abs(float(row["policy_log_prob"]) - float(row["executed_action_log_prob"]))
                for row in policy_rows
                if row["policy_log_prob"] != "" and row["executed_action_log_prob"] != ""
            ]
            split_result[method] = {
                "decision_count": len(action_rows),
                "action_counts": {str(action): action_counts[action] for action in range(5)},
                "current_missing_decision_count": len(missing_rows),
                "current_missing_action_counts": {
                    str(action): missing_counts[action] for action in range(5)
                },
                "current_missing_action_distribution": {
                    str(action): missing_counts[action] / max(len(missing_rows), 1)
                    for action in range(5)
                },
                "action4_attempts": len(action4_rows),
                "action4_successes": sum(
                    row["service_completed"] == "True"
                    and row["migration_success"] == "True"
                    for row in action4_rows
                ),
                "action4_failures": sum(
                    not (
                        row["service_completed"] == "True"
                        and row["migration_success"] == "True"
                    )
                    for row in action4_rows
                ),
                "max_no_progress_streak": max(
                    (int(row["no_progress_streak"]) for row in action_rows), default=0
                ),
                "raw_to_executed_mismatch_count": sum(
                    int(row["raw_env_action"]) != int(row["executed_action"])
                    for row in policy_rows
                ),
                "projection_count": sum(
                    row["projection_applied"] == "True" for row in policy_rows
                ),
                "max_policy_vs_executed_logprob_abs_delta": (
                    max(logprob_deltas) if logprob_deltas else None
                ),
            }
        result[split] = split_result

    for split in result:
        episode_rows = [row for row in evaluation_rows if row["split"] == split]
        result[split]["episode_strata"] = {}
        for pressure in ("low", "high"):
            result[split]["episode_strata"][pressure] = {}
            for method in sorted({str(row["method"]) for row in episode_rows}):
                group = [
                    row
                    for row in episode_rows
                    if row["method"] == method and row["handoff_pressure"] == pressure
                ]
                if not group:
                    continue
                result[split]["episode_strata"][pressure][method] = {
                    "row_n": len(group),
                    "source_window_n": len({str(row["window_id"]) for row in group}),
                    "workflow_completion_rate": float(
                        np.mean([float(row["workflow_completion_rate"]) for row in group])
                    ),
                    "node_coverage_rate": float(
                        np.mean([float(row["node_coverage_rate"]) for row in group])
                    ),
                }
    return result


def _historical_comparison(run_root: Path, repaired: dict[str, Any]) -> dict[str, Any]:
    old_root = run_root.parents[1] / "calibrated_continuous_workflow_pilot_v2_20261006"
    old_aggregate_path = old_root / "aggregate.json"
    if not old_aggregate_path.is_file():
        return {"available": False, "reason": "historical_v2_aggregate_not_present"}
    old = _load_json(old_aggregate_path)["methods"]
    comparison: dict[str, Any] = {"available": True, "interpretation": "descriptive_only_interface_changed"}
    for method in ("sa_ghmappo", "ppo", "mappo", "two_step_cost_rule"):
        old_completion = float(old[method]["metrics"]["workflow_completion_rate"]["mean"])
        new_completion = float(
            repaired["regression"]["methods"][method]["workflow_completion_rate"]["mean"]
        )
        comparison[method] = {
            "historical_v2_completion": old_completion,
            "repaired_regression_completion": new_completion,
            "descriptive_delta": new_completion - old_completion,
        }
    return comparison


def _rebuild_integrity(run_root: Path) -> None:
    files = sorted(
        path
        for path in run_root.rglob("*")
        if path.is_file() and path.name != "artifact_integrity.json"
    )
    _write_json(
        run_root / "artifact_integrity.json",
        {
            "schema_version": "artifact_integrity_v1",
            "files": [
                {
                    "path": str(path.relative_to(run_root)),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    if _load_json(run_root / "run_status.json").get("status") != "complete":
        raise RuntimeError("run is not complete")
    integrity_before = _verify_integrity(run_root)
    if integrity_before["failure_count"]:
        raise RuntimeError(f"input integrity failed: {integrity_before}")
    evaluation_rows = _load_json(run_root / "evaluation_rows.json")
    ledger = _read_csv(run_root / "action_ledger.csv")
    rebuilt = _statistical_rebuild(evaluation_rows)
    action_audit = _action_audit(evaluation_rows, ledger)
    historical = _historical_comparison(run_root, rebuilt)
    _write_json(run_root / "statistical_rebuild.json", rebuilt)
    _write_json(run_root / "action_interface_audit.json", action_audit)
    _write_json(run_root / "historical_regression_comparison.json", historical)
    _write_json(
        run_root / "postprocess_receipt.json",
        {
            "status": "complete",
            "input_integrity": integrity_before,
            "statistical_rebuild": "statistical_rebuild.json",
            "action_interface_audit": "action_interface_audit.json",
            "historical_comparison": "historical_regression_comparison.json",
        },
    )
    _rebuild_integrity(run_root)
    integrity_after = _verify_integrity(run_root)
    if integrity_after["failure_count"]:
        raise RuntimeError(f"output integrity failed: {integrity_after}")
    print(
        json.dumps(
            {
                "status": "complete",
                "integrity_before": integrity_before,
                "integrity_after": integrity_after,
            }
        )
    )


if __name__ == "__main__":
    main()
