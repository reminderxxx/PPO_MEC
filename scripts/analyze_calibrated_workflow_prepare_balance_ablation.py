"""Rebuild the bounded prepare-balance ablation comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ORIGINAL = "sa_ghmappo_original"
CANDIDATE = "sa_ghmappo_no_auxiliary"
PAIR_METRICS = (
    "workflow_completion_rate",
    "on_time_completion_rate",
    "node_coverage_rate",
    "deadline_violation_rate",
    "service_failure_rate",
    "handoff_failure_rate",
    "total_transfer_mb",
    "recompute_seconds",
    "action4_attempts",
    "action4_successes",
    "safe_prepare_attempts",
    "invalid_prepare_attempts",
    "infeasible_target_prepare_attempts",
    "max_consecutive_no_progress_steps",
    "reward",
)
STRATA = (
    "handoff_pressure",
    "link_error_class",
    "capacity",
    "sharing",
    "topology_class",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _window_means(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["window_id"])].append(float(row[metric]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _paired_bootstrap(
    original_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    metric: str,
    *,
    samples: int = 5000,
    seed: int = 7,
) -> dict[str, Any]:
    original = _window_means(original_rows, metric)
    candidate = _window_means(candidate_rows, metric)
    keys = sorted(set(original) & set(candidate))
    deltas = np.asarray([candidate[key] - original[key] for key in keys], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.integers(0, len(deltas), size=len(deltas))
        draws[index] = float(np.mean(deltas[selected]))
    return {
        "direction": "candidate_minus_original",
        "outer_unit": "source_window",
        "outer_n": len(keys),
        "mean_delta": float(np.mean(deltas)),
        "percentile_95_ci": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
        "window_deltas": {key: float(candidate[key] - original[key]) for key in keys},
    }


def _behavior_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actions = [sum(int(row[f"action_{action}"]) for row in rows) for action in range(5)]
    missing = sum(int(row["current_model_missing_decisions"]) for row in rows)
    missing_action4 = sum(int(row["current_missing_action_4"]) for row in rows)
    action4 = sum(int(row["action4_attempts"]) for row in rows)
    successes = sum(int(row["action4_successes"]) for row in rows)
    safe = sum(int(row["safe_prepare_attempts"]) for row in rows)
    invalid = sum(int(row["invalid_prepare_attempts"]) for row in rows)
    infeasible = sum(int(row["infeasible_target_prepare_attempts"]) for row in rows)
    return {
        "action_counts": {str(index): value for index, value in enumerate(actions)},
        "action0_rate": float(actions[0]) / max(sum(actions), 1),
        "action4_rate": float(actions[4]) / max(sum(actions), 1),
        "current_missing_action4": missing_action4,
        "current_missing_decisions": missing,
        "current_missing_action4_rate": float(missing_action4) / max(missing, 1),
        "action4_attempts": action4,
        "realized_prepare_successes": successes,
        "realized_prepare_rate": float(successes) / max(action4, 1),
        "service_safe_prepare_attempts": safe,
        "invalid_prepare_attempts": invalid,
        "infeasible_target_prepare_attempts": infeasible,
    }


def _seed_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for seed in (7, 17, 29):
        group = [row for row in rows if int(row["seed"]) == seed]
        result.append(
            {
                "seed": seed,
                "episode_n": len(group),
                "workflow_completion_rate": float(
                    np.mean([float(row["workflow_completion_rate"]) for row in group])
                ),
                "service_failure_rate": float(
                    np.mean([float(row["service_failure_rate"]) for row in group])
                ),
                "action4_attempts": sum(int(row["action4_attempts"]) for row in group),
                "invalid_prepare_attempts": sum(
                    int(row["invalid_prepare_attempts"]) for row in group
                ),
            }
        )
    return result


def _stratified(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for split in ("regression", "frozen_check"):
        split_rows = [row for row in rows if row["split"] == split]
        for field in STRATA:
            values = sorted({str(row[field]) for row in split_rows})
            for value in values:
                for method in (ORIGINAL, CANDIDATE):
                    group = [
                        row
                        for row in split_rows
                        if row["method"] == method and str(row[field]) == value
                    ]
                    if not group:
                        continue
                    result.append(
                        {
                            "split": split,
                            "stratum": field,
                            "value": value,
                            "method": method,
                            "row_n": len(group),
                            "source_window_n": len({str(row["window_id"]) for row in group}),
                            "workflow_completion_rate": float(
                                np.mean(
                                    [float(row["workflow_completion_rate"]) for row in group]
                                )
                            ),
                            "deadline_violation_rate": float(
                                np.mean(
                                    [float(row["deadline_violation_rate"]) for row in group]
                                )
                            ),
                            "action4_attempts": sum(
                                int(row["action4_attempts"]) for row in group
                            ),
                            "invalid_prepare_attempts": sum(
                                int(row["invalid_prepare_attempts"]) for row in group
                            ),
                        }
                    )
    return result


def _integrity(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_integrity.json"
    )
    _write_json(
        root / "artifact_integrity.json",
        {
            "schema_version": "artifact_integrity_v1",
            "files": [
                {
                    "path": str(path.relative_to(root)),
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
    root = Path(args.run_root).resolve()
    receipt = _load_json(root / "completion_receipt.json")
    if receipt.get("status") != "complete":
        raise RuntimeError("run is not complete")
    rows = _load_json(root / "evaluation_rows.json")
    for row in rows:
        row["on_time_completion_rate"] = float(
            bool(row["workflow_completion_rate"])
            and float(row["deadline_violation_rate"]) <= 0.0
        )
    curves = _load_json(root / "training_curves.json")
    report: dict[str, Any] = {
        "schema_version": "calibrated_workflow_prepare_balance_comparison_v1",
        "run_root": str(root),
        "decision_category": "存在其他未定位问题",
        "candidate_disposition": "reject_auxiliary_removal_as_prepare_balance_fix",
        "claim_boundary": (
            "single-factor nonformal development evidence; not general causality, "
            "innovation, holdout, or paper-ready evidence"
        ),
        "splits": {},
    }
    summary_rows: list[dict[str, Any]] = []
    for split in ("regression", "frozen_check"):
        split_report: dict[str, Any] = {}
        original_rows = [
            row for row in rows if row["split"] == split and row["method"] == ORIGINAL
        ]
        candidate_rows = [
            row for row in rows if row["split"] == split and row["method"] == CANDIDATE
        ]
        split_report["paired_candidate_minus_original"] = {
            metric: _paired_bootstrap(original_rows, candidate_rows, metric)
            for metric in PAIR_METRICS
        }
        split_report["behavior_counts"] = {
            ORIGINAL: _behavior_counts(original_rows),
            CANDIDATE: _behavior_counts(candidate_rows),
        }
        split_report["seed_summary"] = {
            ORIGINAL: _seed_summary(original_rows),
            CANDIDATE: _seed_summary(candidate_rows),
        }
        both_completed = [
            (original, candidate)
            for original in original_rows
            for candidate in candidate_rows
            if int(original["seed"]) == int(candidate["seed"])
            and original["design_id"] == candidate["design_id"]
            and float(original["workflow_completion_rate"]) == 1.0
            and float(candidate["workflow_completion_rate"]) == 1.0
        ]
        split_report["conditional_completed_elapsed"] = {
            "paired_episode_n": len(both_completed),
            "candidate_minus_original_mean_seconds": float(
                np.mean(
                    [
                        float(candidate["completed_sample_elapsed_seconds"])
                        - float(original["completed_sample_elapsed_seconds"])
                        for original, candidate in both_completed
                    ]
                )
            )
            if both_completed
            else None,
            "excludes_any_pair_with_unfinished_episode": True,
        }
        report["splits"][split] = split_report
        for method, method_rows in ((ORIGINAL, original_rows), (CANDIDATE, candidate_rows)):
            behavior = split_report["behavior_counts"][method]
            summary_rows.append(
                {
                    "split": split,
                    "method": method,
                    "workflow_completion_rate": float(
                        np.mean(
                            [float(row["workflow_completion_rate"]) for row in method_rows]
                        )
                    ),
                    "deadline_violation_rate": float(
                        np.mean(
                            [float(row["deadline_violation_rate"]) for row in method_rows]
                        )
                    ),
                    "on_time_completion_rate": float(
                        np.mean(
                            [float(row["on_time_completion_rate"]) for row in method_rows]
                        )
                    ),
                    "service_failure_rate": float(
                        np.mean(
                            [float(row["service_failure_rate"]) for row in method_rows]
                        )
                    ),
                    "total_transfer_mb": float(
                        np.mean([float(row["total_transfer_mb"]) for row in method_rows])
                    ),
                    "recompute_seconds": float(
                        np.mean([float(row["recompute_seconds"]) for row in method_rows])
                    ),
                    "action4_rate": behavior["action4_rate"],
                    "current_missing_action4_rate": behavior[
                        "current_missing_action4_rate"
                    ],
                    "realized_prepare_rate": behavior["realized_prepare_rate"],
                    "invalid_prepare_attempts": behavior["invalid_prepare_attempts"],
                    "max_consecutive_no_progress_steps": float(
                        np.mean(
                            [
                                float(row["max_consecutive_no_progress_steps"])
                                for row in method_rows
                            ]
                        )
                    ),
                }
            )

    report["training_curve_summary"] = {}
    for method in (ORIGINAL, CANDIDATE):
        group = [row for row in curves if row["method"] == method]
        report["training_curve_summary"][method] = {
            "episode_rows": len(group),
            "workflow_completion_rate": float(
                np.mean([float(bool(row["workflow_completed"])) for row in group])
            ),
            "node_coverage_rate": float(
                np.mean([float(row["node_coverage_rate"]) for row in group])
            ),
            "reward": float(np.mean([float(row["reward"]) for row in group])),
        }
    report["interpretation"] = {
        "completion": (
            "Point completion is unchanged on both splits; the failing seed moves "
            "from seed 17 to seed 29 instead of disappearing."
        ),
        "prepare_balance": (
            "The candidate increases total action 4, current-missing action 4, "
            "and invalid prepare attempts on the frozen check."
        ),
        "not_always_action0": (
            "The candidate reduces action 0 rate and increases action 4; it is not "
            "a conservative always-execute collapse."
        ),
        "benefit_source": (
            "Lower transfer/recompute and conditional completed elapsed coexist with "
            "unchanged completion and more service failures/invalid prepares; unfinished "
            "episodes are excluded from elapsed and no speed advantage is claimed."
        ),
        "causal_boundary": (
            "Removing the auxiliary loss does not support it as the sole cause. The "
            "result rejects this deletion as a fix but does not prove the auxiliary loss "
            "is generally beneficial."
        ),
    }

    stratified_rows = _stratified(rows)
    _write_json(root / "comparison_report.json", report)
    _write_csv(root / "comparison_summary.csv", summary_rows)
    _write_csv(root / "stratified_ablation.csv", stratified_rows)
    _write_json(
        root / "analysis_receipt.json",
        {
            "status": "complete",
            "bootstrap_samples": 5000,
            "bootstrap_seed": 7,
            "outer_unit": "source_window",
            "decision_category": report["decision_category"],
        },
    )
    _integrity(root)
    print(json.dumps({"run_root": str(root), "status": "complete"}))


if __name__ == "__main__":
    main()
