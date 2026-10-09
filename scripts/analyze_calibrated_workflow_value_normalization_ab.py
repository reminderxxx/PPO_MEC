"""Build a compact, read-only review package for the completed PopArt A/B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parents[1]
CONTROL = "control_raw_critic_target"
CANDIDATE = "candidate_popart_critic_target"
METHODS = ("sa_ghmappo", "mappo", "ppo")
SEEDS = (7, 17, 29, 43, 61)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(mean(float(row[key]) for row in rows))


def _service_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    completed = [row for row in rows if float(row["workflow_completion_rate"]) > 0.5]
    decision_count = sum(
        sum(int(row[f"action_{action_id}"]) for action_id in range(5)) for row in rows
    )

    def completed_mean(key: str) -> float | None:
        if not completed:
            return None
        return _mean(completed, key)

    return {
        "episode_count": float(len(rows)),
        "completion": _mean(rows, "workflow_completion_rate"),
        "on_time": _mean(rows, "on_time_workflow_completion_rate"),
        "unfinished_after_deadline": _mean(rows, "unfinished_after_deadline_rate"),
        "failure_episode_rate": _mean(rows, "service_failure_rate"),
        "failure_attempt_rate": float(
            sum(int(row["service_failures"]) for row in rows) / max(decision_count, 1)
        ),
        "no_progress_episode_rate": float(
            mean(int(row["max_consecutive_no_progress_steps"]) >= 2 for row in rows)
        ),
        "completed_elapsed_coverage": float(len(completed) / max(len(rows), 1)),
        "completed_elapsed_seconds": completed_mean("completed_sample_elapsed_seconds"),
        "transfer_mb_all": _mean(rows, "total_transfer_mb"),
        "transfer_mb_completed": completed_mean("total_transfer_mb"),
        "recompute_seconds_all": _mean(rows, "recompute_seconds"),
        "recompute_seconds_completed": completed_mean("recompute_seconds"),
        "reward": _mean(rows, "reward"),
    }


def _paired_row(
    identity: dict[str, Any], control: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    row = dict(identity)
    for key in control:
        row[f"control_{key}"] = control[key]
        row[f"candidate_{key}"] = candidate[key]
        if key != "episode_count" and control[key] is not None and candidate[key] is not None:
            row[f"delta_{key}"] = float(candidate[key]) - float(control[key])
        elif key != "episode_count":
            row[f"delta_{key}"] = None
    return row


def _direction(values: list[float], *, higher_is_better: bool) -> dict[str, int]:
    tolerance = 1e-12
    improved = sum(value > tolerance for value in values)
    worsened = sum(value < -tolerance for value in values)
    if not higher_is_better:
        improved, worsened = worsened, improved
    return {
        "improved_seed_count": improved,
        "neutral_seed_count": len(values) - improved - worsened,
        "worsened_seed_count": worsened,
    }


def _service_tables(
    evaluation_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in evaluation_rows:
        grouped[(row["arm"], row["method"], int(row["seed"]))].append(row)
    paired: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in SEEDS:
            control = _service_metrics(grouped[(CONTROL, method, seed)])
            candidate = _service_metrics(grouped[(CANDIDATE, method, seed)])
            paired.append(_paired_row({"method": method, "seed": seed}, control, candidate))

    method_rows: list[dict[str, Any]] = []
    higher_is_better = {
        "completion": True,
        "on_time": True,
        "failure_episode_rate": False,
        "failure_attempt_rate": False,
        "no_progress_episode_rate": False,
        "completed_elapsed_seconds": False,
        "transfer_mb_completed": False,
        "recompute_seconds_completed": False,
        "reward": True,
    }
    for method in METHODS:
        method_pairs = [row for row in paired if row["method"] == method]
        pooled_control = _service_metrics(
            [
                row
                for (arm, row_method, _), rows in grouped.items()
                if arm == CONTROL and row_method == method
                for row in rows
            ]
        )
        pooled_candidate = _service_metrics(
            [
                row
                for (arm, row_method, _), rows in grouped.items()
                if arm == CANDIDATE and row_method == method
                for row in rows
            ]
        )
        summary = _paired_row({"method": method}, pooled_control, pooled_candidate)
        for metric, direction in higher_is_better.items():
            deltas = [
                float(row[f"delta_{metric}"])
                for row in method_pairs
                if row[f"delta_{metric}"] is not None
            ]
            counts = _direction(deltas, higher_is_better=direction)
            for key, value in counts.items():
                summary[f"{metric}_{key}"] = value
        method_rows.append(summary)
    return paired, method_rows


def _behavior_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    eligible = [row for row in rows if row["current_bundle_ready"].lower() == "false"]
    probabilities = [json.loads(row["env_action_probs"]) for row in eligible]
    covered = [values for values in probabilities if len(values) > 4]
    return {
        "current_missing_decisions": float(len(eligible)),
        "action4_probability_coverage": float(len(covered) / max(len(eligible), 1)),
        "action4_mean_probability": float(
            mean(float(values[4]) for values in covered) if covered else 0.0
        ),
        "action4_raw_argmax_rate": float(
            mean(int(row["raw_env_action"]) == 4 for row in eligible)
            if eligible
            else 0.0
        ),
        "action4_executed_rate": float(
            mean(int(row["executed_action"]) == 4 for row in eligible)
            if eligible
            else 0.0
        ),
    }


def _behavior_tables(
    behavior_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in behavior_rows:
        grouped[(row["arm"], row["method"], int(row["seed"]))].append(row)
    paired: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in SEEDS:
            paired.append(
                _paired_row(
                    {"method": method, "seed": seed},
                    _behavior_metrics(grouped[(CONTROL, method, seed)]),
                    _behavior_metrics(grouped[(CANDIDATE, method, seed)]),
                )
            )
    method_rows: list[dict[str, Any]] = []
    for method in METHODS:
        control_rows = [
            row
            for (arm, row_method, _), rows in grouped.items()
            if arm == CONTROL and row_method == method
            for row in rows
        ]
        candidate_rows = [
            row
            for (arm, row_method, _), rows in grouped.items()
            if arm == CANDIDATE and row_method == method
            for row in rows
        ]
        summary = _paired_row(
            {"method": method},
            _behavior_metrics(control_rows),
            _behavior_metrics(candidate_rows),
        )
        method_pairs = [row for row in paired if row["method"] == method]
        for metric in (
            "action4_mean_probability",
            "action4_raw_argmax_rate",
            "action4_executed_rate",
        ):
            counts = _direction(
                [float(row[f"delta_{metric}"]) for row in method_pairs],
                higher_is_better=False,
            )
            for key, value in counts.items():
                summary[f"{metric}_{key}"] = value
        method_rows.append(summary)
    return paired, method_rows


def _optimizer_aggregate(
    rows: list[dict[str, str]], predicate: Callable[[dict[str, str]], bool]
) -> dict[str, float]:
    selected = [row for row in rows if predicate(row)]
    return {
        "global_clip_scale": _mean(selected, "global_clip_scale"),
        "global_clip_fraction": float(
            mean(float(row["global_clip_scale"]) < 1.0 - 1e-12 for row in selected)
        ),
        "pre_clip_total_grad_norm": _mean(selected, "pre_clip_total_grad_norm"),
        "actual_kl": _mean(selected, "approx_kl"),
        "policy_clip_fraction": _mean(selected, "clip_fraction"),
        "entropy": _mean(selected, "entropy"),
    }


def _mechanism_tables(
    update_rows: list[dict[str, Any]], optimizer_rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    updates = {
        (row["arm"], row["method"], int(row["seed"]), int(row["update_index"])): row
        for row in update_rows
    }
    paired: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in SEEDS:
            values: dict[str, dict[str, float]] = {}
            for arm in (CONTROL, CANDIDATE):
                final = updates[(arm, method, seed, 24)]
                optimizer_final = _optimizer_aggregate(
                    optimizer_rows,
                    lambda row, arm=arm, method=method, seed=seed: (
                        row["arm"] == arm
                        and row["method"] == method
                        and int(row["seed"]) == seed
                        and int(row["update_index"]) == 24
                    ),
                )
                all_cell_updates = [
                    updates[(arm, method, seed, update_index)]
                    for update_index in range(1, 25)
                ]
                values[arm] = {
                    "final_rmse_raw": float(final["critic_rmse_denormalized_before_update"]),
                    "final_explained_variance": float(final["explained_variance"]),
                    "final_value_to_policy_grad_ratio": float(
                        final["weighted_value_to_policy_grad_ratio"]
                    ),
                    "mean24_rmse_raw": float(
                        mean(
                            float(row["critic_rmse_denormalized_before_update"])
                            for row in all_cell_updates
                        )
                    ),
                    "mean24_explained_variance": float(
                        mean(float(row["explained_variance"]) for row in all_cell_updates)
                    ),
                    "mean24_value_to_policy_grad_ratio": float(
                        mean(
                            float(row["weighted_value_to_policy_grad_ratio"])
                            for row in all_cell_updates
                        )
                    ),
                    **{f"final_{key}": value for key, value in optimizer_final.items()},
                }
            paired.append(
                _paired_row(
                    {"method": method, "seed": seed},
                    values[CONTROL],
                    values[CANDIDATE],
                )
            )
    method_rows: list[dict[str, Any]] = []
    for method in METHODS:
        group = [row for row in paired if row["method"] == method]
        metrics = [
            key[len("delta_") :]
            for key in group[0]
            if key.startswith("delta_")
        ]
        method_rows.append(
            {
                "method": method,
                **{
                    f"median_delta_{metric}": float(
                        median(float(row[f"delta_{metric}"]) for row in group)
                    )
                    for metric in metrics
                },
                **{
                    f"{metric}_candidate_lower_seed_count": sum(
                        float(row[f"delta_{metric}"]) < -1e-12 for row in group
                    )
                    for metric in metrics
                },
                **{
                    f"{metric}_candidate_higher_seed_count": sum(
                        float(row[f"delta_{metric}"]) > 1e-12 for row in group
                    )
                    for metric in metrics
                },
            }
        )
    return paired, method_rows


def _selection_rows(training_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "arm": row["arm"],
            "method": row["method"],
            "seed": row["seed"],
            "selected_update": row["selected_update"],
            "environment_steps": row["environment_steps"],
            "updates": row["updates"],
            "optimizer_steps": row["optimizer_steps"],
            "episodes_started": row["episodes_started"],
            "episodes_completed_or_truncated": row[
                "episodes_completed_or_truncated"
            ],
            "active_partial_episode_at_budget": row[
                "active_partial_episode_at_budget"
            ],
            "value_normalization_enabled": row["value_normalization_enabled"],
            "selected_checkpoint_sha256": row["selected_checkpoint_sha256"],
        }
        for row in training_summary
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only review root exists: {output_root}")
    output_root.mkdir(parents=True)

    integrity = _read_json(run_root / "artifact_integrity.json")
    integrity_failures: list[str] = []
    for row in integrity["files"]:
        path = run_root / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or _sha256(path) != row["sha256"]
        ):
            integrity_failures.append(str(row["path"]))
    if integrity_failures:
        raise RuntimeError(f"source integrity failures: {integrity_failures}")

    run_manifest = _read_json(run_root / "run_manifest.json")
    completion = _read_json(run_root / "completion_receipt.json")
    gates = _read_json(run_root / "falsification_gates.json")
    training_summary = _read_json(run_root / "training_summary.json")
    checkpoint_candidates = _read_json(run_root / "checkpoint_selection.json")
    update_rows = _read_json(run_root / "update_records.json")
    evaluation_rows = _read_csv(run_root / "evaluation_rows.csv")
    optimizer_rows = _read_csv(run_root / "optimizer_step_records.csv")
    behavior_rows = _read_csv(run_root / "behavior_ledger.csv")

    if len(training_summary) != 30:
        raise RuntimeError("training cell count mismatch")
    if any(
        int(row["environment_steps"]) != 1440
        or int(row["updates"]) != 24
        or int(row["optimizer_steps"]) != 192
        for row in training_summary
    ):
        raise RuntimeError("per-cell budget mismatch")
    if len(checkpoint_candidates) != 120 or len(update_rows) != 720:
        raise RuntimeError("checkpoint/update producer count mismatch")
    if len(optimizer_rows) != 5760 or len(evaluation_rows) != 600:
        raise RuntimeError("optimizer/evaluation producer count mismatch")
    if len(behavior_rows) != int(completion["behavior_rows"]):
        raise RuntimeError("behavior producer count mismatch")

    service_pairs, service_methods = _service_tables(evaluation_rows)
    behavior_pairs, behavior_methods = _behavior_tables(behavior_rows)
    mechanism_pairs, mechanism_methods = _mechanism_tables(
        update_rows, optimizer_rows
    )
    selection_rows = _selection_rows(training_summary)
    _write_csv(output_root / "service_seed_pairs.csv", service_pairs)
    _write_csv(output_root / "service_method_summary.csv", service_methods)
    _write_csv(output_root / "behavior_seed_pairs.csv", behavior_pairs)
    _write_csv(output_root / "behavior_method_summary.csv", behavior_methods)
    _write_csv(output_root / "mechanism_seed_pairs.csv", mechanism_pairs)
    _write_csv(output_root / "mechanism_method_summary.csv", mechanism_methods)
    _write_csv(output_root / "checkpoint_selection_summary.csv", selection_rows)

    _write_json(
        output_root / "gate_review.json",
        {
            "schema_version": "calibrated_workflow_value_normalization_gate_review_v1",
            "implemented_gates": gates,
            "review": {
                "mechanism_gate": {
                    "implemented_result": "pass",
                    "scope": "update-24 training batches for every cell",
                    "selected_checkpoint_aligned": False,
                    "reason": (
                        f"{sum(int(row['selected_update']) < 24 for row in training_summary)} "
                        "of 30 evaluated checkpoints were selected before update 24"
                    ),
                },
                "behavior_gate": {
                    "implemented_result": "fail",
                    "failing_clause": "candidate mean action-4 probability did not decrease",
                    "endogenous_state_visitation": True,
                },
                "service_gate": {
                    "implemented_result": "fail",
                    "failing_method": "ppo",
                    "failing_clause": "on-time completion decreased by 0.10",
                    "mappo_note": "completion/on-time neutral; failure-attempt and no-progress slightly worse",
                },
                "overall": "FALSIFIED_OR_NOT_PROMOTED",
            },
        },
    )
    _write_json(
        output_root / "review_manifest.json",
        {
            "schema_version": "calibrated_workflow_value_normalization_review_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_run": run_root.name,
            "source_git_commit": run_manifest["git_commit"],
            "source_integrity_file_count": len(integrity["files"]),
            "source_integrity_verified": True,
            "source_artifact_integrity_sha256": _sha256(
                run_root / "artifact_integrity.json"
            ),
            "source_run_manifest_sha256": _sha256(run_root / "run_manifest.json"),
            "source_completion_receipt_sha256": _sha256(
                run_root / "completion_receipt.json"
            ),
            "source_gate_sha256": _sha256(run_root / "falsification_gates.json"),
            "budget": {
                "cells": len(training_summary),
                "environment_steps": completion["actual_environment_steps"],
                "updates": completion["actual_updates"],
                "optimizer_steps": completion["actual_optimizer_steps"],
                "evaluation_rows": completion["evaluation_rows"],
                "behavior_rows": completion["behavior_rows"],
            },
            "checkpoint_candidates": len(checkpoint_candidates),
            "selected_update_counts": dict(
                sorted(Counter(str(row["selected_update"]) for row in training_summary).items())
            ),
            "reward_used_for_selection": any(
                bool(row["reward_used"]) for row in checkpoint_candidates
            ),
            "evaluation_or_holdout_used_for_selection": any(
                bool(row["evaluation_or_holdout_used"])
                for row in checkpoint_candidates
            ),
            "claim_boundary": "exposed development data only",
        },
    )
    files = sorted(
        path
        for path in output_root.iterdir()
        if path.is_file() and path.name != "artifact_integrity.json"
    )
    _write_json(
        output_root / "artifact_integrity.json",
        {
            "schema_version": "calibrated_workflow_value_normalization_review_integrity_v1",
            "files": [
                {
                    "path": path.name,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
            ],
        },
    )


if __name__ == "__main__":
    main()
