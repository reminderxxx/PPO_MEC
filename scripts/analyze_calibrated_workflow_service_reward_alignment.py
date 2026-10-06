"""Analyze the frozen service-reward experiment without selecting favorable subsets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "on_time_workflow_completion_rate",
    "workflow_completion_rate",
    "unfinished_after_deadline_rate",
    "service_failure_rate",
    "failed_service_attempt_seconds_proxy",
    "total_transfer_mb",
    "model_prepare_mb",
    "recompute_seconds",
    "invalid_prepare_attempts",
    "max_consecutive_no_progress_steps",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_delta(rows: list[dict[str, Any]], split: str, method: str, metric: str) -> dict[str, Any]:
    subset = [row for row in rows if row["split"] == split and row["method"] == method]
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in subset:
        key = (str(row["source_segment_id"]), str(row["window_id"]))
        grouped[key][str(row["reward_arm"])].append(float(row[metric]))
    deltas = []
    for key, arms in sorted(grouped.items()):
        if "original_reward_v1" not in arms or "service_aligned_v1" not in arms:
            continue
        deltas.append(float(np.mean(arms["service_aligned_v1"]) - np.mean(arms["original_reward_v1"])))
    if not deltas:
        return {"mean": None, "ci95": [None, None], "outer_n": 0}
    rng = np.random.default_rng(20261006)
    values = np.asarray(deltas, dtype=float)
    samples = np.asarray([float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(5000)])
    return {
        "mean": float(np.mean(values)),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "outer_n": len(values),
        "outer_deltas": deltas,
    }


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["reward_arm"]), str(row["method"]))].append(row)
    output: list[dict[str, Any]] = []
    for (split, arm, method), group in sorted(grouped.items()):
        completed = [float(row["completed_sample_elapsed_seconds"]) for row in group if row.get("completed_sample_elapsed_seconds", "") != ""]
        missing_actions = sum(int(row["current_missing_action_4"]) for row in group)
        missing_decisions = sum(int(row["current_model_missing_decisions"]) for row in group)
        item: dict[str, Any] = {
            "split": split,
            "reward_arm": arm,
            "method": method,
            "episode_n": len(group),
            "source_window_n": len({str(row["window_id"]) for row in group}),
            "completed_elapsed_seconds": float(np.mean(completed)) if completed else "",
            "completed_elapsed_coverage": f"{len(completed)}/{len(group)}",
            "current_missing_action4_rate": float(missing_actions) / max(missing_decisions, 1),
        }
        for metric in METRICS:
            item[metric] = float(np.mean([float(row[metric]) for row in group]))
        output.append(item)
    return output


def _seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    learned = [row for row in rows if row["seed"] != "rule"]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in learned:
        grouped[(str(row["split"]), str(row["reward_arm"]), str(row["method"]), str(row["seed"]))].append(row)
    for (split, arm, method, seed), group in sorted(grouped.items()):
        output.append(
            {
                "split": split,
                "reward_arm": arm,
                "method": method,
                "seed": seed,
                "episode_n": len(group),
                "on_time_workflow_completion_rate": float(np.mean([float(row["on_time_workflow_completion_rate"]) for row in group])),
                "workflow_completion_rate": float(np.mean([float(row["workflow_completion_rate"]) for row in group])),
                "service_failure_rate": float(np.mean([float(row["service_failure_rate"]) for row in group])),
                "invalid_prepare_attempts": float(np.mean([float(row["invalid_prepare_attempts"]) for row in group])),
            }
        )
    return output


def _reward_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "reward_node_completion", "reward_workflow_completion", "reward_time_penalty",
        "reward_service_operation_time_penalty", "reward_transfer_penalty",
        "reward_recompute_penalty", "reward_failure_penalty", "reward_deadline_penalty", "reward",
        "rescored_return_original_reward_v1", "rescored_return_service_aligned_v1",
    )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["reward_arm"]), str(row["method"]))].append(row)
    output = []
    for (split, arm, method), group in sorted(grouped.items()):
        item: dict[str, Any] = {"split": split, "reward_arm": arm, "method": method, "episode_n": len(group)}
        for key in keys:
            item[key] = float(np.mean([float(row.get(key, 0.0) or 0.0) for row in group]))
        output.append(item)
    return output


def _negative_regions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    fields = ("handoff_pressure", "link_error_class", "capacity", "sharing", "topology_class")
    for method in ("sa_ghmappo", "mappo", "ppo"):
        for field in fields:
            values = sorted({str(row[field]) for row in rows})
            for value in values:
                subset = [row for row in rows if row["method"] == method and str(row[field]) == value]
                for split in ("regression", "frozen_check"):
                    selected = [row for row in subset if row["split"] == split]
                    arm_values = {}
                    for arm in ("original_reward_v1", "service_aligned_v1"):
                        arm_rows = [row for row in selected if row["reward_arm"] == arm]
                        if arm_rows:
                            arm_values[arm] = float(np.mean([float(row["workflow_completion_rate"]) for row in arm_rows]))
                    if len(arm_values) == 2 and arm_values["service_aligned_v1"] < arm_values["original_reward_v1"]:
                        output.append({"split": split, "method": method, "stratum": field, "value": value, "original_completion": arm_values["original_reward_v1"], "candidate_completion": arm_values["service_aligned_v1"], "delta": arm_values["service_aligned_v1"] - arm_values["original_reward_v1"]})
    return output


def _paper_table(summary: list[dict[str, Any]]) -> str:
    lines = [
        "| split | reward | method | on-time | completion | service-failure episode | failed-attempt proxy s | transfer MB | recompute s | completed elapsed / coverage | missing-model action 4 | invalid prepare | max no-progress |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {split} | {reward_arm} | {method} | {on_time_workflow_completion_rate:.3f} | {workflow_completion_rate:.3f} | {service_failure_rate:.3f} | {failed_service_attempt_seconds_proxy:.2f} | {total_transfer_mb:.2f} | {recompute_seconds:.2f} | {completed_elapsed_seconds} / {completed_elapsed_coverage} | {current_missing_action4_rate:.3f} | {invalid_prepare_attempts:.2f} | {max_consecutive_no_progress_steps:.2f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    receipt = _load(root / "completion_receipt.json")
    if receipt.get("status") != "complete":
        raise RuntimeError("training run is not complete")
    rows = _load(root / "evaluation_rows.json")
    summary = _summary_rows(rows)
    seed_summary = _seed_rows(rows)
    deltas = {
        split: {
            method: {metric: _paired_delta(rows, split, method, metric) for metric in METRICS}
            for method in ("sa_ghmappo", "mappo", "ppo", "two_step_cost_rule")
        }
        for split in ("regression", "frozen_check")
    }
    negative = _negative_regions(rows)
    _write_csv(root / "service_metric_summary.csv", summary)
    _write_csv(root / "seed_summary.csv", seed_summary)
    _write(root / "paired_reward_arm_deltas.json", deltas)
    _write_csv(root / "reward_decomposition.csv", _reward_breakdown(rows))
    if negative:
        _write_csv(root / "negative_regions.csv", negative)
    else:
        _write(root / "negative_regions.json", {"rows": [], "note": "No completion-decrease stratum; inspect other trade-offs in paired deltas."})
    (root / "paper_table.md").write_text(_paper_table(summary), encoding="utf-8")
    _write(root / "analysis_receipt.json", {"status": "complete", "all_seeds_preserved": True, "all_instances_preserved": True, "bootstrap_outer_unit": "source_window", "bootstrap_draws": 5000, "reward_returns_compared_only_by_trajectory_rescoring": True})
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_integrity.json"
    )
    _write(root / "artifact_integrity.json", {"files": [{"path": str(path.relative_to(root)), "sha256": _sha256(path), "bytes": path.stat().st_size} for path in files]})
    print(json.dumps({"run_root": str(root), "status": "complete", "summary_rows": len(summary), "negative_regions": len(negative)}))


if __name__ == "__main__":
    main()
