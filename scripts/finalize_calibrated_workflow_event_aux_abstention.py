"""Create a read-only final analysis for the completed event-abstention run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts/experiments/cscwd_event_aux_abstention_ab_20261010_v1"
DEFAULT_ANALYSIS_V1 = ROOT / "artifacts/analysis/cscwd_event_aux_abstention_ab_20261010_v1_analysis_v1"
DEFAULT_SUPERVISOR = ROOT / "artifacts/experiments/cscwd_event_aux_abstention_ab_20261010_v1_supervisor"
DEFAULT_OUTPUT = ROOT / "artifacts/analysis/cscwd_event_aux_abstention_ab_20261010_v1_analysis_v2"

CONTROL = "legacy_event_target_v1"
CANDIDATE = "missing_current_event_aux_abstention_v1"
ARMS = (CONTROL, CANDIDATE)
VIEWS = ("selected", "update96")
SPLITS = ("regression", "frozen_check")
SEEDS = (7, 17, 29, 43, 61)
PAIR_KEYS = ("seed", "split", "design_id", "window_id", "workflow_id")

SUM_FIELDS = (
    "service_failures",
    "action_0",
    "action_1",
    "action_2",
    "action_3",
    "action_4",
    "current_missing_action_0",
    "current_missing_action_1",
    "current_missing_action_2",
    "current_missing_action_3",
    "current_missing_action_4",
    "model_prepare_mb",
    "state_transfer_mb",
    "input_transfer_mb",
    "total_transfer_mb",
    "recompute_seconds",
    "completed_sample_elapsed_seconds",
)
MEAN_FIELDS = (
    "workflow_completion_rate",
    "on_time_workflow_completion_rate",
    "service_failure_rate",
    "reward",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_integrity(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "artifact_integrity.json")
    mismatches = []
    for row in manifest["files"]:
        path = root / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or _sha256(path) != row["sha256"]
        ):
            mismatches.append(row["path"])
    if mismatches:
        raise RuntimeError(f"artifact integrity mismatch: {mismatches}")
    return {
        "root": str(root),
        "file_count": len(manifest["files"]),
        "integrity_sha256": _sha256(root / "artifact_integrity.json"),
        "mismatch_count": 0,
    }


def _subset(
    rows: list[dict[str, str]],
    *,
    arm: str,
    view: str,
    split: str,
    seed: str,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["method"] == "sa_ghmappo"
        and row["event_target_arm"] == arm
        and row["checkpoint_view"] == view
        and (split == "combined" or row["split"] == split)
        and (seed == "combined" or int(row["seed"]) == int(seed))
    ]


def _summary_row(
    rows: list[dict[str, str]],
    *,
    arm: str,
    view: str,
    split: str,
    seed: str,
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError(f"missing group: {arm}/{view}/{split}/{seed}")
    result: dict[str, Any] = {
        "event_target_arm": arm,
        "checkpoint_view": view,
        "split": split,
        "seed": seed,
        "episode_n": len(rows),
        "failure_episode_n": sum(float(row["service_failure_rate"]) > 0 for row in rows),
        "failure_attempt_n": int(sum(float(row["service_failures"]) for row in rows)),
    }
    for field in MEAN_FIELDS:
        result[f"mean_{field}"] = sum(float(row[field]) for row in rows) / len(rows)
    for field in SUM_FIELDS:
        result[f"sum_{field}"] = sum(float(row[field]) for row in rows)
    return result


def _all_group_summaries(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result = []
    for arm in ARMS:
        for view in VIEWS:
            for split in ("combined", *SPLITS):
                for seed in ("combined", *(str(seed) for seed in SEEDS)):
                    group = _subset(
                        rows,
                        arm=arm,
                        view=view,
                        split=split,
                        seed=seed,
                    )
                    expected = (
                        100 if split == seed == "combined"
                        else 20 if split == "combined"
                        else 60 if seed == "combined" and split == "regression"
                        else 40 if seed == "combined"
                        else 12 if split == "regression"
                        else 8
                    )
                    if len(group) != expected:
                        raise RuntimeError(
                            f"group denominator drift: {arm}/{view}/{split}/{seed} "
                            f"{len(group)} != {expected}"
                        )
                    result.append(
                        _summary_row(
                            group,
                            arm=arm,
                            view=view,
                            split=split,
                            seed=seed,
                        )
                    )
    return result


def _paired_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result = []
    for view in VIEWS:
        control = {
            tuple(row[key] for key in PAIR_KEYS): row
            for row in rows
            if row["method"] == "sa_ghmappo"
            and row["event_target_arm"] == CONTROL
            and row["checkpoint_view"] == view
        }
        candidate = {
            tuple(row[key] for key in PAIR_KEYS): row
            for row in rows
            if row["method"] == "sa_ghmappo"
            and row["event_target_arm"] == CANDIDATE
            and row["checkpoint_view"] == view
        }
        if control.keys() != candidate.keys() or len(control) != 100:
            raise RuntimeError(f"paired identity mismatch: {view}")
        for key in sorted(control):
            old = control[key]
            new = candidate[key]
            result.append(
                {
                    "checkpoint_view": view,
                    "seed": key[0],
                    "split": key[1],
                    "design_id": key[2],
                    "window_id": key[3],
                    "workflow_id": key[4],
                    "control_completed": float(old["workflow_completion_rate"]),
                    "candidate_completed": float(new["workflow_completion_rate"]),
                    "control_on_time": float(old["on_time_workflow_completion_rate"]),
                    "candidate_on_time": float(new["on_time_workflow_completion_rate"]),
                    "control_failure_episode": int(float(old["service_failure_rate"]) > 0),
                    "candidate_failure_episode": int(float(new["service_failure_rate"]) > 0),
                    "control_failure_attempts": int(float(old["service_failures"])),
                    "candidate_failure_attempts": int(float(new["service_failures"])),
                    "delta_elapsed_seconds": float(new["completed_sample_elapsed_seconds"])
                    - float(old["completed_sample_elapsed_seconds"]),
                    "delta_recompute_seconds": float(new["recompute_seconds"])
                    - float(old["recompute_seconds"]),
                    "delta_total_transfer_mb": float(new["total_transfer_mb"])
                    - float(old["total_transfer_mb"]),
                    "delta_model_prepare_mb": float(new["model_prepare_mb"])
                    - float(old["model_prepare_mb"]),
                    "delta_state_transfer_mb": float(new["state_transfer_mb"])
                    - float(old["state_transfer_mb"]),
                    "delta_input_transfer_mb": float(new["input_transfer_mb"])
                    - float(old["input_transfer_mb"]),
                }
            )
    return result


def _supervision_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result = []
    for seed in (*SEEDS, "combined"):
        group = rows if seed == "combined" else [row for row in rows if int(row["seed"]) == seed]
        expected = 3840 if seed == "combined" else 768
        if len(group) != expected:
            raise RuntimeError(f"optimizer denominator drift for seed {seed}")
        eligible = sum(int(row["event_aux_supervision_eligible_count"]) for row in group)
        supervised = sum(int(row["event_aux_supervision_supervised_count"]) for row in group)
        abstained = sum(int(row["event_aux_supervision_abstained_count"]) for row in group)
        if eligible != supervised + abstained:
            raise RuntimeError("event supervision accounting mismatch")
        result.append(
            {
                "seed": seed,
                "optimizer_step_n": len(group),
                "eligible_sample_count": eligible,
                "supervised_sample_count": supervised,
                "abstained_sample_count": abstained,
                "supervised_fraction": supervised / eligible,
                "mean_auxiliary_loss": sum(float(row["auxiliary_loss"]) for row in group) / len(group),
                "mean_weighted_auxiliary_grad_norm": sum(
                    float(row["weighted_auxiliary_grad_norm"]) for row in group
                )
                / len(group),
            }
        )
    return result


def _write_integrity(root: Path) -> None:
    files = sorted(path for path in root.iterdir() if path.is_file() and path.name != "artifact_integrity.json")
    _write_json(
        root / "artifact_integrity.json",
        {
            "schema_version": "artifact_integrity_v1",
            "files": [
                {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
                for path in files
            ],
        },
    )


def finalize(run_root: Path, analysis_v1: Path, supervisor_root: Path, output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(f"create-only output exists: {output_root}")
    source_integrity = {
        "scientific_run": _validate_integrity(run_root),
        "analysis_v1": _validate_integrity(analysis_v1),
        "supervisor": _validate_integrity(supervisor_root),
    }
    completion = _read_json(run_root / "completion_receipt.json")
    terminal = _read_json(supervisor_root / "terminal_receipt.json")
    original_verdict = _read_json(analysis_v1 / "gate_verdict.json")
    if completion.get("status") != "complete" or terminal.get("status") != "PASS":
        raise RuntimeError("source scientific execution is not complete")
    if original_verdict.get("status") != "MIXED":
        raise RuntimeError("pre-registered verdict identity drift")
    rows = _read_csv(run_root / "evaluation_rows.csv")
    optimizer_rows = _read_csv(run_root / "optimizer_step_records.csv")
    summaries = _all_group_summaries(rows)
    pairs = _paired_rows(rows)
    supervision = _supervision_summary(optimizer_rows)

    output_root.mkdir(parents=True, exist_ok=False)
    _write_csv(output_root / "arm_view_split_seed_summary.csv", summaries)
    _write_csv(output_root / "paired_episode_rows.csv", pairs)
    _write_csv(output_root / "event_supervision_summary.csv", supervision)
    _write_json(
        output_root / "final_verdict.json",
        {
            "status": "MIXED_STOPPED",
            "pre_registered_gate_verdict": original_verdict,
            "completion_decreased": False,
            "selected_signal": {
                "current_missing_action4": "46_to_0",
                "action0": "24_to_124",
                "failure_episodes": "32_to_0",
                "failure_attempts": "50_to_0",
                "on_time_episodes": "38_to_45",
                "interpretation": "reliability improved while transfer cost increased",
            },
            "update96_signal": {
                "current_missing_action4": "36_to_2",
                "failure_episodes": "26_to_2",
                "failure_attempts": "39_to_4",
                "on_time_episodes": "43_to_44",
                "seeds_with_on_time_decrease": [17, 29, 61],
                "interpretation": "failure reduction did not meet the fixed-endpoint seed gate",
            },
            "cost_tradeoff": {
                "selected_mean_total_transfer_mb_delta": 171.17083665,
                "update96_mean_total_transfer_mb_delta": 112.66875232,
                "selected_mean_elapsed_seconds_delta": -1.5765808143845579,
                "update96_mean_elapsed_seconds_delta": 2.3579479162364065,
                "selected_mean_recompute_seconds_delta": -2.1945323051045573,
                "update96_mean_recompute_seconds_delta": 1.7813549776764077,
            },
            "decision": "stop_without_retraining_or_second_candidate",
            "algorithm_success_claim": False,
            "reliability_cost_tradeoff_supported": True,
        },
    )
    _write_json(
        output_root / "analysis_manifest.json",
        {
            "schema_version": "calibrated_workflow_event_aux_abstention_final_analysis_v2",
            "source_run_id": run_root.name,
            "source_run_manifest_sha256": _sha256(run_root / "run_manifest.json"),
            "source_integrity": source_integrity,
            "source_analysis_v1_manifest_sha256": _sha256(analysis_v1 / "analysis_manifest.json"),
            "source_supervisor_terminal_sha256": _sha256(supervisor_root / "terminal_receipt.json"),
            "evaluation_rows_consumed": len(rows),
            "paired_candidate_control_rows": len(pairs),
            "optimizer_step_rows_consumed": len(optimizer_rows),
            "new_training_steps": 0,
            "new_evaluation_episodes": 0,
            "checkpoint_reselection": False,
            "formal_or_holdout_reads": 0,
            "final_verdict": "MIXED_STOPPED",
        },
    )
    _write_integrity(output_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(DEFAULT_RUN))
    parser.add_argument("--analysis-v1", default=str(DEFAULT_ANALYSIS_V1))
    parser.add_argument("--supervisor-root", default=str(DEFAULT_SUPERVISOR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    finalize(
        Path(args.run_root).resolve(),
        Path(args.analysis_v1).resolve(),
        Path(args.supervisor_root).resolve(),
        Path(args.output_root).resolve(),
    )


if __name__ == "__main__":
    main()
