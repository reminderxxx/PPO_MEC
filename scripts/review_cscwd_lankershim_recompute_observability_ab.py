"""Read-only stratified review of the completed Lankershim post-fix A/B."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_calibrated_workflow_interface_repair import (  # noqa: E402
    _integrity, _write_csv, _write_json,
)


RUN_ID = "cscwd_phase_v2_recompute_observability_calibration_abstention_ab_20261011_v1"
DEFAULT_RUN_ROOT = ROOT / "artifacts/experiments" / RUN_ID
DEFAULT_ANALYSIS_ROOT = ROOT / "artifacts/analysis" / f"{RUN_ID}_analysis_v1"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/analysis" / f"{RUN_ID}_result_review_v1"
CONTROL = "sa_public_v4"
CANDIDATE = "sa_calibration_abstention_v2"
METRICS = (
    "reward", "workflow_completion_rate", "on_time_workflow_completion_rate",
    "service_failure_rate", "service_failures", "recompute_seconds",
    "total_transfer_mb", "migration_successes", "action_4",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float:
    text = str(value).strip().lower()
    if text == "true":
        return 1.0
    if text in {"false", ""}:
        return 0.0
    return float(text)


def _source_role(row: dict[str, Any]) -> str:
    window_id = str(row["window_id"])
    if "_v19_" in window_id:
        return "selection_v19"
    if "_v749_" in window_id:
        return "check_v749"
    raise RuntimeError(f"unexpected evaluation window: {window_id}")


def _profile(row: dict[str, Any]) -> str:
    return (
        "v1_mechanism_off"
        if str(row["time_profile"]).endswith("_v1")
        else "v2"
    )


def _verify_integrity(root: Path) -> dict[str, Any]:
    manifest_path = root / "artifact_integrity.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = root / item["path"]
        expected_size = item.get("bytes", item.get("size_bytes"))
        if (
            not path.is_file()
            or _sha256(path) != item["sha256"]
            or path.stat().st_size != expected_size
        ):
            raise RuntimeError(f"artifact integrity mismatch: {path}")
    return {
        "verified_file_count": len(manifest["files"]),
        "integrity_manifest_sha256": _sha256(manifest_path),
    }


def _means(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        metric: sum(_float(row[metric]) for row in rows) / len(rows)
        for metric in METRICS
    }


def review(run_root: Path, analysis_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"create-only review root exists: {output_root}")
    run_integrity = _verify_integrity(run_root)
    analysis_integrity = _verify_integrity(analysis_root)
    receipt = json.loads((run_root / "completion_receipt.json").read_text())
    manifest = json.loads((run_root / "run_manifest.json").read_text())
    existing = json.loads((analysis_root / "analysis_summary.json").read_text())
    if (
        receipt.get("scientific_execution_complete") is not True
        or receipt.get("status") != "complete"
        or manifest.get("status") != "complete"
        or existing.get("verdict") != "FAIL"
        or receipt.get("formal_or_holdout_reads") != 0
    ):
        raise RuntimeError("completed development source artifact is required")

    evaluations = _read_csv(run_root / "evaluation_rows.csv")
    behaviors = _read_csv(run_root / "behavior_rows.csv")
    signals = _read_csv(run_root / "training_signal_rows.csv")
    rules = _read_csv(run_root / "rule_evaluation_rows.csv")
    if len(evaluations) != 320 or len(signals) != 23040 or len(rules) != 18:
        raise RuntimeError("source row-count identity drift")

    grouped: defaultdict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in evaluations:
        grouped[(row["arm"], row["view"], _profile(row), _source_role(row))].append(row)
    method_rows = [
        {
            "arm": key[0], "view": key[1], "time_profile_group": key[2],
            "source_role": key[3], "episode_count": len(rows), **_means(rows),
        }
        for key, rows in sorted(grouped.items())
    ]

    episode_key = (
        "view", "seed", "time_profile", "condition_id", "window_id"
    )
    indexed = {
        (row["arm"],) + tuple(row[field] for field in episode_key): row
        for row in evaluations
    }
    delta_rows: list[dict[str, Any]] = []
    for view in ("selected", "fixed_update24"):
        for profile in ("v2", "v1_mechanism_off"):
            for source_role in ("selection_v19", "check_v749"):
                pairs = []
                for key, control in indexed.items():
                    if (
                        key[0] == CONTROL and key[1] == view
                        and _profile(control) == profile
                        and _source_role(control) == source_role
                    ):
                        pairs.append((control, indexed[(CANDIDATE,) + key[1:]]))
                delta_rows.append({
                    "view": view, "time_profile_group": profile,
                    "source_role": source_role, "paired_episode_count": len(pairs),
                    **{
                        f"{metric}_delta": sum(
                            _float(candidate[metric]) - _float(control[metric])
                            for control, candidate in pairs
                        ) / len(pairs)
                        for metric in METRICS
                    },
                })

    rule_groups: defaultdict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rules:
        rule_groups[(
            row["method"], _profile(row), _source_role(row),
            row.get("capability_profile", ""),
        )].append(row)
    rule_rows = [
        {
            "method": key[0], "time_profile_group": key[1],
            "source_role": key[2], "capability_profile": key[3],
            "episode_count": len(rows), **_means(rows),
            "independent_learned_baseline": False,
        }
        for key, rows in sorted(rule_groups.items())
    ]

    behavior_groups: defaultdict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in behaviors:
        key = (row["arm"],) + tuple(row[field] for field in episode_key)
        behavior_groups[key].append(row)
    for rows in behavior_groups.values():
        rows.sort(key=lambda row: int(row["step"]))
    harmed_rows: list[dict[str, Any]] = []
    for key, control_eval in indexed.items():
        if key[0] != CONTROL:
            continue
        candidate_eval = indexed[(CANDIDATE,) + key[1:]]
        if _float(candidate_eval["service_failures"]) <= _float(control_eval["service_failures"]):
            continue
        control_steps = behavior_groups[(CONTROL,) + key[1:]]
        candidate_steps = behavior_groups[(CANDIDATE,) + key[1:]]
        first_divergence = next(
            (
                index for index, (control, candidate) in enumerate(
                    zip(control_steps, candidate_steps)
                )
                if control["executed_action"] != candidate["executed_action"]
            ),
            None,
        )
        first_failure = next(
            (index for index, row in enumerate(candidate_steps) if row["failure_reason"]),
            None,
        )
        divergence_control = control_steps[first_divergence] if first_divergence is not None else {}
        divergence_candidate = candidate_steps[first_divergence] if first_divergence is not None else {}
        failure = candidate_steps[first_failure] if first_failure is not None else {}
        harmed_rows.append({
            "view": key[1], "seed": key[2],
            "time_profile": key[3], "condition_id": key[4],
            "source_role": _source_role(control_eval), "window_id": key[5],
            "control_service_failures": _float(control_eval["service_failures"]),
            "candidate_service_failures": _float(candidate_eval["service_failures"]),
            "reward_delta": _float(candidate_eval["reward"]) - _float(control_eval["reward"]),
            "first_divergence_step": first_divergence,
            "same_public_state_at_first_divergence": (
                divergence_control.get("semantic_state_sha256")
                == divergence_candidate.get("semantic_state_sha256")
            ),
            "control_action_at_first_divergence": divergence_control.get("executed_action"),
            "candidate_action_at_first_divergence": divergence_candidate.get("executed_action"),
            "candidate_label_at_first_divergence": divergence_candidate.get("event_label"),
            "candidate_first_failure_step": first_failure,
            "candidate_first_failure_action": failure.get("executed_action"),
            "candidate_first_failure_reason": failure.get("failure_reason"),
        })

    training = {}
    for arm in (CONTROL, CANDIDATE):
        arm_rows = [row for row in signals if row["arm"] == arm]
        training[arm] = {
            "row_count": len(arm_rows),
            "event_label_counts": dict(Counter(row["event_label"] for row in arm_rows)),
            "event_reason_counts": dict(Counter(row["event_label_reason"] for row in arm_rows)),
            "executed_action_counts": dict(Counter(row["executed_action"] for row in arm_rows)),
        }

    overall = {
        (row["arm"], row["view"]): row
        for row in _read_csv(analysis_root / "method_view_metrics.csv")
    }
    pooled = {}
    for view in ("selected", "fixed_update24"):
        control = overall[(CONTROL, view)]
        candidate = overall[(CANDIDATE, view)]
        pooled[view] = {
            metric: _float(candidate[metric]) - _float(control[metric])
            for metric in (
                "reward", "workflow_completion_rate",
                "on_time_workflow_completion_rate", "service_failure_rate",
                "service_failures", "recompute_seconds", "total_transfer_bytes",
                "migration_successes",
            )
        }

    if any(row["service_failures_delta"] != 0.0 for row in delta_rows if row["time_profile_group"] == "v2"):
        raise RuntimeError("expected v2 service-failure localization drift")
    if not harmed_rows or any(row["time_profile"] != "raw_ngsim_event_time_v1" for row in harmed_rows):
        raise RuntimeError("harmed episode localization drift")

    summary = {
        "schema_version": "cscwd_lankershim_recompute_observability_ab_result_review_v1",
        "status": "complete",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_run_id": RUN_ID,
        "source_git_commit": manifest["git_commit"],
        "source_run_integrity": run_integrity,
        "source_analysis_integrity": analysis_integrity,
        "scientific_terminal_status": "PASS",
        "scientific_verdict": "FAIL",
        "training": training,
        "pooled_sa_deltas": pooled,
        "harmed_evaluation_row_count": len(harmed_rows),
        "harmed_rows_all_v1_mechanism_off": True,
        "v2_service_failure_delta_zero_in_all_strata": True,
        "harmed_first_failure_reasons": dict(Counter(
            row["candidate_first_failure_reason"] for row in harmed_rows
        )),
        "automatic_analysis_correction": {
            "original_artifact_preserved": True,
            "two_exposed_mobility_sources": False,
            "actual_scope": "one_lankershim_recording_two_evaluation_windows",
            "independent_evaluation_unit_count": 0,
            "existing_independent_unit_count_2_interpretable_only_as_window_count": True,
        },
        "mechanism_interpretation": (
            "hard event-label abstention changes the shared-encoder policy and "
            "improves v2 deadline outcomes, but transfers poorly to v1 mechanism-off; "
            "the earliest harmed paths diverge on the same public state and end in "
            "current_rsu_contact_expires_before_commit"
        ),
        "next_single_variable_candidate": (
            "event_auxiliary_head_local_gradient_isolation_from_shared_encoder"
        ),
        "candidate_status": "FROZEN_ONLY_NOT_IMPLEMENTED_OR_RUN",
        "falsification": (
            "reject if fixed-update24 or selected service failure increases in any "
            "profile/source stratum, if v2 on-time gain disappears, or if intervention "
            "is zero under the same matched budget"
        ),
        "claim_boundary": (
            "development only; same recording; no significance, independent "
            "generalization, SA superiority, or paper-ready claim"
        ),
    }
    output_root.mkdir(parents=True)
    _write_json(output_root / "review_summary.json", summary)
    _write_csv(output_root / "method_stratified_metrics.csv", method_rows)
    _write_csv(output_root / "sa_stratified_deltas.csv", delta_rows)
    _write_csv(output_root / "rule_capability_metrics.csv", rule_rows)
    _write_csv(output_root / "harmed_episode_witnesses.csv", harmed_rows)
    _integrity(output_root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(
        review(args.run_root.resolve(), args.analysis_root.resolve(), args.output_root.resolve()),
        ensure_ascii=False, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
