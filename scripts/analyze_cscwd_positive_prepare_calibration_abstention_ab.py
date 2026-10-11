"""Analyze the frozen positive-prepare calibration-abstention development A/B."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_calibrated_workflow_interface_repair import (  # noqa: E402
    _integrity,
    _sha256,
    _write_csv,
    _write_json,
)


SCHEMA_VERSION = "cscwd_positive_prepare_calibration_abstention_analysis_v1"
CONTROL_ARM = "sa_public_v3"
CANDIDATE_ARM = "sa_calibration_abstention"
EXPECTED_ARMS = {CONTROL_ARM, CANDIDATE_ARM, "ppo", "mappo"}
KEY_FIELDS = (
    "view", "seed", "time_profile", "condition_id", "source_segment_id", "window_id"
)
EPISODE_KEY_FIELDS = ("arm",) + KEY_FIELDS
REWARD_METRICS = (
    "reward",
    "reward_node_completion",
    "reward_workflow_completion",
    "reward_time_penalty",
    "reward_transfer_penalty",
    "reward_failure_penalty",
    "reward_deadline_penalty",
)
SERVICE_METRICS = (
    "workflow_completion_rate",
    "on_time_workflow_completion_rate",
    "service_failure_rate",
    "service_failures",
    "unfinished_rate",
    "truncated",
)
COST_METRICS = (
    "elapsed_seconds",
    "total_transfer_bytes",
    "input_transfer_bytes",
    "model_transfer_bytes",
    "state_transfer_bytes",
    "backhaul_transfer_bytes",
    "partial_transfer_bytes",
    "recompute_seconds",
    "migration_successes",
    "target_departed_before_delivery",
)
ALL_METRICS = REWARD_METRICS + SERVICE_METRICS + COST_METRICS


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if text in {"true", "yes"}:
        return 1.0
    if text in {"false", "no", ""}:
        return 0.0
    return float(text)


def _mean(rows: list[dict[str, Any]], metric: str) -> float:
    return float(np.mean([_float(row[metric]) for row in rows]))


def _key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in KEY_FIELDS)


def _episode_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in EPISODE_KEY_FIELDS)


def _augment_episode_metrics(
    evaluation_rows: list[dict[str, str]],
    behavior_rows: list[dict[str, str]],
    expected_arms: set[str],
) -> list[dict[str, Any]]:
    behavior: defaultdict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in behavior_rows:
        if row["arm"] in expected_arms:
            behavior[_episode_key(row)].append(row)
    result: list[dict[str, Any]] = []
    for source in evaluation_rows:
        row: dict[str, Any] = dict(source)
        row["elapsed_seconds"] = _float(
            source.get("completed_sample_elapsed_seconds")
            or source.get("elapsed_seconds_for_failed_or_truncated")
        )
        row["total_transfer_bytes"] = int(
            round(_float(source["total_transfer_mb"]) * 1_000_000.0)
        )
        ledger = behavior[_episode_key(source)]
        row["target_departed_before_delivery"] = sum(
            str(step.get("failure_reason")) == "target_departed_before_delivery"
            for step in ledger
        )
        ledger_migrations = sum(
            str(step.get("migration_success", "")).lower() == "true"
            for step in ledger
        )
        if ledger_migrations != int(_float(source["migration_successes"])):
            raise RuntimeError("behavior/evaluation migration count mismatch")
        result.append(row)
    return result


def _service_nonharm(delta: dict[str, float]) -> bool:
    return bool(
        delta["workflow_completion_rate"] >= -1e-12
        and delta["on_time_workflow_completion_rate"] >= -1e-12
        and delta["service_failure_rate"] <= 1e-12
        and delta["service_failures"] <= 1e-12
        and delta["unfinished_rate"] <= 1e-12
        and delta["truncated"] <= 1e-12
    )


def _service_strict(delta: dict[str, float]) -> bool:
    return bool(
        delta["workflow_completion_rate"] > 1e-12
        or delta["on_time_workflow_completion_rate"] > 1e-12
        or delta["service_failure_rate"] < -1e-12
        or delta["service_failures"] < -1e-12
        or delta["unfinished_rate"] < -1e-12
        or delta["truncated"] < -1e-12
    )


def _resource_nonharm(delta: dict[str, float]) -> bool:
    lower_is_better = (
        "elapsed_seconds", "total_transfer_bytes", "input_transfer_bytes",
        "model_transfer_bytes", "state_transfer_bytes", "backhaul_transfer_bytes",
        "partial_transfer_bytes", "recompute_seconds",
        "target_departed_before_delivery",
    )
    return all(delta[metric] <= 1e-9 for metric in lower_is_better)


def _resource_strict(delta: dict[str, float]) -> bool:
    lower_is_better = (
        "elapsed_seconds", "total_transfer_bytes", "input_transfer_bytes",
        "model_transfer_bytes", "state_transfer_bytes", "backhaul_transfer_bytes",
        "partial_transfer_bytes", "recompute_seconds",
        "target_departed_before_delivery",
    )
    return any(delta[metric] < -1e-9 for metric in lower_is_better)


def analyze(run_root: Path, analysis_root: Path) -> dict[str, Any]:
    if analysis_root.exists():
        raise FileExistsError(f"create-only analysis root exists: {analysis_root}")
    required = {
        name: run_root / name
        for name in (
            "completion_receipt.json", "run_manifest.json", "artifact_integrity.json",
            "evaluation_rows.csv", "behavior_rows.csv", "training_signal_rows.csv",
            "training_updates.csv", "rule_evaluation_rows.csv",
        )
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"complete source artifact is required: {missing}")
    receipt = json.loads(required["completion_receipt.json"].read_text())
    manifest = json.loads(required["run_manifest.json"].read_text())
    manifest_schema = manifest.get("schema_version")
    if (
        receipt.get("scientific_execution_complete") is not True
        or receipt.get("status") != "complete"
        or manifest.get("status") != "complete"
        or manifest_schema not in {
            "cscwd_phase_v2_positive_prepare_calibration_abstention_ab_manifest_v1",
            "cscwd_phase_v2_recompute_observability_calibration_abstention_ab_manifest_v1",
        }
    ):
        raise RuntimeError("source scientific artifact is not contract-complete")
    postfix_run = manifest_schema == (
        "cscwd_phase_v2_recompute_observability_calibration_abstention_ab_manifest_v1"
    )
    control_arm = "sa_public_v4" if postfix_run else CONTROL_ARM
    candidate_arm = "sa_calibration_abstention_v2" if postfix_run else CANDIDATE_ARM
    expected_arms = {control_arm, candidate_arm, "ppo", "mappo"}
    scope = manifest.get("calibration_abstention_scope") or {}
    expected_scope = {
        "raw_producer_confidence_calibrated": False,
        "new_arm_effect": (
            "abstains_positive_prepare_auxiliary_supervision_when_"
            "public_prediction_confidence_is_not_calibrated"
            if postfix_run
            else "removes_all_positive_prepare_auxiliary_supervision"
        ),
        "fine_grained_uncertainty_control": False,
        "three_head_structure_contribution_tested": False,
        "single_variable": "mechanism_aux_positive_prepare_calibration_abstention_enabled",
    }
    if scope != expected_scope:
        raise RuntimeError("calibration-abstention claim scope drift")

    raw_evaluation = _read_csv(required["evaluation_rows.csv"])
    behavior = _read_csv(required["behavior_rows.csv"])
    signals = _read_csv(required["training_signal_rows.csv"])
    updates = _read_csv(required["training_updates.csv"])
    rules = _read_csv(required["rule_evaluation_rows.csv"])
    if len(raw_evaluation) != 320 or len(signals) != 23040 or len(updates) != 480:
        raise RuntimeError("scientific row count drift")
    arm_counts = Counter(row["arm"] for row in raw_evaluation)
    if arm_counts != Counter({arm: 80 for arm in expected_arms}):
        raise RuntimeError(f"learned arm identity drift: {dict(arm_counts)}")
    rows = _augment_episode_metrics(raw_evaluation, behavior, expected_arms)
    control = {_key(row): row for row in rows if row["arm"] == control_arm}
    candidate = {_key(row): row for row in rows if row["arm"] == candidate_arm}
    if control.keys() != candidate.keys() or len(control) != 80:
        raise RuntimeError("paired SA evaluation identity drift")

    paired_rows: list[dict[str, Any]] = []
    for key in sorted(control):
        output: dict[str, Any] = {
            field: value for field, value in zip(KEY_FIELDS, key)
        }
        output.update({
            f"{metric}_control": _float(control[key][metric])
            for metric in ALL_METRICS
        })
        output.update({
            f"{metric}_candidate": _float(candidate[key][metric])
            for metric in ALL_METRICS
        })
        output.update({
            f"{metric}_delta": (
                _float(candidate[key][metric]) - _float(control[key][metric])
            )
            for metric in ALL_METRICS
        })
        paired_rows.append(output)

    method_view_rows: list[dict[str, Any]] = []
    for arm in sorted(expected_arms):
        for view in ("selected", "fixed_update24"):
            subset = [row for row in rows if row["arm"] == arm and row["view"] == view]
            method_view_rows.append({
                "arm": arm,
                "view": view,
                "episode_count": len(subset),
                **{metric: _mean(subset, metric) for metric in ALL_METRICS},
            })

    view_rows: list[dict[str, Any]] = []
    seed_view_rows: list[dict[str, Any]] = []
    any_service_harm = False
    all_view_resource_nonharm = True
    any_view_resource_improvement = False
    reward_lead_views = 0
    favorable_seeds: set[int] = set()
    unfavorable_seeds: set[int] = set()
    for view in ("selected", "fixed_update24"):
        subset = [row for row in paired_rows if row["view"] == view]
        delta = {
            metric: float(np.mean([row[f"{metric}_delta"] for row in subset]))
            for metric in ALL_METRICS
        }
        view_service_nonharm = _service_nonharm(delta)
        view_resource_nonharm = _resource_nonharm(delta)
        view_resource_strict = _resource_strict(delta)
        reward_lead = delta["reward"] > 1e-12
        reward_lead_views += int(reward_lead)
        any_service_harm = any_service_harm or not view_service_nonharm
        all_view_resource_nonharm = all_view_resource_nonharm and view_resource_nonharm
        any_view_resource_improvement = (
            any_view_resource_improvement or view_resource_strict
        )
        view_rows.append({
            "view": view,
            **{f"{metric}_delta": value for metric, value in delta.items()},
            "service_nonharm": view_service_nonharm,
            "service_strict": _service_strict(delta),
            "resource_nonharm": view_resource_nonharm,
            "resource_strict": view_resource_strict,
            "reward_lead": reward_lead,
        })
        for seed in (7, 17, 29, 43, 61):
            seed_subset = [
                row for row in subset if int(row["seed"]) == seed
            ]
            seed_delta = {
                metric: float(np.mean([
                    row[f"{metric}_delta"] for row in seed_subset
                ]))
                for metric in ALL_METRICS
            }
            service_nonharm = _service_nonharm(seed_delta)
            service_strict = _service_strict(seed_delta)
            resource_nonharm = _resource_nonharm(seed_delta)
            resource_strict = _resource_strict(seed_delta)
            any_service_harm = any_service_harm or not service_nonharm
            if not service_nonharm:
                unfavorable_seeds.add(seed)
            elif service_strict or (resource_nonharm and resource_strict):
                favorable_seeds.add(seed)
            seed_view_rows.append({
                "view": view,
                "seed": seed,
                **{f"{metric}_delta": value for metric, value in seed_delta.items()},
                "service_nonharm": service_nonharm,
                "service_strict": service_strict,
                "resource_nonharm": resource_nonharm,
                "resource_strict": resource_strict,
                "reward_lead": seed_delta["reward"] > 1e-12,
            })

    candidate_signals = [
        row for row in signals if row["arm"] == candidate_arm
    ]
    control_signals = [row for row in signals if row["arm"] == control_arm]
    if len(candidate_signals) != 5760 or len(control_signals) != 5760:
        raise RuntimeError("SA training-signal row count drift")
    candidate_positive_prepare_supervised = sum(
        row["event_label"] == "prepare"
        and _float(row["event_supervision_weight"]) > 0.0
        for row in candidate_signals
    )
    control_positive_prepare_supervised = sum(
        row["event_label"] == "prepare"
        and _float(row["event_supervision_weight"]) > 0.0
        for row in control_signals
    )
    intervention_rows = sum(
        row["event_label"] == "abstain"
        and row.get("event_label_reason")
        == "positive_prepare_prediction_confidence_not_calibrated"
        for row in candidate_signals
    )
    implementation_failure = (
        candidate_positive_prepare_supervised != 0
        or (postfix_run and intervention_rows == 0)
    )
    supported = bool(
        not implementation_failure
        and not any_service_harm
        and all_view_resource_nonharm
        and any_view_resource_improvement
        and len(favorable_seeds) >= 3
        and len(unfavorable_seeds) <= 1
    )
    reward_tradeoff = bool(
        reward_lead_views > 0
        and (any_service_harm or not all_view_resource_nonharm)
    )
    if implementation_failure or any_service_harm:
        verdict = "FAIL"
    elif supported:
        verdict = "SUPPORTED_DEVELOPMENT_CANDIDATE"
    else:
        verdict = "MIXED"

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_id": run_root.name,
        "source_git_commit": manifest["git_commit"],
        "source_sha256": {
            name: _sha256(path) for name, path in required.items()
        },
        "evidence_level": "E2_ARTIFACT_AUDITED_DEVELOPMENT_ONLY",
        "verdict": verdict,
        "automatic_next_stage": False,
        "paper_ready": False,
        "development_only": True,
        "independent_unit_count": 2,
        "statistical_significance_claim": False,
        "control_arm": control_arm,
        "candidate_arm": candidate_arm,
        "calibration_abstention_scope": scope,
        "candidate_positive_prepare_supervised_count": (
            candidate_positive_prepare_supervised
        ),
        "control_positive_prepare_supervised_count": (
            control_positive_prepare_supervised
        ),
        "candidate_calibration_abstention_intervention_count": intervention_rows,
        "implementation_failure": implementation_failure,
        "service_nonharm_checked_per_seed_and_view": True,
        "any_seed_view_service_harm": any_service_harm,
        "all_view_resource_nonharm": all_view_resource_nonharm,
        "any_view_resource_improvement": any_view_resource_improvement,
        "reward_lead_view_count": reward_lead_views,
        "reward_tradeoff": reward_tradeoff,
        "favorable_seed_count": len(favorable_seeds),
        "unfavorable_seed_count": len(unfavorable_seeds),
        "views": view_rows,
        "seed_views": seed_view_rows,
        "method_views": method_view_rows,
        "rule_row_count": len(rules),
        "rule_capability_note": (
            "phase-v2 public two-step uses the same public immediate fallback; "
            "privileged exact two-step remains phase-v1 capability reference only"
        ),
        "claim_boundary": (
            "tests positive-prepare auxiliary abstention only; does not test "
            "three-head structural contribution, real-RSU net benefit, "
            "sharing-by-migration interaction, or independent generalization"
        ),
    }
    analysis_root.mkdir(parents=True)
    _write_json(analysis_root / "analysis_summary.json", summary)
    _write_csv(analysis_root / "paired_sa_episode_deltas.csv", paired_rows)
    _write_csv(analysis_root / "method_view_metrics.csv", method_view_rows)
    _write_csv(analysis_root / "seed_view_deltas.csv", seed_view_rows)
    report = f"""# Positive-prepare calibration-abstention A/B

- Verdict: `{verdict}`
- Evidence: development-only; two exposed mobility sources; no significance claim.
- Candidate supervised positive-prepare rows: `{candidate_positive_prepare_supervised}`.
- Reward-leading views: `{reward_lead_views}/2`; reward tradeoff: `{str(reward_tradeoff).lower()}`.
- Seed-view service harm: `{str(any_service_harm).lower()}`.
- Favorable/unfavorable seeds: `{len(favorable_seeds)}/{len(unfavorable_seeds)}`.
- Automatic continuation: disabled.

This run tests only the calibration-abstention auxiliary gate. It does not test
three-head structural contribution, real-RSU net benefit, sharing-by-migration
interaction, independent generalization, or paper readiness.
"""
    (analysis_root / "report.md").write_text(report, encoding="utf-8")
    _integrity(analysis_root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--analysis-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(
        analyze(args.run_root.resolve(), args.analysis_root.resolve()),
        ensure_ascii=False,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
