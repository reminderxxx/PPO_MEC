"""Create the predeclared descriptive verdict for the phase-v2 causal pilot."""

from __future__ import annotations

import argparse
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

from scripts.run_calibrated_workflow_interface_repair import _integrity, _sha256, _write_json  # noqa: E402


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: list[dict[str, str]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def analyze(run_root: Path, analysis_root: Path) -> dict[str, Any]:
    if analysis_root.exists():
        raise FileExistsError(f"create-only analysis root exists: {analysis_root}")
    receipt_path = run_root / "completion_receipt.json"
    manifest_path = run_root / "run_manifest.json"
    integrity_path = run_root / "artifact_integrity.json"
    if not all(path.is_file() for path in (receipt_path, manifest_path, integrity_path)):
        raise RuntimeError("complete source artifact is required")
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if receipt.get("status") != "complete" or manifest.get("status") != "complete":
        raise RuntimeError("source artifact is not terminal complete")
    rows = _read(run_root / "evaluation_rows.csv")
    if len(rows) != 320:
        raise RuntimeError("learned evaluation row count drift")
    key_fields = ("view", "seed", "time_profile", "condition_id", "source_segment_id")
    control = {
        tuple(row[field] for field in key_fields): row
        for row in rows if row["arm"] == "sa_control"
    }
    candidate = {
        tuple(row[field] for field in key_fields): row
        for row in rows if row["arm"] == "sa_candidate"
    }
    if control.keys() != candidate.keys() or len(control) != 80:
        raise RuntimeError("paired SA evaluation identity drift")
    metrics = (
        "workflow_completion_rate", "on_time_workflow_completion_rate",
        "service_failure_rate", "total_transfer_mb", "recompute_seconds",
        "partial_transfer_bytes", "decision_overhead_ms",
    )
    paired = []
    for key in sorted(control):
        row = {field: value for field, value in zip(key_fields, key)}
        row.update({
            f"{metric}_delta": float(candidate[key][metric]) - float(control[key][metric])
            for metric in metrics
        })
        paired.append(row)
    views: dict[str, Any] = {}
    strict_service_improvement = False
    fail = False
    resource_tradeoff = False
    for view in ("selected", "fixed_update24"):
        control_rows = [row for row in control.values() if row["view"] == view]
        candidate_rows = [row for row in candidate.values() if row["view"] == view]
        deltas = {
            metric: _mean(candidate_rows, metric) - _mean(control_rows, metric)
            for metric in metrics
        }
        service_nonharm = (
            deltas["workflow_completion_rate"] >= -1e-12
            and deltas["on_time_workflow_completion_rate"] >= -1e-12
            and deltas["service_failure_rate"] <= 1e-12
        )
        strict = (
            deltas["workflow_completion_rate"] > 1e-12
            or deltas["on_time_workflow_completion_rate"] > 1e-12
            or deltas["service_failure_rate"] < -1e-12
        )
        resource_nonharm = (
            deltas["total_transfer_mb"] <= 1e-12
            and deltas["recompute_seconds"] <= 1e-12
            and deltas["partial_transfer_bytes"] <= 1e-9
        )
        fail = fail or not service_nonharm
        strict_service_improvement = strict_service_improvement or strict
        resource_tradeoff = resource_tradeoff or (strict and not resource_nonharm)
        views[view] = {
            "control_means": {metric: _mean(control_rows, metric) for metric in metrics},
            "candidate_means": {metric: _mean(candidate_rows, metric) for metric in metrics},
            "candidate_minus_control": deltas,
            "service_nonharm": service_nonharm,
            "strict_service_improvement": strict,
            "resource_nonharm": resource_nonharm,
        }
    seed_directions = []
    favorable = unfavorable = 0
    for seed in ("7", "17", "29", "43", "61"):
        seed_rows = [row for row in paired if row["seed"] == seed]
        delta = {
            metric: float(np.mean([row[f"{metric}_delta"] for row in seed_rows]))
            for metric in metrics
        }
        nonharm = (
            delta["workflow_completion_rate"] >= -1e-12
            and delta["on_time_workflow_completion_rate"] >= -1e-12
            and delta["service_failure_rate"] <= 1e-12
        )
        strict = (
            delta["workflow_completion_rate"] > 1e-12
            or delta["on_time_workflow_completion_rate"] > 1e-12
            or delta["service_failure_rate"] < -1e-12
        )
        harmful = not nonharm
        favorable += int(nonharm and strict)
        unfavorable += int(harmful)
        seed_directions.append({
            "seed": int(seed), "deltas": delta,
            "direction": "favorable" if nonharm and strict else "unfavorable" if harmful else "neutral",
        })
    supported = bool(
        not fail and strict_service_improvement and favorable >= 3 and unfavorable <= 1
        and all(view["resource_nonharm"] for view in views.values())
    )
    verdict = (
        "FAIL" if fail else
        "MIXED" if resource_tradeoff or not supported else
        "SUPPORTED_DEVELOPMENT_CANDIDATE"
    )
    summary = {
        "schema_version": "cscwd_phase_v2_causal_public_pilot_analysis_v1",
        "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_id": run_root.name,
        "source_git_commit": manifest["git_commit"],
        "source_sha256": {
            "completion_receipt": _sha256(receipt_path),
            "run_manifest": _sha256(manifest_path),
            "artifact_integrity": _sha256(integrity_path),
            "evaluation_rows": _sha256(run_root / "evaluation_rows.csv"),
        },
        "verdict": verdict, "automatic_next_stage": False,
        "development_only": True, "paper_ready": False,
        "independent_unit_count": 2, "statistical_significance_claim": False,
        "views": views, "seed_directions": seed_directions,
        "favorable_seed_count": favorable, "unfavorable_seed_count": unfavorable,
        "strict_service_improvement": strict_service_improvement,
        "resource_tradeoff": resource_tradeoff,
        "paired_case_count": len(paired),
        "rule_capability_note": "phase-v2 public two-step falls back to the same immediate public score and is not an independent strong baseline",
        "privileged_rule_note": "exact two-step is reported only on phase-v1 capability reference and has no phase-v2 action permission",
    }
    analysis_root.mkdir(parents=True)
    _write_json(analysis_root / "analysis_summary.json", summary)
    with (analysis_root / "paired_sa_deltas.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(paired[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(paired)
    report = f"""# CSCWD phase-v2 public-causal pilot result

- Verdict: `{verdict}`
- Evidence level: development-only, two frozen source windows, no significance claim.
- Learned budget: 20 cells, 23,040 environment steps, 3,840 optimizer steps.
- Selection: one fixed US101/default development case across updates 6/12/18/24; final cases never select checkpoints.
- Candidate favorable seed directions: {favorable}/5; unfavorable: {unfavorable}/5.
- Resource tradeoff observed: `{str(resource_tradeoff).lower()}`.
- Automatic continuation: disabled.

This result does not alter formal/holdout evidence and is not paper-ready.
"""
    (analysis_root / "report.md").write_text(report, encoding="utf-8")
    _integrity(analysis_root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--analysis-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_root.resolve(), args.analysis_root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
