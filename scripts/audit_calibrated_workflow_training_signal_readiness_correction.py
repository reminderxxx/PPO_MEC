"""Publish a sidecar for the corrected training-signal readiness producer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = ROOT / "artifacts/experiments/cscwd_event_aux_abstention_ab_20261010_v1"
DEFAULT_CONTROL = ROOT / "artifacts/experiments/cscwd_causal_prepared_state_visibility_matched_20261010_v1"
DEFAULT_OUTPUT = ROOT / "artifacts/analysis/cscwd_training_signal_readiness_correction_20261010_v1"

EXPECTED = {
    "candidate": {
        "row_count": 28_800,
        "integrity_sha256": "fa03b3fe9eed9af66f96e10e10d90db380b13a28893716071a395a8e4eb129cc",
    },
    "control": {
        "row_count": 115_200,
        "integrity_sha256": "b66dfb946d2fef8c8e93b3ffacd7f332c1280635c7c635ef1f58f7eb8c4e2ac1",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_source(name: str, root: Path) -> dict[str, Any]:
    integrity_path = root / "artifact_integrity.json"
    if _sha256(integrity_path) != EXPECTED[name]["integrity_sha256"]:
        raise RuntimeError(f"{name} source integrity identity mismatch")
    manifest = _read_json(integrity_path)
    for item in manifest["files"]:
        path = root / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(item["bytes"])
            or _sha256(path) != item["sha256"]
        ):
            raise RuntimeError(f"{name} source artifact mismatch: {item['path']}")
    signal_path = root / "training_signal_rows.csv"
    with signal_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    false_count = sum(row["current_bundle_ready"] == "False" for row in rows)
    if len(rows) != EXPECTED[name]["row_count"] or false_count != len(rows):
        raise RuntimeError(f"{name} historical readiness defect witness drift")
    return {
        "run_id": root.name,
        "artifact_integrity_sha256": _sha256(integrity_path),
        "artifact_file_count": len(manifest["files"]),
        "training_signal_rows_sha256": _sha256(signal_path),
        "training_signal_row_count": len(rows),
        "historical_false_count": false_count,
        "historical_field_status": "INVALID_DO_NOT_USE_FOR_READINESS_STRATIFICATION",
    }


def audit(candidate_root: Path, control_root: Path, output_root: Path, expected_commit: str) -> None:
    actual_commit = _git_commit()
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"producer fix commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    if output_root.exists():
        raise FileExistsError(f"create-only output exists: {output_root}")
    sources = {
        "candidate": _validate_source("candidate", candidate_root),
        "control": _validate_source("control", control_root),
    }
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(
        output_root / "correction_receipt.json",
        {
            "schema_version": "calibrated_workflow_training_signal_readiness_correction_v1",
            "status": "PRODUCER_CORRECTED_HISTORICAL_ROWS_PRESERVED",
            "producer_fix_commit": actual_commit,
            "producer": "scripts/run_calibrated_workflow_value_normalization_ab.py::_training_signal_row",
            "old_address": "semantic_state.current_rsu_id",
            "corrected_address": "semantic_state.vehicles[primary_vehicle_id].associated_rsu_id_with_first_vehicle_fallback",
            "sources": sources,
            "historical_rows_rewritten": 0,
            "historical_corrected_values_reconstructed": False,
            "new_training_steps": 0,
            "new_evaluation_episodes": 0,
            "checkpoint_reselection": False,
            "confirmed_impact": [
                "historical_training_signal_rows.current_bundle_ready_is_invalid",
                "future_readiness_stratification_from_this_producer_is_corrected",
            ],
            "not_confirmed_affected": [
                "agent_event_abstention",
                "training_loss",
                "optimizer_receipts",
                "checkpoint_selection",
                "evaluation_rows",
                "event_abstention_mixed_stopped_verdict",
            ],
            "consumer_audit": {
                "event_abstention_finalizer": "does_not_consume_training_signal_readiness",
                "event_abstention_analyzer": "does_not_consume_training_signal_readiness",
                "strong_baseline_analyzer": "uses_training_signal_file_count_only",
                "budget_extension_prefix_identity": "compares_historical_rows_for_identity_not_readiness_truth",
                "value_normalization_behavior_gate": "would_consume_corrected_field_in_future_runs",
            },
            "a_independent_audit": {
                "commit": "7734d5807feaa66ac3f40e89ca8ff86f911f0783",
                "tree": "210c1727c7947ba80dce6066fc158f969b6496f5",
                "report_sha256": "3354ee2704a0ba9f52f3d2ae478afda952202ecaffc64714b73f59e26d28d92c",
                "summary_sha256": "45498f3000fd6d9ea1994437770fef7dc2be24b5f09e2b7632bdc2eb3a14bf33",
                "bounded_manifest_sha256": "859abe71b5465cceeb20787a429ec932274177311d181a724f47e404cc35c4b7",
                "supplement_manifest_sha256": "ee49c70bff27b14e3360e92872dc957669f72722d8944573e21d5e4f79c15bc9",
            },
        },
    )
    receipt = output_root / "correction_receipt.json"
    _write_json(
        output_root / "artifact_integrity.json",
        {
            "schema_version": "artifact_integrity_v1",
            "files": [
                {
                    "path": receipt.name,
                    "sha256": _sha256(receipt),
                    "bytes": receipt.stat().st_size,
                }
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--control-root", default=str(DEFAULT_CONTROL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--expected-git-commit", required=True)
    args = parser.parse_args()
    audit(
        Path(args.candidate_root).resolve(),
        Path(args.control_root).resolve(),
        Path(args.output_root).resolve(),
        args.expected_git_commit,
    )


if __name__ == "__main__":
    main()
