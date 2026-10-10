"""Analyze the SA event-auxiliary abstention experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.analyze_calibrated_workflow_service_feasible_event_target_ab import (  # noqa: E402
    analyze as analyze_service_candidate,
)
from scripts.analyze_calibrated_workflow_service_reward_alignment import _write_csv  # noqa: E402
from scripts.run_calibrated_workflow_interface_repair import (  # noqa: E402
    _integrity,
    _sha256,
    _write_json,
)
from scripts.run_calibrated_workflow_service_feasible_event_target_ab import (  # noqa: E402
    CONTROL_ARM,
    _load_json,
)
from scripts.run_calibrated_workflow_event_aux_abstention_ab import (  # noqa: E402
    CANDIDATE_ARM,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _training_summary(
    source_root: Path,
    control_root: Path,
) -> list[dict[str, Any]]:
    candidate = _read_csv(source_root / "optimizer_step_records.csv")
    control = [
        row
        for row in _read_csv(control_root / "optimizer_step_records.csv")
        if row["method"] == "sa_ghmappo"
    ]
    if len(candidate) != 3840 or len(control) != 3840:
        raise RuntimeError("optimizer-step denominator mismatch")
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for arm, rows in ((CONTROL_ARM, control), (CANDIDATE_ARM, candidate)):
        for row in rows:
            groups[(arm, str(row["seed"]))].append(row)
            groups[(arm, "combined")].append(row)
    result = []
    for (arm, seed), rows in sorted(groups.items()):
        candidate_fields = "event_aux_supervision_eligible_count" in rows[0]
        eligible = (
            sum(int(row["event_aux_supervision_eligible_count"]) for row in rows)
            if candidate_fields
            else None
        )
        supervised = (
            sum(int(row["event_aux_supervision_supervised_count"]) for row in rows)
            if candidate_fields
            else None
        )
        abstained = (
            sum(int(row["event_aux_supervision_abstained_count"]) for row in rows)
            if candidate_fields
            else None
        )
        result.append(
            {
                "event_target_arm": arm,
                "seed": seed,
                "optimizer_step_n": len(rows),
                "mean_auxiliary_loss": float(
                    np.mean([float(row["auxiliary_loss"]) for row in rows])
                ),
                "mean_weighted_auxiliary_grad_norm": float(
                    np.mean(
                        [float(row["weighted_auxiliary_grad_norm"]) for row in rows]
                    )
                ),
                "event_supervision_eligible_count": (
                    eligible if eligible is not None else ""
                ),
                "event_supervision_supervised_count": (
                    supervised if supervised is not None else ""
                ),
                "event_supervision_abstained_count": (
                    abstained if abstained is not None else ""
                ),
                "event_supervision_supervised_fraction": (
                    float(supervised) / eligible
                    if eligible
                    else (0.0 if eligible == 0 else "")
                ),
                "event_supervision_fraction_provenance": (
                    "direct_candidate_optimizer_log"
                    if candidate_fields
                    else "unavailable_in_historical_control_log"
                ),
            }
        )
    return result


def analyze(source_root: Path, control_root: Path, analysis_root: Path) -> None:
    analyze_service_candidate(
        source_root,
        control_root,
        analysis_root,
        control_arm=CONTROL_ARM,
        candidate_arm=CANDIDATE_ARM,
        analysis_schema="calibrated_workflow_event_aux_abstention_ab_analysis_v1",
    )
    training_summary = _training_summary(source_root, control_root)
    summary_path = analysis_root / "auxiliary_training_summary.csv"
    _write_csv(summary_path, training_summary)
    manifest_path = analysis_root / "analysis_manifest.json"
    manifest = _load_json(manifest_path)
    manifest["auxiliary_training_summary_sha256"] = _sha256(summary_path)
    manifest["historical_event_supervision_fraction_recoverable"] = False
    manifest["candidate_event_supervision_fraction_directly_logged"] = True
    _write_json(manifest_path, manifest)
    _integrity(analysis_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--analysis-root", required=True)
    args = parser.parse_args()
    analyze(
        Path(args.source_root).resolve(),
        Path(args.control_root).resolve(),
        Path(args.analysis_root).resolve(),
    )


if __name__ == "__main__":
    main()
