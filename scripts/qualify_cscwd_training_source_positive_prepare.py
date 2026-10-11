"""Qualify a frozen training source using the actual SA event-label consumer."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_calibrated_continuous_workflow_pilot import (  # noqa: E402
    _load_experiment_config,
)
from scripts.run_cscwd_phase_v2_causal_public_pilot import (  # noqa: E402
    _build_cell_agent,
)
from src.agents.causal_public_action_estimator import (  # noqa: E402
    public_prepare_advantage_label,
)


DEFAULT_PROTOCOL = (
    ROOT
    / "configs/experiment/cscwd_training_source_positive_prepare_qualification_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/analysis/cscwd_training_source_positive_prepare_qualification_20261011_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _git_clean() -> bool:
    return not subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _gate_decision(
    *, base_positive_prepare_count: int,
    candidate_prepare_to_abstain_count: int,
    consumer_mismatch_count: int,
    minimum_base_positive_prepare_count: int,
    minimum_candidate_prepare_to_abstain_count: int,
) -> dict[str, Any]:
    reasons = []
    if consumer_mismatch_count:
        reasons.append("actual_training_consumer_mismatch")
    if base_positive_prepare_count < minimum_base_positive_prepare_count:
        reasons.append("nonzero_base_positive_prepare_requirement_failed")
    if candidate_prepare_to_abstain_count < minimum_candidate_prepare_to_abstain_count:
        reasons.append("nonzero_candidate_intervention_requirement_failed")
    return {
        "gate_passed": not reasons,
        "training_authorized": False,
        "training_authorization_reason": (
            "separate_central_authorization_required_after_gate_pass"
            if not reasons
            else "stop_without_training"
        ),
        "failure_reasons": reasons,
    }


def _load_and_validate(protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "cscwd_training_source_positive_prepare_qualification_v1",
        "authorization_scope": "one_create_only_read_existing_probe_no_retry",
        "fixed_source_order": ["lankershim", "us_101"],
        "training_source_segment": "lankershim",
        "minimum_base_positive_prepare_count": 1,
        "minimum_candidate_prepare_to_abstain_count": 1,
        "require_actual_training_consumer_match": True,
        "maximum_total_environment_probe_steps": 192,
        "existing_environment_probe_steps": 96,
        "new_environment_probe_steps": 0,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "checkpoint_reads": 0,
        "formal_or_holdout_reads": 0,
        "automatic_retry": False,
        "on_gate_failure": "stop_without_training",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise RuntimeError(f"qualification protocol drift: {key}")
    for path_key, hash_key in (
        ("source_protocol", "source_protocol_sha256"),
        ("source_manifest", "source_manifest_sha256"),
    ):
        path = Path(protocol[path_key])
        path = path if path.is_absolute() else ROOT / path
        if not path.is_file() or _sha256(path) != protocol[hash_key]:
            raise RuntimeError(f"qualification input identity drift: {path_key}")
    probe_root = ROOT / protocol["existing_probe_root"]
    if (
        _sha256(probe_root / "artifact_integrity.json")
        != protocol["existing_probe_integrity_sha256"]
    ):
        raise RuntimeError("existing probe identity drift")
    integrity = json.loads((probe_root / "artifact_integrity.json").read_text())
    for row in integrity["files"]:
        if _sha256(probe_root / row["path"]) != row["sha256"]:
            raise RuntimeError(f"existing probe file drift: {row['path']}")
    source_manifest = json.loads(Path(protocol["source_manifest"]).read_text())
    selected = source_manifest["selected_windows"]
    if [row["source_segment_id"] for row in selected] != protocol["fixed_source_order"]:
        raise RuntimeError("frozen source order drift")
    training = selected[0]
    for manifest_key, protocol_key in (
        ("window_id", "training_window_id"),
        ("source_time_start", "training_source_time_start"),
        ("source_time_end", "training_source_time_end"),
        ("source_frame_start", "training_source_frame_start"),
        ("source_frame_end", "training_source_frame_end"),
    ):
        if training[manifest_key] != protocol[protocol_key]:
            raise RuntimeError(f"training interval identity drift: {manifest_key}")
    return protocol


def run(protocol_path: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"create-only output exists: {output_root}")
    if not _git_clean():
        raise RuntimeError("qualification checkout must be clean")
    protocol = _load_and_validate(protocol_path)
    source_protocol_path = ROOT / protocol["source_protocol"]
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    base_config_path = ROOT / source_protocol["inputs"]["base_config"]
    config, _ = _load_experiment_config(base_config_path)
    sa_cells = [
        row for row in source_protocol["training"]["learned_cells"]
        if row["method"] == "sa_ghmappo"
    ]
    if [row["arm"] for row in sa_cells] != [
        "sa_public_v3", "sa_calibration_abstention"
    ]:
        raise RuntimeError("SA consumer identity drift")
    torch.manual_seed(7)
    control = _build_cell_agent(sa_cells[0], 7, config)
    torch.manual_seed(7)
    candidate = _build_cell_agent(sa_cells[1], 7, config)

    probe_root = ROOT / protocol["existing_probe_root"]
    states = [json.loads(line) for line in (
        probe_root / "semantic_states.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    if len(states) != protocol["existing_environment_probe_steps"]:
        raise RuntimeError("existing probe step count drift")
    pseudo_targets = {"event_target": 0, "event_soft_target": 0.0}
    rows = []
    for row in states:
        state = row["semantic_state"]
        mask = row["action_mask"]
        base_direct = public_prepare_advantage_label(state, mask)
        gated_direct = public_prepare_advantage_label(
            state, mask,
            positive_prepare_calibration_abstention_enabled=True,
        )
        base_consumer = control._event_auxiliary_target(state, pseudo_targets)
        gated_consumer = candidate._event_auxiliary_target(state, pseudo_targets)
        base_matches = all(
            base_direct[key] == base_consumer[key]
            for key in ("decision", "reason", "event_target", "supervision_weight")
        )
        gated_matches = all(
            gated_direct[key] == gated_consumer[key]
            for key in ("decision", "reason", "event_target", "supervision_weight")
        )
        rows.append({
            "global_step": row["global_step"],
            "semantic_state_sha256": row["semantic_state_sha256"],
            "base_decision": base_direct["decision"],
            "base_reason": base_direct["reason"],
            "base_supervision_weight": base_direct["supervision_weight"],
            "candidate_decision": gated_direct["decision"],
            "candidate_reason": gated_direct["reason"],
            "candidate_supervision_weight": gated_direct["supervision_weight"],
            "base_actual_consumer_match": base_matches,
            "candidate_actual_consumer_match": gated_matches,
        })
    base_positive = sum(
        row["base_decision"] == "prepare"
        and float(row["base_supervision_weight"]) > 0
        for row in rows
    )
    intervention = sum(
        row["base_decision"] == "prepare"
        and float(row["base_supervision_weight"]) > 0
        and row["candidate_decision"] == "abstain"
        and float(row["candidate_supervision_weight"]) == 0
        for row in rows
    )
    mismatches = sum(
        not row["base_actual_consumer_match"]
        or not row["candidate_actual_consumer_match"]
        for row in rows
    )
    gate = _gate_decision(
        base_positive_prepare_count=base_positive,
        candidate_prepare_to_abstain_count=intervention,
        consumer_mismatch_count=mismatches,
        minimum_base_positive_prepare_count=(
            protocol["minimum_base_positive_prepare_count"]
        ),
        minimum_candidate_prepare_to_abstain_count=(
            protocol["minimum_candidate_prepare_to_abstain_count"]
        ),
    )
    summary = {
        "schema_version": "cscwd_training_source_positive_prepare_qualification_result_v1",
        "status": "PASS",
        "git_commit": _git_commit(),
        "protocol_sha256": _sha256(protocol_path),
        "claim_scope": protocol["claim_scope"],
        "source_order": protocol["fixed_source_order"],
        "source_disposition": {
            "lankershim": {
                "role": "training",
                "states_checked": len(rows),
                "base_positive_prepare_count": base_positive,
                "candidate_prepare_to_abstain_count": intervention,
                "actual_training_consumer_mismatch_count": mismatches,
                "base_decision_counts": dict(sorted(Counter(
                    row["base_decision"] for row in rows
                ).items())),
                "base_reason_counts": dict(sorted(Counter(
                    row["base_reason"] for row in rows
                ).items())),
            },
            "us_101": {
                "role": "checkpoint_selection",
                "disposition": "excluded_before_label_or_reward_inspection",
                "states_checked": 0,
            },
        },
        **gate,
        "environment_probe_steps_total": len(rows),
        "new_environment_probe_steps": 0,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "checkpoint_reads": 0,
        "formal_or_holdout_reads": 0,
        "available_authorized_training_sources_after_gate": 0,
        "next_required_input": (
            "a separately preregistered development source/window with raw time/frame "
            "identity disjoint from selection/check intervals and nonzero base-positive "
            "prepare exposure; no source is currently authorized"
        ),
    }
    output_root.mkdir(parents=True)
    _write_json(output_root / "qualification.json", summary)
    _write_json(output_root / "consumer_rows.json", rows)
    files = []
    for path in sorted(output_root.iterdir()):
        if path.name != "artifact_integrity.json":
            files.append({
                "path": path.name,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            })
    _write_json(output_root / "artifact_integrity.json", {
        "schema_version": "cscwd_training_source_positive_prepare_qualification_integrity_v1",
        "files": files,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(
        run(args.protocol.resolve(), args.output.resolve()),
        ensure_ascii=False, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
