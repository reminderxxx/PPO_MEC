"""Run the authorized SA-only service-feasible event-target experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.analyze_calibrated_workflow_service_reward_alignment import _write_csv  # noqa: E402
from scripts.run_calibrated_workflow_interface_repair import (  # noqa: E402
    _git_commit,
    _integrity,
    _sha256,
    _write_json,
)
from scripts.run_calibrated_workflow_prepared_state_visibility_matched import (  # noqa: E402
    LONG_DESIGN,
    NEW_PROFILE,
    _evaluate_checkpoint,
)
from scripts.run_calibrated_workflow_strong_baselines import (  # noqa: E402
    _build_learned,
    _load_inputs,
    _load_json,
    _read_csv,
    _run_learned_cell,
    _validate_protocol,
)


DEFAULT_PROTOCOL = "configs/experiment/calibrated_workflow_service_feasible_event_target_ab_v1.json"
DEFAULT_CONTROL_ROOT = ROOT_DIR / "artifacts/experiments/cscwd_causal_prepared_state_visibility_matched_20261010_v1"
CANDIDATE_ARM = "service_feasible_event_target_v1"
CONTROL_ARM = "legacy_event_target_v1"


def _validate_control(protocol: dict[str, Any], root: Path) -> dict[str, list[dict[str, str]]]:
    frozen = protocol["historical_control"]
    required = {
        "run_manifest.json": frozen["run_manifest_sha256"],
        "completion_receipt.json": frozen["completion_receipt_sha256"],
        "artifact_integrity.json": frozen["artifact_integrity_sha256"],
        "evaluation_rows.csv": frozen["evaluation_rows_sha256"],
        "training_summary.json": frozen["training_summary_sha256"],
        "checkpoint_selection.json": frozen["checkpoint_selection_sha256"],
        "new_selected_evaluation_rows.csv": frozen["new_selected_evaluation_rows_sha256"],
        "new_update96_evaluation_rows.csv": frozen["new_update96_evaluation_rows_sha256"],
        "reused_rule_evaluation_rows.csv": frozen["rule_rows_sha256"],
    }
    if root.name != frozen["run_id"]:
        raise RuntimeError("control run id mismatch")
    for name, expected in required.items():
        path = root / name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"control source identity mismatch: {name}")
    manifest = _load_json(root / "run_manifest.json")
    receipt = _load_json(root / "completion_receipt.json")
    if (
        manifest.get("git_commit") != frozen["scientific_commit"]
        or manifest.get("interface_profile") != NEW_PROFILE
        or receipt.get("status") != "complete"
        or int(receipt.get("learned_environment_steps", -1)) != 115200
        or int(receipt.get("new_evaluation_episodes", -1)) != 1200
    ):
        raise RuntimeError("control scientific identity mismatch")
    integrity = _load_json(root / "artifact_integrity.json")
    for item in integrity["files"]:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"control integrity mismatch: {item['path']}")
    selected = _read_csv(root / "new_selected_evaluation_rows.csv")
    fixed = _read_csv(root / "new_update96_evaluation_rows.csv")
    rules = _read_csv(root / "reused_rule_evaluation_rows.csv")
    if len(selected) != 400 or len(fixed) != 400 or len(rules) != 40:
        raise RuntimeError("control reference row count mismatch")
    return {"selected": selected, "update96": fixed, "rules": rules}


def _load_candidate_inputs(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    identity = protocol["scientific_identity"]
    base = ROOT_DIR / identity["base_config"]
    manifest = ROOT_DIR / identity["workload_manifest"]
    if _sha256(base) != identity["base_config_sha256"] or _sha256(manifest) != identity["workload_manifest_sha256"]:
        raise RuntimeError("base config or workload manifest identity mismatch")
    design = _load_json(ROOT_DIR / LONG_DESIGN)
    _validate_protocol(design)
    config, splits, source_identity = _load_inputs(design)
    config["interface_profile"] = NEW_PROFILE
    config["mechanism_aux_current_service_feasibility_gate_enabled"] = True
    return config, splits, source_identity, design


def _network_identity(config: dict[str, Any]) -> dict[str, Any]:
    legacy_config = {**config, "mechanism_aux_current_service_feasibility_gate_enabled": False}
    legacy = _build_learned("sa_ghmappo", 7, legacy_config, popart_enabled=False)
    candidate = _build_learned("sa_ghmappo", 7, config, popart_enabled=False)
    legacy_state = legacy._network.state_dict()
    candidate_state = candidate._network.state_dict()
    if legacy_state.keys() != candidate_state.keys() or any(
        not torch.equal(legacy_state[key], candidate_state[key]) for key in legacy_state
    ):
        raise RuntimeError("target-only candidate changed initialized network identity")
    legacy_count = sum(int(parameter.numel()) for parameter in legacy._network.parameters())
    candidate_count = sum(int(parameter.numel()) for parameter in candidate._network.parameters())
    if legacy_count != candidate_count or candidate_count != 165512:
        raise RuntimeError("target-only candidate parameter identity mismatch")
    return {
        "legacy_parameter_count": legacy_count,
        "candidate_parameter_count": candidate_count,
        "state_dict_keys": len(legacy_state),
        "initialized_tensors_equal": True,
    }


def preflight(protocol_path: Path, control_root: Path) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    if protocol.get("schema_version") != "calibrated_workflow_service_feasible_event_target_ab_v1":
        raise RuntimeError("candidate protocol version mismatch")
    references = _validate_control(protocol, control_root)
    config, splits, source_identity, _ = _load_candidate_inputs(protocol)
    if protocol["single_variable"]["candidate_event_target"] != "legacy_event_target_and_current_complete_bundle_ready":
        raise RuntimeError("candidate target semantics drift")
    if protocol["training"] != {
        "environment_steps_per_seed": 5760,
        "transitions_per_update": 60,
        "updates_per_seed": 96,
        "optimizer_steps_per_seed": 768,
        "checkpoint_update_candidates": [24, 48, 72, 96],
        "seeds": 5,
        "total_environment_steps": 28800,
        "total_optimizer_steps": 3840,
        "episode_max_steps": 24,
        "new_selected_evaluation_episodes": 100,
        "new_update96_evaluation_episodes": 100,
        "total_new_evaluation_episode_cap": 200,
        "wall_clock_cap_seconds": 3600,
        "automatic_retry": False,
        "scientific_launches": 1,
    }:
        raise RuntimeError("candidate budget drift")
    return {
        "status": "preflight_passed",
        "execution_authorized": bool(protocol["execution_authorized"]),
        "authorization_state": protocol["authorization_state"],
        "branch_gate_status": protocol["symmetric_branch_gate"]["status"],
        "scientific_steps": 0,
        "new_evaluation_episodes": 0,
        "control_rows": {key: len(value) for key, value in references.items()},
        "split_counts": {key: len(value) for key, value in splits.items()},
        "source_identity": source_identity,
        "network_identity": _network_identity(config),
        "protocol_sha256": _sha256(protocol_path),
    }


def _label(rows: list[dict[str, Any]], arm: str, view: str, reused: bool) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "event_target_arm": arm,
            "checkpoint_view": view,
            "reused_without_reevaluation": reused,
        }
        for row in rows
    ]


def execute(protocol_path: Path, control_root: Path, output_root: Path, command: list[str]) -> None:
    protocol = _load_json(protocol_path)
    receipt = preflight(protocol_path, control_root)
    gate = protocol["symmetric_branch_gate"]
    if (
        not protocol["execution_authorized"]
        or protocol["authorization_state"] != "authorized_after_symmetric_branch_pass"
        or gate.get("status") != "PASS"
        or not gate.get("manifest_sha256")
        or not gate.get("report_sha256")
    ):
        raise RuntimeError("candidate scientific execution is not authorized by a PASS branch gate")
    config, splits, source_identity, design = _load_candidate_inputs(protocol)
    references = _validate_control(protocol, control_root)
    output_root.mkdir(parents=False, exist_ok=False)
    _write_json(output_root / "runner_entered.json", {
        "status": "entered", "entered_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(), "protocol_sha256": _sha256(protocol_path),
        "branch_gate_manifest_sha256": gate["manifest_sha256"],
    })
    training = deepcopy(design["training"])
    training.update({
        "environment_steps_per_method_seed": 5760,
        "transitions_per_update": 60,
        "update_opportunities_per_method_seed": 96,
        "expected_optimizer_steps_per_method_seed": 768,
        "checkpoint_update_candidates": [24, 48, 72, 96],
        "fixed_endpoint_diagnostic_updates": [96],
        "evaluation_splits": ["regression", "frozen_check"],
        "episode_max_steps": 24,
    })
    checkpoints = output_root / "checkpoints"
    cells = [
        _run_learned_cell(
            method="sa_ghmappo", seed=int(seed), config=config, splits=splits,
            protocol=training, checkpoints=checkpoints,
            reward_profile="original_reward_v1", popart_enabled=False,
        )
        for seed in protocol["scientific_identity"]["seeds"]
    ]
    candidate_selected = _label(
        [row for cell in cells for row in cell["evaluation_rows"]],
        CANDIDATE_ARM, "selected", False,
    )
    candidate_selected_ledger = _label(
        [row for cell in cells for row in cell["behavior_rows"]],
        CANDIDATE_ARM, "selected", False,
    )
    candidate_fixed: list[dict[str, Any]] = []
    candidate_fixed_ledger: list[dict[str, Any]] = []
    for cell in cells:
        checkpoint = next(row for row in cell["candidates"] if int(row["update_index"]) == 96)
        rows, ledger = _evaluate_checkpoint(
            method="sa_ghmappo", seed=int(checkpoint["seed"]),
            checkpoint=output_root / checkpoint["checkpoint"],
            expected_hash=checkpoint["checkpoint_sha256"], config=config,
            splits=splits, step_cap=24,
        )
        candidate_fixed.extend(_label(rows, CANDIDATE_ARM, "update96", False))
        candidate_fixed_ledger.extend(_label(ledger, CANDIDATE_ARM, "update96", False))
    control_selected = _label(references["selected"], CONTROL_ARM, "selected", True)
    control_fixed = _label(references["update96"], CONTROL_ARM, "update96", True)
    rules = _label(references["rules"], "historical_rule_reference", "selected", True)
    if len(candidate_selected) != 100 or len(candidate_fixed) != 100:
        raise RuntimeError("candidate evaluation budget drift")
    all_rows = control_selected + control_fixed + candidate_selected + candidate_fixed + rules
    identity = {
        (
            row["event_target_arm"], row["checkpoint_view"], row["method"],
            str(row["seed"]), row["split"], row["design_id"],
        )
        for row in all_rows
    }
    if len(identity) != len(all_rows):
        raise RuntimeError("duplicate candidate evaluation identity")
    _write_csv(output_root / "evaluation_rows.csv", all_rows)
    _write_csv(output_root / "candidate_selected_evaluation_rows.csv", candidate_selected)
    _write_csv(output_root / "candidate_update96_evaluation_rows.csv", candidate_fixed)
    _write_csv(output_root / "candidate_selected_behavior_ledger.csv", candidate_selected_ledger)
    _write_csv(output_root / "candidate_update96_behavior_ledger.csv", candidate_fixed_ledger)
    _write_json(output_root / "training_summary.json", [cell["summary"] for cell in cells])
    _write_json(output_root / "update_records.json", [row for cell in cells for row in cell["updates"]])
    _write_csv(output_root / "optimizer_step_records.csv", [row for cell in cells for row in cell["optimizer_rows"]])
    _write_json(output_root / "checkpoint_selection.json", [row for cell in cells for row in cell["candidates"]])
    _write_json(output_root / "training_episode_rows.json", [row for cell in cells for row in cell["training_episodes"]])
    _write_csv(output_root / "training_signal_rows.csv", [row for cell in cells for row in cell["training_signals"]])
    _write_json(output_root / "preflight_receipt.json", receipt)
    _write_json(output_root / "run_manifest.json", {
        "schema_version": "calibrated_workflow_service_feasible_event_target_ab_run_v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(),
        "protocol": {"path": str(protocol_path.relative_to(ROOT_DIR)), "sha256": _sha256(protocol_path)},
        "event_target_arm": CANDIDATE_ARM,
        "mechanism_aux_event_target_semantics": "current_complete_bundle_ready_v1",
        "interface_profile": NEW_PROFILE, "policy_evaluation_mode": "raw_policy",
        "single_variable": protocol["single_variable"], "branch_gate": gate,
        "source_interval_validation": source_identity, "network_identity": _network_identity(config),
        "training_budget": training, "historical_control": protocol["historical_control"],
        "new_evaluation_episode_count": 200, "reused_control_episode_rows": 800,
        "reused_rule_rows": 40, "formal_or_holdout_reads": 0,
        "claim_boundary": protocol["claim_boundary"], "command": command,
    })
    _write_json(output_root / "completion_receipt.json", {
        "status": "complete", "learned_cells": len(cells),
        "learned_environment_steps": sum(cell["summary"]["environment_steps"] for cell in cells),
        "learned_optimizer_steps": sum(cell["summary"]["optimizer_steps"] for cell in cells),
        "new_evaluation_episodes": 200, "control_reused_without_reevaluation": True,
        "formal_or_holdout_reads": 0, "automatic_retry": False,
        "scientific_execution_complete": True,
    })
    _integrity(output_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--control-root", default=str(DEFAULT_CONTROL_ROOT))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-root")
    parser.add_argument("--expected-git-commit")
    args = parser.parse_args()
    if args.preflight == args.run:
        raise RuntimeError("select exactly one of --preflight or --run")
    protocol_path = (ROOT_DIR / args.protocol).resolve()
    control_root = Path(args.control_root).resolve()
    if args.preflight:
        print(json.dumps(preflight(protocol_path, control_root), ensure_ascii=False, sort_keys=True))
        return
    if not args.output_root or not args.expected_git_commit:
        raise RuntimeError("run requires output root and expected git commit")
    if args.expected_git_commit != _git_commit():
        raise RuntimeError("frozen git commit mismatch")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT_DIR, check=True, capture_output=True, text=True)
    if status.stdout.strip():
        raise RuntimeError("scientific run requires a clean checkout")
    execute(protocol_path, control_root, Path(args.output_root).resolve(), sys.argv)


if __name__ == "__main__":
    main()
