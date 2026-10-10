"""Run the preregistered shared prepared-state observation matched experiment.

The historical selected view and rule rows are copied byte-identified from the
completed long-budget run. Only the two update-96 views and the new selected
view execute evaluation episodes; no final evaluation participates in selection.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_calibrated_workflow_interface_repair import (  # noqa: E402
    _evaluate_agent,
    _git_commit,
    _integrity,
    _sha256,
    _write_json,
)
from scripts.run_calibrated_workflow_strong_baselines import (  # noqa: E402
    LEARNED_METHODS,
    _annotate_rows,
    _build_learned,
    _load_inputs,
    _load_json,
    _read_csv,
    _run_learned_cell,
    _validate_protocol,
)
from scripts.analyze_calibrated_workflow_service_reward_alignment import _write_csv  # noqa: E402


DEFAULT_PROTOCOL = "configs/experiment/calibrated_workflow_prepared_state_visibility_matched_protocol_v1.json"
LONG_DESIGN = "configs/experiment/calibrated_workflow_strong_baselines_budget_extension_v1.json"
NEW_PROFILE = "calibrated_workflow_interface_v4_prepared_state_prefix"
OLD_PROFILE = "calibrated_workflow_interface_v3_prefix_only"
DEFAULT_HISTORICAL_ROOT = ROOT_DIR / "artifacts/experiments/cscwd_causal_strong_baselines_budget_extension_20261009_v1"
DEFAULT_RULE_ROOT = Path(
    "/Users/howen/.codex/worktrees/cscwd-window-identity/PPO_MEC/artifacts/experiments/"
    "cscwd_causal_strong_baselines_dev_20261009_v1"
)
EXPECTED_PARAMETER_COUNTS = {
    "sa_ghmappo": (165320, 165512),
    "mappo": (39176, 39624),
    "ppo": (22406, 22854),
    "dt_handoff_drl": (44808, 45256),
}


def _canonical_json_hash(value: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _validate_historical(
    protocol: dict[str, Any], root: Path, rule_root: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]]]:
    frozen = protocol["historical_arm"]
    required = {
        "run_manifest.json": frozen["run_manifest_sha256"],
        "completion_receipt.json": frozen["completion_receipt_sha256"],
        "artifact_integrity.json": frozen["artifact_integrity_sha256"],
        "evaluation_rows.csv": frozen["evaluation_rows_sha256"],
        "checkpoint_selection.json": frozen["checkpoint_selection_sha256"],
        "training_summary.json": frozen["training_summary_sha256"],
    }
    if root.name != frozen["scientific_run_id"]:
        raise RuntimeError("historical run id mismatch")
    for name, expected in required.items():
        path = root / name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"historical source identity mismatch: {name}")
    manifest = _load_json(root / "run_manifest.json")
    receipt = _load_json(root / "completion_receipt.json")
    if manifest.get("git_commit") != frozen["scientific_commit"] or receipt.get("status") != "complete":
        raise RuntimeError("historical scientific identity mismatch")
    rows = _read_csv(root / "evaluation_rows.csv")
    selected = [row for row in rows if row["method"] in LEARNED_METHODS]
    if len(selected) != 400 or len(rows) != 400:
        raise RuntimeError("historical selected row count mismatch")
    rule_path = rule_root / "evaluation_rows.csv"
    if (
        rule_root.name != frozen["rule_source_run_id"]
        or not rule_path.is_file()
        or _sha256(rule_path) != frozen["rule_source_evaluation_rows_sha256"]
    ):
        raise RuntimeError("historical rule source identity mismatch")
    rule_rows = [row for row in _read_csv(rule_path) if row["method"] not in LEARNED_METHODS]
    if len(rule_rows) != 40:
        raise RuntimeError("historical rule row count mismatch")
    candidates = _load_json(root / "checkpoint_selection.json")
    update96 = [row for row in candidates if int(row["update_index"]) == 96]
    if len(update96) != 20:
        raise RuntimeError("historical update-96 checkpoint identity mismatch")
    for row in update96:
        path = root / row["checkpoint"]
        if not path.is_file() or _sha256(path) != row["checkpoint_sha256"]:
            raise RuntimeError(f"historical checkpoint hash mismatch: {path.name}")
    return rows, rule_rows, update96


def _load_matched_inputs(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    scientific = protocol["frozen_scientific_identity"]
    base = ROOT_DIR / scientific["base_config"]
    manifest = ROOT_DIR / scientific["workload_manifest"]
    if _sha256(base) != scientific["base_config_sha256"] or _sha256(manifest) != scientific["workload_manifest_sha256"]:
        raise RuntimeError("base config or workload manifest hash mismatch")
    design = _load_json(ROOT_DIR / LONG_DESIGN)
    _validate_protocol(design)
    config, splits, identity = _load_inputs(design)
    if config["interface_profile"] != OLD_PROFILE:
        raise RuntimeError("historical loader profile drift")
    config["interface_profile"] = NEW_PROFILE
    return config, splits, identity, design


def _parameter_receipt(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method in LEARNED_METHODS:
        old = _build_learned(method, 7, {**config, "interface_profile": OLD_PROFILE}, popart_enabled=False)
        new = _build_learned(method, 7, config, popart_enabled=False)
        old_count = sum(int(item.numel()) for item in old._network.parameters())
        new_count = sum(int(item.numel()) for item in new._network.parameters())
        if (old_count, new_count) != EXPECTED_PARAMETER_COUNTS[method]:
            raise RuntimeError(f"parameter count drift: {method}")
        rows.append({
            "method": method,
            "old_parameter_count": old_count,
            "new_parameter_count": new_count,
            "delta": new_count - old_count,
        })
    return rows


def preflight(protocol_path: Path, historical_root: Path, rule_root: Path) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    if protocol["schema_version"] != "calibrated_workflow_prepared_state_visibility_matched_protocol_v1":
        raise RuntimeError("matched protocol version drift")
    config, splits, identity, design = _load_matched_inputs(protocol)
    historical_rows, rule_rows, update96 = _validate_historical(protocol, historical_root, rule_root)
    document = ROOT_DIR / protocol["diagnostic_basis"]["document"]
    if not document.is_file() or _sha256(document) != protocol["diagnostic_basis"]["document_sha256"]:
        raise RuntimeError("diagnostic basis identity mismatch")
    intervention = protocol["intervention"]
    handoff = ROOT_DIR / "docs/project/cscwd_prepared_state_prefix_interface_20261010.md"
    env_path = ROOT_DIR / "src/envs/core/calibrated_continuous_workflow_env.py"
    if _sha256(handoff) != intervention["implementation_document_sha256"]:
        raise RuntimeError("implementation handoff identity mismatch")
    if _sha256(env_path) != intervention["implementation_env_sha256"]:
        raise RuntimeError("implementation environment identity mismatch")
    if protocol["training_budget"] != {
        "environment_steps_per_method_seed": 5760,
        "transitions_per_update": 60,
        "update_opportunities_per_method_seed": 96,
        "expected_optimizer_steps_per_method_seed": 768,
        "checkpoint_update_candidates": [24, 48, 72, 96],
        "learned_cells": 20,
        "total_environment_steps": 115200,
        "total_update_opportunities": 1920,
        "total_optimizer_steps": 15360,
        "episode_max_steps": 24,
        "wall_clock_cap_seconds": 7200,
        "automatic_retry": False,
        "scientific_launches": 1,
    }:
        raise RuntimeError("matched budget drift")
    if design["training"]["checkpoint_update_candidates"] != [24, 48, 72, 96]:
        raise RuntimeError("selection schedule drift")
    parameters = _parameter_receipt(config)
    return {
        "status": "preflight_passed",
        "execution_authorized": bool(protocol["execution_authorized"]),
        "scientific_steps": 0,
        "new_evaluation_episodes": 0,
        "split_counts": {key: len(value) for key, value in splits.items()},
        "historical_selected_rows": len([row for row in historical_rows if row["method"] in LEARNED_METHODS]),
        "historical_rule_rows": len(rule_rows),
        "historical_update96_checkpoints": len(update96),
        "source_identity": identity,
        "parameter_counts": parameters,
        "protocol_sha256": _sha256(protocol_path),
        "implementation_commit": intervention["implementation_commit"],
        "implementation_document_sha256": intervention["implementation_document_sha256"],
        "implementation_env_sha256": intervention["implementation_env_sha256"],
    }


def _label(rows: list[dict[str, Any]], *, arm: str, view: str, reused: bool) -> list[dict[str, Any]]:
    return [{**row, "observation_arm": arm, "checkpoint_view": view, "reused_without_reevaluation": reused} for row in rows]


def _evaluate_checkpoint(
    *, method: str, seed: int, checkpoint: Path, expected_hash: str,
    config: dict[str, Any], splits: dict[str, list[dict[str, Any]]], step_cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if _sha256(checkpoint) != expected_hash:
        raise RuntimeError(f"checkpoint hash mismatch before evaluation: {checkpoint.name}")
    agent = _build_learned(method, seed, config, popart_enabled=False)
    agent.load(str(checkpoint))
    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for split in ("regression", "frozen_check"):
        one_rows, one_ledger = _evaluate_agent(agent, method, seed, config, splits[split], step_cap)
        one_rows, one_ledger = _annotate_rows(one_rows, one_ledger, splits[split], "original_reward_v1")
        rows.extend(one_rows)
        ledger.extend(one_ledger)
    return rows, ledger


def execute(
    protocol_path: Path, historical_root: Path, rule_root: Path,
    output_root: Path, command: list[str],
) -> None:
    protocol = _load_json(protocol_path)
    receipt = preflight(protocol_path, historical_root, rule_root)
    if not protocol["execution_authorized"] or protocol["authorization_state"] != "authorized_after_independent_preflight":
        raise RuntimeError("scientific execution is not authorized")
    if protocol["intervention"]["implementation_commit"] != "709746bc1f1ea3037f27497bdb51cf6f45c8963c":
        raise RuntimeError("implementation commit drift")
    config, splits, identity, design = _load_matched_inputs(protocol)
    historical_rows, rule_rows, historical_update96 = _validate_historical(
        protocol, historical_root, rule_root
    )
    output_root.mkdir(parents=False, exist_ok=False)
    _write_json(output_root / "runner_entered.json", {
        "status": "entered", "entered_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(), "protocol_sha256": _sha256(protocol_path),
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
    cells = []
    for method in LEARNED_METHODS:
        for seed in protocol["frozen_scientific_identity"]["seeds"]:
            cells.append(_run_learned_cell(
                method=method, seed=int(seed), config=config, splits=splits,
                protocol=training, checkpoints=checkpoints,
                reward_profile="original_reward_v1", popart_enabled=False,
            ))
    new_selected = _label(
        [row for cell in cells for row in cell["evaluation_rows"]],
        arm="prepared_state_visible_v4", view="selected", reused=False,
    )
    new_selected_ledger = _label(
        [row for cell in cells for row in cell["behavior_rows"]],
        arm="prepared_state_visible_v4", view="selected", reused=False,
    )
    new_update96: list[dict[str, Any]] = []
    new_update96_ledger: list[dict[str, Any]] = []
    for cell in cells:
        candidate = next(row for row in cell["candidates"] if int(row["update_index"]) == 96)
        rows, ledger = _evaluate_checkpoint(
            method=str(candidate["method"]), seed=int(candidate["seed"]),
            checkpoint=output_root / candidate["checkpoint"], expected_hash=candidate["checkpoint_sha256"],
            config=config, splits=splits, step_cap=24,
        )
        new_update96.extend(_label(rows, arm="prepared_state_visible_v4", view="update96", reused=False))
        new_update96_ledger.extend(_label(ledger, arm="prepared_state_visible_v4", view="update96", reused=False))
    old_config = {**config, "interface_profile": OLD_PROFILE}
    historical_fixed: list[dict[str, Any]] = []
    historical_fixed_ledger: list[dict[str, Any]] = []
    for candidate in historical_update96:
        rows, ledger = _evaluate_checkpoint(
            method=str(candidate["method"]), seed=int(candidate["seed"]),
            checkpoint=historical_root / candidate["checkpoint"], expected_hash=candidate["checkpoint_sha256"],
            config=old_config, splits=splits, step_cap=24,
        )
        historical_fixed.extend(_label(rows, arm="prepared_state_hidden_v3", view="update96", reused=False))
        historical_fixed_ledger.extend(_label(ledger, arm="prepared_state_hidden_v3", view="update96", reused=False))
    historical_selected = _label(
        [row for row in historical_rows if row["method"] in LEARNED_METHODS],
        arm="prepared_state_hidden_v3", view="selected", reused=True,
    )
    historical_rules = _label(
        rule_rows,
        arm="historical_rule_reference", view="selected", reused=True,
    )
    for name, rows in {
        "new_selected_evaluation_rows.csv": new_selected,
        "new_update96_evaluation_rows.csv": new_update96,
        "historical_selected_evaluation_rows.csv": historical_selected,
        "historical_update96_evaluation_rows.csv": historical_fixed,
        "reused_rule_evaluation_rows.csv": historical_rules,
        "new_selected_behavior_ledger.csv": new_selected_ledger,
        "new_update96_behavior_ledger.csv": new_update96_ledger,
        "historical_update96_behavior_ledger.csv": historical_fixed_ledger,
    }.items():
        _write_csv(output_root / name, rows)
    all_rows = historical_selected + historical_fixed + new_selected + new_update96 + historical_rules
    identity_keys = {
        (row["observation_arm"], row["checkpoint_view"], row["method"], str(row["seed"]), row["split"], row["design_id"])
        for row in all_rows
    }
    if len(identity_keys) != len(all_rows):
        raise RuntimeError("duplicate matched evaluation identity")
    if tuple(map(len, (historical_selected, historical_fixed, new_selected, new_update96, historical_rules))) != (400, 400, 400, 400, 40):
        raise RuntimeError("matched evaluation row budget drift")
    _write_csv(output_root / "evaluation_rows.csv", all_rows)
    _write_json(output_root / "training_summary.json", [cell["summary"] for cell in cells])
    _write_json(output_root / "update_records.json", [row for cell in cells for row in cell["updates"]])
    _write_csv(output_root / "optimizer_step_records.csv", [row for cell in cells for row in cell["optimizer_rows"]])
    _write_json(output_root / "checkpoint_selection.json", [row for cell in cells for row in cell["candidates"]])
    _write_json(output_root / "training_episode_rows.json", [row for cell in cells for row in cell["training_episodes"]])
    _write_csv(output_root / "training_signal_rows.csv", [row for cell in cells for row in cell["training_signals"]])
    _write_json(output_root / "preflight_receipt.json", receipt)
    parameter_counts = _parameter_receipt(config)
    run_manifest = {
        "schema_version": "calibrated_workflow_prepared_state_visibility_matched_run_v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(),
        "protocol": {"path": str(protocol_path.relative_to(ROOT_DIR)), "sha256": _sha256(protocol_path)},
        "implementation_commit": protocol["intervention"]["implementation_commit"],
        "interface_profile": NEW_PROFILE, "historical_interface_profile": OLD_PROFILE,
        "source_interval_validation": identity, "parameter_counts": parameter_counts,
        "training_budget": training, "selection_score": protocol["frozen_scientific_identity"]["selection_score_implementation"],
        "historical_source": str(historical_root), "historical_selected_source_sha256": protocol["historical_arm"]["evaluation_rows_sha256"],
        "historical_rule_source": str(rule_root),
        "historical_rule_source_sha256": protocol["historical_arm"]["rule_source_evaluation_rows_sha256"],
        "evaluation_views": {"primary": "selected", "secondary": "update96"},
        "new_evaluation_episode_count": 1200, "historical_reused_episode_count": 440,
        "formal_or_holdout_reads": 0, "command": command,
        "claim_boundary": protocol["claim_boundary"],
    }
    _write_json(output_root / "run_manifest.json", run_manifest)
    _write_json(output_root / "completion_receipt.json", {
        "status": "complete", "learned_cells": len(cells),
        "learned_environment_steps": sum(row["summary"]["environment_steps"] for row in cells),
        "learned_optimizer_steps": sum(row["summary"]["optimizer_steps"] for row in cells),
        "new_evaluation_episodes": 1200, "historical_selected_reused_episodes": 400,
        "historical_rule_reused_episodes": 40, "formal_or_holdout_reads": 0,
        "automatic_retry": False, "scientific_execution_complete": True,
    })
    _integrity(output_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--historical-root", default=str(DEFAULT_HISTORICAL_ROOT))
    parser.add_argument("--rule-root", default=str(DEFAULT_RULE_ROOT))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-root")
    parser.add_argument("--expected-git-commit")
    args = parser.parse_args()
    if args.preflight == args.run:
        raise RuntimeError("select exactly one of --preflight or --run")
    protocol_path = (ROOT_DIR / args.protocol).resolve()
    historical_root = Path(args.historical_root).resolve()
    rule_root = Path(args.rule_root).resolve()
    if args.preflight:
        print(json.dumps(preflight(protocol_path, historical_root, rule_root), ensure_ascii=False, sort_keys=True))
        return
    if not args.output_root or not args.expected_git_commit:
        raise RuntimeError("run requires output root and expected git commit")
    if args.expected_git_commit != _git_commit():
        raise RuntimeError("frozen git commit mismatch")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT_DIR, check=True, capture_output=True, text=True)
    if status.stdout.strip():
        raise RuntimeError("scientific run requires a clean checkout")
    execute(protocol_path, historical_root, rule_root, Path(args.output_root).resolve(), sys.argv)


if __name__ == "__main__":
    main()
