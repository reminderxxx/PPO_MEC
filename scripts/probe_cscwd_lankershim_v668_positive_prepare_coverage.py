"""One-shot actual-consumer positive-prepare coverage probe on frozen v668."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_cscwd_new_development_reachability import load_traces  # noqa: E402
from scripts.freeze_calibrated_continuous_workflow_pilot import (  # noqa: E402
    _load_experiment_config,
)
from scripts.probe_cscwd_lankershim_event_label_coverage import (  # noqa: E402
    _event_label_diagnostics,
    _select_requested_action,
)
from scripts.run_cscwd_phase_v2_causal_public_pilot import (  # noqa: E402
    _build_cell_agent,
    _phase_env,
)


DEFAULT_PROTOCOL = ROOT / "configs/experiment/cscwd_lankershim_v668_positive_prepare_coverage_probe_v1.json"
DEFAULT_OUTPUT = ROOT / "artifacts/analysis/cscwd_lankershim_v668_positive_prepare_coverage_probe_20261011_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_clean() -> bool:
    return not subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "cscwd_lankershim_v668_positive_prepare_coverage_probe_v1",
        "execution_authorized": True,
        "training_window_id": "cscwd_newdev_lankershim_v668_t1118936135400_1118936254200",
        "condition_id": "default",
        "workload_design_id": "dev_01",
        "seed": 7,
        "environment_step_budget": 192,
        "episode_step_cap": 24,
        "desired_action_cycle": [0, 1, 2, 3, 4],
        "minimum_base_positive_prepare_count": 1,
        "minimum_candidate_prepare_to_abstain_count": 1,
        "selection_and_check_rows_read": False,
        "selection_is_result_blind": True,
        "training_enabled": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "checkpoint_reads": 0,
        "checkpoint_writes": 0,
        "formal_or_holdout_reads": 0,
        "automatic_retry": False,
        "on_gate_failure": "stop_without_training",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise RuntimeError(f"coverage protocol drift: {key}")
    if protocol.get("authorization_scope") not in {
        "one_create_only_nontraining_probe_no_retry",
        "one_create_only_engineering_correction_probe_no_retry",
    }:
        raise RuntimeError("coverage protocol drift: authorization_scope")
    split_path = ROOT / protocol["development_split"]
    source_protocol_path = ROOT / protocol["source_protocol"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    parent_path = Path(split["parent_metadata_candidate_path"])
    if _sha256(parent_path) != protocol["parent_metadata_candidate_sha256"]:
        raise RuntimeError("parent metadata candidate identity drift")
    if _sha256(source_protocol_path) != protocol["source_protocol_sha256"]:
        raise RuntimeError("source protocol identity drift")
    paths = {
        "raw_csv": Path(split["raw_csv_path"]),
        "base_config": ROOT / source_protocol["inputs"]["base_config"],
        "workload_manifest": ROOT / source_protocol["inputs"]["workload_manifest"],
        "phase_contract": ROOT / source_protocol["inputs"]["phase_contract"],
        "base_environment": ROOT / "src/envs/core/calibrated_continuous_workflow_env.py",
        "phase_v2_environment": ROOT / "src/envs/core/raw_ngsim_phase_v2_env.py",
        "public_estimator": ROOT / "src/agents/causal_public_action_estimator.py",
        "sa_core": ROOT / "src/agents/sa_ghmappo_core.py",
        "label_diagnostic_helper": ROOT / "scripts/probe_cscwd_lankershim_event_label_coverage.py",
        "probe_runner": Path(__file__).resolve(),
    }
    for name, expected_hash in protocol["input_sha256"].items():
        if not paths[name].is_file() or _sha256(paths[name]) != expected_hash:
            raise RuntimeError(f"coverage input identity drift: {name}")
    roles = split["selected_windows"]
    if [row["role"] for row in roles] != ["train", "selection", "check"]:
        raise RuntimeError("development role order drift")
    train = roles[0]
    if train["window_id"] != protocol["training_window_id"]:
        raise RuntimeError("training window drift")
    for left, right in zip(roles, roles[1:]):
        if int(right["source_time_start"]) - int(left["source_time_end"]) <= 2500:
            raise RuntimeError("development intervals are not separated")
    return protocol, split, source_protocol


def _gate(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    base_positive = sum(
        row["base_decision"] == "prepare"
        and row["base_supervision_weight"] > 0
        for row in rows
    )
    intervention = sum(
        row["base_decision"] == "prepare"
        and row["base_supervision_weight"] > 0
        and row["candidate_decision"] == "abstain"
        and row["candidate_supervision_weight"] == 0
        for row in rows
    )
    mismatches = sum(
        not row["base_actual_consumer_match"]
        or not row["candidate_actual_consumer_match"]
        for row in rows
    )
    passed = (
        mismatches == 0
        and base_positive >= protocol["minimum_base_positive_prepare_count"]
        and intervention >= protocol["minimum_candidate_prepare_to_abstain_count"]
    )
    return {
        "coverage_gate_passed": passed,
        "base_positive_prepare_count": base_positive,
        "candidate_prepare_to_abstain_count": intervention,
        "actual_training_consumer_mismatch_count": mismatches,
        "matched_training_may_be_frozen": passed,
        "training_started": False,
    }


def run(protocol_path: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"create-only output exists: {output_root}")
    if not _git_clean():
        raise RuntimeError("coverage checkout must be clean")
    protocol, split, source_protocol = _load_protocol(protocol_path)
    train = split["selected_windows"][0]
    traces, trace_identity = load_traces(Path(split["raw_csv_path"]), [train])
    config, _ = _load_experiment_config(
        ROOT / source_protocol["inputs"]["base_config"]
    )
    manifest = json.loads(
        (ROOT / source_protocol["inputs"]["workload_manifest"]).read_text()
    )
    instance = next(
        deepcopy(row) for row in manifest["instances"]
        if row["design_id"] == protocol["workload_design_id"]
    )
    phase = json.loads(
        (ROOT / source_protocol["inputs"]["phase_contract"]).read_text()
    )
    sa_cells = [
        row for row in source_protocol["training"]["learned_cells"]
        if row["method"] == "sa_ghmappo"
    ]
    torch.manual_seed(protocol["seed"])
    control = _build_cell_agent(sa_cells[0], protocol["seed"], config)
    torch.manual_seed(protocol["seed"])
    candidate = _build_cell_agent(sa_cells[1], protocol["seed"], config)
    random.seed(protocol["seed"])
    np.random.seed(protocol["seed"])
    torch.manual_seed(protocol["seed"])
    trace = traces[train["window_id"]]
    env_factory = lambda: _phase_env(
        config, instance, trace, phase, protocol["condition_id"]
    )
    env = env_factory()
    _, info = env.reset()
    episode_index = 1
    pseudo = {"event_target": 0, "event_soft_target": 0.0}
    rows = []
    states = []
    for global_step in range(1, protocol["environment_step_budget"] + 1):
        mask = [bool(value) for value in info["action_mask"]]
        state = deepcopy(info["semantic_state"])
        desired = protocol["desired_action_cycle"][(global_step - 1) % 5]
        requested, fallback = _select_requested_action(desired, mask)
        base = _event_label_diagnostics(state, mask)
        gated = _event_label_diagnostics(
            state, mask,
            positive_prepare_calibration_abstention_enabled=True,
        )
        base_consumer = control._event_auxiliary_target(state, pseudo)
        candidate_consumer = candidate._event_auxiliary_target(state, pseudo)
        base_match = all(base_consumer[key] == base[value] for key, value in (
            ("decision", "event_label"), ("reason", "event_label_reason"),
            ("supervision_weight", "event_supervision_weight"),
        ))
        candidate_match = all(candidate_consumer[key] == gated[value] for key, value in (
            ("decision", "event_label"), ("reason", "event_label_reason"),
            ("supervision_weight", "event_supervision_weight"),
        ))
        states.append({
            "global_step": global_step,
            "episode_index": episode_index,
            "semantic_state_sha256": base["semantic_state_sha256"],
            "action_mask": mask,
            "semantic_state": state,
        })
        _, reward, terminated, truncated, next_info = env.step(requested)
        transition = dict(next_info.get("transition", {}) or {})
        executed = int(transition.get("action", requested))
        if executed != requested:
            raise RuntimeError("coverage action was rewritten")
        rows.append({
            "global_step": global_step,
            "episode_index": episode_index,
            "semantic_state_sha256": base["semantic_state_sha256"],
            "desired_action": desired,
            "executed_action": executed,
            "masked_fallback_used": fallback,
            "reward_observed_not_used_for_selection": float(reward),
            "base_decision": base["event_label"],
            "base_reason": base["event_label_reason"],
            "base_supervision_weight": base["event_supervision_weight"],
            "candidate_decision": gated["event_label"],
            "candidate_reason": gated["event_label_reason"],
            "candidate_supervision_weight": gated["event_supervision_weight"],
            "base_actual_consumer_match": base_match,
            "candidate_actual_consumer_match": candidate_match,
            "action4_legal": base["action4_legal"],
            "action4_current_service": base["action4_current_service"],
            "action4_target_prepare": base["action4_target_prepare"],
            "action4_deadline_fit": base["action4_deadline_fit"],
            "terminated": bool(terminated),
            "environment_truncated": bool(truncated),
        })
        outer_cap = (
            not terminated and not truncated
            and int(env.step_index) >= protocol["episode_step_cap"]
        )
        if terminated or truncated or outer_cap:
            env = env_factory()
            _, info = env.reset()
            episode_index += 1
        else:
            info = next_info
    gate = _gate(rows, protocol)
    summary = {
        "schema_version": "cscwd_lankershim_v668_positive_prepare_coverage_result_v1",
        "status": "PASS",
        "git_commit": _git_commit(),
        "protocol_sha256": _sha256(protocol_path),
        "development_split_sha256": _sha256(ROOT / protocol["development_split"]),
        "parent_metadata_candidate_sha256": protocol["parent_metadata_candidate_sha256"],
        "training_window_id": train["window_id"],
        "trace_identity": trace_identity,
        "historical_exposure_status": "development_unknown_not_independent_generalization",
        "environment_steps": len(rows),
        "new_environment_steps": len(rows),
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "checkpoint_reads": 0,
        "formal_or_holdout_reads": 0,
        "selection_and_check_rows_read": False,
        "base_decision_counts": dict(sorted(Counter(
            row["base_decision"] for row in rows
        ).items())),
        "base_reason_counts": dict(sorted(Counter(
            row["base_reason"] for row in rows
        ).items())),
        "executed_action_counts": dict(sorted(Counter(
            str(row["executed_action"]) for row in rows
        ).items())),
        "unique_semantic_state_count": len({
            row["semantic_state_sha256"] for row in rows
        }),
        **gate,
    }
    output_root.mkdir(parents=True)
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "consumer_rows.json", rows)
    with (output_root / "semantic_states.jsonl").open("w", encoding="utf-8") as handle:
        for row in states:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    files = []
    for path in sorted(output_root.iterdir()):
        if path.name != "artifact_integrity.json":
            files.append({
                "path": path.name,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            })
    _write_json(output_root / "artifact_integrity.json", {
        "schema_version": "cscwd_lankershim_v668_positive_prepare_coverage_integrity_v1",
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
