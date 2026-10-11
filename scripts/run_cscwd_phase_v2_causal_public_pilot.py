"""Run the one-shot phase-v2 public-causal development pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import random
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_cscwd_new_development_reachability import load_traces  # noqa: E402
from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config  # noqa: E402
from scripts.run_calibrated_workflow_interface_repair import (  # noqa: E402
    _build_agent, _git_commit, _integrity, _sha256, _write_csv, _write_json,
)
from scripts.launch_calibrated_workflow_value_normalization_ab import (  # noqa: E402
    SCIENTIFIC_COMPLETION_RECEIPT_SCHEMA_VERSION,
)
from src.agents.causal_public_action_estimator import (  # noqa: E402
    CALIBRATION_ABSTENTION_EVENT_LABEL_SCHEMA_VERSION,
    CausalPublicImmediateRule, CausalPublicTwoStepRule,
    SCHEMA_VERSION as ESTIMATOR_SCHEMA_VERSION,
    EVENT_LABEL_SCHEMA_VERSION, PUBLIC_RULE_SCHEMA_VERSION,
    estimate_public_actions, public_prepare_advantage_label,
)
from src.envs.core.calibrated_continuous_workflow_env import TwoStepCostRule  # noqa: E402
from src.envs.core.raw_ngsim_event_time_env import RawNGSIMEventTimeEnv  # noqa: E402
from src.envs.core.raw_ngsim_phase_v2_env import RawNGSIMPhaseV2Env  # noqa: E402
from src.trainers.ppo_buffer import PPORolloutBuffer  # noqa: E402


DEFAULT_PROTOCOL = "configs/experiment/cscwd_phase_v2_causal_public_pilot_v1.json"
LANKERSHIM_POSTFIX_SCHEMA = (
    "cscwd_phase_v2_recompute_observability_calibration_abstention_ab_v1"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key, "")) for key in fields})


def _git_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    return not result.stdout.strip()


def _input_paths(protocol: dict[str, Any]) -> dict[str, Path]:
    values = protocol["inputs"]
    result = {}
    keys = (
        (
            "base_config", "workload_manifest", "phase_contract", "raw_csv_path",
            "development_split", "coverage_summary", "coverage_integrity",
            "coverage_semantic_states",
        )
        if protocol.get("schema_version") == LANKERSHIM_POSTFIX_SCHEMA
        else (
            "base_config", "workload_manifest", "phase_contract", "raw_csv_path",
            "source_manifest", "phase_v2_preflight", "public_fixture",
            "mechanism_matrix_manifest", "mechanism_matrix_ledger",
            "mechanism_matrix_audit",
        )
    )
    for key in keys:
        path = Path(values[key])
        result[key] = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    return result


def _validate_lankershim_postfix_protocol(
    protocol: dict[str, Any], *, require_clean: bool, require_authorized: bool,
) -> dict[str, Any]:
    if require_authorized and protocol.get("execution_authorized") is not True:
        raise RuntimeError("pilot is not authorized")
    expected_single = {
        "name": "mechanism_aux_positive_prepare_calibration_abstention_enabled",
        "control": False,
        "candidate": True,
        "base_event_semantics": EVENT_LABEL_SCHEMA_VERSION,
        "candidate_semantics": CALIBRATION_ABSTENTION_EVENT_LABEL_SCHEMA_VERSION,
        "sa_only": True,
        "reward_change": False,
        "action_permission_change": False,
        "network_parameter_change": 0,
    }
    if protocol.get("single_variable") != expected_single:
        raise RuntimeError("single-variable contract drift")
    training = protocol["training"]
    if (
        training["seeds"] != [7, 17, 29, 43, 61]
        or int(training["environment_steps_per_cell"]) != 1152
        or int(training["transitions_per_update"]) != 48
        or int(training["update_opportunities_per_cell"]) != 24
        or int(training["optimizer_steps_per_update"]) != 8
        or int(training["optimizer_steps_per_cell"]) != 192
        or training["checkpoint_update_candidates"] != [6, 12, 18, 24]
        or int(training["total_environment_steps"]) != 23040
        or int(training["total_optimizer_steps"]) != 3840
        or training["automatic_retry"]
        or int(training["scientific_launches"]) != 1
    ):
        raise RuntimeError("training budget drift")
    actual_cells = [
        (
            row["arm"], row["method"], bool(row["candidate_enabled"]),
            bool(row.get("positive_prepare_calibration_abstention_enabled", False)),
        )
        for row in training["learned_cells"]
    ]
    if actual_cells != [
        ("sa_public_v4", "sa_ghmappo", True, False),
        ("sa_calibration_abstention_v2", "sa_ghmappo", True, True),
        ("ppo", "ppo", False, False),
        ("mappo", "mappo", False, False),
    ]:
        raise RuntimeError("learned cell identity drift")
    if (
        protocol["objective"]["reward_profile"] != "original_reward_v1"
        or protocol["objective"]["critic_target_normalization"] != "raw_disabled"
    ):
        raise RuntimeError("objective identity drift")

    paths = _input_paths(protocol)
    hashes = protocol["inputs"]["sha256"]
    checks = {
        "workload_manifest": paths["workload_manifest"],
        "phase_contract": paths["phase_contract"],
        "raw_csv": paths["raw_csv_path"],
        "development_split": paths["development_split"],
        "coverage_summary": paths["coverage_summary"],
        "coverage_integrity": paths["coverage_integrity"],
        "coverage_semantic_states": paths["coverage_semantic_states"],
        "phase_v2_environment": ROOT / "src/envs/core/raw_ngsim_phase_v2_env.py",
        "calibrated_environment": (
            ROOT / "src/envs/core/calibrated_continuous_workflow_env.py"
        ),
        "public_estimator": ROOT / "src/agents/causal_public_action_estimator.py",
        "raw_event_time_environment": ROOT / "src/envs/core/raw_ngsim_event_time_env.py",
        "sa_core": ROOT / "src/agents/sa_ghmappo_core.py",
        "agent_builder": ROOT / "scripts/run_calibrated_workflow_interface_repair.py",
        "pilot_runner": Path(__file__).resolve(),
        "ab_analysis": (
            ROOT / "scripts/analyze_cscwd_positive_prepare_calibration_abstention_ab.py"
        ),
        "ab_launcher": (
            ROOT / "scripts/launch_cscwd_lankershim_recompute_observability_ab.py"
        ),
    }
    for name, path in checks.items():
        if not path.is_file() or _sha256(path) != hashes.get(name):
            raise RuntimeError(f"frozen input identity mismatch: {name}")

    split = _load(paths["development_split"])
    phase = _load(paths["phase_contract"])
    coverage = _load(paths["coverage_summary"])
    integrity = _load(paths["coverage_integrity"])
    windows = split.get("selected_windows", [])
    if (
        split.get("schema_version")
        != "cscwd_lankershim_three_role_development_split_v1"
        or [row.get("role") for row in windows] != ["train", "selection", "check"]
        or any(row.get("source_segment_id") != "lankershim" for row in windows)
        or phase.get("time_profile") != "raw_ngsim_event_time_v2"
        or phase.get("source_manifest_sha256") != hashes["development_split"]
        or coverage.get("status") != "PASS"
        or coverage.get("coverage_gate_passed") is not True
        or coverage.get("base_positive_prepare_count") != 46
        or coverage.get("candidate_prepare_to_abstain_count") != 46
        or coverage.get("actual_training_consumer_mismatch_count") != 0
        or coverage.get("selection_and_check_rows_read") is not False
        or coverage.get("optimizer_steps") != 0
        or coverage.get("formal_or_holdout_reads") != 0
    ):
        raise RuntimeError("post-fix coverage or three-role split gate failed")
    integrity_rows = {row["path"]: row["sha256"] for row in integrity["files"]}
    if integrity_rows.get("summary.json") != hashes["coverage_summary"] or (
        integrity_rows.get("semantic_states.jsonl")
        != hashes["coverage_semantic_states"]
    ):
        raise RuntimeError("coverage artifact internal identity mismatch")
    intervals = sorted(
        (int(row["source_time_start"]), int(row["source_time_end"]))
        for row in windows
    )
    if any(left[1] >= right[0] for left, right in zip(intervals, intervals[1:])):
        raise RuntimeError("three-role windows are not time-disjoint")
    if require_clean and not _git_clean():
        raise RuntimeError("scientific checkout must be clean")
    return {
        "status": "PASS",
        "git_commit": _git_commit(),
        "git_clean": _git_clean(),
        "input_sha256": hashes,
        "estimator_schema": ESTIMATOR_SCHEMA_VERSION,
        "event_schema": EVENT_LABEL_SCHEMA_VERSION,
        "calibration_abstention_event_schema": (
            CALIBRATION_ABSTENTION_EVENT_LABEL_SCHEMA_VERSION
        ),
        "rule_schema": PUBLIC_RULE_SCHEMA_VERSION,
        "coverage_gate": {
            "status": "PASS",
            "steps": 192,
            "base_positive_prepare_count": 46,
            "candidate_prepare_to_abstain_count": 46,
            "consumer_mismatch_count": 0,
        },
        "development_identity": (
            "same_lankershim_recording_three_time_disjoint_windows_"
            "historical_check_exposure_unknown"
        ),
    }


def _validate_protocol(
    protocol: dict[str, Any], *, require_clean: bool, require_authorized: bool = True,
) -> dict[str, Any]:
    schema_version = protocol.get("schema_version")
    if schema_version == LANKERSHIM_POSTFIX_SCHEMA:
        return _validate_lankershim_postfix_protocol(
            protocol,
            require_clean=require_clean,
            require_authorized=require_authorized,
        )
    if schema_version not in {
        "cscwd_phase_v2_causal_public_pilot_v1",
        "cscwd_phase_v2_positive_prepare_calibration_abstention_ab_v1",
    }:
        raise RuntimeError("pilot protocol schema drift")
    if require_authorized and not protocol.get("execution_authorized"):
        raise RuntimeError("pilot is not authorized")
    single = protocol["single_variable"]
    legacy = schema_version == "cscwd_phase_v2_causal_public_pilot_v1"
    expected_single = (
        {
            "name": "mechanism_aux_causal_public_prepare_advantage_enabled",
            "control": False,
            "candidate": True,
            "candidate_semantics": "causal_public_prepare_serve_abstain_event_supervision_v3",
            "sa_only": True,
            "reward_change": False,
            "action_permission_change": False,
            "network_parameter_change": 0,
        }
        if legacy
        else {
            "name": "mechanism_aux_positive_prepare_calibration_abstention_enabled",
            "control": False,
            "candidate": True,
            "base_event_semantics": EVENT_LABEL_SCHEMA_VERSION,
            "candidate_semantics": CALIBRATION_ABSTENTION_EVENT_LABEL_SCHEMA_VERSION,
            "sa_only": True,
            "reward_change": False,
            "action_permission_change": False,
            "network_parameter_change": 0,
        }
    )
    if single != expected_single:
        raise RuntimeError("single-variable contract drift")
    training = protocol["training"]
    if (
        training["seeds"] != [7, 17, 29, 43, 61]
        or int(training["environment_steps_per_cell"]) != 1152
        or int(training["transitions_per_update"]) != 48
        or int(training["update_opportunities_per_cell"]) != 24
        or int(training["optimizer_steps_per_update"]) != 8
        or int(training["optimizer_steps_per_cell"]) != 192
        or training["checkpoint_update_candidates"] != [6, 12, 18, 24]
        or int(training["total_environment_steps"]) != 23040
        or int(training["total_optimizer_steps"]) != 3840
        or training["automatic_retry"]
        or int(training["scientific_launches"]) != 1
    ):
        raise RuntimeError("training budget drift")
    expected_cells = (
        [
            ("sa_control", "sa_ghmappo", False, False),
            ("sa_candidate", "sa_ghmappo", True, False),
            ("ppo", "ppo", False, False),
            ("mappo", "mappo", False, False),
        ]
        if legacy
        else [
            ("sa_public_v3", "sa_ghmappo", True, False),
            ("sa_calibration_abstention", "sa_ghmappo", True, True),
            ("ppo", "ppo", False, False),
            ("mappo", "mappo", False, False),
        ]
    )
    actual_cells = [
        (
            row["arm"], row["method"], bool(row["candidate_enabled"]),
            bool(row.get("positive_prepare_calibration_abstention_enabled", False)),
        )
        for row in training["learned_cells"]
    ]
    if actual_cells != expected_cells:
        raise RuntimeError("learned cell identity drift")
    if protocol["objective"]["reward_profile"] != "original_reward_v1":
        raise RuntimeError("reward profile drift")
    if protocol["objective"]["critic_target_normalization"] != "raw_disabled":
        raise RuntimeError("critic normalization drift")
    paths = _input_paths(protocol)
    expected_hashes = protocol["inputs"]["sha256"]
    checks = {
        "workload_manifest": paths["workload_manifest"],
        "phase_contract": paths["phase_contract"],
        "raw_csv": paths["raw_csv_path"],
        "source_manifest": paths["source_manifest"],
        "phase_v2_preflight": paths["phase_v2_preflight"],
        "public_fixture": paths["public_fixture"],
        "mechanism_matrix_manifest": paths["mechanism_matrix_manifest"],
        "mechanism_matrix_ledger": paths["mechanism_matrix_ledger"],
        "mechanism_matrix_audit": paths["mechanism_matrix_audit"],
        "phase_v2_environment": ROOT / "src/envs/core/raw_ngsim_phase_v2_env.py",
        "public_estimator": ROOT / "src/agents/causal_public_action_estimator.py",
    }
    if not legacy:
        checks.update({
            "raw_event_time_environment": (
                ROOT / "src/envs/core/raw_ngsim_event_time_env.py"
            ),
            "sa_core": ROOT / "src/agents/sa_ghmappo_core.py",
            "agent_builder": (
                ROOT / "scripts/run_calibrated_workflow_interface_repair.py"
            ),
            "pilot_runner": Path(__file__).resolve(),
            "ab_analysis": (
                ROOT
                / "scripts/analyze_cscwd_positive_prepare_calibration_abstention_ab.py"
            ),
            "ab_launcher": (
                ROOT
                / "scripts/launch_cscwd_positive_prepare_calibration_abstention_ab.py"
            ),
        })
    for name, path in checks.items():
        if not path.is_file() or _sha256(path) != expected_hashes[name]:
            raise RuntimeError(f"frozen input identity mismatch: {name}")
    phase = _load(paths["phase_contract"])
    source = _load(paths["source_manifest"])
    preflight = _load(paths["phase_v2_preflight"])
    fixture = _load(paths["public_fixture"])
    matrix = _load(paths["mechanism_matrix_manifest"])
    matrix_audit = _load(paths["mechanism_matrix_audit"])
    if (
        phase["time_profile"] != "raw_ngsim_event_time_v2"
        or phase["source_manifest_sha256"] != expected_hashes["source_manifest"]
        or preflight.get("episode_count") != 2
        or preflight.get("real_step_count") != 2
        or fixture.get("schema_version") != "cscwd_raw_phase_v2_public_fixture_v1"
        or len(source.get("selected_windows", [])) != 2
        or matrix.get("schema_version") != "cscwd_raw_phase_v2_mechanism_matrix_v1"
        or matrix.get("episode_count") != 60
        or matrix.get("real_step_count") != 522
        or matrix.get("preview_count_upper_bound") != 731
        or matrix.get("training_steps") != 0
        or matrix_audit.get("all_checks_passed") is not True
        or matrix_audit.get("episode_count") != 60
        or matrix_audit.get("real_step_count") != 522
        or matrix_audit.get("matrix_manifest_sha256") != expected_hashes["mechanism_matrix_manifest"]
        or matrix_audit.get("step_ledger_sha256") != expected_hashes["mechanism_matrix_ledger"]
        or matrix_audit.get("formal_or_holdout_results_read") is not False
    ):
        raise RuntimeError("phase-v2 acceptance or mechanism-matrix semantics drift")
    public_info = fixture["info"]
    canonical = __import__("hashlib").sha256(
        json.dumps(public_info, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    if canonical != fixture["public_info_sha256"]:
        raise RuntimeError("public fixture embedded identity mismatch")
    estimate = estimate_public_actions(
        public_info["semantic_state"], public_info["action_mask"]
    )
    label = public_prepare_advantage_label(
        public_info["semantic_state"], public_info["action_mask"]
    )
    gated_label = public_prepare_advantage_label(
        public_info["semantic_state"], public_info["action_mask"],
        positive_prepare_calibration_abstention_enabled=True,
    )
    immediate = CausalPublicImmediateRule().select_action_from_info(public_info)
    two_step = CausalPublicTwoStepRule().select_action_from_info(public_info)
    if (
        estimate["schema_version"] != ESTIMATOR_SCHEMA_VERSION
        or label["schema_version"] != EVENT_LABEL_SCHEMA_VERSION
        or immediate != 4 or two_step != 4
        or label["decision"] != "prepare"
        or gated_label["decision"] != "abstain"
    ):
        raise RuntimeError("public consumer fixture gate failed")
    if require_clean and not _git_clean():
        raise RuntimeError("scientific checkout must be clean")
    return {
        "status": "PASS",
        "git_commit": _git_commit(),
        "git_clean": _git_clean(),
        "input_sha256": expected_hashes,
        "estimator_schema": ESTIMATOR_SCHEMA_VERSION,
        "event_schema": EVENT_LABEL_SCHEMA_VERSION,
        "calibration_abstention_event_schema": (
            CALIBRATION_ABSTENTION_EVENT_LABEL_SCHEMA_VERSION
        ),
        "rule_schema": PUBLIC_RULE_SCHEMA_VERSION,
        "fixture_label": label["decision"],
        "fixture_calibration_abstention_label": gated_label["decision"],
        "fixture_actions": {"immediate": immediate, "two_step": two_step},
        "phase_v2_two_step_capability": "same_public_immediate_fallback_not_independent_baseline",
        "mechanism_matrix_gate": {
            "status": "PASS", "episodes": 60, "real_steps": 522,
            "preview_count_upper_bound": 731,
            "independent_audit_sha256": expected_hashes["mechanism_matrix_audit"],
        },
    }


def _load_runtime(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = _input_paths(protocol)
    config, _ = _load_experiment_config(paths["base_config"])
    config["reward_profile"] = "original_reward_v1"
    config["interface_profile"] = "calibrated_workflow_interface_v4_prepared_state_prefix"
    manifest = _load(paths["workload_manifest"])
    instance = next(
        deepcopy(row) for row in manifest["instances"]
        if row["design_id"] == protocol["data_assignment"]["workload_design_id"]
    )
    phase = _load(paths["phase_contract"])
    new_split = protocol.get("schema_version") == LANKERSHIM_POSTFIX_SCHEMA
    source = _load(paths["development_split"] if new_split else paths["source_manifest"])
    traces, trace_identity = load_traces(paths["raw_csv_path"], source["selected_windows"])
    identity_key = "role" if new_split else "source_segment_id"
    by_segment = {
        row[identity_key]: traces[row["window_id"]]
        for row in source["selected_windows"]
    }
    source_by_segment = {
        row[identity_key]: row for row in source["selected_windows"]
    }
    expected = {"train", "selection", "check"} if new_split else {"lankershim", "us_101"}
    if set(by_segment) != expected:
        raise RuntimeError("frozen source windows are incomplete")
    return config, instance, {"traces": by_segment, "sources": source_by_segment}, {
        "phase": phase, "trace_identity": trace_identity,
    }


def _phase_env(
    config: dict[str, Any], instance: dict[str, Any], trace: Any,
    phase: dict[str, Any], condition_id: str,
) -> RawNGSIMPhaseV2Env:
    condition = next(row for row in phase["conditions"] if row["condition_id"] == condition_id)
    return RawNGSIMPhaseV2Env(
        config, deepcopy(instance), trace,
        phase_contract=phase["physical_assumptions"], condition=condition,
    )


def _actual_class_gate(
    protocol: dict[str, Any], config: dict[str, Any], instance: dict[str, Any],
    source_bundle: dict[str, Any], runtime: dict[str, Any],
) -> dict[str, Any]:
    paths = _input_paths(protocol)
    if protocol.get("schema_version") == LANKERSHIM_POSTFIX_SCHEMA:
        witness = None
        with paths["coverage_semantic_states"].open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                base = public_prepare_advantage_label(
                    row["semantic_state"], row["action_mask"]
                )
                if base["decision"] == "prepare":
                    witness = row
                    break
        if witness is None:
            raise RuntimeError("coverage artifact has no positive prepare witness")
        label = public_prepare_advantage_label(
            witness["semantic_state"], witness["action_mask"]
        )
        gated_label = public_prepare_advantage_label(
            witness["semantic_state"], witness["action_mask"],
            positive_prepare_calibration_abstention_enabled=True,
        )
        if label["decision"] != "prepare" or gated_label["decision"] != "abstain":
            raise RuntimeError("post-fix intervention witness gate failed")
        sa_cells = [
            row for row in protocol["training"]["learned_cells"]
            if row["method"] == "sa_ghmappo"
        ]
        torch.manual_seed(7)
        control = _build_cell_agent(sa_cells[0], 7, config)
        torch.manual_seed(7)
        candidate = _build_cell_agent(sa_cells[1], 7, config)
        control_state = control._network.state_dict()
        candidate_state = candidate._network.state_dict()
        tensors_equal = control_state.keys() == candidate_state.keys() and all(
            torch.equal(control_state[key], candidate_state[key])
            for key in control_state
        )
        if not tensors_equal:
            raise RuntimeError("SA control/candidate initialized network identity drift")
        control_parameters = sum(int(item.numel()) for item in control._network.parameters())
        candidate_parameters = sum(int(item.numel()) for item in candidate._network.parameters())
        if control_parameters != candidate_parameters:
            raise RuntimeError("SA candidate changed network parameter count")
        env = _phase_env(
            config, instance, source_bundle["traces"]["train"],
            runtime["phase"], "default",
        )
        observation, info = env.reset()
        action, action_info = candidate.act(observation, deepcopy(info))
        if not info["action_mask"][int(action)] or not np.isfinite(float(action_info["log_prob"])):
            raise RuntimeError("candidate action likelihood gate failed")
        return {
            "status": "PASS",
            "witness_semantic_state_sha256": witness["semantic_state_sha256"],
            "base_event_label": label["decision"],
            "base_event_label_reason": label["reason"],
            "candidate_event_label": gated_label["decision"],
            "candidate_event_label_reason": gated_label["reason"],
            "candidate_action": int(action),
            "candidate_action_log_prob_finite": True,
            "sa_initialized_tensors_equal": True,
            "sa_control_parameter_count": control_parameters,
            "sa_candidate_parameter_count": candidate_parameters,
            "environment_steps": 0,
        }
    fixture = _load(paths["public_fixture"])
    env = _phase_env(
        config, instance, source_bundle["traces"]["us_101"], runtime["phase"], "default"
    )
    observation, info = env.reset()
    public_info = {key: info[key] for key in ("semantic_state", "action_mask", "run_metadata")}
    canonical = __import__("hashlib").sha256(
        json.dumps(public_info, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    if canonical != fixture["public_info_sha256"]:
        raise RuntimeError("actual phase-v2 reset differs from frozen public fixture")
    estimate = estimate_public_actions(info["semantic_state"], info["action_mask"])
    label = public_prepare_advantage_label(info["semantic_state"], info["action_mask"])
    gated_label = public_prepare_advantage_label(
        info["semantic_state"], info["action_mask"],
        positive_prepare_calibration_abstention_enabled=True,
    )
    sa_cells = [
        row for row in protocol["training"]["learned_cells"]
        if row["method"] == "sa_ghmappo"
    ]
    if len(sa_cells) != 2:
        raise RuntimeError("expected exactly two SA cells")
    torch.manual_seed(7)
    control = _build_cell_agent(sa_cells[0], 7, config)
    torch.manual_seed(7)
    candidate = _build_cell_agent(sa_cells[1], 7, config)
    control_state = control._network.state_dict()
    candidate_state = candidate._network.state_dict()
    if control_state.keys() != candidate_state.keys() or any(
        not torch.equal(control_state[key], candidate_state[key]) for key in control_state
    ):
        raise RuntimeError("SA control/candidate initialized network identity drift")
    control_parameters = sum(int(item.numel()) for item in control._network.parameters())
    candidate_parameters = sum(int(item.numel()) for item in candidate._network.parameters())
    if control_parameters != candidate_parameters:
        raise RuntimeError("SA candidate changed network parameter count")
    action, action_info = candidate.act(observation, deepcopy(info))
    if not info["action_mask"][int(action)] or not np.isfinite(float(action_info["log_prob"])):
        raise RuntimeError("candidate action likelihood gate failed")
    with np.testing.assert_raises_regex(RuntimeError, "hidden-future"):
        env.clone_for_decision_model()
    before = env.clock_seconds
    _, _, terminated, truncated, after = env.step(4)
    event = after["transition"]
    phase_sum = sum(float(row["elapsed_seconds"]) for row in event["phase_events"])
    if not np.isclose(phase_sum, float(event["step_cost_seconds"])):
        raise RuntimeError("actual phase-v2 elapsed conservation failed")
    if not np.isclose(env.clock_seconds - before, float(event["step_cost_seconds"])):
        raise RuntimeError("actual phase-v2 clock conservation failed")
    if int(event["backhaul_transfer_bytes"]) != int(event["model_transfer_bytes"]) + int(event["state_transfer_bytes"]):
        raise RuntimeError("actual phase-v2 byte conservation failed")
    return {
        "status": "PASS", "public_info_sha256": canonical,
        "estimator_schema": estimate["schema_version"], "event_label": label["decision"],
        "calibration_abstention_event_label": gated_label["decision"],
        "candidate_action": int(action), "candidate_action_log_prob_finite": True,
        "sa_initialized_tensors_equal": True,
        "sa_control_parameter_count": control_parameters,
        "sa_candidate_parameter_count": candidate_parameters,
        "action4_terminated": bool(terminated), "action4_truncated": bool(truncated),
        "action4_service_completed": bool(event["service_completed"]),
        "action4_migration_success": bool(event.get("migration_success", False)),
        "action4_step_cost_seconds": float(event["step_cost_seconds"]),
        "phase_elapsed_conserved": True, "backhaul_bytes_conserved": True,
        "hidden_future_clone_rejected": True,
    }


def _new_training_state(
    env_factory: Callable[[], Any], episode_index: int,
) -> dict[str, Any]:
    env = env_factory()
    observation, info = env.reset()
    return {
        "env": env, "observation": observation, "info": info,
        "episode_index": episode_index + 1, "segment_index": 0,
    }


def _collect_batch(
    *, agent: Any, env_factory: Callable[[], Any], state: dict[str, Any] | None,
    episodes_started: int, transition_count: int, step_cap: int,
    gamma: float, gae_lambda: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    buffer = PPORolloutBuffer()
    while len(rows) < transition_count:
        if state is None:
            state = _new_training_state(env_factory, episodes_started)
            episodes_started += 1
        env = state["env"]
        observation, info = state["observation"], state["info"]
        decision_info = deepcopy(info)
        action, action_info = agent.act(observation, decision_info)
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        executed = int((next_info.get("transition") or {}).get("action", action))
        if executed != int(action):
            raise RuntimeError("training action was rewritten after public masking")
        value = float(action_info.get("value", agent.evaluate_value(observation, decision_info)))
        buffer.add_step(
            observation=observation, action=executed, reward=reward,
            terminated=terminated, truncated=truncated,
            log_prob=float(action_info.get("log_prob", 0.0)), value=value,
            next_observation=next_observation, action_info=action_info,
            decision_info=decision_info, env_info=deepcopy(next_info),
        )
        state["observation"], state["info"] = next_observation, next_info
        boundary = len(rows) + len(buffer) == transition_count
        outer_cap = bool(not terminated and not truncated and int(env.step_index) >= step_cap)
        if terminated or truncated or outer_cap or boundary:
            last_value = 0.0 if terminated else float(agent.evaluate_value(next_observation, next_info))
            buffer.finalize(last_value=last_value, gamma=gamma, gae_lambda=gae_lambda)
            segment = buffer.to_training_rows()
            state["segment_index"] += 1
            for row in segment:
                row["training_episode_index"] = state["episode_index"]
                row["rollout_segment_index"] = state["segment_index"]
                row["bootstrap_value"] = last_value
                row["segment_terminated"] = bool(terminated)
                row["segment_truncated"] = bool(truncated or outer_cap)
                row["batch_boundary"] = bool(boundary and not terminated and not truncated and not outer_cap)
            rows.extend(segment)
            buffer = PPORolloutBuffer()
        if terminated or truncated or outer_cap:
            summary = env.summary()
            episodes.append({
                "training_episode_index": state["episode_index"],
                "steps": summary["steps"],
                "workflow_completed": bool(summary["workflow_completed"]),
                "on_time_workflow_completed": int(summary.get("on_time_workflow_completed", 0)),
                "service_failures": int(summary["service_failures"]),
                "completed_nodes": int(summary["completed_nodes"]),
                "node_count": int(summary["node_count"]),
                "reward": float(summary["reward"]),
                "terminated": bool(terminated), "truncated": bool(truncated or outer_cap),
                "runner_outer_cap": outer_cap,
            })
            state = None
    if len(rows) != transition_count:
        raise RuntimeError("exact transition batch invariant failed")
    return rows, state, episodes_started, episodes


def _decision(agent: Any | None, rule: Any | None, env: Any, observation: Any, info: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    if agent is not None:
        action, action_info = agent.act(observation, info)
        return int(action), dict(action_info)
    if isinstance(rule, TwoStepCostRule):
        return int(rule.select_action(env)), {"capability": "privileged_exact_v1_only"}
    return int(rule.select_action_from_info(info)), {
        "capability": rule.capability_profile,
        "rule_schema_version": rule.rule_schema_version,
        "objective_profile": rule.objective_profile,
    }


def _evaluate_episode(
    *, method: str, seed: int | str, arm: str, view: str,
    env_factory: Callable[[], Any], source: dict[str, Any], condition_id: str,
    time_profile: str, step_cap: int, agent: Any | None = None, rule: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = env_factory()
    observation, info = env.reset()
    previous_deterministic = None
    if agent is not None:
        previous_deterministic = bool(getattr(agent, "_deterministic_action", False))
        agent._deterministic_action = True
    counts: Counter[int] = Counter()
    phase_seconds: defaultdict[str, float] = defaultdict(float)
    ledger: list[dict[str, Any]] = []
    decision_ns = 0
    no_progress = max_no_progress = 0
    while not env.terminated and env.step_index < step_cap:
        before_completed = len(env.completed)
        started = time.perf_counter_ns()
        requested, action_info = _decision(agent, rule, env, observation, info)
        decision_ns += time.perf_counter_ns() - started
        mask = list(info["action_mask"])
        if not (0 <= requested < len(mask) and mask[requested]):
            raise RuntimeError(f"{method} selected masked action {requested}")
        decision_info = deepcopy(info)
        next_observation, reward, terminated, truncated, next_info = env.step(requested)
        event = dict(next_info.get("transition") or {})
        executed = int(event.get("action", requested))
        if executed != requested:
            raise RuntimeError("evaluation action was rewritten")
        counts[executed] += 1
        progressed = len(env.completed) > before_completed
        no_progress = 0 if progressed else no_progress + 1
        max_no_progress = max(max_no_progress, no_progress)
        phases = list(event.get("phase_events", []) or [])
        for phase_event in phases:
            phase_seconds[str(phase_event["phase"])] += float(phase_event["elapsed_seconds"])
        label = public_prepare_advantage_label(
            decision_info["semantic_state"], decision_info["action_mask"],
            positive_prepare_calibration_abstention_enabled=bool(
                getattr(
                    agent,
                    "_mechanism_aux_positive_prepare_calibration_abstention_enabled",
                    False,
                )
                if agent is not None
                else False
            ),
        )
        ledger.append({
            "arm": arm, "view": view, "method": method, "seed": seed,
            "time_profile": time_profile, "condition_id": condition_id,
            "source_segment_id": source["source_segment_id"],
            "window_id": source["window_id"], "step": len(ledger),
            "requested_action": requested, "executed_action": executed,
            "action_mask": json.dumps(mask),
            "action_probabilities": json.dumps(action_info.get("env_action_probs", [])),
            "event_label": label["decision"],
            "event_label_reason": label["reason"],
            "event_label_schema_version": label["schema_version"],
            "event_supervision_weight": label["supervision_weight"],
            "semantic_state_sha256": hashlib.sha256(
                json.dumps(
                    decision_info["semantic_state"], ensure_ascii=False,
                    sort_keys=True, separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "reward": float(reward), "service_completed": bool(event.get("service_completed")),
            "migration_success": bool(event.get("migration_success", False)),
            "failure_reason": event.get("failure_reason") or event.get("admission_rejection_reason"),
            "migration_status": event.get("migration_status"),
            "input_transfer_bytes": int(event.get("input_transfer_bytes", 0)),
            "model_transfer_bytes": int(event.get("model_transfer_bytes", 0)),
            "state_transfer_bytes": int(event.get("state_transfer_bytes", 0)),
            "backhaul_transfer_bytes": int(event.get("backhaul_transfer_bytes", 0)),
            "partial_transfer_bytes": int(event.get("partial_transfer_bytes", 0)),
            "step_cost_seconds": float(event.get("step_cost_seconds", 0.0)),
            "phase_events": json.dumps(phases, sort_keys=True),
            "terminated": bool(terminated), "truncated": bool(truncated),
        })
        observation, info = next_observation, next_info
        if terminated or truncated:
            break
    summary = env.summary()
    completed = bool(summary["workflow_completed"])
    total_bytes = int(summary["total_transfer_bytes"])
    row = {
        "arm": arm, "view": view, "method": method, "seed": seed,
        "time_profile": time_profile, "condition_id": condition_id,
        "source_segment_id": source["source_segment_id"], "window_id": source["window_id"],
        "workflow_completion_rate": float(completed),
        "on_time_workflow_completion_rate": float(summary.get("on_time_workflow_completed", 0)),
        "late_workflow_completion_rate": float(summary.get("late_workflow_completed", 0)),
        "unfinished_rate": float(not completed),
        "service_failure_rate": float(int(summary["service_failures"]) > 0),
        "service_failures": int(summary["service_failures"]),
        "completed_nodes": int(summary["completed_nodes"]), "node_count": int(summary["node_count"]),
        "terminated": bool(summary["terminated"]),
        "truncated": bool(not summary["terminated"]),
        "completed_sample_elapsed_seconds": float(summary["modeled_completion_seconds"]) if completed else "",
        "elapsed_seconds_for_failed_or_truncated": "" if completed else float(summary["modeled_completion_seconds"]),
        "total_transfer_mb": total_bytes / 1_000_000.0,
        "input_transfer_bytes": int(summary["input_transfer_bytes"]),
        "model_transfer_bytes": int(summary["model_transfer_bytes"]),
        "state_transfer_bytes": int(summary["state_transfer_bytes"]),
        "backhaul_transfer_bytes": int(summary.get("backhaul_transfer_bytes", 0)),
        "partial_transfer_bytes": int(summary.get("partial_transfer_bytes", 0)),
        "migration_attempts": int(summary["migration_attempts"]),
        "migration_successes": int(summary["migration_successes"]),
        "state_ready_count": len(env.prepared_state),
        "recompute_seconds": float(summary["recompute_seconds"]),
        "phase_elapsed_seconds": json.dumps(dict(sorted(phase_seconds.items())), sort_keys=True),
        "max_consecutive_no_progress_steps": max_no_progress,
        "decision_overhead_ms": decision_ns / 1_000_000.0,
        "reward": float(summary["reward"]),
        "reward_node_completion": float(summary["reward_node_completion"]),
        "reward_workflow_completion": float(summary["reward_workflow_completion"]),
        "reward_time_penalty": float(summary["reward_time_penalty"]),
        "reward_transfer_penalty": float(summary["reward_transfer_penalty"]),
        "reward_failure_penalty": float(summary["reward_failure_penalty"]),
        "reward_deadline_penalty": float(summary["reward_deadline_penalty"]),
    }
    for action in range(5):
        row[f"action_{action}"] = counts[action]
    if agent is not None:
        agent._deterministic_action = bool(previous_deterministic)
    return row, ledger


def _selection_score(row: dict[str, Any], update_index: int) -> tuple[float, ...]:
    return (
        float(row["on_time_workflow_completion_rate"]),
        float(row["workflow_completion_rate"]),
        -float(row["service_failure_rate"]),
        -float(row["unfinished_rate"]),
        -float(row["total_transfer_mb"]),
        -float(update_index),
    )


def _training_signal(
    row: dict[str, Any], *, arm: str, method: str, seed: int,
    update_index: int, global_step: int,
    positive_prepare_calibration_abstention_enabled: bool = False,
) -> dict[str, Any]:
    event = dict(row.get("env_info", {}).get("transition", {}) or {})
    action_info = dict(row.get("action_info", {}) or {})
    decision = dict(row.get("decision_info", {}) or {})
    label = public_prepare_advantage_label(
        decision["semantic_state"], decision["action_mask"],
        positive_prepare_calibration_abstention_enabled=(
            positive_prepare_calibration_abstention_enabled
        ),
    )
    estimate = label["estimate"]
    action4 = estimate["actions"]["4"]
    return {
        "arm": arm, "method": method, "seed": seed,
        "update_index": update_index, "global_step": global_step,
        "positive_prepare_calibration_abstention_enabled": bool(
            positive_prepare_calibration_abstention_enabled
        ),
        "training_episode_index": row["training_episode_index"],
        "rollout_segment_index": row["rollout_segment_index"],
        "executed_action": int(row["action"]), "reward": float(row["reward"]),
        "value": float(row["value"]), "advantage": float(row["advantage"]),
        "return": float(row["return"]), "old_log_prob": float(row["log_prob"]),
        "bootstrap_value": float(row["bootstrap_value"]),
        "segment_terminated": bool(row["segment_terminated"]),
        "segment_truncated": bool(row["segment_truncated"]),
        "batch_boundary": bool(row["batch_boundary"]),
        "event_label": label["decision"],
        "event_label_reason": label["reason"],
        "event_label_schema_version": label["schema_version"],
        "event_supervision_weight": label["supervision_weight"],
        "semantic_state_sha256": hashlib.sha256(
            json.dumps(
                decision["semantic_state"], ensure_ascii=False,
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "action4_legal": bool(action4["legal"]),
        "action4_current_service": action4["current_service"],
        "action4_target_prepare": action4["target_prepare"],
        "action4_deadline_fit": action4["deadline_fit"],
        "action4_future_readiness_gain": action4["future_readiness_gain"],
        "action_probabilities": json.dumps(action_info.get("env_action_probs", [])),
        "service_completed": bool(event.get("service_completed")),
        "migration_success": bool(event.get("migration_success", False)),
        "failure_reason": event.get("failure_reason") or event.get("admission_rejection_reason"),
        "input_transfer_bytes": int(event.get("input_transfer_bytes", 0)),
        "model_transfer_bytes": int(event.get("model_transfer_bytes", 0)),
        "state_transfer_bytes": int(event.get("state_transfer_bytes", 0)),
        "backhaul_transfer_bytes": int(event.get("backhaul_transfer_bytes", 0)),
        "partial_transfer_bytes": int(event.get("partial_transfer_bytes", 0)),
        "step_cost_seconds": float(event.get("step_cost_seconds", 0.0)),
    }


def _build_cell_agent(cell: dict[str, Any], seed: int, config: dict[str, Any]) -> Any:
    cell_config = deepcopy(config)
    cell_config["mechanism_aux_current_service_feasibility_gate_enabled"] = False
    cell_config["mechanism_aux_missing_current_event_abstention_enabled"] = False
    cell_config["mechanism_aux_causal_public_prepare_advantage_enabled"] = bool(cell["candidate_enabled"])
    cell_config["mechanism_aux_positive_prepare_calibration_abstention_enabled"] = bool(
        cell.get("positive_prepare_calibration_abstention_enabled", False)
    )
    return _build_agent(
        cell["method"], seed, cell_config,
        agent_overrides={"value_normalization_enabled": False},
    )


def _completion_receipt_payload(
    *, expected_commit: str, elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCIENTIFIC_COMPLETION_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "scientific_execution_complete": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": expected_commit,
        "elapsed_seconds": float(elapsed_seconds),
        "learned_cells": 20,
        "environment_steps": 23040,
        "optimizer_steps": 3840,
        "checkpoint_selection_episodes": 80,
        "final_evaluation_episodes": 320,
        "automatic_retry": False,
        "formal_or_holdout_reads": 0,
    }


def execute(protocol_path: Path, output_root: Path, expected_commit: str) -> None:
    if output_root.exists():
        raise FileExistsError(f"create-only output exists: {output_root}")
    protocol = _load(protocol_path)
    gate = _validate_protocol(
        protocol, require_clean=True, require_authorized=True
    )
    if _git_commit() != expected_commit:
        raise RuntimeError("scientific commit identity mismatch")
    output_root.mkdir(parents=True)
    checkpoints = output_root / "checkpoints"
    checkpoints.mkdir()
    _write_json(output_root / "runner_entered.json", {
        "status": "entered", "entered_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": expected_commit,
    })
    started = time.monotonic()
    config, instance, source_bundle, runtime = _load_runtime(protocol)
    actual_class_gate = _actual_class_gate(protocol, config, instance, source_bundle, runtime)
    traces, sources, phase = source_bundle["traces"], source_bundle["sources"], runtime["phase"]
    train_segment = protocol["data_assignment"]["training_source_segment"]
    dev_segment = protocol["data_assignment"]["checkpoint_selection_source_segment"]
    training = protocol["training"]
    step_cap = int(training["episode_max_steps"])

    summaries: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    optimizer_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    training_episode_rows: list[dict[str, Any]] = []
    training_signal_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []

    for cell in training["learned_cells"]:
        for seed in training["seeds"]:
            if time.monotonic() - started > float(training["wall_clock_cap_seconds"]):
                raise TimeoutError("scientific wall-clock cap exceeded")
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            agent = _build_cell_agent(cell, int(seed), config)
            parameter_count = sum(int(p.numel()) for p in agent._network.parameters())
            env_factory = lambda: _phase_env(config, instance, traces[train_segment], phase, "default")
            state = None
            episodes_started = 0
            global_step = 0
            candidates = []
            for update_index in range(1, 25):
                batch, state, episodes_started, episodes = _collect_batch(
                    agent=agent, env_factory=env_factory, state=state,
                    episodes_started=episodes_started,
                    transition_count=int(training["transitions_per_update"]), step_cap=step_cap,
                    gamma=float(config["training"]["gamma"]),
                    gae_lambda=float(config["training"]["gae_lambda"]),
                )
                for row in batch:
                    global_step += 1
                    training_signal_rows.append(_training_signal(
                        row, arm=cell["arm"], method=cell["method"], seed=int(seed),
                        update_index=update_index, global_step=global_step,
                        positive_prepare_calibration_abstention_enabled=bool(
                            cell.get(
                                "positive_prepare_calibration_abstention_enabled",
                                False,
                            )
                        ),
                    ))
                training_episode_rows.extend({
                    "arm": cell["arm"], "method": cell["method"], "seed": seed, **row,
                } for row in episodes)
                result = _json_safe(agent.learn(batch))
                if result.get("policy_update_skipped") or int(result.get("optimizer_step_count", -1)) != 8:
                    raise RuntimeError("optimizer-step contract drift")
                records = list(result.get("optimizer_step_records", []))
                if len(records) != 8:
                    raise RuntimeError("optimizer receipt count drift")
                update_rows.append({
                    "arm": cell["arm"], "method": cell["method"], "seed": seed,
                    "update_index": update_index, "environment_steps": global_step,
                    **{key: value for key, value in result.items() if key != "optimizer_step_records"},
                })
                optimizer_rows.extend({
                    "arm": cell["arm"], "method": cell["method"], "seed": seed,
                    "update_index": update_index, **row,
                } for row in records)
                if update_index in training["checkpoint_update_candidates"]:
                    checkpoint = checkpoints / f"{cell['arm']}_seed{seed}_update{update_index}.pt"
                    agent.save(str(checkpoint))
                    dev_row, _ = _evaluate_episode(
                        method=cell["method"], seed=int(seed), arm=cell["arm"], view="selection_dev",
                        env_factory=lambda: _phase_env(config, instance, traces[dev_segment], phase, "default"),
                        source=sources[dev_segment], condition_id="default",
                        time_profile="raw_ngsim_event_time_v2", step_cap=step_cap, agent=agent,
                    )
                    candidate = {
                        "arm": cell["arm"], "method": cell["method"], "seed": seed,
                        "update_index": update_index,
                        "score": list(_selection_score(dev_row, update_index)),
                        "checkpoint": str(checkpoint.relative_to(output_root)),
                        "checkpoint_sha256": _sha256(checkpoint),
                        "development_row": dev_row,
                        "reward_used_for_selection": False,
                        "final_evaluation_used_for_selection": False,
                    }
                    candidates.append(candidate); candidate_rows.append(candidate)
            if global_step != 1152:
                raise RuntimeError("per-cell environment-step drift")
            best = max(candidates, key=lambda row: tuple(row["score"]))
            fixed = next(row for row in candidates if int(row["update_index"]) == 24)
            selected_path = checkpoints / f"{cell['arm']}_seed{seed}_selected.pt"
            agent.load(str(output_root / best["checkpoint"])); agent.save(str(selected_path))
            view_checkpoints = {"selected": selected_path, "fixed_update24": output_root / fixed["checkpoint"]}
            for view, checkpoint in view_checkpoints.items():
                agent.load(str(checkpoint))
                for segment in protocol["evaluation"]["source_segments"]:
                    for condition in protocol["evaluation"]["phase_v2_conditions"]:
                        row, ledger = _evaluate_episode(
                            method=cell["method"], seed=int(seed), arm=cell["arm"], view=view,
                            env_factory=lambda s=segment, c=condition: _phase_env(config, instance, traces[s], phase, c),
                            source=sources[segment], condition_id=condition,
                            time_profile="raw_ngsim_event_time_v2", step_cap=step_cap, agent=agent,
                        )
                        row["parameter_count"] = parameter_count
                        evaluation_rows.append(row); behavior_rows.extend(ledger)
                    row, ledger = _evaluate_episode(
                        method=cell["method"], seed=int(seed), arm=cell["arm"], view=view,
                        env_factory=lambda s=segment: RawNGSIMEventTimeEnv(config, deepcopy(instance), traces[s]),
                        source=sources[segment], condition_id="mechanism_off_default",
                        time_profile="raw_ngsim_event_time_v1", step_cap=step_cap, agent=agent,
                    )
                    row["parameter_count"] = parameter_count
                    evaluation_rows.append(row); behavior_rows.extend(ledger)
            summaries.append({
                "arm": cell["arm"], "method": cell["method"], "seed": seed,
                "candidate_enabled": bool(cell["candidate_enabled"]),
                "positive_prepare_calibration_abstention_enabled": bool(
                    cell.get(
                        "positive_prepare_calibration_abstention_enabled", False
                    )
                ),
                "parameter_count": parameter_count, "environment_steps": global_step,
                "updates": 24, "optimizer_steps": 192, "episodes_started": episodes_started,
                "active_partial_episode_at_budget": state is not None,
                "selected_update": int(best["update_index"]),
                "selected_checkpoint": str(selected_path.relative_to(output_root)),
                "selected_checkpoint_sha256": _sha256(selected_path),
                "fixed_update24_checkpoint": fixed["checkpoint"],
                "fixed_update24_checkpoint_sha256": fixed["checkpoint_sha256"],
            })
            print(json.dumps({
                "arm": cell["arm"], "seed": seed, "steps": global_step,
                "selected_update": best["update_index"],
            }), flush=True)

    rule_rows: list[dict[str, Any]] = []
    rule_ledgers: list[dict[str, Any]] = []
    public_rules = [CausalPublicImmediateRule(), CausalPublicTwoStepRule()]
    for rule in public_rules:
        for segment in protocol["evaluation"]["source_segments"]:
            for condition in protocol["evaluation"]["phase_v2_conditions"]:
                row, ledger = _evaluate_episode(
                    method=rule.method_name, seed="rule", arm="public_rule", view="fixed_rule",
                    env_factory=lambda s=segment, c=condition: _phase_env(config, instance, traces[s], phase, c),
                    source=sources[segment], condition_id=condition,
                    time_profile="raw_ngsim_event_time_v2", step_cap=step_cap, rule=rule,
                )
                row["capability_profile"] = rule.capability_profile
                row["phase_v2_independent_baseline"] = False
                rule_rows.append(row); rule_ledgers.extend(ledger)
            row, ledger = _evaluate_episode(
                method=rule.method_name, seed="rule", arm="public_rule", view="fixed_rule",
                env_factory=lambda s=segment: RawNGSIMEventTimeEnv(config, deepcopy(instance), traces[s]),
                source=sources[segment], condition_id="mechanism_off_default",
                time_profile="raw_ngsim_event_time_v1", step_cap=step_cap, rule=rule,
            )
            row["capability_profile"] = rule.capability_profile
            row["phase_v2_independent_baseline"] = True
            rule_rows.append(row); rule_ledgers.extend(ledger)
    privileged = TwoStepCostRule()
    for segment in protocol["evaluation"]["source_segments"]:
        row, ledger = _evaluate_episode(
            method="privileged_exact_two_step_v1", seed="rule", arm="privileged_rule",
            view="capability_reference_only",
            env_factory=lambda s=segment: RawNGSIMEventTimeEnv(config, deepcopy(instance), traces[s]),
            source=sources[segment], condition_id="mechanism_off_default",
            time_profile="raw_ngsim_event_time_v1", step_cap=step_cap, rule=privileged,
        )
        row["capability_profile"] = "model_based_hidden_actual_contact_v1"
        row["phase_v2_available"] = False
        rule_rows.append(row); rule_ledgers.extend(ledger)

    if len(summaries) != 20 or len(candidate_rows) != 80 or len(evaluation_rows) != 320:
        raise RuntimeError("scientific row-count identity drift")
    if len(training_signal_rows) != 23040 or len(optimizer_rows) != 3840:
        raise RuntimeError("scientific training-count identity drift")
    calibration_abstention_run = protocol.get("schema_version") in {
        "cscwd_phase_v2_positive_prepare_calibration_abstention_ab_v1",
        LANKERSHIM_POSTFIX_SCHEMA,
    }
    postfix_run = protocol.get("schema_version") == LANKERSHIM_POSTFIX_SCHEMA
    _write_json(output_root / "run_manifest.json", {
        "schema_version": (
            "cscwd_phase_v2_recompute_observability_calibration_abstention_ab_manifest_v1"
            if postfix_run
            else "cscwd_phase_v2_positive_prepare_calibration_abstention_ab_manifest_v1"
            if calibration_abstention_run
            else "cscwd_phase_v2_causal_public_pilot_manifest_v1"
        ),
        "status": "complete", "git_commit": expected_commit,
        "protocol": str(protocol_path.relative_to(ROOT)), "protocol_sha256": _sha256(protocol_path),
        "gate": gate, "actual_class_gate": actual_class_gate,
        "trace_identity": runtime["trace_identity"],
        "reward_profile": "original_reward_v1", "critic_target_normalization": "raw_disabled",
        "calibration_abstention_scope": (
            {
                "raw_producer_confidence_calibrated": False,
                "new_arm_effect": (
                    "abstains_positive_prepare_auxiliary_supervision_when_"
                    "public_prediction_confidence_is_not_calibrated"
                    if postfix_run
                    else "removes_all_positive_prepare_auxiliary_supervision"
                ),
                "fine_grained_uncertainty_control": False,
                "three_head_structure_contribution_tested": False,
                "single_variable": (
                    "mechanism_aux_positive_prepare_calibration_abstention_enabled"
                ),
            }
            if calibration_abstention_run
            else None
        ),
        "development_split_scope": (
            "same_lankershim_recording_time_disjoint_windows_"
            "historical_check_exposure_unknown"
            if postfix_run else None
        ),
        "formal_or_holdout_reads": 0, "raw_or_checkpoint_uploaded": False,
        "counts": {"learned_cells": 20, "training_steps": 23040, "optimizer_steps": 3840,
                   "checkpoint_selection_episodes": 80, "final_learned_evaluation_episodes": 320,
                   "rule_evaluation_episodes": len(rule_rows)},
    })
    _write_json(output_root / "training_summary.json", summaries)
    _write_json(output_root / "checkpoint_candidates.json", candidate_rows)
    _write_table(output_root / "training_updates.csv", update_rows)
    _write_table(output_root / "optimizer_rows.csv", optimizer_rows)
    _write_table(output_root / "training_episode_rows.csv", training_episode_rows)
    _write_table(output_root / "training_signal_rows.csv", training_signal_rows)
    _write_table(output_root / "evaluation_rows.csv", evaluation_rows)
    _write_table(output_root / "behavior_rows.csv", behavior_rows)
    _write_table(output_root / "rule_evaluation_rows.csv", rule_rows)
    _write_table(output_root / "rule_behavior_rows.csv", rule_ledgers)
    _write_json(
        output_root / "completion_receipt.json",
        _completion_receipt_payload(
            expected_commit=expected_commit,
            elapsed_seconds=time.monotonic() - started,
        ),
    )
    _integrity(output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root")
    parser.add_argument("--expected-git-commit")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = _load(protocol_path)
    if args.preflight:
        result = _validate_protocol(
            protocol, require_clean=False, require_authorized=False
        )
        config, instance, source_bundle, runtime = _load_runtime(protocol)
        result["actual_class_gate"] = _actual_class_gate(
            protocol, config, instance, source_bundle, runtime
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return
    if not args.output_root or not args.expected_git_commit:
        raise RuntimeError("--run requires output root and expected commit")
    execute(protocol_path, Path(args.output_root).resolve(), args.expected_git_commit)


if __name__ == "__main__":
    main()
