"""Freeze and audit the two calibrated-workflow reward formulas before training."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config  # noqa: E402
from src.envs.core.calibrated_continuous_workflow_env import (  # noqa: E402
    CalibratedContinuousWorkflowEnv,
    ORIGINAL_REWARD_PROFILE,
    SERVICE_ALIGNED_REWARD_PROFILE,
)
from src.trainers.ppo_buffer import PPORolloutBuffer  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, check=True, capture_output=True, text=True).stdout.strip()


def _score_case(case: dict[str, Any], objective: dict[str, Any]) -> dict[str, Any]:
    gib = float(case["transfer_bytes"]) / float(1024**3)
    original = {
        "reward_node_completion": float(objective["node_completion_reward"]) * int(case["completed_nodes"]),
        "reward_workflow_completion": float(objective["workflow_completion_reward"]) if case["workflow_completed"] else 0.0,
        "reward_time_penalty": -float(objective["time_weight"]) * float(case["elapsed_seconds"]),
        "reward_transfer_penalty": -float(objective["transfer_gib_weight"]) * gib,
        "reward_failure_penalty": -float(objective["failure_penalty"]) * int(case["service_failures"]),
        "reward_deadline_penalty": -float(objective["deadline_penalty"]) if case["workflow_completed"] and case["elapsed_seconds"] > case["deadline_seconds"] else 0.0,
    }
    aligned = objective["reward_profiles"][SERVICE_ALIGNED_REWARD_PROFILE]
    candidate = {
        "reward_node_completion": float(aligned["node_completion_reward"]) * int(case["completed_nodes"]),
        "reward_workflow_completion": float(aligned["workflow_completion_reward"]) if case["workflow_completed"] else 0.0,
        "reward_service_operation_time_penalty": -float(aligned["service_operation_second_weight"]) * float(case["service_operation_seconds"]),
        "reward_transfer_penalty": -float(aligned["transfer_gib_weight"]) * gib,
        "reward_recompute_penalty": -float(aligned["recompute_second_weight"]) * float(case["recompute_seconds"]),
        "reward_failure_penalty": -float(aligned["failed_service_attempt_penalty"]) * int(case["service_failures"]),
        "reward_deadline_penalty": -float(aligned["deadline_miss_penalty"]) if case["elapsed_seconds"] > case["deadline_seconds"] else 0.0,
        "reward_external_truncation_penalty": 0.0,
    }
    output = dict(case)
    output["original_components"] = original
    output["original_return"] = float(sum(original.values()))
    output["candidate_components"] = candidate
    output["candidate_return"] = float(sum(candidate.values()))
    return output


def _transition_invariance(config: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    configs = {}
    for profile in (ORIGINAL_REWARD_PROFILE, SERVICE_ALIGNED_REWARD_PROFILE):
        item = deepcopy(config)
        item["reward_profile"] = profile
        configs[profile] = item
    envs = {profile: CalibratedContinuousWorkflowEnv(item, instance) for profile, item in configs.items()}
    traces: dict[str, list[dict[str, Any]]] = {profile: [] for profile in envs}
    for _ in range(12):
        reference = envs[ORIGINAL_REWARD_PROFILE]
        if reference.terminated:
            break
        action = int(reference.valid_actions()[0])
        for profile, env in envs.items():
            _, _, terminated, truncated, info = env.step(action)
            transition = dict(info["transition"])
            for key in ("reward", "reward_components", "reward_components_by_profile", "active_reward_profile"):
                transition.pop(key, None)
            traces[profile].append(
                {
                    "transition": transition,
                    "semantic_state": info["semantic_state"],
                    "action_mask": info["action_mask"],
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )
    return {
        "same_state_action_transition_trace": traces[ORIGINAL_REWARD_PROFILE] == traces[SERVICE_ALIGNED_REWARD_PROFILE],
        "steps_compared": len(traces[ORIGINAL_REWARD_PROFILE]),
    }


def _bootstrap_audit() -> dict[str, Any]:
    buffer = PPORolloutBuffer()
    buffer.add_step(
        observation=[0.0], action=0, reward=1.0, terminated=False, truncated=True,
        log_prob=0.0, value=0.5, next_observation=[1.0], action_info={}, decision_info={}, env_info={},
    )
    buffer.finalize(last_value=2.0, gamma=0.9, gae_lambda=1.0)
    row = buffer.to_training_rows()[0]
    expected = 1.0 + 0.9 * 2.0
    return {
        "external_truncation_bootstraps": abs(float(row["return"]) - expected) < 1e-9,
        "return": float(row["return"]),
        "expected_return": expected,
        "terminated": row["terminated"],
        "truncated": row["truncated"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/calibrated_workflow_service_reward_alignment_v1.json")
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output already exists: {output_root}")
    output_root.mkdir(parents=True)
    experiment_path = (ROOT_DIR / args.config).resolve()
    experiment = _load(experiment_path)
    base_path = (ROOT_DIR / experiment["base_config"]).resolve()
    manifest_path = (ROOT_DIR / experiment["workload_manifest"]).resolve()
    config, _ = _load_experiment_config(base_path)
    config["objective"].setdefault("reward_profiles", {})[SERVICE_ALIGNED_REWARD_PROFILE] = dict(experiment["service_aligned_reward"])
    manifest = _load(manifest_path)

    cases = [
        {"case": "on_time_complete", "completed_nodes": 5, "total_nodes": 5, "workflow_completed": True, "elapsed_seconds": 50.0, "deadline_seconds": 60.0, "transfer_bytes": 2 * 1024**3, "service_operation_seconds": 30.0, "recompute_seconds": 0.0, "service_failures": 0, "external_truncation": False},
        {"case": "same_completion_late", "completed_nodes": 5, "total_nodes": 5, "workflow_completed": True, "elapsed_seconds": 80.0, "deadline_seconds": 60.0, "transfer_bytes": 2 * 1024**3, "service_operation_seconds": 30.0, "recompute_seconds": 0.0, "service_failures": 0, "external_truncation": False},
        {"case": "same_completion_less_time", "completed_nodes": 5, "total_nodes": 5, "workflow_completed": True, "elapsed_seconds": 45.0, "deadline_seconds": 60.0, "transfer_bytes": 2 * 1024**3, "service_operation_seconds": 20.0, "recompute_seconds": 0.0, "service_failures": 0, "external_truncation": False},
        {"case": "same_completion_more_time", "completed_nodes": 5, "total_nodes": 5, "workflow_completed": True, "elapsed_seconds": 55.0, "deadline_seconds": 60.0, "transfer_bytes": 2 * 1024**3, "service_operation_seconds": 40.0, "recompute_seconds": 0.0, "service_failures": 0, "external_truncation": False},
        {"case": "same_completion_less_transfer", "completed_nodes": 5, "total_nodes": 5, "workflow_completed": True, "elapsed_seconds": 50.0, "deadline_seconds": 60.0, "transfer_bytes": 1 * 1024**3, "service_operation_seconds": 30.0, "recompute_seconds": 0.0, "service_failures": 0, "external_truncation": False},
        {"case": "same_completion_more_transfer", "completed_nodes": 5, "total_nodes": 5, "workflow_completed": True, "elapsed_seconds": 50.0, "deadline_seconds": 60.0, "transfer_bytes": 3 * 1024**3, "service_operation_seconds": 30.0, "recompute_seconds": 0.0, "service_failures": 0, "external_truncation": False},
        {"case": "near_complete_deadline_failure", "completed_nodes": 4, "total_nodes": 5, "workflow_completed": False, "elapsed_seconds": 80.0, "deadline_seconds": 60.0, "transfer_bytes": 1 * 1024**3, "service_operation_seconds": 24.0, "recompute_seconds": 0.0, "service_failures": 0, "external_truncation": False},
        {"case": "repeated_prepare_no_progress", "completed_nodes": 0, "total_nodes": 5, "workflow_completed": False, "elapsed_seconds": 12.0, "deadline_seconds": 60.0, "transfer_bytes": 161061273, "service_operation_seconds": 0.0, "recompute_seconds": 0.0, "service_failures": 6, "external_truncation": True},
        {"case": "unfinished_after_deadline", "completed_nodes": 4, "total_nodes": 5, "workflow_completed": False, "elapsed_seconds": 80.0, "deadline_seconds": 60.0, "transfer_bytes": 0, "service_operation_seconds": 24.0, "recompute_seconds": 0.0, "service_failures": 0, "external_truncation": True},
        {"case": "external_truncation_before_deadline", "completed_nodes": 4, "total_nodes": 5, "workflow_completed": False, "elapsed_seconds": 20.0, "deadline_seconds": 60.0, "transfer_bytes": 0, "service_operation_seconds": 20.0, "recompute_seconds": 0.0, "service_failures": 0, "external_truncation": True}
    ]
    scored = [_score_case(case, config["objective"]) for case in cases]
    by_name = {row["case"]: row for row in scored}
    checks = {
        "on_time_over_late": by_name["on_time_complete"]["candidate_return"] > by_name["same_completion_late"]["candidate_return"],
        "less_time_over_more_time": by_name["same_completion_less_time"]["candidate_return"] > by_name["same_completion_more_time"]["candidate_return"],
        "less_transfer_over_more_transfer": by_name["same_completion_less_transfer"]["candidate_return"] > by_name["same_completion_more_transfer"]["candidate_return"],
        "complete_over_near_failure": by_name["same_completion_late"]["candidate_return"] > by_name["near_complete_deadline_failure"]["candidate_return"],
        "repeat_prepare_not_rewarded": by_name["repeated_prepare_no_progress"]["candidate_return"] < 0.0,
        "unfinished_after_deadline_penalized": by_name["unfinished_after_deadline"]["candidate_components"]["reward_deadline_penalty"] < 0.0,
        "external_truncation_has_no_terminal_penalty": by_name["external_truncation_before_deadline"]["candidate_components"]["reward_external_truncation_penalty"] == 0.0,
    }
    invariance = _transition_invariance(config, manifest["instances"][0])
    bootstrap = _bootstrap_audit()
    passed = all(checks.values()) and invariance["same_state_action_transition_trace"] and bootstrap["external_truncation_bootstraps"]

    _write(output_root / "reward_freeze.json", {"schema_version": experiment["schema_version"], "git_commit": _git_commit(), "interface_profile": experiment["interface_profile"], "hierarchical_action_contract": experiment["hierarchical_action_contract"], "original_formula": "2*completed_nodes + 4*workflow_complete - 0.08*elapsed_seconds - 0.5*transfer_GiB - 3*failed_service_attempts - 4*late_complete", "candidate_formula": "completed_nodes + 100*workflow_complete - 25*first_deadline_miss - 2*failed_service_attempts - 0.02*service_operation_seconds - 0.25*transfer_GiB - 0.05*recompute_seconds", "candidate_weights": experiment["service_aligned_reward"], "service_interruption_measurement": "unavailable", "used_proxy": "failed_service_attempt_seconds_proxy = service_failures * failed_service_seconds", "ordinary_handoff_penalty": False, "prepare_or_action4_bonus": False, "trainer_truncation_penalty": False})
    _write(output_root / "reward_unit_cases.json", {"cases": scored, "checks": checks})
    _write(output_root / "transition_invariance.json", invariance)
    _write(output_root / "truncation_bootstrap.json", bootstrap)
    _write(output_root / "completion_receipt.json", {"status": "complete" if passed else "failed", "passed": passed, "created_at": datetime.now(timezone.utc).isoformat()})
    files = sorted(path for path in output_root.iterdir() if path.is_file())
    _write(output_root / "artifact_integrity.json", {"files": [{"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size} for path in files]})
    if not passed:
        raise RuntimeError("reward preflight failed")
    print(json.dumps({"output_root": str(output_root), "status": "complete"}))


if __name__ == "__main__":
    main()
