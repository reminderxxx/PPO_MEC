"""Bounded synthetic elapsed-time mobility sensitivity, opt-in and no training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

SEED = 7
INSTANCE_IDS = (
    "dev_01", "regression_00", "regression_05", "frozen_check_05",
    "dev_00", "frozen_check_02", "frozen_check_04", "frozen_check_06",
)
METHODS = ("old_sa", "event_abstention_sa", "ppo", "mappo", "two_step")
PROFILES = ("decision_step_original", "synthetic_elapsed_5s_atomic_v1")
RUN_ID = "cscwd_mobility_elapsed_sensitivity_20261010_v1"
PLAN_COMMIT = "428cdc48a9791523d68c57fa8ca7f6372ef256f1"
EXPECTED_SOURCE_SHA = {
    "src/envs/core/calibrated_continuous_workflow_env.py": "c6ded6ce84230519160a40f29e276b5823eb204a35e31c35f914d1c6bfee3a59",
    "src/agents/sa_ghmappo_core.py": "74b1bd1adde037b9e3076057ac32a65c835bc6cca5d3460c5dba2e3429ddf130",
    "scripts/run_calibrated_workflow_event_aux_abstention_ab.py": "07ca5a11cf1c1a5c98b3960ffb558b72a8364d59b622079debced8dacd04bc72",
    "scripts/run_calibrated_workflow_strong_baselines.py": "f8274cec0da7e0973c518e31fe66844ff9a1bbe900f537703f8f8184b981b468",
    "scripts/run_calibrated_workflow_interface_repair.py": "784892709f85d5aca19f889d9d2ed70ba8c983336416136df0d9cce46204df8f",
}
CHECKPOINTS = {
    "old_sa": ("cscwd_causal_prepared_state_visibility_matched_20261010_v1", "sa_ghmappo", "6fca5f2dac35671b445fa1439e71585d3625a293dc7a5d6d414cb25a9f247feb"),
    "event_abstention_sa": ("cscwd_event_aux_abstention_ab_20261010_v1", "sa_ghmappo", "9fdd94cd72227c171b23a686163e47631e46158a0d8dfe791c1dabf2472301fe"),
    "ppo": ("cscwd_causal_prepared_state_visibility_matched_20261010_v1", "ppo", "f93ad95a22a066995bf0deea6f12efe04f6ced2a5f62b021ce082561a531547d"),
    "mappo": ("cscwd_causal_prepared_state_visibility_matched_20261010_v1", "mappo", "5ef457a8403f2b7c83ce299ba9ddc323f9aa949f81ad42e85a2f7ea093d259ce"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def network_sha(network: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class SyntheticElapsedFiveSecondEnv:  # composed dynamically below, after source identity check
    pass


def build_elapsed_env_class(base_class):
    class ElapsedEnv(base_class):
        """One frozen synthetic timeline; all action execution remains inherited."""

        diagnostic_time_profile = "synthetic_elapsed_5s_atomic_v1"

        def _mobility_index(self) -> int:
            slot = math.floor(float(self.clock_seconds) / 5.0)
            return min(max(slot, 0), len(self.instance["rsu_sequence"]) - 1)

        def _offset_in_slot(self) -> float:
            return float(self.clock_seconds) - math.floor(float(self.clock_seconds) / 5.0) * 5.0

        def _contact_budget_seconds(self) -> float:
            current = self._current_rsu_id()
            countdown = 1
            for index, candidate in enumerate(self._predicted_sequence(), start=1):
                if candidate != current:
                    countdown = index
                    break
            return max(0.0, 5.0 * countdown - self._offset_in_slot())

        def _physical_contact_budget_seconds(self) -> float:
            if not self._causal_prediction_enabled() or self._decision_model_mode:
                return self._contact_budget_seconds()
            sequence = self.instance["rsu_sequence"]
            index = self._mobility_index()
            current = str(sequence[index])
            countdown = 1
            horizon = int(self.config["prediction_horizon"])
            for offset, candidate in enumerate(sequence[index + 1 : index + 1 + horizon], start=1):
                if str(candidate) != current:
                    countdown = offset
                    break
            return max(0.0, 5.0 * countdown - self._offset_in_slot())

    return ElapsedEnv


class PreviewFacade:
    """Count model-preview steps without changing the source two-step rule."""

    def __init__(self, env, counter: list[int]):
        self.env = env
        self.counter = counter

    def __getattr__(self, name):
        return getattr(self.env, name)

    def clone_for_decision_model(self):
        return PreviewFacade(self.env.clone_for_decision_model(), self.counter)

    def clone(self):
        return PreviewFacade(self.env.clone(), self.counter)

    def step(self, action):
        self.counter[0] += 1
        return self.env.step(action)


def validate_native_state(env) -> dict[str, Any]:
    for rsu_id, cache in env.caches.items():
        if len(cache.residents) != len(set(cache.residents)):
            raise RuntimeError(f"duplicate resident: {rsu_id}")
        if any(item not in env._object_catalog for item in cache.residents):
            raise RuntimeError(f"unknown resident: {rsu_id}")
        if env._resident_bytes(cache) > cache.capacity_bytes:
            raise RuntimeError(f"capacity overflow: {rsu_id}")
    completed = set(env.completed)
    for rsu_id, state in env.prepared_state.items():
        if not set(state["completed_node_ids"]).issubset(completed):
            raise RuntimeError(f"prepared future prefix: {rsu_id}")
        if state["workflow_id"] != env.instance["workflow_id"]:
            raise RuntimeError(f"prepared workflow mismatch: {rsu_id}")
    return {
        "cache_residents": {key: list(value.residents) for key, value in env.caches.items()},
        "prepared_state": env.prepared_state,
        "completed_node_ids": list(env.completed),
    }


def run_episode(*, base_class, elapsed_class, rule_class, build_learned,
                config: dict, instance: dict, method: str, profile: str, source: Path) -> tuple[dict, list[dict], int]:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    arm_config = {**config, "mechanism_aux_missing_current_event_abstention_enabled": method == "event_abstention_sa"}
    env_class = base_class if profile == "decision_step_original" else elapsed_class
    env = env_class(arm_config, instance)
    observation, info = env.reset()
    agent = None
    checkpoint_sha = None
    parameter_before = None
    if method != "two_step":
        run_id, algorithm, expected_sha = CHECKPOINTS[method]
        checkpoint = source / "artifacts/experiments" / run_id / "checkpoints" / f"{algorithm}_seed{SEED}_selected.pt"
        if sha(checkpoint) != expected_sha:
            raise RuntimeError(f"checkpoint hash drift: {checkpoint}")
        checkpoint_sha = expected_sha
        agent = build_learned(algorithm, SEED, arm_config, popart_enabled=False)
        agent.load(str(checkpoint))
        agent._deterministic_action = True
        parameter_before = network_sha(agent._network)
    rule = rule_class() if method == "two_step" else None
    step_cap = min(24, int(instance["max_steps"]))
    transition_rows = []
    preview_count = [0]
    while not env.terminated and env.step_index < step_cap:
        start_clock = float(env.clock_seconds)
        start_slot = int(env._mobility_index())
        start_rsu = env._current_rsu_id()
        public_contact = float(info["semantic_state"]["calibrated_context"]["contact_budget_seconds"])
        physical_contact = float(env._physical_contact_budget_seconds())
        mask = list(info["action_mask"])
        if agent is not None:
            action, action_info = agent.act(observation, info)
        else:
            action = rule.select_action(PreviewFacade(env, preview_count))
            action_info = {"raw_env_action": action, "final_env_action": action}
        if action not in range(5) or not mask[action]:
            raise RuntimeError(f"illegal public action: {method} {instance['design_id']} {action}")
        next_observation, _, terminated, truncated, next_info = env.step(action)
        event = dict(next_info["transition"])
        finish_clock = float(env.clock_seconds)
        if event["current_rsu_id"] != start_rsu or int(event["step_index"]) != len(transition_rows):
            raise RuntimeError("action executed at a different RSU/decision index")
        expected = (
            (float(arm_config["objective"]["failed_service_seconds"]) if not event["service_completed"] else 0.0)
            + float(event["service_operation_seconds"])
            + float(event["recompute_seconds"])
            + float(event["transfer_seconds"])
        )
        if abs(expected - float(event["step_cost_seconds"])) > 1e-6 or abs(finish_clock - start_clock - expected) > 1e-6:
            raise RuntimeError("step cost / clock conservation failed")
        transition_rows.append({
            "method": method, "profile": profile, "design_id": instance["design_id"],
            "step_index": int(event["step_index"]), "action": int(action),
            "raw_env_action": int(action_info.get("raw_env_action", action)),
            "mask": mask, "start_clock_seconds": start_clock, "finish_clock_seconds": finish_clock,
            "start_slot": start_slot, "finish_slot": int(env._mobility_index()),
            "crossed_slot_count": max(int(math.floor(finish_clock / 5)) - int(math.floor(start_clock / 5)), 0),
            "start_rsu": start_rsu, "finish_rsu": env._current_rsu_id(),
            "public_contact_budget_seconds": public_contact,
            "physical_contact_budget_seconds": physical_contact,
            "service_completed": bool(event["service_completed"]),
            "migration_success": bool(event["migration_success"]),
            "state_transfer_status": (event.get("state_transfer") or {}).get("status"),
            "model_transfer_bytes": int(event["model_transfer_bytes"]),
            "state_transfer_bytes": int(event["state_transfer_bytes"]),
            "input_transfer_bytes": int(event["input_transfer_bytes"]),
            "service_operation_seconds": float(event["service_operation_seconds"]),
            "recompute_seconds": float(event["recompute_seconds"]),
            "transfer_seconds": float(event["transfer_seconds"]),
            "model_load_seconds": float(event["model_load_seconds"]),
            "state_restore_seconds": float(event["state_restore_seconds"]),
            "step_cost_seconds": float(event["step_cost_seconds"]),
            "deadline_seconds": float(event["deadline_seconds"]),
            "deadline_miss_event": bool(event["deadline_miss_event"]),
            "cache_events": event["cache_events"],
        })
        observation, info = next_observation, next_info
        if terminated or truncated:
            break
    if agent is not None and parameter_before != network_sha(agent._network):
        raise RuntimeError("checkpoint parameter changed during sensitivity episode")
    native_state = validate_native_state(env)
    summary = env.summary()
    if abs(sum(row["step_cost_seconds"] for row in transition_rows) - float(summary["modeled_completion_seconds"])) > 1e-6:
        raise RuntimeError("episode elapsed conservation failed")
    for metric in ("model_transfer_bytes", "state_transfer_bytes", "input_transfer_bytes"):
        if sum(row[metric] for row in transition_rows) != int(summary[metric]):
            raise RuntimeError(f"episode byte conservation failed: {metric}")
    row = {
        "method": method, "profile": profile, "design_id": instance["design_id"],
        "split": instance["split"], "seed": SEED if method != "two_step" else "rule",
        "checkpoint_sha256": checkpoint_sha, "parameter_sha256_before_after": parameter_before,
        "steps": len(transition_rows), "preview_model_steps": preview_count[0],
        "completed": bool(summary["workflow_completed"]),
        "on_time": bool(summary["on_time_workflow_completed"]),
        "deadline_missed": bool(summary["deadline_missed"]),
        "service_failures": int(summary["service_failures"]),
        "elapsed_seconds": float(summary["modeled_completion_seconds"]),
        "model_transfer_bytes": int(summary["model_transfer_bytes"]),
        "state_transfer_bytes": int(summary["state_transfer_bytes"]),
        "input_transfer_bytes": int(summary["input_transfer_bytes"]),
        "handoff_failures": int(summary["handoff_failures"]),
        "migration_successes": int(summary["migration_successes"]),
        "action_counts": dict(Counter(str(item["action"]) for item in transition_rows)),
        "native_state": native_state,
    }
    return row, transition_rows, preview_count[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="execute the single frozen sensitivity pass")
    args = parser.parse_args()
    source = args.source_root.resolve()
    for relative, expected in EXPECTED_SOURCE_SHA.items():
        if sha(source / relative) != expected:
            raise RuntimeError(f"source code drift: {relative}")
    sys.path.insert(0, str(source))
    from scripts.run_calibrated_workflow_event_aux_abstention_ab import _load_candidate_inputs
    from scripts.run_calibrated_workflow_strong_baselines import _build_learned
    from src.envs.core.calibrated_continuous_workflow_env import CalibratedContinuousWorkflowEnv, TwoStepCostRule

    protocol_path = source / "configs/experiment/calibrated_workflow_event_aux_abstention_ab_v1.json"
    if sha(protocol_path) != "61295f49d03ce2f9c4f2b8cf644bc66d6a0436f43b10a61ad9b846776b54b6c4":
        raise RuntimeError("protocol identity drift")
    protocol = json.loads(protocol_path.read_text())
    config, splits, _, _ = _load_candidate_inputs(protocol)
    if float(config["mobility_abstraction"]["decision_step_seconds"]) != 5.0 or float(config["objective"]["failed_service_seconds"]) != 2.0:
        raise RuntimeError("synthetic time scale drift")
    instances = {str(item["design_id"]): item for split in ("dev", "regression", "frozen_check") for item in splits[split]}
    if any(identifier not in instances for identifier in INSTANCE_IDS):
        raise RuntimeError("frozen instance missing")
    elapsed_class = build_elapsed_env_class(CalibratedContinuousWorkflowEnv)
    if not args.run:
        print("PRECHECK_PASS; use --run for one frozen sensitivity pass")
        return
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite sensitivity output")
    output.mkdir(parents=True)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    manifest = {
        "schema_version": "cscwd_mobility_elapsed_sensitivity_v1", "run_id": RUN_ID,
        "status": "running", "plan_commit": PLAN_COMMIT, "source_commit": source_commit,
        "source_root": str(source), "source_code_sha256": EXPECTED_SOURCE_SHA,
        "protocol_sha256": sha(protocol_path), "seed": SEED,
        "instance_ids": list(INSTANCE_IDS), "methods": list(METHODS), "profiles": list(PROFILES),
        "actual_environment_steps": 0, "decision_model_preview_steps": 0,
        "completed_episodes": 0, "files": [],
    }
    json_write(output / "run_manifest.json", manifest)
    episode_rows = []
    transition_rows = []
    for identifier in INSTANCE_IDS:
        instance = instances[identifier]
        for method in METHODS:
            for profile in PROFILES:
                row, events, previews = run_episode(
                    base_class=CalibratedContinuousWorkflowEnv, elapsed_class=elapsed_class,
                    rule_class=TwoStepCostRule, build_learned=_build_learned,
                    config=config, instance=instance, method=method, profile=profile, source=source)
                manifest["actual_environment_steps"] += len(events)
                manifest["decision_model_preview_steps"] += previews
                manifest["completed_episodes"] += 1
                if manifest["actual_environment_steps"] > 2880 or manifest["completed_episodes"] > 120:
                    raise RuntimeError("frozen episode or environment-step cap exceeded")
                episode_rows.append(row)
                transition_rows.extend(events)
                print(identifier, method, profile, len(events), "preview", previews, flush=True)
                json_write(output / "run_manifest.json", manifest)
    if len(episode_rows) != 80:
        raise RuntimeError("frozen episode matrix incomplete")
    for name, payload in (("episode_rows.json", episode_rows), ("transition_rows.json", transition_rows)):
        path = output / name
        json_write(path, payload)
        manifest["files"].append({"path": name, "sha256": sha(path), "rows": len(payload)})
    manifest["status"] = "complete"
    json_write(output / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
