"""Align selected checkpoints on one frozen, public-state development probe set.

This is a read-only diagnostic.  It never calls ``learn`` and the continuation
return is explicitly a fixed-behavior probe target, not an estimate of V^pi.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import subprocess
import sys
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.freeze_calibrated_continuous_workflow_pilot import (  # noqa: E402
    _canonical_sha256,
    _load_experiment_config,
)
from scripts.run_calibrated_workflow_interface_repair import _build_agent  # noqa: E402
from scripts.run_calibrated_workflow_service_reward_alignment import (  # noqa: E402
    _resolved_arm_config,
)
from src.envs.core.calibrated_continuous_workflow_env import (  # noqa: E402
    CalibratedContinuousWorkflowEnv,
)


DESIGN_PATH = ROOT_DIR / "configs/experiment/calibrated_workflow_value_normalization_ab_v1.json"
SERVICE_REWARD_PATH = ROOT_DIR / "configs/experiment/calibrated_workflow_service_reward_alignment_v1.json"
CONTROL = "control_raw_critic_target"
CANDIDATE = "candidate_popart_critic_target"
METHODS = ("sa_ghmappo", "mappo", "ppo")
SEEDS = (7, 17, 29, 43, 61)
FIXED_ACTION_PRIORITY = (2, 0, 3, 1, 4)
STATES_PER_INSTANCE = 12
EPISODE_STEP_CAP = 24
TARGET_SEMANTICS = "complete_fixed_behavior_discounted_return_not_vpi"
FORBIDDEN_PUBLIC_KEYS = {"actual_mbps", "actual_link_mbps", "future_actual_mbps"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _state_dict_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda candidate: str(candidate)):
                digest.update(str(key).encode("utf-8"))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"sequence:{len(item)}".encode("utf-8"))
            for child in item:
                update(child)
        elif isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(str(tuple(array.shape)).encode("utf-8"))
            digest.update(array.tobytes())
        else:
            digest.update(
                json.dumps(_json_safe(item), sort_keys=True, default=str).encode("utf-8")
            )

    update(value)
    return digest.hexdigest()


def _integrity(root: Path, schema_version: str) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_integrity.json"
    )
    _write_json(
        root / "artifact_integrity.json",
        {
            "schema_version": schema_version,
            "files": [
                {
                    "path": str(path.relative_to(root)),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
            ],
        },
    )


def _load_protocol() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    design = _read_json(DESIGN_PATH)
    base_path = ROOT_DIR / design["base_config"]
    manifest_path = ROOT_DIR / design["workload_manifest"]
    base, _ = _load_experiment_config(base_path)
    manifest = _read_json(manifest_path)
    if manifest["config_sha256"] != _sha256(base_path):
        raise RuntimeError("frozen base-config hash mismatch")
    if manifest["resolved_config_sha256"] != _canonical_sha256(base):
        raise RuntimeError("frozen resolved-config hash mismatch")
    reward = _read_json(SERVICE_REWARD_PATH)["service_aligned_reward"]
    config = _resolved_arm_config(
        base, {"service_aligned_reward": reward}, design["reward_profile"]
    )
    dev = sorted(
        [row for row in manifest["instances"] if row["split"] == "dev"],
        key=lambda row: str(row["design_id"]),
    )
    if len(dev) != 4:
        raise RuntimeError(f"expected four frozen dev instances, found {len(dev)}")
    return design, config, dev


def _public_payload(observation: Any, info: dict[str, Any]) -> dict[str, Any]:
    return _json_safe(
        {
            "observation": observation,
            "decision_info": {
                "semantic_state": info["semantic_state"],
                "action_mask": info["action_mask"],
                "deterministic_policy": True,
                "run_metadata": dict(info.get("run_metadata", {})),
            },
        }
    )


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_nested_keys(child))
    return keys


def _replay_prefix(
    config: dict[str, Any], instance: dict[str, Any], prefix: list[int]
) -> tuple[CalibratedContinuousWorkflowEnv, Any, dict[str, Any]]:
    env = CalibratedContinuousWorkflowEnv(config, deepcopy(instance))
    observation, info = env.reset()
    for action in prefix:
        if env.terminated or env.step_index >= EPISODE_STEP_CAP:
            raise RuntimeError(f"prefix continues past episode boundary: {prefix}")
        observation, _, terminated, truncated, info = env.step(int(action))
        if (terminated or truncated) and len(prefix) > env.step_index:
            raise RuntimeError(f"prefix continues past terminal state: {prefix}")
    return env, observation, info


def _fixed_probe_return(env: CalibratedContinuousWorkflowEnv, gamma: float) -> dict[str, Any]:
    probe = env.clone()
    discounted_return = 0.0
    discount = 1.0
    actions: list[int] = []
    rewards: list[float] = []
    terminated = bool(probe.terminated)
    truncated = False
    while not terminated and not truncated and probe.step_index < EPISODE_STEP_CAP:
        valid = set(probe.valid_actions())
        action = next((candidate for candidate in FIXED_ACTION_PRIORITY if candidate in valid), None)
        if action is None:
            raise RuntimeError("fixed probe found no valid action")
        _, reward, terminated, truncated, _ = probe.step(action)
        actions.append(action)
        rewards.append(float(reward))
        discounted_return += discount * float(reward)
        discount *= gamma
    complete = bool(terminated or truncated or probe.step_index >= EPISODE_STEP_CAP)
    return {
        "target_semantics": TARGET_SEMANTICS,
        "complete_continuation": complete,
        "discounted_return": float(discounted_return),
        "continuation_actions": actions,
        "continuation_rewards": rewards,
        "continuation_steps": len(actions),
        "continuation_terminated": bool(terminated),
        "continuation_truncated": bool(truncated),
        "continuation_final_step": int(probe.step_index),
    }


def _state_conditions(env: CalibratedContinuousWorkflowEnv, info: dict[str, Any]) -> dict[str, Any]:
    semantic = info["semantic_state"]
    context = semantic.get("calibrated_context", {}) or {}
    deadline = float(env.instance["deadline_seconds"])
    return {
        "current_bundle_ready": bool(context.get("current_bundle_ready", False)),
        "target_bundle_ready": bool(context.get("target_bundle_ready", False)),
        "target_prepare_feasible": bool(info["action_mask"][4]),
        "deadline_status": "after" if float(env.clock_seconds) > deadline else "before_or_at",
        "clock_seconds": float(env.clock_seconds),
        "deadline_seconds": deadline,
        "completed_node_count": len(env.completed),
        "workflow_node_count": len(env.execution_order),
        "progress_fraction": len(env.completed) / max(len(env.execution_order), 1),
    }


def freeze_states(output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(f"create-only output already exists: {output_root}")
    design, config, dev = _load_protocol()
    output_root.mkdir(parents=True)
    gamma = float(config["training"]["gamma"])
    records: list[dict[str, Any]] = []
    for instance in dev:
        queue: deque[list[int]] = deque([[]])
        seen: set[str] = set()
        selected = 0
        while queue and selected < STATES_PER_INSTANCE:
            prefix = queue.popleft()
            env, observation, info = _replay_prefix(config, instance, prefix)
            payload = _public_payload(observation, info)
            state_hash = _stable_hash(payload)
            if state_hash in seen:
                continue
            seen.add(state_hash)
            forbidden = sorted(FORBIDDEN_PUBLIC_KEYS & _nested_keys(payload))
            if forbidden:
                raise RuntimeError(f"future actual-link keys leaked into public state: {forbidden}")
            probe = _fixed_probe_return(env, gamma)
            if not probe["complete_continuation"]:
                raise RuntimeError("fixed continuation did not reach a bounded episode boundary")
            selected += 1
            records.append(
                {
                    "state_id": f"{instance['design_id']}_state_{selected:02d}",
                    "design_id": instance["design_id"],
                    "window_id": instance["window_id"],
                    "source_interval": instance["source_interval"],
                    "selection_rule": "breadth_first_valid_action_prefix_ascending_0_to_4_first_unique_public_states",
                    "prefix_actions": prefix,
                    "prefix_depth": len(prefix),
                    "state_sha256": state_hash,
                    "public_state": payload,
                    "conditions": _state_conditions(env, info),
                    "probe": probe,
                }
            )
            if not env.terminated and env.step_index < min(EPISODE_STEP_CAP, 4):
                for action in sorted(env.valid_actions()):
                    queue.append(prefix + [int(action)])
        if selected != STATES_PER_INSTANCE:
            raise RuntimeError(
                f"could not freeze {STATES_PER_INSTANCE} states for {instance['design_id']}"
            )
    if len(records) != 48 or len({row["state_sha256"] for row in records}) != 48:
        raise RuntimeError("frozen state count or uniqueness mismatch")
    manifest = {
        "schema_version": "calibrated_workflow_selected_checkpoint_state_manifest_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "script_path": str(Path(__file__).resolve().relative_to(ROOT_DIR)),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "design_path": str(DESIGN_PATH.relative_to(ROOT_DIR)),
        "design_sha256": _sha256(DESIGN_PATH),
        "reward_path": str(SERVICE_REWARD_PATH.relative_to(ROOT_DIR)),
        "reward_sha256": _sha256(SERVICE_REWARD_PATH),
        "selection_outcome_blind": True,
        "checkpoint_or_evaluation_inputs_used": False,
        "future_actual_link_information_used": False,
        "state_count": len(records),
        "states_per_instance": STATES_PER_INSTANCE,
        "max_selection_prefix_depth": 4,
        "fixed_action_priority": list(FIXED_ACTION_PRIORITY),
        "gamma": gamma,
        "target_semantics": TARGET_SEMANTICS,
        "records": records,
    }
    _write_json(output_root / "state_manifest.json", manifest)
    _write_json(
        output_root / "freeze_receipt.json",
        {
            "state_manifest_sha256": _sha256(output_root / "state_manifest.json"),
            "state_count": 48,
            "unique_state_count": 48,
            "complete_probe_target_count": sum(
                bool(row["probe"]["complete_continuation"]) for row in records
            ),
            "checkpoint_or_evaluation_inputs_used": False,
        },
    )
    _integrity(
        output_root, "calibrated_workflow_selected_checkpoint_state_integrity_v1"
    )


def _checkpoint_identity(agent: Any) -> dict[str, Any]:
    return {
        "network": _state_dict_hash(agent._network.state_dict()),
        "optimizer": _state_dict_hash(agent._optimizer.state_dict()),
        "popart": _state_dict_hash(agent._popart.state_dict())
        if agent._popart is not None
        else "not_applicable",
        "update_count": int(agent._update_count),
    }


def _restore_state(
    config: dict[str, Any], instances: dict[str, dict[str, Any]], record: dict[str, Any]
) -> tuple[CalibratedContinuousWorkflowEnv, Any, dict[str, Any]]:
    env, observation, info = _replay_prefix(
        config, instances[str(record["design_id"])], list(record["prefix_actions"])
    )
    if _stable_hash(_public_payload(observation, info)) != record["state_sha256"]:
        raise RuntimeError(f"replayed state hash mismatch: {record['state_id']}")
    return env, observation, info


def _agent_row(
    *,
    agent: Any,
    arm: str,
    method: str,
    seed: int,
    selected_update: int,
    record: dict[str, Any],
    observation: Any,
    info: dict[str, Any],
) -> dict[str, Any]:
    decision_info = deepcopy(info)
    decision_info["deterministic_policy"] = True
    decision_info.setdefault("run_metadata", {})["policy_evaluation_mode"] = "raw_policy"
    with torch.no_grad():
        semantic = agent._extract_semantic_state(decision_info)
        output = agent._forward_policy(
            semantic, run_metadata=decision_info.get("run_metadata")
        )
        executed_action, action_info = agent.act(observation, decision_info)
    probs = [float(value) for value in action_info["env_action_probs"]]
    if len(probs) != 5 or not all(math.isfinite(value) for value in probs):
        raise RuntimeError("invalid five-action probability vector")
    probability_argmax = int(np.argmax(np.asarray(probs, dtype=np.float64)))
    best_other = max(probs[:4])
    value = float(output["value"].item())
    normalized_value = float(output["value_normalized"].item())
    if abs(value - float(action_info["value"])) > 1e-5:
        raise RuntimeError("act and direct-forward value mismatch")
    target = float(record["probe"]["discounted_return"])
    error = value - target
    conditions = record["conditions"]
    return {
        "arm": arm,
        "method": method,
        "seed": seed,
        "selected_update": selected_update,
        "state_id": record["state_id"],
        "state_sha256": record["state_sha256"],
        "design_id": record["design_id"],
        "prefix_depth": record["prefix_depth"],
        "current_bundle_ready": conditions["current_bundle_ready"],
        "target_bundle_ready": conditions["target_bundle_ready"],
        "target_prepare_feasible": conditions["target_prepare_feasible"],
        "deadline_status": conditions["deadline_status"],
        "progress_fraction": conditions["progress_fraction"],
        "target_semantics": TARGET_SEMANTICS,
        "fixed_behavior_discounted_return": target,
        "denormalized_value_prediction": value,
        "normalized_value_prediction": normalized_value,
        "fixed_behavior_value_error": error,
        "fixed_behavior_absolute_error": abs(error),
        "fixed_behavior_squared_error": error * error,
        "offline_fixed_behavior_advantage": target - value,
        "env_action_probs": json.dumps(probs, separators=(",", ":")),
        "action4_probability": probs[4],
        "action4_margin_vs_best_other": probs[4] - best_other,
        "probability_argmax_action": probability_argmax,
        "raw_env_action": int(action_info["raw_env_action"]),
        "projected_env_action": int(action_info["projected_env_action"]),
        "executed_action": int(executed_action),
        "action_projection_applied": bool(action_info["action_projection_applied"]),
        "guard_action_delta": bool(action_info["guard_action_delta"]),
    }


def _summarize(rows: list[dict[str, Any]], keys: dict[str, Any]) -> dict[str, Any]:
    n = len(rows)
    return {
        **keys,
        "state_count": n,
        "denormalized_value_mean": mean(row["denormalized_value_prediction"] for row in rows),
        "fixed_behavior_target_mean": mean(
            row["fixed_behavior_discounted_return"] for row in rows
        ),
        "fixed_behavior_rmse": math.sqrt(
            mean(row["fixed_behavior_squared_error"] for row in rows)
        ),
        "fixed_behavior_mae": mean(row["fixed_behavior_absolute_error"] for row in rows),
        "offline_fixed_behavior_advantage_mean": mean(
            row["offline_fixed_behavior_advantage"] for row in rows
        ),
        "action4_probability_mean": mean(row["action4_probability"] for row in rows),
        "action4_margin_mean": mean(row["action4_margin_vs_best_other"] for row in rows),
        "probability_argmax_action4_rate": mean(
            row["probability_argmax_action"] == 4 for row in rows
        ),
        "raw_env_action4_rate": mean(row["raw_env_action"] == 4 for row in rows),
        "projected_env_action4_rate": mean(
            row["projected_env_action"] == 4 for row in rows
        ),
        "executed_action4_rate": mean(row["executed_action"] == 4 for row in rows),
        "projection_rate": mean(row["action_projection_applied"] for row in rows),
    }


def _paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["arm"], row["method"], row["seed"], row["state_id"]): row
        for row in rows
    }
    paired: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in SEEDS:
            for state_id in sorted(
                row["state_id"]
                for row in rows
                if row["arm"] == CONTROL and row["method"] == method and row["seed"] == seed
            ):
                control = indexed[(CONTROL, method, seed, state_id)]
                candidate = indexed[(CANDIDATE, method, seed, state_id)]
                paired.append(
                    {
                        "method": method,
                        "seed": seed,
                        "state_id": state_id,
                        "state_sha256": control["state_sha256"],
                        "control_selected_update": control["selected_update"],
                        "candidate_selected_update": candidate["selected_update"],
                        "target_semantics": TARGET_SEMANTICS,
                        "fixed_behavior_discounted_return": control[
                            "fixed_behavior_discounted_return"
                        ],
                        "control_denormalized_value": control[
                            "denormalized_value_prediction"
                        ],
                        "candidate_denormalized_value": candidate[
                            "denormalized_value_prediction"
                        ],
                        "delta_denormalized_value": candidate[
                            "denormalized_value_prediction"
                        ]
                        - control["denormalized_value_prediction"],
                        "control_fixed_behavior_absolute_error": control[
                            "fixed_behavior_absolute_error"
                        ],
                        "candidate_fixed_behavior_absolute_error": candidate[
                            "fixed_behavior_absolute_error"
                        ],
                        "delta_fixed_behavior_absolute_error": candidate[
                            "fixed_behavior_absolute_error"
                        ]
                        - control["fixed_behavior_absolute_error"],
                        "control_action4_probability": control["action4_probability"],
                        "candidate_action4_probability": candidate["action4_probability"],
                        "delta_action4_probability": candidate["action4_probability"]
                        - control["action4_probability"],
                        "control_action4_margin": control["action4_margin_vs_best_other"],
                        "candidate_action4_margin": candidate[
                            "action4_margin_vs_best_other"
                        ],
                        "delta_action4_margin": candidate[
                            "action4_margin_vs_best_other"
                        ]
                        - control["action4_margin_vs_best_other"],
                        "control_probability_argmax": control[
                            "probability_argmax_action"
                        ],
                        "candidate_probability_argmax": candidate[
                            "probability_argmax_action"
                        ],
                        "probability_argmax_changed": control[
                            "probability_argmax_action"
                        ]
                        != candidate["probability_argmax_action"],
                        "control_raw_action": control["raw_env_action"],
                        "candidate_raw_action": candidate["raw_env_action"],
                        "raw_action_changed": control["raw_env_action"]
                        != candidate["raw_env_action"],
                        "control_projected_action": control["projected_env_action"],
                        "candidate_projected_action": candidate["projected_env_action"],
                        "projected_action_changed": control["projected_env_action"]
                        != candidate["projected_env_action"],
                        "control_executed_action": control["executed_action"],
                        "candidate_executed_action": candidate["executed_action"],
                        "executed_action_changed": control["executed_action"]
                        != candidate["executed_action"],
                    }
                )
    return paired


def _plot_figure(
    output_root: Path,
    method_rows: list[dict[str, Any]],
    service_rows: list[dict[str, Any]],
) -> None:
    labels = ["SA-GHMAPPO", "MAPPO", "PPO"]
    by_method_arm = {
        (row["method"], row["arm"]): row for row in method_rows
    }
    service = {row["method"]: row for row in service_rows}
    raw_rmse = [
        by_method_arm[(method, CONTROL)]["fixed_behavior_rmse"] for method in METHODS
    ]
    popart_rmse = [
        by_method_arm[(method, CANDIDATE)]["fixed_behavior_rmse"] for method in METHODS
    ]
    p4_delta = [
        by_method_arm[(method, CANDIDATE)]["action4_probability_mean"]
        - by_method_arm[(method, CONTROL)]["action4_probability_mean"]
        for method in METHODS
    ]
    completion_delta = [float(service[method]["delta_completion"]) for method in METHODS]
    on_time_delta = [float(service[method]["delta_on_time"]) for method in METHODS]

    width, height = 1200, 410
    panel_width = 360
    panel_lefts = (45, 430, 815)
    plot_top, plot_bottom = 80, 315
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1a202c}.title{font-size:18px;font-weight:700}.panel{font-size:14px;font-weight:700}.axis{font-size:11px}.note{font-size:10px;fill:#4a5568}</style>',
        '<text x="600" y="28" text-anchor="middle" class="title">PopArt mechanism-to-service alignment at selected checkpoints</text>',
    ]

    def text(x: float, y: float, value: str, css: str = "axis", anchor: str = "middle") -> None:
        elements.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="{css}">{html.escape(value)}</text>'
        )

    def axes(panel: int, title: str, ylabel: str, minimum: float, maximum: float) -> tuple[float, float, float]:
        left = panel_lefts[panel]
        right = left + panel_width
        maximum = maximum if maximum > minimum else minimum + 1.0
        elements.append(f'<line x1="{left}" y1="{plot_bottom}" x2="{right}" y2="{plot_bottom}" stroke="#2d3748"/>')
        elements.append(f'<line x1="{left}" y1="{plot_top}" x2="{left}" y2="{plot_bottom}" stroke="#2d3748"/>')
        for fraction in (0.0, 0.5, 1.0):
            y = plot_bottom - fraction * (plot_bottom - plot_top)
            value = minimum + fraction * (maximum - minimum)
            elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e2e8f0"/>')
            text(left - 7, y + 4, f"{value:.3g}", anchor="end")
        text((left + right) / 2, 58, title, "panel")
        text(left - 36, (plot_top + plot_bottom) / 2, ylabel, "axis")
        return left, minimum, maximum

    left, ymin, ymax = axes(0, "(a) Selected-checkpoint critic probe", "fixed-behavior RMSE", 0.0, max(raw_rmse + popart_rmse) * 1.08)
    for index, label in enumerate(labels):
        center = left + 65 + index * 105
        for offset, value, color in ((-15, raw_rmse[index], "#7f8c8d"), (15, popart_rmse[index], "#2b6cb0")):
            bar_height = (value - ymin) / (ymax - ymin) * (plot_bottom - plot_top)
            elements.append(f'<rect x="{center + offset - 12:.1f}" y="{plot_bottom - bar_height:.1f}" width="24" height="{bar_height:.1f}" fill="{color}"/>')
        text(center, plot_bottom + 20, label)
    elements.append('<rect x="70" y="345" width="12" height="12" fill="#7f8c8d"/><text x="88" y="355" class="axis">Raw critic</text>')
    elements.append('<rect x="165" y="345" width="12" height="12" fill="#2b6cb0"/><text x="183" y="355" class="axis">PopArt</text>')

    p4_extent = max(max(abs(value) for value in p4_delta) * 1.15, 0.01)
    left, ymin, ymax = axes(1, "(b) Fixed-state policy shift", "PopArt − raw P(action 4)", -p4_extent, p4_extent)
    zero_y = plot_bottom - (0.0 - ymin) / (ymax - ymin) * (plot_bottom - plot_top)
    elements.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + panel_width}" y2="{zero_y:.1f}" stroke="#1a202c"/>')
    for index, (label, value) in enumerate(zip(labels, p4_delta)):
        center = left + 65 + index * 105
        value_y = plot_bottom - (value - ymin) / (ymax - ymin) * (plot_bottom - plot_top)
        y = min(value_y, zero_y)
        elements.append(f'<rect x="{center - 18}" y="{y:.1f}" width="36" height="{abs(value_y - zero_y):.1f}" fill="{"#2b6cb0" if value <= 0 else "#c53030"}"/>')
        text(center, plot_bottom + 20, label)

    service_extent = max(max(abs(value) for value in completion_delta + on_time_delta) * 1.15, 0.01)
    left, ymin, ymax = axes(2, "(c) Existing service outcome", "PopArt − raw rate", -service_extent, service_extent)
    zero_y = plot_bottom - (0.0 - ymin) / (ymax - ymin) * (plot_bottom - plot_top)
    elements.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + panel_width}" y2="{zero_y:.1f}" stroke="#1a202c"/>')
    for index, label in enumerate(labels):
        center = left + 65 + index * 105
        for offset, value, color in ((-15, completion_delta[index], "#2f855a"), (15, on_time_delta[index], "#dd6b20")):
            value_y = plot_bottom - (value - ymin) / (ymax - ymin) * (plot_bottom - plot_top)
            y = min(value_y, zero_y)
            elements.append(f'<rect x="{center + offset - 12:.1f}" y="{y:.1f}" width="24" height="{abs(value_y - zero_y):.1f}" fill="{color}"/>')
        text(center, plot_bottom + 20, label)
    elements.append('<rect x="845" y="345" width="12" height="12" fill="#2f855a"/><text x="863" y="355" class="axis">Completion</text>')
    elements.append('<rect x="945" y="345" width="12" height="12" fill="#dd6b20"/><text x="963" y="355" class="axis">On-time</text>')
    text(600, 392, "Panel (a) uses a complete fixed-behavior discounted-return probe; it is not an unbiased Vπ estimate.", "note")
    elements.append("</svg>")
    (output_root / "selected_checkpoint_alignment_mechanism.svg").write_text(
        "\n".join(elements) + "\n", encoding="utf-8"
    )


def forward_selected(state_root: Path, run_root: Path, output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(f"create-only output already exists: {output_root}")
    design, config, dev = _load_protocol()
    manifest_path = state_root / "state_manifest.json"
    state_manifest = _read_json(manifest_path)
    if state_manifest["state_count"] != 48:
        raise RuntimeError("frozen state count mismatch")
    if state_manifest["script_sha256"] != _sha256(Path(__file__).resolve()):
        raise RuntimeError("analysis script changed after state freeze")
    if state_manifest["checkpoint_or_evaluation_inputs_used"]:
        raise RuntimeError("state selection was not outcome blind")
    records = list(state_manifest["records"])
    instances = {str(row["design_id"]): row for row in dev}
    summaries = _read_json(run_root / "training_summary.json")
    if len(summaries) != 30:
        raise RuntimeError("expected 30 selected checkpoints")
    output_root.mkdir(parents=True)
    forward_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    for summary in sorted(
        summaries, key=lambda row: (row["arm"], row["method"], int(row["seed"]))
    ):
        arm = str(summary["arm"])
        method = str(summary["method"])
        seed = int(summary["seed"])
        popart = arm == CANDIDATE
        checkpoint = run_root / summary["selected_checkpoint"]
        if _sha256(checkpoint) != summary["selected_checkpoint_sha256"]:
            raise RuntimeError(f"selected checkpoint hash mismatch: {checkpoint}")
        torch.manual_seed(seed)
        agent = _build_agent(
            method,
            seed,
            config,
            agent_overrides={
                "deterministic_action": True,
                "value_normalization_enabled": popart,
                "popart_min_std": 1.0,
                "popart_max_abs_target": 1.0e20,
            },
        )
        agent.load(str(checkpoint))
        before = _checkpoint_identity(agent)
        if before["update_count"] != int(summary["selected_update"]):
            raise RuntimeError("selected update and checkpoint update_count disagree")
        for record in records:
            _, observation, info = _restore_state(config, instances, record)
            forward_rows.append(
                _agent_row(
                    agent=agent,
                    arm=arm,
                    method=method,
                    seed=seed,
                    selected_update=int(summary["selected_update"]),
                    record=record,
                    observation=observation,
                    info=info,
                )
            )
        after = _checkpoint_identity(agent)
        identity_rows.append(
            {
                "arm": arm,
                "method": method,
                "seed": seed,
                "selected_update": int(summary["selected_update"]),
                "selected_checkpoint": summary["selected_checkpoint"],
                "selected_checkpoint_sha256": summary["selected_checkpoint_sha256"],
                "network_sha256_before": before["network"],
                "network_sha256_after": after["network"],
                "network_unchanged": before["network"] == after["network"],
                "optimizer_sha256_before": before["optimizer"],
                "optimizer_sha256_after": after["optimizer"],
                "optimizer_unchanged": before["optimizer"] == after["optimizer"],
                "popart_sha256_before": before["popart"],
                "popart_sha256_after": after["popart"],
                "popart_unchanged": before["popart"] == after["popart"],
                "update_count_before": before["update_count"],
                "update_count_after": after["update_count"],
                "update_count_unchanged": before["update_count"] == after["update_count"],
            }
        )
    if len(forward_rows) != 1440:
        raise RuntimeError(f"expected 1440 forward rows, found {len(forward_rows)}")
    if not all(
        row["network_unchanged"]
        and row["optimizer_unchanged"]
        and row["popart_unchanged"]
        and row["update_count_unchanged"]
        for row in identity_rows
    ):
        raise RuntimeError("read-only identity invariant failed")
    pair_rows = _paired_rows(forward_rows)
    method_seed_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    for arm in (CONTROL, CANDIDATE):
        for method in METHODS:
            for seed in SEEDS:
                subset = [
                    row
                    for row in forward_rows
                    if row["arm"] == arm and row["method"] == method and row["seed"] == seed
                ]
                method_seed_rows.append(_summarize(subset, {"arm": arm, "method": method, "seed": seed}))
            subset = [
                row
                for row in forward_rows
                if row["arm"] == arm and row["method"] == method
            ]
            method_rows.append(_summarize(subset, {"arm": arm, "method": method}))
    service_rows_path = (
        ROOT_DIR
        / "artifacts/analysis/calibrated_workflow_value_normalization_ab_review_20261009_v3/service_method_summary.csv"
    )
    service_rows = list(csv.DictReader(service_rows_path.open(encoding="utf-8", newline="")))
    _write_csv(output_root / "fixed_state_forward_rows.csv", forward_rows)
    _write_csv(output_root / "fixed_state_pair_rows.csv", pair_rows)
    _write_csv(output_root / "checkpoint_identity.csv", identity_rows)
    _write_csv(output_root / "method_seed_summary.csv", method_seed_rows)
    _write_csv(output_root / "method_summary.csv", method_rows)
    _plot_figure(output_root, method_rows, service_rows)
    selected_before_final = sum(int(row["selected_update"]) < 24 for row in summaries)
    _write_json(
        output_root / "alignment_summary.json",
        {
            "schema_version": "calibrated_workflow_selected_checkpoint_alignment_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "source_run": str(run_root.relative_to(ROOT_DIR)),
            "source_run_completion_receipt_sha256": _sha256(
                run_root / "completion_receipt.json"
            ),
            "state_manifest": str(manifest_path.relative_to(ROOT_DIR)),
            "state_manifest_sha256": _sha256(manifest_path),
            "target_semantics": TARGET_SEMANTICS,
            "target_is_unbiased_vpi": False,
            "gae_reconstructed": False,
            "optimizer_update_replayed": False,
            "checkpoint_count": len(identity_rows),
            "state_count": len(records),
            "forward_row_count": len(forward_rows),
            "paired_row_count": len(pair_rows),
            "selected_before_update_24_count": selected_before_final,
            "selected_at_update_24_count": 30 - selected_before_final,
            "network_optimizer_popart_update_count_all_unchanged": True,
            "method_summary": method_rows,
        },
    )
    _integrity(
        output_root, "calibrated_workflow_selected_checkpoint_alignment_integrity_v1"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-states", action="store_true")
    mode.add_argument("--forward", action="store_true")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--state-root")
    parser.add_argument("--run-root")
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    if args.freeze_states:
        freeze_states(output_root)
        return
    if not args.state_root or not args.run_root:
        parser.error("--forward requires --state-root and --run-root")
    forward_selected(
        Path(args.state_root).resolve(), Path(args.run_root).resolve(), output_root
    )


if __name__ == "__main__":
    main()
