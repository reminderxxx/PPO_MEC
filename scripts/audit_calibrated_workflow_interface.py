"""Reproduce the calibrated-pilot interface defects without changing the pilot.

The audit loads the historical environment source from a fixed Git commit and
the preserved v1 checkpoint.  It writes a create-only diagnostic bundle; it
does not train, download data, open holdout data, or mutate historical output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.registry import build_agent
from src.agents.sa_ghmappo_core import 聚合层级动作
from src.encoders import FlatSemanticEncoder, SurrogateFusionEncoder


HISTORICAL_COMMIT = "b418eb4dbfa012e271cb97880693c714e6abf79e"
CONFIG_PATH = Path("configs/experiment/calibrated_continuous_workflow_pilot_v1.json")
MANIFEST_PATH = Path("configs/experiment/calibrated_continuous_workflow_pilot_v1_manifest.json")
CHECKPOINT_PATH = Path(
    "artifacts/benchmarks/calibrated_continuous_workflow_pilot_20261006_v1/"
    "checkpoints/sa_ghmappo_seed7.pt"
)
FAILURE_DESIGN_ID = "evaluation_00"
STOCHASTIC_SEEDS = tuple(range(16))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads((ROOT_DIR / path).read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _historical_env_class(commit: str) -> type[Any]:
    source = subprocess.run(
        ["git", "show", f"{commit}:src/envs/core/calibrated_continuous_workflow_env.py"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module_name = f"historical_calibrated_env_{commit[:12]}"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    exec(compile(source, module_name, "exec"), module.__dict__)
    return module.CalibratedContinuousWorkflowEnv


def _build_agent(*, deterministic: bool) -> Any:
    agent = build_agent(
        "sa_ghmappo",
        random_seed=7,
        learning_rate=3e-4,
        clip_ratio=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        batch_size=32,
        train_epochs=4,
        deterministic_action=deterministic,
    )
    agent.load(str(ROOT_DIR / CHECKPOINT_PATH))
    agent._deterministic_action = deterministic
    return agent


def _run_episode(
    env_class: type[Any],
    config: dict[str, Any],
    instance: dict[str, Any],
    *,
    sample_seed: int,
    deterministic: bool,
) -> dict[str, Any]:
    random.seed(sample_seed)
    np.random.seed(sample_seed)
    torch.manual_seed(sample_seed)
    agent = _build_agent(deterministic=deterministic)
    env = env_class(config, instance)
    observation, info = env.reset()
    actions: list[int] = []
    while not env.terminated and env.step_index < int(instance["max_steps"]):
        action, _ = agent.act(observation, info)
        actions.append(int(action))
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    summary = env.summary()
    return {
        "sample_seed": sample_seed,
        "deterministic": deterministic,
        "workflow_completed": int(summary["workflow_completed"]),
        "completed_nodes": int(summary["completed_nodes"]),
        "service_failures": int(summary["service_failures"]),
        "actions": actions,
    }


def _action4_trace(
    env_class: type[Any],
    config: dict[str, Any],
    instance: dict[str, Any],
) -> dict[str, Any]:
    agent = _build_agent(deterministic=True)
    env = env_class(config, instance)
    observation, info = env.reset()
    rows: list[dict[str, Any]] = []
    while not env.terminated and env.step_index < int(instance["max_steps"]):
        semantic = info["semantic_state"]
        node = semantic["current_workflow_node"]
        current_rsu = env._current_rsu_id()
        target_rsu = env._predicted_handoff_target()
        adapter_id = str(node["required_adapter"])
        with torch.no_grad():
            output = agent._forward_policy(
                semantic,
                run_metadata={"policy_evaluation_mode": "raw_policy"},
            )
            raw_logits = {
                head: [round(float(value), 6) for value in output[f"{head}_logits"].tolist()]
                for head in ("slow", "fast", "event")
            }
            head_probs = {
                head: [
                    round(float(value), 6)
                    for value in torch.softmax(output[f"{head}_logits"], dim=-1).tolist()
                ]
                for head in ("slow", "fast", "event")
            }
            head_argmax = {
                head: int(torch.argmax(output[f"{head}_logits"]).item())
                for head in ("slow", "fast", "event")
            }
            head_argmax_action, head_argmax_reason = 聚合层级动作(
                head_actions=head_argmax,
                use_hierarchy=True,
                event_head_enabled=True,
                adapter_prefetch_enabled=True,
            )
            env_scores = agent._masked_flat_logits(
                agent._hierarchical_env_action_scores(output),
                info["action_mask"],
            )
            env_probs = [
                round(float(value), 6)
                for value in torch.softmax(env_scores, dim=-1).tolist()
            ]
        action, action_info = agent.act(observation, info)
        before = {
            "node_index": env.node_index,
            "step_index": env.step_index,
            "clock_seconds": env.clock_seconds,
            "completed_node_ids": list(env.completed),
            "current_bundle_ready": env._bundle_ready(current_rsu, adapter_id),
            "target_bundle_ready": (
                env._bundle_ready(target_rsu, adapter_id) if target_rsu else None
            ),
        }
        observation, reward, terminated, truncated, next_info = env.step(action)
        transition = next_info["transition"]
        rows.append(
            {
                "step": before["step_index"],
                "node_id": str(node["node_id"]),
                "required_base_model": str(node["required_base_model"]),
                "required_adapter": adapter_id,
                "current_rsu_id": current_rsu,
                "target_rsu_id": target_rsu,
                "before": before,
                "action_mask": list(info["action_mask"]),
                "raw_actor_logits": raw_logits,
                "raw_head_probabilities": head_probs,
                "independent_head_argmax": head_argmax,
                "independent_head_argmax_action": int(head_argmax_action),
                "independent_head_argmax_reason": head_argmax_reason,
                "aggregated_env_action_probabilities": env_probs,
                "reported_raw_head_actions": action_info["raw_head_actions"],
                "reported_projected_head_actions": action_info["projected_head_actions"],
                "raw_env_action": int(action_info["raw_env_action"]),
                "projected_env_action": int(action_info["projected_env_action"]),
                "executed_action": int(action),
                "aggregation_reason": action_info["aggregation_reason"],
                "projection_applied": bool(action_info["action_projection_applied"]),
                "head_log_prob": float(action_info["log_prob"]),
                "env_action_log_prob": float(action_info["env_action_log_prob"]),
                "cache_events": transition["cache_events"],
                "state_transfer": transition["state_transfer"],
                "service_completed": bool(transition["service_completed"]),
                "migration_success": bool(transition["migration_success"]),
                "reward": float(reward),
                "after": {
                    "node_index": env.node_index,
                    "step_index": env.step_index,
                    "clock_seconds": env.clock_seconds,
                    "completed_node_ids": list(env.completed),
                },
            }
        )
        info = next_info
        if terminated or truncated:
            break
    return {
        "instance": {
            field: instance[field]
            for field in ("design_id", "window_id", "workflow_id", "max_steps")
        },
        "checkpoint": str(CHECKPOINT_PATH),
        "trace": rows,
        "summary": env.summary(),
    }


def _feature_vector(encoder: Any, semantic: dict[str, Any]) -> torch.Tensor:
    if isinstance(encoder, FlatSemanticEncoder):
        return torch.cat(
            [
                encoder._build_feature_tensor(semantic),
                encoder._build_centralized_feature_tensor(semantic),
            ]
        ).detach()
    output = encoder(semantic)
    return torch.cat(
        [
            output[key].flatten()
            for key in (
                "shared_embedding",
                "slow_context",
                "fast_context",
                "event_context",
                "critic_context",
            )
        ]
    ).detach()


def _feature_perturbations(
    env_class: type[Any],
    config: dict[str, Any],
    instance: dict[str, Any],
) -> dict[str, Any]:
    agent = _build_agent(deterministic=True)
    env = env_class(config, instance)
    observation, info = env.reset()
    for _ in range(3):
        action, _ = agent.act(observation, info)
        observation, _, _, _, info = env.step(action)
    semantic = info["semantic_state"]
    encoders = {
        "flat": FlatSemanticEncoder(),
        "graph": SurrogateFusionEncoder(),
    }

    def set_link(state: dict[str, Any]) -> None:
        state["calibrated_context"]["link"]["mbps"] = 17.0

    mutations: dict[str, Callable[[dict[str, Any]], None]] = {
        "cache_used_bytes_only": lambda state: state["rsus"][0].__setitem__(
            "cache_used_bytes", 0
        ),
        "cache_capacity_bytes_only": lambda state: state["rsus"][0].__setitem__(
            "cache_capacity", state["rsus"][0]["cache_capacity"] * 2
        ),
        "typed_resident_identity_only": lambda state: state["rsus"][0].__setitem__(
            "typed_resident_object_ids", ["base:family_b", "adapter:alpr"]
        ),
        "calibrated_link_only": set_link,
        "calibrated_state_bytes_only": lambda state: state["calibrated_context"].__setitem__(
            "state_bytes", 999_999_999
        ),
        "node_required_base_only": lambda state: state["current_workflow_node"].__setitem__(
            "required_base_model", "base:family_b"
        ),
        "current_cached_adapter_remove": lambda state: state["rsus"][0].__setitem__(
            "cached_adapter_ids", []
        ),
        "target_cached_adapter_remove": lambda state: state["rsus"][1].__setitem__(
            "cached_adapter_ids", []
        ),
        "public_dwell_time": lambda state: state["predictions"]["dwell_time"].__setitem__(
            "veh_pilot", 19.0
        ),
    }
    results: dict[str, Any] = {}
    for name, mutation in mutations.items():
        results[name] = {}
        for encoder_name, encoder in encoders.items():
            before = _feature_vector(encoder, semantic)
            changed = copy.deepcopy(semantic)
            mutation(changed)
            after = _feature_vector(encoder, changed)
            delta = after - before
            results[name][encoder_name] = {
                "max_abs_delta": float(torch.max(torch.abs(delta)).item()),
                "l2_delta": float(torch.linalg.vector_norm(delta).item()),
            }
    return {
        "state": {
            "design_id": instance["design_id"],
            "node_id": semantic["current_workflow_node"]["node_id"],
            "step_index": semantic["time_index"],
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical_commit", default=HISTORICAL_COMMIT)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output already exists: {output_root}")
    output_root.mkdir(parents=True)

    config = _read_json(CONFIG_PATH)
    manifest = _read_json(MANIFEST_PATH)
    instance = next(
        row for row in manifest["instances"] if row["design_id"] == FAILURE_DESIGN_ID
    )
    env_class = _historical_env_class(args.historical_commit)

    trace = _action4_trace(env_class, config, instance)
    stochastic_rows = [
        _run_episode(
            env_class,
            config,
            instance,
            sample_seed=seed,
            deterministic=False,
        )
        for seed in STOCHASTIC_SEEDS
    ]
    deterministic_row = _run_episode(
        env_class,
        config,
        instance,
        sample_seed=7,
        deterministic=True,
    )
    stochastic_deterministic = {
        "instance": FAILURE_DESIGN_ID,
        "fixed_stochastic_seeds": list(STOCHASTIC_SEEDS),
        "stochastic_completion_rate": float(
            np.mean([row["workflow_completed"] for row in stochastic_rows])
        ),
        "stochastic_rows": stochastic_rows,
        "deterministic_row": deterministic_row,
        "interpretation_boundary": (
            "Checkpoint diagnosis only. The repeated stochastic samples are not independent "
            "episodes, not a new method score, and not a final evaluation."
        ),
    }
    features = _feature_perturbations(env_class, config, instance)
    permission_audit = {
        "two_step_rule_capability": "explicit model-based planner with environment clone access",
        "public_inputs": [
            "full DAG and current node",
            "predicted RSU sequence and confidence",
            "typed cache residents and object catalog",
            "estimated link and measured cost parameters",
        ],
        "additional_computational_capability": [
            "enumerates legal actions by executing exact environment transition clones",
            "observes modeled completion, failure, transfer and completion outcomes for candidate branches",
            "uses a lexicographic objective rather than the scalar RL reward",
        ],
        "leakage_judgment": (
            "The historical v1 clone used the same configured link because no actual/estimated "
            "split existed. It did not open holdout truth, but its exact transition-model and "
            "lexicographic planning capability is stronger than the learned policies' encoder. "
            "It must be labeled a strong model-based rule, not equal information-use capacity."
        ),
    }
    compatibility = {
        "historical_checkpoint": str(CHECKPOINT_PATH),
        "historical_observation_contract": "semantic_state_info_v1 with graph/flat encoder v1",
        "structural_load_into_current_code": True,
        "semantic_comparability_after_encoder_profile_fix": False,
        "reason": (
            "A calibrated encoder profile changes the meaning of existing fixed-size features "
            "to consume byte occupancy, typed bundle readiness, model/state sizes and estimated "
            "link/contact cost. A historical checkpoint may still deserialize by tensor shape, "
            "but its weights were never trained for those semantics and must not be compared as "
            "a compatible repaired policy."
        ),
        "required_action": "retrain all learned methods under the frozen repaired profile",
    }

    _write_json(output_root / "action4_failure_trace.json", trace)
    _write_json(output_root / "sampling_mode_diagnosis.json", stochastic_deterministic)
    _write_json(output_root / "feature_perturbation_audit.json", features)
    _write_json(output_root / "rule_permission_audit.json", permission_audit)
    _write_json(output_root / "checkpoint_compatibility.json", compatibility)
    receipt = {
        "schema_version": "calibrated_workflow_interface_diagnosis_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "historical_git_commit": args.historical_commit,
        "config": {"path": str(CONFIG_PATH), "sha256": _sha256(ROOT_DIR / CONFIG_PATH)},
        "manifest": {"path": str(MANIFEST_PATH), "sha256": _sha256(ROOT_DIR / MANIFEST_PATH)},
        "checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "sha256": _sha256(ROOT_DIR / CHECKPOINT_PATH),
        },
        "downloads": 0,
        "real_model_calls": 0,
        "holdout_calls": 0,
        "training_episodes": 0,
        "fixed_failure_instance": FAILURE_DESIGN_ID,
        "fixed_stochastic_repetitions": len(STOCHASTIC_SEEDS),
    }
    _write_json(output_root / "completion_receipt.json", receipt)
    files = sorted(
        path for path in output_root.iterdir() if path.is_file() and path.name != "artifact_integrity.json"
    )
    _write_json(
        output_root / "artifact_integrity.json",
        {
            "schema_version": "artifact_integrity_v1",
            "files": [
                {
                    "path": path.name,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
            ],
        },
    )
    print(json.dumps({"status": "complete", "output_root": str(output_root)}))


if __name__ == "__main__":
    main()
