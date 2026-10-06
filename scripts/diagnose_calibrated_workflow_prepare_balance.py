"""Diagnose SA prepare/execution balance on frozen development states.

This script is diagnostic only.  It reuses the repaired-interface SA checkpoints,
selects at most twelve states by a fixed state-property rule, and records the
logit/probability path without changing environment state or training weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.freeze_calibrated_continuous_workflow_pilot import (  # noqa: E402
    _canonical_sha256,
    _load_experiment_config,
)
from scripts.run_calibrated_workflow_interface_repair import (  # noqa: E402
    _build_agent,
    _load_json,
)
from src.agents.sa_ghmappo_core import 聚合层级动作  # noqa: E402
from src.encoders.calibrated_workflow_features import (  # noqa: E402
    bundle_ready,
    predicted_target_rsu_id,
    rsu_by_id,
)
from src.envs.core.calibrated_continuous_workflow_env import (  # noqa: E402
    CalibratedContinuousWorkflowEnv,
)


SEEDS = (7, 17, 29)
STRATUM_ORDER = (
    (False, True, False),
    (False, True, True),
    (False, False, False),
    (False, False, True),
    (True, True, False),
    (True, True, True),
    (True, False, False),
    (True, False, True),
)
MAX_STATES = 12
PER_STRATUM_QUOTA = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clone_prepare_feasibility(
    env: CalibratedContinuousWorkflowEnv,
    action_mask: list[bool],
) -> tuple[bool, str]:
    if len(action_mask) <= 4 or not action_mask[4]:
        return False, "action_4_masked_missing_distinct_target"
    clone = env.clone_for_decision_model()
    _, _, _, _, next_info = clone.step(4)
    transition = dict(next_info.get("transition", {}) or {})
    cache_events = list(transition.get("cache_events", []) or [])
    if not cache_events:
        return False, "action_4_no_cache_event"
    reason = str(cache_events[0].get("reason", ""))
    feasible = reason != "contact_budget_exceeded"
    return feasible, reason or "unknown"


def _stage_payload(
    agent: Any,
    output: dict[str, Any],
    action_mask: list[bool],
) -> dict[str, Any]:
    selected, _, _, head_probs, projection = agent._sample_actions(
        output,
        deterministic=True,
        action_mask=action_mask,
    )
    env_action, reason = 聚合层级动作(
        head_actions=selected,
        use_hierarchy=agent._use_hierarchy,
        event_head_enabled=agent._event_head_enabled,
        adapter_prefetch_enabled=agent._adapter_prefetch_enabled,
    )
    scores = agent._masked_flat_logits(
        agent._hierarchical_env_action_scores(output),
        action_mask,
    )
    env_probs = torch.softmax(scores, dim=-1)
    return {
        "slow_logits": [round(float(value), 6) for value in output["slow_logits"].tolist()],
        "fast_logits": [round(float(value), 6) for value in output["fast_logits"].tolist()],
        "event_logits": [round(float(value), 6) for value in output["event_logits"].tolist()],
        "head_probs": head_probs,
        "env_action_probs": [round(float(value), 6) for value in env_probs.tolist()],
        "head_actions": {key: int(value) for key, value in selected.items()},
        "selected_action": int(env_action),
        "aggregation_reason": reason,
        "projection_applied": bool(projection.get("projection_applied", False)),
    }


def _event_stages(
    agent: Any,
    semantic_state: dict[str, Any],
    action_mask: list[bool],
) -> dict[str, dict[str, Any]]:
    with torch.no_grad():
        raw_untempered = agent._network.forward_single(
            semantic_state,
            event_logit_temperature=1.0,
        )
        temperature_only = agent._network.forward_single(
            semantic_state,
            event_logit_temperature=agent._current_event_logit_temperature(),
        )

        original_scale = agent._event_logit_sharpening_final_scale
        original_margin = agent._event_prepare_margin_boost
        agent._event_logit_sharpening_final_scale = 1.0
        agent._event_prepare_margin_boost = 0.0
        try:
            before_event_enhancement = agent._apply_policy_adjustments(
                temperature_only,
                semantic_state,
                run_metadata={"policy_evaluation_mode": "safety_projected"},
            )
        finally:
            agent._event_logit_sharpening_final_scale = original_scale
            agent._event_prepare_margin_boost = original_margin

        timing_features = {
            key: float(value)
            for key, value in agent._build_mechanism_targets(semantic_state).items()
            if key in {"temporal_urgency", "prepare_window_score", "handoff_countdown_steps"}
        }
        scaling = agent._build_event_scaling_summary(
            semantic_state=semantic_state,
            timing_features=timing_features,
        )
        timing_support = max(
            timing_features.get("prepare_window_score", 0.0),
            timing_features.get("temporal_urgency", 0.0),
        )
        sharpen_scale = 1.0 + (
            agent._current_event_logit_sharpening_scale() - 1.0
        ) * (1.0 + agent._event_logit_sharpening_timing_gain * timing_support) * float(
            scaling["event_sharpen_factor"]
        )
        margin_boost = agent._compute_event_prepare_margin_boost(
            semantic_state=semantic_state,
            timing_features=timing_features,
        )
        sharpened = dict(before_event_enhancement)
        sharpened_logits = before_event_enhancement["event_logits"].clone()
        center = sharpened_logits.mean()
        sharpened_logits = (sharpened_logits - center) * sharpen_scale + center
        sharpened["event_logits"] = sharpened_logits

        full = dict(sharpened)
        full_logits = sharpened_logits.clone()
        full_logits[1] = full_logits[1] + margin_boost
        full_logits[0] = full_logits[0] - 0.25 * margin_boost
        full["event_logits"] = full_logits

    return {
        "raw_untempered": _stage_payload(agent, raw_untempered, action_mask),
        "temperature_only": _stage_payload(agent, temperature_only, action_mask),
        "before_event_enhancement": _stage_payload(
            agent, before_event_enhancement, action_mask
        ),
        "after_sharpening_no_margin": _stage_payload(agent, sharpened, action_mask),
        "after_sharpening_and_margin": _stage_payload(agent, full, action_mask),
        "stage_parameters": {
            "active_temperature": round(agent._current_event_logit_temperature(), 6),
            "sharpen_scale": round(float(sharpen_scale), 6),
            "margin_boost": round(float(margin_boost), 6),
        },
    }


def _collect_pool(
    config: dict[str, Any],
    instances: list[dict[str, Any]],
    checkpoint_root: Path,
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    step_cap = int(config["training"]["budget"]["episode_max_steps"])
    for seed in SEEDS:
        agent = _build_agent("sa_ghmappo", seed, config)
        checkpoint = checkpoint_root / f"sa_ghmappo_seed{seed}_selected.pt"
        agent.load(str(checkpoint))
        agent._deterministic_action = True
        for instance_index, instance in enumerate(instances):
            env = CalibratedContinuousWorkflowEnv(config, instance)
            observation, info = env.reset()
            while not env.terminated and env.step_index < step_cap:
                semantic_state = info["semantic_state"]
                action_mask = list(info["action_mask"])
                node = semantic_state.get("current_workflow_node") or {}
                vehicle = (semantic_state.get("vehicles") or [{}])[0]
                current_rsu_id = vehicle.get("associated_rsu_id")
                current_ready = bool(
                    bundle_ready(
                        semantic_state,
                        rsu_by_id(semantic_state, current_rsu_id),
                        node,
                    )
                )
                prepare_feasible, feasibility_reason = _clone_prepare_feasibility(
                    env, action_mask
                )
                action, action_info = agent.act(observation, info)
                near_handoff = bool(
                    float(action_info.get("handoff_countdown_steps", 999.0)) <= 2.5
                )
                pseudo_targets = agent._build_mechanism_targets(semantic_state)
                pool.append(
                    {
                        "sort_key": (
                            int(seed),
                            int(instance_index),
                            int(env.step_index),
                        ),
                        "seed": int(seed),
                        "design_id": str(instance["design_id"]),
                        "step_index": int(env.step_index),
                        "current_rsu_id": current_rsu_id,
                        "target_rsu_id": predicted_target_rsu_id(semantic_state),
                        "current_bundle_ready": current_ready,
                        "near_handoff": near_handoff,
                        "target_prepare_feasible": prepare_feasible,
                        "prepare_feasibility_reason": feasibility_reason,
                        "action_mask": action_mask,
                        "executed_raw_policy_action": int(action),
                        "executed_raw_policy_event_prepare_prob": float(
                            action_info.get("event_prepare_prob", 0.0)
                        ),
                        "executed_raw_policy_event_margin": float(
                            action_info.get("event_margin", 0.0)
                        ),
                        "raw_policy_evaluation": bool(
                            action_info.get("raw_policy_evaluation", False)
                        ),
                        "runtime_event_sharpening_info": dict(
                            action_info.get("event_sharpening_info", {})
                        ),
                        "pseudo_targets": {
                            key: float(value) if isinstance(value, float) else int(value)
                            for key, value in pseudo_targets.items()
                        },
                        "stages": _event_stages(
                            agent,
                            semantic_state,
                            action_mask,
                        ),
                    }
                )
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
    return pool


def _select_states(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[bool, bool, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(pool, key=lambda item: item["sort_key"]):
        key = (
            bool(row["current_bundle_ready"]),
            bool(row["near_handoff"]),
            bool(row["target_prepare_feasible"]),
        )
        grouped[key].append(row)
    selected: list[dict[str, Any]] = []
    for key in STRATUM_ORDER:
        selected.extend(grouped[key][:PER_STRATUM_QUOTA])
        if len(selected) >= MAX_STATES:
            break
    if len(selected) < MAX_STATES:
        selected_ids = {id(row) for row in selected}
        for row in sorted(pool, key=lambda item: item["sort_key"]):
            if id(row) in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(id(row))
            if len(selected) >= MAX_STATES:
                break
    return selected[:MAX_STATES]


def _summarize(pool: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    pool_strata = Counter(
        (
            bool(row["current_bundle_ready"]),
            bool(row["near_handoff"]),
            bool(row["target_prepare_feasible"]),
        )
        for row in pool
    )
    selected_strata = Counter(
        (
            bool(row["current_bundle_ready"]),
            bool(row["near_handoff"]),
            bool(row["target_prepare_feasible"]),
        )
        for row in selected
    )
    raw_actions = Counter(int(row["executed_raw_policy_action"]) for row in selected)
    full_actions = Counter(
        int(row["stages"]["after_sharpening_and_margin"]["selected_action"])
        for row in selected
    )
    no_margin_actions = Counter(
        int(row["stages"]["after_sharpening_no_margin"]["selected_action"])
        for row in selected
    )
    return {
        "pool_state_count": len(pool),
        "selected_state_count": len(selected),
        "selection_rule": {
            "source_split": "dev",
            "iteration_order": "seed_7_17_29_then_manifest_instance_order_then_step_index",
            "stratum_key": [
                "current_bundle_ready",
                "near_handoff_countdown_le_2_5",
                "target_prepare_feasible_by_action4_clone",
            ],
            "stratum_order": [list(key) for key in STRATUM_ORDER],
            "per_stratum_quota": PER_STRATUM_QUOTA,
            "quota_shortfall_fill": "earliest_unselected_state_in_iteration_order",
            "maximum_states": MAX_STATES,
            "logit_or_outcome_blind": True,
        },
        "pool_strata": {str(key): value for key, value in sorted(pool_strata.items())},
        "selected_strata": {
            str(key): value for key, value in sorted(selected_strata.items())
        },
        "runtime_path": {
            "all_selected_states_use_raw_policy_evaluation": all(
                bool(row["raw_policy_evaluation"]) for row in selected
            ),
            "all_runtime_event_sharpening_info_empty": all(
                not row["runtime_event_sharpening_info"] for row in selected
            ),
            "event_margin_and_sharpening_active_in_rollout_or_deterministic_eval": False,
            "event_temperature_active": True,
            "auxiliary_and_temporal_consistency_training_only": True,
        },
        "selected_action_counts": {
            "actual_raw_policy": dict(sorted(raw_actions.items())),
            "counterfactual_full_event_enhancement": dict(sorted(full_actions.items())),
            "counterfactual_no_margin": dict(sorted(no_margin_actions.items())),
        },
        "current_missing_prepare_targets": {
            "state_count": sum(not row["current_bundle_ready"] for row in selected),
            "hard_event_target_1_count": sum(
                (not row["current_bundle_ready"])
                and int(row["pseudo_targets"]["event_target"]) == 1
                for row in selected
            ),
            "actual_action4_count": sum(
                (not row["current_bundle_ready"])
                and int(row["executed_raw_policy_action"]) == 4
                for row in selected
            ),
        },
        "full_dev_trajectory_prepare_targets": {
            "current_missing_state_count": sum(
                not row["current_bundle_ready"] for row in pool
            ),
            "current_missing_hard_event_target_1_count": sum(
                (not row["current_bundle_ready"])
                and int(row["pseudo_targets"]["event_target"]) == 1
                for row in pool
            ),
            "current_missing_temporal_soft_target_gt_0_5_count": sum(
                (not row["current_bundle_ready"])
                and float(row["pseudo_targets"]["event_soft_target"]) > 0.5
                for row in pool
            ),
            "current_missing_action4_count": sum(
                (not row["current_bundle_ready"])
                and int(row["executed_raw_policy_action"]) == 4
                for row in pool
            ),
            "current_missing_infeasible_action4_count": sum(
                (not row["current_bundle_ready"])
                and (not row["target_prepare_feasible"])
                and int(row["executed_raw_policy_action"]) == 4
                for row in pool
            ),
            "current_ready_state_count": sum(
                bool(row["current_bundle_ready"]) for row in pool
            ),
            "current_ready_hard_event_target_1_count": sum(
                bool(row["current_bundle_ready"])
                and int(row["pseudo_targets"]["event_target"]) == 1
                for row in pool
            ),
            "current_ready_action4_count": sum(
                bool(row["current_bundle_ready"])
                and int(row["executed_raw_policy_action"]) == 4
                for row in pool
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json",
    )
    parser.add_argument(
        "--manifest",
        default="configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json",
    )
    parser.add_argument(
        "--baseline_root",
        default="artifacts/benchmarks/calibrated_continuous_workflow_interface_repair_20261006_v1",
    )
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    config_path = (ROOT_DIR / args.config).resolve()
    manifest_path = (ROOT_DIR / args.manifest).resolve()
    baseline_root = (ROOT_DIR / args.baseline_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"create-only output already exists: {output_root}")
    output_root.mkdir(parents=True)

    config, _ = _load_experiment_config(config_path)
    manifest = _load_json(manifest_path)
    if manifest["config_sha256"] != _sha256(config_path):
        raise RuntimeError("frozen manifest config hash mismatch")
    if manifest["resolved_config_sha256"] != _canonical_sha256(config):
        raise RuntimeError("frozen manifest resolved-config hash mismatch")
    if config["interface_profile"] != "calibrated_workflow_interface_v2":
        raise RuntimeError("unexpected interface profile")
    if config["learning_interface"]["hierarchical_action_contract"] != "independent_heads_executed_env_v2":
        raise RuntimeError("unexpected action contract")

    instances = [row for row in manifest["instances"] if row["split"] == "dev"]
    if len(instances) != 4:
        raise RuntimeError("expected four frozen development instances")
    checkpoint_root = baseline_root / "checkpoints"
    checkpoint_hashes = {
        str(seed): _sha256(checkpoint_root / f"sa_ghmappo_seed{seed}_selected.pt")
        for seed in SEEDS
    }
    pool = _collect_pool(config, instances, checkpoint_root)
    selected = _select_states(pool)
    if len(selected) != MAX_STATES:
        raise RuntimeError(f"expected {MAX_STATES} selected states, got {len(selected)}")
    for index, row in enumerate(selected):
        row["diagnostic_state_id"] = f"dev_state_{index:02d}"
        row.pop("sort_key", None)

    summary = _summarize(pool, selected)
    _write_json(output_root / "diagnostic_states.json", selected)
    _write_json(output_root / "diagnostic_summary.json", summary)
    _write_json(
        output_root / "run_manifest.json",
        {
            "schema_version": "calibrated_workflow_prepare_balance_diagnosis_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "config": {"path": args.config, "sha256": _sha256(config_path)},
            "manifest": {"path": args.manifest, "sha256": _sha256(manifest_path)},
            "baseline_root": args.baseline_root,
            "checkpoint_hashes": checkpoint_hashes,
            "seeds": list(SEEDS),
            "source_split": "dev",
            "performance_evaluation": False,
            "training": False,
            "real_model_generate_calls": 0,
            "downloads": 0,
            "old_holdout_calls": 0,
        },
    )
    _write_json(
        output_root / "completion_receipt.json",
        {
            "status": "complete",
            "selected_state_count": len(selected),
            "diagnostic_only": True,
        },
    )
    print(json.dumps({"output_root": str(output_root), "status": "complete"}))


if __name__ == "__main__":
    main()
