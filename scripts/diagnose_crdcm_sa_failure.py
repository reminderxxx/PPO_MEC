#!/usr/bin/env python3
"""Read-only first-order diagnosis of the CRDCM SA performance failure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any
from zoneinfo import ZoneInfo

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.crdcm_agent import (
    CRDCMMAPPOAgent,
    CRDCMPPOAgent,
    CRDCMSAGHMAPPOAgent,
)
from src.encoders.crdcm_observation import finalize_crdcm_observation


DIAGNOSIS_VERSION = "crdcm_sa_first_order_diagnosis_v1"
POLICY_VERSION = "tmc_review_policy_v3_20260621"
ACTION_NAMES = {
    0: "current_rsu_cache_fill",
    1: "predictive_next_rsu_prefetch",
    2: "vehicle_fallback",
    3: "current_rsu_steady_offload",
    4: "handoff_migration_prepare",
}
CHECKPOINT_UPDATES = (4, 8, 12, 16)
PROTECTED_PATHS = (
    "scripts/train_sa_ghmappo_real_sample.py",
    "src/agents/sa_ghmappo_agent.py",
    "src/agents/sa_ghmappo_core.py",
    "src/encoders/fusion_encoder.py",
    "src/evaluators/real_eval_support.py",
    "tests/test_algo_pool_contract.py",
    "tests/test_checkpoint_compat.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def condition_from_run_name(name: str) -> str:
    for condition in (
        "crdcm_full_sa",
        "crdcm_signal_off_sa",
        "crdcm_full_mappo",
        "crdcm_full_ppo",
    ):
        if f"_{condition}_seed" in name:
            return condition
    raise ValueError(f"unknown run name: {name}")


def seed_from_run_name(name: str) -> int:
    match = re.search(r"seed(\d+)$", name)
    if match is None:
        raise ValueError(f"missing seed in run name: {name}")
    return int(match.group(1))


def action_weights(condition: str, action: int) -> dict[str, float]:
    if condition in {"crdcm_full_sa", "crdcm_signal_off_sa"}:
        return {"slow": 1.0, "fast": 1.0, "event": 1.0}
    if condition == "crdcm_full_mappo":
        if action == 4:
            raw = {"slow": 0.3, "fast": 0.1, "event": 1.0}
        elif action in {0, 1}:
            raw = {"slow": 1.0, "fast": 0.2, "event": 0.15}
        else:
            raw = {"slow": 0.3, "fast": 1.0, "event": 0.15}
        floors = {"slow": 0.25, "fast": 0.10, "event": 0.12}
        return {key: max(value, floors[key]) for key, value in raw.items()}
    return {"flat": 1.0}


def canonical_targets(action: int) -> dict[str, int]:
    if action == 0:
        return {"slow": 1, "fast": 0, "event": 0}
    if action == 1:
        return {"slow": 2, "fast": 0, "event": 0}
    if action == 2:
        return {"slow": 0, "fast": 1, "event": 0}
    if action == 4:
        return {"slow": 0, "fast": 0, "event": 1}
    return {"slow": 0, "fast": 0, "event": 0}


def aggregated_probs(head_probs: dict[str, list[float]]) -> list[float]:
    event = head_probs["event"]
    slow = head_probs["slow"]
    fast = head_probs["fast"]
    values = [
        event[0] * slow[1],
        event[0] * slow[2],
        event[0] * slow[0] * fast[1],
        event[0] * slow[0] * fast[0],
        event[1],
    ]
    total = sum(values)
    return [value / total for value in values]


def masked_probs(values: list[float], mask: list[bool] | None) -> list[float]:
    if not mask or len(mask) != len(values):
        return values
    selected = [value if bool(valid) else 0.0 for value, valid in zip(values, mask)]
    total = sum(selected)
    if total <= 0.0:
        return values
    return [value / total for value in selected]


def behavior_log_prob_gradient_norms(
    head_logits: dict[str, list[float]], action: int, mask: list[bool] | None
) -> dict[str, float]:
    tensors = {
        head: torch.tensor(values, dtype=torch.float64, requires_grad=True)
        for head, values in head_logits.items()
    }
    event_log_probs = torch.log_softmax(tensors["event"], dim=-1)
    slow_log_probs = torch.log_softmax(tensors["slow"], dim=-1)
    fast_log_probs = torch.log_softmax(tensors["fast"], dim=-1)
    scores = torch.stack(
        [
            event_log_probs[0] + slow_log_probs[1],
            event_log_probs[0] + slow_log_probs[2],
            event_log_probs[0] + slow_log_probs[0] + fast_log_probs[1],
            event_log_probs[0] + slow_log_probs[0] + fast_log_probs[0],
            event_log_probs[1],
        ]
    )
    if mask and len(mask) == 5:
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        scores = scores.masked_fill(~mask_tensor, -1.0e9)
    log_prob = torch.log_softmax(scores, dim=-1)[action]
    gradients = torch.autograd.grad(
        log_prob, tuple(tensors[head] for head in ("slow", "fast", "event"))
    )
    return {
        head: float(torch.linalg.vector_norm(gradient).item())
        for head, gradient in zip(("slow", "fast", "event"), gradients)
    }


def trace_credit_row(
    *, condition: str, seed: int, scenario: str, unit: str, trace: dict[str, Any]
) -> dict[str, Any]:
    action = int(trace["aggregated_policy_action"])
    mask = [bool(item) for item in trace.get("action_mask", [])]
    head_probs = dict(trace.get("post_crdcm_head_probabilities", {}))
    if condition == "crdcm_full_ppo":
        flat = list(head_probs["flat"])
        env_probs = masked_probs(flat, mask)
        reconstructed_log_prob = math.log(max(env_probs[action], 1e-30))
        irrelevant = []
        weighted_canonical_log_score = reconstructed_log_prob
    else:
        env_probs = masked_probs(aggregated_probs(head_probs), mask)
        reconstructed_log_prob = math.log(max(env_probs[action], 1e-30))
        weights = action_weights(condition, action)
        targets = canonical_targets(action)
        gradient_norms = behavior_log_prob_gradient_norms(
            dict(trace.get("post_crdcm_head_logits", {})), action, mask
        )
        irrelevant = sorted(
            head
            for head, weight in weights.items()
            if weight > 0.0 and gradient_norms[head] <= 1e-10
        )
        weighted_canonical_log_score = sum(
            weights[head] * math.log(max(head_probs[head][targets[head]], 1e-30))
            for head in ("slow", "fast", "event")
        )
    recorded = float(trace.get("raw_policy_log_prob") or 0.0)
    mechanism = dict(trace.get("mechanism_event", {}) or {})
    return {
        "condition_id": condition,
        "seed": seed,
        "scenario_id": scenario,
        "unit_id": unit,
        "step_index": int(trace.get("step_index", 0)),
        "action": action,
        "action_name": ACTION_NAMES[action],
        "executed_action": int(trace["actual_executed_action"]),
        "override_applied": bool(trace.get("override_applied", False)),
        "actor_credit_weight": trace.get("credit_provenance", {}).get(
            "actor_credit_weight"
        ),
        "recorded_log_prob": recorded,
        "reconstructed_behavior_log_prob": reconstructed_log_prob,
        "behavior_log_prob_abs_error": abs(recorded - reconstructed_log_prob),
        "weighted_canonical_head_log_score": weighted_canonical_log_score,
        "irrelevant_credited_heads": "|".join(irrelevant),
        "irrelevant_head_credit": bool(irrelevant),
        "behavior_gradient_norm_slow": (
            None if condition == "crdcm_full_ppo" else gradient_norms["slow"]
        ),
        "behavior_gradient_norm_fast": (
            None if condition == "crdcm_full_ppo" else gradient_norms["fast"]
        ),
        "behavior_gradient_norm_event": (
            None if condition == "crdcm_full_ppo" else gradient_norms["event"]
        ),
        "service_success": bool(mechanism.get("full_service_ready", False)),
        "request_failure": not bool(mechanism.get("full_service_ready", False)),
        "reward": float(trace.get("reward", 0.0)),
        "backhaul_traffic_cost": float(
            mechanism.get("backhaul_traffic_cost", 0.0) or 0.0
        ),
        "migration_overhead": float(
            mechanism.get("adapter_state_migration_overhead", 0.0) or 0.0
        ),
    }


def fixed_base_state(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "time_index": 10,
        "primary_vehicle_id": "veh_1",
        "current_workflow_node": {
            "node_id": "n2",
            "required_adapter": "adapter_perception",
            "input_size": 10.0,
            "output_size": 5.0,
            "predecessors": ["n1"],
            "successors": ["n3"],
        },
        "workflow": {
            "nodes": [
                {"node_id": "n1", "predecessors": [], "successors": ["n2"]},
                {
                    "node_id": "n2",
                    "predecessors": ["n1"],
                    "successors": ["n3"],
                },
                {"node_id": "n3", "predecessors": ["n2"], "successors": []},
            ],
            "completed_node_ids": ["n1"],
            "execution_order": ["n1", "n2", "n3"],
            "current_node_id": "n2",
        },
        "vehicles": [
            {"vehicle_id": "veh_1", "associated_rsu_id": "rsu_a", "speed": 10.0}
        ],
        "rsus": [
            {
                "rsu_id": "rsu_a",
                "cached_adapter_ids": ["adapter_tracking"],
                "cache_capacity": 360.0,
            },
            {"rsu_id": "rsu_b", "cached_adapter_ids": [], "cache_capacity": 360.0},
        ],
        "predictions": {
            "future_load": {"rsu_a": 1.0, "rsu_b": 2.0},
            "predicted_handoff_vehicle_ids": ["veh_1"],
            "predicted_next_rsu_by_vehicle": {"veh_1": "rsu_b"},
            "predicted_first_handoff_rsu_by_vehicle": {"veh_1": "rsu_b"},
            "prediction_confidence_by_vehicle": {"veh_1": 0.8},
            "prediction_uncertainty_by_vehicle": {"veh_1": 0.2},
            "dwell_time": {"veh_1": 2.0},
            "next_rsu_sequence": {"veh_1": ["rsu_b"]},
        },
        "crdcm_observation": finalize_crdcm_observation(payload),
    }


def fixed_payloads() -> dict[str, dict[str, Any]]:
    base = {
        "observed_at_time_index": 10,
        "update_phase": "pre_action_after_causal_prediction",
        "units": {"cache_capacity": "mb", "time": "mobility_step"},
        "sources": {"remaining_dag": "known_submitted_workflow_dag"},
        "required_request": {
            "node_id": "n2",
            "required_base_model_id": "veh_base_v1",
            "required_adapter_id": "adapter_perception",
        },
        "current_rsu": {
            "rsu_id": "rsu_a",
            "capacity_remaining": 140.0,
            "occupancy_rate": 0.6,
            "required_bundle": {
                "base_ready": True,
                "adapter_ready": True,
                "missing_resident_mb": 0.0,
            },
        },
        "predicted_target_rsu": {
            "rsu_id": "rsu_b",
            "capacity_remaining": 100.0,
            "occupancy_rate": 0.7,
            "required_bundle": {
                "base_ready": False,
                "adapter_ready": False,
                "missing_resident_mb": 96.0,
                "eviction_shortfall_mb": 0.0,
            },
        },
        "remaining_dag": {
            "remaining_nodes": 8,
            "remaining_ratio": 0.8,
            "frontier_size": 2,
            "critical_path_length": 5,
            "critical_path_pressure": 0.625,
            "current_adapter_remaining_reuse_count": 3,
            "current_adapter_remaining_reuse_ratio": 0.375,
            "completed_ratio": 0.2,
        },
        "handoff_prediction": {
            "target_rsu_id": "rsu_b",
            "target_available": True,
            "target_differs_from_current": True,
            "confidence": 0.8,
            "uncertainty": 0.2,
            "eta_steps": 2,
        },
        "migration": {
            "enabled": True,
            "state_required": True,
            "state_ready": False,
            "capacity_conflict": False,
            "risk": 0.8,
        },
    }
    warm = deepcopy(base)
    warm["predicted_target_rsu"]["required_bundle"].update(
        {"base_ready": True, "adapter_ready": True, "missing_resident_mb": 0.0}
    )
    warm["migration"].update({"state_ready": True, "risk": 0.1})
    cold = deepcopy(base)
    cold["current_rsu"]["required_bundle"].update(
        {"adapter_ready": False, "missing_resident_mb": 40.0}
    )
    conflict = deepcopy(cold)
    conflict["predicted_target_rsu"]["required_bundle"]["eviction_shortfall_mb"] = 80.0
    conflict["migration"]["capacity_conflict"] = True
    return {
        "target_warm_state_ready": warm,
        "current_adapter_miss_target_cold": cold,
        "capacity_conflict_state_missing": conflict,
    }


def make_agent(condition: str, seed: int):
    common = {
        "random_seed": seed,
        "learning_rate": 3e-4,
        "clip_ratio": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "batch_size": 32,
        "deterministic_action": False,
        "crdcm_feature_mode": (
            "signal_off" if condition == "crdcm_signal_off_sa" else "full"
        ),
    }
    if condition in {"crdcm_full_sa", "crdcm_signal_off_sa"}:
        return CRDCMSAGHMAPPOAgent(**common)
    if condition == "crdcm_full_mappo":
        return CRDCMMAPPOAgent(**common)
    if condition == "crdcm_full_ppo":
        return CRDCMPPOAgent(**common)
    raise ValueError(condition)


def replay_checkpoints(run_dirs: list[Path]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    payloads = fixed_payloads()
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for run_dir in run_dirs:
        condition = condition_from_run_name(run_dir.name)
        seed = seed_from_run_name(run_dir.name)
        for update in CHECKPOINT_UPDATES:
            checkpoint = run_dir / "checkpoints" / f"update_{update:04d}.pt"
            hashes[str(checkpoint.resolve())] = sha256(checkpoint)
            agent = make_agent(condition, seed)
            agent.load(str(checkpoint))
            if int(agent._update_count) != update:
                raise ValueError(f"checkpoint update mismatch: {checkpoint}")
            for probe_name, payload in payloads.items():
                state = fixed_base_state(deepcopy(payload))
                with torch.no_grad():
                    output = agent._forward_policy(state, run_metadata={})
                    if agent._use_hierarchy:
                        env_logits = agent._hierarchical_env_action_scores(output)
                        head_probs = {
                            head: torch.softmax(output[f"{head}_logits"], dim=-1).tolist()
                            for head in ("slow", "fast", "event")
                        }
                    else:
                        env_logits = output["flat_logits"]
                        head_probs = {
                            "flat": torch.softmax(output["flat_logits"], dim=-1).tolist()
                        }
                    env_probs = torch.softmax(env_logits, dim=-1)
                rows.append(
                    {
                        "condition_id": condition,
                        "seed": seed,
                        "update": update,
                        "probe_name": probe_name,
                        "checkpoint_sha256": hashes[str(checkpoint.resolve())],
                        "feature_vector_sha256": canonical_sha256(
                            state["crdcm_observation"]["feature_vector"]
                        ),
                        "env_action_0_probability": float(env_probs[0]),
                        "env_action_1_probability": float(env_probs[1]),
                        "env_action_2_probability": float(env_probs[2]),
                        "env_action_3_probability": float(env_probs[3]),
                        "env_action_4_probability": float(env_probs[4]),
                        "deterministic_argmax_action": int(torch.argmax(env_probs).item()),
                        "deterministic_argmax_action_name": ACTION_NAMES[
                            int(torch.argmax(env_probs).item())
                        ],
                        "env_action_entropy": float(
                            -(env_probs * torch.log(env_probs.clamp_min(1e-30))).sum()
                        ),
                        "value": float(output["value"]),
                        "residual_l2": float(
                            torch.linalg.vector_norm(output["crdcm_residual"])
                        ),
                        "head_probabilities_json": json.dumps(
                            head_probs, ensure_ascii=False, sort_keys=True
                        ),
                    }
                )
    return rows, hashes


def episode_cost_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    info = dict(payload["run_info"])
    endpoint = dict(payload["formal_request_execution_audit"])
    system = dict(payload["system_metrics"])
    traces = list(payload.get("policy_decision_trace_v2", []))
    success_requests = sum(
        bool(item.get("mechanism_event", {}).get("full_service_ready", False))
        for item in traces
    )
    requests = len(traces)
    return {
        "condition_id": str(info["condition_id"]),
        "seed": int(info["seed"]),
        "scenario_id": str(info["scenario_id"]),
        "unit_id": str(info["unit_id"]),
        "request_exposure_fingerprint": str(info["request_exposure_fingerprint"]),
        "workflow_completed": bool(
            endpoint["workflow_completed_under_exogenous_execution"]
        ),
        "requests": requests,
        "successful_requests": success_requests,
        "failed_requests": requests - success_requests,
        "transfer_mb_total": float(endpoint["transfer_mb_per_request"]) * requests,
        "backhaul_total": float(system["backhaul_traffic_cost"]),
        "migration_overhead_total": float(
            system["adapter_state_migration_overhead"]
        ),
        "raw_path": str(path.resolve()),
    }


def aggregate_efficiency(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["condition_id"]].append(row)
    result = []
    for condition, rows in sorted(grouped.items()):
        successful = sum(row["successful_requests"] for row in rows)
        completed = sum(row["workflow_completed"] for row in rows)
        transfer = sum(row["transfer_mb_total"] for row in rows)
        backhaul = sum(row["backhaul_total"] for row in rows)
        migration = sum(row["migration_overhead_total"] for row in rows)
        result.append(
            {
                "condition_id": condition,
                "episodes": len(rows),
                "completed_workflows": completed,
                "completion_rate": completed / len(rows),
                "requests": sum(row["requests"] for row in rows),
                "successful_requests": successful,
                "failed_requests": sum(row["failed_requests"] for row in rows),
                "successful_request_rate": successful
                / max(sum(row["requests"] for row in rows), 1),
                "total_transfer_mb": transfer,
                "total_backhaul_cost": backhaul,
                "total_migration_overhead": migration,
                "transfer_mb_per_successful_request_including_failure_cost": (
                    transfer / successful if successful else None
                ),
                "backhaul_per_successful_request_including_failure_cost": (
                    backhaul / successful if successful else None
                ),
                "transfer_mb_per_completed_workflow_including_failure_cost": (
                    transfer / completed if completed else None
                ),
                "backhaul_per_completed_workflow_including_failure_cost": (
                    backhaul / completed if completed else None
                ),
            }
        )
    return result


def common_success(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["condition_id"], row["seed"], row["scenario_id"], row["unit_id"]): row
        for row in records
    }
    rows = []
    for seed in (1401, 1402, 1403):
        keys = sorted(
            {
                (row["scenario_id"], row["unit_id"])
                for row in records
                if row["seed"] == seed
            }
        )
        for left, right in (
            ("crdcm_full_sa", "crdcm_full_ppo"),
            ("crdcm_full_mappo", "crdcm_full_ppo"),
        ):
            selected = []
            for scenario, unit in keys:
                a = lookup[(left, seed, scenario, unit)]
                b = lookup[(right, seed, scenario, unit)]
                if a["workflow_completed"] and b["workflow_completed"]:
                    selected.append((a, b))
            rows.append(
                {
                    "left_condition": left,
                    "right_condition": right,
                    "seed": seed,
                    "common_success_units": len(selected),
                    "left_transfer_mb_total": sum(a["transfer_mb_total"] for a, _ in selected),
                    "right_transfer_mb_total": sum(b["transfer_mb_total"] for _, b in selected),
                    "left_backhaul_total": sum(a["backhaul_total"] for a, _ in selected),
                    "right_backhaul_total": sum(b["backhaul_total"] for _, b in selected),
                    "selection_warning": "post_outcome_selected_survivor_conditioned_not_causal",
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    source = args.source_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(path.parent for path in source.glob("runs/*/*/summary.json"))
    if len(run_dirs) != 12:
        raise ValueError(f"expected 12 learned run directories, found {len(run_dirs)}")

    protected_before = {path: sha256(ROOT / path) for path in PROTECTED_PATHS}
    replay_rows, checkpoint_hashes_before = replay_checkpoints(run_dirs)

    step_rows: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    episode_payloads: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for path in sorted((source / "evaluation" / "episodes").glob("*/*/*.json")):
        payload = load_json(path)
        info = dict(payload["run_info"])
        condition = str(info["condition_id"])
        seed = int(info["seed"])
        scenario = str(info["scenario_id"])
        unit = str(info["unit_id"])
        episode_payloads[(condition, seed, scenario, unit)] = payload
        episode_records.append(episode_cost_record(path, payload))
        for trace in payload.get("policy_decision_trace_v2", []):
            step_rows.append(
                trace_credit_row(
                    condition=condition,
                    seed=seed,
                    scenario=scenario,
                    unit=unit,
                    trace=trace,
                )
            )
    if len(episode_records) != 144:
        raise ValueError(f"expected 144 learned evaluation episodes, found {len(episode_records)}")

    credit_summary: list[dict[str, Any]] = []
    for condition in sorted({row["condition_id"] for row in step_rows}):
        rows = [row for row in step_rows if row["condition_id"] == condition]
        for seed in (1401, 1402, 1403):
            selected = [row for row in rows if row["seed"] == seed]
            counts = Counter(row["action"] for row in selected)
            credit_summary.append(
                {
                    "condition_id": condition,
                    "seed": seed,
                    "steps": len(selected),
                    "action_0_count": counts[0],
                    "action_1_count": counts[1],
                    "action_2_count": counts[2],
                    "action_3_count": counts[3],
                    "action_4_count": counts[4],
                    "irrelevant_head_credit_steps": sum(
                        row["irrelevant_head_credit"] for row in selected
                    ),
                    "irrelevant_head_credit_rate": sum(
                        row["irrelevant_head_credit"] for row in selected
                    )
                    / max(len(selected), 1),
                    "override_steps": sum(row["override_applied"] for row in selected),
                    "request_failures": sum(row["request_failure"] for row in selected),
                    "max_behavior_log_prob_reconstruction_error": max(
                        row["behavior_log_prob_abs_error"] for row in selected
                    ),
                }
            )

    paired_rows = []
    for seed in (1401, 1402, 1403):
        units = sorted(
            {
                (key[2], key[3])
                for key in episode_payloads
                if key[0] == "crdcm_full_sa" and key[1] == seed
            }
        )
        for scenario, unit in units:
            sa = episode_payloads[("crdcm_full_sa", seed, scenario, unit)]
            ppo = episode_payloads[("crdcm_full_ppo", seed, scenario, unit)]
            if (
                sa["run_info"]["request_exposure_fingerprint"]
                != ppo["run_info"]["request_exposure_fingerprint"]
            ):
                raise ValueError("paired request exposure fingerprint mismatch")
            sa_trace = list(sa["policy_decision_trace_v2"])
            ppo_trace = list(ppo["policy_decision_trace_v2"])
            if len(sa_trace) != len(ppo_trace):
                raise ValueError("paired request trace length mismatch")
            for left, right in zip(sa_trace, ppo_trace):
                sa_success = bool(
                    left.get("mechanism_event", {}).get("full_service_ready", False)
                )
                ppo_success = bool(
                    right.get("mechanism_event", {}).get("full_service_ready", False)
                )
                paired_rows.append(
                    {
                        "seed": seed,
                        "scenario_id": scenario,
                        "unit_id": unit,
                        "step_index": int(left["step_index"]),
                        "node_id": left.get("request", {}).get("node_id"),
                        "sa_action": int(left["actual_executed_action"]),
                        "sa_action_name": ACTION_NAMES[int(left["actual_executed_action"])],
                        "ppo_action": int(right["actual_executed_action"]),
                        "ppo_action_name": ACTION_NAMES[int(right["actual_executed_action"])],
                        "same_action": int(left["actual_executed_action"])
                        == int(right["actual_executed_action"]),
                        "sa_success": sa_success,
                        "ppo_success": ppo_success,
                        "outcome_transition": (
                            "both_success"
                            if sa_success and ppo_success
                            else "sa_only_success"
                            if sa_success
                            else "ppo_only_success"
                            if ppo_success
                            else "both_fail"
                        ),
                        "sa_reward": float(left["reward"]),
                        "ppo_reward": float(right["reward"]),
                    }
                )

    pair_summary = []
    for seed in (1401, 1402, 1403):
        selected = [row for row in paired_rows if row["seed"] == seed]
        transitions = Counter(row["outcome_transition"] for row in selected)
        pair_summary.append(
            {
                "seed": seed,
                "paired_requests": len(selected),
                "same_action_count": sum(row["same_action"] for row in selected),
                "different_action_count": sum(not row["same_action"] for row in selected),
                "both_success": transitions["both_success"],
                "sa_only_success": transitions["sa_only_success"],
                "ppo_only_success": transitions["ppo_only_success"],
                "both_fail": transitions["both_fail"],
                "mean_reward_delta_sa_minus_ppo": fmean(
                    row["sa_reward"] - row["ppo_reward"] for row in selected
                ),
            }
        )

    training_audit = []
    training_trace_summary = []
    for run_dir in run_dirs:
        summary = load_json(run_dir / "summary.json")
        logs = list(summary["update_logs"])
        condition = condition_from_run_name(run_dir.name)
        seed = seed_from_run_name(run_dir.name)
        config = torch.load(
            run_dir / "checkpoints" / "update_0004.pt", map_location="cpu"
        )["config"]
        training_audit.append(
            {
                "condition_id": condition,
                "seed": seed,
                "episodes": int(summary["episodes"]),
                "updates": int(summary["update_count"]),
                "collected_steps": sum(int(row["collected_steps"]) for row in logs),
                "external_override_masked_count": sum(
                    int(row.get("external_override_masked_count", 0) or 0)
                    for row in logs
                ),
                "actor_credit_eligible_count": sum(
                    int(row.get("actor_credit_eligible_count", 0) or 0) for row in logs
                ),
                "critic_only_sample_count": sum(
                    int(row.get("critic_only_sample_count", 0) or 0) for row in logs
                ),
                "mean_actor_loss": fmean(float(row["actor_loss"]) for row in logs),
                "mean_value_loss": fmean(float(row["value_loss"]) for row in logs),
                "mean_policy_entropy": fmean(
                    float(row["policy_entropy"]) for row in logs
                ),
                "mean_approx_kl": fmean(float(row["approx_kl"]) for row in logs),
                "mean_clip_fraction": fmean(
                    float(row["clip_fraction"]) for row in logs
                ),
                "env_action_log_prob_missing_count": sum(
                    int(row.get("env_action_log_prob_missing_count", 0) or 0)
                    for row in logs
                ),
                "encoder_kind": config["encoder_kind"],
                "centralized_critic": bool(config["centralized_critic"]),
                "use_hierarchy": bool(config["use_hierarchy"]),
                "hierarchical_conditioning": bool(config["hierarchical_conditioning"]),
                "head_credit_enabled": bool(config["head_credit_enabled"]),
                "head_credit_protocol": config["head_credit_protocol"],
                "learning_rate": float(config["learning_rate"]),
                "clip_ratio": float(config["clip_ratio"]),
                "entropy_coef": float(config["entropy_coef"]),
                "value_coef": float(config["value_coef"]),
                "train_epochs": int(config["train_epochs"]),
                "event_logit_temperature": float(config["event_logit_temperature"]),
                "event_logit_temperature_final": float(
                    config["event_logit_temperature_final"]
                ),
                "event_temperature_decay_updates": int(
                    config["event_temperature_decay_updates"]
                ),
                "event_logit_sharpening_final_scale": float(
                    config["event_logit_sharpening_final_scale"]
                ),
                "temporal_consistency_coef": float(config["temporal_consistency_coef"]),
                "crdcm_feature_mode": config["crdcm_feature_mode"],
            }
        )
        block_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        completion_by_block: Counter[int] = Counter()
        right_censored_by_block: Counter[int] = Counter()
        episode_count_by_block: Counter[int] = Counter()
        for episode_path in sorted((run_dir / "episodes").glob("episode_*.summary.json")):
            match = re.search(r"episode_(\d+)\.summary\.json$", episode_path.name)
            if match is None:
                raise ValueError(f"unexpected episode path: {episode_path}")
            episode_index = int(match.group(1))
            update_block = int(math.ceil(episode_index / 16.0) * 4)
            payload = load_json(episode_path)
            endpoint = dict(payload["formal_request_execution_audit"])
            episode_count_by_block[update_block] += 1
            completion_by_block[update_block] += bool(
                endpoint["workflow_completed_under_exogenous_execution"]
            )
            right_censored_by_block[update_block] += bool(endpoint["right_censored"])
            run_info = dict(payload["run_info"])
            for trace in payload.get("policy_decision_trace_v2", []):
                block_rows[update_block].append(
                    trace_credit_row(
                        condition=condition,
                        seed=seed,
                        scenario=str(run_info.get("window_id", "training")),
                        unit=str(run_info.get("evaluation_unit_id", episode_path.name)),
                        trace=trace,
                    )
                )
        for update_block in CHECKPOINT_UPDATES:
            rows = block_rows[update_block]
            counts = Counter(row["action"] for row in rows)
            training_trace_summary.append(
                {
                    "condition_id": condition,
                    "seed": seed,
                    "collection_block_ending_update": update_block,
                    "episodes": episode_count_by_block[update_block],
                    "steps": len(rows),
                    "completed_workflows": completion_by_block[update_block],
                    "right_censored_workflows": right_censored_by_block[update_block],
                    "action_0_count": counts[0],
                    "action_1_count": counts[1],
                    "action_2_count": counts[2],
                    "action_3_count": counts[3],
                    "action_4_count": counts[4],
                    "irrelevant_head_credit_steps": sum(
                        row["irrelevant_head_credit"] for row in rows
                    ),
                    "irrelevant_head_credit_rate": sum(
                        row["irrelevant_head_credit"] for row in rows
                    )
                    / max(len(rows), 1),
                    "override_steps": sum(row["override_applied"] for row in rows),
                    "reward_mean": fmean(row["reward"] for row in rows),
                    "reward_min": min(row["reward"] for row in rows),
                    "reward_max": max(row["reward"] for row in rows),
                    "max_behavior_log_prob_reconstruction_error": max(
                        row["behavior_log_prob_abs_error"] for row in rows
                    ),
                }
            )

    checkpoint_hashes_after = {
        path: sha256(Path(path)) for path in checkpoint_hashes_before
    }
    protected_after = {path: sha256(ROOT / path) for path in PROTECTED_PATHS}
    if checkpoint_hashes_before != checkpoint_hashes_after:
        raise RuntimeError("checkpoint bytes changed during read-only replay")
    if protected_before != protected_after:
        raise RuntimeError("protected source changed during diagnosis")

    efficiency_rows = aggregate_efficiency(episode_records)
    common_success_rows = common_success(episode_records)
    write_csv(output / "checkpoint_fixed_observation_replay.csv", replay_rows)
    write_csv(output / "evaluation_step_credit_audit.csv", step_rows)
    write_csv(output / "evaluation_credit_summary.csv", credit_summary)
    write_csv(output / "sa_vs_ppo_request_pairs.csv", paired_rows)
    write_csv(output / "sa_vs_ppo_request_pair_summary.csv", pair_summary)
    write_csv(output / "efficiency_all_episode_summary.csv", efficiency_rows)
    write_csv(output / "common_success_selected_summary.csv", common_success_rows)
    write_csv(output / "training_config_and_credit_audit.csv", training_audit)
    write_csv(output / "training_trace_credit_summary.csv", training_trace_summary)

    reviewed_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    diagnosis = {
        "diagnosis_version": DIAGNOSIS_VERSION,
        "reviewed_at": reviewed_at,
        "literature_cutoff": "2026-09-28",
        "target_venue": "IEEE Transactions on Mobile Computing (TMC)",
        "artifact_run_id": source.name,
        "policy_version": POLICY_VERSION,
        "git_commit": git_value("rev-parse", "HEAD"),
        "evidence_level": "E3_REPRODUCED_OBSERVED_DATA_DEVELOPMENT_PILOT_NOT_HOLDOUT",
        "environment_steps_executed_by_diagnosis": 0,
        "training_updates_executed_by_diagnosis": 0,
        "source_artifact_mutated": False,
        "checkpoint_files_verified_unchanged": len(checkpoint_hashes_before),
        "protected_files_verified_unchanged": len(protected_before),
        "primary_finding": {
            "classification": "implementation_contract_defect_in_hierarchical_actor_credit",
            "statement": (
                "The hierarchical behavior policy samples the exact five-action aggregated "
                "distribution, but the default SA PPO loss trains canonical per-head labels, "
                "including heads whose exact masked behavior-log-probability gradient is zero. "
                "Mask renormalization can cancel event/fast dependencies even for actions 2/3; "
                "flat PPO does not have this behavior/update-distribution mismatch."
            ),
            "performance_causality_status": "UNVERIFIED_without_new_matched_training",
        },
        "inactive_candidate_causes": {
            "external_override_credit_path": {
                "status": "not_active_in_source_run",
                "masked_samples": sum(
                    row["external_override_masked_count"] for row in training_audit
                ),
                "note": (
                    "Critic-only updates can change shared representations and therefore actor "
                    "outputs indirectly, but no source-run sample entered this path."
                ),
            }
        },
        "credit_mismatch_summary": credit_summary,
        "training_trace_credit_summary": training_trace_summary,
        "sa_vs_ppo_request_pair_summary": pair_summary,
        "efficiency_all_episode_summary": efficiency_rows,
        "common_success_warning": (
            "Common-success cost comparisons are post-outcome selected and survivor-conditioned; "
            "they are descriptive only. All-episode totals retain failure costs."
        ),
        "fixed_observation_replay": {
            "rows": len(replay_rows),
            "conditions": 4,
            "seeds": [1401, 1402, 1403],
            "updates": list(CHECKPOINT_UPDATES),
            "synthetic_probe_count": len(fixed_payloads()),
            "historical_state_exact_replay": False,
            "boundary": (
                "Training/evaluation summaries do not persist complete semantic_state. The "
                "checkpoint probe is a fixed causal synthetic-state sensitivity test, not an "
                "exact replay of historical requests."
            ),
        },
        "prioritized_recommendation": {
            "priority": 1,
            "action": (
                "Freeze performance expansion and first add a factorization-consistency validation: "
                "for each legal five-action sample, compare the gradient/ratio of the actual "
                "aggregated behavior log-probability with the current canonical-head PPO loss. "
                "Only in a separate implementation task, replace or mask latent-head actor credit "
                "and then run one pre-registered matched retraining; do not add seeds or tune."
            ),
            "stop_rule": (
                "If the corrected contract does not remove the fixed-observation gradient mismatch "
                "or does not improve pre-registered completion without worsening total failure cost, "
                "stop the complex SA route and retain flat PPO as the supported candidate."
            ),
        },
        "claim_boundary": {
            "safe": [
                "PPO is descriptively ahead in this development pilot (27/36 vs SA 18/36 completions).",
                "The current hierarchical SA actor update credits heads with zero exact masked-behavior gradient.",
                "The defect is a credible first-order explanation requiring matched retraining to establish performance causality.",
            ],
            "prohibited": [
                "SA is more transfer-efficient than PPO.",
                "More training will fix SA.",
                "The credit defect alone caused the entire completion gap.",
                "The development pilot is formal, holdout, paper-ready, or statistically conclusive.",
            ],
        },
        "checkpoint_hashes_before_after_equal": checkpoint_hashes_before
        == checkpoint_hashes_after,
        "protected_hashes_before_after_equal": protected_before == protected_after,
    }
    write_json(output / "diagnosis.json", diagnosis)

    manifest_files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "artifact_integrity_manifest.json"
    )
    write_json(
        output / "artifact_integrity_manifest.json",
        {
            "manifest_version": "artifact_integrity_manifest_v1",
            "generated_at": reviewed_at,
            "files": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in manifest_files
            ],
            "source_checkpoint_hashes_before": checkpoint_hashes_before,
            "source_checkpoint_hashes_after": checkpoint_hashes_after,
            "protected_hashes_before": protected_before,
            "protected_hashes_after": protected_after,
        },
    )
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
