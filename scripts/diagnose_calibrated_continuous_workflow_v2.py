"""Create-only non-degeneracy checks for the calibrated workflow pilot v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.envs.core.calibrated_continuous_workflow_env import (
    CalibratedContinuousWorkflowEnv,
    ImmediateCostRule,
    TwoStepCostRule,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_key(value: Any, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(_contains_key(item, forbidden) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def _future_signature(env: CalibratedContinuousWorkflowEnv) -> str:
    payload = {
        "node_index": env.node_index,
        "completed": env.completed,
        "residents": {key: value.residents for key, value in sorted(env.caches.items())},
        "prepared_state": env.prepared_state,
        "service_failures": env.metrics["service_failures"],
        "modeled_completion_seconds": round(float(env.metrics["modeled_completion_seconds"]), 9),
        "total_transfer_bytes": int(
            env.metrics["model_transfer_bytes"]
            + env.metrics["state_transfer_bytes"]
            + env.metrics["input_transfer_bytes"]
        ),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _expected_transfer_seconds(transition: dict[str, Any], mbps: float, fixed: float) -> float:
    result = 0.0
    for key in ("model_transfer_bytes", "state_transfer_bytes", "input_transfer_bytes"):
        byte_count = int(transition[key])
        if byte_count > 0:
            result += byte_count * 8.0 / (mbps * 1_000_000.0) + fixed
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/calibrated_continuous_workflow_pilot_v2.json")
    parser.add_argument("--manifest", default="configs/experiment/calibrated_continuous_workflow_pilot_v2_manifest.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = ROOT_DIR / args.config
    manifest_path = ROOT_DIR / args.manifest
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"create-only diagnostic already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = _load(config_path)
    manifest = _load(manifest_path)
    if manifest["config_sha256"] != _sha256(config_path):
        raise RuntimeError("frozen manifest config hash mismatch")

    immediate = ImmediateCostRule()
    two_step = TwoStepCostRule()
    action_counts: Counter[int] = Counter()
    two_step_counts: Counter[int] = Counter()
    disagreement_rows: list[dict[str, Any]] = []
    distinct_future_rows: list[dict[str, Any]] = []
    transfer_formula_max_error = 0.0
    semantic_actual_link_leaks = 0
    checked_state_actions = 0
    checked_reachable_states = 0
    search_depth = 4

    for instance in manifest["instances"]:
        frontier: list[tuple[CalibratedContinuousWorkflowEnv, list[int]]] = [
            (CalibratedContinuousWorkflowEnv(config, instance), [])
        ]
        visited: set[str] = set()
        for _depth in range(search_depth):
            next_frontier: list[tuple[CalibratedContinuousWorkflowEnv, list[int]]] = []
            for env, path in frontier:
                state_key = _future_signature(env)
                if state_key in visited or env.terminated:
                    continue
                visited.add(state_key)
                checked_reachable_states += 1
                semantic = env._info()["semantic_state"]
                semantic_actual_link_leaks += int(_contains_key(semantic, "actual_mbps"))
                immediate_action = immediate.select_action(env)
                two_step_action = two_step.select_action(env)
                action_counts[immediate_action] += 1
                two_step_counts[two_step_action] += 1
                if immediate_action != two_step_action:
                    disagreement_rows.append(
                        {
                            "design_id": instance["design_id"],
                            "split": instance["split"],
                            "path": path,
                            "immediate_action": immediate_action,
                            "two_step_action": two_step_action,
                        }
                    )
                signatures: dict[int, str] = {}
                for action in env.valid_actions():
                    trial = env.clone()
                    _, _, terminated, truncated, info = trial.step(action)
                    transition = info["transition"]
                    signatures[action] = _future_signature(trial)
                    actual_mbps = float(
                        instance.get("link_profile", {}).get(
                            "actual_mbps", config["link"]["mbps"]
                        )
                    )
                    expected = _expected_transfer_seconds(
                        transition,
                        actual_mbps,
                        float(config["link"]["fixed_seconds"]),
                    )
                    transfer_formula_max_error = max(
                        transfer_formula_max_error,
                        abs(expected - float(transition["transfer_seconds"])),
                    )
                    checked_state_actions += 1
                    if not terminated and not truncated:
                        next_frontier.append((trial, path + [action]))
                if len(set(signatures.values())) > 1:
                    distinct_future_rows.append(
                        {
                            "design_id": instance["design_id"],
                            "split": instance["split"],
                            "path": path,
                            "legal_action_count": len(signatures),
                            "distinct_future_count": len(set(signatures.values())),
                        }
                    )
            frontier = next_frontier

    checks = {
        "legal_actions_change_future_state_or_cost": len(distinct_future_rows) > 0,
        "symmetric_execution_transfer_formula": transfer_formula_max_error <= 1e-9,
        "decision_state_hides_actual_link_cost": semantic_actual_link_leaks == 0,
        "one_step_rule_is_not_constant": len(action_counts) > 1,
        "two_step_rule_is_not_constant": len(two_step_counts) > 1,
        "one_step_and_two_step_disagree_somewhere": len(disagreement_rows) > 0,
    }
    payload = {
        "schema_version": "calibrated_continuous_workflow_non_degeneracy_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "purpose": "problem non-degeneracy check only; not evidence of RL superiority",
        "config": {"path": args.config, "sha256": _sha256(config_path)},
        "manifest": {"path": args.manifest, "sha256": _sha256(manifest_path)},
        "checks": checks,
        "counts": {
            "instances": len(manifest["instances"]),
            "search_depth": search_depth,
            "checked_reachable_states": checked_reachable_states,
            "checked_reachable_state_actions": checked_state_actions,
            "reachable_states_with_distinct_action_futures": len(distinct_future_rows),
            "immediate_action_counts": dict(sorted(action_counts.items())),
            "two_step_action_counts": dict(sorted(two_step_counts.items())),
            "rule_disagreements": len(disagreement_rows),
            "semantic_actual_link_leaks": semantic_actual_link_leaks,
        },
        "transfer_formula_max_abs_error_seconds": transfer_formula_max_error,
        "distinct_future_examples": distinct_future_rows[:12],
        "rule_disagreement_examples": disagreement_rows[:12],
        "limitations": [
            "The checks inspect reachable states to depth four and their one-step consequences; they do not prove an RL advantage.",
            "A varying or disagreeing rule action does not establish optimal-policy complexity.",
            "Network rates and their estimation errors are artificial assumptions, not wireless measurements.",
        ],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output_path)}), flush=True)
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
