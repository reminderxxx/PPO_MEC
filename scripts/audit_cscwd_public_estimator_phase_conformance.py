"""Create a bounded synthetic conformance receipt for the public estimator.

This audit uses the existing calibrated-workflow fixture and never reads raw
mobility rows, checkpoints, realized future routes, or evaluation outcomes.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_calibrated_continuous_workflow_pilot import (
    _load_experiment_config,
)
from src.agents.causal_public_action_estimator import estimate_public_actions
from src.envs.core.calibrated_continuous_workflow_env import (
    CalibratedContinuousWorkflowEnv,
    PREPARED_STATE_PREFIX_PROFILE,
)
from src.envs.core.causal_rsu_predictor import fit_predictor

RUN_ID = "cscwd_public_estimator_phase_conformance_20261011_v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    config, _ = _load_experiment_config(
        ROOT
        / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json"
    )
    manifest_path = (
        ROOT
        / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    ordered = sorted(manifest["instances"], key=lambda row: str(row["design_id"]))
    model = fit_predictor(
        [row["rsu_sequence"] for row in ordered if row["split"] == "train"]
    )
    instance = deepcopy(
        next(row for row in ordered if row["design_id"] == "regression_08")
    )
    instance["causal_predictor_model"] = model
    instance["link_profile"]["actual_mbps"] = instance["link_profile"][
        "estimated_mbps"
    ]
    return config, instance


def _public_input(
    env: CalibratedContinuousWorkflowEnv,
    config: dict[str, Any],
    *,
    contact_seconds: float | None = None,
) -> tuple[dict[str, Any], list[bool]]:
    info = env._info()
    state = deepcopy(info["semantic_state"])
    state["time_profile"] = "raw_ngsim_event_time_v1"
    context = state["calibrated_context"]
    if contact_seconds is not None:
        context["contact_budget_seconds"] = float(contact_seconds)
    context["time_contract"] = {
        "schema_version": "public_estimator_phase_conformance_v1",
        "clock_seconds": float(env.clock_seconds),
        "deadline_seconds": 100.0,
        "remaining_deadline_seconds": 100.0 - float(env.clock_seconds),
        "trace_remaining_policy_visible": False,
        "contact_scope": "current_rsu",
    }
    context["vehicle_fallback_seconds"] = float(
        config["vehicle"]["fallback_seconds"]
    )
    context["failed_service_seconds"] = float(
        config["objective"]["failed_service_seconds"]
    )
    return state, list(info["action_mask"])


def _prepared_env(
    *, current_ready: bool
) -> tuple[CalibratedContinuousWorkflowEnv, dict[str, Any], str]:
    config, instance = _fixture()
    env = CalibratedContinuousWorkflowEnv(
        {**config, "interface_profile": PREPARED_STATE_PREFIX_PROFILE},
        instance,
    )
    current = env._current_rsu_id()
    target = env._predicted_handoff_target()
    if target is None:
        raise RuntimeError("synthetic fixture has no predicted handoff target")
    node = env._current_node()
    bundle = env._bundle_ids(str(node["required_adapter"]))
    env.caches[current].residents = list(bundle) if current_ready else []
    env.caches[current].last_used = {
        item: 0 for item in env.caches[current].residents
    }
    env.caches[target].residents = []
    env.caches[target].last_used = {}
    return env, config, str(target)


def _transition_view(transition: dict[str, Any]) -> dict[str, Any]:
    return {
        "service_completed": bool(transition["service_completed"]),
        "migration_success": bool(transition["migration_success"]),
        "step_cost_seconds": float(transition["step_cost_seconds"]),
        "model_transfer_bytes": int(transition["model_transfer_bytes"]),
        "state_transfer_bytes": int(transition["state_transfer_bytes"]),
        "state_transfer_status": (
            dict(transition.get("state_transfer") or {}).get("status")
        ),
    }


def _estimate_view(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "current_service",
        "target_prepare",
        "target_state_commit",
        "cost_status",
        "estimated_total_seconds",
        "deadline_fit",
        "target_prepare_contact_fit",
        "raw_full_step_contact_fit",
        "raw_trace_fit",
        "raw_execution_fit",
        "phase_status",
        "estimated_phase_seconds",
        "model_transfer_bytes",
        "state_transfer_bytes",
        "unknown_reasons",
    )
    return {key: deepcopy(row[key]) for key in keys}


def _native_case(
    name: str,
    *,
    current_ready: bool,
    contact_seconds: float | None = None,
) -> dict[str, Any]:
    env, config, _ = _prepared_env(current_ready=current_ready)
    state, mask = _public_input(
        env, config, contact_seconds=contact_seconds
    )
    if contact_seconds is not None:
        env._physical_contact_budget_seconds = (  # type: ignore[method-assign]
            lambda: float(contact_seconds)
        )
    estimate = estimate_public_actions(state, mask)["actions"]["4"]
    _, _, _, _, info = env.step(4)
    transition = info["transition"]
    return {
        "case": name,
        "public_estimate": _estimate_view(estimate),
        "native_transition": _transition_view(transition),
        "estimated_matches_native_cost": (
            estimate["estimated_total_seconds"] is not None
            and abs(
                float(estimate["estimated_total_seconds"])
                - float(transition["step_cost_seconds"])
            )
            <= 1e-9
        ),
    }


def _unknown_case() -> dict[str, Any]:
    env, config, target = _prepared_env(current_ready=True)
    state, mask = _public_input(env, config)
    target_row = next(
        row for row in state["rsus"] if str(row["rsu_id"]) == target
    )
    target_row["cache_used_bytes"] = target_row["cache_capacity"]
    estimate = estimate_public_actions(state, mask)["actions"]["4"]
    return {
        "case": "private_eviction_order_unknown",
        "public_estimate": _estimate_view(estimate),
        "expected_contract": {
            "cost_status": "unknown_required_phase",
            "estimated_total_seconds": None,
            "deadline_fit": "unknown",
        },
    }


def _trace_visibility_case() -> dict[str, Any]:
    env, config, _ = _prepared_env(current_ready=True)
    state, mask = _public_input(env, config)
    left = deepcopy(state)
    right = deepcopy(state)
    left["calibrated_context"]["time_contract"][
        "trace_remaining_seconds"
    ] = 100.0
    right["calibrated_context"]["time_contract"][
        "trace_remaining_seconds"
    ] = 0.0
    left_estimate = estimate_public_actions(left, mask)
    right_estimate = estimate_public_actions(right, mask)
    return {
        "case": "hidden_trace_end",
        "estimates_equal": left_estimate == right_estimate,
        "raw_trace_fit": left_estimate["actions"]["4"]["raw_trace_fit"],
        "forbidden_fields": left_estimate["forbidden_fields"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/analysis" / RUN_ID,
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cases = [
        _native_case("ready_action4_commit", current_ready=True),
        _native_case("missing_current_action4_failure", current_ready=False),
        _native_case(
            "prepare_contact_rollback",
            current_ready=True,
            contact_seconds=8.0,
        ),
        _native_case(
            "prepare_feasible_full_step_contact_insufficient",
            current_ready=True,
            contact_seconds=9.0,
        ),
        _unknown_case(),
        _trace_visibility_case(),
    ]
    receipt = {
        "schema_version": "cscwd_public_estimator_phase_conformance_v1",
        "run_id": RUN_ID,
        "evidence_scope": "synthetic_contract_only_not_algorithm_performance",
        "raw_rows_read": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "evaluation_episodes": 0,
        "cases": cases,
    }
    conformance_path = output_root / "conformance.json"
    conformance_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest = {
        "schema_version": "cscwd_public_estimator_phase_conformance_manifest_v1",
        "run_id": RUN_ID,
        "status": "complete",
        "source_commit": source_commit,
        "files": [
            {
                "path": "conformance.json",
                "sha256": _sha256(conformance_path),
            }
        ],
        "counts": {
            "synthetic_cases": len(cases),
            "raw_rows_read": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "evaluation_episodes": 0,
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "conformance_sha256": _sha256(conformance_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
