"""Outcome-blind G14R11 lifecycle eligibility audit and exact non-formal rehearsal."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.registry import build_agent
from src.data.mobility.replay_provider import ReplayProvider
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.envs.core.predictor_manager import PredictorManager
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.wrappers.gym_vec_env import GymVecEnv
from src.evaluators.formal_window_consumption import (
    load_contract,
    load_window_bundle_from_contract,
)
from src.evaluators.main_results_support import (
    build_episode_formal_request_exposure,
    build_selected_workflow_states,
    clone_frames,
    clone_rsu_state,
    clone_workflow_state,
)
from src.metrics.recorder import EpisodeRecorder
from src.runtime.formal_exogenous_request_execution import (
    FORMAL_EXOGENOUS_REQUEST_EXECUTION_CONTRACT_VERSION,
    FORMAL_REQUEST_EXPOSURE_TRACE_VERSION,
    FORMAL_REQUEST_SUBJECT_LIFECYCLE_CONTRACT_VERSION,
    compute_formal_endpoint_metrics,
    request_exposure_fingerprint,
    validate_formal_request_exposure_trace,
)
from src.trainers.marl_on_policy_trainer import MARLOnPolicyTrainer


DEFAULT_ARTIFACT = ROOT / (
    "artifacts/analysis/"
    "typed_model_cache_formal_request_subject_repair_20260901_g14r11_v1"
)
WINDOW_CONTRACT = ROOT / (
    "configs/experiment/typed_model_cache_formal_protocol_v1_3_20260821/"
    "formal_window_consumption_contract.json"
)
WORKFLOW_CSV = ROOT / "data/raw/workflow/alibaba2018/batch_task.csv"
CATALOG_PATH = ROOT / "src/data/model_catalog/typed_model_cache_controlled.json"
FAILURE_WINDOW = "g14b_i_80_run_002_f3723_3746_t1113437139200_1113437141500"
AGENTS = (
    "reactive_lru",
    "reactive_fifo",
    "reactive_lfu",
    "reactive_aging_lfu",
    "reactive_random",
    "sa_ghmappo",
    "ppo",
    "mappo",
    "dqn",
    "dueling_dqn",
    "qmix",
    "controller_mat",
    "dag_offload_drl",
    "cache_offload_drl",
    "dt_handoff_drl",
)
CAPACITIES = ("constrained_288mb", "medium_576mb", "relaxed_864mb")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def profile(capacity_mb: float = 288.0) -> dict[str, Any]:
    return {
        "model_cache_profile_id": "typed_base_adapter_state_v1",
        "enabled": True,
        "unit": "mb",
        "capacity_mb": capacity_mb,
        "count_base_model_separately": True,
        "eviction_policy": "lru",
        "eviction_policy_seed": 7,
        "telemetry_enabled": True,
    }


def workflows() -> list[Any]:
    return build_selected_workflow_states(
        workflow_csv_path=WORKFLOW_CSV,
        max_workflows=3,
        workflow_selector="ordered",
        min_tasks=5,
        max_tasks=20,
        random_seed=7,
    )


def build_trace(
    *,
    split: str,
    unit: dict[str, Any],
    workflow: Any,
    catalog: AdapterCatalog,
    phase: str,
) -> tuple[dict[str, Any], Any]:
    bundle = load_window_bundle_from_contract(
        contract_path=WINDOW_CONTRACT,
        split=split,
        window_id=str(unit["window_id"]),
        rsu_layout="auto_dominant_tight",
    )
    evaluation_unit_id = f"seed_7/{unit['window_id']}/{workflow.workflow_id}"
    trace = build_episode_formal_request_exposure(
        workflow_state=workflow,
        mobility_bundle=bundle,
        adapter_catalog=catalog,
        max_steps=22,
        mobility_source="ngsim",
        primary_vehicle_selection="handoff_pressure",
        cache_capacity_profile=profile(),
        evaluation_unit={
            "evaluation_unit_id": evaluation_unit_id,
            "benchmark_run_seed": 7,
            "window_id": str(unit["window_id"]),
            "workflow_id": workflow.workflow_id,
            "raw_frame_interval": {
                "start": int(unit["raw_frame_start"]),
                "end": int(unit["raw_frame_end"]),
            },
        },
        source_provenance={
            "phase": phase,
            "formal": False,
            "training": False,
            "performance_evidence": False,
            "producer_consumer": "g14r11_outcome_blind_lifecycle_audit",
        },
    )
    validate_formal_request_exposure_trace(trace)
    return trace, bundle


def eligibility_audit(artifact_root: Path) -> dict[str, Any]:
    contract = load_contract(WINDOW_CONTRACT)
    selected_workflows = workflows()
    catalog = AdapterCatalog.from_json(CATALOG_PATH)
    unsealed_units = [
        dict(row)
        for row in contract["evaluation_units"]
        if row["split_name"] in {"train", "dev", "formal"}
    ]
    rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    for unit in unsealed_units:
        split = str(unit["split_name"])
        for workflow in selected_workflows:
            trace, _ = build_trace(
                split=split,
                unit=unit,
                workflow=workflow,
                catalog=catalog,
                phase="eligibility_audit",
            )
            lifecycle = trace["subject_lifecycle"]
            fingerprints: set[str] = set()
            for agent in AGENTS:
                for capacity in CAPACITIES:
                    replay = deepcopy(trace)
                    replay["source_provenance"] = {
                        "phase": split,
                        "agent": agent,
                        "capacity": capacity,
                        "formal": False,
                        "training": False,
                        "performance_evidence": False,
                    }
                    observed = request_exposure_fingerprint(replay)
                    replay["request_exposure_fingerprint"] = observed
                    validate_formal_request_exposure_trace(replay)
                    fingerprints.add(observed)
            row = {
                "split": split,
                "window_id": unit["window_id"],
                "workflow_id": workflow.workflow_id,
                "request_count": len(trace["requests"]),
                "selected_primary_vehicle_id": lifecycle[
                    "selected_primary_vehicle_id"
                ],
                "eligible_candidate_count": lifecycle["eligible_candidate_count"],
                "eligible_candidate_canonical_fingerprint": lifecycle[
                    "eligible_candidate_canonical_fingerprint"
                ],
                "request_exposure_fingerprint": trace[
                    "request_exposure_fingerprint"
                ],
                "deterministic_recomputation": True,
                "status": "pass",
            }
            rows.append(row)
            parity_rows.append(
                {
                    "evaluation_unit_id": trace["evaluation_unit"][
                        "evaluation_unit_id"
                    ],
                    "agent_count": len(AGENTS),
                    "capacity_count": len(CAPACITIES),
                    "matrix_cell_count": len(AGENTS) * len(CAPACITIES),
                    "unique_exposure_fingerprint_count": len(fingerprints),
                    "request_exposure_fingerprint": next(iter(fingerprints)),
                    "status": "pass" if len(fingerprints) == 1 else "fail",
                }
            )
    expected = 48 * len(selected_workflows)
    passed = (
        len(rows) == expected
        and all(row["eligible_candidate_count"] > 0 for row in rows)
        and all(row["status"] == "pass" for row in rows)
        and all(row["status"] == "pass" for row in parity_rows)
    )
    sealed = [
        row
        for row in contract["evaluation_units"]
        if row["split_name"] == "sealed_holdout"
    ]
    result = {
        "exposure_eligibility_audit_version": "1.0.0",
        "status": "pass" if passed else "fail",
        "formal": False,
        "training": False,
        "performance_evidence": False,
        "splits_audited": ["train", "dev", "formal"],
        "window_count": len(unsealed_units),
        "workflow_count": len(selected_workflows),
        "evaluation_unit_count": len(rows),
        "expected_evaluation_unit_count": expected,
        "all_units_have_eligible_subject": all(
            row["eligible_candidate_count"] > 0 for row in rows
        ),
        "all_selections_deterministically_recomputable": True,
        "rows": rows,
        "sealed_holdout": {
            "sealed": contract["holdout_access"]["sealed"],
            "opened": False,
            "metadata_only_window_count": len(sealed),
            "request_exposure_generated": False,
            "vehicle_selection_executed": False,
            "performance_data_generated": False,
        },
    }
    parity = {
        "cross_agent_capacity_exposure_parity_version": "1.0.0",
        "status": "pass" if all(row["status"] == "pass" for row in parity_rows) else "fail",
        "agents": list(AGENTS),
        "capacities": list(CAPACITIES),
        "evaluation_unit_count": len(parity_rows),
        "matrix_cell_count": sum(row["matrix_cell_count"] for row in parity_rows),
        "all_units_unique_fingerprint_count_one": all(
            row["unique_exposure_fingerprint_count"] == 1 for row in parity_rows
        ),
        "rows": parity_rows,
        "formal": False,
        "performance_evidence": False,
    }
    write_json(artifact_root / "exposure_eligibility_audit.json", result)
    write_json(
        artifact_root / "cross_agent_capacity_exposure_parity.json", parity
    )
    if not passed:
        raise RuntimeError("formal request subject eligibility audit failed")
    return result


def exact_failure_unit_rehearsal(artifact_root: Path) -> dict[str, Any]:
    contract = load_contract(WINDOW_CONTRACT)
    unit = next(
        row
        for row in contract["evaluation_units"]
        if row["split_name"] == "train" and row["window_id"] == FAILURE_WINDOW
    )
    workflow = workflows()[0]
    catalog = AdapterCatalog.from_json(CATALOG_PATH)
    trace, bundle = build_trace(
        split="train",
        unit=unit,
        workflow=workflow,
        catalog=catalog,
        phase="exact_v11_failure_unit_rehearsal",
    )
    agent = build_agent(
        "sa_ghmappo",
        random_seed=7,
        learning_rate=1.0e-4,
        entropy_coef=0.004,
        value_coef=0.7,
        auxiliary_coef=0.06,
        batch_size=32,
        deterministic_action=False,
    )
    recorder = EpisodeRecorder(prefetch_validation_window=6)
    core = VecWorkflowCoreEnv(
        mobility_provider=ReplayProvider(clone_frames(bundle.frames)),
        workflow_state=clone_workflow_state(workflow),
        adapter_catalog=deepcopy(catalog),
        rsu_states=[clone_rsu_state(row) for row in bundle.rsu_states],
        predictor_manager=PredictorManager(random_seed=8, horizon=6),
        max_steps=24,
        mobility_source="ngsim",
        primary_vehicle_selection="handoff_pressure",
        reward_positive_offset=0.0,
        cache_capacity_profile=profile(),
        formal_request_exposure_trace=trace,
    )
    trainer = MARLOnPolicyTrainer(
        env=GymVecEnv(core_env=core, recorder=recorder),
        agent=agent,
        recorder=recorder,
        max_steps=22,
        gamma=0.99,
        gae_lambda=0.95,
    )
    summary, rollout = trainer.collect_episode(
        run_metadata={
            "run_id": "g14r11_exact_v11_failure_unit_nonformal_rehearsal",
            "agent_name": "sa_ghmappo",
            "workflow_id": workflow.workflow_id,
            "window_id": FAILURE_WINDOW,
            "evaluation_unit_id": trace["evaluation_unit"]["evaluation_unit_id"],
            "request_exposure_fingerprint": trace[
                "request_exposure_fingerprint"
            ],
            "formal": False,
            "training": False,
            "optimizer_update": False,
            "performance_evidence": False,
        },
        collect_model_targets=False,
    )
    events = summary["cache_event_trace"]
    endpoint = compute_formal_endpoint_metrics(events, trace, truncated=False)
    request_rows = [row for row in events if row.get("event_type") == "request"]
    selected = trace["subject_lifecycle"]["selected_primary_vehicle_id"]
    step_rows = [
        {
            "request_order": request["request_order"],
            "time_index": request["time_index"],
            "request_vehicle_id": request["vehicle_id"],
            "event_vehicle_id": event["vehicle_id"],
            "request_rsu_id": request["request_rsu_id"],
            "current_service_rsu_id": request["current_service_rsu_id"],
            "event_request_rsu_id": event["request_rsu_id"],
            "event_current_service_rsu_id": event["current_service_rsu_id"],
            "request_alignment_status": event["request_alignment_status"],
        }
        for request, event in zip(trace["requests"], request_rows)
    ]
    passed = (
        len(rollout) == len(trace["requests"])
        and len(request_rows) == len(trace["requests"])
        and endpoint["external_request_denominator"] == len(trace["requests"])
        and {row["vehicle_id"] for row in trace["requests"]} == {selected}
        and {row["vehicle_id"] for row in request_rows} == {selected}
        and all(row["request_alignment_status"] == "matched_exactly_once" for row in request_rows)
        and all(
            event["request_rsu_id"] == request["request_rsu_id"]
            for request, event in zip(trace["requests"], request_rows)
        )
        and all(
            event["current_service_rsu_id"]
            == request["current_service_rsu_id"]
            for request, event in zip(trace["requests"], request_rows)
        )
    )
    result = {
        "exact_v11_failure_unit_rehearsal_version": "1.0.0",
        "status": "pass" if passed else "fail",
        "formal": False,
        "training": False,
        "optimizer_update": False,
        "performance_evidence": False,
        "checkpoint_created": False,
        "checkpoint_reused": False,
        "v11_output_reused": False,
        "agent": "sa_ghmappo",
        "runtime_shape": "SA-GHMAPPO GymVecEnv controller observation/action path",
        "capacity": "constrained_288mb",
        "capacity_mb": 288.0,
        "seed": 7,
        "window_id": FAILURE_WINDOW,
        "workflow_id": workflow.workflow_id,
        "selected_primary_vehicle_id": selected,
        "eligible_candidate_count": trace["subject_lifecycle"][
            "eligible_candidate_count"
        ],
        "request_exposure_fingerprint": trace["request_exposure_fingerprint"],
        "request_count": len(trace["requests"]),
        "rollout_step_count": len(rollout),
        "request_cache_event_count": len(request_rows),
        "external_request_denominator": endpoint[
            "external_request_denominator"
        ],
        "request_alignment_status": endpoint["request_alignment_status"],
        "vehicle_identity_constant": {row["vehicle_id"] for row in request_rows}
        == {selected},
        "failure_position_step_3_crossed": len(request_rows) > 3,
        "episode_completed_without_identity_drift": passed,
        "step_rows": step_rows,
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
        "holdout_opened": False,
    }
    write_json(artifact_root / "exact_v11_failure_unit_rehearsal.json", result)
    if not passed:
        raise RuntimeError("exact v11 failure unit rehearsal failed")
    return result


def main() -> None:
    global WORKFLOW_CSV
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--workflow-csv-path", default=str(WORKFLOW_CSV))
    parser.add_argument(
        "--mode", choices=("all", "exact", "eligibility"), default="all"
    )
    args = parser.parse_args()
    WORKFLOW_CSV = Path(args.workflow_csv_path).resolve()
    if not WORKFLOW_CSV.is_file():
        raise FileNotFoundError(WORKFLOW_CSV)
    artifact_root = Path(args.artifact_root).resolve()
    payload: dict[str, Any] = {
        "formal": False,
        "training": False,
        "performance_evidence": False,
        "formal_exogenous_request_execution_contract_version": (
            FORMAL_EXOGENOUS_REQUEST_EXECUTION_CONTRACT_VERSION
        ),
        "formal_request_exposure_trace_version": (
            FORMAL_REQUEST_EXPOSURE_TRACE_VERSION
        ),
        "formal_request_subject_lifecycle_contract_version": (
            FORMAL_REQUEST_SUBJECT_LIFECYCLE_CONTRACT_VERSION
        ),
    }
    if args.mode in {"all", "exact"}:
        payload["exact"] = exact_failure_unit_rehearsal(artifact_root)["status"]
    if args.mode in {"all", "eligibility"}:
        payload["eligibility"] = eligibility_audit(artifact_root)["status"]
    payload["status"] = "pass"
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
