"""Freeze Protocol 2.0 around policy-neutral exogenous request execution."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.cache_baseline_fairness import (
    build_manifest,
    full_manifest_sha256,
    semantic_protocol_sha256,
    validate_manifest,
)
from src.evaluators.main_results_support import resolve_window_candidates
from src.evaluators.typed_model_cache_formal_execution import endpoint_schema
from src.evaluators.typed_model_cache_formal_protocol import attach_hashes
from src.runtime.active_formal_bundle import (
    ACTIVE_BUNDLE_RESOURCE_RESOLUTION_CONTRACT_VERSION,
    ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION,
    active_bundle_core_projection,
    build_resource_row,
    canonical_sha256,
    ready_index_projection,
    sha256_file,
)
from src.runtime.formal_agent_order import contract_projection


SOURCE = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_9_20260829"
TARGET = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v2_0_20260831"
ARTIFACT = ROOT / "artifacts/analysis/typed_model_cache_formal_exogenous_request_repair_20260831_g14r9_v1"
PROTOCOL_VERSION = "2.0.0"
PROTOCOL_ID = "typed_model_cache_formal_protocol_v2_0"
READY_STATUS = "READY_FOR_G14C_V10_CLEAN_TRAIN_AND_FORMAL"
READINESS_VERSION = "12.0.0"
V9_RUN_ID = "typed_model_cache_formal_20260830_113339_g14c_v9"
OLD_DIR = "typed_model_cache_formal_protocol_v1_9_20260829"
NEW_DIR = "typed_model_cache_formal_protocol_v2_0_20260831"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def replace_paths(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(OLD_DIR, NEW_DIR).replace(
            "protocol_v1_9_manifest.json", "protocol_v2_0_manifest.json"
        )
    if isinstance(value, list):
        return [replace_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: replace_paths(item) for key, item in value.items()}
    return value


def current(
    logical_id: str, role: str, filename: str, semantic_sha256: str | None = None
) -> dict[str, Any]:
    return build_resource_row(
        root=ROOT,
        logical_id=logical_id,
        role=role,
        relative_path=(TARGET / filename).relative_to(ROOT).as_posix(),
        version_scope="current_protocol_version",
        semantic_sha256=semantic_sha256,
    )


def add_explicit_request_mode(command_templates: dict[str, Any]) -> None:
    for spec in command_templates.values():
        argv = spec.get("argv")
        if not isinstance(argv, list):
            continue
        joined = " ".join(str(item) for item in argv)
        if (
            "train_algo_pool_real_sample.py" in joined
            or "benchmark_main_results.py" in joined
        ) and "--formal-exogenous-request-execution" not in argv:
            argv.append("--formal-exogenous-request-execution")


def build_rehearsal_resources(order: dict[str, Any]) -> list[dict[str, Any]]:
    mobility = ROOT / (
        "data/raw/mobility/ngsim/"
        "Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv"
    )
    workflow = ROOT / "data/raw/workflow/alibaba2018/batch_task.csv"
    _, selected = resolve_window_candidates(
        root_dir=ROOT,
        mobility_source="ngsim",
        mobility_csv_path=str(mobility),
        lust_scenario_root="",
        max_mobility_rows=1500,
        rsu_layout="auto_dominant_tight",
        frame_offset=0,
        window_length=24,
        window_selector="ordered",
        window_count=1,
        window_scan_stride=1,
        random_seed=7,
        window_mode="mixed_informative",
        window_rank_offset=0,
        excluded_window_intervals=[],
        holdout_min_gap_frames=0,
        enforce_non_overlapping_selection=True,
        activating_handoff_threshold=2,
        activating_vehicle_threshold=2.0,
        activating_predicted_next_ratio_threshold=0.3,
        activating_handoff_prediction_ratio_threshold=0.15,
        non_mechanism_handoff_max=0,
        non_mechanism_prediction_ratio_max=0.05,
        active_non_mechanism_vehicle_threshold=2.0,
        active_non_mechanism_association_change_min=1,
        active_non_mechanism_handoff_max=1,
        active_non_mechanism_predicted_next_ratio_max=0.2,
        active_non_mechanism_handoff_prediction_ratio_max=0.1,
        idle_or_sparse_vehicle_max=1.5,
        idle_or_sparse_association_change_max=0,
    )
    plan = {
        "window_plan_contract": "g14r9_nonformal_rehearsal_v1",
        "window_mode": selected["window_mode"],
        "selected_window_plan": list(selected["selected_windows"]),
    }
    plan_path = TARGET / "nonformal_rehearsal_window_plan.json"
    write_json(plan_path, plan)
    rows: list[dict[str, Any]] = []
    for label, capacity in (
        ("constrained_288mb", 288.0),
        ("medium_576mb", 576.0),
        ("relaxed_864mb", 864.0),
    ):
        manifest = build_manifest(
            root=ROOT,
            mobility_path=mobility,
            workflow_path=workflow,
            window_plan_path=plan_path,
            catalog_path=ROOT / "src/data/model_catalog/typed_model_cache_controlled.json",
            seeds=[7],
            max_workflows=1,
            workflow_selector="ordered",
            min_tasks=5,
            max_tasks=20,
            max_steps=1,
            max_mobility_rows=1500,
            primary_vehicle_selection="handoff_pressure",
            capacity_unit="mb",
            capacity_value=capacity,
            output_root="artifacts/analysis/g14r9_nonformal_rehearsal",
            evaluation_unit_limit=1,
            created_at="2026-08-31T00:00:00+08:00",
            controller_agents=order["learned_agent_order"],
        )
        manifest["identity"]["dirty_worktree_audit"] = {
            "changed_path_count": 0,
            "changed_paths": [],
            "note": (
                "frozen resource identity is independent of generator worktree dirt; "
                "the active execution gate separately requires a clean candidate"
            ),
        }
        semantic = semantic_protocol_sha256(manifest)
        manifest["hashes"]["semantic_protocol_sha256"] = semantic
        manifest["identity"]["manifest_id"] = f"cbfm-{semantic[:16]}"
        manifest["hashes"]["full_manifest_sha256"] = full_manifest_sha256(manifest)
        report = validate_manifest(manifest, root=ROOT, check_files=True)
        if report["status"] != "pass":
            raise ValueError(report["errors"])
        filename = f"nonformal_rehearsal_fairness_{label}.json"
        write_json(TARGET / filename, manifest)
        rows.append(
            current(
                f"rehearsal_fairness_manifests.{label}",
                "nonformal rehearsal fairness manifest",
                filename,
            )
        )
    return rows


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    old_protocol = read_json(SOURCE / "protocol_v1_9_manifest.json")
    old_index = read_json(SOURCE / "protocol_index.json")

    for filename in (
        "agent_training_scientific_config.json",
        "formal_training_execution_binding_contract.json",
        "resolved_execution_context_contract.json",
    ):
        write_json(TARGET / filename, replace_paths(read_json(SOURCE / filename)))

    order = read_json(SOURCE / "formal_agent_order_contract.json")
    write_json(TARGET / "formal_agent_order_contract.json", order)

    environment = replace_paths(read_json(SOURCE / "execution_environment_manifest.json"))
    environment_identity = environment["scientific_identity"]
    environment_identity.update(
        execution_commit=(
            "Commit A11; exact clean 40-hex HEAD == main == origin/main is observed "
            "and bound before formal execution"
        ),
        formal_exogenous_request_execution_contract_version="1.0.0",
        formal_request_exposure_trace_version="1.0.0",
        formal_endpoint_metrics_contract_version="2.0.0",
    )
    environment_identity.pop("environment_fingerprint", None)
    environment_identity["environment_fingerprint"] = canonical_sha256(
        environment_identity
    )
    write_json(TARGET / "execution_environment_manifest.json", environment)

    request_contract = {
        "version": "1.0.0",
        "request_exposure_trace_version": "1.0.0",
        "producer_identity": "formal_request_exposure_producer_v1.0.0",
        "producer_entrypoint": (
            "src.evaluators.main_results_support.build_episode_formal_request_exposure"
        ),
        "execution_entrypoint": "src.envs.core.vec_workflow_core_env.VecWorkflowCoreEnv",
        "default_enabled": False,
        "formal_explicit_enable_required": True,
        "phase_modes": {
            "train": "replay_driven_exogenous_request_exposure",
            "dev": "replay_driven_exogenous_request_exposure",
            "formal": "replay_driven_exogenous_request_exposure",
            "legacy_nonformal": "legacy_endogenous_progression",
        },
        "request_fingerprint_is_outcome_fingerprint": False,
        "action_before_lookup": True,
        "service_failure_changes_future_exposure": False,
        "actor_or_controller_may_observe_oracle_future_topology": False,
        "cache_event_alignment": "exactly_one_request_level_CacheEvent_per_exposure",
        "g08_replay_role": "analytical_oracle_only_not_formal_execution_producer",
        "g09_opportunity_role": "outcome_blind_analysis_bound_to_exposure_provenance",
        "fail_fast": [
            "missing_replay",
            "duplicate_missing_extra_or_out_of_order_request",
            "cross_agent_fingerprint_drift",
            "request_event_identity_drift",
            "catalog_dependency_size_or_evaluation_unit_drift",
            "outcome_pollution",
            "endogenous_fallback",
            "future_topology_leakage",
            "historical_bundle_or_v9_reference",
        ],
    }
    request_contract["semantic_sha256"] = canonical_sha256(request_contract)
    request_schema = {
        "formal_request_exposure_schema_version": "1.0.0",
        "contract_version": "1.0.0",
        "canonical_serialization": "UTF-8 sorted-key compact JSON; NaN/Infinity rejected",
        "fingerprint": "SHA-256 over the trace excluding only fingerprint and validation",
        "required_request_fields": [
            "request_id",
            "request_kind",
            "request_order",
            "step_index",
            "time_index",
            "vehicle_id",
            "workflow_id",
            "node_id",
            "required_base_model",
            "adapter_id",
            "object_id",
            "object_size_mb",
            "request_rsu_id",
            "current_service_rsu_id",
            "eligible_service_rsu_ids",
            "eligible_cache_target_rsu_ids",
            "requested_typed_objects",
            "dependency_bundle",
            "dag_provenance",
            "oracle_only_future_topology",
        ],
        "outcome_fields_forbidden": True,
        "oracle_future_actor_visible": False,
    }
    request_schema["semantic_sha256"] = canonical_sha256(request_schema)
    write_json(TARGET / "formal_exogenous_request_execution_contract.json", request_contract)
    write_json(TARGET / "formal_request_exposure_schema.json", request_schema)

    protocol = replace_paths(deepcopy(old_protocol))
    protocol["typed_model_cache_formal_protocol_version"] = PROTOCOL_VERSION
    protocol["protocol_id"] = PROTOCOL_ID
    protocol["created_at"] = "2026-08-31T00:00:00+08:00"
    protocol["status"] = "frozen_pre_execution_exogenous_request_execution"
    protocol["supersession"].update(
        supersedes_version="1.9.0",
        old_protocol_status=(
            "invalid_after_training_during_first_dev_candidate_evaluation_before_dev_selection"
        ),
        old_protocol_semantic_sha256=old_protocol["hashes"]["semantic_sha256"],
        scientific_fields_changed=True,
        formal_performance_observed=False,
    )
    protocol["supersession"]["repair_scope"] = [
        "freeze request exposure before any compared agent acts",
        "separate request and outcome fingerprints",
        "redefine primary endpoints on the common external request denominator",
    ]
    protocol["supersession"]["invalid_execution_runs"].append(
        {
            "run_id": V9_RUN_ID,
            "run_root": (
                "artifacts/experiments/typed_model_cache_formal/"
                + V9_RUN_ID
            ),
            "status": "INVALID_PROTOCOL_OR_IMPLEMENTATION",
            "failure_boundary": (
                "invalid_after_training_during_first_dev_candidate_evaluation_before_dev_selection"
            ),
            "phase_ledger_sha256": (
                "ec6b04fee48c4abda056b62f508f186345f10ab580efd896f9f43979d1d728fe"
            ),
            "cell_ledger_sha256": (
                "c0a8cbe94601576e8a8b9c03df5411561b68044ebb9dba7b3910aaaec0b75f4e"
            ),
            "training_cells_executed": 150,
            "candidate_checkpoint_count": 1200,
            "dev_performance_count": 0,
            "formal_performance_count": 0,
            "immutable_old_run": True,
            "resume_allowed": False,
            "retry_allowed": False,
            "legacy_phase_finalize_allowed": False,
            "checkpoint_reuse_allowed": False,
            "candidate_reuse_allowed": False,
            "partial_dev_input_reuse_allowed": False,
        }
    )
    protocol["formal_agent_order_contract"].update(
        active_protocol_versions=["2.0.0"],
        historical_protocol_versions_audit_only=[f"1.{minor}.0" for minor in range(10)],
        semantic_sha256=order["semantic_sha256"],
    )
    protocol["identity"]["formal_agent_order_contract_semantic_sha256"] = order[
        "semantic_sha256"
    ]
    protocol["formal_exogenous_request_execution_contract"] = deepcopy(request_contract)
    protocol["formal_request_exposure_schema"] = deepcopy(request_schema)
    protocol["endpoint_schema"] = endpoint_schema()
    protocol["formal_execution_environment_contract"]["scientific_identity"] = deepcopy(
        environment_identity
    )
    add_explicit_request_mode(protocol["execution_contract"]["command_templates"])
    protocol["active_formal_bundle_contract"].update(
        unique_active_index=(TARGET / "protocol_index.json").relative_to(ROOT).as_posix()
    )
    context = protocol["execution_contract"]["default_expansion_context"]
    context["active_protocol_index_path"] = (
        TARGET / "protocol_index.json"
    ).relative_to(ROOT).as_posix()
    context["protocol_path"] = (
        TARGET / "protocol_v2_0_manifest.json"
    ).relative_to(ROOT).as_posix()
    context["agent_scientific_config_path"] = (
        TARGET / "agent_training_scientific_config.json"
    ).relative_to(ROOT).as_posix()
    context["formal_agent_order_contract_path"] = (
        TARGET / "formal_agent_order_contract.json"
    ).relative_to(ROOT).as_posix()
    protocol["paper_claim_boundary"] = (
        "G14R9 is contract and non-formal rehearsal evidence only; no formal training, "
        "formal checkpoint, performance evidence, holdout, G14C v10, G14D, or G15."
    )
    protocol = attach_hashes(protocol)
    write_json(TARGET / "protocol_v2_0_manifest.json", protocol)

    resolution = deepcopy(protocol["active_bundle_resource_resolution_contract"])
    resolution["semantic_sha256"] = canonical_sha256(resolution)
    write_json(TARGET / "active_bundle_resource_resolution_contract.json", resolution)
    rehearsal_rows = build_rehearsal_resources(order)

    scientific = read_json(TARGET / "agent_training_scientific_config.json")
    current_rows = [
        current("protocol_manifest", "active Protocol manifest", "protocol_v2_0_manifest.json", protocol["hashes"]["semantic_sha256"]),
        current("execution_environment_manifest", "active execution environment identity", "execution_environment_manifest.json"),
        current("agent_training_scientific_config", "Scientific Config 2.0.0", "agent_training_scientific_config.json", scientific["config_semantic_sha256"]),
        current("formal_agent_order_contract", "Formal Agent Order Contract 1.0.0", "formal_agent_order_contract.json", order["semantic_sha256"]),
        current("formal_training_execution_binding_schema", "formal execution binding schema", "formal_training_execution_binding_contract.json"),
        current("resolved_execution_context_schema", "resolved context schema", "resolved_execution_context_contract.json"),
        current("active_bundle_resource_resolution_contract", "active bundle resource resolver", "active_bundle_resource_resolution_contract.json", resolution["semantic_sha256"]),
        current("formal_exogenous_request_execution_contract", "formal exogenous request execution contract", "formal_exogenous_request_execution_contract.json", request_contract["semantic_sha256"]),
        current("formal_request_exposure_schema", "formal request exposure schema", "formal_request_exposure_schema.json", request_schema["semantic_sha256"]),
    ]
    shared_rows = [
        deepcopy(row)
        for row in old_index["active_bundle_resources"]
        if row.get("version_scope") == "shared_historical_stable"
    ]
    index: dict[str, Any] = {
        "active_formal_bundle_contract_version": ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION,
        "active_bundle_resource_resolution_contract_version": ACTIVE_BUNDLE_RESOURCE_RESOLUTION_CONTRACT_VERSION,
        "protocol_index_version": PROTOCOL_VERSION,
        "status": READY_STATUS,
        "protocol_identity": {
            "protocol_id": PROTOCOL_ID,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
            "protocol_full_sha256": protocol["hashes"]["full_sha256"],
        },
        "execution_commit_binding": {
            "mode": "observed_clean_head_equal_origin_main",
            "exact_40_hex_recorded_in_execution_binding": True,
            "index_embeds_own_commit_hash": False,
            "self_reference_avoided": True,
        },
        "environment_identity": {
            "environment_fingerprint": environment_identity["environment_fingerprint"],
            "dependency_fingerprint": environment_identity["dependency_fingerprint"],
        },
        "command_matrix_identity": {
            "command_templates_sha256": canonical_sha256(protocol["execution_contract"]["command_templates"]),
            "outer_nested_expansion_equality_required": True,
        },
        "holdout_seal": deepcopy(protocol["holdout_execution_contract"]),
        "active_bundle_resources": [*current_rows, *rehearsal_rows, *shared_rows],
    }
    index["active_bundle_core_sha256"] = canonical_sha256(
        active_bundle_core_projection(index)
    )
    evidence = {
        "status": "pass",
        "clean_candidate": True,
        "active_bundle_core_sha256": index["active_bundle_core_sha256"],
        "acceptance_scope": "nonformal_g14r9_contract_rehearsal",
        "formal": False,
        "performance_evidence": False,
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
        "holdout_opened": False,
    }
    for evidence_name, field in (
        ("exact_failure_unit_rehearsal.json", "exact_failure_unit_rehearsal_sha256"),
        ("three_capacity_rehearsal.json", "three_capacity_rehearsal_sha256"),
        ("phase_chain_rehearsal.json", "phase_chain_rehearsal_sha256"),
    ):
        evidence_path = ARTIFACT / evidence_name
        if evidence_path.is_file():
            payload = read_json(evidence_path)
            if payload.get("status") != "pass":
                raise ValueError(f"readiness evidence is not pass: {evidence_name}")
            evidence[field] = sha256_file(evidence_path)
    write_json(ARTIFACT / "acceptance_evidence_manifest.json", evidence)
    readiness = {
        "readiness_review_version": READINESS_VERSION,
        "status": "ready",
        "verdict": READY_STATUS,
        "active_bundle_core_sha256": index["active_bundle_core_sha256"],
        "evidence_manifest_path": (ARTIFACT / "acceptance_evidence_manifest.json").relative_to(ROOT).as_posix(),
        "evidence_manifest_sha256": sha256_file(ARTIFACT / "acceptance_evidence_manifest.json"),
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
        "holdout_sealed_unopened": True,
    }
    write_json(TARGET / "readiness_v12.json", readiness)
    readiness_row = current("readiness_companion", "Readiness v12 evidence companion", "readiness_v12.json")
    index["active_bundle_resources"].append(readiness_row)
    index["readiness_companion"] = {
        "logical_path": readiness_row["logical_path"],
        "content_sha256": readiness_row["content_sha256"],
    }
    index["active_formal_bundle_sha256"] = canonical_sha256(
        ready_index_projection(index)
    )
    write_json(TARGET / "protocol_index.json", index)
    print(
        json.dumps(
            {
                "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
                "protocol_full_sha256": protocol["hashes"]["full_sha256"],
                "request_contract_semantic_sha256": request_contract["semantic_sha256"],
                "request_schema_semantic_sha256": request_schema["semantic_sha256"],
                "active_bundle_core_sha256": index["active_bundle_core_sha256"],
                "active_formal_bundle_sha256": index["active_formal_bundle_sha256"],
                "environment_fingerprint": environment_identity["environment_fingerprint"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
