"""Freeze Protocol v1.9 with one validated active-bundle resource resolver."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.typed_model_cache_formal_protocol import attach_hashes
from src.evaluators.cache_baseline_fairness import build_manifest, validate_manifest
from src.evaluators.main_results_support import resolve_window_candidates
from src.runtime.active_formal_bundle import (
    ACTIVE_BUNDLE_RESOURCE_RESOLUTION_CONTRACT_VERSION,
    ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION,
    ACTIVE_PROTOCOL_ID,
    ACTIVE_PROTOCOL_VERSION,
    READINESS_VERSION,
    READY_STATUS,
    active_bundle_core_projection,
    build_resource_row,
    canonical_sha256,
    ready_index_projection,
    sha256_file,
)


V18 = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_8_20260827"
V19 = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_9_20260829"
ARTIFACT = ROOT / "artifacts/analysis/typed_model_cache_formal_bundle_resource_repair_20260829_g14r8_v1"
DEPENDENCY_FINGERPRINT = "88963f6107e2042298da7c6920a5d0a2d50429c92634f3873a03d0ad8f4e2d00"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(path)
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def current(logical_id: str, role: str, name: str, semantic: str | None = None) -> dict[str, Any]:
    return build_resource_row(
        root=ROOT,
        logical_id=logical_id,
        role=role,
        relative_path=(V19 / name).relative_to(ROOT).as_posix(),
        version_scope="current_protocol_version",
        semantic_sha256=semantic,
    )


def main() -> None:
    old_protocol = read_json(V18 / "protocol_v1_8_manifest.json")
    old_index = read_json(V18 / "protocol_index.json")
    protocol = deepcopy(old_protocol)
    protocol["typed_model_cache_formal_protocol_version"] = ACTIVE_PROTOCOL_VERSION
    protocol["protocol_id"] = ACTIVE_PROTOCOL_ID
    protocol["created_at"] = now()
    protocol["status"] = "frozen_pre_execution_active_bundle_resource_resolution"
    protocol["supersession"].update(
        supersedes_version="1.8.0",
        old_protocol_status="invalid_after_training_before_dev_performance_execution",
        old_protocol_semantic_sha256=old_protocol["hashes"]["semantic_sha256"],
        formal_performance_observed=False,
        scientific_fields_changed=False,
    )
    protocol["supersession"]["repair_scope"] = [
        "replace all active raw-index consumers with one validated bundle resolver",
        "pair runtime and fairness resources in frozen capacity order",
        "bind resolved resource audit to execution provenance",
    ]
    protocol["supersession"]["invalid_execution_runs"].append(
        {
            "run_id": "typed_model_cache_formal_20260828_101804_g14c_v8",
            "run_root": "artifacts/experiments/typed_model_cache_formal/typed_model_cache_formal_20260828_101804_g14c_v8",
            "status": "INVALID_PROTOCOL_OR_IMPLEMENTATION",
            "failure_boundary": "invalid_after_training_before_dev_performance_execution",
            "failure_audit_sha256": "2c09cd14028051a012ddedf756bd6b186b4d1680582c5944acc0da986aa40ba5",
            "failure_integrity_sha256": "d2a02fb61bd5b1f9964a7516441ac3ec31d95c0b4451190291be6a9bd1bf3bba",
            "inventory_canonical_sha256": "025b616efcbf9a41289f0a05a0f07bd2a8d1afaa22698ef70fc21c15d034aba5",
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
    protocol["formal_agent_order_contract"]["active_protocol_versions"] = ["1.9.0"]
    protocol["formal_agent_order_contract"]["historical_protocol_versions_audit_only"] = [
        f"1.{minor}.0" for minor in range(9)
    ]
    protocol["active_formal_bundle_contract"].update(
        version=ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION,
        unique_active_index=(V19 / "protocol_index.json").relative_to(ROOT).as_posix(),
    )
    protocol["active_formal_bundle_contract"]["hash_graph"] = [
        "resource content hashes -> active_bundle_core_sha256",
        "core plus acceptance evidence -> Readiness v11 content hash",
        "ready index plus Readiness content hash -> active_formal_bundle_sha256",
    ]
    protocol["active_bundle_resource_resolution_contract"] = {
        "version": ACTIVE_BUNDLE_RESOURCE_RESOLUTION_CONTRACT_VERSION,
        "resource_catalog": "active_bundle_resources",
        "raw_index_layout_is_consumer_api": False,
        "validated_bundle_required": True,
        "resolver": "src.runtime.active_formal_bundle.resolve_active_bundle_resource",
        "group_resolver": "src.runtime.active_formal_bundle.resolve_active_bundle_group",
        "capacity_pair_resolver": "src.runtime.active_formal_bundle.resolve_capacity_resource_pairs",
        "capacity_order": ["constrained_288mb", "medium_576mb", "relaxed_864mb"],
        "fail_fast": [
            "missing", "duplicate", "extra", "role_swap", "content_drift",
            "size_drift", "path_escape", "symlink", "version_scope_drift",
        ],
    }
    context = protocol["execution_contract"]["default_expansion_context"]
    context["active_protocol_index_path"] = (V19 / "protocol_index.json").relative_to(ROOT).as_posix()
    context["protocol_path"] = (V19 / "protocol_v1_9_manifest.json").relative_to(ROOT).as_posix()
    context["agent_scientific_config_path"] = (V19 / "agent_training_scientific_config.json").relative_to(ROOT).as_posix()
    context["formal_agent_order_contract_path"] = (V19 / "formal_agent_order_contract.json").relative_to(ROOT).as_posix()
    protocol["paper_claim_boundary"] = (
        "G14R8 repairs active-bundle resource resolution only; no formal training, "
        "checkpoint production, performance evaluation, holdout access, G14C v9, G14D, or G15."
    )

    environment = read_json(V18 / "execution_environment_manifest.json")
    identity = environment["scientific_identity"]
    identity["execution_commit"] = (
        "Commit A10; exact 40-hex clean HEAD == origin/main is observed and bound before every execution"
    )
    identity["source_root_identity"]["source_tree_sha256"] = (
        "Commit A10 Git tree; runtime active-bundle gate verifies exact clean HEAD"
    )
    identity.pop("environment_fingerprint", None)
    identity["environment_fingerprint"] = canonical_sha256(identity)
    if identity["dependency_fingerprint"] != DEPENDENCY_FINGERPRINT:
        raise ValueError("dependency fingerprint drift")
    protocol["formal_execution_environment_contract"]["scientific_identity"] = deepcopy(identity)
    protocol = attach_hashes(protocol)

    V19.mkdir(parents=True, exist_ok=True)
    write_json(V19 / "protocol_v1_9_manifest.json", protocol)
    write_json(V19 / "execution_environment_manifest.json", environment)
    for name in (
        "agent_training_scientific_config.json",
        "formal_agent_order_contract.json",
        "formal_training_execution_binding_contract.json",
        "resolved_execution_context_contract.json",
    ):
        write_json(V19 / name, read_json(V18 / name))
    resolution_contract = deepcopy(protocol["active_bundle_resource_resolution_contract"])
    resolution_contract["semantic_sha256"] = canonical_sha256(resolution_contract)
    write_json(V19 / "active_bundle_resource_resolution_contract.json", resolution_contract)

    mobility = ROOT / "data/raw/mobility/ngsim/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv"
    workflow = ROOT / "data/raw/workflow/alibaba2018/batch_task.csv"
    _, rehearsal_plan = resolve_window_candidates(
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
    rehearsal_plan = {
        "window_plan_contract": "g14r8_nonformal_rehearsal_v1",
        "window_mode": rehearsal_plan["window_mode"],
        "selected_window_plan": list(rehearsal_plan["selected_windows"]),
    }
    rehearsal_plan_path = V19 / "nonformal_rehearsal_window_plan.json"
    write_json(rehearsal_plan_path, rehearsal_plan)
    rehearsal_rows = []
    capacity_values = {
        "constrained_288mb": 288.0,
        "medium_576mb": 576.0,
        "relaxed_864mb": 864.0,
    }
    learned_order = read_json(V19 / "formal_agent_order_contract.json")[
        "learned_agent_order"
    ]
    for label, capacity in capacity_values.items():
        rehearsal_fairness = build_manifest(
            root=ROOT,
            mobility_path=mobility,
            workflow_path=workflow,
            window_plan_path=rehearsal_plan_path,
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
            output_root="artifacts/analysis/g14r8_nonformal_rehearsal",
            evaluation_unit_limit=1,
            created_at="2026-08-29T00:00:00+08:00",
            controller_agents=learned_order,
        )
        report = validate_manifest(rehearsal_fairness, root=ROOT, check_files=True)
        if report["status"] != "pass":
            raise ValueError(report["errors"])
        name = f"nonformal_rehearsal_fairness_{label}.json"
        write_json(V19 / name, rehearsal_fairness)
        rehearsal_rows.append(
            current(
                f"rehearsal_fairness_manifests.{label}",
                "nonformal rehearsal fairness manifest",
                name,
            )
        )

    current_rows = [
        current("protocol_manifest", "active Protocol manifest", "protocol_v1_9_manifest.json", protocol["hashes"]["semantic_sha256"]),
        current("execution_environment_manifest", "active execution environment identity", "execution_environment_manifest.json"),
        current("agent_training_scientific_config", "Scientific Config 2.0.0", "agent_training_scientific_config.json", "f83587cd13c126a0d8a6bdc26402e34ac1391bd6fc8ef504736458872d649bc8"),
        current("formal_agent_order_contract", "Formal Agent Order Contract 1.0.0", "formal_agent_order_contract.json", "82e562755dadd4341c950bf71efc488d3527b7f45b7f02512f8064d189b655e0"),
        current("formal_training_execution_binding_schema", "formal execution binding schema 1.0.0", "formal_training_execution_binding_contract.json"),
        current("resolved_execution_context_schema", "resolved context schema 2.0.0", "resolved_execution_context_contract.json"),
        current("active_bundle_resource_resolution_contract", "Active Bundle Resource Resolution Contract 1.0.0", "active_bundle_resource_resolution_contract.json", resolution_contract["semantic_sha256"]),
    ]
    shared_rows = [
        deepcopy(row)
        for row in old_index["active_bundle_resources"]
        if row.get("version_scope") == "shared_historical_stable"
    ]
    index: dict[str, Any] = {
        "active_formal_bundle_contract_version": ACTIVE_FORMAL_BUNDLE_CONTRACT_VERSION,
        "active_bundle_resource_resolution_contract_version": ACTIVE_BUNDLE_RESOURCE_RESOLUTION_CONTRACT_VERSION,
        "protocol_index_version": ACTIVE_PROTOCOL_VERSION,
        "status": READY_STATUS,
        "protocol_identity": {
            "protocol_id": ACTIVE_PROTOCOL_ID,
            "protocol_version": ACTIVE_PROTOCOL_VERSION,
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
            "environment_fingerprint": identity["environment_fingerprint"],
            "dependency_fingerprint": DEPENDENCY_FINGERPRINT,
        },
        "command_matrix_identity": {
            "command_templates_sha256": canonical_sha256(protocol["execution_contract"]["command_templates"]),
            "outer_nested_expansion_equality_required": True,
        },
        "holdout_seal": deepcopy(protocol["holdout_execution_contract"]),
        "active_bundle_resources": [*current_rows, *rehearsal_rows, *shared_rows],
    }
    index["active_bundle_core_sha256"] = canonical_sha256(active_bundle_core_projection(index))
    evidence = {
        "status": "pass",
        "clean_candidate": True,
        "active_bundle_core_sha256": index["active_bundle_core_sha256"],
        "resource_resolution_contract_version": ACTIVE_BUNDLE_RESOURCE_RESOLUTION_CONTRACT_VERSION,
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
        "holdout_opened": False,
    }
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
    write_json(V19 / "readiness_v11.json", readiness)
    readiness_row = current("readiness_companion", "Readiness v11 evidence companion", "readiness_v11.json")
    index["active_bundle_resources"].append(readiness_row)
    index["readiness_companion"] = {
        "logical_path": readiness_row["logical_path"],
        "content_sha256": readiness_row["content_sha256"],
    }
    index["active_formal_bundle_sha256"] = canonical_sha256(ready_index_projection(index))
    write_json(V19 / "protocol_index.json", index)
    print(json.dumps({
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "protocol_full_sha256": protocol["hashes"]["full_sha256"],
        "active_bundle_core_sha256": index["active_bundle_core_sha256"],
        "active_formal_bundle_sha256": index["active_formal_bundle_sha256"],
        "environment_fingerprint": identity["environment_fingerprint"],
    }, indent=2))


if __name__ == "__main__":
    main()
