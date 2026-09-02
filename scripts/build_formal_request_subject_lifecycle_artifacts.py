"""Build G14R11 lifecycle repair evidence without formal execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_request_subject_lifecycle_rehearsal import (
    CATALOG_PATH,
    FAILURE_WINDOW,
    WINDOW_CONTRACT,
    build_trace,
    clone_frames,
    clone_rsu_state,
    clone_workflow_state,
    profile,
    workflows,
)
from src.data.mobility.replay_provider import ReplayProvider
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.evaluators.formal_window_consumption import load_contract
from src.runtime.active_formal_bundle import (
    ActiveFormalBundleError,
    validate_active_formal_bundle,
)
from src.runtime.formal_exogenous_request_execution import (
    FormalRequestExposureError,
    eligible_formal_request_subject_ids,
    request_exposure_fingerprint,
    validate_formal_request_exposure_trace,
)


ARTIFACT = ROOT / (
    "artifacts/analysis/"
    "typed_model_cache_formal_request_subject_repair_20260901_g14r11_v1"
)
PROTOCOL_ROOT = ROOT / (
    "configs/experiment/typed_model_cache_formal_protocol_v2_2_20260901"
)
V21_INDEX = ROOT / (
    "configs/experiment/typed_model_cache_formal_protocol_v2_1_20260831/"
    "protocol_index.json"
)
PROTECTED_SHA256 = {
    "scripts/train_sa_ghmappo_real_sample.py": (
        "aed850f5561f94ecba824e22bd323cdd142ee6c74255a3599129a2a6782e0eba"
    ),
    "src/agents/sa_ghmappo_agent.py": (
        "06638c1aea5097a7fa4088db6b77648648655053dc87e1a1c817b09a7709c171"
    ),
    "src/agents/sa_ghmappo_core.py": (
        "9951badce0ce78e608e690d6bed8d07a59d19dfef1e82f94a89d88403ac0d6b9"
    ),
    "src/encoders/fusion_encoder.py": (
        "cde948c13f487790cf255389bc26b7af191ecc66449a7e939b217c638327954d"
    ),
    "src/evaluators/real_eval_support.py": (
        "0a092cc15224b9b1be6a3476555c6e8eb8293573b3e27acf3fa91630db948cb6"
    ),
    "tests/test_algo_pool_contract.py": (
        "41f2ca2f6920940bc11cd16bbc4c96104452c5653812a2b69c0e1a8e6794e75b"
    ),
    "tests/test_checkpoint_compat.py": (
        "6b09b63b4a5cd9b527e7f3a146962ee37b9b1c9f8da78893d213b40bc6dc2cbf"
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8-sig"),
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant in {path}: {item}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_trace() -> tuple[dict[str, Any], Any, Any, AdapterCatalog]:
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
        phase="negative_validation",
    )
    return trace, bundle, workflow, catalog


def negative_validation() -> dict[str, Any]:
    trace, bundle, workflow, catalog = exact_trace()
    rows: list[dict[str, Any]] = []

    def expect_failure(
        case_id: str,
        operation: Callable[[], Any],
        expected_error: str,
    ) -> None:
        try:
            operation()
        except Exception as exc:  # evidence captures the exact fail-closed class/message
            rows.append(
                {
                    "case_id": case_id,
                    "status": "pass",
                    "expected_failure": True,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "expected_error_fragment_present": expected_error in str(exc),
                }
            )
            if expected_error not in str(exc):
                raise AssertionError(f"unexpected error for {case_id}: {exc}") from exc
            return
        raise AssertionError(f"negative case did not fail closed: {case_id}")

    def mutate(
        change: Callable[[dict[str, Any]], None], *, recompute: bool = True
    ) -> dict[str, Any]:
        value = deepcopy(trace)
        change(value)
        if recompute:
            value["request_exposure_fingerprint"] = request_exposure_fingerprint(value)
        return value

    cases: list[tuple[str, Callable[[dict[str, Any]], None], str, bool]] = [
        (
            "missing_lifecycle_field",
            lambda value: value["subject_lifecycle"].pop("eligible_candidate_count"),
            "lifecycle fields drift",
            True,
        ),
        (
            "extra_lifecycle_field",
            lambda value: value["subject_lifecycle"].update(unexpected=True),
            "lifecycle fields drift",
            True,
        ),
        (
            "embedded_lifecycle_version_drift",
            lambda value: value["subject_lifecycle"].update(contract_version="9.0.0"),
            "embedded contract version mismatch",
            True,
        ),
        (
            "top_level_lifecycle_version_drift",
            lambda value: value.update(
                formal_request_subject_lifecycle_contract_version="9.0.0"
            ),
            "lifecycle version mismatch",
            True,
        ),
        (
            "selected_vehicle_tamper",
            lambda value: value["subject_lifecycle"].update(
                selected_primary_vehicle_id="tampered_vehicle"
            ),
            "request vehicle differs",
            True,
        ),
        (
            "request_vehicle_tamper",
            lambda value: value["requests"][3].update(vehicle_id="tampered_vehicle"),
            "request vehicle differs",
            True,
        ),
        (
            "selection_evidence_visibility_tamper",
            lambda value: value["subject_lifecycle"].update(
                selection_evidence_actor_visible=True
            ),
            "leaks to actor",
            True,
        ),
        (
            "reselection_policy_tamper",
            lambda value: value["subject_lifecycle"].update(
                reselection_policy="dynamic"
            ),
            "reselection policy drift",
            True,
        ),
        (
            "outcome_pollution",
            lambda value: value["requests"][0].update(cache_hit=True),
            "outcome field contaminates",
            False,
        ),
        (
            "invalid_v11_run_reference",
            lambda value: value["source_provenance"].update(
                historical_source="typed_model_cache_formal_20260901_155201_g14c_v11"
            ),
            "historical invalid G14C v11",
            False,
        ),
        (
            "invalid_v11_checkpoint_root_reference",
            lambda value: value["source_provenance"].update(
                checkpoint="/private/tmp/ppo_mec_g14c_v11_e19108a_20260901_155201/checkpoints/latest.pt"
            ),
            "historical invalid G14C v11",
            False,
        ),
        (
            "nonfinite_json",
            lambda value: value["requests"][0].update(object_size_mb=math.nan),
            "non-finite JSON",
            False,
        ),
        (
            "fingerprint_tamper",
            lambda value: value.update(request_exposure_fingerprint="0" * 64),
            "canonical fingerprint mismatch",
            False,
        ),
    ]
    for case_id, change, expected_error, recompute in cases:
        expect_failure(
            case_id,
            lambda change=change, recompute=recompute: validate_formal_request_exposure_trace(
                mutate(change, recompute=recompute)
            ),
            expected_error,
        )

    def runtime_env(
        runtime_trace: dict[str, Any], runtime_frames: list[dict[str, Any]]
    ) -> VecWorkflowCoreEnv:
        return VecWorkflowCoreEnv(
            mobility_provider=ReplayProvider(clone_frames(runtime_frames)),
            workflow_state=clone_workflow_state(workflow),
            adapter_catalog=deepcopy(catalog),
            rsu_states=[clone_rsu_state(row) for row in bundle.rsu_states],
            max_steps=24,
            mobility_source="ngsim",
            primary_vehicle_selection="handoff_pressure",
            cache_capacity_profile=profile(),
            formal_request_exposure_trace=runtime_trace,
        )

    selected = trace["subject_lifecycle"]["selected_primary_vehicle_id"]
    missing_frames = clone_frames(bundle.frames)
    missing_frames[4]["vehicles"] = [
        row for row in missing_frames[4]["vehicles"] if row.vehicle_id != selected
    ]
    expect_failure(
        "runtime_subject_disappearance",
        lambda: runtime_env(trace, missing_frames).reset(),
        "selected formal request subject is not eligible",
    )
    physical_frames = clone_frames(bundle.frames)
    changed_subject = next(
        row for row in physical_frames[4]["vehicles"] if row.vehicle_id == selected
    )
    changed_subject.position_x = float(changed_subject.position_x) + 100000.0
    expect_failure(
        "runtime_physical_continuity_failure",
        lambda: runtime_env(trace, physical_frames).reset(),
        "selected formal request subject is not eligible",
    )
    rsu_drift = mutate(
        lambda value: value["requests"][3].update(
            current_service_rsu_id="tampered_rsu"
        )
    )
    expect_failure(
        "runtime_rsu_recomputation_drift",
        lambda: runtime_env(rsu_drift, clone_frames(bundle.frames)).reset(),
        "current RSU drift",
    )
    expect_failure(
        "historical_protocol_2_1_active_start",
        lambda: validate_active_formal_bundle(
            repository_root=ROOT,
            index_path=V21_INDEX,
            require_ready=False,
            require_clean_git=False,
            require_origin_main_match=False,
        ),
        "unique active protocol index",
    )
    no_persistent = [
        {
            "time_index": index,
            "source_segment_run_id": "synthetic",
            "vehicles": [
                {
                    "vehicle_id": f"v{index // 2}",
                    "position_x": float(index),
                    "position_y": 0.0,
                    "speed": 1.0,
                }
            ],
        }
        for index in range(6)
    ]
    if eligible_formal_request_subject_ids(no_persistent, request_count=5):
        raise AssertionError("no-persistent negative fixture unexpectedly has a subject")
    rows.append(
        {
            "case_id": "no_horizon_persistent_subject",
            "status": "pass",
            "expected_failure": True,
            "verdict": "BLOCKED_BY_FORMAL_REQUEST_SUBJECT_ELIGIBILITY",
            "eligible_candidate_count": 0,
        }
    )
    result = {
        "negative_validation_version": "1.0.0",
        "status": "pass" if all(row["status"] == "pass" for row in rows) else "fail",
        "case_count": len(rows),
        "cases": rows,
        "formal": False,
        "training": False,
        "performance_evidence": False,
        "holdout_opened": False,
    }
    write_json(ARTIFACT / "negative_validation.json", result)
    return result


def producer_consumer_matrix() -> dict[str, Any]:
    specs = [
        ("exposure_builder", "producer", "src/evaluators/main_results_support.py", "eligible_formal_request_subject_ids", "direct lifecycle producer"),
        ("runtime_environment", "consumer", "src/envs/core/vec_workflow_core_env.py", "selected formal request subject is not eligible", "direct independent recomputation and fail-fast"),
        ("training", "consumer", "scripts/train_algo_pool_real_sample.py", "FORMAL_REQUEST_SUBJECT_LIFECYCLE_CONTRACT_VERSION", "direct trace provenance"),
        ("dev_selection", "consumer", "scripts/run_typed_model_cache_formal_dev_selection.py", "--formal-exogenous-request-execution", "same Protocol 2.2 execution flag"),
        ("checkpoint_freeze_provenance", "consumer", "src/runtime/formal_training_identity.py", "formal_exogenous_request_execution", "lifecycle semantic hash transitively bound by request contract"),
        ("resolved_execution_context", "consumer", "src/runtime/resolved_formal_execution_context.py", "formal_exogenous_request_execution", "lifecycle semantic hash transitively bound by request contract"),
        ("formal_cache_policy", "consumer", "scripts/run_typed_model_cache_formal_cache_policy.py", "formal_request_subject_lifecycle", "direct lifecycle replay gate"),
        ("formal_controller", "consumer", "scripts/benchmark_main_results.py", "formal_request_subject_lifecycle_contract_version", "direct episode provenance"),
        ("cache_event_alignment", "consumer", "src/runtime/formal_exogenous_request_execution.py", "current_service_rsu_id=request", "direct request/runtime/event alignment companion"),
        ("endpoint_reducer", "consumer", "src/runtime/formal_exogenous_request_execution.py", "compute_formal_endpoint_metrics", "strict aligned-event denominator"),
        ("cross_agent_fairness", "consumer", "scripts/benchmark_main_results.py", "validate_observed_fingerprint_matrix", "one exposure fingerprint per evaluation unit"),
        ("request_replay", "producer_consumer", "src/oracles/cache_request_replay.py", "FORMAL_LIFECYCLE_REQUEST_REPLAY_PRODUCER_VERSION", "reuses direct lifecycle producer and validates evidence"),
        ("future_oracle", "consumer", "scripts/run_future_horizon_cache_oracle.py", "load_and_validate_request_replay", "strict replay consumer; cannot select another subject"),
        ("opportunity_analyzer", "consumer", "scripts/analyze_cache_opportunities.py", "load_and_validate_request_replay", "strict replay consumer; outcome remains separate"),
        ("statistics", "consumer", "scripts/run_typed_model_cache_formal_statistics.py", "validate_protocol_v1_1", "Protocol and resolved-context gate"),
        ("integrity_and_gate", "consumer", "scripts/run_typed_model_cache_formal_protocol.py", "validate_active_formal_bundle", "active bundle gate before run-root write"),
        ("active_bundle", "consumer", "src/runtime/active_formal_bundle.py", "formal_request_subject_lifecycle_contract", "atomic lifecycle resource identity"),
    ]
    rows = []
    for consumer_id, role, relative, token, closure in specs:
        source = (ROOT / relative).read_text(encoding="utf-8")
        present = token in source
        rows.append(
            {
                "consumer_id": consumer_id,
                "role": role,
                "path": relative,
                "required_evidence_token": token,
                "closure": closure,
                "status": "pass" if present else "fail",
            }
        )
    result = {
        "producer_consumer_matrix_version": "1.0.0",
        "status": "pass" if all(row["status"] == "pass" for row in rows) else "fail",
        "row_count": len(rows),
        "rows": rows,
        "scientific_fields_removed_or_relaxed": False,
        "formal_request_vehicle_and_rsu_fields_retained": True,
        "validator_bypass_allowed": False,
    }
    write_json(ARTIFACT / "producer_consumer_matrix.json", result)
    if result["status"] != "pass":
        raise RuntimeError("producer/consumer lifecycle closure is incomplete")
    return result


def protected_files_report() -> dict[str, Any]:
    staged_paths = set(
        subprocess.check_output(
            ["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True
        ).splitlines()
    )
    rows = []
    for relative, expected in PROTECTED_SHA256.items():
        observed = sha256_file(ROOT / relative)
        rows.append(
            {
                "path": relative,
                "starting_sha256": expected,
                "observed_sha256": observed,
                "unchanged": observed == expected,
            }
        )
    protected_staged = sorted(staged_paths.intersection(PROTECTED_SHA256))
    return {
        "status": (
            "pass"
            if all(row["unchanged"] for row in rows) and not protected_staged
            else "fail"
        ),
        "files": rows,
        "protected_staged_paths": protected_staged,
        "staged": bool(protected_staged),
    }


def preliminary() -> None:
    lifecycle = read_json(PROTOCOL_ROOT / "formal_request_subject_lifecycle_contract.json")
    write_json(ARTIFACT / "formal_request_subject_lifecycle_contract.json", lifecycle)
    negative = negative_validation()
    matrix = producer_consumer_matrix()
    protected = protected_files_report()
    review = {
        "readiness_review_version": "14.0.0",
        "status": "pending_clean_candidate_and_full_regression",
        "verdict": "NOT_READY_PENDING_G14R11_ACCEPTANCE",
        "lifecycle_contract_status": "pass",
        "negative_validation_status": negative["status"],
        "producer_consumer_matrix_status": matrix["status"],
        "protected_files_status": protected["status"],
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
        "holdout_sealed_unopened": True,
        "claim_boundary": "execution contract only; not formal, G14, TMC, or paper readiness",
    }
    write_json(ARTIFACT / "readiness_review.json", review)
    print(json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False))


def acceptance() -> None:
    root_cause = read_json(ARTIFACT / "root_cause_audit.json")
    lifecycle = read_json(ARTIFACT / "formal_request_subject_lifecycle_contract.json")
    matrix = read_json(ARTIFACT / "producer_consumer_matrix.json")
    negative = read_json(ARTIFACT / "negative_validation.json")
    exact = read_json(ARTIFACT / "exact_v11_failure_unit_rehearsal.json")
    eligibility = read_json(ARTIFACT / "exposure_eligibility_audit.json")
    parity = read_json(ARTIFACT / "cross_agent_capacity_exposure_parity.json")
    clean = read_json(ARTIFACT / "clean_candidate_validation.json")
    validation = read_json(ARTIFACT / "validation_summary.json")
    index = read_json(PROTOCOL_ROOT / "protocol_index.json")
    protected = protected_files_report()
    checks = {
        "root_cause_audit": "pass" if root_cause.get("phase_0_verdict") == "ROOT_CAUSE_CONFIRMED_IMPLEMENTATION_MAY_PROCEED" else "fail",
        "lifecycle_contract": "pass" if lifecycle.get("version") == "1.0.0" else "fail",
        "producer_consumer_matrix": matrix.get("status"),
        "negative_validation": negative.get("status"),
        "exact_failure_unit_rehearsal": exact.get("status"),
        "exposure_eligibility_audit": eligibility.get("status"),
        "cross_agent_capacity_exposure_parity": parity.get("status"),
        "clean_detached_candidate": clean.get("status"),
        "full_repository_pytest": validation.get("full_repository_pytest", {}).get("status"),
        "smoke_test": validation.get("smoke_test", {}).get("status"),
        "json_round_trip_and_inventory": validation.get("json_round_trip_and_inventory", {}).get("status"),
        "protected_files": protected.get("status"),
    }
    formal_training_count = max(
        int(exact.get("formal_training_count", 0)),
        int(clean.get("formal_training_count", 0)),
    )
    formal_checkpoint_count = max(
        int(exact.get("formal_checkpoint_count", 0)),
        int(clean.get("formal_checkpoint_count", 0)),
    )
    formal_performance_count = max(
        int(exact.get("formal_performance_count", 0)),
        int(clean.get("formal_performance_count", 0)),
    )
    holdout_ok = bool(
        eligibility.get("sealed_holdout", {}).get("sealed") is True
        and eligibility.get("sealed_holdout", {}).get("opened") is False
        and clean.get("holdout_sealed") is True
        and clean.get("holdout_opened") is False
        and clean.get("holdout_capability") is False
    )
    status = "pass" if (
        set(checks.values()) == {"pass"}
        and formal_training_count == 0
        and formal_checkpoint_count == 0
        and formal_performance_count == 0
        and holdout_ok
    ) else "fail"
    evidence = {
        "acceptance_evidence_manifest_version": "1.0.0",
        "status": status,
        "active_bundle_core_sha256": index["active_bundle_core_sha256"],
        "clean_candidate": {
            "status": clean.get("status"),
            "candidate_commit": clean.get("candidate_commit"),
            "detached_head": clean.get("detached_head"),
            "git_clean": clean.get("git_clean"),
            "local_venv_present": clean.get("local_venv_present"),
        },
        "checks": checks,
        "formal_training_count": formal_training_count,
        "formal_checkpoint_count": formal_checkpoint_count,
        "formal_performance_count": formal_performance_count,
        "holdout_sealed_unopened": holdout_ok,
        "g14c_v12_started": False,
        "g14d_started": False,
        "g15_started": False,
    }
    write_json(ARTIFACT / "acceptance_evidence_manifest.json", evidence)
    if status != "pass":
        raise RuntimeError(f"G14R11 acceptance evidence is incomplete: {checks}")
    print(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False))


def finalize() -> None:
    index = read_json(PROTOCOL_ROOT / "protocol_index.json")
    evidence = read_json(ARTIFACT / "acceptance_evidence_manifest.json")
    clean = read_json(ARTIFACT / "clean_candidate_validation.json")
    validation = read_json(ARTIFACT / "validation_summary.json")
    if index.get("status") != "READY_FOR_G14C_V12_CLEAN_TRAIN_AND_FORMAL":
        raise RuntimeError("active Protocol 2.2 index is not ready")
    bundle = validate_active_formal_bundle(
        repository_root=ROOT,
        require_ready=True,
        require_clean_git=False,
        require_origin_main_match=False,
    )
    review = {
        "readiness_review_version": "14.0.0",
        "status": "pass",
        "verdict": "READY_FOR_G14C_V12_CLEAN_TRAIN_AND_FORMAL",
        "protocol_version": "2.2.0",
        "protocol_semantic_sha256": bundle["protocol"]["hashes"]["semantic_sha256"],
        "active_bundle_core_sha256": index["active_bundle_core_sha256"],
        "active_formal_bundle_sha256": bundle["active_formal_bundle_sha256"],
        "acceptance_evidence_manifest_sha256": sha256_file(
            ARTIFACT / "acceptance_evidence_manifest.json"
        ),
        "clean_candidate_status": clean["status"],
        "validation_status": validation["status"],
        "formal_training_count": evidence["formal_training_count"],
        "formal_checkpoint_count": evidence["formal_checkpoint_count"],
        "formal_performance_count": evidence["formal_performance_count"],
        "holdout_sealed_unopened": evidence["holdout_sealed_unopened"],
        "scope_boundary": (
            "execution contract readiness only; no G14C v12 run, formal performance, "
            "holdout, G14D, G15, algorithm advantage, gate, TMC-ready, or paper-ready claim"
        ),
    }
    write_json(ARTIFACT / "readiness_review.json", review)
    inventory = []
    for path in sorted(ARTIFACT.glob("*")):
        if not path.is_file() or path.name == "artifact_integrity_manifest.json":
            continue
        read_json(path)
        inventory.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "strict_json_finite_round_trip": True,
            }
        )
    manifest = {
        "artifact_integrity_manifest_version": "1.0.0",
        "status": "pass",
        "artifact_root": ARTIFACT.relative_to(ROOT).as_posix(),
        "file_count_excluding_self": len(inventory),
        "files": inventory,
        "all_files_strict_json_finite_round_trip": True,
        "inventory_recomputed": True,
    }
    write_json(ARTIFACT / "artifact_integrity_manifest.json", manifest)
    print(json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("preliminary", "acceptance", "finalize"), required=True
    )
    args = parser.parse_args()
    if args.mode == "preliminary":
        preliminary()
    elif args.mode == "acceptance":
        acceptance()
    else:
        finalize()


if __name__ == "__main__":
    main()
