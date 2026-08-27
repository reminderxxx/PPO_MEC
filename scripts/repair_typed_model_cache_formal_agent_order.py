"""Freeze G14R7 Protocol v1.7 and the unified formal agent-order contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.typed_model_cache_formal_protocol import attach_hashes, sha256_file
from src.runtime.formal_agent_order import (
    FORMAL_AGENT_ORDER_CONTRACT_VERSION,
    canonical_sha256,
    contract_projection,
    resolve_formal_agent_order,
)
from src.runtime.formal_execution_environment import (
    probe_python_environment,
    scientific_environment_identity,
)


RUN_ID = "typed_model_cache_formal_agent_order_repair_20260827_g14r7_v1"
ARTIFACT_ROOT = ROOT / "artifacts/analysis" / RUN_ID
V16_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_6_20260825"
V17_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_7_20260827"
V16_PROTOCOL = V16_ROOT / "protocol_v1_6_manifest.json"
V17_PROTOCOL = V17_ROOT / "protocol_v1_7_manifest.json"
ORDER_CONTRACT_PATH = V17_ROOT / "formal_agent_order_contract.json"
SCIENTIFIC_CONFIG = V17_ROOT / "agent_training_scientific_config.json"
V7_RUN_ID = "typed_model_cache_formal_20260826_233222_g14c_v7"
V7_ROOT = ROOT / "artifacts/experiments/typed_model_cache_formal" / V7_RUN_ID
V7_FAILURE_AUDIT_SHA256 = "7fc3685470c1f536def5c504dfbeab83b14dd070a644caefed08e690e10247ba"
V7_FAILURE_INTEGRITY_SHA256 = "ab38f022aa14f51079d74799d73bf88a2382e9809f04a3ba0b22285826e466a2"
SCIENTIFIC_CONFIG_SHA256 = "f83587cd13c126a0d8a6bdc26402e34ac1391bd6fc8ef504736458872d649bc8"
DEPENDENCY_FINGERPRINT = "88963f6107e2042298da7c6920a5d0a2d50429c92634f3873a03d0ad8f4e2d00"
REACTIVE_ORDER = [
    "reactive_lru",
    "reactive_fifo",
    "reactive_lfu",
    "reactive_aging_lfu",
    "reactive_random",
]
LEARNED_ORDER = [
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
]
PROTECTED_FILES = (
    "scripts/train_sa_ghmappo_real_sample.py",
    "src/agents/sa_ghmappo_agent.py",
    "src/agents/sa_ghmappo_core.py",
    "src/encoders/fusion_encoder.py",
    "src/evaluators/real_eval_support.py",
    "tests/test_algo_pool_contract.py",
    "tests/test_checkpoint_compat.py",
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def build_order_contract() -> dict[str, Any]:
    main = [*REACTIVE_ORDER, *LEARNED_ORDER]
    payload: dict[str, Any] = {
        "formal_agent_order_contract_version": FORMAL_AGENT_ORDER_CONTRACT_VERSION,
        "identity_rules": {
            "json_object_insertion_order_is_identity_forbidden": True,
            "alphabetical_order_inference_forbidden": True,
            "same_set_different_order_rejected": True,
            "single_resolver_required": True,
        },
        "reactive_agent_order": REACTIVE_ORDER,
        "learned_agent_order": LEARNED_ORDER,
        "main_benchmark_agent_order": main,
        "checkpoint_free_report_only_agent_roles": [
            {"agent": "popularity_cache_heuristic", "role": "matched_report_only_heuristic"},
            {"agent": "exact_oracle_h1", "role": "exact_oracle_cell"},
            {"agent": "exact_oracle_h3", "role": "exact_oracle_cell"},
            {"agent": "exact_oracle_h6", "role": "exact_oracle_cell"},
            {"agent": "exact_oracle_h12", "role": "exact_oracle_cell"},
        ],
        "row_display_order": main,
        "pairwise_statistics_identity": {
            "candidate_agent": "sa_ghmappo",
            "baseline_agent_order": [name for name in main if name != "sa_ghmappo"],
            "comparison_key_order": ["candidate_agent", "baseline_agent", "metric"],
        },
        "permanently_rejected_run_ids": [
            "typed_model_cache_formal_20260820_g14c_351fdb8_v1",
            "typed_model_cache_formal_20260820_164251_g14c_v2",
            "typed_model_cache_formal_20260820_203430_g14c_v3",
            "typed_model_cache_formal_20260824_110016_g14c_v4",
            "typed_model_cache_formal_20260824_235839_g14c_v4",
            "typed_model_cache_formal_20260825_111625_g14c_v5",
            "typed_model_cache_formal_20260825_135122_g14c_v6",
            V7_RUN_ID,
        ],
    }
    payload["semantic_sha256"] = canonical_sha256(contract_projection(payload))
    return payload


def invalid_v7_reference() -> dict[str, Any]:
    if sha256_file(V7_ROOT / "failure_audit.json") != V7_FAILURE_AUDIT_SHA256:
        raise ValueError("G14C v7 failure audit hash drift")
    if sha256_file(V7_ROOT / "failure_integrity.json") != V7_FAILURE_INTEGRITY_SHA256:
        raise ValueError("G14C v7 failure integrity hash drift")
    return {
        "status": "INVALID_PROTOCOL_OR_IMPLEMENTATION",
        "run_id": V7_RUN_ID,
        "execution_commit": "bfc540c798bae584c9cbca5f382aa9745434a605",
        "failure_boundary": "invalid_after_training_before_dev_performance_execution",
        "failure_audit_path": (V7_ROOT / "failure_audit.json").relative_to(ROOT).as_posix(),
        "failure_audit_sha256": V7_FAILURE_AUDIT_SHA256,
        "failure_integrity_path": (V7_ROOT / "failure_integrity.json").relative_to(ROOT).as_posix(),
        "failure_integrity_sha256": V7_FAILURE_INTEGRITY_SHA256,
        "training_cells_executed": 150,
        "candidate_checkpoint_count": 1200,
        "dev_input_manifest_count": 2,
        "dev_performance_count": 0,
        "selected_checkpoint_count": 0,
        "frozen_checkpoint_count": 0,
        "formal_performance_count": 0,
        "resume_allowed": False,
        "retry_allowed": False,
        "legacy_phase_finalize_allowed": False,
        "checkpoint_reuse_allowed": False,
        "checkpoint_salvage_allowed": False,
        "candidate_reuse_allowed": False,
        "partial_dev_input_reuse_allowed": False,
        "ledger_or_marker_reuse_allowed": False,
        "immutable_old_run": True,
    }


def _add_order_contract_flag(argv: list[str]) -> None:
    if "--formal-agent-order-contract-path" not in argv:
        argv.extend(
            ["--formal-agent-order-contract-path", "{formal_agent_order_contract_path}"]
        )


def build_protocol(shared_python: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    old = read_json(V16_PROTOCOL)
    scientific = read_json(V16_ROOT / "agent_training_scientific_config.json")
    if scientific.get("config_semantic_sha256") != SCIENTIFIC_CONFIG_SHA256:
        raise ValueError("scientific config semantic identity drift")
    order_contract = build_order_contract()
    protocol = deepcopy(old)
    probe = probe_python_environment(shared_python, clean_worktree_root=ROOT)
    environment_identity = scientific_environment_identity(
        probe,
        execution_commit="Commit A8 (runtime binding requires exact observed clean Git HEAD)",
        source_tree_sha256="Commit A8 Git tree (runtime-audited; no self-reference)",
    )
    if environment_identity["dependency_fingerprint"] != DEPENDENCY_FINGERPRINT:
        raise ValueError("dependency fingerprint changed")
    protocol.update(
        typed_model_cache_formal_protocol_version="1.7.0",
        protocol_id="typed_model_cache_formal_protocol_v1_7",
        created_at=now(),
        status="frozen_pre_execution_agent_order_repair_no_performance",
    )
    failures = deepcopy(old["supersession"]["invalid_execution_runs"])
    failures.append(invalid_v7_reference())
    protocol["supersession"] = {
        "supersedes_version": "1.6.0",
        "old_protocol_status": "invalid_after_training_before_dev_performance_execution",
        "old_protocol_semantic_sha256": old["hashes"]["semantic_sha256"],
        "invalid_execution_runs": failures,
        "formal_performance_observed": False,
        "scientific_fields_changed": False,
        "repair_scope": [
            "freeze one versioned formal agent order authority",
            "replace mapping insertion order in dev selection",
            "bind display, checkpoint, fairness, command, and pairwise statistics order",
            "permanently reject every G14C v7 artifact and checkpoint reference",
        ],
    }
    protocol["formal_execution_environment_contract"]["scientific_identity"] = environment_identity
    protocol["identity"]["execution_git_commit_binding"] = (
        "Commit A8; exact observed clean Git HEAD is bound by runtime execution binding"
    )
    protocol["identity"]["formal_agent_order_contract_semantic_sha256"] = order_contract[
        "semantic_sha256"
    ]
    protocol["formal_agent_order_contract"] = {
        "version": FORMAL_AGENT_ORDER_CONTRACT_VERSION,
        "semantic_sha256": order_contract["semantic_sha256"],
        "unique_resolver": "src.runtime.formal_agent_order.resolve_formal_agent_order",
        "json_mapping_order_is_scientific_identity": False,
        "same_set_different_order_rejected": True,
        "active_protocol_versions": ["1.7.0"],
        "historical_protocol_versions_audit_only": [
            "1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0"
        ],
    }
    protocol["training_budget"]["learned_agent_order"] = list(LEARNED_ORDER)
    protocol["statistics"].update(
        candidate_agent="sa_ghmappo",
        baseline_agent_order=[
            name for name in [*REACTIVE_ORDER, *LEARNED_ORDER] if name != "sa_ghmappo"
        ],
        formal_agent_order_contract_semantic_sha256=order_contract["semantic_sha256"],
    )
    protocol["claim_evidence_map"].update(
        paper_display_agent_order=[*REACTIVE_ORDER, *LEARNED_ORDER],
        formal_agent_order_contract_semantic_sha256=order_contract["semantic_sha256"],
    )
    protocol["formal_training_execution_binding_contract"][
        "agent_order_contract_hash_enters_binding_context_and_commands"
    ] = True
    bindings = protocol["execution_contract"]["same_run_resume"]["bindings"]
    if "formal_agent_order_contract_semantic_sha256" not in bindings:
        bindings.append("formal_agent_order_contract_semantic_sha256")
    context = protocol["execution_contract"]["default_expansion_context"]
    context["agent_scientific_config_path"] = SCIENTIFIC_CONFIG.relative_to(ROOT).as_posix()
    context["formal_agent_order_contract_path"] = ORDER_CONTRACT_PATH.relative_to(ROOT).as_posix()
    context["protocol_path"] = V17_PROTOCOL.relative_to(ROOT).as_posix()
    for phase, spec in protocol["execution_contract"]["command_templates"].items():
        argv = spec.get("argv", [])
        if phase in {"dev_select", "formal_statistics"} or "--agents" in argv:
            _add_order_contract_flag(argv)
    protocol["paper_claim_boundary"] = (
        "G14R7 freezes execution/display/statistics order only; it is non-formal, contains no "
        "performance conclusion, does not open holdout, and does not execute G14C v8/G14D/G15."
    )
    protocol = attach_hashes(protocol)
    resolve_formal_agent_order(
        contract=order_contract,
        protocol=protocol,
        scientific_config=scientific,
        reactive_baseline_order=REACTIVE_ORDER,
        command_templates=protocol["execution_contract"]["command_templates"],
    )
    return protocol, scientific, order_contract | {"_environment_probe": probe}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-executable", default=sys.executable)
    args = parser.parse_args()
    shared_python = Path(args.python_executable).absolute()
    if not shared_python.is_file() or not os.access(shared_python, os.X_OK):
        raise FileNotFoundError(shared_python)
    protocol, scientific, contract_bundle = build_protocol(shared_python)
    probe = contract_bundle.pop("_environment_probe")
    order_contract = contract_bundle
    write_json(ORDER_CONTRACT_PATH, order_contract)
    write_json(SCIENTIFIC_CONFIG, scientific)
    write_json(V17_PROTOCOL, protocol)
    environment_manifest = {
        "formal_execution_environment_contract_version": "1.0.0",
        "scientific_identity": protocol["formal_execution_environment_contract"]["scientific_identity"],
        "runtime_location": {
            "resolved_python_absolute_path": str(shared_python),
            "virtual_environment_root": probe["sys_prefix"],
            "site_packages_paths": probe["site_packages"],
        },
        "runtime_location_is_scientific_identity": False,
    }
    write_json(V17_ROOT / "execution_environment_manifest.json", environment_manifest)
    write_json(
        V17_ROOT / "formal_training_execution_binding_contract.json",
        protocol["formal_training_execution_binding_contract"],
    )
    old_index = read_json(V16_ROOT / "protocol_index.json")
    index = deepcopy(old_index)
    index.update(
        protocol_index_version="1.7.0",
        protocol_manifest=V17_PROTOCOL.relative_to(ROOT).as_posix(),
        protocol_semantic_sha256=protocol["hashes"]["semantic_sha256"],
        agent_scientific_config=SCIENTIFIC_CONFIG.relative_to(ROOT).as_posix(),
        agent_scientific_config_semantic_sha256=SCIENTIFIC_CONFIG_SHA256,
        formal_agent_order_contract=ORDER_CONTRACT_PATH.relative_to(ROOT).as_posix(),
        formal_agent_order_contract_version=FORMAL_AGENT_ORDER_CONTRACT_VERSION,
        formal_agent_order_contract_semantic_sha256=order_contract["semantic_sha256"],
        status="PENDING_G14R7_VALIDATION",
    )
    write_json(V17_ROOT / "protocol_index.json", index)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ARTIFACT_ROOT / "g14c_v7_invalid_registration.json", invalid_v7_reference())
    write_json(ARTIFACT_ROOT / "formal_agent_order_contract.json", order_contract)
    write_json(
        ARTIFACT_ROOT / "root_cause_audit.json",
        {
            "status": "confirmed",
            "failure_audit_sha256": V7_FAILURE_AUDIT_SHA256,
            "failure_integrity_sha256": V7_FAILURE_INTEGRITY_SHA256,
            "root_cause": (
                "dev selector used list(protocol['training_budget']['agent_configs']); sorted-key "
                "JSON mapping insertion order became execution identity"
            ),
            "correct_authority": "formal_agent_order_contract.json via resolve_formal_agent_order",
            "failure_boundary": "invalid_after_training_before_dev_performance_execution",
            "training_cells_candidates_dev_selected_frozen_formal": [150, 1200, 0, 0, 0, 0],
            "performance_observed": False,
        },
    )
    write_json(
        ARTIFACT_ROOT / "protected_user_file_hashes_start.json",
        {"capture": "task_start", "files": {name: sha256_file(ROOT / name) for name in PROTECTED_FILES}},
    )
    write_json(ARTIFACT_ROOT / "protocol_v1_7_manifest.json", protocol)
    write_json(
        ARTIFACT_ROOT / "order_reconciliation.json",
        resolve_formal_agent_order(
            contract=order_contract,
            protocol=protocol,
            scientific_config=scientific,
            reactive_baseline_order=REACTIVE_ORDER,
        ),
    )
    print(
        json.dumps(
            {
                "status": "generated_pending_validation",
                "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
                "order_contract_semantic_sha256": order_contract["semantic_sha256"],
                "scientific_config_semantic_sha256": scientific["config_semantic_sha256"],
                "environment_fingerprint": protocol["formal_execution_environment_contract"]["scientific_identity"]["environment_fingerprint"],
                "dependency_fingerprint": protocol["formal_execution_environment_contract"]["scientific_identity"]["dependency_fingerprint"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
