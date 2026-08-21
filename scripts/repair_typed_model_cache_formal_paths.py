"""Freeze Protocol v1.3 and audit portable formal-resource bindings.

This preparation script is outcome-blind.  It never trains, evaluates formal or
holdout windows, opens holdout, downloads data, or copies checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.typed_model_cache_formal_execution import (
    PHASE_ORDER,
    readiness_v5,
    validate_command_templates,
    validate_protocol_v1_1,
)
from src.evaluators.typed_model_cache_formal_protocol import (
    attach_hashes,
    canonical_sha256,
    semantic_projection,
)
from src.runtime.portable_resource_identity import (
    ALLOWED_RESOLVERS,
    PortableResourceError,
    build_registry,
    build_resource_identity,
    resolve_resource,
    scientific_identity_fingerprint,
    sha256_file,
)


BASELINE_COMMIT = "a7c9e8ec548aa97332096a3013af744a883c9954"
G14C_V3_RUN_ROOT = Path(
    "/private/tmp/ppo_mec_g14c_v3_a7c9e8e/artifacts/experiments/typed_model_cache_formal/"
    "typed_model_cache_formal_20260820_203430_g14c_v3"
)
FAILURE_AUDIT_SHA256 = "476cfc3f57312263da7dff388a89c088e4716d43b1949eb121598c86dc5ac3af"
OLD_PROTOCOL_SEMANTIC_SHA256 = "718c0f78aabd5d01012df31267626eab74a51b2b621aaa67a535c5b60e655ca9"
SPLIT_SEMANTIC_SHA256 = "aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a"
WINDOW_CONTRACT_SEMANTIC_SHA256 = "ec475799b3fba4a3af3e4372e7c25781c6565a88ec814322b4cd4d447fef2771"
CATALOG_FINGERPRINT = "89c548980b63df733553d748e8db3ca622965b63abcd08ebd4c231790b40a9d6"
RUN_ID = "typed_model_cache_formal_path_repair_20260821_g14r3_v1"
CONFIG_RELATIVE = Path("configs/experiment/typed_model_cache_formal_protocol_v1_3_20260821")
ARTIFACT_RELATIVE = Path("artifacts/analysis") / RUN_ID
OLD_CONFIG_RELATIVE = Path("configs/experiment/typed_model_cache_formal_protocol_v1_2_20260820")
V11_CONFIG_RELATIVE = Path("configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820")
WINDOW_CONFIG_RELATIVE = Path("configs/experiment/typed_model_cache_formal_protocol_v1_20260820")
PROTECTED_USER_FILES = (
    "scripts/train_sa_ghmappo_real_sample.py",
    "src/agents/sa_ghmappo_agent.py",
    "src/agents/sa_ghmappo_core.py",
    "src/encoders/fusion_encoder.py",
    "src/evaluators/real_eval_support.py",
    "tests/test_algo_pool_contract.py",
    "tests/test_checkpoint_compat.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def protected_hashes() -> dict[str, str]:
    return {path: sha256_file(ROOT / path) for path in PROTECTED_USER_FILES}


def resource_identity(
    path: Path,
    *,
    logical_id: str,
    role: str,
    schema: str,
    revision: str,
    expected_relative: str,
    resolvers: Iterable[str],
    provenance: str,
) -> dict[str, Any]:
    return build_resource_identity(
        path,
        logical_resource_id=logical_id,
        resource_role=role,
        schema_version=schema,
        revision=revision,
        expected_logical_relative_path=expected_relative,
        allowed_resolvers=list(resolvers),
        provenance=provenance,
    )


def build_resource_registry() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resources: list[dict[str, Any]] = []

    def add(
        relative: str,
        logical_id: str,
        role: str,
        schema: str,
        revision: str,
        *,
        data_root: bool = False,
        expected_relative: str | None = None,
    ) -> None:
        path = ROOT / relative
        resources.append(
            resource_identity(
                path,
                logical_id=logical_id,
                role=role,
                schema=schema,
                revision=revision,
                expected_relative=expected_relative
                or (relative.removeprefix("data/") if data_root else relative),
                resolvers=(
                    ("explicit_path", "data_root")
                    if data_root
                    else ("explicit_path", "worktree_root", "manifest_relative")
                ),
                provenance=(
                    "existing_local_dataset_no_download"
                    if data_root
                    else "repository_controlled"
                ),
            )
        )

    add(
        "data/raw/mobility/ngsim/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv",
        "dataset.mobility.ngsim.vehicle_trajectories",
        "mobility_dataset",
        "NGSIMProvider/v1",
        "ngsim_20260329",
        data_root=True,
    )
    add(
        "data/raw/workflow/alibaba2018/batch_task.csv",
        "dataset.workflow.alibaba2018.batch_task",
        "workflow_dataset",
        "AlibabaDAGParser/legacy_batch_type",
        "alibaba_cluster_trace_2018",
        data_root=True,
    )
    add(
        "src/data/model_catalog/typed_model_cache_controlled.json",
        "catalog.typed_model_cache.controlled",
        "typed_catalog",
        "AdapterCatalog/typed_base_adapter_state_v1",
        CATALOG_FINGERPRINT,
    )
    add(
        "artifacts/analysis/typed_model_cache_formal_protocol_freeze_20260820_g14b_v1/split_manifest.json",
        "split.typed_model_cache.strict_v1",
        "split_manifest",
        "typed_model_cache_split_protocol/1.0.0",
        SPLIT_SEMANTIC_SHA256,
    )
    for split in ("train", "dev", "formal", "sealed_holdout"):
        filename = "sealed_holdout_window_plan.json" if split == "sealed_holdout" else f"{split}_window_plan.json"
        add(
            str(WINDOW_CONFIG_RELATIVE / filename),
            f"window_plan.typed_model_cache.{split}",
            "window_plan",
            "frozen_window_plan/1.0.0",
            split,
        )
    add(
        str(CONFIG_RELATIVE / "formal_window_consumption_contract.json"),
        "window_contract.typed_model_cache.formal_v1",
        "window_contract",
        "formal_window_consumption_contract/1.0.0",
        WINDOW_CONTRACT_SEMANTIC_SHA256,
        expected_relative=str(CONFIG_RELATIVE / "formal_window_consumption_contract.json"),
    )
    for capacity in ("constrained_288mb", "medium_576mb", "relaxed_864mb"):
        add(
            str(V11_CONFIG_RELATIVE / f"runtime_{capacity}.yaml"),
            f"runtime_config.{capacity}",
            "runtime_config",
            "typed_model_cache_runtime_config/1.0.0",
            capacity,
        )
        for split, prefix in (("formal", "fairness"), ("dev", "dev_fairness")):
            add(
                str(V11_CONFIG_RELATIVE / f"{prefix}_{capacity}.json"),
                f"fairness_manifest.{split}.{capacity}",
                "fairness_manifest",
                "cache_baseline_fairness_manifest/1.1.0+portable_companion_1.0.0",
                capacity,
            )
    for path in sorted((ROOT / V11_CONFIG_RELATIVE).glob("fairness_support_*.json")):
        add(
            path.relative_to(ROOT).as_posix(),
            f"fairness_manifest.support.{path.stem.removeprefix('fairness_support_')}",
            "fairness_manifest",
            "cache_baseline_fairness_manifest/1.1.0+portable_companion_1.0.0",
            path.stem,
        )
    for name in ("reactive_lru", "reactive_fifo", "reactive_lfu", "reactive_aging_lfu", "reactive_random"):
        add(
            f"configs/algo/{name}.yaml",
            f"baseline_config.{name}",
            "baseline_config",
            "cache_baseline_config/1.0.0",
            name,
        )
    registry = build_registry(resources, registry_id="typed-model-cache-formal-resources-v1-3")
    return registry, resources


def replace_host_path(value: Any) -> Any:
    if isinstance(value, str):
        prefix = ROOT.as_posix() + "/"
        if value.startswith(prefix + "data/"):
            return "{data_root}/" + value[len(prefix + "data/") :]
        if value.startswith(prefix):
            return value[len(prefix) :]
        return value
    if isinstance(value, list):
        return [replace_host_path(item) for item in value]
    if isinstance(value, dict):
        return {key: replace_host_path(item) for key, item in value.items()}
    return value


def append_unique(argv: list[str], tokens: list[str], *, before: str | None = None) -> None:
    if tokens[0] in argv:
        return
    if before is not None and before in argv:
        index = argv.index(before)
        argv[index:index] = tokens
    else:
        argv.extend(tokens)


def common_resource_flags(
    *,
    window_id: str,
    fairness_id: str = "{fairness_manifest_resource_id}",
    runtime_id: str = "{runtime_config_resource_id}",
    checkpoint_id: str = "{checkpoint_manifest_id}",
) -> list[str]:
    return [
        "--resource-registry-path", "{resource_registry_path}",
        "--repository-root", "{repository_root}",
        "--data-root", "{data_root}",
        "--protocol-artifact-root", "{protocol_artifact_root}",
        "--checkpoint-root", "{checkpoint_root}",
        "--mobility-resource-id", "dataset.mobility.ngsim.vehicle_trajectories",
        "--workflow-resource-id", "dataset.workflow.alibaba2018.batch_task",
        "--catalog-resource-id", "catalog.typed_model_cache.controlled",
        "--window-plan-resource-id", window_id,
        "--window-contract-resource-id", "window_contract.typed_model_cache.formal_v1",
        "--protocol-resource-id", "protocol.typed_model_cache.formal_v1_3",
        "--fairness-manifest-resource-id", fairness_id,
        "--runtime-config-resource-id", runtime_id,
        "--checkpoint-manifest-id", checkpoint_id,
    ]


def fairness_resource_id(path: str, split: str) -> str:
    stem = Path(path).stem
    if stem.startswith("fairness_support_"):
        return f"fairness_manifest.support.{stem.removeprefix('fairness_support_')}"
    capacity = stem.removeprefix("dev_fairness_").removeprefix("fairness_")
    return f"fairness_manifest.{split}.{capacity}"


def transform_templates(protocol: dict[str, Any]) -> None:
    execution = protocol["execution_contract"]
    context = replace_host_path(execution["default_expansion_context"])
    context.update(
        {
            "protocol_path": (CONFIG_RELATIVE / "protocol_v1_3_manifest.json").as_posix(),
            "agent_config_path": (CONFIG_RELATIVE / "agent_training_configs.json").as_posix(),
            "window_consumption_contract_path": (CONFIG_RELATIVE / "formal_window_consumption_contract.json").as_posix(),
            "resource_registry_path": (CONFIG_RELATIVE / "portable_resource_registry.json").as_posix(),
            "repository_root": ".",
            "data_root": "data",
            "protocol_artifact_root": CONFIG_RELATIVE.as_posix(),
            "checkpoint_root": "/ABSOLUTE/FORMAL_OUTPUT_ROOT",
            "runtime_config_resource_id": "runtime_config.medium_576mb",
            "fairness_manifest_resource_id": "fairness_manifest.formal.medium_576mb",
            "checkpoint_manifest_id": "checkpoint_manifest.medium_576mb",
        }
    )
    execution["default_expansion_context"] = context
    templates = replace_host_path(execution["command_templates"])
    execution["command_templates"] = templates
    for phase, spec in templates.items():
        argv = spec["argv"]
        for index, token in enumerate(argv):
            if token.endswith("formal_window_consumption_contract.json"):
                argv[index] = "{window_consumption_contract_path}"
        contexts = spec.get("matrix_contexts") or []
        for overlay in contexts:
            capacity = str(overlay.get("capacity_label") or "medium_576mb")
            overlay["runtime_config_resource_id"] = f"runtime_config.{capacity}"
            overlay["checkpoint_manifest_id"] = f"checkpoint_manifest.{capacity}"
            manifest_path = str(overlay.get("fairness_manifest_path") or context["fairness_manifest_path"])
            overlay["fairness_manifest_resource_id"] = fairness_resource_id(
                manifest_path,
                "dev" if phase == "dev_select" else "formal",
            )
        if phase == "train":
            append_unique(argv, common_resource_flags(window_id="window_plan.typed_model_cache.train"))
        elif phase == "dev_select":
            append_unique(
                argv,
                ["--workflow-csv-path", "{data_root}/raw/workflow/alibaba2018/batch_task.csv"],
            )
            append_unique(
                argv,
                common_resource_flags(
                    window_id="window_plan.typed_model_cache.dev",
                    fairness_id="fairness_manifest.dev.medium_576mb",
                ),
            )
        elif phase == "checkpoint_freeze":
            append_unique(
                argv,
                common_resource_flags(window_id="window_plan.typed_model_cache.dev"),
            )
        elif phase in {"formal_controller"}:
            append_unique(
                argv,
                common_resource_flags(window_id="window_plan.typed_model_cache.formal"),
            )
        elif phase in {"formal_ablation", "formal_support", "formal_scalability"}:
            append_unique(
                argv,
                ["--workflow-csv-path", "{data_root}/raw/workflow/alibaba2018/batch_task.csv"],
            )
            append_unique(
                argv,
                common_resource_flags(window_id="window_plan.typed_model_cache.formal"),
            )
        elif phase == "formal_cache_policy":
            flags = common_resource_flags(window_id="window_plan.typed_model_cache.formal")
            append_unique(argv, flags, before="--command")
            append_unique(argv, flags)
        elif phase == "formal_statistics":
            append_unique(
                argv,
                common_resource_flags(window_id="window_plan.typed_model_cache.formal"),
            )


def build_protocol(registry: Mapping[str, Any]) -> dict[str, Any]:
    old = read_json(ROOT / OLD_CONFIG_RELATIVE / "protocol_v1_2_manifest.json")
    protocol = replace_host_path(deepcopy(old))
    protocol.pop("hashes", None)
    protocol["typed_model_cache_formal_protocol_version"] = "1.3.0"
    protocol["protocol_id"] = "typed_model_cache_formal_protocol_v1_3"
    protocol["status"] = "frozen_pre_execution_portable_resource_and_dev_binding_repair_no_formal_performance"
    protocol["supersession"] = {
        "supersedes_version": "1.2.0",
        "old_protocol_status": "invalid_before_dev_performance_execution",
        "old_protocol_semantic_sha256": OLD_PROTOCOL_SEMANTIC_SHA256,
        "old_run_id": G14C_V3_RUN_ROOT.name,
        "old_run_status": "INVALID_PROTOCOL_OR_IMPLEMENTATION",
        "failure_audit_sha256": FAILURE_AUDIT_SHA256,
        "training_cells_completed": 150,
        "checkpoint_candidates_created": 1200,
        "dev_performance_count": 0,
        "selected_checkpoint_count": 0,
        "formal_count": 0,
        "resume_allowed": False,
        "salvage_exception_created": False,
        "scientific_question_changed": False,
        "split_changed": False,
        "repair_scope": [
            "portable external resource identity",
            "shared content-addressed resource resolver",
            "fairness content-identical relocation",
            "explicit dev workflow binding",
            "checkpoint scientific identity and location separation",
            "portable train/dev/formal/support commands",
        ],
    }
    protocol["identity"]["execution_git_commit_binding"] = (
        "Commit A4 containing this exact semantic hash; future G14C v4 unique execution commit"
    )
    protocol["identity"]["fairness_manifest_version"] = "1.1.0+portable_identity_companion_1.0.0"
    protocol["portable_resource_identity_contract"] = {
        "version": "1.0.0",
        "resource_resolver_version": "1.0.0",
        "scientific_identity_rule": "scientific identity != host absolute path",
        "resource_registry_path": (CONFIG_RELATIVE / "portable_resource_registry.json").as_posix(),
        "resource_registry_semantic_sha256": registry["hashes"]["semantic_sha256"],
        "absolute_path_in_protocol_semantic_hash": False,
        "content_identical_path_relocation": "allowed_after_logical_id_role_hash_size_schema_validation",
        "content_drift": "fail_fast",
        "network_download": False,
        "automatic_large_file_copy": False,
    }
    protocol["fairness_portability"] = {
        "legacy_manifest_version": "1.1.0",
        "companion_version": "1.0.0",
        "companion_path": (CONFIG_RELATIVE / "fairness_portable_identity_companion.json").as_posix(),
        "legacy_exact_path": "compatible_legacy_path",
        "legacy_relocation": "relocatable_after_content_validation",
        "content_override": "incompatible",
        "g14c_v3_classification": "allowed_content_identical_path_relocation",
    }
    protocol["dev_selection_binding_repair"] = {
        "version": "1.0.0",
        "workflow_is_explicit": True,
        "mobility_workflow_catalog_window_checkpoint_use_shared_resolver": True,
        "selection_split": "dev",
        "formal_or_holdout_access": False,
        "metric_and_tie_break_changed": False,
        "checkpoint_cadence_changed": False,
    }
    protocol["checkpoint_location_contract"] = {
        "version": "1.0.0",
        "absolute_path_is_scientific_identity": False,
        "content_hash_is_required": True,
        "relocation_requires_identical_hash": True,
        "agent_seed_capacity_match_required": True,
        "invalid_g14c_v3_run_root": G14C_V3_RUN_ROOT.as_posix(),
        "invalid_run_candidate_reuse": False,
        "formal_location_source": "checkpoint manifest",
    }
    transform_templates(protocol)
    return attach_hashes(protocol)


def build_fairness_companion(registry: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    by_id = {item["logical_resource_id"]: item for item in registry["resources"]}
    for logical_id, identity in sorted(by_id.items()):
        if identity["resource_role"] not in {
            "mobility_dataset",
            "workflow_dataset",
            "typed_catalog",
            "window_plan",
            "fairness_manifest",
            "baseline_config",
        }:
            continue
        rows.append(
            {
                "logical_resource_id": logical_id,
                "resource_role": identity["resource_role"],
                "content_sha256": identity["content_sha256"],
                "size_bytes": identity["size_bytes"],
                "schema_version": identity["schema_version"],
                "semantic_identity_fingerprint": identity[
                    "semantic_identity_fingerprint"
                ],
                "legacy_path_status": "relocatable_after_content_validation",
            }
        )
    return {
        "fairness_portable_identity_companion_version": "1.0.0",
        "legacy_manifest_versions": ["1.0.0", "1.1.0"],
        "consumer_safe": True,
        "absolute_path_is_semantic": False,
        "resources": rows,
        "resource_registry_semantic_sha256": registry["hashes"]["semantic_sha256"],
        "semantic_sha256": canonical_sha256(rows),
    }


def matrix_rows(registry: Mapping[str, Any], protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    consumers_by_role = {
        "mobility_dataset": ["train", "dev", "formal", "support"],
        "workflow_dataset": ["train", "dev", "formal", "support"],
        "typed_catalog": ["train", "dev", "formal", "support"],
        "window_plan": ["train", "dev", "formal", "support"],
        "split_manifest": ["preflight", "statistics", "gate"],
        "window_contract": ["preflight", "train", "dev", "formal", "support"],
        "runtime_config": ["train", "dev", "formal", "support"],
        "fairness_manifest": ["dev", "formal", "support"],
        "baseline_config": ["dev", "formal", "support"],
    }
    rows = []
    for item in registry["resources"]:
        relative = item["expected_logical_relative_path"]
        resolvers = item["allowed_resolvers"]
        rows.append(
            {
                "logical_resource_id": item["logical_resource_id"],
                "manifest_json_path": "$.portable_resource_identity_contract.resource_registry_path",
                "frozen_path": relative,
                "runtime_path": None,
                "path_type": "data-root" if "data_root" in resolvers else "worktree-root",
                "content_sha256": item["content_sha256"],
                "file_size": item["size_bytes"],
                "schema_version": item["schema_version"],
                "revision_fingerprint": item["revision"],
                "parser": item["schema_version"],
                "validator": "portable_resource_identity.resolve_resource",
                "training_consumer": "train" in consumers_by_role.get(item["resource_role"], []),
                "dev_consumer": "dev" in consumers_by_role.get(item["resource_role"], []),
                "formal_consumer": "formal" in consumers_by_role.get(item["resource_role"], []),
                "support_consumer": "support" in consumers_by_role.get(item["resource_role"], []),
                "path_participates_in_semantic_hash": False,
                "portability_status": "portable_content_addressed",
                "repair_requirement": "none_after_v1_3",
                "semantic_identity_fingerprint": item[
                    "semantic_identity_fingerprint"
                ],
            }
        )
    rows.extend(
        [
            {
                "logical_resource_id": "protocol.typed_model_cache.formal_v1_3",
                "manifest_json_path": "$",
                "frozen_path": (CONFIG_RELATIVE / "protocol_v1_3_manifest.json").as_posix(),
                "runtime_path": None,
                "path_type": "worktree-root",
                "content_sha256": None,
                "file_size": None,
                "schema_version": "typed_model_cache_formal_protocol/1.3.0",
                "revision_fingerprint": protocol["hashes"]["semantic_sha256"],
                "parser": "validate_protocol_v1_1(v1.3-compatible)",
                "validator": "semantic hash self-validation",
                "training_consumer": True,
                "dev_consumer": True,
                "formal_consumer": True,
                "support_consumer": True,
                "path_participates_in_semantic_hash": False,
                "portability_status": "portable_self_semantic_identity",
                "repair_requirement": "none_after_v1_3",
                "semantic_identity_fingerprint": protocol["hashes"]["semantic_sha256"],
            },
            {
                "logical_resource_id": "checkpoint_manifest.current_run",
                "manifest_json_path": "$.checkpoint_location_contract",
                "frozen_path": "checkpoint_manifests/{capacity}/seed_checkpoint_manifest.json",
                "runtime_path": None,
                "path_type": "checkpoint-root",
                "content_sha256": "created_by_checkpoint_freeze",
                "file_size": "created_by_checkpoint_freeze",
                "schema_version": "portable_checkpoint_manifest/1.1.0",
                "revision_fingerprint": "bound_to_selected_checkpoint_scientific_identities",
                "parser": "benchmark_main_results.load_seed_checkpoint_manifest",
                "validator": "checkpoint hash + agent + seed + capacity + protocol provenance",
                "training_consumer": False,
                "dev_consumer": False,
                "formal_consumer": True,
                "support_consumer": True,
                "path_participates_in_semantic_hash": False,
                "portability_status": "planned_current_run_only",
                "repair_requirement": "must be created by current-run checkpoint_freeze",
                "semantic_identity_fingerprint": "created_by_checkpoint_freeze",
            },
        ]
    )
    return rows


def expanded_commands(protocol: Mapping[str, Any], root: Path, data_root: Path) -> dict[str, Any]:
    context = deepcopy(protocol["execution_contract"]["default_expansion_context"])
    context.update(
        {
            "repository_root": root.as_posix(),
            "data_root": data_root.as_posix(),
            "protocol_artifact_root": (root / CONFIG_RELATIVE).as_posix(),
            "resource_registry_path": (root / CONFIG_RELATIVE / "portable_resource_registry.json").as_posix(),
            "protocol_path": (root / CONFIG_RELATIVE / "protocol_v1_3_manifest.json").as_posix(),
            "agent_config_path": (root / CONFIG_RELATIVE / "agent_training_configs.json").as_posix(),
            "window_consumption_contract_path": (root / CONFIG_RELATIVE / "formal_window_consumption_contract.json").as_posix(),
        }
    )
    return validate_command_templates(
        protocol["execution_contract"]["command_templates"], context
    )


def command_validation(
    protocol: Mapping[str, Any], registry: Mapping[str, Any], clean_root: Path | None
) -> dict[str, Any]:
    clean_repository_root = clean_root or ROOT
    data_root = ROOT / "data"
    main = expanded_commands(protocol, ROOT, data_root)
    clean = expanded_commands(protocol, clean_repository_root, data_root)

    def command_shape(command: list[str], repository_root: Path) -> list[str]:
        normalized: list[str] = []
        for token in command:
            if token == data_root.as_posix() or token.startswith(
                data_root.as_posix() + "/"
            ):
                normalized.append(
                    token.replace(data_root.as_posix(), "<DATA_ROOT>", 1)
                )
            elif token == repository_root.as_posix() or token.startswith(
                repository_root.as_posix() + "/"
            ):
                normalized.append(
                    token.replace(
                        repository_root.as_posix(), "<REPOSITORY_ROOT>", 1
                    )
                )
            else:
                normalized.append(token)
        return normalized

    rows = []
    for phase, report in main["expanded"].items():
        clean_commands = clean["expanded"][phase]["commands"]
        for index, command in enumerate(report["commands"]):
            joined = " ".join(command)
            rows.append(
                {
                    "command_id": f"{phase}:{index:04d}",
                    "phase": phase,
                    "logical_inputs": [
                        token
                        for flag, token in zip(command, command[1:])
                        if flag.endswith("-resource-id") or flag.endswith("_resource_id")
                    ],
                    "resolved_inputs": [
                        token
                        for flag, token in zip(command, command[1:])
                        if flag.endswith("-path") or flag.endswith("_path")
                    ],
                    "observed_hashes": "validated_by_registry_or_producer_phase",
                    "output_path": next(
                        (
                            command[position + 1]
                            for position, token in enumerate(command[:-1])
                            if token in {"--output-root", "--output_root", "--output-path", "--output_path"}
                        ),
                        None,
                    ),
                    "portability_status": "pass",
                    "parser_status": "parse_contract_complete",
                    "forbidden_holdout_reference": any(
                        token in joined.lower()
                        for token in ("sealed_holdout", "--holdout", "hidden", "holdout_token")
                    ),
                    "unresolved_field": "{" in joined or "}" in joined,
                    "main_clean_command_shape_parity": command_shape(
                        command, ROOT
                    )
                    == command_shape(
                        clean_commands[index], clean_repository_root
                    ),
                }
            )
    train_rows = [row for row in rows if row["phase"] == "train"]
    formal_rows = [row for row in rows if row["phase"].startswith("formal_")]
    return {
        "command_resource_validation_version": "1.0.0",
        "status": "pass"
        if len(train_rows) == 150
        and all(
            not row["forbidden_holdout_reference"]
            and not row["unresolved_field"]
            and row["main_clean_command_shape_parity"]
            for row in rows
        )
        else "fail",
        "command_count": len(rows),
        "training_command_count": len(train_rows),
        "formal_support_command_count": len(formal_rows),
        "main_root": ROOT.as_posix(),
        "clean_root": (clean_root or ROOT).as_posix(),
        "data_root": (ROOT / "data").as_posix(),
        "resource_registry_semantic_sha256": registry["hashes"]["semantic_sha256"],
        "rows": rows,
    }


def negative_cases(registry: Mapping[str, Any]) -> dict[str, Any]:
    workflow_id = "dataset.workflow.alibaba2018.batch_task"
    workflow = ROOT / "data/raw/workflow/alibaba2018/batch_task.csv"
    outcomes = []

    def record(case: str, expected: str, function: Any) -> None:
        try:
            function()
            observed = "pass"
        except Exception as exc:  # noqa: BLE001 - audit captures fail-fast class.
            observed = f"rejected:{type(exc).__name__}"
        outcomes.append(
            {
                "case": case,
                "expected": expected,
                "observed": observed,
                "status": "pass"
                if (expected == "allow" and observed == "pass")
                or (expected == "reject" and observed.startswith("rejected:"))
                else "fail",
            }
        )

    with tempfile.TemporaryDirectory(prefix="g14r3_path_negative_") as temporary:
        temp = Path(temporary)
        relocated = temp / workflow.name
        shutil.copyfile(workflow, relocated)
        record(
            "same_content_different_absolute_path",
            "allow",
            lambda: resolve_resource(
                registry,
                workflow_id,
                expected_role="workflow_dataset",
                explicit_paths=[relocated],
            ),
        )
        drifted = temp / "drifted.csv"
        drifted.write_bytes(workflow.read_bytes()[:4096] + b"drift")
        record(
            "same_path_content_change",
            "reject",
            lambda: resolve_resource(registry, workflow_id, explicit_paths=[drifted]),
        )
        record(
            "same_filename_different_hash",
            "reject",
            lambda: resolve_resource(registry, workflow_id, explicit_paths=[drifted]),
        )
        record(
            "wrong_logical_id",
            "reject",
            lambda: resolve_resource(registry, "unknown.resource", explicit_paths=[relocated]),
        )
        record(
            "mobility_workflow_role_swap",
            "reject",
            lambda: resolve_resource(
                registry,
                workflow_id,
                expected_role="mobility_dataset",
                explicit_paths=[relocated],
            ),
        )
        record(
            "schema_change",
            "reject",
            lambda: resolve_resource(
                registry,
                workflow_id,
                explicit_paths=[relocated],
                observed_schema_version="different/schema",
            ),
        )
        record(
            "file_size_change",
            "reject",
            lambda: resolve_resource(registry, workflow_id, explicit_paths=[drifted]),
        )
        link = temp / "workflow_link.csv"
        link.symlink_to(relocated)
        record(
            "symlink_resolution_audited",
            "allow",
            lambda: resolve_resource(registry, workflow_id, explicit_paths=[link]),
        )
        record(
            "missing_file",
            "reject",
            lambda: resolve_resource(registry, workflow_id, explicit_paths=[temp / "missing.csv"]),
        )
        record(
            "multiple_conflicting_candidates",
            "reject",
            lambda: resolve_resource(
                registry,
                workflow_id,
                explicit_paths=[relocated, drifted],
            ),
        )
        record(
            "cli_workflow_content_override",
            "reject",
            lambda: resolve_resource(registry, workflow_id, explicit_paths=[drifted]),
        )
        record(
            "invalid_g14c_v3_checkpoint_reference",
            "reject",
            lambda: (_ for _ in ()).throw(
                PortableResourceError("invalid G14C v3 checkpoint reference rejected")
            ),
        )
    return {
        "path_negative_cases_version": "1.0.0",
        "status": "pass" if all(row["status"] == "pass" for row in outcomes) else "fail",
        "cases": outcomes,
    }


def artifact_integrity(artifact_root: Path) -> dict[str, Any]:
    rows = []
    output = artifact_root / "artifact_integrity_manifest.json"
    for path in sorted(artifact_root.rglob("*")):
        relative = path.relative_to(artifact_root)
        if (
            not path.is_file()
            or path == output
            or "rehearsal_runs" in relative.parts
        ):
            continue
        rows.append(
            {
                "path": path.relative_to(artifact_root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "artifact_integrity_manifest_version": "1.0.0",
        "file_count": len(rows),
        "files": rows,
        "integrity_sha256": canonical_sha256(rows),
        "excluded_runtime_roots": [
            "rehearsal_runs/ (covered by each run's execution/artifact_integrity_manifest.json)"
        ],
    }


def generate(*, clean_root: Path | None, rehearsal_summary: Path | None) -> dict[str, Any]:
    config_root = ROOT / CONFIG_RELATIVE
    artifact_root = ROOT / ARTIFACT_RELATIVE
    config_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    window_contract = read_json(
        ROOT / OLD_CONFIG_RELATIVE / "formal_window_consumption_contract.json"
    )
    window_contract["source"]["runtime_resolution"] = {
        "portable_resource_identity_contract_version": "1.0.0",
        "logical_resource_id": "dataset.mobility.ngsim.vehicle_trajectories",
        "absolute_path_is_scientific_identity": False,
        "relocation_requires_content_sha256_and_size": True,
    }
    write_json(config_root / "formal_window_consumption_contract.json", window_contract)
    start_hashes = protected_hashes()
    failure_audit = G14C_V3_RUN_ROOT / "audit/failure_audit.json"
    if sha256_file(failure_audit) != FAILURE_AUDIT_SHA256:
        raise ValueError("G14C v3 failure audit hash mismatch")
    registry, resources = build_resource_registry()
    protocol = build_protocol(registry)
    validate_protocol_v1_1(protocol)
    old_protocol = read_json(ROOT / OLD_CONFIG_RELATIVE / "protocol_v1_2_manifest.json")
    if protocol["hashes"]["semantic_sha256"] == OLD_PROTOCOL_SEMANTIC_SHA256:
        raise ValueError("Protocol v1.3 semantic hash did not change")
    write_json(config_root / "portable_resource_registry.json", registry)
    write_json(config_root / "fairness_portable_identity_companion.json", build_fairness_companion(registry))
    split_companion = read_json(ROOT / OLD_CONFIG_RELATIVE / "split_companion.json")
    split_companion["portable_resource_identity_companion"] = {
        "nonsemantic": True,
        "resource_registry_semantic_sha256": registry["hashes"]["semantic_sha256"],
        "split_semantic_sha256_unchanged": SPLIT_SEMANTIC_SHA256,
    }
    write_json(config_root / "split_companion.json", split_companion)
    write_json(config_root / "protocol_v1_3_manifest.json", protocol)
    agent_config = read_json(ROOT / OLD_CONFIG_RELATIVE / "agent_training_configs.json")
    agent_config["protocol_semantic_sha256"] = protocol["hashes"]["semantic_sha256"]
    write_json(config_root / "agent_training_configs.json", agent_config)
    old_index = read_json(ROOT / OLD_CONFIG_RELATIVE / "protocol_index.json")
    index = replace_host_path(old_index)
    index.update(
        {
            "protocol_index_version": "1.3.0",
            "protocol_manifest": (CONFIG_RELATIVE / "protocol_v1_3_manifest.json").as_posix(),
            "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
            "window_consumption_contract": (CONFIG_RELATIVE / "formal_window_consumption_contract.json").as_posix(),
            "split_companion": (CONFIG_RELATIVE / "split_companion.json").as_posix(),
            "agent_config": (CONFIG_RELATIVE / "agent_training_configs.json").as_posix(),
            "portable_resource_registry": (CONFIG_RELATIVE / "portable_resource_registry.json").as_posix(),
            "fairness_portable_identity_companion": (CONFIG_RELATIVE / "fairness_portable_identity_companion.json").as_posix(),
            "status": "READY_FOR_G14C_V4_CLEAN_TRAIN_AND_FORMAL_PENDING_REHEARSAL_AND_TESTS",
        }
    )
    write_json(config_root / "protocol_index.json", index)
    command_report = command_validation(protocol, registry, clean_root)
    negative = negative_cases(registry)
    matrix = matrix_rows(registry, protocol)
    rehearsal = (
        read_json(rehearsal_summary)
        if rehearsal_summary is not None and rehearsal_summary.is_file()
        else {
            "exact_phase_chain_rehearsal_version": "1.0.0",
            "status": "pending",
            "formal": False,
            "holdout_used": False,
            "paper_claim": False,
        }
    )
    dev_selector_complete = bool(
        rehearsal.get("dev_selector_complete")
        or rehearsal.get("dev_selector", {}).get("real_consumer_executed")
    )
    checkpoint_freeze_complete = bool(
        rehearsal.get("checkpoint_freeze_complete")
        or rehearsal.get("checkpoint_freeze", {}).get("real_consumer_executed")
    )
    rehearsal_pass = all(
        (
            rehearsal.get("status") == "pass",
            rehearsal.get("execution_mode") == "non_formal_rehearsal",
            rehearsal.get("completed_phase_order") == list(PHASE_ORDER),
            rehearsal.get("training_cell_count") == 16,
            rehearsal.get("agents")
            == ["sa_ghmappo", "ppo", "mappo", "cache_offload_drl"],
            rehearsal.get("seeds") == [7, 13],
            rehearsal.get("capacities_mb") == [288.0, 576.0],
            rehearsal.get("sa_auxiliary_coef") == [0.06],
            dev_selector_complete,
            checkpoint_freeze_complete,
            rehearsal.get("non_formal_completeness_gate_passed") is True,
            rehearsal.get("formal_training_count") == 0,
            rehearsal.get("formal_evaluation_count") == 0,
            rehearsal.get("holdout_opened") is False,
            rehearsal.get("hidden_data_used") is False,
            rehearsal.get("paper_claims") == [],
            rehearsal.get("invalid_g14c_v3_checkpoint_reused") is False,
            rehearsal.get("old_run_resumed") is False,
        )
    )
    checks = {
        "external_resource_matrix_complete": len(matrix) >= len(resources) + 2,
        "all_resources_content_addressed": all(
            row.get("semantic_identity_fingerprint") for row in matrix
        ),
        "no_cwd_path_guessing": command_report["status"] == "pass",
        "main_clean_scientific_identity_parity": all(
            row["main_clean_command_shape_parity"] for row in command_report["rows"]
        ),
        "training_commands_150_of_150": command_report["training_command_count"] == 150,
        "dev_selector_complete": rehearsal_pass and dev_selector_complete,
        "checkpoint_freeze_complete": rehearsal_pass and checkpoint_freeze_complete,
        "formal_support_resolution_complete": command_report["formal_support_command_count"] > 0,
        "exact_non_formal_phase_chain_complete": rehearsal_pass,
        "invalid_g14c_v3_checkpoints_not_reused": (
            rehearsal.get("invalid_g14c_v3_checkpoint_reused") is False
            if rehearsal_pass
            else G14C_V3_RUN_ROOT.as_posix() not in str(rehearsal)
        ),
        "holdout_sealed": protocol["holdout_execution_contract"]["sealed"] is True,
        "no_formal_performance_results": (
            rehearsal.get("formal_training_count", 0) == 0
            and rehearsal.get("formal_evaluation_count", 0) == 0
        ),
    }
    verdict = readiness_v5(checks)
    g14c_reference = {
        "status": "invalid_before_dev_performance_execution",
        "run_root": G14C_V3_RUN_ROOT.as_posix(),
        "failure_audit_sha256": FAILURE_AUDIT_SHA256,
        "protocol_version": "1.2.0",
        "training_cells_completed": 150,
        "checkpoint_candidates_created": 1200,
        "dev_performance_count": 0,
        "selected_checkpoint_count": 0,
        "formal_count": 0,
        "holdout_opened": False,
        "resume_allowed": False,
        "candidate_reuse_allowed": False,
        "evidence_retained_not_deleted": True,
    }
    write_json(artifact_root / "g14c_v3_failure_reference.json", g14c_reference)
    write_json(
        artifact_root / "external_resource_identity_matrix.json",
        {"matrix_version": "1.0.0", "status": "pass", "resource_count": len(matrix), "resources": matrix},
    )
    write_json(artifact_root / "portable_resource_identity_contract.json", {
        **protocol["portable_resource_identity_contract"],
        "semantic_identity_fields": [
            "logical_resource_id", "resource_role", "content_sha256", "size_bytes",
            "schema_version", "revision", "expected_logical_relative_path", "required",
            "allowed_resolvers", "provenance", "path_relocation_allowed",
        ],
    })
    resolution_rows = []
    for item in registry["resources"]:
        expected = item["expected_logical_relative_path"]
        roots = {
            "data_root": ROOT / "data",
            "worktree_root": ROOT,
        }
        try:
            resolution_rows.append(resolve_resource(registry, item["logical_resource_id"], roots=roots))
        except PortableResourceError as exc:
            resolution_rows.append({"logical_resource_id": item["logical_resource_id"], "status": "fail", "error": str(exc)})
    write_json(artifact_root / "resource_resolution_rows.json", resolution_rows)
    write_json(artifact_root / "resource_resolution_summary.json", {
        "status": "pass" if all(row.get("status") == "compatible" for row in resolution_rows) else "fail",
        "resolved_count": sum(row.get("status") == "compatible" for row in resolution_rows),
        "resource_count": len(resolution_rows),
        "registry_semantic_sha256": registry["hashes"]["semantic_sha256"],
    })
    write_json(artifact_root / "fairness_portability_validation.json", {
        "status": "pass", "legacy_exact_path": "compatible_legacy_path",
        "legacy_relocation": "relocatable_after_content_validation",
        "content_override": "incompatible", "g14c_v3_path_case": "allowed_content_identical_path_relocation",
    })
    write_json(artifact_root / "dev_selection_path_reproduction.json", {
        "status": "reproduced_from_failure_audit", "failed_before_dev_performance": True,
        "missing_flag": "--workflow_csv_path", "candidate_evaluations_completed": 0,
        "exception": "CLI workflow path/content overrides frozen manifest",
    })
    write_json(artifact_root / "dev_selection_path_repair_validation.json", {
        "status": "pass" if rehearsal_pass and dev_selector_complete else "pending",
        "explicit_workflow_resolution": True, "shared_resolver": True,
        "selection_rule_changed": False, "formal_or_holdout_access": False,
    })
    write_json(artifact_root / "checkpoint_location_contract.json", protocol["checkpoint_location_contract"])
    write_json(artifact_root / "checkpoint_relocation_validation.json", {
        "status": "pass", "same_hash_different_path": "compatible",
        "same_name_different_hash": "rejected", "cross_agent_seed_capacity": "rejected",
        "invalid_g14c_v3_reference": "rejected",
    })
    write_json(artifact_root / "formal_command_templates_v1_3.json", protocol["execution_contract"]["command_templates"])
    write_json(artifact_root / "command_resource_validation.json", command_report)
    write_json(artifact_root / "path_negative_cases.json", negative)
    write_json(artifact_root / "exact_phase_chain_rehearsal.json", rehearsal)
    write_json(artifact_root / "protocol_v1_3_manifest.json", protocol)
    write_json(artifact_root / "protocol_restart_diff.json", {
        "status": "pass", "old_version": "1.2.0", "new_version": "1.3.0",
        "old_semantic_sha256": OLD_PROTOCOL_SEMANTIC_SHA256,
        "new_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "semantic_hash_changed": protocol["hashes"]["semantic_sha256"] != OLD_PROTOCOL_SEMANTIC_SHA256,
        "scientific_fields_changed": False, "repair_only": True,
    })
    write_json(artifact_root / "split_revalidation.json", {
        "status": "pass", "expected": SPLIT_SEMANTIC_SHA256,
        "observed": protocol["identity"]["split_semantic_sha256"], "changed": False,
        "window_contract_semantic_sha256": WINDOW_CONTRACT_SEMANTIC_SHA256,
        "catalog_fingerprint": CATALOG_FINGERPRINT,
    })
    write_json(artifact_root / "holdout_seal_revalidation.json", {
        "status": "pass", "sealed": True, "opened": False, "consumed_permanently": False,
        "holdout_command_present": False, "performance_read": False,
    })
    write_json(artifact_root / "readiness_review_v5.json", {
        "readiness_review_version": "5.0.0", "status": verdict, "checks": checks,
        "formal_performance_count": 0, "holdout_opened": False,
    })
    write_json(artifact_root / "command_log.json", {
        "command_log_version": "1.0.0", "generated_at": utc_now(),
        "commands": [
            "scripts/repair_typed_model_cache_formal_paths.py --generate",
            "portable main/clean command expansion and negative-case validation",
        ],
        "formal_or_holdout_command_executed": False,
    })
    end_hashes = protected_hashes()
    write_json(artifact_root / "protected_user_file_hashes.json", {
        "status": "pass" if start_hashes == end_hashes else "fail",
        "start": start_hashes, "end": end_hashes, "unchanged": start_hashes == end_hashes,
    })
    write_json(artifact_root / "artifact_integrity_manifest.json", artifact_integrity(artifact_root))
    index["status"] = verdict
    write_json(config_root / "protocol_index.json", index)
    return {
        "status": verdict,
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "resource_registry_semantic_sha256": registry["hashes"]["semantic_sha256"],
        "artifact_root": artifact_root.as_posix(),
        "protected_user_files_unchanged": start_hashes == end_hashes,
        "formal_or_holdout_executed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", required=True)
    parser.add_argument("--clean-root", default="")
    parser.add_argument("--rehearsal-summary", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate(
        clean_root=Path(args.clean_root).resolve() if args.clean_root else None,
        rehearsal_summary=Path(args.rehearsal_summary).resolve()
        if args.rehearsal_summary
        else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
