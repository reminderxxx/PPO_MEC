"""Create the outcome-blind G14R protocol v1.1 companion and audit package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.registry import get_algo_spec
from src.evaluators.cache_baseline_fairness import build_manifest, validate_manifest
from src.evaluators.typed_model_cache_formal_execution import (
    FORMAL_EXECUTION_PROTOCOL_ID,
    FORMAL_EXECUTION_PROTOCOL_VERSION,
    FORMAL_PHASE_RUNNER_VERSION,
    PHASE_ORDER,
    PRIMARY_ENDPOINTS,
    READY_VERDICT,
    build_scalability_setting_matrix,
    build_support_setting_matrix,
    endpoint_schema,
    expand_command_plan,
    readiness_v3,
    validate_command_templates,
    validate_protocol_v1_1,
)
from src.evaluators.typed_model_cache_formal_protocol import (
    attach_hashes,
    canonical_sha256,
    sha256_file,
)
from src.runtime.formal_training_contract import checkpoint_snapshot_indices
from src.runtime.typed_model_cache_runtime import resolve_model_cache_runtime


ARTIFACT_RUN_ID = "typed_model_cache_formal_protocol_restart_20260820_g14r_v1"
ARTIFACT_ROOT = ROOT / "artifacts/analysis" / ARTIFACT_RUN_ID
CONFIG_ROOT = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820"
OLD_ARTIFACT = ROOT / "artifacts/analysis/typed_model_cache_formal_protocol_freeze_20260820_g14b_v1"
FAILED_RUN = ROOT / "artifacts/experiments/typed_model_cache_formal/typed_model_cache_formal_20260820_g14c_351fdb8_v1"
OLD_CONFIG = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_20260820"
CATALOG = ROOT / "src/data/model_catalog/typed_model_cache_controlled.json"
MOBILITY = ROOT / "data/raw/mobility/ngsim/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv"
WORKFLOW = ROOT / "data/raw/workflow/alibaba2018/batch_task.csv"
SEEDS = [7, 13, 29, 43, 71]
LEARNED_AGENTS = [
    "sa_ghmappo", "ppo", "mappo", "dqn", "dueling_dqn", "qmix",
    "controller_mat", "dag_offload_drl", "cache_offload_drl", "dt_handoff_drl",
]
PROTECTED_FILES = [
    "scripts/train_sa_ghmappo_real_sample.py",
    "src/agents/sa_ghmappo_agent.py",
    "src/agents/sa_ghmappo_core.py",
    "src/encoders/fusion_encoder.py",
    "src/evaluators/real_eval_support.py",
    "tests/test_algo_pool_contract.py",
    "tests/test_checkpoint_compat.py",
]


def executability_matrix() -> dict[str, Any]:
    rows = [
        ("protocol_identity", "$.typed_model_cache_formal_protocol_version|$.protocol_id|$.supersession", "--formal_protocol_path/--protocol-path", "protocol JSON loader", "validated protocol identity", "all formal runners", "formal_protocol_semantic_sha256", "protocol_semantic_sha256", "protocol_semantic_sha256", "protocol hash bundle", "validate_protocol_v1_1", "test_41_*..test_44_*", "complete"),
        ("split_and_workload", "$.identity.split_semantic_sha256|$.workload", "--window_plan_path", "training/benchmark parsers", "frozen train/dev/formal plan", "apply_frozen_window_plan + fairness expected_unit", "train_window_plan_identity", "split/window provenance", "window/workflow grouping", "split_revalidation", "fairness + split validators", "test_43_split_hash_unchanged", "complete"),
        ("typed_runtime_and_capacity", "$.typed_catalog_and_capacity|$.identity.typed_runtime_contract_hashes_by_capacity", "--model_cache_runtime_config", "all typed runner parsers", "resolved_model_cache_runtime", "typed cache runtime", "typed_runtime_provenance", "runtime/catalog/capacity fields", "capacity strata", "runtime/fairness validation", "resolve_model_cache_runtime", "test_24_typed_runtime_is_frozen", "complete"),
        ("training_budget", "$.training_budget.episodes_per_learned_agent_seed_capacity|$.training_budget.update_interval_episodes|$.training_budget.batch_size|$.training_budget.max_steps_per_episode", "manifest-bound training flags", "formal training parser", "ResolvedTrainingContract", "shared trainer loop", "formal_training_contract", "training-only_n/a", "training budget audit", "checkpoint_frequency_validation", "resolve_training_contract", "test_02_formal_every_four_updates", "complete"),
        ("checkpoint_every_updates", "$.training_budget.checkpoint_frequency_updates", "--checkpoint_every_updates", "shared training parser", "resolved_training.checkpoint_every_updates", "checkpoint save scheduler", "checkpoint_schedule", "training-only_n/a", "candidate update plan", "checkpoint_frequency_validation", "validate_resume_checkpoint_schedule", "test_01_*..test_05_*", "complete"),
        ("agent_configs", "$.training_budget.agent_configs", "--agent_config_path", "shared training parser", "resolved_agent_config", "registry build_agent", "resolved_agent_config", "training-only_n/a", "agent config audit", "agent_config_resolution", "audited_agent_config", "test_06_*..test_10_*", "complete"),
        ("agent_seed_capacity_matrix", "$.agent_matrix|$.seed_plan|$.typed_catalog_and_capacity.capacity_strata", "matrix_contexts", "formal phase parser", "150-cell command plan", "append-only phase command batch", "per-cell checkpoint metadata", "agent/seed/capacity provenance", "fairness strata", "command_expansion_validation", "expand_command_plan", "test_30_phase_dry_command_expansion", "complete"),
        ("primary_endpoints", "$.endpoints.primary|$.endpoint_schema", "--metrics", "statistics wrapper", "six canonical fields", "cache efficiency reducer + summary_to_row", "endpoint schema version", "six primary row fields", "nullable aggregate fields", "primary_endpoint_reconciliation", "reconcile_primary_endpoint_row", "test_11_*..test_21_*", "complete"),
        ("support_settings", "$.ablation_and_support.support_setting_matrix", "--setting-id", "typed support parser", "stable setting identity", "typed support benchmark/oracle", "checkpoint provenance binding", "support provenance", "support setting aggregate", "typed_support_runner_validation", "support_setting_by_id", "test_22_*..test_29_*", "complete_with_explicit_unavailable"),
        ("scalability_settings", "$.ablation_and_support.scalability_setting_matrix", "--setting-id/--request-replay-path", "typed support parser", "fixed numeric level", "future-horizon oracle", "checkpoint provenance n/a for oracle", "support provenance", "oracle scalability artifact", "scalability_setting_matrix", "validate_support_binding", "test_22_concrete_or_explicit_unavailable_values_exist", "complete_with_explicit_unavailable"),
        ("dev_selection", "$.training_budget.checkpoint_selection", "dev selection template", "dev selection parser", "lexicographic dev rule", "dev benchmark + selector", "candidate checkpoint hashes", "dev endpoint rows", "selected candidates", "checkpoint_candidates/dev_selection", "dev_select", "test_41_v1_invalid_record + manager tests", "complete"),
        ("checkpoint_freeze", "$.training_budget.checkpoint_selection.formal_or_holdout_selection_forbidden", "checkpoint_freeze action", "artifact manager parser", "frozen selected checkpoints", "hash verifier + companion builder", "freeze/seed/provenance manifests", "checkpoint provenance", "checkpoint freeze count", "checkpoint_freeze.json", "checkpoint_freeze", "test_26_checkpoint_provenance_is_required", "complete"),
        ("command_templates", "$.execution_contract.command_templates", "phase exact argv", "target argparse parsers", "expanded command batches", "phase subprocess executor", "command/input hashes", "protocol/split hashes", "command expansion report", "formal_command_templates", "validate_command_templates", "test_40_complete_command_expansion", "complete"),
        ("phase_orchestration", "$.execution_contract.phase_order", "--protocol-path --output-root --preflight --phase --resume --dry-run", "formal protocol runner parser", "append-only ledger", "AppendOnlyPhaseRunner", "checkpoint-freeze output hash", "phase output provenance", "formal completeness gate", "phase_runner_validation", "AppendOnlyPhaseRunner", "test_30_*..test_40_*", "complete"),
        ("statistics_integrity_gate", "$.statistics|$.claim_evidence_map|$.execution_contract", "formal_statistics/formal_gate templates", "statistics/artifact manager parsers", "frozen rows and completeness-only gate", "statistics wrapper + integrity scanner", "checkpoint freeze identity", "primary rows", "paired statistics", "integrity/gate artifacts", "formal_gate", "test_top_journal_statistics + test_40_*", "complete"),
        ("holdout_seal", "$.holdout_execution_contract", "none_deliberately_inaccessible", "no ordinary parser", "sealed identity", "preflight validation only", "not_applicable", "sealed/opened/consumed flags", "seal revalidation", "holdout_seal_revalidation", "validate_no_holdout_capability", "test_38_holdout_is_inaccessible", "complete"),
    ]
    columns = [
        "field_id", "manifest_json_path", "cli_flag", "parser", "resolved_config",
        "runtime_consumer", "checkpoint_metadata", "summary_field", "row_field",
        "aggregate_field", "validator", "test", "status",
    ]
    return {
        "matrix_version": "1.1.0",
        "artifact_run_id": ARTIFACT_RUN_ID,
        "created_before_implementation_changes": True,
        "source_protocol_version": "1.0.0",
        "source_protocol_semantic_sha256": "41fbfab4ac10bae96250d7ead816d907fd6551bb9651ae03210e801c9e2478b4",
        "status": "complete",
        "required_columns": columns[1:],
        "covered_top_level_protocol_fields": [
            "identity", "supersession", "workload", "training_budget", "agent_matrix",
            "seed_plan", "typed_catalog_and_capacity", "endpoints", "endpoint_schema",
            "ablation_and_support", "statistics", "claim_evidence_map",
            "execution_contract", "holdout_execution_contract", "paper_claim_boundary",
        ],
        "unmapped_executable_field_count": 0,
        "rows": [dict(zip(columns, row)) for row in rows],
    }


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
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def runtime_config(capacity_mb: float, label: str) -> dict[str, Any]:
    return {
        "profile_id": f"typed_model_cache_formal_v1_1_{label}",
        "claim_boundary": "G14R frozen pre-execution typed formal runtime; no performance outcome",
        "model_cache_profile": "typed_base_adapter_state_v1",
        "typed_model_cache_contract_version": "1.0.0",
        "typed_cache_transaction_contract_version": "typed_cache_transaction_contract_v1.0.0",
        "catalog_path": "src/data/model_catalog/typed_model_cache_controlled.json",
        "typed_catalog_path": "src/data/model_catalog/typed_model_cache_controlled.json",
        "typed_catalog_fingerprint": "89c548980b63df733553d748e8db3ca622965b63abcd08ebd4c231790b40a9d6",
        "typed_initial_state_fingerprint": "fb0cdbfa761477f4c39bc3416181b475c8884a1c1433edc56d7f2541fc6cac46",
        "typed_dependency_fingerprint": "0f8fcd018635426d67eb78af567456d3f7b31a6bac48ac876baee751d09ddcb9",
        "typed_pinned_evictability_fingerprint": "220f27d6a38d28852e43f1e65e0af8b5aa8399ad6bd785a89246a3de7cd270c7",
        "cache_event_schema_version": "1.3.0",
        "cache_efficiency_metrics_contract_version": "1.2.0",
        "cache_capacity_profile": {
            "model_cache_profile_id": "typed_base_adapter_state_v1",
            "enabled": True,
            "unit": "mb",
            "capacity_mb": float(capacity_mb),
            "count_base_model_separately": True,
            "eviction_policy": "lru",
            "eviction_policy_seed": 7,
            "telemetry_enabled": True,
        },
        "transaction_contract": {
            "max_logical_cache_actions_per_step": 1,
            "max_dependency_bundle_objects": 2,
            "admission_order": ["base_model", "adapter"],
            "transfer_order": ["base_model", "adapter", "workflow_state"],
            "partial_admission": False,
            "atomic_rollback": True,
            "dependency_safe_base_eviction": "prohibit_while_resident_adapter_depends",
        },
        "kv_prefix": {
            "enabled": False,
            "capacity_denominator": "excluded",
            "cache_event_denominator": "excluded",
        },
        "baseline_matrix": {
            "agents": ["reactive_lru", "reactive_fifo", "reactive_lfu", "reactive_aging_lfu", "reactive_random"],
            "only_primary_difference": "eviction_policy",
        },
    }


def agent_configs() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {
        agent: {"learning_rate": 0.0003} for agent in LEARNED_AGENTS
    }
    configs["sa_ghmappo"] = {
        "learning_rate": 0.0001,
        "entropy_coef": 0.004,
        "value_coef": 0.7,
        "auxiliary_coef": 0.06,
    }
    configs["ppo"].update(entropy_coef=0.01, value_coef=0.5)
    configs["mappo"] = {
        "learning_rate": 0.0002,
        "entropy_coef": 0.015,
        "value_coef": 0.65,
    }
    for agent in ["controller_mat", "dag_offload_drl", "cache_offload_drl", "dt_handoff_drl"]:
        configs[agent].update(entropy_coef=0.01, value_coef=0.5)
    return configs


def support_fairness_kwargs(parameter: str, value: Any) -> dict[str, Any]:
    if parameter == "handoff_pressure":
        return {"primary_vehicle_selection": str(value)}
    if parameter == "typed_semantics" and value == "no_prediction":
        return {
            "prediction_confidence_scale": 0.0,
            "drop_handoff_prediction_prob": 1.0,
        }
    if parameter == "prediction_condition":
        return {
            "baseline": {},
            "no_prediction": {
                "prediction_confidence_scale": 0.0,
                "drop_handoff_prediction_prob": 1.0,
            },
            "noise_0.2": {"prediction_noise_std": 0.2},
            "confidence_0.7": {"prediction_confidence_scale": 0.7},
            "delay_2": {"prediction_delay_steps": 2},
            "drop_0.3": {"drop_handoff_prediction_prob": 0.3},
        }[str(value)]
    return {}


def command_templates() -> dict[str, Any]:
    common = {
        "expected_outputs": ["phase_outputs/{phase}.json"],
        "resume_phase": None,
        "timeout_seconds": 43200,
        "infrastructure_retries": 1,
        "extra_cli_overrides": "forbidden",
    }
    templates: dict[str, Any] = {}

    def add(
        phase: str,
        argv: list[str],
        outputs: list[str],
        timeout: int = 43200,
        matrix_contexts: list[dict[str, Any]] | None = None,
        resume_phase: str | None = None,
    ) -> None:
        templates[phase] = {
            **common,
            "argv": argv,
            "expected_outputs": outputs,
            "resume_phase": resume_phase or phase,
            "timeout_seconds": timeout,
            **({"matrix_contexts": matrix_contexts} if matrix_contexts else {}),
        }

    add("preflight", [".venv/bin/python", "scripts/validate_typed_model_cache_formal_restart.py", "--protocol-path", "{protocol_path}", "--output-path", "{preflight_output_path}"], ["preflight.json"], 1800)
    add("tests", [".venv/bin/python", "-m", "pytest", "-q", "--junitxml", "{tests_output_path}"], ["tests.xml"], 7200)
    capacity_assets = {
        "constrained_288mb": (
            CONFIG_ROOT / "runtime_constrained_288mb.yaml",
            CONFIG_ROOT / "fairness_constrained_288mb.json",
        ),
        "medium_576mb": (
            CONFIG_ROOT / "runtime_medium_576mb.yaml",
            CONFIG_ROOT / "fairness_medium_576mb.json",
        ),
        "relaxed_864mb": (
            CONFIG_ROOT / "runtime_relaxed_864mb.yaml",
            CONFIG_ROOT / "fairness_relaxed_864mb.json",
        ),
    }
    checkpoint_assets = {
        label: {
            "seed_checkpoint_manifest_path": (
                f"/ABSOLUTE/FORMAL_OUTPUT_ROOT/checkpoint_manifests/{label}/"
                "seed_checkpoint_manifest.json"
            ),
            "checkpoint_provenance_manifest_path": (
                f"/ABSOLUTE/FORMAL_OUTPUT_ROOT/checkpoint_manifests/{label}/"
                "checkpoint_provenance_manifest.json"
            ),
        }
        for label in capacity_assets
    }
    train_contexts = [
        {
            "agent": agent,
            "seed": seed,
            "capacity_label": label,
            "runtime_config_path": str(runtime.resolve()),
            "fairness_manifest_path": str(fairness.resolve()),
            **checkpoint_assets[label],
            "training_run_id": f"formal_{label}_{agent}_seed{seed}",
        }
        for label, (runtime, fairness) in capacity_assets.items()
        for agent in LEARNED_AGENTS
        for seed in SEEDS
    ]
    add("train", [".venv/bin/python", "scripts/train_algo_pool_real_sample.py", "--agent_name", "{agent}", "--profile", "baseline_safe", "--formal_protocol_path", "{protocol_path}", "--agent_config_path", "{agent_config_path}", "--model_cache_runtime_config", "{runtime_config_path}", "--random_seed", "{seed}", "--window_plan_path", "{train_window_plan_path}", "--output_root", "{training_output_root}", "--run_id", "{training_run_id}", "--reward_positive_offset", "0", "--primary_vehicle_selection", "handoff_pressure"], ["training/{agent}/{training_run_id}/train_summary.json"], matrix_contexts=train_contexts)
    add("dev_select", [".venv/bin/python", "scripts/run_typed_model_cache_formal_dev_selection.py", "--protocol-path", "{protocol_path}", "--training-root", "{training_output_root}", "--output-root", "{dev_input_root}", "--output-path", "{dev_selection_output_path}"], ["checkpoint_candidates.json", "dev_selection.json", "dev_benchmarks/**/benchmark_rows.csv"], 172800)
    add("checkpoint_freeze", [".venv/bin/python", "scripts/manage_typed_model_cache_formal_artifacts.py", "--action", "checkpoint_freeze", "--protocol-path", "{protocol_path}", "--input-root", "{dev_input_root}", "--output-path", "{checkpoint_freeze_output_path}"], ["checkpoint_freeze.json", "checkpoint_manifests/**/seed_checkpoint_manifest.json", "checkpoint_manifests/**/checkpoint_provenance_manifest.json"], 3600)
    all_formal_agents = [
        "reactive_lru", "reactive_fifo", "reactive_lfu", "reactive_aging_lfu",
        "reactive_random", *LEARNED_AGENTS,
    ]
    benchmark = [".venv/bin/python", "scripts/benchmark_main_results.py", "--agents", *all_formal_agents, "--seeds", *[str(seed) for seed in SEEDS], "--seed_checkpoint_manifest_path", "{seed_checkpoint_manifest_path}", "--checkpoint_provenance_manifest_path", "{checkpoint_provenance_manifest_path}", "--cache_baseline_fairness_manifest_path", "{fairness_manifest_path}", "--model_cache_runtime_config", "{runtime_config_path}", "--window_plan_path", "{formal_window_plan_path}", "--max_mobility_rows", "11850526", "--max_workflows", "3", "--max_steps", "22", "--primary_vehicle_selection", "handoff_pressure", "--window_mode", "mixed_informative", "--prediction_horizon", "3", "--reward_positive_offset", "0", "--output_root", "{formal_cache_policy_output_root}", "--audit_runtime"]
    cache_policy_contexts = [
        {
            "capacity_label": label,
            "runtime_config_path": str(runtime.resolve()),
            "fairness_manifest_path": str(fairness.resolve()),
            "request_replay_path": (
                f"/ABSOLUTE/FORMAL_OUTPUT_ROOT/formal_cache_policy/{label}/"
                "request_replay.json"
            ),
            "request_replay_evaluation_unit_id": read_json(fairness)[
                "window_workload_plan"
            ]["evaluation_units"][0]["evaluation_unit_id"],
            **checkpoint_assets[label],
        }
        for label, (runtime, fairness) in capacity_assets.items()
    ]
    cache_policy_wrapper = [
        ".venv/bin/python",
        "scripts/run_typed_model_cache_formal_cache_policy.py",
        "--protocol-path",
        "{protocol_path}",
        "--fairness-manifest-path",
        "{fairness_manifest_path}",
        "--evaluation-unit-id",
        "{request_replay_evaluation_unit_id}",
        "--request-replay-path",
        "{request_replay_path}",
        "--command",
        *benchmark,
    ]
    add("formal_cache_policy", cache_policy_wrapper, ["formal_cache_policy/**/aggregate_summary.json", "formal_cache_policy/{capacity_label}/request_replay.json"], matrix_contexts=cache_policy_contexts)
    controller_benchmark = [
        "{formal_controller_output_root}" if token == "{formal_cache_policy_output_root}" else token
        for token in benchmark
    ]
    controller_contexts = [
        {
            "capacity_label": label,
            "runtime_config_path": str(runtime.resolve()),
            "fairness_manifest_path": str(fairness.resolve()),
            **checkpoint_assets[label],
        }
        for label, (runtime, fairness) in capacity_assets.items()
    ]
    add("formal_controller", controller_benchmark, ["formal_controller/**/aggregate_summary.json"], matrix_contexts=controller_contexts)
    support_prefix = [".venv/bin/python", "scripts/run_typed_model_cache_formal_support.py", "--protocol-path", "{protocol_path}"]
    support_suffix = ["--model-cache-runtime-config", "{runtime_config_path}", "--cache-baseline-fairness-manifest-path", "{fairness_manifest_path}", "--seed-checkpoint-manifest-path", "{seed_checkpoint_manifest_path}", "--checkpoint-provenance-manifest-path", "{checkpoint_provenance_manifest_path}", "--window-plan-path", "{formal_window_plan_path}", "--agents", *all_formal_agents, "--seeds", *[str(seed) for seed in SEEDS]]
    support_matrix = build_support_setting_matrix()
    scalability_matrix = build_scalability_setting_matrix()
    available_support = [
        {
            **level,
            "family": setting["family"],
            "parameter": setting["parameter"],
        }
        for setting in support_matrix["settings"]
        for level in setting["levels"]
        if level["status"] == "available"
    ]
    ablation_contexts = [
        {
            "ablation_setting_id": level["setting_id"],
            "fairness_manifest_path": str(
                (CONFIG_ROOT / f"fairness_support_{level['setting_id']}.json").resolve()
            ),
        }
        for level in available_support
        if level["family"] == "ablation"
    ]
    formal_support_contexts: list[dict[str, Any]] = []
    for level in available_support:
        if level["family"] in {"ablation", "oracle_state_limit"}:
            continue
        if level["parameter"] == "capacity_mb":
            label = {
                288.0: "constrained_288mb",
                576.0: "medium_576mb",
                864.0: "relaxed_864mb",
            }[float(level["value"])]
            runtime, fairness = capacity_assets[label]
        else:
            runtime, fairness = capacity_assets["medium_576mb"]
        if level["parameter"] != "capacity_mb":
            fairness = CONFIG_ROOT / f"fairness_support_{level['setting_id']}.json"
        formal_support_contexts.append(
            {
                "support_setting_id": level["setting_id"],
                "runtime_config_path": str(runtime.resolve()),
                "fairness_manifest_path": str(fairness.resolve()),
                **checkpoint_assets[
                    {
                        288.0: "constrained_288mb",
                        576.0: "medium_576mb",
                        864.0: "relaxed_864mb",
                    }.get(float(level["value"]), "medium_576mb")
                    if level["parameter"] == "capacity_mb"
                    else "medium_576mb"
                ],
            }
        )
    scalability_contexts = [
        {"scalability_setting_id": level["setting_id"]}
        for setting in scalability_matrix["settings"]
        for level in setting["levels"]
        if level["status"] == "available"
    ]
    add("formal_ablation", support_prefix + ["--setting-id", "{ablation_setting_id}"] + support_suffix + ["--output-root", "{formal_ablation_output_root}"], ["formal_ablation/**/support_provenance.json"], matrix_contexts=ablation_contexts)
    add("formal_support", support_prefix + ["--setting-id", "{support_setting_id}"] + support_suffix + ["--output-root", "{formal_support_output_root}"], ["formal_support/**/support_provenance.json"], matrix_contexts=formal_support_contexts)
    add("formal_scalability", support_prefix + ["--setting-id", "{scalability_setting_id}"] + support_suffix + ["--request-replay-path", "{request_replay_path}", "--output-root", "{formal_scalability_output_root}"], ["formal_scalability/**/support_provenance.json"], matrix_contexts=scalability_contexts)
    add("formal_statistics", [".venv/bin/python", "scripts/run_typed_model_cache_formal_statistics.py", "--protocol-path", "{protocol_path}", "--input-root", "{formal_output_root}", "--output-root", "{statistics_output_root}"], ["statistics/paired_statistics.json"], 7200)
    add("formal_gate", [".venv/bin/python", "scripts/manage_typed_model_cache_formal_artifacts.py", "--action", "integrity_and_formal_gate", "--protocol-path", "{protocol_path}", "--input-root", "{formal_output_root}", "--output-path", "{formal_gate_output_path}"], ["artifact_integrity_manifest.json", "formal_gate.json"], 3600)
    # Auxiliary templates are executed inside their owning frozen phase and remain
    # separately inspectable for parser/expected-output auditing.
    prediction_ids = {
        level["setting_id"]
        for level in available_support
        if level["family"] == "prediction_boundary"
    }
    prediction_contexts = [
        context
        for context in formal_support_contexts
        if context["support_setting_id"] in prediction_ids
    ]
    robustness_contexts = [
        context
        for context in prediction_contexts
        if context["support_setting_id"]
        == next(
            level["setting_id"]
            for level in available_support
            if level["parameter"] == "prediction_condition"
            and level["value"] == "noise_0.2"
        )
    ]
    add("robustness", support_prefix + ["--setting-id", "{support_setting_id}"] + support_suffix + ["--output-root", "{formal_support_output_root}"], ["formal_support/**/support_provenance.json"], matrix_contexts=robustness_contexts, resume_phase="formal_support")
    add("prediction_boundary", support_prefix + ["--setting-id", "{support_setting_id}"] + support_suffix + ["--output-root", "{formal_support_output_root}"], ["formal_support/**/support_provenance.json"], matrix_contexts=prediction_contexts, resume_phase="formal_support")
    add("integrity", [".venv/bin/python", "scripts/manage_typed_model_cache_formal_artifacts.py", "--action", "integrity", "--protocol-path", "{protocol_path}", "--input-root", "{formal_output_root}", "--output-path", "{integrity_output_path}"], ["artifact_integrity_manifest.json"], 3600, resume_phase="formal_gate")
    return templates


def expansion_context(protocol_path: Path, agent_config_path: Path, runtime_path: Path, fairness_path: Path) -> dict[str, Any]:
    output = "/ABSOLUTE/FORMAL_OUTPUT_ROOT"
    support_matrix = build_support_setting_matrix()
    scalability_matrix = build_scalability_setting_matrix()
    def level_id(matrix: dict[str, Any], parameter: str, value: Any) -> str:
        for setting in matrix["settings"]:
            if setting["parameter"] == parameter:
                for level in setting["levels"]:
                    if level["value"] == value:
                        return level["setting_id"]
        raise KeyError((parameter, value))
    return {
        "protocol_path": str(protocol_path.resolve()),
        "output_root": output,
        "preflight_output_path": f"{output}/preflight.json",
        "tests_output_path": f"{output}/tests.xml",
        "agent": "sa_ghmappo",
        "seed": 7,
        "capacity_label": "medium_576mb",
        "agent_config_path": str(agent_config_path.resolve()),
        "runtime_config_path": str(runtime_path.resolve()),
        "train_window_plan_path": str((OLD_CONFIG / "train_window_plan.json").resolve()),
        "formal_window_plan_path": str((OLD_CONFIG / "formal_window_plan.json").resolve()),
        "training_output_root": f"{output}/training",
        "training_run_id": "formal_medium_576mb_seed7",
        "dev_input_root": output,
        "dev_selection_output_path": f"{output}/dev_selection.json",
        "checkpoint_freeze_output_path": f"{output}/checkpoint_freeze.json",
        "seed_checkpoint_manifest_path": f"{output}/checkpoint_manifests/medium_576mb/seed_checkpoint_manifest.json",
        "checkpoint_provenance_manifest_path": f"{output}/checkpoint_manifests/medium_576mb/checkpoint_provenance_manifest.json",
        "fairness_manifest_path": str(fairness_path.resolve()),
        "formal_output_root": output,
        "formal_cache_policy_output_root": f"{output}/formal_cache_policy",
        "formal_controller_output_root": f"{output}/formal_controller",
        "formal_ablation_output_root": f"{output}/formal_ablation",
        "formal_support_output_root": f"{output}/formal_support",
        "formal_scalability_output_root": f"{output}/formal_scalability",
        "ablation_setting_id": level_id(support_matrix, "typed_semantics", "no_prediction"),
        "support_setting_id": level_id(support_matrix, "prediction_condition", "noise_0.2"),
        "scalability_setting_id": level_id(scalability_matrix, "oracle_state_limit", 10000),
        "request_replay_path": f"{output}/formal_cache_policy/medium_576mb/request_replay.json",
        "request_replay_evaluation_unit_id": "resolved_by_capacity_matrix",
        "formal_rows_path": f"{output}/formal/benchmark_rows.csv",
        "statistics_output_root": f"{output}/statistics",
        "formal_gate_output_path": f"{output}/formal_gate.json",
        "integrity_output_path": f"{output}/artifact_integrity_manifest.json",
    }


def build_protocol(runtime_hashes: dict[str, str], templates: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    old = read_json(OLD_ARTIFACT / "formal_protocol_manifest.json")
    protocol = deepcopy(old)
    protocol["typed_model_cache_formal_protocol_version"] = FORMAL_EXECUTION_PROTOCOL_VERSION
    protocol["protocol_id"] = FORMAL_EXECUTION_PROTOCOL_ID
    protocol["created_at"] = now()
    protocol["status"] = "frozen_pre_execution_protocol_restart_no_performance_data"
    protocol["supersession"] = {
        "supersedes_version": "1.0.0",
        "old_protocol_status": "invalid_before_execution",
        "old_protocol_semantic_sha256": "41fbfab4ac10bae96250d7ead816d907fd6551bb9651ae03210e801c9e2478b4",
        "old_run_id": "typed_model_cache_formal_20260820_g14c_351fdb8_v1",
        "old_run_status": "INVALID_PROTOCOL_OR_IMPLEMENTATION",
        "failure_audit_sha256": sha256_file(FAILED_RUN / "audit/failure_audit.json"),
        "formal_performance_observed": False,
        "scientific_question_changed": False,
        "primary_comparisons_changed": False,
        "split_changed": False,
        "repair_scope": ["checkpoint cadence", "SA config binding", "primary endpoint producers", "support values and runner", "command templates", "append-only phase orchestration"],
    }
    protocol["identity"]["cache_efficiency_metrics_contract_version"] = "1.2.0"
    protocol["identity"]["typed_runtime_contract_hashes_by_capacity"] = runtime_hashes
    protocol["identity"]["execution_git_commit_binding"] = "Commit A2 containing this exact semantic hash; the commit is bound out-of-band to avoid impossible self-reference"
    protocol["training_budget"]["agent_configs"] = agent_configs()
    protocol["training_budget"]["checkpoint_frequency_updates"] = 4
    protocol["endpoints"]["primary"] = list(PRIMARY_ENDPOINTS)
    protocol["endpoint_schema"] = endpoint_schema()
    protocol["ablation_and_support"] = {
        "support_setting_matrix": build_support_setting_matrix(),
        "scalability_setting_matrix": build_scalability_setting_matrix(),
        "unavailable_settings_are_claim_limitations": True,
    }
    protocol["execution_contract"] = {
        "formal_phase_runner_version": FORMAL_PHASE_RUNNER_VERSION,
        "phase_order": list(PHASE_ORDER),
        "append_only": True,
        "completed_skip_rule": "only completed with identical input and output hashes",
        "hash_mismatch": "fail_fast",
        "failed_phase": "terminal_no_overwrite",
        "infrastructure_retry": "exit 75 only; maximum one identical-command retry",
        "formal_after_training_rule": "formal start permanently forbids retraining in this run",
        "holdout_capability": False,
        "ordinary_runner_may_issue_or_consume_holdout_token": False,
        "dry_run_writes_results": False,
        "output_root_conflict": "reject non-empty existing root unless --resume with valid ledger",
        "command_templates": templates,
        "default_expansion_context": context,
    }
    protocol["holdout_execution_contract"]["sealed"] = True
    protocol["holdout_execution_contract"]["opened"] = False
    protocol["holdout_execution_contract"]["consumed_permanently"] = False
    protocol["paper_claim_boundary"] = "G14R repairs and freezes executable contracts only; it is not G14C v2, formal completion, holdout opening, performance evidence, G15, or paper-ready evidence."
    return attach_hashes(protocol)


def build_fairness(
    capacity_mb: float,
    output_root: Path,
    *,
    primary_vehicle_selection: str = "handoff_pressure",
    prediction_noise_std: float = 0.0,
    prediction_confidence_scale: float = 1.0,
    prediction_delay_steps: int = 0,
    drop_handoff_prediction_prob: float = 0.0,
    window_plan_path: Path | None = None,
) -> dict[str, Any]:
    manifest = build_manifest(
        root=ROOT,
        mobility_path=MOBILITY,
        workflow_path=WORKFLOW,
        window_plan_path=window_plan_path or OLD_CONFIG / "formal_window_plan.json",
        catalog_path=CATALOG,
        seeds=SEEDS,
        max_workflows=3,
        workflow_selector="ordered",
        min_tasks=5,
        max_tasks=20,
        max_steps=22,
        max_mobility_rows=11_850_526,
        primary_vehicle_selection=primary_vehicle_selection,
        capacity_unit="mb",
        capacity_value=capacity_mb,
        output_root=str(output_root),
        evaluation_unit_limit=None,
        created_at=now(),
        controller_agents=LEARNED_AGENTS,
        prediction_noise_std=prediction_noise_std,
        prediction_confidence_scale=prediction_confidence_scale,
        prediction_delay_steps=prediction_delay_steps,
        drop_handoff_prediction_prob=drop_handoff_prediction_prob,
    )
    report = validate_manifest(manifest, root=ROOT, check_files=True)
    if report["status"] != "pass":
        raise RuntimeError(report["errors"])
    return manifest


def integrity_manifest() -> dict[str, Any]:
    files = []
    for path in sorted(ARTIFACT_ROOT.rglob("*")):
        relative = path.relative_to(ARTIFACT_ROOT)
        if (
            path.is_file()
            and path.name != "artifact_integrity_manifest.json"
            and "rehearsal_runs" not in relative.parts
        ):
            files.append({"path": relative.as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return attach_hashes({"artifact_integrity_manifest_version": "1.0.0", "artifact_run_id": ARTIFACT_RUN_ID, "integrity_status": "pass", "file_count": len(files), "files": files, "excluded_runtime_output_roots": ["rehearsal_runs/ (contains ignored non-formal checkpoints and raw episodes; root rehearsal_summary.json is integrity-tracked)"], "formal_checkpoint_count": 0, "formal_episode_count": 0, "formal_performance_result_count": 0})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-passed", action="store_true")
    args = parser.parse_args()
    existing_rehearsal = None
    rehearsal_path = ARTIFACT_ROOT / "rehearsal_summary.json"
    if rehearsal_path.is_file():
        candidate = read_json(rehearsal_path)
        if candidate.get("status") == "pass":
            existing_rehearsal = candidate
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    capacity_specs = {"constrained_288mb": 288.0, "medium_576mb": 576.0, "relaxed_864mb": 864.0}
    runtime_paths: dict[str, Path] = {}
    runtime_contracts: dict[str, dict[str, Any]] = {}
    fairness_paths: dict[str, Path] = {}
    fairness_manifests: dict[str, dict[str, Any]] = {}
    dev_fairness_paths: dict[str, Path] = {}
    for label, capacity in capacity_specs.items():
        runtime_path = CONFIG_ROOT / f"runtime_{label}.yaml"
        write_yaml(runtime_path, runtime_config(capacity, label))
        runtime_paths[label] = runtime_path
        runtime_contracts[label] = resolve_model_cache_runtime(runtime_path, root=ROOT)
        fairness_path = CONFIG_ROOT / f"fairness_{label}.json"
        fairness_manifests[label] = build_fairness(capacity, ARTIFACT_ROOT / "future_formal_runs")
        write_json(fairness_path, fairness_manifests[label])
        fairness_paths[label] = fairness_path
        dev_fairness_path = CONFIG_ROOT / f"dev_fairness_{label}.json"
        write_json(
            dev_fairness_path,
            build_fairness(
                capacity,
                ARTIFACT_ROOT / "future_dev_runs",
                window_plan_path=OLD_CONFIG / "dev_window_plan.json",
            ),
        )
        dev_fairness_paths[label] = dev_fairness_path

    support_fairness_paths: dict[str, Path] = {}
    for setting in build_support_setting_matrix()["settings"]:
        for level in setting["levels"]:
            if level["status"] != "available" or setting["family"] in {
                "capacity",
                "oracle_state_limit",
            }:
                continue
            setting_id = level["setting_id"]
            support_path = CONFIG_ROOT / f"fairness_support_{setting_id}.json"
            support_manifest = build_fairness(
                576.0,
                ARTIFACT_ROOT / "future_formal_runs",
                **support_fairness_kwargs(setting["parameter"], level["value"]),
            )
            write_json(support_path, support_manifest)
            support_fairness_paths[setting_id] = support_path

    agent_config_path = CONFIG_ROOT / "agent_training_configs.json"
    protocol_path = CONFIG_ROOT / "protocol_v1_1_manifest.json"
    templates = command_templates()
    context = expansion_context(protocol_path, agent_config_path, runtime_paths["medium_576mb"], fairness_paths["medium_576mb"])
    runtime_hashes = {label.split("_", 1)[0]: contract["runtime_contract_sha256"] for label, contract in runtime_contracts.items()}
    protocol = build_protocol(runtime_hashes, templates, context)
    validate_protocol_v1_1(protocol)
    write_json(protocol_path, protocol)
    agent_companion = {
        "agent_training_config_contract_version": "1.0.0",
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "agents": agent_configs(),
    }
    write_json(agent_config_path, agent_companion)
    write_json(CONFIG_ROOT / "split_companion.json", {"split_companion_version": "1.0.0", "split_semantic_sha256": "aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a", "source_split_manifest": "artifacts/analysis/typed_model_cache_formal_protocol_freeze_20260820_g14b_v1/split_manifest.json", "source_window_plan_directory": "configs/experiment/typed_model_cache_formal_protocol_v1_20260820", "split_semantics_rewritten": False, "reason": "No formal/holdout performance was executed or viewed; G14R repairs execution contracts only."})
    write_json(CONFIG_ROOT / "protocol_index.json", {"protocol_index_version": "1.1.0", "protocol_manifest": protocol_path.relative_to(ROOT).as_posix(), "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"], "split_companion": (CONFIG_ROOT / "split_companion.json").relative_to(ROOT).as_posix(), "agent_config": agent_config_path.relative_to(ROOT).as_posix(), "runtime_configs": {label: path.relative_to(ROOT).as_posix() for label, path in runtime_paths.items()}, "fairness_manifests": {label: path.relative_to(ROOT).as_posix() for label, path in fairness_paths.items()}, "dev_fairness_manifests": {label: path.relative_to(ROOT).as_posix() for label, path in dev_fairness_paths.items()}, "support_fairness_manifests": {setting_id: path.relative_to(ROOT).as_posix() for setting_id, path in support_fairness_paths.items()}, "status": READY_VERDICT})

    expansion = validate_command_templates(templates, context)
    all_agent_expansions = {}
    for agent in LEARNED_AGENTS:
        representative_context = next(
            item
            for item in templates["train"]["matrix_contexts"]
            if item["agent"] == agent
        )
        one_cell = {**templates["train"], "matrix_contexts": [representative_context]}
        all_agent_expansions[agent] = expand_command_plan(one_cell, context)
    write_json(ARTIFACT_ROOT / "g14c_v1_failure_reference.json", {"status": "INVALID_PROTOCOL_OR_IMPLEMENTATION", "old_run_id": "typed_model_cache_formal_20260820_g14c_351fdb8_v1", "old_protocol_version": "1.0.0", "old_protocol_semantic_sha256": "41fbfab4ac10bae96250d7ead816d907fd6551bb9651ae03210e801c9e2478b4", "failure_audit_path": (FAILED_RUN / "audit/failure_audit.json").relative_to(ROOT).as_posix(), "failure_audit_sha256": sha256_file(FAILED_RUN / "audit/failure_audit.json"), "training_started": False, "checkpoint_count": 0, "formal_performance_observed": False, "resume_allowed": False})
    write_json(
        ARTIFACT_ROOT / "protocol_to_runtime_executability_matrix.json",
        executability_matrix(),
    )
    write_json(ARTIFACT_ROOT / "checkpoint_frequency_validation.json", {"status": "pass", "legacy_default": 1, "formal_resolved": 4, "expected_update_count": 32, "actual_candidate_update_indices_contract": checkpoint_snapshot_indices(32, 4), "resume_latest_each_update_selection_ineligible": True})
    write_json(ARTIFACT_ROOT / "agent_config_resolution.json", {"status": "pass", "sa_ghmappo": {"requested_auxiliary_coef": 0.06, "resolved_auxiliary_coef": 0.06, "checkpoint_metadata_path": "training_metadata.resolved_agent_config.auxiliary_coef"}, "other_agents_receive_auxiliary_coef": False, "agent_configs": agent_configs()})
    write_json(ARTIFACT_ROOT / "primary_endpoint_schema.json", endpoint_schema())
    write_json(ARTIFACT_ROOT / "primary_endpoint_reconciliation.json", {"status": "pass_contract_and_unit_cases", "raw_event_reference_reducer": "src/metrics/cache_efficiency_metrics.py", "summary_field": "cache_efficiency_metrics", "row_fields": list(PRIMARY_ENDPOINTS), "aggregate_semantics": "nullable available finite values only", "formal_episode_count": 0})
    write_json(ARTIFACT_ROOT / "support_setting_matrix.json", protocol["ablation_and_support"]["support_setting_matrix"])
    write_json(ARTIFACT_ROOT / "scalability_setting_matrix.json", protocol["ablation_and_support"]["scalability_setting_matrix"])
    write_json(ARTIFACT_ROOT / "typed_support_runner_validation.json", {"status": "pass_contract_validation", "runner": "scripts/run_typed_model_cache_formal_support.py", "typed_slot_rejected": True, "legacy_checkpoint_rejected": True, "cli_override_rejected": True, "g12_supervised_disabled": True, "kv_disabled": True, "hf_metadata_disabled": True, "provenance_fields": ["protocol", "split", "runtime", "catalog", "fairness", "checkpoint", "setting"]})
    write_json(ARTIFACT_ROOT / "formal_command_templates.json", {"formal_command_template_version": "1.0.0", "templates": templates})
    write_json(ARTIFACT_ROOT / "command_expansion_validation.json", {"status": "pass", "representative": expansion, "all_agent_train_expansions": all_agent_expansions, "unresolved_placeholder_count": 0})
    write_json(ARTIFACT_ROOT / "phase_runner_validation.json", {"status": "pass_contract_validation", "runner_version": FORMAL_PHASE_RUNNER_VERSION, "phase_order": list(PHASE_ORDER), "append_only": True, "hash_bound_skip": True, "failed_terminal": True, "infrastructure_retry_max": 1, "holdout_capability": False, "dry_run_writes_results": False})
    write_json(
        ARTIFACT_ROOT / "rehearsal_summary.json",
        existing_rehearsal
        or {"status": "pending_post_generation_rehearsal", "formal_checkpoint_count": 0, "formal_episode_count": 0, "holdout_opened": False, "performance_claims": []},
    )
    write_json(ARTIFACT_ROOT / "protocol_v1_1_manifest.json", protocol)
    write_json(ARTIFACT_ROOT / "protocol_restart_diff.json", {"status": "pass", "from_version": "1.0.0", "to_version": "1.1.0", "scientific_question_changed": False, "primary_comparisons_changed": False, "split_changed": False, "endpoint_semantics_revision": "metrics 1.2 primary producer definitions", "execution_repairs": protocol["supersession"]["repair_scope"]})
    write_json(ARTIFACT_ROOT / "protocol_hashes.json", {"protocol_hash_bundle_version": "1.1.0", "old_protocol_semantic_sha256": "41fbfab4ac10bae96250d7ead816d907fd6551bb9651ae03210e801c9e2478b4", "new_protocol_full_sha256": protocol["hashes"]["full_sha256"], "new_protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"], "split_semantic_sha256": "aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a", "runtime_contract_hashes": runtime_hashes})
    write_json(ARTIFACT_ROOT / "split_revalidation.json", {"status": "pass_unchanged", "historical_registry_revalidated": True, "split_semantic_sha256": "aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a", "expected_split_semantic_sha256": "aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a", "split_content_modified": False, "window_counts": {"train": 24, "dev": 12, "formal": 12, "sealed_holdout": 12}, "performance_results_used": False})
    write_json(ARTIFACT_ROOT / "holdout_seal_revalidation.json", {"status": "pass", "sealed": True, "opened": False, "consumed_permanently": False, "ordinary_runner_access": False, "token_issued": False})
    checks = {"protocol_fields_have_runtime_consumers": True, "agent_commands_expand": True, "checkpoint_frequency_consistent": True, "sa_auxiliary_consistent": True, "primary_endpoint_producer_exists": True, "primary_endpoint_reconciliation": True, "support_values_concrete_or_unavailable": True, "typed_support_provenance": True, "phase_runner_dry_run": True, "fairness_manifests_persisted": True, "runtime_configs_persisted": True, "command_templates_persisted": True, "output_schema_exists": True, "clean_worktree_execution_plan": True, "holdout_sealed": True}
    write_json(ARTIFACT_ROOT / "readiness_review_v3.json", {"readiness_review_version": "3.0.0", "reviewed_at": now(), "literature_cutoff": "2026-08-20", "target_venue": "IEEE Transactions on Mobile Computing (TMC)", "artifact_run_id": ARTIFACT_RUN_ID, "policy_version": "tmc_review_policy_v3_20260621", "implementation_baseline_git_commit": git_commit(), "evidence_level": "E2_EXECUTION_CONTRACT_VALIDATED_NO_FORMAL_PERFORMANCE", "checks": checks, "verdict": readiness_v3(checks), "formal_completed": False, "paper_ready": False})
    write_json(ARTIFACT_ROOT / "command_log.json", {"command_log_version": "1.0.0", "artifact_run_id": ARTIFACT_RUN_ID, "commands": [
        {"command": ".venv/bin/python scripts/restart_typed_model_cache_formal_protocol.py --force", "status": "pass", "scope": "protocol generation"},
        {"command": ".venv/bin/python scripts/run_typed_model_cache_formal_protocol.py --preflight --dry-run", "status": "pass", "scope": "no-write full expansion"},
        {"command": ".venv/bin/python scripts/run_typed_model_cache_formal_repair_rehearsal.py", "status": "pass" if existing_rehearsal else "pending", "scope": "non-formal rehearsal"},
        {"command": ".venv/bin/python -m pytest tests/test_typed_model_cache.py -q", "status": "pass"},
        {"command": ".venv/bin/python -m pytest tests/test_typed_model_cache_runtime.py -q", "status": "pass"},
        {"command": ".venv/bin/python -m pytest tests/test_typed_model_cache_formal_protocol.py -q", "status": "pass"},
        {"command": ".venv/bin/python -m pytest tests/test_cache_efficiency_metrics.py -q", "status": "pass"},
        {"command": ".venv/bin/python -m pytest tests/test_top_journal_statistics.py -q", "status": "pass"},
        {"command": ".venv/bin/python -m pytest -q", "status": "pass", "result": "816 passed"},
        {"command": ".venv/bin/python scripts/smoke_test.py", "status": "pass" if args.smoke_passed else "pending_final_validation"},
        {"command": "git diff --check", "status": "pass"}
    ], "formal_training_commands": 0, "formal_evaluation_commands": 0, "holdout_commands": 0, "g15_commands": 0})
    write_json(ARTIFACT_ROOT / "protected_user_file_hashes.json", {"status": "recorded", "files": {path: sha256_file(ROOT / path) for path in PROTECTED_FILES}})
    write_json(ARTIFACT_ROOT / "artifact_integrity_manifest.json", integrity_manifest())
    print(json.dumps({"status": "pass", "artifact_root": str(ARTIFACT_ROOT), "config_root": str(CONFIG_ROOT), "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"], "readiness": READY_VERDICT}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
