#!/usr/bin/env python3
"""Freeze G14B historical exclusion, independent splits, and formal protocol.

This command is create-only and result blind.  It does not train an agent, load
a checkpoint, execute an episode, run formal/holdout/hidden, or perform G15.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.registry import build_agent
from src.evaluators.cache_baseline_fairness import build_manifest, validate_manifest
from src.evaluators.main_results_support import build_selected_workflow_states
from src.evaluators.typed_model_cache_formal_protocol import (
    READINESS_REVIEW_VERSION,
    FormalProtocolError,
    HoldoutAccessError,
    InsufficientWindowError,
    append_holdout_execution_record,
    assert_no_semantic_cli_overrides,
    attach_hashes,
    build_agent_matrix,
    build_candidate_inventory,
    build_capacity_strata,
    build_claim_evidence_template,
    build_formal_protocol,
    build_historical_registry,
    build_holdout_seal,
    build_split_manifest,
    build_statistics_protocol,
    canonical_sha256,
    interval_relation,
    protected_file_hashes,
    protocol_hash_changes_on_mutation,
    readiness_verdict,
    scan_ngsim_intervals,
    semantic_projection,
    sha256_file,
    utc_now,
    validate_protocol_manifest,
    validate_split_access,
    write_json,
)
from src.runtime.typed_model_cache_runtime import resolve_model_cache_runtime


RUN_ID = "typed_model_cache_formal_protocol_freeze_20260820_g14b_v1"
CONFIG_DIR = ROOT / "configs" / "experiment" / "typed_model_cache_formal_protocol_v1_20260820"
DEFAULT_OUTPUT = ROOT / "artifacts" / "analysis" / RUN_ID
G14C_OUTPUT_ROOT = ROOT / "artifacts" / "experiments" / "typed_model_cache_g14c_formal_v1"
MOBILITY_PATH = ROOT / "data" / "raw" / "mobility" / "ngsim" / "Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv"
WORKFLOW_PATH = ROOT / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"
CATALOG_PATH = ROOT / "src" / "data" / "model_catalog" / "typed_model_cache_controlled.json"
BASE_RUNTIME_CONFIG = ROOT / "configs" / "benchmark" / "typed_model_cache_controlled_lru.yaml"
G14A_ROOT = ROOT / "artifacts" / "analysis" / "typed_model_cache_runtime_plumbing_validation_20260819_g14a_v1"
G14A_REPORT = G14A_ROOT / "rehearsal_run_manifest.json"
G14A_INTEGRITY = G14A_ROOT / "artifact_integrity_manifest.json"
PROTECTED_FILES = [
    "scripts/train_sa_ghmappo_real_sample.py",
    "src/agents/sa_ghmappo_agent.py",
    "src/agents/sa_ghmappo_core.py",
    "src/encoders/fusion_encoder.py",
    "src/evaluators/real_eval_support.py",
    "tests/test_algo_pool_contract.py",
    "tests/test_checkpoint_compat.py",
]
SPLIT_COUNTS = {"train": 24, "dev": 12, "formal": 12, "sealed_holdout": 12}
WINDOW_LENGTH = 24
MINIMUM_GAP_FRAMES = 24
MINIMUM_VEHICLE_COUNT = 2
SPLIT_SEED = 1401
PREFIX_ROWS = 5_000_000
SCAN_CACHE = Path("/tmp/ppo_mec_g14b_ngsim_interval_cache.pkl")


def parse_args() -> argparse.Namespace:
    assert_no_semantic_cli_overrides(sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--created_at", default="")
    parser.add_argument(
        "--git_commit_binding",
        default="Commit A containing the exact frozen protocol semantic hash",
    )
    return parser.parse_args()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def current_git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_status() -> dict[str, Any]:
    short = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines()
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True).splitlines()
    return {"short": short, "staged": staged}


def resolved_runtimes(capacity_values: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    raw = yaml.safe_load(BASE_RUNTIME_CONFIG.read_text(encoding="utf-8-sig"))
    contracts: dict[str, dict[str, Any]] = {}
    for item in capacity_values:
        config = deepcopy(raw)
        config["cache_capacity_profile"]["capacity_mb"] = float(item["capacity_mb"])
        config["profile_id"] = f"typed_model_cache_formal_{item['stratum']}_v1"
        config["claim_boundary"] = "G14B pre-run frozen protocol; no formal outcome"
        contracts[item["stratum"]] = resolve_model_cache_runtime(config, root=ROOT)
    return contracts["constrained"], contracts


def plan_payload(split: str, windows: list[dict[str, Any]], split_hash: str) -> dict[str, Any]:
    return {
        "protocol_version": "typed_model_cache_split_protocol_v1.0.0",
        "split": split,
        "sealed": split == "sealed_holdout",
        "outcome_blind_selection": True,
        "split_manifest_semantic_sha256": split_hash,
        "selected_window_plan": windows,
    }


def run_help_preflight(command: list[str]) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/ppo_mec_g14b_pycache"
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "return_code": result.returncode,
        "passed": result.returncode == 0,
        "help_output_detected": "usage:" in result.stdout.lower(),
        "stderr": result.stderr[-1000:],
    }


def build_agent_preflight() -> dict[str, Any]:
    matrix = build_agent_matrix()
    names = [item["agent"] for item in matrix["controller_table"]]
    names.extend(item["agent"] for item in matrix["reactive_cache_policy_isolation"])
    rows = []
    for name in names:
        try:
            agent = build_agent(name, random_seed=7)
            rows.append(
                {
                    "agent": name,
                    "status": "pass",
                    "class": type(agent).__name__,
                    "formal_episode_executed": False,
                }
            )
        except Exception as exc:  # pragma: no cover - recorded for the real preflight.
            rows.append({"agent": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}"})
    for item in matrix["exact_oracle_cells"]:
        rows.append(
            {
                "agent": item["agent"],
                "status": "pass_contract_only",
                "exact_only": True,
                "formal_episode_executed": False,
            }
        )
    return {
        "agent_compatibility_audit_version": "1.0.0",
        "passed": all(item["status"].startswith("pass") for item in rows),
        "rows": rows,
        "paper_grade_controller_boundary": matrix["controller_level_boundary"],
    }


def g14a_integrity_check() -> dict[str, Any]:
    required = [
        G14A_ROOT / "producer_consumer_matrix.json",
        G14A_ROOT / "training_entrypoint_audit.json",
        G14A_ROOT / "checkpoint_provenance_validation.json",
        G14A_ROOT / "benchmark_runtime_audit.json",
        G14A_ROOT / "metrics_reconciliation.json",
        G14A_INTEGRITY,
    ]
    rows = []
    for path in required:
        rows.append({"path": relative(path), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None})
    return {"passed": all(item["exists"] for item in rows), "rows": rows}


def fairness_preflight(
    config_stage: Path,
    capacity_strata: dict[str, Any],
    controller_agents: list[str],
    created_at: str,
    runner_max_mobility_rows: int,
) -> dict[str, Any]:
    rows = []
    plan_path = config_stage / "formal_window_plan.json"
    for item in capacity_strata["strata"]:
        try:
            manifest = build_manifest(
                root=ROOT,
                mobility_path=MOBILITY_PATH,
                workflow_path=WORKFLOW_PATH,
                window_plan_path=plan_path,
                catalog_path=CATALOG_PATH,
                seeds=[7, 13, 29, 43, 71],
                max_workflows=3,
                workflow_selector="ordered",
                min_tasks=5,
                max_tasks=20,
                max_steps=22,
                max_mobility_rows=runner_max_mobility_rows,
                primary_vehicle_selection="handoff_pressure",
                capacity_unit="mb",
                capacity_value=float(item["capacity_mb"]),
                output_root=relative(G14C_OUTPUT_ROOT / item["stratum"]),
                evaluation_unit_limit=1,
                created_at=created_at,
                controller_agents=controller_agents,
            )
            validation = validate_manifest(manifest, root=ROOT, check_files=True)
            rows.append(
                {
                    "stratum": item["stratum"],
                    "capacity_mb": item["capacity_mb"],
                    "status": "pass" if validation["status"] == "pass" else "fail",
                    "manifest_semantic_sha256": manifest["hashes"]["semantic_protocol_sha256"],
                    "evaluation_unit_count": len(manifest["window_workload_plan"]["evaluation_units"]),
                }
            )
        except Exception as exc:  # pragma: no cover - recorded by the real run.
            rows.append(
                {
                    "stratum": item["stratum"],
                    "capacity_mb": item["capacity_mb"],
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {"passed": all(item["status"] == "pass" for item in rows), "rows": rows}


def synthetic_negative_preflight(protocol: dict[str, Any], split_manifest: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    sample = deepcopy(split_manifest["splits"]["formal"]["selected_window_plan"][0])
    overlap = deepcopy(sample)
    overlap["window_id"] = "different_id_same_interval"
    relation = interval_relation(sample, overlap, minimum_gap_frames=MINIMUM_GAP_FRAMES)
    rows.append({"case": "same_interval_different_id", "passed": relation["classification"] == "exact_overlap"})

    near = deepcopy(sample)
    shift = WINDOW_LENGTH + MINIMUM_GAP_FRAMES - 1
    sampling = int(sample["sampling_interval"])
    for start_key, end_key in (("raw_frame_start", "raw_frame_end"), ("segment_frame_start", "segment_frame_end")):
        near[start_key] = int(near[start_key]) + shift
        near[end_key] = int(near[end_key]) + shift
    for start_key, end_key in (("raw_time_start", "raw_time_end"), ("time_index_start", "time_index_end")):
        near[start_key] = int(near[start_key]) + shift * sampling
        near[end_key] = int(near[end_key]) + shift * sampling
    relation = interval_relation(sample, near, minimum_gap_frames=MINIMUM_GAP_FRAMES)
    rows.append({"case": "insufficient_gap", "passed": relation["classification"] == "insufficient_gap"})

    try:
        build_split_manifest(
            {**split_manifest, "candidates": split_manifest["splits"]["formal"]["selected_window_plan"][:11], "parameters": {"split_generation_seed": 1, "tie_break": "x", "window_length": 24, "minimum_vehicle_count": 1, "runner_prefix_max_mobility_rows": 1}, "hashes": {"semantic_sha256": "x"}},
            counts={"train": 0, "dev": 0, "formal": 11, "sealed_holdout": 12},
            minimum_gap_frames=MINIMUM_GAP_FRAMES,
        )
        insufficient_rejected = False
    except InsufficientWindowError:
        insufficient_rejected = True
    rows.append({"case": "formal_below_12", "passed": insufficient_rejected})

    try:
        assert_no_semantic_cli_overrides(["--seeds", "7", "13"])
        override_rejected = False
    except FormalProtocolError:
        override_rejected = True
    rows.append({"case": "semantic_cli_override", "passed": override_rejected})

    try:
        validate_split_access("sealed_holdout", caller_role="benchmark_runner")
        ordinary_rejected = False
    except HoldoutAccessError:
        ordinary_rejected = True
    rows.append({"case": "ordinary_runner_holdout_open", "passed": ordinary_rejected})

    rows.append(
        {
            "case": "semantic_mutation_changes_hash",
            "passed": protocol_hash_changes_on_mutation(protocol, "seed_plan.seeds.0", 5),
        }
    )
    changed_time = deepcopy(protocol)
    changed_time["created_at"] = "2099-01-01T00:00:00+00:00"
    rows.append(
        {
            "case": "created_at_semantic_stability",
            "passed": canonical_sha256(semantic_projection(protocol)) == canonical_sha256(semantic_projection(changed_time)),
        }
    )
    return {"passed": all(item["passed"] for item in rows), "rows": rows}


def write_config_stage(
    config_stage: Path,
    split_manifest: dict[str, Any],
    allocation: dict[str, list[dict[str, Any]]],
    protocol: dict[str, Any],
    artifact_path: Path,
) -> None:
    config_stage.mkdir(parents=True, exist_ok=False)
    for split in ("train", "dev", "formal", "sealed_holdout"):
        write_json(
            config_stage / f"{split}_window_plan.json",
            plan_payload(split, allocation[split], split_manifest["hashes"]["semantic_sha256"]),
        )
    write_json(
        config_stage / "protocol_index.json",
        {
            "typed_model_cache_formal_protocol_version": "1.0.0",
            "status": "frozen_pre_training_no_performance_data",
            "artifact_run_id": RUN_ID,
            "artifact_path": relative(artifact_path),
            "formal_protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
            "split_semantic_sha256": split_manifest["hashes"]["semantic_sha256"],
            "execution_commit_binding": "Commit A containing this exact index and protocol hash",
            "holdout_sealed": True,
            "performance_results_present": False,
        },
    )


def artifact_integrity(stage: Path, config_stage: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(stage.rglob("*.json")):
        if path.name == "artifact_integrity_manifest.json":
            continue
        rows.append({"scope": "artifact", "path": path.relative_to(stage).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    for path in sorted(config_stage.rglob("*.json")):
        rows.append({"scope": "config", "path": path.relative_to(config_stage).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return attach_hashes(
        {
            "artifact_integrity_manifest_version": "1.0.0",
            "artifact_run_id": RUN_ID,
            "file_count": len(rows),
            "files": rows,
            "checkpoint_file_count": 0,
            "performance_result_file_count": 0,
        }
    )


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"create-only G14B artifact already exists: {output}")
    if CONFIG_DIR.exists():
        raise FileExistsError(f"create-only G14B config already exists: {CONFIG_DIR}")
    created_at = args.created_at or utc_now()
    protected_before = protected_file_hashes(ROOT, PROTECTED_FILES)
    initial_git = git_status()
    if initial_git["staged"]:
        raise FormalProtocolError("staging area must be empty before G14B freeze")

    with tempfile.TemporaryDirectory(prefix=".ppo_mec_g14b_", dir=ROOT) as temporary:
        temporary_root = Path(temporary)
        stage = temporary_root / "artifact"
        config_stage = temporary_root / "config"
        stage.mkdir(parents=True)

        cache_identity = {
            "path": MOBILITY_PATH.as_posix(),
            "size_bytes": MOBILITY_PATH.stat().st_size,
            "mtime_ns": MOBILITY_PATH.stat().st_mtime_ns,
            "prefix_rows": PREFIX_ROWS,
            "scanner_version": "g14b_v1_compact_prefix_cutoff_ranges",
        }
        cached = None
        if SCAN_CACHE.is_file():
            try:
                with SCAN_CACHE.open("rb") as handle:
                    candidate_cache = pickle.load(handle)
                if candidate_cache.get("identity") == cache_identity:
                    cached = candidate_cache
            except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
                cached = None
        if cached is None:
            inventory, internal = scan_ngsim_intervals(
                MOBILITY_PATH,
                prefix_rows=PREFIX_ROWS,
            )
            with SCAN_CACHE.open("wb") as handle:
                pickle.dump(
                    {"identity": cache_identity, "inventory": inventory, "internal": internal},
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        else:
            inventory, internal = cached["inventory"], cached["internal"]
        registry, registry_validation = build_historical_registry(
            ROOT,
            inventory_internal=internal,
            mobility_sha256=internal["dataset_sha256"],
        )
        candidate_inventory, exclusion_audit = build_candidate_inventory(
            inventory,
            internal,
            registry,
            window_length=WINDOW_LENGTH,
            minimum_gap_frames=MINIMUM_GAP_FRAMES,
            minimum_vehicle_count=MINIMUM_VEHICLE_COUNT,
            split_seed=SPLIT_SEED,
            allowed_source_segments=["i_80"],
            use_full_runner_scope=True,
        )
        split_manifest, split_audit, allocation_bundle = build_split_manifest(
            candidate_inventory,
            counts=SPLIT_COUNTS,
            minimum_gap_frames=MINIMUM_GAP_FRAMES,
            created_at=created_at,
        )
        matrix = allocation_bundle.pop("_pairwise_matrix")
        allocation = allocation_bundle

        base_runtime_raw = yaml.safe_load(BASE_RUNTIME_CONFIG.read_text(encoding="utf-8-sig"))
        base_runtime = resolve_model_cache_runtime(base_runtime_raw, root=ROOT)
        capacity_strata = build_capacity_strata(base_runtime)
        _, runtime_contracts = resolved_runtimes(capacity_strata["strata"])
        runtime_hashes = {
            name: contract["runtime_contract_sha256"] for name, contract in runtime_contracts.items()
        }
        workflow_states = build_selected_workflow_states(
            workflow_csv_path=WORKFLOW_PATH,
            max_workflows=3,
            workflow_selector="ordered",
            min_tasks=5,
            max_tasks=20,
            random_seed=7,
        )
        workflow_ids = [item.workflow_id for item in workflow_states]
        holdout_seal = build_holdout_seal(split_manifest)
        protocol = build_formal_protocol(
            split_manifest=split_manifest,
            historical_registry=registry,
            runtime_contract=runtime_contracts["medium"],
            runtime_hashes_by_capacity=runtime_hashes,
            capacity_strata=capacity_strata,
            dataset_hashes={
                "ngsim_sha256": internal["dataset_sha256"],
                "alibaba_batch_task_sha256": sha256_file(WORKFLOW_PATH),
                "typed_catalog_file_sha256": sha256_file(CATALOG_PATH),
            },
            workflow_ids=workflow_ids,
            fairness_manifest_version="1.1.0",
            holdout_seal=holdout_seal,
            created_at=created_at,
        )
        validate_protocol_manifest(protocol)
        write_config_stage(config_stage, split_manifest, allocation, protocol, output)

        agent_audit = attach_hashes({**build_agent_preflight(), "created_at": created_at})
        controller_agents = [
            item["agent"]
            for item in protocol["agent_matrix"]["controller_table"]
            if item["training_requirement"] != "checkpoint_free"
        ]
        fairness = fairness_preflight(
            config_stage,
            capacity_strata,
            controller_agents,
            created_at,
            int(split_manifest["runner_prefix_max_mobility_rows"]),
        )
        cli_preflights = [
            run_help_preflight([str(ROOT / ".venv" / "bin" / "python"), "scripts/train_sa_ghmappo_real_sample.py", "--help"]),
            run_help_preflight([str(ROOT / ".venv" / "bin" / "python"), "scripts/train_algo_pool_real_sample.py", "--help"]),
            run_help_preflight([str(ROOT / ".venv" / "bin" / "python"), "scripts/benchmark_main_results.py", "--help"]),
            run_help_preflight([str(ROOT / ".venv" / "bin" / "python"), "scripts/analyze_top_journal_statistics.py", "--help"]),
        ]
        negative = synthetic_negative_preflight(protocol, split_manifest)
        g14a = g14a_integrity_check()
        protected_after = protected_file_hashes(ROOT, PROTECTED_FILES)
        protected_unchanged = protected_before == protected_after
        disk = shutil.disk_usage(ROOT)
        output_root_absent = not G14C_OUTPUT_ROOT.exists()
        clean_worktree_plan = {
            "status": "feasible_not_created",
            "candidate_execution_commit_binding": args.git_commit_binding,
            "worktree_root_plan": "/tmp/ppo_mec_g14c_<commit>/",
            "disk_free_bytes": disk.free,
            "minimum_required_free_bytes": 20 * 1024**3,
            "disk_sufficient": disk.free >= 20 * 1024**3,
            "current_worktree_need_not_be_clean_because_user_seven_files_are_preserved": True,
        }
        checks = {
            "g14a_runtime": g14a["passed"],
            "data_hash_and_inventory": bool(inventory["hashes"]["semantic_sha256"]),
            "historical_exclusion": registry_validation["passed"] and exclusion_audit["passed"],
            "split_independence": split_audit["passed"],
            "formal_outer_count": split_audit["outer_cluster_counts"]["formal"] >= 12,
            "sealed_holdout_outer_count": split_audit["outer_cluster_counts"]["sealed_holdout"] >= 12,
            "seed_and_budget_frozen": protocol["seed_plan"]["seeds"] == [7, 13, 29, 43, 71],
            "agent_compatibility": agent_audit["passed"],
            "training_and_benchmark_entrypoints_parse": all(item["passed"] for item in cli_preflights),
            "typed_checkpoint_requirement_frozen": True,
            "fairness_manifest_each_capacity": fairness["passed"],
            "cache_event_and_metrics_contract": protocol["identity"]["cache_event_schema_version"] == "1.3.0" and protocol["identity"]["cache_efficiency_metrics_contract_version"] == "1.1.0",
            "holdout_sealed_unopened": holdout_seal["sealed"] and not holdout_seal["opened"],
            "negative_preflight": negative["passed"],
            "clean_worktree_plan": clean_worktree_plan["disk_sufficient"],
            "g14c_output_root_absent": output_root_absent,
            "user_seven_files_unchanged": protected_unchanged,
        }
        verdict = readiness_verdict(checks)
        readiness = attach_hashes(
            {
                "readiness_review_version": READINESS_REVIEW_VERSION,
                "reviewed_at": created_at,
                "literature_cutoff": "2026-08-20",
                "target_venue": "IEEE Transactions on Mobile Computing",
                "artifact_run_id": RUN_ID,
                "policy_version": "tmc_review_policy_v3_20260621",
                "git_commit": current_git_commit(),
                "candidate_execution_commit_binding": args.git_commit_binding,
                "evidence_level": "E2_PROTOCOL_AND_CONTRACT_VALIDATED_NO_PERFORMANCE_DATA",
                "checks": checks,
                "unresolved_blockers": [name for name, passed in checks.items() if not passed],
                "verdict": verdict,
                "paper_ready": False,
                "formal_completed": False,
                "holdout_opened": False,
                "typed_checkpoint_count": 0,
                "clean_worktree_plan": clean_worktree_plan,
                "initial_git_status": initial_git,
                "protected_user_file_hashes": protected_after,
                "claim_boundary": "Readiness authorizes only a future clean-worktree G14C execution; it is not performance or paper-ready evidence.",
            }
        )
        preflight = attach_hashes(
            {
                "preflight_report_version": "1.0.0",
                "created_at": created_at,
                "passed": verdict == "READY_FOR_G14C_CLEAN_TRAIN_AND_FORMAL",
                "protocol_repeat_build_semantic_hash_stable": protocol["hashes"]["semantic_sha256"] == validate_protocol_manifest(protocol)["semantic_sha256"],
                "semantic_negative_cases": negative,
                "agent_compatibility": agent_audit,
                "capacity_runtime_contracts": {
                    name: {
                        "capacity_mb": contract["cache_capacity_profile"]["capacity_mb"],
                        "runtime_contract_sha256": contract["runtime_contract_sha256"],
                    }
                    for name, contract in runtime_contracts.items()
                },
                "fairness_manifest_preflight": fairness,
                "entrypoint_parse_preflight": cli_preflights,
                "output_root_absent": output_root_absent,
                "clean_worktree_plan": clean_worktree_plan,
                "formal_episode_count": 0,
                "checkpoint_count": 0,
            }
        )

        available_summary = deepcopy(inventory)
        available_summary["duration_accounting"] = {
            **exclusion_audit["duration_accounting"],
            "conservatively_excluded_unknown_segment_count": len(exclusion_audit["conservative_unknown_scope_segments"]),
            "remaining_eligible_non_overlapping_window_count_in_frozen_runner_scope": candidate_inventory["summary"]["eligible_non_overlapping_candidate_count"],
        }
        protocol_hashes = attach_hashes(
            {
                "protocol_hash_bundle_version": "1.0.0",
                "formal_protocol_full_sha256": protocol["hashes"]["full_sha256"],
                "formal_protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
                "split_full_sha256": split_manifest["hashes"]["full_sha256"],
                "split_semantic_sha256": split_manifest["hashes"]["semantic_sha256"],
                "historical_registry_full_sha256": registry["hashes"]["full_sha256"],
                "historical_registry_semantic_sha256": registry["hashes"]["semantic_sha256"],
                "catalog_file_sha256": sha256_file(CATALOG_PATH),
                "catalog_fingerprint": runtime_contracts["medium"]["typed_catalog_fingerprint"],
                "runtime_contract_hashes_by_capacity": runtime_hashes,
            }
        )
        command_log = attach_hashes(
            {
                "command_log_version": "1.0.0",
                "created_at": created_at,
                "commands": [[sys.executable, *sys.argv], *[item["command"] for item in cli_preflights]],
                "prohibited_work_not_executed": [
                    "formal agent training",
                    "formal benchmark",
                    "sealed holdout",
                    "hidden evaluation",
                    "G15",
                    "algorithm/reward/action/network modification",
                    "performance-driven split selection",
                ],
            }
        )

        outputs = {
            "historical_window_usage_registry.json": registry,
            "historical_registry_validation.json": registry_validation,
            "available_interval_inventory.json": available_summary,
            "historical_exclusion_audit.json": exclusion_audit,
            "candidate_window_inventory.json": candidate_inventory,
            "split_manifest.json": split_manifest,
            "split_overlap_matrix.json": matrix,
            "split_independence_audit.json": split_audit,
            "formal_protocol_manifest.json": protocol,
            "protocol_hashes.json": protocol_hashes,
            "agent_compatibility_audit.json": agent_audit,
            "capacity_strata.json": capacity_strata,
            "statistics_protocol.json": attach_hashes({**build_statistics_protocol(), "created_at": created_at}),
            "claim_evidence_template.json": attach_hashes({**build_claim_evidence_template(), "created_at": created_at}),
            "holdout_seal_record.json": holdout_seal,
            "readiness_review_v2.json": readiness,
            "preflight_report.json": preflight,
            "command_log.json": command_log,
        }
        for name, payload in outputs.items():
            write_json(stage / name, payload)
        write_json(stage / "artifact_integrity_manifest.json", artifact_integrity(stage, config_stage))

        if readiness["verdict"] != "READY_FOR_G14C_CLEAN_TRAIN_AND_FORMAL":
            raise FormalProtocolError(
                "G14B readiness blocked: "
                f"{readiness['unresolved_blockers']}; fairness={fairness['rows']}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, output)
        shutil.copytree(config_stage, CONFIG_DIR)

    print(json.dumps({
        "artifact_run_id": RUN_ID,
        "artifact_path": relative(output),
        "config_path": relative(CONFIG_DIR),
        "historical_registry_semantic_sha256": registry["hashes"]["semantic_sha256"],
        "split_semantic_sha256": split_manifest["hashes"]["semantic_sha256"],
        "formal_protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "outer_window_counts": split_audit["outer_cluster_counts"],
        "readiness_verdict": readiness["verdict"],
        "formal_episode_count": 0,
        "holdout_opened": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
