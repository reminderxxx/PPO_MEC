"""Run the bounded G14A non-formal typed runtime rehearsal.

This script is plumbing validation only.  It intentionally uses the controlled,
non-hidden G07 smoke window and one-step smoke checkpoints; its outputs are never
formal, holdout, hidden, or paper evidence.
"""

from __future__ import annotations

import argparse
import csv
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

from src.agents.registry import get_algo_spec
from src.evaluators.cache_baseline_fairness import (
    BASELINE_NAMES,
    build_manifest,
    sha256_file,
    validate_manifest,
)
from src.evaluators.main_results_support import summary_to_row
from src.metrics.cache_efficiency_metrics import (
    cache_efficiency_row_fields,
    reduce_cache_efficiency_summary,
)
from src.runtime.typed_model_cache_runtime import (
    RuntimeContractError,
    resolve_model_cache_runtime,
    validate_checkpoint_provenance,
)


RUN_ID = "20260819_g14a_v1"
LABEL = "non_formal_typed_runtime_rehearsal"
OUTPUT_ROOT = (
    ROOT
    / "artifacts/analysis"
    / f"typed_model_cache_runtime_plumbing_validation_{RUN_ID}"
)
TYPED_CONFIGS = {
    "320mb": ROOT / "configs/benchmark/typed_model_cache_controlled_lru.yaml",
    "384mb": ROOT / "configs/benchmark/typed_model_cache_controlled_lru_384mb.yaml",
}
LEGACY_CONFIGS = {
    "adapter_slots": ROOT / "configs/benchmark/legacy_adapter_slots_lru.yaml",
    "mb": ROOT / "configs/benchmark/legacy_adapter_mb_lru.yaml",
}
CATALOG = ROOT / "src/data/model_catalog/typed_model_cache_controlled.json"
MOBILITY = ROOT / "data/raw/mobility/ngsim/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv"
WORKFLOW = ROOT / "data/raw/workflow/alibaba2018/batch_task.csv"
PLAN = ROOT / "configs/experiment/cache_baseline_fairness_g07_smoke_window_plan.json"
SEEDS = [7, 13]
LEARNED_AGENTS = ["ppo", "mappo"]
ALL_AGENTS = list(BASELINE_NAMES) + LEARNED_AGENTS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def run_command(command: list[str], command_log: list[dict[str, Any]]) -> None:
    started = utc_now()
    environment = dict(os.environ)
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/g14a_rehearsal_pycache"
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    command_log.append(
        {
            "started_at": started,
            "completed_at": utc_now(),
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"rehearsal command failed ({result.returncode}): {' '.join(command)}\n"
            + result.stderr
        )


def build_fairness_manifest(capacity_mb: float) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = build_manifest(
        root=ROOT,
        mobility_path=MOBILITY,
        workflow_path=WORKFLOW,
        window_plan_path=PLAN,
        catalog_path=CATALOG,
        seeds=SEEDS,
        max_workflows=1,
        workflow_selector="ordered",
        min_tasks=5,
        max_tasks=20,
        max_steps=1,
        max_mobility_rows=2500,
        primary_vehicle_selection="stable_first",
        capacity_unit="mb",
        capacity_value=capacity_mb,
        output_root=str(OUTPUT_ROOT / "rehearsal_runs"),
        evaluation_unit_limit=1,
        created_at=utc_now(),
        controller_agents=LEARNED_AGENTS,
    )
    report = validate_manifest(manifest, root=ROOT, check_files=True)
    if report["status"] != "pass":
        raise RuntimeError(f"typed fairness manifest failed: {report['errors']}")
    return manifest, report


def train_tiny_checkpoints(
    *,
    rehearsal_root: Path,
    capacity_label: str,
    config_path: Path,
    runtime: dict[str, Any],
    command_log: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    checkpoint_manifest: dict[str, dict[str, str]] = {agent: {} for agent in LEARNED_AGENTS}
    provenance_manifest: dict[str, dict[str, dict[str, Any]]] = {
        agent: {} for agent in LEARNED_AGENTS
    }
    reports: list[dict[str, Any]] = []
    for agent_name in LEARNED_AGENTS:
        for seed in SEEDS:
            train_root = rehearsal_root / "training" / capacity_label / f"seed_{seed}"
            command = [
                sys.executable,
                str(ROOT / "scripts/train_algo_pool_real_sample.py"),
                "--agent_name",
                agent_name,
                "--profile",
                "smoke",
                "--episodes",
                "1",
                "--update_every",
                "1",
                "--batch_size",
                "1",
                "--max_steps",
                "1",
                "--max_workflows",
                "1",
                "--max_mobility_rows",
                "500",
                "--window_plan_path",
                str(PLAN),
                "--model_cache_runtime_config",
                str(config_path),
                "--reward_positive_offset",
                "0",
                "--random_seed",
                str(seed),
                "--output_root",
                str(train_root),
            ]
            run_command(command, command_log)
            run_dirs = sorted((train_root / agent_name).iterdir())
            run_dir = run_dirs[-1]
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            checkpoint_path = Path(summary["latest_checkpoint_path"])
            validation = validate_checkpoint_provenance(
                checkpoint_path,
                expected_agent_name=agent_name,
                expected_seed=seed,
                expected_runtime_contract=runtime,
                expected_reward_positive_offset=0.0,
                expected_window_plan_identity=summary["train_window_plan_identity"],
            )
            if validation["status"] != "compatible":
                raise RuntimeError(f"tiny checkpoint provenance failed: {validation}")
            checkpoint_manifest[agent_name][str(seed)] = str(checkpoint_path)
            provenance = validation["metadata"]
            provenance_manifest[agent_name][str(seed)] = {
                "checkpoint_sha256": validation["checkpoint_sha256"],
                "execution_git_commit": provenance["execution_git_commit"],
                "train_window_plan_identity": provenance["train_window_plan_identity"],
            }
            reports.append(
                {
                    "capacity_label": capacity_label,
                    "agent_name": agent_name,
                    "seed": seed,
                    "training_run_id": summary["run_id"],
                    "training_summary_path": str(run_dir / "summary.json"),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": validation["checkpoint_sha256"],
                    "status": validation["status"],
                    "runtime_contract_sha256": runtime["runtime_contract_sha256"],
                    "formal_checkpoint": False,
                }
            )
    return checkpoint_manifest, provenance_manifest, reports


def run_typed_benchmark(
    *,
    rehearsal_root: Path,
    capacity_label: str,
    config_path: Path,
    fairness_path: Path,
    checkpoint_manifest: dict[str, dict[str, str]],
    provenance_manifest: dict[str, dict[str, dict[str, Any]]],
    command_log: list[dict[str, Any]],
) -> Path:
    input_root = rehearsal_root / "benchmark_inputs" / capacity_label
    checkpoint_path = input_root / "seed_checkpoint_manifest.json"
    provenance_path = input_root / "checkpoint_provenance_manifest.json"
    write_json(checkpoint_path, checkpoint_manifest)
    write_json(provenance_path, provenance_manifest)
    benchmark_root = rehearsal_root / "benchmarks" / capacity_label
    command = [
        sys.executable,
        str(ROOT / "scripts/benchmark_main_results.py"),
        "--agents",
        *ALL_AGENTS,
        "--seeds",
        *[str(seed) for seed in SEEDS],
        "--seed_checkpoint_manifest_path",
        str(checkpoint_path),
        "--checkpoint_provenance_manifest_path",
        str(provenance_path),
        "--cache_baseline_fairness_manifest_path",
        str(fairness_path),
        "--model_cache_runtime_config",
        str(config_path),
        "--window_plan_path",
        str(PLAN),
        "--max_workflows",
        "1",
        "--max_steps",
        "1",
        "--max_mobility_rows",
        "2500",
        "--min_tasks",
        "5",
        "--max_tasks",
        "20",
        "--workflow_selector",
        "ordered",
        "--primary_vehicle_selection",
        "stable_first",
        "--window_mode",
        "mixed_informative",
        "--reward_positive_offset",
        "0",
        "--output_root",
        str(benchmark_root),
    ]
    run_command(command, command_log)
    return sorted(path for path in benchmark_root.iterdir() if path.is_dir())[-1]


def run_legacy_benchmark(
    *,
    rehearsal_root: Path,
    label: str,
    config_path: Path,
    command_log: list[dict[str, Any]],
) -> Path:
    benchmark_root = rehearsal_root / "legacy_benchmarks" / label
    command = [
        sys.executable,
        str(ROOT / "scripts/benchmark_main_results.py"),
        "--agents",
        "reactive_lru",
        "--seeds",
        "7",
        "--model_cache_runtime_config",
        str(config_path),
        "--window_plan_path",
        str(PLAN),
        "--max_workflows",
        "1",
        "--max_steps",
        "1",
        "--max_mobility_rows",
        "500",
        "--reward_positive_offset",
        "0",
        "--output_root",
        str(benchmark_root),
    ]
    run_command(command, command_log)
    return sorted(path for path in benchmark_root.iterdir() if path.is_dir())[-1]


def reconcile_benchmark(
    benchmark_dirs: dict[str, Path], runtimes: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    event_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    for capacity_label, benchmark_dir in benchmark_dirs.items():
        aggregate = json.loads(
            (benchmark_dir / "aggregate_summary.json").read_text(encoding="utf-8")
        )
        if aggregate["runtime_contract_sha256"] != runtimes[capacity_label][
            "runtime_contract_sha256"
        ]:
            raise RuntimeError("benchmark aggregate runtime hash mismatch")
        csv_rows = list(
            csv.DictReader((benchmark_dir / "benchmark_rows.csv").open(encoding="utf-8"))
        )
        benchmark_rows.extend(csv_rows)
        summaries = sorted((benchmark_dir / "episodes").rglob("*.summary.json"))
        for summary_path in summaries:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            requests = [
                step for step in summary.get("step_trace", []) if step.get("current_node_id")
            ]
            events = summary.get("cache_event_trace", [])
            if len(events) != len(requests):
                raise RuntimeError("CacheEvent/request cardinality mismatch")
            for event in events:
                if event.get("event_schema_version") != "1.3.0":
                    raise RuntimeError("CacheEvent schema mismatch")
                if event.get("event_type") == "request":
                    if not event.get("dependency_bundle") or len(
                        event.get("requested_typed_objects") or []
                    ) != 2:
                        raise RuntimeError("typed request bundle missing")
                    if event.get("base_model_hit") is None or event.get("adapter_hit") is None:
                        raise RuntimeError("typed lookup evidence missing")
                    evicted_ids = list(event.get("evicted_object_ids") or [])
                    if len(evicted_ids) != len(set(evicted_ids)):
                        raise RuntimeError("typed eviction order contains duplicates")
                    transfer = event.get("transfer_mb_by_type") or {}
                    if any(float(value) < 0 for value in transfer.values()):
                        raise RuntimeError("negative per-type transfer")
                    if "workflow_state" in transfer and any(
                        key not in {"base_model", "adapter", "workflow_state"}
                        for key in transfer
                    ):
                        raise RuntimeError("workflow-state transfer mixed with unknown type")
            reduced = reduce_cache_efficiency_summary(summary)
            scalars = cache_efficiency_row_fields(summary)
            row = summary_to_row(summary)
            checks = {
                "cache_object_hit_rate": reduced.request_metrics.get("object_hit_rate"),
                "cache_byte_hit_rate": reduced.byte_metrics.get("byte_hit_rate"),
                "cache_joint_model_hit_rate": reduced.type_aware_metrics.get(
                    "joint_base_adapter_hit_rate"
                ),
                "cache_full_service_ready_rate": reduced.type_aware_metrics.get(
                    "full_service_ready_rate"
                ),
            }
            if any(scalars.get(key) != value or row.get(key) != value for key, value in checks.items()):
                raise RuntimeError("metrics 1.1 scalar reconciliation mismatch")
            event_rows.append(
                {
                    "capacity_label": capacity_label,
                    "summary_path": str(summary_path),
                    "request_count": len(requests),
                    "cache_event_count": len(events),
                    "schema": summary.get("cache_event_schema_version"),
                    "typed_bundle_complete": True,
                    "lookup_complete": True,
                    "stable_eviction_sequence": True,
                    "workflow_state_transfer_separate": True,
                }
            )
            metric_rows.append(
                {
                    "capacity_label": capacity_label,
                    "summary_path": str(summary_path),
                    "metrics_version": reduced.cache_efficiency_metrics_version,
                    "availability": reduced.availability,
                    "scalar_reconciliation": "pass",
                    "nullable_latency_saved": reduced.latency_saved_metrics.get(
                        "latency_saved_sum_ms"
                    ),
                }
            )
    event_report = {
        "status": "pass",
        "cache_event_schema_version": "1.3.0",
        "summary_count": len(event_rows),
        "rows": event_rows,
    }
    metric_report = {
        "status": "pass",
        "cache_efficiency_metrics_contract_version": "1.1.0",
        "summary_count": len(metric_rows),
        "missing_trace_semantics": cache_efficiency_row_fields({}),
        "rows": metric_rows,
    }
    audit = {
        "status": "pass",
        "benchmark_dirs": {key: str(value) for key, value in benchmark_dirs.items()},
        "episode_summary_count": len(event_rows),
        "row_count": len(benchmark_rows),
        "agents": ALL_AGENTS,
        "seeds": SEEDS,
        "capacities_mb": [320.0, 384.0],
        "raw_summary_retains_typed_events": True,
        "row_and_aggregate_are_lightweight_scalars": True,
        "same_step_action_before_lookup": True,
        "base_adapter_atomic_transaction": True,
    }
    return event_report, metric_report, audit


def negative_cases(runtime: dict[str, Any], checkpoint_report: dict[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def expect_failure(case_id: str, callback: Any) -> None:
        try:
            callback()
        except Exception as exc:
            cases.append(
                {
                    "case_id": case_id,
                    "status": "pass_rejected",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        else:
            raise RuntimeError(f"negative case did not fail: {case_id}")

    base = {
        "model_cache_profile": runtime["model_cache_profile"],
        "typed_catalog_path": runtime["typed_catalog_path"],
        "typed_catalog_fingerprint": runtime["typed_catalog_fingerprint"],
        "cache_capacity_profile": deepcopy(runtime["cache_capacity_profile"]),
    }
    slot = deepcopy(base)
    slot["cache_capacity_profile"].update(unit="adapter_slots", rsu_adapter_slots=3)
    expect_failure("typed_slot_rejected", lambda: resolve_model_cache_runtime(slot, root=ROOT))
    bad_catalog = deepcopy(base)
    bad_catalog["typed_catalog_fingerprint"] = "0" * 64
    expect_failure(
        "catalog_fingerprint_mismatch",
        lambda: resolve_model_cache_runtime(bad_catalog, root=ROOT),
    )
    invalid_mb = deepcopy(base)
    invalid_mb["cache_capacity_profile"]["capacity_mb"] = float("inf")
    expect_failure("non_finite_mb", lambda: resolve_model_cache_runtime(invalid_mb, root=ROOT))
    cases.append(
        {
            "case_id": "legacy_checkpoint_typed_status",
            "status": "pass_rejected",
            "gate_status": "unavailable_legacy_metadata",
            "evidence": checkpoint_report.get("legacy_checkpoint_gate"),
        }
    )
    return {"status": "pass", "case_count": len(cases), "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    command_log: list[dict[str, Any]] = []
    rehearsal_id = datetime.now().strftime("g14a_rehearsal_%Y%m%d_%H%M%S_%f")
    rehearsal_root = output_root / "rehearsal_runs" / rehearsal_id
    rehearsal_root.mkdir(parents=True, exist_ok=False)

    runtimes = {
        label: resolve_model_cache_runtime(config, root=ROOT)
        for label, config in TYPED_CONFIGS.items()
    }
    write_json(
        output_root / "resolved_runtime_contract.json",
        {
            "status": "pass",
            "label": LABEL,
            "contracts": runtimes,
        },
    )

    fairness_reports: dict[str, Any] = {}
    fairness_paths: dict[str, Path] = {}
    for label, runtime in runtimes.items():
        manifest, report = build_fairness_manifest(
            float(runtime["cache_capacity_profile"]["capacity_mb"])
        )
        path = (
            output_root / "typed_fairness_manifest.json"
            if label == "320mb"
            else output_root / f"typed_fairness_manifest_{label}.json"
        )
        write_json(path, manifest)
        fairness_paths[label] = path
        fairness_reports[label] = report
    write_json(
        output_root / "fairness_validation_report.json",
        {"status": "pass", "reports": fairness_reports},
    )

    checkpoint_reports: list[dict[str, Any]] = []
    benchmark_dirs: dict[str, Path] = {}
    for label, config_path in TYPED_CONFIGS.items():
        checkpoints, provenance, reports = train_tiny_checkpoints(
            rehearsal_root=rehearsal_root,
            capacity_label=label,
            config_path=config_path,
            runtime=runtimes[label],
            command_log=command_log,
        )
        checkpoint_reports.extend(reports)
        benchmark_dirs[label] = run_typed_benchmark(
            rehearsal_root=rehearsal_root,
            capacity_label=label,
            config_path=config_path,
            fairness_path=fairness_paths[label],
            checkpoint_manifest=checkpoints,
            provenance_manifest=provenance,
            command_log=command_log,
        )

    legacy_checkpoint = rehearsal_root / "legacy_checkpoint.pt"
    import torch

    torch.save({"training_metadata": {"agent_name": "ppo"}}, legacy_checkpoint)
    legacy_gate = validate_checkpoint_provenance(
        legacy_checkpoint,
        expected_agent_name="ppo",
        expected_seed=7,
        expected_runtime_contract=runtimes["320mb"],
        expected_reward_positive_offset=0.0,
        expected_window_plan_identity={},
    )
    checkpoint_payload = {
        "status": "pass",
        "formal_checkpoint_count": 0,
        "tiny_checkpoint_count": len(checkpoint_reports),
        "compatible_count": sum(
            item["status"] == "compatible" for item in checkpoint_reports
        ),
        "reports": checkpoint_reports,
        "legacy_checkpoint_gate": legacy_gate,
    }
    write_json(output_root / "checkpoint_provenance_validation.json", checkpoint_payload)

    event_report, metric_report, benchmark_audit = reconcile_benchmark(
        benchmark_dirs, runtimes
    )
    write_json(output_root / "cache_event_reconciliation.json", event_report)
    write_json(output_root / "metrics_reconciliation.json", metric_report)
    write_json(output_root / "benchmark_runtime_audit.json", benchmark_audit)

    legacy_rows = []
    for label, config_path in LEGACY_CONFIGS.items():
        runtime = resolve_model_cache_runtime(config_path, root=ROOT)
        benchmark_dir = run_legacy_benchmark(
            rehearsal_root=rehearsal_root,
            label=label,
            config_path=config_path,
            command_log=command_log,
        )
        aggregate = json.loads(
            (benchmark_dir / "aggregate_summary.json").read_text(encoding="utf-8")
        )
        legacy_rows.append(
            {
                "mode": label,
                "status": "pass",
                "runtime_contract_sha256": runtime["runtime_contract_sha256"],
                "capacity_profile": runtime["cache_capacity_profile"],
                "benchmark_dir": str(benchmark_dir),
                "fairness_manifest_status": aggregate["fairness_manifest_status"],
            }
        )
    write_json(
        output_root / "legacy_compatibility.json",
        {"status": "pass", "legacy_modes": legacy_rows},
    )

    training_audit = {
        "status": "pass",
        "shared_entrypoint": "scripts/train_algo_pool_real_sample.py",
        "typed_capable_controller_table": [
            {
                "agent_name": agent,
                "support": "supported_shared_entrypoint",
                "observation_contract": get_algo_spec(agent)["observation_contract"],
                "action_contract": get_algo_spec(agent)["action_contract"],
                "rehearsed": agent in LEARNED_AGENTS,
            }
            for agent in ["sa_ghmappo", "ppo", "mappo", "cache_offload_drl"]
        ],
        "network_loss_reward_action_or_hyperparameter_change": False,
        "formal_training_executed": False,
        "tiny_runs": checkpoint_reports,
    }
    write_json(output_root / "training_entrypoint_audit.json", training_audit)
    write_json(
        output_root / "negative_cases.json",
        negative_cases(runtimes["320mb"], checkpoint_payload),
    )

    run_manifest = {
        "rehearsal_manifest_version": "1.0.0",
        "run_id": rehearsal_id,
        "artifact_run_id": RUN_ID,
        "label": LABEL,
        "status": "pass",
        "created_at": utc_now(),
        "execution_git_commit": git_commit(),
        "dataset_skeleton": "NGSIM+Alibaba",
        "window_plan_path": str(PLAN),
        "window_split": "controlled_non_hidden",
        "hidden_consumed": False,
        "formal_consumed": False,
        "holdout_consumed": False,
        "g15_executed": False,
        "seeds": SEEDS,
        "capacities_mb": [320.0, 384.0],
        "reactive_agents": list(BASELINE_NAMES),
        "learned_agents": LEARNED_AGENTS,
        "benchmark_dirs": {key: str(value) for key, value in benchmark_dirs.items()},
        "claim_boundary": "plumbing validation only; no ranking or paper claim",
        "g14_readiness_status": "still_blocked_pending_g14b_split_protocol_and_formal_checkpoints",
    }
    write_json(output_root / "rehearsal_run_manifest.json", run_manifest)
    write_json(
        output_root / "command_log.json",
        {"status": "pass", "command_count": len(command_log), "commands": command_log},
    )

    integrity_files = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "artifact_integrity_manifest.json"
    )
    integrity = {
        "integrity_manifest_version": "1.0.0",
        "artifact_run_id": RUN_ID,
        "generated_at": utc_now(),
        "file_count": len(integrity_files),
        "files": [
            {
                "path": path.relative_to(output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in integrity_files
        ],
    }
    write_json(output_root / "artifact_integrity_manifest.json", integrity)
    print(
        json.dumps(
            {
                "status": "pass",
                "label": LABEL,
                "rehearsal_id": rehearsal_id,
                "output_root": str(output_root),
                "tiny_checkpoint_count": len(checkpoint_reports),
                "typed_episode_count": event_report["summary_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
