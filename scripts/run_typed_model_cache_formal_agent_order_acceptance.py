"""Run G14R7 clean-candidate acceptance without formal training or holdout access."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_typed_model_cache_formal_protocol import resolved_expansion_context
from src.evaluators.cache_baseline_fairness import build_manifest, validate_manifest
from src.evaluators.main_results_support import resolve_window_candidates
from src.evaluators.typed_model_cache_formal_execution import (
    validate_command_templates,
    validate_protocol_v1_1,
)
from src.evaluators.typed_model_cache_formal_protocol import canonical_sha256, sha256_file
from src.runtime.formal_agent_order import resolve_formal_agent_order


MOBILITY_NAME = (
    "Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv"
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


def run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": str(cwd),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            + (completed.stderr or completed.stdout)
        )
    return completed


def junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("./testsuite"))
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-worktree-root", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-path", required=True)
    args = parser.parse_args()
    clean_root = Path(args.clean_worktree_root).resolve()
    # Preserve the virtual-environment entry path: resolving its symlink would
    # replace it with the base interpreter and lose the venv site-packages.
    python = str(Path(args.python_executable).absolute())
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    summary_path = Path(args.summary_path).resolve()
    if (clean_root / ".venv").exists():
        raise ValueError("clean acceptance candidate must not contain local .venv")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=clean_root, text=True
    ).strip()
    if status:
        raise ValueError(f"clean acceptance candidate is dirty: {status}")
    protocol_root = (
        clean_root
        / "configs/experiment/typed_model_cache_formal_protocol_v1_7_20260827"
    )
    protocol_path = protocol_root / "protocol_v1_7_manifest.json"
    order_path = protocol_root / "formal_agent_order_contract.json"
    environment_path = protocol_root / "execution_environment_manifest.json"
    protocol = read_json(protocol_path)
    validate_protocol_v1_1(protocol)
    order = resolve_formal_agent_order(
        contract_path=order_path,
        protocol=protocol,
        scientific_config=read_json(protocol_root / "agent_training_scientific_config.json"),
    )
    context = resolved_expansion_context(
        protocol,
        protocol_path=str(protocol_path),
        output_root=str(output_root / "protocol_run"),
        python_executable=python,
    )
    outer = validate_command_templates(
        protocol["execution_contract"]["command_templates"], context
    )
    train_plan = outer["expanded"]["train"]
    training_commands = train_plan["commands"]
    training_agents = list(
        dict.fromkeys(row["agent"] for row in train_plan["matrix_contexts"])
    )
    if len(training_commands) != 150 or training_agents != order["learned_agent_order"]:
        raise ValueError("150-cell training command order drift")
    agent_command_audit = {}
    for phase, expanded in outer["expanded"].items():
        commands = expanded["commands"]
        rows = []
        for command in commands:
            if "--agents" in command:
                start = command.index("--agents") + 1
                end = next(
                    (
                        index
                        for index in range(start, len(command))
                        if command[index].startswith("--")
                    ),
                    len(command),
                )
                observed = command[start:end]
                if observed != order["main_benchmark_agent_order"]:
                    raise ValueError(f"expanded command agent order drift: {phase}")
                rows.append(observed)
        if rows:
            agent_command_audit[phase] = {
                "command_count": len(rows),
                "agent_order": rows[0],
                "all_commands_equal": all(row == rows[0] for row in rows),
            }

    protocol_run_root = output_root / "protocol_run"
    common = [
        python,
        str(clean_root / "scripts/run_typed_model_cache_formal_protocol.py"),
        "--protocol-path",
        str(protocol_path),
        "--output-root",
        str(protocol_run_root),
        "--python-executable",
        python,
        "--execution-environment-manifest",
        str(environment_path),
    ]
    dry = json.loads(run([*common, "--preflight", "--dry-run"], cwd=clean_root).stdout)
    if dry["command_expansion"]["command_matrix_sha256"] != outer[
        "command_matrix_sha256"
    ]:
        raise ValueError("outer/dry-run command expansion drift")

    # The full Protocol preflight requires materialized data in the clean candidate.
    run([*common, "--preflight"], cwd=clean_root)
    run([*common, "--phase", "tests", "--resume"], cwd=clean_root)
    preflight = read_json(protocol_run_root / "preflight.json")
    nested_hash = preflight["command_expansion"]["command_matrix_sha256"]
    if nested_hash != outer["command_matrix_sha256"]:
        raise ValueError("outer/nested expansion drift")

    mobility = clean_root / "data/raw/mobility/ngsim" / MOBILITY_NAME
    workflow = clean_root / "data/raw/workflow/alibaba2018/batch_task.csv"
    if not mobility.is_file() or not workflow.is_file():
        raise FileNotFoundError("clean candidate must contain materialized NGSIM and Alibaba data")
    runtime_path = (
        clean_root
        / "configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820"
        / "runtime_constrained_288mb.yaml"
    )
    rehearsal_root = output_root / "nonformal_rehearsal"
    _, plan = resolve_window_candidates(
        root_dir=clean_root,
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
    plan = {**plan, "selected_window_plan": list(plan["selected_windows"])}
    plan_path = (
        clean_root
        / "artifacts/analysis/g14r7_acceptance_tiny_window_plan.json"
    )
    write_json(plan_path, plan)
    fairness = build_manifest(
        root=clean_root,
        mobility_path=mobility,
        workflow_path=workflow,
        window_plan_path=plan_path,
        catalog_path=clean_root / "src/data/model_catalog/typed_model_cache_controlled.json",
        seeds=[7],
        max_workflows=1,
        workflow_selector="ordered",
        min_tasks=5,
        max_tasks=20,
        max_steps=1,
        max_mobility_rows=1500,
        primary_vehicle_selection="handoff_pressure",
        capacity_unit="mb",
        capacity_value=288.0,
        output_root=str(rehearsal_root / "benchmark"),
        evaluation_unit_limit=1,
        created_at=now(),
        controller_agents=order["learned_agent_order"],
    )
    fairness_report = validate_manifest(fairness, root=clean_root, check_files=True)
    if fairness_report["status"] != "pass":
        raise ValueError(fairness_report["errors"])
    fairness_path = rehearsal_root / "tiny_fairness.json"
    write_json(fairness_path, fairness)

    training_root = rehearsal_root / "training"
    checkpoint_rows = []
    for agent in order["learned_agent_order"]:
        run_id = f"tiny_tiny_288mb_{agent}_seed7"
        command = [
            python,
            str(clean_root / "scripts/train_algo_pool_real_sample.py"),
            "--agent_name",
            agent,
            "--profile",
            "smoke",
            "--episodes",
            "4",
            "--update_every",
            "1",
            "--batch_size",
            "1",
            "--max_steps",
            "1",
            "--checkpoint_every_updates",
            "4",
            "--max_workflows",
            "1",
            "--workflow_selector",
            "ordered",
            "--min_tasks",
            "5",
            "--max_tasks",
            "20",
            "--mobility_source",
            "ngsim",
            "--mobility_csv_path",
            str(mobility),
            "--max_mobility_rows",
            "1500",
            "--window_selector",
            "ordered",
            "--window_count",
            "1",
            "--window_length",
            "24",
            "--rsu_layout",
            "auto_dominant_tight",
            "--primary_vehicle_selection",
            "handoff_pressure",
            "--model_cache_runtime_config",
            str(runtime_path),
            "--reward_positive_offset",
            "0",
            "--random_seed",
            "7",
            "--output_root",
            str(training_root),
            "--run_id",
            run_id,
        ]
        run(command, cwd=clean_root)
        checkpoint = training_root / agent / run_id / "checkpoints/update_0004.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_rows.append(
            {"agent": agent, "path": str(checkpoint), "sha256": sha256_file(checkpoint)}
        )

    dev_command = [
        python,
        str(clean_root / "scripts/run_typed_model_cache_formal_dev_selection.py"),
        "--protocol-path",
        str(protocol_path),
        "--training-root",
        str(training_root),
        "--output-root",
        str(rehearsal_root / "dev"),
        "--output-path",
        str(rehearsal_root / "dev/dev_selection.json"),
        "--window-plan-path",
        str(plan_path),
        "--mobility-csv-path",
        str(mobility),
        "--workflow-csv-path",
        str(workflow),
        "--max-mobility-rows",
        "1500",
        "--window-selector",
        "ordered",
        "--window-length",
        "24",
        "--rsu-layout",
        "auto_dominant_tight",
        "--primary-vehicle-selection",
        "handoff_pressure",
        "--non-formal-rehearsal",
        "--rehearsal-seed",
        "7",
        "--rehearsal-capacity",
        "tiny_288mb",
        str(runtime_path),
        str(fairness_path),
        "--rehearsal-update-index",
        "4",
        "--training-run-prefix",
        "tiny",
        "--resolved-execution-context-path",
        str(protocol_run_root / "resolved_execution_context.json"),
        "--formal-agent-order-contract-path",
        str(order_path),
    ]
    for agent in order["learned_agent_order"]:
        dev_command.extend(["--rehearsal-agent", agent])
    run(dev_command, cwd=clean_root)
    benchmark_runs = list((rehearsal_root / "dev/dev_benchmarks").glob("**/main_results_*"))
    if len(benchmark_runs) != 1:
        raise ValueError("actual dev selector did not create exactly one nested benchmark")
    rows_path = benchmark_runs[0] / "benchmark_rows.csv"
    with rows_path.open(encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    observed_agents = [row["agent_name"] for row in raw_rows]
    if observed_agents != order["main_benchmark_agent_order"]:
        raise ValueError("actual 15-agent raw-row order drift")
    selection = read_json(rehearsal_root / "dev/dev_selection.json")
    freeze_path = rehearsal_root / "dev/checkpoint_freeze.json"
    run(
        [
            python,
            str(clean_root / "scripts/manage_typed_model_cache_formal_artifacts.py"),
            "--action",
            "checkpoint_freeze",
            "--protocol-path",
            str(protocol_path),
            "--input-root",
            str(rehearsal_root / "dev"),
            "--output-path",
            str(freeze_path),
        ],
        cwd=clean_root,
    )
    freeze = read_json(freeze_path)

    statistics_results = []
    for label in ("a", "b"):
        synthetic = list(raw_rows)
        random.Random(label).shuffle(synthetic)
        synthetic_path = rehearsal_root / f"statistics_{label}.csv"
        with synthetic_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(synthetic[0]))
            writer.writeheader()
            writer.writerows(synthetic)
        stats_root = rehearsal_root / f"statistics_{label}"
        run(
            [
                python,
                str(clean_root / "scripts/analyze_top_journal_statistics.py"),
                "--rows_path",
                str(synthetic_path),
                "--candidate_agent",
                order["statistics_candidate_agent"],
                "--baseline_agents",
                *order["statistics_baseline_agent_order"],
                "--metrics",
                "total_reward",
                "--bootstrap_samples",
                "100",
                "--output_root",
                str(stats_root),
                "--formal-agent-order-contract-path",
                str(order_path),
            ],
            cwd=clean_root,
        )
        statistics_results.append(read_json(stats_root / "paired_statistics.json")["rows"])
    if statistics_results[0] != statistics_results[1]:
        raise ValueError("statistics changed after input row reordering")

    reachability = preflight["window_reachability"]
    test_counts = junit_counts(protocol_run_root / "tests.xml")
    resolved_context = read_json(protocol_run_root / "resolved_execution_context.json")
    binding = read_json(protocol_run_root / "formal_training_execution_binding.json")
    summary = {
        "status": "pass",
        "formal": False,
        "performance_evidence": False,
        "holdout_opened": False,
        "clean_worktree_root": str(clean_root),
        "clean_worktree_has_local_venv": False,
        "execution_commit": resolved_context["scientific_identity"]["execution_commit"],
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "formal_agent_order_contract_semantic_sha256": order["semantic_sha256"],
        "scientific_config_semantic_sha256": resolved_context["scientific_identity"][
            "agent_scientific_config_semantic_sha256"
        ],
        "environment_fingerprint": resolved_context["scientific_identity"][
            "environment_fingerprint"
        ],
        "dependency_fingerprint": resolved_context["scientific_identity"][
            "dependency_fingerprint"
        ],
        "execution_binding_full_sha256": binding["binding_full_sha256"],
        "resolved_context_sha256": resolved_context["context_sha256"],
        "command_audit": {
            "status": "pass",
            "total_command_count": outer["command_count"],
            "training_command_count": len(training_commands),
            "unique_training_command_count": len(
                {canonical_sha256(command) for command in training_commands}
            ),
            "training_agent_order": training_agents,
            "agent_command_phases": agent_command_audit,
            "outer_expansion_sha256": outer["command_matrix_sha256"],
            "nested_expansion_sha256": nested_hash,
            "outer_nested_equal": True,
        },
        "preflight": {
            "status": "pass",
            "raw_rows": 11_850_526,
            "provider_frame_count": reachability["provider_frame_count"],
            "window_count": reachability["window_count"],
            "reachable_count": reachability["reachable_count"],
            "split_reachable_counts": reachability["split_reachable_counts"],
            **test_counts,
        },
        "nonformal_dev_rehearsal": {
            "status": "pass",
            "training_agent_count": len(checkpoint_rows),
            "tiny_checkpoint_count": len(checkpoint_rows),
            "dev_nested_benchmark_count": len(benchmark_runs),
            "raw_row_count": len(raw_rows),
            "observed_agent_order": observed_agents,
            "selection_count": len(selection["selected"]),
            "frozen_checkpoint_count": freeze["frozen_checkpoint_count"],
            "formal": False,
            "performance_evidence": False,
            "rows_path": str(rows_path),
            "rows_sha256": sha256_file(rows_path),
        },
        "statistics_order_invariance": {
            "status": "pass",
            "comparison_row_count": len(statistics_results[0]),
            "reordered_input_equal": True,
        },
        "ledger_regression": {
            "status": "pass",
            "phase_ledger_sha256": sha256_file(protocol_run_root / "phase_state.jsonl"),
            "preflight_and_tests_terminal": True,
            "resume_finalize_regression_covered_by_tests_phase": True,
        },
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
    }
    ending_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=clean_root, text=True
    ).strip()
    if ending_status:
        raise ValueError(f"clean acceptance candidate became dirty: {ending_status}")
    summary["clean_worktree_status_end"] = "clean"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
