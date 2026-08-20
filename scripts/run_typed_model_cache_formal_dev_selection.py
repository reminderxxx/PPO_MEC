"""Evaluate every frozen checkpoint candidate on dev and select outcome-blind winners."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.manage_typed_model_cache_formal_artifacts import dev_select
from src.evaluators.cache_baseline_fairness import (
    BASELINE_NAMES,
    load_and_validate_manifest,
)
from src.evaluators.typed_model_cache_formal_execution import (
    FormalExecutionError,
    validate_protocol_v1_1,
)
from src.evaluators.formal_window_consumption import (
    load_contract as load_window_consumption_contract,
    validate_window_plan_binding,
)
from src.evaluators.typed_model_cache_formal_protocol import sha256_file
from src.runtime.formal_training_contract import checkpoint_snapshot_indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-path", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--formal-window-consumption-contract-path", required=True)
    parser.add_argument("--window-plan-path", required=True)
    parser.add_argument("--mobility-csv-path", required=True)
    parser.add_argument("--max-mobility-rows", type=int, required=True)
    parser.add_argument("--window-selector", choices=["ordered"], required=True)
    parser.add_argument("--window-length", type=int, required=True)
    parser.add_argument("--rsu-layout", required=True)
    parser.add_argument(
        "--primary-vehicle-selection",
        choices=["stable_first", "handoff_pressure"],
        required=True,
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def write_create_only(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def checkpoint_metadata(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise FormalExecutionError(f"checkpoint payload is not an object: {path}")
    metadata = payload.get("training_metadata") or payload.get("checkpoint_metadata")
    if not isinstance(metadata, dict):
        raise FormalExecutionError(f"checkpoint metadata is missing: {path}")
    return dict(metadata)


def finite_mean(rows: list[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) not in {None, ""}]
    if not values:
        raise FormalExecutionError(f"dev endpoint is unavailable: {field}")
    return statistics.fmean(values)


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    validate_protocol_v1_1(protocol)
    window_contract = load_window_consumption_contract(
        args.formal_window_consumption_contract_path
    )
    if (
        protocol.get("execution_contract", {})
        .get("window_consumption_contract", {})
        .get("semantic_sha256")
        != window_contract["hashes"]["semantic_sha256"]
    ):
        raise FormalExecutionError("dev window consumption contract hash mismatch")
    validate_window_plan_binding(
        contract=window_contract,
        plan_path=args.window_plan_path,
        split="dev",
        max_mobility_rows=args.max_mobility_rows,
        mobility_csv_path=args.mobility_csv_path,
        window_selector=args.window_selector,
        window_length=args.window_length,
        rsu_layout=args.rsu_layout,
        primary_vehicle_selection=args.primary_vehicle_selection,
        mode="formal",
    )
    config_root = protocol_path.parent
    index = json.loads((config_root / "protocol_index.json").read_text(encoding="utf-8-sig"))
    learned_agents = list(protocol["training_budget"]["agent_configs"])
    seeds = list(protocol["seed_plan"]["seeds"])
    cadence = int(protocol["training_budget"]["checkpoint_frequency_updates"])
    expected_updates = int(protocol["training_budget"]["expected_update_count"])
    update_indices = checkpoint_snapshot_indices(expected_updates, cadence)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidates: list[dict] = []

    for capacity_label, runtime_relative in index["runtime_configs"].items():
        runtime_path = ROOT / runtime_relative
        fairness_path = ROOT / index["dev_fairness_manifests"][capacity_label]
        fairness, report = load_and_validate_manifest(fairness_path, root=ROOT, check_files=True)
        if report.get("status") != "pass":
            raise FormalExecutionError(f"dev fairness validation failed: {capacity_label}")
        selection = fairness["dataset_provenance"]["selection_filter_parameters"]
        if selection.get("primary_vehicle_selection") != args.primary_vehicle_selection:
            raise FormalExecutionError("dev vehicle selection CLI/fairness mismatch")
        max_steps = {int(unit["max_steps"]) for unit in fairness["window_workload_plan"]["evaluation_units"]}
        if len(max_steps) != 1:
            raise FormalExecutionError("dev fairness manifest has mixed max_steps")
        for update_index in update_indices:
            seed_manifest: dict[str, dict[str, str]] = {}
            provenance_manifest: dict[str, dict[str, dict]] = {}
            metadata_by_agent_seed: dict[tuple[str, int], dict] = {}
            for agent in learned_agents:
                for seed in seeds:
                    checkpoint_path = (
                        Path(args.training_root)
                        / agent
                        / f"formal_{capacity_label}_{agent}_seed{seed}"
                        / "checkpoints"
                        / f"update_{update_index:04d}.pt"
                    ).resolve()
                    if not checkpoint_path.is_file():
                        raise FileNotFoundError(checkpoint_path)
                    metadata = checkpoint_metadata(checkpoint_path)
                    contract = metadata.get("formal_training_contract") or {}
                    schedule = metadata.get("checkpoint_schedule") or {}
                    if (
                        int(metadata.get("update_count", -1)) != update_index
                        or int(schedule.get("checkpoint_every_updates", -1)) != cadence
                        or contract.get("formal_protocol_semantic_sha256")
                        != protocol["hashes"]["semantic_sha256"]
                    ):
                        raise FormalExecutionError("dev checkpoint formal binding mismatch")
                    seed_manifest.setdefault(agent, {})[str(seed)] = str(checkpoint_path)
                    typed = metadata.get("typed_runtime_provenance") or {}
                    provenance_manifest.setdefault(agent, {})[str(seed)] = {
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "execution_git_commit": typed.get("execution_git_commit"),
                        "train_window_plan_identity": typed.get("train_window_plan_identity"),
                    }
                    metadata_by_agent_seed[(agent, seed)] = metadata
            cell_root = output_root / "dev_inputs" / capacity_label / f"update_{update_index:04d}"
            seed_path = cell_root / "seed_checkpoint_manifest.json"
            provenance_path = cell_root / "checkpoint_provenance_manifest.json"
            write_create_only(seed_path, seed_manifest)
            write_create_only(provenance_path, provenance_manifest)
            benchmark_root = output_root / "dev_benchmarks" / capacity_label / f"update_{update_index:04d}"
            command = [
                sys.executable,
                str(ROOT / "scripts/benchmark_main_results.py"),
                "--agents", *BASELINE_NAMES, *learned_agents,
                "--seeds", *[str(seed) for seed in seeds],
                "--seed_checkpoint_manifest_path", str(seed_path),
                "--checkpoint_provenance_manifest_path", str(provenance_path),
                "--cache_baseline_fairness_manifest_path", str(fairness_path),
                "--model_cache_runtime_config", str(runtime_path),
                "--window_plan_path", str(ROOT / fairness["window_workload_plan"]["window_plan_path"]),
                "--formal_window_consumption_contract_path", args.formal_window_consumption_contract_path,
                "--formal_window_split", "dev",
                "--window_consumption_mode", "formal",
                "--mobility_csv_path", args.mobility_csv_path,
                "--window_selector", args.window_selector,
                "--window_length", str(args.window_length),
                "--rsu_layout", args.rsu_layout,
                "--max_mobility_rows", str(selection["max_mobility_rows"]),
                "--max_workflows", str(selection["max_workflows"]),
                "--max_steps", str(next(iter(max_steps))),
                "--workflow_selector", str(selection["workflow_selector"]),
                "--min_tasks", str(selection["min_tasks"]),
                "--max_tasks", str(selection["max_tasks"]),
                "--primary_vehicle_selection", str(selection["primary_vehicle_selection"]),
                "--window_mode", str(selection["window_mode"]),
                "--prediction_horizon", str(selection["prediction_horizon"]),
                "--reward_positive_offset", "0",
                "--output_root", str(benchmark_root),
            ]
            before = set(benchmark_root.iterdir()) if benchmark_root.exists() else set()
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(result.stderr or result.stdout)
            created = sorted(set(benchmark_root.iterdir()) - before)
            if len(created) != 1:
                raise FormalExecutionError("dev benchmark did not create exactly one run")
            with (created[0] / "benchmark_rows.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for agent in learned_agents:
                for seed in seeds:
                    selected_rows = [
                        row
                        for row in rows
                        if row.get("agent_name") == agent and int(row.get("seed", -1)) == seed
                    ]
                    metadata = metadata_by_agent_seed[(agent, seed)]
                    checkpoint_path = Path(seed_manifest[agent][str(seed)])
                    candidates.append(
                        {
                            "agent_name": agent,
                            "seed": seed,
                            "capacity_label": capacity_label,
                            "update_index": update_index,
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_sha256": sha256_file(checkpoint_path),
                            "full_service_ready_byte_hit_rate": finite_mean(selected_rows, "full_service_ready_byte_hit_rate"),
                            "workflow_continuity_rate": finite_mean(selected_rows, "workflow_continuity_rate"),
                            "transfer_mb_per_request": finite_mean(selected_rows, "transfer_mb_per_request"),
                            "end_to_end_workflow_delay": finite_mean(selected_rows, "end_to_end_workflow_delay"),
                            "runtime_contract_sha256": metadata["typed_runtime_provenance"]["runtime_contract_sha256"],
                            "resolved_agent_config": metadata.get("resolved_agent_config"),
                            "checkpoint_schedule": metadata.get("checkpoint_schedule"),
                            "typed_runtime_provenance": metadata.get("typed_runtime_provenance"),
                        }
                    )

    candidates_path = output_root / "checkpoint_candidates.json"
    write_create_only(candidates_path, candidates)
    selection_payload = dev_select(output_root, protocol)
    write_create_only(Path(args.output_path), selection_payload)
    print(json.dumps(selection_payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
