"""Evaluate every frozen checkpoint candidate on dev and select outcome-blind winners."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
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
from src.evaluators.typed_model_cache_formal_protocol import canonical_sha256, sha256_file
from src.evaluators.formal_cell_transaction import (
    CellExecutionIdentity,
    FormalCellLedger,
    atomic_write_json_create_only,
    stable_cell_id,
)
from src.runtime.formal_training_contract import checkpoint_snapshot_indices
from src.runtime.formal_training_identity import (
    FormalTrainingIdentityError,
    learned_agent_rows,
    validate_checkpoint_training_identity,
)
from src.runtime.formal_agent_order import (
    FormalAgentOrderError,
    reject_permanently_invalid_run_references,
    resolve_formal_agent_order,
)
from src.runtime.portable_resource_identity import (
    add_portable_resource_arguments,
    resolve_argument_resources,
)
from src.runtime.resolved_formal_execution_context import (
    load_resolved_formal_execution_context,
    resolved_python_for_nested_consumer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-path", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--formal-window-consumption-contract-path", default="")
    parser.add_argument("--window-plan-path", required=True)
    parser.add_argument("--mobility-csv-path", required=True)
    parser.add_argument("--workflow-csv-path", required=True)
    parser.add_argument("--max-mobility-rows", type=int, required=True)
    parser.add_argument("--window-selector", choices=["ordered"], required=True)
    parser.add_argument("--window-length", type=int, required=True)
    parser.add_argument("--rsu-layout", required=True)
    parser.add_argument(
        "--primary-vehicle-selection",
        choices=["stable_first", "handoff_pressure"],
        required=True,
    )
    parser.add_argument("--non-formal-rehearsal", action="store_true")
    parser.add_argument("--rehearsal-agent", action="append", default=[])
    parser.add_argument("--rehearsal-seed", action="append", type=int, default=[])
    parser.add_argument(
        "--rehearsal-capacity",
        action="append",
        nargs=3,
        metavar=("LABEL", "RUNTIME_CONFIG", "FAIRNESS_MANIFEST"),
        default=[],
    )
    parser.add_argument("--rehearsal-update-index", type=int, default=4)
    parser.add_argument("--training-run-prefix", default="formal")
    parser.add_argument("--resolved-execution-context-path", default="")
    parser.add_argument("--formal-agent-order-contract-path", default="")
    add_portable_resource_arguments(parser)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def write_create_only(path: Path, payload: object) -> None:
    atomic_write_json_create_only(path, payload)


def write_create_only_or_verify(path: Path, payload: object) -> None:
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8-sig"))
        if observed != payload:
            raise FormalExecutionError(f"existing immutable JSON differs: {path}")
        return
    atomic_write_json_create_only(path, payload)


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
    if args.resource_registry_path:
        resolve_argument_resources(
            args,
            bindings=(
                ("mobility_resource_id", "mobility_csv_path", "mobility_dataset"),
                ("workflow_resource_id", "workflow_csv_path", "workflow_dataset"),
                ("window_plan_resource_id", "window_plan_path", "window_plan"),
            ),
        )
    protocol_path = Path(args.protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    validate_protocol_v1_1(protocol)
    nested_python = sys.executable
    protocol_version = protocol["typed_model_cache_formal_protocol_version"]
    resolved_context = None
    if protocol_version in {"1.5.0", "1.6.0", "1.7.0"}:
        if not args.resolved_execution_context_path:
            raise FormalExecutionError(
                "active protocol dev selection requires resolved execution context"
            )
        resolved_context, _ = load_resolved_formal_execution_context(
            args.resolved_execution_context_path,
            protocol=protocol,
            clean_worktree_root=ROOT,
            durable_run_root=Path(args.resolved_execution_context_path).resolve().parent,
            check_git=True,
        )
        nested_python = resolved_python_for_nested_consumer(
            resolved_context, observed_sys_executable=sys.executable
        )
    if args.non_formal_rehearsal:
        if args.formal_window_consumption_contract_path:
            raise FormalExecutionError(
                "non-formal rehearsal must not bind the formal window contract"
            )
        if not args.rehearsal_agent or not args.rehearsal_seed or not args.rehearsal_capacity:
            raise FormalExecutionError("non-formal rehearsal matrix is incomplete")
    else:
        if not args.formal_window_consumption_contract_path:
            raise FormalExecutionError("formal dev selection requires the window contract")
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
    order_audit = None
    if protocol_version == "1.7.0":
        if not args.formal_agent_order_contract_path:
            raise FormalExecutionError("Protocol v1.7 dev selection requires agent order contract")
        scientific = json.loads(
            (config_root / "agent_training_scientific_config.json").read_text(
                encoding="utf-8-sig"
            )
        )
        try:
            order_audit = resolve_formal_agent_order(
                contract_path=args.formal_agent_order_contract_path,
                protocol=protocol,
                scientific_config=scientific,
                reactive_baseline_order=BASELINE_NAMES,
            )
            reject_permanently_invalid_run_references(
                [args.training_root],
                contract=json.loads(
                    Path(args.formal_agent_order_contract_path).read_text(
                        encoding="utf-8-sig"
                    )
                ),
            )
        except FormalAgentOrderError as exc:
            raise FormalExecutionError(str(exc)) from exc
        learned_agents = list(order_audit["learned_agent_order"])
        if args.non_formal_rehearsal and list(args.rehearsal_agent) != learned_agents:
            raise FormalExecutionError(
                "Protocol v1.7 non-formal rehearsal must use the complete learned-agent order"
            )
    else:
        learned_agents = (
            list(args.rehearsal_agent)
            if args.non_formal_rehearsal
            else [str(row["agent"]) for row in learned_agent_rows(protocol)]
        )
    seeds = (
        list(args.rehearsal_seed)
        if args.non_formal_rehearsal
        else list(protocol["seed_plan"]["seeds"])
    )
    cadence = int(protocol["training_budget"]["checkpoint_frequency_updates"])
    expected_updates = int(protocol["training_budget"]["expected_update_count"])
    update_indices = (
        [int(args.rehearsal_update_index)]
        if args.non_formal_rehearsal
        else checkpoint_snapshot_indices(expected_updates, cadence)
    )
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidates: list[dict] = []
    cell_ledger = None
    expected_dev_cell_ids: list[str] = []
    if not args.non_formal_rehearsal and protocol["typed_model_cache_formal_protocol_version"] in {
        "1.4.0", "1.5.0", "1.6.0", "1.7.0"
    }:
        identity_path = output_root / "cell_ledger_identity.json"
        if not identity_path.is_file():
            raise FormalExecutionError(
                "protocol v1.4 dev resume requires the training cell ledger identity"
            )
        identity_payload = json.loads(identity_path.read_text(encoding="utf-8-sig"))
        identity = CellExecutionIdentity(**identity_payload["identity"])
        cell_ledger = FormalCellLedger(
            run_root=output_root,
            identity=identity,
            resume=True,
        )

    capacity_inputs = (
        {
            label: (Path(runtime_path), Path(fairness_path))
            for label, runtime_path, fairness_path in args.rehearsal_capacity
        }
        if args.non_formal_rehearsal
        else {
            label: (
                ROOT / runtime_relative,
                ROOT / index["dev_fairness_manifests"][label],
            )
            for label, runtime_relative in index["runtime_configs"].items()
        }
    )
    for capacity_label, (runtime_path, fairness_path) in capacity_inputs.items():
        fairness, report = load_and_validate_manifest(fairness_path, root=ROOT, check_files=True)
        if report.get("status") != "pass":
            raise FormalExecutionError(f"dev fairness validation failed: {capacity_label}")
        if order_audit is not None:
            try:
                resolve_formal_agent_order(
                    contract_path=args.formal_agent_order_contract_path,
                    protocol=protocol,
                    fairness_manifests=[fairness],
                    reactive_baseline_order=BASELINE_NAMES,
                )
            except FormalAgentOrderError as exc:
                raise FormalExecutionError(str(exc)) from exc
        selection = fairness["dataset_provenance"]["selection_filter_parameters"]
        if selection.get("primary_vehicle_selection") != args.primary_vehicle_selection:
            raise FormalExecutionError("dev vehicle selection CLI/fairness mismatch")
        max_steps = {int(unit["max_steps"]) for unit in fairness["window_workload_plan"]["evaluation_units"]}
        if len(max_steps) != 1:
            raise FormalExecutionError("dev fairness manifest has mixed max_steps")
        for update_index in update_indices:
            coordinates = {
                "capacity_label": capacity_label,
                "update_index": int(update_index),
            }
            expected_dev_cell_ids.append(stable_cell_id("dev_select", coordinates))
            seed_manifest: dict[str, dict[str, str]] = {}
            provenance_manifest: dict[str, dict[str, dict]] = {}
            metadata_by_agent_seed: dict[tuple[str, int], dict] = {}
            for agent in learned_agents:
                for seed in seeds:
                    checkpoint_path = (
                        Path(args.training_root)
                        / agent
                        / f"{args.training_run_prefix}_{capacity_label}_{agent}_seed{seed}"
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
                        or (
                            not args.non_formal_rehearsal
                            and contract.get("formal_protocol_semantic_sha256")
                            != protocol["hashes"]["semantic_sha256"]
                        )
                    ):
                        raise FormalExecutionError("dev checkpoint formal binding mismatch")
                    if (
                        not args.non_formal_rehearsal
                        and protocol_version in {"1.6.0", "1.7.0"}
                    ):
                        scientific_identity = resolved_context["scientific_identity"]
                        try:
                            validate_checkpoint_training_identity(
                                metadata,
                                scientific_config_sha256=str(
                                    scientific_identity[
                                        "agent_scientific_config_semantic_sha256"
                                    ]
                                ),
                                binding_sha256=str(
                                    scientific_identity[
                                        "formal_training_execution_binding_sha256"
                                    ]
                                ),
                                protocol_semantic_sha256=protocol["hashes"][
                                    "semantic_sha256"
                                ],
                                execution_commit=str(
                                    scientific_identity["execution_commit"]
                                ),
                                resolved_context_sha256=str(
                                    resolved_context["context_sha256"]
                                ),
                                formal_agent_order_contract_semantic_sha256=(
                                    order_audit["semantic_sha256"]
                                    if order_audit is not None
                                    else None
                                ),
                            )
                        except FormalTrainingIdentityError as exc:
                            raise FormalExecutionError(str(exc)) from exc
                    seed_manifest.setdefault(agent, {})[str(seed)] = str(checkpoint_path)
                    typed = metadata.get("typed_runtime_provenance") or {}
                    provenance_manifest.setdefault(agent, {})[str(seed)] = {
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "execution_git_commit": typed.get("execution_git_commit"),
                        "train_window_plan_identity": typed.get("train_window_plan_identity"),
                    }
                    metadata_by_agent_seed[(agent, seed)] = metadata
            if cell_ledger is None:
                cell_root = output_root / "dev_inputs" / capacity_label / f"update_{update_index:04d}"
                committed_cell_root = None
                seed_path = cell_root / "seed_checkpoint_manifest.json"
                provenance_path = cell_root / "checkpoint_provenance_manifest.json"
                benchmark_root = output_root / "dev_benchmarks" / capacity_label / f"update_{update_index:04d}"
            else:
                committed_cell_root = (
                    output_root
                    / "dev_benchmarks"
                    / capacity_label
                    / f"update_{update_index:04d}"
                )
                seed_path = committed_cell_root / "seed_checkpoint_manifest.json"
                provenance_path = committed_cell_root / "checkpoint_provenance_manifest.json"
                benchmark_root = committed_cell_root / "benchmark"
            command = [
                nested_python,
                str(ROOT / "scripts/benchmark_main_results.py"),
                "--agents", *BASELINE_NAMES, *learned_agents,
                "--seeds", *[str(seed) for seed in seeds],
                "--seed_checkpoint_manifest_path", str(seed_path),
                "--checkpoint_provenance_manifest_path", str(provenance_path),
                "--cache_baseline_fairness_manifest_path", str(fairness_path),
                "--model_cache_runtime_config", str(runtime_path),
                "--window_plan_path", str(ROOT / fairness["window_workload_plan"]["window_plan_path"]),
                "--mobility_csv_path", args.mobility_csv_path,
                "--workflow_csv_path", args.workflow_csv_path,
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
            if order_audit is not None:
                command.extend(
                    [
                        "--formal-agent-order-contract-path",
                        str(Path(args.formal_agent_order_contract_path).resolve()),
                    ]
                )
            if not args.non_formal_rehearsal:
                command.extend(
                    [
                        "--formal_window_consumption_contract_path",
                        args.formal_window_consumption_contract_path,
                        "--formal_window_split", "dev",
                        "--window_consumption_mode", "formal",
                    ]
                )
            if args.resource_registry_path:
                command.extend(
                    [
                        "--resource-registry-path", args.resource_registry_path,
                        "--repository-root", args.repository_root or str(ROOT),
                        "--data-root", args.data_root,
                        "--protocol-artifact-root", args.protocol_artifact_root,
                        "--checkpoint-root", args.checkpoint_root or str(Path(args.training_root)),
                        "--mobility-resource-id", args.mobility_resource_id,
                        "--workflow-resource-id", args.workflow_resource_id,
                        "--window-plan-resource-id", args.window_plan_resource_id,
                        "--runtime-config-resource-id", f"runtime_config.{capacity_label}",
                        "--fairness-manifest-resource-id", (
                            f"fairness_manifest.rehearsal.{capacity_label}"
                            if args.non_formal_rehearsal
                            else f"fairness_manifest.dev.{capacity_label}"
                        ),
                    ]
                )
            begun = None
            if cell_ledger is not None:
                input_identity = canonical_sha256(
                    {
                        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
                        "formal_agent_order_contract_semantic_sha256": (
                            order_audit["semantic_sha256"] if order_audit else None
                        ),
                        "coordinates": coordinates,
                        "seed_checkpoint_manifest": seed_manifest,
                        "checkpoint_provenance_manifest": provenance_manifest,
                        "fairness_manifest_sha256": sha256_file(fairness_path),
                        "runtime_config_sha256": sha256_file(runtime_path),
                    }
                )
                begun = cell_ledger.begin_cell(
                    phase="dev_select",
                    coordinates=coordinates,
                    command=command,
                    input_hash=input_identity,
                    committed_path=committed_cell_root,
                )
                if begun["status"] == "skipped_committed":
                    committed_candidates = json.loads(
                        (committed_cell_root / "candidate_rows.json").read_text(
                            encoding="utf-8-sig"
                        )
                    )
                    if not isinstance(committed_candidates, list):
                        raise FormalExecutionError("committed dev candidate rows are invalid")
                    candidates.extend(committed_candidates)
                    continue
                staging_root = Path(begun["record"]["staging_path"])
                seed_path = staging_root / "seed_checkpoint_manifest.json"
                provenance_path = staging_root / "checkpoint_provenance_manifest.json"
                benchmark_root = staging_root / "benchmark"
                command[command.index("--seed_checkpoint_manifest_path") + 1] = str(seed_path)
                command[command.index("--checkpoint_provenance_manifest_path") + 1] = str(provenance_path)
                command[command.index("--output_root") + 1] = str(benchmark_root)
            write_create_only(seed_path, seed_manifest)
            write_create_only(provenance_path, provenance_manifest)
            before = set(benchmark_root.iterdir()) if benchmark_root.exists() else set()
            started_ns = time.monotonic_ns()
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            if result.returncode != 0:
                if cell_ledger is not None and begun is not None:
                    cell_ledger.fail_cell(
                        begun["cell_id"],
                        return_code=result.returncode,
                        classification="dev_candidate_evaluation_failure",
                        retryable=result.returncode == 75,
                    )
                raise RuntimeError(result.stderr or result.stdout)
            created = sorted(set(benchmark_root.iterdir()) - before)
            if len(created) != 1:
                raise FormalExecutionError("dev benchmark did not create exactly one run")
            with (created[0] / "benchmark_rows.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            cell_candidates: list[dict] = []
            for agent in learned_agents:
                for seed in seeds:
                    selected_rows = [
                        row
                        for row in rows
                        if row.get("agent_name") == agent and int(row.get("seed", -1)) == seed
                    ]
                    metadata = metadata_by_agent_seed[(agent, seed)]
                    checkpoint_path = Path(seed_manifest[agent][str(seed)])
                    cell_candidates.append(
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
                            "formal_training_contract": metadata.get("formal_training_contract"),
                            "agent_scientific_config_semantic_sha256": metadata.get(
                                "agent_scientific_config_semantic_sha256"
                            ),
                            "formal_training_execution_binding_sha256": metadata.get(
                                "formal_training_execution_binding_sha256"
                            ),
                            "formal_protocol_semantic_sha256": metadata.get(
                                "formal_protocol_semantic_sha256"
                            ),
                            "execution_commit": metadata.get("execution_commit"),
                            "resolved_execution_context_sha256": metadata.get(
                                "resolved_execution_context_sha256"
                            ),
                            "formal_agent_order_contract_semantic_sha256": (
                                order_audit["semantic_sha256"] if order_audit else None
                            ),
                            "non_formal_rehearsal": bool(args.non_formal_rehearsal),
                            "typed_runtime_provenance": metadata.get("typed_runtime_provenance"),
                        }
                    )
            if cell_ledger is not None and begun is not None:
                write_create_only(
                    Path(begun["record"]["staging_path"]) / "candidate_rows.json",
                    cell_candidates,
                )
                cell_ledger.commit_cell(
                    begun["cell_id"],
                    required_paths=[
                        "candidate_rows.json",
                        "seed_checkpoint_manifest.json",
                        "checkpoint_provenance_manifest.json",
                    ],
                    monotonic_started_ns=started_ns,
                )
            candidates.extend(cell_candidates)

    if cell_ledger is not None:
        cell_ledger.assert_complete_matrix(
            phase="dev_select",
            expected_cell_ids=expected_dev_cell_ids,
        )
    candidates_path = output_root / "checkpoint_candidates.json"
    write_create_only_or_verify(candidates_path, candidates)
    selection_payload = dev_select(output_root, protocol)
    selection_payload["non_formal_rehearsal"] = bool(args.non_formal_rehearsal)
    write_create_only_or_verify(Path(args.output_path), selection_payload)
    print(json.dumps(selection_payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
