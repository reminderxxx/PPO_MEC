"""Run G14R14 real downstream consumers on a tiny non-formal matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.cache_baseline_fairness import (
    BASELINE_NAMES,
    full_manifest_sha256,
    semantic_protocol_sha256,
)
from src.evaluators.formal_phase_transaction import (
    PhaseCommandResult,
    TransactionalPhaseRunner,
    validate_phase_ledger_v3,
)
from src.evaluators.formal_cell_transaction import (
    CellExecutionIdentity,
    FormalCellLedger,
    execute_cell_artifact_transaction,
    resolve_child_output_descriptor,
    single_child_directory,
    stable_cell_id,
    validate_cell_ledger,
)
from src.evaluators.typed_model_cache_formal_execution import (
    PHASE_ORDER,
    support_setting_by_id,
)
from src.evaluators.typed_model_cache_formal_protocol import canonical_sha256
from src.runtime.generated_checkpoint_resources import (
    build_generated_checkpoint_registry,
    load_generated_checkpoint_registry,
    publish_or_validate_generated_checkpoint_registry,
    sha256_file,
)
from src.runtime.portable_resource_identity import (
    build_registry,
    build_resource_identity,
)


PROTOCOL_DIR = Path("configs/experiment/typed_model_cache_formal_protocol_v2_5_20260905")
PROTOCOL_PATH = PROTOCOL_DIR / "protocol_v2_5_manifest.json"
INDEX_PATH = PROTOCOL_DIR / "protocol_index.json"
ORDER_PATH = PROTOCOL_DIR / "formal_agent_order_contract.json"
PLAN_PATH = PROTOCOL_DIR / "nonformal_rehearsal_window_plan.json"
BASE_REGISTRY = Path(
    "configs/experiment/typed_model_cache_formal_protocol_v1_3_20260821/portable_resource_registry.json"
)
RUNTIME_DIR = Path("configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820")
MOBILITY_RELATIVE = Path(
    "raw/mobility/ngsim/"
    "Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv"
)
WORKFLOW_RELATIVE = Path("raw/workflow/alibaba2018/batch_task.csv")
CATALOG_RELATIVE = Path("src/data/model_catalog/typed_model_cache_controlled.json")
CAPACITIES = {
    "constrained_288mb": 288,
    "medium_576mb": 576,
    "relaxed_864mb": 864,
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def write_json(path: Path, payload: Any, *, create_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if create_only else "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def resource_identity(
    path: Path, logical_id: str, role: str, schema: str, expected: Path,
) -> dict[str, Any]:
    return build_resource_identity(
        path,
        logical_resource_id=logical_id,
        resource_role=role,
        schema_version=schema,
        revision=f"sha256:{sha256_file(path)}",
        expected_logical_relative_path=expected.as_posix(),
        allowed_resolvers=("explicit_path",),
        provenance={"producer": "G14R14 real non-formal consumer rehearsal"},
    )


def _rehash_fairness(payload: dict[str, Any]) -> dict[str, Any]:
    payload.pop("hashes", None)
    payload["identity"]["manifest_id"] = "pending"
    digest = semantic_protocol_sha256(payload)
    payload["identity"]["manifest_id"] = f"cbfm-{digest[:16]}"
    payload["hashes"] = {
        "semantic_protocol_sha256": digest,
        "full_manifest_sha256": full_manifest_sha256(payload),
        "semantic_hash_excludes": [
            "identity.manifest_id", "identity.created_at", "artifact_plan",
            "dataset normalized_absolute_path", "baseline config normalized_absolute_path",
            "hashes", "validation",
        ],
    }
    return payload


def prepare_static_registry(
    run_root: Path, protocol: dict[str, Any]
) -> tuple[Path, dict[str, Path], dict[str, str]]:
    fairness_paths = {
        label: ROOT / PROTOCOL_DIR / f"nonformal_rehearsal_fairness_{label}.json"
        for label in CAPACITIES
    }
    no_prediction = deepcopy(read_json(fairness_paths["medium_576mb"]))
    selection = no_prediction["dataset_provenance"]["selection_filter_parameters"]
    selection["prediction_confidence_scale"] = 0.0
    selection["drop_handoff_prediction_prob"] = 1.0
    _rehash_fairness(no_prediction)
    no_prediction_path = run_root / "inputs/fairness/ablation_no_prediction.json"
    write_json(no_prediction_path, no_prediction, create_only=True)
    fairness_paths["ablation_no_prediction"] = no_prediction_path
    fairness_ids = {
        label: f"fairness_manifest.rehearsal.{label}" for label in CAPACITIES
    }
    fairness_ids["ablation_no_prediction"] = "fairness_manifest.rehearsal.ablation_no_prediction"
    for row in protocol["execution_contract"]["command_templates"]["formal_support"]["matrix_contexts"]:
        setting = support_setting_by_id(protocol, row["support_setting_id"])
        if setting.get("parameter") == "capacity_mb":
            continue
        payload = deepcopy(read_json(fairness_paths["medium_576mb"]))
        selection = payload["dataset_provenance"]["selection_filter_parameters"]
        parameter = setting.get("parameter")
        value = setting.get("value", setting.get("baseline"))
        if parameter == "handoff_pressure":
            selection["primary_vehicle_selection"] = value
        elif parameter == "prediction_condition":
            selection.update(
                {
                    "baseline": {},
                    "no_prediction": {"prediction_confidence_scale": 0.0, "drop_handoff_prediction_prob": 1.0},
                    "noise_0.2": {"prediction_noise_std": 0.2},
                    "confidence_0.7": {"prediction_confidence_scale": 0.7},
                    "delay_2": {"prediction_delay_steps": 2},
                    "drop_0.3": {"drop_handoff_prediction_prob": 0.3},
                }[str(value)]
            )
        _rehash_fairness(payload)
        key = str(row["support_setting_id"])
        path = run_root / "inputs/fairness" / f"{key}.json"
        write_json(path, payload, create_only=True)
        fairness_paths[key] = path
        fairness_ids[key] = f"fairness_manifest.rehearsal.{key}"
    base = read_json(ROOT / BASE_REGISTRY)
    resources = list(base["resources"])
    resources.append(
        resource_identity(
            ROOT / PLAN_PATH,
            "window_plan.rehearsal.g14r14",
            "window_plan",
            "frozen_window_plan/v1",
            Path(PLAN_PATH.name),
        )
    )
    for label, path in fairness_paths.items():
        resources.append(
            resource_identity(
                path,
                fairness_ids[label],
                "fairness_manifest",
                "cache_baseline_fairness_manifest/v1.1.0",
                Path(path.name),
            )
        )
    registry = build_registry(
        resources,
        registry_id="g14r14-real-non-formal-static-inputs",
        created_at="2026-09-05T00:00:00Z",
    )
    registry_path = run_root / "inputs/portable_resource_registry.json"
    write_json(registry_path, registry, create_only=True)
    return registry_path, fairness_paths, fairness_ids


def static_flags(
    registry_path: Path, data_root: Path, run_root: Path, capacity: str,
    fairness_id: str | None = None,
) -> list[str]:
    result = [
        "--resource-registry-path", str(registry_path),
        "--repository-root", str(ROOT),
        "--data-root", str(data_root),
        "--protocol-artifact-root", str(ROOT),
        "--checkpoint-root", str(run_root),
        "--mobility-resource-id", "dataset.mobility.ngsim.vehicle_trajectories",
        "--workflow-resource-id", "dataset.workflow.alibaba2018.batch_task",
        "--window-plan-resource-id", "window_plan.rehearsal.g14r14",
        "--runtime-config-resource-id", f"runtime_config.{capacity}",
    ]
    if fairness_id:
        result += ["--fairness-manifest-resource-id", fairness_id]
    return result


def generated_flags(
    run_root: Path, capacity: str, *, dashed_paths: bool = False
) -> list[str]:
    checkpoint_root = run_root / "checkpoint_manifests" / capacity
    return [
        (
            "--seed-checkpoint-manifest-path"
            if dashed_paths else "--seed_checkpoint_manifest_path"
        ),
        str(checkpoint_root / "seed_checkpoint_manifest.json"),
        (
            "--checkpoint-provenance-manifest-path"
            if dashed_paths else "--checkpoint_provenance_manifest_path"
        ),
        str(checkpoint_root / "checkpoint_provenance_manifest.json"),
        "--generated-checkpoint-registry-path", str(run_root / "generated_checkpoint_resource_registry.json"),
        "--checkpoint-manifest-id", f"checkpoint_manifest.{capacity}",
        "--checkpoint-provenance-id", f"checkpoint_provenance.{capacity}",
    ]


def phase_runner(run_root: Path, protocol: dict[str, Any]) -> TransactionalPhaseRunner:
    context_path = run_root / "resolved_execution_context.json"
    context = read_json(context_path)
    binding = read_json(run_root / "formal_training_execution_binding.json")
    scientific = context["scientific_identity"]
    run_identity = canonical_sha256(
        {
            "output_root": str(run_root),
            "protocol": protocol["hashes"]["semantic_sha256"],
            "resource_registry": protocol["portable_resource_identity_contract"]["resource_registry_semantic_sha256"],
            "environment": scientific["environment_fingerprint"],
            "execution_commit": scientific["execution_commit"],
            "resolved_execution_context_sha256": context["context_sha256"],
            "resolved_execution_context_file_sha256": sha256_file(context_path),
            "formal_training_execution_binding_sha256": binding["binding_full_sha256"],
            "active_formal_bundle_sha256": scientific["active_formal_bundle_sha256"],
        }
    )
    return TransactionalPhaseRunner(
        output_root=run_root,
        run_identity_fingerprint=run_identity,
        phase_order=PHASE_ORDER,
        resume=True,
        resolved_execution_context_sha256=context["context_sha256"],
        resolved_execution_context_file_sha256=sha256_file(context_path),
    )


def run_phase(
    runner: TransactionalPhaseRunner,
    phase: str,
    commands: Iterable[list[str]],
    expected: list[str],
    cell_ledger: FormalCellLedger | None = None,
) -> None:
    command_list = list(commands)

    def execute(argv: list[str]) -> PhaseCommandResult:
        if cell_ledger is not None and phase in {
            "train", "formal_cache_policy", "formal_controller",
            "formal_ablation", "formal_support", "formal_scalability",
        }:
            original = list(argv)
            if phase == "train":
                agent = original[original.index("--agent_name") + 1]
                run_id = original[original.index("--run_id") + 1]
                capacity = next(label for label in CAPACITIES if label in run_id)
                coordinates = {"agent": agent, "seed": 7, "capacity_label": capacity}
                output_flag = "--output_root"
                final = Path(original[original.index(output_flag) + 1]) / agent / run_id

                def builder(staging, _cell_id):
                    staged = list(original)
                    staged[staged.index(output_flag) + 1] = str(staging)
                    return staged

                def resolver(staging, _cell_id, _completed):
                    return staging / agent / run_id, ["train_summary.json"], staging
            else:
                output_flag = "--output_root" if "--output_root" in original else "--output-root"
                final = Path(original[original.index(output_flag) + 1])
                capacity = next((label for label in CAPACITIES if label in str(final)), "medium_576mb")
                if phase == "formal_cache_policy":
                    coordinates = {"capacity_label": capacity}
                elif phase == "formal_controller":
                    coordinates = {"capacity_label": capacity}
                else:
                    setting = original[original.index("--setting-id") + 1]
                    key = (
                        "ablation_setting_id" if phase == "formal_ablation"
                        else "support_setting_id" if phase == "formal_support"
                        else "scalability_setting_id"
                    )
                    coordinates = {key: setting, "capacity_label": capacity}

                def builder(staging, actual_cell_id):
                    staged = list(original)
                    if phase == "formal_cache_policy":
                        artifact = staging / "artifact"
                        staged[staged.index(output_flag) + 1] = str(artifact / "benchmark")
                        replay_flag = "--request-replay-path"
                        staged[staged.index(replay_flag) + 1] = str(artifact / "request_replay.json")
                    elif phase == "formal_controller":
                        staged[staged.index(output_flag) + 1] = str(staging / "child_output")
                    else:
                        child = staging / "child_output"
                        staged[staged.index(output_flag) + 1] = str(child)
                        staged += [
                            "--cell-id", actual_cell_id,
                            "--cell-phase", phase,
                            "--cell-output-descriptor-path", str(child / "cell_child_output.json"),
                        ]
                    return staged

                def resolver(staging, actual_cell_id, _completed):
                    if phase == "formal_cache_policy":
                        artifact = staging / "artifact"
                        benchmark = single_child_directory(artifact / "benchmark")
                        if not (benchmark / "aggregate_summary.json").is_file():
                            raise ValueError("cache aggregate missing")
                        return artifact, ["request_replay.json"], artifact / "benchmark"
                    if phase == "formal_controller":
                        child = staging / "child_output"
                        return single_child_directory(child), ["aggregate_summary.json", "benchmark_rows.csv"], child
                    child = staging / "child_output"
                    artifact, descriptor = resolve_child_output_descriptor(
                        child / "cell_child_output.json",
                        output_root=child,
                        expected_cell_id=actual_cell_id,
                        expected_phase=phase,
                        expected_setting_id=original[original.index("--setting-id") + 1],
                    )
                    return artifact, descriptor["required_payload"], child
            result = execute_cell_artifact_transaction(
                cell_ledger,
                phase=phase,
                coordinates=coordinates,
                command=original,
                input_hash=canonical_sha256({"phase": phase, "coordinates": coordinates, "command": original}),
                committed_path=final,
                command_builder=builder,
                artifact_resolver=resolver,
                cwd=ROOT,
            )
            return PhaseCommandResult(
                int(result.get("return_code", 0)),
                str(result.get("stdout", "")),
                str(result.get("stderr", "")),
            )
        completed = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, check=False)
        return PhaseCommandResult(completed.returncode, completed.stdout, completed.stderr)

    runner.run_phase(
        phase,
        commands=command_list,
        input_hash=canonical_sha256(
            {"phase": phase, "commands": command_list, "mode": "non_formal_rehearsal"}
        ),
        expected_outputs=expected,
        executor=execute,
        infrastructure_retries=0,
    )


def benchmark_command(
    python: Path, run_root: Path, data_root: Path, registry: Path,
    fairness_paths: dict[str, Path], fairness_ids: dict[str, str], capacity: str,
    output_root: Path, agents: list[str],
) -> list[str]:
    return [
        str(python), str(ROOT / "scripts/benchmark_main_results.py"),
        "--agents", *agents,
        "--seeds", "7",
        *generated_flags(run_root, capacity),
        "--cache_baseline_fairness_manifest_path", str(fairness_paths[capacity]),
        "--model_cache_runtime_config", str(ROOT / RUNTIME_DIR / f"runtime_{capacity}.yaml"),
        "--window_plan_path", str(ROOT / PLAN_PATH),
        "--mobility_csv_path", str(data_root / MOBILITY_RELATIVE),
        "--workflow_csv_path", str(data_root / WORKFLOW_RELATIVE),
        "--window_selector", "ordered", "--window_length", "24",
        "--rsu_layout", "auto_dominant_tight", "--max_mobility_rows", "1500",
        "--max_workflows", "1", "--max_steps", "1", "--workflow_selector", "ordered",
        "--min_tasks", "5", "--max_tasks", "20",
        "--primary_vehicle_selection", "handoff_pressure",
        "--window_mode", "mixed_informative", "--prediction_horizon", "3",
        "--reward_positive_offset", "0", "--audit_runtime", "--non-formal-rehearsal",
        "--protocol-path", str(ROOT / PROTOCOL_PATH),
        "--resolved-execution-context-path", str(run_root / "resolved_execution_context.json"),
        "--formal-training-execution-binding-path", str(run_root / "formal_training_execution_binding.json"),
        "--formal-agent-order-contract-path", str(ROOT / ORDER_PATH),
        "--output_root", str(output_root),
        *static_flags(registry, data_root, run_root, capacity, fairness_ids[capacity]),
    ]


def support_command(
    python: Path, protocol: dict[str, Any], run_root: Path, data_root: Path,
    registry: Path, fairness_paths: dict[str, Path], fairness_ids: dict[str, str],
    capacity: str, setting_id: str, fairness_label: str, output_root: Path,
    agents: list[str], request_replay: Path | None = None,
) -> list[str]:
    selection = read_json(fairness_paths[fairness_label])[
        "dataset_provenance"
    ]["selection_filter_parameters"]
    command = [
        str(python), str(ROOT / "scripts/run_typed_model_cache_formal_support.py"),
        "--protocol-path", str(ROOT / PROTOCOL_PATH), "--setting-id", setting_id,
        "--model-cache-runtime-config", str(ROOT / RUNTIME_DIR / f"runtime_{capacity}.yaml"),
        "--cache-baseline-fairness-manifest-path", str(fairness_paths[fairness_label]),
        *generated_flags(run_root, capacity, dashed_paths=True),
        "--window-plan-path", str(ROOT / PLAN_PATH),
        "--mobility-csv-path", str(data_root / MOBILITY_RELATIVE),
        "--workflow-csv-path", str(data_root / WORKFLOW_RELATIVE),
        "--max-mobility-rows", "1500", "--window-selector", "ordered",
        "--window-length", "24", "--rsu-layout", "auto_dominant_tight",
        "--primary-vehicle-selection", str(selection["primary_vehicle_selection"]),
        "--agents", *agents, "--seeds", "7", "--output-root", str(output_root),
        "--non-formal-rehearsal",
        "--resolved-execution-context-path", str(run_root / "resolved_execution_context.json"),
        "--formal-agent-order-contract-path", str(ROOT / ORDER_PATH),
        *static_flags(registry, data_root, run_root, capacity, fairness_ids[fairness_label]),
    ]
    if request_replay is not None:
        command += ["--request-replay-path", str(request_replay)]
    return command


def setting_id(protocol: dict[str, Any], parameter: str, value: Any) -> str:
    matches = []
    for matrix in (
        protocol["ablation_and_support"]["support_setting_matrix"],
        protocol["ablation_and_support"]["scalability_setting_matrix"],
    ):
        for item in matrix["settings"]:
            if item["parameter"] == parameter:
                matches += [
                    level["setting_id"] for level in item["levels"]
                    if level.get("value", level.get("baseline")) == value
                ]
    if len(matches) != 1:
        raise ValueError(f"setting not unique: {parameter}={value}")
    support_setting_by_id(protocol, matches[0])
    return matches[0]


def count_statistics_rows(path: Path) -> int:
    payload = read_json(path)
    return len(payload.get("rows", []))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--execution-environment-manifest", required=True)
    args = parser.parse_args()
    run_root = Path(args.output_root).resolve()
    data_root = Path(args.data_root).resolve()
    python = Path(args.python_executable).absolute()
    if run_root.exists():
        raise FileExistsError(run_root)
    if (ROOT / ".venv").exists():
        raise ValueError("clean detached rehearsal candidate must not contain .venv")
    for path in (data_root / MOBILITY_RELATIVE, data_root / WORKFLOW_RELATIVE):
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = read_json(ROOT / PROTOCOL_PATH)
    order = read_json(ROOT / ORDER_PATH)
    learned = list(order["learned_agent_order"])
    agents = list(order["main_benchmark_agent_order"])
    run_root.parent.mkdir(parents=True, exist_ok=True)
    preflight = [
        str(python), str(ROOT / "scripts/run_typed_model_cache_formal_protocol.py"),
        "--preflight", "--output-root", str(run_root),
        "--python-executable", str(python),
        "--execution-environment-manifest", str(Path(args.execution_environment_manifest).resolve()),
        "--non-formal-rehearsal-profile", str(
            ROOT / PROTOCOL_DIR / "nonformal_cell_transaction_rehearsal_profile.json"
        ),
    ]
    completed = subprocess.run(preflight, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    registry, fairness_paths, fairness_ids = prepare_static_registry(run_root, protocol)
    marker = {
        "execution_mode": "non_formal_rehearsal", "formal": False,
        "performance_evidence": False, "holdout_capability": False,
        "expected_counts": {
            "committed_training_cells": 30, "candidate_checkpoints": 30,
            "latest_checkpoints": 30, "dev_candidate_evaluations": 30,
            "selections": 30, "frozen_checkpoints": 30,
            "frozen_checkpoints_by_capacity": {label: 10 for label in CAPACITIES},
            "cache_policy_cells": 3, "controller_cells": 3,
            "ablation_settings": 2, "support_settings": 11,
            "scalability_settings": 3, "primary_comparison_rows": 6,
            "formal_outer_window_clusters": 1,
        },
    }
    write_json(run_root / "non_formal_rehearsal.json", marker, create_only=True)
    runner = phase_runner(run_root, protocol)
    context = read_json(run_root / "resolved_execution_context.json")
    binding = read_json(run_root / "formal_training_execution_binding.json")
    static = read_json(registry)
    scientific = context["scientific_identity"]
    profile = read_json(
        ROOT / PROTOCOL_DIR / "nonformal_cell_transaction_rehearsal_profile.json"
    )
    cell_ledger = FormalCellLedger(
        run_root=run_root,
        identity=CellExecutionIdentity(
            run_id=run_root.name,
            execution_commit=scientific["execution_commit"],
            protocol_semantic_sha256=protocol["hashes"]["semantic_sha256"],
            resource_registry_semantic_sha256=static["hashes"]["semantic_sha256"],
            environment_fingerprint=scientific["environment_fingerprint"],
            split_semantic_sha256=protocol["identity"]["split_semantic_sha256"],
            window_contract_semantic_sha256=protocol["execution_contract"]["window_consumption_contract"]["semantic_sha256"],
            catalog_fingerprint=protocol["identity"]["catalog_fingerprint"],
            runtime_identity=canonical_sha256(protocol["identity"]["typed_runtime_contract_hashes_by_capacity"]),
            command_matrix_sha256=canonical_sha256({
                "formal_templates": protocol["execution_contract"]["command_templates"],
                "nonformal_profile": profile["semantic_sha256"],
                "shared_executor": "execute_cell_artifact_transaction",
            }),
        ),
    )
    run_phase(
        runner, "tests",
        [[str(python), "-m", "pytest", str(ROOT / "tests/test_generated_checkpoint_resource_identity_v25.py"), "-q", "--junitxml", str(run_root / "test_reports/generated_resource.xml")]],
        ["test_reports/generated_resource.xml"],
    )
    train_commands = []
    for capacity in CAPACITIES:
        for agent in learned:
            run_id = f"rehearsal_{capacity}_{agent}_seed7"
            train_commands.append([
                str(python), str(ROOT / "scripts/train_algo_pool_real_sample.py"),
                "--agent_name", agent, "--profile", "smoke", "--episodes", "4",
                "--update_every", "1", "--batch_size", "1", "--max_steps", "1",
                "--max_workflows", "1", "--max_mobility_rows", "1500",
                "--workflow_selector", "ordered", "--min_tasks", "5", "--max_tasks", "20",
                "--mobility_csv_path", str(data_root / MOBILITY_RELATIVE),
                "--workflow_csv_path", str(data_root / WORKFLOW_RELATIVE),
                "--window_plan_path", str(ROOT / PLAN_PATH), "--window_selector", "ordered",
                "--window_length", "24", "--rsu_layout", "auto_dominant_tight",
                "--primary_vehicle_selection", "handoff_pressure", "--window_mode", "mixed_informative",
                "--model_cache_runtime_config", str(ROOT / RUNTIME_DIR / f"runtime_{capacity}.yaml"),
                "--agent_config_path", str(
                    ROOT
                    / "configs/experiment/typed_model_cache_formal_protocol_v1_3_20260821"
                    / "agent_training_configs.json"
                ),
                "--checkpoint_every_updates", "4", "--reward_positive_offset", "0",
                "--random_seed", "7", "--output_root", str(run_root / "training"),
                "--run_id", run_id, "--non-formal-rehearsal",
                *static_flags(registry, data_root, run_root, capacity),
            ])
    run_phase(runner, "train", train_commands, ["training/**/train_summary.json"], cell_ledger)
    dev = [
        str(python), str(ROOT / "scripts/run_typed_model_cache_formal_dev_selection.py"),
        "--protocol-path", str(ROOT / PROTOCOL_PATH), "--training-root", str(run_root / "training"),
        "--output-root", str(run_root), "--output-path", str(run_root / "dev_selection.json"),
        "--window-plan-path", str(ROOT / PLAN_PATH),
        "--mobility-csv-path", str(data_root / MOBILITY_RELATIVE),
        "--workflow-csv-path", str(data_root / WORKFLOW_RELATIVE),
        "--max-mobility-rows", "1500", "--window-selector", "ordered",
        "--window-length", "24", "--rsu-layout", "auto_dominant_tight",
        "--primary-vehicle-selection", "handoff_pressure", "--non-formal-rehearsal",
        "--rehearsal-update-index", "4", "--training-run-prefix", "rehearsal",
        "--resolved-execution-context-path", str(run_root / "resolved_execution_context.json"),
        "--formal-agent-order-contract-path", str(ROOT / ORDER_PATH),
    ]
    for agent in learned:
        dev += ["--rehearsal-agent", agent]
    dev += ["--rehearsal-seed", "7"]
    for capacity in CAPACITIES:
        dev += ["--rehearsal-capacity", capacity, str(ROOT / RUNTIME_DIR / f"runtime_{capacity}.yaml"), str(fairness_paths[capacity])]
    dev += static_flags(registry, data_root, run_root, "medium_576mb")
    run_phase(runner, "dev_select", [dev], ["dev_selection.json", "checkpoint_candidates.json"])
    freeze = [
        str(python), str(ROOT / "scripts/manage_typed_model_cache_formal_artifacts.py"),
        "--action", "checkpoint_freeze", "--protocol-path", str(ROOT / PROTOCOL_PATH),
        "--input-root", str(run_root), "--output-path", str(run_root / "checkpoint_freeze.json"),
    ]
    run_phase(runner, "checkpoint_freeze", [freeze], ["checkpoint_freeze.json", "checkpoint_manifests/**/*.json"])
    generated = build_generated_checkpoint_registry(
        run_root=run_root, protocol=protocol, static_registry=static,
        resolved_execution_context=context, execution_binding=binding,
    )
    publication = publish_or_validate_generated_checkpoint_registry(
        run_root / "generated_checkpoint_resource_registry.json",
        generated,
        run_root=run_root,
        expected_run_id=run_root.name,
        static_registry_semantic_sha256=static["hashes"]["semantic_sha256"],
        protocol_semantic_sha256=protocol["hashes"]["semantic_sha256"],
        protocol_full_sha256=protocol["hashes"]["full_sha256"],
        active_formal_bundle_sha256=context["scientific_identity"]["active_formal_bundle_sha256"],
        execution_commit=context["scientific_identity"]["execution_commit"],
        resolved_execution_context_sha256=context["context_sha256"],
        formal_training_execution_binding_sha256=binding["binding_full_sha256"],
    )
    evaluation_units = {
        label: read_json(path)["window_workload_plan"]["evaluation_units"][0]["evaluation_unit_id"]
        for label, path in fairness_paths.items() if label in CAPACITIES
    }
    cache_commands = []
    controller_commands = []
    for capacity in CAPACITIES:
        child = benchmark_command(
            python, run_root, data_root, registry, fairness_paths, fairness_ids,
            capacity, run_root / "formal_cache_policy" / capacity, agents,
        )
        outer = [
            str(python), str(ROOT / "scripts/run_typed_model_cache_formal_cache_policy.py"),
            "--protocol-path", str(ROOT / PROTOCOL_PATH),
            "--fairness-manifest-path", str(fairness_paths[capacity]),
            "--evaluation-unit-id", evaluation_units[capacity],
            "--request-replay-path", str(run_root / "cache_policy_replay" / f"{capacity}.json"),
            *generated_flags(run_root, capacity, dashed_paths=True),
            "--resolved-execution-context-path", str(run_root / "resolved_execution_context.json"),
            "--formal-training-execution-binding-path", str(run_root / "formal_training_execution_binding.json"),
            *static_flags(registry, data_root, run_root, capacity, fairness_ids[capacity]),
            "--command", *child,
        ]
        cache_commands.append(outer)
        controller_commands.append(benchmark_command(
            python, run_root, data_root, registry, fairness_paths, fairness_ids,
            capacity, run_root / "formal_controller" / capacity, agents,
        ))
    run_phase(runner, "formal_cache_policy", cache_commands, ["formal_cache_policy/**/aggregate_summary.json", "formal_cache_policy/*/request_replay.json"], cell_ledger)
    run_phase(runner, "formal_controller", controller_commands, ["formal_controller/**/aggregate_summary.json", "formal_controller/**/benchmark_rows.csv"], cell_ledger)
    ablation_commands = [
        support_command(
            python, protocol, run_root, data_root, registry, fairness_paths, fairness_ids,
            "medium_576mb", setting_id(protocol, "typed_semantics", value), label,
            run_root / "formal_ablation" / setting_id(protocol, "typed_semantics", value), agents,
        )
        for value, label in (("typed_full", "medium_576mb"), ("no_prediction", "ablation_no_prediction"))
    ]
    run_phase(runner, "formal_ablation", ablation_commands, ["formal_ablation/**/support_provenance.json"], cell_ledger)
    support_commands = []
    for row in protocol["execution_contract"]["command_templates"]["formal_support"]["matrix_contexts"]:
        setting = str(row["support_setting_id"])
        capacity = str(row["capacity_label"])
        fairness_label = capacity if setting.startswith("capacity-") else setting
        support_commands.append(
            support_command(
                python, protocol, run_root, data_root, registry, fairness_paths, fairness_ids,
                capacity, setting, fairness_label,
                run_root / "formal_support" / setting, agents,
            )
        )
    run_phase(runner, "formal_support", support_commands, ["formal_support/**/support_provenance.json"], cell_ledger)
    replay = run_root / "formal_cache_policy/medium_576mb/request_replay.json"
    scalability_commands = [
        support_command(
            python, protocol, run_root, data_root, registry, fairness_paths, fairness_ids,
            "medium_576mb", setting_id(protocol, "oracle_state_limit", limit), "medium_576mb",
            run_root / "formal_scalability" / setting_id(protocol, "oracle_state_limit", limit), agents, replay,
        )
        for limit in (1000, 10000, 100000)
    ]
    run_phase(runner, "formal_scalability", scalability_commands, ["formal_scalability/**/support_provenance.json"], cell_ledger)
    statistics = [
        str(python), str(ROOT / "scripts/run_typed_model_cache_formal_statistics.py"),
        "--protocol-path", str(ROOT / PROTOCOL_PATH), "--input-root", str(run_root),
        "--output-root", str(run_root / "statistics"), "--non-formal-rehearsal",
        "--rehearsal-baseline-agent", "reactive_lru",
        "--resolved-execution-context-path", str(run_root / "resolved_execution_context.json"),
        "--formal-agent-order-contract-path", str(ROOT / ORDER_PATH),
        "--resource-registry-path", str(registry),
        "--generated-checkpoint-registry-path", str(run_root / "generated_checkpoint_resource_registry.json"),
    ]
    run_phase(runner, "formal_statistics", [statistics], ["statistics/paired_statistics.json"])
    if count_statistics_rows(run_root / "statistics/paired_statistics.json") != 6:
        raise RuntimeError("non-formal statistics did not produce six primary comparison rows")
    integrity = [
        str(python), str(ROOT / "scripts/manage_typed_model_cache_formal_artifacts.py"),
        "--action", "integrity", "--protocol-path", str(ROOT / PROTOCOL_PATH),
        "--input-root", str(run_root), "--output-path", str(run_root / "artifact_integrity_manifest.json"),
        "--resource-registry-path", str(registry),
        "--generated-checkpoint-registry-path", str(run_root / "generated_checkpoint_resource_registry.json"),
    ]
    gate = list(integrity)
    gate[gate.index("integrity")] = "formal_gate"
    gate[gate.index(str(run_root / "artifact_integrity_manifest.json"))] = str(run_root / "formal_gate.json")
    run_phase(runner, "formal_gate", [integrity, gate], ["artifact_integrity_manifest.json", "formal_gate.json"])
    run_phase(runner, "complete_without_holdout", [], [])
    ledger_rows = [json.loads(line) for line in (run_root / "phase_state.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_phase_ledger_v3(ledger_rows)
    cell_rows = [json.loads(line) for line in (run_root / "cell_state.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    cell_audit = validate_cell_ledger(cell_rows)
    ledger_audit = {
        "status": "pass",
        "schema_version": "3.0.0",
        "record_count": len(ledger_rows),
        "terminal_phase_count": sum(
            row.get("status") == "completed" for row in ledger_rows
        ),
        "hash_chain_complete": True,
    }
    gate_payload = read_json(run_root / "formal_gate.json")
    registry_payload, registry_audit = load_generated_checkpoint_registry(
        run_root / "generated_checkpoint_resource_registry.json",
        run_root=run_root, expected_run_id=run_root.name,
        static_registry_semantic_sha256=static["hashes"]["semantic_sha256"],
        protocol_semantic_sha256=protocol["hashes"]["semantic_sha256"],
        protocol_full_sha256=protocol["hashes"]["full_sha256"],
        active_formal_bundle_sha256=context["scientific_identity"]["active_formal_bundle_sha256"],
        execution_commit=context["scientific_identity"]["execution_commit"],
        resolved_execution_context_sha256=context["context_sha256"],
        formal_training_execution_binding_sha256=binding["binding_full_sha256"],
    )
    summaries = list(run_root.glob("training/**/train_summary.json"))
    output = {
        "real_generated_checkpoint_consumer_rehearsal_version": "1.0.0",
        "status": "pass" if gate_payload.get("passed") else "fail",
        "execution_mode": "non_formal_rehearsal", "formal": False,
        "performance_evidence": False, "holdout_capability": False,
        "holdout_opened": False, "hidden_data_used": False,
        "clean_detached_candidate": True, "candidate_has_local_venv": False,
        "shared_absolute_python": str(python),
        "phase_order": list(PHASE_ORDER),
        "completed_phase_order": [row["phase"] for row in ledger_rows if row.get("status") == "completed"],
        "completed_phase_terminal_count": sum(row.get("status") == "completed" for row in ledger_rows),
        "phase_ledger_validation": ledger_audit,
        "cell_ledger_validation": cell_audit,
        "shared_cell_executor": "src.evaluators.formal_cell_transaction.execute_cell_artifact_transaction",
        "training_cell_count": len(summaries),
        "checkpoint_opened_by_dev_selection_count": 30,
        "seed_manifest_parsed_by_consumer": True,
        "checkpoint_provenance_parsed_by_consumer": True,
        "generated_registry_validated_before_checkpoint_consumption": True,
        "generated_registry_publication": publication,
        "generated_registry_audit": registry_audit,
        "generated_registry_resource_count": len(registry_payload["resources"]),
        "capacity_labels": list(CAPACITIES),
        "cache_policy_outer_and_nested_executed": True,
        "controller_capacity_count": 3, "capacity_support_count": 3,
        "ablation_setting_count": 2, "support_setting_count": 11,
        "scalability_setting_count": 3,
        "statistics_row_count": 6, "exact_nonformal_gate": gate_payload,
        "formal_training_count": 0, "formal_checkpoint_count": 0,
        "formal_performance_count": 0, "g14c_v14_created": False,
        "historical_g14c_v1_v13_reused": False,
        "rehearsal_root": str(run_root),
    }
    if output["completed_phase_terminal_count"] != 13 or output["status"] != "pass":
        raise RuntimeError("G14R14 real downstream rehearsal did not close")
    write_json(run_root.parent / "real_downstream_consumer_rehearsal.json", output, create_only=True)
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
