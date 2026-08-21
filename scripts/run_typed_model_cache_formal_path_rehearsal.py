"""Run the exact G14R3 phase chain on bounded, non-formal, non-hidden data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.cache_baseline_fairness import (
    BASELINE_NAMES,
    build_manifest,
    full_manifest_sha256,
    semantic_protocol_sha256,
    validate_manifest,
)
from src.evaluators.typed_model_cache_formal_execution import (
    AppendOnlyPhaseRunner,
    PHASE_ORDER,
    support_setting_by_id,
    validate_phase_ledger,
)
from src.evaluators.typed_model_cache_formal_protocol import canonical_sha256
from src.runtime.portable_resource_identity import (
    build_registry,
    build_resource_identity,
    scientific_identity_fingerprint,
)
from src.runtime.typed_model_cache_runtime import resolve_model_cache_runtime


CONFIG_RELATIVE = Path(
    "configs/experiment/typed_model_cache_formal_protocol_v1_3_20260821"
)
PLAN_RELATIVE = Path(
    "configs/experiment/cache_baseline_fairness_g07_smoke_window_plan.json"
)
CATALOG_RELATIVE = Path("src/data/model_catalog/typed_model_cache_controlled.json")
MOBILITY_RELATIVE = Path(
    "raw/mobility/ngsim/"
    "Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv"
)
WORKFLOW_RELATIVE = Path("raw/workflow/alibaba2018/batch_task.csv")
AGENTS = ["sa_ghmappo", "ppo", "mappo", "cache_offload_drl"]
ALL_AGENTS = [*BASELINE_NAMES, *AGENTS]
SEEDS = [7, 13]
CAPACITIES = {
    "constrained_288mb": 288.0,
    "medium_576mb": 576.0,
}
INVALID_RUN = Path(
    "/private/tmp/ppo_mec_g14c_v3_a7c9e8e/artifacts/experiments/"
    "typed_model_cache_formal/typed_model_cache_formal_20260820_203430_g14c_v3"
).resolve()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any, *, create_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if create_only else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def portable_input_identity(
    path: Path,
    logical_id: str,
    role: str,
    schema: str,
    expected: Path,
    *,
    allowed: tuple[str, ...],
) -> dict[str, Any]:
    return build_resource_identity(
        path,
        logical_resource_id=logical_id,
        resource_role=role,
        schema_version=schema,
        revision=f"sha256:{path.stat().st_size}",
        expected_logical_relative_path=expected.as_posix(),
        allowed_resolvers=allowed,
        provenance={"producer": "G14R3 exact non-formal rehearsal"},
    )


def normalize_fairness_paths(
    manifest: dict[str, Any], *, data_root: Path, code_root: Path
) -> dict[str, Any]:
    mapping = {
        "ngsim_vehicle_trajectories": (
            "dataset.mobility.ngsim.vehicle_trajectories",
            "mobility_dataset",
            "NGSIMProvider/v1",
            MOBILITY_RELATIVE,
            data_root / MOBILITY_RELATIVE,
            ("explicit_path", "data_root"),
        ),
        "alibaba_cluster_trace_2018_batch_task": (
            "dataset.workflow.alibaba2018.batch_task",
            "workflow_dataset",
            "AlibabaDAGParser/legacy_batch_type",
            WORKFLOW_RELATIVE,
            data_root / WORKFLOW_RELATIVE,
            ("explicit_path", "data_root"),
        ),
        "ppo_mec_sample_adapter_catalog": (
            "catalog.typed_model_cache.controlled",
            "typed_catalog",
            "typed_model_cache_catalog/v1",
            CATALOG_RELATIVE,
            code_root / CATALOG_RELATIVE,
            ("explicit_path", "worktree_root"),
        ),
        "g07_non_hidden_window_plan": (
            "window_plan.rehearsal.controlled_public",
            "window_plan",
            "frozen_window_plan/v1",
            PLAN_RELATIVE,
            code_root / PLAN_RELATIVE,
            ("explicit_path", "worktree_root"),
        ),
    }
    for item in manifest["dataset_provenance"]["inputs"]:
        logical_dataset_id = str(item["logical_dataset_id"])
        logical_id, role, schema, relative, path, allowed = mapping[logical_dataset_id]
        identity = portable_input_identity(
            path, logical_id, role, schema, relative, allowed=allowed
        )
        item["logical_path"] = relative.as_posix()
        item["path_kind"] = (
            "data_root_relative" if "data_root" in allowed else "repository_relative"
        )
        item["normalized_absolute_path"] = str(path.resolve())
        item["portable_identity"] = identity
    manifest.pop("hashes", None)
    manifest["identity"]["manifest_id"] = "pending"
    semantic_hash = semantic_protocol_sha256(manifest)
    manifest["identity"]["manifest_id"] = f"cbfm-{semantic_hash[:16]}"
    manifest["hashes"] = {
        "semantic_protocol_sha256": semantic_hash,
        "full_manifest_sha256": full_manifest_sha256(manifest),
        "semantic_hash_excludes": [
            "identity.manifest_id",
            "identity.created_at",
            "artifact_plan",
            "dataset normalized_absolute_path",
            "baseline config normalized_absolute_path",
            "hashes",
            "validation",
        ],
    }
    report = validate_manifest(manifest, root=code_root, check_files=True)
    if report["status"] != "pass":
        raise RuntimeError(report["errors"])
    return manifest


def build_fairness(
    *,
    code_root: Path,
    data_root: Path,
    capacity_mb: float,
    output_root: Path,
    prediction_noise_std: float = 0.0,
    prediction_confidence_scale: float = 1.0,
    drop_handoff_prediction_prob: float = 0.0,
) -> dict[str, Any]:
    manifest = build_manifest(
        root=code_root,
        mobility_path=data_root / MOBILITY_RELATIVE,
        workflow_path=data_root / WORKFLOW_RELATIVE,
        window_plan_path=code_root / PLAN_RELATIVE,
        catalog_path=code_root / CATALOG_RELATIVE,
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
        output_root=str(output_root),
        evaluation_unit_limit=1,
        created_at=now(),
        controller_agents=AGENTS,
        prediction_noise_std=prediction_noise_std,
        prediction_confidence_scale=prediction_confidence_scale,
        drop_handoff_prediction_prob=drop_handoff_prediction_prob,
    )
    return normalize_fairness_paths(
        manifest, data_root=data_root, code_root=code_root
    )


def prepare_inputs(run_root: Path, data_root: Path) -> dict[str, Any]:
    input_root = run_root / "inputs"
    fairness_root = input_root / "fairness"
    fairness_specs = {
        "constrained_288mb": {},
        "medium_576mb": {},
        "ablation_no_prediction": {
            "prediction_confidence_scale": 0.0,
            "drop_handoff_prediction_prob": 1.0,
        },
        "support_noise_0_2": {"prediction_noise_std": 0.2},
    }
    fairness_paths: dict[str, Path] = {}
    for label, overrides in fairness_specs.items():
        capacity = 288.0 if label == "constrained_288mb" else 576.0
        path = fairness_root / f"{label}.json"
        manifest = build_fairness(
            code_root=ROOT,
            data_root=data_root,
            capacity_mb=capacity,
            output_root=run_root / "execution",
            **overrides,
        )
        write_json(path, manifest, create_only=True)
        fairness_paths[label] = path

    base_registry = json.loads(
        (ROOT / CONFIG_RELATIVE / "portable_resource_registry.json").read_text(
            encoding="utf-8-sig"
        )
    )
    resources = list(base_registry["resources"])
    resources.append(
        portable_input_identity(
            ROOT / PLAN_RELATIVE,
            "window_plan.rehearsal.controlled_public",
            "window_plan",
            "frozen_window_plan/v1",
            PLAN_RELATIVE,
            allowed=("explicit_path", "worktree_root"),
        )
    )
    fairness_ids = {
        "constrained_288mb": "fairness_manifest.rehearsal.constrained_288mb",
        "medium_576mb": "fairness_manifest.rehearsal.medium_576mb",
        "ablation_no_prediction": "fairness_manifest.rehearsal.ablation_no_prediction",
        "support_noise_0_2": "fairness_manifest.rehearsal.support_noise_0_2",
    }
    for label, path in fairness_paths.items():
        resources.append(
            portable_input_identity(
                path,
                fairness_ids[label],
                "fairness_manifest",
                "cache_baseline_fairness_manifest/v1.1.0",
                Path(path.name),
                allowed=("explicit_path",),
            )
        )
    registry = build_registry(
        resources,
        registry_id="g14r3-non-formal-rehearsal-precheckpoint",
        created_at=now(),
    )
    registry_path = input_root / "portable_resource_registry.precheckpoint.json"
    write_json(registry_path, registry, create_only=True)
    return {
        "input_root": input_root,
        "fairness_paths": fairness_paths,
        "fairness_ids": fairness_ids,
        "registry": registry,
        "registry_path": registry_path,
    }


def add_checkpoint_registry(
    prepared: dict[str, Any], execution_root: Path
) -> Path:
    resources = list(prepared["registry"]["resources"])
    for label in CAPACITIES:
        path = execution_root / "checkpoint_manifests" / label / "seed_checkpoint_manifest.json"
        resources.append(
            portable_input_identity(
                path,
                f"checkpoint_manifest.rehearsal.{label}",
                "checkpoint_manifest",
                "portable_checkpoint_manifest/v1.1.0",
                Path(label) / path.name,
                allowed=("explicit_path", "checkpoint_root"),
            )
        )
    registry = build_registry(
        resources,
        registry_id="g14r3-non-formal-rehearsal-postcheckpoint",
        created_at=now(),
    )
    path = prepared["input_root"] / "portable_resource_registry.postcheckpoint.json"
    write_json(path, registry, create_only=True)
    return path


def resource_flags(
    *,
    registry_path: Path,
    data_root: Path,
    execution_root: Path,
    capacity_label: str,
    fairness_id: str | None = None,
    checkpoint_id: str | None = None,
) -> list[str]:
    flags = [
        "--resource-registry-path", str(registry_path),
        "--repository-root", str(ROOT),
        "--data-root", str(data_root),
        "--protocol-artifact-root", str(ROOT),
        "--checkpoint-root", str(execution_root / "checkpoint_manifests"),
        "--mobility-resource-id", "dataset.mobility.ngsim.vehicle_trajectories",
        "--workflow-resource-id", "dataset.workflow.alibaba2018.batch_task",
        "--window-plan-resource-id", "window_plan.rehearsal.controlled_public",
        "--runtime-config-resource-id", f"runtime_config.{capacity_label}",
    ]
    if fairness_id:
        flags.extend(["--fairness-manifest-resource-id", fairness_id])
    if checkpoint_id:
        flags.extend(["--checkpoint-manifest-id", checkpoint_id])
    return flags


def benchmark_command(
    *,
    python: Path,
    execution_root: Path,
    data_root: Path,
    registry_path: Path,
    prepared: dict[str, Any],
    capacity_label: str,
    fairness_label: str,
    output_name: str,
) -> list[str]:
    checkpoint_root = execution_root / "checkpoint_manifests" / capacity_label
    return [
        str(python), str(ROOT / "scripts/benchmark_main_results.py"),
        "--agents", *ALL_AGENTS,
        "--seeds", *[str(seed) for seed in SEEDS],
        "--seed_checkpoint_manifest_path", str(checkpoint_root / "seed_checkpoint_manifest.json"),
        "--checkpoint_provenance_manifest_path", str(checkpoint_root / "checkpoint_provenance_manifest.json"),
        "--cache_baseline_fairness_manifest_path", str(prepared["fairness_paths"][fairness_label]),
        "--model_cache_runtime_config", str(
            ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820"
            / f"runtime_{capacity_label}.yaml"
        ),
        "--window_plan_path", str(ROOT / PLAN_RELATIVE),
        "--mobility_csv_path", str(data_root / MOBILITY_RELATIVE),
        "--workflow_csv_path", str(data_root / WORKFLOW_RELATIVE),
        "--window_selector", "ordered",
        "--window_length", "24",
        "--rsu_layout", "auto_dominant_tight",
        "--max_mobility_rows", "2500",
        "--max_workflows", "1",
        "--max_steps", "1",
        "--workflow_selector", "ordered",
        "--min_tasks", "5",
        "--max_tasks", "20",
        "--primary_vehicle_selection", "stable_first",
        "--window_mode", "mixed_informative",
        "--prediction_horizon", "3",
        "--reward_positive_offset", "0",
        "--audit_runtime",
        "--output_root", str(execution_root / output_name),
        *resource_flags(
            registry_path=registry_path,
            data_root=data_root,
            execution_root=execution_root,
            capacity_label=capacity_label,
            fairness_id=prepared["fairness_ids"][fairness_label],
            checkpoint_id=f"checkpoint_manifest.rehearsal.{capacity_label}",
        ),
    ]


def support_command(
    *,
    python: Path,
    protocol_path: Path,
    execution_root: Path,
    data_root: Path,
    registry_path: Path,
    prepared: dict[str, Any],
    setting_id: str,
    fairness_label: str,
    output_name: str,
    request_replay_path: Path | None = None,
) -> list[str]:
    checkpoint_root = execution_root / "checkpoint_manifests/medium_576mb"
    command = [
        str(python), str(ROOT / "scripts/run_typed_model_cache_formal_support.py"),
        "--protocol-path", str(protocol_path),
        "--setting-id", setting_id,
        "--model-cache-runtime-config", str(
            ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820/"
            "runtime_medium_576mb.yaml"
        ),
        "--cache-baseline-fairness-manifest-path", str(prepared["fairness_paths"][fairness_label]),
        "--seed-checkpoint-manifest-path", str(checkpoint_root / "seed_checkpoint_manifest.json"),
        "--checkpoint-provenance-manifest-path", str(checkpoint_root / "checkpoint_provenance_manifest.json"),
        "--window-plan-path", str(ROOT / PLAN_RELATIVE),
        "--mobility-csv-path", str(data_root / MOBILITY_RELATIVE),
        "--workflow-csv-path", str(data_root / WORKFLOW_RELATIVE),
        "--max-mobility-rows", "2500",
        "--window-selector", "ordered",
        "--window-length", "24",
        "--rsu-layout", "auto_dominant_tight",
        "--primary-vehicle-selection", "stable_first",
        "--agents", *ALL_AGENTS,
        "--seeds", *[str(seed) for seed in SEEDS],
        "--output-root", str(execution_root / output_name),
        "--non-formal-rehearsal",
        *resource_flags(
            registry_path=registry_path,
            data_root=data_root,
            execution_root=execution_root,
            capacity_label="medium_576mb",
            fairness_id=prepared["fairness_ids"][fairness_label],
            checkpoint_id="checkpoint_manifest.rehearsal.medium_576mb",
        ),
    ]
    if request_replay_path is not None:
        command.extend(["--request-replay-path", str(request_replay_path)])
    return command


def setting_id(protocol: dict[str, Any], parameter: str, value: Any) -> str:
    containers = (
        protocol["ablation_and_support"]["support_setting_matrix"],
        protocol["ablation_and_support"]["scalability_setting_matrix"],
    )
    matches = [
        level["setting_id"]
        for container in containers
        for item in container["settings"]
        if item["parameter"] == parameter
        for level in item["levels"]
        if level["value"] == value
    ]
    if len(matches) != 1:
        raise RuntimeError(f"setting lookup failed: {parameter}={value}")
    support_setting_by_id(protocol, matches[0])
    return matches[0]


def run_phase(
    runner: AppendOnlyPhaseRunner,
    phase: str,
    commands: Iterable[list[str]],
    expected: list[str],
) -> None:
    command_list = list(commands)
    runner.run_phase(
        phase,
        command=command_list,
        input_hash=canonical_sha256(
            {"phase": phase, "commands": command_list, "mode": "non_formal_rehearsal"}
        ),
        expected_outputs=expected,
        infrastructure_retries=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    data_root = Path(args.data_root).resolve()
    # Preserve virtual-environment launcher symlinks. Resolving the symlink can
    # silently select the system interpreter and lose the venv dependencies.
    python = Path(args.python).absolute()
    if output_root.exists():
        raise FileExistsError(f"rehearsal output root already exists: {output_root}")
    if not (data_root / MOBILITY_RELATIVE).is_file() or not (
        data_root / WORKFLOW_RELATIVE
    ).is_file():
        raise FileNotFoundError("--data-root must contain the frozen NGSIM and Alibaba inputs")
    if INVALID_RUN == output_root or INVALID_RUN in output_root.parents:
        raise ValueError("rehearsal output cannot reuse the invalid G14C v3 run root")

    os.environ["PYTHONPYCACHEPREFIX"] = "/private/tmp/ppo_mec_g14r3_rehearsal_pycache"
    output_root.mkdir(parents=True, exist_ok=False)
    prepared = prepare_inputs(output_root, data_root)
    protocol_path = ROOT / CONFIG_RELATIVE / "protocol_v1_3_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    execution_root = output_root / "execution"
    runner = AppendOnlyPhaseRunner(protocol=protocol, output_root=execution_root)
    write_json(
        execution_root / "non_formal_rehearsal.json",
        {
            "execution_mode": "non_formal_rehearsal",
            "formal_training": False,
            "formal_evaluation": False,
            "holdout_opened": False,
            "hidden_data_used": False,
            "paper_claims_permitted": False,
        },
        create_only=True,
    )

    preflight_commands = [
        [
            str(python), str(ROOT / "scripts/validate_cache_baseline_fairness_manifest.py"),
            "--manifest_path", str(path),
            "--report_path", str(execution_root / "preflight" / f"{label}.json"),
        ]
        for label, path in prepared["fairness_paths"].items()
    ]
    run_phase(runner, "preflight", preflight_commands, ["preflight/*.json"])
    run_phase(
        runner,
        "tests",
        [[
            str(python), "-m", "pytest", str(ROOT / "tests/test_portable_resource_identity.py"),
            "-q", "--junitxml", str(execution_root / "test_reports/portable_identity.xml"),
        ]],
        ["test_reports/portable_identity.xml"],
    )

    train_commands: list[list[str]] = []
    runtime_root = ROOT / "configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820"
    for capacity_label in CAPACITIES:
        for agent in AGENTS:
            for seed in SEEDS:
                train_commands.append(
                    [
                        str(python), str(ROOT / "scripts/train_algo_pool_real_sample.py"),
                        "--agent_name", agent,
                        "--profile", "smoke",
                        "--episodes", "4",
                        "--update_every", "1",
                        "--batch_size", "1",
                        "--max_steps", "1",
                        "--max_workflows", "1",
                        "--max_mobility_rows", "2500",
                        "--workflow_selector", "ordered",
                        "--min_tasks", "5",
                        "--max_tasks", "20",
                        "--mobility_csv_path", str(data_root / MOBILITY_RELATIVE),
                        "--workflow_csv_path", str(data_root / WORKFLOW_RELATIVE),
                        "--window_plan_path", str(ROOT / PLAN_RELATIVE),
                        "--window_selector", "ordered",
                        "--window_length", "24",
                        "--rsu_layout", "auto_dominant_tight",
                        "--primary_vehicle_selection", "stable_first",
                        "--window_mode", "mixed_informative",
                        "--model_cache_runtime_config", str(runtime_root / f"runtime_{capacity_label}.yaml"),
                        "--agent_config_path", str(ROOT / CONFIG_RELATIVE / "agent_training_configs.json"),
                        "--checkpoint_every_updates", "4",
                        "--reward_positive_offset", "0",
                        "--random_seed", str(seed),
                        "--output_root", str(execution_root / "training"),
                        "--run_id", f"rehearsal_{capacity_label}_{agent}_seed{seed}",
                        *resource_flags(
                            registry_path=prepared["registry_path"],
                            data_root=data_root,
                            execution_root=execution_root,
                            capacity_label=capacity_label,
                        ),
                    ]
                )
    run_phase(runner, "train", train_commands, ["training/**/train_summary.json"])

    dev_command = [
        str(python), str(ROOT / "scripts/run_typed_model_cache_formal_dev_selection.py"),
        "--protocol-path", str(protocol_path),
        "--training-root", str(execution_root / "training"),
        "--output-root", str(execution_root),
        "--output-path", str(execution_root / "dev_selection.json"),
        "--window-plan-path", str(ROOT / PLAN_RELATIVE),
        "--mobility-csv-path", str(data_root / MOBILITY_RELATIVE),
        "--workflow-csv-path", str(data_root / WORKFLOW_RELATIVE),
        "--max-mobility-rows", "2500",
        "--window-selector", "ordered",
        "--window-length", "24",
        "--rsu-layout", "auto_dominant_tight",
        "--primary-vehicle-selection", "stable_first",
        "--non-formal-rehearsal",
        "--rehearsal-agent", *AGENTS,
    ]
    # argparse action=append consumes one value per occurrence.
    agent_offset = dev_command.index("--rehearsal-agent")
    dev_command[agent_offset:agent_offset + 1 + len(AGENTS)] = [
        item for agent in AGENTS for item in ("--rehearsal-agent", agent)
    ]
    for seed in SEEDS:
        dev_command.extend(["--rehearsal-seed", str(seed)])
    for label in CAPACITIES:
        dev_command.extend(
            [
                "--rehearsal-capacity", label,
                str(runtime_root / f"runtime_{label}.yaml"),
                str(prepared["fairness_paths"][label]),
            ]
        )
    dev_command.extend(
        [
            "--rehearsal-update-index", "4",
            "--training-run-prefix", "rehearsal",
            *resource_flags(
                registry_path=prepared["registry_path"],
                data_root=data_root,
                execution_root=execution_root,
                capacity_label="medium_576mb",
            ),
        ]
    )
    run_phase(
        runner,
        "dev_select",
        [dev_command],
        ["dev_selection.json", "checkpoint_candidates.json"],
    )

    freeze_command = [
        str(python), str(ROOT / "scripts/manage_typed_model_cache_formal_artifacts.py"),
        "--action", "checkpoint_freeze",
        "--protocol-path", str(protocol_path),
        "--input-root", str(execution_root),
        "--output-path", str(execution_root / "checkpoint_freeze.json"),
    ]
    run_phase(
        runner,
        "checkpoint_freeze",
        [freeze_command],
        ["checkpoint_freeze.json", "checkpoint_manifests/**/*.json"],
    )
    post_registry = add_checkpoint_registry(prepared, execution_root)

    base_controller = benchmark_command(
        python=python,
        execution_root=execution_root,
        data_root=data_root,
        registry_path=post_registry,
        prepared=prepared,
        capacity_label="medium_576mb",
        fairness_label="medium_576mb",
        output_name="formal_cache_policy",
    )
    unit_id = prepared["fairness_paths"]["medium_576mb"]
    unit_payload = json.loads(unit_id.read_text(encoding="utf-8"))
    evaluation_unit_id = unit_payload["window_workload_plan"]["evaluation_units"][0]["evaluation_unit_id"]
    replay_path = execution_root / "cache_policy/request_replay.json"
    cache_command = [
        str(python), str(ROOT / "scripts/run_typed_model_cache_formal_cache_policy.py"),
        "--protocol-path", str(protocol_path),
        "--fairness-manifest-path", str(prepared["fairness_paths"]["medium_576mb"]),
        "--evaluation-unit-id", evaluation_unit_id,
        "--request-replay-path", str(replay_path),
        "--command", *base_controller,
    ]
    run_phase(
        runner,
        "formal_cache_policy",
        [cache_command],
        ["formal_cache_policy/**/aggregate_summary.json", "cache_policy/request_replay.json"],
    )

    controller_command = benchmark_command(
        python=python,
        execution_root=execution_root,
        data_root=data_root,
        registry_path=post_registry,
        prepared=prepared,
        capacity_label="medium_576mb",
        fairness_label="medium_576mb",
        output_name="formal_controller",
    )
    run_phase(
        runner,
        "formal_controller",
        [controller_command],
        ["formal_controller/**/aggregate_summary.json", "formal_controller/**/benchmark_rows.csv"],
    )

    ablation = support_command(
        python=python,
        protocol_path=protocol_path,
        execution_root=execution_root,
        data_root=data_root,
        registry_path=post_registry,
        prepared=prepared,
        setting_id=setting_id(protocol, "typed_semantics", "no_prediction"),
        fairness_label="ablation_no_prediction",
        output_name="formal_ablation",
    )
    run_phase(
        runner,
        "formal_ablation",
        [ablation],
        ["formal_ablation/**/support_provenance.json"],
    )

    support = support_command(
        python=python,
        protocol_path=protocol_path,
        execution_root=execution_root,
        data_root=data_root,
        registry_path=post_registry,
        prepared=prepared,
        setting_id=setting_id(protocol, "prediction_condition", "noise_0.2"),
        fairness_label="support_noise_0_2",
        output_name="formal_support",
    )
    run_phase(
        runner,
        "formal_support",
        [support],
        ["formal_support/**/support_provenance.json"],
    )

    scalability = support_command(
        python=python,
        protocol_path=protocol_path,
        execution_root=execution_root,
        data_root=data_root,
        registry_path=post_registry,
        prepared=prepared,
        setting_id=setting_id(protocol, "oracle_state_limit", 1000),
        fairness_label="medium_576mb",
        output_name="formal_scalability",
        request_replay_path=replay_path,
    )
    run_phase(
        runner,
        "formal_scalability",
        [scalability],
        ["formal_scalability/**/support_provenance.json"],
    )

    statistics = [
        str(python), str(ROOT / "scripts/run_typed_model_cache_formal_statistics.py"),
        "--protocol-path", str(protocol_path),
        "--input-root", str(execution_root),
        "--output-root", str(execution_root / "statistics"),
    ]
    run_phase(
        runner,
        "formal_statistics",
        [statistics],
        ["statistics/paired_statistics.json"],
    )

    integrity = [
        str(python), str(ROOT / "scripts/manage_typed_model_cache_formal_artifacts.py"),
        "--action", "integrity",
        "--protocol-path", str(protocol_path),
        "--input-root", str(execution_root),
        "--output-path", str(execution_root / "artifact_integrity_manifest.json"),
    ]
    gate = [
        str(python), str(ROOT / "scripts/manage_typed_model_cache_formal_artifacts.py"),
        "--action", "formal_gate",
        "--protocol-path", str(protocol_path),
        "--input-root", str(execution_root),
        "--output-path", str(execution_root / "formal_gate.json"),
    ]
    run_phase(
        runner,
        "formal_gate",
        [integrity, gate],
        ["artifact_integrity_manifest.json", "formal_gate.json"],
    )
    run_phase(runner, "complete_without_holdout", [], [])

    events = runner.events()
    ledger_report = validate_phase_ledger(events)
    freeze = json.loads((execution_root / "checkpoint_freeze.json").read_text(encoding="utf-8"))
    gate_payload = json.loads((execution_root / "formal_gate.json").read_text(encoding="utf-8"))
    selected = json.loads((execution_root / "dev_selection.json").read_text(encoding="utf-8"))
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in execution_root.glob("training/**/train_summary.json")
    ]
    sa_coefficients = {
        float(item["resolved_agent_config"].get("auxiliary_coef"))
        for item in summaries
        if item.get("agent_name") == "sa_ghmappo"
    }
    invalid_reused = any(
        INVALID_RUN == Path(row["checkpoint_path"]).resolve()
        or INVALID_RUN in Path(row["checkpoint_path"]).resolve().parents
        for row in freeze["frozen_checkpoints"]
    )
    summary = {
        "g14r3_non_formal_rehearsal_version": "1.0.0",
        "status": "pass" if gate_payload.get("passed") else "fail",
        "execution_mode": "non_formal_rehearsal",
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "phase_order": list(PHASE_ORDER),
        "completed_phase_order": [
            event["phase"] for event in events if event["status"] == "completed"
        ],
        "phase_ledger_validation": ledger_report,
        "agents": AGENTS,
        "domain_baseline": "cache_offload_drl",
        "seeds": SEEDS,
        "capacities_mb": list(CAPACITIES.values()),
        "training_cell_count": len(summaries),
        "checkpoint_frequency_updates": 4,
        "saved_checkpoint_update_indices": sorted(
            {index for item in summaries for index in item["saved_checkpoint_update_indices"]}
        ),
        "sa_auxiliary_coef": sorted(sa_coefficients),
        "dev_selector": {
            "real_consumer_executed": True,
            "candidate_count": len(json.loads((execution_root / "checkpoint_candidates.json").read_text(encoding="utf-8"))),
            "selected_count": len(selected["selected"]),
            "formal_or_holdout_used": selected["formal_or_holdout_used"],
        },
        "dev_selector_complete": True,
        "checkpoint_freeze": {
            "real_consumer_executed": True,
            "frozen_checkpoint_count": freeze["frozen_checkpoint_count"],
            "freeze_sha256": freeze["freeze_sha256"],
            "location_nonsemantic": True,
        },
        "checkpoint_freeze_complete": True,
        "cache_policy_executed": True,
        "controller_evaluation_executed": True,
        "ablation_support_executed": True,
        "robustness_support_executed": True,
        "scalability_support_executed": True,
        "statistics_executed": True,
        "artifact_integrity_executed": True,
        "non_formal_completeness_gate_passed": bool(gate_payload.get("passed")),
        "formal_training_count": 0,
        "formal_evaluation_count": 0,
        "formal": False,
        "holdout_opened": False,
        "holdout_used": False,
        "hidden_data_used": False,
        "paper_claims": [],
        "paper_claim": False,
        "invalid_g14c_v3_checkpoint_reused": invalid_reused,
        "old_run_resumed": False,
        "rehearsal_root": str(output_root),
        "clean_execution_worktree": str(ROOT),
        "external_data_root": str(data_root),
    }
    if summary["training_cell_count"] != 16:
        raise RuntimeError("rehearsal did not execute exactly 16 training cells")
    if sa_coefficients != {0.06}:
        raise RuntimeError("rehearsal SA coefficient is not exactly 0.06")
    if invalid_reused or summary["formal_training_count"] or summary["formal_evaluation_count"]:
        raise RuntimeError("rehearsal boundary violation")
    write_json(output_root / "rehearsal_summary.json", summary, create_only=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
