#!/usr/bin/env python3
"""Evaluate fixed CRDCM checkpoints on the frozen observed-data paired units."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_mechanism_frozen_checkpoints import _load_workflows
from scripts.run_crdcm_decision_diagnostic import (
    _aggregate,
    _bind_workflow,
    _canonical_sha256,
    _metric_row,
    _sha256,
    _write_csv,
    _write_json,
)
from scripts.run_mechanism_factorial_pilot import build_factorial_catalog
from src.agents.registry import build_agent
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.evaluators.main_results_support import (
    build_episode_formal_request_exposure,
    load_window_bundle,
    run_real_episode,
)


EVALUATION_VERSION = "crdcm_paired_development_evaluation_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _load_checkpoint_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("checkpoint_manifest_version") != "crdcm_checkpoint_manifest_v2":
        raise ValueError("unsupported CRDCM checkpoint manifest")
    entries = list(payload.get("entries", []))
    if not entries:
        raise ValueError("empty CRDCM checkpoint manifest")
    for entry in entries:
        checkpoint = Path(str(entry["checkpoint_path"]))
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if _sha256(checkpoint) != str(entry["checkpoint_sha256"]):
            raise ValueError(f"checkpoint identity mismatch: {checkpoint}")
    return payload


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("study_version") != "crdcm_performance_matrix_v2":
        raise ValueError("unexpected CRDCM matrix version")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()

    checkpoint_manifest_path = args.checkpoint_manifest.resolve()
    checkpoint_manifest = _load_checkpoint_manifest(checkpoint_manifest_path)
    data_root = args.data_root.resolve()
    data = dict(config["training_data"])
    mobility_path = (data_root / str(data["mobility_csv_path"])).resolve()
    workflow_path = (data_root / str(data["workflow_csv_path"])).resolve()
    for role, path in (("mobility", mobility_path), ("workflow", workflow_path)):
        expected = dict(config["expected_sources"][role])
        if path.stat().st_size != int(expected["size_bytes"]):
            raise ValueError(f"{role} source size mismatch")
        if _sha256(path) != str(expected["sha256"]):
            raise ValueError(f"{role} source SHA-256 mismatch")

    evaluation = dict(config["evaluation"])
    all_scenarios = [dict(item) for item in evaluation["scenarios"]]
    all_pairs = [dict(item) for item in evaluation["pairs"]]
    scenarios = all_scenarios[:1] if args.smoke else all_scenarios
    pairs = all_pairs[:1] if args.smoke else all_pairs
    if not args.smoke:
        expected_learned = int(config["budget"]["learned_checkpoint_count"])
        if len(checkpoint_manifest["entries"]) != expected_learned:
            raise ValueError("full evaluation requires all frozen learned checkpoints")

    window_plan_path = (ROOT / str(config["training_window_plan"])).resolve()
    window_plan = json.loads(window_plan_path.read_text(encoding="utf-8"))
    windows = list(window_plan["selected_window_plan"])
    workflow_ids = [str(item["workflow_id"]) for item in pairs]
    workflows = {
        str(item.workflow_id): item
        for item in _load_workflows(workflow_path, workflow_ids)
    }
    source_catalog = AdapterCatalog.from_json(
        ROOT / "src/data/model_catalog/typed_model_cache_controlled.json"
    )
    bundles: dict[int, Any] = {}
    for pair in pairs:
        index = int(pair["window_index"])
        window = dict(windows[index])
        bundle = load_window_bundle(
            root_dir=ROOT,
            mobility_source="ngsim",
            mobility_csv_path=str(mobility_path),
            lust_scenario_root="",
            max_mobility_rows=int(data["max_mobility_rows"]),
            rsu_layout=str(window["recommended_rsu_layout"]),
            frame_offset=int(window["frame_offset"]),
            window_length=int(window["window_length"]),
            random_seed=int(config["seeds"][0]),
        )
        bundle.rsu_metadata.update(
            window_id=window["window_id"],
            window_rank=window["window_rank"],
            window_class=window["window_class"],
        )
        bundles[index] = bundle

    rows: list[dict[str, Any]] = []
    checkpoint_hashes_before = {
        str(entry["checkpoint_path"]): _sha256(Path(str(entry["checkpoint_path"])))
        for entry in checkpoint_manifest["entries"]
    }
    episode_count = 0
    for scenario in scenarios:
        scenario_id = str(scenario["scenario_id"])
        catalog = build_factorial_catalog(
            source_catalog,
            base_sharing_enabled=True,
            initial_adapter_id=str(scenario["initial_adapter_id"]),
        )
        capacity_profile = {
            "model_cache_profile_id": "typed_base_adapter_state_v1",
            "enabled": True,
            "unit": "mb",
            "capacity_mb": float(scenario["capacity_mb"]),
            "count_base_model_separately": True,
            "eviction_policy": "lru",
            "eviction_policy_seed": int(config["seeds"][0]),
            "telemetry_enabled": True,
        }
        mechanism_profile = {
            "mechanism_factorial_profile_version": "controlled_mechanism_factorial_v1",
            "profile_id": f"{scenario_id}_crdcm_observation_v1",
            "base_sharing_enabled": True,
            "workflow_state_migration_enabled": bool(scenario["migration_enabled"]),
            "migration_disabled_fallback": "cold_restart_one_request",
            "crdcm_observation_enabled": True,
            "crdcm_observation_contract_version": "crdcm_observation_v1",
        }
        runtime_contract = {
            "model_cache_profile": "typed_base_adapter_state_v1",
            "typed_catalog_fingerprint": catalog.canonical_fingerprint(),
            "runtime_contract_sha256": _canonical_sha256(
                {"capacity": capacity_profile, "mechanism": mechanism_profile}
            ),
            "cache_event_schema_version": "1.3.0",
            "cache_efficiency_metrics_contract_version": "1.1.0",
        }
        for pair_index, pair in enumerate(pairs):
            window_index = int(pair["window_index"])
            bundle = bundles[window_index]
            workflow = _bind_workflow(
                workflows[str(pair["workflow_id"])],
                catalog,
                str(scenario["workflow_binding"]),
            )
            unit_id = (
                f"{scenario_id}/{bundle.rsu_metadata['window_id']}/"
                f"{workflow.workflow_id}"
            )
            exposure = build_episode_formal_request_exposure(
                workflow_state=workflow,
                mobility_bundle=bundle,
                adapter_catalog=catalog,
                max_steps=int(evaluation["max_steps"]),
                mobility_source="ngsim",
                primary_vehicle_selection="handoff_pressure",
                cache_capacity_profile=capacity_profile,
                evaluation_unit={
                    "evaluation_unit_id": unit_id,
                    "workflow_id": workflow.workflow_id,
                    "window_id": bundle.rsu_metadata["window_id"],
                },
                source_provenance={
                    "study_version": config["study_version"],
                    "evaluation_version": EVALUATION_VERSION,
                    "data_role": evaluation["data_role"],
                },
            )
            for entry in checkpoint_manifest["entries"]:
                condition_id = str(entry["condition_id"])
                agent_name = str(entry["agent_name"])
                agent = build_agent(
                    agent_name,
                    deterministic_action=True,
                    random_seed=int(entry["seed"]),
                    crdcm_feature_mode=str(entry["crdcm_feature_mode"]),
                )
                agent.load(str(entry["checkpoint_path"]))
                summary = run_real_episode(
                    root_dir=ROOT,
                    agent_name=agent_name,
                    checkpoint_map={},
                    workflow_state=workflow,
                    workflow_source_path=str(workflow_path),
                    mobility_bundle=bundle,
                    seed=int(entry["seed"]),
                    max_steps=int(evaluation["max_steps"]),
                    mobility_source="ngsim",
                    primary_vehicle_selection="handoff_pressure",
                    reward_positive_offset=0.0,
                    run_metadata={
                        "script": "scripts/evaluate_crdcm_frozen_checkpoints.py",
                        "study_version": config["study_version"],
                        "evaluation_version": EVALUATION_VERSION,
                        "condition_id": condition_id,
                        "seed": int(entry["seed"]),
                        "scenario_id": scenario_id,
                        "unit_id": unit_id,
                        "training": False,
                        "smoke": bool(args.smoke),
                        "claim_boundary": config["claim_boundary"],
                    },
                    adapter_catalog_override=catalog,
                    workflow_state_override=workflow,
                    cache_capacity_profile=capacity_profile,
                    model_cache_runtime_contract=runtime_contract,
                    formal_request_exposure_trace=exposure,
                    mechanism_profile=mechanism_profile,
                    agent_override=agent,
                )
                controller = f"{condition_id}/seed_{entry['seed']}"
                target = output / "episodes" / condition_id / f"seed_{entry['seed']}" / f"{scenario_id}_{pair_index}.json"
                _write_json(target, summary)
                rows.append(
                    {
                        **_metric_row(
                            summary=summary,
                            controller=controller,
                            scenario_id=scenario_id,
                            unit_id=unit_id,
                        ),
                        "condition_id": condition_id,
                        "seed": int(entry["seed"]),
                        "checkpoint_sha256": str(entry["checkpoint_sha256"]),
                    }
                )
                episode_count += 1

            heuristic = build_agent("crdcm_critical_path_heuristic")
            heuristic_summary = run_real_episode(
                root_dir=ROOT,
                agent_name="crdcm_critical_path_heuristic",
                checkpoint_map={},
                workflow_state=workflow,
                workflow_source_path=str(workflow_path),
                mobility_bundle=bundle,
                seed=int(config["seeds"][0]),
                max_steps=int(evaluation["max_steps"]),
                mobility_source="ngsim",
                primary_vehicle_selection="handoff_pressure",
                reward_positive_offset=0.0,
                run_metadata={
                    "script": "scripts/evaluate_crdcm_frozen_checkpoints.py",
                    "study_version": config["study_version"],
                    "evaluation_version": EVALUATION_VERSION,
                    "condition_id": "crdcm_critical_path_heuristic",
                    "scenario_id": scenario_id,
                    "unit_id": unit_id,
                    "training": False,
                    "smoke": bool(args.smoke),
                    "claim_boundary": config["claim_boundary"],
                },
                adapter_catalog_override=catalog,
                workflow_state_override=workflow,
                cache_capacity_profile=capacity_profile,
                model_cache_runtime_contract=runtime_contract,
                formal_request_exposure_trace=exposure,
                mechanism_profile=mechanism_profile,
                agent_override=heuristic,
            )
            heuristic_target = output / "episodes" / "crdcm_critical_path_heuristic" / f"{scenario_id}_{pair_index}.json"
            _write_json(heuristic_target, heuristic_summary)
            rows.append(
                {
                    **_metric_row(
                        summary=heuristic_summary,
                        controller="crdcm_critical_path_heuristic",
                        scenario_id=scenario_id,
                        unit_id=unit_id,
                    ),
                    "condition_id": "crdcm_critical_path_heuristic",
                    "seed": None,
                    "checkpoint_sha256": None,
                }
            )
            episode_count += 1

    checkpoint_hashes_after = {
        path: _sha256(Path(path)) for path in checkpoint_hashes_before
    }
    if checkpoint_hashes_before != checkpoint_hashes_after:
        raise RuntimeError("evaluation mutated a frozen checkpoint")
    expected_count = (
        len(checkpoint_manifest["entries"]) + 1
    ) * len(scenarios) * len(pairs)
    if episode_count != expected_count:
        raise RuntimeError("evaluation episode accounting mismatch")
    _write_csv(output / "paired_episode_rows.csv", rows)
    _write_json(output / "aggregate.json", _aggregate(rows))
    _write_json(
        output / "evaluation_receipt.json",
        {
            "evaluation_version": EVALUATION_VERSION,
            "status": "SUCCEEDED",
            "smoke": bool(args.smoke),
            "episode_count": episode_count,
            "environment_step_cap": episode_count * int(evaluation["max_steps"]),
            "checkpoint_manifest_path": str(checkpoint_manifest_path),
            "checkpoint_manifest_sha256": _sha256(checkpoint_manifest_path),
            "checkpoint_hashes_before": checkpoint_hashes_before,
            "checkpoint_hashes_after": checkpoint_hashes_after,
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "data_role": evaluation["data_role"],
            "formal": False,
            "holdout": False,
            "claim_boundary": config["claim_boundary"],
            "elapsed_seconds": round(time.time() - started, 3),
        },
    )


if __name__ == "__main__":
    main()
