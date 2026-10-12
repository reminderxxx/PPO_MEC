"""Non-formal multi-window benchmark for the versioned physical VEC environment.

This entrypoint never reads historical checkpoint identities as physical ones.
It requires an explicit environment config and a frozen NGSIM window plan.
All agents execute through the same run_real_episode physical-environment path.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from src.agents.registry import ALGO_REGISTRY, checkpoint_required_agents, list_evaluable_agents
from src.data.model_catalog.adapter_catalog import AdapterCatalog, TYPED_MODEL_CACHE_PROFILE_ID
from src.evaluators.main_results_support import (
    build_selected_workflow_states, load_window_bundle, run_real_episode,
)
from src.metrics.physical_event_metrics import reduce_physical_events
from src.runtime.physical_env_config import (
    PHYSICAL_ENV_CONFIG_VERSION, load_physical_env_config, physical_env_config_identity,
)


BENCHMARK_SCHEMA_VERSION = "physical_env_benchmark_nonformal_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def load_frozen_window_plan(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("frozen window plan must be an object with selected_window_plan")
    if payload.get("mobility_source") not in (None, "ngsim"):
        raise ValueError("physical benchmark requires an NGSIM window plan")
    rows = payload.get("selected_window_plan")
    if not isinstance(rows, list) or not rows:
        raise ValueError("frozen window plan requires a nonempty selected_window_plan")
    required = {"window_id", "frame_offset", "window_length", "window_class"}
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"window plan row {index} lacks required identity fields")
        window_id = row["window_id"]
        offset = row["frame_offset"]
        length = row["window_length"]
        if not isinstance(window_id, str) or not window_id or window_id in seen:
            raise ValueError(f"invalid or duplicate window_id: {window_id}")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError(f"invalid frame_offset for {window_id}")
        if isinstance(length, bool) or not isinstance(length, int) or length < 2:
            raise ValueError(f"invalid window_length for {window_id}")
        seen.add(window_id)
        selected.append(dict(row))
    return selected, payload


def load_checkpoint_manifest(path: Path | None) -> dict[str, dict[str, dict[str, Any]]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("checkpoint manifest must be an object")
    agents = payload.get("agents", payload)
    if not isinstance(agents, dict):
        raise ValueError("checkpoint manifest agents must be an object")
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for agent_name, seed_map in agents.items():
        if str(agent_name).startswith("_"):
            continue
        if not isinstance(seed_map, dict):
            raise ValueError(f"checkpoint manifest agent {agent_name} must map seeds")
        normalized[str(agent_name)] = {}
        for seed, raw_entry in seed_map.items():
            entry = {"path": raw_entry} if isinstance(raw_entry, str) else raw_entry
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"]:
                raise ValueError(f"missing checkpoint path for {agent_name}/{seed}")
            checkpoint_path = Path(entry["path"])
            if not checkpoint_path.is_absolute():
                checkpoint_path = path.parent / checkpoint_path
            normalized[str(agent_name)][str(seed)] = {**entry, "path": str(checkpoint_path.resolve())}
    return normalized


def verify_physical_checkpoint(
    path: Path, *, agent_name: str, seed: int, config_sha256: str,
    manifest_entry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint missing: {path}")
    checksum = _sha256_file(path)
    if manifest_entry is not None and manifest_entry.get("sha256") not in (None, checksum):
        raise ValueError(f"checkpoint sha256 mismatch for {agent_name}/{seed}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("physical checkpoint payload must be an object")
    metadata = payload.get("training_metadata") or payload.get("checkpoint_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("physical checkpoint requires training_metadata")
    for field, expected in (
        ("environment_version", PHYSICAL_ENV_CONFIG_VERSION),
        ("physical_environment_config_sha256", config_sha256),
    ):
        if metadata.get(field) != expected:
            raise ValueError(f"physical checkpoint {field} mismatch for {agent_name}/{seed}")
        if field in payload and payload[field] != expected:
            raise ValueError(f"physical checkpoint top-level {field} mismatch for {agent_name}/{seed}")
        if manifest_entry is not None and field in manifest_entry and manifest_entry[field] != expected:
            raise ValueError(f"checkpoint manifest {field} mismatch for {agent_name}/{seed}")
    if metadata.get("agent_name") not in (None, agent_name):
        raise ValueError(f"checkpoint agent_name mismatch for {agent_name}/{seed}")
    if metadata.get("seed") not in (None, seed):
        raise ValueError(f"checkpoint seed mismatch for {agent_name}/{seed}")
    return {
        "path": str(path.resolve()), "sha256": checksum,
        "environment_version": PHYSICAL_ENV_CONFIG_VERSION,
        "physical_environment_config_sha256": config_sha256,
        "checkpoint_run_id": metadata.get("run_id"),
    }


def _checkpoint_bindings(
    agents: list[str], seeds: list[int], manifest: dict[str, dict[str, dict[str, Any]]],
    config_sha256: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    required = checkpoint_required_agents()
    bindings: dict[str, dict[str, dict[str, Any]]] = {}
    for agent_name in agents:
        bindings[agent_name] = {}
        for seed in seeds:
            if agent_name not in required:
                if agent_name in manifest and str(seed) in manifest[agent_name]:
                    raise ValueError(f"rule agent {agent_name} must not load a checkpoint")
                continue
            entry = manifest.get(agent_name, {}).get(str(seed))
            if entry is None:
                raise ValueError(f"missing physical checkpoint manifest entry for {agent_name}/{seed}")
            bindings[agent_name][str(seed)] = verify_physical_checkpoint(
                Path(entry["path"]), agent_name=agent_name, seed=seed,
                config_sha256=config_sha256, manifest_entry=entry,
            )
    return bindings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a non-formal physical VEC benchmark")
    parser.add_argument("--physical_env_config", required=True, type=Path)
    parser.add_argument("--window_plan_path", required=True, type=Path)
    parser.add_argument("--mobility_csv_path", required=True, type=Path)
    parser.add_argument("--workflow_csv_path", required=True, type=Path)
    parser.add_argument("--adapter_catalog_path", type=Path, default=ROOT_DIR / "src/data/model_catalog/typed_model_cache_controlled.json")
    parser.add_argument("--checkpoint_manifest_path", type=Path)
    parser.add_argument("--agents", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--max_mobility_rows", type=int, default=2500)
    parser.add_argument("--max_workflows", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--workflow_selector", choices=["ordered", "random"], default="ordered")
    parser.add_argument("--rsu_layout", default="auto_dominant_tight")
    parser.add_argument("--primary_vehicle_selection", choices=["stable_first", "handoff_pressure"], default="stable_first")
    parser.add_argument("--output_root", type=Path, default=ROOT_DIR / "artifacts/physical_env/nonformal_benchmark")
    return parser


def run_benchmark(args: argparse.Namespace) -> Path:
    if not args.agents or len(set(args.agents)) != len(args.agents):
        raise ValueError("agents must be nonempty and unique")
    if set(args.agents) - set(list_evaluable_agents()):
        raise ValueError(f"unknown or non-evaluable agents: {sorted(set(args.agents) - set(list_evaluable_agents()))}")
    for agent_name in args.agents:
        policy = ALGO_REGISTRY[agent_name].get("required_eviction_policy")
        if policy not in (None, "lru"):
            raise ValueError(f"physical cache currently supports LRU only; agent {agent_name} requires {policy}")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    if any(value <= 0 for value in (args.max_mobility_rows, args.max_workflows, args.max_steps, args.min_tasks, args.max_tasks)):
        raise ValueError("all workload limits must be positive")
    if args.min_tasks > args.max_tasks:
        raise ValueError("min_tasks must not exceed max_tasks")
    for path in (args.physical_env_config, args.window_plan_path, args.mobility_csv_path, args.workflow_csv_path, args.adapter_catalog_path):
        if not path.is_file():
            raise FileNotFoundError(f"required input missing: {path}")
    kwargs, config = load_physical_env_config(args.physical_env_config)
    config_sha = physical_env_config_identity(config)
    windows, window_plan = load_frozen_window_plan(args.window_plan_path)
    source_path = window_plan.get("mobility_source_path")
    if source_path and Path(source_path).resolve() != args.mobility_csv_path.resolve():
        raise ValueError("window plan mobility_source_path does not match --mobility_csv_path")
    catalog = AdapterCatalog.from_json(args.adapter_catalog_path)
    if catalog.model_cache_profile_id != TYPED_MODEL_CACHE_PROFILE_ID:
        raise ValueError("physical benchmark requires typed model-cache catalog")
    checkpoint_manifest = load_checkpoint_manifest(args.checkpoint_manifest_path)
    checkpoint_bindings = _checkpoint_bindings(args.agents, args.seeds, checkpoint_manifest, config_sha)
    input_sha = {
        "physical_env_config": _sha256_file(args.physical_env_config),
        "window_plan": _sha256_file(args.window_plan_path),
        "mobility_csv": _sha256_file(args.mobility_csv_path),
        "workflow_csv": _sha256_file(args.workflow_csv_path),
        "adapter_catalog": _sha256_file(args.adapter_catalog_path),
    }
    run_identity_payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "physical_environment_config_sha256": config_sha,
        "input_sha256": input_sha,
        "agents": args.agents, "seeds": args.seeds,
        "checkpoints": checkpoint_bindings,
        "windows": windows,
        "max_steps": args.max_steps,
        "max_workflows": args.max_workflows,
        "workflow_selector": args.workflow_selector,
    }
    run_identity = hashlib.sha256(json.dumps(run_identity_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    run_id = f"physical_nonformal_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{run_identity[:12]}"
    output_root = args.output_root / run_id
    if output_root.exists():
        raise FileExistsError(f"run output already exists: {output_root}")
    commit = _git_commit()
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        workflows = build_selected_workflow_states(
            workflow_csv_path=args.workflow_csv_path, max_workflows=args.max_workflows,
            workflow_selector=args.workflow_selector, min_tasks=args.min_tasks,
            max_tasks=args.max_tasks, random_seed=seed,
        )
        if not workflows:
            raise ValueError(f"no Alibaba workflows selected for seed={seed}")
        checkpoint_map = {
            agent_name: checkpoint_bindings[agent_name][str(seed)]["path"]
            for agent_name in args.agents if str(seed) in checkpoint_bindings[agent_name]
        }
        for window in windows:
            bundle = load_window_bundle(
                root_dir=ROOT_DIR, mobility_source="ngsim",
                mobility_csv_path=str(args.mobility_csv_path), max_mobility_rows=args.max_mobility_rows,
                rsu_layout=str(window.get("recommended_rsu_layout", args.rsu_layout)),
                frame_offset=window["frame_offset"], window_length=window["window_length"],
                random_seed=seed, expected_window_id=window["window_id"],
            )
            observed_id = bundle.rsu_metadata.get("window_id")
            if observed_id != window["window_id"]:
                raise ValueError(f"window identity mismatch: planned={window['window_id']}, observed={observed_id}")
            bundle.rsu_metadata["window_class"] = window["window_class"]
            bundle.rsu_metadata["window_rank"] = window.get("window_rank")
            for workflow in workflows:
                for agent_name in args.agents:
                    binding = checkpoint_bindings[agent_name].get(str(seed))
                    identity = {
                        "schema_version": BENCHMARK_SCHEMA_VERSION,
                        "evidence_level": "non_formal_diagnostic",
                        "run_id": run_id, "run_identity_sha256": run_identity,
                        "environment_version": PHYSICAL_ENV_CONFIG_VERSION,
                        "physical_environment_config_sha256": config_sha,
                        "git_commit": commit, "agent_name": agent_name, "seed": seed,
                        "window_id": window["window_id"], "workflow_id": workflow.workflow_id,
                        "checkpoint": binding,
                    }
                    cache_profile = None
                    if ALGO_REGISTRY[agent_name].get("required_eviction_policy") == "lru":
                        cache_profile = {
                            "model_cache_profile_id": TYPED_MODEL_CACHE_PROFILE_ID,
                            "enabled": True, "unit": "mb",
                            "capacity_mb": kwargs["cache_capacity_bytes_per_rsu"] / (1024 * 1024),
                            "eviction_policy": "lru",
                        }
                    summary = run_real_episode(
                        root_dir=ROOT_DIR, agent_name=agent_name, checkpoint_map=checkpoint_map,
                        workflow_state=workflow, workflow_source_path=str(args.workflow_csv_path),
                        mobility_bundle=bundle, seed=seed, max_steps=args.max_steps,
                        mobility_source="ngsim", primary_vehicle_selection=args.primary_vehicle_selection,
                        reward_positive_offset=0.0, run_metadata={
                            "script": "scripts/benchmark_physical_env.py", **identity,
                        },
                        adapter_catalog_override=catalog, cache_capacity_profile=cache_profile,
                        physical_env_config=kwargs,
                    )
                    trace = summary.get("physical_event_trace")
                    if not isinstance(trace, list):
                        raise ValueError("physical run_real_episode summary omitted physical_event_trace")
                    metrics = reduce_physical_events(trace).to_dict()
                    workflow_metrics = summary.get("workflow_physical_metrics")
                    if not isinstance(workflow_metrics, dict):
                        raise ValueError("physical run_real_episode omitted workflow_physical_metrics")
                    matching = [item for item in metrics["requests"] if item["request_id"] == workflow_metrics.get("request_id")]
                    if len(matching) != 1 or matching[0] != workflow_metrics:
                        raise ValueError("workflow physical metrics do not reconcile with event ledger")
                    unit_dir = output_root / "episodes" / window["window_id"] / workflow.workflow_id / agent_name / f"seed_{seed}"
                    _write_json(unit_dir / "events.json", trace)
                    _write_json(unit_dir / "physical_metrics.json", metrics)
                    _write_json(unit_dir / "identity.json", identity)
                    _write_json(unit_dir / "summary.json", summary)
                    row = {
                        "run_id": run_id,
                        "agent_name": agent_name,
                        "seed": seed,
                        "window_id": window["window_id"],
                        "workflow_id": workflow.workflow_id,
                        "episode_completed": bool(summary.get("episode_status", {}).get("completed", False)),
                        "episode_truncated": bool(summary.get("episode_status", {}).get("truncated", False)),
                        "episode_step_count": summary.get("run_info", {}).get("total_steps"),
                        "environment_version": PHYSICAL_ENV_CONFIG_VERSION,
                        "physical_environment_config_sha256": config_sha,
                        "run_identity_sha256": run_identity,
                        "checkpoint_sha256": binding["sha256"] if binding else None,
                        "physical_metrics_availability": workflow_metrics["availability"],
                        "physical_total_latency_s": workflow_metrics["latency_s"],
                        "physical_total_transfer_bytes": workflow_metrics["transfer_bytes"],
                        "all_requests_metrics_availability": metrics["availability"],
                        "event_trace_path": str(unit_dir / "events.json"),
                        "physical_metrics_path": str(unit_dir / "physical_metrics.json"),
                        "summary_path": str(unit_dir / "summary.json"),
                    }
                    rows.append(row)
    _write_json(output_root / "rows.json", rows)
    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with (output_root / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    _write_json(output_root / "run_manifest.json", {
        **run_identity_payload, "run_id": run_id, "run_identity_sha256": run_identity,
        "evidence_level": "non_formal_diagnostic", "formal": False,
        "paper_claim_ready": False, "git_commit": commit,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "physical_env_config": config, "episode_count": len(rows),
        "output_root": str(output_root),
    })
    return output_root


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(run_benchmark(args))


if __name__ == "__main__":
    main()
