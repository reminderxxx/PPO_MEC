"""Generate the controlled G07 validation artifact bundle (never formal/hidden)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.core.cache_eviction import build_eviction_policy
from src.evaluators.cache_baseline_fairness import (
    build_manifest,
    full_manifest_sha256,
    semantic_protocol_sha256,
    sha256_file,
    validate_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate controlled G07 validation artifacts")
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def reseal(payload: dict) -> dict:
    semantic = semantic_protocol_sha256(payload)
    payload["identity"]["manifest_id"] = f"cbfm-{semantic[:16]}"
    payload["hashes"]["semantic_protocol_sha256"] = semantic
    payload["hashes"]["full_manifest_sha256"] = full_manifest_sha256(payload)
    return payload


def manifest_kwargs(output_dir: Path) -> dict:
    return {
        "root": ROOT,
        "mobility_path": ROOT / "data/raw/mobility/ngsim/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv",
        "workflow_path": ROOT / "data/raw/workflow/alibaba2018/batch_task.csv",
        "window_plan_path": ROOT / "configs/experiment/cache_baseline_fairness_g07_smoke_window_plan.json",
        "catalog_path": ROOT / "src/data/model_catalog/sample_model_catalog.json",
        "seeds": [7], "max_workflows": 1, "workflow_selector": "ordered",
        "min_tasks": 5, "max_tasks": 20, "max_steps": 1,
        "max_mobility_rows": 2500, "primary_vehicle_selection": "stable_first",
        "capacity_unit": "adapter_slots", "capacity_value": 3,
        "output_root": str(output_dir / "controlled_benchmark"), "evaluation_unit_limit": 1,
    }


def random_plan(seed: int) -> list[list[str]]:
    policy = build_eviction_policy("random", seed=seed)
    cache = ["a", "b", "c"]
    policy.reset()
    policy.reset(rsu_id="r", initial_resident_ids=cache)
    plans = []
    for step, incoming in enumerate(("d", "e", "f"), 1):
        plan = policy.plan_victims(
            rsu_id="r", resident_ids=cache,
            resident_sizes={item: 1.0 for item in cache}, required_free_capacity=1.0,
            protected_object_id=incoming, capacity_unit="adapter_slots", current_step=step,
        )
        victims = list(plan.ordered_victim_ids)
        plans.append(victims)
        for victim in victims:
            cache.remove(victim)
            policy.on_eviction(rsu_id="r", object_id=victim, current_step=step)
        cache.append(incoming)
        policy.on_admission(rsu_id="r", object_id=incoming, current_step=step)
    return plans


def main() -> None:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "artifacts/analysis" / f"cache_baseline_fairness_manifest_validation_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    first = build_manifest(**manifest_kwargs(output_dir), created_at="2026-08-18T00:00:00Z")
    second = build_manifest(**manifest_kwargs(output_dir), created_at="2026-08-18T00:01:00Z")
    write_json(output_dir / "cache_baseline_fairness_manifest.json", first)
    validation = validate_manifest(first, root=ROOT, check_files=True)
    write_json(output_dir / "validation_report.json", validation)
    write_json(output_dir / "pairwise_protocol_diff.json", validation["pairwise_protocol_diff"])
    write_json(output_dir / "semantic_hash_reproducibility.json", {
        "first_semantic_protocol_sha256": first["hashes"]["semantic_protocol_sha256"],
        "second_semantic_protocol_sha256": second["hashes"]["semantic_protocol_sha256"],
        "semantic_hash_stable": first["hashes"]["semantic_protocol_sha256"] == second["hashes"]["semantic_protocol_sha256"],
        "created_at_changed": first["identity"]["created_at"] != second["identity"]["created_at"],
        "full_hash_changed": first["hashes"]["full_manifest_sha256"] != second["hashes"]["full_manifest_sha256"],
    })
    cases = {}
    mutations = {}
    capacity = deepcopy(first)
    capacity["baseline_matrix"][0]["agent_specific_overrides"] = {"capacity": 4}
    mutations["capacity_drift"] = reseal(capacity)
    seed = deepcopy(first)
    seed["seed_plan"]["per_run"][0]["environment_seed"] = 99
    mutations["seed_drift"] = reseal(seed)
    catalog = deepcopy(first)
    target = next(item for item in catalog["dataset_provenance"]["inputs"] if item["logical_dataset_id"] == "ppo_mec_sample_adapter_catalog")
    target["sha256"] = "0" * 64
    mutations["catalog_hash_drift"] = reseal(catalog)
    admission = deepcopy(first)
    admission["baseline_matrix"][2]["admission_control_identity"] = "drifted_control"
    mutations["admission_control_drift"] = reseal(admission)
    for name, payload in mutations.items():
        case_report = validate_manifest(payload, root=ROOT, check_files=True)
        cases[name] = {"status": case_report["status"], "errors": case_report["errors"]}
    write_json(output_dir / "negative_validation_cases.json", cases)
    benchmark_root = output_dir / "controlled_benchmark"
    command = [
        str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/benchmark_main_results.py"),
        "--agents", *["reactive_lru", "reactive_fifo", "reactive_lfu", "reactive_aging_lfu", "reactive_random"],
        "--cache_baseline_fairness_manifest_path", str(output_dir / "cache_baseline_fairness_manifest.json"),
        "--seeds", "7", "--max_mobility_rows", "2500", "--max_workflows", "1", "--max_steps", "1",
        "--classical_cache_slots", "3", "--workflow_selector", "ordered", "--rsu_layout", "auto_dominant_tight",
        "--window_plan_path", str(ROOT / "configs/experiment/cache_baseline_fairness_g07_smoke_window_plan.json"),
        "--primary_vehicle_selection", "stable_first", "--min_tasks", "5", "--max_tasks", "20",
        "--reward_positive_offset", "0.0", "--output_root", str(benchmark_root),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    (output_dir / "resolved_command.log").write_text(" ".join(command) + "\n\n" + completed.stdout, encoding="utf-8")
    run_dirs = sorted(path for path in benchmark_root.iterdir() if path.is_dir())
    if len(run_dirs) != 1:
        raise RuntimeError(f"expected one controlled benchmark run, got {run_dirs}")
    runtime_audit = json.loads((run_dirs[0] / "fairness_runtime_audit.json").read_text(encoding="utf-8"))
    write_json(output_dir / "observed_request_fingerprints.json", runtime_audit["observed_request_fingerprints"])
    same_seed_a = random_plan(7)
    same_seed_b = random_plan(7)
    other_seed = random_plan(13)
    write_json(output_dir / "random_seed_reproducibility.json", {
        "seed_derivation": "policy_seed_equals_benchmark_run_seed",
        "seed_7_first": same_seed_a, "seed_7_repeat": same_seed_b, "seed_13": other_seed,
        "same_seed_reproducible": same_seed_a == same_seed_b,
        "different_seed_diversity": same_seed_a != other_seed,
    })
    files = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name != "artifact_integrity_manifest.json")
    write_json(output_dir / "artifact_integrity_manifest.json", {
        "integrity_manifest_version": "1.0.0",
        "status": "pass",
        "files": [{"path": path.relative_to(output_dir).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files],
    })
    print(json.dumps({
        "output_dir": str(output_dir), "validation_status": validation["status"],
        "pairwise_comparison_count": validation["pairwise_protocol_diff"]["comparison_count"],
        "negative_cases_all_failed": all(item["status"] == "fail" for item in cases.values()),
        "semantic_hash_stable": first["hashes"]["semantic_protocol_sha256"] == second["hashes"]["semantic_protocol_sha256"],
        "controlled_benchmark_run_dir": str(run_dirs[0]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
