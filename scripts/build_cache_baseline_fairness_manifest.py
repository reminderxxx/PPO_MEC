"""Build and immediately validate a G07 cache-baseline fairness manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.cache_baseline_fairness import build_manifest, validate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-grade cache baseline fairness manifest")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--mobility_path", default=str(ROOT / "data/raw/mobility/ngsim/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv"))
    parser.add_argument("--workflow_path", default=str(ROOT / "data/raw/workflow/alibaba2018/batch_task.csv"))
    parser.add_argument("--window_plan_path", required=True)
    parser.add_argument("--catalog_path", default=str(ROOT / "src/data/model_catalog/sample_model_catalog.json"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--max_workflows", type=int, default=1)
    parser.add_argument("--workflow_selector", default="ordered")
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--max_mobility_rows", type=int, default=2500)
    parser.add_argument("--primary_vehicle_selection", choices=["stable_first", "handoff_pressure"], default="stable_first")
    parser.add_argument("--capacity_unit", choices=["adapter_slots", "mb"], default="adapter_slots")
    parser.add_argument("--capacity_value", type=float, default=3.0)
    parser.add_argument("--artifact_output_root", default="artifacts/analysis/cache_baseline_fairness_runtime")
    parser.add_argument("--evaluation_unit_limit", type=int, default=1)
    parser.add_argument("--controller_agents", nargs="*", default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite existing manifest: {output_path}")
    manifest = build_manifest(
        root=ROOT,
        mobility_path=args.mobility_path,
        workflow_path=args.workflow_path,
        window_plan_path=args.window_plan_path,
        catalog_path=args.catalog_path,
        seeds=args.seeds,
        max_workflows=args.max_workflows,
        workflow_selector=args.workflow_selector,
        min_tasks=args.min_tasks,
        max_tasks=args.max_tasks,
        max_steps=args.max_steps,
        max_mobility_rows=args.max_mobility_rows,
        primary_vehicle_selection=args.primary_vehicle_selection,
        capacity_unit=args.capacity_unit,
        capacity_value=args.capacity_value,
        output_root=args.artifact_output_root,
        evaluation_unit_limit=args.evaluation_unit_limit,
        controller_agents=args.controller_agents,
    )
    report = validate_manifest(manifest, root=ROOT, check_files=True)
    if report["status"] != "pass":
        raise ValueError("manifest validation failed: " + "; ".join(report["errors"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest_path": str(output_path),
        "manifest_id": manifest["identity"]["manifest_id"],
        "semantic_protocol_sha256": manifest["hashes"]["semantic_protocol_sha256"],
        "full_manifest_sha256": manifest["hashes"]["full_manifest_sha256"],
        "validation_status": report["status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
