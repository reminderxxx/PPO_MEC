"""Run an explicitly cross-source, eight-vehicle/four-RSU diagnostic replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.mobility.ngsim_provider import NGSIMProvider
from src.data.workflow.workflow_dataset_builder import WorkflowDatasetBuilder
from src.envs.core.multi_request_replay_env import (
    MultiRequestReplayEnv,
    ReplayCosts,
    select_continuous_window,
)
from src.evaluators.real_sample_support import (
    build_sample_rsus,
    discover_ngsim_csv,
    summarize_mobility_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobility_csv_path", type=Path, default=None)
    parser.add_argument(
        "--workflow_csv_path",
        type=Path,
        default=ROOT_DIR / "data/raw/workflow/alibaba2018/batch_task.csv",
    )
    parser.add_argument("--max_mobility_rows", type=int, default=250_000)
    parser.add_argument("--window_length", type=int, default=120)
    parser.add_argument("--vehicle_count", type=int, default=8)
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--rsu_coverage_factor", type=float, default=0.75)
    parser.add_argument("--compute_slots_per_rsu", type=int, default=1)
    parser.add_argument("--adapter_slots_per_rsu", type=int, default=2)
    parser.add_argument("--link_slots_per_rsu", type=int, default=1)
    parser.add_argument("--adapter_fetch_frames", type=int, default=1)
    parser.add_argument("--handoff_state_frames", type=int, default=1)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT_DIR / "artifacts/diagnostics/eight_vehicle_simulation",
    )
    return parser.parse_args()


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    args = parse_args()
    if args.vehicle_count != 8:
        raise ValueError("This diagnostic entrypoint requires exactly eight vehicles")
    mobility_path = args.mobility_csv_path or discover_ngsim_csv(ROOT_DIR)
    if not args.workflow_csv_path.exists():
        raise FileNotFoundError(args.workflow_csv_path)
    frames = NGSIMProvider(
        csv_path=mobility_path, max_rows=args.max_mobility_rows
    ).get_loaded_frames()
    selected_frames, vehicle_ids, frame_offset = select_continuous_window(
        frames, vehicle_count=8, window_length=args.window_length
    )
    workflows = WorkflowDatasetBuilder().build_selected_alibaba_workflow_states(
        csv_path=args.workflow_csv_path,
        max_workflows=8,
        workflow_selector="ordered",
        min_tasks=args.min_tasks,
        max_tasks=args.max_tasks,
        adapter_assignment_profile="legacy_batch_type",
    )
    if len(workflows) != 8:
        raise RuntimeError(f"expected eight distinct Alibaba DAGs, received {len(workflows)}")
    if args.rsu_coverage_factor <= 0:
        raise ValueError("rsu_coverage_factor must be positive")
    dominant_axis = summarize_mobility_frames(selected_frames)["dominant_axis"]
    rsu_layout = (
        f"custom:axis={dominant_axis},count=4,"
        f"coverage_factor={args.rsu_coverage_factor}"
    )
    rsus, rsu_metadata = build_sample_rsus(frames=selected_frames, rsu_layout=rsu_layout)
    if len(rsus) != 4:
        raise RuntimeError(f"expected four simulated RSUs, received {len(rsus)}")
    assignment = dict(zip(vehicle_ids, workflows))
    costs = ReplayCosts(
        compute_slots_per_rsu=args.compute_slots_per_rsu,
        adapter_slots_per_rsu=args.adapter_slots_per_rsu,
        link_slots_per_rsu=args.link_slots_per_rsu,
        adapter_fetch_frames=args.adapter_fetch_frames,
        handoff_state_frames=args.handoff_state_frames,
    )
    env = MultiRequestReplayEnv(
        frames=selected_frames,
        vehicle_workflows=assignment,
        rsu_states=rsus,
        costs=costs,
    )
    env.reset()
    done = False
    total_reward = 0.0
    while not done:
        _, reward, done, _ = env.step()  # nearest covered RSU baseline
        total_reward += reward

    frame_identity = [{
        "time_index": frame["time_index"],
        "source_segment_id": frame.get("source_segment_id"),
        "vehicles": [vehicle.to_dict() for vehicle in frame["vehicles"]],
    } for frame in selected_frames]
    workflow_graphs = {
        vehicle_id: workflow.to_dict() for vehicle_id, workflow in assignment.items()
    }
    manifest = {
        "experiment_type": "cross_source_trace_driven_simulation_diagnostic",
        "policy": "nearest_covered_rsu",
        "mobility_source": "NGSIM",
        "workflow_source": "Alibaba batch_task",
        "joint_vehicle_request_log": False,
        "simultaneous_arrival_assumed": True,
        "physical_rsus": False,
        "model_inference_executed": False,
        "rsu_and_link_costs_measured": False,
        "rsu_layout_uses_full_window_geometry": True,
        "service_frames_per_node": 1,
        "input_output_bytes_consumed": False,
        "mobility_csv_path": str(mobility_path.resolve()),
        "workflow_csv_path": str(args.workflow_csv_path.resolve()),
        "max_mobility_rows": args.max_mobility_rows,
        "frame_offset": frame_offset,
        "window_length": len(selected_frames),
        "source_segment_id": selected_frames[0].get("source_segment_id"),
        "time_index_start": selected_frames[0]["time_index"],
        "time_index_end": selected_frames[-1]["time_index"],
        "selected_frame_sha256": _fingerprint(frame_identity),
        "workflow_assignment": {
            vehicle_id: workflow.workflow_id for vehicle_id, workflow in assignment.items()
        },
        "workflow_graphs": workflow_graphs,
        "workflow_assignment_sha256": _fingerprint(workflow_graphs),
        "rsu_layout": [rsu.to_dict() for rsu in rsus],
        "rsu_layout_metadata": rsu_metadata,
        "simulated_costs": vars(costs),
    }
    summary = {**env.summary(), "total_reward_diagnostic": round(total_reward, 6)}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "request_events.jsonl").open("w", encoding="utf-8") as handle:
        for event in env.events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(json.dumps({"output_dir": str(output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
