"""Freeze the outcome-blind workload manifest for the calibrated pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_workflows(path: Path, node_min: int, node_max: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not node_min <= int(row["num_tasks"]) <= node_max:
                continue
            if any(
                not math.isfinite(float(node.get("duration_vec", 0) or 0))
                or not math.isfinite(float(node.get("plan_mem", 0) or 0))
                for node in row["nodes"]
            ):
                continue
            row["topology_class"] = (
                "branching"
                if int(row["num_edges"]) >= int(row["num_tasks"])
                or len(row.get("roots", [])) >= 3
                else "simple"
            )
            rows.append(row)
    simple = sorted((row for row in rows if row["topology_class"] == "simple"), key=lambda row: row["workflow_id"])
    branching = sorted((row for row in rows if row["topology_class"] == "branching"), key=lambda row: row["workflow_id"])
    selected: list[dict[str, Any]] = []
    for index in range(max(len(simple), len(branching))):
        if index < len(simple):
            selected.append(simple[index])
        if index < len(branching):
            selected.append(branching[index])
    return selected


def _topological_order(nodes: list[dict[str, Any]]) -> list[str]:
    remaining = {str(node["task_id"]): {str(item) for item in node.get("parents", [])} for node in nodes}
    order: list[str] = []
    while remaining:
        ready = sorted(node_id for node_id, parents in remaining.items() if parents.issubset(set(order)))
        if not ready:
            raise ValueError("workflow is not a DAG")
        for node_id in ready:
            order.append(node_id)
            remaining.pop(node_id)
    return order


def _select_windows(plan: list[dict[str, Any]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [plan[index] for index in list(spec["low_indices"]) + list(spec["high_indices"])]


def _bundle(adapter_id: str, config: dict[str, Any]) -> list[str]:
    return list(config["adapter_to_bundle"][adapter_id])


def _build_instance(
    *,
    split: str,
    ordinal: int,
    window: dict[str, Any],
    workflow: dict[str, Any],
    template: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    sharing = template["sharing"]
    nodes_raw = workflow["nodes"]
    duration_reference = max(median(float(node["duration_vec"]) for node in nodes_raw), 1.0)
    execution_order = _topological_order(nodes_raw)
    successors: dict[str, list[str]] = {str(node["task_id"]): [] for node in nodes_raw}
    edges: list[list[str]] = []
    for node in nodes_raw:
        target = str(node["task_id"])
        for parent in node.get("parents", []):
            source = str(parent)
            edges.append([source, target])
            successors[source].append(target)
    measured_input = int(config["measured_bytes"]["input"])
    measured_state = int(config["measured_bytes"]["state_package"])
    compute_reference = float(config["measured_time_seconds"]["node_compute_reference"])
    nodes: list[dict[str, Any]] = []
    for node in nodes_raw:
        node_id = str(node["task_id"])
        task_name = str(node["task_name"])
        small_adapter = task_name.startswith("M")
        adapter_id = (
            "helmet_shared"
            if small_adapter and sharing == "shared_base"
            else "helmet_distinct"
            if small_adapter
            else "alpr"
        )
        plan_mem = float(node.get("plan_mem", 0.2) or 0.2)
        input_scale = min(max(plan_mem / 0.2, 0.5), 3.0)
        compute_scale = min(max(float(node["duration_vec"]) / duration_reference, 0.4), 2.5)
        state_scale = int(template["state_scale"])
        parents = [str(item) for item in node.get("parents", [])]
        state_bytes = int(round(measured_state * state_scale * (1.0 + 0.25 * len(parents))))
        base_id = config["adapter_to_bundle"][adapter_id][0]
        nodes.append(
            {
                "node_id": node_id,
                "node_name": task_name,
                "required_base_model": base_id,
                "required_adapter": adapter_id,
                "input_size": int(round(measured_input * input_scale)),
                "output_size": state_bytes,
                "input_bytes": int(round(measured_input * input_scale)),
                "state_bytes": state_bytes,
                "compute_seconds": round(compute_reference * compute_scale, 6),
                "predecessors": parents,
                "successors": successors[node_id],
                "trace_duration_raw": node["duration_raw"],
                "trace_plan_mem": plan_mem,
            }
        )
    pressure = "high" if int(window["estimated_handoff_count"]) > 0 else "low"
    block = 2 if pressure == "high" else max(3, len(nodes) // 2)
    rsu_ids = ["rsu_0", "rsu_1", "rsu_2"]
    rsu_sequence = [rsu_ids[(index // block) % len(rsu_ids)] for index in range(len(nodes))]
    actual_mbps = float(template.get("actual_mbps", config["link"]["mbps"]))
    estimated_mbps = float(template.get("estimated_mbps", config["link"]["mbps"]))
    prediction_quality = str(template.get("prediction_quality", "matched"))
    prediction_confidence = float(template.get("prediction_confidence", 0.9))
    prediction_uncertainty = float(template.get("prediction_uncertainty", 0.1))
    capacity_key = f"{'shared' if sharing == 'shared_base' else 'distinct'}_{template['capacity']}"
    capacity = int(config["capacity_bytes"][capacity_key])
    first_adapter = next(node["required_adapter"] for node_id in execution_order for node in nodes if node["node_id"] == node_id)
    alternate = "helmet_shared" if sharing == "shared_base" else "helmet_distinct"
    if first_adapter == alternate:
        alternate = "alpr"
    initial = {
        "rsu_0": _bundle(first_adapter, config),
        "rsu_1": [],
        "rsu_2": [],
    }
    if template["initial_target"] == "competitor":
        initial["rsu_1"] = _bundle(alternate, config)
        initial["rsu_2"] = _bundle(alternate, config)
    elif template["initial_target"] == "all_ready":
        ready = sorted(set(_bundle("alpr", config) + _bundle(alternate, config)))
        initial = {rsu_id: ready for rsu_id in rsu_ids}
    baseline_compute = sum(float(node["compute_seconds"]) for node in nodes)
    deadline = baseline_compute * (1.65 if pressure == "high" else 2.25) + 12.0
    return {
        "design_id": f"{split}_{ordinal:02d}",
        "split": split,
        "window_id": window["window_id"],
        "workflow_id": workflow["workflow_id"],
        "source_interval": {
            "source_segment_id": window["source_segment_id"],
            "time_index_start": window["time_index_start"],
            "time_index_end": window["time_index_end"],
            "frame_offset": window["frame_offset"],
            "window_length": window["window_length"],
        },
        "trace_features": {
            "estimated_handoff_count": window["estimated_handoff_count"],
            "active_vehicle_count_mean": window["active_vehicle_count_mean"],
            "mean_speed_proxy": min(max(float(window["active_vehicle_count_mean"]) / 5.0, 5.0), 35.0),
            "handoff_pressure": pressure,
        },
        "workflow_features": {
            "num_tasks": workflow["num_tasks"],
            "num_edges": workflow["num_edges"],
            "root_count": len(workflow["roots"]),
            "topology_class": workflow["topology_class"],
        },
        "factors": dict(template),
        "rsu_ids": rsu_ids,
        "rsu_sequence": rsu_sequence,
        "link_profile": {
            "actual_mbps": actual_mbps,
            "estimated_mbps": estimated_mbps,
            "error_class": prediction_quality,
        },
        "prediction_profile": {
            "confidence": prediction_confidence,
            "uncertainty": prediction_uncertainty,
            "quality": prediction_quality,
        },
        "cache_capacity_bytes": capacity,
        "initial_residents": initial,
        "nodes": nodes,
        "edges": edges,
        "execution_order": execution_order,
        "deadline_seconds": round(deadline, 6),
        "max_steps": len(nodes) * 2,
        "source_classes": {
            "measured": ["base/adapter bytes", "local load time", "state/input bytes", "state restore overhead"],
            "trace_derived": ["NGSIM window handoff pressure", "Alibaba DAG topology/duration/memory"],
            "literature": [],
            "artificial": ["adapter assignment", "state scale", "link", "decision time scale", "trajectory-workflow pairing", "deadline"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/calibrated_continuous_workflow_pilot_v1.json")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", default="configs/experiment/calibrated_continuous_workflow_pilot_v1_manifest.json")
    args = parser.parse_args()
    config_path = ROOT_DIR / args.config
    output_path = ROOT_DIR / args.output
    if output_path.exists():
        raise FileExistsError(f"create-only manifest already exists: {output_path}")
    config = _load_json(config_path)
    selection = config["selection_protocol"]
    train_plan_path = ROOT_DIR / selection["train_window_plan"]
    eval_plan_path = ROOT_DIR / selection["evaluation_window_plan"]
    train_plan = _load_json(train_plan_path)["selected_window_plan"]
    eval_plan = _load_json(eval_plan_path)["selected_window_plan"]
    train_windows = _select_windows(train_plan, selection["train_windows"])
    dev_windows = _select_windows(train_plan, selection["dev_windows"])
    evaluation_windows = [eval_plan[index] for index in selection["evaluation_indices"]]
    workflow_path = Path(args.data_root).resolve() / Path(selection["workflow_source"]).relative_to("data")
    workflows = _load_workflows(
        workflow_path,
        int(selection["workflow_node_range"][0]),
        int(selection["workflow_node_range"][1]),
    )[: int(selection["workflow_count"])]
    if len(workflows) != int(selection["workflow_count"]):
        raise RuntimeError("not enough eligible Alibaba workflows")
    instances = []
    cursor = 0
    for split, windows in (("train", train_windows), ("dev", dev_windows), ("evaluation", evaluation_windows)):
        for ordinal, window in enumerate(windows):
            template = config["design_templates"][ordinal % len(config["design_templates"])]
            instances.append(
                _build_instance(
                    split=split,
                    ordinal=ordinal,
                    window=window,
                    workflow=workflows[cursor],
                    template=template,
                    config=config,
                )
            )
            cursor += 1
    manifest = {
        "schema_version": config["schema_version"],
        "config_path": args.config,
        "config_sha256": _sha256(config_path),
        "source_files": {
            "train_window_plan": {"path": selection["train_window_plan"], "sha256": _sha256(train_plan_path)},
            "evaluation_window_plan": {"path": selection["evaluation_window_plan"], "sha256": _sha256(eval_plan_path)},
            "workflow_source": {"path": selection["workflow_source"], "sha256": _sha256(workflow_path)},
        },
        "selection_outcome_blind": True,
        "split_counts": {split: sum(item["split"] == split for item in instances) for split in ("train", "dev", "evaluation")},
        "instances": instances,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "split_counts": manifest["split_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
