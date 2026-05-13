"""运行 Alibaba sample workflow 检查。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.workflow.alibaba_dag_parser import AlibabaDAGParser


SCAN_MULTIPLIER = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="解析 Alibaba batch task sample")
    parser.add_argument(
        "--csv_path",
        type=str,
        default=str(ROOT_DIR / "data" / "raw" / "workflow" / "alibaba2018" / "batch_task.csv"),
        help="Alibaba batch_task.csv 路径",
    )
    parser.add_argument(
        "--jsonl_path",
        type=str,
        default=str(ROOT_DIR / "data" / "processed" / "sampled_vec_dags" / "sampled_vec_dags.jsonl"),
        help="可选的 sampled_vec_dags.jsonl 路径",
    )
    parser.add_argument("--prefer_jsonl", action="store_true", help="优先从 sampled_vec_dags.jsonl 读取")
    parser.add_argument("--limit_jobs", type=int, default=3, help="最终展示多少个 workflow")
    parser.add_argument("--min_tasks", type=int, default=5, help="最小任务数")
    parser.add_argument("--max_tasks", type=int, default=20, help="最大任务数")
    parser.add_argument("--preview_nodes", type=int, default=5, help="打印前几个节点摘要")
    parser.add_argument("--preview_edges", type=int, default=8, help="打印前几条边摘要")
    return parser.parse_args()


def load_samples_from_jsonl(jsonl_path: Path, limit_jobs: int) -> list[dict[str, Any]]:
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"指定的 sampled_vec_dags.jsonl 不存在: {jsonl_path}。请检查处理产物路径。"
        )
    samples: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            samples.append(json.loads(stripped))
            if len(samples) >= limit_jobs:
                break
    if not samples:
        raise RuntimeError(f"sampled_vec_dags.jsonl 已找到，但没有读到有效 workflow: {jsonl_path}")
    return samples


def summarize_node(node: dict[str, Any]) -> dict[str, Any]:
    node_id = node.get("node_id")
    if node_id is None and node.get("task_id") is not None:
        node_id = f"task_{node['task_id']}"
    return {
        "node_id": node_id,
        "node_name": node.get("node_name") or node.get("task_name"),
        "required_adapter": node.get("required_adapter"),
        "predecessor_count": len(node.get("predecessors", node.get("parents", []))),
        "successor_count": len(node.get("successors", [])),
    }


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    jsonl_path = Path(args.jsonl_path)

    if args.prefer_jsonl:
        samples = load_samples_from_jsonl(jsonl_path=jsonl_path, limit_jobs=args.limit_jobs)
        source_path = jsonl_path
        source_type = "jsonl"
    else:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Alibaba batch_task.csv 不存在: {csv_path}。请把原始文件放到 data/raw/workflow/alibaba2018/ 下。"
            )
        parser = AlibabaDAGParser(csv_path=csv_path)
        scan_limit = max(args.limit_jobs * SCAN_MULTIPLIER, args.limit_jobs)
        samples = parser.parse_jobs(
            limit_jobs=scan_limit,
            min_tasks=args.min_tasks,
            max_tasks=args.max_tasks,
        )[: args.limit_jobs]
        source_path = csv_path
        source_type = "csv"

    print("Alibaba sample 解析完成")
    print(f"source_type: {source_type}")
    print(f"source_path: {source_path}")
    print(f"workflow_count: {len(samples)}")

    for workflow in samples:
        roots = workflow.get("roots", [])
        print(
            f"workflow_id={workflow.get('workflow_id')} "
            f"tasks={workflow.get('num_tasks', len(workflow.get('nodes', [])))} "
            f"edges={workflow.get('num_edges', len(workflow.get('edges', [])))} "
            f"roots={roots[:5]}"
        )

    first_sample = samples[0]
    print("sample_node_summary:")
    for node in first_sample.get("nodes", [])[: args.preview_nodes]:
        print(f"  {summarize_node(node)}")

    print("sample_edge_summary:")
    for edge in first_sample.get("edges", [])[: args.preview_edges]:
        print(f"  {edge}")


if __name__ == "__main__":
    main()
