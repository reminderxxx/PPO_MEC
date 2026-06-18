"""Alibaba batch task trace 的 DAG 解析器。"""

from __future__ import annotations

import csv
import builtins as _builtins
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


ADAPTER_ASSIGNMENT_LEGACY_BATCH_TYPE = "legacy_batch_type"
ADAPTER_ASSIGNMENT_SEMANTIC_AI_SERVICE = "semantic_ai_service"
SUPPORTED_ADAPTER_ASSIGNMENT_PROFILES = {
    ADAPTER_ASSIGNMENT_LEGACY_BATCH_TYPE,
    ADAPTER_ASSIGNMENT_SEMANTIC_AI_SERVICE,
}


class AlibabaDAGParser:
    """将 Alibaba batch_task.csv 解析为统一 workflow sample。

    当前实现面向“真实数据接入前夜”的科研原型，已经支持：
    - 原始 CSV 存在性检查
    - 按 job 聚合行记录
    - 从 `task_name` 解析依赖关系
    - 构造成统一 workflow sample 字典

    统一 sample 中已经补上最小 VEC 语义字段：
    - `required_base_model`
    - `required_adapter`
    - `input_size`
    - `output_size`
    """

    RAW_COLUMNS = [
        "task_name",
        "instance_num",
        "job_name",
        "task_type",
        "status",
        "start_time",
        "end_time",
        "plan_cpu",
        "plan_mem",
    ]

    def __init__(
        self,
        csv_path: str | Path,
        adapter_assignment_profile: str = ADAPTER_ASSIGNMENT_LEGACY_BATCH_TYPE,
    ) -> None:
        self._csv_path = Path(csv_path)
        self._adapter_assignment_profile = self._normalize_adapter_assignment_profile(adapter_assignment_profile)
        self._validate_source()

    def read_raw_rows(self, limit_rows: int | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self._csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            for index, row in enumerate(reader, start=1):
                if len(row) < len(self.RAW_COLUMNS):
                    raise ValueError(
                        f"Alibaba batch_task.csv 第 {index} 行字段数不足，期望 {len(self.RAW_COLUMNS)} 列。"
                    )
                rows.append(self._normalize_row(row))
                if limit_rows is not None and len(rows) >= limit_rows:
                    break
        return rows

    def collect_jobs(
        self,
        limit_jobs: int = 32,
        min_tasks: int = 5,
        max_tasks: int | None = 64,
    ) -> dict[str, list[dict[str, Any]]]:
        if limit_jobs <= 0:
            raise ValueError("limit_jobs 必须大于 0。")
        jobs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen_job_names: list[str] = []
        with self._csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            for index, row in enumerate(reader, start=1):
                if len(row) < len(self.RAW_COLUMNS):
                    raise ValueError(
                        f"Alibaba batch_task.csv 第 {index} 行字段数不足，期望 {len(self.RAW_COLUMNS)} 列。"
                    )
                raw_job_name = _builtins.str(row[2]).strip()
                if raw_job_name not in jobs and len(seen_job_names) >= limit_jobs:
                    # Alibaba batch_task rows are job-grouped in the consumed trace.
                    # Once the requested job prefix is complete, scanning the rest of
                    # the full CSV only adds latency and can surface unrelated dirty rows.
                    break
                normalized_row = self._normalize_row(row)
                job_name = normalized_row["job_name"]
                if job_name not in jobs:
                    seen_job_names.append(job_name)
                jobs[job_name].append(normalized_row)

        filtered_jobs: dict[str, list[dict[str, Any]]] = {}
        for job_name, rows in jobs.items():
            task_count = len(rows)
            if task_count < min_tasks:
                continue
            if max_tasks is not None and task_count > max_tasks:
                continue
            filtered_jobs[job_name] = rows
        if not filtered_jobs:
            raise RuntimeError(
                "Alibaba batch_task.csv 已找到，但在当前筛选条件下没有收集到可解析 job。"
            )
        return filtered_jobs

    def build_workflow_sample(self, job_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            raise ValueError(f"job={job_name} 没有可解析行记录。")

        node_rows: dict[int, dict[str, Any]] = {}
        edge_pairs: set[tuple[int, int]] = set()
        for row in rows:
            task_id, parent_ids = self.parse_task_name(row["task_name"])
            if task_id is None:
                raise ValueError(f"job={job_name} 出现无法解析的 task_name: {row['task_name']}")
            if task_id not in node_rows:
                node_rows[task_id] = {
                    "task_id": task_id,
                    "task_name": row["task_name"],
                    "parents": list(parent_ids),
                    "task_type": int(row["task_type"]),
                    "instance_num": int(row["instance_num"]),
                    "start_time": int(row["start_time"]),
                    "end_time": int(row["end_time"]),
                    "plan_cpu": float(row["plan_cpu"]),
                    "plan_mem": float(row["plan_mem"]),
                }
            for parent_id in parent_ids:
                edge_pairs.add((parent_id, task_id))

        task_ids = set(node_rows.keys())
        missing_parents = [
            parent_id
            for parent_id, child_id in edge_pairs
            if parent_id not in task_ids or child_id not in task_ids
        ]
        if missing_parents:
            raise ValueError(
                f"job={job_name} 依赖引用了缺失父节点，示例父节点: {missing_parents[:5]}"
            )

        edge_list = sorted(edge_pairs)
        execution_order = self._topological_sort(task_ids=task_ids, edge_list=edge_list)
        successor_map: dict[int, list[int]] = defaultdict(list)
        predecessor_map: dict[int, list[int]] = defaultdict(list)
        for source, target in edge_list:
            successor_map[source].append(target)
            predecessor_map[target].append(source)

        nodes: list[dict[str, Any]] = []
        for task_id in execution_order:
            row = node_rows[task_id]
            task_type = int(row["task_type"])
            plan_cpu = float(row["plan_cpu"])
            plan_mem = float(row["plan_mem"])
            duration = max(1, int(row["end_time"]) - int(row["start_time"]))
            required_adapter, assignment_basis = self._assign_required_adapter(
                task_id=task_id,
                task_type=task_type,
                execution_order=execution_order,
                predecessor_map=predecessor_map,
                successor_map=successor_map,
            )
            nodes.append(
                {
                    "node_id": f"task_{task_id}",
                    "node_name": row["task_name"],
                    "required_base_model": "veh_base_v1",
                    "required_adapter": required_adapter,
                    "input_size": max(8, int(plan_mem * 32.0)),
                    "output_size": max(4, int(plan_cpu * 16.0)),
                    "predecessors": [f"task_{item}" for item in predecessor_map.get(task_id, [])],
                    "successors": [f"task_{item}" for item in successor_map.get(task_id, [])],
                    "raw_profile": {
                        "task_type": task_type,
                        "instance_num": int(row["instance_num"]),
                        "plan_cpu": plan_cpu,
                        "plan_mem": plan_mem,
                        "duration": duration,
                        "adapter_assignment_profile": self._adapter_assignment_profile,
                        "adapter_assignment_basis": assignment_basis,
                    },
                }
            )

        return {
            "workflow_id": job_name,
            "source": "alibaba_batch_task_trace",
            "num_tasks": len(nodes),
            "num_edges": len(edge_list),
            "roots": [node["node_id"] for node in nodes if not node["predecessors"]],
            "execution_order": [node["node_id"] for node in nodes],
            "edges": [(f"task_{source}", f"task_{target}") for source, target in edge_list],
            "nodes": nodes,
            "adapter_assignment_profile": self._adapter_assignment_profile,
            "adapter_assignment_profile_note": self._adapter_assignment_profile_note(),
        }

    def parse_jobs(
        self,
        limit_jobs: int = 32,
        min_tasks: int = 5,
        max_tasks: int | None = 64,
    ) -> list[dict[str, Any]]:
        jobs = self.collect_jobs(limit_jobs=limit_jobs, min_tasks=min_tasks, max_tasks=max_tasks)
        samples: list[dict[str, Any]] = []
        for job_name, rows in jobs.items():
            samples.append(self.build_workflow_sample(job_name=job_name, rows=rows))
        return samples

    def parse_task_name(self, task_name: str) -> tuple[int | None, list[int]]:
        if not isinstance(task_name, str) or not task_name:
            return None, []
        head, *parent_parts = task_name.split("_")
        matched = re.search(r"(\d+)$", head)
        if matched is None:
            return None, []
        task_id = int(matched.group(1))
        parent_ids = [int(item) for item in parent_parts if item.isdigit()]
        return task_id, parent_ids

    def _normalize_row(self, raw_row: list[str]) -> dict[str, Any]:
        return {
            "task_name": str(raw_row[0]).strip(),
            "instance_num": self._safe_int(raw_row[1]),
            "job_name": str(raw_row[2]).strip(),
            "task_type": self._safe_int(raw_row[3]),
            "status": str(raw_row[4]).strip(),
            "start_time": self._safe_int(raw_row[5]),
            "end_time": self._safe_int(raw_row[6]),
            "plan_cpu": self._safe_float(raw_row[7]),
            "plan_mem": self._safe_float(raw_row[8]),
        }

    def _topological_sort(
        self,
        task_ids: set[int],
        edge_list: list[tuple[int, int]],
    ) -> list[int]:
        indegree = {task_id: 0 for task_id in task_ids}
        adjacency: dict[int, list[int]] = defaultdict(list)
        for source, target in edge_list:
            adjacency[source].append(target)
            indegree[target] += 1

        ready_queue = deque(sorted(task_id for task_id, degree in indegree.items() if degree == 0))
        ordered_task_ids: list[int] = []
        while ready_queue:
            current = ready_queue.popleft()
            ordered_task_ids.append(current)
            for target in sorted(adjacency[current]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready_queue.append(target)

        if len(ordered_task_ids) != len(task_ids):
            raise ValueError("Alibaba job 解析结果含环，无法构造成 DAG workflow sample。")
        return ordered_task_ids

    def _normalize_adapter_assignment_profile(self, profile: str) -> str:
        normalized = str(profile or ADAPTER_ASSIGNMENT_LEGACY_BATCH_TYPE).strip()
        if normalized not in SUPPORTED_ADAPTER_ASSIGNMENT_PROFILES:
            supported = ", ".join(sorted(SUPPORTED_ADAPTER_ASSIGNMENT_PROFILES))
            raise ValueError(f"未知 adapter_assignment_profile={profile!r}，支持: {supported}")
        return normalized

    def _adapter_assignment_profile_note(self) -> str:
        if self._adapter_assignment_profile == ADAPTER_ASSIGNMENT_LEGACY_BATCH_TYPE:
            return "Legacy mapping: Alibaba task_type is mapped to adapter_batch_type_<task_type> for backward compatibility."
        return (
            "Controlled AI-service adapter assignment profile based on real Alibaba DAG structure and task_type fields; "
            "the adapter labels are not claimed to be original Alibaba adapter IDs."
        )

    def _assign_required_adapter(
        self,
        *,
        task_id: int,
        task_type: int,
        execution_order: list[int],
        predecessor_map: dict[int, list[int]],
        successor_map: dict[int, list[int]],
    ) -> tuple[str, str]:
        if self._adapter_assignment_profile == ADAPTER_ASSIGNMENT_LEGACY_BATCH_TYPE:
            return f"adapter_batch_type_{task_type}", "legacy_task_type"

        predecessor_count = len(predecessor_map.get(task_id, []))
        successor_count = len(successor_map.get(task_id, []))
        topo_index = execution_order.index(task_id)
        topo_ratio = topo_index / max(len(execution_order) - 1, 1)
        if predecessor_count == 0:
            return "adapter_perception", "source_or_input_node"
        if successor_count == 0:
            return "adapter_control", "sink_or_output_node"
        if predecessor_count >= 2:
            return "adapter_fusion", "multi_parent_fusion_node"
        if topo_ratio >= 0.66:
            return "adapter_intent", "late_topological_position"
        if topo_ratio >= 0.33:
            return "adapter_tracking", "middle_topological_position"
        return "adapter_tracking", "single_parent_early_middle_node"

    def _safe_int(self, raw_value: Any) -> int:
        normalized = _builtins.str(raw_value).strip()
        if normalized == "":
            return 0
        return _builtins.int(_builtins.float(normalized.replace(",", "")))

    def _safe_float(self, raw_value: Any) -> float:
        normalized = _builtins.str(raw_value).strip()
        if normalized == "":
            return 0.0
        return _builtins.float(normalized.replace(",", ""))

    def _validate_source(self) -> None:
        if not self._csv_path.exists():
            raise FileNotFoundError(
                f"Alibaba batch task 原始文件不存在: {self._csv_path}。请先将 batch_task.csv 放到 data/raw/workflow/alibaba2018/ 下。"
            )
        if self._csv_path.suffix.lower() != ".csv":
            raise ValueError(
                f"Alibaba batch task 当前只支持 CSV 输入，收到路径: {self._csv_path}。"
            )
