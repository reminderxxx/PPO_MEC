"""workflow 数据集构造器。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from src.data.workflow.alibaba_dag_parser import AlibabaDAGParser
from src.data.workflow.toy_workflow_generator import ToyWorkflowGenerator
from src.envs.specs import WorkflowGraphState, WorkflowNode


class WorkflowDatasetBuilder:
    """统一构造 toy 与真实 ingress 前夜的 workflow 数据集。"""

    def __init__(self, generator: ToyWorkflowGenerator | None = None) -> None:
        self._generator = generator or ToyWorkflowGenerator()

    def build_toy_dataset(self, size: int = 1, node_count: int = 6) -> list[WorkflowGraphState]:
        return [
            self._generator.generate(workflow_id=f"wf_toy_{index + 1}", node_count=node_count)
            for index in range(size)
        ]

    def build_alibaba_samples(
        self,
        csv_path: str | Path,
        limit_jobs: int = 32,
        min_tasks: int = 5,
        max_tasks: int | None = 64,
        adapter_assignment_profile: str = "legacy_batch_type",
    ) -> list[dict[str, Any]]:
        parser = AlibabaDAGParser(
            csv_path=csv_path,
            adapter_assignment_profile=adapter_assignment_profile,
        )
        return parser.parse_jobs(
            limit_jobs=limit_jobs,
            min_tasks=min_tasks,
            max_tasks=max_tasks,
        )

    def select_alibaba_samples(
        self,
        samples: list[dict[str, Any]],
        max_workflows: int = 1,
        workflow_selector: str = "ordered",
        random_seed: int = 7,
    ) -> list[dict[str, Any]]:
        if not samples:
            raise RuntimeError("没有可供选择的 Alibaba workflow samples。")
        if max_workflows <= 0:
            raise ValueError("max_workflows 必须大于 0。")

        selector = (workflow_selector or "ordered").strip()
        selector_lower = selector.lower()

        if selector_lower in {"ordered", "first", "sequential"}:
            return samples[: min(max_workflows, len(samples))]

        if selector_lower == "random":
            rng = random.Random(random_seed)
            sample_count = min(max_workflows, len(samples))
            return rng.sample(samples, sample_count)

        if selector_lower.startswith("job_name:"):
            target_job_name = selector.split(":", 1)[1].strip()
            selected = [sample for sample in samples if sample.get("workflow_id") == target_job_name]
            if not selected:
                raise ValueError(f"未找到指定 workflow job_name: {target_job_name}")
            return selected[:1]

        if selector_lower.startswith("index:"):
            raw_index = selector.split(":", 1)[1].strip()
            target_index = int(raw_index)
            if target_index < 0 or target_index >= len(samples):
                raise IndexError(
                    f"workflow_selector=index:{target_index} 超出范围，当前样本数为 {len(samples)}。"
                )
            return [samples[target_index]]

        raise ValueError(
            "未知 workflow_selector。支持: ordered, random, job_name:<job>, index:<zero_based_index>"
        )

    def build_selected_alibaba_workflow_states(
        self,
        csv_path: str | Path,
        max_workflows: int = 1,
        workflow_selector: str = "ordered",
        min_tasks: int = 5,
        max_tasks: int | None = 64,
        random_seed: int = 7,
        adapter_assignment_profile: str = "legacy_batch_type",
    ) -> list[WorkflowGraphState]:
        scan_limit = max(max_workflows * 20, max_workflows)
        if workflow_selector.lower().startswith("job_name:"):
            scan_limit = 512
        samples = self.build_alibaba_samples(
            csv_path=csv_path,
            limit_jobs=scan_limit,
            min_tasks=min_tasks,
            max_tasks=max_tasks,
            adapter_assignment_profile=adapter_assignment_profile,
        )
        selected_samples = self.select_alibaba_samples(
            samples=samples,
            max_workflows=max_workflows,
            workflow_selector=workflow_selector,
            random_seed=random_seed,
        )
        return [self.sample_to_workflow_state(sample) for sample in selected_samples]

    def build_alibaba_workflow_states(
        self,
        csv_path: str | Path,
        limit_jobs: int = 16,
        min_tasks: int = 5,
        max_tasks: int | None = 32,
        adapter_assignment_profile: str = "legacy_batch_type",
    ) -> list[WorkflowGraphState]:
        samples = self.build_alibaba_samples(
            csv_path=csv_path,
            limit_jobs=limit_jobs,
            min_tasks=min_tasks,
            max_tasks=max_tasks,
            adapter_assignment_profile=adapter_assignment_profile,
        )
        return [self.sample_to_workflow_state(sample) for sample in samples]

    def sample_to_workflow_state(self, sample: dict[str, Any]) -> WorkflowGraphState:
        nodes = [
            WorkflowNode(
                node_id=node["node_id"],
                node_name=node["node_name"],
                required_base_model=node["required_base_model"],
                required_adapter=node["required_adapter"],
                input_size=int(node["input_size"]),
                output_size=int(node["output_size"]),
                predecessors=list(node.get("predecessors", [])),
                successors=list(node.get("successors", [])),
            )
            for node in sample["nodes"]
        ]
        execution_order = list(sample.get("execution_order", []))
        if not execution_order:
            raise ValueError("workflow sample 缺少 execution_order，无法转成 WorkflowGraphState。")
        return WorkflowGraphState(
            workflow_id=sample["workflow_id"],
            nodes=nodes,
            edges=[tuple(edge) for edge in sample.get("edges", [])],
            execution_order=execution_order,
            current_node_id=execution_order[0],
        )
