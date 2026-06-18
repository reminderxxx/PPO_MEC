"""toy DAG 工作流生成器。"""

from __future__ import annotations

from src.envs.specs import WorkflowGraphState, WorkflowNode


class ToyWorkflowGenerator:
    """生成一个可配置节点数的连续 DAG。"""

    def generate(
        self,
        workflow_id: str = "wf_toy_1",
        node_count: int = 6,
    ) -> WorkflowGraphState:
        if node_count < 5 or node_count > 8:
            raise ValueError("toy workflow 节点数当前仅支持 5 到 8。")

        node_templates = self._build_node_templates()
        edge_templates = [
            ("n1", "n2"),
            ("n1", "n3"),
            ("n2", "n4"),
            ("n3", "n4"),
            ("n4", "n5"),
            ("n5", "n6"),
            ("n6", "n7"),
            ("n7", "n8"),
        ]
        execution_order = [f"n{index}" for index in range(1, node_count + 1)]
        selected_node_ids = set(execution_order)
        nodes = [node_templates[node_id] for node_id in execution_order]
        edges = [
            (source, target)
            for source, target in edge_templates
            if source in selected_node_ids and target in selected_node_ids
        ]
        self._refresh_dependencies(nodes=nodes, edges=edges)
        return WorkflowGraphState(
            workflow_id=workflow_id,
            nodes=nodes,
            edges=edges,
            execution_order=execution_order,
            current_node_id=execution_order[0],
        )

    def _build_node_templates(self) -> dict[str, WorkflowNode]:
        return {
            "n1": WorkflowNode(
                node_id="n1",
                node_name="感知预热",
                required_base_model="veh_base_v1",
                required_adapter="adapter_perception",
                input_size=16,
                output_size=32,
            ),
            "n2": WorkflowNode(
                node_id="n2",
                node_name="车道理解",
                required_base_model="veh_base_v1",
                required_adapter="adapter_lane",
                input_size=32,
                output_size=24,
            ),
            "n3": WorkflowNode(
                node_id="n3",
                node_name="目标跟踪",
                required_base_model="veh_base_v1",
                required_adapter="adapter_tracking",
                input_size=32,
                output_size=24,
            ),
            "n4": WorkflowNode(
                node_id="n4",
                node_name="场景融合",
                required_base_model="veh_base_v1",
                required_adapter="adapter_fusion",
                input_size=48,
                output_size=24,
            ),
            "n5": WorkflowNode(
                node_id="n5",
                node_name="意图推断",
                required_base_model="veh_base_v1",
                required_adapter="adapter_intent",
                input_size=24,
                output_size=16,
            ),
            "n6": WorkflowNode(
                node_id="n6",
                node_name="控制细化",
                required_base_model="veh_base_v1",
                required_adapter="adapter_control",
                input_size=16,
                output_size=8,
            ),
            "n7": WorkflowNode(
                node_id="n7",
                node_name="风险评估",
                required_base_model="veh_base_v1",
                required_adapter="adapter_risk",
                input_size=8,
                output_size=8,
            ),
            "n8": WorkflowNode(
                node_id="n8",
                node_name="协同广播",
                required_base_model="veh_base_v1",
                required_adapter="adapter_coop",
                input_size=8,
                output_size=4,
            ),
        }

    def _refresh_dependencies(
        self,
        nodes: list[WorkflowNode],
        edges: list[tuple[str, str]],
    ) -> None:
        node_map = {node.node_id: node for node in nodes}
        for node in nodes:
            node.predecessors = []
            node.successors = []
        for source, target in edges:
            node_map[source].successors.append(target)
            node_map[target].predecessors.append(source)
