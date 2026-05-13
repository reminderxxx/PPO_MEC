"""核心语义对象定义。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class VehicleState:
    """车辆状态。"""

    vehicle_id: str
    position_x: float
    position_y: float
    speed: float
    base_model_id: str
    associated_rsu_id: str | None = None
    active_workflow_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转成普通字典。"""
        return asdict(self)


@dataclass
class RSUState:
    """RSU 状态。"""

    rsu_id: str
    position_x: float
    position_y: float
    coverage_radius: float
    cached_adapter_ids: list[str] = field(default_factory=list)
    active_vehicle_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转成普通字典。"""
        return asdict(self)


@dataclass
class WorkflowNode:
    """工作流节点。"""

    node_id: str
    node_name: str
    required_base_model: str
    required_adapter: str
    input_size: int
    output_size: int
    predecessors: list[str] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转成普通字典。"""
        return asdict(self)


@dataclass
class WorkflowGraphState:
    """工作流图状态。"""

    workflow_id: str
    nodes: list[WorkflowNode]
    edges: list[tuple[str, str]]
    execution_order: list[str]
    completed_node_ids: list[str] = field(default_factory=list)
    current_node_id: str | None = None
    is_completed: bool = False

    def node_map(self) -> dict[str, WorkflowNode]:
        """返回节点索引。"""
        return {node.node_id: node for node in self.nodes}

    def current_node(self) -> WorkflowNode | None:
        """返回当前节点。"""
        if self.current_node_id is None:
            return None
        return self.node_map().get(self.current_node_id)

    def mark_current_completed(self) -> None:
        """推进到下一个节点。"""
        if self.current_node_id is None:
            self.is_completed = True
            return

        if self.current_node_id not in self.completed_node_ids:
            self.completed_node_ids.append(self.current_node_id)

        remaining = [
            node_id
            for node_id in self.execution_order
            if node_id not in self.completed_node_ids
        ]
        self.current_node_id = remaining[0] if remaining else None
        self.is_completed = self.current_node_id is None

    def to_dict(self) -> dict[str, Any]:
        """转成普通字典。"""
        return {
            "workflow_id": self.workflow_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": list(self.edges),
            "execution_order": list(self.execution_order),
            "completed_node_ids": list(self.completed_node_ids),
            "current_node_id": self.current_node_id,
            "is_completed": self.is_completed,
        }


@dataclass
class CacheObject:
    """缓存对象。"""

    object_id: str
    adapter_id: str
    size_mb: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        """转成普通字典。"""
        return asdict(self)


@dataclass
class AdapterStateBundle:
    """适配器状态迁移包。"""

    bundle_id: str
    adapter_id: str
    state_version: str
    continuity_token: str
    serialized_state_ref: str

    def to_dict(self) -> dict[str, Any]:
        """转成普通字典。"""
        return asdict(self)


@dataclass
class ControlAction:
    """控制动作。"""

    cache_action: dict[str, Any] = field(default_factory=dict)
    offload_action: dict[str, Any] = field(default_factory=dict)
    migration_action: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转成普通字典。"""
        return asdict(self)


@dataclass
class RewardBreakdown:
    """奖励拆解。"""

    total: float
    service_reward: float
    delay_penalty: float
    cache_miss_penalty: float
    migration_cost: float
    continuity_bonus: float
    mechanism_exploration_bonus: float
    constraint_penalty: float

    def to_dict(self) -> dict[str, Any]:
        """转成普通字典。"""
        return asdict(self)


@dataclass
class PredictionSnapshot:
    """数字孪生 / surrogate 预测快照。"""

    snapshot_time: int
    predicted_next_rsu_by_vehicle: dict[str, str | None]
    predicted_handoff_vehicle_ids: list[str]
    surrogate_delay_by_vehicle: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """转成普通字典。"""
        return asdict(self)


@dataclass
class HandoffEvent:
    """切换事件。"""

    vehicle_id: str
    time_index: int
    previous_rsu_id: str | None
    current_rsu_id: str | None
    event_type: str

    def to_dict(self) -> dict[str, Any]:
        """转成普通字典。"""
        return asdict(self)
