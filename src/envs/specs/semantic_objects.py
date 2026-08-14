"""核心语义对象定义。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar


CACHE_EVENT_SCHEMA_VERSION = "1.0.0"
CACHE_EVENT_TYPES = frozenset({"request", "not_applicable"})
CACHE_OBJECT_TYPES = frozenset({"adapter", "not_applicable"})
CACHE_HIT_SOURCES = frozenset(
    {
        "vehicle_local",
        "current_rsu",
        "target_rsu",
        "neighbor_rsu",
        "cloud",
        "unserved",
        "not_applicable",
    }
)


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


@dataclass(frozen=True)
class CacheEvent:
    """Request-level cache lifecycle event schema v1."""

    REQUIRED_FIELDS: ClassVar[tuple[str, ...]]

    event_id: str
    event_schema_version: str
    event_type: str
    time_index: int
    episode_step_index: int
    vehicle_id: str | None
    workflow_id: str | None
    node_id: str | None
    object_id: str | None
    adapter_id: str | None
    object_type: str
    size_mb: float | None
    request_rsu_id: str | None
    selected_target_rsu_id: str | None
    served_rsu_id: str | None
    predicted_next_rsu_id: str | None
    predicted_handoff_target_rsu_id: str | None
    hit_source: str
    cache_lookup_performed: bool
    cache_hit: bool
    was_cached_before: bool
    admission_requested: bool
    admission_added: bool
    admission_reason: str
    cache_target_rsu_id: str | None
    eviction_occurred: bool
    eviction_policy: str
    evicted_object_id: str | None
    evicted_adapter_id: str | None
    eviction_reason: str
    adapter_transfer_size_mb: float
    state_migration_size_mb: float
    transfer_source: str
    migration_requested: bool
    migration_realized: bool
    cache_capacity_enabled: bool
    cache_capacity_unit: str
    cache_capacity_before: float | None
    cache_used_before: float | None
    cache_remaining_before: float | None
    cache_capacity_after: float | None
    cache_used_after: float | None
    cache_remaining_after: float | None
    action_id: int | None
    action_name: str | None
    cache_strategy: str
    offload_mode: str
    service_success: bool
    stall_occurred: bool
    handoff_event_count: int

    def __post_init__(self) -> None:
        if self.event_schema_version != CACHE_EVENT_SCHEMA_VERSION:
            raise ValueError("unsupported cache event schema version")
        if self.event_type not in CACHE_EVENT_TYPES:
            raise ValueError(f"invalid cache event type: {self.event_type}")
        if self.object_type not in CACHE_OBJECT_TYPES:
            raise ValueError(f"invalid cache object type: {self.object_type}")
        if self.hit_source not in CACHE_HIT_SOURCES:
            raise ValueError(f"invalid cache hit source: {self.hit_source}")
        if self.cache_hit and self.hit_source not in {
            "vehicle_local", "current_rsu", "target_rsu", "neighbor_rsu", "cloud"
        }:
            raise ValueError("cache_hit requires a concrete hit_source")
        if not self.cache_hit and self.hit_source in {"current_rsu", "target_rsu", "neighbor_rsu"}:
            raise ValueError("RSU hit_source requires cache_hit=true")
        if self.admission_added and (not self.cache_target_rsu_id or not self.object_id):
            raise ValueError("admission_added requires cache target and object")
        if self.eviction_occurred and not self.evicted_object_id:
            raise ValueError("eviction requires a victim object")
        if not self.cache_capacity_enabled and any(
            value is not None
            for value in (
                self.cache_capacity_before,
                self.cache_used_before,
                self.cache_remaining_before,
                self.cache_capacity_after,
                self.cache_used_after,
                self.cache_remaining_after,
            )
        ):
            raise ValueError("capacity-disabled snapshots must be null")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CacheEvent":
        missing = [field_name for field_name in cls.REQUIRED_FIELDS if field_name not in payload]
        if missing:
            raise ValueError(f"missing cache event fields: {', '.join(missing)}")
        return cls(**{field_name: payload[field_name] for field_name in cls.REQUIRED_FIELDS})


CacheEvent.REQUIRED_FIELDS = tuple(item.name for item in fields(CacheEvent))


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
    positive_offset: float
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
