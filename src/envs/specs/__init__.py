"""环境语义对象与类型定义。"""

from .semantic_objects import (
    AdapterStateBundle,
    CACHE_EVENT_SCHEMA_VERSION,
    CACHE_EVENT_TYPES,
    CACHE_HIT_SOURCES,
    CACHE_OBJECT_TYPES,
    CacheEvent,
    CacheObject,
    ControlAction,
    HandoffEvent,
    PredictionSnapshot,
    RSUState,
    RewardBreakdown,
    VehicleState,
    WorkflowGraphState,
    WorkflowNode,
)
from .action_schema import ActionAdapter, ActionMaskBuilder, ActionSchema, DiscreteActionSpec

__all__ = [
    "ActionAdapter",
    "ActionMaskBuilder",
    "ActionSchema",
    "AdapterStateBundle",
    "CACHE_EVENT_SCHEMA_VERSION",
    "CACHE_EVENT_TYPES",
    "CACHE_HIT_SOURCES",
    "CACHE_OBJECT_TYPES",
    "CacheEvent",
    "CacheObject",
    "ControlAction",
    "DiscreteActionSpec",
    "HandoffEvent",
    "PredictionSnapshot",
    "RSUState",
    "RewardBreakdown",
    "VehicleState",
    "WorkflowGraphState",
    "WorkflowNode",
]
