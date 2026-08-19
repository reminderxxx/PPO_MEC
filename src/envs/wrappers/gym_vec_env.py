"""核心环境的最小 Gym 包装层。"""

from __future__ import annotations

from copy import deepcopy
import numpy as np
from typing import Any

from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs import ActionAdapter, ActionMaskBuilder, ActionSchema, ControlAction
from src.envs.wrappers.observation_normalizer import ObservationNormalizer

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        class _离散空间:
            def __init__(self, n: int) -> None:
                self.n = n

        class _盒空间:
            def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: type) -> None:
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

        class _空间集合:
            Discrete = _离散空间
            Box = _盒空间

        class _基础环境:
            metadata: dict[str, Any] = {}

        class _Gym替身:
            Env = _基础环境

        gym = _Gym替身()
        spaces = _空间集合()


默认动作规范 = ActionSchema.default_vec_workflow_schema()
DECISION_OBSERVATION_TRACE_VERSION = "1.0.0"
FLAT_OBSERVATION_FEATURE_NAMES = (
    "relative_time",
    "vehicle_count",
    "rsu_count",
    "workflow_progress",
    "handoff_event_count",
    "accepted_predicted_handoff_count",
    "current_node_available",
    "current_rsu_cache_size",
    "mean_future_load",
)
动作语义表 = {
    action_id: 默认动作规范.action_name(action_id)
    for action_id in range(默认动作规范.discrete_action_count)
}


class GymVecEnv(gym.Env):
    """单智能体最小包装层。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        core_env: VecWorkflowCoreEnv | None = None,
        recorder: Any | None = None,
    ) -> None:
        super().__init__()
        self._core_env = core_env or VecWorkflowCoreEnv()
        self._recorder = recorder
        self._normalizer = ObservationNormalizer()
        self._action_schema = 默认动作规范
        self._action_adapter = ActionAdapter(self._action_schema)
        self._action_mask_builder = ActionMaskBuilder(self._action_schema)
        self._last_state: dict[str, Any] | None = None
        self._episode_step_index = 0
        # 初始化随机状态生成器，用于可复现性
        self._np_random: np.random.RandomState | None = None
        self._current_seed: int | None = None
        self.action_space = spaces.Discrete(self._action_schema.discrete_action_count)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=2.0,
            shape=(9,),
            dtype=float,
        )

    @property
    def core_env(self) -> VecWorkflowCoreEnv:
        """暴露底层语义环境。"""
        return self._core_env

    @property
    def last_semantic_state(self) -> dict[str, Any] | None:
        """返回最近一次语义状态。"""
        return self._last_state

    @property
    def action_schema(self) -> ActionSchema:
        """Return the semantic action contract exposed by this wrapper."""
        return self._action_schema

    @property
    def current_seed(self) -> int | None:
        """返回当前使用的随机种子，用于可复现性追踪。"""
        return self._current_seed

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[list[float], dict[str, Any]]:
        # 使用 seed 参数初始化随机状态生成器，确保可复现性
        if seed is not None:
            self._np_random = np.random.RandomState(seed)
            self._current_seed = seed
        else:
            # 如果没有提供 seed，创建一个独立的随机状态
            self._np_random = np.random.RandomState()
            self._current_seed = None

        # 将随机状态传递给核心环境（如果支持）
        import inspect
        core_reset_sig = inspect.signature(self._core_env.reset)
        if 'seed' in core_reset_sig.parameters:
            state, info = self._core_env.reset(seed=seed, options=options)
        else:
            state, info = self._core_env.reset()
        self._last_state = state
        self._episode_step_index = 0
        self._normalizer.reset(state)
        semantic_state = self._build_compatible_semantic_state(state)
        action_mask_info = self._action_mask_builder.build_mask_info(semantic_state)
        info = {
            **info,
            "cache_trace_snapshot": self._core_env.export_cache_trace_snapshot(),
            "semantic_state": semantic_state,
            "action_schema": self._action_schema.to_dict(),
            "action_mask": action_mask_info["mask"],
            "action_mask_info": action_mask_info,
            "observation_normalized": True,
            "observation_encoder": "deterministic_scale_v1",
        }
        if self._recorder is not None:
            self._recorder.record_reset(state, info)
        return self._encode_observation(state), info

    def step(self, action: int) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        pre_action_trace_record = self._build_decision_observation_trace_record(action)
        control = self._decode_action(action)
        action_name = 动作语义表.get(action, "unknown_action")
        state, reward, terminated, truncated, info = self._core_env.step(control)
        self._last_state = state
        self._episode_step_index += 1
        semantic_state = self._build_compatible_semantic_state(state)
        action_mask_info = self._action_mask_builder.build_mask_info(semantic_state)
        action_metadata = dict(control.metadata)
        info = {
            **info,
            "cache_trace_snapshot": self._core_env.export_cache_trace_snapshot(),
            "semantic_state": semantic_state,
            "action_schema": self._action_schema.to_dict(),
            "action_mask": action_mask_info["mask"],
            "action_mask_info": action_mask_info,
            "control_action": control.to_dict(),
            "action_id": action,
            "action_name": action_name,
            "action_metadata": action_metadata,
            "action_invalid": bool(action_metadata.get("invalid_action", False)),
            "action_invalid_reason": str(action_metadata.get("invalid_reason", "none")),
            "observation_normalized": True,
            "observation_encoder": "deterministic_scale_v1",
            "decision_observation_trace_record": pre_action_trace_record,
        }
        if self._recorder is not None:
            self._recorder.record_step(
                state=state,
                info=info,
                reward_dict=reward.to_dict(),
                terminated=terminated,
                truncated=truncated,
            )
        return self._encode_observation(state), reward.total, terminated, truncated, info

    def _decode_action(self, action: int) -> ControlAction:
        semantic_state = self._build_compatible_semantic_state(self._last_state or {})
        return self._action_adapter.decode(action, semantic_state)

    def _encode_observation(self, state: dict[str, Any]) -> list[float]:
        raw_observation = self._build_raw_observation(state)
        return self._normalizer.normalize(
            raw_observation=raw_observation,
            state=state,
            episode_step_index=self._episode_step_index,
            max_steps=getattr(self._core_env, "_max_steps", 16),
        )

    def _build_raw_observation(self, state: dict[str, Any]) -> list[float]:
        workflow = state.get("workflow", {})
        completed_node_ids = workflow.get("completed_node_ids", [])
        execution_order = workflow.get("execution_order", [])
        vehicles = state.get("vehicles", [])
        rsus = state.get("rsus", [])
        predictions = state.get("predictions", {})
        current_node = state.get("current_workflow_node")
        primary_vehicle = self._resolve_primary_vehicle_for_observation(state)
        current_rsu_cache_size = 0.0
        if primary_vehicle.get("associated_rsu_id"):
            associated_rsu_id = primary_vehicle["associated_rsu_id"]
            for rsu in rsus:
                if rsu.get("rsu_id") == associated_rsu_id:
                    current_rsu_cache_size = float(len(rsu.get("cached_adapter_ids", [])))
                    break
        progress = 0.0
        if execution_order:
            progress = float(len(completed_node_ids)) / float(len(execution_order))
        predicted_handoffs = predictions.get("predicted_handoff_vehicle_ids", [])
        future_load = predictions.get("future_load", {})
        mean_future_load = 0.0
        if future_load:
            mean_future_load = sum(float(value) for value in future_load.values()) / float(len(future_load))
        raw_observation = [
            float(state.get("time_index", 0)),
            float(len(vehicles)),
            float(len(rsus)),
            progress,
            float(len(state.get("handoff_events", []))),
            float(len(predicted_handoffs)),
            0.0 if current_node is None else 1.0,
            current_rsu_cache_size,
            float(mean_future_load),
        ]
        return raw_observation

    def _build_decision_observation_trace_record(self, action: int) -> dict[str, Any]:
        """Capture the wrapper observation before applying the selected action."""
        state = deepcopy(self._last_state or {})
        semantic_state = self._build_compatible_semantic_state(state)
        raw_observation = self._build_raw_observation(state)
        flattened = self._encode_observation(state)
        feature_name_to_index = {
            name: index for index, name in enumerate(FLAT_OBSERVATION_FEATURE_NAMES)
        }
        feature_values = {
            name: float(flattened[index]) for name, index in feature_name_to_index.items()
        }
        raw_values = {
            name: float(raw_observation[index]) for name, index in feature_name_to_index.items()
        }
        action_mask_info = self._action_mask_builder.build_mask_info(semantic_state)
        action_mask = [bool(value) for value in action_mask_info["mask"]]
        primary_vehicle = self._resolve_primary_vehicle_for_observation(state)
        vehicle_id = str(primary_vehicle.get("vehicle_id")) if primary_vehicle.get("vehicle_id") is not None else None
        predictions = dict(state.get("predictions", {}))
        snapshot = dict(predictions.get("causal_predictor_snapshots_by_vehicle", {}).get(vehicle_id, {})) if vehicle_id else {}
        snapshot_identity = dict(snapshot.get("identity", {}))
        snapshot_time = dict(snapshot.get("causal_time", {}))
        next_step = self._episode_step_index + 1
        request_id = (
            self._recorder.build_decision_request_id(next_step)
            if self._recorder is not None and hasattr(self._recorder, "build_decision_request_id")
            else f"local_episode/request_{next_step:06d}"
        )
        current_rsu_id = primary_vehicle.get("associated_rsu_id")
        predictor_projection = {
            "snapshot_id": snapshot_identity.get("snapshot_id"),
            "availability_mask": predictions.get("causal_snapshot_availability_by_vehicle", {}).get(vehicle_id, 0),
            "predicted_next_rsu_id": predictions.get("predicted_next_rsu_by_vehicle", {}).get(vehicle_id),
            "predicted_handoff_target_rsu_id": predictions.get("predicted_handoff_target_rsu_id_by_vehicle", {}).get(vehicle_id),
        }
        return {
            "decision_observation_trace_version": DECISION_OBSERVATION_TRACE_VERSION,
            "request_id": request_id,
            "step_index": next_step,
            "time_index": int(state.get("time_index", 0)),
            "captured_phase": "pre_action",
            "controller_identity": "single_environment_action_controller",
            "observation_contract_version": "gym_vec_flat_semantic_v1",
            "raw_semantic_fields": {
                "primary_vehicle_id": vehicle_id,
                "current_rsu_id": current_rsu_id,
                "flat_raw_feature_values": raw_values,
                "predictor": predictor_projection,
            },
            "semantic_feature_map": raw_values,
            "flattened_observation": [float(value) for value in flattened],
            "flattened_dimension": len(flattened),
            "feature_name_to_index": feature_name_to_index,
            "flattened_feature_values": feature_values,
            "feature_availability": {
                name: True for name in FLAT_OBSERVATION_FEATURE_NAMES
            },
            "normalization": {
                "kind": "deterministic_scale_and_tanh",
                "version": "deterministic_scale_v1",
                "fitted_on_data": False,
            },
            "information_scopes": {
                "current_local_information": ["current_rsu_id", "current_rsu_cache_size"],
                "cross_rsu_global_information": ["rsu_count", "mean_future_load"],
                "predictor_outputs": ["accepted_predicted_handoff_count", "predictor"],
                "history_derived_information": ["relative_time"],
                "actor_visibility": "flat_observation_plus_agent_semantic_encoder_if_configured",
                "controller_visibility": "semantic_state_and_flat_observation",
                "critic_visibility": "agent_dependent; wrapper_adds_no_future_or_label_fields",
            },
            "predictor_snapshot_provenance": {
                "snapshot_id": snapshot_identity.get("snapshot_id"),
                "contract_version": snapshot_identity.get("contract_version"),
                "generated_at_step": snapshot_time.get("generated_at_step"),
                "observation_as_of_step": snapshot_time.get("observation_as_of_step"),
                "consumed_at_step": snapshot_time.get("consumed_at_step"),
                "age_steps": snapshot_time.get("age_steps"),
                "availability_mask": predictor_projection["availability_mask"],
            },
            "action_mask": action_mask,
            "eligible_actions": [index for index, value in enumerate(action_mask) if value],
            "post_decision_outcome": {"selected_action": int(action)},
            "observation_projections": {
                "actor_local": {
                    "current_rsu_id": current_rsu_id,
                    "current_rsu_cache_size": raw_values["current_rsu_cache_size"],
                },
                "controller_global": {
                    "current_rsu_id": current_rsu_id,
                    "rsu_count": raw_values["rsu_count"],
                    "mean_future_load": raw_values["mean_future_load"],
                },
                "critic_only": {},
                "predictor_augmented": predictor_projection,
            },
            "future_or_outcome_fields_present": False,
        }

    def _resolve_primary_vehicle_for_observation(self, state: dict[str, Any]) -> dict[str, Any]:
        vehicles = list(state.get("vehicles", []))
        primary_vehicle_id = state.get("primary_vehicle_id")
        if primary_vehicle_id is not None:
            for vehicle in vehicles:
                if str(vehicle.get("vehicle_id")) == str(primary_vehicle_id):
                    return dict(vehicle)
        return dict(vehicles[0]) if vehicles else {}

    def _build_compatible_semantic_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """仅在包装层提供旧字段兼容，不回写到底层环境状态。"""
        semantic_state = deepcopy(state)
        predictions = semantic_state.get("predictions", {})
        if "prediction" not in semantic_state:
            semantic_state["prediction"] = {
                "deprecated": True,
                "message": "请改用正式字段 predictions。",
                "predicted_next_rsu_by_vehicle": predictions.get("predicted_next_rsu_by_vehicle", {}),
                "predicted_handoff_vehicle_ids": predictions.get("predicted_handoff_vehicle_ids", []),
                "surrogate_delay_by_vehicle": predictions.get("surrogate_delay_by_vehicle", {}),
            }
        return semantic_state
