"""核心环境的最小 Gym 包装层。"""

from __future__ import annotations

from copy import deepcopy
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

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[list[float], dict[str, Any]]:
        del seed
        del options
        state, info = self._core_env.reset()
        self._last_state = state
        self._episode_step_index = 0
        self._normalizer.reset(state)
        semantic_state = self._build_compatible_semantic_state(state)
        action_mask_info = self._action_mask_builder.build_mask_info(semantic_state)
        info = {
            **info,
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
        workflow = state.get("workflow", {})
        completed_node_ids = workflow.get("completed_node_ids", [])
        execution_order = workflow.get("execution_order", [])
        vehicles = state.get("vehicles", [])
        rsus = state.get("rsus", [])
        predictions = state.get("predictions", {})
        current_node = state.get("current_workflow_node")
        current_rsu_cache_size = 0.0
        if vehicles and vehicles[0].get("associated_rsu_id"):
            associated_rsu_id = vehicles[0]["associated_rsu_id"]
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
        return self._normalizer.normalize(
            raw_observation=raw_observation,
            state=state,
            episode_step_index=self._episode_step_index,
            max_steps=getattr(self._core_env, "_max_steps", 16),
        )

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
