"""Opt-in NGSIM event-time contact gate for the calibrated workflow pilot.

Raw positions drive only physical replay. Public estimates use observed prefixes.
The native step is admitted atomically only while its full cost fits contact.
"""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from src.envs.core.calibrated_continuous_workflow_env import (
    ACTION_NAMES,
    ORIGINAL_REWARD_PROFILE,
    PREPARED_STATE_PREFIX_PROFILE,
    SERVICE_ALIGNED_REWARD_PROFILE,
    CalibratedContinuousWorkflowEnv,
)


PROFILE = "raw_ngsim_event_time_v1"


@dataclass(frozen=True)
class RawVehicleTrace:
    times_seconds: tuple[float, ...]
    xy_metres: tuple[tuple[float, float], ...]
    vehicle_id: str

    def __post_init__(self) -> None:
        if len(self.times_seconds) < 3 or len(self.times_seconds) != len(self.xy_metres):
            raise ValueError("trace requires at least three paired samples")
        if abs(self.times_seconds[0]) > 1e-9:
            raise ValueError("trace must start at zero")
        if any(abs(b - a - 0.1) > 1e-6 for a, b in zip(self.times_seconds, self.times_seconds[1:])):
            raise ValueError("trace requires contiguous 100 ms samples")
        if any(not math.isfinite(value) for xy in self.xy_metres for value in xy):
            raise ValueError("nonfinite trace coordinate")


class RawNGSIMEventTimeEnv(CalibratedContinuousWorkflowEnv):
    """Conservative contact admission; no cross-boundary forwarding engine."""

    diagnostic_time_profile = PROFILE

    def __init__(self, config: dict[str, Any], instance: dict[str, Any], trace: RawVehicleTrace) -> None:
        self.trace = trace
        x0, y0 = trace.xy_metres[0]
        self.geometry = tuple((str(rsu), x0, y0 + 12.0 * index, 10.0)
                              for index, rsu in enumerate(instance["rsu_ids"]))
        if len(self.geometry) != 3:
            raise ValueError("raw NGSIM profile requires three RSUs")
        public_config = deepcopy(config)
        public_config["interface_profile"] = PREPARED_STATE_PREFIX_PROFILE
        super().__init__(public_config, instance)

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset()
        self.clock_seconds = self.trace.times_seconds[1]
        self.metrics["modeled_completion_seconds"] = self.clock_seconds
        return self._observation(), self._info()

    def _causal_prediction_enabled(self) -> bool:
        # The parent predictor consumes step-indexed synthetic RSU sequences.
        return False

    def _observed_index(self) -> int:
        return max(0, min(bisect_right(self.trace.times_seconds, self.clock_seconds + 1e-9) - 1,
                          len(self.trace.times_seconds) - 1))

    def _observed_xy(self) -> tuple[float, float]:
        return self.trace.xy_metres[self._observed_index()]

    def _actual_xy(self, seconds: float) -> tuple[float, float]:
        times = self.trace.times_seconds
        if seconds >= times[-1]:
            return self.trace.xy_metres[-1]
        index = max(0, bisect_right(times, seconds) - 1)
        alpha = (seconds - times[index]) / (times[index + 1] - times[index])
        a, b = self.trace.xy_metres[index:index + 2]
        return (a[0] + alpha * (b[0] - a[0]), a[1] + alpha * (b[1] - a[1]))

    def _association(self, xy: tuple[float, float]) -> str | None:
        covered = [(math.hypot(xy[0] - x, xy[1] - y), rsu)
                   for rsu, x, y, radius in self.geometry
                   if math.hypot(xy[0] - x, xy[1] - y) <= radius + 1e-9]
        return min(covered)[1] if covered else None

    def _current_rsu_id(self) -> str:
        # A structural fallback is needed by the parent cache representation.
        # Public association remains None when no circle covers the vehicle.
        return self._association(self._observed_xy()) or str(self.geometry[0][0])

    def _public_velocity(self) -> tuple[float, float]:
        index = self._observed_index()
        if index == 0:
            return (0.0, 0.0)
        a, b = self.trace.xy_metres[index - 1:index + 1]
        dt = self.trace.times_seconds[index] - self.trace.times_seconds[index - 1]
        return ((b[0] - a[0]) / dt, (b[1] - a[1]) / dt)

    def _predicted_handoff_target(self) -> str | None:
        xy = self._observed_xy()
        vx, vy = self._public_velocity()
        current = self._association(xy)
        for tick in range(1, 101):
            candidate = self._association((xy[0] + vx * tick * 0.1,
                                           xy[1] + vy * tick * 0.1))
            if candidate is not None and candidate != current:
                return candidate
        return None

    def _predicted_sequence(self) -> list[str]:
        target = self._predicted_handoff_target()
        return [target or self._current_rsu_id()] * int(self.config["prediction_horizon"])

    def _contact_budget_seconds(self) -> float:
        xy = self._observed_xy()
        vx, vy = self._public_velocity()
        current = self._association(xy)
        if current is None:
            return 0.0
        _, cx, cy, radius = next(row for row in self.geometry if row[0] == current)
        dx, dy = xy[0] - cx, xy[1] - cy
        speed2 = vx * vx + vy * vy
        if speed2 <= 1e-12:
            return 10.0  # Fixed forecast horizon, not hidden trace length.
        b = dx * vx + dy * vy
        root = (-b + math.sqrt(max(0.0, b * b + speed2 * (radius * radius - dx * dx - dy * dy)))) / speed2
        return max(0.0, min(10.0, root))

    def _physical_contact_budget_seconds(self) -> float:
        current = self._association(self._observed_xy())
        if current is None or self.clock_seconds >= self.trace.times_seconds[-1]:
            return 0.0
        _, cx, cy, radius = next(row for row in self.geometry if row[0] == current)
        start = self.clock_seconds
        p = self._actual_xy(start)
        if math.hypot(p[0] - cx, p[1] - cy) > radius + 1e-9:
            return 0.0
        index = max(0, bisect_right(self.trace.times_seconds, start) - 1)
        for next_index in range(index + 1, len(self.trace.times_seconds)):
            end = self.trace.times_seconds[next_index]
            q = self.trace.xy_metres[next_index]
            if math.hypot(q[0] - cx, q[1] - cy) > radius + 1e-9:
                dx, dy = q[0] - p[0], q[1] - p[1]
                a = dx * dx + dy * dy
                b = 2.0 * ((p[0] - cx) * dx + (p[1] - cy) * dy)
                c = (p[0] - cx) ** 2 + (p[1] - cy) ** 2 - radius ** 2
                fraction = (-b + math.sqrt(max(0.0, b * b - 4.0 * a * c))) / (2.0 * a)
                return max(0.0, start + fraction * (end - start) - self.clock_seconds)
            p, start = q, end
        return max(0.0, self.trace.times_seconds[-1] - self.clock_seconds)

    def _semantic_state(self) -> dict[str, Any]:
        state = super()._semantic_state()
        observed = self._observed_xy()
        vx, vy = self._public_velocity()
        association = self._association(observed)
        vehicle = state["vehicles"][0]
        vehicle.update(position_x=observed[0], position_y=observed[1],
                       speed=math.hypot(vx, vy), associated_rsu_id=association)
        for rsu, (_, x, y, radius) in zip(state["rsus"], self.geometry):
            rsu.update(position_x=x, position_y=y, coverage_radius=radius,
                       active_vehicle_ids=["veh_pilot"] if rsu["rsu_id"] == association else [])
        state["predictions"]["prediction_confidence_by_vehicle"] = {"veh_pilot": 0.5}
        state["predictions"]["prediction_uncertainty_by_vehicle"] = {"veh_pilot": 0.5}
        state["predictions"]["causal_provenance"] = {
            "source": "observed_position_prefix_constant_velocity",
            "observed_sample_count": self._observed_index() + 1,
            "actual_future_used": False,
            "confidence_calibrated": False,
        }
        context = state["calibrated_context"]
        context.update(time_contract={"schema_version": PROFILE,
                                      "clock_seconds": self.clock_seconds,
                                      "deadline_seconds": float(self.instance["deadline_seconds"]),
                                      "remaining_deadline_seconds": float(self.instance["deadline_seconds"]) - self.clock_seconds,
                                      "contact_scope": "current_rsu"},
                       vehicle_fallback_seconds=float(self.config["vehicle"]["fallback_seconds"]),
                       failed_service_seconds=float(self.config["objective"]["failed_service_seconds"]))
        state["time_profile"] = PROFILE
        return state

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        if self._association(self._observed_xy()) is None:
            info["action_mask"] = [False, False, True, False, False]
        return info

    def _reject(self, action: int, reason: str, planned_cost: float) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        remaining = max(0.0, self.trace.times_seconds[-1] - self.clock_seconds)
        elapsed = min(remaining, float(self.config["objective"]["failed_service_seconds"]))
        self.clock_seconds += elapsed
        self.step_index += 1
        self.metrics["service_failures"] += 1
        self.metrics["failed_service_attempt_seconds_proxy"] += elapsed
        self.metrics["modeled_completion_seconds"] = self.clock_seconds
        deadline_miss = not self._deadline_miss_recorded and self.clock_seconds > float(self.instance["deadline_seconds"])
        if deadline_miss:
            self._deadline_miss_recorded = True
            self.metrics["deadline_missed"] = self.metrics["deadline_violations"] = 1
            self.metrics["unfinished_after_deadline"] = 1
        components_by_profile = {
            profile: self._reward_components(profile=profile, completed=False, step_cost=elapsed,
                                             transfer_bytes=0, failure=True, terminal=False,
                                             deadline_miss_event=deadline_miss,
                                             service_operation_seconds=0.0, recompute_seconds=0.0)
            for profile in (ORIGINAL_REWARD_PROFILE, SERVICE_ALIGNED_REWARD_PROFILE)
            if self._reward_profile_available(profile)
        }
        for profile, components in components_by_profile.items():
            self._reward_totals_by_profile[profile] += sum(components.values())
        active = str(self.config.get("reward_profile", ORIGINAL_REWARD_PROFILE))
        components = components_by_profile[active]
        reward = float(sum(components.values()))
        self.metrics["reward"] += reward
        for key, value in components.items():
            if key in self.metrics:
                self.metrics[key] += value
        info = self._info()
        info["transition"] = {"step_index": self.step_index - 1,
                              "node_id": self._current_node()["node_id"],
                              "action": action, "action_name": ACTION_NAMES[action],
                              "service_completed": False, "admission_rejection_reason": reason,
                              "planned_step_cost_seconds": planned_cost,
                              "step_cost_seconds": elapsed, "clock_seconds_after": self.clock_seconds,
                              "model_transfer_bytes": 0, "state_transfer_bytes": 0,
                              "input_transfer_bytes": 0, "transfer_seconds": 0.0,
                              "reward": reward, "reward_components": components}
        info["episode_metrics"] = deepcopy(self.metrics)
        truncated = self.clock_seconds >= self.trace.times_seconds[-1] - 1e-9 or self.step_index >= int(self.instance["max_steps"])
        return self._observation(), reward, False, truncated, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.terminated:
            return self._observation(), 0.0, True, False, self._info()
        if self.clock_seconds >= self.trace.times_seconds[-1] - 1e-9:
            return self._observation(), 0.0, False, True, self._info()
        action = int(action)
        mask = self._info()["action_mask"]
        if action < 0 or action >= len(mask) or not mask[action]:
            self.metrics["invalid_action_count"] += 1
            action = 2 if mask[2] else (3 if mask[3] else next(i for i, valid in enumerate(mask) if valid))
        preview = deepcopy(self)
        _, _, _, _, preview_info = CalibratedContinuousWorkflowEnv.step(preview, action)
        planned = float(preview_info["transition"]["step_cost_seconds"])
        trace_budget = self.trace.times_seconds[-1] - self.clock_seconds
        contact_budget = self._physical_contact_budget_seconds() if action != 2 else math.inf
        if action != 2 and self._association(self._observed_xy()) is None:
            return self._reject(action, "no_current_rsu_coverage", planned)
        if planned > trace_budget + 1e-9:
            return self._reject(action, "trace_end_before_service_commit", planned)
        if planned > contact_budget + 1e-9:
            return self._reject(action, "current_rsu_contact_expires_before_commit", planned)
        obs, reward, terminated, truncated, info = CalibratedContinuousWorkflowEnv.step(self, action)
        info["transition"]["admission_rejection_reason"] = None
        info["episode_metrics"] = deepcopy(self.metrics)
        return obs, reward, terminated, bool(truncated or (not terminated and self.clock_seconds >= self.trace.times_seconds[-1] - 1e-9)), info

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        result.update(time_profile=PROFILE, trace_end_seconds=self.trace.times_seconds[-1],
                      trace_exhausted=bool(self.clock_seconds >= self.trace.times_seconds[-1] - 1e-9))
        return result
