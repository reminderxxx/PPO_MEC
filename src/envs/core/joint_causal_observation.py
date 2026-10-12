"""Causal prediction and delayed shared observations for joint simulation v2.

Frame-based motion extrapolation and JSON wire sizes are controlled proxies,
not calibrated vehicle prediction or measured network costs.
"""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
import json
from typing import Any


class CausalMobilityPredictor:
    """Score only forecasts whose target frame has already been observed."""

    def __init__(self, horizon: int = 3, history_size: int = 64) -> None:
        if horizon < 1 or history_size < 1:
            raise ValueError("horizon and history_size must be positive")
        self.horizon = int(horizon)
        self.history_size = int(history_size)
        self.previous: dict[str, tuple[int, float, float]] = {}
        self.pending: dict[int, dict[str, str | None]] = {}
        self.scores: dict[str, deque] = {}
        self.last_step = -1
        self.last_result: dict = {}

    def predict(self, vehicles: list, associations: dict, rsus: list,
                mapper: Any, step_index: int) -> dict:
        if step_index == self.last_step:
            return deepcopy(self.last_result)
        if step_index < self.last_step:
            raise ValueError("prediction frames must increase; reset predictor between episodes")
        due = self.pending.pop(step_index, {})
        for vehicle_id, target in due.items():
            if vehicle_id in associations:
                self.scores.setdefault(vehicle_id, deque(maxlen=self.history_size)).append(
                    float(target == associations[vehicle_id]))
        # Missing target frames are not scored against a different time.
        self.pending = {step: value for step, value in self.pending.items() if step > step_index}
        projected = []
        next_rsu, sequences, confidence, dwell, sample_counts = {}, {}, {}, {}, {}
        for vehicle in vehicles:
            vid = vehicle.vehicle_id
            old = self.previous.get(vid)
            dx = dy = 0.0
            if old is not None:
                elapsed = step_index - old[0]
                dx = (vehicle.position_x - old[1]) / elapsed
                dy = (vehicle.position_y - old[2]) / elapsed
            hops = []
            for offset in range(1, self.horizon + 1):
                future = deepcopy(vehicle)
                future.position_x += offset * dx
                future.position_y += offset * dy
                target = mapper.associate([future])[vid]
                hops.append(target)
                if offset == self.horizon:
                    projected.append(future)
            changed = [(index + 1, target) for index, target in enumerate(hops)
                       if target is not None and target != associations[vid]]
            next_rsu[vid] = changed[0][1] if changed and old is not None else None
            # Keep one entry per horizon frame: consumers use the index as ETA.
            sequences[vid] = hops
            dwell[vid] = float(changed[0][0]) if changed and old is not None else 0.0
            history = self.scores.get(vid, ())
            sample_counts[vid] = len(history)
            confidence[vid] = (sum(history) / len(history)) if history else 0.0
            self.previous[vid] = (step_index, vehicle.position_x, vehicle.position_y)
        final_associations = mapper.associate(projected)
        self.pending[step_index + self.horizon] = final_associations
        counts = Counter(value for value in final_associations.values() if value is not None)
        result = {
            "predicted_next_rsu_by_vehicle": next_rsu,
            "predicted_first_handoff_rsu_by_vehicle": next_rsu.copy(),
            "predicted_handoff_target_rsu_id_by_vehicle": next_rsu.copy(),
            "next_rsu_sequence": sequences,
            "predicted_handoff_vehicle_ids": [vid for vid, target in next_rsu.items() if target],
            "prediction_confidence_by_vehicle": confidence,
            "prediction_uncertainty_by_vehicle": {vid: 1.0 - value for vid, value in confidence.items()},
            "prediction_observed_sample_count_by_vehicle": sample_counts,
            "dwell_time": dwell,
            "future_load": {rsu.rsu_id: float(counts[rsu.rsu_id]) for rsu in rsus},
            "prediction_source": "causal_constant_velocity_v1",
            "confidence_source": "rolling_observed_horizon_accuracy_not_calibrated_probability",
            "prediction_horizon_frames": self.horizon,
            "prediction_as_of_frame": step_index,
        }
        self.last_step, self.last_result = step_index, deepcopy(result)
        return result


class SharedStateChannel:
    """Fixed-cadence broadcast with delay, age, and a serialized byte ledger."""

    def __init__(self, interval_frames: int = 1, delay_frames: int = 1,
                 enabled: bool = True) -> None:
        if interval_frames < 1 or delay_frames < 0:
            raise ValueError("invalid sharing interval or delay")
        self.interval = int(interval_frames)
        self.delay = int(delay_frames)
        self.enabled = bool(enabled)
        self.pending: deque = deque()
        self.delivered: tuple[int, list[dict]] | None = None
        self.last_step = -1
        self.total_bytes = 0
        self.last_result: dict = {}

    def snapshot(self, rsus: list[dict], step_index: int) -> dict:
        if step_index == self.last_step:
            return deepcopy(self.last_result)
        if step_index < self.last_step:
            raise ValueError("sharing frames must increase; reset channel between episodes")
        transmitted = 0
        if self.enabled and step_index % self.interval == 0:
            payload = deepcopy(rsus)
            transmitted = len(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False, allow_nan=False).encode("utf-8"))
            self.pending.append((step_index + self.delay, step_index, payload))
            self.total_bytes += transmitted
        while self.pending and self.pending[0][0] <= step_index:
            _, sent_at, payload = self.pending.popleft()
            self.delivered = (sent_at, payload)
        observed_at = self.delivered[0] if self.delivered else None
        result = {
            "rsus": deepcopy(self.delivered[1]) if self.delivered else [],
            "metadata": {
                "enabled": self.enabled,
                "available": self.delivered is not None,
                "observed_at_frame": observed_at,
                "age_frames": step_index - observed_at if observed_at is not None else None,
                "delay_frames": self.delay,
                "interval_frames": self.interval,
                "transmitted_bytes": transmitted,
                "total_transmitted_bytes": self.total_bytes,
                "cost_source": "serialized_json_proxy_not_measured_radio",
            },
        }
        self.last_step, self.last_result = step_index, deepcopy(result)
        return result

    @staticmethod
    def actor_rsus(current_rsus: list[dict], local_rsu_id: str | None, snapshot: dict) -> list[dict]:
        remote = {row["rsu_id"]: row for row in snapshot["rsus"]}
        rows = []
        for current in current_rsus:
            rid = current["rsu_id"]
            if rid == local_rsu_id:
                rows.append({**deepcopy(current), "shared_state_available": True, "shared_state_age_frames": 0})
            elif rid in remote:
                rows.append({**deepcopy(remote[rid]), "shared_state_available": True,
                             "shared_state_age_frames": snapshot["metadata"]["age_frames"]})
            else:
                # Only static geometry is public before the first broadcast arrives.
                row = {key: deepcopy(value) for key, value in current.items()
                       if key in {"rsu_id", "position_x", "position_y", "coverage_radius"}}
                rows.append({**row, "shared_state_available": False, "shared_state_age_frames": None})
        return rows
