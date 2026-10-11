"""Opt-in serial raw-time phases with remote source-RSU computation.

The three-RSU geometry and positions remain the v1 diagnostic assumptions.
Radio and backhaul rates are explicit simulation parameters, not measurements.
No decision path may inspect a trace suffix; suffix access is confined to the
real transition's contact and target-delivery checks.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import numpy as np

from src.envs.core.calibrated_continuous_workflow_env import (
    ACTION_NAMES,
    ORIGINAL_REWARD_PROFILE,
    SERVICE_ALIGNED_REWARD_PROFILE,
    CalibratedContinuousWorkflowEnv,
    _network_seconds,
)
from src.envs.core.raw_ngsim_event_time_env import RawNGSIMEventTimeEnv, RawVehicleTrace


PROFILE = "raw_ngsim_event_time_v2"
EPS = 1e-9
SERIAL_PHASE_ORDER = (
    "input_upload", "model_prepare", "current_node_compute", "state_transfer",
    "target_restore", "target_delivery_wait",
)


class RawNGSIMPhaseV2Env(RawNGSIMEventTimeEnv):
    """Five-action workflow with an explicit input, compute and delivery ledger."""

    diagnostic_time_profile = PROFILE

    def __init__(
        self,
        config: dict[str, Any],
        instance: dict[str, Any],
        trace: RawVehicleTrace,
        *,
        phase_contract: dict[str, Any],
        condition: dict[str, Any],
    ) -> None:
        if tuple(phase_contract.get("phase_order", ())) != SERIAL_PHASE_ORDER:
            raise ValueError("raw phase v2 phase order differs from frozen contract")
        if phase_contract.get("service_execution") != "source_rsu_continues_after_input_upload":
            raise ValueError("raw phase v2 requires source-RSU remote execution")
        if phase_contract.get("overlap_enabled") is not False:
            raise ValueError("raw phase v2 requires serial phases")
        if float(phase_contract["queue_seconds"]) != 0.0:
            raise ValueError("raw phase v2 has no calibrated shared queue")
        for key in ("backhaul_fixed_seconds", "backhaul_phase_timeout_seconds", "target_delivery_timeout_seconds"):
            if float(phase_contract[key]) < 0.0:
                raise ValueError(f"negative phase contract parameter: {key}")
        for key in ("backhaul_actual_mbps", "backhaul_estimated_mbps"):
            if float(condition[key]) <= 0.0:
                raise ValueError(f"invalid backhaul rate: {key}")
        self.phase_contract = deepcopy(phase_contract)
        self.phase_condition = deepcopy(condition)
        super().__init__(config, instance, trace)
        if condition.get("cache_profile") == "all_required_bundles_at_all_rsus":
            self._install_ample_cache()
        elif condition.get("cache_profile") != "workload_native":
            raise ValueError("unknown phase cache profile")

    def clone_for_decision_model(self) -> "RawNGSIMPhaseV2Env":
        raise RuntimeError("raw phase v2 forbids exact hidden-future decision preview")

    def _install_ample_cache(self) -> None:
        capacity = int(self.phase_condition["cache_capacity_bytes"])
        required = sorted({
            object_id
            for node in self.instance["nodes"]
            for object_id in self._bundle_ids(str(node["required_adapter"]))
        })
        total = sum(int(self._object_catalog[item]["resident_bytes"]) for item in required)
        if total > capacity:
            raise ValueError("ample cache cannot hold all required bundles")
        for cache in self.caches.values():
            cache.capacity_bytes = capacity
            cache.residents = list(required)
            cache.last_used = {item: 0 for item in required}
        self._validate_all_caches()

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset()
        self.execution_owner_rsu_id: str | None = None
        self.pending_delivery: dict[str, Any] | None = None
        self.metrics.update({
            "backhaul_transfer_bytes": 0,
            "backhaul_transfer_seconds": 0.0,
            "partial_transfer_bytes": 0,
            "delivery_wait_seconds": 0.0,
            "migration_failures": 0,
            "pending_state_count": 0,
            "duplicate_execution_count": 0,
        })
        if self.phase_condition.get("cache_profile") == "all_required_bundles_at_all_rsus":
            self._install_ample_cache()
        return self._observation(), self._info()

    def _semantic_state(self) -> dict[str, Any]:
        state = super()._semantic_state()
        context = state["calibrated_context"]
        context["time_contract"].update(
            schema_version=PROFILE,
            contact_scope="input_upload_only",
            remote_compute_after_upload=True,
        )
        link = dict(self.instance.get("link_profile", {}) or {})
        context["phase_contract"] = {
            "schema_version": PROFILE,
            "phase_order": list(self.phase_contract["phase_order"]),
            "overlap_enabled": False,
            "service_execution": "source_rsu_continues_after_input_upload",
            "input_upload": {
                "estimated_mbps": float(link.get("estimated_mbps", self.config["link"]["mbps"])),
                "fixed_seconds": float(self.config["link"]["fixed_seconds"]),
                "requires_current_contact": True,
            },
            "backhaul": {
                "estimated_mbps": float(self.phase_condition["backhaul_estimated_mbps"]),
                "fixed_seconds": float(self.phase_contract["backhaul_fixed_seconds"]),
                "phase_timeout_seconds": float(self.phase_contract["backhaul_phase_timeout_seconds"]),
            },
            "queue": {"seconds": 0.0, "source_type": self.phase_contract["queue_source_type"]},
            "target_delivery": {
                "timeout_seconds": float(self.phase_contract["target_delivery_timeout_seconds"]),
                "prediction_conditioned": True,
            },
            "source_types": {
                "radio": self.phase_contract["radio_source_type"],
                "backhaul": self.phase_contract["backhaul_source_type"],
            },
        }
        state["time_profile"] = PROFILE
        return state

    def _actual_radio_mbps(self) -> float:
        profile = dict(self.instance.get("link_profile", {}) or {})
        result = float(profile.get("actual_mbps", self.config["link"]["mbps"]))
        if result <= 0.0:
            raise ValueError("raw phase v2 radio rate must be positive")
        return result

    def _network_phase(
        self,
        ledger: list[dict[str, Any]],
        name: str,
        *,
        byte_count: int,
        mbps: float,
        fixed_seconds: float,
        location: str,
        contact_end: float | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[bool, int, str]:
        start = self.clock_seconds
        planned = _network_seconds(byte_count, mbps, fixed_seconds)
        trace_end = self.trace.times_seconds[-1]
        limits: list[tuple[float, str]] = [(trace_end, "trace_end")]
        if contact_end is not None:
            limits.append((contact_end, "current_contact_expired"))
        if timeout_seconds is not None:
            limits.append((start + timeout_seconds, "phase_timeout"))
        limit, reason = min(limits, key=lambda item: item[0])
        elapsed = min(planned, max(0.0, limit - start))
        transferred = (
            byte_count if planned <= elapsed + EPS else
            min(byte_count, max(0, int(max(0.0, elapsed - fixed_seconds) * mbps * 1_000_000.0 / 8.0)))
        )
        complete = planned <= elapsed + EPS
        self.clock_seconds = start + elapsed
        status = "completed" if complete else reason
        ledger.append({
            "phase": name, "location": location, "start_seconds": start,
            "end_seconds": self.clock_seconds, "planned_seconds": planned,
            "elapsed_seconds": elapsed, "planned_bytes": int(byte_count),
            "transferred_bytes": int(transferred), "status": status,
        })
        return complete, int(transferred), status

    def _duration_phase(
        self, ledger: list[dict[str, Any]], name: str, *, seconds: float,
        location: str,
    ) -> tuple[bool, str]:
        start = self.clock_seconds
        elapsed = min(float(seconds), max(0.0, self.trace.times_seconds[-1] - start))
        self.clock_seconds = start + elapsed
        complete = seconds <= elapsed + EPS
        status = "completed" if complete else "trace_end"
        ledger.append({
            "phase": name, "location": location, "start_seconds": start,
            "end_seconds": self.clock_seconds, "planned_seconds": float(seconds),
            "elapsed_seconds": elapsed, "planned_bytes": 0,
            "transferred_bytes": 0, "status": status,
        })
        return complete, status

    def _target_seen_between(self, target: str, start: float, end: float) -> bool:
        if self._association(self._actual_xy(start)) == target:
            return True
        return any(
            start < time <= end and self._association(xy) == target
            for time, xy in zip(self.trace.times_seconds, self.trace.xy_metres)
        )

    def _first_target_entry(self, target: str, start: float, end: float) -> float | None:
        if self._association(self._actual_xy(start)) == target:
            return start
        previous = start
        for sample in self.trace.times_seconds:
            if sample <= start:
                continue
            if sample > end + EPS:
                break
            if self._association(self._actual_xy(sample)) == target:
                lo, hi = previous, sample
                for _ in range(30):
                    midpoint = (lo + hi) / 2.0
                    if self._association(self._actual_xy(midpoint)) == target:
                        hi = midpoint
                    else:
                        lo = midpoint
                return hi
            previous = sample
        if self._association(self._actual_xy(end)) == target:
            return end
        return None

    def _delivery(
        self, ledger: list[dict[str, Any]], target: str,
        *, decision_start: float,
    ) -> tuple[bool, str, float]:
        start = self.clock_seconds
        if self._association(self._actual_xy(start)) == target:
            elapsed = 0.0
            status = "delivered"
        elif self._target_seen_between(target, decision_start, start):
            elapsed = 0.0
            status = "target_departed_before_delivery"
        else:
            end = min(
                start + float(self.phase_contract["target_delivery_timeout_seconds"]),
                self.trace.times_seconds[-1],
            )
            arrival = self._first_target_entry(target, start, end)
            final = arrival if arrival is not None else end
            elapsed = max(0.0, final - start)
            self.clock_seconds = final
            status = (
                "delivered" if arrival is not None else
                "trace_end_pending_delivery" if final >= self.trace.times_seconds[-1] - EPS else
                "target_delivery_timeout"
            )
        ledger.append({
            "phase": "target_delivery_wait", "location": target,
            "start_seconds": start, "end_seconds": self.clock_seconds,
            "planned_seconds": float(self.phase_contract["target_delivery_timeout_seconds"]),
            "elapsed_seconds": elapsed, "planned_bytes": 0,
            "transferred_bytes": 0, "status": status,
        })
        return status == "delivered", status, elapsed

    def _finalize(
        self, *, action: int, requested: int, node_id: str, current: str,
        target: str | None, start: float, ledger: list[dict[str, Any]],
        service_completed: bool, migration_success: bool,
        migration_status: str, failure_reason: str | None,
        input_bytes: int, model_bytes: int, state_bytes: int,
        backhaul_bytes: int, partial_bytes: int,
        transfer_seconds: float, backhaul_seconds: float,
        model_load_seconds: float, state_restore_seconds: float,
        recompute_seconds: float, compute_seconds: float,
        failure_wait_seconds: float, delivery_wait_seconds: float,
        cache_events: list[dict[str, Any]],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_index += 1
        cost = self.clock_seconds - start
        service_operation = compute_seconds + model_load_seconds + state_restore_seconds + failure_wait_seconds
        if not math.isclose(
            cost, transfer_seconds + service_operation + recompute_seconds + delivery_wait_seconds,
            abs_tol=1e-6, rel_tol=0.0,
        ):
            raise RuntimeError("phase seconds do not conserve step cost")
        self.metrics["model_transfer_bytes"] += model_bytes
        self.metrics["state_transfer_bytes"] += state_bytes
        self.metrics["input_transfer_bytes"] += input_bytes
        self.metrics["backhaul_transfer_bytes"] += backhaul_bytes
        self.metrics["partial_transfer_bytes"] += partial_bytes
        self.metrics["transfer_seconds"] += transfer_seconds
        self.metrics["backhaul_transfer_seconds"] += backhaul_seconds
        self.metrics["model_load_seconds"] += model_load_seconds
        self.metrics["state_restore_seconds"] += state_restore_seconds
        self.metrics["recompute_seconds"] += recompute_seconds
        self.metrics["delivery_wait_seconds"] += delivery_wait_seconds
        self.metrics["modeled_completion_seconds"] = self.clock_seconds
        if not service_completed:
            self.metrics["service_failures"] += 1
        if migration_status not in {"not_requested", "delivered"}:
            self.metrics["migration_failures"] += 1
        if migration_status == "trace_end_pending_delivery":
            self.metrics["pending_state_count"] += 1
        if migration_success:
            self.metrics["migration_successes"] += 1
        deadline_miss = not self._deadline_miss_recorded and self.clock_seconds > float(self.instance["deadline_seconds"])
        if deadline_miss:
            self._deadline_miss_recorded = True
            self.metrics["deadline_missed"] = self.metrics["deadline_violations"] = 1
        if self.terminated:
            self.metrics["workflow_completed"] = 1
            if self._deadline_miss_recorded:
                self.metrics["late_workflow_completed"] = 1
            else:
                self.metrics["on_time_workflow_completed"] = 1
        elif self._deadline_miss_recorded:
            self.metrics["unfinished_after_deadline"] = 1
        self.metrics["failed_service_attempt_seconds_proxy"] = (
            float(self.metrics["service_failures"])
            * float(self.config["objective"]["failed_service_seconds"])
        )
        components_by_profile = {
            profile: self._reward_components(
                profile=profile, completed=service_completed, step_cost=cost,
                transfer_bytes=model_bytes + state_bytes + input_bytes,
                failure=not service_completed, terminal=self.terminated,
                deadline_miss_event=deadline_miss,
                service_operation_seconds=service_operation,
                recompute_seconds=recompute_seconds,
            )
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
        if self.pending_delivery is not None and migration_status != "trace_end_pending_delivery":
            self.pending_delivery = None
        event = {
            "step_index": self.step_index - 1, "node_id": node_id,
            "requested_action": requested, "action": action,
            "action_name": ACTION_NAMES[action],
            "action_projection_applied": requested != action,
            "time_profile": PROFILE,
            "execution_owner_rsu_id": self.execution_owner_rsu_id,
            "predicted_target_rsu_id": target,
            "service_completed": service_completed,
            "migration_success": migration_success,
            "migration_status": migration_status,
            "failure_reason": failure_reason,
            "admission_rejection_reason": failure_reason,
            "phase_events": ledger,
            "cache_events": cache_events,
            "state_transfer": {"status": migration_status, "target_rsu_id": target} if action == 4 else None,
            "model_transfer_bytes": model_bytes,
            "state_transfer_bytes": state_bytes,
            "input_transfer_bytes": input_bytes,
            "backhaul_transfer_bytes": backhaul_bytes,
            "partial_transfer_bytes": partial_bytes,
            "transfer_seconds": transfer_seconds,
            "backhaul_transfer_seconds": backhaul_seconds,
            "model_load_seconds": model_load_seconds,
            "state_restore_seconds": state_restore_seconds,
            "recompute_seconds": recompute_seconds,
            "delivery_wait_seconds": delivery_wait_seconds,
            "service_operation_seconds": service_operation,
            "step_cost_seconds": cost,
            "clock_seconds_after": self.clock_seconds,
            "deadline_seconds": float(self.instance["deadline_seconds"]),
            "deadline_miss_event": deadline_miss,
            "reward": reward,
            "reward_components": components,
            "reward_components_by_profile": components_by_profile,
        }
        info = self._info()
        info["transition"] = event
        info["episode_metrics"] = deepcopy(self.metrics)
        truncated = (
            not self.terminated
            and (self.clock_seconds >= self.trace.times_seconds[-1] - EPS
                 or self.step_index >= int(self.instance["max_steps"]))
        )
        return self._observation(), reward, self.terminated, truncated, info

    def _fallback(self, action: int, requested: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        before = self.clock_seconds
        planned_input_bytes = int(self._current_node()["input_bytes"])
        preview = deepcopy(self)
        _, _, _, _, preview_info = CalibratedContinuousWorkflowEnv.step(preview, 2)
        planned = float(preview_info["transition"]["step_cost_seconds"])
        if planned > self.trace.times_seconds[-1] - before + EPS:
            obs, reward, terminated, truncated, info = self._reject(2, "trace_end_before_service_commit", planned)
        else:
            obs, reward, terminated, truncated, info = CalibratedContinuousWorkflowEnv.step(self, 2)
            info["transition"]["admission_rejection_reason"] = None
        event = info["transition"]
        self.execution_owner_rsu_id = "vehicle" if event.get("service_completed") else None
        event.update({
            "requested_action": requested, "action_projection_applied": requested != action,
            "time_profile": PROFILE, "execution_owner_rsu_id": self.execution_owner_rsu_id,
            "predicted_target_rsu_id": None,
            "phase_events": [{
                "phase": "vehicle_fallback_service", "location": "vehicle",
                "start_seconds": before, "end_seconds": self.clock_seconds,
                "planned_seconds": planned, "elapsed_seconds": self.clock_seconds - before,
                "planned_bytes": planned_input_bytes,
                "transferred_bytes": int(event.get("input_transfer_bytes", 0)),
                "status": "completed" if event.get("service_completed") else "trace_end",
            }],
            "migration_status": "not_requested", "failure_reason": event.get("admission_rejection_reason"),
            "backhaul_transfer_bytes": 0, "backhaul_transfer_seconds": 0.0,
            "partial_transfer_bytes": 0, "delivery_wait_seconds": 0.0,
        })
        return obs, reward, terminated, truncated, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.terminated:
            return self._observation(), 0.0, True, False, self._info()
        if self.clock_seconds >= self.trace.times_seconds[-1] - EPS:
            return self._observation(), 0.0, False, True, self._info()
        requested = int(action)
        mask = self._info()["action_mask"]
        action = requested
        if action < 0 or action >= len(mask) or not mask[action]:
            self.metrics["invalid_action_count"] += 1
            action = 2 if mask[2] else (3 if mask[3] else next(i for i, allowed in enumerate(mask) if allowed))
        if action == 2:
            return self._fallback(action, requested)

        start = self.clock_seconds
        node = self._current_node()
        node_id = str(node["node_id"])
        current = self._association(self._observed_xy())
        target = self._predicted_handoff_target()
        if current is None:
            raise RuntimeError("nonfallback action without current coverage passed mask")
        self.execution_owner_rsu_id = None
        if action == 4:
            self.metrics["migration_attempts"] += 1
        ledger: list[dict[str, Any]] = []
        cache_events: list[dict[str, Any]] = []
        input_bytes = model_bytes = state_bytes = backhaul_bytes = partial_bytes = 0
        transfer_seconds = backhaul_seconds = model_load_seconds = state_restore_seconds = 0.0
        recompute_seconds = compute_seconds = failure_wait_seconds = delivery_wait_seconds = 0.0
        service_completed = migration_success = False
        migration_status = "not_requested" if action != 4 else "not_started"
        failure_reason: str | None = None

        contact_end = start + self._physical_contact_budget_seconds()
        upload_ok, transferred, reason = self._network_phase(
            ledger, "input_upload", byte_count=int(node["input_bytes"]),
            mbps=self._actual_radio_mbps(), fixed_seconds=float(self.config["link"]["fixed_seconds"]),
            location=f"vehicle_to_{current}", contact_end=contact_end,
        )
        input_bytes += transferred
        partial_bytes += 0 if upload_ok else transferred
        transfer_seconds += ledger[-1]["elapsed_seconds"]
        if not upload_ok:
            failure_reason = reason
            migration_status = "input_upload_failed" if action == 4 else "not_requested"
            return self._finalize(
                action=action, requested=requested, node_id=node_id, current=current,
                target=target, start=start, ledger=ledger, service_completed=False,
                migration_success=False, migration_status=migration_status,
                failure_reason=failure_reason, input_bytes=input_bytes,
                model_bytes=model_bytes, state_bytes=state_bytes,
                backhaul_bytes=backhaul_bytes, partial_bytes=partial_bytes,
                transfer_seconds=transfer_seconds, backhaul_seconds=backhaul_seconds,
                model_load_seconds=model_load_seconds, state_restore_seconds=state_restore_seconds,
                recompute_seconds=recompute_seconds, compute_seconds=compute_seconds,
                failure_wait_seconds=failure_wait_seconds, delivery_wait_seconds=delivery_wait_seconds,
                cache_events=cache_events,
            )

        self.execution_owner_rsu_id = current

        prepare_rsu = current if action == 0 else target if action in {1, 4} else None
        prepared = False
        if prepare_rsu is not None:
            shadow = deepcopy(self)
            plan = shadow._admit_bundle(prepare_rsu, str(node["required_adapter"]))
            if not plan["committed"]:
                cache_events.append({**plan, "status": "admission_failed"})
                self.metrics["cache_rejections"] += 1
                if action == 4:
                    migration_status = "target_model_admission_failed"
            else:
                ok, moved, reason = self._network_phase(
                    ledger, "model_prepare_network", byte_count=int(plan["transfer_bytes"]),
                    mbps=float(self.phase_condition["backhaul_actual_mbps"]),
                    fixed_seconds=float(self.phase_contract["backhaul_fixed_seconds"]),
                    location=f"repository_to_{prepare_rsu}",
                    timeout_seconds=float(self.phase_contract["backhaul_phase_timeout_seconds"]),
                )
                model_bytes += moved
                backhaul_bytes += moved
                partial_bytes += 0 if ok else moved
                transfer_seconds += ledger[-1]["elapsed_seconds"]
                backhaul_seconds += ledger[-1]["elapsed_seconds"]
                if ok:
                    load_ok, reason = self._duration_phase(
                        ledger, "model_load", seconds=float(plan["load_seconds"]),
                        location=prepare_rsu,
                    )
                    model_load_seconds += ledger[-1]["elapsed_seconds"]
                    ok = load_ok
                if ok:
                    committed = self._admit_bundle(prepare_rsu, str(node["required_adapter"]))
                    if not committed["committed"] or committed["transfer_bytes"] != plan["transfer_bytes"]:
                        raise RuntimeError("typed cache plan changed before commit")
                    cache_events.append({**committed, "status": "completed"})
                    prepared = True
                else:
                    cache_events.append({**plan, "committed": False, "status": reason})
                    if action == 4:
                        migration_status = f"target_model_{reason}"
        elif action == 4:
            migration_status = "no_public_predicted_target"

        if self.clock_seconds >= self.trace.times_seconds[-1] - EPS:
            failure_reason = "trace_end_during_model_prepare"
            if action == 4 and migration_status == "not_started":
                migration_status = failure_reason
        else:
            bundle_ready = self._bundle_ready(current, str(node["required_adapter"]))
            if action == 0 and prepared:
                bundle_ready = True
            if not bundle_ready:
                ok, _ = self._duration_phase(
                    ledger, "failed_service_wait",
                    seconds=float(self.config["objective"]["failed_service_seconds"]),
                    location=current,
                )
                failure_wait_seconds += ledger[-1]["elapsed_seconds"]
                failure_reason = "current_bundle_not_ready" if ok else "trace_end_during_failed_service_wait"
                if action == 4 and migration_status == "not_started":
                    migration_status = "current_service_failed"
            else:
                handoff = bool(self.last_execution_rsu and current != self.last_execution_rsu)
                if handoff:
                    self.metrics["handoff_count"] += 1
                    ready = self.prepared_state.get(current)
                    state_ready = bool(
                        ready and ready.get("workflow_id") == self.instance["workflow_id"]
                        and set(self.completed).issubset(set(ready.get("completed_node_ids", [])))
                    )
                    if not state_ready:
                        self.metrics["handoff_failures"] += 1
                        planned_recompute = self._recompute_seconds(node)
                        ok, _ = self._duration_phase(
                            ledger, "prefix_recompute", seconds=planned_recompute,
                            location=current,
                        )
                        recompute_seconds += ledger[-1]["elapsed_seconds"]
                        if not ok:
                            failure_reason = "trace_end_during_recompute"
                if failure_reason is None:
                    ok, _ = self._duration_phase(
                        ledger, "current_node_compute",
                        seconds=float(node["compute_seconds"]), location=current,
                    )
                    compute_seconds += ledger[-1]["elapsed_seconds"]
                    if not ok:
                        failure_reason = "trace_end_during_compute"
                    else:
                        if node_id in self.completed:
                            self.metrics["duplicate_execution_count"] += 1
                            raise RuntimeError("node would execute twice")
                        self.completed.append(node_id)
                        self.node_index += 1
                        self.last_execution_rsu = current
                        self.metrics["completed_nodes"] += 1
                        self._touch_bundle(current, str(node["required_adapter"]))
                        service_completed = True

        if action == 4 and service_completed:
            if target is None:
                migration_status = "no_public_predicted_target"
            elif not prepared:
                if migration_status == "not_started":
                    migration_status = "target_model_not_ready"
            else:
                self.pending_delivery = {"target_rsu_id": target, "completed_node_ids": list(self.completed)}
                ok, moved, reason = self._network_phase(
                    ledger, "state_transfer", byte_count=int(node["state_bytes"]),
                    mbps=float(self.phase_condition["backhaul_actual_mbps"]),
                    fixed_seconds=float(self.phase_contract["backhaul_fixed_seconds"]),
                    location=f"{current}_to_{target}",
                    timeout_seconds=float(self.phase_contract["backhaul_phase_timeout_seconds"]),
                )
                state_bytes += moved
                backhaul_bytes += moved
                partial_bytes += 0 if ok else moved
                transfer_seconds += ledger[-1]["elapsed_seconds"]
                backhaul_seconds += ledger[-1]["elapsed_seconds"]
                if not ok:
                    migration_status = f"state_transfer_{reason}"
                else:
                    restored, reason = self._duration_phase(
                        ledger, "target_restore",
                        seconds=float(self.config["measured_time_seconds"]["state_restore_overhead"]),
                        location=target,
                    )
                    state_restore_seconds += ledger[-1]["elapsed_seconds"]
                    if not restored:
                        migration_status = "trace_end_during_target_restore"
                    else:
                        delivered, migration_status, waited = self._delivery(
                            ledger, target, decision_start=start,
                        )
                        delivery_wait_seconds += waited
                        if delivered:
                            self.prepared_state[target] = {
                                "completed_node_ids": list(self.completed),
                                "workflow_id": self.instance["workflow_id"],
                                "package_bytes": int(node["state_bytes"]),
                            }
                            migration_success = True
                            self.pending_delivery = None
                if migration_status != "trace_end_pending_delivery":
                    self.pending_delivery = None
        elif action == 4 and migration_status == "not_started":
            migration_status = "current_service_failed"

        return self._finalize(
            action=action, requested=requested, node_id=node_id, current=current,
            target=target, start=start, ledger=ledger,
            service_completed=service_completed, migration_success=migration_success,
            migration_status=migration_status, failure_reason=failure_reason,
            input_bytes=input_bytes, model_bytes=model_bytes, state_bytes=state_bytes,
            backhaul_bytes=backhaul_bytes, partial_bytes=partial_bytes,
            transfer_seconds=transfer_seconds, backhaul_seconds=backhaul_seconds,
            model_load_seconds=model_load_seconds, state_restore_seconds=state_restore_seconds,
            recompute_seconds=recompute_seconds, compute_seconds=compute_seconds,
            failure_wait_seconds=failure_wait_seconds, delivery_wait_seconds=delivery_wait_seconds,
            cache_events=cache_events,
        )

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        result.update(time_profile=PROFILE,
                      phase_condition_id=self.phase_condition["condition_id"],
                      pending_delivery=deepcopy(self.pending_delivery),
                      execution_owner_rsu_id=self.execution_owner_rsu_id)
        return result
