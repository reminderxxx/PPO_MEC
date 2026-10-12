"""Deterministic continuous-time link and compute resource scheduler.

All time values are seconds, transfer sizes are bytes, and compute work uses
the caller's fixed work unit. Callers retain the returned events as the ledger;
``snapshot`` describes only the current live scheduler state.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Any, Mapping


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return number


def _resource_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass
class _Job:
    job_id: str
    sequence: int
    resource_id: str
    total: float
    remaining: float
    submitted_at_s: float
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: str = ""
    started_at_s: float | None = None


class PhysicalRuntime:
    """Advance equally shared links and fixed-slot FIFO compute resources.

    Submissions and cancellation occur at ``now_s``. To submit at a later time,
    first call ``advance_to``. Events at the same instant are returned in job
    submission order, independent of dictionary insertion or resource order.
    Zero-sized jobs complete on the next ``advance_to`` call, including a call
    whose target is the current time.
    """

    def __init__(
        self,
        now_s: float,
        link_capacity_bytes_per_s: Mapping[str, float],
        compute_slots: Mapping[str, int],
        compute_rate_work_units_per_s: Mapping[str, float],
    ) -> None:
        self.now_s = _finite_number(now_s, "now_s")
        self.link_capacity_bytes_per_s = {
            _resource_id(key, "link_id"): _finite_number(value, f"capacity for {key}", positive=True)
            for key, value in link_capacity_bytes_per_s.items()
        }
        self.compute_slots: dict[str, int] = {}
        for key, value in compute_slots.items():
            rsu_id = _resource_id(key, "rsu_id")
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"compute_slots for {rsu_id} must be a positive integer")
            self.compute_slots[rsu_id] = value
        self.compute_rate_work_units_per_s = {
            _resource_id(key, "rsu_id"): _finite_number(value, f"compute rate for {key}", positive=True)
            for key, value in compute_rate_work_units_per_s.items()
        }
        if self.compute_slots.keys() != self.compute_rate_work_units_per_s.keys():
            raise ValueError("compute_slots and compute_rate_work_units_per_s must name the same RSUs")

        self._sequence = 0
        self._transfers: dict[str, _Job] = {}
        self._computes: dict[str, _Job] = {}
        self._compute_waiting: dict[str, deque[str]] = {key: deque() for key in self.compute_slots}
        self._compute_running: dict[str, list[str]] = {key: [] for key in self.compute_slots}

    def _new_job(
        self, *, resource_id: str, total: float, metadata: Mapping[str, Any] | None, kind: str = ""
    ) -> _Job:
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        self._sequence += 1
        return _Job(
            job_id=f"job-{self._sequence:08d}",
            sequence=self._sequence,
            resource_id=resource_id,
            total=total,
            remaining=total,
            submitted_at_s=self.now_s,
            metadata=deepcopy(dict(metadata or {})),
            kind=kind,
        )

    def submit_transfer(
        self,
        link_id: str,
        size_bytes: float,
        kind: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        link_id = _resource_id(link_id, "link_id")
        if link_id not in self.link_capacity_bytes_per_s:
            raise KeyError(f"unknown link_id: {link_id}")
        kind = _resource_id(kind, "kind")
        size = _finite_number(size_bytes, "size_bytes")
        job = self._new_job(resource_id=link_id, total=size, metadata=metadata, kind=kind)
        job.started_at_s = self.now_s
        self._transfers[job.job_id] = job
        return job.job_id

    def submit_compute(
        self, rsu_id: str, work_units: float, metadata: Mapping[str, Any] | None = None
    ) -> str:
        rsu_id = _resource_id(rsu_id, "rsu_id")
        if rsu_id not in self.compute_slots:
            raise KeyError(f"unknown rsu_id: {rsu_id}")
        work = _finite_number(work_units, "work_units")
        job = self._new_job(resource_id=rsu_id, total=work, metadata=metadata)
        self._computes[job.job_id] = job
        if len(self._compute_running[rsu_id]) < self.compute_slots[rsu_id]:
            job.started_at_s = self.now_s
            self._compute_running[rsu_id].append(job.job_id)
        else:
            self._compute_waiting[rsu_id].append(job.job_id)
        return job.job_id

    def _event(self, job: _Job, *, resource_type: str, status: str) -> dict[str, Any]:
        completed = job.total - job.remaining
        if status == "completed":
            completed = job.total
        return {
            "event_type": f"{resource_type}_{status}",
            "status": status,
            "job_id": job.job_id,
            "time_s": self.now_s,
            "resource_type": resource_type,
            "resource_id": job.resource_id,
            "kind": job.kind if resource_type == "transfer" else "compute",
            "submitted_at_s": job.submitted_at_s,
            "started_at_s": job.started_at_s,
            "completed_bytes": completed if resource_type == "transfer" else 0.0,
            "total_bytes": job.total if resource_type == "transfer" else 0.0,
            "completed_work_units": completed if resource_type == "compute" else 0.0,
            "total_work_units": job.total if resource_type == "compute" else 0.0,
            "metadata": deepcopy(job.metadata),
        }

    def cancel_transfer(self, job_id: str) -> dict[str, Any]:
        """Cancel a live transfer and retain bytes consumed through ``now_s``."""
        try:
            job = self._transfers.pop(job_id)
        except KeyError as exc:
            raise KeyError(f"unknown or completed transfer job_id: {job_id}") from exc
        return self._event(job, resource_type="transfer", status="cancelled")

    def _running_jobs(self) -> list[tuple[_Job, float]]:
        result: list[tuple[_Job, float]] = []
        per_link: dict[str, list[_Job]] = {key: [] for key in self.link_capacity_bytes_per_s}
        for job in self._transfers.values():
            per_link[job.resource_id].append(job)
        for link_id, jobs in per_link.items():
            if jobs:
                rate = self.link_capacity_bytes_per_s[link_id] / len(jobs)
                result.extend((job, rate) for job in jobs)
        for rsu_id, job_ids in self._compute_running.items():
            rate = self.compute_rate_work_units_per_s[rsu_id]
            result.extend((self._computes[job_id], rate) for job_id in job_ids)
        return result

    def _start_waiting(self) -> None:
        for rsu_id, waiting in self._compute_waiting.items():
            running = self._compute_running[rsu_id]
            while waiting and len(running) < self.compute_slots[rsu_id]:
                job_id = waiting.popleft()
                self._computes[job_id].started_at_s = self.now_s
                running.append(job_id)

    def next_completion_time(self) -> float | None:
        """Earliest completion under the current link shares and compute slots.

        The estimate becomes stale after any new submission or cancellation.
        A zero-work active job is due at ``now_s``. Queued compute jobs are not
        predicted until they acquire a slot.
        """
        running = self._running_jobs()
        if not running:
            return None
        duration = min(job.remaining / rate for job, rate in running)
        return self.now_s + duration

    def advance_to(self, time_s: float) -> list[dict[str, Any]]:
        """Advance all resources and return ordered completion events through time_s."""
        target = _finite_number(time_s, "time_s")
        if target < self.now_s:
            raise ValueError("time_s must not precede now_s")
        events: list[dict[str, Any]] = []
        while True:
            running = self._running_jobs()
            if not running:
                break
            next_job, next_rate = min(running, key=lambda pair: (pair[0].remaining / pair[1], pair[0].sequence))
            duration = next_job.remaining / next_rate
            next_time = self.now_s + duration
            if next_time > target:
                break
            elapsed = next_time - self.now_s
            due = {
                job.job_id
                for job, rate in running
                if job.remaining / rate <= duration
            }
            for job, rate in running:
                job.remaining = max(0.0, job.remaining - elapsed * rate)
                if job.job_id in due:
                    job.remaining = 0.0
            self.now_s = next_time
            completed_jobs = sorted((job for job, _ in running if job.job_id in due), key=lambda job: job.sequence)
            for job in completed_jobs:
                if job.job_id in self._transfers:
                    del self._transfers[job.job_id]
                    events.append(self._event(job, resource_type="transfer", status="completed"))
                else:
                    del self._computes[job.job_id]
                    self._compute_running[job.resource_id].remove(job.job_id)
                    events.append(self._event(job, resource_type="compute", status="completed"))
            self._start_waiting()
        elapsed = target - self.now_s
        if elapsed:
            for job, rate in self._running_jobs():
                job.remaining = max(0.0, job.remaining - elapsed * rate)
        self.now_s = target
        events.sort(key=lambda event: (event["time_s"], int(event["job_id"].split("-")[1])))
        return events

    def snapshot(self) -> dict[str, Any]:
        """Return a detached view of all in-flight and queued work."""
        def describe(job: _Job, state: str) -> dict[str, Any]:
            return {
                "job_id": job.job_id,
                "resource_id": job.resource_id,
                "kind": job.kind if state == "transfer" else "compute",
                "state": "active" if state == "transfer" else state,
                "submitted_at_s": job.submitted_at_s,
                "started_at_s": job.started_at_s,
                "total": job.total,
                "completed": job.total - job.remaining,
                "remaining": job.remaining,
                "metadata": deepcopy(job.metadata),
            }

        transfers = sorted(self._transfers.values(), key=lambda job: job.sequence)
        computes = sorted(self._computes.values(), key=lambda job: job.sequence)
        running_ids = {job_id for jobs in self._compute_running.values() for job_id in jobs}
        return {
            "now_s": self.now_s,
            "link_capacity_bytes_per_s": dict(self.link_capacity_bytes_per_s),
            "compute_slots": dict(self.compute_slots),
            "compute_rate_work_units_per_s": dict(self.compute_rate_work_units_per_s),
            "transfers": [describe(job, "transfer") for job in transfers],
            "computes": [describe(job, "running" if job.job_id in running_ids else "queued") for job in computes],
        }
