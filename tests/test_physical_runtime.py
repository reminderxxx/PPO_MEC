"""Local contract tests for the deterministic physical resource scheduler."""

from __future__ import annotations

import math

import pytest

from src.envs.core.physical_runtime import PhysicalRuntime


def runtime() -> PhysicalRuntime:
    return PhysicalRuntime(
        now_s=0.0,
        link_capacity_bytes_per_s={"backhaul": 10.0, "uplink": 5.0},
        compute_slots={"rsu-a": 1, "rsu-b": 2},
        compute_rate_work_units_per_s={"rsu-a": 2.0, "rsu-b": 1.0},
    )


def test_shared_link_reallocates_without_losing_transferred_bytes() -> None:
    clock = runtime()
    first = clock.submit_transfer("backhaul", 100.0, "model", {"object_id": "base"})
    assert clock.advance_to(2.0) == []
    assert clock.snapshot()["transfers"][0]["completed"] == pytest.approx(20.0)

    second = clock.submit_transfer("backhaul", 20.0, "input", {"object_id": "request"})
    assert clock.advance_to(5.0) == []
    state = {row["job_id"]: row for row in clock.snapshot()["transfers"]}
    assert state[first]["completed"] == pytest.approx(35.0)
    assert state[second]["completed"] == pytest.approx(15.0)

    events = clock.advance_to(6.0)
    assert [(event["job_id"], event["time_s"]) for event in events] == [(second, 6.0)]
    assert events[0]["completed_bytes"] == 20.0
    assert events[0]["metadata"] == {"object_id": "request"}
    assert clock.snapshot()["transfers"][0]["completed"] == pytest.approx(40.0)
    events = clock.advance_to(12.0)
    assert [(event["job_id"], event["time_s"]) for event in events] == [(first, 12.0)]
    assert events[0]["completed_bytes"] == 100.0


def test_next_completion_time_updates_when_link_share_changes() -> None:
    clock = runtime()
    assert clock.next_completion_time() is None
    first = clock.submit_transfer("backhaul", 10.0, "model")
    assert clock.next_completion_time() == pytest.approx(1.0)
    second = clock.submit_transfer("backhaul", 10.0, "model")
    assert clock.next_completion_time() == pytest.approx(2.0)
    clock.advance_to(0.5)
    assert clock.next_completion_time() == pytest.approx(2.0)
    clock.cancel_transfer(second)
    assert clock.next_completion_time() == pytest.approx(1.25)
    assert clock.advance_to(clock.next_completion_time())[0]["job_id"] == first
    assert clock.next_completion_time() is None


def test_event_loop_can_submit_successor_before_control_step_boundary() -> None:
    clock = runtime()
    predecessor = clock.submit_transfer("backhaul", 10.0, "input", {"task": "a"})
    seen = []
    horizon = 5.0
    while (next_time := clock.next_completion_time()) is not None and next_time <= horizon:
        for event in clock.advance_to(min(next_time, horizon)):
            seen.append(event)
            if event["job_id"] == predecessor:
                clock.submit_compute("rsu-a", 4.0, {"task": "a"})
    clock.advance_to(horizon)
    assert [(event["event_type"], event["time_s"]) for event in seen] == [
        ("transfer_completed", 1.0), ("compute_completed", 3.0)
    ]
    assert seen[1]["started_at_s"] == 1.0


def test_cancellation_keeps_consumed_bytes_and_releases_link_share() -> None:
    clock = runtime()
    first = clock.submit_transfer("backhaul", 40.0, "model")
    second = clock.submit_transfer("backhaul", 40.0, "model")
    clock.advance_to(2.0)
    cancelled = clock.cancel_transfer(second)
    assert cancelled["event_type"] == "transfer_cancelled"
    assert cancelled["time_s"] == 2.0
    assert cancelled["completed_bytes"] == pytest.approx(10.0)
    assert cancelled["total_bytes"] == 40.0
    with pytest.raises(KeyError):
        clock.cancel_transfer(second)
    events = clock.advance_to(5.0)
    assert [event["job_id"] for event in events] == [first]
    assert events[0]["completed_bytes"] == 40.0


def test_compute_slots_fifo_and_completion_starts_next_at_same_time() -> None:
    clock = runtime()
    first = clock.submit_compute("rsu-a", 4.0, {"task": 1})
    second = clock.submit_compute("rsu-a", 2.0, {"task": 2})
    third = clock.submit_compute("rsu-a", 2.0, {"task": 3})
    assert [(row["job_id"], row["state"]) for row in clock.snapshot()["computes"]] == [
        (first, "running"), (second, "queued"), (third, "queued")
    ]
    events = clock.advance_to(4.0)
    assert [(event["job_id"], event["time_s"]) for event in events] == [
        (first, 2.0), (second, 3.0), (third, 4.0)
    ]
    assert [event["started_at_s"] for event in events] == [0.0, 2.0, 3.0]
    assert all(event["completed_bytes"] == 0.0 for event in events)
    assert [event["completed_work_units"] for event in events] == [4.0, 2.0, 2.0]


def test_simultaneous_events_have_global_submission_order() -> None:
    clock = runtime()
    a = clock.submit_transfer("backhaul", 10.0, "model")
    b = clock.submit_compute("rsu-a", 2.0)
    c = clock.submit_transfer("uplink", 5.0, "input")
    events = clock.advance_to(1.0)
    assert [event["job_id"] for event in events] == [a, b, c]
    assert [event["event_type"] for event in events] == [
        "transfer_completed", "compute_completed", "transfer_completed"
    ]
    assert all(event["time_s"] == 1.0 for event in events)


def test_two_compute_slots_run_concurrently() -> None:
    clock = runtime()
    first = clock.submit_compute("rsu-b", 2.0)
    second = clock.submit_compute("rsu-b", 2.0)
    third = clock.submit_compute("rsu-b", 1.0)
    events = clock.advance_to(3.0)
    assert [(event["job_id"], event["time_s"]) for event in events] == [
        (first, 2.0), (second, 2.0), (third, 3.0)
    ]


def test_zero_work_finishes_when_current_time_is_advanced() -> None:
    clock = runtime()
    transfer = clock.submit_transfer("backhaul", 0.0, "state")
    compute = clock.submit_compute("rsu-a", 0.0)
    events = clock.advance_to(0.0)
    assert [event["job_id"] for event in events] == [transfer, compute]
    assert all(event["time_s"] == 0.0 for event in events)


def test_zero_work_queued_at_same_completion_time_keeps_submission_order() -> None:
    clock = runtime()
    running = clock.submit_compute("rsu-a", 2.0)
    queued_zero = clock.submit_compute("rsu-a", 0.0)
    transfer = clock.submit_transfer("backhaul", 10.0, "model")
    events = clock.advance_to(1.0)
    assert [event["job_id"] for event in events] == [running, queued_zero, transfer]
    assert all(event["time_s"] == 1.0 for event in events)


def test_metadata_and_snapshot_are_detached() -> None:
    clock = runtime()
    metadata = {"tags": ["original"]}
    job_id = clock.submit_transfer("backhaul", 10.0, "model", metadata)
    metadata["tags"].append("later")
    snap = clock.snapshot()
    snap["transfers"][0]["metadata"]["tags"].append("snapshot")
    snap["link_capacity_bytes_per_s"]["backhaul"] = 1.0
    event = clock.advance_to(1.0)[0]
    assert event["job_id"] == job_id
    assert event["metadata"] == {"tags": ["original"]}
    assert event["time_s"] == 1.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -1.0, True, "bad"])
def test_rejects_invalid_time_size_and_work(bad: object) -> None:
    clock = runtime()
    with pytest.raises(ValueError):
        clock.advance_to(bad)
    with pytest.raises(ValueError):
        clock.submit_transfer("backhaul", bad, "model")
    with pytest.raises(ValueError):
        clock.submit_compute("rsu-a", bad)


def test_rejects_invalid_config_unknown_resources_and_backward_time() -> None:
    with pytest.raises(ValueError):
        PhysicalRuntime(0, {"link": 0}, {}, {})
    with pytest.raises(ValueError):
        PhysicalRuntime(0, {}, {"rsu": 1}, {})
    with pytest.raises(ValueError):
        PhysicalRuntime(0, {}, {"rsu": True}, {"rsu": 1})
    clock = runtime()
    with pytest.raises(KeyError):
        clock.submit_transfer("missing", 1, "model")
    with pytest.raises(KeyError):
        clock.submit_compute("missing", 1)
    clock.advance_to(2.0)
    with pytest.raises(ValueError):
        clock.advance_to(1.0)
