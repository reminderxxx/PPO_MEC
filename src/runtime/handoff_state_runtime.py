"""Opt-in, node-boundary execution backend; not a replacement for the simulator.

Real model execution and localhost TCP transport are measured separately from
accelerated mobility replay and the legacy simulator's reward clock.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import base64
import json
from pathlib import Path
import socket
import selectors
import struct
import subprocess
import sys
import time
from typing import Any

from src.runtime.workflow_suffix_recovery import (
    EXPECTED_WORKFLOW_ID, canonical_json_bytes, sha256_bytes,
)

RUNTIME_VERSION = "ppo_mec.handoff_execution.v1"
MAX_PACKET_BYTES = 16 * 1024 * 1024
ARMS = ("restart", "recover", "prepare_recover")


def receive_exact(connection: socket.socket, count: int) -> bytes:
    parts = bytearray()
    while len(parts) < count:
        block = connection.recv(min(65536, count - len(parts)))
        if not block:
            raise ConnectionError("truncated handoff packet")
        parts.extend(block)
    return bytes(parts)


def send_packet(port: int, packet: bytes, *, mbps: float, latency_seconds: float) -> dict[str, Any]:
    if not 0 < len(packet) <= MAX_PACKET_BYTES:
        raise ValueError("invalid packet length")
    if mbps <= 0 or latency_seconds < 0:
        raise ValueError("invalid link configuration")
    started = time.monotonic()
    wire = struct.pack("!Q", len(packet)) + packet
    with socket.create_connection(("127.0.0.1", port), timeout=120) as connection:
        connection.settimeout(120)
        time.sleep(latency_seconds)
        send_started = time.monotonic()
        sent = 0
        for offset in range(0, len(wire), 16384):
            block = wire[offset:offset + 16384]
            # Pace every chunk, including the final one: a sub-chunk packet is
            # not accidentally given unlimited bandwidth.
            deadline = send_started + (sent + len(block)) * 8 / (mbps * 1e6)
            time.sleep(max(0.0, deadline - time.monotonic()))
            connection.sendall(block)
            sent += len(block)
        acknowledgement = receive_exact(connection, 64).decode("ascii")
    expected = sha256_bytes(packet)
    if acknowledgement != expected:
        raise ValueError("transport acknowledgement hash mismatch")
    return {"transport": "paced_localhost_tcp", "packet_bytes": len(packet),
            "forward_bytes": len(wire), "ack_bytes": 64,
            "packet_sha256": expected, "mbps": mbps,
            "injected_one_way_latency_seconds": latency_seconds,
            "sender_wall_seconds": time.monotonic() - started}


def receive_packet(listener: socket.socket) -> tuple[bytes, dict[str, Any]]:
    started = time.monotonic()
    listener.settimeout(120)
    with listener.accept()[0] as connection:
        connection.settimeout(120)
        length = struct.unpack("!Q", receive_exact(connection, 8))[0]
        if not 0 < length <= MAX_PACKET_BYTES:
            raise ValueError("invalid received packet length")
        packet = receive_exact(connection, length)
        digest = sha256_bytes(packet)
        connection.sendall(digest.encode("ascii"))
    listener.close()
    return packet, {"packet_bytes": length, "forward_bytes": length + 8,
                    "ack_bytes": 64, "packet_sha256": digest,
                    "receiver_wall_seconds": time.monotonic() - started}


def encode_files(kind: str, files: dict[str, bytes]) -> bytes:
    return canonical_json_bytes({"kind": kind, "files": {
        name: base64.b64encode(value).decode("ascii") for name, value in files.items()}})


def decode_files(packet: bytes) -> tuple[str, dict[str, bytes]]:
    value = json.loads(packet)
    kind = value.get("kind")
    expected = {"state": {"state.json", "manifest.json"}, "input": {"input.png"}}
    if kind not in expected or set(value.get("files", {})) != expected[kind]:
        raise ValueError("invalid packet file allowlist")
    return kind, {name: base64.b64decode(data, validate=True)
                  for name, data in value["files"].items()}


class ModelWorker:
    """One RSU process, isolated command stream and declared dynamic input."""

    def __init__(self, *, python: str, script: Path, plan_path: Path,
                 directory: Path, rsu_id: str, source: bool) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.log = (directory / "stderr.log").open("w")
        self.command = [python, str(script), "--worker", "--plan", str(plan_path),
                        "--directory", str(directory), "--rsu-id", rsu_id]
        if source:
            self.command.append("--source")
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        self.commands: list[dict[str, Any]] = []
        self.pid = self.process.pid

    def call(self, operation: str, **kwargs: Any) -> dict[str, Any]:
        started = time.monotonic()
        request = {"operation": operation, **kwargs}
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=180):
                raise TimeoutError(f"worker operation exceeded 180 seconds: {operation}")
            line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"worker {self.pid} exited: {self.directory / 'stderr.log'}")
        result = json.loads(line)
        entry = {"request": request, "response": result,
                 "parent_wall_seconds": time.monotonic() - started}
        self.commands.append(entry)
        if result.get("status") != "PASS":
            raise RuntimeError(f"worker operation failed: {result}")
        return result

    def close(self) -> None:
        if self.process.poll() is None:
            if self.process.stdin:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        self.log.close()


class HandoffStateRuntime:
    """Only the declared two-node technical DAG is supported in version 1.

    reset/prepare/on_handoff/execute_node are called by the native environment.
    A failed transfer or model call fails closed before logical node completion.
    """

    def __init__(self, *, arm: str, workers: dict[str, ModelWorker], source_rsu: str,
                 mbps: float, latency_seconds: float) -> None:
        if arm not in ARMS:
            raise ValueError("unknown handoff execution arm")
        self.arm, self.workers, self.owner = arm, workers, source_rsu
        self.mbps, self.latency_seconds = mbps, latency_seconds
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.preparations: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.completed: list[str] = []
        self.bound = False
        self.handoff_started: float | None = None
        self.started: float | None = None

    def reset(self, *, workflow: dict[str, Any], vehicle_id: str | None,
              rsu_id: str | None) -> None:
        if self.bound:
            raise ValueError("runtime instances are single-episode; do not reset a consumed backend")
        if workflow["workflow_id"] != EXPECTED_WORKFLOW_ID:
            raise ValueError("execution workflow identity mismatch")
        if [n["node_id"] for n in workflow["nodes"]] != ["n0", "n1"]:
            raise ValueError("execution node contract mismatch")
        if [list(edge) for edge in workflow["edges"]] != [["n0", "n1"]]:
            raise ValueError("execution DAG contract mismatch")
        if (workflow.get("completed_node_ids") or workflow.get("is_completed")
                or workflow.get("current_node_id") != "n0"
                or workflow.get("execution_order") != ["n0", "n1"]):
            raise ValueError("execution workflow must start from the declared unexecuted prefix")
        if any(node["required_adapter"] != "adapter_a" or
               node["required_base_model"] != "veh_base_v1" for node in workflow["nodes"]):
            raise ValueError("logical model mapping does not match the frozen technical backend")
        if not vehicle_id or rsu_id != self.owner:
            raise ValueError("execution source/subject binding mismatch")
        self.vehicle_id, self.bound = vehicle_id, True
        self.started = time.monotonic()

    def prepare(self, *, control: Any, state: dict[str, Any]) -> None:
        if self.arm != "prepare_recover" or control.migration_action.get("mode") != "prepare":
            return
        target = control.migration_action.get("expected_target_rsu_id")
        if target not in self.workers or target == self.owner or target in self.preparations:
            return
        self.preparations[target] = self.pool.submit(self.workers[target].call, "prepare")
        self.events.append({"kind": "prepare_issued", "target_rsu": target,
                            "source": "control.migration_action", "at": time.monotonic()})

    def on_handoff(self, event: dict[str, Any]) -> None:
        if event.get("vehicle_id") != self.vehicle_id or event.get("event_type") != "handoff":
            return
        target = event["current_rsu_id"]
        if event["previous_rsu_id"] != self.owner or target not in self.workers:
            raise ValueError("handoff owner/target identity mismatch")
        if self.completed != ["n0"] or self.handoff_started is not None:
            raise ValueError("handoff requires exactly one committed prefix; duplicate handoff rejected")
        started = time.monotonic()
        self.handoff_started = started
        destination = self.workers[target]
        prepared = target in self.preparations
        preparation_wait_started = time.monotonic()
        if prepared:
            self.preparations[target].result(timeout=120)
        else:
            destination.call("prepare")
        preparation_wait = time.monotonic() - preparation_wait_started
        port = destination.call("listen")["port"]
        receiving = self.pool.submit(destination.call, "receive")
        kind = "input" if self.arm == "restart" else "state"
        sender = self.workers[self.owner].call("send", port=port, kind=kind,
            mbps=self.mbps, latency_seconds=self.latency_seconds)
        receiver = receiving.result(timeout=120)
        if sender["transport"]["packet_sha256"] != receiver["transport"]["packet_sha256"]:
            raise ValueError("source/target transport mismatch")
        self.events.append({"kind": "handoff_completed", "event": deepcopy(event),
            "source_rsu": self.owner, "target_rsu": target,
            "preparation_issued_before_handoff": prepared,
            "preparation_wait_seconds": preparation_wait,
            "handoff_to_state_ready_seconds": time.monotonic() - started,
            "sender": sender, "receiver": receiver})
        self.owner = target

    def execute_node(self, *, node_id: str, rsu_id: str | None) -> dict[str, Any]:
        if not self.bound or rsu_id != self.owner:
            raise ValueError("node execution RSU binding mismatch")
        expected = "n0" if not self.completed else "n1"
        if node_id != expected or len(self.completed) >= 2:
            raise ValueError("node execution order mismatch")
        if node_id == "n0":
            result = self.workers[self.owner].call("n0")
        else:
            operation = "restart" if self.arm == "restart" and self.handoff_started else "suffix"
            result = self.workers[self.owner].call(operation)
        self.completed.append(node_id)
        event = {"kind": "node_completed", "node_id": node_id, "rsu_id": rsu_id,
                 "worker_pid": self.workers[self.owner].pid, "result": result}
        self.events.append(event)
        return event

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        return {"schema_version": RUNTIME_VERSION, "arm": self.arm,
                "completed_nodes": list(self.completed), "owner_rsu": self.owner,
                "workflow_wall_seconds": now - self.started if self.started else None,
                "handoff_to_completion_seconds": now - self.handoff_started if self.handoff_started else None,
                "events": deepcopy(self.events), "reward_clock": "legacy_simulator_not_measured_wall_time"}

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        for worker in self.workers.values():
            worker.close()
