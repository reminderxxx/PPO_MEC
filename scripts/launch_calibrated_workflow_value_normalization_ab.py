"""Persistent, create-only supervisor for the calibrated-workflow PopArt A/B.

The launcher process exits only after a detached supervisor has observed a
child-owned entry receipt.  The detached supervisor then owns stdout/stderr,
the exit sidecar, timeout enforcement, and the terminal receipt.  It never
retries a child.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INTERPRETER = "/Users/howen/Projects/PPO_MEC/.venv/bin/python"
SCIENTIFIC_RUNNER = ROOT_DIR / "scripts/run_calibrated_workflow_value_normalization_ab.py"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _git_status() -> list[str]:
    output = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT_DIR,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return [line for line in output.splitlines() if line.strip()]


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _inventory(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"artifact_integrity.json", ".artifact_integrity.json.tmp"}
    )
    _write_json(
        root / "artifact_integrity.json",
        {
            "schema_version": "persistent_ab_supervisor_integrity_v1",
            "files": [
                {
                    "path": str(path.relative_to(root)),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
            ],
        },
    )


def _frozen_interpreter(value: str) -> str:
    # Deliberately do not resolve symlinks: the authorized venv launch path is
    # part of the execution identity.
    if value != DEFAULT_INTERPRETER:
        raise ValueError(
            f"interpreter identity drift: expected {DEFAULT_INTERPRETER}, got {value}"
        )
    path = Path(value)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"frozen interpreter is not executable: {value}")
    return value


def _build_scientific_plan(args: argparse.Namespace) -> dict[str, Any]:
    interpreter = _frozen_interpreter(args.interpreter)
    authorization_path = (ROOT_DIR / args.authorization_config).resolve()
    authorization = _read_json(authorization_path)
    output_root = Path(args.output_root).resolve()
    supervisor_root = Path(args.supervisor_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"scientific output root already exists: {output_root}")
    if output_root.name != str(authorization["output_run_id"]):
        raise ValueError("authorization output_run_id does not match output root")
    return {
        "schema_version": "persistent_ab_supervisor_plan_v1",
        "kind": "scientific",
        "run_id": str(authorization["output_run_id"]),
        "created_at": _now(),
        "expected_commit": args.expected_commit,
        "interpreter": interpreter,
        "cwd": str(ROOT_DIR),
        "supervisor_root": str(supervisor_root),
        "output_root": str(output_root),
        "authorization_config": str(authorization_path.relative_to(ROOT_DIR)),
        "authorization_sha256": _sha256(authorization_path),
        "child_argv": [
            interpreter,
            str(SCIENTIFIC_RUNNER),
            "--authorization-config",
            str(authorization_path.relative_to(ROOT_DIR)),
            "--output-root",
            str(output_root),
        ],
        "entry_receipt": str(output_root / "runner_entered.json"),
        "expected_exit_code": 0,
        "entry_timeout_seconds": 30.0,
        "wall_clock_cap_seconds": float(
            authorization["scientific_wall_clock_cap_seconds"]
        ),
        "automatic_retry": False,
        "formal_or_holdout_reads": 0,
    }


def _build_probe_plan(args: argparse.Namespace) -> dict[str, Any]:
    interpreter = _frozen_interpreter(args.interpreter)
    supervisor_root = Path(args.supervisor_root).resolve()
    exit_code = 0 if args.probe_case == "success" else 7
    entry_marker = supervisor_root / "probe_entered.json"
    return {
        "schema_version": "persistent_ab_supervisor_plan_v1",
        "kind": "host_acceptance",
        "run_id": f"persistent_supervisor_{args.probe_case}",
        "probe_case": args.probe_case,
        "created_at": _now(),
        "expected_commit": args.expected_commit,
        "interpreter": interpreter,
        "cwd": str(ROOT_DIR),
        "supervisor_root": str(supervisor_root),
        "output_root": None,
        "child_argv": [
            interpreter,
            str(Path(__file__).resolve()),
            "--mode",
            "probe-child",
            "--entry-marker",
            str(entry_marker),
            "--probe-exit-code",
            str(exit_code),
        ],
        "entry_receipt": str(entry_marker),
        "expected_exit_code": exit_code,
        "entry_timeout_seconds": 10.0,
        "wall_clock_cap_seconds": 10.0,
        "automatic_retry": False,
    }


def _validate_launch_identity(plan: dict[str, Any]) -> None:
    if bool(plan.get("automatic_retry")):
        raise RuntimeError("automatic retry is forbidden")
    actual_commit = _current_commit()
    if actual_commit != str(plan["expected_commit"]):
        raise RuntimeError(
            f"execution commit mismatch: expected {plan['expected_commit']}, got {actual_commit}"
        )
    dirty = _git_status()
    if dirty:
        raise RuntimeError(f"execution worktree is not clean: {dirty}")
    _frozen_interpreter(str(plan["interpreter"]))


def _dispatch(plan: dict[str, Any]) -> dict[str, Any]:
    supervisor_root = Path(plan["supervisor_root"])
    if supervisor_root.exists():
        raise FileExistsError(f"supervisor root already exists: {supervisor_root}")
    if plan.get("output_root") and Path(plan["output_root"]).exists():
        raise FileExistsError(f"output root already exists: {plan['output_root']}")
    _validate_launch_identity(plan)
    supervisor_root.mkdir(parents=True)
    plan_path = supervisor_root / "launch_plan.json"
    _write_json(plan_path, plan)
    stdout_path = supervisor_root / "supervisor_stdout.log"
    stderr_path = supervisor_root / "supervisor_stderr.log"
    with stdout_path.open("ab", buffering=0) as stdout_handle, stderr_path.open(
        "ab", buffering=0
    ) as stderr_handle:
        supervisor = subprocess.Popen(
            [
                str(plan["interpreter"]),
                str(Path(__file__).resolve()),
                "--mode",
                "supervise",
                "--plan",
                str(plan_path),
            ],
            cwd=ROOT_DIR,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
            close_fds=True,
        )
    dispatch = {
        "schema_version": "persistent_ab_dispatch_v1",
        "status": "dispatched",
        "dispatched_at": _now(),
        "run_id": plan["run_id"],
        "interpreter": plan["interpreter"],
        "supervisor_pid": supervisor.pid,
        "plan_sha256": _sha256(plan_path),
        "automatic_retry": False,
    }
    _write_json(supervisor_root / "dispatch_receipt.json", dispatch)
    deadline = time.monotonic() + float(plan["entry_timeout_seconds"]) + 5.0
    ack_path = supervisor_root / "launch_ack.json"
    terminal_path = supervisor_root / "terminal_receipt.json"
    while time.monotonic() < deadline:
        if ack_path.exists():
            ack = _read_json(ack_path)
            if ack.get("status") != "entered_child":
                raise RuntimeError(f"invalid launch ACK: {ack}")
            return ack
        if terminal_path.exists():
            raise RuntimeError(
                f"supervisor terminated before child entry: {_read_json(terminal_path)}"
            )
        time.sleep(0.05)
    raise TimeoutError("detached supervisor did not produce a child-entry ACK")


def _terminate_process_group(child: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    child.wait(timeout=5.0)


def _supervise(plan_path: Path) -> int:
    plan = _read_json(plan_path)
    supervisor_root = Path(plan["supervisor_root"])
    terminal_path = supervisor_root / "terminal_receipt.json"
    started_monotonic = time.monotonic()
    child: subprocess.Popen[bytes] | None = None
    terminal: dict[str, Any]
    try:
        _validate_launch_identity(plan)
        _write_json(
            supervisor_root / "supervisor_started.json",
            {
                "status": "started",
                "started_at": _now(),
                "supervisor_pid": os.getpid(),
                "run_id": plan["run_id"],
                "interpreter": plan["interpreter"],
                "plan_sha256": _sha256(plan_path),
            },
        )
        child_stdout = supervisor_root / "child_stdout.log"
        child_stderr = supervisor_root / "child_stderr.log"
        with child_stdout.open("ab", buffering=0) as stdout_handle, child_stderr.open(
            "ab", buffering=0
        ) as stderr_handle:
            child = subprocess.Popen(
                list(plan["child_argv"]),
                cwd=str(plan["cwd"]),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
                close_fds=True,
            )
        child_started_at = _now()
        _write_json(
            supervisor_root / "child_started.json",
            {
                "status": "started",
                "started_at": child_started_at,
                "child_pid": child.pid,
                "interpreter": plan["interpreter"],
                "argv": plan["child_argv"],
                "automatic_retry": False,
            },
        )
        entry_path = Path(plan["entry_receipt"])
        entry_deadline = time.monotonic() + float(plan["entry_timeout_seconds"])
        entry_payload: dict[str, Any] | None = None
        while time.monotonic() < entry_deadline:
            if entry_path.exists():
                entry_payload = _read_json(entry_path)
                break
            if child.poll() is not None:
                break
            time.sleep(0.05)
        if entry_payload is None:
            raise RuntimeError(
                "child exited or timed out before writing the entry receipt"
            )
        ack = {
            "schema_version": "persistent_ab_launch_ack_v1",
            "status": "entered_child",
            "acknowledged_at": _now(),
            "run_id": plan["run_id"],
            "kind": plan["kind"],
            "interpreter": plan["interpreter"],
            "supervisor_pid": os.getpid(),
            "child_pid": child.pid,
            "child_started_at": child_started_at,
            "entered_runner_at": entry_payload.get(
                "entered_at", entry_payload.get("started_at")
            ),
            "entry_receipt": str(entry_path),
            "entry_payload": entry_payload,
            "automatic_retry": False,
        }
        _write_json(supervisor_root / "launch_ack.json", ack)
        timed_out = False
        try:
            return_code = child.wait(timeout=float(plan["wall_clock_cap_seconds"]))
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(child)
            return_code = child.returncode
        exit_sidecar = {
            "schema_version": "persistent_ab_exit_sidecar_v1",
            "recorded_at": _now(),
            "run_id": plan["run_id"],
            "child_pid": child.pid,
            "return_code": return_code,
            "expected_exit_code": int(plan["expected_exit_code"]),
            "timed_out": timed_out,
            "wall_seconds": time.monotonic() - started_monotonic,
            "automatic_retry": False,
        }
        _write_json(supervisor_root / "exit_sidecar.json", exit_sidecar)
        expected_exit = return_code == int(plan["expected_exit_code"])
        scientific_complete = True
        child_receipt: dict[str, Any] | None = None
        if plan["kind"] == "scientific":
            completion_path = Path(plan["output_root"]) / "completion_receipt.json"
            failure_path = Path(plan["output_root"]) / "failure_receipt.json"
            if completion_path.exists():
                child_receipt = _read_json(completion_path)
                scientific_complete = bool(
                    child_receipt.get("scientific_execution_complete")
                    and child_receipt.get("status") == "complete"
                )
            elif failure_path.exists():
                child_receipt = _read_json(failure_path)
                scientific_complete = False
            else:
                scientific_complete = False
        passed = bool(expected_exit and not timed_out and scientific_complete)
        terminal = {
            "schema_version": "persistent_ab_terminal_receipt_v1",
            "status": "PASS" if passed else "FAIL",
            "classification": (
                "SCIENTIFIC_CHILD_COMPLETE"
                if passed and plan["kind"] == "scientific"
                else "HOST_ACCEPTANCE_EXPECTED_EXIT_OBSERVED"
                if passed
                else "CHILD_EXECUTION_FAILURE_NO_RETRY"
            ),
            "finished_at": _now(),
            "run_id": plan["run_id"],
            "kind": plan["kind"],
            "supervisor_pid": os.getpid(),
            "child_pid": child.pid,
            "return_code": return_code,
            "expected_exit_code": int(plan["expected_exit_code"]),
            "timed_out": timed_out,
            "entry_acknowledged": True,
            "child_receipt": child_receipt,
            "automatic_retry": False,
        }
    except BaseException as error:
        if child is not None and child.poll() is None:
            _terminate_process_group(child)
        terminal = {
            "schema_version": "persistent_ab_terminal_receipt_v1",
            "status": "FAIL",
            "classification": "SUPERVISOR_FAILURE_NO_RETRY",
            "finished_at": _now(),
            "run_id": plan.get("run_id"),
            "kind": plan.get("kind"),
            "supervisor_pid": os.getpid(),
            "child_pid": child.pid if child is not None else None,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "automatic_retry": False,
        }
    _write_json(terminal_path, terminal)
    _inventory(supervisor_root)
    return 0 if terminal["status"] == "PASS" else 1


def _verify(supervisor_root: Path, wait_seconds: float, allow_running: bool) -> int:
    deadline = time.monotonic() + max(wait_seconds, 0.0)
    terminal_path = supervisor_root / "terminal_receipt.json"
    while not terminal_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    required_start = [
        supervisor_root / "launch_plan.json",
        supervisor_root / "dispatch_receipt.json",
        supervisor_root / "supervisor_started.json",
        supervisor_root / "child_started.json",
        supervisor_root / "launch_ack.json",
    ]
    missing = [str(path) for path in required_start if not path.exists()]
    if missing:
        raise RuntimeError(f"missing persistent-launch start evidence: {missing}")
    child_started = _read_json(supervisor_root / "child_started.json")
    ack = _read_json(supervisor_root / "launch_ack.json")
    stdout_path = supervisor_root / "child_stdout.log"
    stderr_path = supervisor_root / "child_stderr.log"
    if terminal_path.exists():
        exit_path = supervisor_root / "exit_sidecar.json"
        if not exit_path.exists():
            raise RuntimeError("terminal receipt exists without exit sidecar")
        terminal = _read_json(terminal_path)
        exit_sidecar = _read_json(exit_path)
        result = {
            "status": "terminal",
            "terminal_status": terminal["status"],
            "classification": terminal["classification"],
            "run_id": terminal["run_id"],
            "interpreter": ack["interpreter"],
            "child_pid": child_started["child_pid"],
            "entered_runner_at": ack["entered_runner_at"],
            "return_code": exit_sidecar["return_code"],
            "stdout_bytes": stdout_path.stat().st_size,
            "stderr_bytes": stderr_path.stat().st_size,
            "terminal_receipt": str(terminal_path),
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if terminal["status"] == "PASS" else 1
    if not allow_running:
        raise RuntimeError("terminal receipt is not available")
    child_pid = int(child_started["child_pid"])
    if not _process_alive(child_pid):
        raise RuntimeError("child has exited without terminal receipt")
    print(
        json.dumps(
            {
                "status": "running",
                "run_id": ack["run_id"],
                "interpreter": ack["interpreter"],
                "child_pid": child_pid,
                "entered_runner_at": ack["entered_runner_at"],
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "terminal_receipt": str(terminal_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _probe_child(entry_marker: Path, exit_code: int) -> int:
    _write_json(
        entry_marker,
        {
            "status": "entered",
            "entered_at": _now(),
            "pid": os.getpid(),
            "exit_code": exit_code,
        },
    )
    print(f"probe stdout pid={os.getpid()} exit={exit_code}", flush=True)
    print(f"probe stderr pid={os.getpid()} exit={exit_code}", file=sys.stderr, flush=True)
    time.sleep(0.25)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "launch-scientific",
            "launch-host-acceptance",
            "supervise",
            "verify",
            "probe-child",
        ],
    )
    parser.add_argument("--interpreter", default=DEFAULT_INTERPRETER)
    parser.add_argument("--expected-commit")
    parser.add_argument("--authorization-config")
    parser.add_argument("--output-root")
    parser.add_argument("--supervisor-root")
    parser.add_argument("--probe-case", choices=["success", "nonzero"])
    parser.add_argument("--plan")
    parser.add_argument("--entry-marker")
    parser.add_argument("--probe-exit-code", type=int, default=0)
    parser.add_argument("--wait-seconds", type=float, default=0.0)
    parser.add_argument("--allow-running", action="store_true")
    args = parser.parse_args()

    if args.mode == "supervise":
        return _supervise(Path(args.plan))
    if args.mode == "probe-child":
        return _probe_child(Path(args.entry_marker), args.probe_exit_code)
    if args.mode == "verify":
        return _verify(
            Path(args.supervisor_root).resolve(),
            args.wait_seconds,
            args.allow_running,
        )
    if not args.expected_commit or not args.supervisor_root:
        raise SystemExit("launch modes require --expected-commit and --supervisor-root")
    if args.mode == "launch-scientific":
        if not args.authorization_config or not args.output_root:
            raise SystemExit(
                "launch-scientific requires --authorization-config and --output-root"
            )
        plan = _build_scientific_plan(args)
    else:
        if not args.probe_case:
            raise SystemExit("launch-host-acceptance requires --probe-case")
        plan = _build_probe_plan(args)
    ack = _dispatch(plan)
    print(json.dumps(ack, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
