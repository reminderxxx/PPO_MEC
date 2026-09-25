#!/usr/bin/env python3
"""Launch, supervise, and inspect one frozen G14R22 background command.

This is deliberately not a scheduler.  A job root is create-only, execution is
single-shot, and there is no retry, resume, lock clearing, or second-run path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


HOST_VERSION = "g14r22d_fixed_background_host_v3"
TERMINAL_STATES = {"SUCCEEDED", "FAILED", "INTERRUPTED_OR_UNKNOWN"}
EXECUTION_MODES = {
    "acceptance_non_holdout",
    "formal_holdout",
    "formal_holdout_fixture",
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_identity(command: list[str]) -> str:
    return canonical_sha256(command)


def _ps_value(pid: int, field: str) -> str | None:
    try:
        completed = subprocess.run(
            ["ps", "-o", f"{field}=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value:
        return None
    return value


def _path_identity(value: str) -> dict[str, Any]:
    path = Path(value)
    return {
        "reported_path": value,
        "absolute_path": str(Path(os.path.abspath(value))) if path.is_absolute() else None,
        "resolved_path": str(path.resolve(strict=False)) if path.is_absolute() else None,
    }


def _command_evidence(command: list[str], actual_executable: Mapping[str, Any]) -> dict[str, Any]:
    launch = Path(os.path.abspath(command[0]))
    permitted = {
        str(launch),
        str(launch.resolve(strict=False)),
    }
    for field in ("reported_path", "absolute_path", "resolved_path"):
        value = actual_executable.get(field)
        if isinstance(value, str) and value:
            permitted.add(value)
    return {
        "launch_executable_path": str(launch),
        "resolved_launch_executable_path": str(launch.resolve(strict=False)),
        "actual_process_executable": dict(actual_executable),
        "permitted_live_argv0_paths": sorted(permitted),
        "argv_tail_sha256": canonical_sha256(command[1:]),
        "launch_command_sha256": command_identity(command),
    }


def process_identity(
    pid: int, *, launch_command: list[str] | None = None
) -> dict[str, Any] | None:
    if pid <= 0:
        return None
    started = _ps_value(pid, "lstart")
    live_command = _ps_value(pid, "command")
    executable = _ps_value(pid, "comm")
    if started is None or live_command is None or executable is None:
        return None
    try:
        live_argv = shlex.split(live_command)
    except ValueError:
        live_argv = []
    if not live_argv:
        return None
    actual_executable = _path_identity(executable)
    identity = {
        "pid": pid,
        "process_start_time": started,
        "process_start_token": started,
        "actual_process_executable": actual_executable,
        "live_command_sha256": hashlib.sha256(live_command.encode("utf-8")).hexdigest(),
        "live_command": live_command,
        "live_argv0": live_argv[0],
        "live_argv_tail_sha256": canonical_sha256(live_argv[1:]),
    }
    if launch_command is not None:
        identity["command_identity"] = _command_evidence(
            launch_command, actual_executable
        )
    return identity


def same_process(expected: Mapping[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    observed = process_identity(int(expected.get("pid", -1)))
    if observed is None:
        return False, None
    command = expected.get("command_identity")
    if not isinstance(command, Mapping):
        return False, observed
    expected_actual = command.get("actual_process_executable")
    observed_actual = observed.get("actual_process_executable")
    if not isinstance(expected_actual, Mapping) or not isinstance(observed_actual, Mapping):
        return False, observed
    actual_matches = all(
        observed_actual.get(field) == expected_actual.get(field)
        for field in ("reported_path", "resolved_path")
    )
    live_argv0 = str(observed.get("live_argv0") or "")
    live_argv0_paths = {live_argv0}
    if Path(live_argv0).is_absolute():
        live_argv0_paths.add(str(Path(live_argv0).resolve(strict=False)))
    permitted = set(command.get("permitted_live_argv0_paths") or [])
    return (
        observed.get("process_start_token") == expected.get("process_start_token")
        and actual_matches
        and bool(live_argv0_paths & permitted)
        and observed.get("live_argv_tail_sha256") == command.get("argv_tail_sha256"),
        observed,
    )


def git_value(checkout: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=checkout, text=True
    ).strip()


def validate_frozen_file(binding: Mapping[str, Any], label: str) -> Path:
    path = Path(str(binding.get("path", ""))).resolve()
    expected = str(binding.get("sha256", ""))
    if len(expected) != 64 or not path.is_file() or path.is_symlink():
        raise ValueError(f"invalid frozen {label} binding")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"frozen {label} hash mismatch: {observed} != {expected}")
    if "size_bytes" in binding and path.stat().st_size != binding.get("size_bytes"):
        raise ValueError(f"frozen {label} size mismatch")
    return path


def validate_job_package(
    path: Path, *, require_clean_checkout: bool,
    authorization_boundary: str = "background_launch",
) -> dict[str, Any]:
    package = read_json(path)
    declared = package.get("job_package_sha256")
    projection = {key: value for key, value in package.items() if key != "job_package_sha256"}
    if declared != canonical_sha256(projection):
        raise ValueError("background job package canonical hash mismatch")
    if package.get("host_version") != HOST_VERSION:
        raise ValueError("background host version mismatch")
    if package.get("automatic_retry_count") != 0:
        raise ValueError("background host forbids retry")
    mode = package.get("execution_mode")
    if mode not in EXECUTION_MODES:
        raise ValueError("background execution mode is missing or unsupported")
    if package.get("holdout_opened") is not False:
        raise ValueError("background package must be derived before any holdout opening")
    command = package.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) and item for item in command
    ):
        raise ValueError("background command must be a non-empty string list")
    checkout = Path(str(package.get("executor_checkout", ""))).resolve()
    cwd = Path(str(package.get("cwd", ""))).resolve()
    python_text = package.get("python_executable")
    resolved_python_text = package.get("python_resolved_executable")
    if not isinstance(python_text, str) or not Path(python_text).is_absolute():
        raise ValueError("frozen interpreter path must be an absolute string")
    python = Path(os.path.abspath(python_text))
    if (
        not isinstance(resolved_python_text, str)
        or not Path(resolved_python_text).is_absolute()
        or python.resolve() != Path(resolved_python_text)
    ):
        raise ValueError("frozen interpreter resolved identity mismatch")
    if cwd != checkout or not checkout.is_dir() or not python.is_file():
        raise ValueError("executor checkout/cwd/python binding is invalid")
    if Path(os.path.abspath(command[0])) != python:
        raise ValueError("background command must use the frozen interpreter")
    if git_value(checkout, "rev-parse", "HEAD") != package.get("executor_commit"):
        raise ValueError("executor commit mismatch")
    if git_value(checkout, "rev-parse", "HEAD^{tree}") != package.get("executor_git_tree"):
        raise ValueError("executor tree mismatch")
    if require_clean_checkout and git_value(checkout, "status", "--porcelain"):
        raise ValueError("executor checkout must be clean at launch")
    request_path = validate_frozen_file(package.get("request", {}), "request")
    scientific_path = validate_frozen_file(
        package.get("scientific_command_package", {}), "scientific command package"
    )
    request = read_json(request_path)
    scientific = read_json(scientific_path)
    if mode == "acceptance_non_holdout":
        if scientific.get("acceptance_non_holdout") is not True:
            raise ValueError("non-holdout host requires acceptance_non_holdout=true")
        if any(field in package for field in ("grant", "one_time_token")):
            raise ValueError("non-holdout host must not carry formal authorization references")
    else:
        isolated_fixture = mode == "formal_holdout_fixture"
        if bool(package.get("isolated_fixture_non_scientific")) != isolated_fixture:
            raise ValueError("formal host fixture identity mismatch")
        if scientific.get("acceptance_non_holdout") is True:
            raise ValueError("formal holdout host rejects non-holdout scientific identity")
        grant_path = validate_frozen_file(package.get("grant", {}), "grant")
        token_path = validate_frozen_file(package.get("one_time_token", {}), "one-time token")
        if set(package.get("one_time_token", {})) - {
            "path", "size_bytes", "sha256"
        }:
            raise ValueError("one-time token reference contains forbidden serialized fields")
        from src.evaluators.dedicated_holdout_execution import (
            validate_command_package,
            validate_unsigned_request,
            verify_grant,
        )

        validate_command_package(scientific, isolated_fixture=isolated_fixture)
        validate_unsigned_request(
            request, scientific, isolated_fixture=isolated_fixture
        )
        verify_grant(
            request,
            scientific,
            read_json(grant_path),
            token_path.read_bytes(),
            boundary=authorization_boundary,
        )
        expected_command = [
            str(python),
            str(checkout / "scripts/run_dedicated_public_holdout.py"),
            "--request-path", str(request_path),
            "--command-package-path", str(scientific_path),
            "--grant-path", str(grant_path),
            "--one-time-token-file", str(token_path),
            "--check", "fixture-execute" if isolated_fixture else "execute",
        ]
        if command != expected_command:
            raise ValueError("formal background command is not the fixed dedicated runner")
        if package.get("formal_output_root") != scientific.get("output_root"):
            raise ValueError("formal background output-root binding mismatch")
    environment = package.get("environment")
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ValueError("environment must be an exact string mapping")
    forbidden = [key for key in environment if "OPENAI" in key.upper() or "ANTHROPIC" in key.upper()]
    if forbidden:
        raise ValueError(f"AI/API environment variables are forbidden: {forbidden}")
    return package


def validate_integrity(manifest_path: Path, root: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("status") != "pass" or manifest.get("consumed_permanently") is not True:
        raise ValueError("integrity manifest is not terminal pass/permanently consumed")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("integrity manifest has no files")
    if manifest.get("file_count") != len(files):
        raise ValueError("integrity manifest file_count mismatch")
    if manifest.get("files_canonical_sha256") != canonical_sha256(files):
        raise ValueError("integrity manifest canonical inventory hash mismatch")
    observed_inventory: list[dict[str, Any]] = []
    for target in sorted(root.rglob("*")):
        if target.is_symlink():
            raise ValueError(f"integrity root contains symlink: {target}")
        if target.is_file():
            observed_inventory.append(
                {
                    "path": target.relative_to(root).as_posix(),
                    "size_bytes": target.stat().st_size,
                    "sha256": file_sha256(target),
                }
            )
    seen: set[str] = set()
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "size_bytes", "sha256"}:
            raise ValueError("integrity manifest row schema mismatch")
        relative = row.get("path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("integrity path must be non-empty and relative")
        normalized = Path(relative)
        if any(part in {"", ".", ".."} for part in normalized.parts) or relative in seen:
            raise ValueError("integrity path is duplicate or unsafe")
        seen.add(relative)
        target = (root / normalized).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("integrity path escapes root") from exc
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"integrity file missing: {relative}")
        if target.stat().st_size != int(row.get("size_bytes", -1)):
            raise ValueError(f"integrity size mismatch: {relative}")
        if file_sha256(target) != row.get("sha256"):
            raise ValueError(f"integrity hash mismatch: {relative}")
    if files != observed_inventory:
        raise ValueError("integrity manifest is not an exact published-root inventory")
    return {
        "status": "passed",
        "checked_file_count": len(files),
        "files_canonical_sha256": canonical_sha256(observed_inventory),
        "exact_inventory_match": True,
    }


def scientific_completion(package: Mapping[str, Any], child_return_code: int) -> dict[str, Any]:
    contract = package.get("terminal_contract")
    if not isinstance(contract, dict):
        raise ValueError("terminal_contract missing")
    receipt_path = Path(str(contract.get("scientific_receipt_path", ""))).resolve()
    receipt = read_json(receipt_path)
    if child_return_code != 0:
        raise ValueError(f"child return code is nonzero: {child_return_code}")
    mode = package.get("execution_mode")
    expected_bindings = contract.get("expected_receipt_bindings")
    if not isinstance(expected_bindings, dict) or any(
        receipt.get(field) != expected
        for field, expected in expected_bindings.items()
    ):
        raise ValueError("scientific terminal receipt does not match frozen identities")
    if mode != "acceptance_non_holdout" and (
        receipt.get("status") != "completed_permanently_consumed"
        or receipt.get("acceptance_non_holdout") is not False
        or receipt.get("holdout_opened") is not True
        or receipt.get("consumed_permanently") is not True
        or receipt.get("failure") is not None
    ):
        raise ValueError("formal holdout execution receipt is not a legal completion")
    frozen_scientific_path = validate_frozen_file(
        package.get("scientific_command_package", {}), "scientific command package"
    )
    frozen_scientific = read_json(frozen_scientific_path)
    manifest_path = Path(str(contract.get("integrity_manifest_path", ""))).resolve()
    integrity_root = Path(str(contract.get("integrity_root", ""))).resolve()
    manifest = read_json(manifest_path)
    execution_receipt_path = manifest_path.parent / "execution_receipt.json"
    execution_receipt = read_json(execution_receipt_path)
    if mode == "acceptance_non_holdout":
        if receipt.get("passed") is not True or int(receipt.get("return_code", -1)) != 0:
            raise ValueError(
                "non-holdout receipt does not declare passed=true/return_code=0"
            )
        if receipt.get("holdout_opened") is not False:
            raise ValueError("non-holdout receipt violates holdout_opened=false")
        if receipt.get("capacity_identity_validation", {}).get("status") != "passed":
            raise ValueError("non-holdout receipt lacks capacity identity closure")
        if (
            manifest.get("request_sha256")
            != frozen_scientific.get("acceptance_request_sha256")
            or manifest.get("command_package_sha256")
            != frozen_scientific.get("command_package_sha256")
        ):
            raise ValueError("published integrity identity differs from frozen scientific package")
        if (
            execution_receipt.get("status") != "completed_permanently_consumed"
            or execution_receipt.get("acceptance_non_holdout") is not True
            or execution_receipt.get("holdout_opened") is not False
        ):
            raise ValueError("non-holdout dedicated execution receipt identity/status mismatch")
        opening_receipt_path = None
        opening_receipt_sha256 = None
    else:
        if (
            receipt.get("status") != "completed_permanently_consumed"
            or receipt.get("acceptance_non_holdout") is not False
            or receipt.get("holdout_opened") is not True
            or receipt.get("consumed_permanently") is not True
            or receipt.get("failure") is not None
        ):
            raise ValueError("formal holdout execution receipt is not a legal completion")
        if manifest.get("status") != "pass" or manifest.get(
            "consumed_permanently"
        ) is not True:
            raise ValueError("formal holdout integrity manifest is not a legal completion")
        if (
            manifest.get("request_sha256") != receipt.get("request_sha256")
            or manifest.get("command_package_sha256")
            != receipt.get("command_package_sha256")
        ):
            raise ValueError("formal holdout receipt/integrity identity mismatch")
        opening_receipt_path = Path(
            str(contract.get("opening_receipt_path", ""))
        ).resolve()
        opening = read_json(opening_receipt_path)
        if (
            opening.get("event") != "opened"
            or opening.get("consumed_permanently") is not True
            or opening.get("request_sha256") != receipt.get("request_sha256")
            or opening.get("command_package_sha256")
            != receipt.get("command_package_sha256")
            or opening.get("grant_validation_at_atomic_open", {}).get("status")
            != "pass"
        ):
            raise ValueError("formal holdout opening receipt is invalid")
        opening_receipt_sha256 = file_sha256(opening_receipt_path)
    if (
        execution_receipt.get("request_sha256") != manifest.get("request_sha256")
        or execution_receipt.get("command_package_sha256")
        != manifest.get("command_package_sha256")
    ):
        raise ValueError("dedicated execution receipt identity/status mismatch")
    integrity = validate_integrity(manifest_path, integrity_root)
    return {
        "execution_mode": mode,
        "scientific_receipt_path": str(receipt_path),
        "scientific_receipt_sha256": file_sha256(receipt_path),
        "opening_receipt_path": (
            str(opening_receipt_path) if opening_receipt_path is not None else None
        ),
        "opening_receipt_sha256": opening_receipt_sha256,
        "integrity_manifest_path": str(manifest_path),
        "integrity_manifest_sha256": file_sha256(manifest_path),
        "dedicated_execution_receipt_path": str(execution_receipt_path),
        "dedicated_execution_receipt_sha256": file_sha256(execution_receipt_path),
        "integrity": integrity,
    }


def supervise(package_path: Path, job_root: Path) -> int:
    package = validate_job_package(
        package_path,
        require_clean_checkout=False,
        authorization_boundary="background_supervisor_prechild",
    )
    if Path(str(package.get("job_root", ""))).resolve() != job_root.resolve():
        raise ValueError("job root differs from frozen package binding")
    state_path = job_root / "state.json"
    terminal_path = job_root / "terminal_receipt.json"
    if terminal_path.exists():
        raise ValueError("terminal receipt already exists; second execution is forbidden")
    claim_path = job_root / "supervisor_claim.json"
    try:
        with claim_path.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "host_version": HOST_VERSION,
                    "job_id": package["job_id"],
                    "job_package_sha256": package["job_package_sha256"],
                    "supervisor_pid": os.getpid(),
                    "claimed_at": utc_now(),
                    "single_shot": True,
                    "never_clear_or_reuse": True,
                },
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValueError("supervisor claim already exists; second execution is forbidden") from exc
    stdout_path = job_root / "outer.stdout.log"
    stderr_path = job_root / "outer.stderr.log"
    interrupted: dict[str, Any] = {"signal": None}
    child: subprocess.Popen[bytes] | None = None

    def handle_signal(signum: int, _frame: Any) -> None:
        interrupted["signal"] = signum
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, handle_signal)
    started_at = utc_now()
    with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open("ab", buffering=0) as stderr:
        child = subprocess.Popen(
            package["command"],
            cwd=package["cwd"],
            env=dict(package["environment"]),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        child_identity = None
        for _ in range(50):
            child_identity = process_identity(child.pid, launch_command=package["command"])
            if child_identity is not None:
                break
            time.sleep(0.02)
        running = None
        for _ in range(250):
            if state_path.is_file():
                running = read_json(state_path)
                break
            time.sleep(0.02)
        if running is None:
            raise RuntimeError("launcher state was not published")
        running.update(
            child_process_identity=child_identity,
            child_command_sha256=command_identity(package["command"]),
            child_started_at=started_at,
        )
        atomic_json(state_path, running)
        child_return_code = child.wait()
    status = "FAILED"
    completion: dict[str, Any] | None = None
    scientific_failure_receipt: dict[str, Any] | None = None
    failure: str | None = None
    failure_category: str | None = None
    if interrupted["signal"] is not None:
        status = "INTERRUPTED_OR_UNKNOWN"
        failure_category = "SUPERVISOR_SIGNAL"
        failure = f"supervisor captured signal {interrupted['signal']}"
    elif child_return_code != 0:
        receipt_text = package.get("terminal_contract", {}).get(
            "scientific_receipt_path"
        )
        receipt_path = Path(receipt_text).resolve() if isinstance(receipt_text, str) else None
        if (
            package.get("execution_mode") != "acceptance_non_holdout"
            and receipt_path is not None
            and receipt_path.is_file()
        ):
            failed_receipt = read_json(receipt_path)
            scientific_failure_receipt = {
                "path": str(receipt_path),
                "sha256": file_sha256(receipt_path),
                "status": failed_receipt.get("status"),
                "partial_output_retained": failed_receipt.get(
                    "partial_output_retained"
                ),
                "consumed_permanently": failed_receipt.get(
                    "consumed_permanently"
                ),
            }
            failure_category = (
                "SCIENTIFIC_FAILED_PARTIAL_OUTPUT"
                if failed_receipt.get("partial_output_retained") is True
                else "SCIENTIFIC_FAILED"
            )
        else:
            failure_category = "CHILD_NONZERO_NO_SCIENTIFIC_RECEIPT"
        failure = f"child exited nonzero: {child_return_code}"
    else:
        try:
            completion = scientific_completion(package, child_return_code)
            status = "SUCCEEDED"
        except FileNotFoundError as exc:
            failure_category = "SCIENTIFIC_RECEIPT_MISSING"
            failure = f"terminal validation failed: {type(exc).__name__}: {exc}"
        except Exception as exc:  # terminal validation is intentionally fail-closed
            failure_category = "SCIENTIFIC_RECEIPT_OR_INTEGRITY_INVALID"
            failure = f"terminal validation failed: {type(exc).__name__}: {exc}"
    opening_path = package.get("terminal_contract", {}).get("opening_receipt_path")
    holdout_opened = bool(
        package.get("execution_mode") != "acceptance_non_holdout"
        and isinstance(opening_path, str)
        and Path(opening_path).is_file()
    )
    terminal = {
        "host_version": HOST_VERSION,
        "job_id": package["job_id"],
        "status": status,
        "started_at": started_at,
        "completed_at": utc_now(),
        "child_return_code": child_return_code,
        "captured_signal": interrupted["signal"],
        "failure_category": failure_category,
        "failure": failure,
        "scientific_completion": completion,
        "scientific_failure_receipt": scientific_failure_receipt,
        "outer_stdout_path": str(stdout_path),
        "outer_stderr_path": str(stderr_path),
        "request": package["request"],
        "scientific_command_package": package["scientific_command_package"],
        "executor_commit": package["executor_commit"],
        "executor_git_tree": package["executor_git_tree"],
        "job_package_sha256": package["job_package_sha256"],
        "execution_mode": package["execution_mode"],
        "automatic_retry_count": 0,
        "retry_allowed": False,
        "resume_allowed": False,
        "reopen_allowed": False,
        "automatic_lock_cleanup_allowed": False,
        "holdout_opened": holdout_opened,
    }
    atomic_json(terminal_path, terminal)
    running = read_json(state_path)
    running.update(status=status, terminal_receipt_path=str(terminal_path))
    atomic_json(state_path, running)
    return 0 if status == "SUCCEEDED" else 1


def launch(package_path: Path, job_root: Path) -> dict[str, Any]:
    package = validate_job_package(
        package_path,
        require_clean_checkout=True,
        authorization_boundary="background_launch",
    )
    if Path(str(package.get("job_root", ""))).resolve() != job_root.resolve():
        raise ValueError("job root differs from frozen package binding")
    if job_root.exists():
        raise ValueError("job root already exists; retry/resume/second run are forbidden")
    job_root.mkdir(parents=True, exist_ok=False)
    snapshot_path = job_root / "job_package_snapshot.json"
    snapshot_path.write_bytes(package_path.read_bytes())
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "supervise",
        "--job-package",
        str(snapshot_path),
        "--job-root",
        str(job_root),
    ]
    supervisor = subprocess.Popen(
        command,
        cwd=package["cwd"],
        env=dict(package["environment"]),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    identity = None
    for _ in range(100):
        identity = process_identity(supervisor.pid, launch_command=command)
        if identity is not None:
            break
        time.sleep(0.02)
    if identity is None:
        raise RuntimeError("unable to capture supervisor process identity")
    state = {
        "host_version": HOST_VERSION,
        "job_id": package["job_id"],
        "status": "RUNNING",
        "execution_mode": package["execution_mode"],
        "launched_at": utc_now(),
        "launcher_pid": os.getpid(),
        "supervisor_process_identity": identity,
        "supervisor_command_sha256": command_identity(command),
        "child_process_identity": None,
        "job_package_path": str(snapshot_path),
        "job_package_sha256": package["job_package_sha256"],
        "request": package["request"],
        "scientific_command_package": package["scientific_command_package"],
        "executor_checkout": package["executor_checkout"],
        "executor_commit": package["executor_commit"],
        "executor_git_tree": package["executor_git_tree"],
        "python_executable": package["python_executable"],
        "python_resolved_executable": package["python_resolved_executable"],
        "cwd": package["cwd"],
        "automatic_retry_count": 0,
        "retry_allowed": False,
        "resume_allowed": False,
        "reopen_allowed": False,
        "automatic_lock_cleanup_allowed": False,
        "holdout_opened": False,
    }
    state_path = job_root / "state.json"
    atomic_json(state_path, state)
    # Do not return a misleading RUNNING acknowledgement until the supervisor
    # has published the child birth identity (or a terminal receipt appeared).
    for _ in range(250):
        current = read_json(state_path)
        if current.get("child_process_identity") is not None or (
            job_root / "terminal_receipt.json"
        ).is_file():
            return current
        time.sleep(0.02)
    raise RuntimeError("supervisor did not publish child process identity")


def inspect(job_root: Path) -> dict[str, Any]:
    state_path = job_root / "state.json"
    terminal_path = job_root / "terminal_receipt.json"
    state = read_json(state_path)
    if terminal_path.is_file():
        terminal = read_json(terminal_path)
        status = terminal.get("status")
        if status not in TERMINAL_STATES:
            raise ValueError("invalid terminal state")
        return {
            "observed_status": status,
            "source": "terminal_receipt",
            "state": state,
            "terminal_receipt": terminal,
            "read_only": True,
        }
    matches, observed = same_process(state.get("supervisor_process_identity", {}))
    return {
        "observed_status": "RUNNING" if matches else "INTERRUPTED_OR_UNKNOWN",
        "source": "live_process_identity" if matches else "missing_or_reused_process_identity",
        "state": state,
        "observed_supervisor_process_identity": observed,
        "terminal_receipt": None,
        "read_only": True,
        "note": (
            "SIGKILL, power loss, or host crash cannot guarantee an immediate terminal receipt"
            if not matches else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("launch", "supervise"):
        child = subparsers.add_parser(mode)
        child.add_argument("--job-package", type=Path, required=True)
        child.add_argument("--job-root", type=Path, required=True)
    child = subparsers.add_parser("inspect")
    child.add_argument("--job-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "launch":
        result = launch(args.job_package.resolve(), args.job_root.resolve())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "supervise":
        return supervise(args.job_package.resolve(), args.job_root.resolve())
    print(json.dumps(inspect(args.job_root.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
