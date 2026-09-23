from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.run_g14r22b_background_job import (
    HOST_VERSION,
    canonical_sha256,
    file_sha256,
    inspect,
    launch,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frozen_executor(tmp_path: Path) -> dict[str, object]:
    checkout = tmp_path / "executor"
    checkout.mkdir()
    helper = checkout / "child.py"
    helper.write_text(
        """from __future__ import annotations
import hashlib, json, signal, sys, time
from pathlib import Path
mode, root_text, commit, tree, request_file_hash, package_file_hash, inner_request, inner_package = sys.argv[1:9]
root = Path(root_text); root.mkdir(parents=True, exist_ok=True)
if mode in {'sleep_success', 'sleep'}:
    time.sleep(30 if mode == 'sleep' else 0.5)
if mode == 'fail':
    raise SystemExit(9)
if mode in {'success', 'sleep_success', 'success_extra', 'success_status_fail', 'success_bad_binding'}:
    published = root / 'published'; published.mkdir()
    payload = published / 'payload.txt'; payload.write_text('complete\\n')
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    files = [{'path':'payload.txt','size_bytes':payload.stat().st_size,'sha256':digest}]
    canonical = hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    integrity = {'status':'fail' if mode == 'success_status_fail' else 'pass','consumed_permanently':True,'file_count':len(files),'files':files,'files_canonical_sha256':canonical,'request_sha256':inner_request,'command_package_sha256':inner_package}
    (root / 'integrity.json').write_text(json.dumps(integrity)+'\\n')
    execution = {'status':'completed_permanently_consumed','acceptance_non_holdout':True,'holdout_opened':False,'request_sha256':inner_request,'command_package_sha256':inner_package}
    (root / 'execution_receipt.json').write_text(json.dumps(execution)+'\\n')
    receipt = {'passed':True,'return_code':0,'holdout_opened':False,'executor_commit':'bad' if mode == 'success_bad_binding' else commit,'executor_git_tree':tree,'background_request_sha256':request_file_hash,'command_package_sha256':package_file_hash,'capacity_identity_validation':{'status':'passed'}}
    (root / 'receipt.json').write_text(json.dumps(receipt)+'\\n')
    if mode == 'success_extra':
        (published / 'unlisted.txt').write_text('extra\\n')
""",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=checkout, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "child.py"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=checkout, check=True, capture_output=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=checkout, text=True).strip()
    request = tmp_path / "request.json"
    scientific = tmp_path / "scientific_package.json"
    request.write_text('{"holdout_opened":false}\n', encoding="utf-8")
    scientific.write_text(
        json.dumps(
            {
                "automatic_retry_count": 0,
                "acceptance_request_sha256": "r" * 64,
                "command_package_sha256": "p" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "checkout": checkout,
        "helper": helper,
        "commit": commit,
        "tree": tree,
        "request": request,
        "scientific": scientific,
        "tmp": tmp_path,
    }


def build_package(fixture: dict[str, object], mode: str, job_name: str) -> tuple[Path, Path, Path]:
    tmp = fixture["tmp"]
    assert isinstance(tmp, Path)
    output = tmp / f"{job_name}_scientific"
    job_root = tmp / f"{job_name}_job"
    request = fixture["request"]
    scientific = fixture["scientific"]
    checkout = fixture["checkout"]
    helper = fixture["helper"]
    assert all(isinstance(path, Path) for path in (request, scientific, checkout, helper))
    package = {
        "host_version": HOST_VERSION,
        "job_id": job_name,
        "executor_checkout": str(checkout),
        "executor_commit": fixture["commit"],
        "executor_git_tree": fixture["tree"],
        "cwd": str(checkout),
        "python_executable": sys.executable,
        "environment": {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
        },
        "command": [
            sys.executable,
            str(helper),
            mode,
            str(output),
            str(fixture["commit"]),
            str(fixture["tree"]),
            file_sha256(request),
            file_sha256(scientific),
            "r" * 64,
            "p" * 64,
        ],
        "request": {"path": str(request), "sha256": file_sha256(request)},
        "scientific_command_package": {
            "path": str(scientific),
            "sha256": file_sha256(scientific),
        },
        "terminal_contract": {
            "scientific_receipt_path": str(output / "receipt.json"),
            "integrity_manifest_path": str(output / "integrity.json"),
            "integrity_root": str(output / "published"),
            "expected_receipt_bindings": {
                "executor_commit": fixture["commit"],
                "executor_git_tree": fixture["tree"],
                "command_package_sha256": file_sha256(scientific),
                "background_request_sha256": file_sha256(request),
            },
        },
        "job_root": str(job_root),
        "automatic_retry_count": 0,
        "holdout_opened": False,
    }
    package["job_package_sha256"] = canonical_sha256(package)
    package_path = tmp / f"{job_name}_package.json"
    package_path.write_text(json.dumps(package) + "\n", encoding="utf-8")
    return package_path, job_root, output


def wait_terminal(job_root: Path, timeout: float = 8.0) -> dict[str, object]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = inspect(job_root)
        if result["terminal_receipt"] is not None:
            return result
        time.sleep(0.05)
    raise AssertionError("background job did not become terminal")


def wait_child_identity(job_root: Path) -> dict[str, object]:
    deadline = time.time() + 5
    while time.time() < deadline:
        state = json.loads((job_root / "state.json").read_text())
        if state.get("child_process_identity"):
            return state
        time.sleep(0.02)
    raise AssertionError("child identity was not published")


def test_launcher_exit_does_not_stop_successful_job(frozen_executor: dict[str, object]) -> None:
    package, job_root, _ = build_package(frozen_executor, "sleep_success", "survives_launcher")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_g14r22b_background_job.py"), "launch", "--job-package", str(package), "--job-root", str(job_root)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = wait_terminal(job_root)
    assert result["observed_status"] == "SUCCEEDED"
    receipt = result["terminal_receipt"]
    assert receipt["scientific_completion"]["integrity"]["status"] == "passed"


def test_child_nonzero_is_failed(frozen_executor: dict[str, object]) -> None:
    package, job_root, _ = build_package(frozen_executor, "fail", "child_nonzero")
    launch(package, job_root)
    result = wait_terminal(job_root)
    assert result["observed_status"] == "FAILED"
    assert result["terminal_receipt"]["child_return_code"] == 9


@pytest.mark.parametrize(
    "mode", ["success_extra", "success_status_fail", "success_bad_binding"]
)
def test_incomplete_or_failed_integrity_cannot_succeed(
    frozen_executor: dict[str, object], mode: str
) -> None:
    package, job_root, _ = build_package(frozen_executor, mode, mode)
    launch(package, job_root)
    result = wait_terminal(job_root)
    assert result["observed_status"] == "FAILED"
    assert "terminal validation failed" in result["terminal_receipt"]["failure"]


def test_catchable_signal_is_terminal_unknown(frozen_executor: dict[str, object]) -> None:
    package, job_root, _ = build_package(frozen_executor, "sleep", "catchable_signal")
    state = launch(package, job_root)
    wait_child_identity(job_root)
    supervisor_pid = state["supervisor_process_identity"]["pid"]
    os.kill(supervisor_pid, signal.SIGTERM)
    result = wait_terminal(job_root)
    assert result["observed_status"] == "INTERRUPTED_OR_UNKNOWN"
    assert result["terminal_receipt"]["captured_signal"] == signal.SIGTERM


def test_sigkill_has_no_instant_receipt_and_readonly_inspect_is_unknown(
    frozen_executor: dict[str, object],
) -> None:
    package, job_root, _ = build_package(frozen_executor, "sleep", "host_sigkill")
    state = launch(package, job_root)
    state = wait_child_identity(job_root)
    supervisor_pid = state["supervisor_process_identity"]["pid"]
    os.kill(supervisor_pid, signal.SIGKILL)
    time.sleep(0.1)
    result = inspect(job_root)
    assert result["observed_status"] == "INTERRUPTED_OR_UNKNOWN"
    assert result["terminal_receipt"] is None
    assert not (job_root / "terminal_receipt.json").exists()
    child_pid = state["child_process_identity"]["pid"]
    try:
        os.killpg(child_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_pid_reuse_guard_checks_start_token_and_command_hash(
    frozen_executor: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    package, job_root, _ = build_package(frozen_executor, "sleep", "pid_identity")
    state = launch(package, job_root)
    state = wait_child_identity(job_root)
    expected = state["supervisor_process_identity"]
    monkeypatch.setattr(
        "scripts.run_g14r22b_background_job.process_identity",
        lambda _pid: {
            "pid": expected["pid"],
            "process_start_token": "different birth",
            "live_command_sha256": expected["live_command_sha256"],
            "live_command": expected["live_command"],
        },
    )
    assert inspect(job_root)["observed_status"] == "INTERRUPTED_OR_UNKNOWN"
    os.kill(expected["pid"], signal.SIGTERM)
    wait_terminal(job_root)


def test_job_root_binding_and_single_shot_are_fail_closed(
    frozen_executor: dict[str, object], tmp_path: Path
) -> None:
    package, job_root, _ = build_package(frozen_executor, "sleep_success", "single_shot")
    with pytest.raises(ValueError, match="frozen package binding"):
        launch(package, tmp_path / "different_job_root")
    launch(package, job_root)
    with pytest.raises(ValueError, match="already exists"):
        launch(package, job_root)
    assert wait_terminal(job_root)["observed_status"] == "SUCCEEDED"


def test_direct_second_supervisor_cannot_start_another_child(
    frozen_executor: dict[str, object]
) -> None:
    package, job_root, _ = build_package(
        frozen_executor, "sleep_success", "direct_supervisor_single_shot"
    )
    launch(package, job_root)
    first_state = wait_child_identity(job_root)
    first_child_pid = first_state["child_process_identity"]["pid"]
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_g14r22b_background_job.py"),
            "supervise",
            "--job-package", str(package),
            "--job-root", str(job_root),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "supervisor claim already exists" in completed.stderr
    assert json.loads((job_root / "state.json").read_text())["child_process_identity"]["pid"] == first_child_pid
    assert wait_terminal(job_root)["observed_status"] == "SUCCEEDED"
