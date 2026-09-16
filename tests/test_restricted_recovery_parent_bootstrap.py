"""Real filesystem boundaries for the restricted-recovery parent bootstrap."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from scripts.continuation_executor.locking import writer_lock_path
from scripts.run_typed_model_cache_restricted_recovery import (
    RestrictedRecoveryParentBootstrap,
    _load,
)
from src.runtime.evaluation_only_execution import canonical_sha256
from src.runtime.restricted_recovery import (
    RestrictedRecoveryError,
    validate_restricted_recovery_parent_contract,
    validate_restricted_recovery_request,
)
from tests.test_restricted_recovery import minimal_request


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path("/Users/howen/Projects/PPO_MEC")
PYTHON = str(PROJECT / ".venv/bin/python")
DRIVER = ROOT / "tests/restricted_recovery_public_driver.py"


def _rehash(request: dict) -> None:
    execution = request["recovery_execution"]
    execution["recovery_execution_identity_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in execution.items()
            if key != "recovery_execution_identity_sha256"
        }
    )
    request["authorization_request_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in request.items()
            if key != "authorization_request_sha256"
        }
    )


def _inventory(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(item.relative_to(path).as_posix() for item in path.rglob("*"))


def test_missing_parent_cold_start_is_create_only_and_root_stays_absent(
    tmp_path: Path,
) -> None:
    request = minimal_request(tmp_path)
    root = Path(request["recovery_execution"]["recovery_root"])
    parent = root.parent
    assert not parent.exists()
    with RestrictedRecoveryParentBootstrap(
        request, expected_parent=parent
    ) as guard:
        assert guard.created is True
        guard.verify()
        assert not root.exists()
    assert parent.is_dir() and not parent.is_symlink()
    assert parent.stat().st_mode & 0o777 == 0o700
    assert _inventory(parent) == []


def test_two_independent_processes_concurrently_initialize_missing_parent(
    tmp_path: Path,
) -> None:
    request = minimal_request(tmp_path)
    root = Path(request["recovery_execution"]["recovery_root"])
    parent = root.parent
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    release = tmp_path / "release"
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    processes = []
    ready_paths = []
    for index in range(2):
        ready = tmp_path / f"ready_{index}"
        ready_paths.append(ready)
        processes.append(
            subprocess.Popen(
                [
                    PYTHON,
                    str(DRIVER),
                    "bootstrap",
                    str(tmp_path),
                    str(request_path),
                    str(ready),
                    str(release),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
    deadline = time.monotonic() + 10
    while not all(path.exists() for path in ready_paths):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert not parent.exists()
    release.touch()
    results = [process.communicate(timeout=15) for process in processes]
    assert [process.returncode for process in processes] == [0, 0], results
    payloads = [json.loads(stdout) for stdout, _ in results]
    assert sorted(item["created"] for item in payloads) == [False, True]
    assert parent.is_dir() and parent.stat().st_mode & 0o777 == 0o700
    assert not root.exists() and _inventory(parent) == []


def test_existing_legal_parent_is_revalidated_without_replacement(
    tmp_path: Path,
) -> None:
    request = minimal_request(tmp_path)
    parent = Path(request["recovery_execution"]["recovery_root"]).parent
    parent.mkdir(mode=0o700)
    before = parent.stat()
    with RestrictedRecoveryParentBootstrap(
        request, expected_parent=parent
    ) as guard:
        assert guard.created is False
        guard.verify()
    after = parent.stat()
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)


@pytest.mark.parametrize("kind", ["symlink", "file", "wrong_mode"])
def test_invalid_existing_parent_fails_without_recovery_write(
    tmp_path: Path, kind: str
) -> None:
    request = minimal_request(tmp_path)
    root = Path(request["recovery_execution"]["recovery_root"])
    parent = root.parent
    if kind == "symlink":
        target = tmp_path / "target"
        target.mkdir(mode=0o700)
        parent.symlink_to(target, target_is_directory=True)
    elif kind == "file":
        parent.write_text("not a directory", encoding="utf-8")
    else:
        parent.mkdir(mode=0o755)
    before = _inventory(tmp_path)
    with pytest.raises(RestrictedRecoveryError, match="owner/mode/type"):
        with RestrictedRecoveryParentBootstrap(request, expected_parent=parent):
            pass
    assert _inventory(tmp_path) == before
    assert not root.exists() and not (parent / ".continuation_locks").exists()


def test_unwritable_grandparent_fails_before_parent_creation(tmp_path: Path) -> None:
    request = minimal_request(tmp_path)
    root = Path(request["recovery_execution"]["recovery_root"])
    original_mode = tmp_path.stat().st_mode & 0o777
    tmp_path.chmod(0o500)
    try:
        with pytest.raises(RestrictedRecoveryError, match="identity/access drift"):
            validate_restricted_recovery_parent_contract(
                request["recovery_execution"],
                expected_parent=root.parent,
                check_live=True,
            )
        assert not root.parent.exists()
    finally:
        tmp_path.chmod(original_mode)


def test_parent_replacement_between_validation_and_use_fails_before_run_write(
    tmp_path: Path,
) -> None:
    request = minimal_request(tmp_path)
    root = Path(request["recovery_execution"]["recovery_root"])
    parent = root.parent

    def replace(_guard) -> None:
        parent.rename(tmp_path / "replaced_original_parent")
        parent.mkdir(mode=0o700)

    with pytest.raises(RestrictedRecoveryError, match="replaced or changed"):
        with RestrictedRecoveryParentBootstrap(
            request,
            expected_parent=parent,
            before_use_hook=replace,
        ):
            pass
    assert not root.exists() and not writer_lock_path(root).exists()
    assert not (parent / "cell_state.jsonl").exists()


def test_existing_root_without_initialization_marker_is_read_only_rejection(
    tmp_path: Path,
) -> None:
    request = minimal_request(tmp_path)
    root = Path(request["recovery_execution"]["recovery_root"])
    root.mkdir(parents=True, mode=0o700)
    before = _inventory(root.parent)
    with pytest.raises(RestrictedRecoveryError, match="authorization object is missing"):
        _load(request)
    assert _inventory(root.parent) == before
    assert not writer_lock_path(root).exists()


def test_fault_after_parent_creation_leaves_benign_repeatable_state(
    tmp_path: Path,
) -> None:
    request = minimal_request(tmp_path)
    root = Path(request["recovery_execution"]["recovery_root"])
    parent = root.parent

    def stop(_guard) -> None:
        raise RuntimeError("synthetic stop after parent bootstrap")

    with pytest.raises(RuntimeError, match="synthetic stop"):
        with RestrictedRecoveryParentBootstrap(
            request,
            expected_parent=parent,
            after_create_hook=stop,
        ):
            pass
    assert parent.is_dir() and _inventory(parent) == [] and not root.exists()
    with RestrictedRecoveryParentBootstrap(
        request, expected_parent=parent
    ) as guard:
        assert guard.created is False
        guard.verify()
    assert _inventory(parent) == []


def test_contract_rejects_path_escape_and_wrong_owner_identity(tmp_path: Path) -> None:
    request = minimal_request(tmp_path)
    parent = Path(request["recovery_execution"]["recovery_root"]).parent
    escaped = deepcopy(request)
    escaped["recovery_execution"]["recovery_root"] = str(
        tmp_path / "escaped" / "synthetic_recovery_run"
    )
    _rehash(escaped)
    with pytest.raises(RestrictedRecoveryError, match="parent contract drift"):
        validate_restricted_recovery_request(
            escaped,
            check_live=False,
            _test_recovery_parent=parent,
        )
    wrong_owner = deepcopy(request)
    wrong_owner["recovery_execution"]["parent_bootstrap_contract"][
        "required_parent_owner_uid"
    ] += 1
    _rehash(wrong_owner)
    with pytest.raises(RestrictedRecoveryError, match="parent contract drift"):
        validate_restricted_recovery_request(
            wrong_owner,
            check_live=False,
            _test_recovery_parent=parent,
        )
