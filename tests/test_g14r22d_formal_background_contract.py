from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.evaluators.dedicated_holdout_execution import (
    CAPACITIES,
    GRANT_VALIDITY_CONTRACT,
    HOLDOUT_COMMAND_PACKAGE_VERSION,
    HOLDOUT_GRANT_VERSION,
    HOLDOUT_REQUEST_VERSION,
    LEARNED_AGENTS,
    SEEDS,
    canonical_sha256,
    execute_package,
    file_sha256,
    read_json,
)
from scripts.run_g14r22b_background_job import scientific_completion


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=checkout, text=True).strip()


def make_executor(tmp_path: Path) -> tuple[Path, str, str]:
    frozen = os.environ.get("G14R22D_ACCEPTANCE_EXECUTOR")
    if frozen:
        checkout = Path(frozen).resolve()
        assert git(checkout, "status", "--porcelain") == ""
        return (
            checkout,
            git(checkout, "rev-parse", "HEAD"),
            git(checkout, "rev-parse", "HEAD^{tree}"),
        )
    checkout = tmp_path / "executor"
    for relative in (
        "scripts/run_dedicated_public_holdout.py",
        "scripts/run_g14r22b_background_job.py",
        "scripts/derive_g14r22_holdout_background_job.py",
        "src/evaluators/dedicated_holdout_execution.py",
    ):
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for relative in ("src/__init__.py", "src/evaluators/__init__.py"):
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=checkout, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"], cwd=checkout, check=True
    )
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-m", "fixture executor"], cwd=checkout, check=True, capture_output=True)
    return checkout, git(checkout, "rev-parse", "HEAD"), git(checkout, "rev-parse", "HEAD^{tree}")


def build_fixture(tmp_path: Path, *, fail_first_child: bool = False) -> dict[str, Path | str]:
    checkout, commit, tree = make_executor(tmp_path)
    python = Path(sys.executable).resolve()
    output = tmp_path / "formal_fixture_output"
    child_code = (
        "import csv,hashlib,json,sys;from pathlib import Path;"
        "root=Path(sys.argv[1]);capacity=sys.argv[2];artifact=root/'payload';"
        "artifact.mkdir(parents=True);rows=artifact/'benchmark_rows.csv';"
        "h=rows.open('w',newline='');w=csv.writer(h);"
        "w.writerow(['capacity_label','value']);w.writerow([capacity,1]);"
        "w.writerow([capacity,2]);h.close();data=rows.read_bytes();"
        "files=[{'path':'benchmark_rows.csv','size_bytes':len(data),"
        "'sha256':hashlib.sha256(data).hexdigest()}];"
        "(artifact/'artifact_integrity_manifest.json').write_text("
        "json.dumps({'files':files})+'\\n')"
    )
    scientific_commands = [
        [str(python), "-c", child_code, "{G14R22_CELL_OUTPUT_ROOT}", capacity]
        for capacity in CAPACITIES
    ]
    if fail_first_child:
        scientific_commands[0] = ["/usr/bin/false"]
    stats_code = (
        "import json,sys;from pathlib import Path;root=Path(sys.argv[1]);"
        "root.mkdir(parents=True,exist_ok=True);"
        "row={'holm_preregistered_family_size':1};"
        "(root/'paired_statistics.json').write_text(json.dumps({'rows':[row]})+'\\n');"
        "(root/'paired_statistics.csv').write_text("
        "'holm_preregistered_family_size\\n1\\n')"
    )
    package = {
        "command_package_version": HOLDOUT_COMMAND_PACKAGE_VERSION,
        "execution_mode": "formal_holdout_fixture",
        "isolated_fixture_non_scientific": True,
        "fixture_holdout_policy_runs": 0,
        "executor_checkout": str(checkout),
        "executor_commit": commit,
        "executor_git_tree": tree,
        "output_root": str(output),
        "phases": ["scientific", "statistics", "publication", "integrity"],
        "scientific_matrix": {"rows_per_child": 2, "holm_family_size": 1},
        "commands": {
            "scientific": scientific_commands,
            "statistics": [
                str(python),
                "-c",
                stats_code,
                "{G14R22_STATISTICS_OUTPUT_ROOT}",
                "{G14R22_ROWS_0}",
                "{G14R22_ROWS_1}",
                "{G14R22_ROWS_2}",
            ],
        },
        "automatic_retry_count": 0,
    }
    package["command_package_sha256"] = canonical_sha256(package)
    package_path = tmp_path / "scientific_package.json"
    write_json(package_path, package)

    models = []
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    for capacity in CAPACITIES:
        for agent in LEARNED_AGENTS:
            for seed in SEEDS:
                target = checkpoint_root / f"{capacity}-{agent}-{seed}.pt"
                target.write_bytes(f"{capacity}:{agent}:{seed}".encode())
                models.append(
                    {
                        "capacity_label": capacity,
                        "agent": agent,
                        "seed": seed,
                        "checkpoint_path": str(target),
                        "size_bytes": target.stat().st_size,
                        "checkpoint_sha256": file_sha256(target),
                    }
                )
    source = tmp_path / "checkpoint_source.json"
    write_json(source, {"models": models})
    frozen = tmp_path / "frozen.txt"
    frozen.write_text("frozen\n", encoding="utf-8")
    created = datetime.now(timezone.utc) - timedelta(minutes=2)
    request = {
        "request_version": HOLDOUT_REQUEST_VERSION,
        "created_at": created.isoformat(),
        "status": "READY_FOR_AUTHORIZATION_REVIEW",
        "grant_signed": False,
        "execution_authorized": False,
        "holdout_opened": False,
        "holdout_consumed_permanently": False,
        "command_package_sha256": package["command_package_sha256"],
        "executor_commit": commit,
        "output_root": str(output),
        "grant_validity_contract": GRANT_VALIDITY_CONTRACT,
        "checkpoint_source_reference": {"path": str(source)},
        "frozen_inputs": [
            {
                "path": str(frozen),
                "size_bytes": frozen.stat().st_size,
                "sha256": file_sha256(frozen),
            }
        ],
        "failure_boundary": {
            "before_atomic_open": "not_consumed; no scientific child may start",
            "at_or_after_atomic_open": "permanently_consumed",
            "child_failure": "terminal_failure_no_retry_no_resume_no_reopen",
            "partial_output": "retained_in_staging_and_permanently_consumed",
            "success": "permanently_consumed",
        },
    }
    request["request_sha256"] = canonical_sha256(request)
    request_path = tmp_path / "request.json"
    write_json(request_path, request)
    review = {
        "status": "pass",
        "request_sha256": request["request_sha256"],
        "command_package_sha256": package["command_package_sha256"],
    }
    review_path = tmp_path / "review.json"
    write_json(review_path, review)
    token_bytes = b"fixture-token-secret-never-log-7f0f9a"
    token_path = tmp_path / "token.secret"
    token_path.write_bytes(token_bytes)
    issued = datetime.now(timezone.utc) - timedelta(seconds=5)
    grant = {
        "grant_version": HOLDOUT_GRANT_VERSION,
        "status": "AUTHORIZED_ONE_TIME_HOLDOUT",
        "grant_signed": True,
        "request_sha256": request["request_sha256"],
        "command_package_sha256": package["command_package_sha256"],
        "executor_commit": commit,
        "output_root": str(output),
        "one_time_token_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "independent_review": {
            "path": str(review_path),
            "sha256": file_sha256(review_path),
            "size_bytes": review_path.stat().st_size,
        },
        "issued_at": issued.isoformat(),
        "expires_at": (issued + timedelta(hours=72)).isoformat(),
        "grant_validity_contract": GRANT_VALIDITY_CONTRACT,
    }
    grant_path = tmp_path / "grant.json"
    write_json(grant_path, grant)
    return {
        "checkout": checkout,
        "commit": commit,
        "tree": tree,
        "python": python,
        "output": output,
        "package": package_path,
        "request": request_path,
        "grant": grant_path,
        "token": token_path,
        "token_text": token_bytes.decode(),
        "derived": tmp_path / "derived",
        "job": tmp_path / "job",
    }


def test_public_derivation_launch_and_formal_terminal_are_closed(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    checkout = fixture["checkout"]
    assert isinstance(checkout, Path)
    command = [
        str(fixture["python"]),
        str(checkout / "scripts/derive_g14r22_holdout_background_job.py"),
        "--request-path", str(fixture["request"]),
        "--command-package-path", str(fixture["package"]),
        "--grant-path", str(fixture["grant"]),
        "--one-time-token-file", str(fixture["token"]),
        "--executor-checkout", str(checkout),
        "--python-executable", str(fixture["python"]),
        "--derived-root", str(fixture["derived"]),
        "--job-root", str(fixture["job"]),
        "--isolated-fixture",
    ]
    derived = subprocess.run(command, cwd=checkout, text=True, capture_output=True, check=False)
    assert derived.returncode == 0, derived.stderr
    package_path = Path(fixture["derived"]) / "background_job_package.json"
    launched = subprocess.run(
        [
            str(fixture["python"]),
            str(checkout / "scripts/run_g14r22b_background_job.py"),
            "launch",
            "--job-package", str(package_path),
            "--job-root", str(fixture["job"]),
        ],
        cwd=checkout,
        text=True,
        capture_output=True,
        check=False,
    )
    assert launched.returncode == 0, launched.stderr
    deadline = time.time() + 20
    terminal_path = Path(fixture["job"]) / "terminal_receipt.json"
    while time.time() < deadline and not terminal_path.is_file():
        time.sleep(0.05)
    assert terminal_path.is_file()
    terminal = json.loads(terminal_path.read_text())
    assert terminal["status"] == "SUCCEEDED"
    assert terminal["execution_mode"] == "formal_holdout_fixture"
    assert terminal["holdout_opened"] is True
    assert terminal["scientific_completion"]["integrity"]["status"] == "passed"
    receipt = json.loads((Path(fixture["output"]) / "execution_receipt.json").read_text())
    assert receipt["status"] == "completed_permanently_consumed"
    assert receipt["holdout_opened"] is True
    assert receipt["isolated_fixture_non_scientific"] is True
    assert receipt["fixture_holdout_policy_runs"] == 0
    source_request_before = Path(fixture["request"]).read_bytes()
    source_package_before = Path(fixture["package"]).read_bytes()
    derivation = json.loads(
        (Path(fixture["derived"]) / "derivation_receipt.json").read_text()
    )
    assert derivation["unsigned_sources_unchanged"] is True
    assert Path(fixture["request"]).read_bytes() == source_request_before
    assert Path(fixture["package"]).read_bytes() == source_package_before
    secret = str(fixture["token_text"])
    for root in (fixture["derived"], fixture["job"], fixture["output"]):
        assert isinstance(root, Path)
        for path in root.rglob("*"):
            if path.is_file():
                assert secret.encode() not in path.read_bytes(), path


def test_formal_and_non_holdout_identity_swap_is_rejected(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    checkout = fixture["checkout"]
    assert isinstance(checkout, Path)
    derived = subprocess.run(
        [
            str(fixture["python"]),
            str(checkout / "scripts/derive_g14r22_holdout_background_job.py"),
            "--request-path", str(fixture["request"]),
            "--command-package-path", str(fixture["package"]),
            "--grant-path", str(fixture["grant"]),
            "--one-time-token-file", str(fixture["token"]),
            "--executor-checkout", str(checkout),
            "--python-executable", str(fixture["python"]),
            "--derived-root", str(fixture["derived"]),
            "--job-root", str(fixture["job"]),
            "--isolated-fixture",
        ],
        cwd=checkout,
        text=True,
        capture_output=True,
        check=False,
    )
    assert derived.returncode == 0, derived.stderr
    package_path = Path(fixture["derived"]) / "background_job_package.json"
    package = json.loads(package_path.read_text())
    package["execution_mode"] = "acceptance_non_holdout"
    package["job_package_sha256"] = canonical_sha256(
        {key: value for key, value in package.items() if key != "job_package_sha256"}
    )
    swapped = tmp_path / "swapped.json"
    write_json(swapped, package)
    rejected = subprocess.run(
        [
            str(fixture["python"]),
            str(checkout / "scripts/run_g14r22b_background_job.py"),
            "launch",
            "--job-package", str(swapped),
            "--job-root", str(fixture["job"]),
        ],
        cwd=checkout,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "requires acceptance_non_holdout=true" in rejected.stderr
    assert not Path(fixture["job"]).exists()


def test_formal_child_nonzero_is_terminal_failed_without_retry(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path, fail_first_child=True)
    checkout = fixture["checkout"]
    assert isinstance(checkout, Path)
    derived = subprocess.run(
        [
            str(fixture["python"]),
            str(checkout / "scripts/derive_g14r22_holdout_background_job.py"),
            "--request-path", str(fixture["request"]),
            "--command-package-path", str(fixture["package"]),
            "--grant-path", str(fixture["grant"]),
            "--one-time-token-file", str(fixture["token"]),
            "--executor-checkout", str(checkout),
            "--python-executable", str(fixture["python"]),
            "--derived-root", str(fixture["derived"]),
            "--job-root", str(fixture["job"]),
            "--isolated-fixture",
        ],
        cwd=checkout,
        text=True,
        capture_output=True,
        check=False,
    )
    assert derived.returncode == 0, derived.stderr
    launched = subprocess.run(
        [
            str(fixture["python"]),
            str(checkout / "scripts/run_g14r22b_background_job.py"),
            "launch",
            "--job-package", str(Path(fixture["derived"]) / "background_job_package.json"),
            "--job-root", str(fixture["job"]),
        ],
        cwd=checkout,
        text=True,
        capture_output=True,
        check=False,
    )
    assert launched.returncode == 0, launched.stderr
    terminal_path = Path(fixture["job"]) / "terminal_receipt.json"
    deadline = time.time() + 20
    while time.time() < deadline and not terminal_path.is_file():
        time.sleep(0.05)
    terminal = json.loads(terminal_path.read_text())
    assert terminal["status"] == "FAILED"
    assert terminal["failure_category"] == "SCIENTIFIC_FAILED_PARTIAL_OUTPUT"
    assert terminal["scientific_failure_receipt"]["consumed_permanently"] is True
    assert terminal["holdout_opened"] is True
    assert terminal["retry_allowed"] is False
    assert terminal["resume_allowed"] is False
    assert terminal["reopen_allowed"] is False


def test_atomic_open_rechecks_expiry_without_consuming(
    tmp_path: Path, monkeypatch,
) -> None:
    fixture = build_fixture(tmp_path)
    request = read_json(fixture["request"])
    package = read_json(fixture["package"])
    grant = read_json(fixture["grant"])
    issued = datetime.fromisoformat(grant["issued_at"])
    grant["expires_at"] = (issued + timedelta(seconds=2)).isoformat()
    times = iter(
        [
            issued + timedelta(seconds=1),
            issued + timedelta(seconds=2),
        ]
    )
    monkeypatch.setattr(
        "src.evaluators.dedicated_holdout_execution._utc_now_datetime",
        lambda: next(times),
    )
    try:
        execute_package(
            request,
            package,
            grant,
            Path(fixture["token"]).read_bytes(),
            isolated_fixture=True,
        )
    except Exception as exc:
        assert "future or expired at atomic_open" in str(exc)
    else:
        raise AssertionError("expired grant unexpectedly reached atomic opening")
    assert not Path(fixture["output"]).exists()
    assert not list(Path(fixture["output"]).parent.glob(".formal_fixture_output.opening-*"))


def test_formal_missing_or_wrong_receipt_never_counts_as_success(tmp_path: Path) -> None:
    receipt_path = tmp_path / "missing.json"
    package = {
        "execution_mode": "formal_holdout",
        "terminal_contract": {
            "scientific_receipt_path": str(receipt_path),
            "expected_receipt_bindings": {"request_sha256": "r" * 64},
        },
    }
    try:
        scientific_completion(package, 0)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing formal receipt unexpectedly succeeded")
    write_json(
        receipt_path,
        {
            "status": "completed_permanently_consumed",
            "acceptance_non_holdout": True,
            "holdout_opened": False,
            "request_sha256": "r" * 64,
        },
    )
    try:
        scientific_completion(package, 0)
    except ValueError as exc:
        assert "formal holdout execution receipt" in str(exc)
    else:
        raise AssertionError("wrong-identity formal receipt unexpectedly succeeded")
