"""Real public main bootstrap/handoff acceptance with synthetic child work only."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from scripts.continuation_executor.locking import SingleWriter, writer_lock_path
from src.runtime.evaluation_only_execution import PHASES, canonical_sha256, file_sha256

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2] if ROOT.parent.name == "execution_checkouts" else ROOT
SOURCE_REQUEST = PROJECT / "artifacts/analysis/g14r20_i1_final_acceptance_20260911/authorization_request_release.json"
PYTHON = str(PROJECT / ".venv/bin/python")
DRIVER = ROOT / "tests/evaluation_only_public_driver.py"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def inventory(root):
    return {str(p.relative_to(root)): file_sha256(p) for p in root.rglob("*") if p.is_file()}


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def environment():
    return dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")


@pytest.fixture
def fixture(tmp_path, request):
    evidence = os.environ.get("G14R20_I2_EVIDENCE_ROOT")
    parent = Path(evidence) if evidence else tmp_path.resolve()
    scope = parent / ("synthetic_" + request.node.name.replace("[", "_").replace("]", "").replace("/", "_"))
    scope.mkdir(parents=True, exist_ok=False)
    executor = PROJECT / "artifacts/execution_checkouts/g14r20_i1_executor_release" if request.node.name == "test_old_public_initialization_conflict" else ROOT
    original = json.loads(SOURCE_REQUEST.read_text())
    source = original["model_source_reference"]
    run = scope / "synthetic_evaluation_run"
    package_path = scope / "request.json"
    command = [PYTHON, str(executor / "scripts/prepare_typed_model_cache_evaluation_only.py"),
               "--action", "prepare", "--disposition-eligibility-path", source["disposition_eligibility_path"],
               "--source-run-root", source["source_run_root"], "--scientific-checkout", source["scientific_checkout"],
               "--executor-checkout", str(executor), "--executor-commit",
               subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=executor, text=True).strip(),
               "--python-executable", PYTHON, "--evaluation-run-id", run.name,
               "--evaluation-run-root", str(run), "--output-path", str(package_path)]
    result = subprocess.run(command, cwd=ROOT, env=environment(), text=True, capture_output=True)
    dump(scope / "prepare.command.json", {"argv": command, "cwd": str(ROOT), "returncode": result.returncode})
    (scope / "prepare.stdout").write_text(result.stdout); (scope / "prepare.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    package = json.loads(package_path.read_text()); execution = package["evaluation_execution_contract"]
    dump(scope / "synthetic_binding.json", {"test_only": True, "scope": str(scope),
                                           "request_sha256": package["authorization_request_sha256"]})
    review = {"status": "pass", "reviewer_id": "test-only-fixture-review", "implementation_agent_id": "test-only-fixture-builder",
              "execution_contract_sha256": execution["execution_contract_sha256"],
              "source_reference_sha256": execution["model_source_reference_sha256"], "holdout_capability": False,
              "test_only": True, "scope": str(scope)}
    review_path = scope / "synthetic_review.json"; dump(review_path, review)
    now = datetime.now(timezone.utc)
    grant = {"version": "1.0.0", "status": "AUTHORIZED_FOR_EVALUATION_ONLY",
             "authorization_request_sha256": package["authorization_request_sha256"],
             "evaluation_run_id": run.name, "model_source_reference_sha256": execution["model_source_reference_sha256"],
             "execution_contract_sha256": execution["execution_contract_sha256"], "executor_commit": execution["executor_commit"],
             "phases": list(PHASES), "holdout_capability": False,
             "independent_review": {"path": str(review_path), "sha256": file_sha256(review_path), "size_bytes": review_path.stat().st_size},
             "issued_at": (now-timedelta(minutes=1)).isoformat(), "expires_at": (now+timedelta(hours=2)).isoformat()}
    dump(scope / "synthetic_grant.json", grant)
    return scope, run, package


def command(scope, phase=PHASES[0], fault="none"):
    return [PYTHON, str(DRIVER), "host", str(scope), fault,
            "--authorization-request-path", str(scope / "request.json"),
            "--project-grant-path", str(scope / "synthetic_grant.json"),
            "--phase", phase, "--check", "execute"]


def invoke(fixture, phase=PHASES[0], fault="none", label="execute"):
    scope, run, package = fixture
    argv = command(scope, phase, fault)
    result = subprocess.run(argv, cwd=ROOT, env=environment(), text=True, capture_output=True)
    dump(scope / (label + ".command.json"), {"argv": argv, "cwd": str(ROOT), "returncode": result.returncode,
                                          "environment": {k:environment()[k] for k in ("PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE")}})
    (scope / (label+".stdout")).write_text(result.stdout); (scope / (label+".stderr")).write_text(result.stderr)
    return result


def reject(fixture, **kwargs):
    scope, run, _ = fixture
    count = len(rows(scope / "dispatch.jsonl")); before = inventory(run)
    directories = sorted(str(p.relative_to(run)) for p in run.rglob("*") if p.is_dir())
    result = invoke(fixture, **kwargs)
    assert result.returncode != 0
    assert len(rows(scope / "dispatch.jsonl")) == count
    assert inventory(run) == before
    assert directories == sorted(str(p.relative_to(run)) for p in run.rglob("*") if p.is_dir())
    return result


def test_public_first_and_second_process(fixture):
    scope, run, package = fixture
    assert not run.exists()
    first = invoke(fixture, label="first")
    assert first.returncode == 0, first.stderr
    first_files = inventory(run / PHASES[0]); phase_prefix = (run / "phase_state.jsonl").read_bytes()
    cell_prefix = (run / "cell_state.jsonl").read_bytes()
    initial_files = {p.name: file_sha256(p) for p in run.glob("*.json")}
    assert len(rows(scope / "dispatch.jsonl")) == 3
    second = invoke(fixture, PHASES[1], label="second")
    assert second.returncode == 0, second.stderr
    assert len(rows(scope / "dispatch.jsonl")) == 6
    assert inventory(run / PHASES[0]) == first_files
    assert (run / "phase_state.jsonl").read_bytes().startswith(phase_prefix)
    assert (run / "cell_state.jsonl").read_bytes().startswith(cell_prefix)
    assert all(file_sha256(run/name) == digest for name,digest in initial_files.items())
    completed = [r["phase"] for r in rows(run / "phase_state.jsonl") if r["status"] == "completed"]
    assert completed == list(PHASES[:2])
    assert len({json.loads(p.read_text())["pid"] for p in scope.glob("host_*.json")}) == 2
    context_hash = file_sha256(run / "resolved_execution_context.json")
    assert all(r["resolved_execution_context_file_sha256"] == context_hash for r in rows(run / "phase_state.jsonl"))
    dump(scope / "handoff_verified.json", {"completed_phases": completed, "synthetic_dispatch": 6,
         "scientific_dispatch": 0, "first_phase_unchanged": True, "context_file_sha256": context_hash,
         "formal_execution_authorized": False, "formal_execution_started": False, "holdout_opened": False})


@pytest.mark.parametrize("kind", ["unknown_nonempty", "empty_existing", "missing_marker", "context_bytes", "identity", "missing_cell_ledger", "directory_identity", "phase_context", "duplicate", "jump", "tampered_payload", "invalid_grant", "expired_grant", "held_lock", "permission"])
def test_public_rejections(fixture, kind):
    scope, run, package = fixture
    phase = PHASES[0]
    if kind in {"missing_marker", "context_bytes", "identity", "missing_cell_ledger", "directory_identity", "phase_context", "duplicate", "jump", "tampered_payload"}:
        result = invoke(fixture, label="setup"); assert result.returncode == 0, result.stderr
        phase = PHASES[1]
    if kind == "unknown_nonempty": run.mkdir(); (run / "unknown").write_text("preserve")
    if kind == "empty_existing": run.mkdir()
    if kind == "missing_marker": (run / "evaluation_initialization_complete.json").unlink()
    if kind == "missing_cell_ledger": (run / "cell_state.jsonl").unlink()
    if kind == "context_bytes":
        with (run / "resolved_execution_context.json").open("a") as f: f.write(" ")
    if kind == "identity":
        value = json.loads((run / "evaluation_execution_contract.json").read_text()); value["evaluation_run_id"] = "other"
        dump(run / "evaluation_execution_contract.json", value)
    if kind == "tampered_payload": next((run / PHASES[0]).rglob("aggregate_summary.json")).write_text("{}")
    if kind == "directory_identity":
        value = json.loads((run / "evaluation_initialization_complete.json").read_text())
        value["run_root"] = str(scope / "other")
        dump(run / "evaluation_initialization_complete.json", value)
    if kind == "phase_context":
        from src.evaluators.formal_phase_transaction import _record_hash
        ledger = rows(run / "phase_state.jsonl")
        previous = None
        for row in ledger:
            row["resolved_execution_context_file_sha256"] = "0" * 64
            row["previous_record_hash"] = previous
            row["current_record_hash"] = _record_hash(row)
            previous = row["current_record_hash"]
        (run / "phase_state.jsonl").write_text("".join(json.dumps(row)+"\n" for row in ledger))
    if kind == "duplicate": phase = PHASES[0]
    if kind == "jump": phase = PHASES[2]
    if kind in {"invalid_grant", "expired_grant"}:
        value = json.loads((scope / "synthetic_grant.json").read_text())
        if kind == "invalid_grant": value["status"] = "NOT_AUTHORIZED"
        else: value["expires_at"] = "2000-01-01T00:00:00+00:00"
        dump(scope / "synthetic_grant.json", value)
    if kind == "held_lock":
        path = writer_lock_path(run); path.parent.mkdir(exist_ok=True)
        dump(path, {"state": "held", "process": {"pid": 99999999}, "test_only": True})
        before = path.read_bytes()
    if kind == "permission":
        scope.chmod(0o500)
        argv = command(scope, phase)
        try:
            result = subprocess.run(argv, cwd=ROOT, env=environment(), text=True, capture_output=True)
        finally:
            scope.chmod(0o700)
        (scope / "permission.stderr").write_text(result.stderr)
        (scope / "permission.stdout").write_text(result.stdout)
        dump(scope / "permission.command.json", {"argv": argv, "cwd": str(ROOT), "returncode": result.returncode})
        assert result.returncode != 0 and "parent is not writable" in result.stderr
        assert not rows(scope / "dispatch.jsonl")
    else:
        reject(fixture, phase=phase, label="rejected")
    if kind == "held_lock": assert path.read_bytes() == before
    if kind in {"invalid_grant", "expired_grant", "held_lock", "permission"}: assert not run.exists()


@pytest.mark.parametrize("boundary", ["root_created", "phase_state.jsonl", "evaluation_model_source_reference.json", "evaluation_execution_contract.json", "resolved_execution_context.json", "evaluation_only_state.json", "cell_ledger_identity.json", "cell_state.jsonl", "evaluation_initialization_complete.json"])
def test_public_initialization_faults(fixture, boundary):
    scope, run, _ = fixture
    result = invoke(fixture, fault=boundary, label="fault")
    assert result.returncode != 0 and "test-only initialization write fault" in result.stderr
    assert not rows(scope / "dispatch.jsonl")
    assert run.exists()
    if boundary == "cell_ledger_identity.json": assert not (run / "cell_state.jsonl").exists()
    assert json.loads(writer_lock_path(run).read_text())["state"] == "held"
    reject(fixture, label="retry_rejected")


def test_public_child_failure_cannot_resume(fixture):
    scope, run, _ = fixture
    (scope / "fail_child").touch()
    result = invoke(fixture, label="failed")
    assert result.returncode != 0
    assert len(rows(scope / "dispatch.jsonl")) == 1
    assert any(row["status"] == "failed" for row in rows(run / "phase_state.jsonl"))
    reject(fixture, label="same_rejected")
    reject(fixture, phase=PHASES[1], label="next_rejected")


@pytest.mark.parametrize("boundary", ["before_initialization", "during_child"])
def test_public_concurrent_first_start(fixture, boundary):
    scope, run, _ = fixture
    (scope / ("pause_before_initialize" if boundary == "before_initialization" else "pause_child")).touch()
    argv = command(scope)
    with (scope / "winner.stdout").open("w") as stdout, (scope / "winner.stderr").open("w") as stderr:
        winner = subprocess.Popen(argv, cwd=ROOT, env=environment(), stdout=stdout, stderr=stderr)
        try:
            deadline = time.monotonic() + 30
            ready = scope / ("initializer_ready" if boundary == "before_initialization" else "child_ready")
            while not ready.exists():
                assert winner.poll() is None
                assert time.monotonic() < deadline
                time.sleep(.05)
            if boundary == "before_initialization": assert not run.exists()
            before = inventory(run); lock_before = writer_lock_path(run).read_bytes()
            reject(fixture, label="loser")
            assert inventory(run) == before
            assert writer_lock_path(run).read_bytes() == lock_before
            (scope / ("release_initializer" if boundary == "before_initialization" else "release_child")).touch()
            assert winner.wait(timeout=30) == 0
        finally:
            if winner.poll() is None: winner.terminate(); winner.wait(timeout=10)
    dump(scope / "winner.command.json", {"argv": argv, "cwd": str(ROOT), "returncode": winner.returncode})
    assert len(rows(scope / "dispatch.jsonl")) == 3


def test_continuation_default_still_rejects_missing_root(tmp_path):
    root = tmp_path.resolve() / "missing"
    with pytest.raises(ValueError, match="may not create"):
        with SingleWriter(root, "test", lambda: None): pass
    assert not root.exists()


def test_old_public_initialization_conflict(fixture):
    scope, run, package = fixture
    release = Path(package["evaluation_execution_contract"]["executor_checkout"])
    argv = [PYTHON, str(release / "scripts/run_typed_model_cache_evaluation_only.py"),
            *command(scope)[5:]]
    result = subprocess.run(argv, cwd=release, env=dict(environment(), PYTHONPATH=str(release)),
                            capture_output=True, text=True)
    dump(scope / "old.command.json", {"argv": argv, "cwd": str(release), "returncode": result.returncode})
    (scope / "old.stderr").write_text(result.stderr); (scope / "old.stdout").write_text(result.stdout)
    assert result.returncode != 0 and "phase output root conflict" in result.stderr
    assert set(inventory(run)) == {"evaluation_model_source_reference.json", "evaluation_execution_contract.json",
                                  "resolved_execution_context.json", "evaluation_only_state.json",
                                  "cell_ledger_identity.json", "cell_state.jsonl"}
    assert (run / "cell_state.jsonl").stat().st_size == 0
    assert not writer_lock_path(run).exists()
    assert not rows(scope / "dispatch.jsonl")
