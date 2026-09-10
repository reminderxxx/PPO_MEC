"""Project authorization regression tests; all run/review IO stays in tmp_path.

The public CLI and authorization/file validators are real. Scientific qualification
and execution are replaced at their IO boundaries; no real run, tensor or rollout
is touched. Git observations are fixed test values, while executor file hashes
are checked against the actual implementation files.
"""
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    cli = importlib.import_module("execute_fixed_commit_continuation")
    pa = importlib.import_module("continuation_executor.project_authorization")
    ident = importlib.import_module("continuation_executor.identity")
    scientific = importlib.import_module("continuation_executor.scientific")
    execution = importlib.import_module("continuation_executor.execution")
    run = tmp_path / pa.RUN_ID
    run.mkdir()
    science_root = tmp_path / "frozen_scientific_source"
    science_root.mkdir()
    monkeypatch.setattr(pa, "RUN_ROOT", str(run))
    monkeypatch.setattr(pa, "SCIENTIFIC_ROOT", str(science_root))

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW if tz is not None else NOW.replace(tzinfo=None)

    monkeypatch.setattr(pa, "datetime", Clock)
    commit, tree = "a" * 40, "b" * 40

    def observed_git(root, *args):
        assert Path(root) == ROOT
        return {
            ("rev-parse", "HEAD"): commit,
            ("rev-parse", "HEAD^{tree}"): tree,
            ("status", "--porcelain", "--untracked-files=all"): "",
            ("rev-parse", "--abbrev-ref", "HEAD"): "test-only",
        }[args]

    monkeypatch.setattr(ident, "git", observed_git)
    required = {
        "scripts/execute_fixed_commit_continuation.py",
        "scripts/run_fixed_commit_continuation_acceptance.py",
        "src/runtime/fixed_commit_continuation.py",
    }
    required.update(p.relative_to(ROOT).as_posix()
                    for p in (ROOT / "scripts/continuation_executor").glob("*.py"))
    identity = dict(version="1.0.0", commit=commit, git_tree=tree,
                    files=[dict(path=p, sha256=ident.file_hash(ROOT / p))
                           for p in sorted(required)])

    def file_row(path):
        return dict(path=str(path), sha256=ident.file_hash(path),
                    size_bytes=path.stat().st_size)

    frozen_input = tmp_path / "frozen_input.txt"
    frozen_input.write_text("test-only immutable source identity\n")
    proof = tmp_path / "independent_test_evidence.txt"
    proof.write_text("test-only review evidence; no scientific result\n")
    proposal = dict(run_id=run.name, run_root=str(run),
                    execution_commit=pa.SCIENTIFIC_COMMIT,
                    worktree_root=str(science_root),
                    ledgers=[dict(kind="phase", record_count=15),
                             dict(kind="cell", record_count=348)],
                    evidence=[dict(kind="immutable_test_input", **file_row(frozen_input))])
    contract = dict(
        version="1.0.0", domain="production", proposal_sha256=ident.digest(proposal),
        proposal_file_sha256="c" * 64, executor_identity_sha256=ident.digest(identity),
        run_id=run.name, run_root=str(run), phases=list(pa.PHASES),
        holdout_capability=False, prefixes=deepcopy(proposal["ledgers"]),
        immutable_files=[file_row(frozen_input)], fixture_root=None,
        expires_at="2026-09-30T23:59:59+08:00", revocation_id=pa.MODE + ":" + run.name,
        recovery_owner_sha256=None, recovery_quiescence=None,
        coordination_root=str(tmp_path / ".continuation_locks"),
        command_plan_sha256=ident.digest({"test_only": list(pa.PHASES)}),
    )
    review = dict(
        version="1.0.0", status="pass", reviewer_id="independent-test-reviewer",
        implementation_agent_id="test-implementation-author",
        reviewed_at="2026-09-10T10:00:00+00:00",
        executor_identity_sha256=ident.digest(identity), contract_sha256="",
        decision_sha256=ident.digest(pa.OWNER_DECISION),
        checks={key: True for key in (
            "code_and_tests", "current_dated_source_verification",
            "immutable_input_protection", "real_read_only_qualification",
            "frozen_commands", "no_holdout", "no_scientific_changes")},
        evidence=[file_row(proof)],
    )
    paths = {key: tmp_path / (key + ".json")
             for key in ("proposal", "contract", "identity", "review", "grant")}
    grant = dict(version="1.0.0", mode=pa.MODE,
                 decision_sha256=ident.digest(pa.OWNER_DECISION),
                 contract_sha256="", executor_identity_sha256=ident.digest(identity),
                 independent_review={}, stop_file=pa.stop_path(contract),
                 issued_at="2026-09-10T11:00:00+00:00")

    def write(key, value):
        paths[key].write_bytes(ident.canonical(value))

    def bind():
        """Rebind test integrity hashes; this never changes owner scope rules."""
        write("proposal", proposal)
        contract["proposal_sha256"] = ident.digest(proposal)
        contract["proposal_file_sha256"] = ident.file_hash(paths["proposal"])
        contract["executor_identity_sha256"] = ident.digest(identity)
        review["executor_identity_sha256"] = ident.digest(identity)
        review["contract_sha256"] = ident.digest(contract)
        grant["contract_sha256"] = ident.digest(contract)
        grant["executor_identity_sha256"] = ident.digest(identity)
        write("review", review)
        grant["independent_review"] = file_row(paths["review"])
        for key, value in (("contract", contract), ("identity", identity), ("grant", grant)):
            write(key, value)

    bind()
    calls = []
    phase_records = [dict(phase=phase, status="completed") for phase in (
        "preflight", "tests", "train", "dev_select", "checkpoint_freeze")]

    def validate_proposal(value):
        calls.append("a_proposal_boundary")
        assert value == proposal

    monkeypatch.setattr(cli, "validate_a_proposal", validate_proposal)

    def qualify(actual_proposal, actual_contract):
        calls.append("qualify")
        assert actual_proposal == proposal and actual_contract == contract
        assert Path(actual_contract["run_root"]) == run
        return SimpleNamespace(reconciliation={"test_only": True}, checkpoints=[], origins=[],
                               run_root=run, phase_runner=SimpleNamespace(records=lambda: list(phase_records)))

    def execute(qualified, phase, authorize, actual_identity):
        calls.append("execute")
        assert qualified.run_root == run and phase in pa.PHASES
        assert actual_identity == identity
        assert authorize()["authorization_mode"] == pa.MODE
        receipt = run / "test_only_execution_receipt.json"
        receipt.write_text(json.dumps({"phase": phase, "rollouts": 0}))
        return {"test_only": True, "rollouts": 0}

    monkeypatch.setattr(scientific, "QualifiedRun", qualify)
    monkeypatch.setattr(execution, "execute_phase", execute)

    def argv(*extra):
        return ["--proposal", str(paths["proposal"]), "--contract", str(paths["contract"]),
                "--executor-identity", str(paths["identity"]),
                "--project-authorization", str(paths["grant"]),
                "--phase", pa.PHASES[0], *extra]

    return SimpleNamespace(**locals())


def verify(project, **kwargs):
    p = project
    return p.pa.verify_project_authorization(p.contract, p.proposal, p.identity, p.grant,
                                             now=kwargs.pop("now", NOW), **kwargs)


def test_project_verifies_real_review_and_evidence_files(project):
    report = verify(project)
    assert report["approval_verified"] and report["real_execution_authorized"]
    assert report["authorization_mode"] == project.pa.MODE
    assert report["holdout_capability"] is False
    assert report["historical_approval_claim"] is False
    assert not list(project.run.iterdir())


@pytest.mark.parametrize("case", [
    "cross_run", "domain", "fixture_root", "scientific_commit", "scientific_root",
    "holdout", "recovery", "prefix_count", "prefix_kind", "immutable_missing",
    "immutable_extra", "immutable_changed", "phase_reorder", "unknown_phase",
    "revocation", "late_expiry", "early_expiry",
])
def test_project_exact_scope_rejects_even_rebound_integrity(project, case):
    p, c = project, project.contract
    if case == "cross_run":
        other = p.tmp_path / "another_run"
        c.update(run_id=other.name, run_root=str(other))
        p.proposal.update(run_id=other.name, run_root=str(other))
    elif case == "domain": c["domain"] = "synthetic"
    elif case == "fixture_root": c["fixture_root"] = str(p.tmp_path)
    elif case == "scientific_commit": p.proposal["execution_commit"] = "f" * 40
    elif case == "scientific_root": p.proposal["worktree_root"] += "_other"
    elif case == "holdout": c["holdout_capability"] = True
    elif case == "recovery":
        c["recovery_owner_sha256"] = "a" * 64
        c["recovery_quiescence"] = dict(owner_sha256="a" * 64,
            state="independently_verified", no_live_descendants=True, reference_sha256="b" * 64)
    elif case == "prefix_count":
        p.proposal["ledgers"][0]["record_count"] += 1
        c["prefixes"] = deepcopy(p.proposal["ledgers"])
    elif case == "prefix_kind":
        p.proposal["ledgers"][0]["kind"] = "other"
        c["prefixes"] = deepcopy(p.proposal["ledgers"])
    elif case == "immutable_missing": c["immutable_files"] = []
    elif case == "immutable_extra": c["immutable_files"].append(p.file_row(p.proof))
    elif case == "immutable_changed": c["immutable_files"][0]["sha256"] = "f" * 64
    elif case == "phase_reorder": c["phases"].reverse()
    elif case == "unknown_phase": c["phases"].append("holdout")
    elif case == "revocation": c["revocation_id"] = "another_owner_scope"
    elif case == "late_expiry": c["expires_at"] = "2026-10-01T00:00:00+08:00"
    elif case == "early_expiry": c["expires_at"] = "2026-09-09T23:59:59+08:00"
    p.bind()
    with pytest.raises((p.ident.ContinuationError, TypeError)):
        verify(p)
    assert not list(p.run.iterdir())


@pytest.mark.parametrize("case", ["expired", "future", "predated", "naive_now"])
def test_project_time_limits(project, case):
    p = project
    now = NOW
    if case == "expired": now = datetime(2026, 10, 1, tzinfo=timezone.utc)
    elif case == "future": p.grant["issued_at"] = "2026-09-10T13:00:00+00:00"
    elif case == "predated": p.grant["issued_at"] = "2026-09-09T00:00:00+00:00"
    elif case == "naive_now": now = NOW.replace(tzinfo=None)
    with pytest.raises(p.ident.ContinuationError):
        verify(p, now=now)


@pytest.mark.parametrize("kind", ["file", "directory", "dangling_symlink"])
def test_owner_stop_marker_always_cancels(project, kind):
    p = project
    stop = Path(p.grant["stop_file"])
    stop.parent.mkdir()
    if kind == "file": stop.write_text("stop")
    elif kind == "directory": stop.mkdir()
    else: stop.symlink_to(p.tmp_path / "absent_stop_target")
    with pytest.raises(p.ident.ContinuationError):
        verify(p)


@pytest.mark.parametrize("case", [
    "review_missing", "review_hash", "review_size", "review_tampered",
    "evidence_missing", "evidence_hash", "evidence_tampered", "evidence_empty",
    "review_nonindependent", "review_no_reviewer", "review_not_pass", "review_binding",
    "review_decision", "review_future", "review_extra", "check_missing", "check_false",
    "grant_extra", "grant_decision", "grant_executor", "grant_contract", "grant_stop",
])
def test_project_review_file_and_content_fail_closed(project, case):
    p = project
    if case == "review_missing": p.paths["review"].unlink()
    elif case == "review_hash": p.grant["independent_review"]["sha256"] = "f" * 64
    elif case == "review_size": p.grant["independent_review"]["size_bytes"] += 1
    elif case == "review_tampered": p.paths["review"].write_text("{}")
    elif case == "evidence_missing": p.proof.unlink()
    elif case == "evidence_hash": p.review["evidence"][0]["sha256"] = "f" * 64
    elif case == "evidence_tampered": p.proof.write_text("tampered")
    elif case == "evidence_empty": p.review["evidence"] = []
    elif case == "review_nonindependent": p.review["reviewer_id"] = p.review["implementation_agent_id"]
    elif case == "review_no_reviewer": p.review["reviewer_id"] = " "
    elif case == "review_not_pass": p.review["status"] = "pending"
    elif case == "review_binding": p.review["executor_identity_sha256"] = "f" * 64
    elif case == "review_decision": p.review["decision_sha256"] = "f" * 64
    elif case == "review_future": p.review["reviewed_at"] = "2026-09-10T11:01:00+00:00"
    elif case == "review_extra": p.review["approved"] = True
    elif case == "check_missing": del p.review["checks"]["no_holdout"]
    elif case == "check_false": p.review["checks"]["no_scientific_changes"] = False
    elif case == "grant_extra": p.grant["approved"] = True
    elif case == "grant_decision": p.grant["decision_sha256"] = "f" * 64
    elif case == "grant_executor": p.grant["executor_identity_sha256"] = "f" * 64
    elif case == "grant_contract": p.grant["contract_sha256"] = "f" * 64
    elif case == "grant_stop": p.grant["stop_file"] += "_other"
    if case.startswith(("check_", "review_", "evidence_")) and case not in {
        "review_missing", "review_hash", "review_size", "review_tampered"}:
        p.write("review", p.review)
        p.grant["independent_review"] = p.file_row(p.paths["review"])
    with pytest.raises((p.ident.ContinuationError, OSError)):
        verify(p)


@pytest.mark.parametrize("check", ["qualification", "execute"])
def test_public_main_project_qualification_and_execution(project, capsys, check):
    p = project
    assert p.cli.main(p.argv("--check", check)) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert p.calls == ["a_proposal_boundary", "qualify"] + (["execute"] if check == "execute" else [])
    assert events[0]["event"] == "qualification"
    assert events[0]["authorization"]["authorization_mode"] == p.pa.MODE
    assert events[-1]["status"] == ("completed" if check == "execute" else "qualified")
    assert events[-1]["holdout_capability"] is False
    assert (p.run / "test_only_execution_receipt.json").exists() == (check == "execute")


@pytest.mark.parametrize("case", ["executor_file_hash", "proposal_byte_hash", "missing_review", "finalize"])
def test_public_main_rejects_before_science_or_execution(project, capsys, case):
    p = project
    extra = []
    if case == "executor_file_hash":
        p.identity["files"][0]["sha256"] = "f" * 64
        p.bind()
    elif case == "proposal_byte_hash": p.paths["proposal"].write_bytes(p.paths["proposal"].read_bytes() + b"\n")
    elif case == "missing_review": p.paths["review"].unlink()
    elif case == "finalize": extra = ["--finalize-phase-only"]
    assert p.cli.main(p.argv("--check", "execute", *extra)) == 2
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert report["status"] == "rejected" and report["execution_authorized"] is False
    assert "qualify" not in p.calls and "execute" not in p.calls
    assert not list(p.run.iterdir())


@pytest.mark.parametrize("kind", ["signature", "startup_binding"])
def test_public_project_cannot_mix_crypto_or_test_binding(project, capsys, kind):
    p = project
    extra = ["--approval", str(p.tmp_path / "missing_signature.json")] if kind == "signature" else []
    kwargs = {"_startup_binding": object()} if kind == "startup_binding" else {}
    assert p.cli.main(p.argv(*extra), **kwargs) == 2
    assert "cannot mix" in capsys.readouterr().out
    assert not p.calls and not list(p.run.iterdir())


@pytest.mark.parametrize("phase", ["holdout", "train", "dev_select", "checkpoint_freeze", "unknown"])
def test_public_project_rejects_unapproved_phase_before_read(project, phase):
    p = project
    with pytest.raises(SystemExit) as error:
        p.cli.main(p.argv("--phase", phase, "--check", "execute"))
    assert error.value.code == 2
    assert not p.calls and not list(p.run.iterdir())


@pytest.mark.parametrize("status", ["running", "completion_candidate", "failed"])
def test_public_execute_refuses_cold_phase_recovery(project, capsys, status):
    p = project
    p.phase_records.append(dict(phase=p.pa.PHASES[0], status=status))
    assert p.cli.main(p.argv("--check", "execute")) == 2
    terminal = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert terminal["status"] == "rejected"
    assert "separate recovery authorization" in terminal["reason"]
    assert p.calls == ["a_proposal_boundary", "qualify"]
    assert not list(p.run.iterdir())


@pytest.mark.parametrize("case", ["already_completed", "all_completed", "skip_first", "skip_next"])
def test_public_execute_requires_next_unstarted_phase(project, capsys, case):
    p = project
    requested = p.pa.PHASES[0]
    if case in {"already_completed", "skip_next"}:
        p.phase_records.append(dict(phase=p.pa.PHASES[0], status="completed"))
    elif case == "all_completed":
        p.phase_records.extend(dict(phase=phase, status="completed") for phase in p.pa.PHASES)
    if case == "skip_first": requested = p.pa.PHASES[1]
    elif case == "skip_next": requested = p.pa.PHASES[2]
    assert p.cli.main(p.argv("--check", "execute", "--phase", requested)) == 2
    terminal = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert terminal["status"] == "rejected"
    assert "next unstarted phase" in terminal["reason"]
    assert p.calls == ["a_proposal_boundary", "qualify"]
    assert not list(p.run.iterdir())


def test_public_execute_advances_after_latest_completed_terminal(project, capsys):
    p = project
    p.phase_records.extend(dict(phase=p.pa.PHASES[0], status=status) for status in (
        "running", "completion_candidate", "completed"))
    assert p.cli.main(p.argv("--check", "execute", "--phase", p.pa.PHASES[1])) == 0
    terminal = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert terminal["status"] == "completed" and terminal["phase"] == p.pa.PHASES[1]
    assert p.calls == ["a_proposal_boundary", "qualify", "execute"]
    receipt = json.loads((p.run / "test_only_execution_receipt.json").read_text())
    assert receipt == {"phase": p.pa.PHASES[1], "rollouts": 0}


@pytest.mark.parametrize("change", ["grant", "stop"])
def test_public_execute_rechecks_grant_and_stop_after_qualification(project, monkeypatch, capsys, change):
    p = project
    original = p.scientific.QualifiedRun

    def qualify_then_change(*args):
        result = original(*args)
        if change == "grant":
            p.paths["grant"].write_bytes(p.paths["grant"].read_bytes() + b"\n")
        else:
            stop = Path(p.grant["stop_file"])
            stop.parent.mkdir()
            stop.write_text("owner cancelled")
        return result

    monkeypatch.setattr(p.scientific, "QualifiedRun", qualify_then_change)
    assert p.cli.main(p.argv("--check", "execute")) == 2
    assert p.calls == ["a_proposal_boundary", "qualify", "execute"]
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "rejected"
    assert not list(p.run.iterdir())
