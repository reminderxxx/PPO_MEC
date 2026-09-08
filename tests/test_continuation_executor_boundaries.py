"""B boundaries use an isolated authority; never read a real weight or run."""
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.continuation_executor import PHASES
from scripts.continuation_executor.authorization import (
    approval_message, validate_contract, verify_approval,
)
from scripts.continuation_executor.identity import ContinuationError, canonical, digest, file_hash
from scripts.continuation_executor.isolation import FixtureScope
from scripts.continuation_executor.locking import SingleWriter, writer_lock_path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture(tmp_path):
    run = tmp_path / "synthetic_b"
    run.mkdir()
    p = {"run_id": run.name, "run_root": str(run), "ledgers": []}
    identity = {"version": "1.0.0", "commit": "a" * 40, "git_tree": "b" * 40, "files": []}
    contract = dict(version="1.0.0", domain="synthetic", proposal_sha256=digest(p),
                    proposal_file_sha256="c"*64, executor_identity_sha256=digest(identity),
                    run_id=run.name, run_root=str(run), phases=list(PHASES), holdout_capability=False,
                    prefixes=[], immutable_files=[], fixture_root=str(tmp_path),
                    expires_at=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
                    command_plan_sha256=digest({}), revocation_id="test-grant-1", recovery_owner_sha256=None, recovery_quiescence=None, coordination_root=str(tmp_path/".continuation_locks"))
    authority = dict(domain="synthetic", fixture_root=str(tmp_path), signer_id="fixture-only",
                     key_hex=os.urandom(32).hex(), revoked_ids=[])
    return p, identity, contract, authority


def sign(contract, authority):
    evidence = {k: {"state": "test_only", "reference_sha256": "f"*64} for k in (
        "launch_approval", "release_attestation", "continuation_approval")}
    message = approval_message(contract, evidence)
    return dict(message=message, signer_id=authority["signer_id"], signature=hmac.new(
        bytes.fromhex(authority["key_hex"]), canonical(message), hashlib.sha256).hexdigest())


def test_fixture_scope_signature_and_distinct_evidence(fixture):
    p, identity, contract, authority = fixture
    validate_contract(contract, p, identity)
    report = verify_approval(contract, sign(contract, authority), fixture_authority=authority)
    assert report["approval_verified"] and not report["real_execution_authorized"]


@pytest.mark.parametrize("mutation", ["scope", "expiry", "revoked", "cross_root", "signature", "production"])
def test_test_authority_cannot_escape(fixture, mutation):
    p, identity, contract, authority = fixture
    approval = sign(contract, authority)
    if mutation == "scope": contract["phases"] += ["holdout"]
    if mutation == "expiry": contract["expires_at"] = "2000-01-01T00:00:00+00:00"
    if mutation == "revoked": authority["revoked_ids"] = [contract["revocation_id"]]
    if mutation == "cross_root": authority["fixture_root"] += "_other"
    if mutation == "signature": approval["signature"] = digest(approval["message"])
    if mutation == "production":
        contract["domain"] = "production"
        contract["fixture_root"] = None
        approval = sign(contract, authority)
    with pytest.raises(ContinuationError):
        validate_contract(contract, p, identity)
        verify_approval(contract, approval, fixture_authority=authority)


@pytest.mark.parametrize("phase", ["train", "dev_select", "checkpoint_freeze", "holdout", "unknown"])
def test_public_cli_phase_rejects_before_paths_exist(tmp_path, phase):
    before = list(tmp_path.iterdir())
    result = subprocess.run([sys.executable, "-B", str(ROOT/"scripts/execute_fixed_commit_continuation.py"),
                             "--proposal", str(tmp_path/"absent"), "--contract", str(tmp_path/"absent"),
                             "--executor-identity", str(tmp_path/"absent"), "--phase", phase], capture_output=True)
    assert result.returncode == 2
    assert list(tmp_path.iterdir()) == before


@pytest.mark.parametrize("check", ["qualification", "execute"])
@pytest.mark.parametrize("phase", PHASES)
def test_public_cli_no_approval_no_writes(fixture, tmp_path, check, phase):
    p, identity, contract, authority = fixture
    contract.update(domain="production", fixture_root=None)
    pp = tmp_path/"proposal.json"; pp.write_bytes(canonical(p))
    contract["proposal_file_sha256"] = file_hash(pp)
    cp = tmp_path/"contract.json"; cp.write_bytes(canonical(contract))
    ip = tmp_path/"identity.json"; ip.write_bytes(canonical(identity))
    before = {str(f): file_hash(f) for f in tmp_path.rglob("*") if f.is_file()}
    result = subprocess.run([sys.executable, "-B", str(ROOT/"scripts/execute_fixed_commit_continuation.py"),
                             "--proposal", str(pp), "--contract", str(cp), "--executor-identity", str(ip),
                             "--phase", phase, "--check", check], capture_output=True, text=True)
    assert result.returncode == 2
    assert "approval unavailable" in result.stdout
    assert before == {str(f): file_hash(f) for f in tmp_path.rglob("*") if f.is_file()}


@pytest.mark.parametrize("kind", ["symlink", "escape", "real_resource", "nested_argv", "uri"])
def test_recursive_input_refusal(fixture, tmp_path, kind):
    p, _, _, _ = fixture
    scope = FixtureScope(str(tmp_path), p["run_root"], str(ROOT), sys.executable)
    payload = {"nested": [{"checkpoint_path": str(tmp_path/"test.pt")} ]}
    if kind == "symlink":
        (tmp_path/"test.pt").symlink_to("/etc/passwd")
    if kind == "escape": payload["nested"][0]["checkpoint_path"] = "../outside.pt"
    if kind == "real_resource": payload["nested"][0]["checkpoint_path"] = str(ROOT/"data/raw/real.csv")
    if kind == "nested_argv": payload = {"argv": ["--checkpoint=/some/real.pt"]}
    if kind == "uri": payload = {"resource": "https://example.invalid/weight"}
    with pytest.raises(ContinuationError): scope.value(payload)


def test_lock_refusal_is_before_file_creation(fixture):
    p, _, _, _ = fixture
    def deny(): raise ContinuationError("unapproved")
    with pytest.raises(ContinuationError):
        with SingleWriter(p["run_root"], "x", deny): pass
    assert not list(Path(p["run_root"]).iterdir())


def test_lock_competition_and_repeated_start(fixture):
    p, _, _, _ = fixture
    with SingleWriter(p["run_root"], "x", lambda: None):
        with pytest.raises(BlockingIOError):
            with SingleWriter(p["run_root"], "x", lambda: None): pass
    path = writer_lock_path(p["run_root"])
    inode = path.stat().st_ino
    with SingleWriter(p["run_root"], "x", lambda: None): pass
    assert path.stat().st_ino == inode
    assert json.loads(path.read_text())["state"] == "released"


def test_crash_record_cannot_be_blindly_reclaimed(fixture):
    p, _, _, _ = fixture
    with pytest.raises(RuntimeError):
        with SingleWriter(p["run_root"], "x", lambda: None): raise RuntimeError("crash")
    with pytest.raises(ContinuationError, match="exact approved previous owner"):
        with SingleWriter(p["run_root"], "x", lambda: None): pass
    path = writer_lock_path(p["run_root"])
    with pytest.raises(ContinuationError, match="not proven gone"):
        with SingleWriter(p["run_root"], "x", lambda: None, digest(json.loads(path.read_text()))): pass


def test_planner_projections_equal_frozen_original_main_expressions():
    """Only expression equivalence, explicitly not a producer/consumer test."""
    import ast
    from scripts.continuation_executor.planning import run_identity, cell_identity_fields, phase_plan
    from src.evaluators.typed_model_cache_formal_execution import expand_command_plan
    from src.evaluators.formal_cell_transaction import CellExecutionIdentity
    frozen = Path("/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847")
    source = frozen/"scripts/run_typed_model_cache_formal_protocol.py"
    expected = subprocess.check_output(["git", "-C", str(frozen), "show",
        "a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d:scripts/run_typed_model_cache_formal_protocol.py"])
    assert source.read_bytes() == expected
    tree = ast.parse(expected)
    protocol = json.loads((ROOT/"configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906/protocol_v2_9_manifest.json").read_text())
    context = {"context_sha256": "context", "scientific_identity": {"active_formal_bundle_sha256": "bundle"},
               "resolved_expansion_context": dict(protocol["execution_contract"]["default_expansion_context"])}
    context["resolved_expansion_context"].update(python_executable=sys.executable, repository_root="/fixture/source", clean_worktree_root="/fixture/source", output_root="/fixture/synthetic_b", protocol_path="/fixture/protocol.json", active_formal_bundle_sha256="bundle", active_bundle_resource_resolution_audit_sha256="audit")
    binding = {"binding_full_sha256": "binding"}
    from types import SimpleNamespace
    env = SimpleNamespace(environment_identity={"environment_fingerprint": "environment"},
                          runtime_audit={"observed_execution_commit": "commit"})
    local = dict(protocol=protocol, requested_output_root="/fixture/synthetic_b", environment_resolution=env,
                 resolved_context_payload=context, resolved_context_file_sha256="context-file",
                 execution_binding=binding, active_bundle={"active_formal_bundle_sha256": "bundle"},
                 canonical_sha256=digest, CellExecutionIdentity=CellExecutionIdentity,
                 Path=Path, args=SimpleNamespace(output_root="/fixture/synthetic_b"))
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
    def expr(name):
        nodes = [n.value for n in assignments if any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        return nodes
    original_run = eval(compile(ast.Expression(expr("run_identity")[0]), str(source), "eval"), local)
    assert run_identity(protocol, context, binding, "context-file", "/fixture/synthetic_b", "commit", "environment") == original_run
    original_cell = eval(compile(ast.Expression(expr("cell_identity")[0]), str(source), "eval"), local)
    assert cell_identity_fields(protocol, context, binding, "synthetic_b", "commit", "environment") == original_cell.to_dict()
    for phase in PHASES:
        plan = phase_plan(phase, protocol, context, binding, "context-file", "registry", expand_command_plan)
        base = eval(compile(ast.Expression(expr("input_hash")[0]), str(source), "eval"),
                    {**local, "phase": phase, "command": plan["commands"]})
        original_phase = eval(compile(ast.Expression(expr("input_hash")[1]), str(source), "eval"),
                    {**local, "input_hash": base, "generated_registry_audit": {"registry_canonical_sha256": "registry"}})
        assert plan["input_hash"] == original_phase

    # Evaluate only the frozen original staging builder and identity expression;
    # no public main, transaction, child or file operation is invoked here.
    from scripts.continuation_executor.cells import cell_layout
    from scripts.continuation_executor.planning import cell_input_hash
    import src.evaluators.formal_cell_transaction as cell_native
    build_node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="build"
        and any(isinstance(c,ast.Constant) and c.value=="formal_cache_policy" for c in ast.walk(n)))
    training_branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and any(child is build_node for child in n.orelse))
    selector=training_branch.orelse[:next(i for i,n in enumerate(training_branch.orelse) if n is build_node)]
    original_input=expr("cell_input_hash")[0]
    for phase in PHASES[:5]:
        plan=phase_plan(phase,protocol,context,binding,"context-file","registry",expand_command_plan)
        for original,coordinates in zip(plan["commands"],plan["matrix_contexts"]):
            flag="--output_root" if "--output_root" in original else "--output-root"
            env=dict(local,phase=phase,original=original,coordinates=coordinates,output_flag=flag,
                final_output_root=Path(original[original.index(flag)+1]),stable_cell_id=cell_native.stable_cell_id,
                generated_registry_audit={"registry_canonical_sha256":"registry"},FormalExecutionError=ValueError)
            exec(compile(ast.Module(body=[*selector,build_node],type_ignores=[]),str(source),"exec"),env)
            final,build,_=cell_layout(phase,coordinates,original,cell_native)
            cell_id=cell_native.stable_cell_id(phase,coordinates)
            staging=Path("/fixture/staging")/phase/cell_id
            assert final==env["final_path"]
            assert build(staging,cell_id)==env["build"](staging,cell_id)
            expected=eval(compile(ast.Expression(original_input),str(source),"eval"),env)
            assert cell_input_hash(phase,coordinates,original,protocol,context,"registry")==expected


def test_real_process_crash_requires_approved_recovery(fixture):
    p, _, _, _ = fixture
    code = '''
import os, sys
from scripts.continuation_executor.locking import SingleWriter, writer_lock_path
with SingleWriter(sys.argv[1], "executor", lambda: None):
    print("acquired", flush=True)
    os._exit(17)
'''
    child = subprocess.run([sys.executable, "-B", "-c", code, p["run_root"]], cwd=ROOT,
                           capture_output=True, text=True)
    assert child.returncode == 17 and child.stdout.strip() == "acquired"
    path = writer_lock_path(p["run_root"])
    previous = json.loads(path.read_text())
    with pytest.raises(ContinuationError):
        with SingleWriter(p["run_root"], "executor", lambda: None): pass
    with SingleWriter(p["run_root"], "executor", lambda: None, digest(previous)): pass
    assert json.loads(path.read_text())["state"] == "released"


def test_five_cells_use_old_native_transactions(tmp_path):
    old = "/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847"
    result = subprocess.run([sys.executable, "-I", "-B", str(ROOT/"tests/continuation_native_driver.py"), str(tmp_path)],
        cwd=old, env=dict(os.environ, PYTHONPATH=old, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    report=json.loads(result.stdout)
    assert report["synthetic_child_dispatch_count"] == 5
    assert report["prefix_unchanged"]
    assert report["completed"] == list(PHASES[:5])
    assert all(row["path"].startswith(old+"/") for row in report["origins"].values())


@pytest.mark.parametrize("mutation", ["pythonpath", "cwd", "shadow"])
def test_old_science_loader_rejects_process_pollution(tmp_path, mutation):
    old = "/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847"
    setup = ""
    if mutation == "shadow":
        setup = "sys.path.insert(0,"+repr(str(ROOT))+");import src;sys.path.pop(0)\n"
    code = "import sys\nsys.path.insert(0,"+repr(str(ROOT/"scripts"))+")\n"+setup+"from continuation_executor.scientific import load_native\nload_native("+repr(old)+",'a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d')\n"
    env = dict(os.environ,PYTHONPATH=str(tmp_path) if mutation=="pythonpath" else old,PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1")
    result=subprocess.run([sys.executable,"-I","-B","-c",code],cwd=str(tmp_path) if mutation=="cwd" else old,env=env,capture_output=True,text=True)
    assert result.returncode != 0
    assert "drift" in result.stderr or "shadow scientific" in result.stderr
    assert not list(tmp_path.iterdir())


def test_two_real_processes_only_one_writer(fixture):
    p,_,_,_=fixture
    code="""import sys
from scripts.continuation_executor.locking import SingleWriter
with SingleWriter(sys.argv[1], 'test', lambda: None):
    print('locked',flush=True)
    sys.stdin.readline()
"""
    first=subprocess.Popen([sys.executable,"-B","-c",code,p["run_root"]],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        assert first.stdout.readline().strip()=="locked"
        second=subprocess.run([sys.executable,"-B","-c",code,p["run_root"]],cwd=ROOT,input="\n",capture_output=True,text=True)
        assert second.returncode != 0 and "BlockingIOError" in second.stderr
        first.communicate("\n",timeout=10)
        assert first.returncode == 0
    finally:
        if first.poll() is None:
            first.kill();first.wait()


@pytest.mark.parametrize("case", ["exit75", "terminal", "missing", "corrupt", "descriptor", "provenance", "publication_crash", "candidate_crash", "duplicate_committed", "gate_missing", "gate_false", "revoke_during_phase", "expire_during_phase", "utc_adjustment"])
def test_native_faults_and_restarts(tmp_path, case):
    old = "/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847"
    result = subprocess.run([sys.executable, "-I", "-B", str(ROOT/"tests/continuation_native_driver.py"), str(tmp_path), case],
        cwd=old, env=dict(os.environ, PYTHONPATH=old, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "pass" and report["prefix_unchanged"]
    assert report["synthetic_child_dispatch_count"] > 0


@pytest.mark.parametrize("case", ["truncation", "fork", "out_of_order", "cross_ledger", "immutable_payload"])
def test_native_progress_corruption_rejects_before_write(tmp_path, case):
    old = "/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847"
    result = subprocess.run([sys.executable, "-I", "-B", str(ROOT/"tests/continuation_native_driver.py"), str(tmp_path), case],
        cwd=old, env=dict(os.environ, PYTHONPATH=old, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report=json.loads(result.stdout)
    assert report["status"]=="pass" and report["rejected_before_write"]
    assert report["synthetic_child_dispatch_count"]==0


@pytest.mark.parametrize("entry", ["execute_fixed_commit_continuation.py", "run_fixed_commit_continuation_acceptance.py"])
def test_actual_cli_identity_drift_has_no_fixture_writes(tmp_path, fixture, entry):
    p, identity, contract, authority=fixture
    ip=tmp_path/"identity.json";ip.write_bytes(canonical(identity))
    target=tmp_path/"synthetic_new"
    if entry.startswith("run_"):
        args=["--executor-identity",str(ip),"--fixture-root",str(target)]
    else:
        contract.update(domain="production",fixture_root=None)
        pp=tmp_path/"proposal.json";pp.write_bytes(canonical(p))
        contract["proposal_file_sha256"]=file_hash(pp)
        cp=tmp_path/"contract.json";cp.write_bytes(canonical(contract))
        args=["--executor-identity",str(ip),"--proposal",str(pp),"--contract",str(cp),
              "--phase",PHASES[0],"--check","compatibility"]
    before={str(f):file_hash(f) for f in tmp_path.rglob("*") if f.is_file()}
    result=subprocess.run([sys.executable,"-B",str(ROOT/"scripts"/entry),*args],capture_output=True,text=True)
    assert result.returncode==2 and "fixed commit/tree drift" in result.stdout
    assert not target.exists()
    assert before=={str(f):file_hash(f) for f in tmp_path.rglob("*") if f.is_file()}



def test_actual_foreign_interpreter_rejected_before_scientific_imports(tmp_path):
    old="/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847"
    code="import sys;sys.path.insert(0,"+repr(str(ROOT/"scripts"))+");from continuation_executor.scientific import load_native;load_native("+repr(old)+",'a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d')"
    result=subprocess.run(["/usr/bin/python3","-I","-B","-c",code],cwd=old,
        env=dict(os.environ,PYTHONPATH=old,PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1"),capture_output=True,text=True)
    assert result.returncode!=0 and "actual interpreter differs" in result.stderr
    assert not list(tmp_path.iterdir())



@pytest.mark.parametrize("kind", ["external_src", "critical_package"])
def test_actual_external_import_pollution_rejected(tmp_path, kind):
    old="/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847"
    if kind=="external_src":
        (tmp_path/"src").mkdir();(tmp_path/"src/__init__.py").write_text("")
        setup="sys.path.insert(0,"+repr(str(tmp_path))+")"
        expected="external/current-main src"
    else:
        setup="import types;m=types.ModuleType('torch');m.__version__='2.8.0';m.__file__="+repr(str(tmp_path/"shadow.py"))+";sys.modules['torch']=m"
        expected="critical dependency shadow import"
    before={str(p):file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    code="import sys;sys.path.insert(0,"+repr(str(ROOT/"scripts"))+");"+setup+";from continuation_executor.scientific import load_native;load_native("+repr(old)+",'a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d')"
    result=subprocess.run([sys.executable,"-I","-B","-c",code],cwd=old,
        env=dict(os.environ,PYTHONPATH=old,PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1"),capture_output=True,text=True)
    assert result.returncode!=0 and expected in result.stderr
    assert before=={str(p):file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
