"""Subprocess-only native transaction exercise (not full B acceptance)."""
import hashlib
import hmac
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

# Imported as an absolute script with -I: executor and old science stay separate.
EXECUTOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXECUTOR / "scripts"))
from continuation_executor import PHASES
from continuation_executor.authorization import approval_message, verify_approval, validate_contract
from continuation_executor.identity import canonical, digest, file_hash
from continuation_executor.scientific import load_native, loaded_origins
from continuation_executor.isolation import FixtureScope
from continuation_executor.execution import execute_phase


def main():
    fixture = Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv)>2 else "valid"
    old = Path.cwd()
    native = load_native(str(old), "a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d")
    run = fixture / "synthetic_transaction"
    phase = native["phase"].TransactionalPhaseRunner(output_root=run, run_identity_fingerprint="phase-id",
                                phase_order=native["execution"].PHASE_ORDER)
    identity = native["cell"].CellExecutionIdentity(**dict(run_id=run.name, execution_commit="a"*40,
        protocol_semantic_sha256="b"*64, resource_registry_semantic_sha256="c"*64,
        environment_fingerprint="d"*64, split_semantic_sha256="e"*64,
        window_contract_semantic_sha256="f"*64, catalog_fingerprint="1"*64,
        runtime_identity="2"*64, command_matrix_sha256="3"*64))
    cells = native["cell"].FormalCellLedger(run_root=run, identity=identity)
    for previous in ("train", "dev_select"):
        begun = cells.begin_cell(phase=previous, coordinates={"fixture": previous}, command=[],
                                 input_hash="prior", committed_path=run / "prior" / previous)
        staging = Path(begun["record"]["staging_path"])
        (staging/"payload.json").write_text('{"test_only":true}')
        cells.commit_cell(begun["cell_id"], required_paths=["payload.json"])
    for previous in native["execution"].PHASE_ORDER[:5]:
        phase.run_phase(previous, commands=[], input_hash="prior", expected_outputs=[])
    prefixes = []
    for kind, hash_key, fingerprint in (("phase", "current_record_hash", "phase-id"), ("cell", "current_ledger_hash", identity.fingerprint)):
        path = run / (kind+"_state.jsonl")
        data = path.read_bytes(); rows = [json.loads(line) for line in data.splitlines()]
        prefixes.append(dict(path=str(path), kind=kind, byte_count=len(data), record_count=len(rows),
                             prefix_sha256=hashlib.sha256(data).hexdigest(), terminal_hash=rows[-1][hash_key],
                             run_identity_fingerprint=fingerprint))
    child = fixture / "payload_child.py"
    child.write_text('''import json, sys
from pathlib import Path
from src.evaluators.formal_cell_transaction import write_child_output_descriptor
phase = sys.argv[1]
mode = sys.argv[sys.argv.index("--fault")+1]
log = Path(__file__).with_name("invocations.jsonl")
prior = log.read_text().splitlines() if log.exists() else []
with log.open("a") as stream:stream.write(json.dumps(sys.argv)+"\\n")
if phase == "formal_cache_policy" and mode == "exit75" and not prior:sys.exit(75)
if phase == "formal_cache_policy" and mode == "terminal":sys.exit(9)
if mode in {"revoke_during_phase","expire_during_phase"} and phase=="formal_cache_policy":
    Path(__file__).with_name("revoke.json").write_text("{}")
flag = "--output-root"
out = Path(sys.argv[sys.argv.index(flag)+1]); artifact = out/"benchmark_fixture"
artifact.mkdir(parents=True)
(artifact/"aggregate_summary.json").write_text("{}")
(artifact/"benchmark_rows.csv").write_text("test_only\\n1\\n")
if phase == "formal_cache_policy":
    replay = Path(sys.argv[sys.argv.index("--request-replay-path")+1]); replay.write_text("{}")
elif phase not in {"formal_controller"}:
    (artifact/"support_provenance.json").write_text(json.dumps({"setting_id":("wrong" if mode=="provenance" else "one"), "output":str(artifact)}))
    write_child_output_descriptor(Path(sys.argv[sys.argv.index("--cell-output-descriptor-path")+1]),
        cell_id=sys.argv[sys.argv.index("--cell-id")+1], phase=phase, logical_setting_id=("wrong" if mode=="descriptor" else "one"),
        output_root=out, artifact_root=artifact, producer_kind="test_only",
        required_payload=["aggregate_summary.json","benchmark_rows.csv","support_provenance.json"])
if mode == "missing" and phase == "formal_cache_policy":(artifact/"aggregate_summary.json").unlink()
if mode == "corrupt" and phase == "formal_cache_policy":(artifact/"aggregate_summary.json").write_text("{")
''')
    protocol = {"hashes": {"semantic_sha256":"b"*64}, "formal_nullable_metric_aggregation_contract":{"semantic_sha256":"n"*64}}
    context = {"scientific_identity":{"active_formal_bundle_sha256":"bundle"}}
    plans = {}
    for name, key in zip(PHASES[:5], ("capacity_label", "fixture", "ablation_setting_id", "support_setting_id", "scalability_setting_id")):
        argv = [sys.executable, str(child), name, "--fault", mode, "--output-root", str(run/name)]
        if name == "formal_cache_policy": argv += ["--request-replay-path", str(run/name/"request_replay.json")]
        plans[name] = dict(phase=name, commands=[argv], command_hashes=[digest(argv)],
                          matrix_contexts=[{key:"one"}], expected_outputs=[name+"/**/aggregate_summary.json"],
                          infrastructure_retries=1, input_hash=digest(name))
    for name in PHASES[5:]:
        plans[name]=dict(phase=name,commands=[],command_hashes=[],matrix_contexts=[],expected_outputs=[],infrastructure_retries=1,input_hash=digest(name))
    proposal = dict(run_id=run.name, run_root=str(run), ledgers=prefixes)
    executor_identity = {"test_only": True, "implementation_file": file_hash(EXECUTOR/"scripts/continuation_executor/execution.py")}
    contract = dict(version="1.0.0", domain="synthetic", proposal_sha256=digest(proposal), proposal_file_sha256="test",
        executor_identity_sha256=digest(executor_identity), run_id=run.name, run_root=str(run), phases=list(PHASES),
        holdout_capability=False, prefixes=prefixes, immutable_files=[dict(path=str(child),sha256=file_hash(child),size_bytes=child.stat().st_size)], fixture_root=str(fixture),
        expires_at=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(), revocation_id="fixture",
        command_plan_sha256=digest(plans),recovery_owner_sha256=None, recovery_quiescence=None, coordination_root=str(fixture/".continuation_locks"))
    validate_contract(contract, proposal, executor_identity)
    authority = dict(domain="synthetic", fixture_root=str(fixture), signer_id="test", key_hex=os.urandom(32).hex(), revoked_ids=[])
    evidence = {k:dict(state="test_only",reference_sha256=digest("test")) for k in ("launch_approval","release_attestation","continuation_approval")}
    message = approval_message(contract, evidence)
    approval = dict(message=message, signer_id="test", signature=hmac.new(bytes.fromhex(authority["key_hex"]),canonical(message),hashlib.sha256).hexdigest())
    subject = SimpleNamespace(root=old,run_root=run,native=native,contract=contract,plans=plans,protocol=protocol,context=context,
         registry_sha256="registry",cell_ledger=cells,phase_runner=phase,
         environment=SimpleNamespace(child_environment=dict(os.environ)))
    dispatches = []
    def audit(event,args):
        if event == "subprocess.Popen":
            argv=args[1]
            if argv and argv[0] == sys.executable:
                assert argv[1] == str(child)
                dispatches.append(list(argv))
    sys.addaudithook(audit)
    results=[]
    error=None
    def authorize():
        now=None
        if (fixture/"revoke.json").exists():
            if mode=="revoke_during_phase":authority["revoked_ids"]=[contract["revocation_id"]]
            else:now=datetime.now(timezone.utc)+timedelta(hours=2)
        return verify_approval(contract,approval,fixture_authority=authority,now=now)
    scope=FixtureScope(str(fixture),str(run),str(old),sys.executable)
    if mode in {"truncation", "fork", "out_of_order", "cross_ledger", "immutable_payload"}:
        if mode=="truncation":
            path=run/"phase_state.jsonl";path.write_bytes(path.read_bytes()[:-1])
        elif mode=="immutable_payload":
            (run/"prior/train/payload.json").write_text('{"test_only":"corrupt"}')
        elif mode in {"fork", "out_of_order"}:
            rows=phase.records();row=dict(rows[0])
            target=PHASES[1] if mode=="out_of_order" else PHASES[0]
            row.update(phase=target,input_hash=plans[target]["input_hash"],commands=plans[target]["commands"],
                command_identity=digest(plans[target]["commands"]),sequence_number=len(rows)+1,
                previous_record_hash="0"*64 if mode=="fork" else rows[-1]["current_record_hash"])
            row["current_record_hash"]=native["phase"]._record_hash(row)
            with (run/"phase_state.jsonl").open("a") as stream:stream.write(json.dumps(row)+"\n")
        else:
            rows=cells.records();row=dict(rows[0]);target=PHASES[0]
            row.update(phase=target,sequence_number=len(rows)+1,previous_ledger_hash=rows[-1]["current_ledger_hash"])
            row["current_ledger_hash"]=native["cell"]._record_hash(row)
            with (run/"cell_state.jsonl").open("a") as stream:stream.write(json.dumps(row)+"\n")
        before={str(p):file_hash(p) for p in fixture.rglob("*") if p.is_file()}
        try:execute_phase(subject,PHASES[0],authorize,executor_identity,scope=scope)
        except Exception as exc:error=str(exc)
        else:raise AssertionError("ledger/payload corruption accepted")
        assert not dispatches
        assert before=={str(p):file_hash(p) for p in fixture.rglob("*") if p.is_file()}
        print(json.dumps(dict(status="pass",case=mode,error=error,synthetic_child_dispatch_count=0,rejected_before_write=True)))
        return
    if mode in {"revoke_during_phase","expire_during_phase"}:
        result=execute_phase(subject,PHASES[0],authorize,executor_identity,scope=scope)
        assert result["phase"]==PHASES[0]
        before={str(p):file_hash(p) for p in fixture.rglob("*") if p.is_file()}
        try:execute_phase(subject,PHASES[1],authorize,executor_identity,scope=scope)
        except Exception as exc:error=str(exc)
        else:raise AssertionError("revoked/expired grant admitted next phase")
        assert before=={str(p):file_hash(p) for p in fixture.rglob("*") if p.is_file()}
        assert len(dispatches)==1
        print(json.dumps(dict(status="pass",case=mode,error=error,synthetic_child_dispatch_count=1,
            prefix_unchanged=all(hashlib.sha256(Path(a["path"]).read_bytes()[:a["byte_count"]]).hexdigest()==a["prefix_sha256"] for a in prefixes),admitted_phase_completed=True)))
        return
    if mode=="utc_adjustment":
        ticks=iter([datetime(2030,1,1,tzinfo=timezone.utc),datetime(2020,1,1,tzinfo=timezone.utc)])
        phase._utc_clock=lambda:next(ticks,datetime(2020,1,1,tzinfo=timezone.utc))
    start=0
    if mode in {"publication_crash", "candidate_crash"}:
        pid=os.fork()
        if pid==0:
            target=cells if mode=="publication_crash" else phase
            original_append=target._append
            def crash_at_boundary(row):
                if mode=="publication_crash" and row["status"]=="committed":
                    os._exit(17)
                result=original_append(row)
                if mode=="candidate_crash" and row["status"]=="completion_candidate":
                    os._exit(17)
                return result
            target._append=crash_at_boundary
            execute_phase(subject,PHASES[0],authorize,executor_identity,scope=scope)
            os._exit(99)
        _, status=os.waitpid(pid,0)
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status)==17
        from continuation_executor.locking import writer_lock_path
        lock=writer_lock_path(run)
        inode=lock.stat().st_ino
        previous=json.loads(lock.read_text())
        owner_hash=digest(previous)
        contract["recovery_owner_sha256"]=owner_hash
        contract["recovery_quiescence"]={"owner_sha256":owner_hash,"state":"test_only",
            "no_live_descendants":True,"reference_sha256":digest({"waitpid":pid,"status":status})}
        validate_contract(contract,proposal,executor_identity)
        message=approval_message(contract,evidence)
        approval.update(message=message,signature=hmac.new(bytes.fromhex(authority["key_hex"]),canonical(message),hashlib.sha256).hexdigest())
        results.append(execute_phase(subject,PHASES[0],authorize,executor_identity,scope=scope,
            finalize_only=mode=="candidate_crash"))
        assert lock.stat().st_ino==inode
        assert len((fixture/"invocations.jsonl").read_text().splitlines())==1
        start=1
    for name in PHASES[start:5]:
        try:
            results.append(execute_phase(subject,name,authorize,executor_identity,scope=scope))
        except Exception as exc:
            if mode not in {"terminal","missing","corrupt","descriptor","provenance"}:raise
            error=str(exc)
            count=len(dispatches)
            try:execute_phase(subject,name,authorize,executor_identity,scope=scope)
            except Exception:pass
            else:raise AssertionError("terminal failure was resumed")
            assert len(dispatches)==count
            break
    if mode in {"terminal","missing","corrupt","descriptor","provenance"}:assert error
    else:
        count=len(dispatches)
        execute_phase(subject,PHASES[4],authorize,executor_identity,scope=scope)
        assert len(dispatches)==count, "completed restart dispatched child"
        assert count==(6 if mode=="exit75" else (4 if start else 5))
        if mode=="exit75":
            records=cells.records()
            failed=[r for r in records if r["status"]=="failed_retryable"]
            assert len(failed)==1
            retries=[r for r in records if r.get("cell_id")==failed[0]["cell_id"]]
            assert len({r["command_hash"] for r in retries})==1
    if mode=="utc_adjustment":
        terminal=[r for r in phase.records() if r["phase"]==PHASES[0] and r["status"]=="completed"][0]
        assert terminal["duration_authority"]=="monotonic_clock" and terminal["wall_clock_adjustment_seconds"]<0
    if mode=="duplicate_committed":
        rows=cells.records();row=dict([r for r in rows if r["status"]=="committed"][-1])
        row.update(sequence_number=len(rows)+1,previous_ledger_hash=rows[-1]["current_ledger_hash"])
        row["current_ledger_hash"]=native["cell"]._record_hash(row)
        with (run/"cell_state.jsonl").open("a") as stream:stream.write(json.dumps(row)+"\n")
        try:execute_phase(subject,PHASES[4],authorize,executor_identity,scope=scope)
        except Exception as exc:error=str(exc)
        else:raise AssertionError("duplicate committed ID accepted")
    if mode in {"gate_missing", "gate_false"}:
        for name in PHASES[5:7]:
            phase.run_phase(name,commands=[],input_hash=plans[name]["input_hash"],expected_outputs=[])
        if mode=="gate_false":(run/"formal_gate.json").write_text('{"passed":false}')
        count=len(dispatches)
        try:execute_phase(subject,PHASES[7],authorize,executor_identity,scope=scope)
        except Exception as exc:error=str(exc)
        else:raise AssertionError("invalid gate allowed completion")
        assert len(dispatches)==count
        assert not any(r["phase"]==PHASES[7] for r in phase.records())
    after = [hashlib.sha256(Path(a["path"]).read_bytes()[:a["byte_count"]]).hexdigest()==a["prefix_sha256"] for a in prefixes]
    assert all(after)
    print(json.dumps(dict(status="pass",case=mode,error=error,synthetic_child_dispatch_count=len((fixture/"invocations.jsonl").read_text().splitlines()),prefix_unchanged=all(after),
                         completed=[r["phase"] for r in results],origins=loaded_origins(old))))


if __name__ == "__main__": main()
