"""End-to-end test-only continuation, with actual native statistics and gate."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from . import PHASES
from .authorization import approval_message, validate_contract, verify_approval, load_fixture_authorization
from .execution import execute_phase
from .fixture_inputs import build_core, build_checkpoints, load_prepared_core, write
from .identity import canonical, digest, file_hash, read_json
from .isolation import FixtureScope
from .monitoring import Monitor
from .planning import phase_plan
from .scientific import loaded_origins


def run_chain(fixture, native, executor_identity, *, prepared=False):
    fixture = Path(fixture)
    monitor = Monitor(fixture)
    print("synthetic: preparing core", file=sys.stderr, flush=True)
    if prepared:
        core=load_prepared_core(fixture,native)
        inputs_report={"status":"prepared_input_revalidation", "registry":core["registry_audit"]}
    else:
        core = build_core(fixture, native)
        inputs_report = build_checkpoints(core, native)
    print("synthetic: checkpoint/registry inputs ready", file=sys.stderr, flush=True)
    from .consumer_entry import check_benchmark_main
    consumer_reports=[]
    for case in ("valid", "missing_binding", "missing_nullable", "protocol", "bundle", "binding", "context", "git", "runtime", "sha", "window", "agent", "seed", "capacity"):
        print("synthetic consumer: "+case, file=sys.stderr, flush=True)
        consumer_reports.append(check_benchmark_main(core,monitor,case))
    from .checkpoints import check_frozen_checkpoints
    checkpoint_report=check_frozen_checkpoints(core["source"],core["run"],core["protocol"],core["context"],core["binding"],
        runtime_paths={c:core["inputs"]/("runtime_"+c+".yaml") for c in ("constrained_288mb","medium_576mb","relaxed_864mb")})
    root, protocol, context = core["run"],core["protocol"],core["context"]
    launcher=fixture/"payload_launcher.py"
    executor_scripts=Path(__file__).resolve().parents[1]
    hooks=fixture/"instrumentation"
    hooks.mkdir(exist_ok=True)
    events=fixture/"monitor_events"
    events.mkdir(exist_ok=True)
    (hooks/"sitecustomize.py").write_text(
        "import sys,atexit,json,os,uuid\nfrom pathlib import Path\nsys.path.insert(0,"+repr(str(executor_scripts))+")\n"
        "from continuation_executor.monitoring import Monitor\n"
        "monitor=Monitor("+repr(str(fixture))+")\n"
        "def finish():\n"
        "    sys.setprofile(None)\n"
        "    report=monitor.report()\n"
        "    report['interpreter']=sys.executable\n"
        "    report['entry_argv']=[sys.executable,*sys.argv]\n"
        "    from continuation_executor.scientific import loaded_origins\n"
        "    report['origins']=loaded_origins(Path("+repr(str(core["source"]))+"))\n"
        "    report['sys_path']=sys.path\n"
        "    report['cwd']=os.getcwd()\n"
        "    Path("+repr(str(events))+",str(os.getpid())+'.json').write_text(json.dumps(report))\n"
        "atexit.register(finish)\n")
    launcher.write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0,"+repr(str(executor_scripts))+ ")\n"
        "from continuation_executor.synthetic_payload import main\nmain()\n")
    context_hash=file_hash(root/"resolved_execution_context.json")
    registry_sha=core["registry_audit"]["registry_canonical_sha256"]
    plans={}
    mapping=[]
    # The synthetic adapter changes only the test dispatch implementation. The
    # scientific argv and every coordinate are retained as independent evidence.
    for phase in PHASES:
        frozen = phase_plan(phase,protocol,context,core["binding"],context_hash,registry_sha,native["execution"].expand_command_plan)
        synthetic_protocol=deepcopy(protocol)
        if phase in PHASES[:5]:
            source=synthetic_protocol["execution_contract"]["command_templates"][phase]
            argv=source["argv"]
            source["argv"]=[argv[0],str(launcher),str(fixture),phase,
                "{capacity_label}" if phase=="formal_cache_policy" else
                "{capacity_label}" if phase=="formal_controller" else
                "{ablation_setting_id}" if phase=="formal_ablation" else
                "{support_setting_id}" if phase=="formal_support" else "{scalability_setting_id}",
                "--output-root", "{formal_cache_policy_output_root}" if phase=="formal_cache_policy" else
                "{formal_controller_output_root}" if phase=="formal_controller" else
                "{formal_ablation_output_root}" if phase=="formal_ablation" else
                "{formal_support_output_root}" if phase=="formal_support" else "{formal_scalability_output_root}"]
            if phase=="formal_cache_policy": source["argv"] += ["--request-replay-path","{request_replay_path}"]
        plan=phase_plan(phase,synthetic_protocol,context,core["binding"],context_hash,registry_sha,native["execution"].expand_command_plan)
        # Protocol semantic identity stays the original synthetic fixture identity;
        # no modified protocol file is installed or presented as scientific code.
        plans[phase]=plan
        mapping.append({"phase":phase,"scientific_commands":frozen["commands"],"synthetic_commands":plan["commands"],
                        "coordinates":plan["matrix_contexts"],"expected_outputs":plan["expected_outputs"]})
    prefixes=[]
    for kind,key in (("phase","current_record_hash"),("cell","current_ledger_hash")):
        path=root/(kind+"_state.jsonl");data=path.read_bytes();rows=[__import__('json').loads(line) for line in data.splitlines()]
        prefixes.append(dict(path=str(path),kind=kind,record_count=len(rows),byte_count=len(data),
            prefix_sha256=hashlib.sha256(data).hexdigest(),terminal_hash=rows[-1][key],run_identity_fingerprint=rows[0]["run_identity_fingerprint"]))
    proposal=dict(run_id=root.name,run_root=str(root),ledgers=prefixes)
    contract=dict(version="1.0.0",domain="synthetic",proposal_sha256=digest(proposal),proposal_file_sha256=hashlib.sha256(canonical(proposal)+b"\n").hexdigest(),
        executor_identity_sha256=digest(executor_identity),run_id=root.name,run_root=str(root),phases=list(PHASES),
        holdout_capability=False,prefixes=prefixes,immutable_files=[dict(path=str(path),sha256=file_hash(path),size_bytes=path.stat().st_size)
            for path in (launcher,hooks/"sitecustomize.py",root/"generated_checkpoint_resource_registry.json",
                root/"checkpoint_freeze.json",root/"dev_selection.json",root/"resolved_execution_context.json",root/"formal_training_execution_binding.json")],fixture_root=str(fixture),
        expires_at=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat(),revocation_id="synthetic-chain",
        command_plan_sha256=digest(plans),recovery_owner_sha256=None,recovery_quiescence=None,coordination_root=str(fixture/".continuation_locks"))
    validate_contract(contract,proposal,executor_identity)
    authority=dict(domain="synthetic",fixture_root=str(fixture),signer_id="fixture-only",key_hex=os.urandom(32).hex(),revoked_ids=[])
    evidence={key:dict(state="test_only",reference_sha256=digest(key)) for key in ("launch_approval","release_attestation","continuation_approval")}
    message=approval_message(contract,evidence)
    approval=dict(message=message,signer_id="fixture-only",signature=hmac.new(bytes.fromhex(authority["key_hex"]),canonical(message),hashlib.sha256).hexdigest())
    for filename,payload in (("proposal_test_only.json",proposal),("contract_test_only.json",contract),
            ("approval_test_only.json",approval),("authority_test_only.json",authority),("executor_identity.json",executor_identity),("command_plan_test_only.json",plans)):
        write(fixture/filename,payload)
    stored=load_fixture_authorization(fixture,executor_identity)
    contract,approval,authority,plans=(stored[key] for key in ("contract","approval","authority","plans"))
    subject=SimpleNamespace(root=core["source"],run_root=root,native=native,contract=contract,plans=plans,
        protocol=protocol,context=context,registry_sha256=registry_sha,cell_ledger=core["cells"],phase_runner=core["phase_runner"],
        environment=SimpleNamespace(child_environment=dict(os.environ, PYTHONPATH=str(hooks)+os.pathsep+str(core["source"]))))
    # Rehearsal is explicitly visible to the old integrity/gate consumer. Counts
    # remain the original full matrix, not a reduced empty expected-count map.
    marker=dict(test_only=True,expected_counts={
        "committed_training_cells":150,"candidate_checkpoints":1200,"latest_checkpoints":150,
        "dev_candidate_evaluations":1200,"selections":150,"frozen_checkpoints":150,
        "frozen_checkpoints_by_capacity":{c:50 for c in ("constrained_288mb","medium_576mb","relaxed_864mb")},
        "cache_policy_cells":3,"controller_cells":3,"ablation_settings":2,"support_settings":11,
        "scalability_settings":3,"primary_comparison_rows":84,"formal_outer_window_clusters":12})
    if not (root/"non_formal_rehearsal.json").exists():write(root/"non_formal_rehearsal.json",marker)
    elif read_json(root/"non_formal_rehearsal.json")!=marker:raise ValueError("rehearsal marker drift")
    results=[]
    scope=FixtureScope(str(fixture),str(root),str(core["source"]),sys.executable)
    for phase in PHASES:
        print("synthetic: " + phase, file=sys.stderr, flush=True)
        results.append(execute_phase(subject,phase,lambda:verify_approval(contract,approval,fixture_authority=authority),executor_identity,
            scope=scope))
    gate=read_json(root/"formal_gate.json")
    if gate["passed"] is not True: raise ValueError("synthetic gate did not pass")
    children=[read_json(path) for path in events.glob("*.json")]
    combined=monitor.report()
    combined["child_monitors"]=children
    for field in ("synthetic_child_dispatch_count","scientific_rollout_count","real_v16_dispatch_count","real_v16_write_count"):
        combined[field] += sum(row[field] for row in children)
    from collections import Counter
    all_dispatches=combined["dispatches"]+[command for child in children for command in child["dispatches"]]
    if Counter(tuple(command) for command in all_dispatches)!=Counter(tuple(child["entry_argv"]) for child in children):
        raise ValueError("child dispatch/monitor coverage mismatch")
    if combined["synthetic_child_dispatch_count"]<=0 or any(combined[field] for field in
            ("scientific_rollout_count","real_v16_dispatch_count","real_v16_write_count")):
        raise ValueError("synthetic acceptance execution boundary violated")
    if combined["denied"] or any(child["denied"] for child in children):
        raise ValueError("accepted chain contains denied access")
    return dict(status="pass",serialized_authorization_verified=True,synthetic_proposal_file_sha256=file_hash(fixture/"proposal_test_only.json"),approval_contract_sha256=digest(contract),inputs=inputs_report,consumer_cases=consumer_reports,checkpoint_audit=checkpoint_report,phase_results=results,command_mapping=mapping,
        gate=gate,monitor=combined,actual_parent_environment=native["process_environment"],origins=loaded_origins(core["source"]),
        prefix_unchanged=all(hashlib.sha256(Path(a["path"]).read_bytes()[:a["byte_count"]]).hexdigest()==a["prefix_sha256"] for a in prefixes))
