"""Eight-stage, test-only continuation acceptance under a kernel sandbox."""
from __future__ import annotations
import argparse
import hashlib
import hmac
import importlib.util
import json
import os
from pathlib import Path
import secrets
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCIENCE = Path('/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847')
COMMIT = 'a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d'


def external(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def anchor(path, kind):
    data = path.read_bytes(); rows = [json.loads(line) for line in data.splitlines()]
    key = 'current_record_hash' if kind == 'phase' else 'current_ledger_hash'
    return {'path':str(path),'kind':kind,'record_count':len(rows),'byte_count':len(data),
        'prefix_sha256':hashlib.sha256(data).hexdigest(),'terminal_hash':rows[-1][key],
        'run_identity_fingerprint':rows[-1]['run_identity_fingerprint']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-root', type=Path, required=True)
    parser.add_argument('--lease-fault',choices=['none','expire','revoke'],default='none')
    args = parser.parse_args()
    fixture = args.fixture_root
    security = external('continuation_executor_security')
    security.validate_fixture_root(fixture)
    if not fixture.name.startswith('synthetic_') or fixture.exists():
        raise ValueError('fresh synthetic fixture required')
    fixture.mkdir()
    kernel = external('continuation_fixture_sandbox').verify_kernel_boundary(fixture)
    implementation_commit = security.git(ROOT, 'rev-parse', 'HEAD')
    implementation_files = [{'path': str(path.relative_to(ROOT)), 'sha256':security.file_hash(path)}
        for path in sorted((ROOT / 'scripts').glob('*continuation*.py'))]
    executor_identity = {'version':'1.0.0','implementation_commit':implementation_commit,'files':implementation_files}
    adapter = external('continuation_legacy_adapter')
    runner, source = adapter.load_scientific_modules(SCIENCE, security.file_hash(
        SCIENCE / 'scripts/run_typed_model_cache_formal_protocol.py'), expected_commit=COMMIT)
    monitor = external('continuation_fixture_monitor').FixtureMonitor(fixture_root=fixture,
        scientific_root=SCIENCE, python_executable=sys.executable, implementation_root=ROOT,
        forbidden_roots=['/Users/howen/Projects/PPO_MEC/artifacts/experiments',
                         '/Users/howen/Projects/PPO_MEC/data', SCIENCE/'data', SCIENCE/'artifacts']).install()
    print(json.dumps({'progress':'prepare_test_only_checkpoint_prefix'}), flush=True)
    prepared = external('continuation_fixture_builder').build_fixture(
        fixture, SCIENCE, ROOT, runner, source, adapter,
        implementation_commit=implementation_commit, scientific_commit=COMMIT)
    run = prepared['run_root']; protocol = prepared['protocol']; context = prepared['context_payload']
    binding = prepared['execution_binding']; bundle = prepared['bundle']; environment = prepared['environment']
    anchors = [anchor(run / (kind + '_state.jsonl'), kind) for kind in ['phase','cell']]
    prefix = {row['kind']:Path(row['path']).read_bytes() for row in anchors}
    fixture_proposal = {'test_only':True,'run_id':run.name,'run_root':str(run),'anchors':anchors,
                        'phase_plan_sha256':prepared['command_report']['command_matrix_sha256']}
    contract = {'version':'1.0.0','kind':'fixed_commit_continuation_execution_contract',
        'proposal_sha256':security.digest(fixture_proposal),'proposal_file_sha256':security.digest(fixture_proposal),
        'executor_identity_sha256':security.digest(executor_identity),'run_id':run.name,'run_root':str(run),
        'scientific_worktree':str(SCIENCE),'scientific_commit':COMMIT,'phases':list(adapter.PHASES),
        'write_scope':[str(run)],'ledger_anchors':anchors,'immutable_payload_inventory_sha256':security.digest(anchors),
        'origin_evidence':{'launch_approval':{'status':'unavailable'},'release_attestation':{'status':'unavailable'}},
        'holdout_capability':False,'approval_domain':'synthetic-only'}
    security.validate_execution_contract(contract)
    trust = security.FixtureTrust(fixture, secrets.token_bytes(32), security.digest(executor_identity))
    trust.validate_contract(contract)
    body = security.approval_body(contract, approval_id='synthetic_'+secrets.token_hex(16),
        issued_at=time.time()-1, expires_at=time.time()+3600, domain='synthetic-only')
    approval = {'body':body,'key_id':'test-only',
        'signature':hmac.new(trust.key,security.canonical(body),hashlib.sha256).hexdigest()}
    external('continuation_fixture_authorization').issue(fixture,run,ROOT,parent_approval_sha256=security.digest(approval))
    clock_override=None; revoked_ids=[]; lease_triggered=False
    def admit(phase=adapter.PHASES[0]):
        return security.verify_approval(contract, approval, phase=phase, domain='synthetic-only',
            verify_signature=trust.verify, revoked_ids=revoked_ids, now=clock_override)
    def registry():
        return runner.load_generated_checkpoint_registry(run / 'generated_checkpoint_resource_registry.json',
            run_root=run,expected_run_id=run.name,
            static_registry_semantic_sha256=protocol['portable_resource_identity_contract']['resource_registry_semantic_sha256'],
            protocol_semantic_sha256=protocol['hashes']['semantic_sha256'],protocol_full_sha256=protocol['hashes']['full_sha256'],
            active_formal_bundle_sha256=bundle['active_formal_bundle_sha256'],execution_commit=COMMIT,
            resolved_execution_context_sha256=context['context_sha256'],formal_training_execution_binding_sha256=binding['binding_full_sha256'])[1]
    from src.evaluators.formal_cell_transaction import artifact_inventory, validate_cell_ledger
    from src.evaluators.formal_phase_transaction import validate_phase_ledger_v3
    ledger_check = external('continuation_ledger_validation')
    expected_cells = {phase:{runner.stable_cell_id(phase,row) for row in
        protocol['execution_contract']['command_templates'][phase]['matrix_contexts']} for phase in adapter.PHASES[:5]}
    def reconcile():
        return ledger_check.validate_successors(anchors,root=run,phase_validator=validate_phase_ledger_v3,
            cell_validator=validate_cell_ledger,artifact_inventory=artifact_inventory,expected_cells_by_phase=expected_cells)
    # Validate every active fixture command reference before any continuation lock.
    for phase in adapter.PHASES[:-1]:
        plan = runner.expand_command_plan(protocol['execution_contract']['command_templates'][phase], context['resolved_expansion_context'])
        for command in plan['commands']:
            # Interpreter and the hash-bound external child are code, not inputs.
            for index, value in enumerate(command[4:], 4):
                if value == str(SCIENCE):
                    if command[index-1] != '--scientific-root': raise ValueError('unexpected science root reference')
                    continue
                trust.validate_references(value)
    child_observations=[]
    def observe_result(phase,result):
        nonlocal clock_override,lease_triggered
        for line in result.stdout.splitlines():
            try:payload=json.loads(line)
            except ValueError:continue
            if not isinstance(payload,dict):continue
            report=payload.get('continuation_fixture_monitor',payload)
            if not isinstance(report,dict) or 'counts' not in report or 'fixture_root' not in report:continue
            if report['fixture_root']!=str(fixture):raise ValueError('cross-fixture monitor report')
            child_observations.append({'phase':phase,'monitor':report,
                'dispatch_authority':payload.get('dispatch_authority'),
                'scientific_modules':payload.get('scientific_modules',[])})
            if args.lease_fault!='none' and not lease_triggered and phase==adapter.PHASES[0]:
                lease_triggered=True
                if args.lease_fault=='expire':clock_override=body['expires_at']
                else:revoked_ids.append(body['approval_id'])
    checks=[]; results=[]; error=None
    try:
        reconcile(); registry()
        with security.single_writer(run,admission=admit,owner_identity=executor_identity):
            for phase in adapter.PHASES:
                admit(phase); checks.append(reconcile()); registry_audit=registry()
                print(json.dumps({'progress':'phase','phase':phase}),flush=True)
                results.append({'phase':phase,**adapter.run_continuation_phase(runner,source,phase=phase,
                    protocol=protocol,context_payload=context,execution_binding=binding,bundle=bundle,
                    environment=environment,generated_registry_audit=registry_audit,
                    admission=lambda phase=phase:admit(phase),observe_result=observe_result)})
                checks.append(reconcile())
    except Exception as exc:
        error = str(exc)
    prefix_unchanged=all(Path(row['path']).read_bytes().startswith(prefix[row['kind']]) for row in anchors)
    total_counts={key:monitor.counts[key]+sum(row['monitor']['counts'][key] for row in child_observations)
        for key in ['synthetic_child_dispatch_count','scientific_rollout_count','real_v16_dispatch_count','real_v16_write_count']}
    expected_lease_denial=(args.lease_fault!='none' and len(results)==1 and lease_triggered and
        isinstance(error,str) and ('expired' if args.lease_fault=='expire' else 'revoked') in error)
    report={'lease_fault':args.lease_fault,'lease_triggered':lease_triggered,'expected_lease_denial':expected_lease_denial,
        'child_observations':child_observations,'aggregate_process_counts':total_counts,
        'nested_monitor_boundary':'analyzer process launch observed by statistics parent; analyzer inherits kernel sandbox',
        'test_only':True,'execution_authorized':False,'performance_evidence':False,
        'scope':'full synthetic continuation; not real qualification or real execution approval',
        'executor_identity':executor_identity,'executor_identity_sha256':security.digest(executor_identity),
        'fixture_contract':contract,'approval_domain':'synthetic-only','kernel_boundary':kernel,
        'results':results,'checks':checks,'error':error,'prefix_unchanged':prefix_unchanged,
        'registry_audit':registry(),'monitor':monitor.report(),'scientific_modules':adapter.loaded_origins(SCIENCE)}
    (fixture / 'phase_chain_report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps({'report':str(fixture/'phase_chain_report.json'),'phase_count':len(results),'error':error,
                      'counts':monitor.counts}),flush=True)
    if error and not expected_lease_denial: raise SystemExit(2)


if __name__ == '__main__':
    main()
