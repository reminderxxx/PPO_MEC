"""Independent continuation CLI. No production approval is bundled or installed."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    'external_continuation_security', ROOT / 'scripts/continuation_executor_security.py')
security = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(security)



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--executor-identity', required=True)
    parser.add_argument('--approval')
    parser.add_argument('--proposal')
    parser.add_argument('--finalize-only', action='store_true')
    parser.add_argument('--phase', required=True)
    args = parser.parse_args(argv)
    lock_attempted = False
    phase_started = False
    try:
        if args.phase not in security.PHASES:
            raise security.AdmissionError('unauthorized phase')
        contract = security.validate_execution_contract(security.strict_json(args.contract))
        identity = security.strict_json(args.executor_identity)
        if security.digest(identity) != contract['executor_identity_sha256']:
            raise security.AdmissionError('executor identity reference drift')
        security.verify_executor_identity(identity, ROOT)
        if contract['approval_domain'] != 'production':
            raise security.AdmissionError('synthetic approval cannot enter production CLI')
        if args.approval is None:
            raise security.AdmissionError('independent continuation approval unavailable')
        approval = security.strict_json(args.approval)
        if not args.proposal:
            raise security.AdmissionError('original proposal reference required')
        def admission():
            security.verify_executor_identity(identity, ROOT)
            current = security.strict_json(args.approval)
            return security.verify_production_admission(contract, current, phase=args.phase)
        admission()
        def external(name):
            spec = importlib.util.spec_from_file_location('external_'+name, ROOT/'scripts'/(name+'.py'))
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            return module
        qualification = external('continuation_qualification')
        qualified = qualification.qualify(args.proposal, contract)
        if args.phase == 'complete_without_holdout':
            qualified['runner'].validate_complete_without_holdout_gate(contract['run_root'],qualified['protocol'])
        lock_attempted = True
        with security.single_writer(contract['run_root'], admission=admission, owner_identity=identity):
            # Reconcile after acquisition: another authorized writer may have
            # advanced the ledger while the read-only qualification was running.
            state = qualification.reconcile(qualified, contract)
            phase_started = True
            result = external('continuation_legacy_adapter').run_continuation_phase(
                qualified['runner'], qualified['source'], phase=args.phase,
                protocol=qualified['protocol'], context_payload=qualified['context'],
                execution_binding=qualified['binding'], bundle=qualified['bundle'],
                environment=qualified['environment'], generated_registry_audit=qualified['registry_audit'],
                admission=admission, finalize_only=args.finalize_only)
        print(json.dumps({'version':security.VERSION,'status':'phase_transaction_completed',
            'execution_authorized':True,'phase':args.phase,'result':result,
            'locked_start_reconciliation':state},allow_nan=False,default=str))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({'version': security.VERSION, 'status': 'denied',
                          'reason': str(exc), 'execution_authorized': False,
                          'lock_open_attempted': lock_attempted, 'phase_transaction_started': phase_started,
                          'lock_created': False if not lock_attempted else None,
                          'dispatch_started': False if not phase_started else None}, allow_nan=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
