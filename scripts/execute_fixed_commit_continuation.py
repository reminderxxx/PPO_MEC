"""Separate fixed-commit continuation CLI. A's proposal CLI stays read-only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from continuation_executor import PHASES
from continuation_executor.authorization import validate_contract
from continuation_executor.identity import (
    ContinuationError, file_hash, read_json, verify_executor, validate_a_proposal,
)
from continuation_executor.startup import (
    DEFAULT_WAIT_SECONDS, StartupInterrupted, StartupTimeout, event,
    handoff_material, rejection_code, run_startup_operation, validate_wait_seconds,
)
from continuation_executor.startup_binding import ProductionStartupBinding


def _emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def main(argv=None, *, _startup_binding=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--executor-identity", required=True)
    parser.add_argument("--approval")
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--check", choices=("compatibility", "qualification", "execute"), default="qualification")
    parser.add_argument("--finalize-phase-only", action="store_true")
    parser.add_argument("--startup-wait-seconds", default=DEFAULT_WAIT_SECONDS)
    args = parser.parse_args(argv)
    context = None
    final_authorization = None
    try:
        proposal, contract, identity = (read_json(p) for p in (
            args.proposal, args.contract, args.executor_identity))
        validate_contract(contract, proposal, identity)
        if file_hash(args.proposal) != contract["proposal_file_sha256"]:
            raise ContinuationError("proposal byte identity drift")
        if args.check == "compatibility":
            if contract["domain"] != "production":
                raise ContinuationError("production entry refuses synthetic trust")
            verified_identity = verify_executor(identity, Path(__file__).resolve().parents[1])
            validate_a_proposal(proposal)
            from continuation_executor.scientific import QualifiedRun
            qualified = QualifiedRun(proposal, contract)
            report = {"status": "read_only_compatible", "execution_authorized": False,
                      "executor": verified_identity, "reconciliation": qualified.reconciliation,
                      "plans": qualified.plans, "origins": qualified.origins,
                      "active_resource_count":len(qualified.bundle["resource_ids"]),
                      "registry_audit": qualified.registry_audit, "checkpoint_audit": qualified.checkpoints,
                      "environment": qualified.environment.runtime_audit,
                      "actual_parent_environment": qualified.native["process_environment"],
                      "origin_evidence": proposal["origin_evidence"]}
            _emit(report)
            return 0
        if not args.approval:
            raise ContinuationError("independent continuation approval unavailable")
        approval = read_json(args.approval)
        # Static identity and A structure precede context creation and startup IO.
        verified_identity = verify_executor(identity, Path(__file__).resolve().parents[1])
        validate_a_proposal(proposal)
        binding = _startup_binding or ProductionStartupBinding()
        if binding.domain != contract["domain"]:
            raise ContinuationError("startup binding domain mismatch")
        wait_seconds = validate_wait_seconds(args.startup_wait_seconds)
        context = binding.context(contract)

        def verify(now):
            nonlocal final_authorization
            final_authorization = binding.verify(contract, approval, context, now)
            return final_authorization

        def operate(authorization):
            del authorization
            qualified = binding.qualify(proposal, contract)
            report = {"status": "qualified", "authorization": final_authorization,
                      "executor": verified_identity, **binding.qualification_report(qualified)}
            if args.check == "execute":
                report["result"] = binding.execute(qualified, args.phase,
                    lambda: binding.verify(contract, read_json(args.approval), context,
                                            datetime.now(timezone.utc)),
                    identity, finalize_only=args.finalize_phase_only)
            return report

        authorization, report, material = run_startup_operation(
            context, verify, operate, timeout_seconds=wait_seconds, emit=_emit)
        _emit(event("terminal", "completed" if args.check == "execute" else "qualified",
                    result=report, handoff_material=material,
                    execution_authorized=authorization["real_execution_authorized"]))
    except (ValueError, OSError, KeyError, TypeError, KeyboardInterrupt) as exc:
        status = "timed_out" if isinstance(exc, StartupTimeout) else (
            "interrupted" if isinstance(exc, (StartupInterrupted, KeyboardInterrupt)) else "rejected")
        material = handoff_material(context, status) if context is not None else None
        if material is not None:
            _emit(event("handoff", "verification_required", material=material))
        _emit(event("terminal", status, reason_code=rejection_code(exc), reason=str(exc),
                    handoff_material=material, execution_authorized=False))
        if isinstance(exc, StartupInterrupted):
            return 128 + exc.signum
        if isinstance(exc, KeyboardInterrupt):
            return 130
        if isinstance(exc, StartupTimeout):
            return 124
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
