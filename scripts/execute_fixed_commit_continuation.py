"""Separate fixed-commit continuation CLI. A's proposal CLI stays read-only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from continuation_executor import PHASES
from continuation_executor.authorization import validate_contract, verify_approval
from continuation_executor.identity import (
    ContinuationError, file_hash, read_json, verify_executor, validate_a_proposal,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--executor-identity", required=True)
    parser.add_argument("--approval")
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--check", choices=("compatibility", "qualification", "execute"), default="qualification")
    parser.add_argument("--finalize-phase-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        proposal, contract, identity = (read_json(p) for p in (
            args.proposal, args.contract, args.executor_identity))
        validate_contract(contract, proposal, identity)
        if file_hash(args.proposal) != contract["proposal_file_sha256"]:
            raise ContinuationError("proposal byte identity drift")
        if contract["domain"] != "production":
            raise ContinuationError("production entry refuses synthetic trust")
        if args.check == "compatibility":
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
            print(json.dumps(report, ensure_ascii=False, allow_nan=False))
            return 0
        if not args.approval:
            raise ContinuationError("independent continuation approval unavailable")
        approval = read_json(args.approval)
        # No imports from science, no lock, and no run write before authorization.
        authorization = verify_approval(contract, approval)
        verified_identity = verify_executor(identity, Path(__file__).resolve().parents[1])
        validate_a_proposal(proposal)
        from continuation_executor.scientific import QualifiedRun
        qualified = QualifiedRun(proposal, contract)
        report = {"status": "qualified", "authorization": authorization,
                  "executor": verified_identity, "reconciliation": qualified.reconciliation}
        if args.check == "execute":
            from continuation_executor.execution import execute_phase
            report["result"] = execute_phase(qualified, args.phase,
                lambda: verify_approval(contract, read_json(args.approval)), identity,
                finalize_only=args.finalize_phase_only)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        report = {"status": "rejected", "reason": str(exc), "execution_authorized": False}
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
