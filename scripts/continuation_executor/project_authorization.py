"""One-run local project authorization explicitly approved on 2026-09-10.

This is owner-directed research execution, NOT cryptographic authentication of
historical release or protection against a malicious local administrator. The
separate production-signature path is unchanged. Scientific qualification,
transactions and failure rules are shared, not substituted.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from . import PHASES
from .authorization import timestamp, validate_contract
from .identity import ContinuationError, absolute_path, digest, read_json, verify_file

RUN_ID = "typed_model_cache_formal_20260906_152847_g14c_v16"
RUN_ROOT = "/Users/howen/Projects/PPO_MEC/artifacts/experiments/typed_model_cache_formal/" + RUN_ID
SCIENTIFIC_COMMIT = "a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d"
SCIENTIFIC_ROOT = "/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847"
MODE = "one_run_local_project_authorization_v1"
OWNER_DECISION = {
    "recorded_date": "2026-09-10",
    "source": "current project-owner conversation; not a historical signature",
    "proposal": "是否允许我们把过重的接续授权机制，正式收敛成“你批准范围＋独立核验＋固定输入＋完整追加记录”的一次性实验授权？",
    "user_response": "允许",
    "run_id": RUN_ID,
    "scientific_commit": SCIENTIFIC_COMMIT,
    "phases": list(PHASES),
    "holdout_capability": False,
    "historical_release_attestation": "unavailable_not_backdated",
    "scientific_changes_allowed": False,
    "expires_at": "2026-09-30T23:59:59+08:00",
}


def stop_path(contract):
    return str(absolute_path(contract["coordination_root"]) /
               (digest(RUN_ROOT) + ".project_stop"))


def validate_project_scope(contract, proposal, identity):
    """Validate scope without importing science, writing files or approving IO."""
    validate_contract(contract, proposal, identity)
    if (contract["version"] != "1.0.0" or contract["domain"] != "production"
            or contract["fixture_root"] is not None
            or contract["run_root"] != RUN_ROOT or contract["run_id"] != RUN_ID
            or proposal["execution_commit"] != SCIENTIFIC_COMMIT
            or proposal["worktree_root"] != SCIENTIFIC_ROOT
            or contract["holdout_capability"] is not False
            or contract["recovery_owner_sha256"] is not None
            or contract["recovery_quiescence"] is not None):
        raise ContinuationError("project authorization is limited to exact v16; no recovery")
    anchors = {row["kind"]: row for row in contract["prefixes"]}
    if (len(contract["prefixes"]) != 2 or set(anchors) != {"phase", "cell"}
            or anchors["phase"]["record_count"] != 15
            or anchors["cell"]["record_count"] != 348):
        raise ContinuationError("project authorization requires original freeze boundary")
    # The A proposal includes only immutable source/input/committed evidence;
    # ledger prefixes are handled separately by the unchanged reconciler.
    expected = [{key: row[key] for key in ("path", "sha256", "size_bytes")}
                for row in proposal["evidence"]]
    if contract["immutable_files"] != expected:
        raise ContinuationError("project immutable evidence inventory drift")
    expiry = timestamp(contract["expires_at"])
    if not timestamp("2026-09-10T00:00:00+08:00") < expiry <= timestamp(OWNER_DECISION["expires_at"]):
        raise ContinuationError("project authorization expiry exceeds owner scope")
    if contract["revocation_id"] != MODE + ":" + RUN_ID:
        raise ContinuationError("project cancellation identity drift")


def verify_project_authorization(contract, proposal, identity, grant, *, now=None):
    """Local owner trust + independently recorded review; hashes are integrity only."""
    validate_project_scope(contract, proposal, identity)
    fields = {"version", "mode", "decision_sha256", "contract_sha256",
              "executor_identity_sha256", "independent_review", "stop_file", "issued_at"}
    if (set(grant) != fields or grant["version"] != "1.0.0" or grant["mode"] != MODE
            or grant["decision_sha256"] != digest(OWNER_DECISION)
            or grant["contract_sha256"] != digest(contract)
            or grant["executor_identity_sha256"] != digest(identity)
            or grant["stop_file"] != stop_path(contract)):
        raise ContinuationError("project authorization identity/schema drift")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ContinuationError("project authorization requires timezone-aware time")
    issued = timestamp(grant["issued_at"])
    if issued < timestamp("2026-09-10T00:00:00+08:00") or not issued <= now < timestamp(contract["expires_at"]):
        raise ContinuationError("project authorization future/expired")
    cancellation = absolute_path(grant["stop_file"])
    try:
        cancellation.lstat()
    except FileNotFoundError:
        pass
    else:
        raise ContinuationError("project authorization cancelled by owner stop file")
    review = read_json(verify_file(grant["independent_review"]))
    review_fields = {"version", "status", "reviewer_id", "implementation_agent_id", "reviewed_at",
                     "executor_identity_sha256", "contract_sha256", "decision_sha256",
                     "checks", "evidence"}
    if (set(review) != review_fields or review["version"] != "1.0.0" or review["status"] != "pass"
            or not isinstance(review["reviewer_id"], str) or not review["reviewer_id"].strip()
            or not isinstance(review["implementation_agent_id"], str) or not review["implementation_agent_id"].strip()
            or review["reviewer_id"] == review["implementation_agent_id"]
            or review["executor_identity_sha256"] != digest(identity)
            or review["contract_sha256"] != digest(contract)
            or review["decision_sha256"] != digest(OWNER_DECISION)
            or not timestamp("2026-09-10T00:00:00+08:00") <= timestamp(review["reviewed_at"]) <= issued):
        raise ContinuationError("independent project review missing/mismatched")
    checks = {"code_and_tests", "current_dated_source_verification", "immutable_input_protection",
              "real_read_only_qualification", "frozen_commands", "no_holdout", "no_scientific_changes"}
    if (set(review["checks"]) != checks
            or any(value is not True for value in review["checks"].values())
            or not isinstance(review["evidence"], list) or not review["evidence"]):
        raise ContinuationError("independent project review checks incomplete")
    for row in review["evidence"]:
        verify_file(row)
    return {"approval_verified": True, "domain": "production", "real_execution_authorized": True,
            "authorization_mode": MODE, "contract_sha256": digest(contract),
            "decision_sha256": digest(OWNER_DECISION), "review_sha256": grant["independent_review"]["sha256"],
            "holdout_capability": False, "historical_approval_claim": False}


def run_project_operation(args, proposal, contract, identity, *, emit):
    """Public project branch: validate grant before any science/lock/run IO."""
    from .identity import file_hash
    grant_path = absolute_path(args.project_authorization)
    initial_grant_hash = file_hash(grant_path)

    def authorize():
        if file_hash(grant_path) != initial_grant_hash:
            raise ContinuationError("project authorization changed during operation")
        return verify_project_authorization(contract, proposal, identity, read_json(grant_path))

    authorization = authorize()
    if args.finalize_phase_only:
        raise ContinuationError("cold finalize requires separate owner recovery authorization")
    from .scientific import QualifiedRun
    qualified = QualifiedRun(proposal, contract)
    emit({"event": "qualification", "status": "accepted", "authorization": authorization,
          "reconciliation": qualified.reconciliation, "checkpoint_audit": qualified.checkpoints,
          "origins": qualified.origins})
    result = None
    if args.check == "execute":
        # This grant covers an ordinary forward continuation, not a cold restart
        # of an interrupted phase or a second consumption of a completed phase.
        records = qualified.phase_runner.records()
        latest = {row["phase"]: row["status"] for row in records if row["phase"] in PHASES}
        if any(status != "completed" for status in latest.values()):
            raise ContinuationError("unfinished project phase requires separate recovery authorization")
        remaining = [phase for phase in PHASES if phase not in latest]
        if not remaining or args.phase != remaining[0]:
            raise ContinuationError("project phase must be the next unstarted phase")
        from .execution import execute_phase
        result = execute_phase(qualified, args.phase, authorize, identity)
    emit({"event": "terminal", "status": "completed" if args.check == "execute" else "qualified",
          "authorization_mode": MODE, "real_execution_authorized": True,
          "holdout_capability": False, "phase": args.phase, "result": result})
    return 0
