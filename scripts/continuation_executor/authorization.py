"""Detached approval routing. Production installation remains unavailable.

Test authority is meaningful only inside its independently bounded fixture root.
The public CLI cannot supply or install production trust.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import re
from pathlib import Path

from . import PHASES
from .identity import ContinuationError, absolute_path, canonical, digest, read_json, within


def validate_contract(contract, proposal, executor_identity):
    fields = {"version", "domain", "proposal_sha256", "proposal_file_sha256",
              "executor_identity_sha256", "run_id", "run_root", "phases", "holdout_capability",
              "prefixes", "immutable_files", "fixture_root", "expires_at", "revocation_id",
              "recovery_owner_sha256", "recovery_quiescence", "coordination_root", "command_plan_sha256"}
    if contract.get("version") == "2.0.0":
        fields |= {"scientific_commit", "release_identity", "trust_installation_id"}
    if set(contract) != fields or contract["version"] not in {"1.0.0", "2.0.0"}:
        raise ContinuationError("execution contract schema")
    if contract["version"] == "2.0.0":
        if (not isinstance(contract["revocation_id"], str) or not contract["revocation_id"].strip()
                or contract["scientific_commit"] != "a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d"
                or not isinstance(contract["release_identity"], str) or not contract["release_identity"].strip()
                or not isinstance(contract["trust_installation_id"], str) or not contract["trust_installation_id"].strip()):
            raise ContinuationError("continuation trust scope identity required")
    if contract["domain"] not in {"production", "synthetic"}:
        raise ContinuationError("unknown approval domain")
    if contract["proposal_sha256"] != digest(proposal) or contract["executor_identity_sha256"] != digest(executor_identity):
        raise ContinuationError("proposal/executor identity drift")
    if contract["phases"] != list(PHASES) or contract["holdout_capability"] is not False:
        raise ContinuationError("unauthorized phase/holdout")
    if not isinstance(contract["command_plan_sha256"], str) or not re.fullmatch("[0-9a-f]{64}",contract["command_plan_sha256"]):
        raise ContinuationError("command plan identity required")
    root = absolute_path(contract["run_root"])
    if contract["coordination_root"] != str(root.parent / ".continuation_locks"):
        raise ContinuationError("coordination root must be deterministic for this run")
    if root.name != contract["run_id"] or str(root) != proposal["run_root"] or root.name != proposal["run_id"]:
        raise ContinuationError("cross-run contract")
    if contract["prefixes"] != proposal["ledgers"]:
        raise ContinuationError("approved prefix differs from proposal")
    if contract["domain"] == "synthetic":
        fixture = absolute_path(contract["fixture_root"])
        within(str(root), fixture)
        if root == fixture or not root.name.startswith("synthetic_"):
            raise ContinuationError("synthetic run must be a separate synthetic_* child")
    elif contract["fixture_root"] is not None:
        raise ContinuationError("test authority cannot qualify production")
    recovery = contract["recovery_quiescence"]
    if contract["recovery_owner_sha256"] is None:
        if recovery is not None:
            raise ContinuationError("quiescence evidence without recovery owner")
    elif contract["version"] == "2.0.0":
        if not isinstance(recovery, dict) or not re.fullmatch("[0-9a-f]{64}", str(contract["recovery_owner_sha256"])):
            raise ContinuationError("certified recovery evidence required")
    elif (not isinstance(recovery, dict) or set(recovery) != {
            "owner_sha256", "state", "no_live_descendants", "reference_sha256"}
          or recovery["owner_sha256"] != contract["recovery_owner_sha256"]
          or recovery["state"] != ("test_only" if contract["domain"] == "synthetic" else "independently_verified")
          or recovery["no_live_descendants"] is not True
          or not re.fullmatch("[0-9a-f]{64}", str(recovery["reference_sha256"]))):
        raise ContinuationError("independent recovery quiescence evidence required")
    if not isinstance(contract["immutable_files"], list):
        raise ContinuationError("immutable inventory required")
    return contract


def timestamp(text):
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            raise ValueError("timezone missing")
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError) as exc:
        raise ContinuationError("invalid approval expiry") from exc


def approval_message(contract, evidence):
    return {"version": contract["version"], "domain": contract["domain"],
            "contract_sha256": digest(contract), "evidence": evidence}


def verify_approval(contract, approval, *, fixture_authority=None, test_trust_context=None, now=None):
    """Verify qualification; v2 persists only installed coordination continuity."""
    if contract["domain"] == "production":
        if fixture_authority is not None or test_trust_context is not None:
            raise ContinuationError("test approval cannot become production trust")
        from .production_trust import production_context
        return production_context().verify(contract, approval, now)
    if contract.get("version") == "2.0.0":
        if fixture_authority is not None or test_trust_context is None:
            raise ContinuationError("isolated Ed25519 trust context required")
        if test_trust_context.pin["domain"] != "synthetic":
            raise ContinuationError("test context required")
        return test_trust_context.verify(contract, approval, now)
    if test_trust_context is not None:
        raise ContinuationError("legacy fixture cannot use v2 trust")
    now = now or datetime.now(timezone.utc)
    if now >= timestamp(contract["expires_at"]):
        raise ContinuationError("approval expired")
    if set(approval) != {"message", "signer_id", "signature"}:
        raise ContinuationError("approval evidence schema")
    message = approval["message"]
    if set(message) != {"version", "domain", "contract_sha256", "evidence"}:
        raise ContinuationError("approval message schema")
    if message != approval_message(contract, message["evidence"]):
        raise ContinuationError("approval contract/domain drift")
    evidence = message["evidence"]
    if set(evidence) != {"launch_approval", "release_attestation", "continuation_approval"}:
        raise ContinuationError("three separate evidence states required")
    if fixture_authority is None:
        raise ContinuationError("independent fixture authority required")
    if set(fixture_authority) != {"domain", "fixture_root", "signer_id", "key_hex", "revoked_ids"}:
        raise ContinuationError("fixture authority schema")
    if fixture_authority["domain"] != "synthetic" or fixture_authority["fixture_root"] != contract["fixture_root"]:
        raise ContinuationError("cross-fixture authority")
    if approval["signer_id"] != fixture_authority["signer_id"]:
        raise ContinuationError("fixture signer mismatch")
    if contract["revocation_id"] in fixture_authority["revoked_ids"]:
        raise ContinuationError("approval revoked")
    signature = hmac.new(bytes.fromhex(fixture_authority["key_hex"]), canonical(message), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, approval["signature"]):
        raise ContinuationError("invalid fixture approval signature")
    required_state = "test_only"
    if any(not isinstance(row, dict) or row.get("state") != required_state or not re.fullmatch("[0-9a-f]{64}", str(row.get("reference_sha256"))) for row in evidence.values()):
        raise ContinuationError("independent qualification evidence missing")
    return {"approval_verified": True, "domain": contract["domain"],
            "real_execution_authorized": contract["domain"] == "production",
            "contract_sha256": digest(contract)}


def load_fixture_authorization(fixture_root, expected_executor_identity):
    """Consume the serialized test evidence, including the proposal byte hash."""
    from .identity import file_hash
    root = absolute_path(str(fixture_root))
    filenames = dict(proposal="proposal_test_only.json", contract="contract_test_only.json",
                     approval="approval_test_only.json", authority="authority_test_only.json",
                     identity="executor_identity.json", plans="command_plan_test_only.json")
    paths = {key: within(str(root / name), root) for key, name in filenames.items()}
    values = {key: read_json(path) for key, path in paths.items()}
    contract = values["contract"]
    if contract["domain"] != "synthetic" or contract["fixture_root"] != str(root):
        raise ContinuationError("serialized fixture domain/root mismatch")
    if values["identity"] != expected_executor_identity:
        raise ContinuationError("serialized executor identity drift")
    validate_contract(contract, values["proposal"], values["identity"])
    if file_hash(paths["proposal"]) != contract["proposal_file_sha256"]:
        raise ContinuationError("serialized proposal byte identity drift")
    if digest(values["plans"]) != contract["command_plan_sha256"]:
        raise ContinuationError("serialized command plan drift")
    verify_approval(contract, values["approval"], fixture_authority=values["authority"])
    return values
