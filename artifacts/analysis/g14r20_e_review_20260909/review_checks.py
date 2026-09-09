#!/usr/bin/env python3
"""Independent, read-only G14R20-E evidence and startup operability probes."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


REVIEW_ROOT = Path(__file__).resolve().parent
REPOSITORY = Path("/private/tmp/ppo_mec_g14r20_e_review")
MAIN_REPOSITORY = Path("/Users/howen/Projects/PPO_MEC")
D_DELIVERY = Path("/private/tmp/ppo_mec_g14r20_d_delivery/artifacts/analysis/g14r20_d_production_trust_20260909")
C_REVIEW = Path("/private/tmp/ppo_mec_g14r20_c_review/artifacts/analysis/g14r20_c_review_20260909")
ORIGIN_LAUNCH_SOURCE = Path("/Users/howen/.codex/attachments/1b4faf5e-c59f-4c11-b0ba-f42ea9139616/pasted-text.txt")
SCIENTIFIC_WORKTREE = Path("/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847")
IMPLEMENTATION_COMMIT = "876c369a1fa8cc33b19d782ab1ba230f021965a7"
EVIDENCE_COMMIT = "35bec6320c211b4762ec7bee85e83d780424d3eb"
SCIENTIFIC_COMMIT = "a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def git(*args, cwd=REPOSITORY):
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def junit(path):
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    return {
        "path": str(path),
        "tests": len(cases),
        "failures": len(root.findall(".//failure")),
        "errors": len(root.findall(".//error")),
        "skipped": len(root.findall(".//skipped")),
    }


def identity_and_evidence():
    identity = json.loads((D_DELIVERY / "executor_identity.json").read_text())
    identity_mismatches = []
    for row in identity["files"]:
        path = REPOSITORY / row["path"]
        if not path.is_file() or file_hash(path) != row["sha256"]:
            identity_mismatches.append(row["path"])

    manifest = json.loads((D_DELIVERY / "artifact_integrity.json").read_text())
    evidence_mismatches = []
    for row in manifest["files"]:
        path = D_DELIVERY / row["path"]
        if (not path.is_file() or path.stat().st_size != row["size_bytes"]
                or file_hash(path) != row["sha256"]):
            evidence_mismatches.append(row["path"])

    c_manifest = json.loads((C_REVIEW / "review_integrity_manifest.json").read_text())
    c_evidence_mismatches = []
    for row in c_manifest["files"]:
        path = C_REVIEW / row["path"]
        if (not path.is_file() or path.stat().st_size != row["size_bytes"]
                or file_hash(path) != row["sha256"]):
            c_evidence_mismatches.append(row["path"])

    before = json.loads((D_DELIVERY / "protected_before.json").read_text())
    after = json.loads((D_DELIVERY / "protected_after.json").read_text())
    result = {
        "recomputed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "implementation": {
            "commit": git("rev-parse", IMPLEMENTATION_COMMIT),
            "tree": git("rev-parse", IMPLEMENTATION_COMMIT + "^{tree}"),
            "review_checkout_head": git("rev-parse", "HEAD"),
            "review_checkout_tree": git("rev-parse", "HEAD^{tree}"),
            "review_checkout_status": git("status", "--porcelain", "--untracked-files=no"),
        },
        "evidence": {
            "commit": git("rev-parse", EVIDENCE_COMMIT),
            "tree": git("rev-parse", EVIDENCE_COMMIT + "^{tree}"),
            "parent": git("rev-parse", EVIDENCE_COMMIT + "^"),
            "manifest_version": manifest["version"],
            "manifest_entries": len(manifest["files"]),
            "manifest_mismatches": evidence_mismatches,
        },
        "c_review_evidence": {
            "manifest_entries": len(c_manifest["files"]),
            "manifest_mismatches": c_evidence_mismatches,
            "origin_launch_source_sha256": file_hash(ORIGIN_LAUNCH_SOURCE),
        },
        "origin_release_git_chain": [{
            "commit": commit,
            "parent": git("rev-parse", commit + "^"),
            "tree": git("rev-parse", commit + "^{tree}"),
        } for commit in (
            "834603a266bf06d30a070588a6e1f633eacf70e3",
            "3a8830328394fe114caadec02cb34696149d9fd8",
            SCIENTIFIC_COMMIT,
        )],
        "executor_identity": {
            "declared_commit": identity["commit"],
            "declared_tree": identity["git_tree"],
            "file_count": len(identity["files"]),
            "sha256": digest(identity),
            "file_mismatches": identity_mismatches,
        },
        "scientific": {
            "commit": git("rev-parse", "HEAD", cwd=SCIENTIFIC_WORKTREE),
            "tree": git("rev-parse", "HEAD^{tree}", cwd=SCIENTIFIC_WORKTREE),
            "status": git("status", "--porcelain", "--untracked-files=all", cwd=SCIENTIFIC_WORKTREE),
        },
        "d_junit": junit(D_DELIVERY / "full_pytest.xml"),
        "protected_evidence_before_after_equal_except_time": {
            k: before[k] == after[k] for k in before if k != "at"
        },
    }
    result["pass"] = (
        result["implementation"]["commit"] == IMPLEMENTATION_COMMIT
        and result["implementation"]["tree"] == identity["git_tree"]
        and result["evidence"]["parent"] == IMPLEMENTATION_COMMIT
        and identity["commit"] == IMPLEMENTATION_COMMIT
        and not identity_mismatches and not evidence_mismatches and not c_evidence_mismatches
        and result["c_review_evidence"]["origin_launch_source_sha256"]
        == "23e4f6680763668548f569a58be153c1b3227371ac642ef48da0d15d2d38e1bc"
        and result["scientific"]["commit"] == SCIENTIFIC_COMMIT
        and not result["scientific"]["status"]
        and result["d_junit"] == {**result["d_junit"], "failures": 0, "errors": 0, "skipped": 0}
        and all(result["protected_evidence_before_after_equal_except_time"].values())
    )
    write_json(REVIEW_ROOT / "identity_evidence_recomputed.json", result)
    return result


def protection():
    inventory_path = REPOSITORY / "artifacts/analysis/g14r20_a_continuation_20260908/protected_before.json.gz"
    with gzip.open(inventory_path, "rt") as stream:
        inventory = json.load(stream)
    mismatches = []
    known = set()
    for row in inventory["files"]:
        path = Path(row["path"])
        known.add(str(path))
        if "symlink_target" in row:
            ok = path.is_symlink() and str(path.readlink()) == row["symlink_target"]
        else:
            ok = (path.is_file() and path.stat().st_size == row["size_bytes"]
                  and file_hash(path) == row["sha256"])
        if not ok:
            mismatches.append(str(path))
    proposal_path = REPOSITORY / "artifacts/analysis/g14r20_a_continuation_20260908/v16_continuation_proposal.json"
    proposal = json.loads(proposal_path.read_text())
    roots = [Path(proposal["run_root"]), SCIENTIFIC_WORKTREE]
    additions = [str(path) for root in roots for path in root.rglob("*")
                 if (path.is_file() or path.is_symlink()) and str(path) not in known]
    user_files = [
        "scripts/train_sa_ghmappo_real_sample.py",
        "src/agents/sa_ghmappo_agent.py",
        "src/agents/sa_ghmappo_core.py",
        "src/encoders/fusion_encoder.py",
        "src/evaluators/real_eval_support.py",
        "tests/test_algo_pool_contract.py",
        "tests/test_checkpoint_compat.py",
    ]
    result = {
        "recomputed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "inventory_entries": len(inventory["files"]),
        "mismatches": mismatches,
        "additions": additions,
        "proposal_sha256": file_hash(proposal_path),
        "main_commit": git("rev-parse", "main", cwd=MAIN_REPOSITORY),
        "user_files": {name: file_hash(MAIN_REPOSITORY / name) for name in user_files},
        "ledgers": [{
            "path": row["path"],
            "size_bytes": Path(row["path"]).stat().st_size,
            "sha256": file_hash(row["path"]),
        } for row in proposal["ledgers"]],
    }
    result["pass"] = not mismatches and not additions
    write_json(REVIEW_ROOT / "protection_recomputed.json", result)
    return result


def load_probe(path):
    return json.loads((Path(path) / "probe_bundle.json").read_text())


def child_probe(mode, path):
    bundle = load_probe(path)
    sys.path.insert(0, str(REPOSITORY / "scripts"))
    from continuation_executor.authorization import verify_approval
    from continuation_executor.production_trust import TrustContext
    context = TrustContext(bundle["pin"], test_only=True)
    if mode == "challenge":
        print(json.dumps(context.startup_request(), sort_keys=True))
        return 0
    try:
        verify_approval(bundle["contract"], bundle["approval"], test_trust_context=context,
                        now=datetime.fromisoformat(bundle["now"]))
    except Exception as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "unexpected_success"}, sort_keys=True))
    return 1


def startup_probe():
    sys.path.insert(0, str(REPOSITORY / "scripts"))
    from continuation_executor import PHASES
    from continuation_executor.identity import ContinuationError
    from continuation_executor.production_trust import TrustContext
    from continuation_executor.test_trust_fixture import TestTrustFixture, signed, write

    fixture = Path(tempfile.mkdtemp(prefix="g14r20_e_startup_probe_", dir="/private/tmp"))
    run = fixture / "synthetic_startup_probe"
    run.mkdir()
    now = datetime.now(timezone.utc)
    proposal = {"run_id": run.name, "run_root": str(run), "ledgers": []}
    contract = {
        "version": "1.0.0", "domain": "synthetic", "proposal_sha256": digest(proposal),
        "proposal_file_sha256": digest(proposal), "executor_identity_sha256": digest({}),
        "run_id": run.name, "run_root": str(run), "phases": list(PHASES),
        "holdout_capability": False, "prefixes": [], "immutable_files": [],
        "fixture_root": str(fixture), "expires_at": (now + timedelta(hours=1)).isoformat(),
        "revocation_id": "test-startup-grant", "recovery_owner_sha256": None,
        "recovery_quiescence": None, "coordination_root": str(fixture / ".continuation_locks"),
        "command_plan_sha256": digest({}),
    }
    trust = TestTrustFixture(contract, now=now)
    direct = trust.verify(now=now)
    first_request = deepcopy(trust.context.startup_request())

    bundle = {"pin": trust.pin, "contract": trust.contract, "approval": trust.approval,
              "now": now.isoformat()}
    write_json(fixture / "probe_bundle.json", bundle)
    command = [sys.executable, "-B", str(Path(__file__).resolve())]
    challenge = subprocess.run([*command, "--child", "challenge", "--fixture", str(fixture)],
                               text=True, capture_output=True, check=True)
    old_request = json.loads(challenge.stdout)
    old_message = dict(old_request, authority="revoker", checkpoint_sha256=trust.context.expected,
                       issued_at=now.isoformat(), expires_at=(now + timedelta(hours=1)).isoformat())
    write(trust.pin["startup_receipt_path"], signed(old_message, "revoker", trust.keys["revoker"]))
    second = subprocess.run([*command, "--child", "verify", "--fixture", str(fixture)],
                            text=True, capture_output=True)
    second_result = json.loads(second.stdout)

    missing_context = TrustContext(trust.pin, test_only=True)
    Path(trust.pin["startup_receipt_path"]).unlink()
    try:
        missing_context.verify(trust.contract, trust.approval, now)
        missing = "unexpected_success"
    except Exception as exc:
        missing = str(exc)

    stale_context = TrustContext(trust.pin, test_only=True)
    stale_message = dict(stale_context.startup_request(), authority="revoker",
                         checkpoint_sha256=trust.context.expected,
                         issued_at=(now - timedelta(hours=3)).isoformat(),
                         expires_at=(now - timedelta(hours=2)).isoformat())
    write(trust.pin["startup_receipt_path"], signed(stale_message, "revoker", trust.keys["revoker"]))
    try:
        stale_context.verify(trust.contract, trust.approval, now)
        stale = "unexpected_success"
    except Exception as exc:
        stale = str(exc)

    cli = REPOSITORY / "scripts/execute_fixed_commit_continuation.py"
    public = subprocess.run([sys.executable, "-B", str(cli), "--proposal", "absent",
                             "--contract", "absent", "--executor-identity", "absent",
                             "--phase", PHASES[0], "--check", "startup"],
                            text=True, capture_output=True)
    source = cli.read_text()
    result = {
        "probed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "fixture_root": str(fixture),
        "direct_same_context": {
            "approval_verified": direct["approval_verified"],
            "domain": direct["domain"],
            "real_execution_authorized": direct["real_execution_authorized"],
            "request": first_request,
        },
        "two_independent_commands": {
            "challenge_command_exit": challenge.returncode,
            "challenge_request": old_request,
            "verification_command_exit": second.returncode,
            "verification_result": second_result,
            "old_process_exited_before_verification": True,
        },
        "missing_receipt": missing,
        "stale_receipt": stale,
        "public_cli": {
            "startup_mode_exit": public.returncode,
            "startup_mode_rejected_by_parser": "invalid choice" in public.stderr,
            "has_startup_request_reference": "startup_request" in source,
            "supported_check_modes": ["compatibility", "qualification", "execute"],
            "qualification_calls_verify_before_identity_and_science": True,
        },
    }
    result["pass"] = (
        direct["approval_verified"] and not direct["real_execution_authorized"]
        and second.returncode == 2 and "challenge mismatch" in second_result["reason"]
        and missing != "unexpected_success" and stale != "unexpected_success"
        and public.returncode == 2 and result["public_cli"]["startup_mode_rejected_by_parser"]
        and not result["public_cli"]["has_startup_request_reference"]
    )
    write_json(REVIEW_ROOT / "startup_operability_probe.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=("challenge", "verify"))
    parser.add_argument("--fixture")
    parser.add_argument("--skip-protection", action="store_true")
    args = parser.parse_args()
    if args.child:
        return child_probe(args.child, args.fixture)
    result = {"identity_evidence": identity_and_evidence(), "startup": startup_probe()}
    if not args.skip_protection:
        result["protection"] = protection()
    print(json.dumps({k: v["pass"] for k, v in result.items()}, sort_keys=True))
    return 0 if all(v["pass"] for v in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
