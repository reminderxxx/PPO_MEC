"""Create-only preparation/evidence helpers; never issue a review or dispatch."""
from datetime import datetime, timezone
import gzip
from pathlib import Path

from . import PHASES
from .identity import absolute_path, canonical, digest, file_hash, git, read_json, verify_executor
from .planning import phase_plan
from .project_authorization import MODE, OWNER_DECISION, RUN_ID, RUN_ROOT, SCIENTIFIC_COMMIT, SCIENTIFIC_ROOT, validate_project_scope

PROPOSAL = Path("/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_a_continuation_20260908/v16_continuation_proposal.json")
PROTECTION = PROPOSAL.with_name("protected_before.json.gz")
AUDIT_PARENT = (Path(__file__).resolve().parents[2] / "artifacts/analysis" /
                "g14r20_g_project_authorization_20260910")


def write_new(path, value):
    path = absolute_path(str(path))
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def prepare(output):
    """Invoke in the old scientific cwd with its original Python/environment."""
    output = absolute_path(str(output))
    if output.parent != AUDIT_PARENT or output.exists():
        raise ValueError("new packet must be a unique child of the project audit directory")
    executor_root = Path(__file__).resolve().parents[2]
    names = {"scripts/execute_fixed_commit_continuation.py", "scripts/run_fixed_commit_continuation_acceptance.py", "src/runtime/fixed_commit_continuation.py"}
    names.update(p.relative_to(executor_root).as_posix() for p in (executor_root / "scripts/continuation_executor").glob("*.py"))
    identity = {"version": "1.0.0", "commit": git(executor_root, "rev-parse", "HEAD"),
                "git_tree": git(executor_root, "rev-parse", "HEAD^{tree}"),
                "files": [{"path": name, "sha256": file_hash(executor_root / name)} for name in sorted(names)]}
    verify_executor(identity, executor_root)
    from .scientific import load_native
    native = load_native(SCIENTIFIC_ROOT, SCIENTIFIC_COMMIT)
    proposal = read_json(PROPOSAL)
    run = Path(RUN_ROOT)
    context = read_json(run / "resolved_execution_context.json")
    context_hash = file_hash(run / "resolved_execution_context.json")
    protocol = read_json(context["runtime_location"]["protocol_path"])
    binding = read_json(run / "formal_training_execution_binding.json")
    registry = read_json(run / "generated_checkpoint_resource_registry.json")
    plans = {phase: phase_plan(phase, protocol, context, binding, context_hash,
                registry["registry_canonical_sha256"], native["execution"].expand_command_plan) for phase in PHASES}
    contract = {"version": "1.0.0", "domain": "production", "proposal_sha256": digest(proposal),
                "proposal_file_sha256": file_hash(PROPOSAL), "executor_identity_sha256": digest(identity),
                "run_id": RUN_ID, "run_root": RUN_ROOT, "phases": list(PHASES), "holdout_capability": False,
                "prefixes": proposal["ledgers"], "immutable_files": [
                    {key: row[key] for key in ("path", "sha256", "size_bytes")} for row in proposal["evidence"]],
                "fixture_root": None, "expires_at": OWNER_DECISION["expires_at"],
                "revocation_id": MODE + ":" + RUN_ID, "recovery_owner_sha256": None, "recovery_quiescence": None,
                "coordination_root": str(run.parent / ".continuation_locks"), "command_plan_sha256": digest(plans)}
    validate_project_scope(contract, proposal, identity)
    AUDIT_PARENT.mkdir(exist_ok=True)
    output.mkdir()
    for name, value in (("executor_identity.json", identity), ("execution_contract.json", contract),
                        ("command_plans.json", plans), ("owner_decision.json", OWNER_DECISION)):
        write_new(output / name, value)
    return {"status": "prepared_not_authorized", "output": str(output), "executor": digest(identity),
            "contract": digest(contract), "phase_count": len(plans),
            "command_count": sum(len(p["commands"]) for p in plans.values())}


def protection_snapshot():
    """Independent of scores; verify the exact preserved A inventory."""
    with gzip.open(PROTECTION, "rt", encoding="utf-8") as stream:
        from .identity import strict_json
        original = strict_json(stream.read())
    mismatches = []
    for row in original["files"]:
        p = Path(row["path"])
        if "symlink_target" in row:
            ok = p.is_symlink() and str(p.readlink()) == row["symlink_target"]
        else:
            ok = p.is_file() and p.stat().st_size == row["size_bytes"] and file_hash(p) == row["sha256"]
        if not ok:
            mismatches.append(str(p))
    known = {row["path"] for row in original["files"]}
    additions = [str(p) for root in (Path(RUN_ROOT), Path(SCIENTIFIC_ROOT)) for p in root.rglob("*")
                 if (p.is_file() or p.is_symlink()) and str(p) not in known]
    # Additions must be empty before execution. After continuation they are
    # expected and must be classified against committed native ledger evidence.
    return {"at": datetime.now(timezone.utc).isoformat(), "count": len(known),
            "mismatches": mismatches, "additions": additions,
            "main": git(PROPOSAL.parents[3], "rev-parse", "main"),
            "scientific_head": git(SCIENTIFIC_ROOT, "rev-parse", "HEAD"),
            "proposal_sha256": file_hash(PROPOSAL)}


def read_only_qualification(packet):
    """Run actual original validators under the non-removable no-dispatch monitor."""
    from .monitoring import Monitor
    monitor = Monitor()
    packet = absolute_path(str(packet))
    identity, contract = read_json(packet / "executor_identity.json"), read_json(packet / "execution_contract.json")
    verify_executor(identity, Path(__file__).resolve().parents[2])
    proposal = read_json(PROPOSAL)
    validate_project_scope(contract, proposal, identity)
    from .scientific import QualifiedRun
    run = QualifiedRun(proposal, contract)
    return {"status": "read_only_compatible_not_authorized", "execution_authorized": False,
            "reconciliation": run.reconciliation, "checkpoint_audit": run.checkpoints,
            "registry_audit": run.registry_audit, "active_resource_count": len(run.bundle["resource_ids"]),
            "command_plan_sha256": digest(run.plans), "origins": run.origins,
            "environment": run.environment.runtime_audit, "monitor": monitor.report()}
