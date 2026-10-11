"""Read-only consistency audit for the one-shot repeated-action4 branch artifact."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


RUN_ID = "cscwd_phase_v2_repeated_action4_branches_20261011_v1"
PLAN_COMMIT = "da2c8b7474b0f06d8c9f1fa0a6389c95faf85714"
SUMMARY_SHA = "be57ec84b234e31e9866d768b095d2b25ce58645c7f1c8bc179ff27aee8e9a1d"
CONDITIONS = ("default", "adverse_link", "ample_cache")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def audit(root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("audit output is create-only")
    if root.name != RUN_ID or sha(root / "summary.json") != SUMMARY_SHA:
        raise RuntimeError("frozen branch source identity mismatch")
    start, completion = read(root / "start_receipt.json"), read(root / "completion_receipt.json")
    summary, integrity = read(root / "summary.json"), read(root / "artifact_integrity.json")
    starts, pairs = read(root / "starts.json"), read(root / "paired_branches.json")
    events = [json.loads(line) for line in (root / "event_ledger.jsonl").read_text().splitlines()]
    if (start["git_commit"] != PLAN_COMMIT or completion["git_commit"] != PLAN_COMMIT
            or start["status"] != "STARTED" or completion["status"] != "COMPLETE"
            or completion["summary_sha256"] != SUMMARY_SHA
            or completion["integrity_sha256"] != sha(root / "artifact_integrity.json")):
        raise RuntimeError("branch receipt provenance mismatch")
    listed = set()
    for item in integrity["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or relative in listed:
            raise RuntimeError("invalid artifact integrity path")
        listed.add(relative)
        path = root / relative
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise RuntimeError(f"branch artifact SHA mismatch: {relative}")
    if listed != {Path(name) for name in (
            "start_receipt.json", "starts.json", "paired_branches.json", "event_ledger.jsonl", "summary.json")}:
        raise RuntimeError("branch artifact integrity file set drift")
    if (len(starts) != 3 or len(pairs) != 3 or len(events) != 45
            or summary["environment_step_count"] != 45 or completion["environment_step_count"] != 45
            or summary["branch_count"] != 6 or completion["branch_count"] != 6
            or sum(start["pilot_repeated_count"] for start in starts) != 11):
        raise RuntimeError("branch budget or witness count drift")
    if {row["condition_id"] for row in starts} != set(CONDITIONS):
        raise RuntimeError("start condition grid drift")
    if len({row["public_state_sha256"] for row in starts}) != 3:
        raise RuntimeError("public start identity collision")
    groups = defaultdict(list)
    for event in events:
        if event["condition_id"] not in CONDITIONS or event["branch"] not in (
                "common_prefix", "action4", "action3"):
            raise RuntimeError("unexpected event stratum")
        phases = event["phase_events"]
        if not math.isclose(sum(float(p["elapsed_seconds"]) for p in phases),
                            float(event["step_cost_seconds"]), abs_tol=1e-6, rel_tol=0.0):
            raise RuntimeError("event phase clock mismatch")
        if event["backhaul_transfer_bytes"] != event["model_transfer_bytes"] + event["state_transfer_bytes"]:
            raise RuntimeError("event backhaul bytes mismatch")
        if not event["public_action_mask_before"][event["executed_action"]]:
            raise RuntimeError("event action masked")
        groups[(event["condition_id"], event["branch"])].append(event)
    result = []
    for condition in CONDITIONS:
        prefix = groups[(condition, "common_prefix")]
        if len(prefix) != 1 or prefix[0]["executed_action"] != 4 or not prefix[0]["migration_success"]:
            raise RuntimeError("common prefix not delivered")
        pair = next(row for row in pairs if row["condition_id"] == condition)
        start = next(row for row in starts if row["condition_id"] == condition)
        if pair["public_state_sha256"] != start["public_state_sha256"]:
            raise RuntimeError("paired public state identity mismatch")
        for action in (4, 3):
            name = f"action{action}"
            rows = groups[(condition, name)]
            recorded = pair[name]
            metrics = recorded["metrics"]
            if len(rows) != 7 or recorded["branch_env_steps"] != len(rows) or rows[0]["executed_action"] != action:
                raise RuntimeError("branch action or step count mismatch")
            if rows[0]["public_state_sha256_before"] != start["public_state_sha256"]:
                raise RuntimeError("branches do not share public start state")
            if any(event["executed_action"] not in (2, 3) for event in rows[1:]):
                raise RuntimeError("continuation action contract mismatch")
            for key in ("input_transfer_bytes", "model_transfer_bytes", "state_transfer_bytes",
                        "backhaul_transfer_bytes", "partial_transfer_bytes"):
                if metrics[key] != sum(int(event[key]) for event in rows):
                    raise RuntimeError(f"branch {key} aggregation mismatch")
            if metrics["total_transfer_bytes"] != sum(
                    metrics[key] for key in ("input_transfer_bytes", "model_transfer_bytes", "state_transfer_bytes")):
                raise RuntimeError("branch total bytes mismatch")
            for key, event_key in (("elapsed_seconds", "step_cost_seconds"), ("recompute_seconds", "recompute_seconds"),
                                   ("reward", "reward")):
                if not math.isclose(metrics[key], sum(float(event[event_key]) for event in rows),
                                    abs_tol=1e-6, rel_tol=0.0):
                    raise RuntimeError(f"branch {key} aggregation mismatch")
            if metrics["workflow_completed"] is not True or metrics["on_time"] is not True or metrics["service_failures"] != 0:
                raise RuntimeError("branch service result drift")
        a4, a3 = pair["action4"]["metrics"], pair["action3"]["metrics"]
        if pair["action4"]["first_event"]["migration_status"] != "target_departed_before_delivery":
            raise RuntimeError("first repeated action4 failure drift")
        for key, delta in pair["action3_minus_action4"].items():
            if not math.isclose(a3[key] - a4[key], delta, abs_tol=1e-6, rel_tol=0.0):
                raise RuntimeError(f"paired delta mismatch: {key}")
        if not pair["locally_dominated_in_frozen_continuation"] or a3["total_transfer_bytes"] >= a4["total_transfer_bytes"]:
            raise RuntimeError("frozen local outcome drift")
        result.append({"condition_id": condition, "pilot_repeated_count": start["pilot_repeated_count"],
                       "bytes_saved_by_action3": a4["total_transfer_bytes"] - a3["total_transfer_bytes"],
                       "model_bytes_saved": a4["model_transfer_bytes"] - a3["model_transfer_bytes"],
                       "state_bytes_saved": a4["state_transfer_bytes"] - a3["state_transfer_bytes"],
                       "seconds_saved": a4["elapsed_seconds"] - a3["elapsed_seconds"],
                       "service_equal": True})
    output.mkdir(parents=True)
    audit_result = {"schema_version": "cscwd_phase_v2_repeated_action4_audit_v1",
                    "reviewed_at": datetime.now(timezone.utc).isoformat(), "source_run_id": RUN_ID,
                    "source_summary_sha256": SUMMARY_SHA, "source_integrity_sha256": sha(root / "artifact_integrity.json"),
                    "source_file_count": len(listed), "source_event_rows": len(events),
                    "conditions": result, "all_checks_passed": True,
                    "development_only": True, "formal_or_holdout_results_read": False}
    (output / "audit.json").write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n")
    return audit_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.source_root.resolve(), args.output_root.resolve())
    print(json.dumps({"all_checks_passed": result["all_checks_passed"],
                      "audit_sha256": sha(args.output_root.resolve() / "audit.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
