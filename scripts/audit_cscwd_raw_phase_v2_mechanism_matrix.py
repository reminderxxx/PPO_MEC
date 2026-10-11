"""Read-only cross-check of the frozen 60-cell development mechanism ledger."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_cscwd_new_development_reachability import sha  # noqa: E402


MATRIX_SHA = "cb581f6871c0928f13f58f14fb77568290277247dc7de8c03ed355aeeca4bc2c"
LEDGER_SHA = "bfb2d6743266a24c2a37bf9b714a53f723f0595e6215ec82a8a0d348c9c64028"
V2 = "raw_ngsim_event_time_v2"


def audit(manifest_path: Path, ledger_path: Path) -> dict:
    if sha(manifest_path) != MATRIX_SHA or sha(ledger_path) != LEDGER_SHA:
        raise RuntimeError("frozen mechanism artifact hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    ledgers = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    results = manifest["results"]
    if len(results) != len(ledgers) or len(results) != 60:
        raise RuntimeError("60-cell artifact count mismatch")
    if sum(len(row["steps"]) for row in ledgers) != manifest["real_step_count"]:
        raise RuntimeError("real step count mismatch")
    if sum(row["preview_count_upper_bound"] for row in results) != manifest["preview_count_upper_bound"]:
        raise RuntimeError("preview count mismatch")
    if manifest["real_step_count"] > 2880 or manifest["preview_count_upper_bound"] > 10000:
        raise RuntimeError("frozen budget exceeded")
    unique = set()
    phase_status = Counter()
    migration_status = Counter()
    failures = Counter()
    aggregate = defaultdict(lambda: Counter())
    for result, ledger in zip(results, ledgers):
        key = (result["window_id"], result["condition_id"],
               result["time_profile"], result["method"])
        if key in unique or key != (
            ledger["window_id"], ledger["condition_id"],
            ledger["time_profile"], ledger["method"],
        ):
            raise RuntimeError("duplicate or unpaired matrix cell")
        unique.add(key)
        summary = result["summary"]
        steps = ledger["steps"]
        if not steps or len(steps) != result["steps"]:
            raise RuntimeError("episode step count mismatch")
        if summary["completed_nodes"] > summary["node_count"]:
            raise RuntimeError("duplicate node completion")
        if summary["migration_successes"] > summary["migration_attempts"]:
            raise RuntimeError("migration success exceeds attempts")
        if summary["total_transfer_bytes"] != (
            summary["model_transfer_bytes"] + summary["state_transfer_bytes"]
            + summary["input_transfer_bytes"]
        ):
            raise RuntimeError("episode transfer bytes do not conserve")
        clock = float(steps[0]["clock_seconds_before"])
        completed = successes = service_failures = 0
        for step in steps:
            event = step["transition"]
            if not math.isclose(float(step["clock_seconds_before"]), clock, abs_tol=1e-6):
                raise RuntimeError("noncontiguous episode clock")
            elapsed = float(event["clock_seconds_after"]) - clock
            if not math.isclose(elapsed, float(event["step_cost_seconds"]), abs_tol=1e-6):
                raise RuntimeError("step clock does not conserve")
            clock = float(event["clock_seconds_after"])
            completed += bool(event["service_completed"])
            successes += bool(event.get("migration_success", False))
            service_failures += not bool(event["service_completed"])
            failures[(result["time_profile"], str(
                event.get("failure_reason") or event.get("admission_rejection_reason") or "none"
            ))] += 1
            if result["time_profile"] == V2:
                if not math.isclose(
                    sum(float(phase["elapsed_seconds"]) for phase in event["phase_events"]),
                    float(event["step_cost_seconds"]), abs_tol=1e-6,
                ):
                    raise RuntimeError("v2 phase clock does not conserve")
                if event["backhaul_transfer_bytes"] != (
                    event["model_transfer_bytes"] + event["state_transfer_bytes"]
                ):
                    raise RuntimeError("v2 backhaul bytes do not conserve")
                if event["partial_transfer_bytes"] > (
                    event["input_transfer_bytes"] + event["backhaul_transfer_bytes"]
                ):
                    raise RuntimeError("partial bytes exceed charged bytes")
                migration_status[str(event["migration_status"])] += 1
                for phase in event["phase_events"]:
                    phase_status[(str(phase["phase"]), str(phase["status"]))] += 1
        if not math.isclose(clock, float(summary["modeled_completion_seconds"]), abs_tol=1e-6):
            raise RuntimeError("episode final clock mismatch")
        if completed != summary["completed_nodes"] or successes != summary["migration_successes"]:
            raise RuntimeError("event and summary counts disagree")
        if service_failures != summary["service_failures"]:
            raise RuntimeError("event and summary failures disagree")
        group = aggregate[(result["time_profile"], result["condition_id"], result["method"])]
        group["episodes"] += 1
        group["completed"] += result["workflow_completed"]
        group["on_time"] += result["on_time_complete"]
        group["migration_successes"] += summary["migration_successes"]
        group["service_failures"] += summary["service_failures"]
        group["steps"] += result["steps"]
        group["model_transfer_bytes"] += summary["model_transfer_bytes"]
        group["state_transfer_bytes"] += summary["state_transfer_bytes"]
        group["backhaul_transfer_bytes"] += summary.get("backhaul_transfer_bytes", 0)
        group["modeled_completion_microseconds"] += round(float(summary["modeled_completion_seconds"]) * 1e6)
    if len(unique) != 60 or len(aggregate) != 30 or any(row["episodes"] != 2 for row in aggregate.values()):
        raise RuntimeError("frozen factor grid incomplete")
    return {
        "schema_version": "cscwd_raw_phase_v2_matrix_read_only_audit_v1",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "matrix_manifest_sha256": MATRIX_SHA,
        "step_ledger_sha256": LEDGER_SHA,
        "episode_count": len(unique),
        "real_step_count": manifest["real_step_count"],
        "preview_count_upper_bound": manifest["preview_count_upper_bound"],
        "all_checks_passed": True,
        "phase_status_counts": {f"{a}:{b}": count for (a, b), count in sorted(phase_status.items())},
        "migration_status_counts": dict(sorted(migration_status.items())),
        "failure_reason_counts": {f"{a}:{b}": count for (a, b), count in sorted(failures.items())},
        "aggregate_rows": [
            {"time_profile": profile, "condition_id": condition, "method": method, **dict(aggregate[(profile, condition, method)])}
            for profile, condition, method in sorted(aggregate)
        ],
        "paper_ready": False,
        "formal_or_holdout_results_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", required=True, type=Path)
    parser.add_argument("--step-ledger", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    record = audit(args.matrix_manifest, args.step_ledger)
    output.mkdir(parents=True)
    path = output / "audit.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": str(path), "sha256": sha(path),
                      "episodes": record["episode_count"],
                      "steps": record["real_step_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
