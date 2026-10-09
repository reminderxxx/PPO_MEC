"""Audit hidden prepared-state aliases using only recorded long-budget events."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_calibrated_workflow_interface_repair import _sha256, _write_json

DIAG = ROOT / "artifacts/analysis/cscwd_sa_long_budget_cost_diagnosis_20261009_v1"
B_SOURCE = Path("/Users/howen/.codex/worktrees/causal-budget-extension/PPO_MEC/artifacts/experiments/cscwd_causal_strong_baselines_budget_extension_20261009_v1")
OUTPUT = ROOT / "artifacts/analysis/cscwd_sa_long_budget_state_alias_20261009_v2"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"create-only alias audit root exists: {OUTPUT}")
    diag_receipt = json.loads((DIAG / "completion_receipt.json").read_text())
    if diag_receipt.get("status") != "complete" or diag_receipt.get("replayed_steps") != 2811:
        raise RuntimeError("cost replay not complete")
    diag_inventory = {r["path"]: r for r in json.loads((DIAG / "artifact_integrity.json").read_text())["files"]}
    scientific_inventory = {r["path"]: r for r in json.loads((B_SOURCE / "artifact_integrity.json").read_text())["files"]}
    for path, inventory, name in ((DIAG / "event_rows.jsonl", diag_inventory, "event_rows.jsonl"),
                                  (B_SOURCE / "behavior_ledger.csv", scientific_inventory, "behavior_ledger.csv")):
        if path.stat().st_size != inventory[name]["bytes"] or _sha256(path) != inventory[name]["sha256"]:
            raise RuntimeError(f"source ledger integrity mismatch: {name}")
    events = [json.loads(line) for line in (DIAG / "event_rows.jsonl").read_text().splitlines()]
    ledger = _read_csv(B_SOURCE / "behavior_ledger.csv")
    if len(events) != 2811 or len(ledger) != 2811:
        raise RuntimeError("event/behavior count mismatch")
    records = {(r["split"], r["method"], r["seed"], r["design_id"], int(r["step_index"])): r for r in ledger}
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for event in events:
        key = (event["split"], event["method"], event["seed"], event["design_id"], event["step_index"])
        row = records[key]
        event["recorded_state_ready"] = row["state_ready"] == "True"
        if event["past_handoff_now"] and event["action"] != 2:
            groups[(event["design_id"], event["public_state_sha256"], event["action"])].append(event)
    aliases = []
    for (_, public_hash, action), rows in groups.items():
        if len(rows) > 1 and len({r["recorded_state_ready"] for r in rows}) > 1:
            aliases.append({"public_state_sha256": public_hash, "action": action,
                "instance": rows[0]["design_id"], "rows": len(rows),
                "methods": sorted({r["method"] for r in rows}),
                "outcomes": [{"method": r["method"], "seed": r["seed"], "split": r["split"],
                    "step_index": r["step_index"], "state_ready": r["recorded_state_ready"],
                    "dag_recompute": r["dag_recompute"]} for r in rows]})
    by_episode: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for event in events:
        by_episode[(event["split"], event["method"], event["seed"], event["design_id"])].append(event)
    handoff = defaultdict(Counter)
    opportunity = Counter()
    for key, rows in by_episode.items():
        split, method, _, _ = key
        prepared: dict[str, tuple[int, int]] = {}
        rows.sort(key=lambda r: r["step_index"])
        for index, event in enumerate(rows):
            if event["past_handoff_now"] and event["action"] != 2:
                counts = handoff[(method, split)]
                counts["offloaded_handoffs"] += 1
                if event["recorded_state_ready"]:
                    counts["state_ready"] += 1
                else:
                    counts["state_not_ready"] += 1
                    prior = prepared.get(event["current_rsu"])
                    counts["no_prior_state_prepare" if prior is None else "stale_after_node_progress" if prior[1] else "prior_prepare_but_not_ready"] += 1
                    if event["dag_recompute"] > 0:
                        counts["positive_recompute"] += 1
                        if method == "sa_ghmappo" and split == "frozen_check" and index:
                            before = rows[index - 1]
                            before_key = (before["split"], before["method"], before["seed"], before["design_id"], before["step_index"])
                            record = records[before_key]
                            if before["target_rsu"] == event["current_rsu"]:
                                opportunity["matching_prior_target"] += 1
                                if before["current_ready"] and before["target_prepare_feasible"] and record["prepare_feasibility_reason"] == "committed":
                                    opportunity["prior_publicly_ready_and_preview_committed"] += 1
                                    opportunity[f"prior_action_{before['action']}"] += 1
            if event["node_progressed"]:
                prepared = {rsu: (step, progressed + 1) for rsu, (step, progressed) in prepared.items()}
            if event["migration_prepare_committed"] and event["target_rsu"]:
                prepared[event["target_rsu"]] = (event["step_index"], 0)
    sa_only = [row for row in aliases if len({item["state_ready"] for item in row["outcomes"] if item["method"] == "sa_ghmappo"}) > 1]
    result = {"schema_version": "cscwd_long_budget_state_alias_audit_v1",
        "source_run_id": B_SOURCE.name, "scientific_commit": "d25ebcded6b43b69b82bae825b035adc1d6f19c4",
        "same_public_state_and_action_mixed_state_ready_groups": len(aliases),
        "within_sa_mixed_groups": len(sa_only), "aliases": aliases,
        "handoff_counts": {f"{method}:{split}": dict(counts) for (method, split), counts in sorted(handoff.items())},
        "sa_frozen_immediate_prior_opportunity": dict(opportunity),
        "boundary": "posthoc_hidden_state_audit_no_policy_future_input_no_counterfactual_performance_matrix"}
    OUTPUT.mkdir(parents=True)
    _write_json(OUTPUT / "state_alias_summary.json", result)
    _write_json(OUTPUT / "completion_receipt.json", {"status": "complete", "at": datetime.now(timezone.utc).isoformat(),
        "input_event_rows": len(events), "input_behavior_rows": len(ledger), "training_steps": 0,
        "forward_calls": 0, "local_branch_states": 0, "formal_or_holdout_reads": 0})
    files = sorted(p for p in OUTPUT.iterdir() if p.is_file() and p.name != "artifact_integrity.json")
    _write_json(OUTPUT / "artifact_integrity.json", {"files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": _sha256(p)} for p in files]})
    return result


if __name__ == "__main__":
    outcome = audit()
    print(json.dumps({"status": "complete", "mixed_alias_groups": outcome["same_public_state_and_action_mixed_state_ready_groups"],
                      "within_sa": outcome["within_sa_mixed_groups"]}))
