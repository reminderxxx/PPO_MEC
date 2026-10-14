"""Rebuild a v2 source protocol and verify explicit v3 controls on train/dev only."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.launch_eight_vehicle_joint_experiment import arguments, _hash_file, _load_protocol
from src.envs.core.multi_request_target_control_env import JointRequestControl, MultiRequestTargetControlEnv
from src.envs.core.multi_request_replay_env import ReplayCosts


def select_targets(decisions: dict, policy: str, assignments: dict) -> dict:
    load = Counter()
    selected = {}
    for vid, decision in sorted(decisions.items()):
        state = decision["semantic_state"]
        candidates = state["reachable_execution_rsu_ids"]
        if policy == "associated_reactive":
            target = state["associations"][vid]
        elif assignments.get(vid) in candidates:
            target = assignments[vid]
        else:
            target = min(candidates, key=lambda site: (load[site], site)) if candidates else None
        selected[vid] = target
        if target is not None:
            load[target] += 1
    assignments.update(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--output_root", type=Path, default=ROOT / "artifacts/joint_target_control_engineering")
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if not source.get("development_only") or source.get("environment_revision") != "v2":
        raise ValueError("source must be a v2 development manifest")
    protocol_args = arguments(source["command"][1:])
    for name, path in (("ngsim", protocol_args.mobility_csv_path), ("alibaba", protocol_args.workflow_csv_path)):
        if _hash_file(path) != source["raw_data_sha256"][name]:
            raise ValueError(f"source data hash changed: {name}")
    splits, workflows, rsus = _load_protocol(protocol_args)
    public = [{key: value for key, value in window.items() if key != "frames"} for window in splits[args.split]]
    if public != source["splits"][args.split]:
        raise ValueError("reconstructed split differs from source manifest")
    if not 1 <= args.windows <= len(public):
        raise ValueError("window count exceeds the selected split")
    windows = splits[args.split][:args.windows]
    options = {key: value for key, value in source["environment_options"].items()
               if key not in {"environment_revision", "strict_cold_cache_mask"}}
    options.setdefault("model_cache_config", source["costs"]["model_cache"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_engineering")
    output = args.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    policies = ["associated_reactive", "reachable_balanced_reactive"]
    metadata = {"run_id": run_id, "status": "running",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "command": sys.argv, "source_run_id": source["run_id"],
        "source_manifest_sha256": hashlib.sha256(args.source_manifest.read_bytes()).hexdigest(),
        "split": args.split, "window_count": len(windows), "windows": public[:args.windows],
        "raw_data_sha256": source["raw_data_sha256"], "environment_options": options,
        "policies": policies, "evidence_scope": "engineering_rules_no_RL_no_formal_holdout",
        "limitations": ["frame rates and byte profiles are synthetic",
            "balanced rule coordinates currently active requests without future trajectory access",
            "shared control messages are charged but do not reserve data link slots",
            "v3 registered learning-agent and checkpoint adapters are not implemented"]}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    with (output / "events.jsonl").open("w", encoding="utf-8") as event_file:
        for policy in policies:
            for window in windows:
                env = MultiRequestTargetControlEnv(frames=window["frames"],
                    vehicle_workflows=dict(zip(window["vehicle_ids"], workflows)), rsu_states=rsus,
                    costs=ReplayCosts(**source["costs"]["replay_costs"]), **options)
                assignments, reward = {}, 0.0
                while True:
                    decisions = env.decision_states()
                    targets = select_targets(decisions, policy, assignments)
                    controls = {vid: JointRequestControl(targets[vid], targets[vid],
                        decision["semantic_state"]["current_workflow_node"]["node_id"] if targets[vid] else None)
                        for vid, decision in decisions.items()}
                    _, value, done, info = env.step_controls(controls)
                    reward += value
                    receipt = info["events"][0]["frame_cost_receipt"]
                    assert value == sum(receipt["reward_components"].values())
                    assert all(used <= env.costs.link_slots_per_rsu for used in receipt["link_usage_by_rsu"].values())
                    for event in info["events"]:
                        event_file.write(json.dumps({"policy": policy, "frame_offset": window["frame_offset"],
                                                     **event}) + "\n")
                    if done:
                        break
                receipts = [e["frame_cost_receipt"] for e in env.events if "frame_cost_receipt" in e]
                assert abs(reward - sum(sum(r["reward_components"].values()) for r in receipts)) < 1e-10
                rows.append({"policy": policy, "frame_offset": window["frame_offset"],
                             "reward": reward, "configuration": env.configuration(), **env.summary()})
    (output / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {policy: {"mean_reward": sum(r["reward"] for r in rows if r["policy"] == policy) / len(windows),
                        "mean_completed_requests": sum(r["completed_requests"] for r in rows if r["policy"] == policy) / len(windows)}
               for policy in policies}
    metadata.update(status="complete", summary=summary,
                    finished_at=datetime.now(timezone.utc).isoformat())
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
