"""Bounded fixed-action development sensitivity for raw NGSIM event time."""

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

from scripts.freeze_calibrated_continuous_workflow_pilot import _canonical_sha256, _load_experiment_config
from src.data.mobility.ngsim_event_trace import FROZEN_DESIGN_IDS, load_frozen_traces
from src.envs.core.calibrated_continuous_workflow_env import CalibratedContinuousWorkflowEnv
from src.envs.core.raw_ngsim_event_time_env import RawNGSIMEventTimeEnv


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config_path = ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json"
    manifest_path = ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json"
    config, _ = _load_experiment_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _sha256(config_path) != manifest["config_sha256"] or _canonical_sha256(config) != manifest["resolved_config_sha256"]:
        raise RuntimeError("parent workload identity mismatch")
    source_rows = [next(row for row in manifest["instances"] if row["design_id"] == design_id)
                   for design_id in FROZEN_DESIGN_IDS]
    traces, source = load_frozen_traces(args.raw_csv_path, source_rows)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output exists: {output}")
    output.mkdir(parents=True)
    _write(output / "source_manifest.json", source)
    results = []
    total_steps = 0
    preview_calls = 0
    for row in source_rows:
        design_id = row["design_id"]
        if design_id not in traces:
            continue
        for profile in ("decision_step_original", "raw_ngsim_event_time_v1"):
            for fixed_action in range(5):
                env = (CalibratedContinuousWorkflowEnv(config, row) if profile == "decision_step_original"
                       else RawNGSIMEventTimeEnv(config, row, traces[design_id]))
                _, _ = env.reset()
                reasons: Counter[str] = Counter()
                terminated = truncated = False
                for _ in range(min(int(row["max_steps"]), 24)):
                    _, _, terminated, truncated, info = env.step(fixed_action)
                    total_steps += 1
                    if profile == "raw_ngsim_event_time_v1":
                        preview_calls += 1
                    reason = info["transition"].get("admission_rejection_reason")
                    if reason:
                        reasons[str(reason)] += 1
                    if terminated or truncated:
                        break
                summary = env.summary()
                results.append({"design_id": design_id, "split": row["split"], "profile": profile,
                                "fixed_action": fixed_action, "terminated": terminated, "truncated": truncated,
                                "admission_rejections": dict(reasons), "steps": summary["steps"],
                                "completed_nodes": summary["completed_nodes"],
                                "workflow_completed": summary["workflow_completed"],
                                "service_failures": summary["service_failures"],
                                "model_transfer_bytes": summary["model_transfer_bytes"],
                                "state_transfer_bytes": summary["state_transfer_bytes"],
                                "input_transfer_bytes": summary["input_transfer_bytes"],
                                "modeled_completion_seconds": summary["modeled_completion_seconds"],
                                "reward": summary["reward"]})
                if total_steps > 720 or preview_calls > 10_000 or len(results) > 30:
                    raise RuntimeError("frozen development sensitivity budget exceeded")
    report = {"schema_version": "cscwd_raw_ngsim_event_time_development_sensitivity_v1",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "parent_config_sha256": manifest["config_sha256"],
              "parent_manifest_sha256": _sha256(manifest_path),
              "claim_scope": "already_exposed_development_fixed_action_sensitivity_only",
              "profiles": ["decision_step_original", "raw_ngsim_event_time_v1"],
              "fixed_policy_definition": "repeat action index 0/1/2/3/4 under the public action mask",
              "episode_count": len(results), "real_step_count": total_steps, "preview_call_count": preview_calls,
              "results": results}
    _write(output / "summary.json", report)
    print(json.dumps({"output": str(output), "episodes": len(results), "steps": total_steps,
                      "validated_instances": list(traces), "excluded_instances": [key for key in FROZEN_DESIGN_IDS if key not in traces]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
