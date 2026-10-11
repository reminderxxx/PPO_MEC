"""Hash-locked two-trace reset/step acceptance for opt-in raw phase v2."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_cscwd_new_development_reachability import (  # noqa: E402
    BASE_ENV_SHA, PARENT_SHA, RAW_ENV_SHA, RAW_SHA, SOURCE_SHA, load_traces, sha,
)
from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config  # noqa: E402
from src.envs.core.raw_ngsim_phase_v2_env import RawNGSIMPhaseV2Env  # noqa: E402


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def preflight(raw_path: Path, source_path: Path) -> tuple[dict, dict]:
    expected = (
        (raw_path, RAW_SHA), (source_path, SOURCE_SHA),
        (ROOT / "src/envs/core/raw_ngsim_event_time_env.py", RAW_ENV_SHA),
        (ROOT / "src/envs/core/calibrated_continuous_workflow_env.py", BASE_ENV_SHA),
        (ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json", PARENT_SHA),
    )
    for path, digest in expected:
        if sha(path) != digest:
            raise RuntimeError(f"frozen input mismatch: {path}")
    config_path = ROOT / "configs/experiment/cscwd_raw_phase_v2_development_20261011.json"
    contract = json.loads(config_path.read_text())
    if contract["source_manifest_sha256"] != SOURCE_SHA or contract["raw_source_sha256"] != RAW_SHA:
        raise RuntimeError("phase contract source identity mismatch")
    source = json.loads(source_path.read_text())
    traces, identity = load_traces(raw_path, source["selected_windows"])
    config, _ = _load_experiment_config(
        ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json"
    )
    parent = json.loads((ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json").read_text())
    instance = next(row for row in parent["instances"] if row["design_id"] == "dev_01")
    default = next(row for row in contract["conditions"] if row["condition_id"] == "default")
    rows = []
    public_fixture = None
    for window in source["selected_windows"]:
        window_id = window["window_id"]
        env = RawNGSIMPhaseV2Env(
            config, deepcopy(instance), traces[window_id],
            phase_contract=contract["physical_assumptions"], condition=default,
        )
        _, before_info = env.reset()
        if before_info["semantic_state"]["time_profile"] != contract["time_profile"]:
            raise RuntimeError("phase profile not public")
        mask = list(before_info["action_mask"])
        action = 4 if mask[4] else 3 if mask[3] else 2
        if window["source_segment_id"] == "us_101":
            public_info = {key: before_info[key] for key in ("semantic_state", "action_mask", "run_metadata")}
            public_fixture = {
                "schema_version": "cscwd_raw_phase_v2_public_fixture_v1",
                "source_manifest_sha256": SOURCE_SHA,
                "contract_sha256": sha(config_path),
                "window_id": window_id,
                "design_id": "dev_01",
                "condition_id": "default",
                "step": 0,
                "public_info_sha256": _canonical_sha(public_info),
                "info": public_info,
            }
        start = env.clock_seconds
        _, _, terminated, truncated, after_info = env.step(action)
        event = after_info["transition"]
        if int(event["action"]) != action or not mask[action]:
            raise RuntimeError("mask-legal phase step was rewritten")
        if not math.isclose(sum(float(phase["elapsed_seconds"]) for phase in event["phase_events"]),
                            float(event["step_cost_seconds"]), abs_tol=1e-7):
            raise RuntimeError("phase ledger does not conserve elapsed time")
        if not math.isclose(env.clock_seconds - start, float(event["step_cost_seconds"]), abs_tol=1e-7):
            raise RuntimeError("clock does not conserve step cost")
        if event["backhaul_transfer_bytes"] != event["model_transfer_bytes"] + event["state_transfer_bytes"]:
            raise RuntimeError("backhaul byte split mismatch")
        if env.metrics["completed_nodes"] != int(event["service_completed"]):
            raise RuntimeError("node execution count mismatch")
        rows.append({
            "window_id": window_id, "design_id": "dev_01", "condition_id": "default",
            "public_info_sha256": _canonical_sha(before_info),
            "requested_action": action, "executed_action": event["action"],
            "public_mask": mask, "terminated": bool(terminated), "truncated": bool(truncated),
            "transition": event, "summary": env.summary(),
        })
    if public_fixture is None or len(rows) != 2:
        raise RuntimeError("expected exact two-trace preflight identity")
    return ({
        "schema_version": "cscwd_raw_phase_v2_preflight_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "two_frozen_development_traces_reset_one_legal_step_each_not_performance_matrix",
        "inputs": {
            "raw_source_sha256": RAW_SHA, "source_manifest_sha256": SOURCE_SHA,
            "workload_manifest_sha256": PARENT_SHA,
            "phase_contract_sha256": sha(config_path),
            "raw_v1_environment_sha256": RAW_ENV_SHA,
            "base_environment_sha256": BASE_ENV_SHA,
            "phase_v2_environment_sha256": sha(ROOT / "src/envs/core/raw_ngsim_phase_v2_env.py"),
        },
        "trace_identity": identity,
        "episode_count": len(rows), "real_step_count": len(rows),
        "training_steps": 0, "formal_or_holdout_results_read": False,
        "rows": rows,
    }, public_fixture)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv-path", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    record, public_fixture = preflight(args.raw_csv_path, args.source_manifest)
    output.mkdir(parents=True)
    preflight_path = output / "preflight.json"
    fixture_path = output / "public_fixture.json"
    preflight_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    fixture_path.write_text(json.dumps(public_fixture, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"preflight": str(preflight_path), "preflight_sha256": sha(preflight_path),
                      "public_fixture": str(fixture_path), "public_fixture_sha256": sha(fixture_path),
                      "episodes": record["episode_count"], "steps": record["real_step_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
