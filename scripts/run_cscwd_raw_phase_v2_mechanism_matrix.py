"""Run the pre-frozen two-trace development mechanism matrix exactly once."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_cscwd_new_development_reachability import (  # noqa: E402
    BASE_ENV_SHA, PARENT_SHA, RAW_ENV_SHA, RAW_SHA, SOURCE_SHA,
    legal_action, load_traces, sha,
)
from scripts.freeze_calibrated_continuous_workflow_pilot import _load_experiment_config  # noqa: E402
from src.envs.core.raw_ngsim_event_time_env import RawNGSIMEventTimeEnv  # noqa: E402
from src.envs.core.raw_ngsim_phase_v2_env import RawNGSIMPhaseV2Env  # noqa: E402


CONTRACT_SHA = "2a0cee4b4b2ba512f8015296ee1946698dd51066d1ae5744e22b4d428d176343"
V2_ENV_SHA = "5a5b888f37a3ec8922d74e3b886d325ae96ee266852d338deb37bf0f0b67d683"
PREFLIGHT_SHA = "9b26892522a99c5e50189925c8c08affae89ade55fe8512f8f6c0e9cf09245ed"
PUBLIC_FIXTURE_SHA = "1e7da4d92c5e9387438b2bd33d04d6c1b314cbe4dee4dcd03a72c013a65cd8c2"
V1 = "raw_ngsim_event_time_v1"
V2 = "raw_ngsim_event_time_v2"


def verify_inputs(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    expected = (
        (args.raw_csv_path, RAW_SHA),
        (args.source_manifest, SOURCE_SHA),
        (args.preflight, PREFLIGHT_SHA),
        (args.public_fixture, PUBLIC_FIXTURE_SHA),
        (args.public_estimator_path, args.public_estimator_sha256),
        (ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json", PARENT_SHA),
        (ROOT / "src/envs/core/raw_ngsim_event_time_env.py", RAW_ENV_SHA),
        (ROOT / "src/envs/core/raw_ngsim_phase_v2_env.py", V2_ENV_SHA),
        (ROOT / "src/envs/core/calibrated_continuous_workflow_env.py", BASE_ENV_SHA),
        (ROOT / "configs/experiment/cscwd_raw_phase_v2_development_20261011.json", CONTRACT_SHA),
    )
    if len(args.public_estimator_sha256) != 64:
        raise ValueError("public estimator SHA-256 must be explicit")
    for path, digest in expected:
        if sha(path) != digest:
            raise RuntimeError(f"frozen input mismatch: {path}")
    contract = json.loads((ROOT / "configs/experiment/cscwd_raw_phase_v2_development_20261011.json").read_text())
    source = json.loads(args.source_manifest.read_text())
    preflight = json.loads(args.preflight.read_text())
    fixture = json.loads(args.public_fixture.read_text())
    if source.get("decision") != "READY_FOR_BOUNDED_REACHABILITY":
        raise RuntimeError("source qualification has not passed")
    if preflight["inputs"]["phase_v2_environment_sha256"] != V2_ENV_SHA:
        raise RuntimeError("preflight does not attest the exact phase-v2 implementation")
    if fixture["contract_sha256"] != CONTRACT_SHA or fixture["source_manifest_sha256"] != SOURCE_SHA:
        raise RuntimeError("public fixture source/contract mismatch")
    if len(source["selected_windows"]) != 2 or len(preflight["rows"]) != 2:
        raise RuntimeError("frozen two-trace identity lost")
    matrix = contract["matrix"]
    if matrix["design_ids"] != ["dev_01"] or matrix["time_profiles"] != [V1, V2]:
        raise RuntimeError("frozen design/profile identity lost")
    if len(contract["conditions"]) != 3 or len(matrix["methods"]) != 5:
        raise RuntimeError("frozen matrix dimensions changed")
    if matrix["episode_cap"] != 60 or matrix["real_step_cap"] != 2880 or matrix["preview_cap"] != 10000:
        raise RuntimeError("frozen budget changed")
    return contract, source, preflight, fixture


def load_rules(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("frozen_phase_v2_public_rules", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen public rules")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rules = {
        "causal_public_immediate_rule": module.CausalPublicImmediateRule(),
        "causal_public_two_step_rule": module.CausalPublicTwoStepRule(),
    }
    for rule in rules.values():
        if rule.capability_profile != "public_causal_semantic_state_only_v1" or rule.uses_environment_preview:
            raise RuntimeError("rule has privileged or unexpected capability")
    return rules


def install_ample_cache(env: RawNGSIMEventTimeEnv, capacity: int) -> None:
    required = sorted({
        object_id
        for node in env.instance["nodes"]
        for object_id in env._bundle_ids(str(node["required_adapter"]))
    })
    total = sum(int(env._object_catalog[item]["resident_bytes"]) for item in required)
    if total > capacity:
        raise RuntimeError("ample cache does not fit frozen required bundles")
    for cache in env.caches.values():
        cache.capacity_bytes = capacity
        cache.residents = list(required)
        cache.last_used = {item: 0 for item in required}
    env._validate_all_caches()


def make_env(config: dict, instance: dict, trace, contract: dict, condition: dict, profile: str):
    if profile == V1:
        env = RawNGSIMEventTimeEnv(config, deepcopy(instance), trace)
        env.reset()
        if condition["cache_profile"] == "all_required_bundles_at_all_rsus":
            install_ample_cache(env, int(condition["cache_capacity_bytes"]))
    elif profile == V2:
        env = RawNGSIMPhaseV2Env(
            config, deepcopy(instance), trace,
            phase_contract=contract["physical_assumptions"], condition=condition,
        )
        env.reset()
    else:
        raise RuntimeError(f"unexpected time profile: {profile}")
    return env


def choose(method: str, info: dict, rules: dict, profile: str) -> tuple[int, int]:
    mask = list(info["action_mask"])
    if method == "fallback":
        return 2, 0
    if method == "serve":
        return 3, 0
    if method == "prepare_if_legal_else_serve":
        return (4 if mask[4] else 3), 0
    if method in rules:
        requested = int(rules[method].select_action_from_info(info))
        if method == "causal_public_immediate_rule" or profile == V2:
            return requested, 1
        return requested, 1 + sum(mask)
    raise RuntimeError(f"unexpected frozen method: {method}")


def run_episode(config: dict, instance: dict, trace, contract: dict, condition: dict,
                profile: str, method: str, rules: dict) -> tuple[dict, list[dict], int]:
    env = make_env(config, instance, trace, contract, condition, profile)
    info = env._info()
    ledger = []
    previews = 0
    terminated = truncated = False
    max_steps = int(instance["max_steps"])
    for step in range(max_steps):
        mask = list(info["action_mask"])
        requested, rule_previews = choose(method, info, rules, profile)
        executed = legal_action(requested, mask)
        if executed != requested and method in rules:
            raise RuntimeError("public rule requested a mask-illegal action")
        # v1 always calls one native preview; v2 only does so for vehicle fallback.
        previews += rule_previews + (1 if profile == V1 or executed == 2 else 0)
        start = env.clock_seconds
        _, _, terminated, truncated, info = env.step(executed)
        event = info["transition"]
        if int(event["action"]) != executed:
            raise RuntimeError("executor rewrote a prechecked legal action")
        if not math.isclose(float(event["step_cost_seconds"]), env.clock_seconds - start,
                            abs_tol=1e-6, rel_tol=0.0):
            raise RuntimeError("step clock conservation failed")
        if profile == V2:
            phases = list(event["phase_events"])
            if not math.isclose(sum(float(row["elapsed_seconds"]) for row in phases),
                                float(event["step_cost_seconds"]), abs_tol=1e-6, rel_tol=0.0):
                raise RuntimeError("phase clock conservation failed")
            if int(event["backhaul_transfer_bytes"]) != (
                int(event["model_transfer_bytes"]) + int(event["state_transfer_bytes"])
            ):
                raise RuntimeError("backhaul byte conservation failed")
        ledger.append({
            "step": step, "node_id_before": event["node_id"],
            "requested_action": requested, "executed_action": executed,
            "public_mask": mask, "clock_seconds_before": start,
            "transition": event,
            "terminated": bool(terminated), "truncated": bool(truncated),
        })
        if terminated or truncated:
            break
    summary = env.summary()
    result = {
        "time_profile": profile, "condition_id": condition["condition_id"],
        "method": method, "design_id": instance["design_id"],
        "steps": len(ledger), "terminated": bool(terminated), "truncated": bool(truncated),
        "workflow_completed": int(summary["workflow_completed"]),
        "on_time_complete": bool(summary["workflow_completed"] and
                                 float(summary["modeled_completion_seconds"]) <= float(instance["deadline_seconds"])),
        "summary": summary,
        "action_counts": dict(Counter(str(row["executed_action"]) for row in ledger)),
        "requested_executed_mismatches": sum(
            row["requested_action"] != row["executed_action"] for row in ledger
        ),
        "migration_status_counts": dict(Counter(
            str(row["transition"].get("migration_status", "legacy_v1_unreported")) for row in ledger
        )),
        "preview_count_upper_bound": previews,
    }
    return result, ledger, previews


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv-path", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--public-fixture", type=Path, required=True)
    parser.add_argument("--public-estimator-path", type=Path, required=True)
    parser.add_argument("--public-estimator-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    contract, source, _, _ = verify_inputs(args)
    rules = load_rules(args.public_estimator_path)
    traces, identity = load_traces(args.raw_csv_path, source["selected_windows"])
    config, _ = _load_experiment_config(
        ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json"
    )
    parent = json.loads((ROOT / "configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json").read_text())
    instance = next(row for row in parent["instances"] if row["design_id"] == "dev_01")
    matrix = contract["matrix"]
    results = []
    all_ledger = []
    real_steps = previews = 0
    for window in source["selected_windows"]:
        for condition in contract["conditions"]:
            for profile in matrix["time_profiles"]:
                for method in matrix["methods"]:
                    result, ledger, used_previews = run_episode(
                        config, instance, traces[window["window_id"]], contract,
                        condition, profile, method, rules,
                    )
                    result["window_id"] = window["window_id"]
                    results.append(result)
                    all_ledger.append({
                        "window_id": window["window_id"],
                        "condition_id": condition["condition_id"],
                        "time_profile": profile, "method": method,
                        "steps": ledger,
                    })
                    real_steps += len(ledger)
                    previews += used_previews
                    if (len(results) > matrix["episode_cap"] or real_steps > matrix["real_step_cap"]
                            or previews > matrix["preview_cap"]):
                        raise RuntimeError("frozen development budget exceeded")
    if len(results) != matrix["episode_cap"]:
        raise RuntimeError("frozen matrix cell count incomplete")
    manifest = {
        "schema_version": "cscwd_raw_phase_v2_mechanism_matrix_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "exposed_two_trace_one_workload_development_mechanism_only",
        "identity": {
            "source_manifest_sha256": SOURCE_SHA, "raw_source_sha256": RAW_SHA,
            "workload_manifest_sha256": PARENT_SHA, "contract_sha256": CONTRACT_SHA,
            "raw_v1_environment_sha256": RAW_ENV_SHA,
            "raw_v2_environment_sha256": V2_ENV_SHA,
            "public_estimator_sha256": args.public_estimator_sha256,
            "preflight_sha256": PREFLIGHT_SHA,
            "public_fixture_sha256": PUBLIC_FIXTURE_SHA,
        },
        "trace_identity": identity,
        "episode_count": len(results), "real_step_count": real_steps,
        "preview_count_upper_bound": previews,
        "training_steps": 0, "formal_or_holdout_results_read": False,
        "two_step_v2_capability": "immediate_public_score_fallback_second_step_mobility_unknown",
        "v1_condition_boundary": "adverse_backhaul_condition_has_no_v1_backhaul_effect",
        "results": results,
    }
    output.mkdir(parents=True)
    manifest_path = output / "matrix_manifest.json"
    ledger_path = output / "step_ledger.jsonl"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    with ledger_path.open("w") as stream:
        for row in all_ledger:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
        "ledger": str(ledger_path), "ledger_sha256": sha(ledger_path),
        "episode_count": len(results), "real_step_count": real_steps,
        "preview_count_upper_bound": previews,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
