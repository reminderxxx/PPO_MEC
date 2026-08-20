"""Execute one frozen typed support setting with full provenance binding."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.cache_baseline_fairness import load_and_validate_manifest
from src.evaluators.typed_model_cache_formal_execution import (
    FormalExecutionError,
    support_setting_by_id,
    validate_protocol_v1_1,
    validate_support_binding,
)
from src.evaluators.formal_window_consumption import (
    load_contract as load_window_consumption_contract,
    validate_window_plan_binding,
)
from src.runtime.typed_model_cache_runtime import resolve_model_cache_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-path", required=True)
    parser.add_argument("--setting-id", required=True)
    parser.add_argument("--model-cache-runtime-config", required=True)
    parser.add_argument("--cache-baseline-fairness-manifest-path", required=True)
    parser.add_argument("--seed-checkpoint-manifest-path", required=True)
    parser.add_argument("--checkpoint-provenance-manifest-path", required=True)
    parser.add_argument("--window-plan-path", required=True)
    parser.add_argument("--formal-window-consumption-contract-path", required=True)
    parser.add_argument("--formal-window-split", choices=["formal"], required=True)
    parser.add_argument("--mobility-csv-path", required=True)
    parser.add_argument("--max-mobility-rows", type=int, required=True)
    parser.add_argument("--window-selector", choices=["ordered"], required=True)
    parser.add_argument("--window-length", type=int, required=True)
    parser.add_argument("--rsu-layout", required=True)
    parser.add_argument(
        "--primary-vehicle-selection",
        choices=["stable_first", "handoff_pressure"],
        required=True,
    )
    parser.add_argument("--agents", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--request-replay-path", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def load_json(path: str | Path, label: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise FormalExecutionError(f"{label} must be a JSON object")
    return payload


def benchmark_flags(setting: dict) -> list[str]:
    parameter = setting.get("parameter")
    value = setting.get("value", setting.get("baseline"))
    if parameter == "capacity_mb":
        return []
    if parameter == "handoff_pressure":
        return ["--primary_vehicle_selection", str(value)]
    if parameter == "typed_semantics" and value == "typed_full":
        return []
    if parameter == "typed_semantics" and value == "no_prediction":
        return [
            "--prediction_confidence_scale", "0.0",
            "--drop_handoff_prediction_prob", "1.0",
        ]
    if parameter == "prediction_condition":
        mapping = {
            "baseline": [],
            "no_prediction": ["--prediction_confidence_scale", "0.0", "--drop_handoff_prediction_prob", "1.0"],
            "noise_0.2": ["--prediction_noise_std", "0.2"],
            "confidence_0.7": ["--prediction_confidence_scale", "0.7"],
            "delay_2": ["--prediction_delay_steps", "2"],
            "drop_0.3": ["--drop_handoff_prediction_prob", "0.3"],
        }
        if value in mapping:
            return mapping[value]
    raise FormalExecutionError(
        f"support setting has no safe benchmark_main_results binding: {parameter}={value}"
    )


def stamp_outputs(run_root: Path, provenance: dict) -> None:
    aggregate_path = run_root / "aggregate_summary.json"
    rows_path = run_root / "benchmark_rows.csv"
    aggregate = load_json(aggregate_path, "support aggregate")
    aggregate["support_provenance"] = provenance
    aggregate["support_family"] = provenance["support_family"]
    aggregate["support_setting_id"] = provenance["setting_id"]
    aggregate["support_setting_sha256"] = provenance["support_setting_sha256"]
    aggregate_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with rows_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        row["support_family"] = provenance["support_family"]
        row["support_setting_id"] = provenance["setting_id"]
        row["support_setting_sha256"] = provenance["support_setting_sha256"]
        row["formal_protocol_semantic_sha256"] = provenance["protocol_semantic_sha256"]
        row["split_semantic_sha256"] = provenance["split_semantic_sha256"]
    if rows:
        with rows_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for summary_path in run_root.glob("episodes/**/*.summary.json"):
        summary = load_json(summary_path, "support episode summary")
        summary["support_provenance"] = provenance
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    (run_root / "support_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    protocol = load_json(args.protocol_path, "protocol")
    validate_protocol_v1_1(protocol)
    setting = support_setting_by_id(protocol, args.setting_id)
    runtime = resolve_model_cache_runtime(args.model_cache_runtime_config, root=ROOT)
    fairness, fairness_report = load_and_validate_manifest(
        args.cache_baseline_fairness_manifest_path, root=ROOT, check_files=True
    )
    if fairness_report.get("status") != "pass":
        raise FormalExecutionError("fairness manifest validation failed")
    checkpoint_provenance = load_json(
        args.checkpoint_provenance_manifest_path, "checkpoint provenance"
    )
    load_json(args.seed_checkpoint_manifest_path, "seed checkpoint manifest")
    provenance = validate_support_binding(
        protocol=protocol,
        setting_id=args.setting_id,
        runtime_contract=runtime,
        fairness_manifest=fairness,
        checkpoint_provenance=checkpoint_provenance,
    )
    if setting.get("parameter") == "capacity_mb":
        actual = float(runtime["cache_capacity_profile"]["capacity_mb"])
        expected = float(setting.get("value", setting.get("baseline")))
        if actual != expected:
            raise FormalExecutionError(
                f"capacity support setting mismatch: runtime={actual}, frozen={expected}"
            )
    selection = fairness["dataset_provenance"]["selection_filter_parameters"]
    if selection.get("primary_vehicle_selection") != args.primary_vehicle_selection:
        raise FormalExecutionError("support vehicle selection CLI/fairness mismatch")
    window_contract = load_window_consumption_contract(
        args.formal_window_consumption_contract_path
    )
    expected_contract_hash = (
        protocol.get("execution_contract", {})
        .get("window_consumption_contract", {})
        .get("semantic_sha256")
    )
    if expected_contract_hash != window_contract["hashes"]["semantic_sha256"]:
        raise FormalExecutionError("support window consumption contract hash mismatch")
    validate_window_plan_binding(
        contract=window_contract,
        plan_path=args.window_plan_path,
        split=args.formal_window_split,
        max_mobility_rows=args.max_mobility_rows,
        mobility_csv_path=args.mobility_csv_path,
        window_selector=args.window_selector,
        window_length=args.window_length,
        rsu_layout=args.rsu_layout,
        primary_vehicle_selection=args.primary_vehicle_selection,
        mode="formal",
    )
    parameter = setting.get("parameter")
    value = setting.get("value", setting.get("baseline"))
    expected_selection: dict[str, object] = {}
    if parameter == "handoff_pressure":
        expected_selection["primary_vehicle_selection"] = value
    elif parameter == "typed_semantics" and value == "no_prediction":
        expected_selection.update(
            prediction_confidence_scale=0.0,
            drop_handoff_prediction_prob=1.0,
        )
    elif parameter == "prediction_condition":
        expected_selection.update(
            {
                "baseline": {},
                "no_prediction": {
                    "prediction_confidence_scale": 0.0,
                    "drop_handoff_prediction_prob": 1.0,
                },
                "noise_0.2": {"prediction_noise_std": 0.2},
                "confidence_0.7": {"prediction_confidence_scale": 0.7},
                "delay_2": {"prediction_delay_steps": 2},
                "drop_0.3": {"drop_handoff_prediction_prob": 0.3},
            }[str(value)]
        )
    for field, expected in expected_selection.items():
        if selection.get(field) != expected:
            raise FormalExecutionError(
                f"support fairness manifest does not bind {parameter}: "
                f"{field}={selection.get(field)!r} != {expected!r}"
            )
    if setting.get("parameter") == "oracle_state_limit":
        if not args.request_replay_path:
            raise FormalExecutionError("oracle state-limit setting requires --request-replay-path")
        command = [
            sys.executable,
            str(ROOT / "scripts/run_future_horizon_cache_oracle.py"),
            "--fairness_manifest_path", args.cache_baseline_fairness_manifest_path,
            "--request_replay_path", args.request_replay_path,
            "--output_dir", str(Path(args.output_root) / args.setting_id),
            "--horizons", "1", "3", "6", "12",
            "--state_limit", str(setting.get("value", setting.get("baseline"))),
        ]
        if args.dry_run:
            print(json.dumps({"status": "dry_run_pass", "writes_performed": False, "setting": setting, "support_provenance": provenance, "expanded_command": command}, ensure_ascii=False, indent=2, allow_nan=False))
            return
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)
        run_root = Path(args.output_root) / args.setting_id
        (run_root / "support_provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "pass", "run_root": str(run_root), "support_provenance": provenance}, ensure_ascii=False, indent=2, allow_nan=False))
        return
    benchmark_flags(setting)  # Fail fast when no safe binding is implemented.
    max_steps = {
        int(unit["max_steps"])
        for unit in fairness["window_workload_plan"]["evaluation_units"]
    }
    if len(max_steps) != 1:
        raise FormalExecutionError("support fairness manifest has mixed max_steps")
    command = [
        sys.executable,
        str(ROOT / "scripts/benchmark_main_results.py"),
        "--agents", *args.agents,
        "--seeds", *[str(seed) for seed in args.seeds],
        "--seed_checkpoint_manifest_path", args.seed_checkpoint_manifest_path,
        "--checkpoint_provenance_manifest_path", args.checkpoint_provenance_manifest_path,
        "--cache_baseline_fairness_manifest_path", args.cache_baseline_fairness_manifest_path,
        "--model_cache_runtime_config", args.model_cache_runtime_config,
        "--window_plan_path", args.window_plan_path,
        "--formal_window_consumption_contract_path", args.formal_window_consumption_contract_path,
        "--formal_window_split", args.formal_window_split,
        "--window_consumption_mode", "formal",
        "--mobility_csv_path", args.mobility_csv_path,
        "--window_selector", args.window_selector,
        "--window_length", str(args.window_length),
        "--rsu_layout", args.rsu_layout,
        "--output_root", args.output_root,
        "--max_mobility_rows", str(selection["max_mobility_rows"]),
        "--max_workflows", str(selection["max_workflows"]),
        "--max_steps", str(next(iter(max_steps))),
        "--workflow_selector", str(selection["workflow_selector"]),
        "--min_tasks", str(selection["min_tasks"]),
        "--max_tasks", str(selection["max_tasks"]),
        "--primary_vehicle_selection", str(selection["primary_vehicle_selection"]),
        "--window_mode", str(selection["window_mode"]),
        "--predictor_kind", str(selection["predictor_kind"]),
        "--prediction_horizon", str(selection["prediction_horizon"]),
        "--prediction_noise_std", str(selection["prediction_noise_std"]),
        "--prediction_confidence_scale", str(selection["prediction_confidence_scale"]),
        "--prediction_delay_steps", str(selection["prediction_delay_steps"]),
        "--drop_handoff_prediction_prob", str(selection["drop_handoff_prediction_prob"]),
        "--reward_positive_offset", "0",
        "--audit_runtime",
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_pass",
                    "writes_performed": False,
                    "setting": setting,
                    "support_provenance": provenance,
                    "expanded_command": command,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return
    output_root = Path(args.output_root)
    before = set(output_root.iterdir()) if output_root.exists() else set()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    after = set(output_root.iterdir())
    created = sorted(after - before, key=lambda path: path.stat().st_mtime_ns)
    if len(created) != 1 or not created[0].is_dir():
        raise FormalExecutionError("typed support runner could not identify one new run root")
    stamp_outputs(created[0], provenance)
    print(
        json.dumps(
            {"status": "pass", "run_root": str(created[0]), "support_provenance": provenance},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
