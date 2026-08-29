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
from src.runtime.portable_resource_identity import (
    add_portable_resource_arguments,
    resolve_argument_resources,
)
from src.runtime.resolved_formal_execution_context import (
    load_resolved_formal_execution_context,
    resolved_python_for_nested_consumer,
)
from src.runtime.formal_agent_order import (
    FormalAgentOrderError,
    resolve_formal_agent_order,
)
from src.runtime.active_formal_bundle import (
    ActiveFormalBundleError,
    resolve_active_bundle_resource,
    resolve_active_bundle_group,
    resolve_support_resource,
    validate_active_formal_bundle,
    validate_registered_resource_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-path", required=True)
    parser.add_argument("--setting-id", required=True)
    parser.add_argument("--model-cache-runtime-config", required=True)
    parser.add_argument("--cache-baseline-fairness-manifest-path", required=True)
    parser.add_argument("--seed-checkpoint-manifest-path", required=True)
    parser.add_argument("--checkpoint-provenance-manifest-path", required=True)
    parser.add_argument("--window-plan-path", required=True)
    parser.add_argument("--formal-window-consumption-contract-path", default="")
    parser.add_argument("--formal-window-split", choices=["formal"], default="formal")
    parser.add_argument("--mobility-csv-path", required=True)
    parser.add_argument("--workflow-csv-path", required=True)
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
    parser.add_argument("--non-formal-rehearsal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resolved-execution-context-path", default="")
    parser.add_argument("--formal-agent-order-contract-path", default="")
    add_portable_resource_arguments(parser)
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
    if args.resource_registry_path:
        resolve_argument_resources(
            args,
            bindings=(
                ("mobility_resource_id", "mobility_csv_path", "mobility_dataset"),
                ("workflow_resource_id", "workflow_csv_path", "workflow_dataset"),
                ("window_plan_resource_id", "window_plan_path", "window_plan"),
                ("runtime_config_resource_id", "model_cache_runtime_config", "runtime_config"),
                ("fairness_manifest_resource_id", "cache_baseline_fairness_manifest_path", "fairness_manifest"),
                ("checkpoint_manifest_id", "seed_checkpoint_manifest_path", "checkpoint_manifest"),
            ),
        )
    protocol = load_json(args.protocol_path, "protocol")
    validate_protocol_v1_1(protocol)
    nested_python = sys.executable
    protocol_version = protocol["typed_model_cache_formal_protocol_version"]
    resolved_context = None
    if protocol_version in {"1.5.0", "1.6.0", "1.7.0", "1.8.0", "1.9.0"}:
        if not args.resolved_execution_context_path:
            raise FormalExecutionError(
                "protocol v1.5 support requires resolved execution context"
            )
        resolved_context, _ = load_resolved_formal_execution_context(
            args.resolved_execution_context_path,
            protocol=protocol,
            clean_worktree_root=ROOT,
            durable_run_root=Path(args.resolved_execution_context_path).resolve().parent,
            check_git=True,
        )
        nested_python = resolved_python_for_nested_consumer(
            resolved_context, observed_sys_executable=sys.executable
        )
    resource_resolution_audit = None
    if protocol_version == "1.9.0":
        try:
            bundle = validate_active_formal_bundle(
                repository_root=ROOT,
                index_path=resolved_context["resolved_expansion_context"][
                    "active_protocol_index_path"
                ],
                protocol_path=args.protocol_path,
                require_clean_git=False,
                require_origin_main_match=False,
            )
            if bundle["active_formal_bundle_sha256"] != resolved_context[
                "scientific_identity"
            ].get("active_formal_bundle_sha256"):
                raise ActiveFormalBundleError(
                    "support active bundle differs from resolved context"
                )
            supplied_runtime = Path(args.model_cache_runtime_config).resolve()
            runtime_matches = [
                row
                for row in resolve_active_bundle_group(
                    bundle, "runtime_configs", expected_role="typed runtime config"
                )
                if Path(row["resolved_absolute_path"]) == supplied_runtime
            ]
            if len(runtime_matches) != 1:
                raise ActiveFormalBundleError(
                    "support runtime path is not one registered capacity resource"
                )
            runtime_resource = runtime_matches[0]
            setting = support_setting_by_id(protocol, args.setting_id)
            if setting.get("family") == "capacity":
                label = runtime_resource["logical_id"].split(".", 1)[1]
                fairness_resource = resolve_active_bundle_resource(
                    bundle,
                    f"fairness_manifests.{label}",
                    expected_role="formal fairness manifest",
                )
            else:
                fairness_resource = resolve_support_resource(bundle, args.setting_id)
            validate_registered_resource_path(runtime_resource, supplied_runtime)
            validate_registered_resource_path(
                fairness_resource,
                Path(args.cache_baseline_fairness_manifest_path).resolve(),
            )
            resource_resolution_audit = {
                "active_bundle_resource_resolution_contract_version": "1.0.0",
                "active_bundle_sha256": bundle["active_formal_bundle_sha256"],
                "runtime": runtime_resource,
                "fairness": fairness_resource,
                "validation_status": "validated",
            }
        except (ActiveFormalBundleError, KeyError) as exc:
            raise FormalExecutionError(
                f"support active bundle resource resolution failed: {exc}"
            ) from exc
    setting = support_setting_by_id(protocol, args.setting_id)
    runtime = resolve_model_cache_runtime(args.model_cache_runtime_config, root=ROOT)
    fairness, fairness_report = load_and_validate_manifest(
        args.cache_baseline_fairness_manifest_path, root=ROOT, check_files=True
    )
    if fairness_report.get("status") != "pass":
        raise FormalExecutionError("fairness manifest validation failed")
    order_audit = None
    if protocol_version in {"1.7.0", "1.8.0", "1.9.0"}:
        if not args.formal_agent_order_contract_path:
            raise FormalExecutionError("Protocol v1.7 support requires agent order contract")
        try:
            order_audit = resolve_formal_agent_order(
                contract_path=args.formal_agent_order_contract_path,
                protocol=protocol,
                fairness_manifests=[fairness],
            )
        except FormalAgentOrderError as exc:
            raise FormalExecutionError(str(exc)) from exc
        if list(args.agents) != order_audit["main_benchmark_agent_order"]:
            raise FormalExecutionError("support agent order differs from formal contract")
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
    if order_audit is not None:
        provenance["formal_agent_order_contract_semantic_sha256"] = order_audit[
            "semantic_sha256"
        ]
    if resource_resolution_audit is not None:
        provenance["active_bundle_resource_resolution"] = resource_resolution_audit
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
    if args.non_formal_rehearsal:
        if args.formal_window_consumption_contract_path:
            raise FormalExecutionError(
                "non-formal support rehearsal must not bind the formal window contract"
            )
    else:
        if not args.formal_window_consumption_contract_path:
            raise FormalExecutionError("formal support requires the window contract")
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
            nested_python,
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
    setting_flags = benchmark_flags(setting)
    max_steps = {
        int(unit["max_steps"])
        for unit in fairness["window_workload_plan"]["evaluation_units"]
    }
    if len(max_steps) != 1:
        raise FormalExecutionError("support fairness manifest has mixed max_steps")
    command = [
        nested_python,
        str(ROOT / "scripts/benchmark_main_results.py"),
        "--agents", *args.agents,
        "--seeds", *[str(seed) for seed in args.seeds],
        "--seed_checkpoint_manifest_path", args.seed_checkpoint_manifest_path,
        "--checkpoint_provenance_manifest_path", args.checkpoint_provenance_manifest_path,
        "--cache_baseline_fairness_manifest_path", args.cache_baseline_fairness_manifest_path,
        "--model_cache_runtime_config", args.model_cache_runtime_config,
        "--window_plan_path", args.window_plan_path,
        "--mobility_csv_path", args.mobility_csv_path,
        "--workflow_csv_path", args.workflow_csv_path,
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
    if order_audit is not None:
        command.extend(
            [
                "--formal-agent-order-contract-path",
                str(Path(args.formal_agent_order_contract_path).resolve()),
            ]
        )
    if not args.non_formal_rehearsal:
        command.extend(
            [
                "--formal_window_consumption_contract_path",
                args.formal_window_consumption_contract_path,
                "--formal_window_split",
                args.formal_window_split,
                "--window_consumption_mode",
                "formal",
            ]
        )
    command.extend(setting_flags)
    if args.resource_registry_path:
        command.extend(
            [
                "--resource-registry-path", args.resource_registry_path,
                "--repository-root", args.repository_root or str(ROOT),
                "--data-root", args.data_root,
                "--protocol-artifact-root", args.protocol_artifact_root,
                "--checkpoint-root", args.checkpoint_root,
                "--mobility-resource-id", args.mobility_resource_id,
                "--workflow-resource-id", args.workflow_resource_id,
                "--window-plan-resource-id", args.window_plan_resource_id,
                "--runtime-config-resource-id", args.runtime_config_resource_id,
                "--fairness-manifest-resource-id", args.fairness_manifest_resource_id,
                "--checkpoint-manifest-id", args.checkpoint_manifest_id,
            ]
        )
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
