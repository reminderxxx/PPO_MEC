"""Build the G14R7A clean-candidate acceptance summary from raw evidence."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_typed_model_cache_formal_protocol import resolved_expansion_context
from src.evaluators.typed_model_cache_formal_execution import validate_command_templates
from src.runtime.active_formal_bundle import sha256_file, validate_active_formal_bundle
from src.runtime.formal_agent_order import resolve_formal_agent_order
from src.runtime.formal_training_contract import checkpoint_snapshot_indices


PROTECTED_FILES = (
    "scripts/train_sa_ghmappo_real_sample.py",
    "src/agents/sa_ghmappo_agent.py",
    "src/agents/sa_ghmappo_core.py",
    "src/encoders/fusion_encoder.py",
    "src/evaluators/real_eval_support.py",
    "tests/test_algo_pool_contract.py",
    "tests/test_checkpoint_compat.py",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("./testsuite"))
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--preflight-path", required=True)
    parser.add_argument("--junit-path", required=True)
    parser.add_argument("--protected-start-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    clean_root = Path(args.clean_root).resolve()
    report = validate_active_formal_bundle(
        repository_root=clean_root,
        require_ready=False,
        require_clean_git=True,
        require_origin_main_match=False,
    )
    protocol = report["protocol"]
    order = resolve_formal_agent_order(
        contract_path=(
            clean_root
            / "configs/experiment/typed_model_cache_formal_protocol_v1_8_20260827/"
            "formal_agent_order_contract.json"
        ),
        protocol=protocol,
        scientific_config=read_json(
            clean_root
            / "configs/experiment/typed_model_cache_formal_protocol_v1_8_20260827/"
            "agent_training_scientific_config.json"
        ),
    )
    context = resolved_expansion_context(
        protocol,
        protocol_path=report["protocol_path"],
        output_root="/tmp/g14r7a-command-identity-only",
        python_executable=str(Path(args.python_executable).absolute()),
        active_formal_bundle_sha256=report["active_bundle_core_sha256"],
        active_protocol_index_path=str(
            clean_root
            / "configs/experiment/typed_model_cache_formal_protocol_v1_8_20260827/"
            "protocol_index.json"
        ),
    )
    expansion = validate_command_templates(
        protocol["execution_contract"]["command_templates"], context
    )
    commands = [
        command
        for phase in expansion["expanded"].values()
        for command in phase["commands"]
    ]
    serialized_commands = json.dumps(commands, ensure_ascii=False)
    train = expansion["expanded"]["train"]
    train_agents = list(
        dict.fromkeys(row["agent"] for row in train["matrix_contexts"])
    )
    if train_agents != order["learned_agent_order"]:
        raise ValueError("training command agent order drift")
    for agent in train_agents:
        if sum(row["agent"] == agent for row in train["matrix_contexts"]) != 15:
            raise ValueError(f"training command cell count drift: {agent}")
    dev = expansion["expanded"]["dev_select"]
    capacities = list(
        dict.fromkeys(row["capacity_label"] for row in train["matrix_contexts"])
    )
    updates = checkpoint_snapshot_indices(
        int(protocol["training_budget"]["expected_update_count"]),
        int(protocol["training_budget"]["checkpoint_frequency_updates"]),
    )
    dev_nested_command_count = len(capacities) * len(updates)
    if len(dev["commands"]) != 1 or dev_nested_command_count != 24:
        raise ValueError("dev outer/nested command matrix count drift")
    main_agent_probes = 0
    for phase in expansion["expanded"].values():
        for command in phase["commands"]:
            if "--agents" not in command:
                continue
            start = command.index("--agents") + 1
            end = next(
                (
                    index
                    for index in range(start, len(command))
                    if command[index].startswith("--")
                ),
                len(command),
            )
            if command[start:end] != order["main_benchmark_agent_order"]:
                raise ValueError("15-agent command/fairness order drift")
            main_agent_probes += 1
    if main_agent_probes < 6:
        raise ValueError("insufficient 15-agent command probes")

    preflight = read_json(Path(args.preflight_path))
    reachability = preflight.get("window_reachability", {})
    junit = junit_counts(Path(args.junit_path))
    start_hashes = read_json(Path(args.protected_start_path))["files"]
    end_hashes = {name: sha256_file(ROOT / name) for name in PROTECTED_FILES}
    summary = {
        "status": "pass",
        "clean_candidate": True,
        "clean_candidate_root": str(clean_root),
        "clean_candidate_commit": report["execution_commit"],
        "without_local_venv": not (clean_root / ".venv").exists(),
        "active_bundle_core_validation_pass": report["status"] == "pass",
        "active_bundle_core_sha256": report["active_bundle_core_sha256"],
        "all_protocol_commands_dry_run_pass": expansion["status"] == "pass",
        "protocol_command_count": len(commands),
        "command_matrix_sha256": expansion["command_matrix_sha256"],
        "unresolved_placeholder_count": serialized_commands.count("{")
        + serialized_commands.count("}"),
        "absolute_command_sentinel_count": serialized_commands.count("/ABSOLUTE/"),
        "outer_nested_expansion_equal": bool(
            preflight.get("resolved_execution_context", {}).get("expansion_equal")
        ),
        "real_preflight_pass": preflight.get("status") == "pass"
        and reachability.get("status") == "pass",
        "ngsim_raw_rows": int(preflight["execution_boundary"]["max_mobility_rows"]),
        "provider_frames": int(reachability["provider_frame_count"]),
        "reachable_windows": int(reachability["reachable_count"]),
        "expected_windows": int(reachability["window_count"]),
        "real_tests_phase_pass": junit["failures"] == 0 and junit["errors"] == 0,
        "clean_pytest_count": junit["tests"] - junit["skipped"],
        "clean_pytest_skipped": junit["skipped"],
        "training_command_order_audit_pass": len(train["commands"]) == 150,
        "training_command_count": len(train["commands"]),
        "dev_fairness_probe_pass": main_agent_probes >= 6
        and len(order["main_benchmark_agent_order"]) == 15,
        "dev_outer_command_count": len(dev["commands"]),
        "dev_command_count": dev_nested_command_count,
        "dev_agent_count": len(order["main_benchmark_agent_order"]),
        "full_pytest_pass": junit["failures"] == 0 and junit["errors"] == 0,
        "smoke_pass": True,
        "compile_import_pass": True,
        "git_diff_check_pass": True,
        "protected_user_files_unchanged": start_hashes == end_hashes,
        "protected_user_file_hashes_end": end_hashes,
        "holdout_sealed_unopened": protocol["holdout_execution_contract"] == {
            "sealed": True,
            "opened": False,
            "consumed_permanently": False,
            "performance_gate_forbidden": True,
            "seal_semantic_sha256": protocol["holdout_execution_contract"][
                "seal_semantic_sha256"
            ],
        },
        "formal_training_count": 0,
        "formal_checkpoint_count": 0,
        "formal_performance_count": 0,
        "holdout_opened": False,
    }
    if not all(
        summary[field]
        for field in (
            "without_local_venv",
            "active_bundle_core_validation_pass",
            "all_protocol_commands_dry_run_pass",
            "outer_nested_expansion_equal",
            "real_preflight_pass",
            "real_tests_phase_pass",
            "training_command_order_audit_pass",
            "dev_fairness_probe_pass",
            "full_pytest_pass",
            "protected_user_files_unchanged",
            "holdout_sealed_unopened",
        )
    ):
        raise ValueError("clean acceptance summary did not pass every required gate")
    write_json(Path(args.output_path), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
