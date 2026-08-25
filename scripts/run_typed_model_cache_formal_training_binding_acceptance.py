"""Run G14R6 clean-worktree contract acceptance without training or performance evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_typed_model_cache_formal_protocol import resolved_expansion_context
from src.evaluators.typed_model_cache_formal_execution import (
    validate_command_templates,
    validate_protocol_v1_1,
)
from src.evaluators.typed_model_cache_formal_protocol import canonical_sha256, sha256_file


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


def run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(cwd), "PYTHONNOUSERSITE": "1"},
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-worktree-root", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-path", required=True)
    args = parser.parse_args()
    clean_root = Path(args.clean_worktree_root).resolve()
    python = str(Path(args.python_executable).absolute())
    output_root = Path(args.output_root).resolve()
    summary_path = Path(args.summary_path).resolve()
    if (clean_root / ".venv").exists():
        raise ValueError("clean acceptance worktree must not contain local .venv")
    if output_root.exists():
        raise FileExistsError(output_root)
    protocol_path = clean_root / "configs/experiment/typed_model_cache_formal_protocol_v1_6_20260825/protocol_v1_6_manifest.json"
    environment_path = clean_root / "configs/experiment/typed_model_cache_formal_protocol_v1_6_20260825/execution_environment_manifest.json"
    protocol = read_json(protocol_path)
    validate_protocol_v1_1(protocol)
    context = resolved_expansion_context(
        protocol,
        protocol_path=str(protocol_path),
        output_root=str(output_root),
        python_executable=python,
    )
    outer = validate_command_templates(protocol["execution_contract"]["command_templates"], context)
    train = outer["expanded"]["train"]
    training_commands = train["commands"]
    matrix_contexts = train["matrix_contexts"]
    if len(training_commands) != 150 or len(matrix_contexts) != 150:
        raise ValueError("Protocol v1.6 training matrix must contain 150 cells")
    required_flags = {
        "--agent_scientific_config_path",
        "--formal_training_execution_binding_path",
        "--resolved_execution_context_path",
        "--formal_protocol_path",
    }
    command_identity_rows = []
    for command, coordinates in zip(training_commands, matrix_contexts):
        missing = sorted(required_flags - set(command))
        if missing or "--agent_config_path" in command:
            raise ValueError(f"training command identity flags drift: {missing}")
        command_identity_rows.append(
            {
                "agent": coordinates["agent"],
                "seed": coordinates["seed"],
                "capacity_label": coordinates["capacity_label"],
                "scientific_config_path": command[command.index("--agent_scientific_config_path") + 1],
                "execution_binding_path": command[command.index("--formal_training_execution_binding_path") + 1],
                "resolved_context_path": command[command.index("--resolved_execution_context_path") + 1],
            }
        )
    identity_triplets = {
        (row["scientific_config_path"], row["execution_binding_path"], row["resolved_context_path"])
        for row in command_identity_rows
    }
    if len(identity_triplets) != 1:
        raise ValueError("150 training commands do not share one config/binding/context identity")

    common = [
        python,
        str(clean_root / "scripts/run_typed_model_cache_formal_protocol.py"),
        "--protocol-path", str(protocol_path),
        "--output-root", str(output_root),
        "--python-executable", python,
        "--execution-environment-manifest", str(environment_path),
    ]
    dry_run = run([*common, "--preflight", "--dry-run"], cwd=clean_root)
    dry_payload = json.loads(dry_run.stdout)
    if dry_payload["command_expansion"]["command_matrix_sha256"] != outer["command_matrix_sha256"]:
        raise ValueError("dry-run/outer command matrix mismatch")
    run([*common, "--preflight"], cwd=clean_root)
    run([*common, "--phase", "tests", "--resume"], cwd=clean_root)
    resolved_context = read_json(output_root / "resolved_execution_context.json")
    binding = read_json(output_root / "formal_training_execution_binding.json")
    preflight = read_json(output_root / "preflight.json")
    nested_hash = preflight["command_expansion"]["command_matrix_sha256"]
    if nested_hash != outer["command_matrix_sha256"]:
        raise ValueError("outer/nested command matrix mismatch")

    first_command_by_agent: dict[str, list[str]] = {}
    for command, coordinates in zip(training_commands, matrix_contexts):
        first_command_by_agent.setdefault(str(coordinates["agent"]), list(command))
    if len(first_command_by_agent) != 10:
        raise ValueError("contract entrypoint rehearsal must cover 10 agents")
    rehearsal_root = output_root / "contract_rehearsal"
    agent_results = []
    for agent, original in first_command_by_agent.items():
        command = list(original)
        command[command.index("--output_root") + 1] = str(rehearsal_root)
        command[command.index("--run_id") + 1] = f"contract_preflight_{agent}"
        command.append("--formal_contract_preflight_only")
        run(command, cwd=clean_root)
        artifact = rehearsal_root / agent / f"contract_preflight_{agent}" / "formal_contract_preflight.json"
        payload = read_json(artifact)
        if any(
            (
                payload["formal"], payload["training"], payload["performance_evidence"],
                payload["checkpoint_created"], payload["episode_count"] != 0,
            )
        ):
            raise ValueError(f"contract rehearsal crossed execution boundary: {agent}")
        agent_results.append(
            {
                "agent": agent,
                "status": "pass",
                "resolved_agent_config": payload["resolved_agent_config"],
                "scientific_config_sha256": payload["formal_training_contract"]["agent_scientific_config_semantic_sha256"],
                "execution_binding_sha256": payload["formal_training_contract"]["formal_training_execution_binding_sha256"],
                "artifact_sha256": sha256_file(artifact),
                "episode_count": 0,
                "checkpoint_created": False,
            }
        )
    identity_pairs = {
        (row["scientific_config_sha256"], row["execution_binding_sha256"])
        for row in agent_results
    }
    if len(identity_pairs) != 1:
        raise ValueError("10 agents did not resolve one scientific/binding identity")

    tests_root = ET.parse(output_root / "tests.xml").getroot()
    test_suites = (
        [tests_root]
        if tests_root.tag == "testsuite"
        else list(tests_root.findall("./testsuite"))
    )
    if not test_suites:
        raise ValueError("pytest JUnit XML contains no testsuite")
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in test_suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in test_suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in test_suites)
    reachability = preflight["window_reachability"]
    summary = {
        "status": "pass",
        "formal": False,
        "training": False,
        "performance_evidence": False,
        "checkpoint_count": 0,
        "holdout_opened": False,
        "clean_worktree_root": str(clean_root),
        "clean_worktree_has_local_venv": False,
        "execution_commit": resolved_context["scientific_identity"]["execution_commit"],
        "resolved_shared_python": python,
        "environment_fingerprint": resolved_context["scientific_identity"]["environment_fingerprint"],
        "dependency_fingerprint": resolved_context["scientific_identity"]["dependency_fingerprint"],
        "scientific_config_semantic_sha256": resolved_context["scientific_identity"]["agent_scientific_config_semantic_sha256"],
        "execution_binding_full_sha256": binding["binding_full_sha256"],
        "protocol_semantic_sha256": protocol["hashes"]["semantic_sha256"],
        "resolved_context_sha256": resolved_context["context_sha256"],
        "command_audit": {
            "status": "pass",
            "phase_template_count": outer["phase_count"],
            "command_count": outer["command_count"],
            "training_command_count": len(training_commands),
            "unique_training_command_count": len({canonical_sha256(row) for row in training_commands}),
            "shared_identity_triplet_count": len(identity_triplets),
            "outer_expansion_sha256": outer["command_matrix_sha256"],
            "nested_expansion_sha256": nested_hash,
            "outer_nested_equal": True,
        },
        "entrypoint_rehearsal": {
            "status": "pass",
            "agent_count": len(agent_results),
            "agents": agent_results,
            "formal": False,
            "training": False,
            "performance_evidence": False,
            "checkpoint_count": 0,
        },
        "preflight": {
            "status": "pass",
            "max_mobility_rows": int(context["max_mobility_rows"]),
            "provider_frame_count": reachability["provider_frame_count"],
            "window_count": reachability["window_count"],
            "reachable_count": reachability["reachable_count"],
            "split_reachable_counts": reachability["split_reachable_counts"],
            "tests": tests,
            "failures": failures,
            "errors": errors,
        },
        "ledger_regression": {
            "status": "pass",
            "phase_ledger_path": str(output_root / "phase_state.jsonl"),
            "phase_ledger_sha256": sha256_file(output_root / "phase_state.jsonl"),
            "preflight_and_tests_terminal": True,
            "cell_resume_finalize_regression_covered_by_tests_phase": True,
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
