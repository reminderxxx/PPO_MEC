"""Exercise Protocol 2.1 consumers without training or writing a checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.registry import build_agent
from src.runtime.formal_agent_order import resolve_formal_agent_order
from src.runtime.formal_training_contract import resolve_training_contract
from src.runtime.formal_training_identity import (
    load_strict_json_mapping,
    validate_checkpoint_training_identity,
    validate_execution_binding,
)
from src.runtime.resolved_formal_execution_context import (
    load_resolved_formal_execution_context,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-path", required=True)
    parser.add_argument("--resolved-context-path", required=True)
    parser.add_argument("--execution-binding-path", required=True)
    parser.add_argument("--scientific-config-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    protocol = load_strict_json_mapping(args.protocol_path, "Protocol")
    context, context_report = load_resolved_formal_execution_context(
        args.resolved_context_path,
        protocol=protocol,
        clean_worktree_root=ROOT,
        durable_run_root=Path(args.resolved_context_path).resolve().parent,
        check_git=True,
    )
    binding = load_strict_json_mapping(args.execution_binding_path, "execution binding")
    scientific = load_strict_json_mapping(args.scientific_config_path, "scientific config")
    identity = context["scientific_identity"]
    environment_projection = identity["full_normalized_environment_projection"]
    binding_report = validate_execution_binding(
        binding,
        protocol=protocol,
        scientific_config=scientific,
        execution_commit=identity["execution_commit"],
        environment_identity=environment_projection,
        command_matrix_sha256=context["command_expansion"][
            "resolved_command_matrix_sha256"
        ],
        active_formal_bundle_sha256=identity["active_formal_bundle_sha256"],
    )
    resolved_training = resolve_training_contract(
        agent_name="ppo",
        profile_defaults={
            "episodes": 1,
            "update_every": 1,
            "batch_size": 1,
            "max_steps": 1,
            "checkpoint_every_updates": 1,
            "agent_config": {},
        },
        cli_values={},
        formal_protocol=protocol,
        scientific_config=scientific,
        execution_binding=binding,
        resolved_execution_context=context,
    )
    agent = build_agent("ppo", **resolved_training.agent_config)
    checkpoint_metadata = {
        "agent_scientific_config_semantic_sha256": (
            resolved_training.agent_scientific_config_semantic_sha256
        ),
        "formal_training_execution_binding_sha256": (
            resolved_training.formal_training_execution_binding_sha256
        ),
        "formal_protocol_semantic_sha256": (
            resolved_training.formal_protocol_semantic_sha256
        ),
        "execution_commit": resolved_training.execution_commit,
        "resolved_execution_context_sha256": (
            resolved_training.resolved_execution_context_sha256
        ),
        "formal_agent_order_contract_semantic_sha256": (
            resolved_training.formal_agent_order_contract_semantic_sha256
        ),
        "active_formal_bundle_sha256": (
            resolved_training.active_formal_bundle_sha256
        ),
    }
    checkpoint_report = validate_checkpoint_training_identity(
        checkpoint_metadata,
        scientific_config_sha256=resolved_training.agent_scientific_config_semantic_sha256,
        binding_sha256=resolved_training.formal_training_execution_binding_sha256,
        protocol_semantic_sha256=resolved_training.formal_protocol_semantic_sha256,
        execution_commit=resolved_training.execution_commit,
        resolved_context_sha256=resolved_training.resolved_execution_context_sha256,
        formal_agent_order_contract_semantic_sha256=(
            resolved_training.formal_agent_order_contract_semantic_sha256
        ),
        active_formal_bundle_sha256=resolved_training.active_formal_bundle_sha256,
    )
    order = resolve_formal_agent_order(protocol=protocol, scientific_config=scientific)
    payload = {
        "status": "pass",
        "formal": False,
        "performance_evidence": False,
        "holdout_opened": False,
        "training_executed": False,
        "checkpoint_written": False,
        "formal_row_written": False,
        "execution_binding": binding_report,
        "resolved_context": context_report,
        "environment_identity_projection_contract_version": identity[
            "environment_identity_projection_contract_version"
        ],
        "full_normalized_environment_projection": environment_projection,
        "training_contract_instantiated": True,
        "agent_entrypoint_instantiated": agent.agent_name == "ppo",
        "dev_consumer_agent_order_count": len(order["main_benchmark_agent_order"]),
        "checkpoint_provenance_read": checkpoint_report["status"] == "pass",
    }
    write_json(Path(args.output_path), payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
