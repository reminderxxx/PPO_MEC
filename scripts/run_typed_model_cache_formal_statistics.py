"""Collect frozen formal row artifacts and run the preregistered statistics consumer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.typed_model_cache_formal_execution import (
    FormalExecutionError,
    validate_protocol_v1_1,
)
from src.runtime.portable_resource_identity import add_portable_resource_arguments
from src.runtime.resolved_formal_execution_context import (
    load_resolved_formal_execution_context,
    resolved_python_for_nested_consumer,
)
from src.runtime.formal_agent_order import (
    FormalAgentOrderError,
    resolve_formal_agent_order,
)
from src.runtime.formal_invalid_run_registry import (
    reject_permanently_invalid_formal_references,
)
from src.runtime.formal_protocol_capabilities import get_protocol_capabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-path", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resolved-execution-context-path", default="")
    parser.add_argument("--formal-agent-order-contract-path", default="")
    parser.add_argument("--non-formal-rehearsal", action="store_true")
    parser.add_argument("--rehearsal-baseline-agent", action="append", default=[])
    add_portable_resource_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reject_permanently_invalid_formal_references(
        [args.input_root, args.output_root, args.resolved_execution_context_path]
    )
    protocol = json.loads(Path(args.protocol_path).read_text(encoding="utf-8-sig"))
    validate_protocol_v1_1(protocol)
    nested_python = sys.executable
    protocol_version = protocol["typed_model_cache_formal_protocol_version"]
    capabilities = get_protocol_capabilities(protocol_version)
    if capabilities.persisted_resolved_execution_context_required:
        if not args.resolved_execution_context_path:
            raise FormalExecutionError(
                "protocol v1.5 statistics requires resolved execution context"
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
    order_audit = None
    if capabilities.agent_order_contract_required:
        if not args.formal_agent_order_contract_path:
            raise FormalExecutionError("Protocol v1.7 statistics requires agent order contract")
        try:
            order_audit = resolve_formal_agent_order(
                contract_path=args.formal_agent_order_contract_path,
                protocol=protocol,
            )
        except FormalAgentOrderError as exc:
            raise FormalExecutionError(str(exc)) from exc
    input_root = Path(args.input_root)
    rows = sorted(input_root.glob("formal_controller/**/benchmark_rows.csv"))
    if not rows:
        raise FormalExecutionError("formal controller rows are missing")
    command = [
        nested_python,
        str(ROOT / "scripts/analyze_top_journal_statistics.py"),
    ]
    for path in rows:
        command.extend(["--rows_path", str(path.resolve())])
    command.extend(
        [
            "--output_root",
            str(Path(args.output_root)),
            "--metrics",
            *protocol["endpoints"]["primary"],
            "--outer_cluster_keys",
            "source_segment_run_id",
            "window_id",
            "--inner_cluster_keys",
            "seed",
            "workflow_id",
            "--bootstrap_samples",
            "10000",
            "--random_seed",
            "1401",
        ]
    )
    if order_audit is not None:
        baseline_agents = order_audit["statistics_baseline_agent_order"]
        if args.non_formal_rehearsal:
            baseline_agents = list(args.rehearsal_baseline_agent)
            if not baseline_agents or baseline_agents != [
                agent
                for agent in order_audit["statistics_baseline_agent_order"]
                if agent in set(baseline_agents)
            ]:
                raise FormalExecutionError(
                    "rehearsal baselines must be a non-empty ordered formal subset"
                )
        command.extend(
            [
                "--candidate_agent",
                order_audit["statistics_candidate_agent"],
                "--baseline_agents",
                *baseline_agents,
            ]
        )
        if not args.non_formal_rehearsal:
            command.extend(
                [
                    "--formal-agent-order-contract-path",
                    str(Path(args.formal_agent_order_contract_path).resolve()),
                ]
            )
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    if order_audit is not None and not args.non_formal_rehearsal:
        statistics_payload = json.loads(
            (Path(args.output_root) / "paired_statistics.json").read_text(
                encoding="utf-8-sig"
            )
        )
        if statistics_payload.get(
            "formal_agent_order_contract_semantic_sha256"
        ) != order_audit["semantic_sha256"]:
            raise FormalExecutionError("statistics output order-contract identity drift")
    statistics_payload = json.loads(
        (Path(args.output_root) / "paired_statistics.json").read_text(
            encoding="utf-8-sig"
        )
    )
    nullable_hash = protocol.get(
        "formal_nullable_metric_aggregation_contract", {}
    ).get("semantic_sha256")
    if capabilities.nullable_metric_contract_required:
        statistics_payload[
            "formal_nullable_metric_aggregation_contract_semantic_sha256"
        ] = nullable_hash
        statistics_payload["non_formal_rehearsal"] = bool(
            args.non_formal_rehearsal
        )
        statistics_payload["formal_performance_evidence"] = False if args.non_formal_rehearsal else None
        statistics_payload["formal_agent_order_contract_semantic_sha256"] = (
            order_audit["semantic_sha256"] if order_audit else None
        )
        (Path(args.output_root) / "paired_statistics.json").write_text(
            json.dumps(
                statistics_payload,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": "pass",
                "row_artifact_count": len(rows),
                "output_root": str(Path(args.output_root).resolve()),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
