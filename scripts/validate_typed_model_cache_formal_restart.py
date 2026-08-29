"""Validate protocol v1.1 and its persisted execution companions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.typed_model_cache_formal_execution import (
    validate_command_templates,
    validate_protocol_v1_1,
)
from src.evaluators.formal_window_consumption import (
    load_contract as load_window_consumption_contract,
    validate_reachability,
)
from src.runtime.resolved_formal_execution_context import (
    load_resolved_formal_execution_context,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--window-consumption-contract-path", required=True)
    parser.add_argument("--resolved-execution-context-path", default="")
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol_path).read_text(encoding="utf-8-sig"))
    protocol_report = validate_protocol_v1_1(protocol)
    execution = protocol["execution_contract"]
    context_report = None
    if protocol["typed_model_cache_formal_protocol_version"] in {
        "1.5.0",
        "1.6.0",
        "1.7.0",
        "1.8.0",
        "1.9.0",
    }:
        if not args.resolved_execution_context_path:
            raise ValueError("active protocol preflight requires resolved execution context")
        context_payload, context_report = load_resolved_formal_execution_context(
            args.resolved_execution_context_path,
            protocol=protocol,
            clean_worktree_root=ROOT,
            durable_run_root=Path(args.resolved_execution_context_path).resolve().parent,
            check_git=True,
        )
        expansion_context = context_payload["resolved_expansion_context"]
    else:
        expansion_context = execution["default_expansion_context"]
    command_report = validate_command_templates(
        execution["command_templates"], expansion_context
    )
    if context_report is not None:
        if (
            command_report["command_matrix_sha256"]
            != context_report["outer_expansion_sha256"]
        ):
            raise ValueError("nested/outer resolved command expansion hash mismatch")
        if int(expansion_context.get("max_mobility_rows", 0)) != 11_850_526:
            raise ValueError("formal preflight rejects default or truncated mobility rows")
    window_contract = load_window_consumption_contract(
        args.window_consumption_contract_path
    )
    if (
        execution.get("window_consumption_contract", {}).get("semantic_sha256")
        != window_contract["hashes"]["semantic_sha256"]
    ):
        raise ValueError("protocol/window consumption contract hash mismatch")
    reachability = validate_reachability(args.window_consumption_contract_path)
    if reachability["status"] != "pass" or reachability["reachable_count"] != 60:
        raise ValueError("formal window reachability gate failed")
    payload = {
        "status": "pass",
        "protocol": protocol_report,
        "command_expansion": command_report,
        "resolved_execution_context": (
            {
                **context_report,
                "outer_expansion_sha256": context_report[
                    "outer_expansion_sha256"
                ],
                "nested_expansion_sha256": command_report[
                    "command_matrix_sha256"
                ],
                "expansion_equal": True,
                "resolved_python_absolute_path": context_payload[
                    "runtime_location"
                ]["resolved_python_absolute_path"],
                "python_resolution_source": context_payload["runtime_location"][
                    "python_resolution_source"
                ],
                "clean_import_root": context_payload["runtime_location"][
                    "clean_worktree_root"
                ],
                "durable_run_root": context_payload["runtime_location"][
                    "durable_run_root"
                ],
                "context_sha256": context_payload["context_sha256"],
            }
            if context_report is not None
            else None
        ),
        "window_reachability": reachability,
        "execution_boundary": {
            "max_mobility_rows": 11_850_526,
            "reachable_windows": 60,
            "train_metadata_only": True,
            "dev_metadata_only": True,
            "formal_metadata_only": True,
            "holdout_metadata_only": True,
        },
        "holdout_capability": False,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
