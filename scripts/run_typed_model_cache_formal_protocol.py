"""Run the append-only typed model-cache formal protocol v1.1.

This ordinary runner deliberately exposes no holdout or hidden-data option.
Use ``--dry-run`` during G14R to validate expansion without creating results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.typed_model_cache_formal_execution import (
    AppendOnlyPhaseRunner,
    FormalExecutionError,
    PHASE_ORDER,
    expand_command_plan,
    validate_command_templates,
    validate_protocol_v1_1,
)
from src.evaluators.typed_model_cache_formal_protocol import canonical_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--phase", choices=PHASE_ORDER)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.preflight == bool(args.phase):
        parser.error("select exactly one of --preflight or --phase")
    return args


def load_protocol(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise FormalExecutionError("protocol manifest must be an object")
    validate_protocol_v1_1(payload)
    return payload


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol_path)
    phase = "preflight" if args.preflight else args.phase
    templates = protocol["execution_contract"]["command_templates"]
    context = dict(protocol["execution_contract"]["default_expansion_context"])
    requested_output_root = str(Path(args.output_root).resolve())
    for key, value in list(context.items()):
        if isinstance(value, str) and value.startswith("/ABSOLUTE/FORMAL_OUTPUT_ROOT"):
            context[key] = requested_output_root + value[len("/ABSOLUTE/FORMAL_OUTPUT_ROOT"):]
    context.update(
        protocol_path=str(Path(args.protocol_path).resolve()),
        output_root=requested_output_root,
    )
    if args.dry_run:
        validation = validate_command_templates(templates, context)
        print(
            json.dumps(
                {
                    "status": "dry_run_pass",
                    "selected_phase": phase,
                    "writes_performed": False,
                    "holdout_capability": False,
                    "phase_order": list(PHASE_ORDER),
                    "command_expansion": validation,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return

    if phase == "complete_without_holdout":
        command: list[str] | list[list[str]] = []
        expected_outputs: list[str] = []
        retries = 0
    else:
        spec = templates[phase]
        plan = expand_command_plan(spec, context)
        command = plan["commands"]
        expected_outputs = plan["expected_outputs"]
        retries = int(spec.get("infrastructure_retries", 1))
    runner = AppendOnlyPhaseRunner(
        protocol=protocol,
        output_root=args.output_root,
        resume=args.resume,
    )
    result = runner.run_phase(
        phase,
        command=command,
        input_hash=canonical_sha256(
            {
                "protocol": protocol["hashes"]["semantic_sha256"],
                "phase": phase,
                "commands": command,
            }
        ),
        expected_outputs=expected_outputs,
        infrastructure_retries=retries,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
