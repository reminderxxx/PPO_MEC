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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--window-consumption-contract-path", required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol_path).read_text(encoding="utf-8-sig"))
    protocol_report = validate_protocol_v1_1(protocol)
    execution = protocol["execution_contract"]
    command_report = validate_command_templates(
        execution["command_templates"], execution["default_expansion_context"]
    )
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
        "window_reachability": reachability,
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
