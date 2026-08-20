"""Validate frozen NGSIM window identity without running an agent or episode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.formal_window_consumption import validate_reachability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-consumption-contract-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--validate-window-plan-only", action="store_true", required=True)
    args = parser.parse_args()
    report = validate_reachability(args.window_consumption_contract_path)
    if report["status"] != "pass" or report["reachable_count"] != 60:
        raise RuntimeError("60-window reachability validation failed")
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": "pass",
                "reachable_count": report["reachable_count"],
                "holdout_metadata_only": report["holdout_metadata_only"],
                "output_path": str(output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
