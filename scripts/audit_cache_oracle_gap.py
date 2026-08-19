"""Audit a matched observed baseline against one G08 oracle result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.oracles.future_horizon_cache_oracle import compare_baseline_to_oracle


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cache baseline-to-oracle gap")
    parser.add_argument("--oracle_result_path", required=True)
    parser.add_argument("--observed_baseline_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    output = Path(args.output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite gap audit: {output}")
    oracle = json.loads(Path(args.oracle_result_path).read_text(encoding="utf-8-sig"))
    observed = json.loads(Path(args.observed_baseline_path).read_text(encoding="utf-8-sig"))
    result = compare_baseline_to_oracle(oracle_result=oracle, observed_baseline=observed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
