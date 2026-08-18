"""Recompute the G06 cache-efficiency contract from a raw episode summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.cache_efficiency_metrics import reduce_cache_efficiency_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--output_path")
    parser.add_argument("--reuse_horizons", default="1,3,6,12")
    args = parser.parse_args()
    summary_path = Path(args.summary_path).resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    horizons = tuple(int(item) for item in args.reuse_horizons.split(",") if item.strip())
    result = reduce_cache_efficiency_summary(payload, reuse_horizons=horizons).to_dict()
    result["summary_path"] = str(summary_path)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output_path:
        output_path = Path(args.output_path).resolve()
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite existing audit: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")
        print(output_path)
    else:
        print(encoded)


if __name__ == "__main__":
    main()
