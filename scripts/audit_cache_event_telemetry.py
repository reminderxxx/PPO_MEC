"""Audit a raw episode summary's CacheEvent telemetry against legacy fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.metrics.cache_event_metrics import audit_cache_event_telemetry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    summary_path = Path(args.summary_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{summary_path.stem}.cache_event_telemetry_audit.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    episode_summaries = payload.get("episode_summaries")
    if isinstance(episode_summaries, list):
        if len(episode_summaries) != 1:
            raise ValueError(
                "aggregate summary audit requires exactly one episode_summary"
            )
        payload = episode_summaries[0]
    audit = audit_cache_event_telemetry(payload)
    audit["summary_path"] = str(summary_path)
    output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
