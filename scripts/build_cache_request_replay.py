"""Build a policy-neutral G08 request replay from an explicit G07 manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.oracles.cache_request_replay import build_policy_neutral_replay_from_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build policy-neutral cache request replay")
    parser.add_argument("--fairness_manifest_path", required=True)
    parser.add_argument("--evaluation_unit_id", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite request replay: {output}")
    manifest = json.loads(Path(args.fairness_manifest_path).read_text(encoding="utf-8-sig"))
    replay = build_policy_neutral_replay_from_manifest(
        root=ROOT,
        manifest=manifest,
        evaluation_unit_id=args.evaluation_unit_id,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(replay, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_path": str(output),
        "request_count": len(replay["requests"]),
        "request_replay_fingerprint": replay["request_replay_fingerprint"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
